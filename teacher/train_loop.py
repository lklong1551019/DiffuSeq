"""
Teacher training loop (design: docs/plans/teacher-model.md).

Per update step:
  1. collate a length-bucketed batch                         input_ids [B,S], attention_mask [B,S], labels [B,T]
  2. decoder_input_ids = shift_tokens_right(labels)          [B,T]  (<pad> start token, -100 -> <pad>)
  3. logits = model(input_ids, attention_mask, decoder_input_ids)   [B,T,V]
  4. loss = label-smoothed CE over positions with labels != -100, divided by update_freq
  5. every update_freq batches: clip grads, set LR (inverse sqrt), optimizer step
Per epoch: valid loss + greedy valid BLEU; best checkpoint by BLEU (Hugging Face format); last.pt for resume;
early stop after `patience` epochs without a BLEU gain. After training: beam search on valid and test with
the best checkpoint -> final_metrics.json.
"""

import json
import logging
import math
import os
import random
import time

import numpy as np
import sacrebleu
import torch
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM
from transformers.models.marian.modeling_marian import shift_tokens_right

from .data import IGNORE_INDEX, TokenBucketBatcher, collate, encode_pairs, pair_lengths, read_pairs
from .model import build_model, count_parameters
from .tokenizer import load_tokenizer, train_tokenizer

log = logging.getLogger("train_teacher")


def inverse_sqrt_lr(step, peak, warmup):
    """Linear warmup to `peak` over `warmup` updates, then peak * sqrt(warmup / step). step >= 1."""
    step = max(step, 1)
    return peak * min(step / warmup, math.sqrt(warmup / step))


def label_smoothed_loss(logits, labels, smoothing):
    """Mean label-smoothed cross-entropy over positions whose label != -100 (computed in fp32)."""
    return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), labels.reshape(-1),
                           ignore_index=IGNORE_INDEX, label_smoothing=smoothing)


def forward_batch(model, batch, device, use_bf16):
    """Steps 2-3: decoder inputs from labels, then logits."""
    batch = {k: v.to(device) for k, v in batch.items()}
    cfg = model.config
    decoder_input_ids = shift_tokens_right(batch["labels"], cfg.pad_token_id, cfg.decoder_start_token_id)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                       decoder_input_ids=decoder_input_ids).logits
    return logits, batch["labels"]


@torch.no_grad()
def translate(model, tokenizer, sources, device, num_beams=1, batch_size=64, max_new_tokens=256, use_bf16=False):
    """Translations in input order (batches are formed over length-sorted sources)."""
    model.eval()
    order = sorted(range(len(sources)), key=lambda i: len(sources[i]))
    out = [None] * len(sources)
    for k in range(0, len(order), batch_size):
        idx = order[k:k + batch_size]
        enc = tokenizer([sources[i] for i in idx], return_tensors="pt", padding=True, truncation=True,
                        max_length=max_new_tokens).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
            gen = model.generate(**enc, num_beams=num_beams, max_new_tokens=max_new_tokens)
        for i, text in zip(idx, tokenizer.batch_decode(gen, skip_special_tokens=True)):
            out[i] = text.strip()
    return out


def bleu_chrf(hyps, refs):
    return {"bleu": round(sacrebleu.corpus_bleu(hyps, [refs]).score, 2),
            "chrf": round(sacrebleu.corpus_chrf(hyps, [refs]).score, 2)}


@torch.no_grad()
def valid_loss(model, encoded, batcher, device, use_bf16, smoothing):
    model.eval()
    total, tokens = 0.0, 0
    for idx in batcher.batches(0):
        logits, labels = forward_batch(model, collate(encoded, idx), device, use_bf16)
        n = int((labels != IGNORE_INDEX).sum())
        total += label_smoothed_loss(logits, labels, smoothing).item() * n
        tokens += n
    return total / max(tokens, 1)


def _rng_state():
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def train(args, data_dir, out_dir):
    """Train a teacher on data_dir/{train,valid,test}.jsonl; write everything to out_dir."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    use_bf16 = bool(args.bf16) and device.type == "cuda"
    os.makedirs(out_dir, exist_ok=True)

    train_pairs = read_pairs(os.path.join(data_dir, "train.jsonl"), args.max_train_rows)
    valid_pairs = read_pairs(os.path.join(data_dir, "valid.jsonl"))
    test_pairs = read_pairs(os.path.join(data_dir, "test.jsonl"))
    log.info("pairs: train %d, valid %d, test %d", len(train_pairs), len(valid_pairs), len(test_pairs))

    # Tokenizer: trained on train sources + targets only; reused on resume.
    tok_dir = os.path.join(out_dir, "tokenizer")
    if os.path.exists(os.path.join(tok_dir, "tokenizer.json")):
        tokenizer = load_tokenizer(tok_dir)
    else:
        tokenizer = train_tokenizer([s for s, _ in train_pairs] + [t for _, t in train_pairs],
                                    vocab_size=args.vocab_size, min_frequency=args.min_frequency)
        tokenizer.save_pretrained(tok_dir)
    log.info("tokenizer: %d entries", len(tokenizer))

    train_enc = encode_pairs(train_pairs, tokenizer, args.max_len)
    valid_enc = encode_pairs(valid_pairs, tokenizer, args.max_len)
    train_batcher = TokenBucketBatcher(pair_lengths(train_enc), args.max_tokens, seed=args.seed)
    valid_batcher = TokenBucketBatcher(pair_lengths(valid_enc), args.max_tokens, seed=args.seed, shuffle=False)

    model = build_model(args.preset, tokenizer, dropout=args.dropout).to(device)
    log.info("model: preset %s, %.1fM parameters", args.preset, count_parameters(model) / 1e6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-8,
                                  weight_decay=args.weight_decay)

    state = {"epoch": 0, "update": 0, "best_bleu": -1.0, "bad_epochs": 0}
    last_path = os.path.join(out_dir, "last.pt")
    if args.resume and os.path.exists(last_path):
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        state = ckpt["state"]
        _set_rng_state(ckpt["rng"])
        log.info("resumed from %s: epoch %d, update %d, best valid BLEU %.2f",
                 last_path, state["epoch"], state["update"], state["best_bleu"])

    metrics_path = os.path.join(out_dir, "metrics.jsonl")
    best_dir = os.path.join(out_dir, "best")
    stop = False
    while state["epoch"] < args.max_epochs and not stop:
        model.train()
        epoch_loss, epoch_tokens, t0 = 0.0, 0, time.time()
        optimizer.zero_grad(set_to_none=True)
        for b, idx in enumerate(train_batcher.batches(state["epoch"])):
            logits, labels = forward_batch(model, collate(train_enc, idx), device, use_bf16)
            loss = label_smoothed_loss(logits, labels, args.label_smoothing)
            (loss / args.update_freq).backward()
            n = int((labels != IGNORE_INDEX).sum())
            epoch_loss += loss.item() * n
            epoch_tokens += n
            if (b + 1) % args.update_freq:
                continue
            state["update"] += 1
            lr = inverse_sqrt_lr(state["update"], args.lr, args.warmup)
            for g in optimizer.param_groups:
                g["lr"] = lr
            if args.clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if state["update"] % args.log_interval == 0:
                log.info("epoch %d update %d loss %.4f lr %.2e", state["epoch"] + 1, state["update"],
                         epoch_loss / max(epoch_tokens, 1), lr)
            if args.max_steps and state["update"] >= args.max_steps:
                stop = True
                break

        # ---- end of epoch: validation, checkpoints, early stopping --------------------------------
        elapsed = time.time() - t0
        v_loss = valid_loss(model, valid_enc, valid_batcher, device, use_bf16, args.label_smoothing)
        hyps = translate(model, tokenizer, [s for s, _ in valid_pairs], device, num_beams=args.eval_beams,
                         batch_size=args.eval_batch_size, max_new_tokens=args.max_len, use_bf16=use_bf16)
        v_scores = bleu_chrf(hyps, [t for _, t in valid_pairs])
        state["epoch"] += 1
        improved = v_scores["bleu"] > state["best_bleu"]
        if improved:
            state["best_bleu"], state["bad_epochs"] = v_scores["bleu"], 0
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
        else:
            state["bad_epochs"] += 1
        record = {"epoch": state["epoch"], "update": state["update"],
                  "train_loss": round(epoch_loss / max(epoch_tokens, 1), 4), "valid_loss": round(v_loss, 4),
                  "valid_bleu_greedy": v_scores["bleu"], "valid_chrf_greedy": v_scores["chrf"],
                  "lr": inverse_sqrt_lr(max(state["update"], 1), args.lr, args.warmup),
                  "tokens_per_s": round(epoch_tokens / max(elapsed, 1e-9)), "best": improved}
        with open(metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        log.info("epoch %d: %s", state["epoch"], record)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "state": state,
                    "rng": _rng_state()}, last_path)
        if state["bad_epochs"] >= args.patience:
            log.info("early stop: %d epochs without valid BLEU gain", state["bad_epochs"])
            break

    # ---- final evaluation of the best checkpoint (beam search) -----------------------------------
    best = AutoModelForSeq2SeqLM.from_pretrained(best_dir).to(device)
    final = {"best_valid_bleu_greedy": state["best_bleu"], "beams": args.final_beams}
    for name, pairs in (("valid", valid_pairs), ("test", test_pairs)):
        hyps = translate(best, tokenizer, [s for s, _ in pairs], device, num_beams=args.final_beams,
                         batch_size=args.eval_batch_size, max_new_tokens=args.max_len, use_bf16=use_bf16)
        final[name] = bleu_chrf(hyps, [t for _, t in pairs])
    with open(os.path.join(out_dir, "final_metrics.json"), "w") as f:
        json.dump(final, f, indent=2)
    log.info("final (beam %d): %s", args.final_beams, final)
    return final
