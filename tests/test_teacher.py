"""teacher/ — tokenizer, batching, collate + label shift (docs/plans/teacher-model.md §5), schedule, loss,
tiny end-to-end training, reload through build_kd_dataset.hf_translator, resume."""

import json
import os
import sys
from types import SimpleNamespace

import pytest
import torch
from transformers.models.marian.modeling_marian import shift_tokens_right

import paths
from teacher.data import IGNORE_INDEX, TokenBucketBatcher, collate, encode_pairs, pair_lengths
from teacher.model import build_model, count_parameters
from teacher.tokenizer import EOS_ID, PAD_ID, load_tokenizer, train_tokenizer
from teacher.train_loop import inverse_sqrt_lr, label_smoothed_loss

PAIRS = [("I sing .", "Tôi hát ."), ("Thank you .", "Cảm ơn các bạn ."),
         ("We are trying to prevent an impact .", "Chúng ta đang cố gắng ngăn chặn cuộc va chạm .")] * 20


@pytest.fixture(scope="module")
def tok():
    return train_tokenizer([s for s, _ in PAIRS] + [t for _, t in PAIRS], vocab_size=300, min_frequency=1)


# ------------------------------------------------------------------------------ tokenizer

def test_special_ids_and_eos(tok):
    assert (tok.pad_token_id, tok.unk_token_id, tok.bos_token_id, tok.eos_token_id) == (0, 1, 2, 3)
    ids = tok("Tôi hát .")["input_ids"]
    assert ids[-1] == EOS_ID and ids.count(EOS_ID) == 1


def test_round_trip_is_exact(tok):
    for text in ["Tôi hát .", "Cảm ơn các bạn .", "We are trying to prevent an impact ."]:
        assert tok.decode(tok(text)["input_ids"], skip_special_tokens=True) == text


def test_save_and_reload(tok, tmp_path):
    tok.save_pretrained(tmp_path)
    again = load_tokenizer(str(tmp_path))
    assert again("Cảm ơn các bạn .")["input_ids"] == tok("Cảm ơn các bạn .")["input_ids"]


# ------------------------------------------------------------------------------ batching

def test_batcher_covers_every_index_once_and_respects_budget():
    lengths = [3, 10, 7, 7, 2, 30, 5, 9, 1, 12]
    b = TokenBucketBatcher(lengths, max_tokens=20, seed=1)
    for epoch in (0, 1):
        batches = b.batches(epoch)
        flat = sorted(i for batch in batches for i in batch)
        assert flat == list(range(len(lengths)))
        for batch in batches:
            longest = max(lengths[i] for i in batch)
            assert len(batch) * longest <= 20 or len(batch) == 1   # 30 > budget forms its own batch
    assert b.batches(0) == b.batches(0)                 # deterministic per epoch
    assert b.batches(0) != b.batches(1)                 # reshuffled across epochs


def test_unshuffled_batches_are_length_sorted():
    lengths = [5, 1, 3, 2, 4]
    batches = TokenBucketBatcher(lengths, max_tokens=4, shuffle=False).batches(0)
    flat = [i for batch in batches for i in batch]
    assert [lengths[i] for i in flat] == sorted(lengths)


# ------------------------------------------------------------------------------ collate + label shift (§5)

def test_collate_and_shift_walkthrough(tok):
    enc = encode_pairs([("I sing .", "Tôi hát ."), ("Thank you .", "Cảm ơn các bạn .")], tok)
    batch = collate(enc, [0, 1])
    s_lens, t_lens = [len(s) for s, _ in enc], [len(t) for _, t in enc]
    assert batch["input_ids"].shape == (2, max(s_lens))
    assert batch["labels"].shape == (2, max(t_lens))
    for row in range(2):
        assert batch["attention_mask"][row].sum() == s_lens[row]
        assert (batch["input_ids"][row, s_lens[row]:] == PAD_ID).all()
        assert batch["labels"][row, t_lens[row] - 1] == EOS_ID                  # </s> before padding
        assert (batch["labels"][row, t_lens[row]:] == IGNORE_INDEX).all()     # -100 exactly on padding
    dec = shift_tokens_right(batch["labels"], PAD_ID, PAD_ID)
    assert (dec[:, 0] == PAD_ID).all()                                          # start token = <pad>
    expected = batch["labels"][:, :-1].clone()
    expected[expected == IGNORE_INDEX] = PAD_ID
    assert torch.equal(dec[:, 1:], expected)                                    # position t holds label t-1


def test_encode_cuts_long_sequences_and_keeps_eos(tok):
    (s, t), = encode_pairs([("I sing . " * 50, "Tôi hát .")], tok, max_len=16)
    assert len(s) == 16 and s[-1] == EOS_ID


# ------------------------------------------------------------------------------ schedule + loss

def test_inverse_sqrt_lr_values():
    assert inverse_sqrt_lr(1000, 5e-4, 4000) == pytest.approx(1.25e-4)
    assert inverse_sqrt_lr(4000, 5e-4, 4000) == pytest.approx(5e-4)
    assert inverse_sqrt_lr(16000, 5e-4, 4000) == pytest.approx(2.5e-4)
    assert inverse_sqrt_lr(0, 5e-4, 4000) == pytest.approx(5e-4 / 4000)


def test_loss_ignores_padding_and_matches_manual_nll():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 7)
    labels = torch.tensor([[1, 2, IGNORE_INDEX], [3, IGNORE_INDEX, IGNORE_INDEX]])
    logp = logits.log_softmax(-1)
    manual = -(logp[0, 0, 1] + logp[0, 1, 2] + logp[1, 0, 3]) / 3
    assert label_smoothed_loss(logits, labels, 0.0).item() == pytest.approx(manual.item(), rel=1e-5)
    assert label_smoothed_loss(logits, labels, 0.1).item() != pytest.approx(manual.item())


def test_model_ties_embeddings(tok):
    model = build_model("tiny", tok)
    assert model.lm_head.weight.data_ptr() == model.model.shared.weight.data_ptr()
    assert model.config.decoder_start_token_id == PAD_ID
    assert count_parameters(model) < sum(p.numel() for p in model.parameters()) + 1


# ------------------------------------------------------------------------------ end to end (tiny, CPU)

def write_dataset(root):
    os.makedirs(root, exist_ok=True)
    for split, rows in (("train", PAIRS), ("valid", PAIRS[:6]), ("test", PAIRS[:6])):
        with open(os.path.join(root, f"{split}.jsonl"), "w", encoding="utf-8") as f:
            for s, t in rows:
                f.write(json.dumps({"src": s, "trg": t}, ensure_ascii=False) + "\n")


def teacher_args(**kw):
    sys.path.insert(0, os.path.join(paths.REPO_ROOT, "scripts"))
    import train_teacher
    argv = ["--name", "t", "--preset", "tiny", "--vocab_size", "300", "--min_frequency", "1", "--lr", "3e-3",
            "--warmup", "10", "--max_tokens", "256", "--dropout", "0.0", "--final_beams", "2", "--log_interval", "1000"]
    for k, v in kw.items():
        argv += [f"--{k}", str(v)] if v is not True else [f"--{k}"]
    return train_teacher.parse_args(argv)


def test_tiny_training_learns_and_reloads_for_kd(tmp_path):
    from teacher.train_loop import train
    data, out = str(tmp_path / "data"), str(tmp_path / "out")
    write_dataset(data)
    final = train(teacher_args(max_epochs=8, patience=8), data, out)
    metrics = [json.loads(l) for l in open(os.path.join(out, "metrics.jsonl"))]
    assert metrics[-1]["train_loss"] < metrics[0]["train_loss"] - 0.5
    assert set(final) >= {"valid", "test"} and os.path.exists(os.path.join(out, "best", "config.json"))

    sys.path.insert(0, os.path.join(paths.REPO_ROOT, "scripts"))
    import build_kd_dataset as kd
    translate_batch = kd.hf_translator(SimpleNamespace(
        teacher=os.path.join(out, "best"), src_lang="", tgt_lang="", device="cpu", fp16=False, num_beams=2,
        max_new_tokens=16, input_prefix=""))
    outs = translate_batch(["I sing .", "Thank you ."])
    assert len(outs) == 2 and all(isinstance(o, str) for o in outs)


def test_resume_continues_epochs_and_updates(tmp_path):
    from teacher.train_loop import train
    data, out = str(tmp_path / "data"), str(tmp_path / "out")
    write_dataset(data)
    train(teacher_args(max_epochs=1), data, out)
    first = [json.loads(l) for l in open(os.path.join(out, "metrics.jsonl"))]
    train(teacher_args(max_epochs=2, resume=True), data, out)
    both = [json.loads(l) for l in open(os.path.join(out, "metrics.jsonl"))]
    assert [m["epoch"] for m in both] == [1, 2]
    assert both[1]["update"] == 2 * first[0]["update"]
