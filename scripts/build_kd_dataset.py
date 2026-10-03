"""
build_kd_dataset.py — sequence-level knowledge distillation (KD) data for DiffuSeq.

Replaces the train targets of a dataset (and of its companion variants) with translations produced by an
autoregressive teacher. Non-autoregressive / diffusion students trained on distilled targets see one
translation style per source, which reduces the token repetition caused by multimodal references.

Steps:
  1. load the train split of --src_dataset (plain EN -> VI) and of every --companions dataset
     (e.g. text + AMR) and check that their rows are aligned (same build, same order);
  2. translate every train source with the teacher (batches sorted by length; results cached in a JSONL
     file after every batch, so an interrupted run resumes where it stopped);
  3. replace `trg` of row i in every variant with the teacher output for row i;
  4. drop row i from every variant when any variant's new row exceeds --max_seq_len (same rule as
     prepare_docamr_datasets.py, so all variants keep the same sentences);
  5. write <dataset>_kd-<tag>/ folders: train.jsonl distilled, valid.jsonl / test.jsonl copied unchanged
     (evaluation always uses human references), meta.json with teacher, decoding settings and counts.

Teacher: any Hugging Face encoder-decoder (AutoModelForSeq2SeqLM) — a public en->vi model or a teacher
trained on the same train split. A public model may have seen the IWSLT test talks (TED data is part of
several public corpora); --check_test reports the teacher's BLEU and exact-match rate on the test split
as a leakage signal (see docs/plans/pipeline-fixes-and-amr-redesign.md, Phase 2).

GPU Sharing Rule: the default device is CPU. A GPU run needs --device cuda and the user's approval.

Usage (repo root):
    python scripts/build_kd_dataset.py --teacher <hf id or path> --tag <short name> \
        --src_dataset v2_plain_en_vi_chunk_1 \
        --companions v2_amr_en_vi_chunk_1,v2_text_amr_en_vi_chunk_1 \
        --device cuda --batch_size 32 --num_beams 5 --check_test
"""

import argparse
import json
import logging
import os
import shutil
import sys
import time
from types import SimpleNamespace

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import paths  # noqa: E402
from basic_utils import myTokenizer  # noqa: E402
from prepare_docamr_datasets import TEXT_AMR_SEPARATOR, merged_length  # noqa: E402

log = logging.getLogger("build_kd_dataset")


# --------------------------------------------------------------------------------------------------
# Dataset I/O and alignment
# --------------------------------------------------------------------------------------------------

def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_meta(dataset):
    path = os.path.join(paths.dataset_dir(dataset), "meta.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing: build the dataset with prepare_docamr_datasets.py")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def check_alignment(src_rows, companion_rows, src_meta, companion_metas):
    """
    Raise ValueError unless every companion is row-aligned with the source dataset.

    Checks: (1) same row count; (2) same build arguments in meta.json (one prepare_docamr_datasets call
    writes all variants from the same chunks in the same order); (3) same target text per row;
    (4) text+AMR rows start with the plain source followed by " [SEP] ".
    """
    build_args = {k: v for k, v in src_meta["args"].items() if k != "variants"}
    for name, rows in companion_rows.items():
        if len(rows) != len(src_rows):
            raise ValueError(f"{name}: {len(rows)} train rows vs {len(src_rows)} in the source dataset")
        other = {k: v for k, v in companion_metas[name]["args"].items() if k != "variants"}
        if other != build_args:
            raise ValueError(f"{name}: built with different arguments {other} vs {build_args}")
        for i, (a, b) in enumerate(zip(src_rows, rows)):
            if a["trg"] != b["trg"]:
                raise ValueError(f"{name}: row {i} target differs from the source dataset")
            if companion_metas[name]["variant"] in ("text_amr_en_vi", "text_amr_coref_en_vi") and \
                    not b["src"].startswith(a["src"] + TEXT_AMR_SEPARATOR):
                raise ValueError(f"{name}: row {i} source does not start with the plain source")


# --------------------------------------------------------------------------------------------------
# Translation with cache
# --------------------------------------------------------------------------------------------------

def load_cache(path):
    """Cached translations {source: hypothesis}; an unfinished last line (crash) is ignored."""
    cache = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cache[row["src"]] = row["hyp"]
    return cache


def translate_all(sources, translate_batch, cache_path, batch_size):
    """
    Translate unique sources not yet in the cache; return {source: hypothesis} for all sources.

    Sources are sorted by length so a batch holds similar lengths (less padding); every finished batch
    is appended to the cache file before the next one starts.
    """
    cache = load_cache(cache_path)
    todo = sorted({s for s in sources if s not in cache}, key=len)
    log.info("%d unique sources, %d cached, %d to translate", len(set(sources)), len(cache), len(todo))
    with open(cache_path, "a", encoding="utf-8") as f:
        for i in tqdm(range(0, len(todo), batch_size), desc="translate", unit="batch"):
            batch = todo[i:i + batch_size]
            hyps = translate_batch(batch)
            if len(hyps) != len(batch):
                raise RuntimeError(f"teacher returned {len(hyps)} outputs for {len(batch)} inputs")
            for s, h in zip(batch, hyps):
                cache[s] = h
                f.write(json.dumps({"src": s, "hyp": h}, ensure_ascii=False) + "\n")
            f.flush()
    return cache


def hf_translator(args):
    """translate_batch(list[str]) -> list[str] backed by a Hugging Face encoder-decoder."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.teacher)
    if args.src_lang:
        tok.src_lang = args.src_lang
    model = AutoModelForSeq2SeqLM.from_pretrained(args.teacher).to(args.device).eval()
    if args.device.startswith("cuda") and args.fp16:
        model = model.half()
    gen_kwargs = {"num_beams": args.num_beams, "max_new_tokens": args.max_new_tokens}
    if args.tgt_lang:
        gen_kwargs["forced_bos_token_id"] = tok.convert_tokens_to_ids(args.tgt_lang)

    @torch.no_grad()
    def translate_batch(batch):
        enc = tok([args.input_prefix + s for s in batch], return_tensors="pt", padding=True,
                  truncation=True, max_length=args.max_new_tokens).to(args.device)
        out = model.generate(**enc, **gen_kwargs)
        return [t.strip() for t in tok.batch_decode(out, skip_special_tokens=True)]

    return translate_batch


# --------------------------------------------------------------------------------------------------
# Distilled datasets
# --------------------------------------------------------------------------------------------------

def distill_rows(src_rows, companion_rows, cache, tokenizer, max_seq_len):
    """
    New train rows per variant with trg = teacher output, and the indices dropped by the length filter.

    Returns ({variant name: rows}, dropped indices). Variant "__src__" is the source dataset.
    A row index is dropped from every variant when any variant's new row is longer than max_seq_len
    or the teacher output is empty.
    """
    variants = {"__src__": src_rows, **companion_rows}
    out = {name: [] for name in variants}
    dropped = []
    for i, base in enumerate(src_rows):
        hyp = cache[base["src"]]
        new = {name: dict(rows[i], trg=hyp) for name, rows in variants.items()}
        if not hyp or any(merged_length(r, tokenizer) > max_seq_len for r in new.values()):
            dropped.append(i)
            continue
        for name, row in new.items():
            out[name].append(row)
    return out, dropped


def teacher_test_report(test_rows, translate_batch, batch_size):
    """Teacher corpus BLEU / chrF / exact-match rate on the test split (leakage and quality signal)."""
    import sacrebleu
    srcs = [r["src"] for r in test_rows]
    refs = [r["trg"] for r in test_rows]
    hyps = []
    for i in tqdm(range(0, len(srcs), batch_size), desc="teacher on test", unit="batch"):
        hyps += translate_batch(srcs[i:i + batch_size])
    exact = sum(h.strip() == r.strip() for h, r in zip(hyps, refs)) / max(len(refs), 1)
    return {"bleu": round(sacrebleu.corpus_bleu(hyps, [refs]).score, 2),
            "chrf": round(sacrebleu.corpus_chrf(hyps, [refs]).score, 2),
            "exact_match_rate": round(exact, 4), "rows": len(refs)}


def write_outputs(datasets, distilled, dropped, tag, meta_extra):
    """<dataset>_kd-<tag>/: distilled train, valid/test copied unchanged, meta.json."""
    out_dirs = {}
    for name, rows in distilled.items():
        dataset = datasets[name]
        out_dir = paths.dataset_dir(f"{dataset}_kd-{tag}")
        os.makedirs(out_dir, exist_ok=True)
        write_jsonl(os.path.join(out_dir, "train.jsonl"), rows)
        for split in ("valid", "test"):
            shutil.copyfile(os.path.join(paths.dataset_dir(dataset), f"{split}.jsonl"),
                            os.path.join(out_dir, f"{split}.jsonl"))
        meta = load_meta(dataset)
        meta.update({"kd": {**meta_extra, "base_dataset": dataset, "train_rows": len(rows),
                            "dropped_rows": len(dropped)}})
        with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        out_dirs[name] = out_dir
    return out_dirs


def run(args, translate_batch):
    companions = [c.strip() for c in args.companions.split(",") if c.strip()]
    datasets = {"__src__": args.src_dataset, **{c: c for c in companions}}
    src_meta = load_meta(args.src_dataset)
    metas = {c: load_meta(c) for c in companions}
    if src_meta["variant"] != "plain_en_vi":
        raise ValueError(f"--src_dataset must be a plain_en_vi variant, got {src_meta['variant']}")
    max_seq_len = args.max_seq_len or src_meta["args"]["max_seq_len"]

    src_rows = read_jsonl(os.path.join(paths.dataset_dir(args.src_dataset), "train.jsonl"))
    companion_rows = {c: read_jsonl(os.path.join(paths.dataset_dir(c), "train.jsonl")) for c in companions}
    check_alignment(src_rows, companion_rows, src_meta, metas)
    log.info("aligned: %d train rows in %s + %s", len(src_rows), args.src_dataset, companions)

    cache_path = args.cache or os.path.join(paths.dataset_dir(args.src_dataset), f"kd-{args.tag}_cache.jsonl")
    cache = translate_all([r["src"] for r in src_rows], translate_batch, cache_path, args.batch_size)

    tokenizer = myTokenizer(SimpleNamespace(vocab="bert", config_name=src_meta["args"]["config_name"],
                                            amr_vocab=src_meta["args"]["amr_vocab"], checkpoint_path=""))
    distilled, dropped = distill_rows(src_rows, companion_rows, cache, tokenizer, max_seq_len)
    log.info("distilled: %d rows kept, %d dropped (empty output or > %d tokens)",
             len(distilled["__src__"]), len(dropped), max_seq_len)

    meta_extra = {"teacher": args.teacher, "tag": args.tag, "num_beams": args.num_beams,
                  "max_new_tokens": args.max_new_tokens, "cache": cache_path, "max_seq_len": max_seq_len,
                  "built": time.strftime("%Y-%m-%d %H:%M:%S")}
    if args.check_test:
        test_rows = read_jsonl(os.path.join(paths.dataset_dir(args.src_dataset), "test.jsonl"))
        meta_extra["teacher_on_test"] = teacher_test_report(test_rows, translate_batch, args.batch_size)
        log.info("teacher on test: %s", meta_extra["teacher_on_test"])
    out_dirs = write_outputs(datasets, distilled, dropped, args.tag, meta_extra)
    log.info("SUMMARY written: %s", out_dirs)
    return out_dirs


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--teacher", required=True, help="Hugging Face model id or local path (encoder-decoder)")
    p.add_argument("--tag", required=True, help="short teacher name used in output folder names")
    p.add_argument("--src_dataset", default="v2_plain_en_vi_chunk_1")
    p.add_argument("--companions", default="", help="comma-separated aligned variants, e.g. v2_text_amr_en_vi_chunk_1")
    p.add_argument("--device", default="cpu", help="cpu (default) or cuda — GPU needs the user's approval")
    p.add_argument("--fp16", action="store_true", help="half precision on GPU")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_beams", type=int, default=5)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--src_lang", default="", help="source language code for mBART-style teachers (e.g. en_XX)")
    p.add_argument("--tgt_lang", default="", help="target language token forced as BOS (e.g. vi_VN)")
    p.add_argument("--input_prefix", default="", help="text prepended to every source (e.g. 'en: ' for T5-style)")
    p.add_argument("--max_seq_len", type=int, default=0, help="default: max_seq_len of the source dataset build")
    p.add_argument("--cache", default="", help="translation cache JSONL (default: inside the source dataset folder)")
    p.add_argument("--check_test", action="store_true", help="report teacher BLEU / exact match on the test split")
    return p.parse_args(argv)


def main():
    args = parse_args()
    os.makedirs(paths.LOGS_DIR, exist_ok=True)
    log_path = os.path.join(paths.LOGS_DIR, f"build_kd_dataset_{time.strftime('%Y-%m-%d_%H-%M-%S')}.log")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s",
                        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)])
    run(args, hf_translator(args))
    log.info("log written to %s", log_path)


if __name__ == "__main__":
    main()
