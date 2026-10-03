"""
Length study for the text + AMR source (review of 2026-10-03, §8.2 and §12).

Measures, over every aligned train sentence (chunk 1, TED metadata lines dropped, no length filter), the
merged DiffuSeq length of the text + AMR layout produced by the shipped pipeline:

    len("[CLS] EN [SEP] AMR [SEP]") + 1 separator + len("[CLS] VI [SEP]")

with the real linearizer (diffuseq/amr_linearize.py, sense suffixes kept or dropped) and the real
tokenizer (mBERT + amr_vocab.json). The plain EN -> VI layout is measured for reference.

The first version of this script (§8.2 table) used an ad-hoc linearizer on regex-split sentence blocks;
the table printed here supersedes it for the shipped pipeline.

Run from the repo root (CPU only, thesis_env):
    CUDA_VISIBLE_DEVICES="" python scripts/_oneshot/review_amr_length_study.py
"""

import os
import sys
from collections import Counter
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from basic_utils import myTokenizer  # noqa: E402
from diffuseq.amr_linearize import linearize_sentences  # noqa: E402
from prepare_docamr_datasets import TEXT_AMR_SEPARATOR, chunk_documents, load_split  # noqa: E402


def lengths(texts, hf):
    return np.array([len(x) for x in hf(texts, add_special_tokens=True)["input_ids"]])


def main():
    tok = myTokenizer(SimpleNamespace(vocab="bert", config_name="bert-base-multilingual-cased",
                                      amr_vocab="relations", checkpoint_path=""))
    hf = tok.tokenizer
    chunks = chunk_documents(load_split("train", Counter()), 1, Counter())
    en = [c["en"] for c in chunks]
    vi = [c["vi"] for c in chunks]
    l_vi = lengths(vi, hf)
    rows = [("plain EN -> VI", lengths(en, hf) + 1 + l_vi, None)]
    for drop_sense in (False, True):
        amr = [linearize_sentences(c["trees"], drop_sense=drop_sense).text for c in chunks]
        src = [e + TEXT_AMR_SEPARATOR + a for e, a in zip(en, amr)]
        name = "text + AMR, sense " + ("dropped" if drop_sense else "kept")
        rows.append((name, lengths(src, hf) + 1 + l_vi, lengths(amr, hf) - 2))

    print(f"train sentences: {len(chunks)}")
    print("| layout | AMR tokens p50 / p95 | merged p50 / p95 / p99 | > 128 | > 192 | > 256 | > 320 |")
    print("|---|---|---|---|---|---|---|")
    for name, total, amr_len in rows:
        amr_col = "—" if amr_len is None else f"{np.median(amr_len):.0f} / {np.percentile(amr_len, 95):.0f}"
        over = " | ".join(f"{(total > L).mean() * 100:.1f}%" for L in (128, 192, 256, 320))
        print(f"| {name} | {amr_col} | {np.median(total):.0f} / {np.percentile(total, 95):.0f} / "
              f"{np.percentile(total, 99):.0f} | {over} |")


if __name__ == "__main__":
    main()
