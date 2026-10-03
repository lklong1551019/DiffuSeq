"""
train_teacher.py — train the autoregressive EN->VI teacher used for knowledge distillation.

Design, walkthroughs and settings: docs/plans/teacher-model.md. The teacher sees only the train split of
--dataset (no valid/test sentence), so its translations cannot leak test references into distilled data.

Outputs (teacher_models/<name>/): tokenizer/, best/ (Hugging Face model + tokenizer; pass to
build_kd_dataset.py --teacher), last.pt (resume), metrics.jsonl, final_metrics.json, args.json.

GPU Sharing Rule: the default device is CPU; a GPU run needs --device cuda and the user's approval.

Usage (repo root):
    CUDA_VISIBLE_DEVICES="" python scripts/train_teacher.py --name smoke --preset tiny --max_steps 20 --max_train_rows 2000
    python scripts/train_teacher.py --name iwslt-bpe16k --preset iwslt --device cuda --bf16
    python scripts/train_teacher.py --name iwslt-bpe16k --preset iwslt --device cuda --bf16 --resume
"""

import argparse
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import paths  # noqa: E402
from teacher.train_loop import train  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--name", required=True, help="output folder name under teacher_models/")
    p.add_argument("--dataset", default="v2_plain_en_vi_chunk_1", help="plain EN->VI dataset folder")
    p.add_argument("--preset", default="iwslt", choices=["iwslt", "tiny"])
    p.add_argument("--device", default="cpu", help="cpu (default) or cuda — GPU needs approval")
    p.add_argument("--bf16", action="store_true", help="bf16 autocast on GPU")
    p.add_argument("--vocab_size", type=int, default=16000)
    p.add_argument("--min_frequency", type=int, default=2)
    p.add_argument("--max_len", type=int, default=256, help="max tokens per source / target / generation")
    p.add_argument("--max_tokens", type=int, default=4096, help="rows x longest row per batch")
    p.add_argument("--update_freq", type=int, default=1, help="batches accumulated per update")
    p.add_argument("--lr", type=float, default=5e-4, help="peak learning rate")
    p.add_argument("--warmup", type=int, default=4000, help="warmup updates")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--clip_norm", type=float, default=1.0)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--max_epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=8, help="epochs without valid BLEU gain before stopping")
    p.add_argument("--max_steps", type=int, default=0, help="stop after this many updates (0 = no limit)")
    p.add_argument("--max_train_rows", type=int, default=None, help="use only the first N train pairs (smoke tests)")
    p.add_argument("--eval_beams", type=int, default=1, help="beams for the per-epoch valid BLEU")
    p.add_argument("--final_beams", type=int, default=5, help="beams for the final valid / test BLEU")
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--log_interval", type=int, default=100)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--resume", action="store_true", help="continue from teacher_models/<name>/last.pt")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out_dir = os.path.join(paths.TEACHER_DIR, args.name)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(paths.LOGS_DIR, exist_ok=True)
    log_path = os.path.join(paths.LOGS_DIR, f"train_teacher_{time.strftime('%Y-%m-%d_%H-%M-%S')}.log")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s",
                        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)])
    with open(os.path.join(out_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    final = train(args, paths.dataset_dir(args.dataset), out_dir)
    logging.getLogger("train_teacher").info("SUMMARY %s | log %s", json.dumps(final), log_path)
    return final


if __name__ == "__main__":
    main()
