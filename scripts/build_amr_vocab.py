"""
build_amr_vocab.py — relation labels and -9x frames added to the tokenizer (amr_vocab = "relations").

Scans the TRAIN split only (labels that occur only in valid/test fall back to WordPiece) and uses the
same linearizer as prepare_docamr_datasets.py, so every emitted relation is covered, including
inverse roles (":ARG0-of", ":op2-of") and cross-sentence roles (":same-as"; each document is
linearized as one chunk).

Output: paths.AMR_VOCAB_FILE (datasets/docAMR/output_doc_amr/amr_vocab.json)
    {"relations": [...], "frames": [...], "relation_counts": {...}, "frame_counts": {...}, "source": ...}

Run from the repo root (CPU only, thesis_env):
    python scripts/build_amr_vocab.py
"""

import json
import logging
import re
import os
import sys
import time
from collections import Counter

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import paths  # noqa: E402
from basic_utils import validate_added_tokens  # noqa: E402
from diffuseq.amr_linearize import linearize_sentences, parse_document, relation_labels, special_frames  # noqa: E402
from prepare_docamr_datasets import read_docamr_graph  # noqa: E402

SNT_RE = re.compile(r"^:snt\d+$")


def main():
    os.makedirs(paths.LOGS_DIR, exist_ok=True)
    log_path = os.path.join(paths.LOGS_DIR, f"build_amr_vocab_{time.strftime('%Y-%m-%d_%H-%M-%S')}.log")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s",
                        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)])
    log = logging.getLogger("build_amr_vocab")

    amr_dir = os.path.join(paths.AMR_DOC_DIR, paths.SPLIT_FOLDERS["train"][0])
    files = sorted(f for f in os.listdir(amr_dir) if f.endswith("_docamr_docAMR.out"))
    relations, frames, failed = Counter(), Counter(), 0
    for name in tqdm(files, desc="train docs", unit="doc"):
        try:
            trees = parse_document(read_docamr_graph(os.path.join(amr_dir, name)))
        except Exception:  # noqa: BLE001 - counted and logged
            failed += 1
            log.warning("unparsable: %s", name)
            continue
        lin = linearize_sentences(trees, drop_sense=False)
        # :sntN are document-structure markers; one parser output nests one inside a sentence graph.
        relations.update(r for r in relation_labels(lin) if not SNT_RE.match(r))
        frames.update(special_frames(lin))

    rel_list, frame_list = sorted(relations), sorted(frames)
    validate_added_tokens(rel_list + frame_list)
    out = {
        "relations": rel_list,
        "frames": frame_list,
        "relation_counts": dict(relations.most_common()),
        "frame_counts": dict(frames.most_common()),
        "source": f"train split, {len(files) - failed} of {len(files)} documents",
    }
    with open(paths.AMR_VOCAB_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    log.info("SUMMARY: %d relation labels, %d -9x frames, %d unparsable docs -> %s",
             len(rel_list), len(frame_list), failed, paths.AMR_VOCAB_FILE)
    log.info("log written to %s", log_path)


if __name__ == "__main__":
    main()
