"""
prepare_docamr_datasets.py

Creates two parallel JSONL datasets from the output_doc_amr directory:

1. src_doc_vi_trg_docamr_en/
   - src  : content of .vi doc-N.txt
   - trg  : content of .en doc-N_docamr_docAMR.out

2. src_docamr_en_trg_doc_vi/
   - src  : content of .en doc-N_docamr_docAMR.out
   - trg  : content of .vi doc-N.txt

Each dataset has three splits:
   - train.jsonl  ← train_en-vi.{en,vi}
   - valid.jsonl  ← dev2010.en-vi.{en,vi}
   - test.jsonl   ← tst2015.en-vi.{en,vi}

Usage:
    python prepare_docamr_datasets.py [--input_dir <path>] [--output_dir <path>] [--chunk_size <int>]

Defaults (script lives in repo root):
    input_dir  = datasets/docAMR/output_doc_amr
    output_dir = datasets/docAMR
"""

from __future__ import annotations

import os
import re
import json
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def doc_id_from_name(name: str) -> str:
    """Extract the doc-N stem from a file name, e.g. 'doc-3' from 'doc-3.txt'."""
    return name.split(".")[0]  # 'doc-3'


def read_vi_file(path: str) -> str:
    """
    Read a .vi file, ensure the first line ends with punctuation (.?!),
    and return a list of cleaned sentence lines.
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    if not lines:
        return []
    
    # Check the first line (often a title)
    first_line = lines[0].strip()
    if first_line and first_line[-1] not in ".?!":
        lines[0] = lines[0].rstrip() + ".\n"
        
    # Return cleaned lines
    return [l.strip() for l in lines if l.strip()]


def read_vi_content(path: str) -> str:
    """Read .vi file and return as a single collapsed string."""
    sentences = read_vi_file(path)
    return " ".join(sentences)


def read_en_file(path: str) -> str:
    """
    Read a .en DocAMR file, extract everything starting from the line AFTER 
    the '# ::tok' line, replace all newlines with spaces, and collapse whitespace.
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    start_idx = 0
    for i, line in enumerate(lines):
        if line.startswith("# ::tok"):
            start_idx = i + 1
            break
            
    content = " ".join(lines[start_idx:])
    return re.sub(r'\s+', ' ', content).strip()


def simplify_docamr(amr_str: str) -> str:
    """
    Simplify the docAMR string according to the 'simple' rules:
    - remove (d / document
    - collapse instances like (s1.s / say-01 -> say
    - strip numbers from relations (:ARG1-of -> :ARG-of)
    - remove variable references (e.g., s3.h2, s1.p)
    - remove (, ), /
    """
    # 1. remove (d / document
    amr_str = re.sub(r'\(d\s*/\s*document\b', '', amr_str)
    
    # 2. collapse instances: (var / concept-XX -> concept
    def concept_replacer(m):
        concept = m.group(1)
        concept = re.sub(r'-\d+$', '', concept)
        return concept
    
    amr_str = re.sub(r'\([a-zA-Z0-9_.]+\s*/\s*([a-zA-Z0-9_.-]+)', concept_replacer, amr_str)
    
    # 3. simplify relations: remove digits
    amr_str = re.sub(r':([a-zA-Z]+)\d+(-of)?\b', r':\1\2', amr_str)
    
    # 4. remove variable references like s3.h2
    amr_str = re.sub(r'\b[a-zA-Z]+\d*\.[a-zA-Z0-9]+\b', '', amr_str)
    
    # 5. remove any (, ), /
    amr_str = re.sub(r'[()/]', '', amr_str)
    
    # 6. cleanup extra spaces
    amr_str = re.sub(r'\s+', ' ', amr_str).strip()
    return amr_str



def split_docamr_graph(graph_content: str) -> list[str]:
    """
    Extract individual sentence subgraphs from a DocAMR document graph.
    Looks for ':sntN (' patterns and returns a list of ':sntN (...)' strings.
    """
    # Find all :sntN markers and their positions
    matches = list(re.finditer(r':snt\d+\s+\(', graph_content))
    if not matches:
        return []
    
    sentences = []
    for i in range(len(matches)):
        start = matches[i].start()
        if i + 1 < len(matches):
            end = matches[i+1].start()
        else:
            # Last sentence: find the last ')' of the document
            end = graph_content.rfind(')')
        
        snt_block = graph_content[start:end].strip()
        sentences.append(snt_block)
    return sentences


def find_out_file(en_dir: str, doc_id: str) -> str | None:
    """
    Return the path of the *_docamr_docAMR.out file for the given doc_id
    inside en_dir, or None if not found.
    """
    candidate = os.path.join(en_dir, f"{doc_id}_docamr_docAMR.out")
    return candidate if os.path.isfile(candidate) else None


def build_pairs(en_dir: str, vi_dir: str, chunk_size: int | None = None, simple: bool = True) -> list[dict]:
    """
    Walk the .vi directory, match each doc-N.txt with the corresponding
    doc-N_docamr_docAMR.out in the .en directory.
    
    If chunk_size is provided (>0), splits the document into smaller pairs.
    Each chunk will contain N sentences and a valid DocAMR graph for them.
    """
    pairs = []
    vi_files = sorted(
        [f for f in os.listdir(vi_dir) if f.endswith(".txt")],
        key=lambda x: int(re.search(r"\d+", x).group()),
    )

    missing = 0
    for vi_fname in vi_files:
        doc_id = doc_id_from_name(vi_fname)
        vi_path = os.path.join(vi_dir, vi_fname)
        en_path = find_out_file(en_dir, doc_id)

        if en_path is None:
            logger.warning("No .out file for %s – skipping.", doc_id)
            missing += 1
            continue

        logger.info("  [src-vi] %s", vi_path)
        logger.info("  [src-en] %s", en_path)
        
        if chunk_size and chunk_size > 0:
            vi_sentences = read_vi_file(vi_path)
            en_raw = read_en_file(en_path)
            en_sentences = split_docamr_graph(en_raw)
            
            if not vi_sentences or not en_sentences:
                logger.warning("Empty content for %s – skipping.", doc_id)
                continue
                
            # If mismatch, warn but process based on minimum
            if len(vi_sentences) != len(en_sentences):
                logger.warning("Sentence count mismatch for %s: VI=%d, EN=%d", 
                               doc_id, len(vi_sentences), len(en_sentences))
            
            num_sents = min(len(vi_sentences), len(en_sentences))
            for i in range(0, num_sents, chunk_size):
                vi_chunk = vi_sentences[i : i + chunk_size]
                en_chunk = en_sentences[i : i + chunk_size]
                
                # Join VI
                src_vi = " ".join(vi_chunk)
                
                # Join EN subgraphs and wrap in (d / document ...)
                src_en = "(d / document " + " ".join(en_chunk) + ")"
                if simple:
                    src_en = simplify_docamr(src_en)
                pairs.append({"src_vi": src_vi, "src_en": src_en})
        else:
            vi_text = read_vi_content(vi_path)
            en_text = read_en_file(en_path)
            
            if not vi_text or not en_text:
                logger.warning("Empty content for %s – skipping.", doc_id)
                continue
                
            if simple:
                en_text = simplify_docamr(en_text)
            pairs.append({"src_vi": vi_text, "src_en": en_text})

    logger.info(
        "  Found %d pairs (%d skipped – missing .out).", len(pairs), missing
    )
    return pairs


def write_jsonl(pairs: list[dict], out_path: str, src_key: str, trg_key: str) -> None:
    """Write a .jsonl file where each line is {"src": ..., "trg": ...}."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            record = {"src": pair[src_key], "trg": pair[trg_key]}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("  Wrote %d lines → %s", len(pairs), out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(input_dir: str, output_dir: str, chunk_size: int | None = None, simple: bool = True) -> None:
    # ------------------------------------------------------------------
    # Locate the six sub-folders
    # ------------------------------------------------------------------
    splits = {
        "train": ("train_en-vi.en", "train_en-vi.vi"),
        "valid": ("dev2010.en-vi.en", "dev2010.en-vi.vi"),
        "test":  ("tst2015.en-vi.en", "tst2015.en-vi.vi"),
    }

    # Validate that all required directories exist
    for split_name, (en_folder, vi_folder) in splits.items():
        for folder in (en_folder, vi_folder):
            full = os.path.join(input_dir, folder)
            if not os.path.isdir(full):
                raise FileNotFoundError(
                    f"Expected directory not found: {full}\n"
                    "Please check --input_dir points to 'output_doc_amr'."
                )

    # ------------------------------------------------------------------
    # Dataset 1: src = VI text,  trg = EN DocAMR
    # ------------------------------------------------------------------
    suffix = f"_chunk_{chunk_size}" if chunk_size and chunk_size > 0 else ""
    if simple:
        suffix += "_simple"
    dataset1_dir = os.path.join(output_dir, f"src_doc_vi_trg_docamr_en{suffix}")
    logger.info("=" * 60)
    logger.info("Building dataset 1: src_doc_vi_trg_docamr_en")
    logger.info("=" * 60)

    for split_name, (en_folder, vi_folder) in splits.items():
        logger.info("Processing split: %s", split_name)
        en_dir = os.path.join(input_dir, en_folder)
        vi_dir = os.path.join(input_dir, vi_folder)
        pairs = build_pairs(en_dir, vi_dir, chunk_size=chunk_size, simple=simple)
        out_path = os.path.join(dataset1_dir, f"{split_name}.jsonl")
        write_jsonl(pairs, out_path, src_key="src_vi", trg_key="src_en")

    # ------------------------------------------------------------------
    # Dataset 2: src = EN DocAMR,  trg = VI text
    # ------------------------------------------------------------------
    dataset2_dir = os.path.join(output_dir, f"src_docamr_en_trg_doc_vi{suffix}")
    logger.info("=" * 60)
    logger.info("Building dataset 2: src_docamr_en_trg_doc_vi")
    logger.info("=" * 60)

    for split_name, (en_folder, vi_folder) in splits.items():
        logger.info("Processing split: %s", split_name)
        en_dir = os.path.join(input_dir, en_folder)
        vi_dir = os.path.join(input_dir, vi_folder)
        pairs = build_pairs(en_dir, vi_dir, chunk_size=chunk_size, simple=simple)
        out_path = os.path.join(dataset2_dir, f"{split_name}.jsonl")
        write_jsonl(pairs, out_path, src_key="src_en", trg_key="src_vi")

    logger.info("=" * 60)
    logger.info("Done!")
    logger.info("  Dataset 1 → %s", dataset1_dir)
    logger.info("  Dataset 2 → %s", dataset2_dir)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        description="Prepare paired DocAMR / Vietnamese JSONL datasets."
    )
    parser.add_argument(
        "--input_dir",
        default=os.path.join(script_dir, "datasets", "docAMR", "output_doc_amr"),
        help="Path to the output_doc_amr directory.",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(script_dir, "datasets", "docAMR"),
        help="Directory where the two output dataset folders will be created.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=0,
        help="If specified (>0), split documents into chunks of N sentences."
    )
    parser.add_argument(
        "--simple",
        type=lambda x: (str(x).lower() in ['true', '1', 'yes']),
        default=True,
        help="Whether to simplify DocAMR graphs. Default is True."
    )
    args = parser.parse_args()

    main(args.input_dir, args.output_dir, chunk_size=args.chunk_size, simple=args.simple)
