"""
prepare_docamr_datasets.py — build model-ready JSONL datasets from aligned EN / VI / DocAMR documents.

Inputs (paths.py):
    datasets/docAMR/output_dataset_doc/<split>/doc-N.txt      EN and VI documents, one sentence per line
    datasets/docAMR/output_doc_amr/<split en>/doc-N_docamr_docAMR.out   DocAMR of the EN document

Steps per split:
    1. align:     a document is used only if its EN lines, VI lines and AMR :snt graphs have equal counts
    2. clean:     sentences whose EN side is TED metadata (<url> lines) are dropped (review bug B10)
    3. chunk:     consecutive groups of --chunk_size sentences inside one document
    4. linearize: bracketed AMR via diffuseq/amr_linearize.py (keeps :ARGn numbers and brackets;
                  references to variables outside the chunk are dropped)
    5. variants:  one folder per variant (below), each with train/valid/test.jsonl and meta.json
    6. filter:    a TRAIN chunk is dropped from every variant built in the same call when any of its
                  rows exceeds --max_seq_len, so all variants train on the same sentences;
                  valid/test rows are never dropped (review bug B7)

Variants (folder = <prefix>_<variant>_chunk_<N>):
    plain_en_vi      src EN text                 -> trg VI
    plain_vi_en      src VI text                 -> trg EN
    amr_en_vi        src EN AMR                  -> trg VI        (+ graph_src)
    vi_amr           src VI text                 -> trg EN AMR
    text_amr_en_vi   src "EN [SEP] EN AMR"       -> trg VI        (+ graph_src)
    text_amr_coref_en_vi
                     src "EN [SEP] EN AMR [SEP] :same-as ( antecedent ) ..."  -> trg VI  (+ graph_src)
                     antecedents = earlier-sentence nodes the chunk refers to (diffuseq/amr_linearize.py,
                     find_coref_links / append_coref_context); rows without a cross-sentence reference
                     are identical to text_amr_en_vi
    bidirectional    vi_amr rows (TEXT_TO_AMR) + amr_en_vi rows (AMR_TO_TEXT), with "direction"

Row format: {"src", "trg", ["direction"], ["graph_src"]}. graph_src entries are
[head_pos, dep_pos, label, label_pos]: token positions of the AMR concepts / relation token inside the
tokenized src string ([CLS] = 0, no direction token), see text_datasets.shift_source_graph.

Usage (CPU only; needs penman — thesis_env):
    python scripts/build_amr_vocab.py                       # once: relation labels + -9x frames
    python prepare_docamr_datasets.py --chunk_size 1 --variants plain_en_vi,text_amr_en_vi
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from types import SimpleNamespace

from tqdm import tqdm

import paths
from basic_utils import AMR_TO_TEXT_LABEL, TEXT_TO_AMR_LABEL, load_defaults_config, myTokenizer
from diffuseq.amr_linearize import (append_coref_context, find_coref_links, linearize_sentences,
                                    map_words_to_tokens, parse_document, triples_to_token_graph,
                                    word_char_starts)

VARIANTS = ("plain_en_vi", "plain_vi_en", "amr_en_vi", "vi_amr", "text_amr_en_vi", "text_amr_coref_en_vi",
            "bidirectional")
DEFAULT_COREF = {"max_depth": 2, "max_links": 4, "follow_chain": True}
TEXT_AMR_SEPARATOR = " [SEP] "
METADATA_RE = re.compile(r"(^https?://\S+$)|(</?url>)", re.IGNORECASE)

logger = logging.getLogger("prepare_docamr_datasets")


# --------------------------------------------------------------------------------------------------
# Reading and alignment
# --------------------------------------------------------------------------------------------------

def read_lines(path):
    """Non-empty stripped lines of a text file."""
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def read_docamr_graph(path):
    """PENMAN text of a DocAMR .out file: every line after `# ::tok`, comment lines removed."""
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    start = next((i + 1 for i, line in enumerate(lines) if line.startswith("# ::tok")), 0)
    body = [line for line in lines[start:] if not line.lstrip().startswith("#")]
    return re.sub(r"\s+", " ", " ".join(body)).strip()


def is_metadata(en_sentence):
    """True for TED metadata lines such as 'http://www.ted.com/talks/...</url>'."""
    return bool(METADATA_RE.search(en_sentence))


def doc_number(name):
    return int(re.search(r"\d+", name).group())


def load_split(split, stats):
    """
    Aligned sentences of one split: list of documents, each a list of (en, vi, amr_tree).
    Documents with unequal EN / VI / AMR counts or an unparsable AMR are skipped (counted in stats).
    """
    en_folder, vi_folder = paths.SPLIT_FOLDERS[split]
    en_dir = os.path.join(paths.TEXT_DOC_DIR, en_folder)
    vi_dir = os.path.join(paths.TEXT_DOC_DIR, vi_folder)
    amr_dir = os.path.join(paths.AMR_DOC_DIR, en_folder)
    docs = []
    names = sorted((f for f in os.listdir(vi_dir) if f.endswith(".txt")), key=doc_number)
    for name in tqdm(names, desc=f"read {split}", unit="doc"):
        doc_id = name[:-4]
        en_path, amr_path = os.path.join(en_dir, name), os.path.join(amr_dir, f"{doc_id}_docamr_docAMR.out")
        if not (os.path.exists(en_path) and os.path.exists(amr_path)):
            stats["skip_missing_file"] += 1
            continue
        en, vi = read_lines(en_path), read_lines(os.path.join(vi_dir, name))
        try:
            trees = parse_document(read_docamr_graph(amr_path))
        except Exception as e:  # noqa: BLE001 - malformed parser output, document skipped
            stats["skip_unparsable_amr"] += 1
            logger.warning("skip %s/%s: AMR not parsable (%s)", split, doc_id, str(e).splitlines()[-1][:80])
            continue
        if not (len(en) == len(vi) == len(trees)):
            stats["skip_count_mismatch"] += 1
            logger.warning("skip %s/%s: EN %d, VI %d, AMR %d sentences", split, doc_id, len(en), len(vi), len(trees))
            continue
        docs.append([s for s in zip(en, vi, trees)])
        stats["docs_used"] += 1
    return docs


def chunk_documents(docs, chunk_size, stats):
    """
    Drop metadata sentences, then cut each document into consecutive chunks of chunk_size.

    Each chunk also keeps the document context needed for coreference:
      doc_trees: AMR of every sentence in original document order, None for dropped metadata sentences
      start:     original position of the chunk's first sentence in doc_trees
    """
    chunks = []
    for sentences in docs:
        doc_trees = [None if is_metadata(s[0]) else s[2] for s in sentences]
        kept = [(pos, s) for pos, s in enumerate(sentences) if doc_trees[pos] is not None]
        stats["metadata_sentences_dropped"] += len(sentences) - len(kept)
        for i in range(0, len(kept), chunk_size):
            group = kept[i:i + chunk_size]
            chunks.append({
                "en": " ".join(s[0] for _, s in group),
                "vi": " ".join(s[1] for _, s in group),
                "trees": [s[2] for _, s in group],
                "doc_trees": doc_trees,
                "start": group[0][0],
            })
    return chunks


# --------------------------------------------------------------------------------------------------
# Row construction
# --------------------------------------------------------------------------------------------------

def amr_source(lin, hf_tokenizer, prefix=""):
    """
    Source string and token-position graph for an AMR source, optionally preceded by text.

    Step by step:
      1. src = prefix + lin.text                (prefix = "" or "<EN> [SEP] ")
      2. char start of each AMR word = len(prefix) + its offset inside lin.text
      3. map char starts -> token positions of tokenize(src) ([CLS] = 0)
      4. triples (word indices) -> [head_pos, dep_pos, label, label_pos]
    Example: prefix "I sing . [SEP] ", AMR "( sing :ARG0 i )" -> positions 6 (sing), 7 (:ARG0), 8 (i)
    in [CLS] I sing . [SEP] ( sing :ARG0 i ) [SEP].
    """
    src = prefix + lin.text
    starts = word_char_starts(lin.words, offset=len(prefix))
    word_to_token = map_words_to_tokens(src, starts, hf_tokenizer)
    return src, triples_to_token_graph(lin.triples, word_to_token)


def build_rows(chunk, variant, hf_tokenizer, drop_sense, coref=None):
    """Rows (dicts) of one chunk for one variant; AMR is linearized lazily per call.

    coref: {"max_depth", "max_links", "follow_chain"} for text_amr_coref_en_vi (DEFAULT_COREF if None).
    """
    if variant == "plain_en_vi":
        return [{"src": chunk["en"], "trg": chunk["vi"]}]
    if variant == "plain_vi_en":
        return [{"src": chunk["vi"], "trg": chunk["en"]}]
    lin = linearize_sentences(chunk["trees"], drop_sense=drop_sense)
    if variant == "vi_amr":
        return [{"src": chunk["vi"], "trg": lin.text}]
    if variant == "amr_en_vi":
        src, graph = amr_source(lin, hf_tokenizer)
        return [{"src": src, "trg": chunk["vi"], "graph_src": graph}]
    if variant == "text_amr_en_vi":
        src, graph = amr_source(lin, hf_tokenizer, prefix=chunk["en"] + TEXT_AMR_SEPARATOR)
        return [{"src": src, "trg": chunk["vi"], "graph_src": graph}]
    if variant == "text_amr_coref_en_vi":
        c = dict(DEFAULT_COREF, **(coref or {}))
        links = find_coref_links(chunk["trees"], chunk["doc_trees"], chunk["start"],
                                 follow_chain=c["follow_chain"], max_links=c["max_links"])
        lin = append_coref_context(lin, links, chunk["doc_trees"], drop_sense=drop_sense, max_depth=c["max_depth"])
        src, graph = amr_source(lin, hf_tokenizer, prefix=chunk["en"] + TEXT_AMR_SEPARATOR)
        return [{"src": src, "trg": chunk["vi"], "graph_src": graph}]
    if variant == "bidirectional":
        src, graph = amr_source(lin, hf_tokenizer)
        return [
            {"src": chunk["vi"], "trg": lin.text, "direction": TEXT_TO_AMR_LABEL},
            {"src": src, "trg": chunk["vi"], "direction": AMR_TO_TEXT_LABEL, "graph_src": graph},
        ]
    raise ValueError(f"unknown variant {variant!r}")


def merged_length(row, tokenizer):
    """Length of the merged DiffuSeq sequence: len(src ids) + 1 separator + len(trg ids).

    Matches text_datasets.merge_pair: a row with merged_length <= seq_len is never trimmed.
    The direction token (bidirectional rows) adds one source position.
    """
    src_len = len(tokenizer.encode_token([row["src"]])[0]) + (1 if row.get("direction") else 0)
    return src_len + 1 + len(tokenizer.encode_token([row["trg"]])[0])


# --------------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------------

def setup_logging():
    os.makedirs(paths.LOGS_DIR, exist_ok=True)
    log_path = os.path.join(paths.LOGS_DIR, f"prepare_docamr_datasets_{time.strftime('%Y-%m-%d_%H-%M-%S')}.log")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s",
                        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)])
    return log_path


def main(args):
    log_path = setup_logging()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = set(variants) - set(VARIANTS)
    if unknown:
        raise ValueError(f"unknown variants {sorted(unknown)}; choose from {VARIANTS}")

    tokenizer = myTokenizer(SimpleNamespace(vocab="bert", config_name=args.config_name,
                                            amr_vocab=args.amr_vocab, checkpoint_path=""))
    hf_tokenizer = tokenizer.tokenizer

    summary = {}
    for split in ("train", "valid", "test"):
        stats = Counter()
        chunks = chunk_documents(load_split(split, stats), args.chunk_size, stats)
        stats["chunks"] = len(chunks)
        # rows[v][i] = rows of chunk i for variant v
        coref = {"max_depth": args.coref_depth, "max_links": args.coref_max, "follow_chain": args.coref_follow_chain}
        rows = {v: [build_rows(c, v, hf_tokenizer, args.drop_sense, coref)
                    for c in tqdm(chunks, desc=f"{split}/{v}", unit="chunk")] for v in variants}
        keep = [True] * len(chunks)
        if split == "train":
            for i in tqdm(range(len(chunks)), desc=f"{split} length filter", unit="chunk"):
                keep[i] = all(merged_length(r, tokenizer) <= args.max_seq_len for v in variants for r in rows[v][i])
        stats["train_chunks_dropped_too_long" if split == "train" else "chunks_dropped"] = keep.count(False)
        for variant in variants:
            out_dir = paths.dataset_dir(f"{args.prefix}_{variant}_chunk_{args.chunk_size}")
            os.makedirs(out_dir, exist_ok=True)
            written = 0
            with open(os.path.join(out_dir, f"{split}.jsonl"), "w", encoding="utf-8") as f:
                for i, chunk_rows in enumerate(rows[variant]):
                    if not keep[i]:
                        continue
                    for row in chunk_rows:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        written += 1
            summary.setdefault(variant, {})[split] = {"written": written, "chunks_dropped_too_long": keep.count(False)}
            if variant == "text_amr_coref_en_vi":
                with_ctx = sum(1 for i, chunk_rows in enumerate(rows[variant]) if keep[i]
                               for r in chunk_rows if r["src"].count("[SEP]") >= 2)
                summary[variant][split]["rows_with_coref_context"] = with_ctx
                logger.info("%s/%s: %d rows carry a coreference context", split, variant, with_ctx)
            logger.info("%s/%s: %d rows written, %d chunks dropped (> %d tokens in some variant)", split,
                        variant, written, keep.count(False), args.max_seq_len)
        summary.setdefault("_alignment", {})[split] = dict(stats)
        logger.info("%s alignment: %s", split, dict(stats))

    for variant in variants:
        out_dir = paths.dataset_dir(f"{args.prefix}_{variant}_chunk_{args.chunk_size}")
        meta = {"variant": variant, "args": vars(args), "counts": summary[variant],
                "alignment": summary["_alignment"], "built": time.strftime("%Y-%m-%d %H:%M:%S")}
        with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    logger.info("SUMMARY %s", json.dumps(summary, ensure_ascii=False))
    logger.info("log written to %s", log_path)


def parse_args(argv=None):
    defaults = load_defaults_config()
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--chunk_size", type=int, default=1, help="sentences per example (>= 1)")
    p.add_argument("--variants", default="plain_en_vi,text_amr_en_vi",
                   help=f"comma-separated subset of {','.join(VARIANTS)}")
    p.add_argument("--max_seq_len", type=int, default=defaults["seq_len"],
                   help="train rows with a longer merged sequence are dropped (default: config seq_len)")
    p.add_argument("--drop_sense", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=True,
                   help="grow-01 -> grow (keeps -9x frames)")
    p.add_argument("--amr_vocab", default="relations", help="tokenizer vocabulary used for lengths/positions")
    p.add_argument("--config_name", default=defaults["config_name"])
    p.add_argument("--prefix", default="v2", help="folder prefix; never reuse a folder of a logged run")
    p.add_argument("--coref_depth", type=int, default=DEFAULT_COREF["max_depth"],
                   help="text_amr_coref_en_vi: depth of each antecedent subtree (2 keeps :name strings)")
    p.add_argument("--coref_max", type=int, default=DEFAULT_COREF["max_links"],
                   help="text_amr_coref_en_vi: at most this many antecedents per row")
    p.add_argument("--coref_follow_chain", type=lambda x: str(x).lower() in ("1", "true", "yes"),
                   default=DEFAULT_COREF["follow_chain"],
                   help="text_amr_coref_en_vi: follow :same-as chains to the earliest mention")
    args = p.parse_args(argv)
    if args.chunk_size < 1:
        p.error("--chunk_size must be >= 1")
    return args


if __name__ == "__main__":
    main(parse_args())
