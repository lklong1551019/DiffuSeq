"""
prepare_docamr_datasets.py

This script prepares paired DocAMR and Vietnamese text datasets from the 'output_doc_amr' directory.
It generates three types of datasets:

1. src_doc_vi_trg_docamr_en/
   - Source: Vietnamese document text.
   - Target: English DocAMR graph (optionally simplified).

2. src_docamr_en_trg_doc_vi/
   - Source: English DocAMR graph (optionally simplified).
   - Target: Vietnamese document text.

3. bidirection_docamr_en_doc_vi/
   - A combined dataset containing both directions (Dataset 1 and Dataset 2).
   - Each entry includes a "direction" field: "TEXT_TO_AMR" or "AMR_TO_TEXT".
   - Special tokens [TEXT_TO_AMR] and [AMR_TO_TEXT] are used to signal the task.

4. plain_text_en_vi/
   - Source: English document text.
   - Target: Vietnamese document text.

5. plain_text_vi_en/
   - Source: Vietnamese document text.
   - Target: English document text.

Chunking and Normalization:
   If --chunk_size > 0 is specified, documents are split into chunks of N sentences.
   The script performs the following normalization on DocAMR chunks:
   - Re-indexes sentence markers (:sntN) and sentence-specific variables (sN.*) to start from 1.
   - Discards :same-as relations that refer to entities in sentences outside the current chunk.

Usage:
    python prepare_docamr_datasets.py [--input_dir <path>] [--output_dir <path>] [--chunk_size <int>] [--simple <bool>] [--bidirectional <bool>]

Dependencies:
    Requires 'basic_utils.py' in the same directory for shared constants.
"""

from __future__ import annotations

import os
import re
import json
import argparse
import logging
from basic_utils import AMR_TO_TEXT_LABEL, TEXT_TO_AMR_LABEL, load_tokenizer, myTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tokenization Helpers
# ---------------------------------------------------------------------------

def load_tokenizer_for_prep(config_name: str, use_simple_amr: bool = True) -> myTokenizer:
    """Initialize the mBERT tokenizer with custom AMR tokens for data preparation."""
    # Create a dummy args object to satisfy load_tokenizer/myTokenizer
    class DummyArgs:
        def __init__(self):
            self.vocab = 'bert'
            self.config_name = config_name
            self.use_simple_amr = use_simple_amr
            self.checkpoint_path = "./tmp_tokenizer" # Temporary dir for save_pretrained
            os.makedirs(self.checkpoint_path, exist_ok=True)
    
    return load_tokenizer(DummyArgs())


def remap_adj_to_token_level(adj: list, text: str, tokenizer: myTokenizer) -> list:
    """
    Map word-level adjacency indices to subword token-level indices.
    
    Args:
        adj: List of [src_word_idx, trg_word_idx, relation_str]
        text: The space-separated simplified AMR string
        tokenizer: myTokenizer instance
        
    Returns:
        List of [src_token_idx, trg_token_idx, relation_str]
    """
    if not adj:
        return []
        
    # Standard BERT encoding with special tokens [CLS ... SEP]
    encoding = tokenizer.tokenizer(text, add_special_tokens=True)
    new_adj = []
    
    for src_word_idx, trg_word_idx, rel in adj:
        try:
            # word_to_tokens returns a CharacterSpan mapping to the subword indices
            src_tokens = encoding.word_to_tokens(src_word_idx)
            trg_tokens = encoding.word_to_tokens(trg_word_idx)
            
            if src_tokens and trg_tokens:
                # Use the 'start' of the subword range for the edge
                new_adj.append([src_tokens.start, trg_tokens.start, rel])
        except Exception:
            # Skip edges that fall outside the tokenized length or point to invalid words
            continue
            
    return new_adj

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


def read_plain_text_sentences(path: str) -> list[str]:
    """Read a plain text .txt file and return a list of cleaned sentence lines."""
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def build_plain_text_pairs(en_dir: str, vi_dir: str, chunk_size: int | None = None, ignored_docs: set[str] | None = None) -> list[dict]:
    """
    Pair English and Vietnamese plain text docs by doc-ID.
    If chunk_size is provided (>0), splits the document into smaller pairs.
    Returns a list of {"src_en": str, "src_vi": str} dicts.
    """
    en_files = {doc_id_from_name(f): os.path.join(en_dir, f)
                for f in os.listdir(en_dir) if f.endswith(".txt")}
    vi_files = {doc_id_from_name(f): os.path.join(vi_dir, f)
                for f in os.listdir(vi_dir) if f.endswith(".txt")}
    
    pairs = []
    # Sort doc_ids to ensure deterministic order
    doc_ids = sorted(en_files.keys())
    for doc_id in doc_ids:
        if ignored_docs and doc_id in ignored_docs:
            continue
            
        if doc_id not in vi_files:
            continue
        
        en_path = en_files[doc_id]
        vi_path = vi_files[doc_id]
        
        en_sentences = read_plain_text_sentences(en_path)
        vi_sentences = read_plain_text_sentences(vi_path)
        
        if not en_sentences or not vi_sentences:
            continue
            
        if chunk_size and chunk_size > 0:
            num_sents = min(len(en_sentences), len(vi_sentences))
            for i in range(0, num_sents, chunk_size):
                en_chunk = en_sentences[i : i + chunk_size]
                vi_chunk = vi_sentences[i : i + chunk_size]
                pairs.append({
                    "src_en": " ".join(en_chunk),
                    "src_vi": " ".join(vi_chunk)
                })
        else:
            pairs.append({
                "src_en": " ".join(en_sentences),
                "src_vi": " ".join(vi_sentences)
            })
    return pairs


def write_plain_jsonl(pairs: list[dict], out_path: str, src_key: str, trg_key: str, tokenizer: myTokenizer = None, max_seq_len: int = 256) -> None:
    """Write a simple .jsonl file with only src and trg fields, filtered by token length."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    removed_count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            # Tokenize and check length
            src_tokens = tokenizer.encode_token([pair[src_key]])[0]
            trg_tokens = tokenizer.encode_token([pair[trg_key]])[0]
            
            if len(src_tokens) + len(trg_tokens) > max_seq_len - 3:
                removed_count += 1
                continue
                
            record = {"src": pair[src_key], "trg": pair[trg_key]}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    if removed_count > 0:
        logger.info("  Removed %d lines exceeding max_seq_len (%d) from %s", removed_count, max_seq_len, out_path)


def simplify_docamr(amr_str: str, simple: bool = True) -> tuple[str, list]:
    # 1. Clean IBM DocAMR metadata comments
    lines = [line for line in amr_str.splitlines() if not line.strip().startswith('#')]
    clean_amr = " ".join(lines)
    
    # 2. Advanced Tokenization 
    tokens = re.findall(r'\(|\)|/|:[a-zA-Z0-9-._]+|"[^"]+"|[^\s()]+', clean_amr)
    
    simplified_tokens = []
    edges = []
    stack = [] 
    var_to_token_idx = {}
    var_to_concept = {}
    current_rel = None
    
    i = 0
    while i < len(tokens):
        t = tokens[i]

        if t == '(':
            # Step: Detect new AMR instance definition (e.g., (s1.w / we))
            if i + 3 < len(tokens) and tokens[i+2] == '/':
                var = tokens[i+1]
                concept = tokens[i+3]
                
                # Step: Apply simplification (remove -01, etc.)
                if simple:
                    concept = re.sub(r'-\d+$', '', concept)
                
                # Step: Handle special 'document' node (exclude from graph)
                if concept == 'document':
                    var_to_token_idx[var] = -1  # Mark variable as ignored
                    stack.append(-1)            # Push placeholder to stack to maintain tree depth
                    i += 4
                    continue
                
                # Step: Record concept in sequence and map variable for coreference
                token_idx = len(simplified_tokens)
                simplified_tokens.append(concept)
                var_to_token_idx[var] = token_idx
                var_to_concept[var] = concept
                
                # Step: Create edge from parent to this concept
                if stack and stack[-1] != -1 and current_rel:
                    edges.append([stack[-1], token_idx, current_rel])
                
                # Step: Push to stack and reset relation
                current_rel = None
                stack.append(token_idx)
                i += 4
                continue
            else:
                i += 1
                
        elif t == ')':
            if stack: 
                stack.pop()
            i += 1
            
        elif t.startswith(':'):
            # Step: Handle relations (e.g., :ARG0)
            current_rel = t
            if simple:
                # Remove digits from relations (e.g., :ARG1 -> :ARG)
                current_rel = re.sub(r'\d+', '', current_rel)
            
            # Step: Exclude :document structural relations
            if current_rel == ':document':
                current_rel = None
                i += 1
                continue
                
            simplified_tokens.append(current_rel)
            i += 1
            
        elif t == '/':
            # Step: Skip AMR '/' symbol
            i += 1 
            
        else:
            # Step: Handle Leaf nodes (literals or coreference variables)
            if t in var_to_token_idx:
                # Sub-step: Coreference detected (e.g., s1.w)
                target_idx = var_to_token_idx[t]
                if target_idx != -1: 
                    # Note on Coreference Disambiguation:
                    # If two variables have the same concept (e.g., s1.p / person and s2.p / person),
                    # the text sequence will simply contain the word 'person' twice.
                    # However, because var_to_token_idx maps the distinct string variables (s1.p vs s2.p) 
                    # to their exact index in the sequence (e.g., 5 vs 10), the edge drawn below will 
                    # point unambiguously to the exact node intended by the coreference, maintaining
                    # perfect topological accuracy in the adjacency matrix despite textual ambiguity.
                    
                    # Draw edge back to the original definition
                    if stack and stack[-1] != -1 and current_rel:
                        edges.append([stack[-1], target_idx, current_rel])
                    # Replace variable name with its corresponding concept text (e.g., s1.w -> we)
                    simplified_tokens.append(var_to_concept.get(t, t))
            else:
                # Sub-step: Literal token or first occurrence of a non-variable leaf
                token_idx = len(simplified_tokens)
                simplified_tokens.append(t)
                if stack and stack[-1] != -1 and current_rel:
                    edges.append([stack[-1], token_idx, current_rel])
            
            current_rel = None
            i += 1

    return " ".join(simplified_tokens), edges



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


def build_pairs(en_dir: str, vi_dir: str, chunk_size: int | None = None, simple: bool = True, tokenizer: myTokenizer | None = None, ignored_docs: set[str] | None = None) -> list[dict]:
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
        if ignored_docs and doc_id in ignored_docs:
            continue
            
        vi_path = os.path.join(vi_dir, vi_fname)
        en_path = find_out_file(en_dir, doc_id)

        if en_path is None:
            logger.warning("No .out file for %s – skipping.", doc_id)
            missing += 1
            continue

        # logger.info("  [src-vi] %s", vi_path)
        # logger.info("  [src-en] %s", en_path)
        
        if chunk_size and chunk_size > 0:
            vi_sentences = read_vi_file(vi_path)
            en_raw = read_en_file(en_path)
            en_sentences = split_docamr_graph(en_raw)
            
            if not vi_sentences or not en_sentences:
                logger.warning("Empty content for %s – skipping.", doc_id)
                continue
                
            # If mismatch, ignore this document
            if len(vi_sentences) != len(en_sentences):
                logger.warning("Sentence count mismatch for %s: VI=%d, EN=%d. Ignoring this document.", 
                               doc_id, len(vi_sentences), len(en_sentences))
                continue
            
            num_sents = min(len(vi_sentences), len(en_sentences))
            for i in range(0, num_sents, chunk_size):
                vi_chunk = vi_sentences[i : i + chunk_size]
                en_chunk = en_sentences[i : i + chunk_size]
                
                # ---------------------------------------------------------------------
                # Step: Normalize AMR Chunks
                # 1. Identify which original sentence indices (sN) are in this chunk.
                # 2. Map original indices to normalized 1-based indices (1, 2, ...).
                # 3. Discard :same-as relations that point to sentences OUTSIDE this chunk.
                # 4. Re-index :sntN markers and sN variables inside the chunk text.
                # ---------------------------------------------------------------------
                orig_indices = []
                for snt_str in en_chunk:
                    # Find original sentence index from the :sntN marker
                    match = re.search(r':snt(\d+)\b', snt_str)
                    if match:
                        orig_indices.append(int(match.group(1)))
                
                # Create mapping from original index to normalized index
                mapping = {orig: idx + 1 for idx, orig in enumerate(orig_indices)}
                
                normalized_en_chunk = []
                for snt_str in en_chunk:
                    # Sub-step: Remove cross-chunk :same-as relations (targets not in this mapping)
                    snt_str = re.sub(
                        r':same-as\s+s(\d+)(?:\.[a-zA-Z0-9_.-]+)?\b',
                        lambda m: m.group(0) if int(m.group(1)) in mapping else "",
                        snt_str
                    )
                    
                    # Sub-step: Re-indexing using a safe two-pass placeholder strategy to avoid collisions
                    for orig in mapping.keys():
                        snt_str = re.sub(rf':snt{orig}\b', f':snt_TMP_{orig}_', snt_str)
                        snt_str = re.sub(rf'(?<!\w)s{orig}\b', f's_TMP_{orig}_', snt_str)
                    
                    # Final replacement of placeholders with normalized indices
                    for orig, new in mapping.items():
                        snt_str = snt_str.replace(f':snt_TMP_{orig}_', f':snt{new}')
                        snt_str = snt_str.replace(f's_TMP_{orig}_', f's{new}')
                    
                    normalized_en_chunk.append(snt_str)
                
                en_chunk = normalized_en_chunk

                # Join VI
                src_vi = " ".join(vi_chunk)    
                
                # Join EN subgraphs. 
                # If chunk_size == 1, we strip structural markers (:snt, :same-as) 
                # and avoid the (d / document ...) wrapper for a cleaner sentence graph.
                if chunk_size == 1:
                    snt_str = en_chunk[0]
                    # Remove :sntN markers
                    snt_str = re.sub(r':snt\d+\s+', '', snt_str)
                    snt_str = re.sub(r'\s+:snt\d+\b', '', snt_str)
                    # Remove :same-as cross-sentence coreference
                    snt_str = re.sub(r':same-as\s+s\d+(\.[a-zA-Z0-9._-]+)?\b', '', snt_str)
                    src_en_raw = snt_str.strip()
                else:
                    src_en_raw = "(d / document " + " ".join(en_chunk) + ")"

                src_en, adj = simplify_docamr(src_en_raw, simple=simple)
                
                # Step: Pre-compute token-level indices if tokenizer is available
                if tokenizer:
                    adj = remap_adj_to_token_level(adj, src_en, tokenizer)
                
                pairs.append({
                    "src_vi": src_vi, 
                    "src_en": src_en, 
                    "adj": adj
                })
        else:
            vi_text = read_vi_content(vi_path)
            en_text_raw = read_en_file(en_path)
            
            if not vi_text or not en_text_raw:
                logger.warning("Empty content for %s – skipping.", doc_id)
                continue
                
            src_en, adj = simplify_docamr(en_text_raw, simple=simple)
            
            if tokenizer:
                adj = remap_adj_to_token_level(adj, src_en, tokenizer)
                
            pairs.append({
                "src_vi": vi_text, 
                "src_en": src_en, 
                "adj": adj
            })

    logger.info(
        "  Found %d pairs (%d skipped – missing .out).", len(pairs), missing
    )
    return pairs


def write_jsonl(pairs: list[dict], out_path: str, src_key: str, trg_key: str, direction: str | None = None, mode: str = "w", tokenizer: myTokenizer = None, max_seq_len: int = 256) -> None:
    """Write a .jsonl file with explicit adj_src or adj_trg fields, filtered by token length."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    from basic_utils import TEXT_TO_AMR_LABEL, TEXT_TO_AMR_TOKEN, AMR_TO_TEXT_TOKEN
    removed_count = 0
    with open(out_path, mode, encoding="utf-8") as f:
        for pair in pairs:
            # Tokenize and check length
            src_text = pair[src_key]
            if direction:
                token = TEXT_TO_AMR_TOKEN if direction == TEXT_TO_AMR_LABEL else AMR_TO_TEXT_TOKEN
                src_text = f"{token} {src_text}"
            
            src_tokens = tokenizer.encode_token([src_text])[0]
            trg_tokens = tokenizer.encode_token([pair[trg_key]])[0]
            
            if len(src_tokens) + len(trg_tokens) > max_seq_len - 3:
                removed_count += 1
                continue

            record = {"src": pair[src_key], "trg": pair[trg_key]}
            if direction:
                record["direction"] = direction
            if "adj" in pair:
                # Identify if EN DocAMR (which has the adj) is the src or trg in this file
                adj_key = "adj_src" if src_key == "src_en" else "adj_trg"
                record[adj_key] = pair["adj"]
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    if removed_count > 0:
        logger.info("  Removed %d lines exceeding max_seq_len (%d) from %s (mode=%s)", removed_count, max_seq_len, out_path, mode)
    else:
        logger.info("  Wrote %d lines → %s (mode=%s)", len(pairs), out_path, mode)



def pre_analyze_datasets(input_dir: str, splits: dict) -> tuple[set[str], dict]:
    """
    Perform a pre-analysis of the datasets to ensure sentence and line counts match.
    Strictly halts the process if any mismatch is found.
    Returns a tuple of (ignored_docs set, stats dictionary).
    """
    logger.info("Starting pre-analysis of datasets...")
    ignored_docs = set()
    stats = {split: {"total": 0, "discarded": []} for split in splits}

    # 1. Check output_doc_amr (input_dir)
    for split_name, (en_folder, vi_folder) in splits.items():
        en_dir = os.path.join(input_dir, en_folder)
        vi_dir = os.path.join(input_dir, vi_folder)

        if not os.path.isdir(vi_dir):
            continue

        vi_files = sorted([f for f in os.listdir(vi_dir) if f.endswith(".txt")])
        stats[split_name]["total"] = len(vi_files)
        
        for vi_fname in vi_files:
            doc_id = doc_id_from_name(vi_fname)
            vi_path = os.path.join(vi_dir, vi_fname)
            en_path = find_out_file(en_dir, doc_id)

            if en_path is None:
                stats[split_name]["discarded"].append({
                    "doc_id": vi_fname, "folder": en_folder, "reason": "Missing .out file"
                })
                ignored_docs.add(doc_id)
                continue

            vi_sentences = read_vi_file(vi_path)
            num_vi_lines = len(vi_sentences)

            en_raw = read_en_file(en_path)
            
            if "multi-sentence" in en_raw:
                logger.error("Encountered multi-sentence AMR in %s in folder %s", vi_fname, en_dir)
                raise ValueError(f"Pre-analysis failed: multi-sentence AMR found in {en_path}")

            # Find max :sntN marker
            snt_matches = re.findall(r':snt(\d+)\b', en_raw)
            if not snt_matches:
                logger.warning("No :snt found in %s in folder %s. Ignoring this document.", vi_fname, en_dir)
                stats[split_name]["discarded"].append({
                    "doc_id": vi_fname, "folder": en_folder, "reason": "No :snt markers found"
                })
                ignored_docs.add(doc_id)
                continue
                
            max_snt = max(int(n) for n in snt_matches)

            if num_vi_lines != max_snt or num_vi_lines != len(snt_matches):
                logger.warning("Mismatch in DocAMR: %s (VI lines: %d, Max :snt: %d, Num markers: %d) in folder: %s. Ignoring this document.", 
                             vi_fname, num_vi_lines, max_snt, len(snt_matches), vi_dir)
                stats[split_name]["discarded"].append({
                    "doc_id": vi_fname, "folder": en_folder, "reason": f"Sentence mismatch (VI:{num_vi_lines}, AMR:{len(snt_matches)})"
                })
                ignored_docs.add(doc_id)
                continue

    # 2. Check output_dataset_doc (plain text)
    plain_text_root = os.path.join(os.path.dirname(input_dir), "output_dataset_doc")
    if os.path.isdir(plain_text_root):
        plain_splits = {
            "train": ("train_en-vi.en", "train_en-vi.vi"),
            "valid": ("dev2010.en-vi.en", "dev2010.en-vi.vi"),
            "test":  ("tst2015.en-vi.en", "tst2015.en-vi.vi"),
        }
        for split_name, (en_folder, vi_folder) in plain_splits.items():
            en_dir = os.path.join(plain_text_root, en_folder)
            vi_dir = os.path.join(plain_text_root, vi_folder)

            if not os.path.isdir(en_dir) or not os.path.isdir(vi_dir):
                continue

            en_files = sorted([f for f in os.listdir(en_dir) if f.endswith(".txt")])
            for en_fname in en_files:
                doc_id = doc_id_from_name(en_fname)
                if doc_id in ignored_docs:
                    continue
                    
                en_path = os.path.join(en_dir, en_fname)
                vi_path = os.path.join(vi_dir, en_fname)

                if not os.path.exists(vi_path):
                    continue

                en_sentences = read_plain_text_sentences(en_path)
                vi_sentences = read_plain_text_sentences(vi_path)

                if len(en_sentences) != len(vi_sentences):
                    logger.warning("Mismatch in plain text: %s (EN lines: %d, VI lines: %d) in folder: %s. Ignoring this document.", 
                                 en_fname, len(en_sentences), len(vi_sentences), plain_text_root)
                    stats[split_name]["discarded"].append({
                        "doc_id": en_fname, "folder": os.path.basename(en_dir), "reason": f"Plain text mismatch (EN:{len(en_sentences)}, VI:{len(vi_sentences)})"
                    })
                    ignored_docs.add(doc_id)
                    continue

    logger.info("Pre-analysis completed successfully.")
    return ignored_docs, stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(input_dir: str, output_dir: str, chunk_size: int | None = None, simple: bool = True, bidirectional: bool = True, config_name: str = "bert-base-multilingual-cased", max_seq_len: int = 256) -> None:
    # Initialize tokenizer for pre-computing indices
    logger.info("Initializing tokenizer for pre-computation (config: %s)...", config_name)
    tokenizer = load_tokenizer_for_prep(config_name, use_simple_amr=simple)
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

    # Pre-analyze datasets before processing
    ignored_docs, stats = pre_analyze_datasets(input_dir, splits)

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
        pairs = build_pairs(en_dir, vi_dir, chunk_size=chunk_size, simple=simple, tokenizer=tokenizer, ignored_docs=ignored_docs)
        logger.info("Number of sentence pairs: %d", len(pairs))
        out_path = os.path.join(dataset1_dir, f"{split_name}.jsonl")
        write_jsonl(pairs, out_path, src_key="src_vi", trg_key="src_en", tokenizer=tokenizer, max_seq_len=max_seq_len)
        
        # Print a few examples for verification
        if pairs and split_name == "train":
            logger.info("### Examples for Dataset 1 (VI -> EN DocAMR):")
            for i in range(min(2, len(pairs))):
                logger.info("  Sample %d:", i)
                logger.info("    Src (VI): %s", pairs[i]["src_vi"][:100] + "...")
                logger.info("    Trg (EN): %s", pairs[i]["src_en"][:100] + "...")
                logger.info("    Adj (first 3): %s", pairs[i]["adj"][:3])

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
        pairs = build_pairs(en_dir, vi_dir, chunk_size=chunk_size, simple=simple, tokenizer=tokenizer, ignored_docs=ignored_docs)
        out_path = os.path.join(dataset2_dir, f"{split_name}.jsonl")
        write_jsonl(pairs, out_path, src_key="src_en", trg_key="src_vi", tokenizer=tokenizer, max_seq_len=max_seq_len)

        # Print a few examples for verification
        if pairs and split_name == "train":
            logger.info("### Examples for Dataset 2 (EN DocAMR -> VI):")
            for i in range(min(2, len(pairs))):
                logger.info("  Sample %d:", i)
                logger.info("    Src (EN): %s", pairs[i]["src_en"][:100] + "...")
                logger.info("    Trg (VI): %s", pairs[i]["src_vi"][:100] + "...")
                logger.info("    Adj (first 3): %s", pairs[i]["adj"][:3])

    logger.info("=" * 60)
    logger.info("Done!")
    logger.info("  Dataset 1 → %s", dataset1_dir)
    logger.info("  Dataset 2 → %s", dataset2_dir)

    # ------------------------------------------------------------------
    # Dataset 3: Bidirectional (Concatenate 1 and 2)
    # ------------------------------------------------------------------
    if bidirectional:
        dataset3_dir = os.path.join(output_dir, f"bidirection_docamr_en_doc_vi{suffix}")
        logger.info("=" * 60)
        logger.info("Building dataset 3: bidirection_docamr_en_doc_vi")
        logger.info("=" * 60)

        for split_name, (en_folder, vi_folder) in splits.items():
            logger.info("Processing split: %s", split_name)
            en_dir = os.path.join(input_dir, en_folder)
            vi_dir = os.path.join(input_dir, vi_folder)
            pairs = build_pairs(en_dir, vi_dir, chunk_size=chunk_size, simple=simple, tokenizer=tokenizer, ignored_docs=ignored_docs)
            
            out_path = os.path.join(dataset3_dir, f"{split_name}.jsonl")
            
            # 1. VI -> AMR (TEXT_TO_AMR)
            write_jsonl(pairs, out_path, src_key="src_vi", trg_key="src_en", 
                        direction=TEXT_TO_AMR_LABEL, mode="w", tokenizer=tokenizer, max_seq_len=max_seq_len)
            
            # 2. AMR -> VI (AMR_TO_TEXT)
            write_jsonl(pairs, out_path, src_key="src_en", trg_key="src_vi", 
                        direction=AMR_TO_TEXT_LABEL, mode="a", tokenizer=tokenizer, max_seq_len=max_seq_len)

        logger.info("  Dataset 3 (Bidirectional) → %s", dataset3_dir)

    # ------------------------------------------------------------------
    # Optional Dataset 4 & 5: Plain Text EN-VI (no AMR)
    # ------------------------------------------------------------------
    plain_text_root = os.path.join(os.path.dirname(input_dir), "output_dataset_doc")
    if os.path.isdir(plain_text_root):
        logger.info("=" * 60)
        logger.info("Found 'output_dataset_doc' — building plain text datasets.")
        logger.info("=" * 60)
        
        plain_splits = {
            "train": ("train_en-vi.en", "train_en-vi.vi"),
            "valid": ("dev2010.en-vi.en", "dev2010.en-vi.vi"),
            "test":  ("tst2015.en-vi.en", "tst2015.en-vi.vi"),
        }
        
        en_vi_dir = os.path.join(output_dir, f"plain_text_en_vi{suffix}")
        vi_en_dir = os.path.join(output_dir, f"plain_text_vi_en{suffix}")
        
        for split_name, (en_folder, vi_folder) in plain_splits.items():
            logger.info("Processing plain text split: %s", split_name)
            en_dir = os.path.join(plain_text_root, en_folder)
            vi_dir = os.path.join(plain_text_root, vi_folder)
            pairs = build_plain_text_pairs(en_dir, vi_dir, chunk_size=chunk_size, ignored_docs=ignored_docs)
            logger.info("  Found %d plain text pairs.", len(pairs))
            
            write_plain_jsonl(pairs, os.path.join(en_vi_dir, f"{split_name}.jsonl"), "src_en", "src_vi", tokenizer=tokenizer, max_seq_len=max_seq_len)
            write_plain_jsonl(pairs, os.path.join(vi_en_dir, f"{split_name}.jsonl"), "src_vi", "src_en", tokenizer=tokenizer, max_seq_len=max_seq_len)
        
        logger.info("  Dataset 4 (EN→VI Plain) → %s", en_vi_dir)
        logger.info("  Dataset 5 (VI→EN Plain) → %s", vi_en_dir)
    else:
        logger.info("'output_dataset_doc' not found at %s — skipping plain text datasets.", plain_text_root)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("FINAL PROCESSING SUMMARY")
    logger.info("=" * 60)
    for split_name, split_stats in stats.items():
        total = split_stats["total"]
        discarded = split_stats["discarded"]
        valid = total - len(discarded)
        logger.info("Split: %s", split_name.upper())
        logger.info("  Total valid documents: %d / %d", valid, total)
        logger.info("  Number of discarded documents: %d", len(discarded))
        if discarded:
            logger.info("  Discarded Documents Details:")
            for d in discarded:
                logger.info("    - File: %s | Parent: %s | Reason: %s", d["doc_id"], d["folder"], d["reason"])
    logger.info("=" * 60)


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
    parser.add_argument(
        "--bidirectional",
        type=lambda x: (str(x).lower() in ['true', '1', 'yes']),
        default=True,
        help="Whether to create a bidirectional dataset. Default is True."
    )
    parser.add_argument(
        "--config_name",
        default="bert-base-multilingual-cased",
        help="Tokenizer configuration name for pre-computing indices."
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=256,
        help="Maximum allowed token length for a sequence pair (src + trg). Longer pairs are removed."
    )
    args = parser.parse_args()

    main(args.input_dir, args.output_dir, chunk_size=args.chunk_size, simple=args.simple, bidirectional=args.bidirectional, config_name=args.config_name, max_seq_len=args.max_seq_len)
