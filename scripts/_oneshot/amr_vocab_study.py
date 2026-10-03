"""
AMR vocabulary study (2026-10-03) — numbers behind docs/data/amr-vocabulary.md.

Measures on the aligned train split (metadata lines dropped; chunk 1):
  1. inventory: relation labels, inverse roles, concepts, sense-tagged predicates, -9x frames, frequencies
  2. tokenization cost when relations / frames are NOT added (base mBERT WordPiece)
  3. sharing: AMR concept tokens whose id also occurs in the same row's English sentence
  4. substring damage: plain EN / VI sentences tokenized differently after adding a candidate vocabulary
  5. coverage: test-split frames / relations unseen in train

Run from the repo root (CPU only, thesis_env): CUDA_VISIBLE_DEVICES="" python scripts/_oneshot/amr_vocab_study.py
"""

import json
import os
import random
import re
import sys
from collections import Counter

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import paths  # noqa: E402
from diffuseq.amr_linearize import SENSE_RE, SPECIAL_FRAME_RE, linearize_sentences  # noqa: E402
from prepare_docamr_datasets import chunk_documents, load_split  # noqa: E402

BASE = "bert-base-multilingual-cased"


def tokenizer_with(tokens):
    t = AutoTokenizer.from_pretrained(BASE)
    t.add_tokens(sorted(set(tokens)))
    return t


def concepts_and_relations(chunks, drop_sense):
    concepts, relations = Counter(), Counter()
    for c in chunks:
        lin = linearize_sentences(c["trees"], drop_sense=drop_sense)
        rel_words = {lw for _, lw, _, _ in lin.triples}
        for i, w in enumerate(lin.words):
            if i in rel_words:
                relations[w] += 1
            elif w not in ("(", ")"):
                concepts[w] += 1
    return concepts, relations


def main():
    rng = random.Random(0)
    train = chunk_documents(load_split("train", Counter()), 1, Counter())
    test = chunk_documents(load_split("test", Counter()), 1, Counter())
    out = {}

    # 1. inventory (full train, senses kept)
    concepts, relations = concepts_and_relations(train, drop_sense=False)
    relations = Counter({r: n for r, n in relations.items() if not re.fullmatch(r":snt\d+", r)})
    predicates = Counter({c: n for c, n in concepts.items() if SENSE_RE.search(c) and not SPECIAL_FRAME_RE.search(c)})
    frames91 = Counter({c: n for c, n in concepts.items() if SPECIAL_FRAME_RE.search(c)})
    out["inventory"] = {
        "relation_labels": len(relations), "relation_occurrences": sum(relations.values()),
        "inverse_labels": sum(1 for r in relations if r.endswith("-of")),
        "inverse_occurrences": sum(n for r, n in relations.items() if r.endswith("-of")),
        "distinct_concept_strings": len(concepts),
        "sense_tagged_predicates": len(predicates), "predicate_occurrences": sum(predicates.values()),
        "predicates_seen_lt5": sum(1 for n in predicates.values() if n < 5),
        "predicates_seen_lt20": sum(1 for n in predicates.values() if n < 20),
        "frames_9x": len(frames91), "frame_9x_occurrences": sum(frames91.values()),
        "frames_9x_seen_lt5": sum(1 for n in frames91.values() if n < 5),
        "top_relations": relations.most_common(8), "top_frames_9x": frames91.most_common(6),
    }

    # 2. tokenization cost without added tokens
    base = AutoTokenizer.from_pretrained(BASE)
    pieces = lambda w: len(base.tokenize(w))
    rel_cost = sum(pieces(r) * n for r, n in relations.items()) / sum(relations.values())
    f91_cost = sum(pieces(f) * n for f, n in frames91.items()) / max(sum(frames91.values()), 1)
    pred_cost_kept = sum(pieces(p) * n for p, n in predicates.items()) / sum(predicates.values())
    pred_cost_dropped = sum(pieces(SENSE_RE.sub("", p)) * n for p, n in predicates.items()) / sum(predicates.values())
    out["wordpieces_per_item_without_adding"] = {
        "relation": round(rel_cost, 2), "frame_9x": round(f91_cost, 2),
        "predicate_with_sense": round(pred_cost_kept, 2), "predicate_sense_dropped": round(pred_cost_dropped, 2),
        "examples": {w: base.tokenize(w) for w in (":ARG0", ":ARG1-of", ":polarity", "have-rel-role-91", "abandon-01", "abandon")},
    }

    # 3. sharing with the English sentence (20k sample)
    sample = rng.sample(train, 20000)
    rel_tokens = list(json.load(open(paths.AMR_VOCAB_FILE))["relations"])
    setups = {
        "relations + -9x frames, sense dropped (shipped)": (tokenizer_with(rel_tokens + list(frames91)), True),
        "relations + all predicates as tokens, sense kept": (tokenizer_with(rel_tokens + list(frames91) + list(predicates)), False),
    }
    out["sharing_with_english"] = {}
    for name, (tok, drop) in setups.items():
        shared = total = 0
        for c in sample:
            en_ids = set(tok(c["en"], add_special_tokens=False)["input_ids"])
            lin = linearize_sentences(c["trees"], drop_sense=drop)
            rel_words = {lw for _, lw, _, _ in lin.triples}
            for i, w in enumerate(lin.words):
                if i in rel_words or w in ("(", ")"):
                    continue
                first = tok(w, add_special_tokens=False)["input_ids"][:1]
                total += 1
                shared += bool(first) and first[0] in en_ids
        out["sharing_with_english"][name] = round(shared / total * 100, 1)

    # 4. substring damage on plain text (20k EN + 20k VI sentences)
    plain = [c["en"] for c in sample] + [c["vi"] for c in sample]
    base_ids = base(plain)["input_ids"]
    candidates = {
        "relations + -9x frames (shipped)": rel_tokens + list(frames91),
        "+ all sense-tagged predicates (abandon-01 ...)": rel_tokens + list(frames91) + list(predicates),
        "+ all concepts, senses dropped (bare words)": rel_tokens + [SENSE_RE.sub("", c) for c in concepts],
    }
    out["plain_text_changed_pct"] = {}
    for name, toks in candidates.items():
        ids = tokenizer_with(toks)(plain)["input_ids"]
        out["plain_text_changed_pct"][name] = {
            "added_tokens": len(set(toks)),
            "en": round(np.mean([a != b for a, b in zip(ids[:20000], base_ids[:20000])]) * 100, 2),
            "vi": round(np.mean([a != b for a, b in zip(ids[20000:], base_ids[20000:])]) * 100, 2),
        }

    # 5. test-split coverage by the train inventory
    t_concepts, t_relations = concepts_and_relations(test, drop_sense=False)
    t_pred = Counter({c: n for c, n in t_concepts.items() if SENSE_RE.search(c) and not SPECIAL_FRAME_RE.search(c)})
    out["test_unseen_in_train_pct"] = {
        "relation_occurrences": round(sum(n for r, n in t_relations.items() if r not in relations and not r.startswith(":snt")) / sum(t_relations.values()) * 100, 3),
        "predicate_occurrences": round(sum(n for p, n in t_pred.items() if p not in predicates) / sum(t_pred.values()) * 100, 2),
        "predicate_types": round(sum(1 for p in t_pred if p not in predicates) / len(t_pred) * 100, 1),
    }
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
