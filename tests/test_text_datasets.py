"""diffuseq/text_datasets.py — layout, masks, graph shifting, collate (walkthroughs §3.1–§3.3)."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import diffuseq.text_datasets as td
from basic_utils import AMR_TO_TEXT_LABEL, TEXT_TO_AMR_LABEL

CLS, SEP, PAD = 101, 102, 0
# Real mBERT ids of the §3.1 example: "Thank you ." / "Cảm ơn các bạn ."
SRC = [CLS, 91327, 13028, 119, SEP]
TRG = [CLS, 140, 102539, 375, 10115, 10792, 43094, 119, SEP]


# ------------------------------------------------------------------------------ merge_pair / pad_to

def test_merge_pair_walkthrough_3_1():
    m = td.merge_pair(SRC, TRG, [0] * len(TRG), SEP, seq_len=16)
    assert m["input_ids"] == SRC + [SEP] + TRG
    assert m["input_mask"] == [0] * 6
    assert m["src_len"] == 5 and m["trg_start"] == 6
    ids = td.pad_to(m["input_ids"], PAD, 16)
    mask = td.pad_to(m["input_mask"], 1, 16)
    assert ids == [101, 91327, 13028, 119, 102, 102, 101, 140, 102539, 375, 10115, 10792, 43094, 119, 102, 0]
    assert mask == [0] * 6 + [1] * 10
    assert ids[m["trg_start"]] == CLS


def test_merge_pair_exact_fit_is_not_trimmed():
    # merged length = len(src) + 1 + len(trg) = 5 + 1 + 9 = 15
    m = td.merge_pair(SRC, TRG, [0] * len(TRG), SEP, seq_len=15)
    assert len(m["input_ids"]) == 15 and m["input_ids"] == SRC + [SEP] + TRG


def test_merge_pair_trims_longer_side_first():
    # seq_len 13: src body 4, trg body 8 -> pop trg until 4 + 6 <= 10
    m = td.merge_pair(SRC, TRG, [0] * len(TRG), SEP, seq_len=13)
    assert m["input_ids"][:5] == SRC                          # source untouched
    assert m["input_ids"][5] == SEP and m["input_ids"][-1] == SEP
    assert len(m["input_ids"]) == 13


def test_merge_pair_trims_both_when_equal():
    src = [CLS, 1, 2, 3, SEP]
    trg = [CLS, 4, 5, 6, SEP]
    m = td.merge_pair(src, trg, [0, 0, 0, 1, 0], SEP, seq_len=9)  # bodies 4 + 4 > 6 -> both pop once
    assert m["input_ids"] == [CLS, 1, 2, SEP, SEP, CLS, 4, 5, SEP]
    assert m["rel_mask"] == [0, 0, 0, 0, 0, 0, 0, 0, 0]          # trimmed rel position removed alike


def test_rel_mask_alignment_through_merge():
    rel = [0, 0, 0, 1, 0, 0, 0]                                   # [CLS] ( sing :ARG0 i ) [SEP]
    trg = [CLS, 113, 21253, 119547, 177, 114, SEP]
    m = td.merge_pair(SRC, trg, rel, SEP, seq_len=20)
    assert m["rel_mask"][m["trg_start"] + 3] == 1
    assert sum(m["rel_mask"]) == 1 and len(m["rel_mask"]) == len(m["input_ids"])


def test_pad_to_rejects_overflow():
    with pytest.raises(AssertionError):
        td.pad_to([1, 2, 3], 0, 2)


# ------------------------------------------------------------------------------ rel mask

def test_build_rel_mask_only_for_text_to_amr():
    toks = ["[CLS]", "(", "sing", ":ARG0", "i", ")", "[SEP]"]
    assert td.build_rel_mask(toks, TEXT_TO_AMR_LABEL) == [0, 0, 0, 1, 0, 0, 0]
    assert td.build_rel_mask(toks, AMR_TO_TEXT_LABEL) == [0] * 7
    assert td.build_rel_mask(["[CLS]", "Note", ":", "x", "[SEP]"], None) == [0] * 5   # plain text colon


def test_build_rel_mask_marks_wordpiece_continuations():
    toks = ["[CLS]", ":", "##prep", "x", "[SEP]"]
    assert td.build_rel_mask(toks, TEXT_TO_AMR_LABEL) == [0, 1, 1, 0, 0]


# ------------------------------------------------------------------------------ source graph shift

def test_shift_source_graph_direction_token():
    assert td.shift_source_graph([[2, 4, ":ARG0", 3]], True, 8) == [[3, 5, ":ARG0", 4]]
    assert td.shift_source_graph([[2, 4, ":ARG0", 3]], False, 8) == [[2, 4, ":ARG0", 3]]


def test_shift_source_graph_drops_trimmed_and_sep_positions():
    # src_len 5 -> real tokens at 1..3; after the +1 shift dep (5) and label (4) are out
    assert td.shift_source_graph([[2, 4, ":ARG0", 3]], True, 5) == []
    # src_len 6 -> real tokens 1..4; dep 4 ok, but a node on position 5 ([SEP]) is dropped
    assert td.shift_source_graph([[1, 4, ":r", 2], [1, 5, ":r", 2]], False, 6) == [[1, 4, ":r", 2]]


# ------------------------------------------------------------------------------ decode split

def test_split_source_target():
    seq = list(range(16))
    src, trg = td.split_source_target(seq, [0] * 6 + [1] * 10)
    assert src == list(range(6)) and trg == list(range(6, 16))
    src, trg = td.split_source_target(seq[:20], [0] * 12 + [1] * 4)
    assert len(src) == 12


def test_split_source_target_rejects_malformed_mask():
    with pytest.raises(AssertionError):
        td.split_source_target([1, 2, 3, 4], [0, 1, 0, 1])


# ------------------------------------------------------------------------------ edges + collate

def test_build_edges_levi():
    ei, et = td.build_edges([[6, 8, ":ARG0", 7]], "levi")
    assert ei.tolist() == [[6, 7, 7, 8], [7, 8, 6, 7]]
    assert et.tolist() == [0, 1, 2, 3]


def test_build_edges_edge_attr():
    ei, et = td.build_edges([[6, 8, ":ARG0", 7], [6, 9, ":unseen", 7]], "edge_attr", {":ARG0": 3})
    assert ei.tolist() == [[6, 8, 6, 9], [8, 6, 9, 6]]
    assert et.tolist() == [6, 7, 0, 1]                       # unknown relation -> id 0


def test_build_edges_empty():
    ei, et = td.build_edges([], "levi")
    assert tuple(ei.shape) == (2, 0) and tuple(et.shape) == (0,)


def test_collate_offsets_and_empty_sample():
    L, D = 20, 4

    def sample(edges):
        ei = torch.tensor(edges, dtype=torch.long).t() if edges else torch.empty((2, 0), dtype=torch.long)
        et = torch.zeros(ei.shape[1], dtype=torch.long)
        return np.zeros((L, D), dtype=np.float32), {"input_ids": np.zeros(L, dtype=np.int64),
                                                     "input_mask": np.ones(L, dtype=np.int64),
                                                     "edge_index": ei, "edge_type": et}

    batch, cond = td.collate_with_adj([sample([[6, 7]]), sample([]), sample([[6, 7], [7, 8]])])
    assert tuple(batch.shape) == (3, L, D)
    assert cond["edge_index"].tolist() == [[6, 46, 47], [7, 47, 48]]   # sample 2 offset 2 * 20
    assert cond["edge_type"].shape[0] == 3


# ------------------------------------------------------------------------------ integration

def test_helper_tokenize_end_to_end(my_tokenizer, monkeypatch):
    monkeypatch.setattr(td, "TOKENIZE_NUM_PROC", 1)
    rows = {
        "src": ["Thank you .", "( sing :ARG0 i )"],
        "trg": ["Cảm ơn các bạn .", "Tôi hát ."],
        "direction": [None, AMR_TO_TEXT_LABEL],
        "graph_src": [json.dumps([]), json.dumps([[2, 4, ":ARG0", 3]])],
    }
    ds = td.helper_tokenize(rows, my_tokenizer, seq_len=16)["train"]
    first, second = ds[0], ds[1]
    assert first["input_ids"][:6] == SRC + [SEP] and first["input_mask"] == [0] * 6 + [1] * 10
    for row in (first, second):
        assert len(row["input_ids"]) == len(row["input_mask"]) == len(row["rel_mask"]) == 16
    # direction token inserted at position 1 -> AMR positions move +1
    toks = my_tokenizer.tokenizer.convert_ids_to_tokens(second["input_ids"])
    assert toks[1] == "[AMR_TO_TEXT]"
    graph = json.loads(second["graph"])
    assert graph == [[3, 5, ":ARG0", 4]]
    assert [toks[p] for p in (3, 4, 5)] == ["sing", ":ARG0", "i"]


def test_text_dataset_builds_edges_only_with_graph_encoder(my_tokenizer, monkeypatch):
    monkeypatch.setattr(td, "TOKENIZE_NUM_PROC", 1)
    rows = {"src": ["( sing :ARG0 i )"], "trg": ["Tôi hát ."], "direction": [None],
            "graph_src": [json.dumps([[2, 4, ":ARG0", 3]])]}
    data = td.helper_tokenize(rows, my_tokenizer, seq_len=16)
    emb = torch.nn.Embedding(my_tokenizer.vocab_size, 4)
    plain = td.TextDataset(data, SimpleNamespace(graph_encoder="none"), model_emb=emb)
    assert "edge_index" not in plain[0][1]
    levi = td.TextDataset(data, SimpleNamespace(graph_encoder="gatv2", graph_mode="levi"), model_emb=emb)
    assert levi[0][1]["edge_index"].tolist() == [[2, 3, 3, 4], [3, 4, 2, 3]]
