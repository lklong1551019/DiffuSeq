"""prepare_docamr_datasets.py — alignment, metadata drop (B10), no test filtering (B7), graph positions."""

import json
import os

import pytest

import paths
import prepare_docamr_datasets as prep
from basic_utils import AMR_TO_TEXT_LABEL, TEXT_TO_AMR_LABEL
from diffuseq.text_datasets import merge_pair

EN = ["http://www.ted.com/talks/x</url>", "I sing .", "I want to go ."]
VI = ["http://www.ted.com/talks/x</url>", "Tôi hát .", "Tôi muốn đi ."]
AMR = """# ::id doc-1
# ::tok http : / / www . ted . com / talks / x< / url> <next_sent> I sing . <next_sent> I want to go .
(d / document
   :snt1 (s1.u / url-entity :value "http://www.ted.com/talks/x")
   :snt2 (s2.s / sing-01 :ARG0 (s2.i / i))
   :snt3 (s3.w / want-01 :ARG0 (s3.i / i) :ARG1 (s3.g / go-02 :ARG0 s3.i)))
"""


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    """Three splits, each with doc-1 (aligned) and doc-2 (EN 2 lines, AMR 1 sentence -> skipped)."""
    text_dir, amr_dir = tmp_path / "text", tmp_path / "amr"
    for split, (en_f, vi_f) in paths.SPLIT_FOLDERS.items():
        for d in (text_dir / en_f, text_dir / vi_f, amr_dir / en_f):
            d.mkdir(parents=True)
        (text_dir / en_f / "doc-1.txt").write_text("\n".join(EN) + "\n", encoding="utf-8")
        (text_dir / vi_f / "doc-1.txt").write_text("\n".join(VI) + "\n", encoding="utf-8")
        (amr_dir / en_f / "doc-1_docamr_docAMR.out").write_text(AMR, encoding="utf-8")
        (text_dir / en_f / "doc-2.txt").write_text("A .\nB .\n", encoding="utf-8")
        (text_dir / vi_f / "doc-2.txt").write_text("A .\nB .\n", encoding="utf-8")
        (amr_dir / en_f / "doc-2_docamr_docAMR.out").write_text(
            "# ::tok A .\n(d / document :snt1 (s1.a / a))\n", encoding="utf-8")
    monkeypatch.setattr(paths, "TEXT_DOC_DIR", str(text_dir))
    monkeypatch.setattr(paths, "AMR_DOC_DIR", str(amr_dir))
    monkeypatch.setattr(paths, "DOCAMR_DIR", str(tmp_path / "out"))
    monkeypatch.setattr(paths, "LOGS_DIR", str(tmp_path / "logs"))
    return tmp_path


def test_load_split_alignment_and_skip(corpus):
    from collections import Counter
    stats = Counter()
    docs = prep.load_split("train", stats)
    assert len(docs) == 1 and len(docs[0]) == 3
    assert stats["docs_used"] == 1 and stats["skip_count_mismatch"] == 1


def test_metadata_sentences_dropped_before_chunking(corpus):
    from collections import Counter
    stats = Counter()
    chunks = prep.chunk_documents(prep.load_split("train", stats), chunk_size=2, stats=stats)
    assert stats["metadata_sentences_dropped"] == 1
    assert [c["en"] for c in chunks] == ["I sing . I want to go ."]


def test_rows_per_variant(corpus, my_tokenizer):
    from collections import Counter
    chunk = prep.chunk_documents(prep.load_split("train", Counter()), 1, Counter())[0]
    hf = my_tokenizer.tokenizer
    assert prep.build_rows(chunk, "plain_en_vi", hf, True) == [{"src": "I sing .", "trg": "Tôi hát ."}]
    (row,) = prep.build_rows(chunk, "text_amr_en_vi", hf, True)
    assert row["src"] == "I sing . [SEP] ( sing :ARG0 i )"
    toks = hf.convert_ids_to_tokens(hf(row["src"])["input_ids"])
    assert row["graph_src"] == [[6, 8, ":ARG0", 7]]                 # walkthrough §3.2
    assert [toks[p] for p in (6, 7, 8)] == ["sing", ":ARG0", "i"]
    bi = prep.build_rows(chunk, "bidirectional", hf, True)
    assert [r["direction"] for r in bi] == [TEXT_TO_AMR_LABEL, AMR_TO_TEXT_LABEL]
    assert "graph_src" not in bi[0] and bi[1]["graph_src"] == [[2, 4, ":ARG0", 3]]


def test_merged_length_matches_merge_pair(corpus, my_tokenizer):
    row = {"src": "I sing . [SEP] ( sing :ARG0 i )", "trg": "Tôi hát ."}
    n = prep.merged_length(row, my_tokenizer)
    src = my_tokenizer.encode_token([row["src"]])[0]
    trg = my_tokenizer.encode_token([row["trg"]])[0]
    exact = merge_pair(src, trg, [0] * len(trg), my_tokenizer.sep_token_id, seq_len=n)
    assert len(exact["input_ids"]) == n                              # fits exactly, nothing trimmed
    trimmed = merge_pair(src, trg, [0] * len(trg), my_tokenizer.sep_token_id, seq_len=n - 1)
    assert len(trimmed["input_ids"]) == n - 1


def test_main_filters_train_only(corpus, my_tokenizer):
    args = prep.parse_args(["--chunk_size", "1", "--variants", "plain_en_vi,text_amr_en_vi",
                            "--max_seq_len", "12", "--prefix", "t"])
    prep.main(args)
    out = os.path.join(paths.DOCAMR_DIR, "t_text_amr_en_vi_chunk_1")
    count = lambda split: sum(1 for _ in open(os.path.join(out, f"{split}.jsonl"), encoding="utf-8"))
    assert count("train") == 0                       # both rows exceed 12 tokens -> dropped
    assert count("test") == 2 and count("valid") == 2  # never filtered (B7)
    meta = json.load(open(os.path.join(out, "meta.json"), encoding="utf-8"))
    assert meta["counts"]["train"]["chunks_dropped_too_long"] == 2
    # joint filter: the plain variant drops the same chunks, so both train sets hold the same sentences
    plain = os.path.join(paths.DOCAMR_DIR, "t_plain_en_vi_chunk_1", "train.jsonl")
    assert sum(1 for _ in open(plain, encoding="utf-8")) == 0
    first_test = json.loads(open(os.path.join(out, "test.jsonl"), encoding="utf-8").readline())
    assert "url" not in first_test["src"]
