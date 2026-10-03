"""diffuseq/amr_linearize.py — linearization (bugs B2, B3, B9), parsing, word -> token mapping."""

import penman
import pytest

from diffuseq.amr_linearize import (linearize_sentences, map_words_to_tokens, parse_document,
                                    sanitize_docamr, triples_to_token_graph, word_char_starts)

DOC = (
    "(d / document"
    " :snt1 (s1.s / sing-01 :ARG0 (s1.i / i) :ARG1-of (s1.f / fast-02) :polarity -)"
    " :snt2 (s2.w / want-01 :ARG0 s1.i :ARG1 (s2.g / go-02 :ARG0 s2.w2)"
    "        :same-as s1.s)"
    " :snt3 (s3.c / country :name (s3.n / name :op1 \"U.S.\") :ARG1-of (s3.h / have-org-role-91)))"
)


def trees():
    return parse_document(DOC)


def test_parse_document_orders_sentences():
    ts = trees()
    assert [t.node[0] for t in ts] == ["s1.s", "s2.w", "s3.c"]


def test_single_sentence_file_without_document_root():
    ts = parse_document("(s1.u / url-entity :value \"x\")")
    assert len(ts) == 1 and ts[0].node[0] == "s1.u"


def test_brackets_relations_and_leaves():
    lin = linearize_sentences([trees()[0]], drop_sense=True)
    assert lin.words == ["(", "sing", ":ARG0", "i", ":ARG1-of", "fast", ":polarity", "-", ")"]
    # (head, label, dep, label string), word indices
    assert lin.triples == [(1, 2, 3, ":ARG0"), (1, 4, 5, ":ARG1-of"), (1, 6, 7, ":polarity")]
    assert lin.text.count("(") == lin.text.count(")")


def test_argument_numbers_and_sense_kept_when_requested():
    lin = linearize_sentences([trees()[0]], drop_sense=False)
    assert "sing-01" in lin.words and ":ARG0" in lin.words and ":ARG1-of" in lin.words


def test_special_frames_keep_their_suffix():
    lin = linearize_sentences([trees()[2]], drop_sense=True)
    assert "have-org-role-91" in lin.words
    assert "U.S." in lin.words                      # quotes stripped from constants


def test_cross_sentence_reference_dropped_in_single_sentence_chunk():
    lin = linearize_sentences([trees()[1]], drop_sense=True)
    # s1.i, s2.w2 and the :same-as target s1.s are not defined in sentence 2
    assert lin.words == ["(", "want", ":ARG1", "go", ")"]
    assert not any(w.startswith("s1.") or w.startswith("s2.") for w in lin.words)


def test_reentrancy_resolved_inside_multi_sentence_chunk():
    lin = linearize_sentences(trees()[:2], drop_sense=True)
    want_arg0 = [t for t in lin.triples if t[3] == ":ARG0" and lin.words[t[0]] == "want"]
    assert len(want_arg0) == 1
    head, _, dep, _ = want_arg0[0]
    assert lin.words[dep] == "i" and dep == 3      # edge to the defining occurrence in sentence 1
    same_as = [t for t in lin.triples if t[3] == ":same-as"]
    assert len(same_as) == 1 and lin.words[same_as[0][2]] == "sing"


def test_sanitize_repairs_parser_concepts():
    bad = "(d / document :snt1 (s1.a / 1/3 :mod (s1.o / only)) :snt2 (s2.u / Auf Wiedersehen))"
    with pytest.raises(Exception):
        penman.parse(bad)
    ts = parse_document(bad)
    assert len(ts) == 2
    lin = linearize_sentences(ts)
    assert "1/3" in lin.words and "Auf_Wiedersehen" in lin.words
    assert sanitize_docamr("(s / sing-01)") == "(s / sing-01)"


def test_word_char_starts():
    assert word_char_starts(["(", "sing", ":ARG0"]) == [0, 2, 7]
    assert word_char_starts(["(", "sing"], offset=10) == [10, 12]


def test_map_words_to_tokens_amr_only(my_tokenizer):
    tok = my_tokenizer.tokenizer
    text = "( sing :ARG0 i )"
    words = text.split()
    pos = map_words_to_tokens(text, word_char_starts(words), tok)
    assert pos == [1, 2, 3, 4, 5]
    assert tok.convert_ids_to_tokens(tok(text)["input_ids"])[pos[2]] == ":ARG0"


def test_map_words_to_tokens_with_text_prefix_and_literal(my_tokenizer):
    """Walkthrough §3.2: [CLS] I sing . [SEP] ( sing :ARG0 i ) [SEP]."""
    tok = my_tokenizer.tokenizer
    prefix = "I sing . [SEP] "
    words = ["(", "sing", ":ARG0", "i", ")"]
    pos = map_words_to_tokens(prefix + " ".join(words), word_char_starts(words, len(prefix)), tok)
    assert pos == [5, 6, 7, 8, 9]
    # a literal split by the pre-tokenizer maps to its first piece
    text = "( name :op1 U.S. )"
    w = text.split()
    pos = map_words_to_tokens(text, word_char_starts(w), tok)
    assert tok.convert_ids_to_tokens(tok(text)["input_ids"])[pos[3]] == "U"


def test_triples_to_token_graph_drops_unmapped():
    graph = triples_to_token_graph([(1, 2, 3, ":ARG0"), (1, 4, 5, ":mod")], [0, 2, 3, 4, None, 6])
    assert graph == [[2, 4, ":ARG0", 3]]
