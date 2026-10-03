"""myTokenizer and the AMR added-token vocabulary (review bug B1)."""

import json
import os

import pytest

import paths
from basic_utils import (AMR_TO_TEXT_TOKEN, TEXT_TO_AMR_TOKEN, amr_added_tokens, validate_added_tokens)

PLAIN_SENTENCES = [
    "http://www.ted.com/talks/rachel_pike_the_science_behind_a_climate_headline</url>",
    "I'd like to talk to you today about the scale of the scientific effort.",
    "The information collapsed, and the aches of age went on.",
    "One year later, live with an AK-47 by my side.",
    "Tôi muốn cho các bạn biết về sự to lớn của những nỗ lực khoa học.",
    "Và chúng hoạt động rất tốt, hệ thống thông tin của họ.",
]


def test_validate_rejects_bare_words():
    with pytest.raises(ValueError):
        validate_added_tokens([":ARG0", "ache"])
    with pytest.raises(ValueError):
        validate_added_tokens(["abandon-01"])        # sense-tagged ordinary predicate
    validate_added_tokens([":ARG0", ":ARG1-of", ":prep-with", "have-org-role-91", "include-91"])


def test_legacy_simple_vocab_is_unsafe():
    if not os.path.exists(paths.LEGACY_AMR_TOKENS_SIMPLE):
        pytest.skip("legacy simple token file not present")
    with pytest.raises(ValueError):
        validate_added_tokens(amr_added_tokens("legacy_simple"))


def test_relations_vocab_is_safe_and_has_inverse_roles():
    tokens = amr_added_tokens("relations")
    validate_added_tokens(tokens)
    assert ":ARG0-of" in tokens and ":ARG1" in tokens
    assert "have-org-role-91" in tokens
    assert not any(t.startswith(":snt") for t in tokens)


def test_plain_text_tokenization_unchanged(my_tokenizer, base_tokenizer):
    """Adding the AMR vocabulary must not change how ordinary EN / VI text is tokenized."""
    sentences = list(PLAIN_SENTENCES)
    test_file = os.path.join(paths.DOCAMR_DIR, "plain_text_en_vi_chunk_1", "test.jsonl")
    if os.path.exists(test_file):
        with open(test_file, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                sentences += [row["src"], row["trg"]]
    ours = my_tokenizer.tokenizer(sentences)["input_ids"]
    base = base_tokenizer(sentences)["input_ids"]
    changed = [s for s, a, b in zip(sentences, ours, base) if a != b]
    assert changed == [], changed[:3]


def test_relation_and_direction_tokens_are_single_ids(my_tokenizer):
    tok = my_tokenizer.tokenizer
    assert tok.tokenize(":ARG0-of") == [":ARG0-of"]
    assert tok.tokenize("have-org-role-91") == ["have-org-role-91"]
    assert tok.tokenize(f"{AMR_TO_TEXT_TOKEN} x")[0] == AMR_TO_TEXT_TOKEN
    assert tok.tokenize(TEXT_TO_AMR_TOKEN) == [TEXT_TO_AMR_TOKEN]


def test_encode_wraps_cls_sep(my_tokenizer):
    ids = my_tokenizer.encode_token(["I sing ."])[0]
    tok = my_tokenizer.tokenizer
    assert ids[0] == tok.cls_token_id and ids[-1] == my_tokenizer.sep_token_id
    assert my_tokenizer.vocab_size == len(tok)


def test_saved_run_tokenizer_is_loaded_verbatim(tmp_path):
    """Decoding loads the run folder's tokenizer instead of rebuilding it from the config."""
    from types import SimpleNamespace
    from basic_utils import myTokenizer
    src = myTokenizer(SimpleNamespace(vocab="bert", config_name="bert-base-multilingual-cased",
                                      amr_vocab="none", checkpoint_path=str(tmp_path)))
    reloaded = myTokenizer(SimpleNamespace(vocab="bert", config_name="bert-base-multilingual-cased",
                                           amr_vocab="relations", tokenizer_dir=str(tmp_path),
                                           checkpoint_path=str(tmp_path)))
    assert reloaded.vocab_size == src.vocab_size   # amr_vocab of the args is ignored
