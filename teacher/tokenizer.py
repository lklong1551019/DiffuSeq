"""
Joint EN+VI BPE tokenizer for the teacher.

- Model: BPE, vocabulary --vocab_size (16,000 by default), trained on train-split sources and targets only.
- Normalizer: NFC (composed Vietnamese diacritics, so "ệ" is one code point in every sentence).
- Pre-tokenizer / decoder: Metaspace — spaces become "▁" and are restored on decoding, so
  decode(encode(text)) == NFC(text) exactly (no detokenizer needed for BLEU).
- Post-processor: appends </s> to every sequence.
- Special ids (fixed order): <pad>=0, <unk>=1, <s>=2, </s>=3. Marian uses <pad> as the decoder start token.

Example: "Tôi hát ." -> ["▁Tôi", "▁hát", "▁.", "</s>"] (exact pieces depend on the learned merges).
"""

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers
from transformers import AutoTokenizer, PreTrainedTokenizerFast

SPECIAL_TOKENS = ["<pad>", "<unk>", "<s>", "</s>"]
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3


def train_tokenizer(texts, vocab_size=16000, min_frequency=2):
    """Train the joint BPE on an iterable of raw strings; returns a PreTrainedTokenizerFast."""
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Metaspace()
    tok.decoder = decoders.Metaspace()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=min_frequency,
                                  special_tokens=SPECIAL_TOKENS, show_progress=False)
    tok.train_from_iterator(texts, trainer)
    assert [tok.token_to_id(t) for t in SPECIAL_TOKENS] == [PAD_ID, UNK_ID, BOS_ID, EOS_ID]
    tok.post_processor = processors.TemplateProcessing(single="$A </s>", special_tokens=[("</s>", EOS_ID)])
    return PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", unk_token="<unk>",
                                   bos_token="<s>", eos_token="</s>")


def load_tokenizer(path):
    """Load a tokenizer saved with save_pretrained (also the form build_kd_dataset.py loads)."""
    tok = AutoTokenizer.from_pretrained(path)
    assert tok.pad_token_id == PAD_ID and tok.eos_token_id == EOS_ID, "unexpected special ids"
    return tok
