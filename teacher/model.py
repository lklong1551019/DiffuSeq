"""
Teacher model: Marian encoder-decoder built from a config (random initialization).

Presets:
  iwslt  fairseq transformer_iwslt_de_en sizes: d_model 512, FFN 1024, 6+6 layers, 4 heads (40.3M params
         with a 16k vocabulary)
  tiny   d_model 64, FFN 128, 1+1 layers, 4 heads — CPU tests only

Conventions (Marian): decoder start token = <pad>; </s> ends every sequence; encoder, decoder and output
projection share one embedding matrix; sinusoidal positions; embeddings scaled by sqrt(d_model).
"""

from transformers import MarianConfig, MarianMTModel

PRESETS = {
    "iwslt": dict(d_model=512, encoder_ffn_dim=1024, decoder_ffn_dim=1024, encoder_layers=6, decoder_layers=6,
                  encoder_attention_heads=4, decoder_attention_heads=4),
    "tiny": dict(d_model=64, encoder_ffn_dim=128, decoder_ffn_dim=128, encoder_layers=1, decoder_layers=1,
                 encoder_attention_heads=4, decoder_attention_heads=4),
}


def build_model(preset, tokenizer, dropout=0.3, attention_dropout=0.1, activation_dropout=0.1,
                max_positions=512):
    if preset not in PRESETS:
        raise ValueError(f"preset must be one of {sorted(PRESETS)}, got {preset!r}")
    config = MarianConfig(
        vocab_size=len(tokenizer),
        decoder_vocab_size=len(tokenizer),
        max_position_embeddings=max_positions,
        dropout=dropout,
        attention_dropout=attention_dropout,
        activation_dropout=activation_dropout,
        activation_function="relu",
        scale_embedding=True,
        share_encoder_decoder_embeddings=True,
        tie_word_embeddings=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=tokenizer.pad_token_id,
        forced_eos_token_id=tokenizer.eos_token_id,
        **PRESETS[preset],
    )
    return MarianMTModel(config)


def count_parameters(model):
    """Unique parameters (the tied embedding counted once)."""
    return sum(p.numel() for p in {id(p): p for p in model.parameters()}.values())
