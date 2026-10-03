"""
basic_utils.py — utilities shared by the training and decoding entry points.

  1. myTokenizer:                wraps a HuggingFace tokenizer (BERT family) or a custom vocab file.
  2. amr_added_tokens / validate_added_tokens: the AMR tokens appended to the tokenizer.
  3. load_model_emb:             the embedding table used by the data loader.
  4. load_defaults_config:       defaults from diffuseq/config.json (every key is also a CLI flag).
  5. create_model_and_diffusion: TransformerNetModel + SpacedDiffusion from hyperparameters.
  6. add_dict_to_argparser / args_to_dict / str2bool: argparse helpers.
"""

import argparse
import json
import os
import re
import time

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerFast

import paths
from diffuseq import gaussian_diffusion as gd
from diffuseq.gaussian_diffusion import SpacedDiffusion, space_timesteps
from diffuseq.transformer_model import TransformerNetModel

# Direction tokens prepended to the source of bidirectional datasets.
AMR_TO_TEXT_LABEL = "AMR_TO_TEXT"
TEXT_TO_AMR_LABEL = "TEXT_TO_AMR"
AMR_TO_TEXT_TOKEN = f"[{AMR_TO_TEXT_LABEL}]"
TEXT_TO_AMR_TOKEN = f"[{TEXT_TO_AMR_LABEL}]"

# amr_vocab modes:
#   relations     relation labels + -9x frames from paths.AMR_VOCAB_FILE (default; safe for plain text)
#   none          no AMR tokens (plain-text vocabulary)
#   legacy_full   doc_amrs_token.json — reproduces runs before 2026-10-03
#   legacy_simple doc_amrs_token_simple.json — reproduces runs before 2026-10-03; corrupts
#                 tokenization of ordinary words (review bug B1), never use for new runs
AMR_VOCAB_MODES = ("relations", "none", "legacy_full", "legacy_simple")

# An added token is safe only if it cannot occur inside ordinary text: a relation label
# (":ARG0", ":ARG1-of", ":prep-with") or a hyphenated -9x frame ("have-org-role-91").
# HF added tokens are matched as raw substrings before WordPiece, so a bare word such as "ache"
# would split "rachel" into "r ache l".
_SAFE_ADDED_TOKEN_RE = re.compile(r"^(:[A-Za-z][\w.-]*|[a-z]+(-[a-z]+)*-9\d)$")


def validate_added_tokens(tokens):
    """Raise ValueError if any token could match inside ordinary English or Vietnamese text."""
    unsafe = [t for t in tokens if not _SAFE_ADDED_TOKEN_RE.match(t)]
    if unsafe:
        raise ValueError(
            f"{len(unsafe)} unsafe added tokens (would split ordinary words), e.g. {unsafe[:10]}"
        )


# Vietnamese probe: tones and diacritics carry meaning, so the tokenizer must reproduce them exactly.
# English-only checkpoints fail it (bert-base-uncased lowercases and strips accents: "Tôi muốn" -> "toi muon").
VIETNAMESE_PROBE = "Tôi muốn cho các bạn biết về sự to lớn của những nỗ lực khoa học."


def check_vietnamese_round_trip(tokenizer):
    """Raise ValueError if `tokenizer` cannot reproduce Vietnamese text (lowercasing, accent stripping, [UNK])."""
    ids = tokenizer(VIETNAMESE_PROBE, add_special_tokens=False)["input_ids"]
    decoded = tokenizer.decode(ids, clean_up_tokenization_spaces=True).replace(" .", ".")
    if decoded != VIETNAMESE_PROBE or tokenizer.unk_token_id in ids:
        raise ValueError(
            f"tokenizer cannot represent Vietnamese: {VIETNAMESE_PROBE!r} -> {decoded!r}. "
            "Use a multilingual cased checkpoint (bert-base-multilingual-cased); English-only BERT fits "
            "only English-English tasks."
        )


def amr_added_tokens(mode, vocab_file=None):
    """Return the list of AMR tokens to append to the tokenizer for an `amr_vocab` mode."""
    if mode not in AMR_VOCAB_MODES:
        raise ValueError(f"amr_vocab must be one of {AMR_VOCAB_MODES}, got {mode!r}")
    if mode == "none":
        return []
    if mode == "relations":
        vocab_file = vocab_file or paths.AMR_VOCAB_FILE
        if not os.path.exists(vocab_file):
            raise FileNotFoundError(
                f"{vocab_file} not found; build it with `python scripts/build_amr_vocab.py`"
            )
        with open(vocab_file, encoding="utf-8") as f:
            vocab = json.load(f)
        tokens = list(vocab["relations"]) + list(vocab["frames"])
        validate_added_tokens(tokens)
        return tokens
    legacy = paths.LEGACY_AMR_TOKENS_FULL if mode == "legacy_full" else paths.LEGACY_AMR_TOKENS_SIMPLE
    with open(legacy, encoding="utf-8") as f:
        return json.load(f)


def load_relation_vocab(vocab_file=None):
    """Map relation label -> id for graph edge types. Id 0 is reserved for unknown relations."""
    vocab_file = vocab_file or paths.AMR_VOCAB_FILE
    with open(vocab_file, encoding="utf-8") as f:
        relations = json.load(f)["relations"]
    return {r: i + 1 for i, r in enumerate(relations)}


class myTokenizer():
    """
    Unified tokenizer wrapper.

    Mode 1 — HuggingFace tokenizer (args.vocab == 'bert'):
        - If args.tokenizer_dir holds a saved tokenizer (a training run folder), it is loaded as-is.
          Decoding uses this path so the vocabulary is exactly the one the checkpoint was trained with.
        - Otherwise the base tokenizer `args.config_name` is extended with the AMR tokens of
          `args.amr_vocab` plus the two direction tokens, and saved to args.checkpoint_path.
    Mode 2 — custom vocab file (args.vocab == <path>): one token per line, special tokens
        [START]=0, [END]=1, [UNK]=2, [PAD]=3, direction tokens 4-5.

    Attributes: tokenizer, sep_token_id, pad_token_id, vocab_size (also written to args.vocab_size).
    """

    def __init__(self, args):
        if args.vocab == 'bert':
            tokenizer_dir = getattr(args, 'tokenizer_dir', '') or ''
            if tokenizer_dir and os.path.exists(os.path.join(tokenizer_dir, 'tokenizer_config.json')):
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
                print(f'### Loaded saved tokenizer from {tokenizer_dir} ({len(tokenizer)} entries)')
            else:
                tokenizer = AutoTokenizer.from_pretrained(args.config_name)
                mode = getattr(args, 'amr_vocab', 'relations')
                added = amr_added_tokens(mode)
                if mode == 'legacy_simple':
                    print('### WARNING: amr_vocab=legacy_simple corrupts tokenization (bug B1); '
                          'use only to reproduce runs before 2026-10-03')
                tokenizer.add_tokens(added)
                tokenizer.add_special_tokens(
                    {'additional_special_tokens': [TEXT_TO_AMR_TOKEN, AMR_TO_TEXT_TOKEN]}
                )
                print(f'### amr_vocab={mode}: added {len(added)} AMR tokens ({len(tokenizer)} entries)')
                if getattr(args, 'checkpoint_path', ''):
                    tokenizer.save_pretrained(args.checkpoint_path)
            check_vietnamese_round_trip(tokenizer)
            self.tokenizer = tokenizer
            self.sep_token_id = tokenizer.sep_token_id
            self.pad_token_id = tokenizer.pad_token_id
        else:
            print('#' * 30, 'load vocab from', args.vocab)
            vocab_dict = {
                '[START]': 0, '[END]': 1, '[UNK]': 2, '[PAD]': 3,
                TEXT_TO_AMR_TOKEN: 4, AMR_TO_TEXT_TOKEN: 5
            }
            with open(args.vocab, 'r', encoding='utf-8') as f:
                for row in f:
                    vocab_dict[row.strip().split(' ')[0]] = len(vocab_dict)
            self.tokenizer = vocab_dict
            self.rev_tokenizer = {v: k for k, v in vocab_dict.items()}
            self.sep_token_id = vocab_dict['[END]']
            self.pad_token_id = vocab_dict['[PAD]']
            if int(os.environ.get('LOCAL_RANK', 0)) == 0 and getattr(args, 'checkpoint_path', ''):
                with open(f'{args.checkpoint_path}/vocab.json', 'w') as f:
                    json.dump(vocab_dict, f)

        self.vocab_size = len(self.tokenizer)
        args.vocab_size = self.vocab_size  # the model's embedding table is sized from this

    def encode_token(self, sentences):
        """Encode raw strings to id lists, each wrapped as [CLS] ... [SEP] (or [START] ... [END])."""
        if isinstance(self.tokenizer, dict):
            return [
                [0] + [self.tokenizer.get(x, self.tokenizer['[UNK]']) for x in seq.split()] + [1]
                for seq in sentences
            ]
        if isinstance(self.tokenizer, PreTrainedTokenizerFast):
            return self.tokenizer(sentences, add_special_tokens=True)['input_ids']
        raise TypeError(f"unsupported tokenizer type {type(self.tokenizer)}")

    def decode_token(self, seq):
        """Decode a 1-D id tensor to a string after stripping trailing [PAD] ids."""
        seq = seq.squeeze(-1).tolist()
        while len(seq) > 0 and seq[-1] == self.pad_token_id:
            seq.pop()
        if isinstance(self.tokenizer, dict):
            return " ".join([self.rev_tokenizer[x] for x in seq]).replace('__ ', '').replace('@@ ', '')
        if isinstance(self.tokenizer, PreTrainedTokenizerFast):
            return self.tokenizer.decode(seq)
        raise TypeError(f"unsupported tokenizer type {type(self.tokenizer)}")


def load_model_emb(args, tokenizer):
    """
    Build the embedding table used by TextDataset to turn ids into vectors.

    Note: the training loss embeds `input_ids` with the model's own `word_embedding`
    (GaussianDiffusion.training_losses_seq2seq); the vectors produced here are carried through the
    data loader but not used by the loss.

    Rank 0 creates (or reloads) `random_emb.torch` in args.checkpoint_path and writes a `.done`
    sentinel; other ranks wait for the sentinel.
    """
    model = torch.nn.Embedding(tokenizer.vocab_size, args.hidden_dim)
    path_save = '{}/random_emb.torch'.format(args.checkpoint_path)
    path_save_ind = path_save + ".done"

    if int(os.environ.get('LOCAL_RANK', 0)) == 0:
        if os.path.exists(path_save):
            print('reload the random embeddings', model)
            model.load_state_dict(torch.load(path_save))
        else:
            if getattr(args, 'use_plm_init', 'no') == 'bert':
                from transformers import BertModel
                temp_bert = BertModel.from_pretrained(args.config_name)
                pretrained = temp_bert.embeddings.word_embeddings.weight
                if pretrained.shape[1] != args.hidden_dim:
                    raise ValueError(
                        f"use_plm_init=bert needs hidden_dim={pretrained.shape[1]}, got {args.hidden_dim}"
                    )
                n_pre = pretrained.shape[0]
                with torch.no_grad():
                    torch.nn.init.normal_(model.weight)
                    rows = min(n_pre, tokenizer.vocab_size)
                    model.weight[:rows].copy_(pretrained[:rows])  # extra AMR rows stay random
                del temp_bert
            else:
                print('initializing the random embeddings', model)
                torch.nn.init.normal_(model.weight)
            torch.save(model.state_dict(), path_save)
            os.sync()
            with open(path_save_ind, "x") as _:
                pass
    else:
        while not os.path.exists(path_save_ind):
            time.sleep(1)
        print('reload the random embeddings', model)
        model.load_state_dict(torch.load(path_save))

    return model, tokenizer


def load_tokenizer(args):
    return myTokenizer(args)


def load_defaults_config():
    """Defaults from diffuseq/config.json; every key is also a CLI flag of train.py / samplers."""
    with open(paths.DIFFUSEQ_CONFIG, 'r') as f:
        return json.load(f)


def graph_edge_types(graph_mode, num_relations):
    """Number of edge types the graph encoder embeds (see text_datasets.build_edges)."""
    if graph_mode == 'levi':
        return 4                       # head->label, label->dep, and their reverses
    if graph_mode == 'edge_attr':
        return 2 * (num_relations + 1)  # (relation id incl. unknown 0) x (forward, reverse)
    raise ValueError(f"graph_mode must be 'levi' or 'edge_attr', got {graph_mode!r}")


def create_model_and_diffusion(
    hidden_t_dim,         # size of the sinusoidal timestep embedding
    hidden_dim,           # word-embedding size = diffusion space size
    vocab_size,
    config_name,          # HuggingFace config of the Transformer backbone
    use_plm_init,         # 'bert' = pretrained encoder weights, 'no' = random init
    dropout,
    diffusion_steps,      # T
    noise_schedule,       # 'sqrt' in DiffuSeq
    learn_sigma,
    timestep_respacing,   # '' = all T steps
    predict_xstart,       # DiffuSeq predicts x_0
    rescale_timesteps,
    sigma_small,
    rescale_learned_sigmas,
    use_kl,
    notes,
    learned_mean_embed=False,  # learned soft absorbing state ([MASK]-like vector) of DiffuSeq-v2
    rejection_rate=0.0,
    denoise=False,        # DiffuSeq-v2 discrete noise: replace target positions by mean_embed
    denoise_rate=0.2,     # max replacement probability (scaled by the noise level)
    device="",
    graph_encoder="none",  # 'none' | 'gatv2'
    graph_mode="edge_attr",  # 'edge_attr' | 'levi'
    graph_layers=2,
    graph_heads=4,
    **kwargs,             # other config keys are accepted and ignored
):
    """Construct the denoising Transformer and the diffusion process (see their docstrings)."""
    num_edge_types = 0
    if graph_encoder != 'none':
        num_relations = len(load_relation_vocab()) if graph_mode == 'edge_attr' else 0
        num_edge_types = graph_edge_types(graph_mode, num_relations)

    model = TransformerNetModel(
        input_dims=hidden_dim,
        output_dims=(hidden_dim if not learn_sigma else hidden_dim * 2),
        hidden_t_dim=hidden_t_dim,
        dropout=dropout,
        config_name=config_name,
        vocab_size=vocab_size,
        init_pretrained=use_plm_init,
        logits_mode=kwargs.get('logits_mode', 1),
        learned_mean_embed=learned_mean_embed,
        graph_encoder=graph_encoder,
        graph_layers=graph_layers,
        graph_heads=graph_heads,
        num_edge_types=num_edge_types,
    )

    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)
    if not timestep_respacing:
        timestep_respacing = [diffusion_steps]

    diffusion = SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        rescale_timesteps=rescale_timesteps,
        predict_xstart=predict_xstart,
        learn_sigmas=learn_sigma,
        sigma_small=sigma_small,
        use_kl=use_kl,
        rescale_learned_sigmas=rescale_learned_sigmas,
        rejection_rate=rejection_rate,
        denoise=denoise,
        denoise_rate=denoise_rate,
        mask_docamr_rel=kwargs.get('mask_docamr_rel', False),
        device=device,
        max_T=diffusion_steps,
    )
    return model, diffusion


def add_dict_to_argparser(parser, default_dict):
    """Register each key as --key; bool defaults parse with str2bool, None defaults as str."""
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """Parse 'true/false/yes/no/1/0' (any case) into a bool for argparse."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("boolean value expected")
