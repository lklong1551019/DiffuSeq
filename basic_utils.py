"""
basic_utils.py — Utility functions and classes shared by training and inference scripts.

Responsibilities:
  1. myTokenizer: wraps either HuggingFace BERT tokenizer or a custom BPE vocab dict.
  2. load_model_emb:  initializes or loads random word embeddings (used as diffusion targets).
  3. load_tokenizer:  convenience wrapper around myTokenizer.
  4. load_defaults_config: loads the base hyperparameter config from diffuseq/config.json.
  5. create_model_and_diffusion: instantiates TransformerNetModel + SpacedDiffusion.
  6. add_dict_to_argparser: registers a config dict as CLI flags.
  7. args_to_dict: converts argparse.Namespace back to a keyed dict.
  8. str2bool: parses boolean strings from the command line.
"""

import argparse
import torch
import json, os
import time

# gaussian_diffusion: contains the core GaussianDiffusion math (forward/reverse process)
from diffuseq import gaussian_diffusion as gd
# SpacedDiffusion: supports timestep subsampling (e.g., using only T/2 evenly spaced steps)
# space_timesteps: converts a target step count into a set of timestep indices
from diffuseq.gaussian_diffusion import SpacedDiffusion, space_timesteps
# TransformerNetModel: the BERT-based denoising model (backbone of DiffuSeq)
from diffuseq.transformer_model import TransformerNetModel
from transformers import AutoTokenizer, PreTrainedTokenizerFast
import re

# Bidirectional training special tokens
AMR_TO_TEXT_LABEL = "AMR_TO_TEXT"
TEXT_TO_AMR_LABEL = "TEXT_TO_AMR"
AMR_TO_TEXT_TOKEN = f"[{AMR_TO_TEXT_LABEL}]"
TEXT_TO_AMR_TOKEN = f"[{TEXT_TO_AMR_LABEL}]"

class myTokenizer():
    """
    A unified tokenizer wrapper supporting two vocabulary modes:

    Mode 1 — BERT tokenizer (vocab == 'bert'):
        Uses HuggingFace AutoTokenizer backed by BERT's WordPiece vocabulary.
        Special tokens ([CLS], [SEP], [PAD], [MASK], [UNK]) are handled automatically.
        The tokenizer is saved to checkpoint_path for later use during inference.

    Mode 2 — Custom BPE vocab file (vocab == <path>):
        Reads a vocab file where each line contains a token.
        Builds a dict: {token: id}. Special tokens [START], [END], [UNK], [PAD]
        are prepended with ids 0–3. The vocab is saved as 'vocab.json' by rank 0.

    After construction:
        self.tokenizer:      the underlying tokenizer (HuggingFace or dict)
        self.sep_token_id:   ID of the separator/end token
        self.pad_token_id:   ID of the padding token
        self.vocab_size:     total vocabulary size (also written to args.vocab_size)
    """

    ################################################
    ### You can customize your own tokenizer here. ###
    ################################################

    def __init__(self, args):
        if args.vocab == 'bert':
            # Load the BERT tokenizer from HuggingFace Hub (or cached locally).
            # config_name is typically 'bert-base-uncased' or 'bert-base-multilingual-cased'.
                    
            tokenizer = AutoTokenizer.from_pretrained(args.config_name)
            
            # --- Load Custom AMR & DocAMR Tokens ---
            # Using the pre-generated comprehensive list of tokens
            if getattr(args, 'use_simple_amr', False):
                rel_file = "datasets/docAMR/output_doc_amr/doc_amrs_token_simple.json"
            else:
                rel_file = "datasets/docAMR/output_doc_amr/doc_amrs_token.json"
            if os.path.exists(rel_file):
                with open(rel_file, 'r', encoding='utf-8') as f:
                    all_to_add = json.load(f)
                print(f"### Loaded {len(all_to_add)} custom AMR tokens from {rel_file}")
                tokenizer.add_tokens(all_to_add)
            else:
                raise Exception(f"Token file {rel_file} not found!")

            special_amr_direction_tokens = [TEXT_TO_AMR_TOKEN, AMR_TO_TEXT_TOKEN]
            tokenizer.add_special_tokens({'additional_special_tokens': special_amr_direction_tokens})                
                
            # ------------------------------------------

            self.tokenizer = tokenizer
            self.sep_token_id = tokenizer.sep_token_id  # [SEP]
            self.pad_token_id = tokenizer.pad_token_id  # [PAD]
            # Persist the tokenizer (including added tokens) to the checkpoint dir
            tokenizer.save_pretrained(args.checkpoint_path)
        else:
            # Custom vocab path: each line is "<token> [optional_count]"; we use only the token.
            print('#' * 30, 'load vocab from', args.vocab)
            # Reserve ids 0–5 for special tokens
            vocab_dict = {
                '[START]': 0, '[END]': 1, '[UNK]': 2, '[PAD]': 3, 
                TEXT_TO_AMR_TOKEN: 4, AMR_TO_TEXT_TOKEN: 5
            }
            with open(args.vocab, 'r', encoding='utf-8') as f:
                for row in f:
                    vocab_dict[row.strip().split(' ')[0]] = len(vocab_dict)
            self.tokenizer = vocab_dict
            # Reverse mapping for decoding: id → token
            self.rev_tokenizer = {v: k for k, v in vocab_dict.items()}
            self.sep_token_id = vocab_dict['[END]']
            self.pad_token_id = vocab_dict['[PAD]']
            # Only rank 0 writes the vocab file to avoid race conditions
            if int(os.environ['LOCAL_RANK']) == 0:
                path_save_vocab = f'{args.checkpoint_path}/vocab.json'
                with open(path_save_vocab, 'w') as f:
                    json.dump(vocab_dict, f)

        self.vocab_size = len(self.tokenizer)
        args.vocab_size = self.vocab_size  # propagate vocab size to args for model construction

    def encode_token(self, sentences):
        """
        Tokenize a list of raw text strings into lists of integer token IDs.

        Mode 1 (dict vocab): manually split on whitespace, map each token to its ID,
                             wrap with [START]=0 and [END]=1.
        Mode 2 (HuggingFace): delegate to the HuggingFace tokenizer which handles
                              subword splitting and adds [CLS] / [SEP] automatically.

        Args:
            sentences (list[str]): raw input strings.

        Returns:
            list[list[int]]: token ID sequences (variable length).
        """
        if isinstance(self.tokenizer, dict):
            # Manual BPE vocab: split on whitespace, fall back to [UNK] for unknown tokens
            input_ids = [
                [0] + [self.tokenizer.get(x, self.tokenizer['[UNK]']) for x in seq.split()] + [1]
                for seq in sentences
            ]
        elif isinstance(self.tokenizer, PreTrainedTokenizerFast):
            # HuggingFace fast tokenizer: handles subword tokenization + special tokens
            input_ids = self.tokenizer(sentences, add_special_tokens=True)['input_ids']
        else:
            assert False, "invalid type of vocab_dict"
        return input_ids

    def decode_token(self, seq):
        """
        Convert a tensor (or list) of token IDs back to a human-readable string.

        Trailing PAD tokens are stripped before decoding.

        Mode 1 (dict vocab): map IDs → tokens, join with spaces, strip BPE artifacts.
        Mode 2 (HuggingFace): use the tokenizer's .decode() method.

        Args:
            seq (Tensor | list): 1D sequence of token IDs (possibly with leading batch dim).

        Returns:
            str: decoded string.
        """
        if isinstance(self.tokenizer, dict):
            seq = seq.squeeze(-1).tolist()
            # Strip trailing padding
            while len(seq) > 0 and seq[-1] == self.pad_token_id:
                seq.pop()
            # Clean up BPE continuation markers ('__ ' and '@@ ')
            tokens = " ".join([self.rev_tokenizer[x] for x in seq]).replace('__ ', '').replace('@@ ', '')
        elif isinstance(self.tokenizer, PreTrainedTokenizerFast):
            seq = seq.squeeze(-1).tolist()
            while len(seq) > 0 and seq[-1] == self.pad_token_id:
                seq.pop()
            tokens = self.tokenizer.decode(seq)
        else:
            assert False, "invalid type of vocab_dict"
        return tokens


def load_model_emb(args, tokenizer):
    """
    Initialize or load the word embedding matrix used as diffusion targets.

    The embedding E ∈ R^{vocab_size × hidden_dim} maps token IDs to continuous vectors.
    In DiffuSeq, the forward diffusion noises these embedding vectors, and the model
    learns to denoise back to the original embedding space.

    Initialization strategy (rank-0 only to avoid race conditions):
      - If a saved embedding exists at checkpoint_path/random_emb.torch → reload it.
      - Otherwise → initialize with N(0,1) and save to disk.
    Non-rank-0 processes wait for a sentinel file (random_emb.torch.done) before loading.

    Args:
        args: parsed arguments with checkpoint_path and hidden_dim.
        tokenizer (myTokenizer): tokenizer providing vocab_size.

    Returns:
        (nn.Embedding, myTokenizer): the embedding module and (unchanged) tokenizer.
    """
    model = torch.nn.Embedding(tokenizer.vocab_size, args.hidden_dim)
    path_save = '{}/random_emb.torch'.format(args.checkpoint_path)
    path_save_ind = path_save + ".done"   # sentinel file: signals embedding is ready

    if int(os.environ['LOCAL_RANK']) == 0:
        if os.path.exists(path_save):
            # Reload a previously saved embedding (ensures consistent init across runs)
            print('reload the random embeddings', model)
            model.load_state_dict(torch.load(path_save))
        else:
            if getattr(args, 'use_plm_init', 'no') == 'bert':
                # Seed the standalone embedding from mBERT's pretrained weights so that
                # the diffusion targets and the model's word_embedding start from the same space.
                # Extra rows (for AMR tokens beyond mBERT's 119,547 base vocab) are randomly init.
                print('### Seeding random_emb from pretrained mBERT embeddings (use_plm_init=bert)')
                from transformers import BertModel
                temp_bert = BertModel.from_pretrained(args.config_name)
                pretrained_weight = temp_bert.embeddings.word_embeddings.weight  # [119547, 768]
                mbert_vocab_size = pretrained_weight.shape[0]

                with torch.no_grad():
                    if tokenizer.vocab_size <= mbert_vocab_size:
                        # Vocab fits inside mBERT: copy first vocab_size rows
                        model.weight.copy_(pretrained_weight[:tokenizer.vocab_size, :args.hidden_dim])
                    else:
                        # Copy all pretrained rows then randomly init extra AMR token rows
                        model.weight[:mbert_vocab_size].copy_(pretrained_weight[:, :args.hidden_dim])
                        torch.nn.init.normal_(model.weight[mbert_vocab_size:])
                        print(f'    Pretrained rows: {mbert_vocab_size}, '
                              f'randomly init rows: {tokenizer.vocab_size - mbert_vocab_size}')
                del temp_bert
            else:
                # Fresh random init: N(0,1) is a common embedding initialization
                print('initializing the random embeddings', model)
                torch.nn.init.normal_(model.weight)
            torch.save(model.state_dict(), path_save)
            os.sync()   # flush OS file buffers to disk before writing the sentinel
            with open(path_save_ind, "x") as _:
                pass    # create the sentinel file
    else:
        # Other ranks spin-wait until rank 0 has finished writing
        while not os.path.exists(path_save_ind):
            time.sleep(1)
        print('reload the random embeddings', model)
        model.load_state_dict(torch.load(path_save))

    return model, tokenizer


def load_tokenizer(args):
    """
    Convenience function: constructs and returns a myTokenizer instance.

    Args:
        args: parsed arguments (needs args.vocab and args.config_name).

    Returns:
        myTokenizer
    """
    tokenizer = myTokenizer(args)
    return tokenizer


def load_defaults_config():
    """
    Load the default hyperparameter configuration from diffuseq/config.json.

    This config defines the full set of model and training hyperparameters
    (hidden_dim, diff_steps, noise_schedule, lr, etc.). CLI arguments override
    these defaults.

    Returns:
        dict: key → default value.
    """
    with open('diffuseq/config.json', 'r') as f:
        return json.load(f)


def create_model_and_diffusion(
    hidden_t_dim,         # dimensionality of the timestep embedding input
    hidden_dim,           # word embedding dimension (= model input/output dim)
    vocab_size,           # vocabulary size
    config_name,          # HuggingFace model config name (e.g., 'bert-base-uncased')
    use_plm_init,         # 'bert' to init from BERT weights, 'no' for random init
    dropout,              # dropout probability for the Transformer
    diffusion_steps,      # total number of forward diffusion timesteps T
    noise_schedule,       # noise schedule type: 'sqrt', 'linear', 'cosine', etc.
    learn_sigma,          # if True, model also predicts log-variance (2x output dim)
    timestep_respacing,   # subset of timesteps to use at inference ('' = use all T)
    predict_xstart,       # if True, model predicts x_0 directly; else predicts noise ε
    rescale_timesteps,    # if True, rescale t to [0, 1000] regardless of T
    sigma_small,          # if True, use a smaller sigma for the reverse process
    rescale_learned_sigmas,  # if True, apply a rescaling factor to learned sigmas
    use_kl,               # if True, optimize the ELBO KL term instead of simple MSE
    notes,                # experiment notes (unused in model construction)
    learned_mean_embed=False,  # if True, learn a global mean embedding for conditioning
    rejection_rate=0.0,   # probability of rejecting a diffusion sample (curriculum)
    denoise=False,        # if True, enable denoising mode (partial noising of input)
    denoise_rate=0.2,     # fraction of tokens to denoise in denoise mode
    device="",            # target device (unused here; model moved later)
    enable_gcn=False,     # if True, enable GCN before Transformer
    use_relational_gcn=False, # if True, use RGCN instead of standard GCN
    num_relations=400,    # total number of relation types for R-GCN
    **kwargs,             # absorb any extra config keys silently
):
    """
    Construct the TransformerNetModel (denoising backbone) and SpacedDiffusion
    (noise schedule + loss computation) from hyperparameters.

    Model Architecture:
        TransformerNetModel is a BERT-based encoder that:
          - Embeds noisy continuous token vectors x_t ∈ R^{seq_len × hidden_dim}
          - Adds a sinusoidal + MLP timestep embedding e_t ∈ R^{hidden_dim}
          - Adds positional embeddings
          - Processes through BERT's Transformer encoder layers
          - Projects back to hidden_dim (the denoised x_0 prediction)

    Diffusion Process:
        SpacedDiffusion wraps GaussianDiffusion and supports skipping timesteps
        at inference time. It defines:
          - Forward process: q(x_t | x_0) = N(√ᾱ_t · x_0, (1-ᾱ_t) · I)
          - Loss: MSE between model's x_0 prediction and true x_0 (optionally + VLB term)

    Returns:
        (TransformerNetModel, SpacedDiffusion)
    """
    num_relations = 400
    if enable_gcn:
        # Load relation tokens to get exact count
        use_simple_amr = kwargs.get('use_simple_amr', False)
        if use_simple_amr:
            rel_file = "datasets/docAMR/doc_amrs_token_simple.json"
        else:
            rel_file = "datasets/docAMR/doc_amrs_token.json"
        
        if os.path.exists(rel_file):
            with open(rel_file, 'r', encoding='utf-8') as f:
                tokens = json.load(f)
                # Only count tokens starting with ':' (AMR relations)
                relations = [t for t in tokens if t.startswith(':')]
                num_relations = len(relations)
            print(f"### Setting num_relations to {num_relations} from {rel_file} (filtered to relations only)")
        else:
            print(f"### Warning: {rel_file} not found, using default num_relations=400")

    model = TransformerNetModel(
        input_dims=hidden_dim,
        # If learn_sigma, model outputs 2*hidden_dim: first half = x_0 pred, second = log-var
        output_dims=(hidden_dim if not learn_sigma else hidden_dim * 2),
        hidden_t_dim=hidden_t_dim,
        dropout=dropout,
        config_name=config_name,
        vocab_size=vocab_size,
        init_pretrained=use_plm_init,
        logits_mode=kwargs.get('logits_mode', 1),
        learned_mean_embed=learned_mean_embed,
        enable_gcn=enable_gcn,
        use_relational_gcn=use_relational_gcn,
        num_relations=num_relations,
    )

    # get_named_beta_schedule: computes the β_t noise schedule curve.
    # 'sqrt' schedule: β_t ∝ √t (DiffuSeq's default, better for text than cosine/linear).
    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)

    # If no respacing is specified, use all T timesteps
    if not timestep_respacing:
        timestep_respacing = [diffusion_steps]

    # SpacedDiffusion: wraps GaussianDiffusion and subsets the timestep schedule.
    # At T=5 (as in train.sh), this uses only 5 timesteps for fast prototyping.
    diffusion = SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        rescale_timesteps=rescale_timesteps,
        predict_xstart=predict_xstart,   # DiffuSeq predicts x_0, not noise ε
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
    """
    Register each key in default_dict as a CLI argument on the given parser.

    Type inference:
      - None values → str (user must provide the correct type)
      - bool values → str2bool (handles 'true'/'false'/'yes'/'no' strings)
      - all other types → inferred from the value's Python type

    Args:
        parser (argparse.ArgumentParser): the parser to add arguments to.
        default_dict (dict): mapping of argument name → default value.
    """
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool   # argparse can't natively parse bool from string
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    """
    Extract a subset of fields from an argparse.Namespace as a plain dict.

    Used to pass only the relevant hyperparameters to create_model_and_diffusion,
    filtering out training-only flags (lr, batch_size, etc.).

    Args:
        args (argparse.Namespace): parsed arguments.
        keys (iterable): the keys to extract.

    Returns:
        dict: {key: getattr(args, key)} for each key in keys.
    """
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    Parse a boolean value from a command-line string argument.

    argparse doesn't natively handle bool; passing --flag True would be parsed
    as the string "True" not the Python bool True. This converter handles common
    string representations.

    Reference: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    Args:
        v: the value to convert (may already be a bool if coming from defaults).

    Returns:
        bool

    Raises:
        argparse.ArgumentTypeError: if the string doesn't match any known pattern.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
