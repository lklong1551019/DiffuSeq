"""
rounding.py — Nearest-neighbour rounding utilities for DiffuSeq inference.

After the reverse diffusion process, the model outputs a continuous embedding vector
x̂_0 ∈ R^{seq_len × hidden_dim} for each token position. To decode text, these
continuous vectors must be "rounded" back to the nearest discrete token in the vocabulary.

This module provides three rounding strategies:
  1. get_knn:           brute-force k-NN lookup (cosine or L2).
  2. get_efficient_knn: efficient L2 distance via the ||a-b||² = ||a||²+||b||²-2a·b trick.
  3. denoised_fn_round: used during inference inside the diffusion loop to clamp
                        intermediate predicted x_0 to the nearest token embedding,
                        preventing drift through the continuous embedding space.

Also contains:
  - rounding_func:   converts a list of embeddings to decoded strings.
  - compute_logp:    computes per-token log-probability under a Gaussian token model.
  - get_weights:     extracts and freezes the embedding weight matrix from a model.
"""

import torch
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, default_data_collator, GPT2TokenizerFast
import sys, yaml, os
import json

import numpy as np


def get_knn(model_emb, text_emb, dist='cos'):
    """
    Brute-force k-nearest-neighbour search in embedding space.

    For each query vector in text_emb, finds the k=6 closest vocabulary embeddings
    in model_emb under cosine similarity or L2 distance.

    Args:
        model_emb (Tensor): [vocab_size, dim] embedding matrix (vocabulary embeddings).
        text_emb (Tensor):  [N, dim] query embedding vectors (from the denoised output).
        dist (str):         'cos' for cosine similarity, 'l2' for negative L2 distance.

    Returns:
        (Tensor, Tensor): top-k values and indices, shape [k, N].
    """
    if dist == 'cos':
        # Cosine similarity: model_emb @ text_emb^T → [vocab_size, N]
        adjacency = model_emb @ text_emb.transpose(1, 0).to(model_emb.device)
    elif dist == 'l2':
        # Negative L2 distance: higher = closer
        adjacency = model_emb.unsqueeze(1).expand(-1, text_emb.size(0), -1) - \
                    text_emb.unsqueeze(0).expand(model_emb.size(0), -1, -1)
        adjacency = -torch.norm(adjacency, dim=-1)   # [vocab_size, N]
    topk_out = torch.topk(adjacency, k=6, dim=0)    # top-6 per query token
    return topk_out.values, topk_out.indices


def get_efficient_knn(model_emb, text_emb):
    """
    Memory-efficient L2 nearest-neighbour search using the squared distance identity.

    Computes ||model_emb_i - text_emb_j||² = ||model_emb_i||² + ||text_emb_j||² - 2·(model_emb_i · text_emb_j)
    without materializing the full [vocab × seq_len × dim] difference tensor.

    Args:
        model_emb (Tensor): [vocab_size, dim] vocabulary embedding matrix.
        text_emb (Tensor):  [seq_len, dim] denoised embedding vectors.
                            (can also be [B*seq_len, dim] — any 2D tensor works)

    Returns:
        (Tensor, Tensor):
          - values:  [1, seq_len] negative distances to the nearest token.
          - indices: [1, seq_len] indices of the nearest vocabulary token per position.
    """
    emb_norm = (model_emb ** 2).sum(-1).view(-1, 1)             # [vocab, 1]  ||e_v||²
    text_emb_t = torch.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1)  # [dim, seq_len]
    arr_norm = (text_emb ** 2).sum(-1).view(-1, 1)              # [seq_len, 1] ||x_i||²
    # Squared L2 distances: [vocab, seq_len]
    dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * torch.mm(model_emb, text_emb_t)
    dist = torch.clamp(dist, 0.0, np.inf)   # numerical safety: clamp away small negatives
    # Return top-1 per query position (= nearest vocabulary token)
    topk_out = torch.topk(-dist, k=1, dim=0)   # negate: smaller distance = larger score
    return topk_out.values, topk_out.indices


def rounding_func(text_emb_lst, model, tokenizer, emb_scale_factor=1.0):
    """
    Decode a list of continuous embedding arrays to text strings via L2 nearest-neighbour lookup.

    For each embedding array in text_emb_lst:
      1. Find the nearest vocabulary token for each position using get_knn (L2 mode).
      2. Decode the top-1 token ID sequence to a string using the tokenizer.

    Args:
        text_emb_lst (list[Tensor|np.array]): list of [seq_len, dim] embedding arrays.
        model (nn.Embedding):                 the token embedding matrix.
        tokenizer (myTokenizer):              tokenizer for converting IDs → text.
        emb_scale_factor (float):             optional scale factor for embeddings (unused here).

    Returns:
        list[str]: decoded text strings, one per embedding array.
    """
    decoded_out_lst = []

    model_emb = model.weight     # [vocab_size, dim] embedding weight matrix
    down_proj_emb2 = None

    dist = 'l2'   # use L2 distance for rounding

    for text_emb in text_emb_lst:
        text_emb = torch.tensor(text_emb)
        # Flatten [B, seq_len, dim] → [B*seq_len, dim] if needed
        if len(text_emb.shape) > 2:
            text_emb = text_emb.view(-1, text_emb.size(-1))

        # Find nearest neighbours in the vocabulary embedding space
        val, indices = get_knn(
            (down_proj_emb2 if dist == 'cos' else model_emb),
            text_emb.to(model_emb.device),
            dist=dist
        )

        # Decode the top-1 (nearest) token per position
        decoded_out_lst.append(tokenizer.decode_token(indices[0]))

    return decoded_out_lst


def compute_logp(args, model, x, input_ids):
    """
    Compute per-token negative log-probability under a Gaussian token distribution.

    Models each token as a Gaussian centered at its embedding with fixed σ=0.1.
    The log-probability of generating token v at position i is:

        logp(v | x_i) ∝ -||x_i - emb_v||² / (2σ²)

    This is then used as per-token cross-entropy loss against the ground-truth input_ids.

    Args:
        args:           argparse.Namespace (used for model_arch check).
        model (nn.Embedding): the embedding module (model.weight = [vocab, dim]).
        x (Tensor):     [B, seq_len, dim] predicted embedding sequences.
        input_ids (Tensor): [B, seq_len] ground-truth token IDs.

    Returns:
        Tensor: [B, seq_len] per-token cross-entropy loss values.
    """
    word_emb = model.weight     # [vocab_size, dim]
    sigma = 0.1                 # fixed Gaussian standard deviation

    # (Unused in the active code path but kept for 1D-UNet compatibility)
    if args.model_arch == '1d-unet':
        x = x.permute(0, 2, 1)

    bsz, seqlen, dim = x.shape

    # Compute squared distances between each predicted embedding and each vocabulary embedding.
    # Efficient formulation avoids materializing the full [vocab, B*L, dim] tensor.
    x_flat = x.reshape(-1, x.size(-1)).unsqueeze(0)     # [1, B*L, dim]
    word_emb_flat = word_emb.unsqueeze(1)               # [vocab, 1, dim]
    diff = (x_flat - word_emb_flat) ** 2                # [vocab, B*L, dim]

    # Gaussian log-probability: logp ∝ -||diff||² / (2σ²)
    logp_expanded = -diff.sum(dim=-1) / (2 * sigma ** 2)   # [vocab, B*L]
    logp_expanded = logp_expanded.permute((1, 0))           # [B*L, vocab]

    # Cross-entropy between the Gaussian logits and the ground-truth token IDs
    ce = torch.nn.CrossEntropyLoss(reduction='none')
    loss = ce(logp_expanded, input_ids.view(-1)).view(bsz, seqlen)

    return loss


def get_weights(model, args):
    """
    Extract the (frozen) embedding weight matrix from a model for use in rounding.

    Handles two model types:
      - GPT-style models with a transformer.wte embedding + down_proj layer.
      - Plain nn.Embedding models (DiffuSeq's case) — passed through unchanged.

    After this call, model.weight.requires_grad = False so it won't be updated.

    Args:
        model: an nn.Embedding or a GPT-style model with .transformer.wte.
        args:  argparse.Namespace (used for emb_scale_factor in GPT case).

    Returns:
        nn.Embedding with frozen weights.
    """
    if hasattr(model, 'transformer'):
        # GPT-style: extract embedding, project through down_proj, and rescale
        input_embs = model.transformer.wte
        down_proj = model.down_proj
        model_emb = down_proj(input_embs.weight)
        print(model_emb.shape)
        model = torch.nn.Embedding(model_emb.size(0), model_emb.size(1))
        print(args.emb_scale_factor)
        model.weight.data = model_emb * args.emb_scale_factor
    elif hasattr(model, 'weight'):
        # Plain nn.Embedding: no transformation needed
        pass
    else:
        assert NotImplementedError

    # Freeze the embedding weights so they don't accumulate gradients during inference
    model.weight.requires_grad = False
    return model


def denoised_fn_round(args, model, text_emb, t):
    """
    Clamping function applied to the predicted x_0 during reverse diffusion.

    At each denoising step, the model predicts x̂_0 (the clean embedding). 
    This function projects x̂_0 to the nearest vocabulary embedding and returns
    that embedding. This anchors the diffusion trajectory to the valid token
    embedding manifold, preventing the continuous prediction from drifting into
    regions of embedding space that don't correspond to any real token.

    Called by the diffusion sampler as a post-processing hook on x̂_0 predictions.

    Args:
        args:           argparse.Namespace.
        model (nn.Embedding): the frozen token embedding matrix.
        text_emb (Tensor):    predicted x̂_0, shape [B, seq_len, dim] or [B*L, dim].
        t (int):              current diffusion timestep (unused here; kept for interface compatibility).

    Returns:
        (Tensor, Tensor):
          - new_embeds: [B, seq_len, dim] embeddings of the nearest-neighbour tokens.
          - rounded_tokens: [B, seq_len] integer token IDs of those nearest neighbours.
    """
    model_emb = model.weight    # [vocab_size, dim] frozen vocabulary embeddings
    old_shape = text_emb.shape
    old_device = text_emb.device

    # Flatten to [B*seq_len, dim] for batch matrix operations
    if len(text_emb.shape) > 2:
        text_emb = text_emb.reshape(-1, text_emb.size(-1))

    # Find the nearest vocabulary token for each position using efficient L2 k-NN
    _, indices = get_efficient_knn(model_emb, text_emb.to(model_emb.device))
    # Reshape back to [B, seq_len] and move to original device
    rounded_tokens = indices[0].view(old_shape[:-1]).to(old_device)
    # Look up the embedding of the rounded tokens to get a "clamped" x̂_0
    new_embeds = model(rounded_tokens)

    return new_embeds, rounded_tokens