"""
transformer_model.py — The denoising Transformer backbone for DiffuSeq.

Architecture overview:

  DiffuSeq's core model is a BERT-based Transformer encoder that is conditioned
  on the diffusion timestep t. Given a noisy embedding sequence x_t and timestep t,
  the model predicts the clean embedding sequence x_0.

  Data flow in forward():
    1. Timestep embedding:    t  →  sinusoidal embedding  →  MLP  →  e_t  ∈ R^{hidden_size}
    2. Input projection:      x_t ∈ R^{seq_len × input_dims}
                              → (if input_dims ≠ hidden_size) linear up-project  →  emb_x
    3. Fused input:           emb_inputs = pos_embed(pos) + emb_x + e_t  (broadcast over seq)
    4. Transformer encoder:   emb_inputs  →  BERT encoder (12 layers)  →  hidden states
    5. Output projection:     (if output_dims ≠ hidden_size) linear down-project  →  h
    6. Return h ∈ R^{seq_len × output_dims}  (= predicted x_0 embeddings per token)

  The model also exposes:
    get_embeds(input_ids):  look up token embeddings (used in sampling).
    get_logits(hidden):     project embeddings to vocab logits (for nearest-neighbour decoding).

Weight tying:
    lm_head (vocab projection) shares its weight matrix with word_embedding.
    This ensures the output projection stays in the same embedding space.
"""

from transformers import AutoConfig
from transformers.models.bert.modeling_bert import BertEncoder, BertModel
import torch

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .utils.nn import (
    SiLU,              # Sigmoid Linear Unit activation (smooth alternative to ReLU)
    linear,            # helper that creates nn.Linear layers
    timestep_embedding,  # sinusoidal positional encoding for diffusion timesteps
)


class TransformerNetModel(nn.Module):
    """
    BERT-based denoising Transformer for continuous diffusion over token embeddings.

    The model maps (x_t, t) → x̂_0, where:
        x_t: noisy embedding sequence [batch, seq_len, input_dims]
        t:   diffusion timestep indices [batch]
        x̂_0: predicted clean embedding sequence [batch, seq_len, output_dims]

    Args:
        input_dims (int):         dimensionality of the input/output embeddings
                                  (= hidden_dim in config, typically 128).
        output_dims (int):        output embedding dim (= input_dims, or 2*input_dims if learn_sigma).
        hidden_t_dim (int):       dimensionality of the raw sinusoidal timestep embedding.
        dropout (float):          dropout probability applied after LayerNorm.
        config (BertConfig):      pre-built BERT config (optional; loaded from config_name if None).
        config_name (str):        HuggingFace model name for auto-loading config
                                  (e.g., 'bert-base-uncased').
        vocab_size (int):         vocabulary size for word_embedding and lm_head.
        init_pretrained (str):    'bert' = load pretrained BERT weights;
                                  'no'   = random initialization.
        logits_mode (int):        1 = dot-product (lm_head) logits;
                                  2 = cosine distance negation logits.
        learned_mean_embed (bool):if True, add a learnable mean embedding vector
                                  (used for the denoise_rate curriculum in DiffuSeq).
    """

    def __init__(
        self,
        input_dims,
        output_dims,
        hidden_t_dim,
        dropout=0,
        config=None,
        config_name='bert-base-uncased',
        vocab_size=None,
        init_pretrained='no',
        logits_mode=1,
        learned_mean_embed=False,
    ):
        super().__init__()

        if config is None:
            # Load BERT config from HuggingFace Hub (or cache)
            config = AutoConfig.from_pretrained(config_name)
            config.hidden_dropout_prob = dropout

        self.input_dims = input_dims
        self.hidden_t_dim = hidden_t_dim
        self.output_dims = output_dims
        self.dropout = dropout
        self.logits_mode = logits_mode
        self.hidden_size = config.hidden_size   # BERT's internal hidden size (typically 768)

        # -----------------------------------------------------------------------
        # Token Embedding and Output Projection (weight-tied)
        # -----------------------------------------------------------------------
        # word_embedding: maps token IDs → continuous vectors ∈ R^{input_dims}
        self.word_embedding = nn.Embedding(vocab_size, self.input_dims)

        # lm_head: projects latent vectors back to vocabulary logits ∈ R^{vocab_size}
        # This will produce matrix of shape [input_dims, vocab_size], ex: [128, 30522]
        # After the model produce output of continuous vectors, this will be used to project that vector back to english tokens. 
        # It takes the, for example, 128 dimension ouput and projects it to 30522 dimension,  producing a logit score for every word in the dictionary.        
        self.lm_head = nn.Linear(self.input_dims, vocab_size)
        # Weight tying: lm_head uses the SAME weight matrix as word_embedding.
        # This constrains the embedding and un-embedding to be consistent (saves parameters
        # and empirically improves performance in language models).
        with th.no_grad():
            self.lm_head.weight = self.word_embedding.weight

        # -----------------------------------------------------------------------
        # Timestep Embedding MLP
        # -----------------------------------------------------------------------
        # Converts scalar timestep t into a rich conditioning vector e_t:
        # First, it converts the scalar timestep t into a sinusoidal embedding of dimension hidden_t_dim, for ex, 128.
        #   t → sinusoidal(t, hidden_t_dim)    [dimension: hidden_t_dim]
        # The time_embed NN block takes this sinusoidal embedding and projects 128 features up to 512 features. 
        # Expanding the dimension allows the model to capture more complex patterns in the timestep.
        # SiLU: applies sigmoid linear unit activation function as diffusion models tends to perform better with it, as it is a smooth curve that doesn't kill negatvie gradients.
        #     → Linear(hidden_t_dim, 4*hidden_t_dim) + SiLU
        # Then, projects 512 features into 768 features (BERT's hidden size).
        #     → Linear(4*hidden_t_dim, hidden_size)
        # The output e_t ∈ R^{hidden_size} is broadcast over position to fuse with token embeddings.
        # Model simply adds this time_embed vector to the token embedding before passing them to the Transformer layers.
        time_embed_dim = hidden_t_dim * 4
        self.time_embed = nn.Sequential(
            linear(hidden_t_dim, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, config.hidden_size),
        )

        # -----------------------------------------------------------------------
        # Input Projection (only if input_dims ≠ BERT hidden_size)
        # -----------------------------------------------------------------------
        # DiffuSeq uses hidden_dim=128 while BERT uses hidden_size=768,
        # so an up-projection is needed before the BERT encoder.
        if self.input_dims != config.hidden_size:
            self.input_up_proj = nn.Sequential(
                nn.Linear(input_dims, config.hidden_size),
                nn.Tanh(),
                nn.Linear(config.hidden_size, config.hidden_size)
            )

        # -----------------------------------------------------------------------
        # BERT Transformer Encoder + Positional Embeddings
        # -----------------------------------------------------------------------
        if init_pretrained == 'bert':
            # Initialize from pretrained BERT weights.
            # Shares word embeddings, position embeddings, and LayerNorm with BERT.
            print('initializing from pretrained bert...')
            print(config)
            temp_bert = BertModel.from_pretrained(config_name, config=config)

            self.word_embedding = temp_bert.embeddings.word_embeddings
            with th.no_grad():
                self.lm_head.weight = self.word_embedding.weight

            # Use BERT's encoder (12 self-attention layers)
            self.input_transformers = temp_bert.encoder
            # position_ids: [1, max_pos] integer buffer — pre-computed position indices
            self.register_buffer(
                "position_ids",
                torch.arange(config.max_position_embeddings).expand((1, -1))
            )
            self.position_embeddings = temp_bert.embeddings.position_embeddings
            self.LayerNorm = temp_bert.embeddings.LayerNorm

            # Free BERT's unused components to save memory
            del temp_bert.embeddings
            del temp_bert.pooler

        elif init_pretrained == 'no':
            # Random initialization: use a fresh BERT encoder with no pretrained weights
            self.input_transformers = BertEncoder(config)

            self.register_buffer(
                "position_ids",
                torch.arange(config.max_position_embeddings).expand((1, -1))
            )
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        else:
            assert False, "invalid type of init_pretrained"

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # -----------------------------------------------------------------------
        # Output Projection (only if output_dims ≠ BERT hidden_size)
        # -----------------------------------------------------------------------
        # Maps BERT's hidden states back to the embedding space (hidden_dim=128).
        if self.output_dims != config.hidden_size:
            self.output_down_proj = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.Tanh(),
                nn.Linear(config.hidden_size, self.output_dims)
            )

        # -----------------------------------------------------------------------
        # Optional Learnable Mean Embedding
        # -----------------------------------------------------------------------
        # When learned_mean_embed=True, a trainable global mean vector is added.
        # This helps the model learn the expected center of the embedding space,
        # which can stabilize training under the denoise_rate curriculum.
        if learned_mean_embed:
            self.mean_embed = nn.Parameter(th.randn(input_dims))
            nn.init.normal_(self.mean_embed, mean=0, std=input_dims ** -0.5)
        else:
            self.mean_embed = None

    def get_embeds(self, input_ids):
        """
        Look up embedding vectors for a sequence of token IDs.

        Args:
            input_ids (Tensor): [batch, seq_len] integer token IDs.

        Returns:
            Tensor: [batch, seq_len, input_dims] embedding vectors.
        """
        return self.word_embedding(input_ids)

    def get_logits(self, hidden_repr):
        """
        Project latent embedding vectors to vocabulary logits for decoding.

        Mode 1 (default): straightforward dot product with the embedding matrix
                          → [batch, seq_len, vocab_size] logits.
        Mode 2: cosine-distance-based scoring (negated L2 distance in embedding space)
                → equivalent to nearest-neighbour search in the embedding space.

        Args:
            hidden_repr (Tensor): [batch, seq_len, input_dims] latent vectors.

        Returns:
            Tensor: [batch, seq_len, vocab_size] logit scores.
        """
        if self.logits_mode == 1:
            return self.lm_head(hidden_repr)  # simple linear projection (weight-tied)
        elif self.logits_mode == 2:
            # Efficiently computes negative squared L2 distance using the identity:
            # ||a - b||² = ||a||² + ||b||² - 2 * a·b
            text_emb = hidden_repr
            emb_norm = (self.lm_head.weight ** 2).sum(-1).view(-1, 1)       # [vocab, 1]
            text_emb_t = th.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1)  # [d, B*L]
            arr_norm = (text_emb ** 2).sum(-1).view(-1, 1)                  # [B*L, 1]
            dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * th.mm(self.lm_head.weight, text_emb_t)
            scores = th.sqrt(th.clamp(dist, 0.0, np.inf)).view(
                emb_norm.size(0), hidden_repr.size(0), hidden_repr.size(1)
            )
            scores = -scores.permute(1, 2, 0).contiguous()  # negate: lower dist = higher score
            return scores
        else:
            raise NotImplementedError

    def forward(self, x, timesteps):
        """
        Denoise x_t back toward x_0 given the diffusion timestep t.

        Full data flow:
            x_t [B, L, input_dims]
            → (up-project if needed) → emb_x [B, L, hidden_size]
            → (add) pos_embed(pos) + emb_x + e_t.unsqueeze(1)
            → LayerNorm + Dropout
            → BERT encoder (self-attention)
            → (down-project if needed) → h [B, L, output_dims]

        Args:
            x (Tensor):         [B, seq_len, input_dims] noisy embedding sequence x_t.
            timesteps (Tensor): [B] integer diffusion timestep indices.

        Returns:
            Tensor: [B, seq_len, output_dims] predicted clean embeddings x̂_0.
        """
        # 1. Build timestep conditioning vector: t → sinusoidal → MLP → e_t ∈ R^{hidden_size}
        emb_t = self.time_embed(timestep_embedding(timesteps, self.hidden_t_dim))

        # 2. Project input embeddings to BERT's hidden size (if needed)
        if self.input_dims != self.hidden_size:
            emb_x = self.input_up_proj(x)         # [B, L, hidden_size]
        else:
            emb_x = x

        # 3. Add positional and timestep embeddings to get the fused input
        seq_length = x.size(1)
        position_ids = self.position_ids[:, :seq_length]
        # emb_t is [B, hidden_size]; unsqueeze to [B, 1, hidden_size] then broadcast over seq
        emb_inputs = (
            self.position_embeddings(position_ids)   # [1, L, hidden_size] → broadcast
            + emb_x                                  # [B, L, hidden_size]
            + emb_t.unsqueeze(1).expand(-1, seq_length, -1)  # [B, L, hidden_size]
        )
        emb_inputs = self.dropout(self.LayerNorm(emb_inputs))

        # 4. BERT Transformer encoder: captures bidirectional context across all positions
        input_trans_hidden_states = self.input_transformers(emb_inputs).last_hidden_state

        # 5. Project back to output_dims (if needed)
        if self.output_dims != self.hidden_size:
            h = self.output_down_proj(input_trans_hidden_states)  # [B, L, output_dims]
        else:
            h = input_trans_hidden_states

        # Preserve input dtype (fp16 or fp32) to match what was passed in
        h = h.type(x.dtype)
        return h