"""
transformer_model.py — the denoising Transformer of DiffuSeq.

Given the noisy sequence x_t (source positions clean, target positions noised) and the timestep t,
the model predicts x_0 for every position.

Data flow in forward() (B = batch, L = seq_len, D = hidden_dim, H = backbone hidden size, 768 for BERT):
    0. optional graph encoder:  x [B, L, D] -> x + GATv2 delta on AMR source positions (graph_encoder.py)
    1. timestep embedding:      t [B] -> sinusoidal [B, hidden_t_dim] -> MLP -> e_t [B, H]
    2. input projection:        x [B, L, D] -> [B, L, H]   (Linear-Tanh-Linear when D != H)
    3. fused input:             pos_emb [1, L, H] + x_proj + e_t broadcast over L -> LayerNorm -> Dropout
    4. BERT encoder:            [B, L, H] -> [B, L, H]   (no attention mask: [PAD] is a generated token)
    5. output projection:       [B, L, H] -> [B, L, D]   (Linear-Tanh-Linear when D != H)

The vocabulary head `lm_head` shares its weight with `word_embedding` (weight tying), so decoding
maps predicted vectors back to tokens through the same embedding space.
"""

import numpy as np
import torch
import torch as th
import torch.nn as nn
from transformers import AutoConfig
from transformers.models.bert.modeling_bert import BertEncoder, BertModel

from .utils.nn import SiLU, linear, timestep_embedding


class TransformerNetModel(nn.Module):
    """
    Args:
        input_dims:        embedding size D (diffusion space).
        output_dims:       D, or 2*D when learn_sigma.
        hidden_t_dim:      size of the sinusoidal timestep embedding.
        dropout:           hidden dropout of the backbone.
        config / config_name: backbone config (e.g. 'bert-base-multilingual-cased').
        vocab_size:        rows of word_embedding / lm_head.
        init_pretrained:   'bert' = pretrained encoder + embeddings (requires D == H); 'no' = random.
        logits_mode:       1 = dot product with the tied embedding; 2 = negative L2 distance.
        learned_mean_embed: learn the DiffuSeq-v2 absorbing-state vector used by `denoise`.
        graph_encoder:     'none' | 'gatv2' (see graph_encoder.py).
        graph_layers, graph_heads, num_edge_types: GATv2 settings.
    """

    def __init__(
        self,
        input_dims,
        output_dims,
        hidden_t_dim,
        dropout=0,
        config=None,
        config_name='bert-base-multilingual-cased',
        vocab_size=None,
        init_pretrained='no',
        logits_mode=1,
        learned_mean_embed=False,
        graph_encoder='none',
        graph_layers=2,
        graph_heads=4,
        num_edge_types=0,
    ):
        super().__init__()

        if config is None:
            config = AutoConfig.from_pretrained(config_name)
            config.hidden_dropout_prob = dropout

        self.input_dims = input_dims
        self.hidden_t_dim = hidden_t_dim
        self.output_dims = output_dims
        self.dropout = dropout
        self.logits_mode = logits_mode
        self.hidden_size = config.hidden_size

        self.word_embedding = nn.Embedding(vocab_size, self.input_dims)
        self.lm_head = nn.Linear(self.input_dims, vocab_size)
        with th.no_grad():
            self.lm_head.weight = self.word_embedding.weight

        # t -> sinusoidal(hidden_t_dim) -> Linear(4*hidden_t_dim) -> SiLU -> Linear(H)
        time_embed_dim = hidden_t_dim * 4
        self.time_embed = nn.Sequential(
            linear(hidden_t_dim, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, config.hidden_size),
        )

        if self.input_dims != config.hidden_size:
            self.input_up_proj = nn.Sequential(
                nn.Linear(input_dims, config.hidden_size),
                nn.Tanh(),
                nn.Linear(config.hidden_size, config.hidden_size),
            )

        if init_pretrained == 'bert':
            if input_dims != config.hidden_size:
                # Pretrained word embeddings are H-dimensional; tying them to a D-dimensional head
                # fails at the first forward pass (review bug B8).
                raise ValueError(
                    f"init_pretrained='bert' requires hidden_dim == {config.hidden_size}, got {input_dims}"
                )
            print('initializing from pretrained bert...')
            temp_bert = BertModel.from_pretrained(config_name, config=config)
            self.word_embedding = temp_bert.embeddings.word_embeddings
            if vocab_size is not None and vocab_size != self.word_embedding.num_embeddings:
                # Extra rows (AMR tokens beyond the base vocabulary) are randomly initialized.
                old_emb = self.word_embedding
                self.word_embedding = nn.Embedding(vocab_size, old_emb.embedding_dim)
                with th.no_grad():
                    nn.init.normal_(self.word_embedding.weight)
                    self.word_embedding.weight[:old_emb.num_embeddings] = old_emb.weight
                self.lm_head = nn.Linear(self.input_dims, vocab_size)
            with th.no_grad():
                self.lm_head.weight = self.word_embedding.weight
            self.input_transformers = temp_bert.encoder
            self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
            self.position_embeddings = temp_bert.embeddings.position_embeddings
            self.LayerNorm = temp_bert.embeddings.LayerNorm
            del temp_bert.embeddings
            del temp_bert.pooler
        elif init_pretrained == 'no':
            self.input_transformers = BertEncoder(config)
            self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            raise ValueError(f"invalid init_pretrained {init_pretrained!r}")

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if self.output_dims != config.hidden_size:
            self.output_down_proj = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.Tanh(),
                nn.Linear(config.hidden_size, self.output_dims),
            )

        if learned_mean_embed:
            self.mean_embed = nn.Parameter(th.randn(input_dims))
            nn.init.normal_(self.mean_embed, mean=0, std=input_dims ** -0.5)
        else:
            self.mean_embed = None

        self.graph_encoder = None
        if graph_encoder == 'gatv2':
            from .graph_encoder import GraphEncoder
            self.graph_encoder = GraphEncoder(input_dims, num_edge_types, num_layers=graph_layers,
                                              heads=graph_heads, dropout=dropout)
        elif graph_encoder != 'none':
            raise ValueError(f"graph_encoder must be 'none' or 'gatv2', got {graph_encoder!r}")

    def get_embeds(self, input_ids):
        """[B, L] ids -> [B, L, D] embeddings."""
        return self.word_embedding(input_ids)

    def get_logits(self, hidden_repr):
        """[B, L, D] vectors -> [B, L, vocab] scores (dot product, or negative L2 distance)."""
        if self.logits_mode == 1:
            return self.lm_head(hidden_repr)
        elif self.logits_mode == 2:
            # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b, computed for every (position, vocab entry)
            text_emb = hidden_repr
            emb_norm = (self.lm_head.weight ** 2).sum(-1).view(-1, 1)                # [V, 1]
            text_emb_t = th.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1)     # [D, B*L]
            arr_norm = (text_emb ** 2).sum(-1).view(-1, 1)                            # [B*L, 1]
            dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * th.mm(self.lm_head.weight, text_emb_t)
            scores = th.sqrt(th.clamp(dist, 0.0, np.inf)).view(
                emb_norm.size(0), hidden_repr.size(0), hidden_repr.size(1))
            return -scores.permute(1, 2, 0).contiguous()
        raise NotImplementedError

    def forward(self, x, timesteps, edge_index=None, edge_type=None, **kwargs):
        """x: [B, L, D] noisy sequence; timesteps: [B] -> predicted x_0 [B, L, output_dims]."""
        if self.graph_encoder is not None and edge_index is not None:
            x = self.graph_encoder(x, edge_index, edge_type)

        emb_t = self.time_embed(timestep_embedding(timesteps, self.hidden_t_dim))      # [B, H]
        emb_x = self.input_up_proj(x) if self.input_dims != self.hidden_size else x    # [B, L, H]

        seq_length = x.size(1)
        position_ids = self.position_ids[:, :seq_length]
        emb_inputs = (
            self.position_embeddings(position_ids)
            + emb_x
            + emb_t.unsqueeze(1).expand(-1, seq_length, -1)
        )
        emb_inputs = self.dropout(self.LayerNorm(emb_inputs))

        hidden = self.input_transformers(emb_inputs).last_hidden_state                  # [B, L, H]
        h = self.output_down_proj(hidden) if self.output_dims != self.hidden_size else hidden
        return h.type(x.dtype)
