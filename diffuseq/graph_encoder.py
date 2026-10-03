"""
graph_encoder.py — relation-aware GATv2 over the AMR positions of the source segment.

Replaces gcn.py (review bug B4): GCNConv treats every edge alike (":ARG0" = ":ARG1"), and the old
module silently fell back to nn.Linear when torch_geometric was missing.

Data flow (B = batch, L = seq_len, D = hidden_dim, E = edges in the batch):
    x            [B, L, D]   embeddings entering the denoiser (source positions are clean x_0)
    edge_index   [2, E]      node ids in the flattened batch: node = b * L + pos (collate_with_adj)
    edge_type    [E]         edge-type ids (text_datasets.build_edges), embedded as GATv2 edge features
    1. x_flat = x.reshape(B*L, D)
    2. h = x_flat; per layer: h = h + GELU(GATv2(LayerNorm(h), edge_index, edge_emb))   (pre-norm residual)
    3. delta = out_proj(h)              out_proj is zero-initialised -> delta = 0 at initialisation
    4. delta is kept only on nodes that appear in edge_index (graph nodes); every other position,
       including all noised target positions, passes through unchanged
    5. return x + delta.reshape(B, L, D)

At initialisation the model is therefore identical to the no-graph model; the graph contribution is
learned from zero.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATv2Conv
except ImportError:  # handled in GraphEncoder.__init__ with an explicit error
    GATv2Conv = None


class GraphEncoder(nn.Module):
    def __init__(self, hidden_dim, num_edge_types, num_layers=2, heads=4, dropout=0.1, edge_dim=None):
        super().__init__()
        if GATv2Conv is None:
            raise ImportError(
                "graph_encoder='gatv2' requires torch_geometric (installed in thesis_env). "
                "No silent fallback: a fallback would train without the graph."
            )
        if hidden_dim % heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by graph_heads ({heads})")
        edge_dim = edge_dim or hidden_dim
        self.edge_emb = nn.Embedding(num_edge_types, edge_dim)
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.convs = nn.ModuleList([
            GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, concat=True, edge_dim=edge_dim,
                      add_self_loops=True, fill_value="mean", dropout=dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x, edge_index, edge_type):
        batch, seq_len, dim = x.shape
        if edge_index is None or edge_index.numel() == 0:
            return x
        # Invariant: node ids address positions inside this (micro)batch.
        assert int(edge_index.max()) < batch * seq_len, "edge_index points outside the batch"
        x_flat = x.reshape(batch * seq_len, dim)
        edge_attr = self.edge_emb(edge_type).to(x_flat.dtype)
        h = x_flat
        for norm, conv in zip(self.norms, self.convs):
            h = h + self.dropout(F.gelu(conv(norm(h), edge_index, edge_attr)))
        delta = self.out_proj(h)
        node_mask = torch.zeros(batch * seq_len, 1, dtype=delta.dtype, device=delta.device)
        node_mask[edge_index.reshape(-1)] = 1.0
        return x + (delta * node_mask).reshape(batch, seq_len, dim)
