"""diffuseq/graph_encoder.py — GATv2 module (replaces gcn.py, bug B4)."""

import pytest
import torch

pytest.importorskip("torch_geometric")

from diffuseq.graph_encoder import GraphEncoder  # noqa: E402

B, L, D = 2, 10, 16


def edges():
    # sample 0: 1 <-> 2 ; sample 1: 3 <-> 4  (node = b * L + pos)
    return torch.tensor([[1, 2, 13, 14], [2, 1, 14, 13]]), torch.tensor([0, 1, 0, 1])


def test_zero_init_is_identity():
    enc = GraphEncoder(D, num_edge_types=4, num_layers=2, heads=4, dropout=0.0)
    x = torch.randn(B, L, D)
    ei, et = edges()
    assert torch.equal(enc(x, ei, et), x)


def test_only_graph_nodes_change():
    torch.manual_seed(0)
    enc = GraphEncoder(D, num_edge_types=4, num_layers=2, heads=4, dropout=0.0)
    torch.nn.init.normal_(enc.out_proj.weight)            # leave the zero init
    x = torch.randn(B, L, D)
    ei, et = edges()
    y = enc(x, ei, et)
    changed = ~torch.isclose(y, x).all(-1)                 # [B, L]
    expected = torch.zeros(B, L, dtype=torch.bool)
    expected[0, 1] = expected[0, 2] = expected[1, 3] = expected[1, 4] = True
    assert torch.equal(changed, expected)


def test_edge_type_matters():
    torch.manual_seed(0)
    enc = GraphEncoder(D, num_edge_types=4, num_layers=1, heads=4, dropout=0.0)
    torch.nn.init.normal_(enc.out_proj.weight)
    x = torch.randn(B, L, D)
    ei, _ = edges()
    a = enc(x, ei, torch.tensor([0, 0, 0, 0]))
    b = enc(x, ei, torch.tensor([2, 2, 2, 2]))
    assert not torch.allclose(a, b)


def test_empty_graph_and_out_of_range():
    enc = GraphEncoder(D, num_edge_types=4)
    x = torch.randn(B, L, D)
    assert torch.equal(enc(x, torch.empty((2, 0), dtype=torch.long), torch.empty(0, dtype=torch.long)), x)
    with pytest.raises(AssertionError):
        enc(x, torch.tensor([[0], [B * L]]), torch.tensor([0]))


def test_heads_must_divide_hidden_dim():
    with pytest.raises(ValueError):
        GraphEncoder(10, num_edge_types=4, heads=4)
