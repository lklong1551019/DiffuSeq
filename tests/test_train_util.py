"""train_util.slice_microbatch — per-sample slicing and graph re-numbering (bug B4c, walkthrough §3.3–3.4)."""

import torch

from train_util import slice_microbatch

L, D = 20, 4


def make_batch(b=4):
    batch = torch.arange(b, dtype=torch.float32).view(b, 1, 1).expand(b, L, D).clone()
    cond = {
        "input_ids": torch.arange(b).view(b, 1).expand(b, L).clone(),
        "input_mask": torch.ones(b, L, dtype=torch.long),
        # sample 0: 6->7; sample 1: none; sample 2: 6->7 (46->47); sample 3: 6->7, 7->8 (66->67, 67->68)
        "edge_index": torch.tensor([[6, 46, 66, 67], [7, 47, 67, 68]]),
        "edge_type": torch.tensor([10, 11, 12, 13]),
    }
    return batch, cond


def test_walkthrough_example_sample3_pos6():
    batch, cond = make_batch()
    micro, mc = slice_microbatch(batch, cond, start=2, size=2, device="cpu")
    assert micro[:, 0, 0].tolist() == [2.0, 3.0]
    assert mc["input_ids"][:, 0].tolist() == [2, 3]
    assert mc["edge_index"].tolist() == [[6, 26, 27], [7, 27, 28]]   # 46-40, 66-40, 67-40
    assert mc["edge_type"].tolist() == [11, 12, 13]
    node = mc["edge_index"][0, 1].item()
    assert divmod(node, L) == (1, 6)                                  # sample 1 of micro = batch 3, pos 6


def test_first_microbatch_keeps_only_its_edges():
    batch, cond = make_batch()
    _, mc = slice_microbatch(batch, cond, start=0, size=2, device="cpu")
    assert mc["edge_index"].tolist() == [[6], [7]]
    assert mc["edge_type"].tolist() == [10]


def test_partial_last_microbatch():
    batch, cond = make_batch(b=4)
    micro, mc = slice_microbatch(batch, cond, start=3, size=2, device="cpu")   # 4 % 2 != 0 style tail
    assert micro.shape[0] == 1
    assert mc["edge_index"].tolist() == [[6, 7], [7, 8]]
    assert int(mc["edge_index"].max()) < micro.shape[0] * L


def test_every_edge_assigned_exactly_once():
    batch, cond = make_batch()
    total = 0
    for start in range(0, 4, 3):          # microbatches of 3: [0,3), [3,4)
        micro, mc = slice_microbatch(batch, cond, start, 3, "cpu")
        if mc["edge_index"].numel():
            assert int(mc["edge_index"].max()) < micro.shape[0] * L
        total += mc["edge_index"].shape[1]
    assert total == cond["edge_index"].shape[1]


def test_without_graph_keys():
    batch, cond = make_batch()
    cond.pop("edge_index"), cond.pop("edge_type")
    micro, mc = slice_microbatch(batch, cond, 0, 2, "cpu")
    assert set(mc) == {"input_ids", "input_mask"} and micro.shape[0] == 2
