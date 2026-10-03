"""CPU smoke tests: model + diffusion loss end to end, B8 guard, loading an existing checkpoint."""

import glob
import json
import os
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoConfig

import paths
from diffuseq import gaussian_diffusion as gd
from diffuseq.gaussian_diffusion import SpacedDiffusion, space_timesteps
from diffuseq.transformer_model import TransformerNetModel


class DDPLike(torch.nn.Module):
    """Mimics the DDP wrapper TrainLoop passes to training_losses (`.module`, `.mean_embed`)."""

    def __init__(self, m):
        super().__init__()
        self.module = m

    def forward(self, *a, **k):
        return self.module(*a, **k)

    @property
    def mean_embed(self):
        return self.module.mean_embed


def tiny_model(**kw):
    cfg = AutoConfig.from_pretrained("bert-base-multilingual-cased")
    cfg.num_hidden_layers = 1
    return TransformerNetModel(input_dims=16, output_dims=16, hidden_t_dim=16, config=cfg,
                               vocab_size=1000, learned_mean_embed=True, **kw)


def diffusion():
    return SpacedDiffusion(
        use_timesteps=space_timesteps(100, [100]), betas=gd.get_named_beta_schedule("sqrt", 100),
        rescale_timesteps=True, predict_xstart=True, learn_sigmas=False, sigma_small=False, use_kl=False,
        rescale_learned_sigmas=False, denoise=True, denoise_rate=0.5, device="cpu", max_T=100,
    )


def loss_and_backward(model, extra=None):
    ids = torch.randint(0, 1000, (2, 8))
    mask = torch.tensor([[0] * 4 + [1] * 4] * 2)
    kwargs = {"input_ids": ids, "input_mask": mask, "rel_mask": torch.zeros_like(mask)}
    kwargs.update(extra or {})
    out = diffusion().training_losses(DDPLike(model), None, torch.tensor([5, 50]), model_kwargs=kwargs)
    assert torch.isfinite(out["loss"]).all()
    out["loss"].mean().backward()
    return out


def test_plain_model_loss_backward():
    loss_and_backward(tiny_model())


def test_graph_model_loss_backward():
    pytest.importorskip("torch_geometric")
    model = tiny_model(graph_encoder="gatv2", graph_layers=1, graph_heads=4, num_edge_types=4)
    extra = {"edge_index": torch.tensor([[1, 2, 9, 10], [2, 1, 10, 9]]), "edge_type": torch.tensor([0, 1, 0, 1])}
    loss_and_backward(model, extra)
    assert model.graph_encoder.out_proj.weight.grad is not None    # graph path is in the graph


def test_pretrained_init_requires_matching_dim():
    cfg = AutoConfig.from_pretrained("bert-base-multilingual-cased")
    with pytest.raises(ValueError):
        TransformerNetModel(input_dims=256, output_dims=256, hidden_t_dim=16, config=cfg,
                            config_name="bert-base-multilingual-cased", vocab_size=1000, init_pretrained="bert")


def test_existing_checkpoint_loads_with_its_saved_tokenizer():
    runs = sorted(glob.glob(os.path.join(paths.REPO_ROOT, "diffusion_models", "*", "*", "ema_*.pt")))
    if not runs:
        pytest.skip("no checkpoint on disk")
    from basic_utils import create_model_and_diffusion, load_defaults_config, myTokenizer
    ckpt = runs[0]
    run_dir = os.path.dirname(ckpt)
    with open(os.path.join(run_dir, "training_args.json")) as f:
        targs = json.load(f)
    tok = myTokenizer(SimpleNamespace(vocab="bert", config_name=targs["config_name"], tokenizer_dir=run_dir))
    assert tok.vocab_size == targs["vocab_size"]
    cfg = load_defaults_config()
    cfg.update({k: targs[k] for k in cfg if k in targs})
    cfg["device"] = "cpu"
    model, _ = create_model_and_diffusion(**cfg)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
