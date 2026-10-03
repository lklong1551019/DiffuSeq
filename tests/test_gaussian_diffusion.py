"""Discrete-noise masking (bug B5, walkthrough §3.5) and source anchoring in q_sample."""

import torch

from diffuseq import gaussian_diffusion as gd
from diffuseq.gaussian_diffusion import SpacedDiffusion, build_denoise_mask, space_timesteps


def test_build_denoise_mask_is_per_row():
    random_mask = torch.tensor([[1., 1., 1., 0.], [1., 0., 1., 1.]])
    rel_mask = torch.tensor([[0, 0, 1, 0], [0, 0, 0, 0]])
    out = build_denoise_mask(random_mask, rel_mask, mask_docamr_rel=True)
    assert out.tolist() == [[0., 0., 1., 0.], [1., 0., 1., 1.]]


def test_build_denoise_mask_disabled_or_missing():
    random_mask = torch.tensor([[1., 0.], [0., 1.]])
    rel_mask = torch.tensor([[1, 0], [0, 0]])
    assert torch.equal(build_denoise_mask(random_mask, rel_mask, mask_docamr_rel=False), random_mask)
    assert torch.equal(build_denoise_mask(random_mask, None, mask_docamr_rel=True), random_mask)


def make_diffusion(denoise_rate):
    return SpacedDiffusion(
        use_timesteps=space_timesteps(100, [100]), betas=gd.get_named_beta_schedule("sqrt", 100),
        rescale_timesteps=True, predict_xstart=True, learn_sigmas=False, sigma_small=False, use_kl=False,
        rescale_learned_sigmas=False, denoise=True, denoise_rate=denoise_rate, device="cpu", max_T=100,
    )


def test_q_sample_restores_source_and_noises_target():
    torch.manual_seed(0)
    d = make_diffusion(denoise_rate=0.0)
    x_start = torch.randn(2, 6, 4)
    mask = torch.tensor([[0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 1, 1]])
    x_t = d.q_sample(x_start, torch.tensor([50, 50]), mask=mask, mean_embed=torch.zeros(4))
    src = (mask == 0)
    assert torch.equal(x_t[src], x_start[src])                   # source positions untouched
    assert not torch.allclose(x_t[~src], x_start[~src])          # target positions noised


def test_q_sample_mean_embed_replacement_only_on_target():
    torch.manual_seed(0)
    d = make_diffusion(denoise_rate=1.0)
    x_start = torch.randn(1, 6, 4)
    mean = torch.full((4,), 7.0)
    mask = torch.tensor([[0, 0, 0, 1, 1, 1]])
    x_t = d.q_sample(x_start, torch.tensor([99]), mask=mask, mean_embed=mean)
    replaced = (x_t == 7.0).all(-1)[0]
    assert not replaced[:3].any()                                # never on source
    assert replaced[3:].any()                                    # p ~ 1 at t = T-1
