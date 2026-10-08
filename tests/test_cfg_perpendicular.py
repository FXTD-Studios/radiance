"""CFG++ (Perpendicular): guidance with the part along the conditional
prediction removed.

Until 4.0 the mode only scaled cfg by a cosine of each stage's starting sigma;
nothing was perpendicular. It now also installs a cfg function that keeps only
the component of (cond - uncond) orthogonal to the conditional prediction,
per batch item (projected guidance, as in APG). The cosine stage scale stays:
it arrives as cond_scale.

ComfyUI's contract (comfy/samplers.py cfg_function): `cond` and `uncond` are
x minus the denoised predictions, and the function returns x minus the guided
prediction.
"""
import torch
import pytest

from radiance.sampler_utils import perpendicular_cfg_function


def _args(cond_denoised, uncond_denoised, scale, x=None):
    x = torch.randn_like(cond_denoised) if x is None else x
    return {
        "cond": x - cond_denoised, "uncond": x - uncond_denoised, "cond_scale": scale,
        "input": x, "cond_denoised": cond_denoised, "uncond_denoised": uncond_denoised,
    }


def _guided(args):
    return args["input"] - perpendicular_cfg_function(args)


def test_the_added_guidance_is_orthogonal_to_the_conditional_prediction():
    torch.manual_seed(0)
    cond, uncond = torch.randn(3, 4, 8, 8), torch.randn(3, 4, 8, 8)
    out = _guided(_args(cond, uncond, 5.0))
    added = (out - cond).flatten(1)
    dots = (added * cond.flatten(1)).sum(1)
    assert torch.allclose(dots, torch.zeros(3), atol=1e-3), dots


def test_it_differs_from_plain_cfg_by_exactly_the_parallel_part():
    torch.manual_seed(1)
    cond, uncond = torch.randn(1, 4, 8, 8), torch.randn(1, 4, 8, 8)
    diff = cond - uncond
    par = (diff * cond).sum() / (cond * cond).sum() * cond
    plain = uncond + 4.0 * diff
    assert torch.allclose(_guided(_args(cond, uncond, 4.0)), plain - 3.0 * par, atol=1e-5)


def test_cfg_1_returns_the_conditional_prediction():
    torch.manual_seed(2)
    cond, uncond = torch.randn(2, 4, 8, 8), torch.randn(2, 4, 8, 8)
    assert torch.allclose(_guided(_args(cond, uncond, 1.0)), cond, atol=1e-6)


def test_a_zero_conditional_prediction_does_not_divide_by_zero():
    cond, uncond = torch.zeros(1, 4, 8, 8), torch.randn(1, 4, 8, 8)
    out = _guided(_args(cond, uncond, 3.0))
    assert torch.isfinite(out).all()


def test_each_batch_item_is_projected_on_its_own():
    torch.manual_seed(3)
    cond, uncond = torch.randn(2, 4, 8, 8), torch.randn(2, 4, 8, 8)
    both = _guided(_args(cond, uncond, 6.0, x=torch.zeros_like(cond)))
    one = _guided(_args(cond[1:], uncond[1:], 6.0, x=torch.zeros_like(cond[1:])))
    assert torch.allclose(both[1:], one, atol=1e-5)


def test_video_latents_with_five_dimensions_work():
    torch.manual_seed(4)
    cond, uncond = torch.randn(1, 4, 3, 8, 8), torch.randn(1, 4, 3, 8, 8)
    out = _guided(_args(cond, uncond, 5.0))
    assert out.shape == cond.shape


# ── the node installs it for CFG++ only ─────────────────────────────────────

def _model_options(**overrides):
    from _sampler_harness import run_sampler
    _, rec = run_sampler(**overrides)
    return rec.calls[-1]["model"].model_options


def test_cfg_plus_plus_installs_the_perpendicular_cfg_function():
    opts = _model_options(sampler_mode="CFG++ (Perpendicular)", cfg=5.0)
    assert opts.get("sampler_cfg_function") is perpendicular_cfg_function


def test_standard_mode_installs_no_cfg_function():
    opts = _model_options(sampler_mode="Standard", cfg=5.0)
    assert "sampler_cfg_function" not in opts


def test_the_mode_keeps_its_saved_name():
    from radiance.sampler_utils import SamplerMode
    assert "CFG++ (Perpendicular)" in SamplerMode.ALL
