"""`image/upscale.py` — 1111 statements, and until now none of them ran.

The largest untested module in the repo, and not a dead one: `delivery/handler.py`
imports `RadianceAIUpscale` from it, and `nodes/upscale/upscale.py` routes single
frames here.

`RadianceAIUpscale` needs model weights, so it is out of reach without a
download. The other four classes are pure tensor work and were reachable all
along. What they do is resample and requantise, which is exactly where an HDR
range gets quietly destroyed — a saturating cast or a stray clip in any of them
turns a scene-linear plate into a display-referred one and nothing raises.

So these tests are mostly about range: what survives a conversion, what is
supposed to be clamped, and what comes out the wrong size.
"""
import warnings

import numpy as np
import pytest
import torch

from radiance.image.upscale import (
    RadianceBitDepthConvert,
    RadianceDownscale32bit,
    RadianceSharpen32bit,
    RadianceUpscaleBySize,
)


def hdr_ramp(h=8, w=8, lo=-0.25, hi=64.0):
    """A ramp that leaves [0, 1] at both ends, which is the whole point."""
    return torch.linspace(lo, hi, h * w * 3).reshape(1, h, w, 3).float()


# ── bit depth ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("depth", ["32-bit Float", "16-bit Float"])
def test_the_float_depths_do_not_clamp(depth):
    """Negatives and over-range survive. This is the clamp-free HDR claim, at
    the one node whose job is changing precision."""
    img = hdr_ramp()
    out, _info = RadianceBitDepthConvert().convert(
        image=img, output_depth=depth, dithering="None")
    assert float(out.min()) < 0.0, f"{depth} clamped the negatives away"
    assert float(out.max()) > 1.0, f"{depth} clamped the highlights"
    assert out.dtype == torch.float32, "the node must hand back float32 either way"


@pytest.mark.parametrize("depth,levels", [("16-bit Int", 65535), ("10-bit", 1023), ("8-bit", 255)])
def test_the_integer_depths_clamp_and_quantise(depth, levels):
    """Clamping is correct here — an integer format has no room above 1.0 —
    so this pins the boundary rather than complaining about it."""
    out, _ = RadianceBitDepthConvert().convert(
        image=hdr_ramp(), output_depth=depth, dithering="None")
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
    # Every value must sit on a quantisation step of the target depth.
    steps = out.flatten() * levels
    assert torch.allclose(steps, steps.round(), atol=1e-3), (
        f"{depth} output is not on {levels} levels"
    )


def test_16_bit_float_actually_quantises_to_half():
    """Otherwise the menu entry is a no-op with a label."""
    v = torch.tensor([[[[0.3333333, 0.1, 0.7]]]], dtype=torch.float32)
    out, _ = RadianceBitDepthConvert().convert(
        image=v, output_depth="16-bit Float", dithering="None")
    assert not torch.equal(out, v), "16-bit Float left the values untouched"
    assert torch.equal(out, v.half().float())


def test_a_value_above_the_half_ceiling_becomes_inf(recwarn):
    """Current behaviour, written down rather than assumed.

    float16 tops out at 65504. Converting anything larger gives inf, and inf is
    not a colour: it propagates through the rest of the graph and turns into
    NaN at the first inf-minus-inf. Clamping to 65504 — which is what a half
    EXR write does — would be the kinder answer. Left as-is because it changes
    pixels, and pinned here so the change is a decision rather than a surprise.
    """
    big = torch.tensor([[[[1e5, 70000.0, 65504.0]]]], dtype=torch.float32)
    out, _ = RadianceBitDepthConvert().convert(
        image=big, output_depth="16-bit Float", dithering="None")
    assert torch.isinf(out).sum() == 2
    assert float(out.flatten()[2]) == 65504.0


@pytest.mark.parametrize("dither", ["Ordered", "Floyd-Steinberg"])
def test_dithering_is_deterministic_for_a_seed(dither):
    """A dither that moves between two runs of the same graph makes every
    downstream comparison — and every A/B — meaningless."""
    img = torch.rand(1, 8, 8, 3, generator=torch.Generator().manual_seed(0))
    node = RadianceBitDepthConvert()
    a, _ = node.convert(image=img, output_depth="8-bit", dithering=dither,
                        dither_strength=1.0, seed=7)
    b, _ = node.convert(image=img, output_depth="8-bit", dithering=dither,
                        dither_strength=1.0, seed=7)
    assert torch.equal(a, b), f"{dither} is not reproducible at a fixed seed"


def test_dithering_actually_changes_the_result():
    img = torch.rand(1, 16, 16, 3, generator=torch.Generator().manual_seed(1))
    node = RadianceBitDepthConvert()
    plain, _ = node.convert(image=img, output_depth="8-bit", dithering="None")
    dithered, _ = node.convert(image=img, output_depth="8-bit", dithering="Ordered",
                               dither_strength=1.0, seed=3)
    assert not torch.equal(plain, dithered), "the dither did nothing"


# ── geometry ────────────────────────────────────────────────────────────────

def test_upscale_by_size_hits_the_requested_size_exactly():
    out, w, h = RadianceUpscaleBySize().upscale(
        image=torch.zeros(1, 10, 20, 3), width=100, height=64, method="Nearest",
        maintain_aspect=False, process_in_linear=False, input_color_space="Linear")
    assert (w, h) == (100, 64)
    assert tuple(out.shape) == (1, 64, 100, 3), tuple(out.shape)


def test_maintain_aspect_fit_keeps_the_source_ratio():
    """20x10 into a 100x100 box is 100x50, not 100x100."""
    out, w, h = RadianceUpscaleBySize().upscale(
        image=torch.zeros(1, 10, 20, 3), width=100, height=100, method="Nearest",
        maintain_aspect=True, aspect_mode="fit",
        process_in_linear=False, input_color_space="Linear")
    assert (w, h) == (100, 50)
    assert tuple(out.shape) == (1, 50, 100, 3)


@pytest.mark.parametrize("factor,expect", [(0.5, (8, 8)), (0.25, (4, 4))])
def test_downscale_lands_on_the_expected_size(factor, expect):
    out, w, h = RadianceDownscale32bit().downscale(
        image=torch.zeros(1, 16, 16, 3), scale_factor=factor, method="Lanczos",
        process_in_linear=False, use_gpu=False, input_color_space="Linear")
    assert (w, h) == expect
    assert tuple(out.shape) == (1, expect[1], expect[0], 3)


def test_downscale_never_produces_a_zero_dimension():
    """A scale factor small enough to round a side to zero would otherwise
    hand an empty tensor to whatever comes next."""
    out, w, h = RadianceDownscale32bit().downscale(
        image=torch.zeros(1, 4, 4, 3), scale_factor=0.01, method="Lanczos",
        process_in_linear=False, use_gpu=False, input_color_space="Linear")
    assert w >= 1 and h >= 1
    assert out.numel() > 0


def test_resampling_keeps_scene_linear_values_out_of_the_unit_box():
    """Not a precise-value test — Lanczos rings, and that is correct. The
    property is that an over-range plate is still over-range afterwards."""
    out, _w, _h = RadianceDownscale32bit().downscale(
        image=hdr_ramp(16, 16), scale_factor=0.5, method="Lanczos",
        process_in_linear=False, use_gpu=False, input_color_space="Linear")
    assert float(out.max()) > 1.0, "the downscale clipped the highlights"
    assert torch.isfinite(out).all()


def test_the_srgb_decode_only_runs_when_the_input_is_called_srgb():
    """Feeding scene-linear through an sRGB decode is a silent 2.2-ish gamma
    error. The two paths must not agree, or the switch does nothing."""
    img = hdr_ramp(16, 16)
    node = RadianceDownscale32bit()
    as_srgb, _, _ = node.downscale(image=img, scale_factor=0.5, method="Lanczos",
                                   process_in_linear=True, use_gpu=False,
                                   input_color_space="sRGB")
    as_linear, _, _ = node.downscale(image=img, scale_factor=0.5, method="Lanczos",
                                     process_in_linear=True, use_gpu=False,
                                     input_color_space="Linear")
    assert not torch.allclose(as_srgb, as_linear), (
        "input_color_space made no difference — the decode is running either "
        "way, or not at all"
    )


# ── sharpen ─────────────────────────────────────────────────────────────────

def test_sharpen_at_zero_amount_is_a_pass_through():
    img = hdr_ramp(16, 16)
    out = RadianceSharpen32bit().sharpen(image=img, amount=0.0, radius=1.0, threshold=0.0)
    out = out[0] if isinstance(out, tuple) else out
    assert torch.allclose(out, img, atol=1e-6), "amount=0 changed the picture"


def test_sharpen_keeps_the_shape_and_stays_finite():
    img = hdr_ramp(16, 16)
    out = RadianceSharpen32bit().sharpen(image=img, amount=0.5, radius=1.0, threshold=0.0)
    out = out[0] if isinstance(out, tuple) else out
    assert tuple(out.shape) == tuple(img.shape)
    assert torch.isfinite(out).all(), "unsharp mask produced a non-finite pixel"
