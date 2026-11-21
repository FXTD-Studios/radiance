"""Bit Depth Degrade `restore_from_quantized`, a no-op until 4.0.

It now dequantises: the quantised image is smoothed in a few passes, and after
each pass every pixel is clamped back into the range it could have come from
(half a step around its code value without dither; a step and a half with
dither, which moves values by up to a step before rounding). Off, the node is
unchanged.
"""
import json

import torch
import pytest

from radiance.nodes.color.colorspace import RadianceBitDepthDegrade


def _ramp(h=32, w=96):
    x = torch.linspace(0.0, 1.0, w).view(1, 1, w, 1).expand(1, h, w, 3)
    return x.contiguous()


def _run(img, **kw):
    kw.setdefault("bit_depth", 4)
    kw.setdefault("dither_mode", "none")
    return RadianceBitDepthDegrade().degrade(img, **kw)


def test_restoring_a_banded_ramp_brings_it_closer_to_the_original():
    img = _ramp()
    quantized = _run(img)[0]
    restored = _run(img, restore_from_quantized=True)[0]
    assert ((restored - img) ** 2).mean() < 0.5 * ((quantized - img) ** 2).mean()


def test_the_restored_image_stays_within_half_a_step_of_the_code_values():
    img = _ramp()
    quantized = _run(img)[0]
    restored = _run(img, restore_from_quantized=True)[0]
    assert ((restored - quantized).abs() <= 0.5 / 15 + 1e-6).all()


def test_off_is_unchanged():
    img = _ramp()
    torch.manual_seed(0)
    a = _run(img, dither_mode="none")
    b = _run(img, dither_mode="none", restore_from_quantized=False)
    for x, y in zip(a[:3], b[:3]):
        assert torch.equal(x, y)
    assert "restored" not in json.loads(a[3])


def test_metrics_say_the_output_was_restored_and_measure_it():
    img = _ramp()
    out, delta_amp, _, metrics = _run(img, restore_from_quantized=True, delta_gain=1.0)
    m = json.loads(metrics)
    assert m["restored"] is True
    assert torch.allclose(delta_amp, (img - out).abs().clamp(0, 1), atol=1e-6)


@pytest.mark.parametrize("mode", ["triangular", "floyd-steinberg"])
def test_dithered_input_is_restored_within_a_step_and_a_half(mode):
    torch.manual_seed(1)
    img = _ramp(16, 48)
    torch.manual_seed(1)
    quantized = _run(img, dither_mode=mode)[0]
    torch.manual_seed(1)
    restored = _run(img, dither_mode=mode, restore_from_quantized=True)[0]
    assert ((restored - quantized).abs() <= 1.5 / 15 + 1e-6).all()
    assert ((restored - img) ** 2).mean() < ((quantized - img) ** 2).mean()


def test_alpha_and_one_pixel_images_survive():
    img = torch.rand(1, 1, 1, 4)
    out = _run(img, restore_from_quantized=True)[0]
    assert out.shape == img.shape and torch.equal(out[..., 3], img[..., 3])
