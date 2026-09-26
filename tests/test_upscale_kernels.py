"""Upscale section, 3.5.0: the method menu and the antialiasing control.

Upscale 32-bit / Upscale By Size ran lanczos, lanczos4, mitchell and catrom as
torch bicubic on the default (untiled) path; Downscale 32-bit never read
`antialiasing`; "Auto" colour space meant Linear.
"""
import numpy as np
import pytest
import torch

pytestmark = pytest.mark.real_torch

from radiance.image import upscale as U

KERNELS = ["lanczos", "lanczos4", "bicubic", "mitchell", "catrom", "hermite", "gaussian",
           "bilinear", "nearest"]


def _img(h=24, w=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, h, w, 3, generator=g)


@pytest.mark.parametrize("method", KERNELS)
@pytest.mark.parametrize("size", [(48, 64), (12, 16), (37, 21)])
def test_gpu_and_cpu_paths_give_the_same_picture(method, size):
    img = _img()
    t = U.torch_resize_32bit(img, *size, method)[0].numpy()
    n = U.separable_resize_32bit(img[0].numpy(), *size, method)
    np.testing.assert_allclose(t, n, atol=1e-5)


def test_untiled_methods_are_really_different():
    img = _img()
    outs = {m: U.RadianceProUpscale().upscale(img, 2.0, "Custom", method=m, sharpening=0,
                                              detail_enhancement=0, antialiasing=0,
                                              input_color_space="Linear")[0]
            for m in ("lanczos", "lanczos4", "mitchell", "catrom")}
    names = list(outs)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not torch.allclose(outs[a], outs[b], atol=1e-4), (a, b)


def test_lanczos_rings_on_an_edge_and_bilinear_does_not():
    edge = torch.zeros(1, 4, 16, 3)
    edge[:, :, 8:] = 1.0
    lz = U.torch_resize_32bit(edge, 4, 64, "lanczos")
    bl = U.torch_resize_32bit(edge, 4, 64, "bilinear")
    assert lz.max() > 1.01 and lz.min() < -0.01
    assert bl.max() <= 1.0 + 1e-6 and bl.min() >= -1e-6


@pytest.mark.parametrize("method", KERNELS)
def test_flat_field_stays_flat(method):
    flat = torch.full((1, 9, 13, 3), 0.37)
    for size in ((27, 39), (4, 5)):
        out = U.torch_resize_32bit(flat, *size, method)
        assert torch.allclose(out, torch.full_like(out, 0.37), atol=1e-5)


def test_hdr_values_survive():
    img = _img() * 40.0
    out = U.torch_resize_32bit(img, 48, 64, "mitchell")
    assert out.max() > 30.0


# ── Downscale antialiasing ─────────────────────────────────────────────────

def _checker(n=64):
    # stripes at 1/3 cycle per pixel: far above the 1/8 the 0.25x output can hold
    x = torch.arange(n).float()
    c = (0.5 + 0.5 * torch.cos(2 * torch.pi * x / 3.0)).expand(n, n)
    return c[None, :, :, None].repeat(1, 1, 1, 3)


def _down(img, aa, method="lanczos"):
    return U.RadianceDownscale32bit().downscale(img, 0.25, method, antialiasing=aa, use_gpu=False,
                                                input_color_space="Linear")[0]


def test_antialiasing_now_changes_the_result():
    img = _checker()
    a0, a5, a1 = (_down(img, v) for v in (0.0, 0.5, 1.0))
    assert not torch.allclose(a0, a5, atol=1e-3) and not torch.allclose(a5, a1, atol=1e-3)


def test_antialiasing_suppresses_moire():
    img = _checker()
    # interior only: the replicated border is its own story
    spread = [float(_down(img, v)[:, :, 3:-3].std()) for v in (0.0, 0.5, 1.0)]
    assert spread[0] > 0.05                          # no prefilter: the stripes alias
    assert spread[1] < 0.01 and spread[2] < 0.01     # prefiltered: they average to grey


def test_default_antialiasing_is_the_textbook_filter():
    img = _img(64, 64)
    new = _down(img, 0.5, "catrom")[0].numpy()
    ref = U.separable_resize_32bit(img[0].numpy(), 16, 16, "catrom", aa_width=1.0)
    np.testing.assert_allclose(new, ref, atol=1e-6)


def test_downscale_keeps_alpha_out_of_the_srgb_curve():
    img = torch.cat([torch.full((1, 16, 16, 3), 0.5), torch.full((1, 16, 16, 1), 0.3)], dim=-1)
    out = U.RadianceDownscale32bit().downscale(img, 0.5, "lanczos", use_gpu=False)[0]
    assert torch.allclose(out[..., 3], torch.full((1, 8, 8), 0.3), atol=1e-5)


# ── Auto colour space ──────────────────────────────────────────────────────

def test_auto_detects_sdr_and_hdr():
    assert U.resolve_input_space(torch.rand(1, 4, 4, 3), "Auto") == "sRGB"
    assert U.resolve_input_space(torch.rand(1, 4, 4, 3) * 5, "Auto") == "Linear"
    assert U.resolve_input_space(torch.rand(1, 4, 4, 3) - 0.5, "Auto") == "Linear"
    assert U.resolve_input_space(torch.rand(1, 4, 4, 3) * 5, "sRGB") == "sRGB"


def test_auto_on_an_sdr_image_resamples_in_linear_light():
    img = _img()
    kw = dict(method="lanczos", sharpening=0, detail_enhancement=0, antialiasing=0)
    auto = U.RadianceProUpscale().upscale(img, 2.0, "Custom", input_color_space="Auto", **kw)
    srgb = U.RadianceProUpscale().upscale(img, 2.0, "Custom", input_color_space="sRGB", **kw)
    assert torch.allclose(auto[0], srgb[0]) and "Input: sRGB (Auto)" in auto[3]


def test_large_frames_take_the_gather_path_with_the_same_result(monkeypatch):
    img = _img(20, 30)
    dense = U.torch_resize_32bit(img, 41, 63, "lanczos4")
    monkeypatch.setattr(U, "_DENSE_LIMIT", 0)
    gathered = U.torch_resize_32bit(img, 41, 63, "lanczos4")
    assert torch.allclose(dense, gathered, atol=1e-5)
