"""v3.5.3 P0 regression tests: upscale geometry and HDR (FIX-011, 012, 013).

Runs offline: with downloads disabled the Tier 1 backend falls back to
bicubic, which exercises the same tiling, cascade and accumulation code.
"""
import pytest

torch = pytest.importorskip("torch")
if not hasattr(torch, "einsum"):
    pytest.skip("real torch required", allow_module_level=True)

from radiance.nodes.upscale.upscale import (
    RadianceUpscaleTiler, RadianceUpscaleImage, RadianceUpscaleVideo,
)

TIER1 = "tier1_fast    (Real-ESRGAN — GAN, ms/frame)"


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setenv("RADIANCE_ALLOW_DOWNLOADS", "0")


# ── FIX-011: 8x image / tile geometry ───────────────────────────────────────

def test_tiler_8x_output_and_confidence_geometry():
    img = torch.rand(2, 12, 10, 3)
    up, conf, info = RadianceUpscaleTiler().tile_upscale(img, scale="8× (tile cascade)", tile_size=8,
                                                        overlap=2, model_tier=TIER1)
    assert up.shape == (2, 96, 80, 3)
    assert conf.shape == (2, 96, 80, 3)
    assert "(8×)" in info


def test_image_8x_output_and_confidence_geometry():
    img = torch.rand(1, 12, 10, 3)
    up, conf, _ = RadianceUpscaleImage().upscale_image(img, scale="8× (tile cascade)", tile_size=8,
                                                       overlap=2, model_tier=TIER1, mode="precise")
    assert up.shape == (1, 96, 80, 3)
    assert conf.shape == (1, 96, 80, 3)
    assert float(conf.min()) >= 0.0 and float(conf.max()) <= 1.0


@pytest.mark.parametrize("scale,factor", [("2×", 2), ("4×", 4)])
def test_image_2x_4x_geometry(scale, factor):
    up, conf, _ = RadianceUpscaleImage().upscale_image(torch.rand(1, 9, 7, 3), scale=scale, tile_size=8,
                                                       overlap=2, model_tier=TIER1, mode="precise")
    assert up.shape == (1, 9 * factor, 7 * factor, 3) and conf.shape[:3] == up.shape[:3]


# ── FIX-012: 8x video accumulators and window alignment ─────────────────────

def test_video_8x_geometry_and_frame_count():
    clip = torch.rand(5, 8, 6, 3)
    up, conf, info = RadianceUpscaleVideo().upscale_video(
        clip, scale="8× (tile cascade)", tile_size=8, overlap_spatial=2,
        window_size=3, overlap_temporal=1, flow_compensation=False, model_tier=TIER1)
    assert up.shape == (5, 64, 48, 3)
    assert conf.shape == (5, 64, 48, 3)
    assert "(8×)" in info


@pytest.mark.parametrize("window,overlap", [(4, 2), (3, 1), (4, 3), (16, 4), (2, 5)])
@pytest.mark.parametrize("flow", [False, True])
def test_video_windows_blend_the_same_source_frame(window, overlap, flow):
    """Every output frame must come from its own source frame. Flat frames with
    distinct values make any cross-frame blend visible."""
    B = 9
    vals = torch.linspace(0.05, 0.95, B)
    clip = vals.view(B, 1, 1, 1).expand(B, 8, 8, 3).contiguous()
    up, _, _ = RadianceUpscaleVideo().upscale_video(
        clip, scale="2×", tile_size=8, overlap_spatial=2, window_size=window,
        overlap_temporal=overlap, flow_compensation=flow, model_tier=TIER1, hdr_mode="clamp")
    assert up.shape == (B, 16, 16, 3)
    means = up.mean(dim=(1, 2, 3))
    torch.testing.assert_close(means, vals, atol=2e-3, rtol=0)


# ── FIX-013: no HDR clipping before upscale preprocessing ───────────────────

def test_tiler_keeps_hdr_highlights():
    img = torch.full((1, 8, 8, 3), 0.5)
    img[:, 2:6, 2:6, :] = 6.0
    up, _, info = RadianceUpscaleTiler().tile_upscale(img, scale="2×", tile_size=8, overlap=2,
                                                      model_tier=TIER1)
    assert float(up.max()) > 4.0
    assert "preserved" in info


def test_image_denoise_does_not_clip_hdr():
    img = torch.full((1, 8, 8, 3), 0.5)
    img[:, 2:6, 2:6, :] = 6.0
    up, _, info = RadianceUpscaleImage().upscale_image(img, scale="2×", tile_size=8, overlap=2,
                                                       model_tier=TIER1, mode="precise",
                                                       denoise_pre=0.5, hdr_mode="auto")
    assert "preserve=True" in info
    assert float(up.max()) > 4.0


def test_video_keeps_hdr_highlights():
    clip = torch.full((3, 8, 8, 3), 0.5)
    clip[:, 3:5, 3:5, :] = 8.0
    up, _, _ = RadianceUpscaleVideo().upscale_video(clip, scale="2×", tile_size=8, overlap_spatial=2,
                                                    window_size=2, overlap_temporal=1,
                                                    flow_compensation=False, model_tier=TIER1)
    assert float(up.max()) > 4.0


def test_upscale_does_not_modify_input():
    img = torch.rand(1, 8, 8, 3) * 3
    ref = img.clone()
    RadianceUpscaleImage().upscale_image(img, scale="2×", tile_size=8, overlap=2,
                                         model_tier=TIER1, mode="precise", denoise_pre=0.3)
    assert torch.equal(img, ref)
