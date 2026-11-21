"""Backend truth: a label, a report or a success must say what actually ran.

Phase 3 of the v4 beta audit:
  * a SeedVR2 request that ran SD x4 is reported as SD x4;
  * the video samplers stop on NaN/Inf latents like the image sampler does;
  * a reference image that cannot be applied is an error, not plain text-to-video;
  * an unrecognised model family is logged where it can be seen.
The text-encoder-not-found case is in tests/test_loader_utils.py.
"""
import logging
import re
import types
from pathlib import Path

import pytest

# The pure-Python tests below need no tensors. The ones that do carry the
# real_torch marker, which the light lane skips.
RADIANCE_TORCH_GATED = True

ROOT = Path(__file__).resolve().parent.parent


# ── upscale: which backend ran ───────────────────────────────────────────────

def _up():
    import radiance.nodes.upscale.upscale as up
    return up


def test_seedvr2_request_that_ran_sd_x4_says_so(monkeypatch):
    up = _up()
    monkeypatch.setattr(up, "_load_seedvr2_pipeline",
                        lambda device: (_ for _ in ()).throw(RuntimeError("no seedvr2 module")))
    monkeypatch.setattr(up, "_load_sd_x4_pipeline", lambda device: "sd-pipe")
    monkeypatch.setattr(up, "_sd_x4_infer", lambda pipe, tile, **kw: "sd-result")
    used = {}
    out = up._diffusion_upscale_infer("tile", "cpu", prefer_seedvr2=True, backend_used=used)
    assert out == "sd-result"
    assert used == {"name": "SD x4"}


def test_seedvr2_that_works_is_recorded_as_seedvr2(monkeypatch):
    up = _up()
    monkeypatch.setattr(up, "_load_seedvr2_pipeline", lambda device: "sv-pipe")
    monkeypatch.setattr(up, "_seedvr2_infer", lambda pipe, tile, **kw: "sv-result")
    used = {}
    assert up._diffusion_upscale_infer("tile", "cpu", prefer_seedvr2=True, backend_used=used) == "sv-result"
    assert used == {"name": "SeedVR2"}


def test_the_backend_argument_is_optional(monkeypatch):
    up = _up()
    monkeypatch.setattr(up, "_load_sd_x4_pipeline", lambda device: "p")
    monkeypatch.setattr(up, "_sd_x4_infer", lambda pipe, tile, **kw: "r")
    assert up._diffusion_upscale_infer("tile", "cpu") == "r"


def _report(**stats):
    base = {"diffusion": 0, up_name(): 0, "Real-ESRGAN": 0, "bicubic": 0}
    base.update(stats)
    return types.SimpleNamespace(backend_stats=base)


def up_name():
    return _up()._SD_X4_INSTEAD


def test_report_says_seedvr2_was_not_used_when_every_tile_ran_sd_x4():
    up = _up()
    text = up._backend_report(_report(**{up_name(): 6}), "SeedVR2 (diffusion)")
    assert "NOT used" in text and "SeedVR2 unavailable" in text and "SD x4" in text


def test_report_flags_a_mixed_run():
    up = _up()
    text = up._backend_report(_report(diffusion=2, **{up_name(): 4}), "SeedVR2 (diffusion)")
    assert "some tiles only" in text and "4/6" in text


def test_report_keeps_the_label_when_every_tile_ran_the_requested_backend():
    up = _up()
    assert up._backend_report(_report(diffusion=6), "SeedVR2 (diffusion)") == "SeedVR2 (diffusion)"


def test_the_no_model_case_keeps_its_wording():
    up = _up()
    text = up._backend_report(_report(bicubic=3), "SD x4 upscaler (diffusion)")
    assert "NOT used" in text and "diffusion unavailable" in text and "bicubic" in text


@pytest.mark.real_torch
def test_the_tier3_callable_counts_sd_x4_tiles_separately(monkeypatch):
    torch = pytest.importorskip("torch")
    up = _up()

    def backend(tile, device, **kw):
        kw["backend_used"]["name"] = "SD x4"
        x = tile.permute(0, 3, 1, 2)
        return torch.nn.functional.interpolate(x, scale_factor=4, mode="nearest").permute(0, 2, 3, 1)

    monkeypatch.setattr(up, "_diffusion_upscale_infer", backend)
    fn, label = up._build_upscale_fn("tier3_seedvr2", 4, torch.device("cpu"))
    fn(torch.rand(1, 8, 8, 3))
    assert fn.backend_stats[up._SD_X4_INSTEAD] == 1 and fn.backend_stats["diffusion"] == 0
    assert "NOT used" in up._backend_report(fn, label)


# ── video samplers stop on a diverged latent ─────────────────────────────────

@pytest.mark.real_torch
def test_the_shared_finite_check_names_the_node_that_failed():
    torch = pytest.importorskip("torch")
    from radiance.nodes.generate.sampler import assert_latent_finite
    bad = torch.zeros(1, 4, 2, 2)
    bad[0, 0, 0, 0] = float("nan")
    with pytest.raises(RuntimeError, match=r"\[RadianceVideoSampler\]"):
        assert_latent_finite(bad, node_name="RadianceVideoSampler")
    with pytest.raises(RuntimeError, match=r"\[RadianceSamplerPro\]"):
        assert_latent_finite(bad)


def test_the_video_sampler_uses_the_shared_check_and_no_longer_sanitises():
    src = (ROOT / "nodes" / "video" / "t2v.py").read_text(encoding="utf-8")
    assert 'assert_latent_finite(samples, node_name="RadianceVideoSampler")' in src
    start = src.index("def _comfy_sample")
    end = src.index("def _make_noise")
    assert "nan_to_num" not in src[start:end]


# ── I2V: a reference image that cannot be applied is an error ────────────────

@pytest.mark.real_torch
@pytest.mark.parametrize("method", ["_first_frame_lock", "_prepend_latent"])
def test_a_reference_image_that_cannot_be_applied_raises(method):
    torch = pytest.importorskip("torch")
    from radiance.nodes.video.t2v import RadianceI2VPipeline
    node = RadianceI2VPipeline()
    noise = torch.zeros(1, 16, 2, 4, 4)
    broken = torch.zeros(3)            # not a (B, C, H, W) latent
    args = (noise, broken, 0.5) if method == "_prepend_latent" else (noise, broken, 0.5, noise.shape, 16)
    with pytest.raises(RuntimeError, match="could not apply the reference image"):
        getattr(node, method)(*args)


def test_the_i2v_report_calls_the_heuristic_strategies_heuristic():
    src = (ROOT / "nodes" / "video" / "t2v.py").read_text(encoding="utf-8")
    assert "HEURISTIC I2V" in src
    assert "frame 0 is not guaranteed to match the reference image" in src
    # The class docstring no longer promises per-model APIs that do not exist.
    assert "injected via model's img2vid API" not in src


# ── an unrecognised model family is logged ───────────────────────────────────

def _model(config_cls_name):
    cfg = type(config_cls_name, (), {})()
    return types.SimpleNamespace(model=types.SimpleNamespace(model_config=cfg))


def test_an_unrecognised_family_warns_before_using_sd15_defaults(caplog):
    from radiance.sampler_utils import RadianceModelRegistry
    with caplog.at_level(logging.WARNING):
        assert RadianceModelRegistry.detect(_model("BrandNewFamily")) == "sd1.5"
    assert any("not recognised" in r.getMessage() and "BrandNewFamily" in r.getMessage()
               for r in caplog.records)


@pytest.mark.parametrize("name", ["SD15", "SDXL", "SD20", "SSD1B", "Segmind_Vega"])
def test_a_genuine_sd_family_does_not_warn(caplog, name):
    from radiance.sampler_utils import RadianceModelRegistry
    with caplog.at_level(logging.WARNING):
        assert RadianceModelRegistry.detect(_model(name)) == "sd1.5"
    assert not any("not recognised" in r.getMessage() for r in caplog.records)
