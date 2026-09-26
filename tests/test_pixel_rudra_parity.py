"""RUDRA pixel model: checkpoint loading, per-frame gating, hue and peak safety.

Regression cover for a user report on a clean Flux.2 sunset (Hybrid, v5):
false-colour rings at each channel's clip boundary, white areas cast red
(R about 2.2x G), 4x chroma noise on glints, and single channels at 3,500 nits
under a 1,000-nit peak.
"""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.real_torch

import radiance.pixel_sdr2hdr as px  # noqa: E402
import radiance.nodes.hdr.uplift_universal as uu  # noqa: E402


def _save_safetensors(path, model, config):
    from safetensors.torch import save_file
    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(state, str(path), metadata={"config": json.dumps(config)})


def _randomise(model):
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.05)
    return model


# ── loading ─────────────────────────────────────────────────────────────────

def test_shadow_gated_safetensors_loads_and_matches(tmp_path):
    ref = _randomise(px.SDR2HDRNet(base_channels=8, shadow_conditioning=True)).eval()
    path = tmp_path / "sdr2hdr_shadow_v1.safetensors"
    _save_safetensors(path, ref, {"base_channels": 8, "shadow_conditioning": True})
    model = px.build_pixel_model(*px.read_pixel_checkpoint(path))
    assert model.shadow_gate is not None
    x = torch.rand(1, 3, 32, 48)
    with torch.no_grad():
        a = ref(x, shadow_weight=ref.predict_shadow_weight(x)).hdr
        b = px._predict_frame(model, x, 0, 0, "all", 1.0)
    torch.testing.assert_close(a, b)


def test_state_dict_decides_heads_the_config_forgot(tmp_path):
    ref = _randomise(px.SDR2HDRNet(base_channels=8, shadow_conditioning=True)).eval()
    path = tmp_path / "old_config.safetensors"
    _save_safetensors(path, ref, {"steps": 3000})       # no base_channels, no flag
    model = px.build_pixel_model(*px.read_pixel_checkpoint(path))
    assert model.shadow_gate is not None and model.stem.out_channels == 8


def test_legacy_pt_still_loads(tmp_path):
    ref = _randomise(px.SDR2HDRNet(base_channels=8)).eval()
    path = tmp_path / "sdr2hdr_pixel_image.pt"
    torch.save({"model": ref.state_dict(), "config": {"base_channels": 8}}, path)
    model = px.build_pixel_model(*px.read_pixel_checkpoint(path))
    assert model.shadow_gate is None


def test_temporal_refiner_is_refused_by_name():
    state = px.TemporalHDRRefiner().state_dict()
    with pytest.raises(ValueError, match="temporal refiner"):
        px.build_pixel_model(state, {})


def test_hugging_face_names_are_searched_first():
    pats = px.PIXEL_CHECKPOINT_PATTERNS
    assert pats[0] == "sdr2hdr_shadow_v1.safetensors"
    assert "sdr2hdr/sdr2hdr_shadow_v1.safetensors" in pats
    assert not any("temporal" in p for p in pats)


# ── per-frame decisions and tiling ──────────────────────────────────────────

def test_frame_gates_run_once_not_per_tile(monkeypatch):
    model = _randomise(px.SDR2HDRNet(base_channels=8, shadow_conditioning=True)).eval()
    calls = {"n": 0}
    real = model.shadow_gate.forward

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)
    monkeypatch.setattr(model.shadow_gate, "forward", counting)
    monkeypatch.setattr(px, "_whole_frame_fits", lambda frame: False)
    px._predict_frame(model, torch.rand(1, 3, 256, 256), 128, 32, "all", 1.0)
    assert calls["n"] == 1


def test_whole_frame_is_preferred_when_it_fits():
    model = _randomise(px.SDR2HDRNet(base_channels=8)).eval()
    x = torch.rand(1, 3, 256, 320)
    whole = px._predict_frame(model, x, 0, 0, "all", 1.0)
    asked_for_tiles = px._predict_frame(model, x, 128, 32, "all", 1.0)
    torch.testing.assert_close(whole, asked_for_tiles)


# ── peak limiter bounds every channel ───────────────────────────────────────

def test_no_channel_escapes_the_peak():
    rgb = torch.tensor([[[[40.0, 3.0, 1.0], [5.0, 60.0, 2.0], [0.5, 0.5, 90.0]]]])
    out = uu._soft_peak_limit(rgb, 10.0)
    assert float(out.max()) <= 10.0 + 1e-5
    # hue kept: channel ratios unchanged
    torch.testing.assert_close(out[..., 0] / out[..., 1], rgb[..., 0] / rgb[..., 1])


# ── hue: brightness from the model, colour from the source ──────────────────

@pytest.fixture
def fake_model(monkeypatch):
    """A 'model' that invents red wherever it runs, like v5 on clipped input."""
    def predict(srgb, **kw):
        out = torch.empty_like(srgb[..., :3])
        out[..., 0], out[..., 1], out[..., 2] = 0.04, 0.02, 0.015   # 400/200/150 nits
        return out
    monkeypatch.setattr(px, "predict_pixel_sdr2hdr", predict)


def _run(sdr_code, fake_mask=1.0):
    lin = uu._inverse_oetf(sdr_code, "sRGB")
    mask = torch.full(lin.shape[:-1], fake_mask)
    return lin, uu._RudraRecoveryCore._pixel_reconstruct(
        lin, lin, mask, "", 1.0, 10.0, 0, 0, "highlights", 1.0, highlight_mask=mask)


def test_clipped_white_stays_white(fake_model):
    _, out = _run(torch.ones(1, 2, 2, 3))
    r, g, b = out[0, 0, 0]
    assert float(r / g) == pytest.approx(1.0, abs=1e-5)
    assert float(b / g) == pytest.approx(1.0, abs=1e-5)
    assert float(uu._luma(out).mean()) > 1.5           # it was still lifted


def test_one_clipped_channel_gets_no_invented_lift(fake_model):
    lin, out = _run(torch.tensor([1.0, 0.70, 0.55]).view(1, 1, 1, 3))
    torch.testing.assert_close(out, lin)                 # median code 0.70 < 0.85


def test_learned_lift_never_darkens(monkeypatch):
    monkeypatch.setattr(px, "predict_pixel_sdr2hdr",
                        lambda srgb, **k: torch.zeros_like(srgb[..., :3]))
    lin, out = _run(torch.ones(1, 1, 1, 3))
    torch.testing.assert_close(out, lin)


def test_grey_prediction_is_not_resaturated(monkeypatch):
    """No Rec.2020 -> Rec.709 matrix: the network does not change primaries."""
    monkeypatch.setattr(px, "predict_pixel_sdr2hdr",
                        lambda srgb, **k: torch.full_like(srgb[..., :3], 0.02))
    sdr = torch.tensor([0.30, 0.20, 0.10]).view(1, 1, 1, 3)        # shadow pixel
    lin = uu._inverse_oetf(sdr, "sRGB")
    mask = torch.ones(lin.shape[:-1])
    out = uu._RudraRecoveryCore._pixel_reconstruct(
        lin, lin, mask, "", 1.0, 10.0, 0, 0, "shadows", 1.0,
        highlight_mask=torch.zeros_like(mask))
    torch.testing.assert_close(out, torch.full_like(out, 2.0))
