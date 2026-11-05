"""RUDRA 1.1.0's stages in Radiance's RUDRA ACES 2 path.

* the highlight ceiling (nothing brighter than the view's inverse of white),
  ported from RUDRA's tests/test_aces_ceiling.py;
* the user's highlight / shadow bands (tau_high / tau_low) and Clean shadows,
  off by default;
* the inverse-limit statistic, kept apart from the reconstruction's errors;
* the node options and report lines.
"""
import importlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ocio = pytest.importorskip("PyOpenColorIO")
pytest.importorskip("cv2")

from tests.test_rudra_aces2 import _img, fake_package  # noqa: E402,F401  (fixture)


def _mod():
    return importlib.import_module("radiance.model.rudra_aces2")


def _uu():
    return importlib.import_module("radiance.nodes.hdr.uplift_universal")


# ── the ceiling ──────────────────────────────────────────────────────────────

def test_cap_scales_only_what_is_over_and_keeps_hue():
    m = _mod()
    rgb = np.array([[[1.0, 1.0, 1.0], [400.0, 900.0, 1000.0], [-2.0, 0.5, 0.1]]], np.float32)
    out, n = m.cap_luminance(rgb, 128.0)
    assert n == 1
    y = out.astype(np.float64) @ m.LUMA_AP1
    assert y[0, 1] == pytest.approx(128.0, rel=1e-5)
    assert np.allclose(out[0, 1] / out[0, 1, 2], rgb[0, 1] / rgb[0, 1, 2])
    assert np.array_equal(out[0, 0], rgb[0, 0]) and np.array_equal(out[0, 2], rgb[0, 2])
    assert np.array_equal(m.cap_luminance(rgb, 1e9)[0], rgb)


def test_inverse_white_luma_is_the_views_white():
    m = _mod()
    assert m.inverse_white_luma(m.aces2_view()) == pytest.approx(128.0, rel=0.02)


def test_the_master_never_passes_the_ceiling(fake_package):  # noqa: F811
    m = _mod()
    stats = []
    acescg, *_ = m.run_batch(fake_package, _img().numpy(), stats=stats)
    limit = m.inverse_white_luma(m.aces2_view())
    assert float((acescg.astype(np.float64) @ m.LUMA_AP1).max()) <= limit * (1 + 1e-5)
    assert "ceiling_pixels" in stats[0]


# ── bands ────────────────────────────────────────────────────────────────────

def test_band_weight_is_one_in_the_bands_and_zero_between():
    m = _mod()
    codes = np.array([0, 5, 6, 7, 8, 128, 251, 252, 253, 254, 255], np.float32) / 255
    w = m.user_band_weight(codes, 254, 5)
    assert w[0] == w[1] == 1.0 and w[-1] == w[-2] == 1.0
    assert w[4] == w[5] == w[6] == 0.0
    assert 0.0 < w[2] < 1.0 and 0.0 < w[8] < 1.0
    assert not m.user_band_weight(codes, 256, -1).any()          # both bands off


def test_between_the_bands_is_the_exact_plain_inverse():
    m = _mod()
    src = np.array([[[0.5, 0.5, 0.5], [1.0, 1.0, 1.0], [0.0, 0.0, 0.01]]], np.float32)
    base = np.full_like(src, 2.0)
    pred = np.full_like(src, 7.0)
    out, w, stats = m.apply_user_band(pred, base, src, 254, 5)
    assert np.array_equal(out[0, 0], base[0, 0])
    assert np.array_equal(out[0, 1], pred[0, 1]) and np.array_equal(out[0, 2], pred[0, 2])
    assert stats == {"learned_pixels": 2, "ramp_pixels": 0, "analytic_pixels": 1}


def test_bands_shrink_the_reconstructed_mask(fake_package):  # noqa: F811
    m = _mod()
    codes = _img().numpy()
    _, full, *_ = m.run_batch(fake_package, codes)
    stats = []
    _, banded, *_ = m.run_batch(fake_package, codes, band={"tau_high": 256, "tau_low": -1},
                                stats=stats)
    assert full.max() > 0 and banded.max() == 0
    assert stats[0]["band"]["learned_pixels"] == 0


# ── Clean shadows ────────────────────────────────────────────────────────────

def test_clean_shadows_smooths_only_the_deep_shadows():
    m = _mod()
    rng = np.random.default_rng(0)
    h, w = 24, 24
    acescg = np.full((h, w, 3), 0.002, np.float32) * (1 + 0.08 * rng.standard_normal((h, w, 1))).astype(np.float32)
    acescg[:, 12:] = 0.5
    src = np.zeros((h, w, 3), np.float32)
    src[:, :12] = 20 / 255
    src[:, 12:] = 180 / 255
    out, n = m.clean_shadows(acescg, src)
    assert n == h * 12
    assert np.array_equal(out[:, 12:], acescg[:, 12:])            # bright side untouched
    assert out[2:-2, 2:9].std() < acescg[2:-2, 2:9].std() * 0.7     # noise reduced
    assert out[:, :12].max() < 0.01                                 # the edge does not bleed in


def test_clean_shadows_is_a_no_op_without_shadows():
    m = _mod()
    x = np.full((4, 4, 3), 0.3, np.float32)
    out, n = m.clean_shadows(x, np.full((4, 4, 3), 0.5, np.float32))
    assert n == 0 and out is x


# ── the inverse limit is the view's, not the model's ────────────────────────

class _View:
    """forward = identity, except the first pixel's code always comes back off."""

    def forward(self, rgb):
        out = np.asarray(rgb, np.float32).reshape(-1, 3).copy()
        out[..., 0] = np.where(out[..., 0] > 0.99, out[..., 0] - 5 / 255, out[..., 0])
        return out.reshape(np.shape(rgb))


def test_failures_the_baseline_shares_are_counted_as_the_inverse_limit():
    m = _mod()
    src = np.array([[[1.0, 1.0, 1.0], [0.8, 0.8, 0.8]]], np.float32)
    base = src.copy()
    pred = src.copy()
    pred[0, 1] += 0.2                                  # off by 51 codes; the baseline is not
    _, stats = m.constrain_colour(src, pred, base, np.ones((1, 2), np.float32), _View())
    assert stats["inverse_limit_pixels"] == 1          # pixel 0: the "view" cannot hold it
    assert stats["remaining_display_errors"] == 0      # pixel 1 was pulled back


# ── node options and report ──────────────────────────────────────────────────

def test_options_are_off_by_default():
    uu = _uu()
    assert uu._aces2_options("Off", 254, 5, False) == (None, None)
    band, clean = uu._aces2_options("Highlights and shadows", 999, -9, True)
    assert band == {"tau_high": 256, "tau_low": -1}
    assert clean["max_code"] == 40


def test_new_inputs_are_appended_last_on_both_nodes():
    uu = _uu()
    new = ["aces2_bands", "aces2_tau_high", "aces2_tau_low", "aces2_clean_shadows"]
    for cls in (uu.RadianceSDRToHDRUniversal, uu.RadianceSDRToHDRRecover):
        opt = list(cls.INPUT_TYPES()["optional"])
        assert opt[-4:] == new
        assert cls.INPUT_TYPES()["optional"]["aces2_bands"][1]["default"] == "Off"
        assert cls.INPUT_TYPES()["optional"]["aces2_clean_shadows"][1]["default"] is False


def test_universal_reports_the_stage_statistics(fake_package):  # noqa: F811
    uu = _uu()
    out = uu.RadianceSDRToHDRUniversal().convert(
        _img(), "sRGB", 1000.0, "adaptive", 0.75, 2.0, 0.85, "Linear",
        learned_backend="RUDRA ACES 2", aces2_bands="Highlights and shadows",
        aces2_clean_shadows=True)
    report = out[5]
    for needle in ("highlight ceiling:", "display errors left:", "view's own limit", "bands:",
                   "clean shadows:", "RUDRA ACES 2 preview"):
        assert needle in report, needle


def test_auto_never_downloads_or_runs_the_rudra_3x_models(monkeypatch):
    """Recover on Auto without RUDRA ACES 2 stops and says how to choose the
    RUDRA 3.x models; it does not run (or download) them."""
    uu = _uu()
    import radiance.pixel_sdr2hdr as px
    called = []
    monkeypatch.setattr(uu, "_aces2_available", lambda p="": False)
    monkeypatch.setattr(px, "resolve_pixel_checkpoint", lambda p="": called.append("resolve"))
    monkeypatch.setattr(px, "predict_pixel_sdr2hdr", lambda *a, **k: called.append("run"))
    with pytest.raises(RuntimeError, match="Pixel / Temporal"):
        uu.RadianceSDRToHDRRecover().recover(_img(), "sRGB", 1000.0, 0.98, 0.05, 1.0, 1.0, "Linear")
    assert called == []
