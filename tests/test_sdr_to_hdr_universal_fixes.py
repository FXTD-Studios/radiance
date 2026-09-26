"""Regression tests for the SDR → HDR Universal / Recover bug fixes.

Each test corresponds to a defect that the existing 36-test suite for this node
did not catch, mostly because it probed only the mid-range of the curve or only
the explicit-path configuration.
"""
import pathlib

import pytest

torch = pytest.importorskip("torch")

import radiance.pixel_sdr2hdr as px  # noqa: E402
from radiance.nodes.hdr import uplift_universal as mod  # noqa: E402


@pytest.fixture
def node():
    return mod.RadianceSDRToHDRUniversal()


BASE_KW = dict(
    inverse_oetf="None", peak_nits=1000.0, knee_mode="manual", knee=0.5,
    shoulder_gamma=1.6, temporal_smoothing=0.0, output_encoding="Linear",
)


@pytest.fixture
def fake_rudra(monkeypatch):
    """Mock the pixel model so the blend path runs with real math.

    `rec_value` is in Radiance's working space (scene-linear Rec.709, 1.0 ==
    100 nits); the model contract is Rec.2020 with 1.0 == 10,000 nits, so the
    fake returns rec_value / 100 and the node's own conversion brings it back.
    """
    def _install(rec_value=3.0):
        monkeypatch.setattr(px, "resolve_pixel_checkpoint", lambda p="": pathlib.Path("/tmp/x.pt"))
        # Grey in Rec.2020 is grey in Rec.709, so the matrix is an identity here.
        monkeypatch.setattr(px, "predict_pixel_sdr2hdr",
                            lambda srgb, **k: torch.full_like(srgb[..., :3], rec_value / 100.0))
    return _install


# ── 1. Adaptive knee must survive film resolutions ──────────────────────────
#
# torch.quantile refuses inputs above 2**24 elements along the reduced axis.
# A single 8K frame is 33.2M pixels, so "adaptive" -- the default knee_mode --
# raised RuntimeError on anything past roughly 4K.

@pytest.mark.parametrize("h,w,label", [
    (270, 480, "SD"),
    (1080, 1920, "1080p"),
    (2160, 3840, "4K"),
    (4320, 7680, "8K"),
])
def test_adaptive_knee_survives_large_frames(h, w, label):
    luma = torch.rand(1, h, w)
    knees = mod._adaptive_knees(luma, 0.75, 0.0)
    assert knees.shape == (1,)
    assert 0.05 <= float(knees[0]) <= 0.99, f"{label}: implausible knee {float(knees[0])}"


def test_row_quantile_matches_torch_quantile_below_the_cap():
    """The kthvalue fallback must agree with quantile where both are usable."""
    x = torch.rand(4, 200_000)
    for p in (0.05, 0.5, 0.75, 0.99):
        ref = torch.quantile(x, p, dim=1)
        got = mod._row_quantile(x, p)
        assert torch.allclose(ref, got, atol=1e-4), f"p={p} diverged"


def test_row_quantile_handles_oversized_rows():
    """Above the cap the fallback must return a plausible value, not raise."""
    n = mod._QUANTILE_MAX_ELEMS + 1000
    x = torch.linspace(0.0, 1.0, n).unsqueeze(0)
    got = float(mod._row_quantile(x, 0.75)[0])
    assert abs(got - 0.75) < 1e-3, f"expected ~0.75 on a uniform ramp, got {got}"


# ── 2. The direct-pixel backend must be reachable at stock defaults ─────────
#
# Auto used to gate on `bool(pixel_checkpoint.strip())`, so an installed
# checkpoint with the path widget left blank -- the default -- was never used.

@pytest.mark.parametrize("backend,ckpt", [
    ("Auto", ""),             # the stock defaults
    ("Auto", "/tmp/x.pt"),
    ("Direct Pixel", ""),
])
def test_direct_pixel_runs_when_checkpoint_is_discoverable(node, monkeypatch, backend, ckpt):
    calls = {"n": 0}

    def fake_predict(srgb, **kw):
        calls["n"] += 1
        return torch.full_like(srgb, 0.05)

    monkeypatch.setattr(px, "resolve_pixel_checkpoint", lambda p="": pathlib.Path("/tmp/x.pt"))
    monkeypatch.setattr(px, "predict_pixel_sdr2hdr", fake_predict)

    img = torch.linspace(0, 1, 64).reshape(1, 8, 8, 1).expand(1, 8, 8, 3).contiguous()
    node.convert(image=img, learned_backend=backend, pixel_checkpoint=ckpt,
                 rudra_blend=1.0, **BASE_KW)
    assert calls["n"] == 1, f"direct-pixel backend not used for {backend!r}/{ckpt!r}"


def test_direct_pixel_skipped_when_no_checkpoint_exists(node, monkeypatch):
    """The gate must still be a gate -- no checkpoint, no attempt."""
    calls = {"n": 0}

    def fake_predict(srgb, **kw):
        calls["n"] += 1
        return torch.full_like(srgb, 0.05)

    monkeypatch.setattr(px, "resolve_pixel_checkpoint", lambda p="": None)
    monkeypatch.setattr(px, "predict_pixel_sdr2hdr", fake_predict)

    img = torch.linspace(0, 1, 64).reshape(1, 8, 8, 1).expand(1, 8, 8, 3).contiguous()
    node.convert(image=img, learned_backend="Auto", pixel_checkpoint="",
                 rudra_blend=1.0, **BASE_KW)
    assert calls["n"] == 0


def test_checkpoint_probe_is_not_fooled_by_a_missing_module(monkeypatch):
    """A broken optional import must read as 'unavailable', not crash."""
    import builtins
    real_import = builtins.__import__

    def boom(name, *a, **k):
        if name == "radiance.pixel_sdr2hdr":
            raise ImportError("simulated")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", boom)
    assert mod._pixel_checkpoint_available("") is False


# ── 3. Pixels outside the recovery mask must be preserved exactly ───────────
#
# Both node docstrings promise this. The peak limiter was applied to the blended
# result as well as to the learned signal, so any pixel the deterministic path
# had expanded above 0.9*peak was quietly compressed even where the mask was 0.

def test_pixels_outside_recovery_mask_are_bit_exact(node, fake_rudra):
    fake_rudra(rec_value=3.0)
    # Codes chosen to land above the peak limiter's 0.9*peak knee but below the
    # 0.98 clipping threshold -- i.e. genuinely outside the recovery mask.
    vals = torch.tensor([0.90, 0.95, 0.9664, 0.970, 0.975, 0.979])
    img = vals.view(1, 1, -1, 1).expand(1, 1, 6, 3).contiguous()
    kw = dict(BASE_KW, highlight_threshold=0.98, shadow_threshold=0.001)

    base, _, _, _, _, _ = node.convert(image=img, rudra_blend=0.0, **kw)
    out, _, _, _, _, _ = node.convert(image=img, rudra_blend=1.0, pixel_recovery_mode="all", **kw)

    torch.testing.assert_close(out, base, atol=0.0, rtol=0.0)


def test_peak_is_still_enforced_after_removing_the_second_limiter(node, fake_rudra):
    """Dropping the outer limiter must not let learned radiance exceed peak."""
    fake_rudra(rec_value=1000.0)
    img = torch.linspace(0, 1, 64).reshape(1, 8, 8, 1).expand(1, 8, 8, 3).contiguous()
    out, _, _, _, _, report = node.convert(
        image=img, rudra_blend=1.0, pixel_recovery_mode="all",
        **dict(BASE_KW, peak_nits=200.0, shoulder_gamma=1.0))
    assert "learned recovery: applied" in report
    assert float(mod._luma(out).max()) <= 2.0 + 1e-5


def test_soft_peak_limit_left_alone_is_still_bounded():
    """The limiter itself is unchanged; only how often it runs."""
    peak = 10.0
    rgb = torch.tensor([0.5, 5.0, 9.5, 12.0, 500.0]).view(1, 1, -1, 1).repeat(1, 1, 1, 3)
    out = mod._soft_peak_limit(rgb, peak)
    assert float(mod._luma(out).max()) <= peak + 1e-5
    # below the knee it must be an identity
    torch.testing.assert_close(mod._luma(out)[0, 0, :2],
                               mod._luma(rgb)[0, 0, :2], atol=1e-6, rtol=0)


# ── 4. Empty batches ────────────────────────────────────────────────────────

def test_empty_batch_returns_empty_rather_than_raising(node):
    out, mask, shadows, h_conf, s_conf, _ = node.convert(
        image=torch.zeros(0, 4, 4, 3), **dict(BASE_KW, knee_mode="adaptive", knee=0.75))
    assert out.shape[0] == 0
    for m in (mask, shadows, h_conf, s_conf):
        assert m.shape == (0, 4, 4)


def test_recover_empty_batch_returns_empty():
    rec = mod.RadianceSDRToHDRRecover()
    out, *masks = rec.recover(
        image=torch.zeros(0, 4, 4, 3), inverse_oetf="None", peak_nits=1000.0,
        highlight_threshold=0.98, shadow_threshold=0.05, highlight_strength=1.0,
        shadow_strength=1.0, output_encoding="Linear")
    assert out.shape[0] == 0


# ── 5. Invariants that must not regress ─────────────────────────────────────

def test_expand_is_monotonic_and_hits_reference_white(node):
    """SDR code 1.0 lands on reference white, not on the display peak.

    This asserted 10.0 — peak_nits/100, i.e. a white shirt at 1000 nits. From
    3.4 the expansion targets `reference_white_nits` (ITU-R BT.2408: 203), and
    the range above it is headroom for recovered speculars rather than
    somewhere to put diffuse white.
    """
    ramp = torch.linspace(0, 1, 256).view(1, 1, 256, 1).repeat(1, 1, 1, 3)
    out, _, _, _, _, _ = node.convert(image=ramp, processing_mode="Expand", **BASE_KW)
    y = mod._luma(out)[0, 0]
    assert bool((torch.diff(y) >= -1e-6).all()), "expansion is not monotonic"
    # 3.5.0: Linear is BT.2408-normalised, so reference white is exactly 1.0.
    assert abs(float(y.max()) - 1.0) < 1e-4, "reference white not reached"
    assert float(y[0]) == pytest.approx(0.0, abs=1e-7)


@pytest.mark.parametrize("peak", [200.0, 1000.0, 4000.0, 10000.0])
def test_peak_nits_is_the_ceiling_not_the_target(node, peak):
    """Raising the mastering peak must not move diffuse white.

    It used to: this asserted that SDR white came out at exactly peak_nits, so
    switching a 1000-nit master to 4000 made every white surface four times
    brighter. peak_nits is the ceiling and the encode target; where SDR white
    sits is `reference_white_nits`.
    """
    out, _, _, _, _, _ = node.convert(image=torch.ones(1, 4, 4, 3),
                                   **dict(BASE_KW, peak_nits=peak, processing_mode="Expand"))
    # 3.5.0: output unit is the effective reference white (capped at peak).
    ref = min(203.0, peak)
    y_nits = float(mod._luma(out).max()) * ref
    assert y_nits == pytest.approx(min(203.0, peak), rel=1e-4)
    assert y_nits <= peak + 1e-3, "output exceeded the mastering peak"


@pytest.mark.parametrize("peak", [1000.0, 4000.0, 10000.0])
def test_the_old_behaviour_is_still_reachable(node, peak):
    out, _, _, _, _, _ = node.convert(image=torch.ones(1, 4, 4, 3),
                                   **dict(BASE_KW, peak_nits=peak, processing_mode="Expand",
                                          reference_white_nits=peak))
    # reference white == peak: SDR white at the peak, which is linear 1.0.
    assert float(mod._luma(out).max()) == pytest.approx(1.0, rel=1e-4)


def test_alpha_survives_every_output_encoding(node):
    rgba = torch.cat([torch.full((1, 4, 4, 3), 0.5),
                      torch.full((1, 4, 4, 1), 0.25)], dim=-1)
    for enc in ("Linear", "Linear ACES2065-1 (AP0)", "PQ (HDR10)", "HLG"):
        out, _, _, _, _, _ = node.convert(image=rgba, **dict(BASE_KW, output_encoding=enc))
        assert out.shape[-1] == 4, enc
        assert float(out[0, 0, 0, 3]) == pytest.approx(0.25), enc
