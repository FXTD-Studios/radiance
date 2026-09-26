"""
test_color_math.py — Color science math unit tests.

Covers:
  • sRGB EOTF / OETF roundtrip (numpy + tensor)
  • tensor_linear_to_srgb sign-preservation (regression for BUG-COLORUTILS-1)
  • All log curve encode→decode roundtrips (numpy)
  • GPU tensor log curve encode→decode roundtrips
  • ST.2084 PQ and HLG encode→decode roundtrips
  • ACEScct encode→decode roundtrip
  • Color matrix forward/inverse consistency
  • ACES gamut compress is idempotent inside gamut
"""

import sys
import os
import math
import pytest
import numpy as np

# Add the radiance package root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import color_utils as cu

# Detect whether a *real* torch (not the conftest MagicMock stub) is available.
# The stub has no __version__ string, so this check is False in the stub case.
try:
    import torch as _torch
    HAS_TORCH = isinstance(getattr(_torch, "__version__", None), str)
except ImportError:
    HAS_TORCH = False

_SKIP_NO_TORCH = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

RTOL = 1e-4   # relative tolerance — matches float32 precision
ATOL = 1e-5   # absolute tolerance for near-zero values


def assert_roundtrip(enc, dec, values, rtol=RTOL, atol=ATOL, label=""):
    """Assert enc→dec is identity within tolerance."""
    encoded = enc(values)
    decoded = dec(encoded)
    np.testing.assert_allclose(
        decoded, values, rtol=rtol, atol=atol,
        err_msg=f"{label}: encode→decode roundtrip failed"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  sRGB EOTF / OETF  (NumPy)
# ─────────────────────────────────────────────────────────────────────────────

class TestSRGBNumpy:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    @pytest.fixture
    def sdl_values(self):
        """Linear values covering the full [0, 1] SDR range plus shadow toe."""
        v = np.concatenate([
            np.linspace(0.0, 0.04, 64),
            np.linspace(0.04, 1.0, 512),
        ]).astype(np.float32)
        return v

    def test_srgb_to_linear_midgray(self):
        """sRGB 0.5 linearises to ~0.2140 (per IEC 61966-2-1)."""
        result = cu.srgb_to_linear(np.array([0.5], dtype=np.float32))
        np.testing.assert_allclose(result[0], 0.2140, atol=5e-4)

    def test_linear_to_srgb_midgray(self):
        """Linear 0.18 → sRGB ~0.461."""
        result = cu.linear_to_srgb(np.array([0.18], dtype=np.float32))
        np.testing.assert_allclose(result[0], 0.4613, atol=5e-4)

    def test_srgb_roundtrip(self, sdl_values):
        """sRGB → linear → sRGB roundtrip within float32 tolerance."""
        assert_roundtrip(
            cu.srgb_to_linear, cu.linear_to_srgb, sdl_values,
            label="sRGB numpy"
        )

    def test_linear_roundtrip(self, sdl_values):
        """linear → sRGB → linear roundtrip."""
        assert_roundtrip(
            cu.linear_to_srgb, cu.srgb_to_linear, sdl_values,
            label="linear->sRGB numpy"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  sRGB tensor — regression for BUG-COLORUTILS-1 (sign preservation)
# ─────────────────────────────────────────────────────────────────────────────

@_SKIP_NO_TORCH
class TestSRGBTensor:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    """
    BUG-COLORUTILS-1 regression: tensor_linear_to_srgb must preserve sign
    for sub-zero values that are valid in scene-linear HDR pipelines.
    Pre-fix: values < 0 were clamped to 1e-10 inside pow(), destroying sign.
    """

    def test_positive_sdr_roundtrip(self):
        """tensor sRGB ↔ linear roundtrip for standard [0, 1] values."""
        import torch
        v = torch.linspace(0.0, 1.0, 256, dtype=torch.float32)
        lin = cu.tensor_srgb_to_linear(v)
        back = cu.tensor_linear_to_srgb(lin)
        torch.testing.assert_close(back, v, rtol=1e-4, atol=1e-5)

    def test_sub_zero_sign_preserved(self):
        """
        BUG-COLORUTILS-1 regression: negative linear values must encode to
        negative sRGB and decode back without sign flip.
        Pre-fix: tensor_linear_to_srgb clamped to 1e-10 → always positive.
        """
        import torch
        neg = torch.tensor([-0.5, -0.1, -0.01], dtype=torch.float32)
        encoded = cu.tensor_linear_to_srgb(neg)
        assert (encoded < 0).all(), \
            "BUG-COLORUTILS-1 regression: sub-zero linear values encoded as positive"

    def test_sign_roundtrip_hdr(self):
        """Full sign-preserving roundtrip: negative linear → sRGB → linear."""
        import torch
        neg_values = torch.tensor([-2.0, -1.0, -0.5, -0.01], dtype=torch.float32)
        encoded = cu.tensor_linear_to_srgb(neg_values)
        decoded = cu.tensor_srgb_to_linear(encoded)
        torch.testing.assert_close(decoded, neg_values, rtol=1e-4, atol=1e-5)

    def test_hdr_above_one_roundtrip(self):
        """Linear values > 1.0 (HDR) must survive a sRGB OETF→EOTF roundtrip."""
        import torch
        hdr = torch.tensor([1.5, 2.0, 4.0, 10.0], dtype=torch.float32)
        encoded = cu.tensor_linear_to_srgb(hdr)
        decoded = cu.tensor_srgb_to_linear(encoded)
        torch.testing.assert_close(decoded, hdr, rtol=1e-4, atol=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
#  ARRI LogC3 (NumPy)
# ─────────────────────────────────────────────────────────────────────────────

class TestLogC3Numpy:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    PROBE = np.array([0.0, 0.01, 0.18, 0.5, 1.0, 2.0], dtype=np.float32)

    def test_midgray_placement(self):
        """LogC3 (EI 800): 18% gray → ~0.3910."""
        enc = cu.linear_to_logc3(np.array([0.18], dtype=np.float32), ei=800)
        np.testing.assert_allclose(enc[0], 0.391, atol=5e-3)

    def test_roundtrip_ei800(self):
        assert_roundtrip(
            lambda x: cu.linear_to_logc3(x, 800),
            lambda x: cu.logc3_to_linear(x, 800),
            self.PROBE, label="LogC3 EI800"
        )

    @pytest.mark.parametrize("ei", [160, 400, 800, 1600, 3200])
    def test_roundtrip_multi_ei(self, ei):
        vals = np.array([0.01, 0.18, 1.0], dtype=np.float32)
        assert_roundtrip(
            lambda x: cu.linear_to_logc3(x, ei),
            lambda x: cu.logc3_to_linear(x, ei),
            vals, label=f"LogC3 EI{ei}"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  ARRI LogC4 (NumPy)
# ─────────────────────────────────────────────────────────────────────────────

class TestLogC4Numpy:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    PROBE = np.array([0.001, 0.01, 0.18, 1.0, 4.0, 10.0], dtype=np.float32)
    # Sub-black scene values (t < x < 0) encoded by the log segment;
    # must roundtrip through the FIXED decode (threshold 0, not c).
    PROBE_SUBBLACK = np.array([-0.015, -0.010, -0.005, 0.0], dtype=np.float32)

    def test_midgray_placement(self):
        """LogC4 (ARRI spec 2023): 18% grey → 0.277 (not 0.32 from old wrong constants).

        BUG REGRESSION: previous tensor implementation (A=4296.65, D=11.593) placed
        18% grey at 0.32.  Correct spec constants (a=2231.826, b=0.9071, c=0.0928)
        place it at 0.277.
        """
        enc = cu.linear_to_logc4(np.array([0.18], dtype=np.float32))
        np.testing.assert_allclose(enc[0], 0.277, atol=0.005)

    def test_black_point_placement(self):
        """LogC4: scene-linear 0.0 (absolute black) must encode to c ≈ 0.0928641."""
        enc = cu.linear_to_logc4(np.array([0.0], dtype=np.float32))
        np.testing.assert_allclose(enc[0], 0.0928641308669373, atol=1e-5,
                                   err_msg="Black point should encode to c, not 0.0")

    def test_roundtrip(self):
        assert_roundtrip(
            cu.linear_to_logc4, cu.logc4_to_linear,
            self.PROBE, label="LogC4"
        )

    def test_subblack_roundtrip(self):
        """
        BUG REGRESSION: decode threshold was img >= c (= 0.0928641) instead of
        img >= 0.0, causing sub-black log-encoded values [0, c) to be decoded
        by the wrong (linear) formula with errors up to 0.007 in scene-linear.
        """
        enc = cu.linear_to_logc4(self.PROBE_SUBBLACK)
        dec = cu.logc4_to_linear(enc)
        np.testing.assert_allclose(
            dec, self.PROBE_SUBBLACK, atol=1e-4,
            err_msg="LogC4 sub-black decode threshold bug: y in [0,c) decoded by wrong branch",
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Sony S-Log3 (NumPy)
# ─────────────────────────────────────────────────────────────────────────────

class TestSLog3Numpy:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    PROBE = np.array([0.0, 0.01, 0.18, 0.9, 2.0], dtype=np.float32)

    def test_midgray_placement(self):
        """S-Log3: 18% gray → ~0.41 (per Sony spec)."""
        enc = cu.linear_to_slog3(np.array([0.18], dtype=np.float32))
        np.testing.assert_allclose(enc[0], 0.41, atol=0.01)

    def test_roundtrip(self):
        """
        Regression for S-Log3 operator-precedence bug:
        log10((x + 0.01) / 0.19) was previously log10(x + 0.01) / 0.19
        which produced negative values for 18% gray.
        """
        assert_roundtrip(
            cu.linear_to_slog3, cu.slog3_to_linear,
            self.PROBE, label="S-Log3"
        )

    def test_no_negative_for_midgray(self):
        """Pre-bug: 18% gray encoded to ~-0.56. Post-fix: must be ~+0.41."""
        enc = cu.linear_to_slog3(np.array([0.18], dtype=np.float32))
        assert enc[0] > 0, "S-Log3 BUG: 18% gray encoded as negative value"


# ─────────────────────────────────────────────────────────────────────────────
#  Panasonic V-Log (NumPy)
# ─────────────────────────────────────────────────────────────────────────────

class TestVLogNumpy:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    PROBE = np.array([0.0, 0.01, 0.18, 1.0, 3.0], dtype=np.float32)

    def test_roundtrip(self):
        """
        Regression for V-Log cut_encoded bug:
        was 0.0819, correct is 0.181. Caused a visible kink in darks/mids.
        """
        assert_roundtrip(
            cu.linear_to_vlog, cu.vlog_to_linear,
            self.PROBE, label="V-Log"
        )

    def test_cut_continuity(self):
        """
        Encode/decode must be continuous at the linear-to-log cut point (~0.01).
        Check that values immediately around cut have no discontinuity.
        """
        near_cut = np.array([0.009, 0.0095, 0.01, 0.0105, 0.011], dtype=np.float32)
        enc = cu.linear_to_vlog(near_cut)
        dec = cu.vlog_to_linear(enc)
        # Decoded must match input within float32 noise
        np.testing.assert_allclose(dec, near_cut, atol=1e-4,
                                   err_msg="V-Log: discontinuity at cut point")
        # Encoded must be monotonically increasing
        diffs = np.diff(enc)
        assert (diffs >= 0).all(), "V-Log encode is not monotonic at cut point"


# ─────────────────────────────────────────────────────────────────────────────
#  Canon Log3 (NumPy)
# ─────────────────────────────────────────────────────────────────────────────

class TestCanonLog3Numpy:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    PROBE = np.array([0.0, 0.018, 0.18, 1.0, 2.0], dtype=np.float32)

    def test_midgray_placement(self):
        """
        Canon Log3: 18% gray → 0.343371 per Canon's published spec.

        This previously asserted ~0.336 with a note that the constants did not
        match the spec and "a future fix should re-derive" them. That fix has
        now landed: the implementation was missing the x/0.9 reflectance
        scaling and was using the negative-branch intercept on the positive
        log branch. The assertion tracks the spec, not the implementation.
        """
        enc = cu.linear_to_canonlog3(np.array([0.18], dtype=np.float32))
        np.testing.assert_allclose(enc[0], 0.343371, atol=1e-4)

    def test_roundtrip(self):
        assert_roundtrip(
            cu.linear_to_canonlog3, cu.canonlog3_to_linear,
            self.PROBE, label="Canon Log3"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  RED Log3G10 (NumPy)
# ─────────────────────────────────────────────────────────────────────────────

class TestLog3G10Numpy:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    PROBE = np.array([0.0, 0.01, 0.18, 1.0, 5.0], dtype=np.float32)

    def test_midgray_placement(self):
        """
        Log3G10: 18% gray → 1/3 exactly, per RED's published spec.

        This previously asserted ~0.338 with a note that a future fix should
        re-derive the constants. That fix has now landed: Log3G10 v2 is a
        single continuous curve over (x + 0.01), and the implementation was
        dropping that shift and substituting a linear toe.
        """
        enc = cu.linear_to_log3g10(np.array([0.18], dtype=np.float32))
        np.testing.assert_allclose(enc[0], 1.0 / 3.0, atol=1e-4)

    def test_roundtrip(self):
        """
        Regression for Log3G10 cut_encoded bug:
        was ~0.1133, correct is 0.1016. Caused discontinuity at cut.
        """
        assert_roundtrip(
            cu.linear_to_log3g10, cu.log3g10_to_linear,
            self.PROBE, label="Log3G10"
        )

    def test_cut_continuity(self):
        near_cut = np.array([0.009, 0.0095, 0.01, 0.0105, 0.011], dtype=np.float32)
        enc = cu.linear_to_log3g10(near_cut)
        dec = cu.log3g10_to_linear(enc)
        np.testing.assert_allclose(dec, near_cut, atol=1e-4,
                                   err_msg="Log3G10: discontinuity at cut point")


# ─────────────────────────────────────────────────────────────────────────────
#  DaVinci Intermediate (NumPy)
# ─────────────────────────────────────────────────────────────────────────────

class TestDaVinciIntermediateNumpy:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    PROBE = np.array([0.001, 0.01, 0.18, 1.0, 3.0], dtype=np.float32)

    def test_roundtrip(self):
        """
        Regression for DaVinci Intermediate C1-discontinuous toe bug.
        Old implementation: slope=7.0, intercept=0.07329248 — produced a
        C0 discontinuity of ~0.26 at the cut point and a broken inverse.
        Fixed implementation: C1-continuous constants derived from the log
        segment derivative at cut (slope≈40.78, intercept≈0.311).
        """
        assert_roundtrip(
            cu.linear_to_davinci_intermediate, cu.davinci_intermediate_to_linear,
            self.PROBE, label="DaVinci Intermediate"
        )

    def test_cut_continuity(self):
        """Verify C1 continuity at the cut point (~0.00262)."""
        cut = 0.00262409
        near = np.array([cut * 0.9, cut * 0.99, cut, cut * 1.01, cut * 1.1],
                        dtype=np.float32)
        enc = cu.linear_to_davinci_intermediate(near)
        # Must be strictly increasing (monotonic)
        diffs = np.diff(enc)
        assert (diffs > 0).all(), \
            "DaVinci Intermediate: not monotonic at cut point (C1 discontinuity)"


# ─────────────────────────────────────────────────────────────────────────────
#  ACEScct (NumPy)
# ─────────────────────────────────────────────────────────────────────────────

class TestACEScct:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    PROBE = np.array([0.0, 0.0078, 0.18, 1.0, 3.0], dtype=np.float32)

    def test_roundtrip(self):
        assert_roundtrip(
            cu.linear_to_acescct, cu.acescct_to_linear,
            self.PROBE, label="ACEScct"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  ST.2084 PQ and HLG (NumPy)
# ─────────────────────────────────────────────────────────────────────────────

class TestHDRTransferFunctions:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    def test_pq_roundtrip(self):
        """PQ encode→decode roundtrip at 1000 nits."""
        linear = np.array([0.0, 0.01, 0.1, 0.5, 1.0], dtype=np.float32)
        encoded = cu.linear_to_pq(linear, peak_nits=1000.0)
        decoded = cu.pq_to_linear(encoded, peak_nits=1000.0)
        np.testing.assert_allclose(decoded, linear, rtol=1e-4, atol=1e-6)

    def test_pq_100nit_maps_to_signal(self):
        """
        BT.2408: 100 nit = 1.0 scene-linear at 100-nit display.
        PQ signal for 100/10000 = 0.01 relative luminance ≈ 0.508.
        """
        enc = cu.linear_to_pq(np.array([1.0], dtype=np.float32), peak_nits=100.0)
        # At 100 nit display, 1.0 scene-linear should map to ~0.508 PQ code
        np.testing.assert_allclose(enc[0], 0.508, atol=0.01)

    def test_hlg_roundtrip(self):
        """HLG encode→decode roundtrip."""
        linear = np.array([0.0, 1.0 / 12.0, 0.5, 1.0], dtype=np.float32)
        encoded = cu.linear_to_hlg(linear)
        decoded = cu.hlg_to_linear(encoded)
        np.testing.assert_allclose(decoded, linear, rtol=1e-4, atol=1e-5)

    def test_hlg_boundary(self):
        """HLG segment boundary at E=1/12 must be continuous."""
        boundary = 1.0 / 12.0
        near = np.array([boundary * 0.99, boundary, boundary * 1.01], dtype=np.float32)
        enc = cu.linear_to_hlg(near)
        # Must be monotonically increasing
        assert (np.diff(enc) > 0).all(), "HLG: not monotonic at segment boundary"


# ─────────────────────────────────────────────────────────────────────────────
#  GPU tensor log curves (torch)
# ─────────────────────────────────────────────────────────────────────────────

@_SKIP_NO_TORCH
class TestTensorLogCurves:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    """
    Verify GPU tensor log curves match their NumPy counterparts within float32
    tolerance.  Tests are skipped if torch is not installed (or only stub).
    """

    @pytest.fixture(autouse=True)
    def require_torch(self):
        import torch
        self.torch = torch

    def _probe(self):
        return self.torch.tensor([0.01, 0.18, 1.0, 3.0], dtype=self.torch.float32)

    def _np(self):
        return np.array([0.01, 0.18, 1.0, 3.0], dtype=np.float32)

    def test_logc3_matches_numpy(self):
        t = self._probe()
        np_enc = cu.linear_to_logc3(self._np(), 800)
        t_enc = cu.tensor_linear_to_logc3(t, 800).numpy()
        np.testing.assert_allclose(t_enc, np_enc, rtol=1e-4, atol=1e-5,
                                   err_msg="LogC3 tensor vs numpy mismatch")

    def test_logc3_roundtrip(self):
        t = self._probe()
        enc = cu.tensor_linear_to_logc3(t, 800)
        dec = cu.tensor_logc3_to_linear(enc, 800)
        self.torch.testing.assert_close(dec, t, rtol=1e-4, atol=1e-5)

    def test_logc4_roundtrip(self):
        t = self._probe()
        enc = cu.tensor_linear_to_logc4(t)
        dec = cu.tensor_logc4_to_linear(enc)
        self.torch.testing.assert_close(dec, t, rtol=1e-4, atol=1e-4)

    def test_logc4_matches_numpy(self):
        """
        BUG REGRESSION: tensor_linear_to_logc4 previously used A=4296.65, D=11.593
        (no B/C offset), placing 18% grey at ~0.32 instead of the spec ~0.277 and
        encoding black at 0.0 instead of 0.0928641.  Tensor and numpy must agree.
        """
        np_vals = self._np()
        t_vals  = self._probe()
        np_enc  = cu.linear_to_logc4(np_vals)
        t_enc   = cu.tensor_linear_to_logc4(t_vals).numpy()
        np.testing.assert_allclose(t_enc, np_enc, rtol=1e-4, atol=1e-4,
                                   err_msg="LogC4 tensor vs numpy mismatch (wrong constants)")

    def test_logc4_tensor_midgray(self):
        """Tensor LogC4: 18% grey → ~0.277 (regression against old wrong 0.32)."""
        t = self.torch.tensor([0.18], dtype=self.torch.float32)
        enc = cu.tensor_linear_to_logc4(t)
        assert abs(enc.item() - 0.277) < 0.005, \
            f"LogC4 tensor midgray wrong: got {enc.item():.4f}, expected ~0.277"

    def test_slog3_roundtrip(self):
        """Regression for S-Log3 tensor operator-precedence bug (BUG-1 in color_utils)."""
        t = self.torch.tensor([0.18], dtype=self.torch.float32)
        enc = cu.tensor_linear_to_slog3(t)
        assert enc.item() > 0, "S-Log3 tensor: 18% gray encodes negative (bug regression)"
        dec = cu.tensor_slog3_to_linear(enc)
        self.torch.testing.assert_close(dec, t, rtol=1e-4, atol=1e-5)

    def test_vlog_roundtrip(self):
        """Regression for V-Log tensor cut_encoded bug."""
        t = self._probe()
        enc = cu.tensor_linear_to_vlog(t)
        dec = cu.tensor_vlog_to_linear(enc)
        self.torch.testing.assert_close(dec, t, rtol=1e-4, atol=1e-4)

    def test_log3g10_roundtrip(self):
        """Regression for Log3G10 cut_encoded and device-mismatch bugs."""
        t = self._probe()
        enc = cu.tensor_linear_to_log3g10(t)
        dec = cu.tensor_log3g10_to_linear(enc)
        self.torch.testing.assert_close(dec, t, rtol=1e-4, atol=1e-5)

    def test_davinci_intermediate_roundtrip(self):
        """Regression for DaVinci Intermediate C1-discontinuous toe."""
        t = self._probe()
        enc = cu.tensor_linear_to_davinci_intermediate(t)
        dec = cu.tensor_davinci_intermediate_to_linear(enc)
        self.torch.testing.assert_close(dec, t, rtol=1e-4, atol=1e-5)

    def test_acescct_roundtrip(self):
        t = self._probe()
        enc = cu.tensor_linear_to_acescct(t)
        dec = cu.tensor_acescct_to_linear(enc)
        self.torch.testing.assert_close(dec, t, rtol=1e-4, atol=1e-5)

    def test_pq_roundtrip(self):
        t = self.torch.tensor([0.0, 0.1, 0.5, 1.0], dtype=self.torch.float32)
        enc = cu.tensor_linear_to_pq(t, peak_nits=1000.0)
        dec = cu.tensor_pq_to_linear(enc, peak_nits=1000.0)
        self.torch.testing.assert_close(dec, t, rtol=1e-4, atol=1e-5)

    def test_hlg_roundtrip(self):
        t = self.torch.tensor([0.0, 1.0 / 12.0, 0.5, 1.0], dtype=self.torch.float32)
        enc = cu.tensor_linear_to_hlg(t)
        dec = cu.tensor_hlg_to_linear(enc)
        self.torch.testing.assert_close(dec, t, rtol=1e-4, atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
#  Color matrices
# ─────────────────────────────────────────────────────────────────────────────

class TestColorMatrices:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    def test_srgb_acescg_inverse(self):
        """SRGB_TO_ACESCG and ACESCG_TO_SRGB must be matrix inverses."""
        m1 = cu.SRGB_TO_ACESCG
        m2 = cu.ACESCG_TO_SRGB
        product = m1 @ m2
        np.testing.assert_allclose(product, np.eye(3, dtype=np.float32),
                                   atol=1e-4, err_msg="SRGB⇄ACEScg matrices not inverses")

    def test_srgb_acescg_roundtrip_image(self, linear_image_hwc):
        """sRGB→ACEScg→sRGB roundtrip on a full image."""
        acescg = cu.linear_srgb_to_acescg(linear_image_hwc)
        back = cu.acescg_to_linear_srgb(acescg)
        np.testing.assert_allclose(back, linear_image_hwc, atol=1e-4,
                                   err_msg="sRGB⇄ACEScg image roundtrip failed")

    def test_apply_matrix_transform(self):
        """apply_matrix_transform(img, I) == img (identity matrix)."""
        img = np.random.rand(4, 4, 3).astype(np.float32)
        out = cu.apply_matrix_transform(img, np.eye(3, dtype=np.float32))
        np.testing.assert_array_equal(out, img)


# ─────────────────────────────────────────────────────────────────────────────
#  ACES gamut compress
# ─────────────────────────────────────────────────────────────────────────────

class TestACESGamutCompress:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    def test_in_gamut_unchanged(self):
        """
        Neutral (achromatic) pixels are unchanged by gamut compression.

        A truly "in-gamut" test requires near-achromatic pixels because the
        compression metric is chroma-to-luma ratio (distance from achromatic).
        Pixels at R=G=B have distance=0, which is always below threshold=0.75.
        """
        # Build a gray ramp: R=G=B for every pixel → achromatic → never compressed
        v = np.linspace(0.01, 0.9, 8 * 8, dtype=np.float32)
        img = np.stack([v, v, v], axis=-1).reshape(8, 8, 3)
        out = cu.aces2_gamut_compress(img, threshold=0.75)
        np.testing.assert_allclose(out, img, atol=1e-5,
                                   err_msg="Achromatic image modified by compress")

    def test_out_of_gamut_clamped(self):
        """Out-of-gamut values are brought back toward gamut."""
        # Single pixel: R extremely saturated
        img = np.array([[[3.0, 0.1, 0.1]]], dtype=np.float32)
        out = cu.aces2_gamut_compress(img)
        # The extreme saturation should be reduced
        original_sat = abs(img[0, 0, 0] - img[0, 0, 1])
        result_sat   = abs(out[0, 0, 0] - out[0, 0, 1])
        assert result_sat < original_sat, "Gamut compress did not reduce saturation"

    def test_no_negative_output(self):
        """Compressed image must not introduce negative values."""
        img = np.random.default_rng(1).random((16, 16, 3)).astype(np.float32) * 4.0
        out = cu.aces2_gamut_compress(img)
        assert (out >= -1e-6).all(), "Gamut compress produced negative values"
