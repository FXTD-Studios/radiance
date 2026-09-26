"""
test_hdr_analysis.py — RadianceHDRAnalysis unit tests.

Covers:
  • BT.2408 anchor: 203 nit = scene-linear 1.0
  • peak_nit calculation from luma max
  • is_hdr flag: True iff peak_nit > 203
  • clipped_pct: % of luma > 1.0
  • ev_range: log2(p99 / p01)
  • stats_json is valid JSON with expected keys
  • Constant and mixed images
"""

import os
import sys
import json
import math

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import torch
    # Check for real torch: the stub (MagicMock) has no __version__ string
    HAS_TORCH = isinstance(getattr(torch, "__version__", None), str)
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(
    not HAS_TORCH,
    reason="torch not installed — RadianceHDRAnalysis requires torch"
)


@pytest.fixture(scope="module")
def node():
    """Instantiate RadianceHDRAnalysis once per module."""
    from radiance.nodes.generate.engine import RadianceHDRAnalysis
    return RadianceHDRAnalysis()


def _make_image(values: np.ndarray) -> "torch.Tensor":
    """
    Build a ComfyUI-style (B, H, W, 3) float32 tensor from a flat array of
    luma-equivalent RGB values (r=g=b=v, so luma ≈ v).
    """
    import torch
    v = np.asarray(values, dtype=np.float32)
    # Shape (1, N, 1, 3) — N pixels, 1 row, 1 batch, 3 channels (r=g=b=v)
    rgb = np.stack([v, v, v], axis=-1)[np.newaxis, :, np.newaxis, :]
    return torch.from_numpy(rgb)


# ─────────────────────────────────────────────────────────────────────────────
#  BT.2408 anchor
# ─────────────────────────────────────────────────────────────────────────────

class TestBT2408Anchor:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    def test_linear_1_0_is_203_nit(self, node):
        """Scene-linear 1.0 must map to exactly 203 nit (BT.2408 SDR white)."""
        img = _make_image([1.0])
        peak_nit, ev_range, clipped_pct, is_hdr, stats_json = node.analyse(img)
        assert peak_nit == pytest.approx(203.0, rel=1e-4)

    def test_linear_2_0_is_406_nit(self, node):
        """Scene-linear 2.0 (1 stop above SDR white) → 406 nit."""
        img = _make_image([2.0])
        peak_nit, *_ = node.analyse(img)
        assert peak_nit == pytest.approx(406.0, rel=1e-4)

    def test_linear_0_5_is_101_5_nit(self, node):
        """Scene-linear 0.5 → 101.5 nit."""
        img = _make_image([0.5])
        peak_nit, *_ = node.analyse(img)
        assert peak_nit == pytest.approx(101.5, rel=1e-4)

    def test_peak_nit_scales_linearly(self, node):
        """peak_nit = peak_linear × 203."""
        for v in [0.1, 0.5, 1.0, 2.0, 5.0]:
            img = _make_image([v])
            peak_nit, *_ = node.analyse(img)
            assert peak_nit == pytest.approx(v * 203.0, rel=1e-4), \
                f"Expected {v * 203.0} nit for linear {v}, got {peak_nit}"


# ─────────────────────────────────────────────────────────────────────────────
#  is_hdr flag
# ─────────────────────────────────────────────────────────────────────────────

class TestIsHDRFlag:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    def test_sdr_white_not_hdr(self, node):
        """linear 1.0 → peak_nit == 203 → NOT hdr (strictly >)."""
        img = _make_image([1.0])
        _, _, _, is_hdr, _ = node.analyse(img)
        assert is_hdr is False

    def test_above_sdr_is_hdr(self, node):
        """Any peak above 1.0 linear → is_hdr True."""
        img = _make_image([1.001])
        _, _, _, is_hdr, _ = node.analyse(img)
        assert is_hdr is True

    def test_below_sdr_not_hdr(self, node):
        """All pixels below 1.0 → is_hdr False."""
        img = _make_image([0.0, 0.5, 0.99])
        _, _, _, is_hdr, _ = node.analyse(img)
        assert is_hdr is False

    def test_mixed_hdr_content(self, node):
        """One HDR pixel among many SDR pixels → is_hdr True."""
        vals = [0.2] * 100 + [2.0]
        img = _make_image(vals)
        _, _, _, is_hdr, _ = node.analyse(img)
        assert is_hdr is True


# ─────────────────────────────────────────────────────────────────────────────
#  clipped_pct
# ─────────────────────────────────────────────────────────────────────────────

class TestClippedPct:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    def test_all_sdr_zero_clipped(self, node):
        """No pixels above 1.0 → clipped_pct == 0."""
        img = _make_image([0.0, 0.5, 0.9, 1.0])
        _, _, clipped_pct, _, _ = node.analyse(img)
        assert clipped_pct == pytest.approx(0.0, abs=0.01)

    def test_all_above_one_is_100_pct(self, node):
        """All pixels above 1.0 → clipped_pct == 100."""
        img = _make_image([1.1, 1.5, 2.0, 3.0])
        _, _, clipped_pct, _, _ = node.analyse(img)
        assert clipped_pct == pytest.approx(100.0, abs=0.1)

    def test_half_clipped(self, node):
        """50% of pixels above 1.0 → clipped_pct ≈ 50."""
        vals = [0.5] * 100 + [1.5] * 100
        img = _make_image(vals)
        _, _, clipped_pct, _, _ = node.analyse(img)
        assert clipped_pct == pytest.approx(50.0, abs=1.0)


# ─────────────────────────────────────────────────────────────────────────────
#  ev_range
# ─────────────────────────────────────────────────────────────────────────────

class TestEVRange:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    def test_constant_image_zero_ev(self, node):
        """Constant image → p01 ≈ p99 → ev_range ≈ 0."""
        img = _make_image([0.5] * 200)
        _, ev_range, _, _, _ = node.analyse(img)
        assert ev_range == pytest.approx(0.0, abs=0.5)

    def test_8_stop_range(self, node):
        """
        Image spanning exactly 8 stops: values from 2^-4 to 2^4.
        p01 ≈ 0.0625, p99 ≈ 16.0 → log2(16/0.0625) = log2(256) = 8 stops.
        Use a large enough pixel count for the percentile to be accurate.
        """
        lo, hi = 2**-4, 2**4   # 0.0625 → 16.0
        # Ramp evenly between lo and hi on a log scale
        vals = np.exp(np.linspace(math.log(lo), math.log(hi), 2000)).tolist()
        img = _make_image(vals)
        _, ev_range, _, _, _ = node.analyse(img)
        assert ev_range == pytest.approx(8.0, abs=0.3)


# ─────────────────────────────────────────────────────────────────────────────
#  stats_json
# ─────────────────────────────────────────────────────────────────────────────

class TestStatsJSON:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    _EXPECTED_KEYS = {
        "node", "colorspace", "pixels_sampled",
        "p01", "p50", "p99", "p99.9", "peak_linear",
        "peak_nit", "ev_range", "clipped_pct", "is_hdr", "nit_anchor",
    }

    def test_valid_json(self, node):
        img = _make_image([0.5])
        _, _, _, _, stats_json = node.analyse(img)
        parsed = json.loads(stats_json)  # must not raise
        assert isinstance(parsed, dict)

    def test_expected_keys_present(self, node):
        img = _make_image([0.5])
        _, _, _, _, stats_json = node.analyse(img)
        parsed = json.loads(stats_json)
        for key in self._EXPECTED_KEYS:
            assert key in parsed, f"Missing key in stats_json: {key!r}"

    def test_nit_anchor_is_203(self, node):
        img = _make_image([1.0])
        _, _, _, _, stats_json = node.analyse(img)
        parsed = json.loads(stats_json)
        assert parsed["nit_anchor"] == pytest.approx(203.0)

    def test_peak_nit_consistent_with_return_value(self, node):
        img = _make_image([2.5])
        peak_nit, _, _, _, stats_json = node.analyse(img)
        parsed = json.loads(stats_json)
        assert parsed["peak_nit"] == pytest.approx(peak_nit, rel=1e-3)

    def test_is_hdr_consistent_with_return_value(self, node):
        for v in [0.5, 1.0, 1.5, 3.0]:
            img = _make_image([v])
            _, _, _, is_hdr, stats_json = node.analyse(img)
            parsed = json.loads(stats_json)
            assert parsed["is_hdr"] == is_hdr, \
                f"is_hdr mismatch at linear {v}: return={is_hdr}, json={parsed['is_hdr']}"


# ─────────────────────────────────────────────────────────────────────────────
#  Return types
# ─────────────────────────────────────────────────────────────────────────────

class TestReturnTypes:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    def test_return_signature(self, node):
        """analyse() must return a 5-tuple (float, float, float, bool, str)."""
        img = _make_image([1.0])
        result = node.analyse(img)
        assert len(result) == 5
        peak_nit, ev_range, clipped_pct, is_hdr, stats_json = result
        assert isinstance(peak_nit, float)
        assert isinstance(ev_range, float)
        assert isinstance(clipped_pct, float)
        assert isinstance(is_hdr, bool)
        assert isinstance(stats_json, str)
