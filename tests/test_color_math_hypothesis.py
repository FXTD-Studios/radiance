"""
tests/test_color_math_hypothesis.py
────────────────────────────────────────────────────────────────────────────────
Property-based tests for Radiance color-math and sigma-schedule utilities.

Strategy
────────
We use Hypothesis to generate arbitrary floating-point inputs and verify that
mathematical invariants hold universally, not just for the fixed examples that
hand-written unit tests tend to cover.

Key invariants tested
─────────────────────
1. Round-trip stability   — encode(decode(x)) ≈ x and decode(encode(x)) ≈ x
2. Monotonicity           — encoding functions must be strictly monotone
3. Boundary correctness   — specific anchor points must hit spec values
4. Sigma schedule         — flux_shift_sigmas output is always strictly decreasing

Run
────
    pytest tests/test_color_math_hypothesis.py -v
    pytest tests/test_color_math_hypothesis.py -v --hypothesis-seed=0
"""

from __future__ import annotations

import math
import sys
import types
import unittest
from unittest.mock import MagicMock

# ── Minimal stubs so the module loads without ComfyUI or a real GPU ──────────
_STUB_MODS = [
    "torch", "torch.nn", "torch.nn.functional",
    "comfy", "comfy.samplers", "comfy.sample",
    "comfy.model_management", "comfy.utils", "comfy.model_base",
    "comfy.sd", "comfy.latent_formats", "comfy.nested_tensor",
    "folder_paths",
]
for _mod in _STUB_MODS:
    if _mod not in sys.modules:
        _m = types.ModuleType(_mod)
        _m.__dict__.update({a: MagicMock() for a in dir(MagicMock())})
        sys.modules[_mod] = _m

# Patch torch with a minimal real-ish surface (hypothesis generates numpy
# arrays; we convert them to real torch.Tensor via the real torch import).
try:
    import torch as _real_torch
    HAS_TORCH = isinstance(_real_torch.__version__, str)
except (ImportError, AttributeError):
    HAS_TORCH = False

# ── Hypothesis availability guard ─────────────────────────────────────────────
try:
    from hypothesis import given, settings, assume, HealthCheck
    from hypothesis import strategies as st
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False
    # Define no-op stubs so class bodies with @given decorators don't raise
    # NameError at collection time when Hypothesis is not installed.
    def given(*_a, **_kw):      return lambda f: f          # type: ignore
    def settings(*_a, **_kw):   return lambda f: f          # type: ignore
    def assume(_):              return None                  # type: ignore
    class HealthCheck:          too_slow = None              # type: ignore
    # Strategy objects must be chainable (.map/.filter/.flatmap): @given(...)
    # evaluates its arguments at class-body (collection) time, before
    # skipUnless can skip — a None here aborts the whole CI run.
    class _DummyStrategy:                                    # type: ignore
        def map(self, *a, **k):      return self
        def filter(self, *a, **k):   return self
        def flatmap(self, *a, **k):  return self

    class _DummyStrategies:                                  # type: ignore
        def __getattr__(self, _name):
            return lambda *a, **k: _DummyStrategy()
    st = _DummyStrategies()                                  # type: ignore

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ─────────────────────────────────────────────────────────────────────────────
# Helper: numpy-only log-curve implementations for property testing
# (avoids requiring a real GPU; mirrors the same math as the torch variants)
# ─────────────────────────────────────────────────────────────────────────────

def _np_linear_to_logc4(x: float) -> float:
    """ARRI LogC4 numpy encode — mirrors tensor_linear_to_logc4."""
    a = 2231.826309067637
    b = 0.9071358691330627
    c = 0.0928641308669373
    t = -0.0180569961199123
    s = 0.1135773173772412
    if x >= t:
        inner = a * x + 64.0
        if inner <= 0:
            return c  # clamp to toe
        return ((math.log2(inner) - 6.0) / 14.0) * b + c
    else:
        return (x - t) / s


def _np_logc4_to_linear(y: float) -> float:
    """ARRI LogC4 numpy decode — mirrors tensor_logc4_to_linear."""
    a = 2231.826309067637
    b = 0.9071358691330627
    c = 0.0928641308669373
    t = -0.0180569961199123
    s = 0.1135773173772412
    if y >= 0.0:
        return (2.0 ** (((y - c) / b) * 14.0 + 6.0) - 64.0) / a
    else:
        return y * s + t


def _np_linear_to_slog3(x: float) -> float:
    """Sony S-Log3 numpy encode."""
    cut = 0.011250
    _toe_slope = 76.2102946929 / (0.02125 * 1023.0)
    x_clamped = max(x, -0.01)
    if x_clamped >= cut:
        inner = max((x_clamped + 0.01) / 0.19, 1e-10)
        return (420.0 + math.log10(inner) * 261.5) / 1023.0
    else:
        return _toe_slope * (x_clamped + 0.01) + 95.0 / 1023.0


def _np_slog3_to_linear(y: float) -> float:
    """Sony S-Log3 numpy decode."""
    cut_v = 171.2102946929 / 1023.0
    _toe_slope = 76.2102946929 / (0.02125 * 1023.0)
    if y >= cut_v:
        return 0.19 * (10.0 ** ((y * 1023.0 - 420.0) / 261.5)) - 0.01
    else:
        return (y - 95.0 / 1023.0) / _toe_slope - 0.01


def _np_linear_to_vlog(x: float) -> float:
    """Panasonic V-Log numpy encode."""
    cut = 0.01
    b, c, d = 0.00873, 0.241514, 0.598206
    if x >= cut:
        return c * math.log10(x + b) + d
    else:
        return 5.6 * x + 0.125


def _np_vlog_to_linear(y: float) -> float:
    """Panasonic V-Log numpy decode."""
    b, c, d = 0.00873, 0.241514, 0.598206
    cut_encoded = 0.181000
    if y >= cut_encoded:
        return 10.0 ** ((y - d) / c) - b
    else:
        return (y - 0.125) / 5.6


# ─────────────────────────────────────────────────────────────────────────────
# Sigma schedule helpers
# ─────────────────────────────────────────────────────────────────────────────

def _flux_shift_sigmas(sigmas: list[float], shift: float) -> list[float]:
    """
    Pure-Python replica of sampler_utils.flux_shift_sigmas.
    shift=1.0 is identity. Higher values push weight toward higher sigmas.
    """
    return [
        shift * s / (1.0 + (shift - 1.0) * s)
        for s in sigmas
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Test class
# ─────────────────────────────────────────────────────────────────────────────

@unittest.skipUnless(HAS_HYPOTHESIS, "hypothesis not installed — pip install hypothesis")
class TestLogCurvesHypothesis(unittest.TestCase):

    # ── LogC4 ─────────────────────────────────────────────────────────────────

    @given(st.floats(min_value=0.001, max_value=100.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
    def test_logc4_encode_decode_roundtrip(self, x: float):
        """encode(decode(encode(x))) ≈ encode(x) — round-trip stability in log domain."""
        y = _np_linear_to_logc4(x)
        x_recovered = _np_logc4_to_linear(y)
        y_recovered = _np_linear_to_logc4(x_recovered)
        self.assertAlmostEqual(y, y_recovered, places=5,
            msg=f"LogC4 round-trip failed for x={x}: y={y}, y_recovered={y_recovered}")

    @given(st.floats(min_value=0.001, max_value=100.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
    def test_logc4_decode_encode_roundtrip(self, x: float):
        """decode(encode(decode(y))) ≈ decode(y) — round-trip stability in linear domain."""
        y = _np_linear_to_logc4(x)
        x_r = _np_logc4_to_linear(y)
        self.assertAlmostEqual(x, x_r, places=4,
            msg=f"LogC4 decode round-trip failed for x={x}: x_r={x_r}")

    @given(
        st.floats(min_value=0.001, max_value=50.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.001, max_value=50.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_logc4_monotone(self, x1: float, x2: float):
        """LogC4 encoding must be strictly monotone: if x1 < x2 then encode(x1) < encode(x2)."""
        assume(abs(x1 - x2) > 1e-6)
        y1 = _np_linear_to_logc4(x1)
        y2 = _np_linear_to_logc4(x2)
        if x1 < x2:
            self.assertLess(y1, y2,
                msg=f"LogC4 not monotone: encode({x1})={y1} >= encode({x2})={y2}")

    def test_logc4_midgray_anchor(self):
        """18% grey (0.18 linear) must encode to ≈ 0.277 in LogC4 v1 spec."""
        y = _np_linear_to_logc4(0.18)
        self.assertAlmostEqual(y, 0.277, delta=0.005,
            msg=f"LogC4 mid-grey anchor failed: encode(0.18)={y}, expected ≈0.277")

    def test_logc4_black_anchor(self):
        """0.0 linear must encode to ≈ 0.0928 (LogC4 black not at code 0)."""
        y = _np_linear_to_logc4(0.0)
        self.assertAlmostEqual(y, 0.0928641308669373, places=5,
            msg=f"LogC4 black anchor failed: encode(0.0)={y}")

    # ── S-Log3 ────────────────────────────────────────────────────────────────

    @given(st.floats(min_value=0.001, max_value=100.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
    def test_slog3_encode_decode_roundtrip(self, x: float):
        """S-Log3 round-trip: decode(encode(x)) ≈ x."""
        y = _np_linear_to_slog3(x)
        x_r = _np_slog3_to_linear(y)
        self.assertAlmostEqual(x, x_r, places=4,
            msg=f"S-Log3 round-trip failed for x={x}: x_r={x_r}")

    @given(
        st.floats(min_value=0.001, max_value=50.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.001, max_value=50.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_slog3_monotone(self, x1: float, x2: float):
        """S-Log3 encoding must be strictly monotone."""
        assume(abs(x1 - x2) > 1e-6)
        y1 = _np_linear_to_slog3(x1)
        y2 = _np_linear_to_slog3(x2)
        if x1 < x2:
            self.assertLess(y1, y2,
                msg=f"S-Log3 not monotone: encode({x1})={y1} >= encode({x2})={y2}")

    def test_slog3_midgray_anchor(self):
        """18% grey must encode to ≈ 0.410 (0 dB, 420/1023) in S-Log3."""
        y = _np_linear_to_slog3(0.18)
        expected = 420.0 / 1023.0  # spec table value
        self.assertAlmostEqual(y, expected, delta=0.002,
            msg=f"S-Log3 mid-grey anchor failed: encode(0.18)={y}, expected≈{expected:.4f}")

    def test_slog3_continuity_at_cut(self):
        """S-Log3 must be C0-continuous at the linear/log transition point (cut=0.01125)."""
        cut = 0.011250
        eps = 1e-7
        y_below = _np_linear_to_slog3(cut - eps)
        y_above = _np_linear_to_slog3(cut + eps)
        # At continuity the two values should be nearly equal
        self.assertAlmostEqual(y_below, y_above, delta=0.001,
            msg=f"S-Log3 discontinuous at cut: y_below={y_below}, y_above={y_above}")

    # ── V-Log ─────────────────────────────────────────────────────────────────

    @given(st.floats(min_value=0.001, max_value=100.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
    def test_vlog_encode_decode_roundtrip(self, x: float):
        """V-Log round-trip: decode(encode(x)) ≈ x."""
        y = _np_linear_to_vlog(x)
        x_r = _np_vlog_to_linear(y)
        self.assertAlmostEqual(x, x_r, places=4,
            msg=f"V-Log round-trip failed for x={x}: x_r={x_r}")

    def test_vlog_cut_encoded_threshold(self):
        """The V-Log encoded cut must be ≈ 0.181 (spec). Incorrect value caused visible kink."""
        # Verify the pre-computed constant matches the formula
        b, c, d = 0.00873, 0.241514, 0.598206
        cut_linear = 0.01
        computed = c * math.log10(cut_linear + b) + d
        self.assertAlmostEqual(computed, 0.181000, delta=0.0005,
            msg=f"V-Log cut_encoded mismatch: {computed} vs 0.181000")

    # ─────────────────────────────────────────────────────────────────────────
    # Sigma schedule
    # ─────────────────────────────────────────────────────────────────────────

    @given(
        st.lists(
            st.floats(min_value=0.001, max_value=0.999, allow_nan=False, allow_infinity=False),
            min_size=3, max_size=30,
        ).map(sorted).map(list.__reversed__).map(list),  # always strictly decreasing
        st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_flux_shift_preserves_order(self, sigmas: list[float], shift: float):
        """flux_shift_sigmas must preserve the strict ordering of the input schedule."""
        assume(len(sigmas) >= 2)
        assume(all(sigmas[i] > sigmas[i + 1] for i in range(len(sigmas) - 1)))
        shifted = _flux_shift_sigmas(sigmas, shift)
        # Allow a tiny floating-point tolerance: inputs one ULP apart (~1e-16)
        # can flip order by a few ULPs through the nonlinear transform without
        # that being a real ordering violation. 1e-9 is far above FP noise yet
        # far below any genuine break.
        _TOL = 1e-9
        for i in range(len(shifted) - 1):
            self.assertGreaterEqual(
                shifted[i] - shifted[i + 1], -_TOL,
                msg=f"flux_shift broke order at i={i} with shift={shift}: "
                    f"{shifted[i]} < {shifted[i+1]}"
            )

    @given(st.floats(min_value=0.001, max_value=0.999, allow_nan=False, allow_infinity=False))
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_flux_shift_identity_at_1(self, sigma: float):
        """flux_shift_sigmas with shift=1.0 must be the identity function."""
        shifted = _flux_shift_sigmas([sigma], shift=1.0)
        self.assertAlmostEqual(shifted[0], sigma, places=6,
            msg=f"flux_shift identity failed: sigma={sigma}, shifted={shifted[0]}")

    @given(st.floats(min_value=0.001, max_value=0.999, allow_nan=False, allow_infinity=False))
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_flux_shift_output_in_0_1(self, sigma: float):
        """flux_shift_sigmas output must remain in (0, 1) for sigmas in (0, 1)."""
        for shift in [0.5, 1.0, 2.0, 5.0]:
            shifted = _flux_shift_sigmas([sigma], shift=shift)
            self.assertGreater(shifted[0], 0.0,
                msg=f"flux_shift output ≤ 0: sigma={sigma}, shift={shift}")
            self.assertLess(shifted[0], 1.0,
                msg=f"flux_shift output ≥ 1: sigma={sigma}, shift={shift}")


# ─────────────────────────────────────────────────────────────────────────────
# Non-hypothesis anchor tests (always run, no Hypothesis required)
# ─────────────────────────────────────────────────────────────────────────────

class TestLogCurveAnchors(unittest.TestCase):
    """Fixed-value anchor tests that must always pass."""

    def test_logc4_encode_1_0(self):
        """LogC4: 1.0 linear → encoded ≈ 0.427 per v1 spec constants."""
        y = _np_linear_to_logc4(1.0)
        self.assertAlmostEqual(y, 0.427, delta=0.010)

    def test_slog3_black_level(self):
        """S-Log3: 0.0 linear uses the linear toe (x=0 < cut=0.01125).
        Expected: _toe_slope * (0.0 + 0.01) + 95/1023.
        """
        _toe_slope = 76.2102946929 / (0.02125 * 1023.0)
        expected = _toe_slope * 0.01 + 95.0 / 1023.0
        y = _np_linear_to_slog3(0.0)
        self.assertAlmostEqual(y, expected, places=5)

    def test_vlog_black_level(self):
        """V-Log: 0.0 linear → encoded ≈ 0.125."""
        y = _np_linear_to_vlog(0.0)
        self.assertAlmostEqual(y, 0.125, delta=0.001)

    def test_flux_shift_known_value(self):
        """flux_shift([0.5], shift=2.0) = 2*0.5/(1+0.5*1) = 1.0/1.5 ≈ 0.6667."""
        result = _flux_shift_sigmas([0.5], shift=2.0)
        self.assertAlmostEqual(result[0], 2.0 / 3.0, places=5)

    def test_logc4_large_linear_value(self):
        """LogC4 should handle scene-linear > 1.0 without crashing."""
        y = _np_linear_to_logc4(100.0)
        self.assertTrue(math.isfinite(y), f"LogC4 encode(100.0) is not finite: {y}")
        self.assertGreater(y, 0.5)  # well above mid-grey


if __name__ == "__main__":
    unittest.main()
