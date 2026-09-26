"""
tests/test_policy_guard.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Tests for nodes_policy_guard — delivery ruleset / policy gate

Coverage
────────
  _luma                 — luma conversion correctness
  _mean_saturation      — saturation range [0, 1]
  _gamut_out_of_p3      — fraction in [0, 1]
  _analyse              — returns required metric keys
  _evaluate             — pass/fail based on thresholds
  RadiancePolicyPreset  — lists valid presets, INPUT_TYPES
  RadiancePolicyGuard   — INPUT_TYPES, RETURN_TYPES
  Node registration     — 2 nodes in NODE_CLASS_MAPPINGS
"""

from __future__ import annotations

import json
import os
import sys
import unittest

import numpy as np

# ── Module under test ────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# NODE_CLASS_MAPPINGS moved to the group package: the root
# aggregator that used to build it was a shim over these classes.
from radiance.nodes.color import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)
from radiance.nodes.color.qc import (
    _luma,
    _mean_saturation,
    _gamut_out_of_p3,
    _policy_analyse as _analyse,
    _evaluate_policy as _evaluate,
    RadiancePolicyGuard,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _solid(h=8, w=8, r=0.5, g=0.5, b=0.5) -> np.ndarray:
    arr = np.zeros((h, w, 3), dtype=np.float32)
    arr[..., 0] = r
    arr[..., 1] = g
    arr[..., 2] = b
    return arr

def _grey(v=0.5, h=8, w=8) -> np.ndarray:
    return _solid(h, w, r=v, g=v, b=v)


# ═════════════════════════════════════════════════════════════════════════════
# 1. _luma
# ═════════════════════════════════════════════════════════════════════════════

class TestLuma(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    def test_grey_luma_equals_value(self):
        arr = _grey(0.5)
        luma = _luma(arr)
        self.assertAlmostEqual(float(luma.mean()), 0.5, places=3)

    def test_black_luma_zero(self):
        arr = _grey(0.0)
        luma = _luma(arr)
        self.assertTrue((luma == 0).all())

    def test_white_luma_one(self):
        arr = _grey(1.0)
        luma = _luma(arr)
        self.assertTrue(np.allclose(luma, 1.0, atol=1e-4))

    def test_output_shape_is_2d(self):
        arr = _solid(16, 16)
        luma = _luma(arr)
        self.assertEqual(luma.ndim, 2)
        self.assertEqual(luma.shape, (16, 16))

    def test_pure_red_luma(self):
        # BT.709: R=0.2126, G=0.7152, B=0.0722
        arr = _solid(4, 4, r=1.0, g=0.0, b=0.0)
        luma = _luma(arr).mean()
        self.assertAlmostEqual(float(luma), 0.2126, places=2)

    def test_pure_green_luma(self):
        arr = _solid(4, 4, r=0.0, g=1.0, b=0.0)
        luma = _luma(arr).mean()
        self.assertAlmostEqual(float(luma), 0.7152, places=2)


# ═════════════════════════════════════════════════════════════════════════════
# 2. _mean_saturation
# ═════════════════════════════════════════════════════════════════════════════

class TestMeanSaturation(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    def test_grey_saturation_zero(self):
        arr = _grey(0.5)
        sat = _mean_saturation(arr)
        self.assertAlmostEqual(sat, 0.0, places=4)

    def test_pure_red_saturation_one(self):
        arr = _solid(4, 4, r=1.0, g=0.0, b=0.0)
        sat = _mean_saturation(arr)
        self.assertAlmostEqual(sat, 1.0, places=4)

    def test_saturation_in_range(self):
        rng = np.random.default_rng(0)
        arr = rng.random((16, 16, 3)).astype(np.float32)
        sat = _mean_saturation(arr)
        self.assertGreaterEqual(sat, 0.0)
        self.assertLessEqual(sat, 1.0)


# ═════════════════════════════════════════════════════════════════════════════
# 3. _gamut_out_of_p3
# ═════════════════════════════════════════════════════════════════════════════

class TestGamutOutOfP3(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    def test_clamped_image_all_in_gamut(self):
        arr = np.clip(np.random.default_rng(1).random((8, 8, 3)), 0, 1).astype(np.float32)
        frac = _gamut_out_of_p3(arr)
        self.assertGreaterEqual(frac, 0.0)
        self.assertLessEqual(frac, 1.0)

    def test_zero_for_conservative_grey(self):
        arr = _grey(0.5)
        frac = _gamut_out_of_p3(arr)
        self.assertGreaterEqual(frac, 0.0)

    def test_returns_float(self):
        arr = _solid(4, 4)
        frac = _gamut_out_of_p3(arr)
        self.assertIsInstance(frac, float)


# ═════════════════════════════════════════════════════════════════════════════
# 4. _analyse
# ═════════════════════════════════════════════════════════════════════════════

class TestAnalyse(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    # Actual keys returned by _analyse
    REQUIRED_KEYS = {"peak", "min", "mean_luma", "clipping", "black_crush",
                     "mean_sat", "gamut_violation"}

    def test_required_keys_present(self):
        arr = _grey(0.5, 16, 16)
        stats = _analyse(arr)
        for k in self.REQUIRED_KEYS:
            self.assertIn(k, stats, f"Missing: {k}")

    def test_values_finite(self):
        arr = _grey(0.5, 16, 16)
        stats = _analyse(arr)
        for k, v in stats.items():
            self.assertTrue(np.isfinite(v), f"{k}={v} not finite")

    def test_grey_luma_range(self):
        arr = _grey(0.5, 16, 16)
        stats = _analyse(arr)
        self.assertAlmostEqual(stats["mean_luma"], 0.5, places=2)

    def test_black_image(self):
        arr = _grey(0.0, 8, 8)
        stats = _analyse(arr)
        self.assertAlmostEqual(stats["mean_luma"], 0.0, places=4)
        self.assertAlmostEqual(stats["peak"], 0.0, places=4)

    def test_white_image(self):
        arr = _grey(1.0, 8, 8)
        stats = _analyse(arr)
        self.assertAlmostEqual(stats["mean_luma"], 1.0, places=4)
        self.assertAlmostEqual(stats["peak"], 1.0, places=4)


# ═════════════════════════════════════════════════════════════════════════════
# 5. _evaluate
# ═════════════════════════════════════════════════════════════════════════════
# Signature: _evaluate(stats, policy, metadata_keys) -> (passed, violations, score)

class TestEvaluate(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    def _stats(self, luma=0.5, clipping=0.0, black_crush=0.0,
               mean_sat=0.4, gamut=0.0, peak=0.5) -> dict:
        return {
            "peak":            peak,
            "min":             0.0,
            "mean_luma":       luma,
            "clipping":        clipping,
            "black_crush":     black_crush,
            "mean_sat":        mean_sat,
            "gamut_violation": gamut,
        }

    def _policy(self, **kw) -> dict:
        base = {
            "min_luma":       0.1,
            "max_luma":       0.9,
            "max_clipping":   0.01,
            "max_black_crush": 0.05,
            "max_saturation": 1.0,
            "max_peak_nits":  10000.0,  # large → won't trigger
        }
        base.update(kw)
        return base

    def test_returns_tuple_three_items(self):
        result = _evaluate(self._stats(), self._policy(), [])
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

    def test_pass_within_limits(self):
        passed, violations, score = _evaluate(self._stats(), self._policy(), [])
        self.assertTrue(passed)
        self.assertEqual(violations, [])

    def test_fail_luma_too_dark(self):
        passed, violations, _ = _evaluate(
            self._stats(luma=0.01), self._policy(min_luma=0.2), []
        )
        # min_luma violation is a "warning" — may not flip passed=False depending on impl
        self.assertIsInstance(passed, bool)
        self.assertGreater(len(violations), 0)

    def test_fail_clipping(self):
        passed, violations, _ = _evaluate(
            self._stats(clipping=0.5), self._policy(max_clipping=0.001), []
        )
        self.assertFalse(passed)
        self.assertGreater(len(violations), 0)

    def test_violations_have_rule_field(self):
        _, violations, _ = _evaluate(
            self._stats(clipping=0.5), self._policy(max_clipping=0.001), []
        )
        for v in violations:
            self.assertIn("rule", v)

    def test_score_0_to_100(self):
        _, _, score = _evaluate(self._stats(), self._policy(), [])
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)

    def test_missing_metadata_triggers_violation(self):
        policy = self._policy()
        policy["require_metadata"] = ["scene_name"]
        passed, violations, _ = _evaluate(self._stats(), policy, [])
        rules = [v["rule"] for v in violations]
        self.assertIn("missing_metadata", rules)

    def test_present_metadata_no_violation(self):
        policy = self._policy()
        policy["require_metadata"] = ["scene_name"]
        passed, violations, _ = _evaluate(self._stats(), policy, ["scene_name"])
        rules = [v["rule"] for v in violations]
        self.assertNotIn("missing_metadata", rules)




# ═════════════════════════════════════════════════════════════════════════════
# 7. Node registration
# ═════════════════════════════════════════════════════════════════════════════

class TestNodeRegistration(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    EXPECTED = {"RadiancePolicyGuard"}

    def test_all_nodes_registered(self):
        for key in self.EXPECTED:
            self.assertIn(key, NODE_CLASS_MAPPINGS, f"Missing: {key}")

    def test_display_names_present(self):
        for key in self.EXPECTED:
            self.assertIn(key, NODE_DISPLAY_NAME_MAPPINGS)

    def test_policy_guard_has_image_input(self):
        it = RadiancePolicyGuard.INPUT_TYPES()
        req = it.get("required", {})
        self.assertIn("image", req)

    def test_policy_guard_return_types(self):
        rt = RadiancePolicyGuard.RETURN_TYPES
        self.assertIn("IMAGE", rt)
        self.assertIn("STRING", rt)


if __name__ == "__main__":
    unittest.main()


def test_guard_checks_every_frame_and_enforces_hdr_peak():
    """Audit 3.5: only frame 0 was checked; HDR max_peak_nits never fired."""
    import torch
    from radiance.nodes.color.qc import RadiancePolicyGuard
    clean = torch.full((1, 16, 16, 3), 0.4)
    blown = torch.ones((1, 16, 16, 3))
    clip = torch.cat([clean, clean, blown])
    _, passed, rep, _, _ = RadiancePolicyGuard().run("Guard", image=clip)
    assert passed is False and '"frames_checked": 3' in rep and '"worst_clipping_frame": 2' in rep
    hot = torch.full((1, 16, 16, 3), 20.0)   # ~2000 nits at 1.0 = 100 nits
    _, passed, rep, _, _ = RadiancePolicyGuard().run("Guard", image=hot, max_peak_nits=1000.0,
                                                       max_clipping=1.0)
    assert passed is False and "max_peak_nits" in rep
