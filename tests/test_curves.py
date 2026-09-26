"""
tests/test_curves.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Tests for nodes_curves — HSL helpers + node registration

Coverage
────────
  Pure helpers : _rgb_to_hsl, _hsl_to_rgb round-trip (torch-gated)
  Registration : RadianceHueCurves, RadianceCurves
"""

from __future__ import annotations

import os
import sys
import unittest

import torch
HAS_TORCH = hasattr(torch, "__version__")
skip_no_torch = unittest.skipUnless(HAS_TORCH, "real torch not available")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# NODE_CLASS_MAPPINGS moved to the group package: the root
# aggregator that used to build it was a shim over these classes.
from radiance.nodes.color import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)
from radiance.nodes.color.curves import (
    _rgb_to_hsl,
    _hsl_to_rgb_fast as _hsl_to_rgb,
)

# This module already gates its torch-dependent tests correctly (they skip
# cleanly against conftest's stub), so opt out of the automatic module-level
# skip and keep the rest of the file running on the no-torch CI matrix.
RADIANCE_TORCH_GATED = True



def _t(b=1, h=4, w=4, fill=0.5):
    return torch.full((b, h, w, 3), fill, dtype=torch.float32)


# ═════════════════════════════════════════════════════════════════════════════
# _rgb_to_hsl / _hsl_to_rgb round-trip
# ═════════════════════════════════════════════════════════════════════════════

class TestRGBHSLConversion(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    @skip_no_torch
    def test_output_shape_preserved(self):
        img = _t(1, 8, 8, 0.5)
        hsl = _rgb_to_hsl(img)
        self.assertEqual(hsl.shape, img.shape)

    @skip_no_torch
    def test_round_trip_grey(self):
        """Grey has S=0, L≈fill regardless of H."""
        img = _t(1, 4, 4, 0.4)
        hsl = _rgb_to_hsl(img)
        rgb = _hsl_to_rgb(hsl)
        torch.testing.assert_close(rgb, img, atol=1e-4, rtol=0)

    @skip_no_torch
    def test_round_trip_saturated_red(self):
        img = torch.zeros(1, 4, 4, 3)
        img[..., 0] = 1.0   # pure red
        hsl = _rgb_to_hsl(img)
        rgb = _hsl_to_rgb(hsl)
        torch.testing.assert_close(rgb, img, atol=1e-4, rtol=0)

    @skip_no_torch
    def test_round_trip_saturated_green(self):
        img = torch.zeros(1, 4, 4, 3)
        img[..., 1] = 1.0
        hsl = _rgb_to_hsl(img)
        rgb = _hsl_to_rgb(hsl)
        torch.testing.assert_close(rgb, img, atol=1e-4, rtol=0)

    @skip_no_torch
    def test_round_trip_saturated_blue(self):
        img = torch.zeros(1, 4, 4, 3)
        img[..., 2] = 1.0
        hsl = _rgb_to_hsl(img)
        rgb = _hsl_to_rgb(hsl)
        torch.testing.assert_close(rgb, img, atol=1e-4, rtol=0)

    @skip_no_torch
    def test_round_trip_random(self):
        img = torch.rand(1, 16, 16, 3).clamp(0, 1)
        hsl = _rgb_to_hsl(img)
        rgb = _hsl_to_rgb(hsl)
        torch.testing.assert_close(rgb, img, atol=1e-4, rtol=0)

    @skip_no_torch
    def test_hsl_lightness_grey(self):
        """Grey (equal RGB) → L matches mean, S≈0."""
        img = _t(1, 4, 4, 0.6)
        hsl = _rgb_to_hsl(img)
        L = hsl[..., 2].mean().item()
        S = hsl[..., 1].mean().item()
        self.assertAlmostEqual(L, 0.6, places=3)
        self.assertLess(S, 1e-3)

    @skip_no_torch
    def test_hue_range_0_1(self):
        img = torch.rand(1, 8, 8, 3).clamp(0.05, 0.95)
        hsl = _rgb_to_hsl(img)
        H = hsl[..., 0]
        self.assertGreaterEqual(float(H.min()), 0.0)
        self.assertLessEqual(float(H.max()), 1.0)

    @skip_no_torch
    def test_saturation_range_0_1(self):
        img = torch.rand(1, 8, 8, 3).clamp(0.05, 0.95)
        hsl = _rgb_to_hsl(img)
        S = hsl[..., 1]
        self.assertGreaterEqual(float(S.min()), 0.0)
        self.assertLessEqual(float(S.max()), 1.0 + 1e-4)


# ═════════════════════════════════════════════════════════════════════════════
# Registration
# ═════════════════════════════════════════════════════════════════════════════

class TestCurvesRegistration(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    EXPECTED = ["RadianceHueCurves", "RadianceCurves"]

    def test_all_in_class_mappings(self):
        for name in self.EXPECTED:
            self.assertIn(name, NODE_CLASS_MAPPINGS)

    def test_all_in_display_mappings(self):
        for name in self.EXPECTED:
            self.assertIn(name, NODE_DISPLAY_NAME_MAPPINGS)

    def test_display_names_prefix(self):
        for _, disp in NODE_DISPLAY_NAME_MAPPINGS.items():
            self.assertTrue(disp.startswith("◎"))

    def test_functions_exist(self):
        for name, cls in NODE_CLASS_MAPPINGS.items():
            fn = cls.FUNCTION
            self.assertTrue(hasattr(cls, fn), f"{name}.{fn} not found")

    def test_input_types_structure(self):
        for name, cls in NODE_CLASS_MAPPINGS.items():
            it = cls.INPUT_TYPES()
            self.assertIn("required", it)


if __name__ == "__main__":
    unittest.main(verbosity=2)
