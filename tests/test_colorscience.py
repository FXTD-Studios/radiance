"""
tests/test_colorscience.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Tests for nodes_colorscience — White balance, Bradford CAT, temperature→xy

Coverage
────────
  Pure math   : _xy_to_XYZ, _build_bradford_matrix, _temperature_to_xy
  Node classes : RadianceWhiteBalance, ColorSpaceConvert, ACESTransform
  Registration
"""

from __future__ import annotations

import math
import os
import sys
import unittest

import torch
HAS_TORCH = hasattr(torch, "__version__")
skip_no_torch = unittest.skipUnless(HAS_TORCH, "real torch not available")

# nodes_colorscience uses relative imports (`from .radiance_ocio import …`)
# so it must be imported as part of the radiance package.
# conftest.py installs the radiance.radiance_ocio stub; we just need
# to make sure the radiance package root is on the path so that
# `importlib.import_module("radiance.nodes.color.colorspace")` works.
_RADIANCE_DIR = os.path.dirname(os.path.dirname(__file__))
_MNT_DIR      = os.path.dirname(_RADIANCE_DIR)
if _MNT_DIR not in sys.path:
    sys.path.insert(0, _MNT_DIR)
if _RADIANCE_DIR not in sys.path:
    sys.path.insert(0, _RADIANCE_DIR)

import importlib as _il
_cs_mod = _il.import_module("radiance.nodes.color.colorspace")

from radiance.nodes.color import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)
from radiance.nodes.color.colorspace import (
    _xy_to_XYZ,
    _build_bradford_matrix,
    _temperature_to_xy,
    _ILLUMINANT_XY,
    RadianceWhiteBalance,
)

# This module already gates its torch-dependent tests correctly (they skip
# cleanly against conftest's stub), so opt out of the automatic module-level
# skip and keep the rest of the file running on the no-torch CI matrix.
RADIANCE_TORCH_GATED = True



# ═════════════════════════════════════════════════════════════════════════════
# _xy_to_XYZ
# ═════════════════════════════════════════════════════════════════════════════

class TestXyToXYZ(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    def test_equal_energy_E(self):
        """E illuminant (1/3, 1/3) → XYZ = (1, 1, 1)."""
        xyz = _xy_to_XYZ((1/3, 1/3))
        self.assertAlmostEqual(float(xyz[1]), 1.0, places=4)   # Y = 1 by definition
        self.assertAlmostEqual(float(xyz[0]), float(xyz[1]), places=4)

    @skip_no_torch
    def test_output_shape(self):
        xyz = _xy_to_XYZ((0.3127, 0.3290))
        self.assertEqual(len(xyz), 3)

    def test_d65_y_is_one(self):
        x, y = _ILLUMINANT_XY["D65"]
        xyz = _xy_to_XYZ((x, y))
        self.assertAlmostEqual(float(xyz[1]), 1.0, places=5)

    def test_all_illuminants_positive(self):
        for name, xy in _ILLUMINANT_XY.items():
            xyz = _xy_to_XYZ(xy)
            for v in xyz:
                self.assertGreater(float(v), 0.0, f"Non-positive for {name}")


# ═════════════════════════════════════════════════════════════════════════════
# _build_bradford_matrix
# ═════════════════════════════════════════════════════════════════════════════

class TestBradfordMatrix(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    """Bradford matrix tests require real torch for actual matrix arithmetic."""

    @skip_no_torch
    def test_same_illuminant_is_identity(self):
        """D65→D65 Bradford matrix should be close to identity."""
        M = _build_bradford_matrix("D65", "D65")
        eye = torch.eye(3, dtype=M.dtype)
        self.assertTrue(torch.allclose(M, eye, atol=1e-4))

    @skip_no_torch
    def test_output_shape_3x3(self):
        M = _build_bradford_matrix("D65", "D50")
        self.assertEqual(M.shape, (3, 3))

    @skip_no_torch
    def test_matrix_is_invertible(self):
        M = _build_bradford_matrix("D65", "D50")
        det = float(torch.det(M.double()))
        self.assertGreater(abs(det), 0.1)

    @skip_no_torch
    def test_d65_to_d50_shifts_warm(self):
        """D65→D50 adaptation should warm the image (increase red relative to blue)."""
        M = _build_bradford_matrix("D65", "D50")
        grey = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        adapted = M @ grey
        self.assertGreater(float(adapted[0]), float(adapted[2]))

    @skip_no_torch
    def test_d65_to_d50_inverse_is_d50_to_d65(self):
        M_fwd = _build_bradford_matrix("D65", "D50").double()
        M_bwd = _build_bradford_matrix("D50", "D65").double()
        product = M_fwd @ M_bwd
        eye = torch.eye(3, dtype=torch.float64)
        self.assertTrue(torch.allclose(product, eye, atol=1e-4))

    def test_unknown_illuminant_no_crash(self):
        """Unknown illuminant name → uses D65 fallback without crashing."""
        M = _build_bradford_matrix("UNKNOWN", "D65")
        self.assertIsNotNone(M)


# ═════════════════════════════════════════════════════════════════════════════
# _temperature_to_xy
# ═════════════════════════════════════════════════════════════════════════════

class TestTemperatureToXY(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    def test_returns_tuple(self):
        xy = _temperature_to_xy(6500)
        self.assertIsInstance(xy, tuple)
        self.assertEqual(len(xy), 2)

    def test_d65_approx(self):
        """6500K should be close to D65 (0.3127, 0.3290)."""
        x, y = _temperature_to_xy(6500)
        self.assertAlmostEqual(x, 0.3127, delta=0.01)
        self.assertAlmostEqual(y, 0.3290, delta=0.01)

    def test_warm_is_redder(self):
        """Lower temperature → higher x (redder) chromaticity."""
        x_warm, _ = _temperature_to_xy(3200)
        x_cool, _ = _temperature_to_xy(7000)
        self.assertGreater(x_warm, x_cool)

    def test_values_in_visible_range(self):
        for t in [2000, 3000, 5000, 6500, 9000, 15000]:
            x, y = _temperature_to_xy(t)
            self.assertGreater(x, 0.0, f"x not positive at {t}K")
            self.assertGreater(y, 0.0, f"y not positive at {t}K")
            self.assertLess(x + y, 1.0, f"x+y ≥ 1 at {t}K")

    def test_clamping_low(self):
        """Very low temperature clamped to minimum → no crash."""
        x, y = _temperature_to_xy(100)
        self.assertIsNotNone(x)

    def test_clamping_high(self):
        x, y = _temperature_to_xy(100000)
        self.assertIsNotNone(x)


# ═════════════════════════════════════════════════════════════════════════════
# Node Registration
# ═════════════════════════════════════════════════════════════════════════════

class TestColorscienceRegistration(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    EXPECTED = [
        "RadianceWhiteBalance",
        "RadianceColorSpaceConvert",
        "RadianceACESTransform",
    ]

    def test_all_in_class_mappings(self):
        for name in self.EXPECTED:
            self.assertIn(name, NODE_CLASS_MAPPINGS)

    def test_all_in_display_mappings(self):
        for name in self.EXPECTED:
            self.assertIn(name, NODE_DISPLAY_NAME_MAPPINGS)

    def test_display_names_prefix(self):
        for _, disp in NODE_DISPLAY_NAME_MAPPINGS.items():
            self.assertTrue(disp.startswith("◎"))

    def test_input_types_present(self):
        for name, cls in NODE_CLASS_MAPPINGS.items():
            it = cls.INPUT_TYPES()
            self.assertIn("required", it, f"{name} missing required inputs")

    def test_functions_exist(self):
        for name, cls in NODE_CLASS_MAPPINGS.items():
            fn = cls.FUNCTION
            self.assertTrue(hasattr(cls, fn), f"{name}.{fn} not found")


if __name__ == "__main__":
    unittest.main(verbosity=2)
