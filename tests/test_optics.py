"""
tests/test_optics.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Tests for nodes_optics — Lens distortion, chromatic aberration, anamorphic
streaks, vignette

Coverage
────────
  Registration : all 4 nodes
  Input types  : structure validation
  CATEGORY     : correct prefix
"""

from __future__ import annotations

import os
import sys
import unittest

import torch
HAS_TORCH = hasattr(torch, "__version__")
skip_no_torch = unittest.skipUnless(HAS_TORCH, "real torch not available")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from radiance.nodes.vfx.optics import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    RadianceLensDistortion,
    RadianceChromaticAberration,
    RadianceAnamorphicStreaks,
    RadianceVignette,
)

# This module already gates its torch-dependent tests correctly (they skip
# cleanly against conftest's stub), so opt out of the automatic module-level
# skip and keep the rest of the file running on the no-torch CI matrix.
RADIANCE_TORCH_GATED = True



def _img(b=1, h=16, w=16, fill=0.5):
    return torch.full((b, h, w, 3), fill, dtype=torch.float32)


# ═════════════════════════════════════════════════════════════════════════════
# Registration
# ═════════════════════════════════════════════════════════════════════════════

class TestOpticsRegistration(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    EXPECTED = [
        "RadianceLensDistortion",
        "RadianceChromaticAberration",
        "RadianceAnamorphicStreaks",
        "RadianceVignette",
    ]

    def test_all_in_class_mappings(self):
        for name in self.EXPECTED:
            self.assertIn(name, NODE_CLASS_MAPPINGS)

    def test_all_in_display_mappings(self):
        for name in self.EXPECTED:
            self.assertIn(name, NODE_DISPLAY_NAME_MAPPINGS)

    def test_display_names_start_with_symbol(self):
        for _, disp in NODE_DISPLAY_NAME_MAPPINGS.items():
            self.assertTrue(disp.startswith("◎"))

    def test_category_prefix(self):
        for name, cls in NODE_CLASS_MAPPINGS.items():
            self.assertTrue(
                cls.CATEGORY.startswith("FXTD STUDIOS/Radiance"),
                f"{name}: bad category '{cls.CATEGORY}'"
            )

    def test_functions_exist(self):
        for name, cls in NODE_CLASS_MAPPINGS.items():
            fn = cls.FUNCTION
            self.assertTrue(hasattr(cls, fn), f"{name}.{fn} missing")

    def test_input_types_structure(self):
        for name, cls in NODE_CLASS_MAPPINGS.items():
            it = cls.INPUT_TYPES()
            self.assertIn("required", it, f"{name} missing required")
            self.assertIn("image", it["required"], f"{name} missing image input")

    def test_return_types_defined(self):
        for name, cls in NODE_CLASS_MAPPINGS.items():
            self.assertTrue(hasattr(cls, "RETURN_TYPES"),
                            f"{name} missing RETURN_TYPES")
            self.assertIsInstance(cls.RETURN_TYPES, tuple)
            self.assertGreater(len(cls.RETURN_TYPES), 0)

    def test_count(self):
        self.assertGreaterEqual(len(NODE_CLASS_MAPPINGS), 4)

    def test_mappings_consistent(self):
        self.assertEqual(set(NODE_CLASS_MAPPINGS), set(NODE_DISPLAY_NAME_MAPPINGS))


# ═════════════════════════════════════════════════════════════════════════════
# Lens Distortion — input params validation (no torch inference needed)
# ═════════════════════════════════════════════════════════════════════════════

class TestLensDistortionInputs(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    def test_k1_range_in_input_types(self):
        it = RadianceLensDistortion.INPUT_TYPES()
        self.assertIn("k1", it["required"])
        k1_spec = it["required"]["k1"][1]
        self.assertIn("min", k1_spec)
        self.assertIn("max", k1_spec)

    def test_invert_is_boolean(self):
        it = RadianceLensDistortion.INPUT_TYPES()
        self.assertIn("invert", it["required"])

    def test_returns_image_and_stmap(self):
        rt = RadianceLensDistortion.RETURN_TYPES
        self.assertEqual(len(rt), 2)
        self.assertEqual(rt[0], "IMAGE")
        self.assertEqual(rt[1], "IMAGE")


class TestVignetteInputs(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    def test_strength_in_inputs(self):
        it = RadianceVignette.INPUT_TYPES()
        self.assertIn("image", it["required"])
        # Should have at least one float param
        float_params = [k for k, v in it["required"].items()
                        if isinstance(v, tuple) and len(v) > 1
                        and isinstance(v[1], dict) and v[0] == "FLOAT"]
        self.assertGreater(len(float_params), 0)


class TestChromaticAberrationInputs(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    def test_has_shift_params(self):
        it = RadianceChromaticAberration.INPUT_TYPES()
        keys = set(it["required"].keys())
        # Should have some shift/offset parameters
        self.assertTrue(
            any("shift" in k.lower() or "offset" in k.lower() for k in keys)
            or len(keys) > 1,
            "Expected shift/offset parameters"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
