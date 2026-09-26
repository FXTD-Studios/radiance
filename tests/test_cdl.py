"""
tests/test_cdl.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Tests for nodes_cdl — ASC CDL v1.2

Coverage
────────
  CDL math (pure)  : slope/offset/power/saturation formulas
  XML I/O          : CDL export/import round-trip via temp files
  JSON override    : cdl_data string overrides manual inputs
  Node registration
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

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
from radiance.nodes.color.cdl import (
    RadianceCDLTransform,
    RadianceCDLImport,
    RadianceCDLExport,
)

# This module already gates its torch-dependent tests correctly (they skip
# cleanly against conftest's stub), so opt out of the automatic module-level
# skip and keep the rest of the file running on the no-torch CI matrix.
RADIANCE_TORCH_GATED = True



def _img(b=1, h=4, w=4, fill=0.5):
    return torch.full((b, h, w, 3), fill, dtype=torch.float32)


# ═════════════════════════════════════════════════════════════════════════════
# CDL Math (torch-gated)
# ═════════════════════════════════════════════════════════════════════════════

class TestCDLMath(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    @skip_no_torch
    def test_identity_passthrough(self):
        node = RadianceCDLTransform()
        img = _img(fill=0.4)
        out, info = node.apply(img, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1.0)
        self.assertTrue(torch.allclose(out, img, atol=1e-5))

    @skip_no_torch
    def test_slope_multiplies(self):
        node = RadianceCDLTransform()
        img = _img(fill=0.4)
        out, _ = node.apply(img, 2, 2, 2, 0, 0, 0, 1, 1, 1, 1.0)
        self.assertAlmostEqual(float(out.mean()), 0.8, places=4)

    @skip_no_torch
    def test_offset_shifts(self):
        node = RadianceCDLTransform()
        img = _img(fill=0.3)
        out, _ = node.apply(img, 1, 1, 1, 0.1, 0.1, 0.1, 1, 1, 1, 1.0)
        self.assertAlmostEqual(float(out.mean()), 0.4, places=4)

    @skip_no_torch
    def test_power_gamma(self):
        node = RadianceCDLTransform()
        img = _img(fill=0.5)
        out, _ = node.apply(img, 1, 1, 1, 0, 0, 0, 2, 2, 2, 1.0)
        self.assertAlmostEqual(float(out.mean()), 0.25, places=4)

    @skip_no_torch
    def test_saturation_zero_is_grey(self):
        node = RadianceCDLTransform()
        img = torch.rand(1, 4, 4, 3)
        out, _ = node.apply(img, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0.0)
        # All channels equal (luma only)
        self.assertTrue(torch.allclose(out[..., 0], out[..., 1], atol=1e-4))
        self.assertTrue(torch.allclose(out[..., 1], out[..., 2], atol=1e-4))

    @skip_no_torch
    def test_negative_offset_clamped(self):
        """Negative slope+offset → clamp to 0 before power."""
        node = RadianceCDLTransform()
        img = _img(fill=0.1)
        out, _ = node.apply(img, 1, 1, 1, -0.5, -0.5, -0.5, 1, 1, 1, 1.0)
        self.assertEqual(float(out.min()), 0.0)

    @skip_no_torch
    def test_cdl_info_json_valid(self):
        node = RadianceCDLTransform()
        img = _img()
        _, info = node.apply(img, 1.2, 1.1, 0.9, 0.05, 0, -0.05, 1, 1, 1, 0.9)
        data = json.loads(info)
        self.assertIn("slope", data)
        self.assertIn("saturation", data)

    @skip_no_torch
    def test_cdl_data_json_overrides(self):
        """cdl_data JSON string should override manual inputs."""
        node = RadianceCDLTransform()
        img = _img(fill=0.5)
        cdl = json.dumps({"slope": [2.0, 2.0, 2.0], "offset": [0.0, 0.0, 0.0],
                          "power": [1.0, 1.0, 1.0], "saturation": 1.0})
        out, _ = node.apply(img, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1.0, cdl_data=cdl)
        self.assertAlmostEqual(float(out.mean()), 1.0, places=4)

    @skip_no_torch
    def test_per_channel_slope(self):
        """Different slope per channel → different output means."""
        node = RadianceCDLTransform()
        img = _img(fill=0.5)
        out, _ = node.apply(img, 2, 1, 0.5, 0, 0, 0, 1, 1, 1, 1.0)
        r_mean = float(out[..., 0].mean())
        g_mean = float(out[..., 1].mean())
        b_mean = float(out[..., 2].mean())
        self.assertGreater(r_mean, g_mean)
        self.assertGreater(g_mean, b_mean)


# ═════════════════════════════════════════════════════════════════════════════
# XML I/O — CDL Export / Import round-trip (no torch needed)
# ═════════════════════════════════════════════════════════════════════════════

class TestCDLXMLRoundTrip(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    def _write_cdl(self, path, slope, offset, power, saturation):
        node = RadianceCDLExport()
        node.save(path, *slope, *offset, *power, saturation)

    def test_export_creates_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "out.cdl")
            RadianceCDLExport().save(path, 1.1, 1.0, 0.9, 0.02, 0, -0.02, 1, 1, 1, 0.95)
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 0)

    def test_export_valid_xml(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test.cdl")
            RadianceCDLExport().save(path, 1.1, 1.0, 0.9, 0.02, 0, -0.02, 1, 1, 1, 0.95)
            tree = ET.parse(path)
            root = tree.getroot()
            self.assertIsNotNone(root)

    def test_round_trip_slope(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "rt.cdl")
            slope = [1.2, 0.9, 1.1]
            offset = [0.05, -0.02, 0.03]
            power = [1.0, 0.95, 1.05]
            sat = 0.85

            RadianceCDLExport().save(path, *slope, *offset, *power, sat)
            result = RadianceCDLImport().load(path)
            data = json.loads(result[0])

            self.assertAlmostEqual(data["slope"][0], slope[0], places=4)
            self.assertAlmostEqual(data["slope"][1], slope[1], places=4)
            self.assertAlmostEqual(data["slope"][2], slope[2], places=4)

    def test_round_trip_offset(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "rt2.cdl")
            offset = [0.07, -0.03, 0.05]
            RadianceCDLExport().save(path, 1, 1, 1, *offset, 1, 1, 1, 1.0)
            result = RadianceCDLImport().load(path)
            data = json.loads(result[0])
            for i in range(3):
                self.assertAlmostEqual(data["offset"][i], offset[i], places=4)

    def test_round_trip_saturation(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "rt3.cdl")
            RadianceCDLExport().save(path, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0.72)
            result = RadianceCDLImport().load(path)
            data = json.loads(result[0])
            self.assertAlmostEqual(data["saturation"], 0.72, places=4)

    def test_import_missing_file_returns_identity(self):
        result = RadianceCDLImport().load("/does/not/exist.cdl")
        data = json.loads(result[0])
        # Should return identity-ish defaults (empty dict or defaults)
        # The node returns empty dict on error
        self.assertIsInstance(result, tuple)

    def test_round_trip_individual_float_outputs(self):
        """Import returns (cdl_data, sr, sg, sb, or, og, ob, pr, pg, pb, sat)."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "fl.cdl")
            RadianceCDLExport().save(path, 1.3, 1.0, 0.8, 0.04, 0.0, -0.04, 1.1, 1.0, 0.9, 0.88)
            result = RadianceCDLImport().load(path)
            self.assertEqual(len(result), 11)
            self.assertAlmostEqual(result[1], 1.3, places=3)   # slope_r
            self.assertAlmostEqual(result[11-1], 0.88, places=3)  # saturation

    def test_export_with_cdl_data_override(self):
        """cdl_data JSON string overrides manual inputs in Export too."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ov.cdl")
            cdl = json.dumps({"slope": [1.5, 1.5, 1.5], "offset": [0.1, 0.1, 0.1],
                              "power": [1.0, 1.0, 1.0], "saturation": 0.75})
            RadianceCDLExport().save(path, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1.0, cdl_data=cdl)
            result = RadianceCDLImport().load(path)
            data = json.loads(result[0])
            self.assertAlmostEqual(data["slope"][0], 1.5, places=3)
            self.assertAlmostEqual(data["saturation"], 0.75, places=3)

    def test_export_creates_parent_dir(self):
        """Export should create intermediate directories."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "deep", "nested", "out.cdl")
            RadianceCDLExport().save(path, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1.0)
            self.assertTrue(os.path.exists(path))


# ═════════════════════════════════════════════════════════════════════════════
# Node Registration
# ═════════════════════════════════════════════════════════════════════════════

class TestCDLRegistration(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    EXPECTED = ["RadianceCDLTransform", "RadianceCDLImport", "RadianceCDLExport"]

    def test_all_in_class_mappings(self):
        for name in self.EXPECTED:
            self.assertIn(name, NODE_CLASS_MAPPINGS)

    def test_all_in_display_mappings(self):
        for name in self.EXPECTED:
            self.assertIn(name, NODE_DISPLAY_NAME_MAPPINGS)

    def test_display_names_prefix(self):
        for _, display in NODE_DISPLAY_NAME_MAPPINGS.items():
            self.assertTrue(display.startswith("◎"))

    def test_functions_exist(self):
        for name, cls in NODE_CLASS_MAPPINGS.items():
            fn = getattr(cls, "FUNCTION", None)
            self.assertIsNotNone(fn)
            self.assertTrue(hasattr(cls, fn), f"{name}.{fn} not found")


if __name__ == "__main__":
    unittest.main(verbosity=2)
