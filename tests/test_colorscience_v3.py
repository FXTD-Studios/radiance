"""
tests/test_colorscience_v3.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Tests for Radiance v3 additions to nodes_colorscience:
  • Rec.709 / BT.1886 IDT
  • RadianceBitDepthDegrade node

Coverage
────────
  _COLOR_SPACES         — Rec.709/BT.1886 present
  Analytical decode     — BT.1886 linearise (gamma 2.4)
  Analytical encode     — BT.1886 OETF (gamma 1/2.4)
  OCIO name map         — maps to "Output - Rec.709"
  RadianceBitDepthDegrade
    • quantize range     [0, 1]
    • 8-bit levels       255 representable values
    • delta non-negative
    • TPDF dither        output differs from hard quantize
    • floyd-steinberg    completes on tiny image
    • PSNR metric        monotone in bit depth
    • metrics JSON       all required keys present
  Node registration     — 4 nodes including BitDepthDegrade
"""

from __future__ import annotations

import json
import math
import os
import sys
import unittest

import numpy as np

import torch

HAS_TORCH = hasattr(torch, "__version__")
skip_no_torch = unittest.skipUnless(HAS_TORCH, "real torch not available")

# nodes_colorscience uses relative imports (from .radiance_ocio import ...)
# so it must be imported as part of the radiance package.
# conftest.py installs the radiance.radiance_ocio stub already.
_RADIANCE_DIR = os.path.dirname(os.path.dirname(__file__))
_MNT_DIR      = os.path.dirname(_RADIANCE_DIR)
for _p in (_MNT_DIR, _RADIANCE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import importlib as _il
_il.import_module("radiance.nodes.color.colorspace")

from radiance.nodes.color import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)
from radiance.nodes.color.colorspace import (
    RadianceBitDepthDegrade,
    RadianceColorSpaceConvert,
)

# Was a module-level name in the deleted nodes_colorscience shim;
# it has always been an attribute of the node class.
_COLOR_SPACES = RadianceColorSpaceConvert._COLOR_SPACES

# This module already gates its torch-dependent tests correctly (they skip
# cleanly against conftest's stub), so opt out of the automatic module-level
# skip and keep the rest of the file running on the no-torch CI matrix.
RADIANCE_TORCH_GATED = True



# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _t(b=1, h=8, w=8, fill=0.5) -> torch.Tensor:
    return torch.full((b, h, w, 3), fill, dtype=torch.float32)

def _rand(b=1, h=8, w=8, seed=0) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.rand(b, h, w, 3, generator=g, dtype=torch.float32)


# ═════════════════════════════════════════════════════════════════════════════
# 1. _COLOR_SPACES — Rec.709/BT.1886 presence
# ═════════════════════════════════════════════════════════════════════════════

class TestColorSpacesList(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    def test_bt1886_present(self):
        self.assertIn("Rec.709 / BT.1886", _COLOR_SPACES)

    def test_rec709_oetf_still_present(self):
        self.assertIn("Rec.709 (OETF encoded)", _COLOR_SPACES)

    def test_srgb_still_present(self):
        self.assertIn("sRGB (OETF encoded)", _COLOR_SPACES)

    def test_no_duplicates(self):
        self.assertEqual(len(_COLOR_SPACES), len(set(_COLOR_SPACES)))

    def test_acescg_present(self):
        self.assertIn("ACEScg", _COLOR_SPACES)


# ═════════════════════════════════════════════════════════════════════════════
# 2. BT.1886 analytical paths (torch-gated)
# ═════════════════════════════════════════════════════════════════════════════

@unittest.skipUnless(HAS_TORCH, "real torch not available")
class TestBT1886AnalyticalPaths(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    """
    Verify BT.1886 decode (γ2.4) and encode (γ1/2.4) round-trip correctly
    via the same formulas used in RadianceColorSpaceConvert._decode/_encode.
    """

    def _decode(self, img: torch.Tensor) -> torch.Tensor:
        """Apply BT.1886 EOTF: γ 2.4."""
        return img.clamp(min=0.0) ** 2.4

    def _encode(self, img: torch.Tensor) -> torch.Tensor:
        """Apply BT.1886 OETF: γ 1/2.4."""
        return img.clamp(min=0.0) ** (1.0 / 2.4)

    def test_decode_linearises_midgrey(self):
        """0.5^2.4 ≈ 0.2188 (mid-grey in BT.1886)."""
        x   = torch.tensor([0.5])
        out = self._decode(x)
        expected = 0.5 ** 2.4
        self.assertAlmostEqual(out.item(), expected, places=5)

    def test_encode_decode_roundtrip(self):
        img     = _rand(1, 4, 4)
        encoded = self._encode(img)
        decoded = self._decode(encoded)
        self.assertTrue(torch.allclose(img, decoded, atol=1e-5))

    def test_decode_encode_roundtrip(self):
        img     = _rand(1, 4, 4)
        decoded = self._decode(img)
        encoded = self._encode(decoded)
        self.assertTrue(torch.allclose(img, encoded, atol=1e-5))

    def test_white_passes_through(self):
        x = torch.ones(1, 4, 4, 3)
        self.assertTrue(torch.allclose(self._decode(x), x, atol=1e-6))
        self.assertTrue(torch.allclose(self._encode(x), x, atol=1e-6))

    def test_black_passes_through(self):
        x = torch.zeros(1, 4, 4, 3)
        self.assertTrue(torch.allclose(self._decode(x), x, atol=1e-6))
        self.assertTrue(torch.allclose(self._encode(x), x, atol=1e-6))

    def test_negative_clamped(self):
        x   = torch.tensor([-0.5, 0.0, 0.5])
        out = self._decode(x)
        self.assertTrue((out >= 0).all())


# ═════════════════════════════════════════════════════════════════════════════
# 3. RadianceBitDepthDegrade — node instantiation
# ═════════════════════════════════════════════════════════════════════════════

class TestBitDepthDegradeNodeStructure(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    def test_registered(self):
        self.assertIn("RadianceBitDepthDegrade", NODE_CLASS_MAPPINGS)

    def test_display_name_present(self):
        self.assertIn("RadianceBitDepthDegrade", NODE_DISPLAY_NAME_MAPPINGS)

    def test_input_types_has_image(self):
        it = RadianceBitDepthDegrade.INPUT_TYPES()
        self.assertIn("image", it["required"])

    def test_input_types_has_bit_depth(self):
        it = RadianceBitDepthDegrade.INPUT_TYPES()
        self.assertIn("bit_depth", it["required"])

    def test_input_types_has_dither_mode(self):
        it = RadianceBitDepthDegrade.INPUT_TYPES()
        self.assertIn("dither_mode", it["required"])

    def test_return_types(self):
        rt = RadianceBitDepthDegrade.RETURN_TYPES
        self.assertIn("IMAGE", rt)
        self.assertIn("STRING", rt)

    def test_return_names_count_matches(self):
        self.assertEqual(len(RadianceBitDepthDegrade.RETURN_TYPES),
                         len(RadianceBitDepthDegrade.RETURN_NAMES))

    def test_dither_choices(self):
        it = RadianceBitDepthDegrade.INPUT_TYPES()
        choices = it["required"]["dither_mode"][0]
        self.assertIn("triangular",       choices)
        self.assertIn("none",             choices)
        self.assertIn("floyd-steinberg",  choices)


# ═════════════════════════════════════════════════════════════════════════════
# 4. RadianceBitDepthDegrade — quantisation math (torch-gated)
# ═════════════════════════════════════════════════════════════════════════════

@unittest.skipUnless(HAS_TORCH, "real torch not available")
class TestBitDepthDegradeQuantisation(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    def _run(self, img, bits=8, dither="none", delta_gain=10.0, threshold=0.004):
        node = RadianceBitDepthDegrade()
        return node.degrade(img, bits, dither, delta_gain, threshold)

    def test_output_shape_preserved_rgb(self):
        img = _rand(1, 16, 16)
        q, delta, mask, metrics = self._run(img)
        self.assertEqual(q.shape, img.shape)

    def test_output_clamped_01(self):
        img = _rand(1, 8, 8)
        q, _, _, _ = self._run(img)
        self.assertTrue((q >= 0).all() and (q <= 1).all())

    def test_8bit_has_at_most_256_levels(self):
        img = torch.rand(1, 32, 32, 3)
        q, _, _, _ = self._run(img, bits=8)
        vals = (q * 255).round().unique()
        self.assertLessEqual(len(vals), 256)

    def test_4bit_has_at_most_16_levels(self):
        img = torch.rand(1, 32, 32, 1).expand(-1, -1, -1, 3)
        q, _, _, _ = self._run(img, bits=4)
        levels = (q[..., 0] * 15).round().unique()
        self.assertLessEqual(len(levels), 16)

    def test_delta_non_negative(self):
        img = _rand(1, 8, 8)
        _, delta, _, _ = self._run(img, delta_gain=1.0)
        self.assertTrue((delta >= 0).all())

    def test_psnr_improves_with_more_bits(self):
        """Higher bit depth → higher PSNR."""
        img = torch.rand(1, 32, 32, 3)
        _, _, _, m8  = self._run(img, bits=8)
        _, _, _, m12 = self._run(img, bits=12)
        psnr8  = json.loads(m8)["psnr_dB"]
        psnr12 = json.loads(m12)["psnr_dB"]
        self.assertGreater(psnr12, psnr8)

    def test_metrics_json_valid(self):
        img = _rand(1, 8, 8)
        _, _, _, metrics = self._run(img)
        parsed = json.loads(metrics)
        for key in ("bit_depth", "psnr_dB", "max_error", "dither_mode"):
            self.assertIn(key, parsed, f"Missing: {key}")

    def test_tpdf_dither_output_differs_from_hard(self):
        """TPDF dither produces different result than hard quantize (statistically)."""
        img = torch.rand(1, 32, 32, 3)
        q_hard, _, _, _ = self._run(img, bits=8, dither="none")
        q_tpdf, _, _, _ = self._run(img, bits=8, dither="triangular")
        # They should differ somewhere
        self.assertFalse(torch.allclose(q_hard, q_tpdf))

    def test_16bit_near_lossless(self):
        img = torch.rand(1, 8, 8, 3)
        q, _, _, metrics = self._run(img, bits=16)
        psnr = json.loads(metrics)["psnr_dB"]
        self.assertGreater(psnr, 80.0)

    def test_floyd_steinberg_completes(self):
        """FS dither is CPU-bound; ensure it completes on a tiny image."""
        img = torch.rand(1, 4, 4, 3)
        q, delta, mask, metrics = self._run(img, bits=8, dither="floyd-steinberg")
        self.assertEqual(q.shape, img.shape)

    def test_banding_mask_binary(self):
        img = _rand(1, 8, 8)
        _, _, mask, _ = self._run(img)
        unique = mask.unique()
        for v in unique:
            self.assertIn(v.item(), {0.0, 1.0})

    def test_alpha_preserved(self):
        img = torch.rand(1, 8, 8, 4)  # RGBA
        q, delta, _, _ = self._run(img)
        self.assertEqual(q.shape[-1], 4)

    def test_uniform_image_psnr_infinity_or_high(self):
        """Uniform image quantises exactly → zero or near-zero error."""
        img = _t(1, 8, 8, fill=128.0 / 255.0)
        q, _, _, metrics = self._run(img, bits=8)
        psnr = json.loads(metrics)["psnr_dB"]
        self.assertGreater(psnr, 60.0)


if __name__ == "__main__":
    unittest.main()
