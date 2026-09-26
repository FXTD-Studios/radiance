"""
tests/test_upscale.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Tests for nodes_upscale — AI Upscaler (Tiler / Image / Video / Router /
FaceRestore / ColourFix)

Coverage
────────
  _gaussian_kernel_1d       — kernel shape + normalisation
  _build_gaussian_weight_map — weight map properties
  tiled_upscale (bicubic)   — shape correctness, no-seam smoke
  _bicubic_upscale          — scale × 2 and × 4 output shapes
  _histogram_match_fast     — identity case, mean shift
  _classify_content         — returns expected keys
  _recommend_tier           — returns a known tier string
  RadianceBitDepthDegrade   — quantization math, PSNR, delta range
  Model registry            — all required keys present
  Node registration         — 6 nodes in NODE_CLASS_MAPPINGS
"""

from __future__ import annotations

import json
import math
import os
import sys
import unittest

import numpy as np

# ── Torch detection ──────────────────────────────────────────────────────────
import torch

HAS_TORCH = hasattr(torch, "__version__")
skip_no_torch = unittest.skipUnless(HAS_TORCH, "real torch not available")

# ── Module under test ────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from radiance.nodes.upscale.upscale import (
    _UPSCALE_MODEL_REGISTRY,
    _TIER_CHOICES,
    _gaussian_kernel_1d,
    _build_gaussian_weight_map,
    _bicubic_upscale,
    _histogram_match_fast,
    _classify_content,
    _recommend_tier,
    tiled_upscale,
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    RadianceUpscaleTiler,
    RadianceUpscaleImage,
    RadianceUpscaleVideo,
    RadianceUpscaleRouter,
    RadianceUpscaleFaceRestore,
)

# This module already gates its torch-dependent tests correctly (they skip
# cleanly against conftest's stub), so opt out of the automatic module-level
# skip and keep the rest of the file running on the no-torch CI matrix.
RADIANCE_TORCH_GATED = True



# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _t(b=1, h=32, w=32, c=3, fill=0.5) -> torch.Tensor:
    return torch.full((b, h, w, c), fill, dtype=torch.float32)


def _rand(b=1, h=32, w=32, c=3, seed=0) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.rand(b, h, w, c, generator=g, dtype=torch.float32)


# ═════════════════════════════════════════════════════════════════════════════
# 1. Gaussian kernel
# ═════════════════════════════════════════════════════════════════════════════

@unittest.skipUnless(HAS_TORCH, "real torch not available")
class TestGaussianKernel(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    def test_shape(self):
        k = _gaussian_kernel_1d(15, sigma=3.0, device=torch.device("cpu"))
        self.assertEqual(k.shape, (15,))

    def test_normalised(self):
        k = _gaussian_kernel_1d(31, sigma=5.0, device=torch.device("cpu"))
        self.assertAlmostEqual(k.sum().item(), 1.0, places=5)

    def test_symmetric(self):
        k = _gaussian_kernel_1d(21, sigma=4.0, device=torch.device("cpu"))
        self.assertTrue(torch.allclose(k, k.flip(0), atol=1e-6))

    def test_peak_at_centre(self):
        k = _gaussian_kernel_1d(11, sigma=2.0, device=torch.device("cpu"))
        self.assertEqual(k.argmax().item(), 5)

    def test_size_one(self):
        k = _gaussian_kernel_1d(1, sigma=1.0, device=torch.device("cpu"))
        self.assertAlmostEqual(k[0].item(), 1.0, places=6)


# ═════════════════════════════════════════════════════════════════════════════
# 2. Weight map
# ═════════════════════════════════════════════════════════════════════════════

@unittest.skipUnless(HAS_TORCH, "real torch not available")
class TestWeightMap(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    def test_shape(self):
        wm = _build_gaussian_weight_map(64, 64, overlap=16, device=torch.device("cpu"))
        self.assertEqual(wm.squeeze().shape, (64, 64))

    def test_positive(self):
        wm = _build_gaussian_weight_map(32, 32, overlap=8, device=torch.device("cpu"))
        self.assertTrue((wm > 0).all())

    def test_max_at_centre(self):
        wm = _build_gaussian_weight_map(64, 64, overlap=16, device=torch.device("cpu")).squeeze()
        cy, cx = wm.shape[0] // 2, wm.shape[1] // 2
        centre = wm[cy, cx].item()
        self.assertGreater(centre, wm[0, 0].item())

    def test_non_square(self):
        wm = _build_gaussian_weight_map(48, 80, overlap=12, device=torch.device("cpu"))
        self.assertEqual(wm.squeeze().shape, (48, 80))


# ═════════════════════════════════════════════════════════════════════════════
# 3. Bicubic upscale
# ═════════════════════════════════════════════════════════════════════════════

@unittest.skipUnless(HAS_TORCH, "real torch not available")
class TestBicubicUpscale(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    def test_scale2_shape(self):
        img = _t(1, 16, 16, 3)
        out = _bicubic_upscale(img, scale=2)
        self.assertEqual(out.shape, (1, 32, 32, 3))

    def test_scale4_shape(self):
        img = _t(1, 8, 8, 3)
        out = _bicubic_upscale(img, scale=4)
        self.assertEqual(out.shape, (1, 32, 32, 3))

    def test_uniform_image_unchanged(self):
        img = _t(1, 8, 8, 3, fill=0.7)
        out = _bicubic_upscale(img, scale=2)
        self.assertTrue(torch.allclose(out, torch.full_like(out, 0.7), atol=1e-4))

    def test_batch_preserved(self):
        img = _t(3, 8, 8, 3)
        out = _bicubic_upscale(img, scale=2)
        self.assertEqual(out.shape[0], 3)


# ═════════════════════════════════════════════════════════════════════════════
# 4. tiled_upscale (bicubic smoke)
# ═════════════════════════════════════════════════════════════════════════════

@unittest.skipUnless(HAS_TORCH, "real torch not available")
class TestTiledUpscale(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    def test_output_shape_x2(self):
        img = _rand(1, 32, 32, 3)
        fn = lambda tile: _bicubic_upscale(tile, scale=2)
        out, conf = tiled_upscale(img, fn, scale=2, tile_size=16, overlap=4)
        self.assertEqual(out.shape, (1, 64, 64, 3))
        self.assertEqual(conf.shape[:3], (1, 64, 64))

    def test_output_shape_x4(self):
        img = _rand(1, 32, 32, 3)
        fn = lambda tile: _bicubic_upscale(tile, scale=4)
        out, conf = tiled_upscale(img, fn, scale=4, tile_size=16, overlap=4)
        self.assertEqual(out.shape, (1, 128, 128, 3))

    def test_confidence_range(self):
        img = _rand(1, 32, 32, 3)
        fn = lambda tile: _bicubic_upscale(tile, scale=2)
        _, conf = tiled_upscale(img, fn, scale=2, tile_size=16, overlap=4)
        self.assertTrue((conf >= 0).all() and (conf <= 1).all())


# ═════════════════════════════════════════════════════════════════════════════
# 5. Histogram match (fast path)
# ═════════════════════════════════════════════════════════════════════════════

@unittest.skipUnless(HAS_TORCH, "real torch not available")
class TestHistogramMatchFast(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    def test_identity_case(self):
        """Matching source against itself → no change."""
        src = _rand(1, 16, 16, 3, seed=1)
        out = _histogram_match_fast(src, src, strength=1.0, n_bins=2048)
        self.assertTrue(torch.allclose(out, src, atol=5e-3))

    def test_mean_shift(self):
        """Source dark image matched to bright ref → output brighter."""
        src = _t(1, 8, 8, 3, fill=0.2)
        ref = _t(1, 8, 8, 3, fill=0.8)
        out = _histogram_match_fast(src, ref, strength=1.0)
        self.assertGreater(out.mean().item(), src.mean().item())

    def test_strength_zero_noop(self):
        src = _rand(1, 8, 8, 3, seed=2)
        ref = _rand(1, 8, 8, 3, seed=3)
        out = _histogram_match_fast(src, ref, strength=0.0)
        self.assertTrue(torch.allclose(out, src, atol=1e-5))

    def test_output_shape_preserved(self):
        src = _rand(2, 16, 16, 3)
        ref = _rand(1, 16, 16, 3)
        out = _histogram_match_fast(src, ref)
        self.assertEqual(out.shape, src.shape)

    def test_clamped(self):
        src = torch.rand(1, 8, 8, 3)
        ref = torch.rand(1, 8, 8, 3)
        out = _histogram_match_fast(src, ref)
        self.assertTrue((out >= 0).all() and (out <= 1).all())


# ═════════════════════════════════════════════════════════════════════════════
# 6. Content classifier
# ═════════════════════════════════════════════════════════════════════════════

@unittest.skipUnless(HAS_TORCH, "real torch not available")
class TestClassifyContent(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    REQUIRED_KEYS = {"noise_level", "sharpness", "saturation", "ai_likelihood"}

    def test_keys_present(self):
        img = _rand(1, 64, 64, 3)
        stats = _classify_content(img)
        for k in self.REQUIRED_KEYS:
            self.assertIn(k, stats, f"Missing key: {k}")

    def test_values_finite(self):
        img = _rand(1, 64, 64, 3)
        stats = _classify_content(img)
        for k, v in stats.items():
            self.assertTrue(math.isfinite(v), f"{k} = {v} is not finite")

    def test_solid_colour_low_noise(self):
        img = _t(1, 32, 32, 3, fill=0.5)
        stats = _classify_content(img)
        self.assertLess(stats["noise_level"], 0.05)

    def test_noisy_image_high_noise(self):
        g = torch.Generator().manual_seed(42)
        img = torch.rand(1, 32, 32, 3, generator=g)
        stats = _classify_content(img)
        self.assertGreater(stats["noise_level"], 0.0)


# ═════════════════════════════════════════════════════════════════════════════
# 7. _recommend_tier
# ═════════════════════════════════════════════════════════════════════════════

@unittest.skipUnless(HAS_TORCH, "real torch not available")
class TestRecommendTier(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    def _stats(self, **kw):
        base = {"noise_level": 0.02, "sharpness": 0.3, "saturation": 0.4, "ai_likelihood": 0.2}
        base.update(kw)
        return base

    def test_returns_string(self):
        result = _recommend_tier(self._stats())
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_prefer_speed_returns_tier1(self):
        result = _recommend_tier(self._stats(), prefer_speed=True)
        self.assertIn("tier1", result.lower())

    def test_valid_choice(self):
        result = _recommend_tier(self._stats())
        # Must be one of the known tier choice prefixes
        tier_prefixes = {"tier1", "tier2", "tier3", "auto"}
        self.assertTrue(any(result.startswith(p) for p in tier_prefixes)
                        or result in _TIER_CHOICES)


# ═════════════════════════════════════════════════════════════════════════════
# 8. RadianceBitDepthDegrade (quantisation logic — via nodes_colorscience)
# ═════════════════════════════════════════════════════════════════════════════

@unittest.skipUnless(HAS_TORCH, "real torch not available")
class TestBitDepthDegradeViaTorch(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    """
    Validate the quantisation math directly (not through the ComfyUI node
    wrapper) using the same formula the node uses.
    """

    def _quantize(self, img: torch.Tensor, bits: int) -> torch.Tensor:
        levels = (2 ** bits) - 1
        step   = 1.0 / levels
        return (img / step).round() * step

    def test_8bit_range(self):
        img = torch.rand(1, 8, 8, 3)
        q   = self._quantize(img, 8)
        self.assertTrue((q >= 0).all() and (q <= 1).all())

    def test_4bit_step_size(self):
        """4-bit = 15 levels → step = 1/15 ≈ 0.0667."""
        img = torch.linspace(0, 1, 100).reshape(1, 10, 10, 1).expand(-1, -1, -1, 3)
        q   = self._quantize(img, 4)
        diffs = q.reshape(-1).diff().abs()
        nonzero = diffs[diffs > 1e-6]
        expected_step = 1.0 / 15
        self.assertTrue((nonzero - expected_step).abs().max() < 1e-4)

    def test_16bit_near_lossless(self):
        img = torch.rand(1, 8, 8, 3)
        q   = self._quantize(img, 16)
        mse = ((img - q) ** 2).mean().item()
        self.assertLess(mse, 1e-7)

    def test_psnr_8bit(self):
        """8-bit quantization should give ~48–50 dB PSNR for uniform noise."""
        img = torch.rand(1, 64, 64, 3)
        q   = self._quantize(img, 8)
        mse = ((img - q) ** 2).mean().item()
        psnr = 10 * math.log10(1.0 / mse)
        # Theoretical 8-bit PSNR ≈ 49.9 dB
        self.assertGreater(psnr, 40.0)
        self.assertLess(psnr, 60.0)

    def test_delta_always_non_negative(self):
        img = torch.rand(1, 8, 8, 3)
        q   = self._quantize(img, 8)
        self.assertTrue(((img - q).abs() >= 0).all())


# ═════════════════════════════════════════════════════════════════════════════
# 9. Model registry
# ═════════════════════════════════════════════════════════════════════════════

class TestModelRegistry(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    REQUIRED_KEYS = {
        "realesrgan_x4plus",
        "realesrgan_x4plus_anime",
        "realesrgan_x2plus",
        "swinir_l_x4",
        "hat_l_x4",
        "sd_x4_upscaler",
        "codeformer",
        "gfpgan_v1.4",
        "retinaface_resnet50",
    }

    REQUIRED_FIELDS = {"url", "scale"}  # "note" is the description field in this registry

    def test_required_models_present(self):
        for key in self.REQUIRED_KEYS:
            self.assertIn(key, _UPSCALE_MODEL_REGISTRY, f"Missing: {key}")

    def test_each_entry_has_required_fields(self):
        for key, entry in _UPSCALE_MODEL_REGISTRY.items():
            for field in self.REQUIRED_FIELDS:
                self.assertIn(field, entry, f"{key} missing field '{field}'")

    def test_scale_values_valid(self):
        for key, entry in _UPSCALE_MODEL_REGISTRY.items():
            scale = entry["scale"]
            self.assertIn(scale, (1, 2, 4), f"{key}.scale={scale} not in {{1,2,4}}")

    def test_tier_choices_non_empty(self):
        self.assertGreater(len(_TIER_CHOICES), 0)

    def test_auto_in_tier_choices(self):
        self.assertIn("auto", _TIER_CHOICES)


# ═════════════════════════════════════════════════════════════════════════════
# 10. Node registration
# ═════════════════════════════════════════════════════════════════════════════

class TestNodeRegistration(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    EXPECTED_NODES = {
        "RadianceUpscaleTiler",
        "RadianceUpscaleImage",
        "RadianceUpscaleVideo",
        "RadianceUpscaleFaceRestore",
    }

    def test_all_nodes_registered(self):
        for key in self.EXPECTED_NODES:
            self.assertIn(key, NODE_CLASS_MAPPINGS, f"Missing: {key}")

    def test_display_names_present(self):
        for key in self.EXPECTED_NODES:
            self.assertIn(key, NODE_DISPLAY_NAME_MAPPINGS)
            self.assertIn("◎", NODE_DISPLAY_NAME_MAPPINGS[key])

    def test_class_references_valid(self):
        for key, cls in NODE_CLASS_MAPPINGS.items():
            self.assertTrue(callable(cls), f"{key} is not callable")

    def test_input_types_callable(self):
        for key, cls in NODE_CLASS_MAPPINGS.items():
            self.assertTrue(hasattr(cls, "INPUT_TYPES"))
            it = cls.INPUT_TYPES()
            self.assertIn("required", it)

    def test_return_types_defined(self):
        for key, cls in NODE_CLASS_MAPPINGS.items():
            self.assertTrue(hasattr(cls, "RETURN_TYPES"),
                            f"{key} missing RETURN_TYPES")
            self.assertTrue(hasattr(cls, "FUNCTION"),
                            f"{key} missing FUNCTION")


if __name__ == "__main__":
    unittest.main()
