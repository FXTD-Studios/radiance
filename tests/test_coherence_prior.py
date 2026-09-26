"""
tests/test_coherence_prior.py
─────────────────────────────
Synthetic tests for RadianceHDRCoherencePrior.

Strategy
────────
We manufacture frame sequences that are either perfectly coherent (no EV
variation) or deliberately flickery (large per-pixel EV jumps), then verify:

  1. The coherence map is ≥ 0 and ≤ 1 everywhere.
  2. Coherent regions produce coherence values CLOSE to 1.
  3. Incoherent (flickering) regions produce values CLOSE to 0.
  4. The guided latent is NOT identical to the input latent — noise was scaled.
  5. The coherence_map IMAGE output has the correct spatial size (lat_h × lat_w).
  6. Temporal mean of a steady sequence → coherence ≈ 1.0 everywhere.
  7. Half-frame flicker → coherence near 0 in flicker zone, near 1 elsewhere.

All tests run in CPU-only mode (no CUDA required).
"""
from __future__ import annotations

import unittest

import pytest
import torch

# This module already gates its torch-dependent tests correctly (they skip
# cleanly against conftest's stub), so opt out of the automatic module-level
# skip and keep the rest of the file running on the no-torch CI matrix.
RADIANCE_TORCH_GATED = True


# Real torch is required for actual tensor computations in these tests.
# The conftest stub makes nodes_hdr_smart importable but the stub's tensor
# operations return MagicMocks, which break all numerical assertions.
HAS_TORCH = hasattr(torch, "__version__")

# ── Conditional import ──────────────────────────────────────────────────────
try:
    from radiance.nodes_hdr_smart import RadianceHDRCoherencePrior
    HAS_NODE = HAS_TORCH   # meaningful only when real torch is present
except ImportError:
    HAS_NODE = False

# NOTE: pytestmark applied selectively to torch-dependent test classes only.
# Registration tests run without it.


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_frames(B: int, H: int, W: int, brightness: float | list) -> torch.Tensor:
    """
    Create (B, H, W, 3) float32 image tensor.

    If `brightness` is a scalar, all B frames share the same uniform value.
    If it is a list of length B, frame i has brightness[i] (uniform across pixels).
    """
    if isinstance(brightness, (int, float)):
        brightness = [brightness] * B
    assert len(brightness) == B
    frames = torch.stack([
        torch.full((H, W, 3), float(v), dtype=torch.float32)
        for v in brightness
    ])
    return frames  # (B, H, W, 3)


def _make_latent(B: int, C: int, lat_H: int, lat_W: int) -> dict:
    torch.manual_seed(0)
    return {"samples": torch.randn(B, C, lat_H, lat_W, dtype=torch.float32)}


def _run(frames, latent, sigma=0.5, noise_scale=0.8):
    node = RadianceHDRCoherencePrior()
    return node.inject(frames, latent, sigma=sigma, noise_scale=noise_scale)


# ─────────────────────────────────────────────────────────────────────────────
# Basic output contract
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_NODE,
    reason="radiance.nodes_hdr_smart not importable or real torch not available")
class TestOutputContract:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    def test_returns_two_outputs(self):
        frames  = _make_frames(4, 64, 64, 0.5)
        latent  = _make_latent(4, 4, 8, 8)
        result  = _run(frames, latent)
        assert len(result) == 2

    def test_latent_dict_preserved(self):
        """Output dict must still contain 'samples'."""
        frames  = _make_frames(3, 32, 32, 0.5)
        latent  = _make_latent(3, 4, 4, 4)
        guided, _ = _run(frames, latent)
        assert "samples" in guided

    def test_latent_shape_unchanged(self):
        B, C, lat_H, lat_W = 5, 16, 8, 8
        frames = _make_frames(B, 64, 64, 0.3)
        latent = _make_latent(B, C, lat_H, lat_W)
        guided, _ = _run(frames, latent)
        assert guided["samples"].shape == (B, C, lat_H, lat_W)

    def test_extra_latent_keys_pass_through(self):
        frames  = _make_frames(2, 32, 32, 0.5)
        latent  = _make_latent(2, 4, 4, 4)
        latent["noise_mask"] = torch.ones(2, 1, 4, 4)
        guided, _ = _run(frames, latent)
        assert "noise_mask" in guided

    def test_coherence_map_shape(self):
        """coherence_map must be (1, lat_H, lat_W, 3) for ComfyUI IMAGE."""
        frames  = _make_frames(4, 64, 64, 0.5)
        lat_H, lat_W = 8, 8
        latent  = _make_latent(4, 4, lat_H, lat_W)
        _, coh  = _run(frames, latent)
        assert coh.shape == (1, lat_H, lat_W, 3)

    def test_coherence_map_range(self):
        """All coherence values must lie in [0, 1]."""
        frames  = _make_frames(6, 48, 48, 0.5)
        latent  = _make_latent(6, 4, 6, 6)
        _, coh  = _run(frames, latent)
        assert float(coh.min()) >= 0.0 - 1e-6
        assert float(coh.max()) <= 1.0 + 1e-6

    def test_coherence_map_dtype_float(self):
        frames  = _make_frames(3, 32, 32, 0.5)
        latent  = _make_latent(3, 4, 4, 4)
        _, coh  = _run(frames, latent)
        assert coh.dtype == torch.float32


# ─────────────────────────────────────────────────────────────────────────────
# Coherence map values — steady vs flickery
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_NODE,
    reason="radiance.nodes_hdr_smart not importable or real torch not available")
class TestCoherenceValues:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    def test_steady_sequence_coherence_near_one(self):
        """
        A sequence where every frame is identical → zero EV std →
        coherence map should be ≈ 1 everywhere.
        """
        frames = _make_frames(8, 64, 64, brightness=0.5)
        latent = _make_latent(8, 4, 8, 8)
        _, coh = _run(frames, latent, sigma=0.5)
        mean_coh = float(coh.mean())
        assert mean_coh > 0.95, f"Expected coherence near 1, got {mean_coh:.4f}"

    def test_flickery_sequence_coherence_near_zero(self):
        """
        Alternating very bright / very dark frames → large EV std →
        coherence should be well below 0.5.
        """
        brightness = [0.01 if i % 2 == 0 else 4.0 for i in range(8)]
        frames = _make_frames(8, 64, 64, brightness=brightness)
        latent = _make_latent(8, 4, 8, 8)
        _, coh = _run(frames, latent, sigma=0.5)
        mean_coh = float(coh.mean())
        assert mean_coh < 0.3, f"Expected low coherence for flickery input, got {mean_coh:.4f}"

    def test_single_frame_coherence_is_one(self):
        """
        With only one frame, std = 0, coherence = 1.  Note: torch.std of a
        single sample is NaN/0 depending on correction; node clamps to 1e-6.
        """
        frames = _make_frames(1, 64, 64, brightness=0.5)
        latent = _make_latent(1, 4, 8, 8)
        _, coh = _run(frames, latent, sigma=0.5)
        mean_coh = float(coh.mean())
        assert mean_coh > 0.9, f"Single frame should have coherence≈1, got {mean_coh:.4f}"

    def test_high_sigma_tolerates_flicker(self):
        """
        With a very large sigma (3 EV), even large brightness swings
        should produce coherence > 0.5.
        """
        brightness = [0.05 if i % 2 == 0 else 2.0 for i in range(6)]
        frames = _make_frames(6, 64, 64, brightness=brightness)
        latent = _make_latent(6, 4, 8, 8)
        _, coh = _run(frames, latent, sigma=3.0)
        mean_coh = float(coh.mean())
        assert mean_coh > 0.5, f"High sigma should be tolerant; got {mean_coh:.4f}"

    def test_low_sigma_strict_on_mild_flicker(self):
        """
        sigma=0.1 is very strict — even a small brightness variation (0.45 vs 0.55)
        should drop coherence noticeably below 1.
        """
        brightness = [0.45 if i % 2 == 0 else 0.55 for i in range(6)]
        frames = _make_frames(6, 64, 64, brightness=brightness)
        latent = _make_latent(6, 4, 8, 8)
        _, coh = _run(frames, latent, sigma=0.1)
        mean_coh = float(coh.mean())
        # With sigma=0.1, even 0.3 EV variation should push coherence down
        assert mean_coh < 0.95, f"Tight sigma should penalise mild flicker; got {mean_coh:.4f}"


# ─────────────────────────────────────────────────────────────────────────────
# Noise scaling effect on latent
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_NODE,
    reason="radiance.nodes_hdr_smart not importable or real torch not available")
class TestNoiseScaling:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    def test_guided_latent_differs_from_input(self):
        """Noise gate must change the latent samples."""
        frames  = _make_frames(4, 64, 64, 0.5)
        latent  = _make_latent(4, 4, 8, 8)
        guided, _ = _run(frames, latent, noise_scale=0.8)
        assert not torch.allclose(guided["samples"], latent["samples"])

    def test_zero_noise_scale_leaves_latent_unchanged(self):
        """
        noise_scale=0 → noise_gate = 1.0 everywhere → latent unchanged.
        """
        frames  = _make_frames(4, 64, 64, 0.5)
        latent  = _make_latent(4, 4, 8, 8)
        guided, _ = _run(frames, latent, noise_scale=0.0)
        assert torch.allclose(guided["samples"], latent["samples"], atol=1e-6)

    def test_full_noise_scale_scales_coherent_region(self):
        """
        noise_scale=1.0 + steady frames → coherence≈1 → noise_gate≈0 →
        latent should be close to zero in magnitude.
        """
        frames  = _make_frames(8, 64, 64, 0.5)  # steady
        # Build a latent where all samples are 1.0 so we can measure scaling clearly
        latent  = {"samples": torch.ones(8, 4, 8, 8)}
        guided, _ = _run(frames, latent, noise_scale=1.0, sigma=0.5)
        mean_abs = float(guided["samples"].abs().mean())
        # coherence ≈ 1 → noise_gate ≈ 0 → samples ≈ 0
        assert mean_abs < 0.15, f"Full scale on steady sequence should suppress noise; got {mean_abs:.4f}"

    def test_guided_latent_element_wise_less_than_input(self):
        """
        When noise_scale > 0 and coherence > 0, every element of
        |guided| ≤ |input|  (noise_gate ∈ (0,1]).
        """
        torch.manual_seed(42)
        frames  = _make_frames(4, 64, 64, 0.5)
        samples = torch.rand(4, 4, 8, 8)   # positive values
        latent  = {"samples": samples.clone()}
        guided, _ = _run(frames, latent, noise_scale=0.5)
        assert (guided["samples"].abs() <= samples.abs() + 1e-6).all()


# ─────────────────────────────────────────────────────────────────────────────
# Spatial coherence map structure (half-frame flicker test)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_NODE,
    reason="radiance.nodes_hdr_smart not importable or real torch not available")
class TestSpatialCoherence:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    def test_half_frame_flicker_shows_spatial_structure(self):
        """
        Left half of frame flickers, right half is steady.
        coherence_map left mean < coherence_map right mean.
        """
        B, H, W = 6, 64, 64

        steady_val  = 0.5
        flicker_val = [0.05, 2.0, 0.05, 2.0, 0.05, 2.0]

        frames = torch.full((B, H, W, 3), steady_val)
        for i in range(B):
            # Left half flickers
            frames[i, :, : W // 2, :] = flicker_val[i]

        lat_H, lat_W = H // 8, W // 8
        latent = _make_latent(B, 4, lat_H, lat_W)
        _, coh = _run(frames, latent, sigma=0.5)

        # coh: (1, lat_H, lat_W, 3) — use channel 0
        coh_2d = coh[0, :, :, 0]  # (lat_H, lat_W)
        left_mean  = float(coh_2d[:, : lat_W // 2].mean())
        right_mean = float(coh_2d[:, lat_W // 2 :].mean())

        assert left_mean < right_mean, (
            f"Left (flickery) coherence {left_mean:.3f} should be < "
            f"right (steady) coherence {right_mean:.3f}"
        )
        assert right_mean > 0.7, f"Right half should be highly coherent; got {right_mean:.3f}"


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_NODE,
    reason="radiance.nodes_hdr_smart not importable or real torch not available")
class TestEdgeCases:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    def test_very_dark_frames_no_nan(self):
        """Near-zero luminance must not produce NaN in the coherence map."""
        frames  = _make_frames(4, 32, 32, brightness=1e-6)
        latent  = _make_latent(4, 4, 4, 4)
        guided, coh = _run(frames, latent)
        assert not torch.isnan(guided["samples"]).any()
        assert not torch.isnan(coh).any()

    def test_very_bright_frames_no_nan(self):
        """Very high luminance (HDR) must not produce NaN."""
        frames  = _make_frames(4, 32, 32, brightness=50.0)
        latent  = _make_latent(4, 4, 4, 4)
        guided, coh = _run(frames, latent)
        assert not torch.isnan(guided["samples"]).any()
        assert not torch.isnan(coh).any()

    def test_non_square_frame(self):
        """Different H and W must be handled correctly."""
        frames  = _make_frames(3, 48, 80, brightness=0.5)
        latent  = _make_latent(3, 4, 6, 10)
        guided, coh = _run(frames, latent)
        assert guided["samples"].shape == (3, 4, 6, 10)
        assert coh.shape == (1, 6, 10, 3)

    def test_many_channels_latent(self):
        """LTX-Video uses 128-channel latents."""
        frames  = _make_frames(4, 64, 64, brightness=0.5)
        latent  = _make_latent(4, 128, 8, 8)
        guided, _ = _run(frames, latent)
        assert guided["samples"].shape == (4, 128, 8, 8)

    def test_reproducibility_with_seed(self):
        """Same input must give same output (no internal randomness)."""
        frames = _make_frames(4, 32, 32, brightness=0.4)
        torch.manual_seed(7)
        latent1 = _make_latent(4, 4, 4, 4)
        torch.manual_seed(7)
        latent2 = _make_latent(4, 4, 4, 4)
        g1, c1 = _run(frames, latent1, sigma=0.5, noise_scale=0.8)
        g2, c2 = _run(frames, latent2, sigma=0.5, noise_scale=0.8)
        assert torch.allclose(g1["samples"], g2["samples"])
        assert torch.allclose(c1, c2)


# ─────────────────────────────────────────────────────────────────────────────
# Registration tests — run without torch
# ─────────────────────────────────────────────────────────────────────────────

class TestCoherencePriorRegistration(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    """Registration tests — run even without real torch."""

    def test_class_mappings_present(self):
        """RadianceHDRCoherencePrior must be in NODE_CLASS_MAPPINGS."""
        try:
            from radiance.nodes_hdr_smart import NODE_CLASS_MAPPINGS
            self.assertIn("RadianceHDRCoherencePrior", NODE_CLASS_MAPPINGS)
        except ImportError:
            self.skipTest("nodes_hdr_smart not importable")

    def test_display_names_present(self):
        """RadianceHDRCoherencePrior must be in NODE_DISPLAY_NAME_MAPPINGS."""
        try:
            from radiance.nodes_hdr_smart import NODE_DISPLAY_NAME_MAPPINGS
            self.assertIn("RadianceHDRCoherencePrior", NODE_DISPLAY_NAME_MAPPINGS)
        except ImportError:
            self.skipTest("nodes_hdr_smart not importable")

    def test_display_names_prefix(self):
        """Display name must start with ◎."""
        try:
            from radiance.nodes_hdr_smart import NODE_DISPLAY_NAME_MAPPINGS
            disp_name = NODE_DISPLAY_NAME_MAPPINGS["RadianceHDRCoherencePrior"]
            self.assertTrue(disp_name.startswith("◎"))
        except ImportError:
            self.skipTest("nodes_hdr_smart not importable")

    def test_functions_exist(self):
        """FUNCTION attribute must exist."""
        try:
            from radiance.nodes_hdr_smart import RadianceHDRCoherencePrior
            fn = RadianceHDRCoherencePrior.FUNCTION
            self.assertTrue(hasattr(RadianceHDRCoherencePrior, fn))
        except ImportError:
            self.skipTest("nodes_hdr_smart not importable")

    def test_input_types_have_required(self):
        """INPUT_TYPES() must have 'required' key."""
        try:
            from radiance.nodes_hdr_smart import RadianceHDRCoherencePrior
            it = RadianceHDRCoherencePrior.INPUT_TYPES()
            self.assertIn("required", it)
        except ImportError:
            self.skipTest("nodes_hdr_smart not importable")

    def test_return_types_defined(self):
        """RETURN_TYPES must be defined."""
        try:
            from radiance.nodes_hdr_smart import RadianceHDRCoherencePrior
            self.assertTrue(hasattr(RadianceHDRCoherencePrior, "RETURN_TYPES"))
            self.assertEqual(len(RadianceHDRCoherencePrior.RETURN_TYPES), 2)
        except ImportError:
            self.skipTest("nodes_hdr_smart not importable")
