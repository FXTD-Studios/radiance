import sys
import os
import unittest
import pytest

# Skip the whole file if real torch is not installed.
# The conftest.py installs a MagicMock stub; check for a real version string.
try:
    import torch as _t
    _HAS_TORCH = isinstance(getattr(_t, "__version__", None), str)
except ImportError:
    _HAS_TORCH = False

if not _HAS_TORCH:
    pytest.skip("torch not installed", allow_module_level=True)

import torch

# Add parent directory to sys.path to import radiance modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Augment conftest's comfy stub; never replace it.
#
# This file used to assign `sys.modules['comfy']` wholesale. That worked only
# because it then imported the sampler by bare name and got a private copy of
# the module which captured the replacement. Now that the sampler is a package
# module, replacing the stub tears it out from under every module that had
# already bound to it — `comfy.model_management.get_free_memory` vanishes and
# sixteen nodes stop importing. Add only what this file needs, on top of
# whatever conftest already installed.
import types as _types  # noqa: E402

from torch_mock import MockModel, MockModelSampling, MockSampleModule, MockModelManagement  # noqa: E402

import comfy  # installed by conftest  # noqa: E402
import comfy.sample  # noqa: E402
import comfy.samplers  # noqa: E402

if not hasattr(comfy.samplers, "calculate_sigmas"):
    comfy.samplers.calculate_sigmas = lambda m, s, t: torch.linspace(1.0, 0.0, t + 1)
if not hasattr(comfy, "model_base"):
    comfy.model_base = _types.SimpleNamespace(
        ModelType=type("enum", (), {"FLOW_LTX_AV": "FLOW_LTX_AV"})
    )

from radiance.nodes.generate.sampler import RadianceSamplerPro, get_flux_sigmas

class TestSamplerRegression(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    def test_flux_shift_schedule_denoise_1(self):
        """[C1] Verify get_flux_sigmas applies shift when denoise == 1.0"""
        model = MockModel(model_type="flux")
        steps = 20
        denoise = 1.0
        
        sigmas = get_flux_sigmas(model, "simple", steps, denoise, shift=1.15)
        
        # If the fix for C1 is working, sigmas should have been shifted.
        # Our MockModelSampling.shift returns x * 1.15.
        # If it didn't shift, it would just be torch.linspace(1, 0, steps+1).
        
        base_sigmas = torch.linspace(1.0, 0.0, steps + 1)
        self.assertFalse(torch.allclose(sigmas, base_sigmas), "Sigmas should have been shifted (different from linear)")

    def test_sd_turbo_sigmas_mirrors_comfyui_node(self):
        """get_sd_turbo_sigmas mirrors ComfyUI's own SDTurboScheduler node exactly
        (comfy_extras/nodes_custom_sampler.py) -- 10 fixed discrete timesteps
        (99, 199, ..., 999), picked from the end as denoise decreases."""
        from sampler_utils import get_sd_turbo_sigmas

        class _FakeModelSampling:
            def sigma(self, timesteps):
                return timesteps.float() / 999.0

        class _FakeModel:
            def get_model_object(self, name):
                return _FakeModelSampling()

        # denoise=1.0 -> start_step=0 -> first (highest-noise) timestep = 999
        sigmas = get_sd_turbo_sigmas(_FakeModel(), steps=1, denoise=1.0)
        self.assertEqual(len(sigmas), 2)  # 1 step + terminal zero
        self.assertAlmostEqual(sigmas[0].item(), 1.0, places=5)
        self.assertEqual(sigmas[-1].item(), 0.0)

        # Lower denoise starts further into the schedule -- lower initial noise.
        sigmas_full = get_sd_turbo_sigmas(_FakeModel(), steps=4, denoise=1.0)
        sigmas_half = get_sd_turbo_sigmas(_FakeModel(), steps=2, denoise=0.5)
        self.assertLess(sigmas_half[0].item(), sigmas_full[0].item())

    def test_noise_reproducibility(self):
        """Verify all noise generators are reproducible with the same seed."""
        shape = (1, 4, 32, 32)
        seed = 42
        device = torch.device("cpu")
        dtype = torch.float32
        
        # Internal noise functions from nodes_sampler
        from radiance.nodes.generate.sampler import (
            _perlin_noise, _spectral_noise, _brownian_noise, 
            _simplex_noise, _voronoi_noise, _curl_noise
        )
        
        noise_funcs = {
            "perlin": _perlin_noise,
            "spectral": _spectral_noise,
            "brownian": _brownian_noise,
            "simplex": _simplex_noise,
            "voronoi": _voronoi_noise,
            "curl": _curl_noise
        }
        
        for name, func in noise_funcs.items():
            with self.subTest(noise_type=name):
                # Run twice with same seed
                noise1 = func(shape, device, seed=seed)
                noise2 = func(shape, device, seed=seed)
                
                # Assert equality
                self.assertTrue(torch.allclose(noise1, noise2), f"Noise {name} should be reproducible with seed {seed}")
                
                # Run with different seed
                noise3 = func(shape, device, seed=seed+1)
                self.assertFalse(torch.allclose(noise1, noise3), f"Noise {name} should be different with different seeds")

if __name__ == '__main__':
    unittest.main()


# ─────────────────────────────────────────────────────────────────────────────
# Additional regression tests added in v2.6.0 audit
# ─────────────────────────────────────────────────────────────────────────────

class TestNoiseMaskRegressions(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    """Regression suite for the two NameError fixes in nodes_sampler.py."""

    def test_latent_image_param_not_latent(self):
        """
        Regression for NameError 'noise_mask' (line ~2905) and
        NameError 'latent' (line ~3007).
        The sample() method must accept latent_image (not 'latent').
        """
        import inspect
        from radiance.nodes.generate.sampler import RadianceSamplerPro
        sig = inspect.signature(RadianceSamplerPro.sample)
        self.assertIn("latent_image", sig.parameters,
                      "sample() must accept 'latent_image' — regression for NameError fixes")

    def test_latent_image_copy_in_source(self):
        """Verify latent_image.copy() is used (not the stale latent.copy())."""
        import inspect
        from radiance.nodes.generate.sampler import RadianceSamplerPro
        source = inspect.getsource(RadianceSamplerPro.sample)
        self.assertIn("latent_image.copy()", source,
                      "sample() must use latent_image.copy() — NameError 'latent' fix")
        self.assertNotIn("out = latent.copy()", source,
                         "Stale 'out = latent.copy()' still present")

    def test_noise_mask_get_not_subscript(self):
        """
        Verify noise_mask is extracted via .get() (safe), not direct subscript.
        A direct latent_image['noise_mask'] would KeyError on non-inpaint latents.
        """
        import inspect
        from radiance.nodes.generate.sampler import RadianceSamplerPro
        source = inspect.getsource(RadianceSamplerPro.sample)
        # The fixed line is: noise_mask = latent_image.get("noise_mask")
        self.assertIn('latent_image.get("noise_mask")', source,
                      "noise_mask must be extracted via .get() — regression for NameError fix")


class TestFluxShiftHelper(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    """Pure math tests for flux_shift_sigmas (no model needed)."""

    def setUp(self):
        from radiance.nodes.generate.sampler import flux_shift_sigmas
        self.fn = flux_shift_sigmas

    def test_shift_1_identity(self):
        s = torch.linspace(1.0, 0.0, 10)
        out = self.fn(s, shift=1.0)
        self.assertTrue(torch.allclose(out, s, atol=1e-6),
                        "shift=1.0 must be identity")

    def test_shift_zero_raises(self):
        s = torch.tensor([0.5, 0.3, 0.1])
        with self.assertRaises(Exception):
            self.fn(s, shift=0.0)

    def test_monotone_preserved(self):
        s = torch.linspace(1.0, 0.0, 20)
        out = self.fn(s, shift=2.5)
        diffs = out[1:] - out[:-1]
        self.assertTrue((diffs <= 1e-6).all(),
                        "Shifted sigmas must remain non-increasing")


class TestCFGPlusPlusHelper(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    """Pure math tests for apply_cfg_plus_plus."""

    def setUp(self):
        from radiance.nodes.generate.sampler import apply_cfg_plus_plus
        self.fn = apply_cfg_plus_plus

    def test_full_progress_is_1(self):
        result = self.fn(cfg=7.5, sigma=torch.tensor(0.0), sigma_max=14.6)
        self.assertAlmostEqual(result, 1.0, places=2)

    def test_zero_progress_is_cfg(self):
        result = self.fn(cfg=7.5, sigma=torch.tensor(14.6), sigma_max=14.6)
        self.assertAlmostEqual(result, 7.5, places=2)

    def test_always_in_range(self):
        cfg = 7.5
        sigma_max = 14.6
        for v in [14.6, 10.0, 5.0, 1.0, 0.0]:
            r = self.fn(cfg=cfg, sigma=torch.tensor(v), sigma_max=sigma_max)
            self.assertGreaterEqual(r, 1.0 - 1e-4)
            self.assertLessEqual(r, cfg + 1e-4)
