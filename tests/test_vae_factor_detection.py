"""
Tests for detect_vae_factor() in hdr/vae.py.

comfy.sd.VAE stores its spatial downscale factor as a plain int for most
architectures, but as a (temporal_formula, h, w) tuple for LTX and other
video VAEs. detect_vae_factor() used to only understand the plain-int form,
so LTX silently fell through to VAE_FACTOR_DEFAULT (8) instead of its real
32 -- a 4x underestimate that made RadianceHDRVAEDecode's tiling decision
think a too-large decode fit in a single tile, crashing at high resolutions.

Covers:
  - a tuple-shaped ratio (LTX-style) is read via spacial_compression_decode()
  - a plain-int ratio (every other architecture) is unaffected
  - the class-name and default fallbacks still work when neither is present
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hdr.vae import detect_vae_factor, VAE_FACTOR_DEFAULT


class _FakeVAE:
    """Bare stand-in for comfy.sd.VAE -- only the attributes detect_vae_factor() reads."""
    def __init__(self, downscale_ratio=None, spacial_compression_decode=None, first_stage_model=None):
        if downscale_ratio is not None:
            self.downscale_ratio = downscale_ratio
        if spacial_compression_decode is not None:
            self.spacial_compression_decode = spacial_compression_decode
        if first_stage_model is not None:
            self.first_stage_model = first_stage_model


class TestTupleShapedRatio:
    def test_ltx_style_tuple_ratio_uses_spacial_compression_decode(self):
        """(temporal_formula, 32, 32), matching comfy's real LTX 2.5 VAE."""
        vae = _FakeVAE(
            downscale_ratio=(lambda a: (a + 7) // 8, 32, 32),
            spacial_compression_decode=lambda: 32,
        )
        assert detect_vae_factor(vae) == 32

    def test_no_spacial_compression_decode_falls_through_to_wrong_default(self):
        """Pins the bug this fix addresses: without the new check, a tuple
        ratio and no matching class name silently produced 8, not 32."""
        vae = _FakeVAE(downscale_ratio=(lambda a: (a + 7) // 8, 32, 32))
        assert detect_vae_factor(vae) == VAE_FACTOR_DEFAULT


class TestPlainIntRatioUnaffected:
    """Every non-LTX architecture (SDXL, Flux, WAN, ...) uses a plain int --
    must return the exact same value whether or not spacial_compression_decode
    exists, so this fix cannot regress them."""

    def test_plain_int_via_spacial_compression_decode(self):
        vae = _FakeVAE(downscale_ratio=8, spacial_compression_decode=lambda: 8)
        assert detect_vae_factor(vae) == 8

    def test_plain_int_without_spacial_compression_decode(self):
        vae = _FakeVAE(downscale_ratio=8)
        assert detect_vae_factor(vae) == 8

    def test_spacial_compression_decode_raising_falls_back_to_plain_int(self):
        def _boom():
            raise RuntimeError("simulated older/odd comfy VAE")
        vae = _FakeVAE(downscale_ratio=8, spacial_compression_decode=_boom)
        assert detect_vae_factor(vae) == 8


class TestFallbacksUnchanged:
    def test_unknown_class_with_no_ratio_at_all_uses_default(self):
        vae = _FakeVAE()
        assert detect_vae_factor(vae) == VAE_FACTOR_DEFAULT

    def test_known_class_name_fallback_still_matches(self):
        class HunyuanVideoVAE:
            pass
        vae = _FakeVAE(first_stage_model=HunyuanVideoVAE())
        assert detect_vae_factor(vae) == 8
