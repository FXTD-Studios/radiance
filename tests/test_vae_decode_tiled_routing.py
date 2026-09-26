"""
tests/test_vae_decode_tiled_routing.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Tests that RadianceVAE4KDecode.decode() routes video latents to the right
place under the VAE tiling integration:

  - spatial+temporal tiling both go through the single vae.decode_tiled()
    call (comfy.sd.VAE's own integrated tiler), falling back to plain
    vae.decode() when neither is actually needed. (3.5: the latent RUDRA
    decoders and their separate recursive chunking path are gone.)

See project_radiance_vae_tiling_seams memory for the investigation this
integration replaced (Radiance's own stacked spatial-tile-per-temporal-chunk
decode, which re-decoded a full spatial tile's worth of activations for
every temporal chunk independently).
"""

from __future__ import annotations

import sys
import os
import json
import types
import importlib
import unittest

import pytest

# ── Real torch check ──────────────────────────────────────────────────────────
try:
    import torch
    HAS_TORCH = hasattr(torch, "__version__")
except ImportError:
    HAS_TORCH = False

skip_no_torch = unittest.skipUnless(HAS_TORCH, "PyTorch not available")

# ── Minimal ComfyUI stubs ─────────────────────────────────────────────────────
for _m in ["folder_paths", "comfy", "comfy.utils", "comfy.model_management"]:
    if _m not in sys.modules:
        sys.modules[_m] = types.ModuleType(_m)

_mm = sys.modules["comfy.model_management"]
if not hasattr(_mm, "get_torch_device"):
    _mm.get_torch_device = lambda: torch.device("cpu") if HAS_TORCH else None
if not hasattr(_mm, "soft_empty_cache"):
    _mm.soft_empty_cache = lambda: None  # no-op in tests

_cu = sys.modules["comfy.utils"]
if not hasattr(_cu, "ProgressBar"):
    class _FakeProgressBar:
        def __init__(self, *a, **kw): pass
        def update(self, *a): pass
        def update_absolute(self, *a): pass
    _cu.ProgressBar = _FakeProgressBar

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def _import_vae():
    mod_name = "hdr.vae"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    return importlib.import_module(mod_name)


def _make_decoder():
    return _import_vae().RadianceVAE4KDecode()


class _FakeVideoVAE:
    """Stand-in for comfy.sd.VAE exposing only what decode()'s tiling
    decision touches, with call tracking instead of MagicMock's
    auto-generated (and always-truthy) attributes: detect_vae_factor()
    and the routing logic both branch on real attribute *absence*."""

    def __init__(self, frames_per_lat_frame=2, downscale_ratio=8, temporal_compression=8):
        self.downscale_ratio = downscale_ratio
        self.latent_dim = 3
        self._frames_per_lat_frame = frames_per_lat_frame
        self._temporal_compression = temporal_compression
        self.decode_calls = []
        self.decode_tiled_calls = []

    def temporal_compression_decode(self):
        return self._temporal_compression

    def decode(self, latent):
        self.decode_calls.append(tuple(latent.shape))
        if latent.ndim != 5:
            return torch.zeros(1, 4, 4, 3)
        T = latent.shape[2]
        return torch.zeros(1, T * self._frames_per_lat_frame, 4, 4, 3)

    def decode_tiled(self, samples, tile_x=None, tile_y=None, overlap=None, tile_t=None, overlap_t=None):
        self.decode_tiled_calls.append(
            dict(tile_x=tile_x, tile_y=tile_y, overlap=overlap, tile_t=tile_t, overlap_t=overlap_t)
        )
        T = samples.shape[2] if samples.ndim == 5 else 1
        return torch.zeros(1, T * self._frames_per_lat_frame, 4, 4, 3)


@pytest.mark.real_torch
class TestUnifiedPathFastCase(unittest.TestCase):
    """Small resolution + short clip: neither spatial nor temporal tiling
    is needed, so decode() must skip decode_tiled() and call vae.decode()
    directly, matching pre-fix behavior for the common case."""

    def test_small_video_uses_plain_decode(self):
        decoder = _make_decoder()
        vae = _FakeVideoVAE()
        latent = torch.zeros(1, 4, 4, 32, 32)  # pix 256x256, well under max_tile

        img, _, _ = decoder.decode(
            {"samples": latent}, vae=vae,
            tile_size="Auto", overlap=64,
            hdr_mode="Clip (SDR)", source_space="sRGB",
            display_tonemap="None", hdr_output=False,
            temporal_size="Auto", temporal_overlap=2,
        )

        self.assertEqual(len(vae.decode_calls), 1)
        self.assertEqual(len(vae.decode_tiled_calls), 0)
        self.assertEqual(img.shape[0], 4 * 2)  # 4 latent frames * 2 px/lat


@skip_no_torch
class TestUnifiedPathSpatialOnly(unittest.TestCase):
    """Large resolution, short clip: decode_tiled() must fire for the
    spatial tile, with tile_t/overlap_t omitted (None) so comfy's own
    dispatcher treats the temporal axis as unbounded."""

    def test_large_frame_short_clip_omits_temporal_args(self):
        decoder = _make_decoder()
        vae = _FakeVideoVAE()
        # lat 300x300 * factor 8 = pix 2400x2400 -> exceeds max_tile (1536)
        latent = torch.zeros(1, 4, 1, 300, 300)

        decoder.decode(
            {"samples": latent}, vae=vae,
            tile_size="Auto", overlap=64,
            hdr_mode="Clip (SDR)", source_space="sRGB",
            display_tonemap="None", hdr_output=False,
            temporal_size="Auto", temporal_overlap=2,
        )

        self.assertEqual(len(vae.decode_tiled_calls), 1)
        call = vae.decode_tiled_calls[0]
        self.assertIsNone(call["tile_t"])
        self.assertIsNone(call["overlap_t"])
        self.assertIsNotNone(call["tile_x"])
        self.assertIsNotNone(call["tile_y"])


@skip_no_torch
class TestUnifiedPathTemporalOnly(unittest.TestCase):
    """Small resolution, long clip with a manual temporal_size smaller than
    the clip: decode_tiled() must fire with a real tile_t even though no
    spatial tiling is needed."""

    def test_long_clip_small_frame_sets_temporal_args(self):
        decoder = _make_decoder()
        vae = _FakeVideoVAE()
        latent = torch.zeros(1, 4, 20, 4, 4)  # pix 32x32, 20 latent frames

        decoder.decode(
            {"samples": latent}, vae=vae,
            tile_size="Auto", overlap=64,
            hdr_mode="Clip (SDR)", source_space="sRGB",
            display_tonemap="None", hdr_output=False,
            temporal_size="8", temporal_overlap=2,
        )

        self.assertEqual(len(vae.decode_tiled_calls), 1)
        call = vae.decode_tiled_calls[0]
        self.assertEqual(call["tile_t"], 8)
        self.assertEqual(call["overlap_t"], 2)


@skip_no_torch
class TestMetadataExposesTemporalDecision(unittest.TestCase):
    """The metadata STRING output already reports the resolved tile_size/
    overlap; temporal_size/temporal_chunking mirror that for the temporal
    axis so the Auto decision is visible without reading console logs."""

    def test_fast_path_reports_no_chunking(self):
        decoder = _make_decoder()
        vae = _FakeVideoVAE()
        latent = torch.zeros(1, 4, 4, 32, 32)

        _, meta, _ = decoder.decode(
            {"samples": latent}, vae=vae,
            tile_size="Auto", overlap=64,
            hdr_mode="Clip (SDR)", source_space="sRGB",
            display_tonemap="None", hdr_output=False,
            temporal_size="Auto", temporal_overlap=2,
        )

        metadata = json.loads(meta)
        self.assertFalse(metadata["temporal_chunking"])
        self.assertIsInstance(metadata["temporal_size"], int)

    def test_spatial_only_reports_no_chunking_despite_decode_tiled(self):
        decoder = _make_decoder()
        vae = _FakeVideoVAE()
        latent = torch.zeros(1, 4, 1, 300, 300)

        _, meta, _ = decoder.decode(
            {"samples": latent}, vae=vae,
            tile_size="Auto", overlap=64,
            hdr_mode="Clip (SDR)", source_space="sRGB",
            display_tonemap="None", hdr_output=False,
            temporal_size="Auto", temporal_overlap=2,
        )

        self.assertFalse(json.loads(meta)["temporal_chunking"])

    def test_temporal_chunking_reports_resolved_size(self):
        decoder = _make_decoder()
        vae = _FakeVideoVAE()
        latent = torch.zeros(1, 4, 20, 4, 4)

        _, meta, _ = decoder.decode(
            {"samples": latent}, vae=vae,
            tile_size="Auto", overlap=64,
            hdr_mode="Clip (SDR)", source_space="sRGB",
            display_tonemap="None", hdr_output=False,
            temporal_size="8", temporal_overlap=2,
        )

        metadata = json.loads(meta)
        self.assertTrue(metadata["temporal_chunking"])
        self.assertEqual(metadata["temporal_size"], 8)

    def test_4d_image_has_no_temporal_keys(self):
        """Images never enter the 5D routing branch, so _temporal_metadata
        stays None and the keys must be entirely absent, not False/0."""
        decoder = _make_decoder()
        vae = _FakeVideoVAE()
        latent = torch.zeros(1, 4, 4, 4)

        _, meta, _ = decoder.decode(
            {"samples": latent}, vae=vae,
            tile_size="Auto", overlap=64,
            hdr_mode="Clip (SDR)", source_space="sRGB",
            display_tonemap="None", hdr_output=False,
        )

        metadata = json.loads(meta)
        self.assertNotIn("temporal_size", metadata)
        self.assertNotIn("temporal_chunking", metadata)


if __name__ == "__main__":
    unittest.main()
