"""
tests/test_temporal_chunking.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Tests for temporal chunking in RadianceVAE4KDecode.decode().

Coverage:
  ─ TileEngine.compute_tiles: equal-sized chunks with t_ov > 0
  ─ Overlap trimming math: per-chunk pix_per_lat (ALBABIT-FIX)
      • equal-sized chunks: same result as first-chunk approach (no regression)
      • asymmetric chunks: per-chunk gives correct result; first-chunk would
        not (validates the fix for future causal VAEs with variable ratios)
  ─ decode() smoke: temporal chunking fires, returns correct frame count
      • no overlap (t_ov=0): concatenation only
      • with overlap (t_ov=1): trimming applied, correct output shape
  ─ LATENT_FORMAT_MAP: 12ch → "mochi_12ch", 128ch → "ltx_128ch"
"""

from __future__ import annotations

import sys
import os
import json
import types
import importlib
import unittest
from unittest.mock import MagicMock

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


# ═════════════════════════════════════════════════════════════════════════════
#  TileEngine.compute_tiles — temporal axis
# ═════════════════════════════════════════════════════════════════════════════

@skip_no_torch
class TestComputeTilesTemporal(unittest.TestCase):
    """TileEngine.compute_tiles used for temporal axis should produce
    full-size chunks with consistent overlap."""

    def setUp(self):
        self.TileEngine = _import_vae().TileEngine

    def test_equal_sized_chunks_with_overlap(self):
        # T=9, chunk_size=2, overlap=1 → 8 chunks, all size 2
        chunks = self.TileEngine.compute_tiles(9, 2, 1)
        self.assertEqual(len(chunks), 8)
        for t1, t2 in chunks:
            self.assertEqual(t2 - t1, 2, f"Expected chunk size 2, got {t2-t1} for ({t1},{t2})")

    def test_full_coverage(self):
        # Last chunk must reach T
        chunks = self.TileEngine.compute_tiles(9, 2, 1)
        self.assertEqual(chunks[0][0], 0)
        self.assertEqual(chunks[-1][1], 9)

    def test_single_chunk_when_t_lte_chunk_size(self):
        chunks = self.TileEngine.compute_tiles(3, 5, 1)
        self.assertEqual(chunks, [(0, 3)])

    def test_overlap_at_or_above_tile_size_raises_instead_of_hanging(self):
        """A non-positive stride must raise, not loop forever.

        stride = tile_size - overlap.  At overlap >= tile_size the stride is
        zero or negative, pos never advances past the exit condition, and the
        loop appends tiles until memory is exhausted: ComfyUI hangs with no
        error and nothing in the log.  This was reachable from shipped widget
        defaults on RadianceVAE4KDecode (temporal_size "2" with the default
        temporal_overlap of 2), so it is guarded at the source as well as at
        that call site.
        """
        for tile_size, overlap in ((2, 2), (2, 32), (8, 8), (16, 20)):
            with self.subTest(tile_size=tile_size, overlap=overlap):
                with self.assertRaises(ValueError):
                    self.TileEngine.compute_tiles(100, tile_size, overlap)

    def test_largest_legal_overlap_still_terminates(self):
        """overlap == tile_size - 1 is the boundary and must still work."""
        chunks = self.TileEngine.compute_tiles(10, 4, 3)
        self.assertEqual(chunks[0][0], 0)
        self.assertEqual(chunks[-1][1], 10)
        self.assertLessEqual(len(chunks), 10)
        for t1, t2 in chunks:
            self.assertEqual(t2 - t1, 4)

    def test_degenerate_sizes_raise(self):
        for tile_size, overlap in ((0, 0), (-4, 0), (8, -1)):
            with self.subTest(tile_size=tile_size, overlap=overlap):
                with self.assertRaises(ValueError):
                    self.TileEngine.compute_tiles(100, tile_size, overlap)


# ═════════════════════════════════════════════════════════════════════════════
#  Overlap trimming — per-chunk pix_per_lat (ALBABIT-FIX)
# ═════════════════════════════════════════════════════════════════════════════

def _apply_overlap_trim(chunk_imgs, t_ov):
    """Replicate the ALBABIT-FIX per-chunk trimming logic from decode()."""
    trimmed = []
    for i, (t1, t2, ch) in enumerate(chunk_imgs):
        if i < len(chunk_imgs) - 1:
            pix_per_lat = ch.shape[0] / max(1, t2 - t1)
            pix_ov = max(1, round(t_ov * pix_per_lat))
            trimmed.append(ch[:max(1, ch.shape[0] - pix_ov)])
        else:
            trimmed.append(ch)
    return torch.cat(trimmed, dim=0)


def _apply_overlap_trim_first_chunk_ratio(chunk_imgs, t_ov):
    """Old (buggy) approach: pix_per_lat from first chunk for all."""
    f1_t1, f1_t2, f1_img = chunk_imgs[0]
    pix_per_lat = f1_img.shape[0] / max(1, f1_t2 - f1_t1)
    pix_ov = max(1, round(t_ov * pix_per_lat))
    trimmed = []
    for i, (t1, t2, ch) in enumerate(chunk_imgs):
        trimmed.append(ch[:max(1, ch.shape[0] - pix_ov)] if i < len(chunk_imgs) - 1 else ch)
    return torch.cat(trimmed, dim=0)


@skip_no_torch
class TestOverlapTrimPerChunk(unittest.TestCase):
    """Per-chunk pix_per_lat trimming: correct results for both equal and
    asymmetric chunk pixel ratios."""

    def _make_chunks(self, specs):
        """specs: list of (t1, t2, num_pixel_frames)."""
        return [(t1, t2, torch.zeros(f, 4, 4, 3)) for t1, t2, f in specs]

    def test_equal_chunks_same_result_as_first_chunk(self):
        # Equal-sized lat chunks, same pixel ratio: both approaches identical.
        # 3 chunks of 2 lat frames each → 7 pixel frames each (Mochi causal: 6*(2-1)+1=7)
        chunks = self._make_chunks([(0,2,7), (1,3,7), (2,4,7)])
        t_ov = 1
        out_per_chunk = _apply_overlap_trim(chunks, t_ov)
        out_first_chunk = _apply_overlap_trim_first_chunk_ratio(chunks, t_ov)
        self.assertEqual(out_per_chunk.shape[0], out_first_chunk.shape[0],
                         "Equal chunks: per-chunk and first-chunk approaches must agree")

    def test_asymmetric_chunks_per_chunk_correct(self):
        # Asymmetric: first chunk has 2 lat frames → 7 px (ratio 3.5),
        # last chunk has 1 lat frame → 1 px (ratio 1.0). overlap=1.
        # Per-chunk trim on first chunk: round(1 * 3.5) = 4 px trimmed → 7-4=3 kept
        # First-chunk (buggy): round(1 * 3.5) = 4 px trimmed from BOTH non-last chunks.
        # Last chunk always kept whole.
        chunks = self._make_chunks([(0,2,7), (1,2,7), (1,2,1)])
        t_ov = 1
        out_per_chunk = _apply_overlap_trim(chunks, t_ov)
        # Middle chunk (index 1): pix_per_lat = 7/1 = 7 → pix_ov = 7 → keep max(1, 7-7)=1
        # vs first-chunk (buggy): pix_ov = 4 → keep 7-4=3 (different answer)
        out_first_chunk = _apply_overlap_trim_first_chunk_ratio(chunks, t_ov)
        # Per-chunk: 3 + 1 + 1 = 5 frames
        # First-chunk: 3 + 3 + 1 = 7 frames (wrong for asymmetric case)
        self.assertNotEqual(
            out_per_chunk.shape[0], out_first_chunk.shape[0],
            "Asymmetric chunks must produce different results for per-chunk vs first-chunk"
        )
        # Verify per-chunk gives the mathematically correct result:
        # chunk0: 7px, lat_size=2, pix_per_lat=3.5 → trim round(3.5)=4 → keep 3
        # chunk1: 7px, lat_size=1, pix_per_lat=7.0 → trim round(7.0)=7 → keep max(1,0)=1
        # chunk2: 1px, no trim
        self.assertEqual(out_per_chunk.shape[0], 3 + 1 + 1)

    def test_single_frame_chunks_never_trimmed_to_zero(self):
        # pix_ov must be clamped so we always keep at least 1 frame.
        chunks = self._make_chunks([(0,2,1), (1,2,1)])
        out = _apply_overlap_trim(chunks, t_ov=1)
        self.assertGreaterEqual(out.shape[0], 1)


# ═════════════════════════════════════════════════════════════════════════════
#  decode() smoke: temporal chunking integration
# ═════════════════════════════════════════════════════════════════════════════

@skip_no_torch
class TestTemporalChunkingSmoke(unittest.TestCase):
    """decode() with a 5D latent needing temporal chunking must route through
    vae.decode_tiled() and return the correct frame count."""

    def _make_mock_vae(self, frames_per_lat_frame=6):
        """Mock VAE: vae.decode(5D_latent) → (1, F_out, H, W, 3) tensor.
        F_out = frames_per_lat_frame * T_lat (simple linear ratio for testing).

        ALBABIT-FIX: decode() now routes 5D latents
        through vae.decode_tiled() instead of Radiance's old recursive
        per-chunk vae.decode() loop, so decode_tiled needs the same
        linear-ratio stand-in applied to the whole latent's T (it receives
        the full latent, not pre-chunked).
        """
        mock = MagicMock()
        mock.latent_dim = 3
        mock.temporal_compression_decode.return_value = 1
        def _decode(lat):
            T = lat.shape[2] if lat.ndim == 5 else 1
            return torch.zeros(1, T * frames_per_lat_frame, 4, 4, 3)
        mock.decode.side_effect = _decode
        def _decode_tiled(samples, tile_x=None, tile_y=None, overlap=None, tile_t=None, overlap_t=None):
            T = samples.shape[2] if samples.ndim == 5 else 1
            return torch.zeros(1, T * frames_per_lat_frame, 4, 4, 3)
        mock.decode_tiled.side_effect = _decode_tiled
        return mock

    def test_no_overlap_frame_count(self):
        """temporal_size=3, t_ov=0 on T=6 latent → decode_tiled(tile_t=3), correct frame count."""
        decoder = _make_decoder()
        mock_vae = self._make_mock_vae(frames_per_lat_frame=2)
        # 5D latent: (1, 4, 6, 4, 4) — 6 temporal latent frames
        latent_5d = torch.zeros(1, 4, 6, 4, 4)
        samples = {"samples": latent_5d}

        img, meta, _ = decoder.decode(
            samples, vae=mock_vae,
            tile_size="Auto", overlap=64,
            hdr_mode="Clip (SDR)", source_space="sRGB",
            display_tonemap="None", hdr_output=False,
            temporal_size=3, temporal_overlap=0,
        )
        # 2 chunks of 3 lat frames → 6 px each → 12 total
        self.assertEqual(img.shape[0], 12,
                         f"Expected 12 frames, got {img.shape[0]}")

    def test_with_overlap_frame_count(self):
        """temporal_size=3, t_ov=1 on T=6 latent → decode_tiled(tile_t=3, overlap_t=1)."""
        decoder = _make_decoder()
        mock_vae = self._make_mock_vae(frames_per_lat_frame=2)
        latent_5d = torch.zeros(1, 4, 6, 4, 4)
        samples = {"samples": latent_5d}

        img, meta, _ = decoder.decode(
            samples, vae=mock_vae,
            tile_size="Auto", overlap=64,
            hdr_mode="Clip (SDR)", source_space="sRGB",
            display_tonemap="None", hdr_output=False,
            temporal_size=3, temporal_overlap=1,
        )
        self.assertEqual(img.shape[0], 12)
        self.assertIsInstance(img, torch.Tensor)

    def test_temporal_chunking_restores_alpha_after_stitch(self):
        decoder = _make_decoder()
        mock_vae = self._make_mock_vae(frames_per_lat_frame=2)
        latent_5d = torch.zeros(1, 4, 6, 4, 4)
        alpha = torch.linspace(0.0, 1.0, 12).view(12, 1, 1, 1).expand(12, 4, 4, 3)

        img, meta, _ = decoder.decode(
            {"samples": latent_5d}, vae=mock_vae,
            tile_size="Auto", overlap=64,
            hdr_mode="Clip (SDR)", source_space="sRGB",
            display_tonemap="None", hdr_output=False,
            temporal_size=3, temporal_overlap=1, alpha=alpha,
        )

        self.assertEqual(tuple(img.shape), (12, 4, 4, 4))
        self.assertTrue(torch.allclose(img[..., 3], alpha[..., 0]))
        self.assertTrue(json.loads(meta)["alpha_restored"])

    def test_temporal_chunking_rejects_ambiguous_alpha_batch(self):
        decoder = _make_decoder()
        mock_vae = self._make_mock_vae(frames_per_lat_frame=2)
        with self.assertRaisesRegex(ValueError, "Alpha batch"):
            decoder.decode(
                {"samples": torch.zeros(1, 4, 6, 4, 4)}, vae=mock_vae,
                tile_size="Auto", overlap=64,
                hdr_mode="Clip (SDR)", source_space="sRGB",
                display_tonemap="None", hdr_output=False,
                temporal_size=3, temporal_overlap=1,
                alpha=torch.ones(5, 4, 4, 3),
            )

    def test_temporal_chunking_exports_every_frame_when_requested(self):
        decoder = _make_decoder()
        decoder._save_rhdr = MagicMock(side_effect=lambda _frame, _out, prefix, precision: f"{prefix}.{precision}.rhdr")
        sys.modules["folder_paths"].get_temp_directory = lambda: "temp"
        img, meta, _ = decoder.decode(
            {"samples": torch.zeros(1, 4, 6, 4, 4)},
            vae=self._make_mock_vae(frames_per_lat_frame=2),
            tile_size="Auto", overlap=64,
            hdr_mode="Compress (Log)", source_space="ARRI LogC4",
            display_tonemap="None", hdr_output=True,
            temporal_size=3, temporal_overlap=1,
            export_rhdr=True, rhdr_precision="f32",
        )
        metadata = json.loads(meta)
        self.assertEqual(img.shape[0], 12)
        self.assertEqual(decoder._save_rhdr.call_count, 12)
        self.assertEqual(len(metadata["rhdr_export"]), 12)

    def test_temporal_chunking_not_fired_when_size_zero(self):
        """temporal_size=0 must disable chunking entirely."""
        decoder = _make_decoder()
        mock_vae = self._make_mock_vae(frames_per_lat_frame=2)
        latent_5d = torch.zeros(1, 4, 6, 4, 4)
        samples = {"samples": latent_5d}

        img, _, _ = decoder.decode(
            samples, vae=mock_vae,
            tile_size="Auto", overlap=64,
            hdr_mode="Clip (SDR)", source_space="sRGB",
            display_tonemap="None", hdr_output=False,
            temporal_size=0, temporal_overlap=0,
        )
        # With temporal_size=0, single vae.decode() call on full 6-frame latent
        self.assertEqual(img.shape[0], 12)


# ═════════════════════════════════════════════════════════════════════════════
#  LATENT_FORMAT_MAP — new entries
# ═════════════════════════════════════════════════════════════════════════════

@skip_no_torch
class TestLatentFormatMap(unittest.TestCase):

    def setUp(self):
        vae_mod = _import_vae()
        self.fmt_map = vae_mod.LATENT_FORMAT_MAP

    def test_mochi_12ch(self):
        self.assertEqual(self.fmt_map.get(12), "mochi_12ch")

    def test_ltx_128ch(self):
        self.assertEqual(self.fmt_map.get(128), "ltx_128ch")

    def test_existing_entries_unchanged(self):
        self.assertEqual(self.fmt_map.get(4),  "sd_4ch")
        self.assertEqual(self.fmt_map.get(16), "flux_16ch")

    def test_unknown_channel_count_returns_fallback(self):
        # _latent_format_label() uses LATENT_FORMAT_MAP.get(ch, f"unknown_{ch}ch")
        result = self.fmt_map.get(99, "unknown_99ch")
        self.assertEqual(result, "unknown_99ch")


if __name__ == "__main__":
    unittest.main()
