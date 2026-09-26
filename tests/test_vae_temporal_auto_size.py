"""
Tests for TileEngine.get_optimal_temporal_size() and _resolve_temporal_frames()
in hdr/vae.py.

These back the "Auto" option on RadianceVAE4KDecode's temporal_size widget,
added when temporal chunking was switched from Radiance's own stacked
(spatial-tile-per-temporal-chunk) decode to a single vae.decode_tiled() call
that tiles space and time together. See project_radiance_vae_tiling_seams
memory for the full investigation.

Covers:
  - get_optimal_temporal_size: calibration point, tile-size scaling,
    min/max clamping, compression fallback
  - _resolve_temporal_frames: "Auto", numeric string, raw int, and the
    0/"0"/None "explicitly disabled" sentinel (used internally by the
    recursive chunking call sites to stop further recursion)
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hdr.vae import TileEngine, _resolve_temporal_frames


class TestGetOptimalTemporalSizeCalibration:
    """bytes_per_pixel_frame is tuned so the one real crash/success data
    point from the LTX-2.5 investigation resolves to exactly the verified-
    safe value, not just "somewhere under the verified-crash value"."""

    def test_calibration_point_resolves_to_known_safe_value(self):
        # LTX-2.5, tile=1536px, ~16.8GB budget, 8x temporal compression:
        # 8 latent frames decoded cleanly, 16 hard-crashed.
        frames = TileEngine.get_optimal_temporal_size(
            tile_size_px=1536, temporal_compression=8, vram_budget_gb=16.8,
        )
        assert frames == 8

    def test_calibration_point_stays_well_under_known_crash_value(self):
        frames = TileEngine.get_optimal_temporal_size(
            tile_size_px=1536, temporal_compression=8, vram_budget_gb=16.8,
        )
        assert frames < 16


class TestGetOptimalTemporalSizeScaling:
    def test_smaller_tile_allows_more_temporal_frames(self):
        """Spatial and temporal chunking share one VRAM budget; a smaller
        spatial tile must leave more room for temporal depth."""
        big_tile = TileEngine.get_optimal_temporal_size(
            tile_size_px=1536, temporal_compression=8, vram_budget_gb=16.8,
        )
        small_tile = TileEngine.get_optimal_temporal_size(
            tile_size_px=512, temporal_compression=8, vram_budget_gb=16.8,
        )
        assert small_tile > big_tile

    def test_higher_compression_reduces_latent_frame_count(self):
        """A VAE with a higher temporal compression ratio packs more pixel
        frames per latent frame, so fewer latent frames fit the same
        pixel-frame VRAM budget."""
        low_compression = TileEngine.get_optimal_temporal_size(
            tile_size_px=1024, temporal_compression=4, vram_budget_gb=16.8,
        )
        high_compression = TileEngine.get_optimal_temporal_size(
            tile_size_px=1024, temporal_compression=8, vram_budget_gb=16.8,
        )
        assert high_compression < low_compression


class TestGetOptimalTemporalSizeClamping:
    def test_tiny_budget_clamps_to_min_frames(self):
        frames = TileEngine.get_optimal_temporal_size(
            tile_size_px=1536, temporal_compression=8, vram_budget_gb=0.1,
        )
        assert frames == 2  # default min_frames

    def test_huge_budget_clamps_to_max_frames(self):
        frames = TileEngine.get_optimal_temporal_size(
            tile_size_px=512, temporal_compression=8, vram_budget_gb=500.0,
        )
        assert frames == 64  # default max_frames

    def test_custom_min_max_are_respected(self):
        frames = TileEngine.get_optimal_temporal_size(
            tile_size_px=1536, temporal_compression=8, vram_budget_gb=0.1,
            min_frames=5, max_frames=10,
        )
        assert frames == 5


class TestGetOptimalTemporalSizeCompressionFallback:
    """temporal_compression=None (vae.temporal_compression_decode() returned
    None, e.g. an unrecognized upscale_ratio shape) must not crash and
    should behave like the common 8x case."""

    def test_none_compression_falls_back_to_8x(self):
        with_none = TileEngine.get_optimal_temporal_size(
            tile_size_px=1536, temporal_compression=None, vram_budget_gb=16.8,
        )
        with_eight = TileEngine.get_optimal_temporal_size(
            tile_size_px=1536, temporal_compression=8, vram_budget_gb=16.8,
        )
        assert with_none == with_eight

    def test_zero_compression_falls_back_to_8x(self):
        with_zero = TileEngine.get_optimal_temporal_size(
            tile_size_px=1536, temporal_compression=0, vram_budget_gb=16.8,
        )
        with_eight = TileEngine.get_optimal_temporal_size(
            tile_size_px=1536, temporal_compression=8, vram_budget_gb=16.8,
        )
        assert with_zero == with_eight


class TestResolveTemporalFrames:
    def test_explicit_zero_int_means_disabled(self):
        assert _resolve_temporal_frames(0, 1536, 8) == 0

    def test_explicit_zero_string_means_disabled(self):
        assert _resolve_temporal_frames("0", 1536, 8) == 0

    def test_none_means_disabled(self):
        assert _resolve_temporal_frames(None, 1536, 8) == 0

    def test_auto_with_ts_px_computes_real_value(self):
        resolved = _resolve_temporal_frames("Auto", 1536, 8)
        expected = TileEngine.get_optimal_temporal_size(1536, 8)
        assert resolved == expected

    def test_auto_with_no_ts_px_means_disabled(self):
        """ts_px=None: there is no VRAM
        calibration data for RUDRA's memory profile, so "Auto" keeps that
        path's pre-existing default behavior (no chunking) instead of
        guessing."""
        assert _resolve_temporal_frames("Auto", None, 8) == 0

    def test_numeric_string_preset(self):
        assert _resolve_temporal_frames("16", 1536, 8) == 16

    def test_raw_int_passthrough(self):
        """Internal callers (e.g. recursive chunking
        block) may pass a plain int rather than a widget-style string."""
        assert _resolve_temporal_frames(3, None, 8) == 3
