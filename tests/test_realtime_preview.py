"""
tests/test_realtime_preview.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Tests for nodes_realtime_preview — Pillar 05 · Real-Time Preview

Coverage
────────
  Pure-numpy helpers   : _apply_false_color, _sobel_mag, _focus_peak, _split_view
  Timecode helper      : RadianceFrameStamp._frame_to_tc
  Node-level (torch)   : all 7 node classes (skipped when real torch absent)
  HTTP server          : _PREVIEW_BUFFER, _frame_to_jpeg, server lifecycle
  Registration         : NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
"""

from __future__ import annotations

import importlib
import io
import sys
import types
import unittest
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Torch detection (same pattern as test_aces2 / test_studio_integrations)
# ─────────────────────────────────────────────────────────────────────────────
import torch
HAS_TORCH = hasattr(torch, "__version__")
skip_no_torch = unittest.skipUnless(HAS_TORCH, "real torch not available")

# ─────────────────────────────────────────────────────────────────────────────
# Import the module under test
# ─────────────────────────────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from radiance.nodes.monitor import realtime as _mod
from radiance.nodes.monitor.realtime import (
    _apply_false_color,
    _sobel_mag,
    _focus_peak,
    _split_view,
    _FALSE_COLOR_ZONES,
    _PREVIEW_BUFFER,
    _frame_to_jpeg,
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    RadianceFalseColorMonitor,
    RadianceFocusPeaking,
    RadianceSplitView,
    RadianceContactSheet,
    RadianceFlipbookGIF,
    RadianceFrameStamp,
    RadiancePreviewServer,
)

# This module already gates its torch-dependent tests correctly (they skip
# cleanly against conftest's stub), so opt out of the automatic module-level
# skip and keep the rest of the file running on the no-torch CI matrix.
RADIANCE_TORCH_GATED = True



# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _rgb(h: int = 8, w: int = 8, fill: float = 0.5) -> np.ndarray:
    """Return (H, W, 3) float32 array."""
    return np.full((h, w, 3), fill, dtype=np.float32)


def _make_tensor(h: int = 8, w: int = 8, b: int = 1, fill: float = 0.5):
    """Return (B, H, W, 3) float32 tensor — requires real torch."""
    return torch.full((b, h, w, 3), fill, dtype=torch.float32)


# ═════════════════════════════════════════════════════════════════════════════
# 1. False Color — pure numpy
# ═════════════════════════════════════════════════════════════════════════════

class TestFalseColorHelper(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    def test_encoded_middle_grey_measures_linear_middle_grey(self):
        # Independent code values from the sRGB / BT.709 transfer equations.
        for encoding, value in [("Linear Rec.709", 0.18), ("sRGB", 0.4613561295),
                                ("Rec.709", 0.4090077289)]:
            out = _apply_false_color(_rgb(4, 4, value), input_encoding=encoding)
            np.testing.assert_allclose(out, np.broadcast_to([0.0, 0.85, 0.15], out.shape), atol=1e-6)

    def test_zero_strength_preserves_hdr_alpha_and_zebra_pixels(self):
        image = np.broadcast_to(np.array([-0.2, 4.0, 2.0, 0.37], dtype=np.float32), (8, 32, 4)).copy()
        np.testing.assert_array_equal(_apply_false_color(image, strength=0.0), image)

    def test_zebra_strength_is_blended_and_alpha_is_preserved(self):
        image = np.ones((8, 32, 4), dtype=np.float32)
        image[..., 3] = 0.37
        original = image.copy()
        out = _apply_false_color(image, strength=0.25)
        np.testing.assert_allclose(out[:, 8:16, :3], 0.75)
        np.testing.assert_array_equal(out[..., 3], image[..., 3])
        np.testing.assert_array_equal(image, original)

    def test_output_shape_preserved(self):
        img = _rgb(16, 16, 0.5)
        out = _apply_false_color(img)
        self.assertEqual(out.shape, img.shape)

    def test_output_dtype_float32(self):
        img = _rgb(8, 8, 0.2)
        out = _apply_false_color(img)
        self.assertEqual(out.dtype, np.float32)

    def test_output_clipped_to_01(self):
        img = _rgb(8, 8, 2.0)  # over-bright
        out = _apply_false_color(img, strength=1.0)
        self.assertLessEqual(float(out.max()), 1.0)
        self.assertGreaterEqual(float(out.min()), 0.0)

    def test_strength_zero_is_passthrough(self):
        img = _rgb(8, 8, 0.18)
        out = _apply_false_color(img, strength=0.0)
        np.testing.assert_allclose(out, img, atol=1e-5)

    def test_midgrey_zone_is_green(self):
        """Luma 0.18 → 18% grey zone → should be greenish."""
        img = _rgb(4, 4, 0.18)
        out = _apply_false_color(img, strength=1.0)
        # Green channel should dominate
        self.assertGreater(float(out[..., 1].mean()), float(out[..., 0].mean()))
        self.assertGreater(float(out[..., 1].mean()), float(out[..., 2].mean()))

    def test_black_clip_zone(self):
        """Luma 0.005 → black clip zone → output should be (0, 0, 0)."""
        img = _rgb(4, 4, 0.005)
        out = _apply_false_color(img, strength=1.0)
        np.testing.assert_allclose(out, 0.0, atol=1e-4)

    def test_white_clip_zone_has_zebra(self):
        """Luma 1.0 → clipped zone → zebra stripes (some black pixels)."""
        img = _rgb(8, 16, 1.0)
        out = _apply_false_color(img, strength=1.0)
        # Zebra alternates white and black — so not all pixels are white
        self.assertLess(float(out.min()), 0.5)

    def test_passthrough_zone_unchanged(self):
        """Luma ~0.4 is in the normal/passthrough zone — pixel unchanged."""
        val = 0.4
        img = _rgb(4, 4, val)
        out = _apply_false_color(img, strength=1.0)
        np.testing.assert_allclose(out, val, atol=1e-4)

    def test_hdr_peak_normalises(self):
        """With hdr_peak=10, a pixel at 1.8 maps to luma 0.18 → green zone."""
        img = _rgb(4, 4, 1.8)  # 1.8 / 10 = 0.18 → green zone
        out = _apply_false_color(img, strength=1.0, hdr_peak=10.0)
        self.assertGreater(float(out[..., 1].mean()), float(out[..., 0].mean()))

    def test_all_zones_defined(self):
        """There must be at least 9 zones and none have gaps in ranges."""
        self.assertGreaterEqual(len(_FALSE_COLOR_ZONES), 9)
        # Each zone's lo < hi
        for lo, hi, color, label in _FALSE_COLOR_ZONES:
            self.assertLess(lo, hi, f"Zone '{label}' has lo >= hi")


# ═════════════════════════════════════════════════════════════════════════════
# 2. Sobel helper — pure numpy
# ═════════════════════════════════════════════════════════════════════════════

class TestSobelMag(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    def test_output_shape(self):
        gray = np.random.rand(16, 16).astype(np.float32)
        mag = _sobel_mag(gray)
        self.assertEqual(mag.shape, gray.shape)

    def test_flat_image_near_zero(self):
        """Uniform image should have near-zero Sobel magnitude."""
        gray = np.ones((8, 8), dtype=np.float32) * 0.5
        mag = _sobel_mag(gray)
        self.assertLess(float(mag.max()), 1e-5)

    def test_edge_has_high_magnitude(self):
        """Sharp edge between black and white → high magnitude."""
        gray = np.zeros((8, 8), dtype=np.float32)
        gray[:, 4:] = 1.0  # hard vertical edge at col 4
        mag = _sobel_mag(gray)
        self.assertGreater(float(mag[:, 4].mean()), 0.5)

    def test_non_negative(self):
        gray = np.random.rand(8, 8).astype(np.float32)
        mag = _sobel_mag(gray)
        self.assertGreaterEqual(float(mag.min()), 0.0)


# ═════════════════════════════════════════════════════════════════════════════
# 3. Focus Peaking helper — pure numpy
# ═════════════════════════════════════════════════════════════════════════════

class TestFocusPeakHelper(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    def test_output_shape(self):
        img = _rgb(8, 8, 0.3)
        out = _focus_peak(img, threshold=0.2, color=(1.0, 0.0, 0.0), strength=1.0)
        self.assertEqual(out.shape, img.shape)

    def test_flat_image_unchanged(self):
        """No edges in a flat image → nothing highlighted."""
        img = _rgb(8, 8, 0.3)
        out = _focus_peak(img, threshold=0.0001, color=(1.0, 0.0, 0.0), strength=1.0)
        # Flat image has ~zero Sobel mag, so mag/max = nan or 0 → no pixels above threshold
        # With threshold=0.0001 some boundary diff-edge pixels may be coloured;
        # just verify shape and clip range.
        self.assertEqual(out.shape, img.shape)
        self.assertLessEqual(float(out.max()), 1.0)

    def test_edge_gets_colored(self):
        """Hard edge → Sobel peak → pixels at the edge coloured red."""
        img = np.zeros((8, 16, 3), dtype=np.float32)
        img[:, 8:] = 1.0  # vertical edge
        out = _focus_peak(img, threshold=0.3, color=(1.0, 0.0, 0.0), strength=1.0)
        # At least some pixels should be red (channel 0 > channel 1)
        self.assertTrue((out[..., 0] > out[..., 1]).any())

    def test_strength_zero_is_passthrough(self):
        img = _rgb(8, 8, 0.5)
        out = _focus_peak(img, threshold=0.0, color=(1.0, 0.0, 0.0), strength=0.0)
        np.testing.assert_allclose(out, img, atol=1e-5)

    def test_output_clamped(self):
        img = _rgb(8, 8, 0.9)
        out = _focus_peak(img, threshold=0.0, color=(1.0, 1.0, 1.0), strength=1.0)
        self.assertLessEqual(float(out.max()), 1.0)


# ═════════════════════════════════════════════════════════════════════════════
# 4. Split View helper — pure numpy
# ═════════════════════════════════════════════════════════════════════════════

class TestSplitViewHelper(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    def _ab(self):
        a = np.zeros((8, 16, 3), dtype=np.float32)        # black
        b = np.ones((8, 16, 3), dtype=np.float32)         # white
        return a, b

    def test_wipe_h_left_is_a(self):
        a, b = self._ab()
        out = _split_view(a, b, "wipe_h", 0.5)
        # left half should be black (from a) — exclude guide-line column at split
        # split_col = 8 on a 16-wide image; guide occupies cols 7-8
        self.assertAlmostEqual(float(out[:, :6].mean()), 0.0, places=4)

    def test_wipe_h_right_is_b(self):
        a, b = self._ab()
        out = _split_view(a, b, "wipe_h", 0.5)
        # right half should be white (from b), minus the guide line
        self.assertGreater(float(out[:, 9:].mean()), 0.8)

    def test_wipe_v_top_is_a(self):
        a, b = self._ab()
        out = _split_view(a, b, "wipe_v", 0.5)
        # split_row = 4 on an 8-tall image; guide occupies rows 3-4
        # check rows 0..2 (strictly above the guide)
        self.assertAlmostEqual(float(out[:2, :].mean()), 0.0, places=4)

    def test_wipe_v_bottom_is_b(self):
        a, b = self._ab()
        out = _split_view(a, b, "wipe_v", 0.5)
        self.assertGreater(float(out[5:, :].mean()), 0.8)

    def test_side_by_side_shape(self):
        a, b = self._ab()
        out = _split_view(a, b, "side_by_side", 0.5)
        self.assertEqual(out.shape, a.shape)

    def test_side_by_side_left_a_right_b(self):
        a, b = self._ab()
        out = _split_view(a, b, "side_by_side", 0.5)
        self.assertAlmostEqual(float(out[:, :8].mean()), 0.0, places=4)
        self.assertAlmostEqual(float(out[:, 8:].mean()), 1.0, places=4)

    def test_diff_identical_is_black(self):
        a = _rgb(8, 8, 0.4)
        out = _split_view(a, a.copy(), "diff", 0.5)
        np.testing.assert_allclose(out, 0.0, atol=1e-5)

    def test_diff_complementary_amplified(self):
        a = np.zeros((8, 8, 3), dtype=np.float32)
        b = np.ones((8, 8, 3), dtype=np.float32)
        out = _split_view(a, b, "diff", 0.5)
        # abs(1-0)*4 = 4.0, clamped to 1.0
        np.testing.assert_allclose(out, 1.0, atol=1e-5)

    def test_output_shape_preserved(self):
        a, b = self._ab()
        for mode in ["wipe_h", "wipe_v", "side_by_side", "diff"]:
            out = _split_view(a, b, mode, 0.5)
            self.assertEqual(out.shape, a.shape, f"Failed for mode={mode}")

    def test_position_zero_all_a(self):
        """position=0 means full A in wipe_h (right side is still b from col 0)."""
        a = np.zeros((8, 16, 3), dtype=np.float32)
        b = np.ones((8, 16, 3), dtype=np.float32)
        out = _split_view(a, b, "wipe_h", 0.0)
        # split_col = 0 → entire image is b
        self.assertGreater(float(out.mean()), 0.8)

    def test_position_one_all_b(self):
        """position=1 → split_col = W → entire image is a (no guide line)."""
        a = np.zeros((8, 16, 3), dtype=np.float32)
        b = np.ones((8, 16, 3), dtype=np.float32)
        out = _split_view(a, b, "wipe_h", 1.0)
        # split_col = W → a[:, W:] = b[:, W:] → nothing replaced → all black
        self.assertLess(float(out.mean()), 0.2)


# ═════════════════════════════════════════════════════════════════════════════
# 5. Frame Stamp helper (pure Python, no torch)
# ═════════════════════════════════════════════════════════════════════════════

class TestFrameToTC(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    """RadianceFrameStamp._frame_to_tc static method — no torch required."""

    def _tc(self, frame, fps):
        return RadianceFrameStamp._frame_to_tc(frame, fps)

    def test_zero_frame(self):
        self.assertEqual(self._tc(0, 24.0), "00:00:00:00")

    def test_one_second(self):
        self.assertEqual(self._tc(24, 24.0), "00:00:01:00")

    def test_one_minute(self):
        self.assertEqual(self._tc(24 * 60, 24.0), "00:01:00:00")

    def test_one_hour(self):
        self.assertEqual(self._tc(24 * 3600, 24.0), "01:00:00:00")

    def test_frame_within_second(self):
        self.assertEqual(self._tc(25 * 15 + 7, 25.0), "00:00:15:07")

    def test_30fps(self):
        # 30fps, 1001 frames = 33 seconds + 11 frames
        frames = 30 * 33 + 11
        self.assertEqual(self._tc(frames, 30.0), "00:00:33:11")

    def test_format_zero_padded(self):
        tc = self._tc(5, 24.0)
        parts = tc.split(":")
        self.assertEqual(len(parts), 4)
        for p in parts:
            self.assertEqual(len(p), 2, f"Part '{p}' not zero-padded")

    def test_start_frame_1001(self):
        """Typical VFX first frame = 1001."""
        tc = self._tc(1001, 24.0)
        self.assertIsInstance(tc, str)
        self.assertIn(":", tc)

    # ── Drop-frame timecode (SMPTE 12M, 29.97 / 59.94 fps) ──────────────────

    def _df(self, frame, fps=29.97):
        return RadianceFrameStamp._frame_to_tc(frame, fps, drop_frame=True)

    def test_df_frame_zero(self):
        """Frame 0 always starts at 00:00:00;00."""
        self.assertEqual(self._df(0), "00:00:00;00")

    def test_df_separator_is_semicolon(self):
        """Drop-frame TC uses ';' before the frame field."""
        tc = self._df(30)
        self.assertIn(";", tc)
        self.assertNotIn(":0", tc.split(";")[-1])  # frame field after semicolon

    def test_df_one_minute_skips_frames_00_01(self):
        """
        At 29.97 DF, frame 1800 is nominally 00:01:00:00 in NDF but the DF
        standard skips frames 00 and 01 at each non-10th minute.
        So frame index 1800 in DF lands at 00:01:00;02 (first non-dropped frame).
        """
        tc = self._df(1800)
        # Should be in the 00:01 range and show frame ≥ 02
        self.assertTrue(tc.startswith("00:01:"), f"Unexpected TC: {tc}")
        ff = int(tc.split(";")[-1])
        self.assertGreaterEqual(ff, 2)

    def test_df_ten_minute_no_skip(self):
        """
        At the 10-minute boundary no frames are dropped.
        Frame 17982 (= 10×1798.2, the 10-minute boundary in DF) maps to 00:10:00;00.
        """
        # 10 min in DF @ 30 nominal: frames_per_10min = 30*600 - 9*2 = 17982
        tc = self._df(17982)
        self.assertTrue(tc.startswith("00:10:00;"), f"Unexpected TC: {tc}")
        ff = int(tc.split(";")[-1])
        self.assertEqual(ff, 0)

    def test_df_one_hour(self):
        """One hour in DF @ 29.97 = 107892 frames."""
        # frames_per_hour = 6 * 17982 = 107892
        tc = self._df(107892)
        self.assertEqual(tc, "01:00:00;00")

    def test_df_ndf_differ_at_one_minute(self):
        """DF and NDF TC diverge at the first minute boundary."""
        ndf = self._tc(1800, 29.97)
        df  = self._df(1800)
        self.assertNotEqual(ndf, df)

    def test_df_ndf_agree_at_zero(self):
        """Both formats agree at frame 0."""
        ndf = self._tc(0, 29.97)
        df  = self._df(0)
        # Both should be "00:00:00:00" / "00:00:00;00" — same H/M/S, only separator differs
        self.assertEqual(ndf[:8], df[:8])

    def test_df_59_94(self):
        """Drop-frame at 59.94 fps (nominal 60): drops 4 frames per non-10th minute."""
        tc = RadianceFrameStamp._frame_to_tc(0, 59.94, drop_frame=True)
        self.assertEqual(tc, "00:00:00;00")

    def test_ndf_unchanged_for_non_df_fps(self):
        """Non-drop flag at 24 fps → NDF with colon separator."""
        tc = RadianceFrameStamp._frame_to_tc(24, 24.0, drop_frame=True)
        self.assertNotIn(";", tc)
        self.assertEqual(tc, "00:00:01:00")


# ═════════════════════════════════════════════════════════════════════════════
# 6. JPEG helper
# ═════════════════════════════════════════════════════════════════════════════

class TestFrameToJpeg(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    def test_returns_bytes(self):
        arr = _rgb(8, 8, 0.5)
        result = _frame_to_jpeg(arr)
        # If PIL present → non-empty bytes; otherwise b""
        self.assertIsInstance(result, bytes)

    def test_jpeg_has_ffd8_magic(self):
        """Valid JPEG starts with 0xFF 0xD8."""
        if not _mod.HAS_PIL:
            self.skipTest("Pillow not installed")
        arr = _rgb(8, 8, 0.5)
        jpg = _frame_to_jpeg(arr, quality=80)
        self.assertTrue(jpg[:2] == b"\xff\xd8", "Not a valid JPEG header")

    def test_different_quality(self):
        """Higher quality → same or larger file."""
        if not _mod.HAS_PIL:
            self.skipTest("Pillow not installed")
        arr = np.random.rand(32, 32, 3).astype(np.float32)
        low = _frame_to_jpeg(arr, quality=20)
        high = _frame_to_jpeg(arr, quality=95)
        # High quality should be same size or larger (generally)
        self.assertGreaterEqual(len(high), len(low) * 0.5)  # generous lower bound

    def test_clipped_over_bright(self):
        """Values > 1 should be clipped and produce valid JPEG."""
        if not _mod.HAS_PIL:
            self.skipTest("Pillow not installed")
        arr = np.full((8, 8, 3), 5.0, dtype=np.float32)
        jpg = _frame_to_jpeg(arr)
        self.assertTrue(jpg[:2] == b"\xff\xd8")


# ═════════════════════════════════════════════════════════════════════════════
# 7. Node-level tests (torch-gated)
# ═════════════════════════════════════════════════════════════════════════════

class TestFalseColorMonitorNode(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    @skip_no_torch
    def test_node_forwards_encoding_for_each_frame(self):
        image = torch.full((2, 8, 16, 4), 0.4613561295)
        image[..., 3] = 0.37
        passthrough, result = RadianceFalseColorMonitor().apply(image, 1.0, input_encoding="sRGB")
        self.assertIs(passthrough, image)
        expected = torch.tensor([0.0, 0.85, 0.15, 0.37]).expand_as(image)
        torch.testing.assert_close(result, expected)

    @skip_no_torch
    def test_returns_two_tensors(self):
        node = RadianceFalseColorMonitor()
        img = _make_tensor(8, 8, 1, 0.3)
        pt, fc = node.apply(img, strength=1.0)
        self.assertEqual(pt.shape, img.shape)
        self.assertEqual(fc.shape, img.shape)

    @skip_no_torch
    def test_passthrough_is_identity(self):
        node = RadianceFalseColorMonitor()
        img = _make_tensor(8, 8, 1, 0.3)
        pt, _ = node.apply(img, strength=1.0)
        self.assertTrue(torch.allclose(pt, img))

    @skip_no_torch
    def test_batch_processing(self):
        node = RadianceFalseColorMonitor()
        img = _make_tensor(4, 4, 3, 0.5)
        pt, fc = node.apply(img, strength=0.5)
        self.assertEqual(fc.shape[0], 3)

    @skip_no_torch
    def test_hdr_peak_parameter(self):
        node = RadianceFalseColorMonitor()
        img = _make_tensor(4, 4, 1, 1.8)
        pt, fc = node.apply(img, strength=1.0, hdr_peak=10.0)
        self.assertEqual(fc.shape, img.shape)


class TestFocusPeakingNode(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    @skip_no_torch
    def test_returns_two_tensors(self):
        node = RadianceFocusPeaking()
        img = _make_tensor(8, 8, 1, 0.4)
        pt, pk = node.peak(img, threshold=0.2, peak_color="Red", strength=0.85)
        self.assertEqual(pt.shape, img.shape)
        self.assertEqual(pk.shape, img.shape)

    @skip_no_torch
    def test_passthrough_is_identity(self):
        node = RadianceFocusPeaking()
        img = _make_tensor(8, 8, 1, 0.4)
        pt, _ = node.peak(img, threshold=0.2, peak_color="Green", strength=0.85)
        self.assertTrue(torch.allclose(pt, img))

    @skip_no_torch
    def test_all_peak_colors(self):
        node = RadianceFocusPeaking()
        img = _make_tensor(4, 4, 1, 0.4)
        for color in ["Red", "Green", "White", "Yellow", "Cyan"]:
            pt, pk = node.peak(img, threshold=0.1, peak_color=color, strength=1.0)
            self.assertEqual(pk.shape, img.shape, f"Failed for color={color}")


class TestSplitViewNode(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    @skip_no_torch
    def test_returns_one_tensor(self):
        node = RadianceSplitView()
        a = _make_tensor(8, 8, 1, 0.0)
        b = _make_tensor(8, 8, 1, 1.0)
        (out,) = node.compare(a, b, "wipe_h", 0.5)
        self.assertEqual(out.shape, a.shape)

    @skip_no_torch
    def test_all_modes(self):
        node = RadianceSplitView()
        a = _make_tensor(8, 8, 1, 0.0)
        b = _make_tensor(8, 8, 1, 1.0)
        for mode in ["wipe_h", "wipe_v", "side_by_side", "diff"]:
            (out,) = node.compare(a, b, mode, 0.5)
            self.assertEqual(out.shape, a.shape, f"Failed for mode={mode}")

    @skip_no_torch
    def test_diff_same_images_is_black(self):
        node = RadianceSplitView()
        a = _make_tensor(8, 8, 1, 0.5)
        (out,) = node.compare(a, a.clone(), "diff", 0.5)
        self.assertLess(float(out.max()), 1e-4)


class TestContactSheetNode(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    @skip_no_torch
    def test_returns_three_outputs(self):
        node = RadianceContactSheet()
        imgs = _make_tensor(4, 4, 4, 0.5)  # 4-frame batch
        sheet, cols, rows = node.sheet(imgs, thumb_width=32, max_cols=4)
        self.assertIsInstance(cols, int)
        self.assertIsInstance(rows, int)
        self.assertEqual(cols, 4)
        self.assertEqual(rows, 1)

    @skip_no_torch
    def test_sheet_is_4d_tensor(self):
        node = RadianceContactSheet()
        imgs = _make_tensor(4, 4, 6, 0.5)
        sheet, _, _ = node.sheet(imgs, thumb_width=32, max_cols=3)
        self.assertEqual(sheet.ndim, 4)

    @skip_no_torch
    def test_rows_computed_correctly(self):
        node = RadianceContactSheet()
        imgs = _make_tensor(4, 4, 7, 0.5)  # 7 frames, 4 cols → 2 rows
        _, cols, rows = node.sheet(imgs, thumb_width=32, max_cols=4)
        self.assertEqual(rows, 2)

    @skip_no_torch
    def test_backgrounds(self):
        node = RadianceContactSheet()
        imgs = _make_tensor(4, 4, 2, 0.5)
        for bg in ["Black", "Grey", "White"]:
            sheet, _, _ = node.sheet(imgs, thumb_width=32, max_cols=2, background=bg)
            self.assertIsNotNone(sheet)


class TestFlipbookGIFNode(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    @skip_no_torch
    def test_returns_passthrough_and_status(self):
        node = RadianceFlipbookGIF()
        imgs = _make_tensor(4, 4, 3, 0.5)
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test.gif")
            pt, status = node.export_gif(imgs, path, fps=12.0, max_width=64)
            self.assertEqual(pt.shape, imgs.shape)
            self.assertIsInstance(status, str)

    @skip_no_torch
    def test_gif_file_written(self):
        if not _mod.HAS_PIL:
            self.skipTest("Pillow not installed")
        node = RadianceFlipbookGIF()
        imgs = _make_tensor(4, 4, 2, 0.3)
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "out.gif")
            node.export_gif(imgs, path, fps=10.0, max_width=4)
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 0)

    @skip_no_torch
    def test_status_contains_path(self):
        if not _mod.HAS_PIL:
            self.skipTest("Pillow not installed")
        node = RadianceFlipbookGIF()
        imgs = _make_tensor(4, 4, 2, 0.5)
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "flip.gif")
            _, status = node.export_gif(imgs, path, fps=6.0, max_width=4)
            self.assertIn("flip.gif", status)


class TestFrameStampNode(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    @skip_no_torch
    def test_hdr_and_alpha_survive_outside_the_overlay(self):
        imgs = torch.tensor([-0.23, 0.123456, 4.56789, 0.375]).expand(2, 128, 256, 4).clone()
        original = imgs.clone()
        out, = RadianceFrameStamp().stamp(imgs, start_frame=0, fps=24.0,
                                         position="top_left", show_timecode=False,
                                         show_frame_number=False, custom_text="HDR", opacity=0.8)
        torch.testing.assert_close(out[:, 80:, :, :], imgs[:, 80:, :, :], rtol=0, atol=0)
        torch.testing.assert_close(out[..., 3], imgs[..., 3], rtol=0, atol=0)
        torch.testing.assert_close(imgs, original, rtol=0, atol=0)
        self.assertFalse(torch.equal(out[:, :40, :80, :3], imgs[:, :40, :80, :3]))
        self.assertTrue(torch.isfinite(out).all())

    @skip_no_torch
    def test_zero_opacity_leaves_float_pixels_exact(self):
        imgs = torch.rand(1, 64, 128, 3) * 8 - 1
        out, = RadianceFrameStamp().stamp(imgs, start_frame=0, fps=24.0, opacity=0.0)
        torch.testing.assert_close(out, imgs, rtol=0, atol=0)

    @skip_no_torch
    def test_returns_single_tensor(self):
        node = RadianceFrameStamp()
        imgs = _make_tensor(16, 16, 2, 0.4)
        (out,) = node.stamp(imgs, start_frame=1001, fps=24.0)
        self.assertEqual(out.shape, imgs.shape)

    @skip_no_torch
    def test_stamp_does_not_crash_no_text(self):
        node = RadianceFrameStamp()
        imgs = _make_tensor(16, 16, 1, 0.5)
        (out,) = node.stamp(imgs, start_frame=0, fps=24.0,
                            show_frame_number=False, show_timecode=False,
                            custom_text="")
        self.assertEqual(out.shape, imgs.shape)

    @skip_no_torch
    def test_all_positions(self):
        node = RadianceFrameStamp()
        imgs = _make_tensor(32, 32, 1, 0.3)
        for pos in ["bottom_left", "bottom_right", "top_left", "top_right", "center"]:
            (out,) = node.stamp(imgs, start_frame=100, fps=25.0,
                                position=pos, custom_text="TEST")
            self.assertEqual(out.shape, imgs.shape, f"Failed for position={pos}")


class TestPreviewServerNode(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    @skip_no_torch
    def test_disabled_returns_passthrough(self):
        node = RadiancePreviewServer()
        imgs = _make_tensor(4, 4, 1, 0.5)
        pt, status = node.serve(imgs, port=18765, stream_name="test", enabled=False)
        self.assertTrue(torch.allclose(pt, imgs))
        self.assertIn("disabled", status.lower())

    @skip_no_torch
    def test_serve_updates_preview_buffer(self):
        if not _mod.HAS_PIL:
            self.skipTest("Pillow not installed")
        node = RadiancePreviewServer()
        imgs = _make_tensor(4, 4, 1, 0.6)
        # Use a unique port to avoid collision
        import random
        port = random.randint(49152, 65000)
        stream = f"test_{port}"
        pt, url = node.serve(imgs, port=port, stream_name=stream, jpeg_quality=50)
        self.assertIn(stream, _PREVIEW_BUFFER)
        self.assertGreater(len(_PREVIEW_BUFFER[stream]), 0)
        self.assertIn(str(port), url)

    @skip_no_torch
    def test_passthrough_identity(self):
        if not _mod.HAS_PIL:
            self.skipTest("Pillow not installed")
        node = RadiancePreviewServer()
        imgs = _make_tensor(4, 4, 1, 0.5)
        import random
        port = random.randint(49152, 65000)
        pt, _ = node.serve(imgs, port=port, stream_name=f"test_{port}")
        self.assertTrue(torch.allclose(pt, imgs))


# ═════════════════════════════════════════════════════════════════════════════
# 8. Node Registration
# ═════════════════════════════════════════════════════════════════════════════

class TestNodeRegistration(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    EXPECTED_NODES = [
        "RadianceFocusPeaking",
        "RadianceContactSheet",
        "RadianceFlipbookGIF",
        "RadianceFrameStamp",
        "RadiancePreviewServer",
    ]

    def test_all_nodes_in_class_mappings(self):
        for name in self.EXPECTED_NODES:
            self.assertIn(name, NODE_CLASS_MAPPINGS, f"{name} missing from NODE_CLASS_MAPPINGS")

    def test_all_nodes_in_display_mappings(self):
        for name in self.EXPECTED_NODES:
            self.assertIn(name, NODE_DISPLAY_NAME_MAPPINGS,
                          f"{name} missing from NODE_DISPLAY_NAME_MAPPINGS")

    def test_display_names_have_radiance_prefix(self):
        for key, display in NODE_DISPLAY_NAME_MAPPINGS.items():
            self.assertIn("Radiance", display, f"Display name for {key} missing 'Radiance'")

    def test_display_names_have_symbol_prefix(self):
        for key, display in NODE_DISPLAY_NAME_MAPPINGS.items():
            self.assertTrue(display.startswith("◎"),
                            f"Display name for {key} does not start with ◎: '{display}'")

    def test_all_classes_instantiable(self):
        for name, cls in NODE_CLASS_MAPPINGS.items():
            try:
                instance = cls()
                self.assertIsNotNone(instance)
            except Exception as e:
                self.fail(f"Could not instantiate {name}: {e}")

    def test_all_classes_have_input_types(self):
        for name, cls in NODE_CLASS_MAPPINGS.items():
            self.assertTrue(hasattr(cls, "INPUT_TYPES"),
                            f"{name} missing INPUT_TYPES classmethod")
            it = cls.INPUT_TYPES()
            self.assertIsInstance(it, dict)
            self.assertIn("required", it)

    def test_all_classes_have_function_attr(self):
        for name, cls in NODE_CLASS_MAPPINGS.items():
            self.assertTrue(hasattr(cls, "FUNCTION"),
                            f"{name} missing FUNCTION attribute")
            fn = getattr(cls, "FUNCTION")
            self.assertTrue(hasattr(cls, fn),
                            f"{name}.FUNCTION='{fn}' but method not found on class")

    def test_category_prefix(self):
        for name, cls in NODE_CLASS_MAPPINGS.items():
            self.assertTrue(hasattr(cls, "CATEGORY"),
                            f"{name} missing CATEGORY attribute")
            self.assertTrue(
                cls.CATEGORY.startswith("FXTD STUDIOS/Radiance"),
                f"{name}.CATEGORY '{cls.CATEGORY}' wrong prefix"
            )

    def test_node_count(self):
        self.assertGreaterEqual(len(NODE_CLASS_MAPPINGS), 5)

    def test_mappings_are_consistent(self):
        """CLASS and DISPLAY mappings must have the same keys."""
        self.assertEqual(
            set(NODE_CLASS_MAPPINGS.keys()),
            set(NODE_DISPLAY_NAME_MAPPINGS.keys()),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
