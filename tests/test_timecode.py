"""
tests/test_timecode.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Tests for RadianceFrameStamp._frame_to_tc  (SMPTE 12M timecode)

Coverage
────────
  NDF 24      — basic frames, sub-second, minute, hour roll-overs
  NDF 25      — PAL rates
  NDF 30      — integer 30 fps (NDF)
  DF 29.97    — SMPTE drop-frame at nominal 30 fps (d=2)
               • first frame of each minute has frames 0,1 dropped
               • every 10th minute is NOT dropped (on-the-dot 10/20/30...)
               • key boundary frames: 0, 1799, 1800, 3597, 3598, 17981,
                 17982, 107891, 107892
  DF 59.94    — SMPTE drop-frame at nominal 60 fps (d=4)
               • same structural logic; drop 0-3 at non-10th minutes
  Separator   — ';' for DF, ':' for NDF
  Constants   — _DF_RATES maps {30:2, 60:4}
  Node reg    — RadianceFrameStamp in NODE_CLASS_MAPPINGS

Reference values were computed independently from the SMPTE 12M formula
and cross-checked in the module docstring.
"""

from __future__ import annotations

import os
import sys
import unittest

# ── path setup ───────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch  # noqa: F401  (nodes_realtime_preview imports torch at top-level)

from radiance.nodes.monitor.realtime import (
    RadianceFrameStamp,
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)

# This module already gates its torch-dependent tests correctly (they skip
# cleanly against conftest's stub), so opt out of the automatic module-level
# skip and keep the rest of the file running on the no-torch CI matrix.
RADIANCE_TORCH_GATED = True


_ftc = RadianceFrameStamp._frame_to_tc   # shorthand


# ═════════════════════════════════════════════════════════════════════════════
# 1. NDF — Non-drop frame
# ═════════════════════════════════════════════════════════════════════════════

class TestNDF24(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    """24 fps non-drop (film standard)."""

    def _tc(self, frame: int) -> str:
        return _ftc(frame, fps=24.0, drop_frame=False)

    def test_first_frame(self):
        self.assertEqual(self._tc(0), "00:00:00:00")

    def test_second_frame(self):
        self.assertEqual(self._tc(1), "00:00:00:01")

    def test_last_frame_of_second(self):
        self.assertEqual(self._tc(23), "00:00:00:23")

    def test_first_frame_second_second(self):
        self.assertEqual(self._tc(24), "00:00:01:00")

    def test_last_frame_of_minute(self):
        # 24 × 60 − 1 = 1439
        self.assertEqual(self._tc(1439), "00:00:59:23")

    def test_first_frame_second_minute(self):
        self.assertEqual(self._tc(1440), "00:01:00:00")

    def test_hour_rollover(self):
        # 24 × 3600 = 86400 frames/hour
        self.assertEqual(self._tc(86400), "01:00:00:00")

    def test_separator_is_colon(self):
        tc = self._tc(100)
        self.assertEqual(tc.count(":"), 3)
        self.assertNotIn(";", tc)


class TestNDF25(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    """25 fps non-drop (PAL broadcast)."""

    def _tc(self, frame: int) -> str:
        return _ftc(frame, fps=25.0, drop_frame=False)

    def test_first_frame(self):
        self.assertEqual(self._tc(0), "00:00:00:00")

    def test_last_frame_of_second(self):
        self.assertEqual(self._tc(24), "00:00:00:24")

    def test_minute_boundary(self):
        # 25 × 60 = 1500 frames/min
        self.assertEqual(self._tc(1500), "00:01:00:00")
        self.assertEqual(self._tc(1499), "00:00:59:24")

    def test_hour_rollover(self):
        self.assertEqual(self._tc(25 * 3600), "01:00:00:00")


class TestNDF30(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    """30 fps non-drop."""

    def _tc(self, frame: int) -> str:
        return _ftc(frame, fps=30.0, drop_frame=False)

    def test_first_frame(self):
        self.assertEqual(self._tc(0), "00:00:00:00")

    def test_last_frame_of_second(self):
        self.assertEqual(self._tc(29), "00:00:00:29")

    def test_second_boundary(self):
        self.assertEqual(self._tc(30), "00:00:01:00")

    def test_minute_boundary(self):
        # 30 × 60 = 1800 frames/min
        self.assertEqual(self._tc(1799), "00:00:59:29")
        self.assertEqual(self._tc(1800), "00:01:00:00")

    def test_separator_is_colon(self):
        self.assertNotIn(";", self._tc(999))


# ═════════════════════════════════════════════════════════════════════════════
# 2. DF 29.97  (nominal 30 fps, d=2)
# ═════════════════════════════════════════════════════════════════════════════

class TestDF2997(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    """
    29.97 DF — SMPTE 12M with d=2.

    Key constants
    ─────────────
      frames_per_10min = 30×600 − 9×2 = 17 982
      frames_per_hour  = 6 × 17 982   = 107 892

    Minute structure within each 10-min block
      Minute  0 : frames   0 – 1 799  (1 800 frames, no drop)
      Minute  1 : frames 1 800 – 3 597  (1 798 frames, TC starts ;02)
      Minute  2 : frames 3 598 – 5 395  (1 798 frames, TC starts ;02)
      ...
      Minute  9 : frames 16 184 – 17 981
    """

    def _tc(self, frame: int) -> str:
        return _ftc(frame, fps=29.97, drop_frame=True)

    # ── frame 0 ──────────────────────────────────────────────────────────────
    def test_frame_zero(self):
        self.assertEqual(self._tc(0), "00:00:00;00")

    # ── first second ─────────────────────────────────────────────────────────
    def test_frame_one(self):
        self.assertEqual(self._tc(1), "00:00:00;01")

    def test_frame_29(self):
        self.assertEqual(self._tc(29), "00:00:00;29")

    def test_frame_30(self):
        self.assertEqual(self._tc(30), "00:00:01;00")

    # ── end of first minute (no drop in minute 0) ────────────────────────────
    def test_last_frame_minute_zero(self):
        # frame 1799 = last frame of the full first minute
        self.assertEqual(self._tc(1799), "00:00:59;29")

    # ── start of minute 1 — frames 0 and 1 are DROPPED ───────────────────────
    def test_first_frame_minute_one(self):
        # frame 1800 is the first frame of minute 1; TC jumps to ;02
        self.assertEqual(self._tc(1800), "00:01:00;02")

    def test_second_frame_minute_one(self):
        self.assertEqual(self._tc(1801), "00:01:00;03")

    # ── end of minute 1 ──────────────────────────────────────────────────────
    def test_last_frame_minute_one(self):
        # minute 1 has 1798 frames: 1800..3597
        self.assertEqual(self._tc(3597), "00:01:59;29")

    # ── start of minute 2 ────────────────────────────────────────────────────
    def test_first_frame_minute_two(self):
        self.assertEqual(self._tc(3598), "00:02:00;02")

    def test_second_frame_minute_two(self):
        self.assertEqual(self._tc(3599), "00:02:00;03")

    # ── end of 10-min block ───────────────────────────────────────────────────
    def test_last_frame_10min_block(self):
        # frame 17981 = last frame of block 0
        self.assertEqual(self._tc(17981), "00:09:59;29")

    # ── start of second 10-min block (minute 10) ─────────────────────────────
    def test_first_frame_second_10min_block(self):
        # On-the-dot 10-min boundaries are NOT dropped
        self.assertEqual(self._tc(17982), "00:10:00;00")

    def test_second_frame_second_10min_block(self):
        self.assertEqual(self._tc(17983), "00:10:00;01")

    # ── end of first hour ────────────────────────────────────────────────────
    def test_last_frame_of_first_hour(self):
        self.assertEqual(self._tc(107891), "00:59:59;29")

    def test_first_frame_of_second_hour(self):
        self.assertEqual(self._tc(107892), "01:00:00;00")

    # ── separator ────────────────────────────────────────────────────────────
    def test_separator_is_semicolon(self):
        tc = self._tc(1800)
        self.assertIn(";", tc)
        # Should have exactly 2 colons and 1 semicolon
        self.assertEqual(tc.count(":"), 2)
        self.assertEqual(tc.count(";"), 1)

    # ── frames that must NEVER appear in DF ──────────────────────────────────
    def test_dropped_frames_absent(self):
        """Frames ;00 and ;01 must not appear at non-10th-minute starts."""
        # Non-10th minute boundaries: 1, 2, ..., 9 (within first 10-min block)
        for min_offset in range(1, 10):
            # First frame of each short minute
            frame = 1800 + (min_offset - 1) * 1798
            tc = self._tc(frame)
            ff = int(tc.split(";")[1])
            self.assertGreaterEqual(ff, 2,
                f"minute {min_offset} first frame has ff={ff} (should be ≥2): {tc}")

    # ── monotonicity check ────────────────────────────────────────────────────
    def test_tc_monotone_within_minute(self):
        """Consecutive frames within a minute must produce non-decreasing TC."""
        prev = self._tc(1800)
        for f in range(1801, 1830):
            cur = self._tc(f)
            self.assertGreaterEqual(cur, prev,
                f"TC went backwards: {prev} → {cur} at frame {f}")
            prev = cur


# ═════════════════════════════════════════════════════════════════════════════
# 3. DF 59.94  (nominal 60 fps, d=4)
# ═════════════════════════════════════════════════════════════════════════════

class TestDF5994(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    """
    59.94 DF — SMPTE with d=4.

    Key constants
    ─────────────
      frames_per_10min = 60×600 − 9×4 = 35 964
      frames_per_hour  = 6 × 35 964   = 215 784

    Minute structure within each 10-min block
      Minute  0 : frames     0 – 3 599  (3 600 frames, no drop)
      Minute  1 : frames 3 600 – 7 195  (3 596 frames, TC starts ;04)
      Minute  2 : frames 7 196 – 10 791 (3 596 frames, TC starts ;04)
    """

    def _tc(self, frame: int) -> str:
        return _ftc(frame, fps=59.94, drop_frame=True)

    def test_frame_zero(self):
        self.assertEqual(self._tc(0), "00:00:00;00")

    def test_last_frame_of_first_second(self):
        self.assertEqual(self._tc(59), "00:00:00;59")

    def test_second_boundary(self):
        self.assertEqual(self._tc(60), "00:00:01;00")

    def test_last_frame_minute_zero(self):
        # 60 × 60 − 1 = 3599
        self.assertEqual(self._tc(3599), "00:00:59;59")

    def test_first_frame_minute_one(self):
        # drops 0,1,2,3 → starts at ;04
        self.assertEqual(self._tc(3600), "00:01:00;04")

    def test_second_frame_minute_one(self):
        self.assertEqual(self._tc(3601), "00:01:00;05")

    def test_last_frame_minute_one(self):
        # minute 1 has 3596 frames: 3600..7195
        self.assertEqual(self._tc(7195), "00:01:59;59")

    def test_first_frame_minute_two(self):
        self.assertEqual(self._tc(7196), "00:02:00;04")

    def test_end_of_10min_block(self):
        self.assertEqual(self._tc(35963), "00:09:59;59")

    def test_start_of_second_10min_block(self):
        self.assertEqual(self._tc(35964), "00:10:00;00")

    def test_last_frame_of_first_hour(self):
        self.assertEqual(self._tc(215783), "00:59:59;59")

    def test_first_frame_of_second_hour(self):
        self.assertEqual(self._tc(215784), "01:00:00;00")

    def test_separator_is_semicolon(self):
        tc = self._tc(3600)
        self.assertIn(";", tc)

    def test_dropped_frames_absent_60df(self):
        """Frames ;00, ;01, ;02, ;03 must not appear at non-10th-minute starts."""
        for min_offset in range(1, 10):
            frame = 3600 + (min_offset - 1) * 3596
            tc = self._tc(frame)
            ff = int(tc.split(";")[1])
            self.assertGreaterEqual(ff, 4,
                f"60DF minute {min_offset} first frame has ff={ff}: {tc}")


# ═════════════════════════════════════════════════════════════════════════════
# 4. _DF_RATES constants
# ═════════════════════════════════════════════════════════════════════════════

class TestDFRatesConstant(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    def test_30fps_drops_2(self):
        self.assertEqual(RadianceFrameStamp._DF_RATES[30], 2)

    def test_60fps_drops_4(self):
        self.assertEqual(RadianceFrameStamp._DF_RATES[60], 4)

    def test_only_supported_rates(self):
        # Only 30 and 60 are SMPTE DF rates; 24/25/48/50 are NDF only
        for ndf_fps in [24, 25, 48, 50]:
            self.assertNotIn(ndf_fps, RadianceFrameStamp._DF_RATES)

    def test_frames_per_10min_30df(self):
        d = RadianceFrameStamp._DF_RATES[30]
        self.assertEqual(30 * 600 - 9 * d, 17982)

    def test_frames_per_10min_60df(self):
        d = RadianceFrameStamp._DF_RATES[60]
        self.assertEqual(60 * 600 - 9 * d, 35964)

    def test_frames_per_hour_30df(self):
        d = RadianceFrameStamp._DF_RATES[30]
        fpm10 = 30 * 600 - 9 * d
        self.assertEqual(6 * fpm10, 107892)

    def test_frames_per_hour_60df(self):
        d = RadianceFrameStamp._DF_RATES[60]
        fpm10 = 60 * 600 - 9 * d
        self.assertEqual(6 * fpm10, 215784)


# ═════════════════════════════════════════════════════════════════════════════
# 5. NDF with drop_frame=True on non-DF rates (must not crash)
# ═════════════════════════════════════════════════════════════════════════════

class TestDFRequestedButNotSupported(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    """
    Rates like 24 and 25 have no SMPTE DF definition.
    Passing drop_frame=True should fall through to NDF silently.
    """

    def test_24fps_df_flag_falls_through(self):
        # Should not crash; result should be identical to NDF 24
        tc_ndf = _ftc(1440, fps=24.0, drop_frame=False)
        tc_df  = _ftc(1440, fps=24.0, drop_frame=True)
        self.assertEqual(tc_ndf, tc_df)

    def test_25fps_df_flag_falls_through(self):
        tc_ndf = _ftc(1500, fps=25.0, drop_frame=False)
        tc_df  = _ftc(1500, fps=25.0, drop_frame=True)
        self.assertEqual(tc_ndf, tc_df)


# ═════════════════════════════════════════════════════════════════════════════
# 6. Node registration
# ═════════════════════════════════════════════════════════════════════════════

class TestNodeRegistration(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    def test_frame_stamp_registered(self):
        self.assertIn("RadianceFrameStamp", NODE_CLASS_MAPPINGS)

    def test_frame_stamp_display_name(self):
        self.assertIn("RadianceFrameStamp", NODE_DISPLAY_NAME_MAPPINGS)

    def test_frame_stamp_has_function(self):
        cls = NODE_CLASS_MAPPINGS["RadianceFrameStamp"]
        self.assertTrue(hasattr(cls, "FUNCTION"))

    def test_frame_stamp_has_category(self):
        cls = NODE_CLASS_MAPPINGS["RadianceFrameStamp"]
        self.assertTrue(hasattr(cls, "CATEGORY"))

    def test_frame_stamp_category_under_radiance(self):
        cls = NODE_CLASS_MAPPINGS["RadianceFrameStamp"]
        cat = getattr(cls, "CATEGORY", "")
        self.assertIn("Radiance", cat)

    def test_frame_stamp_input_types_callable(self):
        cls = NODE_CLASS_MAPPINGS["RadianceFrameStamp"]
        it = cls.INPUT_TYPES()
        self.assertIsInstance(it, dict)

    def test_frame_stamp_has_images_input(self):
        cls = NODE_CLASS_MAPPINGS["RadianceFrameStamp"]
        it  = cls.INPUT_TYPES()
        req = it.get("required", {})
        self.assertIn("images", req)

    def test_frame_stamp_has_fps_input(self):
        cls = NODE_CLASS_MAPPINGS["RadianceFrameStamp"]
        it  = cls.INPUT_TYPES()
        req = it.get("required", {})
        opt = it.get("optional", {})
        self.assertIn("fps", {**req, **opt})

    def test_frame_stamp_has_drop_frame_input(self):
        cls = NODE_CLASS_MAPPINGS["RadianceFrameStamp"]
        it  = cls.INPUT_TYPES()
        req = it.get("required", {})
        opt = it.get("optional", {})
        self.assertIn("drop_frame", {**req, **opt})

    def test_frame_stamp_return_types_contain_image(self):
        cls = NODE_CLASS_MAPPINGS["RadianceFrameStamp"]
        self.assertIn("IMAGE", cls.RETURN_TYPES)


if __name__ == "__main__":
    unittest.main()
