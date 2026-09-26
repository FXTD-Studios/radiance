"""
tests/test_scene_cut.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Tests for nodes_scene_cut — Scene-cut detection + per-shot routing

Coverage
────────
  Pure numpy helpers  : _luminance, _histogram_diff, _edge_diff, detect_cuts
  Node registration   : RadianceSceneCutDetect, Split, ShotGradeRouter
"""

from __future__ import annotations

import json
import os
import sys
import unittest

import numpy as np
import torch

HAS_TORCH = hasattr(torch, "__version__")
skip_no_torch = unittest.skipUnless(HAS_TORCH, "real torch not available")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from radiance.nodes.ai.scene_cut import (
    _luminance,
    _histogram_diff,
    _edge_diff,
    detect_cuts,
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    RadianceSceneCutDetect,
    RadianceSceneCutSplit,
    RadianceShotGradeRouter,
)

# This module already gates its torch-dependent tests correctly (they skip
# cleanly against conftest's stub), so opt out of the automatic module-level
# skip and keep the rest of the file running on the no-torch CI matrix.
RADIANCE_TORCH_GATED = True



def _batch(b, h=8, w=8, fill=0.5):
    return np.full((b, h, w, 3), fill, dtype=np.float32)


def _cut_sequence(n_frames=20, cut_at=10, h=8, w=8):
    """Two-shot sequence: frames 0..cut_at-1 black, frames cut_at..n-1 white."""
    frames = np.zeros((n_frames, h, w, 3), dtype=np.float32)
    frames[cut_at:] = 1.0
    return frames


# ═════════════════════════════════════════════════════════════════════════════
# Helper: _luminance
# ═════════════════════════════════════════════════════════════════════════════

class TestLuminance(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    def test_shape(self):
        f = _batch(5, 8, 8)
        lum = _luminance(f)
        self.assertEqual(lum.shape, (5,))

    def test_black_is_zero(self):
        f = _batch(3, 8, 8, 0.0)
        lum = _luminance(f)
        np.testing.assert_allclose(lum, 0.0, atol=1e-6)

    def test_white_near_one(self):
        f = _batch(3, 8, 8, 1.0)
        lum = _luminance(f)
        np.testing.assert_allclose(lum, 1.0, atol=1e-4)

    def test_grey_matches_value(self):
        f = _batch(1, 8, 8, 0.5)
        lum = _luminance(f)
        self.assertAlmostEqual(float(lum[0]), 0.5, places=4)

    def test_different_lumas(self):
        f = np.stack([
            np.full((8, 8, 3), 0.2, dtype=np.float32),
            np.full((8, 8, 3), 0.8, dtype=np.float32),
        ])
        lum = _luminance(f)
        self.assertLess(lum[0], lum[1])


# ═════════════════════════════════════════════════════════════════════════════
# Helper: _histogram_diff
# ═════════════════════════════════════════════════════════════════════════════

class TestHistogramDiff(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    def test_identical_frames_zero(self):
        f = np.full((8, 8, 3), 0.5, dtype=np.float32)
        self.assertAlmostEqual(_histogram_diff(f, f.copy()), 0.0, places=4)

    def test_black_vs_white_large(self):
        a = np.zeros((8, 8, 3), dtype=np.float32)
        b = np.ones((8, 8, 3), dtype=np.float32)
        diff = _histogram_diff(a, b)
        self.assertGreater(diff, 0.5)

    def test_non_negative(self):
        a = np.random.rand(8, 8, 3).astype(np.float32)
        b = np.random.rand(8, 8, 3).astype(np.float32)
        self.assertGreaterEqual(_histogram_diff(a, b), 0.0)

    def test_symmetry(self):
        a = np.random.rand(8, 8, 3).astype(np.float32)
        b = np.random.rand(8, 8, 3).astype(np.float32)
        self.assertAlmostEqual(_histogram_diff(a, b), _histogram_diff(b, a), places=5)


# ═════════════════════════════════════════════════════════════════════════════
# Helper: _edge_diff
# ═════════════════════════════════════════════════════════════════════════════

class TestEdgeDiff(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    def test_identical_near_zero(self):
        f = np.full((8, 8, 3), 0.5, dtype=np.float32)
        self.assertAlmostEqual(_edge_diff(f, f.copy()), 0.0, places=5)

    def test_different_edge_patterns(self):
        # a: horizontal edge at row 4
        a = np.zeros((16, 16, 3), dtype=np.float32)
        a[8:] = 1.0
        # b: vertical edge at col 8
        b = np.zeros((16, 16, 3), dtype=np.float32)
        b[:, 8:] = 1.0
        diff = _edge_diff(a, b)
        self.assertGreater(diff, 0.0)

    def test_non_negative(self):
        a = np.random.rand(8, 8, 3).astype(np.float32)
        b = np.random.rand(8, 8, 3).astype(np.float32)
        self.assertGreaterEqual(_edge_diff(a, b), 0.0)


# ═════════════════════════════════════════════════════════════════════════════
# detect_cuts
# ═════════════════════════════════════════════════════════════════════════════

class TestDetectCuts(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    def test_single_frame_returns_zero(self):
        f = _batch(1, 4, 4)
        cuts, scores = detect_cuts(f)
        self.assertEqual(cuts, [0])
        self.assertEqual(len(scores), 1)

    def test_always_includes_frame_zero(self):
        f = _batch(10, 8, 8, 0.5)
        cuts, _ = detect_cuts(f)
        self.assertEqual(cuts[0], 0)

    def test_identical_frames_no_cut(self):
        f = _batch(10, 8, 8, 0.5)
        cuts, _ = detect_cuts(f, threshold=0.2)
        self.assertEqual(cuts, [0])   # no cuts except the implicit start

    def test_hard_cut_detected(self):
        f = _cut_sequence(n_frames=24, cut_at=12, h=16, w=16)
        cuts, _ = detect_cuts(f, threshold=0.3, min_shot_frames=4)
        self.assertGreater(len(cuts), 1)
        # The detected cut should be near frame 12
        self.assertTrue(any(abs(c - 12) <= 2 for c in cuts))

    def test_scores_shape(self):
        f = _batch(10, 8, 8)
        _, scores = detect_cuts(f)
        self.assertEqual(len(scores), 9)   # B-1

    def test_min_shot_frames_respected(self):
        """Two cuts closer than min_shot_frames → only one detected."""
        f = np.zeros((30, 8, 8, 3), dtype=np.float32)
        f[5:10]  = 1.0   # short shot
        f[10:20] = 0.5
        f[20:]   = 0.0
        cuts, _ = detect_cuts(f, threshold=0.1, min_shot_frames=15)
        # With min_shot_frames=15, the second close cut should be merged
        self.assertLessEqual(len(cuts), 3)

    def test_methods_histogram(self):
        f = _cut_sequence(16, 8, h=8, w=8)
        cuts, _ = detect_cuts(f, method="histogram", threshold=0.3, min_shot_frames=4)
        self.assertGreater(len(cuts), 1)

    def test_methods_edge(self):
        # Flat-colour frames have zero edge maps — need spatial variation with
        # *different edge orientations* so Sobel diff detects the cut.
        # Shot A: vertical edge (left half bright) → column-wise gradient
        # Shot B: horizontal edge (top half bright) → row-wise gradient
        frames = np.zeros((16, 8, 8, 3), dtype=np.float32)
        frames[:8, :, :4, :] = 1.0   # shot A: vertical edge at col 4
        frames[8:, :4, :, :] = 1.0   # shot B: horizontal edge at row 4
        # threshold is now an ABSOLUTE distance, and the edge metric runs on a
        # smaller scale than the histogram one — an orientation flip like this
        # scores ~0.22. Under the old batch-max normalisation any non-zero
        # score could be pushed over any threshold, which is exactly why the
        # number was meaningless.
        cuts, _ = detect_cuts(frames, method="edge", threshold=0.15, min_shot_frames=4)
        self.assertGreater(len(cuts), 1)

    def test_methods_combined(self):
        f = _cut_sequence(16, 8, h=8, w=8)
        cuts, _ = detect_cuts(f, method="combined", threshold=0.3, min_shot_frames=4)
        self.assertGreater(len(cuts), 1)

    def test_scores_non_negative(self):
        f = _batch(5, 8, 8)
        _, scores = detect_cuts(f)
        self.assertTrue((scores >= 0).all())


# ═════════════════════════════════════════════════════════════════════════════
# Node Registration
# ═════════════════════════════════════════════════════════════════════════════

class TestSceneCutRegistration(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    EXPECTED = ["RadianceSceneCutDetect", "RadianceSceneCutSplit", "RadianceShotGradeRouter"]

    def test_all_in_class_mappings(self):
        for name in self.EXPECTED:
            self.assertIn(name, NODE_CLASS_MAPPINGS)

    def test_all_in_display_mappings(self):
        for name in self.EXPECTED:
            self.assertIn(name, NODE_DISPLAY_NAME_MAPPINGS)

    def test_display_names_prefix(self):
        for _, disp in NODE_DISPLAY_NAME_MAPPINGS.items():
            self.assertTrue(disp.startswith("◎"))

    def test_all_instantiable(self):
        for name, cls in NODE_CLASS_MAPPINGS.items():
            try:
                cls()
            except Exception as e:
                self.fail(f"{name} could not be instantiated: {e}")

    def test_input_types_present(self):
        for name, cls in NODE_CLASS_MAPPINGS.items():
            it = cls.INPUT_TYPES()
            self.assertIn("required", it)


if __name__ == "__main__":
    unittest.main(verbosity=2)
