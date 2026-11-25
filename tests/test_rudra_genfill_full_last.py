"""Ported from RUDRA v1.3.0 tests/test_release_1_2_9.py (FXTD-Studios/RUDRA, Apache 2.0): the region re-render
finish of RUDRA 1.2.6 - 1.3.0 in radiance.model.rudra_genfill. The numbers are RUDRA's; only the imports
changed, and tests that need RUDRA's EXR tools or its tiny fill model are left out."""
import cv2  # noqa: F401
import numpy as np
import pytest  # noqa: F401

from radiance.model import rudra_genfill as composite
from radiance.model.rudra_genfill import MaskSettings, TemporalMask, blend_core, plan_windows  # noqa: F401


def test_full_last_makes_the_last_window_whole_and_changes_nothing_else():
    from radiance.model.rudra_genfill import plan_windows
    assert plan_windows(97, 33, 8, full_last=True) == [(0, 0, 33), (25, 33, 58), (50, 58, 83), (64, 83, 97)]
    assert plan_windows(97, 33, 8)[-1] == (75, 83, 97)                   # off: as 1.2.8
    assert plan_windows(20, 33, 8, full_last=True) == [(0, 0, 20)]       # shorter than a window: one window
    for n in range(1, 140):
        for ov in (0, 8, 16):
            p = plan_windows(n, 33, ov, full_last=True)
            written = [i for s, nw, e in p for i in range(nw, e)]
            assert written == list(range(n))                              # every frame written once, in order
            assert all(e - s <= 33 for s, nw, e in p)
            assert all(e - s == 33 for s, nw, e in p) or len(p) == 1     # every window whole

