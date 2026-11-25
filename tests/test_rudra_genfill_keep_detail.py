"""Ported from RUDRA v1.3.0 tests/test_release_1_2_7.py (FXTD-Studios/RUDRA, Apache 2.0): the region re-render
finish of RUDRA 1.2.6 - 1.3.0 in radiance.model.rudra_genfill. The numbers are RUDRA's; only the imports
changed, and tests that need RUDRA's EXR tools or its tiny fill model are left out."""
import cv2  # noqa: F401
import numpy as np
import pytest  # noqa: F401

from radiance.model import rudra_genfill as composite
from radiance.model.rudra_genfill import MaskSettings, TemporalMask, blend_core, plan_windows  # noqa: F401


def _frame(h=1080, w=1920):
    return np.full((h, w, 3), 13, np.uint8)

def test_keep_detail_leaves_a_grating_and_a_lamp_and_redraws_the_door():
    codes = _frame()
    codes[100:800, 600:1300] = 255                      # the door
    for x in range(600, 1300, 6):
        codes[900:1000, x:x + 2] = 255                  # a floor grating: thin bars
    codes[100:130, 100:120] = 255                       # a practical lamp, 600 px (< 0.05 % of the frame)
    w = composite.detail_keep_weight(codes, 240)
    assert w.dtype == np.float32 and w.shape == codes.shape[:2]
    assert w[450, 950] == pytest.approx(1.0)            # the door is redrawn
    assert w[950, 600] < 0.5                            # the grating keeps the master
    assert w[115, 110] == 0.0                           # the lamp too
    assert w[50, 50] == 0.0                             # outside the region: nothing
    assert float((w[100:800, 600:1300] >= 0.5).mean()) > 0.97

def test_keep_detail_is_off_unless_asked():
    from radiance.model.rudra_genfill import Rerender
    r = Rerender()
    assert r.keep_detail is False
