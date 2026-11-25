"""Ported from RUDRA v1.3.0 tests/test_genfill_core.py (FXTD-Studios/RUDRA, Apache 2.0): the region re-render
finish of RUDRA 1.2.6 - 1.3.0 in radiance.model.rudra_genfill. The numbers are RUDRA's; only the imports
changed, and tests that need RUDRA's EXR tools or its tiny fill model are left out."""
import cv2  # noqa: F401
import numpy as np
import pytest  # noqa: F401

from radiance.model import rudra_genfill as composite
from radiance.model.rudra_genfill import MaskSettings, TemporalMask, blend_core, plan_windows  # noqa: F401


def test_blend_core_drops_specks_and_keeps_areas():
    # 1.2.6 (user feedback, 09V.mp4): a near-white door (250-254) peppered with single clipped pixels
    # got its fill in the specks only; areas under min_core of the frame are left to the master.
    from radiance.model.rudra_genfill import blend_core, drop_specks, min_core_pixels
    c = np.full((200, 200, 3), 252, np.uint8)
    c[::7, ::5] = 255                                   # 1,160 single clipped pixels
    c[18:42, 98:132] = 252                              # a ring of near-white round ...
    c[20:40, 100:130] = 255                             # ... one area of 600 px
    s = MaskSettings(min_core=0.01 - 1e-9)              # 399 px at 200 x 200
    core = blend_core(c, s)
    assert core[20:40, 100:130].all() and core.sum() == 600
    assert blend_core(c, MaskSettings(min_core=0.0)).sum() == (c.max(-1) == 255).sum()   # off: every clipped pixel
    assert min_core_pixels((1080, 1920), MaskSettings().min_core) == 103
    assert min_core_pixels((4, 4), MaskSettings().min_core) == 0                         # tiny frames keep all
    two = np.zeros((10, 10), bool); two[0, 0] = two[1, 1] = True                         # 8-connected: one area
    assert drop_specks(two, 2).sum() == 2
    with pytest.raises(ValueError):
        MaskSettings(min_core=0.5).check()

class _View:
    """A stand-in view: Reinhard-like display curve, clipped, per channel (monotone)."""
    def forward(self, x):
        x = np.maximum(x, 0)
        return np.clip(1.25 * x / (1 + x), 0, 1)

def test_finish_rerender_is_off_by_default_and_sets_hue_level_and_time():
    from radiance.model.rudra_genfill import LUMA_AP1, finish_rerender, white_scale
    rng = np.random.default_rng(3)
    h, w = 96, 128
    codes = np.full((h, w, 3), 120, np.uint8)
    codes[20:80, 30:110] = 254
    codes[20:80, 30:50] = 246                                                   # a darker band inside the region
    grey = np.power(codes.astype(np.float32) / 255.0, 2.4)
    source = grey * np.array([1.0, 1.05, 1.05], np.float32)
    master = grey * 0.9
    fill = (rng.uniform(1.0, 2.0, (h, w, 1)) * np.array([0.6, 1.0, 2.0])).astype(np.float32)   # blue, about 1.4
    weight = np.zeros((h, w), np.float32)
    weight[20:80, 30:110] = 1.0
    inside = weight > 0
    lum = lambda x: x @ LUMA_AP1                                                # noqa: E731
    assert finish_rerender(fill, master, source, codes, weight) is fill         # every step off: untouched
    white = white_scale(master, codes)
    assert abs(white - 0.9) < 1e-3
    out = finish_rerender(fill, master, source, codes, weight, hue='source', level=3.0, white=white)
    assert np.array_equal(out[~inside], fill[~inside])                          # outside the weight: as given
    assert abs(float(np.median(lum(out[30:70, 70:100]))) / white - 3.0) < 0.15  # clipped: the level, in SDR whites
    assert float(np.median(lum(out[30:70, 34:42]))) < 0.8 * float(np.median(lum(out[30:70, 70:100])))   # darker source, darker fill
    from radiance.model.rudra_genfill import RERENDER_CEILING_STOPS
    bright = finish_rerender(fill * 64.0, master, source, codes, weight, level=3.0, white=white)   # the model far brighter
    top = 3.0 * 2.0 ** RERENDER_CEILING_STOPS                                   # its own level only to the ceiling (0: not at all)
    assert abs(float(np.median(lum(bright[30:70, 70:100]))) / white - top) < 0.05 * top
    edge = finish_rerender(fill, master, source, np.full_like(codes, 240), weight, hue='source', level=3.0, white=white)
    assert np.allclose(edge[inside], master[inside], rtol=1e-3, atol=1e-5)      # at tau_high the fill is the master: no step
    core = out[30:70, 70:100].reshape(-1, 3).mean(0)
    assert abs(core[1] / core[0] - 1.05) < 0.01 and abs(core[2] / core[0] - 1.05) < 0.01   # the source's colour
    mixed = finish_rerender(fill, master, source, codes, weight, hue='source', colour=0.5)[30:70, 70:100].reshape(-1, 3).mean(0)
    assert 1.05 < mixed[2] / mixed[0] < 2.0 / 0.6                               # between the source's and the model's
    assert np.allclose(lum(finish_rerender(fill, master, source, codes, weight, hue='source')[inside]), lum(fill[inside]), rtol=1e-4)
    state = {}
    a = finish_rerender(fill, master, source, codes, weight, smooth=0.8, state=state)
    assert np.allclose(a[inside], fill[inside], rtol=1e-4)                      # first frame: nothing to follow
    b = finish_rerender(fill * 2.0, master, source, codes, weight, smooth=0.8, state=state)
    assert np.allclose(np.log2(lum(b[inside]) / lum(a[inside])), 0.2, atol=1e-3)   # a one-stop jump arrives as 0.2
    with pytest.raises(ValueError):
        finish_rerender(fill, master, source, codes, weight, hue='other')

def test_finish_rerender_holds_the_fill_where_the_source_holds_still():
    from radiance.model.rudra_genfill import LUMA_AP1, finish_rerender
    h, w = 32, 48
    codes = np.full((h, w, 3), 252, np.uint8)
    source = np.power(codes.astype(np.float32) / 255.0, 2.4)
    fill = np.full((h, w, 3), 2.0, np.float32)
    weight = np.ones((h, w), np.float32)
    lum = lambda x: x @ LUMA_AP1                                                # noqa: E731
    state = {}
    a = finish_rerender(fill, source, source, codes, weight, smooth=0.5, still=0.9, state=state)
    moved = codes.copy()
    moved[:, : w // 2] = 240                                                    # the left half of the source changes
    b = finish_rerender(fill * 2.0, source, source, moved, weight, smooth=0.5, still=0.9, state=state)
    step = np.log2(lum(b) / lum(a))
    assert np.allclose(step[:, w // 2:], 0.1, atol=1e-3)                       # held: 0.9 kept, a stop arrives as 0.1
    assert np.allclose(step[:, : w // 2], 0.5, atol=1e-3)                      # moved: smooth 0.5 stands
    with pytest.raises(ValueError):
        finish_rerender(fill, source, source, codes, weight, still=1.0)

def test_finish_rerender_calm_holds_a_jump_on_still_pixels_and_leaves_a_steady_fill():
    from radiance.model.rudra_genfill import LUMA_AP1, finish_rerender
    h, w = 32, 48
    codes = np.full((h, w, 3), 252, np.uint8)                                   # the source holds still everywhere
    source = np.power(codes.astype(np.float32) / 255.0, 2.4)
    fill = np.full((h, w, 3), 2.0, np.float32)
    weight = np.ones((h, w), np.float32)
    lum = lambda x: x @ LUMA_AP1                                                # noqa: E731
    state = {}
    a = finish_rerender(fill, source, source, codes, weight, smooth=0.5, calm=0.9, state=state)
    nxt = fill.copy()
    nxt[:, : w // 2] *= 2.0                                                     # left: the fill jumps a stop
    nxt[:, w // 2:] *= 2.0 ** 0.05                                              # right: it drifts 0.05 stops
    b = finish_rerender(nxt, source, source, codes, weight, smooth=0.5, calm=0.9, state=state)
    step = np.log2(lum(b) / lum(a))
    assert np.allclose(step[:, : w // 2], 0.1, atol=1e-3)                      # a jump: 0.9 kept
    assert np.allclose(step[:, w // 2:], 0.025, atol=1e-3)                     # steady: smooth 0.5 stands
    state = {}
    finish_rerender(fill, source, source, codes, weight, smooth=0.5, calm=0.9, state=state)
    moved = codes.copy()
    moved[:, : w // 2] = 240                                                    # the source itself moves: nothing is held
    c = finish_rerender(nxt, source, source, moved, weight, smooth=0.5, calm=0.9, state=state)
    assert np.allclose(np.log2(lum(c) / lum(a))[:, : w // 2], 0.5, atol=1e-3)
    off = finish_rerender(nxt, source, source, codes, weight, smooth=0.5, state={'log_y': state['log_y'] * 0, 'on': weight > 0, 'codes': codes})
    assert np.array_equal(off, finish_rerender(nxt, source, source, codes, weight, smooth=0.5, calm=0.0,
                                               state={'log_y': state['log_y'] * 0, 'on': weight > 0, 'codes': codes}))
    with pytest.raises(ValueError):
        finish_rerender(fill, source, source, codes, weight, calm=1.0)

def test_finish_rerender_rise_lifts_the_core_with_its_depth_and_nothing_else():
    from radiance.model.rudra_genfill import LUMA_AP1, finish_rerender
    h, w = 1080, 1080                                                           # px = 1: RERENDER_RISE_PX as written
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    r = np.hypot(yy - h / 2, xx - w / 2)
    code = np.clip(np.rint(270.0 - r * 0.08), 0, 254).astype(np.uint8)         # glare: a full-code core in a ramp
    codes = np.repeat(code[..., None], 3, -1)
    source = np.power(codes.astype(np.float32) / 255.0, 2.4)
    fill = np.full((h, w, 3), 1.0, np.float32)
    weight = (code >= 240).astype(np.float32)
    lum = lambda x: x @ LUMA_AP1                                                # noqa: E731
    flat = finish_rerender(fill, source, source, codes, weight, level=3.0, white=1.0)
    up = finish_rerender(fill, source, source, codes, weight, level=3.0, white=1.0, rise=1.0)
    assert np.allclose(finish_rerender(fill, source, source, codes, weight, level=3.0, white=1.0, rise=0.0), flat)
    gain = np.log2(lum(up) / lum(flat))
    assert np.allclose(gain[code <= 240], 0.0, atol=1e-6)                       # nothing at tau_high or under: no step at the rim
    assert float(gain[(code >= 241) & (code < 253)].max()) < 0.25               # the code ramp: at most a little
    assert float(gain[h // 2, w // 2]) > 0.5 and float(gain.max()) <= 1.0 + 1e-5   # deep in the core: towards rise, never over
    ring = (r > 100) & (r < 140)                                                # just inside the core's rim: a little
    assert 0.0 <= float(gain[ring].mean()) < float(gain[h // 2, w // 2])
    prof = gain[h // 2, w // 2: w // 2 + 200]
    assert np.all(np.diff(prof) <= 1e-5)                                        # climbing inward: no plate, no crease
    with pytest.raises(ValueError):
        finish_rerender(fill, source, source, codes, weight, level=3.0, rise=-1.0)
