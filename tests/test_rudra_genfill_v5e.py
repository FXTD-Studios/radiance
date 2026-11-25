"""Ported from RUDRA v1.3.0 tests/test_genfill_v5e_learned.py (FXTD-Studios/RUDRA, Apache 2.0): the region re-render
finish of RUDRA 1.2.6 - 1.3.0 in radiance.model.rudra_genfill. The numbers are RUDRA's; only the imports
changed, and tests that need RUDRA's EXR tools or its tiny fill model are left out."""
import cv2  # noqa: F401
import numpy as np
import pytest  # noqa: F401

from radiance.model import rudra_genfill as composite
from radiance.model.rudra_genfill import MaskSettings, TemporalMask, blend_core, plan_windows  # noqa: F401

ARGS = dict(hue="source", colour=0.3, level=3.3, smooth=0.8, white=0.8, tau_high=240, dip=0.3, calm=0.9, steady=7,
            hold="thin")                                # the app's finish with v5

def _region(h=270, w=480):
    """A dark wall (code 60) with a clipped patch whose edge climbs through the crossfade (240 .. 255)."""
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(yy - h / 2, xx - w / 2)
    top = np.clip(255 - (r - 60) * 0.5, 60, 255).astype(np.uint8)   # 255 inside r 60, down to 240 at r 90
    top[r > 90] = 60
    codes = np.repeat(top[..., None], 3, -1)
    weight = (codes.max(-1) >= 240).astype(np.float32)
    source = np.power(codes.astype(np.float32) / 255.0, 2.4)
    return codes, source, weight

def _run(master_level, master_noise, cap, frames=10, seed=3):
    """The finish on a still shot: v5's fill (2.0 with 0.1 stops of detail drawn anew each frame) over a master at
    `master_level` (linear) that moves `master_noise` stops from frame to frame (the learned master near the clip)."""
    codes, source, weight = _region()
    rng = np.random.default_rng(seed)
    state, outs = {}, []
    for _ in range(frames):
        d = cv2.GaussianBlur(rng.standard_normal(codes.shape[:2]).astype(np.float32), (0, 0), 3.0)
        fill = np.repeat((2.0 * np.exp2(d * np.float32(0.1 / d.std())))[..., None], 3, -1).astype(np.float32)
        n = rng.standard_normal(codes.shape[:2]).astype(np.float32) * np.float32(master_noise)
        master = np.repeat((np.float32(master_level) * np.exp2(n))[..., None], 3, -1).astype(np.float32)
        master[codes.max(-1) < 240] = 0.05
        outs.append(composite.finish_rerender(fill, master, source, codes, weight, state=state, cap=cap, **ARGS))
    return codes, outs

def _p95_still_clipped(frames, codes, skip=7):
    y = [np.log2(np.maximum(f @ composite.LUMA_AP1, 1e-6)) for f in frames[skip:]]
    sel = codes.max(-1) >= 250
    return float(np.percentile(np.concatenate([np.abs(b - a)[sel] for a, b in zip(y, y[1:])]), 95))

def test_the_cap_keeps_a_bright_unsteady_learned_master_out_of_the_region():
    # As the learned master near the clip: 6 (7.5 x white 0.8), with 0.5 stops of frame-to-frame noise.
    codes, plain = _run(6.0, 0.5, cap=False)
    _, capped = _run(6.0, 0.5, cap=True)
    t_plain, t_cap = _p95_still_clipped(plain, codes), _p95_still_clipped(capped, codes)
    assert t_cap < 0.5 * t_plain, (t_plain, t_cap)
    # ... and the rim is no brighter than the core: the median over code >= 245, as the L gate takes it
    ring = (codes.max(-1) >= 245) & (codes.max(-1) < 253)
    core = codes.max(-1) >= 255
    y, yp = capped[-1] @ composite.LUMA_AP1, plain[-1] @ composite.LUMA_AP1
    assert np.median(y[ring]) <= np.median(y[core]) * 1.05
    assert np.median(yp[ring]) > np.median(yp[core])    # without it the master's rim stood over the fill's core

def test_the_cap_changes_nothing_where_the_master_sits_under_the_level():
    # As the analytic master: near white in the clipped area, under the fill's level (3.3 x white 0.8 = 2.64).
    _, plain = _run(0.8, 0.0, cap=False)
    _, capped = _run(0.8, 0.0, cap=True)
    for a, b in zip(plain, capped):
        assert np.array_equal(a, b)

def test_the_cap_is_off_by_default():
    codes, source, weight = _region()
    fill = np.full(codes.shape, 2.0, np.float32)
    master = np.full(codes.shape, 6.0, np.float32)
    args = dict(ARGS, steady=0, calm=0.0, hold=False)
    a = composite.finish_rerender(fill, master, source, codes, weight, **args)
    b = composite.finish_rerender(fill, master, source, codes, weight, cap=False, **args)
    c = composite.finish_rerender(fill, master, source, codes, weight, cap=True, **args)
    assert np.array_equal(a, b) and not np.array_equal(a, c)

def test_the_mask_takes_a_new_clip_at_its_first_frame_with_first_frame():
    dark = np.full((32, 32, 3), 60, np.uint8)
    lit = dark.copy()
    lit[8:24, 8:24] = 255
    seq = [dark, lit, lit, dark, dark]
    for first, expect in ((False, [False, False, True, False, False]), (True, [False, True, True, False, False])):
        tm = TemporalMask(MaskSettings(tau_high=240, tau_low=-1, dilate=0, first_frame=first))
        got = [bool(tm(c)[0][16, 16]) for c in seq]
        assert got == expect, (first, got)

def test_the_flags_reach_the_fill():
    from radiance.model.rudra_genfill import Rerender
    assert Rerender('x').cap_master is False
    assert Rerender('x', cap_master=True).cap_master is True
    assert MaskSettings().first_frame is False

def _noisy_still(frames=14, seed=11, h=120, w=200):
    """A still shot whose clipped patch sits at codes 249-252 with the codec's +-1 code noise, frame to frame."""
    rng = np.random.default_rng(seed)
    base = np.full((h, w), 60, np.int16)
    base[30:90, 40:160] = 251
    seq = []
    for _ in range(frames):
        n = rng.integers(-1, 2, size=(h, w)).astype(np.int16)
        n = cv2.GaussianBlur(n.astype(np.float32), (0, 0), 1.0)
        c = np.where(base > 200, base + np.round(n * 2), base).clip(0, 255).astype(np.uint8)
        seq.append(np.repeat(c[..., None], 3, -1))
    return seq

def _finish_seq(seq, master_level, **kw):
    state, outs = {}, []
    for codes in seq:
        weight = (codes.max(-1) >= 240).astype(np.float32)
        source = np.power(codes.astype(np.float32) / 255.0, 2.4)
        fill = np.full(codes.shape, 2.0, np.float32)
        master = np.where((codes.max(-1) >= 240)[..., None], np.float32(master_level), np.float32(0.05)).astype(np.float32)
        master = np.broadcast_to(master, codes.shape).astype(np.float32)
        outs.append(composite.finish_rerender(fill, master, source, codes, weight, state=state, **kw))
    return outs

def test_blend_still_settles_the_mix_against_the_codecs_noise():
    seq = _noisy_still()
    base = dict(ARGS, steady=0, hold="thin")
    plain = _finish_seq(seq, 0.8, **base)
    settled = _finish_seq(seq, 0.8, blend_still=0.95, **base)
    codes = seq[-1]
    t_plain, t_set = _p95_still_clipped(plain, codes), _p95_still_clipped(settled, codes)
    assert t_set < 0.5 * t_plain, (t_plain, t_set)

def test_blend_still_follows_a_thin_dark_line_that_drifts():
    """09V's antenna: a 2-px dark line drifting 0.3 px a frame across a clipped sky. Behind it the settled mix must
    not keep it (the faint copies holding on the frame-to-frame change left)."""
    h, w, frames = 60, 160, 40
    xs = np.arange(w, dtype=np.float32)
    seq = []
    for f in range(frames):
        x0 = 40 + 0.3 * f
        cover = np.clip(np.minimum(xs - x0 + 1, x0 + 2 - xs), 0, 1)        # anti-aliased 2-px line
        row = 254 - cover * (254 - 60)
        seq.append(np.repeat(np.repeat(np.round(row)[None, :, None], h, 0), 3, -1).astype(np.uint8))
    base = dict(ARGS, steady=0, hold="thin")
    plain = _finish_seq(seq, 0.8, **base)
    settled = _finish_seq(seq, 0.8, blend_still=0.95, **base)
    x_now = 40 + 0.3 * (frames - 1)
    behind = (xs < x_now - 8) & (xs > 40)                                   # where the line was, 8+ px back
    y_p = np.log2(np.maximum(plain[-1] @ composite.LUMA_AP1, 1e-6))[:, behind]
    y_s = np.log2(np.maximum(settled[-1] @ composite.LUMA_AP1, 1e-6))[:, behind]
    assert np.abs(y_s - y_p).max() < 0.05, float(np.abs(y_s - y_p).max())

def test_blend_still_is_off_by_default_and_needs_state():
    seq = _noisy_still(frames=3)
    base = dict(ARGS, steady=0, hold="thin")
    a = _finish_seq(seq, 0.8, **base)
    b = _finish_seq(seq, 0.8, blend_still=0.0, **base)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))

def test_hold_calm_holds_only_the_held_pixels():
    """A thin strip (held) and a wide patch (not thin): hold_calm changes the strip, never the wide patch."""
    h, w, frames = 1080, 320, 10                       # RERENDER_THIN_PX is in pixels at 1080 lines
    codes = np.full((h, w, 3), 60, np.uint8)
    codes[100:112, 20:300] = 254                       # thin: 12 px
    codes[400:800, 40:280] = 254                       # wide
    weight = (codes.max(-1) >= 240).astype(np.float32)
    source = np.power(codes.astype(np.float32) / 255.0, 2.4)
    master = np.where((codes.max(-1) >= 240)[..., None], np.float32(0.8), np.float32(0.05)).astype(np.float32)
    master = np.broadcast_to(master, codes.shape).astype(np.float32)
    outs = {0.0: [], 0.95: []}
    for hc in outs:
        state = {}
        rng = np.random.default_rng(4)
        for _ in range(frames):
            d = cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), 2.0)
            fill = np.repeat((2.0 * np.exp2(d * np.float32(0.5 / d.std())))[..., None], 3, -1).astype(np.float32)
            outs[hc].append(composite.finish_rerender(fill, master, source, codes, weight, state=state, hold_calm=hc,
                                                      **dict(ARGS, steady=0, hold="thin")))
    strip = np.zeros((h, w), bool); strip[102:110, 30:290] = True
    wide = np.zeros((h, w), bool); wide[500:700, 80:240] = True
    assert _p95_change(outs[0.95][3:], strip) < _p95_change(outs[0.0][3:], strip)
    assert all(np.array_equal(a[wide], b[wide]) for a, b in zip(outs[0.0], outs[0.95]))

def _p95_change(frames, mask):
    y = [np.log2(np.maximum(f @ composite.LUMA_AP1, 1e-6)) for f in frames]
    return float(np.percentile(np.concatenate([np.abs(b - a)[mask] for a, b in zip(y, y[1:])]), 95))

def test_the_new_flags_reach_the_fill():
    from radiance.model.rudra_genfill import Rerender
    r = Rerender('x', blend_still=0.95, hold_calm=0.95)
    assert (r.blend_still, r.hold_calm) == (0.95, 0.95)
    assert Rerender('x').blend_still == 0.0 and Rerender('x').hold_calm == 0.0
