"""Ported from RUDRA v1.3.0 tests/test_release_1_3.py (FXTD-Studios/RUDRA, Apache 2.0): the region re-render
finish of RUDRA 1.2.6 - 1.3.0 in radiance.model.rudra_genfill. The numbers are RUDRA's; only the imports
changed, and tests that need RUDRA's EXR tools or its tiny fill model are left out."""
import cv2  # noqa: F401
import numpy as np
import pytest  # noqa: F401

from radiance.model import rudra_genfill as composite
from radiance.model.rudra_genfill import MaskSettings, TemporalMask, blend_core, plan_windows  # noqa: F401

ARGS = dict(hue="source", colour=0.3, level=3.3, smooth=0.8, white=0.8, tau_high=240, dip=0.3, calm=0.9)

HOLD = dict(ARGS, hold=True)

STEADY = dict(ARGS, steady=7)                   # the app's finish with v5

def _strip(h=1080, w=480):
    """A dark ceiling (code 60) with a 16-px LED strip the source clips at 252-255 and a soft edge, as Porch's
    (frame 80, column 500: 69 73 78 85 105 187 | 255 250 254 254 253 252 ... | 206 111 89 80)."""
    codes = np.full((h, w, 3), 60, np.uint8)
    codes[100:116, 40:440] = 253
    codes[100:116:3, 40:440] = 252
    codes[98, 40:440], codes[99, 40:440], codes[116, 40:440], codes[117, 40:440] = 105, 187, 206, 111
    master = np.full((h, w, 3), 0.03, np.float32)
    master[100:116, 40:440] = 0.8
    weight = (codes.max(-1) >= 240).astype(np.float32)
    source = np.power(codes.astype(np.float32) / 255.0, 2.4)
    return codes, master, source, weight

def _p95_change(frames, mask):
    y = [np.log2(np.maximum(f @ composite.LUMA_AP1, 1e-6)) for f in frames]
    return float(np.percentile(np.concatenate([np.abs(b - a)[mask] for a, b in zip(y, y[1:])]), 95))

def _strip_run(args, frames=12, skip=2):
    codes, master, source, weight = _strip()
    rng = np.random.default_rng(5)
    state, outs, raws = {}, [], []
    for _ in range(frames):
        # v5 draws the strip's detail anew each frame: a few pixels across, about 0.3 stops
        stops = cv2.GaussianBlur(rng.standard_normal(codes.shape[:2]).astype(np.float32), (0, 0), 3.0)
        stops *= np.float32(0.3 / stops.std())
        fill = (np.full(codes.shape, 2.0, np.float32) * np.exp2(stops)[..., None]).astype(np.float32)
        raws.append(fill)
        outs.append(composite.finish_rerender(fill, master, source, codes, weight, state=state, **args))
    strip = np.zeros(codes.shape[:2], bool)
    strip[100:116, 40:440] = True
    return _p95_change(outs[skip:], strip), _p95_change(raws[skip:], strip)

def test_the_hold_steadies_a_still_clipped_strip_beside_the_dark():
    held, raw = _strip_run(HOLD)
    assert raw > 0.3                       # the model's own jitter
    assert held < 0.10, held               # the gate's T (0.031)

def test_without_the_hold_the_finish_is_as_shipped():
    plain, _ = _strip_run(ARGS)
    assert plain > 0.15, plain             # v5 on Porch: 0.229 (this copy 0.222); off by default and in the app

def test_a_strip_that_moves_is_not_held():
    """Where the source changes, nothing is carried: the strip seen one frame on is this frame's fill."""
    codes, master, source, weight = _strip()
    state = {}
    composite.finish_rerender(np.full(codes.shape, 2.0, np.float32), master, source, codes, weight, state=state, **HOLD)
    moved = np.roll(codes, 20, axis=0)
    m2, s2, w2 = np.roll(master, 20, axis=0), np.roll(source, 20, axis=0), np.roll(weight, 20, axis=0)
    out = composite.finish_rerender(np.full(codes.shape, 4.0, np.float32), m2, s2, moved, w2, state=state, **HOLD)
    plain = composite.finish_rerender(np.full(codes.shape, 4.0, np.float32), m2, s2, moved, w2, state={}, **ARGS)
    assert np.allclose(out[120:124], plain[120:124], rtol=1e-5)

def test_the_antenna_against_a_clipped_sky_still_leaves_no_trail():
    """test_release_1_2_8's moving rod, with the sky at code 255 so the new hold could reach it."""
    h, w = 1080, 480
    state = {}
    xs = [60, 120, 180, 240, 300, 360]
    rng = np.random.default_rng(9)
    for x in xs:
        codes = np.full((h, w, 3), 255, np.uint8)
        master = np.full((h, w, 3), 0.8, np.float32)
        for dx, code in ((-2, 238), (-1, 236), (0, 234), (1, 236), (2, 238)):
            codes[:, x + dx] = code
            master[:, x + dx] = 0.4
        t = np.clip((codes.max(-1).astype(np.float32) - 232.0) / 8.0, 0.0, 1.0)
        weight = (t * t * (3.0 - 2.0 * t)).astype(np.float32)
        fill = np.full((h, w, 3), 2.0, np.float32)
        fill[:, x - 3:x + 4] = 0.5                       # the model draws the rod too, darker than the master
        fill *= np.exp2(rng.uniform(-0.2, 0.2, (h, w)))[..., None].astype(np.float32)
        source = np.power(codes.astype(np.float32) / 255.0, 2.4)
        out = composite.finish_rerender(fill, master, source, codes, weight, state=state, **ARGS)
    y = np.log2(np.maximum(out @ composite.LUMA_AP1, 1e-6))
    sky = float(np.median(y[200:880, 420:470]))
    for x in xs[:-2]:
        old = y[200:880, x - 3:x + 4]
        assert float(np.median(old)) > sky - 0.1, (x, float(np.median(old)), sky)

def test_steady_takes_the_shimmer_off_a_still_clipped_strip():
    steady, raw = _strip_run(STEADY, frames=30, skip=7)   # the median from the 7th frame the strip has held
    assert raw > 0.3
    assert steady < 0.08, steady           # the T gate is 0.10 (7 frames 0.068, 5 0.091); v5's own finish 0.238 here, Porch 0.229

def test_steady_off_is_the_finish_as_before():
    codes, master, source, weight = _strip()
    rng = np.random.default_rng(1)
    s0, s1 = {}, {}
    for _ in range(4):
        fill = (np.full(codes.shape, 2.0, np.float32) * np.exp2(rng.uniform(-0.3, 0.3, codes.shape[:2]))[..., None]).astype(np.float32)
        a = composite.finish_rerender(fill, master, source, codes, weight, state=s0, **ARGS)
        b = composite.finish_rerender(fill, master, source, codes, weight, state=s1, **dict(ARGS, steady=1))
        assert np.array_equal(a, b)

def test_steady_median_only_chooses_among_the_models_own_values():
    """Nothing is held of the median's own: a value is out only while the model drew it in most of the last 5
    frames, so a copy drawn once never shows and every copy is gone 2 frames after the model stops."""
    h, w = 8, 8
    codes = np.full((h, w, 3), 255, np.uint8)
    on = np.ones((h, w), bool)
    state = {}
    seq = [0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, -2.0, -2.0, -2.0, -2.0, 0.0, 0.0, 0.0, 0.0]
    got = [float(composite.steady_median(np.full((h, w), v, np.float32), codes, on, state, 5)[0, 0]) for v in seq]
    assert got[:5] == seq[:5]              # until the pixel has held still for a whole window: its own value
    assert got[5] == 0.0                   # drawn once: never shows
    for i in range(4, len(seq)):
        assert (got[i] == -2.0) == (sum(v == -2.0 for v in seq[i - 4:i + 1]) >= 3), i
    assert got[13] == -2.0 and got[14] == 0.0   # the model stopped at 12: gone 2 frames on
    assert set(got) <= {0.0, -2.0}         # only the model's own values

def test_steady_resets_where_the_source_moves():
    h, w = 4, 4
    on = np.ones((h, w), bool)
    state = {}
    still = np.full((h, w, 3), 255, np.uint8)
    for v in (0.0, 0.0, 0.0, 0.0, 0.0):
        composite.steady_median(np.full((h, w), v, np.float32), still, on, state, 5)
    moved = still.copy()
    moved[0, 0] = 200                      # this pixel's source changed: its own value now, no median
    out = composite.steady_median(np.full((h, w), -3.0, np.float32), moved, on, state, 5)
    assert out[0, 0] == -3.0 and out[1, 1] == 0.0

def test_steady_draws_no_trail_behind_a_drifting_antenna():
    """v5b's failure: 09V's antenna drifts and the model draws its own copy a few px behind it. With steady the sky
    behind it is no darker than with v5's own finish, beyond the sky's own noise (the source moved there within the
    last 7 frames, so the median does not act; the 09V ghost gate checks the real shot)."""
    def trail(args):
        h, w = 1080, 480
        state, rng = {}, np.random.default_rng(4)
        for k in range(14):
            x = 100 + 2 * k
            codes = np.full((h, w, 3), 255, np.uint8)
            master = np.full((h, w, 3), 0.8, np.float32)
            for dx, code in ((-2, 238), (-1, 236), (0, 234), (1, 236), (2, 238)):
                codes[:, x + dx] = code
                master[:, x + dx] = 0.4
            t = np.clip((codes.max(-1).astype(np.float32) - 232.0) / 8.0, 0.0, 1.0)
            weight = (t * t * (3.0 - 2.0 * t)).astype(np.float32)
            fill = np.full((h, w, 3), 2.0, np.float32) * np.exp2(rng.uniform(-0.2, 0.2, (h, w)))[..., None].astype(np.float32)
            fill[:, x - 8:x - 5] = 0.3
            source = np.power(codes.astype(np.float32) / 255.0, 2.4)
            out = composite.finish_rerender(fill, master, source, codes, weight, state=state, **args)
        y = np.log2(np.maximum(out @ composite.LUMA_AP1, 1e-6))
        sky = float(np.median(y[200:880, 300:460]))
        return float(np.median(y[200:880, 60:x - 10], axis=0).min() - sky)
    assert trail(STEADY) >= trail(ARGS) - 0.05   # v5 -0.019, steady -0.039 here: the noise of the sky, no copy

THIN = dict(ARGS, hold='thin')

def test_thin_hold_steadies_the_strip():
    held, raw = _strip_run(THIN, frames=30, skip=7)
    assert raw > 0.3
    assert held < 0.06, held               # as the whole hold (v5b); the real Porch replays 0.080 for both

def test_thin_hold_leaves_a_wide_clipped_sky_to_v5s_finish():
    """09V: the antenna drifts across a wide clipped sky. Nothing there is a thin clipped shape, so the finish is
    exactly v5's: the v5b ghosts cannot come back."""
    def run(args):
        h, w = 1080, 480
        state, rng, outs = {}, np.random.default_rng(4), []
        for k in range(10):
            x = 100 + 3 * k
            codes = np.full((h, w, 3), 255, np.uint8)
            master = np.full((h, w, 3), 0.8, np.float32)
            for dx, code in ((-2, 238), (-1, 236), (0, 234), (1, 236), (2, 238)):
                codes[:, x + dx] = code
                master[:, x + dx] = 0.4
            t = np.clip((codes.max(-1).astype(np.float32) - 232.0) / 8.0, 0.0, 1.0)
            weight = (t * t * (3.0 - 2.0 * t)).astype(np.float32)
            fill = np.full((h, w, 3), 2.0, np.float32) * np.exp2(rng.uniform(-0.3, 0.3, (h, w)))[..., None].astype(np.float32)
            fill[:, x - 7:x - 4] = 0.3
            source = np.power(codes.astype(np.float32) / 255.0, 2.4)
            outs.append(composite.finish_rerender(fill, master, source, codes, weight, state=state, **args))
        return outs
    for a, b in zip(run(THIN), run(ARGS)):
        assert np.array_equal(a, b)

def test_the_hold_mode_is_parsed_as_the_app_and_the_v5b_run_give_it():
    import pytest
    from radiance.model.rudra_genfill import hold_mode
    assert hold_mode(True) == 'all' and hold_mode(False) == '' and hold_mode('') == '' and hold_mode(None) == ''
    assert hold_mode('thin') == 'thin'
    with pytest.raises(ValueError):
        hold_mode('everything')
    with pytest.raises(ValueError):
        composite.finish_rerender(np.ones((8, 8, 3), np.float32), np.ones((8, 8, 3), np.float32), np.ones((8, 8, 3), np.float32),
                                  np.full((8, 8, 3), 255, np.uint8), np.ones((8, 8), np.float32), state={'log_y': np.zeros((8, 8), np.float32),
                                  'on': np.ones((8, 8), bool), 'codes': np.full((8, 8, 3), 255, np.uint8), 'trust': np.ones((8, 8), np.float32)},
                                  **dict(ARGS, hold='everything'))
