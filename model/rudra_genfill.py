"""RUDRA AI fill v3: the shape inside fully clipped highlights, drawn by
Wan2.1-VACE-1.3B with the FXTD LoRA; the brightness set by a small estimator.

Port of RUDRA 1.0.1 ``rudra/genfill`` (encoding, masks, composite, brightness,
model, run) working on in-memory frames instead of EXR folders. The steps and
numbers are RUDRA's; what it guarantees is RUDRA's:

* Codes kept: no 8-bit code of the master's ACES 2.0 SDR 100-nit view changes
  (``r_safe_blend``). Pixels the fill does not touch are copied bit for bit.
* Highlights only by default: the fill is blended only where the source is
  fully clipped (code 255), and adds at most ``max_gain_stops`` (3) over the
  master. Shadows (code 0) only when asked for.
* Marked: every generated pixel's blend weight is returned (``rudra.generated``).
* What it draws is plausible, not recovered.

The master is RUDRA's: scene-linear ACEScg, 1.0 = diffuse white, whose ACES
2.0 SDR view is the source (what SDR -> HDR Universal / Recover make with the
RUDRA ACES 2 backend). Needs ``diffusers`` >= 0.35 (WanVACE) at run time;
``peft`` is used for the LoRA when installed, otherwise the LoRA is merged into
the weights directly.

Brought up to RUDRA ``feat/genfill-v4-finish`` 72c4a62 (12 Nov 2026):

* v3, RUDRA 1.1: ``prompt_strength`` (how strongly a region prompt steers its
  region), the flicker ``fallback`` ('spikes': a one-frame flicker is written
  as the master's own frame) and ``focus`` ('region': one box round what the
  shot fills; 'areas': one box per group of clipped areas, each drawn at up to
  the working size, so far-apart highlights in a 4K frame keep their detail).
* v4, region re-render (experimental): :class:`Rerender`. The regions come
  from the SOURCE's codes, the model is conditioned on the route's reading of
  them, and the fill is finished (:func:`finish_rerender`) and blended plainly
  (:func:`plain_blend`): inside the mask the SDR codes are NOT kept, outside it
  every pixel is the master's. Needs the 12 Nov research v4 LoRA (the published v4.3
  needs RUDRA 1.2.4's settings, which this port lacks).
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("radiance.model.rudra_genfill")

AP1_Y = np.array([.2722287168, .6740817658, .0536895174], np.float32)
BASE_PROMPT = ("scene-referred high dynamic range footage, natural light, physically plausible "
               "highlights and shadow detail")
MAX_SEQUENCE_LENGTH = 512
BASE_REPO = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
BASE_DIRNAME = "Wan2.1-VACE-1.3B-diffusers"
BASE_ENV = "RADIANCE_WAN_VACE"
PACKAGE_ENV = "RADIANCE_RUDRA_GENFILL"


# ─────────────────────────────────────────────────────────────────────────────
#  encoding: scene-linear ACEScg <-> the VAE's [-1, 1] (normalised ACEScct)
# ─────────────────────────────────────────────────────────────────────────────

M_DEFAULT = 2.0 ** 14
_A, _B = 10.5402377416545, 0.0729055341958355
_X_BREAK = 0.0078125
_C_BREAK = 0.155251141552511


def _cct_scalar(x: float) -> float:
    return _A * x + _B if x <= _X_BREAK else (math.log2(x) + 9.72) / 17.52


def acescct(x):
    lin = _A * x + _B
    lg = (np.log2(np.clip(x, _X_BREAK, None)) + 9.72) / 17.52
    return np.where(x <= _X_BREAK, lin, lg)


def acescct_inverse(c):
    lin = (c - _B) / _A
    lg = 2.0 ** (np.clip(c, _C_BREAK, None) * 17.52 - 9.72)
    return np.where(c <= _C_BREAK, lin, lg)


def encode(x, M: float = M_DEFAULT):
    lo, hi = _cct_scalar(0.0), _cct_scalar(M)
    v = (acescct(np.clip(x, 0.0, M)) - lo) / (hi - lo)
    return v * 2.0 - 1.0


def decode(y, M: float = M_DEFAULT):
    lo, hi = _cct_scalar(0.0), _cct_scalar(M)
    v = np.clip((y + 1.0) * 0.5, 0.0, 1.0)
    return np.clip(acescct_inverse(v * (hi - lo) + lo), 0.0, M)


# ─────────────────────────────────────────────────────────────────────────────
#  masks
# ─────────────────────────────────────────────────────────────────────────────

MIN_CORE_DEFAULT = 5e-5       # 1.2.6: a share of the frame (about 100 px at 1080p); 0 keeps every clipped pixel


@dataclass
class MaskSettings:
    tau_high: int = 250      # 8-bit code; >= is a highlight to fill (256 = none)
    tau_low: int = -1        # 8-bit code; <= is a shadow to fill (-1 = none)
    dilate: int = 4                       # pixels, at the frame's own resolution
    ema: float = 0.7                      # weight of the previous frame's mask
    feather: int = 0                      # pixels of soft edge for the composite (v1: 6, outward)
    blend_high: int = 255  # the fill is blended only at max code >= this (256 = never)
    blend_low: int = 0    # ... and, with shadows on, at max code <= this
    max_gain_stops: float = 3.0           # v2: the fill's luminance at most this many stops over the master (0 = no cap)
    min_core: float = MIN_CORE_DEFAULT    # 1.2.6: blend-core areas smaller than this share of the frame are left alone
    first_frame: bool = False             # 1.3 (24 Nov 2026), the learned route: a pixel is in from its first clipped frame

    def check(self):
        if not (-1 <= self.tau_low < self.tau_high <= 256):
            raise ValueError(f'need -1 <= tau_low < tau_high <= 256 (got {self.tau_low}, {self.tau_high})')
        if not (0 <= self.blend_high <= 256 and -1 <= self.blend_low <= 255):
            raise ValueError(f'need 0 <= blend_high <= 256 and -1 <= blend_low <= 255 (got {self.blend_high}, {self.blend_low})')
        if not (0.0 <= self.max_gain_stops <= 20.0):
            raise ValueError('max_gain_stops must be in [0, 20]')
        if not (0.0 <= self.min_core < 0.01):
            raise ValueError('min_core must be in [0, 0.01)')
        if not (0.0 <= self.ema < 1.0):
            raise ValueError('ema must be in [0, 1)')
        return self


def _dilate(m: np.ndarray, r: int) -> np.ndarray:
    if r <= 0 or not m.any():
        return m
    import cv2
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(m.astype(np.uint8), k) > 0


def frame_masks(codes: np.ndarray, s: MaskSettings) -> Tuple[np.ndarray, np.ndarray]:
    """codes: (H, W, 3) uint8 SDR codes -> (highlight, shadow) bool masks, dilated."""
    mx = codes.max(-1)
    hi = mx >= s.tau_high if s.tau_high <= 255 else np.zeros(mx.shape, bool)
    lo = mx <= s.tau_low if s.tau_low >= 0 else np.zeros(mx.shape, bool)
    hi, lo = _dilate(hi, s.dilate), _dilate(lo, s.dilate)
    return hi, lo & ~hi


class TemporalMask:
    """Per-pixel EMA of the masks across frames (one instance per shot, frames in order)."""

    def __init__(self, s: MaskSettings):
        self.s = s.check()
        self._hi = self._lo = None

    def __call__(self, codes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        hi, lo = frame_masks(codes, self.s)
        hi, lo = hi.astype(np.float32), lo.astype(np.float32)
        now_hi, now_lo = hi > 0.5, lo > 0.5
        if self._hi is not None and self._hi.shape == hi.shape and self.s.ema > 0:
            a = self.s.ema
            hi = a * self._hi + (1 - a) * hi
            lo = a * self._lo + (1 - a) * lo
        self._hi, self._lo = hi, lo
        h, l = hi >= 0.5, lo >= 0.5
        if self.s.first_frame:
            # 1.3 (24 Nov 2026), the learned route: with the EMA alone a newly clipped pixel joins at its second
            # clipped frame, so for one frame it shows the master. The learned master sits 1-2 stops over the fill
            # there (03V: frame-to-frame jumps of 0.32 EV on still clipped pixels, the T gate's limit 0.10). It now
            # joins at once; when it leaves is still the EMA's (the analytic route keeps the EMA alone).
            h, l = h | now_hi, l | now_lo
        return h, l & ~h


def blend_core(codes: np.ndarray, s: MaskSettings) -> np.ndarray:
    """codes (H, W, 3) uint8 -> bool: where the fill may be blended (no dilation, no EMA)."""
    mx = codes.max(-1)
    core = mx >= s.blend_high if s.blend_high <= 255 else np.zeros(mx.shape, bool)
    if s.tau_low >= 0 and s.blend_low >= 0:
        core = core | (mx <= s.blend_low)
    return drop_specks(core, min_core_pixels(core.shape, s.min_core))


def min_core_pixels(shape, share: float) -> int:
    """The smallest blend-core area kept, in pixels, for a frame of `shape` (0 or 1: keep all)."""
    return int(share * shape[0] * shape[1]) if share > 0 else 0


def drop_specks(core: np.ndarray, min_pixels: int) -> np.ndarray:
    """bool (H, W) -> the same without its 8-connected areas smaller than min_pixels."""
    if min_pixels <= 1 or not core.any():
        return core
    import cv2
    n, labels, stats, _ = cv2.connectedComponentsWithStats(core.astype(np.uint8), connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_pixels
    keep[0] = False
    return keep[labels]


def feather(mask: np.ndarray, radius: int) -> np.ndarray:
    m = mask.astype(np.float32)
    if radius <= 0 or not mask.any():
        return m
    import cv2
    grown = _dilate(mask, radius).astype(np.float32)
    soft = cv2.GaussianBlur(grown, (0, 0), radius / 2.0)
    return np.maximum(m, np.clip(soft, 0, 1)) * grown


# ─────────────────────────────────────────────────────────────────────────────
#  composite: the round-trip-safe blend
# ─────────────────────────────────────────────────────────────────────────────

def _codes(view, rgb: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(view.forward(np.ascontiguousarray(rgb, np.float32)), 0.0, 1.0) * 255.0).astype(np.int16)


def r_safe_blend(master: np.ndarray, fill: np.ndarray, weight: np.ndarray, view,
                 iterations: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    """out = master + alpha * (fill - master) with alpha <= weight chosen so the
    master's SDR view keeps every 8-bit code (bisection; alpha 0 always does)."""
    if master.shape != fill.shape or master.shape[:2] != weight.shape:
        raise ValueError("master, fill and weight must agree in size")
    out = np.array(master, np.float32, copy=True)
    alpha = np.zeros(weight.shape, np.float32)
    idx = np.nonzero(weight > 0)
    if not len(idx[0]):
        return out, alpha
    m = master[idx].astype(np.float32)
    d = fill[idx].astype(np.float32) - m
    w = np.clip(weight[idx].astype(np.float32), 0.0, 1.0)
    ref = _codes(view, m)
    good = (_codes(view, m + w[:, None] * d) == ref).all(-1)
    best = np.where(good, w, 0.0).astype(np.float32)
    bad = np.nonzero(~good)[0]
    if len(bad):
        lo = np.zeros(len(bad), np.float32)
        hi = w[bad].copy()
        mb, db, rb = m[bad], d[bad], ref[bad]
        for _ in range(iterations):
            mid = (lo + hi) * 0.5
            g = (_codes(view, mb + mid[:, None] * db) == rb).all(-1)
            lo = np.where(g, mid, lo)
            hi = np.where(g, hi, mid)
        best[bad] = lo
    alpha[idx] = best
    out[idx] = m + best[:, None] * d
    return out, alpha


# ─────────────────────────────────────────────────────────────────────────────
#  composite (v4, region re-render): the plain blend and the render-time finish
#  RUDRA rudra/genfill/composite.py at 72c4a62, unchanged but for names.
# ─────────────────────────────────────────────────────────────────────────────

def plain_blend(master: np.ndarray, fill: np.ndarray, weight: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """out = master + weight * (fill - master): the region re-render (v4, 9 Nov 2026). Inside the
    mask the SDR codes are NOT kept; outside it (weight 0) every pixel is the master's, bit for bit."""
    if master.shape != fill.shape or master.shape[:2] != weight.shape:
        raise ValueError('master, fill and weight must agree in size')
    alpha = np.clip(weight.astype(np.float32), 0.0, 1.0)
    out = np.array(master, np.float32, copy=True)
    idx = np.nonzero(alpha > 0)
    if len(idx[0]):
        out[idx] = master[idx] + alpha[idx][:, None] * (fill[idx].astype(np.float32) - master[idx])
    return out, alpha


LUMA_AP1 = AP1_Y
RERENDER_HUES = ('model', 'source')
RERENDER_FULL_CODE = 253        # the source's max code at which the fill sits at the whole level
RERENDER_HUE_SIGMA = 8.0        # pixels: the source's colour is taken from its neighbourhood, not per pixel
RERENDER_DETAIL_SIGMA = 64.0    # pixels at 1080 lines: above this scale the level is the source's (or the model's, within the ceiling)
# How far over `level` the model's own level may sit. 0 since 11 Nov 2026: the model's level is not steady
# (Porch: the region's median fell from 5.5 to 3.3 x white over the first 8 frames with 2 stops allowed, and
# held 2.8-3.2 with none), and with the crossfade by code the allowance bought nothing on the held-out clips
# (PU21 at -3 EV -28.1 with it, -27.6 without) or on 09V (no pixel moved).
RERENDER_CEILING_STOPS = 0.0
RERENDER_FRINGE_PX = 12.0       # pixels at 1080 lines: this close to a darker neighbour the fill is the interior's
RERENDER_REACH_PX = 6.0         # pixels at 1080 lines: beside something dark a code takes its brightest neighbour's within this
RERENDER_DARK_CODES = 40        # "something dark": a neighbour this many codes under tau_high (fully so at 80)


RERENDER_RISE_PX = 150.0        # pixels at 1080 lines: with `rise`, the depth into the full-code core over which the lift approaches it


RERENDER_CALM_STOPS = (0.1, 0.3) # stops: with `calm`, a still pixel's fill that would move this much starts to be held; fully from the second
RERENDER_STEADY_MAX = 15       # 1.3: the longest median `steady` takes
RERENDER_RELEASE_PX = 7        # 1.3 research: hold 'release' lets go of a carried dark line narrower than this
RERENDER_RELEASE_STOPS = 0.15  # ... that its closing lifts by more than this
RERENDER_THIN_PX = 20.0        # 1.3: hold 'thin' holds a clipped shape at most this far (px at 1080) from its edge at its widest
RERENDER_THIN_REACH_PX = 40    # ... its widest point looked for this far around (px at 1080)
RERENDER_HOLD_CODE = 250       # 1.3: with `calm`, a still pixel the source clips at this code is held whole (the T gate's clipped)
RERENDER_STILL_TOP = 3.0       # 1.3: with `blend_still`, the crossfade's code (blurred) within this many codes of where its hold began
RERENDER_NEUTRAL = (0.78, 0.88) # the source's min over max code: below the first the region is left to the master, whole from the second


def neutral_weight(codes: np.ndarray) -> np.ndarray:
    """(H, W) in [0, 1]: how near white the source is, from its codes (H, W, 3). The region re-render
    is for an area the source parks near white in every channel: nothing is known there. A saturated
    highlight (a sunset: red at 255, blue at 120) still carries its colour and its gradient, and a fill
    at a level of its own draws a contour across it (11 Nov 2026, the 07V still: the mode whitened the
    sky above a hard edge). There the weight goes to 0 and the master stays."""
    c = codes.astype(np.float32)
    lo, hi = RERENDER_NEUTRAL
    x = np.clip((c.min(-1) / np.maximum(c.max(-1), 1.0) - lo) / (hi - lo), 0.0, 1.0)
    return (x * x * (3.0 - 2.0 * x)).astype(np.float32)


RERENDER_DETAIL_WINDOW = 25        # px at 1080 lines: the box over which a region pixel's neighbours are counted
RERENDER_DETAIL_COVER = (0.45, 0.75)   # the share of that box in the region: below the first the master stays, whole from the second
RERENDER_DETAIL_MIN_AREA = 5e-4    # a region area under this share of the frame keeps the master (a lamp, a glint)


def detail_keep_weight(codes: np.ndarray, tau_high: int) -> np.ndarray:
    """(H, W) in [0, 1]: 1.2.7 (multi-role review of 1.2.6), --rerender-keep-detail. The region re-render
    redraws every pixel at code tau_high or above. Real objects that sit there keep their picture with this
    on: a thin or broken-up part of the region (a floor grating's bright bars, a railing) by how little of
    the box around it is region, and an area under RERENDER_DETAIL_MIN_AREA of the frame (a practical lamp)
    entirely. core/src/fill_region.cpp ai_fill_v4_weight draws the same rule over the viewer."""
    import cv2
    h, w = codes.shape[:2]
    region = (codes.max(-1) >= tau_high).astype(np.uint8)
    k = max(3, int(round(RERENDER_DETAIL_WINDOW * h / 1080.0)) | 1)
    cover = cv2.blur(region.astype(np.float32), (k, k), borderType=cv2.BORDER_REPLICATE)
    lo, hi = RERENDER_DETAIL_COVER
    x = np.clip((cover - lo) / (hi - lo), 0.0, 1.0)
    weight = x * x * (3.0 - 2.0 * x) * region
    n, labels, stats, _ = cv2.connectedComponentsWithStats(region, connectivity=8)
    small = stats[:, cv2.CC_STAT_AREA] < RERENDER_DETAIL_MIN_AREA * h * w
    small[0] = True                                   # the background label
    weight[small[labels]] = 0.0
    return weight.astype(np.float32)


def white_scale(master: np.ndarray, codes: np.ndarray) -> Optional[float]:
    """The master's SDR white on one frame: the median, over mid-tone pixels (every channel between
    codes 50 and 180), of the master's luminance over the source's display light (code ^ 2.4), as
    tools/compare_hdr_renders.py reads its 'scale'. None when the frame has too few mid-tones."""
    mid = np.all((codes > 50) & (codes < 180), -1)
    if int(mid.sum()) < 500:
        return None
    light = np.power(codes[mid].astype(np.float32) / 255.0, 2.4) @ LUMA_AP1
    return float(np.median((master[mid] @ LUMA_AP1) / np.maximum(light, 1e-6)))


def finish_rerender(fill: np.ndarray, master: np.ndarray, source_lin: np.ndarray, codes: np.ndarray, weight: np.ndarray,
                    hue: str = 'model', colour: float = 0.0, level: float = 0.0, smooth: float = 0.0,
                    white: float = 1.0, tau_high: int = 240, state: dict | None = None, dip: float = 0.0,
                    still: float = 0.0, rise: float = 0.0, calm: float = 0.0, hold: bool | str = False,
                    steady: int = 0, cap: bool = False, blend_still: float = 0.0, hold_calm: float = 0.0) -> np.ndarray:
    """The region re-render's fill, finished before the blend (10 Nov 2026, after the v4 gates on 09V:
    the v4 LoRA drew the structure, but too blue, at 1.2 x white where 2.5-4 was asked, its level
    drifting 0.85-1.72 x white along the shot with jumps at the window seams, and brighter where the
    source is darker).

    fill, master, source_lin (the source's display light, code ^ 2.4) (H, W, 3) ACEScg; codes (H, W, 3)
    8-bit; weight (H, W) the blend weight. Every step is off at its default, and then the fill is
    returned as is:
      hue 'source'  the fill keeps its luminance and takes the source's colour (the source's display
                    light blurred RERENDER_HUE_SIGMA px, so near-clip noise does not tint it), mixed
                    with `colour` (0..1) of the model's own
      level > 0     the detail from the model, the level `level` x `white` (the master's SDR white,
                    white_scale): the fill keeps what is finer than RERENDER_DETAIL_SIGMA; above that
                    scale its own level is kept only where brighter and only RERENDER_CEILING_STOPS
                    over (0: not at all; see the constant).
                    That levelled fill is then crossfaded into the MASTER by the source's code, log
                    luminance and colour alike: the master itself at tau_high, the whole fill at
                    RERENDER_FULL_CODE. So the lift starts at nothing where the region starts and a
                    brighter source stays brighter (11 Nov, the Porch shot: a softbox's hot spots,
                    a smooth gradient through tau_high, came out as ovals with an edge when the
                    model's level applied whole from the region's rim).
                    Two edge rules go with it: within RERENDER_FRINGE_PX of a darker neighbour the
                    fill is taken from the interior (the model's fill is soft and dark beside a
                    silhouette), and beside something dark a code takes its brightest neighbour's
                    within RERENDER_REACH_PX (an anti-aliased edge is a mix of figure and sky, not a
                    darker sky; in a gradient there is nothing dark and the codes stand)
      dip > 0       with a level: the model's fine detail goes at most `dip` stops under its surroundings
                    (a soft floor from half of it). 12 Nov 2026, 09V in the app: dark smudges 0.6 to 2.4
                    stops deep beside the figures, a ghost of the silhouette; clouds are 0.05 to 0.12 stops,
                    and nothing in a region parked at white is that much darker than what is round it.
                    Brighter detail is not touched. Off (0) is the fill as gated on 11 Nov
      smooth > 0    the fill's log luminance follows the previous frame's (an EMA, this share kept)
                    wherever both frames are filled; `state` (a dict the caller keeps for the shot)
                    carries it. A newly filled pixel starts from its own value.
      still > 0     where the source holds still (every channel within 2 codes of the previous frame)
                    this share is kept instead of `smooth`: nothing in the source says the light there
                    changed, so the fill should not either. 12 Nov 2026, water.mp4: the static clipped
                    pixels moved 0.19 EV (p95) with smooth 0.8 alone, 0.06 with still 0.9; the
                    analytic master's own were 0.5 to 2.3 EV. Where the source moves, `smooth` stands.
                    It also averages the model's detail away (09V's structure halved): off in the app.
      calm > 0      where the source holds still (as `still`) AND the fill would move by more than
                    RERENDER_CALM_STOPS[0] from the previous frame's, this share is kept, fully so from
                    RERENDER_CALM_STOPS[1]; under it `smooth` stands. 15 Nov 2026: `still` holds every
                    still pixel and so averages a steady picture away over a shot (09V); the flicker
                    it was made for is the model's detail jumping on still pixels inside moving
                    highlights (water: the raw fill moves 1.5 EV p95 there, the door's 0.08). Only a
                    jump is held. Measured on the v4 raw fills with this finish: water frames 30-35,
                    static clipped p95 0.115 -> 0.063 EV, structure 0.112 -> 0.131; 09V frames 40-47,
                    structure 0.0957 both ways, 98.4 % of the region within 0.01 stops of the fill
                    without it. Off (0) is the fill as gated.
      rise > 0      with a level: where the source sits at RERENDER_FULL_CODE the level keeps climbing,
                    up to `rise` stops over `level`, with the depth into that full-code core
                    (rise * (1 - exp(-depth / RERENDER_RISE_PX)), the depth blurred so the core's
                    medial axis leaves no crease). 13 Nov 2026, testwindow.mp4 (sky glare): with the
                    model's level held at the floor (RERENDER_CEILING_STOPS 0) the core was a flat
                    plate at exactly 3.30 x white from 250 px in, a mask-shaped disc at -3 EV where
                    glare keeps rising towards its source. In a clipped core the only evidence left
                    is how far the clip reaches; a wider clip is taken as a brighter one. Fine detail
                    is untouched (it adds no structure, so it does not move the S gate). Off (0) is
                    the fill as gated on 11 Nov.
      steady > 1    1.3 (23 Nov 2026, AI fill v5): where the source clips (RERENDER_HOLD_CODE) and holds
                    still (2 codes), the fill's own log luminance is the median of its last `steady`
                    frames (as many as the pixel has held still), before `smooth` / `calm`. v5 redraws a
                    thin LED strip a little differently every frame (Porch: static clipped p95 0.229 EV,
                    limit 0.10) beside the dark, where the 1.2.8 trust lets `calm` hold nothing; holding
                    regardless (v5b) carried 09V's drifting antenna's ghosts for frames. A median keeps
                    nothing of its own: it only ever chooses among the model's own recent values (where the
                    pixel has held still for all of them), so a copy the model drew in fewer than half of
                    them is dropped and none can outlast the
                    model drawing it by more than steady // 2 frames. Kept in state: the last frames'
                    values (the ring), never an output.
                    0 or 1: off.
      cap           with a level (1.3, 24 Nov 2026, the learned route): in the crossfade the master's share is
                    never brighter than the fill's own value there. The learned ACES 2.0 master rebuilds a clipped
                    area itself, brighter than the fill's level and unsteady near the clip (Porch: 7.8 x white, its
                    still clipped pixels moving 0.75 EV p95; water 24 x white, 0.67 EV), so its share of a region
                    pixel came through as a rim brighter than the core and as flicker (the T gate on 01V, 02V,
                    water; L 4.07 on Porch). The analytic master sits near white there, under the fill's level,
                    where the cap changes nothing; the app sets it on the learned route only. Off: as gated.
      blend_still   with a level and `state` (1.3, 24 Nov 2026): while the code the crossfade follows stays within
                    RERENDER_STILL_TOP of where it was when the hold began, the crossfade's share of the fill (s)
                    keeps this share of the previous frame's. s follows the source's code, and on a still shot a
                    code near the clip moves 1-2 codes from frame to frame (the codec's noise): at code 251 that
                    moves s by up to 0.12, and with the analytic master about 1.7 stops under the fill's level the
                    pixel moved up to 0.2 stops (01V on the analytic route: T 0.164, 98 % of the worst changes
                    within 4 px of a clipped edge; with 0.95, 0.079). 0: as gated.
      hold_calm     with `hold` (1.3, 24 Nov 2026): the share a held pixel keeps for a jump, instead of `calm`.
                    v5 redraws the inside of a thin clipped shape a little differently every frame (03V: T 0.17
                    with the mix settled, 0.094 with 0.95 here); `calm` itself stays as gated, so a wide clipped
                    sky's clouds are finished as before. 0: `calm`."""
    if hue not in RERENDER_HUES:
        raise ValueError(f'hue must be one of {", ".join(RERENDER_HUES)}')
    if (not 0.0 <= smooth < 1.0 or not 0.0 <= colour <= 1.0 or level < 0 or dip < 0 or rise < 0 or not 0.0 <= still < 1.0
            or not 0.0 <= calm < 1.0 or not 0 <= int(steady) <= RERENDER_STEADY_MAX or not 0.0 <= blend_still < 1.0
            or not 0.0 <= hold_calm < 1.0):
        raise ValueError('smooth, still and calm must be in [0, 1), colour in [0, 1], level, dip and rise >= 0, '
                         f'steady 0..{RERENDER_STEADY_MAX}')
    steady = int(steady)
    if hue == 'model' and level <= 0 and smooth <= 0 and still <= 0 and calm <= 0 and steady <= 1:
        return fill
    import cv2
    on = weight > 0
    if not on.any():
        return fill
    y = np.maximum(fill @ LUMA_AP1, 1e-6).astype(np.float32)
    chroma = fill / y[..., None]
    log_y = np.log2(y)
    px = fill.shape[0] / 1080.0
    top_code = codes.max(-1).astype(np.float32)
    if level > 0:                                       # the interior's fill beside a darker neighbour
        r = max(1.0, RERENDER_FRINGE_PX * px)
        d = np.clip(cv2.distanceTransform((top_code >= tau_high - 8).astype(np.uint8), cv2.DIST_L2, 3) / r, 0.0, 1.0)
        conf = (d * d * (3.0 - 2.0 * d) * on).astype(np.float32)
        near_conf = cv2.GaussianBlur(conf, (0, 0), r)
        known = near_conf > 1e-3
        inner_y = cv2.GaussianBlur(log_y * conf, (0, 0), r) / np.maximum(near_conf, 1e-3)
        inner_c = cv2.GaussianBlur(chroma * conf[..., None], (0, 0), r) / np.maximum(near_conf, 1e-3)[..., None]
        log_y = np.where(known, conf * log_y + (1.0 - conf) * inner_y, log_y).astype(np.float32)
        chroma = np.where(known[..., None], conf[..., None] * chroma + (1.0 - conf[..., None]) * inner_c, chroma).astype(np.float32)
    if hue == 'source':
        near = cv2.GaussianBlur(np.ascontiguousarray(source_lin, np.float32), (0, 0), RERENDER_HUE_SIGMA)
        chroma = (1.0 - colour) * near / np.maximum(near @ LUMA_AP1, 1e-6)[..., None] + colour * chroma
    if level > 0:
        sigma = RERENDER_DETAIL_SIGMA * px
        inside = on.astype(np.float32)                  # the model's own neighbourhood: filled pixels only
        low = cv2.GaussianBlur(log_y * inside, (0, 0), sigma) / np.maximum(cv2.GaussianBlur(inside, (0, 0), sigma), 1e-3)
        k = int(round(RERENDER_REACH_PX * px))
        if k > 0:
            el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
            e = np.clip((tau_high - RERENDER_DARK_CODES - cv2.erode(top_code, el)) / float(RERENDER_DARK_CODES), 0.0, 1.0)
            top_code = top_code + e * e * (3.0 - 2.0 * e) * (cv2.dilate(top_code, el) - top_code)
        top = cv2.GaussianBlur(top_code, (0, 0), 2.0)
        s = np.clip((top - tau_high) / max(1.0, float(RERENDER_FULL_CODE - tau_high)), 0.0, 1.0)
        s = s * s * (3.0 - 2.0 * s)
        if blend_still > 0 and state is not None:       # 1.3: the mix settles while its code holds still
            # Held while `top`, the code s is made from (blurred, and lifted beside something dark), stays within
            # RERENDER_STILL_TOP of where the hold began: the codec's noise moves it about there and back. A
            # thin dark object drifting under a pixel a frame (09V's antenna) moves it as little from one frame
            # to the next, but on and on, so it soon leaves the band and the mix follows again. Holding on the
            # frame-to-frame change instead (the pixel's own codes, or `top`) left faint copies of the antenna
            # beside it (24 Nov 2026 replays, tmp/v5_review/09V_*_antenna3x.jpg).
            ps, ref, pon = state.get('s'), state.get('top_ref'), state.get('on')
            if ps is not None and ps.shape == s.shape and ref is not None and ref.shape == top.shape and pon is not None:
                held = (np.abs(top - ref) <= np.float32(RERENDER_STILL_TOP)) & on & pon
                s = np.where(held, np.float32(blend_still) * ps + np.float32(1.0 - blend_still) * s, s)
                ref = np.where(held, ref, top)
            else:
                ref = top
            s = s.astype(np.float32)
            state['s'], state['top_ref'] = s, ref.astype(np.float32)
        floor = np.float32(np.log2(level * white))
        detail = log_y - low
        if dip > 0:
            half = np.float32(0.5 * dip)
            detail = np.where(detail < -half, -half - half * np.tanh((-detail - half) / half), detail).astype(np.float32)
        whole = np.clip(low, floor, floor + RERENDER_CEILING_STOPS) + detail
        if rise > 0:                                    # the full-code core keeps climbing with its depth
            L = RERENDER_RISE_PX * px
            core = ((s >= 0.999) & on).astype(np.uint8)
            if core.any():
                depth = cv2.GaussianBlur(cv2.distanceTransform(core, cv2.DIST_L2, 5).astype(np.float32), (0, 0), 0.5 * L)
                whole = whole + np.float32(rise) * (1.0 - np.exp(-depth / np.float32(L))) * s
        y_m = np.maximum(master @ LUMA_AP1, 1e-6).astype(np.float32)
        log_m = np.log2(y_m).astype(np.float32)
        whole = whole.astype(np.float32)
        log_y = ((1.0 - s) * (np.minimum(log_m, whole) if cap else log_m) + s * whole).astype(np.float32)
        chroma = ((1.0 - s[..., None]) * (master / y_m[..., None]) + s[..., None] * chroma).astype(np.float32)
    if (smooth > 0 or still > 0 or calm > 0 or steady > 1) and state is not None:
        # 1.2.8 (QC of v4.3 on 09V, 21 Nov 2026): what is carried from frame to frame is the fill's own
        # log luminance (`whole` with a level), never the master's share of a region-edge pixel. The
        # carried value had (1 - s) of the master in it, so a dark thin object moving along the edge (09V's
        # antenna, 2-3 px against the sky) left its old positions behind at 0.8, 0.64, ... of its depth: a
        # fan of faint antennas. The master's share is now this frame's, mixed in after the smoothing.
        own = whole if level > 0 else log_y
        # ... and only as much as the previous frame's pixel was the fill's own sky: where the region's edge
        # mixed in the master (s < 1) or the pixel sat within RERENDER_FRINGE_PX of something dark (conf < 1),
        # the model's value there also drew the antenna, so it is carried that much less (the GPU check of
        # 21 Nov 2026 left a faint fan with the first fix alone).
        trust = (s * conf).astype(np.float32) if level > 0 else on.astype(np.float32)
        if steady > 1:                                  # 1.3: the median of the model's own recent values
            own = steady_median(own, codes, on, state, steady)
        prev = state.get('log_y')
        if prev is not None and prev.shape == own.shape:
            both = on & state['on']
            keep = np.full(log_y.shape, np.float32(smooth))
            pc = state.get('codes')
            if (still > 0 or calm > 0) and pc is not None and pc.shape == codes.shape:
                held = np.abs(codes.astype(np.int16) - pc.astype(np.int16)).max(-1) <= 2
                if still > 0:
                    keep[held] = np.float32(still)
                if calm > 0:                            # only a jump is held; a steady picture keeps `smooth`
                    lo, hi = RERENDER_CALM_STOPS
                    t = np.clip((np.abs(own - prev) - lo) / (hi - lo), 0.0, 1.0)
                    t = t * t * (3.0 - 2.0 * t)
                    jump = np.float32(smooth) + (np.float32(calm) - np.float32(smooth)) * t
                    keep = np.where(held, np.maximum(keep, jump), keep).astype(np.float32)
            pt = state.get('trust')
            if pt is not None and pt.shape == keep.shape:
                scaled = (keep * pt).astype(np.float32)
                if hold and calm > 0 and pc is not None and pc.shape == codes.shape:
                    # 1.3 research (23 Nov 2026, v5b; `hold`, off by default and in the app): a pixel the source
                    # clips and holds still is held whole, trust or not. v5 redraws Porch's thin LED strip a little
                    # differently every frame (static clipped p95 0.229 EV, v4.3 0.058) and the strip lies within
                    # RERENDER_FRINGE_PX of the dark ceiling, where the trust above holds nothing; with the hold
                    # 0.080. But the sky beside 09V's drifting antenna is clipped and still too: there the hold
                    # keeps the model's own antennas for frames (v5b: 150-300 px a frame up to 0.87 stops darker
                    # than v5, 4-23 px from it). Not usable as it is.
                    clipped = held & (codes.max(-1) >= RERENDER_HOLD_CODE) & (pc.max(-1) >= RERENDER_HOLD_CODE)
                    mode = hold if isinstance(hold, str) else 'all'
                    if mode == 'dips':
                        # only a dip is held: a pixel getting brighter is never held back, so a dark copy the
                        # model drew cannot stay once it moves on (the v5b ghosts were dark)
                        clipped &= own < prev
                    elif mode == 'release':
                        # held, except where what would be carried is a thin dark line in the last frame (a copy
                        # of something thin): its grey closing over RERENDER_RELEASE_PX lifts it more than
                        # RERENDER_RELEASE_STOPS
                        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (RERENDER_RELEASE_PX, RERENDER_RELEASE_PX))
                        clipped &= (cv2.morphologyEx(prev, cv2.MORPH_CLOSE, k) - prev) <= RERENDER_RELEASE_STOPS
                    elif mode == 'thin':
                        # held only inside a thin clipped shape (a strip light, a tube, a small lamp): where the
                        # widest point of the clipped area within RERENDER_THIN_REACH_PX is at most RERENDER_THIN_PX
                        # from its edge. A wide clipped sky (09V's door, beside the drifting antenna) keeps v5's own
                        # finish, whose trust already holds its interior.
                        top_c = (codes.max(-1) >= RERENDER_HOLD_CODE).astype(np.uint8)
                        dt = cv2.distanceTransform(top_c, cv2.DIST_L2, 3)
                        reach = max(3, int(round(RERENDER_THIN_REACH_PX * px)) | 1)
                        widest = cv2.dilate(dt, cv2.getStructuringElement(cv2.MORPH_RECT, (2 * reach + 1, 2 * reach + 1)))
                        clipped &= widest <= np.float32(RERENDER_THIN_PX * px)
                    elif mode != 'all':
                        raise ValueError(f'hold must be off, all, dips, release or thin, not {hold!r}')
                    if hold_calm > 0:                   # 1.3: the held pixels' own share for a jump
                        jump = np.float32(smooth) + (np.float32(hold_calm) - np.float32(smooth)) * t
                    scaled = np.where(clipped, np.maximum(scaled, jump), scaled).astype(np.float32)
                keep = scaled
            own = own.copy()
            own[both] = keep[both] * prev[both] + (1.0 - keep[both]) * own[both]
            log_y = ((1.0 - s) * (np.minimum(log_m, own) if cap else log_m) + s * own).astype(np.float32) if level > 0 else own
        state['log_y'], state['on'], state['codes'], state['trust'] = own, on, codes.copy(), trust
    out = np.array(fill, np.float32, copy=True)
    out[on] = (np.exp2(log_y)[..., None] * chroma)[on]
    return out


def steady_median(own: np.ndarray, codes: np.ndarray, on: np.ndarray, state: dict, n: int) -> np.ndarray:
    """finish_rerender's `steady` (1.3): own (H, W) the fill's own log luminance this frame -> the same, where the
    source clips (RERENDER_HOLD_CODE) and has held still (2 codes) for all of the last n frames, the median of the
    n values the fill had there. state['steady_ring'] keeps the last n - 1 frames' own values as they came in
    (never an output), state['steady_run'] how many frames each pixel has held, state['steady_codes'] last codes."""
    top = codes.max(-1)
    pc = state.get('steady_codes')
    run = state.get('steady_run')
    if pc is not None and pc.shape == codes.shape:
        held = ((np.abs(codes.astype(np.int16) - pc.astype(np.int16)).max(-1) <= 2) & (top >= RERENDER_HOLD_CODE)
                & (pc.max(-1) >= RERENDER_HOLD_CODE))
    else:
        held = np.zeros(top.shape, bool)
    run = np.where(held, (run if run is not None and run.shape == top.shape else np.zeros(top.shape, np.int16)) + 1,
                   0).astype(np.int16)
    ring = [r for r in state.get('steady_ring', []) if r.shape == own.shape]
    state['steady_ring'] = (ring + [own.astype(np.float32, copy=True)])[-(n - 1):] if n > 1 else []
    state['steady_run'], state['steady_codes'] = run, codes.copy()
    if not ring:
        return own
    if len(ring) < n - 1:
        return own
    # Only where the source has held still for the whole window (a partial 2- or 3-frame median keeps a copy
    # drawn just after something passed). A value is in the output only while the model drew it in most of
    # the last n frames: a copy drawn in fewer than n // 2 + 1 of them never shows, and every copy is gone
    # n // 2 frames after the model stops drawing it.
    sel = (run >= n - 1) & on
    out = own.copy()
    if sel.any():
        out[sel] = np.median(np.stack([f[sel] for f in ring + [own]]), axis=0)
    return out.astype(own.dtype)


# ─────────────────────────────────────────────────────────────────────────────
#  brightness (v3): how bright a clipped area was
# ─────────────────────────────────────────────────────────────────────────────

FEATURES = ["log2_area", "level", "min_chan", "all_three", "ring1_rel", "ring1_near", "ring2_rel", "slope",
            "ring_std", "compact", "frame_rel", "frame_clipped", "centroid_y", "ring_sat"]
MIN_PIXELS = 16


def log2_lum(rgb: np.ndarray) -> np.ndarray:
    return np.log2(np.maximum(rgb @ AP1_Y, 1e-6)).astype(np.float32)


def components(codes: np.ndarray, min_pixels: int = MIN_PIXELS):
    import cv2
    core = (codes.max(-1) >= 255).astype(np.uint8)
    n, lab = cv2.connectedComponents(core, connectivity=8)
    ids = [c for c in range(1, n) if (lab == c).sum() >= min_pixels]
    return lab.astype(np.int32), ids


def _ring(mask: np.ndarray, r0: int, r1: int) -> np.ndarray:
    import cv2
    k = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))  # noqa: E731
    outer = cv2.dilate(mask.astype(np.uint8), k(r1)) > 0
    inner = cv2.dilate(mask.astype(np.uint8), k(r0)) > 0 if r0 > 0 else mask
    return outer & ~inner


def frame_features(codes: np.ndarray, master: np.ndarray):
    ly = log2_lum(master)
    mx = codes.max(-1)
    return {"ly": ly, "median": float(np.median(ly)), "clipped": float((mx >= 255).mean()), "mx": mx}


def component_features(codes: np.ndarray, master: np.ndarray, mask: np.ndarray, frame=None) -> np.ndarray:
    import cv2
    fr = frame or frame_features(codes, master)
    ly, mx = fr["ly"], fr["mx"]
    h, w = mask.shape
    area = float(mask.sum())
    level = float(ly[mask].mean())
    cm = codes[mask].astype(np.float32)
    min_chan = float(cm.min(-1).mean() / 255.0)
    all_three = float((cm.min(-1) >= 255).mean())
    r1, r2 = _ring(mask, 0, 3), _ring(mask, 4, 10)
    r1 &= mx < 255
    r2 &= mx < 255
    l1 = float(ly[r1].mean()) if r1.any() else level
    l2 = float(ly[r2].mean()) if r2.any() else l1
    near = float((mx[r1] >= 250).mean()) if r1.any() else 1.0
    std = float(ly[r1 | r2].std()) if (r1 | r2).any() else 0.0
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    per = float(sum(cv2.arcLength(c, True) for c in cnts)) or 1.0
    ys = np.nonzero(mask)[0]
    rr = master[r1] if r1.any() else master[mask]
    sat = float(((rr.max(-1) - rr.min(-1)) / np.maximum(rr.max(-1), 1e-6)).mean())
    return np.array([np.log2(area / (h * w)), level, min_chan, all_three, l1 - level, near, l2 - level,
                     (l1 - l2) / 5.5, std, min(1.0, 4 * np.pi * area / per ** 2), fr["median"] - level,
                     fr["clipped"], float(ys.mean() / h), sat], np.float32)


class BrightnessEstimator:
    """features -> b (stops over the master), >= 0. Inference only (14 -> 32 -> 32 -> 1 MLP)."""

    def __init__(self, state: dict):
        import torch
        if state.get("format") != "rudra_genfill_brightness_v1" or state.get("features") != FEATURES:
            raise ValueError("not a RUDRA brightness estimator for these features")
        self.net = torch.nn.Sequential(torch.nn.Linear(len(FEATURES), 32), torch.nn.GELU(),
                                       torch.nn.Linear(32, 32), torch.nn.GELU(), torch.nn.Linear(32, 1))
        self.net.load_state_dict({k: torch.tensor(v) for k, v in state["weights"].items()})
        self.mu = np.array(state["mu"], np.float32)
        self.sd = np.array(state["sd"], np.float32)
        self.meta = state.get("meta", {})

    def predict(self, feats: np.ndarray) -> np.ndarray:
        import torch
        x = torch.from_numpy(((np.atleast_2d(feats) - self.mu) / self.sd).astype(np.float32))
        with torch.no_grad():
            return torch.nn.functional.softplus(self.net(x)[:, 0]).numpy()

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class BrightnessSettings:
    estimator: Optional[BrightnessEstimator] = None
    offset: float = 0.0
    shape_gain: float = 1.0
    ema: float = 0.7
    identity: dict = field(default_factory=dict)


def excess_map(codes: np.ndarray, master: np.ndarray, estimator: Optional[BrightnessEstimator],
               offset: float = 0.0):
    lab, ids = components(codes)
    out = np.full(codes.shape[:2], np.nan, np.float32)
    if not ids:
        return out, lab
    fr = frame_features(codes, master)
    feats = np.stack([component_features(codes, master, lab == c, fr) for c in ids])
    b = estimator.predict(feats) if estimator is not None else np.zeros(len(ids), np.float32)
    for c, bc in zip(ids, b):
        out[lab == c] = max(0.0, float(bc) + offset)
    return out, lab


def compose(master: np.ndarray, generated: np.ndarray, b_map: np.ndarray, labels: np.ndarray,
            shape_gain: float = 1.0) -> np.ndarray:
    out = np.array(master, np.float32, copy=True)
    lm, lg = log2_lum(master), log2_lum(generated)
    for c in np.unique(labels[np.isfinite(b_map)]):
        s = (labels == c) & np.isfinite(b_map)
        rel = lg[s] - lm[s]
        shape = rel - rel.mean()
        ly = lm[s] + np.maximum(0.0, b_map[s] + shape_gain * shape)
        g = generated[s]
        yg = np.maximum(g @ AP1_Y, 1e-6)[:, None]
        ratio = np.where(yg > 1e-5, g / yg, master[s] / np.maximum(master[s] @ AP1_Y, 1e-6)[:, None])
        out[s] = ratio * (2.0 ** ly)[:, None]
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  model: Wan2.1-VACE components + RUDRA LoRA (mirrors diffusers' WanVACEPipeline)
# ─────────────────────────────────────────────────────────────────────────────

def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class PromptBank:
    """umT5 prompt embeddings, cached on disk by text (the 11 GB encoder loads at most once)."""

    def __init__(self, base, cache, device="cpu", dtype=None):
        import torch
        self.base, self.device, self.dtype = base, device, dtype or torch.bfloat16
        self.cache = Path(cache) if cache else None
        self._mem: Dict[str, object] = {}
        self._encoder = None

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha256(text.strip().encode()).hexdigest()[:24]

    def has(self, text: str) -> bool:
        """Whether get(text) is instant: in memory or in the disk cache (no umT5 load)."""
        k = self.key(text)
        return k in self._mem or bool(self.cache and (self.cache / f"{k}.pt").exists())

    def _load_encoder(self):
        from transformers import AutoTokenizer, UMT5EncoderModel
        ensure_text_encoder(self.base)
        tok = AutoTokenizer.from_pretrained(self.base, subfolder="tokenizer")
        enc = UMT5EncoderModel.from_pretrained(self.base, subfolder="text_encoder", torch_dtype=self.dtype)
        self._encoder = (tok, enc.to(self.device).eval())

    def release(self):
        import torch
        self._encoder = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def get(self, text: str):
        import torch
        k = self.key(text)
        if k in self._mem:
            return self._mem[k]
        f = self.cache / f"{k}.pt" if self.cache else None
        if f and f.exists():
            e = torch.load(f, map_location="cpu", weights_only=True)
            self._mem[k] = e
            return e
        if self.base is None:
            raise KeyError(f"no embeddings for prompt {text!r} and no text encoder to make them")
        if self._encoder is None:
            logger.info("[RUDRA AI fill] encoding a new prompt with umT5 (loaded once, on %s)", self.device)
            self._load_encoder()
        tok, enc = self._encoder
        with torch.no_grad():
            t = tok([" ".join(text.split())], padding="max_length", max_length=MAX_SEQUENCE_LENGTH,
                    truncation=True, add_special_tokens=True, return_attention_mask=True, return_tensors="pt")
            n = int(t.attention_mask.gt(0).sum())
            e = enc(t.input_ids.to(self.device), t.attention_mask.to(self.device)).last_hidden_state.float().cpu()
        e[:, n:] = 0
        if f:
            try:
                f.parent.mkdir(parents=True, exist_ok=True)
                torch.save(e, f)
                f.with_suffix(".json").write_text(json.dumps({"text": text, "tokens": n}))
            except OSError as exc:
                logger.debug("prompt cache not written: %s", exc)
        self._mem[k] = e
        return e


@dataclass
class FillSettings:
    steps: int = 30
    seed: int = 0
    conditioning_scale: float = 1.0
    guidance: float = 1.0
    negative: str = ""
    prompt: str = BASE_PROMPT
    prompt_highlight: str = ""
    prompt_shadow: str = ""
    # RUDRA 1.1 (5 Nov 2026): how strongly a region prompt steers its region. 1 = the region's
    # prediction replaces the base one there (as before, bit for bit); 0 = the prompt is ignored;
    # above 1 extrapolates away from the base prompt (stronger adherence, experimental: the v3
    # gates were passed at 1).
    prompt_strength: float = 1.0


def checked_prompt_strength(v) -> float:
    """prompt_strength: 0..2, finite."""
    x = float(v)
    if not (math.isfinite(x) and 0.0 <= x <= 2.0):
        raise ValueError(f"prompt_strength must be 0 to 2 (got {v})")
    return x


def load_lora(transformer, path) -> dict:
    """The RUDRA genfill LoRA onto *transformer*: through peft when it is
    installed (what RUDRA does), else merged into the Linear weights
    (W += alpha/r * B @ A), which computes the same function."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file
    with safe_open(str(path), "pt") as f:
        meta = f.metadata() or {}
    if meta.get("format") != "rudra_genfill_lora_v1":
        raise ValueError(f"{path}: not a RUDRA genfill LoRA")
    rank, alpha = int(meta["rank"]), float(meta["alpha"])
    state = load_file(str(path))
    try:
        from peft import LoraConfig
        from peft.utils import set_peft_model_state_dict
    except ImportError:
        LoraConfig = None
    if LoraConfig is not None:
        cfg = LoraConfig(r=rank, lora_alpha=alpha, target_modules=json.loads(meta["targets"]),
                         init_lora_weights=True)
        transformer.add_adapter(cfg, adapter_name="rudra_genfill")
        res = set_peft_model_state_dict(transformer, state, adapter_name="rudra_genfill")
        missing = [k for k in (res.missing_keys if res else []) if "lora_" in k]
        if missing:
            raise ValueError(f"{path}: LoRA weights missing for {len(missing)} modules (first {missing[0]})")
        meta["applied_with"] = "peft"
    else:
        modules = dict(transformer.named_modules())
        scale = alpha / rank
        done = 0
        for key in state:
            if not key.endswith(".lora_A.weight"):
                continue
            name = key[: -len(".lora_A.weight")]
            mod = modules.get(name)
            b_key = name + ".lora_B.weight"
            if mod is None or not hasattr(mod, "weight") or b_key not in state:
                raise ValueError(f"{path}: no module {name} in the base transformer")
            with torch.no_grad():
                delta = (state[b_key].float() @ state[key].float()) * scale
                mod.weight.copy_((mod.weight.float() + delta.to(mod.weight.device)).to(mod.weight.dtype))
            done += 1
        if not done:
            raise ValueError(f"{path}: no LoRA weights found")
        meta["applied_with"] = f"merged ({done} modules)"
    for p in transformer.parameters():
        p.requires_grad_(False)
    return meta


class WanFill:
    def __init__(self, vae, transformer, scheduler, prompts: PromptBank, device="cpu", dtype=None):
        import torch
        self.vae, self.transformer, self.scheduler, self.prompts = vae, transformer, scheduler, prompts
        self.device, self.dtype = torch.device(device), dtype or torch.bfloat16
        self.t_scale = 2 ** sum(vae.config.temperal_downsample)
        self.s_scale = 2 ** len(vae.config.temperal_downsample)
        self.patch = transformer.config.patch_size[1]
        z = vae.config.z_dim
        self._mean = torch.tensor(vae.config.latents_mean, dtype=torch.float32).view(1, z, 1, 1, 1)
        self._inv_std = 1.0 / torch.tensor(vae.config.latents_std, dtype=torch.float32).view(1, z, 1, 1, 1)
        self.identity: dict = {}

    @property
    def multiple(self) -> int:
        return self.s_scale * self.patch

    @classmethod
    def load(cls, base, lora=None, device="cuda", dtype=None, flow_shift: float = 3.0,
             prompt_cache=None, text_device: Optional[str] = None):
        import torch
        try:
            from diffusers import AutoencoderKLWan, WanVACETransformer3DModel
            from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
        except ImportError as exc:
            raise ImportError("RUDRA AI fill needs diffusers >= 0.35 (WanVACE): "
                              "pip install -U 'diffusers>=0.35'") from exc
        dtype = dtype or torch.bfloat16
        vae = AutoencoderKLWan.from_pretrained(base, subfolder="vae", torch_dtype=torch.float32).to(device).eval()
        tr = WanVACETransformer3DModel.from_pretrained(base, subfolder="transformer", torch_dtype=dtype)
        lora_meta = load_lora(tr, lora) if lora else {}
        tr = tr.to(device).eval()
        sch = UniPCMultistepScheduler.from_pretrained(base, subfolder="scheduler")
        sch = UniPCMultistepScheduler.from_config(sch.config, flow_shift=flow_shift)
        pb = PromptBank(base, prompt_cache, device=text_device or device, dtype=dtype)
        self = cls(vae, tr, sch, pb, device=device, dtype=dtype)
        self.identity = {"base": str(base), "lora": str(lora) if lora else None,
                         "lora_sha256": sha256_file(lora) if lora else None,
                         "lora_applied_with": lora_meta.get("applied_with"), "flow_shift": flow_shift,
                         "dtype": str(dtype).replace("torch.", ""), "vae_dtype": "float32"}
        return self

    def to(self, device):
        """Move to *device* (the GPU cache calls this on eviction)."""
        self.vae.to(device)
        self.transformer.to(device)
        return self

    def encode_pixels(self, video):
        import torch
        with torch.no_grad():
            z = self.vae.encode(video.to(self.device, torch.float32)).latent_dist.mode()
        return (z.float() - self._mean.to(z.device)) * self._inv_std.to(z.device)

    def decode_latents(self, z):
        import torch
        z = z.to(self.device, torch.float32) / self._inv_std.to(self.device) + self._mean.to(self.device)
        with torch.no_grad():
            return self.vae.decode(z, return_dict=False)[0].float().clamp(-1, 1)

    def mask_latent(self, mask):
        import torch.nn.functional as F
        _, _, n, h, w = mask.shape
        s = self.s_scale
        nf = (n + self.t_scale - 1) // self.t_scale
        m = mask[0, 0].float().view(n, h // s, s, w // s, s).permute(2, 4, 0, 1, 3).flatten(0, 1)
        return F.interpolate(m.unsqueeze(0), size=(nf, h // s, w // s), mode="nearest-exact")

    def context(self, video, mask):
        import torch
        _, _, n, h, w = video.shape
        if (n - 1) % self.t_scale or h % self.multiple or w % self.multiple:
            raise ValueError(f"frames must be 4k+1 and sizes multiples of {self.multiple} (got {n}x{h}x{w})")
        m = (mask > 0.5).float().to(self.device)
        v = video.to(self.device, torch.float32)
        inactive = self.encode_pixels(v * (1 - m))
        reactive = self.encode_pixels(v * m)
        return torch.cat([inactive, reactive, self.mask_latent(m).to(inactive)], dim=1)

    def _region_weights(self, masks: dict, shape) -> dict:
        import torch
        import torch.nn.functional as F
        out = {}
        f, h, w = shape[2:]
        for k, m in masks.items():
            if m is None:
                continue
            m = m.float()
            n = m.shape[2]
            groups = [m[:, :, :1]] + [m[:, :, 1 + 4 * i:1 + 4 * (i + 1)] for i in range((n - 1) // 4)]
            g = torch.cat([x.amax(2, keepdim=True) for x in groups], 2)
            out[k] = F.adaptive_max_pool3d(g, (f, h, w)).to(self.device)
        return out

    def _predict(self, x, t, embeds, ctx, scale):
        return self.transformer(hidden_states=x.to(self.dtype), timestep=t.expand(x.shape[0]),
                                encoder_hidden_states=embeds.to(self.device, self.dtype),
                                control_hidden_states=ctx.to(self.dtype), control_hidden_states_scale=scale,
                                return_dict=False)[0].float()

    def denoise(self, ctx, s: FillSettings, region_masks: Optional[dict] = None, progress=None):
        import torch
        with torch.no_grad():
            n_layers = len(self.transformer.config.vace_layers)
            scale = torch.full((n_layers,), float(s.conditioning_scale), device=self.device, dtype=self.dtype)
            g = torch.Generator("cpu").manual_seed(int(s.seed))
            shape = (1, self.transformer.config.in_channels) + tuple(ctx.shape[2:])
            x = torch.randn(shape, generator=g, dtype=torch.float32).to(self.device)
            base = self.prompts.get(s.prompt)
            regions = []
            weights = self._region_weights(region_masks or {}, shape)
            for key, text in (("highlight", s.prompt_highlight), ("shadow", s.prompt_shadow)):
                if text.strip() and key in weights and weights[key].any():
                    regions.append((weights[key], self.prompts.get(text)))
            neg = self.prompts.get(s.negative) if s.guidance > 1 else None
            self.scheduler.set_timesteps(s.steps, device=self.device)
            for i, t in enumerate(self.scheduler.timesteps):
                eps = self._predict(x, t, base, ctx, scale)
                for w, e in regions:
                    if s.prompt_strength == 1.0:
                        eps = eps * (1 - w) + self._predict(x, t, e, ctx, scale) * w
                    elif s.prompt_strength != 0.0:
                        eps = eps + (self._predict(x, t, e, ctx, scale) - eps) * (w * float(s.prompt_strength))
                if neg is not None:
                    un = self._predict(x, t, neg, ctx, scale)
                    eps = un + s.guidance * (eps - un)
                x = self.scheduler.step(eps, t, x, return_dict=False)[0]
                if progress:
                    progress(i + 1, len(self.scheduler.timesteps))
            return x


# ─────────────────────────────────────────────────────────────────────────────
#  run: frames in memory
# ─────────────────────────────────────────────────────────────────────────────

def plan_windows(n: int, window: int, overlap: int, full_last: bool = False) -> List[Tuple[int, int, int]]:
    """Frames 0..n-1 -> [(start, first_new, end)]; every frame written once.
    full_last (RUDRA 1.2.9): the last window is a whole window, started earlier (n - window) so it shares
    more frames with the one before, instead of a short window padded with copies of the shot's last frame."""
    if window < 1 or (window - 1) % 4:
        raise ValueError("window must be 4k+1")
    if not 0 <= overlap < window:
        raise ValueError("overlap must be in [0, window)")
    out, written = [], 0
    while written < n:
        start = max(0, written - overlap) if written else 0
        if full_last and written and start + window > n:
            start = max(0, min(start, n - window))
        end = min(n, start + window)
        out.append((start, written, end))
        written = end
    return out


def pad_len(k: int) -> int:
    return k if (k - 1) % 4 == 0 else k + (4 - (k - 1) % 4)


def working_size(h: int, w: int, max_area: int, multiple: int = 16) -> Tuple[int, int]:
    s = min(1.0, math.sqrt(max_area / float(h * w)))
    return (max(multiple, int(h * s // multiple) * multiple),
            max(multiple, int(w * s // multiple) * multiple))


def temporal_excess(prev, cur) -> Optional[float]:
    if prev is None:
        return None
    sel = (cur[2] > 0) | (prev[2] > 0)
    if not sel.any() or prev[0].shape != cur[0].shape:
        return None
    lum = lambda x: np.log2(np.maximum(x @ np.array([0.2722, 0.6741, 0.0537], np.float32), 1e-4))  # noqa: E731
    d_fill = np.abs(lum(cur[1]) - lum(prev[1]))[sel].mean()
    d_master = np.abs(lum(cur[0]) - lum(prev[0]))[sel].mean()
    return float(max(0.0, d_fill - d_master))


def _small(x: np.ndarray, side: int = 256) -> np.ndarray:
    import cv2
    h, w = x.shape[:2]
    s = min(1.0, side / float(max(h, w)))
    return cv2.resize(x, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else x


TEMPORAL_FLAG = 0.25      # stops of frame-to-frame change the fill adds beyond the master's own
FALLBACKS = ("none", "spikes")   # RUDRA 1.1: what a flickering fill frame falls back to (fill_frames)


def _over(temporal) -> bool:
    return bool(temporal is not None and temporal > TEMPORAL_FLAG)


# ── focus (RUDRA 7-8 Nov 2026): what the model sees ───────────────────────────
FOCUS = ("frame", "region", "areas")   # the whole frame, one box round what the shot fills,
                                       # or one box per group of areas
FOCUS_MARGIN = 0.15           # context around the box, a share of its own size each side ...
FOCUS_MIN_MARGIN = 48         # ... and at least this many pixels
FOCUS_MIN_SIDE = 256          # a thin strip gets this much of the frame around it, at least
FOCUS_MAX_SHARE = 0.6         # a box this much of the frame or more: the whole frame is as good
FOCUS_COVER = 0.9             # the box takes the largest areas until this share of the shot's mask
FOCUS_MAX_BOXES = 8           # 'areas': at most this many boxes (closest ones merge first); each is
                              # one more model run per window, so the time grows with the count


def _grow(a: int, b: int, size: int, extra: int) -> Tuple[int, int]:
    """[a, b) with context each side (FOCUS_MARGIN of its size, at least FOCUS_MIN_MARGIN + extra),
    at least FOCUS_MIN_SIDE long, shifted to stay inside [0, size)."""
    m = max(FOCUS_MIN_MARGIN + extra, int(np.ceil(FOCUS_MARGIN * (b - a))) + extra)
    a, b = a - m, b + m
    short = FOCUS_MIN_SIDE - (b - a)
    if short > 0:
        a, b = a - short // 2, b + short - short // 2
    if a < 0:
        a, b = 0, b - a
    if b > size:
        a, b = max(0, a - (b - size)), size
    return max(0, a), min(size, b)


def focus_box(union: np.ndarray, extra: int = 0, cover: float = FOCUS_COVER) -> Optional[Tuple[int, int, int, int]]:
    """The shot's fill masks, OR-ed over every frame (H, W bool) -> (y0, y1, x0, x1): a box with
    context around the largest connected areas, taken until `cover` of the mask is in them, the same
    box for every frame so the fill moves as one picture. Specks far from them do not stretch it to
    the whole frame; the fill leaves them to the master. None when there is nothing to fill or the
    box is most of the frame. `extra`: pixels the blend reaches past the masks (the feather)."""
    import cv2
    H, W = union.shape
    n, _, st, _ = cv2.connectedComponentsWithStats(union.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None
    total, got = float(st[1:, 4].sum()), 0.0
    y0, y1, x0, x1 = H, 0, W, 0
    for k in np.argsort(-st[1:, 4], kind="stable") + 1:
        x, y, w, h, a = (int(v) for v in st[k])
        y0, y1, x0, x1 = min(y0, y), max(y1, y + h), min(x0, x), max(x1, x + w)
        got += a
        if got >= cover * total:
            break
    y0, y1 = _grow(y0, y1, H, extra)
    x0, x1 = _grow(x0, x1, W, extra)
    if (y1 - y0) * (x1 - x0) >= FOCUS_MAX_SHARE * H * W:
        return None
    return y0, y1, x0, x1


def _overlap(a, b) -> bool:
    return a[0] < b[1] and b[0] < a[1] and a[2] < b[3] and b[2] < a[3]


def _merge(a, b):
    return min(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), max(a[3], b[3])


def _area(b) -> int:
    return (b[1] - b[0]) * (b[3] - b[2])


def focus_boxes(union: np.ndarray, extra: int = 0, cover: float = FOCUS_COVER,
                max_boxes: int = FOCUS_MAX_BOXES) -> Optional[List[Tuple[int, int, int, int]]]:
    """'areas': the shot's fill masks, OR-ed over every frame -> boxes that do not overlap, one per
    group of connected areas, each with context (as focus_box), the same boxes for every frame.
    Overlapping boxes merge; while there are more than `max_boxes` the pair whose merged box grows
    least merges. None when there is nothing to fill or the boxes cover most of the frame."""
    import cv2
    H, W = union.shape
    n, _, st, _ = cv2.connectedComponentsWithStats(union.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None
    total, got, boxes = float(st[1:, 4].sum()), 0.0, []
    for k in np.argsort(-st[1:, 4], kind="stable") + 1:
        x, y, w, h, a = (int(v) for v in st[k])
        boxes.append(_grow(y, y + h, H, extra) + _grow(x, x + w, W, extra))
        got += a
        if got >= cover * total:
            break
    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if _overlap(boxes[i], boxes[j]):
                    boxes[i] = _merge(boxes[i], boxes.pop(j))
                    changed = True
                    break
            if changed:
                break
        if not changed and len(boxes) > max_boxes:
            i, j = min(((i, j) for i in range(len(boxes)) for j in range(i + 1, len(boxes))),
                       key=lambda p: _area(_merge(boxes[p[0]], boxes[p[1]])) - _area(boxes[p[0]]) - _area(boxes[p[1]]))
            boxes[i] = _merge(boxes[i], boxes.pop(j))
            changed = True
    if sum(_area(b) for b in boxes) >= FOCUS_MAX_SHARE * H * W:
        return None
    return sorted(boxes)


# ── v4: region re-render ─────────────────────────────────────────────────────

@dataclass
class Rerender:
    """The region re-render (RUDRA AI fill v4): where the codes come from and what the model is
    conditioned on, and the render-time finish (finish_rerender). Defaults: everything off, the
    model's own colour and level. RUDRA_APP_V4 holds the app's settings."""
    condition: str = "bt1886"   # 'bt1886' (code ^ 2.4, the analytic route) or 'aces' (the ACES 2.0 inverse)
    hue: str = "model"     # 'source': the fill's luminance with the source's colour
    colour: float = 0.0    # with hue 'source': the share of the model's own colour kept, 0..1
    level: float = 0.0     # > 0: the level where the source is clipped, in the master's SDR whites
    smooth: float = 0.0    # > 0: the fill's log luminance follows the previous frame's (EMA share kept)
    dip: float = 0.0       # > 0: the fill's fine detail at most this many stops under its surroundings
    # RUDRA 1.2.x and 1.3 (run.py Rerender); finish_rerender says what each does.
    still: float = 0.0     # > 0: the share of the previous frame kept where the source holds still
    independent: bool = False   # every window from the source alone, the overlaps crossfaded (not chained)
    rise: float = 0.0      # > 0: with a level, the full-code core climbs up to this many stops with its depth
    calm: float = 0.0      # > 0: the share kept where the source holds still and the fill would jump
    keep_detail: bool = False   # 1.2.7: thin, broken-up and small parts of the region keep the master
    full_last: bool = False     # 1.2.9: the last window a whole window (plan_windows full_last)
    hold_clipped: str = ""      # 1.3: with calm, still clipped pixels held: '' off, 'all', 'thin' (v5), 'dips', 'release'
    steady: int = 0        # 1.3: > 1, the median of the fill's last frames where the source clips and holds still
    cap_master: bool = False    # 1.3, the learned route: the master's share never over the fill's own value
    blend_still: float = 0.0    # 1.3: the crossfade's mix keeps this share of the previous frame's while its code holds still
    hold_calm: float = 0.0      # 1.3: with hold_clipped, the share a held pixel keeps for a jump instead of calm
    CONDITIONS = ("bt1886", "aces")

    def check(self) -> "Rerender":
        if self.condition not in self.CONDITIONS:
            raise ValueError(f"condition must be one of {', '.join(self.CONDITIONS)}")
        if self.hue not in RERENDER_HUES:
            raise ValueError(f"hue must be one of {', '.join(RERENDER_HUES)}")
        if self.level < 0 or not 0.0 <= self.smooth < 1.0 or not 0.0 <= self.colour <= 1.0 or self.dip < 0:
            raise ValueError("level must be >= 0, smooth in [0, 1), colour in [0, 1] and dip >= 0")
        if (self.rise < 0 or not 0.0 <= self.still < 1.0 or not 0.0 <= self.calm < 1.0
                or not 0 <= int(self.steady) <= RERENDER_STEADY_MAX):
            raise ValueError(f"rise must be >= 0, still and calm in [0, 1), steady 0..{RERENDER_STEADY_MAX}")
        if not (math.isfinite(self.blend_still) and 0.0 <= self.blend_still < 1.0):
            raise ValueError("blend_still must be in [0, 1)")
        if not (math.isfinite(self.hold_calm) and 0.0 <= self.hold_calm < 1.0):
            raise ValueError("hold_calm must be in [0, 1)")
        self.hold_clipped = hold_mode(self.hold_clipped)
        return self

    @property
    def finishes(self) -> bool:
        """Whether finish_rerender has anything to do (run.py's test before it calls it)."""
        return (self.hue != "model" or self.level > 0 or self.smooth > 0 or self.still > 0 or self.calm > 0
                or self.steady > 1)


HOLD_MODES = ("all", "thin", "dips", "release")


def hold_mode(v) -> str:
    """'' (off), or one of HOLD_MODES; True is 'all' (RUDRA run.py hold_mode)."""
    if v is True:
        return "all"
    if v in (None, False, "", "off", 0):
        return ""
    v = str(v)
    if v not in HOLD_MODES:
        raise ValueError(f"rerender hold must be off or one of {', '.join(HOLD_MODES)}, not {v!r}")
    return v


#: RUDRA's app (native/app/genfill_job.cpp at 72c4a62): the amended inference of 11 Nov 2026 as
#: gated, plus the 12 Nov dip limit. Masks: tau_high 240, no shadows, feather 8, no estimator.
RUDRA_APP_V4 = dict(hue="source", colour=0.3, level=3.3, smooth=0.8, dip=0.3)
RUDRA_APP_V4_MASK = dict(tau_high=240, tau_low=-1, feather=8)

#: RUDRA 1.3.0's app (native/app/genfill_job.cpp genfill_arguments, 24 Nov 2026) with AI fill v4.3:
#: the 12 Nov finish, windows from the source alone (independent) with a jump on still pixels held
#: (calm 0.9, 15 Nov), and the last window a whole one (full_last, 1.2.9). Not align, not follow.
RUDRA_APP_V43 = dict(RUDRA_APP_V4, independent=True, calm=0.9, full_last=True)
#: ... and with AI fill v5: the thin hold (research_genfill_v5d_thin_20261123) and the v5e finish
#: (research_genfill_v5e_20261124): the crossfade's mix settles against the codec's noise, a held thin
#: shape keeps more of its last frame for a jump.
RUDRA_APP_V5 = dict(RUDRA_APP_V43, hold_clipped="thin", blend_still=0.95, hold_calm=0.95)
#: ... and on the learned route (condition 'aces'): the master's share capped at the fill, and a newly
#: clipped pixel in the mask at its first frame (MaskSettings.first_frame).
RUDRA_APP_V5_LEARNED = dict(cap_master=True)
RUDRA_APP_V5_LEARNED_MASK = dict(first_frame=True)

#: research_genfill_v4_20261109/rudra_genfill.safetensors, the LoRA the v4 gates were run with.
GENFILL_V4_SHA256 = "95d707eb3e5628d7515187c0752018a23182f186edb060d8652f77a64466375c"
#: The published region re-render LoRAs (fxtdstudios/rudra-models; RUDRA genfill_job.hpp).
GENFILL_V43_SHA256 = "406b1dc7b7d3ada5406b396a6aa8c7e5ec9eac8d0be0d04e8038d68b5bf1b390"   # AI fill v4.3, RUDRA 1.2.4
GENFILL_V5_SHA256 = "5c5d4ab4107088f86bdc8213ae2ec21c1a882cc6d942141f0e12fb7733fda5e6"    # AI fill v5, RUDRA 1.3.0


def condition_of(codes: np.ndarray, condition: str, view) -> np.ndarray:
    """The model's condition for 8-bit codes (H, W, 3): ACEScg float32. 'aces': the ACES 2.0
    inverse (the view's); 'bt1886': the display light, code ^ 2.4, Rec.709 in AP1."""
    if condition == "aces":
        return view.inverse(codes.astype(np.float32) / 255.0)
    from radiance.model.rudra_aces2 import M_REC709_TO_AP1
    lin = np.power(codes.astype(np.float32) / 255.0, 2.4)
    return (lin.astype(np.float64) @ M_REC709_TO_AP1.T).astype(np.float32)


def fill_frames(masters: List[np.ndarray], model, view, s: FillSettings, ms: MaskSettings,
                window: int = 33, overlap: int = 8, max_area: int = 832 * 480,
                brightness: Optional[BrightnessSettings] = None,
                progress: Optional[Callable[..., None]] = None,
                focus: str = "frame", fallback: str = "none",
                rerender: Optional[Rerender] = None,
                source_codes: Optional[List[np.ndarray]] = None) -> Tuple[List[np.ndarray], List[np.ndarray], dict]:
    """ACEScg masters (H,W,3) in frame order -> (filled masters, generated weights, report).

    RUDRA's run.fill_master step for step, frames in memory: the model sees the
    exact inverse of the master's SDR codes (log-encoded) at a working size,
    windows of `window` frames overlapping by `overlap` (filled frames from the
    previous window come in as kept frames, so windows join without a seam),
    and the fill is blended back only where the source holds no information,
    never changing an 8-bit code of the master's SDR view.

    focus: 'frame' (v3) the whole frame at max_area; 'region' one box round everything the shot
    fills; 'areas' one box per group of areas, each drawn on its own at up to max_area. Outside
    the boxes nothing is generated.
    fallback: 'spikes' writes the master's own frame for a frame whose fill jumps more than
    TEMPORAL_FLAG stops beyond the master's own change both into it and out of it; 'none' keeps
    every fill. Either way a fill that is not finite falls back.
    rerender (v4): source_codes, the SDR the master was made from as 8-bit codes (H, W, 3) per
    frame, give the regions and the condition; brightness and the gain cap are off; the fill is
    finished and blended plainly (the codes inside the mask are not kept).
    """
    import cv2
    import torch
    if fallback not in FALLBACKS:
        raise ValueError(f"fallback must be one of {', '.join(FALLBACKS)}")
    if focus not in FOCUS:
        raise ValueError(f"focus must be one of {', '.join(FOCUS)}")
    if not masters:
        return [], [], {"frames": 0}
    H, W = masters[0].shape[:2]
    for m in masters:
        if m.shape[:2] != (H, W):
            raise ValueError("every frame must have the first frame's size")
    if rerender is not None:
        rerender.check()
        if source_codes is None or len(source_codes) != len(masters):
            raise ValueError("the region re-render needs the source's codes for every frame")
        for c in source_codes:
            if c.shape != (H, W, 3):
                raise ValueError("the source's frames must have the master's size")
        ms = MaskSettings(**{**asdict(ms), "blend_high": ms.tau_high, "blend_low": ms.tau_low,
                             "max_gain_stops": 0.0, "min_core": 0.0})   # v4 blends its whole mask, specks too
        brightness = None
    finish_state: dict = {}
    finish_white = 1.0
    if rerender is not None and rerender.level > 0:   # the shot's SDR white: up to 9 frames along it
        picks = sorted({int(k) for k in np.linspace(0, len(masters) - 1, min(9, len(masters)))})
        seen = [white_scale(masters[k], source_codes[k]) for k in picks]
        seen = [x for x in seen if x]
        finish_white = float(np.median(seen)) if seen else 1.0

    def codes_of(i):
        if source_codes is not None and rerender is not None:
            return np.asarray(source_codes[i], np.uint8)
        return np.rint(np.clip(view.forward(masters[i]), 0, 1) * 255).astype(np.uint8)

    box, boxes, inside, outside_share = None, None, None, 0.0
    if focus in ("region", "areas"):                   # a first pass: the masks the fill will use
        pre, union = TemporalMask(ms), np.zeros((H, W), bool)
        for k in range(len(masters)):
            hi, lo = pre(codes_of(k))
            union |= hi | lo
        if focus == "region":
            box = focus_box(union, extra=ms.feather)
            boxes = [box] if box else None
        else:
            boxes = focus_boxes(union, extra=ms.feather)
        if boxes:
            inside = np.zeros((H, W), bool)
            for b in boxes:
                inside[b[0]:b[1], b[2]:b[3]] = True
            outside_share = float((union & ~inside).sum()) / max(1.0, float(union.sum()))
    lanes = []                                       # one per box the model draws (the whole frame if none)
    for b in (boxes or [(0, H, 0, W)]):
        lh, lw = working_size(b[1] - b[0], b[3] - b[2], max_area, model.multiple)
        lanes.append({"box": b, "crop": (slice(b[0], b[1]), slice(b[2], b[3])), "h": lh, "w": lw, "gen_work": {}})
    y0, y1, x0, x1 = lanes[0]["box"]
    h, w = lanes[0]["h"], lanes[0]["w"]
    plan = plan_windows(len(masters), window, overlap, full_last=bool(rerender is not None and rerender.full_last))
    tmask = TemporalMask(ms)
    prev_b = None
    masks: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    codes_all: Dict[int, np.ndarray] = {}
    outs: List[Optional[np.ndarray]] = [None] * len(masters)
    alphas: List[Optional[np.ndarray]] = [None] * len(masters)
    base_gain = working_size(H, W, max_area, model.multiple)[1] / float(W)
    stats = {"frames": len(masters), "windows": len(plan), "generated_pixels": 0, "region_pixels": 0,
             "windows_generated": 0, "working_size": [h, w], "size": [H, W],
             "focus": {"mode": focus, "box": [y0, y1, x0, x1] if box else None,
                       "mask_outside_box": round(outside_share, 4),
                       "detail_gain": round((w / float(x1 - x0)) / base_gain, 3)}}
    if focus == "areas":
        stats["focus"]["boxes"] = [{"box": list(L["box"]), "working_size": [L["h"], L["w"]],
                                    "detail_gain": round((L["w"] / float(L["box"][3] - L["box"][2])) / base_gain, 3)}
                                   for L in lanes] if boxes else []
        stats["focus"]["detail_gain"] = min(b["detail_gain"] for b in stats["focus"]["boxes"]) if boxes else 1.0
    if rerender is not None and rerender.level > 0:
        stats["rerender_white"] = round(finish_white, 4)
    per_frame = []
    prev_small = None
    pending = None                                   # the frame waiting for the next one (fallback)

    def commit(e, spike_reason):
        i, out, alpha = e["i"], e["out"], e["alpha"]
        reason = e["reason"] or spike_reason
        if reason and alpha.any():
            out, alpha = masters[i], np.zeros_like(alpha)
        outs[i], alphas[i] = out, alpha
        per_frame.append({"frame": i, "generated_fraction": float((alpha > 0).mean()),
                          "temporal_excess_stops": e["temporal"], "flagged": _over(e["temporal"]),
                          "fell_back": reason})
        stats["generated_pixels"] += int((alpha > 0).sum())
        stats["region_pixels"] += e["region_px"]
        if progress:
            progress(frame=i + 1, frames=len(masters))

    # Independent windows (RUDRA, 13 Nov 2026): a chained window takes the previous window's generated frames
    # as its context, and the model built on its own detail window after window. Independent, every window
    # starts from the source alone; the frames two windows share are held back (carry) and crossfaded, the
    # later window's share rising across the overlap.
    indep = rerender is not None and rerender.independent
    carry: Dict[int, dict] = {}
    for wi, (start, new, end) in enumerate(plan):
        idxs = list(range(start, end))
        for i in idxs:
            if i not in codes_all:
                codes_all[i] = codes_of(i)
            if i not in masks:
                hi, lo = tmask(codes_all[i])
                if inside is not None:
                    hi, lo = hi & inside, lo & inside
                masks[i] = (hi, lo)
        n_pad = pad_len(len(idxs))
        padded = idxs + [idxs[-1]] * (n_pad - len(idxs))
        decs = {}
        for li, L in enumerate(lanes):
            lcrop, lh, lw, lgen = L["crop"], L["h"], L["w"], L["gen_work"]
            if not any(masks[i][0][lcrop].any() or masks[i][1][lcrop].any() for i in range(new, end)):
                continue
            vid = np.zeros((n_pad, lh, lw, 3), np.float32)
            msk = np.zeros((n_pad, lh, lw), np.float32)
            regions_np = {"highlight": np.zeros((n_pad, lh, lw), np.float32),
                          "shadow": np.zeros((n_pad, lh, lw), np.float32)}
            for k, i in enumerate(padded):
                if i < new and i in lgen and not indep:   # kept frame from the previous window
                    vid[k] = lgen[i]
                    continue
                cond = condition_of(codes_all[i], rerender.condition if rerender else "aces", view)
                vid[k] = cv2.resize(encode(cond[lcrop]).astype(np.float32), (lw, lh), interpolation=cv2.INTER_AREA)
                region = (masks[i][0] | masks[i][1])[lcrop].astype(np.float32)
                msk[k] = cv2.resize(region, (lw, lh), interpolation=cv2.INTER_AREA) > 0
                for key, j in (("highlight", 0), ("shadow", 1)):
                    regions_np[key][k] = cv2.resize(masks[i][j][lcrop].astype(np.float32), (lw, lh),
                                                    interpolation=cv2.INTER_AREA) > 0
            video = torch.from_numpy(vid).permute(3, 0, 1, 2)[None]
            mt = torch.from_numpy(msk)[None, None]
            regions = {k: torch.from_numpy(v)[None, None] for k, v in regions_np.items()}

            def step(a, b, _wi=wi, _li=li):
                if progress:
                    progress(window=_wi + 1, windows=len(plan), step=_li * b + a, steps=b * len(lanes))

            ctx = model.context(video, mt)
            z = model.denoise(ctx, s, regions, progress=step)
            decs[li] = model.decode_latents(z)[0].permute(1, 2, 3, 0).cpu().numpy()
            stats["windows_generated"] += 1
        for k, i in enumerate(idxs):
            if i < new and not (indep and i in carry):
                continue
            dk = {li: dec[k] for li, dec in decs.items()}
            if indep:
                if i in carry:                       # shared with the previous window: crossfade
                    t = np.float32((i - start + 1) / float(new - start + 1))
                    for li, c in carry.pop(i).items():
                        dk[li] = c if li not in dk else ((1.0 - t) * c + t * dk[li]).astype(np.float32)
                if wi < len(plan) - 1 and i >= plan[wi + 1][0]:   # the next window shares it: written then
                    carry[i] = dk
                    continue
            master = masters[i]
            codes = codes_all[i]
            region = masks[i][0] | masks[i][1]
            if dk and region.any():
                fill_lin = np.array(master, np.float32, copy=True)   # outside every box: the master
                for li, dki in dk.items():
                    L = lanes[li]
                    L["gen_work"][i] = dki
                    (a0, a1, b0, b1), lcrop = L["box"], L["crop"]
                    part = cv2.resize(dki, (b1 - b0, a1 - a0), interpolation=cv2.INTER_CUBIC)
                    fill_lin[lcrop] = decode(part.astype(np.float32)).astype(np.float32)
                if brightness is not None:
                    b_map, lab = excess_map(codes, master, brightness.estimator, brightness.offset)
                    if prev_b is not None and prev_b.shape == b_map.shape and brightness.ema > 0:
                        both = np.isfinite(b_map) & np.isfinite(prev_b)
                        b_map[both] = brightness.ema * prev_b[both] + (1 - brightness.ema) * b_map[both]
                    prev_b = b_map
                    fill_lin = compose(master, fill_lin, b_map, lab, brightness.shape_gain)
                if ms.max_gain_stops > 0:
                    lum = lambda x: np.maximum(x @ AP1_Y, 1e-6)  # noqa: E731
                    lim = lum(master) * np.float32(2.0 ** ms.max_gain_stops)
                    yf = lum(fill_lin)
                    fill_lin = fill_lin * np.where(yf > lim, lim / yf, 1.0)[..., None].astype(np.float32)
                core = blend_core(codes, ms) & region
                weight = feather(core, ms.feather) if ms.feather > 0 else core.astype(np.float32)
                if rerender is not None:             # v4: the whole mask, the codes inside it not kept
                    # The dilated, feathered mask reaches dark pixels beside the region (a figure
                    # against the sky): the weight follows the source's codes, a ramp of 8 below tau_high.
                    top = codes.max(-1).astype(np.float32)
                    ramp = np.clip((top - (ms.tau_high - 8)) / 8.0, 0.0, 1.0)
                    w_plain = (weight * ramp * ramp * (3 - 2 * ramp)).astype(np.float32)
                    if rerender.level > 0:           # only where the source is near white in every channel
                        w_plain = (w_plain * neutral_weight(codes)).astype(np.float32)
                    if rerender.keep_detail:         # 1.2.7: a grating, a railing, a lamp keep their picture
                        w_plain = (w_plain * detail_keep_weight(codes, ms.tau_high)).astype(np.float32)
                    if rerender.finishes:
                        fill_lin = finish_rerender(fill_lin, master, condition_of(codes, "bt1886", view), codes,
                                                   w_plain, rerender.hue, rerender.colour, rerender.level,
                                                   rerender.smooth, finish_white, ms.tau_high, finish_state,
                                                   dip=rerender.dip, still=rerender.still, rise=rerender.rise,
                                                   calm=rerender.calm, hold=rerender.hold_clipped or False,
                                                   steady=rerender.steady, cap=rerender.cap_master,
                                                   blend_still=rerender.blend_still, hold_calm=rerender.hold_calm)
                    out, alpha = plain_blend(master, fill_lin, w_plain)
                else:
                    out, alpha = r_safe_blend(master, fill_lin, weight, view)
            else:
                for li, dki in dk.items():
                    lanes[li]["gen_work"][i] = dki
                out, alpha = master, np.zeros((H, W), np.float32)
            reason = None if np.isfinite(out).all() else "non_finite"
            if reason:                               # never return a broken fill
                out, alpha = master, np.zeros((H, W), np.float32)
            cur_small = (_small(master), _small(out), _small(alpha.astype(np.float32)))
            temporal = temporal_excess(prev_small, cur_small)
            prev_small = cur_small
            entry = {"i": i, "out": out, "alpha": alpha, "region_px": int(region.sum()),
                     "temporal": temporal, "reason": reason}
            if pending is not None:
                spike = (fallback == "spikes" and pending["alpha"].any() and _over(pending["temporal"])
                         and _over(temporal))
                commit(pending, "flicker" if spike else None)
            pending = entry
        for L in lanes:                              # keep only what the next window reads
            for i in list(L["gen_work"]):
                if i < (plan[wi + 1][0] if wi < len(plan) - 1 else end - overlap):
                    del L["gen_work"][i]
        for i in list(masks):
            if i < start:
                del masks[i]
        for i in list(codes_all):
            if i < start:
                del codes_all[i]
    if pending is not None:                          # the last frame: nothing after it to spike against
        commit(pending, None)
    stats["flagged_frames"] = sum(1 for f in per_frame if f["flagged"])
    stats["fallback_frames"] = sum(1 for f in per_frame if f["fell_back"])
    stats["generated_fraction"] = stats["generated_pixels"] / float(max(1, len(masters) * H * W))
    contract = ("region re-render (v4): inside the mask the SDR codes are NOT kept (plain_blend); outside it "
                "every pixel is the master's, bit-identical" if rerender is not None else
                "no 8-bit code of the master SDR view changed (r_safe_blend); untouched pixels bit-identical")
    report = {"model": getattr(model, "identity", {}), "settings": asdict(s), "mask": asdict(ms),
              "brightness": None if brightness is None else {**brightness.identity, "offset": brightness.offset,
                                                             "shape_gain": brightness.shape_gain},
              "focus": focus, "fallback": fallback,
              "rerender": asdict(rerender) if rerender is not None else None,
              "window": window, "overlap": overlap, "max_area": max_area, "stats": stats, "frames": per_frame,
              "contract": contract, "release_authorized": False}
    return outs, alphas, report


# ─────────────────────────────────────────────────────────────────────────────
#  finding the files
# ─────────────────────────────────────────────────────────────────────────────

def resolve_base(explicit: str = "") -> Optional[str]:
    """Wan2.1-VACE-1.3B-diffusers: explicit folder, RADIANCE_WAN_VACE, models/radiance/,
    models/diffusers/ (ComfyUI's), or None."""
    from radiance.model.paths import radiance_model_dirs
    cands = []
    for v in (explicit, os.environ.get(BASE_ENV, "")):
        v = (v or "").strip().strip('"').strip("'")
        if v:
            cands.append(Path(v).expanduser())
    for d in radiance_model_dirs():
        cands += [d / BASE_DIRNAME, d.parent / "diffusers" / BASE_DIRNAME]
    for c in cands:
        if (c / "model_index.json").is_file() and (c / "transformer").is_dir() and (c / "vae").is_dir():
            return str(c)
    return None


def resolve_package(explicit: str = "") -> Optional[Path]:
    from radiance.model.rudra_download import GENFILL_V3, find_package
    return find_package(GENFILL_V3, explicit, PACKAGE_ENV)


V4_ENV = "RADIANCE_RUDRA_GENFILL_V4"
V4_DIRNAME = "rudra_genfill_v4"


def resolve_v4_lora(explicit: str = "") -> Optional[Path]:
    """The v4 region re-render's LoRA (never downloaded: the published v4.3 needs RUDRA 1.2.4's
    settings, which this port lacks): *explicit* (the file,
    or a folder holding rudra_genfill.safetensors), RADIANCE_RUDRA_GENFILL_V4, then
    models/radiance/rudra_genfill_v4/rudra_genfill.safetensors or models/radiance/rudra_genfill_v4.safetensors."""
    from radiance.model.paths import radiance_model_dirs
    cands = []
    for v in (explicit, os.environ.get(V4_ENV, "")):
        v = (v or "").strip().strip('"').strip("'")
        if v:
            p = Path(v).expanduser()
            cands.append(p / "rudra_genfill.safetensors" if p.is_dir() else p)
    for d in radiance_model_dirs():
        cands += [d / V4_DIRNAME / "rudra_genfill.safetensors", d / f"{V4_DIRNAME}.safetensors"]
    for c in cands:
        if c.is_file():
            return c
    return None


REGION_MODELS = ("v5", "v4.3", "v4 research")   # the region re-render's LoRAs, newest first
V5_ENV = "RADIANCE_RUDRA_GENFILL_V5"


def region_package(model: str):
    """The rudra_download.Package of a published region LoRA ('v5', 'v4.3'), or None ('v4 research')."""
    from radiance.model.rudra_download import GENFILL_V5, GENFILL_V43
    return {"v5": GENFILL_V5, "v4.3": GENFILL_V43}.get(model)


def resolve_region_lora(model: str = "v5", explicit: str = "") -> Optional[Path]:
    """The region re-render's LoRA for *model*: *explicit* (the file, or a folder holding
    rudra_genfill.safetensors), then for 'v5' RADIANCE_RUDRA_GENFILL_V5 and models/radiance/rudra_genfill_v5/,
    for 'v4.3' the published package in models/radiance/rudra_genfill_v4/ (checked by its hash, so the 12 Nov
    research LoRA in the same folder is not taken for it), for 'v4 research' resolve_v4_lora."""
    if model not in REGION_MODELS:
        raise ValueError(f"region model must be one of {', '.join(REGION_MODELS)}")
    if model == "v4 research":
        return resolve_v4_lora(explicit)
    v = (explicit or "").strip().strip('"').strip("'")
    if v:
        p = Path(v).expanduser()
        p = p / "rudra_genfill.safetensors" if p.is_dir() else p
        return p if p.is_file() else None
    if model == "v5":
        env = os.environ.get(V5_ENV, "").strip().strip('"').strip("'")
        if env:
            p = Path(env).expanduser()
            p = p / "rudra_genfill.safetensors" if p.is_dir() else p
            if p.is_file():
                return p
    from radiance.model.rudra_download import find_package
    folder = find_package(region_package(model))
    return folder / "rudra_genfill.safetensors" if folder is not None else None


def ensure_region_lora(model: str = "v5", explicit: str = "", allow: Optional[bool] = None) -> Optional[Path]:
    """resolve_region_lora, downloading the published v5 or v4.3 package (263 MB) on first use unless
    downloads are off. The 12 Nov research LoRA is never downloaded."""
    found = resolve_region_lora(model, explicit)
    if found is not None or model == "v4 research" or (explicit or "").strip():
        return found
    from radiance.model.rudra_download import download_package
    got = download_package(region_package(model), allow=allow)
    return Path(got) / "rudra_genfill.safetensors" if got is not None else None


def check_runtime() -> None:
    """Fail before any download when this Python cannot run Wan VACE."""
    try:
        import diffusers
        from diffusers import AutoencoderKLWan, WanVACETransformer3DModel  # noqa: F401
    except ImportError as exc:
        import sys
        ver = getattr(sys.modules.get("diffusers"), "__version__", "not installed")
        raise RuntimeError(
            f"RUDRA AI fill needs diffusers 0.35 or newer (found {ver}). Run: "
            f'"{sys.executable}" -m pip install -U "diffusers>=0.35" and restart ComfyUI.') from exc


def ensure_package(explicit: str = "", allow: Optional[bool] = None) -> Optional[Path]:
    """The AI fill v3 package, downloaded on first use (271 MB) unless downloads are off."""
    found = resolve_package(explicit)
    if found is not None:
        return found
    from radiance.model.rudra_download import GENFILL_V3, download_package
    return download_package(GENFILL_V3, allow=allow)


def ensure_base(explicit: str = "", allow: Optional[bool] = None) -> Optional[str]:
    """Wan2.1-VACE-1.3B-diffusers, downloaded on first use into
    models/radiance/ (7.7 GB: VAE, transformer, scheduler, tokenizer; the text
    encoder comes later, only for a custom prompt) unless downloads are off."""
    found = resolve_base(explicit)
    if found is not None:
        return found
    from radiance.model.rudra_download import WAN_BASE, download_package
    got = download_package(WAN_BASE, allow=allow)
    return str(got) if got is not None else None


def ensure_text_encoder(base) -> None:
    """The umT5 text encoder next to *base*: fetched (11.4 GB) the first time a
    prompt that is not cached has to be encoded."""
    if base is None:
        return
    folder = Path(str(base))
    if not folder.is_dir():
        return                                    # a hub id: transformers fetches it itself
    from radiance.model.rudra_download import WAN_TEXT_ENCODER, download_package, package_complete
    te = folder / "text_encoder"
    if package_complete(folder, WAN_TEXT_ENCODER) or (
            (te / "config.json").is_file() and any(te.glob("*.safetensors"))
            and not any(te.glob("*.part"))):
        return
    logger.info("[RUDRA AI fill] a new prompt needs the umT5 text encoder; fetching it once (11.4 GB).")
    if download_package(WAN_TEXT_ENCODER, target=folder) is None:
        raise RuntimeError(
            "The prompt is not cached and the umT5 text encoder is missing. Leave the prompt empty "
            f"(RUDRA's base prompt is cached), allow downloads, or put text_encoder/ from "
            f"https://huggingface.co/{BASE_REPO} in {folder}.")


_MODEL: Dict[str, object] = {}


def get_model(base: str, package: Path, device, keep_loaded: bool = True, lora: Optional[Path] = None) -> WanFill:
    """Wan VACE with a RUDRA LoRA: the v3 package's own, or *lora* (the v4 region re-render's).
    The prompt cache is always the v3 package's: an embedding is the base model's umT5 output for a
    text, the same whichever LoRA is loaded, so v4 reuses the cached base prompt too."""
    lora = Path(lora) if lora else Path(package) / "rudra_genfill.safetensors"
    key = f"{base}|{lora}|{device}"
    if key in _MODEL:
        return _MODEL[key]
    release_model()
    model = WanFill.load(base, lora, device=device,
                         prompt_cache=Path(package) / "prompt_cache", text_device="cpu")
    if keep_loaded:
        _MODEL[key] = model
    return model


def release_model() -> None:
    for m in list(_MODEL.values()):
        try:
            m.to("cpu")
        except Exception:  # noqa: BLE001
            pass
    _MODEL.clear()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
