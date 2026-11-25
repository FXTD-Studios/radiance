"""RUDRA's analytic SDR -> HDR master, ported from the RUDRA desktop app.

The reconstruction RUDRA runs with no model (the app's default route), stage
by stage as ``native/deliver/src/master.cpp grade_master`` runs it:

    1. baseline     the inverse of the curve that made the SDR (Source picker:
                    BT.1886, BT.1886 shadows, ACES, Hable, AgX, camera log,
                    plain clip), white at 203 nits
    2. clip fill    what goes where the SDR clipped: partly clipped channels
                    from the ones that survived, a smooth fill and a dome for
                    a large blown region, a peak and falloff for a lamp or the
                    sun, an artist's peak per region; and the near-clip
                    Highlight detail gain
    3. composite    the fill through the reconstruction gate (Strength)
    4. anchor       mid-tones held where the SDR had them, read through the
                    Source curve, let go above the knee
    5. chroma       the source's colour carried below the clip
    6. grain        flat near-white areas settled so lifted grain does not shimmer
    7. exposure     the master exposure, part of the grade, applied last
    8. scene linear diffuse white (203 nits) = 1.0

Sources (FXTD-Studios/RUDRA, Apache 2.0, commit 74f03b6, 10 Nov 2026):
rudra/source_baseline.py, native/core/src/{baseline,source_curve,clip_fill,
composite,master}.cpp, native/deliver/src/master.cpp. Copyright 2026 FXTD
Studios; see RUDRA's LICENSE and NOTICE. Tested against the native core's own
output (tests/fixtures/rudra_analytic, tests/test_rudra_analytic.py).

Pure numpy, scipy and OpenCV: one frame at a time, (H, W, 3) float SDR codes
in [0, 1] in, (H, W, 3) float32 scene-linear Rec.709 out. Two deliberate
differences from the native code, both inside the parity tolerance: distances
are Euclidean (scipy) where the app uses a (1, sqrt 2) chamfer, and the
per-pixel loops are vectorised.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Optional, Sequence

import numpy as np

__all__ = [
    "SOURCE_CURVES", "SOURCE_CURVE_LABELS", "DEFAULT_SOURCE_CURVE",
    "ClipPeak", "CalibrationPoint", "Region", "RegionBand", "ReferenceFit", "DEFAULT_REGION_BANDS",
    "AnalyticOptions", "AnalyticResult",
    "baseline_network", "clip_fill", "analytic_master", "parse_clip_peaks", "parse_calibration",
    "fit_reference", "apply_region_ev",
]

# ── constants ───────────────────────────────────────────────────────────────
DIFFUSE_WHITE_NITS = 203.0
LOG_SCALE = 16.0            # kAnalyticConstants: log_scale, max_hdr (4 = 40 000 nits), corpus EV 0
MAX_HDR = 4.0
CEILING_NITS = MAX_HDR * 10000.0
KNOTS = 256
_LUMA709 = np.array([0.2126, 0.7152, 0.0722], np.float32)
_LUMA2020 = np.array([0.2627, 0.6780, 0.0593], np.float64)
_TINY = np.float32(1e-8)

# The Source picker: (id, label). bt1886 is RUDRA's default since 13 Oct 2026.
SOURCE_CURVES = ("bt1886", "bt1886_shadows", "aces", "hable", "agx", "camera_log", "clip")
SOURCE_CURVE_LABELS = {
    "bt1886": "BT.1886 (graded / broadcast video)",
    "bt1886_shadows": "BT.1886 shadows (real HDR blacks)",
    "aces": "ACES (CG / games, ACES tone map)",
    "hable": "Filmic (Hable)",
    "agx": "AgX",
    "camera_log": "Camera log through a LUT",
    "clip": "Rec.709, plain clip",
}
DEFAULT_SOURCE_CURVE = "bt1886"


# ── 1. the baseline ─────────────────────────────────────────────────────────
def srgb_to_linear(x):
    x = np.clip(np.asarray(x, np.float32), 0.0, 1.0)
    return np.where(x <= 0.04045, x / np.float32(12.92),
                    np.power((x + np.float32(0.055)) / np.float32(1.055), np.float32(2.4))).astype(np.float32)


def inverse_aces_approx(y):
    """Invert y = x(Ax+B) / (x(Cx+D)+E); clipped values sit just below one (baseline.cpp)."""
    a, b, c, d, e = (np.float32(v) for v in (2.51, 0.03, 2.43, 0.59, 0.14))
    y = np.clip(np.asarray(y, np.float32), 0.0, 0.995)
    qa, qb, qc = y * c - a, y * d - b, y * e
    disc = np.maximum(qb * qb - np.float32(4.0) * qa * qc, 0.0)
    denom = np.minimum(np.float32(2.0) * qa, np.float32(-1e-7))
    s = np.sqrt(disc)
    return np.maximum(np.maximum((-qb - s) / denom, (-qb + s) / denom), 0.0).astype(np.float32)


def _bt709_oetf(x):
    x = np.asarray(x, np.float32)
    return np.where(x < 0.018, 4.5 * x, 1.099 * np.power(np.maximum(x, 0.0), 0.45) - 0.099).astype(np.float32)


def _forward_hable(x):
    a, b, c, d, e, f = (np.float32(v) for v in (0.15, 0.50, 0.10, 0.20, 0.02, 0.30))

    def h(v):
        return (v * (a * v + c * b) + d * e) / (v * (a * v + b) + d * f) - e / f
    return np.clip(h(x * np.float32(2.0)) / h(np.float32(11.2)), 0.0, 1.0).astype(np.float32)


def _forward_agx(x):
    lo, hi = np.float32(-12.47), np.float32(4.03)
    t = np.clip((np.log2(np.maximum(x, np.exp2(lo))) - lo) / (hi - lo), 0.0, 1.0)
    s = 1.0 / (1.0 + np.exp(-(t - 0.6) * 10.0))
    s0 = 1.0 / (1.0 + np.exp(np.float32(0.6 * 10.0)))
    s1 = 1.0 / (1.0 + np.exp(np.float32(-0.4 * 10.0)))
    return np.power(np.clip((s - s0) / (s1 - s0), 0.0, 1.0), 2.2).astype(np.float32)


def _forward_camera_log(x):
    lg = np.clip(np.log2(np.maximum(x, np.float32(1e-6)) / np.float32(0.18)) / np.float32(14.0) + np.float32(0.5), 0.0, 1.0)
    k = np.float32(7.0)
    s = 1.0 / (1.0 + np.exp(-(lg - 0.5) * k))
    s0 = 1.0 / (1.0 + np.exp(0.5 * k))
    s1 = 1.0 / (1.0 + np.exp(-0.5 * k))
    return np.power(np.clip((s - s0) / (s1 - s0), 0.0, 1.0), 2.4).astype(np.float32)


def _table_inverse(forward, display):
    """InverseTable: 8001 log-spaced exposures over 22 stops, bisection and lerp."""
    x = np.exp2(np.linspace(-16.0, 6.0, 8001)).astype(np.float32)
    y = np.maximum.accumulate(forward(x).astype(np.float32))
    v = np.clip(display, 0.0, 0.995).astype(np.float32)
    i = np.searchsorted(y, v, side="left")
    out = np.zeros_like(v)
    hi = i >= len(y)
    mid = (i > 0) & ~hi
    out[hi] = x[-1]
    ii = i[mid]
    y0, y1 = y[ii - 1], y[ii]
    t = np.where(y1 > y0, (v[mid] - y0) / np.where(y1 > y0, y1 - y0, 1.0), 0.0)
    out[mid] = x[ii - 1] + (x[ii] - x[ii - 1]) * t
    return out


_SHADOW_KNEE = 40
_SHADOW_LOG2_SCENE = np.array([0.0,
    -14.685428, -14.249545, -14.009058, -13.741516, -13.537101, -13.242504,
    -13.008029, -12.860731, -12.457915, -12.053934, -12.033934, -12.013934,
    -11.993934, -11.973934, -11.252472, -10.590089, -10.570089, -10.389724,
    -10.164267, -10.013963, -9.968597, -9.929899, -9.870572, -9.778883,
    -9.693991, -9.599686, -9.497595, -9.392905, -9.292866, -9.197775,
    -9.111351, -9.028863, -8.925111, -8.815446, -8.703925, -8.587965,
    -8.450235, -8.357113, -8.253048,
], np.float32)


def _bt709_inverse_oetf(v):
    """The light a BT.709 video code stands for (source_curve.cpp bt709_inverse_oetf)."""
    v = np.clip(np.asarray(v, np.float32), 0.0, 1.0)
    return np.where(v < 0.081, v / 4.5, np.power((v + 0.099) / 1.099, 1.0 / 0.45)).astype(np.float32)


# The curves that read a source's own codes as the BT.709 video codes they are (RUDRA 1.3, 24 Nov 2026).
VIDEO_CODE_CURVES = ("bt1886", "bt1886_shadows")


def _inverse_bt1886(display):
    return np.power(_bt709_oetf(np.clip(display, 0.0, 0.995)), 2.4).astype(np.float32)


def _inverse_bt1886_shadows(display):
    display = np.asarray(display, np.float32)
    knee_display = np.float32(((_SHADOW_KNEE / 255.0 + 0.055) / 1.055) ** 2.4)
    x = np.clip(np.minimum(display, np.float32(0.995)), 0.0, 1.0)
    code = np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1.0 / 2.4) - 0.055).astype(np.float32) * 255.0
    code = np.minimum(code, _SHADOW_KNEE)
    knee = np.log2(_inverse_bt1886(np.asarray([knee_display]))[0])
    table = np.append(_SHADOW_LOG2_SCENE, knee)
    i = np.clip(np.floor(code).astype(int), 1, _SHADOW_KNEE - 1)
    t = code - i
    lifted = np.exp2(table[i] + (table[i + 1] - table[i]) * t)
    lifted = np.where(code < 1.0, code * np.exp2(table[1]), lifted)
    lifted = np.where(code <= 0.0, 0.0, lifted)
    return np.where(display >= knee_display, _inverse_bt1886(display), lifted).astype(np.float32)


def _inverse_curve(curve: str, display):
    if curve == "aces":
        return inverse_aces_approx(display)
    if curve == "bt1886":
        return _inverse_bt1886(display)
    if curve == "bt1886_shadows":
        return _inverse_bt1886_shadows(display)
    if curve == "clip":
        return np.clip(display, 0.0, 0.995).astype(np.float32)
    forward = {"hable": _forward_hable, "agx": _forward_agx, "camera_log": _forward_camera_log}[curve]
    return _table_inverse(forward, display)


@lru_cache(maxsize=None)
def curve_knots(curve: str, own_codes: bool = False) -> np.ndarray:
    """(256,) float32 per-code log2 correction of the curve over the ACES inverse (source_baseline.py).

    ``own_codes`` (RUDRA 1.3, source_curve.cpp, 24 Nov 2026): the SDR holds the source's own code values
    (a still, or a video frame as ComfyUI decodes it), not BT.709 light re-encoded as sRGB (the RUDRA
    command line's video path, which the golden fixtures come from). BT.1886 and BT.1886 shadows then read
    each code as the BT.709 video code it is, so BT.1886 gives code^2.4. Without it an own code went
    through the sRGB decode and then the BT.709 OETF: 0.36 stop dark at code 128, 1.3 at 51, 2.8 at 26.
    The other curves are the same either way.
    """
    if curve not in SOURCE_CURVES:
        raise ValueError(f"unknown source curve {curve!r}; choose one of {SOURCE_CURVES}")
    if curve == "aces":
        out = np.zeros(KNOTS, np.float32)
    else:
        codes = np.arange(KNOTS, dtype=np.float32) / np.float32(255.0)
        display = srgb_to_linear(codes)
        read = _bt709_inverse_oetf(codes) if (own_codes and curve in VIDEO_CODE_CURVES) else display
        want, have = _inverse_curve(curve, read), inverse_aces_approx(display)
        ok = (want > 0) & (have > 0)
        out = np.where(ok, np.log2(np.where(ok, want, 1.0) / np.where(ok, have, 1.0)), 0.0).astype(np.float32)
        out[0] = out[1]
    out.setflags(write=False)
    return out


def _curve_correction_log2(code, knots):
    pos = np.clip(code, 0.0, 1.0).astype(np.float32) * np.float32(KNOTS - 1)
    lo = np.minimum(np.floor(pos).astype(np.int64), KNOTS - 2)
    frac = pos - lo.astype(np.float32)
    v0, v1 = knots[lo], knots[lo + 1]
    return (v0 + (v1 - v0) * frac).astype(np.float32)


_BASE_SCALE = np.float32(203.0 / 10000.0)   # corpus EV 0: 2^0 x 203 / 10 000


def _baseline_exact(code):
    return (inverse_aces_approx(srgb_to_linear(code)) * _BASE_SCALE).astype(np.float32)


_BASELINE_LUT = _baseline_exact(np.arange(KNOTS, dtype=np.float32) / np.float32(255.0))


def _as_codes(code):
    """Integer 8-bit codes when every value is one (k / 255), else None: a 256-entry table is then exact."""
    c = np.asarray(code, np.float32)
    k = np.rint(np.clip(c, 0.0, 1.0) * np.float32(255.0))
    return k.astype(np.int64) if np.array_equal(k / np.float32(255.0), np.clip(c, 0.0, 1.0)) else None


def baseline_value(code):
    """The ACES-inverse baseline (no curve), network units (1.0 = 10 000 nits)."""
    k = _as_codes(code) if np.ndim(code) else None
    if k is not None:
        return _BASELINE_LUT[k]
    return _baseline_exact(code)


def effective_knots(curve: str, extra: Optional[np.ndarray] = None, own_codes: bool = False) -> Optional[np.ndarray]:
    """The picked curve's knots plus a calibration or reference correction (composite.cpp baseline_curve_params)."""
    if extra is None:
        return None if curve == "aces" else curve_knots(curve, own_codes)
    return (curve_knots(curve, own_codes) + np.asarray(extra, np.float32)).astype(np.float32)


def baseline_network(sdr: np.ndarray, curve: str = DEFAULT_SOURCE_CURVE,
                     extra: Optional[np.ndarray] = None, own_codes: bool = False) -> np.ndarray:
    """The corrected baseline: the inverse of the picked curve (and its calibration), network units."""
    c = np.clip(np.asarray(sdr, np.float32), 0.0, 1.0)
    v = baseline_value(c)
    knots = effective_knots(curve, extra, own_codes)
    if knots is not None:
        v = (v * np.exp2(_curve_correction_log2(c, knots))).astype(np.float32)
    return v


# ── calibration and reference (calibration.cpp, reference_fit.cpp) ─────────
@dataclass
class CalibrationPoint:
    """An SDR code (0..255) and what it should be, in nits."""
    code: int
    nits: float


def source_code_nits(curve: str, code) -> np.ndarray:
    """Nits of SDR codes under the source curve's inverse (diffuse white 203 nits)."""
    display = srgb_to_linear(np.asarray(code, np.float32) / np.float32(255.0))
    return _inverse_curve(curve, display).astype(np.float64) * DIFFUSE_WHITE_NITS


def _anchors(points: Sequence[CalibrationPoint]):
    out = {}
    for p in points:
        if 0 <= int(p.code) < KNOTS and p.nits > 0 and math.isfinite(p.nits):
            out[int(p.code)] = float(p.nits)
    return sorted(out.items())


def calibration_knots(points: Sequence[CalibrationPoint], curve: str) -> Optional[np.ndarray]:
    """(256,) log2 correction through the anchors, piecewise-linear in code, flat outside; None for none."""
    pts = _anchors(points)
    if not pts:
        return None
    codes = np.array([c for c, _ in pts])
    have = source_code_nits(curve, codes)
    d = np.where(have > 0, np.log2(np.array([n for _, n in pts]) / np.maximum(have, 1e-300)), 0.0)
    return np.interp(np.arange(KNOTS), codes, d).astype(np.float32)


def calibration_is_monotone(points: Sequence[CalibrationPoint], curve: str) -> bool:
    k = calibration_knots(points, curve)
    if k is None:
        return True
    n = source_code_nits(curve, np.arange(1, KNOTS)) * np.exp2(k[1:].astype(np.float64))
    return bool(np.all(np.diff(n) >= 0))


@dataclass
class ReferenceFit:
    target_log2_nits: np.ndarray        # (256,)
    samples_per_code: np.ndarray        # (256,) int
    residual_mean: float = 0.0
    residual_p95: float = 0.0
    samples: int = 0
    codes_seen: int = 0


_REF_MIN_SAMPLES, _REF_SMOOTH = 16, 3


def _isotonic(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    val, wt, ln = [], [], []
    for yi, wi in zip(y.tolist(), w.tolist()):
        val.append(yi); wt.append(wi); ln.append(1)
        while len(val) > 1 and val[-2] > val[-1]:
            tw = wt[-2] + wt[-1]
            tv = (val[-2] * wt[-2] + val[-1] * wt[-1]) / tw
            val.pop(); wt.pop(); l = ln.pop()
            val[-1] = tv; wt[-1] = tw; ln[-1] += l
    return np.repeat(np.array(val), ln)


def fit_reference(sdr: np.ndarray, reference: np.ndarray) -> ReferenceFit:
    """The baseline fitted to a graded HDR of the same frame (scene-linear Rec.709, 1.0 = 203 nits)."""
    sdr = np.asarray(sdr, np.float32)[..., :3]
    ref = np.asarray(reference, np.float32)[..., :3]
    if ref.shape != sdr.shape:
        raise ValueError(f"the reference is {ref.shape[1]}x{ref.shape[0]}, the frame {sdr.shape[1]}x{sdr.shape[0]}: "
                         "they must be the same size")
    code = np.clip(np.rint(sdr.astype(np.float64) * 255.0), 0, KNOTS - 1).astype(np.int64).ravel()
    v = ref.ravel()
    ok = (v > 0) & np.isfinite(v)
    code, lv = code[ok], np.log2(v[ok].astype(np.float64) * DIFFUSE_WHITE_NITS).astype(np.float32)
    order = np.lexsort((lv, code))
    code, lv = code[order], lv[order]
    counts = np.bincount(code, minlength=KNOTS)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    med = np.full(KNOTS, np.nan)
    seen = counts >= _REF_MIN_SAMPLES
    med[seen] = lv[starts[seen] + counts[seen] // 2]
    if not seen.any():
        raise ValueError(f"the reference gives no SDR code {_REF_MIN_SAMPLES} samples")
    idx = np.nonzero(seen)[0]
    t = np.interp(np.arange(KNOTS), idx, med[idx])
    w = np.arange(-_REF_SMOOTH, _REF_SMOOTH + 1)
    tri = (_REF_SMOOTH + 1 - np.abs(w)).astype(np.float64)
    sm = np.array([(t[np.clip(c + w, 0, KNOTS - 1)] * tri).sum() / tri.sum() for c in range(KNOTS)])
    spc = np.where(seen, counts, 0)
    target = _isotonic(sm, np.maximum(1.0, spc.astype(np.float64))).astype(np.float32)
    inner = (code >= 1) & (code <= KNOTS - 2)
    err = np.abs(lv[inner] - target[code[inner]])
    fit = ReferenceFit(target, spc.astype(np.int64), samples=int(err.size), codes_seen=int(seen.sum()))
    if err.size:
        fit.residual_mean = float(err.astype(np.float64).mean())
        fit.residual_p95 = float(np.partition(err, min(err.size - 1, int(err.size * 0.95)))[min(err.size - 1, int(err.size * 0.95))])
    return fit


def reference_knots(fit: ReferenceFit, curve: str) -> np.ndarray:
    have = np.maximum(source_code_nits(curve, np.arange(KNOTS)), 1e-9)
    return (fit.target_log2_nits.astype(np.float64) - np.log2(have)).astype(np.float32)


def reference_exposure(fit: ReferenceFit, curve: str) -> float:
    p = reference_knots(fit, curve).astype(np.float64)
    w = fit.samples_per_code.astype(np.float64)
    sel = w > 0
    if not sel.any():
        return 0.0
    order = np.argsort(p[sel], kind="stable")
    pv, wv = p[sel][order], w[sel][order]
    return float(pv[np.searchsorted(np.cumsum(wv), wv.sum() * 0.5)])


# ── 2. the clip fill (clip_fill.cpp) ────────────────────────────────────────
@dataclass
class ClipPeak:
    """The artist's peak for the clipped region at a point (x, y across the frame, 0..1)."""
    x: float
    y: float
    nits: float


@dataclass
class ClipRegion:
    pixels: int = 0
    fraction: float = 0.0
    radius_px: float = 0.0
    large: bool = False
    edge_nits: float = 0.0
    peak_nits: float = 0.0
    glow_ev_per_px: float = 0.0
    overridden: bool = False
    x: int = 0
    y: int = 0


@dataclass
class ClipFillResult:
    gain_log2: Optional[np.ndarray]     # (H, W, 3) or None when nothing clipped
    regions: List[ClipRegion] = field(default_factory=list)
    partial_pct: float = 0.0
    full_pct: float = 0.0
    detail_pct: float = 0.0


_SOFT_CODE, _CLIP_CODE = 0.97, 0.995
_LARGE_FRACTION, _LARGE_MAX_EV = 0.005, 1.5
_SOURCE_MIN_EV, _SOURCE_MAX_NITS = 0.0, 4000.0
_MAX_PARTIAL_EV, _GLOW_SCALE = 4.0, 0.5
_MARGIN = 40


def _smooth01(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _sigmoid(x):
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def _clip_weight(code):
    code = np.clip(code, 0.0, 1.0).astype(np.float32)
    w = _smooth01((code - np.float32(_SOFT_CODE)) / np.float32(_CLIP_CODE - _SOFT_CODE)).astype(np.float32)
    return np.where(code > _SOFT_CODE, w, np.float32(0.0)).astype(np.float32)


def _luma(v):
    return (v[..., 0] * _LUMA709[0] + v[..., 1] * _LUMA709[1] + v[..., 2] * _LUMA709[2]).astype(np.float32)


def _distance(seeds: np.ndarray, want_labels=None):
    """Distance to the nearest seed pixel (and its label), Euclidean. inf where there is no seed."""
    from scipy import ndimage
    if not seeds.any():
        d = np.full(seeds.shape, np.inf, np.float32)
        return (d, np.full(seeds.shape, -1, np.int64)) if want_labels is not None else d
    if want_labels is None:
        return ndimage.distance_transform_edt(~seeds).astype(np.float32)
    d, idx = ndimage.distance_transform_edt(~seeds, return_indices=True)
    return d.astype(np.float32), want_labels[idx[0], idx[1]]


def _upsample2(cv: np.ndarray, fh: int, fw: int) -> np.ndarray:
    """Bilinear x2 on pixel centres (fill_missing's pull), indices clamped as the native code clamps them."""
    _, chh, cww = cv.shape
    u = (np.arange(fw) + 0.5) * 0.5 - 0.5
    vv = (np.arange(fh) + 0.5) * 0.5 - 0.5
    x0 = np.clip(np.floor(u).astype(int), 0, cww - 1); y0 = np.clip(np.floor(vv).astype(int), 0, chh - 1)
    x1 = np.minimum(x0 + 1, cww - 1); y1 = np.minimum(y0 + 1, chh - 1)
    fx = np.clip(u - np.floor(u), 0, 1).astype(np.float32); fy = np.clip(vv - np.floor(vv), 0, 1).astype(np.float32)
    rows = cv[:, :, x0] * (1 - fx) + cv[:, :, x1] * fx
    return (rows[:, y0] * (1 - fy)[:, None] + rows[:, y1] * fy[:, None]).astype(np.float32)


def fill_missing(planes: np.ndarray, weight: np.ndarray, relax_iterations: int = 2) -> np.ndarray:
    """Push-pull smooth fill (clip_fill.cpp fill_missing): (K, H, W) planes, known where weight > 0."""
    planes = planes.astype(np.float32).copy()
    k, h, w = planes.shape
    known = weight > 0
    if known.all():
        return planes
    if not known.any():
        return np.zeros_like(planes)

    def push(v, wt):
        fh, fw = wt.shape
        ch, cw = (fh + 1) // 2, (fw + 1) // 2
        pv = np.zeros((v.shape[0], ch * 2, cw * 2), np.float64)
        pw = np.zeros((ch * 2, cw * 2), np.float64)
        pv[:, :fh, :fw] = v
        pw[:fh, :fw] = np.clip(wt, 0.0, None)
        pw[pw <= 0] = 0.0
        W = pw.reshape(ch, 2, cw, 2).sum((1, 3))
        acc = (pv * pw).reshape(v.shape[0], ch, 2, cw, 2).sum((2, 4))
        cv = np.where(W > 0, acc / np.where(W > 0, W, 1.0), 0.0).astype(np.float32)
        return cv, np.minimum(W, 1.0).astype(np.float32)

    levels = [push(planes, np.clip(weight, 0.0, 1.0).astype(np.float32))]
    while levels[-1][1].shape[0] > 1 or levels[-1][1].shape[1] > 1:
        cv, cw_ = levels[-1]
        if (cw_ >= 1.0).all():
            break
        levels.append(push(cv, cw_))

    def relax(v, miss, iterations):
        if not miss.any():
            return v
        hh, ww = miss.shape
        cnt = np.zeros((hh, ww), np.float32)
        cnt[:, 1:] += 1; cnt[:, :-1] += 1; cnt[1:, :] += 1; cnt[:-1, :] += 1
        for _ in range(iterations):
            s = np.zeros_like(v)
            s[:, :, 1:] += v[:, :, :-1]; s[:, :, :-1] += v[:, :, 1:]
            s[:, 1:, :] += v[:, :-1, :]; s[:, :-1, :] += v[:, 1:, :]
            nxt = np.where(cnt > 0, s / np.maximum(cnt, 1), v)
            v = np.where(miss[None], nxt, v).astype(np.float32)
        return v

    def pull(v, wt, coarse):
        cv, _ = coarse
        fh, fw = wt.shape
        chh, cww = cv.shape[1:]
        w0 = np.clip(wt, 0.0, 1.0)
        up = _upsample2(cv, fh, fw)
        blend = np.where(w0 >= 1.0, v, w0 * v + (1.0 - w0) * up).astype(np.float32)
        return blend, w0 <= 0.0

    for li in range(len(levels) - 2, -1, -1):
        fv, fw_ = levels[li]
        fv, miss = pull(fv, fw_, levels[li + 1])
        fv = relax(fv, miss, 4)
        levels[li] = (fv, np.ones_like(fw_))
    planes, miss = pull(planes, weight.astype(np.float32), levels[0])
    return relax(planes, miss, relax_iterations)


def _reconstruct_crop(sdr, peaks, frame):
    from scipy import ndimage
    h, w, _ = sdr.shape
    n = h * w
    total = frame["pixels"]
    wc = _clip_weight(sdr)                                   # (h, w, 3)
    touched_any = (wc > 0).any(-1)
    work = _distance(touched_any) <= _MARGIN
    b = np.where(work[..., None], baseline_value(np.clip(sdr, 0, 1)), np.float32(0.0)).astype(np.float32)
    touched = (wc > 0).sum(-1)
    gone = (wc >= 1.0).sum(-1)
    CLEAN, PARTIAL, FULL = 0, 1, 2
    cls = np.where(touched == 0, CLEAN, np.where(gone == 3, FULL, PARTIAL))
    n_partial, n_full = int((cls == PARTIAL).sum()), int((cls == FULL).sum())
    res = ClipFillResult(None, [], 100.0 * n_partial / total, 100.0 * n_full / total)
    t = b.copy()

    # Stage 1: partly clipped channels from the ones that survived.
    if n_partial > 0:
        clean = (cls == CLEAN) & work
        ly = np.log2(np.maximum(_luma(b), _TINY))
        q = np.where(clean[None], (np.log2(np.maximum(b, _TINY)) - ly[..., None]).transpose(2, 0, 1), 0.0).astype(np.float32)
        q = fill_missing(q, clean.astype(np.float32)).transpose(1, 2, 0)
        part = cls == PARTIAL
        kept = (1.0 - wc.astype(np.float64))
        lb = np.log2(np.maximum(b, _TINY)).astype(np.float64)
        est = (kept * (lb - q)).sum(-1) / np.maximum(kept.sum(-1), 1e-12)
        lift = np.clip(est[..., None] + q - lb, 0.0, _MAX_PARTIAL_EV) * wc
        upd = (b.astype(np.float64) * np.exp2(lift)).astype(np.float32)
        sel = part[..., None] & (wc != 0)
        t = np.where(sel, upd, t)

    # Stages 2 and 3: fully clipped regions.
    if n_full > 0:
        known = (cls != FULL) & work
        f = np.zeros((4, h, w), np.float32)
        lyt = np.log2(np.maximum(_luma(t), _TINY))
        f[0] = np.where(known, lyt, 0.0)
        f[1:] = np.where(known[None], (np.log2(np.maximum(t, _TINY)) - lyt[..., None]).transpose(2, 0, 1), 0.0)
        any_known = n_full < n
        ceil_ly = math.log2(float(baseline_value(np.float32(1.0))))
        if any_known:
            f = fill_missing(f, known.astype(np.float32))
        else:
            f[0] = ceil_ly
        full = cls == FULL
        label, regions = ndimage.label(full, structure=np.ones((3, 3), int))
        label = label.astype(np.int64) - 1                     # -1 outside, 0.. regions
        d_in = _distance(~full) if any_known else np.full((h, w), np.inf, np.float32)
        d_out, near = _distance(full, want_labels=label)
        d_in_eff = d_in if any_known else np.full((h, w), float(max(h, w)), np.float32)
        reg = [ClipRegion() for _ in range(regions)]
        lab_flat = label.ravel()
        inside = lab_flat >= 0
        idx = np.nonzero(inside)[0]
        labs = lab_flat[idx]
        counts = np.bincount(labs, minlength=regions)
        dv = d_in_eff.ravel()[idx].astype(np.float64)
        # deepest point: the first (raster) pixel at the region's largest distance
        order = np.lexsort((idx, -dv, labs))
        first = np.ones(len(order), bool)
        first[1:] = labs[order][1:] != labs[order][:-1]
        deepest = idx[order[first]]
        radius = np.zeros(regions)
        radius[labs[order[first]]] = dv[order[first]]
        for r in range(regions):
            reg[r].pixels = int(counts[r])
        for r, p in zip(labs[order[first]], deepest):
            reg[r].x, reg[r].y = int(p % w), int(p // w)
        # the ring outside each region: its edge level and the glow's slope
        nf = near.ravel()
        do = d_out.ravel().astype(np.float64)
        ring_ok = (cls.ravel() != FULL) & (nf >= 0)
        span = np.clip(radius, 3.0, 32.0)
        ring_ok &= do <= np.where(nf >= 0, span[np.maximum(nf, 0)], 0)
        ri = nf[ring_ok]; rd = do[ring_ok]; rl = f[0].ravel()[ring_ok].astype(np.float64)
        bc = lambda x: np.bincount(ri, weights=x, minlength=regions)
        gn, gsx, gsy, gsxx, gsxy = bc(np.ones_like(rd)), bc(rd), bc(rl), bc(rd * rd), bc(rd * rl)
        edge = rd <= 1.5
        en = np.bincount(ri[edge], minlength=regions).astype(np.float64)
        es = np.bincount(ri[edge], weights=rl[edge], minlength=regions)
        dome = np.zeros(regions)
        centre_nits = np.zeros(regions)
        bl = np.log2(np.maximum(_luma(b), _TINY))
        for r in range(regions):
            R = reg[r]
            R.radius_px = float(radius[r])
            R.fraction = R.pixels / total
            R.large = R.fraction >= _LARGE_FRACTION
            el = es[r] / en[r] if en[r] > 0 else float(f[0][R.y, R.x])
            el = max(el, ceil_ly)
            R.edge_nits = 2.0 ** el * 10000.0
            centre_nits[r] = 2.0 ** max(float(f[0][R.y, R.x]), float(bl[R.y, R.x])) * 10000.0
            var = gn[r] * gsxx[r] - gsx[r] * gsx[r]
            slope = (gn[r] * gsxy[r] - gsx[r] * gsy[r]) / var if gn[r] >= 3 and var > 1e-9 else 0.0
            R.glow_ev_per_px = max(0.0, -slope)
            rad = max(R.radius_px, 1.0)
            a = _GLOW_SCALE * R.glow_ev_per_px * rad * 0.5
            if R.large:
                a = min(max(a, 0.0), _LARGE_MAX_EV)
            else:
                most = max(math.log2(_SOURCE_MAX_NITS / max(centre_nits[r], 1e-6)), 0.0)
                a = min(max(a, _SOURCE_MIN_EV), most)
            dome[r] = a
        for o in peaks:
            if not (o.nits > 0):
                continue
            x = int(o.x * frame["width"]) - frame["x0"]
            y = int(o.y * frame["height"]) - frame["y0"]
            if x < 0 or y < 0 or x >= w or y >= h:
                continue
            if near[y, x] < 0 or d_out[y, x] > 4.0:
                continue
            r = int(near[y, x])
            dome[r] = max(0.0, math.log2(o.nits / max(centre_nits[r], 1e-6)))
            reg[r].overridden = True
        rad_pix = np.maximum(radius[np.maximum(label, 0)], 1.0)
        u = np.clip((d_in.astype(np.float64) if any_known else rad_pix) / rad_pix, 0.0, 1.0)
        ly = np.maximum(f[0].astype(np.float64), bl.astype(np.float64)) + dome[np.maximum(label, 0)] * (2.0 * u - u * u)
        keep = 1.0 - 0.5 * u
        cq = keep[..., None] * f[1:].transpose(1, 2, 0).astype(np.float64)
        norm = np.log2(np.maximum((_LUMA709.astype(np.float64) * np.exp2(cq)).sum(-1), 1e-12))
        lbf = np.log2(np.maximum(b, _TINY)).astype(np.float64)
        lt = np.maximum(ly[..., None] + cq - norm[..., None], lbf)
        filled = (b.astype(np.float64) * np.exp2(lt - lbf)).astype(np.float32)
        t = np.where(full[..., None], filled, t)
        for R in reg:
            R.peak_nits = float(_luma(t[R.y, R.x][None])[0]) * 10000.0
        res.regions = reg

    gain = np.zeros((h, w, 3), np.float32)
    up = work[..., None] & (t > b)
    gain[up] = (np.log2(np.maximum(t, _TINY)) - np.log2(np.maximum(b, _TINY)))[up]
    res.gain_log2 = gain
    return res


def reconstruct_clipped(sdr: np.ndarray, peaks: Sequence[ClipPeak] = ()) -> ClipFillResult:
    h, w, _ = sdr.shape
    wc = _clip_weight(sdr)
    hit = (wc > 0).any(-1)
    if not hit.any():
        return ClipFillResult(None)
    ys, xs = np.nonzero(hit)
    y0, y1 = max(0, ys.min() - _MARGIN), min(h - 1, ys.max() + _MARGIN)
    x0, x1 = max(0, xs.min() - _MARGIN), min(w - 1, xs.max() + _MARGIN)
    frame = {"pixels": h * w, "x0": int(x0), "y0": int(y0), "width": w, "height": h}
    part = _reconstruct_crop(sdr[y0:y1 + 1, x0:x1 + 1], peaks, frame)
    for r in part.regions:
        r.x += int(x0); r.y += int(y0)
    gain = np.zeros((h, w, 3), np.float32)
    gain[y0:y1 + 1, x0:x1 + 1] = part.gain_log2
    part.gain_log2 = gain
    return part


def _gaussian_clamp(a, sigma):
    """Separable Gaussian, edges clamped, radius ceil(3 sigma) (clip_fill.cpp gaussian_blur)."""
    r = max(1, int(math.ceil(3.0 * sigma)))
    i = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-0.5 * i * i / (sigma * sigma)).astype(np.float32)
    k = (k.astype(np.float64) / k.astype(np.float64).sum()).astype(np.float32)
    from scipy import ndimage
    tmp = ndimage.convolve1d(a.astype(np.float64), k.astype(np.float64), axis=1, mode="nearest").astype(np.float32)
    return ndimage.convolve1d(tmp.astype(np.float64), k.astype(np.float64), axis=0, mode="nearest").astype(np.float32)


def near_clip_detail(sdr: np.ndarray, gain: float, sigma: float = 6.0, frm: float = 0.86, to: float = 0.94,
                     floor_stops: float = 0.015, max_stops: float = 1.0) -> Optional[np.ndarray]:
    """Stage 4, Highlight detail: signed log2 gain, (H, W); None at gain 1 (clip_fill.cpp near_clip_detail)."""
    if not (gain > 0.0) or gain == 1.0:
        return None
    v = np.clip(_luma(np.asarray(sdr, np.float32)), 0.0, 1.0)
    to = max(to, frm + 1e-4)
    wgt = _smooth01((v - frm) / (to - frm)).astype(np.float32)
    lum = (np.float32(2.4) * np.log2(np.maximum(v, np.float32(1.0 / 4096.0)))).astype(np.float32)
    low = _gaussian_clamp(wgt * lum, sigma)
    den = _gaussian_clamp(wgt, sigma)
    hp = lum - low / np.maximum(den, 1e-12)
    mag = np.maximum(np.abs(hp) - np.float32(max(floor_stops, 0.0)), 0.0)
    g = np.clip(wgt * np.float32(gain - 1.0) * np.copysign(mag, hp), -max_stops, max_stops).astype(np.float32)
    ok = (wgt > 0) & (den > 1e-3) & (mag > 0)
    return np.where(ok, g, np.float32(0.0)).astype(np.float32)


@dataclass
class _Fields:
    residual: np.ndarray        # (H, W, 3)
    highlight: np.ndarray       # (H, W)


def clip_fill(sdr: np.ndarray, peaks: Sequence[ClipPeak] = (), detail_gain: float = 1.0,
              enabled: bool = True):
    """The fill as composite fields (clip_fill_fields): a residual and a highlight mask. Returns (fields, info)."""
    h, w, _ = sdr.shape
    fields = _Fields(np.zeros((h, w, 3), np.float32), np.zeros((h, w), np.float32))
    if not enabled:
        return fields, ClipFillResult(None)
    info = reconstruct_clipped(sdr, peaks)
    det = near_clip_detail(sdr, detail_gain)
    if info.gain_log2 is None and det is None:
        return fields, info
    code = np.clip(sdr, 0.0, 1.0).astype(np.float32)
    gain = (info.gain_log2 if info.gain_log2 is not None else np.zeros((h, w, 3), np.float32))
    if det is not None:
        gain = gain + det[..., None]
        info.detail_pct = 100.0 * float((det != 0).sum()) / (h * w)
    changed = (gain != 0).any(-1)
    y = _luma(code)
    gate = np.maximum.reduce([_sigmoid((y - np.float32(0.82)) * np.float32(24.0)),
                              _sigmoid((np.float32(0.10) - y) * np.float32(24.0)),
                              np.full_like(y, 1e-6)])
    base = baseline_value(code)
    target = base * np.exp2(gain)
    resid = (np.log1p(target * np.float32(LOG_SCALE)) - np.log1p(base * np.float32(LOG_SCALE))) / gate[..., None]
    fields.residual = np.where((gain != 0) & changed[..., None], resid, np.float32(0.0)).astype(np.float32)
    fields.highlight = changed.astype(np.float32)
    return fields, info


# ── 3. composite (composite.cpp, the analytic route: no learned fields) ─────
@dataclass
class Region:
    """Where the reconstruction may act, by the SDR's Rec.709 luma (RUDRA 1.1 'Limit to a luminance region')."""
    band: bool = False                  # one luma band instead of highlights + shadows
    highlight_from: float = 0.75
    highlight_softness: float = 0.10
    highlight_strength: float = 1.0
    shadow_below: float = 0.15
    shadow_softness: float = 0.10
    shadow_strength: float = 1.0
    band_low: float = 0.50
    band_high: float = 0.90
    band_softness: float = 0.05

    def error(self) -> str:
        v = [self.highlight_from, self.highlight_softness, self.highlight_strength, self.shadow_below,
             self.shadow_softness, self.shadow_strength, self.band_low, self.band_high, self.band_softness]
        if not all(math.isfinite(x) for x in v):
            return "every region value must be finite"
        if any(x < 0 or x > 1 for x in (self.highlight_from, self.shadow_below, self.band_low, self.band_high)):
            return "region thresholds are SDR luma, 0 to 1"
        if any(x < 0 or x > 0.5 for x in (self.highlight_softness, self.shadow_softness, self.band_softness)):
            return "region softness is 0 to 0.5"
        if any(x < 0 or x > 2 for x in (self.highlight_strength, self.shadow_strength)):
            return "region strength is 0 to 2"
        if not self.band_low < self.band_high:
            return "the band's low must be below its high"
        return ""


def _ramp_up(y, at, soft):
    if not soft > 0:
        return (y >= np.float32(at)).astype(np.float32)
    t = np.clip((y - (np.float32(at) - np.float32(0.5) * np.float32(soft))) / np.float32(soft), 0.0, 1.0)
    return (t * t * (np.float32(3.0) - np.float32(2.0) * t)).astype(np.float32)


def _ramp_down(y, at, soft):
    if not soft > 0:
        return (y <= np.float32(at)).astype(np.float32)
    t = np.clip(((np.float32(at) + np.float32(0.5) * np.float32(soft)) - y) / np.float32(soft), 0.0, 1.0)
    return (t * t * (np.float32(3.0) - np.float32(2.0) * t)).astype(np.float32)


def region_weights(y, r: Optional[Region]):
    """The region's highlight and shadow weights at SDR luma y (both 1 with no region)."""
    if r is None:
        one = np.ones_like(y, np.float32)
        return one, one
    if r.band:
        b = _ramp_up(y, r.band_low, r.band_softness) * _ramp_down(y, r.band_high, r.band_softness)
        return (b * np.float32(r.highlight_strength)).astype(np.float32), (b * np.float32(r.shadow_strength)).astype(np.float32)
    return ((_ramp_up(y, r.highlight_from, r.highlight_softness) * np.float32(r.highlight_strength)).astype(np.float32),
            (_ramp_down(y, r.shadow_below, r.shadow_softness) * np.float32(r.shadow_strength)).astype(np.float32))


def composite(sdr, fields: _Fields, curve: str, strength: float = 1.0, mode: str = "all",
              region: Optional[Region] = None, extra_knots: Optional[np.ndarray] = None,
              limit: Optional[np.ndarray] = None, own_codes: bool = False) -> np.ndarray:
    """``limit`` (H, W) in [0, 1]: where the reconstruction may act; 0 is the baseline, bit for bit."""
    c = np.clip(np.asarray(sdr, np.float32), 0.0, 1.0)
    base = baseline_network(c, curve, extra_knots, own_codes)
    y = _luma(c)
    hp = _sigmoid((y - np.float32(0.82)) * np.float32(24.0))
    sp = _sigmoid((np.float32(0.10) - y) * np.float32(24.0))
    if region is not None:
        wh, ws = region_weights(y, region)
        hp, sp = (hp * wh).astype(np.float32), (sp * ws).astype(np.float32)
    gate = {"all": np.maximum(hp, sp), "highlights": hp, "shadows": sp, "off": np.zeros_like(y)}[mode]
    gate = (gate * np.float32(strength)).astype(np.float32)
    log_ceiling = np.float32(math.log1p(MAX_HDR * LOG_SCALE))
    pred_log = np.clip(np.log1p(base * np.float32(LOG_SCALE)) + fields.residual * gate[..., None], 0.0, log_ceiling)
    pred = (np.expm1(pred_log) / np.float32(LOG_SCALE)).astype(np.float32)
    rec = fields.highlight
    if limit is not None:
        rec = (rec * np.clip(np.asarray(limit, np.float32), 0.0, 1.0)).astype(np.float32)
    out = (base + rec[..., None] * (pred - base)).astype(np.float32)
    if region is not None:
        out = np.where((gate == 0)[..., None], base, out)          # outside the region: the baseline itself
    if limit is not None:
        out = np.where((rec == 0)[..., None], base, out)
    return out


# ── 4-7. the master chain (core/master.cpp) ─────────────────────────────────
def _srgb_to_linear_d(v):
    v = np.asarray(v, np.float64)
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def anchor_target_knots(curve: str, extra: Optional[np.ndarray] = None,
                        own_codes: bool = False) -> Optional[np.ndarray]:
    """composite.cpp anchor_target_knots: the curve's inverse over sRGB decode, per code (log2)."""
    if curve == "aces" and extra is None:
        return None
    ramp = np.repeat((np.arange(KNOTS, dtype=np.float32) / np.float32(KNOTS - 1))[:, None], 3, 1)
    base = baseline_network(ramp, curve, extra, own_codes)[:, 1].astype(np.float64) * (10000.0 / 203.0)
    have = srgb_to_linear(np.arange(KNOTS, dtype=np.float32) / np.float32(KNOTS - 1)).astype(np.float64)
    ok = (base > 0) & (have > 0)
    knots = np.where(ok, np.log2(np.where(ok, base, 1.0) / np.where(ok, have, 1.0)), 0.0).astype(np.float32)
    knots[0] = knots[1]
    return knots


def anchor_to_sdr(nits, sdr, knee=0.9, softness=0.04, knots=None):
    s = np.asarray(sdr, np.float64)
    light = _srgb_to_linear_d(s)
    if knots is not None:
        n = len(knots)
        pos = np.clip(s.astype(np.float32), 0, 1) * np.float32(n - 1)
        lo = np.minimum(np.floor(pos).astype(np.int64), n - 2)
        frac = pos - lo.astype(np.float32)
        light = light * np.exp2((knots[lo] + (knots[lo + 1] - knots[lo]) * frac).astype(np.float64))
    target = (light * _LUMA2020).sum(-1) * DIFFUSE_WHITE_NITS
    actual = (nits * _LUMA2020).sum(-1)
    eps = 1e-4
    want = (target + eps) / (actual + eps)
    code = s.max(-1)
    band_sel = (code > knee - softness) & (code < knee + softness) & (actual > 1e-9)
    band = want[band_sel]
    hold = float(np.median(band)) if band.size >= 64 else float(np.median(want))
    r = np.clip((code - (knee - softness)) / (2.0 * softness), 0.0, 1.0)
    r = r * r * (3.0 - 2.0 * r)
    gain = np.where(code > knee - softness, want * (1.0 - r) + hold * r, want)
    gain = np.where(np.isfinite(gain), gain, 1.0)
    return nits * gain[..., None]


def _ramp(code, knee, softness):
    r = np.clip((code - (knee - softness)) / (2.0 * softness), 0.0, 1.0)
    return r * r * (3.0 - 2.0 * r)


def _cv_ksize(sigma):
    return int(round(sigma * 4.0 * 2.0 + 1.0)) | 1


def carry_source_chroma(nits, sdr, knee=0.99, softness=0.01, sigma=2.0):
    import cv2
    s = np.asarray(sdr, np.float64)
    ramp = _ramp(s.max(-1), knee, softness).astype(np.float32)
    k = _cv_ksize(sigma)
    blurred = cv2.GaussianBlur(ramp, (k, k), sigma, borderType=cv2.BORDER_REPLICATE).astype(np.float32)
    source = _srgb_to_linear_d(s) * DIFFUSE_WHITE_NITS
    kk = ((nits * _LUMA2020).sum(-1) + 1e-6) / ((source * _LUMA2020).sum(-1) + 1e-6)
    keep = (np.float32(1.0) - blurred).astype(np.float64)
    out = source * (kk * keep)[..., None] + nits * blurred.astype(np.float64)[..., None]
    return np.where(np.isfinite(out), out, 0.0)


def _source_flatness(sdr):
    import cv2
    s = np.asarray(sdr, np.float64)
    spread = np.zeros(s.shape[:2])
    for x in (s.max(-1), (s * _LUMA2020).sum(-1)):
        m = cv2.boxFilter(x, -1, (7, 7), normalize=True, borderType=cv2.BORDER_REFLECT)
        mm = cv2.boxFilter(x * x, -1, (7, 7), normalize=True, borderType=cv2.BORDER_REFLECT)
        spread = np.maximum(spread, np.sqrt(np.maximum(mm - m * m, 0.0)))
    return _smooth01((3.0 - spread * 255.0) / (3.0 - 1.0))


def settle_highlight_grain(nits, sdr, knee=0.9, softness=0.04, sigma=2.0):
    import cv2
    s = np.asarray(sdr, np.float64)
    ramp = _smooth01((s.max(-1) - (knee - softness)) / (2.0 * softness))
    if not (ramp > 0).any():
        return nits
    flat = _source_flatness(s)
    if not (ramp * flat > 0).any():
        return nits
    luma = np.maximum((nits * _LUMA2020).sum(-1), 0.0)
    k = _cv_ksize(sigma)
    num = cv2.GaussianBlur(flat * luma, (k, k), sigma, borderType=cv2.BORDER_REFLECT)
    den = cv2.GaussianBlur(flat, (k, k), sigma, borderType=cv2.BORDER_REFLECT)
    settled = np.where(den > 1e-6, num / np.maximum(den, 1e-6), luma)
    target = luma + ramp * flat * (settled - luma)
    gain = np.where(luma > 1e-6, target / np.maximum(luma, 1e-6), 1.0)
    out = nits * gain[..., None]
    return np.where(np.isfinite(out), out, 0.0)


# ── Region EV (composite.cpp qualifier_mask / region_ev_gain, master.cpp apply_region_ev) ──
_LUMA2020_F = np.array([np.float32(0.2627), np.float32(0.6780), np.float32(0.0593)], np.float64)


@dataclass
class RegionBand:
    """Push a range of absolute nits (Rec.2020 luma) up or down by ``ev`` stops."""
    low_nits: float
    high_nits: float
    ev: float


DEFAULT_REGION_BANDS = (RegionBand(400.0, 2000.0, 0.0), RegionBand(2000.0, 8000.0, 0.0), RegionBand(0.05, 12.0, 0.0))


def qualifier_mask(luma_nits, low, high, softness_stops):
    log_y = np.log2(np.maximum(luma_nits, 1e-6))
    soft = max(float(softness_stops), 1e-3)
    rise = np.clip((log_y - (math.log2(low) - soft)) / soft, 0.0, 1.0)
    fall = np.clip(((math.log2(high) + soft) - log_y) / soft, 0.0, 1.0)
    m = np.minimum(rise, fall)
    return (m * m * (3.0 - 2.0 * m)).astype(np.float32)


def apply_region_ev(nits: np.ndarray, bands: Sequence[RegionBand], softness_stops: float = 1.0,
                    masks: Sequence[Optional[np.ndarray]] = (), exposure_ev: float = 0.0,
                    ceiling_nits: float = CEILING_NITS) -> np.ndarray:
    """``nits`` (H, W, 3) absolute. The exposure first; the bands read the exposed nits; masks gate their band."""
    exposure = 2.0 ** float(exposure_ev) if math.isfinite(exposure_ev) else 1.0
    graded = [i for i, b in enumerate(bands) if b.ev != 0.0]
    if not graded and exposure == 1.0:
        return nits
    px = nits.astype(np.float64) * exposure
    if graded:
        y = (np.maximum(px, 0.0) * _LUMA2020_F).sum(-1)
        total = np.zeros(y.shape, np.float64)
        for i in graded:
            b = bands[i]
            term = (np.float32(b.ev) * qualifier_mask(y, b.low_nits, b.high_nits, softness_stops)).astype(np.float32)
            if i < len(masks) and masks[i] is not None:
                term = (term * np.asarray(masks[i], np.float32)).astype(np.float32)
            total += term.astype(np.float64)
        px = px * np.exp2(total)[..., None]
    return np.clip(px, 0.0, ceiling_nits)


# ── the whole route ─────────────────────────────────────────────────────────
@dataclass
class AnalyticOptions:
    source_curve: str = DEFAULT_SOURCE_CURVE
    strength: float = 1.0               # how much of the fill (0 = the plain inverse)
    clip_fill: bool = True
    clip_peaks: Sequence[ClipPeak] = ()
    highlight_detail: float = 1.0       # 1 = off; 1.5-2 to judge
    anchor: bool = True
    anchor_knee: float = 0.9
    carry_chroma: bool = True
    settle_grain: bool = True
    exposure_ev: float = 0.0
    region: Optional[Region] = None                 # None: RUDRA's fixed priors (revision 1)
    calibration: Sequence[CalibrationPoint] = ()    # up to three anchors over the source curve
    reference: Optional[ReferenceFit] = None        # replaces the calibration when set
    limit: Optional[np.ndarray] = None              # (H, W) where the reconstruction may act
    mattes: bool = True                             # also build the mattes and pixel classes (RUDRA 1.1)
    # The SDR holds the source's own codes, as every ComfyUI image does (RUDRA 1.3): BT.1886 reads them as
    # video codes. False reproduces the RUDRA command line's video path, which the golden fixtures come from.
    own_codes: bool = True


@dataclass
class AnalyticResult:
    image: np.ndarray                   # (H, W, 3) float32 scene-linear Rec.709, 1.0 = 203 nits
    filled: np.ndarray                  # (H, W) 1 where the fill or the detail gain changed the frame
    estimated: np.ndarray               # (H, W) 1 where every channel clipped (no information in the SDR)
    info: ClipFillResult
    region_weight: Optional[np.ndarray] = None      # (H, W) max of the region's weights, when on
    curve_note: str = ""
    mattes: Optional["Mattes"] = None               # RUDRA 1.1 mattes and pixel classes, before the grade


def curve_correction(opt: "AnalyticOptions"):
    """The calibration or reference correction over the source curve, and a line saying what it did."""
    if opt.reference is not None:
        k = reference_knots(opt.reference, opt.source_curve)
        note = (f"reference: {opt.reference.samples} samples over {opt.reference.codes_seen} codes, exposure "
                f"{reference_exposure(opt.reference, opt.source_curve):+.2f} stops and a per-code curve; "
                f"residual mean {opt.reference.residual_mean:.2f}, p95 {opt.reference.residual_p95:.2f} stops")
        if opt.reference.residual_p95 > 0.5:
            note += " (over half a stop: the grade has a local key the curve cannot carry)"
        return k, note
    pts = _anchors(opt.calibration)
    if not pts:
        return None, ""
    if not calibration_is_monotone(opt.calibration, opt.source_curve):
        raise ValueError("these calibration anchors do not describe a monotone curve over the source curve; "
                         "check the codes and nits (a brighter code must not ask for fewer nits)")
    if len(pts) == 1:
        code, nits = pts[0]
        st = math.log2(nits / max(float(source_code_nits(opt.source_curve, code)), 1e-9))
        return calibration_knots(opt.calibration, opt.source_curve), f"calibration: 1 anchor, exposure {st:+.2f} stops"
    return (calibration_knots(opt.calibration, opt.source_curve),
            f"calibration: {len(pts)} anchors, piecewise-linear in log2 between them, flat outside")


def analytic_master(sdr: np.ndarray, opt: AnalyticOptions = AnalyticOptions()) -> AnalyticResult:
    sdr = np.nan_to_num(np.asarray(sdr, np.float32)[..., :3], nan=0.0, posinf=1.0, neginf=0.0)
    if opt.source_curve not in SOURCE_CURVES:
        raise ValueError(f"unknown source curve {opt.source_curve!r}")
    if opt.region is not None and opt.region.error():
        raise ValueError(opt.region.error())
    extra, note = curve_correction(opt)
    fields, info = clip_fill(sdr, opt.clip_peaks, opt.highlight_detail, opt.clip_fill)
    net = composite(sdr, fields, opt.source_curve, opt.strength, region=opt.region, extra_knots=extra,
                    limit=opt.limit, own_codes=opt.own_codes)
    mattes = None
    if opt.mattes:
        base = baseline_network(np.clip(sdr, 0.0, 1.0), opt.source_curve, extra, opt.own_codes)
        mattes = reconstruction_mattes(sdr, fields, net, base, opt.strength, region=opt.region, limit=opt.limit)
        mattes.settings = _settings_of(opt)
    nits = net.astype(np.float64) * 10000.0
    if opt.anchor:
        nits = anchor_to_sdr(nits, sdr, opt.anchor_knee, 0.04,
                             anchor_target_knots(opt.source_curve, extra, opt.own_codes))
    if opt.carry_chroma:
        nits = carry_source_chroma(nits, sdr)
    if opt.settle_grain:
        nits = settle_highlight_grain(nits, sdr, opt.anchor_knee)
    if opt.exposure_ev:
        nits = np.clip(nits * (2.0 ** float(opt.exposure_ev)), 0.0, CEILING_NITS)
    image = (nits / DIFFUSE_WHITE_NITS).astype(np.float32)
    estimated = (_clip_weight(sdr) >= 1.0).all(-1).astype(np.float32)
    filled = fields.highlight
    if opt.limit is not None:
        filled = (filled * (np.asarray(opt.limit, np.float32) > 0)).astype(np.float32)
    rw = None
    if opt.region is not None:
        wh, ws = region_weights(_luma(np.clip(sdr, 0, 1)), opt.region)
        rw = np.clip(np.maximum(wh, ws), 0.0, 1.0).astype(np.float32)
    return AnalyticResult(image, filled, estimated, info, rw, note, mattes)


def parse_clip_peaks(text: str) -> List[ClipPeak]:
    """Peaks from JSON: [{"x": 0.5, "y": 0.3, "nits": 2500}, ...] (RUDRA's clip_peaks). Empty text: none."""
    text = (text or "").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f'clip_peaks must be JSON like [{{"x": 0.5, "y": 0.3, "nits": 2500}}]: {exc}') from exc
    if isinstance(data, dict):
        data = [data]
    out = []
    for p in data:
        x, y, nits = float(p["x"]), float(p["y"]), float(p["nits"])
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError(f"clip_peaks x and y are 0 to 1 across the frame (got {x}, {y})")
        out.append(ClipPeak(x, y, nits))
    return out


def parse_calibration(text: str, sdr: Optional[np.ndarray] = None) -> List[CalibrationPoint]:
    """Anchors from JSON: [{"code": 118, "nits": 30}, {"x": 0.4, "y": 0.6, "nits": 2.5}, ...].

    ``code`` is the SDR code (0..255). Or ``x`` and ``y`` across the frame (0..1), the
    point the RUDRA app's click would be: its code is the Rec.709 luma of the SDR there.
    At most three anchors (black, an 18 % grey, a highlight). Empty text: none.
    """
    text = (text or "").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f'calibration must be JSON like [{{"code": 118, "nits": 30}}]: {exc}') from exc
    if isinstance(data, dict):
        data = [data]
    if len(data) > 3:
        raise ValueError("calibration takes at most three anchors (black, an 18 % grey, a highlight)")
    out = []
    for p in data:
        nits = float(p["nits"])
        if "code" in p:
            code = int(round(float(p["code"])))
        else:
            if sdr is None:
                raise ValueError("a calibration anchor by x and y needs the frame")
            x, y = float(p["x"]), float(p["y"])
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError(f"calibration x and y are 0 to 1 across the frame (got {x}, {y})")
            h, w = sdr.shape[:2]
            px = np.clip(np.asarray(sdr, np.float32)[min(int(y * h), h - 1), min(int(x * w), w - 1), :3], 0, 1)
            code = int(round(float((px * _LUMA709).sum()) * 255.0))
        if not (0 <= code <= 255) or not nits > 0:
            raise ValueError(f"a calibration anchor needs a code 0..255 and nits above 0 (got {code}, {nits})")
        out.append(CalibrationPoint(code, nits))
    return out



# ── 8. RUDRA 1.1 priorities 3-5: mattes, pixel classes, the regions picture, brackets ──
# composite.cpp classify_pixels / reconstruction_mattes, deliver/master.cpp, view.cpp
# exposure_bracket. Every matte is measured on the composite, before the anchor and the
# grade (no Region EV, no exposure), against its corrected baseline, in network units.

CHANGE_STOPS = 0.1                      # compare.hpp kChangeStops
CHANGE_FLOOR_NITS = 0.05                # compare.hpp kChangeFloorNits
SDR_CLIP_CODE = np.float32(254.0 / 255.0)
SDR_CRUSH_CODE = np.float32(1.0 / 255.0)
PIXEL_CLASSES = ("preserved", "follows", "reconstructed", "estimated", "generated")
PIXEL_CLASS_COLOUR = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.25, 0.78, 0.86],
                               [0.92, 0.30, 0.86], [0.98, 0.78, 0.20]], np.float32)
MAP_GREY = np.float32(0.6)              # view.hpp kMapGrey
MAP_MIX = np.float32(0.85)              # view.hpp kMapMix
BRACKET_STOPS = (-2.0, 0.0, 2.0)        # view.hpp kBracketStops
CONFIDENCE_NOTE = ("a constraint score: 1 preserved / follows, 1/(1+|d|) reconstructed, 0.5/(1+|d|) estimated, "
                   "0 generated; d in stops. Not a probability")
MATTES_NOTE = ("highlight / shadow: the gate's priors as applied (mode, region, strength, recovery mask); changed: 1 "
               "where the reconstruction differs from the baseline; region: the luminance region; class: 0 preserved, "
               "0.25 follows, 0.5 reconstructed, 0.75 estimated, 1 generated; confidence: how tightly the SDR pins the "
               "value (not a probability); generated.highlight / generated.shadow: rudra.generated split by the SDR's "
               "side (max code >= 0.5 is highlight). Before the grade.")


@dataclass
class Mattes:
    """One frame's mattes. Planes (H, W); ``grey`` the 0 EV view of the ungraded composite (H, W, 3) uint8."""
    highlight: np.ndarray
    shadow: np.ndarray
    changed: np.ndarray
    region: Optional[np.ndarray]
    cls: np.ndarray                     # uint8, index into PIXEL_CLASSES
    confidence: np.ndarray
    grey: np.ndarray
    sdr_max: np.ndarray                 # max SDR code per pixel (for the generated split)
    settings: dict = field(default_factory=dict)

    @property
    def share(self) -> List[float]:
        n = max(self.cls.size, 1)
        counts = np.bincount(self.cls.ravel(), minlength=len(PIXEL_CLASSES))
        return [float(c) / n for c in counts[:len(PIXEL_CLASSES)]]

    def with_generated(self, generated: Optional[np.ndarray]) -> "Mattes":
        """Class 4 wherever the AI fill drew (``generated`` > 0), whatever else holds; confidence 0 there."""
        if generated is None:
            return self
        g = np.asarray(generated, np.float32)
        if g.shape != self.cls.shape:
            raise ValueError(f"the generated mask is {g.shape[1]}x{g.shape[0]}; the mattes are "
                             f"{self.cls.shape[1]}x{self.cls.shape[0]}")
        on = g > 0
        return Mattes(self.highlight, self.shadow, self.changed, self.region,
                      np.where(on, np.uint8(4), self.cls).astype(np.uint8),
                      np.where(on, np.float32(0.0), self.confidence).astype(np.float32),
                      self.grey, self.sdr_max, dict(self.settings))

    def channels(self, generated: Optional[np.ndarray] = None) -> dict:
        """The named float planes of ``<name>.mattes.exr``, in RUDRA's order."""
        m = self.with_generated(generated)
        out = {"rudra.class": (m.cls.astype(np.float32) / np.float32(len(PIXEL_CLASSES) - 1)),
               "rudra.confidence": m.confidence, "rudra.highlight": m.highlight, "rudra.shadow": m.shadow,
               "rudra.changed": m.changed.astype(np.float32)}
        if m.region is not None:
            out["rudra.region"] = m.region
        if generated is not None:
            g = np.clip(np.asarray(generated, np.float32), 0.0, 1.0)
            hi = self.sdr_max >= np.float32(0.5)
            out["rudra.generated"] = g
            out["rudra.generated.highlight"] = np.where(hi, g, np.float32(0.0)).astype(np.float32)
            out["rudra.generated.shadow"] = np.where(hi, np.float32(0.0), g).astype(np.float32)
        return out


def _settings_of(opt: "AnalyticOptions") -> dict:
    return {"reconstruction": "analytic (no model)", "source_curve": opt.source_curve,
            "residual_strength": float(opt.strength), "clip_fill": bool(opt.clip_fill),
            "clip_peaks": len(opt.clip_peaks), "highlight_detail": float(opt.highlight_detail),
            "anchored": bool(opt.anchor), "anchor_knee": float(opt.anchor_knee),
            "chroma_carried": bool(opt.carry_chroma), "grain_settled": bool(opt.settle_grain),
            "exposure_ev": float(opt.exposure_ev), "region": None if opt.region is None else vars(opt.region).copy(),
            "calibration": [[p.code, p.nits] for p in opt.calibration] if opt.reference is None else [],
            "reference": opt.reference is not None, "limited": opt.limit is not None,
            "own_codes": bool(opt.own_codes)}


def classify_pixels(sdr, reconstruction, baseline, generated=None):
    """composite.cpp classify_pixels: (class uint8, confidence float32). Network units in."""
    code = np.clip(np.asarray(sdr, np.float32)[..., :3], 0.0, 1.0).max(-1)
    r = np.asarray(reconstruction, np.float32)
    b = np.asarray(baseline, np.float32)
    exact = (r == b).all(-1)
    floor = CHANGE_FLOOR_NITS / 10000.0
    rm = np.maximum(r.max(-1).astype(np.float64), 0.0)
    bm = np.maximum(b.max(-1).astype(np.float64), 0.0)
    d = np.abs(np.log2((rm + floor) / (bm + floor)))
    no_info = (code >= SDR_CLIP_CODE) | (code <= SDR_CRUSH_CODE)
    cls = np.full(code.shape, 2, np.uint8)
    conf = (1.0 / (1.0 + d))
    est = no_info
    cls[est] = 3
    conf = np.where(est, 0.5 / (1.0 + d), conf)
    fol = d <= CHANGE_STOPS
    cls[fol] = 1
    conf = np.where(fol, 1.0, conf)
    cls[exact] = 0
    conf = np.where(exact, 1.0, conf)
    if generated is not None:
        on = np.asarray(generated, np.float32) > 0
        cls[on] = 4
        conf = np.where(on, 0.0, conf)
    return cls, conf.astype(np.float32)


def _srgb_codes(v):
    """view.cpp linear_to_srgb then lround(clamp * 255), float32 as the native view."""
    x = np.clip(np.asarray(v, np.float32), 0.0, 1.0)
    s = np.where(x > np.float32(0.0031308), np.float32(1.055) * np.power(x, np.float32(1.0 / 2.4)) - np.float32(0.055),
                 np.float32(12.92) * x).astype(np.float32)
    return np.floor(np.clip(s, 0.0, 1.0) * np.float32(255.0) + np.float32(0.5)).astype(np.uint8)


def exposure_bracket_network(network, ev: float = 0.0):
    """view.cpp exposure_bracket: SDR sRGB codes, 203 nits at code 255 at 0 EV. Network units in."""
    gain = np.float32(2.0 ** float(ev))
    scale = np.float32(10000.0 / DIFFUSE_WHITE_NITS)
    return _srgb_codes((np.asarray(network, np.float32) * gain) * scale)


def exposure_bracket(scene_linear, ev: float = 0.0):
    """The bracket of a scene-linear picture (1.0 = 203 nits): sRGB codes, uint8 (H, W, 3)."""
    net = np.asarray(scene_linear, np.float32)[..., :3] * np.float32(DIFFUSE_WHITE_NITS / 10000.0)
    return exposure_bracket_network(net, ev)


def code_delta(a, b):
    """view.cpp code_delta: (largest code difference over every channel, share of pixels differing)."""
    a = np.asarray(a, np.int16)
    b = np.asarray(b, np.int16)
    if a.shape != b.shape:
        return 256, 1.0
    worst = np.abs(a - b).max(-1)
    return int(worst.max()) if worst.size else 0, float((worst > 0).mean()) if worst.size else 0.0


def reconstruction_mattes(sdr, fields: _Fields, reconstruction, baseline, strength: float = 1.0,
                          mode: str = "all", region: Optional[Region] = None,
                          limit: Optional[np.ndarray] = None, generated=None) -> Mattes:
    """composite.cpp reconstruction_mattes + classify_pixels + the regions picture's grey. ``limit``
    (Radiance's mask) multiplies the recovery mask as it does in the composite."""
    c = np.clip(np.asarray(sdr, np.float32)[..., :3], 0.0, 1.0)
    y = _luma(c)
    hp = _sigmoid((y - np.float32(0.82)) * np.float32(24.0))
    sp = _sigmoid((np.float32(0.10) - y) * np.float32(24.0))
    rw = None
    if region is not None:
        wh, ws = region_weights(y, region)
        hp, sp = (hp * wh).astype(np.float32), (sp * ws).astype(np.float32)
        rw = np.clip(np.maximum(wh, ws), 0.0, 1.0).astype(np.float32)
    rec = fields.highlight
    if limit is not None:
        rec = (rec * np.clip(np.asarray(limit, np.float32), 0.0, 1.0)).astype(np.float32)
    st = np.float32(strength)
    want_h = mode in ("all", "highlights")
    want_s = mode in ("all", "shadows")
    zero = np.zeros_like(y)
    hl = np.clip(hp * st * rec, 0.0, 1.0).astype(np.float32) if want_h else zero
    sh = np.clip(sp * st * rec, 0.0, 1.0).astype(np.float32) if want_s else zero
    r = np.asarray(reconstruction, np.float32)
    b = np.asarray(baseline, np.float32)
    changed = (r != b).any(-1).astype(np.uint8)
    cls, conf = classify_pixels(c, r, b, generated)
    return Mattes(hl, sh, changed, rw, cls, conf, exposure_bracket_network(r, 0.0), c.max(-1))


def regions_picture(m: Mattes, generated=None) -> np.ndarray:
    """master.cpp's ``<name>.regions.png``: the picture's grey with the class colours mixed in
    (reconstructed cyan, estimated magenta, generated amber) and the region's edge in white. uint8 (H, W, 3)."""
    m = m.with_generated(generated)
    g = m.grey.astype(np.float32)
    lum = (np.float32(0.2126) * g[..., 0] + np.float32(0.7152) * g[..., 1] + np.float32(0.0722) * g[..., 2]) / np.float32(255.0)
    mix = np.where(m.cls >= 2, MAP_MIX, np.float32(0.0)).astype(np.float32)[..., None]
    col = PIXEL_CLASS_COLOUR[m.cls]
    v = (np.float32(1.0) - mix) * (lum * MAP_GREY)[..., None] + mix * col
    if m.region is not None:
        inside = m.region > 0
        edge = np.zeros_like(inside)
        edge[:, :-1] |= inside[:, :-1] != inside[:, 1:]
        edge[:, 1:] |= inside[:, 1:] != inside[:, :-1]
        edge[:-1, :] |= inside[:-1, :] != inside[1:, :]
        edge[1:, :] |= inside[1:, :] != inside[:-1, :]
        v = np.where(edge[..., None], np.float32(1.0), v)
    return np.floor(np.clip(v, 0.0, 1.0) * np.float32(255.0) + np.float32(0.5)).astype(np.uint8)


def class_share_pct(m: Mattes, generated=None) -> dict:
    share = m.with_generated(generated).share
    return {f"{k}_pct": round(100.0 * v, 3) for k, v in zip(PIXEL_CLASSES, share)}
