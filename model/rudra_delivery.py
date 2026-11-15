"""RUDRA's delivery measures, ported for Radiance (RUDRA 1.1-1.2, Apache 2.0).

Ported from FXTD-Studios/RUDRA, commit 74f03b6 (10 Nov 2026):

* ``rudra/delivery/metadata.py`` -- per-frame MaxRGB statistics, shot
  detection, CTA-861.3 MaxCLL / MaxFALL, Dolby Vision L1 per shot (the
  ``dovi_tool generate`` config, profile 8.1), HDR10+ scene statistics and the
  per-frame HDR10+ JSON that x265 (``--dhdr10-info``) and ``hdr10plus_tool``
  read. Nearly verbatim.
* ``native/core/src/hdr10.cpp`` ``master_to_peak`` / ``fit_to_peak`` -- the
  hue-preserving shoulder to a display peak, with RUDRA 1.1's peak fit: "scale"
  (the whole pixel scaled until its largest channel fits: hue and saturation
  kept, luminance lost) or "luminance" (the pixel moved toward its own
  luminance just far enough: luminance and hue direction kept, saturation given
  up, only where the display cannot show it anyway). RUDRA's default is
  "luminance".
* ``rudra/video.py`` -- the flicker measure (how far the HDR's average exposure
  moved from one frame to the next beyond what the source's did, in EV, on a
  36 x 64 area thumbnail, with a per-cell ``local`` version), the cut test of
  its shot smoother, and the static hold: where the source pixel did not change,
  the HDR's gain over the source is held across the shot, reset at each cut.

Conventions are RUDRA's: absolute nits in Rec.2020 primaries for the delivery
measures; network units (1.0 = 10 000 nits) for the hold. Radiance's
scene-linear (1.0 = 203 nits) is converted at the edges. Parity is checked
against RUDRA's own code in tests/test_rudra_delivery.py.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

DIFFUSE_WHITE_NITS = 203.0
LUMA_REC2020 = (0.2627, 0.6780, 0.0593)
LUMA_REC709 = (0.2126, 0.7152, 0.0722)
_LUMA2020_F = np.asarray(LUMA_REC2020, np.float32)

# ── PQ (rudra/hdr10.py, float32 as numpy runs it) ───────────────────────────
PQ_M1, PQ_M2 = 2610.0 / 16384.0, 2523.0 / 32.0
PQ_C1, PQ_C2, PQ_C3 = 3424.0 / 4096.0, 2413.0 / 128.0, 2392.0 / 128.0


def pq_oetf(nits):
    v = np.clip(np.asarray(nits, dtype=np.float32) / 10000.0, 0.0, 1.0)
    p = np.power(v, PQ_M1)
    return np.power((PQ_C1 + PQ_C2 * p) / (1.0 + PQ_C3 * p), PQ_M2)


# ── peak fit (hdr10.cpp master_to_peak / hdr10.hpp fit_to_peak) ─────────────
PEAK_FITS = ("luminance", "scale")


def master_to_peak(rgb_nits: np.ndarray, peak_nits: float = 1000.0, knee_nits: Optional[float] = None,
                   fit: str = "luminance") -> np.ndarray:
    """(H, W, 3) absolute nits in Rec.2020 -> the same, every channel at most ``peak_nits``.

    Below the knee (default 0.75 x peak) luminance is unchanged; above it, it
    approaches the peak smoothly (exponential shoulder) with the pixel's ratios
    kept; then any channel still over the peak is brought in by ``fit``."""
    if not 100.0 <= peak_nits <= 10000.0:
        raise ValueError("peak_nits must be between 100 and 10000")
    knee = 0.75 * peak_nits if knee_nits is None else float(knee_nits)
    if not 0.0 <= knee < peak_nits:
        raise ValueError("knee_nits must be >= 0 and below peak_nits")
    if fit not in PEAK_FITS:
        raise ValueError(f"peak fit must be one of {PEAK_FITS}")
    px = np.maximum(np.asarray(rgb_nits, np.float32), np.float32(0.0))
    kf, span, pf = np.float32(knee), np.float32(peak_nits - knee), np.float32(peak_nits)
    lum = (px[..., 0] * _LUMA2020_F[0] + px[..., 1] * _LUMA2020_F[1]) + px[..., 2] * _LUMA2020_F[2]
    with np.errstate(over="ignore"):   # the branch not taken may overflow far under the knee
        comp = np.where(lum <= kf, lum, kf + span * (np.float32(1.0) - np.exp(-(lum - kf) / span))).astype(np.float32)
    px = px * (comp / np.maximum(lum, np.float32(1e-6)))[..., None]
    cmax = px.max(-1)
    if fit == "scale":
        s = np.minimum(np.float32(1.0), pf / np.maximum(cmax, np.float32(1e-6)))
        return np.clip(px * s[..., None], 0.0, pf).astype(np.float32)
    over = cmax > pf
    keep = over & (comp >= 0) & (comp < pf)
    t = np.where(keep, (pf - comp) / np.where(keep, cmax - comp, np.float32(1.0)), np.float32(1.0))
    toward = comp[..., None] + t[..., None] * (px - comp[..., None])
    scaled = px * (pf / np.maximum(cmax, np.float32(1e-6)))[..., None]
    out = np.where(keep[..., None], toward, np.where(over[..., None], scaled, px))
    return np.clip(out, 0.0, pf).astype(np.float32)


# ── dynamic metadata (rudra/delivery/metadata.py) ───────────────────────────
PERCENTILES = (1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0, 99.98)
HDR10PLUS_INDEX = (1, 5, 10, 25, 50, 75, 90, 95, 99)
# dovi_tool's own generator clamps L1 to these (CM v4.0); the analysis keeps the measured values.
L1_MIN_PQ_MAX, L1_MAX_PQ_MIN, L1_MAX_PQ_MAX, L1_AVG_PQ_MIN_CMV40 = 12, 2081, 4095, 1229


@dataclass
class FrameStats:
    """MaxRGB statistics of one frame, in nits."""
    index: int
    min_nits: float
    avg_nits: float
    max_nits: float
    maxscl_nits: Tuple[float, float, float]
    percentiles_nits: Tuple[float, ...]
    log_hist: Optional[np.ndarray] = field(repr=False, default=None)


def _pq12(nits):
    code = np.round(pq_oetf(np.asarray(nits, dtype=np.float64)) * 4095.0)
    out = np.clip(code, 0, 4095).astype(int)
    return out.tolist() if out.ndim else int(out)


def analyze_frame(rgb_nits: np.ndarray, index: int = 0, hist_bins: int = 64) -> FrameStats:
    arr = np.asarray(rgb_nits, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3), got {arr.shape}")
    arr = np.maximum(np.nan_to_num(arr, nan=0.0, posinf=10000.0), 0.0)
    max_rgb = arr.max(axis=2)
    log_l = np.log2(np.maximum(max_rgb, 1e-4))
    hist, _ = np.histogram(log_l, bins=hist_bins, range=(-14.0, 21.0))
    hist = hist.astype(np.float64) / max(1, max_rgb.size)
    return FrameStats(index, float(max_rgb.min()), float(max_rgb.mean()), float(max_rgb.max()),
                      tuple(float(v) for v in arr.reshape(-1, 3).max(axis=0)),
                      tuple(float(v) for v in np.percentile(max_rgb, PERCENTILES)), hist)


def detect_shots(stats: Sequence[FrameStats], threshold: float = 0.35) -> List[Tuple[int, int]]:
    """[(start, length), ...] by log-luminance histogram distance; 0.35 favours fewer, longer shots."""
    if not stats:
        return []
    cuts = [0]
    for prev, cur in zip(stats, stats[1:]):
        a, b = prev.log_hist, cur.log_hist
        if a is None or b is None:
            continue
        if float(0.5 * np.sum((a - b) ** 2 / (a + b + 1e-9))) > threshold:
            cuts.append(cur.index - stats[0].index)
    cuts.append(len(stats))
    return [(cuts[i], cuts[i + 1] - cuts[i]) for i in range(len(cuts) - 1)]


def shots_from_cuts(cuts: Sequence[int], frame_count: int) -> List[Tuple[int, int]]:
    starts = sorted({0, *(int(c) for c in cuts if 0 < int(c) < frame_count)})
    return [(s, (starts[i + 1] if i + 1 < len(starts) else frame_count) - s) for i, s in enumerate(starts)]


def maxcll_maxfall(stats: Sequence[FrameStats]) -> Tuple[int, int]:
    if not stats:
        raise ValueError("no frames analyzed")
    return int(np.ceil(max(s.max_nits for s in stats))), int(np.ceil(max(s.avg_nits for s in stats)))


def l1_per_shot(stats, shots) -> List[dict]:
    out = []
    for start, length in shots:
        w = stats[start:start + length]
        avg = float(np.mean([s.avg_nits for s in w]))
        lo, hi = min(s.min_nits for s in w), max(s.max_nits for s in w)
        out.append({"start": start, "duration": length, "min_pq": _pq12(lo), "avg_pq": _pq12(avg),
                    "max_pq": _pq12(hi), "min_nits": lo, "avg_nits": avg, "max_nits": hi})
    return out


def _l1_block(shot: dict) -> dict:
    max_pq = min(max(shot["max_pq"], L1_MAX_PQ_MIN), L1_MAX_PQ_MAX)
    return {"Level1": {"min_pq": min(max(shot["min_pq"], 0), L1_MIN_PQ_MAX), "max_pq": max_pq,
                       "avg_pq": min(max(shot["avg_pq"], L1_AVG_PQ_MIN_CMV40), max_pq - 1)}}


def to_dovi_generate_json(stats, shots=None, mastering_peak_nits: float = 1000.0,
                          mastering_min_nits: float = 0.0001) -> dict:
    """``dovi_tool generate --json`` config, profile 8.1, L1 per shot."""
    shots = shots or detect_shots(stats)
    max_cll, max_fall = maxcll_maxfall(stats)
    return {
        "cm_version": "V40", "profile": "8.1", "length": len(stats),
        "level6": {"max_display_mastering_luminance": int(round(mastering_peak_nits)),
                   "min_display_mastering_luminance": int(round(mastering_min_nits * 10000)),
                   "max_content_light_level": max_cll, "max_frame_average_light_level": max_fall},
        "shots": [{"start": s["start"], "duration": s["duration"], "metadata_blocks": [_l1_block(s)]}
                  for s in l1_per_shot(stats, shots)],
    }


def to_hdr10plus_json(stats, shots=None, target_display_nits: int = 400) -> dict:
    """ST 2094-40-style scene statistics in cd/m2 (the analysis payload)."""
    shots = shots or detect_shots(stats)
    scenes = []
    for scene_id, (start, length) in enumerate(shots):
        w = stats[start:start + length]
        scenes.append({
            "SceneId": scene_id, "SceneFirstFrameIndex": start, "SceneFrameNumbers": length,
            "TargetedSystemDisplayMaximumLuminance": target_display_nits,
            "LuminanceParameters": {
                "Units": "cd/m2",
                "AverageMaxRGB": float(np.mean([s.avg_nits for s in w])),
                "MaxSCL": [float(max(s.maxscl_nits[c] for s in w)) for c in range(3)],
                "LuminanceDistributions": {
                    "DistributionIndex": list(PERCENTILES),
                    "DistributionValues": [float(np.max([s.percentiles_nits[i] for s in w]))
                                           for i in range(len(PERCENTILES))]}},
        })
    return {"JSONInfo": {"Generator": "RUDRA delivery", "Version": "1.0", "Units": "cd/m2"},
            "SceneInfoSummary": {"SceneFirstFrameIndex": [s["SceneFirstFrameIndex"] for s in scenes],
                                 "SceneFrameNumbers": [s["SceneFrameNumbers"] for s in scenes]},
            "SceneInfo": scenes}


def _tenth_nits(nits: float) -> int:
    return int(min(max(np.floor(float(nits) * 10.0 + 0.5), 0), 100000))


def to_hdr10plus_x265_json(stats, shots=None) -> dict:
    """Per-frame HDR10+ (profile A, one window) for x265 ``--dhdr10-info`` / ``hdr10plus_tool inject``;
    every frame carries its shot's statistics, so they never change mid-shot."""
    shots = shots or detect_shots(stats)
    frames = []
    for scene_id, (start, length) in enumerate(shots):
        w = stats[start:start + length]
        lum = {"AverageRGB": _tenth_nits(np.mean([s.avg_nits for s in w])),
               "LuminanceDistributions": {
                   "DistributionIndex": list(HDR10PLUS_INDEX),
                   "DistributionValues": [_tenth_nits(max(s.percentiles_nits[i] for s in w))
                                          for i in range(len(PERCENTILES))]},
               "MaxScl": [_tenth_nits(max(s.maxscl_nits[c] for s in w)) for c in range(3)]}
        for k in range(length):
            frames.append({"LuminanceParameters": lum, "NumberOfWindows": 1,
                           "TargetedSystemDisplayMaximumLuminance": 0, "SceneFrameIndex": k,
                           "SceneId": scene_id, "SequenceFrameIndex": start + k})
    return {"JSONInfo": {"HDR10plusProfile": "A", "Version": "1.0"}, "SceneInfo": frames,
            "SceneInfoSummary": {"SceneFirstFrameIndex": [s for s, _ in shots],
                                 "SceneFrameNumbers": [n for _, n in shots]},
            "ToolInfo": {"Tool": "RUDRA", "Version": "1.0"}}


def to_rudra_sidecar(stats, shots=None) -> dict:
    """The self-describing record: every value in nits and in 12-bit PQ."""
    shots = shots or detect_shots(stats)
    max_cll, max_fall = maxcll_maxfall(stats)
    return {
        "generator": "rudra.delivery.metadata",
        "convention": "MaxRGB per pixel; nits absolute; pq codes 12-bit (0-4095)",
        "frame_count": len(stats), "max_cll_nits": max_cll, "max_fall_nits": max_fall,
        "shots": l1_per_shot(stats, shots),
        "frames": [{"index": s.index, "min_nits": s.min_nits, "avg_nits": s.avg_nits, "max_nits": s.max_nits,
                    "min_pq": _pq12(s.min_nits), "avg_pq": _pq12(s.avg_nits), "max_pq": _pq12(s.max_nits),
                    "maxscl_nits": list(s.maxscl_nits),
                    "percentiles": {str(p): v for p, v in zip(PERCENTILES, s.percentiles_nits)}}
                   for s in stats],
    }


def dynamic_metadata_paths(stem) -> Dict[str, Path]:
    """rudra/video.py's names beside a master: ``<stem>_hdr10plus.json`` (x265 / hdr10plus_tool),
    ``_dovi_generate.json`` (dovi_tool), ``_hdr_analysis.json`` and ``_hdr10plus_scenes.json``."""
    stem = Path(stem)
    return {k: stem.with_name(stem.name + suf) for k, suf in
            (("hdr10plus", "_hdr10plus.json"), ("dovi_generate", "_dovi_generate.json"),
             ("analysis", "_hdr_analysis.json"), ("hdr10plus_scenes", "_hdr10plus_scenes.json"))}


def dynamic_metadata(stats, shots, mastering_peak_nits=1000.0, mastering_min_nits=0.0001,
                     target_display_nits=400) -> Dict[str, dict]:
    return {"hdr10plus": to_hdr10plus_x265_json(stats, shots),
            "dovi_generate": to_dovi_generate_json(stats, shots, mastering_peak_nits, mastering_min_nits),
            "analysis": to_rudra_sidecar(stats, shots),
            "hdr10plus_scenes": to_hdr10plus_json(stats, shots, target_display_nits)}


def write_json_atomic(path: Path, data: dict, indent: Optional[int] = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(data, indent=indent), encoding="utf-8", newline="\n")
    part.replace(path)


# ── flicker, cuts and the static hold (rudra/video.py) ──────────────────────
EXPOSURE_THUMB = (36, 64)
EXPOSURE_FLOOR = 2.0 ** -10
CUT_THRESHOLD = 0.15
STATIC_STILL, STATIC_MOVED = 1.5 / 255, 6.0 / 255
STATIC_SDR_FLOOR, STATIC_HDR_FLOOR = 1e-4, 1e-6
STATIC_CLIPPED, STATIC_CRUSHED = 0.98, 0.02
STATIC_HDR_STILL, STATIC_HDR_MOVED = 0.05, 0.25
SDR_TRANSFERS = ("srgb", "rec709", "gamma22", "gamma24")


def area_mean(plane, height=EXPOSURE_THUMB[0], width=EXPOSURE_THUMB[1]):
    """F.interpolate(mode='area') of one plane (adaptive average pooling), float64."""
    plane = np.asarray(plane, np.float64)
    h, w = plane.shape
    integral = np.zeros((h + 1, w + 1))
    integral[1:, 1:] = plane.cumsum(0).cumsum(1)
    y0 = (np.arange(height) * h) // height
    y1 = -((-(np.arange(height) + 1) * h) // height)
    x0 = (np.arange(width) * w) // width
    x1 = -((-(np.arange(width) + 1) * w) // width)
    area = (y1 - y0)[:, None] * (x1 - x0)[None, :]
    return (integral[y1][:, x1] - integral[y0][:, x1] - integral[y1][:, x0] + integral[y0][:, x0]) / area


def exposure_cells(luminance):
    return np.log2(np.maximum(area_mean(luminance), 0.0) + EXPOSURE_FLOOR)


def transfer_to_linear(x, transfer):
    x = np.clip(np.asarray(x, np.float64), 0.0, 1.0)
    if transfer == "srgb":
        return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    if transfer == "rec709":
        return np.where(x < 0.081, x / 4.5, ((x + 0.099) / 1.099) ** (1.0 / 0.45))
    return x ** (2.2 if transfer == "gamma22" else 2.4)


def source_luminance(rgb, transfer="srgb", primaries="rec709"):
    lin = transfer_to_linear(rgb, transfer)
    k = LUMA_REC2020 if primaries == "rec2020" else LUMA_REC709
    return lin[..., 0] * k[0] + lin[..., 1] * k[1] + lin[..., 2] * k[2]


def source_code_luma(rgb, primaries="rec709"):
    x = np.asarray(rgb, np.float64)
    k = LUMA_REC2020 if primaries == "rec2020" else LUMA_REC709
    return x[..., 0] * k[0] + x[..., 1] * k[1] + x[..., 2] * k[2], x.max(axis=2)


def hdr_luminance(network):
    """The HDR's Rec.2020-weighted luminance relative to diffuse white, from network units."""
    x = np.asarray(network, np.float64)
    k = LUMA_REC2020
    return (x[..., 0] * k[0] + x[..., 1] * k[1] + x[..., 2] * k[2]) * 100.0


def thumbnail(rgb):
    """The cut test's 36 x 64 area thumbnail of the SDR codes, (3, 36, 64)."""
    x = np.asarray(rgb, np.float64)
    return np.stack([area_mean(x[..., c]) for c in range(3)])


def detect_cuts(sdr_frames, threshold: float = CUT_THRESHOLD) -> List[int]:
    """ShadowSmoother.update's cut test: the mean absolute change of the thumbnail >= threshold.
    Frame 0 always starts a shot."""
    cuts, previous = [], None
    for i, f in enumerate(sdr_frames):
        t = thumbnail(f)
        if previous is None or float(np.mean(np.abs(t - previous))) >= threshold:
            cuts.append(i)
        previous = t
    return cuts


def flicker_report(sdr_frames, hdr_network_frames, cuts: Sequence[int], transfer="srgb",
                   primaries="rec709") -> dict:
    """rudra/video.py's temporal report: per frame ``flicker_ev`` (the HDR's mean exposure change
    minus the source's) and ``local_flicker_ev`` (the same per thumbnail cell, mean of |.|); frames
    that start a shot are not scored."""
    starts = set(int(c) for c in cuts) | {0}
    frames, previous = [], None
    for i, (sdr, hdr) in enumerate(zip(sdr_frames, hdr_network_frames)):
        exposure = (exposure_cells(source_luminance(sdr, transfer, primaries)), exposure_cells(hdr_luminance(hdr)))
        flicker = local = None
        if i not in starts and previous is not None:
            flicker = abs((float(np.mean(exposure[1])) - float(np.mean(previous[1])))
                          - (float(np.mean(exposure[0])) - float(np.mean(previous[0]))))
            local = float(np.mean(np.abs((exposure[1] - previous[1]) - (exposure[0] - previous[0]))))
        previous = exposure
        frames.append({"index": i, "flicker_ev": flicker, "local_flicker_ev": local, "cut": i in starts})
    scored = [f["flicker_ev"] for f in frames if f["flicker_ev"] is not None]
    local = [f["local_flicker_ev"] for f in frames if f["local_flicker_ev"] is not None]
    peak = max(scored) if scored else None
    return {"frames": frames, "scene_cuts": sorted(starts),
            "temporal": {"frames_scored": len(scored),
                         "flicker_ev_mean": sum(scored) / len(scored) if scored else None,
                         "flicker_ev_max": peak,
                         "flicker_ev_max_frame": next(f["index"] for f in frames if f["flicker_ev"] == peak)
                         if scored else None,
                         "local_flicker_ev_mean": sum(local) / len(local) if local else None,
                         "local_flicker_ev_max": max(local) if local else None}}


def retention_for(fps: float, seconds: float) -> float:
    """exp(-1 / (fps * seconds)); 0 (no hold) for 0 seconds."""
    return math.exp(-1.0 / (float(fps) * seconds)) if seconds > 0 and fps > 0 else 0.0


def mean3x3(plane):
    padded = np.pad(plane, 1, mode="edge")
    h, w = plane.shape
    out = np.zeros((h, w))
    for dy in range(3):
        for dx in range(3):
            out += padded[dy:dy + h, dx:dx + w]
    return out / 9.0


class StaticHold:
    """ShadowSmoother.hold_pixels: where the source pixel is still, the HDR's gain over the source is
    held across the shot; where it moved, or at a cut, the frame is its own. Inside highlights clipped
    in both frames the HDR's own change decides. Only the gain is held, never the pixel."""

    def __init__(self, retention: float):
        self.retention = float(retention)
        self.pixels = None

    def reset(self):
        self.pixels = None

    def factor(self, luma, peak, linear, yh):
        """The per-pixel gain factor for this frame (1 where nothing is held). ``yh``: the HDR's
        Rec.2020 luminance in network units."""
        yh = np.asarray(yh, np.float64)
        gain = (yh + STATIC_HDR_FLOOR) / (linear + STATIC_SDR_FLOOR)
        if self.retention == 0 or self.pixels is None:
            if self.retention:
                self.pixels = (luma, peak, gain, yh)
            return None
        previous, previous_peak, held, previous_yh = self.pixels
        t = np.clip((mean3x3(np.abs(luma - previous)) - STATIC_STILL) / (STATIC_MOVED - STATIC_STILL), 0.0, 1.0)
        hold = self.retention * (1.0 - t * t * (3.0 - 2.0 * t))
        both = np.minimum(peak, previous_peak) >= STATIC_CLIPPED
        if both.any():
            change = np.abs(np.log2(np.maximum(yh, 0.0) + STATIC_HDR_FLOOR)
                            - np.log2(np.maximum(previous_yh, 0.0) + STATIC_HDR_FLOOR))
            th = np.clip((mean3x3(change) - STATIC_HDR_STILL) / (STATIC_HDR_MOVED - STATIC_HDR_STILL), 0.0, 1.0)
            hold = np.where(both, self.retention * (1.0 - th * th * (3.0 - 2.0 * th)), hold)
        hold[((np.maximum(peak, previous_peak) >= STATIC_CLIPPED) & ~both)
             | (np.minimum(peak, previous_peak) <= STATIC_CRUSHED) | (yh < 0)] = 0.0
        held = hold * held + (1.0 - hold) * gain
        self.pixels = (luma, peak, held, yh)
        return held / gain

    def __call__(self, luma, peak, linear, hdr):
        k = LUMA_REC2020
        h64 = np.asarray(hdr, np.float64)
        f = self.factor(luma, peak, linear, h64[..., 0] * k[0] + h64[..., 1] * k[1] + h64[..., 2] * k[2])
        if f is None:
            return hdr
        return (h64 * f[..., None]).astype(np.float32)


def hold_sequence(sdr_frames, hdr_network_frames, cuts: Sequence[int], retention: float,
                  transfer="srgb", primaries="rec709") -> List[np.ndarray]:
    """The static hold over a sequence, reset at each cut (frame 0 always starts a shot)."""
    starts = set(int(c) for c in cuts) | {0}
    hold = StaticHold(retention)
    out = []
    for i, (sdr, hdr) in enumerate(zip(sdr_frames, hdr_network_frames)):
        if i in starts:
            hold.reset()
        luma, peak = source_code_luma(sdr, primaries)
        out.append(hold(luma, peak, source_luminance(sdr, transfer, primaries), hdr))
    return out


def parse_cut_data(text: str, frame_count: int) -> Optional[List[int]]:
    """Cuts from ◎ Scene Cut Detect's cut_data JSON ({"cuts": [...]}), a JSON list, or "0, 12, 40".
    Empty: None (detect)."""
    text = (text or "").strip()
    if not text:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            data = [int(t) for t in text.replace(";", ",").split(",") if t.strip()]
        except ValueError as exc:
            raise ValueError("cut_data must be Scene Cut Detect's cut_data, a JSON list or '0, 12, 40'") from exc
    if isinstance(data, dict):
        data = data.get("cuts", [])
    if not isinstance(data, list):
        raise ValueError("cut_data must hold a list of frame numbers")
    return sorted({0, *(int(c) for c in data if 0 <= int(c) < frame_count)})
