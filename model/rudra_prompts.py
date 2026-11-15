"""Prompt presets and a suggestion for RUDRA's AI fill (RUDRA 1.1, 5 Nov 2026).

Ported from FXTD-Studios/RUDRA ``rudra/genfill/prompts.py``, commit 74f03b6
(Apache 2.0), as it is except for reading files (Radiance hands frames in).

The fill takes an optional highlight and shadow prompt. Writing one from scratch
is guesswork, and a prompt that is not in the prompt cache loads umT5 (about
11 GB, on the CPU) once to embed it. So RUDRA offers a short list of PRESETS
and SUGGESTS one by looking at the shot's fully clipped areas, with no model:

    size and place     a large area at the top of the frame is a sky; a small round one a source
    shape              a filled rectangle with a dark surround is a window; many specks are glints
    the colour around  the ring of pixels just outside the area: blue, warm or neutral
    motion (several    a warm, irregular area whose size jumps from frame to frame is a flame
    frames given)

A suggestion is a starting point the artist can edit; every one says why it was
picked. Nothing here changes the fill: it only fills the prompt boxes.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Preset:
    id: str
    label: str
    text: str
    region: str          # 'highlight' or 'shadow': which prompt box it fills


PRESETS: tuple[Preset, ...] = (
    Preset("sky_blue", "Clear blue sky", "clear blue sky, smooth gradient toward the horizon, soft high clouds", "highlight"),
    Preset("sky_overcast", "Overcast sky", "bright overcast sky, soft even white light, subtle cloud texture", "highlight"),
    Preset("sky_sunset", "Sunset sky", "glowing sunset sky, warm orange to pink gradient, soft clouds", "highlight"),
    Preset("sun_disc", "Sun", "warm sun disc with a soft halo and a smooth glow", "highlight"),
    Preset("light_source", "Lamp / practical", "practical lamp, bright core with a soft falloff", "highlight"),
    Preset("window_daylight", "Window daylight", "window daylight, soft bright exterior, smooth falloff at the frame",
           "highlight"),
    Preset("specular_glints", "Specular glints", "small specular glints on glossy surfaces, sharp bright cores",
           "highlight"),
    Preset("flame", "Flame", "flame core, bright yellow center with an orange glow", "highlight"),
    Preset("shadow_detail", "Shadow detail", "dark shadow detail, soft texture, low light", "shadow"),
)
BY_ID = {p.id: p for p in PRESETS}

LARGE = 0.03          # a component covering at least this share of the frame is large
SOURCE = 0.05         # a round component below this share can be a single source (sun, lamp)
TINY = 0.003          # below this a component is a speck
GLINTS = 3            # this many specks make glints


def cache_key(text: str) -> str:
    """model.PromptBank.key, without importing torch: the prompt cache's file name."""
    return hashlib.sha256(text.strip().encode()).hexdigest()[:24]


def cached(text: str, cache: str | Path | None) -> bool:
    return bool(cache) and (Path(cache) / f"{cache_key(text)}.pt").is_file()


# ------------------------------------------------------------------ analysis
def components(codes: np.ndarray, tau: int = 254) -> list[dict]:
    """Fully clipped areas (max code >= tau) and what is around them."""
    import cv2
    h, w = codes.shape[:2]
    mask = (codes.max(-1) >= tau).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    rgb = codes.astype(np.float32) / 255.0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        a = area / float(h * w)
        comp = (lab == i).astype(np.uint8)
        ring = (cv2.dilate(comp, k) > 0) & (comp == 0)
        rc = rgb[ring].mean(0) if ring.any() else np.array([1.0, 1.0, 1.0], np.float32)
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        per = sum(cv2.arcLength(c, True) for c in contours)
        out.append({
            "area": a, "fill": area / float(max(1, bw * bh)), "aspect": bw / float(max(1, bh)),
            "top": y == 0, "cy": (cent[i][1]) / h,
            "circularity": float(4 * math.pi * area / (per * per)) if per > 0 else 1.0,
            "ring_rgb": [float(v) for v in rc], "ring_luma": float(rc @ np.array([0.2126, 0.7152, 0.0722])),
            "warmth": float(rc[0] - rc[2]), "blueness": float(rc[2] - rc[0]),
        })
    return out


def classify(c: dict) -> tuple[str, str]:
    """(preset id, why) for one clipped area of at least TINY."""
    pct = 100 * c["area"]
    # a filled rectangle in a dark surround, not cut by the frame's top: a window, before the sky test
    if (c["fill"] >= 0.8 and c["area"] >= 0.005 and 0.25 <= c["aspect"] <= 4 and c["ring_luma"] < 0.35
            and not c["top"]):
        return "window_daylight", f"a rectangular clipped area ({pct:.1f}%) with a dark surround"
    if c["area"] >= LARGE and (c["top"] or c["cy"] < 0.45):
        if c["blueness"] > 0.06:
            return "sky_blue", f"a large clipped area ({pct:.0f}% of the frame) at the top, blue around it"
        if c["warmth"] > 0.08:
            return "sky_sunset", f"a large clipped area ({pct:.0f}%) at the top, warm around it"
        return "sky_overcast", f"a large clipped area ({pct:.0f}%) at the top, neutral around it"
    if c["area"] < SOURCE and c["circularity"] > 0.6:
        if c["warmth"] > 0.08:
            return "sun_disc", f"a small round clipped area ({pct:.2f}%), warm around it"
        return "light_source", f"a small round clipped area ({pct:.2f}%), neutral around it"
    if c["warmth"] > 0.08 and c["circularity"] < 0.4:
        return "flame", f"an irregular clipped area ({pct:.2f}%), warm around it"
    return "light_source", f"a clipped area ({pct:.2f}%)"


def suggest(frames: list[np.ndarray], tau_high: int = 250, tau_low: int = -1) -> dict:
    """Ranked suggestions for SDR code frames (one is enough; several let a flame show itself).
    tau_high / tau_low: the fill's own thresholds (the clipped test is code >= max(tau_high, 254),
    the fully clipped core the fill blends into)."""
    if not frames:
        raise ValueError("no frames")
    tau = max(254, int(tau_high))
    votes: dict[str, dict] = {}

    def vote(pid, score, why):
        v = votes.setdefault(pid, {"score": 0.0, "why": why})
        v["score"] += score
        if score >= v.get("best", 0):
            v["best"], v["why"] = score, why
    warm_area = []
    clipped = []
    for codes in frames:
        comps = components(codes, tau)
        clipped.append(sum(c["area"] for c in comps))
        specks = [c for c in comps if c["area"] < TINY]
        if len(specks) >= GLINTS:
            vote("specular_glints", sum(c["area"] for c in specks) + 0.001 * len(specks),
                 f"{len(specks)} small clipped specks")
        warm = 0.0
        for c in comps:
            if c["area"] < TINY:
                continue
            pid, why = classify(c)
            vote(pid, c["area"], why)
            if c["warmth"] > 0.08:
                warm += c["area"]
        warm_area.append(warm)
    if len(frames) >= 3 and np.mean(warm_area) > 0:
        cv = float(np.std(warm_area) / np.mean(warm_area))
        if cv > 0.15:   # a warm area that changes size from frame to frame
            vote("flame", float(np.mean(warm_area)) * (1 + cv) * len(frames),   # scores are divided by n below
                 f"a warm clipped area whose size changes {100 * cv:.0f}% from frame to frame")
    if tau_low >= 0:
        crushed = float(np.mean([(f.max(-1) <= max(1, tau_low)).mean() for f in frames]))
        if crushed > 0.002:
            vote("shadow_detail", crushed, f"{100 * crushed:.1f}% of the frame is crushed to black")
    n = len(frames)
    ranked = sorted(votes.items(), key=lambda kv: -kv[1]["score"])
    return {"clipped_fraction": float(np.mean(clipped)),
            "suggestions": [{**asdict(BY_ID[pid]), "score": v["score"] / n, "reason": v["why"]} for pid, v in ranked]}
