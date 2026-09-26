"""
nodes_scene_cut.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Radiance v3 — Video-First Pipeline · Scene-Cut Detection

Nodes
─────
  RadianceSceneCutDetect    Detect shot boundaries in an IMAGE batch
  RadianceSceneCutSplit     Split a batch into per-shot sub-batches
  RadianceShotGradeRouter   Apply per-shot grade parameters via index lookup
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any

import numpy as np
import torch

log = logging.getLogger("radiance.scene_cut")


# ─────────────────────────────────────────────────────────────────────────────
# Core detection algorithm
# ─────────────────────────────────────────────────────────────────────────────

def _luminance(frames: np.ndarray) -> np.ndarray:
    """(B,H,W,3) → (B,) perceptual luminance per frame."""
    return (0.2126 * frames[:, :, :, 0] +
            0.7152 * frames[:, :, :, 1] +
            0.0722 * frames[:, :, :, 2]).mean(axis=(1, 2))


# ── Putting the two methods on one scale ─────────────────────────────────────
#
# `_histogram_diff` and `_edge_diff` measured different things in different
# units. The histogram distance is an L1 distance between two normalised
# distributions, so it spans 0..2 by construction and a hard cut sits somewhere
# above ~0.3. The edge distance was a mean absolute difference of gradient
# magnitudes, which in practice never left the bottom tenth of its range — and
# which moved with the footage's contrast rather than with the cut, so it had
# no fixed scale even on its own. That part is fixed in `_edge_diff` itself;
# what follows makes the two comparable to each other.
#
# One `threshold` widget therefore could not mean the same thing for both, and
# `combined` was worse than either: 0.6 * histogram + 0.4 * edge added two
# quantities with no common unit, so the nominal "40% edge" contribution was in
# reality a couple of percent, and the blend was a histogram detector wearing a
# hat.
#
# Each raw distance is now divided by the distance that method reports for a
# textbook hard cut, then squashed with r/(1+r) into a bounded confidence in
# [0, 1) where 0.5 means "as different as a hard cut". The squash is strictly
# monotonic, so nothing is clipped away and the score plot keeps its full
# dynamic range — unlike a min(x, 1) clamp, which would flatten every strong
# cut to the same bar.
#
# HISTOGRAM_CUT_REFERENCE is the cut point this node has always documented for
# that method, carried over unchanged.
#
# EDGE_CUT_REFERENCE is new. The tooltip's old figure of 0.15 could not survive
# `_edge_diff` becoming scale-free (see its docstring), and it did not describe
# the old metric either — measured across smooth, mid-detail, detailed and
# hard-edged synthetic plates, real cuts on the old absolute metric landed
# between 0.003 and 0.069, so 0.15 declared every cut a non-cut. 0.30 is the
# median relative distance between unrelated frames across those same plates,
# against a within-shot median of 0.07 for grain, small pans and exposure
# ramps.
#
# Both are the whole calibration and nothing else depends on their being right;
# recalibrating either is a one-line change with `TestTheScalesAreComparable`
# to check it against.
HISTOGRAM_CUT_REFERENCE = 0.30
EDGE_CUT_REFERENCE = 0.30

#: What `_confidence` returns when a raw score sits exactly on its reference.
CUT_CONFIDENCE_AT_REFERENCE = 0.5


def _confidence(raw: float, reference: float) -> float:
    """Map a raw method-specific distance onto the common 0..1 cut confidence.

    Returns exactly `CUT_CONFIDENCE_AT_REFERENCE` when ``raw == reference``,
    approaches 1.0 asymptotically, and is strictly increasing in *raw*.
    """
    r = max(float(raw), 0.0) / reference
    return r / (1.0 + r)


def _histograms(frame: np.ndarray, bins: int = 64) -> list[np.ndarray]:
    """One frame's normalised per-channel histograms, the input `_histogram_diff` compares."""
    out = []
    for c in range(3):
        h, _ = np.histogram(frame[:, :, c].ravel(), bins=bins, range=(0.0, 1.0))
        out.append(h.astype(np.float32) / (h.sum() + 1e-8))
    return out


def _histogram_distance(ha: list[np.ndarray], hb: list[np.ndarray]) -> float:
    score = 0.0
    for c in range(3):
        score += float(np.abs(ha[c] - hb[c]).sum())
    return score / 3.0


def _histogram_diff(a: np.ndarray, b: np.ndarray, bins: int = 64) -> float:
    """Histogram intersection distance between two frames (0=identical, 2=opposite)."""
    return _histogram_distance(_histograms(a, bins), _histograms(b, bins))


#: Box-blur width applied to luma before the gradient. The gradient operator is
#: a high-pass filter and film grain is high-frequency, so without this the
#: metric measures grain more than it measures content. 5 px is the smallest
#: width that brought grain below the level of a real cut in testing; wider
#: keeps helping but starts to miss cuts between similar framings.
EDGE_PREBLUR = 5


def _box_blur(gray: np.ndarray, k: int) -> np.ndarray:
    """k×k box blur via a summed-area table — O(1) per pixel, no scipy."""
    if k <= 1:
        return gray
    pad = k // 2
    padded = np.pad(gray, pad, mode="edge")
    integral = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    integral = np.pad(integral, ((1, 0), (1, 0)))
    out = (
        integral[k:, k:] - integral[:-k, k:] - integral[k:, :-k] + integral[:-k, :-k]
    )
    return out[: gray.shape[0], : gray.shape[1]] / float(k * k)


def _edge_map(img: np.ndarray) -> np.ndarray:
    """Gradient magnitude of the (blurred) luma. Luma only, for speed."""
    gray = (0.2126 * img[:, :, 0] +
            0.7152 * img[:, :, 1] +
            0.0722 * img[:, :, 2])
    gray = _box_blur(gray, EDGE_PREBLUR)
    # Simple finite-difference approximation
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    return np.sqrt(gx ** 2 + gy ** 2)


def _edge_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Relative difference between two frames' edge maps, in [0, 1].

    Catches hard cuts that do not change average colour — the same palette
    reframed, which the histogram method is blind to.

    This used to be `mean(|edge_a − edge_b|)`, an *absolute* difference, and
    that made a fixed threshold impossible in a way no choice of constant could
    repair: the score scales with the frames' own contrast and texture energy,
    not with how different they are. Measured on one synthetic cut, graded
    down:

        contrast   absolute   relative
          100%       0.0369     0.2950
           50%       0.0185     0.2950
           25%       0.0092     0.2950

    The same cut, four times fainter, on the same footage. So a threshold tuned
    on a bright, detailed exterior found nothing in a dim interior, and grain
    on a detailed frame (0.039) outscored a genuine cut in soft content
    (0.003).

    Dividing by the total edge energy of both frames — a Bray–Curtis
    dissimilarity — removes the dependence exactly, as the third column shows,
    and bounds the result in [0, 1] with 0 for identical frames.

    Known limitation, and the reason `combined` is the default rather than
    `edge`: this method still cannot see a cut between two frames that both
    lack edges (a soft gradient cutting to a different soft gradient, or black
    cutting to white), and heavy grain still moves it more than a subtle cut
    does. `KNOWN_ISSUES.md` carries the details.
    """
    return _edge_distance(_edge_map(a), _edge_map(b))


def _edge_distance(mag_a: np.ndarray, mag_b: np.ndarray) -> float:
    """`_edge_diff` on two frames' precomputed edge maps."""
    total = float((mag_a + mag_b).sum())
    if total <= 1e-9:
        # Two perfectly flat frames. They may still differ in level, which is
        # the histogram method's job, not this one's.
        return 0.0
    return float(np.abs(mag_a - mag_b).sum() / total)


def detect_cuts(
    frames: np.ndarray,
    threshold: float = CUT_CONFIDENCE_AT_REFERENCE,
    min_shot_frames: int = 8,
    method: str = "histogram",
) -> tuple[list[int], np.ndarray]:
    """
    Detect hard cuts in a frame sequence.

    Parameters
    ──────────
    frames          (B, H, W, 3) float32 [0, 1]
    threshold       cut confidence above which a cut is declared, on the common
                    0..1 scale every method now reports. 0.5 is "as different
                    as a textbook hard cut" for whichever method is selected;
                    lower is more sensitive. The same number means the same
                    thing for `histogram`, `edge` and `combined`, which is what
                    makes `combined` a real blend rather than a sum of
                    incompatible units.

                    Two earlier meanings this parameter has had, neither of
                    them this one: a fraction of the clip's own maximum (so a
                    cut-free clip always reported a cut), and a raw
                    method-specific distance (so the useful value differed by a
                    factor of two between methods).
    min_shot_frames minimum frames between two cut points
    method          "histogram" | "edge" | "combined"

    Returns
    ───────
    cut_frames      list of frame indices where a new shot begins (always includes 0)
    scores          (B-1,) per-frame cut confidence in [0, 1)
    """
    B = frames.shape[0]
    if B < 2:
        return [0], np.zeros(max(B - 1, 1), dtype=np.float32)

    # 3.5.0: each frame's histograms and edge map are computed once and kept
    # for the next pair; every inner frame used to be analysed twice, once as
    # the later frame of a pair and once as the earlier. Same numbers.
    use_hist = method != "edge"
    use_edge = method != "histogram"

    def features(i: int):
        f = frames[i]
        return (_histograms(f) if use_hist else None,
                _edge_map(f) if use_edge else None)

    scores = np.zeros(B - 1, dtype=np.float32)
    prev = features(0)
    for i in range(B - 1):
        cur = features(i + 1)
        if method == "histogram":
            scores[i] = _confidence(_histogram_distance(prev[0], cur[0]), HISTOGRAM_CUT_REFERENCE)
        elif method == "edge":
            scores[i] = _confidence(_edge_distance(prev[1], cur[1]), EDGE_CUT_REFERENCE)
        else:  # combined — both terms are now confidences, so the weights mean
               # what they say.
            scores[i] = (
                0.6 * _confidence(_histogram_distance(prev[0], cur[0]), HISTOGRAM_CUT_REFERENCE)
                + 0.4 * _confidence(_edge_distance(prev[1], cur[1]), EDGE_CUT_REFERENCE)
            )
        prev = cur

    # Compare the calibrated confidence to the threshold.
    #
    # This used to divide by scores.max() first, which forced the largest
    # inter-frame delta in every batch to exactly 1.0 before comparing. Two
    # consequences: `threshold` had no fixed meaning — the same value meant
    # something different for every clip — and cut-free footage always
    # reported a cut, because after normalisation its biggest ripple sat at
    # 1.0 no matter how small it really was. The scaling above is per-method
    # and fixed, not per-clip, so neither returns.
    #
    # `_make_plot` was already comparing the raw scores to the threshold, so
    # the rendered plot and the returned cut list could disagree with each
    # other on the same input.
    raw_cuts = [i + 1 for i in range(B - 1) if scores[i] >= threshold]
    cuts = [0]
    for c in raw_cuts:
        if c - cuts[-1] >= min_shot_frames:
            cuts.append(c)

    return cuts, scores


# ─────────────────────────────────────────────────────────────────────────────
# Node 1 — RadianceSceneCutDetect
# ─────────────────────────────────────────────────────────────────────────────

class RadianceSceneCutDetect:
    """
    Detect hard shot cuts in a video sequence batch.

    Outputs a JSON list of cut frame indices and a diagnostic plot tensor.
    The cut_data STRING connects directly to RadianceSceneCutSplit.
    """

    CATEGORY = "FXTD STUDIOS/Radiance/◎ VFX"
    DESCRIPTION = "Automatically detect scene cuts in a video using visual change metrics."
    FUNCTION     = "detect"
    RETURN_TYPES = ("STRING", "INT",   "IMAGE")
    RETURN_NAMES = ("cut_data", "shot_count", "score_plot")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Full video sequence as IMAGE batch.",
                }),
                "cut_confidence": ("FLOAT", {
                    "default": 0.5, "min": 0.01, "max": 0.99, "step": 0.01,
                    "tooltip": (
                        "How different two frames must be to count as a cut, on "
                        "a 0–1 scale that means the same thing for every method. "
                        "0.5 is a textbook hard cut; lower is more sensitive. "
                        "Fixed, not relative to the clip, so cut-free footage "
                        "reports no cuts."
                    ),
                }),
                "min_shot_frames": ("INT", {
                    "default": 12, "min": 1, "max": 500,
                    "tooltip": "Minimum frames between detected cuts.",
                }),
                "method": (["histogram", "edge", "combined"], {
                    "default": "combined",
                    "tooltip": (
                        "histogram: colour distribution diff (fast). "
                        "edge: Sobel edge map diff (catches content cuts). "
                        "combined: weighted blend of both (recommended)."
                    ),
                }),
            },
        }

    # ── score plot ────────────────────────────────────────────────────────────

    @staticmethod
    def _make_plot(scores: np.ndarray, cuts: list[int], threshold: float) -> torch.Tensor:
        """Render a 512×128 plot of inter-frame scores as an IMAGE tensor."""
        PW, PH = 512, 128
        plot = np.ones((PH, PW, 3), dtype=np.float32) * 0.12  # dark background

        B = len(scores)
        if B == 0:
            return torch.from_numpy(plot[None])

        # Score bars
        for i, s in enumerate(scores):
            x = int(i / B * PW)
            bar_h = int(min(s, 1.0) * (PH - 16))
            col = [0.2, 0.6, 1.0] if s < threshold else [1.0, 0.3, 0.2]
            plot[PH - bar_h - 1: PH - 1, x: x + max(1, PW // B), :] = col

        # Threshold line
        ty = PH - int(threshold * (PH - 16)) - 1
        plot[ty: ty + 1, :, :] = [1.0, 0.8, 0.0]

        # Cut markers
        for c in cuts:
            if c > 0:
                cx = int((c - 1) / B * PW)
                plot[:, cx: cx + 2, :] = [0.0, 1.0, 0.4]

        return torch.from_numpy(plot[None])  # (1, H, W, 3)

    def detect(
        self,
        images: torch.Tensor,
        cut_confidence: float,
        min_shot_frames: int,
        method: str,
    ):
        # Renamed twice during 3.3.0, both times deliberately: `threshold` →
        # `distance_threshold` → `cut_confidence`.
        #
        # The value's meaning changed each time. It was a fraction of the
        # clip's own maximum; then an absolute inter-frame distance, whose
        # scale still differed by method; it is now a method-independent
        # confidence in 0..1. A widget that kept its name across any of those
        # changes would have carried a saved number into a different meaning
        # and quietly detected different cuts, with nothing on screen to show
        # it. ComfyUI drops a stored value whose key no longer exists, so the
        # rename turns a silent reinterpretation into a visible reset to the
        # new default, which the user can see and re-tune.
        #
        # 3.3.0 has only ever been a beta, so `distance_threshold` never
        # reached a stable release and the second rename costs nobody a
        # migration.
        threshold = cut_confidence
        frames = images.detach().cpu().float().numpy()  # (B,H,W,3)
        cuts, scores = detect_cuts(frames, threshold, min_shot_frames, method)
        shot_count = len(cuts)

        # Build shot table: [{shot_idx, start_frame, end_frame, length}, ...]
        shots = []
        for i, start in enumerate(cuts):
            end = cuts[i + 1] - 1 if i + 1 < len(cuts) else frames.shape[0] - 1
            shots.append({
                "shot": i,
                "start": start,
                "end":   end,
                "length": end - start + 1,
            })

        cut_data = json.dumps({
            "cuts":       cuts,
            "shots":      shots,
            "shot_count": shot_count,
            "threshold":  threshold,
            "method":     method,
            "total_frames": frames.shape[0],
        })

        log.info("SceneCutDetect: %d shots detected in %d frames",
                 shot_count, frames.shape[0])
        for s in shots:
            log.debug("  Shot %02d: frames %d–%d (%d frames)",
                      s["shot"], s["start"], s["end"], s["length"])

        plot = self._make_plot(scores, cuts, threshold)
        return (cut_data, shot_count, plot)


# ─────────────────────────────────────────────────────────────────────────────
# Node 2 — RadianceSceneCutSplit
# ─────────────────────────────────────────────────────────────────────────────

class RadianceSceneCutSplit:
    """
    Split an IMAGE batch into per-shot sub-batches using cut_data from
    RadianceSceneCutDetect.

    OUTPUT_IS_LIST mode: each output slot carries one shot's frames.
    Use shot_index to select a specific shot instead.
    """

    CATEGORY = "FXTD STUDIOS/Radiance/◎ VFX"
    DESCRIPTION = "Split a video at detected scene cut points into discrete segments."
    FUNCTION     = "split"
    RETURN_TYPES = ("IMAGE",  "INT",  "INT",   "INT",   "STRING")
    RETURN_NAMES = ("frames", "shot_index", "start_frame", "end_frame", "shot_info")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images":    ("IMAGE", {
                    "tooltip": "The same frame batch that was analysed by RadianceSceneCutDetect. "
                               "Returned whole if cut_data has no shots.",
                }),
                "cut_data":  ("STRING", {
                    "tooltip": "JSON from RadianceSceneCutDetect.",
                }),
                "shot_index": ("INT", {
                    "default": 0, "min": 0, "max": 9999,
                    "tooltip": "Which shot to extract (0-based). "
                               "Connect shot_count output to know the range.",
                }),
            },
        }

    def split(
        self,
        images: torch.Tensor,
        cut_data: str,
        shot_index: int,
    ):
        # cut_data is a STRING widget with no default, so it arrives as "" for
        # anyone who drops the node and queues before wiring the detector --
        # json.loads("") raised a bare JSONDecodeError traceback. Fail with
        # something that says what to connect instead.
        try:
            data = json.loads(cut_data) if cut_data and cut_data.strip() else {}
        except json.JSONDecodeError as exc:
            raise ValueError(
                "RadianceSceneCutSplit: `cut_data` is not valid JSON "
                f"({exc.msg} at position {exc.pos}). Connect the `cut_data` "
                "output of RadianceSceneCutDetect to this input."
            ) from exc

        if not isinstance(data, dict):
            raise ValueError(
                "RadianceSceneCutSplit: `cut_data` must be a JSON object from "
                f"RadianceSceneCutDetect, got {type(data).__name__}."
            )

        shots = data.get("shots") or []

        if not shots:
            return (images, 0, 0, images.shape[0] - 1,
                    json.dumps({"error": "no shots in cut_data"}))

        idx   = max(0, min(shot_index, len(shots) - 1))
        shot  = shots[idx]
        start = shot["start"]
        end   = shot["end"] + 1   # exclusive for slice

        shot_frames = images[start:end]
        info = json.dumps({
            "shot_index":  idx,
            "start_frame": shot["start"],
            "end_frame":   shot["end"],
            "length":      shot["length"],
            "total_shots": len(shots),
        })

        log.info("SceneCutSplit: shot %d — frames %d–%d (%d frames)",
                 idx, shot["start"], shot["end"], shot["length"])
        return (shot_frames, idx, shot["start"], shot["end"], info)


# ─────────────────────────────────────────────────────────────────────────────
# Node 3 — RadianceShotGradeRouter
# ─────────────────────────────────────────────────────────────────────────────

class RadianceShotGradeRouter:
    """
    Per-shot grade parameter lookup table.

    Store a JSON array of grade entries keyed by shot index.
    Outputs the correct exposure / temperature / saturation floats
    for the current shot_index so downstream grade nodes receive
    shot-specific parameters without manual intervention.

    Grade table format (JSON array, one entry per shot):
    [
      {"exposure": 0.0, "temperature": 0, "saturation": 1.0, "contrast": 1.0},
      {"exposure": 0.3, "temperature": -200, "saturation": 0.9, "contrast": 1.1},
      ...
    ]
    Missing entries fall back to the last defined entry.
    """

    CATEGORY = "FXTD STUDIOS/Radiance/◎ VFX"
    DESCRIPTION = "Route frames to per-shot grade nodes based on scene cut metadata."
    FUNCTION     = "route"
    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT", "FLOAT", "INT")
    RETURN_NAMES = ("exposure", "temperature", "saturation", "contrast", "shot_index")

    # Defaults applied when a key is absent in the table entry
    _DEFAULTS = {
        "exposure":    0.0,
        "temperature": 0.0,
        "saturation":  1.0,
        "contrast":    1.0,
    }

    @classmethod
    def INPUT_TYPES(cls):
        default_table = json.dumps([
            {"exposure": 0.0, "temperature": 0, "saturation": 1.0, "contrast": 1.0},
            {"exposure": 0.3, "temperature": -150, "saturation": 0.9, "contrast": 1.1},
        ], indent=2)
        return {
            "required": {
                "shot_index": ("INT", {
                    "default": 0, "min": 0, "max": 9999,
                    "tooltip": "Current shot index from SceneCutSplit.",
                }),
                "grade_table": ("STRING", {
                    "default": default_table,
                    "multiline": True,
                    "tooltip": "JSON array of per-shot grade parameters.",
                }),
            },
        }

    def route(self, shot_index: int, grade_table: str):
        try:
            table = json.loads(grade_table)
        except json.JSONDecodeError as exc:
            log.warning("ShotGradeRouter: invalid JSON (%s) — using defaults", exc)
            table = []

        # Clamp to last entry if index out of range
        idx   = min(shot_index, len(table) - 1) if table else -1
        entry = table[idx] if idx >= 0 else {}

        exposure    = float(entry.get("exposure",    self._DEFAULTS["exposure"]))
        temperature = float(entry.get("temperature", self._DEFAULTS["temperature"]))
        saturation  = float(entry.get("saturation",  self._DEFAULTS["saturation"]))
        contrast    = float(entry.get("contrast",    self._DEFAULTS["contrast"]))

        log.debug("ShotGradeRouter: shot %d → EV%.2f  temp%.0f  sat%.2f  con%.2f",
                  shot_index, exposure, temperature, saturation, contrast)
        return (exposure, temperature, saturation, contrast, shot_index)


# ─────────────────────────────────────────────────────────────────────────────
# Registration
# ─────────────────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "RadianceSceneCutDetect":  RadianceSceneCutDetect,
    "RadianceSceneCutSplit":   RadianceSceneCutSplit,
    "RadianceShotGradeRouter": RadianceShotGradeRouter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceSceneCutDetect":  "◎ Radiance Scene Cut Detect",
    "RadianceSceneCutSplit":   "◎ Radiance Scene Cut Split",
    "RadianceShotGradeRouter": "◎ Radiance Shot Grade Router",
}
