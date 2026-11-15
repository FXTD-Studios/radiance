"""
◎ Radiance SDR → HDR Analytic (RUDRA)
═════════════════════════════════════

RUDRA's model-free reconstruction, as the RUDRA desktop app runs it by
default: no network, no download, nothing to install.

* **Source curve** — how the SDR was made. The master starts from the exact
  inverse of that curve, diffuse white at 203 nits. BT.1886 (graded and
  broadcast video) is RUDRA's default and scored best on real SDR; BT.1886
  shadows for blacks that look crushed; ACES, Hable or AgX for CG renders and
  games; camera log for a log clip viewed through a LUT; plain clip for video
  simply clipped at white.
* **Clipped highlights** — what goes where the SDR clipped: a channel that
  clipped while the others did not is rebuilt from them, a blown sky gets a
  smooth fill from its edge, a lamp or the sun a peak and a falloff from the
  glow around it. Nothing is made up as texture. ``clip_peaks`` sets a
  region's peak in nits.
* **Highlight detail** — amplifies the structure that survives just under
  white (a sky, a window, an open door); 1 is off, 1.5–2 is the range to judge.
* **Anchor** — mid-tones stay where the SDR had them, read through the source
  curve; above ``anchor_knee`` the reconstruction takes over.
* **Exposure** — moves the whole master, part of the grade.

Masks out: ``filled_mask`` (the fill or the detail gain changed the pixel) and
``estimated_mask`` (every channel clipped: the value is an extrapolation, not
recovered). ``mattes`` carries RUDRA 1.1's mattes and per-pixel classes
(preserved, follows, reconstructed, estimated, generated) to ◎ RUDRA Master
Write; ``regions`` is the class picture to look at. For SDR made through ACES 2.0, SDR → HDR Universal / Recover with
RUDRA ACES 2 is the learned alternative.

Ported from FXTD-Studios/RUDRA (Apache 2.0), radiance.model.rudra_analytic.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import subprocess

import numpy as np
import torch

import comfy.utils

from radiance.color.ops import (
    M_REC709_TO_ACES2065_1,
    M_REC709_TO_ACESCG,
    M_REC709_TO_BT2020,
    apply_matrix_3x3,
    linear_to_hlg_bt2100,
    linear_to_pq_bt2408,
)
from radiance.model.rudra_still import PRIMARIES_NAME, matrix_to_rec709
from radiance.model.rudra_analytic import (
    DEFAULT_REGION_BANDS,
    BRACKET_STOPS,
    CONFIDENCE_NOTE,
    DEFAULT_SOURCE_CURVE,
    DIFFUSE_WHITE_NITS,
    MATTES_NOTE,
    PIXEL_CLASSES,
    SOURCE_CURVE_LABELS,
    AnalyticOptions,
    Region,
    RegionBand,
    analytic_master,
    apply_region_ev,
    class_share_pct,
    code_delta,
    exposure_bracket,
    fit_reference,
    parse_calibration,
    parse_clip_peaks,
    regions_picture,
)

logger = logging.getLogger("radiance.nodes.hdr.rudra_analytic")

_LABEL_TO_ID = {label: cid for cid, label in SOURCE_CURVE_LABELS.items()}
_CURVE_CHOICES = list(SOURCE_CURVE_LABELS.values())
_PRIMARIES_CHOICES = {"Rec.709 / sRGB": "rec709", "Display P3": "p3d65", "Rec.2020": "rec2020",
                      "Adobe RGB (1998)": "adobe_rgb"}
_ENCODINGS = ["Linear (Rec.709)", "Linear ACEScg (AP1)", "Linear ACES2065-1 (AP0)", "PQ (HDR10)", "HLG"]


def _plane(mask, i, h, w):
    """Frame i of a ComfyUI MASK (B, H, W) or (H, W) at h x w, bilinear when the size differs; None for none."""
    if mask is None:
        return None
    m = mask if mask.dim() == 3 else mask.unsqueeze(0)
    a = m[min(i, m.shape[0] - 1)].float().cpu().numpy()
    if a.shape != (h, w):
        import cv2
        a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(a, 0.0, 1.0).astype(np.float32)


def _soft_peak(rgb: torch.Tensor, peak: float, knee_ratio: float = 0.9) -> torch.Tensor:
    """Max-channel shoulder to ``peak`` (linear units), one gain per pixel so the hue holds."""
    m = rgb.amax(dim=-1, keepdim=True)
    knee = knee_ratio * peak
    over = (m - knee).clamp(min=0.0)
    span = max(peak - knee, 1e-6)
    target = torch.where(m > knee, knee + span * (1.0 - torch.exp(-over / span)), m)
    return rgb * (target / m.clamp(min=1e-9))


def encode(linear709: torch.Tensor, encoding: str, peak_nits: float, white_nits: float) -> torch.Tensor:
    """Deliver the master (scene-linear Rec.709, 1.0 = diffuse white) in Radiance's convention."""
    def m(t):
        return t.to(dtype=linear709.dtype, device=linear709.device)
    if encoding == "Linear ACEScg (AP1)":
        return apply_matrix_3x3(linear709, m(M_REC709_TO_ACESCG))
    if encoding == "Linear ACES2065-1 (AP0)":
        return apply_matrix_3x3(linear709, m(M_REC709_TO_ACES2065_1))
    if encoding in ("PQ (HDR10)", "HLG"):
        delivery = apply_matrix_3x3(linear709, m(M_REC709_TO_BT2020)).clamp(min=0.0)
        delivery = _soft_peak(delivery, float(peak_nits) / float(white_nits))
        if encoding == "HLG":
            return linear_to_hlg_bt2100(delivery, reference_white_nits=float(white_nits))
        return linear_to_pq_bt2408(delivery, peak_nits=float(peak_nits), reference_white_nits=float(white_nits))
    return linear709


class RadianceSDRToHDRAnalytic:
    """SDR → HDR with RUDRA's analytic route: the inverse of the source curve plus the clip fill."""

    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    DESCRIPTION = (
        "RUDRA's model-free SDR→HDR (the RUDRA app's default route): the exact inverse of the curve "
        "that made the SDR (BT.1886, BT.1886 shadows, ACES, Hable, AgX, camera log, plain clip), "
        "clipped highlights rebuilt from what survived (partial channels, smooth sky fill, lamp and "
        "sun peaks), Highlight detail, mid-tones anchored to the SDR, master exposure. No model, no "
        "download. Diffuse white = 1.0 (203 nits)."
    )
    FUNCTION = "convert"
    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "STRING", "RUDRA_MATTES", "IMAGE")
    RETURN_NAMES = ("image", "filled_mask", "estimated_mask", "report", "mattes", "regions")
    OUTPUT_TOOLTIPS = (
        "The HDR master in output_encoding.",
        "1 where the clip fill or the detail gain changed the pixel.",
        "1 where every channel clipped: the value is an extrapolation.",
        "What the conversion did, in words.",
        "RUDRA 1.1 mattes and pixel classes per frame, before the grade: for ◎ RUDRA Master Write.",
        "The pixel classes over the picture's grey: reconstructed cyan, estimated magenta, region edge white.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "SDR still, batch or video frames, sRGB-encoded 0..1 "
                                               "(a ComfyUI IMAGE as decoded)."}),
                "source_curve": (_CURVE_CHOICES, {"default": SOURCE_CURVE_LABELS[DEFAULT_SOURCE_CURVE],
                    "tooltip": "How the SDR was made; the master is the inverse of this curve. BT.1886: graded "
                               "and broadcast video, generated video, RUDRA's default. BT.1886 shadows: try it when "
                               "the blacks look crushed (judge on an HDR display). ACES / Hable / AgX: CG and games "
                               "with that tone mapper. Camera log: a log clip viewed through a LUT. Plain clip: video "
                               "simply clipped at white."}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "How much of the clipped-highlight fill: 0 leaves the plain inverse (a flat ceiling "
                               "where the SDR clipped), 1 is RUDRA's default. Lower it where a highlight was only "
                               "just above the clip or for sparks such as fireworks."}),
                "clip_fill": ("BOOLEAN", {"default": True,
                    "tooltip": "Fill clipped highlights: partial channels from the ones that survived, a smooth fill "
                               "and dome for a blown sky, a peak and falloff for lamps and the sun. Off: the plain "
                               "inverse."}),
                "highlight_detail": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 4.0, "step": 0.05,
                    "tooltip": "Amplifies the structure that survives just under white (SDR luma 0.86-0.94 up): a "
                               "sky, a window, an open door. 1 is off; 1.5-2 is the range to judge. Needs clip_fill."}),
                "exposure_ev": ("FLOAT", {"default": 0.0, "min": -3.0, "max": 3.0, "step": 0.05,
                    "tooltip": "Moves the whole master brighter or darker, in stops (part of the grade, applied last)."}),
                "output_encoding": (_ENCODINGS, {"default": "Linear (Rec.709)",
                    "tooltip": "Linear: scene-linear, 1.0 = reference_white_nits (diffuse white). ACEScg / AP0: the "
                               "same in ACES primaries (a grading master). PQ / HLG: BT.2100 delivery code values."}),
            },
            "optional": {
                "clip_peaks": ("STRING", {"default": "", "multiline": True,
                    "tooltip": 'Set the peak of a clipped region: JSON, [{"x": 0.5, "y": 0.3, "nits": 2500}], x and y '
                               'across the frame 0..1 (a point within 4 px of the region picks it). Empty: none.'}),
                "anchor": ("BOOLEAN", {"default": True,
                    "tooltip": "Anchor to source exposure: mid-tones stay where the SDR had them (read through the "
                               "source curve); the reconstruction takes over above anchor_knee."}),
                "anchor_knee": ("FLOAT", {"default": 0.9, "min": 0.5, "max": 0.98, "step": 0.01,
                    "tooltip": "SDR code where the anchor lets go."}),
                "keep_source_colour": ("BOOLEAN", {"default": True,
                    "tooltip": "Carries the source's colour below the clip, so only clipped areas take the "
                               "reconstruction's hue."}),
                "settle_grain": ("BOOLEAN", {"default": True,
                    "tooltip": "Settles flat near-white areas so grain lifted by the expansion does not shimmer."}),
                "peak_nits": ("FLOAT", {"default": 1000.0, "min": 200.0, "max": 10000.0, "step": 50.0,
                    "tooltip": "PQ / HLG only: mastering peak; brighter values roll off to it with the hue kept."}),
                "reference_white_nits": ("FLOAT", {"default": 203.0, "min": 100.0, "max": 1000.0, "step": 1.0,
                    "tooltip": "Linear 1.0 = this many nits (BT.2408: 203). Diffuse white lands on it."}),
                "region": ("RUDRA_REGION", {
                    "tooltip": "From RUDRA Region: limit the reconstruction to a range of the SDR's brightness. "
                               "Every pixel outside it keeps the plain inverse, bit for bit."}),
                "calibration": ("RUDRA_CALIBRATION", {
                    "tooltip": "From RUDRA Calibrate: anchors (an SDR code and its nits) or a fit to a graded HDR "
                               "reference, applied over the source curve."}),
                "mask": ("MASK", {
                    "tooltip": "Where the reconstruction may act (white). Outside it (black) the master is the plain "
                               "inverse of the source curve, bit for bit. One mask for every frame, or one per frame."}),
                "source_primaries": (list(_PRIMARIES_CHOICES), {"default": "Rec.709 / sRGB",
                    "tooltip": "The SDR's primaries. RUDRA makes the master in them and converts at the end, so a "
                               "Display P3 photo keeps its colours. A connected `source` decides instead."}),
                "source": ("RUDRA_SOURCE", {"tooltip": "From ◎ Load Still (RUDRA): the still's own colour tag."}),
            },
        }

    @torch.no_grad()
    def convert(self, image, source_curve, strength, clip_fill, highlight_detail, exposure_ev,
                output_encoding, clip_peaks="", anchor=True, anchor_knee=0.9, keep_source_colour=True,
                settle_grain=True, peak_nits=1000.0, reference_white_nits=203.0, region=None,
                calibration=None, mask=None, source_primaries="Rec.709 / sRGB", source=None):
        curve = _LABEL_TO_ID.get(source_curve, source_curve)
        peaks = parse_clip_peaks(clip_peaks)
        cal = calibration or {}
        opt = AnalyticOptions(
            source_curve=curve, strength=float(strength), clip_fill=bool(clip_fill), clip_peaks=peaks,
            highlight_detail=float(highlight_detail), anchor=bool(anchor), anchor_knee=float(anchor_knee),
            carry_chroma=bool(keep_source_colour), settle_grain=bool(settle_grain), exposure_ev=float(exposure_ev),
            region=region, calibration=cal.get("points", ()), reference=cal.get("reference"),
        )
        prim = (source or {}).get("primaries") or _PRIMARIES_CHOICES.get(source_primaries, "rec709")
        to709 = None if prim == "rec709" else torch.from_numpy(matrix_to_rec709(prim).astype(np.float32))
        img = image if image.dim() == 4 else image.unsqueeze(0)
        frames = img.shape[0]
        out, filled, estimated, stats, pictures = [], [], [], [], []
        bar = comfy.utils.ProgressBar(frames) if frames > 1 else None
        for i in range(frames):
            frame = img[i].float().cpu().numpy()
            opt.limit = _plane(mask, i, frame.shape[0], frame.shape[1])
            r = analytic_master(frame[..., :3], opt)
            t = torch.from_numpy(r.image)
            if to709 is not None:   # the master is in the source's primaries (RUDRA): to Rec.709, then delivered
                t = t @ to709.T
            if r.mattes is not None:
                r.mattes.settings["source_space"] = prim
            t = encode(t, output_encoding, float(peak_nits), float(reference_white_nits))
            if frame.shape[-1] > 3:
                t = torch.cat([t, torch.from_numpy(frame[..., 3:])], dim=-1)
            out.append(t)
            filled.append(torch.from_numpy(r.filled))
            estimated.append(torch.from_numpy(r.estimated))
            pictures.append(torch.from_numpy(regions_picture(r.mattes).astype(np.float32) / 255.0))
            stats.append(r)
            if bar is not None:
                bar.update(1)
        report = self._report(curve, opt, stats, output_encoding)
        if prim != "rec709" or source:
            report += "\nsource: " + ((source or {}).get("note") or f"{PRIMARIES_NAME[prim]}: mastered from {PRIMARIES_NAME[prim]}")
        mattes = {"frames": [r.mattes for r in stats], "encoding": output_encoding}
        return (torch.stack(out).to(image.dtype), torch.stack(filled), torch.stack(estimated), report,
                mattes, torch.stack(pictures))

    @staticmethod
    def _report(curve, opt, results, encoding) -> str:
        frames = len(results)
        partial = sum(r.info.partial_pct for r in results) / max(frames, 1)
        full = sum(r.info.full_pct for r in results) / max(frames, 1)
        regions = [g for r in results for g in r.info.regions]
        big = sorted(regions, key=lambda g: g.pixels, reverse=True)[:8]
        lines = [
            "RUDRA analytic (no model)",
            f"source curve: {SOURCE_CURVE_LABELS.get(curve, curve)}",
            f"frames: {frames}; output: {encoding}",
            f"clip fill: {'on' if opt.clip_fill else 'off'} at strength {opt.strength:.2f}"
            + (f"; highlight detail x{opt.highlight_detail:.2f}" if opt.highlight_detail != 1.0 else ""),
            f"partly clipped pixels: {partial:.2f} %; fully clipped (estimated): {full:.2f} %",
            f"anchor: {'on, knee ' + format(opt.anchor_knee, '.2f') if opt.anchor else 'off'}; "
            f"exposure {opt.exposure_ev:+.2f} EV",
        ]
        if opt.region is not None:
            r = opt.region
            lines.append("region: " + (f"band {r.band_low:.2f}-{r.band_high:.2f} (softness {r.band_softness:.2f})"
                                       if r.band else
                                       f"highlights from {r.highlight_from:.2f} x{r.highlight_strength:.2f}, "
                                       f"shadows below {r.shadow_below:.2f} x{r.shadow_strength:.2f}")
                         + "; outside it the plain inverse")
            share = sum(float((x.region_weight > 0).mean()) for x in results if x.region_weight is not None) / max(frames, 1)
            lines.append(f"region reaches {100 * share:.1f} % of the frame")
        if results and results[0].curve_note:
            lines.append(results[0].curve_note)
        if opt.limit is not None:
            lines.append("mask: the reconstruction acts only inside it")
        for g in big:
            lines.append(f"  region {g.pixels} px ({'sky' if g.large else 'source'}) at ({g.x}, {g.y}): "
                         f"edge {g.edge_nits:.0f} nits, peak {g.peak_nits:.0f} nits"
                         + (" (set by clip_peaks)" if g.overridden else ""))
        shares = [r.mattes.share for r in results if r.mattes is not None]
        if shares:
            mean = [sum(x[k] for x in shares) / len(shares) for k in range(len(PIXEL_CLASSES))]
            lines.append("pixels: " + ", ".join(f"{name} {100 * v:.2f} %" for name, v in zip(PIXEL_CLASSES, mean)
                                                if name != "generated"))
        if opt.clip_peaks and not any(g.overridden for g in regions):
            lines.append("clip_peaks: no point landed within 4 px of a fully clipped region")
        return "\n".join(lines)


class RadianceRudraRegion:
    """RUDRA's 'Limit to a luminance region': where SDR → HDR Analytic may reconstruct."""

    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    DESCRIPTION = (
        "Limit RUDRA's reconstruction to a range of the SDR's brightness (Rec.709 luma 0..1). Highlights + "
        "shadows: each from its own level, with its own softness and strength. Band: only between two levels. "
        "Pixels outside the region keep the plain inverse, bit for bit. Wire into SDR → HDR Analytic's region."
    )
    FUNCTION = "build"
    RETURN_TYPES = ("RUDRA_REGION",)
    RETURN_NAMES = ("region",)

    @classmethod
    def INPUT_TYPES(cls):
        lvl = lambda d, tip: ("FLOAT", {"default": d, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": tip})
        soft = lambda d: ("FLOAT", {"default": d, "min": 0.0, "max": 0.5, "step": 0.01,
                                    "tooltip": "Width of the ramp, in SDR luma (0 is a hard edge)."})
        strength = lambda tip: ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": tip})
        return {
            "required": {
                "mode": (["Highlights + shadows", "Band"], {"default": "Highlights + shadows",
                    "tooltip": "Highlights + shadows: highlights above one level, shadows below another. Band: one "
                               "range of SDR brightness."}),
                "highlight_from": lvl(0.75, "Highlights reconstruct from this SDR luma up (ramp centre)."),
                "highlight_softness": soft(0.10),
                "highlight_strength": strength("How much of the highlight reconstruction (1 = as RUDRA does it). "
                                               "In Band mode it scales the band."),
                "shadow_below": lvl(0.15, "Shadows reconstruct below this SDR luma (ramp centre)."),
                "shadow_softness": soft(0.10),
                "shadow_strength": strength("How much of the shadow reconstruction. In Band mode it scales the band's "
                                            "shadow side."),
                "band_low": lvl(0.50, "Band mode: the band's low edge."),
                "band_high": lvl(0.90, "Band mode: the band's high edge."),
                "band_softness": soft(0.05),
            }
        }

    def build(self, mode, highlight_from, highlight_softness, highlight_strength, shadow_below, shadow_softness,
              shadow_strength, band_low, band_high, band_softness):
        r = Region(band=mode == "Band", highlight_from=float(highlight_from),
                   highlight_softness=float(highlight_softness), highlight_strength=float(highlight_strength),
                   shadow_below=float(shadow_below), shadow_softness=float(shadow_softness),
                   shadow_strength=float(shadow_strength), band_low=float(band_low), band_high=float(band_high),
                   band_softness=float(band_softness))
        if r.error():
            raise ValueError(f"RUDRA Region: {r.error()}")
        return (r,)


class RadianceRudraCalibrate:
    """RUDRA's Calibrate and Reference: correct the source curve through anchors or a graded HDR."""

    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    DESCRIPTION = (
        "Correct the source curve of SDR → HDR Analytic. Anchors: up to three points (black, an 18% grey card, a "
        "highlight), each an SDR code or a position on the frame, with the nits it should be; one anchor shifts the "
        "exposure, two or three bend the curve through them. Reference: fit the curve to a graded HDR of the same "
        "frame (scene-linear Rec.709, 1.0 = reference white); it replaces the anchors."
    )
    FUNCTION = "build"
    RETURN_TYPES = ("RUDRA_CALIBRATION", "STRING")
    RETURN_NAMES = ("calibration", "summary")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The SDR frames SDR → HDR Analytic converts (for positions and the "
                                               "reference fit)."}),
                "anchors": ("STRING", {"default": "", "multiline": True,
                    "tooltip": 'JSON, up to three: [{"code": 118, "nits": 30}] by SDR code, or [{"x": 0.4, "y": 0.6, '
                               '"nits": 2.5}] by position (0..1 across the frame; its code is the SDR luma there). '
                               'Ignored when a reference is connected.'}),
                "frame": ("INT", {"default": 0, "min": 0, "max": 100000,
                    "tooltip": "The frame of the batch the positions and the reference fit read."}),
            },
            "optional": {
                "reference": ("IMAGE", {"tooltip": "A graded HDR of the same frame: scene-linear Rec.709, 1.0 = "
                                                   "reference_white_nits, the SDR's size. Replaces the anchors."}),
                "reference_white_nits": ("FLOAT", {"default": 203.0, "min": 100.0, "max": 1000.0, "step": 1.0,
                    "tooltip": "What 1.0 means in the reference."}),
            },
        }

    def build(self, image, anchors, frame, reference=None, reference_white_nits=203.0):
        img = image if image.dim() == 4 else image.unsqueeze(0)
        i = min(int(frame), img.shape[0] - 1)
        sdr = img[i, ..., :3].float().cpu().numpy()
        if reference is not None:
            ref = reference if reference.dim() == 4 else reference.unsqueeze(0)
            r = ref[min(i, ref.shape[0] - 1), ..., :3].float().cpu().numpy() * (float(reference_white_nits) / 203.0)
            fit = fit_reference(sdr, r)
            summary = (f"reference fit on frame {i}: {fit.samples} samples over {fit.codes_seen} codes; residual "
                       f"mean {fit.residual_mean:.2f}, p95 {fit.residual_p95:.2f} stops")
            if fit.residual_p95 > 0.5:
                summary += " (over half a stop: the grade has a local key the curve cannot carry)"
            return ({"reference": fit, "points": ()}, summary)
        points = parse_calibration(anchors, sdr)
        if not points:
            return ({"points": ()}, "no anchors: the source curve as it is")
        summary = "anchors: " + ", ".join(f"code {p.code} = {p.nits:g} nits" for p in points)
        return ({"points": points}, summary)


class RadianceRudraRegionEV:
    """RUDRA's Region EV: push ranges of nits up or down, each optionally limited by a mask."""

    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    DESCRIPTION = (
        "RUDRA's Region EV grade on a scene-linear HDR image: up to three bands of absolute nits (Rec.2020 luma), "
        "each pushed up or down in stops with a soft edge, each optionally limited to a mask; the exposure first, "
        "the bands read the exposed picture. 0 EV everywhere changes nothing."
    )
    FUNCTION = "grade"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def INPUT_TYPES(cls):
        req = {
            "image": ("IMAGE", {"tooltip": "Scene-linear HDR (Rec.709 or ACES primaries), 1.0 = reference_white_nits."}),
            "exposure_ev": ("FLOAT", {"default": 0.0, "min": -6.0, "max": 6.0, "step": 0.05,
                                      "tooltip": "The whole picture, in stops, before the bands read it."}),
            "softness_stops": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 4.0, "step": 0.05,
                                         "tooltip": "How soft each band's edges are, in stops."}),
        }
        names = ("highlights", "speculars", "shadows")
        for n, (b, name) in enumerate(zip(DEFAULT_REGION_BANDS, names), start=1):
            req[f"band{n}_low_nits"] = ("FLOAT", {"default": b.low_nits, "min": 0.001, "max": 40000.0, "step": 0.01,
                                                  "tooltip": f"Band {n} ({name} by default): low edge, nits."})
            req[f"band{n}_high_nits"] = ("FLOAT", {"default": b.high_nits, "min": 0.002, "max": 40000.0, "step": 0.01,
                                                   "tooltip": f"Band {n}: high edge, nits."})
            req[f"band{n}_ev"] = ("FLOAT", {"default": 0.0, "min": -4.0, "max": 4.0, "step": 0.05,
                                            "tooltip": f"Band {n}: push in stops (0 = untouched)."})
        req["reference_white_nits"] = ("FLOAT", {"default": 203.0, "min": 100.0, "max": 1000.0, "step": 1.0,
                                                 "tooltip": "What 1.0 means in the image."})
        return {"required": req, "optional": {f"mask{n}": ("MASK", {"tooltip": f"Limit band {n} to this mask."})
                                              for n in (1, 2, 3)}}

    def grade(self, image, exposure_ev, softness_stops, band1_low_nits, band1_high_nits, band1_ev, band2_low_nits,
              band2_high_nits, band2_ev, band3_low_nits, band3_high_nits, band3_ev, reference_white_nits=203.0,
              mask1=None, mask2=None, mask3=None):
        bands = [RegionBand(band1_low_nits, band1_high_nits, band1_ev),
                 RegionBand(band2_low_nits, band2_high_nits, band2_ev),
                 RegionBand(band3_low_nits, band3_high_nits, band3_ev)]
        for n, b in enumerate(bands, 1):
            if not b.low_nits < b.high_nits:
                raise ValueError(f"Region EV band {n}: the low edge must be below the high edge")
        img = image if image.dim() == 4 else image.unsqueeze(0)
        white = float(reference_white_nits)
        out = []
        for i in range(img.shape[0]):
            f = img[i].float().cpu().numpy()
            h, w = f.shape[:2]
            masks = [_plane(m, i, h, w) for m in (mask1, mask2, mask3)]
            nits = apply_region_ev(f[..., :3].astype(np.float64) * white, bands, float(softness_stops), masks,
                                   float(exposure_ev))
            rgb = (nits / white).astype(np.float32)
            if f.shape[-1] > 3:
                rgb = np.concatenate([rgb, f[..., 3:]], -1)
            out.append(torch.from_numpy(rgb))
        return (torch.stack(out).to(image.dtype),)


# ── ◎ RUDRA Master Write: the master and what RUDRA writes beside it ────────
# deliver/src/master.cpp (RUDRA 1.1 priorities 3-5): <name>.exr, <name>.mattes.exr,
# <name>.regions.png, <name>.ev-2.png / .ev+0.png / .ev+2.png, <name>.json.

_SPACES = ["From the mattes", "Linear (Rec.709)", "Linear ACEScg (AP1)", "Linear ACES2065-1 (AP0)"]
_CHROMATICITIES = {   # rx, ry, gx, gy, bx, by, wx, wy
    "Linear (Rec.709)": (0.64, 0.33, 0.30, 0.60, 0.15, 0.06, 0.3127, 0.3290),
    "Linear ACEScg (AP1)": (0.713, 0.293, 0.165, 0.830, 0.128, 0.044, 0.32168, 0.33767),
    "Linear ACES2065-1 (AP0)": (0.7347, 0.2653, 0.0, 1.0, 0.0001, -0.077, 0.32168, 0.33767),
}
_CONTAINER_NAMES = {"Linear (Rec.709)": "scene-linear Rec.709", "Linear ACEScg (AP1)": "ACEScg (AP1)",
                    "Linear ACES2065-1 (AP0)": "ACES 2065-1 (AP0)"}
_TO_709 = {"Linear ACEScg (AP1)": M_REC709_TO_ACESCG, "Linear ACES2065-1 (AP0)": M_REC709_TO_ACES2065_1}


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_commit() -> str:
    """The Radiance checkout's commit, as RUDRA stamps its own build; 'unknown' outside a git checkout."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        out = subprocess.run(["git", "-C", root, "rev-parse", "--short=12", "HEAD"], capture_output=True,
                             text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _to_rec709(rgb: np.ndarray, space: str) -> np.ndarray:
    m = _TO_709.get(space)
    if m is None:
        return rgb
    inv = np.linalg.inv(m.double().cpu().numpy())
    return (rgb.astype(np.float64) @ inv.T).astype(np.float32)


def _write_png(path: str, rgb8: np.ndarray) -> None:
    from PIL import Image
    Image.fromarray(np.ascontiguousarray(rgb8), "RGB").save(path, compress_level=6)


def _write_exr(path: str, channels: dict, half: bool, attrs: dict, chromaticities=None) -> None:
    import OpenEXR
    header = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}
    if chromaticities is not None:
        header["chromaticities"] = tuple(float(v) for v in chromaticities)
    header.update({k: str(v) for k, v in attrs.items()})
    dt = np.float16 if half else np.float32
    part = path + ".part"   # a stop or a crash never leaves a short file under the finished name, as RUDRA
    OpenEXR.File(header, {k: np.ascontiguousarray(v, dt) for k, v in channels.items()}).write(part)
    os.replace(part, path)


def _read_exr_rgb(path: str) -> np.ndarray:
    import OpenEXR
    with OpenEXR.File(path, separate_channels=True) as f:
        ch = {k: np.asarray(c.pixels, np.float32) for k, c in f.channels().items()}
    return np.stack([ch["R"], ch["G"], ch["B"]], -1)


class RadianceRudraMasterWrite:
    """The master EXR plus RUDRA 1.1's mattes, regions picture, EV brackets and provenance sidecar."""

    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    DESCRIPTION = (
        "Writes an HDR master the way RUDRA delivers one: <name>.exr (stamped with rudra:* provenance), "
        "<name>.mattes.exr (rudra.highlight, shadow, changed, region, class, confidence, and rudra.generated "
        "split into highlight / shadow when an AI fill mask is connected), <name>.regions.png (what each pixel "
        "is, in colour), EV brackets at -2 / 0 / +2 EV read back from the EXR as written with a "
        "preview-to-export check, and <name>.json with source and checkpoint SHA-256, build, settings and "
        "the pixel-class shares."
    )
    FUNCTION = "write"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("master_path", "sidecar")
    OUTPUT_TOOLTIPS = ("The first frame's master EXR.", "The first frame's sidecar JSON, as written.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The HDR master, scene-linear, 1.0 = 203 nits (diffuse white). "
                                               "Written unchanged. PQ and HLG are refused: write those with "
                                               "the delivery writers."}),
                "filename_prefix": ("STRING", {"default": "rudra_master",
                    "tooltip": "File stem. Frames are <prefix>.<frame, 4 digits>.exr and the files beside them."}),
                "image_space": (_SPACES, {"default": _SPACES[0],
                    "tooltip": "What the image is. 'From the mattes' takes the Analytic node's output_encoding "
                               "(linear Rec.709 when no mattes are connected)."}),
                "bit_depth": (["16-bit Half Float", "32-bit Float"], {"default": "16-bit Half Float",
                    "tooltip": "The master's and the mattes' pixel type. A half master's brackets differ from "
                               "the preview by its rounding; the sidecar records by how much."}),
                "write_mattes": ("BOOLEAN", {"default": True,
                    "tooltip": "<name>.mattes.exr and <name>.regions.png, from the mattes input; without "
                               "it the sidecar records that none were written."}),
                "write_brackets": ("BOOLEAN", {"default": True,
                    "tooltip": "<name>.ev-2.png, .ev+0.png and .ev+2.png: SDR views with 203 nits at code "
                               "255 at 0 EV, from the EXR read back."}),
            },
            "optional": {
                "mattes": ("RUDRA_MATTES", {"tooltip": "From ◎ SDR → HDR Analytic (RUDRA)."}),
                "generated_mask": ("MASK", {"tooltip": "SDR → HDR AI Fill generated_mask: those pixels become "
                                                       "class 'generated' and rudra.generated is written."}),
                "source_path": ("STRING", {"default": "", "tooltip": "The SDR file the frame came from; its "
                                           "SHA-256 is recorded. Empty: not recorded."}),
                "checkpoint_path": ("STRING", {"default": "", "tooltip": "A model file used upstream (the AI "
                                               "fill's, say); its SHA-256 is recorded. Empty: none."}),
                "output_path": ("STRING", {"default": "", "tooltip": "Output folder. Empty: the ComfyUI "
                                           "output folder; relative: inside it; absolute: as it is."}),
                "frame_index": ("INT", {"default": 1, "min": 0, "max": 9999999,
                                        "tooltip": "The first frame's number in the file names."}),
            },
        }

    def write(self, image, filename_prefix, image_space, bit_depth, write_mattes, write_brackets,
              mattes=None, generated_mask=None, source_path="", checkpoint_path="", output_path="",
              frame_index=1):
        from radiance.config.constants import VERSION
        from radiance.path_utils import get_safe_output_dir, strip_path_quotes
        space = image_space
        if space == _SPACES[0]:
            space = mattes.get("encoding", "Linear (Rec.709)") if mattes else "Linear (Rec.709)"
        if space not in _CHROMATICITIES:
            raise ValueError(f"[RUDRA Master Write] the image is {space}; a master is scene-linear. Set the "
                             "Analytic node's output_encoding to a linear one, or write PQ / HLG with the "
                             "delivery writers")
        if write_mattes and not mattes:
            logger.warning("[RUDRA Master Write] write_mattes is on but no mattes are connected: the master, "
                           "brackets and sidecar are written without them (the sidecar says so)")
        if not filename_prefix or any(c in filename_prefix for c in '\\/:*?"<>|'):
            raise ValueError("[RUDRA Master Write] filename_prefix must be a plain name (no folders)")
        img = image if image.dim() == 4 else image.unsqueeze(0)
        frames = img.shape[0]
        mframes = list(mattes["frames"]) if mattes else []
        if mattes and len(mframes) not in (1, frames):
            raise ValueError(f"[RUDRA Master Write] {len(mframes)} frames of mattes for {frames} frames of image")
        half = "16" in bit_depth
        source_sha = ""
        if strip_path_quotes(source_path or ""):
            sp = strip_path_quotes(source_path)
            if not os.path.isfile(sp):
                raise ValueError(f"[RUDRA Master Write] source_path is not a file: {sp}")
            source_sha = _sha256_file(sp)
        ckpt_sha = ""
        cp = strip_path_quotes(checkpoint_path or "")
        if cp:
            if not os.path.isfile(cp):
                raise ValueError(f"[RUDRA Master Write] checkpoint_path is not a file: {cp}")
            ckpt_sha = _sha256_file(cp)
        try:
            import folder_paths
            base = folder_paths.get_output_directory()
        except Exception:
            import tempfile
            base = tempfile.gettempdir()
        out_dir = get_safe_output_dir(base, strip_path_quotes(output_path or ""), allow_absolute=True)
        os.makedirs(out_dir, exist_ok=True)
        app = f"Radiance {VERSION} (RUDRA analytic port)"
        commit = _build_commit()
        first = ("", "")
        bar = comfy.utils.ProgressBar(frames) if frames > 1 else None
        for i in range(frames):
            stem = os.path.join(out_dir, f"{filename_prefix}.{frame_index + i:04d}")
            m = mframes[min(i, len(mframes) - 1)] if mframes else None
            frame = img[i].float().cpu().numpy()
            gen = None
            if generated_mask is not None and m is not None:
                gen = _plane(generated_mask, i, frame.shape[0], frame.shape[1])
            side = self._write_frame(stem, frame, space, half, m, gen, write_mattes, write_brackets,
                                     source_sha, ckpt_sha, app, commit)
            if i == 0:
                first = (stem + ".exr", side)
            if bar is not None:
                bar.update(1)
        return first

    @staticmethod
    def _write_frame(stem, frame, space, half, m, gen, write_mattes, write_brackets, source_sha, ckpt_sha,
                     app, commit) -> str:
        name = os.path.basename(stem)
        rgb = frame[..., :3]
        alpha = frame[..., 3] if frame.shape[-1] > 3 else None
        lin709 = _to_rec709(rgb, space)
        nits_max = np.maximum(lin709.max(-1), 0.0) * DIFFUSE_WHITE_NITS
        maxcll = float(nits_max.max()) if nits_max.size else 0.0
        maxfall = float(nits_max.mean()) if nits_max.size else 0.0
        settings = dict(m.settings) if m is not None else {}
        attrs = {"software": app, "rudra:appVersion": app, "rudra:buildCommit": commit,
                 "rudra:tiled": "False", "rudra:maxCLL": f"{maxcll:.1f}", "rudra:maxFALL": f"{maxfall:.1f}",
                 "rudra:diffuseWhiteNits": "203"}
        if settings:
            attrs["rudra:reconstruction"] = settings.get("reconstruction", "analytic (no model)")
            attrs["rudra:sourceCurve"] = settings.get("source_curve", "")
            attrs["rudra:anchored"] = str(settings.get("anchored", False))
            attrs["rudra:chromaCarried"] = str(settings.get("chroma_carried", False))
            attrs["rudra:grainSettled"] = str(settings.get("grain_settled", False))
            attrs["rudra:sourceSpace"] = settings.get("source_space", "rec709")
            if settings.get("exposure_ev"):
                attrs["rudra:exposureEV"] = f"{settings['exposure_ev']:+.2f}"
        if source_sha:
            attrs["rudra:sourceSha256"] = source_sha
        if ckpt_sha:
            attrs["rudra:checkpointSha256"] = ckpt_sha
        if not half:
            attrs["rudra:sampleType"] = "float32"
        bracket_names = [f"{name}.ev{ev:+.0f}.png" for ev in BRACKET_STOPS]
        if write_brackets:
            attrs["rudra:brackets"] = ", ".join(bracket_names)
        do_mattes = write_mattes and m is not None
        if do_mattes:
            attrs["rudra:mattes"] = f"{name}.mattes.exr"
        channels = {"R": rgb[..., 0], "G": rgb[..., 1], "B": rgb[..., 2]}
        if alpha is not None:
            a = np.clip(alpha, 0.0, 1.0)   # OpenEXR associated alpha: RGB premultiplied, as RUDRA writes it
            channels = {"R": rgb[..., 0] * a, "G": rgb[..., 1] * a, "B": rgb[..., 2] * a, "A": a}
            attrs["rudra:alpha"] = "the source's straight alpha as A; RGB premultiplied by it (OpenEXR associated alpha)"
        _write_exr(stem + ".exr", channels, half, attrs, _CHROMATICITIES[space])

        side = {
            "maxcll_nits": round(maxcll, 1), "maxfall_nits": round(maxfall, 1),
            "peak_nits": round(maxcll, 1), "resolution": [int(rgb.shape[1]), int(rgb.shape[0])],
            "container": _CONTAINER_NAMES[space], "transfer": "linear", "diffuse_white_nits": 203.0,
            "sample_type": "half" if half else "float32",
            "tiled": False, "app_version": app, "build_commit": commit,
            "written": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        if settings:
            side["settings"] = settings
            side["source_curve"] = settings.get("source_curve")
            if settings.get("exposure_ev"):
                side["exposure_ev"] = settings["exposure_ev"]
        if ckpt_sha:
            side["checkpoint_sha256"] = ckpt_sha
        if source_sha:
            side["source_sha256"] = source_sha

        if write_brackets:
            back = _read_exr_rgb(stem + ".exr")
            if alpha is not None:
                a = np.clip(alpha, 0.0, 1.0)[..., None]
                back = np.where(a > 0, back / np.maximum(a, 1e-12), rgb)   # where A is 0 there is no colour
            exported = _to_rec709(back, space)
            worst, differing = 0, 0.0
            for ev, fname in zip(BRACKET_STOPS, bracket_names):
                from_file = exposure_bracket(exported, ev)
                d = code_delta(from_file, exposure_bracket(lin709, ev))
                worst, differing = max(worst, d[0]), max(differing, d[1])
                _write_png(os.path.join(os.path.dirname(stem), fname), from_file)
            side["brackets"] = {
                "files": bracket_names, "ev": list(BRACKET_STOPS),
                "view": "SDR, sRGB, Rec.709; diffuse white (203 nits) at code 255 at 0 EV; from the EXR as written",
                "preview_vs_export": {"max_code_delta": int(worst), "pixels_differing_pct": round(100.0 * differing, 3),
                                      "basis": "the same brackets from the picture in memory against the EXR read "
                                               "back; a half master differs by its half-float rounding"}}
        if write_mattes and m is None:
            side["mattes"] = None
            side["mattes_note"] = "write_mattes was on but no mattes were connected (◎ SDR → HDR Analytic (RUDRA))"
        if do_mattes:
            if m.cls.shape != rgb.shape[:2]:
                raise ValueError(f"[RUDRA Master Write] the mattes are {m.cls.shape[1]}x{m.cls.shape[0]}; the "
                                 f"image is {rgb.shape[1]}x{rgb.shape[0]}")
            planes = m.channels(gen)
            _write_exr(stem + ".mattes.exr", planes, half,
                       {"rudra:mattesOf": f"{name}.exr", "rudra:mattes": MATTES_NOTE, "rudra:appVersion": app})
            _write_png(stem + ".regions.png", regions_picture(m, gen))
            side["mattes"] = {"file": f"{name}.mattes.exr", "channels": list(planes),
                              "basis": "before the grade (no Region EV, no exposure)"}
            side["pixel_classes"] = {"picture": f"{name}.regions.png", **class_share_pct(m, gen),
                                     "confidence": CONFIDENCE_NOTE}
        text = json.dumps(side, indent=2)
        part = stem + ".json.part"
        with open(part, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(part, stem + ".json")
        return text


NODE_CLASS_MAPPINGS = {
    "RadianceSDRToHDRAnalytic": RadianceSDRToHDRAnalytic,
    "RadianceRudraRegion": RadianceRudraRegion,
    "RadianceRudraCalibrate": RadianceRudraCalibrate,
    "RadianceRudraRegionEV": RadianceRudraRegionEV,
    "RadianceRudraMasterWrite": RadianceRudraMasterWrite,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceSDRToHDRAnalytic": "◎ SDR → HDR Analytic (RUDRA)",
    "RadianceRudraRegion": "◎ RUDRA Region",
    "RadianceRudraCalibrate": "◎ RUDRA Calibrate",
    "RadianceRudraRegionEV": "◎ HDR Region EV (RUDRA)",
    "RadianceRudraMasterWrite": "◎ RUDRA Master Write",
}
