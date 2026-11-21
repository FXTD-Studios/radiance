"""
◎ RUDRA delivery: peak fit, dynamic metadata, flicker report, shot hold
═══════════════════════════════════════════════════════════════════════

RUDRA 1.1-1.2's delivery steps for a master or a movie (a ComfyUI batch):

* **◎ HDR Peak Fit (RUDRA)** -- the shoulder to a display peak, and what
  happens to a colour still too bright for it: Keep luminance (RUDRA's
  default: saturation given up, brightness and hue kept) or Scale.
* **◎ HDR Dynamic Metadata (RUDRA)** -- MaxCLL / MaxFALL, Dolby Vision L1 per
  shot (a ``dovi_tool generate`` config), HDR10+ per shot (x265 /
  ``hdr10plus_tool`` JSON and the scene analysis), and the analysis record in
  nits and PQ codes. Shots come from ◎ Scene Cut Detect or are found by
  RUDRA's histogram test.
* **◎ HDR Flicker Report (RUDRA)** -- how far the HDR's exposure moved from
  frame to frame beyond the source's, in EV, overall and per area.
* **◎ HDR Shot Hold (RUDRA)** -- where the source pixel is still, the HDR's
  gain over it is held across the shot (reset at each cut), so expanded codec
  noise and recovered highlights stop shimmering.

Ported from FXTD-Studios/RUDRA (Apache 2.0), radiance.model.rudra_delivery.
"""
from __future__ import annotations

import json
import logging
import os

import numpy as np
import torch

import comfy.utils

from radiance.color.ops import M_REC709_TO_ACES2065_1, M_REC709_TO_ACESCG, M_REC709_TO_BT2020
from radiance.model.rudra_delivery import (
    CUT_THRESHOLD,
    LUMA_REC2020,
    analyze_frame,
    detect_cuts,
    detect_shots,
    dynamic_metadata,
    dynamic_metadata_paths,
    flicker_report,
    master_to_peak,
    maxcll_maxfall,
    parse_cut_data,
    retention_for,
    shots_from_cuts,
    source_code_luma,
    source_luminance,
    StaticHold,
    write_json_atomic,
)

logger = logging.getLogger("radiance.nodes.hdr.rudra_delivery")

_SPACES = ["Linear (Rec.709)", "Linear (Rec.2020)", "Linear ACEScg (AP1)", "Linear ACES2065-1 (AP0)"]
_FITS = {"Keep luminance (RUDRA default)": "luminance", "Scale": "scale"}
_TRANSFERS = {"sRGB": "srgb", "Rec.709": "rec709", "Gamma 2.2": "gamma22", "Gamma 2.4": "gamma24"}
_SPACE_TIP = "What the HDR is: scene-linear in these primaries, 1.0 = reference_white_nits. PQ / HLG are not accepted."
_CUT_TIP = ("Shot starts: ◎ Scene Cut Detect's cut_data, a JSON list or '0, 48, 120'. Empty: RUDRA finds them.")


def _matrix_to_2020(space: str) -> np.ndarray:
    m709 = M_REC709_TO_BT2020.double().cpu().numpy()
    if space == "Linear (Rec.2020)":
        return np.eye(3)
    if space == "Linear ACEScg (AP1)":
        return m709 @ np.linalg.inv(M_REC709_TO_ACESCG.double().cpu().numpy())
    if space == "Linear ACES2065-1 (AP0)":
        return m709 @ np.linalg.inv(M_REC709_TO_ACES2065_1.double().cpu().numpy())
    if space == "Linear (Rec.709)":
        return m709
    raise ValueError(f"unknown image_space {space!r}")


def _frames(image):
    img = image if image.dim() == 4 else image.unsqueeze(0)
    return img


def _nits2020(frame: np.ndarray, m: np.ndarray, white: float) -> np.ndarray:
    """(H, W, >=3) scene-linear -> (H, W, 3) absolute nits in Rec.2020, float32."""
    rgb = frame[..., :3].astype(np.float64)
    return ((rgb @ m.T) * float(white)).astype(np.float32)


def _output_dir(output_path: str) -> str:
    from radiance.path_utils import get_safe_output_dir, strip_path_quotes
    try:
        import folder_paths
        base = folder_paths.get_output_directory()
    except Exception:
        import tempfile
        base = tempfile.gettempdir()
    out = get_safe_output_dir(base, strip_path_quotes(output_path or ""), allow_absolute=True)
    os.makedirs(out, exist_ok=True)
    return out


def _check_prefix(prefix: str, who: str) -> None:
    if not prefix or any(c in prefix for c in '\\/:*?"<>|'):
        raise ValueError(f"[{who}] filename_prefix must be a plain name (no folders)")


def _same_count(sdr, hdr, who):
    if sdr.shape[0] != hdr.shape[0]:
        raise ValueError(f"[{who}] {sdr.shape[0]} SDR frames for {hdr.shape[0]} HDR frames; give the same clip")
    if sdr.shape[1:3] != hdr.shape[1:3]:
        raise ValueError(f"[{who}] the SDR is {sdr.shape[2]}x{sdr.shape[1]}, the HDR {hdr.shape[2]}x{hdr.shape[1]}")


class RadianceRudraPeakFit:
    """Shoulder to a display peak with RUDRA's peak fit."""

    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    DESCRIPTION = (
        "RUDRA's master-to-peak: below the knee (0.75 x peak unless set) nothing moves; above it the "
        "luminance approaches the display peak smoothly with the pixel's ratios kept. A colour still too "
        "bright for the display (a saturated blue sky at 1000 nits) is then fitted: Keep luminance (RUDRA's "
        "default) gives up saturation and keeps its brightness and hue; Scale keeps hue and saturation and "
        "loses brightness. Measured in Rec.2020, scene-linear out in the same space as in."
    )
    FUNCTION = "fit"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "report")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Scene-linear HDR master."}),
                "image_space": (_SPACES, {"default": _SPACES[0], "tooltip": _SPACE_TIP}),
                "peak_nits": ("FLOAT", {"default": 1000.0, "min": 100.0, "max": 10000.0, "step": 50.0,
                                        "tooltip": "The display's peak. No channel ends above it (in Rec.2020)."}),
                "peak_fit": (list(_FITS), {"default": "Keep luminance (RUDRA default)",
                    "tooltip": "For a colour whose luminance fits but one channel does not: give up saturation "
                               "(Keep luminance) or brightness (Scale)."}),
            },
            "optional": {
                "knee_nits": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 9999.0, "step": 10.0,
                                        "tooltip": "Where the shoulder starts. 0: RUDRA's 0.75 x peak."}),
                "reference_white_nits": ("FLOAT", {"default": 203.0, "min": 80.0, "max": 400.0, "step": 1.0,
                                                   "tooltip": "Nits of 1.0 in the image (BT.2408: 203)."}),
            },
        }

    @torch.no_grad()
    def fit(self, image, image_space, peak_nits, peak_fit, knee_nits=0.0, reference_white_nits=203.0):
        m = _matrix_to_2020(image_space)
        inv = np.linalg.inv(m)
        knee = float(knee_nits) if knee_nits and knee_nits > 0 else None
        fit = _FITS.get(peak_fit, peak_fit)
        white = float(reference_white_nits)
        img = _frames(image)
        out, touched, lost = [], 0.0, 0.0
        for i in range(img.shape[0]):
            f = img[i].float().cpu().numpy()
            nits = _nits2020(f, m, white)
            fitted = master_to_peak(nits, float(peak_nits), knee, fit)
            k = np.asarray(LUMA_REC2020)
            before = np.maximum(nits, 0) @ k
            after = fitted @ k
            touched += float((np.abs(fitted - np.maximum(nits, 0)) > 1e-3).any(-1).mean())
            lost += float(np.log2((before.sum() + 1e-6) / (after.sum() + 1e-6)))
            back = ((fitted.astype(np.float64) / white) @ inv.T).astype(np.float32)
            t = torch.from_numpy(back)
            if f.shape[-1] > 3:
                t = torch.cat([t, torch.from_numpy(f[..., 3:])], -1)
            out.append(t)
        n = img.shape[0]
        report = (f"peak fit: {peak_fit} at {peak_nits:.0f} nits, knee {knee if knee is not None else 0.75 * peak_nits:.0f}"
                  f" nits\npixels changed: {100 * touched / n:.2f} %\nframe light given up: {lost / n:.3f} stops (mean)")
        return (torch.stack(out).to(image.dtype), report)


class RadianceRudraDynamicMetadata:
    """HDR10+ and Dolby Vision metadata per shot, MaxCLL / MaxFALL, from the frames."""

    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    DESCRIPTION = (
        "Measures the master as the display will get it (RUDRA's peak fit first, unless turned off) and "
        "writes, beside <prefix>: _hdr10plus.json (x265 --dhdr10-info / hdr10plus_tool inject), "
        "_dovi_generate.json (dovi_tool generate, profile 8.1, L1 per shot), _hdr10plus_scenes.json and "
        "_hdr_analysis.json (every frame in nits and 12-bit PQ). Statistics are MaxRGB per pixel, per "
        "CTA-861.3 and Dolby L1, and never change within a shot."
    )
    FUNCTION = "measure"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("summary", "hdr10plus_json", "dovi_json", "analysis_json", "hdr10plus_path")
    OUTPUT_TOOLTIPS = ("What was measured, per shot.", "The x265 / hdr10plus_tool JSON, as text.",
                       "The dovi_tool generate config, as text.", "Every frame in nits and PQ, as text.",
                       "The written _hdr10plus.json, for Radiance Write's hdr10plus_json (empty with write_files off).")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The movie's frames (a batch), scene-linear HDR."}),
                "image_space": (_SPACES, {"default": _SPACES[0], "tooltip": _SPACE_TIP}),
                "peak_nits": ("FLOAT", {"default": 1000.0, "min": 100.0, "max": 10000.0, "step": 50.0,
                                        "tooltip": "Mastering display peak (Dolby L6, and the peak fit)."}),
                "filename_prefix": ("STRING", {"default": "rudra_master",
                                               "tooltip": "The four files are <prefix>_hdr10plus.json and so on."}),
                "write_files": ("BOOLEAN", {"default": True, "tooltip": "Off: only the outputs."}),
            },
            "optional": {
                "cut_data": ("STRING", {"default": "", "tooltip": _CUT_TIP}),
                "apply_peak_fit": ("BOOLEAN", {"default": True,
                    "tooltip": "Measure after RUDRA's shoulder and peak fit, as the encoder will get it. Off: "
                               "the frames are already mastered to the peak."}),
                "peak_fit": (list(_FITS), {"default": "Keep luminance (RUDRA default)",
                    "tooltip": "For a colour too bright for the display: give up saturation (Keep luminance) "
                               "or brightness (Scale)."}),
                "knee_nits": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 9999.0, "step": 10.0,
                                        "tooltip": "0: RUDRA's 0.75 x peak."}),
                "mastering_min_nits": ("FLOAT", {"default": 0.0001, "min": 0.0, "max": 1.0, "step": 0.0001,
                                                 "tooltip": "Mastering display black (Dolby L6)."}),
                "shot_threshold": ("FLOAT", {"default": 0.35, "min": 0.01, "max": 2.0, "step": 0.01,
                    "tooltip": "With no cut_data: RUDRA's histogram distance that declares a cut. Higher: "
                               "fewer, longer shots (safer: a trim that changes mid-shot pumps)."}),
                "hdr10plus_target_nits": ("INT", {"default": 400, "min": 100, "max": 10000,
                                                  "tooltip": "Targeted display in the scene analysis JSON."}),
                "reference_white_nits": ("FLOAT", {"default": 203.0, "min": 80.0, "max": 400.0, "step": 1.0,
                                                   "tooltip": "Nits of 1.0 in the HDR (BT.2408: 203)."}),
                "output_path": ("STRING", {"default": "", "tooltip": "Folder. Empty: ComfyUI's output folder."}),
            },
        }

    @torch.no_grad()
    def measure(self, image, image_space, peak_nits, filename_prefix, write_files, cut_data="",
                apply_peak_fit=True, peak_fit="Keep luminance (RUDRA default)", knee_nits=0.0,
                mastering_min_nits=0.0001, shot_threshold=0.35, hdr10plus_target_nits=400,
                reference_white_nits=203.0, output_path=""):
        who = "HDR Dynamic Metadata"
        if write_files:
            _check_prefix(filename_prefix, who)
        m = _matrix_to_2020(image_space)
        knee = float(knee_nits) if knee_nits and knee_nits > 0 else None
        fit = _FITS.get(peak_fit, peak_fit)
        img = _frames(image)
        n = img.shape[0]
        stats = []
        bar = comfy.utils.ProgressBar(n) if n > 1 else None
        for i in range(n):
            nits = _nits2020(img[i].float().cpu().numpy(), m, float(reference_white_nits))
            if apply_peak_fit:
                nits = master_to_peak(nits, float(peak_nits), knee, fit)
            stats.append(analyze_frame(nits, i))
            if bar is not None:
                bar.update(1)
        cuts = parse_cut_data(cut_data, n)
        shots = shots_from_cuts(cuts, n) if cuts is not None else detect_shots(stats, float(shot_threshold))
        data = dynamic_metadata(stats, shots, float(peak_nits), float(mastering_min_nits), int(hdr10plus_target_nits))
        max_cll, max_fall = maxcll_maxfall(stats)
        lines = [f"frames: {n}; shots: {len(shots)} ({'from cut_data' if cuts is not None else 'detected'})",
                 f"MaxCLL {max_cll} nits, MaxFALL {max_fall} nits (MaxRGB, Rec.2020"
                 + (f", after the {fit} peak fit at {peak_nits:.0f} nits)" if apply_peak_fit else ")")]
        for s in data["analysis"]["shots"][:12]:
            lines.append(f"  shot at {s['start']} ({s['duration']} frames): L1 min/avg/max PQ "
                         f"{s['min_pq']}/{s['avg_pq']}/{s['max_pq']} = {s['min_nits']:.3f}/{s['avg_nits']:.1f}/"
                         f"{s['max_nits']:.0f} nits")
        if len(shots) > 12:
            lines.append(f"  ... and {len(shots) - 12} more shots")
        hdr10plus_path = ""
        if write_files:
            stem = os.path.join(_output_dir(output_path), filename_prefix)
            paths = dynamic_metadata_paths(stem)
            for key, path in paths.items():
                write_json_atomic(path, data[key], indent=None if key == "hdr10plus" else 2)
            lines.append("written: " + ", ".join(p.name for p in paths.values()))
            hdr10plus_path = str(paths["hdr10plus"])
        return ("\n".join(lines), json.dumps(data["hdr10plus"]), json.dumps(data["dovi_generate"], indent=2),
                json.dumps(data["analysis"], indent=2), hdr10plus_path)


def _shot_starts(cut_data, sdr_np, threshold):
    cuts = parse_cut_data(cut_data, len(sdr_np))
    return (cuts, "from cut_data") if cuts is not None else (detect_cuts(sdr_np, float(threshold)), "detected")


class RadianceRudraFlickerReport:
    """RUDRA's temporal QC: the HDR's exposure change beyond the source's, frame to frame."""

    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    DESCRIPTION = (
        "How far the HDR's average exposure moved from one frame to the next beyond what the source's did, "
        "in EV (flicker_ev), and the same per area of a 36 x 64 grid (local_flicker_ev). Frames that start a "
        "shot are not scored. Mean, max and the worst frame, as RUDRA's movie report gives them."
    )
    FUNCTION = "report"
    RETURN_TYPES = ("STRING", "STRING", "FLOAT")
    RETURN_NAMES = ("report", "json_report", "flicker_ev_max")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sdr": ("IMAGE", {"tooltip": "The source clip, as decoded (0..1 codes)."}),
                "hdr": ("IMAGE", {"tooltip": "The HDR made from it, scene-linear, the same frames."}),
                "image_space": (_SPACES, {"default": _SPACES[0], "tooltip": _SPACE_TIP}),
            },
            "optional": {
                "sdr_transfer": (list(_TRANSFERS), {"default": "sRGB", "tooltip": "How the source codes decode."}),
                "cut_data": ("STRING", {"default": "", "tooltip": _CUT_TIP}),
                "cut_threshold": ("FLOAT", {"default": CUT_THRESHOLD, "min": 0.01, "max": 1.0, "step": 0.01,
                                            "tooltip": "With no cut_data: RUDRA's thumbnail change that starts a shot."}),
                "reference_white_nits": ("FLOAT", {"default": 203.0, "min": 80.0, "max": 400.0, "step": 1.0,
                                                   "tooltip": "Nits of 1.0 in the HDR (BT.2408: 203)."}),
            },
        }

    @torch.no_grad()
    def report(self, sdr, hdr, image_space, sdr_transfer="sRGB", cut_data="", cut_threshold=CUT_THRESHOLD,
               reference_white_nits=203.0):
        s, h = _frames(sdr), _frames(hdr)
        _same_count(s, h, "HDR Flicker Report")
        m = _matrix_to_2020(image_space)
        sdr_np = [s[i, ..., :3].float().cpu().numpy() for i in range(s.shape[0])]
        net = [_nits2020(h[i].float().cpu().numpy(), m, float(reference_white_nits)) / 10000.0
               for i in range(h.shape[0])]
        cuts, how = _shot_starts(cut_data, sdr_np, cut_threshold)
        rep = flicker_report(sdr_np, net, cuts, _TRANSFERS.get(sdr_transfer, sdr_transfer))
        t = rep["temporal"]
        fmt = lambda v: "n/a" if v is None else f"{v:.4f} EV"
        lines = [f"frames: {len(sdr_np)}; scored: {t['frames_scored']}; shots: {len(rep['scene_cuts'])} ({how})",
                 f"flicker: mean {fmt(t['flicker_ev_mean'])}, max {fmt(t['flicker_ev_max'])}"
                 + (f" at frame {t['flicker_ev_max_frame']}" if t["flicker_ev_max_frame"] is not None else ""),
                 f"local flicker: mean {fmt(t['local_flicker_ev_mean'])}, max {fmt(t['local_flicker_ev_max'])}"]
        if t["flicker_ev_max"] is not None and t["flicker_ev_max"] > 0.1:
            lines.append("over 0.1 EV: the HDR pumps where the source did not (try ◎ HDR Shot Hold (RUDRA))")
        return ("\n".join(lines), json.dumps(rep, indent=2), float(t["flicker_ev_max"] or 0.0))


class RadianceRudraShotHold:
    """RUDRA's static hold across a shot, reset at cuts."""

    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    DESCRIPTION = (
        "Where the source pixel did not change from the frame before, the HDR's gain over the source is held "
        "with a time constant, so codec noise lifted by the expansion and jitter in recovered highlights stop "
        "shimmering; where it moved, or at a cut, the frame is its own. Inside highlights clipped in both "
        "frames the HDR's own change decides. Only the gain is held, never the pixel: the source's own "
        "change passes through."
    )
    FUNCTION = "hold"
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "cut_data", "report")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sdr": ("IMAGE", {"tooltip": "The source clip, as decoded (0..1 codes)."}),
                "hdr": ("IMAGE", {"tooltip": "The HDR made from it, scene-linear, the same frames."}),
                "image_space": (_SPACES, {"default": _SPACES[0], "tooltip": _SPACE_TIP}),
                "hold_seconds": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 5.0, "step": 0.05,
                    "tooltip": "The hold's time constant (RUDRA's temporal smoothing, default 0.25 s). 0: off."}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001,
                                  "tooltip": "The clip's frame rate (sets the per-frame retention)."}),
            },
            "optional": {
                "sdr_transfer": (list(_TRANSFERS), {"default": "sRGB", "tooltip": "How the source codes decode."}),
                "cut_data": ("STRING", {"default": "", "tooltip": _CUT_TIP}),
                "cut_threshold": ("FLOAT", {"default": CUT_THRESHOLD, "min": 0.01, "max": 1.0, "step": 0.01,
                                            "tooltip": "With no cut_data: RUDRA's thumbnail change that starts a shot."}),
            },
        }

    @torch.no_grad()
    def hold(self, sdr, hdr, image_space, hold_seconds, fps, sdr_transfer="sRGB", cut_data="",
             cut_threshold=CUT_THRESHOLD):
        s, h = _frames(sdr), _frames(hdr)
        _same_count(s, h, "HDR Shot Hold")
        m = _matrix_to_2020(image_space)
        k = np.asarray(LUMA_REC2020) @ m      # Rec.2020 luminance from the image's own primaries
        sdr_np = [s[i, ..., :3].float().cpu().numpy() for i in range(s.shape[0])]
        cuts, how = _shot_starts(cut_data, sdr_np, cut_threshold)
        retention = retention_for(float(fps), float(hold_seconds))
        transfer = _TRANSFERS.get(sdr_transfer, sdr_transfer)
        holder = StaticHold(retention)
        starts = set(cuts)
        out, held_share = [], []
        for i, f in enumerate(sdr_np):
            if i in starts:
                holder.reset()
            frame = h[i].float().cpu().numpy()
            yh = (frame[..., :3].astype(np.float64) @ k) * (203.0 / 10000.0)   # network units; ratios only
            luma, peak = source_code_luma(f)
            factor = holder.factor(luma, peak, source_luminance(f, transfer), yh)
            if factor is None:
                out.append(torch.from_numpy(frame))
                continue
            held_share.append(float((np.abs(factor - 1.0) > 1e-4).mean()))
            rgb = (frame[..., :3].astype(np.float64) * factor[..., None]).astype(np.float32)
            if frame.shape[-1] > 3:
                rgb = np.concatenate([rgb, frame[..., 3:]], -1)
            out.append(torch.from_numpy(rgb))
        report = (f"hold: {hold_seconds:.2f} s at {fps:g} fps (retention {retention:.3f}); shots: {len(cuts)} ({how})\n"
                  f"pixels held: {100 * (sum(held_share) / len(held_share) if held_share else 0.0):.1f} % of each "
                  f"frame after a shot's first (mean)")
        return (torch.stack(out).to(hdr.dtype), json.dumps({"cuts": cuts}), report)


NODE_CLASS_MAPPINGS = {
    "RadianceRudraPeakFit": RadianceRudraPeakFit,
    "RadianceRudraDynamicMetadata": RadianceRudraDynamicMetadata,
    "RadianceRudraFlickerReport": RadianceRudraFlickerReport,
    "RadianceRudraShotHold": RadianceRudraShotHold,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceRudraPeakFit": "◎ HDR Peak Fit (RUDRA)",
    "RadianceRudraDynamicMetadata": "◎ HDR Dynamic Metadata (RUDRA)",
    "RadianceRudraFlickerReport": "◎ HDR Flicker Report (RUDRA)",
    "RadianceRudraShotHold": "◎ HDR Shot Hold (RUDRA)",
}
