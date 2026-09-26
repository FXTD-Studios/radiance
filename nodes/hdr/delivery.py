"""
nodes_hdr_delivery.py — Radiance HDR Delivery, Monitor & Mastering
════════════════════════════════════════════════════════════════════════════════
(merged: nodes_hdr_delivery + nodes_mastering)

Nodes
─────
  RadianceHDREncode       Scene-linear → PQ (HDR10/DV) or HLG (Broadcast).
                           Replaces: RadiancePQEncoder · RadianceHLGEncoder.
  RadianceHDRMonitor      Scene-linear → SDR/PQ/HLG preview for display.
                           Replaces: RadianceHDRToneMapPreview · RadianceHDRDisplayEncoder.
  RadianceLuminanceGuidance   Mask-based nit-target for HDR synthesis guidance.

All nodes accept scene-linear HDR input (float32, values above 1.0 preserved).
Reference white = 203 cd/m² (BT.2408 recommendation for PQ).
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn.functional as F

# PQ/HLG constants and functions are defined once in color/ops and imported
# here — previously they were copy-pasted from this file into several others.
from radiance.color.ops import (
    PQ_M1 as _PQ_M1,
    PQ_M2 as _PQ_M2,
    PQ_C1 as _PQ_C1,
    PQ_C2 as _PQ_C2,
    PQ_C3 as _PQ_C3,
    M_REC709_TO_BT2020 as _M709_2020,
    linear_to_pq_bt2408 as _linear_to_pq,
    linear_to_hlg as _linear_to_hlg,
    linear_to_hlg_bt2100 as _linear_to_hlg_bt2100,
    apply_matrix_3x3,
)

logger      = logging.getLogger("radiance.hdr_delivery")
diag_logger = logging.getLogger("radiance.diagnostics")


def _apply_bt2020(img: torch.Tensor) -> torch.Tensor:
    """Convert BT.709 primaries to BT.2020 in-place.

    Thin wrapper around apply_matrix_3x3 kept for internal readability.
    """
    return apply_matrix_3x3(img, _M709_2020).clamp(min=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Tone-mapping operators (for SDR monitor preview)
# ─────────────────────────────────────────────────────────────────────────────

def _tm_aces(x):
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return ((x * (a * x + b)) / (x * (c * x + d) + e)).clamp(0.0, 1.0)

def _tm_filmic(x):
    x = (x - 0.004).clamp(min=0.0)
    return (x * (6.2 * x + 0.5)) / (x * (6.2 * x + 1.7) + 0.06)

def _tm_reinhard(x, white=4.0):
    return (x * (1.0 + x / (white * white))) / (1.0 + x)

def _tm_linear(x, exposure=0.0, gamma=2.2):
    return (x * (2.0 ** exposure)).clamp(0.0, 1.0) ** (1.0 / gamma)

_TONE_MAP_OPS = {
    "ACES (Narkowicz)":      _tm_aces,
    "Filmic (Hejl-Burgess)": _tm_filmic,
    "Reinhard Extended":     _tm_reinhard,
    "Exposure + Gamma":      _tm_linear,
}


# ─────────────────────────────────────────────────────────────────────────────
# Node 1 — RadianceHDREncode
# ─────────────────────────────────────────────────────────────────────────────

class RadianceHDREncode:
    """
    ◎ Radiance HDR Encode

    Converts scene-linear HDR to a delivery-ready encoded signal.

    Formats
    ───────
    PQ (HDR10)      ST.2084 PQ — absolute nit encoding for HDR10, Dolby Vision,
                    and Blu-ray HDR. Reference white = 203 cd/m² (BT.2408).
                    Values above 1.0 represent highlights above reference white.
    HLG (Broadcast) BT.2100 HLG — relative encoding used by BBC, NHK, YouTube
                    HDR. The same picture as PQ, transcoded for the 1000-nit
                    reference display (BT.2408): reference white (1.0) at 75 %
                    signal, 1000 nits at 100 %. No peak-nit metadata required.

    Both modes can optionally gamut-convert BT.709 → BT.2020 for delivery.
    OUTPUT_NODE = True — use for final delivery; wire a monitor in parallel.
    """

    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    FUNCTION = "encode"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("encoded_image",)
    OUTPUT_NODE = True
    DESCRIPTION = "Scene-linear HDR → PQ (HDR10/DV) or HLG (Broadcast) delivery signal."

    _FORMATS = ["PQ (HDR10)", "HLG (Broadcast)"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Scene-linear HDR, 1.0 = reference white (203 nits by default). Primaries are either BT.709 (turn apply_bt2020 on) or already BT.2020 (leave it off). Negatives are clipped."}),
                "format": (cls._FORMATS, {"default": "PQ (HDR10)", "tooltip": "PQ: absolute ST.2084 signal clipped at peak_nits. HLG: BT.2100 transcode for a 1000-nit display, reference white at 75% signal."}),
            },
            "optional": {
                "peak_nits": (
                    ["100", "1000", "4000", "10000"],
                    {"default": "1000",
                     "tooltip": "[PQ] Mastering display peak luminance. "
                                "1000 = HDR10, 4000/10000 = Dolby Vision grade."},
                ),
                "reference_white_nits": ("FLOAT", {
                    "default": 203.0, "min": 80.0, "max": 400.0, "step": 1.0,
                    "tooltip": "[PQ / HLG] Nits where linear 1.0 (diffuse white) maps. BT.2408 recommends 203; "
                               "HLG then puts 1.0 at 75 % signal.",
                }),
                "scene_linear_gain": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 8.0, "step": 0.1,
                    "tooltip": "[HLG] Scale scene-linear before encoding. Useful for exposure trim.",
                }),
                "apply_bt2020": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Convert BT.709 primaries to BT.2020. Required for standards-compliant "
                               "HDR10 / HLG delivery. Disable if source is already BT.2020.",
                }),
            },
        }

    def encode(self, image: torch.Tensor, format: str = "PQ (HDR10)",
               peak_nits="1000", reference_white_nits: float = 203.0,
               scene_linear_gain: float = 1.0, apply_bt2020: bool = False):
        # ALBABIT-FIX: peak_nits is a string combo (ComfyUI sends dropdown
        # values as strings) -- convert here, same pattern as nodes/video/hdr.py.
        peak_nits = int(peak_nits)
        img = image[..., :3].clamp(min=0.0).float()

        if apply_bt2020:
            img = _apply_bt2020(img)

        if format == "PQ (HDR10)":
            # ALBABIT-FIX: reference_white_nits was accepted but never passed to
            # _linear_to_pq, which used the hardcoded 203 nits default regardless.
            out = _linear_to_pq(img, peak_nits=peak_nits, reference_white_nits=reference_white_nits)
            logger.info("HDREncode PQ: peak=%d nits  ref=%.0f  bt2020=%s  peak_in=%.3f",
                        peak_nits, reference_white_nits, apply_bt2020, float(image[..., :3].max()))
        else:  # HLG
            # 3.5.0: was the bare OETF on linear, which put reference white
            # (1.0) at 100 % HLG and clipped every highlight. BT.2408 places it
            # at 75 %; the BT.2100 transcode does that and matches OCIO.
            out = _linear_to_hlg_bt2100(img * scene_linear_gain,
                                        reference_white_nits=reference_white_nits)
            logger.info("HDREncode HLG: gain=%.2f  ref=%.0f  bt2020=%s  peak_in=%.3f",
                        scene_linear_gain, reference_white_nits, apply_bt2020,
                        float(image[..., :3].max()))

        diag_logger.info("HDR_ENCODE format=%s peak_in=%.3f", format, float(image[..., :3].max()))

        if image.shape[-1] == 4:
            out = torch.cat([out, image[..., 3:4]], dim=-1)
        return (out,)


# ─────────────────────────────────────────────────────────────────────────────
# Node 2 — RadianceHDRMonitor
# ─────────────────────────────────────────────────────────────────────────────

class RadianceHDRMonitor:
    """
    ◎ Radiance HDR Monitor

    Preview / monitoring node for scene-linear HDR content.
    Does NOT affect your main HDR pipeline — wire in parallel for reference.

    Modes
    ─────
    Preview (SDR)   Tone-maps HDR to display-referred [0,1] for any sRGB monitor.
                    Choose from ACES, Filmic, Reinhard, or Exposure+Gamma operators.
    Rec.2100 PQ     Encodes directly to PQ for HDR display monitoring.
    Rec.2100 HLG    Encodes directly to HLG for broadcast HDR displays.

    OUTPUT_NODE = True — renders a ComfyUI preview panel.
    """

    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    FUNCTION = "monitor"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("preview",)
    OUTPUT_NODE = True
    DESCRIPTION = ("HDR monitor / preview node. "
                   "SDR tone-map or direct PQ/HLG display encoding. "
                   "Wire in parallel — does not affect your HDR pipeline.")

    _MODES = ["Preview (SDR)", "Rec.2100 PQ", "Rec.2100 HLG"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Scene-linear HDR, 1.0 = reference white (203 nits for the PQ and HLG modes). Negatives are clipped."}),
                "mode": (cls._MODES, {"default": "Preview (SDR)", "tooltip": "Preview (SDR): tone-map to 0 to 1 for a normal monitor. Rec.2100 PQ / HLG: encode the signal for an HDR display (no gamut conversion)."}),
            },
            "optional": {
                # ── SDR preview ──────────────────────────────────────────────
                "operator": (
                    list(_TONE_MAP_OPS.keys()),
                    {"default": "ACES (Narkowicz)",
                     "tooltip": "[Preview (SDR)] Tone-mapping operator."},
                ),
                "exposure": ("FLOAT", {
                    "default": 0.0, "min": -6.0, "max": 6.0, "step": 0.1,
                    "tooltip": "[Preview (SDR)] EV offset applied before tone mapping.",
                }),
                "saturation": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "[Preview (SDR)] Post-tonemap saturation scale.",
                }),
                "gamma": ("FLOAT", {
                    "default": 2.2, "min": 1.0, "max": 3.0, "step": 0.05,
                    "tooltip": "[Preview (SDR)] Display gamma, applied as a pure 1/gamma power. 2.2 approximates sRGB, 2.4 matches BT.1886.",
                }),
                "reinhard_white": ("FLOAT", {
                    "default": 4.0, "min": 0.5, "max": 100.0, "step": 0.5,
                    "tooltip": "[Preview (SDR) / Reinhard Extended] White point.",
                }),
                # ── PQ display ───────────────────────────────────────────────
                "peak_nits": ("FLOAT", {
                    "default": 1000.0, "min": 100.0, "max": 10000.0, "step": 100.0,
                    "tooltip": "[Rec.2100 PQ] Mastering display peak luminance in nits.",
                }),
                "gamma_correct_sdr": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "[Preview (SDR)] Apply the 1/gamma display encoding after tone mapping. Applies to every preview operator, including Exposure + Gamma.",
                }),
            },
        }

    def monitor(self, image: torch.Tensor, mode: str = "Preview (SDR)",
                operator: str = "ACES (Narkowicz)",
                exposure: float = 0.0, saturation: float = 1.0,
                gamma: float = 2.2, reinhard_white: float = 4.0,
                peak_nits: float = 1000.0, gamma_correct_sdr: bool = True):
        img = image[..., :3].float().clamp(min=0.0)

        if mode == "Rec.2100 PQ":
            out = _linear_to_pq(img, peak_nits=peak_nits)
        elif mode == "Rec.2100 HLG":
            out = _linear_to_hlg_bt2100(img)
        else:  # Preview (SDR)
            if abs(exposure) > 1e-4:
                img = img * (2.0 ** exposure)

            if operator == "Reinhard Extended":
                tm = _tm_reinhard(img, white=reinhard_white)
            elif operator == "Exposure + Gamma":
                tm = _tm_linear(img, exposure=0.0, gamma=gamma if gamma_correct_sdr else 1.0)
            else:
                tm = _TONE_MAP_OPS.get(operator, _tm_aces)(img)

            if abs(saturation - 1.0) > 1e-4:
                w = img.new_tensor([0.2126, 0.7152, 0.0722])
                luma = (tm * w).sum(dim=-1, keepdim=True)
                tm = (luma + saturation * (tm - luma)).clamp(0.0, 1.0)

            if operator != "Exposure + Gamma" and gamma_correct_sdr and abs(gamma - 1.0) > 1e-4:
                tm = tm.clamp(min=0.0) ** (1.0 / gamma)

            out = tm.clamp(0.0, 1.0)

        if image.shape[-1] == 4:
            out = torch.cat([out, image[..., 3:4]], dim=-1)

        peak_in = float(image[..., :3].max())
        logger.info("HDRMonitor: mode=%s  op=%s  exposure=%.1f EV  peak_in=%.3f",
                    mode, operator if mode == "Preview (SDR)" else "-", exposure, peak_in)
        diag_logger.info("HDR_MONITOR mode=%s peak_in=%.3f", mode, peak_in)
        return (out,)


# ─────────────────────────────────────────────────────────────────────────────
# Node 3 — RadianceLuminanceGuidance (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class RadianceLuminanceGuidance:
    """
    ◎ Radiance Luminance Guidance

    Enables precise control over HDR highlights by providing a nit-target map
    for the HDR Synthesis engine. The masked region will be guided toward
    *target_nits* luminance during energy injection.
    """

    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    DESCRIPTION = "Guide HDR tonemapping using target luminance zone constraints."
    FUNCTION = "generate"
    RETURN_TYPES = ("MASK", "FLOAT", "FLOAT")
    RETURN_NAMES = ("guidance_mask", "target_nits", "feather")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", {"tooltip": "Region whose highlights should be driven towards target_nits, 0 to 1."}),
                "target_nits": ("FLOAT", {
                    "default": 1000.0, "min": 100.0, "max": 10000.0,
                    "tooltip": "Target brightness in nits for the masked region. "
                               "1000 nits = typical HDR10 peak.",
                }),
                "feather": ("FLOAT", {
                    "default": 0.2, "min": 0.0, "max": 1.0,
                    "tooltip": "Gaussian feather radius. Higher = softer mask edge.",
                }),
            },
        }

    def generate(self, mask: torch.Tensor, target_nits: float, feather: float):
        if feather > 0:
            k = max(3, int(feather * 31) | 1)
            sigma = feather * 10.0
            x = torch.linspace(-(k // 2), k // 2, k, device=mask.device, dtype=mask.dtype)
            gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
            gauss = gauss / gauss.sum()
            kernel = (gauss.unsqueeze(0) * gauss.unsqueeze(1)).view(1, 1, k, k)
            m = F.pad(mask.unsqueeze(1), (k // 2,) * 4, mode="reflect")
            m = F.conv2d(m, kernel)
            mask = m.squeeze(1)
        return (mask, target_nits, feather)


# ─────────────────────────────────────────────────────────────────────────────
# Node registry
# ─────────────────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "RadianceHDREncode":          RadianceHDREncode,
    "RadianceHDRMonitor":         RadianceHDRMonitor,
    "RadianceLuminanceGuidance":  RadianceLuminanceGuidance,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceHDREncode":          "◎ Radiance HDR Encode",
    "RadianceHDRMonitor":         "◎ Radiance HDR Monitor",
    "RadianceLuminanceGuidance":  "◎ Radiance Luminance Guidance",
}
