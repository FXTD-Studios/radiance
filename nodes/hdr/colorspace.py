"""
nodes_hdr_colorspace.py — Radiance HDR Color Space Pipeline
════════════════════════════════════════════════════════════════════════════════
(merged: RadianceLinearize + RadiancePrimariesTransform + RadianceChromaAdapt
 → RadianceHDRColorPipeline)

Two nodes:

  1. RadianceHDRColorPipeline
     Full end-to-end pipeline in one node (linearise → adapt → primaries →
     soft-knee compress → VAE-ready).

  2. RadianceColorSpaceInfo
     Passthrough metadata node — outputs colour-space JSON for diagnostics.
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional

import torch
import torch.nn.functional as F

# Shared math primitives — imported once, used throughout this file.
# Previously these constants and helpers were copy-pasted from (or imported via
# private symbols of) nodes_hdr_delivery and nodes_hdr_uplift.
from radiance.color.ops import (
    PQ_M1, PQ_M2, PQ_C1, PQ_C2, PQ_C3, PQ_REF_WHITE_NITS, PQ_MAX_NITS,
    apply_matrix_3x3,
    soft_knee_compress as _soft_knee_compress,
)
# Canonical log-curve decoders — replaces the deprecated color_utils shim.
from radiance.color.transfer import (
    tensor_logc4_to_linear,
    tensor_slog3_to_linear,
)

logger = logging.getLogger("radiance.hdr_colorspace")
diag_logger = logging.getLogger("radiance.diagnostics")


# ─────────────────────────────────────────────────────────────────────────────
# 3×3 colour matrices  (row-vector convention: rgb_out = rgb_in @ M.T)
# ─────────────────────────────────────────────────────────────────────────────

# All matrices are normalised for D65 unless noted.

_MATRICES: dict[str, dict[str, list]] = {
    # Rec.709 primaries → XYZ (D65)
    "rec709_to_xyz": [
        [ 0.4124564,  0.3575761,  0.1804375],
        [ 0.2126729,  0.7151522,  0.0721750],
        [ 0.0193339,  0.1191920,  0.9503041],
    ],
    # XYZ (D65) → Rec.709
    "xyz_to_rec709": [
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252],
    ],
    # BT.2020 primaries → XYZ (D65)
    "bt2020_to_xyz": [
        [ 0.6369580,  0.1446169,  0.1688810],
        [ 0.2627002,  0.6779981,  0.0593017],
        [ 0.0000000,  0.0280727,  1.0609851],
    ],
    # XYZ (D65) → BT.2020
    "xyz_to_bt2020": [
        [ 1.7166512, -0.3556708, -0.2533663],
        [-0.6666844,  1.6164812,  0.0157685],
        [ 0.0176399, -0.0427706,  0.9421031],
    ],
    # ACEScg (AP1) → XYZ (D60)
    "acescg_to_xyz_d60": [
        [ 0.6624541811, 0.1340042065, 0.1561876744],
        [ 0.2722287168, 0.6740817658, 0.0536895174],
        [-0.0055746495, 0.0040607335, 1.0103391003],
    ],
    # XYZ (D60) → ACEScg (AP1)
    "xyz_d60_to_acescg": [
        [ 1.6410233797, -0.3248032942, -0.2364246952],
        [-0.6636628587,  1.6153315917,  0.0167563477],
        [ 0.0117218943, -0.0082844420,  0.9883948585],
    ],
    # DCI-P3 → XYZ (D65)
    "dcip3_to_xyz": [
        [ 0.4865709,  0.2656677,  0.1982173],
        [ 0.2289746,  0.6917385,  0.0792869],
        [ 0.0000000,  0.0451134,  1.0439444],
    ],
    # XYZ (D65) → DCI-P3
    "xyz_to_dcip3": [
        [ 2.4934969, -0.9313836, -0.4027108],
        [-0.8294890,  1.7626641,  0.0236247],
        [ 0.0358458, -0.0761724,  0.9568845],
    ],
}

# Bradford chromatic adaptation matrices  (D65 → target)
_BRADFORD_CAT: dict[str, list] = {
    # 3.5.0: the D65/D60 pair was wrong (D65 white landed 1.2 % off D60 in
    # X); these are Bradford from the D65 and ACES white points, and map one
    # white exactly onto the other.
    "D65_to_D60": [
        [ 1.0130349,  0.0061053, -0.0149709],
        [ 0.0076982,  0.9981634, -0.0050320],
        [-0.0028413,  0.0046852,  0.9245061],
    ],
    "D60_to_D65": [
        [ 0.9872240, -0.0061132,  0.0159533],
        [-0.0075984,  1.0018615,  0.0053300],
        [ 0.0030726, -0.0050960,  1.0816806],
    ],
    "D65_to_D50": [
        [ 1.0478112,  0.0228866, -0.0501270],
        [ 0.0295424,  0.9904844, -0.0170491],
        [-0.0092345,  0.0150436,  0.7521316],
    ],
    "D50_to_D65": [
        [ 0.9554734, -0.0230985,  0.0631188],
        [-0.0283944,  1.0099540,  0.0210418],
        [ 0.0123072, -0.0205069,  1.3299071],
    ],
}

# Compound matrices (pre-multiplied for efficiency)
# Rec.709 D65 → BT.2020 D65 (via XYZ)
_M_709_TO_2020 = [
    [ 0.6274040,  0.3292820,  0.0433136],
    [ 0.0690970,  0.9195400,  0.0113612],
    [ 0.0163916,  0.0880132,  0.8955950],
]
_M_2020_TO_709 = [
    [ 1.6604910, -0.5876411, -0.0728499],
    [-0.1245505,  1.1328999, -0.0083494],
    [-0.0181508, -0.1005789,  1.1187297],
]
# Rec.709 D65 → ACEScg D60 (Bradford-adapted)
_M_709_TO_ACESCG = [
    [ 0.6130974,  0.3395175,  0.0473851],
    [ 0.0701972,  0.9163411,  0.0134617],
    [ 0.0206156,  0.1095698,  0.8698146],
]
# The inverse of the matrix above (it was 4e-3 off, so ACEScg to Rec.709 and
# back drifted); the same values as radiance.color.ops.M_ACESCG_TO_REC709.
_M_ACESCG_TO_709 = [
    [ 1.7050509, -0.6217921, -0.0832588],
    [-0.1302564,  1.1408047, -0.0105483],
    [-0.0240033, -0.1289690,  1.1529723],
]
# DCI-P3 D65 → BT.2020
_M_P3_TO_2020 = [
    [ 0.7539397,  0.1986815,  0.0473788],
    [ 0.0458697,  0.9416312,  0.0124991],
    [-0.0011204,  0.0176004,  0.9835200],
]

_PRIMARIES_MATRICES: dict[tuple, list] = {
    ("Rec.709 (sRGB)",  "BT.2020"):       _M_709_TO_2020,
    ("BT.2020",         "Rec.709 (sRGB)"): _M_2020_TO_709,
    ("Rec.709 (sRGB)",  "ACEScg"):        _M_709_TO_ACESCG,
    ("ACEScg",          "Rec.709 (sRGB)"): _M_ACESCG_TO_709,
    ("DCI-P3 (D65)",    "BT.2020"):       _M_P3_TO_2020,
}

_PRIMARIES_LIST = ["Rec.709 (sRGB)", "BT.2020", "ACEScg", "DCI-P3 (D65)", "XYZ (D65)"]

# Each primaries set as (to-XYZ, from-XYZ, white point). Used to build every
# pair the table above does not carry; the five pairs above keep their
# published matrices so their output is unchanged.
_PRIMARIES_XYZ: dict[str, tuple] = {
    "Rec.709 (sRGB)": (_MATRICES["rec709_to_xyz"], _MATRICES["xyz_to_rec709"], "D65"),
    "BT.2020":        (_MATRICES["bt2020_to_xyz"], _MATRICES["xyz_to_bt2020"], "D65"),
    "DCI-P3 (D65)":   (_MATRICES["dcip3_to_xyz"],  _MATRICES["xyz_to_dcip3"],  "D65"),
    "ACEScg":         (_MATRICES["acescg_to_xyz_d60"], _MATRICES["xyz_d60_to_acescg"], "D60"),
    "XYZ (D65)":      ([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                       [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], "D65"),
}


def _matmul3(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _primaries_matrix(source: str, target: str) -> Optional[List[List[float]]]:
    """RGB-to-RGB matrix from `source` to `target` primaries, with a Bradford
    white-point adaptation when their whites differ (ACEScg is D60, the rest
    D65). None when either name is unknown."""
    if (source, target) in _PRIMARIES_MATRICES:
        return _PRIMARIES_MATRICES[(source, target)]
    if source not in _PRIMARIES_XYZ or target not in _PRIMARIES_XYZ:
        return None
    to_xyz, _, w_src = _PRIMARIES_XYZ[source]
    _, from_xyz, w_dst = _PRIMARIES_XYZ[target]
    m = to_xyz
    if w_src != w_dst:
        m = _matmul3(_BRADFORD_CAT[f"{w_src}_to_{w_dst}"], m)
    return _matmul3(from_xyz, m)


def _primaries_adapt_white(source: str, target: str) -> bool:
    """True when converting source to target already changes the white point."""
    a, b = _PRIMARIES_XYZ.get(source), _PRIMARIES_XYZ.get(target)
    return bool(a and b and a[2] != b[2])


def _apply_matrix(img: torch.Tensor, M: List[List[float]]) -> torch.Tensor:
    """Apply a 3×3 colour matrix to (B,H,W,3) or (H,W,3).

    Thin alias for :func:`radiance.color.ops.apply_matrix_3x3` kept for
    backward compatibility with any submodule that imports this symbol.
    """
    return apply_matrix_3x3(img, M)


# ─────────────────────────────────────────────────────────────────────────────
# EOTFs (display → scene-linear)
# ─────────────────────────────────────────────────────────────────────────────

def _eotf_srgb(v: torch.Tensor) -> torch.Tensor:
    return torch.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)

def _eotf_rec709(v: torch.Tensor) -> torch.Tensor:
    return torch.where(v < 0.081, v / 4.5, ((v + 0.099) / 1.099) ** (1 / 0.45))

def _eotf_gamma22(v: torch.Tensor) -> torch.Tensor:
    return v.clamp(min=0.0) ** 2.2

def _eotf_gamma24(v: torch.Tensor) -> torch.Tensor:
    return v.clamp(min=0.0) ** 2.4

def _eotf_bt1886(v: torch.Tensor) -> torch.Tensor:
    """BT.1886 EOTF — the ITU-R standard for Rec.709 reference displays.

    For ideal displays (Lmin = 0, Lmax = 100 cd/m²) this simplifies to
    pure gamma 2.4.  Mathematically identical to Gamma 2.4 but named
    correctly so broadcast / film-TV colorists recognise it.

    Ref: ITU-R BT.1886 (03/2011) §2.
    """
    return v.clamp(min=0.0) ** 2.4

def _eotf_pq(v: torch.Tensor) -> torch.Tensor:
    """ST.2084 PQ EOTF: signal [0, 1] → scene-linear (ref white = 1.0 at 203 nits).

    Uses the module-level PQ constants imported from radiance.color.ops.

    PQ carries ABSOLUTE luminance normalised by a fixed 10 000 cd/m² ceiling,
    so nothing about the mastering display enters the decode.  The former
    ``peak_nits`` parameter scaled by ``peak_nits / 203`` instead of
    ``10000 / 203``, decoding a real HDR10 file 10x dark at the shipped
    default.  It is removed rather than corrected in place so that a stale
    caller raises instead of quietly changing exposure.
    """
    Vm2 = v.clamp(0.0, 1.0) ** (1.0 / PQ_M2)
    L   = (
        (Vm2 - PQ_C1).clamp(min=0.0)
        / (PQ_C2 - PQ_C3 * Vm2).clamp(min=1e-7)
    ) ** (1.0 / PQ_M1)
    # L is the [0,1] fraction of 10 000 cd/m²; rescale so 203 cd/m² reads 1.0.
    return L * (PQ_MAX_NITS / PQ_REF_WHITE_NITS)

def _eotf_hlg(v: torch.Tensor) -> torch.Tensor:
    """HLG OETF inverse: signal [0,1] → scene-linear (ref white ≈ 1.0)."""
    a = 0.17883277
    b = 0.28466892
    c = 0.55991073
    # The linear segment is (v ** 2) / 3.0, written inline in the torch.where
    # below.  A second, wrong expression for it used to sit here unused
    # ((v/3)**2 * 12, which is 0.333 at v=0.5 where the correct form is 0.083);
    # anyone reconciling the two by "fixing" the where-branch would have broken
    # the HLG EOTF.  Removed rather than left as a trap.
    log_region    = (torch.exp((v - c) / a) + b) / 12.0
    return torch.where(v <= 0.5, (v ** 2) / 3.0, log_region).clamp(min=0.0)

def _eotf_logc4(v: torch.Tensor) -> torch.Tensor:
    """ARRI LogC4 → scene-linear.

    Delegates to the canonical implementation in radiance.color.transfer.
    """
    return tensor_logc4_to_linear(v)


def _eotf_slog3(v: torch.Tensor) -> torch.Tensor:
    """Sony S-Log3 → scene-linear.

    Delegates to the canonical implementation in radiance.color.transfer.
    """
    return tensor_slog3_to_linear(v)

_EOTF_MAP = {
    # ── SDR display EOTFs ─────────────────────────────────────────────────────
    "sRGB":               _eotf_srgb,       # IEC 61966-2-1 — web / monitor
    "Rec.709 (OETF)":     _eotf_rec709,     # BT.709 camera OETF inverse — broadcast signal
    "BT.1886 (TV γ2.4)":  _eotf_bt1886,    # ITU-R BT.1886 — Rec.709 reference display EOTF
    "Gamma 2.2":          _eotf_gamma22,    # Generic γ2.2
    "Gamma 2.4":          _eotf_gamma24,    # Generic γ2.4
    # ── HDR EOTFs ─────────────────────────────────────────────────────────────
    "PQ (ST.2084)":       _eotf_pq,         # ST.2084 — HDR10 / Dolby Vision
    "HLG":                _eotf_hlg,        # Hybrid Log-Gamma — HLG broadcast
    # ── Camera log curves ─────────────────────────────────────────────────────
    "ARRI LogC4":         _eotf_logc4,      # ARRI Alexa 35
    "Sony S-Log3":        _eotf_slog3,      # Sony Venice / FX-series
    # ── Pass-through ──────────────────────────────────────────────────────────
    "Linear (none)":      lambda v: v.clamp(min=0.0),
}


# ─────────────────────────────────────────────────────────────────────────────
# Node 1 — RadianceHDRColorPipeline
# ─────────────────────────────────────────────────────────────────────────────

class RadianceHDRColorPipeline:
    """
    ◎ Radiance HDR Color Pipeline

    Full end-to-end colour pipeline in one node:

        display-referred input
          → inverse EOTF (linearise)
          → chromatic adaptation (optional)
          → primaries transform (optional)
          → soft-knee HDR compression (for VAE)

    Outputs both the VAE-ready compressed image AND the scene-linear passthrough
    so you can branch to diagnostics / EXR save without extra nodes.

    This is the correct node when you have a Rec.709 SDR image and want to
    feed it into a Flux / SDXL / Wan model with Radiance HDR active.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Encoded input image. Negatives are clamped to 0; alpha is passed through.",
                }),
                "encoding": (list(_EOTF_MAP.keys()), {
                    "default": "sRGB",
                    "tooltip": (
                        "Transfer curve of the input, decoded to linear. PQ decodes so 203 nits = 1.0; "
                        "log curves decode to camera scene-linear. Linear (none) only clamps negatives."
                    ),
                }),
                "compression_ratio": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Compression ratio for soft-knee HDR highlight compression.",
                }),
            },
            "optional": {
                "source_primaries": (
                    _PRIMARIES_LIST,
                    {
                        "default": "Rec.709 (sRGB)",
                        "tooltip": (
                            "Primaries of the input. Any pair of the listed primaries converts, with a "
                            "Bradford white-point adaptation where the whites differ (ACEScg is D60)."
                        ),
                    },
                ),
                "target_primaries": (
                    _PRIMARIES_LIST,
                    {
                        "default": "Rec.709 (sRGB)",
                        "tooltip": (
                            "Primaries to convert to. Same as source leaves colours unchanged."
                        ),
                    },
                ),
                "chromatic_adaptation": (
                    ["None"] + list(_BRADFORD_CAT.keys()),
                    {
                        "default": "None",
                        "tooltip": (
                            "Bradford white-point matrix applied to the linear RGB before the primaries "
                            "step. Skipped, with a warning, when the primaries conversion already changes "
                            "the white point (to or from ACEScg), so it is never applied twice."
                        ),
                    },
                ),
                "pq_peak_nits": ("FLOAT", {
                    "default": 1000.0, "min": 100.0, "max": 10000.0, "step": 100.0,
                    "tooltip": (
                        "Has no effect on the decode: PQ is absolute, so it always decodes with "
                        "203 nits = 1.0 whatever this is set to."
                    ),
                }),
            },
        }

    RETURN_TYPES  = ("IMAGE",    "IMAGE",          "FLOAT",        "STRING")
    RETURN_NAMES  = ("vae_image","scene_linear",   "peak_linear",  "colorspace_json")
    FUNCTION      = "pipeline"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    DESCRIPTION   = (
        "Full colour pipeline: linearise → adapt → primaries → HDR compress. "
        "Outputs both VAE-ready and scene-linear images."
    )

    def pipeline(
        self,
        image: torch.Tensor,
        encoding: str = "sRGB",
        compression_ratio: float = 0.5,
        source_primaries: str = "Rec.709 (sRGB)",
        target_primaries: str = "Rec.709 (sRGB)",
        chromatic_adaptation: str = "None",
        pq_peak_nits: float = 1000.0,
    ):
        # 1. Inverse EOTF → scene-linear
        fn = _EOTF_MAP.get(encoding, _eotf_srgb)
        img = image[..., :3].float().clamp(min=0.0)
        # PQ is absolute (10 000 cd/m2 ceiling), so the decode takes no peak:
        # _eotf_pq() dropped its peak_nits argument on purpose and this call
        # still passed it, so every PQ run raised TypeError.
        linear = fn(img)

        # 2. Chromatic adaptation. A primaries conversion between different
        #    white points (anything to or from ACEScg, D60) already adapts, so
        #    a second adaptation on top is skipped rather than applied twice.
        adaptation_applied = "None"
        if chromatic_adaptation != "None":
            if source_primaries != target_primaries and _primaries_adapt_white(source_primaries, target_primaries):
                logger.warning(
                    "RadianceHDRColorPipeline: %s to %s already adapts the white point; "
                    "chromatic_adaptation %s skipped so it is not applied twice.",
                    source_primaries, target_primaries, chromatic_adaptation)
            else:
                M_cat = _BRADFORD_CAT.get(chromatic_adaptation)
                if M_cat:
                    linear = _apply_matrix(linear, M_cat).clamp(min=0.0)
                    adaptation_applied = chromatic_adaptation

        # 3. Primaries transform. Every pair of the listed primaries converts;
        #    pairs outside five hard-coded ones used to pass through unchanged
        #    with no message.
        if source_primaries != target_primaries:
            M_prim = _primaries_matrix(source_primaries, target_primaries)
            if M_prim is None:
                raise ValueError(
                    f"RadianceHDRColorPipeline: no conversion from {source_primaries} to {target_primaries}.")
            linear = _apply_matrix(linear, M_prim).clamp(min=0.0)

        peak_linear = float(linear.max().item())

        # 4. Soft-knee compress → VAE range.
        # _soft_knee_compress is imported at module level from radiance.color.ops.
        compressed = _soft_knee_compress(linear, compression_ratio)

        # Re-attach alpha
        if image.shape[-1] == 4:
            alpha = image[..., 3:4]
            compressed = torch.cat([compressed, alpha], dim=-1)
            scene_out  = torch.cat([linear, alpha], dim=-1)
        else:
            scene_out = linear

        cs_info = {
            "encoding":             encoding,
            "source_primaries":     source_primaries,
            "target_primaries":     target_primaries,
            "chromatic_adaptation": chromatic_adaptation,
            "chromatic_adaptation_applied": adaptation_applied,
            "compression_ratio":    compression_ratio,
            "peak_linear":          round(peak_linear, 4),
        }
        colorspace_json = json.dumps(cs_info)

        logger.info(
            "RadianceHDRColorPipeline: %s→%s  adapt=%s  ratio=%.2f  peak=%.3f",
            encoding, target_primaries, chromatic_adaptation, compression_ratio, peak_linear,
        )
        diag_logger.info(
            "HDR_COLOR_PIPELINE encoding=%s src=%s tgt=%s adapt=%s peak_linear=%.4f",
            encoding, source_primaries, target_primaries, chromatic_adaptation, peak_linear,
        )

        return (compressed, scene_out, peak_linear, colorspace_json)


# ─────────────────────────────────────────────────────────────────────────────
# Node 5 — RadianceColorSpaceInfo
# ─────────────────────────────────────────────────────────────────────────────

class RadianceColorSpaceInfo:
    """
    ◎ Radiance Color Space Info

    Utility node — outputs colour space metadata as a JSON string.
    Use to wire colour space information to:
      • RadianceHDRDiagnostics          (structured report)
      • RadianceHDRDiagnostics          (structured report)
      • ShowText nodes                  (display in graph UI)

    Does not modify the image — purely metadata.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Image passed through unchanged; only its size, channels and batch are read.",
                }),
                "encoding": (list(_EOTF_MAP.keys()), {
                    "default": "sRGB",
                    "tooltip": "Transfer curve to record in the JSON. Metadata only; the image is not decoded.",
                }),
                "primaries": (_PRIMARIES_LIST, {
                    "default": "Rec.709 (sRGB)",
                    "tooltip": "Primaries to record in the JSON. Metadata only; no conversion is done.",
                }),
            },
            "optional": {
                "scene_referred": ("BOOLEAN", {"default": False,
                    "tooltip": "Records in the JSON that the image is scene-referred (values above 1.0 are valid). Metadata only; nothing is clamped either way."
                }),
                "peak_nits": ("FLOAT", {
                    "default": 100.0, "min": 80.0, "max": 10000.0, "step": 10.0,
                    "tooltip": "Mastering display peak in nits to record in the JSON. Metadata only; pixel values are not scaled."
                }),
                "notes": ("STRING", {"default": "", "multiline": False,
                    "tooltip": "Optional free-text notes stored alongside the colorspace metadata JSON output."
                }),
            },
        }

    RETURN_TYPES  = ("IMAGE",  "STRING")
    RETURN_NAMES  = ("image",  "colorspace_json")
    FUNCTION      = "info"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    DESCRIPTION   = "Output colour space metadata as JSON. Image is passed through unchanged."

    def info(
        self,
        image: torch.Tensor,
        encoding: str = "sRGB",
        primaries: str = "Rec.709 (sRGB)",
        scene_referred: bool = False,
        peak_nits: float = 100.0,
        notes: str = "",
    ):
        B, H, W, C = image.shape
        meta = {
            "encoding":       encoding,
            "primaries":      primaries,
            "scene_referred": scene_referred,
            "peak_nits":      peak_nits,
            "resolution":     [W, H],
            "channels":       C,
            "batch":          B,
            "notes":          notes,
        }
        return (image, json.dumps(meta, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# ComfyUI registration
# ─────────────────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "RadianceHDRColorPipeline": RadianceHDRColorPipeline,
    "RadianceColorSpaceInfo":   RadianceColorSpaceInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceHDRColorPipeline": "◎ Radiance HDR Color Pipeline",
    "RadianceColorSpaceInfo":   "◎ Radiance Color Space Info",
}
