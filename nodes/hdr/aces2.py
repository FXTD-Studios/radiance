"""
nodes_aces2.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Radiance · ACES 2.0 Full Implementation

Nodes
─────
  RadianceACES2Tonescale          ACES 2.0 reference tone scale (or a labelled creative curve)
  RadianceACES2ReachGamutCompress ACES 2.0 reach-based gamut compression
  RadianceACES2OutputTransformFull ACES 2.0 Output Transform (OCIO reference; labelled approximation fallback)
  RadianceACESMetadataFile        Write / read ACES Metadata File (AMF) sidecar
  RadianceACES2Compliance         ACES 2.0 Output Check (compare against the OCIO reference)

Math references
───────────────
  • ACES 2.0 DRT design: Daniele Evo tonescale (Scott Dyer / Thomas Mansencal)
      Academy ACES project — github.com/ampas/aces-dev
  • AMF specification: Academy TB-2014-009 / S-2019-001
  • ACES 2.0 reference Output Transforms: OpenColorIO built-in
      studio-config-v4.0.0_aces-v2.0_ocio-v2.5 (radiance.hdr.aces2_ocio)
  • Reach gamut compression: ACES Gamut Compression Implementation Guide
"""

from __future__ import annotations

import json
import logging
import math
import os
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any

import numpy as np
import torch

from radiance.path_utils import strip_path_quotes
from radiance.core.tensor.alpha import alpha_passthrough

log = logging.getLogger("radiance.aces2")

# ─────────────────────────────────────────────────────────────────────────────
# Try defusedxml for safe XML parsing; fall back to stdlib
# ─────────────────────────────────────────────────────────────────────────────
try:
    from defusedxml import ElementTree as SafeET  # type: ignore
    _SAFE_XML = True
except ImportError:
    SafeET = ET  # type: ignore
    _SAFE_XML = False


# ─────────────────────────────────────────────────────────────────────────────
# Colour-science matrices (all float32, AP1 = ACEScg)
# ─────────────────────────────────────────────────────────────────────────────

# AP1 (ACEScg) → XYZ D65
AP1_TO_XYZ = np.array([
    [0.6624541811, 0.1340042065, 0.1561876744],
    [0.2722287168, 0.6740817658, 0.0536895174],
    [-0.0055746495, 0.0040607335, 1.0103391003],
], dtype=np.float32)

# XYZ D65 → AP1 (ACEScg)
XYZ_TO_AP1 = np.array([
    [1.6410233797, -0.3248032942, -0.2364246952],
    [-0.6636628587, 1.6153315917, 0.0167563477],
    [0.0117218943, -0.0082844420, 0.9883948585],
], dtype=np.float32)

# AP1 → sRGB / Rec.709 (D65)
AP1_TO_sRGB = np.array([
    [1.7050509, -0.6217921, -0.0832588],
    [-0.1302564, 1.1408047, -0.0105483],
    [-0.0240033, -0.1289690, 1.1529723],
], dtype=np.float32)

# AP1 → Rec.2020
AP1_TO_Rec2020 = np.array([
    [1.0258246, -0.0200540, -0.0057706],
    [-0.0023054, 1.0045847, -0.0022793],
    [-0.0050569, -0.0252857, 1.0303426],
], dtype=np.float32)

# AP1 → P3-D65
AP1_TO_P3D65 = np.array([
    [1.3792141, -0.3088546, -0.0703595],
    [-0.0693257, 1.0823507, -0.0130250],
    [-0.0021522, -0.0454616, 1.0476138],
], dtype=np.float32)

# AP1 → P3 with the ACES white (x=0.32168, y=0.33767), commonly called D60.
# Both spaces share that white, so no chromatic adaptation is required.
AP1_TO_P3D60 = np.array([
    [1.34954840, -0.28457603, -0.06497236],
    [-0.07004237, 1.08113398, -0.01109161],
    [-0.00251746, -0.04620062, 1.04871808],
], dtype=np.float32)

# AP0 (ACES 2065-1) → AP1 (ACEScg)
AP0_TO_AP1 = np.array([
    [1.4514393161, -0.2365107469, -0.2149285693],
    [-0.0765537734, 1.1762296998, -0.0996759264],
    [0.0083161484, -0.0060324498, 0.9977163014],
], dtype=np.float32)

# sRGB → AP1
sRGB_TO_AP1 = np.array([
    [0.6131, 0.3395, 0.0474],
    [0.0702, 0.9164, 0.0134],
    [0.0206, 0.1096, 0.8698],
], dtype=np.float32)

# Rec.2020 → AP1
Rec2020_TO_AP1 = np.array([
    [0.9788, 0.0165, 0.0047],
    [0.0014, 0.9983, 0.0003],
    [0.0044, 0.0235, 0.9721],
], dtype=np.float32)

# AP1 luminance weights
LUMA_AP1 = np.array([0.2722, 0.6741, 0.0537], dtype=np.float32)


# ═════════════════════════════════════════════════════════════════════════════
# § 1  DANIELE EVO TONESCALE
# ═════════════════════════════════════════════════════════════════════════════

class _DanieleEvoParams:
    """
    Pre-computed constants for the Daniele Evo tonescale.

    The curve is a smooth Hill / Michaelis-Menten rational:

        f(x) = n · (x/K)^g / ((x/K)^g + 1)

    where K is chosen so that:

        f(scene_grey) = grey_target · n

    i.e. the scene grey point (0.18) maps to grey_target fraction (0.10) of
    the display peak.  Below toe_scene the curve is extended linearly (slope
    matching at toe_scene), ensuring zero at zero with no lift.

    Parameters
    ──────────
    peak_nits   : display peak luminance (100 for SDR, 1000/2000/4000 for HDR)
    g           : contrast exponent      (1.15 — ACES 2.0 reference)
    scene_grey  : scene linear reference white / midtone (0.18)
    grey_target : fraction of peak to assign to scene_grey (0.10 standard)
    toe_scene   : scene luminance below which linear toe is applied (0.04)
    """

    # Default ACES 2.0 parameters (per Academy reference design)
    G          = 1.15    # contrast exponent
    SCENE_GREY = 0.18    # 18% grey in scene-linear
    GREY_TGT   = None    # resolved per peak from the published ACES 2.0 table
    TOE_SCENE  = 0.04    # linear toe threshold

    def __init__(
        self,
        peak_nits: float  = 100.0,
        g:         float  = G,
        scene_grey: float = SCENE_GREY,
        grey_target: float = None,
        toe_scene:  float = TOE_SCENE,
    ):
        # grey_target used to default to a flat 0.10 — 10% of peak, whatever
        # the peak was — so 18% grey rendered at 10 nits on SDR but 400 nits on
        # a 4000-nit master, nearly 24x the ACES 2.0 reference. The published
        # table has grey rising gently (10.0 / 14.5 / 16.8 nits at 100 / 1000 /
        # 4000), not tracking the display.
        if grey_target is None:
            from radiance.hdr.tonescale import aces2_midgrey_fraction
            grey_target = aces2_midgrey_fraction(peak_nits)

        self.n          = peak_nits / 100.0   # normalised peak
        self.g          = g
        self.scene_grey = scene_grey
        self.grey_target = grey_target
        self.toe_scene  = toe_scene

        # Solve for K: n * (c/K)^g / ((c/K)^g + 1) = grey_target * n
        #  → (c/K)^g = grey_target / (1 - grey_target)
        #  → K = c / (grey_target / (1 - grey_target))^(1/g)
        ratio = grey_target / (1.0 - grey_target)
        self.K = scene_grey / (ratio ** (1.0 / g))

        # Derivative at toe_scene (for linear extrapolation)
        # f'(x) = n * g * u / (x * (u+1)^2)   where u = (x/K)^g
        u_t = (toe_scene / self.K) ** g
        self.toe_y  = self.n * u_t / (u_t + 1.0)
        self.toe_dy = self.n * g * u_t / (toe_scene * (u_t + 1.0) ** 2)


def _daniele_evo_fwd(
    x: np.ndarray,
    params: _DanieleEvoParams,
) -> np.ndarray:
    """
    Apply the forward Daniele Evo tonescale (per-value, broadcastable).

    Returns display-referred luminance in the range [0, n] where n =
    peak_nits / 100.  Divide by n for a [0,1]-normalised output.
    """
    xp = np.maximum(x, 0.0)

    # Main rational curve
    u = (xp / params.K) ** params.g
    y_main = params.n * u / (u + 1.0)

    # Linear toe below toe_scene
    y_toe = np.maximum(
        params.toe_y + params.toe_dy * (xp - params.toe_scene),
        0.0,
    )

    return np.where(xp < params.toe_scene, y_toe, y_main)


def _daniele_evo_luma_preserving(
    rgb: np.ndarray,
    params: _DanieleEvoParams,
) -> np.ndarray:
    """
    Apply Daniele Evo on luminance, then reconstruct RGB.
    Preserves hue/saturation better than per-channel mapping.
    """
    luma = rgb @ LUMA_AP1   # (H, W)

    luma_mapped = _daniele_evo_fwd(luma, params)

    # Protect against divide-by-zero (pure black pixels)
    scale = np.where(luma > 1e-10, luma_mapped / (luma + 1e-10), 0.0)

    return rgb * scale[..., np.newaxis]


# ═════════════════════════════════════════════════════════════════════════════
# § 2  REACH GAMUT COMPRESSION
# ═════════════════════════════════════════════════════════════════════════════

# ACES 2.0 reach gamut limits per channel (in AP1 space)
# Source: Academy ACES Gamut Compression Implementation Guide v1.0.1
_REACH_LIMIT   = np.array([1.147, 1.264, 1.312], dtype=np.float32)  # Cyan, Magenta, Yellow
_REACH_THRESH  = np.array([0.815, 0.803, 0.880], dtype=np.float32)
_REACH_POWER   = 1.2    # compression power parameter


def _reach_compress_channel(
    dist: np.ndarray,
    threshold: float,
    limit: float,
    power: float = _REACH_POWER,
) -> np.ndarray:
    """
    Smooth power-function compression for one channel distance.

    Maps distances in (threshold, limit] → (threshold, 1.0], smoothly
    asymptoting to 1.0 at the limit.

    compress(d) = threshold + (d - threshold) /
                  (1 + ((d - threshold) / (limit - threshold))^power)^(1/power)
    """
    above = dist > threshold
    xm = np.maximum(dist - threshold, 0.0)
    scale = (limit - threshold)
    compressed = threshold + xm / np.power(1.0 + np.power(xm / scale, power), 1.0 / power)
    return np.where(above, compressed, dist)


def _reach_gamut_compress(
    rgb: np.ndarray,
    strength: float = 1.0,
    limits:    np.ndarray = _REACH_LIMIT,
    thresholds: np.ndarray = _REACH_THRESH,
) -> np.ndarray:
    """
    ACES 2.0 reach-based gamut compression (AP1 working space).

    Algorithm
    ─────────
    1. Find achromatic (max channel) — "nesting" reference.
    2. Compute per-channel distance from achromatic:
           dist_c = (ach - c) / ach
    3. Compress each distance channel with its own (threshold, limit) pair.
    4. Reconstruct RGB from compressed distances.

    Parameters
    ──────────
    rgb        : (H, W, 3) scene-linear ACEScg
    strength   : 0 = bypass, 1 = standard ACES compression, >1 = extra squeeze
    limits     : per-channel reach limits [cyan, magenta, yellow]
    thresholds : per-channel compression thresholds
    """
    if strength <= 0.0:
        return rgb

    # Scale limits toward 1 by (1-strength) for weaker compression
    eff_limits = 1.0 + (limits - 1.0) * strength
    eff_thresh = thresholds * (1.0 - (1.0 - strength) * 0.3)  # slight threshold shift

    ach = np.maximum(
        rgb[..., 0],
        np.maximum(rgb[..., 1], rgb[..., 2]),
    )
    ach = np.maximum(ach, 1e-10)

    # Per-channel distance from achromatic axis (CMY directions)
    dist = (ach[..., np.newaxis] - rgb) / ach[..., np.newaxis]

    # Compress each channel
    dist_c = np.zeros_like(dist)
    for ch in range(3):
        dist_c[..., ch] = _reach_compress_channel(
            dist[..., ch],
            float(eff_thresh[ch]),
            float(eff_limits[ch]),
        )

    return ach[..., np.newaxis] - dist_c * ach[..., np.newaxis]


# ═════════════════════════════════════════════════════════════════════════════
# § 3  EOTF ENCODERS
# ═════════════════════════════════════════════════════════════════════════════

def _pq_encode(linear: np.ndarray) -> np.ndarray:
    """ST.2084 PQ OETF.  Input: display-linear, 1.0 = 100 cd/m².

    No ``peak_nits``: ST.2084 normalises by a fixed 10 000 cd/m² and the
    mastering peak is applied by the caller as a pre-scale.  The parameter used
    to be accepted here and ignored, which read as though it did something.
    """
    L = np.clip(linear * 100.0 / 10000.0, 0, 1)
    m1, m2 = 0.1593017578125, 78.84375
    c1, c2, c3 = 0.8359375, 18.8515625, 18.6875
    Lm1 = np.power(np.maximum(L, 0), m1)
    return np.power((c1 + c2 * Lm1) / (1.0 + c3 * Lm1), m2)


def _hlg_encode(linear: np.ndarray) -> np.ndarray:
    """ARIB STD-B67 HLG OETF.  Input: display-linear [0, 1]."""
    a, b, c = 0.17883277, 0.28466892, 0.55991073
    return np.clip(
        np.where(
            linear <= 1.0 / 12.0,
            np.sqrt(3.0 * np.maximum(linear, 0.0)),
            a * np.log(np.maximum(12.0 * linear - b, 1e-10)) + c,
        ),
        0, 1,
    )


def _srgb_encode(linear: np.ndarray) -> np.ndarray:
    """sRGB / Rec.709 OETF."""
    return np.where(
        linear <= 0.0031308,
        12.92 * np.maximum(linear, 0.0),
        1.055 * np.power(np.maximum(linear, 1e-10), 1.0 / 2.4) - 0.055,
    )


def _as_torch_matrix(values: np.ndarray, image: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(values, device=image.device, dtype=image.dtype)


def _torch_apply_matrix(rgb: torch.Tensor, matrix: np.ndarray) -> torch.Tensor:
    return rgb @ _as_torch_matrix(matrix, rgb).T


def _torch_daniele_evo_fwd(x: torch.Tensor, params: _DanieleEvoParams) -> torch.Tensor:
    xp = x.clamp(min=0.0)
    u = (xp / params.K).clamp(min=0.0) ** params.g
    y_main = params.n * u / (u + 1.0)
    y_toe = (params.toe_y + params.toe_dy * (xp - params.toe_scene)).clamp(min=0.0)
    return torch.where(xp < params.toe_scene, y_toe, y_main)


def _torch_daniele_evo_luma_preserving(rgb: torch.Tensor, params: _DanieleEvoParams) -> torch.Tensor:
    luma_w = _as_torch_matrix(LUMA_AP1, rgb)
    luma = (rgb * luma_w).sum(dim=-1, keepdim=True)
    mapped = _torch_daniele_evo_fwd(luma, params)
    scale = torch.where(luma > 1e-10, mapped / luma.clamp(min=1e-10), torch.zeros_like(luma))
    return rgb * scale


def _torch_reach_compress_channel(
    dist: torch.Tensor,
    threshold: torch.Tensor,
    limit: torch.Tensor,
    power: float = _REACH_POWER,
) -> torch.Tensor:
    above = dist > threshold
    xm = (dist - threshold).clamp(min=0.0)
    scale = (limit - threshold).clamp(min=1e-8)
    compressed = threshold + xm / (1.0 + (xm / scale).pow(power)).pow(1.0 / power)
    return torch.where(above, compressed, dist)


def _torch_reach_gamut_compress(
    rgb: torch.Tensor,
    strength: float = 1.0,
    limits: torch.Tensor | None = None,
    thresholds: torch.Tensor | None = None,
) -> torch.Tensor:
    if strength <= 0.0:
        return rgb
    if limits is None:
        limits = _as_torch_matrix(_REACH_LIMIT, rgb)
    if thresholds is None:
        thresholds = _as_torch_matrix(_REACH_THRESH, rgb)

    eff_limits = 1.0 + (limits - 1.0) * strength
    eff_thresh = thresholds * (1.0 - (1.0 - strength) * 0.3)
    ach = rgb.max(dim=-1, keepdim=True).values.clamp(min=1e-10)
    dist = (ach - rgb) / ach
    dist_c = _torch_reach_compress_channel(dist, eff_thresh.view(1, 1, 1, 3), eff_limits.view(1, 1, 1, 3))
    return ach - dist_c * ach


def _torch_pq_encode(linear: torch.Tensor) -> torch.Tensor:
    """ST.2084 PQ OETF.  Input: display-linear, 1.0 = 100 cd/m².

    See :func:`_pq_encode` for why there is no ``peak_nits`` parameter.
    """
    L = (linear * 100.0 / 10000.0).clamp(0.0, 1.0)
    m1, m2 = 0.1593017578125, 78.84375
    c1, c2, c3 = 0.8359375, 18.8515625, 18.6875
    Lm1 = L.clamp(min=0.0).pow(m1)
    return ((c1 + c2 * Lm1) / (1.0 + c3 * Lm1)).pow(m2)


def _torch_hlg_encode(linear: torch.Tensor) -> torch.Tensor:
    a, b, c = 0.17883277, 0.28466892, 0.55991073
    return torch.where(
        linear <= 1.0 / 12.0,
        torch.sqrt((3.0 * linear).clamp(min=0.0)),
        a * torch.log((12.0 * linear - b).clamp(min=1e-10)) + c,
    ).clamp(0.0, 1.0)


def _torch_srgb_encode(linear: torch.Tensor) -> torch.Tensor:
    return torch.where(
        linear <= 0.0031308,
        12.92 * linear.clamp(min=0.0),
        1.055 * linear.clamp(min=1e-10).pow(1.0 / 2.4) - 0.055,
    )


# ═════════════════════════════════════════════════════════════════════════════
# § 4  AMF BUILDER / PARSER
# ═════════════════════════════════════════════════════════════════════════════

_AMF_NS = "urn:ampas:aces:amf:v1.0"
_AMF_ACES_NS = {"aces": _AMF_NS}

_AMF_TEMPLATE = """\
<?xml version="1.0" encoding="UTF-8"?>
<!--
    ACES Metadata File (AMF)
    Generated by Radiance · Academy TB-2014-009 / S-2019-001
-->
<aces:aces xmlns:aces="{ns}" version="2.0">
  <aces:amfInfo>
    <aces:uuid>{uuid}</aces:uuid>
    <aces:dateTime>{dt}</aces:dateTime>
    <aces:description>{description}</aces:description>
  </aces:amfInfo>
  <aces:clipID>
    <aces:clipName>{clip_name}</aces:clipName>
  </aces:clipID>
  <aces:pipeline>
    <aces:pipelineInfo>
      <aces:systemVersion>
        <aces:majorVersion>2</aces:majorVersion>
        <aces:minorVersion>0</aces:minorVersion>
        <aces:patchVersion>0</aces:patchVersion>
      </aces:systemVersion>
    </aces:pipelineInfo>
    <aces:inputTransform>
      <aces:transformId>{input_transform}</aces:transformId>
    </aces:inputTransform>
    <aces:outputTransform>
      <aces:transformId>{output_transform}</aces:transformId>
      <aces:outputDeviceInfo>
        <aces:peakLuminance>{peak_nits}</aces:peakLuminance>
        <aces:minLuminance>{min_nits}</aces:minLuminance>
      </aces:outputDeviceInfo>
    </aces:outputTransform>
  </aces:pipeline>
</aces:aces>
"""


def _build_amf(
    clip_name:       str,
    input_transform: str,
    output_transform: str,
    peak_nits:       float,
    min_nits:        float,
    description:     str,
) -> str:
    """Return an AMF XML string."""
    import uuid
    uid  = str(uuid.uuid4())
    dt   = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Text is XML-escaped: a description with < or & (or a clip name with
    # them) used to be inserted verbatim and made the file unparseable.
    from xml.sax.saxutils import escape as _xml_escape
    return _AMF_TEMPLATE.format(
        ns               = _AMF_NS,
        uuid             = uid,
        dt               = dt,
        description      = _xml_escape(str(description)),
        clip_name        = _xml_escape(str(clip_name)),
        input_transform  = _xml_escape(str(input_transform)),
        output_transform = _xml_escape(str(output_transform)),
        peak_nits        = peak_nits,
        min_nits         = min_nits,
    )


def _parse_amf(xml_str: str) -> dict:
    """Parse an AMF XML string into a plain dict."""
    root = SafeET.fromstring(xml_str)

    def _find(path: str) -> str:
        el = root.find(path, {"aces": _AMF_NS})
        return el.text.strip() if el is not None and el.text else ""

    return {
        "uuid":             _find("aces:amfInfo/aces:uuid"),
        "dateTime":         _find("aces:amfInfo/aces:dateTime"),
        "description":      _find("aces:amfInfo/aces:description"),
        "clipName":         _find("aces:clipID/aces:clipName"),
        "inputTransform":   _find("aces:pipeline/aces:inputTransform/aces:transformId"),
        "outputTransform":  _find("aces:pipeline/aces:outputTransform/aces:transformId"),
        "peakLuminance":    _find("aces:pipeline/aces:outputTransform/aces:outputDeviceInfo/aces:peakLuminance"),
        "minLuminance":     _find("aces:pipeline/aces:outputTransform/aces:outputDeviceInfo/aces:minLuminance"),
    }


# ═════════════════════════════════════════════════════════════════════════════
# NODE 1 — RadianceACES2Tonescale
# ═════════════════════════════════════════════════════════════════════════════

class RadianceACES2Tonescale:
    """
    Apply the ACES 2.0 tone scale to scene-linear ACEScg.

    grey_target = 0 (default) runs the ACES 2.0 Output Transform's own tone
    scale (radiance.hdr.tonescale.ACES2TonescaleParams), verified against the
    OCIO ACES 2.0 studio config. A grey_target above 0 switches to a creative
    Hill curve that places 18% grey at that fraction of peak; it is labelled
    as such in curve_info and is not the reference curve.

    This is the tone scale only, on luminance or per channel. The full Output
    Transform (JMh, chroma compression, gamut mapping) is
    RadianceACES2OutputTransformFull.
    """

    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    DESCRIPTION = "Apply the ACES 2.0 Tonescale operator with configurable parameters."
    FUNCTION     = "apply"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "curve_info")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Scene-linear ACEScg (AP1) image batch.",
                }),
                "peak_nits": ("FLOAT", {
                    "default": 100.0, "min": 48.0, "max": 10000.0, "step": 1.0,
                    "tooltip": "Display peak luminance in cd/m². "
                               "100 = SDR, 1000/2000/4000 = HDR.",
                }),
                "mode": (["luminance_preserving", "per_channel"], {
                    "default": "luminance_preserving",
                    "tooltip": (
                        "luminance_preserving: tone-map luma then scale RGB — "
                        "preserves hue/saturation.  "
                        "per_channel: apply curve independently to R, G, B — "
                        "may introduce hue shifts but avoids colour casts."
                    ),
                }),
            },
            "optional": {
                "contrast_g": ("FLOAT", {
                    "default": 1.15, "min": 0.8, "max": 1.6, "step": 0.01,
                    "tooltip": "Contrast exponent g. ACES 2.0 reference = 1.15.",
                }),
                "grey_target": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 0.30, "step": 0.005,
                    "tooltip": "0 = ACES 2.0 reference: the Output Transform's own tone scale, "
                               "18% grey at 10.0 / 14.5 / 16.8 nits for 100 / 1000 / 4000-nit peaks. "
                               "Above 0 = creative: 18% grey placed at this fraction of peak on a "
                               "Hill curve (not the ACES 2.0 curve).",
                }),
                "toe_scene": ("FLOAT", {
                    "default": 0.04, "min": 0.005, "max": 0.20, "step": 0.005,
                    "tooltip": "Scene luminance below which a linear toe is applied. "
                               "Prevents gamma lift in deep shadows.",
                }),
            },
        }

    def apply(
        self,
        image: torch.Tensor,
        peak_nits: float,
        mode: str,
        contrast_g: float = 1.15,
        grey_target: float = 0.0,
        toe_scene: float = 0.04,
    ):
        from radiance.core.tensor.alpha import split_rgb_alpha, merge_rgb_alpha
        rgb, alpha = split_rgb_alpha(image)
        rgb = rgb.float()

        # FIX-001: the widget used to default to grey_target=0.10, which bypassed
        # the per-peak ACES grey and put 18% grey at 10% of peak (100 nits on a
        # 1000-nit master). 0 now selects the ACES 2.0 reference tone scale.
        reference = grey_target <= 0.0
        if reference:
            from radiance.hdr.tonescale import ACES2TonescaleParams
            ts = ACES2TonescaleParams(peak_nits, g=contrast_g, t_1=toe_scene)
            n = ts.n
            curve = ts.forward
            grey_nits = float(ts.forward(0.18)) * 100.0
        else:
            params = _DanieleEvoParams(
                peak_nits  = peak_nits,
                g          = contrast_g,
                grey_target= grey_target,
                toe_scene  = toe_scene,
            )
            n = params.n
            curve = lambda x: _torch_daniele_evo_fwd(x, params)  # noqa: E731
            grey_nits = grey_target * peak_nits

        if mode == "luminance_preserving":
            luma_w = _as_torch_matrix(LUMA_AP1, rgb)
            luma = (rgb * luma_w).sum(dim=-1, keepdim=True)
            mapped_l = curve(luma)
            scale = torch.where(luma > 1e-10, mapped_l / luma.clamp(min=1e-10), torch.zeros_like(luma))
            mapped = rgb * scale
        else:
            mapped = curve(rgb)
        out = merge_rgb_alpha((mapped / n).clamp(0.0, 1.0), alpha)

        if reference:
            info = (
                f"ACES 2.0 reference tone scale · peak {peak_nits:g} nits · g={contrast_g:.2f} · "
                f"toe {toe_scene:.3f} · 18% grey -> {grey_nits:.3f} nits"
            )
        else:
            info = (
                f"Creative Hill curve (not ACES 2.0 reference) · peak {peak_nits:g} nits · "
                f"g={contrast_g:.2f} · grey {grey_target*100:.1f}% of peak ({grey_nits:.1f} nits) · "
                f"toe {toe_scene:.3f} | K={params.K:.4f}"
            )
        log.info("ACES2Tonescale: %s", info)

        return (out, info)


# ═════════════════════════════════════════════════════════════════════════════
# NODE 2 — RadianceACES2ReachGamutCompress
# ═════════════════════════════════════════════════════════════════════════════

# FIX-008: gamut compression is colour maths; alpha passes through.
@alpha_passthrough("image")
class RadianceACES2ReachGamutCompress:
    """
    ACES 2.0 Reach-Based Gamut Compression.

    Maps out-of-gamut scene colours in AP1 (ACEScg) space back within the
    target gamut using a per-channel smooth power-function compression.

    The reach gamut is the largest well-defined gamut boundary for each colour
    axis (cyan / magenta / yellow) beyond which compression is applied.  This
    replaces the simpler threshold-clip approach in the legacy ACES node.

    Default limits follow the Academy ACES Gamut Compression Guide v1.0.1:
      Cyan   limit 1.147  threshold 0.815
      Magenta limit 1.264  threshold 0.803
      Yellow  limit 1.312  threshold 0.880
    """

    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    DESCRIPTION = "Compress out-of-gamut values using the ACES 2.0 Reach Gamut method."
    FUNCTION     = "compress"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "compress_info")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Scene-linear ACEScg (AP1) image batch.",
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.5, "step": 0.05,
                    "tooltip": (
                        "Compression strength.  0 = bypass.  "
                        "1.0 = standard ACES 2.0.  "
                        ">1 = more aggressive squeeze for difficult footage."
                    ),
                }),
            },
            "optional": {
                "limit_cyan": ("FLOAT", {
                    "default": 1.147, "min": 1.0, "max": 2.0, "step": 0.01,
                    "tooltip": "Reach gamut limit for cyan channel (R).",
                }),
                "limit_magenta": ("FLOAT", {
                    "default": 1.264, "min": 1.0, "max": 2.0, "step": 0.01,
                    "tooltip": "Reach gamut limit for magenta channel (G).",
                }),
                "limit_yellow": ("FLOAT", {
                    "default": 1.312, "min": 1.0, "max": 2.0, "step": 0.01,
                    "tooltip": "Reach gamut limit for yellow channel (B).",
                }),
                "threshold_cyan": ("FLOAT", {
                    "default": 0.815, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Distance from neutral (on R) where cyan compression starts. Lower protects fewer colours; below strength 1 it is also lowered slightly.",
                }),
                "threshold_magenta": ("FLOAT", {
                    "default": 0.803, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Distance from neutral (on G) where magenta compression starts. Lower protects fewer colours; below strength 1 it is also lowered slightly.",
                }),
                "threshold_yellow": ("FLOAT", {
                    "default": 0.880, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Distance from neutral (on B) where yellow compression starts. Lower protects fewer colours; below strength 1 it is also lowered slightly.",
                }),
            },
        }

    def compress(
        self,
        image: torch.Tensor,
        strength: float,
        limit_cyan:        float = 1.147,
        limit_magenta:     float = 1.264,
        limit_yellow:      float = 1.312,
        threshold_cyan:    float = 0.815,
        threshold_magenta: float = 0.803,
        threshold_yellow:  float = 0.880,
    ):
        frames = image.float()
        limits = torch.tensor(
            [limit_cyan, limit_magenta, limit_yellow],
            device=frames.device,
            dtype=frames.dtype,
        )
        thresholds = torch.tensor(
            [threshold_cyan, threshold_magenta, threshold_yellow],
            device=frames.device,
            dtype=frames.dtype,
        )
        out = _torch_reach_gamut_compress(frames, strength, limits, thresholds)

        # Count pixels brought in-gamut
        before_oog = int(((frames < 0) | (frames > 1)).sum().item())
        after_oog  = int(((out    < 0) | (out    > 1)).sum().item())
        pct_before = before_oog / max(frames.numel(), 1) * 100
        pct_after  = after_oog  / max(out.numel(), 1)   * 100

        info = (
            f"ACES 2.0 reach gamut compress · strength {strength:.2f} | "
            f"out-of-gamut pixels: {pct_before:.2f}% → {pct_after:.2f}%"
        )
        log.info("ACES2ReachGamutCompress: %s", info)

        return (out, info)


# ═════════════════════════════════════════════════════════════════════════════
# NODE 3 — RadianceACES2OutputTransformFull
# ═════════════════════════════════════════════════════════════════════════════

class RadianceACES2OutputTransformFull:
    """
    ACES 2.0 Output Transform.

    Radiance approximation pipeline (engine fallback):
      1. Input colorspace conversion → ACEScg (AP1) scene-linear
      2. Creative white scale + exposure
      3. Reach gamut compression (ACES 2.0 official)
      4. Daniele Evo tonescale (ACES 2.0 official forward curve)
      5. Output gamut matrix (AP1 → sRGB / P3-D65 / Rec.2020)
      6. OETF encoding (sRGB / PQ / HLG / DCI γ2.6)

    engine="Auto" (default) runs the Academy reference through OpenColorIO's
    pinned ACES 2.0 studio config (radiance.hdr.aces2_ocio). The steps above
    are the Radiance approximation, used only when the reference cannot run
    (OCIO < 2.5, or DCI-P3 D60 which the config does not pair) and labelled
    "APPROXIMATION" in transform_info.
    """

    OUTPUT_TRANSFORMS = [
        "ACES 2.0 SDR (sRGB/Rec.709)",
        "ACES 2.0 SDR (P3-D65)",
        "ACES 2.0 HDR (Rec.2100 PQ 1000 nits)",
        "ACES 2.0 HDR (Rec.2100 PQ 2000 nits)",
        "ACES 2.0 HDR (Rec.2100 PQ 4000 nits)",
        "ACES 2.0 HDR (Rec.2100 HLG)",
        "ACES 2.0 Cinema (DCI-P3 D60)",
        "ACES 2.0 Cinema (DCI-P3 D65)",
    ]

    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    DESCRIPTION = "Full ACES 2.0 Output Transform (RRT + ODT) for display rendering."
    FUNCTION     = "transform"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "transform_info")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Scene-linear image in the space chosen by input_colorspace (18% grey = 0.18). A display-encoded sRGB image must be linearised first."}),
                "input_colorspace": (
                    ["ACEScg", "ACES2065-1", "Linear_sRGB", "Linear_Rec2020"],
                    {"default": "ACEScg", "tooltip": "Primaries of the incoming linear image; it is converted to ACEScg (AP1) before the transform. No transfer curve is removed."},
                ),
                "output_transform": (cls.OUTPUT_TRANSFORMS, {
                    "default": "ACES 2.0 SDR (sRGB/Rec.709)",
                    "tooltip": "Target display: sets primaries, encoding (sRGB, PQ, HLG or gamma 2.6) and peak. Output is display-encoded 0 to 1. Cinema D60 targets P3 at ACES white; Cinema D65 targets P3 at D65. Match the display calibration.",
                }),
            },
            "optional": {
                "peak_luminance": ("FLOAT", {
                    "default": 100.0, "min": 48.0, "max": 10000.0, "step": 1.0,
                    "tooltip": "Display peak luminance in nits, used only by the Radiance approximation for the SDR and HLG outputs. "
                               "The OCIO reference fixes the peak by the output's name (SDR and Cinema 100 nits, HLG 1000 nits).",
                }),
                "surround": (["Dark", "Dim", "Average"], {
                    "default": "Dim",
                    "tooltip": "Viewing environment — affects contrast parameter g.",
                }),
                "exposure_adjust": ("FLOAT", {
                    "default": 0.0, "min": -4.0, "max": 4.0, "step": 0.1,
                    "tooltip": "Exposure adjustment in stops before transform.",
                }),
                "creative_white_scale": ("FLOAT", {
                    "default": 1.0, "min": 0.5, "max": 2.0, "step": 0.01,
                    "tooltip": "Linear multiplier on all channels after conversion to ACEScg, so it acts as extra exposure (2.0 = +1 stop), not a white point change. 1.0 = no change.",
                }),
                "gamut_compress_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.5, "step": 0.05,
                    "tooltip": "Reach gamut compression strength. Radiance approximation only: the OCIO reference has its own gamut mapping and ignores this.",
                }),
                "engine": (cls.ENGINES, {
                    "default": cls.ENGINES[0],
                    "tooltip": "Auto: the Academy reference through OpenColorIO's pinned ACES 2.0 studio config "
                               "(OCIO >= 2.5), else the Radiance approximation, labelled as such in transform_info. "
                               "OCIO reference: fail rather than fall back. Radiance approximation: the luminance "
                               "tone scale + reach compression math; not the reference.",
                }),
            },
        }

    ENGINES = ["Auto", "OCIO reference", "Radiance approximation"]

    @staticmethod
    def _peak_nits(output_transform: str, peak_luminance: float) -> float:
        if "4000" in output_transform: return 4000.0
        if "2000" in output_transform: return 2000.0
        if "1000" in output_transform: return 1000.0
        if "Cinema" in output_transform: return 48.0
        return peak_luminance

    @staticmethod
    def _surround_g(surround: str) -> float:
        """ACES 2.0 surround adapts the contrast exponent."""
        return {"Dark": 1.10, "Dim": 1.15, "Average": 1.20}.get(surround, 1.15)

    def transform(
        self,
        image:                    torch.Tensor,
        input_colorspace:         str,
        output_transform:         str,
        peak_luminance:           float = 100.0,
        surround:                 str   = "Dim",
        exposure_adjust:          float = 0.0,
        creative_white_scale:     float = 1.0,
        gamut_compress_strength:  float = 1.0,
        engine:                   str   = "Auto",
    ):
        # FIX-002: production output goes through the Academy reference (OCIO,
        # pinned config). The Radiance math is kept as a labelled approximation.
        from radiance.core.tensor.alpha import split_rgb_alpha, merge_rgb_alpha
        from radiance.hdr import aces2_ocio

        rgb_all, alpha = split_rgb_alpha(image)
        if engine != "Radiance approximation":
            ok, reason = aces2_ocio.reference_available(output_transform)
            if ok:
                return self._transform_reference(
                    rgb_all, alpha, input_colorspace, output_transform,
                    exposure_adjust, creative_white_scale, surround,
                    peak_luminance, gamut_compress_strength)
            if engine == "OCIO reference":
                raise RuntimeError(f"ACES 2.0 Output Transform: OCIO reference unavailable: {reason}")
            log.warning("ACES2FullOT: OCIO reference unavailable (%s); using the Radiance approximation.", reason)
            approx_reason = reason
        else:
            approx_reason = "selected"

        out, info = self._transform_approx(
            rgb_all, input_colorspace, output_transform, peak_luminance, surround,
            exposure_adjust, creative_white_scale, gamut_compress_strength)
        info = f"APPROXIMATION, not the ACES 2.0 reference ({approx_reason}) | " + info
        return (merge_rgb_alpha(out, alpha), info)

    def _transform_reference(self, rgb, alpha, input_colorspace, output_transform,
                             exposure_adjust, creative_white_scale, surround,
                             peak_luminance, gamut_compress_strength):
        from radiance.core.tensor.alpha import merge_rgb_alpha
        from radiance.hdr import aces2_ocio

        gain = (2.0 ** exposure_adjust) * creative_white_scale
        arr = rgb.detach().float().cpu().numpy()
        if gain != 1.0:
            arr = arr * gain
        frames = [aces2_ocio.apply_reference(f, input_colorspace, output_transform) for f in arr.reshape(-1, *arr.shape[-3:])]
        out_np = np.stack(frames).reshape(arr.shape)
        out = torch.from_numpy(out_np).to(device=rgb.device, dtype=torch.float32).clamp(0.0, 1.0)
        ignored = []
        if surround != "Dim":
            ignored.append("surround")
        if gamut_compress_strength != 1.0:
            ignored.append("gamut_compress_strength")
        if ("SDR" in output_transform or "HLG" in output_transform or "Cinema" in output_transform) \
                and peak_luminance != 100.0:
            ignored.append("peak_luminance")
        info = f"ACES 2.0 reference | {input_colorspace} → {output_transform} | {aces2_ocio.describe(output_transform)}"
        if "Cinema" in output_transform:
            info += " | ACES 2.0 100-nit P3-D65 rendering, gamma 2.6 (the pinned config has no 48-nit view)"
        elif "HLG" in output_transform:
            info += " | HLG from the 1000-nit P3-D65-limited rendering"
        if ignored:
            info += " | ignored by the reference: " + ", ".join(ignored)
        log.info("ACES2FullOT: %s", info)
        return (merge_rgb_alpha(out, alpha), info)

    def _transform_approx(self, image, input_colorspace, output_transform, peak_luminance,
                          surround, exposure_adjust, creative_white_scale,
                          gamut_compress_strength):
        rgb = image.float()

        if exposure_adjust != 0.0:
            rgb = rgb * (2.0 ** exposure_adjust)

        if input_colorspace == "ACES2065-1":
            rgb = _torch_apply_matrix(rgb, AP0_TO_AP1)
        elif input_colorspace == "Linear_sRGB":
            rgb = _torch_apply_matrix(rgb, sRGB_TO_AP1)
        elif input_colorspace == "Linear_Rec2020":
            rgb = _torch_apply_matrix(rgb, Rec2020_TO_AP1)

        rgb = rgb * creative_white_scale
        rgb = _torch_reach_gamut_compress(rgb, gamut_compress_strength)

        peak_nits = self._peak_nits(output_transform, peak_luminance)
        is_hdr = "HDR" in output_transform
        g = self._surround_g(surround)
        evo_params = _DanieleEvoParams(peak_nits=peak_nits, g=g)
        rgb = _torch_daniele_evo_luma_preserving(rgb, evo_params)

        peak_scale = peak_nits / 100.0
        rgb = rgb / peak_scale

        is_p3 = "P3" in output_transform
        is_rec2020 = "2100" in output_transform or "Rec.2020" in output_transform
        if is_rec2020:
            rgb = _torch_apply_matrix(rgb, AP1_TO_Rec2020)
        elif is_p3:
            matrix = AP1_TO_P3D60 if "Cinema" in output_transform and "D60" in output_transform else AP1_TO_P3D65
            rgb = _torch_apply_matrix(rgb, matrix)
        else:
            rgb = _torch_apply_matrix(rgb, AP1_TO_sRGB)

        rgb = rgb.clamp(min=0.0)

        is_pq = "PQ" in output_transform
        is_hlg = "HLG" in output_transform
        if is_pq:
            rgb = _torch_pq_encode(rgb * peak_scale)
        elif is_hlg:
            rgb = _torch_hlg_encode(rgb)
        elif "Cinema" in output_transform:
            rgb = rgb.clamp(0.0, 1.0).pow(1.0 / 2.6)
        else:
            rgb = _torch_srgb_encode(rgb)

        out = rgb.clamp(0.0, 1.0)

        info_parts = [
            f"ACES 2.0 Full OT",
            f"{input_colorspace} → {output_transform}",
            f"Daniele Evo (g={self._surround_g(surround):.2f})",
            f"Reach GC (str={gamut_compress_strength:.2f})",
        ]
        if is_hdr:
            info_parts.append(f"Peak {peak_nits:.0f} nits")
        info = " | ".join(info_parts)
        log.info("ACES2FullOT: %s", info)

        return out, info


# ═════════════════════════════════════════════════════════════════════════════
# NODE 4 — RadianceACESMetadataFile
# ═════════════════════════════════════════════════════════════════════════════

class RadianceACESMetadataFile:
    """
    Write or read an ACES Metadata File (AMF) XML sidecar.

    AMF captures the complete ACES colour pipeline in a standardised XML
    format (Academy TB-2014-009 / S-2019-001), enabling downstream tools
    (DaVinci Resolve, Baselight, etc.) to apply the identical colour
    transforms automatically.

    WRITE mode  — generates an AMF and optionally saves it to disk.
    READ  mode  — parses an existing AMF and outputs the key fields as STRING.
    """

    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    DESCRIPTION = "Read or write ACES metadata XML sidecar files for shot archiving."
    FUNCTION     = "run"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("amf_xml",  "amf_summary")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["write", "read"], {
                    "default": "write",
                    "tooltip": "write = create new AMF, read = parse existing.",
                }),
            },
            "optional": {
                # ── WRITE inputs ───────────────────────────────────────────
                "clip_name": ("STRING", {
                    "default": "Untitled",
                    "tooltip": "(write) Clip identifier written into the AMF.",
                }),
                "input_transform": ("STRING", {
                    "default": "urn:ampas:aces:transformId:v1.5:IDT.Academy.ADX10.a1.0.3",
                    "tooltip": "(write) ACES URN for the input transform.",
                }),
                "output_transform": ("STRING", {
                    "default": "urn:ampas:aces:transformId:v2.0:ODT.Academy.Rec709-100nits.a2.0.0",
                    "tooltip": "(write) ACES URN for the output transform.",
                }),
                "peak_nits": ("FLOAT", {
                    "default": 100.0, "min": 48.0, "max": 10000.0, "step": 1.0,
                    "tooltip": "(write) Display peak luminance in nits, stored as the output transform's peakLuminance.",
                }),
                "min_nits": ("FLOAT", {
                    "default": 0.005, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "(write) Display black level in nits, stored as the output transform's minLuminance.",
                }),
                "description": ("STRING", {
                    "default": "Created by Radiance ACES 2.0 pipeline.",
                    "multiline": True,
                    "tooltip": "(write) Free text stored in the AMF's amfInfo description (XML-escaped, so any characters are safe).",
                }),
                "save_path": ("STRING", {
                    "default": "",
                    "tooltip": "(write) Full path to save .amf file. Leave blank to skip.",
                }),
                # ── READ input ─────────────────────────────────────────────
                "amf_xml_in": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "(read) Paste AMF XML here, or load via save_path.",
                }),
            },
        }

    def run(
        self,
        mode:             str,
        clip_name:        str   = "Untitled",
        input_transform:  str   = "",
        output_transform: str   = "",
        peak_nits:        float = 100.0,
        min_nits:         float = 0.005,
        description:      str   = "",
        save_path:        str   = "",
        amf_xml_in:       str   = "",
    ):
        save_path = strip_path_quotes(save_path)
        if mode == "write":
            xml = _build_amf(
                clip_name        = clip_name,
                input_transform  = input_transform,
                output_transform = output_transform,
                peak_nits        = peak_nits,
                min_nits         = min_nits,
                description      = description,
            )

            if save_path:
                path = save_path
                if not path.lower().endswith(".amf"):
                    path += ".amf"
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(xml)
                log.info("AMF written to: %s", path)

            summary = (
                f"AMF WRITE | clip: {clip_name} | "
                f"in: {input_transform[:60]}… | "
                f"out: {output_transform[:60]}… | "
                f"peak: {peak_nits} nits"
            )
            return (xml, summary)

        else:  # read
            src = amf_xml_in.strip()
            if not src and save_path:
                with open(save_path, encoding="utf-8") as fh:
                    src = fh.read()

            if not src:
                return ("", "AMF READ: no XML provided")

            try:
                fields = _parse_amf(src)
            except Exception as exc:
                return (src, f"AMF READ ERROR: {exc}")

            summary = " | ".join(f"{k}: {v}" for k, v in fields.items())
            log.info("AMF READ: %s", summary)
            return (src, summary)


# ═════════════════════════════════════════════════════════════════════════════
# NODE 5 — RadianceACES2Compliance
# ═════════════════════════════════════════════════════════════════════════════

class RadianceACES2Compliance:
    """
    ACES 2.0 Output Check.

    Compares a rendered display image against the Academy reference Output
    Transform (OpenColorIO, pinned ACES 2.0 studio config) for the same
    ACEScg scene image, on every frame, and reports finite values, display
    range and the deviation from the reference.

    FIX-003: this node used to be called an "S-2126 Compliance" diagnostic.
    It never implemented S-2126; its checks were a hard-coded 10%-of-peak
    grey on encoded values plus range heuristics. It is now a scoped
    comparison against the reference, and says when it cannot make one.
    The node ID is unchanged so saved workflows still load.
    """

    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    DESCRIPTION = ("Compare a display render against the ACES 2.0 reference Output Transform (OpenColorIO) "
                   "for the same ACEScg image. Not a certification against any Academy specification.")
    FUNCTION     = "check"
    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("report", "pass_count")

    OUTPUT_TYPES = {
        "SDR_sRGB": "ACES 2.0 SDR (sRGB/Rec.709)",
        "SDR_P3": "ACES 2.0 SDR (P3-D65)",
        "HDR_PQ_1000": "ACES 2.0 HDR (Rec.2100 PQ 1000 nits)",
        "HDR_PQ_2000": "ACES 2.0 HDR (Rec.2100 PQ 2000 nits)",
        "HDR_PQ_4000": "ACES 2.0 HDR (Rec.2100 PQ 4000 nits)",
        "HDR_HLG": "ACES 2.0 HDR (Rec.2100 HLG)",
    }
    # Display code values. 1/1023 is one 10-bit code value.
    MAX_TOLERANCE = 2.0 / 1023.0
    MEAN_TOLERANCE = 0.5 / 1023.0

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scene_image":   ("IMAGE", {
                    "tooltip": "Scene-linear ACEScg image BEFORE the output transform.",
                }),
                "display_image": ("IMAGE", {
                    "tooltip": "Display-encoded image AFTER the output transform under test.",
                }),
                "output_type": (
                    list(cls.OUTPUT_TYPES),
                    {"default": "SDR_sRGB", "tooltip": "ACES 2.0 Output Transform the display image is meant to be. "
                                                      "The reference is rendered for this output and compared."},
                ),
            },
            "optional": {
                "peak_nits": ("FLOAT", {
                    "default": 100.0, "min": 48.0, "max": 10000.0, "step": 1.0,
                    "tooltip": "Shown in the report header only; the peak comes from output_type.",
                }),
            },
        }

    def check(
        self,
        scene_image:   torch.Tensor,
        display_image: torch.Tensor,
        output_type:   str,
        peak_nits:     float = 100.0,
    ):
        from radiance.hdr import aces2_ocio

        s_all = scene_image.detach().cpu().float().numpy()
        d_all = display_image.detach().cpu().float().numpy()
        if s_all.ndim == 3:
            s_all = s_all[None]
        if d_all.ndim == 3:
            d_all = d_all[None]
        s_all = s_all[..., :3]
        d_all = d_all[..., :3]
        results: dict[str, str] = {}

        if s_all.shape != d_all.shape:
            results["shape_match"] = f"FAIL: scene {s_all.shape} vs display {d_all.shape}"
        else:
            results["shape_match"] = f"PASS ({s_all.shape[0]} frame(s))"

        bad_s = int((~np.isfinite(s_all)).sum())
        bad_d = int((~np.isfinite(d_all)).sum())
        results["finite_values"] = ("PASS" if bad_s == 0 and bad_d == 0
                                    else f"FAIL: {bad_s} NaN/Inf in scene, {bad_d} in display")

        finite_d = d_all[np.isfinite(d_all)]
        if finite_d.size and (finite_d.min() < -1e-4 or finite_d.max() > 1.0 + 1e-4):
            results["display_range"] = f"FAIL: values span [{finite_d.min():.4f}, {finite_d.max():.4f}], outside [0, 1]"
        else:
            results["display_range"] = "PASS"

        target = self.OUTPUT_TYPES.get(output_type)
        ok, reason = aces2_ocio.reference_available(target)
        if not ok:
            results["reference_match"] = f"SKIP (no reference: {reason})"
        elif s_all.shape != d_all.shape or bad_s or bad_d:
            results["reference_match"] = "SKIP (inputs failed the checks above)"
        else:
            ref = np.stack([aces2_ocio.apply_reference(f, "ACEScg", target) for f in s_all])
            ref = np.clip(ref, 0.0, 1.0)
            err = np.abs(ref - d_all)
            max_err, mean_err = float(err.max()), float(err.mean())
            worst = int(np.argmax(err.reshape(err.shape[0], -1).max(axis=1)))
            verdict = "PASS" if (max_err <= self.MAX_TOLERANCE and mean_err <= self.MEAN_TOLERANCE) else "FAIL"
            results["reference_match"] = (
                f"{verdict}: max |Δ| {max_err * 1023:.2f}, mean {mean_err * 1023:.3f} "
                f"(10-bit codes; limits {self.MAX_TOLERANCE * 1023:.1f} / {self.MEAN_TOLERANCE * 1023:.1f}); "
                f"worst frame {worst}"
            )

        pass_count  = sum(1 for v in results.values() if v.startswith("PASS"))
        skip_count  = sum(1 for v in results.values() if v.startswith("SKIP"))
        fail_count  = sum(1 for v in results.values() if v.startswith("FAIL"))

        lines = [
            "═══════════════════════════════════════════════",
            "  Radiance ACES 2.0 Output Check",
            f"  Output: {output_type}  Peak: {peak_nits:g} nits",
            f"  Reference: {aces2_ocio.PINNED_CONFIG}",
            "═══════════════════════════════════════════════",
        ]
        for name, verdict in results.items():
            icon = "✓" if verdict.startswith("PASS") else ("⚠" if verdict.startswith("SKIP") else "✗")
            lines.append(f"  {icon}  {name:20s}  {verdict}")
        lines += [
            "───────────────────────────────────────────────",
            f"  PASS {pass_count}  SKIP {skip_count}  FAIL {fail_count}",
            "  Not a certification against an Academy specification.",
            "═══════════════════════════════════════════════",
        ]

        report = "\n".join(lines)
        log.info("ACES2OutputCheck: %d pass, %d skip, %d fail (%s)", pass_count, skip_count, fail_count, output_type)
        return (report, pass_count)


# ─────────────────────────────────────────────────────────────────────────────
# Registration
# ─────────────────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "RadianceACES2Tonescale":          RadianceACES2Tonescale,
    "RadianceACES2ReachGamutCompress": RadianceACES2ReachGamutCompress,
    "RadianceACES2OutputTransformFull": RadianceACES2OutputTransformFull,
    "RadianceACESMetadataFile":        RadianceACESMetadataFile,
    "RadianceACES2Compliance":         RadianceACES2Compliance,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceACES2Tonescale":          "◎ Radiance ACES 2.0 Tonescale",
    "RadianceACES2ReachGamutCompress": "◎ Radiance ACES 2.0 Reach Gamut Compress",
    "RadianceACES2OutputTransformFull": "◎ Radiance ACES 2.0 Output Transform",
    "RadianceACESMetadataFile":        "◎ Radiance ACES Metadata File (AMF)",
    "RadianceACES2Compliance":         "◎ Radiance ACES 2.0 Output Check",
}
