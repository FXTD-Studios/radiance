"""
Radiance VAE 4K — Production HDR Encode / Decode for ComfyUI
═══════════════════════════════════════════════════════════════

ARCHITECTURE OVERVIEW FOR DEBUGGING
────────────────────────────────────
This module implements the full VAE encode→decode pipeline for the Radiance
suite. It handles images up to 8K+ using production-grade cosine-blend tiling,
with full HDR color-science pipelines for 12 color spaces.

CRITICAL SIGNAL-FLOW (understand this before touching ANY code):

  ENCODE: pixels (scene-linear) → Linearize → Exposure → HDR Compress → VAE
  DECODE: VAE → HDR Decompress → Linearize → Inverse Tonemap → Exposure → Target Space

HDR MODES — each has a DIFFERENT valid domain in VAE-encoded space:
  ┌──────────────────┬────────────────┬────────────────────────────────────┐
  │ Mode             │ Encoded Domain │ Notes                              │
  ├──────────────────┼────────────────┼────────────────────────────────────┤
  │ Clip (SDR)       │ [0.0, 1.0]     │ sRGB gamma, hard-clipped           │
  │ Soft Clip        │ [0.0, 1.0]     │ sRGB gamma + tanh rolloff ≥0.85   │
  │ Compress (Log)   │ [~0.0, ~1.04–1.08] │ Log curve — per-profile adaptive  │
  │                  │                    │ ★ v2.3: soft-shouldered, NOT hard │
  │                  │                    │   clamped — see LOG_PROFILE_HDR_  │
  │                  │                    │   PARAMS + _soft_log_shoulder     │
  │ Passthrough      │ [−0.05, 1.5]   │ sRGB gamma, widened for headroom   │
  └──────────────────┴────────────────┴────────────────────────────────────┘

COMPRESS (LOG) — WHY IT NEEDS SPECIAL CARE:
  Log curves are extremely steep near code 1.0. For ARRI LogC4:
    code 0.98 → ~26.0 linear      code 1.00 → ~38.0 linear
    code 1.01 → ~43.0 linear      code 1.02 → ~49.0 linear
  This means VAE reconstruction noise of ±0.01 in coded space becomes ±5–10
  linear units after decompression. The v2.3 pipeline addresses this with:
    1. Soft-shoulder (not hard clamp) at the log curve ceiling — preserves
       legitimate super-whites without hard-cutting at exactly 1.0
    2. Selective highlight denoise in log-coded space BEFORE decompression —
       smooths out VAE noise where it would be exponentially amplified,
       while leaving midtones/shadows untouched
    3. No final linear-domain clamp — full HDR passthrough to output tensor

  ★ PER-PROFILE ADAPTIVE TUNING (v2.3):
    Each of the 6 supported log curves has a different steepness (derivative)
    at code 1.0. Steeper curves amplify VAE noise more aggressively, so they
    get more conservative shoulder/denoise parameters:
      - LogC4, S-Log3 (moderate):      knee=0.96, ceiling=1.08, denoise from 0.80
      - LogC3, V-Log (steep):          knee=0.95, ceiling=1.06–1.07, denoise from 0.78
      - DaVinci Intermediate (v.steep): knee=0.93, ceiling=1.05, denoise from 0.75
      - RED Log3G10 (extreme):         knee=0.92, ceiling=1.04, denoise from 0.72
    See LOG_PROFILE_HDR_PARAMS for full table and rationale.

NOISE MODEL:
  VAE reconstruction noise is roughly uniform in whatever space the VAE sees.
  After log decompression, this uniform noise becomes exponentially larger in
  highlights (because the log→linear curve is steeper there). The correct
  fix is to denoise IN LOG SPACE (where noise is uniform and small) BEFORE
  converting to linear — not to clamp in linear space (which destroys signal).

VERSIONING:
  v2.0 — Initial 4K tiling, multi-format, video support
  v2.1 — BUG-42..46 fixes, frame-loop video decode
  v2.2 — BUG-47..56 fixes, HDR-FIX controls, extended sRGB EOTF
  v2.3 — True HDR Compress(Log): soft-shoulder, highlight denoise,
         clamp-free linear output for all log profiles
  v2.3.5 — BUG-OVEREXPOSE: Fixed overexposed output when target_space="Linear"
           Root cause: missing linear→sRGB re-encode in the SDR display path.
           target_space="Linear" + hdr_output=False now re-encodes to sRGB
           for ComfyUI IMAGE compatibility (same as target_space="sRGB").
           Set hdr_output=True for true scene-linear VFX output.
           Default target_space changed from "Linear" to "sRGB".
           Final clamp extended to cover "Linear"+"Raw" in SDR mode (was sRGB-only).
  v2.3.8 — display_tonemap decoupled from hdr_output.
           display_tonemap is now the SOLE control for tonemapping — fires even
           when hdr_output=True. hdr_output only controls final clamping and
           sRGB re-encode. display_tonemap="None" = guaranteed raw HDR passthrough
           (always overexposed in ComfyUI, intended for OCIO-aware viewers only).
  v2.3.9 — BUG-OVEREXPOSE-INCOMPLETE: Fixed overexposed output for
           target_space="ACEScg" / "ACES 2065-1" / "Rec.2020 Linear" + hdr_output=False.
           v2.3.5 extended the final [0,1] clamp only to target_space="Linear";
           the three scene-referred linear spaces were missed — they passed float
           values > 1.0 to ComfyUI's IMAGE socket, causing blown-out output.
           Fix: clamp ALL target spaces when hdr_output=False (not just display-referred).
           hdr_output=True is unchanged — values > 1.0 preserved for VFX pipelines.
           Added WARNING log when scene-referred target_space is used with hdr_output=False.
  v2.4   — decode_noise_scale: per-profile lerp noise injection into the latent
           before VAE decode. Breaks up coherent decoder grid artifacts in flat
           highlight regions that would be exponentially amplified by log→linear
           decompression. Default 0.0 (disabled). Per-profile recommended values
           in DECODE_NOISE_SCALE_PER_PROFILE (LogC3=0.025 matches LTX default).
           Inspired by LTX LumiVid (arxiv 2604.11788).
"""

import torch
import torch.nn.functional as F
import json
import math
import logging
import struct
import zlib
import os
import uuid
from typing import Tuple, Dict, Any, Optional

import comfy.model_management
import comfy.utils
import folder_paths

logger = logging.getLogger("radiance")

# Local imports
from .utils import tensor_srgb_to_linear, tensor_linear_to_srgb

# Log curve imports from canonical color package
_HAS_LOG_CURVES = True
_HAS_LOGC3 = True

from radiance.color.transfer import (
    tensor_logc4_to_linear,
    tensor_linear_to_logc4,
    tensor_slog3_to_linear,
    tensor_linear_to_slog3,
    tensor_vlog_to_linear,
    tensor_linear_to_vlog,
    tensor_davinci_intermediate_to_linear,
    tensor_linear_to_davinci_intermediate,
    tensor_log3g10_to_linear,
    tensor_linear_to_log3g10,
    tensor_logc3_to_linear,
    tensor_linear_to_logc3,
)


# ═══════════════════════════════════════════════════════════════════════════════
#                    v2.3 CONSTANTS & HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

# Feature 2: Model-architecture → latent downscale factor
VAE_FACTOR_DEFAULT = 8
VAE_FACTOR_MAP: Dict[str, int] = {
    # Video models that may use non-8 factors
    "StableCascadeStageC": 42,
    "StableCascadeStageB": 4,
    "Wan": 8,
    "CogVideo": 8,
    "Cosmos": 8,
    "HunyuanVideo": 8,
    # Most modern image models: 8
}

# Feature 1: channel count → format label
LATENT_FORMAT_MAP: Dict[int, str] = {
    4:   "sd_4ch",        # SD1.x, SD2.x, SDXL
    8:   "sd3_8ch",       # SD3 medium (8-ch)
    12:  "mochi_12ch",    # Mochi (Genmo) causal video VAE
    16:  "flux_16ch",     # Flux, SD3 large, WAN, Chroma, HunyuanVideo
    32:  "cascade_32ch",  # Stable Cascade
    128: "ltx_128ch",     # LTX-Video (all versions), Flux.2 Klein
}

# Feature 4: latent distribution sampling modes
LATENT_SAMPLING_MODES = ["sample", "mean", "mode"]

# Feature 7: Extended color space lists
EXTENDED_SOURCE_SPACES = [
    "Linear",
    "ACEScg",
    "ACES 2065-1",
    "Rec.2020 Linear",
    "sRGB",
    "Raw",
    "ARRI LogC3",   # Alexa Classic / Mini / SXT / LF (LogC3 mode)
    "ARRI LogC4",
    "Sony S-Log3",
    "Panasonic V-Log",
    "DaVinci Intermediate",
    "RED Log3G10",
]

EXTENDED_TARGET_SPACES = [
    "Linear",
    "ACEScg",
    "ACES 2065-1",
    "Rec.2020 Linear",
    "sRGB",
    "Raw",
    "ARRI LogC3",   # Alexa Classic / Mini / SXT / LF (LogC3 mode)
    "ARRI LogC4",
    "Sony S-Log3",
    "Panasonic V-Log",
    "DaVinci Intermediate",
    "RED Log3G10",
]

EXTENDED_LOG_SPACES = [
    "ARRI LogC3",   # Alexa Classic / Mini / SXT / LF (LogC3 mode)
    "ARRI LogC4",
    "Sony S-Log3",
    "Panasonic V-Log",
    "DaVinci Intermediate",
    "RED Log3G10",
]

# ─────────────────────────────────────────────────────────────────────────────
# v2.3: Per-profile adaptive HDR parameters for Compress (Log) decode
# ─────────────────────────────────────────────────────────────────────────────
# Each log curve has a different steepness near code 1.0. Steeper curves
# amplify VAE noise more aggressively after decompression, so they need:
#   - Lower soft-shoulder knee (start compressing earlier)
#   - Tighter ceiling (bound max decompressed linear)
#   - Lower denoise threshold (start smoothing earlier)
#   - Higher denoise strength (suppress more aggressively)
#
# The parameters below were tuned per-curve based on the derivative (slope)
# of the log→linear function at code 1.0:
#
#   ┌─────────────────────┬──────────────┬────────────────┬──────────────────┐
#   │ Profile             │ code 1.0 →   │ slope at 1.0   │ Category         │
#   │                     │ linear       │ (dLin/dCode)   │                  │
#   ├─────────────────────┼──────────────┼────────────────┼──────────────────┤
#   │ ARRI LogC3 (EI800)  │ ~55          │ ~350           │ steep            │
#   │ ARRI LogC4          │ ~38          │ ~250           │ moderate-steep   │
#   │ Sony S-Log3         │ ~38          │ ~240           │ moderate-steep   │
#   │ Panasonic V-Log     │ ~46          │ ~280           │ steep            │
#   │ DaVinci Intermediate│ ~100+        │ ~600+          │ very steep       │
#   │ RED Log3G10         │ ~184         │ ~1100+         │ extremely steep  │
#   └─────────────────────┴──────────────┴────────────────┴──────────────────┘
#
# Format: {space_name: (shoulder_knee, shoulder_ceiling, denoise_threshold, denoise_strength)}

LOG_PROFILE_HDR_PARAMS: Dict[str, Tuple[float, float, float, float]] = {
    # ── Moderate-steep: standard shoulder, standard denoise ──
    "ARRI LogC4":          (0.96, 1.08, 0.80, 0.55),
    "Sony S-Log3":         (0.96, 1.08, 0.80, 0.55),

    # ── Steep: slightly earlier knee, tighter ceiling ──
    "ARRI LogC3":          (0.95, 1.06, 0.78, 0.60),
    "Panasonic V-Log":     (0.95, 1.07, 0.78, 0.60),

    # ── Very steep: conservative shoulder, aggressive denoise ──
    "DaVinci Intermediate": (0.93, 1.05, 0.75, 0.70),

    # ── Extremely steep: most conservative — slope >1100 at code 1.0 ──
    "RED Log3G10":          (0.92, 1.04, 0.72, 0.75),
}

# Default for unknown log profiles (same as LogC4 — safe middle ground)
LOG_PROFILE_HDR_DEFAULT = (0.96, 1.08, 0.80, 0.55)


def _intermediate_device() -> torch.device:
    """Where clip-length accumulators live: ComfyUI's intermediate device.

    STREAM-FIX: the encode and decode frame loops used to accumulate the whole
    clip in VRAM, which made duration a VRAM budget rather than a disk one.
    comfy.sd.VAE already returns encode/decode results on
    ``model_management.intermediate_device()`` (CPU unless the user launched
    with --gpu-only), so accumulating there matches ComfyUI's own contract
    instead of inventing a second one. Falls back to CPU for VAE-like objects
    and test stubs that do not expose the helper.
    """
    fn = getattr(comfy.model_management, "intermediate_device", None)
    if callable(fn):
        try:
            dev = fn()
            if isinstance(dev, torch.device):
                return dev
        except Exception:  # nosec B110 - third-party model_management shims
            pass
    return torch.device("cpu")


def _restore_alpha_channel(img: torch.Tensor, alpha: Optional[torch.Tensor]) -> torch.Tensor:
    if alpha is None:
        return img
    alpha_f = alpha.float()
    if alpha_f.dim() == 3:
        alpha_f = alpha_f.unsqueeze(0)
    if alpha_f.dim() != 4 or alpha_f.shape[-1] < 1:
        raise ValueError("Alpha must have shape (B,H,W) or (B,H,W,C).")
    alpha_ch = alpha_f[..., :1]
    if alpha_ch.shape[1:3] != img.shape[1:3]:
        alpha_ch = F.interpolate(
            alpha_ch.permute(0, 3, 1, 2),
            size=(img.shape[1], img.shape[2]),
            mode="bilinear", align_corners=False,
        ).permute(0, 2, 3, 1)
    if alpha_ch.shape[0] != img.shape[0]:
        if alpha_ch.shape[0] == 1:
            alpha_ch = alpha_ch.expand(img.shape[0], -1, -1, -1)
        else:
            raise ValueError(
                f"Alpha batch must be 1 or match decoded frames ({img.shape[0]}), got {alpha_ch.shape[0]}."
            )
    if alpha_ch.device != img.device:
        alpha_ch = alpha_ch.to(img.device)
    return torch.cat([img[..., :3], alpha_ch], dim=-1)

# ─────────────────────────────────────────────────────────────────────────────
# v2.4: Per-profile decode_noise_scale defaults
# ─────────────────────────────────────────────────────────────────────────────
# Based on LTX LumiVid (arxiv 2604.11788): injecting a small amount of noise
# into the latent BEFORE VAE decode breaks up coherent grid artifacts that the
# VAE's convolutional decoder can introduce in flat highlight regions.  The
# effect is most important for Compress (Log) mode because the subsequent
# log→linear decompression amplifies any residual artifact exponentially in
# the highlight region (slope > 200 at code 1.0 for LogC3/V-Log).
#
# Tuning rationale: steeper log curves amplify latent artifacts more after
# decompression, so they benefit from slightly more noise at the latent level:
#
#   ┌─────────────────────┬─────────────────────────────────────────────────┐
#   │ Profile             │ Recommended scale │ Rationale                   │
#   ├─────────────────────┼───────────────────┼─────────────────────────────┤
#   │ ARRI LogC4          │ 0.018             │ moderate slope ~250         │
#   │ Sony S-Log3         │ 0.018             │ moderate slope ~240         │
#   │ ARRI LogC3          │ 0.025             │ steep ~350; matches LTX     │
#   │ Panasonic V-Log     │ 0.022             │ steep ~280                  │
#   │ DaVinci Intermediate│ 0.030             │ very steep ~600+            │
#   │ RED Log3G10         │ 0.035             │ extreme slope ~1100+        │
#   └─────────────────────┴───────────────────┴─────────────────────────────┘
#
# Formula (lerp, same as LTX):
#   noised_latent = (1 - scale) * latent + scale * randn_like(latent)
#
# At scale=0.025 the latent moves only 2.5% toward pure noise — perceptually
# transparent but enough to de-correlate VAE decoder grid patterns.
# scale=0.0 disables the feature entirely (safe default for existing workflows).

DECODE_NOISE_SCALE_PER_PROFILE: Dict[str, float] = {
    "ARRI LogC4":            0.018,
    "Sony S-Log3":           0.018,
    "ARRI LogC3":            0.025,   # matches LTX default for LogC3
    "Panasonic V-Log":       0.022,
    "DaVinci Intermediate":  0.030,
    "RED Log3G10":           0.035,
}

# ─────────────────────────────────────────────────────────────────────────────
# V-1 FIX: Precomputed color space matrices (module-level constants)
# Previously allocated as new torch.tensor() on EVERY call, causing 6+
# allocations per frame for video. Now created once, moved to device at use.
# ─────────────────────────────────────────────────────────────────────────────
# Encode: working space → Rec.709 linear
_AP1_TO_REC709 = torch.tensor([
    [1.7050509, -0.6217921, -0.0832588],
    [-0.1302564, 1.1408047, -0.0105483],
    [-0.0240033, -0.1289690, 1.1529723],
], dtype=torch.float32).T

# 3.5.0: the two AP0 matrices were not inverses of each other and the
# Rec.709 -> AP0 one was off by 1.2 %. Both now come from color/ops, which
# matches OpenColorIO's ACES studio config to 1e-7.
from radiance.color.ops import (  # noqa: E402
    M_ACES2065_1_TO_REC709 as _M_AP0_709,
    M_REC709_TO_ACES2065_1 as _M_709_AP0,
)
_AP0_TO_REC709 = _M_AP0_709.clone().T

_REC2020_TO_REC709 = torch.tensor([
    [ 1.6604910, -0.5876411, -0.0728499],
    [-0.1245505,  1.1328999, -0.0083494],
    [-0.0181508, -0.1005789,  1.1187297],
], dtype=torch.float32).T

# Decode: Rec.709 linear → working space
_REC709_TO_AP1 = torch.tensor([
    [0.613097, 0.339523, 0.047379],
    [0.070194, 0.916354, 0.013452],
    [0.020616, 0.109570, 0.869815],
], dtype=torch.float32).T

_REC709_TO_AP0 = _M_709_AP0.clone().T

_REC709_TO_REC2020 = torch.tensor([
    [0.6274039, 0.3292830, 0.0433131],
    [0.0690973, 0.9195404, 0.0113623],
    [0.0163914, 0.0880132, 0.8955954],
], dtype=torch.float32).T


from radiance.hdr.decode_meta import (  # noqa: E402  (3.5.0 contract helpers)
    LOG_SPACE_GAMUT,
    TARGET_SPACE_GAMUT,
    _encode_working_gamut,
    _gamut_mat_t,
    latent_fingerprint,
    radiance_meta_is_live,
    strip_hdr_meta,
    verify_radiance_meta,
)


# ─────────────────────────────────────────────────────────────────────────────
# v2.3 Helpers: HDR-safe color transforms + log-domain denoise
# ─────────────────────────────────────────────────────────────────────────────

def _safe_matrix_transform(img: torch.Tensor, mat: torch.Tensor) -> torch.Tensor:
    """Apply a 3×3 color matrix to an image tensor, safe for 4+ channels.

    v2.2 FIX (BUG-48): Previous code did ``img.reshape(-1, 3) @ mat`` which
    crashes with RuntimeError if the image has alpha or other extra channels.
    This helper slices to the first 3 channels, transforms, then recombines.

    Args:
        img: (B, H, W, C) where C ≥ 3.  C > 3 channels pass through untouched.
        mat: (3, 3) color matrix, already transposed for right-multiply.
    """
    if img.shape[-1] == 3:
        shape = img.shape
        return (img.reshape(-1, 3) @ mat).reshape(shape)
    # 4+ channels: split, transform RGB, recombine
    rgb = img[..., :3]
    extra = img[..., 3:]
    shape = rgb.shape
    rgb = (rgb.reshape(-1, 3) @ mat).reshape(shape)
    return torch.cat([rgb, extra], dim=-1)


def _safe_srgb_to_linear_extended(img: torch.Tensor) -> torch.Tensor:
    """sRGB EOTF with correct handling for values outside [0, 1].

    v2.2 FIX (BUG-51): Standard sRGB EOTF is only defined for [0, 1].
    For HDR decode (Soft Clip, Passthrough), recovered values can exceed 1.0.
    This helper:
      • Values in [0, 1]: standard IEC 61966-2-1 EOTF
      • Values > 1.0: continuous power-function extension (no clamp)
      • Values < 0.0: mirrored for symmetry (preserves negatives)

    If the imported ``tensor_srgb_to_linear`` already handles extended range
    correctly (i.e. does NOT clamp to [0, 1]), this wrapper adds negligible
    overhead since the torch.where branches are fused.
    """
    # Decompose: sign-preserving absolute value for the power branch
    abs_img = img.abs()
    sign = img.sign()

    # Standard sRGB EOTF extended to full range
    low_mask = abs_img <= 0.04045
    low = abs_img / 12.92
    high = ((abs_img + 0.055) / 1.055).pow(2.4)
    result = torch.where(low_mask, low, high) * sign

    return result


def _soft_log_shoulder(img: torch.Tensor, knee: float = 0.96, ceiling: float = 1.08) -> torch.Tensor:
    """Soft-shoulder compressor for log-coded VAE output — replaces hard clamp.

    v2.3 FIX (HDR-CLAMP-FREE): Previous versions hard-clamped Compress (Log)
    decode output to [0.0, 1.0]. This destroyed legitimate super-white signal:
    any VAE reconstruction at 1.001 was clipped to 1.0, losing highlight detail.

    This function replaces the hard clamp with a production-grade soft shoulder:
      • Values in [0, knee]   → pass through unchanged (midtones, shadows)
      • Values in [knee, ∞)   → smoothly compressed toward `ceiling` via tanh
      • Values below 0        → clamped at 0 (no valid sub-zero log signal)

    The soft shoulder preserves legitimate super-whites (VAE can reconstruct
    values slightly above 1.0 for very bright highlights) while preventing
    runaway noise from pushing values arbitrarily high.

    Design rationale for defaults:
      knee=0.96:  LogC4 code 0.96 ≈ 22.5 linear (18+ stops above mid-grey).
                  Everything below this is rock-solid signal — never touched.
      ceiling=1.08: LogC4 code 1.08 ≈ 55 linear (21.5+ stops).
                    This allows ~3 extra stops of HDR headroom beyond knee,
                    while still bounding the max log value to prevent the
                    decompression from producing extreme linear outliers.

    Compared to hard clamp at 1.0 (LogC4: max 38 linear):
      Soft shoulder at ceiling=1.08 allows up to ~55 linear — 45% more headroom.

    Args:
        img:     (B, H, W, C) tensor in log-coded domain
        knee:    Below this, values pass through unchanged
        ceiling: Asymptotic maximum that the shoulder approaches
    Returns:
        Soft-shouldered tensor, values in [0, ceiling)
    """
    # Floor: no valid signal below 0 in log-coded space. Log curves encode
    # scene-linear ~−0.015 to code ~0.09 (LogC4), so 0.0 in code is well
    # below anything physical. Clamp the floor to prevent NaN in atanh paths.
    img = torch.clamp(img, min=0.0)

    above = img > knee
    if above.any():
        result = img.clone()
        rng = ceiling - knee
        # v2.3.1 FIX (V-B1): guard against caller-supplied tuples where
        # ceiling <= knee (misconfigured profile). Without the +1e-8 the
        # division becomes NaN/Inf and contaminates the whole tensor.
        excess = (img[above] - knee) / (rng + 1e-8)
        # tanh: maps [0, ∞) → [0, 1), so result → [knee, ceiling)
        result[above] = knee + rng * torch.tanh(excess)
        return result
    return img


def _denoise_log_highlights(
    img: torch.Tensor,
    threshold: float = 0.80,
    strength: float = 0.6,
) -> torch.Tensor:
    """Selective highlight denoise in log-coded space BEFORE decompression.

    v2.3 HDR-CLEAN: This is the key to clean HDR output without clamping.

    PROBLEM:
      VAE reconstruction noise is roughly uniform across the coded domain
      (±0.005–0.02 in log-coded space). After log→linear decompression:
        • Noise at code 0.5 (midtones): ±0.01 coded → ±0.15 linear (invisible)
        • Noise at code 0.9 (highlights): ±0.01 coded → ±2.5 linear (visible)
        • Noise at code 0.98 (super-whites): ±0.01 coded → ±5.0 linear (severe)

      The noise amplification is exponential because the log→linear curve
      is steeper at higher code values. Hard-clamping at 1.0 "fixes" this
      by destroying all HDR content above code 1.0. That's the wrong trade-off.

    SOLUTION:
      Denoise IN LOG SPACE (where noise is uniform and small) BEFORE converting
      to linear. Apply a spatially-adaptive 3×3 box smooth weighted by how
      close each pixel is to the highlight region. Midtones pass through
      untouched; only the noisy highlight region gets smoothed.

    This is the same principle used by DaVinci Resolve's "Highlight Recovery"
    and Nuke's "SoftClip" node — denoise before the steep part of the curve,
    not after.

    Args:
        img:        (B, H, W, C) tensor in log-coded space
        threshold:  Code value above which denoising ramps in (0.80 default =
                    ~10 linear for LogC4 — well into highlights but not midtones)
        strength:   Blend factor at full effect (0.0 = no denoise, 1.0 = full smooth).
                    0.6 default preserves texture while suppressing noise.
    Returns:
        Denoised tensor, same shape and range as input
    """
    if img.shape[1] < 3 or img.shape[2] < 3:
        # Image too small for 3×3 kernel — skip
        return img

    # Compute per-pixel blend weight: 0 below threshold, ramps to `strength`
    # above. Smooth ramp avoids a visible transition edge.
    #
    # blend_alpha is applied per-pixel: output = lerp(original, smoothed, blend_alpha)
    # Using a quadratic ramp for natural falloff:
    #   weight = strength × clamp((code − threshold) / (1.0 − threshold), 0, 1)²
    ramp = torch.clamp((img - threshold) / (1.0 - threshold + 1e-8), 0.0, 1.0)
    blend_alpha = strength * ramp * ramp  # Quadratic ramp for natural falloff

    # 3×3 box blur via F.avg_pool2d (fastest available, no dependencies)
    # Permute BHWC → BCHW for F.avg_pool2d, then back
    x = img.permute(0, 3, 1, 2)  # (B, C, H, W)
    # Reflect-pad to avoid border artifacts (1px each side for 3×3 kernel)
    x_padded = F.pad(x, (1, 1, 1, 1), mode="reflect")
    smoothed = F.avg_pool2d(x_padded, kernel_size=3, stride=1, padding=0)
    smoothed = smoothed.permute(0, 2, 3, 1)  # Back to BHWC

    # Spatially-adaptive blend: midtones untouched, highlights smoothed
    result = img + blend_alpha * (smoothed - img)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Feature 2: Dynamic VAE scale factor detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_vae_factor(vae: Any) -> int:
    """
    v2.0: Read the spatial downscale factor from the VAE object.
    Falls back to 8 for unknown model classes.

    ComfyUI exposes this as:
      vae.spacial_compression_decode()    (comfy.sd.VAE's own helper -- tried first)
      vae.downscale_ratio          (int, most models)
      vae.latent_format.downscale_factor  (some wrappers)
    """
    # ALBABIT-FIX: downscale_ratio is a plain int for most models, but a
    # (temporal_formula, h, w) tuple for LTX -- the int-only checks below
    # silently fell through to VAE_FACTOR_DEFAULT (8) instead of LTX's real
    # 32. spacial_compression_decode() already unwraps both forms; for a
    # plain-int ratio it returns the same value the checks below would.
    compression_decode = getattr(vae, "spacial_compression_decode", None)
    if callable(compression_decode):
        try:
            val = compression_decode()
        except Exception:
            val = None
        # `>= 1` rather than `> 0`: a wrapper answering with a sub-1 constant
        # would truncate to a factor of 0 and divide by it downstream. See the
        # SCALE-FACTOR FIX note on latent_format below.
        if isinstance(val, (int, float)) and int(val) >= 1:
            return int(val)

    for attr in ("downscale_ratio", "latent_downscale_factor"):
        val = getattr(vae, attr, None)
        if isinstance(val, int) and val > 0:
            return val

    # Check through latent_format object
    #
    # SCALE-FACTOR FIX: `latent_format.scale_factor` used to be consulted here
    # as a spatial factor. It is not one -- it is the value-normalisation
    # constant applied to latents (0.18215 for SD, 0.3611 for Flux), so
    # int(val) returned 0 and the caller went on to compute pix_h = lat_h * 0,
    # then divided by it. Unreachable through stock comfy.sd.VAE, which answers
    # spacial_compression_decode() above, but reachable through any
    # third-party VAE wrapper that exposes latent_format and nothing else.
    # Only a genuine spatial factor (>= 1) is accepted now.
    lf = getattr(vae, "latent_format", None)
    if lf is not None:
        val = getattr(lf, "downscale_factor", None)
        if isinstance(val, (int, float)) and int(val) >= 1:
            return int(val)

    # Fallback: check class name against known architectures
    cls_name = type(getattr(vae, "first_stage_model", vae)).__name__
    for key, factor in VAE_FACTOR_MAP.items():
        if key in cls_name:
            return factor

    return VAE_FACTOR_DEFAULT


# ─────────────────────────────────────────────────────────────────────────────
# Feature 1: Latent format label detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_latent_format(vae: Any) -> str:
    """
    v2.0: Return a format string e.g. 'flux_16ch' based on VAE latent channels.
    Compatible with the Radiance Sampler latent_format input socket.
    """
    # Try to get channel count from VAE
    channels = None
    model = getattr(vae, "first_stage_model", None)
    if model is not None:
        # Typical attr names across ComfyUI VAE wrappers
        for attr in ("z_channels", "latent_channels", "out_channels"):
            val = getattr(model, attr, None)
            if isinstance(val, int) and val > 0:
                channels = val
                break
        # Try encoder output shape heuristic
        if channels is None:
            enc = getattr(model, "encoder", None)
            if enc is not None:
                for attr in ("z_channels", "out_channels"):
                    val = getattr(enc, attr, None)
                    if isinstance(val, int) and val > 0:
                        channels = val
                        break

    if channels is not None:
        return LATENT_FORMAT_MAP.get(channels, f"unknown_{channels}ch")

    # Fallback: try class name
    cls = type(getattr(vae, "first_stage_model", vae)).__name__.lower()
    if "flux" in cls:
        return "flux_16ch"
    if "cascade" in cls:
        return "cascade_32ch"
    return "sd_4ch"  # Safe default


def _resolve_temporal_frames(temporal_size: Any, ts_px: Optional[int], temporal_compression: int) -> int:
    """
    Resolve the temporal_size widget value to a count of LATENT frames.

    Accepts "Auto", a numeric preset string (from the widget), or a raw int.
    Recursive chunking call sites pass 0 internally to stop further
    recursion; that is not a value the widget itself offers.

    ts_px=None disables Auto-sizing (resolves to 0 / "no chunking" instead)
    for callers that have no VRAM calibration data of their own.
    """
    if temporal_size in (0, "0", None):
        return 0
    if temporal_size == "Auto":
        if ts_px is None:
            return 0
        return TileEngine.get_optimal_temporal_size(ts_px, temporal_compression)
    return int(temporal_size)


# ─────────────────────────────────────────────────────────────────────────────
# Feature 5: Encode quality metrics
# ─────────────────────────────────────────────────────────────────────────────

def build_encode_quality_report(
    pixels_before: torch.Tensor,
    pixels_after: torch.Tensor,
    latent: torch.Tensor,
    vae_factor: int,
    latent_fmt: str,
    source_space: str,
    hdr_mode: str,
    tile_size: int,
    total_tiles: int,
    encode_time_ms: int,
) -> str:
    """Build a quality metrics JSON for the encode node output."""
    n_total = pixels_before.numel()
    n_clipped = (pixels_before > 1.0).sum().item() + (pixels_before < 0.0).sum().item()
    clip_pct = round(n_clipped / max(1, n_total) * 100, 3)
    nan_count = int(torch.isnan(latent).sum().item() + torch.isinf(latent).sum().item())

    # v2.3.1 FIX (V-B7): 3D-native VAEs (Wan / Cosmos / HunyuanVideo) return
    # 5D latents (B, C, F, H, W) even for single-image encode paths. The
    # original 4-tuple unpack crashed for those VAEs.
    if latent.ndim == 5:
        b, c_lat, _frames, lh, lw = latent.shape
    else:
        b, c_lat, lh, lw = latent.shape
    ph, pw = pixels_before.shape[1:3]

    report = {
        "version": "3.0",
        "input_resolution": f"{pw}x{ph}",
        "latent_resolution": f"{lw}x{lh}",
        "latent_format": latent_fmt,
        "latent_channels": c_lat,
        "vae_factor": vae_factor,
        "source_space": source_space,
        "hdr_mode": hdr_mode,
        "pixel_range": [round(float(pixels_after.min()), 4),
                        round(float(pixels_after.max()), 4)],
        "latent_range": [round(float(latent.min()), 4),
                         round(float(latent.max()), 4)],
        "latent_std": round(float(latent.std()), 4),
        "clipping_pct": clip_pct,
        "nan_inf_count": nan_count,
        "tile_size_px": tile_size,
        "total_tiles": total_tiles,
        "encode_time_ms": encode_time_ms,
    }
    return json.dumps(report, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Feature 4: VAE posterior sampling modes
# ─────────────────────────────────────────────────────────────────────────────

def _encode_with_sampling_mode(vae: Any, pixels: torch.Tensor, mode: str) -> torch.Tensor:
    """
    v2.2: Encode pixels with explicit distribution sampling control.

    mode = "sample"  — standard (random sample from posterior, default ComfyUI)
    mode = "mean"    — use posterior mean (deterministic, best for img2img)
    mode = "mode"    — use mean (same as mean for Gaussian posterior)

    v2.2 FIX (BUG-54): The mean/mode path previously called ``fsm.encode(pixels)``
    directly, but ``pixels`` is in ComfyUI's BHWC [0,1] format. The raw model's
    encoder expects BCHW [-1,1]. This caused mean/mode to silently produce garbage
    latents (or crash on shape mismatch), while "sample" worked correctly because
    it goes through ``vae.encode()`` which handles the conversion internally.

    Fix: preprocess pixels the same way ComfyUI does before calling the raw model.
    """
    if mode == "sample":
        result = vae.encode(pixels)
        if isinstance(result, dict):
            return result["samples"]
        return result

    # ── mean / mode path ──────────────────────────────────────────────

    # Strategy 1: Try ComfyUI's own encode_raw / encode_to_distribution
    # Some ComfyUI versions expose these directly.
    for method_name in ("encode_raw", "encode_to_distribution"):
        fn = getattr(vae, method_name, None)
        if callable(fn):
            try:
                posterior = fn(pixels)
                if hasattr(posterior, "mean"):
                    return posterior.mean
                if hasattr(posterior, "mode"):
                    return posterior.mode()
                if isinstance(posterior, torch.Tensor):
                    return posterior
            except Exception as e:
                # v2.3.1 FIX (V-B-EXC): log-but-continue. Previously this
                # swallowed every exception silently, hiding programming
                # mistakes (TypeError, AttributeError) that would never be
                # "fixed" by falling through to Strategy 2.
                logger.debug(
                    f"[Radiance VAE] {method_name} raised "
                    f"{type(e).__name__}: {e}; trying next strategy."
                )

    # Strategy 2: Access the raw first_stage_model with correct preprocessing.
    # ComfyUI's vae.encode() internally does:
    #   1. BHWC → BCHW  (movedim or permute)
    #   2. [0,1] → [-1,1] scaling
    #   3. Device placement
    #   4. fsm.encode() → posterior
    #   5. posterior.sample()
    # We replicate steps 1-4 and then extract .mean instead of .sample().
    try:
        fsm = vae.first_stage_model
        device = next(fsm.parameters()).device

        # Preprocess: BHWC [0,1] → BCHW [-1,1]
        x = pixels.movedim(-1, 1).to(device=device, dtype=torch.float32)
        x = x * 2.0 - 1.0

        posterior = fsm.encode(x)

        # Handle different return types across model architectures
        if hasattr(posterior, "latent_dist"):
            # Diffusers-style: returns an object with .latent_dist
            posterior = posterior.latent_dist

        # A plain tensor first: torch.Tensor HAS a ``mean`` attribute (the
        # reduction method), so testing ``hasattr(posterior, "mean")`` first
        # returned the bound method as the latent. ComfyUI's first_stage_model
        # returns a tensor, so with latent_sampling="mean" every encode
        # through a real ComfyUI VAE crashed downstream on
        # "'builtin_function_or_method' object has no attribute 'detach'".
        if isinstance(posterior, torch.Tensor):
            latent = posterior
        elif isinstance(getattr(posterior, "mean", None), torch.Tensor):
            latent = posterior.mean
        elif hasattr(posterior, "mode"):
            latent = posterior.mode() if callable(posterior.mode) else posterior.mode
        else:
            raise TypeError(f"Unknown posterior type: {type(posterior)}")

        # Apply scaling factor if the VAE uses one (SD models use 0.18215, etc.)
        scale_factor = getattr(vae, "scale_factor", None)
        if scale_factor is None:
            # Try to find it on the config
            config = getattr(vae, "config", None)
            if config is not None:
                scale_factor = getattr(config, "scaling_factor", None)
        if scale_factor is not None and isinstance(scale_factor, (int, float)):
            latent = latent * scale_factor

        return latent

    except Exception as e:
        logger.warning(
            f"[Radiance 4K Encode v2.3] mean/mode encode failed ({e}), "
            f"falling back to standard sample mode."
        )

    # Strategy 3: Fall back to standard sample (always works)
    result = vae.encode(pixels)
    if isinstance(result, dict):
        return result["samples"]
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#                         TILING ENGINE
# ═══════════════════════════════════════════════════════════════════════════════



class TileEngine:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    """
    Production-grade tiling engine with cosine blend weights.
    Handles arbitrary image sizes, pad-to-multiple, and VRAM-aware sizing.
    """

    @staticmethod
    def get_optimal_tile_size(
        image_h: int,
        image_w: int,
        min_tile: int = 512,
        max_tile: int = 1536,
        vram_budget_gb: float = None,
    ) -> int:
        """
        Determine optimal tile size based on image dimensions and available VRAM.

        Rules:
        - If image fits in a single tile (≤max_tile), use full image
        - Otherwise, pick largest tile that fits in VRAM budget
        - Always returns multiple of 8 (VAE requirement)
        """
        # If image fits in one tile, no tiling needed
        if image_h <= max_tile and image_w <= max_tile:
            # Round up to multiple of 8
            tile = max(image_h, image_w)
            tile = ((tile + 7) // 8) * 8
            return min(tile, max_tile)

        # Query available VRAM
        if vram_budget_gb is None:
            try:
                device = comfy.model_management.get_torch_device()
                if device.type == "cuda":
                    free_mem, total_mem = torch.cuda.mem_get_info(device)
                    # Use 60% of free VRAM for safety (VAE + intermediate buffers)
                    vram_budget_gb = (free_mem * 0.6) / (1024**3)
                else:
                    vram_budget_gb = 4.0  # Conservative default for CPU
            except Exception:
                vram_budget_gb = 4.0

        # Estimate VRAM per tile:
        # Encode: tile_pixels(fp32) + tile_latent(fp32) + VAE weights + intermediates
        # Rough formula: ~20 bytes per pixel for VAE encode/decode
        bytes_per_pixel = 20
        max_pixels = int(vram_budget_gb * (1024**3) / bytes_per_pixel)
        max_side = int(math.sqrt(max_pixels))

        # Clamp to range and round down to multiple of 8
        tile = max(min_tile, min(max_side, max_tile))
        tile = (tile // 8) * 8

        logger.info(
            f"[Radiance 4K] VRAM budget: {vram_budget_gb:.1f}GB -> "
            f"tile size: {tile}px (image: {image_w}x{image_h})"
        )
        return tile

    @staticmethod
    def get_optimal_temporal_size(
        tile_size_px: int,
        temporal_compression: int = 8,
        min_frames: int = 2,
        max_frames: int = 64,
        vram_budget_gb: float = None,
    ) -> int:
        """
        Determine optimal temporal chunk size (in LATENT frames) for 5D video
        VAE decode, given the spatial tile size already chosen. Spatial and
        temporal chunking share one VRAM budget instead of being sized
        independently; a bigger spatial tile leaves less room per frame.

        ALBABIT-FIX (VAE tiling integration): bytes_per_pixel_frame is
        back-solved from the LTX-2.5 crash data (1536px/~16.8GB: 8 frames
        clean, 16 hard-crashed) to land exactly on 8, not just "under 16".
        Rough proxy, not a precise memory model; see
        project_radiance_vae_tiling_seams memory for the investigation.
        """
        if vram_budget_gb is None:
            try:
                device = comfy.model_management.get_torch_device()
                if device.type == "cuda":
                    free_mem, total_mem = torch.cuda.mem_get_info(device)
                    vram_budget_gb = (free_mem * 0.6) / (1024**3)
                else:
                    vram_budget_gb = 4.0
            except Exception:
                vram_budget_gb = 4.0

        bytes_per_pixel_frame = 110
        max_pixel_frames = int(
            vram_budget_gb * (1024**3) / (bytes_per_pixel_frame * tile_size_px * tile_size_px)
        )
        compression = max(1, temporal_compression or 8)
        budget_frames = max_pixel_frames // compression
        frames = max(min_frames, min(budget_frames, max_frames))

        if budget_frames < min_frames:
            # BUDGET-FLOOR FIX: the max(min_frames, ...) floor means a budget
            # that fits ZERO frames still returns min_frames, and this used to
            # be logged as an ordinary budget-derived decision. That is an OOM
            # about to happen, reported as an "optimal" size. Say so, so the
            # operator lowers tile_size or frees VRAM instead of reading the
            # log as confirmation that the chunk size was chosen for them.
            logger.warning(
                f"[Radiance Temporal] VRAM budget {vram_budget_gb:.1f}GB at "
                f"tile={tile_size_px}px fits {budget_frames} latent frames, "
                f"below the {min_frames}-frame floor. Returning {frames} anyway "
                f"and this decode is expected to run out of memory. Lower "
                f"tile_size, lower temporal_size, or free VRAM."
            )
            return frames

        logger.info(
            f"[Radiance Temporal] VRAM budget: {vram_budget_gb:.1f}GB, tile={tile_size_px}px -> "
            f"temporal size: {frames} latent frames"
        )
        return frames

    @staticmethod
    def compute_tiles(total: int, tile_size: int, overlap: int) -> list:
        """
        Compute tile start positions along one axis.
        Returns list of (start, end) tuples.
        Ensures full coverage with consistent overlap.

        ``overlap`` must leave a positive stride.  It used not to be checked:
        with ``overlap >= tile_size`` the stride was zero or negative, ``pos``
        never advanced, and the loop appended tiles until the process ran out of
        memory, with ComfyUI hanging on no error and nothing in the log.  That was
        reachable from shipped widget defaults (temporal_size 2, temporal
        overlap 2), so this raises rather than silently clamping: a caller that
        gets here with a bad overlap has a bug the operator needs to see.
        """
        if tile_size <= 0:
            raise ValueError(f"tile_size must be positive, got {tile_size}")
        if overlap < 0:
            raise ValueError(f"overlap must not be negative, got {overlap}")
        if overlap >= tile_size:
            raise ValueError(
                f"overlap ({overlap}) must be smaller than tile_size "
                f"({tile_size}); an overlap at or above the tile size leaves no "
                f"stride and cannot tile."
            )

        if total <= tile_size:
            return [(0, total)]

        stride = tile_size - overlap
        positions = []
        pos = 0
        while pos < total:
            end = min(pos + tile_size, total)
            start = max(0, end - tile_size)
            positions.append((start, end))
            if end >= total:
                break
            pos += stride

        return positions

    @staticmethod
    def make_cosine_blend_weight_2d(
        tile_h: int,
        tile_w: int,
        overlap_top: int,
        overlap_bottom: int,
        overlap_left: int,
        overlap_right: int,
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        Create 2D cosine blend weight mask for a tile.

        Cosine blending eliminates visible seams — the weight transitions
        smoothly from 0→1 at borders, following a raised cosine curve.
        This is the standard approach used in DJV, Nuke, and Flame for
        tiled compositing.

        Returns: (1, tile_h, tile_w, 1) weight tensor
        """
        weight = torch.ones(tile_h, tile_w, device=device)

        # Cosine ramp: 0 → 1 over `n` pixels
        def cosine_ramp(n):
            if n <= 0:
                return torch.ones(0, device=device)
            t = torch.linspace(0, math.pi / 2, n, device=device)
            return torch.sin(t) ** 2  # Raised cosine (power-of-sine window)

        # Top edge
        if overlap_top > 0:
            ramp = cosine_ramp(overlap_top)
            weight[:overlap_top, :] *= ramp.unsqueeze(1)

        # Bottom edge
        if overlap_bottom > 0:
            ramp = cosine_ramp(overlap_bottom).flip(0)
            weight[-overlap_bottom:, :] *= ramp.unsqueeze(1)

        # Left edge
        if overlap_left > 0:
            ramp = cosine_ramp(overlap_left)
            weight[:, :overlap_left] *= ramp.unsqueeze(0)

        # Right edge
        if overlap_right > 0:
            ramp = cosine_ramp(overlap_right).flip(0)
            weight[:, -overlap_right:] *= ramp.unsqueeze(0)

        return weight.unsqueeze(0).unsqueeze(-1)  # (1, H, W, 1)

    @staticmethod
    def pad_to_multiple(
        tensor: torch.Tensor, multiple: int = 8, mode: str = "reflect"
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Pad image tensor to nearest multiple of `multiple`.
        Input: (B, H, W, C) — ComfyUI IMAGE format
        Returns: padded tensor, (pad_h, pad_w) for later cropping
        """
        b, h, w, c = tensor.shape
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple

        if pad_h == 0 and pad_w == 0:
            return tensor, (0, 0)

        # F.pad expects (N, C, H, W) and padding as (left, right, top, bottom)
        t = tensor.permute(0, 3, 1, 2)  # → (B, C, H, W)
        t = F.pad(t, (0, pad_w, 0, pad_h), mode=mode)
        return t.permute(0, 2, 3, 1), (pad_h, pad_w)  # → (B, H+padH, W+padW, C)


# ═══════════════════════════════════════════════════════════════════════════════
#                         4K VAE ENCODE
# ═══════════════════════════════════════════════════════════════════════════════


class RadianceVAE4KEncode:
    """
    Encode images up to 8K+ to VAE latents using production-grade tiling.

    Features:
    - VRAM-aware auto tile sizing (no OOM)
    - Cosine blend weights (invisible seams)
    - Pad-to-8 for VAE compatibility
    - Color space aware (Linear, sRGB, ACEScg, Log formats)
    - Alpha channel preservation
    - Progress reporting
    """

    SOURCE_SPACES = EXTENDED_SOURCE_SPACES
    LOG_SPACES = EXTENDED_LOG_SPACES

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "pixels": ("IMAGE",),
                "vae": ("VAE",),
                "source_space": (
                    cls.SOURCE_SPACES,
                    {
                        "default": "Linear",
                        "tooltip": "Input color space. Auto-linearized before VAE encode.",
                    },
                ),
            },
            "optional": {
                "tile_size": (
                    ["Auto", "512", "768", "1024", "1280", "1536"],
                    {
                        "default": "Auto",
                        "tooltip": "Tile size in pixels. Auto queries VRAM.",
                    },
                ),
                "overlap": (
                    "INT",
                    {
                        "default": 128, "min": 32, "max": 256, "step": 16,
                        "tooltip": "Overlap between tiles in pixels. 128px optimal for cosine blending.",
                    },
                ),
                "exposure": (
                    "FLOAT",
                    {
                        "default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1,
                        "tooltip": "Exposure adjustment in stops (linear space).",
                    },
                ),
                "alpha_handling": (
                    ["Preserve", "Ignore"],
                    {"default": "Preserve",
                     "tooltip": "Preserve alpha channel for roundtrip."},
                ),
                "hdr_mode": (
                    ["Clip (SDR)", "Soft Clip", "Compress (Log)", "Passthrough"],
                    {
                        "default": "Soft Clip",
                        "tooltip": "HDR handling before VAE encode.",
                    },
                ),
                # v2.0 Feature 4: Latent sampling mode
                "latent_sampling": (
                    LATENT_SAMPLING_MODES,
                    {
                        "default": "sample",
                        "tooltip": (
                            "How to sample from the VAE posterior distribution. "
                            "'mean' gives deterministic, lowest-noise results for img2img. "
                            "'sample' gives diverse outputs (default ComfyUI behaviour)."
                        ),
                    },
                ),
                # v2.0 Feature 8: Tile processing mode
                "processing_mode": (
                    ["sequential", "batched"],
                    {
                        "default": "sequential",
                        "tooltip": (
                            "'sequential' uses minimal VRAM (one tile at a time). "
                            "'batched' groups tiles for 2-4x faster encode on high-VRAM GPUs."
                        ),
                    },
                ),
            },
        }

    # v2.0: 5 outputs — latent_format and quality_report added
    RETURN_TYPES = ("LATENT", "IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("samples", "alpha", "metadata", "latent_format", "quality_report")
    OUTPUT_TOOLTIPS = (
        "Encoded latent — wire to Radiance Sampler or save node.",
        "Alpha channel tensor — wire to Radiance VAE 4K Decode alpha input.",
        "Encode metadata JSON — wire to Decode crop_padding for auto-crop.",
        "Latent format string (e.g. 'flux_16ch') — wire to Radiance Sampler latent_format input.",
        "Quality metrics JSON: clipping %, NaN count, latent range, tile info.",
    )
    FUNCTION = "encode"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    DESCRIPTION = (
        "v2.3 — Universal 4K+ VAE encode. Auto VAE-factor detection, "
        "video 5D latent support, 11 color spaces, mean/sample/mode latent sampling, "
        "batched tile processing, quality report, latent_format output. "
        "Compress (Log) HDR mode now passes full scene-linear range through log "
        "curve without any post-encode clamp — decode side handles noise cleanly."
    )

    # FIX-2: Map each log space to its linear→log converter so Compress (Log)
    # uses the correct curve per source/target space instead of always LogC4.
    _LOG_TO_LINEAR_CONVERTERS = None  # populated lazily after import succeeds
    _LINEAR_TO_LOG_CONVERTERS = None

    @classmethod
    def _get_log_to_linear(cls):
        if not _HAS_LOG_CURVES:
            return {}
        if cls._LOG_TO_LINEAR_CONVERTERS is None:
            cls._LOG_TO_LINEAR_CONVERTERS = {
                "ARRI LogC3": tensor_logc3_to_linear,   # Alexa Classic/Mini/SXT/LF
                "ARRI LogC4": tensor_logc4_to_linear,
                "Sony S-Log3": tensor_slog3_to_linear,
                "Panasonic V-Log": tensor_vlog_to_linear,
                "DaVinci Intermediate": tensor_davinci_intermediate_to_linear,
                "RED Log3G10": tensor_log3g10_to_linear,
            }
        return cls._LOG_TO_LINEAR_CONVERTERS

    @classmethod
    def _get_linear_to_log(cls):
        if not _HAS_LOG_CURVES:
            return {}
        if cls._LINEAR_TO_LOG_CONVERTERS is None:
            cls._LINEAR_TO_LOG_CONVERTERS = {
                "ARRI LogC3": tensor_linear_to_logc3,   # Alexa Classic/Mini/SXT/LF
                "ARRI LogC4": tensor_linear_to_logc4,
                "Sony S-Log3": tensor_linear_to_slog3,
                "Panasonic V-Log": tensor_linear_to_vlog,
                "DaVinci Intermediate": tensor_linear_to_davinci_intermediate,
                "RED Log3G10": tensor_linear_to_log3g10,
            }
        return cls._LINEAR_TO_LOG_CONVERTERS

    def _prepare_for_vae(
        self, img: torch.Tensor, source_space: str, exposure: float, hdr_mode: str
    ) -> torch.Tensor:
        """Full color pipeline: linearize → expose → VAE-space transform.

        v2.2 FIX (BUG-56): "Raw" source space is now an explicit early return.
        Previously "Raw" fell through all branches without matching, then had
        the hdr_mode transform applied (linear_to_srgb + clamp), which made
        "Raw" behave identically to "Linear" — defeating the purpose.
        """

        # Raw: complete passthrough — no linearization, no exposure, no hdr_mode.
        # The data goes to the VAE exactly as the operator provided it.
        if source_space == "Raw":
            return img

        # 1. Linearize log/gamma inputs
        if source_space in self.LOG_SPACES:
            if _HAS_LOG_CURVES:
                img = self._get_log_to_linear()[source_space](img)
                # 3.5.0: the curve alone leaves camera-gamut linear. Compress
                # (Log) keeps it (the round trip back to the same log space is
                # then exact, no gamut clip) and records it as the latent's
                # working gamut. Every other mode feeds a diffusion VAE, which
                # expects Rec.709 imagery, so convert to Rec.709 here.
                if hdr_mode != "Compress (Log)":
                    _m = _gamut_mat_t(LOG_SPACE_GAMUT[source_space], "Rec.709")
                    img = _safe_matrix_transform(img, _m.to(img.device))
            else:
                logger.warning(
                    f"[Radiance 4K Encode] Log curves unavailable — {source_space} "
                    f"will NOT be linearized. Install color_utils.py to fix this."
                )
        elif source_space == "sRGB":
            img = tensor_srgb_to_linear(img)
        elif source_space == "ACEScg":
            # ACEScg → Rec.709 linear (V-1: use precomputed matrix)
            # v2.2 FIX (BUG-48): Safe reshape — only transform first 3 channels.
            # reshape(-1, 3) crashes if img has 4+ channels (alpha from upstream).
            mat = _AP1_TO_REC709.to(img.device)
            img = _safe_matrix_transform(img, mat)
        elif source_space == "ACES 2065-1":
            mat = _AP0_TO_REC709.to(img.device)
            img = _safe_matrix_transform(img, mat)
        elif source_space == "Rec.2020 Linear":
            mat = _REC2020_TO_REC709.to(img.device)
            img = _safe_matrix_transform(img, mat)

        # 2. Exposure (linear domain)
        if exposure != 0.0:
            img = img * (2.0**exposure)

        # 2b. Clamp negatives — but only for modes that need it.
        #
        # v2.3 FIX: Mode-aware negative clamp with production commentary.
        #
        # ★ WHY WE CLAMP NEGATIVES AT ALL:
        #   Scene-linear images from EXR can contain negative values (out-of-gamut
        #   pixels, ringing artifacts from reconstruction filters, below-black in
        #   wide-gamut footage). Each HDR mode handles negatives differently:
        #
        # ★ Compress (Log): Log curves have well-defined below-black mappings.
        #   ARRI LogC4 encodes scene-linear down to ~−0.018 (code ~0.09).
        #   Sony S-Log3 encodes down to ~−0.014. We clamp at −0.03 (50% below
        #   LogC4's floor) so the log curve itself decides the mapping — the
        #   pre-clamp is just a safety net against extreme out-of-gamut values
        #   that would produce NaN in the log function.
        #
        # ★ sRGB modes (Soft Clip, Passthrough, Clip SDR): linear_to_srgb uses
        #   pow(x, 1/2.4) which produces NaN for x < 0. Must clamp at 0.0.
        if hdr_mode == "Compress (Log)":
            # v2.3: Widened from −0.02 → −0.03 to cover all log curve floors
            # without losing any encodable below-black detail.
            img = torch.clamp(img, min=-0.03)
        else:
            img = torch.clamp(img, min=0.0)

        # 3. Transform to VAE input space
        #
        # ★ COMPRESS (LOG) — THE HDR WORKHORSE:
        #   Log curves compress high dynamic range into a VAE-friendly [~0, ~1]
        #   domain while preserving perceptual uniformity. The compressed signal
        #   goes through the VAE which treats it like any [0, 1] image.
        #
        #   On decode, the inverse log curve decompresses back to scene-linear.
        #   The encode→decode roundtrip preserves the FULL HDR range of the
        #   original scene — no hard clamps, no data loss (v2.3).
        #
        #   NOTE: The VAE introduces small reconstruction noise (~±0.01 in coded
        #   space). For midtones this is invisible after decompression. For
        #   highlights (code > 0.9), the steep log→linear curve amplifies this
        #   noise exponentially. The decode side handles this with selective
        #   highlight denoise BEFORE decompression — see _denoise_log_highlights().
        if hdr_mode == "Compress (Log)":
            if _HAS_LOG_CURVES:
                # FIX-2: Use the log curve that matches the source space so the
                # encode and decode curves are symmetric. Previously this was
                # hardcoded to LogC4 regardless of source_space, causing a
                # mismatched encode/decode when shooting S-Log3, V-Log, etc.
                #
                # Preference order:
                #   1. If source_space is itself a log space → use its own curve
                #      (data is already linear after step 1, re-encode to same log)
                #   2. Otherwise default to LogC4 as a neutral choice (widest
                #      dynamic range of any standard log curve: ~26 stops)
                if source_space in self.LOG_SPACES:
                    compress_fn = self._get_linear_to_log()[source_space]
                else:
                    compress_fn = tensor_linear_to_logc4  # neutral default
                img = compress_fn(img)
            else:
                logger.warning(
                    "[Radiance 4K Encode] Compress (Log) HDR mode selected but "
                    "log curves are unavailable — falling back to SDR clamp."
                )
                img = torch.clamp(img, 0.0, 1.0)
        elif hdr_mode == "Passthrough":
            img = tensor_linear_to_srgb(img)
            # v2.2 FIX (BUG-50): Widened from [−0.05, 1.05] → [−0.05, 1.5].
            # Previous ceiling of 1.05 only preserved 1.12 linear (0.17 stops
            # of HDR headroom). New ceiling of 1.5 preserves up to ~2.5 linear
            # (~1.3 stops above SDR white). The VAE may introduce minor
            # reconstruction noise for values above 1.0, but this is a far
            # better trade-off than hard-clipping all HDR content.
            # Values above ~1.5 sRGB push the VAE too far out-of-distribution
            # and produce unacceptable artifacts, so 1.5 is the practical ceiling.
            img = torch.clamp(img, -0.05, 1.5)
        elif hdr_mode == "Soft Clip":
            img = tensor_linear_to_srgb(img)
            # Tanh soft clip at knee=0.85
            # V-5 FIX: Use consistent range constant for encode/decode symmetry.
            knee = 0.85
            rng = 1.0 - knee  # 0.15
            above = img > knee
            if above.any():
                result = img.clone()
                excess = (img[above] - knee) / rng
                result[above] = knee + rng * torch.tanh(excess)
                img = torch.clamp(result, min=0.0)
            else:
                img = torch.clamp(img, 0.0, 1.0)
        else:
            # Clip (SDR)
            img = tensor_linear_to_srgb(img)
            img = torch.clamp(img, 0.0, 1.0)

        return img

    def _tiled_encode(
        self,
        pixels: torch.Tensor,
        vae: Any,
        tile_size: int,
        overlap: int,
        pbar: Any = None,
        latent_sampling: str = "sample",
        processing_mode: str = "sequential",
        vae_factor: int = 8,
    ) -> Dict[str, Any]:
        """
        v2.0: Encode full image in overlapping tiles with cosine blend.

        Args:
            pixels: (B, H, W, 3) float32, already in VAE-space
            vae: ComfyUI VAE model
            tile_size: Tile size in pixels (must be multiple of 8)
            overlap: Overlap in pixels
            pbar: Optional progress bar
            latent_sampling: "sample" / "mean" / "mode"
            processing_mode: "sequential" / "batched"
            vae_factor: Spatial downscale factor (auto-detected, default 8)

        Returns:
            {"samples": (B, C, latH, latW), "radiance_meta": {...}} latent dict
        """
        b, h, w, c = pixels.shape
        device = pixels.device
        scale = vae_factor  # v2.0: no longer hardcoded

        # Compute tile grid
        tiles_y = TileEngine.compute_tiles(h, tile_size, overlap)
        tiles_x = TileEngine.compute_tiles(w, tile_size, overlap)
        total_tiles = len(tiles_y) * len(tiles_x)

        logger.info(
            f"[Radiance 4K Encode] {w}×{h} → {len(tiles_x)}×{len(tiles_y)} "
            f"tiles ({tile_size}px, {overlap}px overlap, {total_tiles} total, "
            f"mode={processing_mode}, factor={scale})"
        )

        lat_c = None
        lat_h = h // scale
        lat_w = w // scale
        lat_overlap = overlap // scale

        output = None
        weight_acc = None

        # Build flat list of all tile coords for batched mode
        all_tiles = [(yi, xi, y1, y2, x1, x2)
                     for yi, (y1, y2) in enumerate(tiles_y)
                     for xi, (x1, x2) in enumerate(tiles_x)]

        tile_idx = 0

        if processing_mode == "batched" and total_tiles > 1:
            # Encode all tiles in one GPU batch (uses more VRAM, much faster)
            tile_list = [pixels[:, y1:y2, x1:x2, :3].contiguous()
                         for (_, _, y1, y2, x1, x2) in all_tiles]
            # Pad tiles to same size (last row/col may be smaller)
            max_th = max(t.shape[1] for t in tile_list)
            max_tw = max(t.shape[2] for t in tile_list)

            padded = []
            for t in tile_list:
                pad_h2 = max_th - t.shape[1]
                pad_w2 = max_tw - t.shape[2]
                if pad_h2 > 0 or pad_w2 > 0:
                    t = F.pad(t.permute(0, 3, 1, 2), (0, pad_w2, 0, pad_h2)).permute(0, 2, 3, 1)
                padded.append(t)

            batch_pixels = torch.cat(padded, dim=0)  # (B*T, H, W, C)
            try:
                # Defensive no_grad: _tiled_encode may be called from a subclass or
                # test harness without the caller's no_grad context. Inference only.
                with torch.no_grad():
                    batch_result = _encode_with_sampling_mode(vae, batch_pixels, latent_sampling)
                if isinstance(batch_result, dict):
                    batch_result = batch_result["samples"]
                batch_result = batch_result.float().cpu()
                lat_c = batch_result.shape[1]
                tile_latents = list(batch_result.chunk(total_tiles, dim=0))
            except Exception as e:
                logger.warning(f"[Radiance] Batched encode failed ({e}), falling back to sequential")
                tile_latents = None
        else:
            tile_latents = None

        for tile_idx_i, (yi, xi, y1, y2, x1, x2) in enumerate(all_tiles):
            if tile_latents is not None:
                # Use pre-computed batch result
                tile_samples = tile_latents[tile_idx_i]
                if tile_samples.ndim == 3:
                    tile_samples = tile_samples.unsqueeze(0)
            else:
                # Sequential: encode one tile at a time
                # Defensive no_grad: idempotent if caller already has it,
                # protective if _tiled_encode is ever invoked standalone.
                tile = pixels[:, y1:y2, x1:x2, :3].contiguous()
                with torch.no_grad():
                    tile_samples = _encode_with_sampling_mode(vae, tile, latent_sampling)
                if isinstance(tile_samples, dict):
                    tile_samples = tile_samples["samples"]
                tile_samples = tile_samples.float().cpu()
                del tile

            # Lazy-init accumulators
            if output is None:
                lat_c = tile_samples.shape[1]
                output = torch.zeros(
                    (b, lat_c, lat_h, lat_w), dtype=torch.float32, device="cpu"
                )
                weight_acc = torch.zeros(
                    (1, 1, lat_h, lat_w), dtype=torch.float32, device="cpu"
                )

            th, tw = tile_samples.shape[2], tile_samples.shape[3]

            # Per-edge overlap flags
            ov_top   = lat_overlap if yi > 0 else 0
            ov_bot   = lat_overlap if yi < len(tiles_y) - 1 else 0
            ov_left  = lat_overlap if xi > 0 else 0
            ov_right = lat_overlap if xi < len(tiles_x) - 1 else 0

            blend_w = TileEngine.make_cosine_blend_weight_2d(
                th, tw, ov_top, ov_bot, ov_left, ov_right, device="cpu"
            )
            blend_w = blend_w.squeeze(-1).unsqueeze(1)

            lx1 = x1 // scale
            ly1 = y1 // scale
            lx2 = lx1 + tw
            ly2 = ly1 + th

            output[:, :, ly1:ly2, lx1:lx2] += tile_samples * blend_w
            weight_acc[:, :, ly1:ly2, lx1:lx2] += blend_w

            tile_idx += 1
            if pbar:
                pbar.update_absolute(tile_idx, total_tiles)

            if tile_latents is None:
                del tile_samples, blend_w

        # V-6 FIX: Call empty_cache once after all tiles, not per-tile.
        # Per-tile empty_cache is a CUDA sync barrier (~1-2ms each) that adds
        # 24-48ms of pure overhead for a 4K image split into 24 tiles.
        if device.type == "cuda":
            torch.cuda.empty_cache()

        weight_acc = torch.clamp(weight_acc, min=1e-3)
        output = output / weight_acc

        return {"samples": output.to(device), "_total_tiles": total_tiles}


    def encode(
        self,
        pixels: torch.Tensor,
        vae: Any,
        source_space: str = "Linear",
        tile_size: str = "Auto",
        overlap: int = 128,
        exposure: float = 0.0,
        alpha_handling: str = "Preserve",
        hdr_mode: str = "Soft Clip",
        latent_sampling: str = "sample",
        processing_mode: str = "sequential",
    ) -> Tuple:
        """v2.3: Universal encode supporting 4D image and 5D video latents.

        Compress (Log) HDR mode now passes full scene-linear range through the
        log curve without any post-encode clamp. The decode side handles VAE
        reconstruction noise with soft-shoulder + highlight denoise.
        """
        import time as _time
        t_start = _time.time()

        # v2.0 Feature 2: Auto-detect VAE factor and format
        vae_factor = detect_vae_factor(vae)
        latent_fmt = detect_latent_format(vae)

        # v2.0 Feature 3: Handle 5D video latents (B, F, H, W, C)
        is_video = pixels.ndim == 5

        # Check if the VAE natively supports 3D latents (e.g., Wan Video, Cosmos)
        is_3d_vae = False
        if hasattr(vae, "latent_dim") and getattr(vae, "latent_dim") == 3:
            is_3d_vae = True

        if is_video and not is_3d_vae:
            # v2.3.1 FIX (V-B16): renamed the frame-count local from `F` to
            # `num_frames`. `F` is the module-level alias for torch.nn.functional
            # imported at the top of the file; re-binding it in function scope
            # turned any later `F.pad` / `F.interpolate` call in this function
            # into an AttributeError. Only the early-return in this branch
            # hid the landmine — harmless today, broken after the next refactor.
            B, num_frames, H, W, C = pixels.shape
            logger.info(
                f"[Radiance 4K Encode v2.3] Video: {B}×{num_frames}×{W}×{H}, space={source_space}"
            )
            # STREAM-FIX: encode frame by frame into ONE pre-allocated CPU
            # buffer instead of a list of GPU latents plus a torch.stack.
            # The list held every frame's latent on the GPU for the whole clip
            # and stack() then allocated a second full-clip copy, so a 500-frame
            # 4K clip at 16 channels needed ~4.1 GB resident plus ~4.1 GB for the
            # stack. Writing each frame into its slice and dropping the frame
            # reference makes the returned latent the only clip-sized tensor
            # alive; peak above it is one frame, whatever the clip length.
            all_frames = None
            frame_pad_h, frame_pad_w = 0, 0
            for fi in range(num_frames):
                frame = pixels[:, fi, ...]  # (B, H, W, C)
                latent_frame, _, _, _, _ = self.encode(
                    frame, vae, source_space, tile_size, overlap, exposure,
                    alpha_handling, hdr_mode, latent_sampling, processing_mode
                )
                lat = latent_frame["samples"]
                if all_frames is None:
                    # (B, C, F, latH, latW), shaped from the first frame so the
                    # channel count comes from the VAE, never assumed. The clip
                    # buffer lives on ComfyUI's intermediate device (CPU unless
                    # --gpu-only), which is where comfy.sd.VAE.encode already
                    # puts its own output; keeping it in VRAM is what made clip
                    # length a VRAM budget.
                    _accum_dev = _intermediate_device()
                    all_frames = torch.empty(
                        (lat.shape[0], lat.shape[1], num_frames) + tuple(lat.shape[2:]),
                        dtype=lat.dtype, device=_accum_dev,
                    )
                    # STREAM-FIX / PAD-FIX: the per-frame call pads to a multiple
                    # of vae_factor and reports the real pad in its own meta.
                    # This branch used to hardcode pad_h/pad_w to 0 on the video
                    # radiance_meta and keep only `samples`, so decode read zeros
                    # and skipped the crop: a 1920x1080 clip through a factor-32
                    # VAE decoded to 1088 tall, the last 8 rows being reflect-pad
                    # of row 1079. Every frame pads identically (same H, W, same
                    # factor), so the first frame's pad is the clip's pad.
                    _fm = latent_frame.get("radiance_meta") or {}
                    frame_pad_h = int(_fm.get("pad_h", 0))
                    frame_pad_w = int(_fm.get("pad_w", 0))
                all_frames[:, :, fi] = lat.to(all_frames.device)
                del lat, latent_frame, frame
            if all_frames is None:                      # num_frames == 0
                all_frames = torch.empty(
                    (B, 0, 0, 0, 0), dtype=torch.float32,
                    device=_intermediate_device())
            # v2.3.1 FIX (V-B23): include radiance_meta on the video latent
            # dict so decode() can auto-recover hdr_mode / source_space /
            # vae_factor on the other side. Previously the key was missing,
            # so a Compress(Log) video roundtrip lost its HDR mode and decode
            # either fell back to Clip(SDR) or required force_hdr_decode=True.
            video_latent = {
                "samples": all_frames,
                "latent_format": latent_fmt,
                "radiance_meta": {
                    "pad_h": frame_pad_h, "pad_w": frame_pad_w,
                    "vae_factor": vae_factor,
                    "latent_format": latent_fmt,
                    "source_space": source_space,
                    "hdr_mode": hdr_mode,
                    "working_gamut": _encode_working_gamut(source_space, hdr_mode),
                    "latent_fingerprint": latent_fingerprint(all_frames),
                },
            }
            # Build a minimal quality/meta output for video
            total_time_ms = int((_time.time() - t_start) * 1000)
            # ALPHA-FIX: the still path extracts pixels[..., 3:4] under
            # alpha_handling="Preserve"; this branch used to return
            # torch.ones unconditionally, so decode composited solid white
            # over a real matte on every RGBA clip. Shape is (B*F, H, W, 1),
            # which is the layout decode()'s frame_alphas chunking expects.
            # Match device of input pixels so downstream nodes don't hit a device
            # mismatch when alpha is moved to GPU alongside the image tensor.
            if C == 4 and alpha_handling == "Preserve":
                alpha_out = (
                    pixels[..., 3:4].reshape(B * num_frames, H, W, 1)
                    .clone().float()
                )
            else:
                alpha_out = torch.ones(
                    (B * num_frames, H, W, 1),
                    dtype=torch.float32, device=pixels.device,
                )
            meta = json.dumps({"node": "RadianceVAE4KEncode", "video": True,
                               "frames": num_frames, "latent_format": latent_fmt,
                               "pad_h": frame_pad_h, "pad_w": frame_pad_w})
            qr = json.dumps({"version": "3.0", "video": True, "frames": num_frames,
                             "latent_format": latent_fmt, "encode_time_ms": total_time_ms})
            return (video_latent, alpha_out, meta, latent_fmt, qr)

        b, h, w, c = pixels.shape
        logger.info(
            f"[Radiance 4K Encode v2.3] Input: {w}×{h} ({b} frames), "
            f"space={source_space}, sampling={latent_sampling}, vae_factor={vae_factor}"
        )

        # Extract alpha
        has_alpha = c == 4
        if has_alpha and alpha_handling == "Preserve":
            alpha = pixels[..., 3:4].clone().float()
        else:
            alpha = torch.ones((b, h, w, 1), dtype=torch.float32, device=pixels.device)

        target_device = comfy.model_management.get_torch_device()
        img = pixels[..., :3].clone().float().to(target_device)

        # V-3 FIX: Clone to CPU immediately instead of keeping a 50MB+ GPU copy.
        # The original pixels are only needed for post-encode quality metrics,
        # which run on CPU anyway. Keeping them on GPU competes with VAE for VRAM.
        pixels_orig = img.detach().cpu()

        # Color pipeline
        img = self._prepare_for_vae(img, source_space, exposure, hdr_mode)

        # Pad to multiple of vae_factor (minimum 8 for all known VAEs)
        pad_multiple = max(vae_factor, 8)
        img, (pad_h, pad_w) = TileEngine.pad_to_multiple(img, pad_multiple)
        padded_h, padded_w = img.shape[1], img.shape[2]

        # Determine tile size
        if tile_size == "Auto":
            ts = TileEngine.get_optimal_tile_size(padded_h, padded_w)
        else:
            ts = int(tile_size)
        ts = (ts // pad_multiple) * pad_multiple

        overlap = min(overlap, ts // 2)
        overlap = (overlap // 8) * 8
        overlap = max(16, overlap)

        # NaN/Inf guard — mode-aware for Passthrough's extended range
        if torch.isnan(img).any() or torch.isinf(img).any():
            logger.warning("[Radiance 4K Encode v2.3] NaN/Inf detected — sanitizing")
            if hdr_mode == "Passthrough":
                img = torch.nan_to_num(img, nan=0.0, posinf=1.5, neginf=-0.05)
            else:
                img = torch.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)

        pbar = comfy.utils.ProgressBar(100)
        total_tiles = 1

        with torch.no_grad():
            if padded_h <= ts and padded_w <= ts:
                # Single tile — use latent_sampling mode
                enc_samples = _encode_with_sampling_mode(vae, img, latent_sampling)
                if isinstance(enc_samples, dict):
                    enc_samples = enc_samples["samples"]
                latent = {
                    "samples": enc_samples,
                    "_total_tiles": 1,
                }
                pbar.update_absolute(100, 100)
            else:
                latent = self._tiled_encode(
                    img, vae, ts, overlap, pbar,
                    latent_sampling=latent_sampling,
                    processing_mode=processing_mode,
                    vae_factor=vae_factor,
                )
                total_tiles = latent.pop("_total_tiles", 1)

        # v2.0 Feature 6: Embed radiance_meta in latent dict for auto-crop in decode
        radiance_meta = {
            "pad_h": pad_h, "pad_w": pad_w,
            "vae_factor": vae_factor, "latent_format": latent_fmt,
            "source_space": source_space, "hdr_mode": hdr_mode,
            "working_gamut": _encode_working_gamut(source_space, hdr_mode),
            # 3.5.0: lets decode tell this latent from a sampled copy that
            # still carries the dict (see verify_radiance_meta).
            "latent_fingerprint": latent_fingerprint(latent["samples"]),
        }
        latent["radiance_meta"] = radiance_meta
        # v2.3.1 FIX (V-B11): mirror the top-level "latent_format" key that the
        # video encode path sets, so downstream consumers that inspect
        # latent["latent_format"] get a consistent shape from both paths.
        latent["latent_format"] = latent_fmt

        # Standard metadata JSON (backward compat wire via crop_padding)
        metadata = {
            "node": "RadianceVAE4KEncode",
            "resolution": f"{w}×{h}",
            "padded": f"{padded_w}×{padded_h}" if pad_h or pad_w else "none",
            "tile_size": ts, "overlap": overlap,
            "source_space": source_space, "hdr_mode": hdr_mode,
            "exposure": exposure, "pad_h": pad_h, "pad_w": pad_w,
            "vae_factor": vae_factor, "latent_format": latent_fmt,
            "latent_sampling": latent_sampling,
        }

        # v2.0 Feature 5: Quality metrics
        encode_time_ms = int((_time.time() - t_start) * 1000)
        quality_report = build_encode_quality_report(
            pixels_orig, img.cpu(), latent["samples"].cpu(),
            vae_factor, latent_fmt, source_space, hdr_mode,
            ts, total_tiles, encode_time_ms,
        )

        return (latent, alpha, json.dumps(metadata, indent=2), latent_fmt, quality_report)



# ═══════════════════════════════════════════════════════════════════════════════
#                         4K VAE DECODE
# ═══════════════════════════════════════════════════════════════════════════════


class RadianceVAE4KDecode:
    """
    Decode VAE latents to 4K+ images using production-grade tiling.

    Features:
    - VRAM-aware auto tile sizing
    - Cosine blend weights (invisible seams)
    - Direct .rhdr export for Radiance Viewer
    - Color space output (Linear, sRGB, ACEScg, Log formats)
    - Alpha channel restoration
    - Inverse tonemap for HDR recovery
    """

    TARGET_SPACES = EXTENDED_TARGET_SPACES
    LOG_SPACES = EXTENDED_LOG_SPACES

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
                "target_space": (
                    cls.TARGET_SPACES,
                    {
                        "default": "sRGB",
                        "tooltip": (
                            "Output color space. "
                            "'sRGB' (default): correct for SaveImage, PreviewImage, and all standard "
                            "ComfyUI nodes — output is display-ready [0,1] sRGB. "
                            "'Linear': scene-linear for VFX pipelines (Nuke, Resolve, Houdini). "
                            "Requires hdr_output=True for true linear passthrough; "
                            "with hdr_output=False the image is re-encoded to sRGB for "
                            "ComfyUI compatibility. "
                            "Log/ACEScg/Rec.2020 spaces are scene-referred — connect to a "
                            "tonemap node before SaveImage. Camera log targets carry their "
                            "own gamut (LogC4 = AWG4, LogC3 = AWG3, S-Log3 = S-Gamut3.Cine, "
                            "V-Log = V-Gamut, DaVinci Intermediate = DWG, Log3G10 = "
                            "REDWideGamutRGB). 'Raw' is linear Rec.709 with no transfer."
                        ),
                    },
                ),
            },
            "optional": {
                "tile_size": (
                    ["Auto", "512", "768", "1024", "1280", "1536"],
                    {
                        "default": "Auto",
                        "tooltip": "Tile size for decode. Auto queries VRAM.",
                    },
                ),
                "overlap": (
                    "INT",
                    {
                        "default": 128,
                        "min": 32,
                        "max": 256,
                        "step": 16,
                        "tooltip": "Overlap between tiles. 128px optimal for cosine blending.",
                    },
                ),
                # ALBABIT-FIX: Temporal chunking for video VAE decode, integrated
                # with spatial tiling via comfy's own vae.decode_tiled() instead
                # of Radiance's own stacked chunk-then-tile loop. See
                # project_radiance_vae_tiling_seams memory.
                "temporal_size": (
                    ["Auto", "2", "4", "8", "16", "32", "64"],
                    {
                        "default": "Auto",
                        "tooltip": (
                            "Temporal chunk size in LATENT frames (not pixel frames — unlike "
                            "ComfyUI's native 'VAE Decode (Tiled)', which counts pixel frames and "
                            "divides internally by the VAE's temporal compression). "
                            "Auto sizes it from the same VRAM budget as tile_size, computed "
                            "jointly since a larger spatial tile leaves less room per frame. "
                            "A manual value forces chunking at that many latent frames "
                            "regardless of tile_size. Images (4D latents) are not affected."
                        ),
                    },
                ),
                "temporal_overlap": (
                    "INT",
                    {
                        "default": 2,
                        "min": 0,
                        "max": 32,
                        "step": 1,
                        "tooltip": (
                            "Overlap in latent frames between temporal chunks. 0 = hard cuts "
                            "at chunk boundaries. A small value (1–2) blends them instead. "
                            "Only active when the video actually needs temporal chunking "
                            "(temporal_size='Auto' decides on its own, or set a manual value "
                            "smaller than the clip's latent frame count)."
                        ),
                    },
                ),
                "exposure_adjust": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -10.0,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": "Post-decode exposure in stops.",
                    },
                ),
                "alpha": (
                    "IMAGE",
                    {"tooltip": "Alpha channel from Radiance VAE 4K Encode."},
                ),
                "hdr_mode": (
                    ["Clip (SDR)", "Soft Clip", "Compress (Log)", "Passthrough"],
                    {
                        "default": "Clip (SDR)",
                        "tooltip": (
                            "Must match encode setting. "
                            "'Compress (Log)' (v2.3): true HDR — soft-shouldered, "
                            "highlight-denoised, no hard clamp. Produces clean HDR output "
                            "across the full log-curve dynamic range. "
                            "'Soft Clip': inverts tanh rolloff to recover highlights. "
                            "'Passthrough': extended sRGB range."
                        ),
                    },
                ),
                "display_tonemap": (
                    ["Reinhard", "ACES Filmic", "None"],
                    {
                        "default": "ACES Filmic",
                        "tooltip": (
                            "v2.3.8: display_tonemap is NOW THE SOLE TONEMAP CONTROL — "
                            "independent of hdr_output. "
                            "'Reinhard' or 'ACES Filmic': tonemap ALWAYS fires for "
                            "Compress(Log), even when hdr_output=True. "
                            "Reinhard → smooth rolloff [0,∞)→[0,1), hue-preserving. "
                            "ACES Filmic → filmic contrast, clips cleanly above ~10 lin. "
                            "'None': NO tonemap regardless of hdr_output. Scene-linear "
                            "values far above 1.0 pass through — GUARANTEED OVEREXPOSURE in "
                            "ComfyUI preview. Use only with an OCIO-aware downstream viewer. "
                            "Ignored for all non-Compress(Log) hdr_modes."
                        ),
                    },
                ),
                "inverse_tonemap": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Expand SDR to HDR (recover highlights).",
                    },
                ),
                "target_stops": (
                    "FLOAT",
                    {
                        "default": 12.0,
                        "min": 8.0,
                        "max": 16.0,
                        "step": 0.5,
                        "tooltip": "Target dynamic range for inverse tonemap.",
                    },
                ),
                "crop_padding": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "JSON metadata from encode (auto-crops padding). Leave empty to skip.",
                    },
                ),
                "export_rhdr": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Export .rhdr sidecar for Radiance Viewer.",
                    },
                ),
                # BUG-G FIX: Precision was previously invisible to the operator
                # (always fp16). Expose it so production shots with values > 65504
                # (sun, explosions, VDB fire) can be delivered in full 32-bit range.
                "rhdr_precision": (
                    ["f16", "f32"],
                    {
                        "default": "f16",
                        "tooltip": (
                            "RHDR export precision. "
                            "'f16' (fp16, default): smaller files, values capped at 65504 — "
                            "adequate for most VFX material. "
                            "'f32' (fp32): full 32-bit range, ~2× file size — "
                            "use for shots with extreme linear values (direct sun, fire VDBs)."
                        ),
                    },
                ),
                "source_space": (
                    EXTENDED_SOURCE_SPACES,
                    {
                        "default": "Linear",
                        "tooltip": (
                            "Source color space used during encode. "
                            "⚠ CRITICAL for hdr_mode='Compress (Log)': must exactly match the "
                            "log curve selected at encode time. Wrong value = wrong decompression "
                            "curve = incorrect colors even if exposure looks OK. "
                            "Valid log spaces: ARRI LogC4, ARRI LogC3, Sony S-Log3, "
                            "Panasonic V-Log, DaVinci Intermediate, RED Log3G10. "
                            "If source_space is set to a non-log value (Linear, sRGB, ACEScg) "
                            "with Compress(Log), a WARNING is logged and LogC4 fallback is used. "
                            "For diffusion model output (SDXL/FLUX/SD1.5), use Clip(SDR) or "
                            "Soft Clip instead — those models output sRGB, not log."
                        ),
                    },
                ),
                # v2.0 Feature 8: Tile processing mode for decode
                # v2.1 FIX (BUG-44): Removed "batched" — decode is much more VRAM-intensive
                # than encode and was never actually implemented for decode. The parameter
                # existed in the UI but silently ran sequential. Encode batched still works.
                "processing_mode": (
                    ["sequential"],
                    {
                        "default": "sequential",
                        "tooltip": "'sequential' processes one tile at a time with minimal VRAM usage.",
                    },
                ),
                # ── HDR-FIX: 32-bit HDR output controls ───────────────────────────────
                "force_hdr_decode": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "HDR DIRECT PATH — enables 32-bit HDR output for direct "
                            "encode→decode pipelines (no sampler between nodes). "
                            "When True: hdr_mode is always respected, even if radiance_meta "
                            "is absent from the latent. "
                            "Keep False when decoding after diffusion sampling — the sampler "
                            "produces sRGB-encoded latents and Log/SoftClip inversion would "
                            "produce incorrect color output."
                        ),
                    },
                ),
                "hdr_output": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "32-BIT HDR OUTPUT — when True, skips the final [0,1] clamp "
                            "so the output IMAGE tensor carries full 32-bit scene-linear "
                            "values (>1.0 for bright highlights, <0.0 for below-black). "
                            "Required for Linear/ACEScg/Log target spaces in HDR VFX pipelines. "
                            "Disable when feeding SDR nodes (preview, PNG export, etc.)."
                        ),
                    },
                ),
                # v2.4: Decode-time noise injection (LTX LumiVid / arxiv 2604.11788)
                "decode_noise_scale": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 0.1,
                        "step": 0.001,
                        "tooltip": (
                            "v2.4 — Decode-time noise injection. "
                            "Mixes a small amount of Gaussian noise into the latent BEFORE "
                            "VAE decode using lerp: noised = (1-scale)*latent + scale*noise. "
                            "Breaks up coherent VAE decoder grid artifacts in flat highlight "
                            "regions. Effect is most valuable for Compress (Log) mode where "
                            "the log→linear decompression exponentially amplifies any residual "
                            "decoder artifact in the highlight band. "
                            "0.0 = disabled (default — safe for existing workflows). "
                            "Per-profile recommended starting points:\n"
                            "  ARRI LogC4:           0.018\n"
                            "  Sony S-Log3:          0.018\n"
                            "  ARRI LogC3:           0.025  (matches LTX default)\n"
                            "  Panasonic V-Log:      0.022\n"
                            "  DaVinci Intermediate: 0.030\n"
                            "  RED Log3G10:          0.035\n"
                            "Ignored for non-Compress(Log) hdr_modes."
                        ),
                    },
                ),
            },
        }

    # v2.0: 3 outputs — latent_format added
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "metadata", "latent_format")
    OUTPUT_TOOLTIPS = (
        "Decoded image tensor.",
        "Decode metadata JSON.",
        "Latent format detected from VAE (e.g. 'flux_16ch').",
    )
    FUNCTION = "decode"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    DESCRIPTION = (
        "v2.4 — Universal 4K+ VAE decode. True HDR Compress(Log) with soft-shoulder, "
        "highlight denoise, and per-profile decode_noise_scale for clean, clamp-free "
        "HDR output. Auto VAE-factor, 11 color spaces, video 5D support, "
        "auto-crop from radiance_meta."
    )

    def _tiled_decode(
        self,
        samples: Dict[str, Any],
        vae: Any,
        tile_size_px: int,
        overlap_px: int,
        pbar: Any = None,
        vae_factor: int = 8,
    ) -> torch.Tensor:
        """
        Decode full latent in overlapping spatial tiles with cosine blend.

        Temporal chunking is handled upstream in decode() before this method
        is called, so samples["samples"] may be 4D (B, C, H, W) or a 5D
        (B, C, T, H, W) chunk whose T dimension fits in VRAM.

        Args:
            samples:      {"samples": latent} — 4D or 5D
            vae:          ComfyUI VAE model
            tile_size_px: Tile size in pixel space
            overlap_px:   Spatial overlap in pixel space
            pbar:         Optional ComfyUI ProgressBar
            vae_factor:   Spatial downscale factor (auto-detected by caller)

        Returns:
            Tuple of:
              - decoded frame tensor (B*F, pixH, pixW, C) or (B, pixH, pixW, C)
              - int | None — pixel frame count if a 3D-native VAE was detected
        """
        latent = samples["samples"]

        b = latent.shape[0]
        lat_h, lat_w = latent.shape[-2], latent.shape[-1]
        device = latent.device
        # v2.1 FIX (BUG-42): Use detected vae_factor instead of hardcoded 8.
        # Stable Cascade Stage C uses factor=42 — hardcoded 8 gave completely
        # wrong tile pixel coordinates, corrupting the output.
        scale = vae_factor

        pix_h = lat_h * scale
        pix_w = lat_w * scale

        # Tile in latent space
        lat_tile = tile_size_px // scale
        lat_overlap = overlap_px // scale

        tiles_y = TileEngine.compute_tiles(lat_h, lat_tile, lat_overlap)
        tiles_x = TileEngine.compute_tiles(lat_w, lat_tile, lat_overlap)
        total_tiles = len(tiles_y) * len(tiles_x)

        logger.info(
            f"[Radiance 4K Decode] {pix_w}×{pix_h} → {len(tiles_x)}×{len(tiles_y)} "
            f"tiles ({tile_size_px}px/{lat_tile}lat, {total_tiles} total)"
        )

        # FIX-3: Do NOT pre-allocate output with a hardcoded 3 channels.
        # The VAE channel count is unknown until the first tile is decoded —
        # inpainting VAEs, future model variants, or custom VAEs may return 4+
        # channels. Allocate lazily from the first tile's actual shape.
        output = None
        weight_acc = None
        out_c = None  # Will be set from first tile

        tile_idx = 0
        _tile_video_frames = None  # Populated on first tile if 3D VAE detected

        for yi, (ly1, ly2) in enumerate(tiles_y):
            for xi, (lx1, lx2) in enumerate(tiles_x):
                # Extract latent tile
                # Use ellipsis to gracefully handle both 4D (H, W) and 5D (F, H, W) slicing
                tile_lat = latent[..., ly1:ly2, lx1:lx2].contiguous()

                # Decode on GPU — defensive no_grad: idempotent if caller has it,
                # protective if _tiled_decode is ever invoked standalone.
                with torch.no_grad():
                    tile_decoded = vae.decode(tile_lat).float().cpu()

                # FIX-4: 3D (temporal) VAEs return (B, F, H, W, C) — a 5D tensor.
                # Reshape to (B*F, H, W, C) so all downstream accumulation logic
                # treats frames as batch elements (standard ComfyUI IMAGE format).
                if tile_decoded.ndim == 5:
                    _B5, _F5, _H5, _W5, _C5 = tile_decoded.shape
                    if _tile_video_frames is None:
                        _tile_video_frames = _F5
                    tile_decoded = tile_decoded.reshape(_B5 * _F5, _H5, _W5, _C5)
                    # Also update b so the accumulator is sized for all frames
                    b = tile_decoded.shape[0]

                # Pixel coordinates
                px1 = lx1 * scale
                py1 = ly1 * scale
                th, tw = tile_decoded.shape[1], tile_decoded.shape[2]

                # FIX-3: Lazy-init accumulators on first tile so channel count is
                # taken from the actual VAE output, not assumed to be 3.
                if output is None:
                    out_c = tile_decoded.shape[3]
                    output = torch.zeros(
                        (b, pix_h, pix_w, out_c), dtype=torch.float32, device="cpu"
                    )
                    weight_acc = torch.zeros(
                        (1, pix_h, pix_w, 1), dtype=torch.float32, device="cpu"
                    )
                    logger.info(f"[Radiance 4K Decode] VAE output channels: {out_c}")

                # Per-edge overlap (pixel space)
                pix_overlap = lat_overlap * scale
                ov_top = pix_overlap if yi > 0 else 0
                ov_bot = pix_overlap if yi < len(tiles_y) - 1 else 0
                ov_left = pix_overlap if xi > 0 else 0
                ov_right = pix_overlap if xi < len(tiles_x) - 1 else 0

                blend_w = TileEngine.make_cosine_blend_weight_2d(
                    th, tw, ov_top, ov_bot, ov_left, ov_right, device="cpu"
                )

                # Accumulate
                output[:, py1 : py1 + th, px1 : px1 + tw, :] += tile_decoded * blend_w
                weight_acc[:, py1 : py1 + th, px1 : px1 + tw, :] += blend_w

                tile_idx += 1
                if pbar:
                    pbar.update_absolute(tile_idx, total_tiles)

                # Free tile references (GPU texture freed by Python GC)
                del tile_lat, tile_decoded, blend_w

        # V-6 FIX: Single empty_cache after all tiles instead of per-tile.
        # Note: If decode OOMs on very large images, restore per-tile cleanup
        # as a fallback — but for most workloads this saves 24-48ms of sync overhead.
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Normalize by accumulated weight (guard against near-zero at tile edges)
        # STREAM-FIX: divide in place. `output` is the full (b, pixH, pixW, C)
        # result, so `output = output / weight_acc` allocated a second copy of
        # the entire decode before freeing the first -- a 2x spike on the one
        # buffer in this method that is already as large as the answer.
        weight_acc.clamp_(min=1e-3)
        output.div_(weight_acc)
        return output.to(device), _tile_video_frames

    def _vae_output_to_target(
        self,
        img: torch.Tensor,
        target_space: str,
        hdr_mode: str,
        exposure: float,
        inverse_tonemap: bool,
        target_stops: float,
        source_space: str = "Linear",
        hdr_output: bool = False,
        display_tonemap: str = "ACES Filmic",
        working_gamut: str = "Rec.709",
    ) -> torch.Tensor:
        """Convert VAE output to target color space.

        v2.3 TRUE HDR pipeline — all log profiles produce clean, clamp-free HDR.

        Pipeline: VAE output → HDR Decompress → Linearize → Inverse Tonemap → Exposure → Target Space

        All operations happen in linear space to prevent double-gamma errors.

        v2.3 CHANGES (Compress (Log) — clamp-free HDR):
          - Hard clamp [0, 1] REMOVED from Compress(Log) pre-decompression
          - Replaced with _soft_log_shoulder() — preserves super-whites up to
            code 1.08 (~55 linear for LogC4, vs 38 with hard clamp at 1.0)
          - Added _denoise_log_highlights() — selective spatial denoise in
            log-coded space BEFORE decompression. Suppresses VAE noise where
            the log→linear curve is steepest (code > 0.80), while passing
            midtones and shadows through untouched. This is the same principle
            used by DaVinci Resolve Highlight Recovery and Nuke SoftClip.
          - Result: clean, noise-free HDR output at full dynamic range, with
            no hard ceiling on recoverable scene-linear values.

        Previous v2.2 fix audit (still active):
          - BUG-48: Color matrices safe for 4+ channels (alpha passthrough)
          - BUG-50: Passthrough clamp widened from [−0.05, 1.1] → [−0.05, 1.6]
          - BUG-52: Soft Clip atanh safety raised from 0.9999 → 0.999999
          - BUG-51: Extended sRGB EOTF for Soft Clip / Passthrough values > 1.0

        HDR-FIX (hdr_output):
          When hdr_output=True, the final result is NOT clamped to [0,1].
          Scene-linear values > 1.0 (bright highlights, sky, fire, explosions)
          and < 0.0 (below-black, out-of-gamut in ACEScg) are preserved as-is
          in the output fp32 tensor. Use target_space="Linear" or "ACEScg" for
          full HDR scene-referred output.
        """

        # Step 1: VAE output → Linear (ALWAYS linearize)
        #
        # ★ PRE-DECOMPRESSION PIPELINE (v2.3 — clamp-free HDR for all log profiles):
        #
        # Each hdr_mode has a different valid range in the encoded space.
        # The pre-clamp MUST handle VAE reconstruction noise WITHOUT destroying
        # legitimate HDR signal. This is mode-specific:
        #
        # ┌──────────────────┬─────────────────────────────────────────────────┐
        # │ Compress (Log)   │ v2.3: Soft shoulder + highlight denoise.       │
        # │                  │ NO hard clamp. Preserves full HDR range.       │
        # │                  │ Denoise in log-coded space BEFORE decompress   │
        # │                  │ to prevent exponential noise amplification.    │
        # ├──────────────────┼─────────────────────────────────────────────────┤
        # │ Soft Clip        │ Hard clamp [0, 1] — tanh ceiling is 1.0       │
        # ├──────────────────┼─────────────────────────────────────────────────┤
        # │ Passthrough      │ Wide clamp [−0.05, 1.6] — extended sRGB       │
        # ├──────────────────┼─────────────────────────────────────────────────┤
        # │ Clip (SDR)       │ Hard clamp [0, 1] — SDR by definition         │
        # └──────────────────┴─────────────────────────────────────────────────┘
        #
        if hdr_mode == "Compress (Log)":
            # ── v2.3 TRUE HDR PIPELINE (replaces v2.2 hard clamp) ──────────
            #
            # PREVIOUS CODE (v2.2):
            #   img = torch.clamp(img, 0.0, 1.0)   ← DESTROYED HDR HEADROOM
            #
            # WHY THE HARD CLAMP WAS WRONG:
            #   The comment said "values above 1.0 are VAE noise, not signal."
            #   This is partially true: MOST above-1.0 values are noise. But
            #   the VAE CAN reconstruct legitimate signal at 1.001–1.05 for
            #   very bright highlights. Hard-clamping at exactly 1.0 means the
            #   maximum recoverable linear value is whatever the log curve maps
            #   to at code 1.0 (LogC4: ~38 linear). With soft shoulder at 1.08,
            #   we can recover up to ~55 linear — 45% more HDR headroom.
            #
            # v2.3 PIPELINE (3 stages):
            #   1. Soft shoulder — gently compress above knee, allow up to ceiling
            #   2. Highlight denoise — smooth VAE noise in coded space BEFORE
            #      the steep log→linear conversion amplifies it exponentially
            #   3. Log-to-linear decompression — clean, full-range HDR output
            #
            # v2.3: Per-profile adaptive parameters. Each log curve has a
            # different steepness near code 1.0, so shoulder knee/ceiling and
            # denoise threshold/strength are tuned per-curve. See the
            # LOG_PROFILE_HDR_PARAMS table for the full rationale.
            #
            # Look up profile: source_space for log inputs, else default (LogC4-like)
            _profile_key = source_space if source_space in LOG_PROFILE_HDR_PARAMS else None
            _shoulder_knee, _shoulder_ceiling, _denoise_thresh, _denoise_str = (
                LOG_PROFILE_HDR_PARAMS.get(_profile_key, LOG_PROFILE_HDR_DEFAULT)
                if _profile_key else LOG_PROFILE_HDR_DEFAULT
            )
            logger.debug(
                f"[Radiance 4K Decode v2.3] Compress(Log) profile='{source_space}' → "
                f"shoulder({_shoulder_knee}/{_shoulder_ceiling}), "
                f"denoise({_denoise_thresh}/{_denoise_str})"
            )
            #
            # Stage 1: Soft log shoulder (replaces hard clamp)
            img = _soft_log_shoulder(img, knee=_shoulder_knee, ceiling=_shoulder_ceiling)
            #
            # Stage 2: Selective highlight denoise in log-coded space
            # This is the key to clean HDR: smooth VAE noise WHERE it matters
            # (highlights) BEFORE log→linear amplifies it.
            # Midtones and shadows pass through untouched.
            img = _denoise_log_highlights(img, threshold=_denoise_thresh, strength=_denoise_str)
        elif hdr_mode == "Soft Clip":
            # Tanh-encoded domain: max encoded value is exactly 1.0 (tanh → 1).
            # Clamp to [0, 1] is correct — anything outside is VAE noise.
            img = torch.clamp(img, 0.0, 1.0)
        elif hdr_mode == "Passthrough":
            # v2.2 FIX (BUG-50): Widened from [−0.05, 1.1] → [−0.05, 1.6].
            # Must match the widened encode clamp [−0.05, 1.5] plus a small
            # margin for tile-boundary reconstruction blur. Previous 1.1 ceiling
            # destroyed all HDR above 1.24 linear (~0.3 stops above SDR white).
            img = torch.clamp(img, -0.05, 1.6)
        else:
            # Clip (SDR): [0, 1] by definition.
            img = torch.clamp(img, 0.0, 1.0)

        # Step 1b: Decode from encoded space → linear
        if hdr_mode == "Compress (Log)":
            if _HAS_LOG_CURVES:
                # Stage 3 of v2.3 pipeline: Log-to-linear decompression.
                #
                # After soft shoulder + highlight denoise, the log-coded signal
                # is clean and bounded. The decompression now produces a smooth,
                # noise-free linear output across the full HDR range.
                #
                # FIX-2 (decode side): Decompress using the curve that matches
                # the encode-side source space, not always LogC4.
                log_to_linear = RadianceVAE4KEncode._get_log_to_linear()
                decompress_fn = log_to_linear.get(source_space, tensor_logc4_to_linear)
                img = decompress_fn(img)
            else:
                logger.warning(
                    "[Radiance 4K Decode] Compress (Log) hdr_mode selected but log "
                    "curves are unavailable — falling back to sRGB linearization. "
                    "Output will be incorrect. Install color_utils.py to fix this."
                )
                img = tensor_srgb_to_linear(img)
        elif hdr_mode == "Soft Clip":
            # v2.2 FIX (BUG-52): Invert tanh soft clip with raised atanh ceiling.
            #
            # Encode:  excess = (v − knee) / rng;  coded = knee + rng × tanh(excess)
            # Decode:  t = (coded − knee) / rng;    excess = atanh(t);  v = knee + excess × rng
            #
            # Previous atanh clamp of 0.9999 gave max recovery of:
            #   atanh(0.9999) ≈ 4.95 → sRGB 1.59 → linear 2.9
            # Raised to 0.999999:
            #   atanh(0.999999) ≈ 7.25 → sRGB 1.94 → linear 4.6
            # This nearly doubles the effective HDR ceiling.
            knee = 0.85
            rng = 1.0 - knee  # 0.15
            above = img > knee
            if above.any():
                recovered = img.clone()
                t = torch.clamp((img[above] - knee) / rng, 0.0, 0.999999)
                excess = torch.atanh(t)
                recovered[above] = knee + excess * rng
                img = recovered
            # v2.2 FIX (BUG-51): Use extended sRGB EOTF because recovered
            # values range from [0, ~1.94] — standard srgb_to_linear may clamp
            # inputs to [0, 1] internally, destroying all recovered highlights.
            img = _safe_srgb_to_linear_extended(img)
        elif hdr_mode == "Passthrough":
            # v2.2 FIX (BUG-51): Passthrough values can be up to 1.6 after
            # the widened clamp. Use extended EOTF to preserve HDR headroom.
            img = _safe_srgb_to_linear_extended(img)
        else:
            # Clip (SDR) / default: standard linearization.
            # Data is in [0, 1] from clamp above, so standard EOTF is safe.
            img = tensor_srgb_to_linear(img)

        # Step 2: Inverse tonemap (SDR→HDR expansion)
        if inverse_tonemap and hdr_mode != "Compress (Log)":
            target_peak = 2.0 ** (target_stops - 8.0)
            if target_peak > 1.0:
                # v2.2 FIX (BUG-48): Safe channel indexing for 4+ channels.
                luma = (
                    0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
                )
                threshold = 0.75
                a = (target_peak - 1.0) / ((1.0 - threshold) ** 2)
                luma_c = torch.clamp(luma, max=1.0)
                expanded = torch.where(
                    luma > threshold,
                    threshold + (luma_c - threshold) + a * (luma_c - threshold) ** 2,
                    luma,
                )
                slope = 1.0 + 2 * a * (1.0 - threshold)
                expanded = torch.where(
                    luma > 1.0, target_peak + (luma - 1.0) * slope, expanded
                )
                ratio = (expanded / (luma + 1e-8)).unsqueeze(-1)
                # Apply ratio to RGB only, leave any extra channels untouched
                if img.shape[-1] > 3:
                    img_rgb = img[..., :3] * ratio
                    img = torch.cat([img_rgb, img[..., 3:]], dim=-1)
                else:
                    img = img * ratio

        # Step 3: Exposure (linear domain)
        if exposure != 0.0:
            # Apply only to RGB channels, leave alpha/extra untouched
            if img.shape[-1] > 3:
                img_rgb = img[..., :3] * (2.0**exposure)
                img = torch.cat([img_rgb, img[..., 3:]], dim=-1)
            else:
                img = img * (2.0**exposure)

        # Step 4: Linear → target space
        #
        # v2.3.7 FINAL FIX (BUG-COMPRESS-LOG-ACEScg-OVEREXPOSE):
        #
        # ROOT CAUSE (confirmed from user screenshot):
        #   Settings: hdr_mode=Compress(Log), target_space=ACEScg, hdr_output=True,
        #             display_tonemap=ACES Filmic, source_space=sRGB
        #
        #   The v2.3.6 auto-tonemap block fired ONLY for display-referred spaces:
        #     _display_referred_space = target_space in ("sRGB","Raw") or
        #                               (target_space=="Linear" and not hdr_output)
        #   ACEScg is NOT in that set → display_tonemap="ACES Filmic" was silently
        #   ignored for ALL scene-referred and log target spaces.
        #   Log decompression produced scene-linear [0..55+].
        #   ACEScg matrix applied to [0..55+] = ACEScg [0..55+].
        #   hdr_output=True skipped all clamping.
        #   ComfyUI preview treats tensor as sRGB [0,1] → massively overexposed.
        #
        # THE FIX — two parts:
        #
        #   Part A: Tonemap fires for ALL target spaces when hdr_output=False.
        #     Old guard: hdr_mode==Compress(Log) AND _display_referred AND not hdr_output
        #     New guard: hdr_mode==Compress(Log) AND not hdr_output
        #     Tonemap runs BEFORE the target_space conversion, so ACEScg/Log/Rec2020
        #     matrix ops all receive properly-ranged [0,1] display-linear data.
        #
        #   Part B: hdr_output=True logs an explicit WARNING so the user knows
        #     their ComfyUI preview will always look overexposed in this mode.
        #     This is CORRECT for VFX pipelines (Nuke/Resolve) but confusing in
        #     ComfyUI's display.
        #
        # CORRECT SETTINGS GUIDE:
        #   Preview / SaveImage  → hdr_output=False,  target_space=sRGB
        #   Nuke / Resolve VFX   → hdr_output=True,   target_space=ACEScg/Linear
        #   Log delivery (EXR)   → hdr_output=True,   target_space=ARRI LogC4 etc.

        # ── v2.3.8 FINAL DESIGN: display_tonemap is the ONLY tonemap control ────
        #
        # PREVIOUS DESIGN (v2.3.7) — WRONG:
        #   Tonemap fired: hdr_mode==Compress(Log) AND not hdr_output
        #   Effect: hdr_output=True silently overrode display_tonemap=Reinhard.
        #   If user set display_tonemap=Reinhard but hdr_output=True (for VFX
        #   pipeline), the tonemap was skipped → overexposed output regardless.
        #   Also: display_tonemap=None was meaningless with hdr_output=False since
        #   the tonemap block ran anyway and "None" just fell through.
        #
        # NEW DESIGN (v2.3.8):
        #   display_tonemap is the SOLE control for tonemapping.
        #   hdr_output controls ONLY the final clamp and sRGB re-encode.
        #
        #   display_tonemap = "Reinhard"    → tonemap always (even hdr_output=True)
        #   display_tonemap = "ACES Filmic" → tonemap always
        #   display_tonemap = "None"        → raw scene-linear [0..55+], ALWAYS blown
        #                                     in ComfyUI preview regardless of hdr_output.
        #                                     Use ONLY when feeding a proper OCIO viewer.
        #
        #   hdr_output = False → after tonemap+target_space: clamp [0,1] + sRGB encode
        #   hdr_output = True  → after tonemap+target_space: NO clamp, raw float tensor
        #
        # CORRECT SETTINGS:
        # ┌─────────────────────┬──────────────────┬────────────┬─────────────────┐
        # │ Use case            │ display_tonemap  │ hdr_output │ target_space    │
        # ├─────────────────────┼──────────────────┼────────────┼─────────────────┤
        # │ Preview / SaveImage │ Reinhard / ACES  │ False      │ sRGB            │
        # │ Nuke (tonemapped)   │ Reinhard / ACES  │ True       │ ACEScg / Linear │
        # │ Nuke (raw HDR)      │ None             │ True       │ ACEScg / Linear │
        # │ Log delivery EXR    │ None             │ True       │ ARRI LogC4 etc  │
        # └─────────────────────┴──────────────────┴────────────┴─────────────────┘

        # Part B: warn when display_tonemap=None with Compress(Log) → blown output
        if hdr_mode == "Compress (Log)" and display_tonemap == "None":
            logger.warning(
                "[Radiance 4K Decode v2.3.8] display_tonemap='None' + Compress(Log): "
                "scene-linear values far above 1.0 pass through without tonemapping. "
                "ComfyUI preview/SaveImage will look overexposed. "
                "Set display_tonemap='ACES Filmic' for filmic contrast, or 'Reinhard' "
                "for a softer technical preview. Use 'None' only when feeding an OCIO-aware viewer."
            )

        # v2.3.9 FIX: Warn when a scene-referred target space is used with
        # hdr_output=False. In that case we now clamp to [0,1] (see final clamp
        # block below), which produces a valid but DISPLAY-REFERRED result — not
        # the wide-gamut / HDR tensor the user might expect from "ACEScg" output.
        # This warning steers them toward the correct settings.
        _SCENE_REFERRED_SPACES = {"ACEScg", "ACES 2065-1", "Rec.2020 Linear"} | set(EXTENDED_LOG_SPACES)
        if target_space in _SCENE_REFERRED_SPACES and not hdr_output:
            logger.warning(
                f"[Radiance 4K Decode v2.3.9] target_space='{target_space}' + hdr_output=False: "
                f"output will be clamped to [0,1] for ComfyUI display compatibility. "
                f"Values above 1.0 in '{target_space}' are discarded. "
                f"For true scene-referred / HDR output set hdr_output=True. "
                f"For log delivery (EXR) also set display_tonemap='None'."
            )

        # BUG 1 FIX: Capture scene-linear BEFORE display_tonemap fires.
        # RHDR sidecar must contain raw scene-linear float data so the Radiance
        # Viewer can apply its own display transform via WebGL shaders.
        # Capturing here (after log decompress + exposure, before tonemap+target_space)
        # gives the correct scene-linear Rec.709 tensor for the RHDR sidecar.
        # This is stored on self so decode() can retrieve it after this call.
        if hdr_mode == "Compress (Log)" and getattr(self, "_want_scene_linear", True):
            self._scene_linear_for_rhdr = img.detach().clone()

        # Part A: apply display_tonemap (independent of hdr_output)
        # Fires for ALL target spaces when hdr_mode=Compress(Log) and tonemap != None.
        # Runs BEFORE target_space conversion so ACEScg/Log/Rec2020 ops get [0,1] data.
        if hdr_mode == "Compress (Log)" and display_tonemap != "None":
            if display_tonemap == "Reinhard":
                # BUG 3 FIX: Luma-preserving Reinhard (replaces per-channel).
                # Per-channel Reinhard compresses R, G, B independently, shifting
                # the R:G:B ratio at high luminance — up to 33%+ hue drift for
                # saturated highlights (e.g. warm street lamps, fire).
                # Luma-based Reinhard compresses the overall luminance signal and
                # scales all channels by the same ratio, preserving chromaticity.
                # Result: correct hue at all luminances, proper desaturation rolloff.
                _luma = (
                    0.2126 * img[..., 0:1].clamp(min=0.0)
                    + 0.7152 * img[..., 1:2].clamp(min=0.0)
                    + 0.0722 * img[..., 2:3].clamp(min=0.0)
                )
                _luma_tm = _luma / (_luma + 1.0)          # compressed luma
                _ratio   = _luma_tm / (_luma + 1e-6)      # scale factor
                img = (img * _ratio).clamp(min=0.0)        # apply to all channels
            elif display_tonemap == "ACES Filmic":
                # Narkowicz ACES fit — filmic contrast, clips above ~10 lin
                v = img.clamp(min=0.0)
                img = (v * (2.51 * v + 0.03)) / (v * (2.43 * v + 0.59) + 0.14)
                img = img.clamp(0.0, 1.0)
            logger.debug(
                f"[Radiance 4K Decode v2.3.8] Compress(Log) tonemap='{display_tonemap}' "
                f"applied before '{target_space}' (hdr_output={hdr_output})."
            )
        # v2.3.9 FIX (BUG-OVEREXPOSE-INCOMPLETE):
        #
        # v2.3.5 fixed overexposure for target_space="Linear" + hdr_output=False
        # by adding tensor_linear_to_srgb re-encode + [0,1] clamp. But the same
        # fix was never applied to "ACEScg", "ACES 2065-1", and "Rec.2020 Linear".
        # With hdr_output=False those paths:
        #   1. Applied the Rec.709→AP1/AP0/Rec2020 matrix (leaving values
        #      potentially > 1.0 for Passthrough/SoftClip content or when
        #      display_tonemap="None" passes scene-linear [0..38+] through)
        #   2. Fell through with _display_referred_space=False → NO final clamp
        #   3. Emitted float values > 1.0 to the ComfyUI IMAGE socket
        #   → PreviewImage / SaveImage treat > 1.0 as blown white →
        #     massively overexposed output.
        #
        # Fix: when hdr_output=False, clamp ALL target spaces to [0,1] after
        # the target-space conversion. "Linear" still gets sRGB re-encode first
        # (v2.3.5 behaviour). For HDR delivery set hdr_output=True — that path
        # intentionally skips the clamp to preserve values > 1.0 for VFX tools.
        # 3.5.0: one gamut step for every target, from the latent's working
        # gamut to the target's own (camera gamut for the log targets), then
        # the transfer. Identity when they already agree, so a Compress(Log)
        # round trip back to its source log space stays exact.
        _dst_gamut = TARGET_SPACE_GAMUT.get(target_space, "Rec.709")
        _gm = _gamut_mat_t(working_gamut, _dst_gamut)
        if _gm is not None:
            img = _safe_matrix_transform(img, _gm.to(img.device))
        if target_space == "Linear":
            if not hdr_output:
                img = tensor_linear_to_srgb(img)
        elif target_space == "sRGB":
            img = tensor_linear_to_srgb(img)
        elif target_space in self.LOG_SPACES and _HAS_LOG_CURVES:
            converters = RadianceVAE4KEncode._get_linear_to_log()
            img = converters[target_space](img)
        elif target_space in self.LOG_SPACES and not _HAS_LOG_CURVES:
            logger.warning(
                f"[Radiance 4K Decode] Target space '{target_space}' is a log format "
                f"but color_utils is not installed — outputting LINEAR data instead."
            )
        # ACEScg / ACES 2065-1 / Rec.2020 Linear: the gamut step above is all.
        # Raw: linear light in Rec.709, no transfer.

        # Final clamp — hdr_output=False only.
        # Clamps every target space to [0,1] so ComfyUI nodes (PreviewImage,
        # SaveImage, downstream samplers) always receive valid display-referred
        # data. hdr_output=True intentionally skips this so scene-linear /
        # wide-gamut / log values > 1.0 are preserved for VFX pipelines.
        if not hdr_output:
            img = torch.clamp(img, 0.0, 1.0)

        # BUG-C FIX: Guarantee fp32 output regardless of which path was taken.
        return img.float()

    @staticmethod
    def _save_rhdr(
        img_np, output_dir: str, prefix: str = "radiance_4k", precision: str = "f16"
    ) -> Optional[str]:
        """Save image as .rhdr compressed float.

        Args:
            img_np:     (H, W, C) numpy array, any float dtype.
            output_dir: Target directory.
            prefix:     Filename stem.
            precision:  "f16" (fp16, default — smaller file, max value 65504) or
                        "f32" (fp32 — full 32-bit range, ~2× file size).
                        BUG-F/G FIX: Previously hard-coded fp16 with no operator
                        control. Production VFX content (sun, explosions, VDBs) can
                        exceed 65504 in linear light — use "f32" for those shots.
        """
        try:
            h, w = img_np.shape[:2]
            c = img_np.shape[2] if img_np.ndim == 3 else 1

            # V-9 FIX: Guard against uint16 header overflow
            if w > 65535 or h > 65535:
                logger.warning(
                    f"[Radiance 4K] RHDR export skipped: dimensions {w}×{h} "
                    f"exceed uint16 header limit (65535)"
                )
                return None

            unique_id = uuid.uuid4().hex[:12]
            filename = f"{prefix}_{unique_id}.rhdr"

            # V-7 FIX: Use safe_join to prevent path traversal via malicious prefix
            try:
                from .path_utils import safe_join
                filepath = safe_join(output_dir, filename)
            except ImportError:
                filepath = os.path.join(output_dir, filename)

            # BUG-F FIX: Support fp32 for scenes with linear values > 65504.
            # Header precision flag: 0 = fp16 (legacy), 1 = fp32.
            if precision == "f32":
                payload = img_np.astype("float32").tobytes()
                prec_flag = 1
            else:
                payload = img_np.astype("float16").tobytes()
                prec_flag = 0

            compressed = zlib.compress(payload, level=6)

            # BUG-G FIX: Precision flag stored in the reserved header byte (was 0).
            header = struct.pack("<4sHHHH", b"RHDR", w, h, c, prec_flag)

            with open(filepath, "wb") as f:
                f.write(header)
                f.write(compressed)

            ratio = len(compressed) / len(payload) * 100
            size_mb = (len(header) + len(compressed)) / (1024 * 1024)
            dtype_str = "fp32" if precision == "f32" else "fp16"
            logger.info(
                f"[Radiance 4K] RHDR export: {filename} "
                f"({w}×{h}, {dtype_str}, {size_mb:.1f}MB, {ratio:.0f}% ratio)"
            )
            return filename
        except Exception as e:
            logger.warning(f"[Radiance 4K] RHDR export failed: {e}")
            return None

    def decode(
        self,
        samples: Dict[str, Any],
        vae: Any,
        target_space: str = "sRGB",
        tile_size: str = "Auto",
        overlap: int = 128,
        exposure_adjust: float = 0.0,
        alpha: torch.Tensor = None,
        hdr_mode: str = "Clip (SDR)",
        display_tonemap: str = "ACES Filmic",
        inverse_tonemap: bool = False,
        target_stops: float = 12.0,
        crop_padding: str = "",
        export_rhdr: bool = False,
        rhdr_precision: str = "f16",
        source_space: str = "Linear",
        processing_mode: str = "sequential",
        force_hdr_decode: bool = False,
        hdr_output: bool = False,
        decode_noise_scale: float = 0.0,
        temporal_size: str = "Auto",
        temporal_overlap: int = 2,
    ) -> Tuple:
        """v2.3.5 TRUE-HDR: Universal decode with 32-bit HDR output support.

        v2.3.5 FIX (BUG-OVEREXPOSE):
          Root cause of overexposed output diagnosed and fixed.
          When target_space="Linear" (the previous default) + hdr_output=False:
            - _vae_output_to_target() applied sRGB→linear at step 1b (correct)
            - Then returned the scene-linear tensor as a ComfyUI IMAGE (wrong)
            - ComfyUI IMAGE convention is sRGB [0,1]; displaying linear [0,1]
              without gamma makes every value appear 1.1–1.5× brighter than correct.
              Result: overexposed image with hyper-saturated colors.
          Fix: target_space="Linear" + hdr_output=False now re-encodes to sRGB
          (identical result to target_space="sRGB"). hdr_output=True preserves
          true scene-linear output for VFX pipelines.
          Default target_space changed from "Linear" → "sRGB".

        v2.3 CHANGES vs v2.2:
          Compress (Log) HDR mode no longer hard-clamps at [0, 1]. Instead:
            1. _soft_log_shoulder() gently compresses above code 0.96, allowing
               legitimate super-whites up to code ~1.08 (~55 linear for LogC4)
            2. _denoise_log_highlights() smooths VAE noise in log-coded space
               BEFORE decompression — prevents exponential noise amplification
            3. Log-to-linear decompression produces clean, full-range HDR

        Parameters (still active):
          force_hdr_decode: When True, respects hdr_mode even without radiance_meta
                            (direct encode→decode, no sampler). Required for HDR output
                            with Compress(Log) or Soft Clip modes post-encoding.
          hdr_output:       When True, disables ALL final encoding and clamping so
                            the output tensor carries full 32-bit scene-linear values.
                            Combine with target_space="Linear" or "ACEScg" for HDR VFX.
                            When False (default), output is always re-encoded to sRGB
                            and clamped to [0,1] for ComfyUI IMAGE compatibility.
        """

        # v2.0 Feature 2: Auto-detect VAE factor
        vae_factor = detect_vae_factor(vae)
        latent_fmt = detect_latent_format(vae)

        # v3.1.1: When the video path decodes frame-by-frame it re-enters this
        # method recursively. The setup diagnostics below (force_hdr note, the
        # Compress(Log) source-space warning) describe decode *settings*, not the
        # frame — so they would otherwise print once for the outer video call and
        # again for every frame. `_quiet_diag` suppresses the per-frame repeats.
        _quiet_diag = getattr(self, "_frame_decode_active", False)

        # STALE-RHDR FIX: `_scene_linear_for_rhdr` is per-instance mutable state
        # and ComfyUI reuses node instances across executions. It used to be
        # cleared only inside the `if export_rhdr:` block, so a run that set it
        # and then exited without exporting left it populated, and the next
        # run's first video frame harvested the previous run's tensor. The
        # shape guard further down catches a resolution change but not a
        # same-resolution stale frame, which is the case that ships wrong
        # pixels. Clear it at the top of every top-level decode; recursive
        # per-frame calls must not clear it, because the outer loop reads what
        # they set.
        if not _quiet_diag:
            self._scene_linear_for_rhdr = None
            # 3.5.0: drop HDR coding keys a sampler carried through unchanged.
            samples, _meta_live = verify_radiance_meta(samples)
            # 3.5.0: the pre-tonemap scene-linear copy is only read by the
            # RHDR export; it used to be cloned on every Compress(Log) decode.
            self._want_scene_linear = bool(export_rhdr) and display_tonemap != "None"

        # v2.0 Feature 6: Read radiance_meta embedded by encode() if present
        radiance_meta = samples.get("radiance_meta", {})
        if radiance_meta:
            latent_fmt = radiance_meta.get("latent_format", latent_fmt)
            vae_factor = radiance_meta.get("vae_factor", vae_factor)

            # v2.2 FIX (BUG-47): Auto-read hdr_mode and source_space from
            # radiance_meta so encode→decode are always symmetric. Previously
            # the encode stored these in radiance_meta but the decode ignored
            # them, relying solely on the user's manual widget selection.
            meta_hdr = radiance_meta.get("hdr_mode")
            meta_src = radiance_meta.get("source_space")
            if meta_hdr and meta_hdr != hdr_mode:
                logger.info(
                    f"[Radiance 4K Decode v2.3] radiance_meta hdr_mode='{meta_hdr}' "
                    f"overrides widget hdr_mode='{hdr_mode}' for encode↔decode symmetry."
                )
                hdr_mode = meta_hdr
            if meta_src and meta_src != source_space:
                logger.info(
                    f"[Radiance 4K Decode v2.3] radiance_meta source_space='{meta_src}' "
                    f"overrides widget source_space='{source_space}' for encode↔decode symmetry."
                )
                source_space = meta_src
        else:
            # BUG-47 / HDR-FIX: No radiance_meta in latent dict.
            # Two legitimate scenarios reach this branch:
            #   A) Latent passed through a sampler — sampler strips custom dict keys.
            #      After diffusion, the VAE output is sRGB-like regardless of encode
            #      settings. Applying Log/SoftClip inversion here → wrong color output.
            #   B) Direct encode→decode with no sampler, but radiance_meta was stripped
            #      by a third-party node. User WANTS the HDR mode respected.
            #
            # Resolution: force_hdr_decode overrides the safety fallback.
            #   force_hdr_decode=False (default) → original BUG-47 safety: warn + Clip(SDR)
            #   force_hdr_decode=True  (HDR mode) → respect user hdr_mode, no override.
            if hdr_mode in ("Compress (Log)", "Soft Clip") and not force_hdr_decode:
                logger.warning(
                    f"[Radiance 4K Decode v2.3] hdr_mode='{hdr_mode}' selected but "
                    f"radiance_meta is absent — latent likely passed through a sampler "
                    f"which strips encode metadata. After diffusion sampling the VAE "
                    f"decodes to sRGB; Log/SoftClip inversion would corrupt color. "
                    f"Falling back to 'Clip (SDR)'. "
                    f"To preserve HDR for direct encode→decode (no sampler): enable "
                    f"force_hdr_decode=True on this node."
                )
                hdr_mode = "Clip (SDR)"
            elif hdr_mode in ("Compress (Log)", "Soft Clip") and force_hdr_decode and not _quiet_diag:
                logger.info(
                    f"[Radiance 4K Decode v2.3] force_hdr_decode=True: respecting "
                    f"hdr_mode='{hdr_mode}' despite absent radiance_meta. "
                    f"Ensure this is a direct encode→decode path (no sampler between "
                    f"encode and decode) for correct results."
                )

        latent = samples["samples"]

        # v2.3.7 FIX — Caveat 1: Validate source_space when hdr_mode=Compress(Log).
        # Compress(Log) inverts a specific log decompression curve selected by
        # source_space. If source_space is not a known log encoding (e.g. "Linear",
        # "sRGB", "ACEScg") the code silently falls back to LogC4 decompression,
        # producing wrong colors with no visible error.
        # Valid source spaces for Compress(Log): the 6 camera log formats.
        #
        # 3.5.0: not when the latent came from VAE Encode (HDR). Its encoder
        # carries a non-log source (Linear, ACEScg, ...) in LogC4 and records
        # that in radiance_meta, so decoding with LogC4 is the exact inverse.
        # The warning fired on every run of the recommended encode/decode
        # pair, telling users correct output was wrong.
        if hdr_mode == "Compress (Log)" and not radiance_meta:
            _valid_log_source = set(LOG_PROFILE_HDR_PARAMS.keys())
            if source_space not in _valid_log_source and not _quiet_diag:
                logger.warning(
                    f"[Radiance 4K Decode v2.3.7] hdr_mode='Compress (Log)' with "
                    f"source_space='{source_space}' and no encoder metadata: decoding "
                    f"with the ARRI LogC4 curve VAE Encode (HDR) uses for non-log "
                    f"sources. That is correct for a latent from VAE Encode (HDR); "
                    f"for anything else pick the log space it was encoded in "
                    f"({', '.join(sorted(_valid_log_source))}), or 'Clip (SDR)' for "
                    f"ordinary diffusion output."
                )

        # 3.5.0: gamut of the linear light inside the latent. Compress(Log)
        # keeps a log source's camera gamut (exact round trip); everything
        # else is Rec.709.
        if hdr_mode == "Compress (Log)":
            working_gamut = ((radiance_meta or {}).get("working_gamut")
                             or LOG_SPACE_GAMUT.get(source_space, "Rec.709"))
        else:
            working_gamut = "Rec.709"

        # v2.1 Video Support: Check for 5D video latent (B, C, F, H, W)
        is_video = latent.ndim == 5

        # Check if the VAE natively supports 3D latents (e.g., Wan Video, Cosmos)
        is_3d_vae = False
        if hasattr(vae, "latent_dim") and getattr(vae, "latent_dim") == 3:
            is_3d_vae = True

        if is_video and not is_3d_vae:
            # v2.3.1 FIX (V-B15): renamed frame-count local from `F` to
            # `num_frames`. `F` is torch.nn.functional at module scope; the
            # in-function re-bind shadowed it for every later call in this
            # function (e.g. `F.interpolate` in the alpha-resize branch).
            # Harmless today because the branch early-returns, but a silent
            # time-bomb for any future refactor that removes the early return.
            B, C, num_frames, H, W = latent.shape
            logger.info(
                f"[Radiance 4K Decode v2.3] Video Latent: {W}×{H} "
                f"({B} batches, {num_frames} frames), format={latent_fmt}"
            )
            # STREAM-FIX: one pre-allocated clip buffer written frame by frame,
            # replacing a list of decoded frames plus a closing torch.cat. At
            # 4K fp32 RGB a frame is ~106 MB, so the old pair cost ~53 GB of
            # list plus ~53 GB for the cat on a 500-frame clip, and with
            # Compress(Log) + export_rhdr the scene-linear list doubled it
            # again. Now the returned batch is the only clip-sized tensor alive
            # and the RHDR sidecars are written as each frame lands, so peak is
            # (returned batch + one frame) and clip length is bounded by disk.
            all_frames = None
            _frame_cursor = 0
            meta_str = None
            # BUG 1 FIX (video path): use scene-linear per frame for RHDR.
            # Each recursive self.decode() sets self._scene_linear_for_rhdr BEFORE
            # its tonemap fires. Harvest it immediately after each frame completes,
            # before the next frame overwrites it.
            _rhdr_dir = folder_paths.get_temp_directory() if export_rhdr else None
            rhdr_filenames = []
            _rhdr_used_scene_linear = 0
            frame_alphas = None
            if alpha is not None:
                if alpha.shape[0] == B * num_frames:
                    # Encode lays alpha out batch-major (b0f0, b0f1, ...);
                    # frame fi needs every batch item's frame fi. chunk()
                    # took consecutive rows, which is only right for B == 1.
                    _a = alpha.reshape(B, num_frames, *alpha.shape[1:])
                    frame_alphas = [_a[:, fi] for fi in range(num_frames)]
                else:
                    frame_alphas = [alpha] * num_frames

            # Suppress duplicate per-frame setup diagnostics in the recursive
            # decode() calls below (see `_quiet_diag`). Reset in `finally` so a
            # failed frame never leaves the flag stuck on.
            self._frame_decode_active = True
            try:
                for fi in range(num_frames):
                    frame_latent = latent[:, :, fi, ...]
                    frame_samples = dict(samples)
                    frame_samples["samples"] = frame_latent

                    frame_alpha = frame_alphas[fi] if frame_alphas else None

                    decoded_frame, meta_str, fmt = self.decode(
                        samples=frame_samples,
                        vae=vae,
                        target_space=target_space,
                        tile_size=tile_size,
                        overlap=overlap,
                        exposure_adjust=exposure_adjust,
                        alpha=frame_alpha,
                        hdr_mode=hdr_mode,
                        display_tonemap=display_tonemap,
                        inverse_tonemap=inverse_tonemap,
                        target_stops=target_stops,
                        crop_padding=crop_padding,
                        export_rhdr=False,   # RHDR handled on concatenated output below
                        rhdr_precision=rhdr_precision,
                        source_space=source_space,
                        processing_mode=processing_mode,
                        force_hdr_decode=force_hdr_decode,
                        hdr_output=hdr_output,
                        decode_noise_scale=decode_noise_scale,
                    )
                    # Collect scene-linear for this frame (set by _vae_output_to_target)
                    _sl = getattr(self, "_scene_linear_for_rhdr", None)
                    self._scene_linear_for_rhdr = None  # clear immediately

                    if all_frames is None:
                        # Clip buffer shaped from the first decoded frame so the
                        # channel count comes from the VAE, never assumed. Lives
                        # on the intermediate device for the same reason encode's
                        # does: clip length must not be a VRAM budget.
                        _per = decoded_frame.shape[0]
                        all_frames = torch.empty(
                            (_per * num_frames,) + tuple(decoded_frame.shape[1:]),
                            dtype=decoded_frame.dtype,
                            device=_intermediate_device(),
                        )
                    _n = decoded_frame.shape[0]
                    all_frames[_frame_cursor:_frame_cursor + _n] = decoded_frame.to(
                        all_frames.device)

                    # v2.1 FIX (BUG-46) + BUG 1 FIX (video): RHDR sidecars are
                    # written here, as the frame lands, rather than from a
                    # second full-clip list after the loop. The source choice is
                    # the same one the post-loop block used to make in bulk:
                    # Compress(Log) with an active tonemap wants the pre-tonemap
                    # scene-linear capture, everything else wants the decoded
                    # frame. Deciding per frame rather than per clip is what
                    # removes the clip-length scene-linear accumulator.
                    if export_rhdr:
                        _use_sl = (
                            hdr_mode == "Compress (Log)"
                            and display_tonemap != "None"
                            and _sl is not None
                        )
                        _rhdr_frame = _sl if _use_sl else decoded_frame
                        if _use_sl:
                            _rhdr_used_scene_linear += 1
                        for _sub in range(_rhdr_frame.shape[0]):
                            frame_np = _rhdr_frame[_sub, ..., :3].cpu().numpy()
                            while frame_np.ndim > 3:
                                frame_np = frame_np[0]
                            fname = self._save_rhdr(
                                frame_np, _rhdr_dir,
                                prefix=f"radiance_4k_f{_frame_cursor + _sub:04d}",
                                precision=rhdr_precision,
                            )
                            if fname:
                                rhdr_filenames.append(fname)
                        del _rhdr_frame

                    _frame_cursor += _n
                    # Drop this frame's references before the next decode runs,
                    # so only one frame's worth of working memory is alive.
                    del decoded_frame, _sl, frame_latent, frame_samples
            finally:
                self._frame_decode_active = False

            if all_frames is None:
                # NUM-FRAMES-0 FIX: `meta_str` is bound inside the loop, so a
                # zero-frame latent used to raise NameError below (after
                # torch.cat([]) raised first). Return an empty batch shaped
                # from the latent instead.
                all_frames = torch.empty(
                    (0, H * vae_factor, W * vae_factor, 3),
                    dtype=torch.float32, device=_intermediate_device(),
                )
            elif _frame_cursor != all_frames.shape[0]:
                # Every frame_latent has identical shape, so this cannot happen
                # with a well-behaved VAE; narrow rather than return a tail of
                # uninitialised memory if one ever returns a ragged batch.
                all_frames = all_frames[:_frame_cursor]

            try:
                meta_json = json.loads(meta_str)
            except (json.JSONDecodeError, TypeError, ValueError):
                meta_json = {}
            meta_json["video"] = True
            meta_json["frames"] = num_frames
            if export_rhdr:
                meta_json["rhdr_scene_linear_frames"] = _rhdr_used_scene_linear
            if rhdr_filenames:
                meta_json["rhdr_export"] = rhdr_filenames
                meta_json["rhdr_precision"] = rhdr_precision

            return (all_frames, json.dumps(meta_json, indent=2), latent_fmt)

        b = latent.shape[0]
        lat_h, lat_w = latent.shape[-2], latent.shape[-1]
        pix_h, pix_w = lat_h * vae_factor, lat_w * vae_factor

        # Spatial tile size in pixel space (must be aligned to vae_factor).
        # ALBABIT-FIX: resolved up front so the temporal Auto-size below can
        # be computed jointly with it. Spatial and temporal chunking share
        # one VRAM budget rather than being sized independently.
        pad_multiple = max(vae_factor, 8)
        if tile_size == "Auto":
            ts_px = TileEngine.get_optimal_tile_size(pix_h, pix_w)
        else:
            ts_px = int(tile_size)
        ts_px = (ts_px // pad_multiple) * pad_multiple

        overlap = min(overlap, ts_px // 2)
        overlap = (overlap // 8) * 8
        overlap = max(16, overlap)

        # Same defensive getattr/callable/try pattern as detect_vae_factor()
        # above: third-party VAE-like objects aren't guaranteed to implement
        # this comfy.sd.VAE convenience method, or to implement it safely.
        _temporal_compression_fn = getattr(vae, "temporal_compression_decode", None)
        _temporal_compression = None
        if callable(_temporal_compression_fn):
            try:
                _temporal_compression = _temporal_compression_fn()
            except Exception:
                _temporal_compression = None
        _temporal_compression = _temporal_compression or 8

        logger.info(
            f"[Radiance 4K Decode v2.3] Latent: {lat_w}x{lat_h} -> "
            f"Output: {pix_w}x{pix_h} ({b} frames, factor={vae_factor}, fmt={latent_fmt})"
        )

        target_device = comfy.model_management.get_torch_device()
        if latent.device != target_device:
            latent = latent.to(target_device)
            samples = dict(samples)
            samples["samples"] = latent

        # v2.4: Decode-time noise injection (LTX LumiVid, arxiv 2604.11788)
        # ─────────────────────────────────────────────────────────────────────
        # A small lerp toward Gaussian noise breaks up coherent VAE decoder
        # grid artifacts in flat highlight areas.  Only active for Compress(Log)
        # where the subsequent log→linear step would exponentially amplify any
        # correlated decoder artifact in the highlight band.
        #
        # Formula: noised = (1 - scale) * latent + scale * randn_like(latent)
        # At scale=0.025 the latent moves 2.5% toward pure noise — perceptually
        # invisible but sufficient to de-correlate decoder grid patterns.
        #
        # User-supplied decode_noise_scale takes precedence.  If 0.0 (the default)
        # the feature is off entirely, preserving existing workflow behaviour.
        _effective_noise_scale = decode_noise_scale
        if _effective_noise_scale > 0.0 and hdr_mode == "Compress (Log)":
            noise = torch.randn_like(latent)
            latent = (1.0 - _effective_noise_scale) * latent + _effective_noise_scale * noise
            samples = dict(samples)
            samples["samples"] = latent
            logger.debug(
                f"[Radiance 4K Decode v2.4] decode_noise_scale={_effective_noise_scale:.4f} "
                f"applied to latent (profile='{source_space}')."
            )

        pbar = comfy.utils.ProgressBar(100)

        decoded_video_frames = None  # Set when 3D VAE returns 5D output
        _temporal_metadata = None  # Set when the decode_tiled() path resolves a temporal decision
        with torch.no_grad():
            if latent.ndim == 5:
                # ALBABIT-FIX: integrated spatial+temporal tiling via comfy's
                # vae.decode_tiled(), replacing the old stacked tiler that
                # re-decoded a full spatial tile per temporal chunk. This is
                # what let LTX-2.5 crash at settings the native "VAE Decode
                # (Tiled)" node handles fine. See project_radiance_vae_tiling_seams.
                lat_T = latent.shape[2]
                temporal_lat = _resolve_temporal_frames(temporal_size, ts_px, _temporal_compression)
                needs_spatial_tiling = not (pix_h <= ts_px and pix_w <= ts_px)
                # temporal_lat==0 means explicitly disabled (see
                # _resolve_temporal_frames); must not be read as "chunk size
                # zero", which would make lat_T > temporal_lat trivially true.
                needs_temporal_chunking = temporal_lat > 0 and lat_T > temporal_lat
                _temporal_metadata = {
                    "temporal_size": temporal_lat,
                    "temporal_chunking": needs_temporal_chunking,
                }

                if not needs_spatial_tiling and not needs_temporal_chunking:
                    img = vae.decode(latent).float()
                else:
                    lat_tile = ts_px // vae_factor
                    lat_overlap = overlap // vae_factor
                    # tile_t/overlap_t=None (omitted by comfy's decode_tiled
                    # dispatcher) means "no temporal limit"; correct when
                    # only spatial tiling is actually needed.
                    tile_t = temporal_lat if needs_temporal_chunking else None
                    t_overlap_lat = min(temporal_overlap, temporal_lat // 2) if needs_temporal_chunking else None
                    logger.info(
                        f"[Radiance 4K Decode] {pix_w}x{pix_h} x {lat_T}f -> "
                        f"spatial tile={lat_tile}lat (needed={needs_spatial_tiling}), "
                        f"temporal chunk={temporal_lat}lat (needed={needs_temporal_chunking})"
                    )
                    img = vae.decode_tiled(
                        latent, tile_x=lat_tile, tile_y=lat_tile, overlap=lat_overlap,
                        tile_t=tile_t, overlap_t=t_overlap_lat,
                    ).float()

                if img.ndim == 5:
                    _B5, _F5, _H5, _W5, _C5 = img.shape
                    decoded_video_frames = _F5
                    img = img.reshape(_B5 * _F5, _H5, _W5, _C5)
                pbar.update_absolute(100, 100)
            elif pix_h <= ts_px and pix_w <= ts_px:
                img = vae.decode(latent).float()
                # FIX-4: 3D (temporal) VAEs return (B, F, H, W, C) — reshape to
                # (B*F, H, W, C) so downstream code handles it as a frame batch.
                if img.ndim == 5:
                    _B5, _F5, _H5, _W5, _C5 = img.shape
                    decoded_video_frames = _F5
                    img = img.reshape(_B5 * _F5, _H5, _W5, _C5)
                pbar.update_absolute(100, 100)
            else:
                img, _tiled_video_frames = self._tiled_decode(
                    samples, vae, ts_px, overlap, pbar,
                    vae_factor=vae_factor,
                )
                if _tiled_video_frames is not None:
                    decoded_video_frames = _tiled_video_frames

        img = img.float()

        # v4.5: source_space auto-detect from tensor statistics
        # Most diffusion VAEs decode to sRGB-like [0,1]. If the user left
        # source_space="Linear" but the tensor looks sRGB (p50 luma ~0.35–0.55
        # after diffusion), applying log→linear transforms will be wrong.
        # Auto-detect runs only when source_space=="Linear" AND hdr_mode is
        # NOT Compress(Log) (log mode requires explicit source_space to invert
        # the right curve). We suggest a warning; we do NOT silently override
        # because for encode→decode pipelines "Linear" may be correct.
        if source_space == "Linear" and hdr_mode not in ("Compress (Log)",) and not _quiet_diag:
            try:
                _sample_px = img[0].reshape(-1, img.shape[-1]) if img.ndim == 4 else img.reshape(-1, img.shape[-1])
                _step = max(1, _sample_px.shape[0] // 50_000)
                _luma = (0.2126*_sample_px[::_step, 0] + 0.7152*_sample_px[::_step, 1] + 0.0722*_sample_px[::_step, 2])
                _p50 = float(_luma.float().cpu().median())
                # Heuristic: scene-linear midgrey (0.18) → p50 ≈ 0.10–0.25
                #             sRGB midgrey (0.46)          → p50 ≈ 0.35–0.60
                #             Very dark diffusion output   → p50 < 0.10 (inconclusive)
                if 0.30 <= _p50 <= 0.70:
                    logger.warning(
                        f"[Radiance 4K Decode v4.5] source_space='Linear' but tensor "
                        f"p50 luma={_p50:.3f} suggests sRGB-encoded output (typical for "
                        f"diffusion models: SDXL/FLUX/SD1.5). "
                        f"If the decode looks too bright, change source_space to 'sRGB'. "
                        f"This warning is suppressed when hdr_mode='Compress (Log)'."
                    )
            except Exception as _exc:
                logger.debug(
                    "[Radiance] decode(): ignoring %s from `_sample_px = img[0].reshape(-1, img.shape[-1]) if img.ndim =…`: %s",
                    type(_exc).__name__, _exc,
                )

        # v2.3: Pre-transform NaN/Inf guard — mode-aware, unconditional.
        # Previously gated on source_space == "Linear", which let NaN values
        # from non-Linear VAE outputs (e.g. sRGB, LogC4) reach _vae_output_to_target
        # unguarded, where log/gamma functions produce further NaN explosions.
        if torch.isnan(img).any() or torch.isinf(img).any():
            logger.warning("[Radiance 4K Decode v2.3] VAE produced NaN/Inf — sanitizing")
            if hdr_mode == "Passthrough":
                img = torch.nan_to_num(img, nan=0.0, posinf=1.5, neginf=-0.05)
            else:
                img = torch.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)

        img = self._vae_output_to_target(
            img, target_space, hdr_mode,
            exposure_adjust, inverse_tonemap, target_stops,
            source_space=source_space,
            hdr_output=hdr_output,
            display_tonemap=display_tonemap,
            working_gamut=working_gamut,
        )

        # v2.3 FIX (BUG-B enhanced): Guard for NaN/Inf *introduced by* the color
        # transform. With v2.3's soft shoulder + denoise pipeline, this should be
        # extremely rare — but we keep the guard for safety:
        #
        # Possible remaining sources of post-transform NaN/Inf:
        #   1. Log decompression of noise that survived the soft shoulder
        #      (values very near ceiling in steep log region → large linear)
        #   2. Inverse tonemap with high target_stops amplifying noise to Inf
        #   3. Out-of-gamut matrix transform producing small negatives → NaN in
        #      downstream sRGB gamma (already handled by _safe_srgb_to_linear_extended)
        #
        # HDR-FIX: posinf cap is mode-aware:
        #   hdr_output=False: cap at 65504 (fp16 max — safe for RHDR export)
        #   hdr_output=True:  cap at fp32 max (~3.4e38) — only true Inf is replaced,
        #                     preserving all valid HDR scene-linear values.
        if torch.isnan(img).any() or torch.isinf(img).any():
            nan_count = int(torch.isnan(img).sum().item())
            inf_count = int(torch.isinf(img).sum().item())
            logger.warning(
                f"[Radiance 4K Decode v2.3] NaN/Inf after color transform — "
                f"sanitizing ({nan_count} NaN, {inf_count} Inf). "
                f"Check VAE reconstruction quality and hdr_mode/source_space pairing."
            )
            posinf_val = 3.4e38 if hdr_output else 65504.0
            img = torch.nan_to_num(img, nan=0.0, posinf=posinf_val, neginf=0.0)

        # v2.0 Feature 6: Auto-crop — prefer radiance_meta, fall back to crop_padding string
        pad_h, pad_w = 0, 0
        if radiance_meta:
            pad_h = radiance_meta.get("pad_h", 0)
            pad_w = radiance_meta.get("pad_w", 0)
        elif crop_padding:
            try:
                meta = json.loads(crop_padding)
                pad_h = meta.get("pad_h", 0)
                pad_w = meta.get("pad_w", 0)
            except (json.JSONDecodeError, TypeError) as _exc:
                logger.debug(
                    "[Radiance] decode(): ignoring %s from `meta = json.loads(crop_padding)`: %s",
                    type(_exc).__name__, _exc,
                )

        if pad_h > 0 or pad_w > 0:
            crop_h = img.shape[1] - pad_h
            crop_w = img.shape[2] - pad_w
            img = img[:, :crop_h, :crop_w, :]
            logger.info(f"[Radiance 4K v2.3] Cropped padding: {pad_h}h, {pad_w}w -> {crop_w}x{crop_h}")

        img = _restore_alpha_channel(img, alpha)

        # Export .rhdr — one sidecar per frame for video, single file for stills
        # BUG 1 FIX: RHDR must be scene-linear float data (Radiance Viewer uses it
        # for proper HDR display via WebGL shaders). Previously saved from the
        # post-processed img (after tonemap + target_space), which gave wrong data
        # when display_tonemap=Reinhard: the viewer would load tonemapped sRGB and
        # apply another display transform on top → double-tonemapped output.
        # Fix: use self._scene_linear_for_rhdr (captured in _vae_output_to_target
        # right before the tonemap step) when Compress(Log) + tonemap was active.
        # For non-Compress(Log) modes and display_tonemap=None, the scene-linear
        # capture is not needed (img is already in the correct state for RHDR).
        rhdr_filenames = []
        if export_rhdr:
            output_dir = folder_paths.get_temp_directory()
            # Select correct source for RHDR:
            # - Compress(Log) + tonemap fired → use scene_linear_for_rhdr (pre-tonemap)
            # - All other cases → use img (either scene-linear or display-referred as appropriate)
            _rhdr_source = img
            _scene_lin = getattr(self, '_scene_linear_for_rhdr', None)
            if (
                hdr_mode == "Compress (Log)"
                and display_tonemap != "None"
                and _scene_lin is not None
                and _scene_lin.shape[:3] == img.shape[:3]
            ):
                _rhdr_source = _scene_lin
                logger.debug(
                    "[Radiance 4K Decode v2.3.8] RHDR saved from scene-linear "
                    "(pre-tonemap) — ensures Radiance Viewer receives correct HDR data."
                )
            num_frames = _rhdr_source.shape[0]
            for fi in range(num_frames):
                frame_np = _rhdr_source[fi, ..., :3].cpu().numpy()
                while frame_np.ndim > 3:
                    frame_np = frame_np[0]
                prefix = f"radiance_4k_f{fi:04d}" if num_frames > 1 else "radiance_4k"
                fname = self._save_rhdr(
                    frame_np, output_dir, prefix=prefix, precision=rhdr_precision
                )
                if fname:
                    rhdr_filenames.append(fname)
            # Clear the cache after use to avoid stale data on next call
            self._scene_linear_for_rhdr = None

        final_h, final_w = img.shape[1], img.shape[2]
        # BUG 2 FIX: report actual output colorspace in metadata.
        # When target_space=Linear + hdr_output=False, the code re-encodes to sRGB
        # for ComfyUI IMAGE compatibility. Metadata must reflect the actual tensor
        # colorspace — downstream tools (Nuke, Resolve) read this for color management.
        _reported_space = target_space
        if target_space == "Linear" and not hdr_output:
            _reported_space = "sRGB (re-encoded from Linear)"
        metadata = {
            "node": "RadianceVAE4KDecode",
            "version": "3.0.0",
            "resolution": f"{final_w}×{final_h}",
            "tile_size": ts_px, "overlap": overlap,
            "target_space": _reported_space, "hdr_mode": hdr_mode,
            "display_tonemap": display_tonemap if hdr_mode == "Compress (Log)" else "N/A",
            "exposure_adjust": exposure_adjust,
            "inverse_tonemap": inverse_tonemap,
            "target_stops": target_stops if inverse_tonemap else "N/A",
            "alpha_restored": alpha is not None,
            "vae_factor": vae_factor, "latent_format": latent_fmt,
            "processing_mode": processing_mode,
        }
        # Video metadata: populated for both 3D-native VAE path and frame-loop path
        if decoded_video_frames is not None:
            metadata["video"] = True
            metadata["frames"] = decoded_video_frames
        elif img.shape[0] > 1:
            # Batch of images (e.g. from frame-loop path arriving here as concatenated frames)
            metadata["video"] = True
            metadata["frames"] = img.shape[0]
        if _temporal_metadata is not None:
            metadata.update(_temporal_metadata)
        if rhdr_filenames:
            metadata["rhdr_export"] = rhdr_filenames[0] if len(rhdr_filenames) == 1 else rhdr_filenames
            metadata["rhdr_precision"] = rhdr_precision

        # ComfyUI and most downstream nodes expect image tensors on CPU.
        # VAE.decode usually handles this; keep it explicit for every path.
        return (img.cpu(), json.dumps(metadata, indent=2), latent_fmt)



# ═══════════════════════════════════════════════════════════════════════════════
#                    4K DIRECT ENCODE → DECODE (ROUNDTRIP)
# ═══════════════════════════════════════════════════════════════════════════════


class RadianceVAE4KRoundtrip:
    """
    Single-node 4K VAE roundtrip: Encode → Decode in one step.

    Use case: Direct 4K export without diffusion — tests VAE reconstruction
    quality, applies color transforms, or serves as a high-quality resampler.

    Also useful for preparing reference frames: encode 4K plate, decode to
    see what the VAE preserves vs destroys, then adjust workflow accordingly.
    """

    SOURCE_SPACES = RadianceVAE4KEncode.SOURCE_SPACES
    TARGET_SPACES = RadianceVAE4KDecode.TARGET_SPACES

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "pixels": ("IMAGE",),
                "vae": ("VAE",),
            },
            "optional": {
                "source_space": (cls.SOURCE_SPACES, {"default": "Linear"}),
                "target_space": (cls.TARGET_SPACES, {"default": "Linear"}),
                "tile_size": (
                    ["Auto", "512", "768", "1024", "1280", "1536"],
                    {"default": "Auto"},
                ),
                "overlap": ("INT", {"default": 128, "min": 32, "max": 256, "step": 16,
                    "tooltip": "Overlap in pixels between tiles during VAE tiling. Larger overlap reduces seam artifacts.",
                }),
                "exposure": (
                    "FLOAT",
                    {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1},
                ),
                "hdr_mode": (
                    ["Clip (SDR)", "Soft Clip", "Compress (Log)", "Passthrough"],
                    {"default": "Soft Clip"},
                ),
                "display_tonemap": (
                    ["Reinhard", "ACES Filmic", "None"],
                    {
                        "default": "ACES Filmic",
                        "tooltip": (
                            "Tonemap for Compress(Log) + display output. "
                            "Independent of hdr_output. "
                            "'Reinhard': smooth rolloff. 'ACES Filmic': filmic contrast. "
                            "'None': raw scene-linear HDR, overexposed in normal preview."
                        ),
                    },
                ),
                "export_rhdr": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Export .rhdr for Radiance Viewer."},
                ),
                "rhdr_precision": (
                    ["f16", "f32"],
                    {
                        "default": "f16",
                        "tooltip": (
                            "RHDR export precision. "
                            "'f16': smaller files, values capped at 65504. "
                            "'f32': full 32-bit range — use for extreme HDR content."
                        ),
                    },
                ),
                # HDR-FIX: Roundtrip is always direct encode→decode (no sampler),
                # so force_hdr_decode and hdr_output default to True here.
                "hdr_output": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "32-BIT HDR OUTPUT — preserves full scene-linear range "
                            "(values >1.0 and <0.0) in the output tensor. "
                            "Default False — matches decode node and works correctly "
                            "with SaveImage/PreviewImage. "
                            "Set True only when feeding VFX pipelines that expect "
                            "scene-linear tensors."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "LATENT", "STRING")
    RETURN_NAMES = ("image", "latent", "metadata")
    FUNCTION = "roundtrip"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    DESCRIPTION = "4K VAE roundtrip in one node: encode → decode with tiling. Test reconstruction or apply color transforms."

    def roundtrip(
        self,
        pixels: torch.Tensor,
        vae: Any,
        source_space: str = "Linear",
        target_space: str = "Linear",
        tile_size: str = "Auto",
        overlap: int = 128,
        exposure: float = 0.0,
        hdr_mode: str = "Soft Clip",
        display_tonemap: str = "ACES Filmic",
        export_rhdr: bool = True,
        rhdr_precision: str = "f16",
        hdr_output: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any], str]:

        # V-4 FIX: Handle both 4D images (B,H,W,C) and 5D video (B,F,H,W,C).
        # Previously both ternary branches returned pixels.shape, crashing on 5D.
        if pixels.ndim == 5:
            b, f, h, w, c = pixels.shape
            logger.info(f"[Radiance 4K Roundtrip v2.3] Video: {w}\u00D7{h} ({f} frames), {source_space} \u2192 {target_space}")
        else:
            b, h, w, c = pixels.shape
            logger.info(f"[Radiance 4K Roundtrip v2.3] {w}\u00D7{h}, {source_space} \u2192 {target_space}")

        # Encode (returns 5-tuple in v2.0)
        encoder = RadianceVAE4KEncode()
        latent, alpha, enc_meta, latent_fmt, _quality = encoder.encode(
            pixels, vae, source_space, tile_size, overlap, exposure, "Preserve", hdr_mode,
        )

        # Map encode hdr_mode to decode hdr_mode
        decode_hdr = {
            "Clip (SDR)": "Clip (SDR)",
            "Soft Clip": "Soft Clip",
            "Compress (Log)": "Compress (Log)",
            "Passthrough": "Passthrough",
        }.get(hdr_mode, "Clip (SDR)")

        # Decode — reverse encode exposure so VAE roundtrip is neutral (returns 3-tuple in v2.0)
        decoder = RadianceVAE4KDecode()
        img, dec_meta, _fmt = decoder.decode(
            latent, vae, target_space, tile_size, overlap,
            exposure_adjust=-exposure, alpha=alpha,
            hdr_mode=decode_hdr,
            display_tonemap=display_tonemap,
            crop_padding=enc_meta,
            export_rhdr=export_rhdr,
            rhdr_precision=rhdr_precision,
            source_space=source_space,
            force_hdr_decode=True,   # Roundtrip is always direct — no sampler in path
            hdr_output=hdr_output,
        )

        # Combined metadata
        combined_meta = {
            "node": "RadianceVAE4KRoundtrip",
            "version": "3.0",
            "latent_format": latent_fmt,
            "encode": json.loads(enc_meta),
            "decode": json.loads(dec_meta),
        }

        return (img, latent, json.dumps(combined_meta, indent=2))
