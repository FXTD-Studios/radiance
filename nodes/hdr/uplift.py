"""
nodes_hdr_uplift.py — Radiance SDR → HDR AI Uplift Pipeline
════════════════════════════════════════════════════════════════════════════════

Three nodes that together form the complete Radiance SDR→HDR reconstruction
pipeline — the core mission of Radiance:

  1. RadianceClipDetector
     Identifies clipped / saturated regions in any SDR image.
     Outputs a soft mask that marks areas where HDR highlight data has been
     lost and needs to be reconstructed by the diffusion model.

  2. RadianceSDRToHDRPrepare
     Prepares an SDR image for AI highlight reconstruction.
     • Applies inverse EOTF (gamma decode → scene-linear)
     • Runs soft-knee compression for VAE compatibility
     • Outputs the VAE-ready image and an inpainting mask (dilated clip mask)
     • Also outputs per-channel norm stats so the decoder can invert them

  3. RadianceHDRHighlightComposite
     Post-generation compositing step.
     • Blends AI-generated HDR highlights back onto the original SDR image
     • Keeps non-clipped regions pixel-perfect from the original
     • Only the reconstructed highlight areas come from the model output
     • Outputs scene-linear HDR ready for RadianceHDRDecoder or EXR save

Pipeline wiring
────────────────
  [LoadImage (SDR)]
       │
       ▼
  [RadianceClipDetector]──clip_mask──┐
       │ clip_fraction               │
       │ visualization               │
       ▼                             │
  [RadianceSDRToHDRPrepare]          │
       │ image (VAE-ready)           │
       │ mask (for inpainting)       │
       │ stats                       │
       ▼                             │
  [VAEEncode]                        │
       │ latent                      │
       ▼                             │
  [KSampler + LoRA model]            │
       │ latent                      │
       ▼                             │
  [VAEDecode]                        │
       │ image (AI HDR output)       │
       ▼                             │
  [RadianceHDRDecoder]               │
       │ image (scene-linear)        │
       ▼                             │
  [RadianceHDRHighlightComposite] ◀──┘ clip_mask
       │ image (final HDR)
       ▼
  [SaveEXR / PreviewHDR]
"""

from __future__ import annotations

import logging
import math
from typing import Tuple

import torch
import torch.nn.functional as F

# Canonical transfer-function decoders — imported from color/transfer.py rather
# than redefined here.  Previously this file contained its own copies of the
# sRGB and Rec.709 EOTF inverses; those are now removed.
from radiance.color.transfer import (
    tensor_srgb_to_linear  as _inverse_eotf_srgb,
    tensor_logc3_to_linear as _inverse_eotf_logc3,
)

# Soft-knee compression is the canonical implementation in color/ops.
# Kept as a module-level alias so internal call-sites remain unchanged.
from radiance.color.ops import soft_knee_compress as _soft_knee_compress

logger      = logging.getLogger("radiance.hdr_uplift")
diag_logger = logging.getLogger("radiance.diagnostics")


# ─────────────────────────────────────────────────────────────────────────────
# EOTF inverses (display-referred → scene-linear)
# Only the functions not yet available in color/transfer.py are defined here.
# ─────────────────────────────────────────────────────────────────────────────

def _inverse_eotf_rec709(img: torch.Tensor) -> torch.Tensor:
    """Rec.709 OETF inverse: camera-signal → scene-linear."""
    return torch.where(
        img < 0.081,
        img / 4.5,
        ((img + 0.099) / 1.099) ** (1.0 / 0.45),
    )


def _inverse_eotf_gamma22(img: torch.Tensor) -> torch.Tensor:
    """Pure gamma 2.2 → scene-linear."""
    return img.clamp(min=0.0) ** 2.2


_INVERSE_EOTF = {
    "sRGB":           _inverse_eotf_srgb,
    "Rec.709":        _inverse_eotf_rec709,
    "Gamma 2.2":      _inverse_eotf_gamma22,
    "Linear (no-op)": lambda x: x.clamp(min=0.0),
}


def _dilate_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Binary dilation via max-pool. mask: (B, H, W) or (B, 1, H, W)."""
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)            # (B, 1, H, W)
    if radius <= 0:
        return mask.squeeze(1)
    k = 2 * radius + 1
    dilated = F.max_pool2d(mask.float(), kernel_size=k, stride=1, padding=radius)
    return dilated.squeeze(1)              # (B, H, W)


def _make_clip_mask(
    img: torch.Tensor,
    threshold: float,
    channel_mode: str,
    soft_edge: float,
) -> torch.Tensor:
    """
    Compute a soft/hard clip mask.

    img: (B, H, W, C)   values in [0, 1]
    Returns: (B, H, W)  1 = clipped, 0 = safe
    """
    rgb = img[..., :3]                    # (B, H, W, 3)

    if channel_mode == "any":
        signal = rgb.amax(dim=-1)
    elif channel_mode == "all":
        signal = rgb.amin(dim=-1)
    elif channel_mode == "luma":
        signal = (rgb * img.new_tensor([0.2126, 0.7152, 0.0722])).sum(dim=-1)
    else:
        raise ValueError(f"Unknown clip channel mode: {channel_mode}")

    if soft_edge > 1e-4:
        over = ((signal - (threshold - soft_edge)) / soft_edge).clamp(0.0, 1.0)
    else:
        over = (signal > threshold).float()

    return over  # (B, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# Node 1 — RadianceClipDetector
# ─────────────────────────────────────────────────────────────────────────────

class RadianceClipDetector:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    """
    ◎ Radiance Clip Detector

    Detects saturated / clipped regions in SDR footage where HDR highlight
    data has been permanently lost and must be reconstructed by the model.

    Outputs a soft mask (1 = clipped, 0 = safe) and a visualisation image
    for quality-control.  Wire the mask to RadianceSDRToHDRPrepare and
    RadianceHDRHighlightComposite.

    Parameters
    ──────────
    threshold    — pixels above this value are considered clipped.
                   0.97 catches near-white as well as pure white.
    channel_mode — "any": flag if any RGB channel is clipped (most sensitive)
                   "all": flag only if all three channels are clipped
                   "luma": flag based on luminance (Y)
    soft_edge    — width of the soft transition zone below the threshold.
                   0 = hard binary mask, 0.05 = gentle feathering.
    dilate_px    — expand the mask by this many pixels to cover fringing
                   near highlight boundaries.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Display-encoded SDR image (code values 0-1). Tested as is, no decoding.",
                }),
                "threshold": ("FLOAT", {
                    "default": 0.97, "min": 0.5, "max": 1.0, "step": 0.005,
                    "tooltip": "Pixels brighter than this are marked as clipped.",
                }),
                "channel_mode": (["any", "all", "luma"], {
                    "default": "any",
                    "tooltip": (
                        "any: flag a pixel when any RGB channel is over threshold. all: only when all "
                        "three are. luma: Rec.709 luma. The selected mode also controls the soft edge."
                    ),
                }),
                "soft_edge": ("FLOAT", {
                    "default": 0.03, "min": 0.0, "max": 0.2, "step": 0.005,
                    "tooltip": "Feathering width below the threshold. 0 = hard binary mask.",
                }),
                "dilate_px": ("INT", {
                    "default": 8, "min": 0, "max": 64, "step": 1,
                    "tooltip": "Expand the clip mask outward to cover highlight fringing.",
                }),
            },
        }

    RETURN_TYPES  = ("MASK",   "FLOAT",          "IMAGE")
    RETURN_NAMES  = ("clip_mask", "clip_fraction", "visualization")
    FUNCTION      = "detect"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    DESCRIPTION   = (
        "Detect clipped highlights in SDR images. "
        "Wire clip_mask → RadianceSDRToHDRPrepare and RadianceHDRHighlightComposite."
    )

    def detect(
        self,
        image: torch.Tensor,
        threshold: float = 0.97,
        channel_mode: str = "any",
        soft_edge: float = 0.03,
        dilate_px: int = 8,
    ):
        # image: (B, H, W, C)
        mask = _make_clip_mask(image, threshold, channel_mode, soft_edge)  # (B,H,W)

        # Dilate — expand mask slightly to cover fringing
        if dilate_px > 0:
            mask_bchw = mask.unsqueeze(1)                   # (B,1,H,W)
            k = 2 * dilate_px + 1
            mask_bchw = F.max_pool2d(
                mask_bchw.float(), kernel_size=k, stride=1, padding=dilate_px
            )
            mask = mask_bchw.squeeze(1)                     # (B,H,W)

        clip_fraction = float(mask.mean().item())

        # Visualisation — overlay red tint on clipped areas
        vis = image[..., :3].clone()
        m   = mask.unsqueeze(-1)                            # (B,H,W,1)
        red = torch.zeros_like(vis)
        red[..., 0] = 1.0
        vis = vis * (1.0 - m * 0.6) + red * (m * 0.6)
        vis = vis.clamp(0.0, 1.0)

        logger.info(
            "RadianceClipDetector: threshold=%.3f  clipped=%.1f%%  dilate=%dpx",
            threshold, clip_fraction * 100, dilate_px,
        )
        diag_logger.info(
            "HDR_CLIP_DETECT threshold=%.3f channel_mode=%s clip_fraction=%.4f dilate_px=%d",
            threshold, channel_mode, clip_fraction, dilate_px,
        )

        return (mask, clip_fraction, vis)


# ─────────────────────────────────────────────────────────────────────────────
# Node 2 — RadianceSDRToHDRPrepare
# ─────────────────────────────────────────────────────────────────────────────

class RadianceSDRToHDRPrepare:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    """
    ◎ Radiance SDR → HDR Prepare

    Converts a display-referred SDR image into a VAE-ready input for AI HDR
    highlight reconstruction.  This is the pre-processing step that sits
    between the raw SDR image and the VAE encoder / KSampler.

    What it does
    ────────────
    1. Inverse EOTF — decode the display gamma to get scene-linear values.
       (sRGB, Rec.709, or Gamma 2.2 depending on the source footage)
    2. HDR extrapolation hint — boosts scene-linear values in the clipped
       regions above 1.0 using a power-curve extrapolation.  This seeds the
       VAE with above-white energy so the model has something to work with
       beyond pure white.
    3. Soft-knee compression — maps the boosted linear values back into [0,1]
       for the VAE (same formula as RadianceHDREncoder).
    4. Per-channel normalisation stats — computed and returned so
       RadianceHDRDecoder can invert the normalisation after generation.
    5. Inpainting mask — the clip_mask (from RadianceClipDetector) is passed
       through as the region where the model should generate new content.

    Outputs
    ───────
    image        — VAE-ready image in [0,1] with extrapolated highlights
    mask         — inpainting mask (1=generate, 0=keep)
    stats_json   — per-channel norm stats for RadianceHDRDecoder
    peak_linear  — estimated peak scene-linear value after extrapolation
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Display-encoded SDR image, decoded to linear with inverse_eotf.",
                }),
                "clip_mask": ("MASK", {
                    "tooltip": (
                        "Clip mask from Clip Detector, same size as image. Highlights are boosted "
                        "inside it and it becomes the feathered inpainting mask output."
                    ),
                }),
                "compression_ratio": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Must match RadianceHDRDecoder. Wire from preset or LoRALoader.",
                }),
            },
            "optional": {
                "inverse_eotf": (
                    ["sRGB", "Rec.709", "Gamma 2.2", "Linear (no-op)"],
                    {
                        "default": "sRGB",
                        "tooltip": (
                            "Transfer curve of the input, decoded to linear before the highlight "
                            "boost. Linear (no-op) only clamps negatives."
                        ),
                    },
                ),
                "highlight_boost": ("FLOAT", {
                    "default": 4.0, "min": 1.0, "max": 32.0, "step": 0.5,
                    "tooltip": (
                        "How bright to push extrapolated highlights in scene-linear. "
                        "4.0 = 2 stops above white. Higher = more vivid reconstructed highlights."
                    ),
                }),
                "boost_gamma": ("FLOAT", {
                    "default": 1.5, "min": 0.5, "max": 4.0, "step": 0.1,
                    "tooltip": "Power curve for highlight boost ramp. Higher = sharper specular peaks.",
                }),
                "mask_feather": ("INT", {
                    "default": 16, "min": 0, "max": 128, "step": 1,
                    "tooltip": "Gaussian feather radius on the inpainting mask edge (pixels).",
                }),
            },
        }

    RETURN_TYPES  = ("IMAGE", "MASK",  "STRING",     "FLOAT")
    RETURN_NAMES  = ("image", "mask",  "stats_json", "peak_linear")
    FUNCTION      = "prepare"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    DESCRIPTION   = (
        "Prepare an SDR image for AI HDR reconstruction. "
        "Applies inverse EOTF, highlight extrapolation, and soft-knee compression."
    )

    def prepare(
        self,
        image: torch.Tensor,
        clip_mask: torch.Tensor,
        compression_ratio: float = 0.5,
        inverse_eotf: str = "sRGB",
        highlight_boost: float = 4.0,
        boost_gamma: float = 1.5,
        mask_feather: int = 16,
    ):
        import json

        # ── 1. Inverse EOTF → scene-linear ────────────────────────────────
        fn = _INVERSE_EOTF.get(inverse_eotf, _inverse_eotf_srgb)
        linear = fn(image[..., :3])       # (B, H, W, 3)

        # ── 2. Highlight extrapolation in clipped regions ──────────────────
        # clip_mask: (B, H, W), 1=clipped
        m = clip_mask.unsqueeze(-1).to(linear.device)   # (B,H,W,1)

        # In clipped areas, the SDR value ≈ 1.0. We extrapolate by computing
        # per-pixel boost based on how close the value was to the threshold.
        # Near the threshold: mild boost. Hard-clipped pure white: full boost.
        boost_map = (linear.max(dim=-1, keepdim=True).values.clamp(0.9, 1.0) - 0.9) / 0.1
        boost_val = 1.0 + (highlight_boost - 1.0) * (boost_map ** boost_gamma)
        linear_boosted = linear * (1.0 - m) + linear * boost_val * m

        peak_linear = float(linear_boosted.max().item())

        # ── 3. Soft-knee compress → VAE range [0,1] ────────────────────────
        compressed = _soft_knee_compress(linear_boosted, compression_ratio)

        # Re-attach alpha if present
        if image.shape[-1] == 4:
            compressed = torch.cat([compressed, image[..., 3:4]], dim=-1)

        # ── 4. Per-channel stats (mirrors RadianceHDRPerChannelNorm) ───────
        rgb = compressed[..., :3]                       # (B, H, W, 3)
        mean = rgb.mean(dim=(1, 2))                     # (B, 3)
        std  = rgb.std(dim=(1, 2)).clamp(min=1e-5)      # (B, 3)
        stats = {
            "mean": mean[0].tolist(),
            "std":  std[0].tolist(),
            "compression_ratio": compression_ratio,
            "inverse_eotf": inverse_eotf,
            "highlight_boost": highlight_boost,
        }
        stats_json = json.dumps(stats)

        # ── 5. Feather the inpainting mask ─────────────────────────────────
        out_mask = clip_mask.clone()
        if mask_feather > 0:
            # AUDIT-FIX (2026-08): the old loop reflect-padded ONCE before the
            # first pass, then after passes 2 and 3 restored the shrunken
            # size by zero-padding only the right/bottom edges -- shifting the
            # feathered mask up-left by ~2x the feather radius (32 px at the
            # default 16) and injecting dark bands. An inpainting mask offset
            # from the clipped highlights it marks guides the AI to repair
            # the wrong pixels. Pad symmetrically inside the loop instead:
            # size is exact after every pass, no shift, no bands.
            k = 2 * mask_feather + 1
            box = torch.ones(1, 1, k, k, device=out_mask.device) / (k * k)
            blurred = out_mask.unsqueeze(1).float()
            for _ in range(3):
                padded = F.pad(blurred, [mask_feather] * 4, mode="reflect")
                blurred = F.conv2d(padded, box, padding=0)
            out_mask = blurred[:, 0]

        logger.info(
            "RadianceSDRToHDRPrepare: eotf=%s  ratio=%.2f  boost=%.1f  peak_linear=%.3f",
            inverse_eotf, compression_ratio, highlight_boost, peak_linear,
        )
        diag_logger.info(
            "HDR_UPLIFT_PREPARE eotf=%s compression_ratio=%.3f boost=%.1f peak_linear=%.3f",
            inverse_eotf, compression_ratio, highlight_boost, peak_linear,
        )

        return (compressed, out_mask, stats_json, peak_linear)


# ─────────────────────────────────────────────────────────────────────────────
# Node 3 — RadianceHDRHighlightComposite
# ─────────────────────────────────────────────────────────────────────────────

class RadianceHDRHighlightComposite:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    """
    ◎ Radiance HDR Highlight Composite

    Final compositing step of the SDR→HDR pipeline.

    After the diffusion model has reconstructed HDR highlights, this node:
    1. Blends the AI-generated HDR output with the original SDR (linearised)
       using the clip_mask as the blend guide.
       • Clipped regions → AI HDR reconstruction
       • Safe regions    → original linearised SDR (pixel-perfect)
    2. Applies an optional highlight blend softening to avoid hard seams
       at the boundary between original and reconstructed content.
    3. Outputs scene-linear HDR ready for EXR save or further grading.

    This approach guarantees that:
    - Non-clipped areas are never degraded by the model
    - Reconstructed highlights are seamlessly blended
    - The transition is smooth and artefact-free

    Inputs
    ──────
    original_image   — original SDR input (same as fed to RadianceClipDetector)
    hdr_image        — scene-linear HDR output from RadianceHDRDecoder
    clip_mask        — from RadianceClipDetector (1=AI region, 0=keep original)
    blend_softness   — Gaussian feathering on the composite edge (pixels)
    shadow_lift      — small lift applied to the original SDR in linear space
                       to prevent pure black from clipping in the blend
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE", {
                    "tooltip": (
                        "Original display-encoded SDR image. Linearised with inverse_eotf and "
                        "used outside the mask; its alpha is carried to the output."
                    ),
                }),
                "hdr_image":      ("IMAGE", {
                    "tooltip": (
                        "Scene-linear HDR reconstruction (e.g. from HDR Decoder). Used inside the "
                        "mask; resized bilinearly if its size differs from original_image."
                    ),
                }),
                "clip_mask":      ("MASK", {
                    "tooltip": "Blend guide from Clip Detector: 1 takes hdr_image, 0 keeps the original.",
                }),
            },
            "optional": {
                "inverse_eotf": (
                    ["sRGB", "Rec.709", "Gamma 2.2", "Linear (no-op)"],
                    {
                        "default": "sRGB",
                        "tooltip": (
                            "Transfer curve of original_image, used to linearise it so it matches "
                            "hdr_image. Use the same setting as SDR to HDR Prepare."
                        ),
                    },
                ),
                "blend_softness": ("INT", {
                    "default": 24, "min": 0, "max": 128, "step": 1,
                    "tooltip": "Feathering on composite edge in pixels. 0 = hard cut.",
                }),
                "highlight_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "How much of the AI highlight reconstruction to use. 1.0 = full.",
                }),
            },
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("image",)
    FUNCTION      = "composite"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    DESCRIPTION   = (
        "Composite AI-reconstructed HDR highlights back onto original linearised SDR. "
        "Non-clipped areas are pixel-perfect original. Only clipped regions come from the model."
    )

    def composite(
        self,
        original_image: torch.Tensor,
        hdr_image: torch.Tensor,
        clip_mask: torch.Tensor,
        inverse_eotf: str = "sRGB",
        blend_softness: int = 24,
        highlight_strength: float = 1.0,
    ):
        device = original_image.device

        # Linearise original SDR
        fn = _INVERSE_EOTF.get(inverse_eotf, _inverse_eotf_srgb)
        linear_sdr = fn(original_image[..., :3].to(device))

        # Align hdr_image spatial size to original if needed
        if hdr_image.shape[1:3] != original_image.shape[1:3]:
            hdr_bchw = hdr_image[..., :3].permute(0, 3, 1, 2)
            hdr_bchw = F.interpolate(
                hdr_bchw,
                size=original_image.shape[1:3],
                mode="bilinear",
                align_corners=False,
            )
            hdr_rgb = hdr_bchw.permute(0, 2, 3, 1)
        else:
            hdr_rgb = hdr_image[..., :3].to(device)

        # Build blend mask (soft)
        blend_mask = clip_mask.clone().float().to(device)
        if blend_softness > 0:
            k = 2 * blend_softness + 1
            bm = blend_mask.unsqueeze(1)
            bm = F.avg_pool2d(bm, kernel_size=k, stride=1, padding=blend_softness)
            blend_mask = bm.squeeze(1).clamp(0.0, 1.0)

        blend_mask = blend_mask * highlight_strength
        m = blend_mask.unsqueeze(-1)   # (B, H, W, 1)

        # Composite: safe regions from original, highlights from AI
        result = linear_sdr * (1.0 - m) + hdr_rgb * m

        # Re-attach alpha if present
        if original_image.shape[-1] == 4:
            result = torch.cat([result, original_image[..., 3:4]], dim=-1)

        peak = float(result.max().item())
        logger.info(
            "RadianceHDRHighlightComposite: blend_softness=%d  strength=%.2f  peak=%.3f",
            blend_softness, highlight_strength, peak,
        )
        diag_logger.info(
            "HDR_COMPOSITE blend_softness=%d strength=%.2f peak_linear=%.3f",
            blend_softness, highlight_strength, peak,
        )

        return (result,)


# ─────────────────────────────────────────────────────────────────────────────
# ComfyUI registration
# ─────────────────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "RadianceClipDetector":          RadianceClipDetector,
    "RadianceSDRToHDRPrepare":       RadianceSDRToHDRPrepare,
    "RadianceHDRHighlightComposite": RadianceHDRHighlightComposite,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceClipDetector":          "◎ Radiance Clip Detector",
    "RadianceSDRToHDRPrepare":       "◎ Radiance SDR → HDR Prepare",
    "RadianceHDRHighlightComposite": "◎ Radiance HDR Highlight Composite",
}
