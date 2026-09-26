import torch
import os
import math
import logging
import uuid
import hashlib
from typing import Dict, Any, Tuple

import folder_paths

# ALBABIT-FIX: guarded like sampler.py/sampler_utils.py already do. An
# unguarded import here would take down this whole node group's __init__.py
# (Sampler, Loader, Prompt, VAE Decode, not just Resolution) on any ComfyUI
# predating NestedTensor.
try:
    import comfy.nested_tensor
    _HAS_NESTED_TENSOR = True
except ImportError:
    _HAS_NESTED_TENSOR = False

from radiance.model.detect import _BASE_VRAM, _BASE_CLIP_VRAM

logger = logging.getLogger("radiance.resolution")


# ═══════════════════════════════════════════════════════════════════════════════
#                         RESOLUTION DATABASE
# ═══════════════════════════════════════════════════════════════════════════════

# Format: (width, height, category, aspect_ratio_label)
PRESETS: Dict[str, Tuple[int, int, str, str]] = {
    # ── Cinema / Film ──
    "4K DCI (4096×2160)": (4096, 2160, "Cinema", "1.90:1"),
    "4K UHD (3840×2160)": (3840, 2160, "Cinema", "16:9"),
    "2K DCI (2048×1080)": (2048, 1080, "Cinema", "1.90:1"),
    "HD 1080p (1920×1080)": (1920, 1080, "Cinema", "16:9"),
    "HD 720p (1280×720)": (1280, 720, "Cinema", "16:9"),
    "Anamorphic 2.39:1 (2048×856)": (2048, 856, "Cinema", "2.39:1"),
    "Anamorphic 2.39:1 (4096×1712)": (4096, 1712, "Cinema", "2.39:1"),
    "Super 35 (2048×1552)": (2048, 1552, "Cinema", "1.32:1"),
    "IMAX (5616×4096)": (5616, 4096, "Cinema", "1.37:1"),
    "Academy 4:3 (1440×1080)": (1440, 1080, "Cinema", "4:3"),
    "VistaVision (3072×2048)": (3072, 2048, "Cinema", "3:2"),
    "8K UHD (7680×4320)": (7680, 4320, "Cinema", "16:9"),
    # ── Social / Delivery ──
    "Instagram Square (1080×1080)": (1080, 1080, "Social", "1:1"),
    "Instagram Story (1080×1920)": (1080, 1920, "Social", "9:16"),
    "YouTube Thumb (1280×720)": (1280, 720, "Social", "16:9"),
    "TikTok (1080×1920)": (1080, 1920, "Social", "9:16"),
}

PRESET_NAMES = ["Custom"] + list(PRESETS.keys())

# ALBABIT-FIX: model-specific behavior is now driven entirely by `model_type`
# (SPATIAL_SCALE, VIDEO_MODEL_TYPES, LATENT_FORMAT_MAP below), not preset
# category. Presets are now plain Cinema/Social resolutions, model-agnostic.
#
# Deferred:
#  - WAN lost its 16px alignment heuristic, falls back to the 8px default now
#    (no SPATIAL_SCALE entry). Revisit if 16px is actually needed.
#  - VIDEO_MODEL_TYPES audited vs comfy_extras/ (2026-06-12), verified
#    TEMPORAL_SCALE: LTXV/Cosmos 8, WAN/HunyuanVideo/CogVideoX 4, Mochi 6
#    ((length-1)//6+1 per nodes_mochi.py). Cosmos Predict2 (stride 4) NOT covered.
#  - Flux.1 vs Flux.2 alignment not further split beyond existing entries.
#  - MiniMax H3 audited (2026-08-18): 17k+5 frame grid has no TEMPORAL_SCALE
#    entry (_minimax_align_frame_count/_minimax_video_latent_t instead).
#    Audio (32ch stereo) out of scope for this node.

# Model types that emit 5D latent (1, C, T, H, W)
VIDEO_MODEL_TYPES = {"WAN (16ch)", "WAN TI2V (48ch)", "LTXV (128ch)", "HunyuanVideo (16ch)", "Mochi (12ch)", "Cosmos World (16ch)", "CogVideoX (16ch)", "MiniMax H3 (24ch)",
                     "HunyuanVideo 1.5 (32ch)", "Kandinsky 5 Video (16ch)"}

#: Ceiling shared by the video_frames widget and the 'Auto (Seconds)' frame
#: computation, so the two ways of asking for a clip length reach the same place.
_MAX_VIDEO_FRAMES = 100000

# Latent format string matching nodes_sampler.py latent_format input
LATENT_FORMAT_MAP = {
    # ALBABIT-FIX: "Manual" is the new default (formerly "Auto (Flux 16ch)") —
    # this node has no `model` input so "Auto" detection was never real.
    # SPATIAL_SCALE/TEMPORAL_SCALE=1 below make width/height/video_frames fully
    # unconstrained; "flux" remains a sensible latent_format fallback, and the
    # user can override channels/format via the `latent_channels` input.
    "Manual": "flux",
    # ALBABIT-FIX: merged with "Lumina2 / Z-Image (16ch)" — both are 16ch, 8px,
    # "flux" latent format with no other distinguishing entries anywhere in
    # resolution.py (SPATIAL_SCALE/TEMPORAL_SCALE/VIDEO_MODEL_TYPES all default).
    "Flux / SD3 / Lumina2 / Z-Image (16ch)": "flux",
    # ALBABIT-FIX: SDXL/SD 1.5/PixArt/AuraFlow are all 4ch, 8px, "sdxl" latent
    # format with no other distinguishing entries -- merged into one option.
    "SDXL / SD 1.5 / PixArt / Aura Flow (4ch)": "sdxl",
    # ALBABIT-FIX: Chroma uses the Flux latent format (16ch, 8px spatial
    # compression) — matches sampler_utils.py's "chroma" -> "flux" mapping.
    "Chroma (16ch)": "chroma",
    # ALBABIT-FIX: Cosmos/CogVideoX/Mochi map to their own sampler model_type
    # (matches sampler_utils.py keys) instead of being aliased to "flux"
    "Cosmos World (16ch)": "cosmos",
    "CogVideoX (16ch)": "cogvideox",
    "Mochi (12ch)": "mochi",
    # ALBABIT-FIX: LTX-Video latent format. "ltxav" (not "ltx") to match the
    # model_type key used by RadianceSamplerPro (sampler_utils.py). Covers
    # both LTX 2.3 and 2.5 -- same 128ch transformer, capability-based
    # detection, and (confirmed against the real checkpoints) identical VAE
    # spatial/temporal compression too, so one entry serves both.
    "LTXV (128ch)": "ltxav",
    # ALBABIT-FIX: Added model types matching the Radiance Video Loader / RUDRA decoder set
    "WAN (16ch)": "wan",
    # ALBABIT-FIX: WAN 2.2 TI2V-5B -- distinct 48ch VAE (comfy.latent_formats.Wan22),
    # real bug fix (was silently defaulting to "WAN (16ch)"'s 16ch/8px, a
    # wrong-shaped-latent crash risk at sampling for this checkpoint).
    "WAN TI2V (48ch)": "wan_ti2v",
    # ALBABIT-FIX: "hunyuan_video" (not "hunyuan") to match sampler_utils.py model_type
    "HunyuanVideo (16ch)": "hunyuan_video",
    # ALBABIT-FIX: Flux.2 latent format (comfy.latent_formats.Flux2)
    "Flux.2 / Flux.2 Klein (128ch)": "flux2",
    # ALBABIT-FIX: matches comfy/supported_models.py's MiniMaxH3.unet_config
    # ("image_model": "minimax_h3"). RadianceSamplerPro doesn't recognize this
    # latent_format yet — Sampler-side support is a separate, later task.
    "MiniMax H3 (24ch)": "minimax_h3",
    # 3.5: ComfyUI 0.32 families. Qwen-Image / Krea 2 ride the Wan 2.1 VAE
    # (16ch, 8px, 2D); HiDream / OmniGen2 / LongCat / Kandinsky 5 image ride
    # the Flux VAE; HunyuanImage 2.1 is 64ch at 32px; HunyuanVideo 1.5 is
    # 32ch at 16px / 4 frames; Kandinsky 5 video uses the HunyuanVideo VAE.
    "Qwen-Image / Krea 2 (16ch)": "qwen_image",
    "HiDream / OmniGen2 / LongCat / Kandinsky 5 Image (16ch)": "flux",
    "HunyuanImage 2.1 (64ch)": "hunyuan_image",
    "HunyuanVideo 1.5 (32ch)": "hunyuan_video_15",
    "Kandinsky 5 Video (16ch)": "kandinsky5",
}

# Common aspect ratios for megapixel target mode
MP_ASPECT_RATIOS = [
    "1:1", "4:3", "3:2", "16:9", "21:9",
    "2.39:1", "1.85:1", "9:16", "2:3", "3:4",
]

MODEL_TYPES = [
    "Manual",
    # ALBABIT-FIX: merged "Flux / SD3 (16ch)" + "Lumina2 / Z-Image (16ch)" — both
    # 16ch/8px/"flux" with no other distinguishing entries in this file.
    "Flux / SD3 / Lumina2 / Z-Image (16ch)",
    "SDXL / SD 1.5 / PixArt / Aura Flow (4ch)",
    "Chroma (16ch)",
    "Cosmos World (16ch)",
    "CogVideoX (16ch)",
    # ALBABIT-FIX: Mochi has its own temporal compression (×6, see TEMPORAL_SCALE)
    # and is a 5D video latent like WAN/LTXV/HunyuanVideo.
    "Mochi (12ch)",
    # ALBABIT-FIX: LTX-Video uses a 128-channel latent (vs 16ch for Flux/SD3).
    # Covers both LTX 2.3 and 2.5 -- confirmed identical VAE compression, see
    # LATENT_FORMAT_MAP above.
    "LTXV (128ch)",
    # ALBABIT-FIX: Added model types matching the Radiance Video Loader / RUDRA decoder set
    "WAN (16ch)",
    # ALBABIT-FIX: WAN 2.2 TI2V-5B -- previously had no dedicated option here at
    # all, forcing users onto "WAN (16ch)" (wrong channel count/spatial scale).
    "WAN TI2V (48ch)",
    "HunyuanVideo (16ch)",
    # ALBABIT-FIX: Flux.2 / Flux.2 Klein — 128ch latent like LTXV, but ×16 spatial
    # downscale (vs ×32 for LTXV) and no 5D/video handling.
    "Flux.2 / Flux.2 Klein (128ch)",
    # ALBABIT-FIX: MiniMax H3 — 24ch video latent, 16px spatial (comfy_extras/
    # nodes_minimax_h3.py's _empty_av_latent: height//16, width//16). Video-only:
    # the model's native audio stream (32ch stereo) isn't produced here — pair
    # with a separate audio latent + "Concat AV Latent" for the full AV pipeline.
    "MiniMax H3 (24ch)",
    # 3.5: ComfyUI 0.32 families.
    "Qwen-Image / Krea 2 (16ch)",
    "HiDream / OmniGen2 / LongCat / Kandinsky 5 Image (16ch)",
    "HunyuanImage 2.1 (64ch)",
    "HunyuanVideo 1.5 (32ch)",
    "Kandinsky 5 Video (16ch)",
]

ORIENTATIONS = ["As Preset", "Landscape", "Portrait", "Square"]

# Latent channels per model type
LATENT_CHANNELS = {
    "Manual": 16,
    "Flux / SD3 / Lumina2 / Z-Image (16ch)": 16,
    "SDXL / SD 1.5 / PixArt / Aura Flow (4ch)": 4,
    "Chroma (16ch)": 16,
    "Cosmos World (16ch)": 16,
    "CogVideoX (16ch)": 16,
    "Mochi (12ch)": 12,
    # ALBABIT-FIX: LTX-Video latent is 128 channels (2.3 and 2.5 alike)
    "LTXV (128ch)": 128,
    # ALBABIT-FIX: Added model types matching the Radiance Video Loader / RUDRA decoder set
    "WAN (16ch)": 16,
    # ALBABIT-FIX: WAN 2.2 TI2V-5B's VAE is comfy.latent_formats.Wan22 (48
    # latent channels) -- real bug fix, see the model_type list comment above.
    "WAN TI2V (48ch)": 48,
    "HunyuanVideo (16ch)": 16,
    # ALBABIT-FIX: Flux.2 latent is 128 channels (comfy.latent_formats.Flux2)
    "Flux.2 / Flux.2 Klein (128ch)": 128,
    # ALBABIT-FIX: MiniMax H3's video stream is 24 latent channels (the 32ch
    # audio stream isn't produced by this node, see MODEL_TYPES comment above).
    "MiniMax H3 (24ch)": 24,
    # 3.5 families (comfy/latent_formats.py).
    "Qwen-Image / Krea 2 (16ch)": 16,
    "HiDream / OmniGen2 / LongCat / Kandinsky 5 Image (16ch)": 16,
    "HunyuanImage 2.1 (64ch)": 64,
    "HunyuanVideo 1.5 (32ch)": 32,
    "Kandinsky 5 Video (16ch)": 16,
}

# ── Per-model latent spatial downscale factor (VAE compression) ─────────────────
# ALBABIT-FIX: This used to be a single global LATENT_SCALE=8 for every model_type,
# which is correct for SD/SDXL/Flux/SD3/WAN/Hunyuan/Mochi/CogVideoX (all ×8 VAEs)
# but produced grossly oversized latents for LTXV (×32) and Flux.2 (×16).
# 8 remains the default for any model_type not listed here.
SPATIAL_SCALE = {
    # ALBABIT-FIX: 32x spatial, confirmed identical for LTX 2.3 and 2.5 against
    # comfy/sd.py's hardcoded ratio for the real diffusion-decoder VAE class
    # (a wrong 16x entry here once produced 2x-oversized output).
    "LTXV (128ch)": 32,
    "Flux.2 / Flux.2 Klein (128ch)": 16,
    # ALBABIT-FIX: WAN 2.2 TI2V-5B's VAE trades channel depth for spatial
    # compression -- comfy.latent_formats.Wan22 sets spacial_downscale_ratio=16
    # (double the standard WAN's implicit 8x). Real bug fix: this model_type had
    # no entry at all, silently falling back to the 8px default -- wrong latent
    # size, not just a metadata inaccuracy.
    "WAN TI2V (48ch)": 16,
    # ALBABIT-FIX: MiniMax H3's video VAE compresses 16x spatially, confirmed
    # against comfy_extras/nodes_minimax_h3.py's _empty_av_latent (plain
    # height//16, width//16 — the bespoke adapt_canvas() short-edge/area-cap
    # logic in that file is a recommended-range helper, not enforced here).
    "MiniMax H3 (24ch)": 16,
    # 3.5: HunyuanImage 2.1's VAE is 32x (comfy.latent_formats.HunyuanImage21,
    # 64ch); HunyuanVideo 1.5's is 16x spatial (HunyuanVideo15,
    # spacial_downscale_ratio = 16). The other new families are 8x.
    "HunyuanImage 2.1 (64ch)": 32,
    "HunyuanVideo 1.5 (32ch)": 16,
    # ALBABIT-FIX: "Manual" uses scale=1 -> _align_up is a no-op, so width/height
    # are fully unconstrained (no rounding, +/- step of 1) for experimental models.
    "Manual": 1,
}

# ALBABIT-FIX: pixel alignment where it differs from the VAE's spatial compression.
# MiniMax H3 compresses 16x but patchifies 2x2, so its keyframe latents (Add Guide,
# Image to Video) need an even latent size: 45 rows (720px) crashes patchify_video.
# The native nodes step width/height by 32 for the same reason.
SPATIAL_ALIGN = {
    "MiniMax H3 (24ch)": 32,
}

# ── Per-model latent temporal downscale factor (3D VAE compression) ─────────────
# ALBABIT-FIX: Restored from previous radiance version — without this, the empty
# video latent's temporal dimension was set to the raw pixel-space frame count
# (e.g. 241 for LTXV), instead of the compressed latent frame count (31), causing
# the sampler to process ~8x more "frames" than necessary. 4 is the default for
# any video model_type not listed here.
TEMPORAL_SCALE = {
    # ALBABIT-FIX: 8x temporal, confirmed identical for LTX 2.3 and 2.5 --
    # comfy/sd.py's downscale_ratio formula for the real diffusion-decoder VAE
    # is `(a+7)//8`, same as SPATIAL_SCALE's note above.
    "LTXV (128ch)": 8,
    "WAN (16ch)": 4,
    # ALBABIT-FIX: WAN 2.2 TI2V-5B keeps the same 4x temporal compression as
    # standard WAN (comfy.latent_formats.Wan22 inherits temporal_downscale_ratio
    # from Wan21, only spacial_downscale_ratio is overridden) -- listed
    # explicitly for clarity even though it matches this table's own default.
    "WAN TI2V (48ch)": 4,
    "HunyuanVideo (16ch)": 4,
    # 3.5: HunyuanVideo 1.5 (temporal_downscale_ratio = 4) and Kandinsky 5
    # video, which decodes through the HunyuanVideo VAE.
    "HunyuanVideo 1.5 (32ch)": 4,
    "Kandinsky 5 Video (16ch)": 4,
    "CogVideoX (16ch)": 4,
    # ALBABIT-FIX: Mochi's VAE temporal compression is ×6 (nodes_mochi.py:
    # (length-1)//6+1), distinct from the 4x default used by WAN/Hunyuan/CogVideoX.
    "Mochi (12ch)": 6,
    # ALBABIT-FIX: Cosmos 1.0 "World" text/image-to-video models (e.g.
    # Cosmos-1_0-Diffusion-7B-Text2World) use a ×8 temporal compression
    # (nodes_cosmos.py), same as LTXV. Cosmos Predict2 (×4) is not covered.
    "Cosmos World (16ch)": 8,
    # ALBABIT-FIX: "Manual" uses stride=1 -> any video_frames value satisfies
    # (n-1)%1==0, so frame-count snapping/validation is fully unconstrained.
    "Manual": 1,
    # ALBABIT-FIX: MiniMax H3 deliberately has NO entry here — its 17k+5 frame
    # grid isn't a fixed divisor (see _minimax_align_frame_count/_minimax_
    # video_latent_t below), so generate() branches on MINIMAX_H3_MODEL_TYPE
    # before ever reaching this table's .get(model_type, 4) fallback.
}

# ── VRAM Estimation Metadata ──────────────────────────────────────────────────
# Bytes per latent element (ComfyUI usually uses float32 internally = 4 bytes)
LATENT_ELEMENT_BYTES = 4
# ALBABIT-FIX: this used to be a separate MODEL_BASE_VRAM dict, hand-duplicated
# from model/detect.py's _BASE_VRAM. It drifted (missing CLIP cost, stale UNET
# numbers), disagreeing with the Loader's own estimate (Flux.2: 20.0 GB shown
# here vs 28.0 GB real). Reuses _BASE_VRAM/_BASE_CLIP_VRAM directly now.

LATENT_SCALE = 8  # VAE downscale factor


# ═══════════════════════════════════════════════════════════════════════════════
#                         HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def _align_up(val: int, scale: int) -> int:
    """Round UP to the nearest multiple of `scale` (never down)."""
    return max(scale, math.ceil(val / scale) * scale)


# ── MiniMax H3 temporal grid ─────────────────────────────────────────────────
# ALBABIT-FIX: mirrors comfy_extras/nodes_minimax_h3.py's align_frame_count()/
# video_latent_t() verbatim — the model's frame count snaps to n%17==5 (5, 22,
# 39...), not the stride*k+1 pattern every other VIDEO_MODEL_TYPES entry uses.
MINIMAX_H3_MODEL_TYPE = "MiniMax H3 (24ch)"
MINIMAX_H3_FPS = 24  # hardcoded FPS in nodes_minimax_h3.py — no variable-fps support
# ALBABIT-FIX: audio latent's own temporal fps (AUDIO_LATENT_FPS in
# nodes_minimax_h3.py), unrelated to the video's 24fps.
MINIMAX_H3_AUDIO_FPS = 40


def _minimax_align_frame_count(n: int) -> int:
    """Snap UP to the nearest valid MiniMax H3 frame count (n % 17 == 5)."""
    n = max(5, n)
    while n % 17 != 5:
        n += 1
    return n


def _minimax_floor_frame_count(n: int) -> int:
    """Snap DOWN to the nearest valid MiniMax H3 frame count (n % 17 == 5).

    Companion to _minimax_align_frame_count (up): _minimax_video_latent_t
    already floors internally for off-grid input, this exposes that same
    floored frame count so audio_t derives from the identical basis instead
    of a separately ceiling-aligned one, which drifted the two out of sync.
    """
    n = max(5, n)
    while n % 17 != 5:
        n -= 1
    return n


def _minimax_audio_latent_t(frame_count: int) -> int:
    """Audio latent frame count for a given (already-aligned) video frame count.

    Mirrors nodes_minimax_h3.py's temporal_shape() exactly: round(duration * 40).
    """
    duration = frame_count / MINIMAX_H3_FPS
    return round(duration * MINIMAX_H3_AUDIO_FPS)


def _minimax_video_latent_t(frame_count: int) -> int:
    """Latent temporal size for a frame count (rounds down to the grid if unaligned)."""
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2


def _estimate_vram(w: int, h: int, c: int, b: int, format_key: str = "flux", spatial_scale: int = 8) -> float:
    """
    Estimate VRAM usage in Gigabytes.
    Includes latent tensor size + estimated model activation overhead.
    """
    # Latent dimensions (model-specific spatial downscale, e.g. 8/16/32)
    lw, lh = w // spatial_scale, h // spatial_scale
    # Tensor size in bytes
    latent_bytes = b * c * lw * lh * LATENT_ELEMENT_BYTES
    # Convert to GB
    latent_gb = latent_bytes / (1024**3)
    # Model overhead: UNET + CLIP/text-encoder, same tables the Loader's own
    # estimate_vram_usage() uses -- keeps this node's readout consistent with
    # the Loader's, instead of a separately-drifting local table.
    key = format_key.lower()
    base_gb = _BASE_VRAM.get(key, 4.0) + _BASE_CLIP_VRAM.get(key, 2.0)
    # Total
    return latent_gb + base_gb


def _apply_orientation(w: int, h: int, orientation: str) -> Tuple[int, int]:
    """Apply orientation override."""
    if orientation == "Landscape":
        return (max(w, h), min(w, h))
    elif orientation == "Portrait":
        return (min(w, h), max(w, h))
    elif orientation == "Square":
        side = max(w, h)
        return (side, side)
    return (w, h)  # "As Preset"


def _gcd_ratio(w: int, h: int) -> str:
    """Calculate simplified aspect ratio string."""
    g = math.gcd(w, h)
    rw, rh = w // g, h // g
    # Simplify common large ratios
    if rw > 50 or rh > 50:
        ratio = w / h
        # Check common cinema ratios
        for name, val in [
            ("1:1", 1.0),
            ("4:3", 4 / 3),
            ("3:2", 3 / 2),
            ("16:9", 16 / 9),
            ("21:9", 21 / 9),
            ("2.39:1", 2.39),
            ("1.85:1", 1.85),
            ("1.90:1", 1.9),
            ("1.37:1", 1.37),
            ("9:16", 9 / 16),
            ("2:3", 2 / 3),
            ("3:4", 3 / 4),
        ]:
            if abs(ratio - val) < 0.02:
                return name
        return f"{ratio:.2f}:1"
    return f"{rw}:{rh}"


def _mp_target_dimensions(mp_target: float, aspect_str: str, align_val: int = 8) -> tuple[int, int]:
    """
    FEATURE: Compute (width, height) from a megapixel target and aspect ratio string.
    The result is aligned to `align_val`px (model-specific, via SPATIAL_SCALE) and
    respects the exact aspect ratio as closely as possible without exceeding the
    MP target.

    Examples:
        _mp_target_dimensions(1.0, "16:9")  → (1360, 768)  ≈ 1.04MP
        _mp_target_dimensions(2.0, "2.39:1") → (2192, 917) ≈ 2.01MP
    """
    # Parse aspect ratio
    aspect_str = aspect_str.strip()
    try:
        if ":" in aspect_str:
            parts = aspect_str.split(":")
            ratio = float(parts[0]) / float(parts[1])
        else:
            ratio = float(aspect_str)
    except (ValueError, ZeroDivisionError):
        ratio = 1.0

    # Solve: w*h = mp_target*1e6, w/h = ratio
    # → h = sqrt(mp*1e6 / ratio), w = h * ratio
    pixels = max(1024.0, mp_target * 1_000_000)
    h_raw = math.sqrt(pixels / ratio)
    w_raw = h_raw * ratio

    w = _align_up(int(round(w_raw)), align_val)
    h = _align_up(int(round(h_raw)), align_val)
    return w, h


def _render_preview_card(
    width: int,
    height: int,
    preset_name: str,
    model_type: str,
    latent_c: int,
    batch_label: str,
    batch_value: str,
    category: str = "",
    enable_video: bool = False,
    video_frames: int = 0,
    frame_rate: float = 24.0,
    vram_est: float = 0.0,
    align_label: str = "8px",
    spatial_scale: int = 8,
) -> "PIL.Image.Image":  # noqa: F821  (forward-ref; PIL imported lazily in body)
    """
    Radiance HUD-style resolution preview card.

    Redesigned v3.6 — Compact HUD Update:
    ┌─ RADIANCE RESOLUTION ────────────────────────────────┐
    │ [CINEMA]                               2.21 MP ●●●○○ │  ← header bar
    ├───────────────────────────────────────────────────────┤
    │  VRAM PRESSURE: [||||||||      ]  12.4 GB    (ALIGNED)│  ← New VRAM bar
    │    ┌─────────────────────────────────────────┐        │
    │    │  · · · · · · · · · · · · · · · · · · ·  │        │  ← AR box
    │    │  · · · · · · ─ ─ ─⬥─ ─ ─ · · · · · ·  │        │
    │    └─────────────────────────────────────────┘        │
    │                    2048 × 1080                        │
    ├──────────────────────┬────────────────────────────────┤
    │  RESOLUTION          │  2048 × 1080  [32px]           │  ← align info
    │  ASPECT RATIO        │  1.90:1                        │
    │  EST. VRAM           │  12.42 GB                      │  ← VRAM info
    │  LATENT              │  256 × 135 × 16ch              │
    ├──────────────────────┴────────────────────────────────┤
    │  PRESET   2K DCI    MODEL   Flux 16ch   BATCH  1      │
    └───────────────────────────────────────────────────────┘
    """
    from PIL import Image, ImageDraw

    # ── Palette — Radiance / ComfyUI dark theme ───────────────────────────────
    # Matches the Autodesk Flame-inspired Radiance UI colour system
    # ── Obsidian Glass & Ocean Cyan Design Overhaul (Apple macOS Style) ──────
    C_BG           = (8, 8, 12)         # Deep Obsidian Black backing
    C_PANEL        = (22, 22, 29)       # Apple dark obsidian glass fill
    C_BORDER       = (45, 45, 55)       # Satin graphite border outline
    C_ACCENT       = (0, 189, 255)      # Premium Ocean Cyan accent
    C_ACCENT_DIM   = (0, 120, 170)      # Muted Ocean Cyan shadow
    C_ACCENT_GLOW  = (72, 183, 255)     # Glowing neon cyan keylights
    C_GRID         = (30, 32, 40)       # Rule-of-thirds grid
    C_CROSS        = (60, 65, 80)       # HUD centering crosshairs
    C_TEXT_HI      = (245, 245, 247)    # Crisp white display value
    C_TEXT_MID     = (160, 160, 168)    # Clean midtone labels
    C_TEXT_DIM     = (124, 124, 132)    # Subdued footers
    C_SEP          = (28, 28, 35)       # Fine graphite separators
    C_ROW_EVEN     = (14, 14, 20)       # Alternating even rows
    C_ROW_ODD      = (20, 20, 26)       # Alternating odd rows
    C_GOOD_MP      = (52, 199, 89)      # macOS green status dot (low density)
    C_MED_MP       = (255, 149, 0)      # macOS orange status dot (medium density)
    C_HIGH_MP      = (255, 59, 48)      # macOS red status dot (high density)
    C_VRAM_BAR     = (30, 30, 38)       # VRAM progress track

    card_w, card_h = 512, 406
    img = Image.new("RGB", (card_w, card_h), C_BG)
    draw = ImageDraw.Draw(img)

    # ── Font loading (cross-platform) ─────────────────────────────────────────
    import platform as _platform
    _plat = _platform.system()
    if _plat == "Windows":
        _wf = os.environ.get("WINDIR", "C:\\Windows")
        font_paths = [
            os.path.join(_wf, "Fonts", "consola.ttf"),
            os.path.join(_wf, "Fonts", "lucon.ttf"),
            os.path.join(_wf, "Fonts", "cour.ttf"),
            os.path.join(_wf, "Fonts", "arial.ttf"),
        ]
    elif _plat == "Darwin":
        font_paths = [
            "/System/Library/Fonts/Menlo.ttc",
            "/System/Library/Fonts/Monaco.ttf",
            "/Library/Fonts/Courier New.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ]
    else:
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
            "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    fT, fL, fS, fXS = None, None, None, None
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                from PIL import ImageFont as IF
                fT  = IF.truetype(fp, 20)   # Title
                fL  = IF.truetype(fp, 13)   # Labels / values
                fS  = IF.truetype(fp, 11)   # Small / footer
                fXS = IF.truetype(fp, 10)   # Tiny (corner HUD)
                break
            except Exception:
                continue
    if fT is None:
        logger.debug("No system font found — using PIL default.")
        from PIL import ImageFont
        fT = fL = fS = fXS = ImageFont.load_default()

    # ── Measurements ──────────────────────────────────────────────────────────
    aspect_str  = _gcd_ratio(width, height)
    megapixels  = (width * height) / 1_000_000
    # ALBABIT-FIX: use per-model spatial downscale (LTXV=32, Flux.2=16, default=8)
    lat_w       = width // spatial_scale
    lat_h       = height // spatial_scale

    HEADER_H    = 34
    STATUS_H    = 22
    AR_TOP      = HEADER_H + STATUS_H + 8
    AR_MARGIN   = 44
    AR_MAX_H    = 160
    ar          = width / height

    # Scale AR box to available area
    area_w = card_w - AR_MARGIN * 2
    if ar >= 1.0:
        bw = area_w
        bh = int(bw / ar)
        if bh > AR_MAX_H:
            bh = AR_MAX_H
            bw = int(bh * ar)
    else:
        bh = AR_MAX_H
        bw = int(bh * ar)
        if bw > area_w:
            bw = area_w
            bh = int(bw / ar)

    bx = (card_w - bw) // 2
    by = AR_TOP

    DIM_LABEL_H = 20      # height of "2048 × 1080" below AR box
    INFO_TOP    = by + bh + DIM_LABEL_H + 8
    INFO_ROW_H  = 26
    INFO_ROWS   = 4       # RESOLUTION, ASPECT, MP, LATENT
    INFO_H      = INFO_ROWS * INFO_ROW_H
    FOOTER_H    = 34

    # The preview is content-driven. Older builds pinned the footer to a fixed
    # 580px canvas, which left a large empty lower screen in ComfyUI.
    card_h = INFO_TOP + INFO_H + FOOTER_H + 8
    img = Image.new("RGB", (card_w, card_h), C_BG)
    draw = ImageDraw.Draw(img)
    FOOTER_Y = card_h - FOOTER_H

    # ═════════════════════════════════════════════════════════════════════════
    # 1. HEADER BAR
    # ═════════════════════════════════════════════════════════════════════════
    draw.rectangle([0, 0, card_w, HEADER_H], fill=(22, 22, 29))
    # Elegant divider line under header
    draw.line([0, HEADER_H, card_w, HEADER_H], fill=C_SEP, width=1)

    # Title (brand-aligned ◎ Resolution)
    draw.text((16, HEADER_H // 2), "◎ RADIANCE RESOLUTION",
              fill=C_ACCENT, font=fT, anchor="lm")

    # Category badge (top right)
    cat_label = (category or "CUSTOM").upper()
    badge_x = card_w - 12
    draw.text((badge_x, HEADER_H // 2), cat_label,
              fill=C_ACCENT_DIM, font=fS, anchor="rm")

    # ═════════════════════════════════════════════════════════════════════════
    # 2. MEGAPIXEL / VRAM STATUS STRIP (below title, above AR box)
    # ═════════════════════════════════════════════════════════════════════════
    # 5-dot indicator: each dot = 2MP, max shown = 10MP
    dot_y = HEADER_H + 5
    dot_r = 3
    dot_spacing = 10
    dot_count = 5
    max_mp = 10.0
    filled = min(dot_count, int(round(megapixels / max_mp * dot_count)))
    dots_total_w = dot_count * dot_spacing
    dot_start_x = card_w - 14 - dots_total_w

    # MP text left of dots
    mp_color = C_GOOD_MP if megapixels <= 2 else (C_MED_MP if megapixels <= 8 else C_HIGH_MP)
    draw.text((dot_start_x - 6, dot_y + dot_r), f"{megapixels:.2f} MP",
              fill=mp_color, font=fS, anchor="rm")

    for di in range(dot_count):
        dx = dot_start_x + di * dot_spacing + dot_r
        color = mp_color if di < filled else C_BORDER
        draw.ellipse([dx - dot_r, dot_y, dx + dot_r, dot_y + dot_r * 2], fill=color)

    # ── VRAM PRESSURE BAR (New v3.5) ──────────────────────────────────────────
    vram_y = HEADER_H + 15
    vram_bar_w = 112
    vram_bar_h = 4
    vram_bar_x = 16
    
    # Label
    draw.text((vram_bar_x, vram_y - 2), "VRAM", fill=C_TEXT_DIM, font=fXS, anchor="lt")
    
    # Track (rounded ends)
    track_x = vram_bar_x + 42
    draw.rounded_rectangle([track_x, vram_y, track_x + vram_bar_w, vram_y + vram_bar_h], radius=2, fill=C_VRAM_BAR)
    
    # Fill based on 24GB max (RTX 3090/4090 standard)
    max_vram = 24.0
    fill_pct = min(1.0, vram_est / max_vram)
    fill_w = int(vram_bar_w * fill_pct)
    fill_color = C_GOOD_MP if vram_est < 12 else (C_MED_MP if vram_est < 20 else C_HIGH_MP)
    
    if fill_w > 0:
        draw.rounded_rectangle([track_x, vram_y, track_x + fill_w, vram_y + vram_bar_h], radius=2, fill=fill_color)
    
    # Value text
    draw.text((track_x + vram_bar_w + 10, vram_y - 2), f"{vram_est:.1f} GB", fill=fill_color, font=fXS, anchor="lt")
    
    # Alignment Badge (right)
    draw.text((card_w - 14, vram_y - 2), f"[{align_label.upper()} ALIGNED]", fill=C_TEXT_DIM, font=fXS, anchor="rt")

    # ═════════════════════════════════════════════════════════════════════════
    # 3. ASPECT RATIO BOX — professional VFX HUD style (macOS Squircled)
    # ═════════════════════════════════════════════════════════════════════════
    # Panel fill using beautiful rounded corner rectangles
    draw.rounded_rectangle([bx, by, bx + bw, by + bh], radius=8, fill=C_PANEL)

    # Rule-of-thirds grid (subtle)
    for xi in [1, 2]:
        gx = bx + bw * xi // 3
        draw.line([gx, by + 1, gx, by + bh - 1], fill=C_GRID, width=1)
    for yi in [1, 2]:
        gy = by + bh * yi // 3
        draw.line([bx + 1, gy, bx + bw - 1, gy], fill=C_GRID, width=1)

    # Crosshair lines (slightly brighter than grid)
    cx, cy = bx + bw // 2, by + bh // 2
    cross_len_h = bw // 2 - 6
    cross_len_v = bh // 2 - 4
    draw.line([cx - cross_len_h, cy, cx - 8, cy], fill=C_CROSS, width=1)
    draw.line([cx + 8, cy, cx + cross_len_h, cy], fill=C_CROSS, width=1)
    draw.line([cx, cy - cross_len_v, cx, cy - 6], fill=C_CROSS, width=1)
    draw.line([cx, cy + 6, cx, cy + cross_len_v], fill=C_CROSS, width=1)

    # Center diamond (Ocean Cyan)
    diam = 5
    draw.polygon([
        (cx,        cy - diam),
        (cx + diam, cy),
        (cx,        cy + diam),
        (cx - diam, cy),
    ], fill=C_ACCENT)

    # Corner brackets — VFX HUD style
    blen = min(14, bw // 5, bh // 4)
    bthk = 1
    corners = [
        (bx,      by,      +1, +1),   # top-left
        (bx + bw, by,      -1, +1),   # top-right
        (bx,      by + bh, +1, -1),   # bottom-left
        (bx + bw, by + bh, -1, -1),   # bottom-right
    ]
    for (ox, oy, sx, sy) in corners:
        draw.line([ox, oy, ox + sx * blen, oy],           fill=C_ACCENT, width=bthk + 1)
        draw.line([ox, oy, ox,             oy + sy * blen], fill=C_ACCENT, width=bthk + 1)

    # Outer border (graphite satin boundary outline matching macOS squircles)
    draw.rounded_rectangle([bx, by, bx + bw, by + bh], radius=8, outline=C_BORDER, width=1)

    # AR label — inside box, top-left corner (subtle)
    ar_label_inside = aspect_str
    draw.text((bx + 6, by + 4), ar_label_inside, fill=C_ACCENT_DIM, font=fXS)

    # Video badge — inside AR box, top-right corner
    if enable_video:
        duration = float(video_frames) / max(frame_rate, 1.0)
        vid_label = f"▶ {video_frames}f  {duration:.1f}s"
        draw.text((bx + bw - 6, by + 4), vid_label,
                  fill=(100, 200, 255), font=fXS, anchor="ra")

    # ═════════════════════════════════════════════════════════════════════════
    # 4. DIMENSION LABEL (below AR box)
    # ═════════════════════════════════════════════════════════════════════════
    dim_y = by + bh + 4
    draw.text((cx, dim_y), f"{width} × {height}",
              fill=C_ACCENT_GLOW, font=fL, anchor="mt")

    # ═════════════════════════════════════════════════════════════════════════
    # 5. INFO GRID (alternating rows)
    # ═════════════════════════════════════════════════════════════════════════
    # Separator above info
    sep_y = INFO_TOP - 4
    draw.line([0, sep_y, card_w, sep_y], fill=C_SEP, width=1)

    LEFT_PAD  = 20
    MID_X     = 210   # divider between label and value
    VALUE_X   = MID_X + 12

    info_rows = [
        ("RESOLUTION",  f"{width} × {height}  ({align_label})",  C_TEXT_HI),
        ("ASPECT RATIO", aspect_str,                          C_TEXT_HI),
        ("EST. VRAM",   f"{vram_est:.2f} GB",                 fill_color),
        ("LATENT",      f"{lat_w} × {lat_h} × {latent_c}ch", C_ACCENT_GLOW),
    ]

    for i, (label, value, val_color) in enumerate(info_rows):
        ry = INFO_TOP + i * INFO_ROW_H
        row_bg = C_ROW_EVEN if i % 2 == 0 else C_ROW_ODD
        draw.rectangle([0, ry, card_w, ry + INFO_ROW_H - 1], fill=row_bg)

        # Vertical divider between label and value
        draw.line([MID_X, ry + 4, MID_X, ry + INFO_ROW_H - 4], fill=C_SEP, width=1)

        # Label (left, vertically centred)
        draw.text((LEFT_PAD, ry + INFO_ROW_H // 2), label,
                  fill=C_TEXT_MID, font=fL, anchor="lm")

        # Value (right of divider)
        draw.text((VALUE_X, ry + INFO_ROW_H // 2), value,
                  fill=val_color, font=fL, anchor="lm")

    # Separator below info
    after_info_y = INFO_TOP + INFO_H
    draw.line([0, after_info_y, card_w, after_info_y], fill=C_SEP, width=1)

    # ═════════════════════════════════════════════════════════════════════════
    # 6. FOOTER STRIP — PRESET / MODEL / BATCH in one line
    # ═════════════════════════════════════════════════════════════════════════
    draw.rectangle([0, FOOTER_Y, card_w, card_h], fill=(22, 22, 29))
    draw.line([0, FOOTER_Y, card_w, FOOTER_Y], fill=C_SEP, width=1)

    fy = FOOTER_Y + FOOTER_H // 2

    # Shorten long preset names
    pname = preset_name if preset_name != "Custom" else "Custom"
    if len(pname) > 22:
        pname = pname[:20] + "…"
    mname = model_type.split("(")[0].strip()
    if len(mname) > 14:
        mname = mname[:12] + "…"

    # Three columns: PRESET | MODEL | BATCH
    col_w = card_w // 3
    for ci, (lbl, val) in enumerate([
        ("PRESET", pname),
        ("MODEL",  mname),
        (batch_label, batch_value),
    ]):
        cx_col = col_w * ci + col_w // 2
        # Vertical separator (not on last)
        if ci > 0:
            draw.line([col_w * ci, FOOTER_Y + 6, col_w * ci, card_h - 6],
                      fill=C_SEP, width=1)
        # Label above, value below — two-line layout
        draw.text((cx_col, fy - 8), lbl,  fill=C_TEXT_DIM,  font=fXS, anchor="mm")
        draw.text((cx_col, fy + 8), val,  fill=C_TEXT_MID, font=fS,  anchor="mm")

    return img



# ═══════════════════════════════════════════════════════════════════════════════
#                        NODE IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════════════════


class RadianceResolution:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Utilities"
    """
    Professional resolution selector with internal preview.

    Outputs an empty LATENT at the selected resolution with correct
    channel count for your model (4ch for SD/SDXL, 16ch for Flux/SD3).

    The internal preview shows a resolution info card with aspect ratio
    visualization, dimensions, megapixel count, and latent info.
    """

    # Per-node preview file tracking — keyed by unique_id.
    # Overwritten on each run so temp files don't accumulate.
    _preview_paths: Dict[str, str] = {}

    @classmethod
    def IS_CHANGED(
        cls,
        preset, width, height, orientation, model_type, batch_size,
        scale_factor=1.0, latent_channels=0, enable_video=False,
        crop_to_broadcast_resolution=True,
        frame_computation="Manual (Frames)", duration_seconds=5.0,
        video_frames=81, frame_rate=24.0, mp_target=0.0,
        mp_aspect_ratio="16:9", unique_id="",
    ):
        """Re-execute only when inputs actually change — avoids redundant renders."""
        state = (
            f"{preset}|{width}|{height}|{orientation}|{model_type}|{batch_size}|"
            f"{scale_factor}|{latent_channels}|{enable_video}|"
            f"{crop_to_broadcast_resolution}|"
            f"{frame_computation}|{duration_seconds}|{video_frames}|"
            f"{frame_rate}|{mp_target}|{mp_aspect_ratio}"
        )
        return hashlib.md5(state.encode()).hexdigest()

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "preset": (
                    PRESET_NAMES,
                    {
                        "default": "Custom",
                        "tooltip": (
                            "Resolution preset (Cinema or Social). The final width/height "
                            "are automatically aligned for the selected 'model_type'. "
                            "Select 'Custom' to use manual width/height."
                        ),
                    },
                ),
                "width": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 64,
                        "max": 16384,
                        "step": 8,
                        "tooltip": "Custom width (only used when preset is 'Custom'). Auto-aligned to 8px (32px for LTX Video).",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 64,
                        "max": 16384,
                        "step": 8,
                        "tooltip": "Custom height (only used when preset is 'Custom'). Auto-aligned to 8px (32px for LTX Video).",
                    },
                ),
                "orientation": (
                    ORIENTATIONS,
                    {
                        "default": "As Preset",
                        "tooltip": "Override orientation. 'As Preset' uses the preset's native orientation.",
                    },
                ),
                "model_type": (
                    MODEL_TYPES,
                    {
                        "default": "Manual",
                        "tooltip": (
                            "Drives pixel alignment, video-latent shape, frame-count "
                            "rules, latent_format, and the Est. VRAM readout.\n"
                            "Flux/SD3/Cosmos = 16ch. SDXL/SD 1.5 = 4ch. Mochi = 12ch.\n"
                            "MiniMax H3 = 24ch video-only (fixed 24fps, 17k+5 frame "
                            "grid); pair with an audio latent + 'Concat AV Latent' "
                            "for the full AV pipeline.\n"
                            "'Manual': no alignment/frame-count constraints; use "
                            "'latent_channels' for experimental/unlisted models.\n"
                            "Est. VRAM assumes a full load; actual usage may be lower "
                            "with DynamicVRAM/CPU offload active."
                        ),
                    },
                ),
                "batch_size": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 64,
                        "step": 1,
                        "tooltip": "Number of latent frames in batch.",
                    },
                ),
            },
            "optional": {
                "scale_factor": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        # ALBABIT-FIX: ComfyUI derives display precision from step's order
                        # of magnitude, not its decimal count -- 0.25 wrongly got precision=1
                        # ("0.3" instead of "0.25"). 0.1 has no such loss. min=0.1 (not 0) to
                        # avoid a literal 0x scale zeroing out width/height.
                        "min": 0.1,
                        "max": 4.0,
                        "step": 0.1,
                        "tooltip": (
                            "Scale the resolution by this factor after preset/custom. "
                            "0.5 = half res, 2.0 = double res. Applied before alignment."
                        ),
                    },
                ),
                "latent_channels": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 256,
                        "step": 1,
                        "tooltip": (
                            "Override latent channel count. 0 = use model_type default. "
                            "Common: 4 (SD/SDXL), 12 (Mochi), 16 (Flux/SD3/Cosmos). "
                            "Set manually for custom architectures."
                        ),
                    },
                ),
                "mp_target": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 64.0,
                        # ALBABIT-FIX: same ComfyUI precision-derivation bug as
                        # scale_factor -- 0.25 truncated to 1 displayed decimal.
                        "step": 0.1,
                        "tooltip": (
                            "MEGAPIXEL TARGET: When > 0, auto-calculates W×H from this MP target "
                            "and mp_aspect_ratio. Overrides preset and custom W/H. "
                            "0 = disabled."
                        ),
                    },
                ),
                "mp_aspect_ratio": (
                    MP_ASPECT_RATIOS,
                    {
                        "default": "16:9",
                        "tooltip": "Aspect ratio for megapixel target mode (only used when mp_target > 0).",
                    },
                ),
                "enable_video": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Enable video sequence mode (replaces batch parameter)."},
                ),
                # ALBABIT-FIX: Restored from previous radiance version, generalized to
                # images too (old fork was video-only). crop_bbox below always reports
                # the diff between the requested size and align_val's padding, for any
                # preset/model_type/custom size, not just a fixed table of broadcast
                # standards like the old fork's 1088->1080 lookup.
                "crop_to_broadcast_resolution": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Compute crop_bbox to remove model-alignment padding "
                            "(e.g. 1920x1088 -> 1920x1080 for LTX's 32px grid). "
                            "Wire crop_bbox into RadianceHDRVAEDecode's crop_bbox "
                            "input to actually apply the crop after decode."
                        ),
                    },
                ),
                # ALBABIT-FIX: Restored from previous radiance version — lets the user pick
                # a target duration in seconds instead of a raw frame count.
                "frame_computation": (
                    ["Manual (Frames)", "Auto (Seconds)"],
                    {"default": "Manual (Frames)",
                     "tooltip": (
                         "How the video frame count is set (video only). Manual uses video_frames as typed; "
                         "Auto (Seconds) uses duration_seconds x frame_rate snapped to the model's stride*k+1 grid."
                     )},
                ),
                "duration_seconds": (
                    "FLOAT",
                    {
                        # DEFECT: this was capped at 120.0 while video_frames
                        # reaches 100000, so 'Auto (Seconds)' could not express
                        # any clip the 'Manual (Frames)' path could. At 24fps
                        # the two ceilings were 120s and ~69 minutes. The cap is
                        # now the frame ceiling at the lowest frame rate the
                        # frame_rate widget allows (100000 frames at 1fps), so
                        # the two entry modes reach the same place.
                        "default": 5.0, "min": 0.1, "max": 100000.0, "step": 0.1,
                        "tooltip": (
                            "Target video duration in seconds (used when "
                            "frame_computation = 'Auto (Seconds)'). Combined with "
                            "frame_rate this must stay within video_frames' 100000 "
                            "ceiling; a longer request is clamped with a warning."
                        ),
                    },
                ),
                "video_frames": (
                    "INT",
                    {
                        "default": 81, "min": 1, "max": _MAX_VIDEO_FRAMES, "step": 1,
                        "tooltip": (
                            "Total number of video frames. "
                            "5D-latent models require (stride*k+1) — e.g. 4k+1 for "
                            "WAN/HunyuanVideo (1, 5, 9, 13...), 8k+1 for LTXV (1, 9, 17...), "
                            "6k+1 for Mochi (1, 7, 13...). MiniMax H3 uses its own 17k+5 "
                            "grid instead (5, 22, 39, 56...). "
                            "A warning is logged if this constraint is violated. "
                            "Ignored when frame_computation = 'Auto (Seconds)'."
                        ),
                    },
                ),
                "frame_rate": (
                    "FLOAT",
                    {"default": 24.0, "min": 1.0, "max": 120.0, "step": 1.0, "tooltip": "Playback frame rate."},
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    # ALBABIT-FIX: Restored from previous radiance version — multi-output (width,
    # height, channels, info, frame_rate, frame_count, latent_format, duration_sec)
    # so this node can drive Sampler Pro / other downstream nodes directly.
    RETURN_TYPES = ("LATENT", "INT", "INT", "INT", "STRING", "FLOAT", "INT", "STRING", "FLOAT", "BOUNDING_BOX")
    RETURN_NAMES = ("latent", "width", "height", "channels", "info", "frame_rate", "frame_count", "latent_format", "duration_sec", "crop_bbox")
    OUTPUT_TOOLTIPS = (
        "Empty latent tensor at the selected resolution.",
        "Final image width (pixels).",
        "Final image height (pixels).",
        "Latent channel count.",
        "Resolution info string.",
        "Playback frame rate. Always the widget value — never 0.0.",
        "Total video frames (or batch size for images).",
        "Latent format string — wire to Sampler Pro latent_format input.",
        "Duration in seconds (video_frames / frame_rate). 0.0 for images.",
        "Crop box {x, y, width, height} to remove model-alignment padding. Wire into RadianceHDRVAEDecode's crop_bbox input.",
    )
    FUNCTION = "generate"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Generate"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Professional resolution selector with internal preview card. "
        "Outputs empty LATENT for Flux/SDXL/SD/Cosmos/CogVideoX with correct channel count. "
        "40+ cinema/social/AI/video presets. WAN frame count validation. LTX 32px alignment. "
        "Manual latent_channels override for custom architectures."
    )

    def generate(
        self,
        preset: str,
        width: int,
        height: int,
        orientation: str,
        model_type: str,
        batch_size: int,
        scale_factor: float = 1.0,
        latent_channels: int = 0,
        enable_video: bool = False,
        crop_to_broadcast_resolution: bool = True,
        frame_computation: str = "Manual (Frames)",
        duration_seconds: float = 5.0,
        video_frames: int = 81,
        frame_rate: float = 24.0,
        mp_target: float = 0.0,
        mp_aspect_ratio: str = "16:9",
        unique_id: str = "",
    ) -> Dict[str, Any]:

        # ── Step 2 (computed early): Determine Alignment Rule (model_type-driven) ──
        # ALBABIT-FIX: alignment is derived solely from SPATIAL_SCALE for the
        # selected model_type (LTXV=32, Flux.2=16, default=8), always rounded UP.
        align_val   = SPATIAL_ALIGN.get(model_type) or SPATIAL_SCALE.get(model_type, 8)
        align_label = f"{align_val}px"

        # ── Megapixel target mode overrides preset/custom ────────────────────────
        if mp_target > 0.0:
            # BUG FIX: warn when mp_target produces a size too small for any model
            min_pixels = 256 * 256  # 65536px — floor below which no model works
            if mp_target * 1_000_000 < min_pixels:
                logger.warning(
                    f"mp_target={mp_target} MP produces < 256×256 pixels. "
                    f"Minimum recommended is 0.065 MP (256×256). Results may be unusable."
                )
            w, h = _mp_target_dimensions(mp_target, mp_aspect_ratio, align_val)
            category = "MP Target"
        # ── Resolve resolution from preset or custom ─────────────────────────────
        elif preset != "Custom" and preset in PRESETS:
            w, h, category, ar_label = PRESETS[preset]
            # BUG FIX: the width/height widgets are fully ignored whenever a
            # named preset is selected — a frequent source of "I typed 1280x720
            # but got a different size" confusion, since the widgets stay
            # visible/editable and look like they should apply. Surface it
            # loudly (WARNING, not debug) whenever the ignored values differ
            # from the preset, so it shows up in the default console output.
            if (width, height) != (w, h):
                logger.warning(
                    f"RadianceResolution: preset '{preset}' is selected, so the "
                    f"width/height widgets ({width}×{height}) are ignored — using "
                    f"the preset's {w}×{h} instead. Set preset to 'Custom' to use "
                    f"manual width/height."
                )
        else:
            w, h = width, height
            category = "Custom"

        # ALBABIT-FIX: captured before scale_factor, matching the old fork's
        # "ignore scale_factor" crop design. scale_factor drives the LTX 2.3
        # LowRes pass; the pipeline's own 2x upscale brings the decode back to
        # the un-scaled size, so crop_bbox must target that, not this call's
        # own scaled-down latent.
        req_w, req_h = w, h

        # Apply scale factor — always surface it (not just on alignment
        # correction): a leftover scale_factor from a previous run silently
        # doubling/halving every subsequent resolution is a common source of
        # "I set X but got a different size" confusion.
        if scale_factor != 1.0:
            w_pre = int(w * scale_factor)
            h_pre = int(h * scale_factor)
            logger.warning(
                f"RadianceResolution: scale_factor={scale_factor} is applied — "
                f"{w}×{h} → {w_pre}×{h_pre} (before alignment). Set scale_factor "
                f"to 1.0 to disable this."
            )
            w, h = w_pre, h_pre

        # Apply alignment — always round UP, never down
        w, h = _align_up(w, align_val), _align_up(h, align_val)

        # ── Step 3: Latent Format & VRAM Estimation (model_type-driven) ──────────
        latent_format = LATENT_FORMAT_MAP.get(model_type, "flux" if LATENT_CHANNELS.get(model_type, 16) >= 16 else "sdxl")

        # ALBABIT-FIX: hoisted, was re-checked 4 times below (audit finding,
        # no behavior change).
        is_minimax_h3 = model_type == MINIMAX_H3_MODEL_TYPE

        # ALBABIT-FIX: Restored from previous radiance version — auto frame count from
        # a target duration, aligned to the model's temporal stride (n*stride + 1).
        if enable_video and frame_computation == "Auto (Seconds)":
            if is_minimax_h3:
                # ALBABIT-FIX: MiniMax H3 has no variable-frame-rate support.
                # nodes_minimax_h3.py's FPS=24 is hardcoded, so the grid alignment
                # always assumes 24fps regardless of the frame_rate widget.
                if frame_rate != MINIMAX_H3_FPS:
                    logger.warning(
                        f"MiniMax H3 is fixed at {MINIMAX_H3_FPS}fps. frame_rate="
                        f"{frame_rate} only affects the frame_rate/duration_sec "
                        f"outputs, not this frame-grid alignment."
                    )
                video_frames = _minimax_align_frame_count(int(round(duration_seconds * MINIMAX_H3_FPS)))
                logger.info(
                    f"Auto-Seconds: {duration_seconds}s @ {MINIMAX_H3_FPS}fps (fixed) -> "
                    f"Aligned to {video_frames} frames (17k+5 grid)"
                )
            else:
                raw_frames = duration_seconds * float(frame_rate)

                stride = TEMPORAL_SCALE.get(model_type, 4)

                video_frames = max(1, int(round(raw_frames / stride)) * stride + 1)
                # duration_seconds now reaches the same ceiling as video_frames,
                # so the product can exceed it. Clamp down to the nearest valid
                # stride*k+1 at or below the limit rather than emitting a frame
                # count the video_frames widget itself could not hold.
                if video_frames > _MAX_VIDEO_FRAMES:
                    clamped = ((_MAX_VIDEO_FRAMES - 1) // stride) * stride + 1
                    logger.warning(
                        f"Auto-Seconds: {duration_seconds}s @ {frame_rate}fps needs "
                        f"{video_frames} frames, above the {_MAX_VIDEO_FRAMES}-frame "
                        f"ceiling. Clamped to {clamped} frames "
                        f"({clamped / max(frame_rate, 1.0):.1f}s). Lower "
                        f"duration_seconds or frame_rate to render the full length."
                    )
                    video_frames = clamped
                logger.info(
                    f"Auto-Seconds: {duration_seconds}s @ {frame_rate}fps -> "
                    f"Aligned to {video_frames} frames (stride {stride})"
                )

        # Estimate VRAM
        v_count = video_frames if enable_video else batch_size
        vram_est = _estimate_vram(w, h, latent_channels or LATENT_CHANNELS.get(model_type, 4), v_count, latent_format, SPATIAL_SCALE.get(model_type, 8))

        # Apply orientation
        w, h = _apply_orientation(w, h, orientation)
        req_w, req_h = _apply_orientation(req_w, req_h, orientation)

        # ── Step 5: Video frame count validation (model_type-driven) ────────────
        # 5D-latent models require frame count = (stride*k + 1): 1, 5, 9, 13...
        # for stride=4 (WAN/HunyuanVideo), or 1, 9, 17... for stride=8 (LTXV), etc.
        if enable_video and is_minimax_h3:
            aligned = _minimax_align_frame_count(video_frames)
            if aligned != video_frames:
                lower = aligned - 17
                neighbors = f"{lower} or {aligned}" if lower >= 5 else str(aligned)
                logger.warning(
                    f"MiniMax H3 requires frame count % 17 == 5 (5, 22, 39, 56...). "
                    f"Got {video_frames}. Nearest valid value(s): {neighbors}. "
                    f"Using {video_frames} may cause sampler errors or incorrect output."
                )
        elif enable_video and model_type in VIDEO_MODEL_TYPES:
            stride = TEMPORAL_SCALE.get(model_type, 4)
            if (video_frames - 1) % stride != 0:
                k_low  = (video_frames - 1) // stride
                v_low  = stride * k_low + 1
                v_high = v_low + stride
                logger.warning(
                    f"{model_type} requires frame count = {stride}k+1 "
                    f"(1, {stride + 1}, {2 * stride + 1}, {3 * stride + 1}...). Got {video_frames}. "
                    f"Nearest valid values: {v_low} or {v_high}. "
                    f"Using {video_frames} may cause sampler errors or incorrect output."
                )

        # ── Latent channels ──────────────────────────────────────────────────────
        if latent_channels > 0:
            latent_c = latent_channels
        else:
            latent_c = LATENT_CHANNELS.get(model_type, 16)

        # ── Determine if this is a video latent (model_type-driven) ────
        # ALBABIT-FIX: "Manual" is not in VIDEO_MODEL_TYPES (so selecting it doesn't
        # auto-enable enable_video in the JS toggle), but if the user explicitly
        # enables video with "Manual" selected, still compute a 5D latent — with
        # TEMPORAL_SCALE["Manual"]=1, i.e. no compression assumed.
        is_video_latent = enable_video and (model_type in VIDEO_MODEL_TYPES or model_type == "Manual")

        # DEFECT: enable_video=True against an IMAGE model_type fell through to
        # the 4D branch below, which builds `actual_batch` INDEPENDENT stills
        # with no temporal relationship at all, while the info string and the
        # preview card still read "VIDEO: 81f @ 24fps" with a duration. The
        # frame-count stride validation above is also skipped entirely, because
        # both of its branches require a video model_type. Nothing anywhere said
        # the output was not video.
        if enable_video and not is_video_latent:
            logger.warning(
                "[Radiance] enable_video=True with model_type='%s', which is an IMAGE "
                "architecture. The output is %d UNRELATED STILL IMAGES in a batch, not "
                "a video clip: there is no temporal axis, no frame-count stride check "
                "and no temporal VAE. Set model_type to a video architecture (%s) or "
                "'Manual', or turn enable_video off and use batch_size.",
                model_type, video_frames,
                ", ".join(sorted(VIDEO_MODEL_TYPES)),
            )

        # DEFECT: batch_size was the one ignored widget in this node with no
        # warning attached. Every other ignored widget (width/height under a
        # preset, frame_rate on MiniMax H3) already says so.
        if enable_video and batch_size != 1:
            logger.warning(
                "[Radiance] batch_size=%d is ignored while enable_video=True. The "
                "latent's batch axis is hard-coded to 1 for video and video_frames "
                "drives the temporal axis instead. Turn enable_video off to batch "
                "stills, or queue the prompt %d times to render %d clips.",
                batch_size, batch_size, batch_size,
            )

        actual_batch = video_frames if enable_video else batch_size

        # ── Create empty latent ──────────────────────────────────────────────────
        # ALBABIT-FIX: use per-model spatial downscale (LTXV=32, Flux.2=16, default=8)
        # instead of the global LATENT_SCALE constant.
        spatial_scale = SPATIAL_SCALE.get(model_type, LATENT_SCALE)
        lat_h = h // spatial_scale
        lat_w = w // spatial_scale

        if is_video_latent:
            if is_minimax_h3:
                # ALBABIT-FIX: 17k+5 grid, not a fixed divisor. See
                # _minimax_video_latent_t (mirrors nodes_minimax_h3.py exactly).
                lat_t = _minimax_video_latent_t(actual_batch)
            else:
                # ALBABIT-FIX: Restored from previous radiance version — compress the raw
                # frame count to the latent's temporal dimension via the 3D VAE block
                # equation: (frames - 1) // temporal_scale + 1.
                temporal_scale = TEMPORAL_SCALE.get(model_type, 4)
                lat_t = (actual_batch - 1) // temporal_scale + 1
            latent = torch.zeros(1, latent_c, lat_t, lat_h, lat_w, dtype=torch.float32)
            logger.info(
                f"Video latent 5D: (1, {latent_c}, {lat_t}, {lat_h}, {lat_w})"
            )
            if is_minimax_h3:
                # ALBABIT-FIX: real bug, found live. MiniMaxH3Model.forward()
                # (comfy/ldm/minimax/model.py) does audio_src = x[1]
                # unconditionally, crashing a video-only latent even for
                # pure T2V. Needs a real NestedTensor(video, audio) pair,
                # silence as zeros, mirroring _empty_av_latent() exactly.
                if not _HAS_NESTED_TENSOR:
                    raise RuntimeError(
                        "MiniMax H3 needs comfy.nested_tensor, which this ComfyUI "
                        "install doesn't have. Update ComfyUI to a version with "
                        "NestedTensor support to use this model_type."
                    )
                # ALBABIT-FIX: real bug, found via code review, not live. Was
                # feeding a ceiling-aligned frame count here while lat_t above
                # uses a floor-aligned one for the same off-grid actual_batch,
                # so video and audio drifted out of sync (audio ran longer).
                # _minimax_floor_frame_count matches what lat_t already uses.
                audio_t = _minimax_audio_latent_t(_minimax_floor_frame_count(actual_batch))
                audio = torch.zeros(1, 32, 2, audio_t, dtype=torch.float32)
                latent = comfy.nested_tensor.NestedTensor((latent, audio))
                logger.info(f"Audio latent (silent): (1, 32, 2, {audio_t})")
        else:
            latent = torch.zeros(actual_batch, latent_c, lat_h, lat_w, dtype=torch.float32)

        latent_dict = {"samples": latent}

        # ── Build info string ────────────────────────────────────────────────────
        megapixels = (w * h) / 1_000_000
        pixel_count = w * h
        ar_str = _gcd_ratio(w, h)
        ch_src = "manual" if latent_channels > 0 else model_type.split("(")[0].strip()

        # Keyed off is_video_latent, not enable_video: a batch of unrelated
        # stills produced by enable_video against an image model_type used to
        # report itself as "VIDEO: 81f @ 24fps" with a duration.
        if is_video_latent:
            batch_label = "VIDEO"
            batch_value = f"{video_frames}f @ {frame_rate}fps"
        elif enable_video:
            batch_label = "STILLS"
            batch_value = f"{video_frames} unrelated frames (no temporal axis)"
        else:
            batch_label = "BATCH"
            batch_value = str(batch_size)

        from radiance.core.logging import supports_unicode
        divider = "│" if supports_unicode() else "|"
        info = (
            f"{w}×{h} ({ar_str}) {megapixels:.2f}MP {divider} "
            f"Latent: {lat_w}×{lat_h}×{latent_c}ch ({ch_src}) {divider} "
            f"{batch_label.capitalize()}: {batch_value}"
        )
        logger.info(info)

        # ── Render internal preview card ─────────────────────────────────────────
        preview_images = []
        try:
            display_model = (
                f"Manual ({latent_c}ch)" if latent_channels > 0 else model_type
            )
            preview_img = _render_preview_card(
                width=w,
                height=h,
                preset_name=preset,
                model_type=display_model,
                latent_c=latent_c,
                batch_label=batch_label,
                batch_value=batch_value,
                category=category,
                enable_video=enable_video,
                video_frames=video_frames,
                frame_rate=frame_rate,
                vram_est=vram_est,
                align_label=align_label,
                spatial_scale=spatial_scale,
            )

            output_dir = folder_paths.get_temp_directory()

            # BUG FIX: Use deterministic filename per node instance (unique_id).
            # Overwrites the previous preview for this node on each run —
            # no temp file accumulation. Falls back to uuid if unique_id not set.
            node_key = unique_id if unique_id else uuid.uuid4().hex[:8]
            preview_filename = f"radiance_resolution_{node_key}.png"
            preview_path = os.path.join(output_dir, preview_filename)

            # Remove old preview file if it exists (in case of uuid fallback path)
            _old = RadianceResolution._preview_paths.get(node_key)
            if _old and _old != preview_path and os.path.exists(_old):
                try:
                    os.remove(_old)
                except OSError as _exc:
                    logger.debug(
                        "[Radiance] generate(): ignoring %s from `os.remove(_old)`: %s",
                        type(_exc).__name__, _exc,
                    )
            RadianceResolution._preview_paths[node_key] = preview_path

            preview_img.save(preview_path, "PNG")
            preview_images.append({"filename": preview_filename, "subfolder": "", "type": "temp"})
            logger.debug(f"Preview saved: {preview_path}")

        except Exception as e:
            logger.warning(f"Preview render failed: {e}")

        # ── Latent format string (model_type-driven) ───────────────────
        latent_fmt = latent_format
        if latent_channels > 0 and not is_video_latent:
            latent_fmt = "flux" if latent_c >= 16 else "sdxl"

        # is_video_latent, not enable_video: a batch of stills has no duration,
        # and reporting one let a downstream muxer stamp a frame rate on it.
        duration_sec = video_frames / frame_rate if is_video_latent else 0.0

        # ALBABIT-FIX: full_w/full_h re-align req_w/req_h on their own rather
        # than reusing w/h, for the same scale_factor reason as above. When
        # disabled, crop_bbox is still a well-formed full-frame box, so wiring
        # it downstream is always harmless regardless of the toggle state.
        full_w, full_h = _align_up(req_w, align_val), _align_up(req_h, align_val)
        if crop_to_broadcast_resolution:
            crop_x, crop_y = (full_w - req_w) // 2, (full_h - req_h) // 2
        else:
            crop_x, crop_y, req_w, req_h = 0, 0, full_w, full_h
        crop_bbox = {"x": crop_x, "y": crop_y, "width": req_w, "height": req_h}

        return {
            "ui": {
                "images": preview_images,
                "computed_width": [w],
                "computed_height": [h],
            },
            "result": (
                latent_dict, w, h, latent_c, info,
                float(frame_rate), int(actual_batch), latent_fmt, duration_sec,
                crop_bbox,
            ),
        }


# ═══════════════════════════════════════════════════════════════════════════════
#                        NODE REGISTRATION
# ═══════════════════════════════════════════════════════════════════════════════

NODE_CLASS_MAPPINGS = {
    "RadianceResolution": RadianceResolution,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceResolution": "◎ Radiance Resolution",
}
