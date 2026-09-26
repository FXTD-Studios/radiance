# ============================================================
# FXTD STUDIOS — Radiance
# nodes_dit_adapter.py  —  DiT Latent Space Adapter
# ============================================================
# Bridges Radiance's HDR color pipeline to Diffusion Transformer
# video models running inside ComfyUI.
#
# Problem this solves
# -------------------
# SD-VAE latents:   [B, 4,  H/8,  W/8]     scale ≈ 0.18215
# LTX-2.x latents:  [B, 128, T/8, H/32, W/32]  scale ≈ 1.0
# HunyuanVideo:     [B, 16,  T/4, H/8,  W/8]   scale ≈ 0.476986
# Wan2.1:           [B, 16,  T/4, H/8,  W/8]   scale ≈ 1.0   (dense DiT)
# Wan2.2 14B:       [B, 16,  T/4, H/8,  W/8]   scale ≈ 1.0   (high/low-noise expert pair)
# Wan2.2 TI2V 5B:   [B, 48,  T/4, H/16, W/16]  scale ≈ 1.0
# HunyuanVideo 1.5: [B, 32,  T/4, H/16, W/16]
# CogVideoX:        [B, 16,  T/4, H/8,  W/8]   scale ≈ 1.15258426
#
# ============================================================

from radiance.config.constants import VERSION as __version__  # noqa: E402

import json
import math
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Lazy torch import
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ---------------------------------------------------------------------------
# Per-model latent specifications
# ---------------------------------------------------------------------------

# Each entry: {channels, spatial_compression, temporal_compression,
#              latent_scale, mean (per-ch or scalar), std (per-ch or scalar)}
# Mean/std are empirical estimates from public model cards and papers.

_MODEL_SPECS: Dict[str, Dict] = {
    "SD-VAE (4ch)": {
        "channels": 4,
        "spatial_compression": 8,
        "temporal_compression": 1,
        "latent_scale": 0.18215,
        "mean": [0.0, 0.0, 0.0, 0.0],
        "std":  [1.0, 1.0, 1.0, 1.0],
        "temporal": False,
        "description": "Stable Diffusion 1.x / 2.x VAE",
    },
    "SDXL-VAE (4ch)": {
        "channels": 4,
        "spatial_compression": 8,
        "temporal_compression": 1,
        "latent_scale": 0.13025,
        "mean": [0.0, 0.0, 0.0, 0.0],
        "std":  [1.0, 1.0, 1.0, 1.0],
        "temporal": False,
        "description": "Stable Diffusion XL VAE",
    },
    "LTX-Video (128ch)": {
        "channels": 128,
        "spatial_compression": 32,
        "temporal_compression": 8,
        "latent_scale": 1.0,
        # Per-channel stats not published; empirical approximations
        "mean": 0.0,
        "std":  1.0,
        "temporal": True,
        "description": "Lightricks LTX-2.x / LTX-2.3",
    },
    "HunyuanVideo (16ch)": {
        "channels": 16,
        "spatial_compression": 8,
        "temporal_compression": 4,
        "latent_scale": 0.476986,
        "mean": 0.0,
        "std":  1.0,
        "temporal": True,
        "description": "Tencent HunyuanVideo",
    },
    "Wan2.1 (16ch)": {
        "channels": 16,
        "spatial_compression": 8,
        "temporal_compression": 4,
        "latent_scale": 1.0,
        "mean": 0.0,
        "std":  1.0,
        "temporal": True,
        "description": "Alibaba Wan2.1 dense DiT",
    },
    "Wan2.2-T2V-14B (16ch)": {
        "channels": 16,
        "spatial_compression": 8,
        "temporal_compression": 4,
        "latent_scale": 1.0,
        "mean": 0.0,
        "std":  1.0,
        "temporal": True,
        "moe": True,
        "description": "Alibaba Wan2.2 T2V 14B, two experts (high-noise and low-noise checkpoints)",
    },
    "Wan2.2-I2V-14B (16ch)": {
        "channels": 16,
        "spatial_compression": 8,
        "temporal_compression": 4,
        "latent_scale": 1.0,
        "mean": 0.0,
        "std":  1.0,
        "temporal": True,
        "moe": True,
        "description": "Alibaba Wan2.2 I2V 14B, two experts (high-noise and low-noise checkpoints)",
    },
    "Wan2.2-TI2V-5B (48ch)": {
        "channels": 48,
        "spatial_compression": 16,
        "temporal_compression": 4,
        "latent_scale": 1.0,
        "mean": 0.0,
        "std":  1.0,
        "temporal": True,
        "description": "Alibaba Wan2.2 TI2V 5B dense, Wan2.2 VAE",
    },
    "HunyuanVideo-1.5 (32ch)": {
        "channels": 32,
        "spatial_compression": 16,
        "temporal_compression": 4,
        "latent_scale": 1.0,
        "mean": 0.0,
        "std":  1.0,
        "temporal": True,
        "description": "Tencent HunyuanVideo 1.5",
    },
    "CogVideoX (16ch)": {
        "channels": 16,
        "spatial_compression": 8,
        "temporal_compression": 4,   # CogVideoX 3D VAE: 4x temporal, (T-1)/4+1 latent frames
        "latent_scale": 1.15258426,
        "mean": 0.0,
        "std":  1.0,
        "temporal": True,
        "description": "Zhipu AI CogVideoX-5B",
    },
    "Mochi-1 (12ch)": {
        "channels": 12,
        "spatial_compression": 8,
        "temporal_compression": 6,
        "latent_scale": 1.0,
        "mean": 0.0,
        "std":  1.0,
        "temporal": True,
        "description": "Genmo Mochi-1",
    },
}

MODEL_NAMES = list(_MODEL_SPECS.keys())


def _get_spec(name: str) -> Dict:
    """A copy: callers update() it with dit_config and must not edit the table."""
    return dict(_MODEL_SPECS.get(name, _MODEL_SPECS["SD-VAE (4ch)"]))


# ===========================================================================
# Registration
# ===========================================================================

NODE_CLASS_MAPPINGS = {}

NODE_DISPLAY_NAME_MAPPINGS = {}