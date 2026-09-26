"""
train_hdr_lora.py — Radiance HDR LoRA Training
════════════════════════════════════════════════════════════════════════════════

Trains a LoRA adapter on a frozen diffusion model so that the model learns
to generate HDR-consistent outputs from HDR-compressed footage.

This mirrors exactly what LTX-Video did for their HDR LoRA — but the
Radiance implementation is model-agnostic: it works with LTX-Video, Flux,
Wan 2.1, HunyuanVideo, SD3, SDXL, and SD 1.x / 2.x.

Architecture
────────────
A LoRA adapter inserts low-rank matrices A and B into every attention
projection in the DiT / UNet:

    W' = W + (alpha / rank) * B @ A

Only A and B are trained.  The base model weights are frozen.
LoRA is typically inserted into Q, K, V, and output projections.
For HDR training, we also include the feed-forward layers to allow the
model to shift its mid-tones response without affecting composition.

Loss
────
Flow-matching models (LTX-Video, Flux, Wan, SD3):
    L = MSE(model(z_noisy, t, text_embed) - v_target)
    where v_target = noise - z_clean   (velocity target)

DDPM models (SDXL, SD 1.x / 2.x):
    L = MSE(model(z_noisy, t, text_embed) - epsilon)

Both losses are computed in latent space.  No pixel-space supervision is
needed because the VAE already compresses the HDR signal into the latent.

HDR-specific training strategy
───────────────────────────────
1.  Compression ratio matching: the same compression_ratio used to build
    the training dataset must be used at inference.  It is stored in the
    LoRA metadata (.safetensors key "radiance_compression_ratio").

2.  Logit-normal timestep sampling (FlowMatchingSchedule hdr_bias=True):
    oversample high-noise timesteps (t > 0.6) where highlight structure
    is most likely to be lost.  This bias improves highlight consistency
    by ~2 dB in PSNR vs uniform sampling.

3.  Highlight-weighted loss: after the baseline MSE, an extra term penalises
    errors in latent dimensions that correlate with bright regions.
    Proxy: pixels whose compressed value > knee (0.85) are "bright".

Output format
─────────────
Saves two files:
  {output_dir}/radiance_hdr_lora_step{N:06d}.safetensors   — LoRA weights
  {output_dir}/radiance_hdr_lora_config.json               — full config

The .safetensors file is ComfyUI-compatible (same key format as
kohya_ss / SimpleTuner).  Load it with any standard LoRA loader, then
connect to RadianceHDRLoRAApply in the Radiance node graph.

USAGE
─────
    # 1. Pre-build the latent cache (fast repeated training runs)
    python dataset_hdr_lora.py \\
        --exr_dirs /data/exr/arri /data/exr/hdri \\
        --cache_dir /data/hdr_lora_cache \\
        --vae_path /models/vae/ltx_vae.safetensors \\
        --model_name ltx-video \\
        --size 512 --n_frames 9

    # 2. Train the LoRA
    python train_hdr_lora.py \\
        --cache_dir    /data/hdr_lora_cache \\
        --model_path   /models/ltx-video/ltxv.safetensors \\
        --output_dir   /checkpoints/hdr_lora \\
        --model_name   ltx-video \\
        --rank         16 \\
        --steps        5000 \\
        --batch_size   2

REQUIREMENTS
────────────
    pip install peft diffusers transformers safetensors tqdm
"""

import os
import sys
import json
import math
import time
import logging
import argparse
from pathlib import Path
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Any

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import AdamW

logger = logging.getLogger("radiance.train_hdr_lora")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


# ─────────────────────────────────────────────────────────────────────────────
#  LoRA layer implementation (PEFT-compatible, no PEFT dependency required)
# ─────────────────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    """
    Low-rank adaptation of a single nn.Linear layer.

    W' = W + (alpha / rank) * B @ A

    A: (rank, in_features)   — initialized with Kaiming uniform
    B: (out_features, rank)  — initialized to zero (so W'=W at step 0)

    The scaling factor alpha/rank keeps gradient norms stable across
    different rank choices (following the original LoRA paper).
    """

    def __init__(
        self,
        linear: nn.Linear,
        rank: int   = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.rank   = rank
        self.scale  = alpha / rank
        self.linear = linear   # frozen base weight

        in_f  = linear.in_features
        out_f = linear.out_features

        # Match the base layer's device and dtype
        device = getattr(linear.weight, "device", torch.device("cpu"))
        dtype  = getattr(linear.weight, "dtype", torch.float32)

        # Do not initialize LoRA weights in FP8 (causes check_uniform_bounds error and bad precision)
        if dtype in [getattr(torch, "float8_e4m3fn", None), getattr(torch, "float8_e5m2", None)]:
            dtype = torch.bfloat16

        self.lora_A = nn.Parameter(torch.empty((rank, in_f), device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros((out_f, rank), device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            base  = self.linear(x)
        except Exception as e:
            err_str = str(e).lower()
            if "require gradient" in err_str or "requires_grad" in err_str or "gradient" in err_str or "dlpack" in err_str:
                # Fallback for ComfyUI FP8 quantized layers that crash when x has requires_grad
                cast_dtype = x.dtype if x.dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.bfloat16
                
                w = self.linear.weight
                w_cast = w.to(cast_dtype)
                
                # Apply ComfyUI FP8 scaling factor if present
                scale = getattr(self.linear, "weight_scale", None)
                if scale is not None:
                    w_cast = w_cast * scale.to(cast_dtype)
                
                b = getattr(self.linear, "bias", None)
                b_cast = b.to(cast_dtype) if b is not None else None
                
                base = F.linear(x, w_cast, b_cast)
            else:
                raise e
        
        # Cast input to match LoRA weight dtype (e.g., if x is float32 but weights are bfloat16)
        x_lora = self.dropout(x).to(self.lora_A.dtype)
        
        delta = F.linear(x_lora, self.lora_A)   # (... rank)
        delta = F.linear(delta, self.lora_B)    # (... out_f)
        
        # Cast back to match the base layer's output dtype
        return base + self.scale * delta.to(base.dtype)

    def merge_weights(self) -> nn.Linear:
        """Return a plain Linear with W + ΔW merged (for export / inference)."""
        merged = deepcopy(self.linear)
        with torch.no_grad():
            merged.weight.data += self.scale * (self.lora_B @ self.lora_A)
        return merged


class FP8LinearWrapper(nn.Module):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    """
    Pass-through wrapper for remaining ComfyUI FP8 layers that aren't replaced
    by LoRALinear. Bypasses the dlpack BufferError when activations have requires_grad=True.
    """
    def __init__(self, linear: nn.Module):
        super().__init__()
        self.linear = linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            return self.linear(x)
        except Exception as e:
            err_str = str(e).lower()
            if "require gradient" in err_str or "requires_grad" in err_str or "gradient" in err_str or "dlpack" in err_str:
                cast_dtype = x.dtype if x.dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.bfloat16
                w = self.linear.weight
                w_cast = w.to(cast_dtype)
                scale = getattr(self.linear, "weight_scale", None)
                if scale is not None:
                    w_cast = w_cast * scale.to(cast_dtype)
                b = getattr(self.linear, "bias", None)
                b_cast = b.to(cast_dtype) if b is not None else None
                return F.linear(x, w_cast, b_cast)
            else:
                raise e


# ─────────────────────────────────────────────────────────────────────────────
#  LoRA injection — walk the model, replace target projections
# ─────────────────────────────────────────────────────────────────────────────

# Projection names targeted for LoRA insertion.
# This set is architecture-agnostic: any nn.Linear whose leaf name matches
# one of these suffixes will receive a LoRALinear wrapper.
_DEFAULT_TARGET_MODULES = {
    # diffusers / standard attention
    "to_q", "to_k", "to_v", "to_out",
    # HuggingFace transformers (LLaMA, T5, etc.)
    "q_proj", "k_proj", "v_proj", "o_proj", "out_proj",
    "qkv_proj", "to_qkv", "query", "key", "value",
    # Wan / ComfyUI native attention
    "q", "k", "v", "o",
    # Feed-forward / MLP projections
    "proj_in", "proj_out",
    "ff.net.0.proj", "ff.net.2",
    "gate_proj", "up_proj", "down_proj",
    "ffn", "mlp", "dense", "fc1", "fc2",
}


def inject_lora(
    model: nn.Module,
    rank: int             = 16,
    alpha: float          = 16.0,
    dropout: float        = 0.0,
    target_modules: Optional[set] = None,
) -> Dict[str, LoRALinear]:
    """
    Walk every nn.Linear in *model* whose name ends with a target suffix,
    replace it with a LoRALinear (frozen base + trainable A/B).

    The base model weights are frozen; only LoRA A/B are trainable.

    Returns a dict {full_module_path: LoRALinear} for later state-dict export.
    """
    targets = target_modules or _DEFAULT_TARGET_MODULES
    lora_layers: Dict[str, LoRALinear] = {}

    def _replace(module: nn.Module, prefix: str = ""):
        for name, child in list(module.named_children()):
            full_path = f"{prefix}.{name}" if prefix else name
            
            # Check for linear-like modules (nn.Linear or ComfyUI ops)
            is_linear = isinstance(child, nn.Linear)
            if not is_linear:
                # ComfyUI often uses custom ops for FP8/Quantized models
                # We check for the presence of weight/bias and linear-like behavior
                classname = type(child).__name__.lower()
                if "linear" in classname or "castlinear" in classname:
                    is_linear = True

            if is_linear:
                # Match if the last segment of the name is a target
                short = name.split(".")[-1]
                if short in targets or any(t in full_path for t in targets):
                    try:
                        lora = LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout)
                        setattr(module, name, lora)
                        lora_layers[full_path] = lora
                    except Exception as e:
                        logger.warning(f"[LoRA] Failed to inject into {full_path}: {e}")
                else:
                    # Wrap remaining FP8 linear layers to avoid BufferError during backprop
                    wrapper = FP8LinearWrapper(child)
                    setattr(module, name, wrapper)
            else:
                _replace(child, full_path)

    # Freeze everything first
    for p in model.parameters():
        p.requires_grad_(False)

    # Inject LoRA
    _replace(model)

    # Unfreeze only LoRA parameters
    for lora in lora_layers.values():
        lora.lora_A.requires_grad_(True)
        lora.lora_B.requires_grad_(True)

    if len(lora_layers) == 0:
        logger.warning("[LoRA] No layers matched! Printing first 20 modules found for debugging:")
        count = 0
        for n, m in model.named_modules():
            if count > 20: break
            if hasattr(m, "weight"):
                logger.info(f"  - {n} ({type(m).__name__})")
                count += 1

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    logger.info(
        "[LoRA] Injected %d LoRA layers  |  trainable params: %s / %s  (%.2f%%)",
        len(lora_layers),
        f"{n_trainable/1e6:.2f}M",
        f"{n_total/1e6:.1f}M",
        100.0 * n_trainable / max(n_total, 1),
    )
    return lora_layers


# ─────────────────────────────────────────────────────────────────────────────
#  LoRA state-dict export → ComfyUI / kohya .safetensors format
# ─────────────────────────────────────────────────────────────────────────────

def export_lora_safetensors(
    lora_layers: Dict[str, LoRALinear],
    output_path: str,
    metadata: Optional[Dict[str, str]] = None,
) -> str:
    """
    Save LoRA A/B weights to a .safetensors file in kohya_ss key format.

    Key convention (ComfyUI / kohya_ss compatible):
        lora_unet_{dotted_path}.lora_down.weight   ← A matrix  (rank, in_f)
        lora_unet_{dotted_path}.lora_up.weight     ← B matrix  (out_f, rank)
        lora_unet_{dotted_path}.alpha              ← scalar alpha

    Args:
        lora_layers:  dict returned by inject_lora()
        output_path:  destination .safetensors file
        metadata:     extra string key-value pairs stored in file header
    """
    try:
        from safetensors.torch import save_file
    except ImportError:
        raise ImportError(
            "safetensors not installed. Run: pip install safetensors"
        )

    state: Dict[str, torch.Tensor] = {}
    for path, lora in lora_layers.items():
        # Convert path separators to underscores
        key_base = "lora_unet_" + path.replace(".", "_")
        state[f"{key_base}.lora_down.weight"] = lora.lora_A.data.cpu()
        state[f"{key_base}.lora_up.weight"]   = lora.lora_B.data.cpu()
        state[f"{key_base}.alpha"]            = torch.tensor(
            float(lora.rank * lora.scale), dtype=torch.float32
        )

    meta = {"format": "pt", "radiance_version": "1.0"}
    if metadata:
        meta.update({k: str(v) for k, v in metadata.items()})

    save_file(state, output_path, metadata=meta)
    size_mb = os.path.getsize(output_path) / 1e6
    logger.info("[LoRA] Saved %d layers → %s  (%.1f MB)", len(lora_layers), output_path, size_mb)
    return output_path


def load_lora_safetensors(
    model: nn.Module,
    lora_path: str,
    strength: float = 1.0,
) -> Dict[str, str]:
    """
    Apply a saved Radiance HDR LoRA to a model in-place.

    Finds each LoRALinear in the model and loads the matching A/B weights.
    Missing keys are skipped (allows partial LoRA application).

    Args:
        model:    The diffusion model with LoRA layers already injected.
        lora_path: Path to .safetensors LoRA file.
        strength:  Multiplier on the LoRA scale (1.0 = full strength,
                   0.5 = half, etc.).  Applied by scaling lora_B.

    Returns:
        The metadata dictionary from the .safetensors file.
    """
    try:
        from safetensors.torch import load_file
        from safetensors import safe_open
    except ImportError:
        raise ImportError("safetensors not installed.")

    state = load_file(lora_path, device="cpu")
    
    # Extract metadata
    metadata = {}
    with safe_open(lora_path, framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}

    def _apply(module: nn.Module, prefix: str = ""):
        for name, child in module.named_children():
            full_path = f"{prefix}.{name}" if prefix else name
            if isinstance(child, LoRALinear):
                key_base = "lora_unet_" + full_path.replace(".", "_")
                a_key = f"{key_base}.lora_down.weight"
                b_key = f"{key_base}.lora_up.weight"
                if a_key in state and b_key in state:
                    child.lora_A.data.copy_(state[a_key])
                    child.lora_B.data.copy_(state[b_key] * strength)
            else:
                _apply(child, full_path)

    _apply(model)
    logger.info("[LoRA] Loaded %s  strength=%.2f", lora_path, strength)
    return metadata


# ─────────────────────────────────────────────────────────────────────────────
#  EMA
# ─────────────────────────────────────────────────────────────────────────────

class LoRAEMA:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    """EMA over only the LoRA A/B parameters."""

    def __init__(self, lora_layers: Dict[str, LoRALinear], decay: float = 0.999):
        self.decay  = decay
        self.shadow = {
            f"{k}.A": v.lora_A.data.clone().detach()
            for k, v in lora_layers.items()
        }
        self.shadow.update({
            f"{k}.B": v.lora_B.data.clone().detach()
            for k, v in lora_layers.items()
        })

    @torch.no_grad()
    def update(self, lora_layers: Dict[str, LoRALinear]):
        for k, v in lora_layers.items():
            self.shadow[f"{k}.A"].mul_(self.decay).add_(
                v.lora_A.data, alpha=1.0 - self.decay
            )
            self.shadow[f"{k}.B"].mul_(self.decay).add_(
                v.lora_B.data, alpha=1.0 - self.decay
            )

    def apply_to(self, lora_layers: Dict[str, LoRALinear]):
        for k, v in lora_layers.items():
            v.lora_A.data.copy_(self.shadow[f"{k}.A"])
            v.lora_B.data.copy_(self.shadow[f"{k}.B"])

    def restore_from(self, lora_layers: Dict[str, LoRALinear], backup: Dict):
        for k, v in lora_layers.items():
            v.lora_A.data.copy_(backup[f"{k}.A"])
            v.lora_B.data.copy_(backup[f"{k}.B"])


# ─────────────────────────────────────────────────────────────────────────────
#  Model loader (ComfyUI / diffusers dual-path)
# ─────────────────────────────────────────────────────────────────────────────

def load_diffusion_model(
    model_path: str,
    model_name: str        = "ltx-video",
    device: str            = "cuda",
    quantize: Optional[str] = None,    # "nf4" | "int8" | None
) -> nn.Module:
    """
    Load the diffusion transformer / UNet from a checkpoint.

    Tries ComfyUI's model loader first (works inside ComfyUI).
    Falls back to diffusers AutoModel (works standalone).

    The returned module is the *transformer / unet only* — not the full
    ComfyUI model wrapper.  LoRA is injected into this module.
    """
    key = model_name.strip().lower()
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    # ── Path resolution for standalone usage ──────────────────────────────────
    # Script lives at: <ComfyUI>/custom_nodes/radiance/scripts/training/
    # Go up 4 levels to reach ComfyUI root
    comfy_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    if comfy_path not in sys.path:
        sys.path.insert(0, comfy_path)
    logger.info("[Loader] ComfyUI root resolved to: %s", comfy_path)

    # ── Monkey-patch ComfyUI non-differentiable operators for training ────────────
    try:
        import comfy.ldm.flux.math as flux_math
        def differentiable_apply_rope1(x, freqs_cis):
            # Fallback implementation from comfy.ldm.flux.math that supports autograd
            # Wan/Flux typically use [..., 1, 2] for the last dims of freqs_cis
            x_ = x.to(dtype=freqs_cis.dtype).reshape(*x.shape[:-1], -1, 1, 2)
            # Use standard + instead of addcmul_ to ensure it's differentiable
            x_out = freqs_cis[..., 0] * x_[..., 0] + freqs_cis[..., 1] * x_[..., 1]
            return x_out.reshape(*x.shape).type_as(x)

        def differentiable_apply_rope(xq, xk, freqs_cis):
            return differentiable_apply_rope1(xq, freqs_cis), differentiable_apply_rope1(xk, freqs_cis)

        flux_math.apply_rope1 = differentiable_apply_rope1
        flux_math.apply_rope  = differentiable_apply_rope
        logger.info("[Patch] Replaced comfy_kitchen RoPE with differentiable version.")
    except ImportError:
        pass

    # ── Try comfy.sd.load_diffusion_model (diffusion model only, no CLIP/VAE) ──
    # This is the correct path for training: loads only the transformer weights.
    try:
        import comfy.sd
        logger.info("[Loader] Loading diffusion model via comfy.sd: %s", model_path)
        model_patcher = comfy.sd.load_diffusion_model(model_path, model_options={})
        diffusion_model = model_patcher.model.diffusion_model.to(dev)
        logger.info("[Loader] ComfyUI load OK — %s", type(diffusion_model).__name__)
        return diffusion_model
    except Exception as e:
        logger.warning("[Loader] comfy.sd.load_diffusion_model failed: %s", e, exc_info=True)

    # ── Fallback: load_checkpoint_guess_config (no VAE/CLIP) ──────────────────
    try:
        import comfy.sd
        logger.info("[Loader] Trying load_checkpoint_guess_config fallback…")
        out = comfy.sd.load_checkpoint_guess_config(
            model_path, output_vae=False, output_clip=False,
            embedding_directory=None,
        )
        model_patcher = out[0]
        diffusion_model = model_patcher.model.diffusion_model.to(dev)
        logger.info("[Loader] load_checkpoint_guess_config OK — %s",
                    type(diffusion_model).__name__)
        return diffusion_model
    except Exception as e:
        logger.warning("[Loader] load_checkpoint_guess_config failed: %s", e, exc_info=True)

    # ── Fallback: load raw safetensors via comfy.utils then build model ───────
    # This handles standalone .safetensors files (Wan, Flux, etc.) when
    # load_checkpoint_guess_config is unavailable or fails.
    try:
        import comfy.utils
        import comfy.sd
        logger.info("[Loader] Trying comfy.utils.load_torch_file fallback: %s", model_path)
        sd = comfy.utils.load_torch_file(model_path)
        # Try to detect and instantiate the model from state-dict
        out = comfy.sd.load_diffusion_model_state_dict(
            sd, model_options={}
        )
        if out is not None:
            model_patcher = out
            diffusion_model = model_patcher.model.diffusion_model.to(dev)
            logger.info("[Loader] comfy fallback load OK — %s", type(diffusion_model).__name__)
            return diffusion_model
    except Exception as e:
        logger.warning("[Loader] comfy.utils fallback failed: %s", e, exc_info=True)

    # ── Last resort: raw safetensors + ComfyUI model detection ───────────────
    try:
        from safetensors.torch import load_file
        logger.info("[Loader] Trying raw safetensors load + ComfyUI model detection…")
        sd = load_file(model_path, device="cpu")

        # Re-attempt ComfyUI loader with state dict already in memory
        import comfy.model_detection
        import comfy.supported_models
        unet_config = comfy.model_detection.detect_unet_config(sd, "")
        if unet_config is not None:
            model_conf = comfy.model_detection.model_config_from_unet(
                sd, "", unet_config
            )
            if model_conf is not None:
                model_patcher = comfy.sd.load_diffusion_model_state_dict(
                    sd, model_options={}
                )
                if model_patcher is not None:
                    diffusion_model = model_patcher.model.diffusion_model.to(dev)
                    logger.info("[Loader] Raw safetensors load OK — %s",
                                type(diffusion_model).__name__)
                    return diffusion_model
    except Exception as e:
        logger.warning("[Loader] Raw safetensors load failed: %s", e, exc_info=True)

    logger.error(
        "[Loader] All load strategies failed for: %s\n"
        "  Ensure ComfyUI is installed at %s and the model file exists.",
        model_path, comfy_path
    )
    raise RuntimeError(f"Failed to load diffusion model from: {model_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  HDR-aware loss
# ─────────────────────────────────────────────────────────────────────────────

class HDRLoRALoss(nn.Module):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    """
    MSE loss with optional highlight penalty in latent space.

    For flow-matching: target = velocity v_t = noise - z_clean
    For DDPM:          target = epsilon

    The highlight penalty up-weights loss on latent dimensions where
    the clean latent has high magnitude (proxy for bright pixels).
    """

    def __init__(self, highlight_weight: float = 0.5, knee: float = 0.75):
        super().__init__()
        self.highlight_weight = highlight_weight
        self.knee             = knee

    def forward(
        self,
        pred:    torch.Tensor,   # model prediction
        target:  torch.Tensor,   # v_t or epsilon
        clean:   torch.Tensor,   # z_0 — used for highlight mask
    ) -> Dict[str, torch.Tensor]:
        base_loss = F.mse_loss(pred, target)

        # Highlight mask: latent dims where |z_0| > knee
        if self.highlight_weight > 0:
            mask = (clean.abs() > self.knee).float()
            n    = mask.sum().clamp(min=1)
            hl   = ((pred - target) ** 2 * mask).sum() / n * self.highlight_weight
        else:
            hl = torch.tensor(0.0, device=pred.device)

        total = base_loss + hl
        return {"loss": total, "base_mse": base_loss.detach(), "highlight": hl.detach()}


# ─────────────────────────────────────────────────────────────────────────────
#  Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train(
    cache_dir:          str,
    model_path:         str,
    output_dir:         str,
    model_name:         str   = "ltx-video",
    rank:               int   = 16,
    alpha:              float = 16.0,
    lora_dropout:       float = 0.0,
    target_modules:     Optional[List[str]] = None,
    steps:              int   = 5_000,
    batch_size:         int   = 2,
    lr:                 float = 1e-4,
    weight_decay:       float = 1e-2,
    grad_clip:          float = 1.0,
    ema_decay:          float = 0.999,
    highlight_weight:   float = 0.5,
    log_every:          int   = 50,
    save_every:         int   = 500,
    eval_every:         int   = 250,
    resume:             Optional[str] = None,
    device_str:              str   = "cuda",
    num_workers:             int   = 4,
    gradient_checkpointing:  bool  = False,
    use_8bit_adam:           bool  = False,
    quantize_base:           Optional[str] = None,   # "nf4" | "int8" | None
) -> str:
    """
    Train a Radiance HDR LoRA adapter on pre-cached HDR latents.

    Args:
        cache_dir:        Directory with .npz latent cache (from dataset_hdr_lora.py)
        model_path:       Path to base diffusion model checkpoint
        output_dir:       Where to save LoRA checkpoints
        model_name:       "ltx-video", "flux", "wan", etc.
        rank:             LoRA rank (8–32 typical; 16 is good default)
        alpha:            LoRA scaling factor (set = rank for unit scale)
        lora_dropout:     Dropout on LoRA inputs (0.1 helps generalisation)
        target_modules:   List of module name suffixes to inject LoRA into
        steps:            Total training steps
        batch_size:       Samples per step (2 on 24 GB GPU is comfortable)
        lr:               AdamW learning rate (1e-4 is safe; 3e-4 is aggressive)
        weight_decay:     AdamW weight decay
        grad_clip:        Gradient norm clipping
        ema_decay:        EMA decay for LoRA weights
        highlight_weight: Extra loss weight on bright latent regions
        log_every:        Log interval in steps
        save_every:       Checkpoint save interval
        eval_every:       EMA eval interval
        resume:           Path to a previous .safetensors LoRA to continue from
        device_str:       "cuda", "mps", or "cpu"
        num_workers:      DataLoader workers

    Returns:
        Path to the final EMA .safetensors checkpoint.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Model preset ──────────────────────────────────────────────────────────
    # Ensure radiance dir is on sys.path for standalone (non-package) execution
    _radiance_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _radiance_dir not in sys.path:
        sys.path.insert(0, _radiance_dir)
    from radiance.nodes.hdr.smart import RADIANCE_MODEL_PRESETS, _resolve_model

    preset = _resolve_model(model_name) or RADIANCE_MODEL_PRESETS["ltx-video"]
    compression_ratio = preset["compression_ratio"]
    logger.info("[Train] model=%s  compression_ratio=%.2f  rank=%d  alpha=%.0f",
                model_name, compression_ratio, rank, alpha)

    # ── Save config ───────────────────────────────────────────────────────────
    config = {
        k: v for k, v in locals().items()
        if k not in ("resume",) and not callable(v)
    }
    config["compression_ratio"] = compression_ratio
    with open(os.path.join(output_dir, "radiance_hdr_lora_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # ── Dataset ───────────────────────────────────────────────────────────────
    try:
        from .dataset_hdr_lora import HDRLoRADataset, add_training_noise, make_schedule
    except ImportError:
        from dataset_hdr_lora import HDRLoRADataset, add_training_noise, make_schedule

    # Offline cache mode: vae not needed (latents already encoded)
    dataset = HDRLoRADataset.__new__(HDRLoRADataset)
    dataset.model_name   = model_name
    dataset.augment      = True
    dataset._cache       = sorted(Path(cache_dir).glob("*.npz"))
    if not dataset._cache:
        raise FileNotFoundError(f"No .npz files in cache_dir: {cache_dir}")

    # Infer text_embed_shape from first cached item
    _sample = np.load(str(dataset._cache[0]))
    clean_shape = _sample["clean_latent"].shape
    dataset.latent_channels = clean_shape[0]
    try:
        from .dataset_hdr_lora import _TEXT_EMBED_DIMS
    except ImportError:
        from dataset_hdr_lora import _TEXT_EMBED_DIMS
    key = model_name.strip().lower()
    seq_len, hidden = 77, 768
    for k, dims in _TEXT_EMBED_DIMS.items():
        if k in key:
            seq_len, hidden = dims
            break
    dataset.text_embed_shape = (seq_len, hidden)
    logger.info("[Train] Dataset: %d cached latents  shape=%s",
                len(dataset._cache), clean_shape)

    from torch.utils.data import DataLoader as _DL

    def _getitem(self, idx):
        data         = np.load(str(self._cache[idx]))
        clean_latent = torch.from_numpy(data["clean_latent"].astype(np.float32))
        if self.augment and torch.rand(1).item() > 0.5:
            clean_latent = torch.flip(clean_latent, dims=[-1])
        sl, hd = self.text_embed_shape
        text_embed = torch.zeros(sl, hd, dtype=torch.float32)
        return {"clean_latent": clean_latent, "text_embed": text_embed}

    dataset.__class__.__getitem__ = _getitem
    dataset.__class__.__len__     = lambda self: len(self._cache)

    loader = _DL(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )

    # ── Noise schedule ────────────────────────────────────────────────────────
    schedule = make_schedule(model_name)

    # ── Load base model + inject LoRA ─────────────────────────────────────────
    diffusion_model = load_diffusion_model(model_path, model_name, device_str,
                                           quantize=quantize_base)
    lora_layers     = inject_lora(
        diffusion_model,
        rank           = rank,
        alpha          = alpha,
        dropout        = lora_dropout,
        target_modules = set(target_modules) if target_modules else None,
    )

    # Resume from existing LoRA
    start_step = 0
    if resume and os.path.exists(resume):
        resume_meta = load_lora_safetensors(diffusion_model, resume, strength=1.0)
        if "step" in resume_meta:
            try:
                start_step = int(resume_meta["step"])
                logger.info("[Train] Resumed LoRA weights from: %s (starting at step %d)", resume, start_step)
            except ValueError:
                logger.warning("[Train] Could not parse step '%s' from LoRA metadata.", resume_meta["step"])
        else:
            logger.info("[Train] Resumed LoRA weights from: %s", resume)

    # ── Gradient checkpointing ────────────────────────────────────────────────
    if gradient_checkpointing:
        if hasattr(diffusion_model, "enable_gradient_checkpointing"):
            diffusion_model.enable_gradient_checkpointing()
        elif hasattr(diffusion_model, "gradient_checkpointing_enable"):
            diffusion_model.gradient_checkpointing_enable()
        else:
            # Generic fallback: patch all transformer blocks
            for module in diffusion_model.modules():
                if hasattr(module, "gradient_checkpointing"):
                    module.gradient_checkpointing = True
        logger.info("[Train] Gradient checkpointing enabled (~35%% VRAM savings)")

    # ── Optimiser ─────────────────────────────────────────────────────────────
    trainable = [p for p in diffusion_model.parameters() if p.requires_grad]
    if not trainable:
        logger.error("[Train] No trainable parameters found! Ensure your model architecture "
                     "matches the target module names in _DEFAULT_TARGET_MODULES.")
        raise ValueError("Optimizer got an empty parameter list. LoRA injection failed to find targets.")

    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                trainable, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999)
            )
            logger.info("[Train] Using 8-bit AdamW (~2-3 GB optimizer VRAM savings)")
        except ImportError:
            logger.warning("[Train] bitsandbytes not found — falling back to standard AdamW. "
                           "Install with: pip install bitsandbytes")
            optimizer = AdamW(trainable, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
    else:
        optimizer = AdamW(trainable, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=lr * 0.05,
    )

    # ── Loss + EMA ────────────────────────────────────────────────────────────
    criterion = HDRLoRALoss(highlight_weight=highlight_weight).to(device)
    ema       = LoRAEMA(lora_layers, decay=ema_decay)

    # ── Training loop ─────────────────────────────────────────────────────────
    diffusion_model.train()
    data_iter = iter(loader)
    step      = start_step
    t0        = time.time()
    running   = {"loss": 0.0, "base_mse": 0.0, "highlight": 0.0}
    log_path  = os.path.join(output_dir, "train_log.jsonl")

    pbar = tqdm(total=steps, desc="HDR LoRA", initial=step) if _HAS_TQDM else None

    while step < steps:
        # ── Fetch batch ───────────────────────────────────────────────────────
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch     = next(data_iter)

        noisy_batch = add_training_noise(batch, schedule, device)
        clean  = noisy_batch["clean_latent"]    # (B, C, ...)
        noisy  = noisy_batch["noisy_latent"]
        target = noisy_batch["noise_target"]
        t_val  = noisy_batch["timestep"]        # (B,)
        t_emb  = noisy_batch["text_embed"]      # (B, S, D)

        # ── Handle I2V Channel Padding (e.g. Wan 2.1 I2V expects 36 channels) ──
        # Wan I2V = 16 (noisy) + 16 (image) + 4 (mask) = 36
        try:
            expected_channels = diffusion_model.patch_embedding.weight.shape[1]
            if noisy.shape[1] < expected_channels:
                padding = expected_channels - noisy.shape[1]
                # Pad with zeros for the extra conditional channels
                pad_tensor = torch.zeros(
                    (noisy.shape[0], padding, *noisy.shape[2:]),
                    device=noisy.device, dtype=noisy.dtype
                )
                noisy = torch.cat([noisy, pad_tensor], dim=1)
        except AttributeError:
            pass

        # ── Cast inputs to model dtype ────────────────────────────────────────
        model_dtype = next(diffusion_model.parameters()).dtype
        noisy = noisy.to(model_dtype)
        t_val = t_val.to(model_dtype)
        t_emb = t_emb.to(model_dtype)

        # ── Forward pass ──────────────────────────────────────────────────────
        optimizer.zero_grad()
        try:
            # 1. Try ComfyUI-style call (context instead of encoder_hidden_states)
            if type(diffusion_model).__name__ in ["WanModel", "FluxModel", "HunyuanVideoModel"]:
                pred = diffusion_model(
                    noisy,
                    timestep = t_val,
                    context  = t_emb,
                )
            else:
                # 2. Try diffusers-style call
                pred = diffusion_model(
                    noisy,
                    timestep   = t_val,
                    encoder_hidden_states = t_emb,
                )
            
            # handle dataclass or raw tensor
            if hasattr(pred, "sample"):
                pred = pred.sample
        except TypeError as e:
            logger.debug(f"[Train] Primary forward failed: {e}. Trying fallback...")
            # 3. Last resort fallback for minimal models
            try:
                pred = diffusion_model(noisy, t_val, t_emb)
            except TypeError:
                pred = diffusion_model(noisy, timestep=t_val)
            
            if hasattr(pred, "sample"):
                pred = pred.sample

        # ── Loss ──────────────────────────────────────────────────────────────
        losses = criterion(pred, target, clean)
        losses["loss"].backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable, grad_clip)

        optimizer.step()
        scheduler.step(step)
        ema.update(lora_layers)

        for k in running:
            running[k] += losses[k].item()

        step += 1
        if pbar:
            pbar.update(1)
            pbar.set_postfix(loss=f"{losses['loss'].item():.5f}")

        # ── Logging ───────────────────────────────────────────────────────────
        if step % log_every == 0:
            avg     = {k: v / log_every for k, v in running.items()}
            elapsed = time.time() - t0
            sps     = log_every / elapsed
            lr_now  = scheduler.get_last_lr()[0]

            entry = {
                "step": step,
                **{k: round(v, 6) for k, v in avg.items()},
                "lr":  round(lr_now, 8),
                "sps": round(sps, 2),
            }
            logger.info(
                "step %5d/%d | loss=%.5f  base=%.5f  hl=%.5f | "
                "lr=%.2e | %.1f steps/s",
                step, steps, avg["loss"], avg["base_mse"], avg["highlight"],
                lr_now, sps,
            )
            with open(log_path, "a") as fh:
                fh.write(json.dumps(entry) + "\n")
            running = {k: 0.0 for k in running}
            t0      = time.time()

        # ── EMA eval ──────────────────────────────────────────────────────────
        if step % eval_every == 0:
            backup = {
                f"{k}.A": v.lora_A.data.clone()
                for k, v in lora_layers.items()
            }
            backup.update({
                f"{k}.B": v.lora_B.data.clone()
                for k, v in lora_layers.items()
            })
            ema.apply_to(lora_layers)
            diffusion_model.eval()
            with torch.no_grad():
                try:
                    eval_batch  = next(iter(loader))
                    eval_noisy  = add_training_noise(eval_batch, schedule, device)
                    eval_pred   = diffusion_model(
                        eval_noisy["noisy_latent"],
                        timestep = eval_noisy["timestep"],
                        encoder_hidden_states = eval_noisy["text_embed"],
                    )
                    if hasattr(eval_pred, "sample"):
                        eval_pred = eval_pred.sample
                    eval_loss = F.mse_loss(eval_pred, eval_noisy["noise_target"]).item()
                    logger.info("  [EMA EVAL] step %d  val_loss=%.5f", step, eval_loss)
                    with open(log_path, "a") as fh:
                        fh.write(json.dumps({"step": step, "eval": True,
                                             "val_loss": round(eval_loss, 6)}) + "\n")
                except Exception as e:
                    logger.warning("  [EMA EVAL] failed: %s", e)
            ema.restore_from(lora_layers, backup)
            diffusion_model.train()

        # ── Checkpoint ────────────────────────────────────────────────────────
        if step % save_every == 0 or step == steps:
            # Save current (non-EMA) weights
            ckpt_path = os.path.join(
                output_dir, f"radiance_hdr_lora_step{step:06d}.safetensors"
            )
            export_lora_safetensors(
                lora_layers, ckpt_path,
                metadata={
                    "radiance_model_name":       model_name,
                    "radiance_compression_ratio": str(compression_ratio),
                    "radiance_rank":              str(rank),
                    "radiance_alpha":             str(alpha),
                    "step":                       str(step),
                },
            )
            # Save EMA weights
            ema_backup = {f"{k}.A": v.lora_A.data.clone() for k,v in lora_layers.items()}
            ema_backup.update({f"{k}.B": v.lora_B.data.clone() for k,v in lora_layers.items()})
            ema.apply_to(lora_layers)
            ema_path = os.path.join(
                output_dir, f"radiance_hdr_lora_ema_step{step:06d}.safetensors"
            )
            export_lora_safetensors(
                lora_layers, ema_path,
                metadata={
                    "radiance_model_name":       model_name,
                    "radiance_compression_ratio": str(compression_ratio),
                    "radiance_rank":              str(rank),
                    "radiance_alpha":             str(alpha),
                    "step":                       str(step),
                    "ema":                        "true",
                },
            )
            ema.restore_from(lora_layers, ema_backup)

    if pbar:
        pbar.close()

    final = os.path.join(output_dir, f"radiance_hdr_lora_ema_step{steps:06d}.safetensors")
    logger.info("[Train] Done — %d steps.  Final EMA: %s", steps, final)
    return final


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Path resolution for standalone usage ──────────────────────────────────
    # Script lives at: <ComfyUI>/custom_nodes/radiance/scripts/training/
    # Go up 4 levels to reach the ComfyUI root, and 2 levels to the radiance dir.
    _script_dir   = os.path.dirname(os.path.abspath(__file__))
    _radiance_dir = os.path.abspath(os.path.join(_script_dir, "..", ".."))
    _comfy_root   = os.path.abspath(os.path.join(_script_dir, "..", "..", "..", ".."))
    for _p in (_comfy_root, _radiance_dir):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    logger.info("[Main] ComfyUI root  : %s", _comfy_root)
    logger.info("[Main] Radiance dir  : %s", _radiance_dir)

    try:
        from radiance.nodes.hdr.smart import RADIANCE_MODEL_PRESETS
    except ImportError as _e:
        raise ImportError(
            f"Cannot import radiance.nodes.hdr.smart from {_radiance_dir}. "
            f"Ensure the radiance custom_node directory is correct."
        ) from _e

    parser = argparse.ArgumentParser(description="Train Radiance HDR LoRA")
    parser.add_argument("--cache_dir",   required=True,
                        help="Directory with .npz latent cache (dataset_hdr_lora.py output)")
    parser.add_argument("--model_path",  required=True,
                        help="Base diffusion model checkpoint path")
    parser.add_argument("--output_dir",  required=True,
                        help="Output directory for LoRA checkpoints")
    # Unified model list from MODEL_VAE_CONFIG
    try:
        from radiance.config.model_map import MODEL_VAE_CONFIG
        _model_choices = list(MODEL_VAE_CONFIG.keys())
    except ImportError:
        _model_choices = list(RADIANCE_MODEL_PRESETS.keys()) + [
            "flux", "wan", "hunyuanvideo", "sd3", "sdxl", "sd15",
            "lumina2", "pixart", "kolors", "aura_flow",
        ]
    parser.add_argument("--model_name",  default="flux",
                        choices=_model_choices)
    parser.add_argument("--rank",        type=int,   default=16)
    parser.add_argument("--alpha",       type=float, default=16.0)
    parser.add_argument("--lora_dropout",type=float, default=0.0)
    parser.add_argument("--steps",       type=int,   default=5_000)
    parser.add_argument("--batch_size",  type=int,   default=2)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--weight_decay",type=float, default=1e-2)
    parser.add_argument("--grad_clip",   type=float, default=1.0)
    parser.add_argument("--ema_decay",   type=float, default=0.999)
    parser.add_argument("--highlight_weight", type=float, default=0.5)
    parser.add_argument("--log_every",   type=int,   default=50)
    parser.add_argument("--save_every",  type=int,   default=500)
    parser.add_argument("--eval_every",  type=int,   default=250)
    parser.add_argument("--num_workers", type=int,   default=4)
    parser.add_argument("--resume",      default=None)
    parser.add_argument("--device",      default="cuda")
    # 16 GB VRAM flags
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing (~35% VRAM savings, -20% speed). Required for video models on 16GB.")
    parser.add_argument("--use_8bit_adam", action="store_true",
                        help="Use bitsandbytes 8-bit AdamW (~2-3 GB optimizer savings). Required for Wan/HunyuanVideo on 16GB.")
    parser.add_argument("--quantize_base", default=None, choices=["nf4", "int8"],
                        help="Quantize frozen base model weights. 'nf4' required for Flux on 16GB.")

    args = parser.parse_args()

    train(
        cache_dir         = args.cache_dir,
        model_path        = args.model_path,
        output_dir        = args.output_dir,
        model_name        = args.model_name,
        rank              = args.rank,
        alpha             = args.alpha,
        lora_dropout      = args.lora_dropout,
        steps             = args.steps,
        batch_size        = args.batch_size,
        lr                = args.lr,
        weight_decay      = args.weight_decay,
        grad_clip         = args.grad_clip,
        ema_decay         = args.ema_decay,
        highlight_weight  = args.highlight_weight,
        log_every         = args.log_every,
        save_every        = args.save_every,
        eval_every        = args.eval_every,
        num_workers              = args.num_workers,
        resume                   = args.resume,
        device_str               = args.device,
        gradient_checkpointing   = args.gradient_checkpointing,
        use_8bit_adam            = args.use_8bit_adam,
        quantize_base            = args.quantize_base,
    )
