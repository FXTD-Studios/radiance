"""
dataset_hdr_lora.py — Radiance HDR LoRA Training Dataset
════════════════════════════════════════════════════════════════════════════════

Produces (clean_latent, noisy_latent, timestep, text_embed) tuples for
LoRA fine-tuning a diffusion model on HDR footage.

This is DIFFERENT from dataset_hdr.py (HDRPairDataset), which builds
(latent, log_coded) pairs for training the TurboDecoder CNN.  LoRA training
operates entirely in latent space and requires:

  • clean_latent   — VAE-encoded HDR-compressed frame / sequence
  • noisy_latent   — clean_latent + noise at a random timestep t
  • timestep       — scalar int drawn from [0, T) per the model's schedule
  • text_embed     — CLIP / T5 text embedding (null / scene-description)

Pipeline per sample
───────────────────
    EXR frame(s)
      → soft-knee compress(compression_ratio)        [Radiance _hdr_soft_compress]
      → [optional] per-channel normalize             [Radiance _compute_channel_stats]
      → VAE encode → clean_latent z
      → noise = randn_like(z)
      → t ~ Uniform[t_min, T)
      → noisy_z = schedule.add_noise(z, noise, t)   [flow-match or DDPM]
      → (noisy_z, z, t, text_embed)

Video models (LTX-Video, Wan, HunyuanVideo)
    Frame sequences are assembled as (T, H, W, C) tensors.
    Temporal dimension is encoded by the video VAE.

Offline cache mode
    Pre-encode all EXR files to .npz (clean_latent + text_embed only;
    noise is always sampled fresh at training time so the model sees
    varied noise per epoch).

REQUIREMENTS
    pip install peft diffusers transformers safetensors tqdm
    pip install openexr imageio imageio-ffmpeg  (for EXR loading)
"""

import os
import json
import math
import logging
import hashlib
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger("radiance.dataset_hdr_lora")

# ── Reuse EXR loader from dataset_hdr.py ────────────────────────────────────
try:
    from .dataset_hdr import load_exr_as_linear
except ImportError:
    from dataset_hdr import load_exr_as_linear

# ── Reuse soft-knee compress from nodes_hdr_encoder.py ──────────────────────
try:
    from radiance.nodes.hdr.encoder import _hdr_soft_compress, _compute_channel_stats
except ImportError:
    from radiance.nodes.hdr.encoder import _hdr_soft_compress, _compute_channel_stats

# ── Model preset table (compression_ratio per model) ────────────────────────
try:
    from radiance.nodes.hdr.smart import RADIANCE_MODEL_PRESETS, _resolve_model
except ImportError:
    from radiance.nodes.hdr.smart import RADIANCE_MODEL_PRESETS, _resolve_model


# ─────────────────────────────────────────────────────────────────────────────
#  Noise schedule helpers (flow-matching and DDPM)
# ─────────────────────────────────────────────────────────────────────────────

class FlowMatchingSchedule:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    """
    Linear flow-matching schedule used by LTX-Video, Flux, SD3, and Wan.

    Forward process:  z_t = (1 - t) * z_0 + t * noise
    Loss target:      v_t = noise - z_0   (velocity prediction)
    t ~ Uniform(0, 1) — but we use logit-normal sampling for bias toward
    high-noise timesteps where HDR highlight structure matters most.

    Args:
        num_train_timesteps: number of discrete timestep bins (typically 1000).
        logit_normal_mean:   mean of the logit-normal distribution (0 = uniform).
        logit_normal_std:    std of the logit-normal distribution.
        hdr_bias:            if True, shift mean toward 0.7 to oversample
                             high-noise steps where HDR recovery is hardest.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        logit_normal_mean: float = 0.0,
        logit_normal_std: float  = 1.0,
        hdr_bias: bool = True,
    ):
        self.T   = num_train_timesteps
        self.mu  = 0.7 if hdr_bias else logit_normal_mean
        self.sig = logit_normal_std

    def sample_timestep(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample t in [0, T) using logit-normal distribution."""
        u = torch.randn(batch_size, device=device) * self.sig + self.mu
        t_frac = torch.sigmoid(u)                              # (0, 1)
        t_int  = (t_frac * self.T).long().clamp(0, self.T - 1)
        return t_int

    def add_noise(
        self,
        clean: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            clean: (B, C, ...) clean latent z_0
            noise: (B, C, ...) unit Gaussian noise
            t:     (B,) integer timestep indices

        Returns:
            noisy:  (B, C, ...) z_t  = (1-alpha)*z_0 + alpha*noise
            target: (B, C, ...) velocity v_t = noise - z_0
        """
        alpha = (t.float() / self.T).view(-1, *([1] * (clean.ndim - 1)))
        noisy  = (1.0 - alpha) * clean + alpha * noise
        target = noise - clean
        return noisy, target


class DDPMSchedule:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    """
    Standard DDPM cosine-noise schedule used by SDXL and SD 1.x / 2.x.

    Forward process:  z_t = sqrt(alpha_bar_t) * z_0 + sqrt(1 - alpha_bar_t) * noise
    Loss target:      epsilon (the noise) — epsilon prediction.
    """

    def __init__(self, num_train_timesteps: int = 1000, beta_schedule: str = "cosine"):
        self.T = num_train_timesteps
        betas  = self._cosine_betas(num_train_timesteps)
        alphas = 1.0 - betas
        self.alpha_bar = torch.cumprod(alphas, dim=0)   # (T,)

    @staticmethod
    def _cosine_betas(T: int, s: float = 8e-3) -> torch.Tensor:
        steps = torch.arange(T + 1, dtype=torch.float64)
        f     = torch.cos(((steps / T) + s) / (1.0 + s) * math.pi * 0.5) ** 2
        alpha_bar = f / f[0]
        betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])
        return betas.clamp(0.0, 0.999).float()

    def sample_timestep(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.T, (batch_size,), device=device)

    def add_noise(
        self,
        clean: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ab = self.alpha_bar.to(clean.device)[t]
        ab = ab.view(-1, *([1] * (clean.ndim - 1)))
        noisy  = torch.sqrt(ab) * clean + torch.sqrt(1.0 - ab) * noise
        target = noise   # epsilon prediction
        return noisy, target


def make_schedule(model_name: str) -> "FlowMatchingSchedule | DDPMSchedule":
    """Return the correct noise schedule for a given model family."""
    flow_models = {"ltx-video", "flux", "wan", "hunyuanvideo", "sd3"}
    preset = _resolve_model(model_name)
    key    = model_name.strip().lower()
    if any(k in key for k in flow_models) or (
        preset and preset.get("latent_channels", 4) >= 16
    ):
        return FlowMatchingSchedule(hdr_bias=True)
    return DDPMSchedule()


# ─────────────────────────────────────────────────────────────────────────────
#  Null-text embeddings (unconditional training)
# ─────────────────────────────────────────────────────────────────────────────

def _null_text_embed(
    seq_len: int,
    hidden_size: int,
    batch_size: int = 1,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Zero-valued text embedding for unconditional LoRA training.

    The LoRA learns HDR-consistent latent distributions without needing
    text descriptions of the training footage.  At inference the user's
    own prompt drives content; the LoRA only shifts the tonal distribution.

    Shape: (B, seq_len, hidden_size)
    """
    return torch.zeros(batch_size, seq_len, hidden_size, device=device)


# Text embedding dimensions per model family
_TEXT_EMBED_DIMS: Dict[str, Tuple[int, int]] = {
    "ltx-video":    (128, 4096),   # T5-XXL
    "flux":         (256, 4096),   # T5-XXL + CLIP-L
    "wan":          (128, 4096),   # T5-XXL
    "hunyuanvideo": (128, 4096),   # T5-XXL
    "sd3":          (154, 4096),   # T5-XXL + CLIP
    "sdxl":         (77,  2048),   # CLIP-G + CLIP-L
    "sd15":         (77,  768),    # CLIP-L
}

def get_null_embed(model_name: str, batch_size: int = 1,
                   device: torch.device = torch.device("cpu")) -> torch.Tensor:
    preset_key = _resolve_model(model_name)
    key = model_name.strip().lower()
    for k, dims in _TEXT_EMBED_DIMS.items():
        if k in key:
            return _null_text_embed(*dims, batch_size=batch_size, device=device)
    return _null_text_embed(77, 768, batch_size=batch_size, device=device)


# ─────────────────────────────────────────────────────────────────────────────
#  Pre-processing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _centre_crop_resize(
    img: np.ndarray,
    target_hw: Tuple[int, int],
) -> np.ndarray:
    """Centre-crop to target aspect ratio, then resize. Operates in linear space."""
    import cv2
    H_tgt, W_tgt = target_hw
    H_src, W_src = img.shape[:2]
    ar_tgt = W_tgt / H_tgt
    ar_src = W_src / H_src
    if ar_src > ar_tgt:
        new_w = int(H_src * ar_tgt)
        x0    = (W_src - new_w) // 2
        img   = img[:, x0 : x0 + new_w]
    else:
        new_h = int(W_src / ar_tgt)
        y0    = (H_src - new_h) // 2
        img   = img[y0 : y0 + new_h]
    return cv2.resize(img, (W_tgt, H_tgt), interpolation=cv2.INTER_AREA).astype(np.float32)


def _random_crop_resize(
    img: np.ndarray,
    target_hw: Tuple[int, int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Random crop to target aspect ratio, then resize."""
    import cv2
    H_tgt, W_tgt = target_hw
    H_src, W_src = img.shape[:2]
    ar_tgt = W_tgt / H_tgt
    min_h  = max(H_tgt, 1)
    max_h  = H_src
    if max_h <= min_h:
        return _centre_crop_resize(img, target_hw)
    crop_h = rng.integers(min_h, max_h + 1)
    crop_w = int(crop_h * ar_tgt)
    if crop_w > W_src:
        crop_w = W_src
        crop_h = int(crop_w / ar_tgt)
    y0 = rng.integers(0, max(1, H_src - crop_h))
    x0 = rng.integers(0, max(1, W_src - crop_w))
    img = img[y0 : y0 + crop_h, x0 : x0 + crop_w]
    return cv2.resize(img, (W_tgt, H_tgt), interpolation=cv2.INTER_AREA).astype(np.float32)


def _compress_frame(
    img_linear: np.ndarray,
    compression_ratio: float,
    per_channel_norm: bool = False,
    norm_center: float = 3.0,
) -> Tuple[np.ndarray, Optional[Dict]]:
    """
    Apply soft-knee HDR compression to a single float32 linear frame.

    Returns:
        compressed: float32 (H, W, 3) in [0, 1]
        stats:      dict with mean/std for per-channel norm (None if disabled)
    """
    t = torch.from_numpy(img_linear.astype(np.float32))
    if t.ndim == 2:
        t = t.unsqueeze(-1).expand(-1, -1, 3)

    compressed = _hdr_soft_compress(t, compression_ratio)

    stats = None
    if per_channel_norm:
        mean, std = _compute_channel_stats(compressed.unsqueeze(0))
        mu  = mean.reshape(1, 1, -1)
        sig = std.reshape(1, 1, -1)
        compressed = ((compressed - mu) / sig + norm_center) / (2.0 * norm_center)
        compressed = compressed.clamp(0.0, 1.0)
        stats = {"mean": mean.tolist(), "std": std.tolist(), "norm_center": norm_center}

    return compressed.numpy().astype(np.float32), stats


# ─────────────────────────────────────────────────────────────────────────────
#  HDR LoRA Dataset — online mode
# ─────────────────────────────────────────────────────────────────────────────

class HDRLoRADataset(Dataset):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    """
    Dataset for LoRA fine-tuning a diffusion model on HDR footage.

    Each item:
        clean_latent  (C, F, H_lat, W_lat) for video  /  (C, H_lat, W_lat) for image
        noise_target  same shape as clean_latent — v_t (flow) or epsilon (DDPM)
        noisy_latent  same shape
        timestep      scalar LongTensor
        text_embed    (seq_len, hidden_size) — zeros for unconditional

    The dataset does NOT add noise to the clean latent.  The DataLoader
    collates clean_latent; the training loop samples noise + timestep per
    batch so each epoch sees fresh noise.  (Noise is only stored in the
    offline cache for reproducible unit tests.)

    Args:
        exr_dirs:          List of directories with .exr / .hdr files.
        vae:               ComfyUI-compatible VAE (encode method).
        model_name:        "ltx-video", "flux", "wan", etc.  Used to look up
                           compression_ratio and text_embed_dims.
        target_hw:         (H, W) spatial resolution for training.
        n_frames:          Number of consecutive frames per video sequence.
                           1 = image mode (no temporal dimension).
        compression_ratio: Override from model preset if set explicitly.
        per_channel_norm:  Mirror vae_per_channel_normalize.
        augment:           Horizontal flip augmentation.
        cache_dir:         If set, pre-encode to .npz on first pass and reload.
        max_samples:       Cap number of source images (for quick experiments).
    """

    _EXTENSIONS = {".exr", ".hdr", ".EXR", ".HDR", ".png", ".PNG"}

    def __init__(
        self,
        exr_dirs: List[str],
        vae,
        model_name: str          = "ltx-video",
        target_hw: Tuple[int,int] = (512, 512),
        n_frames: int             = 1,
        compression_ratio: Optional[float] = None,
        per_channel_norm: bool    = False,
        norm_center: float        = 3.0,
        augment: bool             = True,
        cache_dir: Optional[str]  = None,
        max_samples: int          = -1,
    ):
        self.vae              = vae
        self.model_name       = model_name
        self.target_hw        = target_hw
        self.n_frames         = n_frames
        self.per_channel_norm = per_channel_norm
        self.norm_center      = norm_center
        self.augment          = augment
        self.cache_dir        = cache_dir
        self._rng             = np.random.default_rng(seed=42)

        # Resolve model preset
        preset = _resolve_model(model_name) or RADIANCE_MODEL_PRESETS["ltx-video"]
        self.compression_ratio = (
            compression_ratio if compression_ratio is not None
            else preset["compression_ratio"]
        )
        self.norm_center = norm_center if norm_center != 3.0 else preset["norm_center"]
        self.spatial_factor  = preset["vae_spatial_factor"]
        self.temporal_factor = preset["vae_temporal_factor"]
        self.latent_channels = preset["latent_channels"]

        # Text embed shape
        key = model_name.strip().lower()
        seq_len, hidden = 77, 768
        for k, dims in _TEXT_EMBED_DIMS.items():
            if k in key:
                seq_len, hidden = dims
                break
        self.text_embed_shape = (seq_len, hidden)

        # Scan EXR files
        self.exr_paths = self._scan(exr_dirs)
        if max_samples > 0:
            self.exr_paths = self.exr_paths[:max_samples]

        if not self.exr_paths:
            raise FileNotFoundError(f"No EXR/HDR files found in: {exr_dirs}")

        # Offline cache
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            self._cache = self._build_or_load_cache()
        else:
            self._cache = None

        logger.info(
            "[HDRLoRADataset] model=%s  compression_ratio=%.2f  "
            "n_frames=%d  %d sources  cache=%s",
            model_name, self.compression_ratio, n_frames,
            len(self.exr_paths), cache_dir or "disabled",
        )

    # ── Scan ──────────────────────────────────────────────────────────────────

    def _scan(self, dirs: List[str]) -> List[str]:
        paths = []
        for d in dirs:
            for root, _, files in os.walk(d):
                for f in files:
                    if Path(f).suffix in self._EXTENSIONS:
                        paths.append(os.path.join(root, f))
        return sorted(paths)

    # ── Cache ─────────────────────────────────────────────────────────────────

    def _cache_key(self, path: str) -> str:
        tag = f"{path}|{self.target_hw}|{self.compression_ratio}|{self.per_channel_norm}"
        return hashlib.md5(tag.encode()).hexdigest()[:16]

    def _build_or_load_cache(self) -> List[Path]:
        """Encode all EXR files to .npz in cache_dir if not already done."""
        from tqdm import tqdm
        cached = []
        for path in tqdm(self.exr_paths, desc="Building LoRA cache"):
            key      = self._cache_key(path)
            out_path = Path(self.cache_dir) / f"{key}.npz"
            if not out_path.exists():
                latent = self._encode_path(path)
                if latent is None:
                    continue
                np.savez_compressed(
                    str(out_path),
                    clean_latent = latent.cpu().numpy(),
                    meta         = json.dumps({
                        "source":           path,
                        "model_name":       self.model_name,
                        "compression_ratio": self.compression_ratio,
                        "target_hw":        list(self.target_hw),
                        "n_frames":         self.n_frames,
                    }),
                )
            cached.append(out_path)
        logger.info("[HDRLoRADataset] Cache: %d / %d items ready", len(cached), len(self.exr_paths))
        return cached

    # ── Encode a single EXR path to a clean latent ───────────────────────────

    def _encode_path(self, path: str) -> Optional[torch.Tensor]:
        """Returns clean_latent tensor or None on failure."""
        img_linear = load_exr_as_linear(path)
        if img_linear is None:
            return None

        # Crop + resize
        frame = _centre_crop_resize(img_linear, self.target_hw)

        # Soft-knee compression
        compressed, _ = _compress_frame(
            frame,
            self.compression_ratio,
            per_channel_norm=self.per_channel_norm,
            norm_center=self.norm_center,
        )

        # For video models: replicate single frame to n_frames
        # (real video training would use actual frame sequences)
        if self.n_frames > 1:
            # (T, H, W, C) — replicate + jitter for diversity
            frames = [compressed]
            for _ in range(self.n_frames - 1):
                noise = np.random.normal(0, 0.002, compressed.shape).astype(np.float32)
                frames.append(np.clip(compressed + noise, 0.0, 1.0))
            pixel_batch = torch.from_numpy(np.stack(frames, axis=0))  # (T, H, W, C)
            # Reshape for VAE: (1, T, H, W, C) or (T, 1, C, H, W) depending on backend
        else:
            pixel_batch = torch.from_numpy(compressed).unsqueeze(0)   # (1, H, W, C)

        try:
            with torch.no_grad():
                result = self.vae.encode(pixel_batch)
            if isinstance(result, dict):
                latent = result.get("samples", result.get("latent", None))
                if latent is None:
                    latent = next(iter(result.values()))
            else:
                latent = result
            return latent.squeeze(0).cpu()  # (C, [F,] H_lat, W_lat)
        except Exception as e:
            logger.warning("[HDRLoRADataset] VAE encode failed for %s: %s", path, e)
            return None

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._cache) if self._cache else len(self.exr_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Load clean latent
        if self._cache:
            data         = np.load(str(self._cache[idx]))
            clean_latent = torch.from_numpy(data["clean_latent"].astype(np.float32))
        else:
            path         = self.exr_paths[idx]
            clean_latent = self._encode_path(path)
            if clean_latent is None:
                # Fallback: zeros (prevents DataLoader crash on bad files)
                H_lat = self.target_hw[0] // self.spatial_factor
                W_lat = self.target_hw[1] // self.spatial_factor
                clean_latent = torch.zeros(self.latent_channels, H_lat, W_lat)

        # Horizontal flip augmentation
        if self.augment and torch.rand(1).item() > 0.5:
            clean_latent = torch.flip(clean_latent, dims=[-1])

        # Null text embedding (unconditional)
        seq_len, hidden = self.text_embed_shape
        text_embed = torch.zeros(seq_len, hidden, dtype=torch.float32)

        return {
            "clean_latent": clean_latent,     # (C, [F,] H_lat, W_lat)
            "text_embed":   text_embed,       # (seq_len, hidden_size)
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Collate + noise injection — called inside the training loop
# ─────────────────────────────────────────────────────────────────────────────

def add_training_noise(
    batch: Dict[str, torch.Tensor],
    schedule: "FlowMatchingSchedule | DDPMSchedule",
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Inject noise into a collated batch at random timesteps.

    Called inside the training loop (NOT inside __getitem__) so that
    each training step sees fresh noise for the same clean latent.

    Args:
        batch:    dict with 'clean_latent' (B, C, ...) and 'text_embed' (B, S, D)
        schedule: FlowMatchingSchedule or DDPMSchedule
        device:   target device

    Returns dict with additional keys:
        noisy_latent  (B, C, ...) — z_t
        noise_target  (B, C, ...) — v (flow) or epsilon (DDPM)
        timestep      (B,)        — integer t
    """
    clean  = batch["clean_latent"].to(device)
    B      = clean.shape[0]
    noise  = torch.randn_like(clean)
    t      = schedule.sample_timestep(B, device)
    noisy, target = schedule.add_noise(clean, noise, t)

    return {
        "clean_latent":  clean,
        "noisy_latent":  noisy,
        "noise_target":  target,
        "timestep":      t,
        "text_embed":    batch["text_embed"].to(device),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def make_lora_dataloader(
    exr_dirs: List[str],
    vae,
    model_name: str           = "ltx-video",
    target_hw: Tuple[int,int] = (512, 512),
    n_frames: int             = 1,
    batch_size: int           = 4,
    num_workers: int          = 4,
    cache_dir: Optional[str]  = None,
    compression_ratio: Optional[float] = None,
    per_channel_norm: bool    = False,
    augment: bool             = True,
    max_samples: int          = -1,
) -> DataLoader:
    """
    Convenience factory that returns a DataLoader ready for LoRA training.

    Example:
        loader = make_lora_dataloader(
            exr_dirs     = ["/data/exr/arri", "/data/exr/hdri_haven"],
            vae          = comfy_vae_object,
            model_name   = "ltx-video",
            target_hw    = (512, 512),
            n_frames     = 9,      # 9-frame video clip for LTX temporal VAE
            batch_size   = 2,
            cache_dir    = "/data/hdr_lora_cache",
        )
        for batch in loader:
            noisy_batch = add_training_noise(batch, schedule, device)
    """
    dataset = HDRLoRADataset(
        exr_dirs          = exr_dirs,
        vae               = vae,
        model_name        = model_name,
        target_hw         = target_hw,
        n_frames          = n_frames,
        compression_ratio = compression_ratio,
        per_channel_norm  = per_channel_norm,
        augment           = augment,
        cache_dir         = cache_dir,
        max_samples       = max_samples,
    )
    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = True,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  CLI — pre-build the cache without running full training
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Pre-build Radiance HDR LoRA latent cache")
    parser.add_argument("--exr_dirs",   required=True, nargs="+",
                        help="Directories with EXR/HDR footage")
    parser.add_argument("--cache_dir",  required=True,
                        help="Output directory for .npz latent cache")
    parser.add_argument("--vae_path",   required=True,
                        help="Path to VAE checkpoint (.safetensors / .ckpt)")
    parser.add_argument("--vae_type",   default="ltx-video",
                        choices=list(RADIANCE_MODEL_PRESETS.keys()),
                        help="Model family (for compression_ratio preset)")
    parser.add_argument("--model_name", default="ltx-video")
    parser.add_argument("--size",       type=int, default=512,
                        help="Training image size (square)")
    parser.add_argument("--n_frames",   type=int, default=1,
                        help="Frames per video clip (1 = image mode)")
    parser.add_argument("--max_samples",type=int, default=-1)
    parser.add_argument("--device",     default="cuda")

    args = parser.parse_args()

    # Load VAE
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    try:
        from dataset_hdr import load_vae_standalone
    except ImportError:
        from radiance.dataset_hdr import load_vae_standalone

    try:
        vae = load_vae_standalone(args.vae_path, args.vae_type, args.device)
    except Exception as e:
        logger.error("Failed to load VAE: %s", e)
        sys.exit(1)

    dataset = HDRLoRADataset(
        exr_dirs    = args.exr_dirs,
        vae         = vae,
        model_name  = args.model_name,
        target_hw   = (args.size, args.size),
        n_frames    = args.n_frames,
        cache_dir   = args.cache_dir,
        max_samples = args.max_samples,
    )
    logger.info("Cache built: %d items in %s", len(dataset), args.cache_dir)

    # ── Cleanup ──────────────────────────────────────────────────────────────
    # Explicitly delete the VAE object to trigger ModelPatcher cleanup
    # before Python's global shutdown clears the modules it depends on.
    del vae
    del dataset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()
