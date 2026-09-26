"""Temporal RUDRA residual recovery.

This module defines the Phase 3 checkpoint contract. Unlike the original
RUDRA decoders, the temporal model predicts a signed scene-linear residual and
separate highlight/shadow confidence maps. It never regenerates the full frame.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import torch

from radiance.model.cache import GPUModelCache
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("radiance.temporal_rudra")

# Bounded LRU -- was an unbounded dict holding GPU-resident models.
_TEMPORAL_MODEL_CACHE = GPUModelCache(max_size=1)


def _flow_grid(flow: torch.Tensor) -> torch.Tensor:
    """Convert pixel-space BCHW flow to a grid_sample grid."""
    b, _, h, w = flow.shape
    y, x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=flow.device, dtype=flow.dtype),
        torch.linspace(-1.0, 1.0, w, device=flow.device, dtype=flow.dtype),
        indexing="ij",
    )
    base = torch.stack((x, y), dim=-1).unsqueeze(0).expand(b, -1, -1, -1)
    scale_x = 2.0 / max(w - 1, 1)
    scale_y = 2.0 / max(h - 1, 1)
    offset = torch.stack((flow[:, 0] * scale_x, flow[:, 1] * scale_y), dim=-1)
    return base + offset


class TemporalResidualBlock(nn.Module):
    """Small residual 3D block preserving time and resolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv3d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)


class TemporalRUDRAResidual(nn.Module):
    """Motion-aligned temporal residual model for SDR-to-HDR recovery.

    Input is scene-linear Rec.709 ``[B,T,H,W,3]`` plus highlight and shadow
    evidence masks. Output is a bounded signed residual and independent learned
    confidence maps for every frame.
    """

    def __init__(self, width: int = 48, blocks: int = 4, max_flow: float = 24.0):
        super().__init__()
        self.width = int(width)
        self.blocks = int(blocks)
        self.max_flow = float(max_flow)
        self.flow_head = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 2, 3, padding=1),
        )
        # current RGB + warped previous RGB + warped next RGB + two masks
        self.stem = nn.Conv3d(11, self.width, 3, padding=1)
        self.body = nn.Sequential(
            *[TemporalResidualBlock(self.width) for _ in range(self.blocks)]
        )
        self.residual_head = nn.Conv3d(self.width, 3, 3, padding=1)
        self.highlight_confidence_head = nn.Conv3d(self.width, 1, 3, padding=1)
        self.shadow_confidence_head = nn.Conv3d(self.width, 1, 3, padding=1)

    def _warp(self, neighbor: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        flow = torch.tanh(self.flow_head(torch.cat((neighbor, current), dim=1)))
        flow = flow * self.max_flow
        return F.grid_sample(
            neighbor, _flow_grid(flow), mode="bilinear",
            padding_mode="border", align_corners=True,
        )

    def forward(
        self,
        frames: torch.Tensor,
        highlight_mask: torch.Tensor,
        shadow_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if frames.ndim != 5 or frames.shape[-1] != 3:
            raise ValueError("Temporal RUDRA frames must have shape [B,T,H,W,3].")
        if frames.shape[1] < 3:
            raise ValueError("Temporal RUDRA requires at least three adjacent frames.")

        x = frames.permute(0, 1, 4, 2, 3)
        previous = torch.cat((x[:, :1], x[:, :-1]), dim=1)
        following = torch.cat((x[:, 1:], x[:, -1:]), dim=1)
        b, t, c, h, w = x.shape
        current_flat = x.reshape(b * t, c, h, w)
        previous = self._warp(previous.reshape(b * t, c, h, w), current_flat)
        following = self._warp(following.reshape(b * t, c, h, w), current_flat)
        previous = previous.reshape(b, t, c, h, w)
        following = following.reshape(b, t, c, h, w)

        features = torch.cat(
            (
                x,
                previous,
                following,
                highlight_mask[:, :, None],
                shadow_mask[:, :, None],
            ),
            dim=2,
        ).permute(0, 2, 1, 3, 4)
        features = self.body(F.silu(self.stem(features)))
        residual = torch.tanh(self.residual_head(features)).permute(0, 2, 3, 4, 1)
        highlight_confidence = torch.sigmoid(
            self.highlight_confidence_head(features).squeeze(1)
        )
        shadow_confidence = torch.sigmoid(
            self.shadow_confidence_head(features).squeeze(1)
        )
        return residual, highlight_confidence, shadow_confidence


TEMPORAL_CHECKPOINT_ENV = "RADIANCE_TEMPORAL_RUDRA"
TEMPORAL_CHECKPOINT_PATTERNS = (
    "temporal_rudra_residual_ema.safetensors",
    "temporal_rudra_residual_ema.pth",
    "temporal_rudra_residual*.safetensors",
    "temporal_rudra_residual*.pth",
    "temporal_rudra*.pt",
)


def resolve_temporal_checkpoint(checkpoint_path: str = "") -> Path | None:
    """Explicit path, then ``RADIANCE_TEMPORAL_RUDRA``, then models/radiance."""
    from radiance.model.paths import find_radiance_checkpoint
    return find_radiance_checkpoint(
        TEMPORAL_CHECKPOINT_PATTERNS, explicit_path=checkpoint_path or "",
        env_var=TEMPORAL_CHECKPOINT_ENV,
    )


def _checkpoint_candidates(checkpoint_path: str = "") -> list[Path]:
    """Kept for callers/tests that enumerate candidates; one entry or none."""
    resolved = resolve_temporal_checkpoint(checkpoint_path)
    return [resolved] if resolved is not None else []


def load_temporal_rudra_weights(
    checkpoint_path: str = "",
    device: torch.device | str = "cpu",
) -> TemporalRUDRAResidual | None:
    """Strictly load a Phase 3 temporal checkpoint, or return ``None``."""
    candidate = next((p for p in _checkpoint_candidates(checkpoint_path) if p.is_file()), None)
    if candidate is None:
        return None
    cache_key = (str(candidate.resolve()), str(device))
    if cache_key in _TEMPORAL_MODEL_CACHE:
        return _TEMPORAL_MODEL_CACHE.get(cache_key)  # type: ignore[return-value]

    try:
        if candidate.suffix.lower() == ".safetensors":
            import safetensors.torch
            state = safetensors.torch.load_file(str(candidate), device="cpu")
            metadata = {}
        else:
            checkpoint = torch.load(candidate, map_location="cpu", weights_only=True)
            metadata = checkpoint.get("temporal_rudra", {}) if isinstance(checkpoint, dict) else {}
            if isinstance(checkpoint, dict) and "ema_shadow" in checkpoint:
                state = checkpoint["ema_shadow"]
            elif isinstance(checkpoint, dict) and "model" in checkpoint:
                state = checkpoint["model"]
            else:
                state = checkpoint
        width = int(metadata.get("width", state["stem.weight"].shape[0]))
        blocks = int(metadata.get(
            "blocks",
            len({key.split(".")[1] for key in state if key.startswith("body.")}),
        ))
        max_flow = float(metadata.get("max_flow", 24.0))
        model = TemporalRUDRAResidual(width=width, blocks=blocks, max_flow=max_flow)
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
    except (OSError, RuntimeError, KeyError, ValueError) as exc:
        logger.error("Temporal RUDRA checkpoint %s is incompatible: %s", candidate, exc)
        _TEMPORAL_MODEL_CACHE.put(cache_key, None)
        return None

    _TEMPORAL_MODEL_CACHE.put(cache_key, model)
    logger.info("Loaded temporal RUDRA residual checkpoint: %s", candidate)
    return model


# Per-frame model forwards accumulated into three lists. Without this the
# activation graph of every frame is retained simultaneously -- VRAM grows
# linearly with clip length until it OOMs.


@torch.no_grad()


def recover_temporal_residual(
    source: torch.Tensor,
    deterministic: torch.Tensor,
    highlight_mask: torch.Tensor,
    shadow_mask: torch.Tensor,
    model: TemporalRUDRAResidual,
    peak_scale: float,
    window_size: int = 5,
    preserve_source_outside: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run overlapping temporal windows and apply confidence-aware fallback.

    ``source`` and ``deterministic`` use ``[T,H,W,3]``. Outside both recovery
    masks the source is copied bit-for-bit. Inside a mask, low model confidence
    falls back to the deterministic reconstruction.
    """
    t_count = source.shape[0]
    window = min(max(int(window_size), 5), 9)
    if window % 2 == 0:
        window += 1
    radius = window // 2
    # STREAM-FIX: write each window's centre frame straight into three
    # pre-allocated clip buffers. The three Python lists plus the closing
    # torch.stack held two full-resolution copies of every accumulator for the
    # whole clip, so peak grew with clip length on top of the window forward
    # that is inherently per frame. Peak above the three results is now one
    # window.
    residual = source.new_empty((t_count,) + tuple(source.shape[1:]))
    h_conf = highlight_mask.new_empty((t_count,) + tuple(highlight_mask.shape[1:]))
    s_conf = shadow_mask.new_empty((t_count,) + tuple(shadow_mask.shape[1:]))
    for index in range(t_count):
        indices = [min(max(i, 0), t_count - 1)
                   for i in range(index - radius, index + radius + 1)]
        frames = source[indices].unsqueeze(0)
        highlights = highlight_mask[indices].unsqueeze(0)
        shadows = shadow_mask[indices].unsqueeze(0)
        window_residual, window_h, window_s = model(frames, highlights, shadows)
        residual[index] = window_residual[0, radius]
        h_conf[index] = window_h[0, radius]
        s_conf[index] = window_s[0, radius]
        del frames, highlights, shadows, window_residual, window_h, window_s

    residual.mul_(float(peak_scale))
    h_conf.mul_(highlight_mask)
    s_conf.mul_(shadow_mask)
    confidence = torch.maximum(h_conf, s_conf).clamp(0.0, 1.0)
    region = torch.maximum(highlight_mask, shadow_mask).clamp(0.0, 1.0)
    learned = (source + residual * region.unsqueeze(-1)).clamp(min=0.0)
    recovered_region = torch.lerp(deterministic, learned, confidence.unsqueeze(-1))
    region_rgb = region.unsqueeze(-1)
    if preserve_source_outside:
        output = source + region_rgb * (recovered_region - source)
    else:
        output = deterministic + region_rgb * confidence.unsqueeze(-1) * (learned - deterministic)
    return output, h_conf, s_conf


__all__ = [
    "TemporalRUDRAResidual",
    "load_temporal_rudra_weights",
    "recover_temporal_residual",
]
