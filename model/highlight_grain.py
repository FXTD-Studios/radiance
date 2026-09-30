"""Settle reconstruction grain in flat SDR highlights.

Port of RUDRA 0.9.0-beta.2's luminance-only highlight grain correction.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

_LUMA = (0.2627, 0.6780, 0.0593)


# ALBABIT-FIX: torch, so the pass runs on the GPU (324 ms -> 16 ms per 1080p
# frame). float64 matches RUDRA's numpy/OpenCV version exactly
# (tests/test_highlight_grain.py); MPS has no float64 and computes in float32.
def _reflect_index(n: int, p: int, device) -> torch.Tensor:
    i = torch.arange(-p, n + p, device=device) % (2 * n)
    return torch.where(i < n, i, 2 * n - 1 - i)


def _reflect_pad(x: torch.Tensor, p: int) -> torch.Tensor:
    """OpenCV's BORDER_REFLECT (edge pixel repeated) on the last two dims,
    mirrored again when the image is smaller than the pad, as OpenCV does."""
    x = x.index_select(-1, _reflect_index(x.shape[-1], p, x.device))
    return x.index_select(-2, _reflect_index(x.shape[-2], p, x.device))


def _box(x: torch.Tensor, size: int) -> torch.Tensor:
    return F.avg_pool2d(_reflect_pad(x, size // 2), size, stride=1)


def _gaussian(x: torch.Tensor, sigma: float) -> torch.Tensor:
    # OpenCV's kernel size for a non-8-bit image: round(sigma * 8 + 1), odd.
    radius = (int(round(sigma * 8 + 1)) | 1) // 2
    taps = torch.arange(-radius, radius + 1, dtype=x.dtype, device=x.device)
    kernel = torch.exp(-(taps * taps) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    x = _reflect_pad(x, radius)
    x = F.conv2d(x, kernel.view(1, 1, 1, -1))
    return F.conv2d(x, kernel.view(1, 1, -1, 1))


def settle_highlight_grain(
    hdr_nits: torch.Tensor,
    sdr_srgb: torch.Tensor,
    knee: float = 0.9,
    softness: float = 0.04,
    sigma: float = 2.0,
) -> torch.Tensor:
    """Reduce grain in reconstructed, flat highlights while preserving hue.

    ``hdr_nits`` and ``sdr_srgb`` are [H, W, 3] or [B, H, W, 3] on one device;
    the result has ``hdr_nits``'s shape, dtype and device.
    """
    if hdr_nits.shape != sdr_srgb.shape:
        raise ValueError("HDR and SDR images must have the same shape")
    if not 0.0 < knee < 1.0 or softness <= 0.0 or sigma <= 0.0:
        raise ValueError("invalid highlight grain parameters")

    dtype = torch.float32 if hdr_nits.device.type == "mps" else torch.float64
    single = hdr_nits.dim() == 3
    hdr = (hdr_nits[None] if single else hdr_nits).to(dtype)
    sdr = (sdr_srgb[None] if single else sdr_srgb).to(dtype)
    luma_w = torch.tensor(_LUMA, dtype=dtype, device=hdr.device)

    code = sdr.max(dim=-1).values
    ramp = ((code - (knee - softness)) / (2.0 * softness)).clamp(0.0, 1.0)
    ramp = ramp * ramp * (3.0 - 2.0 * ramp)
    if not bool((ramp > 0.0).any()):
        return hdr_nits

    size = 7
    flat = torch.zeros_like(code)
    for signal in (code, sdr @ luma_w):
        signal = signal[:, None]
        mean = _box(signal, size)
        second = _box(signal * signal, size)
        spread = (second - mean * mean).clamp(min=0.0).sqrt()[:, 0]
        flat = torch.maximum(flat, ((3.0 - spread * 255.0) / 2.0).clamp(0.0, 1.0))
    flat = flat * flat * (3.0 - 2.0 * flat)
    weight = ramp * flat
    if not bool((weight > 0.0).any()):
        return hdr_nits

    luma = (hdr @ luma_w).clamp(min=0.0)
    settled = (_gaussian((flat * luma)[:, None], sigma)
               / _gaussian(flat[:, None], sigma).clamp(min=1e-6))[:, 0]
    target = luma + weight * (settled - luma)
    gain = torch.where(luma > 1e-6, target / luma.clamp(min=1e-6), torch.ones_like(luma))
    out = torch.nan_to_num(hdr * gain[..., None], nan=0.0, posinf=0.0, neginf=0.0)
    return (out[0] if single else out).to(hdr_nits.dtype)
