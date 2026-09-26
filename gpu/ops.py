"""GPU-accelerated image processing operations.

All functions work on torch tensors and fall back gracefully when CUDA
is unavailable (CPU execution is handled by PyTorch automatically).
"""
from __future__ import annotations

import logging
import torch
import torch.nn.functional as F

logger = logging.getLogger("radiance.gpu.ops")


def _gaussian_blur_gpu(img: torch.Tensor, sigma: float = 2.0, kernel_size: int = 7) -> torch.Tensor:
    if img.dim() == 3:
        img_bchw = img.permute(2, 0, 1).unsqueeze(0)
        squeeze = True
    else:
        img_bchw = img
        squeeze = False

    half = kernel_size // 2
    x = torch.arange(-half, half + 1, dtype=torch.float32, device=img.device)
    kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()

    C = img_bchw.shape[1]
    k_h = kernel_1d.view(1, 1, -1, 1).expand(C, 1, -1, 1)
    k_w = kernel_1d.view(1, 1, 1, -1).expand(C, 1, 1, -1)
    pad = half

    out = F.conv2d(img_bchw, k_h, padding=(pad, 0), groups=C)
    out = F.conv2d(out, k_w, padding=(0, pad), groups=C)

    if squeeze:
        out = out.squeeze(0).permute(1, 2, 0)
    return out


def _downsample_gpu(img: torch.Tensor) -> torch.Tensor:
    blurred = _gaussian_blur_gpu(img)
    bchw = blurred.permute(2, 0, 1).unsqueeze(0)
    down = F.avg_pool2d(bchw, kernel_size=2, stride=2, ceil_mode=True)
    return down.squeeze(0).permute(1, 2, 0)


def _upsample_gpu(img: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    bchw = img.permute(2, 0, 1).unsqueeze(0)
    up = F.interpolate(bchw, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return up.squeeze(0).permute(1, 2, 0)


def gpu_laplacian_pyramid_blend(
    low: torch.Tensor,
    high: torch.Tensor,
    mask: torch.Tensor,
    levels: int = 5,
) -> torch.Tensor:
    def _build_gaussian(img, n):
        pyr = [img]
        for _ in range(n - 1):
            pyr.append(_downsample_gpu(pyr[-1]))
        return pyr

    def _build_laplacian(gauss):
        lap = []
        for i in range(len(gauss) - 1):
            upsampled = _upsample_gpu(gauss[i + 1], gauss[i].shape[0], gauss[i].shape[1])
            lap.append(gauss[i] - upsampled)
        lap.append(gauss[-1])
        return lap

    g_low = _build_gaussian(low, levels)
    g_high = _build_gaussian(high, levels)
    g_mask = _build_gaussian(mask, levels)
    l_low = _build_laplacian(g_low)
    l_high = _build_laplacian(g_high)

    blended_lap = []
    for ll, lh, m in zip(l_low, l_high, g_mask):
        m_match = _upsample_gpu(m, ll.shape[0], ll.shape[1]) if m.shape[:2] != ll.shape[:2] else m
        blended_lap.append(ll * m_match + lh * (1.0 - m_match))

    result = blended_lap[-1]
    for i in range(len(blended_lap) - 2, -1, -1):
        result = _upsample_gpu(result, blended_lap[i].shape[0], blended_lap[i].shape[1])
        result = result + blended_lap[i]
    return result


def gpu_local_contrast(
    img: torch.Tensor,
    sigma: float = 50.0,
    amount: float = 0.5,
    luma_only: bool = True,
) -> torch.Tensor:
    if img.dim() == 3:
        img_b = img.unsqueeze(0)
        squeeze = True
    else:
        img_b = img
        squeeze = False

    if luma_only and img_b.shape[-1] >= 3:
        luma = (
            0.2126 * img_b[..., 0] + 0.7152 * img_b[..., 1] + 0.0722 * img_b[..., 2]
        ).unsqueeze(-1)
    else:
        luma = img_b

    kernel_size = max(3, int(sigma) * 2 + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    pad = kernel_size // 2

    luma_bchw = luma.permute(0, 3, 1, 2)
    blurred = F.avg_pool2d(
        F.pad(luma_bchw, (pad, pad, pad, pad), mode="reflect"),
        kernel_size=kernel_size,
        stride=1,
        padding=0,
    )

    blurred_bhwc = blurred.permute(0, 2, 3, 1)
    detail = luma / (blurred_bhwc + 1e-10)
    boost = (1.0 + amount * (detail - 1.0)).clamp(min=0.5, max=2.0)
    result = img_b * boost

    if squeeze:
        result = result.squeeze(0)
    return result


def gpu_memory_info() -> dict:
    """Return available and total GPU memory in GB."""
    info = {"available_gb": 0.0, "total_gb": 0.0}
    try:
        if torch.cuda.is_available():
            free_mem, _ = torch.cuda.mem_get_info(0)
            info["available_gb"] = round(free_mem / (1024 ** 3), 1)
            total_mem = torch.cuda.get_device_properties(0).total_memory
            info["total_gb"] = round(total_mem / (1024 ** 3), 1)
    except Exception as _exc:
        logger.debug(
            "[Radiance] gpu_memory_info(): ignoring %s from `if torch.cuda.is_available():`: %s",
            type(_exc).__name__, _exc,
        )
    return info
