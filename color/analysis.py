import numpy as np
import torch
import math
from typing import Tuple

def safe_tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Safely convert a torch tensor to numpy, handling all dtypes.
    Handles bfloat16 which raises TypeError on direct .numpy().
    Always returns float32 numpy array.
    """
    if tensor.dtype in (torch.bfloat16, torch.float16):
        return tensor.float().cpu().numpy()
    return tensor.cpu().float().numpy()


def compute_data_range(img_np: np.ndarray) -> Tuple[float, float, bool, dict]:
    """
    Compute actual data range, HDR detection, and full scene statistics.
    Returns (min_value, max_value, has_hdr, hdr_stats_dict).

    hdr_stats_dict keys:
      p1, p10, p50, p90, p99, p999  — luminance percentiles (scene-linear Y)
      mean_luma                       — mean scene-linear luma
      clipped_pct                     — % pixels with Y > 1.0
      negative_pct                    — % pixels with Y < 0.0
      scene_linear_peak               — p99.9 as scene-linear value
      ev_range                        — log2(p99/max(p1,1e-6)) in stops
      max_nit_est                     — peak nit estimate (scene_linear_peak × 203)
    """
    flat = img_np.astype(np.float32, copy=False)

    # Sanitize NaN/Inf before any stats
    finite_mask = np.isfinite(flat)
    if not finite_mask.all():
        flat = np.where(finite_mask, flat, 0.0)

    d_min = float(flat.min())
    d_max = float(flat.max())
    if not math.isfinite(d_min):
        d_min = 0.0
    if not math.isfinite(d_max):
        d_max = 1.0
    has_hdr = d_max > 1.0 or d_min < 0.0

    ndim = flat.ndim
    ch = flat.shape[-1] if ndim == 3 else 1

    if ndim == 2 or ch == 1:
        luma = flat.reshape(-1) if ndim == 2 else flat[:, :, 0].reshape(-1)
    elif ch >= 3:
        luma = (0.2126 * flat[:, :, 0] + 0.7152 * flat[:, :, 1] + 0.0722 * flat[:, :, 2]).reshape(-1)
    else:  # 2-channel fallback
        luma = flat[:, :, 0].reshape(-1)

    n_pixels = luma.size
    if n_pixels > 2_000_000:
        step = n_pixels // 2_000_000
        luma = luma[::step]

    def _safe_float(v: float) -> float:
        return round(float(v), 6) if math.isfinite(float(v)) else 0.0

    pcts = np.percentile(luma, [1, 10, 50, 90, 99, 99.9]).tolist()
    p1, p10, p50, p90, p99, p999 = [_safe_float(v) for v in pcts]
    mean_luma = _safe_float(float(np.mean(luma)))

    total = luma.size
    clipped_pct  = _safe_float(float(np.sum(luma > 1.0)) / total * 100.0)
    negative_pct = _safe_float(float(np.sum(luma < 0.0)) / total * 100.0)

    scene_linear_peak = p999
    ev_range = _safe_float(math.log2(max(p99, 1e-6) / max(p1, 1e-6))) if p99 > 0 else 0.0
    max_nit_est = _safe_float(scene_linear_peak * 203.0)

    hdr_stats = {
        "p1": p1, "p10": p10, "p50": p50,
        "p90": p90, "p99": p99, "p999": p999,
        "mean_luma": mean_luma,
        "clipped_pct": clipped_pct,
        "negative_pct": negative_pct,
        "scene_linear_peak": scene_linear_peak,
        "ev_range": ev_range,
        "max_nit_est": max_nit_est,
    }

    return d_min, d_max, has_hdr, hdr_stats
