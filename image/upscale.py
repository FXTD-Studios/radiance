import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import torch
import torch.nn.functional as F
import numpy as np

from radiance.model.cache import GPUModelCache
from radiance.core.tiling import (
    blend_weight_2d, blend_weight_2d_np,
    edge_overlaps_from_coords, clamp_overlap,
)
from typing import Tuple, Dict
import math
from enum import Enum
import logging
import threading

# Optional scipy import for high-quality Gaussian blur and Blue Noise dithering
try:
    from scipy.ndimage import zoom as _scipy_zoom
except ImportError:
    _scipy_zoom = None

# Module logger
logger = logging.getLogger("radiance.image.upscale")

# Global model cache to prevent re-initialization overhead
# Bounded LRU: this was an unbounded dict, and because self.model = model.to(device)
# mutates the cached module in place, the dict held the GPU-resident object. The
# user-facing unload_model toggle only cleared self.model, so it freed nothing.
_MODEL_CACHE = GPUModelCache(max_size=2)
_CACHE_LOCK = threading.RLock()

# ALBABIT-FIX: SUPIR models are diffusion-based (not feedforward upscalers) and
# cannot be identified or loaded by Spandrel. They require their own loading path.
_SUPIR_MODELS = {"SUPIR-v0F_fp16", "SUPIR-v0Q_fp16"}


def _find_supir_dir() -> str:
    """Return the ComfyUI-SUPIR custom-node directory, or None if not installed."""
    # upscale.py lives at:  .../ComfyUI/custom_nodes/radiance/image/upscale.py
    # ComfyUI-SUPIR lives at: .../ComfyUI/custom_nodes/ComfyUI-SUPIR/
    custom_nodes = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    for name in ("ComfyUI-SUPIR", "comfyui-supir", "ComfyUI_SUPIR"):
        d = os.path.join(custom_nodes, name)
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "nodes.py")):
            return d
    return None


# =============================================================================
# TRUE 32-BIT GAUSSIAN BLUR (replaces PIL 8-bit roundtrip)
# =============================================================================


def gaussian_blur_32bit(img: np.ndarray, sigma: float) -> np.ndarray:
    """
    True 32-bit Gaussian blur. No precision loss.
    Uses scipy if available (fast C implementation), otherwise pure numpy fallback.

    Args:
        img: HWC or HW float32 array (values can exceed 0-1 for HDR)
        sigma: Gaussian sigma in pixels

    Returns:
        Blurred float32 array, same shape as input
    """
    if sigma <= 0:
        return img.copy()

    # Try scipy first (significantly faster)
    try:
        from scipy.ndimage import gaussian_filter

        if img.ndim == 3:
            # Blur spatial dims only, not channels
            return gaussian_filter(img, sigma=[sigma, sigma, 0]).astype(np.float32)
        return gaussian_filter(img, sigma=sigma).astype(np.float32)
    except ImportError as _exc:
        logger.debug(
            "[Radiance] gaussian_blur_32bit(): ignoring %s from `from scipy.ndimage import gaussian_filter`: %s",
            type(_exc).__name__, _exc,
        )

    # Pure numpy fallback: separable 1D convolution via np.convolve.
    # NOTE: scipy is strongly recommended for production use — the numpy path is
    # correct but ~10× slower for large sigma values (e.g. sigma=15 for local contrast).
    kernel_radius = int(np.ceil(sigma * 3))
    x = np.arange(-kernel_radius, kernel_radius + 1).astype(np.float32)
    kernel_1d = np.exp(-(x**2) / (2 * sigma**2))
    kernel_1d /= kernel_1d.sum()

    def _convolve_axis(data: np.ndarray, axis: int) -> np.ndarray:
        """Apply 1D convolution along an axis using np.convolve with reflect padding."""
        pad_w = kernel_radius
        # Move target axis to front for easier iteration
        data = np.moveaxis(data, axis, 0)
        shape = data.shape
        flat = data.reshape(shape[0], -1)  # (axis_len, rest)
        result = np.empty_like(flat)
        # Reflect-pad each 1D signal and convolve
        for i in range(flat.shape[1]):
            padded = np.pad(flat[:, i], pad_w, mode="reflect")
            conv = np.convolve(padded, kernel_1d, mode="valid")
            result[:, i] = conv
        return np.moveaxis(result.reshape(shape), 0, axis)

    result = _convolve_axis(img, axis=1)  # Horizontal
    result = _convolve_axis(result, axis=0)  # Vertical
    return result.astype(np.float32)


# =============================================================================
# UPSCALING ALGORITHMS
# =============================================================================


class UpscaleMethod(Enum):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ IO"
    """
    Available upscaling methods.

    NOTE: This enum is currently unused — all method dispatch uses plain strings
    throughout _eval_kernel, _get_kernel_support, and node INPUT_TYPES lists.
    Reserved for a future typed API refactor. Do not remove until that is complete.
    """

    NEAREST = "nearest"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    LANCZOS = "lanczos"
    LANCZOS4 = "lanczos4"
    MITCHELL = "mitchell"
    CATROM = "catrom"
    HERMITE = "hermite"
    GAUSSIAN = "gaussian"


# --- Scalar kernel evaluation functions ---


def _lanczos_scalar(x: float, a: int = 3) -> float:
    """Evaluate Lanczos kernel at a single point."""
    x = abs(x)
    if x == 0:
        return 1.0
    if x >= a:
        return 0.0
    return (a * math.sin(math.pi * x) * math.sin(math.pi * x / a)) / (math.pi**2 * x**2)


def _mitchell_scalar(x: float, B: float = 1 / 3, C: float = 1 / 3) -> float:
    """Evaluate Mitchell-Netravali kernel at a single point."""
    x = abs(x)
    if x < 1:
        return (
            (12 - 9 * B - 6 * C) * x**3 + (-18 + 12 * B + 6 * C) * x**2 + (6 - 2 * B)
        ) / 6
    elif x < 2:
        return (
            (-B - 6 * C) * x**3
            + (6 * B + 30 * C) * x**2
            + (-12 * B - 48 * C) * x
            + (8 * B + 24 * C)
        ) / 6
    return 0.0


def _hermite_scalar(x: float) -> float:
    """Evaluate Hermite kernel at a single point."""
    x = abs(x)
    if x < 1:
        return 2 * x**3 - 3 * x**2 + 1
    return 0.0


def _gaussian_scalar(x: float, sigma: float = 0.7) -> float:
    """Evaluate Gaussian kernel at a single point. sigma=0.7 gives usable width."""
    return math.exp(-(x**2) / (2 * sigma**2))


# --- Vectorized kernel functions (legacy / reserved for future numpy resize path) ---
# NOTE: These five functions are not currently called by any code path.
# They are preserved as they may be wired into a future optimised numpy resize path
# that processes entire weight matrices rather than scalar kernel evaluation.


def lanczos_kernel(x: np.ndarray, a: int = 3) -> np.ndarray:
    """Lanczos kernel function (vectorized)."""
    x = np.abs(x)
    result = np.zeros_like(x)
    mask = x < a
    x_masked = x[mask]
    nonzero = x_masked != 0
    result_masked = np.zeros_like(x_masked)
    if np.any(nonzero):
        x_nz = x_masked[nonzero]
        result_masked[nonzero] = (
            a * np.sin(np.pi * x_nz) * np.sin(np.pi * x_nz / a)
        ) / (np.pi**2 * x_nz**2)
    result_masked[~nonzero] = 1.0
    result[mask] = result_masked
    return result


def mitchell_kernel(x: np.ndarray, B: float = 1 / 3, C: float = 1 / 3) -> np.ndarray:
    """Mitchell-Netravali kernel (vectorized)."""
    x = np.abs(x)
    result = np.zeros_like(x)
    mask1 = x < 1
    x1 = x[mask1]
    result[mask1] = (
        (12 - 9 * B - 6 * C) * x1**3 + (-18 + 12 * B + 6 * C) * x1**2 + (6 - 2 * B)
    ) / 6
    mask2 = (x >= 1) & (x < 2)
    x2 = x[mask2]
    result[mask2] = (
        (-B - 6 * C) * x2**3
        + (6 * B + 30 * C) * x2**2
        + (-12 * B - 48 * C) * x2
        + (8 * B + 24 * C)
    ) / 6
    return result


def catmull_rom_kernel(x: np.ndarray) -> np.ndarray:
    """Catmull-Rom spline kernel (sharper than Mitchell)."""
    return mitchell_kernel(x, B=0, C=0.5)


def hermite_kernel(x: np.ndarray) -> np.ndarray:
    """Hermite kernel (smooth)."""
    x = np.abs(x)
    result = np.zeros_like(x)
    mask = x < 1
    x_m = x[mask]
    result[mask] = 2 * x_m**3 - 3 * x_m**2 + 1
    return result


def gaussian_kernel(x: np.ndarray, sigma: float = 0.7) -> np.ndarray:
    """Gaussian kernel (vectorized). v1.1.0: sigma=0.7 (was 0.5, too narrow)."""
    return np.exp(-(x**2) / (2 * sigma**2))


# --- Kernel support and evaluation ---


def _get_kernel_support(method: str) -> float:
    """Get the support radius for a given kernel method."""
    return {
        "lanczos": 3.0,
        "lanczos4": 4.0,
        "mitchell": 2.0,
        "catrom": 2.0,
        "hermite": 1.0,
        "gaussian": 2.5,
        "bicubic": 2.0,
        "bilinear": 1.0,
        "nearest": 0.5,
    }.get(method, 2.0)


def _eval_kernel(distance: float, method: str) -> float:
    """Evaluate kernel function at a given distance from center."""
    if method == "lanczos":
        return _lanczos_scalar(distance, 3)
    elif method == "lanczos4":
        return _lanczos_scalar(distance, 4)
    elif method == "mitchell":
        return _mitchell_scalar(distance)
    elif method == "catrom":
        return _mitchell_scalar(distance, B=0, C=0.5)
    elif method == "hermite":
        return _hermite_scalar(distance)
    elif method == "gaussian":
        return _gaussian_scalar(distance)
    elif method == "bicubic":
        return _mitchell_scalar(distance, B=0, C=0.5)  # Catmull-Rom for bicubic
    elif method == "bilinear":
        x = abs(distance)
        return max(0.0, 1.0 - x)
    elif method == "nearest":
        return 1.0 if abs(distance) < 0.5 else 0.0
    return 0.0


def _eval_kernel_vectorized(distances: np.ndarray, method: str) -> np.ndarray:
    """
    Evaluate a kernel function over an entire array of distances at once.
    Dramatically faster than calling _eval_kernel scalar in a Python loop.
    Used by the vectorised separable_resize_32bit implementation.
    """
    d = np.asarray(distances, dtype=np.float32)
    if method == "lanczos":
        return lanczos_kernel(d, 3)
    elif method == "lanczos4":
        return lanczos_kernel(d, 4)
    elif method == "mitchell":
        return mitchell_kernel(d)
    elif method == "catrom" or method == "bicubic":
        return catmull_rom_kernel(d)
    elif method == "hermite":
        return hermite_kernel(d)
    elif method == "gaussian":
        return gaussian_kernel(d)
    elif method == "bilinear":
        return np.maximum(0.0, 1.0 - np.abs(d)).astype(np.float32)
    elif method == "nearest":
        return (np.abs(d) < 0.5).astype(np.float32)
    # Unknown method — fall back to bicubic weights
    return catmull_rom_kernel(d)


def _axis_weights(in_len: int, out_len: int, method: str, aa_width: float = 1.0):
    """Source indices and normalised weights for resampling one axis.

    Returns ``(idx, w)``, both ``(out_len, taps)``: output pixel ``i`` is
    ``sum_k src[idx[i, k]] * w[i, k]``. Pixel centres are aligned
    (align_corners=False) and edges are replicated.

    ``aa_width`` scales the kernel when downscaling: 1.0 is the textbook
    prefilter (kernel stretched by the scale factor), 0.0 samples the kernel
    unstretched (sharp, aliases), 2.0 is twice as wide (soft, no moire).
    Upscaling never stretches the kernel.
    """
    support = _get_kernel_support(method)
    scale = in_len / out_len
    filter_scale = max(1.0, scale * max(0.0, aa_width)) if scale > 1.0 else 1.0
    radius = int(np.ceil(support * filter_scale))
    taps = 2 * radius + 1
    centres = (np.arange(out_len, dtype=np.float64) + 0.5) * scale - 0.5
    starts = np.floor(centres).astype(np.int64) - radius
    idx = starts[:, None] + np.arange(taps, dtype=np.int64)[None, :]
    dist = (idx - centres[:, None]) / filter_scale
    w = _eval_kernel_vectorized(dist.astype(np.float32), method).astype(np.float64)
    if method == "nearest":
        # exactly one source pixel, the one the centre falls in
        w = np.zeros_like(w)
        w[np.arange(out_len), np.clip(np.round(centres).astype(np.int64) - starts, 0, taps - 1)] = 1.0
    w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-10)
    return np.clip(idx, 0, in_len - 1), w.astype(np.float32)


def separable_resize_32bit(
    img: np.ndarray, new_h: int, new_w: int, method: str = "lanczos",
    aa_width: float = 1.0,
) -> np.ndarray:
    """
    High-quality separable resize maintaining 32-bit precision (numpy, CPU).

    Same weights as torch_resize_32bit (both come from _axis_weights), so the
    CPU, GPU and tiled paths give the same picture for the same method.
    """
    h, w = img.shape[:2]
    channels = img.shape[2] if img.ndim > 2 else 1

    if new_h == h and new_w == w:
        return img.copy()

    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    img = img.astype(np.float32)

    temp = img
    if new_w != w:
        idx, wt = _axis_weights(w, new_w, method, aa_width)
        temp = (img[:, idx, :] * wt[None, :, :, None]).sum(axis=2)
    result = temp
    if new_h != h:
        idx, wt = _axis_weights(temp.shape[0], new_h, method, aa_width)
        result = np.zeros((new_h, temp.shape[1], temp.shape[2]), np.float32)
        for k in range(idx.shape[1]):
            result += temp[idx[:, k]] * wt[:, k, None, None]

    if channels == 1:
        result = result[:, :, 0]
    return result.astype(np.float32)


_DENSE_LIMIT = 48 * 1024 * 1024   # elements in one resampling matrix (192 MB)


def _resize_axis_torch(x: torch.Tensor, out_len: int, dim: int, method: str,
                       aa_width: float) -> torch.Tensor:
    """Resample BCHW ``x`` along ``dim`` (2 = rows, 3 = columns)."""
    in_len = x.shape[dim]
    if in_len == out_len:
        return x
    idx, wt = _axis_weights(in_len, out_len, method, aa_width)
    idx_t = torch.from_numpy(idx).to(x.device)
    wt_t = torch.from_numpy(wt).to(x.device, x.dtype)
    if out_len * in_len <= _DENSE_LIMIT:
        # One matmul against the (out, in) resampling matrix: BLAS / cuBLAS
        # speed, about 4x the per-tap gather on a CPU at 1080p -> 4K.
        m = torch.zeros(out_len, in_len, device=x.device, dtype=x.dtype)
        m.scatter_add_(1, idx_t, wt_t)
        return torch.matmul(x, m.T) if dim == 3 else torch.matmul(m, x)
    shape = [1] * x.dim()
    shape[dim] = out_len
    out = None
    for k in range(idx.shape[1]):       # one tap at a time: memory stays at one output
        term = x.index_select(dim, idx_t[:, k]) * wt_t[:, k].reshape(shape)
        out = term if out is None else out.add_(term)
    return out


def torch_resize_32bit(
    tensor: torch.Tensor,
    new_h: int,
    new_w: int,
    method: str = "bicubic",
    input_format: str = "BHWC",
    aa_width: float = 1.0,
) -> torch.Tensor:
    """
    Separable resize in float32 on the tensor's device, returned as BHWC.

    3.5.0: every method runs its own kernel. lanczos, lanczos4, mitchell and
    catrom used to be mapped to torch's bicubic (and hermite, gaussian to
    bilinear), so on the untiled path, the default, the method menu changed
    nothing but the label. The weights are the ones separable_resize_32bit
    uses, so GPU and CPU agree.
    """
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    if input_format == "BHWC":
        tensor = tensor.permute(0, 3, 1, 2)
    x = tensor.float()
    x = _resize_axis_torch(x, new_w, 3, method, aa_width)
    x = _resize_axis_torch(x, new_h, 2, method, aa_width)
    return x.permute(0, 2, 3, 1).contiguous()


def resolve_input_space(img: torch.Tensor, input_color_space: str) -> str:
    """'Auto' made real: a ComfyUI IMAGE is sRGB-encoded 0-1 by convention, so
    anything outside 0-1 is taken as linear / HDR. 3.5.0: Auto used to mean
    Linear, whatever came in."""
    if input_color_space != "Auto":
        return input_color_space
    rgb = img[..., :3]
    if rgb.numel() and (float(rgb.max()) > 1.0 + 1e-4 or float(rgb.min()) < -1e-4):
        return "Linear"
    return "sRGB"


# =============================================================================
# DETAIL ENHANCEMENT (v1.1.0: true 32-bit, no PIL roundtrip)
# =============================================================================


def unsharp_mask_32bit(
    img: np.ndarray, amount: float = 1.0, radius: float = 1.0, threshold: float = 0.0
) -> np.ndarray:
    """Unsharp mask in true 32-bit precision."""
    blurred = gaussian_blur_32bit(img, sigma=radius)

    # Detail mask
    mask = img - blurred

    # Apply threshold
    if threshold > 0:
        mask = np.where(np.abs(mask) > threshold, mask, 0)

    return img + mask * amount


def high_pass_sharpen_32bit(
    img: np.ndarray, strength: float = 0.5, radius: float = 3.0
) -> np.ndarray:
    """High-pass sharpening in true 32-bit."""
    low_pass = gaussian_blur_32bit(img, sigma=radius)
    high_pass = img - low_pass
    return img + high_pass * strength


def detail_enhancement_32bit(
    img: np.ndarray,
    detail_strength: float = 0.5,
    edge_strength: float = 0.3,
    local_contrast: float = 0.2,
) -> np.ndarray:
    """Multi-scale detail enhancement in true 32-bit."""
    result = img.copy()

    # Calculate luminance
    if img.ndim == 3 and img.shape[2] >= 3:
        lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    else:
        lum = img[..., 0] if img.ndim == 3 else img

    # Multi-scale detail extraction
    scales = [1.0, 2.0, 4.0]
    detail_layers = []

    prev_level = lum.copy()
    for sigma in scales:
        current_blur = gaussian_blur_32bit(
            prev_level[:, :, np.newaxis] if prev_level.ndim == 2 else prev_level,
            sigma=sigma,
        )
        if current_blur.ndim == 3:
            current_blur = current_blur[..., 0]

        detail = prev_level - current_blur
        detail_layers.append(detail)
        prev_level = current_blur

    # Combine details
    combined_detail = np.zeros_like(lum)
    weights = [0.5, 0.3, 0.2]  # Fine to coarse
    for detail, weight in zip(detail_layers, weights):
        combined_detail += detail * weight * detail_strength

    # Apply to color channels.
    # combined_detail is luma-only, so we add the same luminance delta to all channels.
    # v1.2.0: replaced misleading per-channel loop with a single vectorised broadcast.
    num_channels = min(3, img.shape[2]) if img.ndim == 3 else 1
    if img.ndim == 3:
        result[..., :num_channels] = (
            img[..., :num_channels] + combined_detail[..., np.newaxis]
        )
    else:
        result = img + combined_detail

    # Local contrast enhancement
    if local_contrast > 0:
        lum_3d = lum[:, :, np.newaxis] if lum.ndim == 2 else lum
        local_mean = gaussian_blur_32bit(lum_3d, sigma=15.0)
        if local_mean.ndim == 3:
            local_mean = local_mean[..., 0]

        local_diff = (lum - local_mean) * local_contrast
        if img.ndim == 3:
            result[..., :num_channels] += local_diff[..., np.newaxis]
        else:
            result += local_diff

    return result


# =============================================================================
# ANTI-ALIASING (v1.1.0: true 32-bit)
# =============================================================================


def apply_antialiasing_32bit(img: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """Apply edge-aware antialiasing in true 32-bit."""
    # Calculate luminance
    if img.ndim == 3 and img.shape[2] >= 3:
        lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    else:
        lum = img[..., 0] if img.ndim == 3 else img

    # Edge detection using gradient magnitude
    grad_x = np.abs(np.diff(lum, axis=1, prepend=lum[:, :1]))
    grad_y = np.abs(np.diff(lum, axis=0, prepend=lum[:1, :]))
    edges = np.sqrt(grad_x**2 + grad_y**2)
    edges = edges / (edges.max() + 1e-10)

    # True 32-bit blur
    blurred = gaussian_blur_32bit(img, sigma=1.0)

    # Blend based on edges
    edge_mask = (edges * strength)[..., np.newaxis]
    result = img * (1 - edge_mask) + blurred * edge_mask

    return result


# =============================================================================
# TILE PROCESSING
# =============================================================================


def process_tiles_32bit(
    img: np.ndarray,
    tile_size: int,
    overlap: int,
    process_func,
    output_scale: float = 1.0,
    **kwargs,
) -> np.ndarray:
    """
    Process image in tiles with overlap for seamless results.
    Maintains 32-bit precision throughout.

    v1.1.0 FIX: Eliminated infinite loop when image dimension equals tile_size.
    v1.2.0 FIX: Removed silent wasted test-tile (process_func was called on a
      throw-away tile just to measure output scale — callers always know their scale).
      Accepts explicit output_scale parameter instead.
    v1.2.0 FIX: Edge-tile crop now uses processed.shape[0:2] instead of a
      pre-computed out_tile_size, preventing seam artifacts when the process_func
      output dimensions differ from out_tile_size by rounding (e.g. model output).
    """
    h, w = img.shape[:2]
    channels = img.shape[2] if len(img.shape) > 2 else 1

    scale_h = output_scale
    scale_w = output_scale

    out_h = int(h * scale_h)
    out_w = int(w * scale_w)
    # (out_tile_size is no longer needed: blend weights are now built per tile
    #  from the tile's own dimensions, so there is nothing to pre-size.)
    out_overlap = int(overlap * scale_h)

    # Initialize output and weight buffers
    output = np.zeros((out_h, out_w, channels), dtype=np.float32)
    weights = np.zeros((out_h, out_w), dtype=np.float32)

    # Blend weights are built per tile so edges lying on the image border are
    # NOT ramped (a border pixel is covered by one tile only, so a ramp that
    # starts at 0 normalises to 0 -- a black frame around the output). At most
    # nine distinct variants exist for a given tile size, so they are cached
    # rather than rebuilt: the previous code hoisted a single weight out of the
    # loop for speed, which is what made it border-blind.
    _weight_cache: dict = {}

    def _tile_weight(th, tw, y0, y1_, x0, x1_):
        ov_t, ov_b, ov_l, ov_r = edge_overlaps_from_coords(
            y0, y1_, x0, x1_, out_h, out_w, out_overlap
        )
        key = (th, tw, ov_t, ov_b, ov_l, ov_r)
        w = _weight_cache.get(key)
        if w is None:
            w = blend_weight_2d_np(th, tw, ov_t, ov_b, ov_l, ov_r)
            _weight_cache[key] = w
        return w

    # Calculate tile positions
    overlap = clamp_overlap(tile_size, overlap)
    stride = max(1, tile_size - overlap)
    # FIX 1: "max(1, out_tile_size - out_overlap)" was a bare expression —
    # result computed and immediately discarded. out_stride is never used;
    # output tile positions are derived from input_pos * scale. Removed.

    # v1.1.0 FIX: Precompute tile positions to avoid infinite loops
    y_positions = []
    y = 0
    while y < h:
        y_positions.append(y)
        y += stride
        if y < h and y + tile_size > h:
            last_y = max(0, h - tile_size)
            if last_y != y_positions[-1]:
                y_positions.append(last_y)
            break

    x_positions = []
    x = 0
    while x < w:
        x_positions.append(x)
        x += stride
        if x < w and x + tile_size > w:
            last_x = max(0, w - tile_size)
            if last_x != x_positions[-1]:
                x_positions.append(last_x)
            break

    for y in y_positions:
        for x in x_positions:
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            tile = img[y:y_end, x:x_end]

            # Pad if needed
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                padded = np.zeros((tile_size, tile_size, channels), dtype=np.float32)
                padded[: tile.shape[0], : tile.shape[1]] = tile
                tile = padded

            processed = process_func(tile, **kwargs)

            # v1.2.0 FIX: Use actual processed dimensions rather than pre-computed
            # out_tile_size. Model output can be off by ±1 pixel due to rounding,
            # which would cause seam artifacts if we used the pre-computed value.
            actual_ph, actual_pw = processed.shape[0], processed.shape[1]

            out_y = int(y * scale_h)
            out_x = int(x * scale_w)
            out_y_end = min(out_y + actual_ph, out_h)
            out_x_end = min(out_x + actual_pw, out_w)

            tile_h = out_y_end - out_y
            tile_w = out_x_end - out_x

            weight = _tile_weight(tile_h, tile_w, out_y, out_y_end, out_x, out_x_end)

            # Vectorised over channels: the per-channel Python loop that used to
            # sit here made three strided passes and three kernel launches per
            # tile for no benefit.
            output[out_y:out_y_end, out_x:out_x_end, :] += (
                processed[:tile_h, :tile_w, :channels] * weight[..., None]
            )
            weights[out_y:out_y_end, out_x:out_x_end] += weight

    # Normalize
    weights = np.maximum(weights, 1e-10)
    for c in range(channels):
        output[..., c] /= weights

    return output


def create_tile_weight(size: int, overlap: int) -> np.ndarray:
    """
    Create smooth blending weight for tiles.

    v1.1.0 FIX: Uses outer product of 1D ramps instead of applying
    ramps independently per axis. This prevents corners from being
    darkened by double multiplication.
    """
    if overlap <= 0:
        return np.ones((size, size), dtype=np.float32)

    # Create 1D weight ramp
    weight_1d = np.ones(size, dtype=np.float32)
    ramp = np.linspace(0, 1, overlap, dtype=np.float32)

    weight_1d[:overlap] = ramp
    weight_1d[-overlap:] = ramp[::-1]

    # Outer product gives proper 2D weight without corner artifacts
    weight_2d = np.outer(weight_1d, weight_1d)

    return weight_2d


# =============================================================================
# COLOR SPACE HANDLING
# =============================================================================


def linear_to_srgb_32bit(img: np.ndarray) -> np.ndarray:
    """Convert linear to sRGB in 32-bit."""
    result = np.where(
        img <= 0.0031308,
        img * 12.92,
        1.055 * np.power(np.maximum(img, 0), 1 / 2.4) - 0.055,
    )
    return result.astype(np.float32)


def srgb_to_linear_32bit(img: np.ndarray) -> np.ndarray:
    """Convert sRGB to linear in 32-bit."""
    result = np.where(
        img <= 0.04045, img / 12.92, np.power((np.maximum(img, 0) + 0.055) / 1.055, 2.4)
    )
    return result.astype(np.float32)


# =============================================================================
# GPU-ACCELERATED PROCESSING HELPERS  (operate on BHWC float32 tensors)
# =============================================================================


def _gaussian_blur_gpu(img: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur entirely on GPU. img: (B, H, W, C) float32."""
    if sigma <= 0:
        return img.clone()
    device = img.device
    kernel_size = max(3, int(sigma * 6) | 1)          # always odd
    x = torch.arange(kernel_size, device=device).float() - kernel_size // 2
    gauss_1d = torch.exp(-(x ** 2) / (2.0 * sigma ** 2))
    gauss_1d /= gauss_1d.sum()
    gauss_2d = (gauss_1d.unsqueeze(0) * gauss_1d.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    B, H, W, C = img.shape
    bchw = img.permute(0, 3, 1, 2)                    # (B, C, H, W)
    blurred = F.conv2d(
        bchw,
        gauss_2d.expand(C, 1, -1, -1),
        padding=kernel_size // 2,
        groups=C,
    )
    return blurred.permute(0, 2, 3, 1)                # (B, H, W, C)


def _srgb_to_linear_gpu(img: torch.Tensor) -> torch.Tensor:
    """sRGB → linear, any tensor shape."""
    return torch.where(
        img <= 0.04045,
        img / 12.92,
        torch.pow(torch.clamp((img + 0.055) / 1.055, min=0.0), 2.4),
    )


def _linear_to_srgb_gpu(img: torch.Tensor) -> torch.Tensor:
    """Linear → sRGB, any tensor shape."""
    return torch.where(
        img <= 0.0031308,
        img * 12.92,
        1.055 * torch.pow(torch.clamp(img, min=0.0), 1.0 / 2.4) - 0.055,
    )


def _unsharp_mask_gpu(
    img: torch.Tensor,
    amount: float = 1.0,
    radius: float = 1.0,
    threshold: float = 0.0,
) -> torch.Tensor:
    """Unsharp mask on GPU. img: (B, H, W, C) float32."""
    blurred = _gaussian_blur_gpu(img, sigma=radius)
    mask = img - blurred
    if threshold > 0.0:
        mask = torch.where(mask.abs() > threshold, mask, torch.zeros_like(mask))
    return img + mask * amount


def _high_pass_sharpen_gpu(
    img: torch.Tensor, strength: float = 0.5, radius: float = 3.0
) -> torch.Tensor:
    """High-pass sharpening on GPU. img: (B, H, W, C) float32."""
    low_pass = _gaussian_blur_gpu(img, sigma=radius)
    return img + (img - low_pass) * strength


def _detail_enhancement_gpu(
    img: torch.Tensor, detail_strength: float = 0.5
) -> torch.Tensor:
    """Multi-scale detail enhancement on GPU. img: (B, H, W, C) float32 (RGB or RGBA)."""
    lum = (
        0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    ).unsqueeze(-1)                                    # (B, H, W, 1)
    scales     = [1.0, 2.0, 4.0]
    ms_weights = [0.5, 0.3, 0.2]
    combined = torch.zeros_like(lum)
    prev = lum
    for sigma, w in zip(scales, ms_weights):
        blur = _gaussian_blur_gpu(prev, sigma=sigma)
        combined += (prev - blur) * (w * detail_strength)
        prev = blur
    result = img.clone()
    n_col = min(3, img.shape[-1])
    result[..., :n_col] = img[..., :n_col] + combined
    return result


def _antialiasing_gpu(img: torch.Tensor, strength: float = 0.5) -> torch.Tensor:
    """Edge-aware antialiasing on GPU. img: (B, H, W, C) float32."""
    lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    gx = F.pad((lum[:, :, 1:] - lum[:, :, :-1]).abs(), (1, 0))        # left-pad
    gy = F.pad((lum[:, 1:, :] - lum[:, :-1, :]).abs(), (0, 0, 1, 0))  # top-pad
    edges = (gx ** 2 + gy ** 2).sqrt()
    edges = edges / (edges.amax(dim=(1, 2), keepdim=True) + 1e-10)
    blurred = _gaussian_blur_gpu(img, sigma=1.0)
    mask = (edges * strength).unsqueeze(-1)
    return img * (1.0 - mask) + blurred * mask


# =============================================================================
# FLUX-OPTIMIZED PRESETS
# =============================================================================

FLUX_PRESETS = {
    "Flux Default": {
        "description": "Balanced upscaling optimized for Flux outputs",
        "method": "lanczos",
        "sharpening": 0.3,
        "detail_enhancement": 0.2,
        "antialiasing": 0.3,
        "color_space": "sRGB",
    },
    "Flux Sharp": {
        "description": "Sharp upscaling for detailed Flux images",
        "method": "lanczos4",
        "sharpening": 0.6,
        "detail_enhancement": 0.4,
        "antialiasing": 0.2,
        "color_space": "sRGB",
    },
    "Flux Smooth": {
        "description": "Smooth upscaling for soft Flux renders",
        "method": "mitchell",
        "sharpening": 0.1,
        "detail_enhancement": 0.1,
        "antialiasing": 0.5,
        "color_space": "sRGB",
    },
    "Flux HDR": {
        "description": "HDR-aware upscaling for high dynamic range",
        "method": "lanczos",
        "sharpening": 0.25,
        "detail_enhancement": 0.3,
        "antialiasing": 0.3,
        "color_space": "Linear",
    },
    "Flux Print": {
        "description": "High quality for print output",
        "method": "lanczos4",
        "sharpening": 0.5,
        "detail_enhancement": 0.5,
        "antialiasing": 0.15,
        "color_space": "sRGB",
    },
    "Flux Cinematic": {
        "description": "Film-like upscaling with subtle softness",
        "method": "catrom",
        "sharpening": 0.2,
        "detail_enhancement": 0.15,
        "antialiasing": 0.4,
        "color_space": "Linear",
    },
    "Flux Maximum": {
        "description": "Maximum detail preservation",
        "method": "lanczos4",
        "sharpening": 0.7,
        "detail_enhancement": 0.6,
        "antialiasing": 0.1,
        "color_space": "sRGB",
    },
}


# =============================================================================
# SHARED METHOD LIST (v1.1.0: includes gaussian)
# =============================================================================

METHOD_LIST_FULL = [
    "lanczos",
    "lanczos4",
    "bicubic",
    "mitchell",
    "catrom",
    "hermite",
    "gaussian",
    "bilinear",
    "nearest",
]

METHOD_LIST_QUALITY = [
    "lanczos",
    "lanczos4",
    "bicubic",
    "mitchell",
    "catrom",
    "gaussian",
    "bilinear",
]


# =============================================================================
# MAIN COMFYUI NODES
# =============================================================================


class RadianceProUpscale:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ IO"
    """
    Professional 32-bit upscaler optimized for Flux with HDR support.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        preset_list = ["Custom"] + list(FLUX_PRESETS.keys())

        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Image or batch to resize, display-encoded sRGB or linear float (see "
                    "input_color_space). Alpha is resized separately.",
                }),
                "scale_factor": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 0.1,
                        "max": 8.0,
                        "step": 0.1,
                        "display": "slider",
                        "tooltip": "Output size as a multiple of the input (rounded down to whole pixels). "
                        "Below 1.0 downscales.",
                    },
                ),
                "preset": (preset_list, {
                    "tooltip": "Custom uses the widgets below. Any other preset overrides method, sharpening, "
                    "detail_enhancement and antialiasing, and the HDR/Cinematic presets force process_in_linear on.",
                }),
            },
            "optional": {
                "method": (METHOD_LIST_FULL, {
                    "tooltip": "Resampling kernel, run exactly on every path (GPU, CPU and tiled): lanczos "
                    "(3 lobes), lanczos4, bicubic and catrom (Catmull-Rom), mitchell (B=C=1/3), hermite, "
                    "gaussian, bilinear, nearest.",
                }),
                "sharpening": (
                    "FLOAT",
                    {"default": 0.3, "min": 0.0, "max": 2.0, "step": 0.05,
                     "tooltip": "Unsharp-mask amount applied after resizing. 0 = off, 1 = add the full "
                     "detail difference once."},
                ),
                "sharpen_radius": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.5, "max": 5.0, "step": 0.1,
                     "tooltip": "Gaussian sigma in output pixels for the unsharp mask. Larger sharpens "
                     "coarser detail."},
                ),
                "detail_enhancement": (
                    "FLOAT",
                    {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05,
                     "tooltip": "Multi-scale luma detail boost (blur sigmas 1, 2 and 4 px) added equally "
                     "to RGB after resizing. 0 = off."},
                ),
                "antialiasing": (
                    "FLOAT",
                    {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                     "tooltip": "Edge-aware softening applied last: blends in a 1 px Gaussian blur in "
                     "proportion to edge strength. 0 = off; counteracts sharpening on edges."},
                ),
                "input_color_space": (["sRGB", "Linear", "Auto"], {
                    "tooltip": "Encoding of the input. sRGB allows the linear-light conversion; Linear leaves "
                    "values untouched. Auto: Linear when any value is outside 0-1 (HDR), else sRGB; the "
                    "info output says which.",
                }),
                "process_in_linear": ("BOOLEAN", {"default": True,
                    "tooltip": "Decode sRGB to linear before resampling and re-encode after, for gamma-correct "
                    "filtering. Only acts when input_color_space is sRGB; linear/HDR input is never converted.",
                }),
                "use_tiles": ("BOOLEAN", {"default": False,
                    "tooltip": "Process image in overlapping tiles to handle large images that exceed VRAM.",
                }),
                "tile_size": (
                    "INT",
                    {"default": 512, "min": 128, "max": 2048, "step": 64,
                     "tooltip": "Input tile size in pixels for the tiled CPU path. Used only when tiling "
                     "is on (or forced above 64 MP) and the image is larger than this."},
                ),
                "tile_overlap": (
                    "INT",
                    {"default": 64, "min": 16, "max": 256, "step": 16,
                     "tooltip": "Overlap in input pixels between tiles, blended to hide seams. Tiled "
                     "path only."},
                ),
                "output_bit_depth": (["32-bit Float", "16-bit Float", "8-bit"], {
                    "tooltip": "Precision of the result (always returned as float). 32-bit keeps values "
                    "above 1.0; 16-bit rounds to half precision; 8-bit clamps to 0-1 and quantises to 256 levels.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("upscaled_image", "width", "height", "info")
    FUNCTION = "upscale"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Upscale"
    DESCRIPTION = "Professional 32-bit upscaler optimized for Flux with HDR support."

    def upscale(
        self,
        image: torch.Tensor,
        scale_factor: float,
        preset: str,
        method: str = "lanczos",
        sharpening: float = 0.3,
        sharpen_radius: float = 1.0,
        detail_enhancement: float = 0.2,
        antialiasing: float = 0.3,
        input_color_space: str = "sRGB",
        process_in_linear: bool = True,
        use_tiles: bool = False,
        tile_size: int = 512,
        tile_overlap: int = 64,
        output_bit_depth: str = "32-bit Float",
    ):

        # Apply preset if not custom
        if preset != "Custom" and preset in FLUX_PRESETS:
            p = FLUX_PRESETS[preset]
            method = p.get("method", method)
            sharpening = p.get("sharpening", sharpening)
            detail_enhancement = p.get("detail_enhancement", detail_enhancement)
            antialiasing = p.get("antialiasing", antialiasing)
            if p.get("color_space") == "Linear":
                process_in_linear = True
            logger.info(
                f"Preset '{preset}' applied — overriding: method={method}, "
                f"sharpening={sharpening}, detail={detail_enhancement}, aa={antialiasing}. "
                f"Set preset='Custom' to use your own parameter values."
            )

        batch_size = image.shape[0]
        h, w = image.shape[1], image.shape[2]
        new_h = int(h * scale_factor)
        new_w = int(w * scale_factor)

        # v1.1.0: Auto-Tiling Safety for Large Images
        # v1.2.0 FIX: Check max(input, output) — a 32MP input at scale_factor<1 can still
        # be too large for the un-tiled numpy path even if the output is small.
        SAFE_PIXEL_LIMIT = 64 * 1024 * 1024  # 64 Megapixels
        if max(h * w, new_h * new_w) > SAFE_PIXEL_LIMIT and not use_tiles:
            logger.warning(
                f"Image size exceeds 64MP safety limit "
                f"(input={w}x{h}, output={new_w}x{new_h}). Forcing tiled processing."
            )
            use_tiles = True

        needs_tiling = use_tiles and (h > tile_size or w > tile_size)
        auto_space = input_color_space == "Auto"
        input_color_space = resolve_input_space(image, input_color_space)

        if not needs_tiling:
            # ── GPU batch path (no per-frame loop, no numpy) ──────────────────
            device = torch.device("cuda") if torch.cuda.is_available() else image.device
            img = image.to(device).float()

            has_alpha = img.shape[-1] == 4
            if has_alpha:
                alpha = img[..., 3:4]
                img = img[..., :3]

            if process_in_linear and input_color_space == "sRGB":
                img = _srgb_to_linear_gpu(img)

            upscaled = torch_resize_32bit(img, new_h, new_w, method, input_format="BHWC")

            if detail_enhancement > 0:
                upscaled = _detail_enhancement_gpu(upscaled, detail_enhancement)

            if sharpening > 0:
                upscaled = _unsharp_mask_gpu(upscaled, amount=sharpening, radius=sharpen_radius)

            if antialiasing > 0:
                upscaled = _antialiasing_gpu(upscaled, strength=antialiasing)

            if process_in_linear and input_color_space == "sRGB":
                upscaled = _linear_to_srgb_gpu(upscaled)

            if has_alpha:
                alpha_up = torch_resize_32bit(alpha, new_h, new_w, "lanczos", input_format="BHWC")
                upscaled = torch.cat([upscaled, alpha_up], dim=-1)

            # Bit depth conversion on GPU
            if output_bit_depth == "16-bit Float":
                upscaled = upscaled.half().float()
            elif output_bit_depth == "8-bit":
                upscaled = (upscaled.clamp(0, 1) * 255).round() / 255.0
            # v1.1.1 FIX: 32-bit Float mode (default) no longer clamps to [0,1]
            # preserving HDR range for professional workflows.

            output_tensor = upscaled.cpu()
        else:
            # ── CPU/numpy tiled path (unchanged — tiling logic requires numpy) ─
            results = []

            for b in range(batch_size):
                img = image[b].cpu().numpy().astype(np.float32)

                has_alpha = img.shape[-1] == 4
                if has_alpha:
                    alpha = img[..., 3:4]
                    img = img[..., :3]

                if process_in_linear and input_color_space == "sRGB":
                    img = srgb_to_linear_32bit(img)

                def upscale_tile(tile, **kw):
                    th, tw = tile.shape[:2]
                    new_th = int(th * scale_factor)
                    new_tw = int(tw * scale_factor)
                    return separable_resize_32bit(tile, new_th, new_tw, method)

                upscaled = process_tiles_32bit(
                    img,
                    tile_size,
                    tile_overlap,
                    upscale_tile,
                    output_scale=scale_factor,
                )

                if detail_enhancement > 0:
                    upscaled = detail_enhancement_32bit(upscaled, detail_enhancement)

                if sharpening > 0:
                    upscaled = unsharp_mask_32bit(upscaled, sharpening, sharpen_radius)

                if antialiasing > 0:
                    upscaled = apply_antialiasing_32bit(upscaled, antialiasing)

                if process_in_linear and input_color_space == "sRGB":
                    upscaled = linear_to_srgb_32bit(upscaled)

                if has_alpha:
                    alpha_up = separable_resize_32bit(alpha, new_h, new_w, "lanczos")
                    if len(alpha_up.shape) == 2:
                        alpha_up = alpha_up[:, :, np.newaxis]
                    upscaled = np.concatenate([upscaled, alpha_up], axis=-1)

                if output_bit_depth == "16-bit Float":
                    upscaled = upscaled.astype(np.float16).astype(np.float32)
                elif output_bit_depth == "8-bit":
                    upscaled = np.clip(upscaled, 0, 1)
                    upscaled = np.round(upscaled * 255.0).astype(np.float32) / 255.0

                results.append(upscaled)

            output = np.stack(results, axis=0)
            output_tensor = torch.from_numpy(output).float()

        info = f"Upscaled: {w}x{h} → {new_w}x{new_h} ({scale_factor}x)\n"
        info += f"Method: {method}\n"
        info += f"Preset: {preset}\n"
        info += f"Sharpening: {sharpening}, Detail: {detail_enhancement}\n"
        info += f"Input: {input_color_space}{' (Auto)' if auto_space else ''}\n"
        info += f"Output: {output_bit_depth}"

        return (output_tensor, new_w, new_h, info)


class RadianceUpscaleBySize:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ IO"
    """
    Upscale to exact dimensions with aspect ratio control.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Image or batch to resize, display-encoded sRGB or linear float (see "
                    "input_color_space). Alpha is resized separately.",
                }),
                "width": ("INT", {"default": 2048, "min": 64, "max": 16384, "step": 8,
                    "tooltip": "Target output width in pixels.",
                }),
                "height": (
                    "INT",
                    {"default": 2048, "min": 64, "max": 16384, "step": 8,
                     "tooltip": "Target output height in pixels."},
                ),
                "method": (METHOD_LIST_FULL, {
                    "tooltip": "Resampling kernel, run exactly as named: lanczos (3 lobes), lanczos4, bicubic "
                    "and catrom (Catmull-Rom), mitchell (B=C=1/3), hermite, gaussian, bilinear, nearest.",
                }),
            },
            "optional": {
                "maintain_aspect": ("BOOLEAN", {"default": True,
                    "tooltip": "Keep the source aspect ratio using aspect_mode to fit the width x height box. "
                    "Off resizes to exactly width x height.",
                }),
                "aspect_mode": (["fit", "fill", "stretch"], {
                    "tooltip": "With maintain_aspect on. fit: largest size inside the box. fill: smallest "
                    "size covering the box (not cropped, so one side exceeds it). stretch: exactly width x height.",
                }),
                "sharpening": (
                    "FLOAT",
                    {"default": 0.2, "min": 0.0, "max": 2.0, "step": 0.05,
                     "tooltip": "Unsharp-mask amount (1 px sigma) applied after resizing. 0 = off."},
                ),
                "process_in_linear": ("BOOLEAN", {"default": True,
                    "tooltip": "Decode sRGB to linear before resampling and re-encode after, for gamma-correct "
                    "filtering. Only acts when input_color_space is sRGB; linear/HDR input is never converted.",
                }),
                "input_color_space": (
                    ["sRGB", "Linear", "Auto"],
                    {
                        "default": "sRGB",
                        "tooltip": "Colour space of the input. Linear skips the sRGB-to-linear conversion "
                        "for HDR/linear input. Auto: Linear when any value is outside 0-1, else sRGB.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("upscaled_image", "final_width", "final_height")
    FUNCTION = "upscale"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Upscale"
    DESCRIPTION = "Upscale to exact dimensions with aspect ratio control."

    def upscale(
        self,
        image: torch.Tensor,
        width: int,
        height: int,
        method: str,
        maintain_aspect: bool = True,
        aspect_mode: str = "fit",
        sharpening: float = 0.2,
        process_in_linear: bool = True,
        input_color_space: str = "sRGB",
    ):

        batch_size = image.shape[0]
        orig_h, orig_w = image.shape[1], image.shape[2]

        # Calculate target size
        if maintain_aspect:
            aspect_ratio = orig_w / orig_h

            if aspect_mode == "fit":
                if width / height > aspect_ratio:
                    new_h = height
                    new_w = int(height * aspect_ratio)
                else:
                    new_w = width
                    new_h = int(width / aspect_ratio)
            elif aspect_mode == "fill":
                if width / height > aspect_ratio:
                    new_w = width
                    new_h = int(width / aspect_ratio)
                else:
                    new_h = height
                    new_w = int(height * aspect_ratio)
            else:  # stretch
                new_w = width
                new_h = height
        else:
            new_w = width
            new_h = height

        input_color_space = resolve_input_space(image, input_color_space)
        device = torch.device("cuda") if torch.cuda.is_available() else image.device
        img = image.to(device).float()

        has_alpha = img.shape[-1] == 4
        if has_alpha:
            alpha = img[..., 3:4]
            img = img[..., :3]

        if process_in_linear and input_color_space == "sRGB":
            img = _srgb_to_linear_gpu(img)

        upscaled = torch_resize_32bit(img, new_h, new_w, method, input_format="BHWC")

        if sharpening > 0:
            upscaled = _unsharp_mask_gpu(upscaled, amount=sharpening, radius=1.0)

        # v1.2.0 FIX: gate on input_color_space so HDR/linear inputs are not
        # double-converted. process_in_linear alone was insufficient — the node
        # had no way to tell sRGB inputs from scene-linear inputs.
        if process_in_linear and input_color_space == "sRGB":
            upscaled = _linear_to_srgb_gpu(upscaled)

        if has_alpha:
            alpha_up = torch_resize_32bit(alpha, new_h, new_w, "lanczos", input_format="BHWC")
            upscaled = torch.cat([upscaled, alpha_up], dim=-1)

        return (upscaled.cpu(), new_w, new_h)


class RadianceDownscale32bit:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ IO"
    """
    GPU-accelerated 32-bit downscaling with anti-aliasing.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Image or batch to downscale, display-encoded sRGB or linear float (see "
                    "input_color_space). Values above 1.0 are kept.",
                }),
                "scale_factor": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.01,
                        "max": 1.0,
                        "step": 0.05,
                        "display": "slider",
                        "tooltip": "Output size as a fraction of the input (rounded down, minimum 1 px).",
                    },
                ),
                "method": (METHOD_LIST_QUALITY, {
                    "tooltip": "Resampling kernel, run exactly as named on CUDA and CPU alike: lanczos, "
                    "lanczos4, bicubic (Catmull-Rom), mitchell, catrom, gaussian, bilinear.",
                }),
            },
            "optional": {
                "antialiasing": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                     "tooltip": "Width of the downscaling prefilter. 0.5 = the kernel's textbook width "
                     "(stretched by the scale factor), 0 = no prefilter (sharpest, aliases and moires), "
                     "1 = twice as wide (softest). pre_blur adds a Gaussian on top."},
                ),
                "pre_blur": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.1,
                     "tooltip": "Gaussian sigma in input pixels applied before downscaling to suppress "
                     "aliasing and moire. 0 = off."},
                ),
                "process_in_linear": ("BOOLEAN", {"default": True,
                    "tooltip": "Decode sRGB to linear before resampling and re-encode after, for gamma-correct "
                    "filtering. Only acts when input_color_space is sRGB; linear/HDR input is never converted.",
                }),
                "use_gpu": ("BOOLEAN", {"default": True,
                    "tooltip": "Run on CUDA when available (MPS is not used). Falls back to the CPU path, "
                    "which uses the exact resampling kernels.",
                }),
                # FIX 5: parity with RadianceProUpscale / RadianceUpscaleBySize.
                # Without this param, linear/HDR inputs were always run through
                # sRGB linearization when process_in_linear=True, double-converting.
                "input_color_space": (
                    ["sRGB", "Linear", "Auto"],
                    {
                        "default": "sRGB",
                        "tooltip": (
                            "Colour space of the input image. Linear skips the sRGB-linear conversion "
                            "for HDR or already-linear input. Auto: Linear when any value is outside "
                            "0-1, else sRGB."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("downscaled_image", "width", "height")
    FUNCTION = "downscale"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Upscale"
    DESCRIPTION = "32-bit downscaling with the exact resampling kernel on GPU or CPU and an adjustable anti-aliasing prefilter."

    def downscale(
        self,
        image: torch.Tensor,
        scale_factor: float,
        method: str,
        antialiasing: float = 0.5,
        pre_blur: float = 0.0,
        process_in_linear: bool = True,
        use_gpu: bool = True,
        input_color_space: str = "sRGB",  # FIX 5
    ):

        batch_size = image.shape[0]
        h, w = image.shape[1], image.shape[2]
        new_h = max(1, int(h * scale_factor))
        new_w = max(1, int(w * scale_factor))

        input_color_space = resolve_input_space(image, input_color_space)
        # 3.5.0: antialiasing is the prefilter width (it was never read);
        # 0.5 keeps the textbook filter the node always used.
        aa_width = 2.0 * float(antialiasing)
        linearise = process_in_linear and input_color_space == "sRGB"

        if use_gpu and torch.cuda.is_available():
            try:
                device = torch.device("cuda")
                img = image.to(device).float()
                alpha = img[..., 3:] if img.shape[-1] > 3 else None
                img = img[..., :3]
                if linearise:
                    img = _srgb_to_linear_gpu(img)
                if pre_blur > 0:
                    img = _gaussian_blur_gpu(img, pre_blur)
                result = torch_resize_32bit(img, new_h, new_w, method, aa_width=aa_width)
                if linearise:
                    result = _linear_to_srgb_gpu(result)
                if alpha is not None:
                    alpha = torch_resize_32bit(alpha, new_h, new_w, "lanczos", aa_width=aa_width)
                    result = torch.cat([result, alpha], dim=-1)
                # No output clamp: HDR values survive the downscale.
                return (result.cpu(), new_w, new_h)

            except RuntimeError as e:  # FIX 2: capture + log the error
                logger.warning(
                    f"[RadianceDownscale32bit] GPU path failed "
                    f"({type(e).__name__}: {e}). Falling back to CPU."
                )
                torch.cuda.empty_cache()

        # CPU fallback
        results = []

        for b in range(batch_size):
            img = image[b].cpu().numpy().astype(np.float32)

            has_alpha = img.shape[-1] == 4
            if has_alpha:
                alpha = img[..., 3:4]
                img = img[..., :3]

            if linearise:
                img = srgb_to_linear_32bit(img)

            # Pre-blur in true 32-bit
            if pre_blur > 0:
                img = gaussian_blur_32bit(img, sigma=pre_blur)

            downscaled = separable_resize_32bit(img, new_h, new_w, method, aa_width)

            if linearise:
                downscaled = linear_to_srgb_32bit(downscaled)

            if has_alpha:
                alpha_down = separable_resize_32bit(alpha, new_h, new_w, "lanczos", aa_width)
                if len(alpha_down.shape) == 2:
                    alpha_down = alpha_down[:, :, np.newaxis]
                downscaled = np.concatenate([downscaled, alpha_down], axis=-1)

            results.append(downscaled)

        output = np.stack(results, axis=0)
        output_tensor = torch.from_numpy(output).float()

        return (output_tensor, new_w, new_h)


class RadianceBitDepthConvert:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ IO"
    """
    Convert between bit depths with professional dithering to reduce banding.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image in float32 format."}),
                "output_depth": (
                    ["32-bit Float", "16-bit Float", "16-bit Int", "10-bit", "8-bit"],
                    {"tooltip": "Target bit depth."},
                ),
            },
            "optional": {
                "dithering": (
                    ["None", "Floyd-Steinberg", "Ordered", "Blue Noise", "Random"],
                    {
                        "default": "None",
                        "tooltip": "Dithering algorithm. Floyd-Steinberg = error diffusion, Ordered = Bayer, Blue Noise = high-frequency, Random = simple noise.",
                    },
                ),
                "dither_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.1,
                        "tooltip": "Dithering intensity. 1.0 = standard.",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "tooltip": "Random seed for reproducible Blue Noise and Random dithering.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("converted_image", "bit_depth_info")
    OUTPUT_TOOLTIPS = (
        "Image quantized to target bit depth.",
        "Information about the conversion.",
    )
    FUNCTION = "convert"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Upscale"
    DESCRIPTION = (
        "Convert between bit depths with professional dithering to reduce banding."
    )

    def convert(
        self,
        image: torch.Tensor,
        output_depth: str,
        dithering: str = "None",
        dither_strength: float = 1.0,
        seed: int = 0,
    ):

        info = f"Converted to {output_depth}"

        # ── Float storage: no integer quantisation, so no dithering ──────────
        # FIX-014: "16-bit Float" with any dithering fell through to the
        # integer path with levels defaulting to 255, i.e. it became an 8-bit
        # quantise with values clamped to [0, 1]. Float targets now only round
        # to their storage precision and keep HDR values and negatives.
        if output_depth in ("32-bit Float", "16-bit Float"):
            out = image.float() if output_depth == "32-bit Float" else image.half().float()
            if dithering != "None":
                info += f" (dithering '{dithering}' not applied: float storage is not quantised to integer levels)"
            return (out, info)

        levels_map = {"16-bit Int": 65535, "10-bit": 1023, "8-bit": 255}
        if output_depth not in levels_map:
            raise ValueError(f"Bit Depth Convert: unknown output_depth {output_depth!r}.")
        levels = levels_map[output_depth]

        # ── Floyd-Steinberg: inherently sequential, stays on CPU per-frame ────
        if dithering == "Floyd-Steinberg":
            results = []
            for b in range(image.shape[0]):
                img_np = image[b].cpu().numpy().astype(np.float32)
                img_np = self._apply_dither(img_np, levels, dithering, dither_strength, seed)
                results.append(np.clip(img_np, 0, 1))
            info += f" with {dithering} dithering"
            return (torch.from_numpy(np.stack(results, axis=0)).float(), info)

        # ── GPU path for all remaining cases ──────────────────────────────────
        device = torch.device("cuda") if torch.cuda.is_available() else image.device
        img = image.to(device).float()
        B, H, W, C = img.shape
        n_color = min(3, C)

        if dithering == "Ordered":
            bayer_base = torch.tensor([
                [0, 8, 2, 10], [12, 4, 14, 6],
                [3, 11, 1,  9], [15, 7, 13, 5],
            ], dtype=torch.float32, device=device) / 16.0 - 0.5
            tile_h = (H + 3) // 4
            tile_w = (W + 3) // 4
            noise = bayer_base.repeat(tile_h, tile_w)[:H, :W]          # (H, W)
            noise = noise.unsqueeze(0).unsqueeze(-1) * (dither_strength / levels)
            img = img.clone()
            img[..., :n_color] = img[..., :n_color] + noise
            info += f" with {dithering} dithering"

        elif dithering in ("Blue Noise", "Random"):
            g = torch.Generator()
            g.manual_seed(seed)
            if dithering == "Blue Noise":
                n1 = torch.randn(B, H, W, 1, generator=g).to(device)
                n2 = torch.randn(B, H // 2 + 1, W // 2 + 1, 1, generator=g).to(device)
                n2_up = F.interpolate(
                    n2.permute(0, 3, 1, 2), size=(H, W),
                    mode="bilinear", align_corners=False,
                ).permute(0, 2, 3, 1)
                noise = (n1 - n2_up * 0.5) * (dither_strength / levels)
            else:
                noise = torch.randn(B, H, W, 1, generator=g).to(device) * (dither_strength / levels)
            img = img.clone()
            img[..., :n_color] = img[..., :n_color] + noise
            info += f" with {dithering} dithering"

        # v1.2.0 FIX: Floyd-Steinberg performs quantization IN-PLACE during error
        # diffusion and returns a fully-quantized result — handled above.  All
        # other paths return a PRE-quantization float, so we quantize here.
        result = torch.clamp(torch.round(img * levels) / levels, 0.0, 1.0)
        return (result.cpu(), info)

    def _apply_dither(
        self, img: np.ndarray, levels: int, method: str, strength: float, seed: int = 0
    ) -> np.ndarray:
        """
        Apply dithering before quantization.

        Floyd-Steinberg returns a FULLY QUANTIZED result. All other methods
        return a pre-quantization float (caller applies round/clip).
        """
        h, w = img.shape[:2]
        channels = img.shape[2] if img.ndim == 3 else 1
        rng = np.random.default_rng(seed)

        if method == "Floyd-Steinberg":
            # v1.1.0 FIX: Real error diffusion instead of random noise.
            # v1.2.0 PERF: Scan-line numpy approach — O(H) Python row iterations
            # instead of O(H×W), with O(W) numpy vector operations per row.
            #
            # Within-row 7/16 carry is approximated: all pixels in a row are
            # quantized simultaneously from their pre-carry values, then errors
            # are diffused to the NEXT row with exact FS coefficients.  For
            # typical VFX bit-depth conversions (32→10-bit or 32→8-bit) the
            # visual difference from exact sequential FS is imperceptible.
            result = img.copy()
            for c in range(channels):
                ch = result[..., c] if img.ndim == 3 else result
                for y in range(h):
                    row = ch[y].copy()  # current row values
                    new_row = np.round(row * levels) / levels  # quantize whole row
                    errors = (row - new_row) * strength  # per-pixel errors (w,)
                    ch[y] = new_row

                    # 7/16 forward within-row (approximate: applied after full-row quant)
                    ch[y, 1:] += errors[:-1] * (7.0 / 16)

                    if y + 1 < h:
                        # 5/16 directly below
                        ch[y + 1, :] += errors * (5.0 / 16)
                        # 1/16 below-right
                        ch[y + 1, 1:] += errors[:-1] * (1.0 / 16)
                        # 3/16 below-left
                        ch[y + 1, :-1] += errors[1:] * (3.0 / 16)

            # Return fully-quantized result — convert() must NOT re-round
            return result

        elif method == "Ordered":
            # Bayer 4×4 dithering.
            # v1.2.0 FIX: restrict noise to RGB channels only — alpha should never
            # be dithered as it would corrupt transparency masks.
            bayer = (
                np.array(
                    [
                        [0, 8, 2, 10],
                        [12, 4, 14, 6],
                        [3, 11, 1, 9],
                        [15, 7, 13, 5],
                    ],
                    dtype=np.float32,
                )
                / 16.0
                - 0.5
            )

            bayer_tiled = np.tile(bayer, (h // 4 + 1, w // 4 + 1))[:h, :w]
            noise = bayer_tiled * (strength / levels)  # (h, w) — scalar

            result = img.copy()
            n_color = min(3, channels)  # RGB only, not alpha
            if img.ndim == 3:
                result[..., :n_color] += noise[..., np.newaxis]
            else:
                result += noise
            return result

        elif method == "Blue Noise":
            # High-frequency noise approximation.
            # v1.2.0 FIX: scipy import moved to module scope (was inside method body).
            # v1.2.0 FIX: seeded RNG for reproducibility.
            noise1 = rng.standard_normal((h, w, 1)).astype(np.float32)
            noise2 = rng.standard_normal((h // 2 + 1, w // 2 + 1, 1)).astype(np.float32)

            if _scipy_zoom is not None:
                noise2_up = _scipy_zoom(
                    noise2, [h / noise2.shape[0], w / noise2.shape[1], 1], order=1
                )
                noise2_up = noise2_up[:h, :w, :]
            else:
                noise2_up = np.repeat(np.repeat(noise2, 2, axis=0), 2, axis=1)[
                    :h, :w, :
                ]

            blue_noise = (noise1 - noise2_up * 0.5) * (strength / levels)
            # FIX 3: (h,w,1) noise broadcasts to ALL channels including alpha.
            # Ordered was fixed in v1.2.0; Blue Noise + Random were missed.
            _r3a = img.copy()
            _nc3a = min(3, channels)  # RGB only, never alpha
            if img.ndim == 3:
                _r3a[..., :_nc3a] += blue_noise
            else:
                _r3a += blue_noise
            return _r3a

        elif method == "Random":
            # v1.2.0 FIX: seeded RNG for reproducibility
            noise = rng.standard_normal((h, w, 1)).astype(np.float32) * (
                strength / levels
            )
            # FIX 3: same alpha-channel contamination as Blue Noise. RGB only.
            _r3b = img.copy()
            _nc3b = min(3, channels)  # RGB only, never alpha
            if img.ndim == 3:
                _r3b[..., :_nc3b] += noise
            else:
                _r3b += noise
            return _r3b

        return img


# =============================================================================
# AI UPSCALER INTEGRATION
# =============================================================================


class RadianceAIUpscale:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ IO"
    """
    AI-powered upscaling using neural network models. Supports tiled processing for large images.
    """

    AI_MODELS = [
        "RealESRGAN_x4plus",
        "RealESRGAN_x4plus_anime_6B",
        "RealESRGAN_x2plus",
        "ESRGAN_4x",
        "4x-UltraSharp",
        "4x-AnimeSharp",
        "SwinIR_4x",
        "HAT_4x",
        "SUPIR-v0F_fp16",
        "SUPIR-v0Q_fp16",
    ]

    # 3.5.0: each file pinned to its SHA-256 and size and fetched through
    # radiance.core.model_fetch (verified before it is installed, resumable).
    # SUPIR from Kijai/SUPIR_pruned at a fixed commit; Real-ESRGAN from its
    # original GitHub releases.
    MODEL_FILES = {
        "SUPIR-v0F_fp16": (
            "https://huggingface.co/Kijai/SUPIR_pruned/resolve/eaabd8ecd86906f97626a39f3b9fe882d52d697d/SUPIR-v0F_fp16.safetensors",
            "a8f1846de1985cf0473fac6d8c0ef17c9498a90d1c19906e944134c5572275d0", 2664825592),
        "SUPIR-v0Q_fp16": (
            "https://huggingface.co/Kijai/SUPIR_pruned/resolve/eaabd8ecd86906f97626a39f3b9fe882d52d697d/SUPIR-v0Q_fp16.safetensors",
            "3eef33ec7633122ca23b1e5ef167faa048b5a0845768694d5e8070138ac013ce", 2664858464),
        "RealESRGAN_x4plus": (
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1", 67040989),
        "RealESRGAN_x4plus_anime_6B": (
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
            "f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da", 17938799),
        "RealESRGAN_x2plus": (
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
            "49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb", 67061725),
    }
    MODEL_URLS = {name: spec[0] for name, spec in MODEL_FILES.items()}

    def __init__(self):
        self.model = None
        self.current_model_name = None
        # True when the last upscale() returned the bicubic fallback instead of
        # running a model. The node's outputs cannot carry it, so callers that
        # must not ship a resample as an AI upscale (Delivery) read it here.
        self.used_fallback = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image to upscale."}),
                "model_name": (
                    cls.AI_MODELS,
                    {
                        "default": "RealESRGAN_x4plus",
                        "tooltip": "AI upscaling model. RealESRGAN_x4plus is recommended for general use.",
                    },
                ),
                "mode": (
                    ["Standard", "Refine (HDR)", "Normalize (HDR)"],
                    {
                        "default": "Standard",
                        "tooltip": "Processing mode. 'Standard' = direct upscale. 'Refine (HDR)' = log-compression for highlights. 'Normalize (HDR)' = scales to safe range.",
                    },
                ),
                "tile_size": (
                    "INT",
                    {
                        "default": 512,
                        "min": 128,
                        "max": 1024,
                        "step": 64,
                        "tooltip": "Tile size for processing. Smaller = less VRAM, slower.",
                    },
                ),
                "tile_overlap": (
                    "INT",
                    {
                        "default": 32,
                        "min": 0,
                        "max": 128,
                        "step": 8,
                        "tooltip": "Overlap between tiles to avoid seams.",
                    },
                ),
                "auto_download": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Automatically download models if not found.",
                    },
                ),
            },
            "optional": {
                "unload_model": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Unload model from VRAM after processing to free memory.",
                    },
                ),
                # ALBABIT-FIX: SUPIR-specific inputs. SUPIR_model_loader merges SUPIR weights
                # into an SDXL base model and creates its own internal VAE/CLIP — it does NOT
                # use the vae/clip inputs below. Those are kept for potential future v2 support.
                "sdxl_model_name": ("STRING", {
                    "default": "",
                    "tooltip": "SUPIR only: filename of your SDXL base checkpoint "
                               "(e.g. sd_xl_base_1.0_0.9vae.safetensors). "
                               "Leave empty to auto-detect from the checkpoints folder. "
                               "Ignored for all non-SUPIR models.",
                }),
                "supir_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "SUPIR only: text description for SUPIR upscaling. "
                               "Leave empty for default conditioning. Ignored for all other models.",
                }),
                "vae": ("VAE", {
                    "tooltip": "Reserved for future SUPIR v2 loader support. Currently unused.",
                }),
                "clip": ("CLIP", {
                    "tooltip": "Reserved for future SUPIR v2 loader support. Currently unused.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    OUTPUT_TOOLTIPS = ("Upscaled image.", "Information about the upscaling process.")
    FUNCTION = "upscale"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Upscale"
    DESCRIPTION = "AI-powered upscaling using neural network models. Supports tiled processing for large images."

    def _download_model(self, model_name: str, target_path: str) -> bool:
        """Fetch a pinned model file (see MODEL_FILES). True when it is installed."""
        if model_name not in self.MODEL_FILES:
            return False
        url, sha256, size = self.MODEL_FILES[model_name]
        from radiance.core.model_fetch import ModelFetchError, fetch
        try:
            fetch(url, target_path, sha256=sha256, size=size, label=model_name)
            return True
        except ModelFetchError as e:
            logger.error("%s", e)
            return False

    def _load_supir_model(self, model_name: str, model_path: str, sdxl_model_name: str = ""):
        """Load a SUPIR model via the ComfyUI-SUPIR extension bridge.

        ALBABIT-FIX: Complete rewrite. Debug log revealed three things:
          1. nodes.NODE_CLASS_MAPPINGS DOES contain SUPIR classes — we were looking for
             the wrong key ('SUPIRModelLoader' instead of 'SUPIR_model_loader').
          2. ComfyUI-SUPIR's loader also requires an SDXL base model filename
             (sdxl_model param). A new sdxl_model_name input is now forwarded here.
          3. The loader returns (SUPIRMODEL, SUPIRVAE) — both must be stored in the
             cache tuple for use by _run_supir.
        """
        import sys

        # ALBABIT-FIX: Use nodes.NODE_CLASS_MAPPINGS with the correct key names for this
        # version of ComfyUI-SUPIR. The old code used 'SUPIRModelLoader' which does not
        # exist — the real key is 'SUPIR_model_loader'.
        loader_cls = None
        _ncm = {}

        try:
            import nodes as _comfy_nodes
            _ncm = getattr(_comfy_nodes, "NODE_CLASS_MAPPINGS", {})
            loader_cls = _ncm.get("SUPIR_model_loader")
        except (ImportError, AttributeError) as _exc:
            logger.debug(
                "[Radiance] _load_supir_model(): ignoring %s from `import nodes as _comfy_nodes`: %s",
                type(_exc).__name__, _exc,
            )

        # Fallback: ComfyUI 0.19.x stores the module under its full directory path
        if loader_cls is None:
            supir_dir = _find_supir_dir()
            if supir_dir:
                _mod = sys.modules.get(supir_dir)
                if _mod is not None:
                    _ncm_try = _mod.__dict__.get("NODE_CLASS_MAPPINGS", {})
                    if isinstance(_ncm_try, dict) and "SUPIR_model_loader" in _ncm_try:
                        loader_cls = _ncm_try["SUPIR_model_loader"]
                        _ncm = _ncm_try

        if loader_cls is None:
            supir_dir = _find_supir_dir()
            if supir_dir is None:
                return None, (
                    "SUPIR is a diffusion-based upscaler — Spandrel cannot load it. "
                    "To enable SUPIR in this node:\n"
                    "  1. Install ComfyUI-SUPIR via ComfyUI Manager (search 'SUPIR' by kijai).\n"
                    "  2. Restart ComfyUI so it loads the extension.\n"
                    "  3. Set 'sdxl_model_name' to your SDXL base model filename "
                    "(e.g. sd_xl_base_1.0_0.9vae.safetensors)."
                )
            return None, (
                "ComfyUI-SUPIR is installed but SUPIR_model_loader was not found in "
                "NODE_CLASS_MAPPINGS. Check the ComfyUI startup log for errors in "
                "comfyui-supir, then restart ComfyUI."
            )

        # Build the class map with actual names from this version of ComfyUI-SUPIR
        supir_cls_map = {
            "SUPIR_model_loader": loader_cls,
            "SUPIR_encode":       _ncm.get("SUPIR_encode"),
            "SUPIR_conditioner":  _ncm.get("SUPIR_conditioner"),
            "SUPIR_sample":       _ncm.get("SUPIR_sample"),
            "SUPIR_decode":       _ncm.get("SUPIR_decode"),
        }

        # Auto-detect SDXL model if not specified
        if not sdxl_model_name:
            import folder_paths as _fp
            _ckpts = _fp.get_filename_list("checkpoints")
            for _c in _ckpts:
                if "xl" in _c.lower() and "base" in _c.lower():
                    sdxl_model_name = _c
                    break
            if not sdxl_model_name:
                for _c in _ckpts:
                    if "xl" in _c.lower():
                        sdxl_model_name = _c
                        break
            if not sdxl_model_name:
                return None, (
                    "SUPIR requires an SDXL base model. Set 'sdxl_model_name' to the "
                    "filename of your SDXL checkpoint (e.g. sd_xl_base_1.0_0.9vae.safetensors) "
                    "or place an SDXL model in your checkpoints folder."
                )
            logger.info(f"[RadianceAIUpscale] SUPIR: auto-detected SDXL model '{sdxl_model_name}'")

        try:
            import folder_paths as fp
            loader = loader_cls()
            model_filename = os.path.basename(model_path)
            model_dir      = os.path.dirname(model_path)

            # SUPIR_model_loader looks up both models in "checkpoints" via folder_paths.
            # Temporarily register the SUPIR model's directory there if it is not already.
            _tmp_buckets = []
            for bucket in ("upscale_models", "checkpoints"):
                if model_dir not in fp.get_folder_paths(bucket):
                    fp.add_model_folder_path(bucket, model_dir)
                    _tmp_buckets.append(bucket)

            try:
                result = loader.process(
                    supir_model=model_filename,
                    sdxl_model=sdxl_model_name,
                    diffusion_dtype="auto",
                    fp8_unet=False,
                )
            finally:
                for bucket in _tmp_buckets:
                    try:
                        fp.folder_names_and_paths[bucket][0].remove(model_dir)
                    except (ValueError, KeyError) as _exc:
                        logger.debug(
                            "[Radiance] _load_supir_model(): ignoring %s from `fp.folder_names_and_paths[bucket][0].remove(model_dir)`: %s",
                            type(_exc).__name__, _exc,
                        )

            if not result or result[0] is None:
                return None, "SUPIR_model_loader returned an empty result — check ComfyUI-SUPIR logs."

            # result = (SUPIRMODEL, SUPIRVAE)
            supir_model_obj = result[0]
            supir_vae_obj   = result[1]
            entry = ("supir", supir_model_obj, supir_vae_obj, supir_cls_map)
            _MODEL_CACHE.put(model_name, entry)
            logger.info(f"[RadianceAIUpscale] SUPIR loaded: {model_name} + {sdxl_model_name}")
            return entry, f"SUPIR loaded: {model_name} + {sdxl_model_name}"

        except Exception as e:
            import traceback as _tb
            logger.error(f"[RadianceAIUpscale] SUPIR load failed: {e}\n" + _tb.format_exc())
            return None, f"SUPIR load failed: {e}"

    def _run_supir(self, model_tuple, image, tile_size, tile_overlap, prompt=""):
        """Run SUPIR inference using ComfyUI-SUPIR node classes as a backend.

        ALBABIT-FIX: Full rewrite to match the real ComfyUI-SUPIR API (nodes_v2.py):
          - model_tuple is now ("supir", SUPIRMODEL, SUPIRVAE, cls_map)
          - SUPIR_encode takes SUPIR_VAE (not vae/clip — those are baked into the model)
          - SUPIR_conditioner takes latents + SUPIR_model
          - SUPIR_sample is the diffusion step
          - SUPIR_decode takes SUPIR_VAE
          - All four nodes handle batches internally; we pass the full image batch.
        """
        import inspect

        _, supir_model, supir_vae, supir_cls_map = model_tuple

        def _call(cls_name, method_name, **kwargs):
            cls = supir_cls_map.get(cls_name)
            if cls is None:
                raise RuntimeError(f"{cls_name} not found in ComfyUI-SUPIR node registry — update ComfyUI-SUPIR.")
            obj    = cls()
            method = getattr(obj, method_name, None)
            if method is None:
                raise RuntimeError(f"{cls_name}.{method_name}() not found")
            sig    = inspect.signature(method)
            # Forward only kwargs the method accepts; skip None values.
            filtered = {k: v for k, v in kwargs.items() if k in sig.parameters and v is not None}
            return method(**filtered)

        # ── Step 1 : encode LQ frames to SUPIR latent space ──────────────────
        # SUPIR_encode processes the batch internally frame-by-frame.
        encode_result = _call("SUPIR_encode", "encode",
            SUPIR_VAE=supir_vae,
            image=image,
            use_tiled_vae=True,
            encoder_tile_size=tile_size,
            encoder_dtype="auto",
        )
        latents = encode_result[0]  # {"samples": tensor, "original_size": [H, W]}

        # ── Step 2 : build positive/negative conditioning ─────────────────────
        cond_result = _call("SUPIR_conditioner", "condition",
            SUPIR_model=supir_model,
            latents=latents,
            positive_prompt=prompt or "high quality, detailed",
            negative_prompt="blurry, low quality, noise, artifacts, compression",
        )
        positive = cond_result[0]
        negative = cond_result[1]

        # ── Step 3 : diffusion sampling ───────────────────────────────────────
        sample_result = _call("SUPIR_sample", "sample",
            SUPIR_model=supir_model,
            latents=latents,
            positive=positive,
            negative=negative,
            seed=42,
            steps=45,
            cfg_scale_start=4.0,
            cfg_scale_end=4.0,
            EDM_s_churn=5,
            s_noise=1.003,
            DPMPP_eta=1.0,
            control_scale_start=1.0,
            control_scale_end=1.0,
            restore_cfg=-1.0,
            keep_model_loaded=False,
            sampler="RestoreEDMSampler",
        )
        output_latents = sample_result[0]

        # ── Step 4 : decode latents back to pixel space ───────────────────────
        decode_result = _call("SUPIR_decode", "decode",
            SUPIR_VAE=supir_vae,
            latents=output_latents,
            use_tiled_vae=True,
            decoder_tile_size=tile_size,
        )
        out_img = decode_result[0]  # (B, H, W, C) float32, already cpu

        info = f"SUPIR upscale: {image.shape[0]} frame(s) via ComfyUI-SUPIR"
        return out_img, info

    def _load_model(self, model_name: str, sdxl_model_name: str = "", auto_download: bool = True):
        """Load an upscale model with caching."""
        with _CACHE_LOCK:
            # Check cache first
            if model_name in _MODEL_CACHE:
                return _MODEL_CACHE.get(model_name), "Loaded (Cached)"

            try:
                import folder_paths
                import comfy.utils
            except ImportError:
                return None, "ComfyUI modules not available"

            # Find model path
            model_path = folder_paths.get_full_path(
                "upscale_models", f"{model_name}.pth"
            )
            if model_path is None:
                model_path = folder_paths.get_full_path(
                    "upscale_models", f"{model_name}.safetensors"
                )

            if model_path is None:
                # Try auto-download
                models_dir = folder_paths.get_folder_paths("upscale_models")[0]
                ext = ".safetensors" if "SUPIR" in model_name else ".pth"
                target_path = os.path.join(models_dir, f"{model_name}{ext}")

                # The node's auto_download widget and RADIANCE_ALLOW_DOWNLOADS=0
                # (checked inside the fetch) can both say no.
                if auto_download and self._download_model(model_name, target_path):
                    model_path = target_path
                else:
                    why = ("auto_download is off" if not auto_download
                           else "the download failed or downloads are turned off; see the log")
                    return (
                        None,
                        f"Model {model_name} not found and not downloaded: {why}. "
                        f"Place it in models/upscale_models/",
                    )

            # ALBABIT-FIX: SUPIR models are diffusion-based and cannot be identified by
            # Spandrel (which only handles feedforward upscalers). Route them to a
            # dedicated loader that uses the ComfyUI-SUPIR extension when available.
            if model_name in _SUPIR_MODELS:
                return self._load_supir_model(model_name, model_path, sdxl_model_name)

            # Load standard upscale models with Spandrel
            try:
                sd = comfy.utils.load_torch_file(model_path, safe_load=True)

                try:
                    import spandrel
                except ImportError:
                    return None, "Spandrel library not found. Please update ComfyUI."

                try:
                    # Handle state dict wrapping if necessary
                    if "model" in sd:
                        sd = sd["model"]
                    elif "state_dict" in sd:
                        sd = sd["state_dict"]

                    model_descriptor = spandrel.ModelLoader().load_from_state_dict(sd)
                    upscale_model = model_descriptor.model.eval()

                    _MODEL_CACHE.put(model_name, upscale_model)
                    return upscale_model, f"Loaded: {model_name}"

                except Exception as e:
                    # ALBABIT-FIX: Spandrel raises UnsupportedModelError with no message
                    # when it cannot identify the architecture (str(e) == ""). Replace the
                    # silent empty string with a human-readable diagnostic.
                    err_msg = str(e)
                    if not err_msg:
                        err_msg = (
                            "Architecture not recognised by Spandrel (UnsupportedModelError). "
                            "Ensure the model is a supported ESRGAN / SwinIR / HAT variant "
                            "and that your ComfyUI Spandrel version is up to date."
                        )
                    logger.error(f"Spandrel load failed for {model_name}: {err_msg}")
                    return None, f"Model load error: {err_msg}"

            except Exception as e:
                # v1.1.0 FIX: Removed destructive auto-deletion of "small" model files.
                # Log the error and let the user investigate.
                if os.path.exists(model_path):
                    size = os.path.getsize(model_path)
                    if size < 1000:
                        logger.warning(
                            f"◎ Model file '{model_name}' is suspiciously small ({size} bytes). "
                            f"It may be corrupt or a failed download. Consider deleting and re-downloading: "
                            f"{model_path}"
                        )

                return None, f"Error loading model: {type(e).__name__}: {str(e)}"

    def _fallback_upscale(self, image, model_name):
        """Fallback to algorithmic upscale when AI model unavailable."""
        self.used_fallback = True
        logger.warning("Falling back to bicubic upscale (AI model not loaded)")

        # ── Validate input shape (must be BHWC) ──────────────────────────────
        if not isinstance(image, torch.Tensor) or image.dim() != 4:
            raise RuntimeError(
                f"[RadianceAIUpscale] Expected 4D BHWC tensor, got "
                f"{type(image).__name__} with shape {getattr(image, 'shape', '?')}"
            )

        b, h, w, c = image.shape

        # Detect BCHW-format tensors passed by mistake (channels in dim-1)
        # Typical: B=1, C=3 or 4, H=large, W=large.
        # Heuristic: if c > 32 it is almost certainly H, not channels.
        if c > 32:
            raise RuntimeError(
                f"[RadianceAIUpscale] Input tensor looks like BCHW (last dim={c}), "
                f"expected BHWC. Shape: {tuple(image.shape)}. "
                "Permute the tensor to BHWC before passing to this node."
            )

        scale = 4
        if "x2" in model_name.lower():
            scale = 2
        elif "x8" in model_name.lower():
            scale = 8

        new_h, new_w = h * scale, w * scale

        # ── Memory safety cap: 64 MP per frame ──────────────────────────────
        MAX_OUTPUT_PIXELS = 64_000_000  # 64 megapixels (~16K×4K)
        output_pixels = new_h * new_w
        if output_pixels > MAX_OUTPUT_PIXELS:
            cap_factor = (MAX_OUTPUT_PIXELS / (h * w)) ** 0.5
            safe_scale = max(1, int(cap_factor))
            new_h, new_w = h * safe_scale, w * safe_scale
            logger.warning(
                f"[RadianceAIUpscale] Fallback x{scale} would produce {output_pixels:,} pixels "
                f"per frame. Capping to x{safe_scale} ({new_h}×{new_w}) to avoid OOM."
            )
            scale = safe_scale

        # ALBABIT-FIX: memory warning and interpolation are now per-frame.
        # Previous code called F.interpolate on the full batch tensor (all B frames at
        # once), which for a 121-frame video at 4× resulted in a ~45 GB allocation and
        # a misleading console warning that included the batch dimension in the estimate.
        approx_bytes_per_frame = new_h * new_w * c * 4  # float32, single frame
        if approx_bytes_per_frame * b > 4 * 1024 ** 3:
            logger.warning(
                f"[RadianceAIUpscale] Fallback upscale: {b} frame(s) × "
                f"~{approx_bytes_per_frame / 1024**2:.0f} MB each "
                f"= ~{approx_bytes_per_frame * b / 1024**3:.1f} GB total. "
                f"Consider using a smaller image or fewer frames."
            )

        results = []
        for b_idx in range(b):
            frame = image[b_idx : b_idx + 1].float().cpu().permute(0, 3, 1, 2)
            up = F.interpolate(frame, size=(new_h, new_w), mode="bicubic", align_corners=False)
            results.append(up.permute(0, 2, 3, 1)[0])

        result = torch.stack(results)
        return (result, f"Bicubic {scale}x (AI model not available)")

    def _hdr_compress(self, img: torch.Tensor, mode: str) -> Tuple[torch.Tensor, Dict]:
        """Compress HDR values into a range better handled by AI models."""
        metadata = {}

        if mode == "Refine (HDR)":
            # Logarithmic compression: log(1 + x)
            # Preserves 0 as 0, compresses helps with high highs
            # Models see '1' as '0.69', '10' as '2.4'
            #
            # v1.2.0 FIX: log1p requires non-negative input.  Negative linear values
            # (valid in wide-gamut / ACEScg scenes) are silently clamped to 0, destroying
            # out-of-gamut data.  Warn the operator so they can tone-map beforehand.
            min_val = img.min().item()
            if min_val < 0:
                logger.warning(
                    "[RadianceAIUpscale] _hdr_compress 'Refine (HDR)': input contains "
                    "negative linear values (min=%.4f). These will be clamped to 0 before "
                    "log-compression, permanently destroying out-of-gamut data. "
                    "Apply a tone-mapping node before AI upscale if this is undesirable.",
                    min_val,
                )
            return torch.log1p(torch.clamp(img, min=0)), metadata

        elif mode == "Normalize (HDR)":
            # Linear normalization based on max value
            max_val = torch.max(img)
            # Avoid divide by zero or scaling up LDR images unnecessarily
            min_divisor = 1.0
            divisor = max(max_val.item(), min_divisor)
            metadata["divisor"] = divisor
            return img / divisor, metadata

        return img, metadata

    def _hdr_expand(self, img: torch.Tensor, mode: str, metadata: Dict) -> torch.Tensor:
        """Expand compressed values back to original HDR range."""
        if mode == "Refine (HDR)":
            # Inverse log: exp(x) - 1
            return torch.expm1(img)

        elif mode == "Normalize (HDR)":
            divisor = metadata.get("divisor", 1.0)
            return img * divisor

        return img

    def upscale(
        self,
        image,
        model_name: str = "RealESRGAN_x4plus",
        mode: str = "Standard",
        tile_size: int = 512,
        tile_overlap: int = 32,
        auto_download: bool = True,
        unload_model: bool = False,
        # ALBABIT-FIX: SUPIR optional inputs.
        sdxl_model_name: str = "",
        supir_prompt: str = "",
        vae=None,
        clip=None,
    ):
        """Upscale image using AI model with tiled processing."""
        self.used_fallback = False

        # Load model if needed
        if self.model is None or self.current_model_name != model_name:
            # ALBABIT-FIX: forward sdxl_model_name to _load_supir_model via _load_model routing
            self.model, load_info = self._load_model(model_name, sdxl_model_name=sdxl_model_name,
                                                     auto_download=bool(auto_download))
            self.current_model_name = model_name

            if self.model is None:
                logger.warning(f"{load_info}")
                return self._fallback_upscale(image, model_name)

        # ALBABIT-FIX: Route SUPIR models to the dedicated diffusion inference path.
        # _load_supir_model caches the result as a ("supir", model, nodes_module) tuple.
        if isinstance(self.model, tuple) and self.model[0] == "supir":
            try:
                result_images, info = self._run_supir(
                    self.model, image, tile_size, tile_overlap, supir_prompt
                )
            except Exception as e:
                import traceback as _tb
                logger.error(
                    f"[RadianceAIUpscale] SUPIR inference failed: {e}\n" + _tb.format_exc()
                )
                return self._fallback_upscale(image, model_name)

            if unload_model:
                # Drop the cache entry too. Clearing self.model alone left the
                # module referenced by _MODEL_CACHE -- and since .to(device)
                # mutates in place, that reference was the GPU-resident copy,
                # so empty_cache() reclaimed nothing at all.
                _MODEL_CACHE.pop(self.current_model_name or model_name)
                self.model = None
                self.current_model_name = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                info += " (model unloaded)"

            return (result_images, info)

        try:
            from comfy import model_management

            device = model_management.get_torch_device()
            self.model = self.model.to(device)

            scale = 4
            if "x2" in model_name.lower():
                scale = 2
            elif "x8" in model_name.lower():
                scale = 8

            result_images = []

            for batch_idx in range(image.shape[0]):
                img_in = image[batch_idx : batch_idx + 1]

                # Apply HDR compression
                img_compressed, hdr_meta = self._hdr_compress(img_in, mode)

                # Permute to BCHW for processing
                img = img_compressed.permute(0, 3, 1, 2)
                _, c, h, w = img.shape

                if h <= tile_size and w <= tile_size:
                    with torch.no_grad():
                        img_device = img.to(device)
                        output = self.model(img_device)
                    result_img = output.permute(0, 2, 3, 1)  # Back to BHWC
                else:
                    # Tiled processing
                    new_h, new_w = h * scale, w * scale
                    output = torch.zeros(
                        (1, c, new_h, new_w), dtype=torch.float32, device=device
                    )
                    weight = torch.zeros(
                        (1, 1, new_h, new_w), dtype=torch.float32, device=device
                    )

                    tile_overlap = clamp_overlap(tile_size, tile_overlap)
                    stride = max(1, tile_size - tile_overlap)
                    tiles_x = max(1, math.ceil((w - tile_overlap) / stride))
                    tiles_y = max(1, math.ceil((h - tile_overlap) / stride))
                    _tile_w_cache: dict = {}

                    logger.info(
                        f"Processing {tiles_x * tiles_y} tiles ({tiles_x}x{tiles_y})..."
                    )

                    for ty in range(tiles_y):
                        for tx in range(tiles_x):
                            x1 = min(tx * stride, w - tile_size)
                            y1 = min(ty * stride, h - tile_size)
                            x2 = min(x1 + tile_size, w)
                            y2 = min(y1 + tile_size, h)
                            x1 = max(0, x2 - tile_size)
                            y1 = max(0, y2 - tile_size)

                            tile = img[:, :, y1:y2, x1:x2].to(device)

                            with torch.no_grad():
                                tile_output = self.model(tile)

                            out_x1 = x1 * scale
                            out_y1 = y1 * scale
                            out_x2 = x2 * scale
                            out_y2 = y2 * scale

                            th, tw = tile_output.shape[2], tile_output.shape[3]

                            # Border-aware, and cached: at most nine distinct
                            # weights exist for a tile size, but this rebuilt a
                            # full-size outer product on every tile -- ~670MB of
                            # allocator churn per 4K frame. The ramp also started
                            # at 0 on image-border edges, blacking out the
                            # outermost row and column of the result.
                            _ov_t, _ov_b, _ov_l, _ov_r = edge_overlaps_from_coords(
                                out_y1, out_y2, out_x1, out_x2,
                                new_h, new_w, tile_overlap * scale,
                            )
                            _wkey = (th, tw, _ov_t, _ov_b, _ov_l, _ov_r)
                            tile_weight = _tile_w_cache.get(_wkey)
                            if tile_weight is None:
                                tile_weight = blend_weight_2d(
                                    th, tw, _ov_t, _ov_b, _ov_l, _ov_r,
                                    device=device, dtype=torch.float32,
                                )
                                _tile_w_cache[_wkey] = tile_weight

                            output[:, :, out_y1:out_y2, out_x1:out_x2] += (
                                tile_output * tile_weight
                            )
                            weight[:, :, out_y1:out_y2, out_x1:out_x2] += tile_weight

                    # In-place: the non-in-place form allocated two more
                    # full-resolution tensors (~2GB extra at 4K/4x) at the
                    # moment model weights are also resident.
                    weight.clamp_(min=1e-8)
                    output.div_(weight)
                    result_img = output.permute(0, 2, 3, 1)  # Back to BHWC

                # Expand back to original HDR range
                result_expanded = self._hdr_expand(result_img.cpu(), mode, hdr_meta)
                result_images.append(result_expanded[0])

            result = torch.stack(result_images)

            # For Standard/Normalize modes, we might want to clamp min to 0
            # But avoid clamping max for HDR modes
            result = torch.clamp(result, min=0)

            info = f"Upscaled with {model_name} ({scale}x) [{mode}]"

            if unload_model:
                # Drop the cache entry too. Clearing self.model alone left the
                # module referenced by _MODEL_CACHE -- and since .to(device)
                # mutates in place, that reference was the GPU-resident copy,
                # so empty_cache() reclaimed nothing at all.
                _MODEL_CACHE.pop(self.current_model_name or model_name)
                self.model = None
                self.current_model_name = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                info += " (model unloaded)"
                logger.info(f"Model unloaded from VRAM")

            return (result, info)

        except Exception as e:
            # FIX 4: traceback.print_exc() goes to stderr, bypassing the logger.
            # In ComfyUI server mode the traceback was silently lost.
            import traceback as _tb
            logger.error(
                f"[RadianceAIUpscale] Error during upscale: {e}\n"
                + _tb.format_exc()
            )
            return self._fallback_upscale(image, model_name)


# =============================================================================
# SHARPENING NODE
# v1.2.0: Implemented — was listed in v1.1.0 changelog but class was missing.
# =============================================================================


class RadianceSharpen32bit:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ IO"
    """
    Professional 32-bit sharpening node for ComfyUI.
    Offers two modes: Unsharp Mask (standard) and High-Pass (for heavy sharpening
    without ringing). Both operate in true float32 with no 8-bit roundtrip.
    Full HDR support — values outside [0, 1] are preserved.
    """

    SHARPEN_MODES = ["Unsharp Mask", "High Pass"]

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (
                    cls.SHARPEN_MODES,
                    {
                        "default": "Unsharp Mask",
                        "tooltip": "Unsharp Mask: classic radius/amount sharpening. "
                        "High Pass: overlay-style sharpening with less ringing.",
                    },
                ),
            },
            "optional": {
                "amount": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 5.0,
                        "step": 0.05,
                        "display": "slider",
                        "tooltip": "Sharpening strength. 0 = no change. >1 = heavy sharpen.",
                    },
                ),
                "radius": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": "Blur radius used to detect edges (pixels). "
                        "Larger radius = coarser detail enhancement.",
                    },
                ),
                "threshold": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 0.5,
                        "step": 0.005,
                        "tooltip": "Unsharp Mask only: minimum contrast to sharpen. "
                        "Raise to protect smooth gradients from noise amplification.",
                    },
                ),
                "process_in_linear": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Apply sharpening in linear light for physically correct results. "
                        "Disable if input is already in the target perceptual space.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("sharpened_image",)
    FUNCTION = "sharpen"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Upscale"
    DESCRIPTION = (
        "Professional 32-bit sharpening — Unsharp Mask or High Pass, "
        "true float32 precision, HDR-safe."
    )

    def sharpen(
        self,
        image: torch.Tensor,
        mode: str = "Unsharp Mask",
        amount: float = 0.5,
        radius: float = 1.0,
        threshold: float = 0.0,
        process_in_linear: bool = True,
    ):

        device = torch.device("cuda") if torch.cuda.is_available() else image.device
        img = image.to(device).float()

        # Preserve and strip alpha before processing
        has_alpha = img.shape[-1] == 4
        if has_alpha:
            alpha = img[..., 3:4]
            img = img[..., :3]

        if process_in_linear:
            img = _srgb_to_linear_gpu(img)

        if mode == "Unsharp Mask":
            sharpened = _unsharp_mask_gpu(img, amount=amount, radius=radius, threshold=threshold)
        else:  # High Pass
            sharpened = _high_pass_sharpen_gpu(img, strength=amount, radius=radius)

        if process_in_linear:
            sharpened = _linear_to_srgb_gpu(sharpened)

        if has_alpha:
            sharpened = torch.cat([sharpened, alpha], dim=-1)

        return (sharpened.cpu(),)


# =============================================================================
# NODE MAPPINGS
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "RadianceProUpscale": RadianceProUpscale,
    "RadianceUpscaleBySize": RadianceUpscaleBySize,
    "RadianceDownscale32bit": RadianceDownscale32bit,
    "RadianceBitDepthConvert": RadianceBitDepthConvert,
    "RadianceAIUpscale": RadianceAIUpscale,
    "RadianceSharpen32bit": RadianceSharpen32bit,
    
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceProUpscale": "◎ Radiance Pro Upscale",
    "RadianceUpscaleBySize": "◎ Radiance Upscale By Size",
    "RadianceDownscale32bit": "◎ Radiance Downscale 32-bit",
    "RadianceBitDepthConvert": "◎ Radiance Bit Depth Convert",
    "RadianceAIUpscale": "◎ Radiance AI Upscale",
    "RadianceSharpen32bit": "◎ Radiance Sharpen 32-bit",
    
}
