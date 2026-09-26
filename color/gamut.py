"""Gamut operations — tone mapping, gamut compression, and perceptual spaces."""
from __future__ import annotations

import numpy as np

from radiance.color.matrices import HPE_MATRIX, HPE_MATRIX_INV


# ── ACES Filmic Tonemap (Narkowicz) ───────────────────────────────────────────

def aces_tonemap(img: np.ndarray) -> np.ndarray:
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    res = (img * (a * img + b)) / (img * (c * img + d) + e)
    return np.clip(res, 0.0, 1.0)


# ── Approximate ACES Tone Mapping (Michaelis-Menten) ─────────────────────────

def aces_approx_tonemap(
    img: np.ndarray,
    peak_luminance: float = 1000.0,
    mid_gray: float = 0.18,
    contrast: float = 1.0,
) -> np.ndarray:
    display_scale = 48.0 / peak_luminance
    p = 1.2 * contrast
    img_safe = np.maximum(img, 1e-10)
    img_pow = np.power(img_safe / mid_gray, p)
    tonemapped = img_pow / (img_pow + 1.0)
    result = tonemapped * display_scale

    highlight_threshold = 0.9
    highlight_mask = result > highlight_threshold
    if np.any(highlight_mask):
        excess = result[highlight_mask] - highlight_threshold
        result[highlight_mask] = highlight_threshold + 0.1 * np.tanh(excess * 10)

    return np.clip(result, 0.0, 1.0).astype(np.float32)


# Backward-compatibility alias
aces2_tonemap = aces_approx_tonemap


# ── ACES 2.0 Gamut Compression ───────────────────────────────────────────────

def aces2_gamut_compress(
    img: np.ndarray, threshold: float = 0.75, power: float = 1.2
) -> np.ndarray:
    luma = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    luma = np.maximum(luma, 1e-10)[..., np.newaxis]

    rgb_norm = img / luma
    distance = np.max(np.abs(rgb_norm - 1.0), axis=-1, keepdims=True)

    compress_mask = distance > threshold
    if np.any(compress_mask):
        excess = (distance - threshold) / (1.0 - threshold + 1e-10)
        compressed_distance = threshold + (1.0 - threshold) * (
            1.0 - np.power(np.maximum(1.0 - excess, 0.0), power)
        )
        scale = np.where(compress_mask, compressed_distance / (distance + 1e-10), 1.0)
        img_compressed = luma + (img - luma) * scale
    else:
        img_compressed = img

    return img_compressed.astype(np.float32)


# ── Simplified JMh (perceptual space) ────────────────────────────────────────

_D65_WHITE = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)


def linear_to_jmh(img: np.ndarray, white_point: np.ndarray | None = None) -> np.ndarray:
    if white_point is None:
        white_point = _D65_WHITE

    lms = np.dot(img, HPE_MATRIX.T)
    lms = np.maximum(lms, 1e-10)
    lms_w = np.dot(white_point, HPE_MATRIX.T)
    lms_adapted = lms / lms_w
    lms_c = np.power(lms_adapted, 0.42)

    J = 100.0 * ((lms_c[..., 0] + lms_c[..., 1]) / 2.0)
    a = lms_c[..., 0] - lms_c[..., 1]
    b = 0.5 * (lms_c[..., 0] + lms_c[..., 1]) - lms_c[..., 2]

    M = np.sqrt(a**2 + b**2) * 100.0
    h = np.degrees(np.arctan2(b, a)) % 360.0

    return np.stack([J, M, h], axis=-1).astype(np.float32)


def jmh_to_linear(jmh: np.ndarray, white_point: np.ndarray | None = None) -> np.ndarray:
    if white_point is None:
        white_point = _D65_WHITE

    J, M, h = jmh[..., 0], jmh[..., 1], jmh[..., 2]
    h_rad = np.radians(h)
    a = M / 100.0 * np.cos(h_rad)
    b = M / 100.0 * np.sin(h_rad)

    lms_sum = J / 100.0 * 2.0
    lms_c_0 = (lms_sum + a) / 2.0
    lms_c_1 = (lms_sum - a) / 2.0
    lms_c_2 = lms_sum / 2.0 - b

    lms_c = np.stack([lms_c_0, lms_c_1, lms_c_2], axis=-1)
    lms_adapted = np.power(np.maximum(lms_c, 1e-10), 1.0 / 0.42)
    lms_w = np.dot(white_point, HPE_MATRIX.T)
    lms = lms_adapted * lms_w

    rgb = np.dot(lms, HPE_MATRIX_INV.T)
    return np.maximum(rgb, 0.0).astype(np.float32)
