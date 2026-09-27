"""Settle reconstruction grain in flat SDR highlights.

Port of RUDRA 0.9.0-beta.2's luminance-only highlight grain correction.
"""
from __future__ import annotations

import cv2
import numpy as np

_LUMA = np.array([0.2627, 0.6780, 0.0593], dtype=np.float64)


def settle_highlight_grain(
    hdr_nits: np.ndarray,
    sdr_srgb: np.ndarray,
    knee: float = 0.9,
    softness: float = 0.04,
    sigma: float = 2.0,
) -> np.ndarray:
    """Reduce grain in reconstructed, flat highlights while preserving hue."""
    if hdr_nits.shape != sdr_srgb.shape:
        raise ValueError("HDR and SDR images must have the same shape")
    if not 0.0 < knee < 1.0 or softness <= 0.0 or sigma <= 0.0:
        raise ValueError("invalid highlight grain parameters")

    code = sdr_srgb.max(axis=-1).astype(np.float64)
    ramp = np.clip((code - (knee - softness)) / (2.0 * softness), 0.0, 1.0)
    ramp = ramp * ramp * (3.0 - 2.0 * ramp)
    if not np.any(ramp > 0.0):
        return hdr_nits

    size = 7
    flat = np.zeros(code.shape, dtype=np.float64)
    for signal in (sdr_srgb.max(axis=-1), sdr_srgb @ _LUMA):
        signal = signal.astype(np.float64)
        mean = cv2.boxFilter(signal, -1, (size, size), normalize=True,
                             borderType=cv2.BORDER_REFLECT)
        second = cv2.boxFilter(signal * signal, -1, (size, size), normalize=True,
                               borderType=cv2.BORDER_REFLECT)
        spread = np.sqrt(np.maximum(second - mean * mean, 0.0))
        flat = np.maximum(flat, np.clip((3.0 - spread * 255.0) / 2.0, 0.0, 1.0))
    flat = flat * flat * (3.0 - 2.0 * flat)
    weight = ramp * flat
    if not np.any(weight > 0.0):
        return hdr_nits

    luma = np.maximum(hdr_nits @ _LUMA, 0.0)
    blur = lambda x: cv2.GaussianBlur(x, (0, 0), sigmaX=sigma,
                                      borderType=cv2.BORDER_REFLECT)
    settled = blur(flat * luma) / np.maximum(blur(flat), 1e-6)
    target = luma + weight * (settled - luma)
    gain = np.where(luma > 1e-6, target / np.maximum(luma, 1e-6), 1.0)
    return np.nan_to_num(hdr_nits * gain[..., None], nan=0.0, posinf=0.0, neginf=0.0)
