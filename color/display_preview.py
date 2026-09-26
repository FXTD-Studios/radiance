"""Display-referred previews for the viewers (3.5.0).

The Viewer writes 8-bit PNG previews: it falls back to them and uses them as
the node thumbnail. (The Lite Viewer, retired in 3.5.0, showed them directly.)
Linear sources used to go through ``x / (1 + x)`` with no display encoding,
which is darker than any real view and matches nothing the float path shows.

A preview now goes through the same transform the float path uses by default:

* sRGB-encoded sources (a ComfyUI IMAGE) are already display-referred and are
  written untouched.
* Linear sources go through OpenColorIO's ACES 2.0 studio config, display
  ``sRGB - Display``, view ``ACES 2.0 - SDR 100 nits (Rec.709)``.

The exact OCIO processor is used (no baked LUT: a 65³ shaper LUT was measured
at up to 15 8-bit codes off on saturated highlights). ACES 2.0 costs about
1.2 s per 2K frame on one core; OCIO releases the GIL, so the frame is split
across a thread pool, which scales with the machine's cores.
"""
from __future__ import annotations

import functools
import logging
from typing import Optional

import numpy as np

log = logging.getLogger("radiance.display_preview")

ACES2_CONFIG = "studio-config-v4.0.0_aces-v2.0_ocio-v2.5"
ACES2_DISPLAY = "sRGB - Display"
ACES2_VIEW = "ACES 2.0 - SDR 100 nits (Rec.709)"


@functools.lru_cache(maxsize=8)
def aces2_processor(colorspace: str):
    """Exact OCIO CPU processor (``None`` when OCIO or the config is missing)."""
    try:
        import PyOpenColorIO as ocio  # type: ignore
        cfg = ocio.Config.CreateFromBuiltinConfig(ACES2_CONFIG)
        t = ocio.DisplayViewTransform()
        t.setSrc(colorspace)
        t.setDisplay(ACES2_DISPLAY)
        t.setView(ACES2_VIEW)
        return cfg.getProcessor(t).getDefaultCPUProcessor()
    except Exception as exc:  # noqa: BLE001
        log.debug("[Radiance preview] OCIO ACES 2.0 unavailable for %s: %s", colorspace, exc)
        return None


def _apply_exact(rgb: np.ndarray, cpu) -> np.ndarray:
    import os
    from concurrent.futures import ThreadPoolExecutor
    # A copy, always: applyRGB works in place, and ascontiguousarray returns
    # the caller's own buffer when it is already contiguous float32, so the
    # source frame was being overwritten with display values.
    flat = np.array(rgb.reshape(-1, 3), dtype=np.float32, copy=True)
    workers = max(1, min(16, os.cpu_count() or 1))
    if workers == 1 or flat.shape[0] < 65536:
        cpu.applyRGB(flat)
    else:
        bounds = np.linspace(0, flat.shape[0], workers + 1).astype(int)
        # Views into one contiguous buffer: applyRGB works in place on each.
        chunks = [flat[bounds[i]:bounds[i + 1]] for i in range(workers)]
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(cpu.applyRGB, chunks))
    return flat.reshape(rgb.shape[:-1] + (3,))


def _fallback_view(rgb: np.ndarray) -> np.ndarray:
    """Luminance Reinhard + sRGB OETF, used only when OCIO is unavailable."""
    x = np.maximum(rgb, 0.0)
    y = x @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    x = x * (1.0 / (1.0 + y))[..., None]
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def display_preview(frame: np.ndarray, encoding: str, colorspace: str) -> np.ndarray:
    """Float frame (H, W, C) -> display-referred sRGB-encoded [0, 1], same C.

    Alpha and extra channels are passed through clipped to [0, 1].
    """
    arr = np.asarray(frame, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[..., None]
    rgb = arr[..., :3] if arr.shape[-1] >= 3 else np.repeat(arr[..., :1], 3, axis=-1)
    extra = arr[..., 3:] if arr.shape[-1] > 3 else None
    if encoding == "srgb":
        out = np.clip(rgb, 0.0, 1.0)
    else:
        cpu = aces2_processor(colorspace)
        out = _apply_exact(rgb, cpu) if cpu is not None else _fallback_view(rgb)
        out = np.clip(np.nan_to_num(out, nan=0.0), 0.0, 1.0)
    if arr.shape[-1] == 1:
        out = out[..., :1]
    if extra is not None:
        out = np.concatenate([out, np.clip(extra, 0.0, 1.0)], axis=-1)
    return out.astype(np.float32)


__all__ = ["display_preview", "aces2_processor", "ACES2_VIEW", "ACES2_DISPLAY", "ACES2_CONFIG"]
