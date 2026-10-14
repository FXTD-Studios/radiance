"""Normal-map preparation for the Viewer's 3D Normal channel. numpy only.

A normal pass reaches the Viewer either signed (components in [-1, 1], what
renderers write to EXR) or already encoded for display (n * 0.5 + 0.5, what a
PNG or a normal-map node produces). Nothing in the file says which, so the
choice is made from the values and reported in `info["input"]`; the Viewer
shows it so a wrong guess is visible, not silent.

The result is always unit-length signed vectors plus the 0..1 display encoding.
Pixels with no usable direction (zero length, NaN or Inf) are reported in
`info["invalid_pixels"]`, decoded as (0, 0, 0) and shown as mid grey.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

# Signed data may overshoot -1 a little from filtering; below this it is signed.
SIGNED_MIN = -0.01
_EPS = 1e-8


def prepare_normal(frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Return (unit_vectors HxWx3 float32 in [-1,1], display HxWx3 float32 in [0,1], info)."""
    a = np.asarray(frame)
    if a.ndim != 3 or a.shape[-1] < 3:
        raise ValueError(f"normal needs an HxWx3 (or 4) image, got shape {a.shape}")
    a = a[..., :3].astype(np.float32, copy=True)

    finite = np.isfinite(a).all(axis=-1)
    a[~finite] = 0.0

    signed = bool(a[finite].size and float(a[finite].min()) < SIGNED_MIN)
    v = a if signed else a * 2.0 - 1.0

    length = np.sqrt((v * v).sum(axis=-1, keepdims=True))
    usable = (length[..., 0] > _EPS) & finite
    unit = np.zeros_like(v)
    np.divide(v, length, out=unit, where=usable[..., None])

    display = np.clip(unit * 0.5 + 0.5, 0.0, 1.0).astype(np.float32)
    info = {
        "input": "signed" if signed else "unsigned_0_1",
        "invalid_pixels": int((~usable).sum()),
        "pixels": int(usable.size),
    }
    return unit.astype(np.float32), display, info
