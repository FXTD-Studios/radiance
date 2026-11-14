"""Normal-map preparation for the Viewer's 3D Normal channel. numpy only.

A normal pass reaches the Viewer either signed (components in [-1, 1], what
renderers write to EXR) or already encoded for display (n * 0.5 + 0.5, what a
PNG or a normal-map node produces). Nothing in the file says which, so the
choice is made from the values and reported in `info["input"]`; the Viewer
shows it so a wrong guess is visible, not silent.

Detection runs on the first frame and the answer is used for the whole
sequence, so the reading cannot change from frame to frame.

The result is always unit-length signed vectors plus the 0..1 display encoding.
Pixels with no usable direction (all zero, zero length, NaN or Inf) are reported in
`info["invalid_pixels"]`, decoded as (0, 0, 0) and shown as mid grey.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

# Signed data may overshoot -1 a little from filtering; below this it is signed.
SIGNED_MIN = -0.01
_EPS = 1e-8


def _rgb(frame: np.ndarray) -> np.ndarray:
    a = np.asarray(frame)
    if a.ndim != 3 or a.shape[-1] < 3:
        raise ValueError(f"normal needs an HxWx3 (or 4) image, got shape {a.shape}")
    return a[..., :3].astype(np.float32, copy=True)


def detect_encoding(frame: np.ndarray) -> str:
    """'signed' or 'unsigned_0_1' for a normal frame.

    Any clearly negative component means signed. Otherwise both readings are
    possible (a signed pass of up-facing surfaces has no negatives), so the one
    whose vectors come out closer to unit length wins: real normals are unit
    vectors in the encoding they were written in.
    """
    a = _rgb(frame)
    ok = np.isfinite(a).all(axis=-1) & (np.abs(np.nan_to_num(a)).sum(axis=-1) > 0.0)
    v = a[ok]
    if not v.size:
        return "unsigned_0_1"
    if float(v.min()) < SIGNED_MIN:
        return "signed"
    err_signed = float(np.median(np.abs(np.linalg.norm(v, axis=-1) - 1.0)))
    err_encoded = float(np.median(np.abs(np.linalg.norm(v * 2.0 - 1.0, axis=-1) - 1.0)))
    return "signed" if err_signed < err_encoded else "unsigned_0_1"


def prepare_normal(frame: np.ndarray, encoding: Optional[str] = None
                   ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Return (unit_vectors HxWx3 float32 in [-1,1], display HxWx3 float32 in [0,1], info).

    `encoding` ('signed' or 'unsigned_0_1') skips detection; pass the value
    detected on the first frame so a sequence is read one way throughout.
    """
    a = _rgb(frame)
    if encoding not in (None, "signed", "unsigned_0_1"):
        raise ValueError(f"unknown normal encoding {encoding!r}")

    finite = np.isfinite(a).all(axis=-1)
    a[~finite] = 0.0

    signed = (encoding or detect_encoding(a)) == "signed"
    v = a if signed else a * 2.0 - 1.0

    length = np.sqrt((v * v).sum(axis=-1, keepdims=True))
    # An all-zero pixel is background (no surface) in either encoding; it is not
    # the direction (-1, -1, -1) that an encoded zero would decode to.
    usable = (length[..., 0] > _EPS) & finite & (np.abs(a).sum(axis=-1) > 0.0)
    unit = np.zeros_like(v)
    np.divide(v, length, out=unit, where=usable[..., None])

    display = np.clip(unit * 0.5 + 0.5, 0.0, 1.0).astype(np.float32)
    info = {
        "input": "signed" if signed else "unsigned_0_1",
        "invalid_pixels": int((~usable).sum()),
        "pixels": int(usable.size),
    }
    return unit.astype(np.float32), display, info
