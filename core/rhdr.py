"""
The .rhdr float sidecar format, encoded in one place.

Why this module exists
----------------------
The Viewer loads .rhdr files as its primary display source. They are read by
one parser, ``_parseRHDR`` in ``js/radiance_viewer.js``, but were written by
hand in four places (the Viewer's fp16 and fp32 frame branches, its zdepth
sidecar, and the HDR VAE export), and those copies drifted apart. This module
is the single encoder they share. Change it and ``_parseRHDR`` together.

Layout (little-endian)
----------------------
    [0:4]    magic     b"RHDR"
    [4:6]    width     uint16
    [6:8]    height    uint16
    [8:10]   channels  uint16
    [10:12]  flags     uint16, 0 = fp16 payload, 1 = fp32 payload
    [12:]    zlib stream of the H x W x C samples in C order

The parser rejects a file whose decompressed size is not
width * height * channels * bytes-per-sample.

fp16 samples are always clamped to +-65504 first: an unclamped cast turns a
bright specular into inf, which poisons tonemapping and scopes downstream.
``level`` is the zlib level each writer chose; the Viewer stores (0).

numpy only: no torch, so it imports in the light test lane.
"""
from __future__ import annotations

import struct
import zlib
from typing import Tuple

import numpy as np

__all__ = [
    "MAGIC",
    "HEADER",
    "FLAG_FP16",
    "FLAG_FP32",
    "MAX_DIMENSION",
    "FP16_MAX",
    "encode",
    "write",
    "decode",
]

MAGIC = b"RHDR"
HEADER = struct.Struct("<4sHHHH")
FLAG_FP16 = 0
FLAG_FP32 = 1
MAX_DIMENSION = 65535  # uint16 header fields
FP16_MAX = 65504.0


def _shape(pixels: np.ndarray) -> Tuple[int, int, int]:
    if pixels.ndim == 2:
        h, w = pixels.shape
        c = 1
    elif pixels.ndim == 3:
        h, w, c = pixels.shape
    else:
        raise ValueError(f"RHDR needs an (H, W) or (H, W, C) array, got shape {pixels.shape}")
    if max(w, h, c) > MAX_DIMENSION:
        raise ValueError(
            f"RHDR header fields are uint16: {w}x{h}x{c} exceeds {MAX_DIMENSION}"
        )
    return w, h, c


def _encode_parts(pixels, fp32: bool, level: int) -> Tuple[bytes, bytes]:
    """The header and the zlib stream, kept apart so ``write`` never joins them.

    Joining would copy the whole compressed frame once more; at level 0 that is
    the raw size, about 130 MB for a 4K fp32 RGBA frame.
    """
    pixels = np.asarray(pixels)
    w, h, c = _shape(pixels)
    if fp32:
        samples = pixels.astype(np.float32)
        flags = FLAG_FP32
    else:
        samples = np.clip(pixels, -FP16_MAX, FP16_MAX).astype(np.float16)
        flags = FLAG_FP16
    return HEADER.pack(MAGIC, w, h, c, flags), zlib.compress(samples.tobytes(), level=level)


def encode(pixels, *, fp32: bool = False, level: int = 0) -> bytes:
    """The complete .rhdr file for an (H, W) or (H, W, C) float array.

    fp32 writes the samples as float32 (flags 1). Otherwise they are written as
    float16 (flags 0), clamped to +-65504 first.
    ``level`` is the zlib level; 0 stores, which is what the Viewer writes
    because compressing cost far more time than it saved.

    Raises ValueError for any other rank or a dimension above 65535.
    """
    header, payload = _encode_parts(pixels, fp32, level)
    return header + payload


def write(path: str, pixels, *, fp32: bool = False, level: int = 0) -> int:
    """Write ``encode(pixels, ...)`` to ``path`` and return the file size in bytes.

    Encoding happens before the file is opened, so a ValueError leaves no file.
    The write itself is not atomic, as it was not in any of the writers this
    replaces.
    """
    header, payload = _encode_parts(pixels, fp32, level)
    with open(path, "wb") as f:
        f.write(header)
        f.write(payload)
    return len(header) + len(payload)


def decode(data: bytes) -> Tuple[int, np.ndarray]:
    """``(flags, samples)`` from .rhdr bytes; samples are (H, W, C) float16 or float32.

    Python has no consumer of .rhdr files: this exists to check what the
    writers produce. Raises ValueError where ``_parseRHDR`` would reject the file.
    """
    if len(data) < HEADER.size:
        raise ValueError(f"RHDR file is {len(data)} bytes, shorter than its {HEADER.size}-byte header")
    magic, w, h, c, flags = HEADER.unpack_from(data)
    if magic != MAGIC:
        raise ValueError(f"not an RHDR file (magic {magic!r})")
    dtype = np.float32 if flags & FLAG_FP32 else np.float16
    raw = zlib.decompress(data[HEADER.size:])
    expected = w * h * c * np.dtype(dtype).itemsize
    if len(raw) != expected:
        raise ValueError(f"RHDR payload is {len(raw)} bytes, the header says {expected}")
    return flags, np.frombuffer(raw, dtype=dtype).reshape(h, w, c)
