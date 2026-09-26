"""
Shared tile-blending helpers.

Why this module exists
----------------------
Radiance grew five independent tiling implementations (latent sampling, two
image upscalers, the VAE, and the temporal video path). Four of them built a
blend ramp starting at exactly 0 and applied it to *every* tile edge, including
edges that sit on the image border. Since the accumulated weight is divided out
at the end, a border pixel covered only by ramp-start weights normalises to
``0 / eps == 0`` -- a hard black line one pixel (or one latent pixel, so eight
image pixels) wide around the whole frame, plus a huge noise-amplification ring
immediately inside it.

Only ``hdr/vae.py`` got it right, by zeroing the overlap on edges that are image
borders so no ramp is applied there. That logic now lives here, once.

The rule, stated plainly: **a tile edge that is an image border must not be
ramped.** There is nothing to blend into.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

try:  # torch is optional for the numpy-only callers
    import torch
except Exception:  # pragma: no cover - exercised only in torch-less installs
    torch = None  # type: ignore


__all__ = [
    "clamp_overlap",
    "edge_overlaps",
    "edge_overlaps_from_coords",
    "blend_weight_2d",
    "blend_weight_2d_np",
]


def clamp_overlap(tile_size: int, overlap: int) -> int:
    """
    Constrain overlap to something the tiler can actually step through.

    An overlap >= tile_size makes the stride ``max(1, tile - overlap)`` collapse
    to 1, which turns a 4K image into ~8 million tile inferences -- indis-
    tinguishable from a hang. Exactly one of the five call sites checked this.
    """
    if tile_size <= 1:
        return 0
    return max(0, min(int(overlap), int(tile_size) // 2))


def edge_overlaps(
    index_y: int,
    index_x: int,
    n_tiles_y: int,
    n_tiles_x: int,
    overlap: int,
) -> Tuple[int, int, int, int]:
    """
    Per-edge overlap for the tile at (index_y, index_x).

    Returns ``(top, bottom, left, right)``, with 0 on any edge that lies on the
    image border. Passing these into :func:`blend_weight_2d` is what prevents
    the black-border artefact.
    """
    top = overlap if index_y > 0 else 0
    bottom = overlap if index_y < n_tiles_y - 1 else 0
    left = overlap if index_x > 0 else 0
    right = overlap if index_x < n_tiles_x - 1 else 0
    return top, bottom, left, right


def edge_overlaps_from_coords(
    y1: int, y2: int, x1: int, x2: int,
    height: int, width: int,
    overlap: int,
) -> Tuple[int, int, int, int]:
    """
    Per-edge overlap derived from a tile's own coordinates.

    Equivalent to :func:`edge_overlaps` but usable where tiles are enumerated by
    position rather than by grid index (which is most of the call sites, since
    they clamp and de-duplicate the last tile). An edge touching the image
    bounds gets 0.
    """
    top = overlap if y1 > 0 else 0
    bottom = overlap if y2 < height else 0
    left = overlap if x1 > 0 else 0
    right = overlap if x2 < width else 0
    # Never ramp more than half the tile from either side, or the two ramps
    # overlap and multiply into a dark seam down the middle of the tile.
    half_h = max(0, (y2 - y1) // 2)
    half_w = max(0, (x2 - x1) // 2)
    return min(top, half_h), min(bottom, half_h), min(left, half_w), min(right, half_w)


def _ramp_np(n: int) -> np.ndarray:
    """Raised-cosine ramp rising from 0 to 1 over n samples (exclusive of 0)."""
    if n <= 0:
        return np.ones(0, dtype=np.float32)
    # Offset by one step so the first sample is > 0. Even with border
    # suppression this keeps interior seams from having a true zero, which
    # would leave the seam pixel determined entirely by one tile.
    t = (np.arange(1, n + 1, dtype=np.float32) / (n + 1)) * (math.pi / 2.0)
    return (np.sin(t) ** 2).astype(np.float32)


def blend_weight_2d_np(
    tile_h: int,
    tile_w: int,
    overlap_top: int = 0,
    overlap_bottom: int = 0,
    overlap_left: int = 0,
    overlap_right: int = 0,
) -> np.ndarray:
    """Numpy form. Returns an (H, W) float32 weight in (0, 1]."""
    weight = np.ones((tile_h, tile_w), dtype=np.float32)

    ot = min(int(overlap_top), tile_h)
    ob = min(int(overlap_bottom), tile_h)
    ol = min(int(overlap_left), tile_w)
    orr = min(int(overlap_right), tile_w)

    if ot > 0:
        weight[:ot, :] *= _ramp_np(ot)[:, None]
    if ob > 0:
        weight[tile_h - ob:, :] *= _ramp_np(ob)[::-1][:, None]
    if ol > 0:
        weight[:, :ol] *= _ramp_np(ol)[None, :]
    if orr > 0:
        weight[:, tile_w - orr:] *= _ramp_np(orr)[::-1][None, :]

    return weight


def blend_weight_2d(
    tile_h: int,
    tile_w: int,
    overlap_top: int = 0,
    overlap_bottom: int = 0,
    overlap_left: int = 0,
    overlap_right: int = 0,
    device=None,
    dtype=None,
):
    """
    Torch form. Returns a (1, 1, H, W) weight in (0, 1].

    Shape is chosen for BCHW accumulators; ``.permute``/``.view`` as needed for
    BHWC. Values are strictly positive, so dividing by the accumulated weight is
    safe without an epsilon fudge that silently produces black.
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is not available")

    dtype = dtype or torch.float32
    weight = torch.ones(tile_h, tile_w, device=device, dtype=dtype)

    def ramp(n: int):
        t = (torch.arange(1, n + 1, device=device, dtype=dtype) / (n + 1)) * (math.pi / 2.0)
        return torch.sin(t) ** 2

    ot = min(int(overlap_top), tile_h)
    ob = min(int(overlap_bottom), tile_h)
    ol = min(int(overlap_left), tile_w)
    orr = min(int(overlap_right), tile_w)

    if ot > 0:
        weight[:ot, :] *= ramp(ot).unsqueeze(1)
    if ob > 0:
        weight[tile_h - ob:, :] *= ramp(ob).flip(0).unsqueeze(1)
    if ol > 0:
        weight[:, :ol] *= ramp(ol).unsqueeze(0)
    if orr > 0:
        weight[:, tile_w - orr:] *= ramp(orr).flip(0).unsqueeze(0)

    return weight.unsqueeze(0).unsqueeze(0)
