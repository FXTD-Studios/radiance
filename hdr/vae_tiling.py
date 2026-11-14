"""TileEngine: tile and temporal-chunk sizing and cosine blend weights for
the HDR VAE nodes.

Moved out of hdr/vae.py in 4.0, unchanged, as the first slice of splitting
that file. hdr/vae.py re-exports it, so ``from radiance.hdr.vae import
TileEngine`` keeps working.
"""
import logging
import math
from typing import Tuple

import torch
import torch.nn.functional as F

import comfy.model_management

logger = logging.getLogger("radiance")


class TileEngine:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    """
    Production-grade tiling engine with cosine blend weights.
    Handles arbitrary image sizes, pad-to-multiple, and VRAM-aware sizing.
    """

    @staticmethod
    def get_optimal_tile_size(
        image_h: int,
        image_w: int,
        min_tile: int = 512,
        max_tile: int = 1536,
        vram_budget_gb: float = None,
    ) -> int:
        """
        Determine optimal tile size based on image dimensions and available VRAM.

        Rules:
        - If image fits in a single tile (≤max_tile), use full image
        - Otherwise, pick largest tile that fits in VRAM budget
        - Always returns multiple of 8 (VAE requirement)
        """
        # If image fits in one tile, no tiling needed
        if image_h <= max_tile and image_w <= max_tile:
            # Round up to multiple of 8
            tile = max(image_h, image_w)
            tile = ((tile + 7) // 8) * 8
            return min(tile, max_tile)

        # Query available VRAM
        if vram_budget_gb is None:
            try:
                device = comfy.model_management.get_torch_device()
                if device.type == "cuda":
                    free_mem, total_mem = torch.cuda.mem_get_info(device)
                    # Use 60% of free VRAM for safety (VAE + intermediate buffers)
                    vram_budget_gb = (free_mem * 0.6) / (1024**3)
                else:
                    vram_budget_gb = 4.0  # Conservative default for CPU
            except Exception:
                vram_budget_gb = 4.0

        # Estimate VRAM per tile:
        # Encode: tile_pixels(fp32) + tile_latent(fp32) + VAE weights + intermediates
        # Rough formula: ~20 bytes per pixel for VAE encode/decode
        bytes_per_pixel = 20
        max_pixels = int(vram_budget_gb * (1024**3) / bytes_per_pixel)
        max_side = int(math.sqrt(max_pixels))

        # Clamp to range and round down to multiple of 8
        tile = max(min_tile, min(max_side, max_tile))
        tile = (tile // 8) * 8

        logger.info(
            f"[Radiance 4K] VRAM budget: {vram_budget_gb:.1f}GB -> "
            f"tile size: {tile}px (image: {image_w}x{image_h})"
        )
        return tile

    @staticmethod
    def get_optimal_temporal_size(
        tile_size_px: int,
        temporal_compression: int = 8,
        min_frames: int = 2,
        max_frames: int = 64,
        vram_budget_gb: float = None,
    ) -> int:
        """
        Determine optimal temporal chunk size (in LATENT frames) for 5D video
        VAE decode, given the spatial tile size already chosen. Spatial and
        temporal chunking share one VRAM budget instead of being sized
        independently; a bigger spatial tile leaves less room per frame.

        ALBABIT-FIX (VAE tiling integration): bytes_per_pixel_frame is
        back-solved from the LTX-2.5 crash data (1536px/~16.8GB: 8 frames
        clean, 16 hard-crashed) to land exactly on 8, not just "under 16".
        Rough proxy, not a precise memory model; see
        project_radiance_vae_tiling_seams memory for the investigation.
        """
        if vram_budget_gb is None:
            try:
                device = comfy.model_management.get_torch_device()
                if device.type == "cuda":
                    free_mem, total_mem = torch.cuda.mem_get_info(device)
                    vram_budget_gb = (free_mem * 0.6) / (1024**3)
                else:
                    vram_budget_gb = 4.0
            except Exception:
                vram_budget_gb = 4.0

        bytes_per_pixel_frame = 110
        max_pixel_frames = int(
            vram_budget_gb * (1024**3) / (bytes_per_pixel_frame * tile_size_px * tile_size_px)
        )
        compression = max(1, temporal_compression or 8)
        budget_frames = max_pixel_frames // compression
        frames = max(min_frames, min(budget_frames, max_frames))

        if budget_frames < min_frames:
            # BUDGET-FLOOR FIX: the max(min_frames, ...) floor means a budget
            # that fits ZERO frames still returns min_frames, and this used to
            # be logged as an ordinary budget-derived decision. That is an OOM
            # about to happen, reported as an "optimal" size. Say so, so the
            # operator lowers tile_size or frees VRAM instead of reading the
            # log as confirmation that the chunk size was chosen for them.
            logger.warning(
                f"[Radiance Temporal] VRAM budget {vram_budget_gb:.1f}GB at "
                f"tile={tile_size_px}px fits {budget_frames} latent frames, "
                f"below the {min_frames}-frame floor. Returning {frames} anyway "
                f"and this decode is expected to run out of memory. Lower "
                f"tile_size, lower temporal_size, or free VRAM."
            )
            return frames

        logger.info(
            f"[Radiance Temporal] VRAM budget: {vram_budget_gb:.1f}GB, tile={tile_size_px}px -> "
            f"temporal size: {frames} latent frames"
        )
        return frames

    @staticmethod
    def compute_tiles(total: int, tile_size: int, overlap: int) -> list:
        """
        Compute tile start positions along one axis.
        Returns list of (start, end) tuples.
        Ensures full coverage with consistent overlap.

        ``overlap`` must leave a positive stride.  It used not to be checked:
        with ``overlap >= tile_size`` the stride was zero or negative, ``pos``
        never advanced, and the loop appended tiles until the process ran out of
        memory, with ComfyUI hanging on no error and nothing in the log.  That was
        reachable from shipped widget defaults (temporal_size 2, temporal
        overlap 2), so this raises rather than silently clamping: a caller that
        gets here with a bad overlap has a bug the operator needs to see.
        """
        if tile_size <= 0:
            raise ValueError(f"tile_size must be positive, got {tile_size}")
        if overlap < 0:
            raise ValueError(f"overlap must not be negative, got {overlap}")
        if overlap >= tile_size:
            raise ValueError(
                f"overlap ({overlap}) must be smaller than tile_size "
                f"({tile_size}); an overlap at or above the tile size leaves no "
                f"stride and cannot tile."
            )

        if total <= tile_size:
            return [(0, total)]

        stride = tile_size - overlap
        positions = []
        pos = 0
        while pos < total:
            end = min(pos + tile_size, total)
            start = max(0, end - tile_size)
            positions.append((start, end))
            if end >= total:
                break
            pos += stride

        return positions

    @staticmethod
    def make_cosine_blend_weight_2d(
        tile_h: int,
        tile_w: int,
        overlap_top: int,
        overlap_bottom: int,
        overlap_left: int,
        overlap_right: int,
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        Create 2D cosine blend weight mask for a tile.

        Cosine blending eliminates visible seams — the weight transitions
        smoothly from 0→1 at borders, following a raised cosine curve.
        This is the standard approach used in DJV, Nuke, and Flame for
        tiled compositing.

        Returns: (1, tile_h, tile_w, 1) weight tensor
        """
        weight = torch.ones(tile_h, tile_w, device=device)

        # Cosine ramp: 0 → 1 over `n` pixels
        def cosine_ramp(n):
            if n <= 0:
                return torch.ones(0, device=device)
            t = torch.linspace(0, math.pi / 2, n, device=device)
            return torch.sin(t) ** 2  # Raised cosine (power-of-sine window)

        # Top edge
        if overlap_top > 0:
            ramp = cosine_ramp(overlap_top)
            weight[:overlap_top, :] *= ramp.unsqueeze(1)

        # Bottom edge
        if overlap_bottom > 0:
            ramp = cosine_ramp(overlap_bottom).flip(0)
            weight[-overlap_bottom:, :] *= ramp.unsqueeze(1)

        # Left edge
        if overlap_left > 0:
            ramp = cosine_ramp(overlap_left)
            weight[:, :overlap_left] *= ramp.unsqueeze(0)

        # Right edge
        if overlap_right > 0:
            ramp = cosine_ramp(overlap_right).flip(0)
            weight[:, -overlap_right:] *= ramp.unsqueeze(0)

        return weight.unsqueeze(0).unsqueeze(-1)  # (1, H, W, 1)

    @staticmethod
    def pad_to_multiple(
        tensor: torch.Tensor, multiple: int = 8, mode: str = "reflect"
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Pad image tensor to nearest multiple of `multiple`.
        Input: (B, H, W, C) — ComfyUI IMAGE format
        Returns: padded tensor, (pad_h, pad_w) for later cropping
        """
        b, h, w, c = tensor.shape
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple

        if pad_h == 0 and pad_w == 0:
            return tensor, (0, 0)

        # F.pad expects (N, C, H, W) and padding as (left, right, top, bottom)
        t = tensor.permute(0, 3, 1, 2)  # → (B, C, H, W)
        t = F.pad(t, (0, pad_w, 0, pad_h), mode=mode)
        return t.permute(0, 2, 3, 1), (pad_h, pad_w)  # → (B, H+padH, W+padW, C)
