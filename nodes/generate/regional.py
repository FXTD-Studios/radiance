"""
◎ Radiance — Phase 4: Radiance Sampler Expansion
nodes_regional.py

Provides:
  • RadianceRegionalPrompt — Spatial region conditioning for Flux / SDXL.

Spatial regions are defined as bounding boxes (x0, y0, x1, y1 in [0..1]) or
connected to an external MASK. Each region gets its own positive conditioning
which is merged into the global conditioning via attention-space weighting.

This node does NOT require model surgery — it builds a standard ComfyUI
conditioning list with per-token region masks that are compatible with
Flux, SDXL, and SD3 via the standard area-conditioned inference path
supported in ComfyUI's built-in KSampler.
"""

import json
import logging
from typing import Tuple, List, Optional

import torch

logger = logging.getLogger("radiance.regional")


# ==============================================================================
# Helper: build area conditioning
# ==============================================================================

def _make_area_cond(
    cond_tensors: list,
    x: float, y: float, w: float, h: float,
    strength: float = 1.0,
) -> list:
    """
    Wrap a standard conditioning list with ComfyUI area-conditioning metadata.

    x, y: top-left corner in [0,1] (fraction of image)
    w, h: width and height in [0,1]
    strength: conditioning weight for this region
    """
    out = []
    for c in cond_tensors:
        # c is (tensor, {dict})
        t, d = c
        new_d = dict(d)
        # ComfyUI's area tuple is only interpreted as fractions when the first
        # element is the literal "percentage" marker: resolve_areas_and_cond_
        # masks_multidim (comfy/samplers.py:768) checks `area[0] == "percentage"`
        # and only then multiplies by the latent dims. ComfyUI's own
        # ConditioningSetArea builds ("percentage", height, width, y, x)
        # (nodes.py:224), and that is the form required here.
        #
        # DEFECT: this used to write a bare (h, w, y, x) of floats in [0,1].
        # Without the marker the floats fell straight through to
        # get_area_and_mult, where input_x.narrow(i + 2, 0.0, 0.5) raises
        # "TypeError: narrow(): argument 'start' must be int, not float".
        # Both RadianceRegionalPrompt and RadianceRegionalGrid were therefore
        # non-functional at stock defaults, on ComfyUI 0.32.0 and 0.36.0 alike.
        new_d["area"] = ("percentage", h, w, y, x)
        new_d["strength"] = strength
        new_d["set_area_to_bounds"] = False
        out.append((t, new_d))
    return out


def _is_region(d: dict) -> bool:
    """A conditioning entry restricted to an area (a region from an earlier node
    in a chain, or ComfyUI's Conditioning Set Area), as opposed to the global
    prompt."""
    return isinstance(d.get("area"), tuple)


def _mask_to_area(mask: torch.Tensor) -> Tuple[float, float, float, float]:
    """
    Compute the tight bounding box of a non-zero mask region.
    Returns (x, y, w, h) in [0,1] fractions.
    mask: (H, W) float tensor, values in [0,1].
    """
    nonzero = mask > 0.1
    if not nonzero.any():
        return (0.0, 0.0, 1.0, 1.0)

    H, W = mask.shape[-2], mask.shape[-1]
    rows = nonzero.any(dim=-1).nonzero(as_tuple=True)[0]
    cols = nonzero.any(dim=-2).nonzero(as_tuple=True)[0]

    y0_frac = float(rows.min()) / H
    y1_frac = float(rows.max() + 1) / H
    x0_frac = float(cols.min()) / W
    x1_frac = float(cols.max() + 1) / W

    return (
        x0_frac,
        y0_frac,
        max(x1_frac - x0_frac, 1.0 / W),
        max(y1_frac - y0_frac, 1.0 / H),
    )


# ==============================================================================
# RadianceRegionalPrompt
# ==============================================================================

class RadianceRegionalPrompt:
    """
    ◎ Radiance Regional Prompt

    Define a spatial region with its own positive conditioning that is merged
    into the global conditioning for area-guided synthesis.

    Supports up to 4 chained regions. Each node adds one region and passes
    the accumulated conditioning list downstream via the CONDITIONING output.

    Usage:
      1. Connect your CLIP-encoded global positive conditioning to `base_cond`.
      2. Encode a region-specific prompt: CLIP Text Encode → `region_cond`.
      3. Define the region: bounding box (x, y, w, h) in [0..1] fractions,
         OR connect a MASK (takes priority over the bbox inputs).
      4. Chain multiple RegionalPrompt nodes together via their CONDITIONING outputs.
      5. Feed the final CONDITIONING into your sampler's positive input.

    Coordinates:
      • x, y = top-left corner as fraction of total image width/height
      • w, h = region width/height as fraction of total image
      • Example: top-left quarter → x=0.0, y=0.0, w=0.5, h=0.5

    Compatibility:
      • Flux, SDXL, SD3, SD1.5 via ComfyUI's standard area-conditioned path.
      • For Flux specifically, area conditioning is approximate (Flux uses
        full-sequence attention without spatial masking). Results are best with
        regions that occupy >25% of the image.
    """

    CATEGORY = "FXTD STUDIOS/Radiance/◎ Generate"
    DESCRIPTION = "Apply region-specific text prompts with spatial masks."
    FUNCTION = "apply"
    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "region_info")

    # ip_image / ip_weight are accepted for graph compatibility only and are
    # ignored with a warning. See the note in apply(): IP-Adapter is a model
    # patch, not a conditioning key, so no CONDITIONING node can enable it.

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_cond": ("CONDITIONING", {
                    "tooltip": (
                        "Global positive conditioning, passed through with its strength set to global_strength. "
                        "When chaining, regions from earlier nodes keep their own strength and area."
                    ),
                }),
                "region_cond": ("CONDITIONING", {
                    "tooltip": "Encoded prompt for this region, restricted to the x/y/w/h box (or the mask's bounding box) via ComfyUI area conditioning.",
                }),
                "region_label": ("STRING", {
                    "default": "region_1",
                    "tooltip": "Human-readable label for this region (used in JSON output).",
                }),
                "x": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Left edge of region as fraction of image width.",
                }),
                "y": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Top edge of region as fraction of image height.",
                }),
                "w": ("FLOAT", {
                    "default": 0.5, "min": 0.01, "max": 1.0, "step": 0.01,
                    "tooltip": "Width of region as fraction of image width.",
                }),
                "h": ("FLOAT", {
                    "default": 0.5, "min": 0.01, "max": 1.0, "step": 0.01,
                    "tooltip": "Height of region as fraction of image height.",
                }),
                "region_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Conditioning weight for this region vs global.",
                }),
                "global_strength": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Weight of the global base conditioning passed through.",
                }),
                "merge_mode": (["Additive", "Replace"], {
                    "default": "Additive",
                    "tooltip": (
                        "Additive: region added on top of global (default, safe). "
                        "Replace: region replaces global in its area; the global still "
                        "applies outside it (masked)."
                    ),
                }),
            },
            "optional": {
                "mask": ("MASK", {
                    "tooltip": "Optional. When connected, overrides x/y/w/h with the mask's bounding box.",
                }),
                "ip_image": ("IMAGE", {
                    "tooltip": (
                        "IGNORED. Kept so existing graphs still load. IP-Adapter is a "
                        "model-side attention patch, not a conditioning key, so it "
                        "cannot be enabled from this node. Apply an IPAdapter "
                        "loader/apply node to the MODEL before the sampler instead. "
                        "Connecting this logs a warning and changes nothing."
                    ),
                }),
                "ip_weight": ("FLOAT", {
                    "default": 0.6,
                    "min": 0.0,
                    "max": 1.5,
                    "step": 0.05,
                    "tooltip": (
                        "IGNORED. See ip_image. Set the weight on the IPAdapter node "
                        "that patches the MODEL."
                    ),
                }),
            },
        }

    def apply(
        self,
        base_cond: list,
        region_cond: list,
        region_label: str = "region_1",
        x: float = 0.0,
        y: float = 0.0,
        w: float = 0.5,
        h: float = 0.5,
        region_strength: float = 1.0,
        global_strength: float = 0.5,
        merge_mode: str = "Additive",
        mask: torch.Tensor = None,
        ip_image: torch.Tensor = None,
        ip_weight: float = 0.6,
    ) -> Tuple[list, str]:

        # If a mask is provided, derive bbox from it
        if mask is not None:
            m = mask
            if m.dim() == 3:
                m = m[0]  # take first mask in batch
            x, y, w, h = _mask_to_area(m)
            logger.debug(f"[RegionalPrompt] Mask → bbox: x={x:.3f} y={y:.3f} w={w:.3f} h={h:.3f}")

        # Clamp values
        x = max(0.0, min(1.0 - 0.01, x))
        y = max(0.0, min(1.0 - 0.01, y))
        w = max(0.01, min(1.0 - x, w))
        h = max(0.01, min(1.0 - y, h))

        # Build conditionings
        if merge_mode == "Replace":
            # Region replaces the global in its area only. The global is kept
            # everywhere else through a cond mask with the region cut out; it
            # used to be dropped from the whole frame, leaving everything
            # outside the region with no positive conditioning at all.
            import torch as _torch
            res = 256
            outside = _torch.ones((1, res, res), dtype=_torch.float32)
            y0, y1 = int(round(y * res)), int(round((y + h) * res))
            x0, x1 = int(round(x * res)), int(round((x + w) * res))
            outside[:, y0:y1, x0:x1] = 0.0
            # Regions from earlier nodes in a chain keep their own strength
            # and area; only the global prompt takes global_strength and the
            # cut-out. Every entry used to be reset, so each chained node
            # overwrote the earlier regions' strength. A global already cut by
            # an earlier node keeps that cut as well as this one.
            global_out = []
            for c in base_cond:
                t, d = c
                if _is_region(d):
                    global_out.append((t, d))
                    continue
                nd = dict(d)
                nd["strength"] = global_strength
                prev = d.get("mask")
                if isinstance(prev, _torch.Tensor) and prev.dim() == 3:
                    prev = _torch.nn.functional.interpolate(
                        prev.float().unsqueeze(1), size=(res, res), mode="nearest").squeeze(1)
                    nd["mask"] = prev * outside
                else:
                    nd["mask"] = outside
                nd["mask_strength"] = 1.0
                nd["set_area_to_bounds"] = False
                global_out.append((t, nd))
            result = global_out + _make_area_cond(region_cond, x, y, w, h, region_strength)
        else:
            # Additive: keep global (optionally weight-adjusted) + add region on top
            global_out = []
            for c in base_cond:
                t, d = c
                if _is_region(d):          # an earlier region keeps its strength
                    global_out.append((t, d))
                    continue
                nd = dict(d)
                nd["strength"] = global_strength
                global_out.append((t, nd))
            region_out = _make_area_cond(region_cond, x, y, w, h, region_strength)
            result = global_out + region_out

        # ── IP-Adapter image conditioning for this region ──────────────────────
        # DEFECT: this used to crop/resize ip_image and write it into the
        # conditioning dict under 'cross_attn_controlnet' as
        # {"image": ..., "weight": ..., "type": "ip_adapter"}, then report
        # "ip_adapter": {"enabled": true} and log ip=True.
        #
        # 'cross_attn_controlnet' is a real ComfyUI key, but it is not an
        # IP-Adapter hook and it has never been one. BaseModel.extra_conds
        # (comfy/model_base.py, identical on 0.32.0 and 0.36.0) reads it and
        # wraps it in comfy.conds.CONDCrossAttn, i.e. it expects a TEXT
        # EMBEDDING tensor for a ControlNet's own cross-attention. Its only
        # writer in ComfyUI is ControlNetInpaintingAliMamaApply-style
        # conditioning (comfy_extras/nodes_cond.py), which stores
        # clip.encode_from_tokens output there. Handing CONDCrossAttn a dict
        # is at best ignored and at worst breaks conditioning batching.
        #
        # IP-Adapter is a MODEL patch (attention adapters injected into the
        # UNet), not a conditioning key, so there is no shape of dict this
        # node could write that would make it run. Rather than fake it, this
        # now refuses the input loudly and leaves the conditioning untouched.
        ip_applied = False
        ip_reason = None
        if ip_image is not None:
            ip_reason = (
                "IP-Adapter cannot be driven from a CONDITIONING dict. It is a "
                "model-side attention patch, so it has to be applied to the MODEL "
                "with an IPAdapter loader/apply node before the sampler. "
                "ip_image and ip_weight are ignored here."
            )
            logger.warning(
                "[RegionalPrompt] '%s': %s", region_label, ip_reason,
            )

        region_info = json.dumps({
            "node": "RadianceRegionalPrompt",
            "label": region_label,
            "bbox": {"x": round(x, 4), "y": round(y, 4), "w": round(w, 4), "h": round(h, 4)},
            "mask_provided": mask is not None,
            "ip_adapter": {
                "enabled": ip_applied,
                "weight": ip_weight if ip_applied else None,
                "ignored_reason": ip_reason,
            },
            "region_strength": region_strength,
            "global_strength": global_strength,
            "merge_mode": merge_mode,
            "num_regions_in_output": len(result),
        }, indent=2)

        logger.info(f"[RegionalPrompt] '{region_label}' ({x:.2f},{y:.2f}) {w:.2f}×{h:.2f} strength={region_strength} ip={ip_applied}")
        return (result, region_info)


# ==============================================================================
# RadianceRegionalGrid
# ==============================================================================

class RadianceRegionalGrid:
    """
    ◎ Radiance Regional Grid

    Convenience node that divides the image into a regular grid of cells
    and assigns a separate text conditioning to each cell.

    Input:
      • base_cond      — global positive conditioning (background / fallback)
      • clip            — a CLIP model for encoding each region's prompt text
      • grid_prompts    — JSON array of per-cell prompts in row-major order:
                          ["top-left prompt", "top-right prompt", ...]
      • columns         — number of grid columns
      • rows            — number of grid rows
      • cell_strength   — conditioning weight for each cell region
      • global_strength — weight of the base conditioning passed through

    The number of prompts can be less than columns × rows; remaining cells
    inherit the base conditioning.

    Output: merged CONDITIONING list compatible with ComfyUI's KSampler.
    """

    CATEGORY = "FXTD STUDIOS/Radiance/◎ Generate"
    DESCRIPTION = "Divide the canvas into a grid of independently prompted regions."
    FUNCTION = "apply_grid"
    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "grid_info")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_cond": ("CONDITIONING", {
                    "tooltip": "Global positive conditioning for the whole frame, passed through with its strength set to global_strength. Cells with no prompt get only this.",
                }),
                "clip": ("CLIP", {
                    "tooltip": "Text encoder used to encode each cell's prompt from grid_prompts.",
                }),
                "grid_prompts": ("STRING", {
                    "default": '["subject in left area", "background on right"]',
                    "multiline": True,
                    "tooltip": "JSON array of prompts, one per grid cell, row-major order.",
                }),
                "columns": ("INT", {"default": 2, "min": 1, "max": 8, "step": 1, "tooltip": "Number of columns in the regional prompt grid."}),
                "rows": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1, "tooltip": "Number of rows in the regional prompt grid."}),
                "cell_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Conditioning strength for each individual region cell. Higher values make the model follow regional prompts more closely."
                }),
                "global_strength": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Conditioning strength for the global (full-image) prompt. Blended with cell conditioning at each step."
                }),
            },
        }

    # Calls clip.encode_from_tokens; CLIP parameters require grad after .eval().

    @torch.no_grad()

    def apply_grid(self, base_cond: list, clip, grid_prompts: str,
                   columns: int = 2, rows: int = 1,
                   cell_strength: float = 1.0,
                   global_strength: float = 0.3) -> Tuple[list, str]:

        # Parse grid prompts
        try:
            prompts = json.loads(grid_prompts)
            if isinstance(prompts, str):
                prompts = [prompts]
        except Exception as e:
            logger.warning(f"[RegionalGrid] Parse error: {e}")
            prompts = []

        total_cells = columns * rows
        cell_w = 1.0 / columns
        cell_h = 1.0 / rows

        # Apply global weight to base_cond
        result = []
        for c in base_cond:
            t, d = c
            if _is_region(d):              # an earlier region keeps its strength
                result.append((t, d))
                continue
            nd = dict(d)
            nd["strength"] = global_strength
            result.append((t, nd))

        cell_info = []
        for cell_idx in range(total_cells):
            row_i = cell_idx // columns
            col_i = cell_idx % columns
            cx = col_i * cell_w
            cy = row_i * cell_h

            prompt = prompts[cell_idx] if cell_idx < len(prompts) else None
            if not prompt or not str(prompt).strip():
                continue

            # Encode the cell prompt
            try:
                tokens = clip.tokenize(str(prompt))
                cond_tensor, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
                cell_cond = [(cond_tensor, {"pooled_output": pooled})]
                region_area_cond = _make_area_cond(
                    cell_cond, cx, cy, cell_w, cell_h, cell_strength
                )
                result.extend(region_area_cond)
                cell_info.append({
                    "cell": cell_idx, "row": row_i, "col": col_i,
                    "bbox": {"x": round(cx, 4), "y": round(cy, 4),
                             "w": round(cell_w, 4), "h": round(cell_h, 4)},
                    "prompt": str(prompt)[:80],
                })
                logger.debug(f"[RegionalGrid] Cell {cell_idx} ({row_i},{col_i}) — '{prompt[:40]}'")
            except Exception as e:
                logger.warning(f"[RegionalGrid] Cell {cell_idx} encode failed: {e}")

        grid_info = json.dumps({
            "node": "RadianceRegionalGrid",
            "columns": columns, "rows": rows,
            "total_cells": total_cells,
            "populated_cells": len(cell_info),
            "cell_strength": cell_strength,
            "global_strength": global_strength,
            "cells": cell_info,
        }, indent=2)

        logger.info(f"[RegionalGrid] {columns}×{rows} = {total_cells} cells, {len(cell_info)} populated")
        return (result, grid_info)


# ==============================================================================
# Node registration
# ==============================================================================

NODE_CLASS_MAPPINGS = {
    "RadianceRegionalPrompt": RadianceRegionalPrompt,
    "RadianceRegionalGrid":   RadianceRegionalGrid,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceRegionalPrompt": "◎ Radiance Regional Prompt",
    "RadianceRegionalGrid":   "◎ Radiance Regional Grid",
}
