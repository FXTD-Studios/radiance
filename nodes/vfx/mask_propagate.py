"""Mask propagation along optical flow.

3.5.0: this module also held the Roto node (`RadianceVectorMaskDraw`), which
was removed; draw masks with ComfyUI's mask editor or a SAM / matting node and
propagate them here.
"""
import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger("radiance.vfx.mask_propagate")


class RadianceVideoMaskPropagator:
    """
    ◎ Radiance Video Mask Propagator
    
    A GPU-native mask propagation system. Uses Dense Optical Flow vectors
    to warp and propagate roto masks dynamically across the video sequence timeline.
    """
    
    DESCRIPTION = "Fill empty frames of a mask sequence by warping neighbouring keyframe masks along optical flow. Frames that already contain a mask are kept as keyframes."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK", {"tooltip": "Mask sequence, one per frame. Frames with any mask content are keyframes; empty frames are filled by propagation."}),
                "flow_vectors": ("IMAGE", {"tooltip": "32-bit flow vectors from Radiance Optical Flow."}),
                "propagation_mode": (["Forward", "Backward", "Bidirectional"], {"default": "Bidirectional", "tooltip": "Forward carries masks from earlier frames, Backward from later frames. Bidirectional runs both and keeps the union (maximum) on filled frames."}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("propagated_masks",)
    FUNCTION = "propagate"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ VFX/Masking"

    def propagate(self, masks: torch.Tensor, flow_vectors: torch.Tensor, propagation_mode: str):
        # masks shape [B, H, W]
        # flow_vectors shape [B, H, W, 3] where R=U (dx), G=V (dy)
        B, H, W = masks.shape
        device = masks.device
        
        if B <= 1:
            return (masks,)
            
        prop_masks = masks.clone()
        
        # Grid coordinates mapping [-1, 1] range for grid_sample
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing="ij"
        )
        
        # Radiance Optical Flow convention (measured on a moving plate):
        # flow_vectors[i] lives on frame i and points to where each pixel was
        # in frame i-1, i.e. frame_i(p) ~ frame_{i-1}(p + flow_i(p));
        # flow_vectors[0] is zero. The old code read flow_vectors[i-1] with
        # the opposite sign and a half-pixel-off normalisation, so a mask
        # drifted the wrong way (IoU 0.6 after one frame, 0.0 after three).
        sx = 2.0 / max(W - 1, 1)
        sy = 2.0 / max(H - 1, 1)

        def _warp(src, gx, gy):
            g = torch.stack([gx, gy], dim=-1).unsqueeze(0)
            return F.grid_sample(src.unsqueeze(0).unsqueeze(0), g, mode="bilinear",
                                 padding_mode="zeros", align_corners=True).squeeze(0).squeeze(0)

        # Forward: frame i from frame i-1, sampled at p + flow_i(p).
        if propagation_mode in ("Forward", "Bidirectional"):
            for i in range(1, B):
                if torch.sum(masks[i]) < 1.0:
                    flow = flow_vectors[i].to(device)
                    prop_masks[i] = _warp(prop_masks[i - 1],
                                          grid_x + flow[..., 0] * sx,
                                          grid_y + flow[..., 1] * sy)

        # Backward: frame i from frame i+1, sampled at p - flow_{i+1}(p)
        # (the forward field at i+1 stands in for the inverse field).
        if propagation_mode in ("Backward", "Bidirectional"):
            for i in reversed(range(B - 1)):
                if torch.sum(masks[i]) < 1.0:
                    flow = flow_vectors[i + 1].to(device)
                    warped = _warp(prop_masks[i + 1],
                                   grid_x - flow[..., 0] * sx,
                                   grid_y - flow[..., 1] * sy)
                    prop_masks[i] = torch.max(prop_masks[i], warped)

        logger.info(f"[Video Mask Propagator] Propagated sequence ({propagation_mode} mode) along timeline.")
        return (prop_masks,)
