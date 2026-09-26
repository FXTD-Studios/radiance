import torch
import torch.nn.functional as F
import numpy as np
import logging
import json

from radiance.core.tensor.chunking import FrameSink, chunks, compute_device, frames_per_chunk

logger = logging.getLogger("radiance.vfx.inpaint")

class RadianceHDRCrop:
    """
    ◎ Radiance HDR Crop
    
    Crops around a mask with custom padding context. Auto-aligns crop dimensions
    to multiples of 16 to guarantee compatibility with Wan / LTX video architectures.
    """
    
    DESCRIPTION = "Crop an image batch to the region around a mask, with extra context, for inpainting at full resolution. Pair with Radiance HDR Stitch to put the result back."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Plate or frame batch to crop. Values pass through unchanged, so HDR and scene-linear data are kept."}),
                "mask": ("MASK", {"tooltip": "Region to inpaint. Pixels above 0.05 on any frame define one union box used for every frame, so the crop does not move over time. An empty mask crops the full frame."}),
                "context_padding": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 4.0, "step": 0.05, "tooltip": "Crop size as a multiple of the mask's bounding box, centred on it. 1.0 = tight to the mask, 1.5 = 50% larger for surrounding context."}),
                "force_multiple": ("INT", {"default": 16, "min": 1, "max": 256, "step": 1, "tooltip": "Round the crop width and height up to a multiple of this (16 suits Wan and LTX latents). 1 = no rounding. A crop that reaches the frame edge is limited to the frame size and may not be a multiple."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STITCHER_DATA")
    RETURN_NAMES = ("cropped_image", "cropped_mask", "stitcher_data")
    FUNCTION = "apply"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ VFX/Inpainting"

    def apply(self, image: torch.Tensor, mask: torch.Tensor, context_padding: float, force_multiple: int):
        B, H, W, C = image.shape
        device = image.device
        
        # Ensure mask is [B, H, W]
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).expand(B, -1, -1)
            
        # 1. Compute union bounding box across frames (for stable video cropping)
        # Find active pixels
        active_pixels = (mask > 0.05).nonzero()
        
        if active_pixels.numel() == 0:
            # Fallback to full image if mask is empty
            ymin, xmin, ymax, xmax = 0, 0, H - 1, W - 1
        else:
            # Active coords (ignoring batch index for union)
            y_coords = active_pixels[:, 1]
            x_coords = active_pixels[:, 2]
            
            ymin, ymax = y_coords.min().item(), y_coords.max().item()
            xmin, xmax = x_coords.min().item(), x_coords.max().item()
            
        # Bounding box center and size
        box_h = ymax - ymin + 1
        box_w = xmax - xmin + 1
        cy = ymin + box_h // 2
        cx = xmin + box_w // 2
        
        # Expand box by context padding
        new_h = int(box_h * context_padding)
        new_w = int(box_w * context_padding)
        
        # 2. Force multiples of 16 (or custom VAE downscale factors)
        if force_multiple > 1:
            new_h = ((new_h + force_multiple - 1) // force_multiple) * force_multiple
            new_w = ((new_w + force_multiple - 1) // force_multiple) * force_multiple
            
        # Clamp dims to original frame boundaries
        new_h = min(new_h, H)
        new_w = min(new_w, W)
        
        # Calculate new crop box coords centered around original bbox center
        ymin_new = max(0, cy - new_h // 2)
        xmin_new = max(0, cx - new_w // 2)
        
        # Adjust if box hits bottom/right boundaries
        if ymin_new + new_h > H:
            ymin_new = H - new_h
        if xmin_new + new_w > W:
            xmin_new = W - new_w
            
        ymax_new = ymin_new + new_h
        xmax_new = xmin_new + new_w
        
        # 3. Crop batch
        cropped_img = image[:, ymin_new:ymax_new, xmin_new:xmax_new, :]
        cropped_mask = mask[:, ymin_new:ymax_new, xmin_new:xmax_new]
        
        # Package stitcher coordinates metadata
        stitcher_data = {
            "ymin": ymin_new,
            "xmin": xmin_new,
            "ymax": ymax_new,
            "xmax": xmax_new,
            "h_orig": H,
            "w_orig": W,
            "h_crop": new_h,
            "w_crop": new_w
        }
        
        logger.info(f"[HDR Crop] Cropped region: [{ymin_new}:{ymax_new}, {xmin_new}:{xmax_new}] (Wan-safe alignment).")
        return (cropped_img, cropped_mask, stitcher_data)


class RadianceHDRStitch:
    """
    ◎ Radiance HDR Stitch
    
    Stitches inpainted crops seamlessly back into the original plate.
    Uses ACEScg Linear Laplacian Pyramids and Gaussian edge-aware blending
    to eliminate color seams without dynamic range clamping.
    """
    
    DESCRIPTION = "Composite an inpainted crop from Radiance HDR Crop back into the original plate through its mask, with a feathered or multi-band blend. Output is not clamped above 1.0."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE", {"tooltip": "The full-size plate that was passed to HDR Crop."}),
                "cropped_image": ("IMAGE", {"tooltip": "The processed crop. It must keep the crop's exact width and height, as it is pasted back at the stored coordinates."}),
                "cropped_mask": ("MASK", {"tooltip": "Mask for the crop (normally cropped_mask from HDR Crop). Only this area of the crop replaces the plate."}),
                "stitcher_data": ("STITCHER_DATA", {"tooltip": "Crop coordinates from HDR Crop."}),
                "blend_mode": (["Linear_Laplacian", "Linear_Gaussian", "Standard"], {"default": "Linear_Laplacian", "tooltip": "Linear_Laplacian: 3-level multi-band blend with the feathered mask, negatives clamped to 0. Linear_Gaussian and Standard: plain mix through the feathered mask. No transfer conversion is applied."}),
                "feather_radius": ("INT", {"default": 16, "min": 0, "max": 128, "step": 1, "tooltip": "Mask softening radius in pixels (a box blur of 2 x radius + 1), used by every blend mode. 0 preserves the input mask without softening."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("stitched_image", "stitch_blend_mask")
    FUNCTION = "apply"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ VFX/Inpainting"

    @staticmethod
    def _laplacian_blend(original_image: torch.Tensor, full_cropped_img: torch.Tensor, blend_mask: torch.Tensor) -> torch.Tensor:
        """Linear Laplacian pyramid blend in pure PyTorch (unclamped scene-linear)."""
        levels = 3
        gp_orig = [original_image.permute(0, 3, 1, 2)]
        gp_crop = [full_cropped_img.permute(0, 3, 1, 2)]
        gp_mask = [blend_mask.unsqueeze(1)]
        for l in range(levels - 1):
            gp_orig.append(F.avg_pool2d(gp_orig[-1], 3, stride=2, padding=1))
            gp_crop.append(F.avg_pool2d(gp_crop[-1], 3, stride=2, padding=1))
            gp_mask.append(F.avg_pool2d(gp_mask[-1], 3, stride=2, padding=1))
        lp_orig = []
        lp_crop = []
        for l in range(levels - 1):
            size = gp_orig[l].shape[2:]
            up_orig = F.interpolate(gp_orig[l + 1], size=size, mode="bilinear", align_corners=True)
            up_crop = F.interpolate(gp_crop[l + 1], size=size, mode="bilinear", align_corners=True)
            lp_orig.append(gp_orig[l] - up_orig)
            lp_crop.append(gp_crop[l] - up_crop)
        lp_orig.append(gp_orig[-1])
        lp_crop.append(gp_crop[-1])
        lp_fused = [lp_orig[l] * (1.0 - gp_mask[l]) + lp_crop[l] * gp_mask[l] for l in range(levels)]
        recon = lp_fused[-1]
        for l in reversed(range(levels - 1)):
            size = lp_fused[l].shape[2:]
            recon = F.interpolate(recon, size=size, mode="bilinear", align_corners=True) + lp_fused[l]
        # recon is (B, C, H, W). This used to squeeze(0) first, which on a
        # single frame (B == 1, the common case) left a 3-D tensor and the
        # 4-D permute raised: the default blend mode failed on every still.
        return recon.permute(0, 2, 3, 1).clamp(min=0.0)   # scene-linear, unclamped max

    def apply(self, original_image: torch.Tensor, cropped_image: torch.Tensor, cropped_mask: torch.Tensor, stitcher_data: dict, blend_mode: str, feather_radius: int):
        if blend_mode not in ("Standard", "Linear_Gaussian", "Linear_Laplacian"):
            raise ValueError(f"Unknown stitch blend mode: {blend_mode}")
        B, H, W, C = original_image.shape
        ymin = stitcher_data["ymin"]
        xmin = stitcher_data["xmin"]
        ymax = stitcher_data["ymax"]
        xmax = stitcher_data["xmax"]

        if cropped_mask.dim() == 2:
            cropped_mask = cropped_mask.unsqueeze(0)

        def part(x, a, b):
            return x if x.shape[0] == 1 else x[a:b]

        # 3.5.0: a few frames at a time, on the GPU, and the feather as two
        # 1-D passes (the same zero-padded box mean, O(r) instead of O(r^2)).
        # The whole clip used to go through a 2-D blur and two full-frame
        # pyramids at once: 24 s to paste a 208x240 crop into 24 frames of
        # 1024x576, with about 10 frame-sized buffers alive.
        dev = compute_device()
        stitched_out = FrameSink((B, H, W, C))
        mask_out = FrameSink((B, H, W))
        per = frames_per_chunk(H, W, C, 14.0, dev)
        for a, b in chunks(B, per):
            n = b - a
            orig = original_image[a:b].to(dev, torch.float32)
            orig_mask = torch.zeros((n, H, W), device=dev)
            orig_mask[:, ymin:ymax, xmin:xmax] = part(cropped_mask, a, b).to(dev, torch.float32).clamp(0.0, 1.0)

            if feather_radius > 0:
                k = feather_radius * 2 + 1
                pad = feather_radius
                m = F.avg_pool2d(orig_mask.unsqueeze(1), (1, k), stride=1, padding=(0, pad))
                m = F.avg_pool2d(m, (k, 1), stride=1, padding=(pad, 0))
                blend_mask = m.squeeze(1).clamp(0.0, 1.0)
            else:
                blend_mask = orig_mask
            blend_mask_3d = blend_mask.unsqueeze(-1)

            full_cropped_img = orig.clone()
            full_cropped_img[:, ymin:ymax, xmin:xmax, :] = part(cropped_image, a, b).to(dev, torch.float32)

            if blend_mode in ("Standard", "Linear_Gaussian"):
                stitched = orig * (1.0 - blend_mask_3d) + full_cropped_img * blend_mask_3d
            else:
                stitched = self._laplacian_blend(orig, full_cropped_img, blend_mask)
            stitched_out.put(a, b, stitched)
            mask_out.put(a, b, blend_mask)
            del orig, orig_mask, blend_mask, blend_mask_3d, full_cropped_img, stitched

        stitched, blend_mask = stitched_out.value, mask_out.value
        logger.info(f"[HDR Stitch] Composited crop back into frame using {blend_mode} (Feather: {feather_radius}).")
        return (stitched, blend_mask)


class RadianceTemporalStitchStabilizer:
    """
    ◎ Radiance Temporal Stitch Stabilizer
    
    Eliminates video mask edge jitter/popping by running a high-precision 1D temporal Gaussian filter
    along the sequence timeline for each mask pixel.
    """
    
    DESCRIPTION = "Smooth a mask sequence over time with a per-pixel Gaussian filter across frames, to remove edge jitter and popping before stitching."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK", {"tooltip": "Mask sequence, one mask per frame in timeline order. A single mask is returned unchanged."}),
                "temporal_sigma": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 16.0, "step": 0.5, "tooltip": "Gaussian width in frames; the filter reaches 3 x sigma frames each way. Values of 0.1 or less bypass the node."}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("stabilized_masks",)
    FUNCTION = "apply"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ VFX/Inpainting"

    def apply(self, masks: torch.Tensor, temporal_sigma: float):
        B, H, W = masks.shape
        device = masks.device
        
        if temporal_sigma <= 0.1 or B <= 1:
            return (masks,)
            
        # 1D Gaussian kernel along the time axis (dimension 0)
        radius = int(temporal_sigma * 3)
        radius = max(1, radius)
        kernel_size = radius * 2 + 1
        
        # Build 1D Gaussian kernel
        t = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
        kernel = torch.exp(-0.5 * (t / temporal_sigma)**2)
        kernel = kernel / kernel.sum()
        
        # Reshape masks for 1D convolution
        # Shape becomes [1, H*W, B] where H*W are channels, batch is 1, and B is timeline length
        masks_flat = masks.view(B, H * W).permute(1, 0).unsqueeze(0) # [1, HW, B]
        
        # Pad along timeline boundaries to prevent clipping
        padded = F.pad(masks_flat, (radius, radius), mode="replicate")
        
        # Conv1d
        kernel_1d = kernel.view(1, 1, kernel_size).expand(H * W, 1, -1)
        smoothed = F.conv1d(padded, kernel_1d, groups=H * W) # [1, HW, B]
        
        # Permute back to [B, H, W]
        output = smoothed.squeeze(0).permute(1, 0).view(B, H, W).clamp(0.0, 1.0)
        
        logger.info(f"[Temporal Stitch Stabilizer] Applied temporal smoothing filter (Sigma: {temporal_sigma}).")
        return (output,)
