import torch
import torch.nn.functional as F
import logging

from radiance.core.tensor.chunking import FrameSink, chunks, compute_device, frames_per_chunk

logger = logging.getLogger("radiance.motion_blur")

class RadianceMotionBlur:
    """
    ◎ Radiance Physical Motion Blur
    
    A professional vector-based motion blur engine. 
    Uses motion vectors to perform sub-frame integration in 32-bit linear space.
    
    Includes Shutter Angle control (180° = standard cinema). The blur is a
    plain average of the samples, which conserves energy: a streak from a
    highlight is dimmer than its source, as on a real shutter.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Frames to blur, same size and batch as motion_vectors. Averaged as given, so scene-linear input gives physically correct highlight streaks."}),
                "motion_vectors": ("IMAGE", {"tooltip": "32-bit UV vectors from Radiance Optical Flow."}),
                "shutter_angle": ("FLOAT", {"default": 180.0, "min": 0.0, "max": 720.0, "step": 1.0,
                    "tooltip": "Standard cinema is 180°. Higher = more blur. 360° = full frame motion blur."}),
                "samples": ("INT", {"default": 8, "min": 2, "max": 32, "step": 1,
                    "tooltip": "Number of sub-frame integration samples. Higher = smoother streaks."}),
                "energy_conservation": ("BOOLEAN", {"default": True,
                    "tooltip": "On: plain average, total light conserved. Off: legacy look, the whole "
                               "frame is scaled so its brightest value matches the source peak "
                               "(brightens everything, not only streaks)."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ VFX"
    DESCRIPTION = "Apply physically-based motion blur using optical flow vectors."

    def apply(self, image: torch.Tensor, motion_vectors: torch.Tensor, shutter_angle: float, samples: int, energy_conservation: bool):
        # image (B, H, W, C); motion_vectors (B, H, W, 3), R = U (dx), G = V (dy), in pixels.
        B, H, W, C = image.shape
        # Shutter angle 180 means we blur over 50% of the motion vector length,
        # integrated from -shutter/2 to +shutter/2 (centred shutter).
        shutter_scale = shutter_angle / 360.0

        # 3.5.0: a few frames at a time, on the GPU. The whole clip used to be
        # integrated at once on the input's device (the CPU in ComfyUI), with a
        # copy of the vectors and full-clip grids: +1 GB for 24 frames of
        # 1024x576. The integration itself is unchanged.
        dev = compute_device()
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=dev),
            torch.linspace(-1, 1, W, device=dev),
            indexing="ij",
        )
        base_grid = torch.stack((x, y), dim=-1).unsqueeze(0)          # (1, H, W, 2)
        out = FrameSink((B, H, W, C))
        per = frames_per_chunk(H, W, C, 7.0, dev)
        for a, b in chunks(B, per):
            img_bchw = image[a:b].to(dev, torch.float32).permute(0, 3, 1, 2)
            vec = motion_vectors[a:b].to(dev, torch.float32)
            # Vectors are in pixels. align_corners=True maps [-1, 1] onto W-1 pixel steps.
            dx_norm = (vec[..., 0] * shutter_scale) / (max(W - 1, 1) / 2.0)
            dy_norm = (vec[..., 1] * shutter_scale) / (max(H - 1, 1) / 2.0)
            uv_norm = torch.stack([dx_norm, dy_norm], dim=-1)
            del vec, dx_norm, dy_norm

            accum = torch.zeros_like(img_bchw)
            for s in range(samples):
                # t goes from -0.5 to 0.5
                t = (s / (samples - 1)) - 0.5 if samples > 1 else 0.0
                sample_grid = base_grid + (uv_norm * t)
                accum += F.grid_sample(img_bchw, sample_grid, mode="bilinear", padding_mode="border", align_corners=True)
                del sample_grid
            result = accum / samples
            del accum, uv_norm

            # The average above already conserves energy. The global peak
            # rescale used to run when energy_conservation was ON, which is the
            # opposite of conserving it: one smeared highlight brightened the
            # whole frame. It is now the opt-out legacy look (per frame).
            if not energy_conservation:
                n = b - a
                orig_max = torch.max(img_bchw.reshape(n, C, -1), dim=-1)[0].view(n, C, 1, 1)
                res_max = torch.max(result.reshape(n, C, -1), dim=-1)[0].view(n, C, 1, 1)
                scale = (orig_max / (res_max + 1e-6)).clamp(min=1.0)
                result = (result * scale).clamp(max=orig_max)

            out.put(a, b, result.permute(0, 2, 3, 1))
            del img_bchw, result

        logger.info(f"[Motion Blur] Applied Vector Blur (Shutter: {shutter_angle}, Samples: {samples})")
        return (out.value,)

NODE_CLASS_MAPPINGS = {
    "RadianceMotionBlur": RadianceMotionBlur,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceMotionBlur": "◎ Radiance Physical Motion Blur",
}
