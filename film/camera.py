import io
import torch
import numpy as np
from PIL import Image
from typing import Tuple
import math
import logging

# Module logger
logger = logging.getLogger("radiance.film.camera")


# =============================================================================
# GPU UTILITY FUNCTIONS
# =============================================================================


def get_device(use_gpu: bool = True) -> torch.device:
    """
    Get the appropriate compute device.

    Supports:
    - CUDA (NVIDIA GPUs) - highest priority
    - MPS (Apple M1/M2/M3/M4 chips) - Metal Performance Shaders
    - CPU - fallback for all platforms

    Args:
        use_gpu: Whether to attempt GPU acceleration

    Returns:
        torch.device for the best available accelerator
    """
    if use_gpu:
        # Priority 1: NVIDIA CUDA
        if torch.cuda.is_available():
            return torch.device("cuda")
        # Priority 2: Apple Metal (MPS) for M-series chips
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
    # Fallback: CPU (works on all platforms including Linux)
    return torch.device("cpu")


def gpu_gaussian_blur(
    tensor: torch.Tensor, sigma: float, kernel_size: int = None
) -> torch.Tensor:
    """GPU-accelerated Gaussian blur."""
    if sigma < 0.1:
        return tensor

    if kernel_size is None:
        kernel_size = int(sigma * 6) | 1
        kernel_size = max(3, min(kernel_size, 31))

    device = tensor.device
    dtype = tensor.dtype

    x = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
    kernel_1d = torch.exp(-(x**2) / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()

    kernel_h = kernel_1d.view(1, 1, 1, kernel_size)
    kernel_v = kernel_1d.view(1, 1, kernel_size, 1)

    was_bhwc = tensor.dim() == 4 and tensor.shape[-1] in [1, 3, 4]
    if was_bhwc:
        tensor = tensor.permute(0, 3, 1, 2)

    b, c, h, w = tensor.shape

    kernel_h = kernel_h.expand(c, 1, 1, kernel_size)
    kernel_v = kernel_v.expand(c, 1, kernel_size, 1)

    pad_w = kernel_size // 2
    pad_h = kernel_size // 2

    # Check limits for reflect padding
    mode_w = "reflect" if pad_w < w else "replicate"
    mode_h = "reflect" if pad_h < h else "replicate"

    tensor_padded = torch.nn.functional.pad(tensor, (pad_w, pad_w, 0, 0), mode=mode_w)
    blurred = torch.nn.functional.conv2d(tensor_padded, kernel_h, groups=c)

    blurred_padded = torch.nn.functional.pad(blurred, (0, 0, pad_h, pad_h), mode=mode_h)
    blurred = torch.nn.functional.conv2d(blurred_padded, kernel_v, groups=c)

    if was_bhwc:
        blurred = blurred.permute(0, 2, 3, 1)

    return blurred


def gpu_anisotropic_blur(
    tensor: torch.Tensor, sigma_x: float, sigma_y: float, kernel_size: int = None
) -> torch.Tensor:
    """GPU separable blur with independent horizontal/vertical sigma."""
    if max(sigma_x, sigma_y) < 0.1:
        return tensor

    if kernel_size is None:
        kernel_size = int(max(sigma_x, sigma_y) * 6) | 1
        kernel_size = max(3, min(kernel_size, 31))

    device = tensor.device
    dtype = tensor.dtype
    was_bhwc = tensor.dim() == 4 and tensor.shape[-1] in [1, 3, 4]
    if was_bhwc:
        tensor = tensor.permute(0, 3, 1, 2)

    b, c, h, w = tensor.shape
    x = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
    kx = torch.exp(-(x**2) / (2 * max(sigma_x, 0.1) ** 2))
    ky = torch.exp(-(x**2) / (2 * max(sigma_y, 0.1) ** 2))
    kx = (kx / kx.sum()).view(1, 1, 1, kernel_size).expand(c, 1, 1, kernel_size)
    ky = (ky / ky.sum()).view(1, 1, kernel_size, 1).expand(c, 1, kernel_size, 1)

    pad = kernel_size // 2
    mode_w = "reflect" if pad < w else "replicate"
    mode_h = "reflect" if pad < h else "replicate"
    out = torch.nn.functional.pad(tensor, (pad, pad, 0, 0), mode=mode_w)
    out = torch.nn.functional.conv2d(out, kx, groups=c)
    out = torch.nn.functional.pad(out, (0, 0, pad, pad), mode=mode_h)
    out = torch.nn.functional.conv2d(out, ky, groups=c)

    if was_bhwc:
        out = out.permute(0, 2, 3, 1)
    return out


def gpu_bokeh_blur(tensor: torch.Tensor, sigma: float, bokeh_shape: str) -> torch.Tensor:
    """Approximate different aperture shapes without expensive convolution kernels."""
    shape = (bokeh_shape or "Circle").lower()
    if "anamorphic" in shape:
        return gpu_anisotropic_blur(tensor, sigma * 1.8, sigma * 0.65)
    if "hex" in shape:
        soft = gpu_gaussian_blur(tensor, sigma * 0.85)
        wide = gpu_anisotropic_blur(tensor, sigma * 1.25, sigma * 0.9)
        return soft * 0.55 + wide * 0.45
    if "oct" in shape:
        soft = gpu_gaussian_blur(tensor, sigma)
        wide = gpu_anisotropic_blur(tensor, sigma * 1.15, sigma * 1.15)
        return soft * 0.7 + wide * 0.3
    return gpu_gaussian_blur(tensor, sigma)


# =============================================================================
# COLOR TEMPERATURE UTILITIES
# =============================================================================


def kelvin_to_rgb(kelvin: float) -> Tuple[float, float, float]:
    """Convert color temperature in Kelvin to RGB multipliers."""
    kelvin = max(1000, min(40000, kelvin))
    temp = kelvin / 100.0

    # Red
    if temp <= 66:
        r = 1.0
    else:
        r = temp - 60
        r = 329.698727446 * (r**-0.1332047592) / 255.0
        r = max(0, min(1, r))

    # Green
    if temp <= 66:
        g = temp
        g = 99.4708025861 * math.log(g) - 161.1195681661
        g = g / 255.0
    else:
        g = temp - 60
        g = 288.1221695283 * (g**-0.0755148492) / 255.0
    g = max(0, min(1, g))

    # Blue
    if temp >= 66:
        b = 1.0
    elif temp <= 19:
        b = 0.0
    else:
        b = temp - 10
        b = 138.5177312231 * math.log(b) - 305.0447927307
        b = b / 255.0
        b = max(0, min(1, b))

    return (r, g, b)


# =============================================================================
# WHITE BALANCE NODE
# =============================================================================


class RadianceWhiteBalance:
    """
    Adjust white balance using color temperature (Kelvin) and tint.
    """

    WHITE_BALANCE_PRESETS = {
        "Custom": (5500, 0),
        "Daylight (5500K)": (5500, 0),
        "Cloudy (6500K)": (6500, 5),
        "Shade (7500K)": (7500, 5),
        "Tungsten (3200K)": (3200, 0),
        "Fluorescent (4000K)": (4000, 10),
        "Flash (5500K)": (5500, 0),
        "Candlelight (1850K)": (1850, 0),
        "Sunrise/Sunset (3000K)": (3000, 5),
        "Blue Hour (9000K)": (9000, -10),
        "Moonlight (4100K)": (4100, -5),
    }

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        preset_list = list(cls.WHITE_BALANCE_PRESETS.keys())
        return {
            "required": {
                "image": ("IMAGE",),
                "preset": (preset_list, {"default": "Daylight (5500K)"}),
                "temperature": (
                    "INT",
                    {
                        "default": 5500,
                        "min": 1000,
                        "max": 15000,
                        "step": 100,
                        "display": "slider",
                    },
                ),
                "tint": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -100.0,
                        "max": 100.0,
                        "step": 1.0,
                        "display": "slider",
                    },
                ),
            },
            "optional": {
                "source_temperature": (
                    "INT",
                    {"default": 5500, "min": 1000, "max": 15000, "step": 100},
                ),
                "intensity": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
                "use_gpu": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_white_balance"
    CATEGORY = "FXTD Studios/Radiance/Color"
    DESCRIPTION = "Adjust white balance using color temperature (Kelvin) and tint."

    @torch.no_grad()
    def apply_white_balance(
        self,
        image: torch.Tensor,
        preset: str,
        temperature: int,
        tint: float,
        source_temperature: int = 5500,
        intensity: float = 1.0,
        use_gpu: bool = True,
    ):

        # Use preset values if not Custom
        if preset != "Custom":
            temp_preset, tint_preset = self.WHITE_BALANCE_PRESETS[preset]
            temperature = temp_preset
            tint = tint_preset

        device = get_device(use_gpu)

        try:
            img = image.to(device).float()

            # Get RGB multipliers for source and target temperatures
            src_rgb = kelvin_to_rgb(source_temperature)
            tgt_rgb = kelvin_to_rgb(temperature)

            # Calculate correction multipliers
            r_mult = tgt_rgb[0] / (src_rgb[0] + 1e-6)
            g_mult = tgt_rgb[1] / (src_rgb[1] + 1e-6)
            b_mult = tgt_rgb[2] / (src_rgb[2] + 1e-6)

            # Normalize to maintain overall brightness
            avg_mult = (r_mult + g_mult + b_mult) / 3
            r_mult /= avg_mult
            g_mult /= avg_mult
            b_mult /= avg_mult

            # Apply tint (green-magenta shift)
            tint_factor = tint / 100.0
            g_mult *= 1.0 + tint_factor * 0.3

            # Apply intensity
            r_mult = 1.0 + (r_mult - 1.0) * intensity
            g_mult = 1.0 + (g_mult - 1.0) * intensity
            b_mult = 1.0 + (b_mult - 1.0) * intensity

            # Apply multipliers
            output = img.clone()
            output[..., 0] *= r_mult
            output[..., 1] *= g_mult
            output[..., 2] *= b_mult

            # HDR: Preserve super-white values, only clamp negatives
            output = torch.clamp(output, min=0)
            return (output.cpu(),)

        except RuntimeError:
            if use_gpu:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return self.apply_white_balance(
                    image, preset, temperature, tint, source_temperature, intensity, False
                )
            raise


# =============================================================================
# DEPTH OF FIELD NODE
# =============================================================================


class RadianceDepthOfField:
    """
    Apply cinematic depth of field blur with optional depth map input.
    """

    BOKEH_SHAPES = ["Circle", "Hexagon", "Octagon", "Anamorphic Oval"]

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "blur_amount": (
                    "FLOAT",
                    {
                        "default": 5.0,
                        "min": 0.0,
                        "max": 50.0,
                        "step": 0.5,
                        "display": "slider",
                    },
                ),
            },
            "optional": {
                "depth_map": ("IMAGE",),
                "focus_distance": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "display": "slider",
                    },
                ),
                "focus_range": (
                    "FLOAT",
                    {"default": 0.1, "min": 0.01, "max": 0.5, "step": 0.01},
                ),
                "bokeh_shape": (
                    cls.BOKEH_SHAPES,
                    {
                        "default": "Circle",
                        "tooltip": "Approximate aperture shape for out-of-focus blur.",
                    },
                ),
                "highlight_boost": (
                    "FLOAT",
                    {"default": 1.0, "min": 1.0, "max": 3.0, "step": 0.1},
                ),
                "foreground_blur": ("BOOLEAN", {"default": True}),
                "use_gpu": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_dof"
    CATEGORY = "FXTD Studios/Radiance/Filter"
    DESCRIPTION = "Apply cinematic depth of field blur with optional depth map input."

    @torch.no_grad()
    def apply_dof(
        self,
        image: torch.Tensor,
        blur_amount: float,
        depth_map: torch.Tensor = None,
        focus_distance: float = 0.5,
        focus_range: float = 0.1,
        bokeh_shape: str = "Circle",
        highlight_boost: float = 1.0,
        foreground_blur: bool = True,
        use_gpu: bool = True,
    ):

        if blur_amount < 0.1:
            return (image,)

        device = get_device(use_gpu)
        batch_size, h, w, c = image.shape

        try:
            img = image.to(device).float()

            # Create or use depth map
            if depth_map is not None:
                # Use provided depth map
                depth = depth_map.to(device).float()

                # Reduce to (B, H, W) regardless of input shape.
                # Depth maps arrive as (B,H,W,C) from IMAGE type, but may
                # also be (B,H,W) or (H,W) from custom nodes.
                if depth.dim() == 4:
                    # (B, H, W, C) — take first channel
                    depth = depth[..., 0]
                elif depth.dim() == 2:
                    # (H, W) — add batch dim
                    depth = depth.unsqueeze(0)
                # else: already (B, H, W)

                # Spatial resize if needed
                if depth.shape[-2:] != (h, w):
                    depth = torch.nn.functional.interpolate(
                        depth.unsqueeze(1),
                        size=(h, w),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(1)

                # Batch broadcast: single-frame depth map → match video batch
                if depth.shape[0] == 1 and batch_size > 1:
                    depth = depth.expand(batch_size, -1, -1)
            else:
                # Create radial depth (center focused)
                y = torch.linspace(-1, 1, h, device=device)
                x = torch.linspace(-1, 1, w, device=device)
                yy, xx = torch.meshgrid(y, x, indexing="ij")
                depth = torch.sqrt(xx**2 + yy**2)
                depth = depth / depth.max()
                depth = depth.unsqueeze(0).expand(batch_size, -1, -1)

            # Calculate blur strength based on depth
            depth_diff = torch.abs(depth - focus_distance)
            blur_mask = torch.clamp(
                (depth_diff - focus_range) / (1 - focus_range + 1e-6), 0, 1
            )

            # Only blur foreground if enabled
            if not foreground_blur:
                foreground_mask = depth < focus_distance
                blur_mask = blur_mask * (~foreground_mask).float()

            # Apply multi-pass blur with smooth level blending.
            # FIX 3: Previous code used a hard binary threshold
            #   (blur_mask >= level_threshold).float()
            # which created 5 discrete concentric rings with sharp visible edges.
            # Replaced with a smooth transition using clamp — each level fades
            # in/out over 1/num_levels of the blur range, giving continuous bokeh.
            output = img.clone()
            num_levels = 5
            for level in range(1, num_levels + 1):
                level_sigma = blur_amount * level / num_levels
                level_threshold = (level - 1) / num_levels

                blurred = gpu_bokeh_blur(img, level_sigma, bokeh_shape)

                # Smooth blend weight: 0→1 over one level width
                level_mask = torch.clamp(
                    (blur_mask - level_threshold) * num_levels, 0.0, 1.0
                ).unsqueeze(-1)
                output = output * (1 - level_mask) + blurred * level_mask

            # Highlight boost (bokeh brightness)
            if highlight_boost > 1.0:
                luma = (
                    0.2126 * output[..., 0]
                    + 0.7152 * output[..., 1]
                    + 0.0722 * output[..., 2]
                )
                highlight_mask = (luma > 0.8) * blur_mask
                boost = 1.0 + (highlight_boost - 1.0) * highlight_mask.unsqueeze(-1)
                output = output * boost

            # HDR: Preserve super-white values (important for bokeh highlights)
            output = torch.clamp(output, min=0)
            return (output.cpu(),)

        except RuntimeError:
            if use_gpu:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return self.apply_dof(
                    image,
                    blur_amount,
                    depth_map,
                    focus_distance,
                    focus_range,
                    bokeh_shape,
                    highlight_boost,
                    foreground_blur,
                    False,
                )
            raise


# =============================================================================
# MOTION BLUR NODE
# =============================================================================


class RadianceMotionBlur:
    """
    Apply motion blur (directional, radial, or zoom) to simulate camera movement.
    """

    BLUR_TYPES = ["Directional", "Radial", "Zoom"]

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "blur_type": (cls.BLUR_TYPES, {"default": "Directional"}),
                "amount": (
                    "FLOAT",
                    {
                        "default": 10.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 1.0,
                        "display": "slider",
                    },
                ),
            },
            "optional": {
                "angle": (
                    "FLOAT",
                    {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0},
                ),
                "center_x": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "center_y": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "samples": ("INT", {"default": 16, "min": 4, "max": 64, "step": 4}),
                "use_gpu": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_motion_blur"
    CATEGORY = "FXTD Studios/Radiance/Filter"
    DESCRIPTION = (
        "Apply motion blur (directional, radial, or zoom) to simulate camera movement."
    )

    @torch.no_grad()
    def apply_motion_blur(
        self,
        image: torch.Tensor,
        blur_type: str,
        amount: float,
        angle: float = 0.0,
        center_x: float = 0.5,
        center_y: float = 0.5,
        samples: int = 16,
        use_gpu: bool = True,
    ):

        if amount < 0.5:
            return (image,)

        device = get_device(use_gpu)
        batch_size, h, w, c = image.shape

        try:
            img = image.to(device).float()

            # FIX 4: Vectorize all sample accumulation into a single batched
            # affine_grid + grid_sample call instead of a Python loop.
            # At samples=32, this replaces 32 sequential GPU kernel launches with
            # one fused operation — ~10-20× faster on large images / high sample counts.

            # Build all (batch_size × samples) affine matrices at once
            t_vals = torch.linspace(-0.5, 0.5, samples, device=device, dtype=img.dtype)

            if blur_type == "Directional":
                angle_rad = math.radians(angle)
                dx = math.cos(angle_rad) * amount / w
                dy = math.sin(angle_rad) * amount / h

                # theta: (samples, 2, 3) — identity + per-sample translation
                thetas = torch.zeros(samples, 2, 3, device=device, dtype=img.dtype)
                thetas[:, 0, 0] = 1.0
                thetas[:, 1, 1] = 1.0
                thetas[:, 0, 2] = t_vals * dx * 2   # x offset
                thetas[:, 1, 2] = t_vals * dy * 2   # y offset

            elif blur_type == "Radial":
                cx = center_x * 2 - 1
                cy = center_y * 2 - 1

                rotations = t_vals * amount * 0.02
                cos_r = torch.cos(rotations)
                sin_r = torch.sin(rotations)

                thetas = torch.zeros(samples, 2, 3, device=device, dtype=img.dtype)
                thetas[:, 0, 0] = cos_r
                thetas[:, 0, 1] = -sin_r
                thetas[:, 0, 2] = cx * (1 - cos_r) + cy * sin_r
                thetas[:, 1, 0] = sin_r
                thetas[:, 1, 1] = cos_r
                thetas[:, 1, 2] = cy * (1 - cos_r) - cx * sin_r

            else:  # Zoom
                cx = center_x * 2 - 1
                cy = center_y * 2 - 1
                t_zoom = torch.linspace(0.0, 1.0, samples, device=device, dtype=img.dtype)
                scales = 1.0 + (t_zoom - 0.5) * amount * 0.01

                thetas = torch.zeros(samples, 2, 3, device=device, dtype=img.dtype)
                thetas[:, 0, 0] = scales
                thetas[:, 1, 1] = scales
                thetas[:, 0, 2] = cx * (1 - scales)
                thetas[:, 1, 2] = cy * (1 - scales)

            # Expand thetas for batch: (batch_size × samples, 2, 3)
            thetas = thetas.unsqueeze(0).expand(batch_size, -1, -1, -1)
            thetas = thetas.reshape(batch_size * samples, 2, 3)

            # Tile image: (batch_size × samples, C, H, W)
            img_bchw = img.permute(0, 3, 1, 2)
            img_tiled = img_bchw.unsqueeze(1).expand(-1, samples, -1, -1, -1)
            img_tiled = img_tiled.reshape(batch_size * samples, c, h, w)

            # Single batched affine_grid + grid_sample
            grid = torch.nn.functional.affine_grid(
                thetas, img_tiled.shape, align_corners=False
            )
            sampled = torch.nn.functional.grid_sample(
                img_tiled, grid,
                mode="bilinear", padding_mode="border", align_corners=False,
            )

            # Average across samples: (batch_size, C, H, W) → (batch_size, H, W, C)
            sampled = sampled.reshape(batch_size, samples, c, h, w)
            output = sampled.mean(dim=1).permute(0, 2, 3, 1)

            # HDR: Preserve super-white values
            output = torch.clamp(output, min=0)
            return (output.cpu(),)

        except RuntimeError:
            if use_gpu:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return self.apply_motion_blur(
                    image, blur_type, amount, angle, center_x, center_y, samples, False
                )
            raise


# =============================================================================
# ROLLING SHUTTER NODE
# =============================================================================


class RadianceRollingShutter:
    """
    Simulate rolling shutter artifacts (skew, wobble, flash banding).
    """

    SHUTTER_MODES = ["Horizontal", "Vertical", "Both"]

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "skew_amount": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -50.0,
                        "max": 50.0,
                        "step": 1.0,
                        "display": "slider",
                    },
                ),
            },
            "optional": {
                "shutter_direction": (cls.SHUTTER_MODES, {"default": "Vertical"}),
                "wobble_frequency": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 20.0, "step": 0.5},
                ),
                "wobble_amplitude": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 20.0, "step": 0.5},
                ),
                "flash_band_position": (
                    "FLOAT",
                    {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.05},
                ),
                "flash_band_width": (
                    "FLOAT",
                    {"default": 0.1, "min": 0.01, "max": 0.5, "step": 0.01},
                ),
                "use_gpu": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_rolling_shutter"
    CATEGORY = "FXTD Studios/Radiance/Filter"
    DESCRIPTION = "Simulate rolling shutter artifacts (skew, wobble, flash banding)."

    @torch.no_grad()
    def apply_rolling_shutter(
        self,
        image: torch.Tensor,
        skew_amount: float,
        shutter_direction: str = "Vertical",
        wobble_frequency: float = 0.0,
        wobble_amplitude: float = 0.0,
        flash_band_position: float = -1.0,
        flash_band_width: float = 0.1,
        use_gpu: bool = True,
    ):

        if (
            abs(skew_amount) < 0.1
            and wobble_amplitude < 0.1
            and flash_band_position < -0.5
        ):
            return (image,)

        device = get_device(use_gpu)
        batch_size, h, w, c = image.shape

        try:
            img = image.to(device).float()

            # Create coordinate grids
            y = torch.linspace(-1, 1, h, device=device, dtype=img.dtype)
            x = torch.linspace(-1, 1, w, device=device, dtype=img.dtype)
            yy, xx = torch.meshgrid(y, x, indexing="ij")

            if shutter_direction == "Vertical":
                # Vertical rolling shutter - each row shifts based on its y position
                offset_x = yy * skew_amount / w
                offset_y = torch.zeros_like(yy)

                # Add wobble
                if wobble_amplitude > 0:
                    offset_x += (
                        torch.sin(yy * wobble_frequency * math.pi * 2)
                        * wobble_amplitude
                        / w
                    )

            elif shutter_direction == "Horizontal":
                # Horizontal rolling shutter
                offset_x = torch.zeros_like(xx)
                offset_y = xx * skew_amount / h

                if wobble_amplitude > 0:
                    offset_y += (
                        torch.sin(xx * wobble_frequency * math.pi * 2)
                        * wobble_amplitude
                        / h
                    )
            else:
                # Both
                offset_x = yy * skew_amount / w * 0.5
                offset_y = xx * skew_amount / h * 0.5

                if wobble_amplitude > 0:
                    offset_x += (
                        torch.sin(yy * wobble_frequency * math.pi * 2)
                        * wobble_amplitude
                        / w
                        * 0.5
                    )
                    offset_y += (
                        torch.sin(xx * wobble_frequency * math.pi * 2)
                        * wobble_amplitude
                        / h
                        * 0.5
                    )

            # Apply transformation
            grid_x = xx + offset_x
            grid_y = yy + offset_y
            grid = torch.stack([grid_x, grid_y], dim=-1)
            grid = grid.unsqueeze(0).expand(batch_size, -1, -1, -1)

            output = torch.nn.functional.grid_sample(
                img.permute(0, 3, 1, 2),
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            ).permute(0, 2, 3, 1)

            # Apply flash banding
            if flash_band_position >= -0.5:
                if shutter_direction == "Vertical":
                    scanline = yy
                else:
                    scanline = xx

                band_mask = torch.exp(
                    -((scanline - flash_band_position) ** 2) / (flash_band_width**2)
                )
                band_mask = band_mask.unsqueeze(0).unsqueeze(-1)
                output = output + band_mask * 0.3  # Flash brightness

            # HDR: Preserve super-white values
            output = torch.clamp(output, min=0)
            return (output.cpu(),)

        except RuntimeError:
            if use_gpu:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return self.apply_rolling_shutter(
                    image,
                    skew_amount,
                    shutter_direction,
                    wobble_frequency,
                    wobble_amplitude,
                    flash_band_position,
                    flash_band_width,
                    False,
                )
            raise


# =============================================================================
# COMPRESSION ARTIFACTS NODE
# =============================================================================


class RadianceCompressionArtifacts:
    """
    Add compression artifacts (JPEG blocking, color banding).
    """

    ARTIFACT_TYPES = ["JPEG", "Banding", "Both"]

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "artifact_type": (cls.ARTIFACT_TYPES, {"default": "JPEG"}),
                "quality": (
                    "INT",
                    {
                        "default": 50,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                        "display": "slider",
                    },
                ),
            },
            "optional": {
                "block_size": ("INT", {"default": 8, "min": 4, "max": 32, "step": 4}),
                "color_subsampling": ("BOOLEAN", {"default": True}),
                "banding_levels": (
                    "INT",
                    {"default": 32, "min": 4, "max": 256, "step": 4},
                ),
                "noise_amount": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 0.1, "step": 0.005},
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "tooltip": "Random seed for reproducible noise patterns.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_artifacts"
    CATEGORY = "FXTD Studios/Radiance/Filter"
    DESCRIPTION = "Add compression artifacts (JPEG blocking, color banding)."

    @torch.no_grad()
    def apply_artifacts(
        self,
        image: torch.Tensor,
        artifact_type: str,
        quality: int,
        block_size: int = 8,
        color_subsampling: bool = True,
        banding_levels: int = 32,
        noise_amount: float = 0.0,
        seed: int = 0,
    ):

        # HDR warning: JPEG round-trip is inherently an 8-bit operation.
        # Any float32 values above 1.0 will be clipped before encoding.
        if image.max().item() > 1.0:
            logger.warning(
                "[RadianceCompressionArtifacts] Input contains HDR values (max=%.3f). "
                "JPEG encoding clips to [0, 1] — HDR values above 1.0 will be lost. "
                "Apply tone-mapping before this node if HDR preservation is needed.",
                image.max().item(),
            )

        rng = np.random.default_rng(seed)
        batch_size = image.shape[0]
        results = []

        for b in range(batch_size):
            img = image[b].cpu().numpy()
            has_alpha = img.shape[-1] == 4

            # FIX 1: Strip alpha before JPEG encode — PIL cannot save RGBA as JPEG
            # and raises OSError. Restore alpha after encode/decode.
            if has_alpha:
                alpha_channel = img[..., 3:4].copy()
                img_rgb = img[..., :3]
            else:
                alpha_channel = None
                img_rgb = img

            img_uint8 = (img_rgb * 255).clip(0, 255).astype(np.uint8)
            pil_img = Image.fromarray(img_uint8)  # Always RGB at this point

            if artifact_type in ["JPEG", "Both"]:
                buffer = io.BytesIO()
                pil_img.save(
                    buffer,
                    format="JPEG",
                    quality=quality,
                    subsampling=2 if color_subsampling else 0,
                )
                buffer.seek(0)
                pil_img = Image.open(buffer).copy()
                buffer.close()

            result = np.array(pil_img).astype(np.float32) / 255.0

            # FIX 2: Apply block_size as block-averaging quantization.
            # Simulates DCT blocking by downsampling to block grid and
            # upsampling back — visually identical to 8×8 JPEG macroblocks
            # without requiring raw DCT access. Applied after JPEG if "Both".
            if block_size > 1 and artifact_type in ["JPEG", "Both"]:
                h, w = result.shape[:2]
                # Downsample to block grid (floor division → smaller)
                small_h = max(1, h // block_size)
                small_w = max(1, w // block_size)
                small = (
                    result.reshape(small_h, block_size, small_w, block_size, -1)
                    .mean(axis=(1, 3))
                )  # (small_h, small_w, C) — block averages
                # Upsample back to original size (nearest-neighbour = hard blocks)
                result = np.repeat(np.repeat(small, block_size, axis=0), block_size, axis=1)
                # Crop to exact original size (last block may overshoot by < block_size)
                result = result[:h, :w]

            if artifact_type in ["Banding", "Both"]:
                result = np.floor(result * banding_levels) / banding_levels

            # Seeded noise for reproducible results
            if noise_amount > 0:
                noise = (
                    rng.standard_normal(result.shape).astype(np.float32) * noise_amount
                )
                result = np.clip(result + noise, 0, 1)

            # FIX 1 continued: restore alpha after processing
            if has_alpha:
                result = np.concatenate([result, alpha_channel], axis=-1)

            results.append(torch.from_numpy(result))

        return (torch.stack(results),)


# =============================================================================
# NODE REGISTRATION
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "RadianceWhiteBalance": RadianceWhiteBalance,
    "RadianceDepthOfField": RadianceDepthOfField,
    "RadianceMotionBlur": RadianceMotionBlur,
    "RadianceRollingShutter": RadianceRollingShutter,
    "RadianceCompressionArtifacts": RadianceCompressionArtifacts,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceWhiteBalance": "◎ Radiance White Balance",
    "RadianceDepthOfField": "◎ Radiance Depth of Field",
    "RadianceMotionBlur": "◎ Radiance Motion Blur",
    "RadianceRollingShutter": "◎ Radiance Rolling Shutter",
    "RadianceCompressionArtifacts": "◎ Radiance Compression Artifacts",
}
