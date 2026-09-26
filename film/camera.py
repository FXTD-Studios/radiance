import io
import os
import torch
import numpy as np
from PIL import Image
from typing import Tuple
import math
import logging

from radiance.core.tensor.chunking import FrameSink, chunks, frames_per_chunk

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


def _make_bokeh_kernel(shape: str, radius: int) -> torch.Tensor:
    """
    Build a normalised 2-D bokeh kernel for the requested aperture shape.

    shape : "Circle" | "Hexagon" | "Octagon" | "Anamorphic Oval"
    radius: half-size of the kernel in pixels (kernel will be 2*radius+1 square)
    """
    import numpy as np
    size = 2 * radius + 1
    cx = cy = radius
    y, x = np.mgrid[0:size, 0:size]
    dx = x - cx
    dy = y - cy

    if shape == "Hexagon":
        # Flat-top hexagon: |x| <= r and |x| + |y|*sqrt(3)/3 * 2 <= r * 4/3
        r = radius
        mask = (np.abs(dx) <= r) & (np.abs(dx) + np.abs(dy) * (2.0 / np.sqrt(3)) <= r * 4.0 / 3.0)
    elif shape == "Octagon":
        # Regular octagon: clipped square — max of Chebyshev and offset L1
        r = radius
        mask = (np.abs(dx) <= r) & (np.abs(dy) <= r) & (np.abs(dx) + np.abs(dy) <= r * 1.41)
    elif shape == "Anamorphic Oval":
        # Wide ellipse: 2:1 aspect ratio (wide x, compressed y)
        mask = ((dx / max(radius, 1)) ** 2 + (dy / max(radius * 0.5, 1)) ** 2) <= 1.0
    else:  # Circle (default)
        mask = (dx ** 2 + dy ** 2) <= radius ** 2

    kernel_np = mask.astype(np.float32)
    total = kernel_np.sum()
    if total > 0:
        kernel_np /= total
    return torch.from_numpy(kernel_np)


def _apply_bokeh_kernel(tensor: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Apply a 2-D bokeh kernel to a BHWC tensor via grouped conv2d."""
    was_bhwc = tensor.dim() == 4 and tensor.shape[-1] in [1, 3, 4]
    if was_bhwc:
        tensor = tensor.permute(0, 3, 1, 2)  # BHWC → BCHW

    b, c, h, w = tensor.shape
    k = kernel.shape[0]
    pad = k // 2

    # Expand kernel to (C, 1, k, k) for grouped conv
    k2d = kernel.to(tensor.device, tensor.dtype).view(1, 1, k, k).expand(c, 1, k, k)

    mode = "reflect" if pad < min(h, w) else "replicate"
    t_padded = torch.nn.functional.pad(tensor, (pad, pad, pad, pad), mode=mode)
    out = torch.nn.functional.conv2d(t_padded, k2d, groups=c)

    if was_bhwc:
        out = out.permute(0, 2, 3, 1)  # BCHW → BHWC
    return out


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


# RadianceWhiteBalance moved to nodes_colorscience.py for v2.5.1


# =============================================================================
# DEPTH OF FIELD NODE
# =============================================================================


class RadianceDepthOfField:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ VFX"
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
                "image": ("IMAGE", {
                    "tooltip": "Image or frame batch to defocus. Works on linear or display-encoded "
                    "values; super-whites are kept.",
                }),
                "blur_amount": (
                    "FLOAT",
                    {
                        "default": 5.0,
                        "min": 0.0,
                        "max": 50.0,
                        "step": 0.5,
                        "display": "slider",
                        "tooltip": "Maximum defocus in pixels: Gaussian sigma for Circle (kernel capped "
                        "at 31 px), kernel radius for the other shapes. Below 0.1 the image is returned unchanged.",
                    },
                ),
            },
            "optional": {
                "depth_map": ("IMAGE", {
                    "tooltip": "Depth in 0-1 from the first channel, 0 = near, 1 = far; resized to the "
                    "image. Without it, a radial map is used (sharp centre, blurred corners).",
                }),
                "focus_distance": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "display": "slider",
                        "tooltip": "Depth value that is in focus, on the depth_map scale (0 = near, "
                        "1 = far). With no depth map, 0 is the frame centre.",
                    },
                ),
                "focus_range": (
                    "FLOAT",
                    {"default": 0.1, "min": 0.01, "max": 0.5, "step": 0.01,
                     "tooltip": "Depth distance either side of focus_distance that stays sharp; blur "
                     "then ramps up to full over the rest of the depth range."},
                ),
                "bokeh_shape": (
                    cls.BOKEH_SHAPES,
                    {
                        "default": "Circle",
                        "tooltip": "Aperture shape. Non-circular shapes use a convolution kernel for physically accurate bokeh.",
                    },
                ),
                "highlight_boost": (
                    "FLOAT",
                    {"default": 1.0, "min": 1.0, "max": 3.0, "step": 0.1,
                     "tooltip": "Brightens defocused highlights (luma above 0.8) by up to this factor, "
                     "scaled by the amount of blur. 1.0 = off."},
                ),
                "foreground_blur": ("BOOLEAN", {"default": True,
                    "tooltip": "Blur areas nearer than focus_distance (depth below it). Off keeps the "
                    "foreground sharp and blurs only the background.",
                }),
                "use_gpu": ("BOOLEAN", {"default": True,
                    "tooltip": "Run the effect on GPU via CUDA/MPS. Falls back to CPU if unavailable.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_dof"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ VFX"
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
            # 3.5.0: a few frames at a time. The whole clip used to be on the
            # device at once with the output, each blur level and the blend
            # temporaries (about 5x the clip), so a long 4K clip ran out of
            # VRAM and fell back to the CPU for the whole batch. Each frame is
            # independent, so the result is unchanged.
            num_levels = 5
            use_shaped = bokeh_shape != "Circle"
            level_kernels = []
            for level in range(1, num_levels + 1):
                level_sigma = blur_amount * level / num_levels
                kern = _make_bokeh_kernel(bokeh_shape, max(1, int(level_sigma))) if use_shaped else None
                level_kernels.append((level_sigma, (level - 1) / num_levels, kern))

            if depth_map is not None:
                depth_src = depth_map
                # Reduce to (B, H, W) regardless of input shape.
                # Depth maps arrive as (B,H,W,C) from IMAGE type, but may
                # also be (B,H,W) or (H,W) from custom nodes.
                if depth_src.dim() == 4:
                    depth_src = depth_src[..., 0]         # (B, H, W, C) — take first channel
                elif depth_src.dim() == 2:
                    depth_src = depth_src.unsqueeze(0)    # (H, W) — add batch dim
            else:
                # Radial depth (centre focused), the same for every frame
                y = torch.linspace(-1, 1, h, device=device)
                x = torch.linspace(-1, 1, w, device=device)
                yy, xx = torch.meshgrid(y, x, indexing="ij")
                radial = torch.sqrt(xx**2 + yy**2)
                radial = (radial / radial.max()).unsqueeze(0)

            out = FrameSink((batch_size, h, w, c))
            per = frames_per_chunk(h, w, c, 6.0, device)
            for a, b in chunks(batch_size, per):
                n = b - a
                img = image[a:b].to(device).float()

                if depth_map is not None:
                    depth = (depth_src if depth_src.shape[0] == 1 else depth_src[a:b]).to(device).float()
                    # Spatial resize if needed
                    if depth.shape[-2:] != (h, w):
                        depth = torch.nn.functional.interpolate(
                            depth.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False,
                        ).squeeze(1)
                    # Batch broadcast: single-frame depth map → match video batch
                    if depth.shape[0] == 1 and n > 1:
                        depth = depth.expand(n, -1, -1)
                else:
                    depth = radial.expand(n, -1, -1)

                # Calculate blur strength based on depth
                depth_diff = torch.abs(depth - focus_distance)
                blur_mask = torch.clamp(
                    (depth_diff - focus_range) / (1 - focus_range + 1e-6), 0, 1
                )

                # Only blur foreground if enabled
                if not foreground_blur:
                    foreground_mask = depth < focus_distance
                    blur_mask = blur_mask * (~foreground_mask).float()

                # Multi-pass blur with smooth level blending: each level fades
                # in over 1/num_levels of the blur range (no hard rings).
                output = img.clone()
                for level_sigma, level_threshold, kern in level_kernels:
                    if use_shaped:
                        blurred = _apply_bokeh_kernel(img, kern)
                    else:
                        blurred = gpu_gaussian_blur(img, level_sigma)
                    level_mask = torch.clamp(
                        (blur_mask - level_threshold) * num_levels, 0.0, 1.0
                    ).unsqueeze(-1)
                    output = output * (1 - level_mask) + blurred * level_mask
                    del blurred, level_mask

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
                out.put(a, b, torch.clamp(output, min=0))
                del img, depth, depth_diff, blur_mask, output
            return (out.value,)

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
# ROLLING SHUTTER NODE
# =============================================================================


class RadianceRollingShutter:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ VFX"
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
                "image": ("IMAGE", {
                    "tooltip": "Image or frame batch. Every frame gets the same static distortion.",
                }),
                "skew_amount": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -50.0,
                        "max": 50.0,
                        "step": 1.0,
                        "display": "slider",
                        "tooltip": "Total shear in pixels between the first and last scanline; the sign "
                        "sets the lean. 0 = no skew.",
                    },
                ),
            },
            "optional": {
                "shutter_direction": (cls.SHUTTER_MODES, {
                    "default": "Vertical",
                    "tooltip": "Vertical: scan runs top to bottom, rows shift sideways. Horizontal: "
                    "columns shift up/down. Both: half of each.",
                }),
                "wobble_frequency": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 20.0, "step": 0.5,
                     "tooltip": "Jello wobble cycles per half frame along the scan axis (2x this over "
                     "the full frame). Needs wobble_amplitude above 0."},
                ),
                "wobble_amplitude": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 20.0, "step": 0.5,
                     "tooltip": "Peak-to-peak wobble displacement in pixels (half that in Both mode). "
                     "0 = off."},
                ),
                "flash_band_position": (
                    "FLOAT",
                    {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.05,
                     "tooltip": "Centre of a partial-exposure flash band along the scan axis, -1 = "
                     "top/left edge, 1 = bottom/right. Any value below -0.5 turns the band off."},
                ),
                "flash_band_width": (
                    "FLOAT",
                    {"default": 0.1, "min": 0.01, "max": 0.5, "step": 0.01,
                     "tooltip": "Gaussian width of the flash band as a fraction of half the frame. "
                     "The band adds a flat +0.3 to all channels."},
                ),
                "use_gpu": ("BOOLEAN", {"default": True,
                    "tooltip": "Run the effect on GPU via CUDA/MPS. Falls back to CPU if unavailable.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_rolling_shutter"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ VFX"
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


def _block_average(result: np.ndarray, block_size: int) -> np.ndarray:
    """Replace each block_size x block_size block of an (H, W, C) frame by its
    mean. Blocks cut off at the right or bottom edge are averaged over the
    pixels they have (the whole-block case is the same arithmetic as before;
    an image side that was not a multiple of block_size used to raise)."""
    h, w = result.shape[:2]
    if h % block_size == 0 and w % block_size == 0:
        small = result.reshape(h // block_size, block_size, w // block_size, block_size, -1).mean(axis=(1, 3))
    else:
        sh, sw = -(-h // block_size), -(-w // block_size)
        padded = np.zeros((sh * block_size, sw * block_size, result.shape[2]), dtype=np.float32)
        padded[:h, :w] = result
        count = np.zeros((sh * block_size, sw * block_size, 1), dtype=np.float32)
        count[:h, :w] = 1.0
        sums = padded.reshape(sh, block_size, sw, block_size, -1).sum(axis=(1, 3))
        small = sums / count.reshape(sh, block_size, sw, block_size, 1).sum(axis=(1, 3))
    # Nearest-neighbour upsample back (hard blocks), cropped to the frame.
    return np.repeat(np.repeat(small, block_size, axis=0), block_size, axis=1)[:h, :w]


class RadianceCompressionArtifacts:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ VFX"
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
                "image": ("IMAGE", {
                    "tooltip": "Display-encoded image. It is quantised to 8 bits and clipped to 0-1 "
                    "in every mode, so HDR values are lost; alpha is kept.",
                }),
                "artifact_type": (cls.ARTIFACT_TYPES, {
                    "default": "JPEG",
                    "tooltip": "JPEG: real JPEG round trip plus block averaging (block_size). Banding: "
                    "posterise to banding_levels. Both: JPEG then banding.",
                }),
                "quality": (
                    "INT",
                    {
                        "default": 50,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                        "display": "slider",
                        "tooltip": "JPEG encoder quality, 1 = worst, 100 = best. Ignored in Banding mode.",
                    },
                ),
            },
            "optional": {
                "block_size": ("INT", {"default": 8, "min": 4, "max": 32, "step": 4,
                    "tooltip": "Size in pixels of the square blocks each averaged to one flat colour after the JPEG pass "
                    "(a mosaic, not a DCT setting). Applies in JPEG and Both. Blocks cut off at the right or bottom edge "
                    "are averaged over the pixels they have.",
                }),
                "color_subsampling": ("BOOLEAN", {"default": True,
                    "tooltip": "Apply chroma subsampling (4:2:0) to simulate video codec color compression.",
                }),
                "banding_levels": (
                    "INT",
                    {"default": 32, "min": 4, "max": 256, "step": 4,
                     "tooltip": "Quantisation steps per channel in Banding and Both modes (rounded down, "
                     "so it darkens by up to one step). Fewer = stronger banding."},
                ),
                "noise_amount": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 0.1, "step": 0.005,
                     "tooltip": "Standard deviation of Gaussian noise added at the end, in 0-1 code "
                     "values (0.01 = about 2.5 of 255). 0 = off."},
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
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_artifacts"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ VFX"
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
        peak = float(image.max()) if image.numel() else 0.0
        if peak > 1.0:
            logger.warning(
                "[RadianceCompressionArtifacts] Input contains HDR values (max=%.3f). "
                "JPEG encoding clips to [0, 1] — HDR values above 1.0 will be lost. "
                "Apply tone-mapping before this node if HDR preservation is needed.",
                peak,
            )

        batch_size, height, width, channels = image.shape
        has_alpha = channels == 4

        def degrade(b: int) -> np.ndarray:
            img = image[b].detach().float().cpu().numpy()
            # PIL cannot save RGBA as JPEG, so alpha is split off here and put
            # back after the noise.
            img_rgb = img[..., :3] if has_alpha else img

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

            # block_size as block averaging: each block is replaced by its mean,
            # a mosaic that reads as 8x8 JPEG macroblocks without DCT access.
            # Applied after JPEG in "Both".
            if block_size > 1 and artifact_type in ["JPEG", "Both"]:
                result = _block_average(result, block_size)

            if artifact_type in ["Banding", "Both"]:
                result = np.floor(result * banding_levels) / banding_levels
            return result

        # 3.5.0: frames are encoded in parallel (PIL's JPEG codec runs outside
        # the GIL) and written straight into the output; they were encoded one
        # after another into a list that was then stacked, a second copy of
        # the clip. The noise is still drawn frame by frame in order from one
        # generator, so a seed gives the same noise as before.
        rng = np.random.default_rng(seed)
        out = torch.empty((batch_size, height, width, channels), dtype=torch.float32)
        workers = max(1, min(batch_size, os.cpu_count() or 1, 8))

        def finish(b: int, result: np.ndarray) -> None:
            # Seeded noise for reproducible results
            if noise_amount > 0:
                noise = (
                    rng.standard_normal(result.shape).astype(np.float32) * noise_amount
                )
                result = np.clip(result + noise, 0, 1)
            out[b, ..., :result.shape[-1]] = torch.from_numpy(result)
            if has_alpha:
                out[b, ..., 3] = image[b, ..., 3].detach().float().cpu()

        if workers == 1:
            for b in range(batch_size):
                finish(b, degrade(b))
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="radiance-jpeg") as pool:
                for b, result in enumerate(pool.map(degrade, range(batch_size))):
                    finish(b, result)

        return (out,)


# =============================================================================
# NODE REGISTRATION
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "RadianceDepthOfField": RadianceDepthOfField,
    "RadianceRollingShutter": RadianceRollingShutter,
    "RadianceCompressionArtifacts": RadianceCompressionArtifacts,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceDepthOfField": "◎ Radiance Depth of Field",
    "RadianceRollingShutter": "◎ Radiance Rolling Shutter",
    "RadianceCompressionArtifacts": "◎ Radiance Compression Artifacts",
}
