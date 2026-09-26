import torch
import numpy as np
import logging
from typing import Tuple, Dict, Any

from .utils import tensor_to_numpy_float32, numpy_to_tensor_float32

logger = logging.getLogger("radiance.hdr.recovery")

try:
    from scipy.ndimage import zoom as _scipy_zoom
    SCIPY_AVAILABLE = True
except ImportError:
    _scipy_zoom = None
    SCIPY_AVAILABLE = False


class RadianceHighlightSynthesis:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    """
    Synthesize high dynamic range details in clipped highlights.

    This node "hallucinates" detail in blown-out areas (> threshold) by:
    1. Expanding the dynamic range using a roll-off curve.
    2. Generating synthetic grain/texture that matches the image statistics.
    3. Blending this detail into the expanded highlights to simulate film clipping.

    This is useful for fixing "flat white" skies or light sources in standard generation outputs.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image with clipped highlights, in any encoding (threshold is compared with its Rec.709 luma). Every channel, alpha included if present, receives the detail noise."}),
                "threshold": (
                    "FLOAT",
                    {
                        "default": 0.95,
                        "min": 0.5,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Luminance level above which to synthesize detail.",
                    },
                ),
                "expansion": (
                    "FLOAT",
                    {
                        "default": 1.5,
                        "min": 1.0,
                        "max": 4.0,
                        "step": 0.1,
                        "tooltip": "Highlight gain above the threshold: gain = 1 + (luma - threshold) x (expansion - 1) x 2, faded in by the highlight mask. 1.0 = no expansion.",
                    },
                ),
                "detail_amount": (
                    "FLOAT",
                    {
                        "default": 0.2,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Strength of synthetic grain in highlights.",
                    },
                ),
                "detail_scale": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 5.0,
                        "step": 0.1,
                        "tooltip": "Scale/frequency of the synthetic detail.",
                    },
                ),
                "blend_mode": (["Screen", "Add", "Soft Light"], {"default": "Screen", "tooltip": "How the monochrome detail noise is combined. Add: plain offset. Screen: a + n - a x n, which inverts the noise on values above 1.0. Soft Light: contrast-style blend, normalised by the frame peak."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                    "tooltip": "Random seed for the detail noise; frame N of a batch uses seed + N. The same seed always gives the same pattern.",
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "synthesize"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    DESCRIPTION = "Synthesize high dynamic range details in clipped highlights to simulate film scan quality."

    def synthesize(
        self,
        image: torch.Tensor,
        threshold: float = 0.95,
        expansion: float = 1.5,
        detail_amount: float = 0.2,
        detail_scale: float = 1.0,
        blend_mode: str = "Screen",
        seed: int = 0,
    ) -> Tuple[torch.Tensor]:

        # Working in numpy for complex masking/noise generation
        img_np = tensor_to_numpy_float32(image)

        # Batch handling
        if img_np.ndim == 3:
            img_np = img_np[np.newaxis, ...]

        batch_size, h, w, c = img_np.shape
        result_batch = np.zeros_like(img_np)

        for b in range(batch_size):
            frame = img_np[b]
            
            # v3.1: Deterministic noise per frame using seed + batch index
            rng = np.random.default_rng(seed + b)

            # 1. Calculate Luminance
            if c >= 3:
                luma = (
                    0.2126 * frame[..., 0]
                    + 0.7152 * frame[..., 1]
                    + 0.0722 * frame[..., 2]
                )
            else:
                luma = frame[..., 0]

            # 2. Create Highlight Mask
            # Soft mask starting at threshold
            mask = np.clip((luma - threshold) / (1.0 - threshold + 1e-6), 0, 1)
            # Smooth step for better blending
            mask = mask * mask * (3 - 2 * mask)

            # If no highlights, skip
            if mask.max() < 1e-4:
                result_batch[b] = frame
                continue

            # 3. Expansion (Inverse Tone Mapping curve)
            # Simple quadratic expansion for values above threshold
            # v_expanded = v + (v - threshold)^2 * expansion_factor

            # Identify highlight pixels
            highlight_pixels = luma > threshold

            # Apply expansion to the image
            expanded_frame = frame.copy()

            # Calculate expansion factor map based on luma
            # Start at 1.0, increase with brightness
            expansion_map = 1.0 + (luma - threshold) * (expansion - 1.0) * 2.0
            expansion_map = np.maximum(1.0, expansion_map)

            # Apply expansion only where mask > 0
            # Blend original and expanded based on mask
            expanded_frame = (
                frame * (1.0 - mask[..., np.newaxis])
                + frame * expansion_map[..., np.newaxis] * mask[..., np.newaxis]
            )

            # 4. Synthesize Detail (Grain/Noise)
            # Generate gaussian noise using seeded RNG
            noise = rng.normal(0, 0.5, (h, w)).astype(np.float32)

            # Scale noise (simulate grain size)
            if detail_scale != 1.0:
                if SCIPY_AVAILABLE and _scipy_zoom is not None:
                    # Generate smaller noise and upscale it
                    h_small = int(h / detail_scale)
                    w_small = int(w / detail_scale)
                    noise_small = rng.normal(0, 0.5, (h_small, w_small)).astype(
                        np.float32
                    )
                    noise = _scipy_zoom(noise_small, (h / h_small, w / w_small), order=1)
                    # Handle size mismatch after zoom
                    noise = noise[:h, :w]
                    if noise.shape != (h, w):
                        noise = np.pad(
                            noise,
                            (
                                (0, max(0, h - noise.shape[0])),
                                (0, max(0, w - noise.shape[1])),
                            ),
                        )[:h, :w]
                else:
                    logger.warning(
                        "[HDRRecovery] scipy not available — falling back to standard "
                        "Gaussian noise. Install scipy for film-grain noise synthesis."
                    )

            # Modulate noise by detail_amount and mask
            noise_layer = noise * detail_amount * mask

            # 5. Blend Noise
            final_frame = expanded_frame.copy()

            # Apply noise to all channels (monochromatic grain) or per-channel?
            # Film grain is usually coupled to dye clouds, so often coupled.
            # Let's apply to all channels identically (luminance noise).

            if blend_mode == "Add":
                for ch in range(c):
                    final_frame[..., ch] += noise_layer
            elif blend_mode == "Screen":
                # Screen: a + b - a*b
                # Works well for highlights — doesn't blow out as fast as Add
                for ch in range(c):
                    final_frame[..., ch] = final_frame[..., ch] + noise_layer - (final_frame[..., ch] * noise_layer)
            elif blend_mode == "Soft Light":
                # Soft Light: standard W3C formula
                # We treat noise_layer (centered at 0) as an offset around 0.5
                n = np.clip(noise_layer + 0.5, 0, 1)

                # SOFT-LIGHT-HDR FIX: the W3C formula
                # ``(1 - 2n)·a² + 2n·a`` is only defined on a in [0,1]. Above
                # 1.0 the quadratic term dominates and goes hard negative: this
                # node expands highlights, so a=5.0 at the shipped threshold
                # 0.95 / expansion 1.5 with n=0.7 evaluated to -219.7, which the
                # np.maximum(0.0, ...) below then clamped to PURE BLACK. With
                # detail_amount=1.0 every pixel above roughly 2.0 linear blacked
                # out, on the node whose only job is to put detail into
                # highlights.
                #
                # Normalising by the frame peak puts the curve back on the
                # domain it is defined for, keeps the result non-negative
                # (the formula maps [0,1] -> [0,1] for any n in [0,1]) and is
                # bit-identical to the old behaviour on SDR frames, where the
                # peak is 1.0 and the divide/multiply cancel. The peak is taken
                # over the whole frame rather than per channel so the blend
                # cannot shift hue.
                peak = max(1.0, float(np.max(final_frame)))
                for ch in range(c):
                    a = final_frame[..., ch] / peak
                    # Soft Light formula (Pegtop / W3C)
                    final_frame[..., ch] = (
                        (1.0 - 2.0 * n) * (a**2) + 2.0 * n * a
                    ) * peak

            # Ensure we strictly expanded range (don't clip back to 1.0)
            # But ensure we don't go below 0
            final_frame = np.maximum(0.0, final_frame)

            result_batch[b] = final_frame

        return (numpy_to_tensor_float32(result_batch),)
