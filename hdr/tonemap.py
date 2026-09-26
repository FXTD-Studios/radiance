from __future__ import annotations

import torch
import logging
from typing import Tuple, Dict, Any

# Local imports
from .utils import (
    tensor_srgb_to_linear,
)

logger = logging.getLogger("radiance.hdr.tonemap")

# ─── AgX working space matrices (Blender/Troy Sobotka, BSD-licensed) ─────────
#
# sRGB linear -> AgX working space, in the orientation `einsum('ij,...j->...i')`
# wants, i.e. rows of M such that `out = M @ rgb`.
#
# The comment here used to claim "transposed for einsum" while the literal was
# the GLSL column-major form copied verbatim as rows. That orientation is not
# white-preserving: as coded, `M @ [1,1,1]` was (0.9272, 1.0353, 1.0375), so
# every neutral picked up a warm cast that grew with exposure — measured channel
# spread 0.042 at 0.18 scene-linear and 0.116 at 16.0, roughly 350-600x the
# error of the correct orientation. `M.T @ [1,1,1]` is (1.00014, 0.99996,
# 0.99995), which is the white-preserving one, so the values below are the
# transpose of what was here.
_AGX_M_IN_VALUES = (
    (0.842479062253094, 0.0784335999999992, 0.0792237451477643),
    (0.0423282422610123, 0.878468636469772, 0.0791661274605434),
    (0.0423756549057051, 0.0784336000000002, 0.879142973793104),
)

# PyTorch versions (lazy-init to avoid device conflict)
_AGX_M_IN_T: torch.Tensor | None = None
_AGX_M_OUT_T: torch.Tensor | None = None


def _agx_matrices_gpu(device, dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
    global _AGX_M_IN_T, _AGX_M_OUT_T
    if (
        _AGX_M_IN_T is None
        or _AGX_M_OUT_T is None
        or _AGX_M_IN_T.device != device
        or _AGX_M_IN_T.dtype != dtype
    ):
        _AGX_M_IN_T = torch.tensor(_AGX_M_IN_VALUES, device=device, dtype=dtype)
        _AGX_M_OUT_T = torch.linalg.inv(_AGX_M_IN_T)
    return _AGX_M_IN_T, _AGX_M_OUT_T


# ═══════════════════════════════════════════════════════════════════════════════
#                          TONE MAPPING NODES
# ═══════════════════════════════════════════════════════════════════════════════


class HDRExpandDynamicRange:
    """
    Expand SDR images to HDR dynamic range by recovering highlights and extending stops of exposure latitude.
    """

    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Display-encoded SDR image. Output is linear light; alpha passes through untouched."}),
                "source_gamma": (
                    "FLOAT",
                    {"default": 2.2, "min": 1.0, "max": 3.0, "step": 0.1, "tooltip": "Input decoding. 2.1 to 2.3 uses the exact sRGB curve, 1.0 means already linear, any other value is a pure power (e.g. 2.4, 2.6)."},
                ),
                "highlight_recovery": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1, "tooltip": "Mix between the linearised input (0) and the full expansion (1). Values above 1 push past the expansion curve."},
                ),
                "black_point": (
                    "FLOAT",
                    {"default": 0.0, "min": -0.1, "max": 0.1, "step": 0.001, "tooltip": "Linear value subtracted before expansion, then clipped at 0. Positive crushes blacks, negative lifts them."},
                ),
                "target_stops": (
                    "FLOAT",
                    {"default": 14.0, "min": 8.0, "max": 20.0, "step": 0.5, "tooltip": "Sets the peak that SDR white is mapped to: 2^(stops - 8) in linear, so 14 gives 64.0 and 8 gives no expansion."},
                ),
                "highlight_rolloff": (
                    "FLOAT",
                    {"default": 1.5, "min": 1.0, "max": 3.0, "step": 0.1, "tooltip": "Where expansion starts: linear luma threshold 1 - (rolloff - 1) x 0.2, capped at 0.95 (0.9 at the default, 0.6 at 3.0). Higher starts earlier with a gentler curve."},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "expand"

    DESCRIPTION = "Expand SDR images to HDR dynamic range by recovering highlights and extending stops of exposure latitude."

    def expand(
        self,
        image: torch.Tensor,
        source_gamma: float = 2.2,
        highlight_recovery: float = 1.0,
        black_point: float = 0.0,
        target_stops: float = 14.0,
        highlight_rolloff: float = 1.5,
    ) -> Tuple[torch.Tensor]:

        # GPU-accelerated implementation
        img = image.float()

        # Alpha is a coverage value, not a colour: it must not be linearised,
        # expanded or ratio-scaled. This whole path used to run on the full
        # array, so `tensor_srgb_to_linear` alone took a 50% matte to 0.214 and
        # the highlight ratio moved it again. Split it off and reattach it
        # untouched, the same way nodes/hdr/colorspace.py already does.
        _alpha = img[..., 3:] if img.shape[-1] >= 4 else None
        if _alpha is not None:
            img = img[..., :3]

        # 1. Convert to linear using tensor helper
        linear = tensor_srgb_to_linear(img, source_gamma)

        # 2. Adjust black point
        linear = linear - black_point
        linear = torch.clamp(linear, min=0.0)

        # 3. Calculate Luminance (to preserve color relationships)
        # Using Rec.709 coefficients
        luma = (
            0.2126 * linear[..., 0] + 0.7152 * linear[..., 1] + 0.0722 * linear[..., 2]
        )

        # 4. Expansion Logic
        # Target Peak: 2^(stops - 8). e.g., 14 stops -> 2^6 = 64.0
        target_peak = 2.0 ** (target_stops - 8.0)

        # Use highlight_rolloff to control the threshold softness
        # Higher rolloff = softer transition, starts expansion earlier
        # Range 1.0-3.0 maps to threshold 0.9-0.5
        threshold = (
            1.0 - (highlight_rolloff - 1.0) * 0.2
        )  # rolloff 1.0->0.9, 1.5->0.8, 3.0->0.5
        threshold = max(0.5, min(0.95, threshold))  # Clamp to safe range

        if target_peak > 1.0 and highlight_recovery > 0:
            t = threshold

            # Calculate 'a' coefficient for quadratic curve
            a = (target_peak - 1.0) / ((1.0 - t) ** 2)

            # Clamp luma for curve calculation
            luma_clamped = torch.clamp(luma, max=1.0)

            # Quadratic expansion: y = t + (x - t) + a * (x - t)^2
            # Only apply to values above threshold
            expanded_luma = torch.where(
                luma > t,
                t + (luma_clamped - t) + a * torch.pow(luma_clamped - t, 2),
                luma,
            )

            # For inputs > 1.0, extrapolate linearly with slope at 1.0
            slope_at_1 = 1.0 + 2 * a * (1.0 - t)
            expanded_luma = torch.where(
                luma > 1.0, target_peak + (luma - 1.0) * slope_at_1, expanded_luma
            )

            # Blend based on recovery strength
            final_luma = expanded_luma * highlight_recovery + luma * (
                1.0 - highlight_recovery
            )

            # Apply new luminance to RGB
            # NewColor = OldColor * (NewLuma / OldLuma)
            ratio = final_luma / (luma + 1e-8)

            # Expand dims for RGB multiply
            ratio = ratio.unsqueeze(-1)

            linear = linear * ratio

        if _alpha is not None:
            linear = torch.cat([linear, _alpha], dim=-1)

        return (linear,)


class HDRToneMap:
    """
    Professional HDR tone mapping with presets and advanced controls. GPU-accelerated.
    """

    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"

    TONEMAP_OPERATORS = [
        "filmic_aces",
        "filmic_uncharted2",
        "agx",
        "reinhard",
        "reinhard_extended",
        "reinhard_luminance",
        "linear_clamp",
        "exposure_only",
    ]

    # Tone map look presets
    LOOK_PRESETS = [
        "None (Custom)",
        "◎ Cinematic Film",
        "◎ HDR Display",
        "◎ Web / Social",
        "◎ Print Ready",
        "◎ Game Engine",
        "◎ Photography",
        "◎ Low Key / Dark",
        "◎ High Key / Bright",
    ]

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input HDR or SDR image to tone map"}),
            },
            "optional": {
                "preset": (
                    cls.LOOK_PRESETS,
                    {
                        "default": "◎ Cinematic Film",
                        "tooltip": "Quick look preset. Overrides settings below.",
                    },
                ),
                "operator": (
                    cls.TONEMAP_OPERATORS,
                    {
                        "default": "filmic_aces",
                        "tooltip": "Tone mapping algorithm:\n• filmic_aces: Industry-standard film curve\n• filmic_uncharted2: Game industry favorite\n• agx: Modern Blender default\n• reinhard: Classic, preserves color\n• linear_clamp: Simple clip",
                    },
                ),
                "exposure": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -5.0,
                        "max": 5.0,
                        "step": 0.1,
                        "tooltip": "Exposure adjustment in stops. Negative = darker, Positive = brighter.",
                    },
                ),
                "gamma": (
                    "FLOAT",
                    {
                        "default": 2.2,
                        "min": 1.0,
                        "max": 3.0,
                        "step": 0.1,
                        "tooltip": "Display gamma. 2.2 = sRGB standard. Higher = darker midtones.",
                    },
                ),
                "white_point": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.5,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": "Maximum brightness that maps to white. Higher = more headroom for highlights.",
                    },
                ),
                "contrast": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.5,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Contrast adjustment around midpoint. >1 = more punch.",
                    },
                ),
                "saturation": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Color saturation. 0 = grayscale, 1 = original, >1 = vivid.",
                    },
                ),
                "highlight_compression": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Compress highlights to prevent clipping. 0 = no compression, 1 = full.",
                    },
                ),
                "shadow_lift": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 0.2,
                        "step": 0.01,
                        "tooltip": "Lift shadows to reveal detail. 0 = no lift, 0.1 = subtle.",
                    },
                ),
                "use_gpu": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Use GPU acceleration when available.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_TOOLTIPS = ("Tone-mapped SDR image ready for display or export.",)
    FUNCTION = "tonemap"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    DESCRIPTION = "Professional HDR tone mapping with presets and advanced controls. GPU-accelerated."

    def _gpu_tonemap(
        self, x: torch.Tensor, operator: str, white_point: float
    ) -> torch.Tensor:
        """GPU-accelerated tone mapping."""
        if operator == "reinhard":
            return x / (1.0 + x)
        elif operator == "reinhard_extended":
            white_sq = white_point * white_point
            return (x * (1.0 + x / white_sq)) / (1.0 + x)
        elif operator == "reinhard_luminance":
            # GPU version of luminance-based Reinhard
            luma = 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]
            luma = torch.clamp(luma, min=1e-6)
            white_sq = white_point * white_point
            luma_tm = (luma * (1.0 + luma / white_sq)) / (1.0 + luma)
            scale = (luma_tm / luma).unsqueeze(-1)
            return x * scale
        elif operator == "filmic_aces":
            # FIX #12 (GPU): white_point acts as an input scene-exposure scale.
            # The ACES curve has no built-in white_point parameter; we divide the
            # input by white_point so values at white_point map to the curve's
            # shoulder, matching user expectation (previously ignored silently).
            a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
            x_scaled = x / (white_point + 1e-8)
            return torch.clamp(
                (x_scaled * (a * x_scaled + b)) / (x_scaled * (c * x_scaled + d) + e),
                0,
                1,
            )
        elif operator == "filmic_uncharted2":
            A, B, C, D, E, F = 0.15, 0.50, 0.10, 0.20, 0.02, 0.30

            def curve(v):
                return (
                    (v * (A * v + C * B) + D * E) / (v * (A * v + B) + D * F)
                ) - E / F

            white_scale = 1.0 / curve(torch.tensor(white_point, device=x.device, dtype=x.dtype))
            return curve(x) * white_scale
        elif operator == "agx":
            # Full AgX pipeline: sRGB → AgX gamut → log-sigmoid → AgX⁻¹ → sRGB
            m_in, m_out = _agx_matrices_gpu(x.device, x.dtype)
            # 1. Transform to AgX working space
            x_agx = torch.einsum('ij,...j->...i', m_in, x.clamp(min=0))
            # 2. Log2 inset: working range →  AgX scene (−10 … +6.5 EV)
            x_log = (torch.log2(x_agx.clamp(min=1e-10)) - (-10.0)) / (6.5 - (-10.0))
            x_log = x_log.clamp(0.0, 1.0)
            # 3. Sigmoid contrast curve (fitted to AgX CDL, Troy Sobotka).
            #
            # This previously read:
            #     x_sig = x_log / (1.0 + torch.abs(x_log - 0.5) * 2.0)
            # For x_log > 0.5 the denominator is exactly 2*x_log, so the whole
            # expression reduces to x/(2x) = 0.5 -- every scene-linear value
            # above 2**(0.5*16.5-10) = 0.2973 collapsed to flat mid-grey. A sky
            # at 5.0 and a specular at 100.0 came out identical.
            #
            # Replaced with a genuine symmetric sigmoid, which is monotonic over
            # the whole domain and agrees with the intended shape near 0.5.
            _p = 1.7   # shoulder/toe firmness
            _d = x_log - 0.5
            x_sig = 0.5 + _d / torch.pow(
                1.0 + torch.pow((2.0 * _d.abs()).clamp(min=1e-8), _p), 1.0 / _p
            )
            x_sig = (x_sig - 0.5) * 1.5 + 0.5   # approx CDL slope
            x_sig = x_sig.clamp(0.0, 1.0)
            # 4. Back to display sRGB
            return torch.einsum('ij,...j->...i', m_out, x_sig).clamp(0, 1)
        elif operator == "linear_clamp":
            return torch.clamp(x / white_point, 0, 1)
        else:  # exposure_only
            return torch.clamp(x, 0, 1)

    def tonemap(
        self,
        image: torch.Tensor,
        preset: str = "◎ Cinematic Film",
        operator: str = "filmic_aces",
        exposure: float = 0.0,
        gamma: float = 2.2,
        white_point: float = 1.0,
        contrast: float = 1.0,
        saturation: float = 1.0,
        highlight_compression: float = 0.5,
        shadow_lift: float = 0.0,
        use_gpu: bool = True,
    ) -> Tuple[torch.Tensor]:

        # ALBABIT-FIX: preset used to overwrite operator/exposure/gamma/...
        # unconditionally on every execution, regardless of what the widgets
        # showed. js/radiance_tonemap.js now fills them on selection and
        # flags manual edits with a "●" marker; the widgets are authoritative.

        if use_gpu and image.device.type == "cpu" and torch.cuda.is_available():
            img = image.to(torch.device("cuda"), non_blocking=True).float()
        else:
            img = image.float()

        # Alpha must survive the operator, the contrast, the saturation and
        # above all the gamma encode. Running it through the full stack took an
        # opaque pixel to 0.730 and a 50% matte to 0.607 -- the plate came back
        # 27% transparent with no error anywhere.
        _alpha = img[..., 3:] if img.shape[-1] >= 4 else None
        if _alpha is not None:
            img = img[..., :3]

        # Apply exposure
        img = img * (2.0**exposure)

        # Apply highlight compression (before tone mapping)
        if highlight_compression > 0:
            threshold = 1.0 - highlight_compression * 0.5
            highlight_mask = img > threshold
            compressed = threshold + (img - threshold) / (
                1.0 + (img - threshold) * highlight_compression * 2
            )
            img = torch.where(highlight_mask, compressed, img)

        # Apply shadow lift
        if shadow_lift > 0:
            img = img + shadow_lift * (1.0 - img)

        # Apply tone mapping operator
        result = self._gpu_tonemap(img, operator, white_point)

        # FIX #10: apply contrast and saturation BEFORE gamma encoding.
        # Previous code applied them after gamma, causing nonlinear colour shifts
        # because the 0.5 midpoint has different physical meaning in gamma space.
        # Apply contrast (linear tone-mapped space)
        if contrast != 1.0:
            result = (result - 0.5) * contrast + 0.5

        # Apply saturation adjustment (linear tone-mapped space)
        if saturation != 1.0 and result.shape[-1] >= 3:
            luma = (
                0.2126 * result[..., 0]
                + 0.7152 * result[..., 1]
                + 0.0722 * result[..., 2]
            )
            luma = luma.unsqueeze(-1)
            result = luma + saturation * (result - luma)

        # Apply gamma (display encoding — must be the final step)
        #
        # AgX is the exception: its sigmoid already emits display code values,
        # so the pow() here was a SECOND transfer encode on top. Measured with
        # it: scene 0.00 -> 0.0477 (a black floor of ~12/255, pure black
        # unreachable), 0.02 -> 0.4845, 0.18 -> 0.7042. filmic_aces on the same
        # input gives 0.1260 and 0.5486. A near-black pixel rendered at 48% grey.
        result = torch.clamp(result, 0, 1)
        if str(operator).lower() != "agx":
            result = torch.pow(result, 1.0 / gamma)

        if _alpha is not None:
            result = torch.cat([result, _alpha.to(result.dtype)], dim=-1)

        return (result,)


# =============================================================================
# NODE MAPPINGS
# FIX #7: NODE_CLASS_MAPPINGS was absent — both nodes were invisible to ComfyUI.
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "RadianceHDRExpandDynamicRange": HDRExpandDynamicRange,
    "RadianceHDRToneMap": HDRToneMap,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceHDRExpandDynamicRange": "◎ HDR Expand Dynamic Range",
    "RadianceHDRToneMap": "◎ HDR Tone Map",
}
