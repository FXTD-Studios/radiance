import torch
import math
import numpy as np
import logging
from typing import Tuple, Dict, Any

# Local imports
from .utils import tensor_to_numpy_float32, numpy_to_tensor_float32, linear_to_srgb

# Import from canonical color package (replaces deprecated ..color_utils)
from radiance.color.matrices import (
    SRGB_TO_ACESCG,
    ACESCG_TO_SRGB,
    ACES_AP0_TO_AP1,
    ACESCG_TO_REC2020,
    ACESCG_TO_P3D65,
    linear_srgb_to_acescg,
    acescg_to_linear_srgb,
)
from radiance.color.transfer import (
    linear_to_logc3, logc3_to_linear,
    linear_to_logc4, logc4_to_linear,
    linear_to_slog3, slog3_to_linear,
    linear_to_vlog, vlog_to_linear,
    linear_to_canonlog3, canonlog3_to_linear,
    linear_to_acescct, acescct_to_linear,
    linear_to_davinci_intermediate, davinci_intermediate_to_linear,
    linear_to_pq, pq_to_linear,
    linear_to_hlg, hlg_to_linear,
    tensor_linear_to_logc4, tensor_logc4_to_linear,
    tensor_linear_to_slog3, tensor_slog3_to_linear,
    tensor_linear_to_log3g10, tensor_log3g10_to_linear,
    tensor_linear_to_vlog, tensor_vlog_to_linear,
    tensor_linear_to_davinci_intermediate, tensor_davinci_intermediate_to_linear,
)

logger = logging.getLogger("radiance.hdr.color")

#: Scene-light value that BT.2100's HLG OETF maps to signal 0.75 — the reference
#: white point (~203 nits on a 1000-nit display, per BT.2408). Diffuse white must
#: land here, not at 1.0.
_HLG_DIFFUSE_WHITE_SCENE = 0.26



# ═══════════════════════════════════════════════════════════════════════════════
#                      LUMINANCE WEIGHT CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
# Rec.709 / sRGB luminance coefficients
LUMA_REC709 = (0.2126, 0.7152, 0.0722)
# ACEScg / AP1 luminance coefficients
LUMA_AP1 = (0.2722, 0.6741, 0.0537)
# Rec.2020 luminance coefficients
LUMA_REC2020 = (0.2627, 0.6780, 0.0593)


# ═══════════════════════════════════════════════════════════════════════════════
#                     SIGN-PRESERVING POWER UTILITY
# ═══════════════════════════════════════════════════════════════════════════════


def _sign_pow_torch(x: torch.Tensor, exp: float) -> torch.Tensor:
    """
    Sign-preserving power function for HDR data.
    Preserves negative values through the power curve instead of clamping.
    Pure black (0.0) stays at exactly 0.0 — no epsilon pedestal.
    """
    return torch.sign(x) * torch.pow(torch.clamp(torch.abs(x), min=1e-12), exp)


def _sign_pow_np(x: np.ndarray, exp: float) -> np.ndarray:
    """Numpy sign-preserving power."""
    return np.sign(x) * np.power(np.maximum(np.abs(x), 1e-12), exp)


# ═══════════════════════════════════════════════════════════════════════════════
#                          CORE 32-BIT NODES
# ═══════════════════════════════════════════════════════════════════════════════


class ImageToFloat32:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    """
    Convert images to 32-bit float precision for HDR processing.
    Preserves full dynamic range without clamping.

    v2.1 Fixes:
    - FIX: Normalize is now per-frame, not global batch max
    - FIX: source_gamma is now actually applied (was declared but unused)
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image to convert to float32. Values are not clamped."}),
            },
            "optional": {
                "normalize": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Per frame, divide by the frame's maximum when that maximum is above 1.0, so each frame peaks at 1.0. Frames already within 0 to 1 are left as they are.",
                    },
                ),
                "source_gamma": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 4.0,
                        "step": 0.01,
                        "tooltip": "Source gamma to linearize. 1.0 = already linear, 2.2 = sRGB-ish, 2.6 = DCI.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "convert"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    DESCRIPTION = "Convert images to 32-bit float precision for HDR processing. Preserves full dynamic range without clamping."

    def convert(
        self, image: torch.Tensor, normalize: bool = False, source_gamma: float = 1.0
    ) -> Tuple[torch.Tensor]:
        # Ensure float32
        img = image.float()

        # v2.1 FIX: Apply source gamma linearization (was completely unused before)
        if source_gamma != 1.0:
            img = _sign_pow_torch(img, source_gamma)

        # v2.1 FIX: Normalize PER-FRAME, not global batch max.
        # Old code: img / img.max() — one bright pixel in any frame
        # crushed the entire batch (e.g., 0.5 → 0.01 if max was 50.0).
        if normalize:
            for i in range(img.shape[0]):
                frame_max = img[i].max()
                if frame_max > 1.0:
                    img[i] = img[i] / frame_max

        return (img,)


class Float32ColorCorrect:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    """
    Professional 32-bit color correction with exposure, contrast, saturation,
    gamma, and per-channel lift/gain controls.

    v2.1 Fixes:
    - FIX: Gamma uses sign-preserving power (was hard-clamping negatives to 0)
    - FIX: Gamma epsilon no longer creates pedestal lift on pure black
    - FIX: Contrast pivot at 0.18 (linear mid-gray), not 0.5
    - FIX: Operation order: Lift → Exposure → Gain → Contrast → Gamma → Sat → Brightness
    - FIX: Saturation uses colorspace-aware luminance weights
    - NEW: clamp_output toggle (default OFF for HDR pass-through)
    """

    # Luminance weights by working space
    LUMA_WEIGHTS = {
        "Rec.709 / sRGB": LUMA_REC709,
        "ACEScg / AP1": LUMA_AP1,
        "Rec.2020": LUMA_REC2020,
    }

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "Input image to color correct. Processed in 32-bit float precision."
                    },
                ),
                "exposure": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -10.0,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": "Exposure adjustment in stops. +1 = double brightness, -1 = half brightness.",
                    },
                ),
                "contrast": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 4.0,
                        "step": 0.05,
                        "tooltip": "Contrast multiplier around mid-gray (0.18 in linear). >1 = more contrast, <1 = less.",
                    },
                ),
                "brightness": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -1.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Additive brightness offset. Applied last so it works in the final output space.",
                    },
                ),
                "saturation": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 3.0,
                        "step": 0.05,
                        "tooltip": "Color saturation. 0 = grayscale, 1 = original, >1 = boosted.",
                    },
                ),
            },
            "optional": {
                "gamma": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 4.0,
                        "step": 0.01,
                        "tooltip": "Gamma correction: values are raised to 1/gamma. >1 = brighten midtones, <1 = darken. Sign-preserving for HDR.",
                    },
                ),
                "lift_r": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -0.5,
                        "max": 0.5,
                        "step": 0.01,
                        "tooltip": "Value added to the red channel at every level, applied first, so exposure and gain then scale it. A flat offset, not a shadow-weighted lift.",
                    },
                ),
                "lift_g": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -0.5,
                        "max": 0.5,
                        "step": 0.01,
                        "tooltip": "Value added to the green channel at every level, applied first, so exposure and gain then scale it. A flat offset, not a shadow-weighted lift."
                    },
                ),
                "lift_b": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -0.5,
                        "max": 0.5,
                        "step": 0.01,
                        "tooltip": "Value added to the blue channel at every level, applied first, so exposure and gain then scale it. A flat offset, not a shadow-weighted lift."
                    },
                ),
                "gain_r": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "Red channel gain (multiplier). Applied after lift, before contrast.",
                    },
                ),
                "gain_g": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "Green channel gain (multiplier). Applied after lift and exposure, before contrast."
                    },
                ),
                "gain_b": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "Blue channel gain (multiplier). Applied after lift and exposure, before contrast."
                    },
                ),
                "luma_space": (
                    list(cls.LUMA_WEIGHTS.keys()),
                    {
                        "default": "Rec.709 / sRGB",
                        "tooltip": "Luminance weights for saturation. Match your working colorspace.",
                    },
                ),
                "clamp_output": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Clamp output to [0,1]. OFF = HDR pass-through (preserves super-whites/negatives).",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_TOOLTIPS = ("Color corrected image in 32-bit float.",)
    FUNCTION = "correct"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    DESCRIPTION = "Professional 32-bit color correction with exposure, contrast, saturation, gamma, and per-channel lift/gain controls."

    def correct(
        self,
        image: torch.Tensor,
        exposure: float = 0.0,
        contrast: float = 1.0,
        brightness: float = 0.0,
        saturation: float = 1.0,
        gamma: float = 1.0,
        lift_r: float = 0.0,
        lift_g: float = 0.0,
        lift_b: float = 0.0,
        gain_r: float = 1.0,
        gain_g: float = 1.0,
        gain_b: float = 1.0,
        luma_space: str = "Rec.709 / sRGB",
        clamp_output: bool = False,
    ) -> Tuple[torch.Tensor]:

        # ── Fast path: all defaults → no-op ──
        if (
            exposure == 0.0
            and contrast == 1.0
            and brightness == 0.0
            and saturation == 1.0
            and gamma == 1.0
            and lift_r == 0.0
            and lift_g == 0.0
            and lift_b == 0.0
            and gain_r == 1.0
            and gain_g == 1.0
            and gain_b == 1.0
            and not clamp_output
        ):
            return (image,)

        img = image.clone().float()

        # ══════════════════════════════════════════════════════════════
        # v2.1 FIX: Correct grading order (industry standard)
        # Old order: Exposure → Lift/Gain → Contrast → Brightness → Gamma → Sat
        # New order: Lift → Exposure → Gain → Contrast → Gamma → Sat → Brightness
        # ══════════════════════════════════════════════════════════════

        # 1. LIFT (per-channel shadow offset — first in chain)
        if img.shape[-1] >= 3:
            if lift_r != 0.0:
                img[..., 0] = img[..., 0] + lift_r
            if lift_g != 0.0:
                img[..., 1] = img[..., 1] + lift_g
            if lift_b != 0.0:
                img[..., 2] = img[..., 2] + lift_b

        # 2. EXPOSURE (in stops — multiplicative, before gain)
        if exposure != 0.0:
            img = img * (2.0**exposure)

        # 3. GAIN (per-channel multiplier — after exposure)
        if img.shape[-1] >= 3:
            if gain_r != 1.0:
                img[..., 0] = img[..., 0] * gain_r
            if gain_g != 1.0:
                img[..., 1] = img[..., 1] * gain_g
            if gain_b != 1.0:
                img[..., 2] = img[..., 2] * gain_b

        # 4. CONTRAST around mid-gray
        # v2.1 FIX: Pivot at 0.18 (18% reflectance / linear mid-gray),
        # NOT 0.5. Pivoting at 0.5 in linear space shifts 18% gray to
        # -0.14 at 2x contrast — a massive brightness shift.
        if contrast != 1.0:
            pivot = 0.18
            img = (img - pivot) * contrast + pivot

        # 5. GAMMA (sign-preserving power)
        # v2.1 FIX: Old code did torch.clamp(min=0.0) which destroyed
        # HDR negatives (valid wide-gamut data). Then added 1e-8 epsilon
        # BEFORE pow, creating a pedestal lift where pure black became
        # 0.00023 (visible on 10-bit+ displays).
        # New: sign-preserving power keeps negatives, zero stays zero.
        if gamma != 1.0:
            img = _sign_pow_torch(img, 1.0 / gamma)

        # 6. SATURATION (colorspace-aware luminance weights)
        # v2.1 FIX: Was hardcoded to Rec.709 weights (0.2126, 0.7152, 0.0722)
        # which are wrong for ACEScg data (should be 0.2722, 0.6741, 0.0537).
        if saturation != 1.0 and img.shape[-1] >= 3:
            weights = self.LUMA_WEIGHTS.get(luma_space, LUMA_REC709)
            luma = (
                weights[0] * img[..., 0]
                + weights[1] * img[..., 1]
                + weights[2] * img[..., 2]
            )
            luma = luma.unsqueeze(-1)
            img = luma + saturation * (img - luma)

        # 7. BRIGHTNESS (additive offset — last, in output space)
        if brightness != 0.0:
            img = img + brightness

        # Optional output clamp (OFF by default for HDR pass-through)
        if clamp_output:
            img = torch.clamp(img, 0.0, 1.0)

        return (img,)


class ColorSpaceConvert:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    """
    GPU-accelerated color space conversion (sRGB, ACEScg, ACEScct, Rec.2020, DCI-P3).
    Uses industry-standard matrices.

    v2.1 Fixes:
    - FIX: DaVinci Wide Gamut primaries corrected (was copy-paste of DCI-P3)
    - FIX: Removed needless float64 round-trip (2x memory, zero benefit)
    - FIX: sRGB linearization protected against negative inputs
    """

    COLOR_SPACES = [
        "sRGB",
        "Linear_sRGB",
        "ACEScg",
        "ACEScct",
        "ACES2065-1",
        "Rec709",
        "Rec2020",
        "DCI-P3",
        "Display_P3",
        "DaVinci Wide Gamut",
        "ARRI Wide Gamut 4",
        "S-Gamut3.Cine",
    ]

    CHROMATIC_ADAPTATIONS = ["Bradford", "Von Kries", "XYZ Scaling", "None"]

    # Industry-standard precomputed matrices
    FAST_MATRICES = {
        # ── The three spaces that used to fall through ──────────────────
        #
        # `_get_primaries_key` returns "ACES2065-1", "DCI-P3" and "Display_P3"
        # verbatim, and no "<name>_to_Rec709" key existed for them -- so the
        # lookup below missed, logged a warning nobody reads, and applied NO
        # primaries transform at all. Three of the twelve advertised spaces
        # therefore returned the Rec.709 result byte-identically: (1,0,0) in
        # ACES2065-1 gave [0.613097, 0.070194, 0.020616] where AP0->AP1 should
        # give [1.4514, -0.0766, 0.0083].
        #
        # Derived from the published primaries and white points, Bradford-
        # adapted to D65 where the source white differs (AP0 is D60, DCI-P3 is
        # the DCI white). Each one maps [1,1,1] to [1,1,1] exactly.
        "ACES2065-1_to_Rec709": np.array([
            [2.5216861867, -1.1341309882, -0.3875551985],
            [-0.2764799142, 1.3727190877, -0.0962391734],
            [-0.0153780650, -0.1529753359, 1.1683534008],
        ], dtype=np.float32),
        "Rec709_to_ACES2065-1": np.array([
            [0.4396329819, 0.3829886982, 0.1773783199],
            [0.0897764430, 0.8134394287, 0.0967841283],
            [0.0175411704, 0.1115465533, 0.8709122763],
        ], dtype=np.float32),
        "DCI-P3_to_Rec709": np.array([
            [1.1575164062, -0.1549623781, -0.0025540281],
            [-0.0415000715, 1.0455679231, -0.0040678515],
            [-0.0180500390, -0.0785782727, 1.0966283116],
        ], dtype=np.float32),
        "Rec709_to_DCI-P3": np.array([
            [0.8685797397, 0.1289191385, 0.0025011218],
            [0.0345404103, 0.9618113864, 0.0036482034],
            [0.0167714290, 0.0710399978, 0.9121885732],
        ], dtype=np.float32),
        "Display_P3_to_Rec709": np.array([
            [1.2249401763, -0.2249401763, 0.0000000000],
            [-0.0420569547, 1.0420569547, -0.0000000000],
            [-0.0196375546, -0.0786360456, 1.0982736001],
        ], dtype=np.float32),
        "Rec709_to_Display_P3": np.array([
            [0.8224619687, 0.1775380313, -0.0000000000],
            [0.0331941989, 0.9668058011, -0.0000000000],
            [0.0170826307, 0.0723974407, 0.9105199286],
        ], dtype=np.float32),
        "sRGB_to_ACEScg": np.array(
            [
                [0.613097, 0.339523, 0.047379],
                [0.070194, 0.916354, 0.013452],
                [0.020616, 0.109570, 0.869815],
            ],
            dtype=np.float32,
        ),
        "ACEScg_to_sRGB": np.array(
            [
                [1.704858, -0.621716, -0.083299],
                [-0.130078, 1.140735, -0.010560],
                [-0.023964, -0.128975, 1.153014],
            ],
            dtype=np.float32,
        ),
        "Rec709_to_Rec2020": np.array(
            [
                [0.627404, 0.329283, 0.043313],
                [0.069097, 0.919540, 0.011362],
                [0.016392, 0.088013, 0.895595],
            ],
            dtype=np.float32,
        ),
        "Rec2020_to_Rec709": np.array(
            [
                [1.660496, -0.587656, -0.072840],
                [-0.124547, 1.132895, -0.008348],
                [-0.018154, -0.100597, 1.118751],
            ],
            dtype=np.float32,
        ),
        "Rec709_to_DWG": np.array(
            [
                [0.582254, 0.298395, 0.119351],
                [0.050855, 0.908687, 0.040458],
                [0.015735, 0.121737, 0.862528],
            ],
            dtype=np.float32,
        ),
        "DWG_to_Rec709": np.array(
            [
                [1.751098, -0.568044, -0.183054],
                [-0.097606, 1.116203, -0.018597],
                [-0.035373, -0.163816, 1.199189],
            ],
            dtype=np.float32,
        ),
        "Rec709_to_AWG4": np.array(
            [
                [0.550823, 0.338419, 0.110759],
                [0.056937, 0.867987, 0.075076],
                [0.014080, 0.098117, 0.887803],
            ],
            dtype=np.float32,
        ),
        "AWG4_to_Rec709": np.array(
            [
                [1.858410, -0.728057, -0.130353],
                [-0.122735, 1.181503, -0.058768],
                [-0.030110, -0.126855, 1.156965],
            ],
            dtype=np.float32,
        ),
        "Rec709_to_SGamut3Cine": np.array(
            [
                [0.599083, 0.248925, 0.151992],
                [0.054813, 0.943549, 0.001638],
                [-0.003276, 0.017454, 0.985822],
            ],
            dtype=np.float32,
        ),
        "SGamut3Cine_to_Rec709": np.array(
            [
                [1.764564, -0.473984, -0.290580],
                [-0.102593, 1.073170, 0.029423],
                [0.007069, -0.019043, 1.011974],
            ],
            dtype=np.float32,
        ),
    }

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image encoded as source_space."}),
                "source_space": (cls.COLOR_SPACES, {"default": "sRGB", "tooltip": "Encoding of the input. sRGB and Rec709 are decoded with the sRGB curve and ACEScct with its log curve; every other option is treated as linear with those primaries."}),
                "target_space": (cls.COLOR_SPACES, {"default": "ACEScg", "tooltip": "Encoding of the output. sRGB and Rec709 get the sRGB curve and ACEScct its log curve; every other option is written linear with those primaries."}),
            },
            "optional": {
                "exposure": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -10.0,
                        "max": 10.0,
                        "step": 0.01,
                        "tooltip": "Exposure adjustment in stops (EV)",
                    },
                ),
                "gamma_adjust": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 5.0,
                        "step": 0.01,
                        "tooltip": "Gamma adjustment (1.0 = no change)",
                    },
                ),
                "chromatic_adaptation": (
                    cls.CHROMATIC_ADAPTATIONS,
                    {
                        "default": "Bradford",
                        "tooltip": (
                            "Chromatic adaptation method. NOTE: the D60<->D65 "
                            "adaptation is already baked into the precomputed "
                            "FAST_MATRICES this node converts with (Bradford), "
                            "so changing this does not currently alter the "
                            "output. Kept for workflow compatibility; selecting "
                            "a non-Bradford method logs a warning."
                        ),
                    },
                ),
                "use_gpu": ("BOOLEAN", {"default": True,
                    "tooltip": "Run the effect on GPU via CUDA/MPS. Falls back to CPU if unavailable.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "convert"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    DESCRIPTION = "GPU-accelerated color space conversion (sRGB, ACEScg, ACEScct, Rec.2020, DCI-P3). Uses industry-standard matrices."

    # Color space primaries (xy chromaticity)
    PRIMARIES = {
        "sRGB": np.array([[0.64, 0.33], [0.30, 0.60], [0.15, 0.06]]),
        "Linear_sRGB": np.array([[0.64, 0.33], [0.30, 0.60], [0.15, 0.06]]),
        "ACEScg": np.array([[0.713, 0.293], [0.165, 0.830], [0.128, 0.044]]),
        "ACEScct": np.array([[0.713, 0.293], [0.165, 0.830], [0.128, 0.044]]),
        "ACES2065-1": np.array([[0.7347, 0.2653], [0.0, 1.0], [0.0001, -0.077]]),
        "Rec709": np.array([[0.64, 0.33], [0.30, 0.60], [0.15, 0.06]]),
        "Rec2020": np.array([[0.708, 0.292], [0.170, 0.797], [0.131, 0.046]]),
        "DCI-P3": np.array([[0.680, 0.320], [0.265, 0.690], [0.150, 0.060]]),
        "Display_P3": np.array([[0.680, 0.320], [0.265, 0.690], [0.150, 0.060]]),
        # v2.1 FIX: DaVinci Wide Gamut primaries were WRONG — copy-pasted
        # from DCI-P3 (R=[0.680, 0.320]). Actual DWG has much wider gamut
        # with R=[0.800, 0.313]. This caused all dynamic matrix generation
        # for DWG to silently produce P3 conversions instead.
        "DaVinci Wide Gamut": np.array(
            [[0.8000, 0.3130], [0.1682, 0.9877], [0.0790, -0.1155]]
        ),
        "ARRI Wide Gamut 4": np.array(
            [[0.7347, 0.2653], [0.1424, 0.8576], [0.0991, -0.0308]]
        ),
        "S-Gamut3.Cine": np.array([[0.766, 0.275], [0.225, 0.800], [0.089, -0.087]]),
    }

    # White points (xy chromaticity)
    WHITE_POINTS = {
        "sRGB": np.array([0.3127, 0.3290]),
        "Linear_sRGB": np.array([0.3127, 0.3290]),
        "ACEScg": np.array([0.32168, 0.33767]),
        "ACEScct": np.array([0.32168, 0.33767]),
        "ACES2065-1": np.array([0.32168, 0.33767]),
        "Rec709": np.array([0.3127, 0.3290]),
        "Rec2020": np.array([0.3127, 0.3290]),
        "DCI-P3": np.array([0.314, 0.351]),
        "Display_P3": np.array([0.3127, 0.3290]),
        "DaVinci Wide Gamut": np.array([0.3127, 0.3290]),
        "ARRI Wide Gamut 4": np.array([0.3127, 0.3290]),
        "S-Gamut3.Cine": np.array([0.3127, 0.3290]),
    }

    def _get_primaries_key(self, space: str) -> str:
        """Get the primaries key name for fast matrix lookup."""
        if space in ("sRGB", "Linear_sRGB", "Rec709"):
            return "Rec709"
        elif space in ("ACEScg", "ACEScct"):
            return "AP1"
        elif space == "Rec2020":
            return "Rec2020"
        elif space == "DaVinci Wide Gamut":
            return "DWG"
        elif space == "ARRI Wide Gamut 4":
            return "AWG4"
        elif space == "S-Gamut3.Cine":
            return "SGamut3Cine"
        return space

    def _primaries_to_matrix(
        self, primaries: np.ndarray, white: np.ndarray
    ) -> np.ndarray:
        """Calculate RGB to XYZ matrix from primaries and white point."""

        def xy_to_XYZ(xy):
            return np.array([xy[0] / xy[1], 1.0, (1 - xy[0] - xy[1]) / xy[1]])

        r_XYZ = xy_to_XYZ(primaries[0])
        g_XYZ = xy_to_XYZ(primaries[1])
        b_XYZ = xy_to_XYZ(primaries[2])
        w_XYZ = xy_to_XYZ(white)

        M = np.column_stack([r_XYZ, g_XYZ, b_XYZ])
        S = np.linalg.solve(M, w_XYZ)

        return (M * S).astype(np.float32)

    def _srgb_to_linear(self, rgb: np.ndarray) -> np.ndarray:
        """Convert sRGB to linear. v2.1 FIX: protected against negatives."""
        # v2.1 FIX: np.power on negative base with non-integer exponent
        # produces NaN. Old code: np.power((rgb + 0.055) / 1.055, 2.4)
        # When rgb < -0.055, the base is negative → NaN.
        # Fix: clamp base to >=0 in the power branch. The np.where condition
        # routes negatives to the linear branch (rgb/12.92), but numpy
        # evaluates BOTH branches for all elements, generating warnings.
        return np.where(
            rgb <= 0.04045,
            rgb / 12.92,
            np.power(np.maximum((rgb + 0.055) / 1.055, 0.0), 2.4),
        ).astype(np.float32)

    def _linear_to_srgb(self, rgb: np.ndarray) -> np.ndarray:
        """Convert linear to sRGB."""
        return np.where(
            rgb <= 0.0031308,
            rgb * 12.92,
            1.055 * np.power(np.maximum(rgb, 1e-10), 1 / 2.4) - 0.055,
        ).astype(np.float32)

    def _acescg_to_acescct(self, x: np.ndarray) -> np.ndarray:
        """Convert ACEScg to ACEScct."""
        out = np.zeros_like(x, dtype=np.float32)
        mask = x > 0.0078125
        out[mask] = (np.log2(x[mask]) + 9.72) / 17.52
        out[~mask] = x[~mask] * 10.5402377416545 + 0.0729055341958355
        return out

    def _acescct_to_acescg(self, x: np.ndarray) -> np.ndarray:
        """Convert ACEScct to ACEScg."""
        out = np.zeros_like(x, dtype=np.float32)
        mask = x > 0.155251141552511
        out[mask] = np.power(2, x[mask] * 17.52 - 9.72)
        out[~mask] = (x[~mask] - 0.0729055341958355) / 10.5402377416545
        return out

    def _apply_matrix(self, img: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Apply color matrix transformation."""
        return np.tensordot(img, matrix, axes=([-1], [1])).astype(np.float32)

    def _get_gpu_adaptation_matrix(self, src_white, tgt_white, device):
        """Calculate Bradford adaptation matrix on GPU."""
        if np.allclose(src_white, tgt_white):
            return torch.eye(3, device=device)

        M_bradford = torch.tensor(
            [
                [0.8951, 0.2664, -0.1614],
                [-0.7502, 1.7135, 0.0367],
                [0.0389, -0.0685, 1.0296],
            ],
            dtype=torch.float32,
            device=device,
        )

        def xy_to_XYZ(xy):
            return torch.tensor(
                [xy[0] / xy[1], 1.0, (1 - xy[0] - xy[1]) / xy[1]],
                dtype=torch.float32,
                device=device,
            )

        src_w_node = torch.tensor(src_white, dtype=torch.float32, device=device)
        tgt_w_node = torch.tensor(tgt_white, dtype=torch.float32, device=device)

        src_XYZ = xy_to_XYZ(src_w_node)
        tgt_XYZ = xy_to_XYZ(tgt_w_node)

        src_cone = M_bradford @ src_XYZ
        tgt_cone = M_bradford @ tgt_XYZ

        scale = tgt_cone / (src_cone + 1e-8)
        M_adapt = torch.linalg.inv(M_bradford) @ torch.diag(scale) @ M_bradford
        return M_adapt

    def convert(
        self,
        image: torch.Tensor,
        source_space: str = "sRGB",
        target_space: str = "ACEScg",
        exposure: float = 0.0,
        gamma_adjust: float = 1.0,
        chromatic_adaptation: str = "Bradford",
        use_gpu: bool = True,
    ) -> Tuple[torch.Tensor]:

        # The conversion below goes through a Rec.709 hub using precomputed
        # FAST_MATRICES, which already carry a Bradford D60<->D65 adaptation.
        # `_get_gpu_adaptation_matrix` exists and is called from nowhere in the
        # repo, so this widget has never changed a single pixel: "Bradford" and
        # "None" return bit-identical tensors. Say so rather than letting a
        # colourist believe they picked a CAT.
        if chromatic_adaptation != "Bradford":
            _warn_adaptation_is_baked_in(chromatic_adaptation)

        if source_space == target_space:
            # Still apply exposure/gamma if requested
            if exposure != 0.0 or gamma_adjust != 1.0:
                img = image.clone()
                if exposure != 0.0:
                    img = img * (2.0**exposure)
                if gamma_adjust != 1.0:
                    img = _sign_pow_torch(img, gamma_adjust)
                return (img,)
            return (image,)

        # v2.1 FIX: Removed needless float64. Old code converted to
        # float64 for processing then back to float32 on return — 2x
        # memory cost with zero precision benefit for these operations.
        device = image.device
        img = image.cpu().numpy().astype(np.float32)

        for i, frame in enumerate(img):
            rgb = frame.copy()

            # Linearize input
            if source_space in ("sRGB", "Rec709"):
                rgb = self._srgb_to_linear(rgb)
            elif source_space == "ACEScct":
                rgb = self._acescct_to_acescg(rgb)

            # Apply exposure
            if exposure != 0.0:
                rgb = rgb * (2.0**exposure)

            # Apply gamma adjustment (sign-preserving)
            if gamma_adjust != 1.0:
                rgb = _sign_pow_np(rgb, gamma_adjust)

            # Get primaries keys for fast matrix lookup
            inp = self._get_primaries_key(source_space)
            outp = self._get_primaries_key(target_space)

            # Gamut conversion via Rec.709 hub
            if inp != "Rec709":
                if inp == "AP1":
                    key = "ACEScg_to_sRGB"
                else:
                    key = f"{inp}_to_Rec709"
                if key in self.FAST_MATRICES:
                    rgb = self._apply_matrix(rgb, self.FAST_MATRICES[key])
                else:
                    logger.warning(f"[Radiance] Gamut conversion hub '{key}' not found in FAST_MATRICES. Data may be incorrect.")

            if outp != "Rec709":
                if outp == "AP1":
                    key = "sRGB_to_ACEScg"
                else:
                    key = f"Rec709_to_{outp}"
                if key in self.FAST_MATRICES:
                    rgb = self._apply_matrix(rgb, self.FAST_MATRICES[key])
                else:
                    logger.warning(f"[Radiance] Gamut conversion hub '{key}' not found in FAST_MATRICES. Data may be incorrect.")

            # Apply output transfer function
            if target_space == "ACEScct":
                rgb = self._acescg_to_acescct(rgb)
            elif target_space in ("sRGB", "Rec709"):
                rgb = self._linear_to_srgb(rgb)

            img[i] = rgb

        return (torch.from_numpy(img).to(device),)


# ═══════════════════════════════════════════════════════════════════════════════
#                   SHARED MATRIX CONSTANTS (deduplicated)
# ═══════════════════════════════════════════════════════════════════════════════
# v2.1: These were duplicated across DaVinciWideGamut and ARRIWideGamut4
# classes. Extracted to module level for single source of truth.

_SRGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float32,
)

try:
    _XYZ_TO_SRGB = np.linalg.inv(_SRGB_TO_XYZ).astype(np.float32)
except np.linalg.LinAlgError:
    logger.warning("Failed to invert SRGB_TO_XYZ matrix in color.py. Fallback to identity.")
    _XYZ_TO_SRGB = np.eye(3, dtype=np.float32)

_XYZ_TO_AP1 = np.array(
    [[1.6410, -0.3249, -0.2365], [-0.6636, 1.6153, 0.0168], [0.0117, -0.0084, 0.9884]],
    dtype=np.float32,
)

_AP1_TO_XYZ = np.array(
    [[0.6624, 0.1340, 0.1561], [0.2722, 0.6741, 0.0537], [-0.0056, 0.0040, 1.0103]],
    dtype=np.float32,
)

# ── Chromatic adaptation ────────────────────────────────────────────────────
# AP1 (ACEScg) is defined at D60; sRGB, DaVinci Wide Gamut and ARRI Wide Gamut 4
# are all D65. Crossing between them via XYZ without adapting the white point
# leaves a visible tint on neutrals -- a D65 white arrives in ACEScg as a
# non-neutral triplet. These constants supply the missing step.

_WP_D65_XY = (0.3127, 0.3290)
_WP_D60_XY = (0.32168, 0.33767)   # ACES white

_BRADFORD = np.array(
    [
        [ 0.8951,  0.2664, -0.1614],
        [-0.7502,  1.7135,  0.0367],
        [ 0.0389, -0.0685,  1.0296],
    ],
    dtype=np.float64,
)


def _xy_to_XYZ(xy) -> np.ndarray:
    x, y = xy
    return np.array([x / y, 1.0, (1.0 - x - y) / y], dtype=np.float64)


_warned_adaptation_methods: set = set()


def _warn_adaptation_is_baked_in(method: str) -> None:
    """One warning per method per process — this is a config fact, not an event."""
    if method in _warned_adaptation_methods:
        return
    _warned_adaptation_methods.add(method)
    logger.warning(
        "[Radiance] chromatic_adaptation=%r has no effect. The white-point "
        "adaptation is baked into the precomputed conversion matrices "
        "(Bradford), so every option on this widget produces identical output. "
        "Selecting a different CAT would need the hub conversion rebuilt.",
        method,
    )


def _chromatic_adaptation(src_xy, dst_xy) -> np.ndarray:
    """Bradford von-Kries adaptation matrix, XYZ(src white) -> XYZ(dst white)."""
    src_cone = _BRADFORD @ _xy_to_XYZ(src_xy)
    dst_cone = _BRADFORD @ _xy_to_XYZ(dst_xy)
    scale = np.diag(dst_cone / src_cone)
    return (np.linalg.inv(_BRADFORD) @ scale @ _BRADFORD).astype(np.float32)


_ADAPT_D65_TO_D60 = _chromatic_adaptation(_WP_D65_XY, _WP_D60_XY)
_ADAPT_D60_TO_D65 = _chromatic_adaptation(_WP_D60_XY, _WP_D65_XY)


class DaVinciWideGamut:
    """
    Convert to/from DaVinci Wide Gamut and DaVinci Intermediate.
    """

    TRANSFORMS = [
        "Linear to DaVinci WG",
        "DaVinci WG to Linear",
        "Linear to DaVinci Intermediate",
        "DaVinci Intermediate to Linear",
        "DaVinci WG to ACEScg",
        "ACEScg to DaVinci WG",
    ]

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image in the space named on the left of the chosen transform. 'Linear' means linear Rec.709/sRGB primaries, not display-encoded sRGB."}),
                "transform": (cls.TRANSFORMS, {"default": "Linear to DaVinci WG", "tooltip": "Conversion to run. The DaVinci WG options change primaries only (linear in, linear out); the Intermediate options also apply or remove the DaVinci Intermediate log curve."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "convert"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    DESCRIPTION = "Convert to/from DaVinci Wide Gamut and DaVinci Intermediate."

    # DaVinci Wide Gamut to/from XYZ (D65), per Blackmagic's published values.
    #
    # The third row used to read [-0.0099, -0.0315, 0.9417], which is wrong: a
    # colour matrix's row sums are its implied white point, and that gave
    # (0.9507, 1.0000, 0.9003) against D65's (0.95047, 1.0, 1.08883) -- roughly
    # 17% short on blue, so every neutral picked up a green/yellow cast. Rows 0
    # and 1 were already correct to 4dp. Verified: M @ [1,1,1] == D65 XYZ.
    DWG_TO_XYZ = np.array(
        [
            [0.7006224, 0.1487748, 0.1010587],
            [0.2741185, 0.8736319, -0.1477504],
            [-0.0989629, -0.1378953, 1.3259160],
        ],
        dtype=np.float32,
    )

    # Derived from DWG_TO_XYZ rather than a second literal -- the duplicate
    # literal that used to live here is how the two drifted apart.
    XYZ_TO_DWG = np.linalg.inv(DWG_TO_XYZ).astype(np.float32)

    # v2.1: Use shared module-level matrices
    SRGB_TO_XYZ = _SRGB_TO_XYZ
    XYZ_TO_SRGB = _XYZ_TO_SRGB

    def _davinci_intermediate_encode(self, linear: np.ndarray) -> np.ndarray:
        """Encode linear to DaVinci Intermediate log curve."""
        a = 0.0075
        b = 7.0
        c = 0.07329248
        m = 10.44426855
        lin_cut = 0.00262409

        return np.where(
            linear < lin_cut,
            linear * m,
            c * np.log2(np.maximum(linear + a, 1e-10)) + b * 0.1,
        ).astype(np.float32)

    def _davinci_intermediate_decode(self, encoded: np.ndarray) -> np.ndarray:
        """Decode DaVinci Intermediate to linear."""
        a = 0.0075
        b = 7.0
        c = 0.07329248
        m = 10.44426855
        log_cut = 0.02740668

        return np.where(
            encoded < log_cut, encoded / m, np.power(2.0, (encoded - b * 0.1) / c) - a
        ).astype(np.float32)

    def convert(self, image: torch.Tensor, transform: str) -> Tuple[torch.Tensor]:
        img_full = tensor_to_numpy_float32(image)
        # v2.5.1 FIX: Support batch processing (BUG-008)
        if img_full.ndim == 3:
            img_full = img_full[np.newaxis, ...]
        
        batch, h, w, c = img_full.shape
        result_batch = np.zeros_like(img_full)

        for b in range(batch):
            img = img_full[b]
            if transform == "Linear to DaVinci WG":
                xyz = img @ self.SRGB_TO_XYZ.T
                result = xyz @ self.XYZ_TO_DWG.T

            elif transform == "DaVinci WG to Linear":
                xyz = img @ self.DWG_TO_XYZ.T
                result = xyz @ self.XYZ_TO_SRGB.T

            elif transform == "Linear to DaVinci Intermediate":
                xyz = img @ self.SRGB_TO_XYZ.T
                dwg = xyz @ self.XYZ_TO_DWG.T
                result = self._davinci_intermediate_encode(dwg)

            elif transform == "DaVinci Intermediate to Linear":
                dwg = self._davinci_intermediate_decode(img)
                xyz = dwg @ self.DWG_TO_XYZ.T
                result = xyz @ self.XYZ_TO_SRGB.T

            elif transform == "DaVinci WG to ACEScg":
                # DWG is D65, AP1 is D60 -- adapt, or neutrals pick up a tint.
                xyz = img @ self.DWG_TO_XYZ.T @ _ADAPT_D65_TO_D60.T
                result = xyz @ _XYZ_TO_AP1.T

            else:  # ACEScg to DaVinci WG
                xyz = img @ _AP1_TO_XYZ.T @ _ADAPT_D60_TO_D65.T
                result = xyz @ self.XYZ_TO_DWG.T
            
            result_batch[b] = result

        return (numpy_to_tensor_float32(result_batch),)


class ARRIWideGamut4:
    """
    Convert to/from ARRI Wide Gamut 4 (AWG4) for Alexa 35.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Linear image in the space named on the left of the chosen direction. Remove any LogC4 encoding first: this node converts primaries only."}),
                "direction": (
                    [
                        "AWG4 to ACEScg",
                        "ACEScg to AWG4",
                        "AWG4 to Linear sRGB",
                        "Linear sRGB to AWG4",
                    ],
                    {"default": "AWG4 to ACEScg", "tooltip": "Primaries conversion to run, linear in and linear out. ACEScg options include a Bradford D65 to D60 white adaptation."},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "convert"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    DESCRIPTION = "Convert to/from ARRI Wide Gamut 4 (AWG4) for Alexa 35."

    # ARRI Wide Gamut 4 to XYZ (D65), per ARRI's published values.
    #
    # The third row used to read [-0.0094877, -0.0324927, 0.8954361], giving a
    # white point of (0.9505, 1.0000, 0.8535) -- 22% short on blue. AWG4's
    # primaries are chosen so the true third row is exactly [0, 0, 1.0890578].
    # Verified: M @ [1,1,1] == D65 XYZ.
    AWG4_TO_XYZ = np.array(
        [
            [0.7048583, 0.1297603, 0.1158373],
            [0.2545242, 0.7814777, -0.0360019],
            [0.0000000, 0.0000000, 1.0890578],
        ],
        dtype=np.float32,
    )

    XYZ_TO_AWG4 = np.linalg.inv(AWG4_TO_XYZ).astype(np.float32)

    # v2.1: Use shared module-level matrices
    SRGB_TO_XYZ = _SRGB_TO_XYZ
    XYZ_TO_SRGB = _XYZ_TO_SRGB
    XYZ_TO_AP1 = _XYZ_TO_AP1
    AP1_TO_XYZ = _AP1_TO_XYZ

    def convert(self, image: torch.Tensor, direction: str) -> Tuple[torch.Tensor]:
        img_full = tensor_to_numpy_float32(image)
        # v2.5.1 FIX: Support batch processing (BUG-008)
        if img_full.ndim == 3:
            img_full = img_full[np.newaxis, ...]
        
        batch, h, w, c = img_full.shape
        result_batch = np.zeros_like(img_full)

        for b in range(batch):
            img = img_full[b]
            if direction == "AWG4 to ACEScg":
                # AWG4 is D65, AP1 is D60 -- adapt, or neutrals pick up a tint.
                xyz = img @ self.AWG4_TO_XYZ.T @ _ADAPT_D65_TO_D60.T
                result = xyz @ self.XYZ_TO_AP1.T

            elif direction == "ACEScg to AWG4":
                xyz = img @ self.AP1_TO_XYZ.T @ _ADAPT_D60_TO_D65.T
                result = xyz @ self.XYZ_TO_AWG4.T

            elif direction == "AWG4 to Linear sRGB":
                xyz = img @ self.AWG4_TO_XYZ.T
                result = xyz @ self.XYZ_TO_SRGB.T

            else:  # Linear sRGB to AWG4
                xyz = img @ self.SRGB_TO_XYZ.T
                result = xyz @ self.XYZ_TO_AWG4.T
            
            result_batch[b] = result

        return (numpy_to_tensor_float32(result_batch),)


class ACES2OutputTransform:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    # 3.5.0: hidden from the node menu, still fully working. It duplicates
    # "ACES 2.0 Output Transform" (RadianceACES2OutputTransformFull); saved
    # graphs that use it keep loading and rendering correctly.
    DEPRECATED = True
    """
    Apply ACES 2.0 Output Transform with proper gamut mapping
    for SDR, HDR, or Cinema output.

    v2.1 Fixes:
    - FIX: Tonescale no longer hard-clips to [0,1] for HDR outputs
           (was destroying all highlight gradation above 100 nits)
    - FIX: PQ encoder receives proper [0, peak_scale] range
    - FIX: Highlight desaturation uses peak-aware luma threshold
    """

    OUTPUT_TRANSFORMS = [
        "ACES 2.0 SDR (sRGB/Rec.709)",
        "ACES 2.0 SDR (P3-D65)",
        "ACES 2.0 HDR (Rec.2100 PQ 1000 nits)",
        "ACES 2.0 HDR (Rec.2100 PQ 2000 nits)",
        "ACES 2.0 HDR (Rec.2100 PQ 4000 nits)",
        "ACES 2.0 HDR (Rec.2100 HLG)",
        "ACES 2.0 Cinema (DCI-P3 D60)",
        "ACES 2.0 Cinema (DCI-P3 D65)",
    ]

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "input_colorspace": (
                    ["ACEScg", "ACES2065-1", "Linear_sRGB", "Linear_Rec2020"],
                    {
                        "default": "ACEScg",
                        "tooltip": "Input color space. ACEScg = ACES working space (AP1). Linear_sRGB = standard linear.",
                    },
                ),
                "output_transform": (
                    cls.OUTPUT_TRANSFORMS,
                    {
                        "default": "ACES 2.0 SDR (sRGB/Rec.709)",
                        "tooltip": "Target output. SDR for web/broadcast, HDR for HDR10/Dolby Vision, Cinema for theatrical.",
                    },
                ),
            },
            "optional": {
                "peak_luminance": (
                    "FLOAT",
                    {
                        "default": 100.0,
                        "min": 48.0,
                        "max": 10000.0,
                        "step": 1.0,
                        "tooltip": "SDR peak luminance in nits. Standard SDR = 100.",
                    },
                ),
                "surround": (
                    ["Dark", "Dim", "Average"],
                    {
                        "default": "Dim",
                        "tooltip": "Viewing surround. Dark = cinema, Dim = home theater, Average = office.",
                    },
                ),
                "creative_white_scale": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.5,
                        "max": 1.5,
                        "step": 0.01,
                        "tooltip": "Creative exposure adjustment before tone mapping.",
                    },
                ),
                "exposure_adjust": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -4.0,
                        "max": 4.0,
                        "step": 0.1,
                        "tooltip": "Exposure adjustment in stops.",
                    },
                ),
                "gamut_compress": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Gamut compression strength. 1.0 = standard. Higher = more compression.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("output_image", "transform_info")
    FUNCTION = "apply_transform"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    DESCRIPTION = "Apply ACES 2.0 Output Transform with proper gamut mapping for SDR, HDR, or Cinema output."

    # === Color Space Matrices ===
    ACES_AP0_TO_AP1 = np.array(
        [
            [1.4514393161, -0.2365107469, -0.2149285693],
            [-0.0765537734, 1.1762296998, -0.0996759264],
            [0.0083161484, -0.0060324498, 0.9977163014],
        ],
        dtype=np.float32,
    )

    ACES_AP1_TO_sRGB = np.array(
        [
            [1.7050509, -0.6217921, -0.0832588],
            [-0.1302564, 1.1408047, -0.0105483],
            [-0.0240033, -0.1289690, 1.1529723],
        ],
        dtype=np.float32,
    )

    ACES_AP1_TO_P3D65 = np.array(
        [
            [1.3792141, -0.3088546, -0.0703595],
            [-0.0693257, 1.0823507, -0.0130250],
            [-0.0021522, -0.0454616, 1.0476138],
        ],
        dtype=np.float32,
    )

    ACES_AP1_TO_Rec2020 = np.array(
        [
            [1.0258246, -0.0200540, -0.0057706],
            [-0.0023054, 1.0045847, -0.0022793],
            [-0.0050569, -0.0252857, 1.0303426],
        ],
        dtype=np.float32,
    )

    sRGB_TO_AP1 = np.array(
        [[0.6131, 0.3395, 0.0474], [0.0702, 0.9164, 0.0134], [0.0206, 0.1096, 0.8698]],
        dtype=np.float32,
    )

    Rec2020_TO_AP1 = np.array(
        [[0.9788, 0.0165, 0.0047], [0.0014, 0.9983, 0.0003], [0.0044, 0.0235, 0.9721]],
        dtype=np.float32,
    )

    def _compute_luminance(self, rgb: np.ndarray) -> np.ndarray:
        """Compute luminance using ACEScg (AP1) weights."""
        return (
            LUMA_AP1[0] * rgb[..., 0]
            + LUMA_AP1[1] * rgb[..., 1]
            + LUMA_AP1[2] * rgb[..., 2]
        )

    def _gamut_compress(self, rgb: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """
        Gamut compression to avoid clipping saturated colors.
        Based on the ACES gamut compression algorithm.
        """
        if strength <= 0:
            return rgb

        threshold_cyan = 0.815
        threshold_magenta = 0.803
        threshold_yellow = 0.880
        limit = 1.2

        achromatic = np.maximum(rgb[..., 0], np.maximum(rgb[..., 1], rgb[..., 2]))
        achromatic = np.maximum(achromatic, 1e-10)

        dist_r = (achromatic - rgb[..., 0]) / achromatic
        dist_g = (achromatic - rgb[..., 1]) / achromatic
        dist_b = (achromatic - rgb[..., 2]) / achromatic

        def compress_channel(dist, threshold):
            above = dist > threshold
            compressed = np.where(
                above,
                threshold
                + (dist - threshold)
                / (1.0 + ((dist - threshold) / (limit - threshold)) * strength),
                dist,
            )
            return compressed

        dist_r_comp = compress_channel(dist_r, threshold_cyan)
        dist_g_comp = compress_channel(dist_g, threshold_magenta)
        dist_b_comp = compress_channel(dist_b, threshold_yellow)

        result = np.stack(
            [
                achromatic * (1.0 - dist_r_comp),
                achromatic * (1.0 - dist_g_comp),
                achromatic * (1.0 - dist_b_comp),
            ],
            axis=-1,
        )

        return result

    @staticmethod
    def _tanh_tonescale_scalar(x, peak_scale, contrast, toe_power=2.0, pivot=0.18):
        """The per-channel curve of `_apply_tonescale_drt`, for one value.

        Used to probe where 18% grey lands so the input can be scaled to put
        it on the ACES 2.0 reference. This duplicates the chain in the main
        loop deliberately — the loop is vectorised over an image and this is
        called once on a scalar. `test_aces2_reference_grey.py` asserts the two
        agree, so the copy cannot drift silently.
        """
        channel = max(float(x), 1e-10)

        log_contrast = np.log2(channel / pivot) * contrast
        channel_contrast = float(np.power(2.0, log_contrast) * pivot)

        cc_p = np.power(channel_contrast, toe_power)
        channel_toe = float(
            np.power(cc_p / (cc_p + np.power(0.01, toe_power)), 1.0 / toe_power)
            * channel_contrast
        )

        knee = min(1.0, 0.9 * peak_scale)
        headroom = max(peak_scale - knee, 1e-6)
        if channel_toe < knee:
            return channel_toe
        return float(knee + headroom * np.tanh((channel_toe - knee) / headroom))

    def _tanh_tonescale_grey(self, peak_scale, surround_factor):
        """Where 18% scene grey lands on the uncorrected curve."""
        return self._tanh_tonescale_scalar(0.18, peak_scale, 1.55 * surround_factor)

    def _solve_grey_gain(self, target, peak_scale, surround_factor):
        """Input gain g such that curve(0.18 * g) == target.

        Scaling the input by target/curve(0.18) is not enough: the curve is
        non-linear, so the correction has to be solved rather than computed.
        The curve is monotonic in x, so bisection converges reliably and this
        runs once per call, not per pixel.
        """
        contrast = 1.55 * surround_factor
        f = lambda g: self._tanh_tonescale_scalar(0.18 * g, peak_scale, contrast)

        lo, hi = 1e-4, 1.0
        while f(hi) < target and hi < 1e6:
            hi *= 2.0
        if f(hi) < target:            # curve cannot reach it (peak too low)
            return hi

        for _ in range(60):
            mid = math.sqrt(lo * hi)  # geometric: the curve is log-ish in x
            if f(mid) < target:
                lo = mid
            else:
                hi = mid
        return math.sqrt(lo * hi)

    def _apply_tonescale_drt(
        self,
        rgb: np.ndarray,
        peak_luminance: float = 100.0,
        surround: str = "Dim",
        is_hdr: bool = False,
        grey_anchor: bool = True,
    ) -> np.ndarray:
        """
        Apply ACES 2.0 DRT-style tonescale.
        Uses per-channel path-to-white for natural highlight rolloff.

        v2.1 FIX: No longer hard-clips to [0,1] for HDR outputs.
        The old code returned np.clip(result, 0, 1) which destroyed all
        highlight gradation above 100 nits — making 1000-nit PQ output
        identical to SDR. Now clips to [0, peak_scale] for HDR and [0,1]
        for SDR, preserving the full dynamic range for PQ/HLG encoding.
        """
        surround_factor = {"Dark": 0.9, "Dim": 1.0, "Average": 1.1}.get(surround, 1.0)

        peak_scale = peak_luminance / 100.0

        # Anchor 18% grey where ACES 2.0 puts it.
        #
        # The log-contrast + tanh shaping below is an approximation of the
        # reference DRT, and it held grey at ~18 nits on every peak: the right
        # behaviour in kind (ACES 2.0 does keep the midtone nearly still as the
        # display gets brighter) but ~0.85 stop above the 10-nit SDR reference.
        # Scaling the input so 0.18 lands on the published value keeps the
        # curve's highlight roll-off while putting the midtone where a
        # reference monitor would.
        from radiance.hdr.tonescale import aces2_midgrey_nits

        _target_grey = aces2_midgrey_nits(peak_luminance) / 100.0
        # HLG opts out: it passes a synthetic peak (100 / diffuse-white-scene
        # = ~385) to shape the curve, not a display luminance, and its grey is
        # governed by BT.2408 — reference grey at signal 0.38, about 26 nits on
        # a 1000-nit HLG display — not by the ACES table. Anchoring it here
        # moved HLG grey to signal 0.31 and broke that deliberate calibration.
        if grey_anchor:
            _grey_gain = self._solve_grey_gain(_target_grey, peak_scale, surround_factor)
            rgb = rgb * _grey_gain

        contrast = 1.55 * surround_factor
        pivot = 0.18
        toe_power = 2.0

        result = np.zeros_like(rgb)

        for c in range(3):
            channel = rgb[..., c]
            channel = np.maximum(channel, 1e-10)

            # Log-space contrast around pivot
            log_channel = np.log2(channel / pivot)
            log_contrast = log_channel * contrast
            channel_contrast = np.power(2.0, log_contrast) * pivot

            # Toe (shadows)
            channel_toe = (
                np.power(
                    np.power(channel_contrast, toe_power)
                    / (
                        np.power(channel_contrast, toe_power)
                        + np.power(0.01, toe_power)
                    ),
                    1.0 / toe_power,
                )
                * channel_contrast
            )

            # Shoulder (highlights).
            #
            # Two defects were fixed here:
            #
            # 1. Midtones tracked the peak. The old form ended in
            #    `channel_shoulder / peak_scale`, which divided the WHOLE curve
            #    -- not just the highlight region -- by the peak. Measured: 18%
            #    grey rendered at 1.80 nits on a 1000-nit target and 0.45 nits
            #    on a 4000-nit one. Raising the peak made the picture darker.
            #    Diffuse midtones must sit at the same luminance regardless of
            #    how much highlight headroom the display has.
            #
            # 2. A 10% step at the knee. The branches disagreed there: with
            #    tanh(0) == 0 the upper branch evaluated to `white_scale` while
            #    the lower gave `0.9 * white_scale`, so 0.8999*ws -> 0.89990 and
            #    0.9001*ws -> 0.99999 -- a visible contour ring around every
            #    highlight, and non-monotonic past it.
            #
            # The replacement keeps everything below the knee identity-mapped
            # and rolls the excess into the available headroom with a tanh.
            # It is continuous AND C1 at the knee (d/dx of h*tanh(x/h) is 1 at
            # x=0, matching the identity below), monotonic everywhere, and
            # asymptotes to exactly peak_scale.
            #
            # The knee sits at diffuse white for HDR, and at 0.9 for SDR so the
            # existing SDR roll-off is preserved rather than becoming a clip.
            knee = min(1.0, 0.9 * peak_scale)
            headroom = max(peak_scale - knee, 1e-6)
            excess = channel_toe - knee
            channel_shoulder = np.where(
                channel_toe < knee,
                channel_toe,
                knee + headroom * np.tanh(excess / headroom),
            )

            result[..., c] = channel_shoulder

        # Desaturate very bright highlights (path-to-white)
        luma = self._compute_luminance(result)
        # v2.1 FIX: For HDR, the luma can legitimately exceed 1.0.
        # Scale the saturation threshold relative to peak_scale so
        # highlight desaturation kicks in proportionally.
        if is_hdr:
            # Desaturate above 80% of peak — preserves HDR color
            sat_factor = np.clip(1.0 - np.power(luma / peak_scale, 3.0), 0.3, 1.0)
        else:
            sat_factor = np.clip(1.0 - np.power(luma, 3.0), 0.3, 1.0)

        result_desat = (
            luma[..., np.newaxis]
            + (result - luma[..., np.newaxis]) * sat_factor[..., np.newaxis]
        )

        # v2.1 FIX: Clip to appropriate range based on output type.
        # OLD: return np.clip(result_desat, 0, 1)  ← killed HDR headroom
        if is_hdr:
            # HDR: Allow values up to peak_scale (will be mapped by PQ/HLG encoder)
            return np.clip(result_desat, 0, peak_scale)
        else:
            # SDR/Cinema: Clip to [0, 1] display range
            return np.clip(result_desat, 0, 1)

    def _pq_encode(self, linear: np.ndarray, peak_nits: float = 1000.0) -> np.ndarray:
        """
        Encode to ST.2084 PQ (Perceptual Quantizer).
        v2.1: Input is now scene-referred [0, peak_nits/100] from tonescale,
        not display-referred [0,1].
        """
        # Normalize to 10000 nits reference
        # v2.1 FIX: Input `linear` is now in [0, peak_scale] range from
        # the HDR tonescale, representing [0, peak_nits] scene nits.
        # Map to absolute nits, then normalize to PQ's 10000-nit range.
        L = np.clip(linear * 100.0 / 10000.0, 0, 1)

        # PQ EOTF constants (ST.2084)
        m1 = 0.1593017578125
        m2 = 78.84375
        c1 = 0.8359375
        c2 = 18.8515625
        c3 = 18.6875

        Lm1 = np.power(np.maximum(L, 0), m1)
        pq = np.power((c1 + c2 * Lm1) / (1 + c3 * Lm1), m2)

        return pq

    def _hlg_encode(self, linear: np.ndarray) -> np.ndarray:
        """Encode to ARIB STD-B67 HLG (Hybrid Log-Gamma)."""
        a = 0.17883277
        b = 0.28466892
        c = 0.55991073

        hlg = np.where(
            linear <= 1 / 12,
            np.sqrt(3 * np.maximum(linear, 0)),
            a * np.log(np.maximum(12 * linear - b, 1e-10)) + c,
        )

        return np.clip(hlg, 0, 1)

    def apply_transform(
        self,
        image: torch.Tensor,
        input_colorspace: str,
        output_transform: str,
        peak_luminance: float = 100.0,
        surround: str = "Dim",
        creative_white_scale: float = 1.0,
        exposure_adjust: float = 0.0,
        gamut_compress: float = 1.0,
    ) -> Tuple[torch.Tensor, str]:

        img = tensor_to_numpy_float32(image)
        if img.ndim == 4:
            img = img[0]

        # 1. Apply exposure
        if exposure_adjust != 0:
            img = img * (2.0**exposure_adjust)

        # 2. Convert to ACEScg working space
        if input_colorspace == "ACES2065-1":
            img = img @ self.ACES_AP0_TO_AP1.T
        elif input_colorspace == "Linear_sRGB":
            img = img @ self.sRGB_TO_AP1.T
        elif input_colorspace == "Linear_Rec2020":
            img = img @ self.Rec2020_TO_AP1.T

        # 3. Apply creative adjustment
        img = img * creative_white_scale

        # 4. Gamut compression (before tone mapping)
        img = self._gamut_compress(img, gamut_compress)

        # 5. Determine output parameters
        is_hdr = "HDR" in output_transform
        is_pq = "PQ" in output_transform
        is_hlg = "HLG" in output_transform
        is_p3 = "P3" in output_transform
        is_rec2020 = "2100" in output_transform or "Rec.2100" in output_transform
        is_cinema = "Cinema" in output_transform

        # Extract peak nits for HDR
        if is_hdr and is_pq:
            if "4000" in output_transform:
                peak_nits = 4000.0
            elif "2000" in output_transform:
                peak_nits = 2000.0
            else:
                peak_nits = 1000.0
        elif is_cinema:
            # 48 nits is the LUMINANCE OF CODE VALUE 1.0 on a DCI projector, not
            # a scale factor on the scene. Feeding 48 here made peak_scale 0.48,
            # so the tonescale knee landed at 0.432 with 0.048 of headroom and
            # the shoulder saturated almost immediately: measured, ACEScg neutral
            # 0.40 -> 0.75403, 0.50 -> 0.75405, 8.0 -> 0.75405. A theatrical DCP
            # was flat above 0.4 scene-linear with a peak white of 0.754, i.e.
            # about 27 nits instead of 48. DCI is a full-range [0,1] encode, so
            # the tonescale runs SDR-normalised and the 48 belongs in the info
            # string, not the maths.
            peak_nits = 100.0
        elif is_hlg:
            # HLG is a RELATIVE system. BT.2100's OETF puts HLG reference white
            # (diffuse white, ~203 nits on a 1000-nit display per BT.2408) at
            # scene 0.26 -> signal 0.75, and signal 1.0 is the display peak.
            #
            # This branch used to fall through to `peak_luminance`, which
            # defaults to 100, so the tonescale asymptoted at 1.0 and _hlg_encode
            # mapped diffuse white straight to signal 1.0: measured 18% grey ->
            # 127.5 nits (BT.2408 wants ~26) and diffuse white -> 1000 nits
            # (wants ~203). Every HLG deliverable was roughly 5x over.
            #
            # Run the tonescale with diffuse white at 1.0 and its asymptote at
            # 1/0.26, then scale by 0.26 before the OETF (see step 9).
            peak_nits = 100.0 / _HLG_DIFFUSE_WHITE_SCENE
        else:
            peak_nits = peak_luminance

        # 6. Apply DRT tonescale
        # v2.1 FIX: Pass is_hdr flag so tonescale preserves HDR headroom
        tonemapped = self._apply_tonescale_drt(
            img, peak_luminance=peak_nits, surround=surround, is_hdr=is_hdr,
            grey_anchor=not is_hlg,
        )

        # 7. Convert to output color space
        if is_rec2020:
            output = tonemapped @ self.ACES_AP1_TO_Rec2020.T
        elif is_p3:
            output = tonemapped @ self.ACES_AP1_TO_P3D65.T
        else:  # sRGB/Rec.709
            output = tonemapped @ self.ACES_AP1_TO_sRGB.T

        # 8. Clamp negatives from gamut conversion
        output = np.maximum(output, 0)

        # 9. Apply EOTF encoding
        if is_pq:
            output = self._pq_encode(output, peak_nits)
        elif is_hlg:
            # Place diffuse white at the BT.2100 reference point before the OETF.
            output = self._hlg_encode(output * _HLG_DIFFUSE_WHITE_SCENE)
        elif is_cinema:
            # DCI gamma 2.6
            output = np.power(np.clip(output, 0, 1), 1 / 2.6)
        else:
            # sRGB gamma
            output = linear_to_srgb(np.clip(output, 0, 1))

        output = np.clip(output, 0, 1)

        info = f"ACES 2.0 | {input_colorspace} → {output_transform}"
        if is_cinema:
            # 48 nits is the projector luminance of code value 1.0; the maths
            # above runs SDR-normalised.
            info += " | Peak: 48 nits (DCI white)"
        elif is_hlg:
            # HLG is relative — the display peak is a property of the monitor,
            # not of this encode. What this transform fixes is where diffuse
            # white lands relative to the OETF.
            info += " | HLG relative (diffuse white at the BT.2100 reference)"
        elif is_hdr:
            info += f" | Peak: {peak_nits} nits"
        info += f" | Surround: {surround}"

        return (numpy_to_tensor_float32(output), info)
