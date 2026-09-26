from __future__ import annotations

import torch
import logging
import json
import math
from typing import Tuple

from radiance.radiance_ocio import get_ocio_manager

# Shared torch-math primitives — single source of truth for matrix ops.
# Replaces the fragile try/except private-API import from nodes_hdr_colorspace.
from radiance.color.ops import apply_matrix_3x3, M_REC709_TO_ACESCG, M_ACESCG_TO_REC709

logger = logging.getLogger("radiance.colorscience")

_BRADFORD = torch.tensor([
    [ 0.8951,  0.2664, -0.1614],
    [-0.7502,  1.7135,  0.0367],
    [ 0.0389, -0.0685,  1.0296],
], dtype=torch.float64)

_BRADFORD_INV = torch.inverse(_BRADFORD)

_ILLUMINANT_XY = {
    "D50": (0.3457, 0.3585), "D55": (0.3324, 0.3474), "D60": (0.3217, 0.3377),
    "D65": (0.3127, 0.3290), "D75": (0.2990, 0.3149), "A": (0.4476, 0.4074),
    "B": (0.3484, 0.3516), "C": (0.3101, 0.3162), "E": (0.3333, 0.3333),
}


def _xy_to_XYZ(xy: Tuple[float, float]) -> torch.Tensor:
    x, y = xy
    return torch.tensor([x / y, 1.0, (1.0 - x - y) / y], dtype=torch.float64)


def _build_bradford_matrix(src_illuminant: str, dst_illuminant: str) -> torch.Tensor:
    src_XYZ = _xy_to_XYZ(_ILLUMINANT_XY.get(src_illuminant, _ILLUMINANT_XY["D65"]))
    dst_XYZ = _xy_to_XYZ(_ILLUMINANT_XY.get(dst_illuminant, _ILLUMINANT_XY["D65"]))
    src_cone = _BRADFORD @ src_XYZ
    dst_cone = _BRADFORD @ dst_XYZ
    scale = dst_cone / src_cone.clamp(min=1e-9)
    M = _BRADFORD_INV @ torch.diag(scale) @ _BRADFORD
    return M.float()


def _temperature_to_xy(kelvin: float) -> Tuple[float, float]:
    T = max(1667.0, min(kelvin, 25000.0))
    if T <= 4000:
        x = (-0.2661239e9 / T**3 - 0.2343580e6 / T**2 + 0.8776956e3 / T + 0.179910)
    else:
        x = (-3.0258469e9 / T**3 + 2.1070379e6 / T**2 + 0.2226347e3 / T + 0.240390)
    if T <= 2222:
        y = (-1.1063814 * x**3 - 1.34811020 * x**2 + 2.18555832 * x - 0.20219683)
    elif T <= 4000:
        y = (-0.9549476 * x**3 - 1.37418593 * x**2 + 2.09137015 * x - 0.16748867)
    else:
        y = (3.0817580 * x**3 - 5.87338670 * x**2 + 3.75112997 * x - 0.37001483)
    return (x, y)


class RadianceWhiteBalance:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Color"
    DESCRIPTION = (
        "Shift white balance of a linear image by colour temperature and tint, a Bradford illuminant adaptation, "
        "or per-channel RGB gains. Use it on scene-linear plates; it does not sample a neutral from the image."
    )
    FUNCTION = "apply"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "grade_info")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "RGB image to rebalance, scene-linear (Rec.709 primaries). Apply to linear data, not display-encoded sRGB."}),
                "mode": (["Temperature / Tint", "Illuminant Adapt", "Manual RGB Gain"], {"default": "Temperature / Tint", "tooltip": "Temperature / Tint: adapt from D65 to a Kelvin white plus a green/magenta tint. Illuminant Adapt: Bradford from src to dst illuminant. Manual RGB Gain: multiply channels."}),
                "preset": (["Manual", "Daylight (5500K)", "Tungsten (3200K)", "Fluorescent (4200K)", "Flash (6000K)", "Shade (7500K)"], {"default": "Manual", "tooltip": "Replaces temperature with a preset Kelvin value. Only used in Temperature / Tint mode; Manual uses the temperature slider."}),
                "temperature": ("FLOAT", {"default": 6500.0, "min": 1667.0, "max": 25000.0, "step": 50.0, "tooltip": "Target white in Kelvin (Planckian locus), adapted from D65. Lower values warm the image, higher values cool it; 6500 is close to neutral. Temperature / Tint mode only."}),
                "tint": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.005, "tooltip": "Green channel gain of 2^(-tint/2): positive adds magenta, negative adds green, +/-1 = half a stop. Temperature / Tint mode only."}),
                "src_illuminant": (list(_ILLUMINANT_XY.keys()), {"default": "D65", "tooltip": "White point the image was balanced for (CIE standard illuminant). Illuminant Adapt mode only."}),
                "dst_illuminant": (list(_ILLUMINANT_XY.keys()), {"default": "D50", "tooltip": "White point to adapt to with the Bradford transform. Illuminant Adapt mode only."}),
                "gain_r": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.001, "tooltip": "Linear red multiplier, 1 = unchanged. Manual RGB Gain mode only."}),
                "gain_g": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.001, "tooltip": "Linear green multiplier, 1 = unchanged. Manual RGB Gain mode only."}),
                "gain_b": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.001, "tooltip": "Linear blue multiplier, 1 = unchanged. Manual RGB Gain mode only."}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Linear mix between the original (0) and the corrected image (1)."}),
            },
            "optional": {
                "grade_info_in": ("STRING", {"forceInput": True, "tooltip": "Optional grade_info JSON from an upstream node. Only recorded as \"upstream\" in this node's info output; it does not change the image."}),
                "ocio_context": ("RADIANCE_OCIO", {"tooltip": "From OCIO Context. Makes sure that config is loaded (and loaded first) before this node converts."}),
            },
        }

    def apply(self, image: torch.Tensor, mode: str, preset: str = "Manual",
              temperature: float = 6500.0, tint: float = 0.0,
              src_illuminant: str = "D65", dst_illuminant: str = "D50",
              gain_r: float = 1.0, gain_g: float = 1.0, gain_b: float = 1.0,
              strength: float = 1.0, grade_info_in: str = None):
        img = image.clone()
        device = img.device

        if preset != "Manual" and mode == "Temperature / Tint":
            preset_map = {
                "Daylight (5500K)": 5500.0, "Tungsten (3200K)": 3200.0,
                "Fluorescent (4200K)": 4200.0, "Flash (6000K)": 6000.0, "Shade (7500K)": 7500.0,
            }
            temperature = preset_map.get(preset, temperature)

        if mode == "Temperature / Tint":
            target_xy = _temperature_to_xy(temperature)
            src_xy = _ILLUMINANT_XY["D65"]
            src_XYZ = _xy_to_XYZ(src_xy)
            dst_XYZ = _xy_to_XYZ(target_xy)
            src_cone = _BRADFORD @ src_XYZ
            dst_cone = _BRADFORD @ dst_XYZ
            scale = dst_cone / src_cone.clamp(min=1e-9)
            M = (_BRADFORD_INV @ torch.diag(scale) @ _BRADFORD).float().to(device)
            tint_gain = torch.tensor([1.0, math.pow(2.0, -tint * 0.5), 1.0], dtype=torch.float32, device=device)
            result = torch.einsum('ij,...j->...i', M, img)
            result = result * tint_gain
        elif mode == "Illuminant Adapt":
            M = _build_bradford_matrix(src_illuminant, dst_illuminant).to(device)
            result = torch.einsum('ij,...j->...i', M, img)
        else:
            gains = torch.tensor([gain_r, gain_g, gain_b], dtype=torch.float32, device=device)
            result = img * gains

        if strength < 1.0:
            result = torch.lerp(image, result, strength)

        grade_info = json.dumps({
            "node": "RadianceWhiteBalance", "mode": mode, "preset": preset,
            "temperature": temperature, "tint": tint,
            "src_illuminant": src_illuminant, "dst_illuminant": dst_illuminant,
            "gain_rgb": [gain_r, gain_g, gain_b], "strength": strength,
            "upstream": grade_info_in,
        })
        return (result, grade_info)


class RadianceColorSpaceConvert:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Color"
    DESCRIPTION = (
        "Convert pixel values between working, display and camera-log colour spaces via a Rec.709-primaries linear "
        "intermediate (OCIO for the non-log spaces when a config is loaded). Camera logs are transfer curves only, no gamut change."
    )
    FUNCTION = "apply"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "grade_info")

    _COLOR_SPACES = [
        "Linear sRGB (D65)", "ACEScg", "ACEScc", "ACEScct",
        "sRGB (OETF encoded)", "Rec.709 (OETF encoded)", "Rec.709 / BT.1886",
        "LogC3 (ARRI EI800)", "LogC4 (ARRI Alexa 35)", "F-Log2 (Fujifilm)",
        "C-Log3 (Canon)", "Log3G10 (RED IPP2)", "DaVinci Intermediate",
        "BMD Film Gen5", "V-Log (Panasonic)", "N-Log (Nikon)",
    ]

    # Reference the module-level imports — avoids a second tensor allocation and
    # keeps precision consistent with color/matrices.py (6 d.p. vs 4 d.p. here).
    _M_709_TO_ACESCG = M_REC709_TO_ACESCG
    _M_ACESCG_TO_709 = M_ACESCG_TO_REC709

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image encoded in src_space (or in dst_space when direction is Inverse)."}),
                "src_space": (cls._COLOR_SPACES, {"default": "Linear sRGB (D65)", "tooltip": "Encoding of the input. Camera-log choices decode the log curve only and keep Rec.709 primaries; ACES choices also convert AP1 primaries."}),
                "dst_space": (cls._COLOR_SPACES, {"default": "ACEScg", "tooltip": "Encoding to convert to. Camera-log choices apply the log curve only, on Rec.709 primaries."}),
                "direction": (["Forward", "Inverse"], {"default": "Forward", "tooltip": "Forward converts src_space to dst_space. Inverse swaps them (dst_space to src_space)."}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Mix between input and converted pixel values (0 = input, 1 = full conversion). The mix is done on the raw values of two different encodings."}),
            },
            "optional": {
                "grade_info_in": ("STRING", {"forceInput": True, "tooltip": "Optional grade_info JSON from an upstream node. Only recorded as \"upstream\" in this node's info output; it does not change the image."}),
            },
        }

    # ── Analytical curve table ────────────────────────────────────────────
    #
    # AUDIT-FIX (2026-08): this node used to carry its own inline LogC3 that
    # disagreed with the ARRI spec (18% grey encoded to 0.417 instead of
    # 0.391, black to 0.271 instead of 0.093) AND with its own decode -- 0.18
    # round-tripped through encode+decode to 0.060, a ~1.5-stop error. Worse,
    # ten of the sixteen advertised spaces (ACEScc, ACEScct, LogC4, F-Log2,
    # C-Log3, Log3G10, DaVinci Intermediate, BMD Gen5, V-Log, N-Log) had no
    # analytical implementation at all: whenever OCIO was not configured --
    # the default install -- the "conversion" silently returned the input
    # unchanged. All curves now come from radiance.color.transfer and
    # radiance.color.luts, the audited single source of truth, and an
    # unsupported space raises instead of passing pixels through untouched.
    #
    # Convention: camera-log spaces are treated as transfer curves relative to
    # the Rec.709-primaries working linear (matching color/luts.py). ACEScc,
    # ACEScct and ACEScg additionally apply the Rec.709↔AP1 gamut matrix.

    @staticmethod
    def _via_numpy(fn, img: torch.Tensor) -> torch.Tensor:
        arr = fn(img.detach().cpu().numpy().astype("float32"))
        import numpy as _np
        return torch.from_numpy(_np.ascontiguousarray(arr)).to(img.device, img.dtype)

    def _to_acescg(self, img: torch.Tensor) -> torch.Tensor:
        return torch.einsum('ij,...j->...i', self._M_709_TO_ACESCG.to(img.device), img)

    def _from_acescg(self, img: torch.Tensor) -> torch.Tensor:
        return torch.einsum('ij,...j->...i', self._M_ACESCG_TO_709.to(img.device), img)

    def _encode_from_linear(self, img: torch.Tensor, space: str) -> torch.Tensor:
        """Working linear (Rec.709 primaries, D65) → `space`."""
        from radiance.color import transfer as _tf
        from radiance.color import luts as _luts
        if space == "Linear sRGB (D65)":
            return img
        if space == "sRGB (OETF encoded)":
            return _tf.tensor_linear_to_srgb(img)
        if space == "Rec.709 (OETF encoded)":
            # True BT.709 camera OETF -- previously aliased to the sRGB curve.
            abs_t, sign = img.abs(), img.sign()
            high = 1.099 * torch.pow(abs_t.clamp(min=1e-10), 0.45) - 0.099
            return torch.where(abs_t < 0.018, abs_t * 4.5, high) * sign
        if space == "Rec.709 / BT.1886":
            return img.clamp(min=0.0) ** (1.0 / 2.4)
        if space == "ACEScg":
            return self._to_acescg(img)
        if space == "ACEScc":
            return _tf.tensor_linear_to_acescc(self._to_acescg(img))
        if space == "ACEScct":
            return _tf.tensor_linear_to_acescct(self._to_acescg(img))
        if space == "LogC3 (ARRI EI800)":
            return _tf.tensor_linear_to_logc3(img)
        if space == "LogC4 (ARRI Alexa 35)":
            return _tf.tensor_linear_to_logc4(img)
        if space == "C-Log3 (Canon)":
            return _tf.tensor_linear_to_canonlog3(img)
        if space == "Log3G10 (RED IPP2)":
            return _tf.tensor_linear_to_log3g10(img)
        if space == "DaVinci Intermediate":
            return _tf.tensor_linear_to_davinci_intermediate(img)
        if space == "V-Log (Panasonic)":
            return _tf.tensor_linear_to_vlog(img)
        if space == "F-Log2 (Fujifilm)":
            return self._via_numpy(_luts._lut_flog2, img)
        if space == "BMD Film Gen5":
            return self._via_numpy(_luts._lut_bmd_gen5, img)
        if space == "N-Log (Nikon)":
            return self._via_numpy(_luts._lut_nlog, img)
        raise ValueError(
            f"RadianceColorSpaceConvert: no analytical transform for "
            f"{space!r} and no OCIO config is loaded. Refusing to pass "
            f"pixels through unchanged.")

    def _decode_to_linear(self, img: torch.Tensor, space: str) -> torch.Tensor:
        """`space` → working linear (Rec.709 primaries, D65)."""
        from radiance.color import transfer as _tf
        from radiance.color import luts as _luts
        if space == "Linear sRGB (D65)":
            return img
        if space == "sRGB (OETF encoded)":
            return _tf.tensor_srgb_to_linear(img)
        if space == "Rec.709 (OETF encoded)":
            abs_t, sign = img.abs(), img.sign()
            high = torch.pow(((abs_t + 0.099) / 1.099).clamp(min=1e-10), 1.0 / 0.45)
            return torch.where(abs_t < 0.081, abs_t / 4.5, high) * sign
        if space == "Rec.709 / BT.1886":
            return img.clamp(min=0.0) ** 2.4
        if space == "ACEScg":
            return self._from_acescg(img)
        if space == "ACEScc":
            return self._from_acescg(_tf.tensor_acescc_to_linear(img))
        if space == "ACEScct":
            return self._from_acescg(_tf.tensor_acescct_to_linear(img))
        if space == "LogC3 (ARRI EI800)":
            return _tf.tensor_logc3_to_linear(img)
        if space == "LogC4 (ARRI Alexa 35)":
            return _tf.tensor_logc4_to_linear(img)
        if space == "C-Log3 (Canon)":
            return _tf.tensor_canonlog3_to_linear(img)
        if space == "Log3G10 (RED IPP2)":
            return _tf.tensor_log3g10_to_linear(img)
        if space == "DaVinci Intermediate":
            return _tf.tensor_davinci_intermediate_to_linear(img)
        if space == "V-Log (Panasonic)":
            return _tf.tensor_vlog_to_linear(img)
        if space == "F-Log2 (Fujifilm)":
            return self._via_numpy(_luts._idt_flog2, img)
        if space == "BMD Film Gen5":
            return self._via_numpy(_luts._idt_bmd_gen5, img)
        if space == "N-Log (Nikon)":
            return self._via_numpy(_luts._idt_nlog, img)
        raise ValueError(
            f"RadianceColorSpaceConvert: no analytical transform for "
            f"{space!r} and no OCIO config is loaded. Refusing to pass "
            f"pixels through unchanged.")

    def _try_ocio(self, img: torch.Tensor, src: str, dst: str) -> torch.Tensor | None:
        mgr = get_ocio_manager()
        if not mgr.is_loaded:
            return None
        try:
            config = mgr.config

            def _resolve_name(n):
                try:
                    if config.getColorSpace(n):
                        return n
                except Exception as _exc:
                    logger.debug(
                        "[Radiance] _resolve_name(): ignoring %s from `if config.getColorSpace(n):`: %s",
                        type(_exc).__name__, _exc,
                    )
                # Names in the ACES studio config Radiance configures at
                # startup (color/ocio_setup.py). The old targets ("Linear",
                # "Output - Rec.709", ...) existed in no shipped config, so
                # every lookup missed and OCIO never ran.
                _NAME_MAP = {
                    "Linear sRGB (D65)": "Linear Rec.709 (sRGB)", "ACEScg": "ACEScg",
                    "ACEScc": "ACEScc", "ACEScct": "ACEScct",
                    "sRGB (OETF encoded)": "sRGB Encoded Rec.709 (sRGB)",
                    "Rec.709 / BT.1886": "Gamma 2.4 Encoded Rec.709",
                }
                # Camera logs stay analytic on purpose: this node's documented
                # convention is transfer-only against Rec.709 linear, and the
                # OCIO camera spaces carry their native gamut, so OCIO and the
                # fallback would give different pictures for the same setting.
                # (Read / Write do full camera decodes, gamut included.)
                if n not in _NAME_MAP and n not in ("Linear sRGB (D65)",):
                    return None
                return _NAME_MAP.get(n, n)

            ocio_src = _resolve_name(src)
            ocio_dst = _resolve_name(dst)
            if not ocio_src or not ocio_dst:
                return None
            processor = mgr.get_processor(ocio_src, ocio_dst)
            if not processor:
                return None
            cpu = processor.getDefaultCPUProcessor()
            import numpy as np
            arr = img.cpu().numpy().astype("float32")
            B, H, W, _ = arr.shape
            flat = arr.reshape(-1, 3)
            cpu.applyRGB(flat)
            return torch.from_numpy(flat.reshape(B, H, W, 3)).to(img.device)
        except Exception as e:
            logger.debug(f"[CSConvert] OCIO failed ({e}), using analytical fallback")
            return None

    def apply(self, image: torch.Tensor, src_space: str, dst_space: str,
              direction: str = "Forward", strength: float = 1.0, grade_info_in: str = None,
              ocio_context=None):
        if ocio_context is not None:
            from radiance.nodes.color.ocio import ensure_context_loaded
            ensure_context_loaded(ocio_context)
        if src_space == dst_space:
            return (image, json.dumps({"node": "RadianceColorSpaceConvert", "src": src_space,
                                       "dst": dst_space, "direction": direction, "noop": True,
                                       "upstream": grade_info_in}))

        effective_src = src_space if direction == "Forward" else dst_space
        effective_dst = dst_space if direction == "Forward" else src_space

        result = self._try_ocio(image, effective_src, effective_dst)

        if result is None:
            linear = self._decode_to_linear(image, effective_src)
            result = self._encode_from_linear(linear, effective_dst)

        if strength < 1.0:
            result = torch.lerp(image, result, strength)

        grade_info = json.dumps({
            "node": "RadianceColorSpaceConvert", "src": src_space, "dst": dst_space,
            "direction": direction, "strength": strength, "upstream": grade_info_in,
        })
        return (result, grade_info)


# ACEScg → XYZ (D60) imported from color.ops (authoritative source).
from radiance.color.ops import M_ACESCG_TO_XYZ_D60 as _M_ACESCG_TO_XYZ

# The remaining display-referred matrices are small enough to live here —
# they are specific to the ACES ODT logic below and not shared elsewhere.
_M_D60_TO_D65 = torch.tensor([
    [0.9872240, 0.0000000, 0.0128491],
    [0.0000000, 1.0000000, 0.0000000],
    [0.0000000, 0.0000000, 1.0000000],
], dtype=torch.float32)

_M_XYZ_TO_SRGB = torch.tensor([
    [ 3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [ 0.0556434, -0.2040259,  1.0572252],
], dtype=torch.float32)

_M_XYZ_TO_P3 = torch.tensor([
    [ 2.4934969, -0.9313836, -0.4027108],
    [-0.8294890,  1.7626641,  0.0236247],
    [ 0.0358458, -0.0761724,  0.9568845],
], dtype=torch.float32)

_M_XYZ_TO_REC2020 = torch.tensor([
    [ 1.7166512, -0.3556708, -0.2533663],
    [-0.6666844,  1.6164812,  0.0157685],
    [ 0.0176399, -0.0427706,  0.9421031],
], dtype=torch.float32)


class RadianceACESTransform:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Color"
    DESCRIPTION = (
        "Approximate ACES-style output transform: a fitted filmic tone curve (Narkowicz ACES fit, not the full RRT) "
        "on ACEScg, then conversion and encoding for an sRGB, P3, PQ or HLG display. Use for quick previews, not deliverables."
    )
    FUNCTION = "apply"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "aces_info")

    _ODT_OPTIONS = ["sRGB D65", "DCI-P3 D65", "Rec.2020 PQ (HDR10)", "Rec.2020 HLG"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Scene-linear ACEScg image."}),
                "odt": (cls._ODT_OPTIONS, {"default": "sRGB D65", "tooltip": "Target display. sRGB: sRGB curve. DCI-P3 D65: P3 primaries, 2.6 gamma. Rec.2020 PQ: ST 2084 signal. Rec.2020 HLG: HLG signal, tone-mapped white at signal 1.0."}),
                "exposure_offset": ("FLOAT", {"default": 0.0, "min": -4.0, "max": 4.0, "step": 0.1, "tooltip": "Exposure change in stops applied to the scene-linear input before the tone curve."}),
                "peak_nits": ("FLOAT", {"default": 1000.0, "min": 100.0, "max": 10000.0, "step": 100.0, "tooltip": "Luminance in nits that tone-mapped white (1.0) is encoded at. Only used by Rec.2020 PQ (HDR10); ignored by the other outputs."}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.5, "step": 0.02, "tooltip": "Saturation after the tone curve, around Rec.709-weighted luma: 0 = greyscale, 1 = unchanged. Result is clamped to 0..1."}),
            },
            "optional": {
                "grade_info_in": ("STRING", {"forceInput": True, "tooltip": "Optional grade_info JSON from an upstream node. Only recorded as \"upstream\" in this node's info output; it does not change the image."}),
            },
        }

    @staticmethod
    def _rrt(x: torch.Tensor) -> torch.Tensor:
        a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
        return torch.clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)

    @staticmethod
    def _oetf_srgb(x: torch.Tensor) -> torch.Tensor:
        return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x.clamp(min=0.0) ** (1.0 / 2.4) - 0.055).clamp(0, 1)

    @staticmethod
    def _oetf_pq(linear: torch.Tensor, peak_nits: float) -> torch.Tensor:
        """Scene-linear (in nits) → PQ signal. Note: input is already in nits here."""
        # Import shared constants rather than redefining them.
        from radiance.color.ops import PQ_M1, PQ_M2, PQ_C1, PQ_C2, PQ_C3
        lum = linear.clamp(min=0.0)
        Ym = (lum * peak_nits / 10000.0).clamp(min=0.0) ** PQ_M1
        return ((PQ_C1 + PQ_C2 * Ym) / (1.0 + PQ_C3 * Ym)) ** PQ_M2

    @staticmethod
    def _oetf_hlg(linear: torch.Tensor) -> torch.Tensor:
        """Scene-linear → HLG signal. Delegates to the canonical implementation."""
        from radiance.color.ops import linear_to_hlg
        return linear_to_hlg(linear)

    def apply(self, image: torch.Tensor, odt: str = "sRGB D65", exposure_offset: float = 0.0,
              peak_nits: float = 1000.0, saturation: float = 1.0, grade_info_in: str = None) -> Tuple[torch.Tensor, str]:
        device = image.device
        img = image.float().clone()

        if exposure_offset != 0.0:
            img = img * (2.0 ** exposure_offset)

        img = self._rrt(img)

        if saturation != 1.0 and img.shape[-1] >= 3:
            luma = (0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]).unsqueeze(-1)
            img = luma + saturation * (img - luma)
            img = img.clamp(0, 1)

        M_to_xyz = _M_ACESCG_TO_XYZ.to(device)
        M_d60_d65 = _M_D60_TO_D65.to(device)
        xyz_d65 = torch.einsum('ij,...j->...i', M_d60_d65 @ M_to_xyz, img)

        if odt == "sRGB D65":
            M_out = _M_XYZ_TO_SRGB.to(device)
            rgb = torch.einsum('ij,...j->...i', M_out, xyz_d65).clamp(0, 1)
            result = self._oetf_srgb(rgb)
        elif odt == "DCI-P3 D65":
            M_out = _M_XYZ_TO_P3.to(device)
            rgb = torch.einsum('ij,...j->...i', M_out, xyz_d65).clamp(0, 1)
            result = rgb ** (1.0 / 2.6)
        elif odt == "Rec.2020 PQ (HDR10)":
            M_out = _M_XYZ_TO_REC2020.to(device)
            rgb = torch.einsum('ij,...j->...i', M_out, xyz_d65).clamp(0, 1)
            result = self._oetf_pq(rgb, peak_nits)
        else:
            M_out = _M_XYZ_TO_REC2020.to(device)
            rgb = torch.einsum('ij,...j->...i', M_out, xyz_d65).clamp(0, 1)
            result = self._oetf_hlg(rgb)

        aces_info = json.dumps({
            "node": "RadianceACESTransform", "odt": odt,
            "exposure_offset_stops": exposure_offset, "peak_nits": peak_nits,
            "saturation": saturation, "upstream": grade_info_in,
        }, indent=2)
        return (result, aces_info)


class RadianceBitDepthDegrade:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Color"
    DESCRIPTION = (
        "Quantise an image to a lower bit depth, with optional dither, and output the error and a threshold mask. "
        "Use it to preview banding in 8/10-bit deliveries; feed the display-encoded image as it will be delivered."
    )
    FUNCTION = "degrade"
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("quantized", "delta_amplified", "banding_mask", "metrics")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Display-encoded 0..1 image, quantised as is (no transfer conversion); values outside 0..1 are clipped. Alpha passes through unchanged."}),
                "bit_depth": ("INT", {"default": 8, "min": 4, "max": 16, "step": 1, "display": "slider", "tooltip": "Target bits per channel; the 0..1 range is split into 2^bits - 1 steps."}),
                "dither_mode": (["none", "triangular", "floyd-steinberg"], {"default": "triangular", "tooltip": "none: plain rounding. triangular: random TPDF noise of +/-1 step before rounding. floyd-steinberg: error diffusion (the classic scan-line result, computed a diagonal at a time)."}),
            },
            "optional": {
                "delta_gain": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 100.0, "step": 0.5, "tooltip": "Multiplier on the absolute error |original - quantised| for the delta_amplified output, clipped to 1."}),
                "banding_threshold": ("FLOAT", {"default": 0.004, "min": 0.0005, "max": 0.05, "step": 0.0005, "tooltip": "Per-pixel error (0..1 units, largest channel) above which banding_mask is white. 0.004 is about one 8-bit code value. This flags quantisation error, not detected bands."}),
                "restore_from_quantized": ("BOOLEAN", {"default": False, "tooltip": "Currently has no effect: the node ignores this setting."}),
            },
        }

    @staticmethod
    def _tpdf_dither(img: torch.Tensor, levels: int) -> torch.Tensor:
        lsb = 1.0 / levels
        noise = (torch.rand_like(img) + torch.rand_like(img) - 1.0) * lsb
        return img + noise

    @staticmethod
    def _floyd_steinberg(img: torch.Tensor, levels: int) -> torch.Tensor:
        """Floyd-Steinberg error diffusion, bit-identical to the scan-line loop.

        3.5.0: the scan-line loop visited every pixel of every channel of every
        frame in Python: 8 s per 1024x576 frame, about 2 minutes per 4K frame.
        A pixel at (y, x) only depends on pixels at x' + 2y' < x + 2y, so every
        pixel on one anti-diagonal x + 2y = k is processed together, and so are
        all frames and channels. That is W + 2H steps instead of B*C*H*W. The
        arithmetic is the loop's, in float32, and each pixel receives its
        neighbours' error in the loop's order (1/16, 5/16, 3/16, then 7/16), so
        the result is the same to the bit.
        """
        import numpy as np
        step = 1.0 / (levels - 1)
        B, H, W, C = img.shape
        # (H, W, B*C): one diagonal's pixels gather into an (m, B*C) block.
        p = np.array(img.detach().cpu().float().numpy().transpose(1, 2, 0, 3).reshape(H, W, B * C), copy=True, order="C")
        ys_all, xs_all = np.mgrid[0:H, 0:W]
        key = (xs_all + 2 * ys_all).ravel()
        order = np.argsort(key, kind="stable")
        bounds = np.searchsorted(key[order], np.arange(W + 2 * H), side="left")
        flat_y = ys_all.ravel()[order]
        flat_x = xs_all.ravel()[order]
        # NumPy 1 promotes scalar float32 / Python float to float64; NumPy 2
        # retains float32. Match the scan-line arithmetic on either version.
        calc_dtype = np.asarray(np.float32(0) / step).dtype
        f7, f3, f5, f1 = (calc_dtype.type(v) for v in (7, 3, 5, 1))
        f16 = calc_dtype.type(16)
        for k in range(len(bounds) - 1):
            a, b = bounds[k], bounds[k + 1]
            if a == b:
                continue
            y = flat_y[a:b]
            x = flat_x[a:b]
            old = p[y, x].astype(calc_dtype)
            new = np.round(old / step) * step
            err = old - new
            p[y, x] = new
            # The loop's order into any one pixel: from the row above left to
            # right (1/16, 5/16, 3/16), then from the left (7/16). Steps run in
            # key order, and within this step down-left (3/16) goes before
            # right (7/16), which is the only pair that shares a step.
            dn = y + 1 < H
            sel = dn & (x > 0)                         # down-left, 3/16
            if sel.any():
                p[y[sel] + 1, x[sel] - 1] += err[sel] * f3 / f16
            sel = x + 1 < W                            # right, 7/16
            if sel.any():
                p[y[sel], x[sel] + 1] += err[sel] * f7 / f16
            if dn.any():                               # down, 5/16
                p[y[dn] + 1, x[dn]] += err[dn] * f5 / f16
            sel = dn & (x + 1 < W)                     # down-right, 1/16
            if sel.any():
                p[y[sel] + 1, x[sel] + 1] += err[sel] * f1 / f16
        out = p.reshape(H, W, B, C).transpose(2, 0, 1, 3)
        return torch.from_numpy(np.ascontiguousarray(out)).to(img.device)

    @staticmethod
    def _psnr(original: torch.Tensor, quantized: torch.Tensor) -> float:
        mse = ((original - quantized) ** 2).mean().item()
        if mse == 0.0:
            return float("inf")
        return 10.0 * math.log10(1.0 / mse)

    def degrade(self, image: torch.Tensor, bit_depth: int = 8, dither_mode: str = "triangular",
                delta_gain: float = 10.0, banding_threshold: float = 0.004,
                restore_from_quantized: bool = False):
        img = image[..., :3].float()
        levels = (2 ** bit_depth) - 1
        step = 1.0 / levels

        if dither_mode == "triangular":
            dithered = self._tpdf_dither(img, levels)
        elif dither_mode == "floyd-steinberg":
            dithered = self._floyd_steinberg(img, levels)
        else:
            dithered = img.clone()

        quantized = (dithered / step).round() * step
        quantized = quantized.clamp(0.0, 1.0)

        delta = (img - quantized).abs()
        delta_amp = (delta * delta_gain).clamp(0.0, 1.0)
        banding = (delta.max(dim=-1, keepdim=True).values > banding_threshold).float().expand_as(img)

        psnr_val = self._psnr(img, quantized)
        max_err = delta.max().item()
        dr_loss_stops = math.log2(max(levels, 1) / (2 ** 16 - 1) + 1e-9)

        metrics = json.dumps({
            "bit_depth": bit_depth, "levels": levels, "dither_mode": dither_mode,
            "psnr_dB": round(psnr_val, 2), "max_error": round(max_err, 6),
            "dynamic_range_loss_stops": round(abs(dr_loss_stops), 2),
        }, indent=2)

        def _reattach(out3c):
            if image.shape[-1] == 4:
                return torch.cat([out3c, image[..., 3:4]], dim=-1)
            return out3c

        return (_reattach(quantized), _reattach(delta_amp), _reattach(banding), metrics)
