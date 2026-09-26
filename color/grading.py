import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple

from radiance.color.gamut import aces2_gamut_compress as _aces2_gamut_compress
from radiance.color.luts import (
    _LUT_FUNCTIONS,
    _lut_false_color,
    _lut_clip_check,
    _M_SRGB_TO_ACESCG,
    _M_ACESCG_TO_LIN_SRGB
)

logger = logging.getLogger("radiance.color.grading")


def _kelvin_to_rgb_multipliers(kelvin: float) -> Tuple[float, float, float]:
    """
    Convert color temperature (K) to RGB multipliers.
    Based on Tanner Helland's algorithm, normalized so 6500K = (1,1,1).
    """
    temp = max(1000.0, min(40000.0, kelvin)) / 100.0

    # Red
    if temp <= 66.0:
        r = 255.0
    else:
        r = 329.698727446 * ((temp - 60.0) ** -0.1332047592)
        r = max(0.0, min(255.0, r))

    # Green
    if temp <= 66.0:
        g = 99.4708025861 * np.log(max(temp, 1.0)) - 161.1195681661
    else:
        g = 288.1221695283 * ((temp - 60.0) ** -0.0755148492)
    g = max(0.0, min(255.0, g))

    # Blue
    if temp >= 66.0:
        b = 255.0
    elif temp <= 19.0:
        b = 0.0
    else:
        b = 138.5177312231 * np.log(max(temp - 10.0, 1.0)) - 305.0447927307
        b = max(0.0, min(255.0, b))

    # Normalize to 6500K baseline
    ref_temp = 65.0  # 6500K / 100
    r_ref = 255.0  # At 6500K, temp<=66 so r=255
    g_ref = 99.4708025861 * np.log(ref_temp) - 161.1195681661
    b_ref = 138.5177312231 * np.log(ref_temp - 10.0) - 305.0447927307

    r_ref = max(r_ref, 1.0)
    g_ref = max(g_ref, 1.0)
    b_ref = max(b_ref, 1.0)

    return (r / r_ref, g / g_ref, b / b_ref)


def apply_lut(img: np.ndarray, lut_name: str, intensity: float = 1.0) -> np.ndarray:
    """
    Apply a named LUT to a float32 image.

    Args:
        img: float32 numpy array (H,W,C) or (H,W)
        lut_name: Key from LUT_MODES
        intensity: Blend factor 0.0 (bypass) to 1.0 (full)

    Returns:
        float32 numpy array, same shape (except False Color always returns H,W,3)
    """
    if lut_name == "None" or intensity <= 0.0:
        return img

    handler = _LUT_FUNCTIONS.get(lut_name)
    if handler is None:
        return img

    # False Color / Clip Check — special cases that replace the whole image
    if handler in ("false_color", "clip_check"):
        if handler == "false_color":
            fc = _lut_false_color(img)
        else:
            fc = _lut_clip_check(img)
        if intensity >= 1.0:
            return fc
        # Blend: need to match shapes
        if img.ndim == 2:
            orig = np.stack([img, img, img], axis=-1)
        elif img.ndim == 3 and img.shape[2] == 1:
            orig = np.concatenate([img, img, img], axis=-1)
        elif img.ndim == 3 and img.shape[2] >= 3:
            orig = img[..., :3]
        else:
            orig = img
        return (orig * (1.0 - intensity) + fc * intensity).astype(np.float32)

    # Standard per-channel LUT
    lut_applied = handler(img).astype(np.float32)

    if intensity >= 1.0:
        return lut_applied

    return (img * (1.0 - intensity) + lut_applied * intensity).astype(np.float32)


def apply_grading(
    img: np.ndarray,
    # Basic controls
    exposure: float = 0.0,
    gamma: float = 1.0,  # scalar gamma (used when gamma_rgb is None)
    gain: float = 1.0,
    lift: float = 0.0,
    saturation: float = 1.0,
    temperature: float = 6500.0,
    # Viewer-matched white balance.
    #
    # The WebGL viewer's TEMP/TINT sliders are a plain additive shift in [-2, 2]
    # (`shift.r += temp; shift.b -= temp; shift.g -= tint`), while `temperature`
    # above is a Kelvin white-balance multiply. The delivery handler used to
    # fabricate Kelvin from the slider (6500 + t*3500) and pass that, so the
    # export applied a completely different curve from the one on screen -- and
    # `tint` had nowhere to go at all, so the green/magenta axis was dropped
    # from every master. Pass these instead when matching the viewer; leave
    # `temperature` at 6500 in that case.
    temp_shift: float = 0.0,
    tint_shift: float = 0.0,
    # Extended Resolve-style controls (v3.2 sync)
    offset: float = 0.0,  # global additive offset (applied first)
    contrast: float = 1.0,  # contrast multiplier (pivoted)
    pivot: float = 0.18,  # contrast pivot point
    shadows: float = 0.0,  # shadow lift/crush (-1..1)
    highlights: float = 0.0,  # highlight expand/compress (-1..1)
    hue_shift: float = 0.0,  # degrees (-180..180)
    luma_mix: float = 1.0,   # Preserve original luminance (0.0 = full preservation)
    # Per-channel overrides (v3.5+) — list/tuple of [R, G, B] floats.
    gamma_rgb: Optional[List[float]] = None,
    gain_rgb: Optional[List[float]] = None,
    lift_rgb: Optional[List[float]] = None,
    offset_rgb: Optional[List[float]] = None,
    # LUT
    lut_name: str = "None",
    lut_intensity: float = 1.0,
    # Color Science
    color_science: int = 0, # 0 = Linear/sRGB, 1 = ACEScct
    # Gamut Compression
    gamut_compression: bool = False,
) -> np.ndarray:
    """
    Apply full grading stack to a float32 image.
    Pipeline order is IDENTICAL to the GLSL composite shader so that the
    passthrough IMAGE output matches what the WebGL viewer displays.
    """
    out = img.astype(np.float32, copy=True)

    # Alpha is not a colour. Exposure, contrast and shadows/highlights below all
    # used to operate on the whole array, so a 50% matte came back at 1.059 with
    # exposure +1 and contrast 1.3 -- an over-1.0 "opacity" that clips to fully
    # opaque on write. Lift/gain/gamma, saturation, hue and gamut compression
    # were already [..., :3]-guarded; these three were not.
    _has_alpha = out.ndim == 3 and out.shape[2] >= 4
    
    # Pre-grading luma for luma_mix
    luma_orig = None
    if abs(luma_mix - 1.0) > 0.001:
        luma_orig = 0.2126 * out[..., 0] + 0.7152 * out[..., 1] + 0.0722 * out[..., 2]

    # Fast-path: skip if all controls are at identity
    def _arr_at_identity(arr, identity_val: float) -> bool:
        """True if arr is None/empty or all values equal identity_val within tolerance."""
        if not arr or len(arr) < 3:
            return True
        return all(abs(float(v) - identity_val) < 0.001 for v in arr)

    is_default = (
        abs(exposure) < 0.001
        and abs(gamma - 1.0) < 0.001
        and abs(gain - 1.0) < 0.001
        and abs(lift) < 0.001
        and abs(offset) < 0.001
        and abs(saturation - 1.0) < 0.001
        and abs(contrast - 1.0) < 0.001
        and abs(shadows) < 0.001
        and abs(highlights) < 0.001
        and abs(hue_shift) < 0.1
        and abs(temperature - 6500.0) < 10.0
        and abs(temp_shift) < 0.001
        and abs(tint_shift) < 0.001
        and lut_name == "None"
        and abs(luma_mix - 1.0) < 0.001
        and _arr_at_identity(gamma_rgb, 1.0)
        and _arr_at_identity(gain_rgb, 1.0)
        and _arr_at_identity(lift_rgb, 0.0)
        and _arr_at_identity(offset_rgb, 0.0)
        and not gamut_compression
    )
    if is_default:
        return out

    # ── 1. Offset (global additive shift — handled inside do_lift_gamma_gain)
    # ── 2. Exposure (Linear part)
    if abs(exposure) > 0.001:
        if _has_alpha:
            out[..., :3] *= np.float32(2.0**exposure)
        else:
            out *= np.float32(2.0**exposure)

    # ── 3. White Balance
    if abs(temperature - 6500.0) > 10.0 and out.ndim == 3 and out.shape[2] >= 3:
        r_mult, g_mult, b_mult = _kelvin_to_rgb_multipliers(temperature)
        out[..., 0] *= np.float32(r_mult)
        out[..., 1] *= np.float32(g_mult)
        out[..., 2] *= np.float32(b_mult)

    # Viewer-matched additive shift. Byte-for-byte the shader's applyTempTint:
    #   shift.r += temp;  shift.b -= temp;  shift.g -= tint
    # RGB only -- alpha is not a colour and must not be shifted.
    if (abs(temp_shift) > 0.001 or abs(tint_shift) > 0.001) \
            and out.ndim == 3 and out.shape[2] >= 3:
        out[..., 0] += np.float32(temp_shift)
        out[..., 1] -= np.float32(tint_shift)
        out[..., 2] -= np.float32(temp_shift)
        np.clip(out[..., :3], 0.0, 65504.0, out=out[..., :3])

    # ── 4. Lift / Gain / Gamma (Resolve-style, per-channel aware)
    def do_lift_gamma_gain(color: np.ndarray) -> np.ndarray:
        c = color.copy()
        nc = c.shape[2] if c.ndim == 3 else 0
        has_per_ch = nc >= 3

        _o = np.array(offset_rgb, dtype=np.float32) if (offset_rgb and len(offset_rgb) == 3) else np.array([offset, offset, offset], dtype=np.float32)
        _l = np.array(lift_rgb,   dtype=np.float32) if (lift_rgb   and len(lift_rgb)   == 3) else np.array([lift,   lift,   lift  ], dtype=np.float32)
        _ga = np.array(gain_rgb,  dtype=np.float32) if (gain_rgb   and len(gain_rgb)   == 3) else np.array([gain,   gain,   gain  ], dtype=np.float32)
        _gm = np.array(gamma_rgb, dtype=np.float32) if (gamma_rgb  and len(gamma_rgb)  == 3) else np.array([gamma,  gamma,  gamma ], dtype=np.float32)

        # Offset: per-channel additive shift
        if offset_rgb and len(offset_rgb) == 3 and has_per_ch:
            if np.any(np.abs(_o) > 0.001):
                c[..., :3] += _o
        elif abs(offset) > 0.001:
            c += np.float32(offset)

        # Lift: per-channel shadow offset, luma-pivoted
        if np.any(np.abs(_l) > 0.001) and has_per_ch:
            luma = 0.2126 * c[..., 0] + 0.7152 * c[..., 1] + 0.0722 * c[..., 2]
            pivot_lift = np.clip(1.0 - luma, 0.0, 1.0)[..., np.newaxis]
            c[..., :3] += _l * pivot_lift

        # Gain: per-channel multiplicative slope
        if np.any(np.abs(_ga - 1.0) > 0.001) and has_per_ch:
            c[..., :3] *= _ga

        # Gamma: per-channel power curve
        if np.any(np.abs(_gm - 1.0) > 0.001) and has_per_ch:
            for ch in range(3):
                inv_g = np.float32(1.0 / max(float(_gm[ch]), 0.01))
                if abs(float(_gm[ch]) - 1.0) > 0.001:
                    pos = c[..., ch] > 0
                    c[pos, ch] = np.power(c[pos, ch], inv_g)
        return c

    if color_science == 1 and out.ndim == 3 and out.shape[2] >= 3:
        # ACEScct Pipeline
        acescg = np.tensordot(out[..., :3], _M_SRGB_TO_ACESCG, axes=([2], [1]))
        
        cct = np.empty_like(acescg)
        mask = acescg <= 0.0078125
        cct[mask] = 10.5402377416545 * acescg[mask] + 0.0729055341958355
        clamped = np.clip(acescg[~mask], 1e-10, None)
        cct[~mask] = (np.log2(clamped) + 9.72) / 17.52
        
        cct = do_lift_gamma_gain(cct)
        
        acescg_back = np.empty_like(cct)
        mask2 = cct > 0.155251141552511
        acescg_back[mask2] = np.exp2(cct[mask2] * 17.52 - 9.72)
        acescg_back[~mask2] = (cct[~mask2] - 0.0729055341958355) / 10.5402377416545
        
        out[..., :3] = np.tensordot(acescg_back, _M_ACESCG_TO_LIN_SRGB, axes=([2], [1]))
    else:
        out = do_lift_gamma_gain(out)

    # ── 5. Contrast
    if abs(contrast - 1.0) > 0.001:
        if _has_alpha:
            out[..., :3] = (out[..., :3] - np.float32(pivot)) \
                * np.float32(contrast) + np.float32(pivot)
        else:
            out = (out - np.float32(pivot)) * np.float32(contrast) + np.float32(pivot)

    # ── 6. Shadows / Highlights
    if (
        (abs(shadows) > 0.001 or abs(highlights) > 0.001)
        and out.ndim == 3
        and out.shape[2] >= 3
    ):
        luma = 0.2126 * out[..., 0] + 0.7152 * out[..., 1] + 0.0722 * out[..., 2]
        s_weight = np.power(np.clip(1.0 - luma, 0.0, 1.0), 2.0)[..., np.newaxis]
        h_weight = np.power(np.clip(luma, 0.0, 1.0), 2.0)[..., np.newaxis]
        _s = 1.0 + np.float32(shadows) * s_weight * 0.5
        _h = 1.0 + np.float32(highlights) * h_weight * 0.5
        if _has_alpha:
            out[..., :3] *= _s
            out[..., :3] *= _h
            np.maximum(out[..., :3], 0.0, out=out[..., :3])
        else:
            out *= _s
            out *= _h
            out = np.maximum(out, 0.0)

    # ── 6.5 Gamut Compression
    if gamut_compression and out.ndim == 3 and out.shape[2] >= 3:
        if _aces2_gamut_compress is not None:
            out[..., :3] = _aces2_gamut_compress(out[..., :3])
        else:
            logger.warning(
                "[Radiance Viewer] aces2_gamut_compress unavailable — gamut compression skipped."
            )

    # ── 7. Saturation
    if abs(saturation - 1.0) > 0.001 and out.ndim == 3 and out.shape[2] >= 3:
        luma = (0.2126 * out[..., 0] + 0.7152 * out[..., 1] + 0.0722 * out[..., 2])[..., np.newaxis]
        out[..., :3] = luma + np.float32(saturation) * (out[..., :3] - luma)

    # ── 8. Hue Shift
    if abs(hue_shift) > 0.1 and out.ndim == 3 and out.shape[2] >= 3:
        h_shift = hue_shift / 360.0
        rgb = np.maximum(out[..., :3], 0.0)
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        cmax = np.maximum(np.maximum(r, g), b)
        cmin = np.minimum(np.minimum(r, g), b)
        delta = cmax - cmin + 1e-10

        h = (
            np.where(
                cmax == r,
                (g - b) / delta % 6,
                np.where(cmax == g, (b - r) / delta + 2, (r - g) / delta + 4),
            )
            / 6.0
        )
        s = np.where(cmax > 1e-10, delta / cmax, 0.0)
        v = cmax

        h = (h + h_shift) % 1.0

        i = (h * 6).astype(int)
        f = h * 6 - i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)

        i6 = i % 6
        r_out = np.select(
            [i6 == 0, i6 == 1, i6 == 2, i6 == 3, i6 == 4, i6 == 5], [v, q, p, p, t, v]
        )
        g_out = np.select(
            [i6 == 0, i6 == 1, i6 == 2, i6 == 3, i6 == 4, i6 == 5], [t, v, v, q, p, p]
        )
        b_out = np.select(
            [i6 == 0, i6 == 1, i6 == 2, i6 == 3, i6 == 4, i6 == 5], [p, p, t, v, v, q]
        )

        out[..., 0] = r_out
        out[..., 1] = g_out
        out[..., 2] = b_out

    # ── 8.5 Luma Mix
    if luma_orig is not None:
        luma_curr = 0.2126 * out[..., 0] + 0.7152 * out[..., 1] + 0.0722 * out[..., 2]
        color_with_orig_luma = out * (luma_orig / (luma_curr + 1e-10))[..., np.newaxis]
        out = luma_mix * out + (1.0 - luma_mix) * color_with_orig_luma

    # ── 9. LUT (last in chain)
    if lut_name != "None" and lut_intensity > 0.0:
        out = apply_lut(out, lut_name, lut_intensity)

    return out
