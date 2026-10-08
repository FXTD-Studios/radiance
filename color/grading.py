import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple

from radiance.color.gamut import aces2_gamut_compress as _aces2_gamut_compress
from radiance.color.luts import (
    _LUT_FUNCTIONS,
    _lut_false_color,
    _lut_clip_check,
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


# ── The viewer grade, in Python ──────────────────────────────────────────────
#
# A Deliver master and the Grade Apply node bake the viewer's grade into
# pixels, so this has to be the same maths as js/radiance_grade.js
# gradePixelFull (the CPU form of the viewer's shader), op for op.
# tests/test_viewer_grade_parity.py checks it against values generated from
# that file, so the two cannot drift apart again: they did, and masters
# stopped matching the screen (additive Temperature/Tint, linear contrast,
# Luma Mix restoring the ungraded luminance, tone-mapper ACEScct matrices).

_LUMA = np.array([0.2126, 0.7152, 0.0722])
_PIVOT_DEFAULT = 0.18
_PIVOT_FLOOR = 0.001
_CONTRAST_MAX = 5.0
_CONTRAST_CEILING = 1e30
_GAMMA_FLOOR = 0.01

# Source gamut (the viewer's u_sourceGamut: 0 linear Rec.709, 1 ACEScg,
# 2 ACES2065-1, 3 linear Rec.2020, 4 linear P3-D65) to ACEScg, Bradford, and
# the exact inverses. The same numbers as TO_AP1 / FROM_AP1 in
# js/radiance_grade.js; OpenColorIO agrees to 1e-7.
_TO_AP1 = {
    0: [[0.6130974024011878, 0.3395231461841061, 0.04737945141470665],
        [0.07019372246958168, 0.9163538790573436, 0.01345239847307412],
        [0.02061559288222693, 0.1095697729381354, 0.8698146341796377]],
    2: [[1.451439316145666, -0.2365107468937401, -0.2149285692519255],
        [-0.0765537733960206, 1.176229699833573, -0.09967592643755213],
        [0.008316148425697719, -0.006032449791021028, 0.9977163013653231]],
    3: [[0.9748949779244189, 0.01959910863700534, 0.005505913438576188],
        [0.002179562797703918, 0.9955354688932204, 0.002284968309075179],
        [0.004797239683772727, 0.02453201663458945, 0.9706707436816377]],
    4: [[0.735797914028892, 0.2121664852931746, 0.05203560067793385],
        [0.04717988497673042, 0.9380457009217151, 0.01477441410155395],
        [0.003563664639098911, 0.04114188562513292, 0.955294449735768]],
}
_FROM_AP1 = {
    0: [[1.705050992657983, -0.6217921206570056, -0.08325887200097853],
        [-0.1302564175070434, 1.140804736575402, -0.01054831906835806],
        [-0.02400335680461803, -0.1289689760649706, 1.152972332869588]],
    2: [[0.6954522413574517, 0.1406786964702941, 0.1638690621722542],
        [0.04479456337203774, 0.859671118456422, 0.0955343181715404],
        [-0.005525882558113543, 0.004025210305978663, 1.001500672252135]],
    3: [[1.02582474766601, -0.02005319083821517, -0.005771556827795564],
        [-0.002234369519975978, 1.00458650188848, -0.002352132368503649],
        [-0.005013351468089286, -0.02529007181078517, 1.030303423278875]],
    4: [[1.379214128253342, -0.3088641446737119, -0.07034998357963095],
        [-0.06933485838138222, 1.08229674600235, -0.01296188762096683],
        [-0.002159009513570322, -0.04545932483731564, 1.047618334350886]],
}


def _luma(c: np.ndarray) -> np.ndarray:
    return c @ _LUMA


def _smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _gamut(c: np.ndarray, gamut: int, table: dict) -> np.ndarray:
    if gamut == 1:
        return c
    return c @ np.asarray(table.get(gamut, table[0])).T


def _lin_to_acescct(x: np.ndarray) -> np.ndarray:
    lo = 10.5402377416545 * x + 0.0729055341958355
    hi = (np.log2(np.maximum(x, 1e-300)) + 9.72) / 17.52
    return np.where(x <= 0.0078125, lo, hi)


def _acescct_to_lin(y: np.ndarray) -> np.ndarray:
    hi = np.exp2(np.minimum(y * 17.52 - 9.72, 1000.0))
    lo = (y - 0.0729055341958355) / 10.5402377416545
    return np.where(y > 0.155251141552511, hi, lo)


def _white_balance_gains(temperature: float, tint: float) -> np.ndarray:
    """One unit of Temperature is a stop of red against blue, one of Tint a stop
    of green against magenta; scaled so a neutral keeps its luminance."""
    g = np.array([2.0 ** (temperature / 2.0), 2.0 ** (-tint), 2.0 ** (-temperature / 2.0)])
    return g / float(g @ _LUMA)


def _rgb2hsv(c: np.ndarray) -> np.ndarray:
    # The shader's branchless form, so hues at the seams land where they do there.
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    st1 = g >= b
    p0 = np.where(st1, g, b)
    p1 = np.where(st1, b, g)
    p2 = np.where(st1, 0.0, -1.0)
    p3 = np.where(st1, -1.0 / 3.0, 2.0 / 3.0)
    st2 = r >= p0
    q0 = np.where(st2, r, p0)
    q1 = p1
    q2 = np.where(st2, p2, p3)
    q3 = np.where(st2, p0, r)
    d = q0 - np.minimum(q3, q1)
    e = 1e-10
    return np.stack([np.abs(q2 + (q3 - q1) / (6.0 * d + e)), d / (q0 + e), q0], axis=-1)


def _hsv2rgb(hsv: np.ndarray) -> np.ndarray:
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    out = []
    for k in (1.0, 2.0 / 3.0, 1.0 / 3.0):
        x = h + k
        p = np.abs((x - np.floor(x)) * 6.0 - 3.0)
        out.append(v * (1.0 + (np.clip(p - 1.0, 0.0, 1.0) - 1.0) * s))
    return np.stack(out, axis=-1)


def _sample_curve(table: np.ndarray, x: np.ndarray, ch: int) -> np.ndarray:
    """A 256-entry table read at texel centres: input x reads entry x * 255."""
    f = np.clip(x, 0.0, 1.0) * 255.0
    k = np.minimum(np.floor(f), 254.0).astype(np.int64)
    t = f - k
    return table[k, ch] * (1.0 - t) + table[k + 1, ch] * t


def _curve_table(v) -> Optional[np.ndarray]:
    if v is None:
        return None
    try:
        a = np.asarray(v, dtype=np.float64).reshape(256, 4)
    except (TypeError, ValueError):
        return None
    return a if np.all(np.isfinite(a)) else None


def apply_grading(
    img: np.ndarray,
    # Basic controls
    exposure: float = 0.0,
    gamma: float = 1.0,  # scalar gamma (used when gamma_rgb is None)
    gain: float = 1.0,
    lift: float = 0.0,
    saturation: float = 1.0,
    temperature: float = 6500.0,
    # The viewer's Temperature and Tint, in stops (see _white_balance_gains).
    # `temperature` above is a separate Kelvin multiply for the Grade Apply
    # node; leave it at 6500 when matching the viewer.
    temp_shift: float = 0.0,
    tint_shift: float = 0.0,
    # Extended Resolve-style controls (v3.2 sync)
    offset: float = 0.0,  # global additive offset (applied first)
    contrast: float = 1.0,  # power curve about the pivot
    pivot: float = _PIVOT_DEFAULT,  # contrast pivot point (18% grey)
    shadows: float = 0.0,  # shadow lift/crush (-1..1)
    highlights: float = 0.0,  # highlight expand/compress (-1..1)
    hue_shift: float = 0.0,  # degrees
    luma_mix: float = 1.0,   # 0 keeps the luminance the primaries produced
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
    # The rest of the viewer grade (all neutral by default).
    color_boost: float = 0.0,
    log_shadow: Optional[List[float]] = None,
    log_midtone: Optional[List[float]] = None,
    log_highlight: Optional[List[float]] = None,
    printer_rgb: Optional[List[float]] = None,
    curve_table=None,
    curve_mix: float = 1.0,
    secondary_curve_table=None,
    secondary_curve_mix: float = 0.0,
    source_gamut: int = 0,
) -> np.ndarray:
    """
    Apply the viewer's grade to a float32 image (H, W, C); alpha is untouched.

    The order and the maths are js/radiance_grade.js gradePixelFull:
    exposure, white balance, offset / lift / gain / gamma (in ACEScct when
    color_science is 1), contrast, log wheels, printer lights, shadows /
    highlights, colour boost, curves, hue curves, saturation, hue shift, Luma
    Mix, then the LUT. Computed in float64, returned as float32.
    """
    src = np.asarray(img)
    out = src.astype(np.float32, copy=True)

    def _vec(rgb, scalar):
        if rgb is not None and len(rgb) == 3:
            return np.asarray([float(x) for x in rgb])
        return np.full(3, float(scalar))

    v_off = _vec(offset_rgb, offset)
    v_lift = _vec(lift_rgb, lift)
    v_gain = _vec(gain_rgb, gain)
    v_gamma = _vec(gamma_rgb, gamma)
    v_ls = _vec(log_shadow, 0.0)
    v_lm = _vec(log_midtone, 0.0)
    v_lh = _vec(log_highlight, 0.0)
    v_pl = _vec(printer_rgb, 0.0)
    curves = _curve_table(curve_table) if curve_mix and curve_mix > 0 else None
    hue_curves = _curve_table(secondary_curve_table) if secondary_curve_mix and secondary_curve_mix > 0 else None

    is_default = (
        exposure == 0.0 and temp_shift == 0.0 and tint_shift == 0.0
        and abs(temperature - 6500.0) <= 10.0
        and not v_off.any() and not v_lift.any() and np.all(v_gain == 1.0) and np.all(v_gamma == 1.0)
        and contrast == 1.0 and not v_ls.any() and not v_lm.any() and not v_lh.any() and not v_pl.any()
        and shadows == 0.0 and highlights == 0.0 and color_boost == 0.0 and not gamut_compression
        and curves is None and hue_curves is None and saturation == 1.0 and hue_shift == 0.0
        and luma_mix >= 1.0 and lut_name == "None"
    )
    if is_default:
        return out

    gray = out.ndim < 3 or out.shape[-1] < 3
    c = (np.repeat(out[..., None] if out.ndim < 3 else out[..., :1], 3, axis=-1) if gray
         else out[..., :3]).astype(np.float64)

    # Exposure, clamped to the viewer's +/-12 stops.
    c = c * 2.0 ** min(max(float(exposure), -12.0), 12.0)

    # Kelvin white balance (Grade Apply node only), then the viewer's.
    if abs(temperature - 6500.0) > 10.0:
        c = c * np.asarray(_kelvin_to_rgb_multipliers(temperature))
    if temp_shift or tint_shift:
        c = c * _white_balance_gains(float(temp_shift), float(tint_shift))

    # Offset, lift (pivoted on luminance at white), gain, gamma (positives only).
    def _primaries(x):
        x = x + v_off
        x = x + v_lift * np.clip(1.0 - _luma(x), 0.0, 1.0)[..., None]
        x = x * v_gain
        inv = 1.0 / np.maximum(v_gamma, _GAMMA_FLOOR)
        return np.where(x > 0, np.power(np.maximum(x, 1e-300), inv), x)

    if str(color_science) in ("1", "ACEScct"):
        cct = _lin_to_acescct(_gamut(c, int(source_gamut), _TO_AP1))
        c = _gamut(_acescct_to_lin(_primaries(cct)), int(source_gamut), _FROM_AP1)
    else:
        c = _primaries(c)

    # Contrast: pivot * (c / pivot)^k on positives, negatives untouched.
    k = min(max(float(contrast), 0.0), _CONTRAST_MAX)
    if k != 1.0:
        p = max(float(pivot), _PIVOT_FLOOR)
        with np.errstate(over="ignore"):
            curved = np.minimum(p * np.power(np.maximum(c, 1e-300) / p, k), _CONTRAST_CEILING)
        c = np.where(c > 0, curved, c)

    # Log wheels: a gain weighted by luminance zone.
    if v_ls.any() or v_lm.any() or v_lh.any():
        y = _luma(c)
        sw = 1.0 - _smoothstep(0.0, 0.45, y)
        hw = _smoothstep(0.55, 1.0, y)
        mw = 1.0 - sw - hw
        c = c * (1.0 + v_ls * sw[..., None] + v_lm * mw[..., None] + v_lh * hw[..., None])

    # Printer lights: 2^(points / 50) per channel.
    if v_pl.any():
        c = c * np.power(2.0, v_pl / 50.0)

    # Shadows / highlights.
    if shadows or highlights:
        y = _luma(c)
        sw = (1.0 - _smoothstep(0.0, 0.5, y)) ** 2
        hw = _smoothstep(0.5, 1.0, y) ** 2
        c = c * ((1.0 + shadows * sw * 0.5) * (1.0 + highlights * hw * 0.5))[..., None]

    # Colour boost (vibrance).
    if color_boost:
        y = _luma(c)[..., None]
        sat_est = (c.max(axis=-1) - c.min(axis=-1))[..., None]
        f = 1.0 + (1.0 - sat_est * 0.8) * color_boost
        c = np.maximum(y + (c - y) * f, 0.0)

    # Gamut compression (not a viewer control; kept where it was).
    if gamut_compression:
        if _aces2_gamut_compress is not None:
            c = np.asarray(_aces2_gamut_compress(c.astype(np.float32)), dtype=np.float64)
        else:
            logger.warning("[Radiance Viewer] aces2_gamut_compress unavailable, gamut compression skipped.")

    # Curves: the table on 0..1, the output at 1.0 as a gain above, the output
    # at 0 as an offset below.
    if curves is not None:
        curved = np.empty_like(c)
        for ch in range(3):
            x = c[..., ch]
            curved[..., ch] = np.where(x >= 1.0, x * max(curves[255, ch], 0.0),
                                       np.where(x < 0.0, x + curves[0, ch], _sample_curve(curves, x, ch)))
        c = c + (curved - c) * float(curve_mix)

    # Luma Mix holds the luminance from here.
    ref_luma = _luma(c)

    # Hue vs hue / saturation / luminance.
    if hue_curves is not None:
        hsv = _rgb2hsv(c)
        look = [_sample_curve(hue_curves, hsv[..., 0], ch) for ch in range(3)]
        h = hsv[..., 0] + (look[0] - 0.5)
        h = h - np.floor(h)
        s = np.clip(hsv[..., 1] * look[1] * 2.0, 0.0, 1.0)
        v = np.clip(hsv[..., 2] + (look[2] - 0.5) * 2.0 * 0.5, 0.0, 65504.0)
        c = c + (_hsv2rgb(np.stack([h, s, v], axis=-1)) - c) * float(secondary_curve_mix)

    # Saturation about BT.709 luminance.
    if saturation != 1.0:
        y = _luma(c)[..., None]
        c = y + (c - y) * float(saturation)

    # Hue shift, in degrees.
    if hue_shift:
        hsv = _rgb2hsv(c)
        h = hsv[..., 0] + float(hue_shift) / 360.0
        h = np.where(h > 1.0, h - 1.0, h)
        h = np.where(h < 0.0, h + 1.0, h)
        c = _hsv2rgb(np.stack([h, hsv[..., 1], hsv[..., 2]], axis=-1))

    # Luma Mix: 0 keeps the luminance the primaries produced.
    if luma_mix < 1.0:
        m = min(max(float(luma_mix), 0.0), 1.0)
        y = _luma(c)
        safe_y = np.where(y > 1e-4, y, 1.0)
        held = np.where((y > 1e-4)[..., None], c * (ref_luma / safe_y)[..., None], c + (ref_luma - y)[..., None])
        c = held + (c - held) * m

    if gray:
        out = c[..., 0].astype(np.float32).reshape(out.shape)
    else:
        out[..., :3] = c.astype(np.float32)

    # LUT (last in chain)
    if lut_name != "None" and lut_intensity > 0.0:
        out = apply_lut(out, lut_name, lut_intensity)

    return out


def viewer_grade_kwargs(grading: dict) -> dict:
    """apply_grading arguments for the viewer's grade, as /radiance/deliver
    receives it (js/radiance_viewer.js builds the payload).

    Old saved grades lack newer keys; each falls back to its neutral value, and
    a missing pivot is 18% grey.
    """
    grading = grading or {}

    def _f(v, default):
        if isinstance(v, (list, tuple)):
            v = v[0] if v else default
        try:
            x = float(v)
        except (TypeError, ValueError):
            return default
        return x if np.isfinite(x) else default

    def _rgb(v, default):
        if isinstance(v, (list, tuple)) and len(v) >= 3:
            return [_f(x, default) for x in v[:3]]
        return [_f(v, default)] * 3

    cs = grading.get('colorScience', 0)
    return dict(
        exposure=_f(grading.get('exposure'), 0.0),
        temperature=6500.0,
        temp_shift=_f(grading.get('temperature'), 0.0),
        tint_shift=_f(grading.get('tint'), 0.0),
        offset_rgb=_rgb(grading.get('offset'), 0.0),
        lift_rgb=_rgb(grading.get('lift'), 0.0),
        gain_rgb=_rgb(grading.get('gain'), 1.0),
        gamma_rgb=_rgb(grading.get('gamma'), 1.0),
        contrast=_f(grading.get('contrast'), 1.0),
        pivot=_f(grading.get('pivot'), _PIVOT_DEFAULT),
        saturation=_f(grading.get('saturation'), 1.0),
        shadows=_f(grading.get('shadows'), 0.0),
        highlights=_f(grading.get('highlights'), 0.0),
        hue_shift=_f(grading.get('hue_shift'), 0.0),
        luma_mix=_f(grading.get('lumaMix'), 1.0),
        color_boost=_f(grading.get('colorBoost'), 0.0),
        log_shadow=_rgb(grading.get('logShadow'), 0.0),
        log_midtone=_rgb(grading.get('logMidtone'), 0.0),
        log_highlight=_rgb(grading.get('logHighlight'), 0.0),
        printer_rgb=[_f(grading.get('printerR'), 0.0), _f(grading.get('printerG'), 0.0),
                     _f(grading.get('printerB'), 0.0)],
        curve_table=grading.get('curveTable'),
        curve_mix=_f(grading.get('curveMix'), 1.0),
        secondary_curve_table=grading.get('secondaryCurveTable'),
        secondary_curve_mix=_f(grading.get('secondaryCurveMix'), 0.0),
        color_science=1 if str(cs) in ('1', 'ACEScct') else 0,
        source_gamut=int(_f(grading.get('sourceGamut'), 0.0)),
        lut_name=str(grading.get('lut_name', 'None') or 'None'),
        lut_intensity=_f(grading.get('lut_intensity'), 1.0),
        gamut_compression=bool(grading.get('gamut_compression', False)),
    )


def grading_to_cdl(grading: dict) -> dict:
    """ASC CDL equivalent of a Viewer grade (FIX-006).

    The Viewer applies, in order: exposure (x 2^e), offset (+o), lift
    (luma-pivoted), gain (x g), gamma (^ 1/gamma), then saturation with
    Rec.709 luma. Without lift and the non-SOP controls this is exactly

        ((in * 2^e + o) * g) ^ (1/gamma)  =  (in * slope + offset) ^ power
        slope = 2^e * g,  offset = o * g,  power = 1 / gamma

    The CDL sidecars used to write slope = gain, offset = offset (or the
    Viewer's lift) and power = gamma, i.e. the inverse curve. Controls a CDL
    cannot carry are listed in ``not_represented`` and ``exact`` is False.
    """
    import math

    def _rgb(key, default):
        v = grading.get(key, default)
        if isinstance(v, (int, float)):
            v = [v, v, v]
        try:
            v = [float(x) for x in list(v)[:3]]
        except (TypeError, ValueError):
            v = [default] * 3
        return v if len(v) == 3 else [default] * 3

    def _f(key, default):
        try:
            return float(grading.get(key, default))
        except (TypeError, ValueError):
            return default

    e = _f("exposure", 0.0)
    gain = _rgb("gain", 1.0)
    off = _rgb("offset", 0.0)
    gam = _rgb("gamma", 1.0)
    k = 2.0 ** e
    slope = [k * g for g in gain]
    offset = [o * g for o, g in zip(off, gain)]
    power = [1.0 / max(gm, 0.01) for gm in gam]
    sat = _f("saturation", 1.0)

    checks = {
        "lift": any(abs(x) > 1e-6 for x in _rgb("lift", 0.0)),
        "contrast": abs(_f("contrast", 1.0) - 1.0) > 1e-6,
        "shadows": abs(_f("shadows", 0.0)) > 1e-6,
        "highlights": abs(_f("highlights", 0.0)) > 1e-6,
        "hue_shift": abs(_f("hue_shift", 0.0)) > 1e-6,
        "temperature": abs(_f("temperature", 0.0)) > 1e-6,
        "tint": abs(_f("tint", 0.0)) > 1e-6,
        "lut": str(grading.get("lut_name", "None")) not in ("None", "", "none"),
        "lumaMix": abs(_f("lumaMix", 1.0) - 1.0) > 1e-6,
        "gamut_compression": bool(grading.get("gamut_compression", False)),
        "colorScience ACEScct": str(grading.get("colorScience", "0")) in ("1", "ACEScct"),
    }
    not_represented = [k for k, on in checks.items() if on]
    ok = all(math.isfinite(v) for v in slope + offset + power + [sat])
    return {"slope": slope, "offset": offset, "power": power, "saturation": sat,
            "exact": ok and not not_represented, "not_represented": not_represented}
