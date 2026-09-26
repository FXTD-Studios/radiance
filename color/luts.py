import numpy as np
import logging

from radiance.color.transfer import (
    linear_to_davinci_intermediate as _linear_to_davinci_intermediate,
    linear_to_canonlog3 as _linear_to_canonlog3,
    linear_to_log3g10 as _linear_to_log3g10,
)
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger("radiance.color.luts")

# ═══════════════════════════════════════════════════════════════════════════════
#                           CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

LUT_MODES = [
    # ── Display / Tonemap ──────────────────────────────────────────────────────
    "None",
    "sRGB (Display)",
    "Rec.709 (Broadcast)",
    "Filmic (Cinematic)",
    "Reinhard Tonemap",
    "ACES Filmic",
    # ── Camera Log Encoding (Linear → Log) ────────────────────────────────────
    "LogC3 (ARRI EI800)",
    "LogC4 (ARRI Alexa 35)",
    "F-Log2 (Fujifilm)",
    "C-Log3 (Canon)",
    "Log3G10 (RED IPP2)",
    "DaVinci Intermediate",
    "BMD Film Gen5",
    "V-Log (Panasonic)",
    "RED Log3G10",
    "—",
    "N-Log (Nikon)",
    "Linear to Log (Generic)",
    "IDT: LogC3 → Linear",
    "IDT: LogC4 → Linear",
    "IDT: V-Log → Linear",
    "IDT: Log3G10 → Linear",
    "IDT: DaVinci → Linear",
    "IDT: BMD Gen5 → Linear",
    "IDT: N-Log → Linear",
    # ── Analysis ──────────────────────────────────────────────────────────────
    "False Color (Exposure)",
    "Clip Check (Delivery)",
]

# ACES Output Color Space Transforms (ODT-style)
_M_SRGB_TO_ACESCG = np.array([
    [ 0.59719,  0.35458,  0.04823],
    [ 0.07600,  0.90834,  0.01566],
    [ 0.02840,  0.13383,  0.83777],
], dtype=np.float32)

_M_SRGB_TO_AP0 = np.array([
    [ 0.43963,  0.38298,  0.17739],
    [ 0.08978,  0.81380,  0.09642],
    [ 0.01754,  0.11170,  0.87076],
], dtype=np.float32)

_M_ACESCG_TO_LIN_SRGB = np.array([
    [ 1.60475, -0.53108, -0.07367],
    [-0.10208,  1.10813, -0.00605],
    [-0.00327, -0.07276,  1.07602],
], dtype=np.float32)

_M_ACESCG_TO_AP0 = np.array([
    [ 0.69545,  0.14058,  0.16397],
    [ 0.04479,  0.85963,  0.09558],
    [ -0.00545, 0.00402,  1.00143],
], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
#                    FORWARD CAMERA LOG LUT FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _lut_srgb(x: np.ndarray) -> np.ndarray:
    """Linear → sRGB gamma [IEC 61966-2-1]."""
    return np.clip(
        np.where(
            x <= 0.0031308,
            x * 12.92,
            1.055 * np.power(np.maximum(x, 0.0031308), 1.0 / 2.4) - 0.055,
        ),
        0.0,
        1.0,
    )


def _lut_rec709(x: np.ndarray) -> np.ndarray:
    """Linear → Rec.709 OETF [ITU-R BT.709-6]."""
    return np.clip(
        np.where(
            x < 0.018, x * 4.5, 1.099 * np.power(np.maximum(x, 0.018), 0.45) - 0.099
        ),
        0.0,
        1.0,
    )


def _lut_filmic(x: np.ndarray) -> np.ndarray:
    """Filmic tonemapping (Hable / Uncharted 2 curve)."""
    A, B, C, D, E, F = 0.15, 0.50, 0.10, 0.20, 0.02, 0.30

    def hable(v):
        return ((v * (A * v + C * B) + D * E) / (v * (A * v + B) + D * F)) - E / F

    # np.float32, not a bare float: NEP 50 promotes the whole expression
    # to float64 otherwise, doubling the memory of every frame downstream.
    white_scale = np.float32(1.0) / hable(np.float32(11.2))
    return np.clip(hable(np.maximum(x, 0.0) * 2.0) * white_scale, 0.0, 1.0)


def _lut_reinhard(x: np.ndarray) -> np.ndarray:
    """Reinhard global tonemapping."""
    return np.clip(x / (1.0 + x), 0.0, 1.0)


def _lut_aces_filmic(x: np.ndarray) -> np.ndarray:
    """ACES Filmic tone mapping (Narkowicz fit)."""
    x = np.maximum(x, 0.0)
    return np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0)


def _lut_acescg(x: np.ndarray) -> np.ndarray:
    """Linear sRGB → ACEScg (AP1 scene-linear)."""
    if x.ndim == 3 and x.shape[2] >= 3:
        rgb = x[..., :3]
        out = x.copy()
        out[..., :3] = np.tensordot(rgb, _M_SRGB_TO_ACESCG, axes=([2], [1]))
        return out
    return x


def _lut_aces2065(x: np.ndarray) -> np.ndarray:
    """Linear sRGB → ACES2065-1 (AP0 scene-linear)."""
    if x.ndim == 3 and x.shape[2] >= 3:
        rgb = x[..., :3]
        out = x.copy()
        out[..., :3] = np.tensordot(rgb, _M_SRGB_TO_AP0, axes=([2], [1]))
        return out
    return x


def _lut_acescct_encode(x: np.ndarray) -> np.ndarray:
    """Linear sRGB → ACEScct (log-encoded, AP1 primaries)."""
    if x.ndim != 3 or x.shape[2] < 3:
        return x
    out = x.copy()
    acescg = np.tensordot(out[..., :3], _M_SRGB_TO_ACESCG, axes=([2], [1]))
    cct = np.empty_like(acescg)
    lin_cut = 0.0078125
    mask = acescg <= lin_cut
    cct[mask]  = 10.5402377416545 * acescg[mask] + 0.0729055341958355
    clamped    = np.clip(acescg[~mask], 1e-10, None)
    cct[~mask] = (np.log2(clamped) + 9.72) / 17.52
    out[..., :3] = cct
    return out


def _lut_logc3(x: np.ndarray) -> np.ndarray:
    """Linear → ARRI LogC3 EI800."""
    cut = 0.010591
    a = 5.555556
    b = 0.052272
    c = 0.247190
    d = 0.385537
    e = 5.367655
    f = 0.092809
    return np.clip(
        np.where(x > cut, c * np.log10(np.maximum(a * x + b, 1e-10)) + d, e * x + f),
        0.0,
        1.0,
    )


def _lut_logc4(x: np.ndarray) -> np.ndarray:
    """Linear → ARRI LogC4 (Alexa 35)."""
    a = 2231.82630906768830
    b = 0.90713587487781030
    c = 0.09286412512218964
    s = 0.11359720861058910
    t = -0.01805699611991131
    log_branch = (
        np.log2(np.maximum(a * np.maximum(x, t) + 64.0, 1e-10)) - 6.0
    ) / 14.0 * b + c
    lin_branch = (x - t) / s
    return np.where(x >= t, log_branch, lin_branch)


def _lut_slog3(x: np.ndarray) -> np.ndarray:
    """Linear → Sony S-Log3."""
    cut = 0.01125
    return np.clip(
        np.where(
            x >= cut,
            (420.0 + np.log10(np.maximum(x + 0.01, 1e-10) / 0.19) * 261.5) / 1023.0,
            (x * (171.2102946929 - 95.0) / cut + 95.0) / 1023.0,
        ),
        0.0,
        1.0,
    )


def _lut_vlog(x: np.ndarray) -> np.ndarray:
    """Linear → Panasonic V-Log / V-Log3."""
    return np.clip(
        np.where(
            x < 0.01,
            5.6 * x + 0.125,
            0.241514 * np.log10(np.maximum(x + 0.00873, 1e-10)) + 0.598206,
        ),
        0.0,
        1.0,
    )


def _lut_flog2(x: np.ndarray) -> np.ndarray:
    """Linear → Fujifilm F-Log2."""
    cut1 = 0.000889
    a = 5.555556
    b = 0.064829
    c_c = 0.245281
    d = 0.384316
    e = 8.799461
    f = 0.092864
    return np.clip(
        np.where(
            x >= cut1, c_c * np.log10(np.maximum(a * x + b, 1e-10)) + d, e * x + f
        ),
        0.0,
        1.0,
    )


def _lut_clog3(x: np.ndarray) -> np.ndarray:
    """Linear → Canon C-Log3 v1.2 (delegates to the single source)."""
    return np.clip(_linear_to_canonlog3(x), 0.0, 1.0)


def _lut_log3g10(x: np.ndarray) -> np.ndarray:
    """Linear → RED Log3G10 v2 (delegates to the single source)."""
    return _linear_to_log3g10(x)


def _lut_davinci_intermediate(x: np.ndarray) -> np.ndarray:
    """Linear → DaVinci Intermediate.

    Delegates to radiance.color.transfer, which is the single source for every
    transfer function. This file used to carry its own (correct) copy while
    transfer.py carried a different, broken one -- the two disagreed by a
    factor of 2.6 at 18% grey and only the broken one was wired into the VAE
    and the IO nodes.
    """
    return _linear_to_davinci_intermediate(x)


def _lut_bmd_gen5(x: np.ndarray) -> np.ndarray:
    """Linear → Blackmagic Film Generation 5."""
    A = 0.08692876065491224
    B = 0.005494072432257808
    C = 0.5300133392291939
    D = 8.283605932402494
    E = 0.09246575342465753
    LIN_CUT = 0.005
    return np.where(x >= LIN_CUT, A * np.log(np.maximum(x + B, 1e-10)) + C, D * x + E)


def _lut_nlog(x: np.ndarray) -> np.ndarray:
    """Linear → Nikon N-Log."""
    cut1 = 0.328
    a = 0.635386119257087
    b = 0.0075
    c = 0.1466275659824047
    d = 0.6050830889540567
    log_v = c * np.log(np.maximum(x, 1e-10)) + d
    cbrt_v = a * np.power(np.maximum(x + b, 1e-10), 1.0 / 3.0)
    return np.where(x >= cut1, log_v, cbrt_v)


def _lut_lin_to_log(x: np.ndarray) -> np.ndarray:
    """Generic linear → log (Cineon-style)."""
    return np.clip(
        np.log2(np.maximum(x, 1e-10) * np.float32(5.55) + np.float32(1.0))
        / np.float32(np.log2(6.55)),
        0.0, 1.0,
    ).astype(np.float32)


def _lut_log_to_lin(x: np.ndarray) -> np.ndarray:
    """Generic log → linear (Cineon inverse)."""
    return np.clip((np.power(6.55, np.clip(x, 0.0, 1.0)) - 1.0) / 5.55, 0.0, 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
#                    INVERSE IDT CAMERA LOG FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _idt_logc3(x: np.ndarray) -> np.ndarray:
    """IDT: ARRI LogC3 EI800 → Scene Linear."""
    ENC_CUT = 5.367655 * 0.010591 + 0.092809
    log_out = (np.power(10.0, (x - 0.385537) / 0.247190) - 0.052272) / 5.555556
    lin_out = (x - 0.092809) / 5.367655
    return np.maximum(np.where(x > ENC_CUT, log_out, lin_out), 0.0)


def _idt_logc4(x: np.ndarray) -> np.ndarray:
    """IDT: ARRI LogC4 → Scene Linear."""
    a = 2231.82630906768830
    b = 0.90713587487781030
    c = 0.09286412512218964
    s = 0.11359720861058910
    t = -0.01805699611991131
    lc4_log_cut = (np.log2(a * t + 64.0) - 6.0) / 14.0 * b + c
    log_out = (np.exp2((x - c) / b * 14.0 + 6.0) - 64.0) / a
    lin_out = x * s + t
    return np.where(x >= lc4_log_cut, log_out, lin_out)


def _idt_slog3(x: np.ndarray) -> np.ndarray:
    """IDT: Sony S-Log3 → Scene Linear."""
    CUT_CV = (0.01125 * (171.2102946929 - 95.0) / 0.01125 + 95.0) / 1023.0
    log_out = np.power(10.0, (x * 1023.0 - 420.0) / 261.5) * 0.19 - 0.01
    lin_out = (x * 1023.0 - 95.0) * 0.01125 / (171.2102946929 - 95.0)
    return np.maximum(np.where(x >= CUT_CV, log_out, lin_out), 0.0)


def _idt_vlog(x: np.ndarray) -> np.ndarray:
    """IDT: Panasonic V-Log → Scene Linear."""
    return np.maximum(
        np.where(
            x < 0.181,
            (x - 0.125) / 5.6,
            np.power(10.0, (x - 0.598206) / 0.241514) - 0.00873,
        ),
        0.0,
    )


def _idt_flog2(x: np.ndarray) -> np.ndarray:
    """IDT: Fujifilm F-Log2 → Scene Linear."""
    a = 5.555556
    b = 0.064829
    c_c = 0.245281
    d = 0.384316
    e = 8.799461
    f = 0.092864
    CUT_CV = c_c * np.log10(a * 0.000889 + b) + d
    log_out = (np.power(10.0, (x - d) / c_c) - b) / a
    lin_out = (x - f) / e
    return np.where(x >= CUT_CV, log_out, lin_out)


def _idt_clog3(x: np.ndarray) -> np.ndarray:
    """IDT: Canon C-Log3 v1.2 → Scene Linear."""
    cl3_lin_cv = 1.9754798 * 0.014043 + 0.12512219
    k = 14.98325
    a = 0.36726845
    log_out = (np.power(10.0, (x - 0.12240537) / a) - 1.0) / k * 0.9
    lin_out = (x - 0.12512219) / 1.9754798 * 0.9
    return np.where(x > cl3_lin_cv, log_out, lin_out)


def _idt_log3g10(x: np.ndarray) -> np.ndarray:
    """IDT: RED Log3G10 v2 → Scene Linear."""
    s = np.sign(x)
    return s * (np.power(10.0, np.abs(x) / 0.224282) - 1.0) / 155.975327 - 0.01


def _idt_davinci_intermediate(x: np.ndarray) -> np.ndarray:
    """IDT: DaVinci Intermediate → Scene Linear."""
    DI_A = 0.0075
    DI_B = 7.0
    DI_C = 0.07329248
    DI_M = 10.44426855
    DI_LOG_CUT = 0.02740668
    log_out = np.exp2(x / DI_C - DI_B) - DI_A
    lin_out = x / DI_M
    return np.where(x > DI_LOG_CUT, log_out, lin_out)


def _idt_bmd_gen5(x: np.ndarray) -> np.ndarray:
    """IDT: BMD Film Generation 5 → Scene Linear."""
    A = 0.08692876065491224
    B = 0.005494072432257808
    C = 0.5300133392291939
    D = 8.283605932402494
    E = 0.09246575342465753
    LOG_CUT = D * 0.005 + E
    log_out = np.exp((x - C) / A) - B
    lin_out = (x - E) / D
    return np.where(x >= LOG_CUT, log_out, lin_out)


def _idt_nlog(x: np.ndarray) -> np.ndarray:
    """IDT: Nikon N-Log → Scene Linear."""
    a = 0.635386119257087
    b = 0.0075
    c = 0.1466275659824047
    d = 0.6050830889540567
    CUT_CV = a * (0.328 + b) ** (1.0 / 3.0)
    log_out = np.exp((x - d) / c)
    cbrt_out = np.power(x / a, 3.0) - b
    return np.maximum(np.where(x >= CUT_CV, log_out, cbrt_out), 0.0)


def _lut_passthrough(x: np.ndarray) -> np.ndarray:
    """Placeholder for separator."""
    return x


# ═══════════════════════════════════════════════════════════════════════════════
#                    ANALYSIS & OVERLAYS
# ═══════════════════════════════════════════════════════════════════════════════

def _lut_false_color(img: np.ndarray) -> np.ndarray:
    """ARRI False Color mapping (IRE proxy via sRGB luma)."""
    out = np.empty_like(img)
    if img.ndim == 3 and img.shape[2] >= 3:
        luma = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
        out[..., 0] = luma
        out[..., 1] = luma
        out[..., 2] = luma

        m_red = luma >= 0.99
        m_yel = (luma >= 0.97) & (luma < 0.99)
        m_pnk = (luma >= 0.52) & (luma < 0.56)
        m_grn = (luma >= 0.42) & (luma < 0.45)
        m_cya = (luma >= 0.38) & (luma < 0.40)
        m_pur = luma <= 0.02

        out[m_red] = [1.0, 0.0, 0.0]
        out[m_yel] = [1.0, 1.0, 0.0]
        out[m_pnk] = [1.0, 0.5, 0.8]
        out[m_grn] = [0.0, 0.8, 0.2]
        out[m_cya] = [0.0, 1.0, 1.0]
        out[m_pur] = [0.6, 0.0, 0.8]
    else:
        out = img.copy()
    return out


def _lut_clip_check(
    img: np.ndarray,
    white_ceiling: float = 1.0,
    black_floor: float = 0.0,
) -> np.ndarray:
    """Delivery clip check overlay."""
    out = np.empty((*img.shape[:2], 3), dtype=np.float32)
    rgb = img[..., :3] if img.ndim == 3 and img.shape[2] >= 3 else np.stack([img[..., 0]] * 3, axis=-1)

    luma = (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2])
    grey = np.stack([luma, luma, luma], axis=-1)

    blown = np.any(rgb > white_ceiling, axis=-1)
    crushed = np.any(rgb < black_floor, axis=-1)

    out[:] = grey * 0.35
    out[blown]  = [1.0, 0.05, 0.05]
    out[crushed] = [0.05, 0.15, 1.0]

    return np.clip(out, 0.0, 1.0)


# ── LUT dispatch table ────────────────────────────────────────────────────────
_LUT_FUNCTIONS = {
    "None": None,
    # Display / Tonemap
    "sRGB (Display)": _lut_srgb,
    "Rec.709 (Broadcast)": _lut_rec709,
    "Filmic (Cinematic)": _lut_filmic,
    "Reinhard Tonemap": _lut_reinhard,
    "ACES Filmic": _lut_aces_filmic,
    "ACEScg (AP1)": _lut_acescg,
    "ACES2065-1 (AP0)": _lut_aces2065,
    "ACEScct": _lut_acescct_encode,
    # Camera Log — Forward (Linear → Log)
    "LogC3 (ARRI EI800)": _lut_logc3,
    "LogC4 (ARRI Alexa 35)": _lut_logc4,
    "F-Log2 (Fujifilm)": _lut_flog2,
    "C-Log3 (Canon)": _lut_clog3,
    "Log3G10 (RED IPP2)": _lut_log3g10,
    "DaVinci Intermediate": _lut_davinci_intermediate,
    "BMD Film Gen5": _lut_bmd_gen5,
    "V-Log (Panasonic)": _lut_vlog,
    "RED Log3G10": _lut_log3g10,
    "—": _lut_passthrough,
    "N-Log (Nikon)": _lut_nlog,
    "Linear to Log (Generic)": _lut_lin_to_log,
    "IDT: LogC3 → Linear": _idt_logc3,
    "IDT: LogC4 → Linear": _idt_logc4,
    "IDT: V-Log → Linear": _idt_vlog,
    "IDT: Log3G10 → Linear": _idt_log3g10,
    "IDT: DaVinci → Linear": _idt_davinci_intermediate,
    "IDT: BMD Gen5 → Linear": _idt_bmd_gen5,
    "IDT: N-Log → Linear": _idt_nlog,
    # Analysis
    "False Color (Exposure)": "false_color",
    "Clip Check (Delivery)": "clip_check",
}
