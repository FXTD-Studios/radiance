"""Log and gamma transfer functions — both numpy and GPU-accelerated torch variants.

Every function has a numpy (..._to_linear / linear_to_...) and a tensor
(tensor_..._to_linear / tensor_linear_to_...) form.  The tensor variants avoid
device round-trips by operating directly on GPU tensors.
"""
from __future__ import annotations

import numpy as np
import torch

# ── sRGB Gamma ────────────────────────────────────────────────────────────────


def srgb_to_linear(img: np.ndarray) -> np.ndarray:
    return np.where(
        img <= 0.04045, img / 12.92, np.power((np.maximum(img, 0) + 0.055) / 1.055, 2.4)
    ).astype(np.float32)


def linear_to_srgb(img: np.ndarray) -> np.ndarray:
    return np.where(
        img <= 0.0031308,
        img * 12.92,
        1.055 * np.power(np.maximum(img, 1e-10), 1 / 2.4) - 0.055,
    ).astype(np.float32)


def tensor_srgb_to_linear(tensor: torch.Tensor, gamma: float = 2.2) -> torch.Tensor:
    if gamma == 2.2:
        abs_t = tensor.abs()
        sign = tensor.sign()
        low = abs_t / 12.92
        high = torch.pow((abs_t + 0.055) / 1.055, 2.4)
        return torch.where(abs_t <= 0.04045, low, high) * sign
    abs_t = tensor.abs()
    return torch.pow(abs_t.clamp(min=1e-10), gamma) * tensor.sign()


def tensor_linear_to_srgb(tensor: torch.Tensor) -> torch.Tensor:
    abs_t = tensor.abs()
    sign = tensor.sign()
    low = abs_t * 12.92
    high = 1.055 * torch.pow(abs_t.clamp(min=1e-10), 1 / 2.4) - 0.055
    return torch.where(abs_t <= 0.0031308, low, high) * sign


# ── BT.709 / BT.2020 camera OETF ──────────────────────────────────────────────
#
# ITU-R BT.709-6 §1.2 and ITU-R BT.2020-2 Table 4.  Same shape, different
# constants: BT.2020 tightens alpha and beta for 10/12-bit precision.  BT.709
# rounds them to 1.099 / 0.018.
#
# These were missing entirely.  ``io/writer.py`` called
# ``color_utils.linear_to_rec709`` for every Rec.709 export, the AttributeError
# was swallowed by a blanket except, and the file was written with untouched
# scene-linear values, roughly 2.2 stops dark in the midtones, reported as a
# success.  Rec.2020 fared no better: it applied a 1/2.4 power curve described
# in its own comment as "PQ gamma as a proxy" and hard-clamped to [0,1],
# destroying every highlight above 1.0 on an EXR export.
#
# Both curves here are sign-symmetric and extend the power segment above 1.0,
# so scene-linear values outside [0,1] survive into float formats instead of
# being crushed.  Integer formats quantise at the point of writing, which is
# where clamping belongs.

BT709_ALPHA: float = 1.099
BT709_BETA: float = 0.018
BT2020_ALPHA: float = 1.09929682680944
BT2020_BETA: float = 0.018053968510807


def _oetf_gamma(img: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    a = np.abs(img)
    return (
        np.sign(img)
        * np.where(a < beta, 4.5 * a, alpha * np.power(np.maximum(a, 1e-10), 0.45) - (alpha - 1.0))
    ).astype(np.float32)


def _eotf_gamma(img: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    a = np.abs(img)
    return (
        np.sign(img)
        * np.where(
            a < 4.5 * beta,
            a / 4.5,
            np.power((np.maximum(a, 0.0) + (alpha - 1.0)) / alpha, 1.0 / 0.45),
        )
    ).astype(np.float32)


def linear_to_rec709(img: np.ndarray) -> np.ndarray:
    """Scene-linear → BT.709 OETF encoded (ITU-R BT.709-6 §1.2)."""
    return _oetf_gamma(img, BT709_ALPHA, BT709_BETA)


def rec709_to_linear(img: np.ndarray) -> np.ndarray:
    """BT.709 OETF encoded → scene-linear."""
    return _eotf_gamma(img, BT709_ALPHA, BT709_BETA)


def linear_to_rec2020(img: np.ndarray) -> np.ndarray:
    """Scene-linear → BT.2020 OETF encoded (ITU-R BT.2020-2 Table 4).

    Transfer only.  ``linear_rec709_to_rec2020`` does the primaries.
    """
    return _oetf_gamma(img, BT2020_ALPHA, BT2020_BETA)


def rec2020_to_linear(img: np.ndarray) -> np.ndarray:
    """BT.2020 OETF encoded → scene-linear.  Transfer only."""
    return _eotf_gamma(img, BT2020_ALPHA, BT2020_BETA)


def tensor_linear_to_rec709(tensor: torch.Tensor) -> torch.Tensor:
    a = tensor.abs()
    low = 4.5 * a
    high = BT709_ALPHA * torch.pow(a.clamp(min=1e-10), 0.45) - (BT709_ALPHA - 1.0)
    return torch.where(a < BT709_BETA, low, high) * tensor.sign()


def tensor_rec709_to_linear(tensor: torch.Tensor) -> torch.Tensor:
    a = tensor.abs()
    low = a / 4.5
    high = torch.pow(((a + (BT709_ALPHA - 1.0)) / BT709_ALPHA).clamp(min=0.0), 1.0 / 0.45)
    return torch.where(a < 4.5 * BT709_BETA, low, high) * tensor.sign()


def tensor_linear_to_rec2020(tensor: torch.Tensor) -> torch.Tensor:
    a = tensor.abs()
    low = 4.5 * a
    high = BT2020_ALPHA * torch.pow(a.clamp(min=1e-10), 0.45) - (BT2020_ALPHA - 1.0)
    return torch.where(a < BT2020_BETA, low, high) * tensor.sign()


def tensor_rec2020_to_linear(tensor: torch.Tensor) -> torch.Tensor:
    a = tensor.abs()
    low = a / 4.5
    high = torch.pow(((a + (BT2020_ALPHA - 1.0)) / BT2020_ALPHA).clamp(min=0.0), 1.0 / 0.45)
    return torch.where(a < 4.5 * BT2020_BETA, low, high) * tensor.sign()


# ── ARRI LogC3 ────────────────────────────────────────────────────────────────

LOGC3_EI_PARAMS = {
    160: (0.005561, 5.061087, 0.089004, 0.269035, 0.391007, 6.332427, 0.108361),
    200: (0.006208, 5.168208, 0.076621, 0.265275, 0.391007, 5.842037, 0.099519),
    250: (0.006871, 5.282072, 0.065521, 0.261620, 0.391007, 5.397270, 0.091111),
    320: (0.007622, 5.399335, 0.055194, 0.257766, 0.391007, 4.969419, 0.083295),
    400: (0.008318, 5.510883, 0.046585, 0.254174, 0.391007, 4.606965, 0.076257),
    500: (0.009031, 5.618393, 0.039023, 0.250758, 0.391007, 4.282556, 0.069776),
    640: (0.009840, 5.737055, 0.031538, 0.247070, 0.391007, 3.946374, 0.063409),
    800: (0.010591, 5.555556, 0.052272, 0.247190, 0.385537, 5.367655, 0.092809),
    1000: (0.011361, 5.944966, 0.019018, 0.240020, 0.391007, 3.369506, 0.051759),
    1280: (0.012235, 6.056760, 0.013804, 0.236500, 0.391007, 3.088156, 0.046447),
    1600: (0.013047, 6.161541, 0.009677, 0.233182, 0.391007, 2.852200, 0.041773),
    2000: (0.013901, 6.260724, 0.006210, 0.230014, 0.391007, 2.643126, 0.037413),
    2560: (0.014842, 6.362496, 0.002995, 0.226764, 0.391007, 2.440085, 0.033198),
    3200: (0.015711, 6.456037, 0.000295, 0.223740, 0.391007, 2.265605, 0.029493),
}


def linear_to_logc3(img: np.ndarray, ei: int = 800) -> np.ndarray:
    params = LOGC3_EI_PARAMS.get(ei, LOGC3_EI_PARAMS[800])
    cut, a, b, c, d, e, f = params
    out = np.empty_like(img, dtype=np.float32)
    mask = img > cut
    out[mask] = c * np.log10(a * img[mask] + b) + d
    out[~mask] = e * img[~mask] + f
    return out


def logc3_to_linear(img: np.ndarray, ei: int = 800) -> np.ndarray:
    params = LOGC3_EI_PARAMS.get(ei, LOGC3_EI_PARAMS[800])
    cut, a, b, c, d, e, f = params
    cut_encoded = e * cut + f
    out = np.empty_like(img, dtype=np.float32)
    mask = img > cut_encoded
    out[mask] = (np.power(10.0, (img[mask] - d) / c) - b) / a
    out[~mask] = (img[~mask] - f) / e
    return out


def tensor_linear_to_logc3(tensor: torch.Tensor, ei: int = 800) -> torch.Tensor:
    params = LOGC3_EI_PARAMS.get(ei, LOGC3_EI_PARAMS[800])
    cut, a, b, c, d, e, f = params
    return torch.where(tensor > cut,
                       c * torch.log10(a * tensor + b) + d,
                       e * tensor + f)


def tensor_logc3_to_linear(tensor: torch.Tensor, ei: int = 800) -> torch.Tensor:
    params = LOGC3_EI_PARAMS.get(ei, LOGC3_EI_PARAMS[800])
    cut, a, b, c, d, e, f = params
    cut_encoded = e * cut + f
    return torch.where(tensor > cut_encoded,
                       (torch.pow(10.0, (tensor - d) / c) - b) / a,
                       (tensor - f) / e)


# ── ARRI LogC4 ────────────────────────────────────────────────────────────────

_LOGC4_A = 2231.826309067637
_LOGC4_B = 0.9071358691330627
_LOGC4_C = 0.0928641308669373
_LOGC4_T = -0.0180569961199123
_LOGC4_S = 0.1135773173772412


def linear_to_logc4(img: np.ndarray) -> np.ndarray:
    out = np.empty_like(img, dtype=np.float32)
    mask = img >= _LOGC4_T
    out[mask] = ((np.log2(_LOGC4_A * img[mask] + 64.0) - 6.0) / 14.0) * _LOGC4_B + _LOGC4_C
    out[~mask] = (img[~mask] - _LOGC4_T) / _LOGC4_S
    return out


def logc4_to_linear(img: np.ndarray) -> np.ndarray:
    out = np.empty_like(img, dtype=np.float32)
    mask = img >= 0.0
    out[mask] = (np.power(2.0, ((img[mask] - _LOGC4_C) / _LOGC4_B) * 14.0 + 6.0) - 64.0) / _LOGC4_A
    out[~mask] = img[~mask] * _LOGC4_S + _LOGC4_T
    return out


def tensor_linear_to_logc4(tensor: torch.Tensor) -> torch.Tensor:
    log_val = ((torch.log2((_LOGC4_A * tensor + 64.0).clamp(min=1e-10)) - 6.0) / 14.0) * _LOGC4_B + _LOGC4_C
    lin_val = (tensor - _LOGC4_T) / _LOGC4_S
    return torch.where(tensor >= _LOGC4_T, log_val, lin_val)


def tensor_logc4_to_linear(tensor: torch.Tensor) -> torch.Tensor:
    log_val = (torch.pow(2.0, ((tensor - _LOGC4_C) / _LOGC4_B) * 14.0 + 6.0) - 64.0) / _LOGC4_A
    lin_val = tensor * _LOGC4_S + _LOGC4_T
    return torch.where(tensor >= 0.0, log_val, lin_val)


# ── Sony S-Log3 ───────────────────────────────────────────────────────────────

# Sony S-Log3 toe, per the Sony white paper:
#
#     x <  0.01125 :  (x * (171.2102946929 - 95) / 0.01125 + 95) / 1023
#
# i.e. a slope of 76.2102946929 / (0.01125 * 1023) = 6.6222 applied to x
# DIRECTLY -- there is no +0.01 shift below the cut; that offset belongs to the
# log branch only.
#
# This was 76.2102946929 / (0.02125 * 1023) = 3.5058 with a `+ 0.01` shift on
# the input. Encode and decode were mutually consistent, so every round-trip
# test passed, but the absolute mapping was wrong below 0.011 linear: black
# encoded to 0.127921 instead of 0.092864 (≈36 code values at 10-bit), and
# S-Log3 CV 0.130 decoded to +0.000593 where the spec gives +0.005608 -- a 9.5x
# shadow crush. color/luts.py:194 already carried the correct curve, so the two
# implementations in-tree disagreed.
_TOE_SLOPE_SLOG3 = (171.2102946929 - 95.0) / (0.01125 * 1023.0)  # ≈ 6.6222
_SLOG3_TOE_INTERCEPT = 95.0 / 1023.0


def linear_to_slog3(img: np.ndarray) -> np.ndarray:
    img = np.maximum(img, -0.01)
    cut = 0.011250
    out = np.empty_like(img, dtype=np.float32)
    mask = img >= cut
    out[mask] = (420.0 + np.log10((img[mask] + 0.01) / 0.19) * 261.5) / 1023.0
    out[~mask] = _TOE_SLOPE_SLOG3 * img[~mask] + _SLOG3_TOE_INTERCEPT
    return out


def slog3_to_linear(img: np.ndarray) -> np.ndarray:
    cut_v = 171.2102946929 / 1023.0
    out = np.empty_like(img, dtype=np.float32)
    mask = img >= cut_v
    out[mask] = 0.19 * np.power(10.0, (img[mask] * 1023.0 - 420.0) / 261.5) - 0.01
    out[~mask] = (img[~mask] - _SLOG3_TOE_INTERCEPT) / _TOE_SLOPE_SLOG3
    return out


def tensor_linear_to_slog3(tensor: torch.Tensor) -> torch.Tensor:
    cut = 0.011250
    tensor_clamped = tensor.clamp(min=-0.01)
    log_val = (420.0 + torch.log10((tensor_clamped + 0.01).clamp(min=1e-10) / 0.19) * 261.5) / 1023.0
    lin_val = _TOE_SLOPE_SLOG3 * tensor_clamped + _SLOG3_TOE_INTERCEPT
    return torch.where(tensor_clamped >= cut, log_val, lin_val)


def tensor_slog3_to_linear(tensor: torch.Tensor) -> torch.Tensor:
    cut_v = 171.2102946929 / 1023.0
    log_val = 0.19 * torch.pow(10.0, (tensor * 1023.0 - 420.0) / 261.5) - 0.01
    lin_val = (tensor - _SLOG3_TOE_INTERCEPT) / _TOE_SLOPE_SLOG3
    return torch.where(tensor >= cut_v, log_val, lin_val)


# ── Panasonic V-Log ───────────────────────────────────────────────────────────

def linear_to_vlog(img: np.ndarray) -> np.ndarray:
    cut = 0.01
    b, c, d = 0.00873, 0.241514, 0.598206
    out = np.empty_like(img, dtype=np.float32)
    mask = img >= cut
    out[mask] = c * np.log10(img[mask] + b) + d
    out[~mask] = 5.6 * img[~mask] + 0.125
    return out


def vlog_to_linear(img: np.ndarray) -> np.ndarray:
    b, c, d = 0.00873, 0.241514, 0.598206
    cut_encoded = 0.181  # 5.6 * 0.01 + 0.125
    out = np.empty_like(img, dtype=np.float32)
    mask = img >= cut_encoded
    out[mask] = np.power(10.0, (img[mask] - d) / c) - b
    out[~mask] = (img[~mask] - 0.125) / 5.6
    return out


def tensor_linear_to_vlog(tensor: torch.Tensor) -> torch.Tensor:
    cut = 0.01
    b, c, d = 0.00873, 0.241514, 0.598206
    log_val = c * torch.log10(tensor + b) + d
    lin_val = 5.6 * tensor + 0.125
    return torch.where(tensor >= cut, log_val, lin_val)


def tensor_vlog_to_linear(tensor: torch.Tensor) -> torch.Tensor:
    b, c, d = 0.00873, 0.241514, 0.598206
    cut_encoded = 0.181000
    log_val = torch.pow(10.0, (tensor - d) / c) - b
    lin_val = (tensor - 0.125) / 5.6
    return torch.where(tensor >= cut_encoded, log_val, lin_val)


# ── Canon Log 3 ───────────────────────────────────────────────────────────────

# Canon Log 3 v1.2, per Canon's published specification.
#
# The previous implementation was a two-segment approximation that (a) omitted
# the x/0.9 reflectance scaling entirely, (b) used the negative-branch intercept
# 0.12783901 on the POSITIVE log branch where the spec calls for 0.12240537,
# and (c) used a fitted toe slope of 5.449285 instead of the spec's 1.9754798.
# The result was an 0.0089 discontinuity at the cut (~9 code values at 10-bit,
# a visible band) and an encode/decode pair that disagreed by 4.4e-3 -- so a
# round trip through Canon Log 3 did not return the original image.
# 18% grey now encodes to 0.343371 as specified (was 0.336392).
_CANON_SCALE = 0.9          # linear reflectance -> Canon's normalised x
_CANON_A = 14.98325         # log gain
_CANON_C = 0.36726845       # log scale
_CANON_LIN_SLOPE = 1.9754798
_CANON_LIN_OFFSET = 0.12512219
_CANON_POS_OFFSET = 0.12240537
_CANON_NEG_OFFSET = 0.12783901
_CANON_CUT_LO = -0.009670   # in normalised x
_CANON_CUT_HI = 0.014043    # in normalised x
_CANON_ENC_LO = _CANON_LIN_SLOPE * _CANON_CUT_LO + _CANON_LIN_OFFSET
_CANON_ENC_HI = _CANON_LIN_SLOPE * _CANON_CUT_HI + _CANON_LIN_OFFSET


def linear_to_canonlog3(img: np.ndarray) -> np.ndarray:
    xr = np.asarray(img, dtype=np.float32) / _CANON_SCALE
    out = np.empty_like(xr, dtype=np.float32)
    neg = xr < _CANON_CUT_LO
    pos = xr > _CANON_CUT_HI
    mid = ~(neg | pos)
    out[neg] = -_CANON_C * np.log10(np.maximum(1.0 - _CANON_A * xr[neg], 1e-10)) + _CANON_NEG_OFFSET
    out[mid] = _CANON_LIN_SLOPE * xr[mid] + _CANON_LIN_OFFSET
    out[pos] = _CANON_C * np.log10(np.maximum(_CANON_A * xr[pos] + 1.0, 1e-10)) + _CANON_POS_OFFSET
    return out


def canonlog3_to_linear(img: np.ndarray) -> np.ndarray:
    y = np.asarray(img, dtype=np.float32)
    xr = np.empty_like(y, dtype=np.float32)
    neg = y < _CANON_ENC_LO
    pos = y > _CANON_ENC_HI
    mid = ~(neg | pos)
    xr[neg] = (1.0 - np.power(10.0, (_CANON_NEG_OFFSET - y[neg]) / _CANON_C)) / _CANON_A
    xr[mid] = (y[mid] - _CANON_LIN_OFFSET) / _CANON_LIN_SLOPE
    xr[pos] = (np.power(10.0, (y[pos] - _CANON_POS_OFFSET) / _CANON_C) - 1.0) / _CANON_A
    return xr * _CANON_SCALE


def tensor_linear_to_canonlog3(tensor: torch.Tensor) -> torch.Tensor:
    xr = tensor / _CANON_SCALE
    neg_val = -_CANON_C * torch.log10((1.0 - _CANON_A * xr).clamp(min=1e-10)) + _CANON_NEG_OFFSET
    mid_val = _CANON_LIN_SLOPE * xr + _CANON_LIN_OFFSET
    pos_val = _CANON_C * torch.log10((_CANON_A * xr + 1.0).clamp(min=1e-10)) + _CANON_POS_OFFSET
    out = torch.where(xr > _CANON_CUT_HI, pos_val, mid_val)
    return torch.where(xr < _CANON_CUT_LO, neg_val, out)


def tensor_canonlog3_to_linear(tensor: torch.Tensor) -> torch.Tensor:
    neg_val = (1.0 - torch.pow(10.0, (_CANON_NEG_OFFSET - tensor) / _CANON_C)) / _CANON_A
    mid_val = (tensor - _CANON_LIN_OFFSET) / _CANON_LIN_SLOPE
    pos_val = (torch.pow(10.0, (tensor - _CANON_POS_OFFSET) / _CANON_C) - 1.0) / _CANON_A
    out = torch.where(tensor > _CANON_ENC_HI, pos_val, mid_val)
    return torch.where(tensor < _CANON_ENC_LO, neg_val, out) * _CANON_SCALE


# ── RED Log3G10 ───────────────────────────────────────────────────────────────
#
# RED Log3G10 v2 is a single continuous curve over a shifted input, not a
# piecewise one. The previous implementation dropped the +0.01 shift and bolted
# on a linear toe below 0.01, which put 18% grey at 0.338243 instead of the
# specified 1/3, and made black 0.0 instead of the correct 0.0915. As with
# DaVinci Intermediate and Canon Log 3, color/luts.py already carried the
# correct form -- this is now the single source and luts.py delegates to it.
_LOG3G10_A = 0.224282
_LOG3G10_B = 155.975327
_LOG3G10_SHIFT = 0.01


def linear_to_log3g10(img: np.ndarray) -> np.ndarray:
    x = np.asarray(img, dtype=np.float32) + _LOG3G10_SHIFT
    return (np.sign(x) * _LOG3G10_A
            * np.log10(np.abs(x) * _LOG3G10_B + 1.0)).astype(np.float32)


def log3g10_to_linear(img: np.ndarray) -> np.ndarray:
    y = np.asarray(img, dtype=np.float32)
    x = np.sign(y) * (np.power(10.0, np.abs(y) / _LOG3G10_A) - 1.0) / _LOG3G10_B
    return (x - _LOG3G10_SHIFT).astype(np.float32)


def tensor_linear_to_log3g10(tensor: torch.Tensor) -> torch.Tensor:
    x = tensor + _LOG3G10_SHIFT
    return torch.sign(x) * _LOG3G10_A * torch.log10(x.abs() * _LOG3G10_B + 1.0)


def tensor_log3g10_to_linear(tensor: torch.Tensor) -> torch.Tensor:
    x = torch.sign(tensor) * (torch.pow(10.0, tensor.abs() / _LOG3G10_A) - 1.0) / _LOG3G10_B
    return x - _LOG3G10_SHIFT


# ── ACEScct ───────────────────────────────────────────────────────────────────

_ACESCCT_CUT = 0.0078125
_ACESCCT_A = 10.5402377416545
_ACESCCT_B = 0.0729055341958355
_ACESCCT_CUT_ENCODED = 0.155251141552511


def linear_to_acescct(img: np.ndarray) -> np.ndarray:
    out = np.empty_like(img, dtype=np.float32)
    mask = img > _ACESCCT_CUT
    out[mask] = (np.log2(img[mask]) + 9.72) / 17.52
    out[~mask] = _ACESCCT_A * img[~mask] + _ACESCCT_B
    return out


def acescct_to_linear(img: np.ndarray) -> np.ndarray:
    out = np.empty_like(img, dtype=np.float32)
    mask = img > _ACESCCT_CUT_ENCODED
    out[mask] = np.power(2.0, img[mask] * 17.52 - 9.72)
    out[~mask] = (img[~mask] - _ACESCCT_B) / _ACESCCT_A
    return out


def tensor_linear_to_acescct(tensor: torch.Tensor) -> torch.Tensor:
    log_val = (torch.log2(tensor.clamp(min=1e-10)) + 9.72) / 17.52
    lin_val = _ACESCCT_A * tensor + _ACESCCT_B
    return torch.where(tensor > _ACESCCT_CUT, log_val, lin_val)


def tensor_acescct_to_linear(tensor: torch.Tensor) -> torch.Tensor:
    log_val = torch.pow(2.0, tensor * 17.52 - 9.72)
    lin_val = (tensor - _ACESCCT_B) / _ACESCCT_A
    return torch.where(tensor > _ACESCCT_CUT_ENCODED, log_val, lin_val)


# ── ACEScc ────────────────────────────────────────────────────────────────────
# S-2014-003. Pure log2 encoding with a small-value fold; unlike ACEScct there
# is no linear toe segment. AUDIT-FIX (2026-08): previously absent -- the only
# node advertising ACEScc silently returned its input unchanged.

_ACESCC_MIN = (np.log2(2.0 ** -16) + 9.72) / 17.52          # x <= 0 pins here
_ACESCC_FOLD = 2.0 ** -15
_ACESCC_MAX_ENCODED = (np.log2(65504.0) + 9.72) / 17.52


def linear_to_acescc(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    out = np.empty_like(img, dtype=np.float32)
    neg = img <= 0.0
    fold = (~neg) & (img < _ACESCC_FOLD)
    rest = img >= _ACESCC_FOLD
    out[neg] = _ACESCC_MIN
    out[fold] = (np.log2(2.0 ** -16 + img[fold] * 0.5) + 9.72) / 17.52
    out[rest] = (np.log2(img[rest]) + 9.72) / 17.52
    return out


def acescc_to_linear(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    out = np.empty_like(img, dtype=np.float32)
    low = img <= ((9.72 - 15.0) / 17.52)
    high = img >= _ACESCC_MAX_ENCODED
    mid = (~low) & (~high)
    out[low] = (np.power(2.0, img[low] * 17.52 - 9.72) - 2.0 ** -16) * 2.0
    out[mid] = np.power(2.0, img[mid] * 17.52 - 9.72)
    out[high] = 65504.0
    return out


def tensor_linear_to_acescc(tensor: torch.Tensor) -> torch.Tensor:
    fold_val = (torch.log2(2.0 ** -16 + tensor.clamp(min=0.0) * 0.5) + 9.72) / 17.52
    log_val = (torch.log2(tensor.clamp(min=_ACESCC_FOLD)) + 9.72) / 17.52
    out = torch.where(tensor >= _ACESCC_FOLD, log_val, fold_val)
    return torch.where(tensor <= 0.0, torch.full_like(tensor, float(_ACESCC_MIN)), out)


def tensor_acescc_to_linear(tensor: torch.Tensor) -> torch.Tensor:
    pow_val = torch.pow(2.0, tensor * 17.52 - 9.72)
    low_val = (pow_val - 2.0 ** -16) * 2.0
    out = torch.where(tensor <= (9.72 - 15.0) / 17.52, low_val, pow_val)
    return torch.where(
        tensor >= float(_ACESCC_MAX_ENCODED), torch.full_like(tensor, 65504.0), out)


# ── DaVinci Intermediate ──────────────────────────────────────────────────────

# Blackmagic DaVinci Intermediate, per the published specification.
#
# The previous parameterisation here (A=0.24928, B=444.8616, C=0.0139, D=0.4,
# with a fitted toe slope/intercept) was wrong in every measurable respect:
# 18% grey encoded to 0.874523 instead of 0.336043, black encoded to 0.345557
# instead of 0, the two branches disagreed by 0.064 at the cut, and the decode
# was consequently non-monotonic -- real DI footage at 18% grey decoded to
# -0.003 linear. color/luts.py already carried the correct formulation; this is
# now the single source and luts.py delegates to it.
_DI_A = 0.0075          # linear offset inside the log
_DI_B = 7.0             # log2 offset
_DI_C = 0.07329248      # log scale
_DI_M = 10.44426855     # toe slope
_DI_LIN_CUT = 0.00262409
_DI_CUT_ENCODED = _DI_LIN_CUT * _DI_M   # 0.02740668...


def linear_to_davinci_intermediate(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    out = np.empty_like(img, dtype=np.float32)
    mask = img > _DI_LIN_CUT
    out[mask] = _DI_C * (np.log2(np.maximum(img[mask] + _DI_A, 1e-10)) + _DI_B)
    out[~mask] = img[~mask] * _DI_M
    return out


def davinci_intermediate_to_linear(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    out = np.empty_like(img, dtype=np.float32)
    mask = img > _DI_CUT_ENCODED
    out[mask] = np.power(2.0, img[mask] / _DI_C - _DI_B) - _DI_A
    out[~mask] = img[~mask] / _DI_M
    return out


def tensor_linear_to_davinci_intermediate(tensor: torch.Tensor) -> torch.Tensor:
    log_val = _DI_C * (torch.log2((tensor + _DI_A).clamp(min=1e-10)) + _DI_B)
    lin_val = tensor * _DI_M
    return torch.where(tensor > _DI_LIN_CUT, log_val, lin_val)


def tensor_davinci_intermediate_to_linear(tensor: torch.Tensor) -> torch.Tensor:
    log_val = torch.pow(2.0, tensor / _DI_C - _DI_B) - _DI_A
    lin_val = tensor / _DI_M
    return torch.where(tensor > _DI_CUT_ENCODED, log_val, lin_val)


# ── HDR Transfer Functions (PQ / HLG) ─────────────────────────────────────────

def linear_to_pq(img: np.ndarray, peak_nits: float = 1000.0) -> np.ndarray:
    L = np.clip(img * peak_nits / 10000.0, 0, 1)
    m1, m2 = 0.1593017578125, 78.84375
    c1, c2, c3 = 0.8359375, 18.8515625, 18.6875
    Lm1 = np.power(L, m1)
    return np.power((c1 + c2 * Lm1) / (1 + c3 * Lm1), m2)


def pq_to_linear(img: np.ndarray, peak_nits: float = 1000.0) -> np.ndarray:
    m1, m2 = 0.1593017578125, 78.84375
    c1, c2, c3 = 0.8359375, 18.8515625, 18.6875
    Vm2 = np.power(np.maximum(img, 0), 1 / m2)
    return np.power(np.maximum(Vm2 - c1, 0) / (c2 - c3 * Vm2), 1 / m1) * 10000.0 / peak_nits


def linear_to_hlg(img: np.ndarray) -> np.ndarray:
    a, b, c = 0.17883277, 0.28466892, 0.55991073
    out = np.where(
        img <= 1 / 12,
        np.sqrt(3 * np.maximum(img, 0)),
        a * np.log(np.maximum(12 * img - b, 1e-10)) + c,
    )
    return np.clip(out, 0, 1)


def hlg_to_linear(img: np.ndarray) -> np.ndarray:
    a, b, c = 0.17883277, 0.28466892, 0.55991073
    out = np.where(img <= 0.5, (img**2) / 3, (np.exp((img - c) / a) + b) / 12)
    return np.maximum(out, 0)


def tensor_linear_to_pq(tensor: torch.Tensor, peak_nits: float = 1000.0) -> torch.Tensor:
    L = torch.clamp(tensor * peak_nits / 10000.0, min=0.0)
    m1 = torch.tensor(0.1593017578125, dtype=tensor.dtype, device=tensor.device)
    m2 = torch.tensor(78.84375, dtype=tensor.dtype, device=tensor.device)
    c1, c2, c3 = 0.8359375, 18.8515625, 18.6875
    Lm1 = torch.pow(L, m1)
    return torch.pow((c1 + c2 * Lm1) / (1.0 + c3 * Lm1), m2)


def tensor_pq_to_linear(tensor: torch.Tensor, peak_nits: float = 1000.0) -> torch.Tensor:
    m1 = torch.tensor(0.1593017578125, dtype=tensor.dtype, device=tensor.device)
    m2 = torch.tensor(78.84375, dtype=tensor.dtype, device=tensor.device)
    c1, c2, c3 = 0.8359375, 18.8515625, 18.6875
    Vm2 = torch.pow(torch.clamp(tensor, min=0.0), 1.0 / m2)
    return torch.pow(torch.clamp(Vm2 - c1, min=0.0) / (c2 - c3 * Vm2), 1.0 / m1) * 10000.0 / peak_nits


def tensor_linear_to_hlg(tensor: torch.Tensor) -> torch.Tensor:
    a, b, c = 0.17883277, 0.28466892, 0.55991073
    t = torch.clamp(tensor, min=0.0)
    lin_val = torch.sqrt(3.0 * t)
    log_val = a * torch.log(torch.clamp(12.0 * t - b, min=1e-10)) + c
    return torch.clamp(torch.where(t <= 1.0 / 12.0, lin_val, log_val), 0.0, 1.0)


def tensor_hlg_to_linear(tensor: torch.Tensor) -> torch.Tensor:
    a, b, c = 0.17883277, 0.28466892, 0.55991073
    lin_val = (tensor ** 2) / 3.0
    log_val = (torch.exp((tensor - c) / a) + b) / 12.0
    return torch.clamp(torch.where(tensor <= 0.5, lin_val, log_val), min=0.0)
