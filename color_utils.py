"""
Backward-compatible re-export shim for color_utils.

All functions and constants now live in `radiance.color.*` modules.
This shim re-exports everything so existing imports continue to work.
New code should import from `radiance.color` directly.
"""
from __future__ import annotations

import warnings
warnings.warn(
    "Import from radiance.color_utils is deprecated. Use radiance.color.* modules directly.",
    DeprecationWarning, stacklevel=2,
)

# ── Matrices ────────────────────────────────────────────────────────────────
from radiance.color.matrices import (
    SRGB_TO_ACESCG,
    ACESCG_TO_SRGB,
    ACES_AP0_TO_AP1,
    ACESCG_TO_REC2020,
    ACESCG_TO_P3D65,
    AWG3_TO_ACESCG,
    AWG4_TO_ACESCG,
    SGAMUT3_CINE_TO_ACESCG,
    VGAMUT_TO_ACESCG,
    CINEMA_GAMUT_TO_ACESCG,
    REDWIDEGAMUT_TO_ACESCG,
    DAVINCI_WIDE_TO_ACESCG,
    HPE_MATRIX,
    HPE_MATRIX_INV,
    apply_matrix_transform,
    linear_srgb_to_acescg,
    acescg_to_linear_srgb,
    SRGB_TO_REC2020,
    REC2020_TO_SRGB,
    linear_srgb_to_rec2020,
    rec2020_to_linear_srgb,
)

# ── Transfer functions (numpy) ──────────────────────────────────────────────
from radiance.color.transfer import (
    srgb_to_linear,
    linear_to_srgb,
    linear_to_logc3,
    logc3_to_linear,
    linear_to_logc4,
    logc4_to_linear,
    linear_to_slog3,
    slog3_to_linear,
    linear_to_vlog,
    vlog_to_linear,
    linear_to_canonlog3,
    canonlog3_to_linear,
    linear_to_log3g10,
    log3g10_to_linear,
    linear_to_acescct,
    acescct_to_linear,
    linear_to_davinci_intermediate,
    davinci_intermediate_to_linear,
    linear_to_pq,
    pq_to_linear,
    linear_to_hlg,
    hlg_to_linear,
    linear_to_rec709,
    rec709_to_linear,
    linear_to_rec2020,
    rec2020_to_linear,
)

# ── Transfer functions (torch) ──────────────────────────────────────────────
from radiance.color.transfer import (
    tensor_srgb_to_linear,
    tensor_linear_to_srgb,
    tensor_linear_to_logc3,
    tensor_logc3_to_linear,
    tensor_linear_to_logc4,
    tensor_logc4_to_linear,
    tensor_linear_to_slog3,
    tensor_slog3_to_linear,
    tensor_linear_to_vlog,
    tensor_vlog_to_linear,
    tensor_linear_to_canonlog3,
    tensor_canonlog3_to_linear,
    tensor_linear_to_log3g10,
    tensor_log3g10_to_linear,
    tensor_linear_to_acescct,
    tensor_acescct_to_linear,
    tensor_linear_to_davinci_intermediate,
    tensor_davinci_intermediate_to_linear,
    tensor_linear_to_pq,
    tensor_pq_to_linear,
    tensor_linear_to_hlg,
    tensor_hlg_to_linear,
    tensor_linear_to_rec709,
    tensor_rec709_to_linear,
    tensor_linear_to_rec2020,
    tensor_rec2020_to_linear,
)

# ── Gamut ops ───────────────────────────────────────────────────────────────
from radiance.color.gamut import (
    aces_tonemap,
    aces_approx_tonemap,
    aces2_tonemap,
    aces2_gamut_compress,
    linear_to_jmh,
    jmh_to_linear,
)

# ── Pipeline ────────────────────────────────────────────────────────────────
from radiance.color.pipeline import (
    INPUT_COLORSPACES,
    apply_input_transform,
    apply_output_transform,
)

# ── Luma ────────────────────────────────────────────────────────────────────
from radiance.color.luma import (
    luma_bt709,
    luma_bt709_tensor,
    luma_rec2020_tensor,
)

# ── Tensor/numpy conversions ────────────────────────────────────────────────
from radiance.core.tensor.convert import (
    tensor_to_numpy,
    numpy_to_tensor,
    tensor_to_numpy_float32,
    numpy_to_tensor_float32,
)
