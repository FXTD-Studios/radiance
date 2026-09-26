"""Color science package — matrices, transfer functions, gamut ops, LUT nodes.

Usage:
    from radiance.color.transfer import tensor_logc4_to_linear
    from radiance.color.matrices import SRGB_TO_ACESCG
    from radiance.color.gamut import aces2_gamut_compress
"""
from radiance.color.lut import RadianceLUTApply, RadianceLUTBlend, LUTCache

# Re-export key types for convenience
from radiance.color.matrices import (
    SRGB_TO_ACESCG, ACESCG_TO_SRGB,
    AWG3_TO_ACESCG, AWG4_TO_ACESCG,
    SGAMUT3_CINE_TO_ACESCG, VGAMUT_TO_ACESCG,
    REDWIDEGAMUT_TO_ACESCG, DAVINCI_WIDE_TO_ACESCG,
    CINEMA_GAMUT_TO_ACESCG,
    apply_matrix_transform,
    linear_srgb_to_acescg, acescg_to_linear_srgb,
)
from radiance.color.transfer import (
    tensor_srgb_to_linear, tensor_linear_to_srgb,
    tensor_logc4_to_linear, tensor_linear_to_logc4,
    tensor_logc3_to_linear, tensor_linear_to_logc3,
    tensor_slog3_to_linear, tensor_linear_to_slog3,
    tensor_vlog_to_linear, tensor_linear_to_vlog,
    tensor_canonlog3_to_linear, tensor_linear_to_canonlog3,
    tensor_log3g10_to_linear, tensor_linear_to_log3g10,
    tensor_acescct_to_linear, tensor_linear_to_acescct,
    tensor_davinci_intermediate_to_linear, tensor_linear_to_davinci_intermediate,
    tensor_linear_to_pq, tensor_pq_to_linear,
    tensor_linear_to_hlg, tensor_hlg_to_linear,
    srgb_to_linear, linear_to_srgb,
    logc4_to_linear, linear_to_logc4,
    logc3_to_linear, linear_to_logc3,
    slog3_to_linear, linear_to_slog3,
    vlog_to_linear, linear_to_vlog,
    canonlog3_to_linear, linear_to_canonlog3,
    log3g10_to_linear, linear_to_log3g10,
    acescct_to_linear, linear_to_acescct,
    davinci_intermediate_to_linear, linear_to_davinci_intermediate,
    linear_to_pq, pq_to_linear,
    linear_to_hlg, hlg_to_linear,
)
from radiance.color.gamut import (
    aces_tonemap, aces_approx_tonemap, aces2_tonemap,
    aces2_gamut_compress,
    linear_to_jmh, jmh_to_linear,
)
from radiance.color.pipeline import (
    INPUT_COLORSPACES, apply_input_transform, apply_output_transform,
)
from radiance.color.luma import (
    luma_bt709, luma_bt709_tensor, luma_rec2020_tensor,
)

NODE_CLASS_MAPPINGS = {
    "RadianceLUTApply": RadianceLUTApply,
    "RadianceLUTBlend": RadianceLUTBlend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceLUTApply": "◎ Radiance LUT Apply",
    "RadianceLUTBlend": "◎ Radiance LUT Blend",
}
