"""Input/output colorspace helpers for the HDR pipeline.

These functions are shared by engine nodes (apply_input_transform / apply_output_transform)
and are re-exported from the top-level color_utils.py for backward compatibility.
"""
from __future__ import annotations

import numpy as np
import torch

from radiance.color.transfer import (
    logc3_to_linear, linear_to_logc3,
    logc4_to_linear, linear_to_logc4,
    slog3_to_linear, linear_to_slog3,
    vlog_to_linear, linear_to_vlog,
    davinci_intermediate_to_linear, linear_to_davinci_intermediate,
    acescct_to_linear, linear_to_acescct,
    tensor_srgb_to_linear, tensor_linear_to_srgb,
)
from radiance.color.ops import (
    apply_matrix_3x3, M_REC709_TO_ACESCG, M_ACESCG_TO_REC709,
)

INPUT_COLORSPACES: list[str] = [
    "Linear (sRGB)",
    "sRGB (Standard)",
    "ARRI LogC3",
    "ARRI LogC4",
    "Sony S-Log3",
    "Panasonic V-Log",
    "DaVinci Intermediate",
    "ACEScg",
    "ACEScct",
]

_NUMPY_DECODE_MAP = {
    "ARRI LogC3": logc3_to_linear,
    "ARRI LogC4": logc4_to_linear,
    "Sony S-Log3": slog3_to_linear,
    "Panasonic V-Log": vlog_to_linear,
    "DaVinci Intermediate": davinci_intermediate_to_linear,
    "ACEScct": acescct_to_linear,
}

_NUMPY_ENCODE_MAP = {
    "ARRI LogC3": linear_to_logc3,
    "ARRI LogC4": linear_to_logc4,
    "Sony S-Log3": linear_to_slog3,
    "Panasonic V-Log": linear_to_vlog,
    "DaVinci Intermediate": linear_to_davinci_intermediate,
    "ACEScct": linear_to_acescct,
}


def apply_input_transform(img_tensor: torch.Tensor, colorspace: str) -> torch.Tensor:
    if colorspace == "Linear (sRGB)":
        return img_tensor
    if colorspace == "ACEScg":
        # ACEScg is a different gamut (AP1), not just a curve — convert primaries
        # back to the linear sRGB/Rec.709 working space.
        return apply_matrix_3x3(img_tensor, M_ACESCG_TO_REC709)
    if colorspace == "sRGB (Standard)":
        return tensor_srgb_to_linear(img_tensor)

    device = img_tensor.device
    img_np = img_tensor.cpu().numpy()
    fn = _NUMPY_DECODE_MAP.get(colorspace)
    out_np = fn(img_np) if fn else img_np
    return torch.from_numpy(out_np).to(device)


def apply_output_transform(
    img_tensor: torch.Tensor,
    colorspace: str,
    broadcast_safe: bool = False,
) -> torch.Tensor:
    is_display = colorspace == "sRGB (Standard)"
    if broadcast_safe and is_display:
        a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
        x = img_tensor
        img_tensor = torch.clamp(
            (x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0
        )

    if colorspace == "Linear (sRGB)":
        return img_tensor
    if colorspace == "ACEScg":
        # Convert the linear sRGB/Rec.709 working primaries into ACEScg (AP1).
        # Previously ACEScg was a no-op passthrough, so the data was mislabeled
        # as ACEScg without the gamut matrix — wrong colors in that space only.
        return apply_matrix_3x3(img_tensor, M_REC709_TO_ACESCG)
    if colorspace == "sRGB (Standard)":
        return tensor_linear_to_srgb(img_tensor)

    device = img_tensor.device
    img_np = img_tensor.cpu().numpy()
    fn = _NUMPY_ENCODE_MAP.get(colorspace)
    out_np = fn(img_np) if fn else img_np
    return torch.from_numpy(out_np).to(device)
