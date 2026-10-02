"""What the Viewer's input pixels are: shared by the Viewer node and the
delivery export (FIX-018), so both resolve a source the same way."""
from __future__ import annotations

from typing import Any, Tuple

import torch

VIEWER_INPUT_SPACES = [
    "Auto",
    "sRGB (ComfyUI IMAGE)",
    "Linear Rec.709 (sRGB)",
    "ACEScg",
    "Linear Rec.2020",
    "Linear P3-D65",
    "ACES2065-1",
]
SRGB_COLORSPACE = "sRGB Encoded Rec.709 (sRGB)"


def resolve_viewer_input_space(choice: str, image: Any) -> Tuple[str, str]:
    """-> (encoding "srgb" | "linear", OCIO colour-space name).

    Auto: a ComfyUI IMAGE is display-encoded sRGB by convention and lives in
    0-1. Values above 1.0 or below 0 only come from scene-linear producers, so
    they mark the batch linear (Rec.709 primaries, Radiance's working space).
    The decision is per batch, never per frame, so a clip cannot flip between
    two looks when one frame crosses 1.0.
    """
    if choice and choice != "Auto":
        if choice.startswith("sRGB"):
            return "srgb", SRGB_COLORSPACE
        return "linear", choice
    try:
        t = image if isinstance(image, torch.Tensor) else None
        if t is not None and t.numel():
            rgb = t[..., :3] if t.shape[-1] >= 3 else t
            hi = float(rgb.amax())
            lo = float(rgb.amin())
            if hi > 1.0 + 1e-3 or lo < -1e-3:
                return "linear", "Linear Rec.709 (sRGB)"
    except Exception:  # noqa: BLE001 - fall back to the ComfyUI convention
        pass
    return "srgb", SRGB_COLORSPACE
