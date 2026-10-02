"""ACES 2.0 Output Transforms through the official OCIO implementation.

Radiance's own ACES 2.0 math (Daniele Evo tone scale on luminance plus a reach
gamut compressor) is an approximation of the Output Transform: the reference
works in JMh, with chroma compression and gamut mapping that the
approximation does not have. Production renders go through the Academy
reference as implemented in OpenColorIO, pinned to one built-in config so the
result does not change with whatever OCIO config the user has loaded.

Pinned config: ``studio-config-v4.0.0_aces-v2.0_ocio-v2.5`` (OCIO >= 2.5).
"""
from __future__ import annotations

import functools
import logging
from typing import Optional, Tuple

import numpy as np

log = logging.getLogger("radiance.aces2")

PINNED_CONFIG = "ocio://studio-config-v4.0.0_aces-v2.0_ocio-v2.5"

# Node input colorspace -> colour space name in the pinned config.
INPUT_SPACES = {
    "ACEScg": "ACEScg",
    "ACES2065-1": "ACES2065-1",
    "Linear_sRGB": "Linear Rec.709 (sRGB)",
    "Linear_Rec2020": "Linear Rec.2020",
}

# Node output transform -> (display, view) in the pinned config. Outputs with
# no pairing in the config (DCI-P3 D60 cinema) are left out on purpose; they
# can only run on the labelled approximation.
OUTPUT_VIEWS = {
    "ACES 2.0 SDR (sRGB/Rec.709)": ("sRGB - Display", "ACES 2.0 - SDR 100 nits (Rec.709)"),
    "ACES 2.0 SDR (P3-D65)": ("Display P3 - Display", "ACES 2.0 - SDR 100 nits (P3 D65)"),
    "ACES 2.0 HDR (Rec.2100 PQ 1000 nits)": ("Rec.2100-PQ - Display", "ACES 2.0 - HDR 1000 nits (Rec.2020)"),
    "ACES 2.0 HDR (Rec.2100 PQ 2000 nits)": ("Rec.2100-PQ - Display", "ACES 2.0 - HDR 2000 nits (Rec.2020)"),
    "ACES 2.0 HDR (Rec.2100 PQ 4000 nits)": ("Rec.2100-PQ - Display", "ACES 2.0 - HDR 4000 nits (Rec.2020)"),
    "ACES 2.0 HDR (Rec.2100 HLG)": ("Rec.2100-HLG - Display", "ACES 2.0 - HDR 1000 nits (P3 D65)"),
    "ACES 2.0 Cinema (DCI-P3 D65)": ("P3-D65 - Display", "ACES 2.0 - SDR 100 nits (P3 D65)"),
}


class ACES2ReferenceUnavailable(RuntimeError):
    """The pinned OCIO ACES 2.0 config cannot be used for this request."""


@functools.lru_cache(maxsize=1)
def _config():
    try:
        import PyOpenColorIO as ocio
    except ImportError as exc:
        raise ACES2ReferenceUnavailable(
            "OpenColorIO is not installed (pip install 'opencolorio>=2.5')."
        ) from exc
    try:
        return ocio.Config.CreateFromFile(PINNED_CONFIG)
    except Exception as exc:  # noqa: BLE001 - re-raised with the reason
        raise ACES2ReferenceUnavailable(
            f"OpenColorIO {getattr(ocio, '__version__', '?')} has no built-in "
            f"{PINNED_CONFIG!r}; ACES 2.0 reference transforms need OCIO >= 2.5. ({exc})"
        ) from exc


def reference_available(output_transform: Optional[str] = None) -> Tuple[bool, str]:
    """``(True, "")`` when the reference can run, else ``(False, reason)``."""
    if output_transform is not None and output_transform not in OUTPUT_VIEWS:
        return False, f"{output_transform!r} has no ACES 2.0 view in the pinned OCIO config"
    try:
        _config()
    except ACES2ReferenceUnavailable as exc:
        return False, str(exc)
    return True, ""


@functools.lru_cache(maxsize=32)
def _processor(input_colorspace: str, output_transform: str):
    import PyOpenColorIO as ocio
    if input_colorspace not in INPUT_SPACES:
        raise ACES2ReferenceUnavailable(f"Unknown input colour space {input_colorspace!r}.")
    if output_transform not in OUTPUT_VIEWS:
        raise ACES2ReferenceUnavailable(
            f"{output_transform!r} has no ACES 2.0 view in the pinned OCIO config.")
    display, view = OUTPUT_VIEWS[output_transform]
    cfg = _config()
    proc = cfg.getProcessor(INPUT_SPACES[input_colorspace], display, view,
                            ocio.TRANSFORM_DIR_FORWARD)
    return proc.getDefaultCPUProcessor()


def apply_reference(rgb: np.ndarray, input_colorspace: str, output_transform: str) -> np.ndarray:
    """Run the ACES 2.0 Output Transform on ``rgb`` (..., 3) float32.

    Returns display-encoded values (the display's EOTF inverse applied), as a
    new array. Raises ``ACES2ReferenceUnavailable`` when it cannot run.
    """
    cpu = _processor(input_colorspace, output_transform)
    out = np.ascontiguousarray(rgb, dtype=np.float32).copy()
    flat = out.reshape(-1, 3)
    cpu.applyRGB(flat)
    return flat.reshape(out.shape)


def describe(output_transform: str) -> str:
    display, view = OUTPUT_VIEWS[output_transform]
    return f"OCIO {PINNED_CONFIG.split('//')[1]} · {display} / {view}"


__all__ = [
    "PINNED_CONFIG", "INPUT_SPACES", "OUTPUT_VIEWS", "ACES2ReferenceUnavailable",
    "reference_available", "apply_reference", "describe",
]
