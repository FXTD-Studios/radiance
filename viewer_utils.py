"""
viewer_utils.py — BACKWARD-COMPATIBILITY RE-EXPORT SHIM
======================================================
This module preserves the legacy interface for viewer utility consumers.
Internally, functions are modularized under:
- `radiance.cache`
- `radiance.color.luts`
- `radiance.color.grading`
- `radiance.color.analysis`
- `radiance.io.formats`

New code should import from those submodules directly.
"""
import logging
from typing import Dict, Any, Optional, List, Tuple

# Re-import modular components
from radiance.cache import (
    _VIEWER_CACHE_MAX,
    _VIEWER_CACHE,
    _VIEWER_CACHE_LOCK,
    _viewer_cache_set,
    _viewer_cache_get,
    _VIEWER_PROGRESS_MAX,
    _VIEWER_PROGRESS,
    _VIEWER_PROGRESS_LOCK,
    _progress_set,
    _progress_get,
)

from radiance.color.luts import (
    LUT_MODES,
    _M_SRGB_TO_ACESCG,
    _M_SRGB_TO_AP0,
    _M_ACESCG_TO_LIN_SRGB,
    _M_ACESCG_TO_AP0,
    _lut_srgb,
    _lut_rec709,
    _lut_filmic,
    _lut_reinhard,
    _lut_aces_filmic,
    _lut_acescg,
    _lut_aces2065,
    _lut_acescct_encode,
    _lut_logc3,
    _lut_logc4,
    _lut_slog3,
    _lut_vlog,
    _lut_flog2,
    _lut_clog3,
    _lut_log3g10,
    _lut_davinci_intermediate,
    _lut_bmd_gen5,
    _lut_nlog,
    _lut_lin_to_log,
    _lut_log_to_lin,
    _idt_logc3,
    _idt_logc4,
    _idt_slog3,
    _idt_vlog,
    _idt_flog2,
    _idt_clog3,
    _idt_log3g10,
    _idt_davinci_intermediate,
    _idt_bmd_gen5,
    _idt_nlog,
    _lut_passthrough,
    _lut_false_color,
    _lut_clip_check,
    _LUT_FUNCTIONS,
)

from radiance.color.grading import (
    apply_lut,
    apply_grading,
    _kelvin_to_rgb_multipliers,
)

from radiance.color.analysis import (
    safe_tensor_to_numpy,
    compute_data_range,
)

from radiance.io.formats import (
    HAS_CV2,
    cv2,
    PICK_MAX_DIM,
    RPICK_MAGIC,
    CV2_PNG_COMPRESSION,
    _save_pick_buffer,
    build_cdl_xml,
    save_16bit_png,
)

# ═══════════════════════════════════════════════════════════════════════════════
#                           LEGACY CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_PNG_COMPRESSION = 4
BIT_16_MAX = 65535
BIT_8_MAX = 255
BIT_16_TO_8_DIVISOR = 257

MAX_IMAGE_DIMENSION = 16384
MAX_BATCH_SIZE = 9999
RHDR_MAGIC = b"RHDR"

BIT_DEPTH_MODES = ["8-bit (Fast)", "16-bit (Quality)", "16-bit + HDR Data", "32-bit Float"]


# ═══════════════════════════════════════════════════════════════════════════════
#                           LEGACY TYPE REPRESENTATIONS
# ═══════════════════════════════════════════════════════════════════════════════

class RadianceType(str):
    """
    A specific multi-type class that leverages LiteGraph's native comma-separated type parsing
    on the frontend ("IMAGE,VIDEO") while bypassing Python backend validation.
    """
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Utilities"
    def __ne__(self, __value: object) -> bool:
        return False

# Expose as a multi-type that specifically accepts IMAGE and VIDEO in the UI
image_video_type = RadianceType("IMAGE,VIDEO")
