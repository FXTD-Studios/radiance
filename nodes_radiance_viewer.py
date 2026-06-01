import json
import struct
import torch
import numpy as np
import zlib
import os
import uuid
import logging
import math
import sys
import io
import traceback
from aiohttp import web
from server import PromptServer
import re

# v3.1: Enable OpenEXR support in OpenCV
# Essential for 32-bit float export
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
from typing import Dict, Any, Optional, List, Tuple

import folder_paths

# Import safe path utilities
try:
    from .path_utils import safe_join
except ImportError:
    from path_utils import safe_join

# v3.1: Robust EXR Writer from HDR bridge
try:
    from .hdr.io import write_exr_robust
except ImportError:
    try:
        from hdr.io import write_exr_robust
    except ImportError:
        write_exr_robust = None

# v1.21: cv2 for 16-bit PNG support
try:
    import cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    logging.getLogger("◎ Radiance.viewer").warning(
        "cv2 not available — 16-bit PNG disabled, falling back to 8-bit. "
        "Install opencv-python for 16-bit support."
    )

# Module logger
logger = logging.getLogger("◎ Radiance.viewer")

# Import shared ACES gamut compression — replaces inline duplicate implementation
try:
    from .color_utils import aces2_gamut_compress as _aces2_gamut_compress
except ImportError:
    try:
        from color_utils import aces2_gamut_compress as _aces2_gamut_compress
    except ImportError:
        _aces2_gamut_compress = None


# ═══════════════════════════════════════════════════════════════════════════════
#                           CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_PNG_COMPRESSION = 4
BIT_16_MAX = 65535
BIT_8_MAX = 255
BIT_16_TO_8_DIVISOR = 257  # Correct: 65535 / 255 = 257

MAX_IMAGE_DIMENSION = 16384
MAX_BATCH_SIZE = 100

# ── v3.0 Constants ────────────────────────────────────────────────────────────
PICK_MAX_DIM = 256          # fp32 picking buffer max dimension (per side)
RHDR_MAGIC = b"RHDR"       # fp16 display sidecar magic
RPICK_MAGIC = b"RPIC"      # fp32 picking sidecar magic (v3.0 NEW)

# Try to import imageio for EXR fallback
try:
    import imageio

    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

BIT_DEPTH_MODES = ["8-bit (Fast)", "16-bit (Quality)", "16-bit + HDR Data", "32-bit Float"]
CV2_PNG_COMPRESSION = 4

# v3.1: OCIO Integration — auto-registers /radiance/ocio/* HTTP routes
try:
    from .radiance_ocio import get_ocio_manager, HAS_OCIO, apply_ocio_transform
    if HAS_OCIO:
        logger.info("[Radiance] OCIO engine loaded — /radiance/ocio/* endpoints active")
except ImportError:
    HAS_OCIO = False
    logger.info("[Radiance] OCIO module not found — OCIO features disabled")


# Stores the last processed IMAGE batch for each node to allow HUD-based export.
# v3.1 FIX (B-3): LRU-bounded cache to prevent OOM on long sessions.
# Max 8 nodes cached — evicts oldest entry when full.
import collections
import threading

_VIEWER_CACHE_MAX = 8
_VIEWER_CACHE: collections.OrderedDict = collections.OrderedDict()
_VIEWER_CACHE_LOCK = threading.Lock()


def _viewer_cache_set(key: str, value: torch.Tensor) -> None:
    """Thread-safe LRU insert into the viewer cache."""
    with _VIEWER_CACHE_LOCK:
        if key in _VIEWER_CACHE:
            _VIEWER_CACHE.move_to_end(key)
        _VIEWER_CACHE[key] = value
        while len(_VIEWER_CACHE) > _VIEWER_CACHE_MAX:
            evicted_key, evicted_val = _VIEWER_CACHE.popitem(last=False)
            logger.debug(f"[Radiance] Cache evicted node {evicted_key} (capacity {_VIEWER_CACHE_MAX})")
            del evicted_val  # help GC release tensor


def _viewer_cache_get(key: str) -> Optional[torch.Tensor]:
    """Thread-safe LRU lookup."""
    with _VIEWER_CACHE_LOCK:
        if key in _VIEWER_CACHE:
            _VIEWER_CACHE.move_to_end(key)
            return _VIEWER_CACHE[key]
        return None


# Stores active export progress: {instance_id: {current, total, status, message}}
_VIEWER_PROGRESS: Dict[str, Dict[str, Any]] = {}



# ═══════════════════════════════════════════════════════════════════════════════
#                v3.0 HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def _save_pick_buffer(
    frame: np.ndarray,
    filepath: str,
    max_dim: int = PICK_MAX_DIM,
) -> bool:
    """Save a scene-linear fp32 picking buffer (.rpick) for the color picker.

    Format: 12-byte header  + zlib-compressed IEEE 754 float32 bytes
    Header: magic(4) + width(H) + height(H) + channels(H) + flags(H)
    Flags:  bit0=fp32 (always set), bit1=downsampled

    The buffer is downsampled to max_dim on the longest side so it's cheap
    to fetch (256px ≈ 4 × 256 × 256 × 4 = 256KB uncompressed → ~30KB zlib).
    Float precision is kept full (fp32) so color picker reports true HDR values.
    """
    try:
        h, w = frame.shape[:2]
        c = frame.shape[2] if frame.ndim == 3 else 1
        rgb = frame[..., :3] if c >= 3 else frame

        # Downsample if larger than max_dim
        flags = 1  # bit0 = fp32
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            if HAS_CV2:
                rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                # Simple numpy area-average downsampling fallback
                from PIL import Image as PILImage
                pil = PILImage.fromarray(
                    np.clip(rgb, 0.0, 1.0).astype(np.float32), mode="RGB"
                    if c >= 3 else "L"
                )
                pil = pil.resize((new_w, new_h), PILImage.LANCZOS)
                rgb = np.array(pil, dtype=np.float32)
            h, w = rgb.shape[:2]
            c = rgb.shape[2] if rgb.ndim == 3 else 1
            flags |= 2  # bit1 = downsampled

        buf = rgb.astype(np.float32).tobytes()
        compressed = zlib.compress(buf, level=3)  # level 3 = fast
        header = struct.pack("<4sHHHH", RPICK_MAGIC, w, h, c, flags)
        with open(filepath, "wb") as f:
            f.write(header)
            f.write(compressed)
        return True
    except Exception as e:
        logger.debug(f"[Radiance v3.0] pick buffer save failed: {e}")
        return False


def build_cdl_xml(
    slope: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    power: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    saturation: float = 1.0,
    description: str = "◎ Radiance Viewer Grade",
) -> str:
    """Build an ASC CDL XML string compatible with Nuke, DaVinci Resolve, OCIO.

    ASC CDL v1.2 spec: https://kencolorist.com/p/the-asc-cdl
    SOP order: out = CLAMP( (in * Slope + Offset) ^ Power )
    """
    s = " ".join(f"{v:.6f}" for v in slope)
    o = " ".join(f"{v:.6f}" for v in offset)
    p = " ".join(f"{v:.6f}" for v in power)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<ColorDecisionList xmlns="urn:ASC:CDL:v1.2">\n'
        '  <ColorDecision>\n'
        f'    <!-- {description} -->\n'
        '    <ColorCorrection id="◎ Radiance_grade">\n'
        '      <SOPNode>\n'
        f'        <Slope>{s}</Slope>\n'
        f'        <Offset>{o}</Offset>\n'
        f'        <Power>{p}</Power>\n'
        '      </SOPNode>\n'
        '      <SatNode>\n'
        f'        <Saturation>{saturation:.6f}</Saturation>\n'
        '      </SatNode>\n'
        '    </ColorCorrection>\n'
        '  </ColorDecision>\n'
        '</ColorDecisionList>\n'
    )


# ═══════════════════════════════════════════════════════════════════════════════
#                        BUILT-IN LUT ENGINE v2.3.3
#
#  Analytical LUTs — no file dependencies.
#  Each LUT is a function: float32 linear [0..1+] → float32 [0..1]
#  Applied per-channel in linear space unless noted otherwise.
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
]


# ═══════════════════════════════════════════════════════════════════════════════
#                    CAMERA LOG LUT FUNCTIONS  (v3.2 — Spec-Accurate)
#
#  All forward transforms:
#    • Input: scene-linear float32, 18% grey = 0.18
#    • Output: log-encoded float, clipped to [0, 1]
#
#  Verified 18% grey output values (normalized 0–1):
#    LogC3:   0.391  |  LogC4:  0.278  |  S-Log3:  0.411
#    V-Log:   0.423  |  F-Log2: 0.391  |  C-Log3:  0.343
#    Log3G10: 0.333  |  DaVinci: 0.336 |  BMD Gen5: 0.384 | N-Log: 0.364
#
#  All IDT functions are exact inverses of their forward counterpart.
#  Sources: manufacturer white papers, ACES CLF reference, OCIO configs.
# ═══════════════════════════════════════════════════════════════════════════════


def _lut_srgb(x: np.ndarray) -> np.ndarray:
    """Linear → sRGB gamma  [IEC 61966-2-1]."""
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
    """Linear → Rec.709 OETF  [ITU-R BT.709-6]."""
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

    white_scale = 1.0 / hable(np.array(11.2))
    return np.clip(hable(np.maximum(x, 0.0) * 2.0) * white_scale, 0.0, 1.0)


def _lut_reinhard(x: np.ndarray) -> np.ndarray:
    """Reinhard global tonemapping."""
    return np.clip(x / (1.0 + x), 0.0, 1.0)


def _lut_aces_filmic(x: np.ndarray) -> np.ndarray:
    """ACES Filmic tone mapping (Narkowicz fit)."""
    x = np.maximum(x, 0.0)
    return np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0)


# ── Camera Log Encoding ───────────────────────────────────────────────────────

# ═══════════════════════════════════════════════════════════════════════════════
#        CAMERA LOG LUT FUNCTIONS  (v3.3 — Spec-Accurate)
#
#  All forward transforms (Linear → Log):
#    • Input:  scene-linear float32, 18% grey = 0.18
#    • Output: log-encoded float, clipped to [0, 1]
#
#  Authoritative 18% grey output values (verified against colour-science
#  library, which cites manufacturer specification documents directly):
#    LogC3:   0.391  |  LogC4:  0.278  |  S-Log3:  0.411
#    V-Log:   0.423  |  F-Log2: 0.391  |  C-Log3:  0.343
#    Log3G10: 0.333  |  DaVinci: 0.336 |  BMD Gen5: 0.384 | N-Log: 0.364
#
#  All IDT functions are exact analytic inverses of their forward counterpart.
#  Sources: colour-science library (https://github.com/colour-science/colour),
#           manufacturer white papers, ACES CLF reference, OCIO configs.
# ═══════════════════════════════════════════════════════════════════════════════


def _lut_logc3(x: np.ndarray) -> np.ndarray:
    """Linear → ARRI LogC3 EI800.
    Source: colour-science DATA_ALEXA_LOG_C_CURVE_CONVERSION['SUP 3.x'][800]
    Verified: 18% grey → 0.391007
    """
    # EI800 piecewise constants
    cut = 0.010591
    a = 5.555556
    b = 0.052272
    c = 0.247190
    d = 0.385537
    e = 5.367655
    f = 0.092809  # linear toe (shadow extension)
    return np.clip(
        np.where(x > cut, c * np.log10(np.maximum(a * x + b, 1e-10)) + d, e * x + f),
        0.0,
        1.0,
    )


def _lut_logc4(x: np.ndarray) -> np.ndarray:
    """Linear → ARRI LogC4 (Alexa 35).
    Source: colour-science CONSTANTS_ARRILOGC4 (ARRI Specification 2022)
    Formula: E_p = (log2(a*E + 64) - 6) / 14 * b + c  for E >= t
             E_p = (E - t) / s                          for E < t
    Verified: 18% grey → 0.278396
    """
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
    """Linear → Sony S-Log3.
    Source: colour-science log_encoding_SLog3 (Sony Specification 2014)
    Formula: (420 + log10((x+0.01)/(0.18+0.01)) * 261.5) / 1023
    Key: normalization by /0.19 — not just log10(x+0.01)
    Verified: 18% grey → 0.410557
    """
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
    """Linear → Panasonic V-Log / V-Log3.
    Source: colour-science CONSTANTS_VLOG (Panasonic Specification 2014)
    Verified: 18% grey → 0.423311
    """
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
    """Linear → Fujifilm F-Log2.
    Source: colour-science CONSTANTS_FLOG2 (Fujifilm Specification 2022)
    Piecewise: c*log10(a*x+b)+d for x >= cut1; e*x+f for x < cut1
    Verified: 18% grey → 0.391007
    """
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
    """Linear → Canon C-Log3 v1.2.
    Source: colour-science log_encoding_CanonLog3_v1_2 (Canon Specification 2020)
    Input rescaled by /0.9 before encoding. Three-segment piecewise.
    Cut points (in rescaled space): lower ≈ -0.009670, upper ≈ 0.014043
    Verified: 18% grey → 0.343389
    """
    xr = x / 0.9  # Canon rescaling
    k = 14.98325
    a = 0.36726845
    lo = -0.009670
    hi = 0.014043
    neg = -a * np.log10(np.maximum(-xr * k + 1.0, 1e-10)) + 0.12783901
    lin = 1.9754798 * xr + 0.12512219
    pos = a * np.log10(np.maximum(xr * k + 1.0, 1e-10)) + 0.12240537
    result = np.where(xr < lo, neg, np.where(xr <= hi, lin, pos))
    return result  # C-Log3 values can be outside [0,1] for extreme inputs


def _lut_log3g10(x: np.ndarray) -> np.ndarray:
    """Linear → RED Log3G10 v2 (REDCINE-X PRO / IPP2 pipeline).
    Source: colour-science log_encoding_Log3G10_v2 (Nattress / RED 2016)
    Formula: sign(x+0.01) * 0.224282 * log10(|x+0.01| * 155.975327 + 1)
    Verified: 18% grey → 0.333333 (exactly 1/3 by design)
    """
    xoff = x + 0.01
    return np.sign(xoff) * 0.224282 * np.log10(np.abs(xoff) * 155.975327 + 1.0)


def _lut_davinci_intermediate(x: np.ndarray) -> np.ndarray:
    """Linear → DaVinci Intermediate.
    Source: colour-science CONSTANTS_DAVINCI_INTERMEDIATE (BMD Specification 2020)
    Formula: DI_C * (log2(L + DI_A) + DI_B)  for L > DI_LIN_CUT
             L * DI_M                          for L <= DI_LIN_CUT
    Verified: 18% grey → 0.336043
    """
    DI_A = 0.0075
    DI_B = 7.0
    DI_C = 0.07329248
    DI_M = 10.44426855
    DI_LIN_CUT = 0.00262409
    return np.where(
        x <= DI_LIN_CUT, x * DI_M, DI_C * (np.log2(np.maximum(x + DI_A, 1e-10)) + DI_B)
    )


def _lut_bmd_gen5(x: np.ndarray) -> np.ndarray:
    """Linear → Blackmagic Film Generation 5.
    Source: colour-science CONSTANTS_BLACKMAGIC_FILM_GENERATION_5 (BMD Specification 2021)
    Formula: A*ln(x+B)+C  for x >= LIN_CUT;  D*x+E  for x < LIN_CUT
    Uses natural log (ln), NOT log10.
    Verified: 18% grey → 0.383562
    """
    A = 0.08692876065491224
    B = 0.005494072432257808
    C = 0.5300133392291939
    D = 8.283605932402494
    E = 0.09246575342465753
    LIN_CUT = 0.005
    return np.where(x >= LIN_CUT, A * np.log(np.maximum(x + B, 1e-10)) + C, D * x + E)


def _lut_nlog(x: np.ndarray) -> np.ndarray:
    """Linear → Nikon N-Log.
    Source: colour-science CONSTANTS_NLOG (Nikon Specification 2018)
    Formula: a*(y+b)^(1/3)  for y < cut1 (cube-root toe)
             c*ln(y)+d      for y >= cut1 (log body)
    Uses natural log (ln), NOT log10.
    Verified: 18% grey → 0.363668
    """
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
    return np.clip(np.log2(np.maximum(x, 1e-10) * 5.55 + 1.0) / np.log2(6.55), 0.0, 1.0)


def _lut_log_to_lin(x: np.ndarray) -> np.ndarray:
    """Generic log → linear (Cineon inverse)."""
    return np.clip((np.power(6.55, np.clip(x, 0.0, 1.0)) - 1.0) / 5.55, 0.0, 1.0)


# ── IDT: Camera Log → Scene Linear ────────────────────────────────────────────
# All IDT functions are exact analytic inverses of their forward counterparts.


def _idt_logc3(x: np.ndarray) -> np.ndarray:
    """IDT: ARRI LogC3 EI800 → Scene Linear (exact inverse of _lut_logc3)."""
    # Code-value at cut: e*cut + f = 5.367655*0.010591 + 0.092809 ≈ 0.149658
    ENC_CUT = 5.367655 * 0.010591 + 0.092809
    log_out = (np.power(10.0, (x - 0.385537) / 0.247190) - 0.052272) / 5.555556
    lin_out = (x - 0.092809) / 5.367655
    return np.maximum(np.where(x > ENC_CUT, log_out, lin_out), 0.0)


def _idt_logc4(x: np.ndarray) -> np.ndarray:
    """IDT: ARRI LogC4 → Scene Linear (exact inverse of _lut_logc4)."""
    a = 2231.82630906768830
    b = 0.90713587487781030
    c = 0.09286412512218964
    s = 0.11359720861058910
    t = -0.01805699611991131
    # Code value at cut t
    lc4_log_cut = (np.log2(a * t + 64.0) - 6.0) / 14.0 * b + c
    log_out = (np.exp2((x - c) / b * 14.0 + 6.0) - 64.0) / a
    lin_out = x * s + t
    return np.where(x >= lc4_log_cut, log_out, lin_out)


def _idt_slog3(x: np.ndarray) -> np.ndarray:
    """IDT: Sony S-Log3 → Scene Linear (exact inverse of _lut_slog3).
    Applies the /0.19 normalization: x = 10^((V*1023-420)/261.5) * 0.19 - 0.01
    """
    # CV at cut: (cut*(171.2102946929-95)/0.01125 + 95)/1023
    CUT_CV = (0.01125 * (171.2102946929 - 95.0) / 0.01125 + 95.0) / 1023.0  # = 95/1023
    log_out = np.power(10.0, (x * 1023.0 - 420.0) / 261.5) * 0.19 - 0.01
    lin_out = (x * 1023.0 - 95.0) * 0.01125 / (171.2102946929 - 95.0)
    return np.maximum(np.where(x >= CUT_CV, log_out, lin_out), 0.0)


def _idt_vlog(x: np.ndarray) -> np.ndarray:
    """IDT: Panasonic V-Log → Scene Linear (exact inverse of _lut_vlog)."""
    return np.maximum(
        np.where(
            x < 0.181,  # CONSTANTS_VLOG.cut2
            (x - 0.125) / 5.6,
            np.power(10.0, (x - 0.598206) / 0.241514) - 0.00873,
        ),
        0.0,
    )


def _idt_flog2(x: np.ndarray) -> np.ndarray:
    """IDT: Fujifilm F-Log2 → Scene Linear (exact inverse of _lut_flog2)."""
    a = 5.555556
    b = 0.064829
    c_c = 0.245281
    d = 0.384316
    e = 8.799461
    f = 0.092864
    CUT_CV = c_c * np.log10(a * 0.000889 + b) + d  # ≈ 0.09248
    log_out = (np.power(10.0, (x - d) / c_c) - b) / a
    lin_out = (x - f) / e
    return np.where(x >= CUT_CV, log_out, lin_out)


def _idt_clog3(x: np.ndarray) -> np.ndarray:
    """IDT: Canon C-Log3 v1.2 → Scene Linear (inverse of positive branch).
    For normal scene content (x > CV of linear segment ~0.1527).
    """
    cl3_lin_cv = 1.9754798 * 0.014043 + 0.12512219  # ≈ 0.15277
    k = 14.98325
    a = 0.36726845
    log_out = (np.power(10.0, (x - 0.12240537) / a) - 1.0) / k * 0.9
    lin_out = (x - 0.12512219) / 1.9754798 * 0.9
    return np.where(x > cl3_lin_cv, log_out, lin_out)


def _idt_log3g10(x: np.ndarray) -> np.ndarray:
    """IDT: RED Log3G10 v2 → Scene Linear (exact inverse of _lut_log3g10)."""
    s = np.sign(x)
    return s * (np.power(10.0, np.abs(x) / 0.224282) - 1.0) / 155.975327 - 0.01


def _idt_davinci_intermediate(x: np.ndarray) -> np.ndarray:
    """IDT: DaVinci Intermediate → Scene Linear (exact inverse of _lut_davinci_intermediate)."""
    DI_A = 0.0075
    DI_B = 7.0
    DI_C = 0.07329248
    DI_M = 10.44426855
    DI_LOG_CUT = 0.02740668
    log_out = np.exp2(x / DI_C - DI_B) - DI_A
    lin_out = x / DI_M
    return np.where(x > DI_LOG_CUT, log_out, lin_out)


def _idt_bmd_gen5(x: np.ndarray) -> np.ndarray:
    """IDT: BMD Film Generation 5 → Scene Linear (exact inverse of _lut_bmd_gen5)."""
    A = 0.08692876065491224
    B = 0.005494072432257808
    C = 0.5300133392291939
    D = 8.283605932402494
    E = 0.09246575342465753
    LOG_CUT = D * 0.005 + E  # ≈ 0.13388
    log_out = np.exp((x - C) / A) - B
    lin_out = (x - E) / D
    return np.where(x >= LOG_CUT, log_out, lin_out)


def _idt_nlog(x: np.ndarray) -> np.ndarray:
    """IDT: Nikon N-Log → Scene Linear (exact inverse of _lut_nlog)."""
    a = 0.635386119257087
    b = 0.0075
    c = 0.1466275659824047
    d = 0.6050830889540567
    # CV at cut1=0.328: a * (0.328 + b)^(1/3)
    CUT_CV = a * (0.328 + b) ** (1.0 / 3.0)
    log_out = np.exp((x - d) / c)
    cbrt_out = np.power(x / a, 3.0) - b
    return np.maximum(np.where(x >= CUT_CV, log_out, cbrt_out), 0.0)


def _lut_passthrough(x: np.ndarray) -> np.ndarray:
    """Placeholder for separator."""
    return x


# ── LUT dispatch table ────────────────────────────────────────────────────────
# Keys must match LUT_MODES entries exactly.
_LUT_FUNCTIONS = {
    "None": None,
    # Display / Tonemap
    "sRGB (Display)": _lut_srgb,
    "Rec.709 (Broadcast)": _lut_rec709,
    "Filmic (Cinematic)": _lut_filmic,
    "Reinhard Tonemap": _lut_reinhard,
    "ACES Filmic": _lut_aces_filmic,
    # Camera Log — Forward (Linear → Log)
    "LogC3 (ARRI EI800)": _lut_logc3,
    "LogC4 (ARRI Alexa 35)": _lut_logc4,
    "F-Log2 (Fujifilm)": _lut_flog2,
    "C-Log3 (Canon)": _lut_clog3,
    "Log3G10 (RED IPP2)": _lut_log3g10,
    "DaVinci Intermediate": _lut_davinci_intermediate,
    "BMD Film Gen5": _lut_bmd_gen5,
    "V-Log (Panasonic)": _lut_vlog,
    "RED Log3G10": _lut_log3g10,  # Alias for Log3G10 (RED IPP2)
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
}


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

    # False Color is a special case — replaces the whole image
    if handler == "false_color":
        fc = _lut_false_color(img)
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


# ═══════════════════════════════════════════════════════════════════════════════
#                      COLOR GRADING ENGINE v3.2
#
#  CANONICAL PIPELINE ORDER — must match GLSL composite shader exactly:
#
#   1. Offset (global additive shift)
#   2. Exposure (stops, multiplicative)
#   3. White Balance (temperature / tint multiplicative)
#   4. Lift / Gain / Gamma  (Resolve-style — via applyGrading)
#   5. Contrast  (pivoted)
#   6. Shadows / Highlights (luma-weighted)
#   7. Saturation
#   8. Hue Shift
#   9. LUT (last in chain)
#
#  v3.2 FIX: Python and GLSL were out of sync.
#   Old Python order: Lift → Exposure → Gain → Gamma → Saturation → Temp → LUT
#   New Python order: matches GLSL exactly for correct passthrough output.
# ═══════════════════════════════════════════════════════════════════════════════


def apply_grading(
    img: np.ndarray,
    # Basic controls
    exposure: float = 0.0,
    gamma: float = 1.0,  # scalar gamma (used when gamma_rgb is None)
    gain: float = 1.0,
    lift: float = 0.0,
    saturation: float = 1.0,
    temperature: float = 6500.0,
    # Extended Resolve-style controls (v3.2 sync)
    offset: float = 0.0,  # global additive offset (applied first)
    contrast: float = 1.0,  # contrast multiplier (pivoted)
    pivot: float = 0.18,  # contrast pivot point
    shadows: float = 0.0,  # shadow lift/crush (-1..1)
    highlights: float = 0.0,  # highlight expand/compress (-1..1)
    hue_shift: float = 0.0,  # degrees (-180..180)
    luma_mix: float = 1.0,   # Preserve original luminance (0.0 = full preservation)
    # Per-channel overrides (v3.5+) — list/tuple of [R, G, B] floats.
    # If provided, these override the scalar gamma/gain/lift/offset for each channel.
    gamma_rgb: Optional[List[float]] = None,
    gain_rgb: Optional[List[float]] = None,
    lift_rgb: Optional[List[float]] = None,
    offset_rgb: Optional[List[float]] = None,
    # LUT
    lut_name: str = "None",
    lut_intensity: float = 1.0,
    # Color Science
    color_science: int = 0, # 0 = Linear/sRGB, 1 = ACEScct
    # Sprint 4: Gamut Compression
    gamut_compression: bool = False,
) -> np.ndarray:
    """
    Apply full grading stack to a float32 image.

    Pipeline order is IDENTICAL to the GLSL composite shader so that the
    passthrough IMAGE output matches what the WebGL viewer displays.

    Args:
        img:         float32 (H,W,C), values can exceed [0,1] for HDR
        exposure:    Stops of exposure compensation (-6 to +6)
        gamma:       Artistic midtone gamma (0.1–4.0, 1.0 = identity)
        gain:        Multiplicative gain (0.0–5.0, 1.0 = unity)
        lift:        Shadow additive offset (-1.0–1.0, luma-pivoted)
        saturation:  Color saturation (0.0 = mono, 1.0 = normal, 3.0 = hyper)
        temperature: Color temperature in Kelvin (2000 = warm, 12000 = cool)
        offset:      Global additive offset applied before everything else
        contrast:    Contrast multiplier around pivot (1.0 = identity)
        pivot:       Contrast pivot point (default 0.18 = 18% grey)
        shadows:     Shadow region brightness (-1..1)
        highlights:  Highlight region brightness (-1..1)
        hue_shift:   Hue rotation in degrees
        lut_name:    Name of built-in LUT to apply
        lut_intensity: LUT blend factor (0.0–1.0)
        color_science: 0 = Linear/sRGB, 1 = ACEScct

    Returns:
        float32 numpy array, same shape as input
    """
    out = img.astype(np.float32, copy=True)
    
    # Pre-grading luma for luma_mix
    luma_orig = None
    if abs(luma_mix - 1.0) > 0.001:
        luma_orig = 0.2126 * out[..., 0] + 0.7152 * out[..., 1] + 0.0722 * out[..., 2]

    # Fast-path: skip if all controls are at identity
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
        and lut_name == "None"
    )
    if is_default:
        return out

    # ── 1. Exposure (Linear part) ────────────────────────────────────────────────
    if abs(exposure) > 0.001:
        out *= np.float32(2.0**exposure)

    # ── 3. White Balance ───────────────────────────────────────────────────────
    if abs(temperature - 6500.0) > 10.0 and out.ndim == 3 and out.shape[2] >= 3:
        r_mult, g_mult, b_mult = _kelvin_to_rgb_multipliers(temperature)
        out[..., 0] *= np.float32(r_mult)
        out[..., 1] *= np.float32(g_mult)
        out[..., 2] *= np.float32(b_mult)

    # ── 4. Lift / Gain / Gamma  (Resolve-style, per-channel aware) ───────────

    def do_lift_gamma_gain(color: np.ndarray) -> np.ndarray:
        c = color.copy()
        nc = c.shape[2] if c.ndim == 3 else 0
        has_per_ch = nc >= 3

        # Build per-channel offset array (broadcast-safe)
        _o = np.array(offset_rgb, dtype=np.float32) if (offset_rgb and len(offset_rgb) == 3) else np.array([offset, offset, offset], dtype=np.float32)
        _l = np.array(lift_rgb,   dtype=np.float32) if (lift_rgb   and len(lift_rgb)   == 3) else np.array([lift,   lift,   lift  ], dtype=np.float32)
        _ga = np.array(gain_rgb,  dtype=np.float32) if (gain_rgb   and len(gain_rgb)   == 3) else np.array([gain,   gain,   gain  ], dtype=np.float32)
        _gm = np.array(gamma_rgb, dtype=np.float32) if (gamma_rgb  and len(gamma_rgb)  == 3) else np.array([gamma,  gamma,  gamma ], dtype=np.float32)

        # Offset: per-channel additive shift
        # B-9 FIX: Only fall back to scalar offset when no per-channel array was
        # provided. Previously, if offset_rgb was [0,0,0] the scalar fallback
        # could fire with a non-zero scalar offset value, double-applying.
        if offset_rgb and len(offset_rgb) == 3 and has_per_ch:
            if np.any(np.abs(_o) > 0.001):
                c[..., :3] += _o
        elif abs(offset) > 0.001:
            c += np.float32(offset)

        # Lift: per-channel shadow offset, luma-pivoted so highlights are unaffected
        if np.any(np.abs(_l) > 0.001) and has_per_ch:
            luma = 0.2126 * c[..., 0] + 0.7152 * c[..., 1] + 0.0722 * c[..., 2]
            pivot_lift = np.clip(1.0 - luma, 0.0, 1.0)[..., np.newaxis]  # (H,W,1)
            c[..., :3] += _l * pivot_lift  # broadcasts over (H,W,3)

        # Gain: per-channel multiplicative slope
        if np.any(np.abs(_ga - 1.0) > 0.001) and has_per_ch:
            c[..., :3] *= _ga

        # Gamma: per-channel power curve on positive values only
        if np.any(np.abs(_gm - 1.0) > 0.001) and has_per_ch:
            for ch in range(3):
                inv_g = np.float32(1.0 / max(float(_gm[ch]), 0.01))
                if abs(float(_gm[ch]) - 1.0) > 0.001:
                    pos = c[..., ch] > 0
                    c[pos, ch] = np.power(c[pos, ch], inv_g)
        return c

    if color_science == 1 and out.ndim == 3 and out.shape[2] >= 3:
        # ACEScct Pipeline
        # 1. Linear sRGB -> ACEScg (AP1)
        LIN_SRGB_TO_ACESCG = np.array([
            [0.59719, 0.35458, 0.04823],
            [0.07600, 0.90834, 0.01566],
            [0.02840, 0.13383, 0.83777]
        ], dtype=np.float32)
        
        ACESCG_TO_LIN_SRGB = np.array([
            [ 1.60475, -0.53108, -0.07367],
            [-0.10208,  1.10813, -0.00605],
            [-0.00327, -0.07276,  1.07602]
        ], dtype=np.float32)

        # Matmul expects (..., 3) but dot works if we do reshaping or einsum
        # A @ x for each pixel: out = out @ M.T
        acescg = np.tensordot(out[..., :3], LIN_SRGB_TO_ACESCG, axes=([2], [1]))
        
        # 2. ACEScg -> ACEScct
        cct = np.empty_like(acescg)
        mask = acescg <= 0.0078125
        cct[mask] = 10.5402377416545 * acescg[mask] + 0.0729055341958355
        # Prevent log domain error
        clamped = np.clip(acescg[~mask], 1e-10, None)
        cct[~mask] = (np.log2(clamped) + 9.72) / 17.52
        
        # 3. Apply Grading
        cct = do_lift_gamma_gain(cct)
        
        # 4. ACEScct -> ACEScg
        acescg_back = np.empty_like(cct)
        mask2 = cct > 0.155251141552511
        acescg_back[mask2] = np.exp2(cct[mask2] * 17.52 - 9.72)
        acescg_back[~mask2] = (cct[~mask2] - 0.0729055341958355) / 10.5402377416545
        
        # 5. ACEScg -> Linear sRGB
        out[..., :3] = np.tensordot(acescg_back, ACESCG_TO_LIN_SRGB, axes=([2], [1]))
    else:
        out = do_lift_gamma_gain(out)

    # ── 5. Contrast ────────────────────────────────────────────────────────────
    if abs(contrast - 1.0) > 0.001:
        out = (out - np.float32(pivot)) * np.float32(contrast) + np.float32(pivot)

    # ── 6. Shadows / Highlights ────────────────────────────────────────────────
    if (
        (abs(shadows) > 0.001 or abs(highlights) > 0.001)
        and out.ndim == 3
        and out.shape[2] >= 3
    ):
        luma = 0.2126 * out[..., 0] + 0.7152 * out[..., 1] + 0.0722 * out[..., 2]
        # Quadratic weights — matches GLSL smoothstep approach
        s_weight = np.power(np.clip(1.0 - luma, 0.0, 1.0), 2.0)[..., np.newaxis]
        h_weight = np.power(np.clip(luma, 0.0, 1.0), 2.0)[..., np.newaxis]
        out *= 1.0 + np.float32(shadows) * s_weight * 0.5
        out *= 1.0 + np.float32(highlights) * h_weight * 0.5
        out = np.maximum(out, 0.0)

    # ── 6.5 Gamut Compression ──────────────────────────────────────────────────
    # FIX: replaced inline implementation with aces2_gamut_compress() from
    # color_utils.py. The previous inline code used Rec.709 luma coefficients
    # (0.2126, 0.7152, 0.0722) on what is ACEScg scene-linear data — those
    # coefficients are defined for Rec.709 primaries, not AP1, producing wrong
    # luma-weighting in the compression. It also used a different knee algorithm
    # than color_utils, giving inconsistent results between the viewer and any
    # downstream node that called aces2_gamut_compress() directly.
    if gamut_compression and out.ndim == 3 and out.shape[2] >= 3:
        if _aces2_gamut_compress is not None:
            out[..., :3] = _aces2_gamut_compress(out[..., :3])
        else:
            # Fallback if color_utils unavailable — warn once and skip
            logger.warning(
                "[Radiance Viewer] aces2_gamut_compress unavailable "
                "(color_utils not found) — gamut compression skipped."
            )

    # ── 7. Saturation ──────────────────────────────────────────────────────────
    if abs(saturation - 1.0) > 0.001 and out.ndim == 3 and out.shape[2] >= 3:
        luma = 0.2126 * out[..., 0] + 0.7152 * out[..., 1] + 0.0722 * out[..., 2]
        for c in range(min(out.shape[2], 3)):
            out[..., c] = luma + saturation * (out[..., c] - luma)

    # ── 8. Hue Shift ───────────────────────────────────────────────────────────
    if abs(hue_shift) > 0.1 and out.ndim == 3 and out.shape[2] >= 3:
        h_shift = hue_shift / 360.0
        # B-4 FIX: Clamp to non-negative before HSV — negative RGB from prior
        # grading steps causes broken hue output (cmax < 0 → invalid saturation).
        rgb = np.maximum(out[..., :3], 0.0)
        # Vectorised HSV shift using numpy
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

    # ── 8.5 Luma Mix ───────────────────────────────────────────────────────────
    if luma_orig is not None:
        luma_curr = 0.2126 * out[..., 0] + 0.7152 * out[..., 1] + 0.0722 * out[..., 2]
        color_with_orig_luma = out * (luma_orig / (luma_curr + 1e-10))[..., np.newaxis]
        out = luma_mix * out + (1.0 - luma_mix) * color_with_orig_luma

    # ── 9. LUT (last in chain) ─────────────────────────────────────────────────
    if lut_name != "None" and lut_intensity > 0.0:
        out = apply_lut(out, lut_name, lut_intensity)

    return out


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


# ═══════════════════════════════════════════════════════════════════════════════
#                      TENSOR UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════


def safe_tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Safely convert a torch tensor to numpy, handling all dtypes.
    Handles bfloat16 which raises TypeError on direct .numpy().
    Always returns float32 numpy array.
    """
    if tensor.dtype in (torch.bfloat16, torch.float16):
        return tensor.float().cpu().numpy()
    return tensor.cpu().float().numpy()


def compute_data_range(img_np: np.ndarray) -> Tuple[float, float, bool]:
    """
    Compute actual data range and detect HDR content.
    Returns (min_value, max_value, has_hdr).
    """
    d_min = float(img_np.min())
    d_max = float(img_np.max())

    # Sanitize NaN/Inf for JSON safety
    if not math.isfinite(d_min):
        d_min = 0.0
    if not math.isfinite(d_max):
        d_max = 1.0

    has_hdr = d_max > 1.0 or d_min < 0.0
    return d_min, d_max, has_hdr


# ═══════════════════════════════════════════════════════════════════════════════
#                      16-BIT PNG SAVE (v1.21)
# ═══════════════════════════════════════════════════════════════════════════════


def save_16bit_png(filepath: str, img_uint16: np.ndarray) -> bool:
    """
    Save a uint16 numpy array as a true 16-bit PNG using cv2.

    Args:
        filepath: Output file path
        img_uint16: uint16 array — (H,W,3) RGB, (H,W,4) RGBA, or (H,W) gray

    Returns:
        True on success, False on failure
    """
    if not HAS_CV2:
        logger.error("cv2 required for 16-bit PNG save but not available")
        return False

    try:
        if img_uint16.ndim == 3 and img_uint16.shape[2] == 3:
            bgr = cv2.cvtColor(img_uint16, cv2.COLOR_RGB2BGR)
            return cv2.imwrite(
                filepath, bgr, [cv2.IMWRITE_PNG_COMPRESSION, CV2_PNG_COMPRESSION]
            )
        elif img_uint16.ndim == 3 and img_uint16.shape[2] == 4:
            bgra = cv2.cvtColor(img_uint16, cv2.COLOR_RGBA2BGRA)
            return cv2.imwrite(
                filepath, bgra, [cv2.IMWRITE_PNG_COMPRESSION, CV2_PNG_COMPRESSION]
            )
        elif img_uint16.ndim == 3 and img_uint16.shape[2] == 1:
            return cv2.imwrite(
                filepath,
                img_uint16[:, :, 0],
                [cv2.IMWRITE_PNG_COMPRESSION, CV2_PNG_COMPRESSION],
            )
        elif img_uint16.ndim == 2:
            return cv2.imwrite(
                filepath, img_uint16, [cv2.IMWRITE_PNG_COMPRESSION, CV2_PNG_COMPRESSION]
            )
        else:
            logger.warning(f"Unsupported shape for 16-bit save: {img_uint16.shape}")
            return False
    except Exception as e:
        logger.error(f"cv2 16-bit PNG save failed: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
#                     RADIANCE VIEWER NODE v2.3.3
# ═══════════════════════════════════════════════════════════════════════════════


class RadianceType(str):
    """
    A specific multi-type class that leverages LiteGraph's native comma-separated type parsing 
    on the frontend ("IMAGE,VIDEO") while bypassing Python backend validation.
    """
    def __ne__(self, __value: object) -> bool:
        return False

# Expose as a multi-type that specifically accepts IMAGE and VIDEO in the UI
image_video_type = RadianceType("IMAGE,VIDEO")


class RadianceViewer:
    """
    VFX Industry-Standard Viewer with IMAGE passthrough:
    • Zoom/Pan navigation
    • Exposure / Gamma / Gain / Lift / Saturation / Temperature controls
    • Built-in LUT engine (10 industry-standard LUTs)
    • Channel viewing (RGB/R/G/B/Alpha/Luma)
    • Color picker with HDR value display
    • False color & zebra analysis
    • A/B comparison modes
    • 16-bit + HDR output
    • IMAGE passthrough — no more dead-end node
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": (image_video_type,),
            },
            "optional": {
                "bit_depth": (
                    BIT_DEPTH_MODES,
                    {
                        "default": "16-bit (Quality)",
                        "tooltip": (
                            "8-bit: Fast, 256 levels. "
                            "16-bit: Quality, 65536 levels. "
                            "16-bit + HDR: Adds float16 sidecar for HDR color picker. "
                            "32-bit Float: Full IEEE 754 fp32 sidecar — industry standard, "
                            "matches Nuke/Flame/Resolve viewer precision. Larger files."
                        ),
                    },
                ),
                # ── Compare / Depth ──
                "compare_image": (image_video_type,),
                "zdepth": (
                    image_video_type,
                    {"tooltip": "Z-Depth map to display when pressing Z button"},
                ),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            },
        }

    # v2.3.3: IMAGE passthrough + metadata outputs
    RETURN_TYPES = ()
    FUNCTION = "view"
    CATEGORY = "FXTD Studios/Radiance/Views"
    OUTPUT_NODE = True
    DESCRIPTION = """VFX Industry-Standard Viewer v2.3.3 — Pro Evolution:
• GPU Waveform / RGB Parade / Vectorscope / Histogram scopes
• Power Windows masking (Radial + Box, feather, rotation)
• Comparison Bridge — mouse-draggable wipe + reference shelf (8 stills)
• Anamorphic lens streaks + Brown-Conrady k1/k2 distortion
• Edge-preserving Bilateral Filter denoising (7×7 GPU kernel)
• Channel viewing (RGB/R/G/B/Alpha/Luma), False Color, Zebra
• 16-bit PNG + .rhdr HDR sidecar + .exr export
• IMAGE passthrough — no longer a dead-end node"""

    def view(
        self,
        image: Any,
        bit_depth: str = "16-bit (Quality)",
        # Compare / Depth
        compare_image: Optional[Any] = None,
        zdepth: Optional[Any] = None,
        # Hidden
        prompt: Optional[Any] = None,
        extra_pnginfo: Optional[Any] = None,
        unique_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process and display the image in the Radiance Viewer."""

        # ── Handle Custom VIDEO dicts / Extractor ──
        def _extract_tensor(x: Any) -> Any:
            if hasattr(x, "get_components"):
                try:
                    comps = x.get_components()
                    if hasattr(comps, "images"):
                        return comps.images
                except Exception:
                    pass
            if isinstance(x, dict) and "samples" in x:
                return x["samples"]
            if isinstance(x, (list, tuple)) and len(x) > 0 and isinstance(x[0], torch.Tensor):
                return x[0]
            return x

        image = _extract_tensor(image)
        if compare_image is not None:
            compare_image = _extract_tensor(compare_image)
        if zdepth is not None:
            zdepth = _extract_tensor(zdepth)

        # ── Parse bit depth ──
        use_16bit = "16-bit" in bit_depth and HAS_CV2
        # v4.1: 32-bit Float — full IEEE 754 fp32 RHDR sidecar.
        # Uses RHDR format with flags=1 (fp32 marker) so viewer detects and
        # uploads as Float32 texture instead of HALF_FLOAT. Twice the VRAM of
        # fp16 but eliminates all precision loss — matches Nuke/Flame pipeline.
        use_32bit = bit_depth == "32-bit Float"
        # HDR sidecar: enabled for 16-bit+HDR, 16-bit Quality, and 32-bit Float modes.
        save_hdr_sidecar = use_16bit or use_32bit

        if "16-bit" in bit_depth and not HAS_CV2 and not use_32bit:
            logger.warning(
                "16-bit mode requested but cv2 not available. "
                "Falling back to 8-bit. Install opencv-python for 16-bit support."
            )

        # ── Validate ──
        validation_error = self._validate_image(image, "image")
        if validation_error:
            logger.error(validation_error)
            return {
                "ui": {"radiance_images": [], "error": [validation_error]},
                "result": (),
            }

        try:
            output_dir = folder_paths.get_temp_directory()
            batch_size = image.shape[0] if image.dim() == 4 else 1
            batch_size = min(batch_size, MAX_BATCH_SIZE)

            images_list: List[Dict[str, Any]] = []

            for frame_idx in range(batch_size):
                try:
                    frame_result = self._process_frame(
                        image,
                        frame_idx,
                        output_dir,
                        use_16bit=use_16bit,
                        use_32bit=use_32bit,
                        save_hdr_sidecar=save_hdr_sidecar,
                        prefix="◎ Radiance_viewer",
                    )
                    if frame_result is not None:
                        frame_result["frame"] = frame_idx
                        frame_result["total_frames"] = batch_size
                        images_list.append(frame_result)
                except (RuntimeError, ValueError) as e:
                    logger.warning(f"Error processing frame {frame_idx}: {e}")
                    continue

            # Compare image
            if compare_image is not None:
                compare_list = self._process_compare_image(
                    compare_image,
                    output_dir,
                    use_16bit=use_16bit,
                    save_hdr_sidecar=save_hdr_sidecar,
                )
                images_list.extend(compare_list)

            # Z-Depth
            if zdepth is not None:
                zdepth_list = self._process_zdepth_image(
                    zdepth,
                    output_dir,
                    use_16bit=use_16bit,
                    save_hdr_sidecar=save_hdr_sidecar,
                )
                images_list.extend(zdepth_list)

            # ── Build output IMAGE ──
            output_image = image

            # ── Metadata string ──
            if use_32bit:
                depth_label = "32-bit Float"
            elif use_16bit:
                depth_label = "16-bit"
            else:
                depth_label = "8-bit"

            meta_dict = {
                "bit_depth": depth_label,
                "hdr_sidecar": save_hdr_sidecar,
                "batch_size": batch_size,
            }
            metadata_str = json.dumps(meta_dict, indent=2)

            # ── v3.1: Inject into Terminal Namespace ──
            # Allows experts to inspect live data via the REPL tab
            _TERMINAL_NS["image"] = image
            _TERMINAL_NS["prompt"] = prompt
            _TERMINAL_NS["metadata"] = extra_pnginfo

            # ── v4.2: Delivery Cache (Update node frames) ───────────
            # Use ComfyUI's unique_id (stable graph node ID) as the cache key.
            # v4.1 used str(id(self)) which is the CPython memory address — this
            # changes every execution and is lost on module reload, causing
            # "No cached frame found" on delivery.  unique_id matches
            # this.node.id in LiteGraph JS, so the fallback also works.
            instance_key = str(unique_id) if unique_id else str(id(self))
            _viewer_cache_set(instance_key, image)

            return {
                "ui": {
                    "radiance_images": images_list,
                    "instance_id": [instance_key],
                    "batch_size": [batch_size],
                    "bit_depth": [depth_label],
                },
                "result": (),
            }

        except (RuntimeError, ValueError, TypeError) as e:
            logger.error(f"Error in viewer: {e}")
            return {
                "ui": {"radiance_images": [], "error": [str(e)]},
                "result": (),
            }

    # ─────────────────────────────────────────────────────────────────────
    # Validation
    # ─────────────────────────────────────────────────────────────────────

    def _validate_image(self, image: torch.Tensor, name: str) -> Optional[str]:
        """Validate image tensor. Returns error string or None if valid."""
        if not isinstance(image, torch.Tensor):
            return f"{name} must be a torch.Tensor, got {type(image)}"
        if image.dim() not in (3, 4):
            return f"{name} must be 3D or 4D tensor, got {image.dim()}D"
        if image.dim() == 4:
            _, h, w, c = image.shape
        else:
            h, w = image.shape[0], image.shape[1]
            c = image.shape[2] if image.dim() == 3 else 1
        if h > MAX_IMAGE_DIMENSION or w > MAX_IMAGE_DIMENSION:
            return (
                f"{name} dimensions ({h}x{w}) exceed maximum "
                f"({MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION})"
            )
        if c not in (1, 3, 4):
            return f"{name} has {c} channels, expected 1, 3, or 4"
        return None

    # ─────────────────────────────────────────────────────────────────────
    # Frame Processing (v2.1: grading + LUT + 16-bit)
    # ─────────────────────────────────────────────────────────────────────

    def _process_frame(
        self,
        image: torch.Tensor,
        frame_idx: int,
        output_dir: str,
        use_16bit: bool = True,
        use_32bit: bool = False,
        save_hdr_sidecar: bool = False,
        prefix: str = "◎ Radiance_viewer",
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single frame: grade → LUT → save at selected bit depth.
        """
        # Safe tensor → numpy
        if image.dim() == 4:
            frame = safe_tensor_to_numpy(image[frame_idx])
        else:
            frame = safe_tensor_to_numpy(image)

        # Sanitize NaNs and Infs (Critical for WebGL and PNG safety)
        # Replace NaN with 0.0, Inf with 65504.0 (max float16)
        if not np.isfinite(frame).all():
            frame = np.nan_to_num(frame, nan=0.0, posinf=65504.0, neginf=0.0)

        # Data range of RAW input (before grading)
        d_min, d_max, has_hdr = compute_data_range(frame)

        # ── Apply grading + LUT for viewer display ──
        # FIX v2.3.3: Do NOT apply grading to the preview image saved to disk.
        # The frontend viewer applies grading in real-time via WebGL.
        # If we bake it here, it gets applied twice.

        # ── v2.2 ARCHITECTURE: RHDR is the primary display format ──
        # Like DJV/RV viewing EXR natively: viewer loads compressed half-float
        # directly into GPU and tonemaps in real-time. PNG is reduced to a
        # tiny 256px ComfyUI node thumbnail (required by the framework).
        #
        # .rhdr format: zlib-compressed IEEE 754 float16 — lossless at fp16
        # precision, comparable to EXR DWAA at quality ~85.
        # Compression: ~5-15% of raw → similar file sizes to DWAA EXR.

        unique_id = uuid.uuid4().hex[:12]

        # ── Data Prep ──
        h_frame, w_frame = frame.shape[:2]
        c_frame = frame.shape[2] if frame.ndim == 3 else 1

        # B-8 FIX: RHDR header uses uint16 for width/height (max 65535).
        # Validate dimensions to prevent silent header overflow.
        if w_frame > 65535 or h_frame > 65535:
            logger.error(
                f"Frame dimensions ({w_frame}x{h_frame}) exceed RHDR uint16 header limit (65535). "
                f"Skipping RHDR/EXR sidecar generation for this frame."
            )
            return None

        # Pad RGB to RGBA for WebGL/OpenCV consistency
        # Many Windows/NVIDIA WebGL drivers fail with 3-channel float textures
        if c_frame == 3:
            frame_to_save = np.ones((h_frame, w_frame, 4), dtype=np.float32)
            frame_to_save[..., :3] = frame
            c_frame = 4
        else:
            frame_to_save = frame

        # ── 1. PRIMARY: Save .rhdr sidecar ──────────────────────────────────
        # Supports two precisions controlled by header flags field:
        #   flags = 0  →  fp16 payload  (HALF_FLOAT, half VRAM, existing behaviour)
        #   flags = 1  →  fp32 payload  (FLOAT, full IEEE 754, 32-bit Float mode)
        #
        # Header layout (12 bytes, little-endian):
        #   [0:4]  magic  "RHDR"
        #   [4:6]  width  uint16
        #   [6:8]  height uint16
        #   [8:10] channels uint16
        #   [10:12] flags uint16  — 0=fp16, 1=fp32
        #   [12:]  zlib-compressed pixel data
        rhdr_filename = f"{prefix}_{unique_id}_{frame_idx}.rhdr"
        rhdr_saved = False

        if use_32bit:
            # ── 32-bit Float path: full IEEE 754 fp32, flags=1 ────────────────
            try:
                rhdr_filepath = safe_join(output_dir, rhdr_filename)
                fp32_bytes = frame_to_save.astype(np.float32).tobytes()
                compressed = zlib.compress(fp32_bytes, level=6)
                # flags=1 signals fp32 to the viewer parser
                header = struct.pack("<4sHHHH", b"RHDR", w_frame, h_frame, c_frame, 1)
                with open(rhdr_filepath, "wb") as rhdr_f:
                    rhdr_f.write(header)
                    rhdr_f.write(compressed)
                rhdr_saved = True
                ratio = len(compressed) / len(fp32_bytes) * 100 if fp32_bytes else 0
                logger.debug(
                    f"RHDR fp32 saved: {rhdr_filename} "
                    f"({len(compressed)//1024}KB, {ratio:.0f}% ratio | "
                    f"range [{d_min:.3f}, {d_max:.3f}])"
                )
            except (IOError, OSError, ValueError) as e:
                logger.warning(f"Failed to save fp32 RHDR for frame {frame_idx}: {e}")

        elif use_16bit:
            # ── 16-bit Float path: fp16, flags=0 (existing behaviour) ─────────
            try:
                rhdr_filepath = safe_join(output_dir, rhdr_filename)
                fp16_data = frame_to_save.astype(np.float16).tobytes()
                compressed = zlib.compress(fp16_data, level=6)
                header = struct.pack("<4sHHHH", b"RHDR", w_frame, h_frame, c_frame, 0)
                with open(rhdr_filepath, "wb") as rhdr_f:
                    rhdr_f.write(header)
                    rhdr_f.write(compressed)
                rhdr_saved = True
                ratio = len(compressed) / len(fp16_data) * 100 if fp16_data else 0
                logger.debug(
                    f"RHDR fp16 saved: {rhdr_filename} "
                    f"({len(compressed)//1024}KB, {ratio:.0f}% ratio | "
                    f"range [{d_min:.3f}, {d_max:.3f}])"
                )
            except (IOError, OSError, ValueError) as e:
                logger.warning(f"Failed to save RHDR for frame {frame_idx}: {e}")

        # ── 2. SECONDARY: Save .exr (OpenEXR 32-bit float) for external use ──
        # Requested by users for "Save Image" to export true HDR.
        exr_filename = f"{prefix}_{unique_id}_{frame_idx}.exr"
        exr_saved = False
        
        if write_exr_robust:
            try:
                exr_filepath = safe_join(output_dir, exr_filename)
                # write_exr_robust expects a 3-channel RGB array (ComfyUI standard).
                # frame_to_save may be RGBA (4-channel) if the input was RGB and was
                # padded with alpha=1.0 for WebGL/OpenCV driver compatibility.
                # Passing the 4-channel array causes write_exr_robust to either fail
                # silently (returning False) or write an ABGR-channel-ordered EXR that
                # external viewers decode as black / swapped channels.
                # Fix: always slice to [:, :, :3] before the call so the writer
                # receives the canonical RGB triplet regardless of padding state.
                frame_for_exr = frame_to_save[..., :3]  # RGBA → RGB  (no-op if already 3-ch)
                success = write_exr_robust(
                    exr_filepath,
                    frame_for_exr,
                    bit_depth="32-bit Float",
                    compression="ZIP"
                )
                if success:
                    exr_saved = True
            except Exception as e:
                logger.warning(f"Failed to save EXR for frame {frame_idx}: {e}")
        
        if not exr_saved:
            exr_filename = None

        # v3.1 FIX: Save PNG as RHDR fallback when decompression fails in browser.
        # B-12 FIX: Renamed from misleading "THUMB_MAX". Since RHDR is the primary
        # display source, the PNG only needs to be large enough for a decent fallback.
        # 2048px covers up to ~2K content; for 4K+ the RHDR path handles display.
        FALLBACK_MAX_DIM = 2048

        if has_hdr and d_max > 1.05:
            preview_safe = np.maximum(frame, 0.0)
            preview_image = (preview_safe / (1.0 + preview_safe)).astype(np.float32)
        else:
            preview_image = frame

        # Downsample to thumbnail
        th, tw = preview_image.shape[:2]
        if max(th, tw) > FALLBACK_MAX_DIM:
            scale = FALLBACK_MAX_DIM / max(th, tw)
            new_w, new_h = int(tw * scale), int(th * scale)
            try:
                if HAS_CV2:
                    preview_thumb = cv2.resize(
                        preview_image, (new_w, new_h), interpolation=cv2.INTER_AREA
                    )
                else:
                    from PIL import Image as PILImage

                    pil_full = self._frame_to_pil_8bit(preview_image)
                    if pil_full:
                        pil_full = pil_full.resize((new_w, new_h), PILImage.LANCZOS)
                        preview_thumb = np.array(pil_full).astype(np.float32) / 255.0
                    else:
                        preview_thumb = preview_image
            except Exception as e:
                logger.debug(f"Image resize failed, using original: {e}")
                preview_thumb = preview_image
        else:
            preview_thumb = preview_image

        png_filename = f"{prefix}_{unique_id}_{frame_idx}_thumb.png"
        try:
            png_filepath = safe_join(output_dir, png_filename)
            pil_img = self._frame_to_pil_8bit(preview_thumb)
            if pil_img is None:
                return None
            pil_img.save(png_filepath, compress_level=DEFAULT_PNG_COMPRESSION)
        except (IOError, OSError) as e:
            logger.warning(f"Failed to save thumbnail for frame {frame_idx}: {e}")
            return None

        result: Dict[str, Any] = {
            "filename": png_filename,
            "subfolder": "",
            "type": "temp",
            "data_range": [d_min, d_max],
            "has_hdr": has_hdr,
            "hdr_filename": rhdr_filename if rhdr_saved else None,
            "exr_filename": exr_filename if exr_filename else None,
            # Explicit EXR location metadata — avoids inheriting PNG thumbnail's
            # subfolder/type if the two files are ever stored separately.
            "exr_subfolder": "" if exr_filename else None,
            "exr_type": "temp" if exr_filename else None,
        }

        if rhdr_saved:
            result["hdr_sidecar"] = rhdr_filename
            result["hdr_primary"] = True  # tells viewer to load .rhdr as display source
            # v4.1: tag fp32 sidecars so viewer uses loadFloat32Texture instead of
            # loadFloat16Texture — full pipeline precision end-to-end
            result["hdr_fp32"] = use_32bit

        # v3.0 Feature #5: fp32 picking buffer — true scene-linear HDR color picker
        pick_filename = f"{prefix}_{unique_id}_{frame_idx}.rpick"
        try:
            pick_filepath = safe_join(output_dir, pick_filename)
            if _save_pick_buffer(frame, pick_filepath):
                result["pick_filename"] = pick_filename
                logger.debug(f"[Radiance v3.0] Pick buffer saved: {pick_filename}")
        except ValueError as e:
            logger.debug(f"[Radiance v3.0] Pick buffer path error: {e}")

        return result


    # ─────────────────────────────────────────────────────────────────────
    # PIL 8-bit
    # ─────────────────────────────────────────────────────────────────────

    def _frame_to_pil_8bit(self, frame: np.ndarray) -> Optional["PILImage.Image"]:
        """Convert float32 numpy frame to 8-bit PIL Image."""
        from PIL import Image as PILImage

        img_8bit = self._convert_to_8bit(frame)

        if img_8bit.ndim == 3 and img_8bit.shape[2] == 4:
            return PILImage.fromarray(img_8bit, mode="RGBA")
        elif img_8bit.ndim == 3 and img_8bit.shape[2] == 3:
            return PILImage.fromarray(img_8bit, mode="RGB")
        elif img_8bit.ndim == 3 and img_8bit.shape[2] == 1:
            return PILImage.fromarray(img_8bit[:, :, 0], mode="L")
        elif img_8bit.ndim == 2:
            return PILImage.fromarray(img_8bit, mode="L")

        logger.warning(f"Unexpected 8-bit image shape: {img_8bit.shape}")
        return None

    # ─────────────────────────────────────────────────────────────────────
    # Compare Image
    # ─────────────────────────────────────────────────────────────────────

    def _process_compare_image(
        self,
        compare_image: torch.Tensor,
        output_dir: str,
        use_16bit: bool = True,
        save_hdr_sidecar: bool = False,
    ) -> List[Dict[str, Any]]:
        """Process comparison image. Returns list of image metadata."""
        result: List[Dict[str, Any]] = []

        validation_error = self._validate_image(compare_image, "compare_image")
        if validation_error:
            logger.warning(validation_error)
            return result

        cmp_batch = compare_image.shape[0] if compare_image.dim() == 4 else 1
        cmp_batch = min(cmp_batch, MAX_BATCH_SIZE)

        for cmp_idx in range(cmp_batch):
            try:
                frame_result = self._process_frame(
                    compare_image,
                    cmp_idx,
                    output_dir,
                    use_16bit=use_16bit,
                    save_hdr_sidecar=save_hdr_sidecar,
                    prefix="◎ Radiance_compare",
                )
                if frame_result is not None:
                    frame_result["is_compare"] = True
                    frame_result["frame"] = cmp_idx
                    result.append(frame_result)
            except (RuntimeError, ValueError) as e:
                logger.warning(f"Error processing compare frame {cmp_idx}: {e}")
                continue

        return result

    # ─────────────────────────────────────────────────────────────────────
    # Z-Depth
    # ─────────────────────────────────────────────────────────────────────

    def _process_zdepth_image(
        self,
        zdepth: torch.Tensor,
        output_dir: str,
        use_16bit: bool = True,
        save_hdr_sidecar: bool = False,
    ) -> List[Dict[str, Any]]:
        """Process Z-depth map. 16-bit = 65536 depth levels via cv2."""
        result: List[Dict[str, Any]] = []

        validation_error = self._validate_image(zdepth, "zdepth")
        if validation_error:
            logger.warning(validation_error)
            return result

        depth_batch = zdepth.shape[0] if zdepth.dim() == 4 else 1
        depth_batch = min(depth_batch, MAX_BATCH_SIZE)

        for depth_idx in range(depth_batch):
            try:
                if zdepth.dim() == 4:
                    depth_frame = safe_tensor_to_numpy(zdepth[depth_idx])
                else:
                    depth_frame = safe_tensor_to_numpy(zdepth)

                # Extract single channel
                if depth_frame.ndim == 2:
                    depth_np = depth_frame
                elif depth_frame.ndim == 3:
                    if depth_frame.shape[-1] == 1:
                        depth_np = depth_frame[..., 0]
                    else:
                        depth_np = (
                            0.2126 * depth_frame[..., 0]
                            + 0.7152 * depth_frame[..., 1]
                            + 0.0722 * depth_frame[..., 2]
                        )
                else:
                    continue

                d_min = float(depth_np.min())
                d_max = float(depth_np.max())

                # Sanitize for metadata / JSON safety
                if not math.isfinite(d_min):
                    d_min = 0.0
                if not math.isfinite(d_max):
                    d_max = 1.0

                if d_max > d_min:
                    depth_normalized = (depth_np - d_min) / (d_max - d_min)
                else:
                    depth_normalized = np.zeros_like(depth_np)

                unique_id = uuid.uuid4().hex[:12]
                depth_filename = f"◎ Radiance_zdepth_{unique_id}_{depth_idx}.png"

                try:
                    depth_filepath = safe_join(output_dir, depth_filename)
                except ValueError as e:
                    logger.error(f"Invalid zdepth path: {e}")
                    continue

                try:
                    if use_16bit:
                        depth_16bit = (depth_normalized * BIT_16_MAX).astype(np.uint16)
                        success = save_16bit_png(depth_filepath, depth_16bit)
                        if not success:
                            from PIL import Image as PILImage

                            depth_8bit = (depth_normalized * BIT_8_MAX).astype(np.uint8)
                            PILImage.fromarray(depth_8bit, mode="L").save(
                                depth_filepath, compress_level=DEFAULT_PNG_COMPRESSION
                            )
                    else:
                        from PIL import Image as PILImage

                        depth_8bit = (depth_normalized * BIT_8_MAX).astype(np.uint8)
                        PILImage.fromarray(depth_8bit, mode="L").save(
                            depth_filepath, compress_level=DEFAULT_PNG_COMPRESSION
                        )
                except (IOError, OSError) as e:
                    logger.warning(f"Failed to save zdepth frame {depth_idx}: {e}")
                    continue

                frame_meta: Dict[str, Any] = {
                    "filename": depth_filename,
                    "subfolder": "",
                    "type": "temp",
                    "is_zdepth": True,
                    "frame": depth_idx,
                    "bit_depth": 16 if use_16bit else 8,
                    "depth_range": [d_min, d_max],
                }

                if save_hdr_sidecar:
                    npy_filename = f"◎ Radiance_zdepth_{unique_id}_{depth_idx}_float.rhdr"
                    try:
                        npy_filepath = safe_join(output_dir, npy_filename)
                        fp16_data = depth_np.astype(np.float16).tobytes()
                        compressed = zlib.compress(fp16_data, level=6)
                        dh, dw = depth_np.shape[:2]
                        dc = depth_np.shape[2] if depth_np.ndim == 3 else 1
                        header = struct.pack("<4sHHHH", b"RHDR", dw, dh, dc, 0)
                        with open(npy_filepath, "wb") as rhdr_f:
                            rhdr_f.write(header)
                            rhdr_f.write(compressed)
                        frame_meta["hdr_sidecar"] = npy_filename
                    except (IOError, OSError, ValueError) as e:
                        logger.warning(f"Failed to save depth sidecar {depth_idx}: {e}")

                result.append(frame_meta)

            except (RuntimeError, ValueError) as e:
                logger.warning(f"Error processing zdepth frame {depth_idx}: {e}")
                continue

        return result

    # ─────────────────────────────────────────────────────────────────────
    # Bit Depth Conversion
    # ─────────────────────────────────────────────────────────────────────

    def _convert_to_16bit(self, img_np: np.ndarray) -> np.ndarray:
        """Convert float32 image to uint16. Clamps to [0,1] for display."""
        if img_np.dtype in (np.float32, np.float64):
            return (np.clip(img_np, 0.0, 1.0) * BIT_16_MAX).astype(np.uint16)
        elif img_np.dtype == np.float16:
            return (np.clip(img_np.astype(np.float32), 0.0, 1.0) * BIT_16_MAX).astype(
                np.uint16
            )
        elif img_np.dtype == np.uint16:
            return img_np
        elif img_np.dtype == np.uint8:
            return img_np.astype(np.uint16) * BIT_16_TO_8_DIVISOR
        else:
            img_f = img_np.astype(np.float64)
            mn, mx = img_f.min(), img_f.max()
            if mx > mn:
                return ((img_f - mn) / (mx - mn) * BIT_16_MAX).astype(np.uint16)
            return np.zeros_like(img_f, dtype=np.uint16)

    def _convert_to_8bit(self, img_np: np.ndarray) -> np.ndarray:
        """Convert float32 image to uint8."""
        if img_np.dtype in (np.float32, np.float64, np.float16):
            return (np.clip(img_np.astype(np.float32), 0.0, 1.0) * BIT_8_MAX).astype(
                np.uint8
            )
        elif img_np.dtype == np.uint16:
            return (img_np // BIT_16_TO_8_DIVISOR).astype(np.uint8)
        elif img_np.dtype == np.uint8:
            return img_np
        else:
            img_f = img_np.astype(np.float64)
            mn, mx = img_f.min(), img_f.max()
            if mx > mn:
                return ((img_f - mn) / (mx - mn) * BIT_8_MAX).astype(np.uint8)
            return np.zeros_like(img_f, dtype=np.uint8)


class RadianceGradeApply:
    """
    Dedicated node to bake Radiance Viewer grading math into an image tensor.
    Useful for exporting or passing a graded image downstream in the pipeline.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                # Global
                "exposure": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "offset": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001}),
                # Wheels
                "lift": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001}),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 4.0, "step": 0.01}),
                "gain": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                # Tone
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01}),
                "pivot": ("FLOAT", {"default": 0.18, "min": 0.0, "max": 1.0, "step": 0.01}),
                "shadows": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "highlights": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                # Color
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01}),
                "temperature": ("FLOAT", {"default": 6500.0, "min": 2000.0, "max": 12000.0, "step": 10.0}),
                "hue_shift": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.1}),
                # LUT
                "lut_name": (LUT_MODES, {"default": "None"}),
                "lut_intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                # Color Science
                "color_science": (["Linear (sRGB)", "ACEScct"], {"default": "Linear (sRGB)"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_grade"
    CATEGORY = "FXTD Studios/Radiance/Color"
    DESCRIPTION = "Bakes Radiance grading math into the image tensor permanently."

    def apply_grade(
        self,
        image: torch.Tensor,
        exposure: float, offset: float,
        lift: float, gamma: float, gain: float,
        contrast: float, pivot: float,
        shadows: float, highlights: float,
        saturation: float, temperature: float, hue_shift: float,
        lut_name: str, lut_intensity: float,
        color_science: str = "Linear (sRGB)"
    ) -> Tuple[torch.Tensor]:
        
        batch_size = image.shape[0]
        out_batch = []
        
        for i in range(batch_size):
            frame = image[i].cpu().numpy()
            
            # Apply identical pipeline as RadianceViewer
            graded = apply_grading(
                img=frame,
                exposure=exposure,
                gamma=gamma,
                gain=gain,
                lift=lift,
                saturation=saturation,
                temperature=temperature,
                offset=offset,
                contrast=contrast,
                pivot=pivot,
                shadows=shadows,
                highlights=highlights,
                hue_shift=hue_shift,
                lut_name=lut_name,
                lut_intensity=lut_intensity,
                color_science=1 if color_science == "ACEScct" else 0
            )
            
            # B-5 FIX: Do NOT hard-clamp to [0,1] — this destroys HDR data.
            # Only sanitize NaN/Inf values which would break downstream nodes.
            graded = np.nan_to_num(graded, nan=0.0, posinf=65504.0, neginf=0.0)
            graded_tensor = torch.from_numpy(graded)
            out_batch.append(graded_tensor)
            
        return (torch.stack(out_batch),)


# ── Persistent namespace for the terminal REPL ──────────────────────────────
_TERMINAL_NS: dict = {
    "math": math,
    "os": os,
    "torch": torch,
    "np": np,
    "json": json,
    "time": __import__("time"),
    "cv2": cv2 if HAS_CV2 else None,
}
# Pre-seed it once with folder_paths and PIL if available
try:
    from PIL import Image as PILImage
    _TERMINAL_NS["PIL"] = PILImage
except ImportError:
    pass
# Pre-seed it once with folder_paths if available
try:
    import folder_paths as _fp
    _TERMINAL_NS["folder_paths"] = _fp
except ImportError:
    pass


# B-1 SECURITY FIX: Terminal is disabled by default. Set environment variable
# RADIANCE_ENABLE_TERMINAL=1 to enable it (development use only).
_TERMINAL_ENABLED = os.environ.get("RADIANCE_ENABLE_TERMINAL", "0") == "1"
if not _TERMINAL_ENABLED:
    logger.info(
        "[Radiance] Terminal endpoint DISABLED (security). "
        "Set RADIANCE_ENABLE_TERMINAL=1 to enable for development."
    )


@PromptServer.instance.routes.post('/radiance/terminal')
async def radiance_terminal_endpoint(request):
    # B-1 FIX: Reject all requests unless explicitly opted in
    if not _TERMINAL_ENABLED:
        return web.json_response({
            "output": "Terminal is disabled for security. "
                      "Set environment variable RADIANCE_ENABLE_TERMINAL=1 to enable.",
            "status": "error"
        })

    import threading

    try:
        data = await request.json()
        code = data.get('code', '')
        reset = data.get('_reset', False)

        # ── Reset namespace ──────────────────────────────────────────────
        if reset or code.strip() == '__radiance_reset__ = True':
            global _TERMINAL_NS
            _TERMINAL_NS = {
                "math": math, "os": os, "torch": torch, "np": np,
                "json": json, "time": __import__("time"),
            }
            try:
                import folder_paths as _fp
                _TERMINAL_NS["folder_paths"] = _fp
            except ImportError:
                pass
            return web.json_response({"output": "", "status": "success"})

        # ── Run with 30-second timeout guard ────────────────────────────
        result_box = {"output": "", "status": "success"}

        def _run():
            old_stdout, old_stderr = sys.stdout, sys.stderr
            buf = io.StringIO()
            sys.stdout = buf
            sys.stderr = buf
            try:
                exec(code, _TERMINAL_NS)  # noqa: S102
                result_box["output"] = buf.getvalue()
                result_box["status"] = "success"
            except Exception:
                result_box["output"] = buf.getvalue() + "\n" + traceback.format_exc()
                result_box["status"] = "error"
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=30)

        if t.is_alive():
            result_box["output"] = "⏱ Execution timed out after 30 seconds."
            result_box["status"] = "error"

        return web.json_response(result_box)

    except Exception as e:
        return web.json_response({"output": str(e), "status": "error"})

# ── Apex Delivery Helpers ──────────────────────────────────────────────────

def get_next_version(directory: str, filename_base: str) -> str:
    """Scan directory for existing versions and returns the next one (e.g., v02)."""
    if not os.path.exists(directory):
        return "v01"
    
    # Look for pattern: base_vXX.ext or base_vXX
    # Use re.escape to handle base with dots/etc safely
    pattern = re.compile(rf"{re.escape(filename_base)}_v(\d+)")
    max_v = 0
    
    try:
        for f in os.listdir(directory):
            match = pattern.search(f)
            if match:
                v = int(match.group(1))
                if v > max_v:
                    max_v = v
    except Exception:
        pass
        
    return f"v{max_v + 1:02d}"

def apply_burn_in(image: torch.Tensor, settings: Dict, grading: Dict, version_str: str) -> torch.Tensor:
    """Apply slate, timecode, and version burn-ins using RadianceMetadataOverlay logic."""
    try:
        from .nodes_overlay import RadianceMetadataOverlay
        overlay = RadianceMetadataOverlay()
        
        # Extract settings
        burn_tc = settings.get("burn_in_tc", False)
        burn_ver = settings.get("burn_in_ver", False)
        burn_frame = settings.get("burn_in_frame", False)
        burn_lut = settings.get("burn_in_lut", False)
        burn_custom = settings.get("burn_custom_text", "")
        include_slate = settings.get("include_slate", False)
        filename = settings.get("filename", "output")
        start_frame = max(1, int(settings.get('range_in', 1)))
        
        device = image.device

        # 1. Handle Slate (Prepend 10 frames of black with metadata)
        if include_slate:
            # Create a black frame for the slate (1 frame batch)
            slate_frame = torch.zeros((1, image.shape[1], image.shape[2], 3), device=device)
            
            # Apply high-opacity slate metadata
            slate_result = overlay.overlay_metadata(
                slate_frame,
                enabled=True,
                project="◎ Radiance_PROJECT",
                shot=filename,
                frame=1,
                timecode="00:00:00:00",
                position="Bottom Bar (Slate)",
                font_size=24,
                opacity=1.0,
                user_text=f"VERSION: {version_str}\nEXPORT DATE: PRODUCTION MASTER",
            )[0]
            
            # Repeat for 10 frames
            slate_batch = slate_result.repeat(10, 1, 1, 1)
            image = torch.cat([slate_batch, image], dim=0)

        # 2. Handle Continuous Burn-In (TC / Version / Frame / Custom)
        if burn_tc or burn_ver or burn_frame or burn_lut or burn_custom:
            user_text = version_str if burn_ver else ""
            if burn_lut:
                lut_name = grading.get("lut_name", "None")
                lut_text = f"LUT: {lut_name}" if lut_name != "None" else "LUT: NONE"
                user_text = f"{user_text} | {lut_text}" if user_text else lut_text
            if burn_custom:
                user_text = f"{user_text}\n{burn_custom}" if user_text else burn_custom
                
            tc_str = "00:00:00:00" if burn_tc else ""
            start_frm = start_frame if burn_frame else 0
            
            # Add overlay to all frames
            image = overlay.overlay_metadata(
                image,
                enabled=True,
                project="",
                shot="",
                frame=start_frm,
                timecode=tc_str,
                position="Top Right" if (burn_ver or burn_lut) else "Bottom Center",
                font_size=18,
                opacity=0.6,
                user_text=user_text
            )[0]

        return image
    except Exception as e:
        logger.error(f"Failed to apply burn-in: {e}")
        return image


@PromptServer.instance.routes.post('/radiance/deliver')
async def radiance_deliver_endpoint(request):
    """
    VFX Delivery Endpoint: Receives grading state + export settings from HUD.
    Applies grading, AI-Upscaling, and Burn-ins before high-quality encoding.
    """
    try:
        data = await request.json()
        # Coerce to str — JS may send a numeric node ID (from this.node.id
        # fallback) when instanceId hasn't been set yet.  Cache keys are always
        # strings because unique_id arrives as a string from ComfyUI.
        instance_key = str(data.get('instance_id', ''))
        grading = data.get('grading', {})
        settings = data.get('settings', {})

        images = _viewer_cache_get(instance_key)
        if images is None:
            return web.json_response({"error": "No frames found in cache for this node. Run the workflow first.", "status": "error"})
        
        # ─── Render Range ──────────────────────────────────────────────
        range_in = max(1, int(settings.get('range_in', 1)))
        range_out = int(settings.get('range_out', 0))
        total_frames = images.shape[0]
        start_idx = range_in - 1
        end_idx = range_out if range_out > 0 else total_frames
        start_idx = max(0, min(start_idx, total_frames - 1))
        end_idx = max(start_idx + 1, min(end_idx, total_frames))
        images = images[start_idx:end_idx]
        
        # ─── Process Options ──────────────────────────────────────────
        filename_prefix = settings.get('filename', '◎ Radiance_Deliver')
        output_format = settings.get('format', 'Video — MP4 (H.264)')
        fps = float(settings.get('fps', 24.0))
        quality = int(settings.get('quality', 18))
        output_path = settings.get('path', '')

        # ─── I-7 FIX: Input Sanitization ─────────────────────────────
        # Sanitize filename: allow unicode letters, digits, safe punctuation
        _SAFE_FILENAME_RE = re.compile(r'[^\w\s◎_.() -]', re.UNICODE)
        filename_prefix = _SAFE_FILENAME_RE.sub('', str(filename_prefix)).strip()
        if not filename_prefix:
            filename_prefix = 'Radiance_Deliver'
        if len(filename_prefix) > 200:
            filename_prefix = filename_prefix[:200]

        # Sanitize output_path: canonicalize, prevent traversal
        if output_path:
            output_path = os.path.abspath(os.path.normpath(str(output_path)))
            # Reject paths with traversal artifacts or invalid length
            if '..' in output_path.split(os.sep) or len(output_path) > 1024:
                logger.warning(f"[Deliver] Rejected suspicious output path: {output_path[:100]}")
                return web.json_response({"error": "Invalid output path", "status": "error"})

        # Clamp numeric parameters to sane ranges
        fps = max(1.0, min(fps, 240.0))
        quality = max(0, min(quality, 51))
        
        # NORMALIZED: Use a single color_space variable to avoid default mismatches
        # between the soft-clip logic and the RadianceWrite call.
        color_space = settings.get('colorSpace', 'sRGB (Standard)')
        broadcast_safe = settings.get('soft_clip', True)
        
        # Apex Features
        upscale_2x = settings.get('upscale_2x', False)
        smart_ver = settings.get('smart_versioning', True)
        
        # Handle Versioning
        version_str = "v01"
        if smart_ver:
            import folder_paths
            base_path = output_path if output_path else folder_paths.get_output_directory()
            version_str = get_next_version(base_path, filename_prefix)
            filename_prefix = f"{filename_prefix}_{version_str}"

        # ─── Process Grading ──────────────────────────────────────────
        # Helpers to unpack JS values safely
        def safe_float(v, default):
            """Safely convert any JS value to a Python float."""
            if isinstance(v, (list, tuple)) and len(v) > 0: return float(v[0])
            try: return float(v)
            except (TypeError, ValueError): return default

        def safe_array(v, default_scalar, length=3):
            """Return a float list[length] from a scalar or array JS value."""
            if isinstance(v, (list, tuple)) and len(v) >= length:
                return [float(x) for x in v[:length]]
            try:
                s = float(v)
                return [s] * length
            except (TypeError, ValueError):
                return [default_scalar] * length

        exposure    = safe_float(grading.get('exposure'), 0.0)
        saturation  = safe_float(grading.get('saturation'), 1.0)
        # Temperature from viewer is in internal ±2 units, not Kelvin.
        # Convert: internal 0.0 = 6500K, +1.0 → ~10000K, -1.0 → ~3500K
        _temp_internal = safe_float(grading.get('temperature'), 0.0)
        temperature = 6500.0 + _temp_internal * 3500.0  # maps ±1 → ±3500K
        contrast    = safe_float(grading.get('contrast'), 1.0)
        pivot       = safe_float(grading.get('pivot'), 0.18)
        shadows     = safe_float(grading.get('shadows'), 0.0)
        highlights  = safe_float(grading.get('highlights'), 0.0)
        hue_shift   = safe_float(grading.get('hue_shift'), 0.0)
        lut_name    = grading.get('lut_name', 'None')
        lut_intensity = safe_float(grading.get('lut_intensity'), 1.0)
        gamut_compression = bool(grading.get('gamut_compression', False))

        # Per-channel arrays (Sprint 1 fix: no longer collapsed to [0])
        gamma_rgb  = safe_array(grading.get('gamma'),  1.0)
        gain_rgb   = safe_array(grading.get('gain'),   1.0)
        lift_rgb   = safe_array(grading.get('lift'),   0.0)
        offset_rgb = safe_array(grading.get('offset'), 0.0)

        
        # Apply grading to whole batch
        out_batch = []
        for i in range(images.shape[0]):
            frame_np = images[i].cpu().numpy()
            graded = apply_grading(
                img=frame_np,
                exposure=exposure,
                gamma=1.0,           # scalar path is bypassed by gamma_rgb
                gain=1.0,            # scalar path is bypassed by gain_rgb
                lift=0.0,            # scalar path is bypassed by lift_rgb
                saturation=saturation,
                temperature=temperature,
                offset=0.0,          # scalar path is bypassed by offset_rgb
                contrast=contrast,
                pivot=pivot,
                shadows=shadows,
                highlights=highlights,
                hue_shift=hue_shift,
                # Per-channel arrays:
                gamma_rgb=gamma_rgb,
                gain_rgb=gain_rgb,
                lift_rgb=lift_rgb,
                offset_rgb=offset_rgb,
                lut_name=lut_name,
                lut_intensity=lut_intensity,
                color_science=1 if str(grading.get('colorScience')) in ['1', 'ACEScct'] else 0,
                luma_mix=float(grading.get('lumaMix', 1.0)),
                gamut_compression=gamut_compression
            )
            out_batch.append(torch.from_numpy(graded))

        graded_tensor = torch.stack(out_batch)

        # ─── FX Baking (grain/bloom/halation/diffusion) ───────────────
        # Apply cosmetic FX so the export matches the viewer canvas exactly.
        _grain     = float(grading.get('grain', 0.0))
        _bloom     = float(grading.get('bloom', 0.0))
        _halation  = float(grading.get('halation', 0.0))
        _diffusion = float(grading.get('diffusion', 0.0))
        _denoise   = float(grading.get('denoise', 0.0))

        if _grain > 0.01 or _bloom > 0.01 or _halation > 0.01 or _diffusion > 0.01 or _denoise > 0.01:
            try:
                np_batch = graded_tensor.cpu().numpy()  # (B, H, W, C)
                fx_batch = []
                rng = np.random.default_rng(seed=42)  # deterministic grain for exports
                for i_frame in range(np_batch.shape[0]):
                    f = np_batch[i_frame].astype(np.float32)

                    # Denoise: simple bilateral-approximation via Gaussian smoothing
                    if _denoise > 0.01:
                        try:
                            import cv2 as _cv2
                            sigma = _denoise * 3.0
                            f = _cv2.bilateralFilter((f * 65535).astype(np.uint16), d=5, sigmaColor=sigma*20, sigmaSpace=sigma*20).astype(np.float32) / 65535.0  # noqa: E501
                        except Exception:
                            pass  # cv2 unavailable; skip

                    # Grain: additive Gaussian noise at the correct amplitude
                    if _grain > 0.01:
                        noise = rng.standard_normal(f.shape).astype(np.float32)
                        f = f + noise * _grain * 0.02

                    # Halation: red-channel Gaussian glow on highlights
                    if _halation > 0.01:
                        try:
                            import cv2 as _cv2
                            hi = np.clip(f[..., 0] - 0.8, 0, None)  # highlights only
                            blurred = _cv2.GaussianBlur(hi, (0, 0), sigmaX=_halation * 30)
                            f[..., 0] = np.minimum(f[..., 0] + blurred * _halation * 2.0, 2.0)
                        except Exception:
                            pass

                    # Bloom: additive luminance glow
                    if _bloom > 0.01:
                        try:
                            import cv2 as _cv2
                            lum = 0.2126*f[...,0] + 0.7152*f[...,1] + 0.0722*f[...,2]
                            hi = np.clip(lum - 0.7, 0, None)[..., np.newaxis]
                            blurred = _cv2.GaussianBlur(hi * np.ones((1,1,3), np.float32), (0, 0), sigmaX=_bloom * 40)
                            f = f + blurred * _bloom
                        except Exception:
                            pass

                    # Diffusion: global softening (Pro Mist / Black Mist effect)
                    if _diffusion > 0.01:
                        try:
                            import cv2 as _cv2
                            soft = _cv2.GaussianBlur(f, (0, 0), sigmaX=_diffusion * 20)
                            f = f * (1 - _diffusion * 0.5) + soft * _diffusion * 0.5
                        except Exception:
                            pass

                    fx_batch.append(f)
                graded_tensor = torch.from_numpy(np.stack(fx_batch))
                logger.info(f"[Deliver] FX baked: grain={_grain:.2f} bloom={_bloom:.2f} halation={_halation:.2f} diffusion={_diffusion:.2f}")
            except Exception as e:
                logger.warning(f"[Deliver] FX baking failed (non-fatal): {e}")


        # ─── AI Upscale (2x) ──────────────────────────────────────────
        if upscale_2x:
            try:
                from .image.upscale import RadianceAIUpscale
                upscaler = RadianceAIUpscale()
                # RealESRGAN_x2plus is ideal for performance/quality trade-off
                graded_tensor, _ = upscaler.upscale(
                    image=graded_tensor,
                    model_name="RealESRGAN_x2plus",
                    mode="Refine (HDR)",
                    tile_size=512
                )
            except Exception as e:
                logger.error(f"AI Upscale failed, continuing with original: {e}")

        # ─── Aspect Ratio Blanking ────────────────────────────────────
        aspect_ratio_str = settings.get('aspect_ratio', 'None')
        if aspect_ratio_str != 'None':
            try:
                target_ratio = float(aspect_ratio_str.split(':')[0])
                _, h, w, _ = graded_tensor.shape
                current_ratio = w / h
                if current_ratio > target_ratio + 0.01:
                    target_w = int(h * target_ratio)
                    pad = (w - target_w) // 2
                    graded_tensor[:, :, :pad, :] = 0.0
                    graded_tensor[:, :, -pad:, :] = 0.0
                elif current_ratio < target_ratio - 0.01:
                    target_h = int(w / target_ratio)
                    pad = (h - target_h) // 2
                    graded_tensor[:, :pad, :, :] = 0.0
                    graded_tensor[:, -pad:, :, :] = 0.0
            except Exception as e:
                logger.error(f"Aspect blanking failed: {e}")

        # NOTE: Soft clip (highlight compression) is now consolidated inside
        # apply_output_transform in nodes_io.py. The 'broadcast_safe' flag set
        # above is passed directly to RadianceWrite which calls that function.
        # Removing the duplicate ACES tonemap block that was here.

        graded_tensor = apply_burn_in(graded_tensor, settings, grading, version_str)

        # ─── Integrated QC Pass ───────────────────────────────────────
        qc_report = ""
        try:
            v_min = graded_tensor.min().item()
            v_max = graded_tensor.max().item()
            if v_min < 0.0 or v_max > 1.0:
                qc_report = f"◎ [QC WARNING] Out-of-Gamut detected: Min={v_min:.3f}, Max={v_max:.3f}. (Illegal levels for broadcast)."
            else:
                qc_report = f"◎ [QC PASS] Levels within legal broadcast range (0.0 - 1.0)."
        except Exception:
            pass

        # ─── Save using RadianceWrite Logic ────────────────────────────
        from .nodes_io import RadianceWrite
        writer = RadianceWrite()
        
        # Track progress for writer (approximate since writer is external)
        _VIEWER_PROGRESS[instance_key] = {"current": 90, "total": 100, "status": "encoding", "message": "Encoding Master..."}

        _, path, _ = writer.write(
            filename_prefix=filename_prefix,
            output_format=output_format,
            fps=fps,
            quality=quality,
            output_color_space=color_space,
            image=graded_tensor,
            output_path=output_path,
            broadcast_safe=broadcast_safe,
        )

        # ─── Post-Export: Thumbnails & Sidecars ────────────────────────
        try:
            # Save Thumbnail (.jpg)
            import PIL.Image
            # Thumbnail: use a mid-range frame from the graded tensor
            n_graded = graded_tensor.shape[0]
            thumb_frame = graded_tensor[min(n_graded - 1, 10 if n_graded > 10 else 0)]
            thumb_np = (thumb_frame.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            thumb_img = PIL.Image.fromarray(thumb_np)
            thumb_path = os.path.splitext(path)[0] + "_thumb.jpg"
            thumb_img.save(thumb_path, quality=85)
            
            # ASC CDL Sidecar Export
            if settings.get('export_cdl', False):
                slope = grading.get('gain', 1.0)
                offset = grading.get('offset', 0.0)
                power = grading.get('gamma', 1.0)
                sat = grading.get('saturation', 1.0)
                if not isinstance(slope, list): slope = [slope]*3
                if not isinstance(offset, list): offset = [offset]*3
                if not isinstance(power, list): power = [power]*3
                
                cdl = f"""<ColorCorrection id="◎ Radiance_grade">\n  <SOPNode>\n    <Slope>{slope[0]:.6f} {slope[1]:.6f} {slope[2]:.6f}</Slope>\n    <Offset>{offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}</Offset>\n    <Power>{power[0]:.6f} {power[1]:.6f} {power[2]:.6f}</Power>\n  </SOPNode>\n  <SatNode>\n    <Saturation>{sat:.6f}</Saturation>\n  </SatNode>\n</ColorCorrection>"""
                cdl_path = os.path.splitext(path)[0] + ".cdl"
                with open(cdl_path, "w") as f:
                    f.write(cdl)
            
            # Save Metadata Sidecar (.json)
            meta = {
                "version": version_str,
                "grading": grading,
                "export_settings": settings,
                "qc": qc_report,
                "timestamp": os.path.getmtime(path)
            }
            with open(os.path.splitext(path)[0] + "_meta.json", 'w') as f:
                json.dump(meta, f, indent=4)
                
            # Reveal Folder Hook
            if settings.get('reveal_folder', False):
                import platform, subprocess
                try:
                    p = os.path.abspath(output_path) if output_path else folder_paths.get_output_directory()
                    if platform.system() == "Windows":
                        os.startfile(p)
                    elif platform.system() == "Darwin":
                        subprocess.Popen(["open", p])
                    else:
                        subprocess.Popen(["xdg-open", p])
                except Exception as e:
                    logger.error(f"Launch folder failed: {e}")
                    
        except Exception as e:
            logger.error(f"Sidecar generation failed: {e}")

        _VIEWER_PROGRESS[instance_key] = {"current": 100, "total": 100, "status": "done", "message": "Delivery Complete"}

        return web.json_response({
            "status": "success",
            "path": path,
            "qc": qc_report,
            "message": f"Export delivered successfully: {os.path.basename(path)} ({version_str})"
        })

    except Exception as e:
        logger.error(f"Delivery failed: {traceback.format_exc()}")
        return web.json_response({"error": str(e), "status": "error"})

@PromptServer.instance.routes.get('/radiance/progress')
async def radiance_progress_endpoint(request):
    """Returns active delivery progress for a node instance."""
    instance_id = request.query.get('id')
    if not instance_id:
        return web.json_response({"error": "Missing ID", "status": "error"})
    
    progress = _VIEWER_PROGRESS.get(instance_id, {"current": 0, "total": 100, "status": "idle", "message": "Waiting..."})
    return web.json_response(progress)



# ═══════════════════════════════════════════════════════════════════════════════
#                           NODE REGISTRATION
# ═══════════════════════════════════════════════════════════════════════════════

NODE_CLASS_MAPPINGS = {
    "FXTD_RadianceViewer": RadianceViewer,
    "FXTD_RadianceProViewer": RadianceViewer,
    "FXTD_RadianceGradeApply": RadianceGradeApply,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FXTD_RadianceViewer": "◎ Radiance Viewer",
    "FXTD_RadianceProViewer": "◎ Radiance Pro Viewer (Legacy)",
    "FXTD_RadianceGradeApply": "◎ Radiance Grade Apply",
}
