import torch
import numpy as np
import os
import json
import logging
try:
    import folder_paths
except ImportError:
    folder_paths = None
import datetime
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional

# Local imports
from .utils import numpy_to_tensor_float32

# Imports
__all__ = [
    "write_exr_robust",
    "write_exr_openexr",
    "write_exr_cv2",
    "check_openexr_available",
    "build_radiance_hdr_metadata",
]

try:
    from radiance.color.transfer import (
        linear_to_logc3,
        linear_to_logc4,
        linear_to_slog3,
        srgb_to_linear,
        linear_to_srgb,
    )
    from radiance.color.matrices import (
        apply_matrix_transform,
        AWG3_TO_ACESCG,
        AWG4_TO_ACESCG,
        SGAMUT3_CINE_TO_ACESCG,
        ACESCG_TO_SRGB,
    )

    _HAS_COLOR_UTILS = True
except ImportError:
    _HAS_COLOR_UTILS = False

if not _HAS_COLOR_UTILS:
    try:
        from radiance.color.transfer import (
            linear_to_logc3,
            linear_to_logc4,
            linear_to_slog3,
            srgb_to_linear,
            linear_to_srgb,
        )
        from radiance.color.matrices import (
            apply_matrix_transform,
            AWG3_TO_ACESCG,
            AWG4_TO_ACESCG,
            SGAMUT3_CINE_TO_ACESCG,
            ACESCG_TO_SRGB,
        )
        _HAS_COLOR_UTILS = True
    except ImportError:
        # Deliberately silent: this is the second of two optional import paths
        # for the colour helpers, and it runs at module scope before `logger`
        # exists. _HAS_COLOR_UTILS stays False and every caller checks it.
        pass

logger = logging.getLogger("radiance.hdr.io")

# ═══════════════════════════════════════════════════════════════════════════════
#                           EXR CONSTANTS & HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

# AUDIT-FIX (2026-08): MAX_BATCH_SIZE / MAX_IMAGE_DIMENSION duplicates removed.
# They were stale copies (1000 / 32768) of the authoritative viewer limits in
# radiance/viewer_utils.py (9999 / 16384), referenced by nothing -- a drift
# trap for anyone who imported the wrong pair.

EXR_PIXEL_TYPE = {"UINT": 0, "HALF": 1, "FLOAT": 2}
EXR_LINE_ORDER = {"INCREASING_Y": 0, "DECREASING_Y": 1, "RANDOM_Y": 2}

def _safe_inverse(matrix: np.ndarray, name: str) -> Optional[np.ndarray]:
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError as e:
        logger.warning(f"Failed to invert {name} matrix: {e}")
        return None

try:
    SRGB_TO_ACESCG = _safe_inverse(ACESCG_TO_SRGB, "ACESCG_TO_SRGB")
    ACESCG_TO_AWG3 = _safe_inverse(AWG3_TO_ACESCG, "AWG3_TO_ACESCG")
    ACESCG_TO_AWG4 = _safe_inverse(AWG4_TO_ACESCG, "AWG4_TO_ACESCG")
    ACESCG_TO_SGAMUT = _safe_inverse(SGAMUT3_CINE_TO_ACESCG, "SGAMUT3_CINE_TO_ACESCG")
except NameError:
    SRGB_TO_ACESCG = ACESCG_TO_AWG3 = ACESCG_TO_AWG4 = ACESCG_TO_SGAMUT = None

def float32_to_bytes(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def float16_to_bytes(arr: np.ndarray) -> bytes:
    return arr.astype(np.float16).tobytes()

def parse_timecode(tc_str: str, fps: float = 24.0) -> Optional[Any]:
    """Convert HH:MM:SS:FF or HH:MM:SS;FF string to OpenEXR.TimeCode."""
    try:
        # ALBABIT-FIX: was Imath.TimeCode(..., rate=..., ...) -- the installed
        # Imath.TimeCode no longer accepts a `rate` argument at all (confirmed
        # live: TypeError, silently swallowed by this function's own
        # except-all, so timeCode was never actually produced), and separately
        # the modern OpenEXR.File API only accepts its own OpenEXR.TimeCode
        # type as a header attribute, not Imath.TimeCode (confirmed live:
        # "unrecognized type of attribute"). Rate was only used above for
        # is_drop, which is already computed independently.
        import OpenEXR
        parts = tc_str.replace(";", ":").split(":")
        if len(parts) != 4:
            return None
        h, m, s, f = map(int, parts)
        # Drop frame detection (simplified: if ; is used or if FPS is 29.97/59.94)
        is_drop = ";" in tc_str or abs(fps - 29.97) < 0.01 or abs(fps - 59.94) < 0.01
        tc = OpenEXR.TimeCode()
        tc.hours, tc.minutes, tc.seconds, tc.frame, tc.dropFrame = h, m, s, f, is_drop
        return tc
    except Exception:
        return None

# ═══════════════════════════════════════════════════════════════════════════════
#                           EXR WRITERS
# ═══════════════════════════════════════════════════════════════════════════════

def write_exr_cv2(filepath: str, image: np.ndarray, bit_depth: str = "float32", compression: str = "ZIP") -> bool:
    try:
        import cv2
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        img = image.astype(np.float32)
        if img.ndim == 3:
            if img.shape[2] == 3: img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif img.shape[2] == 4: img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        flags = [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF if "16" in str(bit_depth) else cv2.IMWRITE_EXR_TYPE_FLOAT]
        return cv2.imwrite(filepath, img, flags)
    except Exception: return False

def check_openexr_available() -> bool:
    try:
        import OpenEXR
        return True
    except Exception: return False

_COMP_MAP = {
    "None": "NO_COMPRESSION", "RLE": "RLE_COMPRESSION", "ZIPS": "ZIPS_COMPRESSION",
    "ZIP": "ZIP_COMPRESSION", "PIZ": "PIZ_COMPRESSION", "PXR24": "PXR24_COMPRESSION",
    "B44": "B44_COMPRESSION", "B44A": "B44A_COMPRESSION", "DWAA": "DWAA_COMPRESSION",
    "DWAB": "DWAB_COMPRESSION",
}


def write_exr_openexr(filepath: str, channels: Dict[str, np.ndarray], compression: str = "ZIP", pixel_type: str = "HALF", metadata: Optional[Dict[str, Any]] = None) -> None:
    # ALBABIT-FIX: migrated from the legacy Header()/OutputFile()/writePixels()
    # API, which silently drops every custom AND standard string attribute
    # (confirmed live: 0 of 12 metadata keys ever survive a round trip -- only
    # a C-level stderr "unknown attribute" print, never a Python exception).
    # The modern OpenEXR.File(header_dict, channels_dict) API persists them
    # correctly and infers HALF/FLOAT per channel from the numpy dtype.
    import OpenEXR
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    header: Dict[str, Any] = {
        "compression": getattr(OpenEXR, _COMP_MAP.get(compression, "ZIP_COMPRESSION")),
        "type": OpenEXR.scanlineimage,
    }
    if metadata:
        # Standard OpenEXR attributes and their Imath types
        for k, v in metadata.items():
            if k == "timeCode":
                tc = parse_timecode(str(v))
                if tc: header["timeCode"] = tc
            elif k in ("workflow", "prompt"):
                # ComfyUI provenance — stored under these exact (unprefixed)
                # names so the EXR carries its workflow like a ComfyUI PNG and
                # the js/radiance_exr_workflow.js drag-drop loader can find it.
                # NOTE: plain str, NOT bytes — the modern OpenEXR.File API
                # (unlike the legacy Header()/OutputFile() API this module
                # used to use) rejects bytes attribute values outright
                # ("unrecognized type of attribute"); it wants plain str.
                header[k] = v
            elif k in ["reelName", "capDate", "software", "comments", "owner"]:
                header[k] = v
            elif isinstance(v, (str, int, float)):
                # Prefix Radiance-specific metadata to avoid collisions unless it is standard or Cryptomatte
                key = k if (k.startswith("cryptomatte") or k.startswith("rad_")) else f"rad_{k}"
                header[key] = v

    np_dtype = np.float16 if pixel_type == "HALF" else np.float32
    channels_dict = {n: np.ascontiguousarray(d, dtype=np_dtype) for n, d in channels.items()}
    OpenEXR.File(header, channels_dict).write(filepath)

def write_exr_robust(filepath: str, image: np.ndarray, bit_depth: str = "32-bit Float", compression: str = "ZIP", metadata: Optional[Dict[str, Any]] = None) -> bool:
    # ALBABIT-FIX: was cv2 first, OpenEXR second, with a SimpleEXRWriter
    # last-resort fallback. Three problems: (1) write_exr_cv2() doesn't accept
    # a metadata parameter at all, so metadata was silently never attempted
    # whenever cv2's EXR codec succeeded first -- the opposite priority from
    # _save_exr() (nodes_io.py), which correctly tries OpenEXR first.
    # (2) SimpleEXRWriter is a hand-rolled binary EXR writer that produces
    # structurally invalid files (confirmed live: EXR_ERR_CORRUPT_CHUNK on
    # read-back) while this function still returned True -- silent success
    # on a corrupted delivery, the opposite of the "fail loudly" philosophy
    # this module states elsewhere. (3) OpenEXR>=3.2.0 is a declared hard
    # dependency, so a real EXR writer is expected to always be available.
    #
    # BUG FIX (this session): the channel unpacking below also unconditionally
    # assumed >=3 channels (image[..., 1], image[..., 2]) — any single-channel
    # pass (depth/Z, a matte, cryptomatte ID, etc.) raised IndexError here.
    # Single-channel passes are exactly what RadianceEXRPassesWriter routinely
    # sends through this path (e.g. its "depth" pass). Handle 1/2/3/4 channels
    # like nodes_io.py's _exr_channels() does, instead of assuming RGB.
    if image.ndim == 2:
        image = image[..., None]
    n_ch = image.shape[-1]
    if n_ch == 1:
        channels = {"R": image[..., 0], "G": image[..., 0], "B": image[..., 0]}
    elif n_ch == 2:
        channels = {"R": image[..., 0], "G": image[..., 1]}
    else:
        channels = {"R": image[..., 0], "G": image[..., 1], "B": image[..., 2]}
        if n_ch >= 4:
            channels["A"] = image[..., 3]
    ptype = "HALF" if "16" in str(bit_depth) else "FLOAT"
    if check_openexr_available():
        try:
            write_exr_openexr(filepath, channels, compression, ptype, metadata)
            return True
        except Exception as exc:
            logger.warning(f"[write_exr_robust] OpenEXR write failed ({exc}), trying OpenCV fallback.")
    if write_exr_cv2(filepath, image, bit_depth, compression):
        if metadata:
            logger.warning(f"[write_exr_robust] Wrote '{filepath}' via OpenCV -- metadata was NOT embedded (OpenCV's EXR writer doesn't support it).")
        return True
    logger.error(f"[write_exr_robust] Failed to write EXR '{filepath}': both OpenEXR and OpenCV failed or are unavailable.")
    return False


# ═══════════════════════════════════════════════════════════════════════════════
#               HDR METADATA BUILDER  (v2.4)
# ═══════════════════════════════════════════════════════════════════════════════

_RADIANCE_VERSION = "Radiance v3.0.0"

# Map of log-space names to a brief human-readable curve description embedded
# in the color_pipeline summary and the rad_hdr_curve attribute.
_LOG_CURVE_DESC: Dict[str, str] = {
    "ARRI LogC3":            "ARRI LogC3 (AWG3, ~800% EI, γ≈0.25)",
    "ARRI LogC4":            "ARRI LogC4 (AWG4, ~4500%)",
    "Sony S-Log3":           "Sony S-Log3 (S-Gamut3.Cine)",
    "Panasonic V-Log":       "Panasonic V-Log (V-Gamut)",
    "DaVinci Intermediate":  "DaVinci Intermediate",
    "RED Log3G10":           "RED Log3G10 (REDWideGamutRGB)",
    "Linear":                "Scene-Linear",
    "sRGB":                  "sRGB (display-referred)",
    "ACEScg":                "ACEScg (AP1 linear)",
    "ACES 2065-1":           "ACES 2065-1 (AP0 linear)",
    "Rec.2020 Linear":       "Rec.2020 Scene-Linear",
    "Raw":                   "Raw passthrough (no transform)",
}


def build_radiance_hdr_metadata(
    source_space: str = "Linear",
    hdr_mode: str = "Passthrough",
    decode_noise_scale: float = 0.0,
    exposure: float = 0.0,
    target_space: str = "sRGB",
    display_tonemap: str = "None",
    lora_name: str = "",
    lora_type: str = "IC-LoRA",
) -> Dict[str, Any]:
    """
    Build a standardised EXR attribute dictionary capturing the full HDR
    generation pipeline used by Radiance to produce this image.

    All Radiance-specific keys are prefixed with ``rad_`` to avoid collisions
    with standard OpenEXR attributes.  The two standard attributes
    ``software`` and ``comments`` are also set to give generic viewers a
    human-readable summary without any special tooling.

    Args:
        source_space:        Log/linear input space fed into the VAE encoder
                             (e.g. "ARRI LogC3", "Linear").
        hdr_mode:            Encode HDR mode ("Compress (Log)", "Passthrough",
                             "Clip", "Soft Clip").
        decode_noise_scale:  Latent noise injection scale used at decode time
                             (0 = disabled).
        exposure:            EV exposure shift applied before encoding.
        target_space:        Output color space from the decoder
                             (e.g. "ACEScg", "ARRI LogC4", "sRGB").
        display_tonemap:     Display tonemap operator applied ("Reinhard",
                             "ACES", "Filmic", "None").
        lora_name:           Name of the IC-LoRA / LoRA model used for
                             generation (empty string if none).
        lora_type:           LoRA variant type ("IC-LoRA", "LoRA", etc.).

    Returns:
        Dict mapping EXR attribute name → string value, ready to pass as the
        ``metadata`` argument to ``write_exr_openexr`` / ``write_exr_robust``.
    """
    curve_desc = _LOG_CURVE_DESC.get(source_space, source_space)

    # Build a compact human-readable pipeline summary
    pipeline_steps = [curve_desc]
    if hdr_mode != "Passthrough":
        pipeline_steps.append(f"HDR-{hdr_mode}")
    if decode_noise_scale > 0.0:
        pipeline_steps.append(f"noise={decode_noise_scale:.4f}")
    if exposure != 0.0:
        pipeline_steps.append(f"EV{exposure:+.2f}")
    pipeline_steps.append(f"→ {target_space}")
    if display_tonemap and display_tonemap.lower() not in ("none", ""):
        pipeline_steps.append(f"[TM:{display_tonemap}]")
    color_pipeline = " | ".join(pipeline_steps)

    now_iso = datetime.datetime.now().isoformat(timespec="seconds")

    meta: Dict[str, Any] = {
        # ── Standard OpenEXR attributes (readable by any viewer) ─────────────
        "software": _RADIANCE_VERSION,
        "comments": f"Generated by {_RADIANCE_VERSION}. Pipeline: {color_pipeline}",
        # ── Radiance-specific attributes (rad_ prefix) ────────────────────────
        "rad_generator":         _RADIANCE_VERSION,
        "rad_created":           now_iso,
        "rad_hdr_source_space":  source_space,
        "rad_hdr_mode":          hdr_mode,
        "rad_hdr_exposure":      f"{exposure:+.4f}",
        "rad_hdr_noise_scale":   f"{decode_noise_scale:.6f}",
        "rad_hdr_target_space":  target_space,
        "rad_hdr_tonemap":       display_tonemap if display_tonemap else "None",
        "rad_color_pipeline":    color_pipeline,
        "rad_hdr_curve":         curve_desc,
    }

    if lora_name:
        meta["rad_lora_name"] = lora_name
        meta["rad_lora_type"] = lora_type

    return meta


# ═══════════════════════════════════════════════════════════════════════════════
#                    EXR MULTI-PART WRITER (v2.4 Phase 5.4)
# ═══════════════════════════════════════════════════════════════════════════════

def write_exr_multipart_report(
    filepath: str,
    parts: Dict[str, np.ndarray],
    bit_depth: str = "16-bit Half Float",
    compression: str = "ZIP",
    metadata: Optional[Dict[str, Any]] = None,
    allow_fallback: bool = False,
) -> Dict[str, Any]:
    """
    Write a multi-part EXR v2 file with named layers.

    Each entry in `parts` is written as a separate named part:
      • "beauty"  → channels R, G, B (or R, G, B, A if 4-channel)
      • "depth"   → channel Z
      • "normal"  → channels NX, NY, NZ
      • "albedo"  → channels albedo.R, albedo.G, albedo.B
      • any key   → written with layered channel names (key.R, key.G, ...)

    Standard single-channel parts (keys starting with "depth" or "z") are written
    as a single Z channel.

    Compatibility:
      • Nuke 13+: connects automatically via ReadGeo / Read multi-part
      • DaVinci Resolve: requires "Flatten Layers" mode
      • Fusion: multi-part compatible natively

    Args:
        filepath: Output .exr path.
        parts:    Dict of part_name → numpy array (H, W, C) float32.
        bit_depth: "16-bit Half Float" or "32-bit Float".
        compression: EXR compression (ZIP, PIZ, ZIPS, etc.).
        metadata: Optional dict of string metadata embedded in each part header.

    Returns a receipt (FIX-015/016)::

        {"mode": "multipart" | "per_part" | "failed",
         "complete": bool,            # every requested part is on disk
         "files": [paths written],    # the real files, not the requested name
         "parts": {name: {"status": "written" | "failed" | "skipped",
                          "file": path or None, "error": str or None}},
         "error": str or None}

    With ``allow_fallback`` False (strict) a failed multi-part write writes
    nothing else and reports "failed". With it True, each part goes to its
    own ``<base>.<part>.exr`` and the receipt lists exactly those files.
    """
    receipt: Dict[str, Any] = {"mode": "failed", "complete": False, "files": [],
                               "parts": {}, "error": None}
    for _n, _img in (parts or {}).items():
        receipt["parts"][_n] = {"status": "skipped" if _img is None else "failed",
                                "file": None, "error": None if _img is not None else "no image"}
    requested = [n for n, img in (parts or {}).items() if img is not None]
    if not requested:
        logger.warning("[EXR MultiPart] No parts provided.")
        receipt["error"] = "no parts provided"
        return receipt

    ptype_str = "HALF" if "16" in bit_depth else "FLOAT"

    # ── Attempt via OpenEXR (most correct multi-part support) ─────────────────
    if check_openexr_available():
        try:
            # ALBABIT-FIX: migrated from Header()/MultiPartOutputFile()/
            # writePixels() (the legacy API), which silently drops every
            # custom AND standard string attribute -- same root cause as
            # write_exr_openexr() above. The modern API represents a
            # multi-part file as OpenEXR.File(list_of_Part), where each
            # Part(header_dict, channels_dict, name) carries its own metadata
            # correctly, and infers HALF/FLOAT per channel from the numpy dtype.
            import OpenEXR

            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            comp_value = getattr(OpenEXR, _COMP_MAP.get(compression, "ZIP_COMPRESSION"))
            np_dtype = np.float16 if ptype_str == "HALF" else np.float32

            exr_parts = []
            for part_name, img in parts.items():
                if img is None:
                    continue
                arr = np.asarray(img, dtype=np.float32)
                n_ch = arr.shape[2] if arr.ndim == 3 else 1

                header: Dict[str, Any] = {
                    "compression": comp_value,
                    "type": OpenEXR.scanlineimage,
                }

                # Build channel names
                is_depth = part_name.lower() in ("depth", "z", "zdepth", "depth_z")
                if is_depth or n_ch == 1:
                    ch_names = ["Z"]
                elif n_ch == 2:
                    ch_names = ["R", "G"]
                elif n_ch == 3:
                    ch_names = ["R", "G", "B"] if part_name == "beauty" else \
                               ["NX", "NY", "NZ"] if "normal" in part_name.lower() else \
                               [f"{part_name}.R", f"{part_name}.G", f"{part_name}.B"]
                elif n_ch >= 4:
                    if part_name == "beauty":
                        ch_names = ["R", "G", "B", "A"]
                    else:
                        ch_names = [f"{part_name}.R", f"{part_name}.G",
                                    f"{part_name}.B", f"{part_name}.A"]
                else:
                    ch_names = [part_name]

                if metadata:
                    for k, v in metadata.items():
                        if k in ("workflow", "prompt"):
                            # ComfyUI provenance, unprefixed, plain str — see
                            # write_exr_openexr()'s comment on why NOT bytes.
                            header[k] = v
                        elif k in ["timeCode", "reelName", "capDate", "software", "comments", "owner"]:
                            header[k] = v
                        elif isinstance(v, (str, int, float)):
                            key = k if (k.startswith("cryptomatte") or k.startswith("rad_")) else f"rad_{k}"
                            header[key] = v

                if arr.ndim == 2:
                    arr = arr[..., np.newaxis]
                channels_dict = {
                    cname: np.ascontiguousarray(
                        arr[..., ci] if ci < arr.shape[2] else arr[..., 0], dtype=np_dtype
                    )
                    for ci, cname in enumerate(ch_names)
                }
                exr_parts.append(OpenEXR.Part(header, channels_dict, part_name))

            OpenEXR.File(exr_parts).write(filepath)

            # Read the part names back: the receipt reports what is on disk.
            written_names = [p.name() for p in OpenEXR.File(filepath).parts]
            for name in requested:
                ok = name in written_names
                receipt["parts"][name] = {"status": "written" if ok else "failed",
                                          "file": filepath if ok else None,
                                          "error": None if ok else "part missing after write"}
            receipt["files"] = [filepath]
            receipt["complete"] = all(n in written_names for n in requested)
            receipt["mode"] = "multipart"
            if not receipt["complete"]:
                receipt["error"] = "parts missing after write: " + ", ".join(
                    n for n in requested if n not in written_names)
            logger.info(f"[EXR MultiPart] Written {len(written_names)} parts → {filepath}")
            return receipt

        except Exception as e:
            receipt["error"] = f"OpenEXR multi-part failed: {e}"
            if not allow_fallback:
                logger.error(f"[EXR MultiPart] {receipt['error']}")
                return receipt
            logger.warning(f"[EXR MultiPart] OpenEXR multi-part failed ({e}), falling back to per-layer files.")
    else:
        receipt["error"] = "OpenEXR is not installed"
        if not allow_fallback:
            logger.error("[EXR MultiPart] OpenEXR is not installed; multi-part EXR unavailable.")
            return receipt

    # ── Fallback: write separate EXR files per part ───────────────────────────
    base, _ = os.path.splitext(filepath)
    receipt["mode"] = "per_part"
    for part_name in requested:
        layer_path = f"{base}.{part_name}.exr"
        arr = np.asarray(parts[part_name], dtype=np.float32)
        try:
            ok = bool(write_exr_robust(layer_path, arr, bit_depth, compression, metadata))
            err = None if ok else "writer returned False"
        except Exception as exc:  # noqa: BLE001 - recorded per part
            ok, err = False, str(exc)
        ok = ok and os.path.isfile(layer_path)
        receipt["parts"][part_name] = {"status": "written" if ok else "failed",
                                       "file": layer_path if ok else None, "error": err}
        if ok:
            receipt["files"].append(layer_path)
            logger.info(f"[EXR MultiPart] Fallback→ wrote {layer_path}")

    receipt["complete"] = all(receipt["parts"][n]["status"] == "written" for n in requested)
    logger.warning(
        f"[EXR MultiPart] Wrote {len(receipt['files'])} of {len(requested)} parts as separate EXR files "
        f"instead of one multi-part file. Load each file individually in Nuke."
    )
    return receipt


def write_exr_multipart(
    filepath: str,
    parts: Dict[str, np.ndarray],
    bit_depth: str = "16-bit Half Float",
    compression: str = "ZIP",
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """True only when ``filepath`` is a multi-part EXR holding every part.

    It used to return True after silently writing per-part files under other
    names (or only some of them), so callers reported a file that did not
    exist. Use ``write_exr_multipart_report`` for the fallback and receipt.
    """
    r = write_exr_multipart_report(filepath, parts, bit_depth, compression, metadata,
                                   allow_fallback=False)
    return r["mode"] == "multipart" and r["complete"]
