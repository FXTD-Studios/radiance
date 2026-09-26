# ruff: noqa: F821
# Many names (image_video_type, MAX_IMAGE_DIMENSION, apply_grading, the LUT/IDT
# helpers, etc.) are injected into this module's globals at runtime via
# `globals().update({... getattr(_viewer_utils, name) ...})` below. Static linters
# cannot see through that, so F821 (undefined name) is disabled for this file.
import json
import struct
import torch
import numpy as np
import zlib
import os
import uuid
import wave
import logging
import math
import sys
import io
import traceback
try:
    from aiohttp import web
    from server import PromptServer
except ImportError:  # pragma: no cover - exercised by the CI import smoke-test
    # Guarded because importing any module in this package runs this file, and
    # a bare `from aiohttp import web` therefore made the whole Review group
    # unimportable without a running ComfyUI. `nodes/gizmo.py` and
    # `nodes/pipeline/workspace.py` have always guarded it for the same reason;
    # this one did not, and the root `nodes_realtime_preview.py` shim hid that
    # by being importable on its own — until it was retired into this package.
    #
    # aiohttp is a required dependency of ComfyUI, so this never degrades a
    # real install; it keeps the module readable by tooling that has neither.
    web = None
    PromptServer = None
import re

# v3.1: Enable OpenEXR support in OpenCV
# Essential for 32-bit float export
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
from typing import Dict, Any, Optional, List, Tuple

import folder_paths

# Import safe path utilities
from radiance.path_utils import safe_join

# v3.1: Robust EXR Writer from HDR bridge
try:
    from radiance.hdr.io import write_exr_robust
except ImportError:
    write_exr_robust = None

# v1.21: cv2 for 16-bit PNG support
try:
    import cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    logging.getLogger("radiance.viewer").warning(
        "cv2 not available — 16-bit PNG disabled, falling back to 8-bit. "
        "Install opencv-python for 16-bit support."
    )

# Module logger
logger = logging.getLogger("radiance.viewer")

# Import shared ACES gamut compression — replaces inline duplicate implementation
from radiance.color.gamut import aces2_gamut_compress as _aces2_gamut_compress


# ═══════════════════════════════════════════════════════════════════════════════
#                           CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

from radiance import viewer_utils as _viewer_utils

_VIEWER_UTIL_NAMES = (
    "DEFAULT_PNG_COMPRESSION", "BIT_16_MAX", "BIT_8_MAX", "BIT_16_TO_8_DIVISOR",
    "MAX_IMAGE_DIMENSION", "MAX_BATCH_SIZE", "PICK_MAX_DIM", "RHDR_MAGIC", "RPICK_MAGIC",
    "BIT_DEPTH_MODES", "CV2_PNG_COMPRESSION", "_VIEWER_CACHE_MAX",
    "_VIEWER_CACHE_LOCK", "_viewer_cache_set", "_viewer_cache_get",
    "_VIEWER_PROGRESS", "_VIEWER_PROGRESS_LOCK", "_progress_set", "_progress_get",
    "_save_pick_buffer", "build_cdl_xml", "LUT_MODES", "_lut_srgb", "_lut_rec709",
    "_lut_filmic", "_lut_reinhard", "_lut_aces_filmic", "_M_SRGB_TO_ACESCG",
    "_M_SRGB_TO_AP0", "_M_ACESCG_TO_AP0", "_M_ACESCG_TO_LIN_SRGB", "_lut_acescg",
    "_lut_aces2065", "_lut_acescct_encode", "_lut_logc3", "_lut_logc4", "_lut_slog3",
    "_lut_vlog", "_lut_flog2", "_lut_clog3", "_lut_log3g10", "_lut_davinci_intermediate",
    "_lut_bmd_gen5", "_lut_nlog", "_lut_lin_to_log", "_lut_log_to_lin", "_idt_logc3",
    "_idt_logc4", "_idt_slog3", "_idt_vlog", "_idt_flog2", "_idt_clog3", "_idt_log3g10",
    "_idt_davinci_intermediate", "_idt_bmd_gen5", "_idt_nlog", "_lut_passthrough",
    "_LUT_FUNCTIONS", "_lut_false_color", "_lut_clip_check", "apply_lut", "apply_grading",
    "_kelvin_to_rgb_multipliers", "safe_tensor_to_numpy", "compute_data_range",
    "save_16bit_png", "RadianceType", "image_video_type",
)
globals().update({name: getattr(_viewer_utils, name) for name in _VIEWER_UTIL_NAMES})

# ── 3.5.0: source colour tagging ───────────────────────────────────────────────
# The viewer used to receive every frame as fp32 with no tag and treat all float
# data as scene-linear, so an ordinary sRGB-encoded ComfyUI IMAGE was gamma-
# encoded a second time (washed out), and the frontend fell back to guessing a
# camera log curve from the frame's median. Every frame is now tagged; the names
# are OpenColorIO ACES studio-config colour spaces, so the frontend hands them
# straight to OCIO.
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


def _video_fps(value: Any) -> Optional[float]:
    """Frame rate of a ComfyUI VIDEO input, None for a plain IMAGE batch."""
    try:
        if hasattr(value, "get_components"):
            rate = getattr(value.get_components(), "frame_rate", None)
            if rate:
                return float(rate)
    except Exception:  # noqa: BLE001
        return None
    return None


def _save_preview_audio(audio: Any, output_dir: str) -> Optional[str]:
    """Write a VIDEO's audio as browser-readable PCM without changing its source."""
    if not audio:
        return None
    samples = audio["waveform"][0].detach().to(device="cpu", dtype=torch.float32)
    pcm = (samples.transpose(0, 1).clamp(-1, 1).numpy() * 32767).astype("<i2")
    filename = f"radiance_audio_{uuid.uuid4().hex}.wav"
    with wave.open(os.path.join(output_dir, filename), "wb") as stream:
        stream.setnchannels(samples.shape[0])
        stream.setsampwidth(2)
        stream.setframerate(int(audio["sample_rate"]))
        stream.writeframes(pcm.tobytes())
    return filename


def _viewer_cache_has_node(unique_id: Any) -> bool:
    """True when the delivery cache still holds a frame set for this node."""
    if unique_id is None or not str(unique_id).strip():
        return False
    node = str(unique_id).strip()
    with _viewer_utils._VIEWER_CACHE_LOCK:
        keys = list(_viewer_utils._VIEWER_CACHE.keys()) if hasattr(_viewer_utils, "_VIEWER_CACHE") \
            else []
    return any(k == node or str(k).endswith(":" + node) for k in keys)


def _fp_tensor(value: Any) -> Any:
    if hasattr(value, "get_components"):
        try:
            components = value.get_components()
            audio = getattr(components, "audio", None)
            return (_fp_tensor(components.images), str(components.frame_rate),
                    (_fp_tensor(audio["waveform"]), audio["sample_rate"]) if audio else None)
        except Exception:  # noqa: BLE001
            return repr(type(value))
    if isinstance(value, dict) and "samples" in value:
        value = value["samples"]
    if isinstance(value, torch.Tensor):
        flat = value.detach().reshape(-1)
        step = max(1, flat.numel() // 65536)
        sample = flat[::step].double()
        return (tuple(value.shape), str(value.dtype), round(float(sample.sum()), 6),
                round(float(sample.abs().sum()), 6))
    return repr(value)


def _viewer_fingerprint(image: Any, other: Dict[str, Any]) -> str:
    """Identity of what the viewer would show: inputs plus widget values."""
    parts = [_fp_tensor(image)]
    for k in sorted(other):
        if k in ("prompt", "extra_pnginfo"):
            continue
        parts.append((k, _fp_tensor(other[k])))
    return repr(parts)


# BUG-FIX: compiled at module level — was previously compiled inside the
# hot delivery handler on every request, wasting time on every call.
_SAFE_FILENAME_RE = re.compile(r'[^\w\s◎_.() -]', re.UNICODE)

# ── Viewer temp-file bookkeeping ───────────────────────────────────────────────
# Every execution wrote a fresh uuid4-named set of .rhdr/.exr/.png/.rpick files
# and nothing ever removed them: there was no unlink/rmtree for viewer temp files
# anywhere in the package. With bracketing forced on that is 12 files per frame,
# so scrubbing a slider and re-queueing a 240-frame shot ten times left ~30,000
# orphaned files and >100 GB in the temp directory until ComfyUI restarted.
_VIEWER_TEMP_FILES: Dict[str, List[str]] = {}
_VIEWER_TEMP_LOCK = __import__("threading").Lock()


def _viewer_track_temp(instance_key: str, paths: List[str]) -> None:
    """Record the files written for this instance's current generation."""
    if not instance_key:
        return
    with _VIEWER_TEMP_LOCK:
        _VIEWER_TEMP_FILES[instance_key] = list(paths)


def _viewer_purge_temp(instance_key: str) -> int:
    """Delete the previous generation's files for this instance. Returns the count."""
    if not instance_key:
        return 0
    with _VIEWER_TEMP_LOCK:
        stale = _VIEWER_TEMP_FILES.pop(instance_key, [])
    removed = 0
    for path in stale:
        try:
            os.unlink(path)
            removed += 1
        except FileNotFoundError as _exc:
            logger.debug(
                "[Radiance] _viewer_purge_temp(): ignoring %s from `os.unlink(path)`: %s",
                type(_exc).__name__, _exc,
            )
        except OSError as exc:
            logger.debug("[Radiance] Could not remove stale viewer temp %s: %s", path, exc)
    if removed:
        logger.debug("[Radiance] Purged %d stale viewer temp file(s) for %s", removed, instance_key)
    return removed


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
                "image": (image_video_type, {
                    "tooltip": "IMAGE batch or VIDEO to review. Returned unchanged as an IMAGE; "
                               "input_space decides how it is displayed."}),
            },
            "optional": {
                # ── Compare / Depth ──
                "compare_image": (image_video_type, {
                    "tooltip": "Optional B side (IMAGE or VIDEO) for the viewer's wipe and compare "
                               "modes. Every frame is sent; its display space is resolved like image's."}),
                "zdepth": (
                    image_video_type,
                    {"tooltip": "Z-Depth map to display when pressing Z button"},
                ),
                # ── Exposure bracketing ────────────────────────────────────
                # This was a function default of True with no entry in
                # INPUT_TYPES at all, so there was no widget and no way to turn
                # it off. Every frame was processed three times, and the two
                # bracket passes each wrote a .rhdr, a 32-bit .exr and a .rpick
                # that nothing ever reads: the JS only requests the bracket
                # PNG (imgData.filename), and the sole consumer of the rest was
                # the status string "Low / High cached". 500 frames at 1080p was
                # ~67 GB into temp/, two thirds of it for that label.
                #
                # It is now a visible control, it defaults off, and when it is
                # on it writes only the PNG the viewer actually fetches.
                "exposure_bracketing": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "on",
                        "label_off": "off",
                        "tooltip": (
                            "Write -2 EV / +2 EV preview thumbnails for the analysis shelf. "
                            "One extra PNG per frame per bracket; off by default because a "
                            "long shot pays for it on disk."
                        ),
                    },
                ),
                # ── 3.5.0: what the pixels are, how they travel, how fast ──
                # Appended after exposure_bracketing so saved widget values keep
                # their positions.
                "input_space": (
                    VIEWER_INPUT_SPACES,
                    {
                        "default": "Auto",
                        "tooltip": (
                            "What the incoming pixels are. A ComfyUI IMAGE is sRGB-encoded and is "
                            "shown untouched, exactly as ComfyUI previews it. Linear sources (HDR "
                            "Decode, SDR -> HDR, Read in a linear working space) are shown through "
                            "OpenColorIO's ACES 2.0 SDR view. Auto: linear when any value is above "
                            "1.0 or below 0, otherwise sRGB. Set it explicitly for linear material "
                            "that stays inside 0-1."
                        ),
                    },
                ),
                "float_precision": (
                    ["Half (16-bit)", "Full (32-bit)"],
                    {
                        "default": "Half (16-bit)",
                        "tooltip": (
                            "Precision of the float frames sent to the browser. Half is what "
                            "OpenEXR, RV and Nuke's viewer cache use: exact for display and "
                            "grading, half the size (about 60 MB less per 4K frame). Full keeps "
                            "every fp32 bit for values above 65504 or bit-exact probing."
                        ),
                    },
                ),
                "fps": (
                    "FLOAT",
                    {
                        "default": 0.0, "min": 0.0, "max": 240.0, "step": 0.001,
                        "tooltip": (
                            "Playback rate. 0 = the source's rate (a VIDEO input carries it), "
                            "24 for an image batch. 23.976 / 29.97 / 59.94 are handled as the "
                            "NTSC rates."
                        ),
                    },
                ),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            },
        }

    CATEGORY = "FXTD STUDIOS/Radiance/◎ Display"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "view"
    OUTPUT_NODE = True
    DESCRIPTION = """Radiance Viewer, VFX review for scene-linear and HDR images:
• Simple mode (picture, compare, playback) or Advanced (every panel), switched in its title bar
• Compare against compare_image or a pinned frame: B, wipe, difference, blink
• GPU Waveform / RGB Parade / Vectorscope / Histogram scopes
• Power Windows masking (Radial + Box, feather, rotation)
• Comparison Bridge — mouse-draggable wipe + reference shelf (8 stills)
• Anamorphic lens streaks + Brown-Conrady k1/k2 distortion
• Edge-preserving Bilateral Filter denoising (7×7 GPU kernel)
• Channel viewing (RGB/R/G/B/Alpha/Luma), False Color, Zebra
• 16-bit PNG + .rhdr HDR sidecar + .exr export
• IMAGE passthrough — no longer a dead-end node"""

    @classmethod
    def IS_CHANGED(cls, image=None, unique_id=None, **kwargs):
        """
        Always re-execute.

        This is an OUTPUT_NODE with side effects: it writes preview/RHDR/EXR
        files and — critically — repopulates the process-global frame cache that
        `/radiance/deliver` reads from. Without an IS_CHANGED, ComfyUI caches the
        result and skips the node on re-queue, so once a viewer's entry is
        evicted from the 8-slot cache the delivery endpoint answers "No frames
        found in cache for this node. Run the workflow first." forever: re-queuing
        skips the unchanged node, nothing repopulates the cache, and the only
        escape is to perturb an upstream widget. NaN is ComfyUI's documented
        always-dirty sentinel.

        3.5.0: NaN on every queue also marked every node downstream of the
        viewer dirty (the viewer passes its IMAGE through), so an unchanged
        graph re-ran everything after it. Now: NaN only while this viewer has
        nothing in the delivery cache; otherwise a fingerprint of what it would
        show, so an unchanged viewer is skipped and so is everything after it.
        """
        try:
            if not _viewer_cache_has_node(unique_id):
                return float("NaN")
            return _viewer_fingerprint(image, kwargs)
        except Exception:  # noqa: BLE001 - never block execution on a fingerprint
            return float("NaN")

    def view(
        self,
        image: Any,
        bit_depth: str = "16-bit Half",
        # Compare / Depth
        compare_image: Optional[Any] = None,
        zdepth: Optional[Any] = None,
        # Defaults to off and now matches the INPUT_TYPES widget. It used to
        # default to True with no widget behind it, so it could not be turned
        # off from the graph.
        exposure_bracketing: bool = False,
        input_space: str = "Auto",
        float_precision: str = "",
        fps: float = 0.0,
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
                except Exception as exc:
                    logger.warning("[nodes_radiance_viewer] _extract_tensor: %s", exc)
            if isinstance(x, dict) and "samples" in x:
                return x["samples"]
            if isinstance(x, (list, tuple)) and len(x) > 0 and isinstance(x[0], torch.Tensor):
                return x[0]
            return x

        source_audio = None
        if hasattr(image, "get_components"):
            components = image.get_components()
            source_fps = float(components.frame_rate) if components.frame_rate else None
            source_audio = getattr(components, "audio", None)
            image = components.images
        else:
            source_fps = _video_fps(image)
            image = _extract_tensor(image)
        if compare_image is not None:
            compare_image = _extract_tensor(compare_image)
        if zdepth is not None:
            zdepth = _extract_tensor(zdepth)

        # 3.5.0: float_precision is the visible widget. The legacy bit_depth
        # keyword is still accepted from scripts. Half is the default: RHDR
        # does not need OpenCV, so the old HAS_CV2 gate on it is gone.
        if float_precision:
            bit_depth = "32-bit Float" if float_precision.startswith("Full") else "16-bit Half"
        use_16bit = "16-bit" in bit_depth
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
                "result": (image,),
            }

        source_encoding, source_colorspace = resolve_viewer_input_space(input_space, image)
        play_fps = float(fps) if fps and fps > 0 else (source_fps or 24.0)

        try:
            output_dir = folder_paths.get_temp_directory()
            batch_size = image.shape[0] if image.dim() == 4 else 1
            # No hard cap — process all frames so video sequences play fully.
            if batch_size > 500:
                logger.warning(
                    f"[Viewer] Large batch ({batch_size} frames). "
                    "Processing all frames — this may take a moment."
                )

            images_list: List[Dict[str, Any]] = []

            # Purge the previous generation's temp files for this viewer before
            # writing a new one, so repeated re-queues don't accumulate.
            _purge_key = str(unique_id).strip() if unique_id and str(unique_id).strip() else str(id(self))
            _viewer_purge_temp(_purge_key)

            view_space = (source_encoding, source_colorspace)
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
                        view_space=view_space,
                    )
                    if frame_result is not None:
                        frame_result["frame"] = frame_idx
                        frame_result["total_frames"] = batch_size
                        images_list.append(frame_result)
                        
                    # ── Exposure Bracketing ────────────────────────────────────
                    if exposure_bracketing and frame_result is not None:
                        # Low (-2 EV)
                        # Slice FIRST. These used to multiply the ENTIRE batch
                        # inside the per-frame loop while _process_frame reads
                        # only image[frame_idx] -- 2 full-batch allocations per
                        # frame, i.e. O(B^2) memory traffic. A 120-frame 1080p
                        # RGBA fp32 plate moved ~950 GB for one preview, and
                        # OOM'd outright on a CUDA-resident tensor.
                        #
                        # preview_only: the bracket passes write the PNG and
                        # nothing else. `bracketImages.forEach` in
                        # js/radiance_viewer.js requests only imgData.filename,
                        # so the bracket .rhdr, .exr and .rpick were written,
                        # never fetched, and never deleted until the next
                        # execution purged them. At 1080p that was ~45 GB of the
                        # ~67 GB a 500-frame shot put into temp/.
                        low_res = self._process_frame(
                            image[frame_idx:frame_idx + 1] * 0.25, 0, output_dir,
                            use_16bit=use_16bit, use_32bit=use_32bit, save_hdr_sidecar=save_hdr_sidecar,
                            prefix="◎ Radiance_bracket_low", preview_only=True, view_space=view_space
                        )
                        if low_res:
                            # v4.5 FIX: type must be "temp" — ComfyUI /api/view only
                            # accepts temp/output/input. Custom strings ("bracket_low")
                            # return HTTP 400, making the subsequent RHDR fetch receive
                            # an HTML error page instead of binary data, which causes
                            # RangeError in _parseHDRBuffer. Use bracket_label for
                            # viewer identification instead.
                            low_res["type"] = "temp"
                            low_res["bracket_label"] = "low"
                            low_res["frame"] = frame_idx
                            low_res["total_frames"] = batch_size
                            images_list.append(low_res)
                            
                        # High (+2 EV)
                        high_res = self._process_frame(
                            image[frame_idx:frame_idx + 1] * 4.0, 0, output_dir,
                            use_16bit=use_16bit, use_32bit=use_32bit, save_hdr_sidecar=save_hdr_sidecar,
                            prefix="◎ Radiance_bracket_high", preview_only=True, view_space=view_space
                        )
                        if high_res:
                            high_res["type"] = "temp"
                            high_res["bracket_label"] = "high"
                            high_res["frame"] = frame_idx
                            high_res["total_frames"] = batch_size
                            images_list.append(high_res)

                except (RuntimeError, ValueError) as e:
                    logger.warning(f"Error processing frame {frame_idx}: {e}")
                    continue

            # Compare image
            if compare_image is not None:
                compare_list = self._process_compare_image(
                    compare_image,
                    output_dir,
                    use_16bit=use_16bit,
                    use_32bit=use_32bit,          # BUG-FIX (BUG-2): was omitted — compare always saved fp16
                    save_hdr_sidecar=save_hdr_sidecar,
                    view_space=resolve_viewer_input_space(input_space, compare_image),
                )
                images_list.extend(compare_list)

            # Z-Depth: data, never through a view transform (see _process_zdepth_image).
            if zdepth is not None:
                zdepth_list = self._process_zdepth_image(
                    zdepth,
                    output_dir,
                    use_16bit=use_16bit,
                    use_32bit=use_32bit,          # BUG-FIX (BUG-3): was omitted — depth sidecar always fp16
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

            # Record every file this generation wrote so the next execution can
            # remove it. Entries carry "filename" plus optional hdr/pick sidecars.
            #
            # BUG-FIX: "hdr_sidecar" was missing from this tuple. The zdepth
            # branch stores its RHDR under that key only (_process_zdepth_image
            # never sets hdr_filename), so every execution with a zdepth input
            # orphaned one .rhdr per frame, permanently. That is exactly the
            # leak _viewer_track_temp was written to close. _process_frame sets
            # both keys to the same name, so the set() keeps it to one unlink.
            _written: List[str] = []
            _seen: set = set()
            for _entry in images_list:
                if not isinstance(_entry, dict):
                    continue
                for _k in ("filename", "hdr_filename", "hdr_sidecar",
                           "pick_filename", "exr_filename"):
                    _fn = _entry.get(_k)
                    if _fn and str(_fn) not in _seen:
                        _seen.add(str(_fn))
                        _written.append(os.path.join(output_dir, str(_fn)))
            audio_filename = _save_preview_audio(source_audio, output_dir)
            if audio_filename:
                _written.append(os.path.join(output_dir, audio_filename))
            _viewer_track_temp(_purge_key, _written)

            # ── v4.2: Delivery Cache (Update node frames) ───────────
            # Use ComfyUI's unique_id (stable graph node ID) as the cache key.
            # v4.1 used str(id(self)) which is the CPython memory address — this
            # changes every execution and is lost on module reload, causing
            # "No cached frame found" on delivery.  unique_id matches
            # this.node.id in LiteGraph JS, so the fallback also works.
            # BUG-FIX: empty string is falsy — an empty unique_id must also
            # fall through to the memory-address fallback, not be used as-is.
            # BUG-FIX: unique_id alone is only unique WITHIN one graph. The cache
            # is a process-global, so two workflows that each happen to have a
            # viewer at node id 5 shared the key "5": running the second
            # overwrote the first, and pressing Render in the first tab then
            # exported the OTHER shot's plate, versioned under this shot's
            # filename, with no error and an HTTP 200 "success". Prefixing with
            # the prompt id scopes the key to one execution of one graph. The
            # key round-trips through the UI payload below and comes back on
            # /radiance/deliver, so the JS needs no change -- it just echoes it.
            _node_part = str(unique_id).strip() if unique_id and str(unique_id).strip() else str(id(self))
            _graph_part = ""
            try:
                if isinstance(prompt, dict):
                    _graph_part = str(prompt.get("prompt_id") or prompt.get("client_id") or "")
                if not _graph_part and isinstance(extra_pnginfo, dict):
                    _wf = extra_pnginfo.get("workflow") or {}
                    if isinstance(_wf, dict):
                        _graph_part = str(_wf.get("id") or "")
            except Exception:
                _graph_part = ""
            instance_key = f"{_graph_part}:{_node_part}" if _graph_part else _node_part
            _viewer_cache_set(instance_key, image)

            # ── v6.2: Flicker Heatmap & Cut Markers ─────────────────
            flicker_data = [] # Normalized deltas [0..1]
            cut_indices = []  # Frames indices with cuts
            
            # 1. Calculate automatic flicker data & scene cut markers
            if image.dim() == 4:
                # Optimized torch-side luma calculation
                # B-9 FIX: Safely slice/expand image to RGB format to prevent PyTorch broadcasting crashes on 1-ch grayscale or 4-ch RGBA
                B_img, H_img, W_img, C_img = image.shape
                if C_img == 1:
                    image_rgb = image.expand(B_img, H_img, W_img, 3)
                elif C_img == 4:
                    image_rgb = image[..., :3]
                else:
                    image_rgb = image
                
                coeffs = torch.tensor([0.2126, 0.7152, 0.0722], device=image.device).view(1, 1, 1, 3)
                luma_batch = torch.sum(image_rgb * coeffs, dim=-1) # (B, H, W)
                mean_lumas = torch.mean(luma_batch, dim=(1, 2)).cpu().numpy() # (B,)
                
                if len(mean_lumas) > 1:
                    # Calculate absolute deltas
                    deltas = np.abs(np.diff(mean_lumas))
                    # Normalize: typical flicker is small, so we want to be sensitive.
                    # 0.1 (10% shift) should be near maximum intensity (magenta).
                    flicker_data = [0.0] + (np.clip(deltas / 0.1, 0.0, 1.0)).tolist()
                    
                    # 2. Automated scene cut detection (luma shift > 12% indicates high probability of a camera shot change)
                    for idx in range(1, len(mean_lumas)):
                        if abs(mean_lumas[idx] - mean_lumas[idx - 1]) > 0.12:
                            cut_indices.append(idx)
                else:
                    flicker_data = [0.0]

            # 3.5.0: every frame says what it is, so the viewer never guesses.
            cmp_enc, cmp_cs = (resolve_viewer_input_space(input_space, compare_image)
                               if compare_image is not None else (source_encoding, source_colorspace))
            for _entry in images_list:
                if not isinstance(_entry, dict) or _entry.get("is_zdepth"):
                    continue
                if _entry.get("is_compare"):
                    _entry["source_encoding"], _entry["source_colorspace"] = cmp_enc, cmp_cs
                else:
                    _entry["source_encoding"] = source_encoding
                    _entry["source_colorspace"] = source_colorspace

            ui_payload = {
                "radiance_images": images_list,
                "source_encoding": [source_encoding],
                "source_colorspace": [source_colorspace],
                "fps": [play_fps],
                "audio": [{"filename": audio_filename, "subfolder": "", "type": "temp"}] if audio_filename else [],
                "audio_fps": [source_fps or play_fps],
                "instance_id": [instance_key],
                "batch_size": [batch_size],
                "bit_depth": [depth_label],
                "flicker_data": flicker_data,
                "cut_indices": cut_indices,
            }

            return {
                "ui": ui_payload,
                "result": (output_image,),
            }

        except (RuntimeError, ValueError, TypeError) as e:
            logger.error(f"Error in viewer: {e}")
            return {
                "ui": {"radiance_images": [], "error": [str(e)]},
                "result": (image,),
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
        preview_only: bool = False,
        view_space: Tuple[str, str] = ("srgb", SRGB_COLORSPACE),
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single frame: grade → LUT → save at selected bit depth.

        preview_only writes the 8-bit PNG and nothing else. Used by the
        exposure-bracket passes, whose only consumer on the JS side is
        imgData.filename: the bracket .rhdr / .exr / .rpick were written for
        every frame and fetched by nothing.
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
        d_min, d_max, has_hdr, hdr_stats = compute_data_range(frame)

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

        if preview_only:
            pass  # bracket pass: PNG only, nothing reads the float sidecar
        elif use_32bit:
            # ── 32-bit Float path: full IEEE 754 fp32, flags=1 ────────────────
            try:
                rhdr_filepath = safe_join(output_dir, rhdr_filename)
                fp32_bytes = frame_to_save.astype(np.float32).tobytes()
                compressed = zlib.compress(fp32_bytes, level=1)  # level 1: float data barely compresses
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
                # Clamp to the fp16 range: 1e5 used to become +inf.
                fp16_data = np.clip(frame_to_save, -65504.0, 65504.0).astype(np.float16).tobytes()
                compressed = zlib.compress(fp16_data, level=1)  # level 1: float data barely compresses
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
        
        if write_exr_robust and not preview_only:
            try:
                exr_filepath = safe_join(output_dir, exr_filename)
                # B-10 FIX: Convert to 3-channel RGB array safely regardless of input dimensions (2D vs 3D, grayscale vs RGBA) to prevent indexing crashes
                if frame_to_save.ndim == 2:
                    frame_for_exr = np.stack([frame_to_save] * 3, axis=-1)
                elif frame_to_save.shape[2] == 4:
                    frame_for_exr = frame_to_save[..., :3]
                elif frame_to_save.shape[2] == 1:
                    frame_for_exr = np.stack([frame_to_save[..., 0]] * 3, axis=-1)
                else:
                    frame_for_exr = frame_to_save
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

        # 3.5.0: the PNG is display-referred through the same view the float
        # path shows by default. It used to be x/(1+x) with no display
        # encoding for HDR frames (far too dark) and untouched otherwise, so a
        # linear frame that stayed under 1.05 was written as if it were sRGB.
        _enc, _cs = view_space
        preview_image = frame
        _preview_display = False
        if _enc == "linear":
            th0, tw0 = frame.shape[:2]
            if max(th0, tw0) > 2048 and HAS_CV2:
                # Transform after the downscale: ACES 2.0 on the CPU is the
                # expensive step and it only needs the preview's pixels.
                _sc = 2048 / max(th0, tw0)
                frame_small = cv2.resize(frame.astype(np.float32),
                                         (int(tw0 * _sc), int(th0 * _sc)),
                                         interpolation=cv2.INTER_AREA)
            else:
                frame_small = frame
            from radiance.color.display_preview import display_preview
            preview_image = display_preview(frame_small, "linear", _cs)
            _preview_display = True

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

        channel_names = ["Y"] if c_frame == 1 else ["R", "G", "B", "A"][:c_frame]
        pixel_type = "FLOAT" if use_32bit else ("HALF" if use_16bit else "UINT8")
        frame_metadata = {
            "container": "RHDR" if rhdr_saved else "PNG",
            "compression": "ZLIB" if rhdr_saved else "PNG",
            "pixelType": pixel_type,
            "source": "ComfyUI tensor",
            "exrCompression": "ZIP" if exr_saved else None,
            "channels": [
                {
                    "name": name,
                    "pixelType": pixel_type,
                    "xSampling": 1,
                    "ySampling": 1,
                }
                for name in channel_names
            ],
        }

        # What the PNG fallback actually is, so the viewer's status bar can say
        # so instead of claiming FP32 over it. The viewer reaches the PNG from
        # three paths (no DecompressionStream, RHDR integrity mismatch, texture
        # creation failure) and used to report "FP32 · RGBA32F" regardless,
        # which let a colourist grade a 4K plate against a 2048px tonemapped
        # 8-bit proxy and then press RENDER.
        preview_h, preview_w = preview_thumb.shape[:2]
        result: Dict[str, Any] = {
            "filename": png_filename,
            "subfolder": "",
            "type": "temp",
            "source_width": int(w_frame),
            "source_height": int(h_frame),
            "preview_width": int(preview_w),
            "preview_height": int(preview_h),
            # The PNG is display-referred (ACES 2.0 SDR for linear sources,
            # untouched for sRGB): the viewer must not transform it again.
            "preview_tonemapped": bool(_preview_display),
            "preview_encoding": "display",
            "data_range": [d_min, d_max],
            "hdr_stats": hdr_stats,
            "has_hdr": has_hdr,
            "channel_names": channel_names,
            "metadata": frame_metadata,
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
        if preview_only:
            return result
        try:
            pick_filepath = safe_join(output_dir, pick_filename)
            if _save_pick_buffer(frame, pick_filepath):
                result["pick_filename"] = pick_filename
                logger.debug(f"[Radiance] Pick buffer saved: {pick_filename}")
        except ValueError as e:
            logger.debug(f"[Radiance] Pick buffer path error: {e}")

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
        use_32bit: bool = False,          # BUG-FIX (BUG-2): was missing — compare always saved fp16
        save_hdr_sidecar: bool = False,
        view_space: Tuple[str, str] = ("srgb", SRGB_COLORSPACE),
    ) -> List[Dict[str, Any]]:
        """Process comparison image. Returns list of image metadata."""
        result: List[Dict[str, Any]] = []

        validation_error = self._validate_image(compare_image, "compare_image")
        if validation_error:
            logger.warning(validation_error)
            return result

        cmp_batch = compare_image.shape[0] if compare_image.dim() == 4 else 1
        # No artificial frame cap on compare channel

        for cmp_idx in range(cmp_batch):
            try:
                frame_result = self._process_frame(
                    compare_image,
                    cmp_idx,
                    output_dir,
                    use_16bit=use_16bit,
                    use_32bit=use_32bit,          # BUG-FIX (BUG-2)
                    save_hdr_sidecar=save_hdr_sidecar,
                    prefix="◎ Radiance_compare",
                    view_space=view_space,
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
        use_32bit: bool = False,          # BUG-FIX (BUG-3): was missing — depth sidecar always wrote fp16
        save_hdr_sidecar: bool = False,
    ) -> List[Dict[str, Any]]:
        """Process Z-depth map. 16-bit = 65536 depth levels via cv2."""
        result: List[Dict[str, Any]] = []

        validation_error = self._validate_image(zdepth, "zdepth")
        if validation_error:
            logger.warning(validation_error)
            return result

        depth_batch = zdepth.shape[0] if zdepth.dim() == 4 else 1
        # No artificial frame cap on depth channel

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
                        dh, dw = depth_np.shape[:2]
                        dc = depth_np.shape[2] if depth_np.ndim == 3 else 1
                        # BUG-FIX (BUG-3): flags was hardcoded to 0 (fp16) even in 32-bit Float mode.
                        # Mirror the same fp16/fp32 branching used in _process_frame().
                        if use_32bit:
                            payload = depth_np.astype(np.float32).tobytes()
                            rhdr_flags = 1  # fp32 marker — viewer uses FLOAT texture
                        else:
                            payload = depth_np.astype(np.float16).tobytes()
                            rhdr_flags = 0  # fp16 marker — viewer uses HALF_FLOAT texture
                        compressed = zlib.compress(payload, level=1)  # level 1: float data barely compresses
                        header = struct.pack("<4sHHHH", b"RHDR", dw, dh, dc, rhdr_flags)
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
            # Round, not truncate: truncation darkened every code by half a step.
            return (np.clip(img_np.astype(np.float32), 0.0, 1.0) * BIT_8_MAX + 0.5).astype(
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
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Color"
    """
    Dedicated node to bake Radiance Viewer grading math into an image tensor.
    Useful for exporting or passing a graded image downstream in the pipeline.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Image to grade, treated as linear Rec.709/sRGB primaries. The result is "
                               "not clamped (values above 1.0 are kept); NaN and Inf are zeroed or capped."}),
                # Global
                "exposure": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01,
                    "tooltip": "Exposure in stops, baked into the output: RGB is multiplied by 2^exposure "
                               "before every other control."
                }),
                "offset": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001,
                    "tooltip": "Additive brightness lift applied after exposure. Shifts all tones uniformly."
                }),
                # Wheels
                "lift": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001,
                    "tooltip": "Shadow lift (black point raise). Raises the darkest values without affecting highlights."
                }),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 4.0, "step": 0.01,
                    "tooltip": "Mid-tone power curve, applied as x^(1/gamma) to positive values. "
                               "Values > 1.0 brighten, < 1.0 darken the midtones."
                }),
                "gain": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01,
                    "tooltip": "RGB multiplier (slope), applied after lift. 1.0 = no change."
                }),
                # Tone
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01,
                    "tooltip": "Linear contrast around pivot: (x - pivot) x contrast + pivot, per channel. "
                               "Not an S-curve, so values can go negative or above 1.0."
                }),
                "pivot": ("FLOAT", {"default": 0.18, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Value left unchanged by contrast. 0.18 = 18% grey in linear light."
                }),
                "shadows": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Shadow brightness, no tint: RGB is scaled by 1 + 0.5 x shadows x (1 - luma)^2. "
                               "Positive brightens dark areas, negative darkens them."
                }),
                "highlights": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Highlight brightness, no tint: RGB is scaled by 1 + 0.5 x highlights x luma^2 "
                               "(luma clipped to 1). Positive brightens bright areas, negative darkens them."
                }),
                # Color
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01,
                    "tooltip": "Global colour saturation. 1.0 = original, 0.0 = monochrome, > 1.0 = boosted."
                }),
                "temperature": ("FLOAT", {"default": 6500.0, "min": 2000.0, "max": 12000.0, "step": 10.0,
                    "tooltip": "White balance in absolute Kelvin, 6500 = no change. RGB is multiplied by the "
                               "blackbody colour of this temperature: lower values warm the image, higher cool it."
                }),
                "hue_shift": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.1,
                    "tooltip": "Global hue rotation in degrees. 0 = no change, 180 = complementary hues."
                }),
                # LUT
                "lut_name": (LUT_MODES, {"default": "None",
                    "tooltip": "Built-in transform applied last, after the grade: display/tone-map curves, "
                               "linear-to-camera-log encodes, log-to-linear IDTs, or analysis views "
                               "(False Color, Clip Check). None and the separator do nothing."}),
                "lut_intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Blend strength of the selected lut_name. 0.0 = bypass, 1.0 = full LUT application."
                }),
                # Color Science
                "color_science": (["Linear (sRGB)", "ACEScct"], {"default": "Linear (sRGB)",
                    "tooltip": "Space for offset, lift, gamma and gain. ACEScct converts linear sRGB to "
                               "ACEScct, applies them there and converts back; the other controls stay linear."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_grade"
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


# ── VFX Delivery Endpoint (Extracted) ─────────────────────────────────────────
# Extracted to radiance.delivery.handler for better modularity and reviewability
import radiance.delivery.handler as _delivery_handler



def _radiance_route_once(method, path):
    """Idempotent route decorator — prevents double-registration of this module's
    routes when it is imported under two names (avoids the ComfyUI startup crash
    'method HEAD is already registered').

    BUG-FIX: this used to swallow every exception and silently return an
    identity decorator. If PromptServer.instance was not available at import
    time the /radiance/progress route was simply dropped with no trace, the JS
    poll loop clearInterval'd on its first fetch error, and the export progress
    bar sat at 0% for the whole run with nothing anywhere saying why. The
    fallback still has to be non-fatal, importing this module must not require
    a running ComfyUI, but it now says what it did.
    """
    if PromptServer is None or getattr(PromptServer, "instance", None) is None:
        logger.warning(
            "[Radiance] PromptServer unavailable at import, route %s %s not registered. "
            "Endpoints served from this module (progress polling) will 404.",
            method.upper(), path,
        )
        return lambda fn: fn
    try:
        _reg = getattr(PromptServer.instance, "_radiance_registered_routes", None)
        if _reg is None:
            _reg = set()
            setattr(PromptServer.instance, "_radiance_registered_routes", _reg)
        _key = (method.lower(), path)
        if _key in _reg:
            return lambda fn: fn
        _reg.add(_key)
        return getattr(PromptServer.instance.routes, method)(path)
    except Exception as exc:
        logger.error(
            "[Radiance] Could not register route %s %s: %s. "
            "Progress polling will fail and the progress bar will stay at 0%%.",
            method.upper(), path, exc,
        )
        return lambda fn: fn


@_radiance_route_once('get', '/radiance/progress')
async def radiance_progress_endpoint(request):
    """Returns active delivery progress for a node instance."""
    instance_id = request.query.get('id')
    if not instance_id:
        # status=400: returning 200 made the JS compute
        # (prog.current / prog.total) * 100 on a body with neither key -> NaN,
        # rendering `width: NaN%` and a label of "undefined [NaN%]".
        return web.json_response({"error": "Missing ID", "status": "error",
                                  "current": 0, "total": 100}, status=400)
    
    progress = _progress_get(instance_id)
    return web.json_response(progress)




# ═══════════════════════════════════════════════════════════════════════════════
#   v3.0 VIEWER UPGRADES — Clip Check
# ═══════════════════════════════════════════════════════════════════════════════


# ── 1. Clip Check Overlay ─────────────────────────────────────────────────────

def _lut_clip_check(
    img: np.ndarray,
    white_ceiling: float = 1.0,
    black_floor: float = 0.0,
) -> np.ndarray:
    """Delivery clip check overlay.

    Blown highlights (> white_ceiling) → RED
    Crushed shadows  (< black_floor)   → BLUE
    In-range pixels                    → desaturated grey (luma)

    Args:
        img: float32 (H,W,C)
        white_ceiling: threshold above which pixel is flagged as blown
        black_floor:   threshold below which pixel is flagged as crushed

    Returns:
        float32 (H,W,3) overlay image
    """
    out = np.empty((*img.shape[:2], 3), dtype=np.float32)
    rgb = img[..., :3] if img.ndim == 3 and img.shape[2] >= 3 else np.stack([img[..., 0]] * 3, axis=-1)

    luma = (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2])
    grey = np.stack([luma, luma, luma], axis=-1)

    # Any channel blown?
    blown = np.any(rgb > white_ceiling, axis=-1)
    # Any channel crushed?
    crushed = np.any(rgb < black_floor, axis=-1)

    out[:] = grey * 0.35  # dim grey base
    out[blown]  = [1.0, 0.05, 0.05]   # bright red
    out[crushed] = [0.05, 0.15, 1.0]  # bright blue

    return np.clip(out, 0.0, 1.0)








# ═══════════════════════════════════════════════════════════════════════════════
#                           NODE REGISTRATION
# ═══════════════════════════════════════════════════════════════════════════════

NODE_CLASS_MAPPINGS = {
    "RadianceViewer":           RadianceViewer,
    "RadianceGradeApply":       RadianceGradeApply,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceViewer":           "◎ Radiance Viewer",
    "RadianceGradeApply":       "◎ Radiance Grade Apply",
}
