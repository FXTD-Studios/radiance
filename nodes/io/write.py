"""
nodes_io_unified.py — Radiance v3.1
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Two nodes replace all previous scattered I/O nodes:

  RadianceRead   — universal reader  (image · EXR · video · sequence)
  RadianceWrite  — universal writer  (image · EXR · video · sequence)

Auto-detects format from file extension / path pattern.

Supported READ
──────────────
  Image     .png .jpg .jpeg .tiff .tif .bmp .webp .dpx .hdr
  EXR       .exr  (single frame)
  Video     .mov .mxf .mp4 .m2ts .mkv .webm .avi … → batch of frames
            Decoded through radiance.core.video: one raw ffmpeg pipe, at the
            source's own bit depth (10/12/16-bit survive), with ProRes 4444
            alpha carried through to the MASK output, exact frame ranges, and
            the container's colour tags read rather than ignored.
  Sequence  any of the above with %04d / #### / * pattern, or a directory

Supported WRITE
───────────────
  Image        PNG (8-bit · 16-bit) · JPEG · TIFF (16 · 32f) · DPX · WEBP · Radiance HDR (.hdr)
  EXR          16-bit half · 32-bit float
  Video        MP4 H.264 · MP4 H.265 10-bit · MOV ProRes 422 / 4444 · MOV DNxHR
  Sequence     PNG / TIFF / EXR / DPX / Radiance HDR numbered sequences
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    from ...config.constants import VERSION as _RADIANCE_VERSION
except Exception:  # keep the writer importable even if constants move
    _RADIANCE_VERSION = "unknown"

try:
    import folder_paths as _folder_paths
    _HAS_FOLDER_PATHS = True
except ImportError:
    _HAS_FOLDER_PATHS = False

# Pillow, OpenImageIO, color_utils, core.formats and the video extension table
# all went down with the two engines. What is left in this module is the node
# surface, and it reaches none of them directly any more.

try:
    from ...hdr.io import write_exr_robust, write_exr_multipart
except ImportError:
    try:
        from hdr.io import write_exr_robust, write_exr_multipart  # type: ignore[import]
    except ImportError:
        write_exr_robust = None      # type: ignore[assignment]
        write_exr_multipart = None   # type: ignore[assignment]

try:
    from ...path_utils import get_safe_output_dir, strip_path_quotes
except ImportError:
    try:
        from path_utils import get_safe_output_dir, strip_path_quotes  # type: ignore[import]
    except ImportError:
        def get_safe_output_dir(base, override="", **_):  # type: ignore[misc]
            return override.strip() or base
        def strip_path_quotes(path):  # type: ignore[misc]
            return path.strip().strip('"').strip("'")

log = logging.getLogger("radiance.io_unified")


# ═══════════════════════════════════════════════════════════════════════════
# § 0  Constants
# ═══════════════════════════════════════════════════════════════════════════

# ── The write engine ───────────────────────────────────────────────────────
# Formats, colour-space application, every save backend and the writer body
# itself now live in radiance/io/writer.py, which has no node layer above it.
# They are imported straight back under their old private names so the rest of
# this module -- the reader, RadianceEXRMultiPart, RadianceDigitalCinemaWrite --
# is untouched by the move.
#
# The point of the move is delivery/handler.py, which used to instantiate this
# node to save a file. It calls write_frames() directly now.
from ...io.writer import (  # noqa: F401  (re-exported for this module's other users)
    _FMT_IMAGE,
    _FMT_VIDEO,
    _FMT_SEQ,
    _fmt_stem,
    WRITE_FORMATS,
    OUTPUT_COLOR_SPACES,
    WRITE_WORKING_SPACES,
    EXR_COMPRESSIONS,
    _ffmpeg_bin,
    _ffprobe_bin,
    _ffmpeg_ok,
    _apply_output_colorspace,
    _unique_path,
    _workflow_metadata,
    _save_pil_image,
    _exr_channels,
    _save_exr,
    _save_dpx,
    _save_hdr,
    _coerce_mask_to_alpha,
    _save_video_ffmpeg,
    _write_audio_temp_wav,
    coerce_to_frames as _coerce_to_frames_impl,
    resolve_output_path as _resolve_output_path,
    write_frames as _write_frames,
)


# ── The read engine ──────────────────────────────────────────────
# Input colour spaces, the extension tables, the decoders, path-kind detection,
# the image / sequence / video readers, the write-input coercion helpers and
# the UI probe now live in radiance/io/reader.py, which has no node layer above
# it. They are imported straight back under their old private names so the rest
# of this module -- RadianceRead's widget surface, RadianceEXRMultiPart,
# RadianceDigitalCinemaRead and the HTTP routes -- is untouched by the move.
#
# nodes/pipeline/dcc.py and nodes/pipeline/studio_integrations.py reach in here
# for _read_sequence and _load_video_to_numpy. Those imports keep working
# through this re-export; pointing them at radiance.io.reader directly is a
# caller change and not part of this one.
from ...io.reader import (  # noqa: F401  (re-exported for this module's other users)
    INPUT_COLOR_SPACES,
    READ_WORKING_SPACES,
    _INPUT_DECODERS,
    _IMG_EXT,
    _VID_EXT,
    _EXR_EXT,
    _np_to_tensor,
    _tensor_to_np,
    _rec709_to_linear,
    _apply_input_colorspace,
    _path_kind,
    _explain_unknown,
    _is_16bit_rgb_source,
    _is_float32_tiff,
    _read_image,
    _read_exr_single,
    _read_exr_with_info,
    _trailing_frame_number,
    _window_listed_files,
    _resolve_sequence_paths,
    _read_sequence,
    _read_video,
    _video_frame_range,
    _resolve_video_colorspace,
    _find_video_path,
    _load_video_to_numpy,
    _sequence_frame_range,
    _unpremultiply,
    _describe_read,
    _probe_for_ui,
    read_frames as _read_frames,
)



# ── File browser helpers ──────────────────────────────────────────────────

#: Extensions shown in the browse dropdown
_BROWSEABLE_EXT = _IMG_EXT | _EXR_EXT | _VID_EXT | {".dpx", ".hdr"}

def _get_input_files() -> List[str]:
    """
    Return a sorted list of image / EXR / video files found in ComfyUI's
    input directory.  Called fresh on each INPUT_TYPES() evaluation so new
    uploads appear without restarting the server.

    Falls back to [""] when folder_paths is not available (e.g. test harness).
    """
    if not _HAS_FOLDER_PATHS:
        return [""]
    try:
        input_dir = _folder_paths.get_input_directory()
        files = [
            f for f in sorted(os.listdir(input_dir))
            if os.path.isfile(os.path.join(input_dir, f))
            and Path(f).suffix.lower() in _BROWSEABLE_EXT
        ]
        return [""] + files
    except Exception:
        return [""]


def _resolve_browse(browse: str) -> Optional[str]:
    """
    Convert a browse filename (relative to ComfyUI's input dir) to an
    absolute path.  Returns None if the file cannot be found.
    """
    if not browse or not _HAS_FOLDER_PATHS:
        return None
    try:
        full = os.path.join(_folder_paths.get_input_directory(), browse)
        return full if os.path.isfile(full) else None
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
# § 3  RadianceRead
# ═══════════════════════════════════════════════════════════════════════════

class RadianceRead:
    """
    Universal reader — images, EXR, video, numbered sequences.

    File selection
    ──────────────
    Two ways to choose a file:
      1. browse  — Dropdown + upload button.  Click the upload icon to open a
                   native OS file picker, or drag-and-drop.  Uploaded files land
                   in ComfyUI's input/ folder and are selected automatically.
                   Supports images (PNG/JPG/TIFF/EXR/DPX/HDR) and video
                   (MP4/MOV/MXF/AVI/WebM/MKV).

      2. path    — Type or paste any absolute path, UNC network path, or a
                   sequence pattern (/renders/frame.%04d.exr, /frames/####.png).
                   This field is used only when browse is left blank.

    Path auto-detection
    ───────────────────
    The node inspects the extension and path pattern:
      .exr                → EXR (single frame)
      .png/.jpg/.tiff/…   → image (single frame)
      .mp4/.mov/…         → video (all frames decoded to batch)
      /path/%04d.exr      → sequence (printf pattern)
      /path/frame.####    → sequence (hash pattern)
      /path/              → sequence (directory, sorted)

    """

    CATEGORY = "FXTD STUDIOS/Radiance/◎ IO & Delivery"
    DESCRIPTION = (
        "Read an image, EXR, video or numbered sequence into the pipeline. "
        "Video decodes at the source bit depth through ffmpeg, keeps ProRes "
        "4444 alpha on the mask output, and honours the file's colour tags."
    )
    FUNCTION     = "read"
    # `info` is appended last on purpose: ComfyUI links outputs by index, so
    # every workflow saved against the two-output version keeps working.
    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "info")
    OUTPUT_TOOLTIPS = (
        "Frames as a batch. Scene-linear once color_space has decoded them.",
        "Alpha. For a ProRes 4444, an RGBA EXR or an RGBA sequence this is the "
        "file's own matte; otherwise zeros.",
        "JSON describing what was actually read: resolution, frame count and "
        "range, bit depth, codec, EXR layers and windows, colour tags, "
        "timecode. Nuke's metadata tab, as a wire you can plug in.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "browse": (_get_input_files(), {
                "image_upload": True,
                "tooltip": (
                    "Browse or upload a file from disk.\n"
                    "• Click the upload icon (📎) to open a native file picker.\n"
                    "• Supports images (PNG, JPG, TIFF, EXR, DPX, HDR, WebP) "
                    "and video (MP4, MOV, MXF, AVI, WebM, MKV).\n"
                    "• Uploaded files are copied to ComfyUI's input"
                    "• Leave blank and fill in 'path' below for absolute / "
                    "network / sequence paths."
                ),
            }),
            }, "optional": {
                "media_type": (["Auto", "Image", "Video", "Sequence"], {
                    "default": "Auto",
                    "tooltip": "Override auto-detection. Auto infers from path extension and pattern.",
                }),
                "path": ("STRING", {
                "default": "",
                "multiline": False,
                "placeholder": "/abs/path/file.exr  ·  //nas/share/shot.mov  ·  /seq/frame.%04d.exr",
                "tooltip": (
                    "Optional — used only when 'browse' is left blank.\n"
                    "Accepts any absolute path, UNC network path, or sequence pattern:\n"
                    "  Sequence patterns:  /frames/f.%04d.exr  ·  /frames/f.####.png  ·  /dir/\n"
                    "  Network paths:      /mnt/nas/renders/shot   or   \\\\server\\share\\shot\n"
                    "Format is auto-detected from extension."
                ),
            }),
            "color_space": (INPUT_COLOR_SPACES, {
                "default": "Auto / Linear (pass-through)",
                "tooltip": "Decode the input from this color space to scene-linear before processing.",
            }),
            "start_frame": ("INT", {
                "default": 1001, "min": 0, "max": 99999,
                "tooltip": (
                    "First frame to read.\n"
                    "• Sequence: the frame number in the filename (1001 is the "
                    "usual VFX start).\n"
                    "• Video: a 0-based offset into the clip. The 1001 default "
                    "is a sequence convention, so it is ignored for any clip "
                    "shorter than that rather than reading nothing."
                ),
            }),
            "end_frame": ("INT", {
                "default": 0, "min": 0, "max": 99999,
                "tooltip": "Last frame to read, inclusive. 0 = to the end. Applies to sequences and video.",
            }),
            "frame_step": ("INT", {
                "default": 1, "min": 1, "max": 100,
                "tooltip": "Step size — e.g. 2 reads every other frame. Applies to sequences and video.",
            }),
            "max_video_frames": ("INT", {
                "default": 0, "min": 0, "max": 99999,
                "tooltip": (
                    "Hard cap on decoded video frames (0 = all). Frames are "
                    "float32 RGB in RAM: 240 frames of 4K RGBA is about 31 GB, "
                    "so cap this while building a graph."
                ),
            }),
            "proxy_scale": ("FLOAT", {
                "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                "tooltip": "Downscale factor for proxy preview (0 = full resolution). 0.5 = half res for faster iteration.",
            }),
            "missing_frames": (["Error", "Black", "Skip"], {
                "default": "Skip",
                "tooltip": (
                    "A frame inside a sequence that is not on disk. Black "
                    "inserts a zero frame, Skip omits it, Error raises.\n"
                    "For a read that fails outright, see on_error."
                ),
            }),
            # ── Nuke-parity controls ────────────────────────────────────
            # A STRING rather than a combo on purpose. The layer list depends
            # on the file, and ComfyUI validates a combo's value against the
            # list it was built with -- so a workflow saved with layer="diffuse"
            # would fail to load on a machine where INPUT_TYPES had not seen
            # that file. js/radiance_io.js turns this into a dropdown populated
            # from /radiance/exr_layers, and a typed name still works with no
            # frontend at all.
            "layer": ("STRING", {
                "default": "auto",
                "multiline": False,
                "placeholder": "auto · rgba · diffuse · specular · Z · N",
                "tooltip": (
                    "Which EXR layer to read. 'auto' takes the beauty, or the "
                    "first colour layer, or — for a data-only file such as a "
                    "Z-depth pass — the first layer of any kind.\n"
                    "With the frontend loaded this is a dropdown listing the "
                    "layers actually in the selected file."
                ),
            }),
            "on_error": (["Error", "Black frame"], {
                "default": "Error",
                "tooltip": (
                    "What to do when the read fails — file missing, corrupt, "
                    "unreadable.\n"
                    "• Error: the node goes red and the queue stops. What Nuke "
                    "and every other application does, and the default.\n"
                    "• Black frame: return black and carry on. This is what "
                    "3.1.x always did, silently, which is how a black master "
                    "got delivered."
                ),
            }),
            "raw": ("BOOLEAN", {
                "default": False,
                "label_on": "raw (no transform)",
                "label_off": "managed",
                "tooltip": (
                    "Hand back exactly what is stored in the file: no colour "
                    "space decode, and no conform to the EXR display window "
                    "(so overscan is preserved). Nuke's 'raw data'."
                ),
            }),
            "premultiplied": ("BOOLEAN", {
                "default": False,
                "label_on": "premultiplied (unpremult on read)",
                "label_off": "straight alpha",
                "tooltip": (
                    "Tick when the file's RGB is already multiplied by its "
                    "alpha — the EXR convention — and you want it divided back "
                    "out on read. Off by default, matching Nuke, because "
                    "turning it on changes pixels."
                ),
            }),
            # `reload` used to live in "hidden". ComfyUI only populates a hidden
            # key when its VALUE is one of the magic strings (PROMPT, UNIQUE_ID,
            # EXTRA_PNGINFO, ...) -- a widget spec there is simply dropped, so
            # it never reached read() or IS_CHANGED. The frontend also builds
            # widgets from required/optional only, so js/radiance_io.js could
            # not find a widget named "reload" and bailed before adding the
            # button. The RELOAD button has therefore never rendered, and a user
            # whose file changed on disk without an mtime or size change (a
            # network share, an atomic replace) had no way to force a re-read.
            "reload": ("INT", {
                "default": 0,
                "min": 0,
                "max": 2 ** 31 - 1,
                "tooltip": "Bump to force a re-read of the file, for when the "
                           "contents changed but the timestamp did not.",
            }),
            # ── Colour management (3.5) ─────────────────────────────────
            "working_space": (READ_WORKING_SPACES, {
                "default": "Linear Rec.709 (sRGB)",
                "tooltip": (
                    "Scene-linear space the IMAGE is delivered in. color_space "
                    "is decoded all the way here: transfer AND primaries. "
                    "Linear Rec.709 matches ComfyUI; ACEScg for an ACES pipeline."
                ),
            }),
            "ocio_colorspace": ("STRING", {
                "default": "",
                "placeholder": "e.g. ARRI LogC4 · S-Log3 S-Gamut3.Cine · lin_ap0",
                "tooltip": (
                    "Any colorspace name or alias from the OCIO config. When set "
                    "it overrides color_space and OpenColorIO converts it to the "
                    "working space. Empty = use color_space."
                ),
            }),
            "ocio_config": ("STRING", {
                "default": "",
                "placeholder": "empty = $OCIO, else OCIO's ACES studio config",
                "tooltip": (
                    "Path to a config.ocio, or an ocio:// built-in URI. Empty uses "
                    "$OCIO, and without it OpenColorIO's built-in ACES 2.0 studio "
                    "config (every major camera log). Used by ocio_colorspace and, "
                    "when OCIO is installed, by the named color_space entries."
                ),
            }),
            "hdr_reference_nits": ("FLOAT", {
                "default": 203.0, "min": 48.0, "max": 1000.0, "step": 1.0,
                "tooltip": (
                    "PQ and HLG only: the luminance that scene-linear 1.0 "
                    "(diffuse white) sits at. 203 is BT.2408 and what "
                    "Radiance's HDR nodes use; 100 matches OCIO/ACES conventions."
                ),
            }),
        }}

    @classmethod
    def IS_CHANGED(cls, browse: str = "", media_type: str = "Auto", path: str = "", reload: int = 0, **_kw):
        """
        Tell ComfyUI to re-execute the node when the selected file changes.
        Returns a hash of the resolved path so caching is file-content aware.
        """
        resolved = _resolve_browse(browse) or path.strip()
        if resolved and os.path.isfile(resolved):
            try:
                stat = os.stat(resolved)
                return f"{resolved}:{stat.st_mtime}:{stat.st_size}:reload{reload}"
            except Exception as _exc:
                log.debug(
                    "[Radiance] IS_CHANGED(): ignoring %s from `stat = os.stat(resolved)`: %s",
                    type(_exc).__name__, _exc,
                )
        return float("nan")

    def read(
        self,
        browse: str = "",
        media_type: str = "Auto",
        path:   str = "",
        color_space: str = "Auto / Linear (pass-through)",
        start_frame: int = 1001,
        end_frame:   int = 0,
        frame_step:  int = 1,
        max_video_frames: int = 0,
        proxy_scale: float = 0.0,
        missing_frames: str = "Skip",
        layer: str = "auto",
        on_error: str = "Error",
        raw: bool = False,
        premultiplied: bool = False,
        reload: int = 0,
        working_space: str = "Linear Rec.709 (sRGB)",
        ocio_colorspace: str = "",
        ocio_config: str = "",
        hdr_reference_nits: float = 203.0,
    ):
        # The read itself is radiance/io/reader.py. Two things stay here
        # because the engine cannot have them: `browse` names a file in
        # ComfyUI's input directory and resolving it needs folder_paths, and
        # `reload` exists only to change the IS_CHANGED hash -- the read path
        # has never looked at it, so it is not passed down.
        image, mask, info = _read_frames(
            media_type=media_type,
            path=_resolve_browse(browse) or path,
            color_space=color_space,
            start_frame=start_frame,
            end_frame=end_frame,
            frame_step=frame_step,
            max_video_frames=max_video_frames,
            proxy_scale=proxy_scale,
            missing_frames=missing_frames,
            layer=layer,
            on_error=on_error,
            raw=raw,
            premultiplied=premultiplied,
            working_space=working_space,
            ocio_colorspace=strip_path_quotes(ocio_colorspace or "") if ocio_colorspace else "",
            ocio_config=strip_path_quotes(ocio_config or "") if ocio_config else "",
            hdr_reference_nits=hdr_reference_nits,
        )
        return (image, mask, json.dumps(info, default=str))


# ═══════════════════════════════════════════════════════════════════════════
# § 4  RadianceWrite
# ═══════════════════════════════════════════════════════════════════════════


_PREVIEWABLE = {".png", ".jpg", ".jpeg", ".webp"}


def _history_entries(saved, t0: float) -> list:
    """Files this write produced, as ComfyUI ``{filename, subfolder, type}``.

    ``saved`` is the file for a single image or video, or the sequence
    directory for a sequence; a sequence is listed from files in that
    directory modified since the write started. Files outside ComfyUI's
    output directory are reported with ``type: "absolute"`` and their full
    path, since ``/view`` cannot serve them.
    """
    if not saved:
        return []
    p = Path(str(saved))
    if p.is_dir():
        files = sorted(f for f in p.iterdir() if f.is_file() and f.stat().st_mtime >= t0 - 1.0)
    elif p.is_file():
        files = [p]
    else:
        return []
    out_root = None
    if _HAS_FOLDER_PATHS:
        try:
            out_root = Path(_folder_paths.get_output_directory()).resolve()
        except Exception:  # noqa: BLE001
            out_root = None
    entries = []
    for f in files:
        rf = f.resolve()
        if out_root is not None and (rf == out_root or out_root in rf.parents):
            rel = rf.relative_to(out_root)
            sub = "" if str(rel.parent) == "." else rel.parent.as_posix()
            entries.append({"filename": rf.name, "subfolder": sub, "type": "output"})
        else:
            entries.append({"filename": rf.name, "subfolder": str(rf.parent), "type": "absolute"})
    return entries


def _with_history(res, t0: float):
    """Wrap a write result with a ComfyUI ``ui`` block listing its files.

    ``images`` (which the frontend previews through ``/view``) only gets
    PNG / JPEG / WEBP files inside the output directory; EXR, DPX, TIFF and
    video would show as broken images. Every file is in ``radiance_files``.
    """
    saved = res[0] if isinstance(res, tuple) and res else None
    entries = _history_entries(saved, t0)
    if not entries:
        return res
    ui = {"radiance_files": entries}
    previews = [e for e in entries if e["type"] == "output"
                and Path(e["filename"]).suffix.lower() in _PREVIEWABLE]
    if previews:
        ui["images"] = previews
    return {"ui": ui, "result": res}

class RadianceWrite:
    """
    Universal writer — images, EXR, video, numbered sequences.

    Input
    ─────
    Accepts any ComfyUI output: IMAGE tensor, batched video frames, VHS / VideoHelperSuite
    VIDEO dict, video file path string, or any dict carrying a video path.

    Output path
    ───────────
    Any path the host OS can write to.  Network shares work natively — use the
    path as it appears on the machine running ComfyUI:

      Linux / Mac NFS / SMB   →  /mnt/nas/renders/shot_001
      Windows UNC              →  \\\\server\\share\\renders\\shot_001
      Windows mapped drive     →  Z:\\renders\\shot_001
      S3 / object storage      →  mount with s3fs / rclone first, then use mount path

    The output directory is created automatically (mkdir -p).  Write permission
    on the target location is required; an error is raised if the write fails.

    Format is selected from a flat dropdown.  All format-specific settings
    are optional inputs that are ignored when not relevant:

    ┌──────────────┬─────────────────────────────────────────────────────────┐
    │ Format group │ Key settings used                                        │
    ├──────────────┼─────────────────────────────────────────────────────────┤
    │ Image        │ quality, overwrite                                       │
    │ EXR          │ exr_compression, overwrite                               │
    │ Video        │ fps, quality (CRF), audio_source, broadcast_safe         │
    │ Sequence     │ fps, start_frame, frame_padding, overwrite               │
    └──────────────┴─────────────────────────────────────────────────────────┘

    """

    CATEGORY = "FXTD STUDIOS/Radiance/◎ IO & Delivery"
    DESCRIPTION = "Write images or EXR sequences to disk with configurable format options."
    FUNCTION     = "write"
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    # ALBABIT-FIX: without this, ComfyUI never schedules this node -- it has
    # no outputs for anything else to depend on, and OUTPUT_NODE is the only
    # other way the executor knows to run it. Lost when nodes_io.py was
    # recovered from the working-tree truncation (RadianceDigitalCinemaWrite's
    # docstring nearby references the same incident); the old Radiance
    # registered "◎ Radiance Write" under that shim class instead, which does
    # have this flag, masking the gap here.
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("*", {
                "tooltip": (
                    "Accepts any ComfyUI output: IMAGE tensor, batched video frames, "
                    "VHS"
                    "or any dict/list carrying a video path."
                ),
            }),
            "output_path": ("STRING", {
                "default": str(Path.home() / "radiance_output"),
                "multiline": False,
                "placeholder": "/output/render  or  //nas/share/render  or  Z:/renders/shot",
                "tooltip": (
                    "Output directory + filename stem.  Extension is appended automatically based on format.\n"
                    "For sequences: output_path becomes a folder and frames are named <folder>_<frame>.<ext> "
                    "(e.g. /out/shot -> /out/shot/shot_1001.exr).\n\n"
                    "Network paths are fully supported — use the path as mounted on this machine:\n"
                    "  Linux / Mac  →  /mnt/nas/renders/shot_001\n"
                    "  Windows UNC  →  \\\\server\\share\\renders\\shot_001\n"
                    "  Windows drive→  Z:\\renders\\shot_001\n"
                    "The directory is created automatically (mkdir -p) if it does not exist.\n"
                    "Write permissions on the share are required."
                ),
            }),
            "format": (WRITE_FORMATS, {
                "default": "IMG │ EXR (16-bit half)",
                "tooltip": "Output format.  Extension is appended automatically.",
            }),
        }, "optional": {
            "filename": ("STRING", {
                "default": "",
                "placeholder": "shot_001",
                "tooltip": "Output filename stem (version appended automatically). Leave empty to use output_path as the full stem.",
            }),
            "version": ("INT", {
                "default": 1,
                "min": 0,
                "max": 9999,
                "tooltip": "Version number appended to filename (e.g. shot_001_v0001).",
            }),
            "color_space": (OUTPUT_COLOR_SPACES, {
                "default": "Linear (pass-through)",
                "tooltip": ("File encoding: transfer AND primaries, from working_space. "
                            "Linear (pass-through) writes the working values unchanged. "
                            "EXR and DPX record it in their headers (chromaticities, "
                            "oiio:ColorSpace); video gets primaries/transfer/matrix tags."),
            }),
            "fps": ("FLOAT", {
                "default": 0.0, "min": 0.0, "max": 240.0, "step": 0.001,
                "tooltip": "Frame rate for video and sequence outputs. "
                           "0 = auto-detect from the source video (falls back to 24 if unavailable).",
            }),
            "quality": ("INT", {
                "default": 18, "min": 0, "max": 51,
                "tooltip": "CRF quality for H.264/H.265 (lower = better).  Also JPEG quality 0–100 (remapped).",
            }),
            "exr_compression": (EXR_COMPRESSIONS, {
                "default": "ZIP",
                "tooltip": "EXR compression codec (EXR formats only).",
            }),
            "start_frame": ("INT", {
                "default": 1001, "min": 0, "max": 99999,
                "tooltip": "First frame number for sequences.",
            }),
            "frame_padding": ("INT", {
                "default": 4, "min": 1, "max": 8,
                "tooltip": "Zero-padding width for frame numbers (e.g. 4 → 0001).",
            }),
            "audio_source": ("STRING", {
                "default": "",
                "tooltip": "Path to audio file to mux into video output (optional). "
                           "Takes priority over the 'audio' input when both are set.",
            }),
            "broadcast_safe": ("BOOLEAN", {
                "default": False,
                "tooltip": "8/16-bit stills: clamp to legal range (16–235). Video is always "
                           "encoded legal-range by the RGB→YUV step; float formats are never clamped.",
            }),
            "overwrite": ("BOOLEAN", {
                # AUDIT-UX (2026-08): default was True. Destroying an existing
                # file must be an explicit choice; with versioning built in and
                # a unique-suffix fallback, the safe default costs nothing.
                "default": False,
                "tooltip": "Overwrite existing files.  When disabled (default), a unique suffix is appended instead of destroying the existing file.",
            }),
            "proxy_scale": ("FLOAT", {
                "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                "tooltip": "Downscale output by this factor for proxy preview (0 = full resolution).",
            }),
            "audio": ("AUDIO", {
                "tooltip": "Audio tensor from RadianceVideoLoader (muxed into video output).",
            }),
            "mask": ("MASK", {
                "tooltip": (
                    "Optional alpha/matte, written as the alpha channel of EXR, PNG, "
                    "TIFF, DPX, WEBP and ProRes 4444. Ignored by JPEG, HDR, H.264, "
                    "H.265, ProRes 422 and DNxHR, which have no alpha."
                ),
            }),
            # ── Colour management (3.5) ─────────────────────────────────
            "working_space": (WRITE_WORKING_SPACES, {
                "default": "Linear Rec.709 (sRGB)",
                "tooltip": (
                    "Scene-linear space the incoming IMAGE is in. color_space is "
                    "encoded from here, primaries AND transfer. Match the Read "
                    "node's working_space."
                ),
            }),
            "ocio_colorspace": ("STRING", {
                "default": "",
                "placeholder": "e.g. ACEScct · S-Log3 S-Gamut3.Cine · Rec.1886 Rec.709 - Display",
                "tooltip": (
                    "Any colorspace name or alias from the OCIO config. When set it "
                    "overrides color_space and OpenColorIO encodes the working space "
                    "into it. Empty = use color_space."
                ),
            }),
            "ocio_config": ("STRING", {
                "default": "",
                "placeholder": "empty = $OCIO, else OCIO's ACES studio config",
                "tooltip": "Path to a config.ocio or an ocio:// URI. Empty = $OCIO, else OCIO's built-in ACES 2.0 studio config.",
            }),
            "hdr_reference_nits": ("FLOAT", {
                "default": 203.0, "min": 48.0, "max": 1000.0, "step": 1.0,
                "tooltip": (
                    "PQ and HLG only: the luminance scene-linear 1.0 (diffuse white) "
                    "is written at. 203 is BT.2408 and Radiance's HDR convention; "
                    "100 matches OCIO/ACES. PQ clips at 10,000 nits, HLG at about "
                    "3.8x diffuse white."
                ),
            }),
        }, "hidden": {
            "prompt":        "PROMPT",
            "extra_pnginfo": "EXTRA_PNGINFO",
        }}

    # ── helpers ───────────────────────────────────────────────────────────

    def _read_media(self, source: Any) -> Optional[np.ndarray]:
        """The reader half of input coercion, supplied to the write engine.

        `write_frames` accepts a file path or a VideoHelperSuite-style dict as
        well as a tensor, and resolving either means decoding media. The
        decoders live in this module, a floor above the engine, so the engine
        takes this as a callback rather than importing upward. Returns None
        when there is nothing here to read, which is what makes the engine's
        error message the same one it always was.
        """
        if isinstance(source, dict):
            vpath = _find_video_path(source)
            return _load_video_to_numpy(vpath) if vpath else None
        if isinstance(source, str):
            ext = Path(source).suffix.lower()
            if os.path.isfile(source):
                if ext in _VID_EXT:
                    return _load_video_to_numpy(source)
                if ext in (_IMG_EXT | _EXR_EXT):
                    img_t, _ = _read_image(source)
                    arr = img_t.detach().cpu().float().numpy()
                    return arr[np.newaxis] if arr.ndim == 3 else arr
        return None

    def _coerce_to_frames(self, data: Any) -> Tuple[np.ndarray, Optional[float]]:
        """Kept as a method because saved graphs and tests reach for it."""
        return _coerce_to_frames_impl(data, read_media=self._read_media)

    def _out_path(self, base: str, ext: str, overwrite: bool) -> Path:
        """Kept as a method for the same reason as _coerce_to_frames."""
        return _resolve_output_path(base, ext, overwrite)

    # ── main ──────────────────────────────────────────────────────────────

    def write(
        self,
        image:          torch.Tensor,
        output_path:    str,
        format:         str,
        filename:       str   = "",
        version:        int   = 1,
        color_space:    str  = "Linear (pass-through)",
        fps:            float = 0.0,
        quality:        int   = 18,
        exr_compression: str  = "ZIP",
        start_frame:    int   = 1001,
        frame_padding:  int   = 4,
        audio_source:   str   = "",
        broadcast_safe: bool  = False,
        # Matches INPUT_TYPES, which declares False. It used to default to
        # True here, so any caller that did not pass the widget value (the
        # Digital Cinema shim, the delivery path) got destructive
        # overwrite from a signature that contradicted its own UI.
        overwrite:      bool  = False,
        proxy_scale:    float = 0.0,
        audio:          Any   = None,
        mask:           Any   = None,
        prompt:         Any   = None,
        extra_pnginfo:  Any   = None,
        working_space:  str   = "Linear Rec.709 (sRGB)",
        ocio_colorspace: str  = "",
        ocio_config:    str   = "",
        hdr_reference_nits: float = 203.0,
    ):
        """The node surface. The maths and the file writing are in
        radiance/io/writer.py; this signature is the widget contract and must
        not drift from it, because saved workflows are matched against it by
        name and order.

        Returns ``{"ui": ..., "result": (saved, count)}`` so ComfyUI records
        the written files in ``/history`` (API clients had no way to find
        them). ``result`` keeps the old tuple for programmatic callers.
        """
        t0 = time.time()
        res = _write_frames(
            image=image,
            output_path=output_path,
            format=format,
            filename=filename,
            version=version,
            color_space=color_space,
            fps=fps,
            quality=quality,
            exr_compression=exr_compression,
            start_frame=start_frame,
            frame_padding=frame_padding,
            audio_source=audio_source,
            broadcast_safe=broadcast_safe,
            overwrite=overwrite,
            proxy_scale=proxy_scale,
            audio=audio,
            mask=mask,
            prompt=prompt,
            extra_pnginfo=extra_pnginfo,
            read_media=self._read_media,
            working_space=working_space,
            ocio_colorspace=ocio_colorspace,
            ocio_config=strip_path_quotes(ocio_config or "") if ocio_config else "",
            hdr_reference_nits=hdr_reference_nits,
        )
        return _with_history(res, t0)

# ═══════════════════════════════════════════════════════════════════════════
# § 5  RadianceEXRMultiPart — multi-layer AOV EXR writer
#      (migrated from nodes_io.py, Task #141-fix)
# ═══════════════════════════════════════════════════════════════════════════

_EXR_BIT_DEPTHS   = ["16-bit Half Float", "32-bit Float"]
_EXR_COMPRESSIONS = [
    "ZIP", "ZIPS", "PIZ", "RLE", "Uncompressed",
    "PXR24", "B44", "B44A", "DWAA", "DWAB",
]

def _norm_exr_compression(comp: str) -> str:
    """Normalise UI compression label → write_exr_* keyword."""
    return "NO_COMPRESSION" if comp.lower() == "uncompressed" else comp


def _copy_to_remote_path(local_path: str, remote_path: str) -> bool:
    """Best-effort copy to a remote/UNC path using shutil."""
    try:
        dest = Path(remote_path.strip())
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, dest / Path(local_path).name)
        return True
    except Exception as exc:
        log.warning("[EXRMultiPart] remote copy failed: %s", exc)
        return False


class RadianceEXRMultiPart:
    """Write a named multi-part EXR v2 combining up to 6 AOV layers into one
    file readable by Nuke, DaVinci Resolve (Flatten Layers), and Fusion.

    Standard parts:
      • beauty      → R, G, B (+ A if alpha connected)
      • depth       → Z  (single channel)
      • normal      → NX, NY, NZ
      • albedo      → albedo.R/G/B
      • custom_1/2  → <name>.R/G/B

    Fallback: if OpenEXR v2 multi-part is unavailable the node writes
    separate per-part .exr files instead.
    """

    CATEGORY = "FXTD STUDIOS/Radiance/◎ IO & Delivery"
    DESCRIPTION = ("Write one multi-part OpenEXR frame with named AOV parts (beauty, depth, normal, albedo, "
                   "two custom). Values are written unchanged; if multi-part output fails, one EXR per part is written.")
    FUNCTION    = "write_multipart"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filename_prefix": ("STRING", {"default": "radiance_multipart",
                    "tooltip": "File name stem. The file is written as <prefix>.<frame_index, 4-digit padded>.exr."}),
                "beauty":    ("IMAGE", {
                    "tooltip": "Main image, written as the 'beauty' part (R, G, B, plus A if it has 4 channels). Values are written unchanged, so feed scene-linear data. Each frame of a batch is written to its own file."}),
                "bit_depth": (_EXR_BIT_DEPTHS,   {"default": "16-bit Half Float",
                    "tooltip": "Pixel type for every part: 16-bit half float (smaller, about 11 bits of precision) or 32-bit float."}),
                "compression": (_EXR_COMPRESSIONS, {"default": "ZIP",
                    "tooltip": "EXR compression for every part. ZIP, ZIPS, PIZ, RLE and Uncompressed are lossless; PXR24 is lossy on 32-bit float; B44, B44A, DWAA and DWAB are lossy."}),
            },
            "optional": {
                "depth":         ("IMAGE", {
                    "tooltip": "Depth AOV written as a single Z channel in a 'depth' part, from the first channel (a grey depth image carries the same value in all three); values are not normalised."}),
                "normal":        ("IMAGE", {
                    "tooltip": "Normals AOV written as NX, NY, NZ in a 'normal' part, values unchanged (no 0-1 to -1..1 remap)."}),
                "albedo":        ("IMAGE", {
                    "tooltip": "Albedo AOV written as albedo.R/G/B in an 'albedo' part, values unchanged."}),
                "custom_1":      ("IMAGE", {
                    "tooltip": "Extra AOV written as <custom_1_name>.R/G/B in a part of that name."}),
                "custom_1_name": ("STRING", {"default": "emission",
                    "tooltip": "Part and channel-prefix name for custom_1. Letters, digits, underscore, hyphen or dot only; anything else stops the node."}),
                "custom_2":      ("IMAGE", {
                    "tooltip": "Extra AOV written as <custom_2_name>.R/G/B in a part of that name."}),
                "custom_2_name": ("STRING", {"default": "specular",
                    "tooltip": "Part and channel-prefix name for custom_2. Letters, digits, underscore, hyphen or dot only."}),
                "output_path":   ("STRING", {"default": "",
                    "tooltip": "Output folder. Empty writes to the ComfyUI output folder; a relative path is inside it; an absolute path is used as-is. Created if missing."}),
                "remote_path":   ("STRING", {"default": "",
                    "tooltip": "Optional second folder (for example a NAS or UNC share) the finished file is copied to. A failed copy only logs a warning."}),
                "frame_index":   ("INT",    {"default": 1, "min": 1,
                    "tooltip": "Frame number of the first image in the batch, used in the file name; later frames count up from it. An AOV with one frame is used for every frame."}),
                "custom_metadata": ("STRING", {"default": "", "multiline": True,
                    "tooltip": "Extra header attributes, one key=value per line. Keys are stored with a 'rad_' prefix unless they are standard EXR names (owner, comments, capDate and so on)."}),
            },
        }

    def write_multipart(
        self,
        filename_prefix: str,
        beauty: torch.Tensor,
        bit_depth: str = "16-bit Half Float",
        compression: str = "ZIP",
        depth: Optional[torch.Tensor] = None,
        normal: Optional[torch.Tensor] = None,
        albedo: Optional[torch.Tensor] = None,
        custom_1: Optional[torch.Tensor] = None,
        custom_1_name: str = "emission",
        custom_2: Optional[torch.Tensor] = None,
        custom_2_name: str = "specular",
        output_path: str = "",
        remote_path: str = "",
        frame_index: int = 1,
        custom_metadata: str = "",
    ) -> Tuple[str]:
        import datetime, re as _re

        output_path = strip_path_quotes(output_path)
        remote_path = strip_path_quotes(remote_path)

        _fp = _folder_paths if _HAS_FOLDER_PATHS else None
        base_dir = _fp.get_output_directory() if _fp else tempfile.gettempdir()
        out_dir = get_safe_output_dir(base_dir, output_path, allow_absolute=True)
        os.makedirs(out_dir, exist_ok=True)

        comp      = _norm_exr_compression(compression)

        # Validate custom part names — spaces / special chars corrupt multi-part EXR
        _SAFE = _re.compile(r'^[A-Za-z0-9_\-\.]+$')
        for name in (custom_1_name, custom_2_name):
            if name and not _SAFE.match(name):
                raise ValueError(
                    f"[EXRMultiPart] Invalid part name '{name}'. "
                    "Use only letters, digits, underscore, hyphen, or dot."
                )

        meta: Dict[str, Any] = {
            "software": f"Radiance v{_RADIANCE_VERSION}",
            "rad_version": _RADIANCE_VERSION,
            "created":  datetime.datetime.now().isoformat(),
        }
        for line in custom_metadata.strip().split("\n"):
            if "=" in line:
                k, v = line.split("=", 1)
                meta[k.strip()] = v.strip()

        # 3.5.0: every frame of the batch is written, <prefix>.<frame_index + n>.exr.
        # Only the first frame used to be written and the rest dropped. An AOV
        # with one frame (or a 3-D tensor) is used for every frame.
        batch = int(beauty.shape[0]) if beauty.dim() == 4 else 1
        named = [(beauty, "beauty")]
        for tensor, name in (
            (depth,    "depth"),
            (normal,   "normal"),
            (albedo,   "albedo"),
            (custom_1, custom_1_name or "custom_1"),
            (custom_2, custom_2_name or "custom_2"),
        ):
            if tensor is None:
                continue
            if tensor.dim() == 4 and tensor.shape[0] not in (1, batch):
                raise ValueError(f"[EXRMultiPart] '{name}' has {tensor.shape[0]} frames; beauty has {batch}. "
                                 "Use the same count, or one frame to repeat.")
            named.append((tensor, name))

        def _t(t: torch.Tensor, b: int) -> np.ndarray:
            arr = t if t.dim() == 3 else t[0 if t.shape[0] == 1 else b]
            return arr.detach().float().cpu().numpy()

        def write_frame(b: int) -> str:
            frame_num = str(frame_index + b).zfill(4)
            filepath  = os.path.join(out_dir, f"{filename_prefix}.{frame_num}.exr")
            parts: Dict[str, np.ndarray] = {name: _t(t, b) for t, name in named}

            if write_exr_multipart is not None:
                ok = write_exr_multipart(filepath, parts, bit_depth, comp, meta)
            elif write_exr_robust is not None:
                log.warning("[EXRMultiPart] multi-part unavailable — writing beauty-only EXR")
                ok = write_exr_robust(filepath, parts["beauty"], bit_depth, comp, meta)
            else:
                log.error("[EXRMultiPart] No EXR writer available")
                ok = False

            if not ok:
                raise RuntimeError(f"[EXRMultiPart] Failed to write: {filepath}")

            log.info("[EXRMultiPart] Wrote %d parts → %s", len(parts), filepath)

            if remote_path:
                _copy_to_remote_path(filepath, remote_path)
            return filepath

        workers = max(1, min(batch, os.cpu_count() or 1, 8))
        if workers == 1:
            paths = [write_frame(b) for b in range(batch)]
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="radiance-exr-mp") as pool:
                paths = list(pool.map(write_frame, range(batch)))

        return (paths[0],)


# ═══════════════════════════════════════════════════════════════════════════
# § 6  Registration
# ═══════════════════════════════════════════════════════════════════════════

# Shims for backwards compatibility
class RadianceDigitalCinemaRead:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_path": ("STRING", {"default": "",
                    "tooltip": "Video file, image, or frame sequence (e.g. shot.####.exr). Required: an empty path stops the graph."}),
                "read_mode": (["Auto", "Video", "Sequence", "EXR"], {"default": "Auto",
                    "tooltip": "How to treat source_path. Auto detects from the path; EXR is handled exactly like Sequence."}),
                "start_frame": ("INT", {"default": 1, "min": 1,
                    "tooltip": ("Sequence: the frame number to start at (a number outside the range on disk "
                                "falls back to the first frame). Video: the frame to start at counting from 1, "
                                "so the default 1 is the clip's first frame.")}),
                "frame_limit": ("INT", {"default": 0, "min": 0,
                    "tooltip": "Maximum number of frames to read from start_frame. 0 reads to the end."}),
                "input_colorspace": (["sRGB (Standard)"] + [c for c in INPUT_COLOR_SPACES if c != "sRGB"],
                                     {"default": "sRGB (Standard)",
                                      "tooltip": "Transfer the source was encoded with. It is decoded "
                                                 "to scene-linear; shot_metadata.colorspace records the result."}),
                "fps_override": ("FLOAT", {"default": 0.0, "min": 0.0,
                                           "tooltip": "0 = the source's own rate. Written to shot_metadata.fps."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "RADIANCE_SHOT")
    RETURN_NAMES = ("image", "mask", "shot_metadata")
    FUNCTION = "read"
    CATEGORY = "FXTD Studios/Radiance/IO"
    DESCRIPTION = ("Read a video, image or frame sequence through Radiance Read, decoded to scene-linear, "
                   "and output shot metadata (source info, colorspace, fps) for downstream pipeline nodes.")

    def read(self, source_path, read_mode, start_frame, frame_limit, input_colorspace, fps_override=0.0):
        # An executed read with no source is a mistake, not a placeholder. The
        # shared reader returns an 8x8 black frame for an empty path so a node
        # dropped on the canvas can preview; reached from a queued graph, that
        # frame went downstream as the shot, with RADIANCE_SHOT metadata
        # saying kind "empty" and nothing on screen to say why. Found by the
        # live 3.5 run.
        if not str(source_path or "").strip().strip('"').strip("'"):
            raise ValueError(
                "RadianceDigitalCinemaRead: source_path is empty. Set it to a "
                "video file, an image, or a frame sequence (e.g. shot.####.exr)."
            )
        media_type = "Auto"
        if read_mode == "Video":
            media_type = "Video"
        elif read_mode in ("Sequence", "EXR"):
            media_type = "Sequence"

        cs_in = "sRGB" if input_colorspace == "sRGB (Standard)" else input_colorspace
        reader = RadianceRead()
        # `end_frame` is a frame NUMBER, and `start_frame` here defaults to 1
        # while VFX sequences number from 1001. A `frame_limit` of 10 therefore
        # arrives at the reader as start=1, end=10; `_sequence_frame_range`
        # corrects the start to 1001, and end<start used to send
        # `_resolve_sequence_paths` down its "read everything" fallback --
        # measured: start_frame=1, frame_limit=10 on a 100-frame sequence
        # numbered 1001-1100 read all 100 frames into RAM. The reader now
        # carries the window's LENGTH across that correction (see
        # `_sequence_frame_range`), so the span expressed here is what is read.
        #
        # A video's frames are numbered from 0 in the reader. This node's
        # start_frame counts from 1 (its minimum and default), so it is turned
        # into the 0-based offset here; passed straight through, the default
        # skipped every clip's first frame. The span is carried by
        # max_video_frames, since the reader's end_frame 0 means "to the end".
        clean_path = str(source_path).strip().strip('"').strip("'")
        is_video = media_type == "Video" or (media_type == "Auto" and _path_kind(clean_path) == "video")
        if is_video:
            read_start, read_end = max(int(start_frame) - 1, 0), 0
        else:
            read_start = start_frame
            read_end = (start_frame + frame_limit - 1) if frame_limit > 0 else 0
        img, mask, read_info = reader.read(
            path=source_path,
            media_type=media_type,
            color_space=cs_in,
            start_frame=read_start,
            end_frame=read_end,
            max_video_frames=frame_limit if frame_limit > 0 else 0,
        )

        shot_metadata = {
            "source_path": source_path,
            "read_mode": read_mode,
            "start_frame": start_frame,
            "frame_limit": frame_limit,
            "input_colorspace": input_colorspace,
            "fps_override": fps_override,
        }
        # Fold in what the reader actually found -- resolution, real frame
        # count, codec, bit depth, colour tags, timecode -- rather than echoing
        # back only the widget values it was given.
        try:
            shot_metadata["source"] = json.loads(read_info)
        except (TypeError, ValueError) as _exc:
            log.debug("[Radiance] DigitalCinemaRead: unreadable info payload: %s", _exc)
        src = shot_metadata.get("source") or {}

        # What the pixels are now, the key RadianceLinearCheck reads. It was
        # never written, so Linear Check reported "Unknown" for every shot.
        if cs_in == "ACEScg":
            colorspace = "ACEScg"
        elif cs_in == "Auto / Linear (pass-through)":
            float_src = src.get("kind") == "exr" or str(source_path).lower().endswith((".exr", ".hdr"))
            colorspace = "Linear" if float_src else "Unknown (pass-through, not decoded)"
        else:
            colorspace = f"Linear (decoded from {cs_in})"
        shot_metadata["colorspace"] = colorspace

        # fps_override used to be stored and nothing else.
        src_fps = src.get("fps")
        shot_metadata["fps"] = float(fps_override) if fps_override > 0 else (
            float(src_fps) if src_fps else None)
        shot_metadata["fps_source"] = "override" if fps_override > 0 else (
            "source" if src_fps else "unknown")
        return (img, mask, shot_metadata)


class RadianceDigitalCinemaWrite:
    """Backward-compatible Digital Cinema writer shim.

    Delegates to RadianceWrite. Recovered after the working-tree truncation that
    removed this class; provides the OUTPUT_NODE write surface the pipeline and
    tests expect. Returns a STRING status.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Frames to write, passed to Radiance Write with its defaults (color_space Linear (pass-through), so values are written unchanged)."}),
                # Same default as RadianceWrite, which this node delegates to.
                # It used to default to "", so dropping the node and queueing
                # crashed inside pathlib instead of writing anything.
                "output_path": ("STRING", {
                    "default": str(Path.home() / "radiance_output"),
                    "placeholder": "/output/render  or  Z:/renders/shot",
                    "tooltip": ("Output path, as in Radiance Write. With filename empty it is the full stem "
                                "(single images and video get the extension appended; sequences use it as a folder)."),
                }),
            },
            "optional": {
                "format": (WRITE_FORMATS, {"default": "IMG │ EXR (16-bit half)",
                    "tooltip": "Output format. IMG writes only the first frame of a batch; SEQ and VID write every frame."}),
                "filename": ("STRING", {"default": "",
                    "tooltip": "Optional stem. When set, output_path is treated as a folder and the file is named <filename>_v0001."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "write"
    OUTPUT_NODE = True
    CATEGORY = "FXTD STUDIOS/Radiance/Pipeline"
    DESCRIPTION = ("Simplified writer kept for older pipelines: passes images, output_path, format and filename "
                   "to Radiance Write with every other setting at its default, and returns a status string.")

    def write(self, images, output_path, format="IMG │ EXR (16-bit half)", filename=""):
        writer = RadianceWrite()
        out = writer.write(image=images, output_path=output_path, format=format, filename=filename)
        status = (f"OK: wrote '{filename or output_path}' as {format}",)
        ui = out.get("ui") if isinstance(out, dict) else None
        return {"ui": ui, "result": status} if ui else status


NODE_CLASS_MAPPINGS = {
    "RadianceRead": RadianceRead,
    "RadianceWrite": RadianceWrite,
    "RadianceEXRMultiPart": RadianceEXRMultiPart,
    "RadianceDigitalCinemaRead": RadianceDigitalCinemaRead,
    "RadianceDigitalCinemaWrite": RadianceDigitalCinemaWrite,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceDigitalCinemaWrite": "◎ Radiance Digital Cinema Write",
    "RadianceRead": "◎ Radiance Read",
    "RadianceWrite": "◎ Radiance Write",
    "RadianceEXRMultiPart": "◎ Radiance EXR Multi-Part",
    "RadianceDigitalCinemaRead": "◎ Radiance Digital Cinema Read",
}


# ═══════════════════════════════════════════════════════════════════════════
# § 6  HTTP routes for the Read node's frontend
# ═══════════════════════════════════════════════════════════════════════════

#: Directories the /radiance/media/* routes may inspect.
#:
#: ComfyUI's own input/output/temp trees are always allowed, because that is
#: where uploads land. A facility reads plates off a NAS, so add those mounts
#: with RADIANCE_READ_ROOTS (os.pathsep-separated). Without it the layer
#: dropdown falls back to a plain text field for paths outside the tree, which
#: still works -- you type the layer name.
_ENV_READ_ROOTS = "RADIANCE_READ_ROOTS"


def _allowed_read_roots() -> List[str]:
    roots: List[str] = []
    try:
        import folder_paths  # type: ignore

        for attr in ("get_input_directory", "get_output_directory",
                     "get_temp_directory"):
            getter = getattr(folder_paths, attr, None)
            if callable(getter):
                roots.append(getter())
        models = getattr(folder_paths, "models_dir", None)
        if models:
            roots.append(models)
    except Exception as _exc:
        log.debug("[Radiance] _allowed_read_roots(): no folder_paths: %s", _exc)
    for part in os.environ.get(_ENV_READ_ROOTS, "").split(os.pathsep):
        part = part.strip().strip('"')
        if part:
            roots.append(part)
    return [os.path.abspath(os.path.expanduser(r)) for r in roots if r]


def _is_inside_allowed_read_root(path: str) -> bool:
    """Is this path somewhere the frontend is allowed to ask us to inspect?

    The Read node itself opens any path the user types -- that is the job. This
    check governs the *unauthenticated HTTP route* only, which would otherwise
    be a file-existence oracle for the whole filesystem.
    """
    try:
        resolved = os.path.realpath(path)
    except (OSError, ValueError):
        return False
    for root in _allowed_read_roots():
        try:
            if os.path.commonpath([resolved, os.path.realpath(root)]) == \
                    os.path.realpath(root):
                return True
        except ValueError:      # different drives on Windows
            continue
    return False


def register_read_routes():
    """Endpoints the Read node's widgets call: EXR layers, and media info."""
    try:
        from aiohttp import web
        from server import PromptServer
    except ImportError:
        log.debug("[Radiance/Read] server not available; routes not registered")
        return

    if getattr(PromptServer.instance, "_radiance_read_routes_registered", False):
        return
    PromptServer.instance._radiance_read_routes_registered = True

    def _resolve_query_path(request) -> Tuple[Optional[str], Optional[Any]]:
        raw = (request.query.get("path") or "").strip()
        if not raw:
            return None, web.json_response({"error": "no path given"}, status=400)
        candidate = _resolve_browse(raw) or strip_path_quotes(raw)
        if not os.path.isfile(candidate):
            return None, web.json_response(
                {"error": "not a file", "path": candidate}, status=404)
        if not _is_inside_allowed_read_root(candidate):
            return None, web.json_response({
                "error": "outside the allowed roots",
                "hint": f"Set {_ENV_READ_ROOTS} to the directories Radiance may "
                        f"inspect, os.pathsep-separated.",
            }, status=403)
        return candidate, None

    @PromptServer.instance.routes.get("/radiance/media/layers")
    async def read_layers_endpoint(request):
        """Layer names inside an EXR, for the `layer` dropdown."""
        path, error = _resolve_query_path(request)
        if error is not None:
            return error
        if os.path.splitext(path)[1].lower() not in _EXR_EXT:
            return web.json_response({"layers": [], "reason": "not an EXR"})
        try:
            from ...core import exr as _exr

            return web.json_response({"layers": ["auto"] + _exr.layer_choices(path)})
        except Exception as exc:
            log.debug("[Radiance/Read] layer probe failed for %s: %s", path, exc)
            return web.json_response({"layers": [], "error": str(exc)})

    @PromptServer.instance.routes.get("/radiance/media/info")
    async def read_info_endpoint(request):
        """What the file is, for the info line drawn on the node."""
        path, error = _resolve_query_path(request)
        if error is not None:
            return error
        try:
            return web.json_response(_probe_for_ui(path))
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=200)

    _MIME = {".mp4": "video/mp4", ".m4v": "video/mp4", ".mov": "video/quicktime",
             ".mkv": "video/x-matroska", ".webm": "video/webm", ".avi": "video/x-msvideo"}

    @PromptServer.instance.routes.get("/radiance/media/preview")
    async def read_preview_endpoint(request):
        """Stream a video file for the node's inline <video> preview.

        FileResponse handles HTTP Range requests, so the browser can seek.
        Whether the browser can DECODE it is its own problem -- H.264 MP4/MOV
        will play; ProRes/DNxHR will fire the <video> element's error event,
        and the frontend falls back to /radiance/media/poster.
        """
        path, error = _resolve_query_path(request)
        if error is not None:
            return error
        ext = os.path.splitext(path)[1].lower()
        if ext not in _VID_EXT:
            return web.json_response({"error": "not a video"}, status=400)
        return web.FileResponse(
            path, headers={"Content-Type": _MIME.get(ext, "application/octet-stream")})

    @PromptServer.instance.routes.get("/radiance/media/poster")
    async def read_poster_endpoint(request):
        """First frame of a video as JPEG -- the preview for codecs the
        browser cannot decode (ProRes, DNxHR, MXF...). Cached by mtime."""
        path, error = _resolve_query_path(request)
        if error is not None:
            return error
        if os.path.splitext(path)[1].lower() not in _VID_EXT:
            return web.json_response({"error": "not a video"}, status=400)
        try:
            import hashlib
            import tempfile
            key = hashlib.sha256(
                f"{os.path.realpath(path)}:{os.path.getmtime(path)}".encode()
            ).hexdigest()[:24]
            cache = os.path.join(tempfile.gettempdir(), f"radiance_poster_{key}.jpg")
            if not os.path.isfile(cache):
                import cv2  # type: ignore
                cap = cv2.VideoCapture(path)
                ok, frame = cap.read()
                cap.release()
                if not ok or frame is None:
                    return web.json_response({"error": "cannot decode first frame"},
                                             status=415)
                h, w = frame.shape[:2]
                if max(h, w) > 1024:            # poster, not a plate
                    s = 1024.0 / max(h, w)
                    frame = cv2.resize(frame, (int(w * s), int(h * s)))
                cv2.imwrite(cache, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return web.FileResponse(cache, headers={"Content-Type": "image/jpeg"})
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=500)

    @PromptServer.instance.routes.get("/radiance/media/resolve_write")
    async def resolve_write_endpoint(request):
        """Predict RadianceWrite's output path for the node's UI readout.

        Pure string computation -- no filesystem access, no directory
        creation, so it is not a file-existence oracle and needs no root
        check.
        """
        q = request.query
        try:
            prediction = _predict_write_target(
                output_path=(q.get("output_path") or "").strip(),
                format=(q.get("format") or "").strip(),
                filename=(q.get("filename") or "").strip(),
                version=int(q.get("version") or 1),
                start_frame=int(q.get("start_frame") or 1001),
                frame_padding=int(q.get("frame_padding") or 4),
                overwrite=(q.get("overwrite", "false").lower() in ("1", "true")),
            )
            return web.json_response(prediction)
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=200)

    log.debug("[Radiance/Read] HTTP routes registered: /radiance/media/*")


def _predict_write_target(
    output_path: str,
    format: str,
    filename: str = "",
    version: int = 1,
    start_frame: int = 1001,
    frame_padding: int = 4,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Predict the path RadianceWrite will produce, for the node's UI readout.

    Mirrors RadianceWrite.write()/_dispatch() path construction exactly --
    tests/test_io_audit_regressions.py::TestWritePathPrediction writes real
    files and fails if this drifts from what actually lands on disk. Pure
    string computation: nothing is created or stat'd.
    """
    if not output_path:
        return {"path": "", "note": "set output_path"}

    base = output_path
    if filename.strip():
        base = str(Path(output_path) / f"{filename.strip()}_v{version:04d}")

    stem = _fmt_stem(format)
    note = ""

    if format in _FMT_IMAGE:
        ext_map = {
            "PNG (8-bit)": ".png", "PNG (16-bit)": ".png", "JPEG": ".jpg",
            "TIFF (16-bit)": ".tiff", "TIFF (32-bit float)": ".tiff",
            "DPX": ".dpx", "WEBP": ".webp", "EXR (16-bit half)": ".exr",
            "EXR (32-bit float)": ".exr", "Radiance HDR (.hdr)": ".hdr",
        }
        ext = ext_map.get(stem, "")
        name = Path(base).name
        if not name or name in (".", ".."):
            return {"path": "", "note": "no filename: set the filename widget "
                                        "or a full output_path"}
        if not name.lower().endswith(ext):
            name += ext
        predicted = str(Path(base).with_name(name))
        note = "single image — a multi-frame batch writes frame 1 only; use a SEQ or VID format for all frames"
    elif format in _FMT_VIDEO:
        vid_ext = {"MP4 (H.264)": ".mp4", "MP4 (H.265 10-bit)": ".mp4",
                   "MOV (ProRes 422 HQ)": ".mov", "MOV (ProRes 4444)": ".mov",
                   "MOV (DNxHR HQ)": ".mxf"}.get(stem, ".mp4")
        predicted = base if base.lower().endswith(vid_ext) else base + vid_ext
        if stem == "MOV (DNxHR HQ)":
            note = "DNxHR is written into an MXF container"
    elif format in _FMT_SEQ:
        if "EXR" in stem:
            ext = ".exr"
        elif "TIFF" in stem:
            ext = ".tiff"
        elif "DPX" in stem:
            ext = ".dpx"
        elif "HDR" in stem:
            ext = ".hdr"
        else:
            ext = ".png"
        out_dir = Path(base)
        hashes = "#" * frame_padding
        predicted = str(out_dir / f"{out_dir.name}_{hashes}{ext}")
        note = f"sequence starts at frame {start_frame}"
    else:
        return {"path": "", "note": f"unknown format {format!r}"}

    if not overwrite:
        note = (note + "  ·  " if note else "") + "unique suffix if the file exists"
    return {"path": predicted, "note": note}


# Auto-register on import, the same way the OCIO routes do.
try:
    register_read_routes()
except Exception as _exc:  # pragma: no cover - server not present in tests
    log.debug("[Radiance/Read] route registration deferred: %s", _exc)
