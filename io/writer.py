"""
io/writer.py — the write engine, with no node layer above it.

This is `RadianceWrite`'s body, moved down a floor. Nothing here knows what a
ComfyUI node is, and nothing here imports one.

## Why it moved

`delivery/handler.py` used to do `from radiance.nodes.io.write import
RadianceWrite`, instantiate the node, and call `.write()` on it. The delivery
path reaching up into the node layer is backwards, and it had a concrete cost:
the handler could not be exercised without importing ComfyUI's node surface,
so the one thing that decides whether a master lands on disk correctly was the
one thing hardest to test.

The stated reason it could not move was that `RadianceWrite.write` is 118 lines
leaning on module-level helpers in the same 2972-line file. That was true, and
it is what this file is: the helpers came too. `nodes/io/write.py` imports them
straight back under their old private names, so `RadianceEXRMultiPart`,
`RadianceDigitalCinemaWrite` and the reader are untouched, and
`RadianceWrite.write` is now a signature and a delegation.

## The one thing that did not come down

`coerce_to_frames` accepts a file path or a VideoHelperSuite-style dict, and
resolving either means *reading* media -- several hundred lines of decoder that
belong to the reader, a floor up. Rather than drag the reader down or import it
upward, the reader is passed in: `read_media` is a callable the node layer
supplies and the delivery path does not, because the delivery path hands the
writer a tensor. Given a path and no reader, the error says so.

That is the whole of the layering. There is no lazy upward import hiding in a
function body here; `radiance.io.writer` depends on nothing above it, which is
checkable rather than promised — see tests/test_writer_layering.py.

## The other thing this file is: one frame at a time

Nothing here used to stream, and the cost was that the longest shot Radiance
could write was a property of how much RAM the machine had. Per 1920x1080
float32 RGB frame (24.9 MB) a video write held the source batch, `out_frames`,
the `np.stack` of it, and the raw byte string handed to ffmpeg, all at once:
measured, about 2000 frames at 1080p and 500 at 4K. `out_frames` was the
nastiest of the four, because under "Linear (pass-through)" it held views and
looked free, and under any active transform it held a second full copy, so
turning on a colour transform halved the maximum sequence length with nothing
in the UI to say so.

The shape now is:

    frames (any iterable)
      -> transform_stream()     a generator: colour, clamp, alpha, one frame
      -> dispatch_write()       consumes it, writes, releases, next

`dispatch_write` takes an iterable rather than a list, and the video branch
feeds ffmpeg's stdin as frames arrive rather than stacking the clip and handing
over one buffer. `write_frames` still accepts a tensor, because that is what a
ComfyUI node has; the ceiling there is that tensor plus one working frame. A
caller that does not already hold the shot -- a read-transform-write pipeline
over `reader.iter_sequence_frames` -- holds only the working window, and the
sequence length is then bounded by the disk.

This is measured, not asserted: tests/test_write_streaming.py reports peak
resident set for a 32-frame write and a 512-frame write and requires the
difference to be flat.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import torch

try:
    from ..config.constants import VERSION as _RADIANCE_VERSION
except Exception:  # keep the writer importable even if constants move
    _RADIANCE_VERSION = "3.1.1"

try:
    from PIL import Image as _PIL
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

# DPX has no Pillow plugin at all, in either direction. OpenImageIO is the
# VFX-standard library for it, and without it the DPX entries in the format
# menu are decorative.
try:
    import OpenImageIO as _oiio
    _HAS_OIIO = True
except ImportError:
    _HAS_OIIO = False

from .. import color_utils

try:
    from ..path_utils import strip_path_quotes
except ImportError:
    try:
        from path_utils import strip_path_quotes  # type: ignore[import]
    except ImportError:
        def strip_path_quotes(path):  # type: ignore[misc]
            return path.strip().strip('"').strip("'")

# The containment helper, not os.path.abspath. `resolve_output_path` below used
# to do a bare `Path(base)`, so a relative `output_path` resolved against the
# process working directory -- for ComfyUI that is the install root, and a
# render scattered its output through the installation. `core/system/path_utils
# .resolve_output_path` was written for exactly this class of bug and is what
# RadianceFlipbookGIF and RadianceCDLExport already route through; the flagship
# writer did not. Absolute paths still pass through untouched.
try:
    from ..core.system.path_utils import resolve_output_path as _anchor_output_dir
except ImportError:  # pragma: no cover - only outside the package
    def _anchor_output_dir(path: str) -> str:  # type: ignore[misc]
        return path

log = logging.getLogger("radiance.io.writer")

#: How much of ffmpeg's stderr to quote when an encode fails. Enough for the
#: real reason, not so much that a log line becomes a wall.
_STDERR_TAIL = 2000


# ═══════════════════════════════════════════════════════════════════════════
# § 0  Formats and colour spaces
# ═══════════════════════════════════════════════════════════════════════════

# ── Output formats (for RadianceWrite) ────────────────────────────────────
_FMT_IMAGE = [
    "IMG │ PNG (8-bit)",
    "IMG │ PNG (16-bit)",
    "IMG │ JPEG",
    "IMG │ TIFF (16-bit)",
    "IMG │ TIFF (32-bit float)",
    "IMG │ DPX",
    "IMG │ WEBP",
    "IMG │ EXR (16-bit half)",
    "IMG │ EXR (32-bit float)",
    "IMG │ Radiance HDR (.hdr)",
]
_FMT_VIDEO = [
    "VID │ MP4 (H.264)",
    "VID │ MP4 (H.265 10-bit)",
    "VID │ MOV (ProRes 422 HQ)",
    "VID │ MOV (ProRes 4444)",
    "VID │ MOV (DNxHR HQ)",
]
_FMT_SEQ = [
    "SEQ │ PNG (8-bit)",
    "SEQ │ PNG (16-bit)",
    "SEQ │ TIFF",
    "SEQ │ EXR (16-bit half)",
    "SEQ │ EXR (32-bit float)",
    "SEQ │ DPX",
    "SEQ │ Radiance HDR (.hdr)",
]

WRITE_FORMATS: List[str] = _FMT_IMAGE + _FMT_SEQ + _FMT_VIDEO


def _fmt_stem(fmt: str) -> str:
    """Strip group prefix (e.g. 'IMG │ PNG (8-bit)' → 'PNG (8-bit)')."""
    return fmt.split(" │ ", 1)[-1] if " │ " in fmt else fmt

# ── Output color spaces ────────────────────────────────────────────────────
OUTPUT_COLOR_SPACES = [
    "Linear (pass-through)",
    # display / delivery
    "sRGB",
    "Rec.709 (BT.1886)",
    "Rec.709",                       # camera OETF, pre-3.5 name, kept
    "Rec.2020",                      # BT.2020 primaries + OETF, pre-3.5 name, kept
    "P3-D65 (Gamma 2.6)",
    "PQ (HDR10 / ST.2084)",
    "HLG (Hybrid Log-Gamma)",
    # scene-linear
    "Linear Rec.709 (sRGB)",
    "Linear Rec.2020",
    "Linear P3-D65",
    "ACEScg",
    "ACES2065-1",
    "ACEScct",
    # camera log (transfer + native gamut)
    "ARRI LogC4",
    "ARRI LogC3",
    "Sony S-Log3",
    "Sony S-Log3 S-Gamut3",
    "Panasonic V-Log",
    "Canon Log 3",
    "RED Log3G10",
    "DaVinci Intermediate",
]

from ..color.encodings import WORKING_SPACES as WRITE_WORKING_SPACES  # noqa: E402

# ── EXR compressions ───────────────────────────────────────────────────────
EXR_COMPRESSIONS = ["ZIP", "ZIPS", "PIZ", "RLE", "Uncompressed", "PXR24", "B44", "B44A", "DWAA", "DWAB"]

#: Widget label → OpenEXR compression constant name.
#:
#: ALBABIT-FIX: `exr_compression` was threaded through four call layers
#: (write_frames → dispatch_write → the EXR branches) and then never referenced
#: again -- `_save_exr` hardcoded ZIP_COMPRESSION. Choosing DWAA for a
#: 2000-frame dailies sequence silently got ZIP, roughly 5x the size and time,
#: with the node reporting success. Names rather than values because the
#: constants are resolved off the installed OpenEXR module at write time.
_EXR_COMPRESSION_ATTRS = {
    "ZIP":          "ZIP_COMPRESSION",
    "ZIPS":         "ZIPS_COMPRESSION",
    "PIZ":          "PIZ_COMPRESSION",
    "RLE":          "RLE_COMPRESSION",
    "Uncompressed": "NO_COMPRESSION",
    "NONE":         "NO_COMPRESSION",
    "PXR24":        "PXR24_COMPRESSION",
    "B44":          "B44_COMPRESSION",
    "B44A":         "B44A_COMPRESSION",
    "DWAA":         "DWAA_COMPRESSION",
    "DWAB":         "DWAB_COMPRESSION",
}

#: Sequence frames written at once (see dispatch_write).
_SEQ_WRITERS = min(8, os.cpu_count() or 1)


# ═══════════════════════════════════════════════════════════════════════════
# § 1  ffmpeg and colour-space application
# ═══════════════════════════════════════════════════════════════════════════

def _ffmpeg_bin() -> str:
    """ffmpeg path, PATH first then the imageio-ffmpeg bundle."""
    from ..core.ffmpeg import require_ffmpeg
    return require_ffmpeg()


def _ffmpeg_for(encoder: str) -> str:
    """ffmpeg path that carries `encoder`, PATH first then the bundle."""
    from ..core.ffmpeg import ffmpeg_with_encoder
    return ffmpeg_with_encoder(encoder)


def _ffprobe_bin() -> str:
    """ffprobe path. imageio-ffmpeg does not ship ffprobe, so this can be
    absent even when ffmpeg is present; callers fall back to defaults."""
    from ..core.ffmpeg import ffprobe_exe
    return ffprobe_exe() or "ffprobe"


def _ffmpeg_ok() -> bool:
    # shutil.which alone missed the ffmpeg that imageio-ffmpeg ships, which
    # is the only one most Windows installs have.
    from ..core.ffmpeg import ffmpeg_available
    return ffmpeg_available()


class OutputColour:
    """What the file is encoded as, and how it was reached (for metadata)."""

    def __init__(self, color_space: str = "Linear (pass-through)",
                 working_space: str = "Linear Rec.709 (sRGB)",
                 ocio_colorspace: str = "", ocio_config: str = "",
                 reference_white_nits: float = 203.0):
        from ..color import encodings as _enc
        if working_space not in _enc.WORKING_SPACES:
            raise ValueError(f"Unknown working_space {working_space!r}. "
                             f"Known: {', '.join(_enc.WORKING_SPACES)}")
        self.color_space = color_space
        self.working = working_space
        self.ocio_colorspace = (ocio_colorspace or "").strip()
        self.ocio_config = (ocio_config or "").strip()
        self.ref_nits = float(reference_white_nits or 203.0)
        self.how = ""
        if self.ocio_colorspace:
            cfg = _enc.ocio_config(self.ocio_config)
            if cfg is None:
                raise RuntimeError("ocio_colorspace is set but OpenColorIO is not installed")
            if cfg.getColorSpace(self.ocio_colorspace) is None:
                raise ValueError(f"OCIO colorspace {self.ocio_colorspace!r} is not in "
                                 f"config {cfg.getName()!r}")
            self.label = self.ocio_colorspace
            self.gamut = None
            self.encoding = None
        elif color_space == "Linear (pass-through)":
            self.label = working_space
            self.encoding = _enc.ENCODINGS[working_space]
            self.gamut = self.encoding.gamut
        else:
            try:
                self.encoding = _enc.resolve(color_space)
            except ValueError:
                raise ValueError(
                    f"Unknown output color space {color_space!r}. Known values: "
                    f"{', '.join(OUTPUT_COLOR_SPACES)}") from None
            self.label = self.encoding.name
            self.gamut = self.encoding.gamut

    @property
    def is_linear(self) -> bool:
        return self.encoding is not None and self.encoding.name in (
            "Linear Rec.709 (sRGB)", "Linear Rec.2020", "Linear P3-D65", "ACEScg", "ACES2065-1")

    def apply(self, arr: np.ndarray) -> np.ndarray:
        from ..color import encodings as _enc
        arr = np.asarray(arr, np.float32)
        if arr.ndim == 2:
            return arr
        extra = arr[..., 3:] if arr.shape[-1] > 3 else None
        rgb = arr[..., :3]
        if self.ocio_colorspace:
            cfg = _enc.ocio_config(self.ocio_config)
            src = _enc.ENCODINGS[self.working].ocio
            if cfg.getColorSpace(src) is None:
                src = cfg.getRoleColorSpace("scene_linear")
            out = _enc.ocio_apply(rgb, src, self.ocio_colorspace, self.ocio_config)
            self.how = f"OCIO [{cfg.getName()}] {src} -> {self.ocio_colorspace}"
        elif self.color_space == "Linear (pass-through)":
            out, self.how = rgb, f"pass-through ({self.working})"
        else:
            out, self.how = _enc.encode(rgb, self.color_space, self.working, self.ref_nits)
        out = np.asarray(out, np.float32)
        return np.concatenate([out, extra], axis=-1) if extra is not None else out

    def exr_attributes(self) -> Dict[str, Any]:
        from ..color import encodings as _enc
        attrs: Dict[str, Any] = {"radiance:colorspace": self.label,
                                 "oiio:ColorSpace": self.label}
        if self.gamut:
            attrs["chromaticities"] = tuple(float(v) for v in _enc.chromaticities(self.gamut))
        if self.gamut == "AP0" and self.label == "ACES2065-1":
            attrs["acesImageContainerFlag"] = 1
        return attrs

    def ffmpeg_tags(self) -> List[str]:
        """Colour tags and the RGB->YUV matrix for a video encode."""
        name = (self.encoding.name if self.encoding else "")
        prim = {"Rec.709": "bt709", "Rec.2020": "bt2020", "P3-D65": "smpte432"}.get(self.gamut or "", "")
        trc = {"sRGB": "iec61966-2-1", "Rec.709 (BT.1886)": "bt709",
               "Rec.709 (camera OETF)": "bt709", "Rec.2020 (BT.2020 OETF)": "bt2020-10",
               "PQ (ST.2084)": "smpte2084", "HLG (BT.2100)": "arib-std-b67",
               "Linear Rec.709 (sRGB)": "linear", "Linear Rec.2020": "linear"}.get(name, "")
        matrix = "bt2020nc" if self.gamut == "Rec.2020" else "bt709"
        tags = ["-colorspace", matrix, "-color_range", "tv"]
        if prim:
            tags += ["-color_primaries", prim]
        # Pass-through writes the values unchanged, so the transfer is whatever
        # the IMAGE already carries; a ComfyUI IMAGE is usually display
        # encoded. Tagging it "linear" (the working space's name) told players
        # to decode ordinary 8-bit video as linear light. Leave it unspecified
        # unless an encoding was actually applied.
        if trc and not (self.color_space == "Linear (pass-through)" and not self.ocio_colorspace):
            tags += ["-color_trc", trc]
        return tags

    def yuv_matrix(self) -> str:
        return "bt2020" if self.gamut == "Rec.2020" else "bt709"

    @property
    def is_pass_through(self) -> bool:
        return self.color_space == "Linear (pass-through)" and not self.ocio_colorspace

    def dpx_attributes(self) -> Dict[str, Any]:
        """DPX header fields that describe the pixels actually written (FIX-017).

        Pass-through writes the IMAGE values unchanged, so their transfer is
        whatever the IMAGE carries (in ComfyUI usually display-encoded); it
        used to be labelled with the working space, so a display-encoded
        image was tagged "Linear". It is now "User defined" with no colour
        space claim.
        """
        if self.is_pass_through:
            return {"dpx:Transfer": "User defined", "dpx:Colorimetric": "User defined"}
        name = self.encoding.name if self.encoding else ""
        transfer = ("ITU-R 709-4" if name.startswith("Rec.709") else
                    "Linear" if self.is_linear else
                    "Logarithmic" if name in ("ARRI LogC4", "ARRI LogC3", "Sony S-Log3",
                                              "Sony S-Log3 S-Gamut3", "Panasonic V-Log",
                                              "Canon Log 3", "RED Log3G10", "DaVinci Intermediate",
                                              "ACEScct") else
                    "User defined")
        colorimetric = "ITU-R 709-4" if self.gamut == "Rec.709" else "User defined"
        return {"dpx:Transfer": transfer, "dpx:Colorimetric": colorimetric,
                "oiio:ColorSpace": self.label}


def _apply_output_colorspace(arr: np.ndarray, cs: str) -> np.ndarray:
    """Encode Linear Rec.709 working values as ``cs`` (kept for callers that
    have no working space; the node uses OutputColour). Raises on failure:
    a write that cannot honour the requested encoding is a failed write."""
    return OutputColour(cs).apply(arr)


# ═══════════════════════════════════════════════════════════════════════════
# § 2  Write helpers
# ═══════════════════════════════════════════════════════════════════════════
def _unique_path(path: Path) -> Path:
    """Append _001, _002 ... if path already exists."""
    if not path.exists():
        return path
    stem, ext = path.stem, path.suffix
    i = 1
    while True:
        candidate = path.parent / f"{stem}_{i:03d}{ext}"
        if not candidate.exists():
            return candidate
        i += 1


def _workflow_metadata(prompt=None, extra_pnginfo=None) -> dict:
    """
    Build the ComfyUI provenance metadata dict ({"prompt": json, "workflow":
    json, ...}) exactly as core SaveImage embeds into PNG tEXt chunks, so
    files written by Radiance carry the workflow like a saved PNG does.
    Values are JSON strings; storage encoding is handled per format.
    """
    meta = {}
    try:
        if prompt is not None:
            meta["prompt"] = json.dumps(prompt)
        if extra_pnginfo:
            for key, value in extra_pnginfo.items():
                meta[str(key)] = json.dumps(value)
    except (TypeError, ValueError) as exc:
        log.warning(f"[RadianceWrite] workflow metadata not serialisable: {exc}")
    return meta


def _save_pil_image(arr_f32: np.ndarray, path: Path, fmt: str, quality: int = 18,
                    metadata: dict | None = None) -> None:
    """Save a float32 (H, W, 3) array to an image file via Pillow.

    `quality` is the same widget value used for video CRF (0-51, lower = better).
    JPEG/WEBP use the opposite convention (0-100, higher = better), so it is
    remapped rather than passed through raw -- matching the widget's own
    tooltip ("Also JPEG quality 0-100 (remapped)"), which previously had no
    code behind it at all (see ALBABIT-FIX below).

    PNG output embeds `metadata` (workflow/prompt JSON strings) as tEXt
    chunks — the same convention as ComfyUI core SaveImage, so the file
    can be dragged back into ComfyUI to restore the workflow."""
    if not _HAS_PIL:
        raise ImportError("Pillow required for image writing.")

    # ALBABIT-FIX: tifffile.imwrite() always writes real TIFF bytes -- for
    # "PNG (16-bit)" this silently produced a file with a TIFF magic header
    # under a .png name, rejected by anything that trusts the extension.
    # cv2 writes a genuine 16-bit PNG instead (confirmed via magic-byte +
    # round-trip check) -- already used elsewhere in this module for HDR/DPX.
    if fmt == "PNG (16-bit)":
        import cv2  # type: ignore
        # Rounded, not truncated: truncation biased every sample down by up
        # to one code value.
        arr_u16 = np.rint(np.clip(arr_f32, 0, 1) * 65535).astype(np.uint16)
        if arr_u16.ndim == 3 and arr_u16.shape[-1] == 4:
            ok = cv2.imwrite(str(path), cv2.cvtColor(arr_u16, cv2.COLOR_RGBA2BGRA))
        elif arr_u16.ndim == 3:
            ok = cv2.imwrite(str(path), cv2.cvtColor(arr_u16, cv2.COLOR_RGB2BGR))
        else:
            ok = cv2.imwrite(str(path), arr_u16)
        if not ok:
            raise RuntimeError(f"Failed to write 16-bit PNG {path}")
        return

    # ALBABIT-FIX: was `"16-bit" in fmt or "TIFF" in fmt`, which also matched
    # "TIFF (32-bit float)" (the substring "TIFF" is in both TIFF formats) --
    # every 32-bit TIFF request was silently written as 16-bit instead, and
    # its own dedicated `"32-bit" in fmt` branch below was never reached.
    # "SEQ │ TIFF" (bare, no depth suffix) is still meant to land here.
    if "16-bit" in fmt or ("TIFF" in fmt and "32-bit" not in fmt):
        arr_u16 = np.rint(np.clip(arr_f32, 0, 1) * 65535).astype(np.uint16)
        # ALBABIT-FIX: this used to also do
        # `_PIL.fromarray(arr_u16, mode="I;16" if arr_f32.ndim == 2 else "RGB")`
        # here, whose result was never used (the real write is via tifffile
        # below) -- with Pillow 12.2.0, that dead call raises TypeError for
        # any RGB (ndim==3) array, crashing every 16-bit PNG/TIFF write before
        # tifffile was ever reached. Pillow can't do 16-bit RGB directly
        # anyway; tifffile handles both 2D and 3D arrays natively.
        try:
            import tifffile  # type: ignore
        except ImportError as _exc:
            # AUDIT-FIX (2026-08): this used to log a warning and write 8-bit
            # anyway. A "16-bit" master that is actually 8-bit is a silent
            # quality downgrade of a production asset -- the same downgrade
            # this module's EXR writer refuses to do (see _save_exr).
            raise ImportError(
                f"Writing {fmt!r} requires tifffile (pip install tifffile). "
                "Refusing to silently downgrade a 16-bit write to 8-bit."
            ) from _exc
        tifffile.imwrite(str(path), arr_u16,
                         photometric="rgb" if arr_u16.ndim == 3 and arr_u16.shape[-1] >= 3 else "minisblack",
                         extrasamples=["unassalpha"] if arr_u16.ndim == 3 and arr_u16.shape[-1] == 4 else None,
                         compression="zlib")
        return
    if "32-bit" in fmt:
        try:
            import tifffile  # type: ignore
        except ImportError as _exc:
            # AUDIT-FIX (2026-08): this used to swallow the ImportError at
            # DEBUG level and fall through to the 8-bit Pillow path below --
            # a float TIFF master silently written with 8-bit precision and
            # hard-clipped HDR values, discovered by the round-trip test only
            # because the test environment happened to lack tifffile.
            raise ImportError(
                f"Writing {fmt!r} requires tifffile (pip install tifffile). "
                "Refusing to silently downgrade a 32-bit float write to 8-bit."
            ) from _exc
        a32 = np.asarray(arr_f32, np.float32)
        tifffile.imwrite(str(path), a32,
                         photometric="rgb" if a32.ndim == 3 and a32.shape[-1] >= 3 else "minisblack",
                         extrasamples=["unassalpha"] if a32.ndim == 3 and a32.shape[-1] == 4 else None,
                         compression="zlib")
        return

    arr_u8 = np.rint(np.clip(arr_f32, 0, 1) * 255).astype(np.uint8)
    if ("JPEG" in fmt) and arr_u8.ndim == 3 and arr_u8.shape[-1] == 4:
        arr_u8 = arr_u8[..., :3]          # JPEG has no alpha
    pil = _PIL.fromarray(arr_u8)
    pnginfo = None
    if metadata and str(path).lower().endswith(".png"):
        try:
            from PIL.PngImagePlugin import PngInfo
            pnginfo = PngInfo()
            for _key, _val in metadata.items():
                pnginfo.add_text(_key, _val)
        except Exception:  # noqa: BLE001 — provenance is best-effort
            pnginfo = None
    if "JPEG" in fmt or "WEBP" in fmt:
        # ALBABIT-FIX: quality was hardcoded to 90 here, ignoring the widget
        # entirely -- the tooltip promised JPEG control but nothing in this
        # function ever accepted a quality argument. Remap CRF-scale (0-51,
        # lower = better) to Pillow's JPEG/WEBP scale (0-100, higher = better).
        pil_quality = max(0, min(100, round((1.0 - quality / 51.0) * 100)))
        if "JPEG" in fmt:
            pil.save(str(path), "JPEG", quality=pil_quality)
        else:
            pil.save(str(path), "WEBP", quality=pil_quality, lossless=False)
    elif pnginfo is not None:
        pil.save(str(path), pnginfo=pnginfo)
    else:
        pil.save(str(path))


def _exr_channels(arr_f32: np.ndarray) -> "list[tuple[str, np.ndarray]]":
    """Map an (H,W), (H,W,1), (H,W,3) or (H,W,4) float array to named EXR channels.

    - 1 channel  → R=G=B (grayscale/matte readable in every compositor)
    - 3 channels → R,G,B
    - 4 channels → R,G,B,A (alpha preserved)

    Channels are returned in EXR write order. Alpha is never silently dropped.
    """
    if arr_f32.ndim == 2:
        arr_f32 = arr_f32[..., None]
    if arr_f32.ndim != 3:
        raise ValueError(f"EXR data must be 2-D or 3-D, got shape {arr_f32.shape}")

    c = arr_f32.shape[2]
    if c == 1:
        plane = arr_f32[..., 0]
        return [("R", plane), ("G", plane), ("B", plane)]
    if c == 3:
        return [("R", arr_f32[..., 0]), ("G", arr_f32[..., 1]), ("B", arr_f32[..., 2])]
    if c == 4:
        return [("R", arr_f32[..., 0]), ("G", arr_f32[..., 1]),
                ("B", arr_f32[..., 2]), ("A", arr_f32[..., 3])]
    raise ValueError(f"Unsupported EXR channel count {c}; expected 1, 3, or 4.")


def _resolve_exr_compression(name: str, openexr_module) -> Tuple[Any, str]:
    """Map a widget compression label onto the installed OpenEXR constant.

    An unknown or unavailable label falls back to ZIP and says so, rather than
    raising: a compression the build does not carry is a reason to write the
    frame losslessly, not a reason to lose the render.
    """
    label = (name or "ZIP").strip()
    attr = _EXR_COMPRESSION_ATTRS.get(label) or _EXR_COMPRESSION_ATTRS.get(label.upper())
    if attr is None:
        log.warning("RadianceWrite: unknown EXR compression %r; writing ZIP. "
                    "Known values: %s", name, ", ".join(EXR_COMPRESSIONS))
        return openexr_module.ZIP_COMPRESSION, "ZIP"
    value = getattr(openexr_module, attr, None)
    if value is None:
        log.warning("RadianceWrite: this OpenEXR build has no %s; writing ZIP.", attr)
        return openexr_module.ZIP_COMPRESSION, "ZIP"
    return value, label


def _save_exr(arr_f32: np.ndarray, path: Path, half: bool,
              metadata: dict | None = None,
              compression: str = "ZIP") -> str:
    """Write a float array to OpenEXR, preserving channel count and alpha.

    Returns the backend that wrote the file ("OpenEXR", or an OpenCV
    fallback note listing what it could not write).

    Accepts (H,W), (H,W,1), (H,W,3) or (H,W,4) float data. Scene-linear
    values above 1.0 are preserved. On unrecoverable failure this raises a
    clear error rather than writing a 0-byte placeholder — silent empty/
    downgraded deliveries are never acceptable in a VFX pipeline.

    `compression` is the widget label (see EXR_COMPRESSIONS). It used to be
    accepted three call layers up and dropped on the floor here; see
    `_EXR_COMPRESSION_ATTRS`.
    """
    arr_f32 = np.asarray(arr_f32, dtype=np.float32)
    channels = _exr_channels(arr_f32)
    np_dtype = np.float16 if half else np.float32

    errors: list[str] = []

    # ── Primary: OpenEXR ──────────────────────────────────────────────────
    try:
        import OpenEXR  # type: ignore
        # ALBABIT-FIX: was OpenEXR.Header()/OutputFile()/writePixels() (the
        # legacy dict-based API). That API only recognises a fixed, small set
        # of attribute names internally -- rad_version/software (and even
        # genuinely standard attributes like "comments") were silently
        # rejected with a C-level "unknown attribute" stderr print, never a
        # Python exception, so they never actually reached disk. Confirmed
        # live: 0 of 12 metadata keys survived a round trip. The modern
        # OpenEXR.File(header_dict, channels_dict) API (same installed
        # version) persists arbitrary string/int/float attributes correctly,
        # and infers HALF vs FLOAT per channel from the numpy dtype directly
        # -- no Imath.Channel/PixelType wiring needed.
        _compression_value, _compression_label = _resolve_exr_compression(
            compression, OpenEXR)
        header: Dict[str, Any] = {
            "compression": _compression_value,
            "type": OpenEXR.scanlineimage,
            "rad_version": _RADIANCE_VERSION,
            "software": f"Radiance v{_RADIANCE_VERSION}",
        }
        if metadata:
            # ComfyUI provenance (workflow/prompt) — stored under these exact
            # unprefixed names so the file can be dragged back into ComfyUI to
            # restore the workflow, same convention as write_exr_openexr() in
            # hdr/io.py. The modern API accepts plain str values directly (the
            # byte-encoding the legacy API needed is no longer necessary).
            for _key, _val in metadata.items():
                try:
                    header[_key] = _val
                except Exception:  # noqa: BLE001 — provenance is best-effort
                    pass
        channels_dict = {
            name: np.ascontiguousarray(plane, dtype=np_dtype)
            for name, plane in channels
        }
        OpenEXR.File(header, channels_dict).write(str(path))
        return "OpenEXR"
    except ImportError as exc:
        errors.append(f"OpenEXR module unavailable ({exc})")
    except Exception as exc:  # malformed data, write error, etc.
        errors.append(f"OpenEXR write failed ({exc})")

    # ── Fallback: OpenCV with EXR support (loses alpha for >3 channels) ────
    try:
        import cv2  # type: ignore
        if (compression or "ZIP").strip().upper() != "ZIP":
            # cv2's EXR encoder exposes no compression control, so the file
            # will not be the compression that was asked for. Say so: a DWAA
            # dailies sequence that silently came out ZIP is the defect this
            # parameter was wired up to fix.
            log.warning(
                "RadianceWrite: OpenEXR is unavailable, so %r is being written "
                "through OpenCV, which cannot honour %s compression.",
                str(path), compression,
            )
        # cv2 expects BGR(A); reverse only the colour channels.
        if arr_f32.ndim == 2 or arr_f32.shape[2] == 1:
            cv_arr = arr_f32 if arr_f32.ndim == 3 else arr_f32[..., None]
            cv_arr = np.repeat(cv_arr.reshape(*arr_f32.shape[:2], 1), 3, axis=2)[..., ::-1]
        elif arr_f32.shape[2] == 4:
            rgb = arr_f32[..., :3][..., ::-1]
            cv_arr = np.concatenate([rgb, arr_f32[..., 3:4]], axis=2)  # BGRA
        else:
            cv_arr = arr_f32[..., ::-1]
        ok = cv2.imwrite(
            str(path),
            np.ascontiguousarray(cv_arr.astype(np.float32)),
            [cv2.IMWRITE_EXR_TYPE,
             cv2.IMWRITE_EXR_TYPE_HALF if half else cv2.IMWRITE_EXR_TYPE_FLOAT],
        )
        if ok:
            # FIX-016: the fallback drops every header attribute (colour
            # metadata, workflow) and the compression choice. Say so in the
            # receipt instead of reporting a normal write.
            return (f"OpenCV fallback ({'; '.join(errors)}): header metadata and "
                    f"{compression} compression not written")
        errors.append("cv2.imwrite returned False (OpenCV built without EXR support?)")
    except ImportError as exc:
        errors.append(f"OpenCV unavailable ({exc})")
    except Exception as exc:
        errors.append(f"OpenCV EXR write failed ({exc})")

    # ── No silent downgrade, no empty file: fail loudly. ──────────────────
    raise RuntimeError(
        f"Failed to write EXR '{path}'. Install OpenEXR (pip install OpenEXR Imath) "
        f"or OpenCV built with EXR support. Details: " + "; ".join(errors)
    )


def _save_dpx(arr_f32: np.ndarray, path: Path, attributes: Optional[Dict[str, Any]] = None) -> None:
    """Write a float32 (H, W, 3) array to 10-bit DPX via OpenImageIO.

    ALBABIT-FIX: DPX has no Pillow plugin at all -- "DPX" was offered in the
    format dropdown since v3.1 but never actually worked (write raised
    "unknown file extension: .dpx"; read fell through to the same Pillow
    path and failed identically). 10-bit is the traditional DPX bit depth
    for film/VFX intermediates (SMPTE 268M). write_image()/read_image()
    auto-convert between float32 [0,1] and the packed integer sample format.
    """
    if not _HAS_OIIO:
        raise RuntimeError(
            f"Failed to write DPX '{path}'. Install OpenImageIO (pip install OpenImageIO)."
        )
    raw = np.asarray(arr_f32, dtype=np.float32)
    rgb = raw[..., :3] if raw.ndim == 3 else raw
    if not np.isfinite(rgb).all():
        raise ValueError(f"DPX '{path}': the image has NaN/Inf values; DPX stores 0-1 integers.")
    over = int((rgb > 1.0 + 1e-6).sum())
    if over and (attributes or {}).get("dpx:Transfer") == "Linear":
        # FIX-017: 10-bit integer linear cannot hold scene-linear HDR. Values
        # above 1.0 used to be clipped silently under a "Linear" label.
        raise ValueError(
            f"DPX '{path}': {over} linear value(s) above 1.0 would be clipped. DPX is a 10-bit "
            "integer format: write a log encoding (ARRI LogC, ACEScct, ...) or use EXR for linear HDR.")
    if over or int((rgb < -1e-6).sum()):
        log.warning("DPX '%s': %d value(s) outside 0-1 clipped (DPX stores 0-1 integers).",
                    path, over + int((rgb < -1e-6).sum()))
    arr = np.ascontiguousarray(np.clip(raw, 0, 1), dtype=np.float32)
    h, w = arr.shape[:2]
    c = arr.shape[2] if arr.ndim == 3 else 1

    spec = _oiio.ImageSpec(w, h, c, _oiio.UINT16)
    spec.attribute("oiio:BitsPerSample", 10)
    for _k, _v in (attributes or {}).items():
        spec.attribute(_k, str(_v))
    out = _oiio.ImageOutput.create(str(path))
    if out is None:
        raise RuntimeError(f"No DPX writer available for '{path}': {_oiio.geterror()}")
    try:
        if not out.open(str(path), spec):
            raise RuntimeError(f"Failed to open DPX '{path}' for writing: {_oiio.geterror()}")
        if not out.write_image(arr):
            raise RuntimeError(f"Failed to write DPX pixels to '{path}': {_oiio.geterror()}")
    finally:
        out.close()


def _save_hdr(arr_f32: np.ndarray, path: Path) -> None:
    """Write a float32 (H, W, 3) array to Radiance HDR (.hdr / RGBE) via cv2.

    ALBABIT-FIX: restores a format lost in the v3 rewrite. The pre-v3
    write_hdr_rgbe() (hdr/io.py) is confirmed broken on round-trip (loses
    ~1 stop of range); cv2's native HDR codec (already used to read .hdr)
    is accurate instead.
    """
    arr = np.ascontiguousarray(np.maximum(arr_f32[..., :3], 0.0), dtype=np.float32)
    import cv2  # type: ignore
    if not cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Failed to write Radiance HDR {path}")


def _coerce_mask_to_alpha(mask: Any, n: int, h: int, w: int) -> Optional[np.ndarray]:
    """Normalize a ComfyUI MASK to an (N, H, W) float32 alpha array.

    Resizes to the frame size and matches the frame count (broadcasting a single
    mask, truncating extras, or padding with the last mask). Returns None when no
    usable mask is given.
    """
    if mask is None:
        return None
    m = mask
    if isinstance(m, (list, tuple)):
        m = m[0] if m else None
        if m is None:
            return None
    try:
        if hasattr(m, "detach"):
            a = m.detach().cpu().float()
        else:
            a = torch.as_tensor(np.asarray(m, dtype=np.float32))
        if a.ndim == 4 and a.shape[-1] == 1:   # (N,H,W,1) -> (N,H,W)
            a = a[..., 0]
        if a.ndim == 2:                         # (H,W) -> (1,H,W)
            a = a.unsqueeze(0)
        if a.ndim != 3:
            return None
        if tuple(a.shape[-2:]) != (h, w):
            a = torch.nn.functional.interpolate(
                a.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False
            ).squeeze(1)
        arr = a.numpy().astype(np.float32)
    except Exception as exc:
        log.warning("RadianceWrite: could not interpret mask as alpha (%s); writing without alpha.", exc)
        return None

    if arr.shape[0] == n:
        return arr
    if arr.shape[0] == 1:
        return np.repeat(arr, n, axis=0)
    if arr.shape[0] > n:
        return arr[:n]
    pad = np.repeat(arr[-1:], n - arr.shape[0], axis=0)
    return np.concatenate([arr, pad], axis=0)


def _drain(stream, sink: List[bytes]) -> None:
    """Read a pipe to exhaustion on a thread, so ffmpeg never blocks on a full
    stderr buffer while we are busy feeding its stdin. Same shape, and the same
    reason, as `core/video.py`'s drain on the decode side."""
    try:
        for chunk in iter(lambda: stream.read(65536), b""):
            sink.append(chunk)
    except (OSError, ValueError):  # pragma: no cover - pipe torn down early
        pass


def _frame_to_raw(fr: np.ndarray, src_pix_fmt: str) -> bytes:
    """One frame, as the raw bytes the ffmpeg pipe expects."""
    arr = np.asarray(fr, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    want = 4 if src_pix_fmt in ("rgba64le", "rgba") else 3
    if arr.shape[-1] > want:
        arr = arr[..., :want]
    elif arr.shape[-1] < want:
        arr = np.concatenate([arr, np.ones(arr.shape[:-1] + (1,), np.float32)], axis=-1)
    if src_pix_fmt in ("rgb48le", "rgba64le"):
        return np.ascontiguousarray(
            np.rint(np.clip(arr, 0, 1) * 65535).astype("<u2")).tobytes()
    return np.ascontiguousarray(
        np.rint(np.clip(arr, 0, 1) * 255).astype(np.uint8)).tobytes()


def _save_video_ffmpeg(
    frames: Iterable[np.ndarray],   # (H, W, C) float32 frames in [0, 1]
    output_path: str,
    fmt: str,
    fps: float,
    crf: int,
    audio_source: str,
    overwrite: bool = True,
    timeout: Optional[float] = None,
    on_frame: Optional[Callable[[int], None]] = None,
    colour: Optional["OutputColour"] = None,
    has_alpha: bool = False,
    frame_count: Optional[int] = None,
) -> str:
    """Encode frames to video via ffmpeg, streaming them into its stdin.

    3.5: every codec is fed 16-bit RGB (10-bit codecs got 8-bit before), the
    RGB -> YUV conversion uses the BT.709 / BT.2020 matrix instead of
    swscale's BT.601 default (which shifted every hue on a Rec.709 delivery),
    the file carries primaries / transfer / matrix tags, and ProRes 4444
    carries the mask as alpha.

    ALBABIT-FIX: frames used to go through an 8-bit PNG intermediate, so no
    "10-bit" format ever carried more than 8 bits of real precision. Now
    piped raw via stdin: ProRes 422 HQ/4444 get a genuine 16-bit source (the
    formats that benefit); H.264/H.265/DNxHR HQ stay 8-bit, since their
    "10-bit" is a codec/container property, not source precision.

    ALBABIT-FIX: `frames` used to be one (N,H,W,3) array that the caller built
    with `np.stack`, and the whole clip was then converted to raw bytes in one
    `.tobytes()` -- three simultaneous full copies of the sequence (the source
    batch, the stack, the byte string) before a single frame reached ffmpeg.
    Measured ceiling was about 2000 frames at 1080p. `frames` is now any
    iterable and each frame is converted and written on its own, so peak memory
    is the source frame plus one raw buffer regardless of length. That is what
    a pipe is for.

    ALBABIT-FIX: the encode ran under a hardcoded `timeout=600`, which killed
    long masters mid-file and, because `-y` had already truncated the previous
    version, left neither the old master nor a complete new one. `timeout` now
    defaults to None (no limit), matching the decode side, which made the same
    change for the same reason -- see `core/video.py:decode`. When a caller
    does set a cap and it fires, the partial file is removed rather than left
    looking like a delivery.

    ALBABIT-FIX: stderr was captured by `subprocess.run(capture_output=True)`
    and then discarded, so every encode failure surfaced as a bare exit status
    with no reason. It is drained on a thread and quoted in the exception.
    """
    if not _ffmpeg_ok():
        raise RuntimeError("ffmpeg not found.")

    # (codec, src_pix_fmt, dst_pix_fmt, ext, extra ffmpeg args)
    # ALBABIT-FIX: ProRes was forced to "-qscale:v 9", a fixed quantiser far
    # coarser than Apple's: a 1080p24 4444 master came out at 28 Mb/s, below
    # ProRes 422 Proxy, with visibly smoothed texture. prores_ks's own rate
    # control follows each profile's target, as Apple's encoder does.
    fmt_map = {
        "MP4 (H.264)":        ("libx264",  "rgb48le", "yuv420p",     ".mp4", ["-crf", str(crf), "-preset", "medium"]),
        "MP4 (H.265 10-bit)": ("libx265",  "rgb48le", "yuv420p10le", ".mp4", ["-crf", str(crf), "-preset", "medium", "-tag:v", "hvc1"]),
        "MOV (ProRes 422 HQ)":("prores_ks","rgb48le", "yuv422p10le", ".mov", ["-profile:v", "3", "-vendor", "apl0"]),
        "MOV (ProRes 4444)":  ("prores_ks","rgba64le" if has_alpha else "rgb48le",
                               "yuva444p10le" if has_alpha else "yuv444p10le", ".mov",
                               ["-profile:v", "4", "-vendor", "apl0"]
                               + (["-alpha_bits", "16"] if has_alpha else [])),
        "MOV (DNxHR HQ)":     ("dnxhd",    "rgb48le", "yuv422p",     ".mov", ["-profile:v", "dnxhr_hq"]),
    }
    codec, src_pix_fmt, dst_pix_fmt, ext, extra = fmt_map.get(
        fmt, ("libx264", "rgb48le", "yuv420p", ".mp4", ["-crf", str(crf)])
    )
    colour = colour or OutputColour()
    matrix = colour.yuv_matrix()
    # swscale converts RGB -> YUV with BT.601 unless told otherwise.
    tags = colour.ffmpeg_tags()
    frame_tags = ":".join(
        f"{key[1:]}={value}" for key, value in zip(tags[::2], tags[1::2])
        if key != "-color_range"
    )
    # Stamp the converted frames too: newer ffmpeg filter paths can replace
    # output codec colour options with unspecified input frame metadata.
    extra = ["-vf", f"scale=out_color_matrix={matrix}:out_range=tv:flags=accurate_rnd+full_chroma_int,"
             f"setparams=range=limited:{frame_tags}"] + extra + tags

    # ALBABIT-FIX: the video branch used to build its own output path by string
    # concatenation and never call `resolve_output_path`, so `overwrite=False`
    # was honoured for IMG and SEQ and ignored for video only. Re-queueing a
    # graph handed ffmpeg `-y` and destroyed an approved master. The promised
    # parent-directory creation did not happen for video either. One resolver
    # for all three branches.
    out_path = resolve_output_path(output_path, ext, overwrite)

    iterator = iter(frames)
    try:
        first = np.asarray(next(iterator), dtype=np.float32)
    except StopIteration:
        raise ValueError(
            "RadianceWrite: no frames to encode; nothing was written."
        ) from None
    if first.ndim == 2:
        h, w = first.shape
    else:
        h, w = first.shape[:2]

    # ffmpeg writes a hidden sibling, which replaces out_path only once the
    # encode has succeeded. Writing out_path directly meant `-y` truncated an
    # existing master the moment ffmpeg opened it, so any failure after that
    # (an encoder error, a timeout, a cancel) lost the approved version.
    fd, part_name = tempfile.mkstemp(
        prefix=f".{Path(out_path).stem}.", suffix=f".partial{Path(out_path).suffix}",
        dir=str(Path(out_path).parent))
    os.close(fd)
    part_path = Path(part_name)

    cmd = [
        # `-y` for the sibling this call just created. `overwrite=False` is
        # enforced when the sibling is moved into place, below.
        _ffmpeg_for(codec), "-v", "error", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", src_pix_fmt,
        "-r", str(fps),
        "-i", "pipe:0",
    ]
    if audio_source and os.path.isfile(audio_source):
        cmd += ["-i", audio_source, "-c:a", "aac"]
        if frame_count:
            # ALBABIT-FIX: "-shortest" cut the picture to the audio: a track
            # 30 ms short lost the last frame, a 5 s track on a 10 s shot lost
            # half of it. The audio is padded with silence or trimmed to the
            # picture's length instead.
            end = frame_count / fps
            cmd += ["-af", f"apad=whole_dur={end},atrim=end={end}"]
        else:
            cmd += ["-shortest"]
    cmd += ["-c:v", codec, "-pix_fmt", dst_pix_fmt] + extra + [str(part_path)]

    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except BaseException:
        _discard_partial(part_path)
        raise
    errors: List[bytes] = []
    pump = threading.Thread(target=_drain, args=(proc.stderr, errors), daemon=True)
    pump.start()

    written = 0
    timed_out = False
    frame: Optional[np.ndarray] = first
    try:
        while frame is not None:
            proc.stdin.write(_frame_to_raw(frame, src_pix_fmt))
            written += 1
            if on_frame is not None:
                on_frame(written)
            frame = next(iterator, None)
    except (BrokenPipeError, OSError):
        # ffmpeg exited early. Its stderr says why, and the exit-status check
        # below is where that gets reported -- swallowing the write error here
        # is what lets the real message through instead of "Broken pipe".
        pass
    except BaseException:
        # ALBABIT-FIX: stopped mid-clip (cancelled from ComfyUI, or a frame
        # failed): the partial encode is removed, as after a timeout.
        proc.kill()
        proc.wait()
        _discard_partial(part_path)
        raise
    finally:
        try:
            proc.stdin.close()
        except OSError:  # pragma: no cover - already torn down
            pass
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:  # pragma: no cover - only with a cap
            timed_out = True
            proc.kill()
            proc.wait()
        pump.join(timeout=5)

    tail = b"".join(errors).decode("utf-8", "replace").strip()[-_STDERR_TAIL:]

    if timed_out:
        # Never leave a half-written master behind: it is indistinguishable
        # from a short delivery. The previous master, if any, is untouched.
        _discard_partial(part_path)
        raise RuntimeError(
            f"ffmpeg did not finish encoding {Path(out_path).name} within "
            f"{timeout:g}s after {written} frame(s); the partial file was removed."
            + (f"\nffmpeg said: {tail}" if tail else "")
        )

    if proc.returncode not in (0, None):
        # Like the timeout above: the partial encode goes, and whatever was at
        # out_path before (a previous master) stays as it was.
        _discard_partial(part_path)
        raise RuntimeError(
            f"ffmpeg failed to encode {Path(out_path).name} (exit "
            f"{proc.returncode}) after {written} frame(s). The partial file was removed."
            + (f"\nffmpeg said: {tail}" if tail else
               "\nffmpeg printed nothing on stderr.")
        )

    try:
        if not overwrite and Path(out_path).exists():
            # resolve_output_path chose a free name; something took it since.
            raise FileExistsError(
                f"{out_path} appeared while encoding; not replacing it (overwrite is off).")
        os.replace(part_path, out_path)
    except BaseException:
        _discard_partial(part_path)
        raise
    return str(out_path)


def _discard_partial(path: Path) -> None:
    """Remove an unfinished encode, quietly: the caller is already failing."""
    try:
        Path(path).unlink(missing_ok=True)
    except OSError as _exc:  # pragma: no cover
        log.debug("could not remove the partial encode %s: %s", path, _exc)


def _write_audio_temp_wav(audio: Any) -> Optional[str]:
    """Write a ComfyUI AUDIO dict ({"waveform": tensor, "sample_rate": int}) to a
    temporary 32-bit float PCM WAV file, so it can be fed into the same
    `audio_source` mux path `_save_video_ffmpeg()` already uses for on-disk
    audio files. Returns the temp file path, or None if `audio` isn't a usable
    AUDIO dict. The caller owns the returned file and must delete it.

    # ALBABIT-FIX: the "audio" (AUDIO-type) input was accepted by write()'s
    # signature but never referenced anywhere else in this file -- connecting
    # an AUDIO output here had zero effect. Only the "audio_source" (a STRING
    # path to an existing file) actually worked. No torchaudio/soundfile
    # dependency needed: WAV is simple enough to write by hand (matches the
    # approach radiance.disabled used for the same problem).
    """
    if not isinstance(audio, dict):
        return None
    waveform = audio.get("waveform")
    if waveform is None:
        return None

    import struct

    sr = int(audio.get("sample_rate", 44100))
    wav_np = waveform.detach().cpu().numpy()
    if wav_np.ndim == 3:        # (B, C, T) -> first batch item
        wav_np = wav_np[0]
    if wav_np.ndim == 1:        # (T,) -> (1, T) mono
        wav_np = wav_np[np.newaxis, :]
    n_channels = wav_np.shape[0]
    raw = np.ascontiguousarray(wav_np.T).astype(np.float32).tobytes()  # interleaved (T, C)

    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    block_align = n_channels * 4
    with open(path, "wb") as f:
        f.write(
            b"RIFF" + struct.pack("<I", 36 + len(raw)) +
            b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 3, n_channels, sr,
                sr * block_align, block_align, 32) +
            b"data" + struct.pack("<I", len(raw)) + raw
        )
    return path


# ═══════════════════════════════════════════════════════════════════════════
# § 3  The writer
# ═══════════════════════════════════════════════════════════════════════════

def coerce_to_frames(
    data: Any,
    *,
    read_media: Optional[Callable[[Any], Optional[np.ndarray]]] = None,
) -> Tuple[np.ndarray, Optional[float]]:
    """
    Normalise any ComfyUI output to (N, H, W, C) float32 numpy array.

    Handled input types
    ───────────────────
    torch.Tensor          Standard IMAGE (B,H,W,C) or (H,W,C)
    list / tuple          List of IMAGE tensors — stacked into batch
    str (file path)       Image or video file read from disk
    dict                  VHS, AnimateDiff, VideoHelperSuite and similar;
                          also recursively searched for embedded video paths.
                          A dict with "samples" key (LATENT) raises a helpful error.

    Returns
    -------
    (frames_nhwc, detected_fps)   detected_fps is None when not available.
    """
    detected_fps: Optional[float] = None

    # ── 1. Standard IMAGE tensor ──────────────────────────────────────
    if isinstance(data, torch.Tensor):
        arr = data.detach().cpu().float().numpy()
        if arr.ndim == 3:
            arr = arr[np.newaxis]        # (H,W,C) → (1,H,W,C)
        return arr, detected_fps

    # ── 2. List / tuple of tensors ────────────────────────────────────
    if isinstance(data, (list, tuple)):
        # If all elements are tensors, stack them
        if data and isinstance(data[0], torch.Tensor):
            arr = np.stack(
                [t.detach().cpu().float().numpy() for t in data], axis=0
            )
            return arr, detected_fps
        # Otherwise, recurse on first element that looks like frames.
        #
        # ALBABIT-FIX: this loop used to `except Exception: continue`, throwing
        # away the real reason each element failed -- a missing file, a corrupt
        # MOV, a LATENT that needed a VAE Decode -- and ending on the generic
        # "Cannot extract frames from list", which names nothing anyone can act
        # on. The reasons are collected and reported.
        reasons: List[str] = []
        for item in data:
            try:
                return coerce_to_frames(item, read_media=read_media)
            except Exception as exc:  # noqa: BLE001 - re-raised below, with the reason
                reasons.append(f"{type(item).__name__}: {type(exc).__name__}: {exc}")
        raise ValueError(
            "RadianceWrite: cannot extract frames from list of "
            f"{type(data[0]).__name__ if data else 'empty'}. "
            "Tried each element: " + " | ".join(reasons)
        )

    # ── 3. Dict (VHS, AnimateDiff, custom video nodes …) ─────────────
    if isinstance(data, dict):
        # Grab fps metadata if available
        for fps_key in ("fps", "frame_rate", "framerate"):
            if fps_key in data and isinstance(data[fps_key], (int, float)):
                detected_fps = float(data[fps_key])
                break

        # Check for IMAGE tensor stashed inside the dict
        for key in ("frames", "images", "image", "output", "result"):
            if key in data and isinstance(data[key], torch.Tensor):
                arr, _ = coerce_to_frames(data[key], read_media=read_media)
                return arr, detected_fps

        # LATENT dict — helpful error
        if "samples" in data and "batch_index" not in data:
            raise ValueError(
                "RadianceWrite received a LATENT tensor.  "
                "Decode it with a VAE Decode node first, then connect the IMAGE output."
            )

        # Search for embedded video file path (VHS-style). Reading media is
        # the reader's job, and the reader lives a layer up in the node module,
        # so it is handed in rather than imported. That is the whole reason
        # this function can live down here at all.
        if read_media is not None:
            arr = read_media(data)
            if arr is not None:
                return arr, detected_fps

        raise ValueError(
            f"RadianceWrite: dict input has no recognised IMAGE key or video path.  "
            f"Keys found: {list(data.keys())}"
        )

    # ── 4. String file path ───────────────────────────────────────────
    if isinstance(data, str):
        if read_media is not None:
            arr = read_media(data)
            if arr is not None:
                return arr, detected_fps
        raise ValueError(f"RadianceWrite: path not found or unsupported extension: {data!r}")

    raise ValueError(
        f"RadianceWrite: unsupported input type {type(data).__name__!r}.  "
        "Connect an IMAGE, batched IMAGE, video file path, or video node output."
    )


def resolve_output_path(base: str, ext: str, overwrite: bool) -> Path:
    """Append `ext` to `base`, without Path.with_suffix's truncation.

    `with_suffix` REPLACES everything after the last dot in the final path
    component. VFX filenames are full of dots -- sh010.comp, plate.v2,
    bg.matte -- and `base` here is "<dir>/<filename>_v0001", so
    "sh010.comp_v0001" has a "suffix" of ".comp_v0001" and every version
    collapsed onto the same "sh010.exr". With overwrite defaulting to True,
    each render silently destroyed the previously approved one and reported
    success with the truncated path.

    An empty `base` was worse: Path("") is Path("."), whose name is "", and
    with_suffix raised `ValueError: PosixPath('.') has an empty name` --
    an opaque crash for the ordinary case of leaving `filename` blank.

    ALBABIT-FIX: `base` used to go straight into `Path(base)` with no
    anchoring, so a relative `output_path` resolved against the process working
    directory -- the ComfyUI install root -- and scattered render directories
    through the installation. `core/system/path_utils.resolve_output_path` was
    written for that class of bug and is what RadianceFlipbookGIF and
    RadianceCDLExport already use; the flagship writer did not. Absolute paths
    are honoured untouched, and `..` traversal is rejected there.
    """
    p = Path(_anchor_output_dir(str(base)) if str(base).strip() else base)
    stem = p.name
    if not stem or stem in (".", ".."):
        raise ValueError(
            "RadianceWrite: no output filename. Set the `filename` widget, "
            "or give `output_path` a full file path rather than a directory."
        )
    if not stem.lower().endswith(ext.lower()):
        stem += ext
    p = p.with_name(stem)
    if not overwrite:
        p = _unique_path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def transform_stream(
    frames: Iterable[np.ndarray],
    *,
    color_space: str = "Linear (pass-through)",
    apply_legal_range: bool = False,
    alpha: Optional[np.ndarray] = None,
    colour: Optional["OutputColour"] = None,
) -> Iterator[np.ndarray]:
    """Yield frames with the output transform applied, one at a time.

    ALBABIT-FIX: `write_frames` used to build `out_frames`, a list holding a
    second full-length copy of the sequence, and only then start writing. With
    ``color_space="Linear (pass-through)"`` and no mask or clamp the list held
    views, so it looked free; the moment any transform was active every frame
    became a fresh allocation and the maximum sequence length halved, silently
    and unpredictably -- enabling a colour transform changed how long a shot
    could be. Nothing now holds frame N while frame N+1 is being written: the
    transform is a generator and the writers below consume it.

    `alpha` is the (N,H,W) array from `_coerce_mask_to_alpha`, indexed by
    position, so this is the one thing here that is still O(sequence). It is a
    single-channel mask the caller already had in hand, not a second copy of
    the frames.
    """
    colour = colour or OutputColour(color_space)
    for i, fr in enumerate(frames):
        fr = colour.apply(fr)
        if apply_legal_range:
            # ALBABIT-FIX: legal range limits the colour. It also squeezed the
            # alpha of an RGBA image (transparent became 6% opaque, opaque 92%).
            fr = np.concatenate([np.clip(fr[..., :3], 16 / 255.0, 235 / 255.0), fr[..., 3:]], axis=-1)
        # ALBABIT-FIX: a connected mask is the alpha. It was ignored when the
        # image already carried one (an RGBA VAE's output).
        if alpha is not None and i < len(alpha) and fr.ndim == 3 and fr.shape[-1] in (3, 4):
            fr = np.concatenate([fr[..., :3], alpha[i][..., None]], axis=-1)
        yield fr


def write_frames(
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
    # ALBABIT-FIX: this defaulted to True while RadianceWrite's INPUT_TYPES
    # declares False. ComfyUI supplies the widget value, so graphs were
    # safe, but every programmatic caller -- the Digital Cinema shim,
    # delivery/handler.py -- silently got destructive overwrite from a
    # signature that disagreed with the UI it belongs to.
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
    read_media:     Optional[Callable[[Any], Optional[np.ndarray]]] = None,
    on_frame:       Optional[Callable[[int, int], None]] = None,
    receipt:        Optional[Dict[str, Any]] = None,
):
    output_path = strip_path_quotes(output_path)
    # Built before any frame is touched, so a bad colour setting fails before
    # a file is created.
    colour = OutputColour(color_space, working_space, ocio_colorspace, ocio_config,
                          hdr_reference_nits)
    frames, detected_fps = coerce_to_frames(image, read_media=read_media)   # (N, H, W, C)
    # ALBABIT-FIX: scene-linear HDR (values above 1.0) written unchanged into a
    # video plays back too dark: players read it as display code values.
    if format in _FMT_VIDEO and colour.is_linear and len(frames) and float(frames[0].max()) > 1.0:
        log.warning(
            "RadianceWrite: the image is scene-linear HDR (values above 1.0) and color_space "
            "%r writes it into the %s video unchanged, so it plays back too dark. For an HDR "
            "video set color_space to 'PQ (HDR10 / ST.2084)' or 'HLG (Hybrid Log-Gamma)'; "
            "for SDR, 'Rec.709 (BT.1886)' or 'sRGB' (values above 1.0 clip). Keep "
            "'Linear (pass-through)' for EXR.", color_space, _fmt_stem(format))
    # ALBABIT-FIX: fps==24.0 used to mean "auto-detect", indistinguishable
    # from a user explicitly choosing 24 -- an explicit 24 on a 23.976
    # source was silently overridden. 0.0 is now the unambiguous "auto"
    # sentinel; any other value (including 24.0) is respected as-is.
    if fps <= 0.0:
        fps = detected_fps if detected_fps is not None else 24.0

    # Proxy downscale for faster preview
    if proxy_scale > 0:
        t = torch.from_numpy(frames).movedim(-1, 1)
        t = torch.nn.functional.interpolate(t, scale_factor=proxy_scale, mode="bilinear")
        frames = t.movedim(1, -1).numpy()

    # Build output path from filename + version when provided
    if filename.strip():
        ver_str = f"v{version:04d}"
        output_path = str(Path(output_path) / f"{filename}_{ver_str}")
    n, h, w, c = frames.shape

    # Optional alpha: written as the alpha channel for EXR and PNG formats.
    # ALBABIT-FIX: was EXR-only; the old radiance.disabled fork also wrote
    # alpha for PNG (8-bit via Pillow RGBA, 16-bit via cv2 BGRA) -- both
    # paths already accept a 4-channel array transparently, so only this
    # gating condition needed to change.
    # Alpha travels in every format that can hold it: EXR, PNG, TIFF, DPX,
    # WEBP and ProRes 4444 (was EXR and PNG only).
    _alpha_ok = any(k in format for k in ("EXR", "PNG", "TIFF", "DPX", "WEBP", "ProRes 4444"))
    alpha = _coerce_mask_to_alpha(mask, n, h, w) if mask is not None and _alpha_ok else None

    # Apply color space + broadcast-safe clamp.
    #
    # The clamp is now gated on the output format. It used to be
    # unconditional, so exporting "EXR (32-bit float)" with broadcast_safe
    # on -- which the delivery panel defaults to -- clamped every pixel of a
    # scene-referred master into [16/255, 235/255]: highlights destroyed,
    # blacks lifted 6.3%, and the .exr extension making it look like a
    # legitimate HDR deliverable. Legal-range limiting is a video/8-bit
    # concept and must never touch a float or DPX deliverable.
    _is_float_format = ("EXR" in format or "HDR" in format
                        or "32-bit float" in format or "DPX" in format)
    # Video is always encoded to legal (TV) range by the RGB -> YUV step, so
    # clamping RGB to 16-235 first squeezed the range twice.
    apply_legal_range = bool(broadcast_safe) and not _is_float_format and format not in _FMT_VIDEO
    if broadcast_safe and _is_float_format:
        log.info("RadianceWrite: broadcast_safe ignored for %s "
                 "(legal-range limiting does not apply to float formats)", format)

    out_frames = transform_stream(
        frames, color_space=color_space,
        apply_legal_range=apply_legal_range, alpha=alpha, colour=colour,
    )

    # ALBABIT-FIX: "audio" (AUDIO-type input) was accepted but never used --
    # only "audio_source" (a STRING path to an existing file) actually muxed
    # into video output. When no explicit audio_source is given, fall back
    # to writing the connected AUDIO tensor to a temp WAV so it reaches the
    # same mux path. Only relevant for video output -- image/sequence
    # formats have no audio track.
    effective_audio_source = audio_source
    temp_audio_wav: Optional[str] = None
    if format in _FMT_VIDEO and not audio_source.strip() and audio is not None:
        temp_audio_wav = _write_audio_temp_wav(audio)
        if temp_audio_wav:
            effective_audio_source = temp_audio_wav

    try:
        # ALBABIT-FIX: ProRes 4444 took its alpha from the mask alone, so an
        # RGBA image written without a mask lost its own.
        saved, count = dispatch_write(
            out_frames, output_path, format,
            fps, quality, exr_compression,
            start_frame, frame_padding,
            effective_audio_source, overwrite,
            prompt, extra_pnginfo,
            frame_count=n, colour=colour, has_alpha=alpha is not None or c == 4,
            on_frame=on_frame, receipt=receipt,
        )
        if receipt is not None:
            receipt.update({"format": format, "requested_frames": n, "written_frames": count,
                            "complete": count == (1 if format in _FMT_IMAGE else n),
                            "colour": colour.how or colour.label})
        log.info("RadianceWrite: saved %d frame(s) → %s  [%s]", count, saved, colour.how or colour.label)
        # Return the saved path so programmatic callers (delivery/handler.py)
        # can report it. ComfyUI ignores extra tuple entries for a node whose
        # RETURN_TYPES is empty, so this is safe for graph use.
        return (saved, count)

    except Exception as e:
        # Never silently swallow a delivery failure — surface it so the node
        # turns red in ComfyUI instead of reporting a phantom success.
        log.error("RadianceWrite failed: %s", e)
        raise

    finally:
        if temp_audio_wav and os.path.exists(temp_audio_wav):
            try:
                os.unlink(temp_audio_wav)
            except OSError as _exc:
                log.debug(
                    "[Radiance] write(): ignoring %s from `os.unlink(temp_audio_wav)`: %s",
                    type(_exc).__name__, _exc,
                )


def _sequence_stem(out_dir: Path, seq_stem: str, ext: str, pad_fmt: str,
                   start_frame: int, count: int, overwrite: bool) -> str:
    """Pick the sequence's filename stem once, for the whole sequence.

    ALBABIT-FIX: `overwrite=False` used to apply `_unique_path` PER FRAME, so a
    partial re-render produced `shot_1001_001.exr` for the frames that already
    existed and `shot_1005.exr` for the ones that did not: neither the old
    sequence nor the new one, and the `%04d` glob matched neither. A collision
    is a property of the sequence, so it is decided once, here, and every frame
    then lands under the same stem -- which is what keeps the glob valid.
    """
    if overwrite:
        return seq_stem

    def collides(stem: str) -> bool:
        return any(
            (out_dir / f"{stem}_{pad_fmt % (start_frame + i)}{ext}").exists()
            for i in range(max(1, count))
        )

    if not collides(seq_stem):
        return seq_stem
    i = 1
    while collides(f"{seq_stem}_{i:03d}"):
        i += 1
    return f"{seq_stem}_{i:03d}"


def dispatch_write(
    frames:         Iterable[np.ndarray],
    output_path:    str,
    format:         str,
    fps:            float,
    quality:        int,
    exr_compression: str,
    start_frame:    int,
    frame_padding:  int,
    audio_source:   str,
    overwrite:      bool,
    prompt:         Any = None,
    extra_pnginfo:  Any = None,
    frame_count:    Optional[int] = None,
    encode_timeout: Optional[float] = None,
    colour:         Optional["OutputColour"] = None,
    has_alpha:      bool = False,
    on_frame:       Optional[Callable[[int, int], None]] = None,
    receipt:        Optional[Dict[str, Any]] = None,
) -> Tuple[str, int]:
    """Write `frames` in `format` and return (path, frames written).

    When ``receipt`` is a dict it is filled with the files actually written
    (``files``: [{"path", "frame", "writer"}]), sorted by frame (FIX-016).

    `frames` is any iterable of (H,W,C) float32 frames -- a list, an ndarray,
    or a generator that produces one frame at a time and never holds the
    sequence. It used to be typed `List[np.ndarray]` and the video branch
    stacked it, so the sequence length was bounded by RAM rather than by disk.
    Pass `frame_count` when the length is known and `frames` cannot report it;
    it is only used to size the sequence-collision check and the returned
    count, never to allocate.

    `on_frame(written, frame_count)` follows sequence and video writes, for
    the caller's progress bars.
    """
    n = frame_count
    if n is None:
        n = len(frames) if hasattr(frames, "__len__") else None  # type: ignore[arg-type]
    colour = colour or OutputColour()

    # ── Single image / EXR (first frame only) ──────────────────────────
    if format in _FMT_IMAGE:
        stem = _fmt_stem(format)
        ext_map = {
            "PNG (8-bit)":         ".png",
            "PNG (16-bit)":        ".png",
            "JPEG":                ".jpg",
            "TIFF (16-bit)":       ".tiff",
            "TIFF (32-bit float)": ".tiff",
            "DPX":                 ".dpx",
            "WEBP":                ".webp",
            "EXR (16-bit half)":   ".exr",
            "EXR (32-bit float)":  ".exr",
            "Radiance HDR (.hdr)": ".hdr",
        }
        ext  = ext_map[stem]
        # Only the first frame is wanted, and `frames` may be a generator, so
        # take it by iteration rather than by subscript. The rest is never
        # produced: a single-image write of a 2000-frame batch costs one frame.
        first = next(iter(frames), None)
        if first is None:
            raise ValueError("RadianceWrite: no frames to write.")
        path = resolve_output_path(output_path, ext, overwrite)
        meta = _workflow_metadata(prompt, extra_pnginfo)
        writer = stem
        if "EXR" in stem:
            writer = _save_exr(first, path, half="16-bit" in stem, metadata={**meta, **colour.exr_attributes()},
                               compression=exr_compression)
        elif "DPX" in stem:
            _save_dpx(first, path, colour.dpx_attributes())
        elif "HDR" in stem:
            _save_hdr(first, path)
        else:
            _save_pil_image(first, path, stem, quality, metadata=meta)
        _record(receipt, path, None, writer)
        return str(path), 1

    # ── Video ─────────────────────────────────────────────────────────
    if format in _FMT_VIDEO:
        # Streamed into ffmpeg's stdin frame by frame. This used to be
        # `np.stack(frames, axis=0)`, a full second copy of the sequence, on
        # top of the raw byte string `_save_video_ffmpeg` then built from it.
        encoded = [0]

        def _count(i: int) -> None:
            encoded[0] = i
            if on_frame is not None:
                on_frame(i, n)

        out = _save_video_ffmpeg(
            frames,
            output_path,
            _fmt_stem(format),
            fps,
            quality,
            audio_source,
            overwrite=overwrite,
            timeout=encode_timeout,
            on_frame=_count,
            colour=colour,
            has_alpha=has_alpha,
            frame_count=n,
        )
        _record(receipt, out, None, f"ffmpeg {_fmt_stem(format)}")
        if receipt is not None:
            receipt["frames_encoded"] = encoded[0]
        return str(out), encoded[0]

    # ── Sequences ─────────────────────────────────────────────────────
    if format in _FMT_SEQ:
        stem = _fmt_stem(format)
        # Anchored, for the same reason resolve_output_path is: a relative
        # sequence directory used to be created wherever ComfyUI happened to
        # be started from.
        out_dir  = Path(_anchor_output_dir(str(output_path))
                        if str(output_path).strip() else output_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        seq_stem = out_dir.name
        pad_fmt  = f"%0{frame_padding}d"

        is_exr   = "EXR" in stem
        is_tiff  = "TIFF" in stem
        is_dpx   = "DPX" in stem
        is_hdr   = "HDR" in stem
        half_exr = "16-bit" in stem

        if is_exr:
            ext = ".exr"
        elif is_tiff:
            ext = ".tiff"
        elif is_dpx:
            ext = ".dpx"
        elif is_hdr:
            ext = ".hdr"
        else:
            ext = ".png"

        seq_png_meta = _workflow_metadata(prompt, extra_pnginfo)
        seq_meta = {**seq_png_meta, **colour.exr_attributes()}
        seq_stem = _sequence_stem(out_dir, seq_stem, ext, pad_fmt,
                                  start_frame, n if n is not None else 1,
                                  overwrite)

        def save(fr: np.ndarray, path: Path, frame: int) -> None:
            writer = stem
            if is_exr:
                writer = _save_exr(fr, path, half=half_exr, metadata=seq_meta,
                                   compression=exr_compression)
            elif is_dpx:
                _save_dpx(fr, path, colour.dpx_attributes())
            elif is_hdr:
                _save_hdr(fr, path)
            else:
                _save_pil_image(fr, path, stem, quality, metadata=seq_png_meta)
            _record(receipt, path, frame, writer)

        # `frames` is a generator in the normal path and at most `_SEQ_WRITERS`
        # frames are in flight, so the sequence length is bounded by the disk
        # rather than by RAM.
        # ALBABIT-FIX: several frames are written at once. Compression is most
        # of a frame's cost and runs without the GIL, but frame by frame it used
        # one core: 58 s for 241 1080p frames in 32-bit float EXR.
        written = 0

        def finish(job) -> None:
            nonlocal written
            job.result()
            written += 1
            if on_frame is not None:
                on_frame(written, n)

        with ThreadPoolExecutor(max_workers=_SEQ_WRITERS) as pool:
            in_flight: deque = deque()
            for i, fr in enumerate(frames):
                path = out_dir / f"{seq_stem}_{(pad_fmt % (start_frame + i))}{ext}"
                in_flight.append(pool.submit(save, fr, path, start_frame + i))
                del fr
                if len(in_flight) >= _SEQ_WRITERS:
                    finish(in_flight.popleft())
            for job in in_flight:
                finish(job)

        if receipt is not None:
            receipt.setdefault("files", []).sort(key=lambda e: (e["frame"] is None, e["frame"]))
        return str(out_dir), written

    raise ValueError(f"Unknown format: {format!r}")


_RECEIPT_LOCK = threading.Lock()


def _record(receipt: Optional[Dict[str, Any]], path, frame: Optional[int], writer: Optional[str]) -> None:
    """Add one written file to a dispatch_write receipt (thread-safe)."""
    if receipt is None:
        return
    p = Path(str(path))
    entry = {"path": str(p), "frame": frame, "writer": writer or "",
             "bytes": p.stat().st_size if p.is_file() else 0}
    with _RECEIPT_LOCK:
        receipt.setdefault("files", []).append(entry)
