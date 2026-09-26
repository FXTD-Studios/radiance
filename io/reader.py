"""
io/reader.py — the read engine, with no node layer above it.

This is `RadianceRead`'s body and every decoder under it, moved down a floor,
the same way `io/writer.py` was. Nothing here knows what a ComfyUI node is, and
nothing here imports one.

## Why it moved

`nodes/io/write.py` was 2972 lines and came down to 2201 when the write engine
left. What remained was the reader, and it was the larger half of the problem:
the decoders are the part other things want. `nodes/pipeline/dcc.py` and
`nodes/pipeline/studio_integrations.py` both reach into the node module for
`_read_sequence` and `_load_video_to_numpy` — sideways imports into a module
whose real job is to declare two ComfyUI nodes — and the writer could not take
`coerce_to_frames` all the way down precisely because resolving a path to
frames meant reading media, which lived up here.

So: the decoders, the colour-space decode, the path-kind detection, the image,
sequence and video readers, the write-input coercion helpers and the UI probe
all move to `radiance/io/reader.py`. `nodes/io/write.py` imports them straight
back under their old private names, so `RadianceRead`, `RadianceEXRMultiPart`,
`RadianceDigitalCinemaRead` and the HTTP routes are untouched by the move, and
`RadianceRead.read` becomes a signature and a delegation.

## What stayed up

The browse widget. `_get_input_files` and `_resolve_browse` translate a
filename in ComfyUI's input directory into a path, which needs `folder_paths`
— the one genuinely node-layer thing in the reader. `read_frames` therefore
takes a path and nothing else; the node resolves `browse` first and hands the
result down. That is the mirror image of the writer's `read_media` callback,
and it is the whole of the coupling.

There is no lazy upward import hiding in a function body here;
`radiance.io.reader` depends on nothing above it, which is checkable rather
than promised — see tests/test_reader_layering.py.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch

try:
    from PIL import Image as _PIL
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

# DPX, Cineon and the camera formats have no Pillow plugin. OpenImageIO is the
# VFX-standard library for them, and without it those entries in the format
# tables are decorative.
try:
    import OpenImageIO as _oiio
    _HAS_OIIO = True
except ImportError:
    _HAS_OIIO = False

from .. import color_utils
from ..core import formats as _formats
from ..core.video import VIDEO_EXTENSIONS as _VIDEO_EXTENSIONS

try:
    from ..path_utils import strip_path_quotes
except ImportError:
    try:
        from path_utils import strip_path_quotes  # type: ignore[import]
    except ImportError:
        def strip_path_quotes(path):  # type: ignore[misc]
            return path.strip().strip('"').strip("'")

log = logging.getLogger("radiance.io_unified")


# ── Input color spaces (for RadianceRead decode) ───────────────────────────
# The list used to stop at ACEScct, which left no way to decode the three
# things that arrive in a MOV more often than any camera log: a Rec.709
# delivery, an HDR10 master, and an HLG broadcast file. Radiance already had
# all three inverses in color/transfer.py; only the menu was missing them.
INPUT_COLOR_SPACES = [
    "Auto / Linear (pass-through)",
    # display / delivery
    "sRGB",
    "Rec.709 (BT.1886)",
    "Rec.709 (camera OETF)",
    "Rec.2020 (BT.2020 OETF)",
    "P3-D65 (Gamma 2.6)",
    "PQ (ST.2084)",
    "HLG (BT.2100)",
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

#: What the IMAGE output is in. Every decode lands here (colour.encodings).
from ..color.encodings import WORKING_SPACES as READ_WORKING_SPACES  # noqa: E402


# ── Image file extensions ──────────────────────────────────────────────────
# These three used to be hand-typed sets, and between them they decided what the
# node would even attempt. Nine image extensions meant TGA, SGI, PPM, JP2 and
# PCX were classified "unknown" and refused -- despite the reader underneath
# opening every one of them correctly, measured. core.formats builds the tables
# from what the installed backends actually register, so the answer tracks the
# install instead of a list someone typed in 2024.
_IMG_EXT  = _formats.image_extensions()
_VID_EXT  = set(_VIDEO_EXTENSIONS)
_EXR_EXT  = set(_formats.EXR_EXTENSIONS)



# ═══════════════════════════════════════════════════════════════════════════
# § 1  Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _np_to_tensor(arr: np.ndarray) -> torch.Tensor:
    """(H, W, C) float32 ndarray → (1, H, W, C) float32 tensor.

    This is a pure dtype/shape converter. It performs **no** value
    normalization: scene-linear / HDR / EXR data is preserved exactly,
    including values above 1.0. Callers that read integer formats are
    responsible for dividing by the type maximum (255 / 65535) *before*
    calling this function. Normalizing here based on pixel magnitude
    silently destroyed HDR plates whose highlights exceed 2.0 (the old
    `arr.max() > 2.0` heuristic), so it has been removed.
    """
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    # np.ascontiguousarray guards against torch.from_numpy receiving a
    # non-contiguous view (e.g. a channel slice), which would corrupt layout.
    # OpenEXR's np.frombuffer views are contiguous but read-only; Torch warns
    # that wrapping those can lead to undefined behaviour if later mutated.
    arr = np.ascontiguousarray(arr)
    if not arr.flags.writeable:
        arr = arr.copy()
    return torch.from_numpy(arr).unsqueeze(0)


def _tensor_to_np(t: torch.Tensor) -> np.ndarray:
    """(1, H, W, C) or (H, W, C) tensor → float32 ndarray."""
    arr = t.detach().cpu().float().numpy()
    return arr[0] if arr.ndim == 4 else arr



def _rec709_to_linear(arr: np.ndarray) -> np.ndarray:
    """BT.1886 decode — the display transfer function of a Rec.709 delivery.

    A pure 2.4 power law with black at zero, which is what BT.1886 reduces to
    when Lb = 0. This is the right inverse for a graded Rec.709 MOV or MP4;
    it is *not* the BT.709 camera OETF, which nothing is actually encoded with.
    """
    return np.sign(arr) * np.power(np.abs(arr, dtype=np.float32), 2.4, dtype=np.float32)


#: Input colour space name -> the function that takes it back to scene-linear.
_INPUT_DECODERS = {
    "Rec.709 (BT.1886)":     _rec709_to_linear,
    "sRGB":                  lambda a: color_utils.srgb_to_linear(a),
    "ARRI LogC4":            lambda a: color_utils.logc4_to_linear(a),
    "ARRI LogC3":            lambda a: color_utils.logc3_to_linear(a),
    "Sony S-Log3":           lambda a: color_utils.slog3_to_linear(a),
    "Panasonic V-Log":       lambda a: color_utils.vlog_to_linear(a),
    "Canon Log 3":           lambda a: color_utils.canonlog3_to_linear(a),
    "RED Log3G10":           lambda a: color_utils.log3g10_to_linear(a),
    "DaVinci Intermediate":  lambda a: color_utils.davinci_intermediate_to_linear(a),
    "PQ (ST.2084)":          lambda a: color_utils.pq_to_linear(a),
    "HLG (BT.2100)":         lambda a: color_utils.hlg_to_linear(a),
    "ACEScct":               lambda a: color_utils.acescct_to_linear(a),
}


import contextvars as _contextvars  # noqa: E402

#: Colour settings for the read in progress (working space, OCIO override,
#: HDR reference white, the gamut an EXR declared, and a log of what ran).
#: A context variable so the per-frame helpers below keep their signatures.
_COLOUR: "_contextvars.ContextVar[Optional[dict]]" = _contextvars.ContextVar(
    "radiance_read_colour", default=None)


def _colour_ctx() -> dict:
    ctx = _COLOUR.get()
    if ctx is None:
        ctx = {"working": "Linear Rec.709 (sRGB)", "reference_white_nits": 203.0,
               "ocio_colorspace": "", "ocio_config": "", "file_gamut": None, "applied": []}
    return ctx


def _note(ctx: dict, text: str) -> None:
    if text not in ctx["applied"]:
        ctx["applied"].append(text)


def _apply_input_colorspace(arr: np.ndarray, cs: str) -> np.ndarray:
    """Decode file values to scene-linear in the working space.

    Transfer AND primaries: before 3.5 only the transfer was undone, so a
    camera-log plate came out in the camera's own gamut. Conversion runs
    through OpenColorIO when available (the same result Nuke or Resolve give
    for these names) and the analytic path otherwise; see color/encodings.py.
    A decode that cannot run raises; it used to log and return the file
    values, which then flowed on as if they were linear.
    """
    from ..color import encodings as _enc

    ctx = _colour_ctx()
    working = ctx["working"]
    rgb = np.asarray(arr[..., :3], np.float32)
    extra = arr[..., 3:] if arr.shape[-1] > 3 else None

    ocio_cs = (ctx.get("ocio_colorspace") or "").strip()
    if ocio_cs:
        cfg = _enc.ocio_config(ctx.get("ocio_config", ""))
        if cfg is None:
            raise RuntimeError("ocio_colorspace is set but OpenColorIO is not installed "
                               "(pip install opencolorio)")
        dst = _enc.ENCODINGS[working].ocio
        if cfg.getColorSpace(dst) is None:
            dst = cfg.getRoleColorSpace("scene_linear") if hasattr(cfg, "getRoleColorSpace") else "scene_linear"
            _note(ctx, f"working space {working!r} not in OCIO config {cfg.getName()!r}; "
                       f"used its scene_linear role ({dst})")
        out = _enc.ocio_apply(rgb, ocio_cs, dst, ctx.get("ocio_config", ""))
        _note(ctx, f"OCIO [{cfg.getName()}] {ocio_cs} -> {dst}")
    elif cs == "Auto / Linear (pass-through)":
        gamut = ctx.get("file_gamut")
        if gamut and _enc.LINEAR_FOR_GAMUT.get(gamut) and _enc.LINEAR_FOR_GAMUT[gamut] != working:
            out, how = _enc.decode(rgb, _enc.LINEAR_FOR_GAMUT[gamut], working)
            _note(ctx, f"EXR chromaticities say {gamut}: {how}")
        else:
            out = rgb
            _note(ctx, "pass-through (no transform)")
    else:
        out, how = _enc.decode(rgb, cs, working, ctx["reference_white_nits"])
        _note(ctx, how)
    out = np.asarray(out, np.float32)
    return np.concatenate([out, extra], axis=-1) if extra is not None else out


# ── Path type detection ────────────────────────────────────────────────────

def _path_kind(path: str) -> str:
    """Classify a path: "image" | "exr" | "video" | "sequence" | "unknown".

    One owner, in core.formats, so the browse filter, this detector and the
    readers cannot disagree about what a given extension is.
    """
    return _formats.classify(path)


def _explain_unknown(path: str) -> str:
    """Why a path was refused, and what would make it work."""
    return _formats.explain_unsupported(path)


# ── Image read ────────────────────────────────────────────────────────────

def _is_16bit_rgb_source(path: str, ext: str) -> bool:
    """Cheaply detect a genuine 16-bit-per-channel RGB(A) PNG/TIFF source,
    without a full pixel decode. Returns False (safe default -- existing
    Pillow path) if detection isn't possible or the source isn't 16-bit RGB(A).
    """
    try:
        if ext == ".png":
            # PNG IHDR chunk has a fixed layout: 8-byte signature + 4-byte
            # length + 4-byte "IHDR" + 4-byte width + 4-byte height + 1-byte
            # bit depth + 1-byte color type (2=RGB, 6=RGBA) -- no decode needed.
            with open(path, "rb") as f:
                header = f.read(26)
            if len(header) < 26 or header[:8] != b"\x89PNG\r\n\x1a\n":
                return False
            bit_depth, color_type = header[24], header[25]
            # 0 grey, 2 RGB, 4 grey+alpha, 6 RGBA. Grey was excluded, and
            # Pillow's I;16 -> RGB convert saturates at 255, so a 16-bit
            # depth map or matte read back almost entirely white.
            return bit_depth == 16 and color_type in (0, 2, 4, 6)
        if ext in (".tif", ".tiff"):
            import tifffile  # type: ignore
            with tifffile.TiffFile(path) as tf:
                page = tf.pages[0]
                return page.dtype == np.uint16 and page.samplesperpixel in (1, 2, 3, 4)
    except ImportError:
        # AUDIT-FIX (2026-08): without tifffile a deep TIFF silently fell
        # through to Pillow's 8-bit path -- a 16-bit plate read back crushed
        # to 8-bit precision with no indication anywhere. Warn, loudly.
        log.warning(
            "[Radiance] Cannot probe %s for 16-bit depth: tifffile is not "
            "installed (pip install tifffile). If this file is 16-bit it will "
            "be read at 8-bit precision.", os.path.basename(path))
    except Exception as _exc:
        log.debug(
            "[Radiance] _is_16bit_rgb_source(): ignoring %s from `if ext == '.png':`: %s",
            type(_exc).__name__, _exc,
        )
    return False


def _is_float32_tiff(path: str) -> bool:
    """Cheaply detect a 32-bit float TIFF via tifffile page metadata (no
    full pixel decode) -- Pillow's standard build cannot open this sample
    format at all."""
    try:
        import tifffile  # type: ignore
        with tifffile.TiffFile(path) as tf:
            # Half, single and double float (half is what Nuke and Photoshop
            # write by default; only 32-bit was recognised).
            return np.dtype(tf.pages[0].dtype).kind == "f"
    except ImportError:
        # AUDIT-FIX (2026-08): see _is_16bit_rgb_source -- a float TIFF read
        # without tifffile ends up in Pillow, which either fails or reads it
        # at 8-bit; either way the user should know why.
        log.warning(
            "[Radiance] Cannot probe %s for 32-bit float depth: tifffile is "
            "not installed (pip install tifffile).", os.path.basename(path))
        return False
    except Exception:
        return False


def _read_image(path: str) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Return (IMAGE, MASK) tensors from a single image file."""
    ext = Path(path).suffix.lower()

    if ext == ".exr":
        return _read_exr_single(path)

    if ext == ".hdr":
        import cv2  # type: ignore
        arr = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        if arr is None:
            raise RuntimeError(f"Cannot read HDR file '{path}' (unreadable, truncated, or unsupported).")
        # Radiance .hdr is scene-linear float; keep values as-is (may exceed 1.0).
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB).astype(np.float32)
        return _np_to_tensor(arr), None

    if ext == ".dpx":
        # ALBABIT-FIX: Pillow has no DPX plugin at all; use OpenImageIO, the
        # VFX-industry-standard library for this format.
        if not _HAS_OIIO:
            raise ImportError("Reading DPX requires OpenImageIO (pip install OpenImageIO).")
        inp = _oiio.ImageInput.open(path)
        if inp is None:
            raise RuntimeError(f"Cannot read DPX '{path}': {_oiio.geterror()}")
        try:
            spec = inp.spec()
            pixels = inp.read_image(format=_oiio.FLOAT)
        finally:
            inp.close()
        # read_image() auto-normalises integer DPX samples (e.g. 10-bit) to [0, 1] float.
        arr = np.array(pixels, dtype=np.float32).reshape(spec.height, spec.width, spec.nchannels)
        alpha = arr[..., 3] if spec.nchannels >= 4 else None     # was dropped
        arr = arr[..., :3] if spec.nchannels >= 3 else np.repeat(arr[..., :1], 3, axis=-1)
        return _np_to_tensor(arr), (_np_to_tensor(alpha) if alpha is not None else None)

    # ALBABIT-FIX: a genuine 16-bit-per-channel RGB(A) PNG/TIFF is silently
    # collapsed to 8-bit by Pillow's .convert("RGB"/"RGBA") below -- Pillow has
    # no internal 16-bit RGB mode, and pil.mode reports "RGB" regardless of
    # source depth. cv2 preserves full precision instead. Grayscale 16-bit is
    # untouched -- Pillow already preserves that losslessly via "I"/"I;16".
    if ext in (".png", ".tif", ".tiff") and _is_16bit_rgb_source(path, ext):
        import cv2  # type: ignore
        arr16 = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if arr16 is not None and arr16.dtype == np.uint16:
            if arr16.ndim == 2:
                arr16 = arr16[..., None]
            ch = arr16.shape[-1]
            if ch in (3, 4):
                arr16 = cv2.cvtColor(arr16, cv2.COLOR_BGRA2RGBA if ch == 4 else cv2.COLOR_BGR2RGB)
            a16 = arr16.astype(np.float32) / 65535.0
            if ch in (1, 2):                       # grey, grey + alpha
                rgb16 = np.repeat(a16[..., :1], 3, axis=-1)
                mask16 = a16[..., 1] if ch == 2 else None
            else:
                rgb16 = a16[..., :3]
                mask16 = a16[..., 3] if ch == 4 else None
            return _np_to_tensor(rgb16), (_np_to_tensor(mask16) if mask16 is not None else None)
        # else: fall through to Pillow below -- defensive, shouldn't normally
        # happen since _is_16bit_rgb_source() already confirmed 16-bit.

    # ALBABIT-FIX: a 32-bit float TIFF (RadianceWrite's own "TIFF (32-bit
    # float)" output) can't be opened by Pillow at all -- caught by
    # RadianceRead.read()'s outer handler and surfaced as a silent black
    # image. tifffile (already used to write this format) reads it back correctly.
    if ext in (".tif", ".tiff") and _is_float32_tiff(path):
        import tifffile  # type: ignore
        arr = tifffile.imread(path).astype(np.float32)
        if arr.ndim == 2:
            arr = arr[..., np.newaxis]
        if arr.shape[-1] in (1, 2):
            mask32 = arr[..., 1] if arr.shape[-1] == 2 else None
            arr = np.repeat(arr[..., :1], 3, axis=-1)
        else:
            mask32 = arr[..., 3] if arr.shape[-1] == 4 else None
        rgb32 = arr[..., :3]
        return _np_to_tensor(rgb32), (_np_to_tensor(mask32) if mask32 is not None else None)

    if not _HAS_PIL:
        raise ImportError("Pillow is required to read image files.")
    pil = _PIL.open(path)
    has_alpha = pil.mode in ("RGBA", "LA", "PA")
    pil_rgb = pil.convert("RGBA") if has_alpha else pil.convert("RGB")
    arr = np.array(pil_rgb, dtype=np.float32)
    # ALBABIT-FIX: maxv used to be chosen from `pil` (pre-convert -- still the
    # source bit depth, e.g. 65535 for a 16-bit grayscale "I;16" source) but
    # applied to `arr`, which comes from `pil_rgb` (post-`.convert("RGB"/
    # "RGBA")`, always 8-bit -- Pillow's RGB/RGBA convert targets are always
    # 8-bit per channel, there's no 16-bit RGB convert mode). Dividing 8-bit
    # data by 65535 crushed 16-bit grayscale sources (mattes, depth passes)
    # to ~1/256th of their real brightness, silently. `arr` is always 8-bit
    # range here, so maxv is always 255.
    rgb  = arr[..., :3] / 255.0
    mask = (arr[..., 3:4] / 255.0) if has_alpha else None
    img_t  = _np_to_tensor(rgb)
    mask_t = _np_to_tensor(mask[..., 0]) if mask is not None else None
    return img_t, mask_t


def _read_exr_single(
    path: str,
    layer: Optional[str] = None,
    raw: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Read one layer of an EXR, honouring the display window.

    Delegated to :mod:`radiance.core.exr`. What used to be here pulled R, G, B
    and A out of part 0 and raised on anything else -- so a multi-layer AOV
    render, which is the normal output of Nuke and Arnold, was rejected with a
    message blaming the file. It also read the data window and reshaped to it,
    so an overscan render came back offset and at the wrong resolution with no
    warning.
    """
    from ..core import exr as _exr

    rgb, alpha, _info, _name = _exr.read_layer(path, layer, raw=raw)
    mask = _np_to_tensor(alpha) if alpha is not None else None
    return _np_to_tensor(rgb), mask


def _set_file_gamut(attributes) -> None:
    """Record the gamut an EXR's `chromaticities` attribute declares."""
    from ..color.encodings import gamut_from_chromaticities
    ctx = _COLOUR.get()
    if ctx is not None:
        ctx["file_gamut"] = gamut_from_chromaticities((attributes or {}).get("chromaticities"))


def _read_exr_with_info(path: str, layer: Optional[str] = None, raw: bool = False):
    """As above, but also hands back the probe so the node can report it."""
    from ..core import exr as _exr

    rgb, alpha, info, name = _exr.read_layer(path, layer, raw=raw)
    mask = _np_to_tensor(alpha) if alpha is not None else None
    return _np_to_tensor(rgb), mask, info, name


# ── Sequence read ─────────────────────────────────────────────────────────

def _trailing_frame_number(path: str) -> Optional[int]:
    """Last run of digits in the stem: shot_v002.1042 → 1042, img007 → 7."""
    m = re.search(r"(\d+)\D*$", Path(path).stem)
    return int(m.group(1)) if m else None


def _window_listed_files(
    files: List[str], start: int, end: int, step: int
) -> List[str]:
    """Apply a start/end/step frame window to an explicit file list.

    start/end are FRAME NUMBERS (widget semantics, default 1001..end-of-shot),
    not list indices. When the filenames carry frame numbers, the window is
    reconciled against the range on disk the same way the pattern path does:
    a window that does not intersect the files falls back to the full range
    rather than returning nothing. Files without frame numbers are returned
    positionally with only the step applied.
    """
    if not files:
        return files
    step = max(1, step)
    nums = [_trailing_frame_number(f) for f in files]
    if any(n is None for n in nums) or nums != sorted(nums):
        # No consistent numbering: positional, step only.
        return files[::step]
    first, last = nums[0], nums[-1]
    s = start if first <= start <= last else first
    e = end if s <= end <= last else last
    return [f for f, n in zip(files, nums) if s <= n <= e and (n - s) % step == 0]


def _resolve_sequence_paths(
    pattern: str,
    start: int,
    end: int,
    step: int = 1,
    missing_frames: str = "Skip",
) -> List[str]:
    """
    Expand a sequence pattern to a list of existing file paths.

    Supported patterns:
      /path/frame.%04d.exr
      /path/frame.####.png
      /path/frame.*.png     → sorted glob
      /path/               → directory: sorted image files
    """
    # AUDIT-FIX (2026-08): directory and glob inputs used to slice the sorted
    # file list by LIST INDEX with the widget's frame-number defaults --
    # files[1001:99999] -- so every directory read of a normal-length sequence
    # returned an empty list and raised "No frames found". Frame numbers are
    # now parsed from the filenames and start/end are reconciled against the
    # range actually on disk, matching the %04d/#### pattern path.
    if os.path.isdir(pattern):
        exts = list(_IMG_EXT | _EXR_EXT)
        files = sorted(
            str(f) for f in Path(pattern).iterdir()
            if f.suffix.lower() in exts
        )
        return _window_listed_files(files, start, end, step)

    if "*" in pattern:
        import glob
        return _window_listed_files(sorted(glob.glob(pattern)), start, end, step)

    # Hash style #### → %04d
    hash_match = re.search(r"(#+)", pattern)
    if hash_match:
        hashes = hash_match.group(1)
        pattern = pattern.replace(hashes, f"%0{len(hashes)}d")

    # Probe the directory once to learn which frame numbers actually exist.
    # Walking the caller's upper bound blindly is not viable: "read all frames"
    # arrives here as end=99999, which expanded a 5-frame sequence into ~99k
    # slots, 98,994 of them phantoms, and blew up the downstream torch.cat.
    available: set = set()
    pad_match = re.search(r"%0(\d+)d", pattern)
    if pad_match:
        import glob as _glob
        token = pad_match.group(0)
        prefix, suffix = pattern.split(token, 1)
        for hit in _glob.glob(pattern.replace(token, "[0-9]" * int(pad_match.group(1)))):
            if hit.startswith(prefix) and hit.endswith(suffix):
                num = hit[len(prefix):len(hit) - len(suffix)] if suffix else hit[len(prefix):]
                if num.isdigit():
                    available.add(int(num))

    if available:
        hi = min(end, max(available)) if end >= start else max(available)
        frames = [f for f in range(start, hi + 1, step)]
        # Requested window misses the sequence entirely (e.g. the default
        # start_frame=1001 against a sequence numbered from 1) -- fall back to
        # everything on disk rather than reporting a range full of holes.
        if not any(f in available for f in frames):
            frames = sorted(available)[::step]
    else:
        frames = list(range(start, end + 1, step)) if end >= start else [start]

    paths  = []
    missing = []
    for f in frames:
        p = pattern % f
        if os.path.isfile(p):
            paths.append(p)
        else:
            missing.append(p)
            if missing_frames == "Black":
                paths.append(p)   # keep slot for black-frame insertion
            else:
                # "Skip" means skip: keeping the slot here inserted an 8x8
                # black tile that then failed to concatenate with real frames.
                log.debug("Frame not found: %s", p)

    if missing_frames == "Error" and missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} frame(s), e.g. {missing[0]}"
        )
    return paths


def _read_one_sequence_frame(
    path: str,
    input_cs: str,
    layer: Optional[str],
    raw: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """One frame of a sequence as (1,H,W,C) IMAGE and its alpha, or None."""
    if os.path.splitext(path)[1].lower() in _EXR_EXT:
        img_t, mask_t, _info, _name = _read_exr_with_info(path, layer, raw)
        _set_file_gamut(_info.attributes)
    else:
        img_t, mask_t = _read_image(path)
    arr = _tensor_to_np(img_t)
    if not raw:
        arr = _apply_input_colorspace(arr, input_cs)
    return torch.from_numpy(arr).unsqueeze(0), mask_t


def iter_sequence_frames(
    pattern: str,
    start: int,
    end: int,
    step: int = 1,
    input_cs: str = "Auto / Linear (pass-through)",
    missing_frames: str = "Skip",
    layer: Optional[str] = None,
    raw: bool = False,
) -> Iterator[Tuple[str, torch.Tensor, Optional[torch.Tensor]]]:
    """Yield ``(path, image, alpha)`` one frame at a time, in frame order.

    The batch-returning `_read_sequence` below is built on this and is what the
    node layer calls, because a ComfyUI node genuinely does hand the graph a
    batch. A read-transform-write pipeline does not: it wants a frame, a
    transform and a write, and then the next frame. Consuming this generator
    means the pipeline holds a working window rather than the whole shot, which
    is the difference between a sequence bounded by RAM and one bounded by
    disk.

    A frame that is not on disk yields ``image=None`` so the caller can decide
    what a hole means; `_read_sequence` fills it with black, as it always did.
    """
    paths = _resolve_sequence_paths(pattern, start, end, step, missing_frames)
    if not paths:
        raise FileNotFoundError(f"No frames found for pattern: {pattern}")
    for p in paths:
        if os.path.isfile(p):
            img_t, mask_t = _read_one_sequence_frame(p, input_cs, layer, raw)
            yield p, img_t, mask_t
        else:
            yield p, None, None


def _read_sequence(
    pattern: str,
    start: int,
    end: int,
    step: int,
    input_cs: str,
    missing_frames: str = "Skip",
    layer: Optional[str] = None,
    raw: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], int, int, int, float, str]:
    """Read a frame sequence into a batched IMAGE tensor, keeping its alpha.

    The alpha used to be discarded here -- ``img_t, _ = _read_image(p)`` -- so a
    sequence of RGBA PNGs or EXRs came back with an empty mask, exactly like the
    ProRes 4444 case. A sequence with a matte is the other half of how plates
    arrive.

    This returns the whole batch on purpose: it is what the node hands ComfyUI.
    Anything that does not need the whole shot at once should consume
    `iter_sequence_frames` instead of this.
    """
    paths = _resolve_sequence_paths(pattern, start, end, step, missing_frames)
    if not paths:
        raise FileNotFoundError(f"No frames found for pattern: {pattern}")

    frames: List[Optional[torch.Tensor]] = []
    masks: List[Optional[torch.Tensor]] = []
    blank_slots: List[int] = []
    missing_paths: List[str] = []
    for p in paths:
        if os.path.isfile(p):
            img_t, mask_t = _read_one_sequence_frame(p, input_cs, layer, raw)
            frames.append(img_t)
            masks.append(mask_t)
        else:
            # Placeholder for a missing frame. Size it from a real frame below
            # -- a fixed 8x8 tile cannot concatenate with the rest of the batch.
            blank_slots.append(len(frames))
            missing_paths.append(p)
            frames.append(None)
            masks.append(None)

    real = next((f for f in frames if f is not None), None)
    if real is None:
        raise FileNotFoundError(f"No readable frames found for pattern: {pattern}")
    for i in blank_slots:
        frames[i] = torch.zeros_like(real)
    if missing_paths:
        log.warning(
            "[Radiance/Read] %d frame(s) of %s are not on disk and were filled "
            "with black: %s%s",
            len(missing_paths), pattern,
            ", ".join(os.path.basename(m) for m in missing_paths[:6]),
            " ..." if len(missing_paths) > 6 else "",
        )

    batch = torch.cat(frames, dim=0)   # (N, H, W, C)
    _, h, w, _ = batch.shape

    # An alpha only survives if every frame that has one agrees on the shape.
    # A sequence where half the frames carry a matte is a mistake worth saying
    # out loud rather than papering over.
    with_alpha = [m for m in masks if m is not None]
    alpha_out = None
    if with_alpha:
        if len(with_alpha) != len(masks):
            log.warning(
                "[Radiance/Read] %d of %d frames in %s carry an alpha channel; "
                "the missing ones are opaque in the mask output.",
                len(with_alpha), len(masks), pattern,
            )
        filled = [
            m if m is not None else torch.ones(1, h, w, dtype=with_alpha[0].dtype)
            for m in masks
        ]
        try:
            alpha_out = torch.cat(filled, dim=0)
        except RuntimeError as exc:
            log.warning("[Radiance/Read] alpha channels in %s do not share a "
                        "shape (%s); mask output left empty.", pattern, exc)
            alpha_out = None

    meta = json.dumps({
        "kind": "sequence",
        "pattern": pattern,
        "frame_count": len(frames),
        "missing": [os.path.basename(m) for m in missing_paths],
        "width": w, "height": h,
        "alpha": alpha_out is not None,
    })
    return batch, alpha_out, w, h, len(frames), 24.0, meta


# ── Video read ────────────────────────────────────────────────────────────

def _read_video(
    path: str,
    max_frames: int,
    input_cs: str,
    start_frame: int = 0,
    end_frame: int = 0,
    frame_step: int = 1,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], float, int, int, int, str]:
    """Decode a video file to a batched IMAGE tensor, plus its alpha if it has one.

    Everything here is delegated to :mod:`radiance.core.video`, which pipes raw
    frames from ffmpeg at the source bit depth. The old implementation in this
    slot wrote every frame to a temporary PNG and read them back: measured on a
    4-second 1080p ProRes 422 HQ clip, 25.0 s and ~600 MB of scratch files
    against 12.9 s and none. It also had no way to express a frame range, and it
    silently dropped the alpha channel of every ProRes 4444.

    Returns
    -------
    (images, alpha, fps, width, height, frame_count, metadata_json)
        ``alpha`` is None unless the file actually carries an alpha channel.
    """
    from ..core import video as _video

    info = _video.probe(path)

    # A frame range on the *source* numbering. `end_frame` is inclusive, and 0
    # means "to the end", which is how the sequence reader already behaves.
    start = max(int(start_frame), 0)
    if end_frame and end_frame >= start:
        count = (int(end_frame) - start) // max(int(frame_step), 1) + 1
    else:
        count = 0
    if max_frames > 0:
        count = min(count, max_frames) if count else max_frames

    log.info("[Radiance/Read] %s: %s", os.path.basename(path), info.summary())
    if info.frames_estimated:
        log.debug(
            "%s carries no frame count; %d is estimated from duration x fps.",
            os.path.basename(path), info.frames,
        )

    arr, info = _video.decode(
        path, start=start, count=count, step=max(int(frame_step), 1), info=info,
    )

    # Split alpha off before the transfer decode. A matte is not light: running
    # it through a log or gamma curve is the bug that made a 3.1.x tone map
    # return an opaque pixel at 0.730.
    alpha_np = None
    if arr.shape[-1] == 4:
        alpha_np = np.ascontiguousarray(arr[..., 3])
        arr = np.ascontiguousarray(arr[..., :3])

    resolved_cs = _resolve_video_colorspace(input_cs, info, path)
    arr = _apply_input_colorspace(arr, resolved_cs)

    batch = torch.from_numpy(np.ascontiguousarray(arr))
    alpha_t = torch.from_numpy(alpha_np) if alpha_np is not None else None

    n, h, w, _ = batch.shape
    meta = dict(info.as_dict())
    meta.update({
        "kind": "video",
        "decoded_frames": n,
        "start_frame": start,
        "frame_step": max(int(frame_step), 1),
        "color_space": resolved_cs,
        "color_space_requested": input_cs,
        "alpha": alpha_t is not None,
    })
    return batch, alpha_t, info.fps_float, w, h, n, json.dumps(meta)


#: The sequence-style default for `start_frame`. VFX sequences are numbered
#: from 1001; a clip is numbered from 0, so the same widget means two things.
_SEQUENCE_START_DEFAULT = 1001


def _video_frame_range(path: str, start_frame: int, end_frame: int) -> Tuple[int, int]:
    """Translate the sequence-style frame widgets onto a clip's own numbering.

    `start_frame` defaults to 1001 because that is where a VFX image sequence
    starts. A video's frames are numbered from zero, so taking the widget
    literally would skip the first 1001 frames of every clip -- which, for
    anything under about 40 seconds, means decoding nothing at all and
    reporting an empty file. That is exactly the trap the directory-pattern
    reader fell into.

    So: a start beyond the end of the clip is treated as "the user never
    touched this widget", and said out loud. A start inside the clip is
    honoured, because then it is a deliberate trim.
    """
    from ..core import video as _video

    start = max(int(start_frame), 0)
    end = max(int(end_frame), 0)
    if start <= 0:
        return 0, end

    try:
        info = _video.probe(path)
    except Exception:  # the decoder will produce the real error in a moment
        return 0, end

    if info.frames > 0 and start >= info.frames:
        if start == _SEQUENCE_START_DEFAULT:
            log.debug(
                "start_frame is at its sequence default (%d) and %s has only %d "
                "frame(s), so the whole clip is being read.",
                _SEQUENCE_START_DEFAULT, os.path.basename(path), info.frames,
            )
        else:
            log.warning(
                "[Radiance/Read] start_frame=%d is past the end of %s, which has "
                "%d frame(s). Reading the whole clip instead. Video frames are "
                "numbered from 0, unlike an image sequence.",
                start, os.path.basename(path), info.frames,
            )
        return 0, end
    return start, end


def _resolve_video_colorspace(requested: str, info, path: str) -> str:
    """Pick the decode curve for a video, honouring the file's own tags.

    Nuke, Resolve and RV all read the container's transfer characteristics and
    default to them. Radiance did not read them at all, so a Rec.709 delivery
    was passed through as if it were scene-linear -- the values are gamma
    encoded, so every subsequent exposure, blur and blend was operating on the
    wrong numbers.

    An explicit choice always wins. "Auto" now means what its label promises:
    use the tag if there is one, and say so. An untagged file still passes
    through unchanged, but now says that too, because a silent pass-through is
    exactly how the mistake goes unnoticed.
    """
    from ..core import video as _video

    if requested != "Auto / Linear (pass-through)":
        tagged = _video.suggest_transfer(info)
        if tagged and tagged != requested:
            log.warning(
                "[Radiance/Read] %s is tagged %s (%s), but color_space is set to "
                "%r. Using your setting. Clear it to Auto to follow the file.",
                os.path.basename(path), info.color_transfer, tagged, requested,
            )
        return requested

    suggestion = _video.suggest_transfer(info)
    if suggestion and suggestion != "Auto / Linear (pass-through)":
        log.info(
            "[Radiance/Read] %s is tagged color_trc=%s; decoding it as %s. "
            "Set color_space explicitly to override.",
            os.path.basename(path), info.color_transfer, suggestion,
        )
        return suggestion

    if info.color_transfer:
        log.warning(
            "[Radiance/Read] %s is tagged color_trc=%s, which Radiance has no "
            "inverse for. Passing the values through unchanged -- they are not "
            "scene-linear. Set color_space by hand.",
            os.path.basename(path), info.color_transfer,
        )
    else:
        log.warning(
            "[Radiance/Read] %s carries no colour tags, so its values are being "
            "passed through as if they were already scene-linear. If it is a "
            "Rec.709 delivery or a camera log file, set color_space -- exposure "
            "and blur are only correct in a linear space.",
            os.path.basename(path),
        )
    return requested


# ── Write input coercion ──────────────────────────────────────────────────

def _find_video_path(data: Any, _depth: int = 0) -> Optional[str]:
    """
    Recursively search dict / list / str structures for a video file path.
    Handles VHS, AnimateDiff, and other video-helper dict formats.
    """
    if _depth > 6:
        return None
    if isinstance(data, str):
        if os.path.isfile(data) and Path(data).suffix.lower() in _VID_EXT:
            return data
        return None
    if isinstance(data, dict):
        # Prefer explicit path keys used by common video helper nodes
        for key in ("video_path", "video", "path", "filename", "file", "filepath"):
            if key in data:
                r = _find_video_path(data[key], _depth + 1)
                if r:
                    return r
        for val in data.values():
            r = _find_video_path(val, _depth + 1)
            if r:
                return r
    if isinstance(data, (list, tuple)):
        for item in data:
            r = _find_video_path(item, _depth + 1)
            if r:
                return r
    return None


def _load_video_to_numpy(
    path: str, max_frames: int = 0, *, alpha: bool = False
) -> np.ndarray:
    """Decode a video file path to (N, H, W, C) float32 at the source bit depth.

    This used to try OpenCV first, which was wrong twice over. OpenCV's
    ``VideoCapture`` hands back 8-bit BGR whatever the source is -- a 12-bit
    ProRes 4444 came out on an exact 1/255 grid, measured -- and when it failed
    part-way through a clip the loop simply stopped, returning a short clip with
    no error. A shot that quietly ends early is the kind of bug that survives
    all the way to a review session.

    Alpha is off by default here because the callers of this helper feed
    DCC handoff paths that expect three channels; pass ``alpha=True`` to keep
    the fourth when the file has one.
    """
    from ..core import video as _video

    arr, _info = _video.decode(path, count=max(int(max_frames), 0), alpha=alpha)
    return arr

def _sequence_frame_range(detected, start_frame: int, end_frame: int):
    """Reconcile the widget values with the range that is actually on disk.

    `start_frame` defaults to 1001, which is right for a VFX sequence and wrong
    for everything else. A start outside the range that exists is treated as
    "not set" -- otherwise a sequence numbered from 1 reads nothing, which is
    the documented directory-pattern trap in a new costume.

    ALBABIT-FIX: when the start was corrected, the end was left where it was,
    which threw away the LENGTH the caller asked for. `RadianceDigitalCinemaRead`
    hits this every time: its own `start_frame` defaults to 1, so a 10-frame
    read arrives as start=1, end=10, the start is rewritten to 1001 for a
    sequence numbered from 1001, `end < start`, and `_resolve_sequence_paths`
    falls through to reading the entire sequence. Measured: start_frame=1,
    frame_limit=10 on a 100-frame sequence read all 100 frames into RAM. A
    window is a start and a length; correcting the start has to carry the
    length with it.
    """
    start = start_frame
    span = (end_frame - start_frame + 1) if end_frame >= start_frame > 0 else 0
    if not (detected.first <= start <= detected.last):
        if start not in (0, 1001):
            log.warning(
                "[Radiance/Read] start_frame=%d is outside the range on disk "
                "(%d-%d); reading from %d instead.",
                start, detected.first, detected.last, detected.first,
            )
        start = detected.first
        if span > 0:
            end_frame = start + span - 1
    end = end_frame
    if end <= 0 or end > detected.last:
        end = detected.last
    return start, end


def _unpremultiply(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Divide RGB by alpha, leaving fully transparent pixels alone.

    EXR conventionally stores associated (premultiplied) alpha; ComfyUI's MASK
    is straight by convention. Dividing by zero where alpha is zero would turn
    the transparent region into NaN and poison every downstream operation, so
    those pixels keep their stored value.
    """
    alpha = mask.unsqueeze(-1).to(image.dtype)
    safe = torch.where(alpha > 1e-6, alpha, torch.ones_like(alpha))
    out = torch.where(alpha > 1e-6, image / safe, image)
    return out


def _describe_read(info: dict) -> str:
    """One console line, in the order a Nuke Read's info bar reads."""
    bits = [os.path.basename(str(info.get("path", "")) or "?")]
    if info.get("width"):
        bits.append(f"{info['width']}x{info['height']}")
    frames = info.get("frames", 0)
    if frames and frames > 1:
        bits.append(f"{frames} frames")
    for key in ("kind", "codec", "layer"):
        if info.get(key):
            bits.append(str(info[key]))
    if info.get("bit_depth"):
        bits.append(f"{info['bit_depth']}-bit")
    if info.get("alpha"):
        bits.append("alpha")
    if info.get("overscan"):
        bits.append("overscan conformed")
    if info.get("unpremultiplied"):
        bits.append("unpremultiplied")
    cs = info.get("color_space")
    if cs:
        bits.append(str(cs))
    return " · ".join(bits)



def _probe_for_ui(path: str) -> Dict[str, Any]:
    """A short, cheap description of a file for the node's info line."""
    kind = _path_kind(path)
    out: Dict[str, Any] = {"kind": kind, "name": os.path.basename(path)}

    if kind == "video":
        from ..core import video as _video

        info = _video.probe(path)
        out.update(info.as_dict())
        out["summary"] = info.summary()
        return out

    if kind == "exr":
        from ..core import exr as _exr

        info = _exr.probe(path)
        out.update({
            "width": info.width, "height": info.height,
            "display_width": info.display_width,
            "display_height": info.display_height,
            "overscan": info.has_overscan, "parts": info.parts,
            "layers": info.layer_names, "compression": info.compression,
            "summary": info.summary(),
        })

    sequence = _formats.detect_sequence(path)
    if sequence is not None:
        out.update({
            "sequence": sequence.pattern,
            "first": sequence.first, "last": sequence.last,
            "frames": sequence.count, "missing": len(sequence.missing),
            "summary": (out.get("summary", "") + " · " + sequence.summary()).strip(" ·"),
        })
    if "summary" not in out:
        try:
            image, mask = _read_image(path)
            _, h, w, _ = image.shape
            out.update({"width": w, "height": h, "alpha": mask is not None,
                        "summary": f"{w}x{h}" + (" · alpha" if mask is not None else "")})
        except Exception as exc:
            out["summary"] = f"unreadable: {exc}"
    return out

def read_frames(
    *,
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
    working_space: str = "Linear Rec.709 (sRGB)",
    ocio_colorspace: str = "",
    ocio_config: str = "",
    hdr_reference_nits: float = 203.0,
):
    # Keyword-only. The parameter order below is the node's widget order,
    # which is a contract saved workflows are matched against -- so it cannot
    # be reordered to put `path` first, where a reader would expect it. Rather
    # than leave `read_frames("/plate.exr")` quietly setting `media_type`, the
    # positional form is simply not available.
    #
    # `browse` is the node's: resolving a ComfyUI input-directory filename
    # needs folder_paths, which is exactly the kind of thing that does not
    # belong down here. The node resolves it and hands over a path.
    path = strip_path_quotes(path or "")

    if not path:
        # Not a failure: a node just dropped on the canvas has no file yet.
        log.debug("RadianceRead: no path set yet")
        blank = torch.zeros(1, 8, 8, 3)
        return (blank, blank[..., 0], {"kind": "empty"})

    from ..color.encodings import WORKING_SPACES as _WS
    if working_space not in _WS:
        raise ValueError(f"RadianceRead: unknown working_space {working_space!r}. Known: {', '.join(_WS)}")
    ctx = {"working": working_space, "reference_white_nits": float(hdr_reference_nits or 203.0),
           "ocio_colorspace": (ocio_colorspace or "").strip(), "ocio_config": (ocio_config or "").strip(),
           "file_gamut": None, "applied": []}
    token = _COLOUR.set(ctx)
    try:
        image, mask, info = _read_resolved(
            path, media_type, color_space, start_frame, end_frame,
            frame_step, max_video_frames, missing_frames, layer, raw,
        )
    except Exception as exc:
        log.error("RadianceRead: %s: %s", type(exc).__name__, exc)
        if on_error == "Error":
            # 3.1.x caught everything here and handed back an 8x8 black
            # frame, so a missing plate, a corrupt MOV and a wrong layer
            # name all arrived downstream as black with the node still
            # green -- and the graph carried on and wrote a master out of
            # it. Raising is what Nuke, Resolve and RV all do.
            raise RuntimeError(f"RadianceRead failed on {path!r}: {exc}") from exc
        log.warning(
            "[Radiance/Read] on_error is 'Black frame', so a black 8x8 "
            "frame is being substituted for %s. Nothing downstream can "
            "tell this apart from a genuinely black plate.", path,
        )
        blank = torch.zeros(1, 8, 8, 3)
        return (blank, blank[..., 0],
                {"kind": "error", "path": path, "error": str(exc)})
    finally:
        _COLOUR.reset(token)

    # ── Shared post-processing ────────────────────────────────────────
    if premultiplied and mask is not None and float(mask.abs().max()) > 0:
        image = _unpremultiply(image, mask)
        info["unpremultiplied"] = True

    if proxy_scale > 0:
        image = torch.nn.functional.interpolate(
            image.movedim(-1, 1), scale_factor=proxy_scale, mode="bilinear",
        ).movedim(1, -1)
        if mask is not None:
            mask = torch.nn.functional.interpolate(
                mask.unsqueeze(1), scale_factor=proxy_scale, mode="bilinear",
            ).squeeze(1)
        info["proxy_scale"] = proxy_scale

    n, h, w, _ = image.shape
    info.update({"frames": n, "width": w, "height": h,
                 "alpha": mask is not None,
                 "color_space": "raw (untransformed)" if raw else color_space,
                 "working_space": "file values (raw)" if raw else working_space,
                 "colour_transform": [] if raw else ctx["applied"]})
    if ctx.get("ocio_colorspace") and not raw:
        info["ocio_colorspace"] = ctx["ocio_colorspace"]
    if mask is None:
        mask = torch.zeros(n, h, w)

    log.info("[Radiance/Read] %s", _describe_read(info))
    return (image, mask, info)

def _read_resolved(
    path, media_type, color_space, start_frame, end_frame,
    frame_step, max_video_frames, missing_frames, layer, raw,
):
    """Return (image, mask_or_None, info_dict). Raises on any failure."""
    kind = _path_kind(path) if media_type == "Auto" else media_type.lower()

    # Nuke opens shot.1001.exr and offers you the whole sequence. Radiance
    # read one frame and left you to type a %04d pattern by hand. If the
    # picked file has numbered siblings, this is a sequence.
    if kind in ("image", "exr") and media_type == "Auto":
        detected = _formats.detect_sequence(path)
        if detected is not None:
            log.info(
                "[Radiance/Read] %s is one frame of a sequence: %s. Reading "
                "the whole range. Set media_type to Image for just this frame.",
                os.path.basename(path), detected.summary(),
            )
            path, kind = detected.pattern, "sequence"
            start_frame, end_frame = _sequence_frame_range(
                detected, start_frame, end_frame)

    if kind == "image" and os.path.splitext(path)[1].lower() in _EXR_EXT:
        kind = "exr"     # media_type Image on an EXR: keep layers + chromaticities

    if kind == "image":
        image, mask = _read_image(path)
        if not raw:
            image = _np_to_tensor(
                _apply_input_colorspace(_tensor_to_np(image), color_space))
        return image, mask, {"kind": "image", "path": path}

    if kind == "exr":
        image, mask, exr_info, chosen = _read_exr_with_info(path, layer, raw)
        _set_file_gamut(exr_info.attributes)
        if not raw:
            image = _np_to_tensor(
                _apply_input_colorspace(_tensor_to_np(image), color_space))
        return image, mask, {
            "kind": "exr", "path": path, "layer": chosen,
            "layers": exr_info.layer_names, "parts": exr_info.parts,
            "compression": exr_info.compression,
            "data_window": [exr_info.width, exr_info.height],
            "display_window": [exr_info.display_width, exr_info.display_height],
            "overscan": exr_info.has_overscan,
            "attributes": exr_info.attributes or {},
        }

    if kind == "video":
        v_start, v_end = _video_frame_range(path, start_frame, end_frame)
        batch, alpha, _fps, _w, _h, _n, meta = _read_video(
            path, max_video_frames, "Auto / Linear (pass-through)" if raw
            else color_space,
            start_frame=v_start, end_frame=v_end, frame_step=frame_step,
        )
        return batch, alpha, json.loads(meta)

    if kind == "sequence":
        detected = _formats.describe_pattern(path)
        if detected is not None:
            start_frame, end_frame = _sequence_frame_range(
                detected, start_frame, end_frame)
        batch, alpha, _w, _h, _n, _fps, meta = _read_sequence(
            path, start_frame, end_frame if end_frame > 0 else 99999,
            frame_step, color_space, missing_frames, layer, raw,
        )
        info = json.loads(meta)
        if detected is not None:
            info["available_range"] = [detected.first, detected.last]
        return batch, alpha, info

    raise ValueError(f"RadianceRead cannot read {path!r}. {_explain_unknown(path)}")
