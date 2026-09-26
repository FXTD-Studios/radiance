"""What kind of media is this path, and can we actually open it?

One owner for the question. It used to be answered in three places that
disagreed: ``_IMG_EXT`` in ``nodes_io`` (nine extensions), ``_VID_EXT`` (seven),
and the browse-dropdown filter built from both. Anything outside those sixteen
strings was classified ``"unknown"`` and refused -- even though the reader
underneath opens it fine. Measured: TGA, SGI, PPM, PGM, JP2, PCX and ICO all
decode correctly through the existing code path and were all rejected before
they got there.

So the table here is built from what the installed backends actually register,
not from a list someone typed. Pillow publishes its own extension map; OpenImageIO
publishes its supported formats; EXR, DPX and the float-TIFF path are handled
specially because they need a backend Pillow does not provide. If a user installs
``pillow-heif`` or ``OpenImageIO`` tomorrow, the new formats appear without a code
change -- and if they don't have them, :func:`explain_unsupported` says which
package to install rather than "unknown path type".

Sequence detection
------------------
Nuke opens ``shot.1001.exr`` and offers you the whole sequence with its real
range. Radiance read one frame and moved on. :func:`detect_sequence` closes
that: given any frame of a sequence, it finds the siblings, works out the
padding, and returns a printf pattern plus the first and last frame that exist.
"""
from __future__ import annotations

import functools
import logging
import os
import re
from dataclasses import dataclass
from typing import Optional, Set, Tuple

from .video import VIDEO_EXTENSIONS

logger = logging.getLogger("radiance.formats")

#: Formats with a dedicated reader in nodes_io, independent of Pillow.
EXR_EXTENSIONS = frozenset({".exr", ".sxr", ".mxr"})

#: Needs OpenImageIO. Pillow has no plugin for any of these.
OIIO_ONLY_EXTENSIONS = frozenset({
    ".dpx", ".cin", ".ari", ".arw", ".cr2", ".cr3", ".nef", ".dng", ".raf",
    ".orf", ".rw2", ".iff", ".rla", ".zfile", ".ptex", ".heic", ".heif",
})

#: Read through OpenCV's float paths rather than Pillow.
FLOAT_EXTENSIONS = frozenset({".hdr", ".pic", ".pfm"})

#: Extensions Pillow will happily open but which are documents, not plates.
#: Reading page one of a PDF as a frame is a surprise, not a feature.
_PILLOW_EXCLUDED = frozenset({
    ".pdf", ".ps", ".eps", ".h5", ".hdf", ".grib", ".bufr", ".fit", ".fits",
    ".mpeg", ".mpg",  # these are video; VIDEO_EXTENSIONS owns them
})


@dataclass(frozen=True)
class SequenceInfo:
    """A numbered sequence found on disk, described the way Nuke describes one."""

    pattern: str          #: printf form, e.g. /plates/sh010.%04d.exr
    first: int
    last: int
    padding: int
    count: int            #: how many frames actually exist
    missing: Tuple[int, ...] = ()

    @property
    def is_contiguous(self) -> bool:
        return not self.missing

    def summary(self) -> str:
        span = f"{self.first}-{self.last}"
        if self.missing:
            return f"{self.count} frames, {span} ({len(self.missing)} missing)"
        return f"{self.count} frames, {span}"


@functools.lru_cache(maxsize=1)
def _pillow_extensions() -> frozenset:
    """Everything this Pillow build registered a reader for."""
    try:
        from PIL import Image

        Image.init()
        return frozenset(e.lower() for e in Image.EXTENSION) - _PILLOW_EXCLUDED
    except Exception as exc:  # pragma: no cover - Pillow is a hard dependency
        logger.debug("could not enumerate Pillow formats: %s", exc)
        return frozenset()


@functools.lru_cache(maxsize=1)
def _oiio_extensions() -> frozenset:
    """Everything an installed OpenImageIO can read.

    OIIO publishes ``extension_list`` as ``fmt:ext,ext;fmt:ext``. Parsing it
    beats hard-coding, because which formats a given OIIO build supports
    depends on how it was compiled.
    """
    try:
        import OpenImageIO as oiio
    except Exception:
        return frozenset()
    raw = ""
    for attr in ("extension_list", "get_string_attribute"):
        try:
            raw = (getattr(oiio, attr)("extension_list")
                   if attr == "get_string_attribute" else getattr(oiio, attr))
            if raw:
                break
        except Exception:
            continue
    found: Set[str] = set()
    for group in str(raw).split(";"):
        _, _, exts = group.partition(":")
        for ext in exts.split(","):
            ext = ext.strip().lower()
            if ext:
                found.add(ext if ext.startswith(".") else f".{ext}")
    return frozenset(found)


def image_extensions() -> frozenset:
    """Every still-image extension this install can actually open."""
    return (
        _pillow_extensions()
        | EXR_EXTENSIONS
        | FLOAT_EXTENSIONS
        | (OIIO_ONLY_EXTENSIONS & _oiio_extensions())
        | {".dpx"}  # always listed; the reader raises with an install hint
    ) - VIDEO_EXTENSIONS


def readable_extensions() -> frozenset:
    """Everything RadianceRead will attempt, stills and video together."""
    return image_extensions() | frozenset(VIDEO_EXTENSIONS)


# ── sequence patterns ──────────────────────────────────────────────────────

#: printf (%04d), hash (####), or a glob. Nuke accepts all three too.
_PRINTF_RE = re.compile(r"%0?\d*d")
_HASH_RE = re.compile(r"#+")
#: A frame number in a filename: shot.1001.exr, shot_1001.exr, shot1001.exr.
_FRAME_IN_NAME_RE = re.compile(r"^(?P<stem>.*?)(?P<sep>[._-]?)(?P<frame>\d{2,10})$")


def is_sequence_pattern(path: str) -> bool:
    """Does this path *say* it is a sequence, rather than merely being one frame?"""
    text = str(path).strip()
    return bool(_PRINTF_RE.search(text) or _HASH_RE.search(text) or "*" in text)


def classify(path: str) -> str:
    """``"image"`` | ``"exr"`` | ``"video"`` | ``"sequence"`` | ``"unknown"``."""
    text = str(path).strip()
    if not text:
        return "unknown"
    if is_sequence_pattern(text) or os.path.isdir(text):
        return "sequence"

    ext = os.path.splitext(text)[1].lower()
    if ext in EXR_EXTENSIONS:
        return "exr"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    if ext in image_extensions():
        return "image"
    return "unknown"


def explain_unsupported(path: str) -> str:
    """Why we will not open this, and what would make it work.

    "unknown path type for '/plates/sh010.ari'" tells a user nothing. Naming the
    package that would open it tells them everything.
    """
    ext = os.path.splitext(str(path))[1].lower()
    if not ext:
        return (
            f"{path!r} has no file extension, so there is nothing to detect from. "
            "Set media_type explicitly, or use a sequence pattern "
            "(/dir/f.%04d.exr, /dir/f.####.png, /dir/)."
        )
    if ext in OIIO_ONLY_EXTENSIONS:
        return (
            f"{ext} needs OpenImageIO, which is not installed. "
            "pip install OpenImageIO"
        )
    known = sorted(readable_extensions())
    return (
        f"{ext} is not a format this install can read. Available here: "
        + ", ".join(known[:24])
        + (f", and {len(known) - 24} more" if len(known) > 24 else "")
        + ". Installing OpenImageIO adds DPX, Cineon, ARRI and camera raw."
    )


# ── sequence discovery ─────────────────────────────────────────────────────

def _split_frame(stem: str) -> Optional[Tuple[str, str, str]]:
    m = _FRAME_IN_NAME_RE.match(stem)
    if not m:
        return None
    return m.group("stem"), m.group("sep"), m.group("frame")


def sequence_pattern_for(path: str) -> Optional[str]:
    """``/plates/sh010.1001.exr`` -> ``/plates/sh010.%04d.exr``. None if not numbered."""
    directory, filename = os.path.split(str(path))
    stem, ext = os.path.splitext(filename)
    parts = _split_frame(stem)
    if not parts:
        return None
    prefix, sep, frame = parts
    return os.path.join(directory, f"{prefix}{sep}%0{len(frame)}d{ext}")


def detect_sequence(path: str, *, max_scan: int = 200_000) -> Optional[SequenceInfo]:
    """Given one frame of a sequence, describe the whole thing.

    This is the behaviour that makes Nuke's Read feel like it is doing the work
    for you: point at any frame and you get the range. Returns None when the
    file is not part of a numbered set, or is the only frame in it -- a lone
    ``still.0001.png`` is a still, and treating it as a one-frame sequence would
    be a worse answer than treating it as an image.
    """
    if not os.path.isfile(path):
        return None
    directory, filename = os.path.split(os.path.abspath(path))
    stem, ext = os.path.splitext(filename)
    parts = _split_frame(stem)
    if not parts:
        return None
    prefix, sep, frame_text = parts
    padding = len(frame_text)

    # Match siblings with the same prefix/separator/extension and the same
    # padding. Mixed padding in one directory is a different sequence.
    sibling = re.compile(
        rf"^{re.escape(prefix)}{re.escape(sep)}(\d{{{padding}}}){re.escape(ext)}$",
        re.IGNORECASE,
    )
    try:
        entries = os.listdir(directory)
    except OSError as exc:
        logger.debug("cannot scan %s for a sequence: %s", directory, exc)
        return None
    if len(entries) > max_scan:
        logger.warning(
            "%s holds %d entries; skipping sequence detection. Use an explicit "
            "%%0%dd pattern instead.", directory, len(entries), padding,
        )
        return None

    frames = sorted(
        int(m.group(1)) for m in (sibling.match(e) for e in entries) if m
    )
    if len(frames) < 2:
        return None

    first, last = frames[0], frames[-1]
    present = set(frames)
    missing = tuple(f for f in range(first, last + 1) if f not in present)
    return SequenceInfo(
        pattern=os.path.join(directory, f"{prefix}{sep}%0{padding}d{ext}"),
        first=first,
        last=last,
        padding=padding,
        count=len(frames),
        missing=missing,
    )


def describe_pattern(pattern: str) -> Optional[SequenceInfo]:
    """Range of an explicit ``%04d`` / ``####`` pattern, by looking on disk."""
    text = str(pattern)
    printf = _HASH_RE.sub(lambda m: f"%0{len(m.group(0))}d", text)
    m = _PRINTF_RE.search(printf)
    if not m:
        return None
    directory, filename = os.path.split(os.path.abspath(printf))
    token = m.group(0)
    padding = int(re.sub(r"\D", "", token) or 0)
    head, _, tail = filename.partition(token)
    sibling = re.compile(
        rf"^{re.escape(head)}(\d{{{padding}}}){re.escape(tail)}$"
        if padding else rf"^{re.escape(head)}(\d+){re.escape(tail)}$",
        re.IGNORECASE,
    )
    try:
        entries = os.listdir(directory)
    except OSError:
        return None
    frames = sorted(int(mm.group(1)) for mm in (sibling.match(e) for e in entries) if mm)
    if not frames:
        return None
    present = set(frames)
    return SequenceInfo(
        pattern=os.path.join(directory, filename),
        first=frames[0],
        last=frames[-1],
        padding=padding or len(str(frames[0])),
        count=len(frames),
        missing=tuple(f for f in range(frames[0], frames[-1] + 1) if f not in present),
    )


def reset_cache() -> None:
    """Forget the backend probes (tests, and packages installed mid-session)."""
    _pillow_extensions.cache_clear()
    _oiio_extensions.cache_clear()


__all__ = [
    "EXR_EXTENSIONS",
    "FLOAT_EXTENSIONS",
    "OIIO_ONLY_EXTENSIONS",
    "SequenceInfo",
    "classify",
    "describe_pattern",
    "detect_sequence",
    "explain_unsupported",
    "image_extensions",
    "is_sequence_pattern",
    "readable_extensions",
    "reset_cache",
    "sequence_pattern_for",
]
