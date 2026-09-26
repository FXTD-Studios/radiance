"""Read video the way a finishing house expects it to be read.

What this replaces
------------------
The old path decoded a clip by asking ffmpeg to write every frame to a
temporary directory as a PNG, then read the PNGs back one at a time. Measured
on a 4-second 1920x1080 ProRes 422 HQ clip: 25.0 s and roughly 600 MB of
temporary files, against 4.5 s for a raw pipe. It also dropped the alpha
channel of every ProRes 4444 file, because the frames came back through an
RGB-only reader, and the second decoder in the package (``_load_video_to_numpy``,
OpenCV first) quantised *everything* to 8 bits -- a 12-bit ProRes 4444 came out
on an exact 1/255 grid.

None of that is acceptable for the files this package exists to handle. A
ProRes 4444 with a matte is the single most common way a plate arrives from a
vendor, and silently discarding its fourth channel is worse than refusing to
open it.

What this does instead
----------------------
* **One raw pipe, no disk round-trip.** ffmpeg writes ``rawvideo`` to stdout and
  this module reads it frame by frame, so peak memory is the output array plus
  a single frame rather than the whole clip twice over.
* **Source bit depth.** 16 bits per component for anything above 8-bit, so
  ProRes 10/12-bit, DNxHR HQX and 10-bit HEVC survive. An 8-bit source is
  carried in 16 bits exactly -- there is no cost to being safe here.
* **Alpha, when the file has it.** ProRes 4444, 4444 XQ, and any RGBA-flavoured
  codec decode to four channels. The caller decides what to do with the fourth.
* **The container's colour tags are read**, not ignored: range, matrix,
  primaries and transfer characteristics, plus rotation and start timecode.
  Radiance still will not silently apply a transform you did not ask for, but
  it can now tell you what the file claims to be, and :func:`suggest_transfer`
  maps the tag onto a Radiance input colour space.
* **Exact frame ranges.** ``start``/``count``/``step`` select on the decoder's
  own frame counter, which is frame-accurate for long-GOP codecs too, where a
  time-based seek is not.
* **Failures are failures.** ffmpeg's stderr is captured and raised. A short
  read -- ffmpeg dying mid-clip -- raises instead of returning a truncated
  clip, which is the failure mode that costs a day of someone's time because
  the shot just quietly ends early.

Frame counts
------------
``nb_frames`` is absent from a lot of real media -- MXF and many MOVs report
nothing at all. :func:`probe` falls back to ``duration x fps`` and marks the
result estimated, so a caller can tell a known count from a guess.
"""
from __future__ import annotations

import functools
import json
import logging
import os
import re
import subprocess
import threading
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .ffmpeg import ffmpeg_exe, ffprobe_exe, require_ffmpeg

logger = logging.getLogger("radiance.video")

#: Container extensions this module is willing to treat as video.
VIDEO_EXTENSIONS = frozenset({
    ".mp4", ".mov", ".mxf", ".avi", ".webm", ".mkv", ".m4v", ".m2v",
    ".mpg", ".mpeg", ".ts", ".mts", ".m2ts", ".vob", ".ogv", ".r3d",
    ".braw", ".dv", ".flv", ".wmv", ".asf", ".yuv", ".y4m", ".gif",
})

#: Pixel formats that carry an alpha channel. ffmpeg's naming is regular
#: enough for a prefix test, and the odd ones out are listed explicitly.
_ALPHA_PREFIXES = ("yuva", "gbrap", "rgba", "bgra", "argb", "abgr", "ayuv", "vuya")
_ALPHA_EXACT = frozenset({"ya8", "ya16le", "ya16be", "pal8", "rgba64le", "rgba64be",
                          "bgra64le", "bgra64be", "gray8a"})

#: Codecs that store every frame independently. Worth knowing: a frame range
#: from one of these costs nothing extra, and they are what VFX plates arrive as.
_INTRA_CODECS = frozenset({
    "prores", "dnxhd", "cfhd", "ffv1", "huffyuv", "utvideo", "v210", "v410",
    "rawvideo", "mjpeg", "jpeg2000", "jpegls", "dirac", "dpx", "exr", "qtrle",
    "png", "tiff", "targa", "r10k", "r210", "avrp", "sheervideo",
})

#: ffmpeg colour-transfer tag -> Radiance input colour space name.
#: Only entries Radiance can actually invert are listed; an unmapped tag is
#: reported rather than guessed at.
TRANSFER_TO_RADIANCE: Dict[str, str] = {
    "bt709": "Rec.709 (BT.1886)",
    "bt470bg": "Rec.709 (BT.1886)",
    "smpte170m": "Rec.709 (BT.1886)",
    "bt1361e": "Rec.709 (BT.1886)",
    "iec61966-2-1": "sRGB",
    "iec61966_2_1": "sRGB",
    "srgb": "sRGB",
    "bt2020-10": "Rec.2020 (BT.2020 OETF)",
    "bt2020_10": "Rec.2020 (BT.2020 OETF)",
    "bt2020-12": "Rec.2020 (BT.2020 OETF)",
    "bt2020_12": "Rec.2020 (BT.2020 OETF)",
    "smpte2084": "PQ (ST.2084)",
    "arib-std-b67": "HLG (BT.2100)",
    "arib_std_b67": "HLG (BT.2100)",
    "linear": "Auto / Linear (pass-through)",
}

_PROBE_TIMEOUT = 30.0
#: How much of ffmpeg's stderr to quote back when it fails. Enough to be
#: useful, bounded so a broken file cannot flood the log.
_STDERR_TAIL = 4000


class VideoDecodeError(RuntimeError):
    """ffmpeg could not decode the file, and here is what it said."""


class VideoTruncatedError(VideoDecodeError):
    """The decoder stopped early. The clip you would have got is incomplete."""


@dataclass(frozen=True)
class VideoInfo:
    """What the container claims about itself.

    Every field is what ffprobe reported, not what Radiance would like it to
    be. ``frames_estimated`` is the one to check before trusting ``frames``.
    """

    path: str
    width: int = 0
    height: int = 0
    fps: Fraction = Fraction(24, 1)
    frames: int = 0
    frames_estimated: bool = False
    duration: float = 0.0
    codec: str = ""
    profile: str = ""
    pix_fmt: str = ""
    bit_depth: int = 8
    has_alpha: bool = False
    color_range: str = ""
    color_space: str = ""
    color_primaries: str = ""
    color_transfer: str = ""
    rotation: int = 0
    timecode: str = ""
    field_order: str = ""
    sample_aspect_ratio: str = ""
    audio_streams: int = 0

    @property
    def fps_float(self) -> float:
        return float(self.fps)

    @property
    def is_intra(self) -> bool:
        return self.codec in _INTRA_CODECS

    @property
    def is_hdr(self) -> bool:
        return self.color_transfer in ("smpte2084", "arib-std-b67", "arib_std_b67")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps_float, 6),
            "fps_exact": f"{self.fps.numerator}/{self.fps.denominator}",
            "frames": self.frames,
            "frames_estimated": self.frames_estimated,
            "duration": round(self.duration, 6),
            "codec": self.codec,
            "profile": self.profile,
            "pix_fmt": self.pix_fmt,
            "bit_depth": self.bit_depth,
            "has_alpha": self.has_alpha,
            "color_range": self.color_range,
            "color_space": self.color_space,
            "color_primaries": self.color_primaries,
            "color_transfer": self.color_transfer,
            "rotation": self.rotation,
            "timecode": self.timecode,
            "field_order": self.field_order,
            "sample_aspect_ratio": self.sample_aspect_ratio,
            "audio_streams": self.audio_streams,
            "intra_only": self.is_intra,
            "hdr": self.is_hdr,
        }

    def summary(self) -> str:
        """One line for the console, in the order a colourist would read it."""
        bits = [
            f"{self.width}x{self.height}",
            f"{self.fps_float:g} fps",
            f"{self.frames}{'~' if self.frames_estimated else ''} frames",
            self.codec or "?",
            f"{self.bit_depth}-bit",
        ]
        if self.has_alpha:
            bits.append("alpha")
        if self.color_transfer:
            bits.append(f"trc={self.color_transfer}")
        if self.color_primaries:
            bits.append(f"prim={self.color_primaries}")
        if self.color_range:
            bits.append(f"range={self.color_range}")
        if self.timecode:
            bits.append(f"tc={self.timecode}")
        if self.rotation:
            bits.append(f"rot={self.rotation}deg")
        return " · ".join(bits)


# ── pixel format introspection ─────────────────────────────────────────────

def pix_fmt_has_alpha(pix_fmt: str) -> bool:
    """Does this ffmpeg pixel format carry an alpha channel?"""
    if not pix_fmt:
        return False
    name = pix_fmt.strip().lower()
    if name in _ALPHA_EXACT:
        return name != "pal8"  # pal8 has alpha in theory; in practice, no.
    return name.startswith(_ALPHA_PREFIXES)


def pix_fmt_bit_depth(pix_fmt: str) -> int:
    """Bits per component, from the format name.

    ``yuv422p10le`` -> 10, ``gbrp12le`` -> 12, ``rgb48le`` -> 16, ``rgb24`` -> 8.
    """
    if not pix_fmt:
        return 8
    name = pix_fmt.strip().lower()
    # Planar formats spell the depth out after the plane layout: p10le, p16be.
    m = re.search(r"p(\d+)(?:le|be)?$", name)
    if m:
        return int(m.group(1))
    # Packed formats give total bits across all components: rgb24, rgba64.
    m = re.match(r"^(?:a?[rgb]{3}a?|bgr[a]?|argb|abgr|ya)(\d+)", name)
    if m:
        total = int(m.group(1))
        components = 4 if pix_fmt_has_alpha(name) else 3
        if name.startswith("ya"):
            components = 2
        if total % components == 0:
            return total // components
    if re.search(r"(?:le|be)$", name) and re.search(r"1[026]", name):
        return 16
    return 8


# ── probing ────────────────────────────────────────────────────────────────

def _parse_fraction(text: str, fallback: Fraction) -> Fraction:
    """``"24000/1001"`` -> Fraction. Survives ``"24"``, ``"0/0"`` and junk."""
    if not text:
        return fallback
    try:
        value = Fraction(str(text).strip())
    except (ValueError, ZeroDivisionError):
        return fallback
    return value if value > 0 else fallback


def _rotation_from(stream: Dict[str, Any]) -> int:
    """Display rotation in degrees, from side data or the legacy tag."""
    for side in stream.get("side_data_list", []) or []:
        if "rotation" in side:
            try:
                return int(round(float(side["rotation"]))) % 360
            except (TypeError, ValueError):
                pass
    tag = (stream.get("tags") or {}).get("rotate")
    if tag:
        try:
            return int(round(float(tag))) % 360
        except (TypeError, ValueError):
            pass
    return 0


def _tag(*sources: Optional[Dict[str, Any]], key: str) -> str:
    for src in sources:
        if not src:
            continue
        value = (src.get("tags") or {}).get(key)
        if value:
            return str(value)
    return ""


def probe(path: str, *, timeout: float = _PROBE_TIMEOUT) -> VideoInfo:
    """Read everything the container will tell us about its video stream.

    Never raises for a merely uninformative file: an ffprobe that is missing,
    slow or confused yields a VideoInfo with defaults and an explanation in the
    log. It *does* raise :class:`FileNotFoundError` for a path that is not
    there, because that is a different mistake and deserves a different message.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No such video file: {path}")

    exe = ffprobe_exe()
    if not exe:
        logger.warning(
            "ffprobe was not found, so %s is being decoded without reading its "
            "metadata: frame rate, frame count, colour tags and timecode will "
            "be unavailable. Install ffmpeg (which ships ffprobe) to get them.",
            os.path.basename(path),
        )
        return VideoInfo(path=path)

    cmd = [
        exe, "-v", "quiet", "-print_format", "json",
        "-show_streams", "-show_format", path,
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        data = json.loads(out.stdout or "{}")
    except (subprocess.SubprocessError, OSError, ValueError) as exc:
        logger.warning("ffprobe could not read %s (%s); decoding without metadata.",
                       os.path.basename(path), exc)
        return VideoInfo(path=path)

    streams = data.get("streams") or []
    fmt = data.get("format") or {}
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise VideoDecodeError(
            f"{os.path.basename(path)} has no video stream. "
            f"ffprobe found {len(streams)} stream(s): "
            + (", ".join(s.get("codec_type", "?") for s in streams) or "none")
        )

    fps = _parse_fraction(video.get("avg_frame_rate"), Fraction(0))
    if fps <= 0:
        fps = _parse_fraction(video.get("r_frame_rate"), Fraction(24, 1))

    duration = 0.0
    for candidate in (video.get("duration"), fmt.get("duration")):
        try:
            duration = float(candidate)
            if duration > 0:
                break
        except (TypeError, ValueError):
            continue

    frames, estimated = 0, False
    for key in ("nb_frames", "nb_read_frames", "nb_read_packets"):
        try:
            frames = int(video.get(key) or 0)
        except (TypeError, ValueError):
            frames = 0
        if frames > 0:
            break
    if frames <= 0 and duration > 0 and fps > 0:
        # MXF and plenty of MOVs carry no frame count at all. An estimate that
        # is labelled as one beats a zero that reads like "empty file".
        frames = int(round(duration * float(fps)))
        estimated = True

    pix_fmt = str(video.get("pix_fmt") or "")
    return VideoInfo(
        path=path,
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        fps=fps,
        frames=max(frames, 0),
        frames_estimated=estimated,
        duration=duration,
        codec=str(video.get("codec_name") or ""),
        profile=str(video.get("profile") or ""),
        pix_fmt=pix_fmt,
        bit_depth=pix_fmt_bit_depth(pix_fmt),
        has_alpha=pix_fmt_has_alpha(pix_fmt),
        color_range=str(video.get("color_range") or ""),
        color_space=str(video.get("color_space") or ""),
        color_primaries=str(video.get("color_primaries") or ""),
        color_transfer=str(video.get("color_transfer") or ""),
        rotation=_rotation_from(video),
        timecode=_tag(video, fmt, key="timecode"),
        field_order=str(video.get("field_order") or ""),
        sample_aspect_ratio=str(video.get("sample_aspect_ratio") or ""),
        audio_streams=sum(1 for s in streams if s.get("codec_type") == "audio"),
    )


def suggest_transfer(info: VideoInfo) -> Optional[str]:
    """The Radiance input colour space implied by the file's own tags.

    ``None`` when the file is untagged or tagged with something Radiance has no
    inverse for -- in which case the honest thing is to say so and let the user
    choose, not to guess.
    """
    if not info.color_transfer:
        return None
    return TRANSFER_TO_RADIANCE.get(info.color_transfer.strip().lower())


# ── decoding ───────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def _supports_fps_mode() -> bool:
    """``-fps_mode`` replaced ``-vsync`` in ffmpeg 5.0."""
    exe = ffmpeg_exe()
    if not exe:
        return False
    try:
        out = subprocess.run([exe, "-version"], capture_output=True, text=True,
                             timeout=10).stdout
    except (subprocess.SubprocessError, OSError):
        return False
    m = re.search(r"ffmpeg version n?(\d+)\.", out or "")
    return bool(m) and int(m.group(1)) >= 5


def _select_expression(start: int, count: int, step: int) -> Optional[str]:
    """A frame-accurate ``select`` filter, or None when we want everything.

    Selecting on ``n`` -- the decoder's own frame counter -- rather than seeking
    by time is exact for long-GOP media too. It costs a decode of the frames
    before ``start``, which for the intra codecs VFX plates arrive in is cheap
    and for a long H.264 is the price of being right.
    """
    if start <= 0 and count <= 0 and step <= 1:
        return None
    terms = []
    if start > 0:
        terms.append(f"gte(n\\,{start})")
    if count > 0:
        last = start + (count - 1) * max(step, 1)
        terms.append(f"lte(n\\,{last})")
    if step > 1:
        terms.append(f"not(mod(n-{start}\\,{step}))")
    return "select='" + "*".join(terms) + "'"


def _drain(stream, sink: List[bytes]) -> None:
    try:
        for chunk in iter(lambda: stream.read(65536), b""):
            sink.append(chunk)
    except (OSError, ValueError):  # pragma: no cover - pipe torn down early
        pass


def _read_exactly(stream, size: int) -> bytes:
    """Read exactly ``size`` bytes, or fewer at a clean end of stream."""
    chunks: List[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def decode(
    path: str,
    *,
    start: int = 0,
    count: int = 0,
    step: int = 1,
    alpha: bool = True,
    info: Optional[VideoInfo] = None,
    on_frame: Optional[Callable[[int], None]] = None,
    timeout: Optional[float] = None,
) -> Tuple[np.ndarray, VideoInfo]:
    """Decode to ``(N, H, W, C)`` float32 in 0..1, at the source's bit depth.

    Parameters
    ----------
    start, count, step
        Frame selection on the source's own frame numbering, zero-based.
        ``count=0`` means "to the end".
    alpha
        Keep the alpha channel when the source has one. ``C`` is 4 only when
        the file actually carries alpha and this is True.
    info
        A previously probed :class:`VideoInfo`, to avoid probing twice.
    on_frame
        Called with the running frame count. For progress reporting on a long
        decode; keep it cheap.
    timeout
        Seconds to wait for ffmpeg. ``None`` -- the default -- means no limit,
        because a fixed cap turns a long clip into a spurious failure. The old
        code capped every decode at 300 s regardless of length.

    Raises
    ------
    VideoDecodeError
        ffmpeg failed. The message quotes its stderr.
    VideoTruncatedError
        ffmpeg exited mid-frame, so the clip is incomplete. Raised rather than
        returned, because a short clip that looks fine is the expensive bug.
    """
    exe = require_ffmpeg()
    if info is None:
        info = probe(path)

    want_alpha = bool(alpha and info.has_alpha)
    channels = 4 if want_alpha else 3
    pix_fmt = "rgba64le" if want_alpha else "rgb48le"

    cmd: List[str] = [exe, "-nostdin", "-v", "error", "-i", path]
    select = _select_expression(max(start, 0), max(count, 0), max(step, 1))
    if select:
        cmd += ["-vf", select]
    cmd += ["-fps_mode", "passthrough"] if _supports_fps_mode() else ["-vsync", "0"]
    cmd += ["-map", "0:v:0", "-pix_fmt", pix_fmt, "-f", "rawvideo", "-"]

    width, height = info.width, info.height
    if width <= 0 or height <= 0:
        # No probe data. Decode one frame to a null muxer purely to learn the
        # size, rather than guessing and reshaping garbage.
        width, height = _dimensions_by_decode(exe, path)
    if info.rotation in (90, 270):
        # ffmpeg autorotates by default, so the frames arriving on the pipe are
        # already upright and their width/height are swapped relative to the
        # stored stream dimensions.
        width, height = height, width

    frame_bytes = width * height * channels * 2  # 16 bits per component
    if frame_bytes <= 0:
        raise VideoDecodeError(
            f"Could not determine the frame size of {os.path.basename(path)}."
        )

    logger.debug("decoding %s: %s", os.path.basename(path), " ".join(cmd))

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            bufsize=frame_bytes)
    errors: List[bytes] = []
    pump = threading.Thread(target=_drain, args=(proc.stderr, errors), daemon=True)
    pump.start()

    scale = np.float32(1.0 / 65535.0)
    # Preallocate and convert straight into the output. Collecting frames in a
    # list and np.stack-ing at the end costs a second full copy of the clip --
    # on a 4-second HD ProRes that is another 2.4 GB and, measured, doubles the
    # wall-clock. `expected` is only a hint: a short read returns a view and a
    # long one falls back to appending.
    expected = _expected_frame_count(info, start, count, step)
    out: Optional[np.ndarray] = None
    if expected > 0:
        try:
            out = np.empty((expected, height, width, channels), np.float32)
        except MemoryError:
            logger.debug("could not preallocate %d frames; growing instead", expected)
            out = None
    overflow: List[np.ndarray] = []
    written = 0
    partial = 0
    try:
        while True:
            buf = _read_exactly(proc.stdout, frame_bytes)
            if not buf:
                break
            if len(buf) != frame_bytes:
                partial = len(buf)
                break
            frame = np.frombuffer(buf, dtype="<u2").reshape(height, width, channels)
            if out is not None and written < expected:
                np.multiply(frame, scale, out=out[written])
            else:
                overflow.append(frame * scale)
            written += 1
            if on_frame is not None:
                on_frame(written)
            if count > 0 and written >= count:
                break
    finally:
        try:
            proc.stdout.close()
        except OSError:  # pragma: no cover
            pass
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:  # pragma: no cover - only with a cap
            proc.kill()
            proc.wait()
            raise VideoDecodeError(
                f"ffmpeg did not finish decoding {os.path.basename(path)} "
                f"within {timeout:g}s."
            ) from None
        pump.join(timeout=5)

    stderr = b"".join(errors).decode("utf-8", "replace").strip()
    tail = stderr[-_STDERR_TAIL:]

    if partial:
        raise VideoTruncatedError(
            f"ffmpeg stopped {partial} bytes into frame {written + 1} of "
            f"{os.path.basename(path)}; the file is truncated or corrupt. "
            f"{written} complete frame(s) were decoded before that point."
            + (f"\nffmpeg said: {tail}" if tail else "")
        )

    if not written:
        if proc.returncode not in (0, None):
            raise VideoDecodeError(
                f"ffmpeg failed to decode {os.path.basename(path)} "
                f"(exit {proc.returncode})."
                + (f"\nffmpeg said: {tail}" if tail else "")
            )
        where = f" from frame {start}" if start else ""
        raise VideoDecodeError(
            f"No frames were decoded from {os.path.basename(path)}{where}. "
            + (f"The file reports {info.frames} frame(s)."
               if info.frames else "The file reports no frame count.")
            + (f"\nffmpeg said: {tail}" if tail else "")
        )

    if proc.returncode not in (0, None):
        # Frames did arrive, but ffmpeg still failed. Never return these
        # quietly: a clip that ends early looks exactly like a clip that was
        # meant to end there.
        raise VideoTruncatedError(
            f"ffmpeg exited {proc.returncode} after {written} frame(s) of "
            f"{os.path.basename(path)}, so the decode is incomplete."
            + (f"\nffmpeg said: {tail}" if tail else "")
        )

    if out is None:
        return np.stack(overflow, axis=0), info
    if overflow:
        # The file had more frames than its own metadata claimed. Believe the
        # decoder over the header.
        return np.concatenate([out, np.stack(overflow, axis=0)], axis=0), info
    return (out if written == expected else out[:written]), info


#: An estimated frame count can be nonsense (a bad duration on a damaged file).
#: Cap the speculative allocation at something a workstation can absorb; the
#: decode still returns every frame, it just grows past this point.
_MAX_SPECULATIVE_FRAMES = 4096


def _expected_frame_count(info: VideoInfo, start: int, count: int, step: int) -> int:
    """How many frames the selection should yield, as a preallocation hint."""
    if count > 0:
        return count
    if info.frames <= 0:
        return 0
    remaining = info.frames - max(start, 0)
    if remaining <= 0:
        return 0
    expected = -(-remaining // max(step, 1))  # ceil
    return min(expected, _MAX_SPECULATIVE_FRAMES)


def _dimensions_by_decode(exe: str, path: str) -> Tuple[int, int]:
    """Frame size for a file ffprobe could not describe, by decoding one frame."""
    cmd = [exe, "-nostdin", "-v", "error", "-i", path, "-frames:v", "1",
           "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=120)
    except (subprocess.SubprocessError, OSError) as exc:
        raise VideoDecodeError(f"Could not decode {os.path.basename(path)}: {exc}") from exc
    size = len(out.stdout)
    if size <= 0:
        err = out.stderr.decode("utf-8", "replace").strip()[-_STDERR_TAIL:]
        raise VideoDecodeError(
            f"Could not determine the frame size of {os.path.basename(path)}."
            + (f"\nffmpeg said: {err}" if err else "")
        )
    raise VideoDecodeError(
        f"{os.path.basename(path)} decoded {size} bytes but neither ffprobe nor "
        "the stream header reported its dimensions. Re-wrap the file "
        "(ffmpeg -i in.mov -c copy out.mov) and try again."
    )


def decode_to_tensor(path: str, **kwargs):
    """:func:`decode`, returning a torch tensor. Imported lazily on purpose."""
    import torch

    array, info = decode(path, **kwargs)
    return torch.from_numpy(np.ascontiguousarray(array)), info


__all__ = [
    "VIDEO_EXTENSIONS",
    "TRANSFER_TO_RADIANCE",
    "VideoDecodeError",
    "VideoTruncatedError",
    "VideoInfo",
    "decode",
    "decode_to_tensor",
    "pix_fmt_bit_depth",
    "pix_fmt_has_alpha",
    "probe",
    "suggest_transfer",
]
