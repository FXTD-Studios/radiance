"""Locate the ffmpeg / ffprobe binaries.

Every call site in this package used to hard-code the bare string ``"ffmpeg"``
and rely on it being on PATH. On Windows it usually is not: ffmpeg has no
installer that puts it there by default, so video export failed with a bare
``FileNotFoundError`` on a machine that had a perfectly good ffmpeg sitting in
site-packages the whole time.

``imageio-ffmpeg`` has been a declared dependency since 3.0 and ships exactly
that binary, but nothing in the package ever called it. These helpers close the
gap: PATH first (a system ffmpeg is usually newer and has more codecs compiled
in), then the bundled one.

The lookup is cached because it runs on every frame-range decode.
"""
from __future__ import annotations

import functools
import logging
import os
import shutil
import subprocess
from typing import FrozenSet, Optional

logger = logging.getLogger("radiance.ffmpeg")

#: Set to an absolute path to override discovery entirely.
_ENV_OVERRIDE = "RADIANCE_FFMPEG"
_ENV_OVERRIDE_PROBE = "RADIANCE_FFPROBE"


def _from_env(var: str) -> Optional[str]:
    value = os.environ.get(var, "").strip().strip('"')
    if not value:
        return None
    if os.path.isfile(value) and os.access(value, os.X_OK):
        return value
    logger.warning(
        "%s is set to %r, which is not an executable file. Ignoring it and "
        "falling back to PATH.", var, value,
    )
    return None


def _from_imageio() -> Optional[str]:
    """The ffmpeg bundled with imageio-ffmpeg, if that package is installed."""
    try:
        import imageio_ffmpeg  # type: ignore

        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.debug("imageio-ffmpeg unavailable: %s", exc)
        return None
    return exe if exe and os.path.isfile(exe) else None


@functools.lru_cache(maxsize=1)
def ffmpeg_exe() -> Optional[str]:
    """Absolute path to an ffmpeg binary, or None if there isn't one.

    Order: ``RADIANCE_FFMPEG`` → PATH → the imageio-ffmpeg bundle.
    """
    return _from_env(_ENV_OVERRIDE) or shutil.which("ffmpeg") or _from_imageio()


@functools.lru_cache(maxsize=1)
def ffprobe_exe() -> Optional[str]:
    """Absolute path to ffprobe, or None.

    imageio-ffmpeg ships ffmpeg but *not* ffprobe, so this one really can come
    back empty on a machine where :func:`ffmpeg_exe` succeeds. Callers that use
    ffprobe only to read metadata should degrade to defaults rather than fail.
    """
    from_env = _from_env(_ENV_OVERRIDE_PROBE)
    if from_env:
        return from_env
    found = shutil.which("ffprobe")
    if found:
        return found
    # Some builds drop both binaries in the same directory.
    ff = ffmpeg_exe()
    if ff:
        candidate = os.path.join(
            os.path.dirname(ff), "ffprobe" + (".exe" if os.name == "nt" else "")
        )
        if os.path.isfile(candidate):
            return candidate
    return None


def ffmpeg_available() -> bool:
    return ffmpeg_exe() is not None


def require_ffmpeg() -> str:
    """Return the ffmpeg path or raise with an actionable message."""
    exe = ffmpeg_exe()
    if exe:
        return exe
    raise RuntimeError(
        "ffmpeg was not found. Install it and put it on PATH, run "
        "`pip install imageio-ffmpeg` to use the bundled build, or set "
        f"{_ENV_OVERRIDE} to the full path of an ffmpeg executable."
    )


@functools.lru_cache(maxsize=None)
def _encoders(exe: str) -> FrozenSet[str]:
    """Encoder names an ffmpeg build carries (``ffmpeg -encoders``)."""
    try:
        out = subprocess.run([exe, "-hide_banner", "-encoders"],
                             capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("could not list the encoders of %s: %s", exe, exc)
        return frozenset()
    # Lines read " V....D libx264  description"; the legend's lines read " V..... = Video".
    return frozenset(parts[1] for parts in (line.split() for line in out.splitlines())
                     if len(parts) > 1 and len(parts[0]) == 6 and parts[1] != "=")


@functools.lru_cache(maxsize=None)
def ffmpeg_with_encoder(encoder: str) -> str:
    """The first ffmpeg that carries `encoder`, or raise naming what is missing.

    ALBABIT-FIX: Stability Matrix puts its own ffmpeg first on ComfyUI's PATH,
    a build without libx265, so "MP4 (H.265 10-bit)" failed with "Unknown
    encoder" while the imageio-ffmpeg bundle has it. PATH is still tried
    first; ``RADIANCE_FFMPEG``, when set, stays the only one used.
    """
    env = _from_env(_ENV_OVERRIDE)
    candidates = [env] if env else [c for c in (shutil.which("ffmpeg"), _from_imageio()) if c]
    if not candidates:
        require_ffmpeg()
    for exe in candidates:
        if encoder in _encoders(exe):
            if exe != candidates[0]:
                logger.info("The ffmpeg on PATH (%s) has no %s encoder; using %s.",
                            candidates[0], encoder, exe)
            return exe
    raise RuntimeError(
        f"No ffmpeg found here carries the {encoder} encoder ({', '.join(candidates)}). "
        f"Install an ffmpeg build that has it (the 'essentials' or 'full' builds do) "
        f"and put it on PATH, or set {_ENV_OVERRIDE} to its full path."
    )


def reset_cache() -> None:
    """Forget the cached lookups (tests, and PATH changes at runtime)."""
    ffmpeg_exe.cache_clear()
    ffprobe_exe.cache_clear()
    ffmpeg_with_encoder.cache_clear()
    _encoders.cache_clear()


__all__ = [
    "ffmpeg_exe",
    "ffprobe_exe",
    "ffmpeg_available",
    "ffmpeg_with_encoder",
    "require_ffmpeg",
    "reset_cache",
]
