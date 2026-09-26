"""First-use download of Radiance's pixel SDR-to-HDR checkpoint.

`RadianceSDRToHDRUniversal`'s learned modes need a RUDRA SDR2HDRNet
checkpoint. The one fetched is ``sdr2hdr_shadow_v1.safetensors``, the model
RUDRA ships: the v5 backbone plus its trained shadow gate.
It used to be a manual install: download from Hugging Face, drop it in
``models/radiance``. Anyone who skipped that step got the plain expansion
fallback with only a log line to say why.

This module fetches it on first use instead. The file is small (~5 MB) and
first-party, so unlike the large third-party weights it is allowed by default.
The weights are licensed for NON-COMMERCIAL use (RUDRA's weights licence,
separate from Radiance's GPL-3.0 code), and the download says so in the log.
It is pinned to one Hugging Face commit and verified by size and sha256
before it is moved into place, so a changed or truncated file is never loaded.

Turn it off with any of::

    RADIANCE_ALLOW_DOWNLOADS=0
    HF_HUB_OFFLINE=1
    TRANSFORMERS_OFFLINE=1

The module imports no torch, so it is testable in the lightweight CI lane.
"""
from __future__ import annotations

import hashlib
import logging
import os
import threading
import urllib.request
from pathlib import Path
from typing import Optional

logger = logging.getLogger("radiance.model.pixel_download")

PIXEL_FILENAME = "sdr2hdr_shadow_v1.safetensors"
PIXEL_REPO = "fxtdstudios/RUDRA"
PIXEL_REVISION = "9cecedfa80fd4eaafadc590b6d924c68d86dd12d"
PIXEL_REPO_PATH = f"sdr2hdr/{PIXEL_FILENAME}"
PIXEL_URL = (
    f"https://huggingface.co/{PIXEL_REPO}/resolve/{PIXEL_REVISION}/{PIXEL_REPO_PATH}"
)
PIXEL_PAGE_URL = f"https://huggingface.co/{PIXEL_REPO}/blob/main/{PIXEL_REPO_PATH}"
PIXEL_LICENSE_URL = f"https://huggingface.co/{PIXEL_REPO}/blob/main/LICENSE"
PIXEL_SIZE = 4_878_392
PIXEL_SHA256 = "cffaefc7c2d06fd381b1222fe084cac9019eef881eb533e09c365745f84df2ac"

_TIMEOUT_S = 60
_CHUNK = 1 << 20

_lock = threading.Lock()
_failed = False          # one attempt per process; no retry storm on every queue
_refusal_logged = False  # say "downloads are off" once, not on every queue


def _target_dir() -> Optional[Path]:
    """First Radiance model directory that exists or can be created and written."""
    from radiance.model.paths import radiance_model_dirs
    for d in radiance_model_dirs():
        try:
            d.mkdir(parents=True, exist_ok=True)
            probe = d / ".radiance_write_probe"
            probe.write_bytes(b"")
            probe.unlink()
            return d
        except OSError:
            continue
    return None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch(url: str, dest: Path) -> None:
    """Stream *url* to *dest*. urllib honours HTTPS_PROXY / HTTP_PROXY."""
    req = urllib.request.Request(url, headers={"User-Agent": "radiance-comfyui"})
    with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp, open(dest, "wb") as out:
        while True:
            chunk = resp.read(_CHUNK)
            if not chunk:
                break
            out.write(chunk)


def download_pixel_checkpoint(*, force: bool = False) -> Optional[Path]:
    """Download the pinned pixel checkpoint into ``models/radiance``.

    Returns the installed path, or ``None`` when downloads are off, no model
    directory is writable, or the fetch or verification failed (each logged
    with the manual-install instructions). After one failure the process does
    not try again unless *force* is set.
    """
    global _failed, _refusal_logged
    from radiance.core.consent import downloads_allowed, refusal_message

    if not downloads_allowed(default=True):
        if not _refusal_logged:
            _refusal_logged = True
            logger.warning(refusal_message(
                PIXEL_FILENAME, size_mb=5, dest="ComfyUI/models/radiance/",
                url=PIXEL_PAGE_URL,
            ))
        return None

    with _lock:
        if _failed and not force:
            return None

        target = _target_dir()
        if target is None:
            logger.error(
                "[Radiance] No writable models/radiance folder for %s. Download it "
                "from %s into ComfyUI/models/radiance/.", PIXEL_FILENAME, PIXEL_PAGE_URL,
            )
            _failed = True
            return None

        final = target / PIXEL_FILENAME
        if final.is_file() and final.stat().st_size == PIXEL_SIZE:
            return final       # another thread or process got there first

        part = target / (PIXEL_FILENAME + ".part")
        logger.info("[Radiance] Downloading %s (~5 MB) to %s", PIXEL_FILENAME, target)
        try:
            _fetch(PIXEL_URL, part)
            size = part.stat().st_size
            if size != PIXEL_SIZE:
                raise ValueError(f"size {size} bytes, expected {PIXEL_SIZE}")
            digest = _sha256(part)
            if digest != PIXEL_SHA256:
                raise ValueError(f"sha256 {digest}, expected {PIXEL_SHA256}")
            os.replace(part, final)
        except Exception as exc:  # noqa: BLE001 - network, disk, verification
            _failed = True
            try:
                part.unlink()
            except OSError:
                pass
            logger.error(
                "[Radiance] Could not download %s: %s\n"
                "           Download it manually from %s\n"
                "           and put it in %s", PIXEL_FILENAME, exc, PIXEL_PAGE_URL, target,
            )
            return None

        logger.info(
            "[Radiance] Installed %s\n"
            "           RUDRA weights are licensed for non-commercial use only; "
            "commercial use needs a licence from FXTD Studios. Terms: %s",
            final, PIXEL_LICENSE_URL,
        )
        return final


def _reset_for_tests() -> None:
    global _failed, _refusal_logged
    _failed = False
    _refusal_logged = False
