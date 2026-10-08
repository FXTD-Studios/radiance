"""Download one model file: pinned, verified, resumable, with the right error.

Every Radiance downloader that fetches a single weights file goes through
`fetch()` (3.5.0). Before, each had its own copy: some checked a SHA-256 only
when one happened to be pinned (none were), some wrote straight to the final
path so an interrupted download left a truncated checkpoint that was loaded
next time, none could reach a gated Hugging Face repo, and a 40 GB file that
dropped at 90 % started again from zero.

What `fetch()` guarantees:

* The file lands at `dest` only after its SHA-256 matches the pinned digest.
  A mismatch deletes the download and raises; nothing unverified is kept.
* The download goes to `dest + ".part"` and is resumed from there with an
  HTTP Range request if it was interrupted.
* Hugging Face URLs carry the user's token (``HF_TOKEN``,
  ``HUGGING_FACE_HUB_TOKEN`` or the ``huggingface-cli login`` cache) so gated
  repos work once their licence is accepted; a 401 / 403 says exactly that.
* Consent is checked here too (see `radiance.core.consent`): downloads are on
  by default and ``RADIANCE_ALLOW_DOWNLOADS=0`` or the Hugging Face offline
  flags turn them off.
* A file already at `dest` is checked against the pinned digest once; the
  pass is recorded beside it (``<file>.radiance-sha256``, keyed on size and
  mtime) so a large file is not hashed again on every run. A file that does
  not match is downloaded again, or reported when downloads are off; it is
  never used silently.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import time
import urllib.error
import urllib.request
from typing import Optional

logger = logging.getLogger("radiance.model_fetch")

_CHUNK = 1 << 20
_TIMEOUT_S = 60
_HF_RE = re.compile(r"^https://huggingface\.co/([^/]+/[^/]+)/resolve/")


class ModelFetchError(RuntimeError):
    """A model could not be fetched. The message says why and what to do."""


def hf_token() -> Optional[str]:
    """The user's Hugging Face token, if they have one set up."""
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        tok = (os.environ.get(name) or "").strip()
        if tok:
            return tok
    try:
        from huggingface_hub import get_token  # type: ignore
        tok = get_token()
        return tok.strip() if tok else None
    except Exception:  # noqa: BLE001 - huggingface_hub absent or too old
        return None


def _sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


# Files that passed the digest check in this process, keyed on
# (path, size, mtime_ns) -> sha256. The sidecar below carries the same record
# across restarts; this covers a models folder the process cannot write to.
_VERIFIED: dict = {}
_RECORD_SUFFIX = ".radiance-sha256"


def _stamp(path: str) -> tuple:
    st = os.stat(path)
    return (os.path.abspath(path), st.st_size, st.st_mtime_ns)


def _write_record(path: str, sha256: str, stamp: tuple) -> None:
    with open(path + _RECORD_SUFFIX, "w", encoding="ascii") as fh:
        fh.write(f"{sha256} {stamp[1]} {stamp[2]}\n")


def _remember(path: str, sha256: str) -> None:
    """Record that `path`, as it is now, has digest `sha256`."""
    stamp = _stamp(path)
    _VERIFIED[stamp] = sha256
    try:
        _write_record(path, sha256, stamp)
    except OSError as e:  # read-only models folder: the in-process record still holds
        logger.debug("[Radiance] could not record the checksum pass for %s: %s", path, e)


def _recorded_pass(path: str, sha256: str) -> bool:
    """True when `path` passed against `sha256` and has not changed since."""
    stamp = _stamp(path)
    if _VERIFIED.get(stamp) == sha256:
        return True
    try:
        with open(path + _RECORD_SUFFIX, encoding="ascii") as fh:
            digest, size, mtime_ns = fh.read().split()
    except (OSError, ValueError):
        return False
    if digest == sha256 and int(size) == stamp[1] and int(mtime_ns) == stamp[2]:
        _VERIFIED[stamp] = sha256
        return True
    return False


def _existing_matches(dest: str, expected: str, label: str) -> bool:
    """Check a file already on disk against its pinned digest, hashing it at
    most once while it stays unchanged."""
    if _recorded_pass(dest, expected):
        return True
    logger.info("[Radiance] %s: checking %s against its pinned SHA-256 (once per file).", label, dest)
    try:
        actual = _sha256_of(dest)
    except OSError as e:
        raise ModelFetchError(f"[Radiance] {label}: could not read {dest} to check it ({e}).") from e
    if actual != expected:
        logger.warning("[Radiance] %s: %s does not match its pinned SHA-256 (expected %s, got %s).",
                       label, dest, expected, actual)
        return False
    _remember(dest, expected)
    return True


def _drop_empty(part: str) -> bool:
    """Remove a zero-byte .part file; True if there was nothing worth keeping."""
    try:
        if os.path.isfile(part) and os.path.getsize(part) == 0:
            os.remove(part)
        return not os.path.isfile(part)
    except OSError:
        return True


def _gated_message(label: str, url: str, code: int) -> str:
    m = _HF_RE.match(url)
    page = f"https://huggingface.co/{m.group(1)}" if m else url
    have = ("A token was sent and refused: the licence is probably not accepted yet for this account."
            if hf_token() else "No Hugging Face token is set.")
    return (
        f"[Radiance] {label}: the download was refused (HTTP {code}). This model's repository is gated. {have}\n"
        f"           1. Open {page}, sign in and accept its licence.\n"
        f"           2. Set HF_TOKEN to a read token (or run `huggingface-cli login`) and restart ComfyUI.\n"
        f"           The download then runs automatically on the next queue."
    )


def _progress_bar(total: int):
    try:
        import comfy.utils  # type: ignore
        return comfy.utils.ProgressBar(max(1, total))
    except Exception:  # noqa: BLE001 - outside ComfyUI
        return None


def fetch(
    url: str,
    dest: str,
    *,
    sha256: str,
    size: Optional[int] = None,
    label: Optional[str] = None,
    legacy_offline_env: Optional[str] = None,
) -> str:
    """Make sure `dest` holds the file at `url` with digest `sha256`; return `dest`.

    An existing `dest` is checked against `sha256` the first time it is
    seen and the pass is recorded beside it, so later runs only stat it.
    Until 4.0 an existing file was trusted when its size matched, so a
    corrupt or swapped file of the right size loaded silently. A mismatching
    file is downloaded again, or reported when downloads are off.
    """
    label = label or os.path.basename(dest)
    expected = (sha256 or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ModelFetchError(f"[Radiance] {label}: no SHA-256 is pinned for {url}; refusing to download it.")

    mismatch = False
    if os.path.isfile(dest):
        if size is not None and os.path.getsize(dest) != int(size):
            logger.warning("[Radiance] %s: %s is %d bytes, expected %d; downloading it again.",
                           label, dest, os.path.getsize(dest), int(size))
            mismatch = True
        elif _existing_matches(dest, expected, label):
            return dest
        else:
            mismatch = True

    from radiance.core.consent import ALLOW_ENV, downloads_allowed, refusal_message
    if not downloads_allowed(legacy_offline_env=legacy_offline_env):
        if mismatch:
            raise ModelFetchError(
                f"[Radiance] {label}: {dest} does not match its pinned SHA-256 ({expected}),\n"
                f"           so it was not loaded, and automatic downloads are off "
                f"({ALLOW_ENV}=0, an offline flag or a legacy *_OFFLINE=1 is set).\n"
                f"           Replace it with the file from {url}, or unset that setting to let "
                f"Radiance download it again.")
        raise ModelFetchError(refusal_message(
            label, size_mb=round(size / 1e6) if size else None, dest=dest, url=url))

    os.makedirs(os.path.dirname(os.path.abspath(dest)) or ".", exist_ok=True)
    part = dest + ".part"
    have = os.path.getsize(part) if os.path.isfile(part) else 0
    if size is not None and have > int(size):
        os.remove(part)
        have = 0

    headers = {"User-Agent": "radiance-comfyui"}
    token = hf_token() if _HF_RE.match(url) else None
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if have:
        headers["Range"] = f"bytes={have}-"

    hasher = hashlib.sha256()
    if have:
        with open(part, "rb") as fh:
            for block in iter(lambda: fh.read(_CHUNK), b""):
                hasher.update(block)

    size_txt = f" ({size / 1e9:.2f} GB)" if size and size >= 1e9 else (f" ({size / 1e6:.0f} MB)" if size else "")
    logger.info("[Radiance] Downloading %s%s from %s%s", label, size_txt, url,
                f", resuming at {have / 1e6:.0f} MB" if have else "")
    done = have
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=_TIMEOUT_S) as resp:
            if have and getattr(resp, "status", 200) != 206:
                # The server ignored the Range request: start over.
                have = 0
                hasher = hashlib.sha256()
                mode = "wb"
            else:
                mode = "ab" if have else "wb"
            total = int(size) if size else (have + int(resp.headers.get("Content-Length") or 0))
            bar = _progress_bar(total // _CHUNK if total else 0)
            done, last_pct, last_t = have, -1, time.time()
            with open(part, mode) as out:
                for block in iter(lambda: resp.read(_CHUNK), b""):
                    out.write(block)
                    hasher.update(block)
                    done += len(block)
                    if bar is not None:
                        bar.update_absolute(done // _CHUNK)
                    if total:
                        pct = done * 100 // total
                        if pct >= last_pct + 10 or time.time() - last_t > 30:
                            logger.info("[Radiance]   %s: %d%% (%d / %d MB)", label, pct, done >> 20, total >> 20)
                            last_pct, last_t = pct, time.time()
    except urllib.error.HTTPError as e:
        _drop_empty(part)
        if e.code in (401, 403) and _HF_RE.match(url):
            raise ModelFetchError(_gated_message(label, url, e.code)) from e
        if e.code == 416 and have:          # the part file is already complete
            pass
        else:
            raise ModelFetchError(f"[Radiance] {label}: download failed, HTTP {e.code} from {url}.") from e
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        kept = not _drop_empty(part)
        raise ModelFetchError(f"[Radiance] {label}: download failed ({e})." + (
            f" The partial file is kept at {part} and the next run resumes it." if kept else "")) from e

    actual = hasher.hexdigest() if os.path.getsize(part) == done else _sha256_of(part)
    if actual != expected:
        try:
            os.remove(part)
        except OSError:
            pass
        raise ModelFetchError(
            f"[Radiance] {label}: CHECKSUM MISMATCH (expected {expected}, got {actual}). "
            "The download was deleted; nothing was installed.")
    os.replace(part, dest)
    _remember(dest, expected)
    logger.info("[Radiance] %s verified (sha256 %s…) and installed at %s", label, expected[:12], dest)
    return dest
