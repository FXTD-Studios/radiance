"""One place that decides whether Radiance may fetch model weights.

Several nodes download their weights on first use: Real-ESRGAN, HAT-L, SwinIR,
Depth Anything V2, DSINE. Between them that is 67 MB to 2.4 GB, and it used to
start the moment someone queued a graph — no prompt, no confirmation, and on a
metered or air-gapped machine no way to see it coming until the bandwidth was
already spent.

The gates were also inconsistent: `nodes/upscale` honoured an opt-*out*
(`RADIANCE_UPSCALE_OFFLINE=1`), while `nodes/vfx/multipass` had no gate at all.
Two mechanisms, one of them missing, both invisible from the node UI.

This module is the single decision point. Since 3.5.0 the default is
**download on first use**: every model a node needs is fetched automatically
from a pinned source and checked against its SHA-256 before it is installed
(see `radiance.core.model_fetch`). The operator can always say no:

    RADIANCE_ALLOW_DOWNLOADS=0   never download; missing models raise a
                                 message naming the file, size and folder
    HF_HUB_OFFLINE=1 or TRANSFORMERS_OFFLINE=1   the machine is offline
    RADIANCE_UPSCALE_OFFLINE=1   legacy opt-out, still honoured for upscale
    RADIANCE_LOADER_OFFLINE=1    legacy opt-out, still honoured for Read Models

Nodes with their own switch (Read Models `auto_download`, AI Upscale
`auto_download`, Multipass Estimate `download_missing_models`) also say no
when it is off.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger("radiance.consent")

ALLOW_ENV = "RADIANCE_ALLOW_DOWNLOADS"
LEGACY_UPSCALE_OFFLINE_ENV = "RADIANCE_UPSCALE_OFFLINE"
LEGACY_LOADER_OFFLINE_ENV = "RADIANCE_LOADER_OFFLINE"

_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")


def _flag(name: str) -> Optional[bool]:
    """Tri-state read of an environment flag: True, False, or unset."""
    raw = os.environ.get(name)
    if raw is None:
        return None
    raw = raw.strip().lower()
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    return None


def downloads_allowed(
    *, legacy_offline_env: Optional[str] = None, default: bool = True,
) -> bool:
    """True when the operator has consented to fetching model weights.

    *legacy_offline_env* names an older opt-out variable to keep honouring, so
    existing studio configs do not silently start downloading again.

    *default* is the answer when nothing is set: download (3.5.0). A caller
    can pass False to keep a download ask-first.
    """
    explicit = _flag(ALLOW_ENV)
    if explicit is not None:
        return explicit

    if legacy_offline_env and _flag(legacy_offline_env) is True:
        return False       # the old opt-out said no; that still means no

    if _flag("HF_HUB_OFFLINE") is True or _flag("TRANSFORMERS_OFFLINE") is True:
        return False       # the machine is declared offline

    return bool(default)   # default: download on first use


def refusal_message(
    what: str,
    size_mb: float | int | None = None,
    dest: str | None = None,
    url: str | None = None,
) -> str:
    """The message shown when a download is refused.

    Actionable rather than apologetic: it names what was wanted, how big it is,
    where to put it, and the one setting that changes the answer.
    """
    size = f" (~{size_mb} MB)" if size_mb else ""
    lines = [
        f"[Radiance] '{what}'{size} is not installed and automatic downloads are off",
        f"           ({ALLOW_ENV}=0, HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1 or a legacy *_OFFLINE=1 is set).",
        "           Unset it to let Radiance fetch the file, or install it manually:",
    ]
    if url:
        lines.append(f"             from: {url}")
    if dest:
        lines.append(f"             to:   {dest}")
    return "\n".join(lines)


def require_consent(
    what: str,
    size_mb: float | int | None = None,
    dest: str | None = None,
    url: str | None = None,
    *,
    legacy_offline_env: Optional[str] = None,
) -> bool:
    """Check consent and log the refusal if there is none.

    Returns True when the caller may proceed with the download.
    """
    if downloads_allowed(legacy_offline_env=legacy_offline_env):
        return True
    logger.error(refusal_message(what, size_mb=size_mb, dest=dest, url=url))
    return False
