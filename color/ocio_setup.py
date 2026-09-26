"""Automatic OpenColorIO setup, run once when Radiance loads.

Nothing to configure by hand. On startup:

1. An ``$OCIO`` the user or studio already set is used as-is (validated,
   never overwritten), exactly as Nuke, Resolve and every OCIO host do.
2. Otherwise Radiance uses OpenColorIO's built-in ACES 2.0 **studio** config
   (every ACES space, every major camera log, Rec.709 / Rec.2020 / P3 / PQ /
   HLG displays). It is written once to ``ACES/studio-config.ocio`` inside the
   package, a plain self-contained file with no LUTs, so tools that only
   accept a path (and other custom nodes) can use it too, and ``$OCIO``
   points at it for this process.
3. With an OpenColorIO older than 2.2 (no built-in configs) the bundled
   ``ACES/config.ocio`` (ACES 2.0 CG config) is used instead.

The chosen config is also made OCIO's process-wide current config, so every
OCIO consumer in ComfyUI resolves the same names. Nothing is downloaded.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

log = logging.getLogger("radiance.ocio")

_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACES_DIR = os.path.join(_PKG_DIR, "ACES")
BUNDLED_CG = os.path.join(ACES_DIR, "config.ocio")
STUDIO_FILE = os.path.join(ACES_DIR, "studio-config.ocio")
BUILTIN_URI = "ocio://studio-config-latest"

#: Result of the last configure_ocio() call, for the log line, routes and tests.
STATE: dict = {"configured": False, "source": "", "path": "", "name": "",
               "colorspaces": 0, "ocio_version": "", "error": ""}


def _write_studio_file(ocio) -> Optional[str]:
    """Serialise the built-in studio config to STUDIO_FILE (once per version)."""
    cfg = ocio.Config.CreateFromBuiltinConfig("studio-config-latest")
    text = cfg.serialize()
    try:
        if os.path.isfile(STUDIO_FILE):
            with open(STUDIO_FILE, "r", encoding="utf-8") as f:
                if f.read() == text:
                    return STUDIO_FILE
        os.makedirs(ACES_DIR, exist_ok=True)
        tmp = STUDIO_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, STUDIO_FILE)
        return STUDIO_FILE
    except OSError as exc:
        # A read-only install still works: OCIO accepts the URI directly.
        log.debug("[Radiance OCIO] could not write %s (%s); using %s", STUDIO_FILE, exc, BUILTIN_URI)
        return None


def configure_ocio(environ: Optional[dict] = None) -> dict:
    """Pick, validate and activate the OCIO config. Safe to call repeatedly."""
    env = os.environ if environ is None else environ
    STATE.update(configured=False, source="", path="", name="", colorspaces=0, error="")
    try:
        import PyOpenColorIO as ocio  # type: ignore
    except Exception as exc:  # noqa: BLE001
        STATE["error"] = f"OpenColorIO not installed ({exc}). pip install opencolorio"
        return STATE
    STATE["ocio_version"] = ocio.GetVersion()

    chosen = None
    user = (env.get("OCIO") or "").strip()
    if user:
        try:
            cfg = ocio.Config.CreateFromFile(user)
            cfg.validate()
            chosen, source, path = cfg, "$OCIO (yours)", user
        except Exception as exc:  # noqa: BLE001
            log.warning("[Radiance OCIO] $OCIO=%s is not a usable config (%s); "
                        "falling back to the ACES studio config for Radiance.", user, exc)

    if chosen is None:
        try:
            cfg = ocio.Config.CreateFromBuiltinConfig("studio-config-latest")
            path = _write_studio_file(ocio) or BUILTIN_URI
            chosen, source = cfg, "built-in ACES studio config"
            if not user:
                env["OCIO"] = path
        except Exception:  # noqa: BLE001 - OCIO < 2.2 has no built-ins
            if os.path.isfile(BUNDLED_CG):
                cfg = ocio.Config.CreateFromFile(BUNDLED_CG)
                chosen, source, path = cfg, "bundled ACES CG config", BUNDLED_CG
                if not user:
                    env["OCIO"] = BUNDLED_CG

    if chosen is None:
        STATE["error"] = "no OCIO config available"
        return STATE

    try:
        ocio.SetCurrentConfig(chosen)
    except Exception as exc:  # noqa: BLE001
        log.debug("[Radiance OCIO] SetCurrentConfig failed: %s", exc)
    STATE.update(configured=True, source=source, path=path, name=chosen.getName(),
                 colorspaces=len(list(chosen.getColorSpaceNames())))
    return STATE


def summary() -> str:
    if not STATE["configured"]:
        return f"OCIO not configured: {STATE['error'] or 'not run'}"
    return (f"OCIO {STATE['ocio_version']}: {STATE['name']} "
            f"({STATE['colorspaces']} colorspaces, {STATE['source']})")


__all__ = ["configure_ocio", "summary", "STATE", "STUDIO_FILE", "BUNDLED_CG", "BUILTIN_URI"]
