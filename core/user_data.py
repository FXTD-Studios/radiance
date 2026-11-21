"""Where Radiance keeps what users make: outside the package folder.

A Registry or Manager update replaces the package folder, and everything a user
had saved inside it went with it: the workflow library (projects, versions,
asset bins, thumbnails) and saved gizmos. They now live in ComfyUI's user
directory, ``ComfyUI/user/radiance/``:

    user/radiance/workflows/            the library; official/ is a synced copy
    user/radiance/gizmos/               saved gizmos

The first start after an update copies what the old layout held into the new
one (never overwriting, never deleting the originals). The shipped library
cards are synced into ``workflows/official/`` whenever the package changes, so
the library can keep reading one folder.

Outside ComfyUI (tests, scripts) there is no user directory and the package
folders are used, as before.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("radiance.core.user_data")

MIGRATED_MARK = ".radiance_migrated"
OFFICIAL_MANIFEST = ".radiance_official.json"


def user_root() -> Optional[Path]:
    """``ComfyUI/user/radiance``, or None outside ComfyUI or when it cannot be made."""
    if os.environ.get("RADIANCE_USER_DATA_IN_PACKAGE", "").strip() in ("1", "true", "yes"):
        return None
    try:
        import folder_paths   # ComfyUI's
        getter = getattr(folder_paths, "get_user_directory", None)
        base = getter() if callable(getter) else None
    except Exception:  # noqa: BLE001 - no ComfyUI
        return None
    if not isinstance(base, (str, os.PathLike)) or not str(base):
        return None
    root = Path(base) / "radiance"
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("[Radiance] cannot use %s for user data (%s); keeping it in the package.", root, exc)
        return None
    return root


def _copy_missing(src: Path, dst: Path, skip=lambda rel: False) -> List[str]:
    """Copy every file under ``src`` to the same place under ``dst`` unless it is
    already there. Returns the relative paths copied."""
    copied = []
    for f in sorted(src.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(src)
        if skip(rel):
            continue
        target = dst / rel
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, target)
        copied.append(rel.as_posix())
    return copied


def _shipped_template(rel: Path) -> bool:
    """The package's own workflows/*.json (ComfyUI templates) and official/ cards
    are shipped, not user content. ``_assets_bins.json`` and other ``_*.json`` are."""
    parts = rel.parts
    if parts and parts[0] == "official":
        return True
    return len(parts) == 1 and rel.suffix == ".json" and not rel.name.startswith("_")


def migrate_once(package_dir: Path, user_dir: Path, *, shipped=lambda rel: False) -> List[str]:
    """First start with the new layout: copy what users saved in the package folder."""
    user_dir.mkdir(parents=True, exist_ok=True)
    mark = user_dir / MIGRATED_MARK
    if mark.exists() or not package_dir.is_dir():
        return []
    copied = _copy_missing(package_dir, user_dir, skip=shipped)
    mark.write_text(json.dumps({"from": str(package_dir), "files": len(copied)}), encoding="utf-8")
    if copied:
        logger.info("[Radiance] moved %d saved file(s) from %s to %s; the originals are left in place.",
                    len(copied), package_dir, user_dir)
    return copied


def sync_official(package_official: Path, library_official: Path) -> bool:
    """Make ``library_official`` the shipped cards: copy new and changed ones, remove
    ones no longer shipped. Skipped when the package's cards have not changed since
    the last sync. Returns True when it synced."""
    if not package_official.is_dir():
        return False
    files = sorted(f for f in package_official.rglob("*") if f.is_file())
    manifest = {f.relative_to(package_official).as_posix(): [f.stat().st_size, f.stat().st_mtime_ns] for f in files}
    stamp = library_official / OFFICIAL_MANIFEST
    try:
        if json.loads(stamp.read_text(encoding="utf-8")) == manifest:
            return False
    except (OSError, ValueError):
        pass
    library_official.mkdir(parents=True, exist_ok=True)
    for rel in manifest:
        src, dst = package_official / rel, library_official / rel
        if not dst.exists() or dst.stat().st_size != src.stat().st_size or dst.read_bytes() != src.read_bytes():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    for f in sorted(library_official.rglob("*"), reverse=True):
        rel = f.relative_to(library_official).as_posix()
        if f.is_file() and rel != OFFICIAL_MANIFEST and rel not in manifest:
            f.unlink()
        elif f.is_dir() and not any(f.iterdir()):
            f.rmdir()
    stamp.write_text(json.dumps(manifest), encoding="utf-8")
    return True


def library_dir(package_workflows: Path) -> Path:
    """The workflow library's folder, migrated and synced; the package's when there
    is no ComfyUI user directory."""
    root = user_root()
    if root is None:
        return package_workflows
    lib = root / "workflows"
    try:
        migrate_once(package_workflows, lib, shipped=_shipped_template)
        sync_official(package_workflows / "official", lib / "official")
    except OSError as exc:
        logger.warning("[Radiance] could not set up the library in %s (%s); using the package folder.", lib, exc)
        return package_workflows
    return lib


def gizmos_dir(package_gizmos: Path) -> Path:
    """Saved gizmos' folder, migrated; the package's when there is no user directory."""
    root = user_root()
    if root is None:
        return package_gizmos
    target = root / "gizmos"
    try:
        migrate_once(package_gizmos, target)
    except OSError as exc:
        logger.warning("[Radiance] could not set up gizmos in %s (%s); using the package folder.", target, exc)
        return package_gizmos
    return target
