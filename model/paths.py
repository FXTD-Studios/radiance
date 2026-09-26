"""Where Radiance looks for its own checkpoints.

One resolver for every learned model the suite ships or expects the user to
install (the pixel SDR→HDR network, the temporal residual model, future
weights). It answers two questions:

* :func:`radiance_model_dirs` -- every directory that may hold a Radiance
  checkpoint, in search order.
* :func:`find_radiance_checkpoint` -- the first file matching a set of names
  or globs across those directories, or ``None``.

Search order
------------
1. An explicit path, when the caller has one (widget value).
2. An environment variable, when the caller names one.
3. Every ``radiance`` folder ComfyUI knows about: ``models/radiance`` under
   ``folder_paths.models_dir`` plus anything ``extra_model_paths.yaml`` adds
   under a ``radiance:`` key. The folder is registered with ``folder_paths``
   on first use so the yaml key works without the user having to know the
   registration exists.
4. ``<ComfyUI>/models/radiance`` derived from this package's location, for
   test harnesses and scripts that run without ComfyUI.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

logger = logging.getLogger("radiance.model.paths")

FOLDER_KEY = "radiance"
_SUPPORTED_EXTENSIONS = {".pt", ".pth", ".safetensors", ".ckpt"}

_registered = False


def _package_models_dir() -> Path:
    # <ComfyUI>/custom_nodes/radiance/model/paths.py -> <ComfyUI>/models/radiance
    return Path(__file__).resolve().parents[3] / "models" / FOLDER_KEY


def _register_with_folder_paths() -> None:
    """Register ``models/radiance`` with ComfyUI once, if ComfyUI is present."""
    global _registered
    if _registered:
        return
    _registered = True
    try:
        import folder_paths  # type: ignore
    except ImportError:
        return
    try:
        models_dir = getattr(folder_paths, "models_dir", None)
        if not isinstance(models_dir, str) or not models_dir:
            return
        default_dir = os.path.join(models_dir, FOLDER_KEY)
        add = getattr(folder_paths, "add_model_folder_path", None)
        if callable(add):
            try:
                add(FOLDER_KEY, default_dir, is_default=True)
            except TypeError:  # older ComfyUI without is_default
                add(FOLDER_KEY, default_dir)
        table = getattr(folder_paths, "folder_names_and_paths", None)
        if isinstance(table, dict) and FOLDER_KEY in table:
            entry = table[FOLDER_KEY]
            if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[1], set):
                entry[1].update(_SUPPORTED_EXTENSIONS)
    except Exception as exc:  # noqa: BLE001 - registration is best effort
        logger.debug("[Radiance] models/radiance registration skipped: %s", exc)


def radiance_model_dirs() -> List[Path]:
    """Every directory that may hold a Radiance checkpoint, in search order."""
    _register_with_folder_paths()
    dirs: List[Path] = []
    seen = set()

    def _add(candidate) -> None:
        try:
            p = Path(candidate).expanduser()
        except TypeError:
            return
        key = str(p).lower() if os.name == "nt" else str(p)
        if key in seen:
            return
        seen.add(key)
        dirs.append(p)

    try:
        import folder_paths  # type: ignore
        get = getattr(folder_paths, "get_folder_paths", None)
        if callable(get):
            for d in get(FOLDER_KEY) or ():
                _add(d)
        models_dir = getattr(folder_paths, "models_dir", None)
        if isinstance(models_dir, str) and models_dir:
            _add(os.path.join(models_dir, FOLDER_KEY))
    except Exception:  # noqa: BLE001 - no ComfyUI, or a stubbed folder_paths
        pass

    _add(_package_models_dir())
    return dirs


def find_radiance_checkpoint(
    patterns: Sequence[str],
    explicit_path: str = "",
    env_var: Optional[str] = None,
) -> Optional[Path]:
    """Return the first checkpoint matching ``patterns`` or ``None``.

    ``patterns`` are file names or ``fnmatch`` globs, tried in order inside
    each search directory (so an exact preferred name wins over a wildcard).
    """
    if explicit_path and explicit_path.strip():
        p = Path(explicit_path.strip()).expanduser()
        if p.is_file():
            return p.resolve()
        logger.warning("[Radiance] checkpoint path %s does not exist; searching defaults", p)
    if env_var:
        env_value = os.environ.get(env_var, "").strip()
        if env_value:
            p = Path(env_value).expanduser()
            if p.is_file():
                return p.resolve()
            logger.warning("[Radiance] %s=%s does not exist; searching defaults", env_var, p)
    for directory in radiance_model_dirs():
        if not directory.is_dir():
            continue
        for pattern in patterns:
            if any(ch in pattern for ch in "*?["):
                matches = sorted(directory.glob(pattern))
                matches = [m for m in matches if m.is_file()]
                if matches:
                    return matches[0].resolve()
            else:
                candidate = directory / pattern
                if candidate.is_file():
                    return candidate.resolve()
    return None


def describe_search(patterns: Iterable[str], env_var: Optional[str] = None) -> str:
    """A one-line, user-facing description of where a checkpoint is looked for."""
    dirs = ", ".join(str(d) for d in radiance_model_dirs()[:2]) or "models/radiance"
    names = ", ".join(patterns)
    env = f" or {env_var}" if env_var else ""
    return f"{names} in {dirs}{env}"


__all__ = [
    "FOLDER_KEY",
    "radiance_model_dirs",
    "find_radiance_checkpoint",
    "describe_search",
]
