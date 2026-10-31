"""Isolated Resolve import worker: native scripting failures must not kill ComfyUI."""
from __future__ import annotations

import json
import os
import sys
from typing import List, Optional

def _resolve_module_dirs() -> List[str]:
    """Where Blackmagic installs DaVinciResolveScript.py on each platform."""
    dirs = []
    env = os.environ.get("RESOLVE_SCRIPT_API")
    if env:
        dirs.append(os.path.join(env, "Modules"))
    if sys.platform.startswith("win"):
        base = os.environ.get("PROGRAMDATA", r"C:\ProgramData")
        dirs.append(os.path.join(base, "Blackmagic Design", "DaVinci Resolve", "Support",
                                 "Developer", "Scripting", "Modules"))
    elif sys.platform == "darwin":
        dirs.append("/Library/Application Support/Blackmagic Design/DaVinci Resolve/"
                    "Developer/Scripting/Modules")
    else:
        dirs += ["/opt/resolve/Developer/Scripting/Modules",
                 "/home/resolve/Developer/Scripting/Modules"]
    return dirs


def _resolve_app():
    """(resolve, None) for the running Resolve, or (None, why)."""
    # On Windows, PYTHONHOME is passed in by the caller's spawn env
    # (studio_integrations.import_into_resolve) so fusionscript finds this
    # worker's Python DLL, including in Conda.
    try:
        import DaVinciResolveScript as dvr  # type: ignore
    except ImportError:
        dvr = None
        for d in _resolve_module_dirs():
            if os.path.isfile(os.path.join(d, "DaVinciResolveScript.py")):
                if d not in sys.path:
                    sys.path.append(d)
                try:
                    import DaVinciResolveScript as dvr  # type: ignore  # noqa: F811
                    break
                except ImportError:
                    dvr = None
        if dvr is None:
            return None, ("DaVinci Resolve's scripting module was not found (is Resolve installed? "
                          "RESOLVE_SCRIPT_API can point at its Developer/Scripting folder)")
    try:
        resolve = dvr.scriptapp("Resolve")
    except Exception as e:  # noqa: BLE001 - fusionscript raises all kinds
        return None, f"could not reach Resolve ({e})"
    if resolve is None:
        return None, ("Resolve is not running, or external scripting is off "
                      "(Preferences > System > General > External scripting using: Local)")
    return resolve, None


def _count_present(paths: List[str], first: int, last: int, sequence_pattern: Optional[str]) -> int:
    """How many of the files to import exist on this machine."""
    if not sequence_pattern:
        return sum(1 for p in paths if os.path.isfile(p))
    import re
    m = re.search(r"%0?(\d*)d", sequence_pattern)
    count = 0
    for f in range(first, last + 1):
        name = (sequence_pattern[:m.start()] + str(f).zfill(int(m.group(1) or 0)) + sequence_pattern[m.end():]
                if m else sequence_pattern)
        count += os.path.isfile(name)
    return count


def _clip_frames(item) -> Optional[int]:
    """The frame count Resolve gives the imported clip, or None if it will not say."""
    try:
        v = item.GetClipProperty("Frames")
        return int(str(v).strip()) if str(v).strip() else None
    except Exception:  # noqa: BLE001 - scripting API errors are opaque
        return None


def import_into_resolve(paths: List[str], first: int, last: int, sequence_pattern: Optional[str]) -> str:
    """Import the written files into the current project's Media Pool.
    Returns a status line; never raises."""
    resolve, why = _resolve_app()
    if resolve is None:
        return f"not imported: {why}"
    try:
        project = resolve.GetProjectManager().GetCurrentProject()
        if project is None:
            return "not imported: no project is open in Resolve"
        pool = project.GetMediaPool()
        # E-04: check on this machine what Resolve is about to read, and say
        # what it took, instead of a bare "imported N clips".
        expected = (last - first + 1) if sequence_pattern else len(paths)
        on_disk = _count_present(paths, first, last, sequence_pattern)
        if on_disk == 0:
            where = sequence_pattern or (paths[0] if paths else "")
            return f"not imported: none of the files exist here ({where})"
        if sequence_pattern:
            items = pool.ImportMedia([{"FilePath": sequence_pattern, "StartIndex": first, "EndIndex": last}])
        else:
            items = pool.ImportMedia(paths)
        if not items:
            return "not imported: Resolve refused the files (see its console)"
        frames = _clip_frames(items[0]) if sequence_pattern else None
        detail = f"{on_disk}/{expected} file(s) on disk"
        if frames is not None:
            detail += f", Resolve reports {frames} frame(s)"
            if frames != expected:
                detail += f" (expected {expected})"
        return (f"imported {len(items)} clip(s) into '{project.GetName()}' from "
                f"{sequence_pattern or paths[0]}: {detail}")
    except Exception as e:  # noqa: BLE001 - scripting API errors are opaque
        return f"not imported: {e}"


if __name__ == "__main__":
    request = json.load(sys.stdin)
    result = import_into_resolve(**request)
    print("RADIANCE_RESOLVE_RESULT:" + json.dumps(result), flush=True)
