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
    if sys.platform == "win32":
        # fusionscript must find this worker's Python DLL, including in Conda.
        os.environ["PYTHONHOME"] = sys.base_prefix
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
        if sequence_pattern:
            items = pool.ImportMedia([{"FilePath": sequence_pattern, "StartIndex": first, "EndIndex": last}])
        else:
            items = pool.ImportMedia(paths)
        if not items:
            return "not imported: Resolve refused the files (see its console)"
        return f"imported {len(items)} clip(s) into '{project.GetName()}'"
    except Exception as e:  # noqa: BLE001 - scripting API errors are opaque
        return f"not imported: {e}"


if __name__ == "__main__":
    request = json.load(sys.stdin)
    result = import_into_resolve(**request)
    print("RADIANCE_RESOLVE_RESULT:" + json.dumps(result), flush=True)
