"""
test_packaging.py — keep pyproject's package list in sync with the source tree.

The recurring failure mode this guards against: a new sub-package (e.g. a new
nodes/<group>/ directory) is added in code but NOT listed in
[tool.setuptools] packages, so it imports fine from a source checkout but is
silently dropped from the built wheel / Comfy Registry install.

Pure stdlib, no imports of the package itself (so it runs without torch).
"""
import os
import re

_ROOT = os.path.join(os.path.dirname(__file__), "..")

# Directories that contain __init__.py but are intentionally NOT shipped as
# part of the `radiance` import package.
_EXCLUDE_TOP = {"tests", "docs", "build", "dist", "scripts", "examples", "__pycache__"}


def _listed_packages():
    """Parse the `packages = [...]` array from [tool.setuptools] in pyproject.toml."""
    src = open(os.path.join(_ROOT, "pyproject.toml"), encoding="utf-8").read()
    m = re.search(r"\npackages\s*=\s*\[(.*?)\]", src, re.S)
    assert m, "Could not find a `packages = [...]` list in pyproject.toml"
    return {x.strip().strip('"').strip("'") for x in m.group(1).split(",") if x.strip()}


def _disk_packages():
    """Every directory holding an __init__.py, expressed as radiance.<dotted.path>."""
    found = set()
    for dirpath, dirnames, filenames in os.walk(_ROOT):
        dirnames[:] = [d for d in dirnames if d not in _EXCLUDE_TOP and not d.startswith(".")]
        rel = os.path.relpath(dirpath, _ROOT).replace("\\", "/")
        if rel == ".":
            if "__init__.py" in filenames:
                found.add("radiance")
            continue
        top = rel.split("/")[0]
        if top in _EXCLUDE_TOP:
            continue
        if "__init__.py" in filenames:
            found.add("radiance." + rel.replace("/", "."))
    return found


def test_all_source_packages_are_declared():
    """Every importable package dir on disk must be in pyproject's packages list."""
    listed = _listed_packages()
    disk = _disk_packages()
    missing = sorted(disk - listed)
    assert not missing, (
        "These package directories exist on disk but are NOT in pyproject.toml "
        "[tool.setuptools] packages — they would be dropped from the built wheel:\n"
        + "\n".join(f"  {p}" for p in missing)
    )


def test_no_declared_package_is_missing_on_disk():
    """Every package listed in pyproject must still exist (catches stale entries)."""
    listed = _listed_packages()
    disk = _disk_packages()
    stale = sorted(listed - disk)
    assert not stale, (
        "These packages are listed in pyproject.toml but no longer exist on disk "
        "(remove them from the packages list):\n" + "\n".join(f"  {p}" for p in stale)
    )
