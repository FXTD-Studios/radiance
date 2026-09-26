"""One version everywhere a user or the registry can see it.

The Comfy Registry publishes pyproject's version; the startup log, EXR
metadata and node reports print config.constants.VERSION; the README badge and
the CHANGELOG are what a user reads. Before 3.5.0 they disagreed, and node
descriptions and reports still said 3.0.0 to 3.2.2.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _pyproject_version() -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    return re.search(r'^version\s*=\s*"([^"]+)"', text, re.M).group(1)


def _constants_version() -> str:
    text = (ROOT / "config" / "constants.py").read_text(encoding="utf-8")
    return re.search(r'^VERSION\s*=\s*"([^"]+)"', text, re.M).group(1)


def test_pyproject_constants_package_json_agree():
    v = _pyproject_version()
    assert _constants_version() == v
    assert json.loads((ROOT / "package.json").read_text(encoding="utf-8"))["version"] == v


def test_changelog_top_release_is_this_version():
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    first = re.search(r"^## \[(\d+\.\d+\.\d+)\]", text, re.M).group(1)
    assert first == _pyproject_version()


def test_readme_badge_is_this_version():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"badge/version-{_pyproject_version()}-" in text


def test_no_module_pins_its_own_release_number():
    """Module __version__ strings drifted (3.1.0, 3.2.2); they import VERSION now."""
    offenders = []
    for p in ROOT.rglob("*.py"):
        rel = p.relative_to(ROOT).as_posix()
        if rel.startswith(("tests/", "scripts/")):
            continue
        if re.search(r'^__version__\s*=\s*"\d', p.read_text(encoding="utf-8", errors="ignore"), re.M):
            offenders.append(rel)
    assert not offenders, offenders
