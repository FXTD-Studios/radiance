"""Lightweight release hygiene checks for Radiance.

Python 3.10 needs the tomli backport from the test extras. Newer Python versions
use only the standard library.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
GENERATED_PATTERNS = (
    "core/*.db",
    "workflows/.versions/**",
    "workflows/GENERAL/dashboard_*.rad",
    "workflows/GENERAL/test_*_v*.rad",
    "**/__pycache__",
    "**/*.pyc",
)


def main() -> int:
    errors: list[str] = []
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject.get("project", {})
    comfy = pyproject.get("tool", {}).get("comfy", {})

    _check_project_metadata(project, comfy, errors)
    _check_version_and_count(project, errors)
    _check_readme(errors)
    _check_package_paths(pyproject, errors)
    _check_generated_files(errors)

    if errors:
        print("Release checks failed:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print("Release checks passed.")
    return 0


def _check_project_metadata(project: dict, comfy: dict, errors: list[str]) -> None:
    name = str(project.get("name", ""))
    version = str(project.get("version", ""))

    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9._-]{0,98}", name):
        errors.append(f"invalid project.name: {name!r}")
    if any(token in name for token in ("..", "--", "__")):
        errors.append(f"project.name has consecutive special characters: {name!r}")
    if not re.fullmatch(r"\d+\.\d+\.\d+(?:[-+][A-Za-z0-9.]+)?", version):
        errors.append(f"version is not semantic: {version!r}")

    readme = project.get("readme")
    if not readme or not (ROOT / str(readme)).is_file():
        errors.append("project.readme is missing or does not exist")

    # PEP 639: `license` is an SPDX string and `license-files` names the text.
    # The older table form ({file = ...}) is still accepted.
    license_value = project.get("license")
    if isinstance(license_value, dict):
        license_files = [license_value.get("file")] if license_value.get("file") else []
    else:
        if not license_value:
            errors.append("project.license (SPDX expression) is missing")
        license_files = list(project.get("license-files", []))
    if not license_files:
        errors.append("no licence file declared (project.license-files)")
    for rel in license_files:
        if not (ROOT / str(rel)).is_file():
            errors.append(f"licence file does not exist: {rel}")

    for key in ("PublisherId", "DisplayName"):
        if not comfy.get(key):
            errors.append(f"[tool.comfy].{key} is required")


def _check_version_and_count(project: dict, errors: list[str]) -> None:
    """pyproject, the runtime constant and the README badge must agree."""
    constants = (ROOT / "config" / "constants.py").read_text(encoding="utf-8")
    runtime = re.search(r'^VERSION\s*=\s*"([^"]+)"', constants, re.M)
    if not runtime or runtime.group(1) != str(project.get("version")):
        errors.append(
            f"config/constants.py VERSION {runtime.group(1) if runtime else None!r} "
            f"!= pyproject version {project.get('version')!r}"
        )
    count = re.search(r"^EXPECTED_MIN_NODE_COUNT\s*=\s*(\d+)", constants, re.M)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    badge = re.search(r"badge/nodes-(\d+)-", readme)
    if count and badge and badge.group(1) != count.group(1):
        errors.append(f"README node badge says {badge.group(1)}, EXPECTED_MIN_NODE_COUNT is {count.group(1)}")


def _check_readme(errors: list[str]) -> None:
    readme_path = ROOT / "README.md"
    if not readme_path.is_file():
        errors.append("README.md is missing")
        return

    readme = readme_path.read_text(encoding="utf-8")
    banned_claims = ("150 nodes", "121 Professional Nodes", "zero risk")
    for claim in banned_claims:
        if claim.lower() in readme.lower():
            errors.append(f"README contains stale or unsafe claim: {claim!r}")

    required_sections = ("## Installation", "## Quick start", "## Example workflows",
                         "## Using the Viewer", "## Before you start",
                         "## Documentation", "## Support", "## License")
    for section in required_sections:
        if section not in readme:
            errors.append(f"README missing section: {section}")


def _check_package_paths(pyproject: dict, errors: list[str]) -> None:
    setuptools = pyproject.get("tool", {}).get("setuptools", {})
    for package_name in setuptools.get("packages", []):
        rel = str(package_name).removeprefix("radiance").replace(".", "/").strip("/")
        package_path = ROOT / rel if rel else ROOT
        if not (package_path / "__init__.py").is_file():
            errors.append(f"package listed but missing: {package_name}")
    # A package-data entry that matches nothing is a file that was removed
    # without the manifest being told.
    for pattern in setuptools.get("package-data", {}).get("radiance", []):
        if not any(ROOT.glob(pattern)) and "TEMPLATES" not in pattern:
            errors.append(f"package-data entry matches no file: {pattern}")


def _tracked_files() -> list[str] | None:
    """Files git tracks, or None outside a checkout."""
    import subprocess
    try:
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.splitlines()


def _check_generated_files(errors: list[str]) -> None:
    """Generated files must not be committed (in a checkout) or present (in an unpacked archive).

    Local __pycache__ in a working tree is gitignored and never ships, so a
    checkout is judged by what git tracks rather than by what is on disk.
    """
    import fnmatch
    tracked = _tracked_files()
    for pattern in GENERATED_PATTERNS:
        if tracked is not None:
            flat = pattern.replace("**/", "")
            matches = [f for f in tracked
                       if fnmatch.fnmatch(f, pattern) or fnmatch.fnmatch(f.rsplit("/", 1)[-1], flat)
                       or ("/" + flat.strip("*") + "/") in ("/" + f)]
        else:
            matches = [str(path.relative_to(ROOT)) for path in ROOT.glob(pattern) if path.exists()]
        if matches:
            errors.append(f"generated files present for {pattern}: {', '.join(matches[:5])}")


if __name__ == "__main__":
    raise SystemExit(main())
