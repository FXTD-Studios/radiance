"""The repository's own Python files, for the source-scanning tests.

A plain ``root.rglob("*.py")`` also walks local scratch that is git-ignored
and never shipped (``_audit_tmp/`` holding another project's sources,
``_to_delete/``, build output, virtualenvs), and the scans then fail on code
that is not Radiance. Not a test file (no ``test_`` prefix).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

#: Top-level directories that are never part of the package.
SCRATCH_DIRS = frozenset({"_to_delete", "build", "dist", "__pycache__", "node_modules"})


def _is_scratch(rel_parts) -> bool:
    for part in rel_parts[:-1]:
        if part in SCRATCH_DIRS or part.startswith((".", "_audit")):
            return True
    return False


def repo_py_files(root: Path) -> Iterator[Path]:
    """Every ``*.py`` under ``root``, minus scratch and hidden directories."""
    root = Path(root)
    for p in root.rglob("*.py"):
        if not _is_scratch(p.relative_to(root).parts):
            yield p
