"""Saved gizmos live in the package's gizmos/ folder."""
from pathlib import Path

from radiance.nodes import gizmo


def test_gizmos_are_stored_in_the_package_gizmos_folder():
    """Moving nodes_gizmo.py into nodes/ sent saved gizmos to a stray
    nodes/gizmos/, outside the gizmos/*.gizmo rule in .gitignore."""
    package_gizmos = Path(__file__).resolve().parents[1] / "gizmos"
    assert Path(gizmo.GIZMOS_DIR).resolve() == package_gizmos.resolve()
