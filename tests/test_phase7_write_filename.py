"""E-08: RadianceWrite's filename is joined onto output_path and must not leave it."""
import pytest

from radiance.io.writer import _check_filename_stays_inside


@pytest.mark.parametrize("name", ["shot_v0001", "sh010_v0001", "shots/sh010_v0001", "a.b_v0001"])
def test_plain_names_and_subfolders_are_allowed(name):
    _check_filename_stays_inside(name)


@pytest.mark.parametrize("name", [
    "../../../tmp/x_v0001",
    "../x_v0001",
    "shots/../../x_v0001",
    "/tmp/x_v0001",
    "x\x00y_v0001",
])
def test_names_that_leave_output_path_are_refused(name):
    with pytest.raises(ValueError, match="leaves output_path"):
        _check_filename_stays_inside(name)


def test_write_frames_checks_before_building_the_path():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "io" / "writer.py").read_text(encoding="utf-8")
    i = src.index('_check_filename_stays_inside(f"{filename}_{ver_str}")')
    j = src.index('output_path = str(Path(output_path) / f"{filename}_{ver_str}")')
    assert i < j
