"""tools/check_release_ready.py must run and pass on the tree as committed.

It had crashed since `license` moved to an SPDX string, and it still looked
for README headings that were renamed, so the one release gate nobody ran
had silently stopped working. Running it here keeps it honest.
"""
import runpy
import pathlib

_TOOL = pathlib.Path(__file__).resolve().parent.parent / "tools" / "check_release_ready.py"


def test_release_checks_pass(capsys):
    ns = runpy.run_path(str(_TOOL), run_name="radiance_release_check")
    assert ns["main"]() == 0, capsys.readouterr().out
