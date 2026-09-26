"""
Nodes must not write into ComfyUI's install directory.

Two path widgets defaulted to bare relative strings — CDL Export's
`grading/shot_01_output.cdl` and Flipbook GIF's `preview/flipbook.gif` — and
were resolved with `os.path.abspath()`, which anchors to the *process working
directory*. For ComfyUI that is the install root, so running either node on its
default scattered `grading/` and `preview/` folders through the installation.
The test suite exercised both, which is how those directories kept reappearing
in this repository.

A related shape: `_get_models_dir()` guarded on `if base:` rather than on the
type. A test that stubs `folder_paths` with a bare MagicMock makes
`models_dir` a truthy Mock, and joining it produced a literal
`MagicMock/mock.models_dir/<id>/` tree on disk.
"""
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

try:
    import torch as _t
    _HAS_TORCH = isinstance(getattr(_t, "__version__", None), str)
except ImportError:
    _HAS_TORCH = False

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radiance.path_utils import resolve_input_path, resolve_output_path


@pytest.fixture
def comfy_dirs(tmp_path, monkeypatch):
    """Point the stubbed folder_paths at a scratch ComfyUI tree."""
    out = tmp_path / "output"
    inp = tmp_path / "input"
    out.mkdir()
    inp.mkdir()

    fp = sys.modules["folder_paths"]
    monkeypatch.setattr(fp, "get_output_directory", lambda: str(out), raising=False)
    monkeypatch.setattr(fp, "get_input_directory", lambda: str(inp), raising=False)

    # Somewhere else entirely, standing in for the ComfyUI install root.
    cwd = tmp_path / "install_root"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    return {"output": out, "input": inp, "cwd": cwd}


# ─────────────────────────────────────────────────────────────────────────────
#  The resolver
# ─────────────────────────────────────────────────────────────────────────────

class TestResolveOutputPath:

    def test_a_relative_path_lands_under_output(self, comfy_dirs):
        resolved = Path(resolve_output_path("grading/shot_01.cdl"))
        assert resolved == comfy_dirs["output"] / "grading" / "shot_01.cdl"

    def test_it_does_not_follow_the_working_directory(self, comfy_dirs):
        """The bug: abspath() anchored to CWD, i.e. the ComfyUI install root."""
        resolved = Path(resolve_output_path("grading/shot_01.cdl"))
        assert comfy_dirs["cwd"] not in resolved.parents

    def test_an_absolute_path_is_honoured(self, comfy_dirs, tmp_path):
        target = tmp_path / "elsewhere" / "out.cdl"
        assert Path(resolve_output_path(str(target))) == target

    def test_traversal_out_of_output_is_rejected(self, comfy_dirs):
        with pytest.raises(ValueError):
            resolve_output_path("../../etc/passwd")

    def test_quoted_paths_are_accepted(self, comfy_dirs):
        """Windows "Copy as path" wraps in double quotes."""
        resolved = Path(resolve_output_path('"grading/shot.cdl"'))
        assert resolved == comfy_dirs["output"] / "grading" / "shot.cdl"

    def test_empty_resolves_to_the_output_root(self, comfy_dirs):
        assert Path(resolve_output_path("")) == comfy_dirs["output"]


class TestResolveInputPath:

    def test_a_file_in_the_input_dir_is_found(self, comfy_dirs):
        f = comfy_dirs["input"] / "shot.cdl"
        f.write_text("x", encoding="utf-8")
        assert Path(resolve_input_path("shot.cdl")) == f

    def test_the_output_dir_is_searched_too(self, comfy_dirs):
        """Radiance's own exporters write to output/; reading back is common."""
        d = comfy_dirs["output"] / "grading"
        d.mkdir()
        f = d / "shot.cdl"
        f.write_text("x", encoding="utf-8")
        assert Path(resolve_input_path("grading/shot.cdl")) == f

    def test_input_wins_over_output(self, comfy_dirs):
        (comfy_dirs["input"] / "shot.cdl").write_text("in", encoding="utf-8")
        (comfy_dirs["output"] / "shot.cdl").write_text("out", encoding="utf-8")
        assert Path(resolve_input_path("shot.cdl")).read_text(encoding="utf-8") == "in"

    def test_a_missing_file_points_at_the_input_dir(self, comfy_dirs):
        """So the caller's 'not found' names where the user should look."""
        resolved = Path(resolve_input_path("nope.cdl"))
        assert resolved == comfy_dirs["input"] / "nope.cdl"

    def test_an_absolute_path_is_honoured(self, comfy_dirs, tmp_path):
        target = tmp_path / "somewhere" / "in.cdl"
        assert Path(resolve_input_path(str(target))) == target


# ─────────────────────────────────────────────────────────────────────────────
#  The nodes that caused the pollution
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _HAS_TORCH, reason="needs real torch")
class TestNodesWriteWhereTheyShould:

    def test_cdl_export_on_its_default_writes_to_output(self, comfy_dirs):
        from radiance.nodes.color.cdl import RadianceCDLExport

        default = RadianceCDLExport.INPUT_TYPES()["required"]["file_path"][1]["default"]
        (written,) = RadianceCDLExport().save(
            default, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0
        )

        assert Path(written) == comfy_dirs["output"] / "grading" / "shot_01_output.cdl"
        assert Path(written).is_file()
        assert not (comfy_dirs["cwd"] / "grading").exists(), \
            "the node wrote into the ComfyUI install directory"

    def test_cdl_roundtrips_through_the_default_paths(self, comfy_dirs):
        """Export then Import with stock widget values must find the file."""
        from radiance.nodes.color.cdl import RadianceCDLExport, RadianceCDLImport

        RadianceCDLExport().save(
            "grading/shot_01.cdl", 1.25, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0
        )

        default = RadianceCDLImport.INPUT_TYPES()["required"]["file_path"][1]["default"]
        result = RadianceCDLImport().load(default)

        assert result[1] == pytest.approx(1.25), "the exported CDL was not found"

    def test_flipbook_gif_on_its_default_writes_to_output(self, comfy_dirs):
        pytest.importorskip("PIL")
        from radiance.nodes.monitor.realtime import RadianceFlipbookGIF

        default = RadianceFlipbookGIF.INPUT_TYPES()["required"]["save_path"][1]["default"]
        images = _t.zeros(2, 8, 8, 3)
        _, status = RadianceFlipbookGIF().export_gif(images, default, 12.0, 480)

        assert (comfy_dirs["output"] / "preview" / "flipbook.gif").is_file(), status
        assert not (comfy_dirs["cwd"] / "preview").exists(), \
            "the node wrote into the ComfyUI install directory"


# ─────────────────────────────────────────────────────────────────────────────
#  Mock-shaped models_dir
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _HAS_TORCH, reason="needs real torch")
class TestModelsDirGuard:

    def test_a_mocked_models_dir_does_not_create_directories(self, tmp_path, monkeypatch):
        from radiance.nodes.upscale.upscale import _get_models_dir

        fp = sys.modules["folder_paths"]
        monkeypatch.setattr(fp, "models_dir", MagicMock(), raising=False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path, raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.chdir(tmp_path)

        result = _get_models_dir("upscale_models")

        assert "MagicMock" not in result
        assert not (tmp_path / "MagicMock").exists()

    def test_multipass_models_dir_has_the_same_guard(self, tmp_path, monkeypatch):
        from radiance.nodes.vfx.multipass.core import _get_comfy_models_dir

        fp = sys.modules["folder_paths"]
        monkeypatch.setattr(fp, "models_dir", MagicMock(), raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.chdir(tmp_path)

        result = _get_comfy_models_dir("normal_estimation")

        assert "MagicMock" not in result
        assert not (tmp_path / "MagicMock").exists()


# ─────────────────────────────────────────────────────────────────────────────
#  Ratchet
# ─────────────────────────────────────────────────────────────────────────────

_WRITABLE_SUFFIXES = (
    ".cdl", ".cc", ".ccc", ".gif", ".exr", ".png", ".jpg", ".jpeg", ".tif",
    ".tiff", ".dpx", ".mp4", ".mov", ".mxf", ".json", ".csv", ".txt", ".cube",
)

# Node key -> the resolver its handler routes through. Anything not listed
# here must not default a writable path to a bare relative string.
_ANCHORED = {
    "RadianceCDLExport": "resolve_output_path",
    "RadianceCDLImport": "resolve_input_path",
    "RadianceFlipbookGIF": "resolve_output_path",
}


@pytest.mark.skipif(not _HAS_TORCH, reason="needs the full catalog")
def test_no_node_defaults_a_writable_path_to_a_bare_relative_string():
    """A relative default is only safe if the handler anchors it.

    New nodes copy an existing node's INPUT_TYPES block. Both offenders here
    came from that. If you add a relative path default, route it through
    `resolve_output_path` / `resolve_input_path` and list the node above.
    """
    import inspect

    import radiance

    if radiance._LOAD_RESULT.failures:
        pytest.skip("catalog incomplete: " + ", ".join(
            f.source.label for f in radiance._LOAD_RESULT.failures))

    offenders = []
    for key, cls in radiance.NODE_CLASS_MAPPINGS.items():
        try:
            spec = cls.INPUT_TYPES()
        except Exception:
            continue

        for group in ("required", "optional"):
            for name, decl in (spec.get(group) or {}).items():
                if not (isinstance(decl, (tuple, list)) and len(decl) == 2):
                    continue
                kind, opts = decl
                if kind != "STRING" or not isinstance(opts, dict):
                    continue
                default = opts.get("default", "")
                if not isinstance(default, str) or not default:
                    continue
                if not default.lower().endswith(_WRITABLE_SUFFIXES):
                    continue
                if os.path.isabs(default) or (len(default) > 1 and default[1] == ":"):
                    continue  # an absolute default is explicit, and honoured

                expected = _ANCHORED.get(key)
                if expected is None:
                    offenders.append(f"{key}.{name} = {default!r} (no resolver)")
                    continue

                try:
                    src = inspect.getsource(getattr(cls, cls.FUNCTION))
                except (OSError, TypeError, AttributeError):
                    continue
                if expected not in src:
                    offenders.append(
                        f"{key}.{name} = {default!r} (handler does not call {expected})"
                    )

    assert not offenders, (
        "relative path defaults resolve against the process working directory, "
        "which for ComfyUI is the install root:\n  " + "\n  ".join(offenders)
    )


@pytest.mark.skipif(not _HAS_TORCH, reason="needs the full catalog")
def test_input_types_has_no_filesystem_side_effects(tmp_path, monkeypatch):
    """Enumerating widgets must not touch the disk.

    ComfyUI calls INPUT_TYPES() on every node at startup and again on every
    /object_info request. `RadianceLUTApply.get_lut_files()` used to
    `os.makedirs(models_dir/luts)` from inside it, so simply loading the
    catalog created a directory — and with a mocked folder_paths, the
    `MagicMock/mock.models_dir/<id>/luts` tree this repository kept growing.
    """
    import radiance

    if radiance._LOAD_RESULT.failures:
        pytest.skip("catalog incomplete: " + ", ".join(
            f.source.label for f in radiance._LOAD_RESULT.failures))

    fp = sys.modules["folder_paths"]
    monkeypatch.setattr(fp, "models_dir", MagicMock(), raising=False)
    monkeypatch.chdir(tmp_path)

    before = set(os.listdir(tmp_path))
    for cls in radiance.NODE_CLASS_MAPPINGS.values():
        try:
            cls.INPUT_TYPES()
        except Exception:
            continue  # a raising enumerator is a different bug
    after = set(os.listdir(tmp_path))

    assert after == before, f"INPUT_TYPES() created {sorted(after - before)}"
