"""RadianceWrite reports its files to ComfyUI history (ported from public PR #18).

An OUTPUT_NODE with no ``ui`` block leaves nothing in ``/history``, so API
clients could not find what Write saved. Previews go only to ``images`` for
formats the browser can show; everything is listed in ``radiance_files``.
"""
import sys
import types

import pytest
import torch


@pytest.fixture()
def w(tmp_path, monkeypatch):
    import radiance.nodes.io.write as mod
    fp = types.SimpleNamespace(get_output_directory=lambda: str(tmp_path))
    monkeypatch.setattr(mod, "_folder_paths", fp, raising=False)
    monkeypatch.setattr(mod, "_HAS_FOLDER_PATHS", True)
    return mod


def _img(n=1):
    return torch.rand(n, 8, 8, 3)


def test_png_in_output_dir_is_previewed(w, tmp_path):
    out = w.RadianceWrite().write(_img(), str(tmp_path / "shots" / "a"), "IMG │ PNG (16-bit)", overwrite=True)
    assert isinstance(out, dict) and "ui" in out
    files = out["ui"]["radiance_files"]
    assert len(files) == 1 and files[0]["type"] == "output" and files[0]["subfolder"] == "shots"
    assert out["ui"]["images"] == files
    saved, count = out["result"]
    assert count == 1 and saved.endswith(files[0]["filename"])


def test_exr_is_listed_not_previewed(w, tmp_path):
    out = w.RadianceWrite().write(_img(), str(tmp_path / "b"), "IMG │ EXR (16-bit half)", overwrite=True)
    assert out["ui"]["radiance_files"][0]["filename"].endswith(".exr")
    assert "images" not in out["ui"]


def test_sequence_lists_every_frame(w, tmp_path):
    out = w.RadianceWrite().write(_img(3), str(tmp_path / "seq" / "c"), "SEQ │ PNG (8-bit)", overwrite=True)
    assert len(out["ui"]["radiance_files"]) == 3


def test_outside_output_dir_is_absolute(w, tmp_path, monkeypatch):
    other = tmp_path / "elsewhere"
    fp = types.SimpleNamespace(get_output_directory=lambda: str(tmp_path / "comfy_out"))
    monkeypatch.setattr(w, "_folder_paths", fp, raising=False)
    out = w.RadianceWrite().write(_img(), str(other / "d"), "IMG │ PNG (8-bit)", overwrite=True)
    e = out["ui"]["radiance_files"][0]
    assert e["type"] == "absolute" and "images" not in out["ui"]


def test_digital_cinema_write_keeps_status(w, tmp_path):
    out = w.RadianceDigitalCinemaWrite().write(_img(), str(tmp_path / "e"), "IMG │ PNG (8-bit)")
    assert isinstance(out, dict) and out["result"][0].startswith("OK:")
    assert out["ui"]["radiance_files"]
