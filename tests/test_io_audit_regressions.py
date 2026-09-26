"""Regressions from the 2026-08 full audit: TIFF depth honesty and
directory/glob sequence windowing.
"""
import importlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nodes_io = importlib.import_module("radiance.nodes.io.write")


class TestTiffDepthHonesty:
    """A 16/32-bit TIFF request must never silently produce an 8-bit file."""

    def test_32bit_write_raises_without_tifffile(self, tmp_path, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def _no_tifffile(name, *a, **k):
            if name == "tifffile":
                raise ImportError("No module named 'tifffile'")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", _no_tifffile)
        writer = nodes_io.RadianceWrite()
        with pytest.raises((ImportError, RuntimeError), match="tifffile"):
            writer.write(image=torch.rand(1, 8, 8, 3),
                         output_path=str(tmp_path / "deep"),
                         format="IMG │ TIFF (32-bit float)", overwrite=True)
        produced = list(tmp_path.glob("deep*"))
        assert not produced, "a file was still written after the refusal"

    def test_16bit_write_raises_without_tifffile(self, tmp_path, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def _no_tifffile(name, *a, **k):
            if name == "tifffile":
                raise ImportError("No module named 'tifffile'")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", _no_tifffile)
        writer = nodes_io.RadianceWrite()
        with pytest.raises((ImportError, RuntimeError), match="tifffile"):
            writer.write(image=torch.rand(1, 8, 8, 3),
                         output_path=str(tmp_path / "deep16"),
                         format="IMG │ TIFF (16-bit)", overwrite=True)


class TestWritePathPrediction:
    """/radiance/media/resolve_write drives the Write node's on-node path
    readout. If prediction drifts from what write() actually produces, the UI
    lies about the destination -- these tests write real files and compare."""

    def test_img_prediction_matches_write(self, tmp_path):
        writer = nodes_io.RadianceWrite()
        pred = nodes_io._predict_write_target(
            output_path=str(tmp_path), format="IMG │ EXR (16-bit half)",
            filename="shot", version=3, overwrite=True)
        writer.write(image=torch.rand(1, 8, 8, 3), output_path=str(tmp_path),
                     format="IMG │ EXR (16-bit half)", filename="shot",
                     version=3, overwrite=True)
        produced = [str(p) for p in tmp_path.glob("*.exr")]
        assert produced == [pred["path"]]
        assert "frame 1 only" in pred["note"]

    def test_video_prediction_matches_write(self, tmp_path):
        import shutil
        if shutil.which("ffmpeg") is None:
            pytest.skip("needs ffmpeg")
        writer = nodes_io.RadianceWrite()
        base = str(tmp_path / "clip")
        pred = nodes_io._predict_write_target(
            output_path=base, format="VID │ MP4 (H.264)", overwrite=True)
        writer.write(image=torch.rand(4, 48, 48, 3), output_path=base,
                     format="VID │ MP4 (H.264)", fps=24.0, overwrite=True)
        produced = [str(p) for p in tmp_path.glob("*.mp4")]
        assert produced == [pred["path"]]

    def test_seq_prediction_matches_write(self, tmp_path):
        writer = nodes_io.RadianceWrite()
        base = str(tmp_path / "plate")
        pred = nodes_io._predict_write_target(
            output_path=base, format="SEQ │ EXR (32-bit float)",
            start_frame=1001, frame_padding=4, overwrite=True)
        writer.write(image=torch.rand(2, 8, 8, 3), output_path=base,
                     format="SEQ │ EXR (32-bit float)", start_frame=1001,
                     frame_padding=4, overwrite=True)
        first = pred["path"].replace("####", "1001")
        import os
        assert os.path.isfile(first), (
            f"predicted pattern {pred['path']} does not match written files: "
            f"{sorted(os.listdir(tmp_path / 'plate'))}")

    def test_overwrite_defaults_off(self):
        spec = nodes_io.RadianceWrite.INPUT_TYPES()
        assert spec["optional"]["overwrite"][1]["default"] is False, (
            "overwrite must default to OFF: destroying an existing file has to "
            "be an explicit choice (2026-08 audit)")


class TestSequenceWindowing:
    """Directory/glob reads were sliced by list index with frame-number
    defaults (files[1001:99999]) and always came back empty."""

    @pytest.fixture()
    def exr_dir(self, tmp_path):
        writer = nodes_io.RadianceWrite()
        frames = torch.stack(
            [torch.full((8, 8, 3), i / 10.0) for i in range(5)])
        writer.write(image=frames, output_path=str(tmp_path / "sq"),
                     format="SEQ │ EXR (32-bit float)", overwrite=True)
        return tmp_path / "sq"   # sq_1001.exr .. sq_1005.exr

    def test_directory_read_with_default_window(self, exr_dir):
        reader = nodes_io.RadianceRead()
        out = reader.read(path=str(exr_dir))[0]
        assert out.shape[0] == 5

    def test_directory_read_explicit_window_is_frame_numbers(self, exr_dir):
        reader = nodes_io.RadianceRead()
        out = reader.read(path=str(exr_dir), start_frame=1002, end_frame=1004)[0]
        assert out.shape[0] == 3
        means = [round(float(out[i].mean()), 2) for i in range(3)]
        assert means == [0.1, 0.2, 0.3], (
            f"window selected the wrong frames: {means}")

    def test_glob_read(self, exr_dir):
        reader = nodes_io.RadianceRead()
        out = reader.read(path=str(exr_dir / "sq_*.exr"))[0]
        assert out.shape[0] == 5

    def test_window_helper_unnumbered_files(self):
        files = ["a.png", "b.png", "c.png"]
        assert nodes_io._window_listed_files(files, 1001, 99999, 1) == files
        assert nodes_io._window_listed_files(files, 1001, 99999, 2) == ["a.png", "c.png"]

    def test_window_helper_out_of_range_falls_back(self):
        files = [f"f_{n:04d}.exr" for n in (1, 2, 3)]
        assert nodes_io._window_listed_files(files, 1001, 99999, 1) == files
