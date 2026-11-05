"""The DCC hand-off nodes write frames safely and report leftovers.

These run without real torch (a small stand-in exposes the four tensor methods
the nodes call), so the light CI lane covers them. They sit beside the
`real_torch` node tests in test_dcc_studio.py, which skip there.
"""
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

RADIANCE_TORCH_GATED = True


class _Tensor:
    """Just enough of a torch tensor for the nodes: detach().cpu().float().numpy()."""

    def __init__(self, a):
        self.a = a
        self.ndim = a.ndim
        self.shape = a.shape

    def detach(self):
        return self

    def cpu(self):
        return self

    def float(self):
        return self

    def numpy(self):
        return self.a


def _frames(n):
    return _Tensor(np.ones((n, 8, 8, 3), np.float32))


def _writes_the_file(*args, **kwargs):
    Path(args[1]).write_bytes(b"x")


def _no_partials(folder):
    return [f.name for f in Path(folder).iterdir() if "partial" in f.name]


@patch("radiance.nodes.pipeline.studio_integrations._save_exr", side_effect=_writes_the_file)
def test_nuke_send_reports_frames_left_by_a_longer_take(_exr, tmp_path, monkeypatch):
    from radiance.nodes.pipeline import studio_integrations as si
    monkeypatch.setattr(si, "_for_format", lambda a, *x, **k: a)
    folder = tmp_path / "out"
    folder.mkdir()
    for n in range(1001, 1009):
        (folder / f"shot.{n}.exr").write_bytes(b"old")

    status, _ = si.RadianceNukeSend().run(
        image=_frames(3), nuke_folder=str(folder), filename="shot",
        frame_start=1001, push_to_nuke=False)

    assert "3 EXR frame(s)" in status
    assert "5 frame(s)" in status and "1004-1008" in status
    assert (folder / "shot.1001.exr").read_bytes() == b"x"      # replaced by the new take
    assert (folder / "shot.1008.exr").read_bytes() == b"old"    # never deleted, only reported
    assert _no_partials(folder) == []


@patch("radiance.nodes.pipeline.studio_integrations._save_exr", side_effect=RuntimeError("disk full"))
def test_a_failed_write_keeps_the_frame_already_on_disk(_exr, tmp_path, monkeypatch):
    from radiance.nodes.pipeline import studio_integrations as si
    monkeypatch.setattr(si, "_for_format", lambda a, *x, **k: a)
    folder = tmp_path / "out"
    folder.mkdir()
    (folder / "shot.1001.exr").write_bytes(b"old master")

    with pytest.raises(RuntimeError):
        si.RadianceNukeSend().run(image=_frames(1), nuke_folder=str(folder), filename="shot",
                                  frame_start=1001, push_to_nuke=False)

    assert (folder / "shot.1001.exr").read_bytes() == b"old master"
    assert _no_partials(folder) == []


@patch("radiance.nodes.pipeline.studio_integrations._save_pil_image", side_effect=_writes_the_file)
def test_davinci_send_lands_on_the_final_name(_pil, tmp_path, monkeypatch):
    from radiance.nodes.pipeline import studio_integrations as si
    monkeypatch.setattr(si, "_for_format", lambda a, *x, **k: a)

    si.RadianceDaVinciSend().run(_frames(1), str(tmp_path), "t", "8bit", 1001)

    assert (tmp_path / "t.png").is_file()
    assert _no_partials(tmp_path) == []


@patch("radiance.nodes.pipeline.dcc._save_exr", side_effect=_writes_the_file)
def test_dcc_bridge_export_reports_frames_left_by_a_longer_take(_exr, tmp_path):
    from radiance.nodes.pipeline import dcc
    out = tmp_path / "bridge"
    out.mkdir()
    for n in range(1001, 1006):
        (out / f"f_{n}.exr").write_bytes(b"old")

    status, _ = dcc.RadianceMCP().run(
        mode="Export Frames", source="Images", target="None", output_path=str(out),
        format="EXR (16-bit half)", images=_frames(2), frame_start=1001, filename_prefix="f")

    assert "2 EXR frames" in status
    assert "3 frame(s)" in status and "1003-1005" in status
    assert _no_partials(out) == []
