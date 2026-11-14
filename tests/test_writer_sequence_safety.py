"""Sequence overwrite safety and pipe deadlines (audit 2026-10-16, A6-A8).

Real writer, real PNG files; only the ffmpeg child in the deadline tests is a
stand-in that never answers.
"""
import os
import sys
import time

import numpy as np
import pytest

from radiance.io import writer as W
from radiance.core import video as V


def _frames(n, v=0.25):
    return [np.full((4, 4, 3), v, np.float32) for _ in range(n)]


def _write(out_dir, frames, *, overwrite, start=1001, count="len", receipt=None):
    return W.dispatch_write(
        iter(frames) if count is None else frames, str(out_dir), "SEQ │ PNG (8-bit)",
        24.0, 18, "ZIP", start, 4, "", overwrite,
        frame_count=None if count is None else len(frames), receipt=receipt)


def _seq_names(d):
    return sorted(p.name for p in d.iterdir() if p.suffix == ".png")


def test_a_shorter_overwrite_reports_the_old_tail(tmp_path):
    d = tmp_path / "shot"
    _write(d, _frames(3), overwrite=True)
    receipt = {}
    _write(d, _frames(1, 0.5), overwrite=True, receipt=receipt)
    assert _seq_names(d) == ["shot_1001.png", "shot_1002.png", "shot_1003.png"]  # left in place
    (msg,) = receipt["warnings"]
    assert "2 frame(s)" in msg and "1002-1003" in msg, msg


def test_a_changed_start_frame_reports_the_earlier_frames(tmp_path):
    d = tmp_path / "shot"
    _write(d, _frames(2), overwrite=True, start=1001)
    receipt = {}
    _write(d, _frames(2), overwrite=True, start=1003, receipt=receipt)
    assert "1001-1002" in receipt["warnings"][0]


def test_a_clean_overwrite_has_no_warning(tmp_path):
    d = tmp_path / "shot"
    _write(d, _frames(2), overwrite=True)
    receipt = {}
    _write(d, _frames(2), overwrite=True, receipt=receipt)
    assert not receipt.get("warnings")


def test_an_unknown_length_stream_never_overwrites_with_overwrite_off(tmp_path):
    d = tmp_path / "shot"
    d.mkdir()
    existing = d / "shot_1002.png"
    existing.write_bytes(b"KEEP")
    _write(d, _frames(2), overwrite=False, count=None)
    assert existing.read_bytes() == b"KEEP"
    assert "shot_001_1001.png" in _seq_names(d) and "shot_001_1002.png" in _seq_names(d)


# ── deadlines: a child that never writes ────────────────────────────────────

@pytest.fixture
def stalled_exe(tmp_path):
    if os.name == "nt":
        pytest.skip("uses a POSIX shell script as the stand-in ffmpeg")
    exe = tmp_path / "ffmpeg_stall"
    exe.write_text("#!/bin/sh\nexec sleep 30\n")
    exe.chmod(0o755)
    return str(exe)


def test_decode_timeout_covers_a_stall_before_the_first_byte(stalled_exe, monkeypatch, tmp_path):
    monkeypatch.setattr(V, "require_ffmpeg", lambda: stalled_exe)
    info = V.VideoInfo(path=str(tmp_path / "x.mov"), width=4, height=4, frames=2)
    t0 = time.monotonic()
    with pytest.raises(V.VideoDecodeError, match="within 0.5s"):
        V.decode(info.path, info=info, timeout=0.5)
    assert time.monotonic() - t0 < 10, "the read blocked past the deadline"


def test_decode_without_a_timeout_is_unchanged(monkeypatch, tmp_path):
    if os.name == "nt":
        pytest.skip("POSIX stand-in")
    exe = tmp_path / "ffmpeg_empty"
    exe.write_text("#!/bin/sh\nexit 0\n")
    exe.chmod(0o755)
    monkeypatch.setattr(V, "require_ffmpeg", lambda: str(exe))
    info = V.VideoInfo(path=str(tmp_path / "x.mov"), width=4, height=4, frames=2)
    with pytest.raises(V.VideoDecodeError, match="No frames were decoded"):
        V.decode(info.path, info=info)


def test_encode_timeout_covers_a_blocked_frame_write(stalled_exe, monkeypatch, tmp_path):
    # The stand-in never reads stdin, so the pipe fills and the frame write
    # blocks: the cap used to apply only after the last write.
    monkeypatch.setattr(W, "_ffmpeg_ok", lambda: True)
    monkeypatch.setattr(W, "_ffmpeg_for", lambda codec: stalled_exe)
    frames = (np.full((512, 512, 3), 0.5, np.float32) for _ in range(8))
    out = tmp_path / "clip.mp4"
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="within 0.5s"):
        W._save_video_ffmpeg(frames, str(out), "MP4 (H.264)", 24.0, 18, "", timeout=0.5)
    assert time.monotonic() - t0 < 10
    assert not out.exists() and not list(tmp_path.glob("*.part*")), list(tmp_path.iterdir())
