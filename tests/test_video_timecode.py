"""MOV and MP4 carry the shot's start timecode (4.0), as its EXR frames do."""
from __future__ import annotations

import glob
import json
import subprocess

import pytest
import torch

from radiance.io import writer as W


def test_timecode_strings_match_the_exr_convention():
    assert W._video_timecode(1001, 24.0) == "00:00:41:17"
    assert W._video_timecode(1001, 23.976) == "00:00:41:17"
    assert W._video_timecode(86400, 24.0) == "01:00:00:00"
    assert W._video_timecode(0, 25.0) == "00:00:00:00"
    assert W._video_timecode(24 * 3600 * 25, 25.0) == "00:00:00:00"     # wraps at 24 h


@pytest.mark.parametrize("fmt", ["VID │ MOV (ProRes 422 HQ)", "VID │ MP4 (H.264)", "VID │ MP4 (H.265 10-bit)"])
def test_a_video_starts_at_its_first_frame_s_timecode(tmp_path, fmt):
    from radiance.core.ffmpeg import ffprobe_exe
    from radiance.nodes.io.write import RadianceWrite
    try:
        ffprobe = ffprobe_exe()
    except RuntimeError:
        pytest.skip("no ffprobe")
    RadianceWrite().write(image=torch.rand(3, 64, 64, 3), output_path=str(tmp_path / "edit"), format=fmt,
                          color_space="Rec.709 (BT.1886)", fps=24.0, start_frame=86400, overwrite=True)
    path = [p for p in glob.glob(str(tmp_path / "**" / "*.*"), recursive=True) if p.endswith((".mov", ".mp4"))][0]
    info = json.loads(subprocess.run([ffprobe, "-v", "error", "-show_entries", "stream_tags=timecode:format_tags=timecode",
                                      "-of", "json", path], capture_output=True, text=True).stdout)
    found = {t.get("tags", {}).get("timecode") for t in info.get("streams", [])} | {info.get("format", {}).get("tags", {}).get("timecode")}
    assert "01:00:00:00" in found, info
