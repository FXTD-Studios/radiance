"""Read-path defects B5, B9, B10 and B11 (docs/ai/OPEN_QUESTIONS.md).

B5   An image sequence always reported 24 fps, whatever its headers said.
B9   Without ffprobe every video read failed, although ffmpeg was there.
B10  raw=True on a video still decoded it through the file's colour tags.
B11  OpenImageIO-only extensions other than .dpx went to Pillow and failed.

Torch is not imported at module scope, so the tests that only touch the probe,
the banner parser or the routing also run in the light lane. The ones that
build tensors carry the real_torch marker.
"""
from __future__ import annotations

import json
import os
import subprocess
from fractions import Fraction

import numpy as np
import pytest

from radiance.core import video as V
from radiance.core.ffmpeg import ffmpeg_exe

W, H, N = 64, 48, 5

needs_ffmpeg = pytest.mark.skipif(not ffmpeg_exe(), reason="needs ffmpeg")


def _encode(dst, *args):
    cmd = [ffmpeg_exe(), "-y", "-v", "error", "-f", "lavfi",
           "-i", f"testsrc=size={W}x{H}:rate=24000/1001", "-frames:v", str(N),
           *args, str(dst)]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not os.path.isfile(dst):
        pytest.skip(f"this ffmpeg cannot encode {os.path.basename(str(dst))}: "
                    f"{proc.stderr.decode('utf-8', 'replace')[-300:]}")
    return str(dst)


@pytest.fixture
def tagged_mp4(tmp_path):
    """A Rec.709-tagged H.264, the most common delivery a Read node sees."""
    return _encode(tmp_path / "tagged.mp4", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                   "-color_trc", "bt709", "-color_primaries", "bt709",
                   "-colorspace", "bt709", "-color_range", "tv")


@pytest.fixture
def no_ffprobe(monkeypatch):
    """The imageio-ffmpeg install: an ffmpeg binary and no ffprobe."""
    monkeypatch.setattr(V, "ffprobe_exe", lambda: None)


# ── B9: video without ffprobe ─────────────────────────────────────────────

@needs_ffmpeg
def test_b9_probe_without_ffprobe_reads_the_stream(tagged_mp4, no_ffprobe):
    info = V.probe(tagged_mp4)
    assert (info.width, info.height) == (W, H)
    assert info.fps == Fraction(24000, 1001)
    assert info.codec == "h264"
    assert info.pix_fmt == "yuv420p"
    assert info.color_transfer == "bt709"
    assert info.color_primaries == "bt709"
    assert info.color_range == "tv"
    assert info.frames == N and info.frames_estimated


@needs_ffmpeg
def test_b9_decode_without_ffprobe(tagged_mp4, no_ffprobe):
    arr, info = V.decode(tagged_mp4)
    assert arr.shape == (N, H, W, 3)
    assert info.width == W


@needs_ffmpeg
def test_b9_prores_alpha_and_timecode_without_ffprobe(tmp_path, no_ffprobe):
    path = _encode(tmp_path / "alpha.mov", "-c:v", "prores_ks", "-profile:v", "4444",
                   "-pix_fmt", "yuva444p10le", "-timecode", "01:00:00:00")
    info = V.probe(path)
    assert info.has_alpha and info.bit_depth >= 10
    assert info.timecode == "01:00:00:00"
    arr, _ = V.decode(path, info=info)
    assert arr.shape == (N, H, W, 4)


_BANNER = """\
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'clip.mov':
  Metadata:
    major_brand     : qt
    timecode        : 10:00:00:00
  Duration: 00:00:02.00, start: 0.000000, bitrate: 1138 kb/s
  Stream #0:0[0x1](und): Video: h264 (High 10) (avc1 / 0x31637661), \
yuv420p10le(tv, bt2020nc/bt2020/smpte2084, top first), 1920x1080 [SAR 1:1 DAR 16:9], \
50 kb/s, 29.97 fps, 29.97 tbr, 30k tbn (default)
    Metadata:
      handler_name    : VideoHandler
    Side data:
      displaymatrix: rotation of -90.00 degrees
  Stream #0:1[0x2](eng): Audio: aac (LC) (mp4a / 0x6134706D), 48000 Hz, stereo, fltp, 128 kb/s
  Stream #0:2[0x3](eng): Audio: pcm_s24le, 48000 Hz, 2 channels, s32 (24 bit), 2304 kb/s
At least one output file must be specified
"""


def test_b9_banner_parser_matches_ffprobe_fields():
    info = V._info_from_banner("clip.mov", _BANNER)
    assert (info.width, info.height) == (1920, 1080)
    assert info.fps == Fraction(30000, 1001)
    assert info.duration == pytest.approx(2.0)
    assert info.frames == 60 and info.frames_estimated
    assert info.codec == "h264" and info.profile == "High 10"
    assert info.pix_fmt == "yuv420p10le" and info.bit_depth == 10
    assert info.color_range == "tv"
    assert info.color_space == "bt2020nc"
    assert info.color_primaries == "bt2020"
    assert info.color_transfer == "smpte2084" and info.is_hdr
    assert info.field_order == "tt"
    assert info.sample_aspect_ratio == "1:1"
    assert info.rotation == 270          # ffprobe's convention: -90 % 360
    assert info.timecode == "10:00:00:00"
    assert info.audio_streams == 2


def test_b9_banner_parser_unknown_tags_are_empty():
    text = ("  Duration: N/A, bitrate: N/A\n"
            "  Stream #0:0: Video: ffv1 (FFV1 / 0x31564646), bgr0(pc, gbr/unknown/unknown, "
            "progressive), 64x48, SAR 1:1 DAR 4:3, 24 fps, 24 tbr, 1k tbn\n")
    info = V._info_from_banner("x.mkv", text)
    assert info.codec == "ffv1" and info.profile == ""
    assert info.color_space == "gbr"
    assert info.color_primaries == "" and info.color_transfer == ""
    assert info.fps == Fraction(24) and info.frames == 0
    assert V.suggest_transfer(info) is None


def test_b9_banner_without_video_stream_raises():
    text = "  Stream #0:0: Audio: pcm_s16le, 44100 Hz, 1 channels, s16, 705 kb/s\n"
    with pytest.raises(V.VideoDecodeError, match="no video stream"):
        V._info_from_banner("x.wav", text)


# ── B10: raw video bypasses the colour tags ───────────────────────────────

@pytest.mark.real_torch
@needs_ffmpeg
def test_b10_raw_video_is_the_decoded_file_values(tagged_mp4):
    from radiance.io.reader import read_frames

    expected, _ = V.decode(tagged_mp4)
    image, _mask, info = read_frames(path=tagged_mp4, start_frame=0, raw=True)
    np.testing.assert_array_equal(image.numpy(), expected)
    assert info["colour_transform"] == []

    managed, _m, _i = read_frames(path=tagged_mp4, start_frame=0)
    assert not np.allclose(managed.numpy(), expected), "the tag decode should run when not raw"


@pytest.mark.real_torch
@needs_ffmpeg
def test_b10_raw_video_ignores_an_ocio_override(tagged_mp4):
    from radiance.io.reader import read_frames

    expected, _ = V.decode(tagged_mp4)
    image, _mask, _info = read_frames(path=tagged_mp4, start_frame=0, raw=True,
                                      ocio_colorspace="ARRI LogC4")
    np.testing.assert_array_equal(image.numpy(), expected)
