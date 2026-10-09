"""PQ / HLG video: 10-bit only, and HDR10 static metadata on PQ HEVC.

Before 4.0 a PQ write to "MP4 (H.264)" produced an 8-bit yuv420p file tagged
smpte2084 (banding, not HDR10), and the H.265 PQ master carried colour tags
but no mastering display or content light level SEI, so HDR10 players had no
MaxCLL to tone map against. The ffmpeg command is inspected through a fake
process; one real encode is probed when ffmpeg has libx265 and ffprobe.
"""
from __future__ import annotations

import io
import json
import shutil
import subprocess
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from radiance.io import writer as W

PQ = "PQ (HDR10 / ST.2084)"
HLG = "HLG (Hybrid Log-Gamma)"
P3_1000 = "G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1)"


class _RecordingFfmpeg:
    """Records the command, swallows the frames, writes a stand-in file."""
    calls: list = []

    def __init__(self, cmd, *a, **k):
        type(self).calls.append(cmd)
        self.stdin = io.BytesIO()
        self.stderr = io.BytesIO(b"")
        self.returncode = 0
        Path(cmd[-1]).write_bytes(b"movie")

    def wait(self, timeout=None):
        return 0

    def kill(self):  # pragma: no cover
        pass


@pytest.fixture
def fake_ffmpeg(monkeypatch):
    _RecordingFfmpeg.calls = []
    monkeypatch.setattr(W, "_ffmpeg_ok", lambda: True)
    monkeypatch.setattr(W, "_ffmpeg_for", lambda codec: "ffmpeg")
    monkeypatch.setattr(W.subprocess, "Popen", _RecordingFfmpeg)
    return _RecordingFfmpeg.calls


def _frames(*values, h=8, w=8):
    return [np.full((h, w, 3), v, np.float32) for v in values]


def _real_torch():
    import torch
    if getattr(torch, "__radiance_stub__", False):
        pytest.skip("needs real torch")
    return torch


def _x265_params(cmd):
    assert "-x265-params" in cmd, cmd
    return dict(kv.split("=", 1) for kv in cmd[cmd.index("-x265-params") + 1].split(":"))


def _arg(cmd, key):
    return cmd[cmd.index(key) + 1]


# ── (a) no HDR in 8 bits ─────────────────────────────────────────────────────

@pytest.mark.parametrize("fmt", ["MP4 (H.264)", "MOV (DNxHR HQ)"])
@pytest.mark.parametrize("cs", [PQ, HLG])
def test_hdr_into_an_8_bit_format_is_refused_before_anything_is_written(
        tmp_path, fake_ffmpeg, fmt, cs):
    with pytest.raises(ValueError, match="10-bit") as exc:
        W._save_video_ffmpeg(iter(_frames(0.5, 2.0)), str(tmp_path / "shot"), fmt,
                             24.0, 18, "", colour=W.OutputColour(cs))
    assert "H.265 10-bit" in str(exc.value)
    assert fake_ffmpeg == [], "ffmpeg was started for an 8-bit HDR encode"
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("fmt", ["MP4 (H.265 10-bit)", "MOV (ProRes 422 HQ)", "MOV (ProRes 4444)"])
@pytest.mark.parametrize("cs", [PQ, HLG])
def test_hdr_into_a_10_bit_format_is_written(tmp_path, fake_ffmpeg, fmt, cs):
    W._save_video_ffmpeg(iter(_frames(0.5)), str(tmp_path / "shot"), fmt,
                         24.0, 18, "", colour=W.OutputColour(cs))
    cmd = fake_ffmpeg[0]
    out_pix = cmd[cmd.index("-c:v"):][3]          # -c:v <codec> -pix_fmt <out>
    assert "10le" in out_pix


@pytest.mark.parametrize("cs", ["Rec.709 (BT.1886)", "sRGB", "Linear (pass-through)"])
def test_sdr_h264_is_unchanged(tmp_path, fake_ffmpeg, cs):
    W._save_video_ffmpeg(iter(_frames(0.5)), str(tmp_path / "shot"), "MP4 (H.264)",
                         24.0, 18, "", colour=W.OutputColour(cs))
    assert "-x265-params" not in fake_ffmpeg[0]


def test_the_write_node_path_refuses_pq_h264(tmp_path, fake_ffmpeg):
    torch = _real_torch()
    img = torch.full((2, 8, 8, 3), 0.5)
    with pytest.raises(ValueError, match="10-bit"):
        W.write_frames(img, str(tmp_path / "shot"), "VID │ MP4 (H.264)",
                       color_space=PQ, fps=24.0)
    assert list(tmp_path.iterdir()) == []


# ── (b) colour tags and HDR10 static metadata on HEVC ────────────────────────

def test_pq_hevc_carries_colour_tags_and_hdr10_metadata(tmp_path, fake_ffmpeg):
    W._save_video_ffmpeg(iter(_frames(0.5)), str(tmp_path / "shot"), "MP4 (H.265 10-bit)",
                         24.0, 18, "", colour=W.OutputColour(PQ), light_levels=(812, 406))
    cmd = fake_ffmpeg[0]
    assert _arg(cmd, "-color_primaries") == "bt2020"
    assert _arg(cmd, "-color_trc") == "smpte2084"
    assert _arg(cmd, "-colorspace") == "bt2020nc"
    p = _x265_params(cmd)
    assert p["colorprim"] == "bt2020" and p["transfer"] == "smpte2084"
    assert p["colormatrix"] == "bt2020nc"
    assert p["hdr10"] == "1"
    assert p["master-display"] == P3_1000
    assert p["max-cll"] == "812,406"


def test_pq_hevc_without_measured_levels_says_unknown(tmp_path, fake_ffmpeg):
    """A streamed caller (dispatch_write with a generator) never had the clip
    in hand; CTA-861.3's 0,0 ("unknown") is written rather than a guess."""
    W._save_video_ffmpeg(iter(_frames(0.5)), str(tmp_path / "shot"), "MP4 (H.265 10-bit)",
                         24.0, 18, "", colour=W.OutputColour(PQ))
    assert _x265_params(fake_ffmpeg[0])["max-cll"] == "0,0"


def test_hlg_hevc_is_tagged_arib_with_no_static_metadata(tmp_path, fake_ffmpeg):
    W._save_video_ffmpeg(iter(_frames(0.5)), str(tmp_path / "shot"), "MP4 (H.265 10-bit)",
                         24.0, 18, "", colour=W.OutputColour(HLG))
    cmd = fake_ffmpeg[0]
    assert _arg(cmd, "-color_trc") == "arib-std-b67"
    p = _x265_params(cmd)
    assert p["transfer"] == "arib-std-b67" and p["colorprim"] == "bt2020"
    assert "master-display" not in p and "max-cll" not in p and "hdr10" not in p


def test_sdr_hevc_gets_no_hdr_params(tmp_path, fake_ffmpeg):
    W._save_video_ffmpeg(iter(_frames(0.5)), str(tmp_path / "shot"), "MP4 (H.265 10-bit)",
                         24.0, 18, "", colour=W.OutputColour("Rec.709 (BT.1886)"))
    assert "-x265-params" not in fake_ffmpeg[0]


def test_master_display_string_is_p3_d65_1000_nits():
    assert W._x265_master_display() == P3_1000


# ── MaxCLL / MaxFALL ────────────────────────────────────────────────────────

def test_light_levels_are_measured_per_cta_861_3():
    colour = W.OutputColour(PQ, reference_white_nits=203.0)
    a = np.zeros((4, 4, 3), np.float32)
    a[0, 0] = (5.0, 0.0, 0.0)      # one bright red pixel, linear Rec.709
    b = np.full((4, 4, 3), 1.0, np.float32)   # diffuse white everywhere
    cll, fall = W.hdr_light_levels([a, b], colour)
    # Rec.709 red in Rec.2020 is (0.6274, 0.0691, 0.0164): max channel R.
    assert cll == int(np.ceil(5.0 * 0.627404 * 203.0))
    assert fall == 203                                 # frame b's average
    assert W.hdr_light_levels([b], W.OutputColour(HLG)) is None
    assert W.hdr_light_levels([b], W.OutputColour("Rec.709 (BT.1886)")) is None


def test_light_levels_clip_at_pq_peak_and_ignore_negatives():
    colour = W.OutputColour(PQ)
    hot = np.full((2, 2, 3), 1000.0, np.float32)
    neg = np.full((2, 2, 3), -1.0, np.float32)
    assert W.hdr_light_levels([hot, neg], colour) == (10000, 10000)
    assert W.hdr_light_levels([neg], colour) == (0, 0)


def test_light_levels_follow_the_working_space():
    """ACEScg working values go to Rec.2020 before measuring, as the encode does."""
    from radiance.color.encodings import gamut_matrix
    colour = W.OutputColour(PQ, working_space="ACEScg")
    px = np.array([[[2.0, 0.0, 0.0]]], np.float32)
    want = float((gamut_matrix("AP1", "Rec.2020") @ px[0, 0]).max()) * 203.0
    assert W.hdr_light_levels([px], colour)[0] == int(np.ceil(want))


def test_write_frames_measures_and_passes_light_levels(tmp_path, fake_ffmpeg):
    torch = _real_torch()
    img = torch.stack([torch.full((8, 8, 3), 0.5), torch.full((8, 8, 3), 2.0)])
    W.write_frames(img, str(tmp_path / "shot"), "VID │ MP4 (H.265 10-bit)",
                   color_space=PQ, fps=24.0)
    assert _x265_params(fake_ffmpeg[0])["max-cll"] == "406,406"


# ── one real encode ─────────────────────────────────────────────────────────

def _real_hevc() -> bool:
    try:
        if not W._ffmpeg_ok() or not shutil.which("ffprobe"):
            return False
        from radiance.core.ffmpeg import ffmpeg_with_encoder
        ffmpeg_with_encoder("libx265")
        return True
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _real_hevc(), reason="needs ffmpeg with libx265 and ffprobe")
def test_a_real_pq_hevc_file_carries_hdr10_side_data(tmp_path):
    out = W._save_video_ffmpeg(iter(_frames(0.5, 2.0, h=64, w=64)), str(tmp_path / "shot"),
                               "MP4 (H.265 10-bit)", 24.0, 28, "",
                               colour=W.OutputColour(PQ), light_levels=(406, 300))
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
         "-read_intervals", "%+#1", "-show_entries",
         "stream=pix_fmt,color_transfer,color_primaries,color_space:frame=side_data_list",
         "-of", "json", out], capture_output=True, text=True, check=True)
    info = json.loads(r.stdout)
    s = info["streams"][0]
    assert s["pix_fmt"] == "yuv420p10le"
    assert (s["color_primaries"], s["color_transfer"], s["color_space"]) == \
        ("bt2020", "smpte2084", "bt2020nc")
    side = {d["side_data_type"]: d for d in info["frames"][0].get("side_data_list", [])}
    md = side["Mastering display metadata"]
    assert md["max_luminance"] == "10000000/10000" and md["green_x"] == "13250/50000"
    cll = side["Content light level metadata"]
    assert (cll["max_content"], cll["max_average"]) == (406, 300)
