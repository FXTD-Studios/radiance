"""Reading video the way a facility expects it to be read.

Every test here encodes real media with ffmpeg and decodes it back, because
the failures being pinned were all invisible to a mock: an alpha channel that
was decoded and then discarded, a 12-bit ProRes that arrived on an 8-bit grid,
a clip that ended early with no error. You cannot catch any of those without
running actual bytes through actual ffmpeg.

The whole module skips when ffmpeg is absent. The CI lane has it; a
contributor's laptop might not.

What was wrong before 3.2.0
---------------------------
Measured on this fixture set, against the old implementation:

* ProRes 4444 with a matte decoded to **three channels**. The alpha was read
  off disk and thrown away.
* ``_load_video_to_numpy`` -- the decoder the DCC handoff path used -- put
  **every** source on an exact 1/255 grid, because it tried OpenCV first and
  OpenCV hands back 8-bit BGR regardless of the source. A 12-bit ProRes 4444
  lost four bits per component.
* A 4-second 1080p ProRes 422 HQ clip took **25.0 s** and about **600 MB** of
  temporary PNGs, because every frame was written to disk and read back.
* There was no way to ask for a frame range. ``start_frame`` was documented
  "sequences only" and ignored.
* MXF reported ``nb_frames`` of **0**, and nothing filled it in.
* The container's colour tags were never read.
* Any failure inside ``RadianceRead.read`` was caught and turned into an 8x8
  black frame, so a corrupt file produced a green node and a black master.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from radiance.core import video as V  # noqa: E402
from radiance.core.ffmpeg import ffmpeg_exe, ffprobe_exe  # noqa: E402

pytestmark = pytest.mark.skipif(
    not ffmpeg_exe(), reason="ffmpeg is required to encode the fixtures"
)

WIDTH, HEIGHT, FRAMES = 64, 48, 8


def _source_rgba() -> np.ndarray:
    """(N, H, W, 4) with detail finer than 8 bits and a real alpha wedge.

    Adjacent columns differ by 1/4096, which cannot survive an 8-bit round
    trip: if a decoder quantises, the ramp collapses onto a coarse grid and the
    test says so.
    """
    out = np.zeros((FRAMES, HEIGHT, WIDTH, 4), np.float32)
    ramp = (np.arange(WIDTH, dtype=np.float32) - WIDTH / 2) / 4096.0
    for f in range(FRAMES):
        out[f, :, :, 0] = 0.50 + ramp + f / 512.0
        out[f, :, :, 1] = 0.25 + ramp
        out[f, :, :, 2] = 0.75 + ramp
        out[f, :, :, 3] = np.linspace(0.0, 1.0, WIDTH, dtype=np.float32)
    return out


def _encode(dst: pathlib.Path, raw: bytes, *args: str) -> pathlib.Path:
    cmd = [
        ffmpeg_exe(), "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "rgba64le",
        "-s", f"{WIDTH}x{HEIGHT}", "-r", "24", "-i", "-",
        *args, str(dst),
    ]
    proc = subprocess.run(cmd, input=raw, capture_output=True)
    if proc.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
        pytest.skip(
            f"this ffmpeg cannot encode {dst.name}: "
            + proc.stderr.decode("utf-8", "replace")[-300:]
        )
    return dst


@pytest.fixture(scope="module")
def source() -> np.ndarray:
    return _source_rgba()


@pytest.fixture(scope="module")
def raw(source) -> bytes:
    return np.clip(source * 65535.0 + 0.5, 0, 65535).astype("<u2").tobytes()


@pytest.fixture(scope="module")
def prores4444(tmp_path_factory, raw) -> pathlib.Path:
    d = tmp_path_factory.mktemp("video")
    return _encode(d / "plate_4444.mov", raw,
                   "-c:v", "prores_ks", "-profile:v", "4", "-pix_fmt", "yuva444p10le")


@pytest.fixture(scope="module")
def prores422(tmp_path_factory, raw) -> pathlib.Path:
    d = tmp_path_factory.mktemp("video")
    return _encode(d / "plate_422hq.mov", raw,
                   "-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le")


@pytest.fixture(scope="module")
def h264(tmp_path_factory, raw) -> pathlib.Path:
    d = tmp_path_factory.mktemp("video")
    return _encode(d / "review.mp4", raw,
                   "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "0")


# ── pixel format introspection ─────────────────────────────────────────────

@pytest.mark.parametrize("pix_fmt, expected", [
    ("yuv420p", 8), ("yuv422p10le", 10), ("yuva444p12le", 12),
    ("gbrp16le", 16), ("rgb24", 8), ("rgba64le", 16), ("rgb48le", 16),
    ("", 8),
])
def test_bit_depth_is_read_from_the_pixel_format(pix_fmt, expected):
    assert V.pix_fmt_bit_depth(pix_fmt) == expected


@pytest.mark.parametrize("pix_fmt, expected", [
    ("yuva444p10le", True), ("yuva420p", True), ("rgba", True),
    ("bgra", True), ("gbrap12le", True), ("rgba64le", True),
    ("yuv422p10le", False), ("rgb24", False), ("gbrp12le", False), ("", False),
])
def test_alpha_is_read_from_the_pixel_format(pix_fmt, expected):
    assert V.pix_fmt_has_alpha(pix_fmt) is expected


@pytest.mark.parametrize("text, expected", [
    ("24/1", 24.0), ("24000/1001", 24000 / 1001), ("30", 30.0),
    ("0/0", 25.0), ("", 25.0), ("garbage", 25.0), ("-30/1", 25.0),
])
def test_frame_rate_parsing_survives_what_containers_actually_say(text, expected):
    """`0/0` and a bare integer both appear in the wild.

    The old code did `float(x) for x in fps_str.split("/")`, which raised on
    `"30"` -- and the handler swallowed the width and height along with it.
    """
    from fractions import Fraction
    got = V._parse_fraction(text, Fraction(25, 1))
    assert float(got) == pytest.approx(expected)


# ── probing ────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not ffprobe_exe(), reason="ffprobe is required")
def test_probe_reports_alpha_and_bit_depth_for_prores_4444(prores4444):
    info = V.probe(str(prores4444))
    assert info.has_alpha, "ProRes 4444 carries alpha; the probe must see it"
    assert info.bit_depth >= 10
    assert info.width == WIDTH and info.height == HEIGHT
    assert info.frames == FRAMES and not info.frames_estimated
    assert info.codec == "prores"
    assert info.is_intra, "ProRes is all-intra; frame ranges are cheap"


@pytest.mark.skipif(not ffprobe_exe(), reason="ffprobe is required")
def test_probe_never_reports_zero_frames_when_it_can_estimate(prores422, monkeypatch):
    """MXF and many MOVs carry no frame count. Zero reads like 'empty file'."""
    real = V.probe(str(prores422))
    assert real.frames == FRAMES

    original = subprocess.run

    def strip_nb_frames(cmd, **kwargs):
        out = original(cmd, **kwargs)
        if out.stdout and "nb_frames" in str(out.stdout):
            data = json.loads(out.stdout)
            for stream in data.get("streams", []):
                stream.pop("nb_frames", None)
                stream.pop("nb_read_frames", None)
                stream.pop("nb_read_packets", None)
            out = subprocess.CompletedProcess(cmd, 0, json.dumps(data), "")
        return out

    monkeypatch.setattr(subprocess, "run", strip_nb_frames)
    info = V.probe(str(prores422))
    assert info.frames > 0, "with no nb_frames it must fall back to duration x fps"
    assert info.frames_estimated, "and it must admit the number is an estimate"


def test_probe_refuses_a_file_with_no_video_stream(tmp_path):
    text = tmp_path / "notes.txt"
    text.write_text("this is not a movie")
    with pytest.raises((V.VideoDecodeError, Exception)) as excinfo:
        V.probe(str(text))
    assert "video" in str(excinfo.value).lower()


def test_probe_says_which_file_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError) as excinfo:
        V.probe(str(tmp_path / "absent.mov"))
    assert "absent.mov" in str(excinfo.value)


# ── decoding ───────────────────────────────────────────────────────────────

def test_prores_4444_alpha_survives_the_decode(prores4444, source):
    """The headline regression: a vendor plate's matte used to be discarded."""
    arr, info = V.decode(str(prores4444))
    assert arr.shape == (FRAMES, HEIGHT, WIDTH, 4), (
        f"expected 4 channels from a ProRes 4444, got {arr.shape[-1]}"
    )
    # The alpha plane is stored at 10 bits, so one code is 1/1023. Allow two,
    # which still fails hard if the matte is dropped, flattened or swapped for
    # a colour channel.
    assert np.abs(arr[..., 3] - source[..., 3]).max() < 2.0 / 1023.0


def test_alpha_can_be_declined(prores4444):
    arr, _ = V.decode(str(prores4444), alpha=False)
    assert arr.shape[-1] == 3


@pytest.mark.parametrize("fixture", ["prores4444", "prores422"])
def test_more_than_eight_bits_survive(fixture, request, source):
    """A 10/12-bit source must not come back on an 8-bit grid.

    The source ramp steps by 1/4096 per column. If a decoder quantises to 8
    bits, every value lands exactly on a multiple of 1/255.
    """
    path = request.getfixturevalue(fixture)
    arr, _ = V.decode(str(path))
    red = arr[..., 0].ravel()
    off_grid = np.abs(red * 255.0 - np.round(red * 255.0)).max()
    assert off_grid > 1e-3, (
        "every decoded value sits on an exact 1/255 grid, so the decode "
        "quantised a 10-bit source to 8 bits"
    )
    assert np.abs(arr[..., :3] - source[..., :3]).max() < 0.01


def test_an_eight_bit_source_is_still_exact(h264, source):
    """Lossless H.264 8-bit: the round trip should be within one 8-bit code."""
    arr, info = V.decode(str(h264))
    assert info.bit_depth == 8
    assert arr.shape == (FRAMES, HEIGHT, WIDTH, 3)
    # 4:2:0 chroma on a horizontal ramp costs a little; luma must be tight.
    luma = arr[..., :3].mean(axis=-1)
    ref = source[..., :3].mean(axis=-1)
    assert np.abs(luma - ref).max() < 0.02


# ── frame ranges ───────────────────────────────────────────────────────────

def test_a_frame_range_returns_exactly_those_frames(prores4444, source):
    arr, _ = V.decode(str(prores4444), start=2, count=3)
    assert arr.shape[0] == 3
    assert np.abs(arr[..., :3] - source[2:5, ..., :3]).max() < 0.01


def test_a_step_returns_every_nth_frame(prores4444, source):
    arr, _ = V.decode(str(prores4444), start=1, step=2)
    assert arr.shape[0] == len(range(1, FRAMES, 2))
    assert np.abs(arr[..., :3] - source[1::2, ..., :3]).max() < 0.01


def test_a_range_past_the_end_is_an_error_not_an_empty_array(prores4444):
    """Silently returning nothing is how a shot goes missing from a delivery."""
    with pytest.raises(V.VideoDecodeError) as excinfo:
        V.decode(str(prores4444), start=FRAMES + 50)
    assert "frame" in str(excinfo.value).lower()


def test_the_frame_count_is_the_selection_not_the_clip(prores4444):
    arr, info = V.decode(str(prores4444), count=2)
    assert arr.shape[0] == 2
    assert info.frames == FRAMES, "info still describes the file, not the slice"


# ── failure behaviour ──────────────────────────────────────────────────────

def test_a_truncated_file_raises_instead_of_returning_a_short_clip(tmp_path, prores422):
    """The expensive bug: a clip that ends early looks like a clip that ended.

    Half a ProRes file still decodes a plausible number of frames. Handing
    those back without a word is how a shot arrives at review missing its last
    second.
    """
    data = prores422.read_bytes()
    broken = tmp_path / "truncated.mov"
    broken.write_bytes(data[: int(len(data) * 0.55)])
    with pytest.raises((V.VideoDecodeError, V.VideoTruncatedError)):
        V.decode(str(broken))


def test_ffmpeg_stderr_reaches_the_caller(tmp_path):
    junk = tmp_path / "corrupt.mov"
    junk.write_bytes(os.urandom(50_000))
    with pytest.raises(Exception) as excinfo:
        V.decode(str(junk))
    message = str(excinfo.value)
    assert junk.name in message, "the message must name the file that failed"


def test_no_temporary_files_are_left_behind(prores422, tmp_path, monkeypatch):
    """The old decoder wrote a PNG per frame into a temp dir -- ~600 MB for a
    4-second HD clip. Nothing should touch the filesystem now."""
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setenv("TMPDIR", str(scratch))
    V.decode(str(prores422))
    assert list(scratch.iterdir()) == [], (
        f"the decode left {[p.name for p in scratch.iterdir()]} behind"
    )


# ── colour tags ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("trc, expected", [
    ("bt709", "Rec.709 (BT.1886)"),
    ("smpte170m", "Rec.709 (BT.1886)"),
    ("iec61966-2-1", "sRGB"),
    ("smpte2084", "PQ (ST.2084)"),
    ("arib-std-b67", "HLG (BT.2100)"),
])
def test_a_tagged_transfer_maps_to_a_radiance_colour_space(trc, expected):
    info = V.VideoInfo(path="x.mov", color_transfer=trc)
    assert V.suggest_transfer(info) == expected


def test_an_untagged_file_suggests_nothing():
    """Guessing is worse than asking. An untagged file gets a warning, not a
    transform applied behind the user's back."""
    assert V.suggest_transfer(V.VideoInfo(path="x.mov")) is None
    # A tag with no Radiance inverse (SMPTE 428 is DCI X'Y'Z') is not guessed.
    assert V.suggest_transfer(V.VideoInfo(path="x.mov", color_transfer="smpte428")) is None
    # bt2020-10 now has one (3.5).
    assert V.suggest_transfer(V.VideoInfo(path="x.mov", color_transfer="bt2020-10")) == "Rec.2020 (BT.2020 OETF)"


def test_every_suggestion_is_a_colour_space_the_read_node_offers():
    """A suggestion the widget cannot hold would be worse than none."""
    from radiance.nodes.io.write import INPUT_COLOR_SPACES

    for name in V.TRANSFER_TO_RADIANCE.values():
        assert name in INPUT_COLOR_SPACES, (
            f"{name!r} is suggested by core.video but missing from "
            "INPUT_COLOR_SPACES, so the Read node could never apply it"
        )


def test_every_offered_colour_space_can_actually_be_decoded():
    """The menu listed nine entries; three of them had no inverse wired up.
    Since 3.5 every entry resolves through radiance.color.encodings, transfer
    and primaries, to the working space."""
    from radiance.nodes.io.write import INPUT_COLOR_SPACES
    from radiance.color import encodings as enc

    probe = np.linspace(0.01, 0.99, 32, dtype=np.float32).reshape(1, 1, 32, 1)
    probe = np.repeat(probe, 3, axis=-1)
    for name in INPUT_COLOR_SPACES:
        if name == "Auto / Linear (pass-through)":
            continue
        for working in enc.WORKING_SPACES:
            out, _how = enc.decode(probe.copy(), name, working)
            assert out.shape == probe.shape, f"{name} changed the array shape"
            assert np.isfinite(out).all(), f"{name} -> {working} produced non-finite values"


# ── the Read node itself ───────────────────────────────────────────────────

def test_read_node_puts_prores_4444_alpha_on_the_mask_output(prores4444, source):
    from radiance.nodes.io.write import RadianceRead

    image, mask, _info = RadianceRead().read(browse="", path=str(prores4444))
    assert image.shape == (FRAMES, HEIGHT, WIDTH, 3)
    assert mask.shape == (FRAMES, HEIGHT, WIDTH)
    assert np.abs(mask.numpy() - source[..., 3]).max() < 2.0 / 1023.0
    assert mask.numpy().max() > 0.9 and mask.numpy().min() < 0.1, (
        "the matte came back flat, so it is not the file's alpha"
    )


def test_read_node_leaves_the_mask_empty_when_there_is_no_alpha(prores422):
    from radiance.nodes.io.write import RadianceRead

    _image, mask, _info = RadianceRead().read(browse="", path=str(prores422))
    assert float(mask.abs().max()) == 0.0


def test_read_node_honours_a_frame_range(prores4444):
    from radiance.nodes.io.write import RadianceRead

    image, mask, _info = RadianceRead().read(
        browse="", path=str(prores4444), start_frame=2, end_frame=5)
    assert image.shape[0] == 4
    assert mask.shape[0] == 4


def test_the_sequence_default_start_frame_does_not_swallow_a_clip(prores4444):
    """`start_frame` defaults to 1001 for VFX sequences. Taken literally on a
    clip that is 8 frames long, that means decoding nothing at all."""
    from radiance.nodes.io.write import RadianceRead

    image, _mask, _info = RadianceRead().read(
        browse="", path=str(prores4444), start_frame=1001)
    assert image.shape[0] == FRAMES


def test_read_node_raises_on_a_missing_file(tmp_path):
    """It used to log the error and return an 8x8 black frame, so the graph
    carried on and wrote a master out of black."""
    from radiance.nodes.io.write import RadianceRead

    with pytest.raises(Exception) as excinfo:
        RadianceRead().read(browse="", path=str(tmp_path / "absent.mov"))
    assert "absent.mov" in str(excinfo.value)


def test_read_node_raises_on_a_corrupt_file(tmp_path):
    from radiance.nodes.io.write import RadianceRead

    junk = tmp_path / "corrupt.mov"
    junk.write_bytes(os.urandom(80_000))
    with pytest.raises(Exception):
        RadianceRead().read(browse="", path=str(junk))


def test_a_blank_path_still_returns_a_frame_rather_than_erroring():
    """A node just dropped on the canvas is not a failure."""
    from radiance.nodes.io.write import RadianceRead

    image, mask, _info = RadianceRead().read(browse="", path="")
    assert image.shape[0] == 1 and mask is not None


def test_read_node_reports_what_the_file_is(prores4444, caplog):
    """The console should say what arrived, the way a Nuke Read does."""
    import logging

    from radiance.nodes.io.write import _read_video

    with caplog.at_level(logging.INFO):
        _read_video(str(prores4444), 0, "Auto / Linear (pass-through)")
    logged = " ".join(r.getMessage() for r in caplog.records)
    assert "prores" in logged and "alpha" in logged


def test_the_metadata_carries_the_colour_tags(prores4444):
    from radiance.nodes.io.write import _read_video

    *_rest, meta = _read_video(str(prores4444), 0, "Auto / Linear (pass-through)")
    data = json.loads(meta)
    for key in ("color_transfer", "color_primaries", "color_range", "bit_depth",
                "has_alpha", "codec", "timecode", "fps_exact", "frames"):
        assert key in data, f"{key} is missing from the Read metadata"
    assert data["has_alpha"] is True
    assert data["alpha"] is True


# ── the DCC handoff decoder ────────────────────────────────────────────────

def test_the_handoff_decoder_no_longer_quantises_to_eight_bits(prores4444):
    """`_load_video_to_numpy` used to try OpenCV first, which returns 8-bit BGR
    for every source. A 12-bit ProRes lost four bits per component."""
    from radiance.nodes.io.write import _load_video_to_numpy

    arr = _load_video_to_numpy(str(prores4444))
    red = arr[..., 0].ravel()
    off_grid = np.abs(red * 255.0 - np.round(red * 255.0)).max()
    assert off_grid > 1e-3, "the handoff decoder is still quantising to 8 bits"


def test_the_handoff_decoder_respects_a_frame_cap(prores4444):
    from radiance.nodes.io.write import _load_video_to_numpy

    assert _load_video_to_numpy(str(prores4444), max_frames=3).shape[0] == 3


def test_the_handoff_decoder_raises_rather_than_returning_a_short_clip(tmp_path):
    from radiance.nodes.io.write import _load_video_to_numpy

    junk = tmp_path / "corrupt.mov"
    junk.write_bytes(os.urandom(40_000))
    with pytest.raises(Exception):
        _load_video_to_numpy(str(junk))


# ── extension coverage ─────────────────────────────────────────────────────

@pytest.mark.parametrize("ext", [".mov", ".mxf", ".mp4", ".m2ts", ".mts",
                                 ".mkv", ".webm", ".avi", ".r3d", ".braw"])
def test_the_formats_a_camera_hands_you_are_recognised_as_video(ext):
    """The old seven-entry set classified a .m2ts off a card as 'unknown',
    which then tried to open it as a still image."""
    from radiance.nodes.io.write import _path_kind

    assert _path_kind(f"/plates/sh010{ext}") == "video"


def test_the_extension_set_has_one_owner():
    """Three copies of this list is three chances to disagree."""
    from radiance.nodes.io.write import _VID_EXT

    assert _VID_EXT == set(V.VIDEO_EXTENSIONS)


def test_no_module_decodes_video_through_a_png_temp_directory():
    """The old approach, so it cannot come back by copy-paste."""
    import re

    root = pathlib.Path(__file__).resolve().parent.parent
    offenders = []
    for py in root.rglob("*.py"):
        if "test" in py.parts or "_to_delete" in py.parts:
            continue
        text = py.read_text(encoding="utf-8", errors="ignore")
        # A frame-numbered PNG pattern handed to ffmpeg as an output.
        if re.search(r'f?%0\d+d\.png', text) and "image2" in text:
            offenders.append(str(py.relative_to(root)))
    assert not offenders, (
        f"{offenders} decode video by writing PNGs to a temp directory; use "
        "radiance.core.video.decode instead"
    )
