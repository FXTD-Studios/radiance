"""Exact frame counts, at every length, on every codec Radiance reads back.

The README claims "frame counts are exact from 1 to 100 frames across H.264,
H.265 10-bit and ProRes". Nothing tested that. `test_video_read.py` pins a
single `FRAMES = 8` and asserts a frame count in two places, both ProRes; the
H.264 fixture there has no frame-count assertion at all, and H.265 10-bit is
never encoded in that file, its only appearance being a `test_writes_without_
crash` that asserts a file exists. A decoder that dropped the last frame of a
long-GOP clip, or handed back one frame too many at a keyframe boundary, would
have shipped green.

This file is the sweep the claim describes. Each fixture frame carries its own
index as a 7-bit bar code across the width, so the assertions are not just
"how many frames came back" but "which frames, in what order": a duplicated,
dropped, reordered or repeated-last frame is caught by value, not by a
tolerance on a ramp. The default sweep covers the lengths where off-by-one
errors live; the exhaustive 1-to-100 version is marked `slow`.

The whole module skips when ffmpeg is absent, and an individual codec skips
when this ffmpeg build cannot encode it, the same way test_video_read.py does.
"""
from __future__ import annotations

import os
import pathlib
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from radiance.core import video as V  # noqa: E402
from radiance.core.ffmpeg import ffmpeg_exe, ffprobe_exe  # noqa: E402

pytestmark = pytest.mark.skipif(
    not ffmpeg_exe(), reason="ffmpeg is required to encode the fixtures"
)

# Frame f is encoded as a 7-bit bar code: bit b is a BAR_WIDTH-wide white
# column when it is set in f. Sampled at the centre of each bar it survives
# 4:2:0 chroma and any sane quantiser, so the decoded index is exact rather
# than a value compared against a tolerance. 7 bits covers 0..127, and the
# claim only runs to 100.
INDEX_BITS = 7
BAR_WIDTH = 8
WIDTH, HEIGHT = INDEX_BITS * BAR_WIDTH, 16   # 56x16, both even for yuv420p

#: 1, 2 and 3 are the degenerate lengths where an off-by-one is the whole clip.
#: Every fixture is encoded with a 12-frame GOP, so 11/12/13, 23/24/25 and
#: 47/48/49 straddle a keyframe boundary from both sides. 99 and 100 are the
#: top of the range the README claims.
FRAME_COUNTS = (1, 2, 3, 4, 5, 11, 12, 13, 23, 24, 25, 47, 48, 49, 50, 99, 100)

#: The node-level pass is the same assertion one layer up, so it runs on a
#: subset rather than re-encoding the whole sweep.
NODE_FRAME_COUNTS = (1, 2, 12, 13, 100)

#: (ffmpeg output arguments, container extension, expected codec, expected
#: bit depth). The GOP is pinned at 12 on both inter-frame codecs so the
#: boundary counts above really are boundaries; leaving it to the encoder's
#: default would make the sweep test whatever ffmpeg felt like that day.
CODECS = {
    "h264": (
        ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "0", "-g", "12"],
        "mp4", "h264", 8,
    ),
    "h265_10bit": (
        ["-c:v", "libx265", "-pix_fmt", "yuv420p10le",
         "-x265-params", "log-level=none:keyint=12:lossless=1"],
        "mp4", "hevc", 10,
    ),
    "prores422": (
        ["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le"],
        "mov", "prores", 10,
    ),
}


def _source(frames: int) -> np.ndarray:
    """(N, H, W, 3) where frame f spells out f in white bars."""
    out = np.zeros((frames, HEIGHT, WIDTH, 3), np.float32)
    for f in range(frames):
        for bit in range(INDEX_BITS):
            if (f >> bit) & 1:
                out[f, :, bit * BAR_WIDTH:(bit + 1) * BAR_WIDTH, :] = 1.0
    return out


def _decoded_indices(arr: np.ndarray) -> list:
    """Read each decoded frame's bar code back out."""
    row = arr[:, HEIGHT // 2, :, 0]
    indices = []
    for f in range(arr.shape[0]):
        value = 0
        for bit in range(INDEX_BITS):
            if row[f, bit * BAR_WIDTH + BAR_WIDTH // 2] > 0.5:
                value |= 1 << bit
        indices.append(value)
    return indices


def _encode(root: pathlib.Path, codec: str, frames: int) -> pathlib.Path:
    args, ext, _expect_codec, _expect_depth = CODECS[codec]
    dst = root / f"{codec}_{frames:03d}.{ext}"
    raw = np.clip(_source(frames) * 65535.0 + 0.5, 0, 65535).astype("<u2").tobytes()
    cmd = [
        ffmpeg_exe(), "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb48le",
        "-s", f"{WIDTH}x{HEIGHT}", "-r", "24", "-i", "-",
        *args, str(dst),
    ]
    proc = subprocess.run(cmd, input=raw, capture_output=True)
    if proc.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
        pytest.skip(
            f"this ffmpeg cannot encode {codec}: "
            + proc.stderr.decode("utf-8", "replace")[-300:]
        )
    return dst


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    """Encode (codec, frames) once and hand the same file to every test."""
    root = tmp_path_factory.mktemp("frame_counts")
    cache = {}

    def make(codec: str, frames: int) -> pathlib.Path:
        key = (codec, frames)
        if key not in cache:
            cache[key] = _encode(root, codec, frames)
        return cache[key]

    return make


def _assert_exactly(path: pathlib.Path, frames: int) -> None:
    arr, info = V.decode(str(path))
    assert arr.shape[0] == frames, (
        f"{path.name}: encoded {frames} frames, decoded {arr.shape[0]}")
    assert _decoded_indices(arr) == list(range(frames)), (
        f"{path.name}: the right number of frames came back but not the right "
        f"frames, in order: got {_decoded_indices(arr)}")
    assert info.frames == frames, (
        f"{path.name}: the probe reports {info.frames} frames for a "
        f"{frames}-frame clip")


# ── the fixtures are what they say they are ────────────────────────────────

@pytest.mark.parametrize("codec", sorted(CODECS))
def test_the_fixture_really_is_that_codec(clip, codec):
    """H.265 10-bit was the gap: nothing in the suite ever encoded one.

    A sweep that silently fell back to another codec would prove nothing about
    the one named in the README, so the fixture's own identity is pinned first.
    """
    _args, _ext, expect_codec, expect_depth = CODECS[codec]
    if not ffprobe_exe():
        pytest.skip("ffprobe is required to read the codec back")
    info = V.probe(str(clip(codec, 12)))
    assert info.codec == expect_codec, (
        f"{codec} fixture came back as {info.codec!r}, not {expect_codec!r}")
    assert info.bit_depth == expect_depth, (
        f"{codec} fixture came back at {info.bit_depth} bits, not {expect_depth}")


# ── the sweep ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("frames", FRAME_COUNTS)
@pytest.mark.parametrize("codec", sorted(CODECS))
def test_the_frame_count_is_exact(clip, codec, frames):
    _assert_exactly(clip(codec, frames), frames)


@pytest.mark.parametrize("codec", sorted(CODECS))
def test_the_probe_never_guesses_on_these_containers(clip, codec):
    """`frames_estimated` says "this number came from duration x fps".

    An estimate that happens to be right is not an exact frame count, and it
    is the thing that goes wrong first when a clip is one frame long.
    """
    if not ffprobe_exe():
        pytest.skip("ffprobe is required to read the frame count")
    for frames in (1, 2, 13, 100):
        info = V.probe(str(clip(codec, frames)))
        assert info.frames == frames and not info.frames_estimated, (
            f"{codec} at {frames} frames: probe said {info.frames} "
            f"(estimated={info.frames_estimated})")


@pytest.mark.parametrize("frames", NODE_FRAME_COUNTS)
@pytest.mark.parametrize("codec", sorted(CODECS))
def test_the_read_node_returns_every_frame(clip, codec, frames):
    """The count the graph actually receives, not just the one core.video saw."""
    from radiance.nodes.io.write import RadianceRead

    image, mask, _info = RadianceRead().read(browse="", path=str(clip(codec, frames)))
    assert image.shape[0] == frames, (
        f"RadianceRead handed the graph {image.shape[0]} frames of a "
        f"{frames}-frame {codec} clip")
    assert mask.shape[0] == frames
    assert _decoded_indices(image.numpy()) == list(range(frames))


@pytest.mark.parametrize("codec", sorted(CODECS))
def test_a_frame_range_is_exact_across_a_gop_boundary(clip, codec):
    """Long-GOP seeking is where a count goes wrong without anyone noticing.

    The fixtures have a keyframe every 12 frames, so these ranges start on a
    keyframe, one before it and one after it.
    """
    path = clip(codec, 50)
    for start, count in ((0, 1), (11, 1), (12, 1), (13, 1),
                         (11, 3), (12, 12), (23, 27), (49, 1)):
        arr, _info = V.decode(str(path), start=start, count=count)
        assert arr.shape[0] == count, (
            f"{codec}: start={start} count={count} returned {arr.shape[0]} frames")
        assert _decoded_indices(arr) == list(range(start, start + count)), (
            f"{codec}: start={start} count={count} returned the wrong frames")


# ── the exhaustive version ─────────────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.timeout(600)   # ~30 s per codec; the project default is 30 s total
@pytest.mark.parametrize("codec", sorted(CODECS))
def test_every_frame_count_from_one_to_one_hundred(tmp_path, codec):
    """The literal claim: 1 to 100, every length, no gaps.

    100 encodes per codec, so it is marked slow and the sweep above is what
    runs by default. Run it with `-m slow` before changing anything in
    radiance.core.video's frame selection or its ffmpeg command line.

    Each length is its own encoder-produced file, as before. The time is
    ffmpeg/ffprobe start-up (three processes per length on a 56x16 clip), not
    codec work, so the lengths run on a thread pool: the threads only wait on
    subprocesses. Results are checked in order, so a failure still names the
    first bad length.
    """
    def one(frames):
        return frames, _encode(tmp_path, codec, frames)

    workers = min(8, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        encoded = list(pool.map(one, range(1, 101)))
        decoded = list(pool.map(lambda fp: (fp[0], fp[1], V.decode(str(fp[1]))), encoded))
    for frames, path, (arr, info) in decoded:
        assert arr.shape[0] == frames, (
            f"{path.name}: encoded {frames} frames, decoded {arr.shape[0]}")
        assert _decoded_indices(arr) == list(range(frames)), (
            f"{path.name}: the right number of frames came back but not the right "
            f"frames, in order: got {_decoded_indices(arr)}")
        assert info.frames == frames, (
            f"{path.name}: the probe reports {info.frames} frames for a "
            f"{frames}-frame clip")
