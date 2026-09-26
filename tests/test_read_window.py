"""The read window that read everything.

`RadianceDigitalCinemaRead` asked for ten frames and the reader handed back a
hundred, because correcting the start of a frame window threw the window's
length away. Plus the streaming read that goes beside the batch one.

The delivery half of this pass lives in tests/test_delivery_endpoints.py, next
to the harness that wires the endpoint up.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")


def _write_plates(directory: Path, first: int, count: int, size=16) -> None:
    from PIL import Image
    directory.mkdir(parents=True, exist_ok=True)
    tile = np.zeros((size, size, 3), np.uint8)
    for i in range(count):
        tile[:] = (i % 251)
        Image.fromarray(tile).save(directory / f"plate.{first + i:04d}.png")


# ═══════════════════════════════════════════════════════════════════════════
#  § 1  A frame window whose start was corrected lost its length
# ═══════════════════════════════════════════════════════════════════════════

def test_a_corrected_start_frame_keeps_the_window_length(tmp_path):
    """`RadianceDigitalCinemaRead` computes `end_frame = start_frame +
    frame_limit - 1` on its own `start_frame` default of 1. VFX sequences
    number from 1001, so `_sequence_frame_range` rewrote the start to 1001 and
    left the end at 10; `end < start` sent `_resolve_sequence_paths` down its
    "read the whole sequence" fallback. Measured: `start_frame=1,
    frame_limit=10` on a 100-frame sequence read all 100 frames into RAM.
    """
    from radiance.io.reader import _sequence_frame_range

    class _Detected:
        first, last = 1001, 1100

    start, end = _sequence_frame_range(_Detected(), 1, 10)
    assert (start, end) == (1001, 1010), (start, end)
    assert end - start + 1 == 10, "the window's length was not carried across"


def test_an_unset_end_frame_still_means_to_the_end(tmp_path):
    from radiance.io.reader import _sequence_frame_range

    class _Detected:
        first, last = 1001, 1100

    assert _sequence_frame_range(_Detected(), 1001, 0) == (1001, 1100)
    assert _sequence_frame_range(_Detected(), 1, 0) == (1001, 1100)


def test_a_window_inside_the_range_on_disk_is_left_alone(tmp_path):
    from radiance.io.reader import _sequence_frame_range

    class _Detected:
        first, last = 1001, 1100

    assert _sequence_frame_range(_Detected(), 1020, 1030) == (1020, 1030)


def test_digital_cinema_read_reads_the_frames_it_was_asked_for(tmp_path):
    """The whole point, end to end: 10 frames off a 100-frame sequence."""
    from radiance.nodes.io.write import RadianceDigitalCinemaRead

    src = tmp_path / "plates"
    _write_plates(src, 1001, 100)

    img, _mask, meta = RadianceDigitalCinemaRead().read(
        source_path=str(src / "plate.1001.png"), read_mode="Auto",
        start_frame=1, frame_limit=10, input_colorspace="sRGB (Standard)")

    assert img.shape[0] == 10, (
        f"asked for 10 frames of a 100-frame sequence and got {img.shape[0]}")
    assert meta["frame_limit"] == 10


def test_digital_cinema_read_with_no_limit_still_reads_the_sequence(tmp_path):
    from radiance.nodes.io.write import RadianceDigitalCinemaRead

    src = tmp_path / "plates"
    _write_plates(src, 1001, 12)
    img, _mask, _meta = RadianceDigitalCinemaRead().read(
        source_path=str(src / "plate.1001.png"), read_mode="Auto",
        start_frame=1, frame_limit=0, input_colorspace="sRGB (Standard)")
    assert img.shape[0] == 12


# ═══════════════════════════════════════════════════════════════════════════
#  § 2  The streaming read, beside the batch one
# ═══════════════════════════════════════════════════════════════════════════

def test_iter_sequence_frames_agrees_with_the_batch_read(tmp_path):
    """The streaming path is added beside the batch path, not instead of it,
    and the two must not disagree about pixels."""
    from radiance.io.reader import _read_sequence, iter_sequence_frames

    src = tmp_path / "plates"
    _write_plates(src, 1001, 5)
    pattern = str(src / "plate.%04d.png")

    batch, _alpha, _w, _h, _n, _fps, _meta = _read_sequence(
        pattern, 1001, 1005, 1, "sRGB")
    streamed = [img for _p, img, _m in
                iter_sequence_frames(pattern, 1001, 1005, 1, "sRGB")]

    assert len(streamed) == batch.shape[0] == 5
    for i, frame in enumerate(streamed):
        assert torch.allclose(frame[0], batch[i])


def test_iter_sequence_frames_does_not_read_ahead(tmp_path):
    """A generator that decodes the whole sequence on first `next()` would be
    the same defect wearing an iterator."""
    from radiance.io import reader as R

    src = tmp_path / "plates"
    _write_plates(src, 1001, 6)
    read = []
    original = R._read_one_sequence_frame

    def _spy(path, *a, **k):
        read.append(os.path.basename(path))
        return original(path, *a, **k)

    R._read_one_sequence_frame = _spy
    try:
        it = R.iter_sequence_frames(str(src / "plate.%04d.png"), 1001, 1006)
        assert read == []
        next(it)
        assert read == ["plate.1001.png"]
        next(it)
        assert read == ["plate.1001.png", "plate.1002.png"]
    finally:
        R._read_one_sequence_frame = original
