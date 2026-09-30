"""The write path streams, and the ceiling is the working window.

Nothing in the write path used to stream. Per 1920x1080 float32 RGB frame
(24.9 MB) these were simultaneously resident:

    RadianceWrite -> VID   `frames` + `out_frames` + `np.stack` + raw bytes
    RadianceWrite -> SEQ   `frames` + `out_frames`
    RadianceRead sequence  a list of N tensors + `torch.cat`

so the maximum length of a shot was a property of how much RAM the box had,
measured at roughly 2000 frames at 1080p and 500 at 4K. Worse, `out_frames`
held *views* with `color_space="Linear (pass-through)"` and fresh allocations
the moment any transform was active, so turning on a colour transform halved
the maximum sequence length, silently.

A claim that code "streams" is not a test. These measure: each case runs in a
subprocess and reports `ru_maxrss`, the peak resident set the kernel actually
saw, for a short sequence and a long one. Flat means streaming. The old code
cannot even reach the measurement -- `dispatch_write` was typed
`List[np.ndarray]` and began with `len(frames)` -- which is itself the point.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

_ROOT = Path(__file__).resolve().parent.parent
_PARENT = str(_ROOT.parent)

#: One 256x256 float32 RGB frame, in KB. The numbers below are expressed in
#: multiples of this so they say something about frames rather than about this
#: machine.
_FRAME_KB = (256 * 256 * 3 * 4) / 1024.0      # 768 KB

#: Peak RSS is a whole-process measurement, so it carries the allocator's
#: rounding, import-time fragmentation and glibc arena behaviour. This is the
#: slack allowed on top of the working window, and it is small next to the
#: linear growth it has to be able to tell apart: 400 extra frames is 300 MB.
_SLACK_KB = 24 * 1024


def _run(body: str) -> int:
    """Run `body` in a fresh interpreter and return its peak RSS in KB.

    A subprocess, because `ru_maxrss` is a high-water mark that never falls:
    two measurements in one process would both report the larger. Peak resident
    rather than tracemalloc, because numpy allocates its buffers through malloc
    and not through the Python allocator, so tracemalloc does not see the
    frames at all -- which is exactly the memory in question.
    """
    script = textwrap.dedent(f"""
        import os, sys, tempfile
        if sys.platform == "win32":
            import psutil
            def peak_rss_kb():
                return psutil.Process().memory_info().peak_wset // 1024
        else:
            import resource
            def peak_rss_kb():
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                return rss // 1024 if sys.platform == "darwin" else rss
        sys.path.insert(0, {_PARENT!r})
        import numpy as np
        import torch
        OUT = tempfile.mkdtemp()
        {textwrap.indent(textwrap.dedent(body), ' ' * 8).strip()}
        sys.stderr.write("RSS=%d\\n" % peak_rss_kb())
    """)
    proc = subprocess.run([sys.executable, "-c", script],
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-4000:]
    line = [l for l in proc.stderr.splitlines() if l.startswith("RSS=")]
    assert line, proc.stderr[-4000:]
    return int(line[-1].split("=", 1)[1])


def _frames_src(n: int) -> str:
    return (
        "def gen(n):\n"
        "    for i in range(n):\n"
        "        yield np.full((256, 256, 3), (i %% 17) / 17.0, dtype=np.float32)\n"
        "N = %d\n" % n
    )


# ═══════════════════════════════════════════════════════════════════════════
#  § 1  The measurement
# ═══════════════════════════════════════════════════════════════════════════

_SEQ_BODY = """
%s
from radiance.io.writer import dispatch_write
saved, count = dispatch_write(
    gen(N), os.path.join(OUT, "shot"), "SEQ │ PNG (8-bit)",
    24.0, 18, "ZIP", 1001, 4, "", True, frame_count=N)
assert count == N, count
"""

_VID_BODY = """
%s
from radiance.io.writer import dispatch_write
saved, count = dispatch_write(
    gen(N), os.path.join(OUT, "shot"), "VID │ MP4 (H.264)",
    24.0, 30, "ZIP", 1001, 4, "", True, frame_count=N)
assert count == N, count
"""


@pytest.mark.parametrize("body,label",
                         [(_SEQ_BODY, "sequence"), (_VID_BODY, "video")],
                         ids=["sequence", "video"])
def test_peak_memory_is_flat_in_sequence_length(body, label):
    """32 frames and 512 frames must cost the same peak.

    A per-frame path costs one frame. A path that materialises the sequence
    costs 480 extra frames, 360 MB here and 12 GB at 1080p, which is the wall
    the old writer hit.
    """
    short = _run(body % _frames_src(32))
    long_ = _run(body % _frames_src(512))

    growth_kb = long_ - short
    linear_kb = 480 * _FRAME_KB
    assert growth_kb < _SLACK_KB, (
        f"{label}: peak RSS grew {growth_kb / 1024:.1f} MB going from 32 to 512 "
        f"frames; a materialising write would grow about {linear_kb / 1024:.0f} MB "
        f"and a streaming one about 0"
    )
    # And it must be well under the linear prediction, not merely under a
    # generous constant: an assertion that can be satisfied by a small leak is
    # not a measurement.
    assert growth_kb < linear_kb / 8


_TENSOR_BODY = """
%s
from radiance.io.writer import write_frames
img = torch.zeros(N, 256, 256, 3)
write_frames(image=img, output_path=os.path.join(OUT, "shot"),
             format="SEQ │ PNG (8-bit)", color_space=%r, overwrite=True)
"""


def test_a_colour_transform_does_not_cost_a_second_copy_of_the_shot():
    """Enabling a colour transform must not change how long a shot can be.

    `out_frames` held views under "Linear (pass-through)" and a fresh array per
    frame under anything else, so switching the colour space from pass-through
    to sRGB doubled peak memory and halved the maximum sequence length --
    unpredictably, because nothing in the UI said so.
    """
    n = 400
    flat = _run(_TENSOR_BODY % (_frames_src(n), "Linear (pass-through)"))
    srgb = _run(_TENSOR_BODY % (_frames_src(n), "sRGB"))

    growth_kb = srgb - flat
    second_copy_kb = n * _FRAME_KB
    assert growth_kb < _SLACK_KB, (
        f"turning on a colour transform cost {growth_kb / 1024:.1f} MB extra on "
        f"a {n}-frame shot; a second full copy would be "
        f"{second_copy_kb / 1024:.0f} MB"
    )


_PIPELINE_BODY = """
from radiance.io.reader import iter_sequence_frames
from radiance.io.writer import dispatch_write, transform_stream

SRC = %r
N = %d

def frames():
    for _path, img, _mask in iter_sequence_frames(
            os.path.join(SRC, "plate.%%04d.png"), 1001, 1001 + N - 1):
        yield img[0].numpy()

saved, count = dispatch_write(
    transform_stream(frames(), color_space="sRGB"),
    os.path.join(OUT, "out"), "SEQ │ PNG (8-bit)",
    24.0, 18, "ZIP", 1001, 4, "", True, frame_count=N)
assert count == N, count
"""


def _write_plates(directory: Path, n: int) -> None:
    from PIL import Image
    directory.mkdir(parents=True, exist_ok=True)
    tile = np.zeros((128, 128, 3), np.uint8)
    for i in range(n):
        tile[:] = i % 251
        Image.fromarray(tile).save(directory / f"plate.{1001 + i:04d}.png")


def test_a_read_transform_write_pipeline_holds_a_working_window(tmp_path):
    """The whole point of the streaming read: never hold the shot.

    `_read_sequence` returns a batch, which is right for the node because a
    ComfyUI node hands the graph a batch. A read-transform-write pipeline does
    not want one, and `iter_sequence_frames` is what it consumes instead.
    """
    src = tmp_path / "plates"
    _write_plates(src, 512)

    short = _run(_PIPELINE_BODY % (str(src), 32))
    long_ = _run(_PIPELINE_BODY % (str(src), 512))

    growth_kb = long_ - short
    # 128x128 RGB float32 frames here, because the pipeline decodes PNGs and
    # 512 large ones would make the test slow for no extra signal.
    linear_kb = 480 * (128 * 128 * 3 * 4) / 1024.0
    assert growth_kb < _SLACK_KB, (
        f"peak RSS grew {growth_kb / 1024:.1f} MB over 480 extra frames; "
        f"holding the shot would cost about {linear_kb / 1024:.0f} MB"
    )


# ═══════════════════════════════════════════════════════════════════════════
#  § 2  The mechanism the measurement depends on
# ═══════════════════════════════════════════════════════════════════════════

def test_dispatch_write_accepts_a_generator(tmp_path):
    """`frames` was typed `List[np.ndarray]` and `dispatch_write` opened with
    `n = len(frames)`, so a caller that produced frames lazily could not use
    the writer at all. This raises TypeError on the old engine."""
    from radiance.io.writer import dispatch_write

    def gen():
        for i in range(5):
            yield np.full((8, 8, 3), i / 5.0, np.float32)

    saved, count = dispatch_write(
        gen(), str(tmp_path / "shot"), "SEQ │ PNG (8-bit)",
        24.0, 18, "ZIP", 1001, 4, "", True, frame_count=5)
    assert count == 5
    assert sorted(p.name for p in Path(saved).glob("*.png")) == [
        f"shot_{1001 + i}.png" for i in range(5)]


def test_a_single_image_write_consumes_only_the_first_frame(tmp_path):
    """IMG writes frame 0. It used to index `frames[0]` out of a fully
    materialised list; taking it by iteration means a 2000-frame batch costs
    one frame here, not 2000."""
    from radiance.io.writer import dispatch_write

    produced = []

    def gen():
        for i in range(500):
            produced.append(i)
            yield np.full((8, 8, 3), 0.5, np.float32)

    dispatch_write(gen(), str(tmp_path / "one"), "IMG │ PNG (8-bit)",
                   24.0, 18, "ZIP", 1001, 4, "", True, frame_count=500)
    assert produced == [0], f"{len(produced)} frames were produced for a one-frame write"


def test_transform_stream_is_lazy():
    """The transform must not run ahead of the writer. `out_frames` was a list
    built to completion before the first byte was written."""
    from radiance.io.writer import transform_stream

    produced = []

    def gen():
        for i in range(10):
            produced.append(i)
            yield np.full((4, 4, 3), 0.25, np.float32)

    stream = transform_stream(gen(), color_space="sRGB")
    assert produced == [], "the transform ran before anything consumed it"
    next(stream)
    assert produced == [0]
    next(stream)
    assert produced == [0, 1]


def test_sequence_frames_are_written_a_few_at_a_time(tmp_path, monkeypatch):
    """Frame by frame, compression used one core. Several frames now go to
    writer threads at once, never more than `_SEQ_WRITERS` in flight."""
    import threading
    import time
    import radiance.io.writer as writer

    lock, active, peak, produced = threading.Lock(), [0], [0], []

    def slow_save(arr, path, fmt, quality=18, metadata=None):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
            ahead = len(produced) - int(path.stem.rsplit("_", 1)[1]) + 1001
        time.sleep(0.02)
        with lock:
            active[0] -= 1
        assert ahead <= 4, f"{ahead} frames produced ahead of the one being written"
        path.touch()

    def gen():
        for i in range(12):
            produced.append(i)
            yield np.zeros((8, 8, 3), np.float32)

    monkeypatch.setattr(writer, "_save_pil_image", slow_save)
    monkeypatch.setattr(writer, "_SEQ_WRITERS", 4)
    saved, count = writer.dispatch_write(
        gen(), str(tmp_path / "shot"), "SEQ │ PNG (8-bit)",
        24.0, 18, "ZIP", 1001, 4, "", True, frame_count=12)
    assert count == 12 and len(list(Path(saved).glob("*.png"))) == 12
    assert 1 < peak[0] <= 4, peak[0]


def test_a_failed_frame_stops_the_sequence_with_its_error(tmp_path, monkeypatch):
    import radiance.io.writer as writer

    def save(arr, path, fmt, quality=18, metadata=None):
        if path.stem.endswith("1003"):
            raise OSError("disk full")
        path.touch()

    monkeypatch.setattr(writer, "_save_pil_image", save)
    with pytest.raises(OSError, match="disk full"):
        writer.dispatch_write(
            (np.zeros((8, 8, 3), np.float32) for _ in range(6)),
            str(tmp_path / "shot"), "SEQ │ PNG (8-bit)",
            24.0, 18, "ZIP", 1001, 4, "", True, frame_count=6)


# ═══════════════════════════════════════════════════════════════════════════
#  § 3  Progress
# ═══════════════════════════════════════════════════════════════════════════

class _Bar:
    def __init__(self, total):
        self.total, self.value = total, 0

    def update(self, n):
        self.value += n

    def update_absolute(self, value, total=None):
        self.value = value


def _record_bars(monkeypatch, *modules):
    """Replace ProgressBar in the `comfy.utils` each module holds: other test
    files swap the shared stub, so it may not be the one importable here."""
    bars = []
    for module in modules:
        monkeypatch.setattr(module.comfy.utils, "ProgressBar",
                            lambda total: bars.append(_Bar(total)) or bars[-1])
    return bars


def test_write_nodes_advance_their_progress_bars(tmp_path, monkeypatch):
    """"Write", "Write EXR" and "Write EXR Passes" showed no progress: a
    241-frame EXR sequence ran for a minute with nothing moving."""
    pytest.importorskip("OpenEXR")
    import radiance.nodes.io.write as io_nodes
    import radiance.nodes.vfx.multipass.master as master

    bars = _record_bars(monkeypatch, io_nodes, master)
    io_nodes.RadianceWrite().write(torch.rand(5, 8, 8, 3), str(tmp_path / "w"),
                                   "SEQ │ PNG (8-bit)", filename="shot")
    io_nodes.RadianceEXRMultiPart().write_multipart("mp", torch.rand(3, 8, 8, 3),
                                                    output_path=str(tmp_path / "mp"))
    master.RadianceEXRPassesWriter().write_passes({"beauty": torch.rand(2, 8, 8, 3)}, "p",
                                                 output_path=str(tmp_path / "p"))
    assert [(b.total, b.value) for b in bars] == [(5, 5), (3, 3), (2, 2)]


def test_a_video_reports_each_frame_and_a_cancel_leaves_no_partial_file(tmp_path):
    from radiance.io.writer import _ffmpeg_ok, dispatch_write
    if not _ffmpeg_ok():
        pytest.skip("ffmpeg unavailable")

    def frames():
        return (np.full((16, 16, 3), 0.5, np.float32) for _ in range(4))

    seen = []
    dispatch_write(frames(), str(tmp_path / "a"), "VID │ MP4 (H.264)",
                   24.0, 23, "ZIP", 1001, 4, "", True, frame_count=4,
                   on_frame=lambda written, total: seen.append((written, total)))
    assert seen == [(1, 4), (2, 4), (3, 4), (4, 4)]

    class Cancelled(BaseException):
        """ComfyUI's InterruptProcessingException is a BaseException too."""

    def cancel(written, total):
        if written == 2:
            raise Cancelled

    with pytest.raises(Cancelled):
        dispatch_write(frames(), str(tmp_path / "b"), "VID │ MP4 (H.264)",
                       24.0, 23, "ZIP", 1001, 4, "", True, frame_count=4, on_frame=cancel)
    assert not list(tmp_path.glob("b*")), "the truncated video was left behind"
