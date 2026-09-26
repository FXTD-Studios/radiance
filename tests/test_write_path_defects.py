"""Confirmed defects in the write path, each with the behaviour that proves it.

These are behavioural, not structural: every one of them writes a file, or
fails to, and then looks at what is on disk. A structural assertion that the
code "now calls the right helper" is the kind of test this repository has been
burned by.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from radiance.io import writer as W


def frames(n=3, h=16, w=16, c=3, value=0.5):
    return [np.full((h, w, c), value, np.float32) for _ in range(n)]


def _has_ffmpeg() -> bool:
    try:
        return W._ffmpeg_ok()
    except Exception:
        return False


needs_ffmpeg = pytest.mark.skipif(not _has_ffmpeg(), reason="ffmpeg not available")


# ═══════════════════════════════════════════════════════════════════════════
#  § 1  The video branch ignored overwrite=False and never made its directory
# ═══════════════════════════════════════════════════════════════════════════

@needs_ffmpeg
def test_a_video_write_with_overwrite_false_does_not_destroy_the_master(tmp_path):
    """The one that costs a day: re-queueing a graph overwrote an approved master.

    IMG went through `resolve_output_path`, SEQ did its own mkdir and
    `_unique_path`; VID did neither, built its path by string concatenation and
    handed ffmpeg `-y` unconditionally, so `overwrite=False` was honoured for
    two branches out of three.
    """
    base = tmp_path / "shot"
    first = W._save_video_ffmpeg(iter(frames(6)), str(base), "MP4 (H.264)",
                                 24.0, 30, "", overwrite=False)
    approved = Path(first).read_bytes()
    assert Path(first).name == "shot.mp4"

    second = W._save_video_ffmpeg(iter(frames(6, value=0.1)), str(base),
                                  "MP4 (H.264)", 24.0, 30, "", overwrite=False)
    assert second != first, "the second encode landed on top of the first"
    assert Path(first).read_bytes() == approved, "the approved master was rewritten"
    assert Path(second).exists()


@needs_ffmpeg
def test_a_video_write_creates_its_parent_directory(tmp_path):
    """`mkdir -p` was promised and only IMG and SEQ did it; video failed with a
    bare ffmpeg exit status on a directory that did not exist yet."""
    target = tmp_path / "dailies" / "2026_09_18" / "shot"
    out = W._save_video_ffmpeg(iter(frames(4)), str(target), "MP4 (H.264)",
                               24.0, 30, "", overwrite=True)
    assert Path(out).exists()


@needs_ffmpeg
def test_a_video_write_with_overwrite_true_still_replaces_in_place(tmp_path):
    """The default has to stay the default: `overwrite=True` means one path."""
    base = tmp_path / "shot"
    a = W._save_video_ffmpeg(iter(frames(4)), str(base), "MP4 (H.264)",
                             24.0, 30, "", overwrite=True)
    b = W._save_video_ffmpeg(iter(frames(4)), str(base), "MP4 (H.264)",
                             24.0, 30, "", overwrite=True)
    assert a == b


# ═══════════════════════════════════════════════════════════════════════════
#  § 2  ffmpeg's stderr was captured and thrown away
# ═══════════════════════════════════════════════════════════════════════════

@needs_ffmpeg
def test_a_failed_encode_quotes_what_ffmpeg_said(tmp_path):
    """`subprocess.run(..., capture_output=True)` with `check=True` raises
    CalledProcessError carrying only an exit status, and the handler above
    reported that number. DNxHR refuses frames under 256x120, and saying so is
    the difference between a two-minute fix and an afternoon."""
    with pytest.raises(RuntimeError) as exc:
        W._save_video_ffmpeg(iter(frames(4, h=32, w=32)), str(tmp_path / "m"),
                             "MOV (DNxHR HQ)", 24.0, 18, "")
    message = str(exc.value)
    assert "ffmpeg said:" in message
    assert "256x120" in message, message
    assert "exit" in message


# ═══════════════════════════════════════════════════════════════════════════
#  § 3  The hardcoded 600 s encode cap
# ═══════════════════════════════════════════════════════════════════════════

def test_the_encode_has_no_timeout_by_default():
    """A fixed cap turns a long clip into a spurious failure, which is exactly
    why `core/video.py:decode` made the DECODE timeout None and said so in a
    comment. The encode kept `timeout=600`, so a master longer than ten minutes
    of wall clock was killed mid-file -- over the top of the previous version,
    because `-y` had already truncated it."""
    import inspect
    assert inspect.signature(W._save_video_ffmpeg).parameters["timeout"].default is None
    # And nothing between the node and the encoder reintroduces a cap.
    assert inspect.signature(W.dispatch_write).parameters["encode_timeout"].default is None


class _NeverFinishes:
    """A subprocess that accepts frames and never exits."""
    returncode = None

    def __init__(self, *a, **k):
        import io
        self.stdin = io.BytesIO()
        self.stderr = io.BytesIO(b"")
        self.killed = False

    def wait(self, timeout=None):
        if timeout is not None:
            raise subprocess.TimeoutExpired("ffmpeg", timeout)
        return 0

    def kill(self):
        self.killed = True
        self.returncode = -9


def test_a_timed_out_encode_removes_the_partial_file(tmp_path, monkeypatch):
    """When a caller does set a cap and it fires, what is on disk is a
    truncated master that `-y` has already written over the approved one. It is
    removed, and the error says how far it got."""
    target = tmp_path / "shot.mp4"
    target.write_bytes(b"not really a movie")
    monkeypatch.setattr(W.subprocess, "Popen", _NeverFinishes)

    with pytest.raises(RuntimeError) as exc:
        W._save_video_ffmpeg(iter(frames(3)), str(tmp_path / "shot"),
                             "MP4 (H.264)", 24.0, 30, "", timeout=0.01)

    assert not target.exists(), "a truncated master was left on disk"
    assert "did not finish encoding" in str(exc.value)
    assert "3 frame(s)" in str(exc.value)


# ═══════════════════════════════════════════════════════════════════════════
#  § 4  exr_compression was threaded through four layers and never used
# ═══════════════════════════════════════════════════════════════════════════

_openexr = pytest.importorskip("OpenEXR")


@pytest.mark.parametrize("label,attr", [
    ("ZIP", "ZIP_COMPRESSION"),
    ("PIZ", "PIZ_COMPRESSION"),
    ("DWAA", "DWAA_COMPRESSION"),
    ("Uncompressed", "NO_COMPRESSION"),
])
def test_the_chosen_exr_compression_reaches_the_file(tmp_path, label, attr):
    """`_save_exr` hardcoded ZIP. Selecting DWAA for a 2000-frame dailies
    sequence silently got ZIP, roughly 5x the size and time, and the node
    reported success."""
    path = tmp_path / f"{label}.exr"
    W._save_exr(np.random.rand(64, 64, 3).astype(np.float32), path,
                half=False, compression=label)
    assert _openexr.File(str(path)).header()["compression"] == getattr(_openexr, attr)


def test_a_sequence_write_honours_the_compression_widget(tmp_path):
    """End to end through the parameter's whole journey: write_frames ->
    dispatch_write -> the SEQ EXR branch."""
    out = tmp_path / "shot"
    W.write_frames(image=torch.rand(3, 32, 32, 3),
                   output_path=str(out), format="SEQ │ EXR (32-bit float)",
                   exr_compression="PIZ", overwrite=True)
    written = sorted(out.glob("*.exr"))
    assert len(written) == 3
    for f in written:
        assert _openexr.File(str(f)).header()["compression"] == _openexr.PIZ_COMPRESSION


def test_an_unknown_compression_falls_back_to_zip_rather_than_failing(tmp_path):
    """Losing a render because a label is not in the table is worse than
    writing it losslessly and saying so."""
    path = tmp_path / "odd.exr"
    W._save_exr(np.zeros((8, 8, 3), np.float32), path, half=True,
                compression="Squeezy")
    assert _openexr.File(str(path)).header()["compression"] == _openexr.ZIP_COMPRESSION


# ═══════════════════════════════════════════════════════════════════════════
#  § 5  overwrite=False applied _unique_path per frame
# ═══════════════════════════════════════════════════════════════════════════

def test_a_partial_re_render_keeps_one_stem_for_the_whole_sequence(tmp_path):
    """The defect: `_unique_path` ran per frame, so frames that already existed
    became `shot_1001_001.exr` and frames that did not stayed `shot_1005.exr`
    -- neither the old sequence nor the new one, and the `%04d` glob matched
    neither. A collision is a property of the sequence.
    """
    out = tmp_path / "shot"
    out.mkdir()
    # A previous render that only got as far as frame 1003.
    for i in range(3):
        (out / f"shot_{1001 + i}.png").write_bytes(b"old")

    saved, count = W.dispatch_write(
        frames(6), str(out), "SEQ │ PNG (8-bit)",
        24.0, 18, "ZIP", 1001, 4, "", False, frame_count=6)

    assert count == 6
    fresh = sorted(p.name for p in Path(saved).glob("*.png")
                   if p.read_bytes() != b"old")
    stems = {name.rsplit("_", 1)[0] for name in fresh}
    assert len(stems) == 1, f"the new sequence is split across stems {stems}"
    stem = stems.pop()
    # And the whole new sequence is contiguous under one %04d glob.
    assert fresh == [f"{stem}_{1001 + i}.png" for i in range(6)]
    # The old frames are untouched.
    for i in range(3):
        assert (out / f"shot_{1001 + i}.png").read_bytes() == b"old"


def test_overwrite_true_still_writes_straight_over_a_sequence(tmp_path):
    out = tmp_path / "shot"
    out.mkdir()
    (out / "shot_1001.png").write_bytes(b"old")
    saved, _ = W.dispatch_write(frames(2), str(out), "SEQ │ PNG (8-bit)",
                                24.0, 18, "ZIP", 1001, 4, "", True, frame_count=2)
    assert (out / "shot_1001.png").read_bytes() != b"old"
    assert sorted(p.name for p in Path(saved).glob("*.png")) == [
        "shot_1001.png", "shot_1002.png"]


# ═══════════════════════════════════════════════════════════════════════════
#  § 6  A relative output_path resolved against the process working directory
# ═══════════════════════════════════════════════════════════════════════════

def test_a_relative_output_path_is_anchored_not_left_to_the_cwd(tmp_path, monkeypatch):
    """`Path(base)` with no anchoring resolves against the process working
    directory, which for ComfyUI is the install root -- the bug
    `core/system/path_utils.resolve_output_path` was written for, and which
    RadianceFlipbookGIF already routes through."""
    comfy_out = tmp_path / "comfy" / "output"
    comfy_out.mkdir(parents=True)
    elsewhere = tmp_path / "cwd"
    elsewhere.mkdir()
    monkeypatch.setattr(sys.modules["folder_paths"], "get_output_directory",
                        lambda: str(comfy_out), raising=False)
    monkeypatch.chdir(elsewhere)

    path = W.resolve_output_path("dailies/shot_v0001", ".png", True)

    assert Path(path).is_relative_to(comfy_out), path
    assert not (elsewhere / "dailies").exists(), "wrote into the working directory"


def test_an_absolute_output_path_is_honoured_untouched(tmp_path):
    """Someone who types a full path means it."""
    target = tmp_path / "vendor" / "shot_v0001"
    path = W.resolve_output_path(str(target), ".exr", True)
    assert path == tmp_path / "vendor" / "shot_v0001.exr"
    assert path.parent.is_dir()


def test_a_relative_sequence_directory_is_anchored(tmp_path, monkeypatch):
    comfy_out = tmp_path / "comfy" / "output"
    comfy_out.mkdir(parents=True)
    elsewhere = tmp_path / "cwd"
    elsewhere.mkdir()
    monkeypatch.setattr(sys.modules["folder_paths"], "get_output_directory",
                        lambda: str(comfy_out), raising=False)
    monkeypatch.chdir(elsewhere)

    saved, _ = W.dispatch_write(frames(2), "dailies/shot", "SEQ │ PNG (8-bit)",
                                24.0, 18, "ZIP", 1001, 4, "", True, frame_count=2)
    assert Path(saved).is_relative_to(comfy_out), saved
    assert not (elsewhere / "dailies").exists()


# ═══════════════════════════════════════════════════════════════════════════
#  § 7  overwrite defaulted to True while INPUT_TYPES declared False
# ═══════════════════════════════════════════════════════════════════════════

def test_the_write_engine_defaults_to_the_value_the_widget_declares():
    """ComfyUI supplies the widget value, so graphs were safe; every
    programmatic caller -- the Digital Cinema shim, delivery/handler.py -- got
    destructive overwrite from a signature that disagreed with its own UI."""
    import inspect
    assert inspect.signature(W.write_frames).parameters["overwrite"].default is False


def test_a_programmatic_write_does_not_silently_replace_an_existing_file(tmp_path):
    base = tmp_path / "shot"
    first, _ = W.write_frames(image=torch.full((1, 8, 8, 3), 0.5),
                              output_path=str(base), format="IMG │ PNG (8-bit)")
    original = Path(first).read_bytes()
    second, _ = W.write_frames(image=torch.full((1, 8, 8, 3), 0.9),
                               output_path=str(base), format="IMG │ PNG (8-bit)")
    assert second != first
    assert Path(first).read_bytes() == original


def test_the_node_signature_matches_its_input_types():
    """The node and the engine must not drift apart again."""
    import inspect
    from radiance.nodes.io.write import RadianceWrite
    declared = RadianceWrite.INPUT_TYPES()["optional"]["overwrite"][1]["default"]
    signature = inspect.signature(RadianceWrite.write).parameters["overwrite"].default
    assert signature == declared


# ═══════════════════════════════════════════════════════════════════════════
#  § 8  coerce_to_frames swallowed every decode error in its list loop
# ═══════════════════════════════════════════════════════════════════════════

def test_a_list_that_cannot_be_read_reports_why_each_element_failed():
    """`except Exception: continue` discarded the real reason -- a missing
    file, a LATENT that needed a VAE Decode -- and ended on a generic "Cannot
    extract frames from list" that names nothing anyone can act on."""
    with pytest.raises(ValueError) as exc:
        W.coerce_to_frames([{"samples": object()}, "/no/such/plate.exr"])

    message = str(exc.value)
    assert "VAE Decode" in message, message
    assert "/no/such/plate.exr" in message, message
