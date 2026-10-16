"""Delivery integrity: success must mean the master is on disk and complete.

Covers the Phase 1 fixes from the v4 beta audit:
  * the handler checks the writer's receipt against the disk instead of
    discarding it, and records a shortfall as a warning (status "partial");
  * a sidecar the colourist asked for that failed or only approximates the
    grade is reported;
  * the viewer shows a "partial" delivery as complete-with-warnings, not as
    "FAILED undefined";
  * the DCC hand-off nodes write frames through a temporary file and report
    frames left over from an earlier, longer take.
"""
import json
import os
import re
from pathlib import Path

import pytest

from radiance.core.sequence_safety import stale_sequence_frames, stale_warning, write_atomic

ROOT = Path(__file__).resolve().parent.parent


# ── _verify_written ──────────────────────────────────────────────────────────

def _handler():
    pytest.importorskip("torch")
    from radiance.delivery import handler
    return handler


def _file(tmp_path, name, data=b"x"):
    p = tmp_path / name
    p.write_bytes(data)
    return str(p)


def _receipt(tmp_path, n=2, writer="OpenEXR", complete=True):
    files = [{"path": _file(tmp_path, f"f_{i}.exr"), "frame": i, "writer": writer, "bytes": 1}
             for i in range(n)]
    return {"files": files, "requested_frames": n, "written_frames": n if complete else n - 1,
            "complete": complete}


def test_a_clean_receipt_adds_no_warning(tmp_path):
    w = []
    _handler()._verify_written(_receipt(tmp_path), w)
    assert w == []


def test_no_receipt_is_reported_not_trusted():
    w = []
    _handler()._verify_written({}, w)
    assert len(w) == 1 and "not verified" in w[0]


def test_an_incomplete_write_is_reported(tmp_path):
    w = []
    _handler()._verify_written(_receipt(tmp_path, n=3, complete=False), w)
    assert any("2 of 3" in m for m in w)


def test_a_missing_or_empty_file_is_reported(tmp_path):
    r = _receipt(tmp_path, n=3)
    os.remove(r["files"][0]["path"])
    Path(r["files"][1]["path"]).write_bytes(b"")
    w = []
    _handler()._verify_written(r, w)
    assert any("2 written file(s) are missing or empty" in m for m in w)


def test_the_opencv_exr_fallback_is_reported(tmp_path):
    r = _receipt(tmp_path, writer="OpenCV fallback (no OpenEXR): header metadata and ZIP compression not written")
    w = []
    _handler()._verify_written(r, w)
    assert any("OpenCV fallback" in m for m in w)


def test_warnings_are_deduplicated_across_frames(tmp_path):
    r = _receipt(tmp_path, n=50, writer="OpenCV fallback (x): y")
    w = []
    h = _handler()
    h._verify_written(r, w)
    h._verify_written(r, w)
    assert len(w) == 1


# ── _write_json_atomic ───────────────────────────────────────────────────────

def test_the_meta_sidecar_is_replaced_whole(tmp_path):
    h = _handler()
    p = str(tmp_path / "m.json")
    h._write_json_atomic(p, {"a": 1})
    h._write_json_atomic(p, {"a": 2})
    assert json.loads(Path(p).read_text()) == {"a": 2}
    assert [f.name for f in tmp_path.iterdir()] == ["m.json"]


def test_a_failed_meta_write_keeps_the_earlier_file(tmp_path):
    h = _handler()
    p = str(tmp_path / "m.json")
    h._write_json_atomic(p, {"a": 1})
    with pytest.raises(TypeError):
        h._write_json_atomic(p, {"a": object()})
    assert json.loads(Path(p).read_text()) == {"a": 1}
    assert [f.name for f in tmp_path.iterdir()] == ["m.json"]


# ── write_atomic / stale frames ──────────────────────────────────────────────

def test_write_atomic_replaces_only_on_success(tmp_path):
    final = tmp_path / "frame.1001.exr"
    final.write_bytes(b"old master")

    def boom(tmp):
        tmp.write_bytes(b"partial")
        raise RuntimeError("disk full")

    with pytest.raises(RuntimeError):
        write_atomic(final, boom)
    assert final.read_bytes() == b"old master"
    assert [f.name for f in tmp_path.iterdir()] == ["frame.1001.exr"]


def test_write_atomic_keeps_the_extension_the_writer_needs(tmp_path):
    seen = []
    final = tmp_path / "a.png"
    result = write_atomic(final, lambda tmp: (seen.append(tmp), tmp.write_bytes(b"ok"), "note")[2])
    assert seen[0].suffix == ".png" and seen[0].name.startswith(".")
    assert result == "note" and final.read_bytes() == b"ok"


def test_write_atomic_rejects_a_writer_that_produced_nothing(tmp_path):
    with pytest.raises(RuntimeError):
        write_atomic(tmp_path / "a.exr", lambda tmp: None)


def test_stale_frames_from_a_longer_earlier_take_are_found(tmp_path):
    for n in range(1001, 1011):
        (tmp_path / f"shot.{n}.exr").write_bytes(b"x")
    (tmp_path / "other.1001.exr").write_bytes(b"x")
    stale = stale_sequence_frames(tmp_path, "shot", ".", ".exr", range(1001, 1005))
    assert stale == [1005, 1006, 1007, 1008, 1009, 1010]
    msg = stale_warning(stale, "shot")
    assert "6 frame(s)" in msg and "1005-1010" in msg
    assert stale_warning([], "shot") == ""


def test_no_stale_frames_when_the_range_matches(tmp_path):
    for n in range(1001, 1004):
        (tmp_path / f"shot_{n}.exr").write_bytes(b"x")
    assert stale_sequence_frames(tmp_path, "shot", "_", ".exr", range(1001, 1004)) == []


# ── wiring (source pins: the endpoint is a closure and needs ComfyUI to run) ─

def test_the_handler_passes_a_receipt_and_verifies_it():
    src = (ROOT / "delivery" / "handler.py").read_text(encoding="utf-8")
    assert "receipt=_write_receipt" in src
    assert "_verify_written(_write_receipt, warnings)" in src


@pytest.mark.parametrize("needle", [
    "The CDL sidecar approximates the grade",
    "The CDL sidecar was not written",
    "The AMF sidecar was not written",
    "A sidecar step failed",
])
def test_the_handler_records_sidecar_shortfalls(needle):
    src = (ROOT / "delivery" / "handler.py").read_text(encoding="utf-8")
    assert needle in src


def test_the_meta_sidecar_records_what_was_delivered():
    src = (ROOT / "delivery" / "handler.py").read_text(encoding="utf-8")
    for key in ('"fps"', '"range_in"', '"write_colorspace"', '"writers"', '"warnings"'):
        assert key in src
    assert "_write_json_atomic(os.path.splitext(path)[0] + \"_meta.json\"" in src


def test_the_viewer_shows_a_partial_delivery_as_complete_with_warnings():
    src = (ROOT / "js" / "radiance_viewer.js").read_text(encoding="utf-8")
    assert "result.status === 'success' || result.status === 'partial'" in src
    assert "EXPORT COMPLETE WITH WARNINGS" in src
    assert re.search(r"for \(const w of _dvWarnings\) this\._termLog\('warn'", src)
    assert "result.error || result.message" in src


@pytest.mark.parametrize("rel", ["nodes/pipeline/dcc.py", "nodes/pipeline/studio_integrations.py"])
def test_the_dcc_nodes_write_frames_atomically(rel):
    src = (ROOT / rel).read_text(encoding="utf-8")
    assert "write_atomic(" in src and "stale_warning(" in src
    # No frame is written straight onto its final name any more.
    assert not re.search(r"_save_(exr|pil_image)\([^\n]*\bfpath\b", src)
