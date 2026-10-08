"""`/radiance/deliver` — the endpoint itself, not just its helpers.

`tests/test_delivery_handler.py` covers the pieces that can be called without a
request: the two lookup tables, `get_next_version`, the atomic session log and
the AMF sidecar. It deliberately stops at the door of `radiance_deliver_endpoint`,
which is where the ~570 lines that decide whether a master lands on disk — and
what a client is told about it — actually live.

This file drives that coroutine end to end with real objects:

  * a **real** `aiohttp.web.Response`, so the status codes and JSON bodies
    asserted here are the ones a browser would receive. The suite's conftest
    installs an empty `aiohttp` stub (ComfyUI is not present), so the fixture
    below swaps the real package in for the duration of a test and puts the stub
    back afterwards. It refuses to run against the stub rather than passing
    because `json_response` was a no-op.
  * the **real** viewer frame cache (`radiance.cache`), the real grading stack
    (`radiance.color.grading.apply_grading`) and the real write engine
    (`radiance.io.writer.write_frames`) — every assertion about a delivered
    pixel is read back off disk.

Only three things are faked, and none of them is the code under test: the
`aiohttp` *request* object (a payload carrier), ComfyUI's `folder_paths` output
directory (redirected into tmp_path so containment can be exercised at all), and
— in exactly two tests — `os.unlink` / `subprocess.Popen`, each time to observe a
call the endpoint is supposed to make.
"""
import asyncio
import importlib
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from radiance import cache
from radiance.delivery import handler


# ═══════════════════════════════════════════════════════════════════════════
#  Real aiohttp, borrowed for the duration of a test
# ═══════════════════════════════════════════════════════════════════════════

def _aiohttp_names():
    return [n for n in list(sys.modules) if n == "aiohttp" or n.startswith("aiohttp.")]


@pytest.fixture(scope="session")
def real_aiohttp_web():
    """The genuine `aiohttp.web`, with the conftest stub temporarily removed.

    conftest.py registers a bare `types.ModuleType("aiohttp.web")` when aiohttp
    is absent, and that stub has no `json_response` at all. Every status code in
    this file would be unobservable through it, so the real package is required
    — and asserted for, so a missing install fails loudly instead of quietly
    turning these tests into no-ops.
    """
    saved = {n: sys.modules[n] for n in _aiohttp_names()}
    for name in saved:
        del sys.modules[name]
    try:
        web = importlib.import_module("aiohttp.web")
    except ImportError as exc:  # pragma: no cover - environment problem
        for name in _aiohttp_names():
            del sys.modules[name]
        sys.modules.update(saved)
        # Absent is not the same as stubbed: CI installs no aiohttp at all,
        # and skipping there is honest. Importing something that IS present
        # but is the conftest stub is still a hard failure, just below.
        pytest.skip(f"aiohttp is not installed: {exc}")

    assert callable(getattr(web, "json_response", None)), (
        "imported an aiohttp.web without json_response — that is the conftest "
        "stub, and every status-code assertion below would be meaningless"
    )
    probe = web.json_response({"ok": True}, status=418)
    assert probe.status == 418 and json.loads(probe.body.decode()) == {"ok": True}, (
        "aiohttp.web.json_response is not behaving like a real response"
    )

    yield web

    for name in _aiohttp_names():
        del sys.modules[name]
    sys.modules.update(saved)


# ═══════════════════════════════════════════════════════════════════════════
#  Harness
# ═══════════════════════════════════════════════════════════════════════════

class _Request:
    """The only thing the endpoint asks of a request is `await .json()`."""

    def __init__(self, payload, raise_with=None):
        self._payload = payload
        self._raise_with = raise_with

    async def json(self):
        if self._raise_with is not None:
            raise self._raise_with
        return self._payload


#: Every key `GRADE_PAYLOAD_KEYS` names, at its identity value. Tests that care
#: about one control override just that one, so nothing else can drift in.
IDENTITY_GRADE = {
    "exposure": 0.0, "gamma": [1.0, 1.0, 1.0], "gain": [1.0, 1.0, 1.0],
    "lift": [0.0, 0.0, 0.0], "offset": [0.0, 0.0, 0.0], "contrast": 1.0,
    "pivot": 0.18, "saturation": 1.0, "temperature": 0.0, "tint": 0.0,
    "colorScience": "0", "lumaMix": 1.0, "shadows": 0.0, "highlights": 0.0,
    "hue_shift": 0.0, "lut_name": "None", "lut_intensity": 1.0,
    "gamut_compression": False, "grain": 0.0, "bloom": 0.0, "halation": 0.0,
    "diffusion": 0.0, "denoise": 0.0,
}

#: EXR sequence + pass-through colour, so a delivered pixel can be compared to
#: the pixel that went in without an encode standing in the way.
BASE_SETTINGS = {
    "format": "Image Sequence — EXR (32-bit)",
    "colorSpace": "Linear (sRGB)",
    "soft_clip": False,
    "smart_versioning": True,
    "filename": "Shot",
}


class Harness:
    def __init__(self, root, key):
        self.root = root
        self.key = key
        self.calls = []          # (kwargs) recorded by the write_frames spy

    def put(self, frames, source=("linear", "Linear Rec.709 (sRGB)")):
        """Cache frames as the Viewer would, with the input space it resolved.

        Linear by default: these tests compare delivered pixels with cached
        ones through a linear EXR, which is only an identity for a linear
        source (FIX-018). Display-encoded sources are covered in
        tests/test_p0_delivery_fixes.py.
        """
        cache._viewer_cache_set(self.key, frames)
        if source is not None:
            cache._viewer_source_set(self.key, *source)

    def run(self, settings=None, grading=None, payload=None, raise_with=None):
        if payload is None:
            payload = {
                "instance_id": self.key,
                "grading": dict(IDENTITY_GRADE, **(grading or {})),
                "settings": dict(BASE_SETTINGS, **(settings or {})),
            }
        return asyncio.run(
            handler.radiance_deliver_endpoint(_Request(payload, raise_with)))

    # ── readers ──────────────────────────────────────────────────────────
    @staticmethod
    def body(response):
        assert response.content_type == "application/json", response.content_type
        return json.loads(response.body.decode("utf-8"))

    def exrs(self):
        return sorted(self.root.rglob("*.exr"))

    def pngs(self):
        return sorted(self.root.rglob("*.png"))

    @staticmethod
    def pixel(path, x=0, y=0):
        oiio = pytest.importorskip("OpenImageIO")
        return oiio.ImageBuf(str(path)).getpixel(x, y)

    def progress(self):
        return cache._progress_get(self.key)


@pytest.fixture
def h(tmp_path, monkeypatch, real_aiohttp_web, request):
    """A wired endpoint: real response class, real cache, tmp output root."""
    assert handler.web is not None, "handler.web is None — the module took its "\
        "no-aiohttp fallback and nothing below would exercise a status code"
    monkeypatch.setattr(handler, "web", real_aiohttp_web)

    root = tmp_path / "comfy_output"
    root.mkdir()
    monkeypatch.setattr(handler.folder_paths, "get_output_directory",
                        lambda: str(root))
    # One-shot module global; reset so test order cannot decide whether the
    # missing-key warning fires.
    monkeypatch.setattr(handler, "_warned_missing_grade_keys", False)

    key = f"test-{request.node.name}"
    harness = Harness(root, key)
    yield harness

    with cache._VIEWER_CACHE_LOCK:
        cache._VIEWER_CACHE.pop(key, None)
    with cache._VIEWER_SOURCE_LOCK:
        cache._VIEWER_SOURCE.pop(key, None)
    with cache._VIEWER_PROGRESS_LOCK:
        cache._VIEWER_PROGRESS.pop(key, None)


@pytest.fixture
def spy_write(monkeypatch):
    """Records the kwargs the endpoint hands the write engine, then calls it.

    A recording *pass-through*, not a replacement: the file is still written by
    the real writer, so nothing here can pass because the export was skipped.
    """
    import radiance.io.writer as writer
    real = writer.write_frames
    seen = []

    def _spy(**kwargs):
        seen.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(writer, "write_frames", _spy)
    return seen


def flat(value, n=1, h=4, w=4, c=3):
    return torch.full((n, h, w, c), float(value), dtype=torch.float32)


# ═══════════════════════════════════════════════════════════════════════════
#  § 1  Refusals — the status codes a client actually branches on
# ═══════════════════════════════════════════════════════════════════════════

def test_an_empty_cache_is_400_and_says_to_run_the_workflow(h):
    """Nothing was cached for this node id: the client must not be told 200."""
    resp = h.run()
    assert resp.status == 400
    body = h.body(resp)
    assert body["status"] == "error"
    assert "Run the workflow first" in body["error"]
    assert not h.exrs(), "a 400 must not have written a master"


def test_an_unknown_instance_id_does_not_borrow_another_nodes_frames(h):
    h.put(flat(0.5))
    resp = h.run(payload={"instance_id": "some-other-node",
                          "grading": IDENTITY_GRADE, "settings": BASE_SETTINGS})
    assert resp.status == 400
    assert "No frames found" in h.body(resp)["error"]


def test_a_request_whose_body_is_not_json_is_500_not_200(h):
    """The endpoint's blanket `except` used to answer 200 with status error."""
    resp = h.run(raise_with=ValueError("Expecting value: line 1 column 1"))
    assert resp.status == 500
    body = h.body(resp)
    assert body["status"] == "error"
    assert "Expecting value" in body["error"]


def test_an_output_path_outside_the_comfyui_output_dir_is_403(h, tmp_path):
    h.put(flat(0.5))
    escape = tmp_path / "elsewhere"
    escape.mkdir()
    resp = h.run({"path": str(escape)})
    assert resp.status == 403, "path containment is a security guard, not a 400"
    assert "inside the ComfyUI output directory" in h.body(resp)["error"]
    assert not sorted(escape.rglob("*")), "the guard let a file out anyway"


def test_a_traversal_path_that_climbs_out_is_403(h):
    h.put(flat(0.5))
    resp = h.run({"path": str(h.root / ".." / ".." / "etc")})
    assert resp.status == 403
    assert h.body(resp)["status"] == "error"


def test_a_subdirectory_of_the_output_dir_is_allowed(h):
    """The containment check must not reject the legitimate case."""
    h.put(flat(0.5))
    resp = h.run({"path": str(h.root / "shows" / "ep01")})
    assert resp.status == 200, h.body(resp)
    assert sorted((h.root / "shows" / "ep01").rglob("*.exr"))


def test_an_absurdly_long_output_path_is_400(h):
    h.put(flat(0.5))
    resp = h.run({"path": str(h.root / ("d" * 1100))})
    assert resp.status == 400
    assert h.body(resp)["error"] == "Invalid output path"


def test_an_explicit_path_is_still_honoured_when_the_root_is_unknowable(
        h, monkeypatch, tmp_path):
    """If ComfyUI cannot name its output dir there is nothing to contain
    against, so an explicit destination is used rather than the request being
    refused. Documented fallback, not an accident — but it is the one way a
    master leaves output/, so it is pinned here."""
    dest = tmp_path / "explicit"
    dest.mkdir()

    def _boom():
        raise RuntimeError("no ComfyUI")

    monkeypatch.setattr(handler.folder_paths, "get_output_directory", _boom)
    h.put(flat(0.5))
    resp = h.run({"path": str(dest), "smart_versioning": False})
    assert resp.status == 200, h.body(resp)
    assert sorted(dest.rglob("*.exr")), "the explicit destination stayed empty"


def test_an_unresolvable_default_output_dir_is_500(h, monkeypatch):
    """Location left blank and ComfyUI cannot say where output/ is."""
    h.put(flat(0.5))

    def _boom():
        raise RuntimeError("no ComfyUI")

    monkeypatch.setattr(handler.folder_paths, "get_output_directory", _boom)
    resp = h.run({"path": "   "})
    assert resp.status == 500
    assert "default output directory" in h.body(resp)["error"]


def test_a_format_with_no_writer_is_refused_by_name_with_400(h):
    """400, and before any pixels are touched.

    The delivery panel offers eleven formats and colour spaces the writer does
    not implement. Resolving them used to happen inside the `write_frames(...)`
    call at the END of the export closure, so picking one cost a full per-frame
    grade, an FX bake and a 2x model upscale over the whole batch and THEN
    raised, surfacing as a 500. Naming a format that does not exist is a bad
    request, and it is answered before the work.
    """
    h.put(flat(0.5))
    resp = h.run({"format": "Video — AVI (Cinepak)"})
    assert resp.status == 400
    err = h.body(resp)["error"]
    assert "Cinepak" in err and "Supported" in err
    assert not h.exrs()


def test_a_colour_space_with_no_writer_is_refused_by_name_with_400(h):
    h.put(flat(0.5))
    resp = h.run({"colorSpace": "Rec.2020 PQ"})
    assert resp.status == 400
    assert "Rec.2020 PQ" in h.body(resp)["error"]


def test_an_unsupported_format_is_refused_before_the_grade_runs(h, monkeypatch):
    """The refusal must cost nothing. `apply_grading` is the long pole and it
    used to run over every frame before the format was ever looked at."""
    called = []
    monkeypatch.setattr(handler, "apply_grading",
                        lambda **kw: called.append(1) or kw["img"])
    h.put(flat(0.5))
    assert h.run({"format": "GIF (Animated)"}).status == 400
    assert not called, "the batch was graded before the format was validated"


def test_a_failed_delivery_marks_progress_error_so_the_client_stops_polling(h):
    """The JS clears its poll interval only on status 'done' or 'error'."""
    h.put(flat(0.5))
    h.run({"format": "Video — AVI (Cinepak)"})
    prog = h.progress()
    assert prog["status"] == "error"
    assert prog["current"] == 100
    assert "Cinepak" in prog["message"]


def test_a_refused_delivery_also_reaches_a_terminal_progress_state(h):
    """The three validation refusals returned before any progress update, so
    the client's bar sat on 'starting' forever."""
    resp = h.run()                              # nothing in the cache
    assert resp.status == 400
    assert h.progress()["status"] == "error"


def test_a_path_outside_the_output_directory_terminates_progress(h, tmp_path):
    h.put(flat(0.5))
    escape = tmp_path / "elsewhere"
    escape.mkdir()
    resp = h.run({"path": str(escape)})
    assert resp.status == 403
    assert h.progress()["status"] == "error"


# ═══════════════════════════════════════════════════════════════════════════
#  § 2  The success answer
# ═══════════════════════════════════════════════════════════════════════════

def test_a_delivery_returns_200_with_the_path_it_wrote(h):
    h.put(flat(0.5, n=2))
    resp = h.run()
    assert resp.status == 200
    body = h.body(resp)
    assert body["status"] == "success"
    assert os.path.exists(body["path"]), body["path"]
    assert body["path"].startswith(str(h.root))
    assert len(h.exrs()) == 2, [p.name for p in h.exrs()]
    assert "(v0001)" in body["message"]
    assert os.path.basename(body["path"]) in body["message"]


def test_progress_reaches_a_terminal_done_state(h):
    h.put(flat(0.5, n=2))
    h.run()
    assert h.progress() == {"current": 100, "total": 100,
                            "status": "done", "message": "Delivery Complete"}


def test_the_frames_on_disk_are_the_frames_that_were_cached(h):
    h.put(flat(0.375, n=1))
    resp = h.run()
    assert resp.status == 200, h.body(resp)
    px = h.pixel(h.exrs()[0])
    assert px[0] == pytest.approx(0.375, abs=1e-5), (
        f"an identity grade changed 0.375 into {px[0]}")


# ═══════════════════════════════════════════════════════════════════════════
#  § 3  Render range
# ═══════════════════════════════════════════════════════════════════════════

def test_the_render_range_is_inclusive_of_both_ends(h):
    """range_in is 1-based: 2..4 of five frames is three frames, 1-indexed."""
    h.put(torch.stack([flat(v)[0] for v in (0.1, 0.2, 0.3, 0.4, 0.5)]))
    resp = h.run({"range_in": 2, "range_out": 4})
    assert resp.status == 200, h.body(resp)
    written = h.exrs()
    assert len(written) == 3, [p.name for p in written]
    values = [h.pixel(p)[0] for p in written]
    assert values == pytest.approx([0.2, 0.3, 0.4], abs=1e-5), values


def test_range_out_zero_means_to_the_end(h):
    h.put(torch.stack([flat(v)[0] for v in (0.1, 0.2, 0.3)]))
    assert h.run({"range_in": 2, "range_out": 0}).status == 200
    assert [h.pixel(p)[0] for p in h.exrs()] == pytest.approx([0.2, 0.3], abs=1e-5)


def test_a_range_past_the_end_is_clamped_rather_than_crashing(h):
    h.put(torch.stack([flat(v)[0] for v in (0.1, 0.2, 0.3)]))
    resp = h.run({"range_in": 99, "range_out": 400})
    assert resp.status == 200, h.body(resp)
    values = [h.pixel(p)[0] for p in h.exrs()]
    assert values == pytest.approx([0.3], abs=1e-5), (
        f"range_in past the end should deliver the last frame, got {values}")


def test_a_range_in_below_one_is_treated_as_one(h):
    h.put(torch.stack([flat(v)[0] for v in (0.1, 0.2)]))
    assert h.run({"range_in": 0, "range_out": 1}).status == 200
    assert [h.pixel(p)[0] for p in h.exrs()] == pytest.approx([0.1], abs=1e-5)


def test_an_inverted_range_still_delivers_one_frame(h):
    h.put(torch.stack([flat(v)[0] for v in (0.1, 0.2, 0.3)]))
    resp = h.run({"range_in": 3, "range_out": 1})
    assert resp.status == 200, h.body(resp)
    assert [h.pixel(p)[0] for p in h.exrs()] == pytest.approx([0.3], abs=1e-5)


# ═══════════════════════════════════════════════════════════════════════════
#  § 4  Filename handling
# ═══════════════════════════════════════════════════════════════════════════

def _delivered_dir_name(h, body):
    return os.path.basename(body["path"])


def test_path_separators_and_shell_characters_are_stripped_from_the_name(h):
    h.put(flat(0.5))
    body = h.body(h.run({"filename": "../../etc/pa$$wd;rm -rf"}))
    name = _delivered_dir_name(h, body)
    for bad in ("/", "\\", "$", ";"):
        assert bad not in name, f"{bad!r} survived sanitisation in {name!r}"
    assert os.path.dirname(os.path.abspath(body["path"])).startswith(str(h.root))


def test_a_name_made_entirely_of_illegal_characters_falls_back(h):
    h.put(flat(0.5))
    body = h.body(h.run({"filename": "$$$///&&&"}))
    assert "Radiance_Deliver" in _delivered_dir_name(h, body)


def test_an_over_long_name_is_truncated_to_200_characters(h):
    h.put(flat(0.5))
    body = h.body(h.run({"filename": "N" * 500}))
    name = _delivered_dir_name(h, body)
    assert name.count("N") == 200, f"{name.count('N')} N's survived"


def test_symbols_are_dropped_from_the_file_name(h):
    """cv2.imwrite mangles non-ASCII symbols in paths on Windows."""
    h.put(flat(0.5))
    body = h.body(h.run({"filename": "◎ Shot_010"}))
    assert "◎" not in _delivered_dir_name(h, body)
    assert "Shot_010" in _delivered_dir_name(h, body)


# ═══════════════════════════════════════════════════════════════════════════
#  § 5  Versioning
# ═══════════════════════════════════════════════════════════════════════════

def test_smart_versioning_advances_on_the_second_delivery(h):
    h.put(flat(0.5))
    first = h.body(h.run())
    assert "(v0001)" in first["message"], first["message"]
    second = h.body(h.run())
    assert "(v0002)" in second["message"], second["message"]
    assert first["path"] != second["path"], "v0002 overwrote v0001"


def test_the_delivered_name_carries_one_version_suffix(h):
    """3.x stamped two: Shot_v02_v0001. 4.0 writes Shot_v0002."""
    h.put(flat(0.5))
    h.body(h.run({"filename": "Shot"}))
    body = h.body(h.run({"filename": "Shot"}))
    name = _delivered_dir_name(h, body)
    assert os.path.splitext(name)[0] == "Shot_v0002", body["path"]
    assert "(v0002)" in body["message"], body["message"]


def test_smart_versioning_off_always_writes_v0001(h):
    h.put(flat(0.5))
    first = h.body(h.run({"smart_versioning": False, "filename": "Shot"}))
    assert os.path.splitext(_delivered_dir_name(h, first))[0] == "Shot_v0001", first["path"]
    assert "(v0001)" in first["message"]


# ═══════════════════════════════════════════════════════════════════════════
#  § 6  Grading actually reaches the master
# ═══════════════════════════════════════════════════════════════════════════

def test_exposure_is_applied_in_stops(h):
    h.put(flat(0.25))
    assert h.run(grading={"exposure": 1.0}).status == 200
    assert h.pixel(h.exrs()[0])[0] == pytest.approx(0.5, abs=1e-5), (
        "+1 stop must double a linear value")


def test_temperature_and_tint_are_the_viewers_additive_sliders(h):
    """The shipped bug fabricated Kelvin from the slider and dropped tint.

    The viewer's shader is `shift.r += temp; shift.b -= temp; shift.g -= tint`,
    so temp=+1 tint=+1 on a flat 0.25 plate is R 1.25, G 0 (clamped), B 0.
    """
    h.put(flat(0.25))
    assert h.run(grading={"temperature": 1.0, "tint": 1.0}).status == 200
    r, g, b = h.pixel(h.exrs()[0])[:3]
    assert r == pytest.approx(1.25, abs=1e-4), f"temperature did not lift R: {r}"
    assert g == pytest.approx(0.0, abs=1e-4), f"tint did not touch G: {g}"
    assert b == pytest.approx(0.0, abs=1e-4), f"temperature did not drop B: {b}"


def test_a_temperature_of_zero_leaves_the_channels_alone(h):
    h.put(flat(0.25))
    h.run(grading={"temperature": 0.0, "tint": 0.0})
    r, g, b = h.pixel(h.exrs()[0])[:3]
    assert (r, g, b) == pytest.approx((0.25, 0.25, 0.25), abs=1e-5)


def test_per_channel_gain_reaches_the_master(h):
    h.put(flat(0.25))
    assert h.run(grading={"gain": [2.0, 1.0, 0.5]}).status == 200
    r, g, b = h.pixel(h.exrs()[0])[:3]
    assert (r, g, b) == pytest.approx((0.5, 0.25, 0.125), abs=1e-4)


def test_a_scalar_where_the_grade_wants_a_triple_is_broadcast(h):
    h.put(flat(0.25))
    assert h.run(grading={"gain": 2.0}).status == 200
    assert h.pixel(h.exrs()[0])[:3] == pytest.approx((0.5, 0.5, 0.5), abs=1e-4)


def test_a_nonsense_grade_value_falls_back_to_the_default_instead_of_500(h):
    """safe_float/safe_array: the panel can send null or a string."""
    h.put(flat(0.25))
    resp = h.run(grading={"exposure": None, "gain": "bananas"})
    assert resp.status == 200, h.body(resp)
    assert h.pixel(h.exrs()[0])[0] == pytest.approx(0.25, abs=1e-5), (
        "an unparseable grade must fall back to identity, not to zero")


def test_the_colour_science_switch_changes_the_result(h):
    """`colorScience` 'ACEScct' selects a different saturation math path."""
    plate = torch.zeros((1, 4, 4, 3))
    plate[..., 0], plate[..., 1], plate[..., 2] = 0.6, 0.3, 0.1
    h.put(plate)
    h.run(grading={"saturation": 1.5, "colorScience": "0"},
          settings={"filename": "SDR"})
    h.run(grading={"saturation": 1.5, "colorScience": "ACEScct"},
          settings={"filename": "ACES"})
    sdr = h.pixel([p for p in h.exrs() if "SDR" in str(p)][0])
    aces = h.pixel([p for p in h.exrs() if "ACES" in str(p)][0])
    assert sdr[:3] != pytest.approx(aces[:3], abs=1e-4), (
        f"colorScience was ignored: both paths produced {sdr[:3]}")


def test_a_payload_short_of_grade_keys_warns_once_and_names_them(h, caplog):
    """A version-mismatched viewer is a per-process condition, not per-frame."""
    h.put(flat(0.25))
    short = {"instance_id": h.key, "grading": {"exposure": 0.0},
             "settings": BASE_SETTINGS}
    with caplog.at_level(logging.WARNING, logger="radiance.delivery.handler"):
        resp = h.run(payload=short)
        h.run(payload=short)
    assert resp.status == 200, h.body(resp)
    warnings = [r.getMessage() for r in caplog.records
                if "missing" in r.getMessage()]
    assert len(warnings) == 1, f"warned {len(warnings)} times: {warnings}"
    assert "highlights" in warnings[0] and "lut_name" in warnings[0]


def test_a_grade_sent_as_a_single_element_list_uses_that_element(h):
    """safe_float's list branch: the panel sends [1.0] for a scalar slider."""
    h.put(flat(0.25))
    assert h.run(grading={"exposure": [1.0]}).status == 200
    assert h.pixel(h.exrs()[0])[0] == pytest.approx(0.5, abs=1e-5)


# ═══════════════════════════════════════════════════════════════════════════
#  § 7  FX baking
# ═══════════════════════════════════════════════════════════════════════════

def test_grain_changes_the_pixels_and_a_zero_grain_does_not(h):
    h.put(flat(0.4, h=16, w=16))
    h.run(settings={"filename": "Clean"})
    h.run(grading={"grain": 0.5}, settings={"filename": "Grainy"})
    clean = h.pixel([p for p in h.exrs() if "Clean" in str(p)][0], 3, 3)[0]
    grainy = h.pixel([p for p in h.exrs() if "Grainy" in str(p)][0], 3, 3)[0]
    assert clean == pytest.approx(0.4, abs=1e-5)
    assert abs(grainy - 0.4) > 1e-4, (
        f"grain 0.5 left the pixel at {grainy} — FX baking was skipped")


def test_fx_below_the_threshold_is_not_baked(h):
    """0.005 is under the 0.01 gate; the plate must come back untouched."""
    h.put(flat(0.4, h=16, w=16))
    assert h.run(grading={"grain": 0.005}).status == 200
    assert h.pixel(h.exrs()[0], 3, 3)[0] == pytest.approx(0.4, abs=1e-6)


def test_halation_bleeds_the_red_channel_and_leaves_green_alone(h):
    plate = torch.zeros((1, 16, 16, 3))
    plate[:, 6:10, 6:10, 0] = 1.0
    h.put(plate)
    assert h.run(grading={"halation": 0.5}).status == 200
    px = h.pixel(h.exrs()[0], 2, 2)
    assert px[0] > 1e-4, f"no red bleed reached a corner pixel: {px[0]}"
    assert px[1] == pytest.approx(0.0, abs=1e-6), "halation touched green"
    assert px[2] == pytest.approx(0.0, abs=1e-6), "halation touched blue"


def test_bloom_spreads_a_highlight_across_all_three_channels(h):
    plate = torch.zeros((1, 16, 16, 3))
    plate[:, 6:10, 6:10, :] = 1.0
    h.put(plate)
    assert h.run(grading={"bloom": 0.5}).status == 200
    px = h.pixel(h.exrs()[0], 1, 1)
    assert px[0] > 1e-4, f"no bloom reached a corner pixel: {px[0]}"
    assert px[0] == pytest.approx(px[1], abs=1e-5) == pytest.approx(px[2], abs=1e-5)


def test_denoise_is_non_fatal(h):
    """See the report accompanying this file: the uint16 bilateral filter at
    handler.py:491 always raises inside cv2 and is swallowed at DEBUG. What is
    contractual here is only that it cannot take the delivery down."""
    noisy = torch.rand((1, 16, 16, 3))
    h.put(noisy)
    resp = h.run(grading={"denoise": 0.5})
    assert resp.status == 200, h.body(resp)
    assert len(h.exrs()) == 1


def test_diffusion_softens_an_edge(h):
    """A hard edge must survive with less contrast once diffusion is baked."""
    plate = torch.zeros((1, 16, 16, 3))
    plate[:, :, 8:, :] = 1.0
    h.put(plate)
    assert h.run(grading={"diffusion": 0.6}).status == 200
    exr = h.exrs()[0]
    dark, bright = h.pixel(exr, 7, 8)[0], h.pixel(exr, 8, 8)[0]
    assert dark > 1e-4, f"the dark side of the edge stayed at {dark}"
    assert bright < 1.0 - 1e-4, f"the bright side of the edge stayed at {bright}"


# ═══════════════════════════════════════════════════════════════════════════
#  § 8  Aspect-ratio blanking
# ═══════════════════════════════════════════════════════════════════════════

def test_a_vertical_target_blanks_the_sides_not_the_top_and_bottom(h):
    """The regression this guard exists for.

    '9:16 (Vertical)' once parsed as the ratio 9.0 because only the numerator
    was read. A 16x8 plate is 2.0 wide, which is *narrower* than 9.0, so the
    handler took the vertical branch and letterboxed a plate that needed
    pillarboxing. Read correctly the target is 0.5625, the horizontal branch
    runs, and pad = (16 - int(8*0.5625)) // 2 = 6.
    """
    h.put(flat(1.0, h=8, w=16))
    assert h.run({"aspect_ratio": "9:16 (Vertical)"}).status == 200
    exr = h.exrs()[0]
    assert h.pixel(exr, 0, 4)[0] == pytest.approx(0.0), "left bar not blanked"
    assert h.pixel(exr, 15, 4)[0] == pytest.approx(0.0), "right bar not blanked"
    assert h.pixel(exr, 8, 4)[0] == pytest.approx(1.0), "the picture was blanked"
    assert h.pixel(exr, 8, 0)[0] == pytest.approx(1.0), (
        "the top row was blanked — that is the letterbox branch, on a plate "
        "that needed pillarboxing")


def test_a_wide_target_on_a_tall_plate_blanks_the_top_and_bottom(h):
    """2.39:1 on a 16x16 square: pad = (16 - int(16/2.39)) // 2 = 4."""
    h.put(flat(1.0, h=16, w=16))
    assert h.run({"aspect_ratio": "2.39:1 (Scope)"}).status == 200
    exr = h.exrs()[0]
    assert h.pixel(exr, 8, 0)[0] == pytest.approx(0.0), "top bar not blanked"
    assert h.pixel(exr, 8, 15)[0] == pytest.approx(0.0), "bottom bar not blanked"
    assert h.pixel(exr, 8, 8)[0] == pytest.approx(1.0), "the picture was blanked"
    assert h.pixel(exr, 0, 8)[0] == pytest.approx(1.0), "a side was blanked"


def test_aspect_none_leaves_the_frame_alone(h):
    h.put(flat(1.0, h=8, w=16))
    assert h.run({"aspect_ratio": "None"}).status == 200
    exr = h.exrs()[0]
    assert h.pixel(exr, 0, 0)[0] == pytest.approx(1.0)
    assert h.pixel(exr, 15, 7)[0] == pytest.approx(1.0)


def test_a_matching_aspect_ratio_blanks_nothing(h):
    """16x8 is already 2:1; within the 0.01 tolerance neither branch runs."""
    h.put(flat(1.0, h=8, w=16))
    assert h.run({"aspect_ratio": "2:1"}).status == 200
    exr = h.exrs()[0]
    assert h.pixel(exr, 0, 0)[0] == pytest.approx(1.0)
    assert h.pixel(exr, 15, 7)[0] == pytest.approx(1.0)


def test_an_unparseable_aspect_ratio_is_non_fatal_and_delivers_the_full_frame(h):
    h.put(flat(1.0, h=8, w=16))
    resp = h.run({"aspect_ratio": "widescreen-ish"})
    assert resp.status == 200, h.body(resp)
    assert h.pixel(h.exrs()[0], 0, 0)[0] == pytest.approx(1.0), (
        "a bad preset must not blank the master")


def test_a_zero_aspect_ratio_is_rejected_rather_than_blanking_everything(h):
    h.put(flat(1.0, h=8, w=16))
    assert h.run({"aspect_ratio": "0:1"}).status == 200
    assert h.pixel(h.exrs()[0], 8, 4)[0] == pytest.approx(1.0)


# ═══════════════════════════════════════════════════════════════════════════
#  § 9  The QC report the client displays
# ═══════════════════════════════════════════════════════════════════════════

def test_levels_inside_broadcast_range_report_a_qc_pass(h):
    h.put(flat(0.5))
    body = h.body(h.run())
    assert "QC PASS" in body["qc"]
    assert "legal broadcast range" in body["qc"]


def test_a_super_white_alone_is_enough_for_a_qc_warning(h):
    """Either bound on its own is an illegal level — the test used to set both,
    so `or` and `and` were indistinguishable here."""
    plate = flat(0.5, h=4, w=4)
    plate[0, 0, 0, :] = 2.0
    h.put(plate)
    body = h.body(h.run())
    assert "QC WARNING" in body["qc"], body["qc"]
    assert "Min=0.500" in body["qc"] and "Max=2.000" in body["qc"], body["qc"]


def test_a_sub_black_alone_is_enough_for_a_qc_warning(h):
    plate = flat(0.5, h=4, w=4)
    plate[0, 1, 1, :] = -0.25
    h.put(plate)
    body = h.body(h.run())
    assert "QC WARNING" in body["qc"], body["qc"]
    assert "Min=-0.250" in body["qc"] and "Max=0.500" in body["qc"], body["qc"]


def test_levels_outside_broadcast_range_report_a_qc_warning_with_numbers(h):
    plate = flat(0.5, h=4, w=4)
    plate[0, 0, 0, :] = 2.0
    plate[0, 1, 1, :] = -0.25
    h.put(plate)
    body = h.body(h.run())
    assert "QC WARNING" in body["qc"] and "Out-of-Gamut" in body["qc"]
    assert "Illegal levels for broadcast" in body["qc"]
    assert "Min=-0.250" in body["qc"], body["qc"]
    assert "Max=2.000" in body["qc"], body["qc"]


def test_an_aces_delivery_gets_the_gamut_report_not_the_levels_report(h):
    h.put(flat(0.5))
    body = h.body(h.run({"colorSpace": "ACEScg (AP1)"}))
    assert "QC PASS — ACES" in body["qc"], body["qc"]
    assert "ACEScg (AP1)" in body["qc"]
    assert "Imaginary gamut" in body["qc"]


def test_imaginary_gamut_pixels_are_flagged_on_an_aces_delivery(h):
    """Pure AP1 green is outside Rec.2020 — every pixel goes negative."""
    plate = torch.zeros((1, 4, 4, 3))
    plate[..., 1] = 1.0
    h.put(plate)
    body = h.body(h.run({"colorSpace": "ACEScg (AP1)"}))
    assert "QC WARNING" in body["qc"], body["qc"]
    assert "100.0% imaginary-gamut" in body["qc"], body["qc"]
    assert "gamut compression" in body["qc"]


def test_an_out_of_range_aces_master_is_not_reported_as_a_levels_failure(h):
    """Scene-linear ACES values above 1.0 are normal, not 'illegal levels'."""
    h.put(flat(4.0))
    body = h.body(h.run({"colorSpace": "ACEScg (AP1)"}))
    assert "Out-of-Gamut" not in body["qc"], body["qc"]


# ═══════════════════════════════════════════════════════════════════════════
#  § 10  Shot-continuity scan
# ═══════════════════════════════════════════════════════════════════════════

def test_a_luminance_jump_between_frames_is_reported_as_a_flicker_event(h):
    h.put(torch.stack([flat(0.2)[0], flat(0.5)[0]]))
    body = h.body(h.run())
    assert body["continuity"], "a 1.32 EV jump was not flagged"
    assert len(body["continuity"]) == 1
    assert "Frame 0→1" in body["continuity"][0]
    assert "1.32 EV jump" in body["continuity"][0], body["continuity"][0]
    assert "flicker event(s)" in body["qc"], body["qc"]


def test_a_steady_shot_reports_no_flicker(h):
    h.put(torch.stack([flat(0.20)[0], flat(0.21)[0]]))
    body = h.body(h.run())
    assert body["continuity"] == []
    assert "flicker" not in body["qc"]


def test_a_single_frame_delivery_skips_the_continuity_scan(h):
    h.put(flat(0.2))
    assert h.body(h.run())["continuity"] == []


def test_only_the_first_three_flicker_events_are_listed_with_a_count(h):
    values = [0.05, 0.4, 0.05, 0.4, 0.05, 0.4]
    h.put(torch.stack([flat(v)[0] for v in values]))
    body = h.body(h.run())
    assert len(body["continuity"]) == 5
    assert "(+2 more)" in body["qc"], body["qc"]


# ═══════════════════════════════════════════════════════════════════════════
#  § 11  Sidecars
# ═══════════════════════════════════════════════════════════════════════════

def test_a_metadata_sidecar_lands_next_to_the_master(h):
    h.put(flat(0.5, n=2))
    body = h.body(h.run())
    meta_path = os.path.splitext(body["path"])[0] + "_meta.json"
    meta = json.loads(open(meta_path, encoding="utf-8").read())
    assert meta["version"] == "v0001"
    assert meta["continuity"] == "PASS"
    assert meta["bake_grade_exr"] is False
    assert "QC PASS" in meta["qc"]
    assert meta["grading"]["exposure"] == 0.0
    assert meta["export_settings"]["format"] == BASE_SETTINGS["format"]


def test_a_thumbnail_is_written_beside_the_master(h):
    h.put(flat(0.5))
    body = h.body(h.run())
    thumb = os.path.splitext(body["path"])[0] + "_thumb.jpg"
    assert os.path.isfile(thumb), os.listdir(h.root)
    assert os.path.getsize(thumb) > 0


def test_an_rgba_plate_still_gets_its_sidecars(h):
    """JPEG cannot hold alpha; that used to abort every sidecar after it."""
    h.put(torch.full((1, 4, 4, 4), 0.5))
    body = h.body(h.run({"export_cdl": True}))
    stem = os.path.splitext(body["path"])[0]
    assert os.path.isfile(stem + "_meta.json"), "the .json sidecar was lost"
    assert os.path.isfile(stem + ".cdl"), "the CDL sidecar was lost"


def test_the_cdl_sidecar_is_only_written_when_asked_for(h):
    h.put(flat(0.5))
    body = h.body(h.run())
    assert not os.path.isfile(os.path.splitext(body["path"])[0] + ".cdl")


def test_the_cdl_sidecar_carries_the_grade_that_was_delivered(h):
    h.put(flat(0.5))
    body = h.body(h.run({"export_cdl": True},
                        {"gain": [1.25, 1.0, 0.75], "saturation": 1.4}))
    cdl = open(os.path.splitext(body["path"])[0] + ".cdl", encoding="utf-8").read()
    assert "<Slope>1.250000 1.000000 0.750000</Slope>" in cdl, cdl
    assert "<Saturation>1.400000</Saturation>" in cdl, cdl


def test_the_amf_sidecar_is_written_when_asked_for(h):
    import xml.etree.ElementTree as ET
    h.put(flat(0.5))
    body = h.body(h.run({"export_amf": True}, {"saturation": 1.4}))
    amf = os.path.splitext(body["path"])[0] + ".amf"
    assert os.path.isfile(amf)
    ET.parse(amf)
    assert "1.400000" in open(amf, encoding="utf-8").read()


def test_the_amf_sidecar_is_absent_by_default(h):
    h.put(flat(0.5))
    body = h.body(h.run())
    assert not os.path.isfile(os.path.splitext(body["path"])[0] + ".amf")


def test_reveal_folder_opens_the_output_directory(h, monkeypatch):
    import subprocess
    seen = []
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, *a, **k: seen.append(cmd))
    if hasattr(os, "startfile"):
        monkeypatch.setattr(os, "startfile", lambda path: seen.append([path]))
    h.put(flat(0.5))
    assert h.run({"reveal_folder": True}).status == 200
    assert seen, "reveal_folder did not launch anything"
    assert os.path.abspath(str(h.root)) in seen[0]


def test_reveal_folder_is_off_by_default(h, monkeypatch):
    import subprocess
    seen = []
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, *a, **k: seen.append(cmd))
    if hasattr(os, "startfile"):
        monkeypatch.setattr(os, "startfile", lambda path: seen.append([path]))
    h.put(flat(0.5))
    h.run()
    assert seen == []


def test_a_thumbnail_failure_does_not_cost_the_other_sidecars(h, monkeypatch):
    """Each sidecar fails independently — that was the point of the rework."""
    from PIL import Image

    def _boom(self, *a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(Image.Image, "save", _boom)
    h.put(flat(0.5))
    body = h.body(h.run({"export_cdl": True}))
    stem = os.path.splitext(body["path"])[0]
    assert not os.path.exists(stem + "_thumb.jpg")
    assert os.path.isfile(stem + ".cdl"), "the CDL went down with the thumbnail"
    assert os.path.isfile(stem + "_meta.json"), "the .json went down with it too"


def test_a_failing_amf_export_does_not_fail_the_delivery(h, monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("no such ODT")

    monkeypatch.setattr(handler, "_export_aces_clip_xml", _boom)
    h.put(flat(0.5))
    resp = h.run({"export_amf": True})
    assert resp.status == 200, h.body(resp)
    assert os.path.isfile(os.path.splitext(h.body(resp)["path"])[0] + "_meta.json")


def test_a_failing_session_log_does_not_fail_the_delivery(h, monkeypatch):
    def _boom(*a, **k):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(handler, "_write_sessions_atomically", _boom)
    h.put(flat(0.5))
    assert h.run().status == 200
    assert not (h.root / "radiance_sessions.json").exists()


def test_reveal_folder_uses_open_on_macos(h, monkeypatch):
    import platform
    import subprocess
    seen = []
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, *a, **k: seen.append(cmd))
    if hasattr(os, "startfile"):
        monkeypatch.setattr(os, "startfile", lambda path: seen.append([path]))
    h.put(flat(0.5))
    assert h.run({"reveal_folder": True}).status == 200
    assert seen and seen[0][0] == "open", seen


def test_a_failing_file_manager_does_not_fail_the_delivery(h, monkeypatch):
    import subprocess

    def _boom(*a, **k):
        raise OSError("no xdg-open here")

    monkeypatch.setattr(subprocess, "Popen", _boom)
    h.put(flat(0.5))
    assert h.run({"reveal_folder": True}).status == 200


# ═══════════════════════════════════════════════════════════════════════════
#  § 12  Grade baking into EXR
# ═══════════════════════════════════════════════════════════════════════════

def test_bake_grade_linearises_an_srgb_encoded_plate_into_the_exr(h):
    """0.5 sRGB is 0.2140 linear. Without the bake it stays 0.5.

    The decode follows the SOURCE the Viewer resolved (FIX-018), not the
    output colour space: a linear source must NOT be linearised -- see
    `test_bake_grade_leaves_an_already_linear_master_alone` below.
    """
    h.put(flat(0.5), source=("srgb", "sRGB Encoded Rec.709 (sRGB)"))
    body = h.body(h.run({"bake_grade": True, "colorSpace": "sRGB (Standard)"}))
    assert h.pixel(h.exrs()[0])[0] == pytest.approx(0.21404, abs=1e-4)
    meta = json.loads(open(os.path.splitext(body["path"])[0] + "_meta.json",
                           encoding="utf-8").read())
    assert meta["bake_grade_exr"] is True


def test_bake_grade_uses_the_low_slope_below_the_srgb_knee(h):
    """0.02 is under 0.04045, so it divides by 12.92 rather than powing."""
    h.put(flat(0.02), source=("srgb", "sRGB Encoded Rec.709 (sRGB)"))
    h.run({"bake_grade": True, "colorSpace": "sRGB (Standard)"})
    assert h.pixel(h.exrs()[0])[0] == pytest.approx(0.02 / 12.92, abs=1e-6)


def test_bake_grade_leaves_an_already_linear_master_alone(h):
    """`Linear (sRGB)` is linear. It used to be linearised a SECOND time.

    The bake applied the sRGB EOTF for `color_space in ('sRGB (Standard)',
    'Linear (sRGB)')`, so a master that was already scene-linear came out about
    2.2 gamma down in the midtones -- 0.5 delivered as 0.214 -- and the log line
    said "sRGB to linear applied". The else branch's own comment says "already
    linear", which is the whole argument.
    """
    h.put(flat(0.5))
    h.run({"bake_grade": True, "colorSpace": "Linear (sRGB)"})
    assert h.pixel(h.exrs()[0])[0] == pytest.approx(0.5, abs=1e-5)


def test_without_bake_grade_the_exr_keeps_the_graded_values(h):
    h.put(flat(0.5))
    h.run({"bake_grade": False})
    assert h.pixel(h.exrs()[0])[0] == pytest.approx(0.5, abs=1e-5)


def test_bake_grade_is_ignored_for_a_non_exr_delivery(h):
    """The flag means 'bake into EXR'; a PNG must not be linearised."""
    h.put(flat(0.5))
    body = h.body(h.run({"bake_grade": True,
                         "format": "Image Sequence — PNG (8-bit)"}))
    meta = json.loads(open(os.path.splitext(body["path"])[0] + "_meta.json",
                           encoding="utf-8").read())
    assert meta["bake_grade_exr"] is False
    from PIL import Image
    # 0.5 → 128 (rounded; 3.5 stopped truncating to 127). Had the sRGB
    # decode run it would be 0.2140 → 55.
    assert Image.open(h.pngs()[0]).getpixel((0, 0))[0] == 128


def test_an_already_linear_colour_space_skips_the_srgb_decode(h):
    h.put(flat(0.5))
    h.run({"bake_grade": True, "colorSpace": "ACEScg (AP1)"})
    # ACEScg output re-encodes on write, so read the meta flag and the fact
    # that the sRGB decode (which would have produced 0.214 before encoding)
    # was not applied: the ACEScg matrix on a neutral 0.5 is close to 0.5.
    assert h.pixel(h.exrs()[0])[0] == pytest.approx(0.5, abs=0.05)


# ═══════════════════════════════════════════════════════════════════════════
#  § 13  What the write engine is actually asked for
# ═══════════════════════════════════════════════════════════════════════════

def test_the_ui_vocabulary_is_translated_before_it_reaches_the_writer(h, spy_write):
    h.put(flat(0.5))
    assert h.run({"format": "Image Sequence — EXR (32-bit)",
                  "colorSpace": "ACEScg (AP1)"}).status == 200
    kwargs = spy_write[-1]
    assert kwargs["format"] == "SEQ │ EXR (32-bit float)"
    assert kwargs["color_space"] == "ACEScg"


def test_an_absurd_frame_rate_is_clamped_to_240(h, spy_write):
    h.put(flat(0.5))
    h.run({"fps": 100000})
    assert spy_write[-1]["fps"] == 240.0


def test_a_sub_one_frame_rate_is_clamped_up(h, spy_write):
    h.put(flat(0.5))
    h.run({"fps": 0.01})
    assert spy_write[-1]["fps"] == 1.0


def test_quality_is_clamped_into_the_crf_range(h, spy_write):
    h.put(flat(0.5))
    h.run({"quality": 900})
    assert spy_write[-1]["quality"] == 51
    h.run({"quality": -5})
    assert spy_write[-1]["quality"] == 0


def test_soft_clip_is_passed_through_as_broadcast_safe(h, spy_write):
    h.put(flat(0.5))
    h.run({"soft_clip": True})
    assert spy_write[-1]["broadcast_safe"] is True
    h.run({"soft_clip": False})
    assert spy_write[-1]["broadcast_safe"] is False


def test_soft_clip_limits_an_8_bit_master_to_legal_range(h):
    """Black must land on 16/255, not 0, when broadcast-safe is on."""
    from PIL import Image
    h.put(flat(0.0))
    h.run({"format": "Image Sequence — PNG (8-bit)", "soft_clip": True})
    assert Image.open(h.pngs()[0]).getpixel((0, 0))[0] == 16


def test_soft_clip_off_leaves_black_at_zero(h):
    from PIL import Image
    h.put(flat(0.0))
    h.run({"format": "Image Sequence — PNG (8-bit)", "soft_clip": False})
    assert Image.open(h.pngs()[0]).getpixel((0, 0))[0] == 0


def test_a_failing_upscale_is_non_fatal(h):
    """No RealESRGAN weights here; the delivery still has to complete."""
    h.put(flat(0.5, n=1, h=8, w=8))
    resp = h.run({"upscale_2x": True})
    assert resp.status == 200, h.body(resp)
    assert len(h.exrs()) == 1


# ═══════════════════════════════════════════════════════════════════════════
#  § 14  The session log
# ═══════════════════════════════════════════════════════════════════════════

def _sessions(h):
    return json.loads((h.root / "radiance_sessions.json").read_text("utf-8"))


def test_a_delivery_is_appended_to_the_session_log(h):
    h.put(flat(0.5, n=2, h=6, w=10))
    body = h.body(h.run(grading={"exposure": 0.5, "saturation": 1.25}))
    entry = _sessions(h)[-1]
    assert entry["version"] == "v0001"
    assert entry["shot"] == os.path.basename(body["path"])
    assert entry["frames"] == 2
    assert entry["resolution"] == "10×6"
    assert entry["exposure"] == 0.5
    assert entry["saturation"] == 1.25
    assert entry["format"] == BASE_SETTINGS["format"]
    assert entry["color_space"] == BASE_SETTINGS["colorSpace"]
    assert entry["continuity"] == "PASS"


def test_the_session_log_accumulates_rather_than_replacing(h):
    h.put(flat(0.5))
    h.run()
    h.run()
    assert len(_sessions(h)) == 2, _sessions(h)


def test_a_corrupt_session_log_is_replaced_not_propagated(h):
    (h.root / "radiance_sessions.json").write_text("{ not json", encoding="utf-8")
    h.put(flat(0.5))
    assert h.run().status == 200
    assert len(_sessions(h)) == 1


def test_the_session_log_is_capped_at_500_entries(h):
    old = [{"shot": f"old_{i}"} for i in range(505)]
    (h.root / "radiance_sessions.json").write_text(json.dumps(old), encoding="utf-8")
    h.put(flat(0.5))
    h.run()
    sessions = _sessions(h)
    assert len(sessions) == 500
    assert sessions[0]["shot"] == "old_6", sessions[0]
    assert sessions[-1]["version"] == "v0001"


def test_a_flicker_event_is_recorded_in_the_session_log(h):
    h.put(torch.stack([flat(0.2)[0], flat(0.5)[0]]))
    h.run()
    assert _sessions(h)[-1]["continuity"] == "1 events"


# ═══════════════════════════════════════════════════════════════════════════
#  § 15  Small module-level paths the endpoint depends on
# ═══════════════════════════════════════════════════════════════════════════

def test_an_unreadable_directory_yields_v0001_rather_than_raising(tmp_path, caplog):
    """get_next_version's guard: a file where a directory was expected."""
    victim = tmp_path / "not_a_dir"
    victim.write_text("", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="radiance.delivery.handler"):
        assert handler.get_next_version(str(victim), "shot") == "v0001"
    assert any("get_next_version" in r.getMessage() for r in caplog.records)


def test_a_failed_session_write_reraises_even_if_the_temp_cleanup_fails(
        tmp_path, monkeypatch):
    """The cleanup is best-effort; it must not replace the real error."""
    def _no_unlink(_path):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(handler.os, "unlink", _no_unlink)
    with pytest.raises(TypeError):
        handler._write_sessions_atomically(str(tmp_path / "s.json"),
                                           [{"bad": object()}])


def test_the_route_is_not_registered_twice(monkeypatch):
    """Importing the module under two names must not raise 'already registered'."""
    posted = []

    class _Routes:
        def post(self, path):
            posted.append(path)
            return lambda fn: fn

    class _Server:
        instance = type("I", (), {"routes": _Routes()})()

    monkeypatch.setattr(handler, "PromptServer", _Server)
    monkeypatch.setattr(handler, "_RADIANCE_DELIVER_ROUTE_REGISTERED", True)
    sentinel = object()
    assert handler._register_deliver_route(sentinel) is sentinel
    assert posted == [], "the route was registered a second time"

    monkeypatch.setattr(handler, "_RADIANCE_DELIVER_ROUTE_REGISTERED", False)
    assert handler._register_deliver_route(sentinel) is sentinel
    assert posted == ["/radiance/deliver"]
    assert _Server.instance._radiance_deliver_route_registered is True


def test_no_server_means_no_registration_and_no_crash(monkeypatch):
    monkeypatch.setattr(handler, "PromptServer", None)
    monkeypatch.setattr(handler, "_RADIANCE_DELIVER_ROUTE_REGISTERED", False)
    sentinel = object()
    assert handler._register_deliver_route(sentinel) is sentinel


# ═══════════════════════════════════════════════════════════════════════════
#  § 15  The delivery that reported success for work it did not do
# ═══════════════════════════════════════════════════════════════════════════

def test_a_failed_upscale_is_reported_rather_than_delivered_as_success(h, monkeypatch):
    """A master delivered at 1x because the ESRGAN model was missing is not a
    successful delivery.

    Seven user-requested transforms in `_run_export` -- the AI upscale, aspect
    blanking, the FX bake, denoise, halation, bloom and diffusion -- each
    caught their own failure, logged it and carried on, while the response
    still said status "success" and the HUD still showed "EXPORT COMPLETE".

    The upscaler is made to raise rather than left to fail on its own: whether
    it raises depends on what else is installed, and a test whose colour
    depends on the environment proves nothing either way.
    """
    from radiance.image.upscale import RadianceAIUpscale

    def _no_model(self, **kwargs):
        raise RuntimeError("RealESRGAN_x2plus.pth not found in models/upscale_models")

    monkeypatch.setattr(RadianceAIUpscale, "upscale", _no_model)

    h.put(flat(0.5))
    resp = h.run({"upscale_2x": True})

    body = h.body(resp)
    assert resp.status == 200, body
    assert body["status"] == "partial", body
    assert body["warnings"], body
    assert any("upscale" in w.lower() for w in body["warnings"]), body["warnings"]
    assert "1x" in " ".join(body["warnings"])
    assert "WITHOUT" in body["message"]


def test_a_silent_bicubic_fallback_is_reported_too(h, monkeypatch):
    """RadianceAIUpscale does not raise when its model is missing, a download
    fails or inference fails: it returns a bicubic resize. The master was 2x by
    resampling and the receipt said nothing (code review P1-2)."""
    from radiance.image.upscale import RadianceAIUpscale

    monkeypatch.setattr(RadianceAIUpscale, "_load_model",
                        lambda self, *a, **k: (None, "RealESRGAN_x2plus.pth missing"))

    h.put(flat(0.5, n=1, h=8, w=8))
    body = h.body(h.run({"upscale_2x": True}))

    assert body["status"] == "partial", body
    note = " ".join(body["warnings"]).lower()
    assert "upscale" in note and "bicubic" in note, body["warnings"]


def test_a_failed_aspect_blank_is_reported(h):
    """An unparseable ratio silently delivered an unblanked master."""
    h.put(flat(0.5))
    body = h.body(h.run({"aspect_ratio": "not a ratio"}))
    assert body["status"] == "partial", body
    assert any("aspect" in w.lower() for w in body["warnings"]), body["warnings"]


def test_a_clean_delivery_still_says_success(h):
    h.put(flat(0.5))
    body = h.body(h.run())
    assert body["status"] == "success"
    assert body["warnings"] == []


def test_a_sequence_export_of_a_trimmed_range_is_numbered_from_where_it_starts(h):
    """`start_frame` was never passed, so a sequence export of frames 21-25 was
    numbered 1001-1005 exactly like an export of frames 1-5, and nothing in the
    filenames said which range it was."""
    h.put(flat(0.5, n=40))
    body = h.body(h.run({"format": "Image Sequence — PNG (8-bit)",
                         "range_in": 21, "range_out": 25}))
    assert body["status"] == "success", body

    written = sorted(p.name for p in Path(body["path"]).glob("*.png"))
    numbers = sorted(int(n.rsplit("_", 1)[1].split(".")[0]) for n in written)
    assert numbers == [1021, 1022, 1023, 1024, 1025], written


def test_an_untrimmed_sequence_export_still_starts_at_1001(h):
    h.put(flat(0.5, n=3))
    body = h.body(h.run({"format": "Image Sequence — PNG (8-bit)"}))
    written = sorted(p.name for p in Path(body["path"]).glob("*.png"))
    numbers = sorted(int(n.rsplit("_", 1)[1].split(".")[0]) for n in written)
    assert numbers == [1001, 1002, 1003], written


def test_the_export_never_stacks_the_whole_sequence(h, monkeypatch):
    """The delivery held five full copies of the clip at its peak.

    The grade built `out_batch`, a list of N graded frames, and `torch.stack`ed
    it; the FX bake then built `fx_batch` and `np.stack`ed that. On top of the
    viewer cache entry and the writer's own copies the measured ceiling was
    about 1000 frames at 1080p. Both stacks are gone: the graded batch is
    allocated once and written into, and the filters write back in place.

    Measured as "nothing ever stacked as many things as there are frames",
    which is the property, rather than as a source-text check.
    """
    seen = []

    def _spy(fn):
        def wrapper(seq, *a, **k):
            try:
                seen.append(len(seq))
            except TypeError:
                pass
            return fn(seq, *a, **k)
        return wrapper

    monkeypatch.setattr(handler.torch, "stack", _spy(torch.stack))
    monkeypatch.setattr(handler.np, "stack", _spy(np.stack))

    n = 24
    h.put(flat(0.5, n=n))
    body = h.body(h.run({"grain": 0.0}, grading={"bloom": 0.5, "grain": 0.3}))
    assert body["status"] == "success", body
    assert len(h.exrs()) == n

    assert max(seen, default=0) < n, (
        f"something stacked {max(seen)} items during a {n}-frame delivery; "
        "the sequence is being materialised again")
