"""`delivery/handler.py` — the path that decides whether a master lands on disk.

452 statements at 10%, and the reason that mattered is on the record: the
handler used to call the writer with `filename_prefix`, `output_format` and
`output_color_space`, none of which are parameters, and omit the required
`format`. Every delivery raised TypeError, the handler swallowed it, and the
endpoint returned HTTP 200 with status "error".

The write engine moved down a floor to `radiance/io/writer.py` specifically so
this could be exercised without ComfyUI. Nothing had used that yet. This does.

The centre of the file is the two lookup tables. They translate what the
Viewer's delivery panel offers into what the writer accepts, which makes them a
contract between two modules that are edited at different times for different
reasons — exactly where names drift apart in silence.
"""
import json
import os
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import torch

from radiance.delivery.handler import (
    _UI_TO_WRITE_COLORSPACE,
    _UI_TO_WRITE_FORMAT,
    _export_aces_clip_xml,
    _resolve_write_colorspace,
    _resolve_write_format,
    _write_sessions_atomically,
    get_next_version,
)
from radiance.io.writer import OUTPUT_COLOR_SPACES, WRITE_FORMATS, write_frames


# ── the contract between the panel and the writer ───────────────────────────

def test_every_offered_format_is_one_the_writer_implements():
    """Against the writer's own table, not a list retyped here."""
    unknown = sorted(v for v in _UI_TO_WRITE_FORMAT.values() if v not in WRITE_FORMATS)
    assert not unknown, (
        f"the delivery panel offers {unknown}, which write_frames does not accept"
    )


def test_every_offered_colour_space_is_one_the_writer_implements():
    unknown = sorted(v for v in _UI_TO_WRITE_COLORSPACE.values()
                     if v not in OUTPUT_COLOR_SPACES)
    assert not unknown, (
        f"the delivery panel offers {unknown}, which write_frames does not accept"
    )


def test_an_unsupported_format_is_refused_by_name():
    """Refusing loudly is the point: the shipped bug was a silent TypeError."""
    with pytest.raises(ValueError) as exc:
        _resolve_write_format("Video — AVI (Cinepak)")
    assert "Cinepak" in str(exc.value)
    assert "MP4" in str(exc.value), "the error should list what is supported"


def test_an_unsupported_colour_space_is_refused_by_name():
    with pytest.raises(ValueError) as exc:
        _resolve_write_colorspace("Rec.2020 PQ")
    assert "Rec.2020 PQ" in str(exc.value)


@pytest.mark.parametrize("ui", list(_UI_TO_WRITE_FORMAT))
def test_each_offered_format_resolves(ui):
    assert _resolve_write_format(ui) in WRITE_FORMATS


# ── the thing the extraction was for ────────────────────────────────────────

def test_the_handler_can_write_a_master_with_no_comfyui_present(tmp_path):
    """End to end through the same call the endpoint makes.

    The AST test in test_writer_layering.py checks the handler *names* real
    parameters. This runs them: resolve through the handler's own tables, hand
    write_frames a tensor, and require a file. If the signature drifts again,
    this fails where a graph would.
    """
    frames = torch.linspace(-0.2, 8.0, 2 * 8 * 8 * 3).reshape(2, 8, 8, 3).float()
    path, count = write_frames(
        image=frames,
        output_path=str(tmp_path),
        format=_resolve_write_format("Image Sequence — EXR (32-bit)"),
        filename="master",
        color_space=_resolve_write_colorspace("Linear (sRGB)"),
        fps=24.0,
        quality=18,
        broadcast_safe=False,
    )
    assert count == 2, f"asked for two frames, wrote {count}"
    written = sorted(tmp_path.rglob("*.exr"))
    assert len(written) == 2, [p.name for p in written]
    assert all(p.stat().st_size > 0 for p in written)
    assert path


def test_a_linear_delivery_does_not_clamp_the_master(tmp_path):
    """`Linear (sRGB)` resolves to pass-through, so scene-linear values have to
    survive the delivery path the same way they survive the node."""
    frames = torch.full((1, 4, 4, 3), 6.5)
    write_frames(
        image=frames, output_path=str(tmp_path),
        format=_resolve_write_format("Image Sequence — EXR (32-bit)"),
        filename="hdr",
        color_space=_resolve_write_colorspace("Linear (sRGB)"),
        fps=24.0, quality=18, broadcast_safe=False,
    )
    exr = sorted(tmp_path.rglob("*.exr"))[0]
    oiio = pytest.importorskip("OpenImageIO")
    buf = oiio.ImageBuf(str(exr))
    px = buf.getpixel(0, 0)
    assert px[0] == pytest.approx(6.5, rel=1e-4), (
        f"a 6.5 delivery came back as {px[0]} — the master was clamped"
    )


# ── versioning ──────────────────────────────────────────────────────────────

def test_the_first_version_is_v01_when_nothing_is_there(tmp_path):
    assert get_next_version(str(tmp_path / "nope"), "shot") == "v01"
    assert get_next_version(str(tmp_path), "shot") == "v01"


def test_the_version_follows_the_highest_on_disk(tmp_path):
    for name in ("shot_v01.mov", "shot_v02.mov", "shot_v07.mov"):
        (tmp_path / name).touch()
    assert get_next_version(str(tmp_path), "shot") == "v08", (
        "a gap in the numbering must not hand back a version that already exists"
    )


def test_versions_stay_two_digits_until_they_cannot(tmp_path):
    (tmp_path / "shot_v09.mov").touch()
    assert get_next_version(str(tmp_path), "shot") == "v10"
    (tmp_path / "shot_v99.mov").touch()
    assert get_next_version(str(tmp_path), "shot") == "v100"


def test_another_shots_versions_do_not_count(tmp_path):
    (tmp_path / "other_v42.mov").touch()
    assert get_next_version(str(tmp_path), "shot") == "v01"


def test_a_regex_special_character_in_the_name_is_not_a_pattern(tmp_path):
    """`shot.a` must not match `shotXa`. The base goes through re.escape."""
    (tmp_path / "shotXa_v03.mov").touch()
    assert get_next_version(str(tmp_path), "shot.a") == "v01"


# ── the session log ─────────────────────────────────────────────────────────

def test_the_session_log_is_written_and_reads_back(tmp_path):
    target = tmp_path / "logs" / "sessions.json"
    sessions = [{"id": "a", "frames": 12}, {"id": "b", "frames": 3}]
    _write_sessions_atomically(str(target), sessions)
    assert json.loads(target.read_text(encoding="utf-8")) == sessions


def test_the_session_log_leaves_no_temp_file_behind(tmp_path):
    target = tmp_path / "sessions.json"
    _write_sessions_atomically(str(target), [{"id": "a"}])
    leftovers = [p.name for p in tmp_path.iterdir() if p.name != "sessions.json"]
    assert not leftovers, f"temp files survived the write: {leftovers}"


def test_a_failed_session_write_does_not_destroy_the_previous_log(tmp_path):
    """The reason it is atomic. A serialisation failure half way through must
    leave the old log intact rather than a truncated one."""
    target = tmp_path / "sessions.json"
    _write_sessions_atomically(str(target), [{"id": "good"}])
    with pytest.raises(TypeError):
        _write_sessions_atomically(str(target), [{"id": object()}])
    assert json.loads(target.read_text(encoding="utf-8")) == [{"id": "good"}]
    leftovers = [p.name for p in tmp_path.iterdir() if p.name != "sessions.json"]
    assert not leftovers, f"the failed write left {leftovers}"


# ── the ACES sidecar ────────────────────────────────────────────────────────

def _amf(tmp_path, grading, colour="ACEScg (AP1)"):
    media = tmp_path / "shot_v03.exr"
    media.touch()
    _export_aces_clip_xml(str(media), grading, colour, "v03")
    return tmp_path / "shot_v03.amf"


def test_the_amf_lands_next_to_the_media_and_parses(tmp_path):
    amf = _amf(tmp_path, {"gain": [1.1, 1.0, 0.9], "offset": [0.0, 0.0, 0.0],
                          "gamma": [1.0, 1.0, 1.0], "saturation": 1.0})
    assert amf.is_file()
    ET.parse(amf)          # raises if the XML is malformed


def test_the_cdl_values_are_the_grade_that_was_passed(tmp_path):
    """A sidecar carrying someone else's numbers is worse than no sidecar."""
    amf = _amf(tmp_path, {"gain": [1.25, 1.0, 0.75], "offset": [0.01, 0.0, -0.02],
                          "gamma": [0.9, 1.0, 1.1], "saturation": 1.4})
    text = amf.read_text(encoding="utf-8")
    assert "1.250000 1.000000 0.750000" in text, "Slope is not the gain"
    assert "0.010000 0.000000 -0.020000" in text, "Offset is not the offset"
    assert "0.900000 1.000000 1.100000" in text, "Power is not the gamma"
    assert "1.400000" in text, "Saturation missing"


def test_a_scalar_grade_value_is_broadcast_to_three(tmp_path):
    """The Viewer can send a single number where the CDL wants a triple."""
    amf = _amf(tmp_path, {"gain": 1.5, "offset": 0.0, "gamma": 1.0, "saturation": 1.0})
    assert "1.500000 1.500000 1.500000" in amf.read_text(encoding="utf-8")


def test_an_empty_grade_writes_an_identity_cdl(tmp_path):
    amf = _amf(tmp_path, {})
    text = amf.read_text(encoding="utf-8")
    assert "1.000000 1.000000 1.000000" in text
    assert "0.000000 0.000000 0.000000" in text


@pytest.mark.parametrize("colour,expected", [
    ("ACEScg (AP1)", "ACEScg_to_ACES"),
    ("sRGB (Standard)", "sRGB_100nits"),
    ("ACEScct", "ACEScct_to_ACES"),
])
def test_the_output_transform_follows_the_colour_space(tmp_path, colour, expected):
    out = tmp_path / colour.replace(" ", "_").replace("(", "").replace(")", "")
    out.mkdir()
    assert expected in _amf(out, {}, colour).read_text(encoding="utf-8")


def test_an_unknown_colour_space_falls_back_rather_than_writing_nothing(tmp_path):
    amf = _amf(tmp_path, {}, "Some Show LUT")
    assert "Rec709_100nits" in amf.read_text(encoding="utf-8")
