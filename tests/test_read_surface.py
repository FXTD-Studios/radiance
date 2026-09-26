"""The Read node should open what a comp application opens.

The reference point throughout is Nuke's Read: point it at a file and it works
out the format, the range, the layers and the windows for you, and it tells you
what it found. Radiance's Read opened nine image extensions, one EXR layer, and
told you nothing.

Everything below runs against real files written to disk. A test that mocks the
reader cannot see that `.tga` was refused by the *detector* while the reader
underneath handled it perfectly, which is what was actually happening.
"""
from __future__ import annotations

import collections
import contextlib
import importlib
import json
import os
import pathlib
import re
import sys
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")
PIL = pytest.importorskip("PIL.Image")

from radiance.core import formats as F  # noqa: E402

_ROOT = pathlib.Path(__file__).resolve().parent.parent


# ── fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def rgba8() -> np.ndarray:
    a = np.zeros((16, 24, 4), np.uint8)
    a[..., 0] = np.linspace(0, 255, 24, dtype=np.uint8)[None, :]
    a[..., 1] = 64
    a[..., 2] = 128
    a[..., 3] = np.linspace(0, 255, 24, dtype=np.uint8)[None, :]
    return a


@pytest.fixture
def exr_layers(tmp_path) -> pathlib.Path:
    """A multi-layer EXR: the normal output of a Nuke or Arnold render."""
    OpenEXR = pytest.importorskip("OpenEXR")
    if not hasattr(OpenEXR, "File"):
        pytest.skip("this OpenEXR build has no multi-part File API")
    h, w = 16, 24
    flat = lambda v: np.full((h, w), v, np.float32)  # noqa: E731
    channels = {
        "R": flat(0.1), "G": flat(0.2), "B": flat(0.3), "A": flat(0.5),
        "diffuse.R": flat(0.4), "diffuse.G": flat(0.45), "diffuse.B": flat(0.5),
        "specular.R": flat(0.6), "specular.G": flat(0.65), "specular.B": flat(0.7),
        "Z": flat(12.5),
    }
    dst = tmp_path / "sh010_render.exr"
    with OpenEXR.File({"type": OpenEXR.scanlineimage}, channels) as f:
        f.write(str(dst))
    return dst


@pytest.fixture
def exr_overscan(tmp_path) -> pathlib.Path:
    """Data window larger than the display window — what Nuke writes by default."""
    OpenEXR = pytest.importorskip("OpenEXR")
    if not hasattr(OpenEXR, "File"):
        pytest.skip("this OpenEXR build has no multi-part File API")
    px = np.zeros((30, 40, 3), np.float32)
    px[6:24, 8:32] = 1.0          # exactly the display-window region
    header = {
        "dataWindow": (np.array([-8, -6], np.int32), np.array([31, 23], np.int32)),
        "displayWindow": (np.array([0, 0], np.int32), np.array([23, 17], np.int32)),
        "type": OpenEXR.scanlineimage,
    }
    dst = tmp_path / "overscan.exr"
    with OpenEXR.File(header, {"RGB": px}) as f:
        f.write(str(dst))
    return dst


@pytest.fixture
def exr_depth_only(tmp_path) -> pathlib.Path:
    OpenEXR = pytest.importorskip("OpenEXR")
    if not hasattr(OpenEXR, "File"):
        pytest.skip("this OpenEXR build has no multi-part File API")
    dst = tmp_path / "sh010_Z.exr"
    with OpenEXR.File({"type": OpenEXR.scanlineimage},
                      {"Z": np.full((16, 24), 3.25, np.float32)}) as f:
        f.write(str(dst))
    return dst


@pytest.fixture
def png_sequence(tmp_path, rgba8) -> pathlib.Path:
    d = tmp_path / "plates"
    d.mkdir()
    for frame in range(1001, 1009):
        PIL.fromarray(rgba8).save(d / f"sh010.{frame:04d}.png")
    return d / "sh010.1004.png"


# ── format coverage ────────────────────────────────────────────────────────

@pytest.mark.parametrize("ext", [".tga", ".sgi", ".ppm", ".pgm", ".jp2",
                                 ".pcx", ".ico", ".psd", ".dds", ".qoi"])
def test_formats_pillow_can_open_are_not_refused(ext, tmp_path, rgba8):
    """These all decoded correctly and were all rejected before they got there.

    ``_IMG_EXT`` was nine hand-typed strings, so the detector said "unknown" for
    everything else and the node refused the file -- while the reader underneath
    opened it without complaint.
    """
    from radiance.nodes.io.write import _path_kind

    if ext not in F.image_extensions():
        pytest.skip(f"this Pillow build has no {ext} reader")
    assert _path_kind(f"/plates/sh010{ext}") == "image"


def test_the_image_extension_table_is_not_a_hand_typed_list():
    """It must reflect what the install can do, not what someone remembered."""
    exts = F.image_extensions()
    assert len(exts) > 30, (
        f"only {len(exts)} image extensions; the table is not being built from "
        "the installed backends"
    )
    for required in (".png", ".jpg", ".tif", ".exr", ".hdr", ".dpx", ".tga"):
        assert required in exts, f"{required} is missing from the readable set"


def test_documents_are_not_treated_as_plates():
    """Pillow will open a PDF. Reading page one as a frame is a surprise."""
    for ext in (".pdf", ".ps", ".eps"):
        assert ext not in F.image_extensions()


def test_video_and_image_tables_do_not_overlap():
    from radiance.core.video import VIDEO_EXTENSIONS

    overlap = F.image_extensions() & set(VIDEO_EXTENSIONS)
    assert not overlap, f"{sorted(overlap)} is claimed by both tables"


def test_an_unsupported_format_says_what_would_open_it():
    """"unknown path type" tells a user nothing."""
    message = F.explain_unsupported("/plates/sh010.ari")
    assert "OpenImageIO" in message, message

    message = F.explain_unsupported("/plates/sh010.wat")
    assert ".wat" in message and "png" in message.lower()


def test_the_frontend_video_list_matches_python():
    """Three copies of this list is three chances to disagree, and they did:
    the JS listed .webp (a still) and omitted .mxf (a plate)."""
    from radiance.core.video import VIDEO_EXTENSIONS

    js = (_ROOT / "js" / "radiance_io.js").read_text(encoding="utf-8")
    match = re.search(r"const VIDEO_EXTENSIONS = \[(.*?)\];", js, re.S)
    assert match, "the frontend no longer declares VIDEO_EXTENSIONS"
    listed = set(re.findall(r'"(\.[a-z0-9]+)"', match.group(1)))
    assert listed == set(VIDEO_EXTENSIONS), (
        f"frontend-only: {sorted(listed - set(VIDEO_EXTENSIONS))}; "
        f"python-only: {sorted(set(VIDEO_EXTENSIONS) - listed)}"
    )


# ── sequence detection ─────────────────────────────────────────────────────

def test_one_frame_of_a_sequence_finds_the_whole_range(png_sequence):
    """Nuke opens shot.1001.exr and offers you the sequence. This did not."""
    info = F.detect_sequence(str(png_sequence))
    assert info is not None
    assert (info.first, info.last, info.count) == (1001, 1008, 8)
    assert info.padding == 4
    assert info.pattern.endswith("sh010.%04d.png")
    assert info.is_contiguous


def test_a_gap_in_the_sequence_is_reported(png_sequence):
    os.remove(png_sequence.parent / "sh010.1003.png")
    info = F.detect_sequence(str(png_sequence))
    assert info.missing == (1003,), info.missing
    assert not info.is_contiguous
    assert "1 missing" in info.summary()


def test_a_lone_numbered_file_is_a_still_not_a_sequence(tmp_path, rgba8):
    """`render.0001.png` on its own is a still. Calling it a one-frame sequence
    would be a worse answer than leaving it alone."""
    PIL.fromarray(rgba8).save(tmp_path / "render.0001.png")
    assert F.detect_sequence(str(tmp_path / "render.0001.png")) is None


def test_an_unnumbered_file_is_never_a_sequence(tmp_path, rgba8):
    PIL.fromarray(rgba8).save(tmp_path / "reference.png")
    assert F.detect_sequence(str(tmp_path / "reference.png")) is None


@pytest.mark.parametrize("pattern, expected", [
    ("/plates/f.%04d.exr", True), ("/plates/f.####.exr", True),
    ("/plates/f.*.exr", True), ("/plates/f.1001.exr", False),
    ("/plates/f.exr", False),
])
def test_explicit_patterns_are_recognised(pattern, expected):
    assert F.is_sequence_pattern(pattern) is expected


def test_an_explicit_pattern_reports_the_range_on_disk(png_sequence):
    pattern = str(png_sequence.parent / "sh010.%04d.png")
    info = F.describe_pattern(pattern)
    assert info is not None
    assert (info.first, info.last, info.count) == (1001, 1008, 8)


def test_reading_one_frame_reads_the_sequence(png_sequence):
    from radiance.nodes.io.write import RadianceRead

    image, mask, info = RadianceRead().read(browse="", path=str(png_sequence))
    assert image.shape[0] == 8, "picking one frame should offer the whole range"
    assert json.loads(info)["kind"] == "sequence"


def test_media_type_image_still_reads_exactly_one_frame(png_sequence):
    """The escape hatch, for when you really do want the single frame."""
    from radiance.nodes.io.write import RadianceRead

    image, _mask, info = RadianceRead().read(
        browse="", path=str(png_sequence), media_type="Image")
    assert image.shape[0] == 1
    assert json.loads(info)["kind"] == "image"


def test_a_sequence_keeps_its_alpha(png_sequence):
    """`img_t, _ = _read_image(p)` threw the matte away for every frame."""
    from radiance.nodes.io.write import RadianceRead

    _image, mask, info = RadianceRead().read(browse="", path=str(png_sequence))
    assert json.loads(info)["alpha"] is True
    assert float(mask.max()) > 0.9, "the sequence's alpha did not reach the mask"
    assert float(mask.min()) < 0.1


def test_the_sequence_start_frame_default_does_not_swallow_a_sequence(tmp_path, rgba8):
    """A sequence numbered from 1 read nothing, because start_frame is 1001."""
    d = tmp_path / "frames"
    d.mkdir()
    for frame in range(1, 6):
        PIL.fromarray(rgba8).save(d / f"take.{frame:04d}.png")

    from radiance.nodes.io.write import RadianceRead

    image, _mask, _info = RadianceRead().read(browse="", path=str(d / "take.0003.png"))
    assert image.shape[0] == 5


# ── EXR layers ─────────────────────────────────────────────────────────────

def test_a_multi_layer_exr_lists_its_layers(exr_layers):
    from radiance.core import exr as E

    info = E.probe(str(exr_layers))
    assert "diffuse" in info.layer_names
    assert "specular" in info.layer_names
    assert "Z" in info.layer_names
    assert info.beauty() is not None and info.beauty().is_beauty


def test_each_layer_reads_its_own_pixels(exr_layers):
    from radiance.core import exr as E

    expected = {"diffuse": 0.4, "specular": 0.6, "Z": 12.5}
    for name, value in expected.items():
        rgb, _alpha, _info, chosen = E.read_layer(str(exr_layers), name)
        assert chosen == name
        assert rgb[0, 0, 0] == pytest.approx(value, abs=1e-4), (
            f"layer {name} returned {rgb[0, 0, 0]}, not its own pixels"
        )


def test_the_beauty_is_the_default_layer(exr_layers):
    from radiance.core import exr as E

    rgb, alpha, _info, chosen = E.read_layer(str(exr_layers))
    assert chosen in ("RGBA", "RGB")
    assert rgb[0, 0, 0] == pytest.approx(0.1, abs=1e-4)
    assert alpha is not None and alpha[0, 0] == pytest.approx(0.5, abs=1e-4)


def test_a_depth_only_exr_reads_instead_of_raising(exr_depth_only):
    """It used to raise, and the message blamed the file:

        '...' has no standard R/G/B channels ... which RadianceRead does not
        support -- it only reads standard RGB(A) EXR.

    A Z pass is a normal render output, not a malformed file.
    """
    from radiance.nodes.io.write import RadianceRead

    image, _mask, info = RadianceRead().read(browse="", path=str(exr_depth_only))
    assert image.shape == (1, 16, 24, 3)
    assert float(image.max()) == pytest.approx(3.25, abs=1e-4)
    assert json.loads(info)["layer"] == "Z"


def test_an_unknown_layer_name_lists_the_real_ones(exr_layers):
    from radiance.core import exr as E

    with pytest.raises(E.EXRReadError) as excinfo:
        E.read_layer(str(exr_layers), "difuse")
    message = str(excinfo.value)
    assert "diffuse" in message and "specular" in message


def test_the_layer_widget_accepts_a_channel_suffix(exr_layers):
    """A user pasting "diffuse.R" out of a channel list means "diffuse"."""
    from radiance.core import exr as E

    _rgb, _alpha, _info, chosen = E.read_layer(str(exr_layers), "diffuse.R")
    assert chosen == "diffuse"


def test_layer_choices_puts_the_beauty_first(exr_layers):
    from radiance.core import exr as E

    choices = E.layer_choices(str(exr_layers))
    assert choices and choices[0] in ("RGBA", "RGB")


def test_reading_a_named_layer_through_the_node(exr_layers):
    from radiance.nodes.io.write import RadianceRead

    image, _mask, info = RadianceRead().read(
        browse="", path=str(exr_layers), layer="specular")
    assert float(image[0, 0, 0, 0]) == pytest.approx(0.6, abs=1e-4)
    assert json.loads(info)["layer"] == "specular"


# ── display window ─────────────────────────────────────────────────────────

def test_an_overscan_exr_is_conformed_to_its_display_window(exr_overscan):
    """Documented as a known limitation through 3.1.x: the reader used the data
    window and returned an offset frame at the wrong resolution, silently."""
    from radiance.nodes.io.write import RadianceRead

    image, _mask, info = RadianceRead().read(browse="", path=str(exr_overscan))
    assert image.shape == (1, 18, 24, 3), "not conformed to the display window"
    assert json.loads(info)["overscan"] is True
    # The white region was authored to be exactly the display window.
    assert float(image.min()) == pytest.approx(1.0, abs=1e-5)


def test_raw_keeps_the_overscan(exr_overscan):
    from radiance.nodes.io.write import RadianceRead

    image, _mask, _info = RadianceRead().read(
        browse="", path=str(exr_overscan), raw=True)
    assert image.shape == (1, 30, 40, 3), "raw should hand back the data window"


def test_the_info_output_reports_both_windows(exr_overscan):
    from radiance.nodes.io.write import RadianceRead

    _image, _mask, info = RadianceRead().read(browse="", path=str(exr_overscan))
    data = json.loads(info)
    assert data["data_window"] == [40, 30]
    assert data["display_window"] == [24, 18]


# ── error policy ───────────────────────────────────────────────────────────

def test_a_missing_file_raises_by_default(tmp_path):
    from radiance.nodes.io.write import RadianceRead

    with pytest.raises(Exception) as excinfo:
        RadianceRead().read(browse="", path=str(tmp_path / "absent.exr"))
    assert "absent.exr" in str(excinfo.value)


def test_black_frame_is_available_but_says_so(tmp_path, caplog):
    """The 3.1.x behaviour, kept as an explicit choice rather than a silent one."""
    import logging

    from radiance.nodes.io.write import RadianceRead

    with caplog.at_level(logging.WARNING):
        image, _mask, info = RadianceRead().read(
            browse="", path=str(tmp_path / "absent.exr"), on_error="Black frame")
    assert float(image.abs().max()) == 0.0
    assert json.loads(info)["kind"] == "error"
    assert any("Black frame" in r.getMessage() for r in caplog.records), (
        "substituting black must be announced, or it is the old silent bug"
    )


def test_an_unreadable_format_names_the_fix(tmp_path):
    from radiance.nodes.io.write import RadianceRead

    junk = tmp_path / "plate.wat"
    junk.write_bytes(b"nope")
    with pytest.raises(Exception) as excinfo:
        RadianceRead().read(browse="", path=str(junk))
    assert ".wat" in str(excinfo.value)


# ── premultiplied alpha ────────────────────────────────────────────────────

def test_premultiplied_divides_rgb_back_out(tmp_path):
    from radiance.nodes.io.write import RadianceRead

    OpenEXR = pytest.importorskip("OpenEXR")
    if not hasattr(OpenEXR, "File"):
        pytest.skip("needs the OpenEXR File API")
    h, w = 8, 8
    alpha = np.full((h, w), 0.5, np.float32)
    rgba = np.stack([np.full((h, w), 0.25, np.float32)] * 3 + [alpha], axis=-1)
    dst = tmp_path / "premult.exr"
    with OpenEXR.File({"type": OpenEXR.scanlineimage}, {"RGBA": rgba}) as f:
        f.write(str(dst))

    plain, _m, _i = RadianceRead().read(browse="", path=str(dst))
    assert float(plain[0, 0, 0, 0]) == pytest.approx(0.25, abs=1e-4)

    unpremult, _m, info = RadianceRead().read(
        browse="", path=str(dst), premultiplied=True)
    assert float(unpremult[0, 0, 0, 0]) == pytest.approx(0.5, abs=1e-4)
    assert json.loads(info)["unpremultiplied"] is True


def test_unpremultiply_never_produces_nan(tmp_path):
    """Dividing by a zero alpha would poison every downstream operation."""
    from radiance.nodes.io.write import _unpremultiply

    image = torch.full((1, 4, 4, 3), 0.5)
    mask = torch.zeros(1, 4, 4)
    out = _unpremultiply(image, mask)
    assert torch.isfinite(out).all()
    assert torch.equal(out, image), "fully transparent pixels should be left alone"


# ── raw ────────────────────────────────────────────────────────────────────

def test_raw_skips_the_colour_transform(tmp_path, rgba8):
    from radiance.nodes.io.write import RadianceRead

    PIL.fromarray(rgba8[..., :3]).save(tmp_path / "srgb.png")
    managed, _m, _i = RadianceRead().read(
        browse="", path=str(tmp_path / "srgb.png"), color_space="sRGB")
    raw, _m, info = RadianceRead().read(
        browse="", path=str(tmp_path / "srgb.png"), color_space="sRGB", raw=True)
    assert not torch.allclose(managed, raw), "raw applied the transform anyway"
    assert "raw" in json.loads(info)["color_space"]


# ── the info output ────────────────────────────────────────────────────────

def test_every_read_returns_three_outputs(png_sequence):
    from radiance.nodes.io.write import RadianceRead

    assert len(RadianceRead().read(browse="", path=str(png_sequence))) == 3


def test_the_node_declares_the_info_output_last():
    """Appended, not inserted: ComfyUI links outputs by index, so a workflow
    saved against the two-output version must keep working."""
    from radiance.nodes.io.write import RadianceRead

    assert RadianceRead.RETURN_TYPES[:2] == ("IMAGE", "MASK")
    assert RadianceRead.RETURN_NAMES == ("image", "mask", "info")
    assert len(RadianceRead.OUTPUT_TOOLTIPS) == len(RadianceRead.RETURN_TYPES)


def test_the_info_output_is_valid_json_for_every_media_type(
        png_sequence, exr_layers, tmp_path, rgba8):
    from radiance.nodes.io.write import RadianceRead

    PIL.fromarray(rgba8).save(tmp_path / "still.png")
    cases = [str(png_sequence), str(exr_layers), str(tmp_path / "still.png"), ""]
    for path in cases:
        _image, _mask, info = RadianceRead().read(browse="", path=path)
        data = json.loads(info)
        assert "kind" in data, f"{path or '(blank)'} produced {info!r}"


def test_an_empty_read_is_not_an_error():
    """A node just dropped on the canvas has no file yet."""
    from radiance.nodes.io.write import RadianceRead

    image, mask, info = RadianceRead().read(browse="", path="")
    assert image.shape[0] == 1 and mask is not None
    assert json.loads(info)["kind"] == "empty"


# ── the widgets themselves ─────────────────────────────────────────────────

def test_every_widget_the_signature_accepts_is_declared():
    """A widget missing from INPUT_TYPES silently keeps its default forever —
    which is exactly how the RELOAD button went missing for two releases."""
    import inspect

    from radiance.nodes.io.write import RadianceRead

    spec = RadianceRead.INPUT_TYPES()
    declared = set(spec.get("required", {})) | set(spec.get("optional", {}))
    accepted = set(inspect.signature(RadianceRead.read).parameters) - {"self"}
    assert declared <= accepted, f"declared but not accepted: {declared - accepted}"
    assert accepted <= declared, f"accepted but never declared: {accepted - declared}"


def test_every_widget_has_a_tooltip():
    """Fifteen widgets is a lot to guess at."""
    from radiance.nodes.io.write import RadianceRead

    spec = RadianceRead.INPUT_TYPES()
    missing = [
        name for group in ("required", "optional")
        for name, entry in spec.get(group, {}).items()
        if not (isinstance(entry, (list, tuple)) and len(entry) > 1
                and isinstance(entry[1], dict) and entry[1].get("tooltip"))
    ]
    assert not missing, f"no tooltip on: {missing}"


def test_the_layer_widget_is_a_string_not_a_combo():
    """Deliberate. ComfyUI validates a combo against the list INPUT_TYPES built,
    so a workflow saved with layer="diffuse" would fail to load on a machine
    that had never seen that file. The frontend adds the dropdown."""
    from radiance.nodes.io.write import RadianceRead

    entry = RadianceRead.INPUT_TYPES()["optional"]["layer"]
    assert entry[0] == "STRING"
    assert entry[1]["default"] == "auto"


def test_on_error_defaults_to_raising():
    from radiance.nodes.io.write import RadianceRead

    entry = RadianceRead.INPUT_TYPES()["optional"]["on_error"]
    assert entry[1]["default"] == "Error", (
        "defaulting to a black frame is the bug this replaced"
    )


# ── the HTTP routes ────────────────────────────────────────────────────────

def test_the_layer_route_is_sandboxed(monkeypatch, tmp_path):
    """An unauthenticated GET taking a path is a file-existence oracle for the
    whole filesystem unless it is bounded. The Read *node* still opens any path
    the user types -- that is the job; this bounds only the HTTP route."""
    from radiance.nodes.io import write as nodes_io

    allowed = tmp_path / "plates"
    allowed.mkdir()
    (allowed / "sh010.exr").write_bytes(b"")
    monkeypatch.setattr(nodes_io, "_allowed_read_roots", lambda: [str(allowed)])

    assert nodes_io._is_inside_allowed_read_root(str(allowed / "sh010.exr"))
    assert not nodes_io._is_inside_allowed_read_root("/etc/passwd")
    assert not nodes_io._is_inside_allowed_read_root(str(tmp_path / "elsewhere.exr"))


def test_the_sandbox_is_not_defeated_by_dot_dot(monkeypatch, tmp_path):
    from radiance.nodes.io import write as nodes_io

    allowed = tmp_path / "allowed"
    allowed.mkdir()
    (tmp_path / "secret.txt").write_text("not yours")
    monkeypatch.setattr(nodes_io, "_allowed_read_roots", lambda: [str(allowed)])

    for escape in (
        str(allowed / ".." / "secret.txt"),
        str(allowed) + "/../secret.txt",
        str(allowed / "a" / ".." / ".." / "secret.txt"),
    ):
        assert not nodes_io._is_inside_allowed_read_root(escape), escape


def test_the_sandbox_is_not_defeated_by_a_symlink(monkeypatch, tmp_path):
    """`realpath`, not `abspath` — a link inside the root must not lead out."""
    from radiance.nodes.io import write as nodes_io

    allowed = tmp_path / "allowed"
    allowed.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("not yours")
    link = allowed / "shortcut.txt"
    try:
        link.symlink_to(secret)
    except (OSError, NotImplementedError):
        pytest.skip("this filesystem has no symlinks")
    monkeypatch.setattr(nodes_io, "_allowed_read_roots", lambda: [str(allowed)])
    assert not nodes_io._is_inside_allowed_read_root(str(link))


def test_a_prefix_collision_is_not_inside_the_root(monkeypatch, tmp_path):
    """/plates_private must not match a /plates root by string prefix."""
    from radiance.nodes.io import write as nodes_io

    (tmp_path / "plates").mkdir()
    (tmp_path / "plates_private").mkdir()
    monkeypatch.setattr(
        nodes_io, "_allowed_read_roots", lambda: [str(tmp_path / "plates")])
    assert not nodes_io._is_inside_allowed_read_root(
        str(tmp_path / "plates_private" / "secret.exr"))


def test_the_ui_probe_describes_each_media_type(exr_layers, png_sequence):
    from radiance.nodes.io.write import _probe_for_ui

    exr = _probe_for_ui(str(exr_layers))
    assert exr["kind"] == "exr" and "layer" in exr["summary"]

    seq = _probe_for_ui(str(png_sequence))
    assert "1001-1008" in seq["summary"], seq["summary"]


# ── route registration ─────────────────────────────────────────────────────
#
# AUDIT-FIX (2026-09): the idempotency test below was the one test in the suite
# with no assertion at all. It called `register_read_routes()` twice and passed
# if nothing raised, against conftest.py's `_FakeRoutes`, whose `get`/`post`
# return an identity decorator and keep no state. So the duplicate-registration
# crash it names was doubly unreachable: nothing was asserted, and nothing the
# fake was handed could ever raise. tests/test_ocio_endpoints.py:167 records the
# same defect in the same fake; this file did not get the fix.
#
# The machinery below borrows the real aiohttp so the second pass meets the real
# UrlDispatcher, which is the thing that raises "method HEAD is already
# registered".


def _aiohttp_names():
    return [n for n in list(sys.modules) if n == "aiohttp" or n.startswith("aiohttp.")]


@contextlib.contextmanager
def _real_aiohttp_web():
    """Hand back the real `aiohttp.web`, past conftest's stub.

    conftest installs a bare `aiohttp` ModuleType when ComfyUI is absent. It has
    no `UrlDispatcher`, no `json_response`, and no duplicate detection, which is
    exactly why registering twice against it proved nothing.
    """
    saved = {n: sys.modules[n] for n in _aiohttp_names()}
    for name in saved:
        del sys.modules[name]
    try:
        web = importlib.import_module("aiohttp.web")
    except ImportError:  # pragma: no cover - aiohttp is a ComfyUI dependency
        for name in _aiohttp_names():
            del sys.modules[name]
        sys.modules.update(saved)
        pytest.skip("this environment has no real aiohttp")
    if not hasattr(web, "UrlDispatcher"):  # pragma: no cover - belt and braces
        for name in _aiohttp_names():
            del sys.modules[name]
        sys.modules.update(saved)
        pytest.fail("imported an aiohttp.web with no UrlDispatcher, so it is a stub")
    try:
        yield web
    finally:
        for name in _aiohttp_names():
            del sys.modules[name]
        sys.modules.update(saved)


class _DispatcherRoutes:
    """ComfyUI's `PromptServer.instance.routes`, backed by a real dispatcher.

    ComfyUI hands out a `web.RouteTableDef`, which only appends; the duplicate
    RuntimeError surfaces later, when the app adds that table to its
    `UrlDispatcher`. Registering straight into a dispatcher raises from the same
    code for the same reason, without needing a running Application.
    """

    def __init__(self, dispatcher):
        self._dispatcher = dispatcher

    def get(self, path):
        def deco(fn):
            self._dispatcher.add_get(path, fn)
            return fn
        return deco

    def post(self, path):
        def deco(fn):
            self._dispatcher.add_post(path, fn)
            return fn
        return deco


@contextlib.contextmanager
def _prompt_server_on(dispatcher):
    """Install a `server` module whose PromptServer routes into *dispatcher*.

    conftest's own `server` stub is put back afterwards, so nothing leaks into
    the rest of the suite. A fresh PromptServer instance also means a fresh
    `_radiance_read_routes_registered` flag: the module already registered its
    routes once at import time, against conftest's fake.
    """
    module = types.ModuleType("server")
    module.PromptServer = types.SimpleNamespace(
        instance=types.SimpleNamespace(routes=_DispatcherRoutes(dispatcher)))

    saved = sys.modules.get("server")
    sys.modules["server"] = module
    try:
        yield
    finally:
        if saved is not None:
            sys.modules["server"] = saved
        else:
            sys.modules.pop("server", None)


def _registered(dispatcher):
    """Counter of (method, path) for everything currently in the dispatcher."""
    return collections.Counter(
        (r.method, r.resource.canonical) for r in dispatcher.routes())


def test_route_registration_is_idempotent():
    """Registering twice used to crash ComfyUI at startup with
    'method HEAD is already registered'.

    ComfyUI puts `custom_nodes/` on `sys.path`, so a user script doing
    `import viewer` produces a second copy of a module and a second registration
    pass. `register_read_routes` survives that by flagging the PromptServer
    singleton, and this asserts the flag does its job: the routes land once, the
    second pass adds nothing, and it does not raise on the way through.
    """
    from radiance.nodes.io.write import register_read_routes

    with _real_aiohttp_web() as web:
        dispatcher = web.UrlDispatcher()

        # Harness check, not a claim about Radiance: this must be the real
        # dispatcher, the one that refuses a duplicate. A stub that accepts
        # everything is what made the old version of this test unfalsifiable.
        async def _probe(request):  # pragma: no cover - never called
            return None
        dispatcher.add_get("/radiance/_duplicate_probe", _probe)
        with pytest.raises(RuntimeError, match="already registered"):
            dispatcher.add_get("/radiance/_duplicate_probe", _probe)

        with _prompt_server_on(dispatcher):
            register_read_routes()
            first = _registered(dispatcher)
            # Without this the whole test could pass on a dispatcher nothing
            # ever reached, which is the defect it replaces.
            assert first, (
                "no route reached the dispatcher, so registering a second time "
                "proves nothing"
            )
            assert first[("HEAD", "/radiance/media/info")] == 1, (
                "aiohttp registers HEAD alongside GET, and HEAD is what the "
                "startup crash named; without it this is not the code path "
                f"that crashed ComfyUI: {sorted(first)}"
            )

            register_read_routes()          # the second import's pass
            second = _registered(dispatcher)

    repeated = {k: n for k, n in first.items() if n > 1}
    assert not repeated, f"the first pass already registered something twice: {repeated}"

    added = second - first
    assert not added, (
        "the second registration pass reached the dispatcher again: "
        f"{sorted(added.elements())} landed a second time. Every one of those "
        "shadows a live route, and an adjacent repeat is the aiohttp "
        "RuntimeError that crashed ComfyUI at startup"
    )
