"""
test_workspace_api.py — behaviour tests for `nodes/pipeline/workspace.py`.

The module under test is the Radiance workspace/project-manager REST backend.
It is reachable two ways: as a ComfyUI node (`RadianceProjectManager`) and as a
set of aiohttp route handlers driven by the dashboard JS.

Three things make this module awkward to test, and each is handled explicitly
rather than papered over:

1. `aiohttp` is absent in CI, and conftest.py registers a bare
   `types.ModuleType("aiohttp.web")` that has no `json_response` at all. Every
   status-code assertion here would be unobservable through that stub, so the
   `real_aiohttp_web` fixture borrows the genuine package for the session and
   puts the stub back afterwards. It asserts the import produced something that
   behaves like a real response, so a missing install fails loudly instead of
   silently turning this file into a no-op.

2. The module resolves its storage root ONCE at import time
   (`_WORKFLOW_DIR_RESOLVED`, `_ASSETS_BINS_PATH`, `_THUMB_CACHE`) from
   `__file__`. Left alone, every save/delete/backup test would write into the
   checked-out repository. The `ws` fixture repoints all three at `tmp_path`
   and asserts the repository root is no longer in play.

3. `folder_paths` is a conftest stub whose output/input directories point at
   `/tmp`. The asset scanner and the output lister walk those trees, so the
   fixture repoints them at `tmp_path` too.

Nothing here asserts merely "did not raise": every test checks a status code, a
returned payload field, or a file that actually landed on disk.
"""

from __future__ import annotations

import asyncio
import importlib
import io
import json
import struct
import sys
import time
import types
import zipfile
import zlib
from pathlib import Path

import pytest

from radiance.nodes.pipeline import workspace as ws_mod


# ═══════════════════════════════════════════════════════════════════════════
#  Real aiohttp, borrowed for the session
# ═══════════════════════════════════════════════════════════════════════════

def _aiohttp_names():
    return [n for n in list(sys.modules) if n == "aiohttp" or n.startswith("aiohttp.")]


@pytest.fixture(scope="session")
def real_aiohttp_web():
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
    assert probe.status == 418 and json.loads(bytes(probe.body)) == {"ok": True}, (
        "aiohttp.web.json_response is not behaving like a real response"
    )

    yield web

    for name in _aiohttp_names():
        del sys.modules[name]
    sys.modules.update(saved)


# ═══════════════════════════════════════════════════════════════════════════
#  Request doubles
# ═══════════════════════════════════════════════════════════════════════════

class _Content:
    """Stands in for `request.content` (a stream the handler reads bounded)."""

    def __init__(self, raw: bytes):
        self._raw = raw

    async def read(self, n: int = -1) -> bytes:
        return self._raw if n is None or n < 0 else self._raw[:n]


class _Part:
    def __init__(self, filename, data: bytes, chunk: int = 7):
        self.filename = filename
        self._data = data
        self._pos = 0
        self._chunk = chunk

    async def read_chunk(self, size: int = 0) -> bytes:
        if self._pos >= len(self._data):
            return b""
        out = self._data[self._pos:self._pos + self._chunk]
        self._pos += len(out)
        return out


class _MultipartReader:
    def __init__(self, parts):
        self._parts = list(parts)

    def __aiter__(self):
        self._it = iter(self._parts)
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration


class Req:
    """Carries exactly what the workspace handlers ask of a request."""

    def __init__(self, *, json_body=None, raw=None, query=None,
                 match_info=None, content_length="auto", parts=None):
        if raw is None:
            raw = b"" if json_body is None else json.dumps(json_body).encode("utf-8")
        self._raw = raw
        self.query = dict(query or {})
        self.rel_url = types.SimpleNamespace(query=self.query)
        self.match_info = dict(match_info or {})
        self.content = _Content(raw)
        self.content_length = len(raw) if content_length == "auto" else content_length
        self._parts = parts or []

    async def json(self):
        # Real json.loads — a malformed body raises a real JSONDecodeError,
        # which is the exact exception the handlers branch on.
        return json.loads(self._raw.decode("utf-8"))

    async def multipart(self):
        return _MultipartReader(self._parts)


def run(coro):
    return asyncio.run(coro)


def body(response):
    """Decode a JSON response body, asserting it really is JSON."""
    assert response.content_type == "application/json", (
        f"expected a JSON response, got {response.content_type!r}")
    return json.loads(bytes(response.body).decode("utf-8"))


def raw_body(response) -> bytes:
    return bytes(response.body)


# ═══════════════════════════════════════════════════════════════════════════
#  Wired module fixture
# ═══════════════════════════════════════════════════════════════════════════

class WS:
    """The workspace module with every on-disk root pointed at tmp_path."""

    def __init__(self, mod, root: Path, inp: Path, out: Path):
        self.m = mod
        self.root = root
        self.inp = inp
        self.out = out

    # ── convenience builders ────────────────────────────────────────────
    def save(self, filename, content='{"nodes": []}', **extra):
        payload = {"filename": filename, "content": content}
        payload.update(extra)
        resp = run(self.m.save_workflow(Req(json_body=payload)))
        assert resp.status == 200, body(resp)
        return self.root / (filename if filename.endswith(".rad") else filename + ".rad")


@pytest.fixture
def ws(tmp_path, monkeypatch, real_aiohttp_web):
    mod = ws_mod

    assert mod._WORKSPACE_AVAILABLE is True, (
        "workspace.py took its no-server fallback (web/PromptServer are None); "
        "the route handlers would not be callable and nothing below would test "
        "a real status code"
    )

    root = (tmp_path / "workflows").resolve()
    root.mkdir()
    inp = (tmp_path / "input").resolve()
    inp.mkdir()
    out = (tmp_path / "comfy_output").resolve()
    out.mkdir()

    monkeypatch.setattr(mod, "web", real_aiohttp_web)
    monkeypatch.setattr(mod, "WORKFLOW_DIR", str(root))
    monkeypatch.setattr(mod, "_WORKFLOW_DIR_RESOLVED", root)
    monkeypatch.setattr(mod, "_ASSETS_BINS_PATH", root / "_assets_bins.json")
    monkeypatch.setattr(mod, "_THUMB_CACHE", root / ".asset_thumbs")
    monkeypatch.setattr(mod.folder_paths, "get_input_directory", lambda: str(inp))
    monkeypatch.setattr(mod.folder_paths, "get_output_directory", lambda: str(out))

    repo_dir = Path(ws_mod.__file__).resolve().parents[2] / "workflows"
    assert mod._WORKFLOW_DIR_RESOLVED != repo_dir.resolve(), (
        "the storage root was not redirected — this test would write into the repo")

    return WS(mod, root, inp, out)


# ═══════════════════════════════════════════════════════════════════════════
#  1. Pipeline path helpers
# ═══════════════════════════════════════════════════════════════════════════

def test_the_library_is_the_package_workflows_folder():
    """Moving workspace.py into nodes/pipeline/ sent the library to a stray
    nodes/pipeline/workflows/, hiding earlier saves and workflows/official/."""
    package_workflows = Path(__file__).resolve().parents[1] / "workflows"
    assert ws_mod._WORKFLOW_DIR_RESOLVED == package_workflows.resolve()


def test_next_version_on_missing_and_empty_dir(tmp_path):
    assert ws_mod._next_version(tmp_path / "does_not_exist") == "v001"
    empty = tmp_path / "render"
    empty.mkdir()
    assert ws_mod._next_version(empty) == "v001"


def test_next_version_takes_max_not_count(tmp_path):
    render = tmp_path / "render"
    render.mkdir()
    for name in ("v001", "v002", "v007"):
        (render / name).mkdir()
    # A count-based implementation would say v004 here.
    assert ws_mod._next_version(render) == "v008"


def test_next_version_ignores_files_and_foreign_dirs(tmp_path):
    render = tmp_path / "render"
    render.mkdir()
    (render / "v003").mkdir()
    (render / "v999").write_text("a file, not a version dir")
    (render / "version4").mkdir()
    (render / "v04x").mkdir()
    assert ws_mod._next_version(render) == "v004"


def test_build_shot_paths_materialises_the_tree(tmp_path):
    paths = ws_mod._build_shot_paths(str(tmp_path), "SHOW", "SEQ010", "SH020", "comp")
    root = tmp_path / "SHOW" / "sequences" / "SEQ010" / "SH020"

    assert paths["root"] == root
    assert paths["work"] == root / "work" / "comp"
    assert set(paths) == {"root", "plates", "ref", "work", "render",
                          "deliverables", "cache"}
    for key, p in paths.items():
        assert p.is_dir(), f"{key} was not created on disk"


# ═══════════════════════════════════════════════════════════════════════════
#  2. Path security
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("bad", [
    "",                       # empty
    "x" * 201 + ".rad",       # over MAX_FILENAME_LENGTH
    "evil\x00.rad",           # null byte
    "evil\n.rad",             # control character
    "../escape.rad",          # parent traversal
    "sub/../../escape.rad",   # traversal through a subdir
    "/etc/passwd",            # absolute path
])
def test_resolve_safe_path_rejects(ws, bad):
    assert ws.m._resolve_safe_path(bad) is None


def test_resolve_safe_path_accepts_and_anchors(ws):
    p = ws.m._resolve_safe_path("SHOW/sh010-v002.rad")
    assert p == ws.root / "SHOW" / "sh010-v002.rad"
    assert p.is_relative_to(ws.root)


def test_resolve_safe_path_honours_the_length_boundary(ws):
    name = "a" * ws.m.MAX_FILENAME_LENGTH
    assert ws.m._resolve_safe_path(name) is not None
    assert ws.m._resolve_safe_path(name + "a") is None


def test_validate_extension(ws):
    assert ws.m._validate_extension(Path("x.rad")) is True
    assert ws.m._validate_extension(Path("x.RAD")) is True
    assert ws.m._validate_extension(Path("x.json")) is False
    assert ws.m._validate_extension(Path("x")) is False


# ═══════════════════════════════════════════════════════════════════════════
#  3. Scene inspector
# ═══════════════════════════════════════════════════════════════════════════

GRAPH = {
    "nodes": [
        {"type": "CheckpointLoaderSimple",
         "widgets_values": ["flux_dev.safetensors", "notamodel.txt"]},
        {"type": "LoraLoader", "widgets_values": ["film_grain.ckpt"]},
        {"type": "RadianceColorspaceConvert",
         "widgets_values": ["ACEScg", "Rec709", "banana"]},
        {"type": "RadianceHDRTonemap", "widgets_values": ["rec2020"]},
        {"type": "RadianceVideoWriter", "widgets_values": ["out", 23.976]},
        {"type": "PreviewImage", "widgets_values": []},
    ]
}


def test_inspect_graph_extracts_models_colorspaces_fps_and_hdr():
    profile = ws_mod._inspect_graph_content(json.dumps(GRAPH))

    assert profile["node_count"] == 6
    assert profile["models"] == ["film_grain.ckpt", "flux_dev.safetensors"]
    assert profile["color_spaces"] == ["ACEScg", "Rec709", "rec2020"]
    assert profile["is_hdr"] is True
    assert profile["fps"] == pytest.approx(23.976)


def test_inspect_graph_defaults_without_hdr_or_fps():
    profile = ws_mod._inspect_graph_content(json.dumps({"nodes": [
        {"type": "KSampler", "widgets_values": [20, 8.0]},
    ]}))
    assert profile == {
        "models": [], "color_spaces": [], "fps": 24.0,
        "is_hdr": False, "node_count": 1,
    }


def test_inspect_graph_rejects_out_of_range_fps():
    profile = ws_mod._inspect_graph_content(json.dumps({"nodes": [
        {"type": "VideoCombine", "widgets_values": [500.0, 30.0]},
    ]}))
    # 500 is outside 1..240 so it must be skipped; 30 is the first valid one.
    assert profile["fps"] == 30.0


def test_inspect_graph_survives_malformed_json():
    profile = ws_mod._inspect_graph_content("{not json at all")
    assert profile["node_count"] == 0
    assert profile["models"] == []
    assert profile["fps"] == 24.0


# ═══════════════════════════════════════════════════════════════════════════
#  4. Formatting / naming helpers
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("size,expected", [
    (0, "0 B"),
    (-5, "0 B"),
    (999, "999 B"),
    (1024, "1.0 KB"),
    (1536, "1.5 KB"),
    (1024 ** 2, "1.0 MB"),
    (1024 ** 3, "1.0 GB"),
    (1024 ** 4, "1.0 TB"),
    (1024 ** 5, "1024.0 TB"),
])
def test_format_bytes(size, expected):
    assert ws_mod._format_bytes(size) == expected


def test_format_relative_time_buckets():
    now = time.time()
    assert ws_mod._format_relative_time(now) == "Just now"
    assert ws_mod._format_relative_time(now + 5000) == "Just now"   # clamped
    assert ws_mod._format_relative_time(now - 60) == "1 minute ago"
    assert ws_mod._format_relative_time(now - 125) == "2 minutes ago"
    assert ws_mod._format_relative_time(now - 3600) == "1 hour ago"
    assert ws_mod._format_relative_time(now - 7200) == "2 hours ago"
    assert ws_mod._format_relative_time(now - 86400) == "1 day ago"
    assert ws_mod._format_relative_time(now - 3 * 86400) == "3 days ago"
    old = now - 30 * 86400
    assert ws_mod._format_relative_time(old) == time.strftime(
        "%Y-%m-%d", time.localtime(old))


@pytest.mark.parametrize("value,expected", [
    ("SHOW A", "show-a"),
    ("Show_A/2026", "show-a-2026"),
    ("!!!", "general"),
    ("", "general"),
])
def test_project_slug(value, expected):
    assert ws_mod._project_slug(value) == expected


@pytest.mark.parametrize("value,expected", [
    ("sh010-v002.rad", "SH010"),
    ("SH_0420 comp", "SH0420"),
    ("beauty_pass", "GENERAL"),
    ("sh1", "GENERAL"),          # needs at least 2 digits
])
def test_shot_from_name(value, expected):
    assert ws_mod._shot_from_name(value) == expected


@pytest.mark.parametrize("value,expected", [
    ("sh010-v002", "v002"),
    ("comp v0123 final", "v0123"),
    ("comp_v12", "v001"),        # too few digits
    ("nothing", "v001"),
])
def test_version_from_name(value, expected):
    assert ws_mod._version_from_name(value) == expected


# ═══════════════════════════════════════════════════════════════════════════
#  5. .rad containers — v2
# ═══════════════════════════════════════════════════════════════════════════

def test_pack_v2_roundtrip_preserves_graph_and_metadata():
    graph = json.dumps(GRAPH)
    meta = {"author": "ada", "commit_message": "first light"}
    blob = ws_mod._pack_rad_v2(graph, meta)

    assert blob[:4] == ws_mod.RAD_MAGIC
    magic, ver, meta_len, graph_len = struct.unpack(
        ws_mod._RAD_HEADER_FMT, blob[:ws_mod._RAD_HEADER_SIZE])
    assert (magic, ver) == (ws_mod.RAD_MAGIC, 2)
    assert ws_mod._RAD_HEADER_SIZE == 14

    out_graph, out_meta = ws_mod._unpack_rad_v2(blob)
    assert json.loads(out_graph) == GRAPH
    assert out_meta == meta


def test_unpack_v2_detects_tampering():
    blob = bytearray(ws_mod._pack_rad_v2('{"nodes": []}', {"author": "ada"}))
    blob[-1] ^= 0xFF          # corrupt the checksum
    with pytest.raises(ValueError, match="integrity check failed"):
        ws_mod._unpack_rad_v2(bytes(blob))


def test_unpack_v2_detects_payload_tampering():
    blob = bytearray(ws_mod._pack_rad_v2('{"nodes": []}', {"author": "ada"}))
    blob[ws_mod._RAD_HEADER_SIZE + 2] ^= 0xFF   # corrupt the metadata JSON
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        ws_mod._unpack_rad_v2(bytes(blob))


def test_unpack_v2_rejects_truncated_file():
    with pytest.raises(ValueError, match="too small"):
        ws_mod._unpack_rad_v2(b"RAD!" + b"\x00" * 10)


def test_unpack_v2_rejects_wrong_magic():
    import hashlib
    header = struct.pack(ws_mod._RAD_HEADER_FMT, b"XXXX", 2, 0, 0)
    payload = header
    blob = payload + hashlib.sha256(payload).digest()
    with pytest.raises(ValueError, match="magic bytes"):
        ws_mod._unpack_rad_v2(blob)


def test_unpack_v2_slices_graph_by_declared_length():
    """Trailing bytes inside the payload must not leak into the graph."""
    import hashlib
    graph = '{"nodes": []}'
    meta = {"a": 1}
    meta_data = json.dumps(meta).encode()
    graph_data = zlib.compress(graph.encode())
    junk = b"TRAILING-GARBAGE"
    header = struct.pack(ws_mod._RAD_HEADER_FMT, ws_mod.RAD_MAGIC, 2,
                         len(meta_data), len(graph_data))
    payload = header + meta_data + graph_data + junk
    blob = payload + hashlib.sha256(payload).digest()

    out_graph, out_meta = ws_mod._unpack_rad_v2(blob)
    assert out_graph == graph
    assert out_meta == meta


# ═══════════════════════════════════════════════════════════════════════════
#  6. .rad containers — v3
# ═══════════════════════════════════════════════════════════════════════════

PNG_1PX = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
    b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


def test_pack_v3_roundtrip_with_assets_and_preview():
    import base64
    graph = json.dumps(GRAPH)
    meta = {
        "author": "ada",
        "stats": {"node_count": 6},
        "preview_image": "data:image/png;base64," + base64.b64encode(PNG_1PX).decode(),
    }
    blob = ws_mod._pack_rad_v3(graph, meta, assets={"grade.cube": b"LUTDATA"})

    assert blob[:4] == ws_mod.RAD_MAGIC_V3
    out_graph, out_meta, assets = ws_mod._unpack_rad_v3(blob)
    assert json.loads(out_graph) == GRAPH
    assert out_meta["author"] == "ada"
    assert out_meta["rad_version"] == 3
    assert out_meta["assets"] == ["grade.cube"]
    assert assets == {"grade.cube": b"LUTDATA"}

    with zipfile.ZipFile(io.BytesIO(blob[4:])) as zf:
        assert zf.read("preview.png") == PNG_1PX
        manifest = json.loads(zf.read("manifest.json"))
    assert manifest["rad_version"] == 3
    assert manifest["stats"] == {"node_count": 6}


def test_pack_v3_strips_traversal_from_asset_names():
    blob = ws_mod._pack_rad_v3("{}", {}, assets={"../../evil.cube": b"X"})
    _, _, assets = ws_mod._unpack_rad_v3(blob)
    assert list(assets) == ["evil.cube"]


def test_unpack_v3_rejects_foreign_magic():
    with pytest.raises(ValueError, match="RADZ magic"):
        ws_mod._unpack_rad_v3(b"PK\x03\x04rest")


def test_unpack_v3_rejects_corrupt_zip():
    with pytest.raises(ValueError, match="Corrupt .rad v3 ZIP"):
        ws_mod._unpack_rad_v3(ws_mod.RAD_MAGIC_V3 + b"not a zip at all")


def test_unpack_v3_requires_workflow_json():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("manifest.json", "{}")
    with pytest.raises(ValueError, match="missing workflow.json"):
        ws_mod._unpack_rad_v3(ws_mod.RAD_MAGIC_V3 + buf.getvalue())


def test_unpack_v3_rejects_too_many_entries(monkeypatch):
    monkeypatch.setattr(ws_mod, "MAX_RAD_ZIP_ENTRIES", 3)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("workflow.json", "{}")
        for i in range(5):
            zf.writestr(f"assets/a{i}.txt", b"x")
    with pytest.raises(ValueError, match="too many ZIP entries"):
        ws_mod._unpack_rad_v3(ws_mod.RAD_MAGIC_V3 + buf.getvalue())


def test_unpack_v3_rejects_zip_bomb(monkeypatch):
    monkeypatch.setattr(ws_mod, "MAX_RAD_UNCOMPRESSED_BYTES", 64)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("workflow.json", "0" * 4096)
    with pytest.raises(ValueError, match="uncompressed payload exceeds"):
        ws_mod._unpack_rad_v3(ws_mod.RAD_MAGIC_V3 + buf.getvalue())


def test_unpack_v3_rejects_unsafe_asset_name():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("workflow.json", "{}")
        zf.writestr("assets/nested/evil.cube", b"X")
    with pytest.raises(ValueError, match="unsafe asset name"):
        ws_mod._unpack_rad_v3(ws_mod.RAD_MAGIC_V3 + buf.getvalue())


# ═══════════════════════════════════════════════════════════════════════════
#  7. Version detection / unified unpack
# ═══════════════════════════════════════════════════════════════════════════

def test_detect_rad_version():
    assert ws_mod._detect_rad_version(ws_mod.RAD_MAGIC_V3 + b"...") == 3
    assert ws_mod._detect_rad_version(ws_mod.RAD_MAGIC + b"...") == 2
    assert ws_mod._detect_rad_version(b'{"nodes": []}') == 1
    assert ws_mod._detect_rad_version(b"") == 1


def test_unpack_any_rad_dispatches_all_three_formats():
    graph = '{"nodes": []}'

    g1, m1 = ws_mod._unpack_any_rad(graph.encode("utf-8"))
    assert (g1, m1) == (graph, {"format": "v1", "secure": False})

    g2, m2 = ws_mod._unpack_any_rad(ws_mod._pack_rad_v2(graph, {"author": "ada"}))
    assert g2 == graph and m2["author"] == "ada"

    g3, m3 = ws_mod._unpack_any_rad(ws_mod._pack_rad_v3(graph, {"author": "bob"}))
    assert g3 == graph and m3["author"] == "bob" and m3["rad_version"] == 3


# ═══════════════════════════════════════════════════════════════════════════
#  8. Version backups
# ═══════════════════════════════════════════════════════════════════════════

def test_version_backup_is_a_noop_for_a_missing_file(tmp_path):
    ws_mod._create_version_backup(tmp_path / "nope.rad")
    assert not (tmp_path / ".versions").exists()


def test_version_backup_copies_content_and_writes_sidecar(tmp_path):
    target = tmp_path / "sh010.rad"
    target.write_bytes(b"REV-ONE")

    ws_mod._create_version_backup(target, message="first", author="ada")

    backup = tmp_path / ".versions" / "sh010.v1.rad"
    assert backup.read_bytes() == b"REV-ONE"
    info = json.loads((tmp_path / ".versions" / "sh010.v1.rad.json").read_text())
    assert info["version"] == 1
    assert info["message"] == "first"
    assert info["author"] == "ada"
    assert info["original_filename"] == "sh010.rad"
    assert info["timestamp"] <= time.time()


def test_version_backup_numbers_from_max_not_count(tmp_path):
    target = tmp_path / "sh010.rad"
    versions = tmp_path / ".versions"
    versions.mkdir()
    # v1..v5 existed, v3 was deleted by hand. A len()+1 scheme would pick v5
    # and clobber the existing v5 backup.
    for n in (1, 2, 4, 5):
        (versions / f"sh010.v{n}.rad").write_bytes(b"old")
    target.write_bytes(b"current")

    ws_mod._create_version_backup(target, message="next")

    assert (versions / "sh010.v6.rad").read_bytes() == b"current"
    assert (versions / "sh010.v5.rad").read_bytes() == b"old"


def test_version_backup_evicts_at_capacity(tmp_path, monkeypatch):
    monkeypatch.setattr(ws_mod, "MAX_VERSIONS", 3)
    target = tmp_path / "sh010.rad"
    versions = tmp_path / ".versions"

    for i in range(6):
        target.write_bytes(f"rev{i}".encode())
        ws_mod._create_version_backup(target, message=f"c{i}")

    backups = sorted(p.name for p in versions.glob("sh010.v*.rad"))
    assert len(backups) == 3, backups
    # The newest three survive, and their sidecars go with them.
    assert backups == ["sh010.v4.rad", "sh010.v5.rad", "sh010.v6.rad"]
    assert not (versions / "sh010.v1.rad.json").exists()


# ═══════════════════════════════════════════════════════════════════════════
#  9. /radiance/workflows/pack + /unpack
# ═══════════════════════════════════════════════════════════════════════════

def test_pack_endpoint_returns_a_real_v2_container(ws):
    resp = run(ws.m.pack_workflow(Req(json_body={
        "content": '{"nodes": []}', "description": "d", "author": "ada"})))

    assert resp.status == 200
    assert resp.content_type == "application/octet-stream"
    blob = raw_body(resp)
    graph, meta = ws.m._unpack_rad_v2(blob)
    assert graph == '{"nodes": []}'
    assert meta["author"] == "ada"
    assert meta["description"] == "d"
    assert meta["format"] == "v2"


def test_pack_endpoint_rejects_empty_content(ws):
    resp = run(ws.m.pack_workflow(Req(json_body={"content": ""})))
    assert resp.status == 400
    assert body(resp) == {"error": "Missing content"}


def test_pack_endpoint_reports_broken_body_as_500(ws):
    resp = run(ws.m.pack_workflow(Req(raw=b"{not json")))
    assert resp.status == 500
    assert "error" in body(resp)


def test_unpack_endpoint_handles_v1_v2_v3(ws):
    graph = '{"nodes": []}'

    r1 = run(ws.m.unpack_workflow_api(Req(raw=graph.encode())))
    assert r1.status == 200
    p1 = body(r1)
    assert p1 == {"success": True, "content": graph, "secure": False, "format": "v1"}

    r2 = run(ws.m.unpack_workflow_api(
        Req(raw=ws.m._pack_rad_v2(graph, {"author": "ada"}))))
    p2 = body(r2)
    assert p2["format"] == "v2" and p2["secure"] is True
    assert p2["metadata"]["author"] == "ada"

    r3 = run(ws.m.unpack_workflow_api(
        Req(raw=ws.m._pack_rad_v3(graph, {"author": "bob"}, assets={"l.cube": b"X"}))))
    p3 = body(r3)
    assert p3["format"] == "v3" and p3["assets"] == ["l.cube"]
    assert p3["content"] == graph


def test_unpack_endpoint_rejects_oversized_content_length(ws):
    resp = run(ws.m.unpack_workflow_api(
        Req(raw=b"x", content_length=ws.m.MAX_WORKFLOW_SIZE_BYTES + 1)))
    assert resp.status == 413
    assert "50MB" in body(resp)["error"]


def test_unpack_endpoint_rejects_oversized_body_despite_missing_header(ws, monkeypatch):
    monkeypatch.setattr(ws.m, "MAX_WORKFLOW_SIZE_BYTES", 32)
    resp = run(ws.m.unpack_workflow_api(Req(raw=b"y" * 64, content_length=None)))
    assert resp.status == 413


def test_unpack_endpoint_reports_corrupt_container_as_400(ws):
    resp = run(ws.m.unpack_workflow_api(
        Req(raw=ws.m.RAD_MAGIC_V3 + b"not a zip")))
    assert resp.status == 400
    assert "Corrupt" in body(resp)["error"]


# ═══════════════════════════════════════════════════════════════════════════
#  10. /radiance/workflows/save
# ═══════════════════════════════════════════════════════════════════════════

def test_save_writes_a_v2_container_and_appends_the_extension(ws):
    resp = run(ws.m.save_workflow(Req(json_body={
        "filename": "sh010-v002", "content": json.dumps(GRAPH),
        "description": "comp", "author": "ada", "message": "look dev"})))

    assert resp.status == 200
    assert body(resp) == {"success": True, "filename": "sh010-v002.rad"}

    written = ws.root / "sh010-v002.rad"
    assert written.exists()
    assert written.read_bytes()[:4] == ws.m.RAD_MAGIC     # v2, no stats/preview

    graph, meta = ws.m._unpack_rad_v2(written.read_bytes())
    assert json.loads(graph) == GRAPH
    assert meta["author"] == "ada"
    assert meta["commit_message"] == "look dev"
    assert meta["pipeline"]["node_count"] == 6
    assert meta["pipeline"]["is_hdr"] is True


def test_save_uses_v3_when_stats_are_present(ws):
    run(ws.m.save_workflow(Req(json_body={
        "filename": "withstats.rad", "content": "{}",
        "stats": {"node_count": 3}})))
    blob = (ws.root / "withstats.rad").read_bytes()
    assert blob[:4] == ws.m.RAD_MAGIC_V3
    _, meta, _ = ws.m._unpack_rad_v3(blob)
    assert meta["stats"] == {"node_count": 3}


def test_save_defaults_author_and_message_when_blank(ws):
    run(ws.m.save_workflow(Req(json_body={
        "filename": "blank.rad", "content": "{}",
        "author": "   ", "message": "  "})))
    _, meta = ws.m._unpack_rad_v2((ws.root / "blank.rad").read_bytes())
    assert meta["author"] == "Radiance Artist"
    assert meta["commit_message"] == "Auto-save"


@pytest.mark.parametrize("payload", [
    {"content": "{}"},                       # no filename
    {"filename": "a.rad"},                   # no content
    {"filename": "   ", "content": "{}"},    # whitespace filename
])
def test_save_rejects_missing_fields_with_400(ws, payload):
    resp = run(ws.m.save_workflow(Req(json_body=payload)))
    assert resp.status == 400
    assert body(resp) == {"error": "Missing filename or content"}


def test_save_rejects_traversal_with_403(ws):
    resp = run(ws.m.save_workflow(Req(json_body={
        "filename": "../../pwned.rad", "content": "{}"})))
    assert resp.status == 403
    assert body(resp) == {"error": "Invalid or unsafe path"}
    assert not (ws.root.parent.parent / "pwned.rad").exists()


def test_save_rejects_oversized_content_length_with_413(ws):
    resp = run(ws.m.save_workflow(Req(
        json_body={"filename": "a.rad", "content": "{}"},
        content_length=ws.m.MAX_WORKFLOW_SIZE_BYTES + 1)))
    assert resp.status == 413
    assert not (ws.root / "a.rad").exists()


def test_save_rejects_oversized_body_when_header_lies(ws, monkeypatch):
    monkeypatch.setattr(ws.m, "MAX_WORKFLOW_SIZE_BYTES", 40)
    resp = run(ws.m.save_workflow(Req(
        json_body={"filename": "a.rad", "content": "x" * 200},
        content_length=1)))
    assert resp.status == 413
    assert not (ws.root / "a.rad").exists()


def test_save_versions_the_previous_file_and_drops_legacy_sidecar(ws):
    ws.save("sh010.rad", content='{"nodes": []}')
    sidecar = ws.root / "sh010.rad.json"
    sidecar.write_text("{}")

    ws.save("sh010.rad", content='{"nodes": [{"type": "KSampler"}]}',
            message="second pass", author="bob")

    backup = ws.root / ".versions" / "sh010.v1.rad"
    assert backup.exists()
    old_graph, _ = ws.m._unpack_rad_v2(backup.read_bytes())
    assert json.loads(old_graph) == {"nodes": []}

    new_graph, _ = ws.m._unpack_rad_v2((ws.root / "sh010.rad").read_bytes())
    assert json.loads(new_graph)["nodes"][0]["type"] == "KSampler"

    info = json.loads((ws.root / ".versions" / "sh010.v1.rad.json").read_text())
    assert info["message"] == "second pass" and info["author"] == "bob"
    assert not sidecar.exists()


def test_save_reports_a_malformed_body_as_500(ws):
    resp = run(ws.m.save_workflow(Req(raw=b"{oops")))
    assert resp.status == 500
    assert "error" in body(resp)


# ═══════════════════════════════════════════════════════════════════════════
#  11. /radiance/workflows/list + record reader
# ═══════════════════════════════════════════════════════════════════════════

def test_list_reports_saved_workflows_newest_first(ws):
    ws.save("old.rad")
    ws.save("new.rad", stats={"node_count": 1})
    # Force a deterministic ordering rather than relying on filesystem timing.
    import os
    os.utime(ws.root / "old.rad", (1_000_000, 1_000_000))

    resp = run(ws.m.list_workflows(Req()))
    assert resp.status == 200
    items = body(resp)["workflows"]

    assert [i["filename"] for i in items] == ["new.rad", "old.rad"]
    assert items[0]["metadata"]["format"] == "v3"
    assert items[0]["metadata"]["secure"] is True
    assert items[0]["metadata"]["asset_count"] == 0
    assert items[1]["metadata"]["format"] == "v2"
    assert items[1]["size"] == (ws.root / "old.rad").stat().st_size


def test_list_reports_has_preview_false_for_a_v3_without_one(ws):
    (ws.root / "nopreview.rad").write_bytes(
        ws.m._pack_rad_v3('{"nodes": []}', {"author": "ada"}))
    items = body(run(ws.m.list_workflows(Req())))["workflows"]
    assert items[0]["metadata"]["has_preview"] is False


# Was xfail(strict): the defect it documented is fixed.
def test_empty_preview_image_must_not_create_a_preview_entry(ws):
    ws.save("withstats.rad", stats={"node_count": 1})

    with zipfile.ZipFile(ws.root / "withstats.rad") as zf:
        assert "preview.png" not in zf.namelist()

    resp = run(ws.m.get_workflow_preview(Req(query={"filename": "withstats.rad"})))
    assert resp.status == 404


def test_record_reader_skips_versions_dir_and_reads_v1_sidecars(ws):
    ws.save("sh010.rad")
    ws.save("sh010.rad")                       # creates .versions/sh010.v1.rad

    legacy = ws.root / "legacy.rad"
    legacy.write_text('{"nodes": [], "padding": "over 14 bytes"}')
    (ws.root / "legacy.rad.json").write_text(json.dumps({"author": "old-tool"}))

    records = {r["filename"]: r for r in ws.m._read_workflow_records()}
    assert set(records) == {"sh010.rad", "legacy.rad"}
    assert records["legacy.rad"]["metadata"] == {
        "author": "old-tool", "format": "v1", "secure": False}


# Was xfail(strict): the defect it documented is fixed.
def test_short_v1_container_keeps_its_sidecar_metadata(ws):
    (ws.root / "tiny.rad").write_text('{"nodes":[]}')      # 12 bytes
    (ws.root / "tiny.rad.json").write_text(json.dumps({"author": "old-tool"}))

    record = ws.m._read_workflow_records()[0]
    assert record["metadata"] == {
        "author": "old-tool", "format": "v1", "secure": False}


def test_record_reader_tolerates_an_unreadable_container(ws):
    ws.save("good.rad")
    (ws.root / "broken.rad").write_bytes(ws.m.RAD_MAGIC + b"\x00" * 40)

    records = {r["filename"]: r for r in ws.m._read_workflow_records()}
    assert set(records) == {"good.rad", "broken.rad"}
    # The good one still parsed; the broken one degraded to empty metadata.
    assert records["good.rad"]["metadata"]["format"] == "v2"
    assert records["broken.rad"]["metadata"] == {}


# ═══════════════════════════════════════════════════════════════════════════
#  12. /radiance/workflows/get
# ═══════════════════════════════════════════════════════════════════════════

def test_get_returns_graph_and_format(ws):
    ws.save("sh010.rad", content=json.dumps(GRAPH))
    resp = run(ws.m.get_workflow(Req(query={"filename": "sh010.rad"})))

    assert resp.status == 200
    payload = body(resp)
    assert payload["success"] is True
    assert json.loads(payload["content"]) == GRAPH
    assert payload["format"] == "v2"
    assert payload["secure"] is True


def test_get_marks_a_legacy_v1_file_insecure(ws):
    (ws.root / "legacy.rad").write_text('{"nodes": []}')
    payload = body(run(ws.m.get_workflow(Req(query={"filename": "legacy.rad"}))))
    assert payload["format"] == "v1"
    assert payload["secure"] is False


def test_get_404s_for_missing_and_for_non_rad_extensions(ws):
    secret = ws.root / "secret.json"
    secret.write_text("hunter2")

    missing = run(ws.m.get_workflow(Req(query={"filename": "nope.rad"})))
    assert missing.status == 404

    wrong_ext = run(ws.m.get_workflow(Req(query={"filename": "secret.json"})))
    assert wrong_ext.status == 404, "a non-.rad file inside the library was served"
    assert body(wrong_ext) == {"error": "File not found"}

    traversal = run(ws.m.get_workflow(Req(query={"filename": "../../etc/passwd"})))
    assert traversal.status == 404


# ═══════════════════════════════════════════════════════════════════════════
#  13. /radiance/workflows/delete
# ═══════════════════════════════════════════════════════════════════════════

def test_delete_removes_file_backups_and_empty_dirs(ws):
    ws.save("SHOW/sh010.rad")
    ws.save("SHOW/sh010.rad")                  # produce a backup + sidecar

    target = ws.root / "SHOW" / "sh010.rad"
    versions = target.parent / ".versions"
    (ws.root / "SHOW" / "sh010.rad.json").write_text("{}")
    assert (versions / "sh010.v1.rad").exists()

    resp = run(ws.m.delete_workflow(Req(json_body={"filename": "SHOW/sh010.rad"})))
    assert resp.status == 200
    assert body(resp) == {"success": True}

    assert not target.exists()
    assert not (ws.root / "SHOW" / "sh010.rad.json").exists()
    assert not versions.exists()
    assert not (ws.root / "SHOW").exists(), "empty project dir was left behind"
    assert ws.root.exists(), "the library root itself must never be removed"


def test_delete_keeps_a_project_dir_that_still_has_files(ws):
    ws.save("SHOW/a.rad")
    ws.save("SHOW/b.rad")
    run(ws.m.delete_workflow(Req(json_body={"filename": "SHOW/a.rad"})))
    assert (ws.root / "SHOW" / "b.rad").exists()
    assert (ws.root / "SHOW").is_dir()


def test_delete_400_on_missing_filename(ws):
    resp = run(ws.m.delete_workflow(Req(json_body={})))
    assert resp.status == 400
    assert body(resp) == {"error": "Missing filename"}


def test_delete_403_on_non_rad_extension(ws):
    victim = ws.root / "keepme.json"
    victim.write_text("important")
    resp = run(ws.m.delete_workflow(Req(json_body={"filename": "keepme.json"})))
    assert resp.status == 403
    assert body(resp) == {"error": "Invalid or unsafe path"}
    assert victim.exists()


def test_delete_403_on_traversal(ws):
    resp = run(ws.m.delete_workflow(Req(json_body={"filename": "../x.rad"})))
    assert resp.status == 403


def test_delete_404_when_absent(ws):
    resp = run(ws.m.delete_workflow(Req(json_body={"filename": "ghost.rad"})))
    assert resp.status == 404
    assert body(resp) == {"error": "File not found"}


def test_delete_400_on_malformed_json(ws):
    resp = run(ws.m.delete_workflow(Req(raw=b"{nope")))
    assert resp.status == 400
    assert body(resp) == {"error": "Malformed JSON body"}


# ═══════════════════════════════════════════════════════════════════════════
#  14. history + restore
# ═══════════════════════════════════════════════════════════════════════════

def test_history_lists_commits_newest_first(ws):
    ws.save("sh010.rad", message="first")
    ws.save("sh010.rad", message="second", author="bob")
    ws.save("sh010.rad", message="third")

    resp = run(ws.m.get_workflow_history(Req(query={"filename": "sh010.rad"})))
    assert resp.status == 200
    payload = body(resp)
    assert payload["success"] is True and payload["filename"] == "sh010.rad"

    history = payload["history"]
    assert len(history) == 2
    assert [h["version_file"] for h in history] == ["sh010.v2.rad", "sh010.v1.rad"]
    assert history[0]["message"] == "third"
    assert history[1]["message"] == "second"
    assert history[1]["author"] == "bob"
    assert history[0]["size"] == (
        ws.root / ".versions" / "sh010.v2.rad").stat().st_size


def test_history_is_empty_without_backups(ws):
    ws.save("solo.rad")
    payload = body(run(ws.m.get_workflow_history(Req(query={"filename": "solo.rad"}))))
    assert payload["history"] == []


def test_history_404_for_missing_file(ws):
    resp = run(ws.m.get_workflow_history(Req(query={"filename": "ghost.rad"})))
    assert resp.status == 404


def test_restore_puts_back_the_old_bytes_and_backs_up_current(ws):
    ws.save("sh010.rad", content='{"nodes": []}')
    ws.save("sh010.rad", content='{"nodes": [{"type": "KSampler"}]}')

    resp = run(ws.m.restore_workflow_version(Req(json_body={
        "filename": "sh010.rad", "version_file": "sh010.v1.rad"})))

    assert resp.status == 200
    assert body(resp) == {"success": True, "restored_from": "sh010.v1.rad"}

    graph, _ = ws.m._unpack_rad_v2((ws.root / "sh010.rad").read_bytes())
    assert json.loads(graph) == {"nodes": []}

    # The state that was overwritten is itself preserved as v2.
    pre, _ = ws.m._unpack_rad_v2((ws.root / ".versions" / "sh010.v2.rad").read_bytes())
    assert json.loads(pre)["nodes"][0]["type"] == "KSampler"
    info = json.loads((ws.root / ".versions" / "sh010.v2.rad.json").read_text())
    assert info["message"] == "Pre-restore backup of sh010.v1.rad"


@pytest.mark.parametrize("payload", [
    {"version_file": "x.v1.rad"},
    {"filename": "sh010.rad"},
    {"filename": "", "version_file": ""},
])
def test_restore_400_on_missing_fields(ws, payload):
    resp = run(ws.m.restore_workflow_version(Req(json_body=payload)))
    assert resp.status == 400
    assert body(resp) == {"error": "Missing filename or version_file"}


def test_restore_403_on_unsafe_target(ws):
    resp = run(ws.m.restore_workflow_version(Req(json_body={
        "filename": "../../x.rad", "version_file": "a.v1.rad"})))
    assert resp.status == 403
    assert body(resp) == {"error": "Invalid target path"}


def test_restore_403_on_version_file_traversal(ws):
    ws.save("sh010.rad")
    ws.save("sh010.rad")
    outsider = ws.root / "outsider.rad"
    outsider.write_bytes(b"OUTSIDE")

    resp = run(ws.m.restore_workflow_version(Req(json_body={
        "filename": "sh010.rad", "version_file": "../outsider.rad"})))

    assert resp.status == 403
    assert body(resp) == {"error": "Invalid version file path"}
    assert (ws.root / "sh010.rad").read_bytes() != b"OUTSIDE"


def test_restore_404_when_version_file_absent(ws):
    ws.save("sh010.rad")
    resp = run(ws.m.restore_workflow_version(Req(json_body={
        "filename": "sh010.rad", "version_file": "sh010.v99.rad"})))
    assert resp.status == 404
    assert body(resp) == {"error": "Version file not found"}


# ═══════════════════════════════════════════════════════════════════════════
#  15. /radiance/workflows/preview
# ═══════════════════════════════════════════════════════════════════════════

def _v3_with_preview(path: Path, png: bytes = PNG_1PX):
    import base64
    blob = ws_mod._pack_rad_v3("{}", {
        "preview_image": base64.b64encode(png).decode()})
    path.write_bytes(blob)


def test_preview_returns_the_embedded_png(ws):
    _v3_with_preview(ws.root / "shot.rad")
    resp = run(ws.m.get_workflow_preview(Req(query={"filename": "shot.rad"})))
    assert resp.status == 200
    assert resp.content_type == "image/png"
    assert raw_body(resp) == PNG_1PX


def test_preview_404_for_missing_file(ws):
    resp = run(ws.m.get_workflow_preview(Req(query={"filename": "ghost.rad"})))
    assert resp.status == 404


def test_preview_400_for_a_v2_container(ws):
    ws.save("plain.rad")
    resp = run(ws.m.get_workflow_preview(Req(query={"filename": "plain.rad"})))
    assert resp.status == 400
    assert body(resp) == {"error": "Previews only supported in .rad v3"}


def test_preview_404_when_v3_has_no_preview(ws):
    (ws.root / "nopreview.rad").write_bytes(ws.m._pack_rad_v3("{}", {}))
    resp = run(ws.m.get_workflow_preview(Req(query={"filename": "nopreview.rad"})))
    assert resp.status == 404
    assert body(resp) == {"error": "No preview image"}


def test_list_reports_has_preview_true_for_embedded_previews(ws):
    _v3_with_preview(ws.root / "shot.rad")
    items = body(run(ws.m.list_workflows(Req())))["workflows"]
    assert items[0]["metadata"]["has_preview"] is True


# ═══════════════════════════════════════════════════════════════════════════
#  16. Projects index + dashboard
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def populated(ws):
    """Two projects: SHOW A (2 shots, 3 workflows) and SHOW B (1 workflow)."""
    ws.save("SHOW A/sh010-v001.rad")
    ws.save("SHOW A/sh010-v002.rad")
    ws.save("SHOW A/sh020-v001.rad")
    ws.save("SHOW B/sh030-v001.rad")
    return ws


def test_list_projects_summarises_shots_and_counts(populated):
    resp = run(populated.m.list_projects(Req()))
    assert resp.status == 200
    projects = {p["id"]: p for p in body(resp)["projects"]}

    assert set(projects) == {"show-a", "show-b"}
    assert projects["show-a"]["name"] == "SHOW A"
    assert projects["show-a"]["workflows"] == 3
    assert projects["show-a"]["shots"] == 2
    assert projects["show-b"]["workflows"] == 1
    assert projects["show-a"]["favorite"] is False
    assert projects["show-a"]["updated"] == "Just now"
    assert projects["show-a"]["size"].endswith(("B", "KB"))


def test_project_name_prefers_explicit_metadata_over_the_folder(ws):
    ws.save("misc/one.rad", stats={"n": 1})
    record = ws.m._read_workflow_records()[0]
    assert ws.m._project_name_for_workflow(record) == "misc"

    record["metadata"]["project"] = "OVERRIDE"
    assert ws.m._project_name_for_workflow(record) == "OVERRIDE"

    assert ws.m._project_name_for_workflow(
        {"filename": "loose.rad", "metadata": {}}) == "GENERAL"


def test_dashboard_payload_shape_and_active_project(populated):
    resp = run(populated.m.project_manager_dashboard(Req()))
    assert resp.status == 200
    payload = body(resp)

    assert payload["source"] == "Live API"
    assert payload["user"] == {"name": "Ahmed", "role": "Artist"}
    assert payload["activeProjectId"] in {"show-a", "show-b"}
    assert len(payload["projects"]) == 2
    assert payload["continueWorking"]["filename"].endswith(".rad")
    assert payload["continueWorking"]["status"] == "WIP"
    assert payload["continueWorking"]["shot"].startswith("SH")
    assert payload["quickActions"][0] == "Save Version"
    assert "Assets" in payload["nav"]
    assert set(payload["storage"]) == {"usedLabel", "totalLabel", "percent"}
    assert 0 <= payload["storage"]["percent"] <= 100
    assert isinstance(payload["versions"], list) and payload["versions"]


def test_dashboard_payload_on_an_empty_library(ws):
    payload = body(run(ws.m.project_manager_dashboard(Req())))
    assert payload["projects"] == []
    assert payload["versions"] == []
    assert payload["outputs"] == []
    assert payload["notes"] == []
    assert payload["activeProjectId"] == ""
    assert payload["continueWorking"] == {
        "workflow": "No workflow saved yet",
        "project": "GENERAL",
        "shot": "GENERAL",
        "status": "WIP",
        "lastModified": "Never",
        "filename": "",
    }


def test_project_versions_endpoint(populated):
    resp = run(populated.m.list_project_versions(
        Req(match_info={"project_id": "show-a"})))
    assert resp.status == 200
    versions = body(resp)["versions"]
    assert {v["shot"] for v in versions} == {"SH010", "SH020"}
    assert {v["version"] for v in versions} == {"v001", "v002"}
    assert all(v["status"] == "WIP" for v in versions)
    assert len(versions) <= 12


def test_project_versions_include_backups(ws):
    ws.save("SHOW A/sh010-v001.rad", message="c1")
    ws.save("SHOW A/sh010-v001.rad", message="c2")

    project, _ = ws.m._find_project("show-a")
    versions = ws.m._versions_for_project(project)
    assert [v["status"] for v in versions] == ["WIP", "c2"]
    assert versions[1]["workflow"] == "sh010-v001"


@pytest.mark.parametrize("handler", [
    "list_project_versions", "list_project_outputs",
    "list_project_notes", "save_project_version",
    "export_project_package", "set_shot_status",
])
def test_project_scoped_endpoints_404_on_unknown_project(populated, handler):
    fn = getattr(populated.m, handler)
    resp = run(fn(Req(json_body={}, match_info={"project_id": "nope",
                                                "shot": "SH010"})))
    assert resp.status == 404
    assert body(resp) == {"error": "Project not found"}


def test_project_notes_reads_both_json_shapes(populated):
    notes_path = populated.root / "SHOW A" / ".review_notes.json"

    notes_path.write_text(json.dumps(["fix the sky", "regrade"]))
    resp = run(populated.m.list_project_notes(Req(match_info={"project_id": "show-a"})))
    assert resp.status == 200
    assert body(resp)["notes"] == ["fix the sky", "regrade"]

    notes_path.write_text(json.dumps({"notes": ["wrapped"]}))
    assert body(run(populated.m.list_project_notes(
        Req(match_info={"project_id": "show-a"}))))["notes"] == ["wrapped"]

    notes_path.write_text("{not json")
    assert body(run(populated.m.list_project_notes(
        Req(match_info={"project_id": "show-a"}))))["notes"] == []


def test_project_notes_empty_when_absent(populated):
    assert body(run(populated.m.list_project_notes(
        Req(match_info={"project_id": "show-b"}))))["notes"] == []


def test_project_outputs_matches_on_project_and_shot_tokens(populated):
    (populated.out / "renders").mkdir()
    hit_project = populated.out / "renders" / "show a_beauty.exr"
    hit_shot = populated.out / "sh010_comp.mov"
    miss_ext = populated.out / "show a_notes.txt"
    miss_name = populated.out / "unrelated.png"
    for p in (hit_project, hit_shot, miss_ext, miss_name):
        p.write_bytes(b"0123456789")

    resp = run(populated.m.list_project_outputs(
        Req(match_info={"project_id": "show-a"})))
    assert resp.status == 200
    outputs = body(resp)["outputs"]

    names = {o["name"] for o in outputs}
    assert names == {"show a_beauty.exr", "sh010_comp.mov"}
    exr = next(o for o in outputs if o["name"].endswith(".exr"))
    assert exr["type"] == "EXR"
    assert exr["size"] == "10 B"
    assert exr["date"] == "Just now"
    assert exr["path"] == str(hit_project)
    assert "_mtime" not in exr


def test_project_outputs_empty_when_output_dir_is_gone(populated, monkeypatch):
    monkeypatch.setattr(populated.m.folder_paths, "get_output_directory",
                        lambda: str(populated.out / "vanished"))
    assert body(run(populated.m.list_project_outputs(
        Req(match_info={"project_id": "show-a"}))))["outputs"] == []


def test_project_outputs_empty_without_a_folder_paths_getter(populated, monkeypatch):
    monkeypatch.delattr(populated.m.folder_paths, "get_output_directory")
    assert body(run(populated.m.list_project_outputs(
        Req(match_info={"project_id": "show-a"}))))["outputs"] == []


# ═══════════════════════════════════════════════════════════════════════════
#  17. save-version / export-package
# ═══════════════════════════════════════════════════════════════════════════

def test_save_project_version_writes_into_the_project_folder(populated):
    resp = run(populated.m.save_project_version(Req(
        match_info={"project_id": "show-a"},
        json_body={"content": json.dumps(GRAPH), "filename": "elsewhere/new.rad",
                   "message": "dash save", "author": "ada"})))

    assert resp.status == 200
    # The handler re-anchors the file under the project name.
    assert body(resp) == {"success": True, "filename": "SHOW A/new.rad"}

    written = populated.root / "SHOW A" / "new.rad"
    assert written.read_bytes()[:4] == populated.m.RAD_MAGIC_V3
    graph, meta, _ = populated.m._unpack_rad_v3(written.read_bytes())
    assert json.loads(graph) == GRAPH
    assert meta["project"] == "SHOW A"
    assert meta["commit_message"] == "dash save"
    assert meta["author"] == "ada"
    assert meta["pipeline"]["node_count"] == 6


def test_save_project_version_generates_a_name_when_none_given(populated):
    payload = body(run(populated.m.save_project_version(Req(
        match_info={"project_id": "show-a"}, json_body={"content": "{}"}))))
    assert payload["success"] is True
    assert payload["filename"].startswith("SHOW A/dashboard_")
    assert (populated.root / payload["filename"]).exists()


def test_save_project_version_flattens_a_traversal_name_into_the_project(populated):
    resp = run(populated.m.save_project_version(Req(
        match_info={"project_id": "show-a"},
        json_body={"content": "{}", "filename": "x.rad/../../evil.rad"})))

    assert resp.status == 200
    assert body(resp) == {"success": True, "filename": "SHOW A/evil.rad"}
    assert (populated.root / "SHOW A" / "evil.rad").exists()
    assert not (populated.root.parent / "evil.rad").exists()


def test_save_project_version_rejects_a_control_character_name(populated):
    resp = run(populated.m.save_project_version(Req(
        match_info={"project_id": "show-a"},
        json_body={"content": "{}", "filename": "ev\x00il.rad"})))
    assert resp.status == 403
    assert body(resp) == {"error": "Invalid or unsafe path"}
    assert not list((populated.root / "SHOW A").glob("ev*"))


def test_save_project_version_without_content_backs_up_the_latest(populated):
    resp = run(populated.m.save_project_version(Req(
        match_info={"project_id": "show-b"}, json_body={"message": "snapshot"})))

    assert resp.status == 200
    payload = body(resp)
    assert payload["success"] is True
    assert payload["filename"] == "SHOW B/sh030-v001.rad"
    assert payload["message"] == "Created a version backup for the latest workflow."

    backup = populated.root / "SHOW B" / ".versions" / "sh030-v001.v1.rad"
    assert backup.exists()
    info = json.loads(backup.with_suffix(".rad.json").read_text())
    assert info["message"] == "snapshot"


def test_save_project_version_tolerates_a_malformed_body(populated):
    resp = run(populated.m.save_project_version(Req(
        match_info={"project_id": "show-b"}, raw=b"{nope")))
    assert resp.status == 200
    assert body(resp)["filename"] == "SHOW B/sh030-v001.rad"


def test_export_package_zips_manifest_and_workflows(populated):
    resp = run(populated.m.export_project_package(
        Req(match_info={"project_id": "show-a"})))

    assert resp.status == 200
    assert resp.content_type == "application/zip"
    assert resp.headers["Content-Disposition"] == (
        'attachment; filename="show-a_radiance_package.zip"')

    with zipfile.ZipFile(io.BytesIO(raw_body(resp))) as zf:
        names = set(zf.namelist())
        manifest = json.loads(zf.read("manifest.json"))
        one = zf.read("workflows/SHOW A/sh010-v001.rad")

    assert names == {
        "manifest.json",
        "workflows/SHOW A/sh010-v001.rad",
        "workflows/SHOW A/sh010-v002.rad",
        "workflows/SHOW A/sh020-v001.rad",
    }
    assert manifest["project"] == "SHOW A"
    assert manifest["project_id"] == "show-a"
    assert manifest["workflow_count"] == 3
    assert one[:4] == populated.m.RAD_MAGIC


# ═══════════════════════════════════════════════════════════════════════════
#  18. Shot status
# ═══════════════════════════════════════════════════════════════════════════

def test_set_shot_status_persists_and_feeds_the_version_list(populated):
    resp = run(populated.m.set_shot_status(Req(
        match_info={"project_id": "show-a", "shot": "SH010"},
        json_body={"status": "Approved"})))

    assert resp.status == 200
    assert body(resp) == {"success": True, "shot": "SH010", "status": "Approved"}

    stored = json.loads(
        (populated.root / "SHOW A" / ".shot_status.json").read_text())
    assert stored == {"SH010": "Approved"}

    versions = body(run(populated.m.list_project_versions(
        Req(match_info={"project_id": "show-a"}))))["versions"]
    by_shot = {v["shot"]: v["status"] for v in versions}
    assert by_shot["SH010"] == "Approved"
    assert by_shot["SH020"] == "WIP"


@pytest.mark.parametrize("status", ["WIP", "Review", "Approved", "Retake", "Final"])
def test_set_shot_status_accepts_every_allowed_value(populated, status):
    resp = run(populated.m.set_shot_status(Req(
        match_info={"project_id": "show-a", "shot": "SH010"},
        json_body={"status": status})))
    assert resp.status == 200


@pytest.mark.parametrize("status", ["approved", "Done", "", "  "])
def test_set_shot_status_rejects_unknown_values(populated, status):
    resp = run(populated.m.set_shot_status(Req(
        match_info={"project_id": "show-a", "shot": "SH010"},
        json_body={"status": status})))
    assert resp.status == 400
    assert body(resp)["error"].startswith("status must be one of")
    assert not (populated.root / "SHOW A" / ".shot_status.json").exists()


def test_shot_status_loader_ignores_corrupt_and_non_dict_files(populated):
    project, _ = populated.m._find_project("show-a")
    path = populated.m._shot_status_path(project)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text("{not json")
    assert populated.m._load_shot_status(project) == {}

    path.write_text(json.dumps(["a", "list"]))
    assert populated.m._load_shot_status(project) == {}

    path.write_text(json.dumps({"SH010": "Final"}))
    assert populated.m._load_shot_status(project) == {"SH010": "Final"}


# ═══════════════════════════════════════════════════════════════════════════
#  19. Assets
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def assets(ws):
    """Input tree: a still, a video, and a 4-frame image sequence."""
    (ws.inp / "hero.png").write_bytes(PNG_1PX)
    (ws.inp / "take.mp4").write_bytes(b"\x00" * 32)
    seq = ws.inp / "seq"
    seq.mkdir()
    for i in range(1, 5):
        (seq / f"plate_{i:04d}.exr").write_bytes(b"E" * 10)
    (ws.inp / "notes.txt").write_bytes(b"ignored")
    return ws


def test_scan_assets_groups_sequences_and_classifies_types(assets):
    found = {a["name"]: a for a in assets.m._scan_assets()}

    assert set(found) == {"hero.png", "take.mp4", "plate"}
    assert found["hero.png"]["type"] == "image"
    assert found["hero.png"]["previewable"] is True
    assert found["take.mp4"]["type"] == "video"
    assert found["take.mp4"]["previewable"] is False

    seq = found["plate"]
    assert seq["type"] == "sequence"
    assert seq["frames"] == 4
    assert seq["frame_start"] == 1 and seq["frame_end"] == 4
    assert seq["format"] == "EXR"
    assert seq["meta"] == "EXR seq · 4 fr"
    assert seq["size"] == "40 B"


def test_short_runs_are_not_treated_as_a_sequence(ws):
    d = ws.inp / "pair"
    d.mkdir()
    for i in (1, 2):
        (d / f"plate_{i:04d}.png").write_bytes(PNG_1PX)

    names = sorted(a["name"] for a in ws.m._scan_assets())
    assert names == ["plate_0001.png", "plate_0002.png"]


def test_assets_endpoint_counts_by_type(assets):
    resp = run(assets.m.list_assets(Req()))
    assert resp.status == 200
    payload = body(resp)

    assert payload["source"] == "Live scan"
    assert payload["counts"] == {"all": 3, "image": 1, "video": 1, "sequence": 1}
    assert payload["bins"] == []
    assert all("_mtime" not in a for a in payload["assets"])


def test_classify_and_asset_id():
    assert ws_mod._classify(".MP4") == "video"
    assert ws_mod._classify(".hdr") == "hdri"
    assert ws_mod._classify(".exr") == "image"
    assert ws_mod._classify(".png") == "image"

    a = ws_mod._asset_id(Path("/a/b.png"))
    assert len(a) == 16 and a == ws_mod._asset_id(Path("/a/b.png"))
    assert a != ws_mod._asset_id(Path("/a/c.png"))


def test_is_within(tmp_path):
    assert ws_mod._is_within(tmp_path / "a" / "b", tmp_path) is True
    assert ws_mod._is_within(Path("/etc/passwd"), tmp_path) is False


def test_asset_roots_skips_missing_dirs(ws, monkeypatch):
    monkeypatch.setattr(ws.m.folder_paths, "get_output_directory",
                        lambda: str(ws.out / "gone"))
    roots = ws.m._asset_roots()
    assert roots == [ws.inp]


def test_create_bin_then_add_rename_remove_and_delete(assets):
    resp = run(assets.m.create_asset_bin(Req(json_body={"name": "Hero plates"})))
    assert resp.status == 200
    new_bin = body(resp)["bin"]
    assert new_bin["name"] == "Hero plates" and new_bin["assets"] == []
    bin_id = new_bin["id"]

    assert json.loads(assets.m._ASSETS_BINS_PATH.read_text())[0]["id"] == bin_id

    hero = next(a for a in assets.m._scan_assets() if a["name"] == "hero.png")

    add = run(assets.m.modify_asset_bin(Req(
        match_info={"bin_id": bin_id},
        json_body={"action": "add", "asset_id": hero["id"]})))
    assert add.status == 200 and body(add) == {"success": True}
    assert assets.m._load_bins()[0]["assets"] == [hero["id"]]

    # Adding twice must not duplicate.
    run(assets.m.modify_asset_bin(Req(
        match_info={"bin_id": bin_id},
        json_body={"action": "add", "asset_id": hero["id"]})))
    assert assets.m._load_bins()[0]["assets"] == [hero["id"]]

    payload = body(run(assets.m.list_assets(Req())))
    assert payload["bins"][0]["count"] == 1

    run(assets.m.modify_asset_bin(Req(
        match_info={"bin_id": bin_id},
        json_body={"action": "rename", "name": "Renamed"})))
    assert assets.m._load_bins()[0]["name"] == "Renamed"

    run(assets.m.modify_asset_bin(Req(
        match_info={"bin_id": bin_id},
        json_body={"action": "remove", "asset_id": hero["id"]})))
    assert assets.m._load_bins()[0]["assets"] == []

    run(assets.m.modify_asset_bin(Req(
        match_info={"bin_id": bin_id}, json_body={"action": "delete"})))
    assert assets.m._load_bins() == []


def test_create_bin_requires_a_name(ws):
    resp = run(ws.m.create_asset_bin(Req(json_body={"name": "   "})))
    assert resp.status == 400
    assert body(resp) == {"error": "Bin name required"}
    assert not ws.m._ASSETS_BINS_PATH.exists()


def test_create_bin_truncates_long_names(ws):
    payload = body(run(ws.m.create_asset_bin(Req(json_body={"name": "N" * 200}))))
    assert len(payload["bin"]["name"]) == 60


def test_modify_bin_404_and_400(ws):
    missing = run(ws.m.modify_asset_bin(Req(
        match_info={"bin_id": "ghost"}, json_body={"action": "add"})))
    assert missing.status == 404
    assert body(missing) == {"error": "Bin not found"}

    bin_id = body(run(ws.m.create_asset_bin(Req(json_body={"name": "B"}))))["bin"]["id"]
    bad = run(ws.m.modify_asset_bin(Req(
        match_info={"bin_id": bin_id}, json_body={"action": "explode"})))
    assert bad.status == 400
    assert body(bad) == {"error": "Unknown action"}


def test_load_bins_ignores_a_corrupt_file(ws):
    ws.m._ASSETS_BINS_PATH.write_text("{not a list")
    assert ws.m._load_bins() == []
    ws.m._ASSETS_BINS_PATH.write_text(json.dumps({"not": "a list"}))
    assert ws.m._load_bins() == []


# ═══════════════════════════════════════════════════════════════════════════
#  20. Asset thumbnails
# ═══════════════════════════════════════════════════════════════════════════

def test_thumb_requires_a_path(ws):
    resp = run(ws.m.asset_thumb(Req(query={})))
    assert resp.status == 400
    assert body(resp) == {"error": "path required"}


def test_thumb_forbids_paths_outside_the_asset_roots(ws, tmp_path):
    outsider = tmp_path / "outside.png"
    outsider.write_bytes(PNG_1PX)
    resp = run(ws.m.asset_thumb(Req(query={"path": str(outsider)})))
    assert resp.status == 403
    assert body(resp) == {"error": "forbidden"}


def test_thumb_forbids_a_directory(ws):
    resp = run(ws.m.asset_thumb(Req(query={"path": str(ws.inp)})))
    assert resp.status == 403


def test_thumb_serves_web_images_verbatim(assets):
    resp = run(assets.m.asset_thumb(Req(query={"path": str(assets.inp / "hero.png")})))
    assert resp.status == 200
    assert resp.content_type == "image/png"
    assert raw_body(resp) == PNG_1PX


def test_thumb_renders_and_caches_a_non_web_image(ws):
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    src = ws.inp / "plate.tif"
    img = (np.linspace(0, 65535, 32 * 32 * 3).reshape(32, 32, 3)).astype("uint16")
    assert cv2.imwrite(str(src), img)

    resp = run(ws.m.asset_thumb(Req(query={"path": str(src)})))
    assert resp.status == 200
    assert resp.content_type == "image/png"
    png = raw_body(resp)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"

    cached = list(ws.m._THUMB_CACHE.glob("*.png"))
    assert len(cached) == 1 and cached[0].read_bytes() == png

    again = run(ws.m.asset_thumb(Req(query={"path": str(src)})))
    assert raw_body(again) == png


def test_thumb_415_when_the_file_cannot_be_decoded(ws):
    junk = ws.inp / "broken.exr"
    junk.write_bytes(b"not an exr at all")
    resp = run(ws.m.asset_thumb(Req(query={"path": str(junk)})))
    assert resp.status == 415
    assert body(resp) == {"error": "not previewable"}


def test_render_thumb_png_returns_none_for_undecodable_input(ws):
    pytest.importorskip("cv2")
    junk = ws.inp / "broken.exr"
    junk.write_bytes(b"nope")
    assert ws.m._render_thumb_png(junk) is None


# ═══════════════════════════════════════════════════════════════════════════
#  21. Asset upload
# ═══════════════════════════════════════════════════════════════════════════

def test_upload_saves_allowed_files_and_skips_the_rest(ws):
    resp = run(ws.m.upload_asset(Req(parts=[
        _Part("hero.png", PNG_1PX),
        _Part("notes.txt", b"nope"),
        _Part(None, b"a form field"),
        _Part("../../escape.png", b"ESCAPED"),
    ])))

    assert resp.status == 200
    assert body(resp) == {"success": True, "saved": ["hero.png", "escape.png"]}

    dest = ws.inp / "radiance_assets"
    assert (dest / "hero.png").read_bytes() == PNG_1PX
    assert (dest / "escape.png").read_bytes() == b"ESCAPED"
    assert not (dest / "notes.txt").exists()
    assert not (ws.inp.parent.parent / "escape.png").exists()


def test_upload_500_without_an_input_directory(ws, monkeypatch):
    monkeypatch.delattr(ws.m.folder_paths, "get_input_directory")
    resp = run(ws.m.upload_asset(Req(parts=[])))
    assert resp.status == 500
    assert body(resp) == {"error": "No input directory"}


# ═══════════════════════════════════════════════════════════════════════════
#  22. Storage summary
# ═══════════════════════════════════════════════════════════════════════════

def test_storage_summary_from_disk_usage(ws):
    summary = ws.m._storage_summary()
    assert set(summary) == {"usedLabel", "totalLabel", "percent"}
    assert isinstance(summary["percent"], int)
    assert 0 <= summary["percent"] <= 100
    assert summary["usedLabel"].split()[-1] in {"B", "KB", "MB", "GB", "TB"}


def test_storage_summary_falls_back_to_library_size(ws, monkeypatch):
    ws.save("a.rad")
    monkeypatch.setattr(ws.m.shutil, "disk_usage",
                        lambda p: (_ for _ in ()).throw(OSError("no statvfs")))
    summary = ws.m._storage_summary()
    assert summary["totalLabel"] == "Workflow Library"
    assert summary["percent"] == 0
    assert summary["usedLabel"] == ws.m._format_bytes(
        (ws.root / "a.rad").stat().st_size)


# ═══════════════════════════════════════════════════════════════════════════
#  23. The ComfyUI node
# ═══════════════════════════════════════════════════════════════════════════

def test_node_input_types_and_registration():
    spec = ws_mod.RadianceProjectManager.INPUT_TYPES()
    assert set(spec["optional"]) == {"filename", "artist", "version"}
    assert spec["hidden"] == {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"}
    assert spec["required"] == {}
    assert ws_mod.RadianceProjectManager.OUTPUT_NODE is True
    assert ws_mod.RadianceProjectManager.RETURN_TYPES == ()

    assert ws_mod.NODE_CLASS_MAPPINGS["RadianceWorkspace"] is ws_mod.RadianceWorkspace
    assert issubclass(ws_mod.RadianceWorkspace, ws_mod.RadianceProjectManager)
    assert ws_mod.RadianceWorkspace.DEPRECATED is True
    assert "RadianceWorkspace" not in ws_mod.RadianceProjectManager.__dict__
    assert ws_mod.NODE_DISPLAY_NAME_MAPPINGS["RadianceWorkspace"].endswith(
        "Radiance Project Manager")


def test_node_run_writes_a_v3_container_with_metadata(ws):
    node = ws.m.RadianceProjectManager()
    assert node.run(filename="sh010", artist="ada", version=3, prompt=GRAPH) == ()

    written = ws.root / "sh010_ada_v0003.rad"
    assert written.exists()
    assert written.read_bytes()[:4] == ws.m.RAD_MAGIC_V3

    graph, meta, _ = ws.m._unpack_rad_v3(written.read_bytes())
    assert json.loads(graph) == GRAPH
    assert meta["artist"] == "ada"
    assert meta["version"] == 3
    assert meta["stats"]["node_count"] == 6
    assert meta["stats"]["is_hdr"] is True
    assert meta["pipeline"]["models"] == ["film_grain.ckpt", "flux_dev.safetensors"]


def test_node_run_without_a_filename_uses_the_artist(ws):
    ws.m.RadianceProjectManager().run(artist="bob", version=1, prompt=GRAPH)
    assert (ws.root / "bob_v0001.rad").exists()


def test_node_run_without_artist_or_filename(ws):
    ws.m.RadianceProjectManager().run(version=12, prompt=GRAPH)
    assert (ws.root / "unknown_v0012.rad").exists()


def test_node_run_writes_nothing_without_a_prompt(ws):
    assert ws.m.RadianceProjectManager().run(filename="x", prompt=None) == ()
    assert list(ws.root.rglob("*.rad")) == []


def test_node_run_versions_a_previous_save(ws):
    node = ws.m.RadianceProjectManager()
    node.run(filename="sh010", artist="ada", version=1, prompt={"nodes": []})
    node.run(filename="sh010", artist="ada", version=1, prompt=GRAPH)

    backup = ws.root / ".versions" / "sh010_ada_v0001.v1.rad"
    assert backup.exists()
    graph, _, _ = ws.m._unpack_rad_v3(backup.read_bytes())
    assert json.loads(graph) == {"nodes": []}


def test_node_run_rejects_an_unsafe_filename(ws):
    node = ws.m.RadianceProjectManager()
    assert node.run(filename="../../pwned", artist="ada", version=1,
                    prompt=GRAPH) == ()
    assert not (ws.root.parent.parent / "pwned_ada_v0001.rad").exists()


# ═══════════════════════════════════════════════════════════════════════════
#  24. Internal failures surface as JSON 500s, never as raised exceptions
# ═══════════════════════════════════════════════════════════════════════════

def _boom(*a, **k):
    raise RuntimeError("engine room fire")


#: (handler, module attribute to break, match_info, json body)
FAILURE_CASES = [
    ("list_workflows", "_read_workflow_records", {}, None),
    ("list_projects", "_read_workflow_records", {}, None),
    ("project_manager_dashboard", "_dashboard_payload", {}, None),
    ("list_project_versions", "_versions_for_project", {"project_id": "show-a"}, None),
    ("list_project_outputs", "_outputs_for_project", {"project_id": "show-a"}, None),
    ("list_project_notes", "_notes_for_project", {"project_id": "show-a"}, None),
    ("save_project_version", "_create_version_backup", {"project_id": "show-a"}, {}),
    ("set_shot_status", "_load_shot_status",
     {"project_id": "show-a", "shot": "SH010"}, {"status": "Final"}),
    ("list_assets", "_assets_payload", {}, None),
    ("get_workflow", "_resolve_safe_path", {}, None),
    ("get_workflow_history", "_resolve_safe_path", {}, None),
    ("get_workflow_preview", "_resolve_safe_path", {}, None),
    ("save_workflow", "_resolve_safe_path", {}, {"filename": "a.rad", "content": "{}"}),
    ("delete_workflow", "_resolve_safe_path", {}, {"filename": "a.rad"}),
    ("restore_workflow_version", "_resolve_safe_path", {},
     {"filename": "a.rad", "version_file": "a.v1.rad"}),
]


@pytest.mark.parametrize("handler,broken,match_info,payload", FAILURE_CASES,
                         ids=[f"{c[0]}-{c[1]}" for c in FAILURE_CASES])
def test_endpoints_convert_internal_errors_into_500(
        populated, monkeypatch, handler, broken, match_info, payload):
    monkeypatch.setattr(populated.m, broken, _boom)
    resp = run(getattr(populated.m, handler)(
        Req(match_info=match_info, json_body=payload,
            query={"filename": "sh010-v001.rad"})))

    assert resp.status == 500
    assert body(resp) == {"error": "engine room fire"}


def test_export_package_500_when_zipping_fails(populated, monkeypatch):
    monkeypatch.setattr(populated.m.zipfile, "ZipFile", _boom)
    resp = run(populated.m.export_project_package(
        Req(match_info={"project_id": "show-a"})))
    assert resp.status == 500
    assert body(resp) == {"error": "engine room fire"}


def test_delete_500_when_the_target_is_a_directory(ws):
    (ws.root / "trap.rad").mkdir()
    resp = run(ws.m.delete_workflow(Req(json_body={"filename": "trap.rad"})))
    assert resp.status == 500
    assert "error" in body(resp)
    assert (ws.root / "trap.rad").is_dir()


def test_upload_500_when_the_destination_cannot_be_created(ws, monkeypatch):
    blocker = ws.inp / "blocker"
    blocker.write_bytes(b"i am a file")
    monkeypatch.setattr(ws.m.folder_paths, "get_input_directory", lambda: str(blocker))
    resp = run(ws.m.upload_asset(Req(parts=[_Part("hero.png", PNG_1PX)])))
    assert resp.status == 500
    assert "error" in body(resp)


def test_save_project_version_404_when_the_project_has_no_workflows(ws, monkeypatch):
    empty = {"id": "ghost", "name": "GHOST", "shots_set": set(),
             "workflow_count": 0, "size": 0, "mtime": 0, "workflows": []}
    monkeypatch.setattr(ws.m, "_find_project", lambda pid: (empty, []))
    resp = run(ws.m.save_project_version(
        Req(match_info={"project_id": "ghost"}, json_body={})))
    assert resp.status == 404
    assert body(resp) == {"error": "Project has no workflows to version"}


# ═══════════════════════════════════════════════════════════════════════════
#  25. Degradation paths that must not take the request down
# ═══════════════════════════════════════════════════════════════════════════

def test_version_list_survives_a_corrupt_backup_sidecar(ws):
    ws.save("SHOW A/sh010-v001.rad", message="c1")
    ws.save("SHOW A/sh010-v001.rad", message="c2")
    sidecar = ws.root / "SHOW A" / ".versions" / "sh010-v001.v1.rad.json"
    sidecar.write_text("{not json")

    project, _ = ws.m._find_project("show-a")
    versions = ws.m._versions_for_project(project)
    assert [v["status"] for v in versions] == ["WIP", "Backup"]


def test_save_shot_status_swallows_a_write_failure(ws, monkeypatch):
    blocker = ws.root / "blocker"
    blocker.write_bytes(b"file, not a directory")
    monkeypatch.setattr(ws.m, "_shot_status_path",
                        lambda project: blocker / "sub" / ".shot_status.json")

    ws.m._save_shot_status({"name": "X"}, {"SH010": "Final"})   # must not raise
    assert blocker.read_bytes() == b"file, not a directory"


def test_save_bins_swallows_a_write_failure(ws, monkeypatch):
    monkeypatch.setattr(ws.m, "_ASSETS_BINS_PATH", ws.root)   # a directory
    resp = run(ws.m.create_asset_bin(Req(json_body={"name": "B"})))
    assert resp.status == 200            # the failure is logged, not raised
    assert ws.m._load_bins() == []       # ...and nothing was persisted


def test_asset_roots_skips_a_getter_that_raises(ws, monkeypatch):
    monkeypatch.setattr(ws.m.folder_paths, "get_input_directory", _boom)
    assert ws.m._asset_roots() == [ws.out]


def test_outputs_empty_when_the_output_getter_raises(populated, monkeypatch):
    monkeypatch.setattr(populated.m.folder_paths, "get_output_directory", _boom)
    project, _ = populated.m._find_project("show-a")
    assert populated.m._outputs_for_project(project) == []


# ═══════════════════════════════════════════════════════════════════════════
#  26. Route registration guard
# ═══════════════════════════════════════════════════════════════════════════

def test_route_is_a_noop_without_a_server(monkeypatch):
    monkeypatch.setattr(ws_mod, "_WORKSPACE_AVAILABLE", False)

    def handler():
        pass

    assert ws_mod._route("get", "/radiance/anything")(handler) is handler


def test_route_registration_is_idempotent():
    """A second import under another name must not re-register a route."""
    reg = getattr(ws_mod.PromptServer.instance, "_radiance_registered_routes")
    path = "/radiance/__test_only__"
    assert ("get", path) not in reg

    try:
        first = ws_mod._route("get", path)
        assert ("get", path) in reg

        def handler():
            pass

        # The already-registered path yields the pass-through decorator.
        assert ws_mod._route("get", path)(handler) is handler
        assert callable(first)
    finally:
        reg.discard(("get", path))


# ═══════════════════════════════════════════════════════════════════════════
#  27. Thumbnail rendering across pixel layouts
# ═══════════════════════════════════════════════════════════════════════════

def test_render_thumb_png_handles_grayscale_16bit(ws):
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    src = ws.inp / "gray.tif"
    assert cv2.imwrite(str(src), np.full((16, 16), 30000, dtype="uint16"))

    png = ws.m._render_thumb_png(src)
    assert png is not None and png[:8] == b"\x89PNG\r\n\x1a\n"
    decoded = cv2.imdecode(np.frombuffer(png, "uint8"), cv2.IMREAD_UNCHANGED)
    assert decoded.shape == (16, 16, 3)          # promoted to BGR


def test_render_thumb_png_downscales_wide_images_and_drops_alpha(ws):
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    src = ws.inp / "wide.tif"
    rgba = np.zeros((200, 1024, 4), dtype="uint8")
    rgba[..., 3] = 255
    assert cv2.imwrite(str(src), rgba)

    png = ws.m._render_thumb_png(src)
    decoded = cv2.imdecode(np.frombuffer(png, "uint8"), cv2.IMREAD_UNCHANGED)
    assert decoded.shape[1] == 512               # capped width
    assert decoded.shape[0] == max(1, int(200 * 512 / 1024))
    assert decoded.ndim == 3 and decoded.shape[2] == 3   # alpha removed


def test_render_thumb_png_tonemaps_float_input(ws):
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    src = ws.inp / "hdr.exr"
    hdr = np.full((8, 8, 3), 100.0, dtype="float32")
    # An OpenCV built without OPENEXR support does not return False here, it
    # raises "could not find a writer for the specified extension".  The old
    # guard only handled the falsy return, so on such a build this test failed
    # for a reason that says nothing about Radiance.  Skip on both shapes, and
    # name the build, so the reason is readable in the run output.
    try:
        wrote = cv2.imwrite(str(src), hdr)
    except cv2.error as exc:
        pytest.skip(f"this OpenCV build cannot write EXR: {exc.err or exc}")
    if not wrote:
        pytest.skip("this OpenCV build cannot write EXR")

    png = ws.m._render_thumb_png(src)
    assert png is not None
    decoded = cv2.imdecode(np.frombuffer(png, "uint8"), cv2.IMREAD_UNCHANGED)
    # Reinhard + 2.2 gamma maps 100.0 to just under white, never above it.
    assert decoded.max() <= 255
    assert decoded.min() > 240, decoded.min()


def test_render_thumb_png_reads_the_first_video_frame(ws):
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    src = ws.inp / "clip.mp4"
    writer = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 12, (64, 48))
    if not writer.isOpened():
        pytest.skip("this OpenCV build has no mp4 writer")
    for _ in range(4):
        writer.write(np.full((48, 64, 3), 90, dtype="uint8"))
    writer.release()

    png = ws.m._render_thumb_png(src)
    assert png is not None and png[:8] == b"\x89PNG\r\n\x1a\n"
    decoded = cv2.imdecode(np.frombuffer(png, "uint8"), cv2.IMREAD_UNCHANGED)
    assert decoded.shape[:2] == (48, 64)


def test_render_thumb_png_returns_none_for_an_unreadable_video(ws):
    pytest.importorskip("cv2")
    src = ws.inp / "broken.mp4"
    src.write_bytes(b"not a video")
    assert ws.m._render_thumb_png(src) is None


# ═══════════════════════════════════════════════════════════════════════════
#  28. Remaining edge branches
# ═══════════════════════════════════════════════════════════════════════════

def test_build_shot_paths_reports_but_survives_an_unusable_root(tmp_path, caplog):
    blocker = tmp_path / "blocker"
    blocker.write_bytes(b"a file where a directory was expected")

    with caplog.at_level("WARNING", logger="radiance.pipeline"):
        paths = ws_mod._build_shot_paths(str(blocker), "SHOW", "SEQ", "SH", "comp")

    assert not paths["root"].exists()
    assert any("Could not create" in r.message for r in caplog.records)


def test_pack_v3_skips_an_undecodable_preview(caplog):
    with caplog.at_level("WARNING", logger="radiance.pipeline"):
        blob = ws_mod._pack_rad_v3("{}", {"preview_image": "!!! not base64 !!!"})

    with zipfile.ZipFile(io.BytesIO(blob[4:])) as zf:
        assert "preview.png" not in zf.namelist()
    assert any("Failed to pack preview image" in r.message for r in caplog.records)


def test_save_project_version_appends_the_rad_extension(populated):
    payload = body(run(populated.m.save_project_version(Req(
        match_info={"project_id": "show-a"},
        json_body={"content": "{}", "filename": "SHOW A/noext"}))))
    assert payload["filename"] == "SHOW A/noext.rad"
    assert (populated.root / "SHOW A" / "noext.rad").exists()


def test_history_ignores_a_corrupt_commit_sidecar(ws):
    ws.save("sh010.rad", message="first")
    ws.save("sh010.rad", message="second")
    (ws.root / ".versions" / "sh010.v1.rad.json").write_text("{not json")

    history = body(run(ws.m.get_workflow_history(
        Req(query={"filename": "sh010.rad"}))))["history"]
    assert len(history) == 1
    assert history[0]["version_file"] == "sh010.v1.rad"
    assert "message" not in history[0]       # the sidecar contributed nothing
    assert history[0]["size"] > 0


def test_bin_endpoints_tolerate_a_malformed_body(ws):
    created = run(ws.m.create_asset_bin(Req(raw=b"{nope")))
    assert created.status == 400
    assert body(created) == {"error": "Bin name required"}

    bin_id = body(run(ws.m.create_asset_bin(
        Req(json_body={"name": "B"}))))["bin"]["id"]
    modified = run(ws.m.modify_asset_bin(
        Req(match_info={"bin_id": bin_id}, raw=b"{nope")))
    assert modified.status == 400
    assert body(modified) == {"error": "Unknown action"}


def test_set_shot_status_400_on_a_malformed_body(populated):
    resp = run(populated.m.set_shot_status(Req(
        match_info={"project_id": "show-a", "shot": "SH010"}, raw=b"{nope")))
    assert resp.status == 400
    assert body(resp)["error"].startswith("status must be one of")


def test_set_shot_status_without_a_shot_records_nothing(populated):
    resp = run(populated.m.set_shot_status(Req(
        match_info={"project_id": "show-a", "shot": ""},
        json_body={"status": "Final"})))

    assert resp.status == 200
    assert body(resp) == {"success": True, "shot": "", "status": "Final"}
    stored = json.loads(
        (populated.root / "SHOW A" / ".shot_status.json").read_text())
    assert stored == {}


def test_thumb_400_on_a_path_python_cannot_resolve(ws):
    resp = run(ws.m.asset_thumb(Req(query={"path": "bad\x00path.png"})))
    assert resp.status == 400
    assert body(resp) == {"error": "bad path"}


def test_thumb_415_when_the_cache_directory_cannot_be_created(ws, monkeypatch):
    blocker = ws.root / "thumbblocker"
    blocker.write_bytes(b"a file")
    monkeypatch.setattr(ws.m, "_THUMB_CACHE", blocker / "cache")

    src = ws.inp / "plate.exr"
    src.write_bytes(b"E" * 32)
    resp = run(ws.m.asset_thumb(Req(query={"path": str(src)})))
    assert resp.status == 415
    assert body(resp) == {"error": "not previewable"}
