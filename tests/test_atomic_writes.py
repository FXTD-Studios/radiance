"""The small-file atomic writers: what each caller sees.

Written before the writers were moved onto core/sequence_safety, and run
against both versions: the final bytes, the exception on failure, the earlier
file kept, no temporary file left behind, the sessions log's file mode.
"""
import json
import os
import stat
import sys
import types
from pathlib import Path

import pytest

from radiance.delivery import handler as dh
from radiance.nodes.pipeline import workspace as ws


def _leftovers(d: Path, keep):
    return sorted(p.name for p in d.iterdir() if p.name not in keep)


class _Unserialisable:
    pass


# ── delivery: session log ────────────────────────────────────────────────────

def test_sessions_log_content_and_directory(tmp_path):
    target = tmp_path / "sub" / "s.json"
    dh._write_sessions_atomically(str(target), [{"id": "a"}])
    assert json.loads(target.read_text()) == [{"id": "a"}]
    assert target.read_text() == json.dumps([{"id": "a"}], indent=2)
    assert _leftovers(target.parent, {"s.json"}) == []


def test_sessions_log_failure_keeps_the_previous_file(tmp_path):
    target = tmp_path / "s.json"
    dh._write_sessions_atomically(str(target), [{"id": "good"}])
    with pytest.raises(TypeError):
        dh._write_sessions_atomically(str(target), [{"id": _Unserialisable()}])
    assert json.loads(target.read_text()) == [{"id": "good"}]
    assert _leftovers(tmp_path, {"s.json"}) == []


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
def test_sessions_log_is_private_to_the_user(tmp_path):
    target = tmp_path / "s.json"
    dh._write_sessions_atomically(str(target), [])
    assert stat.S_IMODE(os.stat(target).st_mode) == 0o600


# ── delivery: sidecar JSON ───────────────────────────────────────────────────

def test_json_sidecar_content(tmp_path):
    p = tmp_path / "x_meta.json"
    dh._write_json_atomic(str(p), {"a": 1})
    assert p.read_text() == json.dumps({"a": 1}, indent=4)
    assert _leftovers(tmp_path, {"x_meta.json"}) == []


def test_json_sidecar_failure_keeps_the_previous_file(tmp_path):
    p = tmp_path / "x_meta.json"
    dh._write_json_atomic(str(p), {"a": 1})
    with pytest.raises(TypeError):
        dh._write_json_atomic(str(p), {"a": _Unserialisable()})
    assert json.loads(p.read_text()) == {"a": 1}
    assert _leftovers(tmp_path, {"x_meta.json"}) == []


# ── workspace ────────────────────────────────────────────────────────────────

def test_workspace_bytes_and_text(tmp_path):
    p = tmp_path / "a.rad"
    ws._atomic_write_bytes(p, b"\x00\x01binary")
    assert p.read_bytes() == b"\x00\x01binary"
    t = tmp_path / "b.json"
    ws._atomic_write_text(str(t), "naïve ✓")
    assert t.read_bytes() == "naïve ✓".encode("utf-8")
    assert _leftovers(tmp_path, {"a.rad", "b.json"}) == []


def test_workspace_failed_replace_keeps_the_previous_file(tmp_path, monkeypatch):
    p = tmp_path / "a.json"
    ws._atomic_write_text(p, "old")

    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="disk full"):
        ws._atomic_write_text(p, "new")
    monkeypatch.undo()
    assert p.read_text() == "old"
    assert _leftovers(tmp_path, {"a.json"}) == []


# ── OCIO studio config ───────────────────────────────────────────────────────

def _fake_ocio(text):
    cfg = types.SimpleNamespace(serialize=lambda: text)
    return types.SimpleNamespace(Config=types.SimpleNamespace(CreateFromBuiltinConfig=lambda name: cfg))


def test_studio_config_written_once_and_kept_when_unchanged(tmp_path, monkeypatch):
    from radiance.color import ocio_setup as oc
    target = tmp_path / "ACES" / "studio-config.ocio"
    monkeypatch.setattr(oc, "ACES_DIR", str(target.parent))
    monkeypatch.setattr(oc, "STUDIO_FILE", str(target))
    assert oc._write_studio_file(_fake_ocio("ocio_profile_version: 2.4\n")) == str(target)
    assert target.read_text() == "ocio_profile_version: 2.4\n"
    before = target.stat().st_mtime_ns
    os.utime(target, ns=(before - 10_000_000_000, before - 10_000_000_000))
    stamped = target.stat().st_mtime_ns
    assert oc._write_studio_file(_fake_ocio("ocio_profile_version: 2.4\n")) == str(target)
    assert target.stat().st_mtime_ns == stamped, "an unchanged config was rewritten"
    assert _leftovers(target.parent, {"studio-config.ocio"}) == []


def test_studio_config_unwritable_returns_none(tmp_path, monkeypatch):
    from radiance.color import ocio_setup as oc
    blocker = tmp_path / "ACES"
    blocker.write_text("a file where the folder should be")
    monkeypatch.setattr(oc, "ACES_DIR", str(blocker))
    monkeypatch.setattr(oc, "STUDIO_FILE", str(blocker / "studio-config.ocio"))
    assert oc._write_studio_file(_fake_ocio("x")) is None


# ── gizmo save route ─────────────────────────────────────────────────────────

class _Req:
    def __init__(self, data):
        self._data = data
        self.content_length = len(json.dumps(data))

    async def json(self):
        return self._data


def _save_gizmo(name="My Fx", nodes=None):
    import asyncio
    from radiance.nodes import gizmo
    return asyncio.run(gizmo.create_gizmo_api(_Req({"name": name, "nodes": nodes or [{"id": 1, "type": "A"}]})))


def test_gizmo_save_writes_json_and_keeps_the_previous_as_bak(tmp_path, monkeypatch):
    from radiance.nodes import gizmo
    monkeypatch.setattr(gizmo, "GIZMOS_DIR", str(tmp_path))
    # The light lane's aiohttp stand-in has no json_response: record the reply.
    monkeypatch.setattr(gizmo.web, "json_response", lambda body, status=200: {"body": body, "status": status}, raising=False)
    reply = _save_gizmo(nodes=[{"id": 1, "type": "A"}])
    assert reply["status"] == 200 and reply["body"]["success"] is True, reply
    path = tmp_path / "My_Fx.gizmo"
    first = json.loads(path.read_text())
    assert first["name"] == "My_Fx" and first["nodes"] == [{"id": 1, "type": "A"}]
    assert path.read_text() == json.dumps(first, indent=2)
    _save_gizmo(nodes=[{"id": 2, "type": "B"}])
    assert json.loads(path.read_text())["nodes"] == [{"id": 2, "type": "B"}]
    assert json.loads((tmp_path / "My_Fx.gizmo.bak").read_text())["nodes"] == [{"id": 1, "type": "A"}]
    assert _leftovers(tmp_path, {"My_Fx.gizmo", "My_Fx.gizmo.bak"}) == []
