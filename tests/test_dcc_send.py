"""Send to Nuke, Send to DaVinci Resolve and the bridge, end to end (3.5.0).

The Nuke listener script runs here against a stand-in `nuke` module, so the
real wire protocol, HMAC check and load_exr action are exercised; Resolve's
scripting module is replaced by a recorder."""
import importlib.util
import json
import os
import socket
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.delenv("RADIANCE_DCC_AUTH_TOKEN", raising=False)


# ── A stand-in Nuke ─────────────────────────────────────────────────────────

class _Knob:
    def __init__(self):
        self.v = None

    def setValue(self, v):
        self.v = v

    def value(self):
        return self.v

    def execute(self):
        pass


class _Node:
    def __init__(self, cls):
        self.cls, self.name, self.inputs = cls, None, {}
        self._knobs = {k: _Knob() for k in ("file", "first", "last", "origfirst", "origlast", "raw",
                                             "colorspace", "reload")}

    def Class(self):
        return self.cls

    def knobs(self):
        return self._knobs

    def __getitem__(self, k):
        return self._knobs[k]

    def setName(self, n):
        if not n.replace("_", "a").isalnum() or n[0].isdigit():
            raise ValueError(f"invalid node name {n!r}")     # as Nuke does
        self.name = n

    def setInput(self, i, node):
        self.inputs[i] = node


def _fake_nuke():
    nuke = types.ModuleType("nuke")
    nuke.GUI = False
    nuke.nodes = {}
    nuke.current = [1]

    def createNode(cls, inpanel=False):
        n = _Node(cls)
        nuke.nodes[id(n)] = n
        return n

    def toNode(name):
        return next((n for n in nuke.nodes.values() if n.name == name), None)

    nuke.createNode = createNode
    nuke.toNode = toNode
    def gui_only(fn):
        if not nuke.GUI:
            raise RuntimeError("Invalid function call in non-gui mode")
        return fn()
    nuke.executeInMainThread = gui_only
    nuke.message = lambda m: gui_only(lambda: None)
    nuke.frame = lambda f=None: nuke.current.__setitem__(0, f) if f is not None else nuke.current[0]
    nukescripts = types.ModuleType("nukescripts")
    nukescripts.PythonPanel = object
    return nuke, nukescripts


@pytest.fixture(params=[False, True], ids=["terminal", "gui"])
def nuke_listener(monkeypatch, request):
    nuke, nukescripts = _fake_nuke()
    monkeypatch.setitem(sys.modules, "nuke", nuke)
    monkeypatch.setitem(sys.modules, "nukescripts", nukescripts)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    monkeypatch.setenv("RADIANCE_NUKE_PORT", str(port))
    spec = importlib.util.spec_from_file_location("start_nuke_server", ROOT / "scripts" / "start_nuke_server.py")
    srv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(srv)
    nuke.GUI = request.param
    t = threading.Thread(target=srv.start_radiance_server, daemon=True)
    t.start()
    for _ in range(50):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.1).close()
            break
        except OSError:
            time.sleep(0.05)
    yield nuke, port
    srv.RUNNING = False
    t.join(timeout=2)


def test_push_to_nuke_works_with_no_configuration(nuke_listener, tmp_path):
    """Both sides find the same token in ~/.radiance/dcc_token; the listener
    used to refuse everything unless both environments set a variable."""
    nuke, port = nuke_listener
    from radiance.nodes.pipeline.studio_integrations import RadianceNukeSend
    folder = tmp_path / "my renders"
    status, _ = RadianceNukeSend().run(torch.rand(3, 8, 8, 3), str(folder), "shot-010_comp",
                                       push_to_nuke=True, nuke_port=port)
    assert "nuke push: OK" in status, status
    read = nuke.toNode("shot_010_comp")
    assert read is not None and read.Class() == "Read"
    assert read["file"].value() == str(folder / "shot-010_comp.####.exr").replace("\\", "/")
    assert (read["first"].value(), read["last"].value(), read["raw"].value()) == (1001, 1003, True)
    nk = (folder / "shot-010_comp.nk").read_text()
    assert 'file "' in nk and "name shot_010_comp" in nk


def test_the_listener_still_refuses_a_wrong_token(nuke_listener, monkeypatch):
    nuke, port = nuke_listener
    from radiance.tools.nuke_connector import NukeConnector
    monkeypatch.setenv("RADIANCE_DCC_AUTH_TOKEN", "not-the-token")
    ok, msg = NukeConnector(port=port).ping()
    assert not ok and "Authentication failed" in msg


def test_nuke_node_names_are_made_valid():
    from radiance.tools.nuke_connector import nuke_node_name
    assert nuke_node_name("shot-010 comp") == "shot_010_comp"
    assert nuke_node_name("010_plate") == "R_010_plate"
    assert nuke_node_name("---") == "RadianceStream"


def test_terminal_bridge_runs_on_calling_thread(monkeypatch):
    nuke, nukescripts = _fake_nuke()
    monkeypatch.setitem(sys.modules, "nuke", nuke)
    monkeypatch.setitem(sys.modules, "nukescripts", nukescripts)
    spec = importlib.util.spec_from_file_location("terminal_nuke_server", ROOT / "scripts" / "start_nuke_server.py")
    srv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(srv)
    seen = []
    monkeypatch.setattr(srv, "start_radiance_server", lambda: seen.append(threading.get_ident()))
    srv.start_bridge()
    assert seen == [threading.get_ident()]
    assert srv._SERVER_THREAD is None


# ── Colour of what is sent ──────────────────────────────────────────────────

def test_input_space_converts_to_what_each_format_expects(tmp_path, monkeypatch):
    import radiance.nodes.pipeline.studio_integrations as si
    got = {}
    monkeypatch.setattr(si, "_save_pil_image", lambda a, p, f: got.__setitem__(Path(p).suffix, a.copy()))
    monkeypatch.setattr(si, "_save_exr", lambda a, p, half=True: got.__setitem__(Path(p).suffix, a.copy()))
    lin = torch.full((1, 2, 2, 3), 0.18)
    si.RadianceDaVinciSend().run(lin, str(tmp_path), "a", "16bit", input_space="Scene-linear")
    assert got[".tif"][0, 0, 0] == pytest.approx(0.4614, abs=1e-3)       # 18 % grey, sRGB-encoded
    srgb = torch.full((1, 2, 2, 3), 0.4614)
    si.RadianceDaVinciSend().run(srgb, str(tmp_path), "b", "EXR", input_space="sRGB display")
    assert got[".exr"][0, 0, 0] == pytest.approx(0.18, abs=1e-3)
    si.RadianceDaVinciSend().run(lin, str(tmp_path), "c", "16bit")      # As is: unchanged
    assert got[".tif"][0, 0, 0] == pytest.approx(0.18)


# ── Resolve Media Pool import ───────────────────────────────────────────────

def _fake_resolve(monkeypatch, tmp_path, running=True):
    """A Python scripting module loaded by the real isolated worker."""
    capture = tmp_path / "resolve-calls.json"
    module = tmp_path / "DaVinciResolveScript.py"
    module.write_text(
        "import json\nfrom pathlib import Path\nfrom types import SimpleNamespace\n"
        "def import_media(arg):\n"
        f"    Path({str(capture)!r}).write_text(json.dumps([arg]))\n"
        "    return ['clip']\n"
        "pool = SimpleNamespace(ImportMedia=import_media)\n"
        "project = SimpleNamespace(GetMediaPool=lambda: pool, GetName=lambda: 'Show')\n"
        "resolve = SimpleNamespace(GetProjectManager=lambda: SimpleNamespace(GetCurrentProject=lambda: project))\n"
        f"def scriptapp(name): return {'resolve' if running else 'None'}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    return capture


def test_resolve_import_of_a_sequence_uses_the_frame_pattern(tmp_path, monkeypatch):
    import radiance.nodes.pipeline.studio_integrations as si
    pool = _fake_resolve(monkeypatch, tmp_path)
    monkeypatch.setattr(si, "_save_pil_image", lambda *a, **k: None)
    status, _ = si.RadianceDaVinciSend().run(torch.rand(3, 4, 4, 3), str(tmp_path), "plate", "16bit",
                                             import_to_media_pool=True)
    assert "imported 1 clip(s) into 'Show'" in status
    assert json.loads(pool.read_text()) == [[{"FilePath": str(tmp_path / "plate.%04d.tif"), "StartIndex": 1001, "EndIndex": 1003}]]


def test_resolve_not_running_is_reported_not_raised(tmp_path, monkeypatch):
    import radiance.nodes.pipeline.studio_integrations as si
    _fake_resolve(monkeypatch, tmp_path, running=False)
    monkeypatch.setattr(si, "_save_pil_image", lambda *a, **k: None)
    status, _ = si.RadianceDaVinciSend().run(torch.rand(1, 4, 4, 3), str(tmp_path), "still", "8bit",
                                             import_to_media_pool=True)
    assert "not imported: Resolve is not running" in status


@pytest.mark.skipif(sys.platform != "win32", reason="Windows fusionscript DLL discovery")
def test_resolve_worker_sets_python_home_without_changing_host(tmp_path, monkeypatch):
    import radiance.nodes.pipeline.studio_integrations as si
    monkeypatch.delenv("PYTHONHOME", raising=False)
    (tmp_path / "DaVinciResolveScript.py").write_text(
        "import os, sys\n"
        "if os.environ.get('PYTHONHOME') != sys.base_prefix: os._exit(23)\n"
        "def scriptapp(name): return None\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    assert "Resolve is not running" in si.import_into_resolve([], 1, 1, None)
    assert "PYTHONHOME" not in os.environ


def test_resolve_native_failure_cannot_terminate_the_host(tmp_path, monkeypatch):
    import radiance.nodes.pipeline.studio_integrations as si
    (tmp_path / "DaVinciResolveScript.py").write_text("import os\nos._exit(23)\n")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    media = tmp_path / "preserved.exr"
    media.write_bytes(b"exported media")
    status = si.import_into_resolve([str(media)], 1, 1, None)
    assert "exited with code 23" in status
    assert media.read_bytes() == b"exported media"


def test_resolve_timeout_is_reported(tmp_path, monkeypatch):
    import radiance.nodes.pipeline.studio_integrations as si
    def timeout(*args, **kwargs):
        raise si.subprocess.TimeoutExpired(args[0], kwargs["timeout"])
    monkeypatch.setattr(si.subprocess, "run", timeout)
    assert "timed out" in si.import_into_resolve([], 1, 1, None)


# ── Bridge ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("payload_size", [200, 4096, 65536])
def test_bridge_refuses_an_endless_line(monkeypatch, payload_size):
    import radiance.nodes.pipeline.dcc as dcc
    monkeypatch.setattr(dcc, "_MAX_LINE", 64)
    a, b = socket.socketpair()
    t = threading.Thread(target=dcc._handle, args=(b,), daemon=True)
    t.start()
    try:
        a.sendall(b"x" * payload_size)
        a.settimeout(3)
        assert b"request too large" in a.recv(1024)
        t.join(timeout=3)
        assert not t.is_alive(), "oversized requests must not keep the handler alive"
    finally:
        a.close()
        t.join(timeout=3)
