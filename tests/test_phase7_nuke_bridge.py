"""E-11 and E-04: the Nuke listener refuses replayed and stale frames, caps its
threads, and does not say "loaded" for a path missing on the Nuke host.

The listener script runs against a stand-in `nuke` module, as in
test_dcc_send.py, so the real wire protocol is exercised."""
import importlib.util
import json
import socket
import struct
import sys
import threading
import time
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("RADIANCE_DCC_AUTH_TOKEN", "test-token-0123456789")


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
        self._knobs = {k: _Knob() for k in ("file", "first", "last", "origfirst", "origlast", "raw", "colorspace", "reload")}

    def Class(self):
        return self.cls

    def knobs(self):
        return self._knobs

    def __getitem__(self, k):
        return self._knobs[k]

    def setName(self, n):
        self.name = n

    def setInput(self, i, node):
        self.inputs[i] = node


def _fake_nuke():
    nuke = types.ModuleType("nuke")
    nuke.GUI = False      # set after the script loads (it builds menus when GUI)
    nuke.nodes = {}
    nuke.current = [1]
    nuke.NUKE_VERSION_STRING = "17.0v1"

    def createNode(cls, inpanel=False):
        n = _Node(cls)
        nuke.nodes[id(n)] = n
        return n

    nuke.createNode = createNode
    nuke.toNode = lambda name: next((n for n in nuke.nodes.values() if n.name == name), None)
    nuke.executeInMainThread = lambda fn: fn()
    nuke.message = lambda m: None
    nuke.frame = lambda f=None: nuke.current.__setitem__(0, f) if f is not None else nuke.current[0]
    nukescripts = types.ModuleType("nukescripts")
    nukescripts.PythonPanel = object
    return nuke, nukescripts


@pytest.fixture
def listener(monkeypatch):
    nuke, nukescripts = _fake_nuke()
    monkeypatch.setitem(sys.modules, "nuke", nuke)
    monkeypatch.setitem(sys.modules, "nukescripts", nukescripts)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    monkeypatch.setenv("RADIANCE_NUKE_PORT", str(port))
    spec = importlib.util.spec_from_file_location("nuke_server_p7", ROOT / "scripts" / "start_nuke_server.py")
    srv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(srv)
    nuke.GUI = True       # a thread per connection, as in a GUI session
    t = threading.Thread(target=srv.start_radiance_server, daemon=True)
    t.start()
    for _ in range(50):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.1).close()
            break
        except OSError:
            time.sleep(0.05)
    yield srv, nuke, port
    srv.RUNNING = False
    t.join(timeout=2)


def _send(port, frame):
    with socket.create_connection(("127.0.0.1", port), timeout=5) as c:
        c.sendall(frame)
        data = b""
        while b"__RADIANCE_END__" not in data:
            chunk = c.recv(4096)
            if not chunk:
                break
            data += chunk
    return data.decode().replace("\n__RADIANCE_END__\n", "").strip()


def _frame(cmd, ts_ms=None, nonce=None, token="test-token-0123456789"):
    from radiance.tools.nuke_connector import build_v3_header
    payload = cmd.encode()
    return build_v3_header(token, payload, ts_ms=ts_ms, nonce=nonce) + payload


PING = json.dumps({"action": "ping"})


def test_a_signed_frame_runs_once_and_its_replay_is_refused(listener):
    _, _, port = listener
    frame = _frame(PING)
    assert _send(port, frame) == "RADIANCE_PONG"
    assert "Replay or stale" in _send(port, frame)
    assert _send(port, _frame(PING)) == "RADIANCE_PONG"      # a fresh nonce works


def test_a_stale_timestamp_is_refused(listener):
    srv, _, port = listener
    old = int((time.time() - srv.REPLAY_WINDOW_S - 5) * 1000)
    assert "Replay or stale" in _send(port, _frame(PING, ts_ms=old))


def test_a_wrong_token_does_not_burn_the_nonce(listener):
    _, _, port = listener
    nonce = b"N" * 16
    assert "Authentication failed" in _send(port, _frame(PING, nonce=nonce, token="wrong"))
    assert _send(port, _frame(PING, nonce=nonce)) == "RADIANCE_PONG"


def test_protocol_2_is_refused_with_an_upgrade_message(listener):
    import hashlib
    import hmac
    _, _, port = listener
    payload = PING.encode()
    sig = hmac.new(b"test-token-0123456789", payload, hashlib.sha256).digest()
    reply = _send(port, b"RCMD" + struct.pack("<B", 2) + sig + struct.pack("<I", len(payload)) + payload)
    assert "Protocol 2 is replayable" in reply


def test_connections_beyond_the_cap_are_refused(listener):
    srv, _, port = listener
    held = []
    try:
        for _ in range(srv.MAX_CONNECTIONS):
            c = socket.create_connection(("127.0.0.1", port), timeout=5)
            c.sendall(b"RC")          # an incomplete header keeps its thread busy
            held.append(c)
        time.sleep(0.3)
        assert "Too many connections" in _send(port, _frame(PING))
    finally:
        for c in held:
            c.close()
    time.sleep(5.5)                    # the held threads time out (5 s) and free their slots
    assert _send(port, _frame(PING)) == "RADIANCE_PONG"


def test_load_exr_refuses_a_path_missing_on_the_nuke_host(listener, tmp_path):
    _, nuke, port = listener
    cmd = json.dumps({"action": "load_exr", "filepath": str(tmp_path / "missing.####.exr"),
                      "node_name": "Plate", "first_frame": 1001, "last_frame": 1003})
    reply = _send(port, _frame(cmd))
    assert reply.startswith("ERROR:") and "does not exist on the Nuke host" in reply
    assert nuke.toNode("Plate") is None


def test_load_exr_reports_the_range_the_read_holds(listener, tmp_path):
    _, nuke, port = listener
    for f in (1001, 1002, 1003):
        (tmp_path / f"plate.{f:04d}.exr").write_bytes(b"x")
    cmd = json.dumps({"action": "load_exr", "filepath": str(tmp_path / "plate.####.exr"),
                      "node_name": "Plate", "first_frame": 1001, "last_frame": 1003})
    reply = _send(port, _frame(cmd))
    assert reply.endswith("[1001-1003]"), reply
    assert nuke.toNode("Plate")["file"].value().endswith("plate.####.exr")


def test_first_frame_path_expands_hashes_and_printf():
    spec = importlib.util.spec_from_file_location("p7_ffp", ROOT / "scripts" / "start_nuke_server.py")
    src = (ROOT / "scripts" / "start_nuke_server.py").read_text(encoding="utf-8")
    ns = {}
    start = src.index("def _first_frame_path(")
    end = src.index("\n\n\n", start)
    exec(src[start:end], ns)                       # the helper alone: no nuke import
    f = ns["_first_frame_path"]
    assert f("/a/plate.####.exr", 1001) == "/a/plate.1001.exr"
    assert f("/a/plate.%04d.exr", 7) == "/a/plate.0007.exr"
    assert f("/a/plate.exr", 7) == "/a/plate.exr"
