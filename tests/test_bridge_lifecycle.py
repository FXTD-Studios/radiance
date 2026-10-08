"""DCC Bridge: concurrent connections are capped and the server stops.

Each accepted connection used to get its own thread with no limit, so a
client opening idle connections could pile up threads for as long as each
one's 15 s read timeout allowed. stop_server() existed but nothing called it.
"""
import json
import socket
import time

import pytest

dcc = pytest.importorskip("radiance.nodes.pipeline.dcc")


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _connect(port, attempts=60):
    for _ in range(attempts):
        try:
            return socket.create_connection(("127.0.0.1", port), timeout=2.0)
        except OSError:
            time.sleep(0.05)
    raise AssertionError("bridge never accepted a connection")


def _ask(sock, cmd="ping"):
    sock.sendall((json.dumps({"cmd": cmd}) + "\n").encode())
    return json.loads(sock.makefile("r").readline())


@pytest.fixture
def bridge(monkeypatch):
    monkeypatch.setattr(dcc, "_MAX_CONNECTIONS", 2)
    port = _free_port()
    dcc.start_server(port, "127.0.0.1")
    yield port
    dcc.stop_server()


def test_connections_beyond_the_cap_are_refused_with_json(bridge):
    a, b = _connect(bridge), _connect(bridge)
    try:
        assert _ask(a)["result"] == "pong"
        assert _ask(b)["result"] == "pong"
        c = _connect(bridge)
        try:
            reply = json.loads(c.makefile("r").readline())
            assert reply["ok"] is False and "busy" in reply["error"]
            assert c.recv(1) == b""  # and the connection is closed
        finally:
            c.close()
    finally:
        a.close()
        b.close()

    # A freed slot is usable again.
    for _ in range(40):
        d = _connect(bridge)
        try:
            reply = _ask(d)
        finally:
            d.close()
        if reply.get("ok"):
            break
        time.sleep(0.05)
    assert reply == {"ok": True, "result": "pong"}


def test_stop_server_closes_the_listener_and_can_restart(monkeypatch):
    port = _free_port()
    dcc.start_server(port, "127.0.0.1")
    s = _connect(port)
    assert _ask(s)["result"] == "pong"
    s.close()
    assert dcc.stop_server() == "Bridge stopped."
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
    dcc.start_server(port, "127.0.0.1")
    try:
        s = _connect(port)
        assert _ask(s)["result"] == "pong"
        s.close()
    finally:
        dcc.stop_server()


def test_start_server_registers_an_exit_hook(monkeypatch):
    registered = []
    monkeypatch.setattr(dcc.atexit, "register", lambda fn, *a, **k: registered.append(fn))
    monkeypatch.setattr(dcc, "_ATEXIT_REGISTERED", False)
    port = _free_port()
    dcc.start_server(port, "127.0.0.1")
    try:
        assert registered == [dcc.stop_server]
    finally:
        dcc.stop_server()


# ── review: one peer cannot hold every slot ─────────────────────────────────
# The read timeout is per byte, so 16 connections that each trickle a byte
# every few seconds held every slot for as long as they liked.

def test_one_address_cannot_take_every_slot(monkeypatch):
    monkeypatch.setattr(dcc, "_MAX_CONNECTIONS", 4)
    monkeypatch.setattr(dcc, "_MAX_PER_ADDRESS", 1)
    port = _free_port()
    dcc.start_server(port, "127.0.0.1")
    try:
        a = _connect(port)
        try:
            assert _ask(a)["result"] == "pong"
            b = _connect(port)
            try:
                reply = json.loads(b.makefile("r").readline())
                assert reply["ok"] is False and "this address" in reply["error"]
            finally:
                b.close()
        finally:
            a.close()
    finally:
        dcc.stop_server()


def test_a_request_line_that_never_ends_is_cut_off(monkeypatch):
    monkeypatch.setattr(dcc, "_LINE_DEADLINE_S", 0.4)
    port = _free_port()
    dcc.start_server(port, "127.0.0.1")
    try:
        s = _connect(port)
        try:
            t0 = time.monotonic()
            for ch in b'{"cmd": "pi':
                try:
                    s.sendall(bytes([ch]))
                except OSError:
                    break                # cut off mid-line, as intended
                time.sleep(0.06)
                if time.monotonic() - t0 > 2:
                    break
            s.settimeout(3)
            reply = json.loads(s.makefile("r").readline())
            assert reply["ok"] is False and "too slow" in reply["error"]
        finally:
            s.close()
    finally:
        dcc.stop_server()
