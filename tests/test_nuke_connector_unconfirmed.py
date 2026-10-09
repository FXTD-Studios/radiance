"""A command Nuke never confirmed is not reported as a success.

NukeConnector used to break out of its read loop on a timeout (or on a
disconnect) and return (True, ...) with whatever it had read, usually nothing.
A push that Nuke never acknowledged then printed "nuke push: OK ()".
"""
import socket
import threading

import pytest

from radiance.tools.nuke_connector import NukeConnector


@pytest.fixture(autouse=True)
def _token(monkeypatch):
    monkeypatch.setenv("RADIANCE_DCC_AUTH_TOKEN", "test-token")


def _listener(reply: bytes = b"", hold_s: float = 2.0):
    """A listener that reads the request, sends `reply`, then either holds the
    connection open for `hold_s` (a busy Nuke) or closes it (hold_s=0)."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    release = threading.Event()

    def run():
        conn, _ = srv.accept()
        try:
            conn.recv(65536)
            if reply:
                conn.sendall(reply)
            if hold_s:
                release.wait(hold_s)
        finally:
            conn.close()
            srv.close()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return port, release, t


def test_read_timeout_is_not_success():
    port, release, t = _listener(hold_s=5.0)
    try:
        ok, msg = NukeConnector(port=port).send_command('{"action": "ping"}', timeout=0.3)
    finally:
        release.set()
        t.join(2)
    assert ok is False, msg
    assert msg.startswith("UNCONFIRMED:")
    assert "may or may not have run" in msg


def test_partial_reply_then_timeout_is_not_success():
    port, release, t = _listener(reply=b"RADIANCE_PO", hold_s=5.0)
    try:
        ok, msg = NukeConnector(port=port).send_command('{"action": "ping"}', timeout=0.3)
    finally:
        release.set()
        t.join(2)
    assert ok is False and msg.startswith("UNCONFIRMED:"), msg


def test_disconnect_before_the_end_marker_is_not_success():
    port, _release, t = _listener(reply=b"", hold_s=0)
    ok, msg = NukeConnector(port=port).send_command('{"action": "ping"}', timeout=2.0)
    t.join(2)
    assert ok is False and msg.startswith("UNCONFIRMED:"), msg


def test_a_complete_reply_is_still_success():
    port, _release, t = _listener(reply=b"OK\n__RADIANCE_END__\n", hold_s=0)
    ok, msg = NukeConnector(port=port).send_command('{"action": "ping"}', timeout=2.0)
    t.join(2)
    assert (ok, msg) == (True, "OK")


def test_error_reply_is_still_reported_as_error():
    port, _release, t = _listener(reply=b"ERROR: bad\n__RADIANCE_END__\n", hold_s=0)
    ok, msg = NukeConnector(port=port).send_command('{"action": "ping"}', timeout=2.0)
    t.join(2)
    assert (ok, msg) == (False, "ERROR: bad")


@pytest.mark.parametrize("where", ["dcc", "studio"])
def test_push_status_says_unconfirmed_not_ok_or_failed(where, monkeypatch):
    from radiance.tools import nuke_connector

    def unconfirmed(self, **kwargs):
        return (False, "UNCONFIRMED: sent to Nuke at 127.0.0.1:1, but no reply within 15s; "
                       "the command may or may not have run in Nuke")

    monkeypatch.setattr(nuke_connector.NukeConnector, "load_exr", unconfirmed)
    if where == "dcc":
        dcc = pytest.importorskip("radiance.nodes.pipeline.dcc")
        status = dcc._push_to_nuke(["/r/a_1001.exr"], "Read1", 1001, 1001, "127.0.0.1", 1)
    else:
        si = pytest.importorskip("radiance.nodes.pipeline.studio_integrations")
        status = si.RadianceNukeSend._push_to_nuke("/r/a.exr", "Read1", 1, 1, "127.0.0.1", 1)
    assert status.startswith("UNCONFIRMED ("), status
    assert "OK" not in status and "FAILED" not in status


@pytest.mark.parametrize("reply", [b"ERROR: Timeout\n__RADIANCE_END__\n",
                                   b"PENDING: Nuke is busy\n__RADIANCE_END__\n"])
def test_a_busy_nuke_is_unconfirmed_not_failed(reply):
    # The listener waits 10 s for Nuke's main thread and then answers
    # "ERROR: Timeout" (4.0 listeners: "PENDING: ..."), well inside the
    # client's 15 s. The command is still queued in Nuke and usually runs,
    # so FAILED was wrong.
    port, _release, t = _listener(reply=reply, hold_s=0)
    ok, msg = NukeConnector(port=port).send_command('{"action": "ping"}', timeout=2.0)
    t.join(2)
    assert ok is False and msg.startswith("UNCONFIRMED:"), msg
    assert "busy" in msg


def test_the_listener_says_pending_not_error_when_nuke_is_busy():
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[1] / "scripts" / "start_nuke_server.py").read_text(encoding="utf-8")
    assert '"ERROR: Timeout"' not in src
    assert "PENDING:" in src
