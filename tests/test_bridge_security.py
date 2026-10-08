"""Security tests for the DCC bridge.

History: this file used to assert that an AST allowlist validator (`_validate`)
accepted "safe" expressions and rejected "dangerous" ones. A 2026-07 audit
demonstrated the allowlist was escapable -- it permitted `ast.Assign` and
attribute access on the name `json`, so rebinding that name reached the real
`builtins` module in two statements and gave arbitrary code execution:

    json = json.codecs      # the real codecs module
    json = json.builtins    # the real builtins module
    json.exec(...)          # full RCE

Rather than harden a blocklist over attacker-supplied Python -- which is not a
defensible boundary -- the `exec` command and its sandbox were removed. These
tests now assert that removal, so the capability cannot quietly return.
"""
import json as _json

import pytest

dcc = pytest.importorskip("radiance.nodes.pipeline.dcc")


class _FakeConn:
    """Minimal socket stand-in that records what the handler sends back."""

    def __init__(self, lines):
        self._to_send = b"".join((line + "\n").encode() for line in lines)
        self._pos = 0
        self.sent = b""
        self.closed = False

    def settimeout(self, _t):
        pass

    def recv(self, n):
        if self._pos >= len(self._to_send):
            return b""
        chunk = self._to_send[self._pos:self._pos + n]
        self._pos += n
        return chunk

    def sendall(self, data):
        self.sent += data

    def close(self):
        self.closed = True


def _responses(conn):
    return [_json.loads(line) for line in conn.sent.decode().splitlines() if line.strip()]


# ── The sandbox must stay gone ──────────────────────────────────────────────

@pytest.mark.parametrize("name", [
    "_validate", "_exec_sandbox", "_SAFE_BUILTINS",
    "_ALLOWED_AST_NODES", "_ALLOWED_ATTR_OWNERS", "_dynamic_exec_enabled",
])
def test_sandbox_scaffolding_is_absent(name):
    assert not hasattr(dcc, name), (
        f"{name} is back. The bridge sandbox was removed because its allowlist "
        f"was escapable to RCE; re-adding it reintroduces that hole."
    )


def test_module_has_no_eval_or_exec():
    import inspect
    src = inspect.getsource(dcc)
    assert "eval(" not in src
    # `exec` only survives as the command name in the refusal branch.
    assert "exec(" not in src


# ── The exec command must be refused, whatever it carries ───────────────────

@pytest.mark.parametrize("code", [
    "1 + 2",                                # previously allowed
    "json.dumps({'a': 1})",                 # previously allowed
    "y = 5",                                # the assignment that enabled escape
    "json = json.codecs",                   # step 1 of the published escape
    "__import__('os').system('id')",
    "open('/etc/passwd').read()",
    "().__class__.__bases__",
])
def test_exec_command_is_refused(code):
    conn = _FakeConn([_json.dumps({"cmd": "exec", "code": code})])
    dcc._handle(conn, ("127.0.0.1", 12345))
    replies = _responses(conn)
    assert replies, "handler produced no response"
    assert replies[0]["ok"] is False
    assert "removed" in replies[0]["error"].lower()


def test_exec_refused_even_from_loopback():
    """Loopback was the only gate on the old exec path; it is no longer enough."""
    conn = _FakeConn([_json.dumps({"cmd": "exec", "code": "1 + 1"})])
    dcc._handle(conn, ("127.0.0.1", 9999))
    assert _responses(conn)[0]["ok"] is False


def test_exec_refused_with_dev_mode_set(monkeypatch):
    """RADIANCE_DEV used to unlock eval/exec. It must no longer do so."""
    monkeypatch.setenv("RADIANCE_DEV", "1")
    conn = _FakeConn([_json.dumps({"cmd": "exec", "code": "1 + 1"})])
    dcc._handle(conn, ("127.0.0.1", 9999))
    assert _responses(conn)[0]["ok"] is False


# ── Supported commands still work ───────────────────────────────────────────

def test_ping_still_works():
    conn = _FakeConn([_json.dumps({"cmd": "ping"})])
    dcc._handle(conn, ("127.0.0.1", 9999))
    assert _responses(conn)[0] == {"ok": True, "result": "pong"}


def test_status_still_works():
    conn = _FakeConn([_json.dumps({"cmd": "status"})])
    dcc._handle(conn, ("127.0.0.1", 9999))
    reply = _responses(conn)[0]
    assert reply["ok"] is True and reply["result"]["mode"] == "bridge"


def test_unknown_command_is_rejected():
    conn = _FakeConn([_json.dumps({"cmd": "definitely_not_a_command"})])
    dcc._handle(conn, ("127.0.0.1", 9999))
    assert _responses(conn)[0]["ok"] is False


# ── Bind policy ─────────────────────────────────────────────────────────────

def test_remote_bridge_disabled_by_default(monkeypatch):
    monkeypatch.delenv("RADIANCE_ALLOW_REMOTE_BRIDGE", raising=False)
    assert dcc._remote_bridge_allowed() is False
    monkeypatch.setenv("RADIANCE_ALLOW_REMOTE_BRIDGE", "1")
    assert dcc._remote_bridge_allowed() is True


# ── Remote queue needs the DCC token (code review P2-6) ─────────────────────
# With RADIANCE_ALLOW_REMOTE_BRIDGE=1 the bridge listened on the network and
# relayed `queue` to ComfyUI's /prompt for anyone who could reach the port:
# any installed node, so effectively code execution. A non-loopback bridge now
# requires the shared DCC token (~/.radiance/dcc_token, or
# RADIANCE_DCC_AUTH_TOKEN), the same secret the Nuke listener uses.

@pytest.fixture
def _queued(monkeypatch):
    import urllib.request

    sent = []

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"prompt_id": "x"}'

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: sent.append(req) or _Resp())
    monkeypatch.setenv("RADIANCE_DCC_AUTH_TOKEN", "s3cret-token")
    return sent


def _queue(token=None):
    msg = {"cmd": "queue", "prompt": {"1": {"class_type": "PreviewImage", "inputs": {}}}}
    if token is not None:
        msg["token"] = token
    conn = _FakeConn([_json.dumps(msg)])
    dcc._handle(conn, ("10.0.0.7", 50000))
    return _responses(conn)[0]


@pytest.mark.parametrize("token", [None, "", "wrong", "s3cret-token "])
def test_a_remote_bridge_refuses_queue_without_the_token(monkeypatch, _queued, token):
    monkeypatch.setattr(dcc, "_BOUND_LOOPBACK", False)
    reply = _queue(token)
    assert reply["ok"] is False and "token" in reply["error"].lower()
    assert _queued == [], "the prompt reached ComfyUI without the token"


def test_a_remote_bridge_queues_with_the_token(monkeypatch, _queued):
    monkeypatch.setattr(dcc, "_BOUND_LOOPBACK", False)
    assert _queue("s3cret-token")["ok"] is True
    assert len(_queued) == 1


def test_a_loopback_bridge_still_queues_without_a_token(monkeypatch, _queued):
    monkeypatch.setattr(dcc, "_BOUND_LOOPBACK", True)
    assert _queue()["ok"] is True
    assert len(_queued) == 1
