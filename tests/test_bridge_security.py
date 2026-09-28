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


# ── queue must be signed ────────────────────────────────────────────────────
# queue submits a workflow, and write nodes take absolute paths. Unsigned, any
# local process, a LAN peer with RADIANCE_ALLOW_REMOTE_BRIDGE, or a web page
# POSTing to 127.0.0.1:1987 could run workflows and write files.

TOKEN = "t" * 64
PROMPT = {"3": {"class_type": "SaveImage", "inputs": {"filename_prefix": "x"}}}


@pytest.fixture
def submitted(monkeypatch):
    """Records what reaches ComfyUI's /prompt instead of sending it."""
    import urllib.request
    monkeypatch.setenv("RADIANCE_DCC_AUTH_TOKEN", TOKEN)
    monkeypatch.setattr(dcc, "_SEEN_NONCES", {}, raising=False)
    calls = []

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"prompt_id": "abc"}'

    def fake_urlopen(req, timeout=None):
        calls.append(_json.loads(req.data))
        return _Resp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return calls


def _send(*messages):
    conn = _FakeConn([m if isinstance(m, str) else _json.dumps(m) for m in messages])
    dcc._handle(conn, ("127.0.0.1", 9999))
    return _responses(conn)


def test_a_signed_queue_reaches_comfyui(submitted):
    from radiance.core.dcc_auth import sign_queue
    reply = _send(sign_queue(PROMPT, TOKEN))[0]
    assert reply["ok"] is True
    assert submitted == [{"prompt": PROMPT, "client_id": "radiance_dcc_bridge"}]


def test_an_unsigned_queue_is_refused(submitted):
    reply = _send({"cmd": "queue", "prompt": PROMPT})[0]
    assert reply["ok"] is False and "unsigned" in reply["error"]
    assert submitted == []


def test_a_wrong_token_is_refused(submitted):
    from radiance.core.dcc_auth import sign_queue
    reply = _send(sign_queue(PROMPT, "wrong"))[0]
    assert reply["ok"] is False and "signature" in reply["error"]
    assert submitted == []


def test_a_tampered_prompt_is_refused(submitted):
    from radiance.core.dcc_auth import sign_queue
    msg = sign_queue(PROMPT, TOKEN)
    msg["prompt"] = {"3": {"class_type": "SaveImage", "inputs": {"filename_prefix": "/etc/x"}}}
    assert _send(msg)[0]["ok"] is False
    assert submitted == []


def test_key_order_does_not_break_the_signature(submitted):
    from radiance.core.dcc_auth import sign_queue
    msg = sign_queue({"b": 1, "a": 2}, TOKEN)
    line = _json.dumps({**msg, "prompt": {"a": 2, "b": 1}})
    assert _send(line)[0]["ok"] is True


def test_a_replayed_request_is_refused(submitted):
    from radiance.core.dcc_auth import sign_queue
    msg = sign_queue(PROMPT, TOKEN)
    first, second = _send(msg, msg)
    assert first["ok"] is True
    assert second["ok"] is False and "replayed" in second["error"]
    assert len(submitted) == 1


def test_a_stale_request_is_refused(submitted):
    from radiance.core.dcc_auth import queue_signature
    ts, nonce = 1_000_000, "n" * 32
    msg = {"cmd": "queue", "prompt": PROMPT, "ts": ts, "nonce": nonce,
           "sig": queue_signature(TOKEN, ts, nonce, PROMPT)}
    reply = _send(msg)[0]
    assert reply["ok"] is False and "stale" in reply["error"]
    assert submitted == []


def test_an_http_request_is_dropped_before_its_body_runs(submitted):
    """A browser's no-cors POST: request line, headers, then a JSON body."""
    from radiance.core.dcc_auth import sign_queue
    body = _json.dumps(sign_queue(PROMPT, TOKEN))
    replies = _send("POST / HTTP/1.1", "Host: 127.0.0.1:1987",
                    "Content-Type: text/plain", "", body)
    assert replies == []
    assert submitted == []
