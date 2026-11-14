"""The shared secret between ComfyUI and the Radiance listener in Nuke.

The Nuke listener (scripts/start_nuke_server.py) accepts only commands signed
with HMAC-SHA256 under a shared token. That token used to come only from
RADIANCE_DCC_AUTH_TOKEN, which had to be set, identically, in the environment
of both ComfyUI and Nuke. It is unset by default, so out of the box the
listener refused every command and "push to Nuke" always failed.

Since 3.5.0 the token is found the same way on both sides:

1. RADIANCE_DCC_AUTH_TOKEN, if set (studios, or Nuke on another machine);
2. otherwise the file ~/.radiance/dcc_token, created on first use with a
   random 256-bit token and readable only by the user.

Both programs running as the same user on one machine therefore agree with no
configuration. For a remote Nuke, copy that file (or set the variable) there.
The Nuke script carries its own copy of `load_or_create_token` because it runs
inside Nuke, where this package is not importable; keep the two identical.

The DCC Bridge node, when bound to a network address, takes `queue` only
signed with the same token (`sign_queue`); the token itself never goes over
the wire.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import time
from pathlib import Path
from typing import Optional

ENV = "RADIANCE_DCC_AUTH_TOKEN"


def token_path() -> Path:
    return Path(os.path.expanduser("~")) / ".radiance" / "dcc_token"


def load_or_create_token() -> str:
    tok = (os.environ.get(ENV) or "").strip()
    if tok:
        return tok
    path = token_path()
    try:
        tok = path.read_text(encoding="utf-8").strip()
        if tok:
            return tok
    except OSError:
        pass
    tok = secrets.token_hex(32)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(tok)
    except FileExistsError:
        # The other program created it between our read and our write.
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    return tok


def queue_signature(token: str, prompt, ts: int, nonce: str) -> str:
    """HMAC-SHA256 of a DCC Bridge `queue` request, hex.

    Signed: "radiance-queue", the Unix time, a single-use nonce and the prompt
    as compact JSON with sorted keys (json.dumps(prompt, sort_keys=True,
    separators=(",", ":"))), joined by newlines. A client in another language
    must serialise the prompt the same way.
    """
    body = json.dumps(prompt, sort_keys=True, separators=(",", ":"))
    message = f"radiance-queue\n{int(ts)}\n{nonce}\n{body}".encode("utf-8")
    return hmac.new(token.encode("utf-8"), message, hashlib.sha256).hexdigest()


def sign_queue(prompt, token: Optional[str] = None) -> dict:
    """The DCC Bridge `queue` message for `prompt`, signed with the shared token."""
    token = token if token is not None else load_or_create_token()
    ts = int(time.time())
    nonce = secrets.token_hex(16)
    return {"cmd": "queue", "prompt": prompt, "ts": ts, "nonce": nonce,
            "sig": queue_signature(token, prompt, ts, nonce)}
