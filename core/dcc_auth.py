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
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import time
from pathlib import Path

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


# ── Signed commands for the ComfyUI-side DCC Bridge (nodes/pipeline/dcc.py) ──
#
# The bridge's `queue` command submits a workflow to ComfyUI, and Radiance
# write nodes take absolute output paths, so an unsigned `queue` let anything
# that could reach the bridge port run workflows and write files. It is now
# signed with the same token as the Nuke listener:
#
#   {"cmd": "queue", "prompt": {...}, "ts": <unix seconds>, "nonce": "<hex>",
#    "sig": hmac_sha256(token, "queue\n<ts>\n<nonce>\n" + canonical(prompt))}
#
# canonical() is compact JSON with sorted keys, so client and server hash the
# same bytes whatever key order the client's JSON library produced. A request
# older than MAX_SKEW seconds, or a nonce seen before, is refused.

MAX_SKEW = 300


def canonical(prompt) -> str:
    return json.dumps(prompt, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def queue_signature(token: str, ts, nonce: str, prompt) -> str:
    data = f"queue\n{ts}\n{nonce}\n{canonical(prompt)}".encode("utf-8")
    return hmac.new(token.encode("utf-8"), data, hashlib.sha256).hexdigest()


def sign_queue(prompt, token: str | None = None) -> dict:
    """A signed bridge `queue` request for `prompt`, ready to json.dumps."""
    token = token if token is not None else load_or_create_token()
    if not token:
        raise RuntimeError("No DCC token: set RADIANCE_DCC_AUTH_TOKEN or make "
                           "~/.radiance/dcc_token writable.")
    ts = int(time.time())
    nonce = secrets.token_hex(16)
    return {"cmd": "queue", "prompt": prompt, "ts": ts, "nonce": nonce,
            "sig": queue_signature(token, ts, nonce, prompt)}
