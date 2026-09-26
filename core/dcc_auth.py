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

import os
import secrets
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
