"""Shared secret resolution helpers.

Workflow JSON is easy to share accidentally, so new nodes should store
environment variable names instead of raw credentials.
"""
from __future__ import annotations

import os
import re
from typing import Mapping, Optional

_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def normalize_env_name(value: object) -> str:
    text = str(value or "").strip()
    if text.lower().startswith("env:"):
        text = text[4:].strip()
    if not text or not _ENV_NAME_RE.match(text):
        return ""
    return text


def resolve_secret(
    explicit_value: object = "",
    env_var: object = "",
    default_env_var: object = "",
    environ: Optional[Mapping[str, str]] = None,
) -> str:
    env = os.environ if environ is None else environ
    for name in (normalize_env_name(env_var), normalize_env_name(default_env_var)):
        if not name:
            continue
        value = env.get(name, "")
        if value:
            return value
    return str(explicit_value or "")
