"""
test_node_keys_snapshot.py — freeze the public node-key set.

NODE_CLASS_MAPPINGS keys are public API: saved ComfyUI workflows reference them.
A refactor must never add, drop, or rename a key by accident. This test compares
the live key set against a committed golden snapshot
(``tests/node_keys_snapshot.json``) and fails loudly on any drift.

This is the safety net for the structural refactor phases — with it green, file
moves and layer consolidation provably do not change what workflows can load.

Workflow:
  • First run on a real-deps environment (or with RADIANCE_UPDATE_SNAPSHOT=1)
    writes the snapshot, then SKIPS with a note to commit it.
  • Every later run compares and fails on any added/removed key.
  • When a key change is *intentional*, regenerate:
        RADIANCE_UPDATE_SNAPSHOT=1 python -m pytest tests/test_node_keys_snapshot.py
    and commit the updated json in the same PR.

Only runs under real dependencies: with torch stubbed (the fast CI matrix) some
node modules don't register, so the key set would be incomplete and misleading.
"""
import json
import os

import pytest

_SNAPSHOT = os.path.join(os.path.dirname(__file__), "node_keys_snapshot.json")


def _real_dependencies() -> bool:
    """True only when the real torch is importable (not the test stub)."""
    try:
        import torch
        return isinstance(getattr(torch, "__version__", None), str)
    except Exception:
        return False


def _current_keys():
    import radiance
    return sorted(radiance.NODE_CLASS_MAPPINGS.keys())


@pytest.mark.skipif(
    not _real_dependencies(),
    reason="needs real torch — the stubbed matrix under-registers nodes",
)
def test_node_keys_match_snapshot():
    current = _current_keys()

    # Bootstrap / intentional regeneration.
    if os.environ.get("RADIANCE_UPDATE_SNAPSHOT") or not os.path.exists(_SNAPSHOT):
        with open(_SNAPSHOT, "w", encoding="utf-8") as fh:
            json.dump(current, fh, indent=2)
        pytest.skip(
            f"node_keys_snapshot.json (re)generated with {len(current)} keys — "
            "commit it to lock the public node-key set."
        )

    with open(_SNAPSHOT, encoding="utf-8") as fh:
        expected = json.load(fh)

    added = sorted(set(current) - set(expected))
    removed = sorted(set(expected) - set(current))
    assert not added and not removed, (
        "Public node-key set changed (saved workflows depend on these keys):\n"
        f"  added  : {added}\n"
        f"  removed: {removed}\n"
        "If this change is intentional, regenerate the snapshot with "
        "RADIANCE_UPDATE_SNAPSHOT=1 and commit it in the same PR."
    )
