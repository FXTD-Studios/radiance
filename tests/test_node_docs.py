"""Every visible node says what it does, and every input says what it does.

ComfyUI shows a node's DESCRIPTION and each input's "tooltip" on hover; they
are the documentation that travels with the package. In 3.5.0, 14 nodes had
no description and 561 of 1259 inputs had no tooltip. This keeps it at zero:
a new node or input fails here until it is documented. Hidden (DEPRECATED)
nodes are exempt; they exist so saved graphs open.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
if not isinstance(getattr(torch, "__version__", None), str):
    pytest.skip("needs real torch: the stubbed lane under-registers nodes", allow_module_level=True)

pytestmark = pytest.mark.real_torch


def _visible_nodes():
    import radiance
    from radiance.config.constants import EXPECTED_MIN_NODE_COUNT
    if radiance._LOAD_RESULT.failures:
        pytest.skip("environment is short a runtime dependency")
    assert len(radiance.NODE_CLASS_MAPPINGS) >= EXPECTED_MIN_NODE_COUNT, "catalog did not fully load"
    return {k: c for k, c in radiance.NODE_CLASS_MAPPINGS.items() if not getattr(c, "DEPRECATED", False)}


def test_every_visible_node_has_a_description():
    missing = sorted(k for k, c in _visible_nodes().items()
                     if not str(getattr(c, "DESCRIPTION", "") or "").strip())
    assert not missing, f"nodes with no DESCRIPTION: {missing}"


def test_every_input_of_a_visible_node_has_a_tooltip():
    missing = []
    for key, cls in sorted(_visible_nodes().items()):
        spec = cls.INPUT_TYPES()
        for section in ("required", "optional"):
            for name, entry in (spec.get(section) or {}).items():
                opts = entry[1] if isinstance(entry, (list, tuple)) and len(entry) > 1 and isinstance(entry[1], dict) else {}
                if not str(opts.get("tooltip", "")).strip():
                    missing.append(f"{key}.{name}")
    assert not missing, f"{len(missing)} inputs with no tooltip: {missing[:40]}"
