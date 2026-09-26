"""
test_node_load_completeness.py — catch node modules that silently fail to import.

The production loader (registry.load_node_mappings) wraps each module import in a
try/except and *continues* on failure, so a broken module (e.g. an ImportError for
a missing function) drops its nodes with only a log line — the pack still "loads",
just with whole nodes missing. That's how the Multipass Master node shipped
unregistered. These tests import every node module directly so any such failure is
loud and automatic.

Runs under the conftest torch/comfy stubs, so no GPU or ComfyUI required.
"""
import importlib
import os

import pytest


def _node_module_names():
    """All shipping node modules: top-level nodes_*.py + everything under nodes/."""
    import radiance
    root = os.path.dirname(radiance.__file__)
    names = set()

    # legacy top-level nodes_*.py
    for fn in os.listdir(root):
        if fn.startswith("nodes_") and fn.endswith(".py"):
            names.add("radiance." + fn[:-3])

    # organized nodes/<group>/...  (skip dunder/private files; those are loaders)
    nodes_dir = os.path.join(root, "nodes")
    for dirpath, _dirs, files in os.walk(nodes_dir):
        if "__pycache__" in dirpath:
            continue
        for fn in files:
            if fn.endswith(".py") and not fn.startswith("_"):
                rel = os.path.relpath(os.path.join(dirpath, fn), root)[:-3]
                names.add("radiance." + rel.replace(os.sep, "."))
    return sorted(names)


def test_no_silent_node_import_failures():
    """Every node module must import cleanly — no silently-skipped modules."""
    failures = {}
    for mod in _node_module_names():
        try:
            importlib.import_module(mod)
        except Exception as exc:  # noqa: BLE001 - we want every failure, not the first
            failures[mod] = f"{type(exc).__name__}: {exc}"
    assert not failures, (
        "These node modules failed to import and would be SILENTLY SKIPPED at "
        "runtime (their nodes would be missing from the menu):\n"
        + "\n".join(f"  {m}\n      {e}" for m, e in sorted(failures.items()))
    )


def test_multipass_group_complete():
    """Regression guard for the Multipass Master import bug: all 5 multipass nodes
    must register (Master/AOV Reader/Composite/Relight + EXR Passes Writer)."""
    import radiance
    keys = set(radiance.NODE_CLASS_MAPPINGS)
    expected = {
        "RadianceMultipassMaster",
        "RadianceMultipassEstimate",
        "RadianceMultipassAOVReader",
        "RadianceMultipassComposite",
        "RadianceMultipassRelight",
        "RadianceEXRPassesWriter",
    }
    missing = sorted(expected - keys)
    assert not missing, f"Multipass nodes missing from registry (import failure?): {missing}"
