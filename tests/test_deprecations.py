"""The nodes deprecated in 4.0 (nodes/deprecations.py)."""
import glob
import json
import os
import re

from radiance.nodes.deprecations import DEPRECATED_NODES, apply_deprecations

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _registered_ids():
    ids = set()
    for p in glob.glob(os.path.join(ROOT, "nodes", "**", "*.py"), recursive=True):
        ids.update(re.findall(r'^\s*"(Radiance\w+)"\s*:\s*[A-Za-z_][\w.]*\s*,', open(p, encoding="utf-8").read(), re.M))
    return ids


def test_the_audit_list_is_37_registered_nodes():
    assert len(DEPRECATED_NODES) == 37
    missing = sorted(set(DEPRECATED_NODES) - _registered_ids())
    assert not missing, f"not registered under these ids: {missing}"


def test_no_shipped_workflow_uses_a_deprecated_node():
    used = {}
    for p in glob.glob(os.path.join(ROOT, "workflows", "*.json")):
        for n in json.load(open(p, encoding="utf-8")).get("nodes", []):
            if n.get("type") in DEPRECATED_NODES:
                used.setdefault(os.path.basename(p), []).append(n["type"])
    assert not used, used


def test_marking_hides_the_node_and_says_what_to_use():
    class Old:
        DESCRIPTION = "Does a thing."

    class Kept:
        pass

    marked = apply_deprecations({"RadianceSplitView": Old, "RadianceViewer": Kept})
    assert marked == ["RadianceSplitView"]
    assert Old.DEPRECATED is True
    assert Old.DESCRIPTION.startswith("Deprecated in Radiance 4.0: use Viewer (compare).")
    assert Old.DESCRIPTION.endswith("Does a thing.")
    assert not hasattr(Kept, "DEPRECATED")
    apply_deprecations({"RadianceSplitView": Old})          # twice: the note is not repeated
    assert Old.DESCRIPTION.count("Deprecated in Radiance") == 1
