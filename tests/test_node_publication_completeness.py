"""Every NODE_CLASS_MAPPINGS in the distribution, not just the ones we remember.

This is the third time the same defect has shipped, and each time the test that
should have caught it was looking at a smaller part of the tree than the defect
was hiding in:

  * 2026-07, `tests/test_node_smoke.py` scanned only the root `nodes_*.py`
    layer, so retiring that layer made keys vanish from the scan rather than
    fail it;
  * 2026-08, `tests/test_node_publication.py` widened the scan to `nodes/`,
    which found seventeen finished nodes no group's `__init__.py` had ever
    listed. That scan still stops at `nodes/`, and skips `__init__.py` files;
  * 2026-09, twenty-three more, in `radiance/image`, `radiance/hdr` and
    `radiance/film`. Each declares a complete NODE_CLASS_MAPPINGS in its
    `__init__.py`. They are not node groups (nodes/catalog.py), and
    `fold_in_module_nodes` cannot cross a package boundary, so nothing in the
    load chain has ever read a single one of those dicts.

So this file scans the WHOLE package, every `.py` file including `__init__.py`,
and asserts two things about every mapping it finds: that the key reaches
ComfyUI, and that it reaches ComfyUI pointing at the class the mapping names.
The second half matters because `radiance/film/__init__.py` used to declare
RadianceFilmGrain and RadianceMotionBlur as classes that were NOT the ones
shipping under those keys -- a key-only check calls that published and moves on.

Nothing here is hand-listed. Writing a new node with a mapping entry and
forgetting to register it turns this suite red on the next run, whichever
package it was written in. A node may still be withheld, by naming it in its
module's WITHHELD_NODES or in one of the two allow-lists below, and an
allow-list entry is a sentence explaining the decision, not a key.
"""
import ast
import importlib
import os
import pathlib
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radiance.nodes.aggregate import WITHHELD_ATTR  # noqa: E402

RADIANCE_ROOT = pathlib.Path(__file__).resolve().parent.parent

#: Directories that are not package source: build artifacts hold copies of it,
#: and the test tree declares throwaway mappings of its own.
_IGNORED_PARTS = {"__pycache__", "build", "dist", ".git", "tests", "scripts"}


# ─────────────────────────────────────────────────────────────────────────────
#  Withheld, deliberately. Each entry is a decision someone made and wrote
#  down. Removing a node's reason for being here means registering it.
# ─────────────────────────────────────────────────────────────────────────────

#: key -> why this key is declared in source and not published at all.
UNPUBLISHED_KEYS = {
    "RadianceSharpen32bit": (
        "Declared in image/upscale.py's own mapping but not in the "
        "image/__init__.py list the 2026-09-18 registration pass worked from, "
        "so it was never assessed with the other five in that module. It is a "
        "complete node (32-bit unsharp/high-pass, HDR-safe) and on current "
        "evidence it should ship; it is held only until the owner confirms it "
        "does not overlap RadianceUpscaleTiler's sharpening, which is a "
        "product call, not a registration one."
    ),
    "RadianceHDRHistogram": (
        "Complete node, and its IMAGE output is 5-D. hdr/analysis.py:340 calls "
        "numpy_to_tensor_float32(hist_np).unsqueeze(0) under a comment saying "
        "that helper returns (H, W, C) for a single frame; it does not, it adds "
        "the batch dimension itself at hdr/utils.py:44, so the node emits "
        "(1, 1, H, W, C). ComfyUI cannot render that and no downstream node can "
        "read it. Nothing caught it because the node never shipped and nothing "
        "ever executed it -- tests/test_node_functional.py catches it the "
        "moment it is registered. Drop the unsqueeze and it publishes, but "
        "whether the stale comment or the helper is the thing to correct is the "
        "owner's call, so it is not shipped broken in the meantime."
    ),
}

#: key -> why the class this mapping names is not the class that ships under
#: the key. A rival implementation, in other words.
SUPERSEDED_IMPLEMENTATIONS: dict = {
    # Empty since 2026-09-25. radiance.film declared rival RadianceFilmGrain and
    # RadianceMotionBlur classes; the owner kept the ones radiance.nodes.vfx
    # ships and the film/ rivals were deleted.
}


# ─────────────────────────────────────────────────────────────────────────────
#  Discovery
# ─────────────────────────────────────────────────────────────────────────────

def _source_files():
    for path in sorted(RADIANCE_ROOT.rglob("*.py")):
        rel = path.relative_to(RADIANCE_ROOT)
        if any(p in _IGNORED_PARTS or p.endswith(".egg-info") for p in rel.parts):
            continue
        yield path, rel


def _declarations():
    """Every NODE_CLASS_MAPPINGS literal in the package, by AST.

    AST rather than import, because `radiance/image/__init__.py` is exactly the
    file a scan must not miss and conftest replaces `radiance.image` with a
    stub, so an import-driven scan reads an empty mapping and reports nothing
    wrong. Returns [(relative path, module name, [keys], withheld keys)].
    """
    out = []
    for path, rel in _source_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue

        keys, withheld = [], set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if not isinstance(target, ast.Name):
                    continue
                if target.id == "NODE_CLASS_MAPPINGS" and isinstance(node.value, ast.Dict):
                    keys += [k.value for k in node.value.keys
                             if isinstance(k, ast.Constant) and isinstance(k.value, str)]
                elif target.id == WITHHELD_ATTR and isinstance(
                        node.value, (ast.Set, ast.List, ast.Tuple)):
                    withheld |= {e.value for e in node.value.elts
                                 if isinstance(e, ast.Constant)}
        if keys:
            out.append((str(rel), _module_name(rel), keys, withheld))
    return out


def _module_name(rel: pathlib.Path) -> str:
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1][: -len(".py")]
    return ".".join(["radiance", *parts])


def _live():
    import radiance

    if radiance._LOAD_RESULT.failures:
        pytest.skip("catalog incomplete: " + ", ".join(
            f.source.label for f in radiance._LOAD_RESULT.failures))
    return radiance.NODE_CLASS_MAPPINGS


_DECLARATIONS = _declarations()


# ─────────────────────────────────────────────────────────────────────────────
#  The scan reaches what it is supposed to reach
# ─────────────────────────────────────────────────────────────────────────────

class TestTheScanSeesTheWholePackage:
    """Pins the part that was wrong all three times: the scan's reach."""

    @pytest.mark.parametrize("rel", [
        "image/__init__.py",
        "hdr/__init__.py",
        "film/__init__.py",
        "color/__init__.py",
        "nodes/upscale/__init__.py",
        "nodes/hdr/__init__.py",
        "nodes/vfx/__init__.py",
        "nodes/monitor/realtime.py",
    ])
    def test_a_file_that_declares_nodes_is_in_the_scan(self, rel):
        """Each of these hid a node from ComfyUI at some point.

        A scan limited to `nodes/`, or one that skips `__init__.py`, drops
        entries here and reports success, which is what happened in 2026-08.
        """
        scanned = {d[0].replace(os.sep, "/") for d in _DECLARATIONS}
        # color/__init__.py re-exports rather than declaring; accept either the
        # package __init__ or the leaf module that owns the keys.
        if rel == "color/__init__.py" and rel not in scanned:
            assert "color/lut.py" in scanned
            return
        assert rel in scanned, (
            f"{rel} declares nodes and the scan does not reach it; a node "
            "added there would never fail this suite"
        )

    def test_the_scan_is_not_confined_to_the_nodes_package(self):
        outside = sorted(d[0].replace(os.sep, "/") for d in _DECLARATIONS
                         if not d[0].replace(os.sep, "/").startswith("nodes/"))
        assert outside, (
            "no declarations found outside nodes/, so the scan has narrowed back "
            "to the subtree that hid twenty-three nodes until 2026-09-18"
        )

    def test_every_declaring_module_imports(self):
        """A module that cannot be imported cannot be checked below.

        Without this, breaking an implementation module's imports would quietly
        turn the identity check into a no-op for every node it declares.
        """
        broken = {}
        for rel, module, _keys, _withheld in _DECLARATIONS:
            try:
                importlib.import_module(module)
            except Exception as exc:
                broken[rel] = f"{type(exc).__name__}: {exc}"
        assert not broken, f"declaring modules that do not import: {broken}"


# ─────────────────────────────────────────────────────────────────────────────
#  Nothing declared is unreachable
# ─────────────────────────────────────────────────────────────────────────────

class TestEveryDeclaredNodeReachesTheMenu:

    def test_every_declared_key_is_registered(self):
        live = set(_live())
        orphans = {}
        for rel, _module, keys, withheld in _DECLARATIONS:
            gone = sorted({k for k in keys
                           if k not in live
                           and k not in withheld
                           and k not in UNPUBLISHED_KEYS})
            if gone:
                orphans[rel] = gone
        assert not orphans, (
            "these nodes are written and ComfyUI never sees them. Register "
            "them in the owning radiance/nodes/<group>/__init__.py, or name "
            "them in the module's WITHHELD_NODES or in UNPUBLISHED_KEYS in "
            f"this file with a reason: {orphans}"
        )

    def test_every_declared_class_is_the_class_that_ships(self):
        """A key can be registered and still not publish the node in question.

        `radiance/film/__init__.py` declares RadianceFilmGrain, the key is in
        the catalog, and the class behind it is somebody else's node. Checking
        keys alone reports that as published.
        """
        live = _live()
        rivals = {}
        for rel, module, keys, withheld in _DECLARATIONS:
            try:
                declared = getattr(importlib.import_module(module),
                                   "NODE_CLASS_MAPPINGS", {}) or {}
            except Exception:
                continue  # test_every_declaring_module_imports owns this case
            for key, cls in declared.items():
                if key in withheld or key in UNPUBLISHED_KEYS:
                    continue
                if key in SUPERSEDED_IMPLEMENTATIONS:
                    continue
                shipped = live.get(key)
                if shipped is not None and shipped is not cls:
                    rivals[f"{rel}::{key}"] = (
                        f"declares {cls.__module__}.{cls.__qualname__}, "
                        f"ships {shipped.__module__}.{shipped.__qualname__}"
                    )
        assert not rivals, (
            "two implementations are competing for one menu key. Decide which "
            "one ships, then either register it or record the decision in "
            f"SUPERSEDED_IMPLEMENTATIONS in this file: {rivals}"
        )

    def test_every_registered_node_has_the_comfyui_surface(self):
        """Registering something that is not a node is its own kind of broken."""
        incomplete = {}
        for key, cls in _live().items():
            missing = [a for a in ("INPUT_TYPES", "RETURN_TYPES", "FUNCTION", "CATEGORY")
                       if not hasattr(cls, a)]
            fn = getattr(cls, "FUNCTION", None)
            if fn and not hasattr(cls, fn):
                missing.append(f"FUNCTION={fn!r} names no method")
            if missing:
                incomplete[key] = missing
        assert not incomplete, f"registered without a node surface: {incomplete}"


# ─────────────────────────────────────────────────────────────────────────────
#  The allow-lists are a ratchet, not a drawer
# ─────────────────────────────────────────────────────────────────────────────

class TestTheAllowListsStayHonest:

    @pytest.mark.parametrize("key", sorted(UNPUBLISHED_KEYS))
    def test_an_unpublished_key_is_still_declared_somewhere(self, key):
        declared = {k for _rel, _m, keys, _w in _DECLARATIONS for k in keys}
        assert key in declared, (
            f"UNPUBLISHED_KEYS['{key}'] names a node nobody declares any more; "
            "delete the entry"
        )

    @pytest.mark.parametrize("key", sorted(UNPUBLISHED_KEYS))
    def test_an_unpublished_key_really_is_unpublished(self, key):
        assert key not in _live(), (
            f"{key} ships now, so remove it from UNPUBLISHED_KEYS; a stale "
            "entry is somewhere the next orphan can hide"
        )

    @pytest.mark.parametrize("key", sorted(SUPERSEDED_IMPLEMENTATIONS))
    def test_a_superseded_key_still_has_two_implementations(self, key):
        """When the rivalry is resolved the entry must go, not linger."""
        live = _live()
        assert key in live, f"{key} is not registered at all; this is not a rivalry"

        implementations = set()
        for _rel, module, _keys, _w in _DECLARATIONS:
            try:
                declared = getattr(importlib.import_module(module),
                                   "NODE_CLASS_MAPPINGS", {}) or {}
            except Exception:
                continue
            if key in declared:
                implementations.add(declared[key])
        assert len(implementations) > 1, (
            f"only one class is declared for {key} now; the decision has been "
            "made, so remove it from SUPERSEDED_IMPLEMENTATIONS"
        )

    @pytest.mark.parametrize(
        "key", sorted(set(UNPUBLISHED_KEYS) | set(SUPERSEDED_IMPLEMENTATIONS)))
    def test_the_entry_carries_a_written_reason(self, key):
        """A bare key is how a node gets withheld by accident and stays that way."""
        reason = UNPUBLISHED_KEYS.get(key) or SUPERSEDED_IMPLEMENTATIONS[key]
        assert isinstance(reason, str) and len(reason.split()) >= 10, (
            f"the allow-list entry for {key} has to say why, in a sentence"
        )
