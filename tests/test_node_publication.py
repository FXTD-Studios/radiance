"""A node that exists must reach the menu, or say in writing why it does not.

Radiance has now found the same defect three times:

  * issue #40 — RadianceSamplerPro read an energy mask out of CONDITIONING, and
    no node in the package wrote one, so the feature could not be reached from
    a graph at all;
  * RadianceGradeApply — a finished viewer node whose only trace anywhere was
    an entry in `SECTION_OVERRIDES` saying which menu it would go in if anyone
    ever published it;
  * twenty more, found by widening this suite's source scan past the retired
    root `nodes_*.py` layer: every group's `__init__.py` hand-copied a
    selection of its modules' node keys, and whatever nobody remembered to copy
    simply never existed as far as ComfyUI was concerned.

`nodes/aggregate.py` inverts the default so writing the node is enough. These
tests hold that line: the aggregation actually runs, it does not overwrite a
group's deliberate choices, and withholding a node stays possible but has to be
declared.
"""
import os
import sys
import types

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radiance.nodes.aggregate import (  # noqa: E402
    WITHHELD_ATTR,
    fold_in_module_nodes,
)
from radiance.nodes.catalog import NODE_GROUPS  # noqa: E402


def _live():
    import radiance

    if radiance._LOAD_RESULT.failures:
        pytest.skip("catalog incomplete: " + ", ".join(
            f.source.label for f in radiance._LOAD_RESULT.failures))
    return radiance.NODE_CLASS_MAPPINGS


# ── the mechanism ────────────────────────────────────────────────────────────

class TestFoldInModuleNodes:

    def _package(self, name, modules):
        """Register a throwaway package with the given leaf modules."""
        pkg = types.ModuleType(name)
        pkg.__path__ = []                      # a package, with no files
        sys.modules[name] = pkg
        for leaf, attrs in modules.items():
            mod = types.ModuleType(f"{name}.{leaf}")
            for k, v in attrs.items():
                setattr(mod, k, v)
            sys.modules[f"{name}.{leaf}"] = mod
        return pkg

    def test_a_key_only_the_leaf_module_declares_gets_published(self, monkeypatch):
        pkg = "radiance_test_group_a"
        self._package(pkg, {"leaf": {
            "NODE_CLASS_MAPPINGS": {"Forgotten": object},
            "NODE_DISPLAY_NAME_MAPPINGS": {"Forgotten": "Forgotten Node"},
        }})
        monkeypatch.setattr(
            "radiance.nodes.aggregate._leaf_modules", lambda _p: (f"{pkg}.leaf",))

        classes, displays = {}, {}
        added = fold_in_module_nodes(pkg, classes, displays)

        assert added == ("Forgotten",)
        assert classes["Forgotten"] is object
        assert displays["Forgotten"] == "Forgotten Node"

    def test_the_groups_own_mapping_wins(self, monkeypatch):
        """An explicit entry may deliberately point a key at a different class."""
        pkg = "radiance_test_group_b"

        class Chosen: pass
        class NotChosen: pass

        self._package(pkg, {"leaf": {"NODE_CLASS_MAPPINGS": {"Key": NotChosen}}})
        monkeypatch.setattr(
            "radiance.nodes.aggregate._leaf_modules", lambda _p: (f"{pkg}.leaf",))

        classes = {"Key": Chosen}
        added = fold_in_module_nodes(pkg, classes, {})

        assert added == ()
        assert classes["Key"] is Chosen

    def test_a_withheld_key_is_not_published(self, monkeypatch):
        pkg = "radiance_test_group_c"
        self._package(pkg, {"leaf": {
            "NODE_CLASS_MAPPINGS": {"Shipped": object, "Held": object},
            WITHHELD_ATTR: {"Held"},
        }})
        monkeypatch.setattr(
            "radiance.nodes.aggregate._leaf_modules", lambda _p: (f"{pkg}.leaf",))

        classes = {}
        added = fold_in_module_nodes(pkg, classes, {})

        assert added == ("Shipped",)
        assert "Held" not in classes

    def test_one_broken_module_does_not_cost_the_group(self, monkeypatch, caplog):
        """An optional dependency should not take ten other nodes down with it."""
        pkg = "radiance_test_group_d"
        self._package(pkg, {"good": {"NODE_CLASS_MAPPINGS": {"Good": object}}})

        monkeypatch.setattr(
            "radiance.nodes.aggregate._leaf_modules",
            lambda _p: (f"{pkg}.good", f"{pkg}.missing"))

        with caplog.at_level("WARNING"):
            added = fold_in_module_nodes(pkg, {}, {})

        assert added == ("Good",)
        assert "did not import" in caplog.text

    def test_a_module_without_mappings_is_ignored(self, monkeypatch):
        pkg = "radiance_test_group_e"
        self._package(pkg, {"helpers": {"SOME_CONSTANT": 3}})
        monkeypatch.setattr(
            "radiance.nodes.aggregate._leaf_modules", lambda _p: (f"{pkg}.helpers",))

        assert fold_in_module_nodes(pkg, {}, {}) == ()


# ── the mechanism is actually wired up ───────────────────────────────────────

class TestEveryGroupAggregates:

    @pytest.mark.parametrize("group", [g.module_path for g in NODE_GROUPS])
    def test_the_group_calls_the_sweep(self, group):
        """Not "the file mentions it" — the source of the loaded module."""
        import importlib
        import inspect

        try:
            module = importlib.import_module(group)
        except Exception as exc:
            pytest.skip(f"{group} does not import here: {exc}")

        source = inspect.getsource(module)
        assert "fold_in_module_nodes(" in source, (
            f"{group} hand-lists its nodes with no sweep behind it — a node "
            "added to one of its modules would not reach ComfyUI"
        )


class TestNothingIsLostBetweenModuleAndMenu:

    def test_every_key_a_node_module_declares_reaches_the_catalog(self):
        """The whole point, asserted against the live catalog.

        Deliberate omissions are allowed; they have to be spelled with
        WITHHELD_NODES in the module that declares the key, which this reads
        rather than accepting silence.
        """
        import ast
        import pathlib

        live = set(_live())
        root = pathlib.Path(__file__).resolve().parent.parent
        missing = {}

        for path in sorted((root / "nodes").rglob("*.py")):
            if path.name == "__init__.py":
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:
                continue

            declared, withheld = [], set()
            for node in ast.walk(tree):
                if not isinstance(node, ast.Assign):
                    continue
                for target in node.targets:
                    if not isinstance(target, ast.Name):
                        continue
                    if target.id == "NODE_CLASS_MAPPINGS" and isinstance(node.value, ast.Dict):
                        declared += [k.value for k in node.value.keys
                                     if isinstance(k, ast.Constant)]
                    elif target.id == WITHHELD_ATTR and isinstance(node.value, (ast.Set, ast.List, ast.Tuple)):
                        withheld |= {e.value for e in node.value.elts
                                     if isinstance(e, ast.Constant)}

            gone = [k for k in declared if k not in live and k not in withheld]
            if gone:
                missing[str(path.relative_to(root))] = gone

        assert not missing, (
            "these nodes exist in source but never reach ComfyUI, and are not "
            f"declared withheld: {missing}"
        )


class TestDeprecatedAliasesLoadWithoutCluttering:
    """Old keys must keep loading; the menu must not show the node twice."""

    ALIASES = {
        "RadianceImageLoader": "RadianceUnifiedLoader",
        "RadianceControlApply": "RadianceControlNetApply",
        "RadianceWorkspace": "RadianceProjectManager",
    }

    @pytest.mark.parametrize("alias,canonical", sorted(ALIASES.items()))
    def test_the_alias_is_registered_so_saved_workflows_load(self, alias, canonical):
        live = _live()
        assert alias in live, (
            f"a workflow saved against {alias} would open with a missing-node box"
        )
        assert canonical in live

    @pytest.mark.parametrize("alias,canonical", sorted(ALIASES.items()))
    def test_the_alias_behaves_exactly_like_the_node_it_aliases(self, alias, canonical):
        live = _live()
        assert issubclass(live[alias], live[canonical])

    @pytest.mark.parametrize("alias,canonical", sorted(ALIASES.items()))
    def test_only_the_alias_is_flagged_deprecated(self, alias, canonical):
        """A shared class object would have hidden the canonical node too.

        This is why the aliases are subclasses: ComfyUI reads DEPRECATED off
        the class it is handed, and both keys used to point at the same one.
        """
        live = _live()
        assert getattr(live[alias], "DEPRECATED", False) is True
        assert getattr(live[canonical], "DEPRECATED", False) is not True
