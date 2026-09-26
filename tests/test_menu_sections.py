"""Menu placement is declared, not guessed.

`classify_menu_section` reads the node key, display name, module path and
category as one string and returns on the first keyword rule that matches, so
placement depended on rule ORDER rather than on anything about the node.
`"mask"` is tested for VFX several rules before `"conditioning"` is tested for
Generate, which put RadianceEnergyMask — a node that produces CONDITIONING for
the sampler — one menu away from the only node that reads it. Each such case
had to be noticed by a human and patched into SECTION_OVERRIDES afterwards.

`NODE_SECTIONS` now declares the section for every registered node, and these
tests make an omission fail at commit time instead of shipping a node into the
wrong menu.
"""
import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radiance.nodes.branding import (  # noqa: E402
    MENU_ROOT,
    MENU_STRUCTURE,
    NODE_SECTIONS,
    SECTION_OVERRIDES,
    classify_menu_section,
)


def _live():
    import radiance

    if radiance._LOAD_RESULT.failures:
        pytest.skip("catalog incomplete: " + ", ".join(
            f.source.label for f in radiance._LOAD_RESULT.failures))
    return radiance.NODE_CLASS_MAPPINGS


class TestEveryNodeDeclaresItsSection:

    def test_no_registered_node_is_missing_from_the_table(self):
        """Add the node to NODE_SECTIONS in nodes/branding.py — one line."""
        missing = sorted(set(_live()) - set(NODE_SECTIONS) - set(SECTION_OVERRIDES))
        assert not missing, (
            f"{len(missing)} node(s) would fall back to keyword guessing: {missing}"
        )

    def test_the_table_names_no_node_that_stopped_existing(self):
        live = set(_live())
        stale = sorted(k for k in NODE_SECTIONS if k not in live)
        assert not stale, f"NODE_SECTIONS lists nodes that are gone: {stale}"

    def test_every_declared_section_is_a_real_menu_section(self):
        unknown = sorted({s for s in NODE_SECTIONS.values() if s not in MENU_STRUCTURE})
        assert not unknown, f"sections not in MENU_STRUCTURE: {unknown}"


class TestDeclarationBeatsGuessing:

    def test_a_declared_section_is_used_verbatim(self):
        class Fake:
            CATEGORY = "somewhere/else"

        # "mask" in the key would send this to VFX by keyword.
        section = classify_menu_section("RadianceEnergyMask", Fake, "Energy Mask")
        assert section == "Generate"

    def test_the_energy_mask_regression_specifically(self):
        """The node that exposed the problem: it feeds the sampler."""
        live = _live()
        assert live["RadianceEnergyMask"].CATEGORY == f"{MENU_ROOT}/Generate"

    def test_an_undeclared_node_still_gets_a_section_but_warns(self, caplog):
        """Fallback must not break a third-party or in-progress node."""
        class Fake:
            CATEGORY = f"{MENU_ROOT}/Color"

        with caplog.at_level("WARNING"):
            section = classify_menu_section("SomeUnknownNode", Fake, "Unknown")

        assert section in MENU_STRUCTURE
        assert "no declared menu section" in caplog.text


class TestCategoriesActuallyApplied:

    def test_every_node_category_sits_under_the_menu_root(self):
        bad = {k: c.CATEGORY for k, c in _live().items()
               if not str(getattr(c, "CATEGORY", "")).startswith(MENU_ROOT)}
        assert not bad, f"nodes outside the Radiance menu: {bad}"

    def test_the_applied_category_matches_the_declaration(self):
        mismatched = {}
        for key, cls in _live().items():
            declared = SECTION_OVERRIDES.get(key) or NODE_SECTIONS.get(key)
            if declared is None:
                continue
            applied = str(cls.CATEGORY).rsplit("/", 1)[-1]
            if applied != declared:
                mismatched[key] = (declared, applied)
        assert not mismatched, f"declared vs applied disagree: {mismatched}"


class TestOverridesAreNotDeadWeight:

    def test_overrides_only_cover_nodes_that_are_not_registered(self):
        """A registered node belongs in NODE_SECTIONS, not in the override map.

        RadianceGradeApply sat in SECTION_OVERRIDES for months while never
        being registered — the override was the only trace that anyone knew
        the node existed.
        """
        live = set(_live())
        redundant = sorted(k for k in SECTION_OVERRIDES if k in live)
        assert not redundant, (
            f"these are registered and should be declared in NODE_SECTIONS: {redundant}"
        )

    def test_an_override_still_names_a_class_that_exists_in_the_source(self):
        """An override for a class nobody wrote is a typo, not a decision."""
        import subprocess
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for key in SECTION_OVERRIDES:
            found = subprocess.run(
                ["grep", "-rq", f"class {key}", "--include=*.py", root],
                capture_output=True,
            )
            assert found.returncode == 0, f"SECTION_OVERRIDES['{key}'] names no class"
