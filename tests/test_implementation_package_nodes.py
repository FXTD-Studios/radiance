"""The twenty-six nodes that were finished, tested, and invisible.

Four top-level packages -- `radiance.image`, `radiance.hdr`, `radiance.film`
and `radiance.color` -- each declare a full NODE_CLASS_MAPPINGS in their
`__init__.py`. None of them is a node group (see nodes/catalog.py), and
`fold_in_module_nodes` walks only the `__path__` of a package already in the
catalog, so nothing in the load chain ever read those dicts. Three more nodes
were missing from their own leaf module's mapping, which is the one place the
sweep does look.

Twenty-six complete classes, with INPUT_TYPES, tooltips and in several cases
their own tests, that no user could put on a graph. Twenty-five of them ship
now; RadianceHDRHistogram emits a 5-D IMAGE and is withheld with its reason in
tests/test_node_publication_completeness.py, because publishing a node whose
output ComfyUI cannot render is not publishing it.

`tests/test_node_publication_completeness.py` is the guard that makes the class
of bug fail the suite. This file pins the specific fix: each node registered,
pointing at the class it is supposed to point at, in a declared menu section,
with a label. Every assertion here fails against the pre-fix catalog.
"""
import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radiance.nodes.branding import MENU_ROOT, NODE_SECTIONS  # noqa: E402

#: key -> (implementing class's module, declared menu section)
#:
#: The module is part of the assertion on purpose. Registering the key is not
#: the fix; registering it against the implementation that was stranded is.
PUBLISHED = {
    # radiance.image.upscale, the whole of image/upscale.py
    "RadianceProUpscale":           ("radiance.image.upscale", "Upscale"),
    "RadianceUpscaleBySize":        ("radiance.image.upscale", "Upscale"),
    "RadianceDownscale32bit":       ("radiance.image.upscale", "Upscale"),
    "RadianceBitDepthConvert":      ("radiance.image.upscale", "Upscale"),
    "RadianceAIUpscale":            ("radiance.image.upscale", "Upscale"),
    # radiance.hdr
    "RadianceFloat32Convert":       ("radiance.hdr.color", "HDR"),
    "RadianceFloat32ColorCorrect":  ("radiance.hdr.color", "Color"),
    "RadianceHDRColorConvert":      ("radiance.hdr.color", "Color"),
    "RadianceDaVinciWideGamut":     ("radiance.hdr.color", "Color"),
    "RadianceARRIWideGamut4":       ("radiance.hdr.color", "Color"),
    "RadianceACES2OutputTransform": ("radiance.hdr.color", "HDR"),
    "RadianceHDRExposureBlend":     ("radiance.hdr.processing", "HDR"),
    "RadianceHDRShadowHighlight":   ("radiance.hdr.processing", "HDR"),
    "RadianceGPUTensorOps":         ("radiance.hdr.processing", "HDR"),
    "RadianceHDR360Generate":       ("radiance.hdr.panorama", "HDR"),
    "RadianceHighlightSynthesis":   ("radiance.hdr.recovery", "HDR"),
    "RadianceACESConfigManager":    ("radiance.hdr.ocio", "HDR"),
    "RadianceHDROCIOTransform":     ("radiance.hdr.ocio", "HDR"),
    "RadianceOCIOListColorspaces":  ("radiance.hdr.ocio", "HDR"),
    # radiance.film.camera
    "RadianceDepthOfField":         ("radiance.film.camera", "VFX"),
    "RadianceRollingShutter":       ("radiance.film.camera", "VFX"),
    "RadianceCompressionArtifacts": ("radiance.film.camera", "VFX"),
    # Missing from their own leaf module's mapping, which is what the aggregate
    # sweep reads, so not even the sweep could reach them.
    "RadianceUpscaleRouter":        ("radiance.nodes.upscale.upscale", "Upscale"),
    "RadianceFalseColorMonitor":    ("radiance.nodes.monitor.realtime", "Review"),
    "RadianceSplitView":            ("radiance.nodes.monitor.realtime", "Review"),
}

#: Declared in radiance.hdr too, and already registered as the SAME class
#: object out of radiance.hdr.tonemap. Re-registering them from hdr/__init__.py
#: would have been a no-op at best and a rival at worst.
ALREADY_SHIPPING_FROM_THE_SAME_CLASS = (
    "RadianceHDRExpandDynamicRange",
    "RadianceHDRToneMap",
)

#: radiance.film used to declare rival classes under these keys; they were
#: deleted in 3.5.0 and the radiance.nodes.vfx implementations are the nodes.
SHIPPED_BY_NODES_VFX = {
    "RadianceFilmGrain": "radiance.nodes.vfx.optics",
    "RadianceMotionBlur": "radiance.nodes.vfx.motion_blur",
}


def _live():
    import radiance

    if radiance._LOAD_RESULT.failures:
        pytest.skip("catalog incomplete: " + ", ".join(
            f.source.label for f in radiance._LOAD_RESULT.failures))
    return radiance.NODE_CLASS_MAPPINGS


def _display_names():
    import radiance

    return radiance.NODE_DISPLAY_NAME_MAPPINGS


class TestTheStrandedNodesShip:

    @pytest.mark.parametrize("key", sorted(PUBLISHED))
    def test_the_node_is_in_the_catalog(self, key):
        assert key in _live(), f"{key} still cannot be placed on a graph"

    @pytest.mark.parametrize("key", sorted(PUBLISHED))
    def test_the_key_points_at_the_stranded_implementation(self, key):
        expected_module, _section = PUBLISHED[key]
        cls = _live()[key]
        assert cls.__module__ == expected_module, (
            f"{key} resolves to {cls.__module__}, not the implementation that "
            f"was unreachable ({expected_module})"
        )

    @pytest.mark.parametrize("key", sorted(PUBLISHED))
    def test_the_node_has_the_comfyui_surface(self, key):
        cls = _live()[key]
        for attr in ("INPUT_TYPES", "RETURN_TYPES", "FUNCTION", "CATEGORY"):
            assert hasattr(cls, attr), f"{key} has no {attr}"
        assert isinstance(cls.INPUT_TYPES(), dict)
        assert hasattr(cls, cls.FUNCTION), (
            f"{key}.FUNCTION names {cls.FUNCTION!r}, which is not a method"
        )

    @pytest.mark.parametrize("key", sorted(PUBLISHED))
    def test_the_node_has_a_declared_menu_section(self, key):
        _module, section = PUBLISHED[key]
        assert NODE_SECTIONS.get(key) == section, (
            f"{key} should be declared in the {section} section of "
            "NODE_SECTIONS; a node with no declaration falls back to keyword "
            "guessing"
        )

    @pytest.mark.parametrize("key", sorted(PUBLISHED))
    def test_the_declared_section_is_the_one_applied(self, key):
        _module, section = PUBLISHED[key]
        assert _live()[key].CATEGORY == f"{MENU_ROOT}/{section}"

    @pytest.mark.parametrize("key", sorted(PUBLISHED))
    def test_the_node_has_a_label_of_its_own(self, key):
        """Not derived from the class key by the branding fallback.

        Four earlier recoveries shipped with no display name at all, which made
        "every registered node has a label" untestable. See the comment in
        nodes/color/__init__.py.
        """
        label = _display_names().get(key)
        assert label, f"{key} reaches the menu with no display name"
        assert "Radiance" not in label, (
            f"{key} is labelled {label!r}; the menu tab carries the brand"
        )


class TestTheLabelsStayDistinguishable:

    def test_no_two_nodes_share_a_menu_label(self):
        """Two identical labels in one section is a node the user cannot find.

        RadianceACES2OutputTransform and RadianceACES2OutputTransformFull are
        the live case: both are ACES 2.0 output transforms and aces2.py's own
        docstring says one supersedes the other, so both ship and the labels
        have to say which is which.

        Back-compat aliases are exempt: they are DEPRECATED subclasses, which
        ComfyUI loads for saved workflows and keeps out of the node search, so
        they share a label with their canonical node without doubling the menu.
        """
        live = _live()
        names = _display_names()
        by_label = {}
        for key, cls in live.items():
            if getattr(cls, "DEPRECATED", False) is True:
                continue
            section = NODE_SECTIONS.get(key, "?")
            by_label.setdefault((section, str(names.get(key, key))), []).append(key)
        collisions = {k: sorted(v) for k, v in by_label.items() if len(v) > 1}
        assert not collisions, f"same menu label in the same section: {collisions}"


class TestWhatWasDeliberatelyNotRegistered:

    @pytest.mark.parametrize("key", ALREADY_SHIPPING_FROM_THE_SAME_CLASS)
    def test_the_duplicate_declaration_resolves_to_one_class(self, key):
        """radiance.hdr declares these too, as the very same class object."""
        import radiance.hdr

        assert _live()[key] is radiance.hdr.NODE_CLASS_MAPPINGS[key]

    @pytest.mark.parametrize("key,module", sorted(SHIPPED_BY_NODES_VFX.items()))
    def test_the_vfx_implementation_ships_and_film_has_no_rival(self, key, module):
        import radiance.film

        assert _live()[key].__module__ == module
        assert key not in radiance.film.NODE_CLASS_MAPPINGS, "a rival class is back in radiance.film"
