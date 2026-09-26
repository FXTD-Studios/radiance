"""Publishing a node is the default; withholding one has to be written down.

Every node group is a package of leaf modules, each declaring its own
``NODE_CLASS_MAPPINGS``, and each group's ``__init__.py`` then hand-copied a
selection of those keys into the group mapping that ComfyUI actually reads.
Nothing checked the two against each other, so a node was published only if a
human remembered to add a second line in a second file after writing it.

Seventeen did not get that line. They were finished — real classes, working
``INPUT_TYPES``, tooltips, tests in some cases — and they had never once
appeared in anyone's menu:

    RadianceHDRTurboEncoder, RadianceHDRPerChannelNorm,
    RadianceHDRPerChannelDenorm, RadianceACESMetadataFile,
    RadianceACES2Compliance, RadianceColorSpaceInfo,
    RadianceLuminanceGuidance, RadianceHDRBlendValidator, RadianceHDRAnalysis,
    RadianceNDISender, RadianceShotGradeRouter, RadianceAudioCut,
    RadianceAudioTranscribe, RadianceLinearCheck, RadianceCinemaStudio,
    RadianceCameraSync, RadianceVideoPromptBuilder

That is the same defect as issue #40 (a sampler feature no node could reach)
and as RadianceGradeApply (a viewer node whose only trace was an entry in
`SECTION_OVERRIDES`), and finding it a third time by hand is not a strategy.

`fold_in_module_nodes` inverts the default. A group's ``__init__.py`` keeps its
explicit mapping — that stays the readable index of what the group is, and an
explicit entry always wins — and then calls this to sweep its own package for
anything it missed. Adding a node to a leaf module is now enough to ship it.

Withholding is still possible and now has to be deliberate: name the key in a
module-level ``WITHHELD_NODES`` set, with a comment saying why.
"""
from __future__ import annotations

import importlib
import logging
import pkgutil
import sys
from typing import Any, Dict, Iterable, Set, Tuple

#: A leaf module may set this to keep a declared node out of the catalog.
WITHHELD_ATTR = "WITHHELD_NODES"


def _leaf_modules(package_name: str) -> Iterable[str]:
    """Import-able module names directly inside *package_name*, excluding itself.

    Deliberately one level deep, and deliberately confined to the package it is
    given. The sweep publishes whatever a group's own leaf modules declare; it
    is not a discovery mechanism for the rest of the distribution.

    That boundary is why the implementation packages (`radiance.image`,
    `radiance.hdr`, `radiance.film`, `radiance.color`) were invisible here, and
    it still earns its keep: `radiance.film` used to declare RadianceFilmGrain
    and RadianceMotionBlur pointing at different classes from the ones
    radiance.nodes.vfx ships under those keys, and a recursive sweep would have
    decided that collision by import order rather than by anyone choosing. Cross-package
    publication goes through a group's ``__init__.py``, one named key at a time
    -- see nodes/catalog.py and nodes/color/__init__.py.
    """
    package = sys.modules.get(package_name)
    if package is None or not hasattr(package, "__path__"):
        return ()
    return tuple(
        f"{package_name}.{info.name}"
        for info in pkgutil.iter_modules(package.__path__)
        if not info.ispkg and not info.name.startswith("_")
    )


def fold_in_module_nodes(
    package_name: str,
    class_mappings: Dict[str, Any],
    display_mappings: Dict[str, Any],
    logger: "logging.Logger | None" = None,
) -> Tuple[str, ...]:
    """Merge every leaf module's node mappings into the group's, in place.

    Keys the group already declares are left alone: the explicit mapping is the
    group's own statement of intent and may deliberately point a key at a
    different class. Only keys the group does not mention are added.

    A leaf module that fails to import is skipped with a warning rather than
    taking the whole group down — the same policy the catalog loader uses, and
    the reason an optional dependency does not cost you the other ten groups.

    Returns the keys this call added, so a group can log or test them.
    """
    log = logger or logging.getLogger(package_name)
    added: list[str] = []

    for module_name in sorted(_leaf_modules(package_name)):
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - optional deps vary by host
            log.warning("Node module %s did not import: %s", module_name, exc)
            log.debug("Import failure detail for %s", module_name, exc_info=True)
            continue

        module_classes = getattr(module, "NODE_CLASS_MAPPINGS", None)
        if not isinstance(module_classes, dict):
            continue

        withheld: Set[str] = set(getattr(module, WITHHELD_ATTR, ()) or ())
        module_displays = getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", {}) or {}

        for key, cls in module_classes.items():
            if key in class_mappings or key in withheld:
                continue
            class_mappings[key] = cls
            if key in module_displays:
                display_mappings.setdefault(key, module_displays[key])
            added.append(key)

    if added:
        log.debug(
            "%s: published %d node(s) declared in leaf modules but not listed in "
            "the group mapping: %s",
            package_name, len(added), ", ".join(sorted(added)),
        )
    return tuple(added)


__all__ = ["WITHHELD_ATTR", "fold_in_module_nodes"]
