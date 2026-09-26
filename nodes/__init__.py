"""Aggregate the organized Radiance node groups for ComfyUI."""
from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

from radiance.nodes.branding import apply_radiance_branding
from radiance.nodes.catalog import enabled_node_group_specs
from radiance.nodes.registry import NodeLoadFailure, load_node_mappings

logger = logging.getLogger("radiance.nodes")


#: Group imports that failed during catalog load.
#:
#: `load_node_mappings` already collected these; this module used to throw them
#: away, which is why an entry-point health check could see zero failures while
#: a whole group (18 nodes behind one import) had quietly gone missing. The
#: entry point reads this to name the modules in its startup ERROR.
NODE_LOAD_FAILURES: Tuple[NodeLoadFailure, ...] = ()


def _load_registered_node_groups() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load the declarative node catalog into ComfyUI mapping dictionaries."""

    global NODE_LOAD_FAILURES

    load_result = load_node_mappings(
        enabled_node_group_specs(),
        logger=logger,
        context="Radiance node catalog",
    )
    NODE_LOAD_FAILURES = load_result.failures
    return load_result.class_mappings, load_result.display_name_mappings


def _load_dynamic_gizmos() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load user-generated gizmo nodes after the static catalog is ready."""

    try:
        from radiance.nodes.gizmo import load_dynamic_gizmos
    except Exception as exc:  # pragma: no cover - optional runtime feature
        logger.warning("Failed to import dynamic Gizmo loader: %s", exc)
        logger.debug("Gizmo loader import failure details", exc_info=True)
        return {}, {}

    try:
        return load_dynamic_gizmos()
    except Exception as exc:  # pragma: no cover - optional runtime feature
        logger.warning("Failed to load dynamically generated Gizmos: %s", exc)
        logger.debug("Gizmo load failure details", exc_info=True)
        return {}, {}


NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = _load_registered_node_groups()

_gizmo_classes, _gizmo_display_names = _load_dynamic_gizmos()
NODE_CLASS_MAPPINGS.update(_gizmo_classes)
NODE_DISPLAY_NAME_MAPPINGS.update(_gizmo_display_names)
apply_radiance_branding(NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "NODE_LOAD_FAILURES"]
