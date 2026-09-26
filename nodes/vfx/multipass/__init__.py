"""VFX multipass node group."""
from __future__ import annotations

import logging

from ...registry import load_node_group

logger = logging.getLogger("radiance.nodes.vfx.multipass")

SOURCE_MODULES = (
    "relight_comp",
    "master",
    "aov_reader",
    "estimate",
)

_load_result = load_node_group(__name__, child_source_modules=SOURCE_MODULES, logger=logger)

NODE_CLASS_MAPPINGS = _load_result.class_mappings
NODE_DISPLAY_NAME_MAPPINGS = _load_result.display_name_mappings

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "SOURCE_MODULES"]
