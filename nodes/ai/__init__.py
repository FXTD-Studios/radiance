"""AI assist and knowledge node group."""
from __future__ import annotations

import logging

from radiance.nodes.ai.scene_cut import (
    RadianceSceneCutDetect,
    RadianceSceneCutSplit,
)
from radiance.nodes.aggregate import fold_in_module_nodes

logger = logging.getLogger("radiance.nodes.ai")

NODE_CLASS_MAPPINGS = {
    "RadianceSceneCutDetect": RadianceSceneCutDetect,
    "RadianceSceneCutSplit": RadianceSceneCutSplit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceSceneCutDetect": "◎ Scene Cut Detect",
    "RadianceSceneCutSplit": "◎ Scene Cut Split",
}

# Publishing is the default: sweep this package for nodes its leaf modules
# declare but the mapping above does not list. See nodes/aggregate.py — this is
# what stopped seventeen finished nodes from ever reaching the menu.
fold_in_module_nodes(__name__, NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, logger)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
