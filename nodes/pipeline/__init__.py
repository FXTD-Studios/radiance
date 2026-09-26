"""Pipeline orchestration node group."""
from __future__ import annotations

import logging

from radiance.nodes.pipeline.workspace import RadianceProjectManager
# nodes_layout.py is currently empty — no Layout node classes defined yet.
from radiance.nodes.pipeline.overlay import RadianceBlendComposite
from radiance.nodes.pipeline.dcc import RadianceMCP
from radiance.nodes.pipeline.studio_integrations import RadianceDaVinciSend, RadianceNukeSend
from radiance.nodes.aggregate import fold_in_module_nodes

logger = logging.getLogger("radiance.nodes.pipeline")

NODE_CLASS_MAPPINGS = {
    "RadianceProjectManager": RadianceProjectManager,
    "RadianceBlendComposite": RadianceBlendComposite,
    "RadianceMCP": RadianceMCP,
    "RadianceNukeSend": RadianceNukeSend,
    "RadianceDaVinciSend": RadianceDaVinciSend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceProjectManager": "◎ Project Manager",
    "RadianceBlendComposite": "◎ Blend Composite",
    "RadianceMCP": "◎ Radiance DCC Bridge",
    "RadianceNukeSend": "◎ Radiance Send to Nuke",
    "RadianceDaVinciSend": "◎ Radiance Send to DaVinci Resolve",
}

# Publishing is the default: sweep this package for nodes its leaf modules
# declare but the mapping above does not list. See nodes/aggregate.py — this is
# what stopped seventeen finished nodes from ever reaching the menu.
fold_in_module_nodes(__name__, NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, logger)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
