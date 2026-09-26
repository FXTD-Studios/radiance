"""Upscaling node group."""
from __future__ import annotations

import logging

from radiance.nodes.upscale.upscale import (
    RadianceUpscaleTiler,
    RadianceUpscaleImage,
    RadianceUpscaleVideo,
    RadianceUpscaleFaceRestore,
    RadianceUpscaleRouter,
)
# `radiance.image.upscale` is an implementation package, not a node group: it
# is not in nodes/catalog.py and the aggregate sweep never reaches it, because
# `_leaf_modules` only walks the __path__ of the package it is given and skips
# sub-packages. Its five nodes therefore declared a complete
# NODE_CLASS_MAPPINGS in image/__init__.py that nothing in the load chain ever
# read. This group's __init__ is the registration layer, exactly as
# nodes/color/__init__.py registers RadianceLUTApply out of radiance.color.lut.
# Import the leaf module, not the package: conftest shadows `radiance.image`
# with a stub for `defects`, so the package __init__ does not run under test.
from radiance.image.upscale import (
    RadianceProUpscale,
    RadianceUpscaleBySize,
    RadianceDownscale32bit,
    RadianceBitDepthConvert,
    RadianceAIUpscale,
)
from radiance.nodes.aggregate import fold_in_module_nodes

logger = logging.getLogger("radiance.nodes.upscale")

NODE_CLASS_MAPPINGS = {
    "RadianceUpscaleTiler": RadianceUpscaleTiler,
    "RadianceUpscaleImage": RadianceUpscaleImage,
    "RadianceUpscaleVideo": RadianceUpscaleVideo,
    "RadianceUpscaleFaceRestore": RadianceUpscaleFaceRestore,
    "RadianceUpscaleRouter": RadianceUpscaleRouter,
    # ── from radiance.image.upscale (never reachable before 2026-09-18) ────
    "RadianceProUpscale": RadianceProUpscale,
    "RadianceUpscaleBySize": RadianceUpscaleBySize,
    "RadianceDownscale32bit": RadianceDownscale32bit,
    "RadianceBitDepthConvert": RadianceBitDepthConvert,
    "RadianceAIUpscale": RadianceAIUpscale,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceUpscaleTiler": "◎ Upscale Tiler",
    "RadianceUpscaleImage": "◎ Upscale Image",
    "RadianceUpscaleVideo": "◎ Upscale Video",
    "RadianceUpscaleFaceRestore": "◎ Upscale Face Restore",
    "RadianceUpscaleRouter": "◎ Upscale Router",
    # "Pro" is stripped by nodes/branding.py as a marketing word, so the label
    # image/__init__.py intended ("Radiance Pro Upscale") would reach the menu
    # as a bare "Upscale". Name the precision instead, which is what the node
    # is actually for.
    "RadianceProUpscale": "◎ Upscale 32-bit",
    "RadianceUpscaleBySize": "◎ Upscale By Size",
    "RadianceDownscale32bit": "◎ Downscale 32-bit",
    "RadianceBitDepthConvert": "◎ Bit Depth Convert",
    "RadianceAIUpscale": "◎ AI Upscale",
}

# Publishing is the default: sweep this package for nodes its leaf modules
# declare but the mapping above does not list. See nodes/aggregate.py — this is
# what stopped seventeen finished nodes from ever reaching the menu.
fold_in_module_nodes(__name__, NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, logger)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
