"""HDR pipeline node group."""
from __future__ import annotations

import logging

from radiance.nodes.hdr.aces2 import (
    RadianceACES2Tonescale,
    RadianceACES2ReachGamutCompress,
    RadianceACES2OutputTransformFull,
)
from radiance.nodes.hdr.colorspace import (
    RadianceHDRColorPipeline,
)
from radiance.nodes.hdr.delivery import (
    RadianceHDREncode,
    RadianceHDRMonitor,
)
from radiance.nodes.hdr.encoder import (
    RadianceHDRLatentEncoder,
)
from radiance.nodes.hdr.smart import (
    RadianceHDRAutoLogSelect,
    RadianceHDRDiagnostics,
)
from radiance.nodes.hdr.uplift import (
    RadianceClipDetector,
    RadianceSDRToHDRPrepare,
    RadianceHDRHighlightComposite,
)
from radiance.nodes.hdr.uplift_universal import (
    RadianceSDRToHDRRecover,
    RadianceSDRToHDRUniversal,
)
from radiance.nodes.hdr.synthesis import (
    RadianceSDRtoHDRExpand,
    RadianceHDRSynthesisEngine,
    RadianceRelightEngine,
)
from radiance.nodes.hdr.tonemap import (
    HDRExpandDynamicRange,
    HDRToneMap,
)
# `radiance.hdr` is an implementation package, not a node group: it is not in
# nodes/catalog.py and the aggregate sweep never reaches it, because
# `_leaf_modules` only walks the __path__ of the package it is given and skips
# sub-packages. Its fifteen remaining nodes therefore declared a complete
# NODE_CLASS_MAPPINGS in hdr/__init__.py that nothing in the load chain ever
# read. This group's __init__ is the registration layer, exactly as
# nodes/color/__init__.py registers RadianceLUTApply out of radiance.color.lut,
# and exactly as nodes/hdr/tonemap.py already re-exports radiance.hdr.tonemap.
# Import the leaf modules rather than the package so the dependency each node
# actually needs is visible at the import that pulls it in.
from radiance.hdr.color import (
    ImageToFloat32,
    Float32ColorCorrect,
    ColorSpaceConvert,
    DaVinciWideGamut,
    ARRIWideGamut4,
    ACES2OutputTransform,
)
from radiance.hdr.processing import (
    HDRExposureBlend,
    HDRShadowHighlightRecovery,
    GPUTensorOps,
)
from radiance.hdr.panorama import HDR360Generate
from radiance.hdr.recovery import RadianceHighlightSynthesis
from radiance.hdr.ocio import (
    ACESConfigManager,
    OCIOColorTransform,
    OCIOListColorspaces,
)
from radiance.nodes.aggregate import fold_in_module_nodes
logger = logging.getLogger("radiance.nodes.hdr")

NODE_CLASS_MAPPINGS = {
    "RadianceACES2Tonescale": RadianceACES2Tonescale,
    "RadianceACES2ReachGamutCompress": RadianceACES2ReachGamutCompress,
    "RadianceACES2OutputTransformFull": RadianceACES2OutputTransformFull,
    "RadianceHDRColorPipeline": RadianceHDRColorPipeline,
    "RadianceHDREncode": RadianceHDREncode,
    "RadianceHDRMonitor": RadianceHDRMonitor,
    "RadianceHDRAutoLogSelect": RadianceHDRAutoLogSelect,
    "RadianceHDRDiagnostics": RadianceHDRDiagnostics,
    "RadianceClipDetector": RadianceClipDetector,
    "RadianceSDRToHDRPrepare": RadianceSDRToHDRPrepare,
    "RadianceHDRHighlightComposite": RadianceHDRHighlightComposite,
    "RadianceSDRtoHDRExpand": RadianceSDRtoHDRExpand,
    "RadianceSDRToHDRRecover": RadianceSDRToHDRRecover,
    "RadianceSDRToHDRUniversal": RadianceSDRToHDRUniversal,
    "RadianceHDRExpandDynamicRange": HDRExpandDynamicRange,
    "RadianceHDRToneMap": HDRToneMap,
    "RadianceHDRSynthesisEngine": RadianceHDRSynthesisEngine,
    "RadianceRelightEngine": RadianceRelightEngine,
    "RadianceHDRLatentEncoder": RadianceHDRLatentEncoder,
    # ── from radiance.hdr (never reachable before 2026-09-18) ──────────────
    # RadianceHDRExpandDynamicRange and RadianceHDRToneMap are deliberately
    # absent from this block: hdr/__init__.py declares them too, and they are
    # already registered above as the same class objects out of
    # radiance.hdr.tonemap.
    "RadianceFloat32Convert": ImageToFloat32,
    "RadianceFloat32ColorCorrect": Float32ColorCorrect,
    # Key avoids a clash with nodes.color's RadianceColorSpaceConvert; the two
    # are different nodes, not two spellings of one.
    "RadianceHDRColorConvert": ColorSpaceConvert,
    "RadianceDaVinciWideGamut": DaVinciWideGamut,
    "RadianceARRIWideGamut4": ARRIWideGamut4,
    "RadianceACES2OutputTransform": ACES2OutputTransform,
    # RadianceHDRHistogram is NOT here. hdr/analysis.py:340 calls
    # numpy_to_tensor_float32(...).unsqueeze(0) on a comment claiming that
    # helper returns (H, W, C); it already adds the batch dimension itself
    # (hdr/utils.py:44), so the node emits a 5-D (1, 1, H, W, C) IMAGE that
    # ComfyUI cannot render and no downstream node can read. Nothing caught it
    # because the node never shipped and nothing ever executed it. It is one
    # line from working, but which way to correct it is the owner's call, so it
    # is withheld with a reason in
    # tests/test_node_publication_completeness.py rather than published broken.
    "RadianceHDRExposureBlend": HDRExposureBlend,
    "RadianceHDRShadowHighlight": HDRShadowHighlightRecovery,
    "RadianceGPUTensorOps": GPUTensorOps,
    "RadianceHDR360Generate": HDR360Generate,
    "RadianceHighlightSynthesis": RadianceHighlightSynthesis,
    "RadianceACESConfigManager": ACESConfigManager,
    # Key avoids a clash with nodes.color's OCIO nodes.
    "RadianceHDROCIOTransform": OCIOColorTransform,
    "RadianceOCIOListColorspaces": OCIOListColorspaces,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceACES2Tonescale": "◎ ACES 2.0 Tonescale",
    "RadianceACES2ReachGamutCompress": "◎ ACES 2.0 Gamut Compress",
    "RadianceACES2OutputTransformFull": "◎ ACES 2.0 Output Transform",
    "RadianceHDRColorPipeline": "◎ HDR Color Pipeline",
    "RadianceHDREncode": "◎ HDR Encode",
    "RadianceHDRMonitor": "◎ HDR Monitor",
    "RadianceHDRAutoLogSelect": "◎ HDR Auto Log Select",
    "RadianceHDRDiagnostics": "◎ HDR Diagnostics",
    "RadianceClipDetector": "◎ Clip Detector",
    "RadianceSDRToHDRPrepare": "◎ SDR to HDR Prepare",
    "RadianceHDRHighlightComposite": "◎ HDR Highlight Composite",
    "RadianceSDRtoHDRExpand": "◎ SDR to HDR Expand",
    "RadianceSDRToHDRRecover": "◎ SDR → HDR Recover",
    "RadianceSDRToHDRUniversal": "◎ SDR → HDR Universal",
    "RadianceHDRExpandDynamicRange": "◎ HDR Expand Dynamic Range",
    "RadianceHDRToneMap": "◎ HDR Tone Map",
    "RadianceHDRSynthesisEngine": "◎ HDR Synthesis Engine",
    "RadianceRelightEngine": "◎ Relight Engine",
    "RadianceHDRLatentEncoder": "◎ HDR Latent Encoder",
    "RadianceFloat32Convert": "◎ Float32 Convert",
    "RadianceFloat32ColorCorrect": "◎ Float32 Color Correct",
    "RadianceHDRColorConvert": "◎ HDR Color Convert",
    "RadianceDaVinciWideGamut": "◎ DaVinci Wide Gamut",
    "RadianceARRIWideGamut4": "◎ ARRI Wide Gamut 4",
    # nodes/hdr/aces2.py says in its own docstring that it supersedes this one
    # "for high-accuracy deliveries requiring full Academy S-2126 compliance",
    # so both ship and the label has to say which is which. Without the suffix
    # the two would land in the menu under one identical name.
    "RadianceACES2OutputTransform": "◎ ACES 2.0 Output Transform (Legacy)",
    "RadianceHDRExposureBlend": "◎ HDR Exposure Blend",
    "RadianceHDRShadowHighlight": "◎ HDR Shadow / Highlight Recovery",
    "RadianceGPUTensorOps": "◎ GPU Tensor Ops",
    "RadianceHDR360Generate": "◎ HDR 360 Generate",
    "RadianceHighlightSynthesis": "◎ Highlight Synthesis",
    "RadianceACESConfigManager": "◎ ACES Config Manager",
    "RadianceHDROCIOTransform": "◎ OCIO Transform (HDR)",
    "RadianceOCIOListColorspaces": "◎ OCIO List Colorspaces",
}

# Publishing is the default: sweep this package for nodes its leaf modules
# declare but the mapping above does not list. See nodes/aggregate.py — this is
# what stopped seventeen finished nodes from ever reaching the menu.
fold_in_module_nodes(__name__, NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, logger)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
