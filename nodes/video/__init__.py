"""Video generation and HDR-video node group."""
from __future__ import annotations

import logging

from radiance.nodes.video.t2v import (
    RadianceVideoModelInfo,
    RadianceVideoLatentNoise,
    RadianceVideoCondMerge,
    RadianceVideoSampler,
    RadianceT2VPipeline,
    RadianceI2VPipeline,
    RadianceVideoBatchDecode,
    RadianceVideoExport,
)
from radiance.nodes.video.hdr import (
    RadianceVideoHDRConditioner,
    RadianceVideoHDRDecode,
    RadianceVideoFrameRouter,
    RadianceVideoAssembler,
)
from radiance.nodes.aggregate import fold_in_module_nodes

logger = logging.getLogger("radiance.nodes.video")

NODE_CLASS_MAPPINGS = {
    "RadianceVideoModelInfo": RadianceVideoModelInfo,
    "RadianceVideoLatentNoise": RadianceVideoLatentNoise,
    "RadianceVideoCondMerge": RadianceVideoCondMerge,
    "RadianceVideoSampler": RadianceVideoSampler,
    "RadianceT2VPipeline": RadianceT2VPipeline,
    "RadianceI2VPipeline": RadianceI2VPipeline,
    "RadianceVideoBatchDecode": RadianceVideoBatchDecode,
    "RadianceVideoExport": RadianceVideoExport,
    "RadianceVideoHDRConditioner": RadianceVideoHDRConditioner,
    "RadianceVideoHDRDecode": RadianceVideoHDRDecode,
    "RadianceVideoFrameRouter": RadianceVideoFrameRouter,
    "RadianceVideoAssembler": RadianceVideoAssembler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceVideoModelInfo": "◎ Video Model Info",
    "RadianceVideoLatentNoise": "◎ Video Latent Noise",
    "RadianceVideoCondMerge": "◎ Video Cond Merge",
    "RadianceVideoSampler": "◎ Video Sampler",
    "RadianceT2VPipeline": "◎ T2V Pipeline",
    "RadianceI2VPipeline": "◎ I2V Pipeline",
    "RadianceVideoBatchDecode": "◎ Video Batch Decode",
    "RadianceVideoExport": "◎ Video Export",
    "RadianceVideoHDRConditioner": "◎ Video HDR Conditioner",
    "RadianceVideoHDRDecode": "◎ Video HDR Decode",
    "RadianceVideoFrameRouter": "◎ Video Frame Router",
    "RadianceVideoAssembler": "◎ Video Assembler",
}

# Publishing is the default: sweep this package for nodes its leaf modules
# declare but the mapping above does not list. See nodes/aggregate.py — this is
# what stopped seventeen finished nodes from ever reaching the menu.
fold_in_module_nodes(__name__, NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, logger)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
