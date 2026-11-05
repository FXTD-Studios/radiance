"""Nodes deprecated in 4.0: hidden from the node menu, still loaded.

A saved graph that uses one keeps opening and running; ComfyUI marks the node
as deprecated, and its description says what to use instead. The list comes
from the November 2026 node audit (37 nodes with no shipped workflow, whose
job another node already does). They are removed in a later release, once the
nodes that replace them can take over a saved graph's settings.
"""

from __future__ import annotations

from typing import Mapping

#: node id -> what to use instead (shown in the node's description).
DEPRECATED_NODES: dict[str, str] = {
    # SDR -> HDR: the manual chain from before RUDRA, and the orphan HDR LoRA chain
    "RadianceSDRtoHDRExpand": "SDR → HDR Universal (mode Expand)",
    "RadianceHDRExpandDynamicRange": "SDR → HDR Universal (mode Expand)",
    "RadianceSDRToHDRPrepare": "SDR → HDR Universal",
    "RadianceHDRHighlightComposite": "SDR → HDR Universal, or SDR → HDR AI Fill",
    "RadianceHighlightSynthesis": "SDR → HDR AI Fill",
    "RadianceHDRSynthesisEngine": "SDR → HDR Universal, or SDR → HDR AI Fill",
    "RadianceLuminanceGuidance": "SDR → HDR Universal",
    "RadianceHDRLoRALoader": "SDR → HDR Universal (no HDR LoRA was published)",
    "RadianceHDRLoRAApply": "SDR → HDR Universal (no HDR LoRA was published)",
    "RadianceHDRAutoLogSelect": "VAE Encode (HDR)",
    "RadianceHDRColorPipeline": "OCIO ColorSpace",
    "RadianceHDRPerChannelNorm": "VAE Encode (HDR)",
    "RadianceHDRPerChannelDenorm": "VAE Decode (HDR)",
    "RadianceHDRBlendValidator": "HDR Diagnostics",
    # HDR, ACES and measurement
    "RadianceACESTransform": "ACES 2.0 Output Transform",
    "RadianceHDRAnalysis": "HDR Diagnostics",
    "RadianceFloat32Convert": "ColorCorrect",
    "RadianceGPUTensorOps": "ColorCorrect or Grade",
    "RadianceColorSpaceInfo": "Read (its info output)",
    # Read and Write
    "RadianceDigitalCinemaRead": "Read",
    "RadianceDigitalCinemaWrite": "Write",
    # Review
    "RadianceSplitView": "Viewer (compare)",
    "RadianceHDRMonitor": "Viewer",
    # Pipeline
    "RadianceLinearCheck": "Read",
    # Video: the older all-in-one video family
    "RadianceVideoSampler": "Sampler",
    "RadianceT2VPipeline": "Loader, Prompt, Resolution and Sampler",
    "RadianceI2VPipeline": "Loader, Prompt, Resolution and Sampler",
    "RadianceVideoModelInfo": "Loader",
    "RadianceVideoLatentNoise": "Resolution",
    "RadianceVideoBatchDecode": "VAE Decode (HDR)",
    "RadianceVideoHDRConditioner": "SDR → HDR Universal after decoding",
    "RadianceVideoPromptBuilder": "Prompt",
    "RadianceVideoHDRDecode": "VAE Decode (HDR), then Write",
    "RadianceVideoExport": "Write",
    "RadianceVideoAssembler": "Write (sequences and video)",
    # VFX
    "RadianceRelightEngine": "Multipass Relight",
    "RadianceCameraSync": "Multipass Relight",
}


def apply_deprecations(class_mappings: Mapping[str, type]) -> list[str]:
    """Mark the listed nodes deprecated. Returns the ids that were marked."""
    marked = []
    for node_id, instead in DEPRECATED_NODES.items():
        cls = class_mappings.get(node_id)
        if cls is None:
            continue
        cls.DEPRECATED = True
        note = f"Deprecated in Radiance 4.0: use {instead}. Saved graphs still open and run."
        desc = getattr(cls, "DESCRIPTION", "") or ""
        if not desc.startswith("Deprecated in Radiance"):
            cls.DESCRIPTION = f"{note}\n\n{desc}".strip()
        marked.append(node_id)
    return marked
