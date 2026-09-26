"""Generation, sampling, and model-loading node group."""
from __future__ import annotations

import logging

from radiance.nodes.generate.sampler import RadianceSamplerPro
from radiance.nodes.generate.engine import (
    RadianceHDRVAEDecode,
    RadianceHDRVAEEncode,
)
from radiance.nodes.generate.loader import (
    RadianceControlNetApply,
    RadianceLoraStack,
    RadianceUnifiedLoader,
    RadianceVideoLoader,
)
from radiance.nodes.generate.lora import RadianceHDRLoRALoader, RadianceHDRLoRAApply
from radiance.nodes.generate.prompt import (
    RadianceCinematicPromptEncoder,
)
from radiance.nodes.generate.regional import RadianceRegionalPrompt, RadianceRegionalGrid
from radiance.nodes.generate.resolution import RadianceResolution
from radiance.nodes.generate.denoise import RadianceDenoise
from radiance.nodes.generate.energy import RadianceEnergyMask
from radiance.nodes.aggregate import fold_in_module_nodes

logger = logging.getLogger("radiance.nodes.generate")

NODE_CLASS_MAPPINGS = {
    "RadianceSamplerPro": RadianceSamplerPro,
    "RadianceHDRVAEDecode": RadianceHDRVAEDecode,
    # 3.5.0: the encoder HDR VAE Decode's Auto/Direct HDR log inversion was
    # built for. Implemented and tested since v2 as RadianceVAE4KEncode, never
    # registered, so no latent on the menu could be decoded back to HDR.
    "RadianceHDRVAEEncode": RadianceHDRVAEEncode,
    "RadianceLoraStack": RadianceLoraStack,
    "RadianceUnifiedLoader": RadianceUnifiedLoader,
    # Written, complete and importable since v3, but never listed here,
    # so it never appeared in ComfyUI's node menu.
    "RadianceControlNetApply": RadianceControlNetApply,
    "RadianceVideoLoader": RadianceVideoLoader,
    "RadianceHDRLoRALoader": RadianceHDRLoRALoader,
    "RadianceHDRLoRAApply": RadianceHDRLoRAApply,
    "RadianceCinematicPromptEncoder": RadianceCinematicPromptEncoder,
    "RadianceRegionalPrompt": RadianceRegionalPrompt,
    "RadianceRegionalGrid": RadianceRegionalGrid,
    "RadianceResolution": RadianceResolution,
    "RadianceDenoise": RadianceDenoise,
    "RadianceEnergyMask": RadianceEnergyMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceSamplerPro": "◎ Radiance Sampler Pro",
    "RadianceHDRVAEDecode": "◎ HDR VAE Decode",
    "RadianceHDRVAEEncode": "◎ HDR VAE Encode",
    "RadianceLoraStack": "◎ LoRA Stack",
    "RadianceUnifiedLoader": "◎ Radiance Read Models",
    "RadianceControlNetApply": "◎ Radiance ControlNet Apply",
    "RadianceVideoLoader": "◎ Video Loader",
    "RadianceHDRLoRALoader": "◎ HDR LoRA Loader",
    "RadianceHDRLoRAApply": "◎ HDR LoRA Apply",
    "RadianceCinematicPromptEncoder": "◎ Cinematic Prompt Encoder",
    "RadianceRegionalPrompt": "◎ Regional Prompt",
    "RadianceRegionalGrid": "◎ Regional Grid",
    "RadianceResolution": "◎ Resolution",
    "RadianceDenoise": "◎ Denoise",
    "RadianceEnergyMask": "◎ Energy Mask",
}

# Publishing is the default: sweep this package for nodes its leaf modules
# declare but the mapping above does not list. See nodes/aggregate.py — this is
# what stopped seventeen finished nodes from ever reaching the menu.
fold_in_module_nodes(__name__, NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, logger)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
