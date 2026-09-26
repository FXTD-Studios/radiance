"""VFX, compositing, and optical-effects node group."""
from __future__ import annotations

import logging

from radiance.nodes.vfx.depth import RadianceDepthMapGenerator
from radiance.nodes.vfx.motion import RadianceOpticalFlow
from radiance.nodes.vfx.motion_blur import RadianceMotionBlur
from radiance.nodes.vfx.optics import (
    RadianceLensDistortion,
    RadianceChromaticAberration,
    RadianceAnamorphicStreaks,
    RadianceFilmGrain,
    RadianceVignette,
)
from radiance.nodes.vfx.multipass import (
    NODE_CLASS_MAPPINGS as MP_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as MP_DISPLAY_NAMES,
)

# ── Phase 1, 2, 3 and 5 new high-precision linear VFX nodes ───────────
from radiance.nodes.vfx.masking import (
    RadianceSAMModelLoader,
    RadianceSAMGenerator,
    RadianceMultiMaskVisualPicker,
    RadianceLinearMatting,
)
from radiance.nodes.vfx.plate import (
    RadianceHDRGrainMatcher,
    RadianceSubpixelStabilizer,
)
from radiance.nodes.vfx.inpaint import (
    RadianceHDRCrop,
    RadianceHDRStitch,
    RadianceTemporalStitchStabilizer,
)
from radiance.nodes.vfx.mask_propagate import RadianceVideoMaskPropagator
# `radiance.film` is an implementation package, not a node group: it is not in
# nodes/catalog.py and the aggregate sweep never reaches it, because
# `_leaf_modules` only walks the __path__ of the package it is given and skips
# sub-packages. Its three camera-artefact nodes therefore declared a complete
# NODE_CLASS_MAPPINGS in film/__init__.py that nothing in the load chain ever
# read. This group's __init__ is the registration layer, exactly as
# nodes/color/__init__.py registers RadianceLUTApply out of radiance.color.lut.
#
# All three film/ nodes. film/ also held rival RadianceFilmGrain and
# RadianceMotionBlur classes under the keys this group ships from optics.py
# and motion_blur.py; they were deleted in 3.5.0 (the owner kept these).
from radiance.film.camera import (
    RadianceDepthOfField,
    RadianceRollingShutter,
    RadianceCompressionArtifacts,
)
from radiance.nodes.aggregate import fold_in_module_nodes

logger = logging.getLogger("radiance.nodes.vfx")

NODE_CLASS_MAPPINGS = {
    # Existing VFX nodes
    "RadianceDepthMapGenerator": RadianceDepthMapGenerator,
    "RadianceOpticalFlow": RadianceOpticalFlow,
    "RadianceMotionBlur": RadianceMotionBlur,
    "RadianceLensDistortion": RadianceLensDistortion,
    "RadianceChromaticAberration": RadianceChromaticAberration,
    "RadianceAnamorphicStreaks": RadianceAnamorphicStreaks,
    "RadianceFilmGrain": RadianceFilmGrain,
    "RadianceVignette": RadianceVignette,
    **MP_MAPPINGS,
    
    # Phase 1: Masking & Matting
    "RadianceSAMModelLoader": RadianceSAMModelLoader,
    "RadianceSAMGenerator": RadianceSAMGenerator,
    "RadianceMultiMaskVisualPicker": RadianceMultiMaskVisualPicker,
    "RadianceLinearMatting": RadianceLinearMatting,
    
    # Phase 2: Plate Prep
    "RadianceHDRGrainMatcher": RadianceHDRGrainMatcher,
    "RadianceSubpixelStabilizer": RadianceSubpixelStabilizer,
    
    # Phase 3: Regional Inpainting Crop & Stitch
    "RadianceHDRCrop": RadianceHDRCrop,
    "RadianceHDRStitch": RadianceHDRStitch,
    "RadianceTemporalStitchStabilizer": RadianceTemporalStitchStabilizer,
    
    # Phase 5: Advanced Rotoscoping & Propagation
    "RadianceVideoMaskPropagator": RadianceVideoMaskPropagator,

    # ── from radiance.film.camera (never reachable before 2026-09-18) ──────
    "RadianceDepthOfField": RadianceDepthOfField,
    "RadianceRollingShutter": RadianceRollingShutter,
    "RadianceCompressionArtifacts": RadianceCompressionArtifacts,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Existing display names
    "RadianceDepthMapGenerator": "◎ Depth Map Generator",
    "RadianceOpticalFlow": "◎ Optical Flow",
    "RadianceMotionBlur": "◎ Physical Motion Blur",
    "RadianceLensDistortion": "◎ Lens Distortion",
    "RadianceChromaticAberration": "◎ Chromatic Aberration",
    "RadianceAnamorphicStreaks": "◎ Anamorphic Streaks",
    "RadianceFilmGrain": "◎ Film Grain (Simple)",
    "RadianceVignette": "◎ Vignette",
    **MP_DISPLAY_NAMES,
    
    # Phase 1: Masking & Matting display names
    "RadianceSAMModelLoader": "◎ SAM Model Loader (not shipped)",
    "RadianceSAMGenerator": "◎ SAM Mask Generator (not shipped)",
    "RadianceMultiMaskVisualPicker": "◎ SAM Multi-Mask Picker",
    "RadianceLinearMatting": "◎ Linear Alpha Matting",
    
    # Phase 2: Plate Prep display names
    "RadianceHDRGrainMatcher": "◎ HDR Grain Matcher",
    "RadianceSubpixelStabilizer": "◎ Subpixel Plate Stabilizer",
    
    # Phase 3: Regional Inpainting display names
    "RadianceHDRCrop": "◎ HDR Inpaint Crop",
    "RadianceHDRStitch": "◎ HDR Inpaint Stitch",
    "RadianceTemporalStitchStabilizer": "◎ Temporal Stitch Stabilizer",
    
    # Phase 5: Advanced Rotoscoping display names
    "RadianceVideoMaskPropagator": "◎ Video Mask Propagator",

    # Camera-artefact display names. nodes/branding.py already carried a
    # TERM_OVERRIDES label for RadianceDepthOfField ("Defocus"), written for a
    # menu the node had never been in.
    "RadianceDepthOfField": "◎ Depth of Field",
    "RadianceRollingShutter": "◎ Rolling Shutter",
    "RadianceCompressionArtifacts": "◎ Compression Artifacts",
}

# Publishing is the default: sweep this package for nodes its leaf modules
# declare but the mapping above does not list. See nodes/aggregate.py — this is
# what stopped seventeen finished nodes from ever reaching the menu.
fold_in_module_nodes(__name__, NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, logger)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
