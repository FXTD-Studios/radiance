"""Color-science node group — organized internal modules."""
from __future__ import annotations

import logging

from radiance.nodes.color.cdl import (
    RadianceCDLTransform,
    RadianceCDLImport,
    RadianceCDLExport,
)
from radiance.color.lut import (
    RadianceLUTApply,
    RadianceLUTBlend,
)
from radiance.nodes.color.colorspace import (
    RadianceWhiteBalance,
    RadianceColorSpaceConvert,
    RadianceACESTransform,
    RadianceBitDepthDegrade,
)
from radiance.nodes.color.curves import (
    RadianceHueCurves,
    RadianceCurves,
)
from radiance.nodes.color.grade import (
    RadianceGrade,
    RadianceApplyGradeInfo,
    RadianceGradeMatch,
)
from radiance.nodes.color.ocio import (
    RadianceOCIOContext,
)
from radiance.nodes.color.qc import (
    RadianceQC,
    RadiancePolicyGuard,
)
from radiance.nodes.aggregate import fold_in_module_nodes

logger = logging.getLogger("radiance.nodes.color")

NODE_CLASS_MAPPINGS = {
    "RadianceCDLTransform": RadianceCDLTransform,
    "RadianceCDLImport": RadianceCDLImport,
    "RadianceCDLExport": RadianceCDLExport,
    "RadianceWhiteBalance": RadianceWhiteBalance,
    "RadianceColorSpaceConvert": RadianceColorSpaceConvert,
    "RadianceACESTransform": RadianceACESTransform,
    "RadianceHueCurves": RadianceHueCurves,
    "RadianceCurves": RadianceCurves,
    "RadianceGrade": RadianceGrade,
    "RadianceApplyGradeInfo": RadianceApplyGradeInfo,
    "RadianceGradeMatch": RadianceGradeMatch,
    "RadianceOCIOContext": RadianceOCIOContext,
    "RadianceQC": RadianceQC,
    # The v3 reorganisation transcribed these mapping dicts by hand and dropped
    # entries on the way. The classes were written, complete and importable the
    # whole time -- they simply never appeared in ComfyUI's node menu.
    "RadianceBitDepthDegrade": RadianceBitDepthDegrade,
    "RadiancePolicyGuard": RadiancePolicyGuard,
    "RadianceLUTApply": RadianceLUTApply,
    "RadianceLUTBlend": RadianceLUTBlend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceCDLTransform": "◎ Radiance CDL Transform",
    "RadianceCDLImport": "◎ Radiance CDL Import",
    "RadianceCDLExport": "◎ Radiance CDL Export",
    "RadianceWhiteBalance": "◎ Radiance White Balance",
    "RadianceColorSpaceConvert": "◎ Radiance Colorspace Convert",
    "RadianceACESTransform": "◎ Radiance ACES Transform",
    "RadianceHueCurves": "◎ Radiance Hue Curves",
    "RadianceCurves": "◎ Radiance Curves",
    "RadianceGrade": "◎ Radiance Grade",
    "RadianceApplyGradeInfo": "◎ Radiance Apply Grade Info",
    "RadianceGradeMatch": "◎ Radiance Grade Match",
    "RadianceOCIOContext": "◎ Radiance OCIO Context",
    "RadianceQC": "◎ Radiance QC",
    # These four were registered with no display name, so the branding
    # layer had to derive a label from the class key. Harmless in the menu,
    # but it made "every registered node has a display name" untestable.
    "RadiancePolicyGuard": "◎ Radiance Policy Guard",
    "RadianceLUTApply": "◎ Radiance LUT Apply",
    "RadianceLUTBlend": "◎ Radiance LUT Blend",
    "RadianceBitDepthDegrade": "◎ Radiance Bit-Depth Degrade",
}

# Publishing is the default: sweep this package for nodes its leaf modules
# declare but the mapping above does not list. See nodes/aggregate.py — this is
# what stopped seventeen finished nodes from ever reaching the menu.
fold_in_module_nodes(__name__, NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, logger)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
