from .utils import (
    tensor_to_numpy_float32,
    numpy_to_tensor_float32,
)

from .color import (
    ImageToFloat32,
    Float32ColorCorrect,
    ColorSpaceConvert,
    DaVinciWideGamut,
    ARRIWideGamut4,
    ACES2OutputTransform,
)

from .tonemap import HDRExpandDynamicRange, HDRToneMap

from .analysis import HDRHistogram

from .processing import HDRExposureBlend, HDRShadowHighlightRecovery, GPUTensorOps

from .panorama import HDR360Generate

from .recovery import RadianceHighlightSynthesis

from .ocio import ACESConfigManager, OCIOColorTransform, OCIOListColorspaces

# ── All keys use the Radiance prefix convention ───────────────────────────────
# Keys that previously lacked the prefix have been renamed to avoid collisions
# with other packages and to appear under the Radiance namespace in ComfyUI.

NODE_CLASS_MAPPINGS = {
    # hdr/color.py
    "RadianceFloat32Convert":           ImageToFloat32,
    "RadianceFloat32ColorCorrect":      Float32ColorCorrect,
    "RadianceHDRColorConvert":          ColorSpaceConvert,   # avoids clash w/ RadianceColorSpaceConvert
    "RadianceDaVinciWideGamut":         DaVinciWideGamut,
    "RadianceARRIWideGamut4":           ARRIWideGamut4,
    "RadianceACES2OutputTransform":     ACES2OutputTransform,
    # hdr/tonemap.py
    "RadianceHDRExpandDynamicRange":    HDRExpandDynamicRange,
    "RadianceHDRToneMap":               HDRToneMap,
    # hdr/analysis.py
    "RadianceHDRHistogram":             HDRHistogram,
    # hdr/processing.py
    "RadianceHDRExposureBlend":         HDRExposureBlend,
    "RadianceHDRShadowHighlight":       HDRShadowHighlightRecovery,
    "RadianceGPUTensorOps":             GPUTensorOps,
    # hdr/panorama.py
    "RadianceHDR360Generate":           HDR360Generate,
    # hdr/recovery.py
    "RadianceHighlightSynthesis":       RadianceHighlightSynthesis,
    # hdr/ocio.py
    "RadianceACESConfigManager":        ACESConfigManager,
    "RadianceHDROCIOTransform":         OCIOColorTransform,  # avoids clash w/ color/ RadianceOCIOColorTransform
    "RadianceOCIOListColorspaces":      OCIOListColorspaces,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceFloat32Convert":           "◎ Radiance Float32 Convert",
    "RadianceFloat32ColorCorrect":      "◎ Radiance Float32 Color Correct",
    "RadianceHDRColorConvert":          "◎ Radiance HDR Color Convert",
    "RadianceDaVinciWideGamut":         "◎ Radiance DaVinci Wide Gamut",
    "RadianceARRIWideGamut4":           "◎ Radiance ARRI Wide Gamut 4",
    "RadianceACES2OutputTransform":     "◎ Radiance ACES 2.0 Output Transform",
    "RadianceHDRExpandDynamicRange":    "◎ Radiance Expand Dynamic Range",
    "RadianceHDRToneMap":               "◎ Radiance HDR Tone Map",
    "RadianceHDRHistogram":             "◎ Radiance HDR Histogram",
    "RadianceHDRExposureBlend":         "◎ Radiance HDR Exposure Blend",
    "RadianceHDRShadowHighlight":       "◎ Radiance HDR Shadow / Highlight Recovery",
    "RadianceGPUTensorOps":             "◎ Radiance GPU Tensor Ops",
    "RadianceHDR360Generate":           "◎ Radiance HDR 360 Generate",
    "RadianceHighlightSynthesis":       "◎ Radiance Highlight Synthesis",
    "RadianceACESConfigManager":        "◎ Radiance ACES Config Manager",
    "RadianceHDROCIOTransform":         "◎ Radiance OCIO Transform (HDR)",
    "RadianceOCIOListColorspaces":      "◎ Radiance OCIO List Colorspaces",
}
