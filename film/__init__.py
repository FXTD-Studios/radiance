from .camera import (
    RadianceDepthOfField,
    RadianceRollingShutter,
    RadianceCompressionArtifacts,
)

__all__ = [
    "RadianceDepthOfField",
    "RadianceRollingShutter",
    "RadianceCompressionArtifacts",
]

NODE_CLASS_MAPPINGS = {
    "RadianceDepthOfField": RadianceDepthOfField,
    "RadianceRollingShutter": RadianceRollingShutter,
    "RadianceCompressionArtifacts": RadianceCompressionArtifacts,
    
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceDepthOfField": "◎ Radiance Depth of Field",
    "RadianceRollingShutter": "◎ Radiance Rolling Shutter",
    "RadianceCompressionArtifacts": "◎ Radiance Compression Artifacts",
    
}
