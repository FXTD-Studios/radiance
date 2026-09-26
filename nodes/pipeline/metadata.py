import logging

logger = logging.getLogger("radiance.metadata")

class RadianceLinearCheck:
    """
    ◎ Radiance Linear Check
    
    Validates that the incoming image is tagged as 'Linear'.
    If not, it can optionally raise a warning or error.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image passed through unchanged. Its pixels are not inspected; only the shot metadata's colorspace tag is checked."}),
                "shot_metadata": ("RADIANCE_SHOT", {"tooltip": "Shot metadata whose 'colorspace' tag is tested. Tags containing 'Linear', or exactly 'ACEScg', pass; anything else (or a missing tag, read as 'Unknown') fails."}),
                "action": (["Log Warning", "Strict Error", "Ignore"], {"default": "Log Warning", "tooltip": "What to do when the tag is not linear: Log Warning writes to the console and continues, Strict Error stops the graph, Ignore does nothing."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "RADIANCE_SHOT")
    RETURN_NAMES = ("image", "shot_metadata")
    FUNCTION = "check"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ IO & Delivery"
    DESCRIPTION = "Check that the shot metadata tags the image as scene-linear (Linear or ACEScg) before linear-only nodes. It reads the metadata tag only, not the pixel values."

    def check(self, image, shot_metadata, action):
        cs = shot_metadata.get("colorspace", "Unknown")
        if cs != "Linear" and cs != "ACEScg" and "Linear" not in cs:
            msg = f"[Radiance Pipeline] WARNING: Non-linear colorspace '{cs}' detected in a linear-only node. This will cause incorrect grading results."
            if action == "Strict Error":
                raise ValueError(msg)
            elif action == "Log Warning":
                logger.warning(msg)
        
        return (image, shot_metadata)

NODE_CLASS_MAPPINGS = {
    "RadianceLinearCheck": RadianceLinearCheck,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceLinearCheck": "◎ Radiance Linear Check",
}
