import torch
import logging

logger = logging.getLogger("radiance.overlay")


# ─────────────────────────────────────────────────────────────────────────────
#  RadianceBlendComposite  (v1.0)
# ─────────────────────────────────────────────────────────────────────────────
#  RadianceBlendComposite  (v1.0)
# ─────────────────────────────────────────────────────────────────────────────

BLEND_MODES = ["Normal", "Add", "Screen", "Multiply", "Overlay", "Soft Light", "Difference", "Divide"]


class RadianceBlendComposite:
    """
    Composite two images using industry-standard blend modes.
    All math is in scene-linear fp32; HDR values survive Add and Screen.
    An optional mask (MASK) controls compositing coverage per-pixel.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base":    ("IMAGE", {"tooltip": "Bottom layer (background)."}),
                "blend":   ("IMAGE", {"tooltip": "Top layer (foreground)."}),
                "mode":    (BLEND_MODES, {"default": "Normal", "tooltip": "Blend operation of blend over base, in fp32. Add, Multiply and Divide keep HDR values above 1.0; Screen clamps both layers to 0-1 first; Overlay and Soft Light assume 0-1 display-range values."}),
                "opacity": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                     "tooltip": "Overall strength of the blend layer."},
                ),
            },
            "optional": {
                "mask": (
                    "MASK",
                    {"tooltip": "Optional per-pixel mask (grayscale). White = full blend, Black = base only."},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "composite"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ VFX"
    DESCRIPTION = "Composite two images using Normal, Add, Screen, Multiply, Overlay, Soft Light, Difference, or Divide blend modes."

    def composite(
        self,
        base: torch.Tensor,
        blend: torch.Tensor,
        mode: str = "Normal",
        opacity: float = 1.0,
        mask: torch.Tensor = None,
    ) -> tuple:
        A = base.float()
        B = blend.float()

        # Match spatial dimensions — broadcast blend to base if needed
        if B.shape[1:3] != A.shape[1:3]:
            import torch.nn.functional as F
            B = F.interpolate(
                B.permute(0, 3, 1, 2),
                size=(A.shape[1], A.shape[2]),
                mode="bilinear", align_corners=False,
            ).permute(0, 2, 3, 1)

        # Match batch dimension
        if B.shape[0] == 1 and A.shape[0] > 1:
            B = B.expand_as(A)

        # ── Blend operations ────────────────────────────────────────────────
        if mode == "Normal":
            blended = B
        elif mode == "Add":
            blended = A + B
        elif mode == "Screen":
            blended = 1.0 - (1.0 - A.clamp(0, 1)) * (1.0 - B.clamp(0, 1))
        elif mode == "Multiply":
            blended = A * B
        elif mode == "Overlay":
            low  = 2.0 * A * B
            high = 1.0 - 2.0 * (1.0 - A) * (1.0 - B)
            blended = torch.where(A < 0.5, low, high)
        elif mode == "Soft Light":
            # Pegtop formula (no clamping)
            blended = (1.0 - 2.0 * B) * A * A + 2.0 * B * A
        elif mode == "Difference":
            blended = torch.abs(A - B)
        elif mode == "Divide":
            blended = A / (B + 1e-7)
        else:
            blended = B

        # ── Opacity + mask compositing ───────────────────────────────────────
        alpha = opacity
        if mask is not None:
            # mask shape: (B, H, W) or (H, W) → expand to (B, H, W, 1)
            m = mask.float()
            if m.dim() == 2:
                m = m.unsqueeze(0)
            m = m.unsqueeze(-1)  # (B, H, W, 1)
            if m.shape[0] == 1 and A.shape[0] > 1:
                m = m.expand(A.shape[0], -1, -1, -1)
            alpha_map = m * opacity
            out = A * (1.0 - alpha_map) + blended * alpha_map
        else:
            out = A * (1.0 - alpha) + blended * alpha

        return (out,)



NODE_CLASS_MAPPINGS = {
    "RadianceBlendComposite":  RadianceBlendComposite,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceBlendComposite":  "◎ Radiance Blend Composite",
}

