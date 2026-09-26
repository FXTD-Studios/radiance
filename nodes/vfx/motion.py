import torch
import logging
import json

try:
    from radiance.nodes.vfx.multipass.core import _flow_to_hsv_image, _optical_flow
except Exception:
    from .multipass.core import _flow_to_hsv_image, _optical_flow
from radiance.core.tensor.chunking import chunks, frames_per_chunk

logger = logging.getLogger("radiance.motion")

class RadianceOpticalFlow:
    """
    ◎ Radiance Optical Flow
    
    Generates high-precision 32-bit UV motion vectors between consecutive frames.
    Compatible with Nuke's VectorBlur and Radiance Motion Coherence patches.
    
    Uses DIS (Dense Inverse Search), which this docstring claimed for some
    time while the node actually ran pyramidal Lucas-Kanade. It does now.
    Lucas-Kanade is still selectable, and is the fallback if OpenCV cannot be
    imported.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Batch of frames to analyze."}),
                "preset": (["Fast", "Medium", "Ultra"], {"default": "Medium",
                    "tooltip": "Solver effort. DIS: ultrafast, fast or medium preset (mostly a speed trade, similar accuracy). Lucas-Kanade: window radius 3, 5 or 9 px."}),
                "flow_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1,
                    "tooltip": "Scale factor for output vectors. 1.0 = pixel units."}),
                "visualize": ("BOOLEAN", {"default": False,
                    "tooltip": "Outputs a color-coded visualization of the motion field."}),
            },
            "optional": {
                "solver": (["Auto", "DIS", "Lucas-Kanade"], {
                    "default": "Auto",
                    "tooltip": (
                        "Auto uses DIS, and falls back to Lucas-Kanade only if "
                        "OpenCV is missing.\n"
                        "DIS holds a dense field out to about 20 px of motion "
                        "and is roughly 11x faster.\n"
                        "Lucas-Kanade is the previous solver: accurate to about "
                        "8 px, after which the field thins out even though the "
                        "median stays close."
                    ),
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("motion_vectors", "visualization", "stats")
    FUNCTION = "analyze"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ VFX"
    DESCRIPTION = "Estimate dense optical flow between adjacent frames."

    @torch.no_grad()
    def analyze(self, images: torch.Tensor, preset: str, flow_scale: float,
                visualize: bool, solver: str = "Auto"):
        B, H, W, C = images.shape
        device = images.device
        
        if B < 2:
            empty = torch.zeros((B, H, W, 3), device=device)
            return (empty, empty, json.dumps({"error": "Batch size must be >= 2"}))

        # One dial, two solvers. Lucas-Kanade reads it as an integration
        # radius; DIS reads it as its own preset. Mapped to all three of DIS's
        # presets rather than to its default, because otherwise Fast, Medium
        # and Ultra are three positions that do the same thing -- measured at
        # 1.4, 3.6 and 8.6 ms for the same accuracy on a 160x384 frame, so what
        # the dial buys under DIS is speed rather than precision.
        radius = {"Fast": 3, "Medium": 5, "Ultra": 9}.get(preset, 5)
        dis_preset = {"Fast": "ultrafast", "Medium": "fast",
                      "Ultra": "medium"}.get(preset, "fast")
        luma = (0.2126 * images[..., 0] + 0.7152 * images[..., 1] + 0.0722 * images[..., 2])
        luma_norm = (torch.log1p(luma.clamp(min=0.0) * 10.0) / 2.4).clamp(0.0, 1.0)

        method = {"Auto": "auto", "DIS": "dis",
                  "Lucas-Kanade": "lucas-kanade"}.get(solver, "auto")

        # 3.5.0: frame pairs go to DIS a chunk at a time, so it builds its
        # solver once per chunk instead of once per pair, and results are
        # written into the outputs as they come (the per-frame lists and the
        # stack were two copies of each output). Every pair is solved and
        # visualised on its own as before.
        vectors_tensor = torch.zeros((B, H, W, 3), device=device, dtype=torch.float32)
        visuals_tensor = torch.zeros((B, H, W, 3), device=device, dtype=torch.float32)
        # Lucas-Kanade stays one pair per call: its batched solve does not give
        # exactly the per-pair result (up to 0.07 px apart), and it is the
        # fallback for an install without OpenCV.
        use_lk = method == "lucas-kanade"
        if method == "auto":
            try:
                import cv2  # noqa: F401,PLC0415
            except ImportError:
                use_lk = True
        per = 1 if use_lk else frames_per_chunk(H, W, 1, 24.0, device)
        for a, b in chunks(B - 1, per):
            u, v = _optical_flow(luma_norm[a + 1:b + 1], luma_norm[a:b], method=method,
                                 window_radius=radius, preset=dis_preset)
            u = u * flow_scale
            v = v * flow_scale
            vectors_tensor[a + 1:b + 1, ..., 0] = u
            vectors_tensor[a + 1:b + 1, ..., 1] = v
            if visualize:
                visuals_tensor[a + 1:b + 1] = _flow_to_hsv_image(u, v)
            del u, v

        stats = json.dumps({
            "frames": B,
            "resolution": f"{W}x{H}",
            "preset": preset,
            "mean_mag": float(torch.norm(vectors_tensor, dim=-1).mean().item())
        })
        
        logger.info(f"[Optical Flow] Analyzed {B} frames. Mean motion: {json.loads(stats)['mean_mag']:.2f}px")
        
        return (vectors_tensor, visuals_tensor, stats)

NODE_CLASS_MAPPINGS = {
    "RadianceOpticalFlow": RadianceOpticalFlow,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceOpticalFlow": "◎ Radiance Optical Flow",
}
