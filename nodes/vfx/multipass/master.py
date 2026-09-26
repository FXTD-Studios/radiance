import os
import json
import logging
import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

from ....performance import perf_finish, perf_start
from ....core.system.path_utils import get_safe_output_dir, safe_join, strip_path_quotes

# Core library imports
from .core import (
    _to_3ch_image, _luminance, _LUMA_WEIGHTS, _threshold_mask,
    _guided_filter_diffuse, _scharr_edges, _colorfulness,
    _ssao_multisampled, _reflection_mask, _albedo_retinex,
    _emission_glow, _roughness_from_specular, _transmission_mask,
    _optical_flow, _flow_to_hsv_image, _object_id_matte,
    _depth_anything_v2_infer, _normal_from_dsine, _surface_normals_gradient,
    _curvature_from_normals, _world_position_from_depth,
    _AUTO_DEPTH_CHOICES, _NORMAL_CONVENTIONS, _DA_CHOICE_TO_KEY,
    _highpass_filter, _metallic_mask
)

logger = logging.getLogger("radiance.vfx.multipass.master")


def _match_optional_image(x: torch.Tensor, batch: int, height: int, width: int) -> torch.Tensor:
    out = x.float()
    if out.shape[1] != height or out.shape[2] != width:
        out = F.interpolate(out.permute(0, 3, 1, 2), size=(height, width), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
    if out.shape[0] == batch:
        return out.contiguous()
    if out.shape[0] == 1:
        return out.expand(batch, -1, -1, -1).contiguous()
    if out.shape[0] > batch:
        return out[:batch].contiguous()
    raise ValueError(f"Batch mismatch: expected {batch} frames or a single broadcast frame, got {out.shape[0]}")

# Try importing EXR writer utilities from io/hdr
try:
    from ....hdr.io import write_exr_multipart, write_exr_openexr
    _HAS_EXR = True
except ImportError:
    try:
        from hdr.io import write_exr_multipart, write_exr_openexr  # type: ignore[import]
        _HAS_EXR = True
    except ImportError:
        _HAS_EXR = False

# Try importing ComfyUI folder_paths
try:
    import folder_paths  # type: ignore
    _HAS_FOLDER_PATHS = True
except ImportError:
    _HAS_FOLDER_PATHS = False


def _workflow_metadata(prompt=None, extra_pnginfo=None) -> Dict[str, Any]:
    """
    Build the ComfyUI provenance metadata dict ({"prompt": json, "workflow":
    json, ...}) exactly as core SaveImage embeds into PNG tEXt chunks and as
    RadianceWrite embeds into EXR headers, so multipass/AOV EXRs carry the
    workflow like a saved PNG does too. Values are JSON strings; write_exr_openexr
    / write_exr_multipart store "workflow"/"prompt" as unprefixed EXR string
    attributes (see hdr/io.py).
    """
    meta: Dict[str, Any] = {}
    try:
        if prompt is not None:
            meta["prompt"] = json.dumps(prompt)
        if extra_pnginfo:
            for key, value in extra_pnginfo.items():
                meta[str(key)] = json.dumps(value)
    except (TypeError, ValueError) as exc:
        logger.warning("[EXR Passes Writer] workflow metadata not serialisable: %s", exc)
    return meta


def _generate_cryptomatte_manifest(K: int) -> str:
    import json
    phi = 1.6180339887
    manifest = {}
    for k in range(K):
        hue = (k * phi) % 1.0
        sat, val = 0.85, 0.90
        h6  = hue * 6.0
        idx = int(h6) % 6
        f   = h6 - int(h6)
        p_  = val * (1 - sat)
        q_  = val * (1 - f * sat)
        t_  = val * (1 - (1 - f) * sat)
        lut = [(val,t_,p_),(q_,val,p_),(p_,val,t_),(p_,q_,val),(t_,p_,val),(val,p_,q_)]
        r, g, b = lut[idx]

        # Convert float RGB to 8-bit hex color string
        hex_color = f"{int(round(r*255)):02x}{int(round(g*255)):02x}{int(round(b*255)):02x}"
        manifest[f"cluster_{k}"] = hex_color
    return json.dumps(manifest)


def _image_to_3ch(image: torch.Tensor) -> torch.Tensor:
    x = image.float()
    if x.shape[-1] == 3:
        return x.contiguous()
    if x.shape[-1] > 3:
        return x[..., :3].contiguous()
    if x.shape[-1] == 1:
        return x.expand(-1, -1, -1, 3).contiguous()
    pad = x[..., -1:].expand(-1, -1, -1, 3 - x.shape[-1])
    return torch.cat([x, pad], dim=-1).contiguous()


def _pass_channels_for_multilayer(name: str, image: np.ndarray) -> Dict[str, np.ndarray]:
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[..., np.newaxis]

    channels: Dict[str, np.ndarray] = {}
    count = arr.shape[2] if arr.ndim == 3 else 1
    if name == "beauty":
        names = ["R", "G", "B", "A"][:count]
    elif name == "depth":
        # ALBABIT-FIX: was ["depth.R","depth.G","depth.B"] since extract() always
        # triples depth to 3 identical channels for ComfyUI's IMAGE type -- no
        # compositor auto-recognizes "depth.R/G/B" as a Z-depth buffer. Bare "Z"
        # (matching write_exr_multipart()'s already-correct convention) is what
        # Nuke/Fusion look for; the duplicate channels carry no extra data anyway.
        names = ["Z"]
    elif name == "normal":
        names = ["normal.NX", "normal.NY", "normal.NZ", "normal.A"][:count]
    elif name == "motion_vector":
        names = ["MV.X", "MV.Y"][:count]
    elif name == "alpha":
        names = ["A"]
    else:
        names = [f"{name}.{ch}" for ch in ["R", "G", "B", "A"][:count]]

    for idx, channel_name in enumerate(names):
        channels[channel_name] = arr[..., idx if idx < count else 0]
    return channels


#: Frames written at once by the EXR Passes Writer. OpenEXR compresses outside
#: the GIL, so this scales with cores; each worker holds one frame's passes.
_EXR_WRITE_THREADS = max(1, min(8, os.cpu_count() or 1))


def _write_exr_singlepart_multilayer(
    filepath: str,
    parts: Dict[str, np.ndarray],
    bit_depth: str,
    compression: str,
    metadata: Dict[str, Any],
) -> bool:
    channels: Dict[str, np.ndarray] = {}
    for name, image in parts.items():
        if image is None:
            continue
        channels.update(_pass_channels_for_multilayer(name, image))

    if not channels:
        return False

    pixel_type = "HALF" if "16" in bit_depth else "FLOAT"
    write_exr_openexr(filepath, channels, compression, pixel_type, metadata)
    return True


_LEGACY_MULTIPASS_STOP = (
    "Multipass Extract is retired in Radiance 3.5.0. Its albedo, specular, roughness, "
    "metallic, emission, transmission, reflection and object-ID passes were image filters, "
    "not measurements, and its depth-derived passes had no real scale. Use 'Multipass "
    "Estimate' (MoGe-2 geometry, Marigold materials and lighting) for passes from a plate, "
    "or 'Read AOVs' for renderer passes."
)


class RadianceMultipassMaster:
    CATEGORY = "FXTD STUDIOS/Radiance/VFX"
    DESCRIPTION = "Retired: use Multipass Estimate. Kept so saved graphs open and explain the replacement."
    DEPRECATED = True   # hidden from the menu; saved graphs still open

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "beauty": ("IMAGE",),
            },
            "optional": {
                "depth_map": ("IMAGE",),
                "normal_map": ("IMAGE",),
                "prev_frame": ("IMAGE",),
                "source_passes": ("RADIANCE_PASSES",),

                # General Settings
                "luma_weights": (list(_LUMA_WEIGHTS.keys()), {"default": "Rec.709 / sRGB"}),

                # Depth (Auto-download) Settings
                "auto_depth_model": (_AUTO_DEPTH_CHOICES, {"default": "disabled"}),
                "depth_near_is_white": ("BOOLEAN", {"default": True}),
                "depth_scale": ("FLOAT", {"default": 10.0, "min": 0.01, "max": 1000.0, "step": 0.1}),
                "fov_degrees": ("FLOAT", {"default": 60.0, "min": 10.0, "max": 120.0, "step": 1.0}),

                # Normals Settings
                "dsine_model_path": ("STRING", {"default": "auto"}),
                "normal_strength": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 20.0, "step": 0.1}),
                "normal_convention": (_NORMAL_CONVENTIONS, {"default": "OpenGL (Y-Up)"}),

                # PBR Settings
                "albedo_shading_radius": ("FLOAT", {"default": 80.0, "min": 10.0, "max": 300.0, "step": 5.0}),
                "albedo_eps": ("FLOAT", {"default": 0.001, "min": 0.0001, "max": 0.1, "step": 0.0005}),
                "specular_floor": ("BOOLEAN", {"default": True}),
                "roughness_fine_radius": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 10.0, "step": 0.5}),
                "roughness_coarse_radius": ("FLOAT", {"default": 15.0, "min": 3.0, "max": 60.0, "step": 1.0}),
                "transmission_sensitivity": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 10.0, "step": 0.25}),

                # Highpass Settings
                "highpass_radius": ("FLOAT", {"default": 8.0, "min": 0.1, "max": 100.0, "step": 0.5}),
                "highpass_strength": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "highpass_contrast": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),

                # Lighting & Mask Settings
                "shadow_threshold": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 0.60, "step": 0.01}),
                "highlight_threshold": ("FLOAT", {"default": 0.75, "min": 0.40, "max": 1.0, "step": 0.01}),
                "mask_softness": ("FLOAT", {"default": 0.15, "min": 0.01, "max": 0.50, "step": 0.01}),
                "ao_radius": ("FLOAT", {"default": 15.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "ao_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.1}),
                "ao_samples": ("INT", {"default": 8, "min": 4, "max": 32, "step": 4}),

                # Motion & ID Settings
                "lk_window_radius": ("INT", {"default": 7, "min": 1, "max": 32}),
                "motion_coherence": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 0.95, "step": 0.05}),
                "batch_is_sequence": ("BOOLEAN", {"default": False}),
                "object_id_segments": ("INT", {"default": 16, "min": 2, "max": 64}),
                "object_id_spatial_weight": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 2.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = (
        "RADIANCE_PASSES", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE",
        "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE",
        "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE"
    )
    RETURN_NAMES = (
        "passes", "beauty", "albedo", "normal_map", "depth", "roughness", "specular",
        "metallic", "ao", "emission", "transmission", "highpass", "world_position",
        "curvature", "shadow_mask", "midtone_mask", "highlight_mask", "reflection_mask",
        "motion_vector", "segmentation_id", "motion_visualization", "alpha"
    )

    FUNCTION = "extract"

    def extract(self, beauty: torch.Tensor, **kwargs) -> Tuple:
        raise RuntimeError(f"[Multipass Extract] {_LEGACY_MULTIPASS_STOP}")

    def _legacy_extract(
        self,
        beauty: torch.Tensor,
        depth_map: Optional[torch.Tensor] = None,
        normal_map: Optional[torch.Tensor] = None,
        prev_frame: Optional[torch.Tensor] = None,
        source_passes: Optional[Dict[str, Any]] = None,
        luma_weights: str = "Rec.709 / sRGB",
        auto_depth_model: str = "disabled",
        depth_near_is_white: bool = True,
        depth_scale: float = 10.0,
        fov_degrees: float = 60.0,
        dsine_model_path: str = "auto",
        normal_strength: float = 2.0,
        normal_convention: str = "OpenGL (Y-Up)",
        albedo_shading_radius: float = 80.0,
        albedo_eps: float = 0.001,
        specular_floor: bool = True,
        roughness_fine_radius: float = 2.0,
        roughness_coarse_radius: float = 15.0,
        transmission_sensitivity: float = 2.0,
        highpass_radius: float = 8.0,
        highpass_strength: float = 1.0,
        highpass_contrast: float = 1.0,
        shadow_threshold: float = 0.20,
        highlight_threshold: float = 0.75,
        mask_softness: float = 0.15,
        ao_radius: float = 15.0,
        ao_strength: float = 1.0,
        ao_samples: int = 8,
        lk_window_radius: int = 7,
        motion_coherence: float = 0.5,
        batch_is_sequence: bool = False,
        object_id_segments: int = 16,
        object_id_spatial_weight: float = 0.25,
    ) -> Tuple:
        img = _image_to_3ch(beauty)
        B, H, W, _ = img.shape
        device = beauty.device
        _perf = perf_start(device)
        weights = _LUMA_WEIGHTS.get(luma_weights, _LUMA_WEIGHTS["Rec.709 / sRGB"])
        luma = _luminance(img, weights)
        source_present = set(source_passes.get("_present", ())) if source_passes else set()
        if source_passes and not source_present:
            source_present = {k for k, v in source_passes.items() if isinstance(v, torch.Tensor) and not k.startswith("_")}
        if depth_map is None and source_passes and "depth" in source_present:
            depth_map = source_passes.get("depth")
        if normal_map is None and source_passes and "normal" in source_present:
            normal_map = source_passes.get("normal")

        # ── 1. Depth Map ──────────────────────────────────────────────────────
        depth_scalar: Optional[torch.Tensor] = None
        if depth_map is not None:
            d_in = _match_optional_image(depth_map, B, H, W)
            pass_depth = d_in.contiguous()
            depth_scalar = pass_depth[..., 0]
        elif auto_depth_model != "disabled":
            da_key = _DA_CHOICE_TO_KEY.get(auto_depth_model)
            if da_key:
                auto_d = _depth_anything_v2_infer(img, model_key=da_key)
                if auto_d is not None:
                    pass_depth = _to_3ch_image(auto_d.to(device))
                    depth_scalar = auto_d.to(device)
                else:
                    pass_depth = torch.zeros(B, H, W, 3, device=device, dtype=torch.float32)
            else:
                pass_depth = torch.zeros(B, H, W, 3, device=device, dtype=torch.float32)
        else:
            pass_depth = torch.zeros(B, H, W, 3, device=device, dtype=torch.float32)

        # ── 2. Normals ────────────────────────────────────────────────────────
        if normal_map is not None:
            n_in = _match_optional_image(normal_map, B, H, W)
            signed_n = n_in[..., :3]
            if float(signed_n.detach().min()) >= -0.001 and float(signed_n.detach().max()) <= 1.001:
                signed_n = signed_n * 2.0 - 1.0
            signed_n = signed_n / torch.sqrt((signed_n * signed_n).sum(dim=-1, keepdim=True).clamp(min=1e-8))
            pass_normal = (signed_n * 0.5 + 0.5).contiguous()
            normal_source = "input normal_map"
        else:
            dsine_result = _normal_from_dsine(img, dsine_model_path, normal_convention)
            if dsine_result is not None:
                pass_normal = dsine_result
                normal_source = "DSINE"
            else:
                pass_normal = _surface_normals_gradient(luma, normal_strength, normal_convention)
                # Recorded in the passes so the fallback is visible downstream
                # (it used to be a log line only).
                normal_source = "luma gradient (DSINE unavailable)"

        # ── 3. Geometry Utilities ─────────────────────────────────────────────
        pass_curvature = _curvature_from_normals(pass_normal)

        if depth_scalar is not None:
            pass_world_pos = _world_position_from_depth(depth_scalar, fov_degrees, depth_scale, depth_near_is_white)
        else:
            pass_world_pos = torch.full((B, H, W, 3), 0.5, device=device, dtype=torch.float32)

        # ── 4. Diffuse & Specular ─────────────────────────────────────────────
        pass_diffuse = _guided_filter_diffuse(img, 20.0, eps=0.01)
        pass_specular = img - pass_diffuse
        if specular_floor:
            pass_specular = pass_specular.clamp(min=0.0)

        # ── 5. Edge & Colorfulness ────────────────────────────────────────────
        edge_map = _scharr_edges(luma)
        colorf_map = _colorfulness(img, weights)

        # ── 6. Tone Masks ─────────────────────────────────────────────────────
        shadow_m = _threshold_mask(luma, shadow_threshold, mask_softness, above=False)
        highlight_m = _threshold_mask(luma, highlight_threshold, mask_softness, above=True)
        midtone_m = (1.0 - shadow_m - highlight_m).clamp(min=0.0)
        _sum = (shadow_m + midtone_m + highlight_m).clamp(min=1e-8)

        pass_shadow = _to_3ch_image(shadow_m / _sum)
        pass_midtone = _to_3ch_image(midtone_m / _sum)
        pass_highlight = _to_3ch_image(highlight_m / _sum)

        # ── 7. Lighting & Occlusion ───────────────────────────────────────────
        if depth_scalar is not None and ao_strength > 0.0:
            ao_map = _ssao_multisampled(depth_scalar, pass_normal, ao_radius, ao_strength, ao_samples, depth_near_is_white)
            pass_ao = _to_3ch_image(ao_map)
        else:
            pass_ao = torch.zeros(B, H, W, 3, device=device, dtype=torch.float32)

        pass_reflection = _reflection_mask(pass_specular, colorf_map)

        # ── 8. Material & PBR Passes ──────────────────────────────────────────
        pass_metallic = _metallic_mask(img, pass_specular, colorf_map)
        pass_albedo = _albedo_retinex(img, luma, shading_radius=albedo_shading_radius, eps=albedo_eps)
        pass_emission = _emission_glow(luma, colorf_map, radius=albedo_shading_radius * 0.375, boost=1.5)
        pass_roughness = _roughness_from_specular(pass_specular, fine_radius=roughness_fine_radius, coarse_radius=roughness_coarse_radius)
        pass_transmission = _transmission_mask(img, luma, edge_map, highlight_m, colorf_map, sensitivity=transmission_sensitivity)

        # ── 9. Highpass Filter ────────────────────────────────────────────────
        pass_highpass = _highpass_filter(img, highpass_radius, highpass_strength, highpass_contrast)

        # ── 10. Motion Vectors ────────────────────────────────────────────────
        if prev_frame is not None:
            pf_in = _match_optional_image(prev_frame, B, H, W)
            prev_luma = _luminance(pf_in, weights)
            # `auto` is DIS where OpenCV is importable, which is every supported
            # install; lk_window_radius still governs the Lucas-Kanade fallback.
            flow_u, flow_v = _optical_flow(
                luma, prev_luma, method="auto", window_radius=lk_window_radius)

            # Apply temporal coherence / EMA smoothing to reduce sub-pixel jitter
            if motion_coherence > 0.0 and batch_is_sequence:
                smooth_u = torch.zeros_like(flow_u)
                smooth_v = torch.zeros_like(flow_v)

                for b in range(B):
                    if b == 0:
                        smooth_u[0] = flow_u[0]
                        smooth_v[0] = flow_v[0]
                    else:
                        smooth_u[b] = flow_u[b] * (1.0 - motion_coherence) + smooth_u[b-1] * motion_coherence
                        smooth_v[b] = flow_v[b] * (1.0 - motion_coherence) + smooth_v[b-1] * motion_coherence

                flow_u = smooth_u
                flow_v = smooth_v

            pass_motion = torch.stack([flow_u, flow_v, torch.zeros_like(flow_u)], dim=-1)
            pass_motion_viz = _flow_to_hsv_image(flow_u, flow_v)
        else:
            pass_motion = torch.zeros(B, H, W, 3, device=device, dtype=torch.float32)
            pass_motion_viz = torch.zeros_like(pass_motion)

        # ── 11. Object ID / Cryptomatte ───────────────────────────────────────
        pass_object_id = _object_id_matte(img, luma, n_segments=object_id_segments, spatial_weight=object_id_spatial_weight)

        out_beauty = img.contiguous()
        out_albedo = pass_albedo.contiguous()
        out_normal = pass_normal.contiguous()
        out_depth = pass_depth.contiguous()
        out_roughness = pass_roughness.contiguous()
        out_specular = pass_specular.contiguous()
        out_metallic = pass_metallic.contiguous()
        out_ao = pass_ao.contiguous()
        out_emission = pass_emission.contiguous()
        out_transmission = pass_transmission.contiguous()
        out_highpass = pass_highpass.contiguous()
        out_world_pos = pass_world_pos.contiguous()
        out_curvature = pass_curvature.contiguous()
        out_shadow = pass_shadow.contiguous()
        out_midtone = pass_midtone.contiguous()
        out_highlight = pass_highlight.contiguous()
        out_reflection = pass_reflection.contiguous()
        out_motion = pass_motion.contiguous()
        out_motion_viz = pass_motion_viz.contiguous()
        out_object_id = pass_object_id.contiguous()
        out_alpha = (
            _match_optional_image(source_passes["alpha"], B, H, W)[..., :1]
            if source_passes and isinstance(source_passes.get("alpha"), torch.Tensor)
            else beauty.float()[..., 3:4] if beauty.shape[-1] >= 4
            else torch.ones(B, H, W, 1, device=device, dtype=torch.float32)
        )
        out_alpha = out_alpha.expand(-1, -1, -1, 3).contiguous()

        # ── Bundle dictionary ──
        passes_dict = {
            "beauty": out_beauty,
            "albedo": out_albedo,
            "normal": out_normal,
            "depth": out_depth,
            "roughness": out_roughness,
            "specular": out_specular,
            "metallic": out_metallic,
            "ao": out_ao,
            "emission": out_emission,
            "transmission": out_transmission,
            "highpass": out_highpass,
            "world_position": out_world_pos,
            "curvature": out_curvature,
            "shadow_mask": out_shadow,
            "midtone_mask": out_midtone,
            "highlight_mask": out_highlight,
            "reflection_mask": out_reflection,
            "motion_vector": out_motion,
            "motion_visualization": out_motion_viz,
            "object_id": out_object_id,
            "alpha": out_alpha,
        }
        if source_passes:
            for name in source_present:
                value = source_passes.get(name)
                if name in passes_dict and isinstance(value, torch.Tensor):
                    passes_dict[name] = _match_optional_image(value, B, H, W)
            for name, value in source_passes.items():
                if name.startswith("_") and name != "_present":
                    passes_dict[name] = value
            passes_dict["_source_present"] = sorted(source_present)
        passes_dict["_normal_source"] = normal_source
        passes_dict["_present"] = sorted(name for name, value in passes_dict.items() if isinstance(value, torch.Tensor))

        perf_finish(logger, "Multipass Extract", _perf, device)
        return (
            passes_dict, passes_dict["beauty"], passes_dict["albedo"], passes_dict["normal"],
            passes_dict["depth"], passes_dict["roughness"], passes_dict["specular"],
            passes_dict["metallic"], passes_dict["ao"], passes_dict["emission"],
            passes_dict["transmission"], passes_dict["highpass"], passes_dict["world_position"],
            passes_dict["curvature"], passes_dict["shadow_mask"], passes_dict["midtone_mask"],
            passes_dict["highlight_mask"], passes_dict["reflection_mask"], passes_dict["motion_vector"],
            passes_dict["object_id"], passes_dict["motion_visualization"], passes_dict["alpha"]
        )


class RadianceEXRPassesWriter:
    CATEGORY = "FXTD STUDIOS/Radiance/Load & Save"
    DESCRIPTION = "Write all passes inside the RADIANCE_PASSES bundle into a single-part multilayer or true multi-part OpenEXR file."
    FUNCTION = "write_passes"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        compressions = [
            "ZIP", "ZIPS", "PIZ", "RLE", "Uncompressed",
            "PXR24", "B44", "B44A", "DWAA", "DWAB"
        ]
        return {
            "required": {
                "passes": ("RADIANCE_PASSES", {"tooltip": "Pass bundle to write; must contain a beauty pass, and every pass must match its width and height. Pixel values are written as is, with no colour conversion."}),
                "filename_prefix": ("STRING", {"default": "radiance_vfx_passes", "tooltip": "File name stem, not a path. Files are named <prefix>.<frame>.exr with a 4-digit padded frame number."}),
                "bit_depth": (["16-bit Half Float", "32-bit Float"], {"default": "16-bit Half Float", "tooltip": "EXR pixel type. Files containing depth, world_position, motion_vector or object_id passes are always written as 32-bit float."}),
                "compression": (compressions, {"default": "ZIP", "tooltip": "EXR compression. ZIP, ZIPS, PIZ, RLE and Uncompressed are lossless; lossy B44/B44A/DWAA/DWAB raise an error when data passes (depth, position, motion, object ID) are present."}),
            },
            "optional": {
                "output_path": ("STRING", {"default": "", "tooltip": "Output folder. Empty = ComfyUI output folder; a relative path is a subfolder of it; an absolute path is used as is. Created if missing."}),
                "remote_path": ("STRING", {"default": "", "tooltip": "Optional absolute folder (for example a NAS share) that each written file is also copied to. A failed copy only logs a warning."}),
                "frame_index": ("INT", {"default": 1001, "min": 0, "max": 999999, "tooltip": "Frame number of the first image in the batch; later images count up from it."}),
                "exr_layout": (["Single-part multilayer", "Multi-part"], {"default": "Single-part multilayer", "tooltip": "Single-part multilayer: all passes as named channel layers in one part (widest compatibility). Multi-part: one EXR 2.0 part per pass."}),
                "custom_metadata": ("STRING", {"default": "", "multiline": True, "tooltip": "Extra EXR header attributes, one key=value per line, stored as strings. Lines without = are ignored."}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    def write_passes(
        self,
        passes: Dict[str, torch.Tensor],
        filename_prefix: str,
        bit_depth: str = "16-bit Half Float",
        compression: str = "ZIP",
        output_path: str = "",
        remote_path: str = "",
        frame_index: int = 1001,
        exr_layout: str = "Single-part multilayer",
        custom_metadata: str = "",
        prompt: Optional[Any] = None,
        extra_pnginfo: Optional[Any] = None,
    ) -> Tuple[str]:
        import datetime
        import tempfile
        import shutil

        output_path = strip_path_quotes(output_path)
        remote_path = strip_path_quotes(remote_path)

        # Determine target directory
        if _HAS_FOLDER_PATHS:
            base_dir = folder_paths.get_output_directory()
        else:
            base_dir = tempfile.gettempdir()

        # ALBABIT-FIX: was hand-rolled (os.path.isabs check + manual join),
        # with no anti-traversal protection for relative output_path values.
        # get_safe_output_dir() is the same helper RadianceWrite/RadianceEXRMultiPart
        # already use for this exact pattern.
        out_dir = get_safe_output_dir(base_dir, output_path, allow_absolute=True)

        prefix = filename_prefix.strip()
        if not prefix or prefix in (".", "..") or os.path.basename(prefix) != prefix:
            raise ValueError("[EXR Passes Writer] filename_prefix must be a filename, not a path.")

        # Standard Metadata dictionary
        meta: Dict[str, Any] = {
            "software": "Radiance VFX Multipass v3.1",
            "created": datetime.datetime.now().isoformat(),
        }

        # Custom user-defined metadata lines
        for line in custom_metadata.strip().split("\n"):
            if "=" in line:
                k, v = line.split("=", 1)
                meta[k.strip()] = v.strip()

        # ComfyUI workflow/prompt provenance — same convention as RadianceWrite,
        # so multipass EXRs can be dragged back into ComfyUI to restore the graph.
        meta.update(_workflow_metadata(prompt, extra_pnginfo))

        # Retrieve beauty shape to detect batch size
        beauty = passes.get("beauty")
        if not isinstance(beauty, torch.Tensor):
            raise ValueError("[EXR Passes Writer] passes dictionary must contain a 'beauty' tensor.")
        if beauty.dim() != 4 or beauty.shape[-1] not in (1, 2, 3, 4):
            raise ValueError("[EXR Passes Writer] beauty must have shape (B,H,W,C).")

        B, H, W, _ = beauty.shape
        # 3.5.0: passes are converted one frame at a time as each file is
        # written (a GPU or half-precision pass used to be copied whole to the
        # CPU first), and frames are written in parallel: OpenEXR compresses
        # outside the GIL, so the files no longer queue behind each other.
        # Each file's contents are unchanged.
        pass_tensors: Dict[str, torch.Tensor] = {}
        for name, tensor in passes.items():
            if name.startswith("_") or not isinstance(tensor, torch.Tensor):
                continue
            if tensor.dim() not in (3, 4):
                raise ValueError(f"[EXR Passes Writer] pass '{name}' must have shape (H,W,C) or (B,H,W,C).")
            shape = tensor.shape[-3:]
            if shape[0] != H or shape[1] != W or shape[2] not in (1, 2, 3, 4):
                raise ValueError(f"[EXR Passes Writer] pass '{name}' does not match beauty dimensions/channels.")
            if tensor.dim() == 4 and tensor.shape[0] not in (1, B):
                raise ValueError(f"[EXR Passes Writer] pass '{name}' batch must be 1 or {B}, got {tensor.shape[0]}.")
            pass_tensors[name] = tensor

        data_passes = {"depth", "world_position", "motion_vector", "object_id"}
        if data_passes.intersection(pass_tensors):
            if compression.upper() in {"B44", "B44A", "DWAA", "DWAB"}:
                raise ValueError("[EXR Passes Writer] lossy compression is not allowed with data AOVs; use ZIP, ZIPS, PIZ, RLE, or Uncompressed.")
            if "16" in bit_depth:
                logger.info("[EXR Passes Writer] Promoting data-AOV file to 32-bit float.")
                bit_depth = "32-bit Float"

        if not _HAS_EXR or (exr_layout != "Single-part multilayer" and write_exr_multipart is None):
            raise RuntimeError(
                "[EXR Passes Writer] EXR writing modules not found or import failed."
            )
        # Normalise compression label
        comp = "None" if compression.lower() == "uncompressed" else compression

        def write_frame(b: int) -> str:
            frame_num = str(frame_index + b).zfill(4)
            filename = f"{prefix}.{frame_num}.exr"
            filepath = safe_join(out_dir, filename)

            parts: Dict[str, np.ndarray] = {}
            for name, tensor in pass_tensors.items():
                frame = tensor if tensor.dim() == 3 else tensor[0 if tensor.shape[0] == 1 else b]
                parts[name] = frame.detach().float().cpu().numpy()

            # Write the EXR file
            if exr_layout == "Single-part multilayer":
                ok = _write_exr_singlepart_multilayer(filepath, parts, bit_depth, comp, meta)
            else:
                ok = write_exr_multipart(filepath, parts, bit_depth, comp, meta)
            if not ok:
                raise RuntimeError(f"[EXR Passes Writer] Failed to write EXR file: {filepath}")

            logger.info("[EXR Passes Writer] Saved %d layers → %s", len(parts), filepath)

            # Best effort copy to remote/NAS path
            if remote_path:
                try:
                    remote_dir = get_safe_output_dir(
                        strip_path_quotes(remote_path).strip(), "", allow_absolute=True)
                    os.makedirs(remote_dir, exist_ok=True)
                    dest = safe_join(remote_dir, filename)
                    shutil.copy2(filepath, dest)
                    logger.info("[EXR Passes Writer] Copied to remote → %s", dest)
                except Exception as ex:
                    logger.warning("[EXR Passes Writer] Remote copy failed: %s", ex)
            return filepath

        workers = max(1, min(B, _EXR_WRITE_THREADS))
        if workers == 1:
            saved_paths: List[str] = [write_frame(b) for b in range(B)]
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="radiance-exr") as pool:
                saved_paths = list(pool.map(write_frame, range(B)))   # frame order; first error raises

        return (saved_paths[0] if saved_paths else "",)


NODE_CLASS_MAPPINGS = {
    "RadianceMultipassMaster": RadianceMultipassMaster,
    "RadianceEXRPassesWriter": RadianceEXRPassesWriter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceMultipassMaster": "Multipass Extract (Legacy)",
    "RadianceEXRPassesWriter": "Write EXR Passes"
}
