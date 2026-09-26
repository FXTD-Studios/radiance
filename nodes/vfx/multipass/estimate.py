"""Multipass Estimate: render-style passes from a plate, from trained models only.

This replaces Multipass Extract, whose material and lighting passes were
image filters (Retinex albedo, "specular" as image minus a blur, roughness
from local contrast, masks from saturation and brightness). Those looked like
AOVs but measured nothing. Every pass here comes from a model trained to
predict that quantity, or is computed from those predictions with the maths a
renderer would use.

Geometry, MoGe-2 (ComfyUI native, Microsoft, MIT):
    depth           metric z distance from the camera, metres (0 where there is
                    no surface, e.g. sky; see geometry_mask)
    world_position  camera-space position in metres, OpenGL axes (x right,
                    y up, camera looking down -z), which is what Relight uses
    normal_map      camera-space normals, 0..1 encoded, OpenGL or DirectX
    ao              ground-truth-style ambient occlusion (GTAO integral) over
                    the metric geometry, 1 = open, 0 = fully occluded
    curvature       mean curvature in 1/metre, signed (convex > 0)
    geometry_mask   1 where MoGe found a surface

Materials, Marigold IID Appearance v1.1 (ETH Zurich, OpenRAIL++-M):
    albedo          linear base colour
    roughness       linear 0..1
    metallic        linear 0..1

Lighting, Marigold IID Lighting v1.1, scaled to the plate:
    diffuse_lighting   a * albedo_L * shading
    specular_lighting  b * residual (non-diffuse light: reflections, highlights)
    The model predicts shading and residual only up to scale, so a and b are
    fitted per frame by least squares against the linear beauty. The fit
    error is reported in `info`.

Measured:
    motion_vector   backward optical flow to the previous frame (DIS), pixels,
                    +x right, +y up (Nuke convention)

All colour passes, and the beauty in `passes`, are scene-linear.
"""
from __future__ import annotations

import json
import logging
import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger("radiance.vfx.multipass.estimate")

BEAUTY_ENCODINGS = ("sRGB (display)", "Linear (scene)")
NORMAL_CONVENTIONS = ("OpenGL (Y-Up)", "DirectX (Y-Down)")


# ── Colour ──────────────────────────────────────────────────────────────────

def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(min=0.0)
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x.pow(1.0 / 2.4) - 0.055)


# ── Geometry ────────────────────────────────────────────────────────────────

def opencv_to_opengl(v: torch.Tensor) -> torch.Tensor:
    """(..., 3) camera-space vectors: OpenCV (x right, y down, z forward) -> OpenGL (x right, y up, z back)."""
    return torch.stack([v[..., 0], -v[..., 1], -v[..., 2]], dim=-1)


def encode_normals(n_gl: torch.Tensor, convention: str) -> torch.Tensor:
    n = F.normalize(n_gl, dim=-1)
    if convention.startswith("DirectX"):
        n = torch.stack([n[..., 0], -n[..., 1], n[..., 2]], dim=-1)
    return (n * 0.5 + 0.5).clamp(0.0, 1.0)


def focal_px(intrinsics: torch.Tensor, width: int) -> torch.Tensor:
    """MoGe returns normalised intrinsics (fx in image widths). (B,3,3) -> (B,) focal length in pixels."""
    return intrinsics[:, 0, 0].float() * float(width)


def _blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur of (B,H,W,C)."""
    r = max(1, int(math.ceil(3 * sigma)))
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=x.device, dtype=x.dtype) / sigma) ** 2)
    k = k / k.sum()
    C = x.shape[-1]
    t = x.permute(0, 3, 1, 2)
    t = F.conv2d(F.pad(t, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1).expand(C, 1, 1, -1), groups=C)
    t = F.conv2d(F.pad(t, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1).expand(C, 1, -1, 1), groups=C)
    return t.permute(0, 2, 3, 1)


def _grid(h: int, w: int, device) -> torch.Tensor:
    ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) / h * 2 - 1
    xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) / w * 2 - 1
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx, gy], dim=-1)          # (H,W,2), grid_sample order


def depth_edges(depth: torch.Tensor, valid: torch.Tensor, rel: float = 0.04) -> torch.Tensor:
    """(B,H,W) bool: pixels next to a depth discontinuity (relative jump above `rel`), or invalid."""
    z = torch.where(valid, depth, torch.zeros_like(depth))
    e = ~valid.clone()
    dz_u = (z[:, :, 1:] - z[:, :, :-1]).abs() / z[:, :, 1:].clamp(min=1e-4)
    dz_v = (z[:, 1:, :] - z[:, :-1, :]).abs() / z[:, 1:, :].clamp(min=1e-4)
    ju = dz_u > rel
    jv = dz_v > rel
    e[:, :, 1:] |= ju
    e[:, :, :-1] |= ju
    e[:, 1:, :] |= jv
    e[:, :-1, :] |= jv
    return e


def normals_from_positions(P: torch.Tensor, fallback: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """Normals of the point map itself (OpenGL camera space), `fallback` at depth edges.

    Occlusion is computed from the positions, so the tangent plane has to come
    from the same surface. MoGe's predicted normals are smoother and disagree
    with the point map near silhouettes, which makes the plane cut through
    real geometry and draws a dark outline around every object.
    """
    du = torch.zeros_like(P)
    dv = torch.zeros_like(P)
    du[:, :, 1:-1] = P[:, :, 2:] - P[:, :, :-2]
    dv[:, 1:-1, :] = P[:, 2:, :] - P[:, :-2, :]
    n = F.normalize(torch.cross(dv, du, dim=-1), dim=-1)
    ok = ~edges
    ok[:, :, 0] = ok[:, :, -1] = False
    ok[:, 0, :] = ok[:, -1, :] = False
    ok &= n.norm(dim=-1) > 0.5
    return torch.where(ok.unsqueeze(-1), n, fallback)


def fill_edges(x: torch.Tensor, edges: torch.Tensor, grow: int = 1, window: int = 7) -> torch.Tensor:
    """Replace (B,H,W) values at depth edges with the mean of nearby non-edge values.

    Monocular point maps have "flying pixels" along silhouettes, halfway
    between the foreground and the background. They are not surfaces, and any
    occlusion measured there draws an outline no renderer would produce.
    """
    bad = edges.float().unsqueeze(1)
    if grow > 0:
        bad = F.max_pool2d(bad, 2 * grow + 1, 1, grow)
    good = 1.0 - bad
    pad = window // 2
    num = F.avg_pool2d(x.unsqueeze(1) * good, window, 1, pad, count_include_pad=False)
    den = F.avg_pool2d(good, window, 1, pad, count_include_pad=False)
    filled = torch.where(den > 1e-3, num / den.clamp(min=1e-3), x.unsqueeze(1))
    return torch.where(bad > 0, filled, x.unsqueeze(1))[:, 0]


def gtao(
    positions: torch.Tensor,
    normals: torch.Tensor,
    valid: torch.Tensor,
    f_px: torch.Tensor,
    radius_m: float,
    slices: int = 8,
    steps: int = 8,
    max_side: int = 1024,
    occluder_ok: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Ground-truth ambient occlusion (Jimenez et al. 2016, as in XeGTAO).

    positions, normals: (B,H,W,3) OpenGL camera space, metres, unit normals.
    valid: (B,H,W) bool. f_px: (B,) focal length in pixels at full size.
    Returns visibility (B,H,W) in [0,1], cosine-weighted, 1 = unoccluded.

    Runs at up to `max_side` pixels on the long edge and is upsampled; the
    radius is in metres, so the result does not depend on that working size.
    `occluder_ok` (B,H,W) bool marks pixels that may occlude others; the rest
    (sky, flying pixels on silhouettes) are pushed out of range.
    """
    B, H, W, _ = positions.shape
    scale = min(1.0, max_side / max(H, W))
    h, w = max(1, round(H * scale)), max(1, round(W * scale))
    if (h, w) != (H, W):
        def down(t):
            return F.interpolate(t.permute(0, 3, 1, 2), size=(h, w), mode="nearest").permute(0, 2, 3, 1)
        P = down(positions)
        N = F.normalize(down(normals), dim=-1)
        V_ok = down(valid.float().unsqueeze(-1))[..., 0] > 0.5
        O_ok = down(occluder_ok.float().unsqueeze(-1))[..., 0] > 0.5 if occluder_ok is not None else V_ok
        f = f_px * (w / W)
    else:
        P, N, V_ok, f = positions, normals, valid, f_px
        O_ok = occluder_ok if occluder_ok is not None else valid
    dev = P.device
    P = torch.where(V_ok.unsqueeze(-1), P, torch.zeros_like(P))
    far = torch.tensor([0.0, 0.0, -1.0e6], device=dev)
    P_occ = torch.where((O_ok & V_ok).unsqueeze(-1), P, far.expand_as(P))
    view = F.normalize(-P, dim=-1)
    depth = (-P[..., 2]).clamp(min=1e-4)
    r_px = (radius_m * f.view(B, 1, 1) / depth).clamp(max=0.25 * max(h, w))      # screen radius per pixel

    falloff_range = 0.615 * radius_m
    falloff_from = radius_m * (1.0 - 0.615)
    falloff_mul = -1.0 / falloff_range
    falloff_add = falloff_from / falloff_range + 1.0

    base = _grid(h, w, dev).unsqueeze(0).expand(B, -1, -1, -1)
    P_bchw = P_occ.permute(0, 3, 1, 2)
    to_ndc = torch.tensor([2.0 / w, 2.0 / h], device=dev)
    visibility = torch.zeros(B, h, w, device=dev)

    for k in range(slices):
        phi = math.pi * (k + 0.5) / slices
        c, s = math.cos(phi), math.sin(phi)
        dir3 = torch.tensor([c, s, 0.0], device=dev).expand(B, h, w, 3)          # view-space slice direction, y up
        screen = torch.tensor([c, -s], device=dev)                                # pixels, y down
        ortho = dir3 - (dir3 * view).sum(-1, keepdim=True) * view
        axis = F.normalize(torch.cross(ortho, view, dim=-1), dim=-1)
        proj_n = N - axis * (N * axis).sum(-1, keepdim=True)
        proj_len = proj_n.norm(dim=-1).clamp(min=1e-6)
        sign_n = torch.sign((ortho * proj_n).sum(-1))
        cos_n = ((proj_n * view).sum(-1) / proj_len).clamp(-1.0, 1.0)
        n = sign_n * torch.acos(cos_n)
        hc0 = torch.cos(n + math.pi / 2)
        hc1 = torch.cos(n - math.pi / 2)
        for j in range(steps):
            t = ((j + 0.5) / steps) ** 2
            off = (1.0 + t * (r_px - 1.0)).clamp(min=1.0)                        # pixels, at least one
            delta = (off.unsqueeze(-1) * screen) * to_ndc                          # (B,h,w,2)
            for side in (1.0, -1.0):
                sp = F.grid_sample(P_bchw, base + side * delta, mode="nearest",
                                   padding_mode="border", align_corners=False).permute(0, 2, 3, 1)
                d = sp - P
                dist = d.norm(dim=-1).clamp(min=1e-6)
                shc = ((d / dist.unsqueeze(-1)) * view).sum(-1)
                wgt = (dist * falloff_mul + falloff_add).clamp(0.0, 1.0)
                if side > 0:
                    hc0 = torch.maximum(hc0, torch.lerp(torch.cos(n + math.pi / 2), shc, wgt))
                else:
                    hc1 = torch.maximum(hc1, torch.lerp(torch.cos(n - math.pi / 2), shc, wgt))
        h0 = -torch.acos(hc1.clamp(-1, 1))
        h1 = torch.acos(hc0.clamp(-1, 1))
        h0 = n + (h0 - n).clamp(-math.pi / 2, math.pi / 2)
        h1 = n + (h1 - n).clamp(-math.pi / 2, math.pi / 2)
        sin_n = torch.sin(n)
        iarc0 = (cos_n + 2 * h0 * sin_n - torch.cos(2 * h0 - n)) / 4
        iarc1 = (cos_n + 2 * h1 * sin_n - torch.cos(2 * h1 - n)) / 4
        visibility += proj_len * (iarc0 + iarc1)

    visibility = (visibility / slices).clamp(0.0, 1.0)
    visibility = torch.where(V_ok, visibility, torch.ones_like(visibility))
    if (h, w) != (H, W):
        visibility = F.interpolate(visibility.unsqueeze(1), size=(H, W), mode="bilinear",
                                   align_corners=False)[:, 0]
    return visibility


def mean_curvature(normals: torch.Tensor, depth: torch.Tensor, valid: torch.Tensor, f_px: torch.Tensor,
                   edges: Optional[torch.Tensor] = None, sigma_px: float = 1.5) -> torch.Tensor:
    """Mean curvature H = div(N)/2 in 1/metre from camera-space unit normals (OpenGL).

    Screen derivatives are converted to metric ones with the pixel footprint
    depth / f. Convex surfaces (a sphere seen from outside) are positive.
    """
    B, H, W, _ = normals.shape
    if sigma_px > 0:
        normals = F.normalize(_blur(normals, sigma_px), dim=-1)     # measure at a few-pixel scale, not per-pixel noise
    nx = normals[..., 0]
    ny = normals[..., 1]
    dnx_du = torch.zeros_like(nx)
    dny_dv = torch.zeros_like(ny)
    dnx_du[:, :, 1:-1] = (nx[:, :, 2:] - nx[:, :, :-2]) * 0.5
    dny_dv[:, 1:-1, :] = (ny[:, 2:, :] - ny[:, :-2, :]) * 0.5          # v runs down the image
    ok = valid.clone()
    if edges is not None:
        grow = F.max_pool2d(edges.float().unsqueeze(1), 2 * int(math.ceil(2 * sigma_px)) + 1, 1,
                            int(math.ceil(2 * sigma_px)))[:, 0] > 0
        ok &= ~grow                                                       # curvature is undefined across a silhouette
    ok[:, :, 1:-1] &= valid[:, :, 2:] & valid[:, :, :-2]
    ok[:, 1:-1, :] &= valid[:, 2:, :] & valid[:, :-2, :]
    ok[:, 0, :] = ok[:, -1, :] = False
    ok[:, :, 0] = ok[:, :, -1] = False
    per_m = f_px.view(B, 1, 1) / depth.clamp(min=1e-4)
    div = (dnx_du - dny_dv) * per_m                                       # y up in view space
    return torch.where(ok, 0.5 * div, torch.zeros_like(div))


# ── Lighting scale ──────────────────────────────────────────────────────────

def fit_lighting(
    beauty_lin: torch.Tensor, diffuse_unit: torch.Tensor, residual: torch.Tensor, usable: torch.Tensor,
) -> Tuple[float, float, float]:
    """Non-negative least squares for beauty ~= a * diffuse_unit + b * residual.

    All (H,W,3). `usable` (H,W) excludes clipped and invalid pixels. Returns
    (a, b, relative RMS error over the usable pixels).
    """
    m = usable.unsqueeze(-1).expand_as(beauty_lin)
    y = beauty_lin[m].double()
    x1 = diffuse_unit[m].double()
    x2 = residual[m].double()
    if y.numel() < 16 or float(y.abs().sum()) == 0.0:
        return 1.0, 1.0, float("nan")

    def solve_one(x):
        den = float((x * x).sum())
        return max(0.0, float((x * y).sum()) / den) if den > 0 else 0.0

    A = torch.stack([x1, x2], dim=1)
    try:
        sol = torch.linalg.lstsq(A, y.unsqueeze(1)).solution[:, 0]
        a, b = float(sol[0]), float(sol[1])
    except Exception:  # noqa: BLE001 - singular system
        a, b = -1.0, -1.0
    # A rank-deficient fit (e.g. a zero residual pass) can return a feasible
    # but inferior solution. Compare the interior with both boundary fits.
    cand = [(solve_one(x1), 0.0), (0.0, solve_one(x2))]
    if math.isfinite(a) and math.isfinite(b) and a >= 0 and b >= 0:
        cand.append((a, b))
    a, b = min(cand, key=lambda ab: float(((ab[0] * x1 + ab[1] * x2 - y) ** 2).sum()))
    err = float(torch.sqrt(((a * x1 + b * x2 - y) ** 2).mean()) / torch.sqrt((y ** 2).mean()).clamp(min=1e-12))
    return a, b, err


# ── Node ────────────────────────────────────────────────────────────────────

def _match(x: torch.Tensor, B: int, H: int, W: int) -> torch.Tensor:
    x = x.float()
    if x.shape[1:3] != (H, W):
        x = F.interpolate(x.permute(0, 3, 1, 2), size=(H, W), mode="bilinear",
                          align_corners=False).permute(0, 2, 3, 1)
    if x.shape[0] != B:
        x = x[:1].expand(B, -1, -1, -1) if x.shape[0] == 1 else x[:B]
    return x


def _rgb(x: torch.Tensor) -> torch.Tensor:
    return x[..., :3] if x.shape[-1] >= 3 else x[..., :1].expand(-1, -1, -1, 3)


def _three(x: torch.Tensor) -> torch.Tensor:
    return x.unsqueeze(-1).expand(*x.shape, 3).contiguous()


def _interrupt_check() -> None:
    try:
        import comfy.model_management as mm  # type: ignore
    except ImportError:
        return
    check = getattr(mm, "throw_exception_if_processing_interrupted", None)
    if callable(check):
        check()


def _compute_device() -> torch.device:
    try:
        import comfy.model_management as mm  # type: ignore
        return torch.device(mm.get_torch_device())
    except Exception:  # noqa: BLE001
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _free_vram(device: torch.device, need_bytes: float) -> None:
    try:
        import comfy.model_management as mm  # type: ignore
        mm.free_memory(need_bytes, device)
    except Exception:  # noqa: BLE001
        pass


class RadianceMultipassEstimate:
    CATEGORY = "FXTD STUDIOS/Radiance/VFX"
    DESCRIPTION = (
        "Estimate render passes from a plate with trained models: MoGe-2 for metric geometry "
        "(depth, position, normals, AO, curvature) and Marigold IID for materials (albedo, "
        "roughness, metallic) and lighting (diffuse and specular light). No image-filter guesses."
    )
    FUNCTION = "estimate"

    RETURN_TYPES = (
        "RADIANCE_PASSES", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE",
        "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING",
    )
    RETURN_NAMES = (
        "passes", "beauty", "albedo", "roughness", "metallic", "diffuse_lighting",
        "specular_lighting", "normal_map", "depth", "world_position", "ao", "curvature",
        "motion_vector", "geometry_mask", "info",
    )
    OUTPUT_TOOLTIPS = (
        "Every estimated pass plus the linear beauty, for Write EXR Passes and Relight.",
        "The plate, scene-linear.",
        "Base colour, scene-linear (Marigold IID Appearance).",
        "Roughness 0..1 (Marigold IID Appearance).",
        "Metallic 0..1 (Marigold IID Appearance).",
        "Diffuse light contribution, scene-linear, scaled to the plate (Marigold IID Lighting).",
        "Non-diffuse light (reflections, highlights), scene-linear, scaled to the plate.",
        "Camera-space normals, 0..1 encoded (MoGe-2).",
        "Metric z depth in metres; 0 where there is no surface (MoGe-2).",
        "Camera-space position in metres, OpenGL axes (MoGe-2).",
        "Ambient occlusion from the metric geometry, 1 = open.",
        "Mean curvature in 1/metre, convex positive.",
        "Backward optical flow to the previous frame in pixels, +y up.",
        "1 where MoGe found a surface, 0 for sky and invalid pixels.",
        "JSON: models, estimated field of view, lighting scale factors and fit error.",
    )

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "beauty": ("IMAGE", {"tooltip": "The plate. One image or a batch of frames."}),
                "beauty_encoding": (list(BEAUTY_ENCODINGS), {
                    "default": BEAUTY_ENCODINGS[0],
                    "tooltip": "How the plate is encoded. Ordinary images and video frames are sRGB. "
                               "Choose Linear for scene-linear EXR plates; values above 1.0 are clipped "
                               "for the models and left out of the lighting fit."}),
                "geometry": ("BOOLEAN", {"default": True, "tooltip": "Depth, position, normals, AO and curvature (MoGe-2, 662 MB)."}),
                "materials": ("BOOLEAN", {"default": True, "tooltip": "Albedo, roughness, metallic (Marigold IID Appearance, 2.5 GB)."}),
                "lighting": ("BOOLEAN", {"default": True, "tooltip": "Diffuse and specular lighting (Marigold IID Lighting, 1.7 GB; shares files with Appearance)."}),
                "download_missing_models": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Fetch missing weights from Hugging Face on first run (pinned versions). "
                               "RADIANCE_ALLOW_DOWNLOADS=0 or HF_HUB_OFFLINE=1 always blocks this."}),
            },
            "optional": {
                "prev_frame": ("IMAGE", {"tooltip": "Previous frame for motion vectors of a single image."}),
                "moge_model": ("MOGE_MODEL", {"tooltip": "Optional: a model from ComfyUI's Load MoGe Model. Must be MoGe-2 for normals."}),
                "geometry_detail": ("INT", {"default": 9, "min": 0, "max": 9, "tooltip": "MoGe resolution level: 0 fastest, 9 most detail."}),
                "fov_x_degrees": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 170.0, "step": 0.1,
                                             "tooltip": "Known horizontal field of view. 0 = estimate it."}),
                "normal_convention": (list(NORMAL_CONVENTIONS), {"default": NORMAL_CONVENTIONS[0],
                                                                  "tooltip": "Encoding of the camera-space normal pass (0..1, +Z toward the camera). "
                                                                             "DirectX (Y-Down) flips the green channel; match the renderer or tool reading it."}),
                "ao_radius_m": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 50.0, "step": 0.01,
                                          "tooltip": "Occlusion search radius in metres of scene space."}),
                "ao_quality": ("INT", {"default": 8, "min": 2, "max": 32, "tooltip": "AO slices and steps per slice."}),
                "material_steps": ("INT", {"default": 4, "min": 1, "max": 50, "tooltip": "Marigold denoising steps. 4 is the trained default."}),
                "material_resolution": ("INT", {"default": 768, "min": 0, "max": 2048, "step": 64,
                                                "tooltip": "Marigold processing size (long edge). 768 is the trained size; 0 = native."}),
                "ensemble_size": ("INT", {"default": 1, "min": 1, "max": 10,
                                          "tooltip": "Average several Marigold predictions. Slower, steadier."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF, "control_after_generate": "fixed",
                                 "tooltip": "Marigold noise seed, reused on every frame to limit flicker."}),
                "batch_is_sequence": ("BOOLEAN", {"default": True,
                                                  "tooltip": "Treat the batch as consecutive frames for motion vectors."}),
            },
        }

    def estimate(
        self,
        beauty: torch.Tensor,
        beauty_encoding: str = BEAUTY_ENCODINGS[0],
        geometry: bool = True,
        materials: bool = True,
        lighting: bool = True,
        download_missing_models: bool = True,
        prev_frame: Optional[torch.Tensor] = None,
        moge_model: Any = None,
        geometry_detail: int = 9,
        fov_x_degrees: float = 0.0,
        normal_convention: str = NORMAL_CONVENTIONS[0],
        ao_radius_m: float = 0.5,
        ao_quality: int = 8,
        material_steps: int = 4,
        material_resolution: int = 768,
        ensemble_size: int = 1,
        seed: int = 0,
        batch_is_sequence: bool = True,
    ) -> Tuple:
        from . import estimate_models as em
        from .core import _flow_to_hsv_image, _luminance, _optical_flow  # noqa: F401

        if beauty.dim() != 4:
            raise ValueError("[Multipass Estimate] beauty must be an IMAGE batch (B,H,W,C).")
        B, H, W = beauty.shape[:3]
        out_dev = beauty.device
        rgb = _rgb(beauty.float())
        if beauty_encoding.startswith("Linear"):
            lin = rgb.clamp(min=0.0)
            srgb = linear_to_srgb(lin)
            clipped = (rgb >= 1.0).any(-1)
        else:
            srgb = rgb.clamp(0.0, 1.0)
            lin = srgb_to_linear(srgb)
            clipped = (rgb >= 0.995).any(-1)
        alpha = beauty.float()[..., 3] if beauty.shape[-1] >= 4 else None

        info: Dict[str, Any] = {"frames": B, "size": [W, H], "encoding": beauty_encoding}
        passes: Dict[str, Any] = {"beauty": lin.contiguous()}
        zeros3 = torch.zeros(B, H, W, 3, device=out_dev)
        device = _compute_device()
        progress = None
        try:
            import comfy.utils  # type: ignore
            progress = comfy.utils.ProgressBar(B * (int(geometry) + int(materials) + int(lighting)) or 1)
        except Exception:  # noqa: BLE001
            pass

        def tick():
            if progress is not None:
                progress.update(1)

        valid = torch.ones(B, H, W, dtype=torch.bool, device=out_dev)

        # ── Geometry ────────────────────────────────────────────────────────
        if geometry:
            model = em.prepare_moge(moge_model) if moge_model is not None else em.load_moge(download_missing_models)
            if str(getattr(model, "version", "v2")).lstrip("vV").startswith("1"):
                logger.warning("[Multipass Estimate] MoGe-1 has no normal head; normals are derived from the point map.")
            pts, dep, msk, nrm, K = [], [], [], [], []
            fov = None if fov_x_degrees <= 0 else float(fov_x_degrees)
            for i in range(B):
                _interrupt_check()
                with torch.inference_mode():
                    o = model.infer(srgb[i:i + 1].permute(0, 3, 1, 2).contiguous(),
                                    resolution_level=int(geometry_detail), fov_x=fov,
                                    force_projection=True, apply_mask=False)
                pts.append(o["points"].float().to(out_dev))
                dep.append(o["depth"].float().to(out_dev))
                msk.append((o["mask"] if "mask" in o else torch.isfinite(o["depth"])).to(out_dev).bool())
                if "normal" in o:
                    nrm.append(o["normal"].float().to(out_dev))
                else:
                    from comfy_extras.nodes_moge import _normals_from_points  # type: ignore
                    nrm.append(_normals_from_points(o["points"].float()).to(out_dev))
                K.append(o["intrinsics"].float().to(out_dev))
                tick()
            points = torch.cat(pts)
            depth_m = torch.cat(dep)
            valid = torch.cat(msk) & torch.isfinite(depth_m) & (depth_m > 0)
            valid &= torch.isfinite(points).all(-1)
            normals_cv = torch.cat(nrm)
            intr = torch.cat(K)
            if points.shape[1:3] != (H, W):
                raise RuntimeError("[Multipass Estimate] MoGe returned a different resolution than the plate.")

            pos_gl = torch.where(valid.unsqueeze(-1), opencv_to_opengl(torch.nan_to_num(points)), torch.zeros_like(points))
            n_gl = F.normalize(opencv_to_opengl(torch.nan_to_num(normals_cv)), dim=-1)
            n_gl = torch.where(valid.unsqueeze(-1), n_gl, torch.tensor([0.0, 0.0, 1.0], device=out_dev).expand_as(n_gl))
            depth_z = torch.where(valid, torch.nan_to_num(depth_m), torch.zeros_like(depth_m))
            f = focal_px(intr, W)

            edges = depth_edges(depth_z, valid)
            n_geo = normals_from_positions(pos_gl, n_gl, edges)
            vis = gtao(pos_gl, n_geo, valid, f, float(ao_radius_m), slices=int(ao_quality), steps=int(ao_quality),
                       occluder_ok=valid & ~edges)
            vis = torch.where(valid, fill_edges(vis, edges & valid), vis)
            curv = mean_curvature(n_gl, depth_z, valid, f, edges=edges)

            passes.update({
                "depth": _three(depth_z),
                "world_position": pos_gl.contiguous(),
                "normal": encode_normals(n_gl, normal_convention).contiguous(),
                "ao": _three(vis),
                "curvature": _three(curv),
                "geometry_mask": _three(valid.float()),
            })
            fov_est = [round(math.degrees(2 * math.atan(0.5 / float(k[0, 0]))), 2) for k in intr]
            vd = depth_z[valid]
            info["geometry"] = {
                "model": "MoGe-2 ViT-L normal (MIT)" if moge_model is None else "MoGe (connected model)",
                "fov_x_degrees": fov_est if fov is None else [fov] * B,
                "depth_range_m": [round(float(vd.min()), 3), round(float(vd.max()), 3)] if vd.numel() else None,
                "surface_fraction": round(float(valid.float().mean()), 4),
                "normal_convention": normal_convention,
                "ao_radius_m": ao_radius_m,
            }

        # ── Materials and lighting ──────────────────────────────────────────
        if materials or lighting:
            if geometry:
                del model
                if device.type == "cpu":
                    em.release_moge_cache()     # one large model resident at a time on the CPU
            _free_vram(device, 4.5e9)
            iid = em.load_marigold(download_missing_models, device).to(device)
            try:
                res = int(material_resolution) if material_resolution > 0 else 0
                albedo_l, rough_l, metal_l, diff_l, spec_l = [], [], [], [], []
                fits = []
                # All frames through one model, then the other: on the CPU only
                # one UNet is resident, so interleaving would reload it per frame.
                if materials:
                    names = iid.target_names("appearance")
                    for i in range(B):
                        _interrupt_check()
                        x = srgb[i:i + 1].permute(0, 3, 1, 2).contiguous()
                        pa = iid.predict("appearance", x, steps=material_steps, resolution=res,
                                         ensemble=ensemble_size, seed=seed).to(out_dev)
                        alb = pa[names.index("albedo")].permute(1, 2, 0)
                        mat = pa[names.index("material")].permute(1, 2, 0)
                        albedo_l.append(srgb_to_linear(alb))
                        rough_l.append(_three(mat[..., 0].clamp(0, 1)))
                        metal_l.append(_three(mat[..., 1].clamp(0, 1)))
                        tick()
                if lighting:
                    names = iid.target_names("lighting")
                    for i in range(B):
                        _interrupt_check()
                        x = srgb[i:i + 1].permute(0, 3, 1, 2).contiguous()
                        pl = iid.predict("lighting", x, steps=material_steps, resolution=res,
                                         ensemble=ensemble_size, seed=seed).to(out_dev)
                        a_lin = pl[names.index("albedo")].permute(1, 2, 0)
                        shading = pl[names.index("shading")].permute(1, 2, 0)
                        residual = pl[names.index("residual")].permute(1, 2, 0)
                        diffuse_unit = a_lin * shading
                        usable = (~clipped[i]) & valid[i]
                        a, b, err = fit_lighting(lin[i], diffuse_unit, residual, usable)
                        diff_l.append(a * diffuse_unit)
                        spec_l.append(b * residual)
                        fits.append({"diffuse_scale": round(a, 5), "specular_scale": round(b, 5),
                                     "rebuild_error": None if math.isnan(err) else round(err, 4)})
                        tick()
            finally:
                em.release_marigold(iid)
            if materials:
                passes.update({
                    "albedo": torch.stack(albedo_l).contiguous(),
                    "roughness": torch.stack(rough_l).contiguous(),
                    "metallic": torch.stack(metal_l).contiguous(),
                })
                info["materials"] = {"model": "Marigold IID Appearance v1.1 (OpenRAIL++-M)",
                                     "steps": material_steps, "resolution": res, "ensemble": ensemble_size, "seed": seed}
            if lighting:
                passes.update({
                    "diffuse_lighting": torch.stack(diff_l).contiguous(),
                    "specular_lighting": torch.stack(spec_l).contiguous(),
                })
                info["lighting"] = {
                    "model": "Marigold IID Lighting v1.1 (OpenRAIL++-M)",
                    "per_frame": fits,
                    "note": "beauty ~= diffuse_lighting + specular_lighting; rebuild_error is RMS relative "
                            "to the beauty over unclipped surface pixels.",
                }

        # ── Motion ──────────────────────────────────────────────────────────
        luma = _luminance(srgb, (0.2126, 0.7152, 0.0722))
        prev_luma = None
        if prev_frame is not None:
            p = _rgb(_match(prev_frame, B, H, W).to(out_dev))
            p_srgb = linear_to_srgb(p.clamp(min=0)) if beauty_encoding.startswith("Linear") else p.clamp(0, 1)
            prev_luma = _luminance(p_srgb, (0.2126, 0.7152, 0.0722))
            if batch_is_sequence and B > 1:
                prev_luma = torch.cat([prev_luma[:1], luma[:-1]])
        elif batch_is_sequence and B > 1:
            prev_luma = torch.cat([luma[:1], luma[:-1]])
        if prev_luma is not None:
            u, v = _optical_flow(luma, prev_luma, method="auto")
            if prev_frame is None:
                u[0] = 0.0
                v[0] = 0.0
            passes["motion_vector"] = torch.stack([u, -v, torch.zeros_like(u)], dim=-1).contiguous()
            info["motion"] = {"method": "DIS optical flow", "direction": "backward (to previous frame)",
                              "units": "pixels, +x right, +y up"}

        passes["alpha"] = _three(alpha.clamp(0, 1)) if alpha is not None else torch.ones(B, H, W, 3, device=out_dev)
        passes["_present"] = sorted(k for k, t in passes.items() if isinstance(t, torch.Tensor))
        passes["_estimator"] = "RadianceMultipassEstimate"
        passes["_normal_convention"] = normal_convention
        info["passes"] = passes["_present"]
        not_run = [n for n, on in (("geometry", geometry), ("materials", materials), ("lighting", lighting)) if not on]
        if not_run:
            info["not_run"] = not_run

        def get(name):
            t = passes.get(name)
            return t if isinstance(t, torch.Tensor) else zeros3

        return (
            passes, passes["beauty"], get("albedo"), get("roughness"), get("metallic"),
            get("diffuse_lighting"), get("specular_lighting"), get("normal"), get("depth"),
            get("world_position"), get("ao"), get("curvature"), get("motion_vector"),
            get("geometry_mask"), json.dumps(info, indent=2),
        )


NODE_CLASS_MAPPINGS = {"RadianceMultipassEstimate": RadianceMultipassEstimate}
NODE_DISPLAY_NAME_MAPPINGS = {"RadianceMultipassEstimate": "Multipass Estimate"}
