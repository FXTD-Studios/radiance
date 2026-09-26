import json
import logging
import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from ....performance import perf_finish, perf_start
from ....core.tensor.chunking import FrameSink, chunks, compute_device, frames_per_chunk

_NORMAL_INPUTS = ["OpenGL (Y-Up)", "DirectX (Y-Down)"]
_LIGHT_TYPES = ["Directional", "Point"]
logger = logging.getLogger("radiance.vfx.multipass.relight_comp")


def _resize_bhwc(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if x.shape[1] == height and x.shape[2] == width:
        return x
    return F.interpolate(
        x.float().permute(0, 3, 1, 2),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).permute(0, 2, 3, 1)


def _match_batch(x: torch.Tensor, batch: int) -> torch.Tensor:
    if x.shape[0] == batch:
        return x
    if x.shape[0] > batch:
        return x[:batch]
    if x.shape[0] == 1:
        return x.expand(batch, -1, -1, -1)
    raise ValueError(f"Batch mismatch: expected {batch} frames or a single broadcast frame, got {x.shape[0]}")


def _match_image(
    x: torch.Tensor,
    batch: int,
    height: int,
    width: int,
    channels: int = 3,
) -> torch.Tensor:
    out = _match_batch(_resize_bhwc(x.float(), height, width), batch)
    if out.shape[-1] == channels:
        return out.contiguous()
    if channels == 1:
        return out[..., :1].contiguous()
    if out.shape[-1] > channels:
        return out[..., :channels].contiguous()
    if out.shape[-1] == 1:
        return out.expand(-1, -1, -1, channels).contiguous()
    pad = out[..., -1:].expand(-1, -1, -1, channels - out.shape[-1])
    return torch.cat([out, pad], dim=-1).contiguous()


def _scalar_pass(
    x: Optional[torch.Tensor],
    batch: int,
    height: int,
    width: int,
    default: float,
    device: torch.device,
    clamp: bool = True,
) -> torch.Tensor:
    if x is None:
        out = torch.full((batch, height, width), default, device=device, dtype=torch.float32)
    else:
        out = _match_image(x, batch, height, width, channels=1)[..., 0].to(device=device)
    return out.clamp(0.0, 1.0) if clamp else out


def _color_tensor(r: float, g: float, b: float, device: torch.device) -> torch.Tensor:
    return torch.tensor([r, g, b], device=device, dtype=torch.float32).view(1, 1, 1, 3)


def _normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / torch.sqrt((v * v).sum(dim=-1, keepdim=True).clamp(min=eps))


def _normals_are_encoded(normal_map: torch.Tensor) -> bool:
    """True when the pass is 0..1 encoded (decode to -1..1), decided on the whole clip."""
    n = normal_map[..., :3]
    return float(n.detach().min()) >= -0.001 and float(n.detach().max()) <= 1.001


def _decode_normal_map(normal_map: torch.Tensor, convention: str, encoded: Optional[bool] = None) -> torch.Tensor:
    n = normal_map.float()[..., :3]
    if encoded is None:
        encoded = _normals_are_encoded(n)
    if encoded:
        n = n * 2.0 - 1.0
    if convention == "DirectX (Y-Down)":
        n = torch.stack([n[..., 0], -n[..., 1], n[..., 2]], dim=-1)
    return _normalize(n)


def _view_positions(
    batch: int,
    height: int,
    width: int,
    device: torch.device,
    depth_map: Optional[torch.Tensor],
    depth_scale: float,
) -> torch.Tensor:
    y = torch.linspace(1.0, -1.0, height, device=device, dtype=torch.float32)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    aspect = float(width) / max(float(height), 1.0)
    if depth_map is None:
        z = torch.zeros((batch, height, width), device=device, dtype=torch.float32)
    else:
        z = _scalar_pass(depth_map, batch, height, width, 0.5, device, clamp=False)
        z = (z - 0.5) * float(depth_scale)
    return torch.stack(
        [
            xx.view(1, height, width).expand(batch, -1, -1) * aspect,
            yy.view(1, height, width).expand(batch, -1, -1),
            z,
        ],
        dim=-1,
    )


def _blur_bhwc(x: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return x
    b, h, w, c = x.shape
    radius = min(int(radius), max(0, h - 1), max(0, w - 1))
    if radius <= 0:
        return x
    k = radius * 2 + 1
    x4 = x.permute(0, 3, 1, 2).reshape(b * c, 1, h, w)
    x4 = F.pad(x4, (radius, radius, radius, radius), mode="reflect")
    x4 = F.avg_pool2d(x4, kernel_size=k, stride=1)
    return x4.reshape(b, c, h, w).permute(0, 2, 3, 1)


class RadianceMultipassRelight:
    CATEGORY = "FXTD STUDIOS/Radiance/VFX"
    DESCRIPTION = (
        "Relights a CG or estimated plate from its albedo and normal passes with one directional or point light "
        "(GGX specular, Lambert diffuse), using roughness, metallic, AO and shadow passes when supplied. "
        "Use it to change key light direction or colour in comp without re-rendering; outputs are scene-linear."
    )

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "albedo": ("IMAGE", {"tooltip": "Diffuse albedo (base colour) pass, scene-linear with no lighting baked in. Also used as the specular colour where metallic is 1."}),
                "normal_map": ("IMAGE", {"tooltip": "Normal pass: 0..1 encoded values are decoded to -1..1 automatically, signed passes are used as is. View direction is +Z toward the camera."}),
            },
            "optional": {
                "beauty": ("IMAGE", {"tooltip": "Original beauty render, scene-linear. Only used when mix_with_beauty is above 0."}),
                "roughness": ("IMAGE", {"tooltip": "Roughness pass, first channel, 0 = mirror, 1 = fully rough (clamped to 0.045 minimum). Defaults to 0.5 when not connected."}),
                "metallic": ("IMAGE", {"tooltip": "Metalness pass, first channel, 0 = dielectric, 1 = metal (no diffuse, albedo tints the reflection). Defaults to 0 when not connected."}),
                "specular": ("IMAGE", {"tooltip": "Dielectric specular level pass, 0..1, scaling the default 4% reflectance at normal incidence (1 = 4%). Defaults to 1 when not connected."}),
                "ao": ("IMAGE", {"tooltip": "Ambient occlusion as renderers write it: 1 = open, 0 = fully occluded."}),
                "alpha": ("IMAGE", {"tooltip": "Coverage matte, first channel, 0..1. Passed through to the alpha output and used for output_premultiplied; defaults to 1."}),
                "shadow_mask": ("IMAGE", {"tooltip": "Shadow pass, first channel: 1 = fully shadowed, 0 = lit. Blocks direct diffuse and specular light but not ambient."}),
                "depth_map": ("IMAGE", {"tooltip": "Normalised 0..1 depth pass. Only used by a Point light when world_position is not connected, to give pixels a synthetic Z."}),
                "world_position": ("IMAGE", {"tooltip": "Position pass in camera space metres (OpenGL axes: X right, Y up, +Z toward the camera). Only used by a Point light, and overrides depth_map."}),
                "normal_convention": (_NORMAL_INPUTS, {"default": "OpenGL (Y-Up)", "tooltip": "Green channel convention of the normal pass. DirectX (Y-Down) flips the Y component before lighting."}),
                "light_type": (_LIGHT_TYPES, {"default": "Directional", "tooltip": "Directional: light_x/y/z is a direction, no falloff. Point: light_x/y/z is a position, with soft distance falloff 1 / (1 + 0.08 d^2)."}),
                "light_x": ("FLOAT", {"default": -0.35, "min": -10.0, "max": 10.0, "step": 0.01, "tooltip": "Light X (+ = screen right). Directional: component of the direction toward the light. Point: position in world_position units, or in screen units (X spans +/- aspect ratio) without it."}),
                "light_y": ("FLOAT", {"default": 0.45, "min": -10.0, "max": 10.0, "step": 0.01, "tooltip": "Light Y (+ = up). Directional: component of the direction toward the light. Point: position in world_position units, or in screen units (Y spans -1..1) without it."}),
                "light_z": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, "tooltip": "Light Z (+ = toward the camera). Directional: component of the direction toward the light, so positive values light the front. Point: position along the camera axis."}),
                "light_r": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 8.0, "step": 0.01, "tooltip": "Linear red multiplier of the light colour. Also tints the ambient term."}),
                "light_g": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 8.0, "step": 0.01, "tooltip": "Linear green multiplier of the light colour. Also tints the ambient term."}),
                "light_b": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 8.0, "step": 0.01, "tooltip": "Linear blue multiplier of the light colour. Also tints the ambient term."}),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.05, "tooltip": "Linear multiplier on the direct light (diffuse and specular). Does not affect ambient."}),
                "ambient": ("FLOAT", {"default": 0.03, "min": 0.0, "max": 4.0, "step": 0.01, "tooltip": "Flat fill light level, tinted by the light colour and multiplied by the AO pass. Applied to diffuse only."}),
                "specular_intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 8.0, "step": 0.05, "tooltip": "Linear multiplier on the specular term only. 0 gives a purely diffuse relight."}),
                "depth_scale": ("FLOAT", {"default": 10.0, "min": 0.01, "max": 1000.0, "step": 0.1, "tooltip": "Z range, in screen units (image height = 2), that the 0..1 depth pass spans around mid grey. Only used for a Point light with depth_map and no world_position."}),
                "depth_near_is_white": ("BOOLEAN", {"default": True, "tooltip": "Set on when near pixels are white in depth_map. Only used for a Point light with depth_map and no world_position."}),
                "mix_with_beauty": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Linear blend of the relit result toward the beauty input: 0 = relight only, 1 = original beauty. Needs beauty connected."}),
                "output_premultiplied": ("BOOLEAN", {"default": False, "tooltip": "Multiply the relit output (and the beauty mix) by alpha. Off leaves it unpremultiplied."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("relit", "diffuse_light", "specular_light", "lighting", "alpha", "relight_info")
    FUNCTION = "relight"

    def relight(
        self,
        albedo: torch.Tensor,
        normal_map: torch.Tensor,
        beauty: Optional[torch.Tensor] = None,
        roughness: Optional[torch.Tensor] = None,
        metallic: Optional[torch.Tensor] = None,
        specular: Optional[torch.Tensor] = None,
        ao: Optional[torch.Tensor] = None,
        alpha: Optional[torch.Tensor] = None,
        shadow_mask: Optional[torch.Tensor] = None,
        depth_map: Optional[torch.Tensor] = None,
        world_position: Optional[torch.Tensor] = None,
        normal_convention: str = "OpenGL (Y-Up)",
        light_type: str = "Directional",
        light_x: float = -0.35,
        light_y: float = 0.45,
        light_z: float = 1.0,
        light_r: float = 1.0,
        light_g: float = 1.0,
        light_b: float = 1.0,
        intensity: float = 1.0,
        ambient: float = 0.03,
        specular_intensity: float = 1.0,
        depth_scale: float = 10.0,
        depth_near_is_white: bool = True,
        mix_with_beauty: float = 0.0,
        output_premultiplied: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
        batch, height, width, _ = albedo.shape
        # 3.5.0: shaded a few frames at a time on the GPU. The whole clip used
        # to be shaded at once on the input's device (the CPU in ComfyUI) with
        # about 25 frame-sized temporaries, and unconnected passes were built
        # as full frames of a constant: 24 frames of 1024x576 ran out of
        # memory in 6 GB. Unconnected passes and per-light constants are now
        # broadcast values.
        device = compute_device()
        _perf = perf_start(device)
        encoded = _normals_are_encoded(normal_map.float())

        def part(x: Optional[torch.Tensor], a: int, b: int) -> Optional[torch.Tensor]:
            if x is None or x.shape[0] == 1:
                return x
            if x.shape[0] < batch:
                raise ValueError(f"Batch mismatch: expected {batch} frames or a single broadcast frame, got {x.shape[0]}")
            return x[a:b]

        def scalar(x: Optional[torch.Tensor], a: int, b: int, n: int, default: float) -> torch.Tensor:
            if x is None:
                return torch.full((1, 1, 1), float(default), device=device, dtype=torch.float32)
            return _scalar_pass(part(x, a, b), n, height, width, default, device)

        outs = [FrameSink((batch, height, width, 3)) for _ in range(5)]
        per = frames_per_chunk(height, width, 3, 30.0, device)
        for a, b in chunks(batch, per):
            n = b - a
            base = _match_image(part(albedo, a, b), n, height, width, 3).to(device=device).clamp(min=0.0)
            normals = _decode_normal_map(
                _match_image(part(normal_map, a, b), n, height, width, 3).to(device=device),
                normal_convention, encoded,
            )
            rough = scalar(roughness, a, b, n, 0.5).clamp(0.045, 1.0)
            metal = scalar(metallic, a, b, n, 0.0)
            spec = (
                1.0 if specular is None
                else _match_image(part(specular, a, b), n, height, width, 3).to(device=device).clamp(0.0, 1.0)
            )
            # Renderer convention (Arnold, Cycles, Karma, Multipass Estimate):
            # white is open. This used to read the pass as an occlusion amount,
            # which inverted every real AO pass loaded through Read AOVs.
            accessibility = scalar(ao, a, b, n, 1.0)
            alpha_s = scalar(alpha, a, b, n, 1.0)
            shadow = scalar(shadow_mask, a, b, n, 0.0)
            visibility = (1.0 - shadow).clamp(0.0, 1.0)

            if light_type == "Point":
                if world_position is not None:
                    positions = _match_image(part(world_position, a, b), n, height, width, 3).to(device=device)
                else:
                    positions = _view_positions(n, height, width, device, part(depth_map, a, b), depth_scale)
                    # +Z is toward the camera, and _view_positions maps white to
                    # +Z, so near-is-white depth is already right; a distance
                    # style pass (near is black) is the one to flip. The flip was
                    # on the wrong setting, so near pixels sat behind far ones
                    # either way.
                    if depth_map is not None and not depth_near_is_white:
                        positions[..., 2] = -positions[..., 2]
                light_pos = _color_tensor(light_x, light_y, light_z, device)
                l_vec = light_pos - positions
                dist2 = (l_vec * l_vec).sum(dim=-1, keepdim=True).clamp(min=1e-6)
                light_dir = _normalize(l_vec)
                attenuation = 1.0 / (1.0 + dist2 * 0.08)
                del l_vec, positions
            else:
                light_dir = _normalize(_color_tensor(light_x, light_y, light_z, device))   # (1,1,1,3)
                attenuation = 1.0

            view_dir = _normalize(_color_tensor(0.0, 0.0, 1.0, device))                   # (1,1,1,3)
            half_vec = _normalize(light_dir + view_dir)

            ndotl = (normals * light_dir).sum(dim=-1, keepdim=True).clamp(0.0, 1.0)
            ndotv = (normals * view_dir).sum(dim=-1, keepdim=True).clamp(0.0, 1.0)
            ndoth = (normals * half_vec).sum(dim=-1, keepdim=True).clamp(0.0, 1.0)
            vdoth = (view_dir * half_vec).sum(dim=-1, keepdim=True).clamp(0.0, 1.0)
            del normals, light_dir, half_vec

            light_color = _color_tensor(light_r, light_g, light_b, device)
            direct_scalar = ndotl * visibility.unsqueeze(-1) * attenuation * float(intensity)
            diffuse_light = light_color * direct_scalar

            rough_v = rough.unsqueeze(-1)
            alpha_ggx = (rough_v * rough_v).clamp(0.002, 1.0)
            alpha2 = alpha_ggx * alpha_ggx
            denom = (ndoth * ndoth * (alpha2 - 1.0) + 1.0)
            d_ggx = alpha2 / (math.pi * denom * denom + 1e-8)
            k = ((rough_v + 1.0) * (rough_v + 1.0)) / 8.0
            g_l = ndotl / (ndotl * (1.0 - k) + k + 1e-8)
            g_v = ndotv / (ndotv * (1.0 - k) + k + 1e-8)
            f0_dielectric = 0.04 * spec
            metal_v = metal.unsqueeze(-1)
            f0 = f0_dielectric * (1.0 - metal_v) + base * metal_v
            fresnel = f0 + (1.0 - f0) * torch.pow((1.0 - vdoth).clamp(0.0, 1.0), 5.0)
            spec_brdf = (d_ggx * g_l * g_v * fresnel) / (4.0 * ndotl * ndotv + 1e-6)
            del denom, d_ggx, g_l, g_v, f0, fresnel
            specular_light = (
                spec_brdf
                * ndotl
                * light_color
                * visibility.unsqueeze(-1)
                * attenuation
                * float(intensity)
                * float(specular_intensity)
            ).clamp(min=0.0)
            del spec_brdf

            ambient_light = light_color * float(ambient) * accessibility.unsqueeze(-1)
            diffuse = base * (1.0 - metal_v) * (diffuse_light + ambient_light)
            relit = (diffuse + specular_light).clamp(min=0.0)
            del diffuse
            if output_premultiplied:
                relit = relit * alpha_s.unsqueeze(-1)

            if beauty is not None and mix_with_beauty > 0.0:
                src = _match_image(part(beauty, a, b), n, height, width, 3).to(device=device).clamp(min=0.0)
                mix = float(max(0.0, min(1.0, mix_with_beauty)))
                if output_premultiplied:
                    src = src * alpha_s.unsqueeze(-1)
                relit = relit * (1.0 - mix) + src * mix

            lighting = (diffuse_light + ambient_light + specular_light).clamp(min=0.0)
            full = (n, height, width, 3)
            outs[0].put(a, b, relit.expand(full))
            outs[1].put(a, b, diffuse_light.expand(full))
            outs[2].put(a, b, specular_light.expand(full))
            outs[3].put(a, b, lighting.expand(full))
            outs[4].put(a, b, alpha_s.unsqueeze(-1).expand(full))
            del base, relit, diffuse_light, specular_light, lighting, ambient_light

        info = {
            "mode": "real_pbr_relight",
            "required_passes": ["albedo", "normal_map"],
            "optional_passes_used": {
                "roughness": roughness is not None,
                "metallic": metallic is not None,
                "specular": specular is not None,
                "ao": ao is not None,
                "alpha": alpha is not None,
                "shadow_mask": shadow_mask is not None,
                "depth_map": depth_map is not None,
                "world_position": world_position is not None,
            },
            "missing_optional_defaults": {
                "roughness": 0.5 if roughness is None else None,
                "metallic": 0.0 if metallic is None else None,
                "specular": 1.0 if specular is None else None,
                "ao": 1.0 if ao is None else None,
                "alpha": 1.0 if alpha is None else None,
                "shadow_mask": 0.0 if shadow_mask is None else None,
            },
            "light_type": light_type,
            "normal_convention": normal_convention,
            "output_premultiplied": output_premultiplied,
            "note": "This node consumes supplied utility/PBR passes; it does not extract or hallucinate missing passes from beauty.",
        }

        perf_finish(logger, "Multipass Relight", _perf, device)
        return (*[o.value for o in outs], json.dumps(info, indent=2))


class RadianceMultipassComposite:
    CATEGORY = "FXTD STUDIOS/Radiance/VFX"
    DESCRIPTION = (
        "Alpha-over composite of a foreground onto a background with optional depth holdout, background shadow "
        "and light wrap. Use it to put a (relit) CG element into a plate; feed scene-linear images for correct edges."
    )

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "foreground": ("IMAGE", {"tooltip": "Foreground colour. Sets the output size and batch; its pixels are ignored when relit_foreground is connected."}),
                "alpha": ("IMAGE", {"tooltip": "Foreground matte, first channel, 0..1 (1 = foreground fully covers the background)."}),
            },
            "optional": {
                "background": ("IMAGE", {"tooltip": "Background plate, resized to the foreground. Black when not connected."}),
                "relit_foreground": ("IMAGE", {"tooltip": "Replaces the foreground pixels (for example the relit output of Multipass Relight). Same premultiplication as set by premultiplied_input."}),
                "foreground_depth": ("IMAGE", {"tooltip": "Foreground depth pass, first channel. Used for the depth holdout only when background_depth is also connected."}),
                "background_depth": ("IMAGE", {"tooltip": "Background depth pass in the same units and polarity as foreground_depth. Background pixels nearer than the foreground hold it out."}),
                "shadow_mask": ("IMAGE", {"tooltip": "Shadow cast onto the background, first channel, 1 = full shadow. Darkens the background by shadow_strength before the over."}),
                "alpha_invert": ("BOOLEAN", {"default": False, "tooltip": "Use 1 - alpha as the matte. Applied after unpremultiplying, which still uses the original alpha."}),
                "premultiplied_input": ("BOOLEAN", {"default": False, "tooltip": "Set on when the foreground is already multiplied by alpha; it is divided by alpha first to avoid a double premultiply."}),
                "depth_near_is_white": ("BOOLEAN", {"default": True, "tooltip": "Depth polarity of both depth passes. On: larger values are nearer. Off: smaller values are nearer (distance style Z)."}),
                "depth_bias": ("FLOAT", {"default": 0.01, "min": -0.25, "max": 0.25, "step": 0.001, "tooltip": "Depth tolerance in depth-pass units. Positive values keep the foreground in front when depths are nearly equal, negative favours the background."}),
                "shadow_strength": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "How much shadow_mask darkens the background: 0 = none, 1 = black where the mask is 1."}),
                "light_wrap": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Amount of blurred background added to the foreground's soft edges. 0 = off; needs background connected."}),
                "light_wrap_radius": ("INT", {"default": 8, "min": 0, "max": 64, "step": 1, "tooltip": "Box blur radius in pixels for the wrapped background. The edge band uses half this radius."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("composite", "premultiplied_foreground", "holdout_mask", "depth_matte", "comp_info")
    FUNCTION = "composite"

    def composite(
        self,
        foreground: torch.Tensor,
        alpha: torch.Tensor,
        background: Optional[torch.Tensor] = None,
        relit_foreground: Optional[torch.Tensor] = None,
        foreground_depth: Optional[torch.Tensor] = None,
        background_depth: Optional[torch.Tensor] = None,
        shadow_mask: Optional[torch.Tensor] = None,
        alpha_invert: bool = False,
        premultiplied_input: bool = False,
        depth_near_is_white: bool = True,
        depth_bias: float = 0.01,
        shadow_strength: float = 0.35,
        light_wrap: float = 0.0,
        light_wrap_radius: int = 8,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
        batch, height, width, _ = foreground.shape
        # 3.5.0: composited a few frames at a time on the GPU (was the whole
        # clip on the input's device, the CPU in ComfyUI, with full-size
        # zeros / ones for unconnected inputs). Every step is per frame.
        device = compute_device()
        _perf = perf_start(device)
        fg_src = relit_foreground if relit_foreground is not None else foreground

        def part(x: Optional[torch.Tensor], a: int, b: int) -> Optional[torch.Tensor]:
            if x is None or x.shape[0] == 1:
                return x
            if x.shape[0] < batch:
                raise ValueError(f"Batch mismatch: expected {batch} frames or a single broadcast frame, got {x.shape[0]}")
            return x[a:b]

        outs = [FrameSink((batch, height, width, 3)) for _ in range(4)]
        per = frames_per_chunk(height, width, 3, 12.0, device)
        for a, b in chunks(batch, per):
            n = b - a
            full = (n, height, width, 3)
            fg = _match_image(part(fg_src, a, b), n, height, width, 3).to(device=device).clamp(min=0.0)
            matte = _scalar_pass(part(alpha, a, b), n, height, width, 1.0, device)
            if premultiplied_input:
                straight_fg = torch.where(
                    matte.unsqueeze(-1) > 1e-8,
                    fg / matte.unsqueeze(-1).clamp(min=1e-8),
                    torch.zeros_like(fg),
                )
            else:
                straight_fg = fg
            if alpha_invert:
                matte = 1.0 - matte

            if background is None:
                bg = torch.zeros((1, 1, 1, 3), device=device, dtype=torch.float32)
            else:
                bg = _match_image(part(background, a, b), n, height, width, 3).to(device=device).clamp(min=0.0)

            if shadow_mask is not None and shadow_strength > 0.0:
                sh = _scalar_pass(part(shadow_mask, a, b), n, height, width, 0.0, device)
                bg = bg * (1.0 - sh.unsqueeze(-1) * float(shadow_strength)).clamp(0.0, 1.0)

            front_mask = torch.ones((1, 1, 1), device=device, dtype=torch.float32)
            if foreground_depth is not None and background_depth is not None:
                fg_z = _scalar_pass(part(foreground_depth, a, b), n, height, width, 0.5, device, clamp=False)
                bg_z = _scalar_pass(part(background_depth, a, b), n, height, width, 0.5, device, clamp=False)
                if depth_near_is_white:
                    front_mask = (fg_z >= (bg_z - float(depth_bias))).float()
                else:
                    front_mask = (fg_z <= (bg_z + float(depth_bias))).float()

            visible_alpha = (matte * front_mask).clamp(0.0, 1.0)

            if light_wrap > 0.0 and background is not None:
                soft_bg = _blur_bhwc(bg, int(light_wrap_radius))
                edge = (_blur_bhwc(visible_alpha.unsqueeze(-1), max(1, int(light_wrap_radius) // 2))[..., 0] - visible_alpha).clamp(0.0, 1.0)
                straight_fg = straight_fg + soft_bg * edge.unsqueeze(-1) * float(light_wrap)

            premult = straight_fg * visible_alpha.unsqueeze(-1)
            composite = premult + bg * (1.0 - visible_alpha).unsqueeze(-1)
            outs[0].put(a, b, composite.expand(full))
            outs[1].put(a, b, premult.expand(full))
            outs[2].put(a, b, visible_alpha.unsqueeze(-1).expand(full))
            outs[3].put(a, b, front_mask.unsqueeze(-1).expand(full))
            del fg, matte, straight_fg, bg, visible_alpha, premult, composite

        info = {
            "mode": "alpha_over_depth_comp",
            "relit_foreground_used": relit_foreground is not None,
            "background_used": background is not None,
            "depth_holdout_used": foreground_depth is not None and background_depth is not None,
            "shadow_mask_used": shadow_mask is not None,
            "light_wrap_used": bool(light_wrap > 0.0 and background is not None),
            "note": "Composite uses supplied alpha/depth/shadow data only; it does not infer hidden mattes or object IDs.",
        }

        perf_finish(logger, "Multipass Composite", _perf, device)
        return (*[o.value for o in outs], json.dumps(info, indent=2))


NODE_CLASS_MAPPINGS = {
    "RadianceMultipassRelight": RadianceMultipassRelight,
    "RadianceMultipassComposite": RadianceMultipassComposite,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceMultipassRelight": "Multipass Relight",
    "RadianceMultipassComposite": "Multipass Composite",
}
