import torch
import torch.nn.functional as F
import numpy as np

from radiance.core.tensor.chunking import FrameSink, chunks, compute_device, frames_per_chunk

class RadianceSDRtoHDRExpand:
    """
    ◎ Radiance SDR → HDR Expand

    Expands dynamic range from SDR footage via an inverse OETF pass and a
    mathematical highlight expansion. Does not reconstruct clipped detail.
    """
    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    DESCRIPTION = ("Expand an SDR image into HDR headroom via inverse OETF and "
                   "mathematical highlight expansion. Does not reconstruct clipped detail.")
    FUNCTION = "apply"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "SDR image, display-encoded as chosen in inverse_oetf. Alpha, if present, passes through untouched."}),
                "inverse_oetf": (["None", "sRGB", "Rec.709"], {"default": "sRGB", "tooltip": "Transfer curve removed before expansion, giving linear light. None = the image is already linear. The output stays linear."}),
                "threshold": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Linear luminance (after the inverse OETF) above which expansion begins. 0.8 = 80% of SDR white in linear light."}),
                "expansion_gain": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 100.0, "step": 0.1, "tooltip": "Multiplier on the added highlight energy: luma gains gain x (luma - threshold)^gamma. With the defaults SDR white (1.0) rises to about 1.6, not 5x."}),
                "expansion_gamma": ("FLOAT", {"default": 1.2, "min": 0.1, "max": 5.0, "step": 0.01, "tooltip": "Exponent on the amount luma exceeds the threshold. Above 1.0 the expansion starts more gently and adds less; below 1.0 it rises faster just above the threshold."}),
                "smoothness": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.5, "step": 0.01, "tooltip": "Width, in linear luma units, of the soft onset at the threshold (a sigmoid on luma, not a spatial blur). 0 = hard onset."}),
            }
        }

    @torch.no_grad()
    def apply(self, image: torch.Tensor, inverse_oetf: str, threshold: float, 
              expansion_gain: float, expansion_gamma: float, smoothness: float):
        img = image.clone()

        # ALPHA-OETF FIX: the inverse OETF ran over the WHOLE tensor before the
        # RGB slice was taken, so a 4-channel input had its alpha gamma-decoded
        # as if it were a colour channel and the decoded matte was concatenated
        # back below: alpha 0.5 came out 0.2140. Alpha is linear coverage and
        # carries no transfer function. Split first, decode colour only.
        # uplift_universal._inverse_oetf is called this way for the same reason.
        RGB = img[..., :3]
        extra = img[..., 3:]

        if inverse_oetf == "sRGB":
            RGB = torch.where(RGB <= 0.04045, RGB / 12.92, ((RGB + 0.055) / 1.055) ** 2.4)
        elif inverse_oetf == "Rec.709":
            RGB = torch.where(RGB < 0.0812, RGB / 4.5, ((RGB + 0.099) / 1.099) ** (1.0 / 0.45))

        luma = 0.2126 * RGB[..., 0] + 0.7152 * RGB[..., 1] + 0.0722 * RGB[..., 2]
        
        diff = luma - threshold
        # AUDIT-FIX (2026-08): this mask was computed and then never used --
        # the `smoothness` widget was a dead control (relu() below already
        # hard-gates the expansion at the threshold). It now feathers the
        # expansion onset as the tooltip has always promised.
        mask = torch.sigmoid(diff / max(smoothness, 0.0001)) if smoothness > 0 else (diff > 0).float()

        highlight_amt = F.relu(diff)
        expansion = (highlight_amt ** expansion_gamma) * expansion_gain * mask
        
        luma_safe = torch.clamp(luma, min=1e-6)
        ratio = RGB / luma_safe.unsqueeze(-1)
        
        expanded_RGB = RGB + (ratio * expansion.unsqueeze(-1))
        
        if extra.shape[-1] > 0:
            result = torch.cat([expanded_RGB, extra], dim=-1)
        else:
            result = expanded_RGB

        return (result,)


class RadianceHDRSynthesisEngine:
    """
    ◎ Radiance HDR Synthesis Engine
    
    Heuristic highlight expansion. The image is split into a Laplacian
    pyramid; only the low-pass base is lifted (a luma-weighted power curve
    toward energy_target, gated to highlights), then the detail bands are
    added back unchanged. Clipped detail is not invented: a flat clipped
    area stays flat, only brighter. For learned recovery use SDR → HDR
    Universal with Recover.
    """
    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    DESCRIPTION = "Synthesise HDR imagery from SDR input and optional guidance signals."
    FUNCTION = "synthesize"
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "highlight_mask")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "SDR image. Highlights are found where the low-pass luma passes about 0.7; no transfer curve is removed, so linearise first for a linear result."}),
                "energy_target": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Approximate brightness multiplier reached at SDR white (10.0 lifts 1.0 to about 9.6, roughly 3.3 stops). 1.0 = no lift."}),
                "recovery_iters": ("INT", {"default": 3, "min": 0, "max": 8, "step": 1,
                    "tooltip": "Pyramid depth. The lift is applied to the 1/2^N low-pass, so detail "
                               "finer than about 2^N px keeps its original contrast. 0 lifts the whole "
                               "image. It does not reconstruct clipped detail."}),
                "chroma_preservation": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "0 adds neutral highlight energy, reducing saturation as brightness rises. 1 preserves the original low-pass RGB ratios. Both use the same lifted luminance."}),
            },
            "optional": {
                "guidance_mask":  ("MASK",  {"tooltip": "Per-pixel guidance mask from Luminance Guidance."}),
                "guidance_nits":  ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10000.0, "step": 50.0,
                    "tooltip": "Target peak in nits inside guidance_mask, on the package's scale (203 nits = 1.0). 0 = ignore the mask and use energy_target everywhere."}),
            }
        }

    def _build_pyramid(self, img, iters):
        pyramid = []
        curr = img
        for _ in range(iters):
            down = F.interpolate(curr, scale_factor=0.5, mode="bilinear", align_corners=False)
            up = F.interpolate(down, size=(curr.shape[2], curr.shape[3]), mode="bilinear", align_corners=False)
            diff = curr - up
            pyramid.append(diff)
            curr = down
        pyramid.append(curr)
        return pyramid

    def _reconstruct_pyramid(self, pyramid):
        curr = pyramid.pop()
        while pyramid:
            level = pyramid.pop()
            up = F.interpolate(curr, size=(level.shape[2], level.shape[3]), mode="bilinear", align_corners=False)
            curr = up + level
        return curr

    @torch.no_grad()
    def synthesize(self, image: torch.Tensor, energy_target: float, recovery_iters: int, chroma_preservation: float,
                   guidance_mask: torch.Tensor = None, guidance_nits: float = 0.0):
        # image shape (B, H, W, C)
        B, H, W, C = image.shape
        img = image.permute(0, 3, 1, 2).float() # (B, C, H, W)
        
        # 1. Decompose into Laplacian Pyramid
        pyramid = self._build_pyramid(img[:, :3, :, :], recovery_iters)
        
        # 2. Extract Base Layer (The low-frequency luminance)
        base = pyramid[-1]
        base_H, base_W = base.shape[2], base.shape[3]
        
        # 3. Intelligent Energy Projection
        # We identify areas that are clipped (near 1.0) and "lift" them towards the target
        luma = 0.2126 * base[:, 0:1, :, :] + 0.7152 * base[:, 1:2, :, :] + 0.0722 * base[:, 2:3, :, :]
        
        # Highlight mask (areas near or above SDR white)
        mask = torch.sigmoid((luma - 0.7) * 10.0)
        
        # Dynamic Target Logic (Luminance Guidance)
        if guidance_mask is not None and guidance_nits > 0.0:
            # guidance_mask shape: (B, H, W) — comes from RadianceLuminanceGuidance
            g_mask = guidance_mask.unsqueeze(1).float()  # (B, 1, H, W)
            g_mask_low = F.interpolate(g_mask, size=(base_H, base_W), mode="bilinear", align_corners=False)

            # Target nits on the package's scale, 1.0 = 203 nits (BT.2408
            # reference white), the same scale as the output. It used 100.
            from radiance.color.encodings import DEFAULT_REFERENCE_WHITE_NITS
            g_target = guidance_nits / DEFAULT_REFERENCE_WHITE_NITS

            # Blend global energy_target with local guidance target
            effective_target = torch.lerp(torch.full_like(luma, energy_target), torch.full_like(luma, g_target), g_mask_low)
        else:
            effective_target = energy_target

        # Non-linear energy lift
        energy_lift = torch.pow(luma.clamp(min=1e-6), 2.2) * (effective_target - 1.0)
        # Neutral energy desaturates highlights; proportional energy retains
        # the source RGB ratios. Their luminance is identical.
        neutral_lifted = base + luma * energy_lift * mask
        ratio_lifted = base + base * energy_lift * mask
        base_lifted = torch.lerp(neutral_lifted, ratio_lifted, chroma_preservation)
            
        pyramid[-1] = base_lifted
        
        # 5. Reconstruct with Detail Retention
        result_rgb = self._reconstruct_pyramid(pyramid)
        
        # 6. Final compositing
        if C > 3:
            result = torch.cat([result_rgb, img[:, 3:, :, :]], dim=1)
        else:
            result = result_rgb
            
        # Return image and mask
        mask_out = F.interpolate(mask, size=(H, W), mode="bilinear").permute(0, 2, 3, 1).expand(-1, -1, -1, 3)
        return (result.permute(0, 2, 3, 1), mask_out)


class RadianceRelightEngine:
    """
    ◎ Radiance Relight Engine
    
    Adds one directional light to an image from a normal map: a Lambert
    diffuse plus Blinn-Phong specular pass, in float. The pass is added to
    the image, not multiplied by an albedo, so it brightens rather than
    re-renders. No environment map. A RADIANCE_CAMERA sets the view point
    for the specular term.
    """
    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    DESCRIPTION = "Add a directional light (Lambert + Blinn-Phong) to an image from its normal map."
    FUNCTION = "apply"
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "lighting_pass_only")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image to light. The lighting pass is added on top of it, not multiplied by an albedo."}),
                "normal_map": ("IMAGE", {"tooltip": "Normal map encoded 0 to 1 (RGB = XYZ x 0.5 + 0.5). Resized to the image if the sizes differ."}),
                "light_dir_x": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01, "tooltip": "Light direction X component. Normalized internally — sets the horizontal angle of the synthetic light."}),
                "light_dir_y": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01, "tooltip": "Light direction Y component. Positive = light from above."}),
                "light_dir_z": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01, "tooltip": "Light direction Z component. Positive = light in front of surface."}),
                "light_color_r": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "Red component of the synthetic light color. Values > 1.0 produce HDR emission."}),
                "light_color_g": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "Green component of the synthetic light color."}),
                "light_color_b": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "Blue component of the synthetic light color."}),
                "diffuse_intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Lambertian diffuse reflection strength. Controls broad, soft illumination."}),
                "specular_intensity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Specular highlight strength. Higher values produce brighter, more visible glints."}),
                "specular_roughness": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01, "tooltip": "Surface roughness for Blinn-Phong specular. Low = sharp glints (metallic), high = soft broad highlights (matte)."}),
            },
            "optional": {
                "camera": ("RADIANCE_CAMERA", {"tooltip": "Optional camera. Only the position in its transform is used, as the viewpoint for the specular term relative to a -1 to 1 image plane. Without it a viewer at z = 2 is assumed."}),
            }
        }

    @torch.no_grad()
    def apply(self, image: torch.Tensor, normal_map: torch.Tensor, 
              light_dir_x: float, light_dir_y: float, light_dir_z: float,
              light_color_r: float, light_color_g: float, light_color_b: float,
              diffuse_intensity: float, specular_intensity: float, specular_roughness: float,
              camera: dict = None):
        
        # 3.5.0: lit a few frames at a time on the GPU (was the whole clip at
        # once on the input's device, the CPU in ComfyUI, with about 6 full
        # temporaries). The light, view and half vectors are built once.
        device = compute_device()
        B_img, H_img, W_img, C_img = image.shape

        L = torch.tensor([light_dir_x, -light_dir_y, light_dir_z], device=device, dtype=torch.float32)
        L = F.normalize(L, p=2, dim=0)

        # ── View Vector (V) ──
        y_c, x_c = torch.meshgrid(
            torch.linspace(-1, 1, H_img, device=device),
            torch.linspace(-1, 1, W_img, device=device),
            indexing='ij'
        )
        if camera is not None and "transform" in camera:
            cam_mat = np.array(camera["transform"])
            cam_pos = torch.tensor(cam_mat[:3, 3], device=device, dtype=torch.float32)
            surface_pos = torch.stack([x_c, -y_c, torch.zeros_like(x_c)], dim=-1)
            V = cam_pos.view(1, 1, 1, 3) - surface_pos.unsqueeze(0)
            V = F.normalize(V, p=2, dim=-1)
        else:
            cam_z = 2.0
            V = torch.stack([-x_c, y_c, torch.full_like(x_c, cam_z)], dim=-1)
            V = F.normalize(V, p=2, dim=-1).unsqueeze(0)
        H = F.normalize(L.view(1, 1, 1, 3) + V, p=2, dim=-1)
        del V, x_c, y_c

        shininess = max(0.001, min(2.0 / (specular_roughness**2) - 2.0, 2048.0))
        light_color = torch.tensor([light_color_r, light_color_g, light_color_b], device=device, dtype=torch.float32).view(1, 1, 1, 3)

        def lighting_for(nm: torch.Tensor) -> torch.Tensor:
            normals = nm.to(device, torch.float32)
            if tuple(normals.shape[1:3]) != (H_img, W_img):
                normals = F.interpolate(normals.permute(0, 3, 1, 2),
                                        size=(H_img, W_img),
                                        mode="bilinear").permute(0, 2, 3, 1)
            normals = normals[..., :3] * 2.0 - 1.0
            N = normals / torch.norm(normals, p=2, dim=-1, keepdim=True).clamp(min=1e-6)
            diffuse = F.relu(torch.sum(N * L.view(1, 1, 1, 3), dim=-1, keepdim=True))
            specular = torch.pow(F.relu(torch.sum(N * H, dim=-1, keepdim=True)), shininess)
            return diffuse * diffuse_intensity * light_color + specular * specular_intensity * light_color

        four = C_img > 3

        def lighting_image(lp: torch.Tensor) -> torch.Tensor:
            return torch.cat([lp, torch.ones_like(lp[..., :1])], dim=-1) if four else lp

        # A single normal frame lights every image frame; the lighting output is
        # then that one frame, as before.
        shared = lighting_for(normal_map[:1]) if normal_map.shape[0] == 1 else None
        relit = FrameSink((B_img, H_img, W_img, C_img))
        lighting_out = FrameSink((1 if shared is not None else B_img, H_img, W_img, 4 if four else 3))
        if shared is not None:
            lighting_out.put(0, 1, lighting_image(shared))
        per = frames_per_chunk(H_img, W_img, C_img, 8.0, device)
        for a, b in chunks(B_img, per):
            img = image[a:b].to(device, torch.float32)
            lp = shared if shared is not None else lighting_for(normal_map[a:b])
            out = img[..., :3] + lp
            if four:
                out = torch.cat([out, img[..., 3:]], dim=-1)
            relit.put(a, b, out)
            if shared is None:
                lighting_out.put(a, b, lighting_image(lp))
            del img, lp, out

        return (relit.value, lighting_out.value)


NODE_CLASS_MAPPINGS = {
    "RadianceSDRtoHDRExpand": RadianceSDRtoHDRExpand,
    "RadianceHDRSynthesisEngine": RadianceHDRSynthesisEngine,
    "RadianceRelightEngine": RadianceRelightEngine,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceSDRtoHDRExpand": "◎ Radiance SDR → HDR Expand",
    "RadianceHDRSynthesisEngine": "◎ Radiance HDR Synthesis Engine",
    "RadianceRelightEngine": "◎ Radiance Relight Engine",
}
