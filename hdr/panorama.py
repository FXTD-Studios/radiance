import torch
import numpy as np
import logging
from typing import Tuple, Dict, Any
import cv2

# Local imports
from .utils import tensor_to_numpy_float32, numpy_to_tensor_float32

logger = logging.getLogger("radiance.hdr.panorama")

# ═══════════════════════════════════════════════════════════════════════════════
#                          HDR 360 PANORAMA NODES
# ═══════════════════════════════════════════════════════════════════════════════


class HDR360Generate:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    """
    Generate 360° equirectangular panoramas for HDRI environment mapping in 3D applications.
    """

    PROJECTION_TYPES = ["Equirectangular", "Cube_Map", "Mirror_Ball", "Angular_Map"]
    INTERPOLATION_MODES = ["Bilinear", "Bicubic", "Lanczos", "Nearest"]
    FILL_MODES = ["Mirror", "Repeat", "Black", "Edge"]

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "source_image": (
                    "IMAGE",
                    {
                        "tooltip": "Input image to project. For best results, use a wide or panoramic image."
                    },
                ),
                "projection_type": (
                    cls.PROJECTION_TYPES,
                    {
                        "default": "Equirectangular",
                        "tooltip": "Projection type: Equirectangular (standard HDRI), Cube_Map (6-face), Mirror_Ball (chrome ball), Angular_Map (light probe).",
                    },
                ),
                "output_width": (
                    "INT",
                    {
                        "default": 4096,
                        "min": 512,
                        "max": 16384,
                        "step": 64,
                        "tooltip": "Output panorama width in pixels. Standard HDRI sizes: 2048 (preview), 4096 (standard), 8192 (high), 16384 (ultra).",
                    },
                ),
                "output_height": (
                    "INT",
                    {
                        "default": 2048,
                        "min": 256,
                        "max": 8192,
                        "step": 64,
                        "tooltip": "Output panorama height. For equirectangular, height should be half of width (2:1 ratio).",
                    },
                ),
            },
            "optional": {
                "horizontal_fov": (
                    "FLOAT",
                    {
                        "default": 360.0,
                        "min": 30.0,
                        "max": 360.0,
                        "step": 1.0,
                        "tooltip": "Horizontal field of view in degrees. 360° = full sphere, less = partial panorama.",
                    },
                ),
                "vertical_fov": (
                    "FLOAT",
                    {
                        "default": 180.0,
                        "min": 15.0,
                        "max": 180.0,
                        "step": 1.0,
                        "tooltip": "Vertical field of view in degrees. 180° = pole to pole, less = band around horizon.",
                    },
                ),
                "rotation_x": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -180.0,
                        "max": 180.0,
                        "step": 1.0,
                        "tooltip": "Rotation around X axis (pitch) in degrees. Tilts the panorama up/down.",
                    },
                ),
                "rotation_y": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -180.0,
                        "max": 180.0,
                        "step": 1.0,
                        "tooltip": "Rotation around Y axis (yaw) in degrees. Rotates the panorama left/right.",
                    },
                ),
                "rotation_z": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -180.0,
                        "max": 180.0,
                        "step": 1.0,
                        "tooltip": "Rotation around Z axis (roll) in degrees. Tilts the horizon.",
                    },
                ),
                "interpolation": (
                    cls.INTERPOLATION_MODES,
                    {
                        "default": "Lanczos",
                        "tooltip": "Sampling interpolation: Lanczos (highest quality), Bicubic (good), Bilinear (fast), Nearest (pixelated).",
                    },
                ),
                "fill_mode": (
                    cls.FILL_MODES,
                    {
                        "default": "Mirror",
                        "tooltip": "How to fill areas outside source image: Mirror (reflects), Repeat (tiles), Black (transparent), Edge (extends edge pixels).",
                    },
                ),
                "exposure_adjust": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -5.0,
                        "max": 5.0,
                        "step": 0.1,
                        "tooltip": "Exposure adjustment in stops. +1 = double brightness, -1 = half brightness.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("panorama", "projection_map")
    OUTPUT_TOOLTIPS = ("Generated 360° panorama.", "UV projection map for debugging.")
    FUNCTION = "generate"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Upscale"
    DESCRIPTION = "Generate 360° equirectangular panoramas for HDRI environment mapping in 3D applications."

    def _create_rotation_matrix(self, rx: float, ry: float, rz: float) -> np.ndarray:
        """Create 3D rotation matrix from Euler angles (in degrees)."""
        rx, ry, rz = np.radians([rx, ry, rz])

        Rx = np.array(
            [[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]]
        )

        Ry = np.array(
            [[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]]
        )

        Rz = np.array(
            [[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]]
        )

        return Rz @ Ry @ Rx

    def _equirectangular_to_xyz(self, width: int, height: int) -> np.ndarray:
        """Convert equirectangular coordinates to 3D unit sphere coordinates."""
        u = np.linspace(0, 1, width)
        v = np.linspace(0, 1, height)
        u, v = np.meshgrid(u, v)

        theta = (u - 0.5) * 2 * np.pi
        phi = (0.5 - v) * np.pi

        x = np.cos(phi) * np.sin(theta)
        y = np.sin(phi)
        z = np.cos(phi) * np.cos(theta)

        return np.stack([x, y, z], axis=-1)

    def _xyz_to_equirectangular(self, xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert 3D coordinates back to equirectangular UV coordinates."""
        x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]

        theta = np.arctan2(x, z)
        phi = np.arcsin(np.clip(y, -1, 1))

        u = theta / (2 * np.pi) + 0.5
        v = 0.5 - phi / np.pi

        return u, v

    def _apply_fill_mode(
        self, u: np.ndarray, v: np.ndarray, fill_mode: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply fill mode for out-of-bounds coordinates."""
        if fill_mode == "Mirror":
            # FIX #14: the previous formula produced wrong values at integer UV
            # boundaries (u=0, u=1, u=2 gave 0 instead of the correct 1/0/1 from
            # a triangle wave).  The correct mirror formula is simply abs(mod(x,2)-1).
            u = np.abs(np.mod(u, 2.0) - 1.0)
            v = np.abs(np.mod(v, 2.0) - 1.0)
            mask = np.ones_like(u)
        elif fill_mode == "Repeat":
            u = np.mod(u, 1)
            v = np.mod(v, 1)
            mask = np.ones_like(u)
        elif fill_mode == "Edge":
            u = np.clip(u, 0, 1)
            v = np.clip(v, 0, 1)
            mask = np.ones_like(u)
        else:  # Black
            mask = ((u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)).astype(np.float32)
            u = np.clip(u, 0, 1)
            v = np.clip(v, 0, 1)

        return u, v, mask

    def _sample_image(
        self,
        img: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        interpolation: str,
        fill_mode: str = "Mirror",
    ) -> np.ndarray:
        """Sample image at UV coordinates with specified interpolation."""

        h, w = img.shape[:2]
        x = u * (w - 1)
        y = v * (h - 1)

        interp_map = {
            "Nearest": cv2.INTER_NEAREST,
            "Bilinear": cv2.INTER_LINEAR,
            "Bicubic": cv2.INTER_CUBIC,
            "Lanczos": cv2.INTER_LANCZOS4,
        }

        map_x = x.astype(np.float32)
        map_y = y.astype(np.float32)

        # FIX #16: use BORDER_CONSTANT (fill=0) for Black fill mode.
        # Previously BORDER_REFLECT was used unconditionally. For Black fill,
        # _apply_fill_mode clips u,v to [0,1] before the mask is applied; the
        # interpolation kernel at the very boundary then sampled reflected pixels
        # instead of the correct zero fill, leaking source-image edge content into
        # the transparent/black border area before the mask multiplication.
        if fill_mode == "Black":
            border_mode = cv2.BORDER_CONSTANT
        else:
            border_mode = cv2.BORDER_REFLECT

        result = cv2.remap(
            img,
            map_x,
            map_y,
            interp_map.get(interpolation, cv2.INTER_LANCZOS4),
            borderMode=border_mode,
            borderValue=0,
        )
        return result

    def generate(
        self,
        source_image: torch.Tensor,
        projection_type: str = "Equirectangular",
        output_width: int = 4096,
        output_height: int = 2048,
        horizontal_fov: float = 360.0,
        vertical_fov: float = 180.0,
        rotation_x: float = 0.0,
        rotation_y: float = 0.0,
        rotation_z: float = 0.0,
        interpolation: str = "Lanczos",
        fill_mode: str = "Mirror",
        exposure_adjust: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        img = tensor_to_numpy_float32(source_image)
        if img.ndim == 4:
            img = img[0]

        src_h, src_w = img.shape[:2]

        # Generate equirectangular UV grid
        xyz = self._equirectangular_to_xyz(output_width, output_height)

        # Apply rotation if specified
        if rotation_x != 0 or rotation_y != 0 or rotation_z != 0:
            R = self._create_rotation_matrix(rotation_x, rotation_y, rotation_z)
            xyz = xyz @ R.T

        # Project based on type
        if projection_type == "Equirectangular":
            u, v = self._xyz_to_equirectangular(xyz)
            # Apply FOV scaling
            h_scale = horizontal_fov / 360.0
            v_scale = vertical_fov / 180.0
            u = (u - 0.5) / h_scale + 0.5
            v = (v - 0.5) / v_scale + 0.5

        elif projection_type == "Cube_Map":
            # FIX 3: FOV params silently ignored for non-equirectangular projections.
            # Log a warning so users know the inputs have no effect.
            if horizontal_fov != 360.0 or vertical_fov != 180.0:
                logger.warning(
                    "[HDR360] horizontal_fov/vertical_fov only apply to Equirectangular"
                    f" — ignored for Cube_Map."
                )
            x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
            abs_x, abs_y, abs_z = np.abs(x), np.abs(y), np.abs(z)

            # FIX #15: previous code used abs_x/abs_y/abs_z in the UV numerators,
            # losing the sign of the dominant axis.  This caused the +X and −X faces
            # (and similarly for Y and Z) to produce identical UV coordinates — half
            # the sphere mapped to the wrong source region.  Now each face is selected
            # by sign AND dominant magnitude, with the correct signed UV formula.
            face_x_pos = (x >= 0) & (abs_x >= abs_y) & (abs_x >= abs_z)
            face_x_neg = (x < 0) & (abs_x >= abs_y) & (abs_x >= abs_z)
            face_y_pos = (y >= 0) & (abs_y > abs_x) & (abs_y >= abs_z)
            face_y_neg = (y < 0) & (abs_y > abs_x) & (abs_y >= abs_z)
            face_z_pos = (z >= 0) & (abs_z > abs_x) & (abs_z > abs_y)
            # FIX 2: was a bare expression — result was computed but immediately
            # discarded (allocated a full H×W bool array for nothing). The -Z face
            # worked only because the np.where else-arm implicitly handles it.
            # Assigned properly so the code is explicit and NameError-safe.
            face_z_neg = (z < 0) & (abs_z > abs_x) & (abs_z > abs_y)  # noqa: F841

            u = np.where(
                face_x_pos,
                0.5 + (-z) / (2.0 * abs_x + 1e-8),
                np.where(
                    face_x_neg,
                    0.5 + (z) / (2.0 * abs_x + 1e-8),
                    np.where(
                        face_y_pos,
                        0.5 + (x) / (2.0 * abs_y + 1e-8),
                        np.where(
                            face_y_neg,
                            0.5 + (x) / (2.0 * abs_y + 1e-8),
                            np.where(
                                face_z_pos,
                                0.5 + (x) / (2.0 * abs_z + 1e-8),
                                0.5 + (-x) / (2.0 * abs_z + 1e-8),
                            ),
                        ),
                    ),
                ),
            )

            v = np.where(
                face_x_pos,
                0.5 + (y) / (2.0 * abs_x + 1e-8),
                np.where(
                    face_x_neg,
                    0.5 + (y) / (2.0 * abs_x + 1e-8),
                    np.where(
                        face_y_pos,
                        0.5 + (-z) / (2.0 * abs_y + 1e-8),
                        np.where(
                            face_y_neg,
                            0.5 + (z) / (2.0 * abs_y + 1e-8),
                            np.where(
                                face_z_pos,
                                0.5 + (y) / (2.0 * abs_z + 1e-8),
                                0.5 + (y) / (2.0 * abs_z + 1e-8),
                            ),
                        ),
                    ),
                ),
            )

        elif projection_type == "Mirror_Ball":
            if horizontal_fov != 360.0 or vertical_fov != 180.0:
                logger.warning(
                    "[HDR360] horizontal_fov/vertical_fov only apply to Equirectangular"
                    f" — ignored for Mirror_Ball."
                )
            x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
            # FIX #17: removed unused variable r = sqrt(x²+y²) that was computed
            # but never referenced; the formula correctly uses m as the denominator.
            m = 2 * np.sqrt(x**2 + y**2 + (z + 1) ** 2 + 1e-8)
            u = x / m + 0.5
            v = y / m + 0.5

        elif projection_type == "Angular_Map":
            if horizontal_fov != 360.0 or vertical_fov != 180.0:
                logger.warning(
                    "[HDR360] horizontal_fov/vertical_fov only apply to Equirectangular"
                    f" — ignored for Angular_Map."
                )
            x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
            r = np.arccos(np.clip(z, -1, 1)) / np.pi
            phi = np.arctan2(y, x)
            u = r * np.cos(phi) * 0.5 + 0.5
            v = r * np.sin(phi) * 0.5 + 0.5

        # Apply fill mode
        u, v, mask = self._apply_fill_mode(u, v, fill_mode)

        # Sample the source image
        panorama = self._sample_image(img, u, v, interpolation, fill_mode)

        # Apply mask for black fill mode
        if fill_mode == "Black":
            panorama = panorama * mask[..., np.newaxis]

        # Apply exposure adjustment
        if exposure_adjust != 0:
            panorama = panorama * (2.0**exposure_adjust)

        # Create UV map for visualization
        uv_map = np.stack([u, v, mask], axis=-1).astype(np.float32)

        # FIX: bypass numpy_to_tensor_float32 — its batch-dim behavior varies
        # between package implementations. In this subpackage it already adds
        # a batch dim, so our previous .unsqueeze(0) created a 5D tensor
        # (1, 1, H, W, C). ComfyUI's save_images iterated dim-0 giving PIL
        # a 4D array — PIL typekey (1,1,4096,3) → KeyError.
        # torch.from_numpy gives exact (H,W,C), .unsqueeze(0) gives (1,H,W,C).
        # Ensure panorama is 3D before conversion — cv2.remap always returns
        # (out_H, out_W, C) for color inputs, but be explicit for safety.
        if panorama.ndim == 2:
            panorama = panorama[..., np.newaxis]  # grayscale → (H,W,1)
        if uv_map.ndim == 2:
            uv_map = uv_map[..., np.newaxis]
        panorama_t = torch.from_numpy(panorama.astype(np.float32)).unsqueeze(0)  # (1,H,W,C)
        uv_map_t   = torch.from_numpy(uv_map.astype(np.float32)).unsqueeze(0)   # (1,H,W,3)
        return (panorama_t, uv_map_t)


# =============================================================================
# NODE MAPPINGS
# FIX #13: NODE_CLASS_MAPPINGS was absent — HDR360Generate was invisible to
# ComfyUI and could not be used in any workflow.
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "RadianceHDR360Generate": HDR360Generate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceHDR360Generate": "◎ HDR 360 Panorama Generate",
}
