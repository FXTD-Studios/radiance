"""
◎ Radiance Multipass AOV Reader

Reads a *real* multilayer OpenEXR (renderer AOVs) and splits its named layers
into the same outputs as RadianceMultipassMaster — so genuine render passes
(diffuse/albedo, N, P, Z, specular, AO, cryptomatte, …) flow straight into the
same EXR-passes writer and relight/comp chain.

This is the "make it real" counterpart to the Master extractor: instead of
*estimating* passes from a 2D beauty, it ingests ground-truth layers produced by
Arnold / Redshift / Karma / Cycles / V-Ray, preserving scene-linear values
(no normalization). Layers that are absent in the EXR come through as black, so
the Master node can still be used to gap-fill the missing ones.

Channel/layer mapping is alias-based and case-insensitive (e.g. "diffuse",
"diffuse_color", "albedo", "basecolor" all map to the albedo slot). A per-read
report of which layers were found and how they mapped is logged.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .core import _flow_to_hsv_image
from ....core.system.path_utils import strip_path_quotes

logger = logging.getLogger("radiance.vfx_multipass.aov_reader")

# Output slot -> accepted layer-name aliases (all lowercased, dots/spaces/dashes
# stripped to underscores before matching).
_LAYER_ALIASES: Dict[str, Tuple[str, ...]] = {
    "beauty":         ("", "rgba", "rgb", "beauty", "composite", "combined", "final"),
    "albedo":         ("albedo", "diffuse", "diffuse_color", "diffusecolor", "basecolor", "base_color", "diff", "diffuse_albedo"),
    "normal":         ("n", "normal", "normals", "normal_world", "worldnormal", "world_normal"),
    "depth":          ("z", "depth", "zdepth", "z_depth"),
    "roughness":      ("roughness", "rough"),
    "specular":       ("specular", "spec", "reflect", "reflection_color"),
    "metallic":       ("metallic", "metalness", "metal"),
    "ao":             ("ao", "occlusion", "ambient_occlusion", "ambocc"),
    "emission":       ("emission", "emit", "emissive", "incandescence", "self_illumination"),
    "transmission":   ("transmission", "refraction", "transmissive", "refract"),
    "highpass":       ("highpass", "high_pass"),
    "world_position": ("p", "position", "worldposition", "world_position", "point", "world_point"),
    "curvature":      ("curvature", "curv"),
    "shadow_mask":    ("shadow", "shadow_mask", "shadows"),
    "midtone_mask":   ("midtone", "midtone_mask", "midtones"),
    "highlight_mask": ("highlight", "highlight_mask", "highlights"),
    "reflection_mask": ("reflection_mask", "reflectionmask"),
    "motion_vector":  ("mv", "motion", "motionvector", "motion_vector", "velocity", "vector", "motionvectors"),
    "object_id":      ("id", "crypto", "cryptomatte", "objectid", "object_id", "segmentation", "matteid"),
}

# Output order MUST match RadianceMultipassMaster so this node is a drop-in.
_OUTPUT_ORDER: Tuple[str, ...] = (
    "beauty", "albedo", "normal", "depth", "roughness", "specular", "metallic",
    "ao", "emission", "transmission", "highpass", "world_position", "curvature",
    "shadow_mask", "midtone_mask", "highlight_mask", "reflection_mask",
    "motion_vector", "object_id",
)


def _norm_layer_name(name: str) -> str:
    return name.strip().lower().replace(".", "_").replace(" ", "_").replace("-", "_")


def _suffix_rank(suf: str) -> Tuple[int, str]:
    order = {
        "R": 0, "G": 1, "B": 2, "A": 3,
        "X": 0, "Y": 1, "Z": 2,
        "NX": 0, "NY": 1, "NZ": 2,
    }
    s = suf.upper()
    return (order.get(s, 99), suf)


def _layer_key_for_channel(channel_name: str, default_layer: str = "") -> Tuple[str, str]:
    if "." in channel_name:
        return channel_name.rsplit(".", 1)
    if channel_name.upper() == "Z":
        return "depth", "Z"
    return default_layer, channel_name


def _layers_from_channel_groups(
    groups: Dict[str, Dict[str, np.ndarray]],
    h: int,
    w: int,
) -> Dict[str, np.ndarray]:
    layers: Dict[str, np.ndarray] = {}
    for layer, suffices in groups.items():
        ordered = sorted(suffices.items(), key=lambda kv: _suffix_rank(kv[0]))
        planes = [np.asarray(plane, dtype=np.float32).reshape(h, w) for _suf, plane in ordered]
        if planes:
            layers[layer] = np.stack(planes, axis=-1)
    return layers


def _window_bounds(window) -> Tuple[int, int, int, int]:
    return (window.min.x, window.min.y, window.max.x, window.max.y)


def _canvas_bounds(data_windows, display_window) -> Tuple[int, int, int, int]:
    bounds = [_window_bounds(display_window)] + [_window_bounds(window) for window in data_windows]
    return (
        min(item[0] for item in bounds), min(item[1] for item in bounds),
        max(item[2] for item in bounds), max(item[3] for item in bounds),
    )


def _place_in_canvas(plane: np.ndarray, data_window, canvas_bounds) -> np.ndarray:
    dx0, dy0, dx1, dy1 = canvas_bounds
    x0, y0, x1, y1 = _window_bounds(data_window)
    canvas = np.zeros((dy1 - dy0 + 1, dx1 - dx0 + 1), dtype=np.float32)
    canvas[y0 - dy0:y1 - dy0 + 1, x0 - dx0:x1 - dx0 + 1] = plane
    return canvas


def _read_multipart_exr(OpenEXR, Imath, path: str) -> Tuple[Dict[str, np.ndarray], int, int, Dict[str, Any]]:
    if not hasattr(OpenEXR, "MultiPartInputFile"):
        raise RuntimeError("OpenEXR binding does not expose MultiPartInputFile")

    f = OpenEXR.MultiPartInputFile(path)
    try:
        part_count = None
        for attr in ("parts", "numParts", "num_parts"):
            value = getattr(f, attr, None)
            if value is None:
                continue
            part_count = int(value() if callable(value) else value)
            break
        if part_count is None:
            raise RuntimeError("OpenEXR MultiPartInputFile did not expose a part count")

        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        groups: Dict[str, Dict[str, np.ndarray]] = {}
        headers = [f.header(part_index) for part_index in range(part_count)]
        first_display = _window_bounds(headers[0].get("displayWindow", headers[0]["dataWindow"]))
        for hdr in headers[1:]:
            if _window_bounds(hdr.get("displayWindow", hdr["dataWindow"])) != first_display:
                raise RuntimeError("Multipart EXR parts have different displayWindow bounds")
        data_windows = [hdr["dataWindow"] for hdr in headers]
        canvas = _canvas_bounds(data_windows, headers[0].get("displayWindow", data_windows[0]))
        first_w = canvas[2] - canvas[0] + 1
        first_h = canvas[3] - canvas[1] + 1

        for part_index, hdr in enumerate(headers):
            dw = hdr["dataWindow"]
            w = dw.max.x - dw.min.x + 1
            h = dw.max.y - dw.min.y + 1

            part_name = str(hdr.get("name", f"part{part_index}"))
            for chname in list(hdr["channels"].keys()):
                buf = None
                last_error: Optional[Exception] = None
                for args in (
                    (part_index, chname, pt),
                    (part_index, chname),
                    (chname, pt, part_index),
                    (chname, pt),
                ):
                    try:
                        buf = f.channel(*args)
                        break
                    except Exception as exc:
                        last_error = exc
                        continue
                if buf is None:
                    raise RuntimeError(f"Could not read channel '{chname}' from part '{part_name}': {last_error}")

                layer, suffix = _layer_key_for_channel(chname, default_layer=part_name)
                plane = np.frombuffer(buf, dtype=np.float32).reshape(h, w)
                groups.setdefault(layer, {})[suffix] = _place_in_canvas(plane, dw, canvas)

        metadata = {
            "data_windows": [_window_bounds(window) for window in data_windows],
            "display_window": first_display,
            "canvas_window": canvas,
        }
        return _layers_from_channel_groups(groups, first_h, first_w), first_h, first_w, metadata
    finally:
        close = getattr(f, "close", None)
        if callable(close):
            close()


def _read_singlepart_exr(OpenEXR, Imath, path: str) -> Tuple[Dict[str, np.ndarray], int, int, Dict[str, Any]]:
    f = OpenEXR.InputFile(path)
    try:
        hdr = f.header()
        dw = hdr["dataWindow"]
        display = hdr.get("displayWindow", dw)
        w = dw.max.x - dw.min.x + 1
        h = dw.max.y - dw.min.y + 1
        canvas = _canvas_bounds([dw], display)
        display_w = canvas[2] - canvas[0] + 1
        display_h = canvas[3] - canvas[1] + 1
        pt = Imath.PixelType(Imath.PixelType.FLOAT)

        groups: Dict[str, Dict[str, np.ndarray]] = {}
        for chname in list(hdr["channels"].keys()):
            layer, suffix = _layer_key_for_channel(chname)
            buf = f.channel(chname, pt)
            plane = np.frombuffer(buf, dtype=np.float32).reshape(h, w)
            groups.setdefault(layer, {})[suffix] = _place_in_canvas(plane, dw, canvas)
        metadata = {
            "data_window": _window_bounds(dw),
            "display_window": _window_bounds(display),
            "canvas_window": canvas,
        }
        return _layers_from_channel_groups(groups, display_h, display_w), display_h, display_w, metadata
    finally:
        f.close()


def _read_multilayer_exr(path: str) -> Tuple[Dict[str, np.ndarray], int, int, Dict[str, Any]]:
    """Return ({layer_name: (H,W,C) float32}, H, W) for a multilayer EXR.

    Channels are grouped by the text before the final '.'; the default layer
    (channels with no '.') is keyed as "". Channel suffixes are ordered
    canonically (R,G,B,A then X,Y,Z then alphabetical) so colour and vector
    layers assemble predictably. Scene-linear values are preserved exactly.
    """
    try:
        import OpenEXR  # type: ignore
        import Imath    # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Reading multilayer EXR requires the OpenEXR + Imath packages "
            "(pip install OpenEXR Imath)."
        ) from exc

    if not os.path.isfile(path):
        raise RuntimeError(f"EXR not found: {path}")

    if hasattr(OpenEXR, "MultiPartInputFile"):
        try:
            return _read_multipart_exr(OpenEXR, Imath, path)
        except Exception as exc:
            logger.debug("[Radiance AOV Reader] Multipart read path skipped: %s", exc)

    return _read_singlepart_exr(OpenEXR, Imath, path)


def _to_image_tensor(arr: Optional[np.ndarray], h: int, w: int) -> torch.Tensor:
    """Coerce a layer to a (1, H, W, 3) float32 IMAGE tensor (scene-linear preserved)."""
    if arr is None:
        return torch.zeros(1, h, w, 3, dtype=torch.float32)
    a = np.ascontiguousarray(arr.astype(np.float32))
    if a.ndim == 2:
        a = a[..., None]
    c = a.shape[2]
    if c == 1:
        a = np.repeat(a, 3, axis=2)          # scalar -> grey RGB
    elif c == 2:
        a = np.concatenate([a, np.zeros_like(a[..., :1])], axis=2)  # XY -> XY0
    elif c >= 4:
        a = a[..., :3]                        # drop alpha/extra for IMAGE slot
    return torch.from_numpy(a).unsqueeze(0)


def _motion_visualization(arr: Optional[np.ndarray], h: int, w: int) -> torch.Tensor:
    if arr is None or arr.ndim < 3 or arr.shape[2] < 2:
        return torch.zeros(1, h, w, 3, dtype=torch.float32)
    vectors = torch.from_numpy(np.ascontiguousarray(arr[..., :2], dtype=np.float32))
    return _flow_to_hsv_image(vectors[..., 0].unsqueeze(0), vectors[..., 1].unsqueeze(0))


class RadianceMultipassAOVReader:
    """◎ Multipass: AOV Reader — split a real multilayer EXR into Radiance passes."""

    CATEGORY = "FXTD STUDIOS/Radiance/VFX"
    DESCRIPTION = (
        "Read a real multilayer/AOV OpenEXR and split its named layers into the "
        "same passes as the Master extractor. Ground-truth renderer passes — not "
        "estimates. Missing layers come through black."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "exr_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to a multilayer/AOV EXR (Arnold/Redshift/Karma/Cycles/V-Ray).",
                }),
            },
            "optional": {
                "beauty_layer": ("STRING", {"default": "auto",
                    "tooltip": "Override the beauty layer name, or 'auto' to detect (RGBA / 'beauty')."}),
                "albedo_layer": ("STRING", {"default": "auto",
                    "tooltip": "Override the albedo/diffuse layer name, or 'auto'."}),
                "normal_layer": ("STRING", {"default": "auto",
                    "tooltip": "Override the normal layer name, or 'auto'."}),
                "depth_layer": ("STRING", {"default": "auto",
                    "tooltip": "Override the depth (Z) layer name, or 'auto'."}),
            },
        }

    # Mirrors RadianceMultipassMaster exactly so the two are interchangeable.
    RETURN_TYPES = (
        "RADIANCE_PASSES", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE",
        "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE",
        "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE",
    )
    RETURN_NAMES = (
        "passes", "beauty", "albedo", "normal_map", "depth", "roughness", "specular",
        "metallic", "ao", "emission", "transmission", "highpass", "world_position",
        "curvature", "shadow_mask", "midtone_mask", "highlight_mask", "reflection_mask",
        "motion_vector", "segmentation_id", "motion_visualization", "alpha",
    )
    FUNCTION = "read_passes"

    def read_passes(
        self,
        exr_path: str,
        beauty_layer: str = "auto",
        albedo_layer: str = "auto",
        normal_layer: str = "auto",
        depth_layer: str = "auto",
    ) -> Tuple:
        exr_path = strip_path_quotes(exr_path)
        layers, h, w, window_metadata = _read_multilayer_exr(exr_path)

        # Normalized lookup of available layers.
        norm_to_raw = {_norm_layer_name(k): k for k in layers}

        manual = {
            "beauty": beauty_layer, "albedo": albedo_layer,
            "normal": normal_layer, "depth": depth_layer,
        }

        resolved: Dict[str, Optional[np.ndarray]] = {}
        report: List[str] = []
        used_layers = set()

        for slot in _OUTPUT_ORDER:
            chosen_raw: Optional[str] = None

            # 1. explicit manual override (exact or normalized match)
            ov = manual.get(slot, "auto")
            if ov and ov.strip().lower() not in ("", "auto"):
                key = _norm_layer_name(ov)
                chosen_raw = norm_to_raw.get(key) or (ov if ov in layers else None)
                if chosen_raw is None:
                    raise ValueError(f"[Radiance AOV Reader] Layer override '{ov}' for {slot} was not found.")

            # 2. alias auto-match: exact first, then prefix for multi-char aliases
            #    so 'crypto' matches 'crypto00'/'crypto01', 'diffuse' matches
            #    'diffuse_indirect', etc. Short aliases (n, z, p, id, mv) stay exact.
            if chosen_raw is None:
                avail_sorted = sorted(norm_to_raw)
                for alias in _LAYER_ALIASES.get(slot, ()):  # ordered by preference
                    if alias in norm_to_raw:
                        chosen_raw = norm_to_raw[alias]
                        break
                    if len(alias) >= 4:
                        pref = next((n for n in avail_sorted if n.startswith(alias)), None)
                        if pref is not None:
                            chosen_raw = norm_to_raw[pref]
                            break

            if chosen_raw is not None and chosen_raw in layers:
                resolved[slot] = layers[chosen_raw]
                used_layers.add(chosen_raw)
                report.append(f"{slot} <- '{chosen_raw or 'RGBA'}'")
            else:
                resolved[slot] = None
                report.append(f"{slot} <- (none, black)")

        if resolved["beauty"] is None:
            logger.warning(
                "[Radiance AOV Reader] No beauty/RGBA layer matched in %s; beauty output is black. "
                "Set beauty_layer explicitly if your renderer uses a custom name.",
                os.path.basename(exr_path),
            )

        unmapped = sorted(set(layers) - used_layers)
        logger.info(
            "[Radiance AOV Reader] %s (%dx%d): %d layers, mapped: %s%s",
            os.path.basename(exr_path), w, h, len(layers), "; ".join(report),
            f"  | unmapped: {', '.join(repr(u) for u in unmapped)}" if unmapped else "",
        )

        images = {slot: _to_image_tensor(resolved[slot], h, w) for slot in _OUTPUT_ORDER}
        beauty_arr = resolved["beauty"]
        alpha = (
            _to_image_tensor(beauty_arr[..., 3:4], h, w)
            if beauty_arr is not None and beauty_arr.shape[-1] >= 4
            else torch.ones(1, h, w, 3, dtype=torch.float32)
        )
        motion_visualization = _motion_visualization(resolved["motion_vector"], h, w)

        passes_dict = dict(images)
        # Keep the internal dict key as object_id for compatibility with the
        # EXR-passes writer / relight nodes, while the socket label reads
        # 'segmentation_id' to avoid implying real Cryptomatte support.
        passes_dict["object_id"] = images["object_id"]
        passes_dict["motion_visualization"] = motion_visualization
        passes_dict["alpha"] = alpha
        passes_dict["_present"] = [slot for slot in _OUTPUT_ORDER if resolved[slot] is not None]
        if beauty_arr is not None and beauty_arr.shape[-1] >= 4:
            passes_dict["_present"].append("alpha")
        passes_dict.update({f"_{key}": value for key, value in window_metadata.items()})

        return (passes_dict,) + tuple(images[slot] for slot in _OUTPUT_ORDER) + (motion_visualization, alpha)


NODE_CLASS_MAPPINGS = {
    "RadianceMultipassAOVReader": RadianceMultipassAOVReader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceMultipassAOVReader": "Multipass AOV Reader",
}
