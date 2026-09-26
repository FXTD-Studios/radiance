import torch
import torch.nn.functional as F
import json
import logging
import numpy as np

logger = logging.getLogger("radiance.3d")

class RadianceCameraSync:
    """
    ◎ Radiance Camera Sync
    
    Imports camera metadata from JSON to drive optical nodes.

    Keys: focal_length (mm), f_stop, shutter_angle (degrees; "shutter" is
    accepted as an alias), transform (4x4). An animated camera is a "frames"
    list of those dicts; frame_offset picks the entry (clamped to the list).
    Alembic (.abc) is not read: export the camera to JSON.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "camera_data": ("STRING", {"multiline": True, "default": "{}", "tooltip": "JSON camera data: {\"focal_length\": 35.0, \"f_stop\": 2.8, \"shutter_angle\": 180.0, \"transform\": [...]} or {\"frames\": [ ... ]}"}),
                "frame_offset": ("INT", {"default": 0, "min": -10000, "max": 10000, "step": 1, "tooltip": "Index into a \"frames\" list (animated camera). Ignored for a single camera."}),
            },
            "optional": {
                "camera_file": ("STRING", {"default": "", "tooltip": "Path to a .json camera file. Takes precedence over camera_data."}),
            }
        }

    RETURN_TYPES = ("RADIANCE_CAMERA", "FLOAT", "FLOAT", "INT")
    RETURN_NAMES = ("camera", "focal_length", "f_stop", "shutter_angle")
    FUNCTION = "sync"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ VFX"
    DESCRIPTION = "Read a JSON camera (focal length, f-stop, shutter, transform) for the optical nodes."

    def sync(self, camera_data, frame_offset, camera_file=""):
        import os
        path = str(camera_file or "").strip().strip('"').strip("'")
        # Errors raise. A missing file, an .abc or bad JSON used to fall back
        # to a default 35 mm f/2.8 camera with the node still green.
        if path:
            if path.lower().endswith(".abc"):
                raise ValueError("Camera Sync reads JSON only; Alembic (.abc) is not supported. "
                                 "Export the camera to JSON.")
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Camera Sync: camera_file not found: {path}")
            with open(path, "r") as f:
                data = json.load(f)
        else:
            try:
                data = json.loads(camera_data or "{}")
            except json.JSONDecodeError as exc:
                raise ValueError(f"Camera Sync: camera_data is not valid JSON: {exc}") from exc

        frame_used = None
        frames = data.get("frames") if isinstance(data, dict) else None
        if isinstance(frames, list) and frames:
            frame_used = max(0, min(int(frame_offset), len(frames) - 1))
            data = {**{k: v for k, v in data.items() if k != "frames"}, **frames[frame_used]}

        focal_length = float(data.get("focal_length", 35.0))
        f_stop = float(data.get("f_stop", 2.8))
        shutter_angle = float(data.get("shutter_angle", data.get("shutter", 180.0)))
        transform = data.get("transform", np.eye(4).tolist())
        defaults = [k for k in ("focal_length", "f_stop", "shutter_angle")
                    if k not in data and not (k == "shutter_angle" and "shutter" in data)]

        camera_obj = {
            "focal_length": focal_length,
            "f_stop": f_stop,
            "shutter_angle": shutter_angle,
            "transform": transform,
            "frame_offset": frame_offset,
            "frame_used": frame_used,
            "defaults_used": defaults,
        }

        logger.info(f"[Camera Sync] focal {focal_length}mm, f/{f_stop}, shutter {shutter_angle}"
                    + (f", frame {frame_used}" if frame_used is not None else "")
                    + (f" (defaults for {', '.join(defaults)})" if defaults else ""))

        return (camera_obj, focal_length, f_stop, int(round(shutter_angle)))


NODE_CLASS_MAPPINGS = {
    "RadianceCameraSync": RadianceCameraSync,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceCameraSync": "◎ Radiance Camera Sync",
}
