"""
Studio integration nodes for sending renders to DCC applications.

RadianceNukeSend  — Export EXR + .nk Read-node snippet for Nuke, optionally
                    loading it into a running Nuke through the Radiance listener.
RadianceDaVinciSend — Export to a DaVinci Resolve media folder, optionally
                    importing it into the open project's Media Pool.
"""
from __future__ import annotations

import logging
import json
import subprocess
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from radiance.nodes.io.write import _save_exr, _save_pil_image

logger = logging.getLogger("radiance.nodes.studio")

# ── Colour of the pixels being sent ─────────────────────────────────────────
# EXR is scene-linear by convention; PNG and TIFF hold a display (sRGB) signal.
# The nodes used to write the tensor as it came to every format, so one of the
# two was always wrong: a scene-linear frame went to Resolve as a dark TIFF,
# a display-referred one to Nuke as an EXR with the sRGB curve baked in.
# "As is" keeps that behaviour for graphs that already compensate.
INPUT_SPACES = ["As is (no conversion)", "Scene-linear", "sRGB display"]


def _srgb_encode(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, None)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)


def _srgb_decode(v: np.ndarray) -> np.ndarray:
    v = np.clip(v, 0.0, None)
    return np.where(v <= 0.04045, v / 12.92, np.power((v + 0.055) / 1.055, 2.4))


def _for_format(frame: np.ndarray, input_space: str, linear_file: bool) -> np.ndarray:
    """`frame` converted to what the file format expects (colour channels only)."""
    if input_space.startswith("As is"):
        return frame
    is_linear = input_space == "Scene-linear"
    if is_linear == linear_file:
        return frame
    out = frame.copy()
    rgb = out[..., :3]
    out[..., :3] = _srgb_decode(rgb) if linear_file else _srgb_encode(rgb)
    return out.astype(np.float32)


# ── Nuke ────────────────────────────────────────────────────────────────────

_NUKE_NK_TEMPLATE = """\
version {nuke_version}
Read {{
 inputs 0
 file "{filepath}"
 first {first}
 last {last}
 origfirst {first}
 origlast {last}
 raw true
 name {node_name}
 xpos 0
 ypos 0
}}
"""


def _nk_escape(path: str) -> str:
    """A path as a double-quoted TCL string in a .nk file."""
    return path.replace("\\", "/").replace('"', '\\"').replace("[", "\\[").replace("$", "\\$")


class RadianceNukeSend:
    CATEGORY = "FXTD STUDIOS/Radiance/07 Pipeline & DCC"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("status", "render_path")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Frame to export. A batch is written as a numbered EXR sequence.",
                }),
                "nuke_folder": ("STRING", {
                    "default": "",
                    "placeholder": "/renders/nuke_exchange/",
                    "tooltip": "Output folder for image + .nk file. Created if missing.",
                }),
                "filename": ("STRING", {
                    "default": "radiance_out",
                    "tooltip": "Base name for the EXR file(s). The Read node gets the same name with "
                               "anything Nuke does not allow in a node name replaced by _.",
                }),
            },
            "optional": {
                "frame_start": ("INT", {
                    "default": 1001, "min": 0, "max": 999999,
                    "tooltip": "Starting frame number for the EXR sequence.",
                }),
                "push_to_nuke": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Also create or update the Read node in a running Nuke. Needs the Radiance "
                               "listener running there (scripts/start_nuke_server.py). Both sides share a "
                               "token from ~/.radiance/dcc_token, created automatically, or from "
                               "RADIANCE_DCC_AUTH_TOKEN; for Nuke on another machine copy that file or set the "
                               "variable there.",
                }),
                "nuke_host": ("STRING", {
                    "default": "127.0.0.1",
                    "tooltip": "Nuke listener host (used only when push_to_nuke is on). RADIANCE_NUKE_HOST "
                               "replaces the default.",
                }),
                "nuke_port": ("INT", {
                    "default": 1986, "min": 1024, "max": 65535,
                    "tooltip": "Nuke listener port (used only when push_to_nuke is on). RADIANCE_NUKE_PORT "
                               "replaces the default.",
                }),
                "half_float": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Write 16-bit half EXR (True) or 32-bit float EXR (False).",
                }),
                "input_space": (INPUT_SPACES, {
                    "default": INPUT_SPACES[0],
                    "tooltip": "What the image holds. EXR is written scene-linear: sRGB display input is "
                               "linearised first. As is: written unchanged, as before.",
                }),
            },
        }

    DESCRIPTION = (
        "Export image as EXR and write a .nk Read-node snippet for direct Nuke import. "
        "Optionally load it into a running Nuke through the Radiance listener."
    )
    FUNCTION = "run"

    def run(
        self,
        image: torch.Tensor,
        nuke_folder: str,
        filename: str,
        frame_start: int = 1001,
        push_to_nuke: bool = False,
        nuke_host: str = "127.0.0.1",
        nuke_port: int = 1986,
        half_float: bool = True,
        input_space: str = INPUT_SPACES[0],
    ) -> Tuple[str, str]:
        from radiance.config.env import get_nuke_host, get_nuke_port
        from radiance.path_utils import strip_path_quotes
        if nuke_host == "127.0.0.1":
            nuke_host = get_nuke_host()
        if nuke_port == 1986:
            nuke_port = get_nuke_port()

        nuke_folder = strip_path_quotes(nuke_folder or "")
        if not nuke_folder:
            return ("Error: nuke_folder is required.", "")

        frames = image.detach().cpu().float().numpy()
        if frames.ndim == 3:
            frames = frames[np.newaxis]

        out_dir = Path(nuke_folder)
        out_dir.mkdir(parents=True, exist_ok=True)

        written = []
        for i in range(len(frames)):
            frame_num = frame_start + i
            if len(frames) == 1:
                fname = f"{filename}.exr"
            else:
                fname = f"{filename}.{frame_num:04d}.exr"
            fpath = out_dir / fname
            _save_exr(_for_format(frames[i], input_space, linear_file=True), fpath, half=half_float)
            written.append(str(fpath))

        first_frame = frame_start
        last_frame = frame_start + len(frames) - 1

        if len(frames) == 1:
            file_field = str(out_dir / f"{filename}.exr").replace("\\", "/")
        else:
            file_field = str(out_dir / f"{filename}.####.exr").replace("\\", "/")

        from radiance.tools.nuke_connector import nuke_node_name
        node_name = nuke_node_name(filename)
        # The path is quoted (a folder with a space broke the Read node) and
        # the node name is one Nuke accepts (a hyphen used to break it).
        nk_content = _NUKE_NK_TEMPLATE.format(
            nuke_version="15.0",
            filepath=_nk_escape(file_field),
            first=first_frame,
            last=last_frame,
            node_name=node_name,
        )

        nk_path = out_dir / f"{filename}.nk"
        nk_path.write_text(nk_content, encoding="utf-8")

        status = f"{len(written)} EXR frame(s) + .nk → {out_dir}"
        render_path = str(out_dir)

        if push_to_nuke:
            push_status = self._push_to_nuke(
                file_field, node_name, first_frame, last_frame, nuke_host, nuke_port
            )
            status += f" | nuke push: {push_status}"

        return (status, render_path)

    @staticmethod
    def _push_to_nuke(
        filepath: str,
        node_name: str,
        first_frame: int,
        last_frame: int,
        host: str,
        port: int,
    ) -> str:
        try:
            from radiance.tools.nuke_connector import NukeConnector
        except ImportError:
            return "NukeConnector not available"

        try:
            conn = NukeConnector(host=host, port=port)
            ok, msg = conn.load_exr(
                filepath=filepath,
                node_name=node_name,
                first_frame=first_frame,
                last_frame=last_frame,
                current_frame=first_frame,
                color_space="linear",
                connect_viewer=True,
                raw=True,
            )
        except Exception as e:  # noqa: BLE001 - the files are written; report, don't fail the node
            return f"FAILED ({e})"
        if ok:
            return f"OK ({msg})"
        return f"FAILED ({msg})"


# ── DaVinci Resolve ─────────────────────────────────────────────────────────

def import_into_resolve(paths: List[str], first: int, last: int, sequence_pattern: Optional[str]) -> str:
    """Run the native Resolve API outside ComfyUI so a DLL crash is contained."""
    worker = Path(__file__).resolve().parents[2] / "tools" / "resolve_import.py"
    request = dict(paths=paths, first=first, last=last, sequence_pattern=sequence_pattern)
    try:
        completed = subprocess.run(
            [sys.executable, str(worker)], input=json.dumps(request),
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except subprocess.TimeoutExpired:
        return "not imported: Resolve scripting timed out after 30 seconds; exported files are intact"
    except OSError as exc:
        return f"not imported: could not start Resolve scripting worker ({exc})"
    if completed.returncode:
        return (f"not imported: Resolve scripting worker exited with code {completed.returncode}; "
                "check the Resolve scripting/Python installation. Exported files are intact")
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith("RADIANCE_RESOLVE_RESULT:"):
            try:
                status = json.loads(line.partition(":")[2])
                if isinstance(status, str):
                    return status
            except ValueError:
                break
    return "not imported: Resolve scripting worker returned no valid result; exported files are intact"


class RadianceDaVinciSend:
    CATEGORY = "FXTD STUDIOS/Radiance/07 Pipeline & DCC"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("status", "render_path")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Frame to export. Batches write numbered files.",
                }),
                "resolve_folder": ("STRING", {
                    "default": "",
                    "placeholder": "/tmp/resolve_exchange",
                    "tooltip": "DaVinci Resolve shared media folder. Created if missing.",
                }),
                "filename": ("STRING", {
                    "default": "radiance_out",
                    "tooltip": "Base filename (no extension).",
                }),
                "bit_depth": (["16bit", "8bit", "EXR"], {
                    "default": "16bit",
                    "tooltip": "16bit: 16-bit TIFF. 8bit: PNG. EXR: 16-bit half-float EXR.",
                }),
            },
            "optional": {
                "frame_start": ("INT", {
                    "default": 1001, "min": 0, "max": 999999,
                    "tooltip": "Starting frame number for numbered sequences.",
                }),
                "input_space": (INPUT_SPACES, {
                    "default": INPUT_SPACES[0],
                    "tooltip": "What the image holds. TIFF and PNG are written as sRGB display images "
                               "(scene-linear input is encoded with the sRGB curve, clipped at 1.0); EXR "
                               "is written scene-linear (sRGB input is linearised). As is: unchanged, as before.",
                }),
                "import_to_media_pool": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Also import the files into the open project's Media Pool, through Resolve's "
                               "scripting API. Needs Resolve running on this machine with Preferences > "
                               "System > General > External scripting using: Local (Resolve Studio). "
                               "Each run imports again.",
                }),
            },
        }

    DESCRIPTION = (
        "Export the current image to a DaVinci Resolve media folder as 16-bit TIFF, 8-bit PNG or EXR, "
        "and optionally import it into the open project's Media Pool."
    )
    FUNCTION = "run"

    def run(
        self,
        image: torch.Tensor,
        resolve_folder: str,
        filename: str,
        bit_depth: str = "16bit",
        frame_start: int = 1001,
        input_space: str = INPUT_SPACES[0],
        import_to_media_pool: bool = False,
    ) -> Tuple[str, str]:
        from radiance.path_utils import strip_path_quotes
        resolve_folder = strip_path_quotes(resolve_folder or "")
        if not resolve_folder:
            return ("Error: resolve_folder is required.", "")

        frames = image.detach().cpu().float().numpy()
        if frames.ndim == 3:
            frames = frames[np.newaxis]

        out_dir = Path(resolve_folder)
        out_dir.mkdir(parents=True, exist_ok=True)

        ext = {"EXR": "exr", "8bit": "png"}.get(bit_depth, "tif")
        written = []
        for i in range(len(frames)):
            frame_num = frame_start + i
            fname = f"{filename}.{ext}" if len(frames) == 1 else f"{filename}.{frame_num:04d}.{ext}"
            fpath = out_dir / fname
            if bit_depth == "EXR":
                _save_exr(_for_format(frames[i], input_space, linear_file=True), fpath, half=True)
            elif bit_depth == "8bit":
                _save_pil_image(_for_format(frames[i], input_space, linear_file=False), fpath, "8-bit PNG")
            else:  # 16bit
                _save_pil_image(_for_format(frames[i], input_space, linear_file=False), fpath, "16-bit TIFF")
            written.append(str(fpath))

        status = f"{len(written)} frame(s) [{bit_depth}] → {out_dir}"
        if import_to_media_pool:
            pattern = None
            if len(written) > 1:
                pattern = str(out_dir / f"{filename}.%04d.{ext}")
            status += " | resolve: " + import_into_resolve(
                written, frame_start, frame_start + len(written) - 1, pattern)
        return (status, str(out_dir))


NODE_CLASS_MAPPINGS = {
    "RadianceNukeSend": RadianceNukeSend,
    "RadianceDaVinciSend": RadianceDaVinciSend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceNukeSend": "◎ Radiance Send to Nuke",
    "RadianceDaVinciSend": "◎ Radiance Send to DaVinci Resolve",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
