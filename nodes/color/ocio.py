import os
import logging
import folder_paths
from radiance.radiance_ocio import get_ocio_manager, HAS_OCIO
from radiance.path_utils import strip_path_quotes

logger = logging.getLogger("radiance.ocio")


class RadianceOCIOContext:
    """Load an OpenColorIO config for the session.

    3.5.0: the node had no output node flag and nothing consumed its output,
    so ComfyUI never ran it and the config was never loaded. It is now an
    output node (it runs whenever it is in the graph) and its ocio_context
    output can be wired into Color Space Convert, which then runs after it.
    A config that fails to load is an error instead of a status field nobody
    read. The working_space widget, stored and never used, is gone.
    """

    CATEGORY = "FXTD STUDIOS/Radiance/◎ Color"
    DESCRIPTION = ("Load an OpenColorIO config into Radiance's shared OCIO manager for this session. "
                   "Color Space Convert uses it for the working and display spaces; wire ocio_context "
                   "into it to be sure the config is loaded first.")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config_path": ("STRING", {
                    "default": "C:/ACES/config.ocio", "multiline": False,
                    "tooltip": "Path to a config.ocio file. A relative path is looked for in ComfyUI's input/ then output/ folder. The loaded config replaces the session-wide one until another is loaded.",
                }),
            },
        }

    RETURN_TYPES = ("RADIANCE_OCIO",)
    RETURN_NAMES = ("ocio_context",)
    OUTPUT_TOOLTIPS = ("The loaded config (path and name). Optional input of Color Space Convert.",)
    FUNCTION = "set_context"

    @classmethod
    def IS_CHANGED(cls, config_path, **_):
        path = _resolve_config_path(config_path)
        try:
            return f"{path}:{os.path.getmtime(path)}"
        except OSError:
            return path

    def set_context(self, config_path, **_stale):
        # **_stale: a saved graph from before 3.5.0 still sends working_space.
        if not HAS_OCIO:
            raise RuntimeError("OCIO Context: PyOpenColorIO is not installed "
                               "(pip install opencolorio).")
        path = _resolve_config_path(config_path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"OCIO Context: no config at {path!r}.")
        mgr = get_ocio_manager()
        if not mgr.load_config(path):
            raise RuntimeError(f"OCIO Context: {path!r} is not a config OpenColorIO can load; "
                               "see the console for OCIO's own message.")
        n_spaces = len(mgr.get_scene_color_spaces())
        text = f"OCIO: {mgr.config_name} ({n_spaces} colour spaces)\n{path}"
        context = {"status": "active", "path": mgr.config_path or path,
                   "config_name": mgr.config_name}
        return {"ui": {"text": [text]}, "result": (context,)}


def _resolve_config_path(config_path: str) -> str:
    config_path = strip_path_quotes(config_path or "")
    if config_path and not os.path.isabs(config_path):
        for search_dir in (folder_paths.get_input_directory(), folder_paths.get_output_directory()):
            candidate = os.path.join(search_dir, config_path)
            if os.path.exists(candidate):
                return candidate
    return config_path


def ensure_context_loaded(ocio_context) -> None:
    """Load ``ocio_context``'s config if another one replaced it since."""
    if not isinstance(ocio_context, dict) or ocio_context.get("status") != "active":
        return
    mgr = get_ocio_manager()
    path = ocio_context.get("path")
    if path and getattr(mgr, "config_path", None) != path:
        mgr.load_config(path)
