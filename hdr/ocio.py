import logging

logger = logging.getLogger("radiance.hdr.ocio")

import os
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import torch

# NOTE: tensor_to_numpy_float32 / numpy_to_tensor_float32 are imported here
# for use by OCIOColorTransform. Previously these were imported but never used
# because the image-transform node was missing entirely.
from .utils import tensor_to_numpy_float32, numpy_to_tensor_float32

# color_utils is NOT imported here — those symbols (SRGB_TO_ACESCG, linear_to_logc4,
# etc.) were previously imported but never referenced anywhere in this module.
# OCIO handles all color math internally; the fallback path that would have used
# color_utils was never implemented. Removing the dead import block eliminates
# confusing "color_utils not found" warnings that had nothing to do with OCIO.

# Check for PyOpenColorIO
try:
    import PyOpenColorIO as OCIO

    HAS_OCIO = True
    logger.info(f"PyOpenColorIO {OCIO.__version__} loaded.")
except ImportError:
    HAS_OCIO = False
    # FIX: was logger.info — missing OCIO disables ALL image processing in this
    # module, which is a significant failure, not informational noise.
    logger.warning(
        "[Radiance OCIO] PyOpenColorIO not found — OCIO color transforms disabled. "
        "Install with: pip install opencolorio"
    )

# ── Shared constants ──────────────────────────────────────────────────────────

# FIX: was duplicated magic number (50) in _list_colorspaces AND list_spaces.
_CS_DISPLAY_LIMIT = 50


# ── Shared helpers ────────────────────────────────────────────────────────────


def _is_unconfigured_default(cfg) -> bool:
    """True when `cfg` is OCIO's built-in fallback rather than a real config.

    UNREACHABLE-FALLBACK FIX: step 3 of _resolve_config() used to accept
    whatever ``OCIO.GetCurrentConfig()`` returned as long as it was not None.
    In OCIO v2 that call never returns None and never raises: with $OCIO unset
    it hands back a built-in default, measured here on PyOpenColorIO 2.5.2 as
    a nameless config carrying exactly one colorspace, "raw". So steps 4 and 5,
    the ACES config this package bundles and the models/ACES/config.ocio that
    the "Download ACES 2.0" button installs, were dead code on every machine
    where PyOpenColorIO imports, which is every machine the OCIO nodes run on.
    Every transform then failed on colorspace names that config does not have.

    A config a host application genuinely set (Nuke, Resolve, a $OCIO export)
    has a name, colorspaces, or both. The built-in fallback has neither.
    """
    try:
        if (cfg.getName() or "").strip():
            return False
    except Exception:  # nosec B110 - older/odd bindings
        return False
    try:
        return len(list(cfg.getColorSpaces())) < 2
    except Exception:  # nosec B110 - OCIO v1 has no getColorSpaces()
        try:
            return cfg.getNumColorSpaces() < 2
        except Exception:  # nosec B110
            return False


def _resolve_colorspace_name(config, name: str) -> "Optional[str]":
    """Resolve `name` to a canonical colorspace name, or None if unknown.

    ALIAS-VALIDATION FIX: the transform node pre-validated against a set built
    from ``ColorSpace.getName()`` alone, but OCIO resolves aliases and roles
    too. In the ACES CG config this package bundles, "ACES - ACEScg",
    "ACES - ACES2065-1" and "sRGB - Texture" exist ONLY as aliases of
    "ACEScg", "ACES2065-1" and "sRGB Encoded Rec.709 (sRGB)" -- and two of
    those three are names the node's own tooltip tells the user to type. The
    node rejected them while OCIO itself would have accepted them. Ask OCIO.
    """
    if not name:
        return None
    try:
        cs = config.getColorSpace(name)
    except Exception:  # nosec B110 - v1 bindings raise on unknown names
        cs = None
    if cs is not None:
        try:
            return cs.getName()
        except Exception:  # nosec B110
            return name
    # OCIO v1 fallback: no alias support there, so a plain name match is all
    # there is.
    for known, _family in _iter_colorspaces(config):
        if known == name:
            return known
    return None


def _resolve_config(ocio_config_path: str = "") -> "Optional[OCIO.Config]":
    """
    Resolve an OCIO config from (in priority order):
      1. Explicit path argument
      2. OCIO environment variable
      3. Active process config (OCIO.GetCurrentConfig), when one was actually
         configured -- see _is_unconfigured_default() for why that qualifier
         has to be there
      4. Local Radiance ACES folder  (../ACES/config.ocio relative to this file)
      5. ComfyUI models/ACES/config.ocio

    Returns an OCIO.Config object, or None if nothing is found.
    Centralises what was previously copy-pasted across three nodes.
    """
    if not HAS_OCIO:
        return None

    # 1. Explicit path
    if ocio_config_path and os.path.exists(ocio_config_path):
        return OCIO.Config.CreateFromFile(ocio_config_path)

    # 2. Environment variable
    env_path = os.environ.get("OCIO", "")
    if env_path and (os.path.exists(env_path) or env_path.startswith("ocio://")):
        return OCIO.Config.CreateFromFile(env_path)

    # 3. Active process config, if a host application set one
    try:
        cfg = OCIO.GetCurrentConfig()
        if cfg is not None and not _is_unconfigured_default(cfg):
            return cfg
    except Exception:  # nosec B110
        pass

    # 4. Local ACES folder shipped alongside the Radiance package
    current_dir = os.path.dirname(os.path.realpath(__file__))
    radiance_dir = os.path.dirname(current_dir)  # hdr/ -> radiance/
    local_aces = os.path.join(radiance_dir, "ACES", "config.ocio")
    if os.path.exists(local_aces):
        return OCIO.Config.CreateFromFile(local_aces)

    # 5. ComfyUI models folder
    try:
        import folder_paths

        aces_path = os.path.join(folder_paths.models_dir, "ACES", "config.ocio")
        if os.path.exists(aces_path):
            return OCIO.Config.CreateFromFile(aces_path)
    except ImportError as _exc:
        logger.debug(
            "[Radiance] _resolve_config(): ignoring %s from `import folder_paths`: %s",
            type(_exc).__name__, _exc,
        )

    return None


def _iter_colorspaces(config):
    """
    Yield (name, family) tuples for every colorspace in the config.

    FIX: Previous code used OCIO v1 index-based getNumColorSpaces() /
    getColorSpaceNameByIndex(i) loop throughout the module.  OCIO v2 provides
    a proper iterator via getColorSpaces().  We try v2 first and fall back to
    the v1 index loop so both library generations are supported.
    """
    try:
        # OCIO v2: getColorSpaces() returns an iterator of ColorSpace objects
        for cs in config.getColorSpaces():
            yield cs.getName(), (cs.getFamily() or "")
    except (AttributeError, TypeError):
        # OCIO v1 fallback
        for i in range(config.getNumColorSpaces()):
            name = config.getColorSpaceNameByIndex(i)
            cs = config.getColorSpace(name)
            family = (cs.getFamily() if cs else "") or ""
            yield name, family


def _iter_look_names(config):
    """
    Yield every look name in the config (OCIO v2 / v1 compatible).

    FIX: Previous code used v1 index loop: getLookNameByIndex(i).
    """
    try:
        yield from config.getLookNames()  # OCIO v2
    except AttributeError:
        for i in range(config.getNumLooks()):  # OCIO v1
            yield config.getLookNameByIndex(i)


def _iter_displays_views(config):
    """
    Yield (display, view) pairs (OCIO v2 / v1 compatible).

    FIX: Previous code used integer-indexed getDisplay(i) and getView(display, v).
    OCIO v2 exposes named iterators instead.
    """
    try:
        # OCIO v2
        for display in config.getDisplays():
            for view in config.getViews(display):
                yield display, view
    except (AttributeError, TypeError):
        # OCIO v1 fallback
        for d in range(config.getNumDisplays()):
            display = config.getDisplay(d)
            for v in range(config.getNumViews(display)):
                view = config.getView(display, v)
                yield display, view


def _format_colorspaces(config, config_path: str = "") -> str:
    """
    Build a human-readable colorspace listing from a config.
    Single implementation shared by ACESConfigManager and OCIOListColorspaces.

    FIX: Previously duplicated in _list_colorspaces() and list_spaces().
    """
    spaces = []
    for name, family in _iter_colorspaces(config):
        spaces.append(f"{name} [{family}]" if family else name)

    looks = list(_iter_look_names(config))

    by_display: Dict[str, List[str]] = {}
    for display, view in _iter_displays_views(config):
        by_display.setdefault(display, []).append(view)
    displays = [f"{d}: {', '.join(v)}" for d, v in by_display.items()]

    header = ""
    if config_path:
        header = f"Config: {os.path.basename(config_path)}\n"
    header += f"Total Colorspaces: {len(spaces)}\n" + "\u2500" * 40

    result = header + "\n\u2550\u2550\u2550 COLORSPACES \u2550\u2550\u2550\n"
    result += "\n".join(spaces[:_CS_DISPLAY_LIMIT])
    if len(spaces) > _CS_DISPLAY_LIMIT:
        result += f"\n\u2026 and {len(spaces) - _CS_DISPLAY_LIMIT} more"

    if looks:
        result += "\n\n\u2550\u2550\u2550 LOOKS \u2550\u2550\u2550\n" + "\n".join(looks)

    if displays:
        result += (
            "\n\n\u2550\u2550\u2550 DISPLAYS / VIEWS \u2550\u2550\u2550\n"
            + "\n".join(displays)
        )

    return result


# ===============================================================================
#                          OCIO COLOR TRANSFORM  (new node)
# ===============================================================================


class OCIOColorTransform:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    """
    Apply an OCIO colorspace transform to an image.

    This is the core image-processing node that was previously missing from this
    module — ACESConfigManager and OCIOListColorspaces only manage the config,
    they never touched pixel data.  Without this node, log image output was
    impossible regardless of the OCIO setup.

    Workflow examples
    -----------------
    - ACEScg -> ACEScct    :  source="ACEScg", target="ACEScct"  (the defaults;
      both exist in the ACES CG config this package bundles)
    - ACEScg -> sRGB display: source="ACES - ACEScg", target="sRGB - Display"
      (the first is an alias, which resolves; see _resolve_colorspace_name)
    - Linear -> ARRI LogC4 :  source="ACEScg", target="ACES - ARRI LogC4 (EI800)",
      which needs the ACES Studio config, not the bundled CG one
    - Roundtrip test       :  forward then inverse to check reconstruction error
    - Apply a look         :  fill 'look' with e.g. "ACES 1.3 Reference Gamut Compress"

    Color pipeline
    --------------
    1. Optional pre-exposure adjustment (linear multiply 2^stops in source space)
    2. OCIO ColorSpaceTransform  source -> target  (or inverse)
    3. Look is baked in via LookTransform when specified (applied inside the CST)

    Tensor format
    -------------
    ComfyUI IMAGE is (B, H, W, C) float32.  OCIO CPU processor expects (H, W, C)
    float32 contiguous arrays.  The batch loop processes each frame independently
    to keep peak memory low and ensure correct per-frame processing.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image encoded in source_colorspace. RGB goes through OCIO; alpha is passed through unchanged."}),
                # DEFAULTS FIX: these used to default to "Linear" and
                # "ACES - ARRI LogC4 (EI800)". Neither name exists in the ACES
                # CG config this package bundles (the camera log spaces are in
                # the Studio config, not CG), so the node raised its own
                # "colorspace not in config" error at its own untouched
                # defaults. "ACEScg" and "ACEScct" are the scene_linear and
                # compositing_log roles of the bundled config, present under
                # those exact names.
                "source_colorspace": (
                    "STRING",
                    {
                        "default": "ACEScg",
                        "tooltip": (
                            "OCIO colorspace name of the input image. "
                            "Connect ACESConfigManager -> List Colorspaces to see valid names. "
                            "Aliases and roles are accepted: in the bundled ACES CG config "
                            "'ACEScg', 'ACES - ACEScg' and 'lin_ap1' all resolve. "
                            "Camera log spaces such as 'ACES - ARRI LogC4 (EI800)' need the "
                            "ACES Studio config, not the bundled CG one."
                        ),
                    },
                ),
                "target_colorspace": (
                    "STRING",
                    {
                        "default": "ACEScct",
                        "tooltip": (
                            "Desired output colorspace. Any OCIO name, alias or role in the "
                            "active config. Bundled CG config log target: 'ACEScct'. "
                            "Camera log targets ('ACES - ARRI LogC4 (EI800)', "
                            "'ACES - Sony S-Log3 SGamut3.Cine') require the ACES Studio config."
                        ),
                    },
                ),
            },
            "optional": {
                "ocio_config_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Path to .ocio config file. Leave empty to use the OCIO environment "
                            "variable or the locally installed ACES config."
                        ),
                    },
                ),
                "look": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Optional OCIO Look name to bake in, e.g. "
                            "'ACES 1.3 Reference Gamut Compress'. "
                            "Leave empty for a plain colorspace transform with no look."
                        ),
                    },
                ),
                "exposure_stops": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -10.0,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": (
                            "Pre-transform exposure in stops (multiplies by 2^stops). "
                            "Applied to the raw source values before the OCIO transform, so it is "
                            "true exposure only when the source colorspace is linear."
                        ),
                    },
                ),
                "direction": (
                    ["Forward", "Inverse"],
                    {
                        "default": "Forward",
                        "tooltip": (
                            "Forward: source -> target. "
                            "Inverse: target -> source (useful for roundtrip tests "
                            "or undoing a previously applied transform)."
                        ),
                    },
                ),
                "clamp_output": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Clamp output to [0, 1]. Disable for HDR / log outputs where "
                            "values legitimately exceed display range."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "metadata")
    FUNCTION = "apply_transform"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    DESCRIPTION = (
        "Apply an OCIO colorspace transform to an image. "
        "Supports any source/target pair in the active OCIO config, "
        "including all log formats (LogC4, S-Log3, V-Log, DaVinci Intermediate, Log3G10)."
    )

    # ── internal helpers ──────────────────────────────────────────────────────

    def _get_cpu_processor(self, config, src: str, dst: str, look: str, direction: str):
        """
        Build and return an OCIO CPU processor for the requested transform.

        When a look is specified the entire pipeline (CST + look) is expressed
        as a LookTransform so the look is baked atomically rather than applied
        as a separate pass.  Direction is applied at the processor level so
        both forward and inverse share identical calling code in apply_transform.
        """
        ocio_dir = (
            OCIO.TransformDirection.TRANSFORM_DIR_FORWARD
            if direction == "Forward"
            else OCIO.TransformDirection.TRANSFORM_DIR_INVERSE
        )

        if look and look.strip():
            # LookTransform wraps the CST + the look in one operation.
            transform = OCIO.LookTransform(
                src=src,
                dst=dst,
                looks=look.strip(),
                direction=OCIO.TransformDirection.TRANSFORM_DIR_FORWARD,
            )
        else:
            transform = OCIO.ColorSpaceTransform(
                src=src,
                dst=dst,
                direction=OCIO.TransformDirection.TRANSFORM_DIR_FORWARD,
            )

        processor = config.getProcessor(transform, ocio_dir)
        return processor.getDefaultCPUProcessor()

    def _apply_to_frame(
        self, frame: np.ndarray, cpu_proc, exposure_stops: float
    ) -> np.ndarray:
        """
        Apply pre-exposure then OCIO transform to a single (H, W, C) float32 frame.

        applyRGB is OCIO's in-place RGB processor for contiguous float32 arrays.
        Only the first 3 channels are sent to OCIO; any alpha channel is
        preserved separately and reattached after the transform.
        """
        h, w, c = frame.shape

        # Preserve alpha — OCIO only handles RGB
        if c == 4:
            alpha = frame[..., 3:4].copy()
            rgb = np.ascontiguousarray(frame[..., :3], dtype=np.float32)
        else:
            alpha = None
            rgb = np.ascontiguousarray(frame, dtype=np.float32)

        # Pre-transform exposure (linear multiply, applied in source space)
        if exposure_stops != 0.0:
            rgb *= float(2.0**exposure_stops)

        # OCIO in-place transform — applyRGB modifies the array directly
        cpu_proc.applyRGB(rgb)

        # Reattach alpha
        if alpha is not None:
            return np.concatenate([rgb, alpha], axis=-1)
        return rgb

    # ── main entry point ──────────────────────────────────────────────────────

    def apply_transform(
        self,
        image: torch.Tensor,
        source_colorspace: str,
        target_colorspace: str,
        ocio_config_path: str = "",
        look: str = "",
        exposure_stops: float = 0.0,
        direction: str = "Forward",
        clamp_output: bool = False,
    ) -> Tuple[torch.Tensor, str]:

        if not HAS_OCIO:
            raise RuntimeError(
                "[Radiance OCIO] PyOpenColorIO is required for OCIOColorTransform. "
                "Install with: pip install opencolorio"
            )

        # Resolve config
        config = _resolve_config(ocio_config_path)
        if config is None:
            raise RuntimeError(
                "[Radiance OCIO] No OCIO config found. "
                "Set the OCIO environment variable, provide ocio_config_path, "
                "or use ACESConfigManager to download the ACES 2.0 config."
            )

        config_name = (
            os.path.basename(ocio_config_path) if ocio_config_path else "active config"
        )
        logger.info(
            f"[Radiance OCIO] Transform: '{source_colorspace}' -> '{target_colorspace}' "
            f"| look='{look or 'none'}' | dir={direction} | exp={exposure_stops:+.2f} stops "
            f"| config={config_name}"
        )

        # Validate colorspace names before building the processor so the error
        # message is actionable ("use List OCIO Colorspaces") rather than a raw
        # OCIO exception with an opaque internal traceback. Resolution goes
        # through OCIO so aliases and roles are accepted, which is what the
        # node's own tooltip promises. See _resolve_colorspace_name().
        if _resolve_colorspace_name(config, source_colorspace) is None:
            raise ValueError(
                f"[Radiance OCIO] Source colorspace '{source_colorspace}' not in config. "
                f"Use 'List OCIO Colorspaces' to see available names."
            )
        if _resolve_colorspace_name(config, target_colorspace) is None:
            raise ValueError(
                f"[Radiance OCIO] Target colorspace '{target_colorspace}' not in config. "
                f"Use 'List OCIO Colorspaces' to see available names."
            )

        # Build OCIO CPU processor
        cpu_proc = self._get_cpu_processor(
            config, source_colorspace, target_colorspace, look, direction
        )

        # Process each frame in the batch
        b, h, w, c = image.shape
        out_frames: List[np.ndarray] = []

        for i in range(b):
            frame_np = tensor_to_numpy_float32(image[i])  # (H, W, C)
            frame_out = self._apply_to_frame(frame_np, cpu_proc, exposure_stops)
            out_frames.append(frame_out)

        # Stack frames and convert back to ComfyUI tensor
        out_np = np.stack(out_frames, axis=0)  # (B, H, W, C)
        out_tensor = numpy_to_tensor_float32(out_np)  # (B, H, W, C)

        if clamp_output:
            out_tensor = torch.clamp(out_tensor, 0.0, 1.0)

        import json

        metadata = json.dumps(
            {
                "node": "RadianceHDROCIOTransform",
                "config": config_name,
                "source_colorspace": source_colorspace,
                "target_colorspace": target_colorspace,
                "look": look or "",
                "direction": direction,
                "exposure_stops": exposure_stops,
                "clamp_output": clamp_output,
                "frames": b,
                "resolution": f"{w}x{h}",
            },
            indent=2,
        )

        logger.info(
            f"[Radiance OCIO] Done — {b} frame(s) {w}x{h}, clamped={clamp_output}"
        )
        return (out_tensor, metadata)


# ===============================================================================
#                          ACES CONFIG MANAGER
# ===============================================================================


class ACESConfigManager:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    """
    Detect, download, and manage ACES OCIO configurations for professional color workflows.
    """

    COMMON_PATHS = [
        # Windows
        r"C:\ACES\config.ocio",
        r"C:\Program Files\ACES\config.ocio",
        r"C:\Users\Public\ACES\config.ocio",
        # Linux / macOS
        "/opt/ACES/config.ocio",
        "/usr/share/ACES/config.ocio",
        "~/ACES/config.ocio",
    ]

    # Official ACES 2.0 config — OCIO 2.5 / v4.0.0
    ACES2_CONFIG_URL = (
        "https://github.com/AcademySoftwareFoundation/OpenColorIO-Config-ACES/"
        "releases/download/v4.0.0/cg-config-v4.0.0_aces-v2.0_ocio-v2.5.ocio"
    )

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "action": (
                    [
                        "Detect Config",
                        "Download ACES 2.0",
                        "List Colorspaces",
                        "Get Config Info",
                    ],
                    {
                        "default": "Detect Config",
                        "tooltip": "Action to perform.",
                    },
                ),
            },
            "optional": {
                "custom_config_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Custom path to .ocio config. Leave empty for auto-detection.",
                    },
                ),
                "install_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Where to install downloaded ACES config. Defaults to <radiance>/ACES/.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("config_path", "colorspaces_list", "status_info")
    FUNCTION = "manage_config"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    DESCRIPTION = "Detect, download, and manage ACES OCIO configurations."
    OUTPUT_NODE = True

    # ── internal helpers ──────────────────────────────────────────────────────

    def _find_existing_config(self) -> Tuple[str, str]:
        """Find existing ACES config on the system."""
        try:
            env_config = os.environ.get("OCIO", "")
            if env_config and os.path.exists(env_config):
                return env_config, f"Found config from OCIO environment: {env_config}"

            for path in self.COMMON_PATHS:
                expanded = os.path.expanduser(path)
                if os.path.exists(expanded):
                    return expanded, f"Found config at: {expanded}"

            try:
                import folder_paths

                aces_path = os.path.join(folder_paths.models_dir, "ACES", "config.ocio")
                if os.path.exists(aces_path):
                    return aces_path, f"Found config in ComfyUI models: {aces_path}"
            except ImportError as _exc:
                logger.debug(
                    "[Radiance] _find_existing_config(): ignoring %s from `import folder_paths`: %s",
                    type(_exc).__name__, _exc,
                )

            current_dir = os.path.dirname(os.path.realpath(__file__))
            radiance_dir = os.path.dirname(current_dir)
            local_aces = os.path.join(radiance_dir, "ACES", "config.ocio")
            if os.path.exists(local_aces):
                return local_aces, f"Found config in Radiance folder: {local_aces}"

            return "", "No ACES config found. Use 'Download ACES 2.0' to install."
        except Exception as e:
            return "", f"Error detecting config: {e}"

    def _list_colorspaces(self, config_path: str) -> str:
        """List available colorspaces — delegates to shared helper."""
        if not HAS_OCIO:
            return "PyOpenColorIO not installed. Run: pip install opencolorio"
        if not config_path or not os.path.exists(config_path):
            return "No valid config path provided."
        try:
            config = OCIO.Config.CreateFromFile(config_path)
            return _format_colorspaces(config, config_path)
        except Exception as e:
            return f"Error reading config: {e}"

    def _download_aces_config(self, install_path: str) -> Tuple[str, str]:
        """Download the official ACES 2.0 config."""
        import urllib.request

        if not install_path:
            current_dir = os.path.dirname(os.path.realpath(__file__))
            radiance_dir = os.path.dirname(current_dir)
            install_path = os.path.join(radiance_dir, "ACES")

        os.makedirs(install_path, exist_ok=True)
        config_file = os.path.join(install_path, "config.ocio")

        if os.path.exists(config_file):
            return config_file, f"ACES config already exists at: {config_file}"

        try:
            logger.info(f"Downloading ACES 2.0 config to {config_file}...")
            # FIX: Previously called urllib.request.install_opener() which installs
            # the custom opener PROCESS-WIDE, affecting every subsequent urllib call
            # made by any other code in the ComfyUI session (other nodes, extensions,
            # the server itself).  Use opener.open() locally instead so the custom
            # User-Agent header is scoped to this single download request only.
            opener = urllib.request.build_opener()
            opener.addheaders = [("User-agent", "Mozilla/5.0")]
            with opener.open(self.ACES2_CONFIG_URL) as response:
                with open(config_file, "wb") as f:
                    f.write(response.read())

            os.environ["OCIO"] = config_file
            return (
                config_file,
                f"Successfully downloaded ACES 2.0 config to: {config_file}\n"
                f"OCIO environment variable set for this session.",
            )
        except Exception as e:
            return (
                "",
                f"Download failed: {e}\n\nManual download URL:\n{self.ACES2_CONFIG_URL}",
            )

    def _get_config_info(self, config_path: str) -> str:
        """Get detailed info about the config."""
        if not HAS_OCIO:
            return "PyOpenColorIO not installed. Run: pip install opencolorio"
        if not config_path or not os.path.exists(config_path):
            return "No valid config path provided."
        try:
            config = OCIO.Config.CreateFromFile(config_path)
            # FIX: getDisplay(i) with integer index is OCIO v1 style.
            # Use _iter_displays_views() which handles both v1 and v2.
            display_names = sorted({d for d, _ in _iter_displays_views(config)})
            info = [
                f"Config Path:   {config_path}",
                f"Description:   {config.getDescription()}",
                f"Search Paths:  {config.getSearchPath()}",
                f"Colorspaces:   {sum(1 for _ in _iter_colorspaces(config))}",
                f"Looks:         {', '.join(_iter_look_names(config)) or 'none'}",
                f"Displays:      {', '.join(display_names)}",
                "",
                "Common ACES Colorspaces:",
                "  - ACES - ACEScg              (CG working space)",
                "  - ACES - ACES2065-1          (archival / interchange)",
                "  - Output - sRGB              (web / SDR display)",
                "  - Output - Rec.2100-PQ       (HDR display)",
                "  - ACES - ARRI LogC4 (EI800)  (log output)",
                "  - ACES - Sony S-Log3 SGamut3.Cine (log output)",
            ]
            return "\n".join(info)
        except Exception as e:
            return f"Error reading config: {e}"

    # ── main entry point ──────────────────────────────────────────────────────

    def manage_config(
        self,
        action: str,
        custom_config_path: str = "",
        install_path: str = "",
    ) -> Tuple[str, str, str]:

        try:
            config_path = custom_config_path or ""
            colorspaces_list = ""
            status_info = ""

            if action == "Detect Config":
                if not custom_config_path:
                    config_path, status_info = self._find_existing_config()
                else:
                    config_path = custom_config_path
                    status_info = (
                        f"Using custom config: {config_path}"
                        if os.path.exists(config_path)
                        else "Custom config not found!"
                    )
                if config_path and os.path.exists(config_path):
                    colorspaces_list = self._list_colorspaces(config_path)

            elif action == "Download ACES 2.0":
                config_path, status_info = self._download_aces_config(install_path)
                if config_path:
                    colorspaces_list = self._list_colorspaces(config_path)

            elif action == "List Colorspaces":
                if not custom_config_path:
                    config_path, _ = self._find_existing_config()
                else:
                    config_path = custom_config_path
                colorspaces_list = self._list_colorspaces(config_path)
                status_info = f"Listed colorspaces from: {config_path}"

            elif action == "Get Config Info":
                if not custom_config_path:
                    config_path, _ = self._find_existing_config()
                else:
                    config_path = custom_config_path
                status_info = self._get_config_info(config_path)

            return (config_path, colorspaces_list, status_info)

        except Exception as e:
            error_msg = f"ACESConfigManager error: {e}"
            logger.error(error_msg)
            return ("", "", error_msg)


# ===============================================================================
#                          LIST OCIO COLORSPACES
# ===============================================================================


class OCIOListColorspaces:
    """
    List all colorspaces, looks, and display/view transforms in an OCIO config.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {},
            "optional": {
                "ocio_config_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Path to .ocio config. Leave empty to use OCIO env var or local ACES config.",
                    },
                ),
                "filter_family": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Only show colorspaces whose family or name contains this string (case-insensitive).",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("colorspaces_list",)
    FUNCTION = "list_spaces"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    DESCRIPTION = "List all colorspaces available in an OCIO configuration file."
    OUTPUT_NODE = True

    def list_spaces(
        self,
        ocio_config_path: str = "",
        filter_family: str = "",
    ) -> Tuple[str]:

        if not HAS_OCIO:
            return (
                "PyOpenColorIO not installed. Install with: pip install opencolorio",
            )

        try:
            config = _resolve_config(ocio_config_path)
            if config is None:
                return (
                    "No OCIO config found. "
                    "Use ACESConfigManager to download or set the OCIO environment variable.",
                )

            # FIX: Previously duplicated the 50-cap and index-iteration logic.
            # Now delegates to the shared _format_colorspaces helper (or filtered variant).
            if filter_family:
                flt = filter_family.lower()
                spaces = [
                    (f"{n} [{f}]" if f else n)
                    for n, f in _iter_colorspaces(config)
                    if flt in f.lower() or flt in n.lower()
                ]
                looks = list(_iter_look_names(config))
                result = f"Filter: '{filter_family}' -- {len(spaces)} match(es)\n"
                result += "\n".join(spaces[:_CS_DISPLAY_LIMIT])
                if len(spaces) > _CS_DISPLAY_LIMIT:
                    result += f"\n... and {len(spaces) - _CS_DISPLAY_LIMIT} more"
                if looks:
                    result += "\n\n=== LOOKS ===\n" + "\n".join(looks)
                return (result,)

            return (_format_colorspaces(config, ocio_config_path),)

        except Exception as e:
            return (f"Error: {e}",)


# ===============================================================================
#                          NODE REGISTRATION
# ===============================================================================

NODE_CLASS_MAPPINGS = {
    # Key renamed to RadianceHDROCIOTransform to avoid collision with
    # color/transform.py which also registers RadianceOCIOColorTransform.
    # hdr/__init__.py re-exports this class under RadianceHDROCIOTransform.
    "RadianceHDROCIOTransform": OCIOColorTransform,
    "RadianceACESConfigManager": ACESConfigManager,
    "RadianceOCIOListColorspaces": OCIOListColorspaces,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceHDROCIOTransform": "◎ Radiance OCIO Transform (HDR)",
    "RadianceACESConfigManager": "◎ Radiance ACES Config Manager",
    "RadianceOCIOListColorspaces": "◎ Radiance OCIO List Colorspaces",
}
