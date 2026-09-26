import os
import logging
import hashlib
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger("radiance.ocio")

# ═══════════════════════════════════════════════════════════════════════════════
#                        OCIO AVAILABILITY
# ═══════════════════════════════════════════════════════════════════════════════

try:
    import PyOpenColorIO as OCIO

    HAS_OCIO = True
    OCIO_VERSION = OCIO.__version__ if hasattr(OCIO, "__version__") else "unknown"
    logger.info(f"[Radiance OCIO] PyOpenColorIO {OCIO_VERSION} available")
except ImportError:
    HAS_OCIO = False
    OCIO = None
    OCIO_VERSION = None
    logger.info(
        "[Radiance OCIO] PyOpenColorIO not installed — OCIO features disabled. "
        "Install with: pip install opencolorio"
    )


# ═══════════════════════════════════════════════════════════════════════════════
#                        CONFIG DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════════

# Common OCIO config locations (searched in order)
_OCIO_SEARCH_PATHS = [
    # Environment variable (highest priority, industry standard)
    lambda: os.environ.get("OCIO"),
    # The config Radiance set up automatically (ocio_setup)
    lambda: os.path.join(os.path.dirname(os.path.realpath(__file__)), "ACES", "studio-config.ocio"),
    # ACES configs in common locations
    lambda: _find_file("/usr/share/ocio", "config.ocio"),
    lambda: _find_file(os.path.expanduser("~/.config/ocio"), "config.ocio"),
    lambda: _find_file(os.path.expanduser("~/ocio"), "config.ocio"),
    # Windows common paths
    lambda: _find_file("C:/ACES", "config.ocio"),
    lambda: _find_file("C:/ocio", "config.ocio"),
    # macOS
    lambda: _find_file("/Library/Application Support/ocio", "config.ocio"),
]


def _find_file(directory: str, filename: str) -> Optional[str]:
    """Search directory tree (1 level deep) for a file."""
    if not directory or not os.path.isdir(directory):
        return None
    # Direct match
    direct = os.path.join(directory, filename)
    if os.path.isfile(direct):
        return direct
    # One level deep (e.g., ~/ocio/aces_1.2/config.ocio)
    try:
        for entry in os.listdir(directory):
            sub = os.path.join(directory, entry, filename)
            if os.path.isfile(sub):
                return sub
    except OSError as _exc:
        logger.debug(
            "[Radiance] _find_file(): ignoring %s from `for entry in os.listdir(directory):`: %s",
            type(_exc).__name__, _exc,
        )
    return None


def _download_default_config() -> Optional[str]:
    """The config used when none is found. Name kept for callers.

    It used to download the ACES CG config from GitHub at startup, without
    download consent. OpenColorIO ships the ACES studio config built in, so
    nothing is fetched: radiance.color.ocio_setup writes it to
    ACES/studio-config.ocio (or falls back to the bundled CG config).
    """
    try:
        from radiance.color.ocio_setup import configure_ocio
        state = configure_ocio()
        return state["path"] if state.get("configured") else None
    except Exception as e:  # noqa: BLE001
        logger.error(f"[Radiance OCIO] automatic config setup failed: {e}")
        return None


def discover_ocio_config() -> Optional[str]:
    """
    Auto-discover the active OCIO config file.

    Search order:
      1. $OCIO environment variable (industry standard)
      2. Common system paths (/usr/share/ocio, ~/ocio, etc.)
      3. The ACES studio config Radiance sets up automatically (no download)

    Returns:
        Absolute path to config.ocio, or None if not found.
    """
    for finder in _OCIO_SEARCH_PATHS:
        try:
            path = finder()
            if path and os.path.isfile(path):
                logger.info(f"[Radiance OCIO] Config discovered: {path}")
                return os.path.abspath(path)
        except Exception:
            continue
            
    # Fallback to auto-downloading config
    return _download_default_config()


# ═══════════════════════════════════════════════════════════════════════════════
#                        CONFIG MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class OCIOConfigManager:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Utilities"
    """
    Manages a single OCIO config and caches baked 3D LUTs.

    Thread Safety:
        Read-only access to config is safe across threads.
        LUT cache uses a simple dict (GIL-protected for single writes).
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config = None
        self.config_path = None
        self.config_name = "None"
        self._lut_cache: Dict[str, np.ndarray] = {}
        self._max_cache_entries = 32

        if not HAS_OCIO:
            return

        # Try explicit path, then auto-discover
        path = config_path or discover_ocio_config()
        if path:
            self.load_config(path)

    def load_config(self, config_path: str) -> bool:
        """Load an OCIO config file. Returns True on success."""
        if not HAS_OCIO:
            logger.warning("[Radiance OCIO] Cannot load config — PyOpenColorIO not installed")
            return False

        try:
            config_path = os.path.abspath(config_path)
            if not os.path.isfile(config_path):
                logger.error(f"[Radiance OCIO] Config not found: {config_path}")
                return False

            self.config = OCIO.Config.CreateFromFile(config_path)
            self.config_path = config_path
            self._lut_cache.clear()

            # Extract a human-readable name
            desc = self.config.getDescription() or ""
            if desc:
                # Take first line of description
                self.config_name = desc.split("\n")[0].strip()[:80]
            else:
                self.config_name = os.path.basename(os.path.dirname(config_path))

            n_cs = len(list(self.config.getColorSpaces()))
            n_disp = len(self.config.getDisplays())
            logger.info(
                f"[Radiance OCIO] Loaded: {self.config_name} "
                f"({n_cs} color spaces, {n_disp} displays) "
                f"from {config_path}"
            )
            return True

        except Exception as e:
            logger.error(f"[Radiance OCIO] Failed to load config: {e}")
            self.config = None
            self.config_path = None
            self.config_name = "None"
            return False

    @property
    def is_loaded(self) -> bool:
        return self.config is not None

    # ─────────────────────────────────────────────────────────────────────
    # Enumeration
    # ─────────────────────────────────────────────────────────────────────

    def get_scene_color_spaces(self) -> List[Dict[str, str]]:
        """Get all scene-referred color spaces (for IDT / working space selection)."""
        if not self.is_loaded:
            return []
        result = []
        for cs in self.config.getColorSpaces():
            name = cs.getName()
            family = cs.getFamily() or ""
            # OCIO v2: use ReferenceSpaceType; v1: heuristic on family
            is_scene = True
            if hasattr(cs, "getReferenceSpaceType"):
                is_scene = cs.getReferenceSpaceType() == OCIO.REFERENCE_SPACE_SCENE
            elif "display" in family.lower() or "output" in family.lower():
                is_scene = False
            if is_scene:
                result.append({"name": name, "family": family})
        return result

    def get_display_color_spaces(self) -> List[Dict[str, str]]:
        """Get all display-referred color spaces."""
        if not self.is_loaded:
            return []
        result = []
        for cs in self.config.getColorSpaces():
            name = cs.getName()
            family = cs.getFamily() or ""
            is_display = False
            if hasattr(cs, "getReferenceSpaceType"):
                is_display = cs.getReferenceSpaceType() == OCIO.REFERENCE_SPACE_DISPLAY
            elif "display" in family.lower() or "output" in family.lower():
                is_display = True
            if is_display:
                result.append({"name": name, "family": family})
        return result

    def get_displays(self) -> List[str]:
        """Get available display devices (e.g., 'sRGB', 'Rec.709', 'ACES')."""
        if not self.is_loaded:
            return []
        return list(self.config.getDisplays())

    def get_views(self, display: str) -> List[str]:
        """Get available views for a display (e.g., 'ACES 1.0 SDR-video', 'Raw')."""
        if not self.is_loaded:
            return []
        try:
            return list(self.config.getViews(display))
        except Exception as exc:
            logger.warning("[radiance_ocio] get_views: %s", exc)
            return []

    def get_display_view_pairs(self) -> List[Dict[str, str]]:
        """
        Get all display/view combinations — the primary user-facing list.
        This is what appears in the viewer's OCIO dropdown.

        Returns:
            List of {"display": "sRGB", "view": "ACES 1.0 SDR-video", "label": "sRGB / ACES 1.0 SDR-video"}
        """
        if not self.is_loaded:
            return []
        pairs = []
        for display in self.get_displays():
            for view in self.get_views(display):
                pairs.append({
                    "display": display,
                    "view": view,
                    "label": f"{display} / {view}",
                })
        return pairs

    def get_roles(self) -> Dict[str, str]:
        """Get OCIO roles (scene_linear, compositing_log, color_timing, etc.)."""
        if not self.is_loaded:
            return {}
        roles = {}
        # Standard roles
        for role_name in [
            "scene_linear", "compositing_log", "color_timing",
            "data", "reference", "rendering", "default",
            "aces_interchange", "cie_xyz_d65_interchange",
        ]:
            try:
                cs = self.config.getColorSpace(
                    self.config.getColorSpaceFromRole(role_name) if hasattr(self.config, "getColorSpaceFromRole")
                    else role_name
                )
                if cs:
                    roles[role_name] = cs.getName()
            except Exception as exc:
                logger.warning("[radiance_ocio] get_roles: %s", exc)
        
        # Heuristic for common missing roles in older configs
        if "scene_linear" not in roles:
            # Look for ACEScg or linear spaces
            for cs in self.config.getColorSpaces():
                name = cs.getName().lower()
                if "acescg" in name or ("linear" in name and "srgb" not in name):
                    roles["scene_linear"] = cs.getName()
                    break
        
        return roles

    def get_processor(self, src: str, dst: str):
        """Get an OCIO processor for a transform between two spaces/roles."""
        if not self.is_loaded:
            return None
        try:
            return self.config.getProcessor(src, dst)
        except Exception as e:
            logger.error(f"[Radiance OCIO] getProcessor failed ({src} -> {dst}): {e}")
            return None

    def get_config_info(self) -> Dict[str, Any]:
        """Full config summary for the frontend."""
        if not self.is_loaded:
            return {
                "loaded": False,
                "name": "None",
                "error": "No OCIO config loaded" if HAS_OCIO else "PyOpenColorIO not installed",
                "install_hint": "pip install opencolorio" if not HAS_OCIO else None,
            }
        return {
            "loaded": True,
            "name": self.config_name,
            "path": self.config_path,
            "ocio_version": OCIO_VERSION,
            "displays": self.get_displays(),
            "display_view_pairs": self.get_display_view_pairs(),
            "scene_color_spaces": self.get_scene_color_spaces(),
            "roles": self.get_roles(),
            "cache_entries": len(self._lut_cache),
        }

    # ─────────────────────────────────────────────────────────────────────
    # 3D LUT Baking
    # ─────────────────────────────────────────────────────────────────────

    def bake_display_view_lut(
        self,
        display: str,
        view: str,
        input_space: Optional[str] = None,
        lut_size: int = 33,
    ) -> Optional[np.ndarray]:
        """
        Bake an OCIO display/view transform into a 3D LUT.

        This is the core operation — converts an OCIO processor into a
        float32 3D LUT cube that can be uploaded directly to WebGL's
        sampler3D via the existing loadLUT() method.

        Args:
            display:     Display name (e.g., "sRGB", "Rec.709")
            view:        View name (e.g., "ACES 1.0 SDR-video", "Raw")
            input_space: Source color space (default: scene_linear role)
            lut_size:    Cube size (33 = standard, 65 = high quality)

        Returns:
            Float32Array of shape (size³ × 3) in R,G,B interleaved order,
            ready for WebGL texImage3D upload. None on failure.
        """
        if not self.is_loaded:
            return None

        # Cache key
        cache_key = self._make_cache_key(display, view, input_space, lut_size)
        if cache_key in self._lut_cache:
            logger.debug(f"[Radiance OCIO] Cache hit: {display}/{view}")
            return self._lut_cache[cache_key]

        try:
            # Resolve input space — default to scene_linear role
            if not input_space:
                input_space = self.config.getColorSpace(
                    OCIO.ROLE_SCENE_LINEAR
                ).getName()

            # Build the OCIO processor
            processor = self._build_processor(display, view, input_space)

            if processor is None:
                return None

            cpu = processor.getDefaultCPUProcessor()

            # Generate the 3D LUT lattice
            lut = self._bake_processor_to_cube(cpu, lut_size)

            # Cache with eviction
            if len(self._lut_cache) >= self._max_cache_entries:
                # Remove oldest entry
                oldest = next(iter(self._lut_cache))
                del self._lut_cache[oldest]
            self._lut_cache[cache_key] = lut

            logger.info(
                f"[Radiance OCIO] Baked {lut_size}³ LUT: "
                f"{input_space} → {display}/{view} "
                f"({lut.nbytes // 1024}KB)"
            )
            return lut

        except Exception as e:
            logger.error(f"[Radiance OCIO] Bake failed ({display}/{view}): {e}")
            return None

    def bake_colorspace_lut(
        self,
        src_space: str,
        dst_space: str,
        lut_size: int = 33,
    ) -> Optional[np.ndarray]:
        """
        Bake a direct color space → color space transform into a 3D LUT.

        Useful for IDTs (e.g., "ARRI LogC3" → "ACES - ACEScg") or
        custom transforms between any two named color spaces.

        Args:
            src_space: Source color space name
            dst_space: Destination color space name
            lut_size:  Cube size (33 or 65)

        Returns:
            Float32Array for WebGL, or None on failure.
        """
        if not self.is_loaded:
            return None

        cache_key = hashlib.md5(
            repr(("cs", src_space, dst_space, lut_size)).encode()
        ).hexdigest()

        if cache_key in self._lut_cache:
            return self._lut_cache[cache_key]

        try:
            processor = self.config.getProcessor(src_space, dst_space)
            cpu = processor.getDefaultCPUProcessor()
            lut = self._bake_processor_to_cube(cpu, lut_size)

            if len(self._lut_cache) >= self._max_cache_entries:
                oldest = next(iter(self._lut_cache))
                del self._lut_cache[oldest]
            self._lut_cache[cache_key] = lut

            logger.info(
                f"[Radiance OCIO] Baked {lut_size}³ LUT: "
                f"{src_space} → {dst_space} ({lut.nbytes // 1024}KB)"
            )
            return lut

        except Exception as e:
            logger.error(f"[Radiance OCIO] Bake failed ({src_space} → {dst_space}): {e}")
            return None

    def _build_processor(self, display: str, view: str, input_space: str):
        """Build an OCIO processor for a display/view transform."""
        try:
            # OCIO v2 API (preferred)
            if hasattr(OCIO, "DisplayViewTransform"):
                dvt = OCIO.DisplayViewTransform()
                dvt.setSrc(input_space)
                dvt.setDisplay(display)
                dvt.setView(view)
                return self.config.getProcessor(dvt)
            else:
                # OCIO v1 fallback
                return self.config.getProcessor(
                    inputColorSpace=input_space,
                    display=display,
                    view=view,
                )
        except Exception as e:
            logger.error(f"[Radiance OCIO] Processor build failed: {e}")
            return None

    def _bake_processor_to_cube(self, cpu_processor, size: int) -> np.ndarray:
        """
        Evaluate an OCIO CPU processor across a 3D lattice to produce
        a float32 RGB cube suitable for WebGL texImage3D.

        The lattice is evaluated in the correct order for OpenGL 3D
        textures: R varies fastest (innermost), then G, then B (outermost).

        Args:
            cpu_processor: OCIO CPUProcessor with applyRGB()
            size:          Lattice points per axis (33 or 65)

        Returns:
            np.ndarray of shape (size*size*size, 3), dtype float32
        """
        n = size

        # Lattice: R fastest, then G, then B, matching texImage3D(TEXTURE_3D)
        # memory layout. Built with numpy rather than a triple loop -- at 65
        # that loop was 274,625 iterations to produce a fixed ramp.
        axis = np.arange(n, dtype=np.float32) / (n - 1)
        coords = np.stack([
            np.tile(axis, n * n),                  # R cycles every element
            np.repeat(np.tile(axis, n), n),        # G every n
            np.repeat(axis, n * n),                # B every n*n
        ], axis=1).astype(np.float32)

        # Apply on the array, not pixel by pixel.
        #
        # This used to build a Python list per lattice point and call
        # applyRGB(list). OCIO's binding converts a list to a temporary buffer,
        # transforms that, and discards it -- the list is not modified. So the
        # loop wrote the input straight back and every baked LUT was an exact
        # identity: 18% grey through ACEScg -> sRGB Display came out as 0.18
        # instead of 0.47, and the viewer showed an untransformed picture under
        # the name of the transform the user had picked.
        #
        # Given a contiguous float32 array, applyRGB does modify in place, and
        # it is also ~35x faster than the loop was. The `hasattr(applyRGB)`
        # branch that used to guard a "batch API if available" fallback is gone
        # with it: applyRGB has always been present, so the fallback was
        # unreachable and the slow path was the only path.
        result = np.ascontiguousarray(coords)
        cpu_processor.applyRGB(result)
        return result.astype(np.float32)

    def _make_cache_key(
        self, display: str, view: str, input_space: Optional[str], size: int
    ) -> str:
        # repr of a tuple, not a colon-joined string: a display or view name
        # containing a colon would otherwise let two different transforms hash
        # to the same key and serve each other's LUT.
        raw = repr(("dv", display, view, input_space, size))
        return hashlib.md5(raw.encode()).hexdigest()

    def clear_cache(self):
        """Release all cached LUTs."""
        n = len(self._lut_cache)
        self._lut_cache.clear()
        if n > 0:
            logger.info(f"[Radiance OCIO] Cleared {n} cached LUTs")


# ═══════════════════════════════════════════════════════════════════════════════
#                     GLOBAL SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

# Auto-initialize on module load — the config manager is lightweight when
# no config is found (no OCIO calls until explicitly needed).
_ocio_manager: Optional[OCIOConfigManager] = None


#: Directories an OCIO config may be loaded from over the HTTP route.
#:
#: Defaults to the bundled ACES config and the ComfyUI models tree. Add more
#: with RADIANCE_OCIO_ROOTS (os.pathsep-separated), or with the standard $OCIO
#: environment variable, which a studio will already have set.
def _allowed_ocio_roots():
    roots = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "ACES")]
    for var in ("RADIANCE_OCIO_ROOTS", "OCIO"):
        value = os.environ.get(var, "")
        for part in value.split(os.pathsep):
            part = part.strip().strip('"')
            if not part:
                continue
            roots.append(part if os.path.isdir(part) else os.path.dirname(part))
    try:
        import folder_paths  # type: ignore
        _models_dir = getattr(folder_paths, "models_dir", None)
        if isinstance(_models_dir, str) and _models_dir:
            roots.append(_models_dir)
    except Exception:
        pass
    return [os.path.abspath(os.path.expanduser(r)) for r in roots if r]


def _is_inside_allowed_ocio_root(path: str) -> bool:
    resolved = os.path.realpath(path)
    for root in _allowed_ocio_roots():
        try:
            if os.path.commonpath([resolved, os.path.realpath(root)]) == os.path.realpath(root):
                return True
        except ValueError:      # different drives on Windows
            continue
    return False


def get_ocio_manager() -> OCIOConfigManager:
    """Get or create the global OCIO config manager."""
    global _ocio_manager
    if _ocio_manager is None:
        _ocio_manager = OCIOConfigManager()
    return _ocio_manager


# ═══════════════════════════════════════════════════════════════════════════════
#                     HTTP API ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

def register_ocio_routes():
    """Register OCIO HTTP endpoints on the ComfyUI PromptServer."""
    try:
        from aiohttp import web
        from server import PromptServer
    except ImportError:
        logger.warning("[Radiance OCIO] Cannot register routes — server not available")
        return

    # Idempotent: re-running this (duplicate import) would crash ComfyUI startup
    # with "method HEAD is already registered". Register the OCIO routes once.
    if getattr(PromptServer.instance, "_radiance_ocio_routes_registered", False):
        return
    PromptServer.instance._radiance_ocio_routes_registered = True

    @PromptServer.instance.routes.get("/radiance/ocio/config")
    async def ocio_config_endpoint(request):
        """Return full OCIO config info for the frontend HUD."""
        mgr = get_ocio_manager()
        return web.json_response(mgr.get_config_info())

    @PromptServer.instance.routes.post("/radiance/ocio/load")
    async def ocio_load_endpoint(request):
        """Load or reload an OCIO config from a specific path."""
        try:
            data = await request.json()
            config_path = data.get("path", "")

            if not config_path:
                return web.json_response(
                    {"error": "Missing 'path' parameter", "status": "error"}
                )

            # Containment, not just resolution.
            #
            # The comment here said "resolve and validate path" and then only
            # called abspath + isfile, so this unauthenticated route accepted an
            # arbitrary absolute path and answered "File not found" or "loaded"
            # -- a file-existence oracle for anything on the host, and an
            # invitation to hand a parser a file it was never meant to see.
            config_path = os.path.abspath(os.path.expanduser(config_path))
            if not _is_inside_allowed_ocio_root(config_path):
                logger.warning(
                    "[Radiance OCIO] Rejected a config load outside the allowed "
                    "roots: %s", config_path,
                )
                return web.json_response(
                    {"error": "Path is outside the allowed OCIO directories. "
                              "Set RADIANCE_OCIO_ROOTS to add locations.",
                     "status": "error"},
                    status=403,
                )
            if not os.path.isfile(config_path):
                return web.json_response(
                    {"error": f"File not found: {config_path}", "status": "error"},
                    status=404,
                )

            mgr = get_ocio_manager()
            success = mgr.load_config(config_path)

            if success:
                return web.json_response({
                    "status": "success",
                    "config": mgr.get_config_info(),
                })
            else:
                return web.json_response(
                    {"error": "Failed to parse OCIO config", "status": "error"}
                )
        except Exception as e:
            return web.json_response({"error": str(e), "status": "error"})

    @PromptServer.instance.routes.post("/radiance/ocio/bake")
    async def ocio_bake_endpoint(request):
        """
        Bake an OCIO display/view transform to a 3D LUT and return as binary.

        Request JSON:
            { "display": "sRGB", "view": "ACES 1.0 SDR-video", "size": 33 }
            OR
            { "src": "ARRI LogC3", "dst": "ACEScg", "size": 33 }

        Response:
            Binary float32 data (size³ × 3 floats) with Content-Type application/octet-stream.
            The JS frontend creates a Float32Array from this and calls renderer.loadLUT().
        """
        try:
            data = await request.json()
            mgr = get_ocio_manager()

            if not mgr.is_loaded:
                return web.json_response(
                    {"error": "No OCIO config loaded", "status": "error"}
                )

            lut_size = min(max(int(data.get("size", 33)), 9), 65)

            # Display/View mode
            if "display" in data and "view" in data:
                lut = mgr.bake_display_view_lut(
                    display=data["display"],
                    view=data["view"],
                    input_space=data.get("input_space"),
                    lut_size=lut_size,
                )
            # ColorSpace → ColorSpace mode
            elif "src" in data and "dst" in data:
                lut = mgr.bake_colorspace_lut(
                    src_space=data["src"],
                    dst_space=data["dst"],
                    lut_size=lut_size,
                )
            else:
                return web.json_response(
                    {
                        "error": "Provide either {display, view} or {src, dst}",
                        "status": "error",
                    }
                )

            if lut is None:
                return web.json_response(
                    {"error": "LUT baking failed — check server logs", "status": "error"}
                )

            # Return raw float32 bytes — the frontend wraps in Float32Array
            response = web.Response(
                body=lut.tobytes(),
                content_type="application/octet-stream",
                headers={
                    "X-Radiance-LUT-Size": str(lut_size),
                    "X-Radiance-LUT-Channels": "3",
                    "X-Radiance-LUT-Dtype": "float32",
                },
            )
            return response

        except Exception as e:
            logger.error(f"[Radiance OCIO] Bake endpoint error: {e}")
            return web.json_response({"error": str(e), "status": "error"})

    @PromptServer.instance.routes.get("/radiance/ocio/displays")
    async def ocio_displays_endpoint(request):
        """Quick endpoint: list display/view pairs for the HUD dropdown."""
        mgr = get_ocio_manager()
        return web.json_response({
            "loaded": mgr.is_loaded,
            "config_name": mgr.config_name,
            "pairs": mgr.get_display_view_pairs() if mgr.is_loaded else [],
        })

    logger.info("[Radiance OCIO] HTTP routes registered: /radiance/ocio/*")


# Auto-register routes on import
try:
    register_ocio_routes()
except Exception as e:
    logger.debug(f"[Radiance OCIO] Route registration deferred: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#                   NUMPY LUT APPLICATION (CPU fallback)
# ═══════════════════════════════════════════════════════════════════════════════

def apply_ocio_transform(
    img: np.ndarray,
    display: str,
    view: str,
    input_space: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    Apply an OCIO display/view transform to a numpy image (CPU path).

    This is the fallback for delivery/export when GPU is not available.
    For real-time viewing, the baked 3D LUT + WebGL path is used instead.

    Args:
        img:         float32 numpy array (H, W, 3)
        display:     OCIO display name
        view:        OCIO view name
        input_space: Source color space (default: scene_linear role)

    Returns:
        Transformed float32 numpy array, or None on failure.
    """
    mgr = get_ocio_manager()
    if not mgr.is_loaded:
        return None

    try:
        if not input_space:
            input_space = mgr.config.getColorSpace(
                OCIO.ROLE_SCENE_LINEAR
            ).getName()

        processor = mgr._build_processor(display, view, input_space)
        if processor is None:
            return None

        cpu = processor.getDefaultCPUProcessor()

        # OCIO operates in-place on contiguous float32 RGB
        out = img[..., :3].astype(np.float32).copy()
        h, w = out.shape[:2]

        # applyRGB mutates a contiguous float32 buffer in place. Handed a
        # Python list it transforms a temporary and discards it, which is how
        # this function returned its input unchanged for every pixel, the same
        # defect the LUT bake carried. Give it the array.
        if hasattr(cpu, "applyRGB"):
            flat = np.ascontiguousarray(out.reshape(-1, 3), dtype=np.float32)
            cpu.applyRGB(flat)
            out = flat.reshape(h, w, 3)

        return out

    except Exception as e:
        logger.error(f"[Radiance OCIO] CPU transform failed: {e}")
        return None
