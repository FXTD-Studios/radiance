from __future__ import annotations

import asyncio
import os
import io
import base64
import json
import time
import shutil
import traceback
import logging
import zipfile
from pathlib import Path

# aiohttp and PromptServer are only available inside a live ComfyUI process.
# Guard the import so the module can be loaded safely in test runners, linters,
# and CI environments that lack the ComfyUI server context.
try:
    from aiohttp import web
    from server import PromptServer
    _WORKSPACE_AVAILABLE = True
except ImportError:
    web = None                # type: ignore[assignment]
    PromptServer = None       # type: ignore[assignment]
    _WORKSPACE_AVAILABLE = False

import folder_paths

logger = logging.getLogger("radiance.pipeline")

# ═══════════════════════════════════════════════════════════════════════════════
#                           CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

MAX_WORKFLOW_SIZE_MB = 50
MAX_WORKFLOW_SIZE_BYTES = MAX_WORKFLOW_SIZE_MB * 1024 * 1024
MAX_FILENAME_LENGTH = 200
ALLOWED_EXTENSIONS = {".rad"}
MAX_VERSIONS = 50
MAX_RAD_ZIP_ENTRIES = 256
MAX_RAD_UNCOMPRESSED_BYTES = MAX_WORKFLOW_SIZE_BYTES


# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline path helpers
# ─────────────────────────────────────────────────────────────────────────────

def _next_version(render_root: Path) -> str:
    """
    Scan render_root for subdirectories named vNNN and return the next one.
    Example: render/ contains v001/, v002/ → returns "v003".
    Returns "v001" if the directory is empty or does not exist.
    """
    import re as _re
    max_v = 0
    if render_root.exists():
        for d in render_root.iterdir():
            if d.is_dir():
                m = _re.match(r'^v(\d+)$', d.name)
                if m:
                    max_v = max(max_v, int(m.group(1)))
    return f"v{max_v + 1:03d}"


def _build_shot_paths(pipe_root: str, show: str, sequence: str, shot: str, task: str) -> dict:
    """
    Build and materialise the standard VFX shot directory tree.

    Layout::

        {pipe_root}/{show}/sequences/{seq}/{shot}/
            plates/          <- read-only input EXRs (from editorial)
            ref/             <- reference stills, EDL, client notes
            work/{task}/     <- .rad workflow files
            render/          <- versioned render output (render/v001/, render/v002/ ...)
            deliverables/    <- QC-approved, locked outputs
            cache/           <- temp / intermediate (auto-deletable)
    """
    root = Path(pipe_root) / show / "sequences" / sequence / shot
    paths = {
        "root":         root,
        "plates":       root / "plates",
        "ref":          root / "ref",
        "work":         root / "work" / task,
        "render":       root / "render",
        "deliverables": root / "deliverables",
        "cache":        root / "cache",
    }
    for p in paths.values():
        try:
            p.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.warning("[Radiance] Could not create %s: %s", p, exc)
    return paths


# ═══════════════════════════════════════════════════════════════════════════════
#                       NODE: RadianceProjectManager
# ═══════════════════════════════════════════════════════════════════════════════

class RadianceProjectManager:
    """
    Pipeline project manager — save, list, load, delete, and inspect .rad workflow containers.
    Uses the existing backend (v2/v3 .rad packing, version control, scene inspector).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "filename": ("STRING", {
                    "default": "",
                    "placeholder": "shot_001",
                    "tooltip": "Workflow filename stem (version appended automatically).",
                }),
                "artist": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Artist name saved in workflow metadata.",
                }),
                "version": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 9999,
                    "tooltip": "Version number for the saved workflow.",
                }),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION    = "run"
    CATEGORY    = "FXTD STUDIOS/Radiance/◎ Pipeline"
    OUTPUT_NODE = True
    DESCRIPTION = "Save the current workflow graph as a .rad container with artist and version metadata."

    def run(self, filename="", artist="", version=1, prompt=None, extra_pnginfo=None):
        stem = f"{filename}_{artist or 'unknown'}_v{version:04d}" if filename else f"{artist or 'unknown'}_v{version:04d}"
        return self._save(stem, artist, version, prompt, extra_pnginfo)

    def _save(self, filename, artist, version, prompt, extra_pnginfo):
        if not prompt:
            return ()

        # Store the UI workflow (extra_pnginfo["workflow"]) when ComfyUI gives
        # it: it is what the library reopens with loadGraphData. `prompt` is the
        # API graph ({id: {class_type, inputs}}), kept only as the fallback.
        workflow = extra_pnginfo.get("workflow") if isinstance(extra_pnginfo, dict) else None
        graph_data = workflow if isinstance(workflow, dict) and workflow.get("nodes") else prompt
        graph_json = json.dumps(graph_data)

        # Build pipeline metadata via scene inspector
        tech_profile = _inspect_graph_content(graph_json)

        metadata = {
            "artist": artist or "",
            "version": version,
            "saved_at": time.time(),
            "stats": {
                "models": tech_profile.get("models", []),
                "color_spaces": tech_profile.get("color_spaces", []),
                "fps": tech_profile.get("fps", 24.0),
                "is_hdr": tech_profile.get("is_hdr", False),
                "node_count": tech_profile.get("node_count", 0),
            },
            "pipeline": tech_profile,
        }

        filepath = _resolve_safe_path(filename if filename.endswith(".rad") else filename + ".rad")
        if filepath is None:
            return ()

        filepath.parent.mkdir(parents=True, exist_ok=True)
        _create_version_backup(filepath, message=f"Save v{version}", author=artist or "unknown")

        # Use v3 container for richer metadata
        binary = _pack_rad_v3(graph_json, metadata)
        filepath.write_bytes(binary)

        count = tech_profile.get("node_count", 0)
        return ()


# ═══════════════════════════════════════════════════════════════════════════════
#                       NODE: RadiancePipelineContext
# ═══════════════════════════════════════════════════════════════════════════════


# Compatibility alias. `RadianceProjectManager` is the published key (the
# pipeline group registers it); `RadianceWorkspace` is the older name and has to
# keep loading for workflows saved against it. Subclassed rather than aliased so
# the DEPRECATED flag hides this entry from the node search without hiding the
# canonical node, which is the same class object.
class RadianceWorkspace(RadianceProjectManager):
    """Deprecated alias of RadianceProjectManager. Loads, does not list."""
    DEPRECATED = True


NODE_CLASS_MAPPINGS = {
    "RadianceWorkspace":        RadianceWorkspace,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceWorkspace":        "◎ Radiance Project Manager",
}


# ═══════════════════════════════════════════════════════════════════════════════
#                         PATH SECURITY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

# ALBABIT-FIX: the package's workflows/ folder, not one beside this file. Moving
# the module into nodes/pipeline/ had sent the library to a stray folder there,
# hiding earlier saves and the shipped workflows/official/.
WORKFLOW_DIR = str(Path(__file__).resolve().parents[2] / "workflows")
OFFICIAL_DIR = os.path.join(WORKFLOW_DIR, "official")
os.makedirs(WORKFLOW_DIR, exist_ok=True)
os.makedirs(OFFICIAL_DIR, exist_ok=True)

# Pre-resolve once at startup for consistent comparison
_WORKFLOW_DIR_RESOLVED = Path(WORKFLOW_DIR).resolve()


def _resolve_safe_path(filename: str) -> Path | None:
    """
    Resolve a user-supplied filename to a safe absolute path within WORKFLOW_DIR.
    Returns None if the resolved path escapes the allowed directory.
    Uses pathlib.is_relative_to() which is immune to the os.sep prefix trick.
    """
    if not filename or len(filename) > MAX_FILENAME_LENGTH:
        return None

    # Reject null bytes and other control characters
    if any(ord(c) < 32 for c in filename):
        return None

    candidate = (_WORKFLOW_DIR_RESOLVED / filename).resolve()

    try:
        candidate.relative_to(_WORKFLOW_DIR_RESOLVED)
    except ValueError:
        return None

    return candidate


def _validate_extension(filepath: Path) -> bool:
    """Ensure the file has an allowed extension."""
    return filepath.suffix.lower() in ALLOWED_EXTENSIONS


# ═══════════════════════════════════════════════════════════════════════════════
#                         SCENE INSPECTOR (VFX PIPELINE)
# ═══════════════════════════════════════════════════════════════════════════════

def _inspect_graph_content(graph_json: str) -> dict:
    """
    Scan the serialised ComfyUI graph to extract VFX pipeline metadata.

    Extracts:
      - models:       .safetensors / .ckpt filenames from any loader node
      - color_spaces: colorspace strings from OCIO / Color / HDR nodes
      - is_hdr:       True if any HDR node is present
      - fps:          first float widget value from any fps-bearing node
      - node_count:   total number of nodes in the graph
    """
    profile = {
        "models":       set(),
        "color_spaces": set(),
        "fps":          24.0,
        "is_hdr":       False,
        "node_count":   0,
    }

    try:
        data  = json.loads(graph_json)
        nodes = data.get("nodes")
        if nodes is None:
            # API format: {id: {"class_type": ..., "inputs": {...}}}. Read it as
            # nodes whose widget values are the literal inputs (lists are links).
            nodes = [
                {"type": n.get("class_type", ""),
                 "widgets_values": [v for v in (n.get("inputs") or {}).values()
                                    if not isinstance(v, (list, dict))]}
                for n in data.values()
                if isinstance(n, dict) and "class_type" in n
            ]
        profile["node_count"] = len(nodes)

        for node in nodes:
            ctype  = node.get("type", "")
            inputs = node.get("widgets_values", [])

            # Models — any loader or checkpoint node
            if any(k in ctype for k in ("Loader", "Checkpoint", "Lora", "LoRA")):
                for val in inputs:
                    if isinstance(val, str) and val.endswith((".safetensors", ".ckpt", ".pt")):
                        profile["models"].add(val)

            # Color spaces — OCIO, Color, HDR nodes
            if any(k in ctype for k in ("OCIO", "Color", "HDR", "Colorspace", "Grade")):
                for val in inputs:
                    if isinstance(val, str) and any(
                        x in val.lower()
                        for x in ("aces", "rec709", "rec2020", "linear", "srgb", "logc", "slog", "vlog", "davinci")
                    ):
                        profile["color_spaces"].add(val)
                if "HDR" in ctype:
                    profile["is_hdr"] = True

            # FPS — grab first numeric widget from timing nodes
            if any(k in ctype for k in ("FPS", "Timeline", "Video", "Sequence", "ProjectManager")):
                for val in inputs:
                    if isinstance(val, float) and 1.0 <= val <= 240.0:
                        profile["fps"] = val
                        break

    except Exception as exc:
        logger.warning("[Radiance] Scene Inspector failed: %s", exc)

    return {
        "models":       sorted(profile["models"]),
        "color_spaces": sorted(profile["color_spaces"]),
        "fps":          profile["fps"],
        "is_hdr":       profile["is_hdr"],
        "node_count":   profile["node_count"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
#                         VERSION CONTROL BACKEND (v3.5)
# ═══════════════════════════════════════════════════════════════════════════════

def _create_version_backup(filepath: Path, message: str = "Auto-save", author: str = "Radiance Artist") -> None:
    """
    Rotate existing file into a .vN backup before overwrite.
    Keeps at most MAX_VERSIONS backups, deleting the oldest.
    Also writes a .vN.json sidecar with commit metadata (message, author, timestamp).
    """
    if not filepath.exists():
        return

    versions_dir = filepath.parent / ".versions"
    versions_dir.mkdir(exist_ok=True)

    stem = filepath.stem
    suffix = filepath.suffix

    # FIX 4: Find the actual highest version number instead of using len(existing)+1.
    # len(existing)+1 gives wrong numbers when backups have been deleted — e.g. if
    # v1..v5 exist and v3 is deleted, len=4 → next_num=5, which collides with v5.
    # Parse the numeric suffix from each filename and take max+1.
    import re as _re

    def _ver_of(path) -> int:
        m = _re.search(r"\.v(\d+)" + _re.escape(suffix) + r"$", path.name)
        return int(m.group(1)) if m else 0

    # Sort by version NUMBER, not by name. Lexicographic order puts v10 before
    # v2, so once the counter passed ten the eviction below deleted a newer
    # backup and kept an older one.
    existing = sorted(versions_dir.glob(f"{stem}.v*{suffix}"), key=_ver_of)
    max_ver = max((_ver_of(p) for p in existing), default=0)
    next_num = max_ver + 1

    # Evict oldest if at capacity
    if len(existing) >= MAX_VERSIONS:
        for old in existing[: len(existing) - MAX_VERSIONS + 1]:
            old.unlink(missing_ok=True)
            old.with_suffix(old.suffix + ".json").unlink(missing_ok=True)

    backup_name = f"{stem}.v{next_num}{suffix}"
    backup_path = versions_dir / backup_name
    shutil.copy2(str(filepath), str(backup_path))
    
    # Save commit metadata sidecar
    meta_path = backup_path.with_suffix(backup_path.suffix + ".json")
    commit_info = {
        "version": next_num,
        "message": message,
        "author": author,
        "timestamp": time.time(),
        "original_filename": filepath.name
    }
    meta_path.write_text(json.dumps(commit_info, indent=2), encoding="utf-8")
    
    logger.info(f"[Radiance] Commit: {message} ({backup_name})")


# ═══════════════════════════════════════════════════════════════════════════════
#                       ASSETS LIBRARY API ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════


import hashlib
import zlib
import struct
import re

# ═══════════════════════════════════════════════════════════════════════════════
#                         SECURE .RAD CONTAINER (v2)
# ═══════════════════════════════════════════════════════════════════════════════

RAD_MAGIC    = b"RAD!"
RAD_MAGIC_V3 = b"RADZ"           # v3 ZIP container magic
RAD_VERSION  = 2
# FIX 6: Format string uses unsigned short H (was signed h) for the version field.
# Signed short has no meaningful advantage here and could cause confusion if version
# numbers ever exceed 32767. H is the correct type for a version/flag field.
_RAD_HEADER_FMT  = ">4sHII"   # magic(4s) ver(H) meta_len(I) graph_len(I) = 14 bytes
_RAD_HEADER_SIZE = struct.calcsize(_RAD_HEADER_FMT)   # always 14


# ───────────────────────────────────────────────────────────────────────────────
#               .rad v3 — ZIP-based archive (Phase 5.1)
# Layout inside ZIP:
#   manifest.json     — metadata + asset list + rad_version=3
#   workflow.json     — raw ComfyUI graph JSON
#   assets/<name>     — optional linked assets (images, LUTs, etc.)
# ───────────────────────────────────────────────────────────────────────────────

def _pack_rad_v3(
    graph_json: str,
    metadata: dict,
    assets: dict | None = None,
) -> bytes:
    """
    Pack a workflow + metadata + optional assets into a .rad v3 ZIP container.

    Args:
        graph_json: The ComfyUI graph JSON string.
        metadata:   Arbitrary metadata dict (author, description, timestamps).
        assets:     Optional dict of {asset_name: bytes} to embed in assets/.

    Returns:
        Raw bytes of the ZIP-based .rad v3 container.

    Format:
        A standard Python zipfile.ZIP_DEFLATED archive with:
          manifest.json  — {rad_version:3, metadata:{...}, assets:[name,...]}
          workflow.json  — raw graph JSON
          assets/<name>  — one entry per asset in `assets` dict
    """
    manifest = {
        "rad_version": 3,
        "metadata": metadata,
        "assets": list((assets or {}).keys()),
        "packed_at": time.time(),
        "stats": metadata.get("stats", {}),  # Promote stats to top level for fast access
    }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.writestr("manifest.json",  json.dumps(manifest, indent=2))
        zf.writestr("workflow.json",  graph_json)
        for name, data in (assets or {}).items():
            # Sanitise asset name: no path traversal
            safe_name = Path(name).name
            zf.writestr(f"assets/{safe_name}", data)
        
        # New in v3.5: Store a high-res preview image if provided
        # Key presence is not a preview: save_workflow always sets this key,
        # so an empty string wrote a zero-byte preview.png and every listing
        # then advertised has_preview for a file with no preview.
        if metadata.get("preview_image"):
            try:
                # Expecting base64 string
                p_data = metadata["preview_image"]
                if p_data.startswith("data:image"):
                    p_data = p_data.split(",")[1]
                zf.writestr("preview.png", base64.b64decode(p_data))
            except Exception as e:
                logger.warning(f"[Radiance] Failed to pack preview image: {e}")

    raw = buf.getvalue()
    # Prefix 4-byte magic so we can detect v3 by header without parsing the ZIP
    return RAD_MAGIC_V3 + raw


def _unpack_rad_v3(data: bytes) -> tuple[str, dict, dict]:
    """
    Unpack a .rad v3 ZIP container.

    Returns:
        (graph_json, metadata, assets_dict)
        assets_dict = {asset_name: bytes}
    """
    if not data.startswith(RAD_MAGIC_V3):
        raise ValueError("Not a .rad v3 container (missing RADZ magic)")

    zip_data = data[4:]  # Strip the 4-byte magic prefix
    buf = io.BytesIO(zip_data)

    try:
        zf = zipfile.ZipFile(buf, mode="r")
    except zipfile.BadZipFile as e:
        raise ValueError(f"Corrupt .rad v3 ZIP: {e}")

    with zf:
        infos = zf.infolist()
        if len(infos) > MAX_RAD_ZIP_ENTRIES:
            raise ValueError(".rad v3: too many ZIP entries")

        total_uncompressed = sum(info.file_size for info in infos)
        if total_uncompressed > MAX_RAD_UNCOMPRESSED_BYTES:
            raise ValueError(
                f".rad v3: uncompressed payload exceeds {MAX_WORKFLOW_SIZE_MB}MB limit"
            )

        names = [info.filename for info in infos]

        if "workflow.json" not in names:
            raise ValueError(".rad v3: missing workflow.json")
        graph_json = zf.read("workflow.json").decode("utf-8")

        metadata = {}
        if "manifest.json" in names:
            manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
            metadata = manifest.get("metadata", {})
            metadata["rad_version"] = 3
            metadata["assets"] = manifest.get("assets", [])

        assets = {}
        for n in names:
            if n.startswith("assets/") and not n.endswith("/"):
                asset_name = n[len("assets/"):]
                if Path(asset_name).name != asset_name or any(ord(c) < 32 for c in asset_name):
                    raise ValueError(".rad v3: unsafe asset name")
                assets[asset_name] = zf.read(n)

    return graph_json, metadata, assets


def _detect_rad_version(data: bytes) -> int:
    """
    Detect the .rad format version from the first 4 bytes.
    Returns 1, 2, or 3.
    """
    if data[:4] == RAD_MAGIC_V3:
        return 3
    if data[:4] == RAD_MAGIC:
        return 2
    return 1  # Legacy plain-text


def _unpack_any_rad(data: bytes) -> tuple[str, dict]:
    """
    Unified unpacker: automatically handles v1, v2, and v3 .rad files.
    Returns (graph_json, metadata). Assets are discarded in v3 (use _unpack_rad_v3 directly).
    """
    ver = _detect_rad_version(data)
    if ver == 3:
        graph_json, metadata, _ = _unpack_rad_v3(data)
        return graph_json, metadata
    elif ver == 2:
        return _unpack_rad_v2(data)
    else:
        return data.decode("utf-8"), {"format": "v1", "secure": False}


def _pack_rad_v2(graph_json: str, metadata: dict) -> bytes:
    """
    Packs a workflow and metadata into a secure binary container.
    Format: [MAGIC][VER][META_LEN][GRAPH_LEN][META_JSON][Z_GRAPH_DATA][SHA256]
    """
    meta_data  = json.dumps(metadata).encode("utf-8")
    graph_data = zlib.compress(graph_json.encode("utf-8"))

    # FIX 6: use _RAD_HEADER_FMT (unsigned H) instead of inline ">4shII"
    header  = struct.pack(_RAD_HEADER_FMT, RAD_MAGIC, RAD_VERSION, len(meta_data), len(graph_data))
    payload = header + meta_data + graph_data

    checksum = hashlib.sha256(payload).digest()
    return payload + checksum

def _unpack_rad_v2(data: bytes) -> tuple[str, dict]:
    """
    Unpacks a v2 binary .rad file. Verifies integrity.
    """
    min_size = _RAD_HEADER_SIZE + 32   # header + SHA256
    if len(data) < min_size:
        raise ValueError("File too small to be a valid .rad v2 container")

    payload           = data[:-32]
    expected_checksum = data[-32:]
    actual_checksum   = hashlib.sha256(payload).digest()

    if actual_checksum != expected_checksum:
        raise ValueError(
            "File integrity check failed (SHA256 mismatch). "
            "The file may be corrupted or tampered with."
        )

    # FIX 6: unpack with unsigned H
    magic, ver, meta_len, graph_len = struct.unpack(_RAD_HEADER_FMT, payload[:_RAD_HEADER_SIZE])
    if magic != RAD_MAGIC:
        raise ValueError("Invalid .rad container magic bytes")

    meta_start = _RAD_HEADER_SIZE
    meta_end   = meta_start + meta_len
    meta_json  = payload[meta_start:meta_end].decode("utf-8")
    metadata   = json.loads(meta_json)

    graph_start = meta_end
    # FIX 5: use graph_len from the header to slice exactly — previously
    # payload[graph_start:] was used which silently accepted any trailing bytes.
    graph_data  = zlib.decompress(payload[graph_start:graph_start + graph_len])
    return graph_data.decode("utf-8"), metadata


# ── HTTP Route registration helper ─────────────────────────────────────────────
# Returns a no-op decorator when the ComfyUI server is not available (e.g., test
# runners, CI, linters). This lets all route handlers be defined normally without
# conditional indentation throughout the file.

def _route(method: str, path: str):
    """Decorator: register an aiohttp route only when PromptServer is present."""
    if not _WORKSPACE_AVAILABLE:
        return lambda f: f
    # Idempotent: the module can be imported under two names (radiance.X and X),
    # which would otherwise register every route twice and crash ComfyUI startup
    # with "method HEAD is already registered". Track (method, path) on the
    # PromptServer singleton so a duplicate import re-uses the first registration.
    _reg = getattr(PromptServer.instance, "_radiance_registered_routes", None)
    if _reg is None:
        _reg = set()
        setattr(PromptServer.instance, "_radiance_registered_routes", _reg)
    _key = (method.lower(), path)
    if _key in _reg:
        return lambda f: f
    _reg.add(_key)
    return getattr(PromptServer.instance.routes, method)(path)


def _format_bytes(size: int | float) -> str:
    """Return a compact file size label for dashboard API payloads."""
    units = ("B", "KB", "MB", "GB", "TB")
    value = float(max(size, 0))
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return "0 B"


def _format_relative_time(timestamp: float) -> str:
    """Return an artist-friendly relative timestamp."""
    delta = max(0, time.time() - timestamp)
    if delta < 60:
        return "Just now"
    if delta < 3600:
        minutes = int(delta // 60)
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    if delta < 86400:
        hours = int(delta // 3600)
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    if delta < 604800:
        days = int(delta // 86400)
        return f"{days} day{'s' if days != 1 else ''} ago"
    return time.strftime("%Y-%m-%d", time.localtime(timestamp))


def _project_slug(value: str) -> str:
    """Stable project id used by the Project Manager API."""
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "general"


def _shot_from_name(value: str) -> str:
    """Extract a VFX-style shot code from a filename or metadata string."""
    # Not \b: "_" is a word character, so "sh010_v002" (the Project Manager's
    # own naming) did not match. Bounded by anything but a letter or digit.
    match = re.search(r"(?<![a-z0-9])sh[-_ ]?(\d{2,5})(?![a-z0-9])", value, flags=re.IGNORECASE)
    if match:
        return f"SH{match.group(1)}"
    return "GENERAL"


def _version_from_name(value: str) -> str:
    """Extract a vNNN/vNNNN version token from a filename."""
    match = re.search(r"(?<![a-z0-9])v(\d{3,4})(?![a-z0-9])", value, flags=re.IGNORECASE)
    if match:
        return f"v{match.group(1)}"
    return "v001"


def _read_workflow_records() -> list[dict]:
    """Read the existing workflow library and expose records for API routes."""
    workflows = []

    for rad_file in _WORKFLOW_DIR_RESOLVED.rglob("*.rad"):
        if ".versions" in rad_file.parts:
            continue

        try:
            rel_path = rad_file.relative_to(_WORKFLOW_DIR_RESOLVED)
        except ValueError:
            continue

        stat = rad_file.stat()
        meta = {}

        try:
            with open(rad_file, "rb") as f:
                # A v1 container is plain-text JSON of any length, so a short
                # read is not an error here. Only the binary v2 branch below
                # needs the full header, and it checks for itself.
                header_sample = f.read(_RAD_HEADER_SIZE)

            if header_sample.startswith(RAD_MAGIC_V3):
                full_data = rad_file.read_bytes()
                _, meta_v3, assets_v3 = _unpack_rad_v3(full_data)
                meta = meta_v3
                meta["format"] = "v3"
                meta["secure"] = True
                meta["asset_count"] = len(assets_v3)

                with zipfile.ZipFile(rad_file, "r") as zf:
                    meta["has_preview"] = "preview.png" in zf.namelist()
            elif (header_sample.startswith(RAD_MAGIC)
                  and len(header_sample) >= _RAD_HEADER_SIZE):
                magic, ver, meta_len, graph_len = struct.unpack(_RAD_HEADER_FMT, header_sample)
                with open(rad_file, "rb") as f:
                    f.seek(_RAD_HEADER_SIZE)
                    meta_json = f.read(meta_len).decode("utf-8")
                meta = json.loads(meta_json)
                meta["format"] = "v2"
                meta["secure"] = True
            else:
                meta_path = rad_file.with_suffix(rad_file.suffix + ".json")
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                meta["format"] = "v1"
                meta["secure"] = False
        except Exception as meta_err:
            logger.warning("[Radiance] Failed to read metadata for %s: %s", rel_path, meta_err)

        filename = str(rel_path).replace("\\", "/")
        workflows.append({
            "filename": filename,
            "path": rad_file,
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            "metadata": meta,
        })

    workflows.sort(key=lambda x: x["mtime"], reverse=True)
    return workflows


def _project_name_for_workflow(workflow: dict) -> str:
    metadata = workflow.get("metadata") or {}
    explicit = metadata.get("project") or metadata.get("show")
    if explicit:
        return str(explicit)

    parts = workflow["filename"].split("/")
    if len(parts) > 1:
        return parts[0]
    return "GENERAL"


def _build_project_index(workflows: list[dict]) -> dict[str, dict]:
    projects: dict[str, dict] = {}

    for workflow in workflows:
        project_name = _project_name_for_workflow(workflow)
        project_id = _project_slug(project_name)
        shot = _shot_from_name(workflow["filename"])
        project = projects.setdefault(project_id, {
            "id": project_id,
            "name": project_name,
            "shots_set": set(),
            "workflow_count": 0,
            "size": 0,
            "mtime": 0,
            "workflows": [],
        })
        project["shots_set"].add(shot)
        project["workflow_count"] += 1
        project["size"] += workflow["size"]
        project["mtime"] = max(project["mtime"], workflow["mtime"])
        project["workflows"].append(workflow)

    return projects


def _project_summary(project: dict) -> dict:
    return {
        "id": project["id"],
        "name": project["name"],
        "shots": len(project["shots_set"]),
        "workflows": project["workflow_count"],
        "updated": _format_relative_time(project["mtime"]),
        "favorite": False,
        "size": _format_bytes(project["size"]),
    }


def _find_project(project_id: str) -> tuple[dict | None, list[dict]]:
    workflows = _read_workflow_records()
    projects = _build_project_index(workflows)
    return projects.get(project_id), workflows


def _shot_status_path(project: dict) -> Path:
    """Where a project's shot statuses live: <library>/<project name>/.

    The name comes from metadata inside any .rad in the library, shared ones
    included, so "../../x" would have put the file outside it. Raises
    ValueError for a name that does not stay inside the library.
    """
    root = _WORKFLOW_DIR_RESOLVED
    # Lexical, so only leaving the library is refused; a name that stays
    # inside keeps the location it always had.
    path = Path(os.path.normpath(root / str(project["name"]) / ".shot_status.json"))
    try:
        path.relative_to(root)
    except ValueError:
        raise ValueError(f"project name {project['name']!r} leaves the workflow library") from None
    return path


def _load_shot_status(project: dict) -> dict:
    try:
        sp = _shot_status_path(project)
        if sp.exists():
            data = json.loads(sp.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception as _exc:
        logger.debug(
            "[Radiance] _load_shot_status(): ignoring %s from `sp = _shot_status_path(project)`: %s",
            type(_exc).__name__, _exc,
        )
    return {}


def _save_shot_status(project: dict, mapping: dict) -> None:
    try:
        sp = _shot_status_path(project)
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("[Radiance] Failed to save shot status: %s", exc)


def _versions_for_project(project: dict) -> list[dict]:
    versions = []
    status_map = _load_shot_status(project)

    for workflow in sorted(project["workflows"], key=lambda w: w["mtime"], reverse=True):
        metadata = workflow.get("metadata") or {}
        name = Path(workflow["filename"]).stem
        versions.append({
            "version": str(metadata.get("version") or _version_from_name(name)),
            "status": str(status_map.get(str(metadata.get("shot") or _shot_from_name(name))) or metadata.get("status") or metadata.get("review_status") or "WIP"),
            "shot": str(metadata.get("shot") or _shot_from_name(name)),
            "workflow": name,
            "filename": workflow["filename"],
            "updated": _format_relative_time(workflow["mtime"]),
        })

        versions_dir = workflow["path"].parent / ".versions"
        if versions_dir.exists():
            for backup in sorted(versions_dir.glob(f"{workflow['path'].stem}.v*.rad"), key=lambda p: p.stat().st_mtime, reverse=True)[:5]:
                meta_file = backup.with_suffix(backup.suffix + ".json")
                backup_meta = {}
                if meta_file.exists():
                    try:
                        backup_meta = json.loads(meta_file.read_text(encoding="utf-8"))
                    except Exception:
                        backup_meta = {}
                versions.append({
                    "version": _version_from_name(backup.name),
                    "status": str(backup_meta.get("message") or "Backup"),
                    "shot": str(metadata.get("shot") or _shot_from_name(name)),
                    "workflow": name,
                    "filename": workflow["filename"],
                    "updated": _format_relative_time(backup.stat().st_mtime),
                })

    return versions[:12]


def _outputs_for_project(project: dict) -> list[dict]:
    output_dir_getter = getattr(folder_paths, "get_output_directory", None)
    if not output_dir_getter:
        return []

    try:
        output_root = Path(output_dir_getter()).resolve()
    except Exception:
        return []

    if not output_root.exists():
        return []

    project_tokens = {project["name"].lower(), project["id"].replace("-", "_").lower()}
    shot_tokens = {shot.lower() for shot in project["shots_set"] if shot != "GENERAL"}
    allowed = {".exr", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".mov", ".mp4"}
    outputs = []

    try:
        for item in output_root.rglob("*"):
            if not item.is_file() or item.suffix.lower() not in allowed:
                continue
            haystack = item.name.lower()
            if project_tokens and any(token in haystack for token in project_tokens) or any(token in haystack for token in shot_tokens):
                stat = item.stat()
                outputs.append({
                    "name": item.name,
                    "type": item.suffix.lstrip(".").upper(),
                    "size": _format_bytes(stat.st_size),
                    "date": _format_relative_time(stat.st_mtime),
                    "path": str(item),
                    "_mtime": stat.st_mtime,
                })
    except Exception as exc:
        logger.warning("[Radiance] Failed to scan outputs for project %s: %s", project["name"], exc)

    outputs.sort(key=lambda item: item.get("_mtime", 0), reverse=True)
    for item in outputs:
        item.pop("_mtime", None)
    return outputs[:20]


def _notes_for_project(project: dict) -> list[str]:
    notes_path = _WORKFLOW_DIR_RESOLVED / project["name"] / ".review_notes.json"
    if notes_path.exists():
        try:
            data = json.loads(notes_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [str(item) for item in data]
            if isinstance(data, dict) and isinstance(data.get("notes"), list):
                return [str(item) for item in data["notes"]]
        except Exception as exc:
            logger.warning("[Radiance] Failed to read notes for project %s: %s", project["name"], exc)

    return []


def _storage_summary() -> dict:
    try:
        usage = shutil.disk_usage(_WORKFLOW_DIR_RESOLVED)
        used = usage.total - usage.free
        percent = int((used / usage.total) * 100) if usage.total else 0
        return {
            "usedLabel": _format_bytes(used),
            "totalLabel": _format_bytes(usage.total),
            "percent": percent,
        }
    except Exception:
        total = sum(item["size"] for item in _read_workflow_records())
        return {
            "usedLabel": _format_bytes(total),
            "totalLabel": "Workflow Library",
            "percent": min(100, total // (1024 * 1024)),
        }


def _dashboard_payload() -> dict:
    workflows = _read_workflow_records()
    project_index = _build_project_index(workflows)
    project_items = sorted(project_index.values(), key=lambda project: project["mtime"], reverse=True)
    projects = [_project_summary(project) for project in project_items]

    active_project = None
    if projects:
        active_project = project_index.get(projects[0]["id"])

    latest = workflows[0] if workflows else None
    latest_name = Path(latest["filename"]).stem if latest else "No workflow saved yet"
    latest_project = _project_name_for_workflow(latest) if latest else "GENERAL"
    latest_metadata = latest.get("metadata", {}) if latest else {}

    versions = _versions_for_project(active_project) if active_project else []
    outputs = _outputs_for_project(active_project) if active_project else []
    notes = _notes_for_project(active_project) if active_project else []

    return {
        "user": {
            "name": "Ahmed",
            "role": "Artist",
        },
        "storage": _storage_summary(),
        "continueWorking": {
            "workflow": latest_name,
            "project": latest_project,
            "shot": str(latest_metadata.get("shot") or _shot_from_name(latest_name)),
            "status": str(latest_metadata.get("status") or latest_metadata.get("review_status") or "WIP"),
            "lastModified": _format_relative_time(latest["mtime"]) if latest else "Never",
            "filename": latest["filename"] if latest else "",
        },
        "projects": projects,
        "versions": versions,
        "outputs": outputs,
        "notes": notes,
        "quickActions": [
            "Save Version",
            "Compare Versions",
            "Import Workflow",
            "Export Package",
            "Project Settings",
        ],
        "nav": [
            "Dashboard",
            "Projects",
            "Shots",
            "Workflows",
            "Versions",
            "Review",
            "Assets",
            "Settings",
        ],
        "source": "Live API",
        "activeProjectId": active_project["id"] if active_project else "",
    }


@_route("post", "/radiance/workflows/pack")
async def pack_workflow(request):
    """
    Utility endpoint that returns a binary .rad blob for the provided graph.
    Used by the JS side for 'Export' without actually saving to the server's library.
    """
    try:
        data = await request.json()
        content = data.get("content", "")
        description = data.get("description", "")
        
        if not content:
            return web.json_response({"error": "Missing content"}, status=400)
            
        metadata = {
            "description": description,
            "saved_at": time.time(),
            "format": "v2",
            "author": data.get("author", "Radiance Artist")
        }
        
        binary_rad = _pack_rad_v2(content, metadata)
        
        return web.Response(body=binary_rad, content_type='application/octet-stream')
    except Exception as e:
        logger.exception("[Radiance] pack_workflow failed")
        return web.json_response({"error": str(e)}, status=500)

@_route("post", "/radiance/workflows/unpack")
async def unpack_workflow_api(request):
    """
    Utility endpoint that takes a binary .rad blob and returns the unpacked 
    graph JSON and metadata. Handles v1, v2, and v3.
    """
    try:
        content_length = request.content_length
        if content_length is not None and content_length > MAX_WORKFLOW_SIZE_BYTES:
            return web.json_response({"error": f"Payload exceeds {MAX_WORKFLOW_SIZE_MB}MB limit"}, status=413)

        body = await request.content.read(MAX_WORKFLOW_SIZE_BYTES + 1)
        if len(body) > MAX_WORKFLOW_SIZE_BYTES:
            return web.json_response({"error": f"Payload exceeds {MAX_WORKFLOW_SIZE_MB}MB limit"}, status=413)

        ver  = _detect_rad_version(body)

        if ver == 3:
            graph_json, metadata, assets = _unpack_rad_v3(body)
            return web.json_response({
                "success":  True,
                "content":  graph_json,
                "metadata": metadata,
                "secure":   True,
                "format":   "v3",
                "assets":   list(assets.keys()),
            })
        elif ver == 2:
            graph_json, metadata = _unpack_rad_v2(body)
            return web.json_response({
                "success":  True,
                "content":  graph_json,
                "metadata": metadata,
                "secure":   True,
                "format":   "v2",
            })
        else:
            content = body.decode("utf-8")
            return web.json_response({
                "success": True,
                "content": content,
                "secure":  False,
                "format":  "v1",
            })
    except Exception as e:
        return web.json_response({"error": str(e)}, status=400)

@_route("post", "/radiance/workflows/save")
async def save_workflow(request):
    try:
        # FIX 2: Content-Length is an optional HTTP header that can be absent or
        # spoofed. The pre-read check below is a fast-path guard only — it cannot
        # be relied upon for enforcement. A post-read check (after reading the body)
        # is the only reliable protection against oversized payloads.
        content_length = request.content_length
        if content_length is not None and content_length > MAX_WORKFLOW_SIZE_BYTES:
            return web.json_response({"error": f"Payload exceeds {MAX_WORKFLOW_SIZE_MB}MB limit"}, status=413)

        data = await request.json()

        # Post-read size enforcement — serialise the body back and check actual length.
        raw_size = len(json.dumps(data).encode("utf-8"))
        if raw_size > MAX_WORKFLOW_SIZE_BYTES:
            return web.json_response({"error": f"Payload exceeds {MAX_WORKFLOW_SIZE_MB}MB limit"}, status=413)
        filename = data.get("filename", "").strip()
        description = data.get("description", "").strip()
        content = data.get("content", "")

        if not filename or not content:
            return web.json_response({"error": "Missing filename or content"}, status=400)

        if not filename.endswith(".rad"):
            filename += ".rad"

        filepath = _resolve_safe_path(filename)
        if filepath is None:
            return web.json_response({"error": "Invalid or unsafe path"}, status=403)

        # Prepare modern container metadata
        commit_message = data.get("message", "Auto-save").strip() or "Auto-save"
        author = data.get("author", "Radiance Artist").strip() or "Radiance Artist"

        # Industry-Level: Inspect Graph for Pipeline Metadata
        technical_profile = _inspect_graph_content(content)
        
        metadata = {
            "description": description,
            "author": author,
            "commit_message": commit_message,
            "stats": data.get("stats", {}),
            "preview_image": data.get("preview_image", ""),
            "pipeline": technical_profile # The Scene Inspector results
        }
        
        # Use v3 ZIP container if preview or stats are present, else v2 for speed
        if metadata["preview_image"] or metadata["stats"]:
            binary_rad = _pack_rad_v3(content, metadata)
            filename_final = filename
        else:
            binary_rad = _pack_rad_v2(content, metadata)
            filename_final = filename
        
        # Auto-version existing file before overwrite
        filepath.parent.mkdir(parents=True, exist_ok=True)
        _create_version_backup(filepath, message=commit_message, author=author)

        filepath.write_bytes(binary_rad)

        # Clean up legacy sidecar if it exists
        meta_path = filepath.with_suffix(filepath.suffix + ".json")
        meta_path.unlink(missing_ok=True)

        logger.info(f"[Radiance] Saved secure .rad workflow: {filename}")
        return web.json_response({"success": True, "filename": filename})

    except Exception as e:
        logger.exception("[Radiance] save_workflow failed")
        return web.json_response({"error": str(e)}, status=500)


@_route("get", "/radiance/workflows/list")
async def list_workflows(request):
    try:
        workflows = [
            {
                "filename": item["filename"],
                "size": item["size"],
                "mtime": item["mtime"],
                "metadata": item["metadata"],
            }
            for item in _read_workflow_records()
        ]
        return web.json_response({"workflows": workflows})

    except Exception as e:
        logger.exception("[Radiance] list_workflows failed")
        return web.json_response({"error": str(e)}, status=500)


@_route("get", "/radiance/projects/dashboard")
async def project_manager_dashboard(request):
    """Build the dashboard payload OFF the event loop.

    `_dashboard_payload` walks the entire ComfyUI output tree with a stat() per
    entry and fully unzips every .rad it finds. That ran inline on aiohttp's
    loop, so a studio output/ directory with a few hundred thousand frames froze
    ComfyUI's websocket for the whole walk: progress bar, node highlighting,
    queue view and /prompt all stalled mid-render, with nothing to indicate the
    dashboard was the cause.
    """
    try:
        loop = asyncio.get_running_loop()
        payload = await loop.run_in_executor(None, _dashboard_payload)
        return web.json_response(payload)
    except Exception as e:
        logger.exception("[Radiance] project_manager_dashboard failed")
        return web.json_response({"error": str(e)}, status=500)


@_route("get", "/radiance/projects")
@_route("get", "/radiance/projects/recent")
async def list_projects(request):
    try:
        workflows = _read_workflow_records()
        project_items = sorted(
            _build_project_index(workflows).values(),
            key=lambda project: project["mtime"],
            reverse=True,
        )
        projects = [
            _project_summary(project)
            for project in project_items
        ]
        return web.json_response({"projects": projects})
    except Exception as e:
        logger.exception("[Radiance] list_projects failed")
        return web.json_response({"error": str(e)}, status=500)


@_route("get", "/radiance/projects/{project_id}/versions")
async def list_project_versions(request):
    try:
        project_id = request.match_info.get("project_id", "")
        project, _ = _find_project(project_id)
        if project is None:
            return web.json_response({"error": "Project not found"}, status=404)
        return web.json_response({"versions": _versions_for_project(project)})
    except Exception as e:
        logger.exception("[Radiance] list_project_versions failed")
        return web.json_response({"error": str(e)}, status=500)


@_route("get", "/radiance/projects/{project_id}/outputs")
async def list_project_outputs(request):
    try:
        project_id = request.match_info.get("project_id", "")
        project, _ = _find_project(project_id)
        if project is None:
            return web.json_response({"error": "Project not found"}, status=404)
        return web.json_response({"outputs": _outputs_for_project(project)})
    except Exception as e:
        logger.exception("[Radiance] list_project_outputs failed")
        return web.json_response({"error": str(e)}, status=500)


@_route("get", "/radiance/projects/{project_id}/notes")
async def list_project_notes(request):
    try:
        project_id = request.match_info.get("project_id", "")
        project, _ = _find_project(project_id)
        if project is None:
            return web.json_response({"error": "Project not found"}, status=404)
        return web.json_response({"notes": _notes_for_project(project)})
    except Exception as e:
        logger.exception("[Radiance] list_project_notes failed")
        return web.json_response({"error": str(e)}, status=500)


@_route("post", "/radiance/projects/{project_id}/save-version")
async def save_project_version(request):
    try:
        project_id = request.match_info.get("project_id", "")
        project, _ = _find_project(project_id)
        if project is None:
            return web.json_response({"error": "Project not found"}, status=404)

        try:
            data = await request.json()
        except Exception:
            data = {}

        content = data.get("content")
        message = str(data.get("message") or "Dashboard save version")
        author = str(data.get("author") or "Radiance Artist")

        if content:
            filename = data.get("filename", "").strip()
            if not filename:
                filename = f"{project['name']}/dashboard_{int(time.time())}.rad"
            else:
                if not filename.endswith(".rad"):
                    filename += ".rad"
                # Keep workflow categorized within the current project
                parts = filename.replace("\\", "/").split("/")
                if len(parts) == 1 or parts[0] != project['name']:
                    filename = f"{project['name']}/{parts[-1]}"

            filepath = _resolve_safe_path(filename)
            if filepath is None or not _validate_extension(filepath):
                return web.json_response({"error": "Invalid or unsafe path"}, status=403)

            filepath.parent.mkdir(parents=True, exist_ok=True)
            _create_version_backup(filepath, message=message, author=author)
            metadata = {
                "project": project["name"],
                "author": author,
                "commit_message": message,
                "saved_at": time.time(),
                "pipeline": _inspect_graph_content(content),
            }
            filepath.write_bytes(_pack_rad_v3(content, metadata))
            
            # Clean up legacy sidecar if it exists
            meta_path = filepath.with_suffix(filepath.suffix + ".json")
            meta_path.unlink(missing_ok=True)

            return web.json_response({"success": True, "filename": filename})

        latest = max(project["workflows"], key=lambda workflow: workflow["mtime"], default=None)
        if latest is None:
            return web.json_response({"error": "Project has no workflows to version"}, status=404)

        _create_version_backup(latest["path"], message=message, author=author)
        return web.json_response({
            "success": True,
            "filename": latest["filename"],
            "message": "Created a version backup for the latest workflow.",
        })
    except Exception as e:
        logger.exception("[Radiance] save_project_version failed")
        return web.json_response({"error": str(e)}, status=500)


@_route("post", "/radiance/projects/{project_id}/export-package")
async def export_project_package(request):
    try:
        project_id = request.match_info.get("project_id", "")
        project, _ = _find_project(project_id)
        if project is None:
            return web.json_response({"error": "Project not found"}, status=404)

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            manifest = {
                "project": project["name"],
                "project_id": project["id"],
                "exported_at": time.time(),
                "workflow_count": project["workflow_count"],
            }
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))
            for workflow in project["workflows"]:
                zf.write(workflow["path"], arcname=f"workflows/{workflow['filename']}")

        filename = f"{project['id']}_radiance_package.zip"
        return web.Response(
            body=buf.getvalue(),
            content_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except Exception as e:
        logger.exception("[Radiance] export_project_package failed")
        return web.json_response({"error": str(e)}, status=500)


@_route("get", "/radiance/workflows/get")
@_route("post", "/radiance/workflows/get")
async def get_workflow(request):
    try:

        filename = request.rel_url.query.get("filename", "").strip()
        filepath = _resolve_safe_path(filename)

        # FIX 3: _validate_extension() was defined but never called here.
        # Without it, any file type inside WORKFLOW_DIR could be read by name.
        if filepath is None or not _validate_extension(filepath) or not filepath.exists():
            return web.json_response({"error": "File not found"}, status=404)

        raw_data = filepath.read_bytes()
        ver = _detect_rad_version(raw_data)

        # Unified v1 / v2 / v3 dispatch
        graph_json, metadata = _unpack_any_rad(raw_data)
        return web.json_response({
            "success":  True,
            "content":  graph_json,
            "metadata": metadata,
            "secure":   ver >= 2,
            "format":   f"v{ver}",
        })

    except Exception as e:
        logger.exception("[Radiance] get_workflow failed")
        return web.json_response({"error": str(e)}, status=500)


@_route("post", "/radiance/workflows/delete")
async def delete_workflow(request):
    try:
        data = await request.json()
        filename = data.get("filename", "").strip()
        if not filename:
            return web.json_response({"error": "Missing filename"}, status=400)

        filepath = _resolve_safe_path(filename)
        # FIX 3: validate extension — _validate_extension was never called in delete.
        if filepath is None or not _validate_extension(filepath):
            return web.json_response({"error": "Invalid or unsafe path"}, status=403)

        if not filepath.exists():
            return web.json_response({"error": "File not found"}, status=404)

        filepath.unlink()
        logger.info(f"[Radiance] Deleted workflow: {filename}")

        # Also remove sidecar if exists
        meta_path = filepath.with_suffix(filepath.suffix + ".json")
        meta_path.unlink(missing_ok=True)

        # Also clean up all backups and sidecars in .versions
        versions_dir = filepath.parent / ".versions"
        if versions_dir.exists():
            for backup in versions_dir.glob(f"{filepath.stem}.v*{filepath.suffix}"):
                backup.unlink(missing_ok=True)
                backup.with_suffix(backup.suffix + ".json").unlink(missing_ok=True)
            
            # Clean up the .versions directory if it is now empty
            try:
                if not any(versions_dir.iterdir()):
                    versions_dir.rmdir()
            except OSError as _exc:
                logger.debug(
                    "[Radiance] delete_workflow(): ignoring %s from `if not any(versions_dir.iterdir()):`: %s",
                    type(_exc).__name__, _exc,
                )

        # Clean up empty parent directories (but never WORKFLOW_DIR itself)
        try:
            parent = filepath.parent
            while parent != _WORKFLOW_DIR_RESOLVED:
                if not any(parent.iterdir()):
                    parent.rmdir()
                    parent = parent.parent
                else:
                    break
        except OSError as _exc:
            logger.debug(
                "[Radiance] delete_workflow(): ignoring %s from `parent = filepath.parent`: %s",
                type(_exc).__name__, _exc,
            )

        return web.json_response({"success": True})

    except json.JSONDecodeError:
        return web.json_response({"error": "Malformed JSON body"}, status=400)
    except Exception as e:
        logger.exception("[Radiance] delete_workflow failed")
        return web.json_response({"error": str(e)}, status=500)


@_route("get", "/radiance/workflows/history")
async def get_workflow_history(request):
    """
    Returns the version history for a specific .rad file.
    """
    try:
        filename = request.rel_url.query.get("filename", "").strip()
        filepath = _resolve_safe_path(filename)
        if filepath is None or not filepath.exists():
            return web.json_response({"error": "File not found"}, status=404)

        versions_dir = filepath.parent / ".versions"
        history = []

        if versions_dir.exists():
            # Find all .vN.rad files
            for v_file in versions_dir.glob(f"{filepath.stem}.v*.rad"):
                meta_file = v_file.with_suffix(v_file.suffix + ".json")
                v_info = {
                    "version_file": v_file.name,
                    "timestamp": v_file.stat().st_mtime,
                    "size": v_file.stat().st_size,
                }
                
                if meta_file.exists():
                    try:
                        v_info.update(json.loads(meta_file.read_text(encoding="utf-8")))
                    except Exception as _exc:
                        logger.debug(
                            "[Radiance] get_workflow_history(): ignoring %s from `v_info.update(json.loads(meta_file.read_text(encoding='utf-8…`: %s",
                            type(_exc).__name__, _exc,
                        )
                
                history.append(v_info)

        history.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        return web.json_response({"success": True, "filename": filename, "history": history})

    except Exception as e:
        logger.exception("[Radiance] get_workflow_history failed")
        return web.json_response({"error": str(e)}, status=500)


@_route("post", "/radiance/workflows/restore")
async def restore_workflow_version(request):
    """
    Restores a specific version from the .versions folder.
    """
    try:
        data = await request.json()
        filename = data.get("filename", "").strip()
        version_file = data.get("version_file", "").strip()

        if not filename or not version_file:
            return web.json_response({"error": "Missing filename or version_file"}, status=400)

        target_path = _resolve_safe_path(filename)
        if target_path is None:
            return web.json_response({"error": "Invalid target path"}, status=403)

        # Security: resolve and contain version_file within the .versions directory.
        # A raw path join without resolve() allows traversal via e.g. "../../etc/passwd".
        versions_dir = (target_path.parent / ".versions").resolve()
        source_path = (versions_dir / version_file).resolve()
        try:
            source_path.relative_to(versions_dir)
        except ValueError:
            return web.json_response({"error": "Invalid version file path"}, status=403)
        if not source_path.exists() or not source_path.is_file():
            return web.json_response({"error": "Version file not found"}, status=404)

        # Before restoring, back up current state as a 'Restore Point' commit
        _create_version_backup(target_path, message=f"Pre-restore backup of {version_file}")

        # Perform restore
        shutil.copy2(str(source_path), str(target_path))
        
        logger.info(f"[Radiance] Restored {filename} from {version_file}")
        return web.json_response({"success": True, "restored_from": version_file})

    except Exception as e:
        logger.exception("[Radiance] restore_workflow_version failed")
        return web.json_response({"error": str(e)}, status=500)


@_route("get", "/radiance/workflows/preview")
async def get_workflow_preview(request):
    """
    Returns the binary preview.png from a .rad v3 file.
    """
    try:
        filename = request.rel_url.query.get("filename", "").strip()
        filepath = _resolve_safe_path(filename)
        
        if filepath is None or not filepath.exists():
            return web.json_response({"error": "File not found"}, status=404)

        with open(filepath, "rb") as _pf:
            _magic = _pf.read(4)
        if _detect_rad_version(_magic) != 3:
            return web.json_response({"error": "Previews only supported in .rad v3"}, status=400)

        with zipfile.ZipFile(filepath, "r") as zf:
            if "preview.png" not in zf.namelist():
                return web.json_response({"error": "No preview image"}, status=404)
            
            preview_data = zf.read("preview.png")
            return web.Response(body=preview_data, content_type='image/png')

    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


# ═══════════════════════════════════════════════════════════════════════════════
#                           ASSET MANAGER  (v3.1.1)
# ═══════════════════════════════════════════════════════════════════════════════
import re as _re
import hashlib as _hashlib

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp", ".gif"}
_HDR_EXTS = {".exr", ".hdr"}
_VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}
_WEB_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
_ASSET_EXTS = _IMAGE_EXTS | _HDR_EXTS | _VIDEO_EXTS
_ASSETS_BINS_PATH = _WORKFLOW_DIR_RESOLVED / "_assets_bins.json"
_ASSET_SCAN_LIMIT = 600
_SEQ_RE = _re.compile(r"^(.*?)(\d{2,})(\.[^.]+)$")


def _asset_roots() -> "list[Path]":
    roots = []
    for getter in ("get_input_directory", "get_output_directory"):
        fn = getattr(folder_paths, getter, None)
        if fn:
            try:
                p = Path(fn()).resolve()
                if p.exists():
                    roots.append(p)
            except Exception as _exc:
                logger.debug(
                    "[Radiance] _asset_roots(): ignoring %s from `p = Path(fn()).resolve()`: %s",
                    type(_exc).__name__, _exc,
                )
    return roots


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except Exception:
        return False


def _asset_id(path: Path) -> str:
    return _hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:16]


def _classify(ext: str) -> str:
    e = ext.lower()
    if e in _VIDEO_EXTS:
        return "video"
    if e == ".hdr":
        return "hdri"
    return "image"


def _load_bins() -> "list[dict]":
    try:
        if _ASSETS_BINS_PATH.exists():
            data = json.loads(_ASSETS_BINS_PATH.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
    except Exception as _exc:
        logger.debug(
            "[Radiance] _load_bins(): ignoring %s from `if _ASSETS_BINS_PATH.exists():`: %s",
            type(_exc).__name__, _exc,
        )
    return []


def _save_bins(bins: "list[dict]") -> None:
    try:
        _ASSETS_BINS_PATH.write_text(json.dumps(bins, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("[Radiance] Failed to save asset bins: %s", exc)


def _scan_assets() -> "list[dict]":
    roots = _asset_roots()
    files = []
    for root in roots:
        try:
            for item in root.rglob("*"):
                if not item.is_file() or item.suffix.lower() not in _ASSET_EXTS:
                    continue
                files.append((root, item))
                if len(files) > _ASSET_SCAN_LIMIT * 4:
                    break
        except Exception as exc:
            logger.warning("[Radiance] asset scan failed under %s: %s", root, exc)

    seq_groups: "dict[tuple, list]" = {}
    singles = []
    for root, item in files:
        m = _SEQ_RE.match(item.name)
        if m and item.suffix.lower() in (_IMAGE_EXTS | _HDR_EXTS):
            key = (str(item.parent), m.group(1), m.group(3).lower())
            seq_groups.setdefault(key, []).append((root, item, int(m.group(2))))
        else:
            singles.append((root, item))

    assets = []
    for key, members in seq_groups.items():
        if len(members) >= 3:
            members.sort(key=lambda t: t[2])
            _, first_item, first_n = members[0]
            _, _last_item, last_n = members[-1]
            total = 0
            mtime = 0.0
            for _r, it, _n in members:
                try:
                    st = it.stat(); total += st.st_size; mtime = max(mtime, st.st_mtime)
                except Exception as _exc:
                    logger.debug(
                        "[Radiance] _scan_assets(): ignoring %s from `st = it.stat()`: %s",
                        type(_exc).__name__, _exc,
                    )
            ext = key[2]
            prefix = (key[1].rstrip("._-") or first_item.stem)
            assets.append({
                "id": _asset_id(first_item.parent / (key[1] + "####" + ext)),
                "name": Path(prefix).name,
                "type": "sequence",
                "format": ext.lstrip(".").upper(),
                "frames": len(members),
                "frame_start": first_n,
                "frame_end": last_n,
                "meta": "%s seq · %d fr" % (ext.lstrip(".").upper(), len(members)),
                "size": _format_bytes(total),
                "date": _format_relative_time(mtime),
                "path": str(first_item),
                "previewable": False,
                "_mtime": mtime,
            })
        else:
            for mem in members:
                singles.append((mem[0], mem[1]))

    for root, item in singles:
        try:
            st = item.stat()
        except Exception:
            continue
        ext = item.suffix.lower()
        assets.append({
            "id": _asset_id(item),
            "name": item.name,
            "type": _classify(ext),
            "format": ext.lstrip(".").upper(),
            "meta": ext.lstrip(".").upper(),
            "size": _format_bytes(st.st_size),
            "date": _format_relative_time(st.st_mtime),
            "path": str(item),
            "previewable": ext in _WEB_IMAGE_EXTS,
            "_mtime": st.st_mtime,
        })

    assets.sort(key=lambda a: a.get("_mtime", 0), reverse=True)
    return assets[:_ASSET_SCAN_LIMIT]


def _assets_payload() -> dict:
    assets = _scan_assets()
    by_id = {a["id"] for a in assets}
    bins = _load_bins()
    for b in bins:
        b["count"] = sum(1 for aid in b.get("assets", []) if aid in by_id)
    counts = {
        "all": len(assets),
        "image": sum(1 for a in assets if a["type"] in ("image", "hdri")),
        "video": sum(1 for a in assets if a["type"] == "video"),
        "sequence": sum(1 for a in assets if a["type"] == "sequence"),
    }
    for a in assets:
        a.pop("_mtime", None)
    return {"assets": assets, "bins": bins, "counts": counts, "source": "Live scan"}


@_route("get", "/radiance/assets")
async def list_assets(request):
    try:
        return web.json_response(_assets_payload())
    except Exception as e:
        logger.exception("[Radiance] list_assets failed")
        return web.json_response({"error": str(e)}, status=500)


@_route("post", "/radiance/assets/bins")
async def create_asset_bin(request):
    try:
        data = await request.json()
    except Exception:
        data = {}
    name = str(data.get("name") or "").strip()
    if not name:
        return web.json_response({"error": "Bin name required"}, status=400)
    bins = _load_bins()
    bin_id = _hashlib.sha1((name + str(time.time())).encode()).hexdigest()[:10]
    new_bin = {"id": bin_id, "name": name[:60], "assets": []}
    bins.append(new_bin)
    _save_bins(bins)
    return web.json_response({"success": True, "bin": new_bin})


@_route("post", "/radiance/assets/bins/{bin_id}")
async def modify_asset_bin(request):
    bin_id = request.match_info.get("bin_id", "")
    try:
        data = await request.json()
    except Exception:
        data = {}
    action = str(data.get("action") or "")
    asset_id = str(data.get("asset_id") or "")
    bins = _load_bins()
    target = next((b for b in bins if b.get("id") == bin_id), None)
    if target is None:
        return web.json_response({"error": "Bin not found"}, status=404)
    if action == "add" and asset_id:
        if asset_id not in target.setdefault("assets", []):
            target["assets"].append(asset_id)
    elif action == "remove" and asset_id:
        target["assets"] = [a for a in target.get("assets", []) if a != asset_id]
    elif action == "rename":
        target["name"] = str(data.get("name") or target["name"])[:60]
    elif action == "delete":
        bins = [b for b in bins if b.get("id") != bin_id]
    else:
        return web.json_response({"error": "Unknown action"}, status=400)
    _save_bins(bins)
    return web.json_response({"success": True})


_THUMB_CACHE = _WORKFLOW_DIR_RESOLVED / ".asset_thumbs"


def _render_thumb_png(target: Path) -> "bytes | None":
    """Best-effort: render a tonemapped first-frame PNG for EXR/HDR/TIFF/video.
    Returns None if the optional cv2 dependency is missing or decode fails."""
    ext = target.suffix.lower()
    try:
        import numpy as np
        import cv2  # type: ignore
    except Exception:
        return None
    try:
        if ext in (".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"):
            cap = cv2.VideoCapture(str(target))
            ok, img = cap.read()
            cap.release()
            if not ok or img is None:
                return None
        else:
            img = cv2.imread(str(target), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
            if img is None:
                return None
            if np.issubdtype(img.dtype, np.floating):
                x = np.clip(img.astype("float32"), 0, None)
                x = x / (1.0 + x)                      # Reinhard tonemap
                x = np.power(np.clip(x, 0, 1), 1.0 / 2.2)  # display gamma
                img = (x * 255.0).astype("uint8")
            elif img.dtype == np.uint16:
                img = (img / 257.0).astype("uint8")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        h, w = img.shape[:2]
        if w > 512:
            img = cv2.resize(img, (512, max(1, int(h * 512 / w))), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".png", img)
        return buf.tobytes() if ok else None
    except Exception:
        return None


@_route("get", "/radiance/assets/thumb")
async def asset_thumb(request):
    raw = request.query.get("path", "")
    if not raw:
        return web.json_response({"error": "path required"}, status=400)
    if "\x00" in raw:
        return web.json_response({"error": "bad path"}, status=400)
    try:
        target = Path(raw).resolve()
    except Exception:
        return web.json_response({"error": "bad path"}, status=400)
    if not any(_is_within(target, root) for root in _asset_roots()) or not target.is_file():
        return web.json_response({"error": "forbidden"}, status=403)

    ext = target.suffix.lower()
    if ext in _WEB_IMAGE_EXTS:
        ctype = {
            ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".webp": "image/webp", ".gif": "image/gif", ".bmp": "image/bmp",
        }.get(ext, "application/octet-stream")
        return web.Response(body=target.read_bytes(), content_type=ctype)

    # EXR / HDR / TIFF / video -> rendered + cached PNG thumbnail (best-effort).
    try:
        mtime = target.stat().st_mtime
        key = _hashlib.sha1(("%s:%s" % (target, mtime)).encode()).hexdigest()[:20]
        _THUMB_CACHE.mkdir(parents=True, exist_ok=True)
        cached = _THUMB_CACHE / (key + ".png")
        if cached.exists():
            return web.Response(body=cached.read_bytes(), content_type="image/png")
        png = _render_thumb_png(target)
        if png:
            try:
                cached.write_bytes(png)
            except Exception as _exc:
                logger.debug(
                    "[Radiance] asset_thumb(): ignoring %s from `cached.write_bytes(png)`: %s",
                    type(_exc).__name__, _exc,
                )
            return web.Response(body=png, content_type="image/png")
    except Exception as e:
        logger.debug("[Radiance] thumb render failed: %s", e)
    return web.json_response({"error": "not previewable"}, status=415)


def _claim_upload_path(dest_dir: Path, name: str):
    """Create a new, empty file for ``name`` in dest_dir and return (path, handle).

    Never replaces a file: a taken name becomes stem_1.ext, stem_2.ext, ...
    Exclusive creation ("xb") also holds against a concurrent upload.
    """
    stem, suffix = Path(name).stem, Path(name).suffix
    for n in range(10000):
        candidate = dest_dir / (name if n == 0 else f"{stem}_{n}{suffix}")
        try:
            return candidate, open(candidate, "xb")
        except FileExistsError:
            continue
    raise FileExistsError(f"No free name for {name} in {dest_dir}")


@_route("post", "/radiance/assets/upload")
async def upload_asset(request):
    try:
        fn = getattr(folder_paths, "get_input_directory", None)
        if not fn:
            return web.json_response({"error": "No input directory"}, status=500)
        dest_dir = Path(fn()).resolve() / "radiance_assets"
        dest_dir.mkdir(parents=True, exist_ok=True)
        reader = await request.multipart()
        saved = []
        renamed = {}
        async for part in reader:
            if part.filename:
                safe_name = os.path.basename(part.filename)
                if Path(safe_name).suffix.lower() not in _ASSET_EXTS:
                    continue
                # Opening with "wb" replaced an existing file of the same name,
                # so a workflow that referenced it silently read the new one.
                dest, fh = _claim_upload_path(dest_dir, safe_name)
                try:
                    with fh:
                        while True:
                            chunk = await part.read_chunk()
                            if not chunk:
                                break
                            fh.write(chunk)
                except BaseException:
                    # An aborted upload must not leave a truncated file behind.
                    dest.unlink(missing_ok=True)
                    raise
                saved.append(dest.name)
                if dest.name != safe_name:
                    renamed[safe_name] = dest.name
        result = {"success": True, "saved": saved}
        if renamed:
            result["renamed"] = renamed
        return web.json_response(result)
    except Exception as e:
        logger.exception("[Radiance] upload_asset failed")
        return web.json_response({"error": str(e)}, status=500)


@_route("post", "/radiance/projects/{project_id}/shots/{shot}/status")
async def set_shot_status(request):
    try:
        project_id = request.match_info.get("project_id", "")
        shot = request.match_info.get("shot", "")
        project, _ = _find_project(project_id)
        if project is None:
            return web.json_response({"error": "Project not found"}, status=404)
        try:
            data = await request.json()
        except Exception:
            data = {}
        status = str(data.get("status") or "").strip()
        allowed = {"WIP", "Review", "Approved", "Retake", "Final"}
        if status not in allowed:
            return web.json_response({"error": "status must be one of %s" % sorted(allowed)}, status=400)
        try:
            _shot_status_path(project)
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        mapping = _load_shot_status(project)
        if shot:
            mapping[shot] = status
        _save_shot_status(project, mapping)
        return web.json_response({"success": True, "shot": shot, "status": status})
    except Exception as e:
        logger.exception("[Radiance] set_shot_status failed")
        return web.json_response({"error": str(e)}, status=500)
