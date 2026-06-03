import os
import io
import base64
import json
import time
import shutil
import traceback
import logging
from pathlib import Path
from aiohttp import web
from server import PromptServer

import folder_paths

logger = logging.getLogger("Radiance.Workspace")

# ═══════════════════════════════════════════════════════════════════════════════
#                           CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

MAX_WORKFLOW_SIZE_MB = 50
MAX_WORKFLOW_SIZE_BYTES = MAX_WORKFLOW_SIZE_MB * 1024 * 1024
MAX_FILENAME_LENGTH = 200
ALLOWED_EXTENSIONS = {".rad"}
MAX_VERSIONS = 10  # Auto-versioning backup depth


class RadianceWorkspace:
    """
    A unified node that sits on the ComfyUI canvas acting as a hub for
    saving and loading entire workflows securely as .rad files.

    The actual file generation and graph rebuilding is handled natively
    by the companion JavaScript extension: js/radiance_workspace.js
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "noop"
    CATEGORY = "FXTD Studios/Radiance/Layout"
    OUTPUT_NODE = False

    def noop(self, **kwargs):
        """
        No execution needed. This node exists purely to provide the
        UI buttons and anchor point on the canvas.
        """
        return ()


NODE_CLASS_MAPPINGS = {
    "RadianceWorkspace": RadianceWorkspace
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceWorkspace": "◎ Radiance Workspace"
}


# ═══════════════════════════════════════════════════════════════════════════════
#                         PATH SECURITY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

WORKFLOW_DIR = os.path.join(os.path.dirname(__file__), "workflows")
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
#                         AUTO-VERSIONING
# ═══════════════════════════════════════════════════════════════════════════════

def _create_version_backup(filepath: Path) -> None:
    """
    Rotate existing file into a .v1, .v2, ... backup before overwrite.
    Keeps at most MAX_VERSIONS backups, deleting the oldest.
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
    existing = sorted(versions_dir.glob(f"{stem}.v*{suffix}"))
    max_ver = 0
    for p in existing:
        m = _re.search(r"\.v(\d+)" + _re.escape(suffix) + r"$", p.name)
        if m:
            max_ver = max(max_ver, int(m.group(1)))
    next_num = max_ver + 1

    # Evict oldest if at capacity
    if len(existing) >= MAX_VERSIONS:
        for old in existing[: len(existing) - MAX_VERSIONS + 1]:
            old.unlink(missing_ok=True)

    backup_name = f"{stem}.v{next_num}{suffix}"
    shutil.copy2(str(filepath), str(versions_dir / backup_name))
    logger.info(f"[Radiance] Backed up {filepath.name} → .versions/{backup_name}")


# ═══════════════════════════════════════════════════════════════════════════════
#                       ASSETS LIBRARY API ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════


import hashlib
import zlib
import struct

# ═══════════════════════════════════════════════════════════════════════════════
#                         SECURE .RAD CONTAINER (v2)
# ═══════════════════════════════════════════════════════════════════════════════

RAD_MAGIC = b"RAD!"
RAD_VERSION = 2
# FIX 6: Format string uses unsigned short H (was signed h) for the version field.
# Signed short has no meaningful advantage here and could cause confusion if version
# numbers ever exceed 32767. H is the correct type for a version/flag field.
_RAD_HEADER_FMT = ">4sHII"   # magic(4s) ver(H) meta_len(I) graph_len(I) = 14 bytes
_RAD_HEADER_SIZE = struct.calcsize(_RAD_HEADER_FMT)   # always 14

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

@PromptServer.instance.routes.post("/radiance/workflows/pack")
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

@PromptServer.instance.routes.post("/radiance/workflows/unpack")
async def unpack_workflow_api(request):
    """
    Utility endpoint that takes a binary .rad blob and returns the unpacked 
    graph JSON and metadata. Used for importing local files securely.
    """
    try:
        # Read the raw binary body
        body = await request.read()
        
        if body.startswith(RAD_MAGIC):
            graph_json, metadata = _unpack_rad_v2(body)
            return web.json_response({
                "success": True, 
                "content": graph_json, 
                "metadata": metadata,
                "secure": True
            })
        else:
            # Fallback for v1 (text)
            content = body.decode("utf-8")
            return web.json_response({
                "success": True, 
                "content": content,
                "secure": False
            })
    except Exception as e:
        return web.json_response({"error": str(e)}, status=400)

@PromptServer.instance.routes.post("/radiance/workflows/save")
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

        # Prepare modern v2 container
        metadata = {
            "description": description,
            "saved_at": time.time(),
            "format": "v2",
            "author": data.get("author", "Radiance Artist")
        }
        
        binary_rad = _pack_rad_v2(content, metadata)
        
        # Auto-version existing file before overwrite
        filepath.parent.mkdir(parents=True, exist_ok=True)
        _create_version_backup(filepath)

        filepath.write_bytes(binary_rad)

        # Clean up legacy sidecar if it exists
        meta_path = filepath.with_suffix(filepath.suffix + ".json")
        meta_path.unlink(missing_ok=True)

        logger.info(f"[Radiance] Saved secure .rad workflow: {filename}")
        return web.json_response({"success": True, "filename": filename})

    except Exception as e:
        logger.exception("[Radiance] save_workflow failed")
        return web.json_response({"error": str(e)}, status=500)


@PromptServer.instance.routes.get("/radiance/workflows/list")
async def list_workflows(request):
    try:
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
                # FIX 1: Previously read only 1024 bytes, but meta_len can exceed
                # 1024 - _RAD_HEADER_SIZE = 1010 bytes. When it does, the metadata
                # JSON is silently truncated and json.loads raises JSONDecodeError,
                # caught by the outer except — metadata is lost without any warning.
                # Fix: read the header first (14 bytes) to get meta_len, then read
                # exactly that many additional bytes.
                with open(rad_file, "rb") as f:
                    header_bytes = f.read(_RAD_HEADER_SIZE)
                    if len(header_bytes) < _RAD_HEADER_SIZE:
                        raise ValueError("File too small to be a valid .rad container")
                    header_sample = header_bytes  # kept for startswith check below

                if header_sample.startswith(RAD_MAGIC):
                    # v2 Binary: read exact metadata length from header
                    magic, ver, meta_len, graph_len = struct.unpack(_RAD_HEADER_FMT, header_sample)
                    with open(rad_file, "rb") as f:
                        f.seek(_RAD_HEADER_SIZE)
                        meta_json = f.read(meta_len).decode("utf-8")
                    meta = json.loads(meta_json)
                    meta["format"] = "v2"
                    meta["secure"] = True
                else:
                    # v1 Legacy: Look for sidecar
                    meta_path = rad_file.with_suffix(rad_file.suffix + ".json")
                    if meta_path.exists():
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    meta["format"] = "v1"
                    meta["secure"] = False
            except Exception as meta_err:
                logger.warning(f"[Radiance] Failed to read metadata for {rel_path}: {meta_err}")

            workflows.append({
                "filename": str(rel_path).replace("\\", "/"),
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "metadata": meta,
            })

        workflows.sort(key=lambda x: x["mtime"], reverse=True)
        return web.json_response({"workflows": workflows})

    except Exception as e:
        logger.exception("[Radiance] list_workflows failed")
        return web.json_response({"error": str(e)}, status=500)


@PromptServer.instance.routes.get("/radiance/workflows/get")
@PromptServer.instance.routes.post("/radiance/workflows/get")
async def get_workflow(request):
    try:

        filename = request.rel_url.query.get("filename", "").strip()
        filepath = _resolve_safe_path(filename)

        # FIX 3: _validate_extension() was defined but never called here.
        # Without it, any file type inside WORKFLOW_DIR could be read by name.
        if filepath is None or not _validate_extension(filepath) or not filepath.exists():
            return web.json_response({"error": "File not found"}, status=404)

        raw_data = filepath.read_bytes()
        
        # Check if it's v1 (text) or v2 (binary)
        if raw_data.startswith(RAD_MAGIC):
            # v2 Secure Binary
            graph_json, metadata = _unpack_rad_v2(raw_data)
            return web.json_response({
                "success": True, 
                "content": graph_json, 
                "metadata": metadata,
                "secure": True
            })
        else:
            # v1 Legacy Plain Text
            content = raw_data.decode("utf-8")
            return web.json_response({
                "success": True, 
                "content": content,
                "secure": False
            })

    except Exception as e:
        logger.exception("[Radiance] get_workflow failed")
        return web.json_response({"error": str(e)}, status=500)


@PromptServer.instance.routes.post("/radiance/workflows/delete")
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

        # Clean up empty parent directories (but never WORKFLOW_DIR itself)
        try:
            parent = filepath.parent
            while parent != _WORKFLOW_DIR_RESOLVED:
                if not any(parent.iterdir()):
                    parent.rmdir()
                    parent = parent.parent
                else:
                    break
        except OSError:
            pass  # Directory not empty or permission issue, safe to ignore

        return web.json_response({"success": True})

    except json.JSONDecodeError:
        return web.json_response({"error": "Malformed JSON body"}, status=400)
    except Exception as e:
        logger.exception("[Radiance] delete_workflow failed")
        return web.json_response({"error": str(e)}, status=500)
