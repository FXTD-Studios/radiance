import asyncio
import json
import os
import tempfile
import threading
import re
import math
import traceback
import logging
import torch
import numpy as np
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
try:
    from aiohttp import web
    from server import PromptServer
except ImportError:  # pragma: no cover - exercised by the CI import smoke-test
    # Same guard, same reason as nodes/monitor/viewer.py: viewer.py imports
    # this module at import time, so an unguarded aiohttp here made the whole
    # Review node group unimportable without a running ComfyUI. aiohttp is a
    # required ComfyUI dependency, so a real install never takes this branch.
    web = None
    PromptServer = None
import folder_paths

# Re-import modular components
from radiance.cache import _viewer_cache_get, _viewer_source_get, _progress_set
from radiance.config.constants import VERSION as _RADIANCE_VERSION
from radiance.color.grading import apply_grading

logger = logging.getLogger("radiance.delivery.handler")

_SAFE_FILENAME_RE = re.compile(r'[^\w\s_.() -]', re.UNICODE)

# ── UI → write-engine vocabulary ────────────────────────────────────────────
# The delivery panel (js/radiance_viewer.js) and the write engine
# (radiance/io/writer.py) grew separate names for the same things; the JS
# comment still claims they
# "MUST exactly match", but they have not matched for some time. Translating at
# this boundary keeps both sides untouched. Anything absent from these tables
# has no writer implementation and is rejected explicitly rather than silently
# falling through to a wrong encoder.
_UI_TO_WRITE_FORMAT = {
    'Video — MP4 (H.264)':          'VID │ MP4 (H.264)',
    'Video — MP4 (H.265 10-bit)':   'VID │ MP4 (H.265 10-bit)',
    'Video — MOV (ProRes 422 HQ)':  'VID │ MOV (ProRes 422 HQ)',
    'Video — MOV (ProRes 4444)':    'VID │ MOV (ProRes 4444)',
    'Image Sequence — PNG (8-bit)': 'SEQ │ PNG (8-bit)',
    'Image Sequence — PNG (16-bit)':'SEQ │ PNG (16-bit)',
    'Image Sequence — EXR (32-bit)':'SEQ │ EXR (32-bit float)',
}

_UI_TO_WRITE_COLORSPACE = {
    'Linear (sRGB)':    'Linear (pass-through)',
    'sRGB (Standard)':  'sRGB',
    'ACEScg (AP1)':     'ACEScg',
    'ARRI LogC3':       'ARRI LogC3',
    'ARRI LogC4':       'ARRI LogC4',
    'Sony S-Log3':      'Sony S-Log3',
}


#: Every grading key `/radiance/deliver` reads out of the payload.
#:
#: This list is the contract between js/radiance_viewer.js and this module, and
#: it went out of sync silently: the viewer sent 11 keys while this file read 6
#: more that never arrived -- shadows, highlights, hue_shift, lut_name,
#: lut_intensity and gamut_compression -- each falling back to its identity
#: default. All six are live viewer controls driving real shader uniforms, so a
#: colourist could set Shadows -0.60 and Highlights +0.35, press RENDER, and get
#: a master with neither, a 200 response and "EXPORT COMPLETE" on the HUD. In
#: the other direction `tint` was sent and read nowhere.
#:
#: tests/test_delivery_contract.py compares this list against the keys the JS
#: actually sends, in both directions, so the two cannot drift again.
GRADE_PAYLOAD_KEYS = (
    "exposure", "gamma", "gain", "lift", "offset",
    "contrast", "pivot", "saturation", "temperature", "tint",
    "colorScience", "lumaMix",
    "shadows", "highlights", "hue_shift",
    "lut_name", "lut_intensity", "gamut_compression",
    "grain", "bloom", "halation", "diffusion", "denoise",
)

_warned_missing_grade_keys = False


def _warn_on_missing_grade_keys(grading: dict) -> None:
    """Say so when the payload is short of what the grade reads.

    Once per process: a missing key is a client/server version mismatch, not a
    per-frame condition, and the whole point is that it should be impossible to
    miss rather than impossible to ignore.
    """
    global _warned_missing_grade_keys
    if _warned_missing_grade_keys:
        return
    missing = [k for k in GRADE_PAYLOAD_KEYS if k not in (grading or {})]
    if missing:
        _warned_missing_grade_keys = True
        logger.warning(
            "[Radiance] The delivery payload is missing %d grading key(s): %s. "
            "Those controls will export at their default value even if they are "
            "set in the viewer. This usually means the browser is running a "
            "cached older radiance_viewer.js -- hard-refresh ComfyUI.",
            len(missing), ", ".join(missing),
        )


#: Serialises the read-modify-write of the session log.
#:
#: Deliveries run in a thread-pool executor, so two exports finishing close
#: together both read the file, both append their own entry, and the second
#: write loses the first. The write itself also went straight to the final
#: path, so a crash or a full disk mid-dump left a truncated file that the next
#: read discarded entirely -- the whole history, not just the last entry.
_SESSIONS_LOCK = threading.Lock()


def _write_sessions_atomically(path: str, sessions: list) -> None:
    """Write the session log via a temp file + os.replace, under a lock."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    with _SESSIONS_LOCK:
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".radiance_sessions.",
                                   suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(sessions, handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)          # atomic on POSIX and on Windows
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise


def _resolve_write_format(ui_format: str) -> str:
    if ui_format in _UI_TO_WRITE_FORMAT:
        return _UI_TO_WRITE_FORMAT[ui_format]
    raise ValueError(
        f"Delivery format {ui_format!r} has no writer implementation. "
        f"Supported: {', '.join(sorted(_UI_TO_WRITE_FORMAT))}"
    )


def _resolve_write_colorspace(ui_cs: str) -> str:
    if ui_cs in _UI_TO_WRITE_COLORSPACE:
        return _UI_TO_WRITE_COLORSPACE[ui_cs]
    raise ValueError(
        f"Output colour space {ui_cs!r} is not supported by the writer. "
        f"Supported: {', '.join(sorted(_UI_TO_WRITE_COLORSPACE))}"
    )


_LINEAR_SOURCES = ("Linear Rec.709 (sRGB)", "Linear Rec.2020", "Linear P3-D65", "ACEScg", "ACES2065-1")


def _resolve_export_source(instance_key: str, images) -> tuple:
    """``(encoding, working_space)`` of the cached Viewer frames.

    encoding is "srgb" or "linear"; working_space is the writer's name for
    the linear space the export is written from. Falls back to the Viewer's
    Auto rule when no record exists (a cache filled by an older build).
    """
    rec = _viewer_source_get(instance_key)
    if rec is None:
        from radiance.color.viewer_space import resolve_viewer_input_space
        rec = resolve_viewer_input_space("Auto", images)
    encoding, colorspace = rec
    if encoding == "linear":
        return "linear", colorspace if colorspace in _LINEAR_SOURCES else "Linear Rec.709 (sRGB)"
    return "srgb", "Linear Rec.709 (sRGB)"


def _srgb_decode_np(frame: "np.ndarray") -> "np.ndarray":
    """sRGB EOTF on the RGB channels (sign-preserving); alpha untouched. Returns a copy."""
    out = np.array(frame, dtype=np.float32, copy=True)
    rgb = out[..., :3]
    a = np.abs(rgb)
    lin = np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
    out[..., :3] = lin * np.sign(rgb)
    return out


def _srgb_encode_display(t: "torch.Tensor") -> "torch.Tensor":
    """Linear -> sRGB display values in [0, 1] (for thumbnails); alpha kept."""
    out = t.detach().float().clone()
    rgb = out[..., :3].clamp(0.0, 1.0)
    out[..., :3] = torch.where(rgb <= 0.0031308, rgb * 12.92,
                               1.055 * rgb.clamp(min=1e-12) ** (1 / 2.4) - 0.055)
    return out


def _note(warnings: list, message: str) -> None:
    """Record something the operator asked for that did not happen.

    Deduplicated, because the FX failures are raised inside the per-frame loop
    and a 2000-frame export would otherwise hand the client the same sentence
    2000 times. The first occurrence is the informative one.
    """
    if message not in warnings:
        warnings.append(message)


def get_next_version(directory: str, filename_base: str) -> str:
    """The next version after the highest on disk, as the writer stamps it (``v0002``).

    Four digits since 4.0, matching ``write_frames``, so a delivery carries one
    suffix (``Shot_v0002.mov``) instead of two (``Shot_v02_v0001.mov``). The scan
    reads the first ``_v<digits>`` after the base, so 3.x names keep counting.
    """
    if not os.path.exists(directory):
        return "v0001"
    
    pattern = re.compile(rf"{re.escape(filename_base)}_v(\d+)")
    max_v = 0
    
    try:
        for f in os.listdir(directory):
            match = pattern.search(f)
            if match:
                v = int(match.group(1))
                if v > max_v:
                    max_v = v
    except Exception as exc:
        logger.warning("[radiance.delivery.handler] get_next_version: %s", exc)
        
    return f"v{max_v + 1:04d}"


def _export_aces_clip_xml(media_path: str, grading: dict, color_space: str, version_str: str) -> None:
    """Write an ACES Metadata File (.amf) alongside the exported media."""
    # FIX-006: map the Viewer grade to SOP properly (power = 1/gamma,
    # offset scaled by gain, exposure folded into slope).
    from radiance.color.grading import grading_to_cdl
    _cdl = grading_to_cdl(grading)
    slope, offset, power, sat = _cdl["slope"], _cdl["offset"], _cdl["power"], _cdl["saturation"]

    # Map output color_space to ACES ODT URN
    _ODT_MAP = {
        'ACEScg (AP1)':          'urn:ampas:aces:transformId:v2.0:ACEScsc.Academy.ACEScg_to_ACES.a2.v1',
        'ACES2065-1 (AP0)':      'urn:ampas:aces:transformId:v2.0:ACEScsc.Academy.Identity.a2.v1',
        'ACEScct':               'urn:ampas:aces:transformId:v2.0:ACEScsc.Academy.ACEScct_to_ACES.a2.v1',
        'Linear (sRGB)':         'urn:ampas:aces:transformId:v2.0:ODT.Academy.Rec709_100nits_dim.a1.0.3',
        'sRGB (Standard)':       'urn:ampas:aces:transformId:v2.0:ODT.Academy.sRGB_100nits_dim.a1.0.3',
        'ARRI LogC3':            'urn:ampas:aces:transformId:v2.0:IDT.ARRI.Alexa-v3-LogC-EI800.a1.v2',
        'DaVinci Intermediate':  'urn:ampas:aces:transformId:v2.0:IDT.BlackmagicDesign.DaVinci_Intermediate.a1.v1',
    }
    odt_urn = _ODT_MAP.get(color_space, 'urn:ampas:aces:transformId:v2.0:ODT.Academy.Rec709_100nits_dim.a1.0.3')
    idt_urn = 'urn:ampas:aces:transformId:v2.0:IDT.Academy.Rec709_100nits.a1.0.3'  # default: Rec.709

    # Build XML tree
    ns = 'urn:ampas:aces:schema:v1.0'
    ET.register_namespace('', ns)
    ET.register_namespace('xsi', 'http://www.w3.org/2001/XMLSchema-instance')

    root = ET.Element(f'{{{ns}}}aces_metadata_file')
    root.set('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
    root.set('version', '1.0')

    # Header
    hdr = ET.SubElement(root, f'{{{ns}}}amf_info')
    ET.SubElement(hdr, f'{{{ns}}}description').text = f'Radiance v{_RADIANCE_VERSION} — ACES grade export ({version_str})'
    ET.SubElement(hdr, f'{{{ns}}}uuid').text = f'urn:uuid:{os.urandom(16).hex()}'
    ET.SubElement(hdr, f'{{{ns}}}date_time').text = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

    # Pipeline
    pipeline = ET.SubElement(root, f'{{{ns}}}pipeline')
    pipeline.set('name', f'Radiance_{version_str}')

    # IDT
    idt_node = ET.SubElement(pipeline, f'{{{ns}}}input_transform')
    ET.SubElement(idt_node, f'{{{ns}}}transform_id').text = idt_urn

    # Look Transform (CDL)
    look = ET.SubElement(pipeline, f'{{{ns}}}look_transform')
    look.set('applied', 'true')
    ET.SubElement(look, f'{{{ns}}}description').text = 'ASC CDL via FXTD Radiance'

    cdl_node = ET.SubElement(look, f'{{{ns}}}cdl_transform')
    sop = ET.SubElement(cdl_node, f'{{{ns}}}SOPNode')
    ET.SubElement(sop, f'{{{ns}}}Slope').text  = ' '.join(f'{v:.6f}' for v in slope)
    ET.SubElement(sop, f'{{{ns}}}Offset').text = ' '.join(f'{v:.6f}' for v in offset)
    ET.SubElement(sop, f'{{{ns}}}Power').text  = ' '.join(f'{v:.6f}' for v in power)
    sat_node = ET.SubElement(cdl_node, f'{{{ns}}}SatNode')
    ET.SubElement(sat_node, f'{{{ns}}}Saturation').text = f'{sat:.6f}'

    # ODT
    odt_node = ET.SubElement(pipeline, f'{{{ns}}}output_transform')
    ET.SubElement(odt_node, f'{{{ns}}}transform_id').text = odt_urn

    # Write
    tree = ET.ElementTree(root)
    ET.indent(tree, space='  ')
    amf_path = os.path.splitext(media_path)[0] + '.amf'
    tree.write(amf_path, encoding='utf-8', xml_declaration=True)
    logger.info(f'[Radiance] ACES Metadata File written: {amf_path}')


# Idempotent registration, matching nodes/monitor/viewer.py's _radiance_route_once
# and radiance_ocio.py's flag. Without it, importing this module twice under two
# module paths (delivery/__init__.py imports it unconditionally) raises
# "method POST is already registered" at startup and the whole pack fails to load
# -- exactly the crash those two guards were written for.
_RADIANCE_DELIVER_ROUTE_REGISTERED = getattr(
    getattr(PromptServer, "instance", None), "_radiance_deliver_route_registered", False)


def _register_deliver_route(handler):
    # `PromptServer` is None when aiohttp/ComfyUI are absent (see the guarded
    # import at the top). Reading `.instance` off None at module scope raised
    # AttributeError and took the whole Review group down with it.
    if _RADIANCE_DELIVER_ROUTE_REGISTERED or PromptServer is None:
        logger.debug("[Deliver] Route already registered or no server; skipping.")
        return handler
    try:
        PromptServer.instance.routes.post('/radiance/deliver')(handler)
        PromptServer.instance._radiance_deliver_route_registered = True
    except Exception as exc:  # pragma: no cover - startup-only
        logger.warning("[Deliver] Could not register /radiance/deliver: %s", exc)
    return handler


def _progress_refuse(instance_key, message: str) -> None:
    """Terminate the progress entry on a refusal.

    The JS poll loop only clears its interval on status 'done' or 'error', so
    an early return that left the entry on 'starting' pinned the client's
    progress bar for the life of the process.
    """
    try:
        _progress_set(instance_key, {"current": 0, "total": 100,
                                     "status": "error", "message": message})
    except Exception as _exc:  # progress is cosmetic, never the reason a call fails
        logger.debug("[Radiance] could not set refusal progress: %s", _exc)


async def radiance_deliver_endpoint(request):
    """
    VFX Delivery Endpoint: Receives grading state + export settings from HUD.
    Applies grading, AI-Upscaling, and Burn-ins before high-quality encoding.
    """
    instance_key = ""
    try:
        data = await request.json()
        instance_key = str(data.get('instance_id', ''))
        grading = data.get('grading', {})
        settings = data.get('settings', {})

        images = _viewer_cache_get(instance_key)
        # Initialise progress immediately. It was only ever set at 90% and 100%,
        # so a client polling before the first update read a stale value from a
        # previous run -- and a failure left it pinned at 90% for the life of
        # the process, because the except branch never reset it.
        _progress_set(instance_key, {"current": 0, "total": 100,
                                     "status": "starting", "message": "Preparing…"})
        if images is None:
            _progress_refuse(instance_key, "No frames in cache")
            return web.json_response({"error": "No frames found in cache for this node. Run the workflow first.", "status": "error"}, status=400)
        
        # ─── Render Range ──────────────────────────────────────────────
        range_in = max(1, int(settings.get('range_in', 1)))
        range_out = int(settings.get('range_out', 0))
        total_frames = images.shape[0]
        start_idx = range_in - 1
        end_idx = range_out if range_out > 0 else total_frames
        start_idx = max(0, min(start_idx, total_frames - 1))
        end_idx = max(start_idx + 1, min(end_idx, total_frames))
        images = images[start_idx:end_idx]

        # FIX-018: what the cached pixels are, as the Viewer resolved them.
        # The export used to assume Linear Rec.709 for every source, so a
        # display-encoded ComfyUI IMAGE delivered as "sRGB" was sRGB-encoded a
        # second time (washed out), and as EXR it was labelled linear.
        source_encoding, source_colorspace = _resolve_export_source(instance_key, images)
        
        # ─── Process Options ──────────────────────────────────────────
        filename_prefix = settings.get('filename', 'Radiance_Deliver')
        output_format = settings.get('format', 'Video — MP4 (H.264)')
        fps = float(settings.get('fps', 24.0))
        quality = int(settings.get('quality', 18))
        output_path = settings.get('path', '')

        # ─── Input Sanitization ───────────────────────────────────────
        filename_prefix = _SAFE_FILENAME_RE.sub('', str(filename_prefix)).strip()
        if not filename_prefix:
            filename_prefix = 'Radiance_Deliver'
        if len(filename_prefix) > 200:
            filename_prefix = filename_prefix[:200]

        # Resolve the default BEFORE the sandbox check. Previously the entire
        # check lived inside `if output_path:`, so leaving Location blank -- which
        # the UI placeholder ("Default Output Folder") actively invites -- skipped
        # containment altogether and produced a *relative* path: the master landed
        # in ComfyUI's working directory rather than output/, and the security
        # guard was inert in exactly the default case.
        if not str(output_path).strip():
            try:
                output_path = folder_paths.get_output_directory()
            except Exception as exc:
                logger.error("[Deliver] Could not resolve the ComfyUI output directory: %s", exc)
                return web.json_response(
                    {"error": "Could not resolve the default output directory.",
                     "status": "error"}, status=500)

        if output_path:
            output_path = os.path.abspath(os.path.normpath(str(output_path)))
            if len(output_path) > 1024:
                logger.warning(f"[Deliver] Rejected oversized output path (len={len(output_path)})")
                _progress_refuse(instance_key, "Invalid output path")
                return web.json_response({"error": "Invalid output path", "status": "error"}, status=400)
            try:
                _allowed_root = os.path.abspath(folder_paths.get_output_directory())
            except Exception:
                _allowed_root = None
            if _allowed_root:
                try:
                    _rel = os.path.relpath(output_path, _allowed_root)
                    _outside = _rel.startswith("..")
                except ValueError:
                    _outside = True
                if _outside:
                    logger.warning(f"[Deliver] Rejected output path outside ComfyUI output dir: {output_path[:120]}")
                    _progress_refuse(instance_key, "Output path outside the output directory")
                    return web.json_response({"error": "Output path must be inside the ComfyUI output directory.", "status": "error"}, status=403)

        # Clamp numeric parameters to sane ranges
        fps = max(1.0, min(fps, 240.0))
        quality = max(0, min(quality, 51))

        color_space = settings.get('colorSpace', 'sRGB (Standard)')
        broadcast_safe = settings.get('soft_clip', True)
        
        upscale_2x = settings.get('upscale_2x', False)
        smart_ver = settings.get('smart_versioning', True)

        # Resolve the output vocabulary BEFORE anything expensive runs.
        #
        # `_resolve_write_format` / `_resolve_write_colorspace` raise for the
        # eleven entries the delivery panel offers and the writer does not
        # implement, and they used to be evaluated inside the `write_frames(...)`
        # call at the very END of `_run_export` -- after per-frame grading, the
        # FX bake, the 2x model upscale and the QC pass had all run over the
        # whole batch. Picking "ProRes 4444 XQ" therefore cost minutes of work
        # and then a 500. The check is free; it belongs here.
        try:
            write_format = _resolve_write_format(output_format)
            write_colorspace = _resolve_write_colorspace(color_space)
        except ValueError as exc:
            logger.warning("[Deliver] Rejected before grading: %s", exc)
            # Terminal progress, at 100, exactly as the post-grading failure
            # path reports it -- the JS clears its poll interval on status
            # 'error' and the bar should not be left short of the end.
            try:
                _progress_set(instance_key, {"current": 100, "total": 100,
                                             "status": "error",
                                             "message": str(exc)[:200]})
            except Exception as _exc:
                logger.debug("[Radiance] could not set refusal progress: %s", _exc)
            # 400, not 500: the request named a format or a colour space that
            # does not exist. It used to surface as a 500 only because the
            # ValueError escaped from the middle of the export closure.
            return web.json_response({"error": str(exc), "status": "error"}, status=400)


        # Handle Versioning
        # One suffix: write_frames stamps `_v{version:04d}` onto filename_prefix.
        # This used to append its own `_v02` first, giving Shot_v02_v0001.
        version_str = "v0001"
        if smart_ver:
            base_path = output_path if output_path else folder_paths.get_output_directory()
            version_str = get_next_version(base_path, filename_prefix)
        version_num = int(version_str[1:])

        # ─── Process Grading ──────────────────────────────────────────
        def safe_float(v, default):
            if isinstance(v, (list, tuple)) and len(v) > 0: return float(v[0])
            try: return float(v)
            except (TypeError, ValueError): return default

        def safe_array(v, default_scalar, length=3):
            if isinstance(v, (list, tuple)) and len(v) >= length:
                return [float(x) for x in v[:length]]
            try:
                s = float(v)
                return [s] * length
            except (TypeError, ValueError):
                return [default_scalar] * length

        # Every key this function reads. The viewer must send all of them --
        # `_warn_on_missing_grade_keys` says so out loud rather than letting a
        # silent .get() default turn a colourist's decision into a no-op.
        _warn_on_missing_grade_keys(grading)

        exposure    = safe_float(grading.get('exposure'), 0.0)
        saturation  = safe_float(grading.get('saturation'), 1.0)

        # TEMP/TINT are the viewer's additive sliders in [-2, 2], not Kelvin.
        # This used to be `temperature = 6500.0 + temp * 3500.0`, feeding a
        # Kelvin white-balance multiply that has nothing to do with the shader's
        # `shift.r += temp` -- so the export applied a different curve from the
        # one on screen. `tint` was read nowhere at all, so the green/magenta
        # axis was silently dropped from every master.
        temp_shift  = safe_float(grading.get('temperature'), 0.0)
        tint_shift  = safe_float(grading.get('tint'), 0.0)
        temperature = 6500.0
        contrast    = safe_float(grading.get('contrast'), 1.0)
        pivot       = safe_float(grading.get('pivot'), 0.18)
        shadows     = safe_float(grading.get('shadows'), 0.0)
        highlights  = safe_float(grading.get('highlights'), 0.0)
        hue_shift   = safe_float(grading.get('hue_shift'), 0.0)
        lut_name    = grading.get('lut_name', 'None')
        lut_intensity = safe_float(grading.get('lut_intensity'), 1.0)
        gamut_compression = bool(grading.get('gamut_compression', False))

        gamma_rgb  = safe_array(grading.get('gamma'),  1.0)
        gain_rgb   = safe_array(grading.get('gain'),   1.0)
        lift_rgb   = safe_array(grading.get('lift'),   0.0)
        offset_rgb = safe_array(grading.get('offset'), 0.0)

        # ── Everything below runs OFF the event loop ─────────────────────
        # This entire block used to execute inside `async def`, on aiohttp's
        # event loop thread: per-frame NumPy grading, cv2 filters, a 2x model
        # upscale, and a blocking ffmpeg subprocess. A 240-frame 1080p export
        # therefore froze ComfyUI's whole websocket for minutes -- progress bar,
        # node highlighting and queue view all stalled, and the client's own
        # poll of /radiance/progress could not be answered, so the bar sat at 0%
        # and jumped straight to 100%. `_progress_set(..., 90, "encoding")` was
        # unobservable by construction.
        #
        # It is now a closure handed to run_in_executor, so it captures the
        # locals above unchanged while the loop stays free to serve /progress.
        def _run_export():
            # Every user-requested transform that did not happen.
            #
            # AUDIT-FIX: the AI upscale, the aspect blanking, the FX bake, the
            # denoise, the halation, the bloom and the diffusion each caught
            # their own failure, logged it and carried on, while the response
            # still said status "success" and the HUD still showed "EXPORT
            # COMPLETE". A master delivered at 1x because the ESRGAN model was
            # missing is not a successful delivery. Anything appended here
            # travels back to the client and turns the response into "partial".
            warnings: list = []

            # Grade into ONE preallocated batch, frame by frame.
            #
            # This used to build `out_batch`, a list of N graded frames, and
            # then `torch.stack` it -- two more full copies of the sequence on
            # top of the viewer cache entry, before the FX bake made two more.
            # Measured ceiling for a delivery was about 1000 frames at 1080p.
            # The destination is allocated once and each frame is written into
            # its slot, so the grade costs the output batch plus one frame.
            # Frame 0 is graded first purely to learn the output shape, rather
            # than assuming the grade cannot change the channel count.
            graded_tensor = None
            _n_frames = max(1, int(images.shape[0]))
            for i in range(images.shape[0]):
                # Grading is the long pole; report it. Now that the work runs in
                # an executor the event loop is free to answer /radiance/progress,
                # so this is actually observable.
                if i % 4 == 0 or i == _n_frames - 1:
                    _progress_set(instance_key, {
                        "current": int(5 + 80 * i / _n_frames), "total": 100,
                        "status": "grading", "message": f"Grading frame {i + 1}/{_n_frames}"})
                frame_np = images[i].cpu().numpy()
                if source_encoding == "srgb":
                    # FIX-018: the Viewer linearises a display-encoded source
                    # BEFORE grading (radiance_webgl.js "1b. Linearize"); the
                    # export graded the encoded values. Decode first, as the
                    # Viewer does, so the grade and the master match it.
                    frame_np = _srgb_decode_np(frame_np)
                graded = apply_grading(
                    img=frame_np,
                    exposure=exposure,
                    gamma=1.0,
                    gain=1.0,
                    lift=0.0,
                    saturation=saturation,
                    temperature=temperature,
                    temp_shift=temp_shift,
                    tint_shift=tint_shift,
                    offset=0.0,
                    contrast=contrast,
                    pivot=pivot,
                    shadows=shadows,
                    highlights=highlights,
                    hue_shift=hue_shift,
                    gamma_rgb=gamma_rgb,
                    gain_rgb=gain_rgb,
                    lift_rgb=lift_rgb,
                    offset_rgb=offset_rgb,
                    lut_name=lut_name,
                    lut_intensity=lut_intensity,
                    color_science=1 if str(grading.get('colorScience')) in ['1', 'ACEScct'] else 0,
                    luma_mix=float(grading.get('lumaMix', 1.0)),
                    gamut_compression=gamut_compression
                )
                if graded_tensor is None:
                    graded_tensor = torch.empty(
                        (images.shape[0],) + tuple(graded.shape),
                        dtype=torch.float32)
                graded_tensor[i] = torch.from_numpy(
                    np.ascontiguousarray(graded, dtype=np.float32))
                del graded

            # ─── FX Baking ────────────────────────────────────────────────
            _grain     = float(grading.get('grain', 0.0))
            _bloom     = float(grading.get('bloom', 0.0))
            _halation  = float(grading.get('halation', 0.0))
            _diffusion = float(grading.get('diffusion', 0.0))
            _denoise   = float(grading.get('denoise', 0.0))

            if _grain > 0.01 or _bloom > 0.01 or _halation > 0.01 or _diffusion > 0.01 or _denoise > 0.01:
                try:
                    # A view of the graded batch, written back in place. The
                    # filters used to collect every frame into `fx_batch` and
                    # `np.stack` it into a new tensor, so the FX pass cost two
                    # more full copies of the sequence on top of the grade's.
                    np_batch = graded_tensor.cpu().numpy()
                    rng = np.random.default_rng(seed=42)
                    for i_frame in range(np_batch.shape[0]):
                        f = np_batch[i_frame].astype(np.float32)

                        if _denoise > 0.01:
                            try:
                                import cv2 as _cv2
                                sigma = _denoise * 3.0
                                # OpenCV's bilateral filter takes 8U or 32F
                                # only. uint16 raised every time and the
                                # DEBUG-level except below swallowed it, so
                                # Denoise silently did nothing.
                                f = _cv2.bilateralFilter(
                                    np.ascontiguousarray(f, dtype=np.float32),
                                    d=5, sigmaColor=sigma, sigmaSpace=sigma * 20,
                                )
                            except ImportError as _exc:
                                logger.warning(
                                    "[Radiance] Denoise needs OpenCV and it is "
                                    "not installed (%s); the master was written "
                                    "without it.", _exc,
                                )
                                _note(warnings,
                                      f"Denoise skipped: OpenCV is not installed ({_exc})")
                            except Exception as _exc:
                                logger.warning(
                                    "[Radiance] Denoise failed (%s: %s); the "
                                    "master was written without it.",
                                    type(_exc).__name__, _exc,
                                )
                                _note(warnings,
                                      f"Denoise failed: {type(_exc).__name__}: {_exc}")

                        if _grain > 0.01:
                            noise = rng.standard_normal(f.shape).astype(np.float32)
                            f = f + noise * _grain * 0.02

                        if _halation > 0.01:
                            try:
                                import cv2 as _cv2
                                hi = np.clip(f[..., 0] - 0.8, 0, None)
                                blurred = _cv2.GaussianBlur(hi, (0, 0), sigmaX=_halation * 30)
                                f[..., 0] = np.minimum(f[..., 0] + blurred * _halation * 2.0, 2.0)
                            except Exception as exc:
                                logger.warning("[radiance.delivery.handler]: %s", exc)
                                _note(warnings, f"Halation failed: {type(exc).__name__}: {exc}")

                        if _bloom > 0.01:
                            try:
                                import cv2 as _cv2
                                lum = 0.2126*f[...,0] + 0.7152*f[...,1] + 0.0722*f[...,2]
                                hi = np.clip(lum - 0.7, 0, None)[..., np.newaxis]
                                blurred = _cv2.GaussianBlur(hi * np.ones((1,1,3), np.float32), (0, 0), sigmaX=_bloom * 40)
                                f = f + blurred * _bloom
                            except Exception as exc:
                                logger.warning("[radiance.delivery.handler]: %s", exc)
                                _note(warnings, f"Bloom failed: {type(exc).__name__}: {exc}")

                        if _diffusion > 0.01:
                            try:
                                import cv2 as _cv2
                                soft = _cv2.GaussianBlur(f, (0, 0), sigmaX=_diffusion * 20)
                                f = f * (1 - _diffusion * 0.5) + soft * _diffusion * 0.5
                            except Exception as exc:
                                logger.warning("[radiance.delivery.handler]: %s", exc)
                                _note(warnings, f"Diffusion failed: {type(exc).__name__}: {exc}")

                        np_batch[i_frame] = f
                        del f
                    logger.info(f"[Deliver] FX baked: grain={_grain:.2f} bloom={_bloom:.2f} halation={_halation:.2f} diffusion={_diffusion:.2f}")
                except Exception as e:
                    logger.warning(f"[Deliver] FX baking failed: {e}")
                    _note(warnings, f"FX bake did not run: {type(e).__name__}: {e}")

            # ─── AI Upscale (2x) ──────────────────────────────────────────
            if upscale_2x:
                try:
                    # Imported from the library, not from the node surface.
                    #
                    # This used to read `from radiance.nodes_upscale import
                    # RadianceAIUpscale` — delivery reaching up into the ComfyUI
                    # node layer for a class that lives one directory away. The
                    # stated reason was that tests/conftest.py replaced
                    # `radiance.image` with a stub, so the real submodule could
                    # not be imported under the harness. That was a fixable
                    # stub, not a fact about the code: the stub now keeps a
                    # __path__, so only `defects` is faked and every other
                    # submodule loads from disk.
                    from radiance.image.upscale import RadianceAIUpscale
                    upscaler = RadianceAIUpscale()
                    graded_tensor, upscale_info = upscaler.upscale(
                        image=graded_tensor,
                        model_name="RealESRGAN_x2plus",
                        mode="Refine (HDR)",
                        tile_size=512
                    )
                    # A missing model, failed download or failed inference does
                    # not raise: upscale() returns a bicubic resize instead.
                    if getattr(upscaler, "used_fallback", False):
                        _note(warnings,
                              f"AI upscale did not run, the master was resampled "
                              f"(bicubic), not AI-upscaled: {upscale_info}")
                except Exception as e:
                    logger.error(f"AI Upscale failed, continuing with original: {e}")
                    _note(warnings,
                          f"AI upscale did not run, the master is 1x: "
                          f"{type(e).__name__}: {e}")

            # ─── Aspect Ratio Blanking ────────────────────────────────────
            aspect_ratio_str = settings.get('aspect_ratio', 'None')
            if aspect_ratio_str != 'None':
                try:
                    # Parse BOTH terms. This used to read only the numerator, so
                    # "9:16 (Vertical)" became the ratio 9.0 instead of 0.5625 --
                    # a 1920x1080 plate then took the horizontal branch and was
                    # blanked down to a 214px letterbox slit, delivered as success.
                    # The other three presets survived only because their
                    # denominator happens to be 1.
                    ratio_token = aspect_ratio_str.split()[0]
                    num, _, den = ratio_token.partition(':')
                    target_ratio = float(num) / float(den or 1.0)
                    if target_ratio <= 0.0:
                        raise ValueError(f"non-positive aspect ratio {aspect_ratio_str!r}")

                    _, h, w, _ = graded_tensor.shape
                    current_ratio = w / h
                    # `pad > 0` guards are required: `t[:, :, -0:, :] = 0` is
                    # `t[:, :, 0:, :] = 0`, which blanks the ENTIRE frame.
                    if current_ratio > target_ratio + 0.01:
                        pad = (w - int(h * target_ratio)) // 2
                        if pad > 0:
                            graded_tensor[:, :, :pad, :] = 0.0
                            graded_tensor[:, :, -pad:, :] = 0.0
                    elif current_ratio < target_ratio - 0.01:
                        pad = (h - int(w / target_ratio)) // 2
                        if pad > 0:
                            graded_tensor[:, :pad, :, :] = 0.0
                            graded_tensor[:, -pad:, :, :] = 0.0
                except Exception as e:
                    logger.error(f"Aspect blanking failed: {e}")
                    _note(warnings,
                          f"Aspect blanking to {aspect_ratio_str} did not run: "
                          f"{type(e).__name__}: {e}")

            # ─── Integrated QC Pass ───────────────────────────────────────
            qc_report = ""
            try:
                v_min = graded_tensor.min().item()
                v_max = graded_tensor.max().item()

                is_aces_cs = color_space in ('ACEScg (AP1)', 'ACES2065-1 (AP0)', 'ACEScct')

                if is_aces_cs:
                    M_AP1_TO_REC2020 = np.array([
                        [ 1.70505,  -0.62179, -0.08326],
                        [-0.13026,   1.14080, -0.01054],
                        [-0.02400,  -0.12897,  1.15297],
                    ], dtype=np.float32)
                    try:
                        sample = graded_tensor[0].cpu().numpy()
                        if sample.ndim == 3 and sample.shape[2] >= 3:
                            rgb_ap1 = sample[..., :3]
                            rec2020 = np.tensordot(rgb_ap1, M_AP1_TO_REC2020, axes=([2], [1]))
                            imaginary_pct = float(np.mean(np.any(rec2020 < -0.001, axis=-1)) * 100)
                            if imaginary_pct > 0.5:
                                qc_report = (
                                    f"◎ [QC WARNING] {imaginary_pct:.1f}% imaginary-gamut pixels detected "
                                    f"(outside Rec.2020 after AP1→Rec.2020 transform). "
                                    f"Apply gamut compression before display-referred delivery."
                                )
                            else:
                                qc_report = (
                                    f"◎ [QC PASS — ACES] {color_space} · "
                                    f"Imaginary gamut: {imaginary_pct:.2f}% (within Rec.2020 tolerance)."
                                )
                        else:
                            qc_report = f"◎ [QC PASS — ACES] {color_space} output."
                    except Exception as _e:
                        qc_report = f"◎ [QC PASS — ACES] {color_space} (gamut check skipped: {_e})."
                elif v_min < 0.0 or v_max > 1.0:
                    qc_report = (
                        f"◎ [QC WARNING] Out-of-Gamut detected: "
                        f"Min={v_min:.3f}, Max={v_max:.3f}. "
                        f"(Illegal levels for broadcast)."
                    )
                else:
                    qc_report = "◎ [QC PASS] Levels within legal broadcast range (0.0 - 1.0)."
            except Exception as exc:
                logger.warning("[radiance.delivery.handler]: %s", exc)

            # ─── Save through the write engine ─────────────────────────────
            #
            # radiance.io.writer, not radiance.nodes.io.write. This used to
            # instantiate the node and call .write() on it -- the delivery path
            # reaching up into the node layer, which is why the handler could
            # not be exercised without importing ComfyUI's node surface. The
            # engine is a floor below both of us now and imports nothing above
            # itself.
            from radiance.io.writer import write_frames

            is_exr = 'EXR' in output_format or 'exr' in output_format.lower()
            bake_grade_exr = settings.get('bake_grade', False) and is_exr

            # FIX-018: hand the writer linear values in the source's space.
            # A display-encoded (sRGB) source is decoded once here, so every
            # output encoding -- sRGB included -- is applied exactly once.
            # This replaces the old bake branch that guessed the source from
            # the OUTPUT colour space. The thumbnail keeps the display frame.
            # graded_tensor is linear in the source's space: a display-encoded
            # source was decoded before grading (above). The thumbnail is a
            # display picture, so it is sRGB-encoded from the linear grade.
            n_graded = graded_tensor.shape[0]
            thumb_frame = _srgb_encode_display(graded_tensor[min(n_graded - 1, 10 if n_graded > 10 else 0)])
            write_working_space = source_colorspace

            if bake_grade_exr:
                # The EXR holds scene-linear values in the source's space; the
                # writer must not re-encode them.
                write_colorspace_effective = 'Linear (pass-through)'
                logger.info(f'[Deliver] Grade baked into linear EXR ({write_working_space})')
            else:
                write_colorspace_effective = write_colorspace

            # ─── Shot Continuity QC ───────────────────────────────────────
            continuity_report = []
            if graded_tensor.shape[0] > 1:
                try:
                    frames_np = graded_tensor.cpu().numpy()
                    def _frame_midluma(f):
                        luma = 0.2126*f[...,0] + 0.7152*f[...,1] + 0.0722*f[...,2]
                        s = np.sort(luma.ravel())
                        return float(s[len(s)//2])
                    mid_lumas = [_frame_midluma(frames_np[i]) for i in range(frames_np.shape[0])]
                    for i in range(1, len(mid_lumas)):
                        p0, p1 = max(mid_lumas[i-1], 1e-6), max(mid_lumas[i], 1e-6)
                        ev_delta = abs(math.log2(p1 / p0))
                        if ev_delta > 0.5:
                            continuity_report.append(
                                f'Frame {i-1}→{i}: {ev_delta:.2f} EV jump (mid-luma {p0:.3f}→{p1:.3f})'
                            )
                    if continuity_report:
                        logger.warning(f'[Deliver] Continuity issues: {len(continuity_report)} flicker events')
                except Exception as _e:
                    logger.debug(f'[Deliver] Continuity scan failed: {_e}')

            _progress_set(instance_key, {"current": 90, "total": 100, "status": "encoding", "message": "Encoding Master..."})

            # NOTE: these kwargs must track write_frames' real signature.
            # They previously read filename_prefix / output_format /
            # output_color_space, none of which exist, and omitted the required
            # `format` -- so every delivery raised TypeError, was swallowed by the
            # handler below, and returned HTTP 200 with status "error".
            #
            # No read_media callback: the delivery path always hands the writer
            # a tensor, so the branch that needs a decoder is never reached.
            #
            # `format` and `color_space` were resolved before any of the work
            # above ran; see the validation block in the endpoint.
            #
            # `start_frame` is passed because it was not, and a sequence export
            # of a trimmed range was therefore always numbered from 1001 no
            # matter where the range started -- frame 120 of the shot delivered
            # as 1001, and no way to tell from the filenames. `range_in` is
            # 1-based into the cached batch, so frame 1 maps to 1001, which is
            # the numbering the rest of the pipeline assumes.
            path, _count = write_frames(
                image=graded_tensor,
                output_path=output_path,
                format=write_format,
                filename=filename_prefix,
                version=version_num,
                color_space=write_colorspace_effective,
                working_space=write_working_space,
                fps=fps,
                quality=quality,
                broadcast_safe=broadcast_safe,
                start_frame=1000 + range_in,
            )

            # ─── Post-Export: Thumbnails & Sidecars ────────────────────────
            try:
                import PIL.Image
                thumb_np = (thumb_frame.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                thumb_img = PIL.Image.fromarray(thumb_np)
                # JPEG cannot store an alpha channel: PIL raises
                # "cannot write mode RGBA as JPEG". Because this was the FIRST
                # statement in the shared sidecar try-block, an RGBA plate
                # aborted the CDL export, the AMF export, the .json sidecar and
                # the reveal-folder hook as well -- while the response still
                # said "success". Each sidecar now also fails independently.
                if thumb_img.mode not in ("RGB", "L"):
                    thumb_img = thumb_img.convert("RGB")
                thumb_path = os.path.splitext(path)[0] + "_thumb.jpg"
                try:
                    thumb_img.save(thumb_path, quality=85)
                except Exception as _te:
                    logger.warning("[Deliver] Thumbnail failed (non-fatal): %s", _te)
            
                # ASC CDL Sidecar Export
                if settings.get('export_cdl', False):
                    from radiance.color.grading import grading_to_cdl
                    _cdl = grading_to_cdl(grading)
                    slope, offset, power, sat = (_cdl["slope"], _cdl["offset"],
                                                 _cdl["power"], _cdl["saturation"])
                    if not _cdl["exact"]:
                        logger.warning("[Deliver] CDL sidecar cannot carry: %s; it approximates the grade.",
                                       ", ".join(_cdl["not_represented"]))
                    cdl_path = os.path.splitext(path)[0] + ".cdl"
                    try:
                        from radiance.io.formats import write_cdl_file
                        write_cdl_file(cdl_path, slope, offset, power, sat,
                                       description=f"Radiance grade {version_str}",
                                       cc_id=os.path.splitext(os.path.basename(path))[0])
                    except Exception as _ce:
                        logger.warning("[Deliver] CDL export failed: %s", _ce)

                # AMF Export
                if settings.get('export_amf', False):
                    try:
                        _export_aces_clip_xml(path, grading, color_space, version_str)
                    except Exception as _ae:
                        logger.warning("[Deliver] AMF export failed: %s", _ae)
            
                # Save Metadata Sidecar (.json)
                meta = {
                    "version": version_str,
                    "grading": grading,
                    "export_settings": settings,
                    "qc": qc_report,
                    "continuity": continuity_report if continuity_report else "PASS",
                    "bake_grade_exr": bake_grade_exr,
                    "timestamp": os.path.getmtime(path)
                }
                with open(os.path.splitext(path)[0] + "_meta.json", 'w') as f:
                    json.dump(meta, f, indent=4)
                
                # Reveal Folder Hook
                if settings.get('reveal_folder', False):
                    import platform, subprocess
                    try:
                        p = os.path.abspath(output_path)
                        if platform.system() == "Windows":
                            os.startfile(p)
                        elif platform.system() == "Darwin":
                            subprocess.Popen(["open", p])
                        else:
                            subprocess.Popen(["xdg-open", p])
                    except Exception as e:
                        logger.error(f"Launch folder failed: {e}")
                    
            except Exception as e:
                logger.error(f"Sidecar generation failed: {e}")

            _progress_set(instance_key, {"current": 100, "total": 100, "status": "done", "message": "Delivery Complete"})

            # Session Tracker
            try:
                _sessions_path = os.path.join(folder_paths.get_output_directory(), 'radiance_sessions.json')
                _sessions = []
                if os.path.exists(_sessions_path):
                    try:
                        with open(_sessions_path, 'r') as _sf:
                            _sessions = json.load(_sf)
                    except Exception:
                        _sessions = []
                _session_entry = {
                    "timestamp":    datetime.now().isoformat(timespec='seconds'),
                    "shot":         os.path.basename(path),
                    "version":      version_str,
                    "format":       output_format,
                    "color_space":  color_space,
                    "frames":       int(graded_tensor.shape[0]),
                    "resolution":   f"{graded_tensor.shape[2]}×{graded_tensor.shape[1]}",
                    "exposure":     round(exposure, 3),
                    "saturation":   round(saturation, 3),
                    "gain_rgb":     [round(v, 4) for v in gain_rgb],
                    "lift_rgb":     [round(v, 4) for v in lift_rgb],
                    "gamma_rgb":    [round(v, 4) for v in gamma_rgb],
                    "qc":           qc_report[:120] if qc_report else "",
                    "continuity":   f"{len(continuity_report)} events" if continuity_report else "PASS",
                    "bake_grade":   bake_grade_exr,
                }
                _sessions.append(_session_entry)
                if len(_sessions) > 500:
                    _sessions = _sessions[-500:]
                _write_sessions_atomically(_sessions_path, _sessions)
                logger.debug(f'[Deliver] Session logged → {_sessions_path}')
            except Exception as _e:
                logger.debug(f'[Deliver] Session log failed (non-fatal): {_e}')

            _cont_warn = ""
            if continuity_report:
                _cont_warn = f" ⚠ {len(continuity_report)} flicker event(s): " + "; ".join(continuity_report[:3])
                if len(continuity_report) > 3:
                    _cont_warn += f" (+{len(continuity_report)-3} more)"

            return path, qc_report, _cont_warn, continuity_report, warnings

        loop = asyncio.get_running_loop()
        path, qc_report, _cont_warn, continuity_report, warnings = (
            await loop.run_in_executor(None, _run_export))

        # AUDIT-FIX: this returned status "success" unconditionally, so a
        # master delivered at 1x because the ESRGAN model was missing, or
        # without the FX the colourist baked in, or unblanked, read as a clean
        # delivery to every client and put "EXPORT COMPLETE" on the HUD. The
        # response now carries what did not happen, and says "partial" when
        # anything did not.
        _status = "partial" if warnings else "success"
        _message = f"Export delivered successfully: {os.path.basename(path)} ({version_str})"
        if warnings:
            _message = (
                f"Export delivered WITHOUT {len(warnings)} requested step(s): "
                f"{os.path.basename(path)} ({version_str}). " + " ".join(warnings)
            )
        return web.json_response({
            "status": _status,
            "path": path,
            "qc": qc_report + _cont_warn,
            "continuity": continuity_report,
            "warnings": warnings,
            "message": _message,
        })

    except Exception as e:
        logger.error(f"Delivery failed: {traceback.format_exc()}")
        # Mark the run failed so the client's poll loop terminates. Without this
        # the progress entry stayed at 90%/"encoding" forever, and the JS only
        # clears its interval on status in ('done', 'error').
        if instance_key:
            try:
                _progress_set(instance_key, {"current": 100, "total": 100,
                                             "status": "error", "message": str(e)[:200]})
            except Exception as _exc:
                logger.debug(
                    "[Radiance] radiance_deliver_endpoint(): ignoring %s from `_progress_set(instance_key, {'current': 100, 'total': 100, '…`: %s",
                    type(_exc).__name__, _exc,
                )
        # status=500, not the aiohttp default of 200 -- a failed delivery that
        # answers 200 reads as success to every client.
        return web.json_response({"error": str(e), "status": "error"}, status=500)


_register_deliver_route(radiance_deliver_endpoint)
