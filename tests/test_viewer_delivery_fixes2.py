"""Regression tests for the second round of viewer / delivery fixes.

These cover the items the first pass reported but did not fix: the blocking
export, the scope read-back format, the frame cache, temp-file accumulation,
and the medium-severity batch.
"""
import ast
import os
import pathlib
import tempfile
import types

import pytest

torch = pytest.importorskip("torch")

# Most assertions here read source text and run fine against conftest's torch
# stub, so opt out of the automatic module-level skip and gate the handful
# of tests that build real tensors with @pytest.mark.real_torch.
RADIANCE_TORCH_GATED = True

_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _src(rel):
    return (_ROOT / rel).read_text(encoding="utf-8")


# ── 1. The export must not block the event loop ─────────────────────────────

def test_delivery_runs_in_an_executor():
    """
    348 lines of NumPy grading, cv2 filters, a model upscale and a blocking
    ffmpeg call used to run on aiohttp's event loop, freezing ComfyUI's
    websocket for the whole export.
    """
    src = _src("delivery/handler.py")
    assert "run_in_executor" in src, "the export is back on the event loop"
    assert "import asyncio" in src


def test_export_closure_has_exactly_one_return():
    """The closure must return the values the HTTP response needs."""
    tree = ast.parse(_src("delivery/handler.py"))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_run_export")

    own = []

    def visit(node):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            if isinstance(child, ast.Return):
                own.append(child)
            visit(child)

    visit(fn)
    assert len(own) == 1, f"_run_export has {len(own)} top-level returns"
    assert ast.unparse(own[0]) == (
        "return (path, qc_report, _cont_warn, continuity_report, warnings)")


def test_progress_is_initialised_and_reset_on_failure():
    src = _src("delivery/handler.py")
    assert '"status": "starting"' in src, "progress never initialised"
    assert '"status": "error"' in src, "progress not reset on failure"
    assert '"status": "grading"' in src, "no per-frame progress during the long pole"


# ── 2. Scope read-back must match the attachment format ─────────────────────

def test_scope_readback_matches_the_fbo_type():
    """
    The scope FBO is created at pipelinePrecision (f32 -> RGBA32F) but the
    read-back was hard-coded to UNSIGNED_BYTE, which WebGL2 only accepts for
    normalized fixed-point buffers -> INVALID_OPERATION and blank scopes.
    """
    src = _src("js/radiance_webgl.js")
    assert "_scopeFBOType" in src, "the FBO type is not recorded"
    assert "gl.readPixels(0, 0, size, size, gl.RGBA, gl.FLOAT" in src, \
        "no float read-back path"
    assert "_scopePixelsF32" in src, "no Float32Array staging buffer"


def test_scope_fallback_path_still_uses_bytes():
    """The RGBA8 fallback must keep its UNSIGNED_BYTE read."""
    src = _src("js/radiance_webgl.js")
    assert "gl.readPixels(0, 0, size, size, gl.RGBA, gl.UNSIGNED_BYTE, pixels)" in src


def test_webgl_destroy_releases_context_and_clears_maps():
    src = _src("js/radiance_webgl.js")
    body = src[src.index("    destroy("):][:4000]
    assert "this.textures = {}" in body and "this.programs = {}" in body
    assert "loseContext" in body


# ── 3. Frame cache: bytes, not entries; detached CPU copies ─────────────────

def _load_cache_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("_rad_cache", _ROOT / "cache.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cache_evicts_on_a_byte_budget():
    m = _load_cache_module()
    m._VIEWER_CACHE_MAX_BYTES = 4 * 1024 * 1024
    for i in range(6):
        m._viewer_cache_set(f"k{i}", torch.zeros(1, 512, 512, 3))  # ~3 MB each
    total = sum(m._tensor_nbytes(v) for v in m._VIEWER_CACHE.values())
    assert total <= m._VIEWER_CACHE_MAX_BYTES + 4 * 1024 * 1024
    assert m._viewer_cache_get("k5") is not None, "the newest entry was evicted"


@pytest.mark.real_torch
def test_cache_stores_a_detached_cpu_copy():
    """
    The node returns the same object it caches, so an un-copied entry could be
    mutated in place by a downstream node before delivery read it.
    """
    m = _load_cache_module()
    src = torch.ones(1, 4, 4, 3)
    m._viewer_cache_set("alias", src)
    cached = m._viewer_cache_get("alias")
    src[0, 0, 0, 0] = 99.0
    assert float(cached[0, 0, 0, 0]) == 1.0, "cache entry aliases the caller's tensor"
    assert cached.device.type == "cpu"


def test_cache_never_evicts_its_only_entry():
    """A single plate larger than the whole budget must still be retrievable."""
    m = _load_cache_module()
    m._VIEWER_CACHE_MAX_BYTES = 1024
    m._viewer_cache_set("big", torch.zeros(1, 256, 256, 3))
    assert m._viewer_cache_get("big") is not None


def test_cache_key_is_scoped_to_the_graph():
    """node id alone collides across workflows and exported the wrong plate."""
    src = _src("nodes/monitor/viewer.py")
    assert "_graph_part" in src and "prompt_id" in src, \
        "the cache key is back to a bare node id"


# ── 4. Temp files ───────────────────────────────────────────────────────────

def test_temp_file_helpers_purge_the_previous_generation():
    src = _src("nodes/monitor/viewer.py")
    start = src.index("_VIEWER_TEMP_FILES")
    end = src.index("class RadianceViewer:")
    ns = {"Dict": dict, "List": list, "os": os,
          "logger": types.SimpleNamespace(debug=lambda *a, **k: None)}
    exec(src[start:end], ns)

    d = tempfile.mkdtemp()
    paths = [os.path.join(d, f"f{i}.rhdr") for i in range(5)]
    for p in paths:
        open(p, "w").write("x")

    ns["_viewer_track_temp"]("n5", paths)
    assert ns["_viewer_purge_temp"]("n5") == 5
    assert not any(os.path.exists(p) for p in paths)
    # idempotent, and tolerant of unknown keys / already-deleted files
    assert ns["_viewer_purge_temp"]("n5") == 0
    assert ns["_viewer_purge_temp"]("never-seen") == 0


def test_viewer_purges_before_writing_and_tracks_after():
    src = _src("nodes/monitor/viewer.py")
    assert "_viewer_purge_temp(_purge_key)" in src
    assert "_viewer_track_temp(_purge_key" in src


def test_float_payloads_use_cheap_compression():
    """level 6 cost ~0.6s per 32MB frame for ~2% over level 1."""
    src = _src("nodes/monitor/viewer.py")
    assert "level=6" not in src, "zlib level 6 is back on a float payload"
    assert src.count("level=1") >= 3


# ── 5. Medium batch ─────────────────────────────────────────────────────────

def test_rgba_thumbnail_is_converted_before_jpeg_save():
    src = _src("delivery/handler.py")
    assert 'thumb_img.convert("RGB")' in src, "RGBA -> JPEG will raise again"


def test_sidecars_fail_independently():
    """
    The thumbnail was the first statement in a shared try, so an RGBA plate
    aborted the CDL, AMF and .json sidecars too — while reporting success.
    """
    src = _src("delivery/handler.py")
    assert "CDL export failed" in src
    assert "AMF export failed" in src
    assert "Thumbnail failed" in src


def test_progress_route_returns_400_on_missing_id():
    src = _src("nodes/monitor/viewer.py")
    idx = src.index('"error": "Missing ID"')
    assert "status=400" in src[idx:idx + 300], "progress route still answers 200 on error"


def test_deliver_route_registration_is_idempotent():
    src = _src("delivery/handler.py")
    assert "_register_deliver_route" in src
    assert "@PromptServer.instance.routes.post('/radiance/deliver')" not in src, \
        "the bare decorator is back; a double import will crash startup"


def test_localstorage_reads_are_guarded():
    src = _src("js/radiance_viewer.js")
    assert "static readJSON(" in src
    assert "JSON.parse(localStorage.getItem('radiance_presets')" not in src, \
        "an unguarded preset read is back"


def test_node_lifecycle_hooks_chain():
    """Assigning onRemoved/onSelected raw discarded other extensions' handlers."""
    src = _src("js/radiance_viewer.js")
    assert "_prevOnRemoved?.apply" in src
    assert "_prevOnSelected?.apply" in src


def test_curve_editor_has_a_destroy_and_is_called():
    src = _src("js/radiance_viewer.js")
    cls = src[src.index("class RadianceCurveEditor {"):][:3000]
    assert "destroy()" in cls, "RadianceCurveEditor still has no teardown"
    assert "_winMouseUp" in cls and "_winPanMove" in cls and "_winPanUp" in cls
    body = src[src.index("    destroy("):][:6000]
    assert "curveEditor?.destroy" in body or "curveEditor.destroy" in body
