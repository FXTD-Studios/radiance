"""The Viewer node's side of a real session: temp files, names, disk, cache.

From the October 2026 viewer review:

H15  The temp purge was keyed by node id alone and ran before the new frames
     were written. A second workflow with a viewer at the same id wiped the
     first one's frames, and a re-run deleted frames the open viewer had not
     loaded yet, which then 404'd.
M13  Every file was named with its own uuid, so the frontend had to save the
     whole result list (about 1.4 KB a frame) in the workflow to find them
     again. One token per run lets it save the token and the count.
M14  A .rpick per frame that nothing reads; no warning before a run that
     will not fit on the temp disk; a delivery cache that copied a plate
     even when it was larger than the whole cache budget.
"""
import os
import pathlib
import tempfile

import pytest

torch = pytest.importorskip("torch")

RADIANCE_TORCH_GATED = True

_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _src(rel):
    return (_ROOT / rel).read_text(encoding="utf-8")


def _temp_helpers():
    """The temp bookkeeping functions, run from the source (no torch needed)."""
    import threading
    import types

    import typing

    src = _src("nodes/monitor/viewer.py")
    start = src.index("def _viewer_graph_id(")
    end = src.index("class RadianceViewer:")
    ns = {"Dict": dict, "List": list, "Any": typing.Any, "Optional": typing.Optional, "os": os,
          "threading": threading, "logger": types.SimpleNamespace(debug=lambda *a, **k: None)}
    exec(src[start:end], ns)
    return ns


# ── H15: purge by graph and node, after writing ─────────────────────────────

def test_tracking_returns_only_the_previous_files_the_new_run_does_not_reuse():
    ns = _temp_helpers()
    d = tempfile.mkdtemp()
    old = [os.path.join(d, f"a{i}") for i in range(3)]
    new = [old[2], os.path.join(d, "b0")]
    assert ns["_viewer_track_temp"]("g:5", old) == []
    assert ns["_viewer_track_temp"]("g:5", new) == old[:2]
    for p in old[:2]:
        open(p, "w").close()
    assert ns["_viewer_unlink"](old[:2]) == 2
    assert ns["_viewer_unlink"](old[:2]) == 0          # already gone is fine


def test_the_graph_is_the_workflow_id():
    ns = _temp_helpers()
    graph = ns["_viewer_graph_id"]
    assert graph({"5": {}}, {"workflow": {"id": "wf-A"}}) == "wf-A"
    assert graph(None, None) == ""
    assert graph({}, {"workflow": "not a dict"}) == ""


@pytest.fixture(autouse=True)
def _real_shared_modules():
    """Undo stub path modules other test files install (see test_viewer_paging_and_temp)."""
    import importlib
    import sys

    stubbed = {}
    for name in ("radiance.path_utils", "radiance.color_utils", "radiance.hdr.utils"):
        mod = sys.modules.get(name)
        if mod is not None and getattr(mod, "__file__", None) is None:
            stubbed[name] = mod
            del sys.modules[name]
            importlib.import_module(name)
    from radiance.core.system.path_utils import safe_join as real_safe_join
    rebound = []
    mod = sys.modules.get("radiance.nodes.monitor.viewer")
    if mod is not None and getattr(mod, "safe_join", None) is not real_safe_join:
        rebound.append((mod, mod.safe_join))
        mod.safe_join = real_safe_join
    try:
        yield
    finally:
        for m, previous in rebound:
            m.safe_join = previous
        for name, m in stubbed.items():
            sys.modules[name] = m


@pytest.fixture
def temp_out(monkeypatch):
    import folder_paths
    from radiance.nodes.monitor import viewer as _viewer

    d = tempfile.mkdtemp()
    seen = set()
    for mod in (folder_paths, _viewer.folder_paths):
        if id(mod) in seen:
            continue
        seen.add(id(mod))
        monkeypatch.setattr(mod, "get_temp_directory", lambda: d, raising=False)
    return d


def _names(result):
    return {e["filename"] for e in result["ui"]["radiance_images"]}


@pytest.mark.real_torch
def test_two_workflows_with_the_same_node_id_keep_their_own_frames(temp_out):
    from radiance.nodes.monitor.viewer import RadianceViewer

    image = torch.rand(2, 16, 24, 3)
    a = RadianceViewer().view(image, unique_id="5", extra_pnginfo={"workflow": {"id": "wf-A"}})
    RadianceViewer().view(image, unique_id="5", extra_pnginfo={"workflow": {"id": "wf-B"}})
    left = set(os.listdir(temp_out))
    assert _names(a) <= left, "running workflow B deleted workflow A's frames"


@pytest.mark.real_torch
def test_a_rerun_writes_its_frames_before_the_old_ones_go(temp_out, monkeypatch):
    from radiance.nodes.monitor import viewer as _viewer

    seen_at_purge = []
    unlink = _viewer._viewer_unlink

    def spy(stale, key=""):
        seen_at_purge.append(set(os.listdir(temp_out)))
        return unlink(stale, key)

    monkeypatch.setattr(_viewer, "_viewer_unlink", spy)
    node = _viewer.RadianceViewer()
    first = node.view(torch.rand(2, 16, 24, 3), unique_id="7", extra_pnginfo={"workflow": {"id": "wf"}})
    second = node.view(torch.rand(2, 16, 24, 3), unique_id="7", extra_pnginfo={"workflow": {"id": "wf"}})
    assert _names(second) <= seen_at_purge[-1], "the old frames were purged before the new ones existed"
    after = set(os.listdir(temp_out))
    assert not (_names(first) & after), "the previous run's frames were never purged"
    assert _names(second) <= after


# ── M13 / M14: one token per run, no .rpick, a disk warning ────────────────

@pytest.mark.real_torch
def test_one_token_names_every_file_of_a_run_and_no_rpick_is_written(temp_out):
    from radiance.nodes.monitor.viewer import RadianceViewer

    result = RadianceViewer().view(torch.rand(3, 16, 24, 3), compare_image=torch.rand(3, 16, 24, 3),
                                   zdepth=torch.rand(3, 16, 24, 1), unique_id="3")
    token = result["ui"]["file_token"][0]
    assert len(token) == 12
    files = os.listdir(temp_out)
    assert not [f for f in files if f.endswith(".rpick")], "a .rpick nobody reads is still written"
    for entry in result["ui"]["radiance_images"]:
        assert "pick_filename" not in entry
        assert f"_{token}_" in entry["filename"], entry["filename"]
    main = [e for e in result["ui"]["radiance_images"]
            if not e.get("is_compare") and not e.get("is_zdepth")]
    for i, e in enumerate(main):
        assert e["filename"] == f"Radiance_viewer_{token}_{i}_thumb.png"
        assert e["hdr_sidecar"] == f"Radiance_viewer_{token}_{i}.rhdr"


@pytest.mark.real_torch
def test_brackets_are_named_by_their_timeline_frame(temp_out):
    from radiance.nodes.monitor.viewer import RadianceViewer

    result = RadianceViewer().view(torch.rand(3, 16, 24, 3), exposure_bracketing=True, unique_id="4")
    lows = sorted(e["filename"] for e in result["ui"]["radiance_images"] if e.get("bracket_label") == "low")
    assert len(set(lows)) == 3, f"bracket files of different frames share a name: {lows}"


@pytest.mark.real_torch
def test_a_run_that_will_not_fit_warns_and_still_writes(temp_out, monkeypatch):
    import collections
    import shutil

    from radiance.nodes.monitor.viewer import RadianceViewer

    usage = collections.namedtuple("usage", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda p: usage(10_000, 9_000, 1_000))
    result = RadianceViewer().view(torch.rand(2, 16, 24, 3), unique_id="6")
    warnings = result["ui"].get("warnings") or []
    assert warnings and "temp disk" in warnings[0], warnings
    assert len(result["ui"]["radiance_images"]) == 2, "the warning stopped the run"


def test_the_frontend_shows_the_warning():
    src = _src("js/radiance_viewer.js")
    at = src.index("nodeType.prototype.onExecuted = function")
    assert "message.warnings" in src[at:at + 4000]


# ── M14: the delivery cache does not copy a plate over its whole budget ────

def _load_cache_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("_rad_cache_session", _ROOT / "cache.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.real_torch
def test_an_oversize_plate_is_cached_without_a_copy():
    m = _load_cache_module()
    m._VIEWER_CACHE_MAX_BYTES = 1024
    plate = torch.zeros(1, 64, 64, 3)
    m._viewer_cache_set("big", plate)
    assert m._viewer_cache_get("big").data_ptr() == plate.data_ptr(), \
        "a plate larger than the whole budget was copied"
    m._VIEWER_CACHE_MAX_BYTES = 1 << 30
    small = torch.zeros(1, 4, 4, 3)
    m._viewer_cache_set("small", small)
    assert m._viewer_cache_get("small").data_ptr() != small.data_ptr(), "small plates are still copied"
