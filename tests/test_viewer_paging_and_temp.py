"""The viewer's disk and state defects: wasted writes, orphaned temps, lies.

Four separate faults are covered here, all in nodes/monitor.

1. Exposure bracketing was a function default of True with no entry in
   INPUT_TYPES, so it could not be turned off, and each bracket pass wrote a
   .rhdr, a full 32-bit .exr and a .rpick that nothing ever fetched: the JS
   bracket loader requests only imgData.filename. 500 frames at 1080p was
   ~67 GB into temp/, two thirds of it backing the string "Low / High cached".

2. The zdepth RHDR sidecar is stored under frame_meta["hdr_sidecar"] and never
   sets hdr_filename, and _viewer_purge_temp collected only ("filename",
   "hdr_filename", "pick_filename", "exr_filename"). Every execution with a
   zdepth input orphaned one .rhdr per frame, forever.

3. RadianceLiteViewer wrote one FULL RESOLUTION RGBA PNG per frame despite a
   docstring promising "compact temp PNG previews", and had no purge path at
   all. (Removed in 3.5.0: the Viewer's Simple mode replaces it.)

4. RadiancePreviewServer._ensure_server caught OSError, logged, and returned,
   and serve() then returned a working-looking URL for a server that never
   bound. Changing the port leaked the previous HTTPServer.

Plus the JS/Python protocol agreement for the proxy-provenance fields the
status-bar badge needs, which only a test that reads both sides can hold.
"""
import logging
import os
import pathlib
import tempfile

import pytest

torch = pytest.importorskip("torch")

# Source-level assertions run fine under conftest's torch stub; the ones that
# execute the node are marked real_torch.
RADIANCE_TORCH_GATED = True

_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _src(rel):
    return (_ROOT / rel).read_text(encoding="utf-8")


@pytest.fixture(autouse=True)
def _real_shared_modules():
    """
    Undo the stub modules other test files leave in sys.modules.

    tests/test_exr_metadata.py and tests/test_exr_multipass_workflow.py install
    placeholder radiance.path_utils / radiance.color_utils modules at
    collection time, whose safe_join returns None. They collect before this
    file does, so a node imported here would write every temp file to the path
    None. These tests execute the real nodes, so they need the real modules;
    the stubs are put back afterwards so nothing else sees a change.
    """
    import importlib
    import sys

    stubbed = {}
    for name in ("radiance.path_utils", "radiance.color_utils", "radiance.hdr.utils"):
        mod = sys.modules.get(name)
        if mod is not None and getattr(mod, "__file__", None) is None:
            stubbed[name] = mod
            del sys.modules[name]
            importlib.import_module(name)

    # A node module imported while a stub was in place holds the stub's
    # safe_join, which returns None. Rebind it in place; reloading the package
    # instead would rebuild the node classes the shared registry points at.
    from radiance.core.system.path_utils import safe_join as real_safe_join
    rebound = []
    for name in ("radiance.nodes.monitor.viewer",):
        mod = sys.modules.get(name)
        if mod is not None and getattr(mod, "safe_join", None) is not real_safe_join:
            rebound.append((mod, mod.safe_join))
            mod.safe_join = real_safe_join

    try:
        yield
    finally:
        for mod, previous in rebound:
            mod.safe_join = previous
        for name, mod in stubbed.items():
            sys.modules[name] = mod


@pytest.fixture
def temp_out(monkeypatch):
    """Point the nodes at a scratch temp directory.

    Other test files replace sys.modules["folder_paths"] wholesale, so the node
    modules can be holding a different module object than a fresh
    `import folder_paths` here returns. Patch the one they actually bound.
    """
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


def _listdir(d):
    return sorted(os.listdir(d))


# ── 1. Exposure bracketing ──────────────────────────────────────────────────

def test_exposure_bracketing_is_a_control_the_user_can_see():
    from radiance.nodes.monitor.viewer import RadianceViewer

    spec = RadianceViewer.INPUT_TYPES()
    assert "exposure_bracketing" in spec.get("optional", {}), \
        "exposure_bracketing is back to a hidden function default with no widget"
    kind, opts = spec["optional"]["exposure_bracketing"]
    assert kind == "BOOLEAN"
    assert opts["default"] is False, \
        "bracketing defaults on again — that is ~45 GB per 500-frame shot nobody asked for"
    assert opts.get("tooltip")


def test_the_frontend_no_longer_pins_bracketing_on():
    src = _src("js/radiance_viewer.js")
    at = src.index("const hiddenViewerDefaults = {")
    body = src[at:src.index("};", at)]
    assert "exposure_bracketing" not in body, \
        "the JS forces exposure_bracketing back on and hides the widget"


@pytest.mark.real_torch
def test_bracketing_is_off_by_default(temp_out):
    from radiance.nodes.monitor.viewer import RadianceViewer

    RadianceViewer().view(torch.rand(3, 16, 24, 3), unique_id="n1")
    assert not [f for f in _listdir(temp_out) if "bracket" in f], \
        "bracket files written with no one asking for them"


@pytest.mark.real_torch
def test_bracket_passes_write_only_the_png_the_viewer_fetches(temp_out):
    """The bracket .rhdr / .exr / .rpick were written and never read."""
    from radiance.nodes.monitor.viewer import RadianceViewer

    result = RadianceViewer().view(
        torch.rand(4, 16, 24, 3), exposure_bracketing=True, unique_id="n1")

    bracket_files = [f for f in _listdir(temp_out) if "bracket" in f]
    assert bracket_files, "bracketing was requested and produced nothing"
    for name in bracket_files:
        assert name.endswith(".png"), f"bracket pass still writes {name}"

    # And the payload still carries what the analysis shelf reads.
    labels = {e.get("bracket_label") for e in result["ui"]["radiance_images"]}
    assert {"low", "high"} <= labels
    for entry in result["ui"]["radiance_images"]:
        if entry.get("bracket_label"):
            assert entry.get("filename")
            assert not entry.get("hdr_filename")
            assert not entry.get("exr_filename")
            assert not entry.get("pick_filename")


@pytest.mark.real_torch
def test_bracketing_multiplies_the_write_volume_by_one_file_not_four(temp_out):
    """The whole point: a bracket frame costs one PNG, not a full frame set."""
    from radiance.nodes.monitor.viewer import RadianceViewer

    RadianceViewer().view(torch.rand(5, 16, 24, 3), exposure_bracketing=True, unique_id="n1")
    files = _listdir(temp_out)
    main = [f for f in files if "bracket" not in f]
    brackets = [f for f in files if "bracket" in f]
    # 5 frames x 4 artefacts for the main pass, 5 x 2 PNGs for the brackets.
    assert len(main) == 20, main
    assert len(brackets) == 10, brackets


# ── 2. The zdepth sidecar leak ──────────────────────────────────────────────

@pytest.mark.real_torch
def test_zdepth_sidecar_is_purged_on_the_next_execution(temp_out):
    """
    _process_zdepth_image stores its RHDR under "hdr_sidecar" only. That key
    was missing from the purge collection, so one .rhdr per frame leaked on
    every execution with a zdepth input, forever.
    """
    from radiance.nodes.monitor.viewer import RadianceViewer

    node = RadianceViewer()
    image = torch.rand(3, 16, 24, 3)
    depth = torch.rand(3, 16, 24, 1)

    node.view(image, zdepth=depth, unique_id="n7")
    first = {f for f in _listdir(temp_out) if f.startswith("◎ Radiance_zdepth")}
    assert any(f.endswith(".rhdr") for f in first), "no zdepth sidecar was written"

    node.view(image, zdepth=depth, unique_id="n7")
    survivors = first & set(_listdir(temp_out))
    assert not survivors, f"orphaned zdepth temp files from the previous run: {sorted(survivors)}"


def test_purge_collects_the_key_the_zdepth_branch_actually_uses():
    src = _src("nodes/monitor/viewer.py")
    at = src.index("_written: List[str] = []")
    body = src[at:at + 900]
    assert '"hdr_sidecar"' in body, \
        "the purge collection dropped hdr_sidecar again — zdepth leaks one .rhdr per frame"


# ── 3. Proxy provenance, and the protocol that carries it ───────────────────

@pytest.mark.real_torch
def test_frame_payload_declares_what_the_png_actually_is(temp_out):
    """
    The PNG fallback is 8-bit, capped at FALLBACK_MAX_DIM = 2048, and Reinhard
    tonemapped when d_max > 1.05. The viewer's badge cannot tell the colourist
    that without being told, and it used to read FP32 over it.
    """
    from radiance.nodes.monitor.viewer import RadianceViewer

    hdr_plate = torch.rand(1, 2400, 3000, 3) * 8.0
    out = RadianceViewer().view(hdr_plate, unique_id="n2")
    entry = out["ui"]["radiance_images"][0]

    assert entry["source_width"] == 3000 and entry["source_height"] == 2400
    assert max(entry["preview_width"], entry["preview_height"]) == 2048, \
        "the fallback PNG is capped at 2048 and the payload must say so"
    assert entry["preview_tonemapped"] is True, \
        "an HDR plate's proxy is tonemapped and the payload must say so"


@pytest.mark.real_torch
def test_an_sdr_plate_is_not_reported_as_tonemapped(temp_out):
    from radiance.nodes.monitor.viewer import RadianceViewer

    out = RadianceViewer().view(torch.rand(1, 32, 32, 3), unique_id="n3")
    assert out["ui"]["radiance_images"][0]["preview_tonemapped"] is False


def test_proxy_provenance_keys_agree_on_both_sides():
    """
    A payload key the node writes and the browser never reads, or the reverse,
    only shows up at runtime. Both sides are read here.
    """
    py = _src("nodes/monitor/viewer.py")
    js = _src("js/radiance_viewer.js")

    for key in ("source_width", "source_height", "preview_tonemapped"):
        assert f'"{key}"' in py, f"the node stopped emitting {key}"
        assert f"imgData.{key}" in js, f"the viewer stopped reading {key}"

    # ...and the badge is the consumer, so it has to actually use them.
    at = js.index("    _updateBitDepthBadge() {")
    badge = js[at:js.index("\n    }", at)]
    for prop in ("source_width", "source_height", "preview_tonemapped"):
        assert prop in badge, f"_updateBitDepthBadge ignores {prop}"


def test_the_badge_cannot_claim_fp32_over_the_proxy():
    """
    inputLabel was initialised to 'FP32' and the `else if (this.image)` branch
    set it to 'FP32' again, so the status bar read "FP32 · RGBA32F" while the
    viewer displayed an 8-bit tonemapped PNG.
    """
    js = _src("js/radiance_viewer.js")
    at = js.index("    _updateBitDepthBadge() {")
    badge = js[at:js.index("\n    }", at)]

    assert "PROXY 8-BIT" in badge, "the badge has no label for the 8-bit fallback"

    # The branch that runs when there is no float data must mark the proxy.
    fallback = badge[badge.index("} else if (this.image"):]
    assert "isProxy = true" in fallback.split("} else {")[0], \
        "the no-float-data branch does not mark the display as a proxy"
    assert "inputLabel = 'FP32'" not in fallback, \
        "the fallback branch still labels the 8-bit proxy FP32"

    # FP32 may only be set where a float source is actually loaded.
    float_branch = badge[badge.index("if (hdr && hdr.data)"):badge.index("} else if (this.image")]
    assert badge.count("inputLabel = 'FP32'") == float_branch.count("inputLabel = 'FP32'"), \
        "FP32 is set outside the float-source branch"

    # And the fallback reason is surfaced, not only console.warn'd.
    assert "_currentFallbackReason" in badge


def test_every_fallback_path_records_a_reason():
    js = _src("js/radiance_viewer.js")
    assert js.count("_noteHDRFallback(") >= 2, \
        "a path that drops to the 8-bit proxy is back to only warning the console"
    assert "payload.fallbackReason" in js


# ── 5. The preview server ───────────────────────────────────────────────────

class _FakeServer:
    instances = []

    def __init__(self, addr, handler):
        self.addr = addr
        self.shutdown_called = False
        self.closed = False
        _FakeServer.instances.append(self)

    def serve_forever(self):
        return None

    def shutdown(self):
        self.shutdown_called = True

    def server_close(self):
        self.closed = True


@pytest.fixture
def realtime_module(monkeypatch):
    from radiance.nodes.monitor import realtime as R
    monkeypatch.setattr(R, "_SERVERS", {}, raising=False)
    _FakeServer.instances = []
    return R


def test_a_failed_bind_does_not_hand_back_a_working_looking_url(realtime_module, monkeypatch):
    R = realtime_module

    def _refuse(addr, handler):
        raise OSError(98, "Address already in use")

    monkeypatch.setattr(R, "HTTPServer", _refuse)
    node = R.RadiancePreviewServer()
    _, url = node.serve(torch.zeros(1, 8, 8, 3), port=8765, stream_name="s")

    assert not url.startswith("http"), \
        "a port that could not be bound still reported a usable URL"
    assert "8765" in url and "ERROR" in url


def test_a_successful_bind_still_reports_the_url(realtime_module, monkeypatch):
    R = realtime_module
    monkeypatch.setattr(R, "HTTPServer", _FakeServer)
    node = R.RadiancePreviewServer()
    _, url = node.serve(torch.zeros(1, 8, 8, 3), port=8801, stream_name="s")
    assert url == "http://localhost:8801/"


def test_changing_the_port_stops_the_old_server(realtime_module, monkeypatch):
    """
    _SERVERS was never pruned and nothing called shutdown(), so every port edit
    leaked an HTTPServer, its thread and its listening socket for the life of
    the process.
    """
    R = realtime_module
    monkeypatch.setattr(R, "HTTPServer", _FakeServer)
    node = R.RadiancePreviewServer()

    node.serve(torch.zeros(1, 8, 8, 3), port=8811, stream_name="s")
    node.serve(torch.zeros(1, 8, 8, 3), port=8812, stream_name="s")

    assert list(R._SERVERS) == [8812], f"leaked servers: {sorted(R._SERVERS)}"
    first = _FakeServer.instances[0]
    assert first.shutdown_called and first.closed, "the server on the old port was leaked"


def test_re_running_on_the_same_port_reuses_the_server(realtime_module, monkeypatch):
    R = realtime_module
    monkeypatch.setattr(R, "HTTPServer", _FakeServer)
    node = R.RadiancePreviewServer()
    node.serve(torch.zeros(1, 8, 8, 3), port=8821, stream_name="s")
    node.serve(torch.zeros(1, 8, 8, 3), port=8821, stream_name="s")
    assert len(_FakeServer.instances) == 1, "the server was rebound on every execution"
    assert not _FakeServer.instances[0].shutdown_called


# ── 6. The dropped progress route ───────────────────────────────────────────

def test_a_dropped_route_is_reported(monkeypatch, caplog):
    """
    _radiance_route_once swallowed every exception and returned an identity
    decorator, so if PromptServer.instance was unavailable at import the
    /radiance/progress route vanished silently: the JS poll loop
    clearInterval'd on its first fetch error and the progress bar sat at 0%
    for the whole export with nothing saying why.
    """
    from radiance.nodes.monitor import viewer as V

    monkeypatch.setattr(V, "PromptServer", None, raising=False)
    with caplog.at_level(logging.WARNING, logger="radiance.viewer"):
        deco = V._radiance_route_once("get", "/radiance/progress")

    assert deco(lambda: None) is not None  # still non-fatal
    assert any("/radiance/progress" in r.getMessage() for r in caplog.records), \
        "the route was dropped without a word"


def test_a_registration_error_is_reported(monkeypatch, caplog):
    from radiance.nodes.monitor import viewer as V

    class _Boom:
        class instance:
            @property
            def routes(self):
                raise RuntimeError("no routes")

    monkeypatch.setattr(V, "PromptServer", _Boom, raising=False)
    with caplog.at_level(logging.ERROR, logger="radiance.viewer"):
        deco = V._radiance_route_once("get", "/radiance/progress")
    assert deco(lambda: None) is not None
    assert any("/radiance/progress" in r.getMessage() for r in caplog.records)


# ── 7. Per-instance state ───────────────────────────────────────────────────

def test_the_viewer_keeps_no_per_execution_state_on_the_instance():
    """
    ComfyUI reuses node instances across executions, so anything stashed on
    self survives into the next run. RadianceViewer defines no __init__ and
    assigns nothing to self; this pins that, because the stale-state hazard
    other nodes in this package hit is exactly one `self.x = ...` away.
    """
    import ast

    tree = ast.parse(_src("nodes/monitor/viewer.py"))
    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == "RadianceViewer")

    assigned = [
        t.attr
        for node in ast.walk(cls)
        if isinstance(node, ast.Assign)
        for t in node.targets
        if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self"
    ]
    assert not assigned, f"RadianceViewer stashes per-execution state on self: {assigned}"
    assert not any(isinstance(n, ast.FunctionDef) and n.name == "__init__" for n in cls.body)


@pytest.mark.real_torch
def test_two_executions_on_one_instance_do_not_bleed(temp_out):
    from radiance.nodes.monitor.viewer import RadianceViewer

    node = RadianceViewer()
    first = node.view(torch.rand(4, 16, 24, 3), unique_id="n9")
    second = node.view(torch.rand(2, 16, 24, 3), unique_id="n9")

    assert first["ui"]["batch_size"] == [4]
    assert second["ui"]["batch_size"] == [2]
    assert len(second["ui"]["radiance_images"]) == 2, \
        "the second execution carried frames over from the first"
