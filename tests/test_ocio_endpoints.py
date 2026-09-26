"""`radiance_ocio.py` — the HTTP surface and the display/view bake.

`tests/test_ocio_manager.py` covers the manager: the colour-space bake that
used to return an exact identity, the cache key, the root guard as a function.
It stops short of the four routes registered on ComfyUI's PromptServer, which
is where a browser actually meets this module:

    GET  /radiance/ocio/config     what the HUD reads on open
    GET  /radiance/ocio/displays   the dropdown
    POST /radiance/ocio/load       an unauthenticated path parameter
    POST /radiance/ocio/bake       raw float32 LUT bytes straight into WebGL

and the *display/view* bake behind that last one — a different code path from
the colour-space bake (`DisplayViewTransform`, the `scene_linear` role as the
implicit source), and the one the viewer's dropdown actually calls.

Everything here runs the real thing:

  * the real `radiance_ocio.py`, loaded from its file. conftest.py installs a
    `HAS_OCIO = False` stub at `sys.modules["radiance.radiance_ocio"]`, so an
    ordinary import gets a MagicMock and proves nothing.
  * the real PyOpenColorIO. tests/test_node_smoke.py installs a MagicMock at
    `sys.modules["PyOpenColorIO"]`, and it is collected first, so
    `importorskip` here would hand back the mock — every OCIO call would fail
    as an AttributeError rather than skip, or worse, quietly answer.
  * the real `aiohttp.web`, so the status codes and bodies asserted below are
    the ones a browser receives. conftest installs an empty `aiohttp` stub with
    no `json_response` at all when ComfyUI is absent; the fixture swaps the real
    package in and asserts it got one, rather than passing because
    `json_response` was a no-op.
  * a real `PromptServer` stand-in that *records* what gets registered, instead
    of conftest's identity decorator which drops handlers on the floor.

The LUT assertions are on numbers: bytes off the response, compared against a
processor built independently from the same config, plus the value a colourist
checks first (18% grey). An identity LUT is the defect this module already
shipped once, so "a LUT came back" is never the assertion.
"""
import asyncio
import contextlib
import importlib
import importlib.util
import json
import logging
import os
import pathlib
import sys
import types

import numpy as np
import pytest


# ═══════════════════════════════════════════════════════════════════════════
#  The real PyOpenColorIO, borrowed past the suite's MagicMock
# ═══════════════════════════════════════════════════════════════════════════

def _is_real(mod):
    """A real extension module, not a stand-in."""
    return (mod is not None
            and getattr(mod, "__file__", None) is not None
            and hasattr(getattr(mod, "Config", None), "CreateFromBuiltinConfig"))


@contextlib.contextmanager
def _real_pyocio():
    """Hand back the real PyOpenColorIO for the duration of the block.

    Same reasoning as tests/test_ocio_manager.py: another test file installs a
    MagicMock under this name and is collected first, so the stub has to be
    lifted out by hand and put back afterwards.
    """
    saved = sys.modules.get("PyOpenColorIO")
    if _is_real(saved):
        yield saved
        return
    sys.modules.pop("PyOpenColorIO", None)
    real = None
    try:
        try:
            real = importlib.import_module("PyOpenColorIO")
        except ImportError:
            real = None
        yield real if _is_real(real) else None
    finally:
        if saved is not None:
            sys.modules["PyOpenColorIO"] = saved
        else:
            sys.modules.pop("PyOpenColorIO", None)


# The module under test, loaded from its file.
#
# The name it is registered under is deliberate. coverage.py resolves
# `--cov=radiance` to a *package name* and matches executed files by module
# name, so a module loaded by path under a bare name is traced as "not part of
# radiance" and reports 0% however hard it is exercised — which is exactly what
# happens to test_ocio_manager.py's `radiance_ocio_under_test` today. Loading
# the same file under a `radiance.`-prefixed name leaves conftest's
# `radiance.radiance_ocio` stub untouched for the rest of the suite and makes
# the work these tests do visible in the coverage report.
_REAL = pathlib.Path(__file__).resolve().parent.parent / "radiance_ocio.py"
with _real_pyocio() as _pyocio:
    if _pyocio is None:
        pytest.skip("PyOpenColorIO is not installed", allow_module_level=True)
    PyOCIO = _pyocio
    _spec = importlib.util.spec_from_file_location(
        "radiance.radiance_ocio_http_under_test", _REAL
    )
    ocio_module = importlib.util.module_from_spec(_spec)
    sys.modules["radiance.radiance_ocio_http_under_test"] = ocio_module
    _spec.loader.exec_module(ocio_module)

if not getattr(ocio_module, "HAS_OCIO", False):   # pragma: no cover - belt and braces
    pytest.skip("PyOpenColorIO is not usable here", allow_module_level=True)

OCIOConfigManager = ocio_module.OCIOConfigManager

DISPLAY = "sRGB - Display"
VIEW = "Un-tone-mapped"          # ACEScg -> sRGB display encoding, no tone map
TONEMAPPED_VIEW = "ACES 2.0 - SDR 100 nits (Rec.709)"
RAW_VIEW = "Raw"                 # identity by definition — the control


# ═══════════════════════════════════════════════════════════════════════════
#  The real aiohttp.web, borrowed past conftest's empty stub
# ═══════════════════════════════════════════════════════════════════════════

def _aiohttp_names():
    return [n for n in list(sys.modules) if n == "aiohttp" or n.startswith("aiohttp.")]


@pytest.fixture(scope="module")
def real_aiohttp_web():
    saved = {n: sys.modules[n] for n in _aiohttp_names()}
    for name in saved:
        del sys.modules[name]
    try:
        web = importlib.import_module("aiohttp.web")
    except ImportError as exc:  # pragma: no cover - environment problem
        for name in _aiohttp_names():
            del sys.modules[name]
        sys.modules.update(saved)
        pytest.fail(f"these tests need the real aiohttp: {exc}")

    assert callable(getattr(web, "json_response", None)), (
        "imported an aiohttp.web with no json_response — that is conftest's "
        "stub, and every status code below would be unobservable"
    )
    probe = web.json_response({"ok": True}, status=418)
    assert probe.status == 418 and json.loads(probe.body.decode()) == {"ok": True}, (
        "aiohttp.web.json_response is not behaving like a real response"
    )

    yield web

    for name in _aiohttp_names():
        del sys.modules[name]
    sys.modules.update(saved)


# ═══════════════════════════════════════════════════════════════════════════
#  A PromptServer that records what was registered
# ═══════════════════════════════════════════════════════════════════════════

class _RecordingRoutes:
    """conftest's fake routes object returns an identity decorator and forgets
    the handler. These tests need the handlers, and need to see duplicates."""

    def __init__(self):
        self.registered = []          # [(method, path, handler)] in order

    def _record(self, method, path):
        def deco(fn):
            self.registered.append((method, path, fn))
            return fn
        return deco

    def get(self, path):
        return self._record("GET", path)

    def post(self, path):
        return self._record("POST", path)


@contextlib.contextmanager
def _fake_prompt_server():
    """Install a `server` module whose PromptServer records route registration.

    Fixture-scoped and restored: conftest's own `server` stub is put back, so
    nothing leaks into the rest of the suite.
    """
    routes = _RecordingRoutes()
    instance = types.SimpleNamespace(routes=routes)
    module = types.ModuleType("server")
    module.PromptServer = types.SimpleNamespace(instance=instance)

    saved = sys.modules.get("server")
    sys.modules["server"] = module
    try:
        yield routes
    finally:
        if saved is not None:
            sys.modules["server"] = saved
        else:
            sys.modules.pop("server", None)


@contextlib.contextmanager
def _capture_logs(level=logging.WARNING):
    """caplog cannot see this module's logger: `radiance` sets propagate=False
    (core/logging.py), so records never reach the root handler pytest installs.
    Attach a handler where the records actually are."""
    records = []

    class _Sink(logging.Handler):
        def emit(self, record):
            records.append(record)

    logger = ocio_module.logger
    handler = _Sink(level=level)
    logger.addHandler(handler)
    previous = logger.level
    logger.setLevel(min(previous or level, level))
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)


# ═══════════════════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def config_path(tmp_path_factory):
    """A real OCIO config on disk, so load_config runs its normal path."""
    cfg = PyOCIO.Config.CreateFromBuiltinConfig("studio-config-latest")
    p = tmp_path_factory.mktemp("ocio_http") / "config.ocio"
    p.write_text(cfg.serialize(), encoding="utf-8")
    return str(p)


@pytest.fixture(scope="module")
def manager(config_path):
    m = OCIOConfigManager()
    assert m.load_config(config_path) is True
    return m


@pytest.fixture
def fresh_manager(config_path):
    m = OCIOConfigManager()
    assert m.load_config(config_path) is True
    return m


@pytest.fixture
def empty_manager(monkeypatch):
    """A manager with nothing loaded. Discovery is suppressed because a bare
    manager finds the bundled ACES config, which is the intended behaviour."""
    monkeypatch.setattr(ocio_module, "discover_ocio_config", lambda: None)
    m = OCIOConfigManager()
    assert m.is_loaded is False
    return m


@pytest.fixture(scope="module")
def routes(real_aiohttp_web):
    """The four handlers, as registered by register_ocio_routes() itself."""
    with _fake_prompt_server() as recorded:
        ocio_module.register_ocio_routes()
        assert recorded.registered, (
            "register_ocio_routes() registered nothing — the rest of this file "
            "would be testing an empty dict"
        )
        yield recorded


@pytest.fixture(scope="module")
def handler(routes):
    table = {(m, p): fn for m, p, fn in routes.registered}
    return lambda method, path: table[(method, path)]


@pytest.fixture
def serving(monkeypatch):
    """Point the module's singleton at a given manager for one test."""
    def _use(mgr):
        monkeypatch.setattr(ocio_module, "_ocio_manager", mgr)
        assert ocio_module.get_ocio_manager() is mgr
        return mgr
    return _use


class FakeRequest:
    """Only what the handlers touch: an awaitable JSON body."""

    def __init__(self, payload=None, raises=None):
        self._payload = payload
        self._raises = raises

    async def json(self):
        if self._raises is not None:
            raise self._raises
        return self._payload


def call(coro):
    return asyncio.run(coro)


def body(response):
    return json.loads(response.body.decode())


def lattice(n):
    """The cube a bake is built from: R fastest, then G, then B."""
    axis = np.arange(n, dtype=np.float32) / (n - 1)
    return np.stack([
        np.tile(axis, n * n),
        np.repeat(np.tile(axis, n), n),
        np.repeat(axis, n * n),
    ], axis=1).astype(np.float32)


def reference_display_view(config_path, display, view, src, n):
    """What OCIO computes for this display/view, built from the config file
    rather than from the manager, so the bake is checked against something
    other than itself."""
    cfg = PyOCIO.Config.CreateFromFile(config_path)
    dvt = PyOCIO.DisplayViewTransform()
    dvt.setSrc(src)
    dvt.setDisplay(display)
    dvt.setView(view)
    ref = lattice(n)
    cfg.getProcessor(dvt).getDefaultCPUProcessor().applyRGB(ref)
    return ref


def grey_index(n):
    """Index of the lattice point nearest 18% grey."""
    i = round(0.18 * (n - 1))
    return i + n * i + n * n * i


# ═══════════════════════════════════════════════════════════════════════════
#  Route registration
# ═══════════════════════════════════════════════════════════════════════════

def test_the_four_routes_are_registered_on_their_methods_and_paths(routes):
    """The paths are a contract with radiance_viewer.js. A GET registered where
    the frontend POSTs is a 405 at runtime and nothing else."""
    assert [(m, p) for m, p, _ in routes.registered] == [
        ("GET", "/radiance/ocio/config"),
        ("POST", "/radiance/ocio/load"),
        ("POST", "/radiance/ocio/bake"),
        ("GET", "/radiance/ocio/displays"),
    ]


def test_registering_twice_registers_the_routes_once(routes):
    """A duplicate import used to crash ComfyUI startup with 'method HEAD is
    already registered', which takes the whole server down, not just OCIO."""
    before = len(routes.registered)
    ocio_module.register_ocio_routes()
    assert len(routes.registered) == before, (
        "the second registration was not suppressed — aiohttp raises on a "
        "duplicate route and ComfyUI never finishes starting"
    )


@pytest.mark.parametrize("missing", ["server", "aiohttp"])
def test_nothing_is_registered_when_the_host_is_absent(real_aiohttp_web, missing):
    """Radiance imports outside ComfyUI (tests, tooling, `python -c`). The
    absent host must be a logged warning, not an exception during import."""
    with _fake_prompt_server() as recorded, _capture_logs() as records:
        saved = {n: sys.modules[n] for n in list(sys.modules)
                 if n == missing or n.startswith(missing + ".")}
        for name in saved:
            del sys.modules[name]
        # A None in sys.modules is what makes `import <missing>` raise
        # ImportError, which is the branch under test.
        sys.modules[missing] = None
        try:
            ocio_module.register_ocio_routes()
        finally:
            sys.modules.pop(missing, None)
            sys.modules.update(saved)

    assert recorded.registered == [], (
        f"routes were registered even though {missing} could not be imported"
    )
    assert any("server not available" in r.getMessage() for r in records), (
        "a missing host was swallowed silently: "
        f"{[r.getMessage() for r in records]}"
    )


# ═══════════════════════════════════════════════════════════════════════════
#  GET /radiance/ocio/config
# ═══════════════════════════════════════════════════════════════════════════

def test_config_endpoint_describes_the_loaded_config(handler, serving, manager,
                                                     config_path):
    serving(manager)
    response = call(handler("GET", "/radiance/ocio/config")(FakeRequest()))
    assert response.status == 200
    payload = body(response)
    assert payload["loaded"] is True
    assert payload["path"] == config_path
    assert DISPLAY in payload["displays"]
    assert payload["roles"]["scene_linear"] == "ACEScg"
    labels = [p["label"] for p in payload["display_view_pairs"]]
    assert f"{DISPLAY} / {VIEW}" in labels
    assert any(cs["name"] == "ACEScg" for cs in payload["scene_color_spaces"])
    assert payload["ocio_version"] == PyOCIO.__version__


def test_config_endpoint_says_so_when_nothing_is_loaded(handler, serving,
                                                        empty_manager):
    """The HUD renders this state; it needs the reason, not an empty list."""
    serving(empty_manager)
    response = call(handler("GET", "/radiance/ocio/config")(FakeRequest()))
    assert response.status == 200
    payload = body(response)
    assert payload["loaded"] is False
    assert payload["error"] == "No OCIO config loaded"
    assert payload["install_hint"] is None      # OCIO *is* installed here
    assert "display_view_pairs" not in payload


# ═══════════════════════════════════════════════════════════════════════════
#  GET /radiance/ocio/displays
# ═══════════════════════════════════════════════════════════════════════════

def test_displays_endpoint_lists_every_display_view_pair(handler, serving, manager):
    serving(manager)
    response = call(handler("GET", "/radiance/ocio/displays")(FakeRequest()))
    assert response.status == 200
    payload = body(response)
    assert payload["loaded"] is True
    assert payload["config_name"] == manager.config_name
    assert payload["pairs"] == manager.get_display_view_pairs()
    assert {"display": DISPLAY, "view": VIEW,
            "label": f"{DISPLAY} / {VIEW}"} in payload["pairs"]
    assert len(payload["pairs"]) >= len(manager.get_displays())


def test_displays_endpoint_returns_an_empty_dropdown_when_unloaded(
        handler, serving, empty_manager):
    serving(empty_manager)
    payload = body(call(handler("GET", "/radiance/ocio/displays")(FakeRequest())))
    assert payload == {"loaded": False, "config_name": "None", "pairs": []}


def test_the_displays_endpoint_does_not_query_a_config_that_is_not_loaded(
        handler, serving):
    """`loaded: false` has to be answered from the flag, not by asking the
    config anyway and hoping it comes back empty. OCIO's bindings raise on an
    unloaded config, and this route is what the HUD calls on every open — a
    throw here is an empty dropdown with no explanation in it."""
    class _Tripwire:
        is_loaded = False
        config_name = "None"

        def get_display_view_pairs(self):
            raise AssertionError(
                "the endpoint queried the config while nothing was loaded")

    serving(_Tripwire())
    payload = body(call(handler("GET", "/radiance/ocio/displays")(FakeRequest())))
    assert payload == {"loaded": False, "config_name": "None", "pairs": []}


# ═══════════════════════════════════════════════════════════════════════════
#  POST /radiance/ocio/load
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def allowed_root(monkeypatch, tmp_path):
    """A directory the load route is permitted to read configs from."""
    root = tmp_path / "configs"
    root.mkdir()
    monkeypatch.setenv("RADIANCE_OCIO_ROOTS", str(root))
    return root


def load(handler, payload):
    return call(handler("POST", "/radiance/ocio/load")(FakeRequest(payload)))


def test_load_without_a_path_is_rejected(handler):
    response = load(handler, {})
    assert response.status == 200
    assert body(response) == {"error": "Missing 'path' parameter", "status": "error"}


def test_load_refuses_a_path_outside_the_allowed_roots(handler, serving, manager,
                                                       allowed_root, tmp_path):
    """This route is unauthenticated. Handing it an arbitrary absolute path
    turns it into a parser pointed at any file on the host."""
    serving(manager)
    outside = tmp_path / "elsewhere" / "config.ocio"
    outside.parent.mkdir()
    outside.write_text("ocio_profile_version: 2", encoding="utf-8")

    with _capture_logs() as records:
        response = load(handler, {"path": str(outside)})

    assert response.status == 403
    payload = body(response)
    assert payload["status"] == "error"
    assert "outside the allowed OCIO directories" in payload["error"]
    assert any("Rejected a config load" in r.getMessage() for r in records)
    assert manager.config_path != str(outside), "the rejected config was loaded anyway"


def test_load_of_a_missing_file_inside_a_root_is_a_404(handler, allowed_root):
    missing = allowed_root / "not-here.ocio"
    response = load(handler, {"path": str(missing)})
    assert response.status == 404
    assert body(response)["error"] == f"File not found: {missing}"


def test_a_missing_file_outside_the_roots_is_403_not_404(handler, allowed_root,
                                                         tmp_path):
    """The precedence matters on its own. If existence is checked first, the
    route answers 404 for absent and 403 for present, and an unauthenticated
    caller can map the filesystem one path at a time."""
    absent = tmp_path / "outside" / "nope.ocio"
    present = tmp_path / "outside" / "yes.ocio"
    present.parent.mkdir()
    present.write_text("ocio_profile_version: 2", encoding="utf-8")

    assert load(handler, {"path": str(absent)}).status == 403
    assert load(handler, {"path": str(present)}).status == 403


def test_load_of_a_traversal_out_of_a_root_is_refused(handler, allowed_root,
                                                      tmp_path):
    secret = tmp_path / "secret.ocio"
    secret.write_text("ocio_profile_version: 2", encoding="utf-8")
    traversal = str(allowed_root / ".." / "secret.ocio")
    assert load(handler, {"path": traversal}).status == 403


def test_load_installs_the_config_and_reports_it(handler, serving, empty_manager,
                                                 allowed_root, config_path):
    """The success path end to end: the manager was empty, the route was given
    a real config, and afterwards the dropdown has entries in it."""
    mgr = serving(empty_manager)
    target = allowed_root / "config.ocio"
    target.write_text(pathlib.Path(config_path).read_text(encoding="utf-8"),
                      encoding="utf-8")

    response = load(handler, {"path": str(target)})

    assert response.status == 200
    payload = body(response)
    assert payload["status"] == "success"
    assert payload["config"]["loaded"] is True
    assert payload["config"]["path"] == str(target)
    assert DISPLAY in payload["config"]["displays"]
    assert mgr.is_loaded is True
    assert mgr.get_views(DISPLAY), "the loaded config serves no views"


def test_load_of_an_unparseable_file_reports_failure_and_leaves_nothing_loaded(
        handler, serving, fresh_manager, allowed_root):
    """A half-loaded manager would serve a dropdown built from the old config
    and bake through the new one. This one starts loaded, so the reset is
    visible rather than assumed."""
    junk = allowed_root / "junk.ocio"
    junk.write_text("this is not a config\n:::", encoding="utf-8")
    mgr = serving(fresh_manager)
    assert mgr.is_loaded is True

    response = load(handler, {"path": str(junk)})

    assert response.status == 200
    assert body(response) == {"error": "Failed to parse OCIO config",
                              "status": "error"}
    assert mgr.is_loaded is False
    assert mgr.config_path is None
    assert mgr.config_name == "None"


def test_load_with_an_unreadable_body_answers_with_the_error(handler):
    response = call(handler("POST", "/radiance/ocio/load")(
        FakeRequest(raises=ValueError("Expecting value: line 1 column 1"))))
    assert response.status == 200
    payload = body(response)
    assert payload["status"] == "error"
    assert "Expecting value" in payload["error"]


def test_the_comfyui_models_directory_is_an_allowed_root(monkeypatch, tmp_path):
    """`folder_paths` is ComfyUI's, and may be absent entirely — the guard has
    to work either way. Here it is present and its models dir is permitted."""
    models = tmp_path / "models"
    (models / "ocio").mkdir(parents=True)
    stub = types.ModuleType("folder_paths")
    stub.models_dir = str(models)
    saved = sys.modules.get("folder_paths")
    sys.modules["folder_paths"] = stub
    monkeypatch.delenv("RADIANCE_OCIO_ROOTS", raising=False)
    monkeypatch.delenv("OCIO", raising=False)
    try:
        assert ocio_module._is_inside_allowed_ocio_root(
            str(models / "ocio" / "config.ocio")) is True
        assert ocio_module._is_inside_allowed_ocio_root(
            str(tmp_path / "config.ocio")) is False
    finally:
        if saved is not None:
            sys.modules["folder_paths"] = saved
        else:
            sys.modules.pop("folder_paths", None)


def test_the_root_guard_survives_folder_paths_being_absent(monkeypatch, tmp_path):
    saved = sys.modules.get("folder_paths")
    sys.modules["folder_paths"] = None       # import raises, as outside ComfyUI
    monkeypatch.setenv("RADIANCE_OCIO_ROOTS", str(tmp_path))
    try:
        assert ocio_module._is_inside_allowed_ocio_root(
            str(tmp_path / "config.ocio")) is True
    finally:
        if saved is not None:
            sys.modules["folder_paths"] = saved
        else:
            sys.modules.pop("folder_paths", None)


# ═══════════════════════════════════════════════════════════════════════════
#  POST /radiance/ocio/bake — the bytes that become the viewer's LUT
# ═══════════════════════════════════════════════════════════════════════════

def bake(handler, payload):
    return call(handler("POST", "/radiance/ocio/bake")(FakeRequest(payload)))


def lut_of(response, size):
    """The response body read back the way radiance_viewer.js reads it."""
    assert response.status == 200
    assert response.content_type == "application/octet-stream"
    assert response.headers["X-Radiance-LUT-Size"] == str(size)
    assert response.headers["X-Radiance-LUT-Channels"] == "3"
    assert response.headers["X-Radiance-LUT-Dtype"] == "float32"
    assert len(response.body) == size ** 3 * 3 * 4, "wrong number of bytes for the cube"
    return np.frombuffer(response.body, dtype=np.float32).reshape(-1, 3)


def test_bake_returns_the_display_view_transform_ocio_computes(
        handler, serving, manager, config_path):
    """The whole point of the endpoint, checked against a processor built from
    the config file rather than against the manager that produced it."""
    serving(manager)
    n = 9
    lut = lut_of(bake(handler, {"display": DISPLAY, "view": VIEW, "size": n}), n)
    ref = reference_display_view(config_path, DISPLAY, VIEW, "ACEScg", n)
    assert np.array_equal(lut, ref), f"max deviation {np.abs(lut - ref).max()}"


def test_the_baked_display_view_lut_is_not_an_identity(handler, serving, manager):
    """The defect this module shipped once: a LUT bit-identical to its own
    input lattice, served under the name of the transform the user picked."""
    serving(manager)
    n = 9
    lut = lut_of(bake(handler, {"display": DISPLAY, "view": VIEW, "size": n}), n)
    assert not np.array_equal(lut, lattice(n)), (
        "the baked LUT is its own input lattice — the display transform is not "
        "being applied at all"
    )


def test_18_percent_grey_comes_back_where_the_display_puts_it(
        handler, serving, manager):
    """ACEScg 0.1875 through sRGB - Display / Un-tone-mapped is ~0.47. The
    broken bake returned 0.1875, which reads as a flat grade, not as a bug."""
    serving(manager)
    n = 33
    lut = lut_of(bake(handler, {"display": DISPLAY, "view": VIEW, "size": n}), n)
    assert lut[grey_index(n)][0] == pytest.approx(0.470, abs=0.01), lut[grey_index(n)]


def test_a_tone_mapped_view_differs_from_the_un_tone_mapped_one(
        handler, serving, manager):
    """Two views of the same display must not produce the same LUT — the whole
    dropdown is otherwise decorative."""
    serving(manager)
    n = 9
    plain = lut_of(bake(handler, {"display": DISPLAY, "view": VIEW, "size": n}), n)
    tonemapped = lut_of(
        bake(handler, {"display": DISPLAY, "view": TONEMAPPED_VIEW, "size": n}), n)
    assert not np.array_equal(plain, tonemapped)
    i = grey_index(n)
    assert tonemapped[i][0] < plain[i][0], (
        "the ACES tone map should pull 18% grey below the un-tone-mapped "
        f"encoding: {tonemapped[i][0]} vs {plain[i][0]}"
    )


def test_the_raw_view_is_the_only_one_that_comes_back_as_the_lattice(
        handler, serving, manager):
    """The control for every 'not an identity' assertion above: Raw *is* an
    identity, so a bake that returns the lattice for everything cannot hide
    behind it."""
    serving(manager)
    n = 9
    raw = lut_of(bake(handler, {"display": DISPLAY, "view": RAW_VIEW, "size": n}), n)
    assert np.array_equal(raw, lattice(n))


def test_the_baked_lattice_runs_r_fastest(handler, serving, manager):
    """The order the bytes arrive in is the order texImage3D reads them. Swap
    R and B and every LUT is a channel swap with a plausible histogram."""
    serving(manager)
    n = 9      # the endpoint clamps below 9, so this is the smallest real cube
    lut = lut_of(bake(handler, {"display": DISPLAY, "view": RAW_VIEW, "size": n}), n)
    assert lut[1][0] > lut[0][0], "R did not advance between the first two entries"
    assert lut[1][1] == lut[0][1], "G moved when only R should have"
    assert lut[1][2] == lut[0][2], "B moved when only R should have"
    assert lut[n][1] > lut[0][1], "G did not advance after n entries"
    assert lut[n * n][2] > lut[0][2], "B did not advance after n*n entries"

    b_fastest = lattice(n)[:, ::-1]
    assert not np.array_equal(lut, b_fastest), "the lattice is running B fastest"


def test_bake_accepts_a_colour_space_pair_as_well(handler, serving, manager,
                                                  config_path):
    serving(manager)
    n = 9
    lut = lut_of(bake(handler, {"src": "ACEScg", "dst": DISPLAY, "size": n}), n)
    cfg = PyOCIO.Config.CreateFromFile(config_path)
    ref = lattice(n)
    cfg.getProcessor("ACEScg", DISPLAY).getDefaultCPUProcessor().applyRGB(ref)
    assert np.array_equal(lut, ref)
    assert not np.array_equal(lut, lattice(n))


def test_an_explicit_input_space_changes_the_result(handler, serving, manager):
    """`input_space` is what makes the endpoint usable for footage that is not
    already in the scene_linear role."""
    serving(manager)
    n = 9
    default = lut_of(bake(handler, {"display": DISPLAY, "view": VIEW, "size": n}), n)
    explicit = lut_of(bake(handler, {"display": DISPLAY, "view": VIEW,
                                     "input_space": "ACEScct", "size": n}), n)
    assert not np.array_equal(default, explicit), (
        "input_space was ignored — the LUT is the same as the scene_linear one"
    )
    same = lut_of(bake(handler, {"display": DISPLAY, "view": VIEW,
                                 "input_space": "ACEScg", "size": n}), n)
    assert np.array_equal(default, same), (
        "the default input space is not the scene_linear role"
    )


@pytest.mark.parametrize("asked, expected", [(1, 9), (8, 9), (9, 9), (33, 33),
                                             (65, 65), (300, 65)])
def test_the_requested_lut_size_is_clamped_to_what_webgl_can_take(
        handler, serving, manager, asked, expected):
    """Unclamped, a caller can ask for a 4096³ cube and the server tries."""
    serving(manager)
    response = bake(handler, {"display": DISPLAY, "view": RAW_VIEW, "size": asked})
    lut = lut_of(response, expected)
    assert lut.shape == (expected ** 3, 3)


def test_bake_without_a_config_is_an_error_not_an_empty_lut(handler, serving,
                                                            empty_manager):
    serving(empty_manager)
    response = bake(handler, {"display": DISPLAY, "view": VIEW, "size": 9})
    assert response.status == 200
    assert body(response) == {"error": "No OCIO config loaded", "status": "error"}


def test_bake_needs_one_of_the_two_shapes_of_request(handler, serving, manager):
    serving(manager)
    for payload in ({}, {"display": DISPLAY}, {"src": "ACEScg"}, {"size": 33}):
        response = bake(handler, payload)
        assert response.status == 200
        assert body(response) == {
            "error": "Provide either {display, view} or {src, dst}",
            "status": "error",
        }


def test_an_unknown_display_fails_rather_than_returning_a_lattice(handler, serving,
                                                                  manager):
    """Silence here is the identity bug wearing a different hat: the viewer
    would load a LUT that does nothing under the name of a transform."""
    serving(manager)
    response = bake(handler, {"display": "NoSuchDisplay", "view": VIEW, "size": 9})
    assert response.status == 200
    assert body(response) == {"error": "LUT baking failed — check server logs",
                              "status": "error"}


def test_an_unknown_colour_space_pair_fails_too(handler, serving, manager):
    serving(manager)
    response = bake(handler, {"src": "NotASpace", "dst": DISPLAY, "size": 9})
    assert body(response)["error"] == "LUT baking failed — check server logs"


def test_a_nonsense_size_is_reported_rather_than_crashing_the_request(
        handler, serving, manager):
    serving(manager)
    response = bake(handler, {"display": DISPLAY, "view": VIEW, "size": "big"})
    assert response.status == 200
    payload = body(response)
    assert payload["status"] == "error"
    assert "invalid literal for int()" in payload["error"]


def test_the_second_identical_bake_is_served_from_the_cache(handler, serving,
                                                            fresh_manager):
    serving(fresh_manager)
    n = 9
    first = bake(handler, {"display": DISPLAY, "view": VIEW, "size": n})
    assert len(fresh_manager._lut_cache) == 1
    second = bake(handler, {"display": DISPLAY, "view": VIEW, "size": n})
    assert len(fresh_manager._lut_cache) == 1, "the same request was baked twice"
    assert first.body == second.body


# ═══════════════════════════════════════════════════════════════════════════
#  The display/view bake, at the manager
# ═══════════════════════════════════════════════════════════════════════════

def test_display_view_bake_defaults_to_the_scene_linear_role(manager):
    implicit = manager.bake_display_view_lut(DISPLAY, VIEW, None, 9)
    explicit = manager.bake_display_view_lut(DISPLAY, VIEW, "ACEScg", 9)
    assert implicit is not None
    assert np.array_equal(implicit, explicit)


def test_display_view_bake_and_colour_space_bake_agree_on_the_same_transform(
        manager):
    """'sRGB - Display / Un-tone-mapped' is the display encoding and nothing
    else, so it must equal the plain colour-space conversion into it."""
    dv = manager.bake_display_view_lut(DISPLAY, VIEW, "ACEScg", 9)
    cs = manager.bake_colorspace_lut("ACEScg", DISPLAY, 9)
    assert np.allclose(dv, cs, atol=1e-6), f"max deviation {np.abs(dv - cs).max()}"


def test_an_unknown_view_bakes_to_nothing(manager):
    assert manager.bake_display_view_lut(DISPLAY, "NoSuchView", "ACEScg", 9) is None


def test_the_lut_cache_is_bounded_and_evicts_the_oldest(fresh_manager):
    """32 entries of 65³ float32 is ~100MB; unbounded, the viewer's dropdown is
    a memory leak with a UI."""
    fresh_manager._max_cache_entries = 2
    views = ["Un-tone-mapped", "Video (colorimetric)", "Raw"]
    first = fresh_manager.bake_display_view_lut(DISPLAY, views[0], "ACEScg", 5)
    fresh_manager.bake_display_view_lut(DISPLAY, views[1], "ACEScg", 5)
    assert len(fresh_manager._lut_cache) == 2
    fresh_manager.bake_display_view_lut(DISPLAY, views[2], "ACEScg", 5)
    assert len(fresh_manager._lut_cache) == 2, "the cache grew past its limit"
    again = fresh_manager.bake_display_view_lut(DISPLAY, views[0], "ACEScg", 5)
    assert again is not first, "the oldest entry was not the one evicted"
    assert np.array_equal(again, first), "the re-bake disagrees with the original"


def test_a_baked_lut_is_float32_and_finite(manager):
    """texImage3D takes the buffer as-is: a float64 cube is twice the bytes and
    the wrong stride, and a NaN is a black pixel on screen."""
    lut = manager.bake_display_view_lut(DISPLAY, TONEMAPPED_VIEW, "ACEScg", 9)
    assert lut.dtype == np.float32
    assert lut.shape == (9 ** 3, 3)
    assert np.isfinite(lut).all()


# ═══════════════════════════════════════════════════════════════════════════
#  Config discovery
# ═══════════════════════════════════════════════════════════════════════════

def test_discovery_prefers_the_ocio_environment_variable(monkeypatch, config_path):
    monkeypatch.setenv("OCIO", config_path)
    assert ocio_module.discover_ocio_config() == os.path.abspath(config_path)


def test_discovery_looks_one_level_down_a_search_directory(monkeypatch, tmp_path,
                                                           config_path):
    """`~/ocio/aces_1.2/config.ocio` is how these are laid out in the wild."""
    nested = tmp_path / "aces_1.2"
    nested.mkdir()
    (nested / "config.ocio").write_text("x", encoding="utf-8")
    assert ocio_module._find_file(str(tmp_path), "config.ocio") == str(
        nested / "config.ocio")
    assert ocio_module._find_file(str(tmp_path), "nothing.ocio") is None
    assert ocio_module._find_file(str(tmp_path / "absent"), "config.ocio") is None


def test_discovery_falls_back_to_the_download_when_nothing_is_on_disk(monkeypatch):
    """The last resort is a network fetch; it must be reached, and its failure
    must be a None rather than an exception out of import."""
    monkeypatch.setattr(ocio_module, "_OCIO_SEARCH_PATHS", [lambda: None])
    calls = []

    def _fake_download():
        calls.append(1)
        return None

    monkeypatch.setattr(ocio_module, "_download_default_config", _fake_download)
    assert ocio_module.discover_ocio_config() is None
    assert calls == [1], "the download fallback was never reached"


# ═══════════════════════════════════════════════════════════════════════════
#  The CPU transform path
# ═══════════════════════════════════════════════════════════════════════════

def test_the_cpu_transform_declines_without_a_config(serving, empty_manager):
    serving(empty_manager)
    img = np.full((2, 2, 3), 0.18, dtype=np.float32)
    assert ocio_module.apply_ocio_transform(img, DISPLAY, VIEW) is None


def test_the_cpu_transform_keeps_the_shape_and_dtype(serving, manager):
    serving(manager)
    img = np.full((4, 3, 3), 0.18, dtype=np.float32)
    out = ocio_module.apply_ocio_transform(img, DISPLAY, VIEW)
    assert out is not None
    assert out.shape == (4, 3, 3)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()


def test_the_cpu_transform_returns_none_for_an_unknown_view(serving, manager):
    serving(manager)
    img = np.full((2, 2, 3), 0.18, dtype=np.float32)
    assert ocio_module.apply_ocio_transform(img, DISPLAY, "NoSuchView") is None


# Was xfail(strict): the defect it documented is fixed.
def test_the_cpu_transform_actually_applies_the_transform(serving, manager):
    """18% grey through sRGB - Display / Un-tone-mapped is ~0.47, on the GPU
    path and on this one. They are the same picture at different resolutions."""
    serving(manager)
    img = np.full((2, 2, 3), 0.18, dtype=np.float32)
    out = ocio_module.apply_ocio_transform(img, DISPLAY, VIEW)
    assert not np.allclose(out, img), "the CPU transform returned its input"
    assert out[0, 0, 0] == pytest.approx(0.46, abs=0.02), out[0, 0]


# ═══════════════════════════════════════════════════════════════════════════
#  Processors, failures, and the install that has no OCIO at all
# ═══════════════════════════════════════════════════════════════════════════

def test_get_processor_answers_for_a_real_pair_and_declines_for_a_bad_one(
        manager, empty_manager):
    processor = manager.get_processor("ACEScg", DISPLAY)
    assert processor is not None
    assert hasattr(processor, "getDefaultCPUProcessor")
    assert manager.get_processor("ACEScg", "NotASpace") is None
    assert empty_manager.get_processor("ACEScg", DISPLAY) is None


def test_a_failure_inside_the_lattice_evaluation_is_an_error_body_not_a_500(
        handler, serving, fresh_manager, monkeypatch):
    """OCIO raising mid-bake must reach the browser as the same JSON error as
    any other bake failure — a traceback out of the handler is a 500 with no
    body, and the viewer shows nothing at all."""
    class _Exploding:
        def applyRGB(self, _array):
            raise RuntimeError("OCIO exploded")

    class _Processor:
        def getDefaultCPUProcessor(self):
            return _Exploding()

    monkeypatch.setattr(fresh_manager, "_build_processor",
                        lambda *a, **k: _Processor())
    serving(fresh_manager)

    with _capture_logs(logging.ERROR) as records:
        assert fresh_manager.bake_display_view_lut(DISPLAY, VIEW, "ACEScg", 9) is None
    assert any("OCIO exploded" in r.getMessage() for r in records)
    assert fresh_manager._lut_cache == {}, "a failed bake was cached"

    response = bake(handler, {"display": DISPLAY, "view": VIEW, "size": 9})
    assert response.status == 200
    assert body(response)["error"] == "LUT baking failed — check server logs"


def test_the_colour_space_cache_is_bounded_too(fresh_manager):
    fresh_manager._max_cache_entries = 2
    first = fresh_manager.bake_colorspace_lut("ACEScg", DISPLAY, 5)
    fresh_manager.bake_colorspace_lut("ACEScg", "Display P3 - Display", 5)
    fresh_manager.bake_colorspace_lut("ACEScg", "ACEScct", 5)
    assert len(fresh_manager._lut_cache) == 2
    assert fresh_manager.bake_colorspace_lut("ACEScg", DISPLAY, 5) is not first


# ── the auto-download fallback ──────────────────────────────────────────────

@pytest.fixture
def download_target(monkeypatch, tmp_path):
    """Redirect the download's ACES/ directory out of the checkout.

    _download_default_config() writes next to radiance_ocio.py, which is the
    repository. Pointing the module's __file__ at a tmp dir keeps the test off
    the working tree while still running the real function.
    """
    monkeypatch.setattr(ocio_module, "__file__", str(tmp_path / "radiance_ocio.py"))
    return tmp_path / "ACES" / "config.ocio"


def test_the_default_config_is_set_up_without_the_network(monkeypatch, download_target):
    """3.5: nothing is downloaded. The fallback used to fetch a config from
    GitHub at import time, without download consent; OpenColorIO ships the
    ACES studio config built in, and ocio_setup writes and activates it."""
    import urllib.request

    def _boom(req):        # pragma: no cover - reaching it is the failure
        raise AssertionError("the OCIO fallback touched the network")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    from radiance.color import ocio_setup
    monkeypatch.setattr(ocio_setup, "ACES_DIR", str(download_target.parent))
    monkeypatch.setattr(ocio_setup, "STUDIO_FILE", str(download_target.parent / "studio-config.ocio"))
    monkeypatch.delenv("OCIO", raising=False)
    path = ocio_module._download_default_config()
    assert path == str(download_target.parent / "studio-config.ocio")
    assert "studio-config" in open(path, encoding="utf-8").read(4000)


def test_the_default_config_never_raises(monkeypatch, download_target):
    """This runs during `import radiance`; an exception here would take the
    whole node load down."""
    from radiance.color import ocio_setup

    def _fail(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(ocio_setup, "configure_ocio", _fail)
    with _capture_logs(logging.ERROR) as records:
        assert ocio_module._download_default_config() is None
    assert any("disk full" in r.getMessage() for r in records)


# ── PyOpenColorIO not installed at all ──────────────────────────────────────

@pytest.fixture(scope="module")
def no_ocio_module():
    """The same file, imported on a machine with no PyOpenColorIO.

    This is the common install: OCIO is an optional dependency, and every one
    of these paths runs on someone's box.
    """
    saved = sys.modules.get("PyOpenColorIO")
    sys.modules["PyOpenColorIO"] = None      # makes the import raise ImportError
    try:
        spec = importlib.util.spec_from_file_location(
            "radiance.radiance_ocio_no_pyocio", _REAL)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["radiance.radiance_ocio_no_pyocio"] = mod
        spec.loader.exec_module(mod)
    finally:
        if saved is not None:
            sys.modules["PyOpenColorIO"] = saved
        else:
            sys.modules.pop("PyOpenColorIO", None)
    assert mod.HAS_OCIO is False, "PyOpenColorIO was importable after all"
    assert mod.OCIO is None and mod.OCIO_VERSION is None
    yield mod
    sys.modules.pop("radiance.radiance_ocio_no_pyocio", None)


def test_without_pyopencolorio_the_manager_stays_empty_and_says_why(
        no_ocio_module, config_path):
    mgr = no_ocio_module.OCIOConfigManager()
    assert mgr.is_loaded is False
    with _capture_logs() as records:
        assert mgr.load_config(config_path) is False
    assert any("PyOpenColorIO not installed" in r.getMessage() for r in records)
    assert mgr.bake_display_view_lut(DISPLAY, VIEW, "ACEScg", 9) is None
    assert mgr.get_displays() == []
    assert mgr.get_display_view_pairs() == []
    assert mgr.get_roles() == {}


def test_without_pyopencolorio_the_endpoints_hand_the_hud_an_install_hint(
        no_ocio_module, real_aiohttp_web):
    """The HUD has a place to put this. It must be the hint, not a stack
    trace and not an empty dropdown with no explanation."""
    with _fake_prompt_server() as recorded:
        no_ocio_module.register_ocio_routes()
    table = {(m, p): fn for m, p, fn in recorded.registered}
    assert len(table) == 4

    payload = body(call(table[("GET", "/radiance/ocio/config")](FakeRequest())))
    assert payload["loaded"] is False
    assert payload["error"] == "PyOpenColorIO not installed"
    assert payload["install_hint"] == "pip install opencolorio"

    payload = body(call(table[("GET", "/radiance/ocio/displays")](FakeRequest())))
    assert payload == {"loaded": False, "config_name": "None", "pairs": []}

    payload = body(call(table[("POST", "/radiance/ocio/bake")](
        FakeRequest({"display": DISPLAY, "view": VIEW}))))
    assert payload == {"error": "No OCIO config loaded", "status": "error"}
