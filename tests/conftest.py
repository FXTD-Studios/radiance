"""
conftest.py — Radiance test suite shared fixtures and ComfyUI mocks.

ComfyUI is not installed in the test environment, so we stub the modules
that Radiance imports at the top of each nodes_*.py file.  The stubs expose
just enough surface to let the modules import cleanly; node logic that
actually calls into ComfyUI (e.g. comfy.sample.sample) is tested via
integration tests that mock the call-site, not here.
"""

import logging
import os
import sys
import types
import importlib
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import numpy as np

# Tests never touch the network. The pixel SDR-to-HDR checkpoint downloads by
# default on first use; this keeps the suite offline unless a test opts in.
os.environ.setdefault("RADIANCE_ALLOW_DOWNLOADS", "0")

RADIANCE_ROOT = Path(__file__).resolve().parent.parent
RADIANCE_PARENT = RADIANCE_ROOT.parent
if str(RADIANCE_PARENT) not in sys.path:
    sys.path.insert(0, str(RADIANCE_PARENT))

# Make `import radiance` resolve regardless of the checkout directory name.
# The package imports itself as `radiance`, which normally requires the repo
# directory to be named exactly "radiance". CI on a mirror/fork checked out as
# e.g. "radiance-beta" would otherwise fail every `import radiance` with
# ModuleNotFoundError → collection errors → exit code 2. Register the package
# from its __init__.py under the stable name so the dir name no longer matters.
if RADIANCE_ROOT.name != "radiance" and (RADIANCE_ROOT / "__init__.py").exists():
    # Install a lazy meta-path finder that maps the import name `radiance` to the
    # repo root. This makes the FIRST `import radiance` run __init__ through the
    # normal import machinery — exactly as it would if the checkout directory were
    # named "radiance" — instead of eagerly pre-importing here (which would cache
    # partial submodules under the test stubs and skew node-count tests).
    import importlib.abc as _iabc
    import importlib.util as _iutil

    class _RadianceNameFinder(_iabc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name != "radiance":
                return None
            return _iutil.spec_from_file_location(
                "radiance",
                str(RADIANCE_ROOT / "__init__.py"),
                submodule_search_locations=[str(RADIANCE_ROOT)],
            )

    if not any(type(f).__name__ == "_RadianceNameFinder" for f in sys.meta_path):
        sys.meta_path.insert(0, _RadianceNameFinder())


# ─────────────────────────────────────────────────────────────────────────────
#  Torch stub — installed BEFORE any radiance import so modules that do
#  `import torch` at the top level don't crash the collection phase.
#  Real torch tests are guarded with HAS_TORCH / pytest.skipif in each file.
# ─────────────────────────────────────────────────────────────────────────────

def _make_torch_stub():
    """
    Minimal torch stub that lets module-level `import torch` succeed.
    Actual tensor operations are NOT supported — those tests skip themselves.
    """
    try:
        import torch  # use the real thing if available
        return torch
    except ImportError:
        pass

    torch_mod = types.ModuleType("torch")

    # Basic sentinel class so isinstance checks don't crash
    class _FakeTensor:
        pass

    class _FakeModule:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return MagicMock()

        def eval(self):
            return self

        def to(self, *args, **kwargs):
            return self

        def state_dict(self):
            return {}

    class _FakeSequential(_FakeModule):
        def __init__(self, *layers):
            super().__init__()
            self.layers = layers

    class _FakeNN(types.ModuleType):
        Module = _FakeModule
        Sequential = _FakeSequential
        Conv2d = MagicMock
        ReLU = MagicMock
        Upsample = MagicMock
        Linear = MagicMock
        SiLU = MagicMock
        AdaptiveAvgPool2d = MagicMock
        functional = MagicMock()

    torch_mod.Tensor      = _FakeTensor
    torch_mod.Generator   = MagicMock
    torch_mod.nn          = _FakeNN("torch.nn")
    torch_mod.device      = MagicMock()
    torch_mod.zeros       = MagicMock()
    torch_mod.ones        = MagicMock()
    torch_mod.randn       = MagicMock()
    torch_mod.from_numpy  = MagicMock()
    torch_mod.no_grad     = MagicMock()
    torch_mod.float32     = "float32"
    torch_mod.float16     = "float16"
    torch_mod.bfloat16    = "bfloat16"
    torch_mod.cuda        = MagicMock()
    torch_mod.tensor      = MagicMock(return_value=MagicMock())
    torch_mod.inverse     = MagicMock(return_value=MagicMock())
    torch_mod.diag        = MagicMock(return_value=MagicMock())
    torch_mod.stack       = MagicMock(return_value=MagicMock())
    torch_mod.cat         = MagicMock(return_value=MagicMock())
    torch_mod.clamp       = MagicMock(return_value=MagicMock())
    torch_mod.where       = MagicMock(return_value=MagicMock())
    torch_mod.det         = MagicMock(return_value=MagicMock())
    torch_mod.float64     = "float64"
    torch_mod.eye        = MagicMock(return_value=MagicMock())
    torch_mod.long        = MagicMock(return_value=MagicMock())
    torch_mod.arange      = MagicMock(return_value=MagicMock())
    torch_mod.max         = MagicMock(return_value=MagicMock())
    torch_mod.min         = MagicMock(return_value=MagicMock())
    torch_mod.abs         = MagicMock(return_value=MagicMock())
    torch_mod.sqrt        = MagicMock(return_value=MagicMock())
    torch_mod.exp         = MagicMock(return_value=MagicMock())
    torch_mod.pow         = MagicMock(return_value=MagicMock())
    torch_mod.clamp_      = MagicMock(return_value=MagicMock())
    torch_mod.zeros_like  = MagicMock(return_value=MagicMock())
    torch_mod.ones_like   = MagicMock(return_value=MagicMock())
    torch_mod.full        = MagicMock(return_value=MagicMock())
    torch_mod.load        = MagicMock(return_value=MagicMock())
    torch_mod.save        = MagicMock()
    torch_mod.inference_mode = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

    # Mark the stub so tests can tell it apart from the real package.
    #
    # Without this there is no reliable signal: the stub is a real ModuleType
    # (so `isinstance(torch, MagicMock)` is False) and every attribute access
    # returns a MagicMock that answers `hasattr` for anything (so
    # `hasattr(torch.zeros(1), "shape")` is True too). Both idioms were in use
    # in this suite and both reported "real torch is available".
    torch_mod.__radiance_stub__ = True
    return torch_mod


_torch_stub = _make_torch_stub()
_torch_mod = sys.modules.setdefault("torch", _torch_stub)
if not hasattr(_torch_mod, "Generator"):
    _torch_mod.Generator = MagicMock
if not hasattr(_torch_mod, "nn"):
    _torch_mod.nn = getattr(_torch_stub, "nn", MagicMock())
sys.modules.setdefault("torch.nn", getattr(_torch_mod, "nn", MagicMock()))
# Ensure torch.nn.functional exists so `import torch.nn.functional as F` succeeds
if "torch.nn.functional" not in sys.modules:
    _nn_functional = types.ModuleType("torch.nn.functional")
    for _fn in ("grid_sample", "pad", "interpolate", "conv2d", "relu",
                "softmax", "normalize", "avg_pool2d", "max_pool2d"):
        setattr(_nn_functional, _fn, MagicMock())
    sys.modules["torch.nn.functional"] = _nn_functional

# Every module conftest fabricates carries this attribute. A test that needs to
# know "is this the host package or our stand-in?" reads the marker instead of
# inferring it from behaviour: a MagicMock answers hasattr for anything, and a
# types.ModuleType is not distinguishable from a real module by isinstance, so
# every behavioural guess in this suite has been wrong at least once. See the
# real-torch gate below, which is the same idea for torch.
STUB_MARKER = "__radiance_test_stub__"


def _mark_stub(module):
    """Tag a fabricated module so callers can check, not guess."""
    setattr(module, STUB_MARKER, True)
    return module


def is_test_stub(module) -> bool:
    """True when `module` is one conftest fabricated rather than the real thing."""
    return bool(getattr(module, STUB_MARKER, False))


if "node_helpers" not in sys.modules:
    _node_helpers = _mark_stub(types.ModuleType("node_helpers"))
    _node_helpers.conditioning_set_values = MagicMock(side_effect=lambda conditioning, values: conditioning)
    sys.modules["node_helpers"] = _node_helpers

if "aiohttp" not in sys.modules:
    _aiohttp = _mark_stub(types.ModuleType("aiohttp"))
    _aiohttp_web = _mark_stub(types.ModuleType("aiohttp.web"))
    _aiohttp.web = _aiohttp_web
    sys.modules["aiohttp"] = _aiohttp
    sys.modules["aiohttp.web"] = _aiohttp_web

if "server" not in sys.modules:
    _server = _mark_stub(types.ModuleType("server"))

    class _FakeRoutes:
        """A routes object that remembers what was registered on it.

        AUDIT-FIX (2026-09): `get`/`post` used to return a bare identity
        decorator and keep no state at all, so a test could drive a whole
        register_*_routes() call and then have nothing to assert against --
        registration tests either passed vacuously or had to ship their own
        recording double (test_ocio_endpoints.py::_RecordingRoutes documents
        exactly that). Recording here lets the shared fake answer the two
        questions those tests actually ask: which paths were registered, and
        were any of them registered twice.
        """

        def __init__(self):
            self.registered = []  # [(method, path, handler)] in registration order

        def _record(self, method, path):
            def deco(fn):
                self.registered.append((method, path, fn))
                return fn
            return deco

        def get(self, path):
            return self._record("GET", path)

        def post(self, path):
            return self._record("POST", path)

        # ── readers, so assertions do not have to unpack tuples themselves ──
        def paths(self, method=None):
            return [p for m, p, _ in self.registered if method in (None, m)]

        def handler(self, path, method=None):
            for m, p, fn in self.registered:
                if p == path and method in (None, m):
                    return fn
            return None

        def clear(self):
            self.registered.clear()

    class _FakePromptServer:
        instance = type("PromptServerInstance", (), {"routes": _FakeRoutes()})()
    _server.PromptServer = _FakePromptServer
    sys.modules["server"] = _server

if "comfy_api" not in sys.modules:
    # ALBABIT-FIX: the V3 node API (comfy_api.latest.io), for Radiance's V3 nodes.
    # A node class derives INPUT_TYPES, RETURN_* and FUNCTION from its schema in
    # the V1 form real ComfyUI gives it: lists, "COMBO" with an options list,
    # FUNCTION "EXECUTE_NORMALIZED" returning a NodeOutput.
    _io = _mark_stub(types.ModuleType("comfy_api.latest.io"))

    class _classproperty:
        def __init__(self, fget):
            self.fget = fget

        def __get__(self, obj, owner):
            return self.fget(owner)

    class _V3Input:
        io_type = None

        def __init__(self, id, display_name=None, optional=False, tooltip=None, **options):
            self.id, self.optional = id, optional
            self.options = {"tooltip": tooltip, **options}

        def as_v1(self):
            return (self.io_type, {k: v for k, v in self.options.items() if v is not None})

    class _V3StringInput(_V3Input):
        def __init__(self, id, multiline=False, placeholder=None, default=None,
                     dynamic_prompts=None, force_input=None, **kw):
            super().__init__(id, default=default, multiline=multiline, placeholder=placeholder,
                             dynamicPrompts=dynamic_prompts, forceInput=force_input, **kw)

    class _V3ComboInput(_V3Input):
        def __init__(self, id, options=(), default=None, **kw):
            super().__init__(id, default=default, multiselect=False, options=list(options), **kw)

    class _V3IntInput(_V3Input):
        def __init__(self, id, default=None, min=None, max=None, step=None, **kw):
            super().__init__(id, default=default, min=min, max=max, step=step, **kw)

    class _V3Output:
        io_type = None

        def __init__(self, id=None, display_name=None, tooltip=None, is_output_list=False):
            self.display_name, self.tooltip = display_name, tooltip

    def _v3_type(name, io_type, input_cls=_V3Input):
        return type(name, (), {
            "io_type": io_type,
            "Input": type("Input", (input_cls,), {"io_type": io_type}),
            "Output": type("Output", (_V3Output,), {"io_type": io_type}),
        })

    _io.Clip = _v3_type("Clip", "CLIP")
    _io.String = _v3_type("String", "STRING", _V3StringInput)
    _io.Combo = _v3_type("Combo", "COMBO", _V3ComboInput)
    _io.Int = _v3_type("Int", "INT", _V3IntInput)
    _io.Conditioning = _v3_type("Conditioning", "CONDITIONING")

    class _NodeOutput:
        def __init__(self, *args, ui=None, expand=None, block_execution=None):
            self.args, self.ui, self.expand = args, ui, expand

        @property
        def result(self):
            return self.args if self.args else None

        @classmethod
        def from_dict(cls, data):
            return cls(*data.get("result", ()), ui=data.get("ui"), expand=data.get("expand"))

        def __getitem__(self, index):
            return self.args[index]

    class _Schema:
        def __init__(self, node_id, display_name=None, category="sd", description="",
                     search_aliases=None, inputs=(), outputs=(), is_output_node=False,
                     is_deprecated=False, **_):
            self.node_id, self.display_name, self.category = node_id, display_name, category
            self.description, self.search_aliases = description, search_aliases
            self.inputs, self.outputs = list(inputs), list(outputs)
            self.is_output_node, self.is_deprecated = is_output_node, is_deprecated

    class _ComfyNode:
        FUNCTION = "EXECUTE_NORMALIZED"
        CATEGORY = _classproperty(lambda cls: cls.define_schema().category)
        DESCRIPTION = _classproperty(lambda cls: cls.define_schema().description)
        DEPRECATED = _classproperty(lambda cls: cls.define_schema().is_deprecated)
        OUTPUT_NODE = _classproperty(lambda cls: cls.define_schema().is_output_node)
        RETURN_TYPES = _classproperty(lambda cls: [o.io_type for o in cls.define_schema().outputs])
        RETURN_NAMES = _classproperty(
            lambda cls: [o.display_name or o.io_type for o in cls.define_schema().outputs])
        OUTPUT_TOOLTIPS = _classproperty(lambda cls: [o.tooltip for o in cls.define_schema().outputs])

        @classmethod
        def INPUT_TYPES(cls):
            out = {"required": {}, "optional": {}}
            for i in cls.define_schema().inputs:
                out["optional" if i.optional else "required"][i.id] = i.as_v1()
            return out

        @classmethod
        def EXECUTE_NORMALIZED(cls, *args, **kwargs):
            out = cls.execute(*args, **kwargs)
            if isinstance(out, dict):
                return _NodeOutput.from_dict(out)
            if isinstance(out, tuple):
                return _NodeOutput(*out)
            return out

    _io.ComfyNode, _io.Schema, _io.NodeOutput = _ComfyNode, _Schema, _NodeOutput
    _comfy_api = _mark_stub(types.ModuleType("comfy_api"))
    _comfy_api_latest = _mark_stub(types.ModuleType("comfy_api.latest"))
    _comfy_api.latest, _comfy_api_latest.io = _comfy_api_latest, _io
    sys.modules["comfy_api"] = _comfy_api
    sys.modules["comfy_api.latest"] = _comfy_api_latest
    sys.modules["comfy_api.latest.io"] = _io

# AUDIT-FIX (2026-09): this stub used to be installed unconditionally, which
# meant the real radiance_ocio.py (370 statements) was never imported by the
# suite even on the lane where PyOpenColorIO IS installed: 0% coverage on the
# module that owns every OCIO transform, and no test touching the real manager.
# Stub it only when OCIO is genuinely absent; when it is present the real module
# loads and the OCIO paths run for real.
try:
    import PyOpenColorIO as _PyOCIO  # noqa: F401

    HAS_REAL_OCIO = True
except ImportError:
    HAS_REAL_OCIO = False

if not HAS_REAL_OCIO:
    _radiance_ocio_stub = _mark_stub(types.ModuleType("radiance.radiance_ocio"))
    # AUDIT-FIX (2026-08): the manager mock must report is_loaded=False. A bare
    # MagicMock() is truthy for every attribute, so RadianceColorSpaceConvert's
    # _try_ocio() saw is_loaded=True, got a MagicMock "processor" whose applyRGB
    # was a no-op, and returned the INPUT UNCHANGED -- every colour-space test
    # through the node was silently validating an identity transform. With
    # is_loaded=False the nodes exercise their real analytical fallback in tests.
    _ocio_mgr_mock = MagicMock()
    _ocio_mgr_mock.is_loaded = False
    _radiance_ocio_stub.get_ocio_manager = MagicMock(return_value=_ocio_mgr_mock)
    _radiance_ocio_stub.HAS_OCIO = False
    sys.modules.setdefault("radiance.radiance_ocio", _radiance_ocio_stub)

# `radiance.nodes_hdr_colorspace` used to be stubbed here. That module was
# retired with the rest of the flat nodes_*.py layer, so the stub stood in for
# a file rather than for a dependency: nothing imports the name any more, and a
# stub for something that does not exist is a place for a stale expectation to
# hide.


# ─────────────────────────────────────────────────────────────────────────────
#  Real-torch gate
#
#  CI's lightweight matrix installs no torch, so the MagicMock stub above is
#  active. Every "self-skip" idiom in the suite was defeated by it:
#
#    * `pytest.importorskip("torch")` succeeds -- the stub IS importable.
#    * `isinstance(torch, MagicMock)` is False -- the stub is a ModuleType.
#    * `hasattr(torch.zeros(1), "shape")` is True -- MagicMock answers hasattr.
#
#  so torch-dependent tests ran against mocks and failed. The workaround was a
#  hand-maintained `--ignore=` list in .github/workflows/ci.yml, which went
#  stale the moment a new torch-using test file was added: CI had been red
#  since 2026-07-10 with 13 failures in two files nobody had added to the list.
#
#  The gate below replaces that list with something that cannot go stale:
#
#    1. `HAS_REAL_TORCH` is the single source of truth, keyed off the marker
#       the stub sets on itself.
#    2. Any test marked `@pytest.mark.real_torch` skips when the stub is active.
#    3. Any test module that imports torch at module scope -- unguarded
#       `import torch` / `from torch import ...`, or `importorskip("torch")` --
#       is skipped wholesale, automatically, with no list to maintain. A module
#       that gates itself per-test opts out by setting
#       `RADIANCE_TORCH_GATED = True` at module scope.
#
#  Nothing is skipped when real torch is present, so the test-full lane still
#  executes every one of these.
# ─────────────────────────────────────────────────────────────────────────────

HAS_REAL_TORCH = not getattr(sys.modules.get("torch"), "__radiance_stub__", False)

_TORCH_GATE_OPT_OUT = "RADIANCE_TORCH_GATED"
_torch_need_cache: dict = {}


def _module_needs_real_torch(path) -> bool:
    """True if this test file binds torch at module scope without a guard.

    A guarded import (`try: import torch / except ImportError:`) is left alone:
    the module has said it handles absence itself.
    """
    key = str(path)
    if key in _torch_need_cache:
        return _torch_need_cache[key]

    verdict = False
    try:
        import ast

        tree = ast.parse(Path(path).read_text(encoding="utf-8"))

        guarded = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                for child in ast.walk(node):
                    guarded.add(id(child))

        for node in tree.body:
            if id(node) in guarded:
                continue
            if isinstance(node, ast.Import):
                if any(a.name == "torch" or a.name.startswith("torch.")
                       for a in node.names):
                    verdict = True
            elif isinstance(node, ast.ImportFrom):
                if node.module and (node.module == "torch"
                                    or node.module.startswith("torch.")):
                    verdict = True
            elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                call = node.value
                name = getattr(call.func, "attr", getattr(call.func, "id", ""))
                if name == "importorskip" and call.args:
                    arg = call.args[0]
                    if isinstance(arg, ast.Constant) and str(arg.value).split(".")[0] == "torch":
                        verdict = True
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                call = node.value
                name = getattr(call.func, "attr", getattr(call.func, "id", ""))
                if name == "importorskip" and call.args:
                    arg = call.args[0]
                    if isinstance(arg, ast.Constant) and str(arg.value).split(".")[0] == "torch":
                        verdict = True
    except Exception:  # pragma: no cover - unparseable file fails elsewhere
        verdict = False

    _torch_need_cache[key] = verdict
    return verdict


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "real_torch: test needs the real torch package, not the conftest stub",
    )


def pytest_report_header(config):
    """Say up front which lane this is, and what it cannot test.

    A gate that skips quietly is only half a gate: the CI log for the
    lightweight lane showed `1483 passed` and nothing about the 1577 tests that
    never ran or why. These two lines appear above every run, so a lane that is
    short a dependency announces it before the first test rather than only in a
    -rs summary nobody passes.
    """
    missing = []
    try:
        from radiance.config.dependencies import (
            CORE_DEPENDENCIES,
            OPTIONAL_DEPENDENCIES,
            missing_dependencies,
        )

        missing = [
            spec.display_name
            for spec in missing_dependencies(
                tuple(CORE_DEPENDENCIES) + tuple(OPTIONAL_DEPENDENCIES)
            )
        ]
    except Exception:  # pragma: no cover - reported by the import tests
        missing = ["<could not query radiance.config.dependencies>"]

    return [
        "radiance lane: real torch={}, real OCIO={}".format(
            "yes" if HAS_REAL_TORCH else "NO (conftest MagicMock stub is active)",
            "yes" if HAS_REAL_OCIO else "NO (radiance.radiance_ocio is stubbed)",
        ),
        "radiance declared dependencies absent here: {}".format(
            ", ".join(missing) if missing else "none"
        ),
    ]


def pytest_collection_modifyitems(config, items):
    if HAS_REAL_TORCH or os.environ.get("RADIANCE_DISABLE_TORCH_GATE"):
        return

    marker_skip = pytest.mark.skip(
        reason="requires real torch (the conftest MagicMock stub is active)"
    )
    module_skip = pytest.mark.skip(
        reason="module imports torch at module scope; the conftest MagicMock "
               "stub is active, so it cannot run here (covered by the "
               "test-full CI lane)"
    )

    for item in items:
        if item.get_closest_marker("real_torch"):
            item.add_marker(marker_skip)
            continue
        module = getattr(item, "module", None)
        if module is not None and getattr(module, _TORCH_GATE_OPT_OUT, False):
            continue
        fspath = getattr(item, "fspath", None)
        if fspath is not None and _module_needs_real_torch(str(fspath)):
            item.add_marker(module_skip)


# ── radiance.image subpackage (used by nodes_qc via `from .image import defects`) ─
_image_pkg = _mark_stub(types.ModuleType("radiance.image"))
_defects_stub = _mark_stub(types.ModuleType("radiance.image.defects"))
_defects_stub.analyze_levels         = MagicMock(return_value={"crushed": 0.0, "clipped": 0.0})
_defects_stub.check_gamut            = MagicMock(return_value={"out_of_gamut_pct": 0.0})
_defects_stub.detect_banding         = MagicMock(return_value={"risk_pct": 0.0, "detected": False})
_defects_stub.analyze_noise          = MagicMock(return_value={"level": 0.0})
_defects_stub.detect_compression_artifacts = MagicMock(return_value={"detected": False})
_defects_stub.analyze_focus          = MagicMock(return_value={"sharpness": 1.0})
_image_pkg.defects = _defects_stub
# Keep the package importable as a real package. Without a __path__ this stub
# shadowed the whole subpackage, so `import radiance.image.upscale` failed even
# though the file exists — which is the reason delivery/handler.py reached for
# a node class instead of the library function it actually wanted. Only
# `defects` is stubbed; every other submodule loads from disk.
_image_pkg.__path__ = [str(Path(__file__).resolve().parent.parent / "image")]
sys.modules.setdefault("radiance.image",         _image_pkg)
sys.modules.setdefault("radiance.image.defects", _defects_stub)
# Also expose as bare `image` and `image.defects` for flat imports
_bare_image_pkg = sys.modules.get("image") or types.ModuleType("image")
_bare_image_pkg.defects = _defects_stub
# Same __path__ as the packaged alias above. Without it the two spellings
# disagree: `radiance.image.upscale` imports and the flat `image.upscale` does
# not, which is a difference between two names for one directory.
if not hasattr(_bare_image_pkg, "__path__"):
    _bare_image_pkg.__path__ = [str(Path(__file__).resolve().parent.parent / "image")]
sys.modules.setdefault("image",         _bare_image_pkg)
sys.modules.setdefault("image.defects", _defects_stub)


# ─────────────────────────────────────────────────────────────────────────────
#  ComfyUI stub modules
# ─────────────────────────────────────────────────────────────────────────────

def _make_comfy_stubs():
    """
    Build a minimal comfy.* module tree so Radiance nodes can be imported
    without a live ComfyUI installation.
    """
    # comfy (top-level)
    comfy = types.ModuleType("comfy")

    # comfy.samplers
    samplers = types.ModuleType("comfy.samplers")
    samplers.KSampler = MagicMock()
    samplers.SAMPLER_NAMES = [
        "euler", "euler_ancestral", "dpm_fast", "dpm_adaptive",
        "dpmpp_2s_ancestral", "dpmpp_sde", "dpmpp_2m", "dpmpp_3m_sde",
        "ddim", "uni_pc",
    ]
    samplers.SCHEDULER_NAMES = [
        "normal", "karras", "exponential", "sgm_uniform",
        "simple", "ddim_uniform", "beta",
    ]

    def _calculate_sigmas(model_sampling, scheduler, steps):
        """Monotonically decreasing sigmas, the only property callers rely on.

        test_sampler_regression.py used to install its own `comfy` stub with
        this function and then import the sampler by bare name, which gave it a
        private module that captured that stub. Now that the sampler is a
        package module, whichever stub was installed first wins and the test's
        own is overwritten — so the shared stub has to carry it.
        """
        try:
            import torch as _torch
            if isinstance(getattr(_torch, "__version__", None), str):
                return _torch.linspace(1.0, 0.0, int(steps) + 1)
        except ImportError:
            pass
        import numpy as _np
        return _np.linspace(1.0, 0.0, int(steps) + 1)

    samplers.calculate_sigmas = _calculate_sigmas
    comfy.samplers = samplers

    # comfy.sample
    sample = types.ModuleType("comfy.sample")
    sample.sample = MagicMock(return_value=MagicMock())
    comfy.sample = sample

    # comfy.model_management
    mm = types.ModuleType("comfy.model_management")
    mm.get_torch_device = MagicMock(return_value="cpu")
    mm.load_model_gpu = MagicMock()
    mm.soft_empty_cache = MagicMock()
    mm.unload_all_models = MagicMock()
    mm.get_free_memory = MagicMock(return_value=8 * 1024**3)
    mm.get_total_memory = MagicMock(return_value=16 * 1024**3)
    comfy.model_management = mm

    # comfy.utils
    utils = types.ModuleType("comfy.utils")
    utils.ProgressBar = MagicMock()
    comfy.utils = utils

    # comfy.model_base
    model_base = types.ModuleType("comfy.model_base")
    ModelType = types.SimpleNamespace(
        FLOW=1, FLOW_LTX_AV=2, EPS=3, V_PREDICTION=4,
    )
    model_base.ModelType = ModelType
    comfy.model_base = model_base

    # comfy.sd
    sd_mod = types.ModuleType("comfy.sd")
    sd_mod.VAE = MagicMock()
    comfy.sd = sd_mod

    # comfy.latent_formats
    latent_formats = types.ModuleType("comfy.latent_formats")
    latent_formats.SD15 = MagicMock()
    latent_formats.SDXL = MagicMock()
    comfy.latent_formats = latent_formats

    # comfy.nested_tensor  (optional — some builds don't have it)
    nested_tensor = types.ModuleType("comfy.nested_tensor")
    nested_tensor.NestedTensor = None
    comfy.nested_tensor = nested_tensor

    # comfy.cldm.control_types
    cldm = types.ModuleType("comfy.cldm")
    control_types = types.ModuleType("comfy.cldm.control_types")
    control_types.UNION_CONTROLNET_TYPES = {}
    cldm.control_types = control_types
    comfy.cldm = cldm

    # folder_paths stub
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_filename_list = MagicMock(return_value=[])
    folder_paths.get_full_path = MagicMock(return_value=None)
    folder_paths.get_input_directory  = MagicMock(return_value="/tmp")
    folder_paths.get_output_directory = MagicMock(return_value="/tmp/comfy_output")
    folder_paths.filter_files_content_types = MagicMock(return_value=[])
    folder_paths.get_annotated_filepath = MagicMock(side_effect=lambda name: name)
    folder_paths.exists_annotated_filepath = MagicMock(return_value=True)
    folder_paths.models_dir = "/tmp/comfy_models"
    folder_paths.output_directory = "/tmp/comfy_output"

    stubs = {
        "comfy": comfy,
        "comfy.samplers": samplers,
        "comfy.sample": sample,
        "comfy.model_management": mm,
        "comfy.utils": utils,
        "comfy.model_base": model_base,
        "comfy.sd": sd_mod,
        "comfy.latent_formats": latent_formats,
        "comfy.nested_tensor": nested_tensor,
        "comfy.cldm": cldm,
        "comfy.cldm.control_types": control_types,
        "folder_paths": folder_paths,
    }
    for _mod in stubs.values():
        _mark_stub(_mod)
    return stubs


# Install stubs before any radiance import can happen
_STUBS = _make_comfy_stubs()
for name, mod in _STUBS.items():
    if name not in sys.modules:
        sys.modules[name] = mod


# ─────────────────────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def np_rng():
    """Deterministic NumPy RNG for all tests."""
    return np.random.default_rng(seed=42)


@pytest.fixture(scope="session")
def linear_gradient_1d():
    """Linear ramp [0, 2] with 1024 samples — covers SDR and HDR range."""
    return np.linspace(0.0, 2.0, 1024, dtype=np.float32)


@pytest.fixture(scope="session")
def linear_image_hwc(np_rng):
    """(64, 64, 3) float32 scene-linear image with values in [0, 4]."""
    img = np_rng.random((64, 64, 3), dtype=np.float32) * 4.0
    return img.astype(np.float32)


@pytest.fixture
def caplog(caplog):
    """Capture Radiance's non-propagating logger without changing host logging."""
    logger = logging.getLogger("radiance")
    logger.addHandler(caplog.handler)
    try:
        yield caplog
    finally:
        logger.removeHandler(caplog.handler)
