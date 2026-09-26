"""Functional smoke test: actually CALL every node.

`test_node_smoke.py` checks that each node *declares* itself correctly —
INPUT_TYPES parses, RETURN_TYPES exists, FUNCTION names a real method, the class
instantiates. It never calls anything. A node can pass all of that and raise on
the first line of its FUNCTION.

This module closes that gap. For every key in NODE_CLASS_MAPPINGS it:

  1. builds an argument set from the node's own INPUT_TYPES — real tensors for
     IMAGE/MASK/LATENT, declared defaults for scalars, the first option for
     combos;
  2. calls FUNCTION;
  3. asserts the return is a tuple whose length matches RETURN_TYPES, and that
     IMAGE/MASK outputs have the shape and dtype ComfyUI requires;
  4. asserts the input tensors were not mutated in place — ComfyUI hands the
     same tensor to every consumer of a link, so an in-place write corrupts
     other branches of the graph.

A node that needs something this harness cannot synthesise — a MODEL, a VAE, a
file on disk, a GPU — SKIPS with that reason stated. A skip is a known gap; a
pass is a node that demonstrably runs. Neither is silent.
"""
from __future__ import annotations

import copy
import pathlib
import sys
import types

import pytest

torch = pytest.importorskip("torch")

RADIANCE_TORCH_GATED = True

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_ROOT.parent))


# ── Types this harness cannot fabricate ─────────────────────────────────────
#
# Anything requiring loaded weights, a ComfyUI runtime object, or another
# node's bespoke payload. Listing them explicitly means a NEW opaque type shows
# up as a skip with a name rather than as a mystery error.
_OPAQUE = {
    "MODEL", "VAE", "CONDITIONING", "CONTROL_NET", "STYLE_MODEL",
    "CLIP_VISION", "CLIP_VISION_OUTPUT", "GLIGEN", "UPSCALE_MODEL", "SIGMAS",
    "SAMPLER", "GUIDER", "NOISE", "PHOTOMAKER", "AUDIO", "SAM_MODEL",
    "BBOX_DETECTOR", "SEGM_DETECTOR", "RADIANCE_GRADE_INFO", "RADIANCE_CDL",
    "RADIANCE_AOV", "RADIANCE_PASSES", "RADIANCE_LORA_STACK",
    "RADIANCE_MODEL_META", "RADIANCE_SHOT", "RADIANCE_PROJECT",
}

_H = _W = 64
_B = 1


def _image():
    return torch.rand(_B, _H, _W, 3, dtype=torch.float32)


def _mask():
    return torch.rand(_B, _H, _W, dtype=torch.float32)


def _latent():
    return {"samples": torch.randn(_B, 4, _H // 8, _W // 8, dtype=torch.float32)}


class _StubClip:
    """A text encoder with SDXL's tokenizer layout (CLIP-G + CLIP-L): one token
    per word in 77-token chunks, zero conditioning of the real shapes. Enough
    for the prompt nodes to run their whole path; a node that needs a real
    embedding shows up as a failure, not a skip."""

    def tokenize(self, text):
        ids = [(1000 + i, 1.0) for i, _ in enumerate(str(text).split())]
        chunks = []
        while True:
            body, ids = ids[:75], ids[75:]
            chunks.append([(49406, 1.0)] + body + [(49407, 1.0)] * (76 - len(body)))
            if not ids:
                break
        return {"g": [list(c) for c in chunks], "l": [list(c) for c in chunks]}

    def encode_from_tokens(self, tokens, return_pooled=False):
        n = len(tokens["l"])
        cond, pooled = torch.zeros(1, 77 * n, 2048), torch.zeros(1, 1280)
        return (cond, pooled) if return_pooled else cond

    def encode_from_tokens_scheduled(self, tokens):
        cond, pooled = self.encode_from_tokens(tokens, return_pooled=True)
        return [[cond, {"pooled_output": pooled}]]


def _synthesise(spec):
    """Build one argument from a single INPUT_TYPES entry.

    Returns (value, None) or (None, reason-it-cannot-be-built).
    """
    if not isinstance(spec, (list, tuple)) or not spec:
        return None, "malformed spec"

    kind, config = spec[0], (spec[1] if len(spec) > 1 else {})
    if not isinstance(config, dict):
        config = {}

    # A combo is declared as a list of options in the type slot.
    if isinstance(kind, (list, tuple)):
        options = [o for o in kind]
        if not options:
            return None, "empty combo"
        default = config.get("default")
        return (default if default in options else options[0]), None

    if kind == "IMAGE":
        return _image(), None
    if kind == "MASK":
        return _mask(), None
    if kind == "LATENT":
        return _latent(), None
    if kind == "CLIP":
        return _StubClip(), None
    if kind == "INT":
        return int(config.get("default", 1)), None
    if kind == "FLOAT":
        return float(config.get("default", 0.5)), None
    if kind == "BOOLEAN":
        return bool(config.get("default", False)), None
    if kind == "STRING":
        # `forceInput` means ComfyUI draws no widget at all: the value can only
        # arrive over a link, so there is no default to stand in for and "" is
        # not a value any real graph would produce. Treat it like an opaque
        # type rather than feeding empty text into a parser — that was reported
        # as a node failure (RadianceHDRPerChannelDenorm on an empty
        # stats_json) when it is really an un-runnable input.
        if config.get("forceInput"):
            return None, "needs a STRING from an upstream node"
        # An empty field with a placeholder is one the user is meant to fill
        # in (a prompt): run it the way a user would, with the placeholder's
        # words typed in.
        if not config.get("default") and config.get("placeholder"):
            return str(config["placeholder"]), None
        return str(config.get("default", "")), None
    if kind in _OPAQUE:
        return None, f"needs a {kind}"
    return None, f"unknown input type {kind}"


def _build_args(cls):
    spec = cls.INPUT_TYPES()
    kwargs, skips = {}, []
    for section in ("required", "optional"):
        for name, entry in (spec.get(section) or {}).items():
            value, why = _synthesise(entry)
            if why is not None:
                if section == "required":
                    skips.append(f"{name}: {why}")
                continue
            kwargs[name] = value
    return kwargs, skips


def _snapshot(kwargs):
    """Deep copy of every tensor argument, for the mutation check."""
    out = {}
    for k, v in kwargs.items():
        if torch.is_tensor(v):
            out[k] = v.detach().clone()
        elif isinstance(v, dict) and torch.is_tensor(v.get("samples")):
            out[k] = v["samples"].detach().clone()
    return out


def _assert_unmutated(before, kwargs):
    for k, original in before.items():
        now = kwargs[k]
        if isinstance(now, dict):
            now = now["samples"]
        assert torch.equal(original, now), (
            f"the node mutated its {k!r} input in place. ComfyUI passes the same "
            "tensor object to every consumer of a link, so this corrupts other "
            "branches of the graph."
        )


# ── The reasons a call is allowed not to run ────────────────────────────────
#
# AUDIT-FIX (2026-09): this used to be a list of substrings matched against
# str(exc), and it swallowed real bugs. Free text cannot separate "this machine
# lacks a model file" from "this node has an argument-contract bug", and no
# wording change makes it able to. Four fabricated bug messages, four skips:
#
#   ValueError("mask requires the same batch size as image")  -> "requires"
#   KeyError("colour space 'ACEScg' not found in the LUT table") -> "not found"
#   RuntimeError("tile size does not exist for this resolution") -> "does not exist"
#   ValueError("peak_luminance is required for HDR output")   -> "is required for"
#
# Nothing below reads an exception message. A call is excused only when the
# environment is CHECKED to be short of something, by one of two routes:
#
#   1. _absence_signal() -- the exception, or a link in its cause chain, IS a
#      structural absence: RadianceDependencyError (the package's own signal),
#      an ImportError for a module that genuinely will not import, a
#      FileNotFoundError, or an AttributeError against one of conftest's stub
#      modules. A node computing a wrong tile size raises none of these.
#   2. _ENVIRONMENT_GATED -- an explicit, per-node predicate that interrogates
#      this environment and returns a reason, or None to let the failure stand.
#      Four entries, each one a named function you can read.
#
# Anything else fails the test, whatever it says.

try:
    from radiance.core.errors import RadianceDependencyError as _DependencyError
except Exception:  # pragma: no cover - import failure is reported elsewhere
    class _DependencyError(Exception):
        pass

try:
    from radiance.config.dependencies import module_available as _module_available
except Exception:  # pragma: no cover - import failure is reported elsewhere
    def _module_available(name):
        import importlib.util

        if name in sys.modules:
            return True
        try:
            return importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            return False


# conftest.py tags every module it fabricates with this attribute so callers can
# check rather than guess; see conftest.STUB_MARKER.
_STUB_MARKER = "__radiance_test_stub__"


def _cause_chain(exc):
    """The exception and everything it was raised from, outermost first.

    `__cause__` (an explicit `raise ... from ...`) is followed first, and
    `__context__` only where Python itself would print it. A node that wraps a
    real absence keeps that absence reachable; a node with a bug does not
    acquire one.
    """
    seen, chain, cur = set(), [], exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        chain.append(cur)
        if cur.__cause__ is not None:
            cur = cur.__cause__
        elif not cur.__suppress_context__:
            cur = cur.__context__
        else:
            cur = None
    return chain


class _NoNetwork(OSError):
    """Raised only by the _no_network fixture below.

    A node that reaches for the network is reaching for a model file that is
    not on this machine, which is a gap. The harness raises its OWN type for it
    so the classifier recognises the situation structurally: a node cannot
    produce this exception by wording an error like the fixture's.
    """


def _absence_signal(exc):
    """Reason this exception structurally means "absent", or None."""
    for link in _cause_chain(exc):
        if isinstance(link, _NoNetwork):
            return "the node tried to download a model; no network in tests"

        if isinstance(link, _DependencyError):
            return ("the node raised RadianceDependencyError, the package's own "
                    "missing-dependency signal")

        if isinstance(link, ImportError):
            # ImportError comes from the import machinery, not from argument
            # validation. Where it names the module (`.name`), confirm the
            # module really is unimportable -- a fabricated
            # ModuleNotFoundError("No module named 'torch'") on a machine that
            # has torch is a bug, not a gap.
            name = getattr(link, "name", None)
            if name and _module_available(name):
                continue
            return (f"an import failed for {name or 'an optional dependency'}, "
                    "which is not importable in this environment")

        if isinstance(link, FileNotFoundError):
            # The type exists to say "this path is not here". The harness
            # fabricates no real files, so it cannot supply one.
            return "a file the node needs is not present in this environment"

        if isinstance(link, AttributeError):
            # Python 3.10+ records the object the lookup failed on. When that
            # object is one of conftest's stand-ins, the node reached for a
            # piece of the host runtime that is not installed here.
            obj = getattr(link, "obj", None)
            if isinstance(obj, types.ModuleType) and getattr(obj, _STUB_MARKER, False):
                return (f"the node needs the real {obj.__name__}; conftest's "
                        "stand-in does not implement it")
    return None


def _declared_input(cls, section, arg):
    """The (type, config) a node declares for one input, or (None, {})."""
    try:
        entry = (cls.INPUT_TYPES().get(section) or {}).get(arg)
    except Exception:  # pragma: no cover - INPUT_TYPES is covered by test_node_smoke
        return None, {}
    if not isinstance(entry, (list, tuple)) or not entry:
        return None, {}
    config = entry[1] if len(entry) > 1 else {}
    return entry[0], (config if isinstance(config, dict) else {})


def _unfabricable_path(arg):
    """Gate: a required STRING path input the harness had no file to fill.

    Checked twice over: the node declares `arg` as a required STRING with no
    usable default, AND the value this call actually passed was empty. A real
    path would close the gate, so the day the harness can supply one the node
    is executed and any failure is reported as one.
    """

    def predicate(cls, kwargs):
        kind, config = _declared_input(cls, "required", arg)
        if kind != "STRING" or str(config.get("default", "")) != "":
            return None
        if str(kwargs.get(arg, "")) != "":
            return None
        return f"{arg} is a required file path and this environment has no file to name"

    return predicate


def _unfabricable_opaque(arg):
    """Gate: an optional input of a type this harness cannot build at all.

    _build_args only records a skip for un-fabricable REQUIRED inputs, so a node
    that declares e.g. an optional VAE and then needs it is called without one
    and fails. Checked against _OPAQUE, the same list the required path uses,
    and against the call actually made.
    """

    def predicate(cls, kwargs):
        kind, _ = _declared_input(cls, "optional", arg)
        if kind not in _OPAQUE or kwargs.get(arg) is not None:
            return None
        return f"needs the optional {arg} ({kind}), which the harness cannot build"

    return predicate


def _no_pixel_checkpoint(cls, kwargs):
    """Gate: no SDR→HDR pixel checkpoint is resolvable on this machine.

    Asked of the same resolver the node uses, so on a machine with
    sdr2hdr_pixel_image.pt installed the gate closes and Recover runs for real.
    """
    try:
        from radiance.pixel_sdr2hdr import resolve_pixel_checkpoint
    except Exception:
        return "radiance.pixel_sdr2hdr is not importable in this environment"
    if resolve_pixel_checkpoint(str(kwargs.get("pixel_checkpoint", ""))) is None:
        return "no SDR→HDR pixel checkpoint is installed in models/radiance"
    return None


def _ocio_config_lacks_colorspaces(cls, kwargs):
    """Gate: the active OCIO config does not define the names this call used.

    Asked of the config itself, through the same resolver the node uses. On a
    machine whose config DOES define them the gate closes and the transform is
    executed for real.
    """
    try:
        from radiance.hdr.ocio import _iter_colorspaces, _resolve_config
    except Exception:
        return "radiance.hdr.ocio is not importable in this environment"

    config = _resolve_config(str(kwargs.get("ocio_config_path", "")))
    if config is None:
        return "no OCIO config is resolvable in this environment"

    known = {name for name, _ in _iter_colorspaces(config)}
    wanted = [
        str(kwargs.get(arg, ""))
        for arg in ("source_colorspace", "target_colorspace")
    ]
    missing = [name for name in wanted if name and name not in known]
    if missing:
        return f"the active OCIO config defines none of {missing}"
    return None


def _missing_file(arg):
    """Gate: a path input whose value (a placeholder default) names no file.

    Like _unfabricable_path, for inputs that ship a placeholder default such
    as "/path/to/audio.wav" instead of an empty string. The node now raises on
    it (3.5); before, it returned an empty result that looked like a real one.
    """

    def predicate(cls, kwargs):
        import os
        value = str(kwargs.get(arg, ""))
        if value and os.path.isfile(value):
            return None
        return f"{arg}={value!r} names no file in this environment"

    return predicate


def _retired(cls, exc):
    """A hidden node kept only so saved graphs open, raising by design.

    SAM Loader / Generator (no SAM runtime ships) and Multipass Extract
    (image-filter passes, replaced by Multipass Estimate). The node must be
    DEPRECATED and its error must say what to use instead, so a real crash in
    a hidden node is still reported.
    """
    if not getattr(cls, "DEPRECATED", False) or not isinstance(exc, RuntimeError):
        return None
    msg = str(exc)
    if "SAM" in cls.__name__ or "Multipass Estimate" in msg:
        return "retired node, raises by design with its replacement named"
    return None


def _no_estimate_models(cls, kwargs):
    """Gate: Multipass Estimate needs MoGe-2 and both Marigold IID models (~4.9 GB)."""
    from radiance.nodes.vfx.multipass import estimate_models as em
    missing = [n for n, ok in (
        ("MoGe-2", em.find_moge()),
        ("Marigold appearance", em.find_marigold(em.MARIGOLD_APPEARANCE)),
        ("Marigold lighting", em.find_marigold(em.MARIGOLD_LIGHTING)),
    ) if not ok]
    if missing:
        return f"model weights not installed here: {', '.join(missing)} (fake-model tests in test_multipass_estimate.py)"
    return None


# The nodes that genuinely cannot run here, each with the check that says so.
# Short on purpose: every entry costs a node's worth of execution coverage, so
# a new one has to earn its place, and the predicate has to be something a
# reader can verify.
_ENVIRONMENT_GATED = {
    # Needs a .safetensors LoRA on disk.
    "RadianceHDRLoRALoader": _unfabricable_path("lora_path"),
    # Needs a multi-layer EXR rendered by a 3D package; nothing here makes one.
    "RadianceMultipassAOVReader": _unfabricable_path("exr_path"),
    # Needs a real clip or sequence; an empty path now raises (3.5) instead of
    # returning an 8x8 black shot.
    "RadianceDigitalCinemaRead": _unfabricable_path("source_path"),
    # Runs the RUDRA pixel model and, by design, raises rather than falling
    # back when no checkpoint is installed.
    "RadianceSDRToHDRRecover": _no_pixel_checkpoint,
    # Placeholder default path; a missing file raises (3.5) instead of "[]".
    "RadianceAudioCut": _missing_file("audio_filepath"),
    # Four-plus GB of weights; the maths and the pass bundle are tested with
    # fake models in test_multipass_estimate.py.
    "RadianceMultipassEstimate": _no_estimate_models,
    # Ships defaults that the ACES CG config bundled with the package does not
    # define, so the harness cannot name a colorspace the config knows.
    "RadianceHDROCIOTransform": _ocio_config_lacks_colorspaces,
}


def _environment_gap(key, cls, kwargs, exc):
    """Why this call could not run here, or None if the failure is the node's."""
    reason = _absence_signal(exc) or _retired(cls, exc)
    if reason:
        return reason
    gate = _ENVIRONMENT_GATED.get(key)
    if gate is not None:
        return gate(cls, kwargs)
    return None


def _all_nodes():
    try:
        import radiance
    except Exception as exc:  # pragma: no cover - covered by test_node_smoke
        pytest.skip(f"radiance did not import: {exc}")
    return radiance.NODE_CLASS_MAPPINGS


try:
    _NODES = _all_nodes()
except Exception:  # pragma: no cover
    _NODES = {}

_KEYS = sorted(_NODES)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """A test must never download a model.

    RadianceUpscaleImage fetched a 67 MB Real-ESRGAN checkpoint from the
    internet on its first functional run — silently, with no consent prompt, on
    what a user would experience as "I dropped a node and pressed queue". The
    behaviour is worth knowing about; a test suite that depends on it is not.
    """
    import socket
    import urllib.request

    def _blocked(*args, **kwargs):
        raise _NoNetwork("network access is blocked in tests; this node needs a "
                         "model file that is not present")

    monkeypatch.setattr(urllib.request, "urlopen", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    try:
        import requests
        monkeypatch.setattr(requests, "get", _blocked)
    except ImportError:
        pass


@pytest.mark.real_torch
@pytest.mark.parametrize("key", _KEYS)
def test_node_executes(key):
    cls = _NODES[key]
    kwargs, skips = _build_args(cls)
    if skips:
        pytest.skip("; ".join(skips))

    node = cls()
    fn = getattr(node, cls.FUNCTION)
    before = _snapshot(kwargs)

    try:
        result = fn(**copy.copy(kwargs))
    except Exception as exc:
        reason = _environment_gap(key, cls, kwargs, exc)
        if reason:
            pytest.skip(f"{reason} ({type(exc).__name__}: {str(exc)[:120]})")
        raise

    _assert_unmutated(before, kwargs)

    declared = getattr(cls, "RETURN_TYPES", ())
    if not declared:
        # OUTPUT_NODE with no graph outputs; a dict or tuple is both fine.
        assert result is None or isinstance(result, (tuple, dict, list)), (
            f"{key} declares no RETURN_TYPES but returned {type(result).__name__}"
        )
        return

    if isinstance(result, dict):
        assert "result" in result, (
            f"{key} returned a ui dict without a 'result' key, so nothing "
            "reaches its output sockets"
        )
        result = result["result"]

    assert isinstance(result, tuple), (
        f"{key}.{cls.FUNCTION} returned {type(result).__name__}, not a tuple. "
        "ComfyUI unpacks the return positionally against RETURN_TYPES."
    )
    assert len(result) == len(declared), (
        f"{key} declares {len(declared)} outputs {declared} but returned "
        f"{len(result)}"
    )

    for value, kind in zip(result, declared):
        if kind == "IMAGE" and value is not None:
            assert torch.is_tensor(value), f"{key} IMAGE output is not a tensor"
            assert value.ndim == 4 and value.shape[-1] in (1, 3, 4), (
                f"{key} IMAGE output has shape {tuple(value.shape)}; ComfyUI "
                "requires (B, H, W, C)"
            )
            assert value.dtype in (torch.float32, torch.float16, torch.bfloat16), (
                f"{key} IMAGE output dtype is {value.dtype}, not float"
            )
            assert torch.isfinite(value).all(), f"{key} IMAGE output contains NaN or inf"
        elif kind == "MASK" and value is not None:
            assert torch.is_tensor(value), f"{key} MASK output is not a tensor"
            assert value.ndim in (2, 3), (
                f"{key} MASK output has shape {tuple(value.shape)}; ComfyUI "
                "requires (B, H, W) or (H, W)"
            )
            assert torch.isfinite(value).all(), f"{key} MASK output contains NaN or inf"
        elif kind == "LATENT" and value is not None:
            assert isinstance(value, dict) and "samples" in value, (
                f"{key} LATENT output is not a dict with a 'samples' key"
            )


# ── The classifier is itself under test ─────────────────────────────────────
#
# These are the four messages that defeated the old substring list, plus the
# shapes that must still be excused. If anyone reintroduces text matching, the
# first four of these go red.

_BUG_MESSAGES = [
    ValueError("mask requires the same batch size as image"),
    KeyError("colour space 'ACEScg' not found in the LUT table"),
    RuntimeError("tile size does not exist for this resolution"),
    ValueError("peak_luminance is required for HDR output"),
    RuntimeError("PyOpenColorIO returned a null processor"),
    OSError("network access is blocked in tests"),
    ImportError("No module named 'torch'", name="torch"),
]


@pytest.mark.parametrize("exc", _BUG_MESSAGES, ids=lambda e: type(e).__name__)
def test_a_bug_is_not_an_environment_gap(exc):
    """No wording makes a node bug look like a missing dependency.

    The last entry matters most: a node that RAISES ModuleNotFoundError for a
    module this interpreter can import is lying, and the classifier checks the
    claim instead of believing the message.
    """
    assert _absence_signal(exc) is None, (
        f"{type(exc).__name__}({exc}) was classified as an environment gap; "
        "the classifier is reading the message again"
    )


def test_a_real_absence_is_still_excused():
    """The structural signals have to keep working, or everything turns red."""
    assert _absence_signal(ImportError("no gdal", name="radiance_no_such_module"))
    assert _absence_signal(FileNotFoundError(2, "No such file", "/nope.cube"))
    assert _absence_signal(_DependencyError("OpenImageIO is not installed"))

    wrapped = None
    try:
        try:
            raise ImportError("torchvision is missing", name="radiance_no_such_module")
        except ImportError as inner:
            raise RuntimeError("could not load the depth model") from inner
    except RuntimeError as outer:
        wrapped = outer
    assert _absence_signal(wrapped), "a wrapped absence must stay reachable"


def test_the_gated_node_list_stays_short_and_real():
    """Every gate names a node that exists, and there are few enough to read."""
    assert len(_ENVIRONMENT_GATED) <= 8, (
        "the environment-gate list is growing into the substring list it "
        "replaced; each entry costs a node's worth of execution coverage"
    )
    if not _NODES:
        pytest.skip("radiance did not import; test_node_smoke reports that")
    unknown = sorted(set(_ENVIRONMENT_GATED) - set(_NODES))
    assert not unknown, f"gates named for nodes that no longer exist: {unknown}"


def test_the_harness_actually_covered_the_catalog():
    """A harness that silently covers nothing is worse than no harness."""
    import radiance
    from radiance.config.constants import EXPECTED_MIN_NODE_COUNT

    if radiance._LOAD_RESULT.failures:
        pytest.skip("environment is short a runtime dependency")
    assert len(_KEYS) >= EXPECTED_MIN_NODE_COUNT, (
        f"only {len(_KEYS)} nodes were collected for functional testing"
    )


@pytest.mark.real_torch
def test_hdr_color_pipeline_decodes_pq():
    """Every PQ run raised TypeError: the node still passed peak_nits to an
    _eotf_pq() that dropped it. PQ code 0.58 is about 203 nits, i.e. 1.0."""
    import torch
    from radiance.nodes.hdr.colorspace import RadianceHDRColorPipeline
    node = RadianceHDRColorPipeline()
    img = torch.full((1, 4, 4, 3), 0.5807)
    out = getattr(node, node.FUNCTION)(img, encoding="PQ (ST.2084)")[1]   # scene-linear output
    assert abs(float(out[0, 0, 0, 0]) - 1.0) < 0.02
