"""
test_exr_multipass_workflow.py — ComfyUI workflow provenance in multipass/AOV EXRs.

RadianceEXRPassesWriter (nodes/vfx/multipass/master.py) writes via
hdr/io.py's write_exr_openexr (single-part multilayer) and write_exr_multipart
(true multi-part). Both must embed "workflow"/"prompt" as unprefixed EXR
header string attributes — the same convention RadianceWrite uses for its
EXR output and that js/radiance_exr_workflow.js parses on drag-drop — while
still prefixing arbitrary custom metadata keys with "rad_" to avoid collisions
with standard OpenEXR attributes (regression coverage for that existing rule).
"""
import json
import os
import sys
import tempfile
import types
import unittest
import importlib.util

import numpy as np

_RADIANCE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _RADIANCE_ROOT)

_STUBBED_MODULE_NAMES = (
    "torch", "folder_paths", "radiance", "radiance.hdr", "radiance.hdr.utils",
    "radiance.color_utils", "radiance.color", "radiance.color.transfer",
    "radiance.color.matrices", "radiance.path_utils", "radiance.hdr.io",
)
_PREVIOUS_MODULES = {name: sys.modules.get(name) for name in _STUBBED_MODULE_NAMES}

try:
    import OpenEXR  # noqa: F401
    HAS_OPENEXR = True
except ImportError:
    HAS_OPENEXR = False

# ── Stub heavy dependencies so hdr/io.py can be imported without GPU/torch ──
# (same technique as tests/test_exr_metadata.py)
_torch_stub = types.ModuleType("torch")
_torch_stub.Tensor = object
sys.modules.setdefault("torch", _torch_stub)
sys.modules.setdefault("folder_paths", types.ModuleType("folder_paths"))

_rad_pkg = types.ModuleType("radiance")
_rad_pkg.__path__ = [_RADIANCE_ROOT]
_rad_pkg.__package__ = "radiance"
sys.modules.setdefault("radiance", _rad_pkg)

_rad_hdr_pkg = types.ModuleType("radiance.hdr")
_rad_hdr_pkg.__path__ = [os.path.join(_RADIANCE_ROOT, "hdr")]
_rad_hdr_pkg.__package__ = "radiance.hdr"
sys.modules.setdefault("radiance.hdr", _rad_hdr_pkg)

_rad_hdr_utils = types.ModuleType("radiance.hdr.utils")
_rad_hdr_utils.numpy_to_tensor_float32 = lambda x: x
sys.modules.setdefault("radiance.hdr.utils", _rad_hdr_utils)

_cu = types.ModuleType("radiance.color_utils")
for _n in ["linear_to_logc3", "linear_to_logc4", "linear_to_slog3",
           "srgb_to_linear", "linear_to_srgb", "apply_matrix_transform",
           "AWG3_TO_ACESCG", "AWG4_TO_ACESCG", "SGAMUT3_CINE_TO_ACESCG", "ACESCG_TO_SRGB"]:
    setattr(_cu, _n, None)
sys.modules.setdefault("radiance.color_utils", _cu)

# hdr/io.py actually does `from radiance.color.transfer import (...)` and
# `from radiance.color.matrices import (...)` — stub those submodules
# directly so the real color/__init__.py -> color/lut.py chain (which
# evaluates `torch.device` in a class-body annotation and crashes against
# this lightweight torch stub) never executes.
_rad_color_pkg = types.ModuleType("radiance.color")
_rad_color_pkg.__path__ = [os.path.join(_RADIANCE_ROOT, "color")]
_rad_color_pkg.__package__ = "radiance.color"
sys.modules.setdefault("radiance.color", _rad_color_pkg)

_rad_color_transfer = types.ModuleType("radiance.color.transfer")
for _n in ["linear_to_logc3", "linear_to_logc4", "linear_to_slog3",
           "srgb_to_linear", "linear_to_srgb"]:
    setattr(_rad_color_transfer, _n, lambda *a, **kw: None)
sys.modules.setdefault("radiance.color.transfer", _rad_color_transfer)

_rad_color_matrices = types.ModuleType("radiance.color.matrices")
for _n in ["apply_matrix_transform", "AWG3_TO_ACESCG", "AWG4_TO_ACESCG",
           "SGAMUT3_CINE_TO_ACESCG", "ACESCG_TO_SRGB"]:
    setattr(_rad_color_matrices, _n, None)
sys.modules.setdefault("radiance.color.matrices", _rad_color_matrices)

_pu = types.ModuleType("radiance.path_utils")
for _fn in ["safe_join", "get_safe_output_dir", "get_safe_input_path", "get_next_index"]:
    setattr(_pu, _fn, lambda *a, **kw: None)
sys.modules.setdefault("radiance.path_utils", _pu)

_io_path = os.path.join(_RADIANCE_ROOT, "hdr", "io.py")
_io_spec = importlib.util.spec_from_file_location("radiance.hdr.io", _io_path)
_io_mod = importlib.util.module_from_spec(_io_spec)
_io_mod.__package__ = "radiance.hdr"
sys.modules["radiance.hdr.io"] = _io_mod
_io_spec.loader.exec_module(_io_mod)

write_exr_openexr = _io_mod.write_exr_openexr
write_exr_multipart = _io_mod.write_exr_multipart

# The loaded functions retain their module globals; restore the process module
# table so this lightweight import harness cannot poison unrelated tests.
for _name, _previous in _PREVIOUS_MODULES.items():
    if _previous is None:
        sys.modules.pop(_name, None)
    else:
        sys.modules[_name] = _previous

WORKFLOW = {"nodes": [{"id": 1, "type": "RadianceEXRPassesWriter"}], "version": 0.4}
PROMPT = {"1": {"class_type": "RadianceEXRPassesWriter", "inputs": {}}}


def _read_header_attr(path, name):
    import OpenEXR
    hdr = OpenEXR.InputFile(path).header()
    val = hdr[name]
    return val.decode("utf-8") if isinstance(val, bytes) else val


@unittest.skipUnless(HAS_OPENEXR, "OpenEXR not installed")
class TestSinglePartMultilayerWorkflowEmbed(unittest.TestCase):
    def _write(self, metadata):
        fd, path = tempfile.mkstemp(suffix=".exr")
        os.close(fd)
        self.addCleanup(os.unlink, path)
        h, w = 4, 4
        channels = {
            "R": np.random.rand(h, w).astype(np.float32),
            "G": np.random.rand(h, w).astype(np.float32),
            "B": np.random.rand(h, w).astype(np.float32),
        }
        write_exr_openexr(path, channels, "ZIP", "HALF", metadata)
        return path

    def test_workflow_and_prompt_unprefixed_roundtrip(self):
        meta = {
            "workflow": json.dumps(WORKFLOW),
            "prompt": json.dumps(PROMPT),
            "software": "Radiance VFX Multipass v3.1",
        }
        path = self._write(meta)
        self.assertEqual(json.loads(_read_header_attr(path, "workflow")), WORKFLOW)
        self.assertEqual(json.loads(_read_header_attr(path, "prompt")), PROMPT)

    def test_custom_metadata_key_not_stored_unprefixed(self):
        """Arbitrary custom keys (e.g. from the custom_metadata textbox) must
        never land under their bare name — only "workflow"/"prompt" are
        stored unprefixed. (Whether the rad_-prefixed form round-trips
        depends on the installed OpenEXR binding's handling of plain-str
        custom attributes — see the docstring on write_exr_openexr's
        metadata loop — so this only pins the collision-avoidance guarantee,
        not that specific storage detail.)"""
        meta = {"workflow": json.dumps(WORKFLOW), "shot": "sh010"}
        path = self._write(meta)
        hdr_keys = OpenEXR.InputFile(path).header().keys()
        self.assertNotIn("shot", hdr_keys)
        self.assertIn("workflow", hdr_keys)


@unittest.skipUnless(HAS_OPENEXR, "OpenEXR not installed")
class TestMultiPartWorkflowEmbed(unittest.TestCase):
    def test_workflow_and_prompt_unprefixed_in_every_part(self):
        """
        write_exr_multipart() has two code paths: true multi-part
        (OpenEXR.MultiPartOutputFile) when the installed OpenEXR binding
        supports it, else a documented fallback that writes one separate
        ``{base}.{part_name}.exr`` file per pass (see hdr/io.py's
        except-block). Newer OpenEXR Python bindings (observed: 3.4.x) no
        longer expose MultiPartOutputFile at all, so the fallback is the
        common case, not an edge case — this test must check whichever
        path actually ran rather than assuming true multi-part output.
        """
        parts = {
            "beauty": np.random.rand(4, 4, 3).astype(np.float32),
            "depth": np.random.rand(4, 4, 1).astype(np.float32),
        }
        meta = {"workflow": json.dumps(WORKFLOW), "prompt": json.dumps(PROMPT)}
        fd, path = tempfile.mkstemp(suffix=".exr")
        os.close(fd)
        base, _ = os.path.splitext(path)
        fallback_paths = [f"{base}.{name}.exr" for name in parts]
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        for fp in fallback_paths:
            self.addCleanup(lambda fp=fp: os.path.exists(fp) and os.unlink(fp))

        ok = write_exr_multipart(path, parts, "16-bit Half Float", "ZIP", meta)
        self.assertTrue(ok)

        # Pick whichever file(s) actually got EXR content: `path` already
        # exists as an empty file from mkstemp() regardless of which code
        # path ran, so `os.path.exists(path)` alone can't tell them apart —
        # check for actual written bytes instead. True multi-part writes to
        # `path`; the per-layer fallback writes to `fallback_paths` instead
        # and leaves `path` empty.
        check_paths = [path] if os.path.getsize(path) > 0 else fallback_paths
        self.assertTrue(check_paths and all(os.path.exists(p) for p in check_paths),
                         "write_exr_multipart returned True but wrote no file")

        for p in check_paths:
            hdr = OpenEXR.InputFile(p).header()
            self.assertEqual(json.loads(
                hdr["workflow"].decode("utf-8") if isinstance(hdr["workflow"], bytes) else hdr["workflow"]
            ), WORKFLOW)
            self.assertEqual(json.loads(
                hdr["prompt"].decode("utf-8") if isinstance(hdr["prompt"], bytes) else hdr["prompt"]
            ), PROMPT)


if __name__ == "__main__":
    unittest.main()
