"""
test_exr_workflow_embed.py — ComfyUI workflow provenance in written files.

RadianceWrite embeds "workflow" and "prompt" (the same metadata ComfyUI core
SaveImage puts into PNG tEXt chunks) into:
  • EXR header string attributes  — requires OpenEXR (self-skips without it)
  • PNG tEXt chunks               — requires Pillow  (self-skips without it)

Also validates the byte-level header layout that js/radiance_exr_workflow.js
walks in the browser, so the JS drag-drop loader contract is pinned here.
"""
import json
import struct
import sys
import types
import unittest
import tempfile
import os
from pathlib import Path

import numpy as np

# ComfyUI stubs (nodes_io imports folder_paths) — add-only, never replace
for _mod in ["folder_paths", "comfy", "comfy.utils"]:
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)
_fp = sys.modules["folder_paths"]
for _name, _val in [("get_output_directory", lambda: "/tmp/comfy_output"),
                    ("get_input_directory", lambda: "/tmp/comfy_input"),
                    ("get_temp_directory", lambda: "/tmp/comfy_temp")]:
    if not hasattr(_fp, _name):
        setattr(_fp, _name, _val)

try:
    import OpenEXR  # noqa: F401
    HAS_OPENEXR = True
except ImportError:
    HAS_OPENEXR = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

WORKFLOW = {"nodes": [{"id": 1, "type": "KSampler"}], "version": 0.4}
PROMPT = {"1": {"class_type": "KSampler", "inputs": {}}}


def _io():
    """Import nodes_io as `radiance.nodes.io.write` so its internal `from . import
    color_utils` relative imports resolve. Importing it as a bare top-level
    `nodes_io` module (the previous approach here) leaves it with no parent
    package, and that relative import raises ImportError before any test in
    this file can even run — this mirrors the working pattern already used in
    tests/test_io.py's `_import_io()`."""
    import importlib
    if "radiance.nodes.io.write" in sys.modules:
        return sys.modules["radiance.nodes.io.write"]
    if "nodes_io" in sys.modules:
        return sys.modules["radiance.nodes.io.write"]
    root = Path(__file__).parent.parent
    parent = str(root.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    try:
        return importlib.import_module("radiance.nodes.io.write")
    except ImportError:
        sys.path.insert(0, str(root))
        return importlib.import_module("radiance.nodes.io.write")


class TestWorkflowMetadataBuilder(unittest.TestCase):
    def test_builds_pnginfo_convention(self):
        io = _io()
        meta = io._workflow_metadata(PROMPT, {"workflow": WORKFLOW})
        self.assertEqual(json.loads(meta["prompt"]), PROMPT)
        self.assertEqual(json.loads(meta["workflow"]), WORKFLOW)

    def test_none_inputs_give_empty_dict(self):
        io = _io()
        self.assertEqual(io._workflow_metadata(None, None), {})


@unittest.skipUnless(HAS_OPENEXR, "OpenEXR not installed")
class TestExrEmbed(unittest.TestCase):
    def _write(self, metadata):
        io = _io()
        fd, path = tempfile.mkstemp(suffix=".exr")
        os.close(fd)
        self.addCleanup(os.unlink, path)
        arr = np.random.rand(8, 8, 3).astype(np.float32) * 4.0  # HDR values
        io._save_exr(arr, Path(path), half=True, metadata=metadata)
        return path

    def test_workflow_roundtrip_via_openexr(self):
        io = _io()
        meta = io._workflow_metadata(PROMPT, {"workflow": WORKFLOW})
        path = self._write(meta)
        hdr = OpenEXR.InputFile(path).header()
        got = hdr["workflow"]
        got = got.decode("utf-8") if isinstance(got, bytes) else got
        self.assertEqual(json.loads(got), WORKFLOW)
        got_p = hdr["prompt"]
        got_p = got_p.decode("utf-8") if isinstance(got_p, bytes) else got_p
        self.assertEqual(json.loads(got_p), PROMPT)

    def test_no_metadata_still_writes(self):
        path = self._write(None)
        self.assertGreater(os.path.getsize(path), 0)

    def test_js_header_walk_contract(self):
        """Replicate js/radiance_exr_workflow.js's parser byte-for-byte."""
        io = _io()
        meta = io._workflow_metadata(PROMPT, {"workflow": WORKFLOW})
        blob = open(self._write(meta), "rb").read()
        self.assertEqual(blob[:4], bytes([0x76, 0x2F, 0x31, 0x01]))

        attrs, off = {}, 8
        while off < len(blob) and blob[off] != 0:
            end = blob.index(b"\0", off); name = blob[off:end].decode(); off = end + 1
            end = blob.index(b"\0", off); typ = blob[off:end].decode(); off = end + 1
            size = struct.unpack_from("<i", blob, off)[0]; off += 4
            if typ == "string":
                attrs[name] = blob[off:off + size].decode("utf-8")
            off += size
        self.assertEqual(json.loads(attrs["workflow"]), WORKFLOW)
        self.assertEqual(json.loads(attrs["prompt"]), PROMPT)


@unittest.skipUnless(HAS_PIL, "Pillow not installed")
class TestPngEmbed(unittest.TestCase):
    def test_png_text_chunks_roundtrip(self):
        io = _io()
        meta = io._workflow_metadata(PROMPT, {"workflow": WORKFLOW})
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        self.addCleanup(os.unlink, path)
        arr = np.random.rand(8, 8, 3).astype(np.float32)
        io._save_pil_image(arr, Path(path), "PNG (8-bit)", metadata=meta)
        img = Image.open(path)
        self.assertEqual(json.loads(img.text["workflow"]), WORKFLOW)
        self.assertEqual(json.loads(img.text["prompt"]), PROMPT)


if __name__ == "__main__":
    unittest.main()
