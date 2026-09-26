"""
tests/test_viewer_scopes.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Tests for the v3.0 viewer upgrade nodes (nodes_radiance_viewer.py):

  Overlay helpers : _lut_clip_check
"""

from __future__ import annotations

import os
import sys
import types
import unittest

import numpy as np
try:
    import torch
    HAS_TORCH = hasattr(torch, "__version__")
    if HAS_TORCH and not hasattr(torch, "device"):
        torch.device = type("device", (), {})
    if HAS_TORCH and not hasattr(torch, "tensor"):
        torch.tensor = lambda *a, **kw: None
    if HAS_TORCH and not hasattr(torch, "inverse"):
        torch.inverse = lambda *a, **kw: None
    if HAS_TORCH and not hasattr(torch, "no_grad"):
        class DummyNoGrad:
            def __init__(self, *a, **kw): pass
            def __call__(self, func): return func
            def __enter__(self): pass
            def __exit__(self, *a): pass
        torch.no_grad = DummyNoGrad
    if HAS_TORCH and not hasattr(torch, "Generator"):
        torch.Generator = type("Generator", (), {})
except ImportError:
    torch = types.ModuleType("torch")
    torch.Tensor = type("Tensor", (), {})
    torch.device = type("device", (), {})
    torch.Generator = type("Generator", (), {})
    torch.tensor = lambda *a, **kw: None
    torch.inverse = lambda *a, **kw: None
    class DummyNoGrad:
        def __init__(self, *a, **kw): pass
        def __call__(self, func): return func
        def __enter__(self): pass
        def __exit__(self, *a): pass
    torch.no_grad = DummyNoGrad
    torch.float16 = "float16"
    torch.bfloat16 = "bfloat16"
    torch.float32 = "float32"
    torch.float64 = "float64"
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    sys.modules["torch"] = torch
    HAS_TORCH = False






_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.dirname(_ROOT))

skip_no_torch = unittest.skipUnless(HAS_TORCH, "real torch not available")

# ── Minimal stubs so the module loads without a full ComfyUI environment ──────

def _install_stubs():
    for mod in ("folder_paths", "aiohttp", "server"):
        if mod not in sys.modules:
            m = types.ModuleType(mod)
            if mod == "aiohttp":
                m.web = types.SimpleNamespace()
            if mod == "server":
                ps = types.SimpleNamespace()
                ps.instance = types.SimpleNamespace()
                ps.instance.routes = types.SimpleNamespace(
                    post=lambda p: (lambda f: f),
                    get=lambda p: (lambda f: f),
                )
                m.PromptServer = ps
            sys.modules[mod] = m

_install_stubs()

# NOTE: this file used to carry a defensive re-patch of the torch stub because
# test_sdr_conditioning.py replaced sys.modules["torch"] at collection time.
# That root cause is fixed (stubs are now augmented in place, never replaced),
# so the workaround has been removed. All shared stubbing lives in conftest.py —
# test files must never replace sys.modules entries, only add missing ones.

# Load nodes_radiance_viewer fresh from disk, bypassing any stale sys.modules
# cache left behind by tests that ran before us with stub torch environments.
import importlib.util as _ilu

# The viewer module itself, not the deleted root shim that re-exported it.
# The shim was a second name for this file, which is exactly the double import
# `_radiance_route_once` exists to survive.
_VIEWER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "nodes", "monitor", "viewer.py",
)
_IMPORT_ERROR = None
try:
    _spec = _ilu.spec_from_file_location("_nrv_fresh", _VIEWER_PATH)
    nrv = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(nrv)
except Exception as _import_err:
    import types as _types
    nrv = _types.ModuleType("_nrv_fresh")
    nrv.NODE_CLASS_MAPPINGS = {}
    nrv.NODE_DISPLAY_NAME_MAPPINGS = {}
    _IMPORT_ERROR = str(_import_err)


def _rand_frame(h=32, w=32, c=3) -> np.ndarray:
    return np.random.rand(h, w, c).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
#   Clip Check Overlay
# ═══════════════════════════════════════════════════════════════════════════════

class TestClipCheck(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    def test_output_shape(self):
        img = _rand_frame(16, 16, 3)
        out = nrv._lut_clip_check(img)
        self.assertEqual(out.shape, (16, 16, 3))

    def test_output_dtype(self):
        out = nrv._lut_clip_check(_rand_frame())
        self.assertEqual(out.dtype, np.float32)

    def test_values_clipped_to_01(self):
        img = np.ones((4, 4, 3), dtype=np.float32) * 2.0  # all blown
        out = nrv._lut_clip_check(img)
        self.assertLessEqual(out.max(), 1.0)
        self.assertGreaterEqual(out.min(), 0.0)

    def test_blown_pixels_are_red(self):
        img = np.zeros((1, 3, 3), dtype=np.float32)
        img[0, 2, :] = 1.5  # third pixel blown
        out = nrv._lut_clip_check(img)
        self.assertGreater(out[0, 2, 0], 0.8, "Blown pixel R should be high")
        self.assertLess(out[0, 2, 1], 0.3, "Blown pixel G should be low")

    def test_crushed_pixels_are_blue(self):
        # black_floor = 0.05 → pixel at 0.0 is crushed
        img = np.full((1, 1, 3), 0.0, dtype=np.float32)
        out = nrv._lut_clip_check(img, black_floor=0.05)
        self.assertGreater(out[0, 0, 2], 0.8, "Crushed pixel B should be high")
        self.assertLess(out[0, 0, 0], 0.3, "Crushed pixel R should be low")

    def test_in_range_pixels_are_desaturated(self):
        img = np.full((1, 1, 3), 0.5, dtype=np.float32)
        out = nrv._lut_clip_check(img)
        # Grey: R ≈ G ≈ B and all low (attenuated)
        self.assertAlmostEqual(float(out[0, 0, 0]), float(out[0, 0, 1]), places=4)
        self.assertAlmostEqual(float(out[0, 0, 1]), float(out[0, 0, 2]), places=4)

    def test_4channel_input(self):
        img = np.ones((4, 4, 4), dtype=np.float32) * 0.5
        out = nrv._lut_clip_check(img)
        self.assertEqual(out.shape, (4, 4, 3))


if __name__ == "__main__":
    unittest.main(verbosity=2)
