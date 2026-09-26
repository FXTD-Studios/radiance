"""
Tests for AVControl-style SDR reference conditioning in RadianceSamplerPro
(nodes_sampler.py  v4.3).

Covers:
  _encode_sdr_reference               (requires real torch — skipped without it)
    • output shape matches work_latent exactly (4-D image, 5-D video)
    • spatial resize when reference is a different resolution
    • temporal broadcast for video latents
    • batch broadcast from single reference to N-batch latent
    • channel trim / pad when VAE output channels differ

  Init-blend arithmetic               (numpy-backed, no real torch needed)
    • lerp formula correct for blend in {0, 0.5, 1}
    • result values bounded between latent and sdr

  Post-CFG anchor function            (numpy-backed, no real torch needed)
    • step < inject_steps  → x0 nudged toward SDR reference
    • step >= inject_steps → x0 returned unmodified
    • blend strength decays as sdr_blend × decay^step
    • step counter increments on every call
    • shape mismatch → returns x0 unmodified (safe fallback)

  _build_latent_meta                  (numpy-backed sigmas, no real torch needed)
    • sdr_conditioning block present when sdr_blend > 0
    • sdr_conditioning block absent when sdr_blend == 0
    • round-trip JSON parse with correct keys

  INPUT_TYPES contract                (no torch needed)
    • sdr_reference, sdr_vae, sdr_blend, sdr_inject_steps, sdr_decay all in 'optional'
"""

import sys
import os
import json
import types
import unittest

import numpy as np

# ── Locate the repo root ──────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

# ── Detect whether real PyTorch is installed ──────────────────────────────────
_HAS_REAL_TORCH = False
try:
    import importlib as _il
    _spec = _il.util.find_spec("torch")
    if _spec is not None:
        import torch as _t
        _HAS_REAL_TORCH = hasattr(_t, "rand") and hasattr(_t, "zeros")
except Exception:
    pass

# ── Build a numpy-backed "torch" stub that covers operations in our tests ─────
#    Only installed in sys.modules if real torch is absent.

class _NpTensor:
    """
    Minimal numpy-backed tensor that mimics the PyTorch Tensor API used by
    _encode_sdr_reference, the post-CFG closure, and the blend arithmetic.
    """
    def __init__(self, data):
        if isinstance(data, _NpTensor):
            self._d = data._d
        else:
            self._d = np.asarray(data, dtype=np.float32)

    # ── Properties ────────────────────────────────────────────────────────────
    @property
    def shape(self):
        return _ShapeProxy(self._d.shape)

    @property
    def ndim(self):
        return self._d.ndim

    @property
    def dtype(self):
        return np.float32

    @property
    def device(self):
        return "cpu"

    # ── Torch-compatible methods ───────────────────────────────────────────────
    def to(self, *args, **kwargs):
        return self          # no-op — everything is already on CPU / float32

    def detach(self):
        return self

    def float(self):
        return self

    def clamp(self, min=None, max=None):
        return _NpTensor(np.clip(self._d, min, max))

    def expand(self, *shape):
        return _NpTensor(np.broadcast_to(self._d, shape).copy())

    def expand_as(self, other):
        return _NpTensor(np.broadcast_to(self._d, _data(other).shape).copy())

    def contiguous(self):
        return _NpTensor(np.ascontiguousarray(self._d))

    def unsqueeze(self, dim):
        return _NpTensor(np.expand_dims(self._d, dim))

    def mean(self):
        return float(self._d.mean())

    def item(self):
        return float(self._d.flat[0])

    def clone(self):
        return _NpTensor(self._d.copy())

    def cpu(self):
        return self

    def numpy(self):
        return self._d

    @property
    def T(self):
        return _NpTensor(self._d.T)

    def transpose(self, dim0, dim1):
        axes = list(range(self._d.ndim))
        axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
        return _NpTensor(np.transpose(self._d, axes))

    def dim(self):
        return self._d.ndim

    def size(self, dim=None):
        if dim is None:
            return self._d.shape
        return self._d.shape[dim]

    def squeeze(self, dim=None):
        if dim is None:
            return _NpTensor(self._d.squeeze())
        return _NpTensor(self._d.squeeze(axis=dim))

    def permute(self, *dims):
        return _NpTensor(np.transpose(self._d, dims))

    def __getitem__(self, idx):
        return _NpTensor(self._d[idx])

    def __setitem__(self, idx, val):
        self._d[idx] = _data(val)

    def __len__(self):
        return self._d.shape[0]

    # ── Arithmetic ────────────────────────────────────────────────────────────
    def __add__(self, other):  return _NpTensor(self._d + _data(other))
    def __radd__(self, other): return _NpTensor(_data(other) + self._d)
    def __sub__(self, other):  return _NpTensor(self._d - _data(other))
    def __rsub__(self, other): return _NpTensor(_data(other) - self._d)
    def __mul__(self, other):  return _NpTensor(self._d * _data(other))
    def __rmul__(self, other): return _NpTensor(_data(other) * self._d)
    def __truediv__(self, other): return _NpTensor(self._d / _data(other))
    def __neg__(self):         return _NpTensor(-self._d)

    # ── Comparison (for sigmas > 0 etc.) ─────────────────────────────────────
    def __gt__(self, other): return _NpTensor(self._d > _data(other))
    def __ge__(self, other): return _NpTensor(self._d >= _data(other))
    def __le__(self, other): return _NpTensor(self._d <= _data(other))

    def any(self): return bool(self._d.any())
    def min(self): return float(self._d.min())
    def max(self): return float(self._d.max())

    # ── Indexing ──────────────────────────────────────────────────────────────
    def __getitem__(self, idx):
        if isinstance(idx, _NpTensor):
            idx = idx._d.astype(bool)
        result = self._d[idx]
        if result.ndim == 0:
            return float(result)
        return _NpTensor(result)

    def __repr__(self):
        return f"_NpTensor({self._d})"


class _ShapeProxy(tuple):
    """Tuple subclass so shape comparisons work with torch.Size notation."""
    pass


def _data(x):
    if isinstance(x, _NpTensor):
        return x._d
    return x


class _TorchStub:
    """Module-level torch stub backed by _NpTensor."""
    Tensor = _NpTensor
    float32 = np.float32

    @staticmethod
    def zeros(*shape, **kw):
        if len(shape) == 1 and hasattr(shape[0], "__len__"):
            shape = tuple(shape[0])
        return _NpTensor(np.zeros(shape, dtype=np.float32))

    @staticmethod
    def ones(*shape, **kw):
        if len(shape) == 1 and hasattr(shape[0], "__len__"):
            shape = tuple(shape[0])
        return _NpTensor(np.ones(shape, dtype=np.float32))

    @staticmethod
    def rand(*shape):
        return _NpTensor(np.random.rand(*shape).astype(np.float32))

    @staticmethod
    def full(shape, fill_value, **kw):
        return _NpTensor(np.full(shape, fill_value, dtype=np.float32))

    @staticmethod
    def full_like(other, fill_value):
        return _NpTensor(np.full(_data(other).shape, fill_value, dtype=np.float32))

    @staticmethod
    def zeros_like(other):
        return _NpTensor(np.zeros_like(_data(other), dtype=np.float32))

    @staticmethod
    def ones_like(other):
        return _NpTensor(np.ones_like(_data(other), dtype=np.float32))

    @staticmethod
    def tensor(data, **kw):
        return _NpTensor(np.array(data, dtype=np.float32))

    @staticmethod
    def cat(tensors, dim=0):
        return _NpTensor(np.concatenate([_data(t) for t in tensors], axis=dim))

    @staticmethod
    def allclose(a, b, atol=1e-8, rtol=1e-5):
        return np.allclose(_data(a), _data(b), atol=atol, rtol=rtol)

    @staticmethod
    def device(s):
        return s


if not _HAS_REAL_TORCH:
    # CRITICAL: do NOT replace sys.modules["torch"] here. This file is imported
    # at pytest COLLECTION time; swapping in a fresh module would evict the
    # conftest torch stub (which carries Generator, nn, no_grad, dtypes, …) for
    # every module imported afterwards — e.g. test_node_load_completeness
    # importing radiance.film.grain, whose `generator: torch.Generator`
    # signature is evaluated at import. That was CI failure
    # "AttributeError: module 'torch' has no attribute 'Generator'".
    # Instead, AUGMENT the already-installed stub in place so its identity —
    # and every attribute other tests rely on — is preserved.
    _stub = _TorchStub()
    _torch_mod = sys.modules.get("torch")
    if _torch_mod is None:  # standalone run without conftest
        import importlib as _il2
        _torch_mod = _il2.util.module_from_spec(
            _il2.util.spec_from_loader("torch", loader=None)
        )
        sys.modules["torch"] = _torch_mod
    for _attr in dir(_stub):
        if not _attr.startswith("__"):
            try:
                setattr(_torch_mod, _attr, getattr(_stub, _attr))
            except Exception:
                pass

# Now import torch — gets real torch or the augmented stub
import torch   # noqa: E402  (used in test bodies below)

# ── comfy stubs ───────────────────────────────────────────────────────────────────
# Augment — never replace — comfy stubs already installed by conftest.py.
# Replacing sys.modules entries at collection time evicts the conftest stubs
# (which carry get_free_memory, soft_empty_cache, …) for every module imported
# afterwards, breaking e.g. radiance.nodes.generate.denoise in
# test_node_load_completeness.

def _ensure_mod(name, parent=None, attr=None):
    mod = sys.modules.get(name)
    if mod is None:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
    if parent is not None and getattr(parent, attr, None) is None:
        setattr(parent, attr, mod)
    return mod

_comfy = _ensure_mod("comfy")
_comfy_samplers = _ensure_mod("comfy.samplers", _comfy, "samplers")
if getattr(_comfy_samplers, "KSampler", None) is None:
    _comfy_samplers.KSampler = types.SimpleNamespace()
_comfy_samplers.KSampler.SAMPLERS = ["euler"]
_comfy_samplers.KSampler.SCHEDULERS = ["simple", "normal"]

_comfy_sample = _ensure_mod("comfy.sample", _comfy, "sample")
_comfy_sample.prepare_noise = lambda lat, seed, _: _NpTensor(np.zeros_like(_data(lat)))
_comfy_sample.sample_custom = lambda *a, **kw: a[4]   # returns latent unchanged

_comfy_mm = _ensure_mod("comfy.model_management", _comfy, "model_management")
_comfy_mm.get_torch_device = lambda: "cpu"
if not hasattr(_comfy_mm, "get_free_memory"):   # standalone run without conftest
    _comfy_mm.get_free_memory = lambda *a, **kw: 8 * 1024**3

_comfy_utils = _ensure_mod("comfy.utils", _comfy, "utils")
_comfy_utils.ProgressBar = lambda n: types.SimpleNamespace(
    update_absolute=lambda *a, **kw: None, update=lambda *a, **kw: None
)

if "folder_paths" not in sys.modules:
    sys.modules["folder_paths"] = types.ModuleType("folder_paths")

# ── Load the sampler ─────────────────────────────────────────────────────────
#
# This used to load the module from its file path under the bare name
# "nodes_sampler", giving the process a second, independent copy of a module the
# package also imports — the same double-import the aiohttp route guards exist
# to survive. It also swallowed every exception from `exec_module`, so an import
# error surfaced later as `AttributeError: module 'nodes_sampler' has no
# attribute 'RadianceSamplerPro'` with the real cause discarded.

# This module already gates its torch-dependent tests correctly (they skip
# cleanly against conftest's stub), so opt out of the automatic module-level
# skip and keep the rest of the file running on the no-torch CI matrix.
RADIANCE_TORCH_GATED = True

from radiance.nodes.generate.sampler import RadianceSamplerPro  # noqa: E402
from radiance.sampler_utils import _build_latent_meta  # noqa: E402


# ── Numpy-based sigmas helper ─────────────────────────────────────────────────
def _np_sigmas(*vals):
    """Return a _NpTensor (or real torch tensor if available) for sigmas."""
    if _HAS_REAL_TORCH:
        import torch as _rt
        return _rt.tensor(list(vals), dtype=_rt.float32)
    return _NpTensor(np.array(vals, dtype=np.float32))


# ── Fake VAE (works with both real torch and the NpTensor stub) ───────────────
class _FakeVAE:
    def __init__(self, out_channels=4, scale_factor=8):
        self._c = out_channels
        self._sf = scale_factor

    def encode(self, pixels):
        if _HAS_REAL_TORCH:
            import torch as _rt
            B = pixels.shape[0] if hasattr(pixels.shape, '__getitem__') else 1
            H = pixels.shape[1]
            W = pixels.shape[2]
            lh = max(1, H // self._sf)
            lw = max(1, W // self._sf)
            return {"samples": _rt.zeros(B, self._c, lh, lw)}
        else:
            d = _data(pixels) if isinstance(pixels, _NpTensor) else np.asarray(pixels)
            B, H, W, C = d.shape
            lh = max(1, H // self._sf)
            lw = max(1, W // self._sf)
            return {"samples": _NpTensor(np.zeros((B, self._c, lh, lw), dtype=np.float32))}


# ── Tensor constructor helper (works in both environments) ────────────────────
def _t(*args, **kw):
    """Create a tensor using real torch or the NpTensor stub."""
    if _HAS_REAL_TORCH:
        import torch as _rt
        return _rt.tensor(*args, **kw)
    return _NpTensor(np.array(args[0] if args else [], dtype=np.float32))


def _zeros(*shape):
    if _HAS_REAL_TORCH:
        import torch as _rt
        return _rt.zeros(*shape)
    return _NpTensor(np.zeros(shape, dtype=np.float32))


def _ones(*shape):
    if _HAS_REAL_TORCH:
        import torch as _rt
        return _rt.ones(*shape)
    return _NpTensor(np.ones(shape, dtype=np.float32))


def _rand(*shape):
    if _HAS_REAL_TORCH:
        import torch as _rt
        return _rt.rand(*shape)
    return _NpTensor(np.random.rand(*shape).astype(np.float32))


def _full(shape, val):
    if _HAS_REAL_TORCH:
        import torch as _rt
        return _rt.full(shape, val)
    return _NpTensor(np.full(shape, val, dtype=np.float32))


def _allclose(a, b, atol=1e-5):
    if _HAS_REAL_TORCH:
        import torch as _rt
        return bool(_rt.allclose(a, b, atol=atol))
    return bool(np.allclose(_data(a), _data(b), atol=atol))


# ═════════════════════════════════════════════════════════════════════════════
#               _encode_sdr_reference   (requires real torch)
# ═════════════════════════════════════════════════════════════════════════════

@unittest.skipUnless(_HAS_REAL_TORCH, "requires real PyTorch")
class TestEncodeSDRReference(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    def _node(self):
        return RadianceSamplerPro()

    def test_4d_output_shape_matches_work_latent(self):
        import torch as _rt
        work = _rt.zeros(1, 4, 16, 24)
        ref  = _rt.rand(1, 128, 192, 3)
        vae  = _FakeVAE(out_channels=4, scale_factor=8)
        out  = self._node()._encode_sdr_reference(ref, vae, work)
        self.assertEqual(tuple(out.shape), tuple(work.shape))

    def test_4d_output_dtype_float32(self):
        import torch as _rt
        work = _rt.zeros(1, 4, 16, 24)
        ref  = _rt.rand(1, 128, 192, 3)
        vae  = _FakeVAE(out_channels=4)
        out  = self._node()._encode_sdr_reference(ref, vae, work)
        self.assertEqual(out.dtype, _rt.float32)

    def test_spatial_resize_to_target_size(self):
        import torch as _rt
        work = _rt.zeros(1, 4, 32, 32)
        ref  = _rt.rand(1, 64, 64, 3)        # 8× → VAE gives 8×8 → needs resize to 32×32
        vae  = _FakeVAE(out_channels=4, scale_factor=8)
        out  = self._node()._encode_sdr_reference(ref, vae, work)
        self.assertEqual(tuple(out.shape[-2:]), (32, 32))

    def test_5d_output_shape_matches_video_latent(self):
        import torch as _rt
        T   = 8
        work = _rt.zeros(1, 4, T, 16, 24)
        ref  = _rt.rand(1, 128, 192, 3)
        vae  = _FakeVAE(out_channels=4, scale_factor=8)
        out  = self._node()._encode_sdr_reference(ref, vae, work)
        self.assertEqual(tuple(out.shape), tuple(work.shape))

    def test_5d_temporal_broadcast_uniform(self):
        import torch as _rt
        T   = 6
        work = _rt.zeros(1, 4, T, 8, 8)
        ref  = _rt.rand(1, 64, 64, 3)
        vae  = _FakeVAE(out_channels=4, scale_factor=8)
        out  = self._node()._encode_sdr_reference(ref, vae, work)
        for t in range(1, T):
            self.assertTrue(
                bool(_rt.allclose(out[:, :, 0], out[:, :, t])),
                f"Frame {t} differs from frame 0 — broadcast not uniform",
            )

    def test_batch_broadcast_single_ref_to_n_batch(self):
        import torch as _rt
        B   = 4
        work = _rt.zeros(B, 4, 16, 16)
        ref  = _rt.rand(1, 128, 128, 3)
        vae  = _FakeVAE(out_channels=4, scale_factor=8)
        out  = self._node()._encode_sdr_reference(ref, vae, work)
        self.assertEqual(out.shape[0], B)

    def test_channel_trim_extra_vae_channels(self):
        import torch as _rt
        work = _rt.zeros(1, 4, 16, 16)
        ref  = _rt.rand(1, 128, 128, 3)
        vae  = _FakeVAE(out_channels=16, scale_factor=8)
        out  = self._node()._encode_sdr_reference(ref, vae, work)
        self.assertEqual(out.shape[1], 4)

    def test_channel_pad_fewer_vae_channels(self):
        import torch as _rt
        work = _rt.zeros(1, 16, 16, 16)
        ref  = _rt.rand(1, 128, 128, 3)
        vae  = _FakeVAE(out_channels=4, scale_factor=8)
        out  = self._node()._encode_sdr_reference(ref, vae, work)
        self.assertEqual(out.shape[1], 16)


# ═════════════════════════════════════════════════════════════════════════════
#               Init-blend arithmetic      (numpy-backed)
# ═════════════════════════════════════════════════════════════════════════════

class TestInitBlendArithmetic(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    def _lerp(self, a, b, blend):
        return (1.0 - blend) * a + blend * b

    def test_blend_zero_leaves_latent_unchanged(self):
        latent = _rand(1, 4, 8, 8)
        sdr    = _full((1, 4, 8, 8), 5.0)
        result = self._lerp(latent, sdr, 0.0)
        self.assertTrue(_allclose(result, latent))

    def test_blend_one_replaces_with_sdr(self):
        latent = _rand(1, 4, 8, 8)
        sdr    = _full((1, 4, 8, 8), 5.0)
        result = self._lerp(latent, sdr, 1.0)
        self.assertTrue(_allclose(result, sdr))

    def test_blend_half_is_mean(self):
        latent = _zeros(1, 4, 8, 8)
        sdr    = _full((1, 4, 8, 8), 2.0)
        result = self._lerp(latent, sdr, 0.5)
        expected = _ones(1, 4, 8, 8)    # (0 + 2) / 2 = 1
        self.assertTrue(_allclose(result, expected))

    def test_exact_lerp_formula(self):
        b = 0.35
        a_val, b_val = 2.0, 10.0
        result_val = (1 - b) * a_val + b * b_val
        self.assertAlmostEqual(result_val, 0.65 * 2.0 + 0.35 * 10.0, places=6)

    def test_result_bounded_between_endpoints(self):
        lo, hi = 1.0, 3.0
        latent = _full((1, 4, 8, 8), lo)
        sdr    = _full((1, 4, 8, 8), hi)
        for blend in [0.1, 0.3, 0.5, 0.7, 0.9]:
            result = self._lerp(latent, sdr, blend)
            arr = _data(result) if isinstance(result, _NpTensor) else result.numpy()
            self.assertTrue(
                (arr >= lo - 1e-6).all() and (arr <= hi + 1e-6).all(),
                f"blend={blend}: value out of [{lo},{hi}]"
            )


# ═════════════════════════════════════════════════════════════════════════════
#               Post-CFG anchor function   (numpy-backed)
# ═════════════════════════════════════════════════════════════════════════════

class TestPostCFGAnchor(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    """
    Re-creates the _sdr_post_cfg closure exactly as written in sample() and
    exercises it with _NpTensor / real-tensor arguments.
    """

    def _make_anchor(self, sdr_latent, sdr_blend, sdr_inject_steps, sdr_decay):
        _step_counter = [0]
        _sdr_ref_lat  = sdr_latent  # already detached (NpTensor has .detach())

        def _sdr_post_cfg(args):
            step  = _step_counter[0]
            _step_counter[0] += 1
            if step >= sdr_inject_steps:
                return args["denoised"]
            blend = sdr_blend * (sdr_decay ** step)
            if blend < 1e-5:
                return args["denoised"]
            x0  = args["denoised"]
            ref = _sdr_ref_lat
            if hasattr(ref, "to"):
                ref = ref.to(device=getattr(x0, "device", "cpu"),
                             dtype=getattr(x0, "dtype", None))
            if _shape(ref) != _shape(x0):
                if _ndim(ref) == _ndim(x0):
                    if _shape(ref)[0] == 1 and _shape(x0)[0] > 1:
                        ref = ref.expand_as(x0)
                    else:
                        return x0
                else:
                    return x0
            return (1.0 - blend) * x0 + blend * ref

        return _sdr_post_cfg, _step_counter

    def _args(self, x0):
        return {"denoised": x0}

    def _val(self, t):
        """Extract scalar mean from tensor or NpTensor."""
        if isinstance(t, _NpTensor):
            return float(t._d.mean())
        if hasattr(t, "mean"):
            v = t.mean()
            return float(v.item() if hasattr(v, "item") else v)
        return float(np.mean(t))

    def test_step0_nudges_x0_toward_sdr(self):
        sdr = _full((1, 4, 8, 8), 10.0)
        x0  = _zeros(1, 4, 8, 8)
        fn, _ = self._make_anchor(sdr, sdr_blend=0.4, sdr_inject_steps=5, sdr_decay=1.0)
        result = fn(self._args(x0))
        # expected mean = 0.6*0 + 0.4*10 = 4.0
        self.assertAlmostEqual(self._val(result), 4.0, places=4)

    def test_blend_decays_across_steps(self):
        sdr    = _full((1, 4, 4, 4), 100.0)
        x0     = _zeros(1, 4, 4, 4)
        blend0 = 0.5
        decay  = 0.7
        fn, _  = self._make_anchor(sdr, blend0, sdr_inject_steps=10, sdr_decay=decay)
        for step in range(5):
            result = fn(self._args(x0))
            expected = blend0 * (decay ** step) * 100.0
            self.assertAlmostEqual(self._val(result), expected, places=3,
                                   msg=f"step={step}")

    def test_step_counter_increments(self):
        sdr   = _zeros(1, 4, 4, 4)
        fn, counter = self._make_anchor(sdr, 0.3, sdr_inject_steps=10, sdr_decay=1.0)
        for _ in range(7):
            fn(self._args(_rand(1, 4, 4, 4)))
        self.assertEqual(counter[0], 7)

    def test_step_at_inject_boundary_is_passthrough(self):
        sdr    = _full((1, 4, 4, 4), 50.0)
        x0     = _rand(1, 4, 4, 4)
        inject = 3
        fn, _ = self._make_anchor(sdr, 0.5, inject, 1.0)
        for _ in range(inject):               # exhaust active range
            fn(self._args(_zeros(1, 4, 4, 4)))
        result = fn(self._args(x0))           # step == inject_steps
        self.assertTrue(_allclose(result, x0))

    def test_step_beyond_inject_is_passthrough(self):
        sdr = _full((1, 4, 4, 4), 50.0)
        x0  = _rand(1, 4, 4, 4)
        fn, _ = self._make_anchor(sdr, 0.5, sdr_inject_steps=2, sdr_decay=1.0)
        for _ in range(10):
            fn(self._args(_zeros(1, 4, 4, 4)))
        result = fn(self._args(x0))
        self.assertTrue(_allclose(result, x0))

    def test_inject_steps_zero_always_passthrough(self):
        sdr = _full((1, 4, 4, 4), 99.0)
        x0  = _rand(1, 4, 4, 4)
        fn, _ = self._make_anchor(sdr, 0.5, sdr_inject_steps=0, sdr_decay=1.0)
        result = fn(self._args(x0))
        self.assertTrue(_allclose(result, x0))

    def test_spatial_mismatch_returns_x0_unchanged(self):
        sdr = _full((1, 4, 8, 8), 5.0)     # 8×8
        x0  = _rand(1, 4, 16, 16)          # 16×16 — mismatch
        fn, _ = self._make_anchor(sdr, 0.5, sdr_inject_steps=10, sdr_decay=1.0)
        result = fn(self._args(x0))
        self.assertTrue(_allclose(result, x0))

    def test_batch_broadcast_in_anchor(self):
        sdr = _full((1, 4, 8, 8), 5.0)   # batch=1
        x0  = _zeros(4, 4, 8, 8)          # batch=4
        fn, _ = self._make_anchor(sdr, 0.4, sdr_inject_steps=5, sdr_decay=1.0)
        result = fn(self._args(x0))
        # expected = 0.6*0 + 0.4*5 = 2.0
        self.assertAlmostEqual(self._val(result), 2.0, places=4)

    def test_full_decay_schedule(self):
        sdr    = _ones(1, 4, 4, 4)
        x0     = _zeros(1, 4, 4, 4)
        blend0 = 0.5
        decay  = 0.8
        inject = 6
        fn, _ = self._make_anchor(sdr, blend0, inject, decay)
        for n in range(inject):
            result = fn(self._args(x0))
            expected = blend0 * (decay ** n)
            self.assertAlmostEqual(self._val(result), expected, places=5,
                                   msg=f"n={n}")


# ─────────────────────────────────────────────────────────────────────────────
# Shape / ndim helpers that work for both _NpTensor and real tensors
# ─────────────────────────────────────────────────────────────────────────────
def _shape(t):
    if isinstance(t, _NpTensor):
        return tuple(t._d.shape)
    s = t.shape
    return tuple(s) if not isinstance(s, tuple) else s


def _ndim(t):
    if isinstance(t, _NpTensor):
        return t._d.ndim
    return t.ndim


# ═════════════════════════════════════════════════════════════════════════════
#               _build_latent_meta — SDR section   (numpy-backed)
# ═════════════════════════════════════════════════════════════════════════════

class TestBuildLatentMetaSDR(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    def _meta(self, sdr_blend=0.0, sdr_inject_steps=0, sdr_decay=0.65):
        return json.loads(_build_latent_meta(
            detected_type="flux",
            steps=20,
            scheduler="simple",
            flux_shift=1.0,
            denoise=1.0,
            sigmas=_np_sigmas(1.0, 0.5, 0.0),
            ays_active=False,
            pag_active=False,
            noise_type="Gaussian",
            tile_mode=False,
            multi_cond_mode="Off",
            clip_target="Auto",
            seed=42,
            total_time_ms=1500,
            latent_format="video",
            frames=None,
            sdr_blend=sdr_blend,
            sdr_inject_steps=sdr_inject_steps,
            sdr_decay=sdr_decay,
        ))

    def test_no_sdr_block_when_blend_zero(self):
        meta = self._meta(sdr_blend=0.0)
        self.assertNotIn("sdr_conditioning", meta)

    def test_sdr_block_present_when_blend_positive(self):
        meta = self._meta(sdr_blend=0.35, sdr_inject_steps=6, sdr_decay=0.65)
        self.assertIn("sdr_conditioning", meta)

    def test_sdr_blend_correct(self):
        meta = self._meta(sdr_blend=0.4, sdr_inject_steps=5, sdr_decay=0.7)
        self.assertAlmostEqual(meta["sdr_conditioning"]["sdr_blend"], 0.4, places=4)

    def test_sdr_inject_steps_correct(self):
        meta = self._meta(sdr_blend=0.3, sdr_inject_steps=8)
        self.assertEqual(meta["sdr_conditioning"]["sdr_inject_steps"], 8)

    def test_sdr_decay_correct(self):
        meta = self._meta(sdr_blend=0.3, sdr_inject_steps=4, sdr_decay=0.8)
        self.assertAlmostEqual(meta["sdr_conditioning"]["sdr_decay"], 0.8, places=4)

    def test_valid_json_with_sdr(self):
        raw = _build_latent_meta(
            detected_type="ltxv", steps=10, scheduler="normal",
            flux_shift=1.0, denoise=0.85,
            sigmas=_np_sigmas(0.9, 0.4, 0.0),
            ays_active=False, pag_active=False, noise_type="Gaussian",
            tile_mode=False, multi_cond_mode="Off", clip_target="Auto",
            seed=0, total_time_ms=999, sdr_blend=0.35,
            sdr_inject_steps=6, sdr_decay=0.65,
        )
        parsed = json.loads(raw)
        self.assertIn("sdr_conditioning", parsed)


# ═════════════════════════════════════════════════════════════════════════════
#               INPUT_TYPES contract         (no torch needed)
# ═════════════════════════════════════════════════════════════════════════════

class TestInputTypesContract(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    @classmethod
    def setUpClass(cls):
        cls.optional = RadianceSamplerPro.INPUT_TYPES().get("optional", {})

    def test_sdr_reference_in_optional(self):
        self.assertIn("sdr_reference", self.optional)

    def test_sdr_vae_in_optional(self):
        self.assertIn("sdr_vae", self.optional)

    def test_sdr_blend_in_optional(self):
        self.assertIn("sdr_blend", self.optional)

    def test_sdr_inject_steps_in_optional(self):
        self.assertIn("sdr_inject_steps", self.optional)

    def test_sdr_decay_in_optional(self):
        self.assertIn("sdr_decay", self.optional)

    def test_sdr_blend_default_in_0_1(self):
        spec = self.optional["sdr_blend"]
        default = spec[1].get("default", 0.35) if (isinstance(spec, tuple) and
                  len(spec) > 1 and isinstance(spec[1], dict)) else 0.35
        self.assertGreater(default, 0.0)
        self.assertLess(default, 1.0)

    def test_sdr_inject_steps_default_positive(self):
        spec = self.optional["sdr_inject_steps"]
        default = spec[1].get("default", 6) if (isinstance(spec, tuple) and
                  len(spec) > 1 and isinstance(spec[1], dict)) else 6
        self.assertGreater(default, 0)

    def test_sdr_vae_type_is_vae(self):
        spec = self.optional["sdr_vae"]
        if isinstance(spec, tuple):
            self.assertEqual(spec[0], "VAE")

    def test_sdr_reference_type_is_image(self):
        spec = self.optional["sdr_reference"]
        if isinstance(spec, tuple):
            self.assertEqual(spec[0], "IMAGE")


if __name__ == "__main__":
    unittest.main()
