"""
tests/test_vae_streaming_memory.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Clip length must be bounded by disk, not by memory.

The HDR VAE path used to hold the whole clip several times over:

  * video ENCODE accumulated every frame's latent in a list, on the GPU, then
    torch.stack allocated a second full-clip copy. 4K at 16 channels is about
    8.3 MB per latent frame, so 500 frames cost ~4.1 GB resident plus ~4.1 GB
    for the stack.
  * video DECODE accumulated every decoded frame in a list then torch.cat.
    4K fp32 RGB is about 106 MB per frame, so 500 frames cost ~53 GB plus
    another ~53 GB for the cat, and with hdr_mode="Compress (Log)" plus
    export_rhdr a second full-clip list of scene-linear frames doubled it again.
  * SDR->HDR Universal kept roughly five full-size RGB copies resident at once
    (the input clone, `lin`, `expanded_hdr`, `hdr` and `out`), about 125 MB per
    1080p frame.

The nodes' contract is to return an IMAGE batch (decode, uplift) or a latent
(encode) to ComfyUI, so ONE clip-sized tensor is unavoidable. The claim these
tests measure is everything above that one buffer: it must be flat in frame
count.

A streaming claim nobody measures is exactly the failure this repo is
correcting, so these tests measure rather than inspect. PeakTracker below is a
TorchDispatchMode that sees every ATen call, registers a weakref to each output
tensor and recomputes live storage bytes, so `peak` is a true peak across the
whole call and not a sample taken at a convenient moment.

Measured on this suite's geometry, before and after the streaming rework:

                     overhead at 4 frames   overhead at 32 frames
    decode, before          4.915 MB               25.166 MB   (one whole clip)
    decode, after           5.702 MB                5.702 MB   (exactly flat)
    encode, before          1.360 MB                4.260 MB
    encode, after           1.491 MB                1.491 MB   (exactly flat)

`overhead` is peak live bytes minus the returned result minus the caller's own
input, so it is the working memory the node adds on top of what it was asked
to produce and what it was handed.
"""

from __future__ import annotations

import gc
import importlib
import os
import sys
import types
import weakref

import pytest

torch = pytest.importorskip("torch")

# Real torch only: the conftest stub has no submodules, and a bare import
# here failed collection on the lightweight CI lane.
TorchDispatchMode = pytest.importorskip("torch.utils._python_dispatch").TorchDispatchMode  # noqa: E402

RADIANCE_TORCH_GATED = True


# ── ComfyUI stubs ────────────────────────────────────────────────────────────

for _m in ["folder_paths", "comfy", "comfy.utils", "comfy.model_management"]:
    if _m not in sys.modules:
        sys.modules[_m] = types.ModuleType(_m)

_mm = sys.modules["comfy.model_management"]
if not hasattr(_mm, "get_torch_device"):
    _mm.get_torch_device = lambda: torch.device("cpu")
if not hasattr(_mm, "soft_empty_cache"):
    _mm.soft_empty_cache = lambda: None

_cu = sys.modules["comfy.utils"]
if not hasattr(_cu, "ProgressBar"):
    class _FakeProgressBar:
        def __init__(self, *a, **kw):
            pass

        def update(self, *a):
            pass

        def update_absolute(self, *a):
            pass

    _cu.ProgressBar = _FakeProgressBar

sys.modules["comfy"].model_management = sys.modules["comfy.model_management"]
sys.modules["comfy"].utils = sys.modules["comfy.utils"]

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def _vae_mod():
    return importlib.import_module("radiance.hdr.vae")


# ── The instrument ───────────────────────────────────────────────────────────

class PeakTracker(TorchDispatchMode):
    """Peak live torch storage bytes across every ATen call in the block.

    Every op output is registered by weakref together with its storage pointer
    and size. After each op the registry is pruned of dead tensors and the live
    total recomputed over distinct storages, so views and aliases are counted
    once. Pruning keeps the registry the size of the live set, so the per-op
    cost is proportional to how many tensors are alive, not to how many have
    ever existed.
    """

    def __init__(self):
        super().__init__()
        self._refs = []
        self.peak = 0
        self.live = 0

    def _register(self, out):
        stack = [out]
        while stack:
            obj = stack.pop()
            if isinstance(obj, torch.Tensor):
                try:
                    storage = obj.untyped_storage()
                    ptr, nbytes = storage.data_ptr(), storage.nbytes()
                except Exception:
                    continue
                if ptr:
                    self._refs.append((weakref.ref(obj), ptr, nbytes))
            elif isinstance(obj, (list, tuple)):
                stack.extend(obj)

    def _recompute(self):
        alive, seen, total = [], set(), 0
        for ref, ptr, nbytes in self._refs:
            if ref() is None:
                continue
            alive.append((ref, ptr, nbytes))
            if ptr not in seen:
                seen.add(ptr)
                total += nbytes
        self._refs = alive
        self.live = total
        if total > self.peak:
            self.peak = total

    def __torch_dispatch__(self, func, types_, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        self._register(out)
        self._recompute()
        return out


def test_peak_tracker_separates_the_two_shapes():
    """The instrument must be calibrated before it is trusted.

    A list-then-cat accumulation has overhead equal to the result (linear in
    frame count). A pre-allocate-and-write accumulation has overhead equal to
    one frame (flat). If this test ever fails, every measurement below means
    nothing.
    """
    frame_bytes = 1 * 32 * 32 * 3 * 4

    def _list_then_cat(n):
        with PeakTracker() as tracker:
            frames = [torch.zeros(1, 32, 32, 3) for _ in range(n)]
            out = torch.cat(frames, dim=0)
            del frames
            return tracker.peak, out.numel() * 4

    def _prealloc(n):
        with PeakTracker() as tracker:
            out = None
            for i in range(n):
                frame = torch.zeros(1, 32, 32, 3)
                if out is None:
                    out = torch.empty((n, 32, 32, 3))
                out[i] = frame
                del frame
            return tracker.peak, out.numel() * 4

    cat_overhead = [peak - res for peak, res in (_list_then_cat(n) for n in (4, 32))]
    pre_overhead = [peak - res for peak, res in (_prealloc(n) for n in (4, 32))]

    assert cat_overhead[1] > 4 * cat_overhead[0], (
        f"list+cat overhead should grow with frame count, got {cat_overhead}")
    assert pre_overhead[0] == pre_overhead[1] == frame_bytes, (
        f"pre-allocated overhead should be exactly one frame, got {pre_overhead}")


# ── Fakes ────────────────────────────────────────────────────────────────────

class _FrameVAE:
    """A 4D image VAE: the case encode()/decode() handle with a frame loop.

    Deliberately not a MagicMock: both paths branch on real attribute absence
    (`hasattr(vae, "latent_dim")` decides the 3D-native route) and a MagicMock
    hands out a truthy attribute for every name asked of it.

    128 latent channels is a real count (LTX-Video, Flux2), chosen here so the
    latent accumulation the encode test is measuring is not swamped by the
    per-frame pixel working set.
    """

    factor = 8
    latent_channels = 128

    def __init__(self):
        self.downscale_ratio = self.factor

    def decode(self, latent):
        b, _c, h, w = latent.shape
        return torch.zeros(b, h * self.factor, w * self.factor, 3)

    def encode(self, pixels):
        b, h, w, _c = pixels.shape
        return torch.zeros(
            b, self.latent_channels, h // self.factor, w // self.factor)


SHORT, LONG = 4, 32


def _decode_overhead(num_frames, lat_hw=32):
    """Peak live bytes above (returned batch + caller's latent)."""
    node = _vae_mod().RadianceVAE4KDecode()
    vae = _FrameVAE()
    latent = torch.zeros(1, _FrameVAE.latent_channels, num_frames, lat_hw, lat_hw)
    input_bytes = latent.numel() * latent.element_size()
    gc.collect()
    with PeakTracker() as tracker:
        img, _meta, _fmt = node.decode({"samples": latent}, vae)
        result_bytes = img.numel() * img.element_size()
    frame_bytes = result_bytes / num_frames
    del node, vae, latent, img
    gc.collect()
    return tracker.peak - result_bytes - input_bytes, result_bytes, frame_bytes


def _encode_overhead(num_frames, pix_hw=128):
    node = _vae_mod().RadianceVAE4KEncode()
    vae = _FrameVAE()
    pixels = torch.zeros(1, num_frames, pix_hw, pix_hw, 3)
    input_bytes = pixels.numel() * pixels.element_size()
    gc.collect()
    with PeakTracker() as tracker:
        latent, _alpha, _meta, _fmt, _qr = node.encode(pixels, vae)
        samples = latent["samples"]
        result_bytes = samples.numel() * samples.element_size()
    del node, vae, pixels, latent, samples
    gc.collect()
    return tracker.peak - result_bytes - input_bytes, result_bytes


# ── The measurements ─────────────────────────────────────────────────────────

def test_video_decode_working_memory_is_flat_in_frame_count():
    """Peak above the returned IMAGE batch must not grow with clip length.

    With the list + torch.cat accumulation this was one extra whole clip:
    measured 4.915 MB of overhead at 4 frames and 25.166 MB at 32, i.e.
    exactly the size of the returned batch. Duration was therefore a RAM
    budget. The streaming loop measures 5.702 MB at both lengths.
    """
    over_short, res_short, frame_bytes = _decode_overhead(SHORT)
    over_long, res_long, _ = _decode_overhead(LONG)

    assert res_long == res_short * (LONG // SHORT), (
        "the returned batch must itself scale with frame count, or this test "
        "is measuring the wrong thing")
    assert over_long <= over_short * 1.25, (
        f"decode working memory grew with clip length: "
        f"{over_short / 1e6:.3f} MB at {SHORT} frames, "
        f"{over_long / 1e6:.3f} MB at {LONG} frames. It must be a function of "
        f"the window, not the clip.")
    # And the ceiling is a handful of frames of colour-pipeline working set,
    # not a second clip.
    assert over_long < 0.5 * res_long, (
        f"decode overhead {over_long / 1e6:.3f} MB is more than half the "
        f"returned batch ({res_long / 1e6:.3f} MB)")


def test_video_encode_working_memory_is_flat_in_frame_count():
    """Same claim for encode: the returned latent is the only clip.

    The list of per-frame latents plus torch.stack cost two full clips of
    headroom, both in VRAM. Measured overhead was 1.360 MB at 4 frames and
    4.260 MB at 32; the streaming loop measures 1.491 MB at both.
    """
    over_short, res_short = _encode_overhead(SHORT)
    over_long, res_long = _encode_overhead(LONG)

    assert res_long == res_short * (LONG // SHORT)
    assert over_long <= over_short * 1.25, (
        f"encode working memory grew with clip length: "
        f"{over_short / 1e6:.3f} MB at {SHORT} frames, "
        f"{over_long / 1e6:.3f} MB at {LONG} frames")


def test_decoded_frames_are_released_as_the_loop_advances():
    """The per-frame buffers must die, not pile up behind the clip buffer.

    Complements the byte measurement with a direct liveness count: at the point
    the VAE is asked for frame k, only a couple of single-frame IMAGE tensors
    should exist. Under the old code `decoded_frames` held k of them, so this
    count WAS the frame index.
    """
    node = _vae_mod().RadianceVAE4KDecode()
    frames, lat_hw = 24, 16
    pix = lat_hw * _FrameVAE.factor
    counts = []

    class _CountingVAE(_FrameVAE):
        def decode(self, latent):
            gc.collect()
            counts.append(sum(
                1 for obj in gc.get_objects()
                if type(obj) is torch.Tensor
                and tuple(obj.shape) == (1, pix, pix, 3)))
            return super().decode(latent)

    latent = torch.zeros(1, _FrameVAE.latent_channels, frames, lat_hw, lat_hw)
    node.decode({"samples": latent}, _CountingVAE())

    assert len(counts) == frames
    assert max(counts) <= 3, (
        f"single-frame tensors alive per decode call: {counts}; they are "
        f"being retained instead of written into the clip buffer")
    assert counts[-1] <= counts[0] + 1, (
        f"liveness grew across the clip: first {counts[0]}, last {counts[-1]}")


def test_uplift_universal_holds_one_full_size_copy_at_the_output_encode():
    """SDR->HDR Universal's accumulation must be the only accumulation.

    `convert` used to keep roughly five full-size RGB copies resident at once,
    about 125 MB per 1080p frame. Counted at the moment the output encode runs
    (the last full-size allocation in the method), the old code had five
    full-size float32 tensors of the frame shape alive: the caller's IMAGE, the
    node's clone of it, `lin`, `expanded_hdr` and `hdr`. It is now two: the
    caller's IMAGE, which is not the node's to free, and `hdr` itself.
    """
    mod = importlib.import_module("radiance.nodes.hdr.uplift_universal")
    node = mod.RadianceSDRToHDRUniversal()
    image = torch.rand(2, 128, 128, 3)
    frame_bytes = image.numel() * image.element_size()
    shape = tuple(image.shape)
    live_at_encode = []

    original = mod._encode_output

    def _counting(hdr, *args, **kwargs):
        gc.collect()
        live_at_encode.append(sum(
            1 for obj in gc.get_objects()
            if type(obj) is torch.Tensor
            and tuple(obj.shape) == shape
            and obj.dtype == torch.float32))
        return original(hdr, *args, **kwargs)

    mod._encode_output = _counting
    try:
        gc.collect()
        with PeakTracker() as tracker:
            out = node.convert(
                image, "sRGB", 1000.0, "adaptive", 0.75, 1.6, 0.0,
                "Linear Rec.709", processing_mode="Expand",
            )[0]
    finally:
        mod._encode_output = original

    assert out.shape == image.shape
    assert live_at_encode == [2], (
        f"full-size tensors alive when the output is encoded: "
        f"{live_at_encode}; expected the caller's input and `hdr` only")
    # Peak covers the whole method, including the intermediate the expansion
    # maths genuinely needs. Measured 7.33 full-size copies before, 6.67 after.
    assert tracker.peak <= 7.0 * frame_bytes, (
        f"peak {tracker.peak / frame_bytes:.2f} full-size copies")
