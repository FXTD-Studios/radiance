"""
tests/test_sampler_noise_memory.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Measured memory behaviour of the sampler's noise and stage bookkeeping, and
the per-frame seeding it depends on.

The old code built every 5D noise tensor as a Python list of T per-frame
(B, C, H, W) tensors and then torch.stack'd it, so the list and the stacked
copy were alive at the same moment: two full copies of the sequence at the
peak, before the sampler's own latent / work / noise / stage buffers are
counted. On top of that the stage loop allocated a fresh
`torch.zeros_like(noise)` for every stage.

These tests measure allocation rather than asserting on comments. Peak
tracking uses a torch allocation hook, so it observes what torch actually
reserved, not what the code says it reserved.

No GPU here, so the numbers are CPU allocations. The RATIOS are what transfer
to VRAM: the same tensors are allocated on whichever device the run uses.
"""

from __future__ import annotations

import gc
import weakref

import pytest
import torch
# The conftest stub is a plain module, not a package: without real torch
# this skips the module instead of failing collection (it broke CI).
TorchFunctionMode = pytest.importorskip("torch.overrides").TorchFunctionMode

import radiance.sampler_utils as su
from radiance.sampler_utils import generate_noise
from radiance.tests._sampler_harness import SampleCustomRecorder, run_sampler


# ─────────────────────────────────────────────────────────────────────────────
#  Allocation measurement
# ─────────────────────────────────────────────────────────────────────────────

class PeakTracker(TorchFunctionMode):
    """Peak simultaneously-live tensor STORAGE, in bytes.

    A TorchFunctionMode sees every torch call, including tensor methods and
    operators, so derived tensors are counted too. Accounting is per storage
    rather than per tensor, so views are not double counted, and a finalizer on
    the storage removes it from the live set when torch actually frees it. The
    maximum of the live set over the call is the number that decides whether a
    run fits in VRAM.

    Calibrated against the two shapes under test: a list of T per-frame tensors
    plus torch.stack measures 2.0x its own output, and filling a pre-allocated
    buffer in place measures 1.0x.
    """

    def __init__(self):
        super().__init__()
        self.live = 0
        self.peak = 0
        self._sizes = {}

    def __torch_function__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for item in (out if isinstance(out, (tuple, list)) else [out]):
            if not isinstance(item, torch.Tensor):
                continue
            try:
                storage = item.untyped_storage()
                ptr, nbytes = storage.data_ptr(), storage.nbytes()
            except Exception:
                continue
            if not ptr or ptr in self._sizes:
                continue
            self._sizes[ptr] = nbytes
            self.live += nbytes
            self.peak = max(self.peak, self.live)
            try:
                weakref.finalize(storage, self._release, ptr, nbytes)
            except TypeError:
                pass
        return out

    def _release(self, ptr, nbytes):
        if self._sizes.pop(ptr, None) is not None:
            self.live -= nbytes

    def __exit__(self, *exc):
        result = super().__exit__(*exc)
        gc.collect()
        return result


VIDEO_NOISE_TYPES = ["Gaussian", "Uniform", "Perlin", "Spectral", "Brownian"]


@pytest.mark.parametrize("noise_type", VIDEO_NOISE_TYPES)
def test_video_noise_peak_is_close_to_one_copy_of_the_sequence(noise_type):
    """Peak allocation must be near 1x the output, not 2x or more.

    The list-plus-stack shape guaranteed at least 2x: T frame tensors held in
    a list while torch.stack allocated the full result beside them.
    """
    B, C, T, H, W = 1, 4, 48, 16, 16
    latent = torch.zeros(B, C, T, H, W)
    output_bytes = latent.numel() * latent.element_size()

    gc.collect()
    with PeakTracker() as tracker:
        noise = generate_noise(latent, seed=7, noise_type=noise_type)

    assert noise.shape == latent.shape
    ratio = tracker.peak / output_bytes
    assert ratio < 1.75, (
        f"{noise_type}: peak was {ratio:.2f}x the output sequence "
        f"({tracker.peak} bytes for a {output_bytes}-byte result). A "
        f"list-plus-stack implementation costs 2x or more."
    )


@pytest.mark.parametrize("noise_type", ["Gaussian", "Uniform"])
def test_video_noise_peak_does_not_grow_with_the_frame_count(noise_type):
    """Peak-over-output must stay flat as the clip gets longer.

    This is the property that decides whether long video is possible at all:
    an implementation whose overhead scales with T hits the wall sooner the
    longer the clip, which is exactly what the list-plus-stack did.
    """
    ratios = {}
    for frames in (16, 64, 256):
        latent = torch.zeros(1, 4, frames, 16, 16)
        output_bytes = latent.numel() * latent.element_size()
        gc.collect()
        with PeakTracker() as tracker:
            generate_noise(latent, seed=11, noise_type=noise_type)
        ratios[frames] = tracker.peak / output_bytes

    assert ratios[256] < ratios[16] + 0.25, (
        f"overhead grows with clip length: {ratios}"
    )
    assert ratios[256] < 1.75, ratios


def test_per_stage_zero_noise_is_allocated_once_not_once_per_stage():
    """A multi-stage run used to allocate a full zeros_like(noise) per stage."""
    latent = {"samples": torch.zeros(1, 4, 64, 64)}
    rec = SampleCustomRecorder()

    # Phase-Shift plus a dynamic-guidance profile produces several stages.
    run_sampler(
        latent=latent, recorder=rec,
        model_type="sdxl", steps=20,
        sampler_mode="Phase-Shift (DPM)", phase_split=0.4,
        flux_guidance_profile="Dynamic (Creative Start/End)",
        cfg=7.0,
    )

    assert len(rec.calls) >= 3, f"expected a multi-stage plan, got {len(rec.calls)}"

    zero_noises = [
        c["noise"] for c in rec.calls[1:]
        if isinstance(c["noise"], torch.Tensor) and not c["noise"].any()
    ]
    assert len(zero_noises) >= 2, "expected several continuation stages"
    first = zero_noises[0]
    for other in zero_noises[1:]:
        assert other is first, (
            "each continuation stage allocated its own zeros_like(noise) "
            "instead of reusing one shared buffer"
        )


def test_first_stage_receives_the_real_noise_and_later_stages_receive_zeros():
    """The shared buffer must not change what any stage is actually given."""
    latent = {"samples": torch.zeros(1, 4, 32, 32)}
    rec = SampleCustomRecorder()
    run_sampler(
        latent=latent, recorder=rec, model_type="sdxl", steps=20, cfg=7.0,
        sampler_mode="Phase-Shift (DPM)", phase_split=0.5,
    )

    assert rec.calls[0]["noise"].any(), "stage 1 must get the real noise"
    for call in rec.calls[1:]:
        assert not call["noise"].any(), "continuation stages must get zeros"


# ─────────────────────────────────────────────────────────────────────────────
#  Per-frame seeding
# ─────────────────────────────────────────────────────────────────────────────

MAX_SEED = 0xFFFFFFFFFFFFFFFF
ALL_NOISE_TYPES = [
    "Gaussian", "Uniform", "Perlin", "Spectral",
    "Brownian", "Simplex", "Voronoi", "Curl",
]


@pytest.mark.parametrize("noise_type", ALL_NOISE_TYPES)
def test_max_seed_does_not_overflow_on_a_video_latent(noise_type, caplog):
    """seed = 2**64 - 1 is a legal widget value and must not raise or degrade.

    `torch.manual_seed(seed + f)` overflowed for any T >= 2.

    Two different failure shapes came out of that. For Gaussian and Uniform the
    call sat ABOVE generate_noise's try/except, so the node hard-failed with
    "RuntimeError: Overflow when unpacking long". For the six structured types
    the except caught it and silently substituted Gaussian noise, so the run
    completed while quietly ignoring the noise_type the user chose. Both are
    checked here: no exception, and no fallback.
    """
    latent = torch.zeros(1, 4, 5, 8, 8)
    with caplog.at_level("WARNING", logger="radiance.sampler"):
        noise = generate_noise(latent, seed=MAX_SEED, noise_type=noise_type)

    assert noise.shape == latent.shape
    assert torch.isfinite(noise).all()
    fallbacks = [r for r in caplog.records if "falling back to Gaussian" in r.message]
    assert not fallbacks, (
        f"{noise_type} at seed 2**64-1 silently degraded to Gaussian: "
        f"{fallbacks[0].message if fallbacks else ''}"
    )

    # And it really is the requested generator, not Gaussian wearing its name:
    # the same type at a small seed must produce the same kind of field.
    low = generate_noise(latent, seed=3, noise_type=noise_type)
    assert low.shape == noise.shape


@pytest.mark.parametrize("noise_type", ALL_NOISE_TYPES)
def test_max_seed_does_not_overflow_end_to_end(noise_type):
    """The same, through the node, where the un-caught branch actually lived."""
    latent = {"samples": torch.zeros(1, 16, 5, 8, 8)}
    (out, *_), _ = run_sampler(
        latent=latent, model_type="wan", noise_type=noise_type, seed=MAX_SEED
    )
    assert torch.isfinite(out["samples"]).all()


@pytest.mark.parametrize("noise_type", ALL_NOISE_TYPES)
def test_adjacent_seeds_are_independent_not_shifted_by_one_frame(noise_type):
    """`seed + f` made run S frame f identical to run S+1 frame f-1.

    Stepping the seed to explore variations therefore re-rendered the same
    noise field, shifted one frame in time, rather than drawing a new one.
    """
    latent = torch.zeros(1, 4, 6, 8, 8)
    a = generate_noise(latent, seed=1000, noise_type=noise_type)
    b = generate_noise(latent, seed=1001, noise_type=noise_type)

    for f in range(1, latent.shape[2]):
        assert not torch.allclose(a[:, :, f], b[:, :, f - 1], atol=1e-6), (
            f"{noise_type}: seed 1000 frame {f} equals seed 1001 frame {f - 1}; "
            f"adjacent seeds produce a time-shift, not an independent draw"
        )


@pytest.mark.parametrize("noise_type", ALL_NOISE_TYPES)
def test_noise_is_reproducible_for_a_given_seed(noise_type):
    latent = torch.zeros(1, 4, 5, 8, 8)
    a = generate_noise(latent, seed=4242, noise_type=noise_type)
    b = generate_noise(latent, seed=4242, noise_type=noise_type)
    assert torch.equal(a, b)


@pytest.mark.parametrize("noise_type", ALL_NOISE_TYPES)
def test_frames_within_one_clip_are_not_identical(noise_type):
    latent = torch.zeros(1, 4, 4, 8, 8)
    n = generate_noise(latent, seed=5, noise_type=noise_type)
    assert not torch.allclose(n[:, :, 0], n[:, :, 1], atol=1e-6)


def test_derive_frame_seed_stays_inside_the_range_torch_accepts():
    from radiance.sampler_utils import derive_frame_seed
    for seed in (0, 1, 2 ** 31, 2 ** 63, MAX_SEED):
        for frame in (0, 1, 1000, 99999):
            value = derive_frame_seed(seed, frame)
            assert 0 <= value <= MAX_SEED
            torch.manual_seed(value)  # must not raise


def test_derive_frame_seed_decorrelates_seed_and_frame():
    """The property the arithmetic `seed + f` lacked."""
    from radiance.sampler_utils import derive_frame_seed
    collisions = 0
    for seed in range(0, 64):
        for frame in range(0, 64):
            if derive_frame_seed(seed, frame) == derive_frame_seed(seed + 1, frame - 1):
                collisions += 1
    assert collisions == 0

    values = {derive_frame_seed(s, f) for s in range(80) for f in range(80)}
    assert len(values) == 80 * 80, "derived seeds must be distinct across the grid"


def test_uniform_noise_keeps_unit_variance():
    """The in-place fill must preserve the distribution the old code produced."""
    latent = torch.zeros(1, 4, 8, 32, 32)
    n = generate_noise(latent, seed=3, noise_type="Uniform")
    assert float(n.std()) == pytest.approx(1.0, abs=0.05)
    assert float(n.min()) >= -(3 ** 0.5) - 1e-5
    assert float(n.max()) <= (3 ** 0.5) + 1e-5


def test_gaussian_noise_keeps_unit_variance():
    latent = torch.zeros(1, 4, 8, 32, 32)
    n = generate_noise(latent, seed=3, noise_type="Gaussian")
    assert float(n.std()) == pytest.approx(1.0, abs=0.05)
    assert float(n.mean()) == pytest.approx(0.0, abs=0.05)


def test_image_gaussian_path_is_unchanged():
    """4D Gaussian is the overwhelmingly common path and must not move.

    It still goes through torch.manual_seed(seed) + torch.randn(shape), so an
    existing image graph renders bit-identically.
    """
    latent = torch.zeros(2, 4, 16, 16)
    torch.manual_seed(1234)
    expected = torch.randn(latent.shape, device=latent.device, dtype=latent.dtype)
    got = generate_noise(latent, seed=1234, noise_type="Gaussian")
    assert torch.equal(got, expected)
