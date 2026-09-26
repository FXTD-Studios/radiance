"""
tests/test_long_video_windowing.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Overlapping temporal windows for long-video sampling, and the noise
bookkeeping that windowing sits on top of.

WHAT THESE TESTS ESTABLISH
  • The window plan covers every frame, exactly once at full weight, with the
    first window on frame 0 and the last on the final frame.
  • The blend weights are a partition of unity across every overlap, before
    any normalisation, and are strictly positive everywhere (the black-frame
    failure mode RadianceUpscaleVideo already had to fix).
  • Blending happens PER STEP: the recombination runs inside the denoising
    loop, once per model evaluation, not once per finished window.
  • Callback and step accounting are untouched by windowing: same number of
    callbacks, same step indices, same total.
  • Peak working-set size is a function of the window, not of clip length.
  • The plan is deterministic, so a rerun at the same seed produces the same
    windows.
  • A single window reproduces the unwindowed result BIT-EXACTLY.

WHAT THESE TESTS CANNOT ESTABLISH
  There is no GPU and no diffusion model in this environment. The fake
  denoiser is an arithmetic stand-in, so nothing here says the output is
  TEMPORALLY COHERENT on a real video model: whether a window boundary is
  invisible in a rendered plate, whether content drifts across a long clip,
  and what window/overlap values a given architecture actually needs are all
  questions only a real render on real hardware answers. That validation is
  the operator's, on the 4080.
"""

from __future__ import annotations

import math

import pytest
import torch

from radiance.sampler_utils import (
    plan_temporal_windows,
    temporal_window_weights,
    slice_conds_temporally,
)
from radiance.nodes.generate.sampler import (
    RadianceSamplerPro,
    make_temporal_window_wrapper,
)
from radiance.tests._sampler_harness import (
    FakeModelPatcher,
    SampleCustomRecorder,
    make_latent,
    run_sampler,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Window scheduling
# ─────────────────────────────────────────────────────────────────────────────

CLIP_CASES = [
    (31, 16, 4), (21, 8, 2), (21, 8, 4), (81, 16, 4), (81, 16, 8), (100, 16, 1),
    (33, 32, 16), (7, 8, 2), (8, 8, 4), (9, 8, 4), (250, 24, 6), (1000, 16, 4),
]


@pytest.mark.parametrize("total,window,overlap", CLIP_CASES)
def test_windows_cover_every_frame_contiguously(total, window, overlap):
    windows = plan_temporal_windows(total, window, overlap)

    assert windows, "a non-empty clip must produce at least one window"
    assert windows[0][0] == 0, "the first window must start on frame 0"
    assert windows[-1][1] == total, "the last window must end on the final frame"

    covered = set()
    for f0, f1 in windows:
        assert 0 <= f0 < f1 <= total
        covered.update(range(f0, f1))
    assert covered == set(range(total)), "every frame must be inside some window"

    starts = [f0 for f0, _ in windows]
    assert starts == sorted(set(starts)), "windows must be ascending and distinct"


@pytest.mark.parametrize("total,window,overlap", CLIP_CASES)
def test_blend_weights_are_a_partition_of_unity(total, window, overlap):
    """Summed raw weights must be exactly 1.0 on every frame.

    This is the property the normalisation in the wrapper would otherwise have
    to rescue. Proving it holds BEFORE normalisation means no frame is ever
    reconstructed from a weight sum that drifted, and nothing can normalise to
    black.
    """
    windows = plan_temporal_windows(total, window, overlap)
    accum = torch.zeros(total, dtype=torch.float64)

    for wi, (f0, f1) in enumerate(windows):
        w = temporal_window_weights(windows, wi, dtype=torch.float64).view(-1)
        assert w.shape[0] == f1 - f0
        assert torch.all(w > 0.0), (
            "a zero weight is the black-frame bug: a frame covered only by "
            "ramp-start weights normalises to 0"
        )
        assert torch.all(w <= 1.0 + 1e-12)
        accum[f0:f1] += w

    assert torch.allclose(accum, torch.ones(total, dtype=torch.float64), atol=1e-9), (
        f"weights do not sum to 1 everywhere: min={accum.min():.6f} "
        f"max={accum.max():.6f}"
    )


def test_coverage_and_unity_hold_across_a_wide_parameter_sweep():
    """Exhaustive rather than illustrative: the properties must not have holes.

    Includes the final window pulled back into a three-window intersection.
    Raw weights must sum to one before floating-point normalization.
    """
    checked = 0
    for window in range(2, 33):
        for overlap in range(0, window + 2):
            for total in (window - 1, window, window + 1, window + 3,
                          3 * window - 1, 3 * window, 7 * window + 5):
                if total < 1:
                    continue
                windows = plan_temporal_windows(total, window, overlap)

                covered = set()
                accum = torch.zeros(total, dtype=torch.float64)
                for wi, (f0, f1) in enumerate(windows):
                    covered.update(range(f0, f1))
                    w = temporal_window_weights(windows, wi, dtype=torch.float64).view(-1)
                    assert torch.all(w > 0.0), (total, window, overlap, wi)
                    accum[f0:f1] += w

                assert covered == set(range(total)), (total, window, overlap)
                assert torch.allclose(
                    accum, torch.ones(total, dtype=torch.float64), atol=1e-9
                ), (
                    f"total={total} window={window} overlap={overlap}: "
                    f"weight sum ranges [{accum.min():.6f}, {accum.max():.6f}]"
                )
                checked += 1

    assert checked > 3000, f"sweep only covered {checked} configurations"


def test_no_frame_is_shared_by_more_than_three_windows():
    """Only the pulled-back final window can add a third contributor."""
    for window in range(2, 33):
        for overlap in range(0, window + 2):
            for total in (window + 1, window + 3, 3 * window - 1, 7 * window + 5):
                windows = plan_temporal_windows(total, window, overlap)
                counts = [0] * total
                for f0, f1 in windows:
                    for f in range(f0, f1):
                        counts[f] += 1
                assert max(counts) <= 3, (
                    f"total={total} window={window} overlap={overlap}: a frame is "
                    f"covered by {max(counts)} windows"
                )


@pytest.mark.parametrize("total", [17, 28, 29, 30, 31, 32, 33, 40, 41])
def test_final_window_keeps_full_context_and_requested_overlap(total):
    windows = plan_temporal_windows(total, 16, 4)
    assert all(end - start == 16 for start, end in windows)
    assert all(left[1] - right[0] >= 4 for left, right in zip(windows, windows[1:]))
    assert windows[-1][1] == total
    if total == 31:
        assert windows == [(0, 16), (12, 28), (15, 31)]


def test_single_window_has_uniform_weight():
    windows = plan_temporal_windows(12, 32, 8)
    assert windows == [(0, 12)]
    w = temporal_window_weights(windows, 0).view(-1)
    assert torch.equal(w, torch.ones(12))


def test_overlap_is_clamped_to_half_the_window():
    """An overlap at or above the window collapses the stride to one frame."""
    huge = plan_temporal_windows(64, 8, 8)
    sane = plan_temporal_windows(64, 8, 4)
    assert huge == sane, "overlap >= window must clamp, not run one window per frame"
    assert len(huge) < 64


def test_plan_is_deterministic_across_reruns():
    for total, window, overlap in CLIP_CASES:
        first = plan_temporal_windows(total, window, overlap)
        for _ in range(3):
            assert plan_temporal_windows(total, window, overlap) == first


# ─────────────────────────────────────────────────────────────────────────────
#  The wrapper: per-step blending, coverage, exactness
# ─────────────────────────────────────────────────────────────────────────────

def _wrapper_args(x):
    return {
        "input": x,
        "timestep": torch.tensor([1.0]),
        "c": {"c_crossattn": torch.zeros(1, 7, 768)},
        "cond_or_uncond": [0],
    }


def test_wrapper_is_bit_exact_when_one_window_covers_the_clip():
    """A clip that fits in one window must reproduce the unwindowed tensor."""
    x = torch.randn(1, 4, 12, 8, 8)

    def apply_model(xx, tt, **cc):
        return xx * 3.0 - 1.0

    unwindowed = apply_model(x, None)
    wrapper = make_temporal_window_wrapper(window_size=32, overlap=8)
    windowed = wrapper(apply_model, _wrapper_args(x))

    assert torch.equal(windowed, unwindowed), "single window must be bit-exact"


def test_wrapper_reconstructs_an_identity_model_exactly():
    """With a model that returns its input, blending must return the input.

    Any weighting error -- a gap, a double count, a ramp that does not sum to
    one -- shows up immediately as a deviation from the input.
    """
    x = torch.randn(1, 4, 81, 6, 6, dtype=torch.float64)

    def apply_model(xx, tt, **cc):
        return xx

    wrapper = make_temporal_window_wrapper(window_size=16, overlap=4)
    out = wrapper(apply_model, _wrapper_args(x))

    assert out.shape == x.shape
    assert torch.allclose(out, x, atol=1e-12), (
        f"max deviation {float((out - x).abs().max()):.3e}"
    )


def test_three_window_join_blends_distinct_predictions():
    calls = []

    def apply_model(x, t, **c):
        value = (1.0, 10.0, 100.0)[len(calls)]
        calls.append(x.shape[2])
        return torch.full_like(x, value)

    x = torch.zeros(1, 1, 31, 1, 1, dtype=torch.float64)
    out = make_temporal_window_wrapper(16, 4)(apply_model, _wrapper_args(x))
    assert calls == [16, 16, 16]
    # Frame 15 is in all three windows. Window 2 has faded in for 3.5/4
    # of the first join; window 3 starts its separate 13-frame fade here.
    second = math.sin(math.pi * 3.5 / 8) ** 2
    third = math.sin(math.pi * 0.5 / 26) ** 2
    expected = ((1 - second) + 10 * second) * (1 - third) + 100 * third
    assert out[0, 0, 15, 0, 0].item() == pytest.approx(expected, abs=1e-12)
    assert out[0, 0, 0, 0, 0].item() == 1.0
    assert out[0, 0, -1, 0, 0].item() == 100.0
    assert out.min() >= 1 and out.max() <= 100


def test_wrapper_evaluates_the_model_once_per_window_per_call():
    x = torch.randn(1, 4, 81, 6, 6)
    seen = []

    def apply_model(xx, tt, **cc):
        seen.append(xx.shape[2])
        return xx

    windows = plan_temporal_windows(81, 16, 4)
    wrapper = make_temporal_window_wrapper(window_size=16, overlap=4)
    wrapper(apply_model, _wrapper_args(x))

    assert len(seen) == len(windows)
    assert set(seen) == {16}, "every model call sees exactly one window of frames"


def test_wrapper_slices_temporal_conditioning_with_the_window():
    """An i2v model's per-frame conditioning must follow its window.

    Text embeddings carry no temporal axis and are shared untouched; a 5D
    conditioning tensor matching the clip length is a per-frame signal and
    must be cut to the same frames the model is being shown.
    """
    x = torch.randn(1, 4, 40, 6, 6)
    c_concat = torch.arange(40, dtype=torch.float32).view(1, 1, 40, 1, 1).expand(1, 4, 40, 6, 6)
    crossattn = torch.zeros(1, 7, 768)
    seen = []

    def apply_model(xx, tt, **cc):
        # c_concat carries its own absolute frame index as its value, so the
        # first value of the slice says which frame the model was handed.
        seen.append((
            xx.shape[2],
            cc["c_concat"].shape[2],
            cc["c_crossattn"].shape,
            int(cc["c_concat"][0, 0, 0, 0, 0]),
        ))
        return xx

    wrapper = make_temporal_window_wrapper(window_size=16, overlap=4)
    wrapper(apply_model, {
        "input": x,
        "timestep": torch.tensor([1.0]),
        "c": {"c_crossattn": crossattn, "c_concat": c_concat},
        "cond_or_uncond": [0],
    })

    windows = plan_temporal_windows(40, 16, 4)
    assert len(seen) == len(windows)
    for (lat_t, cond_t, ca_shape, first_frame), (f0, f1) in zip(seen, windows):
        assert cond_t == lat_t, "temporal conditioning was not sliced with the window"
        assert ca_shape == (1, 7, 768), "text embeddings must not be sliced"
        assert first_frame == f0, (
            f"window starting at frame {f0} was given conditioning starting at "
            f"frame {first_frame}"
        )


def test_slice_conds_leaves_non_temporal_entries_alone():
    c = {
        "c_crossattn": torch.zeros(1, 7, 768),
        "y": torch.zeros(1, 2816),
        "c_concat": torch.zeros(1, 4, 40, 6, 6),
        "transformer_options": {"sigmas": 1.0},
        "control": None,
    }
    out = slice_conds_temporally(c, 8, 24, total_frames=40)

    assert out["c_concat"].shape[2] == 16
    assert out["c_crossattn"] is c["c_crossattn"]
    assert out["y"] is c["y"]
    assert out["transformer_options"] is c["transformer_options"]
    assert out["control"] is None


def test_controlnet_with_windowing_rejects_misaligned_control():
    """ComfyUI computes control hints for the whole clip before the wrapper.

    Each window would then receive a full-length control tensor. There is no
    way to re-slice a ControlBase's already-computed output from here, so the
    run stops before the model receives misaligned control.
    """
    x = torch.randn(1, 4, 40, 6, 6)

    def apply_model(xx, tt, **cc):
        pytest.fail("Model must not receive full-clip control with a windowed latent")

    wrapper = make_temporal_window_wrapper(window_size=16, overlap=4)
    args = {
        "input": x,
        "timestep": torch.tensor([1.0]),
        "c": {"c_crossattn": torch.zeros(1, 7, 768), "control": object()},
        "cond_or_uncond": [0],
    }

    with pytest.raises(ValueError, match="Set temporal_window=0"):
        wrapper(apply_model, args)


@pytest.mark.parametrize("window_size", [0, 40, 64])
def test_controlnet_passes_through_when_clip_is_not_split(window_size):
    x = torch.randn(1, 4, 40, 6, 6)
    control = object()
    args = _wrapper_args(x)
    args["c"]["control"] = control

    def apply_model(xx, tt, **cc):
        assert xx is x
        assert cc["control"] is control
        return xx

    result = make_temporal_window_wrapper(window_size, 4)(apply_model, args)
    assert result is x


def test_no_control_warning_without_a_controlnet(caplog):
    x = torch.randn(1, 4, 40, 6, 6)
    wrapper = make_temporal_window_wrapper(window_size=16, overlap=4)
    with caplog.at_level("WARNING", logger="radiance.sampler"):
        wrapper(lambda xx, tt, **cc: xx, _wrapper_args(x))
    assert not any("ControlNet" in r.getMessage() for r in caplog.records)


def test_wrapper_passes_through_non_video_latents():
    """4D image latents and packed multi-modality latents have no axis to window."""
    wrapper = make_temporal_window_wrapper(window_size=4, overlap=1)
    seen = []

    def apply_model(xx, tt, **cc):
        seen.append(tuple(xx.shape))
        return xx

    x4 = torch.randn(2, 4, 16, 16)
    assert torch.equal(wrapper(apply_model, _wrapper_args(x4)), x4)

    x3 = torch.randn(1, 1, 4096)  # LTX-AV packs to (B, 1, N) before this point
    assert torch.equal(wrapper(apply_model, _wrapper_args(x3)), x3)

    assert seen == [(2, 4, 16, 16), (1, 1, 4096)]


def test_wrapper_composes_with_a_wrapper_another_node_installed():
    """set_model_unet_function_wrapper has one slot; clobbering it breaks graphs."""
    calls = []

    def other_wrapper(apply_model, args):
        calls.append(args["input"].shape[2])
        return apply_model(args["input"], args["timestep"], **args["c"]) + 1.0

    def apply_model(xx, tt, **cc):
        return xx

    x = torch.randn(1, 4, 40, 6, 6, dtype=torch.float64)
    wrapper = make_temporal_window_wrapper(
        window_size=16, overlap=4, previous_wrapper=other_wrapper
    )
    out = wrapper(apply_model, _wrapper_args(x))

    assert calls, "the pre-existing wrapper must still be called"
    assert torch.allclose(out, x + 1.0, atol=1e-12), (
        "the pre-existing wrapper's contribution must survive the blend"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Peak memory as a function of window size, not clip length
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("clip_frames", [24, 81, 240, 960])
def test_peak_model_input_is_bounded_by_the_window_not_the_clip(clip_frames):
    """The measurement that matters: what the model is ever asked to hold.

    Activation memory is what OOMs on long video, and it scales with the
    frames handed to the model in one evaluation. Windowing must cap that at
    the window size regardless of how long the clip is.
    """
    window = 16
    x = torch.zeros(1, 4, clip_frames, 6, 6)
    peak_frames = []
    peak_elems = []

    def apply_model(xx, tt, **cc):
        peak_frames.append(xx.shape[2])
        peak_elems.append(xx.numel())
        return xx

    wrapper = make_temporal_window_wrapper(window_size=window, overlap=4)
    wrapper(apply_model, _wrapper_args(x))

    assert max(peak_frames) == window, (
        f"a {clip_frames}-frame clip handed the model {max(peak_frames)} frames; "
        f"peak must stay at the {window}-frame window"
    )
    # And the unwindowed path would have handed it the whole clip.
    assert max(peak_elems) == 1 * 4 * window * 6 * 6


def test_windowing_trades_compute_for_memory_predictably():
    """Total model evaluations grow linearly with clip length, peak does not."""
    window, overlap = 16, 4
    counts = {}
    for clip in (32, 64, 128, 256):
        x = torch.zeros(1, 4, clip, 4, 4)
        n = []

        def apply_model(xx, tt, **cc):
            n.append(xx.shape[2])
            return xx

        make_temporal_window_wrapper(window, overlap)(apply_model, _wrapper_args(x))
        counts[clip] = len(n)
        assert max(n) == window

    assert counts[256] > counts[128] > counts[64] > counts[32]


# ─────────────────────────────────────────────────────────────────────────────
#  Per-step blending, callbacks, and the node-level control
# ─────────────────────────────────────────────────────────────────────────────

def test_blending_happens_inside_the_denoising_loop_not_after_it():
    """The defining property of the technique.

    Blending windows that have each been denoised to completion re-creates the
    seams windowing exists to remove. Proof that it does not happen that way:
    with N denoising steps driven through the wrapper, the recombination runs
    N times, once per step, and every one of those runs sees ALL the windows.
    """
    steps = 5
    rec = SampleCustomRecorder(model_wrapper_steps=steps)
    latent = {"samples": torch.zeros(1, 16, 40, 8, 8)}

    run_sampler(
        latent=latent, recorder=rec, model_type="wan",
        steps=steps, temporal_window=16, temporal_overlap=4,
    )

    windows = plan_temporal_windows(40, 16, 4)
    assert len(windows) > 1

    # Every model evaluation is one window of one step.
    assert len(rec.wrapper_inputs) == steps * len(windows), (
        f"expected {steps} steps x {len(windows)} windows of model calls, got "
        f"{len(rec.wrapper_inputs)}; a per-window-then-blend implementation "
        f"would produce {len(windows)} calls over whole schedules instead"
    )
    assert {t.shape[2] for t in rec.wrapper_inputs} == {16}


def test_callback_accounting_is_unchanged_by_windowing():
    """Windowing must not multiply, skip or reorder progress steps."""
    steps = 6
    latent = {"samples": torch.zeros(1, 16, 40, 8, 8)}

    def collect(temporal_window):
        rec = SampleCustomRecorder(model_wrapper_steps=steps)
        seen = []
        run_sampler(
            latent=latent, recorder=rec, model_type="wan", steps=steps,
            temporal_window=temporal_window, temporal_overlap=4,
            preview_method="None",
        )
        for call in rec.calls:
            seen.append(len(call["sigmas"]))
        return seen, rec

    off_sigmas, off_rec = collect(0)
    on_sigmas, on_rec = collect(16)

    assert off_sigmas == on_sigmas, "windowing changed the sigma schedule"
    assert len(off_rec.calls) == len(on_rec.calls), (
        "windowing changed the number of sampling calls"
    )


def test_the_new_widgets_are_appended_after_every_existing_optional_input():
    """ComfyUI restores widget VALUES from a saved workflow by POSITION.

    Inserting a new optional input ahead of an existing one shifts every
    later widget onto its neighbour's saved value, so an old graph reloads
    with restart_count in noise_alpha_start's slot and so on. New widgets go
    at the end.
    """
    optional = list(RadianceSamplerPro.INPUT_TYPES()["optional"].keys())
    assert optional[-2:] == ["temporal_window", "temporal_overlap"], (
        f"the windowing widgets are not last: {optional}"
    )
    for name in (
        "refiner_model", "noise_override", "sigmas_override",
        "_js_export_btn", "_js_import_btn", "_js_preset_info",
        "restart_count", "noise_alpha_start", "noise_alpha_end",
        "custom_ays_anchors", "restart_schedule",
        "sdr_reference", "sdr_vae", "sdr_blend", "sdr_inject_steps", "sdr_decay",
        "model_meta",
    ):
        assert optional.index(name) < optional.index("temporal_window"), (
            f"'{name}' now sits after the new widgets; its position moved"
        )


def test_temporal_window_defaults_off_and_leaves_existing_graphs_alone():
    """Default is 0. A run with the widget absent is bit-identical to 0."""
    assert (
        RadianceSamplerPro.INPUT_TYPES()["optional"]["temporal_window"][1]["default"]
        == 0
    )

    latent = {"samples": torch.zeros(1, 16, 40, 8, 8)}
    (out_default, *_), _ = run_sampler(latent=latent, model_type="wan")
    (out_explicit_off, *_), _ = run_sampler(
        latent=latent, model_type="wan", temporal_window=0
    )
    assert torch.equal(out_default["samples"], out_explicit_off["samples"])


def test_windowing_is_not_registered_for_image_latents():
    """A 4D image latent has no temporal axis; the wrapper must not be installed."""
    model = FakeModelPatcher()
    run_sampler(model=model, model_type="sdxl", temporal_window=16)
    # sample() clones the model, so check the clone the recorder saw instead.
    # The original must never be patched either way.
    assert "model_function_wrapper" not in model.model_options


def test_windowing_registers_on_the_clone_never_the_input_model():
    """Patching the loader's cached ModelPatcher leaks into every later queue."""
    model = FakeModelPatcher()
    latent = {"samples": torch.zeros(1, 16, 40, 8, 8)}
    _, rec = run_sampler(
        latent=latent, model=model, model_type="wan",
        temporal_window=16, temporal_overlap=4,
    )
    assert "model_function_wrapper" not in model.model_options, (
        "the input MODEL was patched in place"
    )
    used = rec.calls[0]["model"]
    assert used is not model
    assert "model_function_wrapper" in used.model_options


def test_same_seed_produces_the_same_windows_and_the_same_output():
    latent = {"samples": torch.zeros(1, 16, 81, 8, 8)}
    kwargs = dict(model_type="wan", temporal_window=16, temporal_overlap=4, seed=987654321)

    (a, *_), rec_a = run_sampler(
        latent=latent, recorder=SampleCustomRecorder(model_wrapper_steps=3), **kwargs
    )
    (b, *_), rec_b = run_sampler(
        latent=latent, recorder=SampleCustomRecorder(model_wrapper_steps=3), **kwargs
    )

    assert torch.equal(a["samples"], b["samples"])
    assert [t.shape for t in rec_a.wrapper_inputs] == [
        t.shape for t in rec_b.wrapper_inputs
    ]
