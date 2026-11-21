"""A refiner takes over where the base model stops, on the same schedule: shifted by flux_shift
like the base stage, and on sigmas_override when one is supplied."""
import json

import torch

from _sampler_harness import FakeModelPatcher, SampleCustomRecorder, make_latent, run_sampler


def _stages(**kw):
    rec = SampleCustomRecorder()
    refiner = FakeModelPatcher()
    run_sampler(recorder=rec, model_type="flux", scheduler="simple", steps=8,
                refiner_model=refiner, refiner_start_step=4, **kw)
    base = [c["sigmas"] for c in rec.calls if c["model"].model is not refiner.model]
    ref = [c["sigmas"] for c in rec.calls if c["model"].model is refiner.model]
    assert base and ref, "the refiner never ran"
    base, ref = base[-1], ref[0]
    assert torch.isclose(base[-1], ref[0]), (base, ref)
    return base, ref


def test_both_stages_follow_the_shifted_schedule():
    base, ref = _stages(flux_shift=1.0)
    plain = torch.cat([base, ref[1:]])
    expected = 2.0 * plain / (1.0 + plain)       # shift*s / (1 + (shift-1)*s) at shift 2
    base, ref = _stages(flux_shift=2.0)
    assert torch.allclose(torch.cat([base, ref[1:]]), expected), (base, ref)


def test_the_refiner_keeps_the_supplied_sigmas():
    override = torch.linspace(1.0, 0.0, 9) ** 3
    base, ref = _stages(sigmas_override=override)
    assert torch.allclose(torch.cat([base, ref[1:]]), override), (base, ref)


def _wan_stages(unet, refiner_start_step, **kw):
    rec = SampleCustomRecorder()
    refiner = FakeModelPatcher()
    meta = json.dumps({"arch": "wan", "unet_file": unet, "loras": []})
    run_sampler(latent=make_latent((1, 16, 2, 8, 8)), recorder=rec, preset="Auto", model_type="wan",
                model_meta=meta, scheduler="simple", steps=20, refiner_model=refiner,
                refiner_start_step=refiner_start_step, **kw)
    base = [c["sigmas"] for c in rec.calls if c["model"].model is not refiner.model]
    ref = [c["sigmas"] for c in rec.calls if c["model"].model is refiner.model]
    return base, ref


def test_wan22_low_noise_expert_takes_over_at_half_the_steps():
    # At the default start (20 of 20) the low-noise expert never ran; its templates hand over at 10.
    base, ref = _wan_stages("wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors", 20)
    assert ref and len(base[-1]) == 11 and len(ref[0]) == 11, (base, ref)
    base, ref = _wan_stages("wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors", 6)
    assert len(base[-1]) == 7, base


def test_wan22_hand_over_follows_a_supplied_schedule():
    # sigmas_override sets the step count (8 here, steps says 20): the hand-over is at 4 of 8.
    base, ref = _wan_stages("wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors", 20,
                            sigmas_override=torch.linspace(1.0, 0.0, 9))
    assert ref and len(base[-1]) == 5 and len(ref[0]) == 5, (base, ref)


def test_other_refiners_keep_their_start():
    base, ref = _wan_stages("wan2.1_t2v_14B_fp16.safetensors", 20)
    assert not ref, ref
