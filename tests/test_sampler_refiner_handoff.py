"""A refiner takes over where the base model stops, on the same schedule: shifted by flux_shift
like the base stage, and on sigmas_override when one is supplied."""
import torch

from _sampler_harness import FakeModelPatcher, SampleCustomRecorder, run_sampler


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
