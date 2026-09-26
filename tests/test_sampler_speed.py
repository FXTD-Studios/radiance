"""Sampler speed defects found in the 3.5 live logs.

1. cfg 1.0 was replaced by the architecture's base CFG in the Auto preset
   whenever model_meta was absent, so a turbo / distilled checkpoint ran an
   extra unconditional forward pass on every step (and the wrong look).
2. Dynamic CFG boosted cfg 1.0 to 1.2 early on, switching that pass on.
3. Every stage called load_model_gpu() before sample_custom, which loads the
   model again itself; the live log shows each model "prepared" twice.
4. log_tensor built its stats (float copy + 4 GPU syncs) even with DEBUG off.
5. gc.collect() + empty_cache() before sampling cost ~0.5 s per run.
"""
import inspect
import json
import logging
import re
from unittest.mock import MagicMock

import torch
import pytest

from _sampler_harness import run_sampler
import radiance.sampler_utils as su
from radiance.nodes.generate.sampler import RadianceSamplerPro


@pytest.mark.parametrize("preset", ["Auto", "Custom"])
@pytest.mark.parametrize("extra_shift", [1.0, 2.0])
def test_auto_preserves_native_schedule_and_explicit_extra_shift(monkeypatch, preset, extra_shift):
    # ModelSamplingSD3 has already shifted this schedule by 8. Applying the
    # architecture default again would turn sigma 0.8 into ~0.970, not 0.8.
    native_sigmas = torch.tensor([1.0, 0.95, 0.8, 0.3, 0.0])
    monkeypatch.setattr(su.comfy.samplers, "calculate_sigmas",
                        lambda model, scheduler, steps: native_sigmas.clone())
    _, recorder = run_sampler(preset=preset, model_type="wan", steps=4,
                              scheduler="simple", flux_shift=extra_shift, cfg=5.0)
    expected = extra_shift * native_sigmas / (1 + (extra_shift - 1) * native_sigmas)
    assert len(recorder.calls) == 1
    torch.testing.assert_close(recorder.calls[0]["sigmas"], expected)


def test_auto_keeps_cfg_1_without_model_meta():
    _, rec = run_sampler(preset="Auto", model_type="z_image", cfg=1.0, steps=8)
    assert all(c["cfg"] == 1.0 for c in rec.calls)


def test_auto_applies_base_cfg_when_model_meta_names_the_checkpoint():
    meta = json.dumps({"arch": "z_image", "unet_file": "z_image_bf16.safetensors"})
    _, rec = run_sampler(preset="Auto", model_type="auto", model_meta=meta, cfg=1.0, steps=8)
    assert rec.calls[0]["cfg"] == 4.0


def test_auto_keeps_turbo_at_cfg_1_with_model_meta():
    meta = json.dumps({"arch": "z_image", "unet_file": "z_image_turbo_bf16.safetensors"})
    _, rec = run_sampler(preset="Auto", model_type="auto", model_meta=meta, cfg=1.0, steps=8)
    assert rec.calls[0]["cfg"] == 1.0


def test_dynamic_cfg_does_not_turn_on_the_uncond_pass():
    for step in range(0, 20):
        assert su.compute_dynamic_cfg(1.0, step, 20, 1.0) == 1.0
    assert su.compute_dynamic_cfg(5.0, 0, 20, 1.0) > 5.0      # still shapes real CFG


@pytest.mark.parametrize("model_type,early,late", [("flux", 0.6, 0.95), ("sdxl", 1.2, 0.7)])
@pytest.mark.parametrize("denoise", [1.0, 0.5])
def test_dynamic_stages_reach_the_guidance_targets(model_type, early, late, denoise):
    _, recorder = run_sampler(model_type=model_type, steps=40, denoise=denoise,
                              cfg=5.0, flux_guidance=5.0,
                              flux_guidance_profile="Dynamic (Creative Start/End)")
    values = [call["positive"][0][1]["guidance"] if model_type == "flux" else call["cfg"]
              for call in recorder.calls]
    assert values[0] == pytest.approx(5.0 * early)
    assert any(value == pytest.approx(5.0) for value in values), values
    assert values[-1] == pytest.approx(5.0 * late)
    for left, right in zip(recorder.calls, recorder.calls[1:]):
        torch.testing.assert_close(left["sigmas"][-1], right["sigmas"][0])


@pytest.mark.parametrize("model_type", ["flux", "sdxl"])
def test_dynamic_partial_range_stays_within_the_requested_steps(model_type):
    _, recorder = run_sampler(model_type=model_type, steps=40, start_step=12, end_step=28,
                              cfg=5.0, flux_guidance=5.0,
                              flux_guidance_profile="Dynamic (Creative Start/End)")
    assert sum(len(call["sigmas"]) - 1 for call in recorder.calls) == 16
    values = [call["positive"][0][1]["guidance"] if model_type == "flux" else call["cfg"]
              for call in recorder.calls]
    assert all(value == pytest.approx(5.0) for value in values)


def _code_lines(fn):
    return [ln for ln in inspect.getsource(fn).splitlines() if not ln.strip().startswith("#")]


def test_no_second_model_load_per_stage():
    code = "\n".join(_code_lines(RadianceSamplerPro.sample))
    assert not re.search(r"\bload_model_gpu\s*\(", code)
    assert not re.search(r"\bgc\.collect\s*\(", code)


def test_log_tensor_costs_nothing_without_debug():
    su.logger.setLevel(logging.INFO)
    t = MagicMock()
    su.log_tensor("x", t)
    t.float.assert_not_called()
