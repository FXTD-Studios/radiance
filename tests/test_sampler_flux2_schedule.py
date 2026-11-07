"""FLUX.2's "simple" schedule follows the token count each model call sees, on every path:
the whole latent, one tile in tile_mode, and a refiner taking over."""
import types
from unittest.mock import patch

import torch

from _sampler_harness import FakeModelPatcher, run_sampler
from radiance.nodes.generate import sampler as sampler_node

LATENT = {"samples": torch.zeros(1, 128, 64, 48)}
FLUX2 = dict(model_type="flux2-klein", scheduler="simple")


def _recording_schedule():
    calls = []

    def get_schedule(steps, seq):
        calls.append(seq)
        return torch.linspace(1.0, 0.0, steps + 1)
    return calls, types.SimpleNamespace(get_schedule=get_schedule)


def _patched(fake):
    return patch.dict(sampler_node.get_flux_sigmas.__globals__, {"nodes_flux": fake})


def test_tile_mode_schedules_for_one_tile():
    calls, fake = _recording_schedule()
    with _patched(fake):
        run_sampler(latent=dict(LATENT), **FLUX2, tile_mode=True, tile_size=32, tile_overlap=8)
        run_sampler(latent=dict(LATENT), **FLUX2)
    assert calls == [32 * 32, 64 * 48]


def test_a_flux2_refiner_takes_over_on_the_same_schedule():
    calls, fake = _recording_schedule()
    with _patched(fake):
        run_sampler(latent=dict(LATENT), **FLUX2, refiner_model=FakeModelPatcher(), refiner_start_step=4)
    assert len(calls) >= 2 and set(calls) == {64 * 48}


def test_the_image_size_schedule_is_never_served_from_the_cache():
    calls, fake = _recording_schedule()
    cache = sampler_node.get_flux_sigmas.__globals__["SigmaCache"]()
    model = FakeModelPatcher()
    with _patched(fake):
        for seq in (1024, 4096):
            sampler_node.compute_base_sigmas(model, "simple", 8, "simple", 1.0, 1.0, cache, image_seq_len=seq)
    assert calls == [1024, 4096]
