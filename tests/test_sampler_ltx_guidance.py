"""LTX-AV guidance in the Sampler: the ltx_guidance recipes, ComfyUI's STG and
modality patches, and the guidance rescale computed per modality."""

import logging
import types

import pytest
import torch

import radiance.nodes.generate.sampler as sampler_module
from radiance.sampler_utils import make_guidance_rescale_patch, resolve_ltx_guidance

from _sampler_harness import FakeModelPatcher, SampleCustomRecorder, _FakeNestedTensor, run_sampler


def test_recipes_fill_the_settings_and_custom_keeps_them():
    assert resolve_ltx_guidance("Modality + STG (early)", 0.0, 1.0, 1.0) == (1.0, 0.5, 3.0)
    assert resolve_ltx_guidance("Off", 2.0, 0.3, 5.0) == (0.0, 1.0, 1.0)
    assert resolve_ltx_guidance("Custom", 2.0, 0.3, 5.0) == (2.0, 0.3, 5.0)


def test_rescale_brings_video_and_audio_back_to_their_own_std():
    torch.manual_seed(0)
    v, a = 600, 40
    cond = torch.cat((torch.randn(1, 1, v), torch.randn(1, 1, a) * 0.2), dim=-1)
    guided = torch.cat((cond[..., :v] * 1.5, cond[..., v:] * 4.0), dim=-1)
    args = {"denoised": guided, "cond_denoised": cond}

    out = make_guidance_rescale_patch(1.0, v)(args)
    assert torch.allclose(out[..., :v].std(), cond[..., :v].std(), rtol=1e-4)
    assert torch.allclose(out[..., v:].std(), cond[..., v:].std(), rtol=1e-4)
    # One std for both follows the video: the audio stays far above its own level.
    joint = make_guidance_rescale_patch(1.0)(args)
    assert joint[..., v:].std() / cond[..., v:].std() > 2.0


def test_rescale_without_a_split_is_the_plain_formula():
    torch.manual_seed(1)
    cond, guided = torch.randn(2, 4, 8, 8), torch.randn(2, 4, 8, 8) * 3.0
    out = make_guidance_rescale_patch(0.7)({"denoised": guided, "cond_denoised": cond})
    dims = [1, 2, 3]
    rescaled = guided * (cond.std(dim=dims, keepdim=True) / guided.std(dim=dims, keepdim=True))
    assert torch.allclose(out, rescaled * 0.7 + guided * 0.3, atol=1e-6)


class _NestedLatent(_FakeNestedTensor):
    """The video stream's shape, as comfy.nested_tensor.NestedTensor reports it."""

    @property
    def shape(self):
        return self.tensors[0].shape

    @property
    def ndim(self):
        return self.tensors[0].ndim


class _LtxAvModel(FakeModelPatcher):
    def __init__(self):
        super().__init__(latent_format="ltxav")
        self.model.memory_required = lambda *args, **kwargs: 0

    def clone(self):
        other = super().clone()
        other.__class__ = _LtxAvModel
        return other

    def get_model_object(self, name):
        if name == "diffusion_model":
            return types.SimpleNamespace(separate_audio_and_video_latents=None,
                                         recombine_audio_and_video_latents=None)
        return super().get_model_object(name)


def _guidance_node(label, calls):
    class Node:
        @staticmethod
        def execute(model, *args):
            calls.append((label, args))
            model = model.clone()
            model.set_model_sampler_post_cfg_function(label)
            return types.SimpleNamespace(args=(model,))
    return Node


def _run_ltx_av(monkeypatch, **overrides):
    calls = []
    monkeypatch.setattr(sampler_module, "LTXVSpatioTemporalGuidance", _guidance_node("stg", calls))
    monkeypatch.setattr(sampler_module, "LTXVModalityGuidance", _guidance_node("modality", calls))
    latent = {"samples": _NestedLatent((torch.zeros(1, 128, 3, 4, 6), torch.zeros(1, 8, 5, 16)))}
    recorder = SampleCustomRecorder(denoise=lambda latent, noise, sigmas: latent)
    kwargs = dict(model_type="ltxav", scheduler="simple", cfg=3.0)
    kwargs.update(overrides)
    _, recorder = run_sampler(latent=latent, model=_LtxAvModel(), recorder=recorder, **kwargs)
    post_cfg = recorder.calls[0]["model"].model_options.get("sampler_post_cfg_function", [])
    return calls, post_cfg


def test_a_recipe_registers_stg_then_modality_before_the_rescale(monkeypatch):
    calls, post_cfg = _run_ltx_av(monkeypatch, ltx_guidance="Modality + STG (early)", guidance_rescale_phi=0.45)
    assert calls == [("stg", (1.0, "28", 0.0, 0.5)), ("modality", (3.0, 0.0, 1.0))]
    assert post_cfg[:2] == ["stg", "modality"] and callable(post_cfg[2])


def test_the_rescale_follows_ltx_guidance_at_cfg_1(monkeypatch, caplog):
    with caplog.at_level(logging.WARNING, logger="radiance.sampler"):
        _, post_cfg = _run_ltx_av(monkeypatch, ltx_guidance="Modality", guidance_rescale_phi=0.45, cfg=1.0)
    assert post_cfg[0] == "modality" and callable(post_cfg[1])
    assert not any("ignored at cfg" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("mode, expected", [
    ("Off", []),
    ("Modality", ["modality"]),
    ("Custom", ["stg"]),
])
def test_only_the_active_guidance_runs(monkeypatch, mode, expected):
    calls, _ = _run_ltx_av(monkeypatch, ltx_guidance=mode, stg_scale=2.0, stg_end=1.0, modality_scale=1.0)
    assert [label for label, _ in calls] == expected


def test_ltx_guidance_does_nothing_on_other_models(monkeypatch):
    calls = []
    monkeypatch.setattr(sampler_module, "LTXVSpatioTemporalGuidance", _guidance_node("stg", calls))
    monkeypatch.setattr(sampler_module, "LTXVModalityGuidance", _guidance_node("modality", calls))
    run_sampler(ltx_guidance="Modality + STG (full)")
    assert calls == []


def test_an_older_comfyui_without_the_nodes_fails_clearly(monkeypatch):
    monkeypatch.setattr(sampler_module, "LTXVSpatioTemporalGuidance", None)
    monkeypatch.setattr(sampler_module, "LTXVModalityGuidance", None)
    latent = {"samples": _NestedLatent((torch.zeros(1, 128, 3, 4, 6), torch.zeros(1, 8, 5, 16)))}
    with pytest.raises(ValueError, match="Update ComfyUI, or set ltx_guidance to Off"):
        run_sampler(latent=latent, model=_LtxAvModel(), model_type="ltxav", scheduler="simple",
                    ltx_guidance="Modality")
