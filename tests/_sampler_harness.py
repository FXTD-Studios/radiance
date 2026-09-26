"""Shared in-process harness for driving RadianceSamplerPro.sample() end to end.

ComfyUI is stubbed by conftest, so the pieces `sample()` actually calls into --
`comfy.sample.sample_custom`, `comfy.samplers.sampler_object`,
`comfy.model_management` -- are replaced here with recording fakes. That makes
the node's own control flow (stage planning, step ranges, tiling, windowing,
noise bookkeeping) testable on CPU without a model.

Nothing here simulates a diffusion model. Every fake denoiser is a fixed,
cheap tensor op, which is enough to prove scheduling, coverage, blending
arithmetic, callback accounting and memory behaviour, and is not enough to say
anything about image quality. See the module docstrings of the tests that use
it for what that boundary means.
"""

from __future__ import annotations

import sys
import types
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import torch

import comfy.sample
import comfy.samplers
import comfy.model_management

from radiance.nodes.generate.sampler import RadianceSamplerPro


#: Sentinel for "this attribute did not exist before we patched it".
_MISSING = object()


class _FakeNestedTensor:
    """Stand-in for comfy.nested_tensor.NestedTensor (video + audio streams)."""

    is_nested = True

    def __init__(self, tensors):
        self.tensors = tuple(tensors)

    def unbind(self):
        return self.tensors


def _prepare_noise_like_comfy(latent_image, seed, noise_inds=None):
    """Same contract as comfy.sample.prepare_noise: seeded noise, latent shape."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) & 0xFFFFFFFFFFFFFFFF)
    return torch.randn(
        latent_image.shape, generator=generator, device="cpu",
        dtype=latent_image.dtype, layout=latent_image.layout,
    )


class FakeModelPatcher:
    """Enough of comfy.model_patcher.ModelPatcher for the sampler's use of it."""

    def __init__(self, latent_format: str = "flux", context_dim: Optional[int] = None):
        self.model_options: Dict[str, Any] = {}
        self.load_device = "cpu"
        self.clone_count = 0
        self.model = types.SimpleNamespace(
            latent_format=latent_format,
            model_config=types.SimpleNamespace(
                unet_config={"context_dim": context_dim}
            ),
        )

    def clone(self):
        other = FakeModelPatcher()
        other.model_options = dict(self.model_options)
        other.model = self.model
        other.load_device = self.load_device
        other.clone_count = self.clone_count + 1
        return other

    # ── patch API surface the sampler touches ────────────────────────────────
    def set_model_unet_function_wrapper(self, fn):
        self.model_options["model_function_wrapper"] = fn

    def set_model_sampler_post_cfg_function(self, fn):
        self.model_options.setdefault("sampler_post_cfg_function", []).append(fn)

    def set_model_sampler_cfg_function(self, fn, disable_cfg1_optimization=False):
        self.model_options["sampler_cfg_function"] = fn

    def set_model_attn1_patch(self, fn):
        self.model_options["attn1_patch"] = fn

    def get_model_object(self, name):
        if name == "model_sampling":
            return types.SimpleNamespace(sigma_min=0.002, sigma_max=14.6)
        # LTX-AV probing looks for diffusion_model; a plain model has none.
        raise AttributeError(name)


class SampleCustomRecorder:
    """Replaces comfy.sample.sample_custom and records every call.

    `denoise` is applied once per call and stands in for running the whole
    supplied sigma schedule. It is deterministic, so a run can be compared with
    another run bit-for-bit.
    """

    def __init__(self, denoise=None, model_wrapper_steps: int = 0):
        self.calls: List[Dict[str, Any]] = []
        self.denoise = denoise or (lambda latent, noise, sigmas: latent + noise * 0.5)
        #: number of fake denoising steps to drive through the registered unet
        #: wrapper per sample_custom call, so wrapper behaviour is observable.
        self.model_wrapper_steps = model_wrapper_steps
        self.wrapper_inputs: List[torch.Tensor] = []

    def __call__(self, model, noise, cfg, sampler, sigmas, positive, negative,
                 latent_image, noise_mask=None, callback=None, disable_pbar=False,
                 seed=0, **kwargs):
        self.calls.append({
            "model": model,
            "noise": noise,
            "cfg": cfg,
            "sigmas": sigmas,
            "positive": positive,
            "negative": negative,
            "latent_image": latent_image,
            "seed": seed,
            "callback": callback,
            "disable_pbar": disable_pbar,
        })

        result = self.denoise(latent_image, noise, sigmas)

        wrapper = None
        if hasattr(model, "model_options"):
            wrapper = model.model_options.get("model_function_wrapper")

        n_steps = max(0, min(self.model_wrapper_steps, max(0, len(sigmas) - 1)))
        for step in range(n_steps):
            if wrapper is not None:
                def apply_model(x, t, **c):
                    self.wrapper_inputs.append(x)
                    return x * 2.0
                result = wrapper(
                    apply_model,
                    {
                        "input": result,
                        "timestep": torch.tensor([float(sigmas[step])]),
                        "c": {"c_crossattn": torch.zeros(1, 4, 8)},
                        "cond_or_uncond": [0],
                    },
                )
            if callback is not None:
                callback(step, result, result, n_steps)
        return result


class SamplerEnv:
    """Context manager installing the fakes and restoring what was there."""

    def __init__(self, recorder: Optional[SampleCustomRecorder] = None):
        self.recorder = recorder or SampleCustomRecorder()
        self._saved: Dict[str, Any] = {}

    def __enter__(self):
        # conftest stubs comfy.nested_tensor.NestedTensor as None while the
        # import still succeeds, so _HAS_NESTED_TENSOR is True and every
        # isinstance(x, _NestedTensor) raises TypeError. Give the modules a real
        # class for the duration so the sampler's normal path can run.
        import radiance.sampler_utils as _su
        import radiance.nodes.generate.sampler as _sm
        self._saved["_su_nt"] = _su._NestedTensor
        self._saved["_sm_nt"] = _sm._NestedTensor
        _su._NestedTensor = _FakeNestedTensor
        _sm._NestedTensor = _FakeNestedTensor
        self._su, self._sm = _su, _sm

        # Other test modules in the suite install their own comfy stubs, which
        # replaces the module OBJECTS. Two aliases can therefore disagree:
        # `import comfy.sample as cs` binds getattr(comfy, "sample") in
        # preference to sys.modules["comfy.sample"], and the code under test
        # uses both spellings. Patch every alias that resolves, or the code
        # calls somebody else's MagicMock and gets a MagicMock back.
        # Take the `comfy` roots from the modules under test as well as from
        # sys.modules: a test that swapped sys.modules["comfy"] leaves the
        # already-imported sampler holding the previous object in its globals,
        # and that is the one its `comfy.sample.prepare_noise` call resolves.
        roots = []
        for candidate in (
            sys.modules.get("comfy"),
            getattr(_su, "comfy", None),
            getattr(_sm, "comfy", None),
        ):
            if candidate is not None and not any(candidate is r for r in roots):
                roots.append(candidate)

        self._targets = {}
        for sub, names in (
            ("sample", ("prepare_noise", "sample_custom")),
            ("samplers", ("sampler_object",)),
            ("model_management", ("load_model_gpu",)),
        ):
            modules = []
            candidates = [sys.modules.get(f"comfy.{sub}")]
            candidates += [getattr(root, sub, None) for root in roots]
            for candidate in candidates:
                if candidate is not None and not any(candidate is m for m in modules):
                    modules.append(candidate)
            self._targets[sub] = (modules, names)

        replacements = {
            "prepare_noise": _prepare_noise_like_comfy,
            "sample_custom": self.recorder,
            "sampler_object": lambda name: f"<sampler {name}>",
            "load_model_gpu": MagicMock(),
        }
        for sub, (modules, names) in self._targets.items():
            for index, module in enumerate(modules):
                for name in names:
                    self._saved[(sub, index, name)] = getattr(module, name, _MISSING)
                    setattr(module, name, replacements[name])
        return self

    def __exit__(self, *exc):
        for sub, (modules, names) in self._targets.items():
            for index, module in enumerate(modules):
                for name in names:
                    saved = self._saved[(sub, index, name)]
                    if saved is _MISSING:
                        if hasattr(module, name):
                            delattr(module, name)
                    else:
                        setattr(module, name, saved)
        self._su._NestedTensor = self._saved["_su_nt"]
        self._sm._NestedTensor = self._saved["_sm_nt"]
        return False


def make_latent(shape=(1, 4, 16, 16)) -> Dict[str, torch.Tensor]:
    torch.manual_seed(1234)
    return {"samples": torch.randn(*shape)}


def make_cond(dim: int = 768) -> list:
    return [(torch.zeros(1, 7, dim), {})]


DEFAULT_KWARGS = dict(
    preset="Custom",
    steps=8,
    start_step=0,
    end_step=0,
    cfg=1.0,
    audio_cfg=0.0,
    sampler="euler",
    sampler_mode="Standard",
    phase_split=0.4,
    scheduler="normal",
    scheduler_mode="Manual",
    denoise=1.0,
    flux_shift=1.0,
    flux_guidance=3.5,
    flux_guidance_profile="Static",
    add_noise=True,
    return_with_leftover_noise=False,
    seed=42,
    model_type="sdxl",
    noise_type="Gaussian",
    preview_method="None",
)


def run_sampler(latent=None, model=None, recorder=None, **overrides):
    """Drive RadianceSamplerPro.sample() with the fakes installed.

    Returns ``(result_tuple, recorder)``.
    """
    node = RadianceSamplerPro()
    latent = latent if latent is not None else make_latent()
    model = model if model is not None else FakeModelPatcher()
    kwargs = dict(DEFAULT_KWARGS)
    kwargs.update(overrides)

    with SamplerEnv(recorder) as env:
        out = node.sample(
            model=model,
            positive=make_cond(),
            negative=make_cond(),
            latent_image=latent,
            **kwargs,
        )
    return out["result"], env.recorder
