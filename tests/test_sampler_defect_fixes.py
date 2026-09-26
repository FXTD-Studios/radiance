"""
tests/test_sampler_defect_fixes.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Confirmed defects across RadianceSamplerPro, sampler_utils, the tensor
contract helpers, RadianceRegionalPrompt / RadianceRegionalGrid and
RadianceResolution.

Every test here fails on the code as it stood before these fixes. The shared
theme is silence: each defect produced a plausible-looking result and a log
line claiming success, so nothing downstream could tell a good run from a
broken one.
"""

from __future__ import annotations

import json
import logging

import pytest
import torch

import radiance.sampler_utils as su
from radiance.nodes.generate.sampler import (
    RadianceSamplerPro,
    _make_energy_cfg_patch,
)

# Imported lazily inside the tests that need them: neither symbol exists on the
# pre-fix code, and a module-level import would turn a per-test failure into a
# collection error that says nothing about which defect is being pinned.
try:
    from radiance.nodes.generate.sampler import assert_latent_finite
except ImportError:  # pragma: no cover - only on the pre-fix code
    assert_latent_finite = None
from radiance.core.tensor.contract import ensure_4d, ensure_5d
from radiance.nodes.generate.regional import (
    RadianceRegionalGrid,
    RadianceRegionalPrompt,
    _make_area_cond,
)
from radiance.tests._sampler_harness import (
    FakeModelPatcher,
    SampleCustomRecorder,
    SamplerEnv,
    _FakeNestedTensor,
    run_sampler,
)


# ═════════════════════════════════════════════════════════════════════════════
#  Regional conditioning: two nodes that did not work at stock defaults
# ═════════════════════════════════════════════════════════════════════════════

class TestRegionalAreaFormat:
    """`area` without the "percentage" marker crashes ComfyUI's sampler.

    resolve_areas_and_cond_masks_multidim (comfy/samplers.py:768, identical in
    0.32.0 and 0.36.0) only multiplies an area by the latent dims when
    `area[0] == "percentage"`. Without the marker the raw floats reach
    get_area_and_mult, where `input_x.narrow(i + 2, 0.0, 0.5)` raises
    "TypeError: narrow(): argument 'start' must be int, not float".
    """

    @staticmethod
    def _resolve_like_comfy(area, dims):
        """Verbatim transcription of ComfyUI's own area resolution."""
        if area[0] == "percentage":
            a = area[1:]
            a_len = len(a) // 2
            resolved = ()
            for d in range(len(dims)):
                resolved += (max(1, round(a[d] * dims[d])),)
            for d in range(len(dims)):
                resolved += (round(a[d + a_len] * dims[d]),)
            return resolved
        return tuple(area)

    def test_area_carries_the_percentage_marker(self):
        out = _make_area_cond([(torch.zeros(1, 7, 768), {})], x=0.25, y=0.5, w=0.5, h=0.25)
        area = out[0][1]["area"]
        assert area[0] == "percentage", (
            f"area={area!r} has no 'percentage' marker, so ComfyUI passes the "
            f"raw floats to narrow() and raises TypeError"
        )
        assert area == ("percentage", 0.25, 0.5, 0.5, 0.25)

    def test_resolved_area_is_all_integers(self):
        """The property narrow() needs: every element an int after resolution."""
        out = _make_area_cond([(torch.zeros(1, 7, 768), {})], x=0.25, y=0.5, w=0.5, h=0.25)
        resolved = self._resolve_like_comfy(out[0][1]["area"], dims=(64, 64))
        assert all(isinstance(v, int) for v in resolved), resolved
        # h=0.25 of 64 rows, w=0.5 of 64 cols, at y=0.5, x=0.25.
        assert resolved == (16, 32, 32, 16)

    def test_narrow_accepts_the_resolved_area(self):
        """End to end against the torch call that actually raised."""
        out = _make_area_cond([(torch.zeros(1, 7, 768), {})], x=0.25, y=0.5, w=0.5, h=0.25)
        area = self._resolve_like_comfy(out[0][1]["area"], dims=(64, 64))
        latent = torch.zeros(1, 4, 64, 64)
        cropped = latent.narrow(2, area[2], area[0]).narrow(3, area[3], area[1])
        assert cropped.shape == (1, 4, 16, 32)

    def test_regional_prompt_node_produces_a_usable_area(self):
        node = RadianceRegionalPrompt()
        cond, info = node.apply(
            base_cond=[(torch.zeros(1, 7, 768), {})],
            region_cond=[(torch.zeros(1, 7, 768), {})],
        )
        areas = [d["area"] for _, d in cond if "area" in d]
        assert areas, "the node emitted no area conditioning at all"
        for area in areas:
            assert area[0] == "percentage"
            resolved = self._resolve_like_comfy(area, dims=(60, 104))
            assert all(isinstance(v, int) for v in resolved)

    def test_regional_grid_node_produces_a_usable_area(self):
        class _Clip:
            def tokenize(self, text):
                return {"l": [[]]}

            def encode_from_tokens(self, tokens, return_pooled=False):
                return torch.zeros(1, 7, 768), torch.zeros(1, 768)

        node = RadianceRegionalGrid()
        cond, info = node.apply_grid(
            base_cond=[(torch.zeros(1, 7, 768), {})],
            clip=_Clip(),
            grid_prompts='["left", "right"]',
            columns=2, rows=1,
        )
        areas = [d["area"] for _, d in cond if "area" in d]
        assert len(areas) == 2
        for area in areas:
            assert area[0] == "percentage"
            assert all(
                isinstance(v, int)
                for v in self._resolve_like_comfy(area, dims=(60, 104))
            )


class TestRegionalIPAdapterIsHonest:
    """ip_image wrote a dict into a ComfyUI key that wants a text embedding.

    'cross_attn_controlnet' is read by BaseModel.extra_conds and handed to
    comfy.conds.CONDCrossAttn, i.e. it is a ControlNet cross-attention
    embedding slot. It is not, and never has been, an IP-Adapter hook.
    IP-Adapter is a model-side attention patch, so no CONDITIONING node can
    enable it. The node used to report "ip_adapter": {"enabled": true} and log
    ip=True anyway.
    """

    def test_no_bogus_key_is_written_into_the_conditioning(self):
        node = RadianceRegionalPrompt()
        cond, info = node.apply(
            base_cond=[(torch.zeros(1, 7, 768), {})],
            region_cond=[(torch.zeros(1, 7, 768), {})],
            ip_image=torch.rand(1, 64, 64, 3),
            ip_weight=0.8,
        )
        for _, d in cond:
            assert "cross_attn_controlnet" not in d, (
                "a dict was written into ComfyUI's CONDCrossAttn slot"
            )

    def test_region_info_does_not_claim_ip_adapter_ran(self):
        node = RadianceRegionalPrompt()
        _, info = node.apply(
            base_cond=[(torch.zeros(1, 7, 768), {})],
            region_cond=[(torch.zeros(1, 7, 768), {})],
            ip_image=torch.rand(1, 64, 64, 3),
        )
        parsed = json.loads(info)
        assert parsed["ip_adapter"]["enabled"] is False
        assert parsed["ip_adapter"]["ignored_reason"]

    def test_connecting_ip_image_warns(self, caplog):
        node = RadianceRegionalPrompt()
        with caplog.at_level(logging.WARNING, logger="radiance.regional"):
            node.apply(
                base_cond=[(torch.zeros(1, 7, 768), {})],
                region_cond=[(torch.zeros(1, 7, 768), {})],
                ip_image=torch.rand(1, 64, 64, 3),
            )
        assert any("IP-Adapter" in r.message for r in caplog.records)


# ═════════════════════════════════════════════════════════════════════════════
#  The un-sampled latent return
# ═════════════════════════════════════════════════════════════════════════════

class TestEmptyStepRangeFailsLoudly:
    """A collapsed step range returned the INPUT latent, logging success.

    `splits = {effective_start, effective_end}` became a one-element set, the
    stage loop never ran, and `sample()` returned the un-denoised input while
    logging "Sampling complete". Both widgets are min 0 / max 200, so all three
    cases below are a couple of clicks away.
    """

    CASES = [
        pytest.param(10, 10, 20, id="start==end"),
        pytest.param(50, 0, 20, id="start>steps,end=0"),
        pytest.param(200, 200, 20, id="both-at-the-widget-maximum"),
    ]

    @pytest.mark.parametrize("start_step,end_step,steps", CASES)
    def test_it_raises_instead_of_returning_the_input(self, start_step, end_step, steps):
        rec = SampleCustomRecorder()
        with pytest.raises(RuntimeError, match="Empty step range"):
            run_sampler(
                recorder=rec, steps=steps,
                start_step=start_step, end_step=end_step,
            )
        assert not rec.calls, "no sampling should have been attempted"

    def test_a_reversed_range_is_repaired_rather_than_collapsed(self):
        """start_step=20, end_step=5 used to collapse to the empty range (5, 5).

        `effective_start = min(start_step, effective_end)` pulled the start down
        to the end instead of swapping, so the run returned the input latent.
        validate_step_range swaps it into the real range the user described.
        """
        rec = SampleCustomRecorder()
        run_sampler(recorder=rec, steps=20, start_step=20, end_step=5)
        assert rec.calls, "steps 5-20 is a real range and must be sampled"
        # 15 steps of schedule means 16 sigma values.
        assert len(rec.calls[0]["sigmas"]) == 16

    def test_a_normal_range_still_samples(self):
        rec = SampleCustomRecorder()
        (out, *_), _ = run_sampler(recorder=rec, steps=20, start_step=0, end_step=0)
        assert rec.calls
        assert out["samples"] is not None


# ═════════════════════════════════════════════════════════════════════════════
#  tile_mode
# ═════════════════════════════════════════════════════════════════════════════

class TestTileModeNoLongerDiscardsSettingsSilently:
    """tile_mode ran tile_sample once and `break`-ed out of the stage loop.

    Phase-shift, refiner_model, dynamic guidance, start_step, end_step and
    add_noise were all dropped without a word, and flux_guidance never reached
    the conditioning at all, because the only apply_flux_guidance call lives
    inside the loop that was skipped.
    """

    @staticmethod
    def _tile_calls(recorder):
        return recorder.calls

    def test_add_noise_false_is_honoured(self):
        rec = SampleCustomRecorder()
        run_sampler(
            recorder=rec, model_type="sdxl", tile_mode=True,
            tile_size=32, tile_overlap=8, add_noise=False,
            latent={"samples": torch.zeros(1, 4, 64, 64)},
        )
        assert rec.calls
        for call in rec.calls:
            assert not call["noise"].any(), (
                "add_noise=False was ignored: tiling still injected noise"
            )

    def test_add_noise_true_still_injects(self):
        rec = SampleCustomRecorder()
        run_sampler(
            recorder=rec, model_type="sdxl", tile_mode=True,
            tile_size=32, tile_overlap=8, add_noise=True,
            latent={"samples": torch.zeros(1, 4, 64, 64)},
        )
        assert any(c["noise"].any() for c in rec.calls)

    def test_start_and_end_step_slice_the_schedule(self):
        def sigma_count(**kw):
            rec = SampleCustomRecorder()
            run_sampler(
                recorder=rec, model_type="sdxl", steps=20, tile_mode=True,
                tile_size=32, tile_overlap=8,
                latent={"samples": torch.zeros(1, 4, 64, 64)}, **kw
            )
            return len(rec.calls[0]["sigmas"])

        full = sigma_count()
        partial = sigma_count(start_step=5, end_step=12)
        assert partial < full, (
            f"start_step/end_step were ignored: tiling ran {partial} sigmas "
            f"for steps 5-12 and {full} for the whole schedule"
        )
        assert partial == 12 - 5 + 1

    def test_flux_guidance_reaches_the_conditioning(self):
        rec = SampleCustomRecorder()
        run_sampler(
            recorder=rec, model_type="flux", tile_mode=True,
            tile_size=32, tile_overlap=8, flux_guidance=7.5,
            latent={"samples": torch.zeros(1, 16, 64, 64)},
        )
        guidances = [
            d.get("guidance") for _, d in rec.calls[0]["positive"]
        ]
        assert 7.5 in guidances, (
            f"flux_guidance never reached the conditioning in tile mode: {guidances}"
        )

    @pytest.mark.parametrize("kwargs,needle", [
        ({"sampler_mode": "Phase-Shift (Euler >> DPM)"}, "sampler_mode"),
        ({"flux_guidance_profile": "Dynamic (Creative Start/End)", "cfg": 7.0},
         "Dynamic"),
    ])
    def test_unsupported_settings_are_named_in_a_warning(self, kwargs, needle, caplog):
        with caplog.at_level(logging.WARNING, logger="radiance.sampler"):
            run_sampler(
                model_type="sdxl", tile_mode=True, tile_size=32, tile_overlap=8,
                latent={"samples": torch.zeros(1, 4, 64, 64)}, **kwargs
            )
        messages = [r.getMessage() for r in caplog.records]
        assert any("tile_mode=True ignores" in m for m in messages), messages
        assert any(needle in m for m in messages), (
            f"the warning did not name the ignored setting: {messages}"
        )

    def test_no_warning_when_nothing_is_ignored(self, caplog):
        with caplog.at_level(logging.WARNING, logger="radiance.sampler"):
            run_sampler(
                model_type="sdxl", tile_mode=True, tile_size=32, tile_overlap=8,
                latent={"samples": torch.zeros(1, 4, 64, 64)},
            )
        assert not any(
            "tile_mode=True ignores" in r.getMessage() for r in caplog.records
        )


class TestFailedTileFailsTheRun:
    """A failed tile was replaced by the un-denoised input slice.

    The condition that makes a tile fail is almost always OOM at high
    resolution, which is the exact condition tiling exists to avoid. The old
    code feathered raw latent noise into the plate and reported success.
    """

    @staticmethod
    def _tile_with(denoise):
        rec = SampleCustomRecorder(denoise=denoise)
        with SamplerEnv(rec):
            return su.tile_sample(
                model=FakeModelPatcher(),
                noise=torch.zeros(1, 4, 64, 64),
                latent_samples=torch.ones(1, 4, 64, 64),
                positive=[], negative=[],
                sigmas=torch.linspace(1.0, 0.0, 5),
                sampler_obj=object(), seed=0,
                tile_size=32, tile_overlap=8,
            )

    @staticmethod
    def _oom(latent, noise, sigmas):
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

    def test_a_failing_tile_raises(self):
        with pytest.raises(RuntimeError) as exc:
            self._tile_with(self._oom)
        assert "Tiled sampling failed" in str(exc.value)
        assert "out-of-memory" in str(exc.value)

    def test_the_original_error_is_preserved(self):
        with pytest.raises(RuntimeError) as exc:
            self._tile_with(self._oom)
        assert "CUDA out of memory" in str(exc.value)
        assert isinstance(exc.value.__cause__, RuntimeError)

    def test_the_error_names_the_tile_and_the_settings(self):
        with pytest.raises(RuntimeError) as exc:
            self._tile_with(self._oom)
        message = str(exc.value)
        assert "tile_size=32" in message
        assert "tile_overlap=8" in message
        assert "tile 1/" in message

    def test_the_undenoised_input_is_not_substituted(self):
        """The old fallback: `t_out = t_latent`, feathered in, run reports success.

        The input latent here is all ones and the denoiser returns all zeros,
        so any surviving input slice shows up as a non-zero region.
        """
        calls = {"n": 0}

        def fail_one_tile(latent, noise, sigmas):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("CUDA out of memory")
            return torch.zeros_like(latent)

        with pytest.raises(RuntimeError):
            self._tile_with(fail_one_tile)

    def test_a_successful_tiling_run_still_blends(self):
        rec = SampleCustomRecorder()
        with SamplerEnv(rec):
            out = su.tile_sample(
                model=FakeModelPatcher(),
                noise=torch.zeros(1, 4, 64, 64),
                latent_samples=torch.ones(1, 4, 64, 64),
                positive=[], negative=[],
                sigmas=torch.linspace(1.0, 0.0, 5),
                sampler_obj=object(), seed=0,
                tile_size=32, tile_overlap=8,
            )
        assert out.shape == (1, 4, 64, 64)
        assert torch.isfinite(out).all()


# ═════════════════════════════════════════════════════════════════════════════
#  PAG
# ═════════════════════════════════════════════════════════════════════════════

class TestPagScaleUsesItsWholeRange:
    """pag_scale was clamped to 1.0 while the widget is min 0.0 / max 5.0."""

    @staticmethod
    def _run_patch(pag_scale):
        model = FakeModelPatcher()
        patched = su.apply_pag_to_model(model, pag_scale, cfg=7.0)
        patch = patched.model_options["attn1_patch"]
        torch.manual_seed(0)
        q = torch.randn(2, 6, 8)
        k = torch.randn(2, 6, 8)
        v = torch.randn(2, 6, 8)
        return patch(q, k, v, {"cond_or_uncond": [0, 1], "block": ("middle", 0)})

    def test_scales_above_one_are_distinguishable(self):
        _, k1, v1 = self._run_patch(1.0)
        _, k5, v5 = self._run_patch(5.0)
        assert not torch.allclose(k1, k5), (
            "pag_scale 1.0 and 5.0 produced bit-identical keys; four fifths of "
            "the slider was dead"
        )
        assert not torch.allclose(v1, v5)

    @pytest.mark.parametrize("scale", [1.5, 2.0, 3.0, 5.0])
    def test_every_step_above_one_differs_from_the_last(self, scale):
        _, k_prev, _ = self._run_patch(scale - 0.5)
        _, k_now, _ = self._run_patch(scale)
        assert not torch.allclose(k_prev, k_now)

    def test_values_at_or_below_one_are_unchanged(self):
        """Existing graphs using 0 < pag_scale <= 1 must render identically."""
        _, k, v = self._run_patch(0.4)
        torch.manual_seed(0)
        q = torch.randn(2, 6, 8)
        k_in = torch.randn(2, 6, 8)
        v_in = torch.randn(2, 6, 8)
        expected_k = k_in.clone()
        expected_k[0:1] = k_in[0:1]  # cond slice untouched
        expected_k[1:2] = k_in[1:2] * 0.6 + q[1:2] * 0.4
        assert torch.allclose(k, expected_k, atol=1e-6)


class TestPagIsHonestAtLowCfg:
    """The patch only touches the uncond batch, which cfg <= 1.0 never runs.

    ComfyUI skips the unconditional pass entirely below cfg 1.0, and at exactly
    1.0 the CFG combine is `uncond + 1.0 * (cond - uncond)` == cond, so a
    perturbed uncond cannot reach the output either way. The "PAG applied at
    scale" log fired at registration regardless, and this sampler's own cfg
    default is 1.0.
    """

    @pytest.mark.parametrize("cfg", [0.0, 0.5, 1.0])
    def test_it_warns_and_does_not_patch(self, cfg, caplog):
        model = FakeModelPatcher()
        with caplog.at_level(logging.WARNING, logger="radiance.sampler"):
            result = su.apply_pag_to_model(model, 3.0, cfg=cfg)
        assert result is model, "a patch was installed that cannot fire"
        assert "attn1_patch" not in model.model_options
        assert any("no effect at cfg" in r.getMessage() for r in caplog.records)

    def test_it_does_not_claim_pag_was_applied(self, caplog):
        with caplog.at_level(logging.INFO, logger="radiance.sampler"):
            su.apply_pag_to_model(FakeModelPatcher(), 3.0, cfg=1.0)
        assert not any(
            "perturbation applied" in r.getMessage() for r in caplog.records
        )

    def test_it_still_patches_above_one(self):
        model = FakeModelPatcher()
        result = su.apply_pag_to_model(model, 3.0, cfg=7.0)
        assert "attn1_patch" in result.model_options

    def test_the_node_passes_cfg_through(self, caplog):
        with caplog.at_level(logging.WARNING, logger="radiance.sampler"):
            run_sampler(model_type="flux", pag_scale=2.0, cfg=1.0,
                        latent={"samples": torch.zeros(1, 16, 32, 32)})
        assert any("no effect at cfg" in r.getMessage() for r in caplog.records)


# ═════════════════════════════════════════════════════════════════════════════
#  conditioning_clip_target
# ═════════════════════════════════════════════════════════════════════════════

class TestConditioningClipTargetIsHonest:
    """route_conditioning wrote a key nothing reads, and reported success.

    `encoder_target` appears nowhere in ComfyUI 0.32.0, ComfyUI 0.36.0 or
    Radiance. It cannot be made to work from a sampler either: which encoder
    produced an embedding is fixed at encode time.
    """

    @pytest.mark.parametrize("target", ["clip_l", "clip_g", "t5xxl"])
    def test_no_dead_key_is_written(self, target):
        cond = [(torch.zeros(1, 7, 768), {})]
        out = su.route_conditioning(cond, target)
        for entry in out:
            assert "encoder_target" not in entry[1], (
                "a key no ComfyUI version reads was written into the conditioning"
            )

    def test_it_warns_rather_than_reporting_success(self, caplog):
        with caplog.at_level(logging.WARNING, logger="radiance.sampler"):
            su.route_conditioning([(torch.zeros(1, 7, 768), {})], "clip_g")
        assert any(
            "is ignored" in r.getMessage() for r in caplog.records
        ), "the no-op was still silent"

    def test_auto_is_a_true_passthrough(self, caplog):
        cond = [(torch.zeros(1, 7, 768), {})]
        with caplog.at_level(logging.WARNING, logger="radiance.sampler"):
            assert su.route_conditioning(cond, "Auto") is cond
        assert not caplog.records

    def test_the_node_does_not_log_a_successful_routing(self, caplog):
        with caplog.at_level(logging.INFO, logger="radiance.sampler"):
            run_sampler(model_type="sdxl", conditioning_clip_target="clip_g")
        assert not any(
            "Conditioning routed to" in r.getMessage() for r in caplog.records
        )


# ═════════════════════════════════════════════════════════════════════════════
#  sigmas_override
# ═════════════════════════════════════════════════════════════════════════════

class TestSigmasOverrideIsNotClamped:
    """"Bypassing internal sigma computation" then ran correct_sigma_end.

    A custom SIGMAS deliberately ending at 0.03, to leave residual noise for a
    following pass, was silently driven to 0 and fully denoised.
    """

    LEFTOVER = torch.tensor([1.0, 0.7, 0.4, 0.15, 0.03])

    def test_a_non_zero_terminal_sigma_survives(self):
        rec = SampleCustomRecorder()
        (_, sigmas, *_), _ = run_sampler(
            recorder=rec, model_type="sdxl",
            sigmas_override=self.LEFTOVER.clone(),
        )
        assert float(sigmas[-1]) == pytest.approx(0.03), (
            f"the supplied terminal sigma was clamped to {float(sigmas[-1])}"
        )

    def test_the_sampler_is_given_the_unclamped_schedule(self):
        rec = SampleCustomRecorder()
        run_sampler(
            recorder=rec, model_type="sdxl",
            sigmas_override=self.LEFTOVER.clone(),
        )
        assert float(rec.calls[0]["sigmas"][-1]) == pytest.approx(0.03)

    def test_terminal_sigma_to_zero_still_clamps_on_request(self):
        rec = SampleCustomRecorder()
        (_, sigmas, *_), _ = run_sampler(
            recorder=rec, model_type="sdxl",
            sigmas_override=self.LEFTOVER.clone(),
            terminal_sigma_to_zero=True,
        )
        assert float(sigmas[-1]) == 0.0

    def test_leaving_it_unclamped_is_logged(self, caplog):
        with caplog.at_level(logging.INFO, logger="radiance.sampler"):
            run_sampler(
                model_type="sdxl", sigmas_override=self.LEFTOVER.clone()
            )
        assert any("un-clamped" in r.getMessage() for r in caplog.records)


# ═════════════════════════════════════════════════════════════════════════════
#  guidance_rescale_phi
# ═════════════════════════════════════════════════════════════════════════════

def test_guidance_rescale_warns_when_cfg_makes_it_inert(caplog):
    """Skipped at cfg <= 1.0 with no else and no warning.

    The widget tooltip recommends 0.7 and the sampler's own cfg default is 1.0,
    so setting it and getting nothing was the common case.
    """
    with caplog.at_level(logging.WARNING, logger="radiance.sampler"):
        run_sampler(model_type="flux", guidance_rescale_phi=0.7, cfg=1.0,
                    latent={"samples": torch.zeros(1, 16, 32, 32)})
    assert any(
        "guidance_rescale_phi" in r.getMessage() and "ignored at cfg" in r.getMessage()
        for r in caplog.records
    )


def test_guidance_rescale_is_silent_when_it_applies(caplog):
    with caplog.at_level(logging.WARNING, logger="radiance.sampler"):
        run_sampler(model_type="sdxl", guidance_rescale_phi=0.7, cfg=7.0)
    assert not any("ignored at cfg" in r.getMessage() for r in caplog.records)


# ═════════════════════════════════════════════════════════════════════════════
#  Animated energy masks
# ═════════════════════════════════════════════════════════════════════════════

class TestAnimatedEnergyMask:
    """A per-frame mask was collapsed to frame 0 for the whole clip.

    `m.reshape(-1, 1, H, W)` folds T into dim 0 for video, then
    `m.shape[0] != B` took the `m[:1].expand(B, ...)` branch meant for genuine
    batch mismatches. A (1, 1, 21, 480, 832) mask became frame 0's mask applied
    to all 21 latent frames, with no warning.
    """

    @staticmethod
    def _run(mask, cond_shape=(1, 4, 6, 8, 8)):
        patch = _make_energy_cfg_patch([(mask, 1.0)])
        cond = torch.ones(cond_shape)
        uncond = torch.zeros(cond_shape)
        return patch({
            "cond_denoised": cond,
            "uncond_denoised": uncond,
            "cond_scale": 1.0,
            "denoised": cond,
        })

    def test_a_per_frame_mask_varies_across_frames(self):
        T = 6
        # Frame f is masked only in its own quadrant: a mask that genuinely
        # animates, so collapsing it to frame 0 is visible in the output.
        mask = torch.zeros(T, 8, 8)
        for f in range(T):
            mask[f, f, :] = 1.0

        out = self._run(mask)

        # Every pair of frames must differ: the mask moves, so the field must.
        # Comparing per-frame SUMS would not catch this, because each frame's
        # mask has the same total area.
        for f in range(1, T):
            assert not torch.allclose(out[0, 0, 0], out[0, 0, f]), (
                f"frame {f} got the same energy field as frame 0: the animated "
                f"mask was collapsed to a single frame"
            )
        # And specifically, frame f must be boosted on row f and nowhere else.
        for f in range(T):
            assert float(out[0, 0, f, f, 0]) > float(out[0, 0, f, (f + 1) % 8, 0]), (
                f"frame {f} was not masked on its own row"
            )

    def test_frame_zero_is_not_broadcast_to_the_whole_clip(self):
        T = 5
        mask = torch.zeros(T, 8, 8)
        mask[0] = 1.0  # only frame 0 is masked

        out = self._run(mask, cond_shape=(1, 4, T, 8, 8))
        frame0 = float(out[0, 0, 0].mean())
        rest = [float(out[0, 0, f].mean()) for f in range(1, T)]
        assert all(abs(r - frame0) > 1e-6 for r in rest), (
            "frames 1+ got frame 0's mask; the clip was flattened"
        )

    def test_a_five_dimensional_mask_animates_too(self):
        """The shape RadianceEnergyMask emits for a video mask source."""
        T = 4
        mask = torch.zeros(1, 1, T, 8, 8)
        for f in range(T):
            mask[0, 0, f, f, :] = 1.0
        mask = mask.reshape(-1, 8, 8)  # what normalize_energy_mask produces

        out = self._run(mask, cond_shape=(1, 4, T, 8, 8))
        for f in range(1, T):
            assert not torch.allclose(out[0, 0, 0], out[0, 0, f])

    def test_a_still_mask_still_applies_to_every_frame(self):
        mask = torch.zeros(1, 8, 8)
        mask[0, :4, :] = 1.0
        out = self._run(mask, cond_shape=(1, 4, 5, 8, 8))
        per_frame = [float(out[0, 0, f].sum()) for f in range(5)]
        assert len(set(per_frame)) == 1, "a still mask must not vary across frames"

    def test_a_mismatched_frame_count_warns(self, caplog):
        mask = torch.zeros(3, 8, 8)
        mask[:, :4, :] = 1.0
        with caplog.at_level(logging.WARNING, logger="radiance.sampler"):
            self._run(mask, cond_shape=(1, 4, 7, 8, 8))
        assert any("frame" in r.getMessage() for r in caplog.records), (
            "a mask that matches neither the batch nor the clip was silently "
            "reduced to its first frame"
        )

    def test_image_latents_are_unaffected(self):
        mask = torch.zeros(1, 8, 8)
        mask[0, :4, :] = 1.0
        out = self._run(mask, cond_shape=(2, 4, 8, 8))
        assert out.shape == (2, 4, 8, 8)
        assert float(out[0, 0, 0, 0]) > float(out[0, 0, 7, 0])


# ═════════════════════════════════════════════════════════════════════════════
#  Non-finite guard
# ═════════════════════════════════════════════════════════════════════════════

class TestNonFiniteGuard:
    """The guard could not fire on the path that most needed it.

    `torch.isfinite(_s)` raises TypeError on a NestedTensor, and the
    surrounding `except Exception` swallowed it into a `logger.debug` that
    ComfyUI's default INFO level does not print. And on the tensor path the
    remedy was `nan_to_num(nan=0.0)`, but latent 0.0 is not black, so a blown
    up run shipped as a plausible-looking wrong plate.
    """

    def test_a_clean_latent_passes(self):
        assert_latent_finite(torch.randn(1, 4, 8, 8))

    def test_nan_fails_the_node(self):
        bad = torch.randn(1, 4, 8, 8)
        bad[0, 0, 0, 0] = float("nan")
        with pytest.raises(RuntimeError, match="NaN or Inf"):
            assert_latent_finite(bad)

    def test_inf_fails_the_node(self):
        bad = torch.randn(1, 4, 8, 8)
        bad[0, 1, 2, 3] = float("inf")
        with pytest.raises(RuntimeError, match="NaN or Inf"):
            assert_latent_finite(bad)

    def test_it_does_not_silently_substitute_zeros(self):
        bad = torch.full((1, 4, 8, 8), float("nan"))
        with pytest.raises(RuntimeError) as exc:
            assert_latent_finite(bad)
        assert "0.0 is not black" in str(exc.value)

    def test_a_nested_tensor_is_checked_not_skipped(self, monkeypatch):
        """The LTX-AV / MiniMax H3 path, where the guard used to be blind."""
        import radiance.nodes.generate.sampler as sampler_mod

        monkeypatch.setattr(sampler_mod, "_HAS_NESTED_TENSOR", True)
        monkeypatch.setattr(sampler_mod, "_NestedTensor", _FakeNestedTensor)

        good = _FakeNestedTensor((torch.zeros(1, 4, 8, 8), torch.zeros(1, 32, 2, 4)))
        sampler_mod.assert_latent_finite(good)

        video = torch.zeros(1, 4, 8, 8)
        video[0, 0, 0, 0] = float("nan")
        bad = _FakeNestedTensor((video, torch.zeros(1, 32, 2, 4)))
        with pytest.raises(RuntimeError, match="stream 0"):
            sampler_mod.assert_latent_finite(bad)

    def test_an_uncheckable_latent_warns_at_warning_level(self, caplog):
        """The old code buried this at debug, invisible at ComfyUI's INFO."""
        with caplog.at_level(logging.WARNING, logger="radiance.sampler"):
            assert_latent_finite(object())
        assert any("UNCHECKED" in r.getMessage() for r in caplog.records)

    def test_the_node_fails_on_a_diverged_run(self):
        def blow_up(latent, noise, sigmas):
            return torch.full_like(latent, float("nan"))

        rec = SampleCustomRecorder(denoise=blow_up)
        with pytest.raises(RuntimeError, match="NaN or Inf"):
            run_sampler(recorder=rec, model_type="sdxl")


# ═════════════════════════════════════════════════════════════════════════════
#  noise_alpha_end
# ═════════════════════════════════════════════════════════════════════════════

class TestNoiseAlphaRamp:
    """noise_alpha_end appeared only in a predicate and an f-string.

    The tooltip and the comment beside it both promised a cosine interpolation
    across the denoising trajectory. No interpolation existed anywhere in the
    file.
    """

    RAMP = staticmethod(
        getattr(RadianceSamplerPro, '_noise_alpha_at', None) or (lambda *a: None)
    )

    def test_the_ramp_hits_both_endpoints(self):
        assert self.RAMP(0, 20, 0.9, 0.1) == pytest.approx(0.9)
        assert self.RAMP(20, 20, 0.9, 0.1) == pytest.approx(0.1)

    def test_the_ramp_is_a_cosine_not_a_line(self):
        mid = self.RAMP(10, 20, 1.0, 0.0)
        assert mid == pytest.approx(0.5), "midpoint of a cosine ramp is the mean"
        quarter = self.RAMP(5, 20, 1.0, 0.0)
        linear_quarter = 0.75
        assert quarter != pytest.approx(linear_quarter, abs=1e-3), (
            "the ramp is linear, not the promised cosine interpolation"
        )
        assert quarter == pytest.approx(1.0 - (1.0 - 0.0) * (1 - 0.70710678) / 2, abs=1e-6)

    def test_the_ramp_is_monotonic(self):
        values = [self.RAMP(s, 20, 1.0, 0.0) for s in range(21)]
        assert values == sorted(values, reverse=True)

    def test_equal_endpoints_are_flat(self):
        for step in range(21):
            assert self.RAMP(step, 20, 0.6, 0.6) == pytest.approx(0.6)

    def test_noise_alpha_end_changes_the_injected_noise(self):
        """The defect proper: the widget had no effect on any run.

        start_step > 0 injects part way down the trajectory, which is where the
        ramp has something to say.
        """
        def noise_for(noise_alpha_end):
            rec = SampleCustomRecorder()
            run_sampler(
                recorder=rec, model_type="sdxl", steps=20,
                start_step=10, end_step=20,
                noise_type="Perlin",
                noise_alpha_start=1.0, noise_alpha_end=noise_alpha_end,
                seed=99,
            )
            return rec.calls[0]["noise"]

        a = noise_for(1.0)
        b = noise_for(0.0)
        assert not torch.allclose(a, b), (
            "noise_alpha_end made no difference to the noise actually injected"
        )

    def test_it_warns_when_the_end_value_cannot_apply(self, caplog):
        with caplog.at_level(logging.WARNING, logger="radiance.sampler"):
            run_sampler(
                model_type="sdxl", noise_type="Perlin",
                noise_alpha_start=1.0, noise_alpha_end=0.2, start_step=0,
            )
        assert any(
            "noise_alpha_end" in r.getMessage() and "no effect" in r.getMessage()
            for r in caplog.records
        )

    def test_gaussian_noise_is_untouched_by_the_ramp(self):
        """noise_type=Gaussian means there is no structured noise to fade."""
        rec_a, rec_b = SampleCustomRecorder(), SampleCustomRecorder()
        run_sampler(recorder=rec_a, model_type="sdxl", noise_type="Gaussian",
                    noise_alpha_start=0.2, noise_alpha_end=0.9, seed=5)
        run_sampler(recorder=rec_b, model_type="sdxl", noise_type="Gaussian", seed=5)
        assert torch.equal(rec_a.calls[0]["noise"], rec_b.calls[0]["noise"])


# ═════════════════════════════════════════════════════════════════════════════
#  Tensor contract
# ═════════════════════════════════════════════════════════════════════════════

class TestTensorContractVisibility:
    """A 5D -> 4D flatten destroys temporal coherence and was logged at debug.

    ComfyUI's default level is INFO, so a WAN (1, 16, 21, 60, 104) latent
    becoming 21 independent images left no trace the operator could see.
    """

    def test_flattening_video_frames_logs_at_warning(self, caplog):
        latent = torch.zeros(1, 16, 21, 60, 104)
        with caplog.at_level(logging.WARNING, logger="radiance.core.tensor.contract"):
            out = ensure_4d(latent, "SamplerPro")
        assert out.shape == (21, 16, 60, 104)

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warnings, "the rank change was invisible at ComfyUI's default level"
        message = warnings[0].getMessage()
        assert "21" in message and "temporal coherence" in message

    def test_a_single_frame_squeeze_stays_quiet(self):
        """F == 1 loses nothing, so it must not cry wolf."""
        latent = torch.zeros(1, 16, 1, 60, 104)
        logger = logging.getLogger("radiance.core.tensor.contract")
        records = []
        handler = logging.Handler()
        handler.emit = records.append
        logger.addHandler(handler)
        try:
            out = ensure_4d(latent, "SamplerPro")
        finally:
            logger.removeHandler(handler)
        assert out.shape == (1, 16, 60, 104)
        assert not [r for r in records if r.levelno >= logging.WARNING]

    def test_a_four_dimensional_tensor_is_a_passthrough(self):
        latent = torch.zeros(2, 4, 64, 64)
        assert ensure_4d(latent, "x") is latent


class TestTensorContractRejectsNestedTensors:
    """A bare AttributeError named neither the node nor the input.

    NestedTensor exposes .ndim and .shape but neither .unsqueeze nor .permute,
    so it sailed past the rank checks and died inside the reshape.
    """

    class _Nested:
        ndim = 5
        shape = (1, 16, 21, 60, 104)

    @pytest.mark.parametrize("fn", [ensure_4d, ensure_5d])
    def test_it_raises_a_readable_type_error(self, fn):
        with pytest.raises(TypeError) as exc:
            fn(self._Nested(), "SamplerPro")
        message = str(exc.value)
        assert "SamplerPro" in message, "the error does not say which node"
        assert "NestedTensor" in message or "_Nested" in message
        assert "unpack" in message, "the error offers no way forward"

    @pytest.mark.parametrize("fn", [ensure_4d, ensure_5d])
    def test_it_is_not_an_attribute_error(self, fn):
        with pytest.raises(TypeError):
            fn(self._Nested(), "SamplerPro")


class TestSamplerKeepsPackedVideoLatents:
    """3.5.0 made ensure_5d reject NestedTensor, and SamplerPro still called it
    on every video latent, so LTX-AV and MiniMax H3 failed before sampling."""

    class _Reached(Exception):
        pass

    def test_a_packed_latent_reaches_sampling(self):
        reached = self._Reached

        class Packed(_FakeNestedTensor):
            shape = (1, 24, 2, 4, 4)
            ndim = 5

            def to(self, *args, **kwargs):
                raise reached

        latent = {"samples": Packed((torch.zeros(1, 24, 2, 4, 4), torch.zeros(1, 32, 2, 8)))}
        with pytest.raises(self._Reached):
            run_sampler(latent=latent, model_type="minimax", steps=2)


def test_regional_replace_keeps_the_global_outside_the_region():
    """Audit 3.5: Replace dropped the base from the whole frame."""
    import torch
    from radiance.nodes.generate.regional import RadianceRegionalPrompt
    base = [(torch.zeros(1, 4, 8), {"pooled_output": torch.zeros(1, 8)})]
    region = [(torch.ones(1, 4, 8), {"pooled_output": torch.zeros(1, 8)})]
    out = RadianceRegionalPrompt().apply(base, region, x=0.5, y=0.0, w=0.5, h=1.0,
                                         merge_mode="Replace")[0]
    assert len(out) == 2
    g = out[0][1]
    assert "area" not in g and g["mask"].shape == (1, 256, 256)
    assert float(g["mask"][0, 10, 10]) == 1.0 and float(g["mask"][0, 10, 200]) == 0.0
    assert out[1][1]["area"][0] == "percentage"
