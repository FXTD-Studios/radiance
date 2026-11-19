"""Tests for sampler_utils.py model_meta parsing / distillation refinement."""
import json

from radiance.sampler_utils import (CFG_GUIDED_MODELS, GUIDANCE_EMBED_MODELS, parse_model_meta,
                                    refine_distillation_from_meta)


class TestParseModelMeta:
    def test_empty_string(self):
        assert parse_model_meta("") == ("", "", [])

    def test_malformed_json(self):
        assert parse_model_meta("{not json") == ("", "", [])

    def test_valid_json(self):
        meta = json.dumps({"arch": "flux2-klein", "unet_file": "flux-2-klein-9b.safetensors",
                           "loras": [{"name": "style.safetensors", "model_str": 1.0, "clip_str": 1.0}]})
        assert parse_model_meta(meta) == ("flux2-klein", "flux-2-klein-9b.safetensors", ["style.safetensors"])

    def test_missing_fields(self):
        assert parse_model_meta(json.dumps({"arch": "flux2"})) == ("flux2", "", [])


class TestRefineDistillationFromMeta:
    def test_no_unet_file_returns_none(self):
        assert refine_distillation_from_meta("flux2-klein", "") is None

    def test_unrelated_type_returns_none(self):
        assert refine_distillation_from_meta("sdxl", "sd_xl_base_1.0.safetensors") is None

    def test_klein_base_variant(self):
        result = refine_distillation_from_meta("flux2-klein", "flux-2-klein-base-4b.safetensors")
        assert result == {"cfg": 4.0, "steps": 50}

    def test_klein_distilled_variant(self):
        result = refine_distillation_from_meta("flux2-klein", "flux-2-klein-9b.safetensors")
        assert result == {"cfg": 1.0, "steps": 4}

    def test_klein_runs_a_real_cfg(self):
        # The Klein DiT has no guidance_in: a flux_guidance value would be ignored.
        assert "flux2-klein" in CFG_GUIDED_MODELS
        assert "flux2-klein" not in GUIDANCE_EMBED_MODELS

    def test_klein_base_case_insensitive(self):
        result = refine_distillation_from_meta("flux2-klein", "FLUX-2-KLEIN-BASE-9B-FP8.safetensors")
        assert result == {"cfg": 4.0, "steps": 50}

    def test_flux2_dev_not_affected(self):
        # Flux.2 Dev has no distilled counterpart -- always None, MODEL_DEFAULTS'
        # guidance=4.0 fallback applies unchanged.
        assert refine_distillation_from_meta("flux2", "flux2-dev.safetensors") is None

    def test_flux1_schnell(self):
        result = refine_distillation_from_meta("flux", "flux1-schnell-fp8.safetensors")
        assert result == {"guidance": 0.0, "steps": 4}

    def test_flux1_dev_not_affected(self):
        assert refine_distillation_from_meta("flux", "flux1-dev-fp8.safetensors") is None

    def test_flux1_krea_dev_guidance_only(self):
        # BFL's model card gives no steps recommendation for Krea Dev --
        # unlike Klein/Schnell, the result must have no "steps" key at all.
        result = refine_distillation_from_meta("flux", "flux1-krea-dev.safetensors")
        assert result == {"guidance": 4.5}
        assert "steps" not in result

    def test_flux1_kontext_dev_guidance_only(self):
        result = refine_distillation_from_meta("flux", "flux1-dev-kontext_fp8_scaled.safetensors")
        assert result == {"guidance": 2.5}

    def test_sdxl_turbo(self):
        result = refine_distillation_from_meta("sdxl", "sd_xl_turbo_1.0_fp16.safetensors")
        assert result == {"cfg": 1.0, "steps": 1, "sampler": "euler_ancestral"}

    def test_sd35_turbo(self):
        result = refine_distillation_from_meta("sd3.5", "sd3.5_large_turbo.safetensors")
        assert result == {"cfg": 1.6, "steps": 4}

    def test_qwen_image_edit_2511(self):
        # The official templates' KSampler widgets read 40/3, but switches feed
        # it 40 steps and cfg 4 when the Lightning LoRA is off.
        for f in ("qwen_image_edit_2511_int8_convrot.safetensors", "Qwen-Image-Edit-2511-bf16.safetensors"):
            assert refine_distillation_from_meta("qwen_image", f) == {"cfg": 4.0, "steps": 40}

    def test_qwen_image_lightning_lora_sets_its_steps_at_cfg_1(self):
        # The official templates' LoRAs, over Edit 2511's own 40 steps, cfg 4.
        for lora, steps in (("Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors", 4),
                            ("qwen/Qwen-Image-Lightning-8steps-V1.0.safetensors", 8)):
            loras = ["style.safetensors", lora]
            assert refine_distillation_from_meta("qwen_image", "qwen_image_edit_2511_int8_convrot.safetensors",
                                                 loras) == {"cfg": 1.0, "steps": steps}
            assert refine_distillation_from_meta("qwen_image", "", loras) == {"cfg": 1.0, "steps": steps}

    def test_qwen_image_2512_runs_50_steps(self):
        assert refine_distillation_from_meta("qwen_image", "qwen_image_2512_fp8_e4m3fn.safetensors") == {"steps": 50}

    def test_minimax_turbo_lora_sets_its_steps(self):
        for lora, steps in (("minimax_h3_fl2v_turbo_8step_v1.1_768p_comfyui_bf16.safetensors", 8),
                            ("minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors", 4)):
            assert refine_distillation_from_meta("minimax", "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
                                                 [lora]) == {"cfg": 1.0, "steps": steps}
        assert refine_distillation_from_meta("minimax", "minimax_h3_fl2va_pruned_int8_convrot.safetensors") is None

    def test_turbo_steps_come_from_the_lora_file_name_not_its_folder(self):
        unet = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
        assert refine_distillation_from_meta("minimax", unet, ["turbo_8step/style.safetensors"]) is None
        assert refine_distillation_from_meta(
            "minimax", unet, ["lightx2v\\minimax_h3_fl2v_turbo_8step_v1.1_768p_comfyui_bf16.safetensors"]
        ) == {"cfg": 1.0, "steps": 8}

    def test_a_lora_at_model_strength_0_is_left_out(self):
        meta = json.dumps({"arch": "minimax", "unet_file": "unet.safetensors", "loras": [
            {"name": "minimax_h3_fl2v_turbo_8step_v1.1_768p_comfyui_bf16.safetensors", "model_str": 0, "clip_str": 1.0},
            {"name": "style.safetensors", "model_str": 0.8, "clip_str": 1.0}]})
        assert parse_model_meta(meta) == ("minimax", "unet.safetensors", ["style.safetensors"])

    def test_lightning_lora_only_for_qwen_image(self):
        assert refine_distillation_from_meta("sdxl", "sd_xl_base_1.0.safetensors",
                                             ["sdxl_lightning_4step_lora.safetensors"]) is None

    def test_qwen_image_and_older_edits_not_affected(self):
        for f in ("qwen_image_fp8_e4m3fn.safetensors", "qwen_image_edit_2509_fp8_e4m3fn.safetensors"):
            assert refine_distillation_from_meta("qwen_image", f) is None

    def test_sd35_medium_not_affected(self):
        # "turbo" substring absent -- SD3.5 Medium has no Turbo variant.
        assert refine_distillation_from_meta("sd3.5", "sd3.5_medium.safetensors") is None

    def test_ltx_distilled_runs_at_cfg_1(self):
        for f in ("ltx-2.3-22b-distilled-1.1.safetensors", "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"):
            assert refine_distillation_from_meta("ltxav", f) == {"cfg": 1.0}

    def test_wan22_experts_run_euler_at_cfg_3_5(self):
        # Wan 2.2 14B's templates run euler at cfg 3.5-4; Wan 2.1 keeps its own uni_pc at cfg 6.
        for f in ("wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors", "Wan/wan2.2_i2v_low_noise_14B_fp16.safetensors"):
            assert refine_distillation_from_meta("wan", f) == {"sampler": "euler", "cfg": 3.5}
        assert refine_distillation_from_meta("wan", "wan2.1_t2v_14B_fp16.safetensors") is None

    def test_wan22_lightx2v_lora_runs_its_steps_at_cfg_1(self):
        # The lightx2v 4-step LoRAs of Wan 2.2's templates: 4 steps, cfg 1, euler.
        lora = "wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors"
        assert refine_distillation_from_meta("wan", "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors", [lora]) == {
            "cfg": 1.0, "steps": 4, "sampler": "euler"}

    def test_wan21_lora_follows_the_steps_in_its_name_too(self):
        # A step count in a LoRA's name means a step distillation, whatever the Wan checkpoint;
        # lightx2v's Wan 2.1 LoRAs name none and keep Wan 2.1's defaults.
        wan21 = "wan2.1_t2v_14B_fp16.safetensors"
        assert refine_distillation_from_meta("wan", wan21, ["wan2.1_t2v_lightx2v_4steps_lora.safetensors"]) == {
            "cfg": 1.0, "steps": 4, "sampler": "euler"}
        assert refine_distillation_from_meta(
            "wan", wan21, ["Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors"]) is None

    def test_ltx_dev_not_affected_even_with_the_distilled_lora(self):
        # The Dev workflows add the distilled LoRA at a low strength and keep the Dev recipe.
        assert refine_distillation_from_meta("ltxav", "ltx-2.3-22b-dev.safetensors",
                                             ["ltx-2.3-22b-distilled-lora-384.safetensors"]) is None


def _auto_ltx_cfg(unet, denoise):
    from radiance.nodes.generate.sampler import RadianceSamplerPro
    meta = json.dumps({"arch": "ltxav", "unet_file": unet, "loras": []})
    _, kwargs, _ = RadianceSamplerPro()._configure_model_and_defaults(
        None, "auto", "Auto", model_meta=meta, cfg=1.0, flux_guidance=3.5, steps=3, denoise=denoise,
        sampler="euler", scheduler="simple", scheduler_mode="Manual")
    return kwargs["cfg"]


def test_auto_ltx_cfg_follows_the_stage():
    # cfg 1 is the upscaled pass's (denoise below 1) and the Distilled model's own value:
    # Auto used to raise it to 3, a negative pass on every step. Dev LowRes still gets 3.
    dev, distilled = "ltx-2.5-22b-dev-transformer-bf16.safetensors", "ltx-2.3-22b-distilled-1.1.safetensors"
    assert _auto_ltx_cfg(dev, 1.0) == 3.0
    assert _auto_ltx_cfg(dev, 0.45) == 1.0
    assert _auto_ltx_cfg(distilled, 1.0) == 1.0
    assert _auto_ltx_cfg(distilled, 0.375) == 1.0


def test_auto_wan22_follows_its_templates():
    from radiance.nodes.generate.sampler import RadianceSamplerPro
    meta = json.dumps({"arch": "wan", "unet_file": "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors", "loras": []})
    _, kwargs, _ = RadianceSamplerPro()._configure_model_and_defaults(
        None, "auto", "Auto", model_meta=meta, cfg=1.0, flux_guidance=3.5, steps=20, denoise=1.0,
        sampler="euler", scheduler="simple", scheduler_mode="Manual")
    assert (kwargs["sampler"], kwargs["cfg"], kwargs["steps"]) == ("euler", 3.5, 20)
