"""Tests for sampler_utils.py model_meta parsing / distillation refinement."""
import json

from radiance.sampler_utils import parse_model_meta, refine_distillation_from_meta


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
        assert result == {"guidance": 4.0, "steps": 50}

    def test_klein_distilled_variant(self):
        result = refine_distillation_from_meta("flux2-klein", "flux-2-klein-9b.safetensors")
        assert result == {"guidance": 1.0, "steps": 4}

    def test_klein_base_case_insensitive(self):
        result = refine_distillation_from_meta("flux2-klein", "FLUX-2-KLEIN-BASE-9B-FP8.safetensors")
        assert result == {"guidance": 4.0, "steps": 50}

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

    def test_lightning_lora_only_for_qwen_image(self):
        assert refine_distillation_from_meta("sdxl", "sd_xl_base_1.0.safetensors",
                                             ["sdxl_lightning_4step_lora.safetensors"]) is None

    def test_qwen_image_and_older_edits_not_affected(self):
        for f in ("qwen_image_fp8_e4m3fn.safetensors", "qwen_image_edit_2509_fp8_e4m3fn.safetensors"):
            assert refine_distillation_from_meta("qwen_image", f) is None

    def test_sd35_medium_not_affected(self):
        # "turbo" substring absent -- SD3.5 Medium has no Turbo variant.
        assert refine_distillation_from_meta("sd3.5", "sd3.5_medium.safetensors") is None
