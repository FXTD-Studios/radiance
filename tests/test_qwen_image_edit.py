"""
Tests for Qwen-Image Edit 2511 support.

Qwen-Image Edit 2511 is a Qwen-Image checkpoint: same DiT keys, same 16ch
VAE and Qwen2.5-VL-7B text encoder, so Radiance resolves it to qwen_image.
"""
import json

import pytest

from _sampler_harness import FakeModelPatcher, make_latent, run_sampler
from radiance.config.model_map import (
    CHECKPOINT_PRESETS, RADIANCE_MODEL_MAP, VIDEO_MODEL_TYPES, VIDEO_PRESET_NAMES,
)

PRESET = "Qwen-Image Edit 2511"


class TestQwenImageEditLoader:

    def test_preset_selects_qwen_image_with_default_dtypes(self):
        assert CHECKPOINT_PRESETS[PRESET] == {
            "model_type": "qwen_image", "weight_dtype": "default", "clip_dtype": "default",
        }

    def test_offered_by_the_image_loader_not_the_video_loader(self):
        from radiance.nodes.generate.loader import MODEL_TYPES
        assert "qwen_image" in MODEL_TYPES
        assert "qwen_image" not in VIDEO_MODEL_TYPES
        assert PRESET not in VIDEO_PRESET_NAMES

    def test_catalogue_pins_the_templates_files(self):
        expected = {
            "qwen_image_edit_2511_int8_convrot.safetensors": ("Qwen-Image-Edit_ComfyUI", "diffusion_models"),
            "qwen_image_edit_2511_fp8mixed.safetensors": ("Qwen-Image-Edit_ComfyUI", "diffusion_models"),
            "qwen_2.5_vl_7b_fp8_scaled.safetensors": ("HunyuanVideo_1.5_repackaged", "text_encoders"),
            "qwen_image_vae.safetensors": ("Qwen-Image_ComfyUI", "vae"),
        }
        for fname, (repo, kind) in expected.items():
            entry = RADIANCE_MODEL_MAP[fname]
            assert entry["type"] == kind
            assert entry["url"].startswith(f"https://huggingface.co/Comfy-Org/{repo}/resolve/")
            assert entry["url"].endswith(f"split_files/{kind}/{fname}")


class TestQwenImageEditSampler:

    def test_auto_preset_applies_the_template_values_from_model_meta(self):
        """Qwen-Image's own defaults are 20 steps, cfg 2.5; only the file name
        tells the Edit checkpoint apart."""
        from radiance.nodes.generate.sampler import RadianceSamplerPro
        meta = json.dumps({"arch": "qwen_image", "unet_file": "qwen_image_edit_2511_int8_convrot.safetensors"})
        detected, kwargs, _ = RadianceSamplerPro()._configure_model_and_defaults(
            None, "auto", "Auto", model_meta=meta, cfg=1.0, flux_guidance=3.5, steps=20,
            sampler="euler", scheduler="normal", scheduler_mode="Auto (Match Steps)",
        )
        assert detected == "qwen_image"
        assert (kwargs["cfg"], kwargs["steps"], kwargs["sampler"], kwargs["scheduler"]) == (4.0, 40, "euler", "simple")

    @pytest.mark.parametrize("shape", [(1, 16, 1, 8, 8), (1, 16, 8, 8)])
    def test_the_latent_is_sampled_in_5d_like_the_native_ksampler(self, shape):
        """Qwen-Image's latent format is Wan 2.1's (latent_dimensions 3). A VAE
        Encode latent is (B, C, 1, H, W); an empty 4D one gains the frame axis,
        as comfy.sample.fix_empty_latent_channels does. Squeezed to 4D, the
        format's 5D mean broadcast it into 16 frames of noise."""
        _, rec = run_sampler(latent=make_latent(shape), model=FakeModelPatcher(latent_dimensions=3),
                             model_type="qwen_image")
        call = rec.calls[0]
        assert call["latent_image"].shape == call["noise"].shape == (1, 16, 1, 8, 8)
