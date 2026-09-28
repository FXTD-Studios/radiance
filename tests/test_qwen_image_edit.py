"""
Tests for Qwen-Image Edit 2511 support.

Qwen-Image Edit 2511 is a Qwen-Image checkpoint: same DiT keys, same 16ch
VAE and Qwen2.5-VL-7B text encoder, so Radiance resolves it to qwen_image.
"""
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
