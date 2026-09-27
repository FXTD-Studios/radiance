"""
Tests for Qwen-Image 2.1 support.

Qwen-Image 2.1 is not Qwen-Image with new weights: it has its own DiT keys,
a 64-channel RGBA VAE (16x) and a Qwen3-VL-8B text encoder that comfy.sd
routes to the right encoder only under CLIPType.QWEN_IMAGE.
"""
import sys
import types

import torch
import pytest

from radiance.model import detect as D

HAS_TORCH = isinstance(getattr(torch, "__version__", None), str)
pytestmark = pytest.mark.skipif(
    not HAS_TORCH, reason="writes real safetensors files, needs real torch/safetensors."
)


def _write_shaped_checkpoint(path, tensors):
    from safetensors.torch import save_file
    save_file(tensors, path)


class TestQwenImage21AutoDetect:

    def test_detected_from_its_text_norm_key(self, tmp_path):
        # Widths of the real qwen_image_2.1 checkpoints (bf16 and int8_convrot).
        path = str(tmp_path / "qwen21.safetensors")
        _write_shaped_checkpoint(path, {
            "txt_in.text_norm.weight": torch.zeros(4096),
            "img_in.weight": torch.zeros(4096, 64),
            "modulation.1.weight": torch.zeros(1),
            "proj_out.weight": torch.zeros(64, 8),
        })
        assert D.detect_model_type(path) == "qwen_image21"

    def test_qwen_image_is_still_qwen_image(self, tmp_path):
        path = str(tmp_path / "qwen.safetensors")
        _write_shaped_checkpoint(path, {
            "txt_norm.weight": torch.zeros(3584),
            "img_in.weight": torch.zeros(3072, 64),
        })
        assert D.detect_model_type(path) == "qwen_image"

    def test_comfy_fallback_maps_qwen_image21(self, tmp_path, monkeypatch):
        class QwenImage21:
            latent_format = types.SimpleNamespace(latent_channels=64)

        fake = types.ModuleType("comfy.model_detection")
        fake.model_config_from_unet = lambda sd, prefix, metadata=None: QwenImage21()
        monkeypatch.setitem(sys.modules, "comfy.model_detection", fake)
        monkeypatch.setitem(sys.modules, "comfy", types.ModuleType("comfy"))
        sys.modules["comfy"].model_detection = fake

        path = str(tmp_path / "mystery.safetensors")
        _write_shaped_checkpoint(path, {"img_in.weight": torch.zeros(8, 64)})
        assert D.detect_model_type(path) == "qwen_image21"


class TestQwenImage21Tables:

    def test_latent_is_64ch_at_16x_with_no_temporal_axis(self):
        assert D.LATENT_CHANNELS["qwen_image21"] == 64
        assert D.VAE_SPATIAL_FACTOR["qwen_image21"] == 16
        assert "qwen_image21" not in D.VAE_TEMPORAL_FACTOR
        assert D.latent_format("qwen_image21") == "qwen_image21_64ch"

    def test_one_llm_encoder_slot(self):
        assert D.CLIP_SLOT_ORDER["qwen_image21"] == ["llm_encoder"]

    def test_has_vram_estimates(self):
        assert D._BASE_VRAM["qwen_image21"] > 0
        assert D._BASE_CLIP_VRAM["qwen_image21"] > 0

    def test_clip_type_is_qwen_image(self, monkeypatch):
        """No CLIPType is named after 2.1: without the override the generic
        lookup finds nothing and falls back to STABLE_DIFFUSION."""
        clip_type = types.SimpleNamespace(
            FLUX="FLUX", SD3="SD3", STABLE_DIFFUSION="STABLE_DIFFUSION", QWEN_IMAGE="QWEN_IMAGE",
        )
        monkeypatch.setattr(D.comfy.sd, "CLIPType", clip_type, raising=False)
        assert D.get_clip_type_enum("qwen_image21") == "QWEN_IMAGE"
