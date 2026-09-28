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


class TestQwenImage21Loader:

    def test_both_presets_select_the_model_type_with_default_dtypes(self):
        from radiance.config.model_map import CHECKPOINT_PRESETS
        for name in ("Qwen-Image 2.1", "Qwen-Image 2.1 (Low VRAM)"):
            assert CHECKPOINT_PRESETS[name] == {
                "model_type": "qwen_image21", "weight_dtype": "default", "clip_dtype": "default",
            }

    def test_offered_by_the_image_loader_not_the_video_loader(self):
        from radiance.config.model_map import VIDEO_PRESET_NAMES, VIDEO_MODEL_TYPES
        from radiance.nodes.generate.loader import MODEL_TYPES
        assert "qwen_image21" in MODEL_TYPES
        assert "qwen_image21" not in VIDEO_MODEL_TYPES
        assert not {"Qwen-Image 2.1", "Qwen-Image 2.1 (Low VRAM)"} & VIDEO_PRESET_NAMES

    def test_catalogue_pins_the_official_files(self):
        from radiance.config.model_map import RADIANCE_MODEL_MAP
        expected = {
            "qwen_image_2.1_bf16.safetensors": "diffusion_models",
            "qwen_image_2.1_int8_convrot.safetensors": "diffusion_models",
            "qwen3vl_8b_bf16.safetensors": "text_encoders",
            "qwen3vl_8b_int8_convrot.safetensors": "text_encoders",
            "qwen3vl_8b_w4a8.safetensors": "text_encoders",
            "qwen_image_2.1_vae_bf16.safetensors": "vae",
        }
        for fname, kind in expected.items():
            entry = RADIANCE_MODEL_MAP[fname]
            assert entry["type"] == kind
            assert entry["url"].startswith("https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/")
            assert entry["url"].endswith(f"{kind}/{fname}")

    def test_prompt_enhancers_are_not_catalogued_as_encoders(self):
        from radiance.config.model_map import RADIANCE_MODEL_MAP
        assert not [f for f in RADIANCE_MODEL_MAP if "_pe_" in f]


class TestQwenImage21Resolution:

    MODEL_TYPE = "Qwen-Image 2.1 (64ch)"

    def test_tables_agree_on_the_model_type(self):
        from radiance.nodes.generate import resolution as R
        assert self.MODEL_TYPE in R.MODEL_TYPES
        assert R.LATENT_FORMAT_MAP[self.MODEL_TYPE] == "qwen_image21"
        assert R.LATENT_CHANNELS[self.MODEL_TYPE] == 64
        assert R.SPATIAL_SCALE[self.MODEL_TYPE] == 16
        assert self.MODEL_TYPE not in R.VIDEO_MODEL_TYPES
        assert self.MODEL_TYPE not in R.SPATIAL_ALIGN

    @pytest.mark.parametrize("w, h", [(1024, 1024), (1280, 720), (2048, 2048)])
    def test_latent_is_64ch_at_one_sixteenth_and_keeps_the_size(self, w, h):
        """720 rows is a 45-row latent. The DiT has no patchify, so an odd
        latent is valid and 1280x720 stays exact."""
        from radiance.nodes.generate.resolution import RadianceResolution
        latent, out_w, out_h, c, _info, _fr, _frames, fmt, *_ = RadianceResolution().generate(
            preset="Custom", width=w, height=h, orientation="As Preset",
            model_type=self.MODEL_TYPE, batch_size=1, unique_id="test",
        )["result"]
        assert tuple(latent["samples"].shape) == (1, 64, h // 16, w // 16)
        assert (out_w, out_h, c, fmt) == (w, h, 64, "qwen_image21")


class TestQwenImage21Sampler:

    def test_tables_follow_the_official_template(self):
        from radiance.sampler_utils import MODEL_DEFAULTS, MODEL_TYPES, CFG_GUIDED_MODELS, GUIDANCE_EMBED_MODELS
        d = MODEL_DEFAULTS["qwen_image21"]
        assert (d["cfg"], d["sampler"], d["scheduler"], d["steps"]) == (1.0, "euler", "simple", 25)
        assert "qwen_image21" in MODEL_TYPES
        assert "qwen_image21" in CFG_GUIDED_MODELS
        assert "qwen_image21" not in GUIDANCE_EMBED_MODELS

    def test_detected_from_its_model_config(self):
        """Without model_meta only detect_by_sampling matched, and it reports
        every flow model with 0-1 sigmas as "flux"."""
        from radiance.sampler_utils import detect_model_type

        class QwenImage21:
            pass

        model = types.SimpleNamespace(model=types.SimpleNamespace(model_config=QwenImage21()))
        assert detect_model_type(model) == "qwen_image21"

    def test_auto_preset_applies_the_defaults_from_model_meta(self):
        import json
        from radiance.nodes.generate.sampler import RadianceSamplerPro
        meta = json.dumps({"arch": "qwen_image21", "unet_file": "qwen_image_2.1_bf16.safetensors"})
        detected, kwargs, _ = RadianceSamplerPro()._configure_model_and_defaults(
            None, "auto", "Auto", model_meta=meta, cfg=1.0, flux_guidance=3.5, steps=20,
            sampler="euler", scheduler="normal", scheduler_mode="Auto (Match Steps)",
        )
        assert detected == "qwen_image21"
        assert (kwargs["cfg"], kwargs["steps"], kwargs["sampler"], kwargs["scheduler"]) == (1.0, 25, "euler", "simple")


class _RGBAVae:
    """Stands in for Qwen-Image 2.1's VAE: 64 latent channels at 16x, RGBA in
    and out, decoding a left-to-right alpha gradient."""
    downscale_ratio = 16
    latent_channels = 64
    output_channels = 4

    def encode(self, pixels):
        self.encoded = pixels
        b, h, w, _ = pixels.shape
        return torch.zeros(b, 64, h // 16, w // 16)

    def decode(self, latent):
        b, _, h, w = latent.shape
        image = torch.full((b, h * 16, w * 16, 4), 0.5)
        image[..., 3] = torch.linspace(0.0, 1.0, w * 16)
        return image


class TestQwenImage21VaeDecodeHDR:

    @pytest.mark.parametrize("target", ["sRGB", "Linear", "ACEScg", "ARRI LogC4"])
    def test_colour_targets_leave_the_alpha_alone(self, target):
        """The colour transform curved every channel: an alpha of 0.5 came
        out at 0.23 in Linear."""
        from radiance.nodes.generate.engine import RadianceHDRVAEDecode
        vae, latent = _RGBAVae(), torch.zeros(1, 64, 2, 2)
        image = RadianceHDRVAEDecode().apply(
            {"samples": latent}, vae, decode_mode="Sampler (SDR-safe)", target_space=target,
        )["result"][0]
        assert image.shape[-1] == 4
        assert torch.allclose(image[..., 3], vae.decode(latent)[..., 3])

    def test_latent_format_label_reads_the_vae_channel_count(self):
        from radiance.hdr.vae import detect_latent_format
        assert detect_latent_format(_RGBAVae()) == "qwen_image21_64ch"


class TestQwenImage21VaeEncodeHDR:

    def test_the_alpha_reaches_an_rgba_vae(self):
        """Only the RGB used to go in, so comfy padded the alpha opaque and
        transparency never reached the latent."""
        from radiance.nodes.generate.engine import RadianceHDRVAEEncode
        vae, pixels = _RGBAVae(), torch.rand(1, 32, 32, 4)
        RadianceHDRVAEEncode().encode(pixels, vae, source_space="sRGB", hdr_mode="Clip (SDR)")
        assert vae.encoded.shape[-1] == 4
        assert torch.allclose(vae.encoded[..., 3].cpu(), pixels[..., 3])

    def test_an_rgb_vae_still_gets_rgb(self):
        from radiance.nodes.generate.engine import RadianceHDRVAEEncode
        vae = _RGBAVae()
        vae.output_channels = 3
        RadianceHDRVAEEncode().encode(torch.rand(1, 32, 32, 4), vae, source_space="sRGB", hdr_mode="Clip (SDR)")
        assert vae.encoded.shape[-1] == 3


class TestSamplerSeedDefault:

    def test_the_seed_defaults_to_one(self):
        """With seed 0 Qwen-Image 2.1 edits fail or come out oversaturated,
        with the native nodes too."""
        from radiance.nodes.generate.sampler import RadianceSamplerPro
        assert RadianceSamplerPro.INPUT_TYPES()["required"]["seed"][1]["default"] == 1
