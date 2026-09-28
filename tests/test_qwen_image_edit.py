"""
Tests for Qwen-Image Edit 2511 support.

Qwen-Image Edit 2511 is a Qwen-Image checkpoint: same DiT keys, same 16ch
VAE and Qwen2.5-VL-7B text encoder, so Radiance resolves it to qwen_image.
"""
import json

import pytest
import torch
from comfy_api.latest import io

from _sampler_harness import FakeModelPatcher, make_latent, run_sampler
from radiance.config.model_map import (
    CHECKPOINT_PRESETS, RADIANCE_MODEL_MAP, VIDEO_MODEL_TYPES, VIDEO_PRESET_NAMES,
)
from radiance.nodes.generate import prompt

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

    def test_tile_mode_tiles_the_single_frame_5d_latent(self):
        """The 5D latent used to skip tiling: every check read ndim == 4."""
        (samples, *_), rec = run_sampler(
            latent={"samples": torch.zeros(1, 16, 1, 64, 64)},
            model=FakeModelPatcher(latent_dimensions=3), model_type="qwen_image",
            tile_mode=True, tile_size=32, tile_overlap=8)
        assert len(rec.calls) > 1
        assert all(c["latent_image"].shape[:3] == (1, 16, 1) for c in rec.calls)
        assert all(c["latent_image"].shape[-2:] == (32, 32) for c in rec.calls)
        assert samples["samples"].shape == (1, 16, 1, 64, 64)
        assert torch.isfinite(samples["samples"]).all()

    def test_a_4d_noise_override_fits_the_5d_latent(self):
        noise = torch.randn(1, 16, 8, 8)
        _, rec = run_sampler(
            latent=make_latent((1, 16, 8, 8)), model=FakeModelPatcher(latent_dimensions=3),
            model_type="qwen_image", noise_override={"samples": noise})
        assert torch.equal(rec.calls[0]["noise"], noise.unsqueeze(2))

    def test_a_missing_latent_names_the_prompt_output(self):
        """Prompt.latent is None without reference images; it used to be called an IMAGE."""
        from _sampler_harness import DEFAULT_KWARGS, SamplerEnv, make_cond
        from radiance.nodes.generate.sampler import RadianceSamplerPro
        kwargs = {**DEFAULT_KWARGS, "model_type": "qwen_image"}
        with SamplerEnv(), pytest.raises(RuntimeError, match="image_1") as exc:
            RadianceSamplerPro().sample(model=FakeModelPatcher(latent_dimensions=3),
                                        positive=make_cond(), negative=make_cond(),
                                        latent_image=None, **kwargs)
        assert "IMAGE" not in str(exc.value)


class _FakeClip:
    def tokenize(self, text, **kwargs):
        return {"qwen25_7b": [[(1, 1.0)] * 5]}

    def encode_from_tokens_scheduled(self, tokens):
        return [["cond", {}]]


class TestQwenImageEditPromptReferences:
    """Wired as the official 2511 templates: image1 through FluxKontextImageScale,
    image2 and image3 as they come, each prompt through TextEncodeQwenImageEditPlus."""

    META = json.dumps({"arch": "qwen_image"})
    SCALED = torch.full((1, 48, 80, 3), 7.0)

    @pytest.fixture
    def encoded(self, monkeypatch):
        calls = []

        class _Scale:
            @classmethod
            def execute(cls, image):
                return io.NodeOutput(self.SCALED)

        class _Empty:
            @classmethod
            def execute(cls, width, height, batch_size=1):
                return io.NodeOutput({"samples": ("empty", width, height)})

        class _EditPlus:
            @classmethod
            def execute(cls, clip, prompt, vae=None, image1=None, image2=None, image3=None):
                calls.append((prompt, vae, image1, image2, image3))
                return io.NodeOutput([[f"cond:{prompt}", {}]])

        monkeypatch.setattr(prompt.nodes_flux, "FluxKontextImageScale", _Scale, raising=False)
        monkeypatch.setattr(prompt.nodes_sd3, "EmptySD3LatentImage", _Empty, raising=False)
        monkeypatch.setattr(prompt.nodes_qwen, "TextEncodeQwenImageEditPlus", _EditPlus, raising=False)
        return calls

    def _run(self, images, **kwargs):
        return prompt.RadianceCinematicPromptEncoder.execute(
            _FakeClip(), base_prompt="Change the leather to fur", model_meta=self.META,
            images=images, **kwargs)["result"]

    def test_both_prompts_read_the_images_in_order(self, encoded):
        fur = torch.ones(1, 20, 20, 3)
        pos, neg, text, neg_text, arch, _count, latent = self._run(
            {"image_2": fur, "image_1": torch.zeros(1, 30, 40, 3)}, vae="vae", negative_strength="Off")
        assert [call[0] for call in encoded] == [text, neg_text]
        # The negative reuses the positive's reference latents, so only the
        # positive VAE-encodes the images.
        assert [call[1] for call in encoded] == ["vae", None]
        for _prompt, _vae, image1, image2, image3 in encoded:
            assert image1 is self.SCALED and image2 is fur and image3 is None
        assert (pos, neg) == ([[f"cond:{text}", {}]], [[f"cond:{neg_text}", {}]])
        assert (arch, latent) == ("qwen_image", {"samples": ("empty", 80, 48)})

    def test_alpha_is_dropped_before_the_encoder(self, encoded):
        rgba = torch.ones(1, 20, 20, 4)
        self._run({"image_1": torch.zeros(1, 30, 40, 3), "image_2": rgba}, vae="vae")
        assert all(call[3].shape[-1] == 3 for call in encoded)

    def test_the_negative_carries_the_positives_reference_latents(self, monkeypatch):
        from radiance.nodes.generate.prompt import nodes_edit_model  # noqa: F401
        vae_calls = []

        class _EditPlus:
            @classmethod
            def execute(cls, clip, prompt, vae=None, image1=None, image2=None, image3=None):
                vae_calls.append(vae)
                refs = {"reference_latents": ["ref1"]} if vae is not None else {}
                return io.NodeOutput([[f"cond:{prompt}", refs]])

        class _RefLatent:
            @classmethod
            def execute(cls, conditioning, latent):
                return io.NodeOutput([[c[0], {**c[1], "reference_latents": [
                    *c[1].get("reference_latents", []), latent["samples"]]}] for c in conditioning])

        monkeypatch.setattr(prompt.nodes_flux, "FluxKontextImageScale",
                            type("S", (), {"execute": classmethod(lambda cls, image: io.NodeOutput(image))}),
                            raising=False)
        monkeypatch.setattr(prompt.nodes_sd3, "EmptySD3LatentImage",
                            type("E", (), {"execute": classmethod(
                                lambda cls, width, height, batch_size=1: io.NodeOutput({"samples": None}))}),
                            raising=False)
        monkeypatch.setattr(prompt.nodes_qwen, "TextEncodeQwenImageEditPlus", _EditPlus, raising=False)
        monkeypatch.setattr(prompt.nodes_edit_model, "ReferenceLatent", _RefLatent, raising=False)
        pos, neg, *_ = self._run({"image_1": torch.zeros(1, 30, 40, 3)}, vae="vae", negative_strength="Off")
        assert vae_calls == ["vae", None]
        assert neg[0][1]["reference_latents"] == pos[0][1]["reference_latents"] == ["ref1"]

    def test_plain_qwen_image_with_references_warns(self, encoded, caplog):
        meta = json.dumps({"arch": "qwen_image", "unet_file": "qwen_image_fp8_e4m3fn.safetensors"})
        with caplog.at_level("WARNING"):
            prompt.RadianceCinematicPromptEncoder.execute(
                _FakeClip(), base_prompt="x", model_meta=meta,
                images={"image_1": torch.zeros(1, 30, 40, 3)})
        assert any("plain Qwen-Image" in r.getMessage() for r in caplog.records)

    def test_the_edit_checkpoint_does_not_warn(self, encoded, caplog):
        meta = json.dumps({"arch": "qwen_image", "unet_file": "qwen_image_edit_2511_fp8mixed.safetensors"})
        with caplog.at_level("WARNING"):
            prompt.RadianceCinematicPromptEncoder.execute(
                _FakeClip(), base_prompt="x", model_meta=meta,
                images={"image_1": torch.zeros(1, 30, 40, 3)})
        assert not any("plain Qwen-Image" in r.getMessage() for r in caplog.records)

    def test_the_vae_is_optional(self, encoded):
        """As on the native encoder: the images then reach the model through the text encoder only."""
        self._run({"image_1": torch.zeros(1, 30, 40, 3)})
        assert [call[1] for call in encoded] == [None, None]

    def test_a_fourth_image_is_refused(self, encoded):
        images = {f"image_{n}": torch.zeros(1, 8, 8, 3) for n in range(1, 5)}
        with pytest.raises(ValueError, match="3 reference images"):
            self._run(images, vae="vae")
