"""
Reference images on the Prompt for the models that are not Qwen-Image 2.1
(tests/test_qwen_image21.py covers that one).
"""
import json

import pytest
import torch
from comfy_api.latest import io

from radiance.nodes.generate import prompt


class _FakeClip:
    def tokenize(self, text, **kwargs):
        return {"t5xxl": [[(1, 1.0)] * 5]}

    def encode_from_tokens_scheduled(self, tokens):
        return [["cond", {}]]


class _FakeVae:
    def __init__(self):
        self.encoded = []

    def encode(self, pixels):
        self.encoded.append(pixels)
        return f"latent{len(self.encoded)}"


class TestFlux2References:
    """Wired as the official Flux.2 Dev and Klein edit templates. Radiance
    resolves Flux.2 Dev to flux2 and the Klein models to flux2-klein."""

    @pytest.fixture(params=["flux2", "flux2-klein"])
    def arch(self, request):
        self.META = json.dumps({"arch": request.param})
        return request.param

    @pytest.fixture
    def natives(self, monkeypatch, arch):
        calls = {"scale": []}

        class _Scale:
            @classmethod
            def execute(cls, image, upscale_method, megapixels, resolution_steps):
                calls["scale"].append((upscale_method, megapixels, resolution_steps))
                return io.NodeOutput(torch.full((1, 48, 64, 4), float(len(calls["scale"]))))

        class _Empty:
            @classmethod
            def execute(cls, width, height, batch_size=1):
                return io.NodeOutput({"samples": ("empty", width, height)})

        class _Reference:
            @classmethod
            def execute(cls, conditioning, latent=None):
                return io.NodeOutput([[c[0], {**c[1], "refs": c[1].get("refs", []) + [latent["samples"]]}]
                                      for c in conditioning])

        monkeypatch.setattr(prompt.nodes_post_processing, "ImageScaleToTotalPixels", _Scale, raising=False)
        monkeypatch.setattr(prompt.nodes_flux, "EmptyFlux2LatentImage", _Empty, raising=False)
        monkeypatch.setattr(prompt.nodes_edit_model, "ReferenceLatent", _Reference, raising=False)
        return calls

    def _run(self, **kwargs):
        vae = _FakeVae()
        images = {"image_2": torch.zeros(1, 20, 20, 3), "image_1": torch.zeros(1, 30, 40, 4)}
        out = prompt.RadianceCinematicPromptEncoder.execute(
            _FakeClip(), base_prompt="Let this character hold the bag", model_meta=self.META,
            vae=vae, images=images, **kwargs)
        return vae, out["result"]

    @pytest.mark.parametrize("negative_mode", ["Auto", "Always encode"])
    def test_every_image_joins_both_prompts_in_order(self, natives, negative_mode):
        """Auto zeroes the negative on Flux.2; the references stay on it, as
        ConditioningZeroOut then ReferenceLatent gives."""
        vae, (pos, neg, *_rest, latent) = self._run(negative_mode=negative_mode)
        assert pos[0][1]["refs"] == ["latent1", "latent2"]
        assert neg[0][1]["refs"] == ["latent1", "latent2"]
        assert natives["scale"] == [("lanczos", 1.0, 1)] * 2
        assert all(p.shape[-1] == 3 for p in vae.encoded)   # the VAE gets RGB, as VAEEncode
        assert latent == {"samples": ("empty", 64, 48)}     # image_1's scaled size

    def test_reference_images_need_the_vae(self, arch):
        with pytest.raises(ValueError, match="vae"):
            prompt.RadianceCinematicPromptEncoder.execute(
                _FakeClip(), base_prompt="a bag", model_meta=self.META,
                images={"image_1": torch.zeros(1, 32, 32, 3)})


class _FakeNestedTensor:
    is_nested = True

    def __init__(self, tensors):
        self.tensors = list(tensors)


class TestMiniMaxFrames:
    """Wired as MiniMaxH3ImageToVideo: image_1 the first frame, image_2 the last,
    at the size and frame count of the latent the Sampler samples on."""

    META = json.dumps({"arch": "minimax"})

    @pytest.fixture
    def native(self, monkeypatch):
        calls = []

        class _ImageToVideo:
            @classmethod
            def execute(cls, **kwargs):
                calls.append(kwargs)
                return io.NodeOutput([["pos", {"minimax_keyframes": ["kf"]}]], {"samples": "av"})

        monkeypatch.setattr(prompt.nodes_minimax_h3, "MiniMaxH3ImageToVideo", _ImageToVideo, raising=False)
        monkeypatch.setattr(prompt.node_helpers, "conditioning_set_values",
                            lambda cond, values: [[c[0], {**c[1], **values}] for c in cond])
        return calls

    def _run(self, images, **kwargs):
        video_latent = {"samples": _FakeNestedTensor((torch.zeros(1, 24, 37, 46, 80), torch.zeros(1, 32, 2, 207)))}
        out = prompt.RadianceCinematicPromptEncoder.execute(
            _FakeClip(), base_prompt="She turns to the camera", model_meta=self.META,
            vae="vae", images=images, video_latent=video_latent, **kwargs)
        return out["result"]

    @pytest.mark.parametrize("negative_mode", ["Auto", "Always encode"])
    def test_the_frames_go_to_the_native_node_and_both_prompts(self, native, negative_mode):
        first, last = torch.zeros(1, 32, 32, 3), torch.ones(1, 32, 32, 3)
        pos, neg, text, *_rest, latent = self._run({"image_2": last, "image_1": first},
                                                   negative_mode=negative_mode)
        (kwargs,) = native
        assert (kwargs["prompt"], kwargs["vae"]) == (text, "vae")
        assert (kwargs["width"], kwargs["height"], kwargs["length"]) == (1280, 736, 124)
        assert kwargs["first_frame"] is first and kwargs["last_frame"] is last
        assert pos[0][1]["minimax_keyframes"] == neg[0][1]["minimax_keyframes"] == ["kf"]
        assert latent == {"samples": "av"}

    def test_each_image_keeps_its_frame_alone(self, native):
        image = torch.zeros(1, 32, 32, 3)
        self._run({"image_1": image})
        self._run({"image_2": image})
        assert (native[0]["first_frame"], native[0]["last_frame"]) == (image, None)
        assert (native[1]["first_frame"], native[1]["last_frame"]) == (None, image)

    def test_a_third_image_is_refused(self, native):
        with pytest.raises(ValueError, match="image_3"):
            self._run({f"image_{i}": torch.zeros(1, 32, 32, 3) for i in (1, 2, 3)})

    def test_a_latent_that_is_no_minimax_video_is_refused(self, native):
        # The Resolution left on a still image gives [B, 4, H/8, W/8]: no time axis to read.
        with pytest.raises(ValueError, match=r"MiniMax H3 \(24ch\)"):
            prompt.RadianceCinematicPromptEncoder.execute(
                _FakeClip(), base_prompt="She turns", model_meta=self.META, vae="vae",
                images={"image_1": torch.zeros(1, 32, 32, 3)}, video_latent={"samples": torch.zeros(1, 4, 64, 64)})
        assert not native

    def test_the_frames_need_the_video_latent(self, native):
        with pytest.raises(ValueError, match="video_latent"):
            prompt.RadianceCinematicPromptEncoder.execute(
                _FakeClip(), base_prompt="She turns", model_meta=self.META, vae="vae",
                images={"image_1": torch.zeros(1, 32, 32, 3)})
