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
        with pytest.raises(ValueError, match="first_frame and last_frame"):
            self._run({f"image_{i}": torch.zeros(1, 32, 32, 3) for i in (1, 2, 3)})

    def test_first_frame_and_last_frame_go_to_the_native_node(self, native):
        first, last = torch.zeros(1, 32, 32, 3), torch.ones(1, 32, 32, 3)
        self._run(None, first_frame=first, last_frame=last)
        self._run(None, last_frame=last)
        assert (native[0]["first_frame"], native[0]["last_frame"]) == (first, last)
        assert (native[1]["first_frame"], native[1]["last_frame"]) == (None, last)

    def test_frames_and_images_together_are_refused(self, native):
        with pytest.raises(ValueError, match="not both"):
            self._run({"image_1": torch.zeros(1, 32, 32, 3)}, last_frame=torch.ones(1, 32, 32, 3))

    def test_another_family_leaves_the_frames_out_with_a_warning(self, native, monkeypatch):
        warnings = []
        monkeypatch.setattr(prompt.logger, "warning", lambda message, *args: warnings.append(message % args))
        prompt.RadianceCinematicPromptEncoder.execute(
            _FakeClip(), base_prompt="A cat", model_meta=json.dumps({"arch": "flux3"}), vae="vae",
            first_frame=torch.zeros(1, 32, 32, 3))
        assert not native
        assert any("first_frame and last_frame are read by MiniMax H3 and Wan" in w and "'flux3'" in w for w in warnings)

    def test_a_latent_that_is_no_minimax_video_is_refused(self, native):
        # The Resolution left on a still image gives [B, 4, H/8, W/8]: no time axis to read.
        with pytest.raises(ValueError, match=r"MiniMax H3 \(24ch\)"):
            prompt.RadianceCinematicPromptEncoder.execute(
                _FakeClip(), base_prompt="She turns", model_meta=self.META, vae="vae",
                images={"image_1": torch.zeros(1, 32, 32, 3)}, video_latent={"samples": torch.zeros(1, 4, 64, 64)})
        assert not native

    def test_a_batch_of_videos_is_refused(self, native):
        latent = {"samples": torch.zeros(2, 24, 37, 46, 80)}
        with pytest.raises(ValueError, match="video_latent holds 2"):
            prompt.RadianceCinematicPromptEncoder.execute(
                _FakeClip(), base_prompt="She turns", model_meta=self.META, vae="vae",
                first_frame=torch.zeros(1, 32, 32, 3), video_latent=latent)
        assert not native

    def test_the_frames_need_the_video_latent(self, native):
        with pytest.raises(ValueError, match="video_latent"):
            prompt.RadianceCinematicPromptEncoder.execute(
                _FakeClip(), base_prompt="She turns", model_meta=self.META, vae="vae",
                images={"image_1": torch.zeros(1, 32, 32, 3)})


class TestWanFrames:
    """Wired as the official Wan templates: WanImageToVideo, WanFirstLastFrameToVideo with a last
    frame, Wan22ImageToVideoLatent on the 5B, at the size and frame count of video_latent."""

    @pytest.fixture
    def natives(self, monkeypatch):
        calls = []
        for name in ("WanImageToVideo", "WanFirstLastFrameToVideo", "Wan22ImageToVideoLatent"):
            def execute(cls, _name=name, **kwargs):
                calls.append((_name, kwargs))
                if _name == "Wan22ImageToVideoLatent":
                    return io.NodeOutput({"samples": "ti2v"})
                return io.NodeOutput([["pos", {"by": _name}]], [["neg", {"by": _name}]], {"samples": "i2v"})
            monkeypatch.setattr(prompt.nodes_wan, name, type(name, (), {"execute": classmethod(execute)}),
                                raising=False)
        return calls

    def _run(self, arch, latent_shape, **kwargs):
        out = prompt.RadianceCinematicPromptEncoder.execute(
            _FakeClip(), base_prompt="She turns to the camera", model_meta=json.dumps({"arch": arch}),
            vae="vae", video_latent={"samples": torch.zeros(latent_shape)}, **kwargs)
        return out["result"]

    def test_a_first_frame_goes_to_wan_image_to_video_at_the_videos_size(self, natives):
        first = torch.zeros(1, 32, 32, 3)
        pos, neg, *_rest, latent = self._run("wan", (1, 16, 21, 88, 160), first_frame=first)
        ((name, kwargs),) = natives
        assert name == "WanImageToVideo"
        assert (kwargs["width"], kwargs["height"], kwargs["length"], kwargs["batch_size"]) == (1280, 704, 81, 1)
        assert kwargs["start_image"] is first and kwargs["vae"] == "vae"
        assert pos[0][1]["by"] == neg[0][1]["by"] == "WanImageToVideo"
        assert latent == {"samples": "i2v"}

    def test_a_last_frame_goes_to_the_first_last_frame_node(self, natives):
        first, last = torch.zeros(1, 32, 32, 3), torch.ones(1, 32, 32, 3)
        self._run("wan", (1, 16, 21, 88, 160), first_frame=first, last_frame=last)
        self._run("wan", (1, 16, 21, 88, 160), last_frame=last)
        assert [name for name, _ in natives] == ["WanFirstLastFrameToVideo"] * 2
        assert (natives[0][1]["start_image"], natives[0][1]["end_image"]) == (first, last)
        assert (natives[1][1]["start_image"], natives[1][1]["end_image"]) == (None, last)

    def test_the_5b_puts_its_first_frame_in_the_latent(self, natives):
        first = torch.zeros(1, 32, 32, 3)
        *_rest, latent = self._run("wan_ti2v", (1, 48, 31, 44, 80), first_frame=first)
        ((name, kwargs),) = natives
        assert name == "Wan22ImageToVideoLatent"
        assert (kwargs["width"], kwargs["height"], kwargs["length"]) == (1280, 704, 121)
        assert kwargs["start_image"] is first and latent == {"samples": "ti2v"}

    def test_the_5b_refuses_a_last_frame(self, natives):
        with pytest.raises(ValueError, match="first frame only"):
            self._run("wan_ti2v", (1, 48, 31, 44, 80), last_frame=torch.ones(1, 32, 32, 3))
        assert not natives

    def test_a_latent_that_is_no_wan_video_is_refused(self, natives):
        with pytest.raises(ValueError, match=r"WAN TI2V \(48ch\)"):
            self._run("wan_ti2v", (1, 16, 21, 88, 160), first_frame=torch.zeros(1, 32, 32, 3))
        assert not natives

    def test_without_a_frame_wan_encodes_the_text_alone(self, natives):
        *_rest, latent = self._run("wan", (1, 16, 21, 88, 160))
        assert not natives and latent is None
