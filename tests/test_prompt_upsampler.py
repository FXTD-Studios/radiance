"""The Prompt's optional upsampler for FLUX 3 Image Klein and Qwen-Image 2.1.

ComfyUI's TextGenerate is replaced by a recorder: the tests check what the
Prompt hands it (system prompt, sampling, images) and that the rewrite is what
gets encoded, not the model's output.
"""

import json
import sys
import types

import pytest
import torch

from comfy_api.latest import io


class Flux3Tokenizer:
    pass


class Qwen3VLTokenizer:          # stands in for Qwen-Image 2.1's encoder
    pass


class _Clip:
    def __init__(self, tokenizer_cls=Flux3Tokenizer):
        self.tokenizer = tokenizer_cls()
        self.texts = []

    def tokenize(self, text, **kwargs):
        if text != "test":
            self.texts.append(text)
        return {"qwen3vl_4b": [[(1, 1.0)] * 5]}

    def encode_from_tokens_scheduled(self, tokens):
        return [["cond", {}]]


@pytest.fixture
def textgen(monkeypatch):
    """A recording TextGenerate with the real node's parameters."""
    calls = []

    class TextGenerate:
        @classmethod
        def execute(cls, clip, prompt, max_length, sampling_mode, image=None, thinking=False,
                    use_default_template=True, video=None, audio=None, mtp="auto", system_prompt=""):
            calls.append(dict(clip=clip, prompt=prompt, max_length=max_length, sampling_mode=sampling_mode,
                              image=image, thinking=thinking, use_default_template=use_default_template,
                              mtp=mtp, system_prompt=system_prompt))
            return io.NodeOutput("  A detailed caption of a red fox standing in fresh snow at sunrise.  ", "")

    mod = types.ModuleType("comfy_extras.nodes_textgen")
    mod.TextGenerate = TextGenerate
    import comfy_extras
    monkeypatch.setattr(comfy_extras, "nodes_textgen", mod, raising=False)
    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_textgen", mod)
    return calls


def _encode(clip=None, **kwargs):
    from radiance.nodes.generate import prompt
    kwargs.setdefault("base_prompt", "a red fox in snow")
    return prompt.RadianceCinematicPromptEncoder.execute(clip or _Clip(), **kwargs)["result"]


def test_off_by_default_and_the_upsampler_is_never_called(textgen):
    clip = _Clip()
    _pos, _neg, text, *_ = _encode(clip, upsampler_clip=object())
    assert textgen == []
    assert "fox" in text


def test_flux3_rewrite_is_what_gets_encoded(textgen):
    from radiance.nodes.generate import upsampler
    up, clip = object(), _Clip()
    _pos, _neg, text, _n, arch, *_ = _encode(clip, upsampler_clip=up, upsample_prompt=True, upsampler_seed=7)
    assert arch == "flux3"
    assert text == "A detailed caption of a red fox standing in fresh snow at sunrise."
    assert clip.texts[0] == text
    (call,) = textgen
    assert call["clip"] is up
    assert "a red fox in snow" in call["prompt"]
    assert call["system_prompt"] == upsampler.system_prompt("flux3", "t2i")
    assert call["system_prompt"].startswith("You are the prompt upsampler of a text-to-image model")
    # The official FLUX 3 Klein template's settings.
    assert call["max_length"] == 3584 and call["thinking"] is False and call["mtp"] == "off"
    s = call["sampling_mode"]
    assert (s["sampling_mode"], s["temperature"], s["top_k"], s["top_p"], s["seed"], s["presence_penalty"]) == \
        ("on", 1.0, 1, 0.95, 7, 1.5)
    assert call["image"] is None


def test_qwen_image21_uses_its_own_instructions_and_thinking(textgen, monkeypatch):
    from radiance.nodes.generate import prompt, upsampler
    monkeypatch.setattr(prompt, "_detect_arch_from_clip", lambda *a, **k: "qwen_image21")
    _encode(_Clip(Qwen3VLTokenizer), upsampler_clip=object(), upsample_prompt=True)
    (call,) = textgen
    assert call["system_prompt"] == upsampler.system_prompt("qwen_image21", "t2i")
    assert call["system_prompt"].startswith("# Image Prompt Rewriting Expert")
    assert call["max_length"] == 4096 and call["thinking"] is True and call["mtp"] == "auto"
    assert call["sampling_mode"]["top_k"] == 20


def test_reference_images_switch_to_the_edit_instructions_and_reach_the_upsampler(textgen, monkeypatch):
    from radiance.nodes.generate import prompt, upsampler

    class _Vae:
        def encode(self, pixels):
            b, h, w, _ = pixels.shape
            return torch.ones(b, 128, h // 16, w // 16)

    class ReferenceLatent:
        @classmethod
        def execute(cls, conditioning, latent=None):
            return io.NodeOutput(conditioning)
    monkeypatch.setattr(prompt.nodes_edit_model, "ReferenceLatent", ReferenceLatent, raising=False)
    batched = []

    class BatchImagesNode:
        @classmethod
        def execute(cls, images):
            batched.append([tuple(v.shape) for v in images.values()])
            return io.NodeOutput(torch.cat(list(images.values()), dim=0))
    monkeypatch.setattr(prompt.nodes_post_processing, "BatchImagesNode", BatchImagesNode, raising=False)
    images = {"image_1": torch.zeros(1, 64, 64, 4), "image_2": torch.zeros(1, 64, 64, 3)}
    _encode(base_prompt="In <ref_image_0>, make the dress red", vae=_Vae(), images=images, resolution=0,
            upsampler_clip=object(), upsample_prompt=True)
    (call,) = textgen
    assert call["system_prompt"] == upsampler.system_prompt("flux3", "edit")
    assert call["sampling_mode"]["presence_penalty"] == 0.0
    assert batched == [[(1, 64, 64, 3), (1, 64, 64, 3)]]           # RGB, every reference
    assert call["image"].shape == (2, 64, 64, 3)


def test_the_flux3_layout_table_stays_out_of_the_upsampler_and_is_kept(textgen):
    table = json.dumps([{"id": "obj_1", "bbox": [100, 200, 800, 900], "desc": "a red silk dress"}])
    clip = _Clip()
    _pos, _neg, text, *_ = _encode(clip, base_prompt=f"A woman wearing <obj_1>. {table}",
                                   upsampler_clip=object(), upsample_prompt=True)
    assert "bbox" not in textgen[0]["prompt"]
    assert text.endswith(" " + table)
    assert text.startswith("A detailed caption")


def test_on_without_the_upsampler_clip_says_what_to_load(textgen):
    with pytest.raises(ValueError, match="qwen3.5_9b_flux3_prompt_upsampler"):
        _encode(upsample_prompt=True)


def test_another_family_is_refused(textgen, monkeypatch):
    from radiance.nodes.generate import prompt
    monkeypatch.setattr(prompt, "_detect_arch_from_clip", lambda *a, **k: "sdxl")
    with pytest.raises(ValueError, match="FLUX 3 Image Klein and Qwen-Image 2.1"):
        _encode(upsampler_clip=object(), upsample_prompt=True)
    assert textgen == []


def test_an_empty_rewrite_is_an_error_not_an_empty_prompt(textgen, monkeypatch):
    import comfy_extras
    def blank(cls, clip, prompt, max_length, sampling_mode, image=None, thinking=False,
              use_default_template=True, mtp="auto", system_prompt=""):
        return io.NodeOutput("   ", "")
    monkeypatch.setattr(comfy_extras.nodes_textgen.TextGenerate, "execute", classmethod(blank))
    with pytest.raises(ValueError, match="returned no text"):
        _encode(upsampler_clip=object(), upsample_prompt=True)


def test_inputs_are_appended_after_every_older_one():
    from radiance.nodes.generate import prompt
    names = [i.id for i in prompt.RadianceCinematicPromptEncoder.define_schema().inputs]
    assert names[-4:] == ["upsampler_clip", "upsample_prompt", "upsampler_seed", "layout"]
    assert names.index("images") == len(names) - 5


def test_the_system_prompts_ship_with_the_node():
    from radiance.nodes.generate import upsampler
    for arch, spec in upsampler.UPSAMPLERS.items():
        for mode in ("t2i", "edit"):
            assert len(upsampler.system_prompt(arch, mode)) > 1000, (arch, mode)


# ── The Loader's upsampler slot ──────────────────────────────────────────────

def test_the_loader_loads_the_upsampler_once_as_load_clip_does(monkeypatch):
    from radiance.nodes.generate import loader
    loads = []
    monkeypatch.setattr(loader.folder_paths, "get_full_path_or_raise", lambda kind, name: f"{kind}/{name}",
                        raising=False)
    monkeypatch.setattr(loader.folder_paths, "get_folder_paths", lambda kind: [], raising=False)
    monkeypatch.setattr(loader, "_file_fingerprint", lambda path: "fp")
    monkeypatch.setattr(loader, "get_clip_type_enum", lambda arch: f"CLIPType for {arch}")
    monkeypatch.setattr(loader.comfy.sd, "load_clip",
                        lambda ckpt_paths, clip_type, embedding_directory, model_options=None:
                        loads.append((ckpt_paths, clip_type)) or "llm",
                        raising=False)
    monkeypatch.setattr(loader, "_upsampler_cache", loader.LRUCache())
    info = []
    for _ in range(2):
        assert loader._load_upsampler("pup.safetensors", "flux3", True, info) == "llm"
    assert loads == [(["text_encoders/pup.safetensors"], "CLIPType for flux3")]
    assert info[-1] == "UPSAMPLER: pup.safetensors (cached)"


def test_the_upsampler_follows_the_offload_device(monkeypatch):
    """CPU offload put the text encoder on the CPU and the 9B upsampler on the GPU
    (CodeRabbit review on PR #71); the device is part of the cache key."""
    from radiance.nodes.generate import loader
    opts = []
    monkeypatch.setattr(loader.folder_paths, "get_full_path_or_raise", lambda kind, name: f"{kind}/{name}",
                        raising=False)
    monkeypatch.setattr(loader.folder_paths, "get_folder_paths", lambda kind: [], raising=False)
    monkeypatch.setattr(loader, "_file_fingerprint", lambda path: "fp")
    monkeypatch.setattr(loader, "get_clip_type_enum", lambda arch: arch)
    monkeypatch.setattr(loader.comfy.sd, "load_clip",
                        lambda ckpt_paths, clip_type, embedding_directory, model_options=None:
                        opts.append(model_options) or object(), raising=False)
    monkeypatch.setattr(loader, "_upsampler_cache", loader.LRUCache())
    import torch
    cpu = torch.device("cpu")
    loader._load_upsampler("pup.safetensors", "flux3", True, [], cpu)
    loader._load_upsampler("pup.safetensors", "flux3", True, [])          # another device: a new load
    loader._load_upsampler("pup.safetensors", "flux3", True, [], cpu)     # cached
    assert opts == [{"load_device": cpu}, {}]
