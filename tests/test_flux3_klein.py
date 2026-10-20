"""
FLUX 3 Image Klein (Black Forest Labs, 15 October 2026): Loader, Sampler,
Resolution and Prompt.

A 5.6B DiT with its own block layout (per-mode stacks, no double_blocks),
the Flux.2 VAE (128ch, 16x) and a Qwen3-VL-4B text encoder that ComfyUI
loads only under CLIPType.FLUX3. Base and step-distilled share the
architecture; the distilled one adds a flow-map head (ref_time_in), which
is how the Sampler tells them apart without a file name.

tests/fixtures/flux3_klein_keys.json holds the state-dict key names of
ComfyUI's own Flux3 module (comfy/ldm/flux3/model.py), base and distilled.
"""
import json
import os
import sys
import types

import pytest
import torch
from comfy_api.latest import io

from radiance.model import detect as D

HAS_TORCH = isinstance(getattr(torch, "__version__", None), str)
KEYS = json.load(open(os.path.join(os.path.dirname(__file__), "fixtures", "flux3_klein_keys.json")))
BASE_FILE = "flux3_image_klein_int8_convrot.safetensors"
SD_FILE = "flux3_image_klein_sd_int8_convrot.safetensors"


def _write(path, keys):
    from safetensors.torch import save_file
    save_file({k: torch.zeros(shape) for k, shape in keys.items()}, path)


# ── Loader: detection and tables ─────────────────────────────────────────────

@pytest.mark.skipif(not HAS_TORCH, reason="writes real safetensors files")
class TestDetection:

    @pytest.mark.parametrize("variant", ["base", "distilled"])
    def test_both_variants_detect_as_flux3(self, tmp_path, variant):
        path = str(tmp_path / f"{variant}.safetensors")
        _write(path, KEYS[variant])
        assert D.detect_model_type(path) == "flux3"

    def test_flux2_klein_is_still_flux2_klein(self, tmp_path):
        # Flux.2 Klein keys (double_stream_modulation_img, 20 single blocks).
        keys = {"double_stream_modulation_img.lin.weight": [4, 4], "img_in.weight": [4, 128],
                "double_blocks.0.img_attn.norm.key_norm.scale": [4]}
        keys.update({f"single_blocks.{i}.linear1.weight": [4, 4] for i in range(20)})
        path = str(tmp_path / "klein2.safetensors")
        _write(path, keys)
        assert D.detect_model_type(path) == "flux2-klein"

    def test_comfy_fallback_maps_the_flux3_config_class(self):
        assert D._COMFY_CONFIG_TO_ARCH["Flux3"] == "flux3"


class TestTables:

    def test_flux2_vae_figures(self):
        assert D.LATENT_CHANNELS["flux3"] == 128
        assert D.VAE_SPATIAL_FACTOR["flux3"] == 16
        assert D._FORMAT_MAP["flux3"] == D._FORMAT_MAP["flux2-klein"]

    def test_one_text_encoder_slot(self):
        assert D.CLIP_SLOT_ORDER["flux3"] == ["llm_encoder"]

    def test_vae_config_is_not_flux1s(self):
        # "flux3" contains "flux": the substring fallback used to land on the 16ch entry.
        from radiance.config.model_map import resolve_model_vae_config
        cfg = resolve_model_vae_config("flux3")
        assert cfg["latent_channels"] == 128 and cfg["text_embed_hidden"] == 20480

    def test_vram_tables_have_their_own_entry(self):
        assert D._BASE_VRAM["flux3"] == 11.2 and D._BASE_CLIP_VRAM["flux3"] == 8.9
        assert D.estimate_vram_usage("flux3", "default") == pytest.approx(20.1)

    def test_presets(self):
        from radiance.config.model_map import CHECKPOINT_PRESETS
        from radiance.nodes.generate.loader import MODEL_TYPES, RadianceUnifiedLoader
        presets = RadianceUnifiedLoader.INPUT_TYPES()["required"]["preset"][0]
        for name in ("Flux.3 Klein", "Flux.3 Klein (Low VRAM)"):
            assert CHECKPOINT_PRESETS[name]["model_type"] == "flux3"
            assert name in presets
        assert "flux3" in MODEL_TYPES
        names = list(CHECKPOINT_PRESETS)
        assert names.index("Flux.3 Klein") == names.index("Flux.2 (Low VRAM)") + 1


class TestClipType:

    def _clip_type(self, monkeypatch, **members):
        # The comfy.sd detect.py holds: other tests swap sys.modules["comfy"].
        fake = types.SimpleNamespace(STABLE_DIFFUSION="SD", FLUX="FLUX", SD3="SD3", **members)
        monkeypatch.setattr(D.comfy.sd, "CLIPType", fake, raising=False)
        return D.get_clip_type_enum("flux3")

    def test_uses_clip_type_flux3(self, monkeypatch):
        assert self._clip_type(monkeypatch, FLUX3="FLUX3", FLUX2="FLUX2") == "FLUX3"

    def test_an_older_comfyui_stops_instead_of_loading_the_wrong_encoder(self, monkeypatch):
        with pytest.raises(RuntimeError, match="Update ComfyUI"):
            self._clip_type(monkeypatch, FLUX2="FLUX2")


# ── Sampler ─────────────────────────────────────────────────────────────────

class Flux3:
    """Stands in for comfy.ldm.flux3.model.Flux3 (the class name is what is read)."""

    def __init__(self, flow_map):
        self.flow_map = flow_map


def _model(flow_map):
    config = type("Flux3", (), {})()          # comfy.supported_models.Flux3
    dm = Flux3(flow_map)
    return types.SimpleNamespace(
        model=types.SimpleNamespace(model_config=config),
        get_model_object=lambda name: dm if name == "diffusion_model" else None)


def _auto(model, meta="", cfg=1.0, steps=20):
    from radiance.nodes.generate.sampler import RadianceSamplerPro
    detected, kwargs, _ = RadianceSamplerPro()._configure_model_and_defaults(
        model, "auto", "Auto", model_meta=meta, cfg=cfg, flux_guidance=3.5, steps=steps,
        sampler="euler", scheduler="normal", scheduler_mode="Auto (Match Steps)")
    return detected, kwargs


class TestSampler:

    def test_config_class_detects_flux3(self):
        from radiance.sampler_utils import detect_model_type
        assert detect_model_type(_model(False)) == "flux3"

    def test_flux3_is_cfg_guided_not_guidance_embedded(self):
        from radiance.sampler_utils import CFG_GUIDED_MODELS, GUIDANCE_EMBED_MODELS, MODEL_TYPES
        assert "flux3" in MODEL_TYPES and "flux3" in CFG_GUIDED_MODELS
        assert "flux3" not in GUIDANCE_EMBED_MODELS

    def test_base_runs_the_templates_cfg_and_steps_without_model_meta(self):
        detected, kw = _auto(_model(flow_map=False))
        assert detected == "flux3"
        assert (kw["cfg"], kw["steps"], kw["sampler"], kw["scheduler"]) == (3.5, 50, "euler", "simple")
        assert kw["flux_guidance"] == 0.0

    def test_distilled_runs_16_steps_at_cfg_1_without_model_meta(self):
        _, kw = _auto(_model(flow_map=True))
        assert (kw["cfg"], kw["steps"], kw["scheduler"]) == (1.0, 16, "simple")

    def test_the_weights_win_over_a_misleading_file_name(self):
        meta = json.dumps({"arch": "flux3", "unet_file": SD_FILE})
        _, kw = _auto(_model(flow_map=False), meta=meta)
        assert (kw["cfg"], kw["steps"]) == (3.5, 50)

    @pytest.mark.parametrize("unet, expected", [(BASE_FILE, (3.5, 50)), (SD_FILE, (1.0, 16)),
                                                ("flux3_image_klein_sd_bf16.safetensors", (1.0, 16))])
    def test_file_name_decides_when_the_model_does_not_say(self, unet, expected):
        meta = json.dumps({"arch": "flux3", "unet_file": unet})
        _, kw = _auto(None, meta=meta)
        assert (kw["cfg"], kw["steps"]) == expected

    def test_a_value_the_user_set_is_kept(self):
        _, kw = _auto(_model(flow_map=False), cfg=4.5, steps=30)
        assert (kw["cfg"], kw["steps"]) == (4.5, 30)

    def test_flux2_klein_rule_does_not_touch_flux3(self):
        from radiance.sampler_utils import refine_distillation_from_meta
        assert refine_distillation_from_meta("flux3", BASE_FILE) is None
        assert refine_distillation_from_meta("flux2-klein", "flux-2-klein-4b.safetensors") == {"guidance": 1.0, "steps": 4}


# ── Resolution ──────────────────────────────────────────────────────────────

class TestResolution:

    def test_model_type(self):
        from radiance.nodes.generate import resolution as R
        mt = "Flux.3 Klein (128ch)"
        assert mt in R.MODEL_TYPES and mt not in R.VIDEO_MODEL_TYPES
        assert (R.LATENT_CHANNELS[mt], R.SPATIAL_SCALE[mt], R.LATENT_FORMAT_MAP[mt]) == (128, 16, "flux3")
        assert R.MODEL_TYPE_BY_ARCH["flux3"] == mt

    def test_vram_estimate_reads_flux3_not_flux2_dev(self):
        from radiance.nodes.generate import resolution as R
        flux3 = R._estimate_vram(1024, 1024, 128, 1, "flux3", 16)
        assert flux3 == pytest.approx(11.2 + 8.9, abs=0.01)


# ── Prompt ──────────────────────────────────────────────────────────────────

class Flux3Tokenizer:
    pass


class MageFlowTokenizer:
    pass


class _Clip:
    def __init__(self, tokenizer_cls=Flux3Tokenizer):
        self.tokenizer = tokenizer_cls()
        self.texts = []

    def tokenize(self, text, **kwargs):
        if text != "test":              # the encoder's own arch probe
            self.texts.append(text)
        return {"qwen3vl_4b": [[(1, 1.0)] * 5]}

    def encode_from_tokens_scheduled(self, tokens):
        return [["cond", {}]]


class _Vae:
    def __init__(self):
        self.encoded = []

    def encode(self, pixels):
        self.encoded.append(tuple(pixels.shape))
        b, h, w, _ = pixels.shape
        return torch.ones(b, 128, h // 16, w // 16)


@pytest.fixture(autouse=True)
def reference_latent(monkeypatch):
    """ComfyUI's ReferenceLatent: appends the latent to reference_latents."""
    from radiance.nodes.generate import prompt

    class ReferenceLatent:
        @classmethod
        def execute(cls, conditioning, latent=None):
            return io.NodeOutput([[c[0], {**c[1], "reference_latents": c[1].get("reference_latents", []) + [latent["samples"]]}]
                                  for c in conditioning])
    monkeypatch.setattr(prompt.nodes_edit_model, "ReferenceLatent", ReferenceLatent, raising=False)


@pytest.fixture
def resize(monkeypatch):
    from radiance.nodes.generate import prompt
    calls = []

    def _upscale(samples, width, height, method, crop):
        calls.append((width, height, method, crop))
        return torch.zeros(samples.shape[0], samples.shape[1], height, width)
    monkeypatch.setattr(prompt.comfy.utils, "common_upscale", _upscale, raising=False)
    return calls


def _encode(clip=None, **kwargs):
    from radiance.nodes.generate import prompt
    kwargs.setdefault("base_prompt", "In <ref_image_0>, make the dress red")
    return prompt.RadianceCinematicPromptEncoder.execute(clip or _Clip(), **kwargs)["result"]


class TestPrompt:

    def test_the_text_encoder_resolves_to_flux3(self):
        from radiance.nodes.generate import prompt
        assert prompt._detect_arch_from_clip(_Clip(), "Auto") == "flux3"
        assert prompt._detect_arch_from_clip(_Clip(MageFlowTokenizer), "Auto") != "flux3"

    def test_prose_and_both_prompts_encoded(self):
        clip = _Clip()
        pos, neg, text, _neg_text, arch, *_ = _encode(clip, base_prompt="a red fox in snow")
        assert arch == "flux3"
        assert len(clip.texts) == 2      # the base model runs real cfg: the negative is encoded

    def test_reference_images_join_both_prompts_and_size_the_latent(self, resize):
        vae = _Vae()
        images = {"image_1": torch.zeros(1, 768, 1536, 4), "image_2": torch.zeros(1, 512, 512, 3)}
        pos, neg, text, *_, latent = _encode(vae=vae, images=images, resolution=1024)
        assert [r.shape for r in pos[0][1]["reference_latents"]] == [r.shape for r in neg[0][1]["reference_latents"]]
        assert len(pos[0][1]["reference_latents"]) == 2
        assert resize[0] == (1448, 724, "lanczos", "disabled")      # 1 MP, aspect kept
        assert resize[1] == (1024, 1024, "lanczos", "disabled")
        assert all(s[-1] == 3 for s in vae.encoded)                 # RGB to the VAE
        assert latent["samples"].shape == (1, 128, 724 // 16, 1448 // 16)
        assert latent["samples"].abs().sum() == 0
        assert "<ref_image_0>" in text

    def test_resolution_0_keeps_native_size_up_to_16_mp(self, resize):
        _encode(vae=_Vae(), images={"image_1": torch.zeros(1, 640, 960, 3)}, resolution=0)
        assert resize == []                                         # native size, no resample
        _encode(vae=_Vae(), images={"image_1": torch.zeros(1, 4800, 4800, 3)}, resolution=0)
        assert resize[-1][0] * resize[-1][1] <= 16 * 1024 * 1024 * 1.001

    def test_more_than_10_references_is_an_error(self, resize):
        images = {f"image_{i}": torch.zeros(1, 32, 32, 3) for i in range(1, 12)}
        with pytest.raises(ValueError, match="reads 10 reference images"):
            _encode(vae=_Vae(), images=images)

    def test_references_need_the_vae(self):
        with pytest.raises(ValueError, match="Flux.3 Klein reads reference images as VAE latents"):
            _encode(images={"image_1": torch.zeros(1, 32, 32, 3)})

    def test_a_layout_table_reaches_the_encoder_verbatim(self):
        from radiance.nodes.generate import prompt
        table = json.dumps([{"id": "obj_1", "bbox": [0.1, 0.2, 0.8, 0.9], "desc": "a red silk dress, flowing"}],
                           ensure_ascii=False)
        clip = _Clip()
        _pos, _neg, text, *_ = _encode(clip, base_prompt=f"A woman in a ballroom wearing <obj_1>. {table}")
        assert text.endswith(" " + table)
        assert clip.texts[0] == text
        assert prompt._split_flux3_layout("no table [here]") == ("no table [here]", "")


class TestLayoutInput:
    """The layout input the Viewer's FLUX 3 Layout Boxes writes (4.0.0)."""

    ROWS = [{"id": "obj_1", "bbox": [0, 0, 400, 1000], "desc": "a stormy sky"},
            {"id": "obj_2", "bbox": [450, 300, 980, 700], "desc": "a lighthouse"}]

    def test_the_table_is_appended_after_the_prompt(self):
        clip = _Clip()
        _pos, _neg, text, *_ = _encode(clip, base_prompt="<obj_2> under <obj_1>",
                                       layout=json.dumps(self.ROWS, indent=2))
        assert text.endswith(" " + json.dumps(self.ROWS, ensure_ascii=False))
        assert clip.texts[0] == text

    def test_empty_or_blank_changes_nothing(self):
        a = _encode(base_prompt="a red fox")[2]
        assert _encode(base_prompt="a red fox", layout="  ")[2] == a
        assert _encode(base_prompt="a red fox", layout="[]")[2] == a

    def test_only_flux3_reads_it(self, monkeypatch):
        from radiance.nodes.generate import prompt
        monkeypatch.setattr(prompt, "_detect_arch_from_clip", lambda *a, **k: "flux2-klein")
        with pytest.raises(ValueError, match="FLUX 3 Image Klein only"):
            _encode(base_prompt="a red fox", layout=json.dumps(self.ROWS))

    def test_two_tables_are_refused(self):
        table = json.dumps(self.ROWS[:1])
        with pytest.raises(ValueError, match="Keep one"):
            _encode(base_prompt=f"a sky {table}", layout=json.dumps(self.ROWS))

    @pytest.mark.parametrize("bad, match", [
        ("{not json", "not JSON"),
        ('{"id": "obj_1"}', "JSON list"),
        ('[{"id": "obj_1", "bbox": [0.1, 0.2, 0.8]}]', "box 1 needs a bbox"),
        ('[{"id": "obj_1", "bbox": [0, 0, 1200, 500]}]', "0-1000 grid"),
        ('[{"id": "obj_1", "bbox": [600, 0, 100, 500]}]', "0-1000 grid"),
    ])
    def test_a_broken_table_says_what_is_wrong(self, bad, match):
        with pytest.raises(ValueError, match=match):
            _encode(base_prompt="a red fox", layout=bad)
