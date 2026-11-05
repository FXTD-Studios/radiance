"""
The Loader took over the Video Loader (4.0): one node lists every preset and
model type and loads the audio VAE, the latent upscale model and Wan 2.2's
low-noise expert, while the old RadianceVideoLoader key still opens saved
graphs with its own widget and output order.
"""
import json
import re
import types
from pathlib import Path

import pytest

from radiance.config.model_map import CHECKPOINT_PRESETS
from radiance.nodes.generate import loader
from radiance.nodes.generate.loader import RadianceUnifiedLoader, RadianceVideoLoader

JS = (Path(__file__).resolve().parents[1] / "js" / "radiance_loader.js").read_text(encoding="utf-8")

# The Video Loader's widgets and outputs as graphs saved before 4.0 hold them.
OLD_VIDEO_WIDGETS = [
    "preset", "unet_name", "weight_dtype", "model_type", "vae_name", "audio_vae_name",
    "upscale_model_name", "clip_l", "clip_g", "t5xxl", "llm_encoder", "text_projection",
    "clip_dtype", "offload_mode", "check_vram", "use_cache", "lora_on_error", "auto_download",
]
OLD_VIDEO_OUTPUTS = ("model", "model_low_noise", "clip", "vae", "audio_vae", "lora_stack",
                     "upscale_model", "model_meta")


def _widgets(cls):
    spec = cls.INPUT_TYPES()
    return [name for section in ("required", "optional") for name, (kind, *_) in spec[section].items()
            if kind != "LORA_STACK"]


class TestOneLoader:

    def test_it_lists_every_preset_and_model_type(self):
        required = RadianceUnifiedLoader.INPUT_TYPES()["required"]
        assert required["preset"][0] == list(CHECKPOINT_PRESETS)
        assert required["model_type"][0] == loader.MODEL_TYPES
        assert {"LTX Video 2.5 Dev", "Wan 2.2", "Flux.2", "SDXL"} <= set(required["preset"][0])

    def test_the_video_inputs_sit_under_vae_name(self):
        widgets = _widgets(RadianceUnifiedLoader)
        assert widgets[4:7] == ["vae_name", "audio_vae_name", "upscale_model_name"]
        assert widgets[-2:] == ["model_shift", "upsampler"]

    def test_the_video_outputs_come_after_the_image_ones(self):
        """Saved graphs keep their outputs in the saved order: appending is what keeps their links."""
        assert RadianceUnifiedLoader.RETURN_NAMES == (
            "model", "clip", "vae", "lora_stack", "model_meta", "upsampler_clip",
            "model_low_noise", "audio_vae", "upscale_model")
        assert len(RadianceUnifiedLoader.RETURN_TYPES) == len(RadianceUnifiedLoader.RETURN_NAMES)

    def test_saved_loaders_are_restored_by_the_order_they_were_saved_in(self):
        body = re.search(r"const LOADER_SAVED_WIDGET_ORDER = \[(.*?)\];", JS, re.S).group(1)
        saved = re.findall(r'"([^"]+)"', body)
        assert saved == [n for n in _widgets(RadianceUnifiedLoader)
                         if n not in ("audio_vae_name", "upscale_model_name")]


class TestOldVideoLoaderKey:

    def test_it_keeps_its_widget_order(self):
        assert _widgets(RadianceVideoLoader)[:len(OLD_VIDEO_WIDGETS)] == OLD_VIDEO_WIDGETS

    def test_it_keeps_its_output_order(self):
        assert RadianceVideoLoader.RETURN_NAMES == OLD_VIDEO_OUTPUTS
        assert len(RadianceVideoLoader.RETURN_TYPES) == len(OLD_VIDEO_OUTPUTS)

    def test_it_returns_the_loaders_outputs_in_that_order(self, monkeypatch):
        names = RadianceUnifiedLoader.RETURN_NAMES
        monkeypatch.setattr(RadianceUnifiedLoader, "load_radiance_stack", lambda self, *a, **k: names)
        assert RadianceVideoLoader().load_radiance_stack("LTX Video 2.5 Dev", "u", "default", "Auto-Detect", "v") \
            == OLD_VIDEO_OUTPUTS


class TestLoad:

    @pytest.fixture
    def run(self, monkeypatch):
        """Runs the Loader with every file load stubbed out; returns its outputs
        and the audio VAE / upscale model files it was asked to load."""
        asked = {}

        class _WanModel(str):
            # Wan's sampling settings: a shift, ComfyUI's default time scale (multiplier 1000).
            model = types.SimpleNamespace(model_config=types.SimpleNamespace(sampling_settings={"shift": 8.0}))

        def unet(path, name, *a):
            return (_WanModel(f"unet:{name}"), "vae", "baked audio", 0.0, False, 0.0, False)

        stubs = {
            "_ensure_model_exists": lambda name, *a: name,
            "resolve_architecture": lambda *a: ("wan", "wan", "Wan21"),
            "setup_offload_mode": lambda *a: None,
            "estimate_vram_for_load": lambda *a, **k: (0.0, 0.0, 0.0),
            "load_unet_and_baked_vae": unet,
            "load_clip_stack": lambda *a: ("clip", [], 0.0, False),
            "load_standalone_vae": lambda *a: ("vae", 0.0, False),
            "apply_lora_stack": lambda model, clip, *a: (model, clip, [], None),
            "print_premium_loader_hud": lambda **k: None,
            "_load_audio_vae": lambda name, *a: asked.setdefault("audio", name) and "audio vae",
            "_load_upscale_model": lambda name, *a: asked.setdefault("upscale", name) and "upscaler",
        }
        for name, stub in stubs.items():
            monkeypatch.setattr(loader, name, stub)

        def go(unet_name, **kwargs):
            out = RadianceUnifiedLoader().load_radiance_stack(
                "Custom", unet_name, "default", "Auto-Detect", "vae.safetensors",
                check_vram="Off", use_cache="Off", **kwargs)
            return out, json.loads(out[4]), asked
        return go

    def test_a_video_model_fills_the_video_outputs(self, run):
        out, meta, asked = run("wan2.2_t2v_low_noise_14B.safetensors",
                               audio_vae_name="audio.safetensors", upscale_model_name="up.safetensors")
        assert out[0] == "unet:wan2.2_t2v_high_noise_14B.safetensors", "model is the high-noise expert"
        assert out[6] == "unet:wan2.2_t2v_low_noise_14B.safetensors"
        assert out[7] == "audio vae" and out[8] == "upscaler"
        assert asked == {"audio": "audio.safetensors", "upscale": "up.safetensors"}
        assert meta["wan_moe_companion"] == "wan2.2_t2v_high_noise_14B.safetensors"
        assert meta["audio_vae"] == "audio.safetensors"
        assert meta["upscale_model"] == "up.safetensors"

    def test_model_shift_is_ignored_for_a_model_not_sampled_like_qwen_image(self, run, monkeypatch):
        """ModelSamplingAuraFlow sets the time scale to 1: on Wan it would break the
        schedule, and only the high-noise expert would get it."""
        patched = []

        class _AuraFlow:
            def patch_aura(self, model, shift):
                patched.append(model)
                return ("shifted",)

        monkeypatch.setattr(loader.nodes_model_advanced, "ModelSamplingAuraFlow", _AuraFlow, raising=False)
        out, meta, _ = run("wan2.2_t2v_low_noise_14B.safetensors", model_shift=3.0)
        assert patched == []
        assert out[0] == "unet:wan2.2_t2v_high_noise_14B.safetensors"
        assert out[6] == "unet:wan2.2_t2v_low_noise_14B.safetensors"
        assert meta["model_shift"] == 0.0

    def test_a_baked_audio_vae_comes_from_the_checkpoint(self, run):
        out, meta, asked = run("ltx.safetensors", audio_vae_name="Baked Audio VAE (from UNET)")
        assert out[7] == "baked audio"
        assert "audio" not in asked

    def test_an_image_model_leaves_the_video_outputs_empty(self, run):
        out, meta, asked = run("flux2-dev.safetensors")
        assert out[:3] == ("unet:flux2-dev.safetensors", "clip", "vae")
        assert out[6] is None and out[8] is None
        assert asked == {}
        assert meta["wan_moe_companion"] is None and meta["upscale_model"] is None


class TestUpscaleModel:
    """The latent upscale model goes through ComfyUI's own Load Latent Upscale Model."""

    @pytest.fixture
    def native(self, monkeypatch):
        calls = []

        class _Native:
            result = "patcher"

            @classmethod
            def execute(cls, name):
                calls.append(name)
                return types.SimpleNamespace(args=[cls.result])

        monkeypatch.setattr(loader.folder_paths, "get_full_path", lambda kind, name: f"/models/{name}")
        monkeypatch.setattr(loader, "_file_fingerprint", lambda path: "fp")
        monkeypatch.setattr(loader, "LatentUpscaleModelLoader", _Native)
        return _Native, calls

    def test_loads_through_the_native_node(self, native):
        cls, calls = native
        info = []
        assert loader._load_upscale_model("up.safetensors", False, "|", info) == "patcher"
        assert calls == ["up.safetensors"]
        assert info[-1].startswith("UPSCALE MODEL: up.safetensors (")

    def test_an_unknown_architecture_loads_nothing(self, native):
        cls, _ = native
        cls.result = None
        info = []
        assert loader._load_upscale_model("odd.safetensors", False, "|", info) is None
        assert info == ["UPSCALE MODEL: unrecognized architecture (odd.safetensors)"]

    def test_an_older_comfyui_without_the_node_loads_nothing(self, native, monkeypatch):
        monkeypatch.setattr(loader, "LatentUpscaleModelLoader", None)
        info = []
        assert loader._load_upscale_model("up.safetensors", False, "|", info) is None
        assert info == ["UPSCALE MODEL: 'up.safetensors' needs a newer ComfyUI"]

    def test_a_missing_file_loads_nothing(self, native, monkeypatch):
        monkeypatch.setattr(loader.folder_paths, "get_full_path", lambda kind, name: None)
        info = []
        assert loader._load_upscale_model("gone.safetensors", False, "|", info) is None
        assert info == ["UPSCALE MODEL: 'gone.safetensors' not found"]


def test_the_vae_is_read_with_its_metadata(monkeypatch):
    """As ComfyUI's Load VAE: LTX 2.3's video VAE needs it to pick its config."""
    seen = {}
    monkeypatch.setattr(loader.comfy.utils, "load_torch_file",
                        lambda path, **k: seen.update(k) or ({}, {"config": "{}"}), raising=False)
    assert RadianceUnifiedLoader._load_vae_sd_metadata("vae.safetensors") == ({}, {"config": "{}"})
    assert seen == {"return_metadata": True}
