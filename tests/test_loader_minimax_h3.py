"""
Tests for MiniMax H3 support in the model Loader (model/detect.py,
config/model_map.py, loader_utils.py, nodes/generate/loader.py).

Covers:
  - Auto-Detect: the audio_patch_proj/video_patch_proj safetensors-key
    heuristic, using the same synthetic-checkpoint technique as the existing
    Flux.2/Klein tests in test_model_detect.py
  - Table consistency across config/model_map.py and model/detect.py,
    including the MODEL_TYPES list in nodes/generate/loader.py itself (a
    SEPARATE list from VIDEO_MODEL_TYPES. "ltxav" was once unselectable
    for exactly this reason, per that file's own comment)
  - construct_audio_vae(): AUDIO_VAE_KEY_REMAP dispatch (ltxav remaps and
    filters, minimax does neither), and the mutate-the-caller's-sd-in-place
    side effect load_unet_and_baked_vae relies on
"""
import sys
import os

import torch
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HAS_TORCH = isinstance(getattr(torch, "__version__", None), str)
pytestmark = pytest.mark.skipif(
    not HAS_TORCH, reason="writes real safetensors files, needs real torch/safetensors."
)


def _write_fake_checkpoint(path, keys):
    from safetensors.torch import save_file
    save_file({k: torch.zeros(1) for k in keys}, path)


class TestMiniMaxH3AutoDetect:
    """Mirrors test_model_detect.py's synthetic-checkpoint pattern."""

    def test_detected_from_audio_and_video_patch_proj_keys(self, tmp_path):
        from radiance.model.detect import detect_model_type
        path = str(tmp_path / "minimax.safetensors")
        _write_fake_checkpoint(path, [
            "audio_patch_proj.weight", "audio_patch_proj.bias",
            "video_patch_proj.weight", "video_patch_proj.bias",
            "blocks.0.attn.qkv_proj.weight",
        ])
        assert detect_model_type(path) == "minimax"

    def test_neither_marker_alone_is_sufficient(self, tmp_path):
        """Guards the AND (not OR) between the two markers. A checkpoint
        with only one of them must not be misdetected as MiniMax H3."""
        from radiance.model.detect import detect_model_type
        path = str(tmp_path / "audio_only.safetensors")
        _write_fake_checkpoint(path, ["audio_patch_proj.weight", "blocks.0.attn.qkv_proj.weight"])
        assert detect_model_type(path) != "minimax"


class TestMiniMaxH3Tables:
    """Table consistency across config/model_map.py and model/detect.py."""

    def test_listed_in_checkpoint_presets(self):
        from radiance.config.model_map import CHECKPOINT_PRESETS
        assert "MiniMax H3" in CHECKPOINT_PRESETS
        assert "MiniMax H3 (Low VRAM)" in CHECKPOINT_PRESETS

    def test_both_presets_resolve_to_minimax_model_type(self):
        from radiance.config.model_map import CHECKPOINT_PRESETS
        assert CHECKPOINT_PRESETS["MiniMax H3"]["model_type"] == "minimax"
        assert CHECKPOINT_PRESETS["MiniMax H3 (Low VRAM)"]["model_type"] == "minimax"

    def test_both_presets_keep_default_dtype(self):
        # MiniMax's own quantization (int8-convrot / nvfp4-awq) isn't a
        # generic dtype-cast target. Forcing one would fight comfy.sd's
        # own native quantization detection (comfy/sd.py's
        # detect_layer_quantization), unlike LTX's Low VRAM sibling.
        from radiance.config.model_map import CHECKPOINT_PRESETS
        for name in ("MiniMax H3", "MiniMax H3 (Low VRAM)"):
            assert CHECKPOINT_PRESETS[name]["weight_dtype"] == "default"
            assert CHECKPOINT_PRESETS[name]["clip_dtype"] == "default"

    def test_listed_in_video_preset_names(self):
        from radiance.config.model_map import VIDEO_PRESET_NAMES
        assert "MiniMax H3" in VIDEO_PRESET_NAMES
        assert "MiniMax H3 (Low VRAM)" in VIDEO_PRESET_NAMES

    def test_listed_in_video_model_types(self):
        from radiance.config.model_map import VIDEO_MODEL_TYPES
        assert "minimax" in VIDEO_MODEL_TYPES

    def test_listed_in_loader_model_types_dropdown(self):
        """Regression guard: nodes/generate/loader.py's own MODEL_TYPES list
        is separate from VIDEO_MODEL_TYPES (which only filters it). "ltxav"
        was once unselectable in Custom mode for exactly this reason, per
        that file's own comment on its MODEL_TYPES list."""
        from radiance.nodes.generate.loader import MODEL_TYPES
        assert "minimax" in MODEL_TYPES

    def test_latent_channels(self):
        from radiance.model.detect import LATENT_CHANNELS
        assert LATENT_CHANNELS["minimax"] == 24

    def test_clip_slot_order_is_single_llm_encoder(self):
        from radiance.model.detect import CLIP_SLOT_ORDER
        assert CLIP_SLOT_ORDER["minimax"] == ["llm_encoder"]

    def test_has_vram_estimates(self):
        from radiance.model.detect import _BASE_VRAM, _BASE_CLIP_VRAM
        assert "minimax" in _BASE_VRAM
        assert "minimax" in _BASE_CLIP_VRAM

    def test_model_map_lists_every_diffusion_model_variant_with_hf_urls(self):
        from radiance.config.model_map import RADIANCE_MODEL_MAP
        for fname in (
            "minimax_h3_fl2va_bf16.safetensors",
            "minimax_h3_fl2va_pruned_bf16.safetensors",
            "minimax_h3_fl2va_int8_convrot.safetensors",
            "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
            "minimax_h3_fl2va_pruned_fp8_scaled.safetensors",
        ):
            assert fname in RADIANCE_MODEL_MAP
            entry = RADIANCE_MODEL_MAP[fname]
            assert entry["type"] == "diffusion_models"
            assert entry["url"].startswith("https://huggingface.co/Comfy-Org/MiniMax-H3/")
            assert entry["url"].endswith(fname)

    def test_model_map_lists_video_and_audio_vae(self):
        from radiance.config.model_map import RADIANCE_MODEL_MAP
        for fname in ("minimax_h3_video_vae_fp16.safetensors", "minimax_h3_audio_vae_fp32.safetensors"):
            assert fname in RADIANCE_MODEL_MAP
            assert RADIANCE_MODEL_MAP[fname]["type"] == "vae"

    def test_model_map_lists_every_text_encoder_variant(self):
        from radiance.config.model_map import RADIANCE_MODEL_MAP
        for fname in (
            "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
            "qwen3vl_32b_minimax_h3_bf16.safetensors",
        ):
            assert fname in RADIANCE_MODEL_MAP
            entry = RADIANCE_MODEL_MAP[fname]
            assert entry["type"] == "text_encoders"
            assert entry["url"].startswith("https://huggingface.co/Comfy-Org/MiniMax-H3/")
            assert entry["url"].endswith(fname)


class TestConstructAudioVae:
    """AUDIO_VAE_KEY_REMAP dispatch. Uses a faithful reimplementation of
    comfy.utils.state_dict_prefix_replace since the test stub only defines
    ProgressBar (same technique as test_energy_prioritized_sampling.py's
    pack/unpack reimplementation)."""

    @pytest.fixture(autouse=True)
    def _fake_prefix_replace(self, monkeypatch):
        import comfy.utils

        def _replace(state_dict, replace_prefix, filter_keys=False):
            out = {} if filter_keys else state_dict
            for rp, new_prefix in replace_prefix.items():
                matched = [k for k in list(state_dict.keys()) if k.startswith(rp)]
                for k in matched:
                    w = state_dict.pop(k)
                    out[new_prefix + k[len(rp):]] = w
            return out

        monkeypatch.setattr(comfy.utils, "state_dict_prefix_replace", _replace, raising=False)

    def test_no_remap_entry_for_minimax(self):
        from radiance.model.detect import AUDIO_VAE_KEY_REMAP
        assert AUDIO_VAE_KEY_REMAP.get("minimax") is None

    def test_ltxav_remaps_and_filters_to_only_matched_keys(self):
        from loader_utils import construct_audio_vae
        import comfy.sd
        sd = {"audio_vae.encoder.weight": "E", "vocoder.up.weight": "V", "unrelated.weight": "U"}
        construct_audio_vae(sd, metadata=None, resolved_type="ltxav")
        called_sd = comfy.sd.VAE.call_args.kwargs["sd"]
        assert called_sd == {"autoencoder.encoder.weight": "E", "vocoder.up.weight": "V"}

    def test_ltxav_pops_matched_keys_out_of_the_callers_sd(self):
        """Regression guard for the mutation side effect load_unet_and_baked_vae
        relies on (see construct_audio_vae's docstring): the caller's own `sd`
        must lose the audio-VAE keys so its own main-model extraction doesn't
        see them mixed back in."""
        from loader_utils import construct_audio_vae
        sd = {"audio_vae.encoder.weight": "E", "unrelated.weight": "U"}
        construct_audio_vae(sd, metadata=None, resolved_type="ltxav")
        assert sd == {"unrelated.weight": "U"}

    def test_minimax_applies_no_remap(self):
        from loader_utils import construct_audio_vae
        import comfy.sd
        sd = {"pre_block.attn.zero_k_bias": "X", "other.weight": "Y"}
        construct_audio_vae(sd, metadata=None, resolved_type="minimax")
        called_sd = comfy.sd.VAE.call_args.kwargs["sd"]
        assert called_sd == {"pre_block.attn.zero_k_bias": "X", "other.weight": "Y"}

    def test_minimax_does_not_mutate_the_callers_sd(self):
        from loader_utils import construct_audio_vae
        sd = {"pre_block.attn.zero_k_bias": "X"}
        original = dict(sd)
        construct_audio_vae(sd, metadata=None, resolved_type="minimax")
        assert sd == original
