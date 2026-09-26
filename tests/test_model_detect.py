"""Tests for model/detect.py architecture heuristics."""
import torch
import pytest

from radiance.model.detect import detect_model_type

HAS_TORCH = isinstance(getattr(torch, "__version__", None), str)
pytestmark = pytest.mark.skipif(
    not HAS_TORCH, reason="writes real safetensors files, needs real torch/safetensors."
)


def _flux2_keys(n_single_blocks, n_double_blocks=8):
    keys = ["double_stream_modulation_img.weight"]
    keys += [f"double_blocks.{i}.img_attn.qkv.weight" for i in range(n_double_blocks)]
    keys += [f"single_blocks.{i}.linear1.weight" for i in range(n_single_blocks)]
    return keys


def _write_fake_checkpoint(path, keys):
    from safetensors.torch import save_file
    save_file({k: torch.zeros(1) for k in keys}, path)


# ALBABIT-FIX: Flux.2 Dev and Flux.2 Klein share the double_stream_modulation_img
# key -- only single_blocks depth tells them apart (measured on real checkpoints:
# Dev=48, Klein 9B=24, Klein Base 4B=20).
def test_flux2_dev_detected_by_single_block_count(tmp_path):
    path = str(tmp_path / "dev.safetensors")
    _write_fake_checkpoint(path, _flux2_keys(n_single_blocks=48))
    assert detect_model_type(path) == "flux2"


def test_flux2_klein_9b_detected_by_single_block_count(tmp_path):
    path = str(tmp_path / "klein9b.safetensors")
    _write_fake_checkpoint(path, _flux2_keys(n_single_blocks=24))
    assert detect_model_type(path) == "flux2-klein"


def test_flux2_klein_base_4b_detected_by_single_block_count(tmp_path):
    path = str(tmp_path / "klein_base_4b.safetensors")
    _write_fake_checkpoint(path, _flux2_keys(n_single_blocks=20, n_double_blocks=5))
    assert detect_model_type(path) == "flux2-klein"


# ── 3.5: ComfyUI 0.32 families ───────────────────────────────────────────────
#
# Signatures are the keys comfy/model_detection.py branches on. Shapes matter
# where the same key names carry two families (Kandinsky 5 video vs image,
# Qwen-Image vs Mage-Flow, LongCat vs Flux).

def _write_shaped_checkpoint(path, tensors):
    from safetensors.torch import save_file
    save_file(tensors, path)


def test_hunyuan_video_15_detected_by_vision_in(tmp_path):
    path = str(tmp_path / "hv15.safetensors")
    _write_fake_checkpoint(path, [
        "txt_in.individual_token_refiner.blocks.0.norm1.weight",
        "img_in.proj.weight", "vector_in.in_layer.weight",
        "vision_in.proj.0.weight",
    ])
    assert detect_model_type(path) == "hunyuan_video_15"


def test_hunyuan_image_21_detected_by_missing_vector_in(tmp_path):
    path = str(tmp_path / "hi21.safetensors")
    _write_fake_checkpoint(path, [
        "txt_in.individual_token_refiner.blocks.0.norm1.weight",
        "img_in.proj.weight", "byt5_in.fc1.weight",
    ])
    assert detect_model_type(path) == "hunyuan_image"


def test_hunyuan_video_still_detected(tmp_path):
    """The new Hunyuan matchers must not steal the original family."""
    path = str(tmp_path / "hv.safetensors")
    _write_fake_checkpoint(path, [
        "txt_in.individual_token_refiner.blocks.0.norm1.weight",
        "img_in.proj.weight", "vector_in.in_layer.weight",
    ])
    assert detect_model_type(path) == "hunyuan_video"


def test_longcat_image_detected_by_context_width(tmp_path):
    path = str(tmp_path / "longcat.safetensors")
    _write_shaped_checkpoint(path, {
        "double_blocks.0.img_attn.qkv.weight": torch.zeros(1),
        "txt_in.weight": torch.zeros(3072, 3584),
        "img_in.weight": torch.zeros(3072, 64),
    })
    assert detect_model_type(path) == "longcat_image"


def test_flux_with_vector_in_is_still_flux(tmp_path):
    path = str(tmp_path / "flux.safetensors")
    _write_shaped_checkpoint(path, {
        "double_blocks.0.img_attn.qkv.weight": torch.zeros(1),
        "txt_in.weight": torch.zeros(3072, 4096),
        "vector_in.in_layer.weight": torch.zeros(3072, 768),
    })
    assert detect_model_type(path) == "flux"


@pytest.mark.parametrize("model_dim,expected", [
    (4096, "kandinsky5"), (1792, "kandinsky5"), (2560, "kandinsky5_image"),
])
def test_kandinsky5_video_and_image_split_on_model_dim(tmp_path, model_dim, expected):
    path = str(tmp_path / f"k5_{model_dim}.safetensors")
    _write_shaped_checkpoint(path, {
        "visual_transformer_blocks.0.cross_attention.key_norm.weight": torch.zeros(1),
        "visual_embeddings.in_layer.bias": torch.zeros(model_dim),
    })
    assert detect_model_type(path) == expected


def test_hidream_omnigen2_krea2_by_signature_key(tmp_path):
    cases = {
        "hidream": ["caption_projection.0.linear.weight"],
        "omnigen2": ["time_caption_embed.timestep_embedder.linear_1.bias"],
        "krea2": ["txtfusion.projector.weight"],
    }
    for arch, keys in cases.items():
        path = str(tmp_path / f"{arch}.safetensors")
        _write_fake_checkpoint(path, keys)
        assert detect_model_type(path) == arch, arch


def test_boogu_is_not_reported_as_omnigen2(tmp_path):
    path = str(tmp_path / "boogu.safetensors")
    _write_fake_checkpoint(path, [
        "time_caption_embed.timestep_embedder.linear_1.bias",
        "double_stream_layers.0.img_instruct_attn.processor.img_to_q.weight",
    ])
    assert detect_model_type(path) != "omnigen2"


def test_qwen_image_detected_by_txt_norm_width(tmp_path):
    path = str(tmp_path / "qwen.safetensors")
    _write_shaped_checkpoint(path, {
        "txt_norm.weight": torch.zeros(3584),
        "img_in.weight": torch.zeros(3072, 64),
        "transformer_blocks.0.img_mod.1.weight": torch.zeros(1),
    })
    assert detect_model_type(path) == "qwen_image"


def test_mage_flow_is_not_reported_as_qwen_image(tmp_path):
    path = str(tmp_path / "mage.safetensors")
    _write_shaped_checkpoint(path, {
        "txt_norm.weight": torch.zeros(2560),
        "proj_out.weight": torch.zeros(128, 2560),
    })
    assert detect_model_type(path) != "qwen_image"


def test_unknown_checkpoint_falls_back_to_comfy_detector(tmp_path, monkeypatch):
    """When no heuristic matches, ComfyUI's own detector decides, from shapes."""
    import sys
    import types
    from radiance.model import detect as D

    seen = {}

    class _Cfg:
        latent_format = types.SimpleNamespace(latent_channels=16)

    class QwenImage(_Cfg):
        pass

    def fake_model_config_from_unet(sd, prefix, metadata=None):
        seen["keys"] = sorted(sd.keys())
        seen["shape"] = tuple(sd["img_in.weight"].shape)
        seen["small"] = sd["some_norm.weight"]
        return QwenImage()

    fake = types.ModuleType("comfy.model_detection")
    fake.model_config_from_unet = fake_model_config_from_unet
    monkeypatch.setitem(sys.modules, "comfy.model_detection", fake)
    monkeypatch.setitem(sys.modules, "comfy", types.ModuleType("comfy"))
    sys.modules["comfy"].model_detection = fake

    path = str(tmp_path / "mystery.safetensors")
    _write_shaped_checkpoint(path, {
        "img_in.weight": torch.zeros(3072, 64),
        "some_norm.weight": torch.ones(8),
        "big.weight": torch.zeros(2048, 1024),
    })
    assert D.detect_model_type(path) == "qwen_image"
    assert seen["shape"] == (3072, 64)
    assert torch.is_tensor(seen["small"]), "small tensors are read for real"
    assert "big.weight" in seen["keys"]


def test_comfy_fallback_reports_unmapped_families_and_returns_none(tmp_path, monkeypatch, caplog):
    import sys
    import types
    from radiance.model import detect as D

    class Ideogram4:
        latent_format = types.SimpleNamespace(latent_channels=128)

    fake = types.ModuleType("comfy.model_detection")
    fake.model_config_from_unet = lambda sd, prefix, metadata=None: Ideogram4()
    monkeypatch.setitem(sys.modules, "comfy.model_detection", fake)
    monkeypatch.setitem(sys.modules, "comfy", types.ModuleType("comfy"))
    sys.modules["comfy"].model_detection = fake

    path = str(tmp_path / "ideo.safetensors")
    _write_fake_checkpoint(path, ["embed_image_indicator.weight"])
    with caplog.at_level("WARNING", logger="radiance.model.detect"):
        assert D.detect_model_type(path) is None
    assert any("Ideogram4" in r.getMessage() and "128" in r.getMessage() for r in caplog.records)
