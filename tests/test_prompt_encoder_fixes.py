"""Cinematic Encoder step 1 (3.5.0): no BREAK, no truncation, every modern
text encoder detected, the subject never rewritten, and no negative encode
on guidance-distilled models.

Each test failed on the pre-change tree.
"""
import json

import pytest

from radiance.nodes.generate import prompt as P
from radiance.nodes.generate.prompt import RadianceCinematicPromptEncoder, _detect_arch_from_clip

from test_prompt_encoder import FakeClip


LONG_SUBJECT = ("A weathered fisherman mends a torn red net on a wooden dock at dawn, gulls "
                "circling overhead, mist rolling off the grey harbour, his calloused hands "
                "moving with decades of practice while a small dog sleeps beside a coil of rope. ") * 2


def _run(keys, **kw):
    clip = FakeClip(keys)
    out = RadianceCinematicPromptEncoder().execute(clip, **kw)
    return clip, out


@pytest.mark.real_torch
def test_sdxl_long_prompt_keeps_camera_and_has_no_break():
    clip, out = _run(("g", "l"), base_prompt=LONG_SUBJECT, camera_type="ARRI Alexa 35",
                     lighting="Rembrandt Lighting")
    text = out["result"][2]
    assert "BREAK" not in text
    assert "ARRI Alexa 35" in text and "Rembrandt" in text
    # every word reached the encoder: FakeClip makes one token per word
    encoded = clip.encoded_tokens[0]["g"]
    real = sum(1 for ch in encoded for t in ch if t[0] != 49407)
    assert real == len(text.split())


@pytest.mark.parametrize("key,arch", [
    ("umt5xxl", "wan"), ("mistral3_24b", "flux2"), ("qwen3_4b", "qwen3_llm"),
    ("qwen3_8b", "qwen3_llm"), ("gemma2_2b", "lumina2"), ("qwen25_7b", "qwen25_llm"),
    ("pile_t5xl", "aura_flow"), ("some_future_llm", "llm"),
])
def test_modern_encoders_are_detected_and_get_prose(key, arch):
    assert _detect_arch_from_clip(FakeClip((key,)), "Auto", None) == arch
    assert P._is_prose_arch(arch)


@pytest.mark.parametrize("meta_arch", ["flux2", "flux2-klein", "z_image", "lumina2", "chroma",
                                       "qwen_image", "hidream", "cosmos", "hunyuan_video_15"])
def test_loader_arch_names_get_prose(meta_arch):
    arch = _detect_arch_from_clip(FakeClip(("l",)), "Auto", json.dumps({"arch": meta_arch}))
    assert arch == meta_arch and P._is_prose_arch(arch)


def test_clip_models_keep_the_tag_path():
    for keys, arch in ((("g", "l"), "sdxl"), (("l",), "sd1.5")):
        assert _detect_arch_from_clip(FakeClip(keys), "Auto", None) == arch
        assert not P._is_prose_arch(arch)


@pytest.mark.real_torch
def test_subject_is_never_rewritten():
    subject = "An accurate close-up of a watchmaker at work on the design of a gear, the process of generation."
    _, out = _run(("t5xxl", "l"), base_prompt=subject)
    text = out["result"][2]
    for word in ("accurate", "design", "process", "generation"):
        assert word in text
    assert "architectural composition" not in text


@pytest.mark.real_torch
def test_flux_skips_negative_encode_by_default():
    clip, out = _run(("t5xxl", "l"), base_prompt="a lighthouse at night")
    assert len(clip.encoded_tokens) == 1          # positive only
    assert out["ui"]["negative_skipped"] == [True]
    assert out["result"][3] == ""


@pytest.mark.real_torch
def test_flux_encodes_negative_when_user_typed_one_or_forced():
    clip, _ = _run(("t5xxl", "l"), base_prompt="a lighthouse", negative_prompt="fog")
    assert len(clip.encoded_tokens) == 2
    clip, _ = _run(("t5xxl", "l"), base_prompt="a lighthouse", negative_mode="Always encode")
    assert len(clip.encoded_tokens) == 2


@pytest.mark.real_torch
@pytest.mark.parametrize("keys", [("g", "l"), ("umt5xxl",), ("t5xxl", "g", "l")])
def test_cfg_models_still_encode_negative(keys):
    clip, out = _run(keys, base_prompt="a lighthouse")
    assert len(clip.encoded_tokens) == 2 and out["ui"]["negative_skipped"] == [False]


@pytest.mark.real_torch
def test_zero_conditioning_matches_positive_shape():
    import torch
    pos = [[torch.ones(1, 256, 4096), {"pooled_output": torch.ones(1, 768), "guidance": 3.5}]]
    neg = P._zero_conditioning(pos)
    assert neg[0][0].shape == pos[0][0].shape and float(neg[0][0].abs().sum()) == 0.0
    assert float(neg[0][1]["pooled_output"].abs().sum()) == 0.0
    assert neg[0][1]["guidance"] == 3.5 and pos[0][1]["pooled_output"].sum() > 0


def test_widgets_keep_their_saved_order():
    # Appended after negative_prompt so saved workflows keep their widget order
    # (vae and model_meta are sockets, never widgets). vae follows model_meta,
    # the last V1 input, so model_meta keeps its input slot index.
    names = list(RadianceCinematicPromptEncoder.INPUT_TYPES()["optional"])
    assert names.index("negative_mode") == names.index("negative_prompt") + 1
    assert names[names.index("negative_mode") + 1:] == ["model_meta", "vae", "resolution", "images"]
