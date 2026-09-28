"""Cinematic Encoder, 3.5.0: what reaches the text encoder.

Most of these failed on the 3.5.0 tree; the rest pin what must not change.
"""
import ast
import json
import re
from pathlib import Path

import pytest

from radiance.nodes.generate import prompt as P
from radiance.nodes.generate.prompt import RadianceCinematicPromptEncoder, build_cinematic_prompt_v3

from test_prompt_encoder import FakeClip

ROOT = Path(__file__).resolve().parents[1]
C = P.CinematicDatasets
FIELDS = list(P._FIELD_TO_DATASET)


def _build(arch, preset=None, subject="A detective in the rain", **kw):
    cfg = C.PRESET_CONFIGS[preset] if preset else {}
    args = {f: cfg.get(f, "None") for f in FIELDS}
    args.update(kw)
    return build_cinematic_prompt_v3(subject, target_arch=arch, **args)


# ── 1. No menu label turns into a weight ───────────────────────────────────

@pytest.mark.parametrize("field", FIELDS)
def test_no_label_carries_a_bracket(field):
    for value in P._FIELD_TO_DATASET[field]:
        assert "(" not in P._label(value) and ")" not in P._label(value), value


@pytest.mark.parametrize("arch", ["sdxl", "sd1.5", "flux", "wan", "qwen_image"])
@pytest.mark.parametrize("preset", [p for p in C.PRESET_CONFIGS])
def test_every_preset_encodes_without_brackets(arch, preset):
    positive, _neg, _ = _build(arch, preset)
    assert "(" not in positive and ")" not in positive, positive


def test_user_brackets_are_kept():
    positive, _neg, _ = _build("sdxl", "→ Film Noir", subject="a (red:1.3) coat")
    assert "(red:1.3)" in positive


def test_unknown_label_loses_its_brackets():
    assert P._label("Some New Lens (Soft)") == "Some New Lens Soft"


# ── 5. Prose reads as English ──────────────────────────────────────────────

def test_prose_uses_an_before_vowel_sounds():
    positive, _, _ = _build("flux", lens_focal="85mm Portrait Prime", camera_type="ARRI Alexa 35")
    assert "an 85mm" in positive and "an ARRI" in positive and "a 85mm" not in positive


@pytest.mark.parametrize("framing", C.FRAMING[1:])
@pytest.mark.parametrize("arch", ["sdxl", "flux"])
def test_every_framing_opens_a_sentence(framing, arch):
    positive, _, _ = _build(arch, framing=framing)
    assert "Rule of Thirds of" not in positive and "Composition of a" not in positive
    assert positive.startswith(("A ", "An ")) and "detective in the rain" in positive
    assert " of A detective" not in positive          # article lowered after the opener


@pytest.mark.parametrize("lighting", C.LIGHTING[1:])
def test_every_lighting_reads_as_a_phrase(lighting):
    positive, _, _ = _build("flux", lighting=lighting)
    assert "Lit by " in positive and "bathed in" not in positive


def test_prose_has_no_header_labels():
    positive, _, _ = _build("flux", "→ Film Noir")
    assert "Aesthetic:" not in positive and "Cinematic technique" not in positive


@pytest.mark.parametrize("arch", ["qwen_image", "ltxav", "sdxl"])
def test_the_prompt_goes_out_as_typed(arch):
    """A period changed an edit instruction's conditioning. It only ends the
    subject's sentence when the builder adds one after it."""
    subject = "Convert this image to pop art poster style"
    assert _build(arch, subject=subject)[0] == subject
    assert _build(arch, subject=subject + "!")[0] == subject + "!"
    assert _build(arch, subject=subject, lighting=C.LIGHTING[1])[0].startswith(subject + ". ")
    assert _build(arch, subject=subject + ".", lighting=C.LIGHTING[1])[0].startswith(subject + ". ")


# ── 2. Negatives on real-CFG video models ──────────────────────────────────

@pytest.mark.parametrize("arch", ["wan", "ltxv", "ltxav", "hunyuan_video"])
def test_cfg_video_models_keep_their_negative(arch):
    _, negative, _ = _build(arch, negative_strength="Aggressive")
    assert "deformed" in negative and "mutated" in negative
    assert arch not in P._WEAK_NEG_ARCHS


@pytest.mark.parametrize("arch", ["flux", "flux2", "minimax"])
def test_distilled_models_are_still_soft(arch):
    _, negative, _ = _build(arch, negative_strength="Aggressive")
    assert "deformed" not in negative and arch in P._WEAK_NEG_ARCHS


@pytest.mark.real_torch
def test_wan_negative_field_is_not_flagged_as_ignored():
    out = RadianceCinematicPromptEncoder().execute(FakeClip(("umt5xxl",)), base_prompt="a ship")
    assert out["ui"]["weak_neg_arch"] == [False]


# ── 3. "text" only where lettering cannot be drawn ─────────────────────────

@pytest.mark.parametrize("arch", ["flux", "sd3", "qwen_image", "hidream", "wan"])
def test_text_models_are_not_told_to_avoid_text(arch):
    _, negative, _ = _build(arch, negative_strength="Standard")
    assert "text" not in [t.strip() for t in negative.split(",")]


@pytest.mark.parametrize("arch", ["sdxl", "sd1.5"])
def test_clip_models_still_avoid_text(arch):
    _, negative, _ = _build(arch, negative_strength="Soft")
    assert "text" in [t.strip() for t in negative.split(",")]


# ── 4. Presets without the frontend ────────────────────────────────────────

@pytest.mark.real_torch
def test_preset_applies_in_an_api_workflow():
    out = RadianceCinematicPromptEncoder().execute(
        FakeClip(("g", "l")), base_prompt="a detective", style_preset="→ Film Noir")
    text = out["result"][2]
    assert "ARRI Alexa 35" in text and "film noir lighting" in text and "low-angle" in text


@pytest.mark.real_torch
def test_preset_keeps_widgets_the_user_set():
    out = RadianceCinematicPromptEncoder().execute(
        FakeClip(("g", "l")), base_prompt="a detective", style_preset="→ Film Noir",
        camera_type="RED Komodo")
    text = out["result"][2]
    assert "RED Komodo" in text and "ARRI Alexa 35" not in text
    assert "Kodak Tri-X 400" in text          # hidden fields still come from the preset


# ── 6. No hidden default prompt ────────────────────────────────────────────

def test_base_prompt_default_is_empty_with_a_placeholder():
    spec = RadianceCinematicPromptEncoder.INPUT_TYPES()["optional"]["base_prompt"][1]
    assert spec["default"] == "" and spec["placeholder"]


def test_empty_prompt_is_refused_with_a_clear_message():
    with pytest.raises(ValueError, match="prompt is empty"):
        RadianceCinematicPromptEncoder().execute(FakeClip(("g", "l")))


# ── 7. Token count on LLM encoders ─────────────────────────────────────────

class _PaddedClip(FakeClip):
    """Pads to 512 with the key's own pad id, as ComfyUI's T5 / LLM tokenizers do."""

    def tokenize(self, text):
        words = [(i + 1000, 1.0) for i, _ in enumerate(str(text).split())]
        return {k: [words + [(P._PAD_IDS[k], 1.0)] * (512 - len(words))] for k in self.keys}


@pytest.mark.parametrize("key", ["umt5xxl", "mistral3_24b", "qwen3_4b", "qwen25_7b",
                                 "gemma2_2b", "pile_t5xl", "gemma3_12b"])
def test_token_count_ignores_padding(key):
    text = "a lighthouse keeper climbs the spiral stairs at dusk"
    assert P._real_token_count(_PaddedClip((key,)), text) == len(text.split())


def test_token_count_reads_t5_not_clip_on_flux():
    assert P._count_key({"l": [[(1, 1.0)]], "t5xxl": [[(1, 1.0)]]}) == "t5xxl"
    assert P._count_key({"g": [[(1, 1.0)]], "l": [[(1, 1.0)]]}) == "l"


# ── 9. One preset table ────────────────────────────────────────────────────

def test_frontend_keeps_no_copy_of_the_presets():
    js = (ROOT / "js" / "radiance_prompt.js").read_text(encoding="utf-8")
    assert "/radiance/prompt/presets" in js
    for name in C.PRESET_CONFIGS:
        assert json.dumps(name)[1:-1] not in js and name not in js, name


def test_preset_route_payload_is_the_widget_half():
    data = P.preset_widget_values()
    assert set(data) == set(C.PRESET_CONFIGS)
    for name, cfg in data.items():
        assert set(cfg) <= P._WIDGET_BACKED_PRESET_FIELDS
        for k, v in cfg.items():
            assert v in P._FIELD_TO_DATASET[k]
