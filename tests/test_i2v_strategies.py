"""I2V strategies write what ComfyUI's models read.

clip_vision_inject used to store a 224 px tensor under "clip_vision_input",
a key nothing in ComfyUI reads; concat_channels concatenated channels onto
the noise itself, which no model accepts. Both now write the keys
ComfyUI's WanImageToVideo writes, and refuse when they cannot work.
"""
import sys
import types

import pytest
import torch

import radiance.nodes.video.t2v as t2v


class _Weight:
    def __init__(self, in_ch):
        self.shape = (5120, in_ch, 1, 2, 2)


class _FakeModel:
    def __init__(self, in_ch, latent_ch=16):
        dm = types.SimpleNamespace(patch_embedding=types.SimpleNamespace(weight=_Weight(in_ch)))
        fmt = types.SimpleNamespace(latent_channels=latent_ch, latent_dimensions=3)
        self.model = types.SimpleNamespace(diffusion_model=dm, latent_format=fmt)


class _FakeVAE:
    def __init__(self, ch=16, sc=8):
        self.ch = ch
        self.sc = sc
        self.encoded = []

    def spacial_compression_decode(self):
        return self.sc

    def temporal_compression_decode(self):
        return 4

    def encode(self, img):
        self.encoded.append(tuple(img.shape))
        n, h, w, _ = img.shape
        t = (n - 1) // 4 + 1
        return torch.zeros(1, self.ch, t, h // self.sc, w // self.sc)


class _FakeClip:
    def tokenize(self, text):
        return text

    def encode_from_tokens(self, tokens, return_pooled=True):
        return torch.zeros(1, 4, 8), torch.zeros(1, 8)


@pytest.fixture
def captured(monkeypatch):
    calls = {}

    def fake_sample(model, noise, steps, cfg, sampler, sched, pos, neg, latent, denoise=1.0, seed=0):
        calls.update(noise=noise, pos=pos, neg=neg, denoise=denoise)
        return noise

    monkeypatch.setattr(t2v, "HAS_COMFY", True)
    monkeypatch.setattr(t2v, "_comfy_sample", fake_sample)
    monkeypatch.setattr(t2v.RadianceI2VPipeline, "_decode_preview",
                        lambda self, vae, latent, n: (torch.zeros(n, 8, 8, 3), None))
    return calls


def _run(model, vae, strategy, **kw):
    node = t2v.RadianceI2VPipeline()
    img = torch.rand(1, 64, 96, 3)
    return node.generate(model, _FakeClip(), vae, img, "a", "b", 9, 1,
                         i2v_strategy=strategy, **kw)


def test_concat_channels_writes_wan_concat_keys(captured):
    vae = _FakeVAE()
    _run(_FakeModel(36), vae, "concat_channels")
    for cond in (captured["pos"], captured["neg"]):
        d = cond[0][1]
        assert "concat_latent_image" in d and "concat_mask" in d
        mask = d["concat_mask"]
        assert float(mask[:, :, 0].max()) == 0.0 and float(mask[:, :, 1:].min()) == 1.0
    assert captured["noise"].shape[1] == 16, "noise must keep the model's latent channels"
    assert captured["denoise"] == 1.0
    assert vae.encoded[0][0] == 9, "the reference is padded to the clip length before encode"


def test_auto_picks_concat_only_when_the_model_has_the_input(captured):
    _, _, report = _run(_FakeModel(36), _FakeVAE(), "auto")
    assert "Resolved strategy: concat_channels" in report
    _, _, report = _run(_FakeModel(16), _FakeVAE(), "auto")
    assert "Resolved strategy: first_frame_lock" in report


def test_concat_refuses_a_model_without_the_input(captured):
    with pytest.raises(ValueError, match="image-concat input"):
        _run(_FakeModel(16), _FakeVAE(), "concat_channels")


def test_clip_vision_inject_needs_the_output_and_uses_comfys_key(captured):
    with pytest.raises(ValueError, match="clip_vision_output connected"):
        _run(_FakeModel(36), _FakeVAE(), "clip_vision_inject")
    cvo = object()
    _run(_FakeModel(36), _FakeVAE(), "clip_vision_inject", clip_vision_output=cvo)
    assert captured["pos"][0][1]["clip_vision_output"] is cvo
    assert captured["neg"][0][1]["clip_vision_output"] is cvo
    assert "clip_vision_input" not in captured["pos"][0][1]


def test_noise_shape_follows_the_model_not_the_ltx_default(captured):
    # No dit_config: the table default is LTX-Video (128 ch, /32). A Wan 2.2
    # TI2V 5B model (48 ch) with its /16 VAE must get a 48 ch /16 latent.
    _, _, report = _run(_FakeModel(48, latent_ch=48), _FakeVAE(ch=48, sc=16), "first_frame_lock")
    assert captured["noise"].shape[1] == 48
    assert captured["noise"].shape[-2:] == (4, 6)
    assert "Latent spec from model/VAE" in report


def test_presets_are_the_real_table_not_one_fallback():
    assert t2v._get_spec("LTX-Video (128ch)")["channels"] == 128
    assert t2v._get_spec("Wan2.2-TI2V-5B (48ch)")["spatial_compression"] == 16


def test_a_failed_reference_encode_is_an_error(captured):
    class Broken(_FakeVAE):
        def encode(self, img):
            raise RuntimeError("OOM")
    with pytest.raises(RuntimeError, match="VAE encode of the reference image failed"):
        _run(_FakeModel(16), Broken(), "first_frame_lock")


def test_hdr_strength_is_the_descriptor_weight():
    node = t2v.RadianceT2VPipeline()
    assert node._hdr_tokens("1000", "BT.2020", "PQ (ST.2084)", 0.0) == ""
    neutral = node._hdr_tokens("1000", "BT.2020", "PQ (ST.2084)", 0.5)
    assert "(" not in neutral and "1000 nits HDR" in neutral
    assert node._hdr_tokens("1000", "BT.2020", "PQ (ST.2084)", 1.0).endswith(":2.0)")


@pytest.mark.parametrize("control,values,expected", [
    ("peak_nits", ["100", "203", "400", "600", "1000", "4000", "10000"],
     ["100 nits SDR", "203 nits HDR", "400 nits HDR", "600 nits HDR",
      "1000 nits HDR", "4000 nits HDR", "10000 nits HDR"]),
    ("gamut", ["BT.2020", "P3-D65", "P3-DCI", "BT.709", "ACEScg"],
     ["rec2020", "P3 D65", "DCI-P3 theatrical", "Rec709", "aces"]),
    ("eotf", ["PQ (ST.2084)", "HLG (BT.2100)", "Linear", "sRGB / BT.1886"],
     ["HDR10 PQ", "HLG broadcast", "linear EXR", "sRGB BT1886"]),
])
def test_video_look_controls_reach_both_text_encoders(control, values, expected):
    class RecordingClip(_FakeClip):
        def tokenize(self, text):
            self.text = text
            return super().tokenize(text)

    t2v_node = t2v.RadianceT2VPipeline()
    i2v_node = t2v.RadianceI2VPipeline()
    clip = RecordingClip()
    seen = set()
    for value, descriptor in zip(values, expected):
        settings = dict(peak_nits="100", gamut="BT.709", eotf="Linear")
        settings[control] = value
        suffix = t2v_node._hdr_tokens(**settings, strength=0.5)
        assert descriptor in suffix
        assert t2v_node._encode_text(clip, "a scene, " + suffix)
        assert descriptor in clip.text
        assert i2v_node._encode_text(clip, "a scene", **settings)
        assert clip.text == "a scene, " + suffix
        seen.add(clip.text)
    assert len(seen) == len(values), "every offered control value must change conditioning text"
    i2v_node._encode_text(clip, "negative prompt")
    assert clip.text == "negative prompt", "negative prompts must not gain HDR descriptors"


def test_dit_config_does_not_edit_the_spec_table():
    before = dict(t2v._get_spec("Wan2.1 (16ch)"))
    t2v._resolve_dit_config('{"model_name": "Wan2.1 (16ch)", "channels": 99}')
    assert t2v._get_spec("Wan2.1 (16ch)") == before


def test_batch_decode_output_linear_linearises():
    import radiance.nodes.video.t2v as m

    class V:
        def decode(self, s):
            return torch.full((1, 4, 4, 3), 0.5)
    frames, n, report = m.RadianceVideoBatchDecode().decode(V(), {"samples": torch.zeros(1, 4, 4, 4)},
                                                            output_linear=True)
    assert abs(float(frames.max()) - 0.21404) < 1e-3
    assert "scene-linear" in report


def test_video_export_exr_writes_real_exrs(tmp_path):
    frames = torch.rand(2, 8, 8, 3) * 4.0
    _, n, report = t2v.RadianceVideoExport().export(frames, "exr_sequence", output_folder=str(tmp_path))
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == ["radiance_video_000000.exr", "radiance_video_000001.exr"]
    assert "Saved 2 EXR frames" in report
