"""Upscale nodes report what actually ran.

Found by the 3.5 release audit: a creative (diffusion) run with no diffusion
model was labelled "SD x4 upscaler (diffusion)"; face restore counted a crop
pasted back unchanged as restored; model_tier "auto" and mode "balanced" did
nothing; UpscaleVideo crashed whenever sharpness_boost > 0.
"""
import pytest
import torch

import radiance.nodes.upscale.upscale as up


@pytest.fixture
def no_models(monkeypatch):
    def fail(*a, **k):
        raise RuntimeError("weights not installed")
    monkeypatch.setattr(up, "_load_realesrgan", fail)
    monkeypatch.setattr(up, "_load_tier2", fail)
    monkeypatch.setattr(up, "_diffusion_upscale_infer", lambda *a, **k: None)
    monkeypatch.setattr(up, "_compute_device", lambda images: torch.device("cpu"))


def test_diffusion_fallback_is_reported_not_labelled_as_diffusion(no_models):
    img = torch.rand(1, 32, 32, 3)
    out, _, info = up.RadianceUpscaleImage().upscale_image(img, "2×", mode="creative", tile_size=128, overlap=32)
    assert out.shape == (1, 64, 64, 3)
    assert "NOT used" in info and "bicubic" in info


def test_missing_tier1_says_bicubic_is_not_ai(no_models):
    _, label = up._build_upscale_fn("tier1_fast", 2, torch.device("cpu"))
    assert "NOT an AI upscale" in label and "Real-ESRGAN unavailable" in label


def test_auto_tier_uses_content_analysis(no_models):
    img = torch.rand(1, 32, 32, 3)
    _, _, info = up.RadianceUpscaleImage().upscale_image(img, "2×", model_tier="auto", tile_size=128, overlap=32)
    assert "auto ->" in info


def test_balanced_mode_sharpens(no_models):
    img = torch.rand(1, 32, 32, 3)
    node = up.RadianceUpscaleImage()
    a, _, _ = node.upscale_image(img, "2×", mode="precise", tile_size=128, overlap=32)
    b, _, info = node.upscale_image(img, "2×", mode="balanced", tile_size=128, overlap=32)
    assert not torch.allclose(a, b)
    assert "sharpness boost: 0.25" in info


def test_video_sharpness_boost_runs(no_models):
    frames = torch.rand(6, 32, 32, 3)
    out = up.RadianceUpscaleVideo().upscale_video(frames, "2×", tile_size=128, overlap_spatial=32,
                                                  window_size=4, overlap_temporal=1,
                                                  flow_compensation=False, sharpness_boost=0.3)
    assert out[0].shape == (6, 64, 64, 3)


def test_face_crop_without_inference_is_not_counted_restored(monkeypatch):
    class Useless(torch.nn.Module):
        pass
    with pytest.raises(RuntimeError):
        up._restore_face_crop(Useless(), torch.rand(40, 40, 3), "gfpgan_v1.4", 0.7, torch.device("cpu"))

    monkeypatch.setattr(up, "_compute_device", lambda images: torch.device("cpu"))
    monkeypatch.setattr(up, "_load_face_restore_model", lambda k, d: Useless())
    monkeypatch.setattr(up, "_detect_faces", lambda frame, min_face_px, pad_frac: [(4, 4, 36, 36)])
    _, _, info = up.RadianceUpscaleFaceRestore().restore_faces(torch.rand(1, 48, 48, 3), "codeformer")
    assert "Faces restored    : 0" in info and "Crops failed" in info


def test_face_restore_without_models_says_so(monkeypatch):
    def fail(k, d):
        raise RuntimeError(f"{k} not installed")
    monkeypatch.setattr(up, "_compute_device", lambda images: torch.device("cpu"))
    monkeypatch.setattr(up, "_load_face_restore_model", fail)
    monkeypatch.setattr(up, "_detect_faces", lambda frame, min_face_px, pad_frac: [])
    _, _, info = up.RadianceUpscaleFaceRestore().restore_faces(torch.rand(1, 48, 48, 3))
    assert "NONE LOADED" in info and "codeformer: codeformer not installed" in info
