"""Every Hugging Face load goes through the consent gate.

Depth Map Generator, the SD x4 and SeedVR2 upscalers and character CLIP
embedding called `from_pretrained` directly, so they downloaded (up to
2.4 GB) with `RADIANCE_ALLOW_DOWNLOADS=0`. Without consent they must now ask
for the cached copy only.
"""
from __future__ import annotations

import os
import sys
import types
from unittest import mock

import pytest

torch = pytest.importorskip("torch")
if not isinstance(getattr(torch, "__version__", None), str):
    pytest.skip("needs real torch", allow_module_level=True)

pytestmark = pytest.mark.real_torch


class _Recorder:
    def __init__(self):
        self.calls = []

    def from_pretrained(self, *args, **kwargs):
        self.calls.append(kwargs)
        raise OSError("not in the local cache")


@pytest.fixture
def no_consent(monkeypatch):
    monkeypatch.setenv("RADIANCE_ALLOW_DOWNLOADS", "0")


def test_depth_generator_stays_local_and_says_why(no_consent):
    from radiance.nodes.vfx import depth
    proc, model = _Recorder(), _Recorder()
    fake = types.SimpleNamespace(AutoImageProcessor=proc, AutoModelForDepthEstimation=model)
    with mock.patch.dict(sys.modules, {"transformers": fake}):
        depth._processor_cache.clear()
        with pytest.raises(FileNotFoundError, match="automatic downloads are off"):
            depth.download_and_load_model("Small (25M - Fast)", torch.device("cpu"))
    assert proc.calls and all(c.get("local_files_only") is True for c in proc.calls + model.calls)


def test_upscalers_stay_local(no_consent):
    from radiance.nodes.upscale import upscale
    rec = _Recorder()
    fake = types.SimpleNamespace(StableDiffusionUpscalePipeline=rec, DiffusionPipeline=rec)
    with mock.patch.dict(sys.modules, {"diffusers": fake, "seedvr2": None}):
        for loader in (upscale._load_sd_x4_pipeline, upscale._load_seedvr2_pipeline):
            try:
                loader(torch.device("cpu"))
            except Exception:  # noqa: BLE001 - each loader reports failure its own way
                pass
    assert rec.calls, "no loader reached from_pretrained"
    assert all(c.get("local_files_only") is True for c in rec.calls)


def test_character_clip_stays_local(no_consent):
    from radiance.nodes.video import character
    rec = _Recorder()
    fake = types.SimpleNamespace(CLIPModel=rec, CLIPProcessor=rec)
    with mock.patch.dict(sys.modules, {"transformers": fake}):
        with pytest.raises(OSError):
            character._clip_embed_transformers(None)
    assert rec.calls and rec.calls[0].get("local_files_only") is True


def test_consent_given_allows_download(monkeypatch):
    monkeypatch.setenv("RADIANCE_ALLOW_DOWNLOADS", "1")
    from radiance.nodes.video import character
    rec = _Recorder()
    fake = types.SimpleNamespace(CLIPModel=rec, CLIPProcessor=rec)
    with mock.patch.dict(sys.modules, {"transformers": fake}):
        with pytest.raises(OSError):
            character._clip_embed_transformers(None)
    assert rec.calls[0].get("local_files_only") is False
