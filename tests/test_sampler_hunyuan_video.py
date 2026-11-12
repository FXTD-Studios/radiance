"""HunyuanVideo 1.0 is guidance-distilled: like ComfyUI's template (BasicGuider, FluxGuidance 6),
the Sampler gives it no CFG and writes flux_guidance into the conditioning."""
import torch

from _sampler_harness import SampleCustomRecorder, run_sampler
from radiance.sampler_utils import CFG_GUIDED_MODELS, GUIDANCE_EMBED_MODELS, get_model_defaults


def test_hunyuan_video_defaults_to_its_embedded_guidance():
    defaults = get_model_defaults("hunyuan_video")
    assert (defaults["cfg"], defaults["guidance"]) == (1.0, 6.0)
    assert "hunyuan_video" in GUIDANCE_EMBED_MODELS and "hunyuan_video" not in CFG_GUIDED_MODELS


def test_hunyuan_video_gets_flux_guidance_in_its_conditioning():
    rec = SampleCustomRecorder()
    run_sampler(recorder=rec, model_type="hunyuan_video", cfg=1.0, flux_guidance=6.0,
                latent={"samples": torch.zeros(1, 16, 3, 16, 16)})
    assert rec.calls
    for call in rec.calls:
        assert call["cfg"] == 1.0
        assert call["positive"][0][1]["guidance"] == 6.0
