"""Tests for the LTX-2.3 entries added to RADIANCE_MODEL_MAP
(config/model_map.py) to close a gap found while cross-checking Flux.2:
several filenames already referenced by the Loader's LTX-2.3 preset hints
had no matching catalogue entry. Mirrors test_model_map_flux2.py's
structure.
"""
import re


class TestLtx23DiffusionModels:
    def test_distilled_no_suffix_still_real_but_superseded(self):
        from radiance.config.model_map import RADIANCE_MODEL_MAP
        fname = "ltx-2.3-22b-distilled.safetensors"
        assert fname in RADIANCE_MODEL_MAP
        entry = RADIANCE_MODEL_MAP[fname]
        assert entry["type"] == "diffusion_models"
        assert entry["url"].startswith("https://huggingface.co/Lightricks/LTX-2.3/")

    def test_dev_fp8_on_its_own_repo(self):
        """fp8 lives on Lightricks/LTX-2.3-fp8, a separate repo from the
        main Lightricks/LTX-2.3 one, not a subfolder of it."""
        from radiance.config.model_map import RADIANCE_MODEL_MAP
        fname = "ltx-2.3-22b-dev-fp8.safetensors"
        assert fname in RADIANCE_MODEL_MAP
        entry = RADIANCE_MODEL_MAP[fname]
        assert entry["type"] == "diffusion_models"
        assert re.fullmatch(rf"https://huggingface\.co/Lightricks/LTX-2\.3-fp8/resolve/[0-9a-f]{{40}}/{re.escape(fname)}",
                            entry["url"])


class TestLtx23Upscaler:
    def test_x2_1_1_catalogued_as_latent_upscale_model(self):
        from radiance.config.model_map import RADIANCE_MODEL_MAP
        fname = "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
        assert fname in RADIANCE_MODEL_MAP
        assert RADIANCE_MODEL_MAP[fname]["type"] == "latent_upscale_models"

    def test_x2_1_0_deliberately_not_catalogued(self):
        """Removed from the Lightricks repo (confirmed by Albabit); its old
        URL 404s now. Still a valid unet_hints fallback for local files, but
        must never be offered for auto-download."""
        from radiance.config.model_map import RADIANCE_MODEL_MAP
        assert "ltx-2.3-spatial-upscaler-x2-1.0.safetensors" not in RADIANCE_MODEL_MAP


class TestLtx23TextEncoders:
    def test_gemma_variants_on_comfy_org_ltx2(self):
        from radiance.config.model_map import RADIANCE_MODEL_MAP
        for fname in ("gemma_3_12B_it.safetensors", "gemma_3_12B_it_fp4_mixed.safetensors"):
            assert fname in RADIANCE_MODEL_MAP
            entry = RADIANCE_MODEL_MAP[fname]
            assert entry["type"] == "text_encoders"
            assert re.match(r"https://huggingface\.co/Comfy-Org/ltx-2/resolve/[0-9a-f]{40}/split_files/text_encoders/",
                            entry["url"])

    def test_text_projection_on_kijai_repo(self):
        from radiance.config.model_map import RADIANCE_MODEL_MAP
        fname = "ltx-2.3_text_projection_bf16.safetensors"
        assert fname in RADIANCE_MODEL_MAP
        entry = RADIANCE_MODEL_MAP[fname]
        assert entry["type"] == "text_encoders"
        assert entry["url"].startswith("https://huggingface.co/Kijai/LTX2.3_comfy/")


class TestLtx23VaeOnKijaiRepo:
    def test_video_and_audio_vae(self):
        """Community repackaging (Kijai/LTX2.3_comfy), not an official
        Lightricks/Comfy-Org repo. Confirmed acceptable as a catalogue
        source by Albabit."""
        from radiance.config.model_map import RADIANCE_MODEL_MAP
        for fname in ("LTX23_video_vae_bf16.safetensors", "LTX23_audio_vae_bf16.safetensors"):
            assert fname in RADIANCE_MODEL_MAP
            entry = RADIANCE_MODEL_MAP[fname]
            assert entry["type"] == "vae"
            assert re.match(r"https://huggingface\.co/Kijai/LTX2\.3_comfy/resolve/[0-9a-f]{40}/vae/", entry["url"])
