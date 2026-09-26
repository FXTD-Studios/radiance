"""Tests for the Flux.2 Dev/Klein entries added to RADIANCE_MODEL_MAP
(config/model_map.py). Covers only the file catalogue itself, mirroring
TestMiniMaxH3Tables's model_map coverage in test_loader_minimax_h3.py.
Auto-Detect and CHECKPOINT_PRESETS already have their own coverage
elsewhere (test_model_detect.py) and are untouched here.
"""


class TestFlux2DevModelMap:
    def test_dev_diffusion_models(self):
        from radiance.config.model_map import RADIANCE_MODEL_MAP
        for fname, repo in (
            ("flux2-dev.safetensors", "black-forest-labs/FLUX.2-dev"),
            ("flux2_dev_fp8mixed.safetensors", "Comfy-Org/flux2-dev"),
        ):
            assert fname in RADIANCE_MODEL_MAP
            entry = RADIANCE_MODEL_MAP[fname]
            assert entry["type"] == "diffusion_models"
            assert entry["url"].startswith(f"https://huggingface.co/{repo}/")
            assert entry["url"].endswith(fname)

    def test_dev_text_encoders(self):
        from radiance.config.model_map import RADIANCE_MODEL_MAP
        for fname in (
            "mistral_3_small_flux2_bf16.safetensors",
            "mistral_3_small_flux2_fp8.safetensors",
            "mistral_3_small_flux2_fp4_mixed.safetensors",
        ):
            assert fname in RADIANCE_MODEL_MAP
            entry = RADIANCE_MODEL_MAP[fname]
            assert entry["type"] == "text_encoders"
            assert entry["url"].startswith("https://huggingface.co/Comfy-Org/flux2-dev/")
            assert entry["url"].endswith(fname)

    def test_dev_vae(self):
        from radiance.config.model_map import RADIANCE_MODEL_MAP
        assert "flux2-vae.safetensors" in RADIANCE_MODEL_MAP
        assert RADIANCE_MODEL_MAP["flux2-vae.safetensors"]["type"] == "vae"


class TestFlux2KleinModelMap:
    def test_klein_9b_diffusion_models_on_black_forest_labs(self):
        from radiance.config.model_map import RADIANCE_MODEL_MAP
        for fname, repo in (
            ("flux-2-klein-9b.safetensors", "black-forest-labs/FLUX.2-klein-9B"),
            ("flux-2-klein-9b-fp8.safetensors", "black-forest-labs/FLUX.2-klein-9b-fp8"),
            ("flux-2-klein-base-9b.safetensors", "black-forest-labs/FLUX.2-klein-base-9B"),
            ("flux-2-klein-base-9b-fp8.safetensors", "black-forest-labs/FLUX.2-klein-base-9b-fp8"),
        ):
            assert fname in RADIANCE_MODEL_MAP
            entry = RADIANCE_MODEL_MAP[fname]
            assert entry["type"] == "diffusion_models"
            assert entry["url"].startswith(f"https://huggingface.co/{repo}/")
            assert entry["url"].endswith(fname)

    def test_klein_9b_text_encoders(self):
        from radiance.config.model_map import RADIANCE_MODEL_MAP
        for fname in (
            "qwen_3_8b.safetensors",
            "qwen_3_8b_fp8mixed.safetensors",
            "qwen_3_8b_fp4mixed.safetensors",
        ):
            assert fname in RADIANCE_MODEL_MAP
            entry = RADIANCE_MODEL_MAP[fname]
            assert entry["type"] == "text_encoders"
            assert entry["url"].startswith(
                "https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b/"
            )

    def test_klein_4b_diffusion_models_split_across_two_orgs(self):
        """Unlike the 9B tier, Klein 4B's full-precision UNET lives on
        Comfy-Org (bundled with its text encoders), only the -fp8 tier is
        on black-forest-labs. Verified directly against the real repos
        before writing this."""
        from radiance.config.model_map import RADIANCE_MODEL_MAP
        for fname, repo in (
            ("flux-2-klein-4b.safetensors", "Comfy-Org/vae-text-encorder-for-flux-klein-4b"),
            ("flux-2-klein-4b-fp8.safetensors", "black-forest-labs/FLUX.2-klein-4b-fp8"),
            ("flux-2-klein-base-4b.safetensors", "Comfy-Org/vae-text-encorder-for-flux-klein-4b"),
            ("flux-2-klein-base-4b-fp8.safetensors", "black-forest-labs/FLUX.2-klein-base-4b-fp8"),
        ):
            assert fname in RADIANCE_MODEL_MAP
            entry = RADIANCE_MODEL_MAP[fname]
            assert entry["type"] == "diffusion_models"
            assert entry["url"].startswith(f"https://huggingface.co/{repo}/")
            assert entry["url"].endswith(fname)

    def test_klein_4b_text_encoders(self):
        from radiance.config.model_map import RADIANCE_MODEL_MAP
        for fname in ("qwen_3_4b.safetensors", "qwen_3_4b_fp4_flux2.safetensors"):
            assert fname in RADIANCE_MODEL_MAP
            entry = RADIANCE_MODEL_MAP[fname]
            assert entry["type"] == "text_encoders"
            assert entry["url"].startswith(
                "https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-4b/"
            )
