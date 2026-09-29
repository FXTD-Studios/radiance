"""
Resolution's model_meta input. The front end sets model_type from the Loader
on model_meta; at execution "Manual" still follows the Loader's architecture
(an auto-detecting Loader, an API run), and any other model_type is kept.
"""
import json

import pytest

pytest.importorskip("torch")

from radiance.nodes.generate.resolution import RadianceResolution


def _generate(model_type, arch=None):
    meta = json.dumps({"arch": arch}) if arch else ""
    latent, _w, _h, channels, _info, _fr, _frames, fmt, *_ = RadianceResolution().generate(
        preset="Custom", width=1024, height=1024, orientation="As Preset",
        model_type=model_type, batch_size=1, model_meta=meta, unique_id="model-meta-test",
    )["result"]
    return tuple(latent["samples"].shape), channels, fmt


@pytest.mark.parametrize("arch, shape, fmt", [
    ("qwen_image", (1, 16, 128, 128), "qwen_image"),
    ("qwen_image21", (1, 64, 64, 64), "qwen_image21"),
    ("sdxl", (1, 4, 128, 128), "sdxl"),
])
def test_manual_follows_the_loaders_architecture(arch, shape, fmt):
    assert _generate("Manual", arch) == (shape, shape[1], fmt)


def test_a_model_type_chosen_by_hand_is_kept():
    assert _generate("SDXL / SD 1.5 / PixArt / Aura Flow (4ch)", "qwen_image")[1:] == (4, "sdxl")


def test_without_model_meta_manual_stays_manual():
    assert _generate("Manual") == _generate("Manual", "unknown_arch")
