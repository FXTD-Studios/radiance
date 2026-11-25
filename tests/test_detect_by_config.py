"""sampler_utils.detect_by_config: the config class name decides the model type.

ComfyUI's config classes carry suffixes (WAN21_T2V, HunyuanVideoI2V), so the
table is matched by substring. It was matched in insertion order, and "Flux"
comes before "Flux2", so every Flux.2 model without a connected model_meta was
detected as Flux.1 and picked up Flux.1 defaults (code review P2-1). The
longest pattern now wins.
"""
import os
import sys

import pytest

# sampler_utils' import chain needs real torch (see test_sampler_minimax_h3.py).
import torch  # noqa: F401

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sampler_utils import detect_by_config  # noqa: E402


class _FakeModelWithConfig:
    """detect_by_config only reads type(model.model.model_config).__name__."""

    def __init__(self, config_cls_name):
        config_cls = type(config_cls_name, (), {})
        self.model = type("M", (), {"model_config": config_cls()})()


@pytest.mark.parametrize("config_cls, expected", [
    ("Flux2", "flux2"),
    ("Flux", "flux"),
    ("FluxSchnell", "flux"),
    ("FluxInpaint", "flux"),
    ("Chroma", "chroma"),
    ("ChromaRadiance", "chroma"),
    ("WAN21_T2V", "wan"),
    ("WAN22_I2V", "wan"),
    ("LTXV", "ltxv"),
    ("LTXAV", "ltxav"),
    ("HunyuanVideo", "hunyuan_video"),
    ("HunyuanVideoI2V", "hunyuan_video"),
    ("Lumina2", "lumina2"),
    ("ZImage", "z_image"),
    ("CogVideoX", "cogvideox"),
    ("Mochi", "mochi"),
    ("MiniMaxH3", "minimax"),
    ("QwenImage21", "qwen_image21"),
])
def test_config_class_maps_to_its_model_type(config_cls, expected):
    assert detect_by_config(_FakeModelWithConfig(config_cls)) == expected


def test_an_unknown_config_class_is_left_to_the_other_detectors():
    assert detect_by_config(_FakeModelWithConfig("SomethingNew")) is None
