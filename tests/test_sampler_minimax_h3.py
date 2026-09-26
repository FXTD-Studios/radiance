"""
Tests for MiniMax H3 support in RadianceSamplerPro / sampler_utils.py.

MiniMax H3's reference pipeline uses BasicGuider (comfy_extras/nodes_minimax_h3.py's
official T2V template), which has no cfg input and no guidance-embed mechanism
either -- unlike every other architecture in this codebase, cfg genuinely has zero
effect. See project_radiance_minimax_h3 memory for the full design rationale.

Covers:
  - Table consistency (MODEL_TYPES/VIDEO_MODEL_TYPES/WORKFLOW_PRESETS list it,
    CFG_GUIDED_MODELS/GUIDANCE_EMBED_MODELS deliberately don't)
  - MODEL_DEFAULTS["minimax"] against the real official workflow template's
    KSamplerSelect/BasicScheduler values
  - detect_by_config / detect_by_architecture recognize a fake MiniMax H3 model
"""
import sys
import os

# ALBABIT-FIX: unguarded so conftest.py's AST-based gate auto-skips this whole
# file on the lightweight CI lane (sampler_utils.py's own import chain needs
# real torch, same situation as test_sampler_regression.py).
import torch  # noqa: F401

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sampler_utils import (
    MODEL_TYPES,
    VIDEO_MODEL_TYPES,
    CFG_GUIDED_MODELS,
    GUIDANCE_EMBED_MODELS,
    MODEL_DEFAULTS,
    WORKFLOW_PRESETS,
    detect_by_config,
    detect_by_architecture,
)


class TestMiniMaxH3Registration:
    """Every model-taxonomy table must agree, or the Sampler falls back to
    wrong defaults (e.g. real CFG scaling on a BasicGuider-only pipeline)."""

    def test_listed_in_model_types(self):
        assert "minimax" in MODEL_TYPES

    def test_listed_in_video_model_types(self):
        assert "minimax" in VIDEO_MODEL_TYPES

    def test_not_in_cfg_guided_models(self):
        """cfg has zero effect on BasicGuider -- Dynamic CFG scheduling would
        just be wasted computation, not a real guidance ramp."""
        assert "minimax" not in CFG_GUIDED_MODELS

    def test_not_in_guidance_embed_models(self):
        """No flux_guidance-equivalent mechanism exists for this pipeline
        either -- it isn't guidance-embed, it's no-guidance-at-all."""
        assert "minimax" not in GUIDANCE_EMBED_MODELS

    def test_listed_in_workflow_presets(self):
        assert "[V] MiniMax H3 T2V (20 steps)" in WORKFLOW_PRESETS


class TestMiniMaxH3Defaults:
    """Hand-verified against the official T2V workflow template's subgraph
    (KSamplerSelect=res_multistep, BasicScheduler=simple/20 steps/denoise=1,
    BasicGuider has zero widgets -- confirmed no cfg field exists at all)."""

    def test_defaults_entry_exists(self):
        assert "minimax" in MODEL_DEFAULTS

    def test_cfg_pinned_inert(self):
        assert MODEL_DEFAULTS["minimax"]["cfg"] == 1.0

    def test_sampler_matches_official_template(self):
        assert MODEL_DEFAULTS["minimax"]["sampler"] == "res_multistep"

    def test_scheduler_matches_official_template(self):
        assert MODEL_DEFAULTS["minimax"]["scheduler"] == "simple"

    def test_steps_matches_official_template(self):
        assert MODEL_DEFAULTS["minimax"]["steps"] == 20

    def test_shift_left_neutral(self):
        """No ModelSamplingXXX/shift-adjustment node exists anywhere in the
        reference pipeline -- 1.0 (flux_shift_sigmas' own no-op value)."""
        assert MODEL_DEFAULTS["minimax"]["shift"] == 1.0


class _FakeModelWithConfig:
    """Stands in for a real MODEL: detect_by_config only reads
    type(model.model.model_config).__name__, never instantiates or reads
    attributes off the config object itself."""

    def __init__(self, config_cls_name):
        config_cls = type(config_cls_name, (), {})
        self.model = type("M", (), {"model_config": config_cls()})()


class _FakeModelWithDiffusionModel:
    """Stands in for a real MODEL: detect_by_architecture only reads the
    diffusion_model object's own class name/module, via get_model_object."""

    def __init__(self, cls_name, module_name):
        diffusion_cls = type(cls_name, (), {"__module__": module_name})
        self._diffusion_model = diffusion_cls()

    def get_model_object(self, name):
        return self._diffusion_model if name == "diffusion_model" else None


class TestMiniMaxH3Detection:
    def test_detect_by_config_recognizes_minimax_h3_config_class(self):
        model = _FakeModelWithConfig("MiniMaxH3")
        assert detect_by_config(model) == "minimax"

    def test_detect_by_architecture_falls_back_on_diffusion_model_name(self):
        model = _FakeModelWithDiffusionModel("MiniMaxH3Model", "comfy.ldm.minimax.model")
        assert detect_by_architecture(model) == "minimax"
