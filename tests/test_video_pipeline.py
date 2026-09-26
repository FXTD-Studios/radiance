"""
tests/test_video_pipeline.py
Unit tests for the Radiance v3 Video-First Pipeline modules:
  • radiance/nodes/ai/scene_cut.py  — detect_cuts(), SceneCutDetect, Split, Router
"""

import sys
import os
import json
import types
import importlib
import pytest
from pathlib import Path

import numpy as np

# ── ComfyUI stubs ─────────────────────────────────────────────────────────────
for _mod in ["folder_paths", "comfy", "comfy.utils"]:
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)
_fp = sys.modules["folder_paths"]
if not hasattr(_fp, "get_output_directory"):
    _fp.get_output_directory = lambda: "/tmp/comfy_output"

# ── torch guard ───────────────────────────────────────────────────────────────
try:
    import torch
    HAS_TORCH = hasattr(torch, "__version__")
except ImportError:
    HAS_TORCH = False

skip_no_torch = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")

# ── module imports ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))


def _import(mod_name):
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    return importlib.import_module(mod_name)


# ═════════════════════════════════════════════════════════════════════════════
# radiance/nodes/ai/scene_cut.py  — detect_cuts (pure numpy, no torch required)
# ═════════════════════════════════════════════════════════════════════════════

class TestDetectCuts:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    """Test the core detect_cuts() algorithm directly."""

    def setup_method(self):
        self.fn = _import("radiance.nodes.ai.scene_cut").detect_cuts

    def _make_shots(self, shot_lengths, h=32, w=32):
        """Build a synthetic batch with hard cuts between shots."""
        frames = []
        for i, length in enumerate(shot_lengths):
            colour = np.ones((h, w, 3), dtype=np.float32) * (i * 0.3 % 1.0)
            frames.extend([colour.copy()] * length)
        return np.stack(frames, axis=0)

    def test_no_cuts_uniform(self):
        frames = np.ones((20, 32, 32, 3), dtype=np.float32) * 0.5
        cuts, scores = self.fn(frames, threshold=0.3, min_shot_frames=4)
        assert cuts == [0]   # only the implicit start
        assert len(scores) == 19

    def test_single_hard_cut(self):
        """Two clearly different shots → exactly one internal cut."""
        frames = self._make_shots([10, 10], h=32, w=32)
        cuts, scores = self.fn(frames, threshold=0.20, min_shot_frames=4)
        assert len(cuts) == 2
        assert cuts[0] == 0
        assert cuts[1] == 10  # cut at frame 10

    def test_multiple_cuts(self):
        frames = self._make_shots([8, 8, 8, 8], h=32, w=32)
        cuts, scores = self.fn(frames, threshold=0.15, min_shot_frames=4)
        assert len(cuts) == 4

    def test_min_shot_frames_enforced(self):
        """Two cuts very close together — min_shot_frames should suppress one."""
        frames = self._make_shots([2, 2, 20], h=32, w=32)
        cuts, _ = self.fn(frames, threshold=0.15, min_shot_frames=5)
        # First cut (at frame 2) is too close to start — should be suppressed
        for c in cuts[1:]:
            assert c - cuts[cuts.index(c) - 1] >= 5

    def test_scores_length(self):
        frames = np.random.rand(15, 16, 16, 3).astype(np.float32)
        cuts, scores = self.fn(frames)
        assert len(scores) == 14  # B-1

    def test_cuts_always_start_at_zero(self):
        frames = np.random.rand(10, 16, 16, 3).astype(np.float32)
        cuts, _ = self.fn(frames)
        assert cuts[0] == 0

    def test_single_frame_no_crash(self):
        frames = np.ones((1, 16, 16, 3), dtype=np.float32)
        cuts, scores = self.fn(frames)
        assert cuts == [0]

    def test_methods(self):
        frames = self._make_shots([10, 10])
        for method in ("histogram", "edge", "combined"):
            cuts, _ = self.fn(frames, threshold=0.15, min_shot_frames=4, method=method)
            assert len(cuts) >= 1


@skip_no_torch
class TestSceneCutDetectNode:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    def setup_method(self):
        self.mod = _import("radiance.nodes.ai.scene_cut")
        self.cls = self.mod.RadianceSceneCutDetect

    def test_registered(self):
        assert "RadianceSceneCutDetect" in self.mod.NODE_CLASS_MAPPINGS

    def test_detect_returns_three_outputs(self):
        frames = np.zeros((20, 32, 32, 3), dtype=np.float32)
        import torch
        t = torch.from_numpy(frames)
        cut_data, shot_count, plot = self.cls().detect(
            t, cut_confidence=0.5, min_shot_frames=8, method="combined"
        )
        assert isinstance(cut_data, str)
        assert isinstance(shot_count, int)
        assert shot_count >= 1
        assert plot.shape[-1] == 3   # RGB plot

    def test_cut_data_is_valid_json(self):
        import torch
        frames = np.random.rand(20, 32, 32, 3).astype(np.float32)
        cut_data, _, _ = self.cls().detect(
            torch.from_numpy(frames), 0.3, 8, "histogram"
        )
        data = json.loads(cut_data)
        assert "cuts" in data
        assert "shots" in data
        assert "shot_count" in data
        assert data["cuts"][0] == 0


@skip_no_torch
class TestSceneCutSplitNode:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    def setup_method(self):
        self.mod = _import("radiance.nodes.ai.scene_cut")
        self.split = self.mod.RadianceSceneCutSplit
        self.detect = self.mod.RadianceSceneCutDetect

    def test_split_extracts_correct_frames(self):
        import torch
        # Shot 0: black (10 frames), Shot 1: white (10 frames)
        black = np.zeros((10, 16, 16, 3), dtype=np.float32)
        white = np.ones( (10, 16, 16, 3), dtype=np.float32)
        frames = torch.from_numpy(np.concatenate([black, white], axis=0))

        cut_data, _, _ = self.detect().detect(frames, 0.2, 5, "histogram")

        shot0, idx, s, e, info = self.split().split(frames, cut_data, 0)
        assert shot0.shape[0] > 0
        assert idx == 0
        assert s == 0

    def test_shot_index_clamping(self):
        import torch
        frames = torch.zeros(10, 16, 16, 3)
        cut_data = json.dumps({
            "cuts": [0, 5], "shots": [
                {"shot": 0, "start": 0, "end": 4, "length": 5},
                {"shot": 1, "start": 5, "end": 9, "length": 5},
            ], "shot_count": 2
        })
        _, idx, _, _, _ = self.split().split(frames, cut_data, 999)
        assert idx == 1   # clamped to last valid shot


@skip_no_torch
class TestShotGradeRouter:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    def setup_method(self):
        self.cls = _import("radiance.nodes.ai.scene_cut").RadianceShotGradeRouter

    def test_returns_defaults_for_empty_table(self):
        ev, temp, sat, con, idx = self.cls().route(0, "[]")
        assert ev   == 0.0
        assert temp == 0.0
        assert sat  == 1.0
        assert con  == 1.0

    def test_reads_correct_entry(self):
        table = json.dumps([
            {"exposure": 0.5, "temperature": 100, "saturation": 0.8, "contrast": 1.2},
            {"exposure": 1.0, "temperature": 200, "saturation": 0.7, "contrast": 0.9},
        ])
        ev, temp, sat, con, idx = self.cls().route(1, table)
        assert ev   == pytest.approx(1.0)
        assert temp == pytest.approx(200.0)

    def test_clamps_to_last_entry(self):
        table = json.dumps([
            {"exposure": 0.0},
            {"exposure": 1.5},
        ])
        ev, *_ = self.cls().route(999, table)
        assert ev == pytest.approx(1.5)

    def test_invalid_json_uses_defaults(self):
        ev, temp, sat, con, _ = self.cls().route(0, "not_json")
        assert ev   == 0.0
        assert sat  == 1.0


# ═════════════════════════════════════════════════════════════════════════════
# NODE_CLASS_MAPPINGS completeness
# ═════════════════════════════════════════════════════════════════════════════

@skip_no_torch
class TestAllModulesRegistered:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    def test_scene_cut_nodes_registered(self):
        mod = _import("radiance.nodes.ai.scene_cut")
        for key in ("RadianceSceneCutDetect", "RadianceSceneCutSplit",
                    "RadianceShotGradeRouter"):
            assert key in mod.NODE_CLASS_MAPPINGS, f"{key} missing"

    def test_all_display_names_use_radiance_prefix(self):
        for mod_name in ("radiance.nodes.ai.scene_cut",):
            mod = _import(mod_name)
            for key, name in mod.NODE_DISPLAY_NAME_MAPPINGS.items():
                assert name.startswith("◎"), \
                    f"{mod_name}.{key} display name missing ◎ prefix: {name!r}"


def test_pipelines_refuse_an_image_model_with_a_clear_message():
    """Live 3.5 run: Z-Image into T2V failed inside the model's forward with
    "too many values to unpack (expected 4)". It must name the cause instead."""
    import types
    import pytest
    from radiance.nodes.video import t2v as T

    image_model = types.SimpleNamespace(model=types.SimpleNamespace(
        latent_format=type("Flux", (), {"latent_dimensions": 2})()))
    with pytest.raises(ValueError, match="needs a video model.*image model"):
        T._require_video_model(image_model, "RadianceT2VPipeline")

    video_model = types.SimpleNamespace(model=types.SimpleNamespace(
        latent_format=type("Wan21", (), {"latent_dimensions": 3})()))
    T._require_video_model(video_model, "RadianceT2VPipeline")      # no raise
    T._require_video_model(object(), "RadianceT2VPipeline")         # unknown: let through
