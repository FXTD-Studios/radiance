"""
Tests for MiniMax H3 support in RadianceResolution (nodes/generate/resolution.py).

MiniMax H3's frame count snaps to a 17k+5 grid (5, 22, 39...), not the plain
stride*k+1 pattern every other VIDEO_MODEL_TYPES entry uses, and the model has
no variable-frame-rate support (fixed 24fps). See project_radiance_minimax_h3
memory and comfy_extras/nodes_minimax_h3.py (align_frame_count/video_latent_t/
temporal_shape) for the reference formulas these mirror.

Covers:
  - Table consistency (MODEL_TYPES/VIDEO_MODEL_TYPES/LATENT_CHANNELS/
    SPATIAL_SCALE/LATENT_FORMAT_MAP all agree on the same key)
  - _minimax_align_frame_count / _minimax_video_latent_t against hand-verified
    values, including the floor-snap behavior for off-grid input
  - generate() end-to-end: latent shape, Auto-Seconds fixed-24fps alignment,
    Manual-mode warn-without-override, and the frame_rate-mismatch warning
"""
import sys
import os
from unittest.mock import MagicMock

# ALBABIT-FIX: unguarded so conftest.py's AST-based gate auto-skips this whole
# file on the lightweight CI lane (stub torch can't back real .shape asserts).
import torch  # noqa: F401

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nodes.generate.resolution as resolution_module
from nodes.generate.resolution import (
    RadianceResolution,
    MODEL_TYPES,
    VIDEO_MODEL_TYPES,
    LATENT_CHANNELS,
    SPATIAL_SCALE,
    SPATIAL_ALIGN,
    TEMPORAL_SCALE,
    LATENT_FORMAT_MAP,
    MINIMAX_H3_MODEL_TYPE,
    MINIMAX_H3_FPS,
    MINIMAX_H3_AUDIO_FPS,
    _minimax_align_frame_count,
    _minimax_floor_frame_count,
    _minimax_video_latent_t,
    _minimax_audio_latent_t,
)


class _FakeNestedTensor:
    """conftest.py stubs comfy.nested_tensor.NestedTensor to None ("some
    builds don't have it"). MiniMax H3 genuinely needs a working one, so
    this augments the stub rather than replacing conftest's module wholesale
    (same convention as test_sampler_regression.py's torch_mock augmentation).
    Mirrors the real class's constructor + shape/tensors access only, all
    generate() and these tests actually touch."""

    def __init__(self, tensors):
        self.tensors = list(tensors)
        self.is_nested = True

    @property
    def shape(self):
        return self.tensors[0].shape


import comfy.nested_tensor  # installed by conftest  # noqa: E402
comfy.nested_tensor.NestedTensor = _FakeNestedTensor


class TestMiniMaxH3Registration:
    """Every model_type table must agree on the same key, or generate() falls
    back to wrong defaults (e.g. 4ch/8px) silently instead of erroring."""

    def test_listed_in_model_types(self):
        assert MINIMAX_H3_MODEL_TYPE in MODEL_TYPES

    def test_listed_in_video_model_types(self):
        assert MINIMAX_H3_MODEL_TYPE in VIDEO_MODEL_TYPES

    def test_latent_channels_is_24(self):
        assert LATENT_CHANNELS[MINIMAX_H3_MODEL_TYPE] == 24

    def test_spatial_scale_is_16(self):
        assert SPATIAL_SCALE[MINIMAX_H3_MODEL_TYPE] == 16

    def test_pixel_alignment_is_32(self):
        assert SPATIAL_ALIGN[MINIMAX_H3_MODEL_TYPE] == 32

    def test_latent_format_matches_supported_models_unet_config(self):
        assert LATENT_FORMAT_MAP[MINIMAX_H3_MODEL_TYPE] == "minimax_h3"

    def test_deliberately_absent_from_temporal_scale(self):
        """The 17k+5 grid isn't a fixed divisor. generate() must branch on
        MINIMAX_H3_MODEL_TYPE before ever reaching TEMPORAL_SCALE.get()."""
        assert MINIMAX_H3_MODEL_TYPE not in TEMPORAL_SCALE

    def test_fps_matches_native_node_hardcoded_constant(self):
        assert MINIMAX_H3_FPS == 24


class TestMinimaxAlignFrameCount:
    """Mirrors comfy_extras/nodes_minimax_h3.py's align_frame_count()."""

    def test_already_on_grid_is_unchanged(self):
        for n in (5, 22, 39, 56, 124):
            assert _minimax_align_frame_count(n) == n

    def test_rounds_up_to_next_grid_point(self):
        # Hand-verified: 80 % 17 == 12, nearest grid points are 73 and 90.
        assert _minimax_align_frame_count(80) == 90

    def test_floors_below_five_up_to_five(self):
        assert _minimax_align_frame_count(1) == 5
        assert _minimax_align_frame_count(0) == 5

    def test_one_above_grid_point_rounds_up_a_full_cycle(self):
        assert _minimax_align_frame_count(23) == 39


class TestMinimaxFloorFrameCount:
    """Companion to _minimax_align_frame_count, snapping down instead of up."""

    def test_already_on_grid_is_unchanged(self):
        for n in (5, 22, 39, 56, 124):
            assert _minimax_floor_frame_count(n) == n

    def test_rounds_down_to_previous_grid_point(self):
        # Same hand-verified pair as the align-up test: 80 % 17 == 12,
        # nearest grid points are 73 and 90; this one floors to 73.
        assert _minimax_floor_frame_count(80) == 73

    def test_never_goes_below_five(self):
        assert _minimax_floor_frame_count(1) == 5
        assert _minimax_floor_frame_count(0) == 5

    def test_matches_what_video_latent_t_already_uses_internally(self):
        """_minimax_video_latent_t floors off-grid input internally (see
        TestMinimaxVideoLatentT.test_off_grid_input_floor_snaps_to_lower_grid_neighbor).
        This must produce the identical t for the floored frame count as for
        the raw one, or audio_t (derived from the floored count) would drift
        out of sync with the video stream it's supposed to pair with."""
        for raw in (80, 81, 89):
            floored = _minimax_floor_frame_count(raw)
            assert _minimax_video_latent_t(floored) == _minimax_video_latent_t(raw)


class TestMinimaxVideoLatentT:
    """Mirrors comfy_extras/nodes_minimax_h3.py's video_latent_t()."""

    def test_base_case_at_or_below_five(self):
        assert _minimax_video_latent_t(5) == 2
        assert _minimax_video_latent_t(1) == 2

    def test_hand_verified_grid_values(self):
        assert _minimax_video_latent_t(22) == 7
        assert _minimax_video_latent_t(124) == 37

    def test_off_grid_input_floor_snaps_to_lower_grid_neighbor(self):
        """video_latent_t(n) for unaligned n equals video_latent_t() at the
        nearest LOWER grid point (integer division floors), matching how the
        generic stride*k+1 models already behave on off-grid video_frames,
        so generate() can call it directly without pre-aligning."""
        assert _minimax_video_latent_t(80) == _minimax_video_latent_t(73) == 22


class TestMinimaxAudioLatentT:
    """Mirrors comfy_extras/nodes_minimax_h3.py's temporal_shape()'s audio_t:
    round((frame_count / 24) * 40). Hand-verified, not re-derived from the
    same formula under test."""

    def test_minimum_grid_value(self):
        assert _minimax_audio_latent_t(5) == 8

    def test_hand_verified_grid_values(self):
        assert _minimax_audio_latent_t(22) == 37
        assert _minimax_audio_latent_t(124) == 207
        assert _minimax_audio_latent_t(141) == 235

    def test_fps_constant_matches_reference(self):
        assert MINIMAX_H3_AUDIO_FPS == 40


class TestGenerateMiniMaxH3Latent:
    """End-to-end generate() calls. Exercises the three MiniMax-specific
    branch points (Auto-Seconds, frame-count validation, latent construction)
    together, not just the pure helpers in isolation."""

    def _generate(self, node, **overrides):
        params = dict(
            preset="Custom", width=1344, height=768, orientation="As Preset",
            model_type=MINIMAX_H3_MODEL_TYPE, batch_size=1,
            enable_video=True, frame_computation="Manual (Frames)",
            duration_seconds=5.0, video_frames=124, frame_rate=24.0,
            unique_id="test",
        )
        params.update(overrides)
        return node.generate(**params)["result"]

    def test_on_grid_manual_frames_produces_correct_shape(self):
        node = RadianceResolution()
        latent, w, h, c, info, fr, frames, fmt, dur, crop_bbox = self._generate(node, video_frames=124)
        # width=1344, height=768 already on the 32px alignment; latent is //16 -> 84x48.
        assert tuple(latent["samples"].shape) == (1, 24, 37, 48, 84)
        assert (w, h, c) == (1344, 768, 24)
        assert fmt == "minimax_h3"
        assert frames == 124

    def test_latent_is_a_genuine_video_audio_pair(self):
        """MiniMaxH3Model.forward() does audio_src = x[1] unconditionally
        (comfy/ldm/minimax/model.py). A plain video-only tensor crashes
        with IndexError the instant sampling starts, even for pure T2V with
        no real audio content. Regression guard for that real, live-found bug."""
        node = RadianceResolution()
        latent, *_ = self._generate(node, video_frames=124)
        samples = latent["samples"]
        assert samples.is_nested is True
        assert len(samples.tensors) == 2
        video, audio = samples.tensors
        assert tuple(video.shape) == (1, 24, 37, 48, 84)
        # audio_t = round((124/24)*40) = 207, silence encoded as zeros.
        assert tuple(audio.shape) == (1, 32, 2, 207)
        assert torch.count_nonzero(audio) == 0

    def test_off_grid_manual_frames_warns_but_does_not_override(self, monkeypatch):
        # ALBABIT-FIX: radiance's "radiance" logger has propagate=False
        # (core/logging.py), so caplog's root-only handler never sees these
        # records. Mocking the module's logger directly sidesteps that.
        mock_logger = MagicMock()
        monkeypatch.setattr(resolution_module, "logger", mock_logger)
        node = RadianceResolution()
        latent, w, h, c, info, fr, frames, fmt, dur, crop_bbox = self._generate(node, video_frames=80)
        # video_frames output stays the user's raw value (warn-only, matching
        # the existing Manual-mode behavior for every other video model_type)...
        assert frames == 80
        # ...but the actual tensor floor-snaps to a decodable grid value.
        assert latent["samples"].shape[2] == 22
        text = "\n".join(str(call) for call in mock_logger.warning.call_args_list)
        assert "17" in text and "80" in text

    def test_off_grid_manual_frames_keep_video_and_audio_in_sync(self):
        """Real bug caught in code review: audio_t used to derive from a
        ceiling-aligned frame count (90) while video used a floor-aligned one
        (73) for this same off-grid input, drifting the two streams out of
        sync. Both must now derive from the identical floored basis."""
        node = RadianceResolution()
        latent, *_ = self._generate(node, video_frames=80)
        video, audio = latent["samples"].tensors
        assert video.shape[2] == _minimax_video_latent_t(73)
        # audio_t = round((73/24)*40) = 122, not round((90/24)*40) = 150.
        assert audio.shape[3] == 122

    def test_auto_seconds_uses_fixed_24fps_regardless_of_frame_rate_widget(self):
        node = RadianceResolution()
        latent, w, h, c, info, fr, frames, fmt, dur, crop_bbox = self._generate(
            node, frame_computation="Auto (Seconds)", duration_seconds=5.0, frame_rate=30.0,
        )
        # 5.0s * 24fps (fixed) = 120 -> aligned up to 124, NOT the 150 that
        # 5.0*30 would have produced.
        assert frames == 124
        assert latent["samples"].shape[2] == _minimax_video_latent_t(124)
        # duration_sec is still the widget's frame_rate passthrough contract.
        assert dur == 124 / 30.0

    def test_auto_seconds_frame_rate_mismatch_warns(self, monkeypatch):
        mock_logger = MagicMock()
        monkeypatch.setattr(resolution_module, "logger", mock_logger)
        node = RadianceResolution()
        self._generate(
            node, frame_computation="Auto (Seconds)", duration_seconds=5.0, frame_rate=30.0,
        )
        text = "\n".join(str(call) for call in mock_logger.warning.call_args_list)
        assert "24" in text and "30" in text

    def test_auto_seconds_at_native_frame_rate_is_silent(self, monkeypatch):
        # Note: the test stub's folder_paths lacks get_temp_directory, so
        # generate()'s (unrelated) preview-card save always warns here too.
        # Check specifically for the fps-mismatch warning, not "zero warnings".
        mock_logger = MagicMock()
        monkeypatch.setattr(resolution_module, "logger", mock_logger)
        node = RadianceResolution()
        self._generate(
            node, frame_computation="Auto (Seconds)", duration_seconds=5.0, frame_rate=24.0,
        )
        text = "\n".join(str(call) for call in mock_logger.warning.call_args_list)
        assert "fixed at" not in text

    def test_width_height_align_to_32px_but_latent_stays_16x(self):
        node = RadianceResolution()
        latent, w, h, c, info, fr, frames, fmt, dur, crop_bbox = self._generate(node, width=1350, height=770)
        assert (w, h) == (1376, 800)
        assert tuple(latent["samples"].shape[-2:]) == (h // 16, w // 16)

    def test_720p_gets_an_even_latent_so_keyframes_can_patchify(self):
        # 720 // 16 = 45 rows is odd and crashes patchify_video on keyframe latents.
        node = RadianceResolution()
        latent, w, h, c, info, fr, frames, fmt, dur, crop_bbox = self._generate(node, width=1280, height=720)
        assert (w, h) == (1280, 736)
        assert latent["samples"].shape[-2] % 2 == 0
        assert crop_bbox == {"x": 0, "y": 8, "width": 1280, "height": 720}


class TestNonMiniMaxRegressionGuard:
    """resolution.py had no test coverage at all before this file. The three
    generate() branch points touched for MiniMax H3 (Auto-Seconds, frame-count
    validation, latent construction) each gained an `if MINIMAX... else:` split,
    so a representative non-MiniMax video model_type (LTXV) is checked here to
    guard the untouched-but-reindented else-branches, not as a full audit."""

    def _generate(self, node, **overrides):
        params = dict(
            preset="Custom", width=1280, height=704, orientation="As Preset",
            model_type="LTXV (128ch)", batch_size=1,
            enable_video=True, frame_computation="Manual (Frames)",
            duration_seconds=3.0, video_frames=81, frame_rate=24.0,
            unique_id="ltx-regression",
        )
        params.update(overrides)
        return node.generate(**params)["result"]

    def test_ltxv_manual_frames_still_uses_stride_formula(self):
        node = RadianceResolution()
        latent, w, h, c, info, fr, frames, fmt, dur, crop_bbox = self._generate(node, video_frames=81)
        assert latent["samples"].shape[2] == (81 - 1) // 8 + 1
        assert fmt == "ltxav"

    def test_ltxv_auto_seconds_still_uses_frame_rate_widget(self):
        node = RadianceResolution()
        latent, w, h, c, info, fr, frames, fmt, dur, crop_bbox = self._generate(
            node, frame_computation="Auto (Seconds)", duration_seconds=3.0, frame_rate=24.0,
        )
        # 3.0*24=72 -> round(72/8)*8+1 = 73 (unlike MiniMax, not fixed-24fps-only).
        assert frames == 73

    def test_ltxv_off_grid_frames_still_warn_only_not_overridden(self, monkeypatch):
        mock_logger = MagicMock()
        monkeypatch.setattr(resolution_module, "logger", mock_logger)
        node = RadianceResolution()
        latent, w, h, c, info, fr, frames, fmt, dur, crop_bbox = self._generate(node, video_frames=80)
        assert frames == 80
        text = "\n".join(str(call) for call in mock_logger.warning.call_args_list)
        assert "stride*k" in text or "8k+1" in text or "requires frame count" in text
