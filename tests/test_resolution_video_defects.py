"""
tests/test_resolution_video_defects.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RadianceResolution's video path: three widgets that lied about what they
produced.

  • enable_video=True against an IMAGE model_type produced N unrelated stills,
    skipped the frame-count stride validation entirely, and reported
    "VIDEO: 81f @ 24fps" with a duration.
  • batch_size was ignored for video (the batch axis is hard-coded to 1) and
    was the only ignored widget in the node with no warning attached.
  • duration_seconds was capped at 120.0 while video_frames reaches 100000, so
    'Auto (Seconds)' could not express any clip 'Manual (Frames)' could.
"""

from __future__ import annotations

import logging

import pytest
import torch

from radiance.nodes.generate.resolution import RadianceResolution


IMAGE_MODEL = "SDXL (4ch)"
VIDEO_MODEL = "WAN (16ch)"

BASE = dict(
    preset="Custom",
    width=512,
    height=512,
    orientation="Landscape",
    batch_size=1,
    crop_to_broadcast_resolution=False,
)


def run(**overrides):
    node = RadianceResolution()
    kwargs = dict(BASE)
    kwargs.update(overrides)
    out = node.generate(**kwargs)
    (latent, w, h, ch, info, fps, frame_count,
     latent_fmt, duration_sec, crop_bbox) = out["result"]
    return {
        "samples": latent["samples"],
        "info": info,
        "duration_sec": duration_sec,
        "frame_count": frame_count,
        "frame_rate": fps,
    }


def _widget(name):
    for group in ("required", "optional"):
        spec = RadianceResolution.INPUT_TYPES().get(group, {})
        if name in spec:
            return spec[name][1]
    raise KeyError(name)


# ─────────────────────────────────────────────────────────────────────────────
#  enable_video against an image architecture
# ─────────────────────────────────────────────────────────────────────────────

class TestVideoFlagAgainstAnImageModel:

    def test_it_warns_that_the_output_is_not_video(self, caplog):
        with caplog.at_level(logging.WARNING, logger="radiance.resolution"):
            run(model_type=IMAGE_MODEL, enable_video=True, video_frames=81)
        messages = [r.getMessage() for r in caplog.records]
        assert any("UNRELATED STILL IMAGES" in m for m in messages), (
            f"a batch of stills was reported as video with no warning: {messages}"
        )

    def test_the_info_string_does_not_claim_video(self):
        info = run(model_type=IMAGE_MODEL, enable_video=True, video_frames=81)["info"]
        assert "video" not in info.lower(), (
            f"info still advertises a video clip for an image model: {info!r}"
        )
        assert "stills" in info.lower()

    def test_no_duration_is_reported_for_a_batch_of_stills(self):
        result = run(model_type=IMAGE_MODEL, enable_video=True,
                     video_frames=81, frame_rate=24.0)
        assert result["duration_sec"] == 0.0, (
            f"a batch of unrelated stills reported a {result['duration_sec']}s "
            f"duration, which a downstream muxer will act on"
        )

    def test_the_latent_really_is_four_dimensional(self):
        """Confirms the defect is real and not just a labelling problem."""
        samples = run(model_type=IMAGE_MODEL, enable_video=True,
                      video_frames=81)["samples"]
        assert samples.ndim == 4
        assert samples.shape[0] == 81

    def test_a_real_video_model_is_unaffected(self, caplog):
        with caplog.at_level(logging.WARNING, logger="radiance.resolution"):
            result = run(model_type=VIDEO_MODEL, enable_video=True,
                         video_frames=81, frame_rate=24.0)
        assert result["samples"].ndim == 5
        assert "video" in result["info"].lower()
        assert result["duration_sec"] == pytest.approx(81 / 24.0)
        assert not any(
            "UNRELATED STILL IMAGES" in r.getMessage() for r in caplog.records
        )

    def test_manual_model_type_still_counts_as_video(self):
        """'Manual' is the documented escape hatch for unlisted architectures."""
        result = run(model_type="Manual", enable_video=True, video_frames=16,
                     latent_channels=16)
        assert result["samples"].ndim == 5
        assert "video" in result["info"].lower()


# ─────────────────────────────────────────────────────────────────────────────
#  batch_size
# ─────────────────────────────────────────────────────────────────────────────

class TestBatchSizeIsIgnoredLoudly:

    def test_it_warns_when_ignored(self, caplog):
        with caplog.at_level(logging.WARNING, logger="radiance.resolution"):
            run(model_type=VIDEO_MODEL, enable_video=True,
                video_frames=17, batch_size=4)
        assert any(
            "batch_size=4 is ignored" in r.getMessage() for r in caplog.records
        ), "the one ignored widget in this node with no warning attached"

    def test_the_batch_axis_really_is_hard_coded_to_one(self):
        samples = run(model_type=VIDEO_MODEL, enable_video=True,
                      video_frames=17, batch_size=4)["samples"]
        assert samples.shape[0] == 1

    def test_no_warning_at_batch_size_one(self, caplog):
        with caplog.at_level(logging.WARNING, logger="radiance.resolution"):
            run(model_type=VIDEO_MODEL, enable_video=True,
                video_frames=17, batch_size=1)
        assert not any("is ignored" in r.getMessage() for r in caplog.records)

    def test_batch_size_still_works_for_images(self, caplog):
        with caplog.at_level(logging.WARNING, logger="radiance.resolution"):
            samples = run(model_type=IMAGE_MODEL, enable_video=False,
                          batch_size=4)["samples"]
        assert samples.shape[0] == 4
        assert not any("batch_size" in r.getMessage() for r in caplog.records)


# ─────────────────────────────────────────────────────────────────────────────
#  duration_seconds ceiling
# ─────────────────────────────────────────────────────────────────────────────

class TestDurationSecondsReachesTheFrameCeiling:
    """Long-clip cases run at the smallest usable frame size on purpose.

    The node allocates the latent for real, and 100000 frames at 512x512 is a
    multi-gigabyte tensor. Frame COUNT is what these tests are about, so the
    spatial size is kept at 64x64 (an 8x8 latent) to keep the allocation small.
    """

    TINY = dict(width=64, height=64)

    def test_the_widget_ceiling_matches_the_frame_ceiling(self):
        duration_max = _widget("duration_seconds")["max"]
        frames_max = _widget("video_frames")["max"]
        assert duration_max > 120.0, (
            f"duration_seconds is still capped at {duration_max}s while "
            f"video_frames reaches {frames_max}; the two entry modes cannot "
            f"express the same clip"
        )
        assert duration_max >= frames_max / _widget("frame_rate")["min"]

    def test_a_ten_minute_clip_is_reachable_by_seconds(self):
        result = run(
            model_type=VIDEO_MODEL, enable_video=True,
            frame_computation="Auto (Seconds)",
            duration_seconds=600.0, frame_rate=24.0, **self.TINY,
        )
        assert result["frame_count"] > 14000, (
            f"600s at 24fps produced only {result['frame_count']} frames"
        )
        assert result["duration_sec"] == pytest.approx(600.0, abs=1.0)

    def test_an_over_ceiling_request_is_clamped_with_a_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="radiance.resolution"):
            result = run(
                model_type=VIDEO_MODEL, enable_video=True,
                frame_computation="Auto (Seconds)",
                duration_seconds=99000.0, frame_rate=24.0, **self.TINY,
            )
        assert result["frame_count"] <= 100000
        assert any("ceiling" in r.getMessage() for r in caplog.records)

    def test_the_clamped_count_is_still_on_the_models_stride_grid(self):
        result = run(
            model_type=VIDEO_MODEL, enable_video=True,
            frame_computation="Auto (Seconds)",
            duration_seconds=99000.0, frame_rate=24.0, **self.TINY,
        )
        # WAN needs 4k + 1.
        assert (result["frame_count"] - 1) % 4 == 0

    def test_a_short_clip_is_unchanged(self):
        result = run(
            model_type=VIDEO_MODEL, enable_video=True,
            frame_computation="Auto (Seconds)",
            duration_seconds=5.0, frame_rate=24.0, **self.TINY,
        )
        assert result["frame_count"] == 121  # round(120/4)*4 + 1
