"""
Tests for RadianceResolution's crop_bbox output (nodes/generate/resolution.py).

Restored from the previous radiance version, generalized: the old fork only
recognized a fixed table of known broadcast standards (1088->1080, 736/768->720,
2176->2160) and only for enable_video. This version reports the diff between the
requested size and whatever align_val (model-specific, SPATIAL_SCALE) padded it
to, for any preset/model_type/custom size, for images too. See
project_radiance_beta memory for the full design discussion.
"""
import sys
import os

import torch  # noqa: F401

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodes.generate.resolution import RadianceResolution


def _generate(node, **overrides):
    params = dict(
        preset="Custom", width=1920, height=1080, orientation="As Preset",
        model_type="LTXV (128ch)", batch_size=1,
        unique_id="crop-bbox-test",
    )
    params.update(overrides)
    return node.generate(**params)["result"]


class TestCropBboxPadding:
    def test_1080p_on_ltxv_32px_grid_crops_back_from_1088(self):
        # 1080 is not a multiple of 32 -> aligned up to 1088; crop_bbox undoes it.
        node = RadianceResolution()
        _latent, w, h, *_rest, crop_bbox = _generate(node)
        assert (w, h) == (1920, 1088)
        assert crop_bbox == {"x": 0, "y": 4, "width": 1920, "height": 1080}

    def test_already_aligned_size_gets_a_full_frame_no_op_box(self):
        # 1920x1920 is already a multiple of 32 on both axes -> nothing to crop.
        node = RadianceResolution()
        _latent, w, h, *_rest, crop_bbox = _generate(node, width=1920, height=1920)
        assert (w, h) == (1920, 1920)
        assert crop_bbox == {"x": 0, "y": 0, "width": 1920, "height": 1920}

    def test_toggle_off_always_reports_the_full_padded_frame(self):
        node = RadianceResolution()
        _latent, w, h, *_rest, crop_bbox = _generate(
            node, crop_to_broadcast_resolution=False,
        )
        assert (w, h) == (1920, 1088)
        assert crop_bbox == {"x": 0, "y": 0, "width": 1920, "height": 1088}

    def test_default_8px_alignment_still_works_for_a_non_video_model(self):
        # SDXL uses the SPATIAL_SCALE fallback (8px). 1001 is not 8-aligned
        # (rounds up to 1008), unlike the LTXV/32px cases above.
        node = RadianceResolution()
        _latent, w, h, *_rest, crop_bbox = _generate(
            node,
            model_type="SDXL / SD 1.5 / PixArt / Aura Flow (4ch)",
            width=1001, height=1001,
        )
        assert (w, h) == (1008, 1008)
        assert crop_bbox == {"x": 3, "y": 3, "width": 1001, "height": 1001}

    def test_orientation_swap_keeps_the_crop_on_the_right_axis(self):
        # Portrait-shaped request, swapped to landscape *after* alignment. The
        # padding must end up on the axis it actually lands on post-swap, not
        # wherever it was pre-swap.
        node = RadianceResolution()
        _latent, w, h, *_rest, crop_bbox = _generate(
            node, width=1080, height=1920, orientation="Landscape",
        )
        assert (w, h) == (1920, 1088)
        assert crop_bbox == {"x": 0, "y": 4, "width": 1920, "height": 1080}

    def test_applies_to_still_images_not_just_video(self):
        # enable_video defaults to False here; the old fork gated this whole
        # mechanism on enable_video, this version does not.
        node = RadianceResolution()
        _latent, w, h, *_rest, crop_bbox = _generate(node, enable_video=False)
        assert (w, h) == (1920, 1088)
        assert crop_bbox == {"x": 0, "y": 4, "width": 1920, "height": 1080}

    def test_scale_factor_is_ignored_for_the_crop_target(self):
        # LTX 2.3's LowRes -> 2x latent upscale -> HighRes pattern: this
        # Resolution call's own w/h output (960x544) is the scaled-down LowRes
        # latent size, but the pipeline's upscale brings the actual decode
        # back to 1920x1088, so crop_bbox must target that instead.
        node = RadianceResolution()
        _latent, w, h, *_rest, crop_bbox = _generate(
            node, scale_factor=0.5, enable_video=True,
            frame_computation="Auto (Seconds)", duration_seconds=10.04,
        )
        assert (w, h) == (960, 544)  # this call's own (scaled) latent size, unaffected
        assert crop_bbox == {"x": 0, "y": 4, "width": 1920, "height": 1080}
