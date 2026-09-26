import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from radiance.nodes.vfx.multipass import aov_reader
from radiance.nodes.vfx.multipass.core import _ssao_multisampled
from radiance.nodes.vfx.multipass.master import (
    RadianceEXRPassesWriter,
    RadianceMultipassMaster,
    _match_optional_image,
    _pass_channels_for_multilayer,
)
from radiance.nodes.vfx.multipass.relight_comp import (
    RadianceMultipassComposite,
    RadianceMultipassRelight,
)
import pytest

# Only the tests marked @pytest.mark.real_torch below need genuine tensors; the
# rest run fine against conftest's stub. Opt out of the automatic module-level
# skip so they keep running on the no-torch CI matrix.
RADIANCE_TORCH_GATED = True


try:
    import OpenEXR  # noqa: F401
    HAS_OPENEXR = True
except ImportError:
    HAS_OPENEXR = False


class TestMultipassContracts(unittest.TestCase):
    @pytest.mark.real_torch
    def test_optional_batch_allows_singleton_and_rejects_partial_batch(self):
        self.assertEqual(_match_optional_image(torch.zeros(1, 4, 4, 3), 3, 4, 4).shape[0], 3)
        with self.assertRaisesRegex(ValueError, "Batch mismatch"):
            _match_optional_image(torch.zeros(2, 4, 4, 3), 3, 4, 4)

    @pytest.mark.real_torch
    def test_relight_reads_ao_in_renderer_convention(self):
        # 1 = open, 0 = occluded, as Arnold/Cycles/Karma and Multipass Estimate write it.
        relight = RadianceMultipassRelight()
        albedo = torch.ones(1, 2, 2, 3)
        normal = torch.tensor([0.5, 0.5, 1.0]).view(1, 1, 1, 3).expand_as(albedo)
        common = dict(albedo=albedo, normal_map=normal, intensity=0.0, ambient=1.0)
        open_result = relight.relight(ao=torch.ones_like(albedo), **common)[0]
        blocked_result = relight.relight(ao=torch.zeros_like(albedo), **common)[0]
        default_result = relight.relight(**common)[0]
        self.assertTrue(torch.allclose(default_result, open_result))
        self.assertTrue(torch.allclose(open_result, torch.ones_like(open_result)))
        self.assertTrue(torch.allclose(blocked_result, torch.zeros_like(blocked_result)))

    @pytest.mark.real_torch
    def test_relight_defaults_to_straight_rgb(self):
        relight = RadianceMultipassRelight()
        albedo = torch.ones(1, 1, 1, 3)
        normal = torch.tensor([0.5, 0.5, 1.0]).view(1, 1, 1, 3)
        alpha = torch.full_like(albedo, 0.25)
        straight = relight.relight(albedo, normal, alpha=alpha, intensity=0.0, ambient=1.0)[0]
        premult = relight.relight(
            albedo, normal, alpha=alpha, intensity=0.0, ambient=1.0, output_premultiplied=True
        )[0]
        self.assertTrue(torch.allclose(straight, torch.ones_like(straight)))
        self.assertTrue(torch.allclose(premult, torch.full_like(premult, 0.25)))

    @pytest.mark.real_torch
    def test_premultiplied_foreground_respects_depth_holdout(self):
        comp = RadianceMultipassComposite()
        foreground = torch.tensor([0.5, 0.0, 0.0]).view(1, 1, 1, 3)
        alpha = torch.full((1, 1, 1, 3), 0.5)
        background = torch.ones(1, 1, 1, 3)
        result = comp.composite(
            foreground, alpha, background=background,
            foreground_depth=torch.ones_like(alpha), background_depth=torch.zeros_like(alpha),
            premultiplied_input=True, depth_near_is_white=False,
        )[0]
        self.assertTrue(torch.allclose(result, background))

    @pytest.mark.real_torch
    def test_ssao_preserves_occlusion_shape_and_range(self):
        depth = torch.linspace(0.0, 1.0, 64).reshape(1, 8, 8)
        ao = _ssao_multisampled(depth, None, 2.0, 1.0, 4, False)
        self.assertEqual(tuple(ao.shape), (1, 8, 8))
        self.assertGreaterEqual(float(ao.min()), 0.0)
        self.assertLessEqual(float(ao.max()), 1.0)

    def test_motion_channels_are_raw_xy(self):
        channels = _pass_channels_for_multilayer("motion_vector", np.zeros((2, 2, 3), np.float32))
        self.assertEqual(set(channels), {"MV.X", "MV.Y"})
        alpha = _pass_channels_for_multilayer("alpha", np.zeros((2, 2, 3), np.float32))
        self.assertEqual(set(alpha), {"A"})

    def test_data_window_is_placed_at_its_display_offset(self):
        class Point:
            def __init__(self, x, y):
                self.x, self.y = x, y

        class Window:
            def __init__(self, x0, y0, x1, y1):
                self.min, self.max = Point(x0, y0), Point(x1, y1)

        data = Window(1, 1, 1, 1)
        display = Window(0, 0, 2, 2)
        bounds = aov_reader._canvas_bounds([data], display)
        canvas = aov_reader._place_in_canvas(np.asarray([[7.0]], np.float32), data, bounds)
        self.assertEqual(canvas.shape, (3, 3))
        self.assertEqual(float(canvas[1, 1]), 7.0)

    def test_bare_z_channel_maps_to_depth(self):
        self.assertEqual(aov_reader._layer_key_for_channel("Z"), ("depth", "Z"))

    @pytest.mark.real_torch
    def test_master_gap_fill_preserves_present_renderer_pass(self):
        master = RadianceMultipassMaster()
        beauty = torch.rand(1, 8, 8, 3)
        depth = torch.full((1, 8, 8, 3), 0.5)
        normal = torch.tensor([0.5, 0.5, 1.0]).view(1, 1, 1, 3).expand(1, 8, 8, 3)
        renderer_albedo = torch.full((1, 8, 8, 3), 0.25)
        result = master._legacy_extract(
            beauty, depth_map=depth, normal_map=normal,
            source_passes={"albedo": renderer_albedo, "_present": ["albedo"]},
            ao_strength=0.0, object_id_segments=2,
        )
        self.assertTrue(torch.equal(result[0]["albedo"], renderer_albedo))
        self.assertTrue(torch.equal(result[2], renderer_albedo))
        self.assertEqual(result[0]["motion_vector"].shape[-1], 3)
        self.assertEqual(len(result), 22)

    def test_master_is_retired_with_a_pointer_to_estimate(self):
        self.assertTrue(RadianceMultipassMaster.DEPRECATED)
        with self.assertRaisesRegex(RuntimeError, "Multipass Estimate"):
            RadianceMultipassMaster().extract(torch.zeros(1, 4, 4, 3))

    def test_writer_rejects_filename_paths_before_writing(self):
        writer = RadianceEXRPassesWriter()
        passes = {"beauty": torch.zeros(1, 2, 2, 3)}
        with self.assertRaisesRegex(ValueError, "filename_prefix"):
            writer.write_passes(passes, "../escape", output_path=tempfile.mkdtemp())

    @pytest.mark.real_torch
    def test_writer_rejects_lossy_data_aov_compression(self):
        writer = RadianceEXRPassesWriter()
        passes = {
            "beauty": torch.zeros(1, 2, 2, 3),
            "depth": torch.zeros(1, 2, 2, 3),
        }
        with self.assertRaisesRegex(ValueError, "lossy compression"):
            writer.write_passes(passes, "safe", compression="DWAA", output_path=tempfile.mkdtemp())

    @pytest.mark.real_torch
    @unittest.skipUnless(HAS_OPENEXR, "OpenEXR not installed")
    def test_writer_reader_roundtrip_preserves_data_contracts(self):
        with tempfile.TemporaryDirectory() as directory:
            motion = torch.zeros(1, 2, 3, 3)
            motion[..., 0] = 1.25
            motion[..., 1] = -0.5
            passes = {
                "beauty": torch.full((1, 2, 3, 3), 0.2),
                "alpha": torch.full((1, 2, 3, 3), 0.7),
                "depth": torch.full((1, 2, 3, 3), 4.0),
                "motion_vector": motion,
            }
            path = RadianceEXRPassesWriter().write_passes(
                passes, "roundtrip", bit_depth="32-bit Float", output_path=directory
            )[0]
            result = aov_reader.RadianceMultipassAOVReader().read_passes(path)
        self.assertTrue(torch.allclose(result[4], torch.full_like(result[4], 4.0)))
        self.assertTrue(torch.allclose(result[18][..., :2], motion[..., :2]))
        self.assertTrue(torch.allclose(result[-1], torch.full_like(result[-1], 0.7)))

    @pytest.mark.real_torch
    def test_reader_rejects_invalid_manual_override_and_preserves_alpha(self):
        rgba = np.ones((2, 3, 4), dtype=np.float32)
        with mock.patch.object(aov_reader, "_read_multilayer_exr", return_value=({"beauty": rgba}, 2, 3, {})):
            reader = aov_reader.RadianceMultipassAOVReader()
            with self.assertRaisesRegex(ValueError, "was not found"):
                reader.read_passes("unused.exr", beauty_layer="missing")
            result = reader.read_passes("unused.exr", beauty_layer="beauty")
        self.assertEqual(len(result), 22)
        self.assertIn("alpha", result[0]["_present"])
        self.assertTrue(torch.equal(result[-1], torch.ones(1, 2, 3, 3)))


if __name__ == "__main__":
    unittest.main()
