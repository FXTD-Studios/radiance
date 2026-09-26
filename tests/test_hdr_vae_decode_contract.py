import importlib.util
import json
import os
import sys
import types
import unittest

import torch
import pytest

# Only the tests marked @pytest.mark.real_torch below need genuine tensors; the
# rest run fine against conftest's stub. Opt out of the automatic module-level
# skip so they keep running on the no-torch CI matrix.
RADIANCE_TORCH_GATED = True



ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULE_NAMES = (
    "radiance", "radiance.hdr", "radiance.hdr.vae", "radiance.hdr.decode_meta",
    "radiance.color", "radiance.color.transfer", "radiance.color.pipeline",
)
PREVIOUS = {name: sys.modules.get(name) for name in MODULE_NAMES}


def _package(name):
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module
    return module


_package("radiance")
_package("radiance.hdr")
_package("radiance.color")


class FakeDecode:
    last_kwargs = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",), "vae": ("VAE",),
                "target_space": (["sRGB", "Linear"], {"default": "sRGB"}),
            },
            "optional": {
                "force_hdr_decode": ("BOOLEAN", {"default": False}),
                "hdr_mode": (["Clip (SDR)", "Compress (Log)"], {"default": "Clip (SDR)"}),
                "source_space": (["sRGB", "ARRI LogC4"], {"default": "sRGB"}),
                "display_tonemap": (["None", "Reinhard"], {"default": "None"}),
                "hdr_output": ("BOOLEAN", {"default": False}),
                "export_rhdr": ("BOOLEAN", {"default": False}),
                "rhdr_precision": (["f16", "f32"], {"default": "f16"}),
                "crop_padding": ("STRING", {"default": ""}),
                "processing_mode": (["sequential"], {"default": "sequential"}),
            },
        }

    def decode(self, **kwargs):
        FakeDecode.last_kwargs = kwargs
        channels = 4 if kwargs.get("alpha") is not None else 3
        image = torch.zeros(1, 2, 2, channels)
        image[0, 1, 1, :] = 5.0  # distinct value at (y=1, x=1), for crop_bbox offset tests
        metadata = {
            "resolution": "2x2",
            "latent_format": "engine_fmt",
        }
        if kwargs.get("export_rhdr"):
            metadata["rhdr_export"] = "preserved.rhdr"
        return image, json.dumps(metadata), "returned_fmt"


# 3.5.0: the fingerprint / gamut helpers are pure torch; load the real ones.
_meta_spec = importlib.util.spec_from_file_location(
    "radiance.hdr.decode_meta", os.path.join(ROOT, "hdr", "decode_meta.py"))
_meta_module = importlib.util.module_from_spec(_meta_spec)
sys.modules["radiance.hdr.decode_meta"] = _meta_module
_meta_spec.loader.exec_module(_meta_module)

vae_module = types.ModuleType("radiance.hdr.vae")
vae_module.RadianceVAE4KDecode = FakeDecode


class FakeEncode:
    """engine.py subclasses the 4K encoder as RadianceHDRVAEEncode."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}, "optional": {
            "hdr_mode": (["Clip (SDR)", "Compress (Log)"], {"default": "Clip (SDR)"})}}


vae_module.RadianceVAE4KEncode = FakeEncode
sys.modules["radiance.hdr.vae"] = vae_module


transfer_module = types.ModuleType("radiance.color.transfer")
for name in ("tensor_linear_to_logc4", "tensor_linear_to_slog3", "tensor_srgb_to_linear", "tensor_linear_to_srgb"):
    setattr(transfer_module, name, lambda value: value)
sys.modules["radiance.color.transfer"] = transfer_module

pipeline_module = types.ModuleType("radiance.color.pipeline")
pipeline_module.apply_input_transform = lambda value, _space: value
pipeline_module.apply_output_transform = lambda value, _space: value
pipeline_module.INPUT_COLORSPACES = ["Linear (sRGB)"]
sys.modules["radiance.color.pipeline"] = pipeline_module

spec = importlib.util.spec_from_file_location(
    "hdr_vae_engine_contract", os.path.join(ROOT, "nodes", "generate", "engine.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
RadianceHDRVAEDecode = module.RadianceHDRVAEDecode

for name, previous in PREVIOUS.items():
    if previous is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = previous


class TestHDRVAEDecodeContract(unittest.TestCase):
    def setUp(self):
        RadianceHDRVAEDecode._engine = FakeDecode()
        FakeDecode.last_kwargs = None
        self.samples = {"samples": torch.zeros(1, 4, 2, 2)}

    def test_ui_defaults_are_sampler_safe_and_export_is_opt_in(self):
        types_map = RadianceHDRVAEDecode.INPUT_TYPES()
        optional = types_map["optional"]
        self.assertEqual(optional["decode_mode"][1]["default"], "Auto (Recommended)")
        self.assertEqual(optional["hdr_mode"][1]["default"], "Clip (SDR)")
        self.assertEqual(optional["source_space"][1]["default"], "sRGB")
        self.assertFalse(optional["export_rhdr"][1]["default"])

    @pytest.mark.real_torch
    def test_sampler_mode_blocks_log_inversion_and_preserves_engine_metadata(self):
        result = RadianceHDRVAEDecode().apply(
            self.samples, object(), hdr_mode="Compress (Log)",
            source_space="ARRI LogC4", display_tonemap="Reinhard",
            decode_mode="Sampler (SDR-safe)", export_rhdr=True,
        )
        kwargs = FakeDecode.last_kwargs
        self.assertEqual(kwargs["hdr_mode"], "Clip (SDR)")
        self.assertEqual(kwargs["source_space"], "sRGB")
        self.assertFalse(kwargs["force_hdr_decode"])
        self.assertFalse(kwargs["export_rhdr"])
        metadata = json.loads(result["result"][1])
        self.assertEqual(metadata["resolution"], "2x2")
        self.assertNotIn("rhdr_export", metadata)
        self.assertNotIn("rudra_decoder", metadata)

    @pytest.mark.real_torch
    def test_legacy_direct_mode_name_still_selects_direct_hdr(self):
        """Graphs saved before 3.5 carry "Direct HDR / RUDRA"."""
        result = RadianceHDRVAEDecode().apply(
            self.samples, object(), decode_mode="Direct HDR / RUDRA",
        )
        meta = json.loads(result["result"][1])
        self.assertEqual(meta["decode_mode"], "Direct HDR")
        self.assertIn("hdr_path", meta)

    @pytest.mark.real_torch
    def test_direct_hdr_mode_owns_scene_linear_contract(self):
        """A log-encoded latent (HDR Encode stamps radiance_meta) is inverted."""
        alpha = torch.ones(1, 2, 2, 3)
        lat = torch.zeros(1, 4, 2, 2)
        samples = {"samples": lat,
                   "radiance_meta": {"hdr_mode": "Compress (Log)",
                                     "latent_fingerprint": _meta_module.latent_fingerprint(lat)}}
        written = {}

        def fake_write(image, precision):
            written["max"] = float(image[..., :3].max()); written["shape"] = tuple(image.shape)
            return ["preserved.rhdr"]

        orig = RadianceHDRVAEDecode.__dict__["_write_rhdr"]
        RadianceHDRVAEDecode._write_rhdr = staticmethod(fake_write)
        try:
            result = RadianceHDRVAEDecode().apply(
                samples, object(), alpha=alpha,
                decode_mode="Direct HDR", export_rhdr=True, hdr_scale_factor=2.0,
            )
        finally:
            RadianceHDRVAEDecode._write_rhdr = orig
        kwargs = FakeDecode.last_kwargs
        self.assertEqual(kwargs["hdr_mode"], "Compress (Log)")
        self.assertEqual(kwargs["source_space"], "ARRI LogC4")
        self.assertEqual(kwargs["target_space"], "Linear")
        self.assertEqual(kwargs["display_tonemap"], "None")
        self.assertTrue(kwargs["force_hdr_decode"])
        self.assertTrue(kwargs["hdr_output"])
        # 3.5.0: the wrapper writes RHDR itself, after scale and crop, so the
        # sidecar matches the IMAGE output.
        self.assertFalse(kwargs["export_rhdr"])
        self.assertEqual(written["max"], 10.0)            # 5.0 * hdr_scale_factor
        metadata = json.loads(result["result"][1])
        self.assertTrue(metadata["alpha_restored"])
        self.assertEqual(float(result["result"][0][..., 3].max()), 5.0)   # fake alpha, not scaled
        self.assertEqual(metadata["latent_format"], "returned_fmt")
        self.assertEqual(metadata["rhdr_export"], "preserved.rhdr")

    @pytest.mark.real_torch
    def test_direct_hdr_on_a_plain_latent_uses_pixel_rudra_not_log_inversion(self):
        """3.5: a sampler latent was never log-encoded. Inverting the log curve
        on it overexposes the whole frame; the latent RUDRA decoder used to
        cover this case. Now: decode SDR, then reconstruct to scene-linear
        with the RUDRA pixel model."""
        seen = {}

        def fake_pixel(sdr, peak, is_video=False):
            seen["sdr_max"] = float(sdr.max()); seen["peak"] = peak
            return sdr * 4.0, "mode: Hybrid\npath: direct-pixel (highlights)\nlearned recovery: applied", \
                "RUDRA pixel SDR->HDR (direct-pixel (highlights))"

        orig = RadianceHDRVAEDecode.__dict__["_pixel_hdr"]
        RadianceHDRVAEDecode._pixel_hdr = staticmethod(fake_pixel)
        try:
            result = RadianceHDRVAEDecode().apply(
                self.samples, object(), decode_mode="Direct HDR", hdr_peak_nits=1600.0,
            )
        finally:
            RadianceHDRVAEDecode._pixel_hdr = orig
        kwargs = FakeDecode.last_kwargs
        self.assertFalse(kwargs["force_hdr_decode"], "log inversion ran on a plain latent")
        self.assertEqual(kwargs["hdr_mode"], "Clip (SDR)")
        self.assertEqual(kwargs["target_space"], "sRGB")
        self.assertEqual(seen["peak"], 1600.0)
        image, meta = result["result"][0], json.loads(result["result"][1])
        self.assertEqual(float(image.max()), 20.0)
        self.assertTrue(meta["hdr_path"].startswith("RUDRA pixel"))
        self.assertEqual(meta["target_space"], "Linear")
        self.assertIn("learned recovery: applied", meta["sdr_to_hdr_report"])
        self.assertFalse(result["ui"]["log_overexposure_risk"][0])

    @pytest.mark.real_torch
    def test_auto_mode_recognizes_direct_hdr_encode_metadata(self):
        lat = torch.zeros(1, 4, 2, 2)
        samples = {
            "samples": lat,
            "radiance_meta": {"hdr_mode": "Compress (Log)", "source_space": "ARRI LogC4",
                              "latent_fingerprint": _meta_module.latent_fingerprint(lat)},
        }
        RadianceHDRVAEDecode().apply(samples, object())
        self.assertTrue(FakeDecode.last_kwargs["force_hdr_decode"])
        self.assertEqual(FakeDecode.last_kwargs["target_space"], "Linear")

    @pytest.mark.real_torch
    def test_auto_mode_ignores_metadata_a_sampler_carried_through(self):
        """3.5.0: KSampler copies the latent dict, so radiance_meta survives
        sampling. Auto used to log-invert the diffused latent (blown frames)."""
        encoded = torch.zeros(1, 4, 2, 2)
        sampled = torch.randn(1, 4, 2, 2)
        samples = {
            "samples": sampled,
            "radiance_meta": {"hdr_mode": "Compress (Log)", "source_space": "ARRI LogC4",
                              "pad_h": 0, "pad_w": 0,
                              "latent_fingerprint": _meta_module.latent_fingerprint(encoded)},
        }
        result = RadianceHDRVAEDecode().apply(samples, object())
        kwargs = FakeDecode.last_kwargs
        self.assertFalse(kwargs["force_hdr_decode"])
        self.assertEqual(kwargs["hdr_mode"], "Clip (SDR)")
        self.assertNotIn("hdr_mode", kwargs["samples"]["radiance_meta"])
        self.assertIn("pad_h", kwargs["samples"]["radiance_meta"])   # geometry kept
        meta = json.loads(result["result"][1])
        self.assertEqual(meta["decode_mode"], "Sampler (SDR-safe)")
        self.assertFalse(meta["radiance_meta_live"])

    @pytest.mark.real_torch
    def test_sampler_mode_honours_a_linear_target(self):
        """A visible target_space is delivered: Linear no longer needs the
        hidden hdr_output to avoid being re-encoded to sRGB."""
        RadianceHDRVAEDecode().apply(self.samples, object(), target_space="Linear",
                                     decode_mode="Sampler (SDR-safe)", hdr_output=False)
        self.assertTrue(FakeDecode.last_kwargs["hdr_output"])
        RadianceHDRVAEDecode().apply(self.samples, object(), target_space="sRGB",
                                     decode_mode="Sampler (SDR-safe)", hdr_output=True)
        self.assertFalse(FakeDecode.last_kwargs["hdr_output"])

    @pytest.mark.real_torch
    def test_crop_bbox_is_clamped_to_the_image(self):
        result = RadianceHDRVAEDecode().apply(
            self.samples, object(), crop_bbox={"x": 1, "y": 1, "width": 50, "height": 50},
        )
        self.assertEqual(tuple(result["result"][0].shape), (1, 1, 1, 3))
        result = RadianceHDRVAEDecode().apply(
            self.samples, object(), crop_bbox={"x": 9, "y": 9, "width": 4, "height": 4},
        )
        self.assertEqual(tuple(result["result"][0].shape), (1, 2, 2, 3))

    def test_invalid_latent_payload_fails_early(self):
        with self.assertRaisesRegex(RuntimeError, "4D image latent or 5D video latent"):
            RadianceHDRVAEDecode().apply({"samples": "bad"}, object())

    @pytest.mark.real_torch
    def test_av_latent_decodes_only_the_video_stream(self):
        """MiniMax H3 (and any future AV model) hands this node a nested
        video+audio latent. This node only does HDR/tonemap/RUDRA on the
        video stream, same split ComfyUI's own LTXVSeparateAVLatent uses
        (unbind()[0] = video, [1] = audio); audio is a separate node's job."""
        video = torch.zeros(1, 24, 3, 2, 2)
        audio = torch.zeros(1, 32, 2, 5)

        class FakeAVLatent:
            is_nested = True

            def unbind(self):
                return [video, audio]

        RadianceHDRVAEDecode().apply({"samples": FakeAVLatent()}, object())
        self.assertIs(FakeDecode.last_kwargs["samples"]["samples"], video)

    @pytest.mark.real_torch
    def test_crop_bbox_slices_the_decoded_image(self):
        """crop_bbox comes from RadianceResolution (offset, then size), same
        contract as ComfyUI's own BOUNDING_BOX convention."""
        result = RadianceHDRVAEDecode().apply(
            self.samples, object(), crop_bbox={"x": 1, "y": 1, "width": 1, "height": 1},
        )
        image = result["result"][0]
        self.assertEqual(tuple(image.shape), (1, 1, 1, 3))
        self.assertEqual(image[0, 0, 0, 0].item(), 5.0)

    @pytest.mark.real_torch
    def test_no_crop_bbox_leaves_the_image_untouched(self):
        result = RadianceHDRVAEDecode().apply(self.samples, object())
        self.assertEqual(tuple(result["result"][0].shape), (1, 2, 2, 3))


if __name__ == "__main__":
    unittest.main()
