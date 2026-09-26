"""
Tests for build_radiance_hdr_metadata (hdr/io.py  v2.4).

Covers:
  - Required keys always present (rad_generator, rad_hdr_source_space, …)
  - Standard EXR attrs present: software, comments
  - rad_ prefix correct on all Radiance-specific attrs
  - lora_name / lora_type embedded when provided, absent when not
  - color_pipeline summary is correctly formatted for each log profile
  - All values are strings (EXR string attributes)
  - decode_noise_scale=0.0 not shown in pipeline summary; >0 shown
  - Exposure sign (+/-) reflected in pipeline summary
  - Round-trip: JSON encode_metadata from RadianceVAE4KEncode → metadata
  - Idempotent: calling twice with same args produces identical output
    (except rad_created timestamp — excluded from comparison)
"""

import sys
import os
import json
import unittest
import types
import importlib.util

_RADIANCE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _RADIANCE_ROOT)

# ── Stub heavy dependencies so hdr/io.py can be imported without GPU ─────────

# torch
_torch_stub = types.ModuleType("torch")
_torch_stub.Tensor = object
sys.modules.setdefault("torch", _torch_stub)

# folder_paths
sys.modules.setdefault("folder_paths", types.ModuleType("folder_paths"))

# Set up a fake radiance.hdr package hierarchy so relative imports resolve.
# hdr/io.py does: from .utils import numpy_to_tensor_float32
# That resolves to  radiance.hdr.utils  if the module is loaded under that name.

_rad_pkg = types.ModuleType("radiance")
_rad_pkg.__path__ = [_RADIANCE_ROOT]
_rad_pkg.__package__ = "radiance"
sys.modules.setdefault("radiance", _rad_pkg)

_rad_hdr_pkg = types.ModuleType("radiance.hdr")
_rad_hdr_pkg.__path__ = [os.path.join(_RADIANCE_ROOT, "hdr")]
_rad_hdr_pkg.__package__ = "radiance.hdr"
sys.modules.setdefault("radiance.hdr", _rad_hdr_pkg)

_rad_hdr_utils = types.ModuleType("radiance.hdr.utils")
_rad_hdr_utils.numpy_to_tensor_float32 = lambda x: x
sys.modules.setdefault("radiance.hdr.utils", _rad_hdr_utils)

# radiance.color_utils and radiance.path_utils (tried via ..color_utils in io.py)
_cu = types.ModuleType("radiance.color_utils")
for _n in ["linear_to_logc3", "linear_to_logc4", "linear_to_slog3",
           "srgb_to_linear", "linear_to_srgb", "apply_matrix_transform",
           "AWG3_TO_ACESCG", "AWG4_TO_ACESCG", "SGAMUT3_CINE_TO_ACESCG", "ACESCG_TO_SRGB"]:
    setattr(_cu, _n, None)
sys.modules.setdefault("radiance.color_utils", _cu)

_pu = types.ModuleType("radiance.path_utils")
for _fn in ["safe_join", "get_safe_output_dir", "get_safe_input_path", "get_next_index"]:
    setattr(_pu, _fn, lambda *a, **kw: None)
sys.modules.setdefault("radiance.path_utils", _pu)

# ── Load hdr/io.py under its proper dotted name ───────────────────────────────

_io_path = os.path.join(_RADIANCE_ROOT, "hdr", "io.py")
_io_spec  = importlib.util.spec_from_file_location("radiance.hdr.io", _io_path)
_io_mod   = importlib.util.module_from_spec(_io_spec)
_io_mod.__package__ = "radiance.hdr"   # lets relative imports resolve
sys.modules["radiance.hdr.io"] = _io_mod
_io_spec.loader.exec_module(_io_mod)

build_radiance_hdr_metadata = _io_mod.build_radiance_hdr_metadata


# ═════════════════════════════════════════════════════════════════════════════
#                              TEST CASES
# ═════════════════════════════════════════════════════════════════════════════

_REQUIRED_RAD_KEYS = [
    "rad_generator",
    "rad_created",
    "rad_hdr_source_space",
    "rad_hdr_mode",
    "rad_hdr_exposure",
    "rad_hdr_noise_scale",
    "rad_hdr_target_space",
    "rad_hdr_tonemap",
    "rad_color_pipeline",
    "rad_hdr_curve",
]

_REQUIRED_STANDARD_KEYS = ["software", "comments"]


class TestBuildRadianceHDRMetadata(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    """Core invariants on build_radiance_hdr_metadata."""

    def _build(self, **kwargs):
        return build_radiance_hdr_metadata(**kwargs)

    # ── Required keys ─────────────────────────────────────────────────────────

    def test_required_rad_keys_present(self):
        """All rad_* required keys must be present in the output."""
        meta = self._build()
        for key in _REQUIRED_RAD_KEYS:
            self.assertIn(key, meta, f"Required key '{key}' missing from metadata")

    def test_standard_exr_keys_present(self):
        """software and comments standard EXR attributes must be present."""
        meta = self._build()
        for key in _REQUIRED_STANDARD_KEYS:
            self.assertIn(key, meta, f"Standard EXR key '{key}' missing")

    def test_all_values_are_strings(self):
        """Every value must be a str (EXR string attributes are plain strings)."""
        meta = self._build(
            source_space="ARRI LogC3",
            hdr_mode="Compress (Log)",
            decode_noise_scale=0.025,
            exposure=0.5,
            target_space="ACEScg",
            display_tonemap="None",
            lora_name="LTX-HDR",
        )
        for k, v in meta.items():
            self.assertIsInstance(v, str, f"Value for '{k}' is {type(v).__name__}, expected str")

    # ── rad_ prefix rules ─────────────────────────────────────────────────────

    def test_radiance_attrs_use_rad_prefix(self):
        """All Radiance-specific attrs (not 'software'/'comments') must start with rad_."""
        meta = self._build()
        for k in meta:
            if k not in ("software", "comments"):
                self.assertTrue(
                    k.startswith("rad_"),
                    f"Key '{k}' is not a standard EXR attr but lacks 'rad_' prefix",
                )

    # ── lora_name ─────────────────────────────────────────────────────────────

    def test_lora_name_embedded_when_provided(self):
        """rad_lora_name and rad_lora_type present when lora_name is given."""
        meta = self._build(lora_name="LTX-2.3-22b-IC-LoRA-HDR", lora_type="IC-LoRA")
        self.assertIn("rad_lora_name", meta)
        self.assertIn("rad_lora_type", meta)
        self.assertEqual(meta["rad_lora_name"], "LTX-2.3-22b-IC-LoRA-HDR")
        self.assertEqual(meta["rad_lora_type"], "IC-LoRA")

    def test_lora_name_absent_when_empty(self):
        """rad_lora_name must NOT be present when lora_name=''."""
        meta = self._build(lora_name="")
        self.assertNotIn("rad_lora_name", meta)
        self.assertNotIn("rad_lora_type", meta)

    # ── Source space ──────────────────────────────────────────────────────────

    def test_source_space_embedded(self):
        for space in ["ARRI LogC3", "ARRI LogC4", "Sony S-Log3", "RED Log3G10", "Linear"]:
            with self.subTest(space=space):
                meta = self._build(source_space=space)
                self.assertEqual(meta["rad_hdr_source_space"], space)

    def test_unknown_source_space_falls_back(self):
        """Unknown source space is stored verbatim without crashing."""
        meta = self._build(source_space="CustomSpace XYZ")
        self.assertEqual(meta["rad_hdr_source_space"], "CustomSpace XYZ")

    # ── hdr_mode ─────────────────────────────────────────────────────────────

    def test_hdr_mode_stored(self):
        for mode in ["Compress (Log)", "Passthrough", "Clip", "Soft Clip"]:
            with self.subTest(mode=mode):
                meta = self._build(hdr_mode=mode)
                self.assertEqual(meta["rad_hdr_mode"], mode)

    # ── Noise scale ──────────────────────────────────────────────────────────

    def test_noise_scale_stored_as_string(self):
        meta = self._build(decode_noise_scale=0.025)
        self.assertEqual(meta["rad_hdr_noise_scale"], "0.025000")

    def test_zero_noise_scale_stored(self):
        meta = self._build(decode_noise_scale=0.0)
        self.assertEqual(meta["rad_hdr_noise_scale"], "0.000000")

    def test_nonzero_noise_in_pipeline_summary(self):
        """Non-zero noise_scale should appear in color_pipeline summary."""
        meta = self._build(decode_noise_scale=0.025)
        self.assertIn("noise=", meta["rad_color_pipeline"])

    def test_zero_noise_not_in_pipeline_summary(self):
        """Zero noise_scale should NOT clutter the pipeline summary."""
        meta = self._build(decode_noise_scale=0.0)
        self.assertNotIn("noise=", meta["rad_color_pipeline"])

    # ── Exposure ─────────────────────────────────────────────────────────────

    def test_positive_exposure_in_pipeline_summary(self):
        meta = self._build(exposure=1.0)
        self.assertIn("EV+", meta["rad_color_pipeline"])

    def test_negative_exposure_in_pipeline_summary(self):
        meta = self._build(exposure=-2.0)
        self.assertIn("EV-", meta["rad_color_pipeline"])

    def test_zero_exposure_not_in_pipeline_summary(self):
        """Zero exposure should not add EV noise to the pipeline summary."""
        meta = self._build(exposure=0.0)
        self.assertNotIn("EV", meta["rad_color_pipeline"])

    # ── Target space in pipeline ──────────────────────────────────────────────

    def test_target_space_in_pipeline_summary(self):
        meta = self._build(target_space="ACEScg")
        self.assertIn("ACEScg", meta["rad_color_pipeline"])

    # ── Tonemap ──────────────────────────────────────────────────────────────

    def test_tonemap_none_not_in_summary(self):
        """display_tonemap='None' should not append a [TM:None] suffix."""
        meta = self._build(display_tonemap="None")
        self.assertNotIn("[TM:", meta["rad_color_pipeline"])

    def test_tonemap_reinhard_in_summary(self):
        meta = self._build(display_tonemap="Reinhard")
        self.assertIn("[TM:Reinhard]", meta["rad_color_pipeline"])

    # ── software / comments ──────────────────────────────────────────────────

    def test_software_contains_radiance(self):
        meta = self._build()
        self.assertIn("Radiance", meta["software"])

    def test_comments_contains_pipeline(self):
        """comments field must include the pipeline summary for generic viewers."""
        meta = self._build(source_space="ARRI LogC3", target_space="ACEScg")
        self.assertIn("ACEScg", meta["comments"])

    # ── Idempotency (sans timestamp) ──────────────────────────────────────────

    def test_idempotent_output(self):
        """Two calls with same args produce identical output (except rad_created)."""
        kwargs = dict(
            source_space="ARRI LogC3",
            hdr_mode="Compress (Log)",
            decode_noise_scale=0.025,
            exposure=-0.5,
            target_space="ACEScg",
            display_tonemap="None",
            lora_name="test-lora",
        )
        m1 = self._build(**kwargs)
        m2 = self._build(**kwargs)
        for k in m1:
            if k == "rad_created":
                continue
            self.assertEqual(m1[k], m2[k], f"Key '{k}' differs between two calls")


class TestEncodeMetadataRoundtrip(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    """
    Simulate the RadianceVAE4KEncode → RadianceWrite pipeline:
    encode_metadata JSON is parsed and fed into build_radiance_hdr_metadata.
    """

    _SAMPLE_ENCODE_META = json.dumps({
        "node": "RadianceVAE4KEncode",
        "resolution": "1920×1080",
        "source_space": "ARRI LogC3",
        "hdr_mode": "Compress (Log)",
        "exposure": -0.5,
        "pad_h": 0,
        "pad_w": 0,
        "vae_factor": 8,
        "latent_format": "video",
        "latent_sampling": "normal",
        "decode_noise_scale": 0.025,
    })

    def _parse_and_build(self, encode_metadata_json: str, **overrides):
        enc = json.loads(encode_metadata_json)
        return build_radiance_hdr_metadata(
            source_space       = enc.get("source_space", "Linear"),
            hdr_mode           = enc.get("hdr_mode", "Passthrough"),
            decode_noise_scale = float(enc.get("decode_noise_scale", 0.0)),
            exposure           = float(enc.get("exposure", 0.0)),
            **overrides,
        )

    def test_source_space_round_trips(self):
        meta = self._parse_and_build(self._SAMPLE_ENCODE_META)
        self.assertEqual(meta["rad_hdr_source_space"], "ARRI LogC3")

    def test_hdr_mode_round_trips(self):
        meta = self._parse_and_build(self._SAMPLE_ENCODE_META)
        self.assertEqual(meta["rad_hdr_mode"], "Compress (Log)")

    def test_noise_scale_round_trips(self):
        meta = self._parse_and_build(self._SAMPLE_ENCODE_META)
        self.assertEqual(meta["rad_hdr_noise_scale"], "0.025000")

    def test_exposure_round_trips(self):
        # rad_hdr_exposure stores the raw signed float string (e.g. "-0.5000")
        # The "EV" prefix only appears in the pipeline summary, not in this attribute.
        meta = self._parse_and_build(self._SAMPLE_ENCODE_META)
        self.assertIn("-0.5", meta["rad_hdr_exposure"])

    def test_lora_name_injected_at_write_time(self):
        """lora_name comes from the write node (not encode), injected separately."""
        meta = self._parse_and_build(
            self._SAMPLE_ENCODE_META, lora_name="LTX-2.3-22b-IC-LoRA-HDR"
        )
        self.assertEqual(meta["rad_lora_name"], "LTX-2.3-22b-IC-LoRA-HDR")

    def test_empty_encode_metadata_does_not_crash(self):
        """Missing encode_metadata must produce valid baseline metadata."""
        meta = build_radiance_hdr_metadata()
        self.assertIn("software", meta)
        self.assertIn("rad_generator", meta)

    def test_partial_encode_meta_missing_keys_use_defaults(self):
        """Partial encode_metadata (missing decode_noise_scale) defaults to 0."""
        enc_json = json.dumps({"source_space": "Sony S-Log3", "hdr_mode": "Clip"})
        meta = self._parse_and_build(enc_json)
        self.assertEqual(meta["rad_hdr_source_space"], "Sony S-Log3")
        self.assertEqual(meta["rad_hdr_noise_scale"], "0.000000")


class TestLogProfileCurveDescriptions(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    """Verify that the known log profiles produce sensible curve descriptions."""

    def test_arri_logc3_description(self):
        meta = build_radiance_hdr_metadata(source_space="ARRI LogC3")
        self.assertIn("LogC3", meta["rad_hdr_curve"])

    def test_red_log3g10_description(self):
        meta = build_radiance_hdr_metadata(source_space="RED Log3G10")
        self.assertIn("Log3G10", meta["rad_hdr_curve"])

    def test_linear_description(self):
        meta = build_radiance_hdr_metadata(source_space="Linear")
        self.assertIn("Scene-Linear", meta["rad_hdr_curve"])

    def test_unknown_space_uses_verbatim(self):
        meta = build_radiance_hdr_metadata(source_space="Mystery Curve v9")
        self.assertEqual(meta["rad_hdr_curve"], "Mystery Curve v9")


if __name__ == "__main__":
    unittest.main()
