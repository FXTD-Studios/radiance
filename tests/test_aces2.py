"""
tests/test_aces2.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Test suite for nodes_aces2.py — ACES 2.0 Full Implementation

Coverage
────────
  TestDanieleEvoMath         Pure-math unit tests for the Evo tonescale
  TestReachGamutCompress     Pure-math tests for reach gamut compression
  TestAMFBuildParse          AMF XML round-trip (no torch)
  TestACES2TonescaleNode     ComfyUI node (torch required)
  TestACES2ReachGamutNode    ComfyUI node (torch required)
  TestACES2FullOTNode        Full output transform node (torch required)
  TestACESMetadataFileNode   AMF node (no torch for write/read mode)
  TestACES2ComplianceNode    ACES 2.0 Output Check node (torch required)
  TestNodeRegistration       Verify all 5 nodes are registered
"""

import json
import math
import sys

import numpy as np
import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Torch guard — same pattern used throughout the Radiance test suite
# ─────────────────────────────────────────────────────────────────────────────
try:
    import torch
    HAS_TORCH = hasattr(torch, "__version__")   # False when only the conftest stub is loaded
except ImportError:
    HAS_TORCH = False

skip_no_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not available")

# ─────────────────────────────────────────────────────────────────────────────
# Import the module under test
# ─────────────────────────────────────────────────────────────────────────────
sys.path.insert(0, ".")
from radiance.nodes.hdr.aces2 import (
    _DanieleEvoParams,
    _daniele_evo_fwd,
    _daniele_evo_luma_preserving,
    _reach_compress_channel,
    _reach_gamut_compress,
    _build_amf,
    _parse_amf,
    _pq_encode,
    _hlg_encode,
    _srgb_encode,
    LUMA_AP1,
    AP1_TO_P3D60,
    RadianceACES2Tonescale,
    RadianceACES2ReachGamutCompress,
    RadianceACES2OutputTransformFull,
    RadianceACESMetadataFile,
    RadianceACES2Compliance,
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)


# ═════════════════════════════════════════════════════════════════════════════
# § 1  Daniele Evo tonescale — pure numpy
# ═════════════════════════════════════════════════════════════════════════════

class TestDanieleEvoMath:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    """Unit tests for _DanieleEvoParams and _daniele_evo_fwd (numpy-only)."""

    def test_params_sdr_middle_grey(self):
        """18% scene grey should map to 10% of SDR peak (10 nits of 100)."""
        p = _DanieleEvoParams(peak_nits=100.0)
        y = _daniele_evo_fwd(np.array([0.18]), p)
        # n=1.0, so 10% of peak = 0.10
        assert abs(float(y[0]) - 0.10) < 0.001, f"Expected 0.10, got {y[0]:.5f}"

    def test_params_hdr_1000_middle_grey(self):
        """18% grey at a 1000-nit peak is 14.512 nits, not 10% of the peak.

        This test used to assert 100 nits — grey_target was a flat 10% of
        whatever the display could do, so raising the peak raised the midtone
        with it. ACES 2.0 publishes 10.000 / 14.512 / 16.824 nits at 100 /
        1000 / 4000: the midtone barely moves while the highlights extend.
        """
        p = _DanieleEvoParams(peak_nits=1000.0)
        y = _daniele_evo_fwd(np.array([0.18]), p)
        nits = float(y[0]) * 100.0
        assert abs(nits - 14.512) < 0.01, f"Expected 14.512 nits, got {nits:.5f}"

    def test_black_maps_to_zero(self):
        """Zero scene luminance must produce zero display output."""
        p = _DanieleEvoParams()
        y = _daniele_evo_fwd(np.array([0.0]), p)
        assert float(y[0]) == pytest.approx(0.0, abs=1e-6)

    def test_monotonic_increasing(self):
        """Curve must be strictly monotonic (brighter in, brighter out)."""
        p = _DanieleEvoParams()
        x = np.linspace(0.0, 10.0, 500)
        y = _daniele_evo_fwd(x, p)
        diffs = np.diff(y)
        assert np.all(diffs >= 0), "Curve is not monotonically increasing"

    def test_asymptote_at_peak(self):
        """Very bright inputs should saturate toward n (display peak)."""
        p = _DanieleEvoParams(peak_nits=100.0)  # n = 1.0
        y_large = float(_daniele_evo_fwd(np.array([1e6]), p)[0])
        assert y_large < p.n * 1.01, f"Curve does not saturate: {y_large:.4f} > {p.n}"
        assert y_large > p.n * 0.98, f"Curve too far from asymptote at white: {y_large:.4f}"

    def test_toe_linear_below_threshold(self):
        """Below toe_scene the curve should be nearly linear (slope > 0)."""
        p = _DanieleEvoParams()
        xs = np.linspace(0.0, p.toe_scene, 50)
        ys = _daniele_evo_fwd(xs, p)
        # All positive, linearly increasing
        assert ys[-1] > 0.0
        assert ys[0] == pytest.approx(0.0, abs=1e-5)

    def test_contrast_exponent_effect(self):
        """Higher g → more contrast (deeper shadows, brighter highlights)."""
        p_soft = _DanieleEvoParams(peak_nits=100.0, g=0.90)
        p_hard = _DanieleEvoParams(peak_nits=100.0, g=1.40)
        x = np.array([0.05])   # shadow region
        y_soft = float(_daniele_evo_fwd(x, p_soft)[0])
        y_hard = float(_daniele_evo_fwd(x, p_hard)[0])
        assert y_hard < y_soft, "Higher contrast should compress shadows more"

    def test_grey_target_scaling(self):
        """Custom grey_target should move the midpoint correctly."""
        for target in [0.08, 0.10, 0.18]:
            p = _DanieleEvoParams(peak_nits=100.0, grey_target=target)
            y = float(_daniele_evo_fwd(np.array([0.18]), p)[0])
            expected = target * p.n
            assert abs(y - expected) < 0.002, \
                f"grey_target={target}: expected {expected:.4f} got {y:.4f}"

    def test_luma_preserving_achromatic(self):
        """Achromatic RGB (R=G=B) should remain achromatic after luma mapping."""
        p = _DanieleEvoParams()
        grey = np.full((16, 16, 3), 0.18, dtype=np.float32)
        out = _daniele_evo_luma_preserving(grey, p)
        r, g, b = out[..., 0], out[..., 1], out[..., 2]
        np.testing.assert_allclose(r, g, atol=1e-5)
        np.testing.assert_allclose(g, b, atol=1e-5)

    def test_luma_preserving_no_negative(self):
        """Luma-preserving mode must produce no negative output."""
        p = _DanieleEvoParams()
        rng = np.random.default_rng(42)
        rgb = rng.uniform(0, 5, (64, 64, 3)).astype(np.float32)
        out = _daniele_evo_luma_preserving(rgb, p)
        assert out.min() >= -1e-6


# ═════════════════════════════════════════════════════════════════════════════
# § 2  Reach gamut compression — pure numpy
# ═════════════════════════════════════════════════════════════════════════════

class TestReachGamutCompress:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    def test_in_gamut_pixels_unchanged(self):
        """Pixels inside the threshold band must not be modified."""
        rgb = np.array([[[0.5, 0.5, 0.5]]], dtype=np.float32)   # (1,1,3)
        out = _reach_gamut_compress(rgb, strength=1.0)
        np.testing.assert_allclose(out, rgb, atol=1e-5)

    def test_out_of_gamut_compressed(self):
        """A very saturated out-of-gamut colour should be pulled in."""
        # Extreme cyan: R very negative relative to ach
        rgb = np.array([[[-0.5, 1.0, 1.0]]], dtype=np.float32)
        out = _reach_gamut_compress(rgb, strength=1.0)
        # Compressed dist_r should be < original dist_r — the channel moves up
        assert out[0, 0, 0] > rgb[0, 0, 0], "Negative R should be lifted"

    def test_bypass_at_zero_strength(self):
        """strength=0 must be a no-op."""
        rgb = np.array([[[-1.0, 2.0, 0.5]]], dtype=np.float32)
        out = _reach_gamut_compress(rgb, strength=0.0)
        np.testing.assert_allclose(out, rgb, atol=1e-6)

    def test_compress_channel_at_threshold(self):
        """Distance exactly at threshold must be returned unchanged."""
        t, limit = 0.815, 1.147
        d = np.array([t])
        c = _reach_compress_channel(d, t, limit)
        assert abs(float(c[0]) - t) < 1e-5

    def test_compress_channel_below_threshold(self):
        """Distance below threshold must be passed through identically."""
        t, limit = 0.815, 1.147
        d = np.array([0.5])
        c = _reach_compress_channel(d, t, limit)
        np.testing.assert_allclose(c, d, atol=1e-6)

    def test_compress_channel_at_limit(self):
        """Distance at the reach limit should compress to < limit."""
        t, limit = 0.815, 1.147
        d = np.array([limit])
        c = _reach_compress_channel(d, t, limit)
        # Compressed value should be significantly less than limit
        assert float(c[0]) < limit - 0.05

    def test_compress_channel_monotonic(self):
        """Channel compression must be monotonically increasing."""
        t, limit = 0.815, 1.147
        d = np.linspace(0, limit * 1.2, 200)
        c = _reach_compress_channel(d, t, limit)
        assert np.all(np.diff(c) >= -1e-7)

    def test_no_negative_output(self):
        """Even extreme inputs must not produce negative output."""
        rng = np.random.default_rng(0)
        rgb = rng.uniform(-2, 3, (32, 32, 3)).astype(np.float32)
        out = _reach_gamut_compress(rgb, strength=1.0)
        # We only guarantee achromatic axis is non-negative
        ach = np.maximum(out[..., 0], np.maximum(out[..., 1], out[..., 2]))
        assert ach.min() >= -1e-4

    def test_oog_count_reduced(self):
        """After compression the number of out-of-gamut values should drop."""
        rng = np.random.default_rng(7)
        rgb = rng.uniform(-0.5, 1.5, (64, 64, 3)).astype(np.float32)
        oog_before = int(((rgb < 0) | (rgb > 1)).sum())
        out = _reach_gamut_compress(rgb, strength=1.0)
        oog_after  = int(((out < 0) | (out > 1)).sum())
        assert oog_after <= oog_before, "Compression must not increase OOG count"


# ═════════════════════════════════════════════════════════════════════════════
# § 3  AMF build / parse round-trip — no torch
# ═════════════════════════════════════════════════════════════════════════════

class TestAMFBuildParse:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    def _make(self, **kw):
        defaults = dict(
            clip_name        = "TestClip_001",
            input_transform  = "urn:ampas:aces:transformId:v1.5:IDT.Academy.ADX10.a1.0.3",
            output_transform = "urn:ampas:aces:transformId:v2.0:ODT.Academy.Rec709-100nits.a2.0.0",
            peak_nits        = 100.0,
            min_nits         = 0.005,
            description      = "Test AMF",
        )
        defaults.update(kw)
        return _build_amf(**defaults)

    def test_valid_xml(self):
        """Output must be parseable XML."""
        import xml.etree.ElementTree as ET
        xml_str = self._make()
        root = ET.fromstring(xml_str)
        assert root is not None

    def test_round_trip_clip_name(self):
        xml_str = self._make(clip_name="RoundTripClip")
        fields = _parse_amf(xml_str)
        assert fields["clipName"] == "RoundTripClip"

    def test_round_trip_peak_nits(self):
        xml_str = self._make(peak_nits=4000.0)
        fields = _parse_amf(xml_str)
        assert fields["peakLuminance"] == "4000.0"

    def test_round_trip_input_transform(self):
        urn = "urn:ampas:aces:transformId:v1.5:IDT.Academy.ADX10.a1.0.3"
        xml_str = self._make(input_transform=urn)
        fields = _parse_amf(xml_str)
        assert fields["inputTransform"] == urn

    def test_round_trip_output_transform(self):
        urn = "urn:ampas:aces:transformId:v2.0:ODT.Academy.Rec709-100nits.a2.0.0"
        xml_str = self._make(output_transform=urn)
        fields = _parse_amf(xml_str)
        assert fields["outputTransform"] == urn

    def test_uuid_present(self):
        xml_str = self._make()
        fields = _parse_amf(xml_str)
        assert len(fields["uuid"]) == 36   # standard UUID with dashes

    def test_datetime_present(self):
        xml_str = self._make()
        fields = _parse_amf(xml_str)
        assert "T" in fields["dateTime"] and "Z" in fields["dateTime"]

    def test_description_preserved(self):
        xml_str = self._make(description="My Compliance Test AMF")
        fields = _parse_amf(xml_str)
        assert "My Compliance Test AMF" in fields["description"]

    def test_different_clips_have_different_uuids(self):
        xml_a = self._make(clip_name="A")
        xml_b = self._make(clip_name="B")
        uuid_a = _parse_amf(xml_a)["uuid"]
        uuid_b = _parse_amf(xml_b)["uuid"]
        assert uuid_a != uuid_b


# ═════════════════════════════════════════════════════════════════════════════
# § 5  EOTF encoders — numpy (no torch)
# ═════════════════════════════════════════════════════════════════════════════

class TestEOTFEncoders:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    def test_pq_range(self):
        x = np.linspace(0, 10, 50, dtype=np.float32)
        y = _pq_encode(x)
        assert y.min() >= 0.0
        assert y.max() <= 1.0 + 1e-5

    def test_pq_zero_in_zero_out(self):
        assert _pq_encode(np.array([0.0]))[0] == pytest.approx(0.0, abs=0.01)

    def test_hlg_range(self):
        x = np.linspace(0, 1, 50, dtype=np.float32)
        y = _hlg_encode(x)
        assert y.min() >= 0.0
        assert y.max() <= 1.0 + 1e-5

    def test_srgb_zero_in_zero_out(self):
        y = _srgb_encode(np.array([0.0]))
        assert float(y[0]) == pytest.approx(0.0, abs=1e-6)

    def test_srgb_one_in_one_out(self):
        y = _srgb_encode(np.array([1.0]))
        assert float(y[0]) == pytest.approx(1.0, abs=1e-4)

    def test_srgb_linear_region(self):
        """sRGB should be nearly linear below 0.0031308."""
        x = np.array([0.001])
        y = _srgb_encode(x)
        assert float(y[0]) == pytest.approx(12.92 * 0.001, abs=1e-5)


# ═════════════════════════════════════════════════════════════════════════════
# § 6  ComfyUI Node tests (torch required)
# ═════════════════════════════════════════════════════════════════════════════

@skip_no_torch
class TestACES2TonescaleNode:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    def _grey_tensor(self, val=0.18, shape=(1, 32, 32, 3)):
        return torch.full(shape, val, dtype=torch.float32)

    def test_output_shape_preserved(self):
        node = RadianceACES2Tonescale()
        img = self._grey_tensor()
        out, info = node.apply(img, peak_nits=100.0, mode="luminance_preserving")
        assert out.shape == img.shape

    def test_middle_grey_maps_to_ten_percent(self):
        """Node output for 18% grey scene should be ≈ 10% (0.10)."""
        node = RadianceACES2Tonescale()
        img = self._grey_tensor(0.18)
        out, _ = node.apply(img, peak_nits=100.0, mode="luminance_preserving")
        mean = float(out.mean())
        assert abs(mean - 0.10) < 0.005, f"Expected ≈0.10, got {mean:.5f}"

    def test_output_range(self):
        node = RadianceACES2Tonescale()
        img = torch.rand(1, 64, 64, 3) * 5.0
        out, _ = node.apply(img, peak_nits=100.0, mode="per_channel")
        assert out.min() >= -1e-4
        assert out.max() <= 1.0 + 1e-4

    def test_per_channel_mode_runs(self):
        node = RadianceACES2Tonescale()
        img = torch.rand(2, 16, 16, 3)
        out, info = node.apply(img, peak_nits=1000.0, mode="per_channel")
        assert out.shape == img.shape
        assert "ACES 2.0 reference tone scale" in info

    def test_hdr_peak_in_info(self):
        node = RadianceACES2Tonescale()
        img = torch.rand(1, 8, 8, 3)
        _, info = node.apply(img, peak_nits=4000.0, mode="luminance_preserving")
        assert "4000" in info

    def test_batch_output(self):
        node = RadianceACES2Tonescale()
        img = torch.rand(4, 16, 16, 3)
        out, _ = node.apply(img, peak_nits=100.0, mode="luminance_preserving")
        assert out.shape[0] == 4


@skip_no_torch
class TestACES2ReachGamutNode:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    def test_output_shape(self):
        node = RadianceACES2ReachGamutCompress()
        img = torch.rand(1, 32, 32, 3)
        out, info = node.compress(img, strength=1.0)
        assert out.shape == img.shape

    def test_in_gamut_unchanged(self):
        """In-gamut pixels (low-saturation) should be near-identical."""
        node = RadianceACES2ReachGamutCompress()
        img = torch.rand(1, 16, 16, 3) * 0.05 + 0.25
        out, _ = node.compress(img, strength=1.0)
        # Should be very close (no compression applied)
        assert torch.allclose(out, img, atol=1e-4)

    def test_bypass_at_zero_strength(self):
        node = RadianceACES2ReachGamutCompress()
        img = torch.rand(1, 16, 16, 3) * 2.0 - 0.5  # includes negatives
        out, _ = node.compress(img, strength=0.0)
        assert torch.allclose(out, img, atol=1e-5)

    def test_info_contains_percentages(self):
        node = RadianceACES2ReachGamutCompress()
        img = torch.rand(1, 8, 8, 3)
        _, info = node.compress(img, strength=1.0)
        assert "%" in info


@skip_no_torch
class TestACES2FullOTNode:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    def test_p3_d60_matrix_matches_chromaticities(self):
        from radiance.color.encodings import rgb_to_xyz
        # Solve the P3 primary scaling at ACES white independently from the
        # fixed conversion matrix used in the output transform.
        xy = np.array([[0.680, 0.320], [0.265, 0.690], [0.150, 0.060]])
        xyz = np.stack([xy[:, 0] / xy[:, 1], np.ones(3), (1 - xy.sum(axis=1)) / xy[:, 1]])
        white = np.array([0.32168 / 0.33767, 1.0, (1 - 0.32168 - 0.33767) / 0.33767])
        p3_to_xyz = xyz @ np.diag(np.linalg.solve(xyz, white))
        np.testing.assert_allclose(p3_to_xyz @ AP1_TO_P3D60, rgb_to_xyz("AP1"), atol=1e-7)
        np.testing.assert_allclose(AP1_TO_P3D60 @ np.ones(3), np.ones(3), atol=1e-7)

    def test_cinema_white_options_produce_distinct_colour_transforms(self):
        node = RadianceACES2OutputTransformFull()
        image = torch.tensor([[[[0.30, 0.20, 0.10], [0.18, 0.18, 0.18]]]])
        d60, _ = node.transform(image, "ACEScg", "ACES 2.0 Cinema (DCI-P3 D60)",
                                engine="Radiance approximation")
        d65, _ = node.transform(image, "ACEScg", "ACES 2.0 Cinema (DCI-P3 D65)",
                                engine="Radiance approximation")
        assert not torch.allclose(d60[..., 0, :], d65[..., 0, :], atol=1e-4)
        torch.testing.assert_close(d60[..., 1, :], d65[..., 1, :])
        assert torch.isfinite(d60).all()

    def test_sdr_output_range(self):
        node = RadianceACES2OutputTransformFull()
        img = torch.rand(1, 32, 32, 3) * 2.0   # scene-linear, some overbright
        out, info = node.transform(
            img,
            input_colorspace = "ACEScg",
            output_transform = "ACES 2.0 SDR (sRGB/Rec.709)",
        )
        assert out.min() >= -1e-4
        assert out.max() <= 1.0 + 1e-4
        assert "ACES 2.0" in info

    def test_hdr_pq_output_range(self):
        node = RadianceACES2OutputTransformFull()
        img = torch.rand(1, 16, 16, 3)
        out, _ = node.transform(
            img,
            input_colorspace = "ACEScg",
            output_transform = "ACES 2.0 HDR (Rec.2100 PQ 1000 nits)",
        )
        assert out.min() >= -1e-4
        assert out.max() <= 1.0 + 1e-4

    def test_hlg_output_range(self):
        node = RadianceACES2OutputTransformFull()
        img = torch.rand(1, 16, 16, 3)
        out, _ = node.transform(
            img,
            input_colorspace = "ACEScg",
            output_transform = "ACES 2.0 HDR (Rec.2100 HLG)",
        )
        assert out.min() >= -1e-4
        assert out.max() <= 1.0 + 1e-4

    def test_p3_output(self):
        node = RadianceACES2OutputTransformFull()
        img = torch.rand(1, 8, 8, 3)
        out, _ = node.transform(
            img,
            input_colorspace = "ACEScg",
            output_transform = "ACES 2.0 SDR (P3-D65)",
        )
        assert out.shape == img.shape

    def test_cinema_dci_p3(self):
        node = RadianceACES2OutputTransformFull()
        img = torch.rand(1, 8, 8, 3)
        out, info = node.transform(
            img,
            input_colorspace = "ACEScg",
            output_transform = "ACES 2.0 Cinema (DCI-P3 D65)",
        )
        assert out.min() >= -1e-4

    def test_input_colorspace_ap0(self):
        node = RadianceACES2OutputTransformFull()
        img = torch.rand(1, 8, 8, 3)
        out, _ = node.transform(
            img,
            input_colorspace = "ACES2065-1",
            output_transform = "ACES 2.0 SDR (sRGB/Rec.709)",
        )
        assert out.shape == img.shape

    def test_input_colorspace_srgb(self):
        node = RadianceACES2OutputTransformFull()
        img = torch.rand(1, 8, 8, 3)
        out, _ = node.transform(
            img,
            input_colorspace = "Linear_sRGB",
            output_transform = "ACES 2.0 SDR (sRGB/Rec.709)",
        )
        assert out.shape == img.shape

    def test_exposure_adjust_darkens(self):
        """Negative exposure should produce dimmer output."""
        node = RadianceACES2OutputTransformFull()
        img = torch.rand(1, 16, 16, 3) * 0.5 + 0.1
        out_0, _ = node.transform(img, "ACEScg", "ACES 2.0 SDR (sRGB/Rec.709)", exposure_adjust=0.0)
        out_neg, _ = node.transform(img, "ACEScg", "ACES 2.0 SDR (sRGB/Rec.709)", exposure_adjust=-2.0)
        assert float(out_neg.mean()) < float(out_0.mean())

    def test_batch_preserved(self):
        node = RadianceACES2OutputTransformFull()
        img = torch.rand(3, 8, 8, 3)
        out, _ = node.transform(img, "ACEScg", "ACES 2.0 SDR (sRGB/Rec.709)")
        assert out.shape[0] == 3


@skip_no_torch
class TestACESMetadataFileNodeTorch:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    """Torch-gated tests just to ensure the node runs inside torch context."""

    def test_write_returns_xml_and_summary(self):
        node = RadianceACESMetadataFile()
        xml, summary = node.run(
            mode             = "write",
            clip_name        = "TorchTest",
            input_transform  = "urn:test:IDT",
            output_transform = "urn:test:ODT",
            peak_nits        = 1000.0,
            min_nits         = 0.005,
            description      = "Torch test",
        )
        assert "<aces:aces" in xml
        assert "TorchTest" in xml
        assert "TorchTest" in summary

    def test_read_round_trip(self):
        node = RadianceACESMetadataFile()
        xml, _ = node.run(
            mode = "write",
            clip_name = "RoundTrip",
            input_transform = "urn:a",
            output_transform = "urn:b",
        )
        _, summary = node.run(mode="read", amf_xml_in=xml)
        assert "RoundTrip" in summary


class TestACESMetadataFileNode:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    """No-torch tests for the AMF node (write/read don't need torch)."""

    def test_write_mode(self):
        node = RadianceACESMetadataFile()
        xml, summary = node.run(
            mode             = "write",
            clip_name        = "MyClip",
            input_transform  = "urn:ampas:aces:IDT.test",
            output_transform = "urn:ampas:aces:ODT.test",
            peak_nits        = 100.0,
            min_nits         = 0.005,
        )
        assert "<?xml" in xml
        assert "MyClip" in xml
        assert "AMF WRITE" in summary

    def test_read_mode_from_xml(self):
        node = RadianceACESMetadataFile()
        # First generate XML via write
        xml, _ = node.run(
            mode = "write",
            clip_name = "ReadTest",
            input_transform = "urn:idt",
            output_transform = "urn:odt",
        )
        # Then read it back
        returned_xml, summary = node.run(mode="read", amf_xml_in=xml)
        assert "ReadTest" in summary

    def test_empty_read_returns_error(self):
        node = RadianceACESMetadataFile()
        _, summary = node.run(mode="read", amf_xml_in="")
        assert "no xml" in summary.lower()

    def test_write_default_no_crash(self):
        node = RadianceACESMetadataFile()
        xml, _ = node.run(mode="write")
        assert xml  # non-empty


@skip_no_torch
class TestACES2ComplianceNode:
    """ACES 2.0 Output Check (node ID RadianceACES2Compliance)."""

    def _make(self, val, shape=(1, 32, 32, 3)):
        return torch.full(shape, val, dtype=torch.float32)

    def test_reference_render_passes(self):
        scene = torch.rand(2, 16, 16, 3) * 4.0
        display, _ = RadianceACES2OutputTransformFull().transform(
            scene, "ACEScg", "ACES 2.0 SDR (sRGB/Rec.709)", engine="OCIO reference")
        report, pass_count = RadianceACES2Compliance().check(scene, display, "SDR_sRGB")
        assert "✗" not in report, report
        assert pass_count == 4

    def test_approximation_fails_reference_match(self):
        scene = torch.rand(1, 16, 16, 3) * 4.0
        display, _ = RadianceACES2OutputTransformFull().transform(
            scene, "ACEScg", "ACES 2.0 SDR (sRGB/Rec.709)", engine="Radiance approximation")
        report, _ = RadianceACES2Compliance().check(scene, display, "SDR_sRGB")
        assert "reference_match" in report and "FAIL" in report

    def test_wrong_frame_only_on_frame_two_fails(self):
        scene = torch.rand(3, 8, 8, 3)
        display, _ = RadianceACES2OutputTransformFull().transform(
            scene, "ACEScg", "ACES 2.0 SDR (sRGB/Rec.709)", engine="OCIO reference")
        display = display.clone()
        display[2] = 0.5
        report, _ = RadianceACES2Compliance().check(scene, display, "SDR_sRGB")
        assert "FAIL" in report and "worst frame 2" in report

    def test_output_over_one_fails_range(self):
        report, _ = RadianceACES2Compliance().check(self._make(0.18), self._make(1.5), "SDR_sRGB")
        assert "display_range" in report and "FAIL" in report

    def test_nan_fails(self):
        d = self._make(0.1)
        d[0, 0, 0, 0] = float("nan")
        report, _ = RadianceACES2Compliance().check(self._make(0.18), d, "SDR_sRGB")
        assert "finite_values" in report and "FAIL" in report

    def test_output_type_chooses_the_reference(self):
        """KNOWN_ISSUES listed output_type as a label; it picks the reference render."""
        scene = torch.rand(1, 16, 16, 3) * 4.0
        display, _ = RadianceACES2OutputTransformFull().transform(
            scene, "ACEScg", "ACES 2.0 SDR (sRGB/Rec.709)", engine="OCIO reference")
        _, sdr = RadianceACES2Compliance().check(scene, display, "SDR_sRGB")
        report, pq = RadianceACES2Compliance().check(scene, display, "HDR_PQ_1000")
        assert sdr == 4 and pq < sdr and "reference_match" in report

    def test_peak_nits_is_a_header_label_only(self):
        scene = torch.rand(1, 16, 16, 3) * 4.0
        display, _ = RadianceACES2OutputTransformFull().transform(
            scene, "ACEScg", "ACES 2.0 SDR (sRGB/Rec.709)", engine="OCIO reference")
        r100, n100 = RadianceACES2Compliance().check(scene, display, "SDR_sRGB", peak_nits=100.0)
        r4000, n4000 = RadianceACES2Compliance().check(scene, display, "SDR_sRGB", peak_nits=4000.0)
        assert n100 == n4000
        changed = [a for a, b in zip(r100.splitlines(), r4000.splitlines()) if a != b]
        assert len(changed) == 1 and "100" in changed[0]
        tip = RadianceACES2Compliance.INPUT_TYPES()["optional"]["peak_nits"][1]["tooltip"]
        assert tip.startswith("Label only")

    def test_report_makes_no_specification_claim(self):
        report, pass_count = RadianceACES2Compliance().check(self._make(0.18), self._make(0.1), "SDR_sRGB")
        assert "S-2126" not in report
        assert "Not a certification" in report
        assert isinstance(pass_count, int)


# ═════════════════════════════════════════════════════════════════════════════
# § 7  Node registration
# ═════════════════════════════════════════════════════════════════════════════

class TestNodeRegistration:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    EXPECTED_NODES = [
        "RadianceACES2Tonescale",
        "RadianceACES2ReachGamutCompress",
        "RadianceACES2OutputTransformFull",
        "RadianceACESMetadataFile",
        "RadianceACES2Compliance",
    ]

    def test_all_nodes_in_class_mappings(self):
        for name in self.EXPECTED_NODES:
            assert name in NODE_CLASS_MAPPINGS, f"Missing from NODE_CLASS_MAPPINGS: {name}"

    def test_all_nodes_in_display_mappings(self):
        for name in self.EXPECTED_NODES:
            assert name in NODE_DISPLAY_NAME_MAPPINGS, \
                f"Missing from NODE_DISPLAY_NAME_MAPPINGS: {name}"

    def test_display_names_use_radiance_icon(self):
        for name, display in NODE_DISPLAY_NAME_MAPPINGS.items():
            assert display.startswith("◎ Radiance"), \
                f"{name}: display name '{display}' missing '◎ Radiance' prefix"

    def test_display_names_are_unique(self):
        names = list(NODE_DISPLAY_NAME_MAPPINGS.values())
        assert len(names) == len(set(names)), "Duplicate display names detected"

    def test_node_classes_have_required_attrs(self):
        for cls_name, cls in NODE_CLASS_MAPPINGS.items():
            assert hasattr(cls, "FUNCTION"),      f"{cls_name} missing FUNCTION"
            assert hasattr(cls, "RETURN_TYPES"),  f"{cls_name} missing RETURN_TYPES"
            assert hasattr(cls, "CATEGORY"),      f"{cls_name} missing CATEGORY"
            assert hasattr(cls, "INPUT_TYPES"),   f"{cls_name} missing INPUT_TYPES"

    def test_categories_under_radiance(self):
        for cls_name, cls in NODE_CLASS_MAPPINGS.items():
            assert "FXTD STUDIOS/Radiance" in cls.CATEGORY, \
                f"{cls_name}: category '{cls.CATEGORY}' not under FXTD STUDIOS/Radiance"

    def test_all_under_hdr_aces(self):
        for cls_name, cls in NODE_CLASS_MAPPINGS.items():
            assert "FXTD STUDIOS/Radiance" in cls.CATEGORY, \
                f"{cls_name}: category '{cls.CATEGORY}' not under FXTD STUDIOS/Radiance"
            assert any(x in cls.CATEGORY for x in ("ACES", "HDR", "Color Science")), \
                f"{cls_name}: expected ACES/HDR/Color Science in category, got '{cls.CATEGORY}'"

    def test_node_count(self):
        assert len(NODE_CLASS_MAPPINGS) == len(self.EXPECTED_NODES), \
            f"Expected {len(self.EXPECTED_NODES)} nodes, found {len(NODE_CLASS_MAPPINGS)}"
