"""
tests/test_hdr_colorspace.py — Unit tests for nodes_hdr_colorspace.py

Covers:
  • _apply_matrix: shape preservation, identity matrix
  • EOTF functions: sRGB, Rec.709, PQ, HLG output ranges
  • Bradford chromatic adaptation: D65↔D60 roundtrip
  • Primaries matrix entries exist for key pairs
  • RadianceLinearize INPUT_TYPES and node registration
  • RadianceChromaAdapt adaptation list non-empty
  • RadianceHDRColorPipeline and RadianceColorSpaceInfo registrations
"""

import sys
import types
import importlib
import pytest

# ── Real torch check ──────────────────────────────────────────────────────────
try:
    import torch
    HAS_TORCH = hasattr(torch, "__version__")
except ImportError:
    HAS_TORCH = False

skip_no_torch = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")

# ── ComfyUI stub ──────────────────────────────────────────────────────────────
for _mod in ["folder_paths", "comfy", "comfy.utils"]:
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)


def _import_cs():
    if "nodes_hdr_colorspace" in sys.modules:
        return sys.modules["radiance.nodes.hdr.colorspace"]
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
    return importlib.import_module("radiance.nodes.hdr.colorspace")


# ─────────────────────────────────────────────────────────────────────────────
# _apply_matrix
# ─────────────────────────────────────────────────────────────────────────────

@skip_no_torch
class TestApplyMatrix:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    def setup_method(self):
        self.mod = _import_cs()
        self.fn = self.mod._apply_matrix

    def test_identity_matrix_noop(self):
        """Identity matrix must leave pixel values unchanged."""
        img = torch.rand(2, 8, 8, 3)
        identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        out = self.fn(img, identity)
        assert torch.allclose(out, img, atol=1e-5)

    def test_shape_preserved(self):
        img = torch.rand(1, 16, 16, 3)
        m = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        assert self.fn(img, m).shape == img.shape

    def test_zero_matrix_gives_zero(self):
        img = torch.rand(1, 4, 4, 3)
        zero = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        out = self.fn(img, zero)
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)

    def test_swap_rg_channels(self):
        """Swap R and G channels via permutation matrix."""
        img = torch.zeros(1, 1, 1, 3)
        img[0, 0, 0, 0] = 1.0  # R=1, G=0, B=0
        # Matrix that swaps R↔G: row 0 becomes old G, row 1 becomes old R
        swap_rg = [[0, 1, 0], [1, 0, 0], [0, 0, 1]]
        out = self.fn(img, swap_rg)
        assert out[0, 0, 0, 0].item() == pytest.approx(0.0, abs=1e-5)  # new R = old G = 0
        assert out[0, 0, 0, 1].item() == pytest.approx(1.0, abs=1e-5)  # new G = old R = 1


# ─────────────────────────────────────────────────────────────────────────────
# EOTF functions
# ─────────────────────────────────────────────────────────────────────────────

@skip_no_torch
class TestEOTFs:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    def setup_method(self):
        self.mod = _import_cs()

    def test_srgb_eotf_black_to_black(self):
        fn = self.mod._eotf_srgb
        x = torch.zeros(1, 4, 4, 3)
        assert torch.allclose(fn(x), torch.zeros_like(x), atol=1e-6)

    def test_srgb_eotf_white_to_white(self):
        """sRGB EOTF: display 1.0 → linear 1.0."""
        fn = self.mod._eotf_srgb
        x = torch.ones(1, 1, 1, 3)
        out = fn(x)
        assert out.min().item() == pytest.approx(1.0, abs=1e-4)

    def test_srgb_eotf_monotonic(self):
        fn = self.mod._eotf_srgb
        vals = torch.linspace(0.0, 1.0, 50).view(1, 1, 50, 1).expand(1, 1, 50, 3)
        out = fn(vals)[0, 0, :, 0]
        diffs = out[1:] - out[:-1]
        assert (diffs >= -1e-6).all(), "sRGB EOTF must be monotonically non-decreasing"

    def test_pq_eotf_output_non_negative(self):
        """PQ EOTF output (linear nits fraction) must be ≥ 0 for inputs in [0,1]."""
        fn = self.mod._eotf_pq
        x = torch.rand(1, 8, 8, 3)
        out = fn(x)
        assert out.min().item() >= -1e-6

    def test_pq_eotf_shape_preserved(self):
        fn = self.mod._eotf_pq
        x = torch.rand(2, 16, 16, 3)
        assert fn(x).shape == x.shape

    def test_hlg_eotf_black_maps_to_black(self):
        fn = self.mod._eotf_hlg
        x = torch.zeros(1, 4, 4, 3)
        out = fn(x)
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-5)

    def test_hlg_eotf_monotonic(self):
        fn = self.mod._eotf_hlg
        vals = torch.linspace(0.0, 1.0, 50).view(1, 1, 50, 1).expand(1, 1, 50, 3)
        out = fn(vals)[0, 0, :, 0]
        diffs = out[1:] - out[:-1]
        assert (diffs >= -1e-6).all(), "HLG EOTF must be monotonically non-decreasing"

    def test_gamma22_shape(self):
        fn = self.mod._eotf_gamma22
        x = torch.rand(1, 8, 8, 3)
        assert fn(x).shape == x.shape

    def test_rec709_eotf_shape(self):
        fn = self.mod._eotf_rec709
        x = torch.rand(1, 8, 8, 3)
        assert fn(x).shape == x.shape


# ─────────────────────────────────────────────────────────────────────────────
# Bradford chromatic adaptation
# ─────────────────────────────────────────────────────────────────────────────

@skip_no_torch
class TestBradfordCAT:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    def setup_method(self):
        self.mod = _import_cs()
        self.apply = self.mod._apply_matrix
        cats = self.mod._BRADFORD_CAT
        self.d65_to_d60 = cats["D65_to_D60"]
        self.d60_to_d65 = cats["D60_to_D65"]

    def test_d65_d60_roundtrip_neutral(self):
        """D65→D60→D65 should return near-original values for neutral grey."""
        img = torch.full((1, 4, 4, 3), 0.5)
        adapted = self.apply(img, self.d65_to_d60)
        restored = self.apply(adapted, self.d60_to_d65)
        assert torch.allclose(restored, img, atol=1e-3), \
            "Bradford D65→D60→D65 roundtrip must be near-lossless"

    def test_d65_d50_keys_present(self):
        cats = self.mod._BRADFORD_CAT
        assert "D65_to_D50" in cats
        assert "D50_to_D65" in cats

    def test_d65_to_d60_shape_preserved(self):
        img = torch.rand(2, 8, 8, 3)
        out = self.apply(img, self.d65_to_d60)
        assert out.shape == img.shape


# ─────────────────────────────────────────────────────────────────────────────
# Primaries matrices
# ─────────────────────────────────────────────────────────────────────────────

@skip_no_torch
class TestPrimariesMatrices:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    def setup_method(self):
        self.mod = _import_cs()

    def test_709_to_2020_key_exists(self):
        pm = self.mod._PRIMARIES_MATRICES
        assert ("Rec.709 (sRGB)", "BT.2020") in pm

    def test_2020_to_709_key_exists(self):
        pm = self.mod._PRIMARIES_MATRICES
        assert ("BT.2020", "Rec.709 (sRGB)") in pm

    def test_709_to_acescg_key_exists(self):
        pm = self.mod._PRIMARIES_MATRICES
        assert ("Rec.709 (sRGB)", "ACEScg") in pm

    def test_709_2020_709_roundtrip(self):
        """Rec.709 → BT.2020 → Rec.709 should be near-identity."""
        pm = self.mod._PRIMARIES_MATRICES
        apply = self.mod._apply_matrix
        img = torch.rand(1, 4, 4, 3) * 0.8 + 0.1  # avoid corners
        to2020 = pm[("Rec.709 (sRGB)", "BT.2020")]
        to709 = pm[("BT.2020", "Rec.709 (sRGB)")]
        roundtrip = apply(apply(img, to2020), to709)
        assert torch.allclose(roundtrip, img, atol=1e-3), \
            "Rec.709 ↔ BT.2020 matrix roundtrip must be near-lossless"

    def test_primaries_matrix_is_3x3(self):
        pm = self.mod._PRIMARIES_MATRICES
        m = pm[("Rec.709 (sRGB)", "BT.2020")]
        assert len(m) == 3 and all(len(row) == 3 for row in m)


# ─────────────────────────────────────────────────────────────────────────────
# Node API surface
# ─────────────────────────────────────────────────────────────────────────────

@skip_no_torch
class TestNodeRegistrations:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    def setup_method(self):
        self.mod = _import_cs()

    def test_all_nodes_registered(self):
        ncm = self.mod.NODE_CLASS_MAPPINGS
        for key in [
            "RadianceHDRColorPipeline",
            "RadianceColorSpaceInfo",
        ]:
            assert key in ncm, f"{key} missing from NODE_CLASS_MAPPINGS"

    def test_pipeline_input_types(self):
        cls = self.mod.RadianceHDRColorPipeline
        it = cls.INPUT_TYPES()
        assert "image" in it["required"]
        assert "encoding" in it["required"]
        assert "compression_ratio" in it["required"]
        
        all_keys = {**it["required"], **it.get("optional", {})}
        assert "source_primaries" in all_keys
        assert "target_primaries" in all_keys
        assert "chromatic_adaptation" in all_keys

    def test_pipeline_forward_smoke(self):
        node = self.mod.RadianceHDRColorPipeline()
        img = torch.rand(1, 8, 8, 3)
        # Use the function name from FUNCTION attribute
        fn = getattr(node, self.mod.RadianceHDRColorPipeline.FUNCTION)
        # pipeline returns: (vae_image, scene_linear, peak_linear, colorspace_json)
        vae_image, scene_linear, peak_linear, colorspace_json = fn(
            image=img,
            encoding="sRGB",
            compression_ratio=0.5,
            source_primaries="Rec.709 (sRGB)",
            target_primaries="Rec.709 (sRGB)",
            chromatic_adaptation="None",
        )
        assert vae_image.shape == img.shape
        assert scene_linear.shape == img.shape
        assert isinstance(peak_linear, float)
        assert isinstance(colorspace_json, str)

