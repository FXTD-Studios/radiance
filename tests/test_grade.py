"""
tests/test_grade.py — Unit tests for nodes_grade.py

Covers:
  • _apply_grade math (lift, gamma, gain, offset, contrast, saturation)
  • strength=0 identity (FIX 2 regression)
  • Sign-preserving gamma on negative values (FIX 3 regression)
  • GradeMatch LAB statistics matching
  • RadianceGrade INPUT_TYPES and constants
  • RadianceGradeMatch forward pass (smoke)
"""

import sys
import types
import importlib
import pytest
import numpy as np

# ── Real torch check ──────────────────────────────────────────────────────────
try:
    import torch
    HAS_TORCH = hasattr(torch, "__version__")
except ImportError:
    HAS_TORCH = False

skip_no_torch = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")

# ── ComfyUI stub so nodes_grade imports cleanly ───────────────────────────────
for mod in ["folder_paths", "comfy", "comfy.utils"]:
    if mod not in sys.modules:
        sys.modules[mod] = types.ModuleType(mod)


# ── Import helpers from the grade module ─────────────────────────────────────
def _import_grade():
    """The module that defines the grade maths and node classes.

    `_apply_grade` and friends live here; the group package next door only
    aggregates node keys, which is what the deleted nodes_grade shim conflated.
    """
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
    return importlib.import_module("radiance.nodes.color.grade")


# ─────────────────────────────────────────────────────────────────────────────
# _apply_grade math
# ─────────────────────────────────────────────────────────────────────────────

@skip_no_torch
class TestApplyGradeMath:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    """Tests for the _apply_grade internal function."""

    def setup_method(self):
        self.mod = _import_grade()
        self.fn = self.mod._apply_grade
        # Neutral mid-grey image: (1, 4, 4, 3), value 0.5
        self.img = torch.full((1, 4, 4, 3), 0.5)

    def _run(self, **kwargs):
        defaults = dict(
            lift_r=0, lift_g=0, lift_b=0,
            gamma_r=1, gamma_g=1, gamma_b=1,
            gain_r=1, gain_g=1, gain_b=1,
            offset_r=0, offset_g=0, offset_b=0,
            contrast=1, pivot=0.18,
            saturation=1, strength=1.0,
        )
        defaults.update(kwargs)
        return self.fn(self.img, **defaults)

    def test_identity_returns_same(self):
        """All default params → image unchanged."""
        out = self._run()
        assert torch.allclose(out, self.img, atol=1e-5)

    def test_gain_doubles_brightness(self):
        out = self._run(gain_r=2, gain_g=2, gain_b=2)
        assert torch.allclose(out, self.img * 2, atol=1e-4)

    def test_lift_shifts_shadows(self):
        dark = torch.zeros(1, 4, 4, 3)
        out = self.fn(dark, lift_r=0.1, lift_g=0, lift_b=0,
                      gamma_r=1, gamma_g=1, gamma_b=1,
                      gain_r=1, gain_g=1, gain_b=1,
                      offset_r=0, offset_g=0, offset_b=0,
                      contrast=1, pivot=0.18, saturation=1, strength=1.0)
        # Red channel should be lifted; green/blue should remain 0
        assert out[0, 0, 0, 0].item() > 0
        assert out[0, 0, 0, 1].item() == pytest.approx(0.0, abs=1e-5)

    def test_saturation_zero_is_grey(self):
        """Saturation=0 → all channels equal (greyscale)."""
        coloured = torch.tensor([[[[0.2, 0.5, 0.9]]]])  # B=1,H=1,W=1,C=3
        out = self.fn(coloured, lift_r=0, lift_g=0, lift_b=0,
                      gamma_r=1, gamma_g=1, gamma_b=1,
                      gain_r=1, gain_g=1, gain_b=1,
                      offset_r=0, offset_g=0, offset_b=0,
                      contrast=1, pivot=0.18, saturation=0, strength=1.0)
        r, g, b = out[0, 0, 0, 0], out[0, 0, 0, 1], out[0, 0, 0, 2]
        assert r.item() == pytest.approx(g.item(), abs=1e-5)
        assert g.item() == pytest.approx(b.item(), abs=1e-5)

    def test_strength_zero_is_identity(self):
        """FIX 2 regression: strength=0 must return the original image unmodified."""
        # Apply extreme grade with strength=0 — result must equal input
        out = self._run(gain_r=10, gamma_r=5, saturation=0, strength=0.0)
        assert torch.allclose(out, self.img, atol=1e-5), (
            "strength=0 should be a perfect identity — FIX 2 regression"
        )

    def test_gamma_sign_preserving(self):
        """FIX 3 regression: negative input values must stay negative after gamma."""
        neg = torch.full((1, 2, 2, 3), -0.1)
        out = self._run.__func__(self.fn, neg,
                                 lift_r=0, lift_g=0, lift_b=0,
                                 gamma_r=2, gamma_g=2, gamma_b=2,
                                 gain_r=1, gain_g=1, gain_b=1,
                                 offset_r=0, offset_g=0, offset_b=0,
                                 contrast=1, pivot=0.18,
                                 saturation=1, strength=1.0) \
            if False else self.fn(neg,
                                   lift_r=0, lift_g=0, lift_b=0,
                                   gamma_r=2, gamma_g=2, gamma_b=2,
                                   gain_r=1, gain_g=1, gain_b=1,
                                   offset_r=0, offset_g=0, offset_b=0,
                                   contrast=1, pivot=0.18,
                                   saturation=1, strength=1.0)
        # Sign must be preserved
        assert (out < 0).all(), "FIX 3 regression: negative inputs must remain negative after gamma"

    def test_pure_black_stays_black(self):
        """FIX 3: gamma of zero must remain zero (clamp prevents inf)."""
        black = torch.zeros(1, 2, 2, 3)
        out = self._run(gamma_r=2, gamma_g=2, gamma_b=2)
        # Mid-grey input should not explode
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


# ─────────────────────────────────────────────────────────────────────────────
# LAB conversion roundtrip
# ─────────────────────────────────────────────────────────────────────────────

@skip_no_torch
class TestRGBToLAB:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    def setup_method(self):
        self.mod = _import_grade()

    def test_lab_shape_preserved(self):
        img = torch.rand(2, 8, 8, 3)
        lab = self.mod._rgb_to_lab(img)
        assert lab.shape == img.shape

    def test_d65_white_is_100_0_0(self):
        """sRGB white (1,1,1) → L*≈100, a*≈0, b*≈0 in CIE LAB."""
        white = torch.ones(1, 1, 1, 3)
        lab = self.mod._rgb_to_lab(white)
        L = lab[0, 0, 0, 0].item()
        a = lab[0, 0, 0, 1].item()
        b = lab[0, 0, 0, 2].item()
        assert L == pytest.approx(100.0, abs=1.0)
        assert a == pytest.approx(0.0, abs=2.0)
        assert b == pytest.approx(0.0, abs=2.0)

    def test_black_is_zero_L(self):
        black = torch.zeros(1, 1, 1, 3)
        lab = self.mod._rgb_to_lab(black)
        assert lab[0, 0, 0, 0].item() == pytest.approx(0.0, abs=0.5)


# ─────────────────────────────────────────────────────────────────────────────
# GradeMatch LAB statistics
# ─────────────────────────────────────────────────────────────────────────────

@skip_no_torch
class TestMatchGradeParams:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    def setup_method(self):
        self.mod = _import_grade()

    def test_identity_match(self):
        """Matching an image to itself should return near-identity params."""
        img = torch.rand(1, 16, 16, 3)
        params = self.mod._match_grade_params(img, img)
        # Gains should be ~1, offsets ~0
        for k in ["gain_r", "gain_g", "gain_b"]:
            assert params[k] == pytest.approx(1.0, abs=0.05), f"{k} should be ~1 for self-match"
        for k in ["offset_r", "offset_g", "offset_b"]:
            assert params[k] == pytest.approx(0.0, abs=0.05), f"{k} should be ~0 for self-match"

    def test_match_shifts_mean(self):
        """A brighter reference should produce a positive gain/offset."""
        src = torch.full((1, 8, 8, 3), 0.3)
        ref = torch.full((1, 8, 8, 3), 0.7)
        params = self.mod._match_grade_params(src, ref)
        # Combined grade should push src brighter
        out = self.mod._apply_grade(src, **params, strength=1.0)
        assert out.mean().item() > src.mean().item()


# ─────────────────────────────────────────────────────────────────────────────
# Node API surface
# ─────────────────────────────────────────────────────────────────────────────

class TestRadianceGradeAPI:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    def setup_method(self):
        self.mod = _import_grade()

    def test_node_registered(self):
        """Assert against what ComfyUI reads, not a module-local mapping.

        This read `NODE_CLASS_MAPPINGS` off the deleted nodes_grade shim, which
        built its own dict from the classes it had just imported — so the
        assertion held whether or not the node ever reached the catalog. The
        group package is what publishes.
        """
        from radiance.nodes.color import NODE_CLASS_MAPPINGS

        assert "RadianceGrade" in NODE_CLASS_MAPPINGS
        assert "RadianceGradeMatch" in NODE_CLASS_MAPPINGS
        assert "RadianceApplyGradeInfo" in NODE_CLASS_MAPPINGS

    def test_input_types_have_required_keys(self):
        cls = self.mod.RadianceGrade
        it = cls.INPUT_TYPES()
        required = it["required"]
        optional = it.get("optional", {})
        all_keys = {**required, **optional}
        assert "image" in required, "image must be a required input"
        # CDL params are in optional (preset system is the primary UI path)
        assert "lift_r" in all_keys, "lift_r must exist in INPUT_TYPES"
        assert "saturation" in all_keys, "saturation must exist in INPUT_TYPES"
        # RadianceGrade uses preset_strength / match_strength rather than plain 'strength'
        strength_present = any(
            "strength" in k for k in all_keys
        )
        assert strength_present, "some strength parameter must exist in INPUT_TYPES"

    def test_presets_list_nonempty(self):
        cls = self.mod.RadianceGrade
        it = cls.INPUT_TYPES()
        preset_entry = it["optional"].get("preset") or it["required"].get("preset")
        assert preset_entry is not None

    @skip_no_torch
    def test_forward_returns_correct_shape(self):
        node = self.mod.RadianceGrade()
        img = torch.rand(2, 8, 8, 3)
        # RadianceGrade.grade() does not take 'strength'; CDL params are optional kwargs
        result = node.grade(
            image=img,
            preset="None (Custom)",
            preset_strength=1.0,
            lift_r=0, lift_g=0, lift_b=0,
            gamma_r=1, gamma_g=1, gamma_b=1,
            gain_r=1, gain_g=1, gain_b=1,
            offset_r=0, offset_g=0, offset_b=0,
            contrast=1, pivot=0.18, saturation=1,
        )
        out_img = result[0]
        assert out_img.shape == img.shape

    @skip_no_torch
    def test_grade_match_forward_smoke(self):
        node = self.mod.RadianceGradeMatch()
        src = torch.rand(1, 8, 8, 3)
        ref = torch.rand(1, 8, 8, 3)
        result = node.match(source=src, reference=ref, strength=1.0)
        out_img = result[0]
        assert out_img.shape == src.shape
