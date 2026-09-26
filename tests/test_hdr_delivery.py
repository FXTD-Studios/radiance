"""
tests/test_hdr_delivery.py — Unit tests for nodes_hdr_delivery.py

Covers:
  • _linear_to_pq: ST.2084 OETF output range and known reference values
  • _linear_to_hlg: HLG OETF output range and known reference values
  • PQ ↔ linear partial roundtrip (encode → decode monotonicity)
  • RadiancePQEncoder output shape, dtype, clamp range
  • RadianceHLGEncoder output shape, dtype, clamp range
  • RadianceHDRToneMapPreview smoke test
  • Node registrations present
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

# ── ComfyUI stub ──────────────────────────────────────────────────────────────
for mod in ["folder_paths", "comfy", "comfy.utils"]:
    if mod not in sys.modules:
        sys.modules[mod] = types.ModuleType(mod)


def _import_delivery():
    if "nodes_hdr_delivery" in sys.modules:
        return sys.modules["radiance.nodes.hdr.delivery"]
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
    return importlib.import_module("radiance.nodes.hdr.delivery")


# ─────────────────────────────────────────────────────────────────────────────
# _linear_to_pq (ST.2084 OETF)
# ─────────────────────────────────────────────────────────────────────────────

@skip_no_torch
class TestLinearToPQ:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    def setup_method(self):
        self.mod = _import_delivery()
        self.fn = self.mod._linear_to_pq

    def test_output_range_clamped_0_to_1(self):
        """PQ output must be in [0, 1] for all non-negative linear inputs."""
        x = torch.linspace(0, 2.0, 200)
        out = self.fn(x.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(1, 1, 200, 3))
        assert out.min().item() >= -1e-6
        assert out.max().item() <= 1.0 + 1e-6

    def test_black_maps_to_zero(self):
        x = torch.zeros(1, 4, 4, 3)
        out = self.fn(x)
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-5)

    def test_reference_white_203_nits(self):
        """BT.2408: scene-linear 1.0 = 203 nits → PQ value should be ~0.5807."""
        # Reference: PQ code for 203 nits / 10000 nits peak
        x = torch.ones(1, 1, 1, 3)  # scene-linear 1.0
        out = self.fn(x, peak_nits=10000.0)
        pq_val = out[0, 0, 0, 0].item()
        # PQ(203/10000) ≈ 0.5807 per ST.2084
        assert pq_val == pytest.approx(0.5807, abs=0.01)

    @pytest.mark.parametrize("peak_nits", [500.0, 1000.0, 2000.0, 4000.0, 10000.0])
    def test_reference_white_is_peak_independent(self, peak_nits):
        """Diffuse white sits at 203 nits whatever the mastering peak is.

        ST.2084 normalises by a fixed 10 000 cd/m²; the mastering peak clips
        luminance, it does not scale it.  The older implementation divided by
        ``peak_nits`` instead, which is correct only at 10 000 and is where the
        previous single-value test happened to sit, so the defect was invisible:
        at the shipped default of 1000 it encoded diffuse white at PQ 0.8290,
        which a conforming display shows at 2030 nits rather than 203.
        """
        x = torch.ones(1, 1, 1, 3)
        out = self.fn(x, peak_nits=peak_nits)
        assert out[0, 0, 0, 0].item() == pytest.approx(0.58069, abs=1e-4)

    def test_absolute_luminance_ladder(self):
        """Known ST.2084 code values for known absolute luminances.

        Checked against the curve itself rather than against this module, so a
        change to the normaliser fails here whichever direction it moves.
        """
        # (nits, PQ code) from ST.2084 with the 10 000 cd/m² ceiling, computed
        # from the rational constants (m1 = 2610/16384, m2 = 2523/4096*128,
        # c1 = 3424/4096, c2 = 2413/4096*32, c3 = 2392/4096*32) rather than
        # read back out of this module.
        ladder = [(1.0, 0.149946), (10.0, 0.299699), (100.0, 0.508078),
                  (203.0, 0.580689), (1000.0, 0.751827), (4000.0, 0.902572),
                  (10000.0, 1.0)]
        for nits, expected in ladder:
            scene_linear = nits / 203.0
            x = torch.full((1, 1, 1, 3), scene_linear)
            out = self.fn(x, peak_nits=10000.0)
            assert out[0, 0, 0, 0].item() == pytest.approx(expected, abs=2e-4), (
                f"{nits} nits should encode to PQ {expected}"
            )

    def test_peak_nits_clips_rather_than_scales(self):
        """Above the mastering peak the signal clips, and below it is untouched.

        This is the assertion that separates "peak clips" from "peak scales":
        under the old implementation a 1000-nit master and a 4000-nit master
        disagreed about every value, including diffuse white.  Under the correct
        one they agree everywhere below 1000 nits and differ only above it.
        """
        # 500 nits: under both masters, identical.
        x = torch.full((1, 1, 1, 3), 500.0 / 203.0)
        assert self.fn(x, peak_nits=1000.0)[0, 0, 0, 0].item() == pytest.approx(
            self.fn(x, peak_nits=4000.0)[0, 0, 0, 0].item(), abs=1e-6
        )
        # 4000 nits: the 1000-nit master clips it to its own peak.
        y = torch.full((1, 1, 1, 3), 4000.0 / 203.0)
        clipped = self.fn(y, peak_nits=1000.0)[0, 0, 0, 0].item()
        at_peak = self.fn(torch.full((1, 1, 1, 3), 1000.0 / 203.0),
                          peak_nits=1000.0)[0, 0, 0, 0].item()
        assert clipped == pytest.approx(at_peak, abs=1e-6)
        assert self.fn(y, peak_nits=4000.0)[0, 0, 0, 0].item() > clipped

    def test_agrees_with_the_aces2_encoder(self):
        """The package's two PQ encoders must not disagree by a factor of ten.

        ``nodes/hdr/aces2.py`` normalises by 10 000 on an ACES 1.0 = 100 nits
        scale.  Fed the same absolute luminance, both must produce the same
        code value.  They differed by 10x until the normaliser was fixed here.
        """
        from radiance.nodes.hdr.aces2 import _torch_pq_encode
        for nits in (1.0, 100.0, 203.0, 1000.0, 4000.0):
            # This module's scale is 1.0 = 203 nits; aces2's is 1.0 = 100 nits.
            mine = self.fn(
                torch.full((1, 1, 1, 3), nits / 203.0), peak_nits=10000.0
            )[0, 0, 0, 0].item()
            theirs = _torch_pq_encode(
                torch.full((1, 1, 1, 3), nits / 100.0)
            )[0, 0, 0, 0].item()
            assert mine == pytest.approx(theirs, abs=1e-4), (
                f"the two PQ encoders disagree at {nits} nits"
            )

    def test_monotonic(self):
        """PQ encoding must be strictly monotonically increasing."""
        vals = torch.linspace(0.01, 1.0, 50).view(1, 1, 50, 1).expand(1, 1, 50, 3)
        out = self.fn(vals)[0, 0, :, 0]
        diffs = out[1:] - out[:-1]
        assert (diffs >= 0).all(), "PQ OETF must be monotonically non-decreasing"

    def test_shape_preserved(self):
        x = torch.rand(2, 8, 8, 3)
        out = self.fn(x)
        assert out.shape == x.shape


# ─────────────────────────────────────────────────────────────────────────────
# _linear_to_hlg (HLG OETF)
# ─────────────────────────────────────────────────────────────────────────────

@skip_no_torch
class TestLinearToHLG:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    def setup_method(self):
        self.mod = _import_delivery()
        self.fn = self.mod._linear_to_hlg

    def test_output_range_0_to_1(self):
        """HLG output must stay in [0, 1] for inputs in [0, 1]."""
        x = torch.rand(1, 8, 8, 3)
        out = self.fn(x)
        assert out.min().item() >= -1e-6
        assert out.max().item() <= 1.0 + 1e-6

    def test_black_maps_to_zero(self):
        x = torch.zeros(1, 4, 4, 3)
        out = self.fn(x)
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-5)

    def test_reference_0_5_maps_to_0_5(self):
        """HLG: input 1/12 maps to 0.5 (boundary between linear and log segments)."""
        # At x=1/12, HLG transitions from linear to log; encoded value = 0.5
        x = torch.full((1, 1, 1, 3), 1.0 / 12.0)
        out = self.fn(x)
        val = out[0, 0, 0, 0].item()
        assert val == pytest.approx(0.5, abs=0.02)

    def test_monotonic(self):
        vals = torch.linspace(0.001, 1.0, 50).view(1, 1, 50, 1).expand(1, 1, 50, 3)
        out = self.fn(vals)[0, 0, :, 0]
        diffs = out[1:] - out[:-1]
        assert (diffs >= 0).all(), "HLG OETF must be monotonically non-decreasing"

    def test_shape_preserved(self):
        x = torch.rand(2, 16, 16, 3)
        out = self.fn(x)
        assert out.shape == x.shape


# ─────────────────────────────────────────────────────────────────────────────
# PQ encode → decode monotonicity
# ─────────────────────────────────────────────────────────────────────────────

@skip_no_torch
def test_pq_decode_monotonic():
    """Encoded PQ values should be monotonically related to input luminance."""
    mod = _import_delivery()
    inputs = torch.linspace(0.0, 5.0, 30)
    encoded = []
    for v in inputs:
        x = torch.full((1, 1, 1, 3), v.item())
        pq = mod._linear_to_pq(x)[0, 0, 0, 0].item()
        encoded.append(pq)
    for i in range(1, len(encoded)):
        assert encoded[i] >= encoded[i - 1] - 1e-6, \
            f"PQ encoding not monotonic at input {inputs[i].item():.3f}"


# ─────────────────────────────────────────────────────────────────────────────
# RadianceHDREncode node (merged PQ + HLG)
# ─────────────────────────────────────────────────────────────────────────────

@skip_no_torch
class TestRadianceHDREncode:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    def setup_method(self):
        self.mod = _import_delivery()
        self.node = self.mod.RadianceHDREncode()

    def test_pq_output_shape(self):
        img = torch.rand(1, 16, 16, 3)
        result = self.node.encode(image=img, format="PQ (HDR10)", peak_nits=1000)
        assert result[0].shape == img.shape

    def test_pq_output_clamped(self):
        img = torch.rand(1, 8, 8, 3) * 5.0
        out = self.node.encode(image=img, format="PQ (HDR10)", peak_nits=1000)[0]
        assert out.min().item() >= -1e-4
        assert out.max().item() <= 1.0 + 1e-4

    def test_pq_alpha_preserved(self):
        img = torch.rand(1, 8, 8, 4)
        img[..., 3] = 0.75
        out = self.node.encode(image=img, format="PQ (HDR10)", peak_nits=1000)[0]
        if out.shape[-1] == 4:
            assert torch.allclose(out[..., 3], img[..., 3], atol=1e-4)

    def test_hlg_output_shape(self):
        img = torch.rand(1, 16, 16, 3)
        result = self.node.encode(image=img, format="HLG (Broadcast)")
        assert result[0].shape == img.shape

    def test_hlg_output_clamped(self):
        img = torch.rand(1, 8, 8, 3)
        out = self.node.encode(image=img, format="HLG (Broadcast)")[0]
        assert out.min().item() >= -1e-4
        assert out.max().item() <= 1.0 + 1e-4


# ─────────────────────────────────────────────────────────────────────────────
# RadianceHDRMonitor node (merged tone-map preview)
# ─────────────────────────────────────────────────────────────────────────────

@skip_no_torch
def test_monitor_preview_smoke():
    mod = _import_delivery()
    if not hasattr(mod, "RadianceHDRMonitor"):
        pytest.skip("RadianceHDRMonitor not present")
    node = mod.RadianceHDRMonitor()
    img = torch.rand(1, 8, 8, 3) * 4.0
    result = node.monitor(image=img, mode="Preview (SDR)")
    assert result[0].shape == img.shape
    assert result[0].max().item() <= 1.0 + 1e-4


# ─────────────────────────────────────────────────────────────────────────────
# Node registrations
# ─────────────────────────────────────────────────────────────────────────────

@skip_no_torch
class TestNodeRegistrations:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    def setup_method(self):
        self.mod = _import_delivery()

    def test_hdr_encode_registered(self):
        assert "RadianceHDREncode" in self.mod.NODE_CLASS_MAPPINGS

    def test_hdr_monitor_registered(self):
        assert "RadianceHDRMonitor" in self.mod.NODE_CLASS_MAPPINGS

    def test_helper_functions_exported(self):
        assert hasattr(self.mod, "_linear_to_pq")
        assert hasattr(self.mod, "_linear_to_hlg")
