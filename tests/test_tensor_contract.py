"""
test_tensor_contract.py — ensure_4d / ensure_5d shape contract tests.

Covers:
  • ensure_5d
      - 4D (B,C,H,W) → 5D (B,C,1,H,W) by adding singleton F dim
      - 5D input returned unchanged
      - 3D input raises ValueError
  • ensure_4d
      - 5D with F=1 → 4D (B,C,H,W) by squeezing F
      - 5D with F>1 → 4D (B*F,C,H,W) by flattening F into batch
      - 4D input returned unchanged
      - 3D input raises ValueError
  • Round-trip: ensure_4d(ensure_5d(t)) == t  for F=1
  • Context_label is included in ValueError messages
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import torch
    # Check for real torch: the stub (MagicMock) has no __version__ string
    HAS_TORCH = isinstance(getattr(torch, "__version__", None), str)
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(
    not HAS_TORCH,
    reason="torch not installed — tensor_contract requires torch"
)


@pytest.fixture(scope="module")
def ensure_4d():
    from tensor_contract import ensure_4d
    return ensure_4d


@pytest.fixture(scope="module")
def ensure_5d():
    from tensor_contract import ensure_5d
    return ensure_5d


# ─────────────────────────────────────────────────────────────────────────────
#  ensure_5d
# ─────────────────────────────────────────────────────────────────────────────

class TestEnsure5D:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    def test_4d_gains_singleton_frame_dim(self, ensure_5d):
        """(B,C,H,W) → (B,C,1,H,W)."""
        t = torch.zeros(2, 4, 8, 16)          # B=2 C=4 H=8 W=16
        out = ensure_5d(t)
        assert out.shape == (2, 4, 1, 8, 16)

    def test_4d_data_preserved(self, ensure_5d):
        """Values must be identical after adding the frame dimension."""
        t = torch.randn(1, 3, 4, 4)
        out = ensure_5d(t)
        assert torch.equal(out.squeeze(2), t)

    def test_5d_passthrough(self, ensure_5d):
        """5D input is returned as-is."""
        t = torch.zeros(1, 4, 3, 8, 8)
        out = ensure_5d(t)
        assert out is t
        assert out.shape == (1, 4, 3, 8, 8)

    def test_3d_raises_value_error(self, ensure_5d):
        """Anything other than 4D or 5D raises ValueError."""
        t = torch.zeros(4, 8, 8)
        with pytest.raises(ValueError):
            ensure_5d(t)

    def test_context_label_in_error(self, ensure_5d):
        """The context_label must appear in the error message."""
        t = torch.zeros(8, 8)   # 2D
        with pytest.raises(ValueError, match="MyContext"):
            ensure_5d(t, context_label="MyContext")

    def test_various_shapes(self, ensure_5d):
        """Parametric: different (B,C,H,W) sizes all gain exactly one F=1 dim."""
        for B, C, H, W in [(1, 1, 1, 1), (3, 16, 64, 64), (1, 4, 512, 512)]:
            t = torch.zeros(B, C, H, W)
            out = ensure_5d(t)
            assert out.shape == (B, C, 1, H, W), \
                f"Failed for shape ({B},{C},{H},{W})"


# ─────────────────────────────────────────────────────────────────────────────
#  ensure_4d
# ─────────────────────────────────────────────────────────────────────────────

class TestEnsure4D:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    def test_5d_f1_squeezes(self, ensure_4d):
        """(B,C,1,H,W) → (B,C,H,W) by squeezing the F=1 frame dim."""
        t = torch.zeros(2, 4, 1, 8, 16)
        out = ensure_4d(t)
        assert out.shape == (2, 4, 8, 16)

    def test_5d_f1_data_preserved(self, ensure_4d):
        """Values must be identical after squeezing F=1."""
        t = torch.randn(1, 3, 1, 4, 4)
        out = ensure_4d(t)
        assert torch.equal(out, t.squeeze(2))

    def test_5d_f3_flattens_to_batch(self, ensure_4d):
        """(B,C,F,H,W) with F>1 → (B*F, C, H, W)."""
        B, C, F, H, W = 2, 4, 3, 8, 8
        t = torch.zeros(B, C, F, H, W)
        out = ensure_4d(t)
        assert out.shape == (B * F, C, H, W)   # (6, 4, 8, 8)

    def test_5d_f3_data_ordering(self, ensure_4d):
        """
        After flattening F>1, the first B frames belong to batch item 0,
        the next B frames to batch item 1, etc.  (permute-then-reshape order)
        """
        # Build tensor where t[b, c, f] = b*100 + f (H=W=1 for simplicity)
        B, C, F = 2, 1, 3
        t = torch.zeros(B, C, F, 1, 1)
        for b in range(B):
            for f in range(F):
                t[b, 0, f, 0, 0] = b * 100 + f
        out = ensure_4d(t)   # (B*F, C, 1, 1)
        # permute(0,2,1,3,4) → (B,F,C,H,W) then reshape → (B*F,C,H,W)
        expected_values = [b * 100 + f for b in range(B) for f in range(F)]
        for i, ev in enumerate(expected_values):
            assert out[i, 0, 0, 0].item() == pytest.approx(ev)

    def test_4d_passthrough(self, ensure_4d):
        """4D input is returned as-is."""
        t = torch.zeros(1, 3, 8, 8)
        out = ensure_4d(t)
        assert out is t
        assert out.shape == (1, 3, 8, 8)

    def test_3d_raises_value_error(self, ensure_4d):
        """Anything other than 4D or 5D raises ValueError."""
        t = torch.zeros(3, 8, 8)
        with pytest.raises(ValueError):
            ensure_4d(t)

    def test_context_label_in_error(self, ensure_4d):
        """The context_label must appear in the error message."""
        t = torch.zeros(8)   # 1D
        with pytest.raises(ValueError, match="VideoDecoder"):
            ensure_4d(t, context_label="VideoDecoder")

    def test_various_f1_shapes(self, ensure_4d):
        """Parametric: different (B,C,1,H,W) sizes all squeeze to (B,C,H,W)."""
        for B, C, H, W in [(1, 1, 1, 1), (3, 16, 64, 64), (1, 4, 128, 128)]:
            t = torch.zeros(B, C, 1, H, W)
            out = ensure_4d(t)
            assert out.shape == (B, C, H, W), \
                f"Failed for shape ({B},{C},1,{H},{W})"


# ─────────────────────────────────────────────────────────────────────────────
#  Round-trips
# ─────────────────────────────────────────────────────────────────────────────

class TestRoundTrip:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    def test_4d_to_5d_to_4d(self, ensure_4d, ensure_5d):
        """4D → ensure_5d → ensure_4d must recover the original tensor exactly."""
        t = torch.randn(2, 4, 8, 8)
        rt = ensure_4d(ensure_5d(t))
        assert torch.equal(rt, t)

    def test_5d_f1_to_4d_to_5d(self, ensure_4d, ensure_5d):
        """5D(F=1) → ensure_4d → ensure_5d must recover the original shape."""
        t = torch.randn(2, 4, 1, 8, 8)
        rt = ensure_5d(ensure_4d(t))
        assert rt.shape == t.shape
        assert torch.equal(rt, t)

    def test_value_preservation_across_roundtrip(self, ensure_4d, ensure_5d):
        """Values must be byte-identical after a full round-trip."""
        t = torch.randn(3, 16, 32, 32)
        t_copy = t.clone()
        rt = ensure_4d(ensure_5d(t))
        assert torch.equal(rt, t_copy)
