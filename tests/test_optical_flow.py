"""Optical flow has to survive more than a couple of pixels of motion.

The single-scale Lucas-Kanade this replaces linearised brightness constancy as
`It = f2 - f1`, which only holds while the displacement stays inside the
gradient's support. Measured recovery was ~100% at 1 px, 17% at 3 px and 1% at
5 px, so `RadianceVideoMaskPropagator` was effectively static on anything but
the slowest moves — while still returning a confident-looking flow field.

The pyramid does not change the solver; it changes the scale it runs at, so
each level only ever solves for sub-pixel residual motion.
"""
import os
import sys

import pytest

try:
    import torch
    _HAS_TORCH = isinstance(getattr(torch, "__version__", None), str)
except ImportError:
    _HAS_TORCH = False

if not _HAS_TORCH:
    pytest.skip("torch not installed", allow_module_level=True)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radiance.nodes.vfx.multipass.core import _optical_flow_lk, _warp_by_flow


@pytest.fixture(scope="module")
def plate():
    """A plate with texture at several scales.

    Deliberately not random noise: noise has energy only at the finest scale,
    so it vanishes on the coarse pyramid levels and understates what the
    solver does on real footage. Overlapping sinusoids give the multi-scale
    structure a real plate has.
    """
    h, w = 96, 200
    ys, xs = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing="ij",
    )
    plate = (0.5
             + 0.20 * torch.sin(xs / 7.0)
             + 0.15 * torch.cos(ys / 5.0)
             + 0.10 * torch.sin((xs + ys) / 11.0)
             + 0.05 * torch.sin(xs / 2.5))
    return plate.unsqueeze(0)


def _shift_pair(plate, dx=0, dy=0, h=64, w=128):
    """Two crops of one plate, the second displaced by (dx, dy)."""
    a = plate[:, dy:dy + h, dx:dx + w]
    b = plate[:, 0:h, 0:w]
    return a, b


def _centre(t, m=20):
    """Median of the interior — edges are unconstrained by definition."""
    return float(t[:, m:-m, m:-m].median())


class TestHorizontalRecovery:

    @pytest.mark.parametrize("shift", [1, 2, 3, 5, 8])
    def test_recovers_the_displacement(self, plate, shift):
        """3 px used to come back at 17%, 5 px at 1%."""
        a, b = _shift_pair(plate, dx=shift)
        u, _ = _optical_flow_lk(a, b)
        got = _centre(u)
        assert got == pytest.approx(shift, rel=0.10), \
            f"{shift} px shift recovered as {got:.2f}"

    @pytest.mark.parametrize("shift,min_fraction", [(1, 0.9), (3, 0.9), (5, 0.8), (8, 0.7)])
    def test_the_field_is_accurate_per_pixel_not_just_on_average(
        self, plate, shift, min_fraction
    ):
        """A correct median over a noisy field would not propagate a mask.

        This is the property mask warping actually needs: most of the frame
        within half a pixel, not merely centred on the right answer.
        """
        a, b = _shift_pair(plate, dx=shift)
        u, _ = _optical_flow_lk(a, b)
        interior = u[:, 20:-20, 20:-20]
        within = float(((interior - shift).abs() < 0.5).float().mean())
        assert within >= min_fraction, \
            f"{shift} px: only {within * 100:.0f}% of the field within 0.5 px"


class TestVerticalAndCrossAxis:

    @pytest.mark.parametrize("shift", [1, 3, 5])
    def test_vertical_recovery(self, plate, shift):
        a, b = _shift_pair(plate, dy=shift)
        _, v = _optical_flow_lk(a, b)
        assert _centre(v) == pytest.approx(shift, rel=0.15)

    def test_a_pure_horizontal_move_does_not_leak_into_v(self, plate):
        a, b = _shift_pair(plate, dx=4)
        _, v = _optical_flow_lk(a, b)
        assert abs(_centre(v)) < 0.5, "vertical flow appeared from a horizontal move"

    def test_a_pure_vertical_move_does_not_leak_into_u(self, plate):
        a, b = _shift_pair(plate, dy=4)
        u, _ = _optical_flow_lk(a, b)
        assert abs(_centre(u)) < 0.5, "horizontal flow appeared from a vertical move"


class TestDegenerateInputs:

    def test_identical_frames_give_zero_flow(self, plate):
        a = plate[:, :64, :128]
        u, v = _optical_flow_lk(a, a.clone())
        assert abs(_centre(u)) < 0.05
        assert abs(_centre(v)) < 0.05

    def test_no_second_frame_gives_zeros(self, plate):
        a = plate[:, :64, :128]
        u, v = _optical_flow_lk(a, None)
        assert u.shape == a.shape and v.shape == a.shape
        assert float(u.abs().max()) == 0.0
        assert float(v.abs().max()) == 0.0

    def test_a_flat_frame_does_not_produce_nonsense(self):
        """Textureless input is singular; it must resolve to ~no motion."""
        flat = torch.full((1, 48, 64), 0.5)
        u, v = _optical_flow_lk(flat, flat.clone())
        assert torch.isfinite(u).all() and torch.isfinite(v).all()
        assert float(u.abs().max()) < 1.0

    def test_a_frame_smaller_than_the_pyramid_still_works(self):
        """The level count must adapt rather than downsample into nothing."""
        torch.manual_seed(2)
        small = torch.rand(1, 16, 20)
        u, v = _optical_flow_lk(small, small.clone())
        assert u.shape == small.shape
        assert torch.isfinite(u).all() and torch.isfinite(v).all()

    def test_batches_are_handled_independently(self, plate):
        a1, b1 = _shift_pair(plate, dx=3)
        a2, b2 = _shift_pair(plate, dx=0)
        u, _ = _optical_flow_lk(torch.cat([a1, a2]), torch.cat([b1, b2]))
        assert _centre(u[:1]) == pytest.approx(3, rel=0.2)
        assert abs(_centre(u[1:])) < 0.5


class TestWarpHelper:

    def test_warping_by_the_recovered_flow_realigns_the_frames(self, plate):
        """The end-to-end property mask propagation actually depends on."""
        a, b = _shift_pair(plate, dx=3)
        u, v = _optical_flow_lk(a, b)

        before = float((a - b).abs()[:, 20:-20, 20:-20].mean())
        after = float((a - _warp_by_flow(b, u, v)).abs()[:, 20:-20, 20:-20].mean())
        assert after < before * 0.2, \
            f"warping by the flow barely improved alignment: {before:.4f} -> {after:.4f}"

    def test_zero_flow_is_an_identity_warp(self, plate):
        a = plate[:, :64, :128]
        z = torch.zeros_like(a)
        assert torch.allclose(_warp_by_flow(a, z, z), a, atol=1e-5)
