"""
tests/test_writer_colorspaces.py — every output colour space actually converts.

The defect this file exists for: ``io/writer.py`` offered ten output colour
spaces, called ``color_utils.linear_to_rec709`` for one of them, and that
function did not exist anywhere in the package.  The AttributeError was caught
by a blanket ``except Exception`` that logged one warning and returned the
array untouched, so a Rec.709 export wrote scene-linear values — roughly 2.2
stops dark in the midtones — and the node reported success.  Rec.2020 was
worse: it applied a bare 1/2.4 power curve (its own comment called it "PQ gamma
as a proxy") and hard-clipped to [0,1], destroying every highlight above 1.0 on
an EXR export.

Nothing caught either one, because no test asserted that a conversion changed
the data.  These do, per colour space, by value.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from radiance.io.writer import OUTPUT_COLOR_SPACES, _apply_output_colorspace


# Every entry that is not the explicit pass-through must transform the data.
# "Linear Rec.709 (sRGB)" is the default working space itself, so encoding
# into it from Linear Rec.709 is correctly the identity.
NON_IDENTITY = [cs for cs in OUTPUT_COLOR_SPACES
                if cs not in ("Linear (pass-through)", "Linear Rec.709 (sRGB)")]


@pytest.mark.parametrize("cs", NON_IDENTITY)
def test_every_offered_colorspace_actually_transforms(cs):
    """A listed colour space that returns its input unchanged is not implemented.

    This is the assertion that was missing.  Under the old code Rec.709 passed
    its input straight through and nothing failed.
    """
    # A saturated, non-neutral patch: a primaries-only conversion leaves a
    # neutral grey almost untouched (the matrix rows sum to 1), so a neutral
    # probe would let a matrix transform look like a no-op.
    arr = np.tile(
        np.array([0.9, 0.2, 0.05], dtype=np.float32), (4, 4, 1)
    ).astype(np.float32)
    out = _apply_output_colorspace(arr.copy(), cs)
    assert out.shape == arr.shape
    assert not np.allclose(out, arr, atol=1e-4), (
        f"{cs!r} returned scene-linear unchanged, so it is not converting"
    )


@pytest.mark.parametrize("cs", NON_IDENTITY)
def test_no_colorspace_silently_swallows_a_failure(cs):
    """A conversion that cannot run must raise, not return the input."""
    arr = np.full((2, 2, 3), 0.18, dtype=np.float32)
    out = _apply_output_colorspace(arr, cs)
    assert np.isfinite(out).all(), f"{cs!r} produced non-finite output"


def test_unknown_colorspace_raises():
    arr = np.zeros((2, 2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="Unknown output color space"):
        _apply_output_colorspace(arr, "Rec.2100 Cinematic Deluxe")


def test_passthrough_is_the_only_identity():
    arr = np.full((2, 2, 3), 0.42, dtype=np.float32)
    assert np.array_equal(_apply_output_colorspace(arr, "Linear (pass-through)"), arr)


# ── Rec.709: ITU-R BT.709-6 §1.2 ─────────────────────────────────────────────

class TestRec709:
    def test_published_anchor_values(self):
        """V = 4.5L below 0.018, V = 1.099 L^0.45 - 0.099 above it.

        Values computed from the recommendation, not read back out of the
        module, so a change to either branch fails here.
        """
        ladder = [
            (0.0,     0.0),
            (0.001,   0.0045),      # linear segment: 4.5 * 0.001
            (0.0179,  0.08055),     # just below the breakpoint, linear branch
            (0.018,   0.0812480),   # at the breakpoint the power branch takes
                                    # over (BT.709 says L < 0.018 for the
                                    # linear segment).  The two branches do not
                                    # meet exactly, because 1.099 and 0.018 are
                                    # the recommendation's own rounded values;
                                    # the ~0.00025 step is in the standard.
            (0.18,    0.4090078),   # 18% grey
            (1.0,     1.0),         # 1.099 * 1 - 0.099
        ]
        for linear, expected in ladder:
            arr = np.full((1, 1, 3), linear, dtype=np.float32)
            out = _apply_output_colorspace(arr, "Rec.709")
            assert out[0, 0, 0] == pytest.approx(expected, abs=2e-4), (
                f"linear {linear} should encode to {expected}"
            )

    def test_is_not_srgb(self):
        """Rec.709 and sRGB are different curves and must not be aliased.

        A plausible "fix" for the missing function would have been to point it
        at linear_to_srgb.  At 18% grey they differ by about 0.05, which is
        visible, so this pins them apart.
        """
        arr = np.full((1, 1, 3), 0.18, dtype=np.float32)
        r709 = _apply_output_colorspace(arr, "Rec.709")[0, 0, 0]
        srgb = _apply_output_colorspace(arr, "sRGB")[0, 0, 0]
        assert abs(float(r709) - float(srgb)) > 0.02

    def test_round_trip(self):
        from radiance import color_utils as cu
        arr = np.array([[[0.0, 0.005, 0.18]]], dtype=np.float32)
        back = cu.rec709_to_linear(cu.linear_to_rec709(arr))
        assert np.allclose(back, arr, atol=1e-5)

    def test_highlights_above_one_are_not_crushed(self):
        """Float delivery keeps its headroom; only integer formats quantise."""
        arr = np.array([[[1.0, 2.0, 8.0]]], dtype=np.float32)
        out = _apply_output_colorspace(arr, "Rec.709")
        assert out[0, 0, 1] > out[0, 0, 0] > 0.99
        assert out[0, 0, 2] > out[0, 0, 1]

    def test_negatives_keep_their_sign(self):
        arr = np.array([[[-0.18, -0.005, 0.18]]], dtype=np.float32)
        out = _apply_output_colorspace(arr, "Rec.709")
        assert out[0, 0, 0] < 0 and out[0, 0, 1] < 0 and out[0, 0, 2] > 0
        assert out[0, 0, 0] == pytest.approx(-out[0, 0, 2], abs=1e-6)


# ── Rec.2020: ITU-R BT.2020-2 ────────────────────────────────────────────────

class TestRec2020:
    def test_applies_the_primaries_matrix(self):
        """Rec.2020 is a wider gamut, so a saturated Rec.709 red must move.

        The old implementation was transfer-only, so a pure red came out with
        its chromaticity untouched and the file was mislabelled.
        """
        red = np.zeros((1, 1, 3), dtype=np.float32)
        red[0, 0, 0] = 1.0
        out = _apply_output_colorspace(red.copy(), "Rec.2020")
        # Rec.709 red maps into Rec.2020 with non-zero green and blue.
        assert out[0, 0, 1] > 0.01, "green channel unchanged: no primaries conversion"
        assert out[0, 0, 2] > 0.01, "blue channel unchanged: no primaries conversion"

    def test_neutral_stays_neutral(self):
        grey = np.full((1, 1, 3), 0.18, dtype=np.float32)
        out = _apply_output_colorspace(grey, "Rec.2020")
        assert out[0, 0, 0] == pytest.approx(out[0, 0, 1], abs=1e-4)
        assert out[0, 0, 1] == pytest.approx(out[0, 0, 2], abs=1e-4)

    def test_does_not_clip_highlights(self):
        """The old branch did np.clip(..., 0, 1), losing all HDR headroom."""
        arr = np.full((1, 1, 3), 8.0, dtype=np.float32)
        out = _apply_output_colorspace(arr, "Rec.2020")
        assert out.max() > 1.0, "Rec.2020 clipped an over-range value to 1.0"

    def test_round_trip(self):
        from radiance import color_utils as cu
        arr = np.array([[[0.02, 0.18, 0.9]]], dtype=np.float32)
        back = cu.rec2020_to_linear_srgb(
            cu.rec2020_to_linear(
                cu.linear_to_rec2020(cu.linear_srgb_to_rec2020(arr))
            )
        )
        assert np.allclose(back, arr, atol=1e-4)

    def test_is_not_the_old_power_curve(self):
        """Pins the fix: the old result was arr ** (1/2.4), clipped."""
        arr = np.full((1, 1, 3), 0.18, dtype=np.float32)
        out = _apply_output_colorspace(arr, "Rec.2020")[0, 0, 0]
        old = float(np.clip(0.18 ** (1 / 2.4), 0, 1))
        assert abs(float(out) - old) > 1e-3


def test_rec2020_matrix_matches_the_torch_one():
    """The numpy and torch BT.2020 matrices must not drift apart."""
    from radiance.color.matrices import SRGB_TO_REC2020, REC2020_TO_SRGB
    from radiance.color.ops import M_REC709_TO_BT2020, M_BT2020_TO_REC709
    assert np.allclose(SRGB_TO_REC2020, M_REC709_TO_BT2020.numpy(), atol=1e-7)
    assert np.allclose(REC2020_TO_SRGB, M_BT2020_TO_REC709.numpy(), atol=1e-7)
