"""Both ACES 2.0 tone scales must put 18% grey where the standard puts it.

Radiance shipped two of them and they disagreed with each other by up to 24x:

  * `RadianceACES2Tonescale` used the real Daniele Evo curve but pinned grey to
    a fixed 10% *of peak*, so it rendered at 10 nits on SDR and 400 on a
    4000-nit master.
  * `RadianceACES2OutputTransform` used a log-contrast + tanh approximation
    that held grey at ~18 nits on every peak — nearly the right behaviour in
    kind, ~0.85 stop bright at SDR.

There was no judgement call to make in the end. ACES 2.0 publishes the mapping
(Output Transforms → Tone Mapping): an ACES value of 0.18 lands at 10.000 nits
on a 100-nit display, rising gently to 16.824 at 4000. The same table gives
ACES value 1.0 → 45.757 nits at 100-nit peak, i.e. the familiar ~48-nit diffuse
white, which is how you can tell those cells are luminances.

https://docs.acescentral.com/system-components/output-transforms/technical-details/tone-mapping/

Why the published table is transcribed into this file
-----------------------------------------------------
It has to be pinned against something outside the module. Every assertion here
used to be parametrised over `ACES2_MIDGREY_ANCHORS`, the tuple that
`radiance.hdr.tonescale` ships, and then asserted that `aces2_midgrey_nits`
returned those same numbers. That function is a log-log interpolator over that
very tuple: at an anchor the interpolation parameter is zero, so it hands back
the table entry by construction. Replacing the shipped table with the
fabricated ((100, 999.0), (500, 998.0), ...) left the whole file green, so a
typo in a published digit, 13.193 becoming 13.913, would have shipped with the
suite passing while the README quoted the digits as measured.

No third-party library implements the ACES 2.0 Output Transform to check
against: colour-science 0.4.x carries ACEScc/ACEScct/ACESproxy encodings only,
with no ACES 2 DRT or tone scale. So the published table is transcribed below
as literal data, and the shipped table is compared against it. The interpolator
is exercised where interpolation actually does something, strictly between two
anchors, against a closed form rather than a re-run of the module's own loop.
"""
import math
import os
import sys

import numpy as np
import pytest

try:
    import torch as _t
    _HAS_TORCH = isinstance(getattr(_t, "__version__", None), str)
except ImportError:
    _HAS_TORCH = False

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radiance.hdr.tonescale import (  # noqa: E402
    ACES2_MIDGREY_ANCHORS,
    SCENE_MIDGREY,
    aces2_midgrey_fraction,
    aces2_midgrey_nits,
)

# The published ACES 2.0 Output Transform table, transcribed by hand from
#   https://docs.acescentral.com/system-components/output-transforms/technical-details/tone-mapping/
# ("Tone Mapping" → "Values", page revision of 10 September 2025).
#
# Keys are the display peak luminance in nits. Values are the display
# luminance, in nits, that the Output Transform assigns to each of the four
# published ACES input values. The 1.0 column is what tells you the cells are
# luminances rather than normalised code values: 45.757 nits on a 100-nit
# display is the familiar ~48-nit diffuse white.
#
# This is deliberately a second, independent copy of the numbers that
# radiance.hdr.tonescale ships. It is the only thing in the repository that can
# catch a mistyped digit in that table, so do not derive it from the module.
PUBLISHED_ACES_INPUTS = (0.0, 0.18, 1.0, 2.0)
PUBLISHED_OUTPUT_TRANSFORM = {
    #  peak      ACES 0.0   ACES 0.18   ACES 1.0   ACES 2.0
    100.0:      (0.000,      10.000,      45.757,     63.988),
    500.0:      (0.000,      13.193,      89.098,    158.949),
    1000.0:     (0.000,      14.512,     106.564,    205.783),
    2000.0:     (0.000,      15.747,     121.664,    248.779),
    4000.0:     (0.000,      16.824,     133.883,    284.433),
}

#: The 0.18 column, as (peak, nits) pairs in ascending peak order.
PUBLISHED_MIDGREY = tuple(
    (peak, row[PUBLISHED_ACES_INPUTS.index(0.18)])
    for peak, row in sorted(PUBLISHED_OUTPUT_TRANSFORM.items())
)
PUBLISHED_BY_PEAK = dict(PUBLISHED_MIDGREY)


class TestTheReferenceTable:

    def test_the_shipped_table_is_the_published_table(self):
        """The assertion the old file was missing.

        Everything else in the module is downstream of these ten numbers, and
        nothing else compares them to a source outside the module.
        """
        assert len(ACES2_MIDGREY_ANCHORS) == len(PUBLISHED_MIDGREY), (
            f"the shipped table has {len(ACES2_MIDGREY_ANCHORS)} anchors, "
            f"the published one has {len(PUBLISHED_MIDGREY)}")
        for (peak, grey), (pub_peak, pub_grey) in zip(ACES2_MIDGREY_ANCHORS,
                                                      PUBLISHED_MIDGREY):
            assert float(peak) == pytest.approx(pub_peak, abs=1e-9), (
                f"shipped anchor peak {peak} is not the published {pub_peak}")
            assert float(grey) == pytest.approx(pub_grey, abs=1e-9), (
                f"the shipped 18% grey for a {pub_peak:.0f}-nit peak is {grey}, "
                f"the ACES Output Transform publishes {pub_grey}")

    def test_the_table_ships_the_18_percent_column(self):
        """Which column of the published table the module claims to carry."""
        assert SCENE_MIDGREY == 0.18
        assert SCENE_MIDGREY in PUBLISHED_ACES_INPUTS

    @pytest.mark.parametrize("peak,expected", PUBLISHED_MIDGREY)
    def test_published_anchors_are_exact(self, peak, expected):
        # Parametrised over the transcribed table, not over the shipped one:
        # that is the difference between a pin and a tautology.
        assert aces2_midgrey_nits(peak) == pytest.approx(expected, abs=1e-6)

    @pytest.mark.parametrize("lo,hi", list(zip(PUBLISHED_MIDGREY,
                                               PUBLISHED_MIDGREY[1:])))
    def test_it_interpolates_to_the_geometric_mean_between_anchors(self, lo, hi):
        """An anchor proves nothing about the interpolator; a point between two does.

        The documented rule is a straight line in log(peak) against log(grey),
        so at the geometric mean of two peaks the answer is the geometric mean
        of their two published greys. That is a closed form, not a second run
        of the module's own loop, and it moves if either anchor moves.
        """
        (peak_lo, grey_lo), (peak_hi, grey_hi) = lo, hi
        midpoint = math.sqrt(peak_lo * peak_hi)
        assert aces2_midgrey_nits(midpoint) == pytest.approx(
            math.sqrt(grey_lo * grey_hi), rel=1e-9), (
            f"log-log interpolation between the {peak_lo:.0f}- and "
            f"{peak_hi:.0f}-nit anchors is wrong at {midpoint:.1f} nits")

    def test_grey_rises_with_peak_but_nowhere_near_proportionally(self):
        """The whole bug in one assertion: 40x the peak, 1.68x the grey."""
        ratio = aces2_midgrey_nits(4000) / aces2_midgrey_nits(100)
        assert ratio == pytest.approx(
            PUBLISHED_BY_PEAK[4000.0] / PUBLISHED_BY_PEAK[100.0], rel=1e-9)
        assert ratio < 2.0, "18% grey is tracking the display peak again"

    def test_it_is_monotonic(self):
        vals = [aces2_midgrey_nits(p) for p in range(100, 8000, 50)]
        assert all(b > a for a, b in zip(vals, vals[1:]))

    def test_it_interpolates_between_anchors(self):
        assert PUBLISHED_BY_PEAK[100.0] < aces2_midgrey_nits(250) < PUBLISHED_BY_PEAK[500.0]
        assert PUBLISHED_BY_PEAK[1000.0] < aces2_midgrey_nits(1500) < PUBLISHED_BY_PEAK[2000.0]

    def test_it_extrapolates_instead_of_clamping(self):
        """A 10000-nit target should get an answer, not the 4000-nit one."""
        assert aces2_midgrey_nits(10000) > aces2_midgrey_nits(4000)
        assert aces2_midgrey_nits(50) < aces2_midgrey_nits(100)

    @pytest.mark.parametrize("peak,expected", PUBLISHED_MIDGREY)
    def test_the_fraction_form_agrees(self, peak, expected):
        assert aces2_midgrey_fraction(peak) == pytest.approx(expected / peak, rel=1e-9)

    def test_sdr_fraction_is_the_old_hardcoded_value(self):
        """0.10 was right — for 100 nits only, which is why SDR looked fine."""
        assert aces2_midgrey_fraction(100.0) == pytest.approx(
            PUBLISHED_BY_PEAK[100.0] / 100.0, abs=1e-9)
        assert aces2_midgrey_fraction(100.0) == pytest.approx(0.10, abs=1e-9)


@pytest.mark.skipif(not _HAS_TORCH, reason="needs real torch")
class TestDanieleEvoNode:

    @pytest.mark.parametrize("peak,expected", PUBLISHED_MIDGREY)
    def test_it_hits_the_published_grey(self, peak, expected):
        from radiance.nodes.hdr.aces2 import _DanieleEvoParams, _daniele_evo_fwd

        params = _DanieleEvoParams(peak_nits=peak)
        nits = float(_daniele_evo_fwd(np.array([0.18]), params)[0]) * 100.0
        assert nits == pytest.approx(expected, abs=0.01)

    def test_grey_no_longer_tracks_the_display(self):
        """It used to render 400 nits at a 4000-nit peak — 24x reference."""
        from radiance.nodes.hdr.aces2 import _DanieleEvoParams, _daniele_evo_fwd

        nits = float(_daniele_evo_fwd(np.array([0.18]), _DanieleEvoParams(peak_nits=4000.0))[0]) * 100.0
        assert nits < 20.0, f"18% grey at {nits:.1f} nits on a 4000-nit master"

    def test_an_explicit_grey_target_is_still_honoured(self):
        """The parameter stays available for deliberate deviation."""
        from radiance.nodes.hdr.aces2 import _DanieleEvoParams, _daniele_evo_fwd

        params = _DanieleEvoParams(peak_nits=1000.0, grey_target=0.05)
        nits = float(_daniele_evo_fwd(np.array([0.18]), params)[0]) * 100.0
        assert nits == pytest.approx(0.05 * 1000.0, rel=0.01)


@pytest.mark.skipif(not _HAS_TORCH, reason="needs real torch")
class TestLegacyOutputTransform:

    @staticmethod
    def _grey(peak):
        from radiance.hdr.color import ACES2OutputTransform

        rgb = np.full((1, 1, 1, 3), 0.18, dtype=np.float32)
        out = ACES2OutputTransform()._apply_tonescale_drt(
            rgb, peak_luminance=peak, is_hdr=(peak > 100))
        return float(out[0, 0, 0, 0]) * 100.0

    @pytest.mark.parametrize("peak,expected", PUBLISHED_MIDGREY)
    def test_it_hits_the_published_grey(self, peak, expected):
        assert self._grey(peak) == pytest.approx(expected, abs=0.05)

    def test_the_two_tone_scales_now_agree(self):
        """They disagreed by up to 24x. Same input, same peak, same answer."""
        from radiance.nodes.hdr.aces2 import _DanieleEvoParams, _daniele_evo_fwd

        for peak, _ in PUBLISHED_MIDGREY:
            evo = float(_daniele_evo_fwd(np.array([0.18]),
                                         _DanieleEvoParams(peak_nits=peak))[0]) * 100.0
            assert self._grey(peak) == pytest.approx(evo, rel=0.01), \
                f"the two tone scales disagree at {peak} nits"

    def test_a_neutral_stays_neutral(self):
        """The grey solve must not tint anything."""
        from radiance.hdr.color import ACES2OutputTransform

        rgb = np.full((1, 4, 4, 3), 0.18, dtype=np.float32)
        out = ACES2OutputTransform()._apply_tonescale_drt(rgb, peak_luminance=1000.0, is_hdr=True)
        assert out[..., 0] == pytest.approx(out[..., 1], abs=1e-6)
        assert out[..., 1] == pytest.approx(out[..., 2], abs=1e-6)

    def test_it_is_still_monotonic(self):
        from radiance.hdr.color import ACES2OutputTransform

        x = np.linspace(0.0, 40.0, 20001, dtype=np.float32)
        rgb = np.stack([x] * 3, axis=-1)[None, None]
        y = ACES2OutputTransform()._apply_tonescale_drt(rgb, peak_luminance=1000.0, is_hdr=True)[0, 0, :, 0]
        assert bool((np.diff(y) >= -1e-6).all()), "tonescale is not monotonic"

    def test_it_still_reaches_the_requested_peak(self):
        from radiance.hdr.color import ACES2OutputTransform

        rgb = np.full((1, 1, 1, 3), 10000.0, dtype=np.float32)
        out = ACES2OutputTransform()._apply_tonescale_drt(
            rgb, peak_luminance=4000.0, is_hdr=True)
        assert float(out[0, 0, 0, 0]) * 100.0 > 3900.0


@pytest.mark.skipif(not _HAS_TORCH, reason="needs real torch")
def test_the_scalar_probe_matches_the_vectorised_loop():
    """`_tanh_tonescale_scalar` duplicates the per-channel chain deliberately.

    The gain solve needs to evaluate the curve on one value, and the main loop
    is vectorised over an image. This asserts the copy has not drifted — the
    reason for the duplication is speed, not divergence.
    """
    from radiance.hdr.color import ACES2OutputTransform

    t = ACES2OutputTransform()
    for peak in (100.0, 1000.0, 4000.0):
        peak_scale = peak / 100.0
        for x in (0.01, 0.18, 0.5, 1.0, 4.0):
            rgb = np.full((1, 1, 1, 3), x, dtype=np.float32)
            # Undo the grey anchoring so the raw curve is what gets compared.
            gain = t._solve_grey_gain(
                __import__("radiance.hdr.tonescale", fromlist=["x"]).aces2_midgrey_nits(peak) / 100.0,
                peak_scale, 1.0)
            loop = float(t._apply_tonescale_drt(rgb, peak_luminance=peak, is_hdr=(peak > 100))[0, 0, 0, 0])
            scalar = t._tanh_tonescale_scalar(x * gain, peak_scale, 1.55)
            assert loop == pytest.approx(scalar, rel=1e-4, abs=1e-6), \
                f"scalar probe and vectorised loop disagree at x={x}, peak={peak}"
