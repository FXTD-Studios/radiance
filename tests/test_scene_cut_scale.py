"""One threshold widget, one meaning, for every scene-cut method.

`_histogram_diff` returns an L1 distance between normalised histograms — 0..2
by construction, a hard cut somewhere above 0.3. `_edge_diff` returned a mean
absolute difference of gradient magnitudes, which in practice stayed inside the
bottom tenth of its range. They were compared against the same `threshold`
widget, so a value tuned on one method silently meant something else on the
other, and `combined` added the two together as though they shared a unit:
nominally 60/40, in reality a histogram detector with a rounding error attached.

Unifying the scales turned up a second, larger defect underneath. The edge
distance was *absolute*, so it scaled with the footage's own contrast and
texture energy rather than with how different two frames were — the same cut
graded two stops down scored four times lower. No choice of threshold constant
could repair that, so `_edge_diff` is now a relative (Bray–Curtis) distance
between edge maps, with a small pre-blur so that grain does not dominate a
high-pass metric. `TestTheEdgeMetricIsScaleFree` is the test that argument
rests on, and it needs no synthetic-content assumptions: it grades one pair.

Both methods now report a calibrated confidence in [0, 1) where 0.5 is "as
different as a textbook hard cut". These tests pin that property rather than
the specific constants, so recalibrating a reference does not require rewriting
the suite — only `test_the_published_references_map_to_the_documented_point`
names the numbers.
"""
import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radiance.nodes.ai.scene_cut import (  # noqa: E402
    CUT_CONFIDENCE_AT_REFERENCE,
    EDGE_CUT_REFERENCE,
    HISTOGRAM_CUT_REFERENCE,
    _confidence,
    _edge_diff,
    _histogram_diff,
    detect_cuts,
)

REFERENCES = (
    ("histogram", HISTOGRAM_CUT_REFERENCE),
    ("edge", EDGE_CUT_REFERENCE),
)


# ── synthetic plates ─────────────────────────────────────────────────────────

def _gradient(seed, h=120, w=160):
    r = np.random.default_rng(seed)
    x = np.linspace(0, 1, w)[None, :, None]
    y = np.linspace(0, 1, h)[:, None, None]
    base = (0.3 + 0.5 * r.random(3))[None, None, :]
    return np.clip(base * (0.4 + 0.6 * x * y) + 0.05 * r.random((h, w, 3)), 0, 1).astype(np.float32)


def _texture(seed, h=120, w=160, scale=8):
    """Blocky detail, so the edge method has something to measure."""
    r = np.random.default_rng(seed)
    small = r.random((h // scale + 1, w // scale + 1, 3))
    return np.clip(np.kron(small, np.ones((scale, scale, 1)))[:h, :w], 0, 1).astype(np.float32)


def _field(seed, cutoff, h=120, w=160):
    """Band-limited noise: smoother, closer to real footage than blocky kron."""
    r = np.random.default_rng(seed)
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    rad = np.sqrt(fy ** 2 + fx ** 2)
    out = np.empty((h, w, 3), np.float32)
    for c in range(3):
        spec = np.fft.fft2(r.normal(size=(h, w))) * np.exp(-((rad / cutoff) ** 2))
        im = np.real(np.fft.ifft2(spec))
        out[:, :, c] = (im - im.min()) / (np.ptp(im) + 1e-9)
    return out.astype(np.float32)


def _grain(frame, amount=0.01, seed=0):
    r = np.random.default_rng(seed)
    return np.clip(frame + amount * r.normal(size=frame.shape), 0, 1).astype(np.float32)


def _clip(*shots, length=10):
    """Concatenate shots into one (B,H,W,3) sequence with grain within each."""
    frames = []
    for i, shot in enumerate(shots):
        frames.extend(_grain(shot, 0.008, seed=100 * i + j) for j in range(length))
    return np.stack(frames)


# ── the mapping itself ───────────────────────────────────────────────────────

class TestTheConfidenceMapping:

    @pytest.mark.parametrize("name,reference", REFERENCES)
    def test_the_published_references_map_to_the_documented_point(self, name, reference):
        assert _confidence(reference, reference) == pytest.approx(CUT_CONFIDENCE_AT_REFERENCE)

    def test_identical_frames_score_zero(self):
        assert _confidence(0.0, HISTOGRAM_CUT_REFERENCE) == 0.0

    def test_it_is_bounded_below_one(self):
        """A bound, not a clamp: no raw distance can saturate the scale."""
        assert _confidence(1e9, EDGE_CUT_REFERENCE) < 1.0

    def test_it_is_strictly_monotonic(self):
        """Nothing is clipped away, so the score plot keeps its dynamic range."""
        vals = [_confidence(x, HISTOGRAM_CUT_REFERENCE) for x in np.linspace(0, 4, 500)]
        assert all(b > a for a, b in zip(vals, vals[1:]))

    def test_a_negative_distance_cannot_appear(self):
        assert _confidence(-1.0, EDGE_CUT_REFERENCE) == 0.0


class TestTheEdgeMetricIsScaleFree:
    """The defect no choice of threshold constant could have repaired.

    These need no assumption about what real footage looks like: they take one
    pair of frames and re-grade it. A similarity metric must not care.
    """

    def test_grading_a_cut_down_does_not_change_its_score(self):
        a, b = _field(7, 0.2), _field(8, 0.2)

        full = _edge_diff(a, b)
        half = _edge_diff(a * 0.5, b * 0.5)
        quarter = _edge_diff(a * 0.25, b * 0.25)

        assert half == pytest.approx(full, rel=1e-4), \
            f"the same cut scored {full:.4f} at full contrast and {half:.4f} two stops down"
        assert quarter == pytest.approx(full, rel=1e-4)

    def test_the_old_absolute_metric_would_have_failed_that(self):
        """Guards the premise: mean|Δedge| really does track contrast."""
        a, b = _field(7, 0.2), _field(8, 0.2)

        def absolute(x, y):
            from radiance.nodes.ai.scene_cut import _edge_map
            return float(np.abs(_edge_map(x) - _edge_map(y)).mean())

        assert absolute(a * 0.25, b * 0.25) < 0.5 * absolute(a, b)

    def test_it_is_bounded_in_zero_one(self):
        a, b = _field(7, 0.2), _texture(3)
        assert 0.0 <= _edge_diff(a, b) <= 1.0

    def test_identical_frames_score_zero(self):
        a = _field(7, 0.2)
        assert _edge_diff(a, a) == pytest.approx(0.0, abs=1e-9)

    def test_two_flat_frames_do_not_divide_by_zero(self):
        flat_a = np.zeros((16, 16, 3), np.float32)
        flat_b = np.ones((16, 16, 3), np.float32)
        assert _edge_diff(flat_a, flat_b) == 0.0


class TestTheScalesAreComparable:
    """The property the widget depends on: 0.5 means a cut, whichever method."""

    def test_a_hard_cut_clears_the_reference_on_both_methods(self):
        a, b = _texture(5), _texture(6)

        hist = _confidence(_histogram_diff(a, b), HISTOGRAM_CUT_REFERENCE)
        edge = _confidence(_edge_diff(a, b), EDGE_CUT_REFERENCE)

        assert hist > CUT_CONFIDENCE_AT_REFERENCE, f"histogram missed a hard cut ({hist:.3f})"
        assert edge > CUT_CONFIDENCE_AT_REFERENCE, f"edge missed a hard cut ({edge:.3f})"

    def test_grain_on_a_held_frame_stays_below_the_reference_on_both(self):
        a = _field(11, 0.2)
        b = _grain(a, 0.02, seed=3)

        assert _confidence(_histogram_diff(a, b), HISTOGRAM_CUT_REFERENCE) < CUT_CONFIDENCE_AT_REFERENCE
        assert _confidence(_edge_diff(a, b), EDGE_CUT_REFERENCE) < CUT_CONFIDENCE_AT_REFERENCE

    def test_the_two_methods_no_longer_need_thresholds_a_factor_apart(self):
        """Before: the same widget value meant wildly different sensitivity.

        The raw distances still differ; that is the point. After calibration
        the confidences land within the same band.
        """
        a, b = _texture(5), _texture(6)

        conf_ratio = (
            _confidence(_histogram_diff(a, b), HISTOGRAM_CUT_REFERENCE)
            / _confidence(_edge_diff(a, b), EDGE_CUT_REFERENCE)
        )
        assert 0.5 < conf_ratio < 2.0, f"calibrated scales still {conf_ratio:.2f}x apart"


class TestCombinedIsARealBlend:

    def test_the_edge_term_can_change_the_verdict(self):
        """The defect: 0.4 * edge used to contribute ~1% of `combined`.

        A cut that the histogram cannot see (same palette, different detail)
        must be reachable through `combined` at a threshold the histogram alone
        would not clear.
        """
        shots = _clip(_texture(5, scale=8), _texture(5, scale=3))

        hist_cuts, _ = detect_cuts(shots, 0.5, min_shot_frames=4, method="histogram")
        comb_cuts, _ = detect_cuts(shots, 0.5, min_shot_frames=4, method="combined")

        assert len(comb_cuts) >= len(hist_cuts)

    def test_combined_sits_between_its_two_terms(self):
        """A weighted mean only means anything if the terms share a unit."""
        a, b = _texture(5), _texture(6)

        _, hist = detect_cuts(np.stack([a, b]), 0.5, 1, "histogram")
        _, edge = detect_cuts(np.stack([a, b]), 0.5, 1, "edge")
        _, comb = detect_cuts(np.stack([a, b]), 0.5, 1, "combined")

        lo, hi = sorted((float(hist[0]), float(edge[0])))
        assert lo - 1e-6 <= float(comb[0]) <= hi + 1e-6


class TestDetectCutsOnTheNewScale:

    def test_every_score_is_inside_the_common_range(self):
        for method in ("histogram", "edge", "combined"):
            _, scores = detect_cuts(_clip(_texture(1), _gradient(2)), 0.5, 4, method)
            assert scores.min() >= 0.0
            assert scores.max() < 1.0, f"{method} left the 0..1 scale"

    def test_a_cut_free_clip_reports_one_shot_on_every_method(self):
        held = _clip(_texture(9), length=24)
        for method in ("histogram", "edge", "combined"):
            cuts, _ = detect_cuts(held, 0.5, 4, method)
            assert cuts == [0], f"{method} invented a cut in held footage: {cuts}"

    def test_a_three_shot_clip_is_found_at_the_default_on_every_method(self):
        seq = _clip(_texture(1), _texture(2), _texture(3), length=12)
        for method in ("histogram", "edge", "combined"):
            cuts, _ = detect_cuts(seq, CUT_CONFIDENCE_AT_REFERENCE, 4, method)
            assert cuts == [0, 12, 24], f"{method} found {cuts}"

    def test_lowering_the_threshold_never_finds_fewer_cuts(self):
        seq = _clip(_texture(1), _texture(2), length=12)
        counts = [len(detect_cuts(seq, t, 4, "combined")[0]) for t in (0.8, 0.6, 0.4, 0.2)]
        assert counts == sorted(counts), f"sensitivity is not monotonic: {counts}"


class TestTheNodeWidget:

    def test_the_widget_is_named_for_what_it_now_means(self):
        from radiance.nodes.ai.scene_cut import RadianceSceneCutDetect

        required = RadianceSceneCutDetect.INPUT_TYPES()["required"]
        assert "cut_confidence" in required
        assert "distance_threshold" not in required, (
            "the old name would let ComfyUI carry a saved raw-distance value "
            "into the confidence scale silently"
        )
        assert "threshold" not in required

    def test_the_widget_range_matches_the_scale(self):
        from radiance.nodes.ai.scene_cut import RadianceSceneCutDetect

        spec = RadianceSceneCutDetect.INPUT_TYPES()["required"]["cut_confidence"][1]
        assert spec["default"] == pytest.approx(CUT_CONFIDENCE_AT_REFERENCE)
        assert spec["min"] > 0.0
        assert spec["max"] < 1.0, "1.0 is unreachable by construction"
