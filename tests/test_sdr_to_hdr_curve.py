"""The flagship SDR→HDR curve, measured against the standards it cites.

`RadianceSDRToHDRUniversal` is a knee-plus-shoulder inverse tone mapper, which
is the same family as ITU-R BT.2446 Method B. Benchmarking it against that
report turned up two defects that no amount of parameter tuning could reach.

**1. SDR diffuse white was mapped to the display peak.** The expansion targeted
`peak_nits` at SDR code 1.0, so a white shirt, a page of text or a cloud came
out at the full mastering peak — measured 1000.00 nits at the shipped defaults
against the 203 nits ITU-R BT.2408 defines as HDR Reference White. Nearly five
times reference, on exactly the value a viewer's eye uses to judge exposure for
the whole frame. BT.2446 Method B scales SDR by ~2 to reach 203 and expands
*only* above the breakpoint, by 2.3x in display light.

**2. The knee was not C¹.** Below it the curve is the identity, gradient 1.
Above it the gradient was `(peak−k)·gamma·t^(gamma−1)/(1−k)`, which tends to
zero for any gamma > 1. Measured at knee 0.75, peak_scale 10: gradient 1.000
below, 0.083 above at the old default gamma of 1.6, and exactly 0.000 at 2.5.
That is a ridge in any smooth gradient crossing the knee, and since adaptive
mode moves the knee per frame, on video it crawls. BT.2446 Method B: "the
gradient of the exponential function is set to unity at the breakpoint".

References:
  ITU-R BT.2446-1 (03/2021), Method B, §5.1
  ITU-R BT.2408, HDR Reference White = 203 cd/m²
"""
import os
import sys

import pytest

try:
    import torch as _t
    _HAS_TORCH = isinstance(getattr(_t, "__version__", None), str)
except ImportError:
    _HAS_TORCH = False

if not _HAS_TORCH:
    pytest.skip("torch not installed", allow_module_level=True)

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radiance.nodes.hdr.uplift_universal import (  # noqa: E402
    RadianceSDRToHDRUniversal,
    _soft_knee_expand,
    _soft_peak_limit,
)

#: ITU-R BT.2408. A white shirt, a page, a cloud.
BT2408_REFERENCE_WHITE_NITS = 203.0


def _patch_strip(values):
    """(1, 1, N, 3) neutral patches at the given SDR code values."""
    return torch.tensor([[[[v, v, v] for v in values]]], dtype=torch.float32)


def _convert(image, **kw):
    args = dict(
        inverse_oetf="sRGB", peak_nits=1000.0,
        reference_white_nits=BT2408_REFERENCE_WHITE_NITS,
        knee_mode="manual", knee=0.75, shoulder_gamma=2.0,
        temporal_smoothing=0.0, output_encoding="Linear",
        processing_mode="Expand",
    )
    args.update(kw)
    out, *rest = RadianceSDRToHDRUniversal().convert(image, **args)
    # 3.5.0: Linear output is BT.2408-normalised (1.0 = reference white, the
    # effective one: capped at the peak). Convert back to 1.0 = 100 nits so
    # _nits() reads absolute luminance.
    ref = min(max(float(args["reference_white_nits"]), 100.0),
              max(float(args["peak_nits"]), 100.0))
    out = torch.cat([out[..., :3] * (ref / 100.0), out[..., 3:]], dim=-1)
    return (out, *rest)


def _nits(out, index=0):
    return float(out[0, 0, index, 0]) * 100.0


# ── where SDR white lands ────────────────────────────────────────────────────

class TestReferenceWhite:

    def test_sdr_diffuse_white_lands_on_the_bt2408_reference(self):
        out, *_ = _convert(_patch_strip([1.0]))
        assert _nits(out) == pytest.approx(BT2408_REFERENCE_WHITE_NITS, abs=0.5)

    def test_it_used_to_land_on_the_display_peak(self):
        """The defect, still reachable, because some grades do want it."""
        out, *_ = _convert(_patch_strip([1.0]), reference_white_nits=1000.0)
        assert _nits(out) == pytest.approx(1000.0, rel=1e-3)

    def test_reference_white_is_independent_of_the_display_peak(self):
        """Raising the mastering ceiling must not move diffuse white."""
        for peak in (600.0, 1000.0, 4000.0):
            out, *_ = _convert(_patch_strip([1.0]), peak_nits=peak)
            assert _nits(out) == pytest.approx(BT2408_REFERENCE_WHITE_NITS, abs=0.5), \
                f"diffuse white moved when peak_nits changed to {peak}"

    def test_reference_white_can_never_exceed_the_display_peak(self):
        out, *_ = _convert(_patch_strip([1.0]), peak_nits=250.0,
                           reference_white_nits=900.0)
        assert _nits(out) <= 250.0 + 1e-3

    def test_midtones_are_left_alone(self):
        """BT.2446 requirement 2: raising mean luminance causes discomfort."""
        codes = [0.0, 0.4585, 0.5, 0.75]        # 0.4585 sRGB ≈ 0.18 linear
        out, *_ = _convert(_patch_strip(codes))
        for i, code in enumerate(codes):
            linear = code / 12.92 if code <= 0.04045 else ((code + 0.055) / 1.055) ** 2.4
            assert _nits(out, i) == pytest.approx(linear * 100.0, abs=0.05), \
                f"midtone at code {code} moved"

    def test_the_curve_is_monotonic_across_the_whole_range(self):
        codes = torch.linspace(0.0, 1.0, 512).tolist()
        out, *_ = _convert(_patch_strip(codes))
        y = out[0, 0, :, 0]
        assert bool((y.diff() >= -1e-6).all()), "the transfer curve is not monotonic"


# ── the join ─────────────────────────────────────────────────────────────────

class TestTheKneeIsContinuousInGradient:

    @staticmethod
    def _gradients_either_side(knee, peak, gamma, eps=1e-5):
        # float64: a finite difference of 1e-5 on values near 0.75 in float32
        # is measuring rounding as much as slope (it reads 1.0014 for a segment
        # that is exactly the identity).
        k = torch.tensor([knee], dtype=torch.float64)
        at = torch.tensor([[knee]], dtype=torch.float64)
        below = torch.tensor([[knee - eps]], dtype=torch.float64)
        above = torch.tensor([[knee + eps]], dtype=torch.float64)
        g_below = float((_soft_knee_expand(at, k, peak, gamma)
                         - _soft_knee_expand(below, k, peak, gamma)) / eps)
        g_above = float((_soft_knee_expand(above, k, peak, gamma)
                         - _soft_knee_expand(at, k, peak, gamma)) / eps)
        return g_below, g_above

    @pytest.mark.parametrize("gamma", [2.0, 2.5, 4.0, 6.0])
    def test_the_gradient_is_continuous_at_the_knee(self, gamma):
        below, above = self._gradients_either_side(0.75, 10.0, gamma)
        assert below == pytest.approx(1.0, abs=1e-3)
        assert above == pytest.approx(1.0, rel=0.01), (
            f"gradient steps from {below:.3f} to {above:.3f} at the knee — "
            "a visible ridge in any gradient crossing it"
        )

    def test_the_old_curve_would_have_failed_that(self):
        """Guards the premise: a bare power shoulder collapses to zero slope."""
        knee, peak, gamma, eps = 0.75, 10.0, 2.5, 1e-5

        def old(luma):
            t = ((luma - knee) / (1.0 - knee))
            t = t.clamp(0.0, 1.0)
            return torch.where(luma > knee, knee + (peak - knee) * t ** gamma, luma)

        above = float((old(torch.tensor([knee + eps])) - old(torch.tensor([knee]))) / eps)
        assert above < 0.01, "premise gone: the old shoulder was not flat at the knee"

    def test_value_continuity_too(self):
        k = torch.tensor([0.75])
        eps = 1e-6
        at = _soft_knee_expand(torch.tensor([[0.75]]), k, 10.0, 2.0)
        below = _soft_knee_expand(torch.tensor([[0.75 - eps]]), k, 10.0, 2.0)
        assert float(at) == pytest.approx(float(below), abs=1e-5)

    def test_the_shoulder_still_reaches_the_target_at_code_one(self):
        for gamma in (1.0, 2.0, 6.0):
            y = _soft_knee_expand(torch.tensor([[1.0]]), torch.tensor([0.75]), 10.0, gamma)
            assert float(y) == pytest.approx(10.0, rel=1e-5)

    @pytest.mark.parametrize("gamma", [1.0, 1.5, 2.0, 3.0, 6.0])
    def test_the_shoulder_is_monotonic(self, gamma):
        luma = torch.linspace(0.0, 1.0, 20001).unsqueeze(0)
        y = _soft_knee_expand(luma, torch.tensor([0.75]), 10.0, gamma)
        assert bool((y.diff() >= -1e-7).all())

    def test_gamma_below_one_is_clamped_not_honoured(self):
        """Below 1 the shoulder cannot have unity gradient; 1.0 is the floor."""
        a = _soft_knee_expand(torch.tensor([[0.9]]), torch.tensor([0.75]), 10.0, 0.4)
        b = _soft_knee_expand(torch.tensor([[0.9]]), torch.tensor([0.75]), 10.0, 1.0)
        assert float(a) == pytest.approx(float(b))

    def test_the_widget_range_matches_what_the_curve_supports(self):
        spec = RadianceSDRToHDRUniversal.INPUT_TYPES()["required"]["shoulder_gamma"][1]
        assert spec["min"] >= 1.0
        assert spec["default"] >= 2.0, "the default should sit in the C¹ range"


# ── the limiter was already right; keep it that way ──────────────────────────

class TestThePeakLimiter:

    def test_it_is_continuous_at_its_own_knee(self):
        """Regression guard: this half of the curve was never the problem."""
        peak, eps = 10.0, 1e-4
        knee = peak * 0.9

        def y_of(v):
            rgb = torch.full((1, 1, 1, 3), v, dtype=torch.float32)
            out = _soft_peak_limit(rgb, peak)
            return float(out[0, 0, 0, 0])

        below = (y_of(knee) - y_of(knee - eps)) / eps
        above = (y_of(knee + eps) - y_of(knee)) / eps
        assert below == pytest.approx(above, rel=0.05)

    def test_nothing_escapes_the_mastering_peak(self):
        rgb = torch.rand(1, 8, 8, 3) * 50.0
        out = _soft_peak_limit(rgb, 10.0)
        luma = (0.2126 * out[..., 0] + 0.7152 * out[..., 1] + 0.0722 * out[..., 2])
        assert float(luma.max()) <= 10.0 + 1e-4


# ── the node contract did not change shape ───────────────────────────────────

class TestTheNodeStillHonoursItsContract:

    def test_the_outputs_and_the_alpha_passthrough(self):
        rgba = torch.cat([_patch_strip([0.5, 1.0]),
                          torch.full((1, 1, 2, 1), 0.25)], dim=-1)
        out, hi, sh, hc, sc, report = _convert(rgba)
        assert out.shape[-1] == 4
        assert float(out[0, 0, 0, 3]) == pytest.approx(0.25)
        for m in (hi, sh, hc, sc):
            assert m.shape == (1, 1, 2)
        assert isinstance(report, str) and report

    def test_an_empty_batch_does_not_crash(self):
        empty = torch.zeros((0, 4, 4, 3))
        out, *_ = _convert(empty)
        assert out.shape[0] == 0

    def test_expand_mode_needs_no_model(self):
        out, *_ = _convert(_patch_strip([1.0]), processing_mode="Expand", vae=None)
        assert torch.isfinite(out).all()


# ── the report ───────────────────────────────────────────────────────────────

class TestTheReportSaysWhatActuallyRan:
    """Recover and Hybrid are the reason this node is called Universal.

    On a machine with no checkpoint installed they are bit-identical to Expand,
    because every learned backend is wrapped in try/except and only logs on the
    way past. Measured on a clean install before this output existed:

        pixel checkpoint discoverable:      False
        Hybrid output identical to Expand:  True
        Recover output identical to Expand: True

    Five normal-looking outputs and nothing on the graph to say the node's
    headline feature never engaged. The `report` output is that signal.
    """

    @pytest.fixture(autouse=True)
    def no_checkpoint(self, monkeypatch):
        import radiance.pixel_sdr2hdr as pixel
        monkeypatch.setattr(pixel, "resolve_pixel_checkpoint", lambda *a, **k: None)

    @staticmethod
    def _report(**kw):
        img = torch.rand(1, 16, 16, 3)
        img[:, :4, :4, :] = 1.0                  # a clipped highlight to recover
        return _convert(img, **kw)[5]

    def test_expand_says_it_did_not_ask_for_recovery(self):
        r = self._report(processing_mode="Expand")
        assert "mode: Expand" in r
        assert "not requested" in r
        assert "NOT APPLIED" not in r

    @pytest.mark.parametrize("mode", ["Hybrid", "Recover"])
    def test_a_learned_mode_with_no_backend_says_so_in_capitals(self, mode):
        r = self._report(processing_mode=mode)
        assert "NOT APPLIED" in r, r
        assert "identical to Expand" in r, r
        assert mode in r

    @pytest.mark.parametrize("mode", ["Hybrid", "Recover"])
    def test_it_names_the_thing_the_user_has_to_install(self, mode):
        r = self._report(processing_mode=mode)
        assert "models/radiance" in r or "RADIANCE_SDR2HDR_PIXEL" in r, r

    def test_a_zero_blend_is_reported_as_the_reason_when_it_is(self):
        r = self._report(processing_mode="Hybrid", rudra_blend=0.0)
        assert "rudra_blend is 0" in r, r

    def test_the_report_carries_the_numbers_that_decide_the_look(self):
        r = self._report(processing_mode="Expand", reference_white_nits=203.0,
                         peak_nits=4000.0)
        assert "203 nits" in r
        assert "4000 nits" in r
        assert "BT.2408" in r

    def test_a_learned_mode_that_did_nothing_logs_a_warning(self, caplog):
        with caplog.at_level("WARNING"):
            self._report(processing_mode="Hybrid")
        assert "deterministic Expand output" in caplog.text

    def test_expand_does_not_warn(self, caplog):
        with caplog.at_level("WARNING"):
            self._report(processing_mode="Expand")
        assert "deterministic Expand output" not in caplog.text

    def test_the_report_survives_an_empty_batch(self):
        out, _, _, _, _, report = _convert(torch.zeros((0, 4, 4, 3)))
        assert out.shape[0] == 0
        assert isinstance(report, str) and report

    def test_the_description_warns_before_the_user_runs_it(self):
        """The report is after the fact; the node description is before."""
        text = RadianceSDRToHDRUniversal.DESCRIPTION
        assert "checkpoint" in text.lower()
        assert "fall back" in text.lower() or "falls back" in text.lower()
