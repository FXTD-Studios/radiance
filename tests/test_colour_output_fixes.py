"""Regression tests for the pre-release colour fixes.

Four defects, all of the same shape: the transform ran, returned a plausible
image, and was wrong. Nothing raised, so 1891 tests had nothing to catch.
"""
import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

RADIANCE_TORCH_GATED = True

_SDR = "ACES 2.0 SDR (sRGB/Rec.709)"
_PQ1000 = "ACES 2.0 HDR (Rec.2100 PQ 1000 nits)"
_HLG = "ACES 2.0 HDR (Rec.2100 HLG)"
_DCI = "ACES 2.0 Cinema (DCI-P3 D65)"


def _aces(transform, scene):
    from radiance.hdr.color import ACES2OutputTransform

    out = ACES2OutputTransform().apply_transform(
        torch.full((1, 2, 2, 3), float(scene)), "ACEScg", transform,
        peak_luminance=100.0, surround="Dim",
    )[0]
    return out[0, 0, 0]


# ── 1. DCI no longer flat-lines at 0.4 scene ────────────────────────────────

@pytest.mark.real_torch
def test_cinema_reaches_full_white_instead_of_clipping_at_027_nits():
    """
    peak_nits was 48 for cinema, which is the projector luminance of code value
    1.0, not a scale on the scene. It made peak_scale 0.48, so the tonescale
    knee landed at 0.432 with 0.048 of headroom and everything above ~0.4 scene
    collapsed: measured 0.40 -> 0.75403, 0.50 -> 0.75405, 8.0 -> 0.75405. A DCP
    master was flat above 0.4 with a peak white of 0.754 — about 27 nits, not 48.
    """
    assert float(_aces(_DCI, 8.0)[0]) > 0.99, "cinema still cannot reach white"

    ramp = [float(_aces(_DCI, v)[0]) for v in (0.18, 0.30, 0.40, 0.50)]
    assert ramp == sorted(ramp), "the cinema ramp is not monotonic"
    assert ramp[3] - ramp[2] > 0.05, (
        f"0.40 -> {ramp[2]:.5f} and 0.50 -> {ramp[3]:.5f} are still nearly "
        "identical; the shoulder is saturating again"
    )


@pytest.mark.real_torch
def test_cinema_still_reports_48_nits_to_the_user():
    from radiance.hdr.color import ACES2OutputTransform

    info = ACES2OutputTransform().apply_transform(
        torch.full((1, 2, 2, 3), 0.18), "ACEScg", _DCI,
        peak_luminance=100.0, surround="Dim",
    )[1]
    assert "48" in info


# ── 2. HLG is a relative system ─────────────────────────────────────────────

@pytest.mark.real_torch
def test_hlg_places_18_percent_grey_at_the_bt2408_reference():
    """
    peak_nits fell through to `peak_luminance` (default 100) for HLG, so the
    tonescale asymptoted at 1.0 and _hlg_encode mapped diffuse white straight to
    signal 1.0 — the display peak. Measured before: 18% grey -> signal 0.672
    (~127 nits on a 1000-nit display, where BT.2408 wants ~26).
    """
    signal = float(_aces(_HLG, 0.18)[0])
    assert 0.34 < signal < 0.42, (
        f"18% grey encodes to HLG signal {signal:.4f}; BT.2408 puts HLG Grey at "
        "about 0.38"
    )


@pytest.mark.real_torch
def test_hlg_no_longer_pins_diffuse_white_to_the_display_peak():
    signal = float(_aces(_HLG, 1.0)[0])
    assert signal < 0.99, (
        "diffuse white is still encoding to HLG signal 1.0, i.e. the full "
        "display peak"
    )


@pytest.mark.real_torch
def test_hlg_is_monotonic_and_bounded():
    vals = [float(_aces(_HLG, v)[0]) for v in (0.0, 0.18, 0.5, 1.0, 2.0, 8.0, 64.0)]
    assert vals == sorted(vals)
    assert vals[-1] <= 1.0 + 1e-6


def test_hlg_reference_constant_is_documented():
    import pathlib

    src = (pathlib.Path(__file__).resolve().parent.parent / "hdr/color.py").read_text(encoding="utf-8")
    assert "_HLG_DIFFUSE_WHITE_SCENE" in src
    assert "BT.2408" in src or "BT.2100" in src


# ── 3. The paths that were already right must stay right ────────────────────

@pytest.mark.real_torch
# Updated 2026-08-16 when 18% grey was anchored to the ACES 2.0 reference.
# These are not "whatever the code now prints" — each is derivable:
#   SDR  0.18 -> 0.3492 = the sRGB encode of 10.000 nits on a 100-nit display
#                         (1.055*0.10^(1/2.4)-0.055), the ACES 2.0 SDR grey.
#                         The old 0.4610 encoded ~18 nits, 0.85 stop bright.
#   PQ   0.18 -> 0.3298 = the PQ signal for 14.512 nits, the ACES 2.0 value at
#                         a 1000-nit peak. The old 0.3478 was PQ for 17.97.
#   PQ   8.00 -> 0.7518 unchanged: the highlight end of the curve did not move,
#                         only where the midtone sits on it.
@pytest.mark.parametrize("transform,scene,expected", [
    (_SDR, 0.18, 0.3492),
    (_SDR, 0.50, 0.7284),
    (_PQ1000, 0.18, 0.3298),
    (_PQ1000, 1.00, 0.5827),
    (_PQ1000, 8.00, 0.7518),
])
def test_sdr_and_pq_are_untouched_by_the_hlg_and_cinema_fix(transform, scene, expected):
    assert abs(float(_aces(transform, scene)[0]) - expected) < 5e-4


@pytest.mark.real_torch
@pytest.mark.parametrize("transform", [_SDR, _PQ1000, _HLG, _DCI])
def test_every_transform_preserves_neutrals(transform):
    rgb = _aces(transform, 0.5)
    assert float(rgb.max() - rgb.min()) < 1e-5, f"{transform} tints a neutral"


# ── 4. AgX: matrix orientation and the double transfer encode ───────────────

def _agx(scene, operator="agx"):
    from radiance.hdr.tonemap import HDRToneMap

    return HDRToneMap().tonemap(
        torch.full((1, 2, 2, 3), float(scene)), preset="Custom", operator=operator,
        exposure=0.0, gamma=2.2, white_point=1.0, contrast=1.0, saturation=1.0,
        highlight_compression=0.0, shadow_lift=0.0, use_gpu=False,
    )[0][0, 0, 0]


@pytest.mark.real_torch
@pytest.mark.parametrize("scene", [0.18, 1.0, 16.0])
def test_agx_does_not_tint_neutrals(scene):
    """
    The inset matrix was the GLSL column-major form used as rows, so
    `M @ [1,1,1]` was (0.9272, 1.0353, 1.0375) instead of white. Measured
    channel spread on a neutral: 0.0420 at 0.18 and 0.1155 at 16.0 — a visible
    warm cast on every grey, growing with exposure.
    """
    rgb = _agx(scene)
    assert float(rgb.max() - rgb.min()) < 1e-3, (
        f"AgX spreads a neutral by {float(rgb.max() - rgb.min()):.5f} at scene "
        f"{scene}; the inset matrix is back in the wrong orientation"
    )


def test_agx_inset_matrix_is_white_preserving():
    from radiance.hdr.tonemap import _AGX_M_IN_VALUES

    m = np.array(_AGX_M_IN_VALUES, dtype=np.float64)
    white = m @ np.ones(3)
    assert np.allclose(white, 1.0, atol=1e-3), (
        f"M @ [1,1,1] = {white}; einsum('ij,...j->...i') needs the "
        "white-preserving orientation"
    )


@pytest.mark.real_torch
def test_agx_is_not_transfer_encoded_twice():
    """
    The AgX sigmoid already emits display code values and `pow(1/gamma)` ran on
    top. Measured with the double encode: scene 0.0 -> 0.0477, a black floor of
    ~12/255 that made pure black unreachable, and 0.02 -> 0.4845 (a near-black
    pixel at 48% grey).
    """
    assert float(_agx(0.0)[0]) < 0.01, "AgX still has a raised black floor"
    assert float(_agx(0.02)[0]) < 0.30, "near-black is still rendering as mid grey"


@pytest.mark.real_torch
def test_agx_is_monotonic_and_rolls_off():
    vals = [float(_agx(v)[0]) for v in (0.0, 0.02, 0.18, 1.0, 4.0, 16.0, 64.0)]
    assert vals == sorted(vals)
    assert vals[-1] <= 1.0
    assert vals[-1] > 0.85, "the shoulder is crushing highlights"


@pytest.mark.real_torch
def test_other_operators_still_get_their_gamma_encode():
    """The AgX exemption must not leak into the operators that need the encode."""
    assert abs(float(_agx(0.18, operator="filmic_aces")[0]) - 0.5486) < 5e-3


# ── 5. Alpha is not a colour ────────────────────────────────────────────────

@pytest.mark.real_torch
@pytest.mark.parametrize("alpha", [1.0, 0.5, 0.25, 0.0])
def test_tone_mapping_leaves_alpha_alone(alpha):
    """Measured before: 1.0 -> 0.72974, 0.5 -> 0.60691. A plate came back 27%
    transparent, and a 50% matte 21% too opaque."""
    from radiance.hdr.tonemap import HDRToneMap

    rgba = torch.full((1, 4, 4, 4), 0.5)
    rgba[..., 3] = alpha
    out = HDRToneMap().tonemap(
        rgba, preset="Custom", operator="reinhard", exposure=0.0, gamma=2.2,
        white_point=1.0, contrast=1.3, saturation=1.2,
        highlight_compression=0.5, shadow_lift=0.1, use_gpu=False,
    )[0]
    assert out.shape[-1] == 4
    assert abs(float(out[0, 0, 0, 3]) - alpha) < 1e-6


@pytest.mark.real_torch
@pytest.mark.parametrize("alpha", [1.0, 0.5, 0.0])
def test_dynamic_range_expansion_leaves_alpha_alone(alpha):
    """Measured before: 0.5 -> 0.21404, via tensor_srgb_to_linear on all four."""
    from radiance.hdr.tonemap import HDRExpandDynamicRange

    rgba = torch.full((1, 4, 4, 4), 0.5)
    rgba[..., 3] = alpha
    out = HDRExpandDynamicRange().expand(
        rgba, source_gamma=2.2, highlight_recovery=1.0, black_point=0.0,
        target_stops=14.0, highlight_rolloff=1.5,
    )[0]
    assert out.shape[-1] == 4
    assert abs(float(out[0, 0, 0, 3]) - alpha) < 1e-6


@pytest.mark.real_torch
@pytest.mark.parametrize("alpha", [1.0, 0.5, 0.25])
def test_the_delivery_grade_leaves_alpha_alone(alpha):
    """
    exposure, contrast and shadows/highlights all ran on the full array — lift,
    gain, gamma, saturation and gamut compression were already guarded, these
    three were not. Measured with exposure +1, contrast 1.3, shadows +0.4,
    highlights -0.3: alpha 1.0 -> 2.6509, 0.5 -> 1.0591, 0.25 -> 0.5066.
    """
    from radiance.color.grading import apply_grading

    rgba = np.full((4, 4, 4), 0.5, dtype=np.float32)
    rgba[..., 3] = alpha
    out = apply_grading(rgba, exposure=1.0, contrast=1.3, shadows=0.4,
                        highlights=-0.3, saturation=1.4)
    assert abs(float(out[0, 0, 3]) - alpha) < 1e-6


@pytest.mark.real_torch
def test_rgb_only_input_still_works_everywhere():
    """The alpha split must not change the 3-channel path."""
    from radiance.color.grading import apply_grading
    from radiance.hdr.tonemap import HDRExpandDynamicRange, HDRToneMap

    rgb_t = torch.full((1, 4, 4, 3), 0.5)
    assert HDRToneMap().tonemap(
        rgb_t, preset="Custom", operator="reinhard", exposure=0.0, gamma=2.2,
        white_point=1.0, contrast=1.0, saturation=1.0,
        highlight_compression=0.0, shadow_lift=0.0, use_gpu=False,
    )[0].shape[-1] == 3
    assert HDRExpandDynamicRange().expand(
        rgb_t, source_gamma=2.2, highlight_recovery=1.0, black_point=0.0,
        target_stops=14.0, highlight_rolloff=1.5,
    )[0].shape[-1] == 3
    out = apply_grading(np.full((4, 4, 3), 0.5, dtype=np.float32), exposure=1.0,
                        contrast=1.3, shadows=0.4)
    assert out.shape[-1] == 3
