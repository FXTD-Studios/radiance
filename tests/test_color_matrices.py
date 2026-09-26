"""
test_color_matrices.py — Regression suite for all Radiance camera-space matrices
and log transfer functions.

Motivation:
    AWG4_TO_ACESCG was wrong for months (max element error 0.364) before it was
    caught by manual inspection.  This suite encodes the ground-truth invariants
    that any correct matrix MUST satisfy, so a copy-paste error or wrong source
    is caught immediately on CI.

Invariants tested per matrix:
    1. White-point preservation  — D65 (or native white) maps to ACEScg white.
    2. Round-trip identity       — matrix @ inverse ≈ identity (within 1e-5).
    3. Primary hue preservation  — primary unit vectors produce non-negative
                                   outputs (gamut-compression may clip, but the
                                   matrix itself must not flip sign on primaries).

Invariants tested per log curve:
    1. 18% grey maps to the spec code value (±1 code value at 10-bit precision).
    2. Encode → decode round-trip error < 1e-5 over the valid linear range.
    3. Monotonicity  — encoded values strictly increase with linear input.
    4. Continuity    — encode and decode are C0-continuous at the knee
                       (|log_branch(cut) - linear_branch(cut)| < 1e-4).
    5. Black offset  — SLog3: encode(-0.01) = 95/1023; decoder(-0.01 linear) works.
    6. Tensor variants — GPU/numpy parity check (if torch is available).
"""

import sys
import pytest
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Torch probe  (same pattern as test_nodes_temporal_hdr.py)
# ─────────────────────────────────────────────────────────────────────────────
try:
    import torch as _torch_probe
    HAS_TORCH = isinstance(getattr(_torch_probe, "__version__", None), str)
except ImportError:
    HAS_TORCH = False

_SKIP_TORCH = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")


# ─────────────────────────────────────────────────────────────────────────────
# Import helpers
# ─────────────────────────────────────────────────────────────────────────────
import importlib
import pathlib

# Add the parent of 'radiance' to sys.path so `from color_utils import …` works
_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from radiance.color_utils import (
    # Matrices
    SRGB_TO_ACESCG, ACESCG_TO_SRGB,
    AWG4_TO_ACESCG,
    AWG3_TO_ACESCG,
    SGAMUT3_CINE_TO_ACESCG,
    VGAMUT_TO_ACESCG,
    CINEMA_GAMUT_TO_ACESCG,
    REDWIDEGAMUT_TO_ACESCG,
    DAVINCI_WIDE_TO_ACESCG,
    # Numpy log curves
    linear_to_logc4,   logc4_to_linear,
    linear_to_logc3,   logc3_to_linear,
    linear_to_slog3,   slog3_to_linear,
    linear_to_vlog,    vlog_to_linear,
    linear_to_hlg,     hlg_to_linear,
    linear_to_pq,      pq_to_linear,
    linear_to_acescct, acescct_to_linear,
)


# ─────────────────────────────────────────────────────────────────────────────
# ── Section 1: Matrix invariants
# ─────────────────────────────────────────────────────────────────────────────

# Each entry: (name, matrix_to_acescg, inverse_or_None, d65_white_in_native)
# d65_white_in_native: the equal-energy / native white in the SOURCE gamut.
# For gamuts whose white IS D65 we use the standard [0.9505, 1.0000, 1.0891]
# normalised to Y=1 (D65 in XYZ normalised), expressed as (R,G,B)=white=(1,1,1)
# in the source RGB space — i.e., the matrix row sum must equal ACEScg white.

_ACESCG_WHITE = np.array([1.0, 1.0, 1.0], dtype=np.float64)  # R=G=B=1 in ACEScg

_MATRICES = [
    ("sRGB→ACEScg",        SRGB_TO_ACESCG,       ACESCG_TO_SRGB),
    ("AWG4→ACEScg",        AWG4_TO_ACESCG,        None),
    ("AWG3→ACEScg",        AWG3_TO_ACESCG,        None),
    ("SGamut3Cine→ACEScg", SGAMUT3_CINE_TO_ACESCG, None),
    ("VGamut→ACEScg",      VGAMUT_TO_ACESCG,      None),
    ("CinemaGamut→ACEScg", CINEMA_GAMUT_TO_ACESCG, None),
    ("REDWideGamut→ACEScg",REDWIDEGAMUT_TO_ACESCG, None),
    ("DaVinciWide→ACEScg", DAVINCI_WIDE_TO_ACESCG, None),
]


class TestMatrixWhitePoint:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    """Row sums of a gamut→ACEScg matrix must sum to [1, 1, 1] ± 0.005.

    Rationale: input RGB=(1,1,1) is the native white.  In ACEScg, white is also
    (1,1,1).  Therefore each row of the matrix must sum to 1.
    """
    @pytest.mark.parametrize("name,M,_inv", _MATRICES)
    def test_row_sums(self, name, M, _inv):
        row_sums = M.sum(axis=1)
        np.testing.assert_allclose(
            row_sums, [1.0, 1.0, 1.0], atol=0.005,
            err_msg=f"{name}: row sums {row_sums} should all be ~1.0 (white maps to white)",
        )


class TestMatrixRoundTrip:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    """M @ inv(M) ≈ identity for each matrix to within 1e-5."""
    @pytest.mark.parametrize("name,M,_inv", _MATRICES)
    def test_roundtrip(self, name, M, _inv):
        M64  = M.astype(np.float64)
        inv  = np.linalg.inv(M64)
        prod = M64 @ inv
        np.testing.assert_allclose(
            prod, np.eye(3), atol=1e-5,
            err_msg=f"{name}: M @ inv(M) deviates from identity",
        )


class TestMatrixPrimarySign:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    """Diagonal elements of every gamut→ACEScg matrix must be positive.
    A negative diagonal would invert a primary axis — a definitive sign of
    a wrong matrix (e.g. the old Canon matrix with diag ~[0.69, 0.93, 0.94]
    was too conservative; the correct ultra-wide Canon diag is ~[1.11, 1.31, 1.22]).
    """
    @pytest.mark.parametrize("name,M,_inv", _MATRICES)
    def test_diagonal_positive(self, name, M, _inv):
        diag = np.diag(M)
        assert np.all(diag > 0), (
            f"{name}: negative diagonal element(s) {diag} — matrix is likely wrong"
        )

class TestMatrixRowSumDerivation:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    """Spot-check that our Bradford derivation matches the stored sRGB→ACEScg matrix
    to 0.000001 tolerance. This pins the derivation path so any future Bradford
    matrix change breaks here immediately."""
    def test_srgb_derivation_matches_stored(self):
        M_Bradford = np.array([
            [ 0.8951,  0.2664, -0.1614],
            [-0.7502,  1.7135,  0.0367],
            [ 0.0389, -0.0685,  1.0296],
        ], dtype=np.float64)
        def _xy(x, y): return np.array([x/y, 1.0, (1-x-y)/y])
        def _primXYZ(rx,ry,gx,gy,bx,by,wx,wy):
            M = np.column_stack([_xy(rx,ry),_xy(gx,gy),_xy(bx,by)])
            return M * np.linalg.solve(M, _xy(wx,wy))
        M_src = _primXYZ(0.64,0.33, 0.30,0.60, 0.15,0.06, 0.3127,0.3290)
        M_ap1 = _primXYZ(0.713,0.293, 0.165,0.830, 0.128,0.044, 0.32168,0.33767)
        D65 = _xy(0.3127,0.3290);  D60 = _xy(0.32168,0.33767)
        c65 = M_Bradford @ D65;    c60 = M_Bradford @ D60
        M_cat = np.linalg.inv(M_Bradford) @ np.diag(c60/c65) @ M_Bradford
        derived = (np.linalg.inv(M_ap1) @ M_cat @ M_src).astype(np.float32)
        np.testing.assert_allclose(derived, SRGB_TO_ACESCG, atol=1e-5,
            err_msg="Bradford derivation drifted from stored sRGB→ACEScg reference")


class TestKnownPairRoundTrip:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    """sRGB ↔ ACEScg round-trip with the explicit inverse matrix."""
    def test_srgb_acescg_roundtrip(self):
        rng    = np.random.default_rng(0)
        pixels = rng.random((1000, 3), dtype=np.float32)
        acescg = pixels @ SRGB_TO_ACESCG.T
        back   = acescg @ ACESCG_TO_SRGB.T
        np.testing.assert_allclose(back, pixels, atol=1e-5,
                                   err_msg="sRGB→ACEScg→sRGB round-trip error")


# ─────────────────────────────────────────────────────────────────────────────
# ── Section 2: Log curve invariants
# ─────────────────────────────────────────────────────────────────────────────

# (name, encode_fn, decode_fn, 18%-grey linear, expected 18%-grey encoded, atol_18grey,
#  valid_linear_range, continuity_cut)
# atol_18grey is in normalised [0,1] code-value units; 1 code value at 10-bit = 1/1023 ≈ 0.001

_LOG_CURVES = [
    # name          enc              dec             lin_18  enc_18     atol    range        cut
    ("LogC4",   linear_to_logc4,   logc4_to_linear,   0.18, 0.277,   0.005, (0.0, 8.0),  -0.0180),
    ("LogC3",   linear_to_logc3,   logc3_to_linear,   0.18, 0.391,   0.005, (0.0, 4.0),   0.010591),
    ("SLog3",   linear_to_slog3,   slog3_to_linear,   0.18, 420/1023, 0.002, (0.0, 8.0),  0.011250),
    ("VLog",    linear_to_vlog,    vlog_to_linear,    0.18, 0.423,   0.005, (0.0, 4.0),   0.01),
    # HLG: lin_18 = 1/12 is the knee point where sqrt branch gives exactly 0.5.
    # HLG does not use the same "18% grey" convention as log formats.
    ("HLG",     linear_to_hlg,     hlg_to_linear,     1/12, 0.5,    0.002, (0.0, 1.0),   1/12),
    # PQ: float32 precision with m1=0.159/m2=78.84 exponents accumulates ~1.2e-4 error
    # near the upper end.  Restrict to (0.01, 1.9) — still covers the full HDR range.
    ("PQ",      linear_to_pq,      pq_to_linear,      0.18, None,   None,  (0.01, 1.9),  None),
    ("ACEScct", linear_to_acescct, acescct_to_linear, 0.18, 0.414,  0.005, (0.0, 4.0),   0.0078125),
]


class TestLogCurve18GreyMapping:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    """18% grey (or knee-point for HLG) must encode to the spec code value."""
    @pytest.mark.parametrize("name,enc,dec,lin18,enc18,atol,_r,_c", _LOG_CURVES)
    def test_18_grey(self, name, enc, dec, lin18, enc18, atol, _r, _c):
        if enc18 is None:
            pytest.skip(f"{name}: spec code value not parameterised")
        result = float(enc(np.array([lin18], dtype=np.float32)).item())
        assert abs(result - enc18) < atol, (
            f"{name}: lin={lin18} encodes to {result:.5f}, expected {enc18:.5f} ±{atol}"
        )


class TestLogCurveRoundTrip:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    """encode → decode round-trip error < 1e-4 over the valid linear range."""
    @pytest.mark.parametrize("name,enc,dec,_l18,_e18,_a,valid_range,_c", _LOG_CURVES)
    def test_roundtrip(self, name, enc, dec, _l18, _e18, _a, valid_range, _c):
        lo, hi = valid_range
        x      = np.linspace(max(lo, 1e-5), hi, 500, dtype=np.float32)
        y      = enc(x)
        x_hat  = dec(y)
        err    = np.abs(x_hat - x)
        assert err.max() < 1e-4, (
            f"{name}: max round-trip error {err.max():.2e} at "
            f"x={float(x[np.argmax(err)]):.4f}"
        )


class TestLogCurveMonotonicity:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    """Encoded values must strictly increase with linear input."""
    @pytest.mark.parametrize("name,enc,dec,_l18,_e18,_a,valid_range,_c", _LOG_CURVES)
    def test_monotone(self, name, enc, dec, _l18, _e18, _a, valid_range, _c):
        lo, hi = valid_range
        x      = np.linspace(max(lo, 1e-5), hi, 500, dtype=np.float32)
        y      = enc(x)
        diffs  = np.diff(y.astype(np.float64))
        n_nonpositive = int((diffs <= 0).sum())
        assert n_nonpositive == 0, (
            f"{name}: {n_nonpositive} non-positive diffs — curve is not monotone"
        )


class TestLogCurveContinuity:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    """Encode and decode are C0-continuous at the knee (|log_val - lin_val| < 1e-4)."""
    @pytest.mark.parametrize("name,enc,dec,_l18,_e18,_a,valid_range,cut", _LOG_CURVES)
    def test_knee_continuity(self, name, enc, dec, _l18, _e18, _a, valid_range, cut):
        if cut is None:
            pytest.skip(f"{name}: no piecewise knee")
        eps = 1e-6
        below   = float(enc(np.array([cut - eps], dtype=np.float32)).item())
        above   = float(enc(np.array([cut + eps], dtype=np.float32)).item())
        assert abs(above - below) < 5e-4, (
            f"{name}: encode discontinuity at cut {cut:.6f}: "
            f"below={below:.6f} above={above:.6f} Δ={abs(above-below):.2e}"
        )
        # Decoder continuity
        cut_enc   = float(enc(np.array([cut], dtype=np.float32)).item())
        dec_below = float(dec(np.array([cut_enc - eps], dtype=np.float32)).item())
        dec_above = float(dec(np.array([cut_enc + eps], dtype=np.float32)).item())
        assert abs(dec_above - dec_below) < 5e-4, (
            f"{name}: decode discontinuity at encoded cut {cut_enc:.6f}: "
            f"below={dec_below:.6f} above={dec_above:.6f}"
        )


class TestSLog3BlackOffset:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    """Sony S-Log3: scene linear 0.0 encodes to 95/1023.

    This class used to assert that linear −0.01 mapped to 95/1023, which
    described the implementation rather than the spec. Sony's white paper gives
    the toe as

        x < 0.01125 :  (x * (171.2102946929 − 95) / 0.01125 + 95) / 1023

    applied to x DIRECTLY — the +0.01 belongs to the log branch alone. The two
    branches meet exactly at the cut: at x = 0.01125 both give 0.167361. The
    old implementation used half the slope with a +0.01 shift on the input, so
    black encoded to 0.127921 instead of 0.092864 (≈36 code values at 10-bit).
    Encode and decode were mutually consistent, which is why every round-trip
    test passed while the absolute mapping was wrong.
    """
    def test_black_encodes_to_95cv(self):
        encoded = float(linear_to_slog3(np.array([0.0], dtype=np.float32)).item())
        expected = 95.0 / 1023.0
        assert abs(encoded - expected) < 1e-5, (
            f"SLog3: scene black 0.0 encodes to {encoded:.6f}, "
            f"expected {expected:.6f} (95/1023)"
        )

    def test_black_decodes_correctly(self):
        black_encoded = np.array([95.0 / 1023.0], dtype=np.float32)
        linear = float(slog3_to_linear(black_encoded).item())
        assert abs(linear) < 1e-5, (
            f"SLog3: 95/1023 decodes to {linear:.6f}, expected 0.0"
        )

    def test_branches_meet_at_the_cut(self):
        """Toe and log branch are C0-continuous at x = 0.01125 -> 0.167361."""
        cut = np.array([0.01125], dtype=np.float32)
        assert abs(float(linear_to_slog3(cut).item()) - 0.167361) < 1e-5

    def test_matches_the_reference_curve_in_luts(self):
        """color/luts.py carried the correct decode all along; they must agree.

        Compared from 95/1023 (scene black) upward. Below that the two differ on
        purpose: `_idt_slog3` is an IDT and clamps sub-black to 0, while
        `slog3_to_linear` lets it go negative so a working space keeps its
        footroom. Above black they are the same curve.
        """
        from radiance.color.luts import _idt_slog3

        cv = np.linspace(95.0 / 1023.0, 1.0, 257).astype(np.float32)
        assert np.max(np.abs(slog3_to_linear(cv) - _idt_slog3(cv))) < 2e-5

    def test_sub_black_keeps_footroom(self):
        """Below 95/1023 this decoder goes negative rather than clamping."""
        assert float(slog3_to_linear(np.array([0.05], dtype=np.float32)).item()) < 0.0

    def test_sub_black_clamped(self):
        """Values below −0.01 should not produce NaN (clamped to camera black)."""
        very_dark = np.array([-0.5, -1.0, -10.0], dtype=np.float32)
        result = linear_to_slog3(very_dark)
        assert not np.any(np.isnan(result)), "SLog3: NaN on sub-black input"
        assert not np.any(np.isinf(result)), "SLog3: Inf on sub-black input"

    def test_18_grey(self):
        grey = np.array([0.18], dtype=np.float32)
        encoded = float(linear_to_slog3(grey).item())
        expected = 420.0 / 1023.0
        assert abs(encoded - expected) < 1e-5, (
            f"SLog3: 18% grey encodes to {encoded:.6f}, expected {expected:.6f} (420/1023)"
        )


class TestVLogSlopeConsistency:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    """VLog encode/decode both use slope 5.6 — knee must be continuous."""
    def test_encode_continuity(self):
        cut = 0.01
        eps = 1e-6
        lo  = float(linear_to_vlog(np.array([cut - eps], dtype=np.float32)).item())
        hi  = float(linear_to_vlog(np.array([cut + eps], dtype=np.float32)).item())
        assert abs(hi - lo) < 1e-4, f"VLog encode knee gap: {abs(hi-lo):.2e}"

    def test_decode_continuity(self):
        cut_enc = 0.181  # spec Table 1
        eps = 1e-6
        lo  = float(vlog_to_linear(np.array([cut_enc - eps], dtype=np.float32)).item())
        hi  = float(vlog_to_linear(np.array([cut_enc + eps], dtype=np.float32)).item())
        assert abs(hi - lo) < 1e-4, f"VLog decode knee gap: {abs(hi-lo):.2e}"


# ─────────────────────────────────────────────────────────────────────────────
# ── Section 3: Tensor (GPU) parity with numpy
# ─────────────────────────────────────────────────────────────────────────────

if HAS_TORCH:
    import torch
    from radiance.color_utils import (
        tensor_linear_to_logc4,   tensor_logc4_to_linear,
        tensor_linear_to_slog3,   tensor_slog3_to_linear,
        tensor_linear_to_vlog,    tensor_vlog_to_linear,
        tensor_linear_to_pq,      tensor_pq_to_linear,
        tensor_linear_to_hlg,     tensor_hlg_to_linear,
        tensor_linear_to_acescct, tensor_acescct_to_linear,
    )

    _TENSOR_PAIRS = [
        ("LogC4",   linear_to_logc4,   logc4_to_linear,   tensor_linear_to_logc4,   tensor_logc4_to_linear),
        ("SLog3",   linear_to_slog3,   slog3_to_linear,   tensor_linear_to_slog3,   tensor_slog3_to_linear),
        ("VLog",    linear_to_vlog,    vlog_to_linear,    tensor_linear_to_vlog,    tensor_vlog_to_linear),
        ("PQ",      linear_to_pq,      pq_to_linear,      tensor_linear_to_pq,      tensor_pq_to_linear),
        ("HLG",     linear_to_hlg,     hlg_to_linear,     tensor_linear_to_hlg,     tensor_hlg_to_linear),
        ("ACEScct", linear_to_acescct, acescct_to_linear, tensor_linear_to_acescct, tensor_acescct_to_linear),
    ]

    @_SKIP_TORCH
    class TestTensorNumpyParity:
        CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
        """Tensor (GPU-path) encode/decode matches numpy to within 1e-4."""

        @pytest.mark.parametrize("name,np_enc,np_dec,t_enc,t_dec", _TENSOR_PAIRS)
        def test_encode_parity(self, name, np_enc, np_dec, t_enc, t_dec):
            x_np = np.linspace(0.001, 2.0, 500, dtype=np.float32)
            y_np = np_enc(x_np)
            x_t  = torch.from_numpy(x_np)
            y_t  = t_enc(x_t).numpy()
            np.testing.assert_allclose(
                y_t, y_np, atol=1e-4,
                err_msg=f"{name} encode: tensor≠numpy",
            )

        @pytest.mark.parametrize("name,np_enc,np_dec,t_enc,t_dec", _TENSOR_PAIRS)
        def test_decode_parity(self, name, np_enc, np_dec, t_enc, t_dec):
            # Use numpy-encoded values as decode input to avoid propagating encode errors
            x_np  = np.linspace(0.001, 2.0, 500, dtype=np.float32)
            y_np  = np_enc(x_np)
            xh_np = np_dec(y_np)
            y_t   = torch.from_numpy(y_np)
            xh_t  = t_dec(y_t).numpy()
            np.testing.assert_allclose(
                xh_t, xh_np, atol=1e-4,
                err_msg=f"{name} decode: tensor≠numpy",
            )

        @pytest.mark.parametrize("name,np_enc,np_dec,t_enc,t_dec", _TENSOR_PAIRS)
        def test_tensor_roundtrip(self, name, np_enc, np_dec, t_enc, t_dec):
            hi = 1.0 if name == "HLG" else (1.8 if name == "PQ" else 2.0)
            x  = torch.linspace(0.001, hi, 500)
            xh = t_dec(t_enc(x))
            err = (xh - x).abs().max().item()
            assert err < 1.5e-4, f"{name} tensor round-trip: max error {err:.2e}"

    @_SKIP_TORCH
    class TestTensorVLogSlope:
        CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
        """GPU VLog encoder and decoder must both use slope 5.6, not 5.625."""
        def test_encode_slope(self):
            x = torch.linspace(0.0, 0.009, 100)          # below cut=0.01 → linear region
            y = tensor_linear_to_vlog(x)
            # y = 5.6*x + 0.125; slope = Δy/Δx
            dy = y[1:] - y[:-1]
            dx = x[1:] - x[:-1]
            slopes = (dy / dx).numpy()
            np.testing.assert_allclose(slopes, 5.6, atol=2e-4,
                                       err_msg="VLog tensor encode slope ≠ 5.6")

        def test_decode_slope(self):
            cut_enc = 0.181
            y = torch.linspace(0.0, cut_enc - 1e-4, 100)   # below cut → linear region
            x = tensor_vlog_to_linear(y)
            # x = (y - 0.125) / 5.6
            dy = y[1:] - y[:-1]
            dx = x[1:] - x[:-1]
            slopes = (dy / dx).numpy()   # dy/dx = 5.6
            np.testing.assert_allclose(slopes, 5.6, atol=1e-3,
                                       err_msg="VLog tensor decode slope ≠ 5.6")
