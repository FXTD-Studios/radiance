"""
tests/test_exr_roundtrip.py
────────────────────────────
EXR round-trip fidelity tests for the Radiance soft-knee HDR codec.

What "EXR round-trip" means here
─────────────────────────────────
A real EXR file stores float32 scene-linear RGB.  The Radiance HDR pipeline
compresses those values into [0, 1] for VAE encoding, then decompresses after
VAE decode.  We measure the *codec* fidelity (compress → decompress) — the
VAE itself is an external model with its own reconstruction error budget.

This suite:
  1. Builds synthetic HDR scenes that match real EXR content:
       • Overcast sky    0.0 – 0.3 linear
       • Interior        0.0 – 1.0 linear (display-referred)
       • Golden-hour     0.0 – 4.0 linear (1–2 stops over)
       • Fire / flame    0.0 – 8.0 linear (3 stops over)
       • Midday sun      0.0 – 16.0 linear (4 stops over)
       • Arc-welder      0.0 – 64.0 linear (6 stops over, pathological)

  2. Runs compress → decompress for compression_ratio ∈ {0.0, 0.25, 0.5,
     0.75, 1.0}.

  3. Verifies:
       • PSNR ≥ target_dB for each (scene, ratio) pair (pure numpy, no torch)
       • Monotonicity: higher ratio → less clipping of HDR content
       • Zero-preservation: black pixels stay black
       • Midtone fidelity: 18% grey within 0.5 % error for ratio ≤ 0.5
       • No NaN / Inf at any ratio

All tests run in pure numpy — no torch, no VAE, no GPU required.
Torch-conditional variants are included for regression on the node classes.
"""

import math
import numpy as np
import pytest

# ── Torch-conditional import ─────────────────────────────────────────────────
# Use __version__ presence to distinguish real torch from the test stub
# installed by conftest.py (the stub has no __version__).
try:
    import torch as _torch_probe
    HAS_TORCH = isinstance(getattr(_torch_probe, "__version__", None), str)
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    import torch
    from radiance.nodes.hdr.encoder import _hdr_soft_compress, _hdr_soft_decompress

# ─────────────────────────────────────────────────────────────────────────────
# Pure-numpy reference implementations (match the torch originals exactly)
# ─────────────────────────────────────────────────────────────────────────────

def _np_compress(img: np.ndarray, ratio: float) -> np.ndarray:
    """Numpy mirror of _hdr_soft_compress."""
    x = np.maximum(img, 0.0)
    reinhard = x / (1.0 + x)
    clamped  = np.minimum(x, 1.0)
    if ratio <= 0.0:
        return clamped
    if ratio >= 1.0:
        return reinhard
    return clamped * (1.0 - ratio) + reinhard * ratio


def _np_decompress(img: np.ndarray, ratio: float) -> np.ndarray:
    """Numpy mirror of _hdr_soft_decompress (two-regime analytical inverse)."""
    r   = float(ratio)
    eps = 1e-7
    y   = np.clip(img, 0.0, 1.0 - eps)

    if r <= 0.0:
        return y
    if r >= 1.0:
        return y / (1.0 - y + eps)

    y_break     = 1.0 - r * 0.5
    x_hdr       = (y - 1.0 + r) / (1.0 - y + eps)
    one_minus_r = 1.0 - r
    one_minus_y = 1.0 - y
    disc        = one_minus_y ** 2 + 4.0 * one_minus_r * y
    x_sdr       = (-one_minus_y + np.sqrt(np.maximum(disc, 0.0))) / (2.0 * one_minus_r + eps)
    return np.where(y > y_break, x_hdr, x_sdr)


def _psnr(original: np.ndarray, reconstructed: np.ndarray, peak: float) -> float:
    """
    Peak Signal-to-Noise Ratio in dB.

    PSNR = 10 · log10(peak² / MSE)

    For HDR content we use the *scene peak* as the reference so we're
    measuring fidelity over the full dynamic range, not just [0,1].
    """
    mse = float(np.mean((original - reconstructed) ** 2))
    if mse == 0.0:
        return float("inf")
    return 10.0 * math.log10((peak ** 2) / mse)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic HDR scene generators
# ─────────────────────────────────────────────────────────────────────────────

# (name, peak_linear, target_psnr_db per ratio)
# PSNR targets are conservative — the codec intentionally sacrifices some
# highlight precision to fit the VAE's [0,1] domain.
SCENES = [
    # name              peak      r=0    r=0.25  r=0.5  r=0.75  r=1.0
    #
    # r=0 (hard clamp): PSNR is LOW for HDR scenes because all content above 1.0
    # is clipped to 1.0 and the decompress is identity — those highlights are
    # permanently lost.  This is intentional: ratio=0 is the "no tone-mapping"
    # mode, not a round-trip codec.  Targets reflect the actual MSE from clipping.
    #
    ("overcast_sky",    0.30,   [99.0,  99.0,   99.0,  99.0,   99.0]),
    ("interior_sdr",    1.00,   [99.0,  55.0,   50.0,  46.0,   42.0]),
    ("golden_hour",     4.00,   [ 8.0,  35.0,   40.0,  44.0,   44.0]),
    ("fire_flame",      8.00,   [ 6.0,  28.0,   36.0,  40.0,   42.0]),
    ("midday_sun",     16.00,   [ 5.0,  22.0,   32.0,  38.0,   41.0]),
    ("arc_welder",     64.00,   [ 3.0,  16.0,   26.0,  34.0,   39.0]),
]

RATIOS = [0.0, 0.25, 0.5, 0.75, 1.0]

RNG = np.random.default_rng(seed=42)


def _make_hdr_image(peak: float, shape=(64, 64, 3)) -> np.ndarray:
    """
    Synthetic scene-linear HDR image matching real EXR content distribution.

    Uses a mixture of:
      • Dark shadows     (0 – 0.05 × peak)
      • Mid-tones        (0.05 – 0.5 × peak)
      • Bright highlights (0.5 – peak)

    Returned as float32, shape (H, W, C), values in [0, peak].
    """
    H, W, C = shape
    N = H * W * C

    # Stratified sampling across the dynamic range
    shadow     = RNG.uniform(0.0,         0.05 * peak, size=N // 3).astype(np.float32)
    midtone    = RNG.uniform(0.05 * peak, 0.5  * peak, size=N // 3).astype(np.float32)
    highlight  = RNG.uniform(0.5  * peak, peak,        size=N - 2 * (N // 3)).astype(np.float32)

    flat = np.concatenate([shadow, midtone, highlight])
    RNG.shuffle(flat)
    return flat.reshape(H, W, C)


# ─────────────────────────────────────────────────────────────────────────────
# PSNR round-trip tests  (pure numpy, always run)
# ─────────────────────────────────────────────────────────────────────────────

class TestPSNRRoundtrip:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    """
    Parametric PSNR tests across 6 HDR scene types × 5 compression ratios.
    Each test checks that the codec meets the minimum PSNR target.
    """

    @pytest.mark.parametrize("scene,peak,psnr_targets", SCENES)
    @pytest.mark.parametrize("ratio_idx,ratio", enumerate(RATIOS))
    def test_psnr_meets_target(self, scene, peak, psnr_targets, ratio_idx, ratio):
        img   = _make_hdr_image(peak)
        comp  = _np_compress(img, ratio)
        recon = _np_decompress(comp, ratio)
        db    = _psnr(img, recon, peak)
        target = psnr_targets[ratio_idx]
        assert db >= target, (
            f"scene={scene}  ratio={ratio}  PSNR={db:.1f} dB < target {target} dB"
        )

    @pytest.mark.parametrize("scene,peak,_", SCENES)
    def test_no_nan_or_inf(self, scene, peak, _):
        img = _make_hdr_image(peak)
        for ratio in RATIOS:
            comp  = _np_compress(img, ratio)
            recon = _np_decompress(comp, ratio)
            assert not np.isnan(recon).any(),  f"NaN  in recon: scene={scene} ratio={ratio}"
            assert not np.isinf(recon).any(),  f"Inf  in recon: scene={scene} ratio={ratio}"


class TestCodecProperties:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"

    def test_black_preserved_all_ratios(self):
        """Pixel value = 0 must reconstruct as 0 for all ratios."""
        black = np.zeros((1, 1, 3), dtype=np.float32)
        for ratio in RATIOS:
            recon = _np_decompress(_np_compress(black, ratio), ratio)
            assert float(np.abs(recon).max()) < 1e-6, \
                f"Black not preserved at ratio={ratio}: got {recon.max()}"

    def test_compress_output_in_unit_range(self):
        """Compressed values must always be in [0, 1]."""
        for scene, peak, _ in SCENES:
            img  = _make_hdr_image(peak)
            for ratio in RATIOS:
                comp = _np_compress(img, ratio)
                assert float(comp.min()) >= -1e-6, \
                    f"Compressed below 0: scene={scene} ratio={ratio} min={comp.min()}"
                assert float(comp.max()) <= 1.0 + 1e-6, \
                    f"Compressed above 1: scene={scene} ratio={ratio} max={comp.max()}"

    def test_midtone_18pct_grey_fidelity(self):
        """
        18% grey (scene-linear 0.18) must reconstruct within 0.5% for
        compression_ratio ≤ 0.5 (low-compression regime).
        """
        grey = np.full((1, 1, 3), 0.18, dtype=np.float32)
        for ratio in [0.0, 0.25, 0.5]:
            comp  = _np_compress(grey, ratio)
            recon = _np_decompress(comp, ratio)
            err   = abs(float(recon.mean()) - 0.18) / 0.18
            assert err < 0.005, \
                f"18% grey error {err*100:.3f}% > 0.5% at ratio={ratio}"

    def test_monotone_highlight_recovery(self):
        """
        Higher compression_ratio must recover more highlight energy.
        For a 4-stop overexposed image, PSNR should increase monotonically
        with ratio (less clipping loss → better reconstruction).
        """
        img = _make_hdr_image(peak=4.0)
        prev_psnr = -1.0
        for ratio in RATIOS:
            comp  = _np_compress(img, ratio)
            recon = _np_decompress(comp, ratio)
            db    = _psnr(img, recon, peak=4.0)
            assert db >= prev_psnr - 1.0, (
                f"PSNR dropped unexpectedly at ratio={ratio}: "
                f"{db:.1f} dB < prev {prev_psnr:.1f} dB"
            )
            prev_psnr = db

    def test_round_trip_identity_ratio_zero(self):
        """
        ratio=0 is hard-clamp. For SDR image (peak ≤ 1.0), the round-trip
        must be near-lossless (PSNR ≥ 60 dB).
        """
        img  = _make_hdr_image(peak=1.0)
        comp = _np_compress(img, 0.0)
        recon = _np_decompress(comp, 0.0)
        db   = _psnr(img, recon, peak=1.0)
        assert db >= 60.0, f"SDR identity (ratio=0) PSNR {db:.1f} dB < 60 dB"

    def test_reinhard_round_trip_ratio_one(self):
        """
        ratio=1 is full Reinhard. decompress(compress(x)) ≈ x everywhere
        (Reinhard is analytically invertible).  PSNR ≥ 55 dB for 4-stop HDR.
        """
        img   = _make_hdr_image(peak=4.0)
        comp  = _np_compress(img, 1.0)
        recon = _np_decompress(comp, 1.0)
        db    = _psnr(img, recon, peak=4.0)
        assert db >= 55.0, f"Reinhard round-trip PSNR {db:.1f} dB < 55 dB"

    def test_compress_is_monotone_increasing(self):
        """compress(x) must be monotone — larger x → larger y."""
        x = np.linspace(0.0, 10.0, 500, dtype=np.float32)
        for ratio in RATIOS:
            y = _np_compress(x.reshape(-1, 1, 1), ratio).flatten()
            diffs = np.diff(y)
            assert (diffs >= -1e-7).all(), \
                f"compress not monotone at ratio={ratio}: min diff={diffs.min()}"

    def test_decompress_is_monotone_increasing(self):
        """decompress(y) must be monotone — larger y → larger x."""
        y = np.linspace(0.0, 0.999, 500, dtype=np.float32)
        for ratio in RATIOS:
            x = _np_decompress(y.reshape(-1, 1, 1), ratio).flatten()
            diffs = np.diff(x)
            assert (diffs >= -1e-7).all(), \
                f"decompress not monotone at ratio={ratio}: min diff={diffs.min()}"

    def test_compress_preserves_relative_order(self):
        """Two pixels with x1 < x2 must satisfy compress(x1) ≤ compress(x2)."""
        x1 = np.array([[[0.1, 0.5, 2.0]]], dtype=np.float32)
        x2 = np.array([[[0.2, 1.0, 4.0]]], dtype=np.float32)
        for ratio in RATIOS:
            c1 = _np_compress(x1, ratio)
            c2 = _np_compress(x2, ratio)
            assert (c1 <= c2 + 1e-7).all(), \
                f"Relative order not preserved at ratio={ratio}"


class TestSceneSpecificPSNR:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    """Named PSNR assertions for clarity — supplements parametric tests."""

    def test_interior_sdr_ratio05_psnr_50db(self):
        """Interior SDR scene at ratio=0.5 → PSNR ≥ 50 dB."""
        img   = _make_hdr_image(peak=1.0)
        comp  = _np_compress(img, 0.5)
        recon = _np_decompress(comp, 0.5)
        db    = _psnr(img, recon, peak=1.0)
        assert db >= 50.0, f"Interior SDR PSNR {db:.1f} dB < 50 dB"

    def test_fire_scene_ratio075_psnr_40db(self):
        """Fire scene (8× overexposed) at ratio=0.75 → PSNR ≥ 40 dB."""
        img   = _make_hdr_image(peak=8.0)
        comp  = _np_compress(img, 0.75)
        recon = _np_decompress(comp, 0.75)
        db    = _psnr(img, recon, peak=8.0)
        assert db >= 40.0, f"Fire scene PSNR {db:.1f} dB < 40 dB"

    def test_arc_welder_ratio1_psnr_39db(self):
        """Pathological arc-welder (64× overexposed) at full Reinhard → ≥ 39 dB."""
        img   = _make_hdr_image(peak=64.0)
        comp  = _np_compress(img, 1.0)
        recon = _np_decompress(comp, 1.0)
        db    = _psnr(img, recon, peak=64.0)
        assert db >= 39.0, f"Arc-welder PSNR {db:.1f} dB < 39 dB"

    def test_golden_hour_best_ratio_is_high(self):
        """
        For golden-hour (4× peak), the optimal compression_ratio is 0.5+.
        Verify PSNR(ratio=0.75) > PSNR(ratio=0.0) by ≥ 10 dB.
        """
        img = _make_hdr_image(peak=4.0)

        def _rt(r):
            return _psnr(_make_hdr_image(4.0), _np_decompress(_np_compress(_make_hdr_image(4.0), r), r), 4.0)

        db_high = _psnr(img, _np_decompress(_np_compress(img, 0.75), 0.75), 4.0)
        db_low  = _psnr(img, _np_decompress(_np_compress(img, 0.00), 0.00), 4.0)
        assert db_high - db_low >= 10.0, (
            f"ratio=0.75 ({db_high:.1f} dB) not ≥ 10 dB above ratio=0.0 ({db_low:.1f} dB)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Torch-conditional node-class round-trip
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestTorchRoundtrip:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    """
    Verifies that the torch node implementations produce numerically
    identical results to the numpy reference implementations.
    """

    def _make_tensor(self, peak: float, shape=(1, 32, 32, 3)) -> "torch.Tensor":
        np_img = _make_hdr_image(peak, shape=(shape[1], shape[2], shape[3]))
        return torch.from_numpy(np_img).unsqueeze(0)

    @pytest.mark.parametrize("ratio", RATIOS)
    def test_torch_vs_numpy_compress(self, ratio):
        """torch compress == numpy compress to float32 precision."""
        t  = self._make_tensor(peak=4.0)
        np_img = t.squeeze(0).numpy()

        t_out = _hdr_soft_compress(t, ratio).squeeze(0).numpy()
        np_out = _np_compress(np_img, ratio)

        np.testing.assert_allclose(t_out, np_out, atol=1e-6,
            err_msg=f"torch vs numpy compress mismatch at ratio={ratio}")

    @pytest.mark.parametrize("ratio", RATIOS)
    def test_torch_vs_numpy_decompress(self, ratio):
        """torch decompress == numpy decompress to float32 precision."""
        t  = self._make_tensor(peak=4.0)
        comp_np = _np_compress(t.squeeze(0).numpy(), ratio)
        comp_t  = torch.from_numpy(comp_np).unsqueeze(0)

        t_out  = _hdr_soft_decompress(comp_t, ratio).squeeze(0).numpy()
        np_out = _np_decompress(comp_np, ratio)

        np.testing.assert_allclose(t_out, np_out, atol=1e-5,
            err_msg=f"torch vs numpy decompress mismatch at ratio={ratio}")

    @pytest.mark.parametrize("scene,peak,psnr_targets", SCENES)
    def test_torch_psnr_roundtrip(self, scene, peak, psnr_targets):
        """Torch end-to-end PSNR at ratio=0.5 must meet ≥ numpy target."""
        t       = self._make_tensor(peak)
        ratio   = 0.5
        comp    = _hdr_soft_compress(t, ratio)
        recon   = _hdr_soft_decompress(comp, ratio)
        orig_np = t.squeeze(0).numpy()
        rec_np  = recon.squeeze(0).numpy()
        db      = _psnr(orig_np, rec_np, peak)
        target  = psnr_targets[RATIOS.index(0.5)]
        assert db >= target, (
            f"torch PSNR for {scene}: {db:.1f} dB < {target} dB"
        )
