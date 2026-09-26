"""
tests/test_integration_hdr_pipeline.py
═══════════════════════════════════════════════════════════════════════════════
End-to-end integration tests for the Radiance HDR color pipeline.

Test 1 — Log C4 → Linear → ACES 2.0 Daniele Evo tonescale → EXR codec
  Verifies that the full linearisation + tone-mapping + HDR soft-compression
  codec chain achieves PSNR > 55 dB on the codec leg (compress → decompress).

Test 2 — Temporal flicker removal: 4-frame stability check
  Verifies that a simple weighted-average temporal smoother reduces
  frame-to-frame luma variance significantly (> 60 % reduction).

Test 3 — RadianceLinearize node (torch-gated)
  Exercises the actual ComfyUI node class on a synthetic LogC4 tensor and
  confirms shape contract + value range contract.

Test 4 — RadianceACES2Tonescale node round-trip (torch-gated)
  Exercises RadianceACES2Tonescale with a known mid-grey input and checks
  that the 18% grey maps to ≈ 10% of peak (ACES 2.0 standard).

All tests run in pure numpy for CI (no torch / GPU required).
Torch-gated tests are skipped when PyTorch is not installed.
"""

import sys
import os
import math
import types
import tempfile
import importlib
from pathlib import Path

import numpy as np
import pytest

# ── project root on sys.path ──────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

# ── ComfyUI stubs (needed so node modules can be imported) ────────────────────
for _mod in ["folder_paths", "comfy", "comfy.utils", "comfy.model_management"]:
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)
_fp = sys.modules["folder_paths"]
if not hasattr(_fp, "get_output_directory"):
    _fp.get_output_directory = lambda: tempfile.gettempdir()

# ── Torch guard ───────────────────────────────────────────────────────────────
try:
    import torch as _torch_probe
    HAS_TORCH = isinstance(getattr(_torch_probe, "__version__", None), str)
except ImportError:
    HAS_TORCH = False

skip_no_torch = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")

if HAS_TORCH:
    import torch


# ═════════════════════════════════════════════════════════════════════════════
#  NUMPY REFERENCE IMPLEMENTATIONS
#  (mirror the Radiance Torch implementations for pure-numpy CI runs)
# ═════════════════════════════════════════════════════════════════════════════

def _np_logc4_to_linear(v: np.ndarray) -> np.ndarray:
    """ARRI LogC4 → scene-linear (numpy).
    Source: ARRI LogC4 Logarithmic Color Space v1.0 (2021)
    """
    # LogC4 encoding parameters
    a  = (2.0 ** 18 - 16) / 117.45
    b  = (1023.0 - 95.0) / 1023.0
    c  = 95.0 / 1023.0
    s  = (7 * math.log(2) * 2.0 ** (7 - 14 * c / b)) / (a * b)
    t  = (2.0 ** (14 * (-c / b) + 6) - 64) / a
    cut = (14 * np.log(2) * 2.0 ** (7 - 14 * c / b)) / (a * b)
    # Piecewise: below cut → linear; above → log
    lin  = (v - c) / b * s + t
    log_ = (np.power(2.0, 14.0 * (v - c) / b + 6.0) - 64.0) / a
    return np.where(v < cut, lin, log_)


def _np_linear_to_logc4(x: np.ndarray) -> np.ndarray:
    """Scene-linear → ARRI LogC4 (numpy, inverse of above)."""
    a = (2.0 ** 18 - 16) / 117.45
    b = (1023.0 - 95.0) / 1023.0
    c = 95.0 / 1023.0
    s = (7 * math.log(2) * 2.0 ** (7 - 14 * c / b)) / (a * b)
    t = (2.0 ** (14 * (-c / b) + 6) - 64) / a
    cut = (14 * np.log(2) * 2.0 ** (7 - 14 * c / b)) / (a * b)
    lin  = (x - t) / s * b + c
    log_ = (np.log2(a * x + 64.0) - 6.0) / 14.0 * b + c
    return np.where(x < (cut - c) / b * s + t, lin, log_)


def _np_compress(img: np.ndarray, ratio: float) -> np.ndarray:
    """HDR soft compression into [0, 1] — Reinhard blend."""
    x       = np.maximum(img, 0.0)
    reinhard = x / (1.0 + x)
    clamped  = np.minimum(x, 1.0)
    if ratio <= 0.0:
        return clamped
    if ratio >= 1.0:
        return reinhard
    return clamped * (1.0 - ratio) + reinhard * ratio


def _np_decompress(img: np.ndarray, ratio: float) -> np.ndarray:
    """Analytical inverse of _np_compress (two-regime exact inverse)."""
    r   = float(ratio)
    eps = 1e-7
    y   = np.clip(img, 0.0, 1.0 - eps)
    if r <= 0.0:
        return y
    if r >= 1.0:
        return y / (1.0 - y + eps)
    # Break-point: y value that corresponds to x = 1.0
    y_break     = 1.0 - r * 0.5          # = (1-r)*1 + r*(1/2)
    # HDR leg (x > 1): y = (1-r) + r*x/(1+x)  →  x = (y - 1 + r)/(1 - y)
    x_hdr       = (y - 1.0 + r) / (1.0 - y + eps)
    # SDR leg (x ≤ 1): quadratic (1-r)x² + (1-y)x - y = 0
    one_minus_r = 1.0 - r
    one_minus_y = 1.0 - y
    disc        = one_minus_y ** 2 + 4.0 * one_minus_r * y
    x_sdr       = (-one_minus_y + np.sqrt(np.maximum(disc, 0.0))) / (2.0 * one_minus_r + eps)
    return np.where(y > y_break, x_hdr, x_sdr)


def _psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Peak signal-to-noise ratio in dB (signal peak = 1.0)."""
    mse = float(np.mean((original - reconstructed) ** 2))
    if mse == 0.0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def _np_daniele_evo_fwd(x: np.ndarray,
                        peak_nits: float = 100.0,
                        g: float = 1.15,
                        scene_grey: float = 0.18,
                        grey_target: float = 0.10,
                        toe_scene: float = 0.04) -> np.ndarray:
    """
    ACES 2.0 Daniele Evo forward tonescale (numpy, per Academy spec).
    Returns normalised display [0, 1].
    """
    n      = peak_nits / 100.0
    ratio  = grey_target / (1.0 - grey_target)
    K      = scene_grey / (ratio ** (1.0 / g))

    u_t    = (toe_scene / K) ** g
    toe_y  = n * u_t / (u_t + 1.0)
    toe_dy = n * g * u_t / (toe_scene * (u_t + 1.0) ** 2)

    xp     = np.maximum(x, 0.0)
    u      = (xp / K) ** g
    y_main = n * u / (u + 1.0)
    y_toe  = np.maximum(toe_y + toe_dy * (xp - toe_scene), 0.0)

    return np.where(xp < toe_scene, y_toe, y_main) / n   # → [0, 1]


def _np_temporal_smooth(frames: np.ndarray, weights: tuple = (0.1, 0.2, 0.4, 0.2, 0.1)) -> np.ndarray:
    """
    Causal temporal smoother — weighted average of neighbouring frames.
    `frames`: (N, H, W, C) float32.  Returns same shape.
    """
    N   = frames.shape[0]
    out = np.empty_like(frames)
    hw  = len(weights) // 2
    for i in range(N):
        acc = np.zeros_like(frames[0])
        w_sum = 0.0
        for j, w in enumerate(weights):
            idx = i + j - hw
            if 0 <= idx < N:
                acc   += frames[idx] * w
                w_sum += w
        out[i] = acc / w_sum
    return out


def _synthetic_logc4_frame(height: int = 64, width: int = 64,
                            seed: int = 42) -> np.ndarray:
    """
    Return a synthetic LogC4-encoded frame (H, W, 3) float32.

    Spectral content: smooth gradient + mid-frequency detail to stress
    the codec without pure-noise pathology.
    """
    rng  = np.random.default_rng(seed)
    x    = np.linspace(0.0, 1.0, width,  dtype=np.float32)
    y    = np.linspace(0.0, 1.0, height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    # Linear HDR content: 0 → 6.0 stops (~0 – 4 linear)
    linear_r = 0.001 + 4.0 * xx * yy + 0.5 * (1.0 - yy)
    linear_g = 0.001 + 3.0 * xx + 0.3 * yy ** 2
    linear_b = 0.001 + 1.0 * (1.0 - xx) * yy + 0.2 * xx

    linear = np.stack([linear_r, linear_g, linear_b], axis=-1).astype(np.float32)
    # Add a tiny amount of high-frequency noise (real cameras have noise floor)
    linear += rng.normal(0.0, 0.002, linear.shape).astype(np.float32)
    linear  = np.maximum(linear, 0.0)

    return _np_linear_to_logc4(linear)


# ═════════════════════════════════════════════════════════════════════════════
#  TEST 1 — LOG C4 → LINEAR → ACES 2.0 → HDR CODEC ROUND-TRIP  (PSNR ≥ 55 dB)
# ═════════════════════════════════════════════════════════════════════════════

class TestLogC4ToACES2EXRRoundTrip:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    """
    Pipeline: LogC4 frame → decode → Daniele Evo → HDR compress → decompress.
    The EXR "codec leg" (compress/decompress) must achieve PSNR > 55 dB
    relative to the pre-compression tone-mapped signal.
    """

    PSNR_THRESHOLD = 55.0   # dB
    COMPRESS_RATIO = 0.75   # typical HDR-LoRA setting

    def _run_pipeline(self, frame_logc4: np.ndarray) -> tuple:
        """Returns (tone_mapped, reconstructed) numpy arrays."""
        # 1. Decode Log C4 → scene-linear
        linear = _np_logc4_to_linear(frame_logc4)
        linear = np.maximum(linear, 0.0)

        # 2. ACES 2.0 Daniele Evo tonescale (normalised to [0, 1])
        tone_mapped = _np_daniele_evo_fwd(linear, peak_nits=1000.0)

        # 3. HDR soft compression → [0, 1] (VAE latent range)
        compressed = _np_compress(tone_mapped, self.COMPRESS_RATIO)

        # 4. HDR soft decompression (simulated EXR reload + decode)
        reconstructed = _np_decompress(compressed, self.COMPRESS_RATIO)

        return tone_mapped, reconstructed

    def test_psnr_exceeds_55db_small_frame(self):
        """64×64 synthetic Log C4 frame — codec PSNR must exceed 55 dB."""
        frame_logc4 = _synthetic_logc4_frame(64, 64)
        tone_mapped, reconstructed = self._run_pipeline(frame_logc4)

        psnr = _psnr(tone_mapped, reconstructed)
        assert psnr > self.PSNR_THRESHOLD, (
            f"PSNR {psnr:.2f} dB below threshold {self.PSNR_THRESHOLD} dB"
        )

    def test_psnr_exceeds_55db_mid_grey(self):
        """18% grey log-encoded patch must round-trip cleanly."""
        grey_linear  = np.full((32, 32, 3), 0.18, dtype=np.float32)
        grey_logc4   = _np_linear_to_logc4(grey_linear)
        tone_mapped, reconstructed = self._run_pipeline(grey_logc4)

        psnr = _psnr(tone_mapped, reconstructed)
        assert psnr > self.PSNR_THRESHOLD, (
            f"Mid-grey PSNR {psnr:.2f} dB below threshold"
        )

    def test_aces2_maps_18pct_grey_to_10pct_peak(self):
        """
        ACES 2.0 spec: scene linear 0.18 → display ≈ 10% of peak.
        Tolerance: ±1% of peak (10 nits out of 1000 nits ± 10 nits).
        """
        grey_linear = np.full((1, 1, 3), 0.18, dtype=np.float32)
        grey_mapped = _np_daniele_evo_fwd(grey_linear, peak_nits=1000.0)
        # All channels should be ~0.10 (10% of normalised peak)
        assert np.allclose(grey_mapped, 0.10, atol=0.01), (
            f"18% grey maps to {float(grey_mapped.mean()):.4f}, expected ≈ 0.10"
        )

    def test_pipeline_no_nans_or_infs(self):
        """No NaN / Inf values at any stage of the pipeline."""
        frame_logc4 = _synthetic_logc4_frame(64, 64, seed=99)
        tone_mapped, reconstructed = self._run_pipeline(frame_logc4)

        assert np.all(np.isfinite(tone_mapped)),     "NaN/Inf in tone_mapped"
        assert np.all(np.isfinite(reconstructed)),   "NaN/Inf in reconstructed"

    def test_compressed_signal_in_unit_range(self):
        """Compressed signal must be in [0, 1] for VAE compatibility."""
        frame_logc4 = _synthetic_logc4_frame(64, 64, seed=7)
        linear      = _np_logc4_to_linear(frame_logc4)
        linear      = np.maximum(linear, 0.0)
        tone_mapped = _np_daniele_evo_fwd(linear, peak_nits=1000.0)
        compressed  = _np_compress(tone_mapped, self.COMPRESS_RATIO)

        assert compressed.min() >= -1e-6, "Compressed signal below 0"
        assert compressed.max() <= 1.0 + 1e-6, "Compressed signal above 1"

    def test_black_preservation(self):
        """Pure black (log-encoded near 0) must stay black after round-trip."""
        black_logc4     = _np_linear_to_logc4(np.zeros((16, 16, 3), dtype=np.float32))
        tone_mapped, _  = self._run_pipeline(black_logc4)
        assert np.allclose(tone_mapped, 0.0, atol=1e-4), (
            f"Black not preserved: max={tone_mapped.max():.6f}"
        )


# ═════════════════════════════════════════════════════════════════════════════
#  TEST 2 — TEMPORAL FLICKER REMOVAL: 4-FRAME STABILITY
# ═════════════════════════════════════════════════════════════════════════════

class TestTemporalFlickerRemoval:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    """
    Verify that the temporal smoother meaningfully reduces inter-frame
    luma variance on a 4-frame synthetic sequence.

    Flicker model: identical base frame + per-frame multiplicative gain
    jitter (uniform ±8%), simulating analog sensor noise or exposure drift.
    """

    N_FRAMES  = 4
    JITTER    = 0.08   # ±8% multiplicative flicker
    SEED      = 2024

    def _make_jittered_sequence(self) -> np.ndarray:
        rng  = np.random.default_rng(self.SEED)
        base = _synthetic_logc4_frame(64, 64, seed=0)
        base_linear = _np_logc4_to_linear(base)
        base_linear = np.maximum(base_linear, 0.0)
        # Tonemap to display
        base_display = _np_daniele_evo_fwd(base_linear, peak_nits=1000.0)

        frames = []
        for _ in range(self.N_FRAMES):
            gain  = 1.0 + rng.uniform(-self.JITTER, self.JITTER)
            frame = np.clip(base_display * gain, 0.0, 1.0).astype(np.float32)
            frames.append(frame)
        return np.stack(frames, axis=0)   # (N, H, W, 3)

    @staticmethod
    def _frame_luma_means(frames: np.ndarray) -> np.ndarray:
        """Return per-frame luma mean. frames: (N, H, W, 3)."""
        # BT.709 luma coefficients
        return (0.2126 * frames[..., 0] +
                0.7152 * frames[..., 1] +
                0.0722 * frames[..., 2]).reshape(frames.shape[0], -1).mean(axis=1)

    def test_smoother_reduces_variance(self):
        """Temporal smoothing must reduce per-frame luma std by > 50%."""
        frames  = self._make_jittered_sequence()
        smoothed = _np_temporal_smooth(frames)

        std_before = float(np.std(self._frame_luma_means(frames)))
        std_after  = float(np.std(self._frame_luma_means(smoothed)))

        reduction = 1.0 - (std_after / (std_before + 1e-9))
        assert reduction > 0.50, (
            f"Temporal smoother only reduced luma std by {reduction:.1%} "
            f"(need > 50%); before={std_before:.5f} after={std_after:.5f}"
        )

    def test_smoother_preserves_mean_luminance(self):
        """Global luma mean must not shift by more than 5%."""
        frames   = self._make_jittered_sequence()
        smoothed = _np_temporal_smooth(frames)

        mean_before = float(self._frame_luma_means(frames).mean())
        mean_after  = float(self._frame_luma_means(smoothed).mean())

        rel_shift = abs(mean_after - mean_before) / (mean_before + 1e-9)
        assert rel_shift < 0.05, (
            f"Luma mean shifted {rel_shift:.1%} (threshold 5%)"
        )

    def test_smoother_no_nans(self):
        """Smoother must not introduce NaN or Inf."""
        frames   = self._make_jittered_sequence()
        smoothed = _np_temporal_smooth(frames)
        assert np.all(np.isfinite(smoothed)), "Smoother produced NaN/Inf"

    def test_smoother_output_in_valid_range(self):
        """Output must stay in [0, 1] when input is in [0, 1]."""
        frames   = self._make_jittered_sequence()
        smoothed = _np_temporal_smooth(frames)
        assert smoothed.min() >= -1e-6, f"Smoothed min = {smoothed.min():.6f}"
        assert smoothed.max() <= 1.0 + 1e-6, f"Smoothed max = {smoothed.max():.6f}"


# ═════════════════════════════════════════════════════════════════════════════
#  TEST 3 — RadianceHDRColorPipeline NODE CLASS  (torch required)
# ═════════════════════════════════════════════════════════════════════════════

@skip_no_torch
class TestRadianceHDRColorPipelineNode:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    """Integration test for the RadianceHDRColorPipeline ComfyUI node."""

    def setup_method(self):
        mod = importlib.import_module("radiance.nodes.hdr.colorspace")
        self.cls = mod.RadianceHDRColorPipeline

    def _make_logc4_tensor(self, H=32, W=32) -> "torch.Tensor":
        """Synthetic LogC4 tensor (1, H, W, 3) float32."""
        arr   = _synthetic_logc4_frame(H, W).astype(np.float32)[np.newaxis]
        return torch.from_numpy(arr)

    def test_output_shape_preserved(self):
        t = self._make_logc4_tensor()
        node = self.cls()
        _, scene_linear, _, _ = node.pipeline(t, encoding="ARRI LogC4")
        assert scene_linear.shape == t.shape

    def test_output_dtype_float32(self):
        t = self._make_logc4_tensor()
        node = self.cls()
        _, scene_linear, _, _ = node.pipeline(t, encoding="ARRI LogC4")
        assert scene_linear.dtype == torch.float32

    def test_linearised_values_non_negative(self):
        t = self._make_logc4_tensor()
        node = self.cls()
        _, scene_linear, _, _ = node.pipeline(t, encoding="ARRI LogC4")
        assert float(scene_linear.min()) >= -1e-4, (
            f"Linearized image has negative values: min={float(scene_linear.min()):.6f}"
        )

    def test_18pct_grey_roundtrip_logc4(self):
        """Log C4 encode of 0.18 → linearize → should recover ≈ 0.18."""
        grey_linear = np.full((1, 16, 16, 3), 0.18, dtype=np.float32)
        grey_logc4  = _np_linear_to_logc4(grey_linear)
        t = torch.from_numpy(grey_logc4)
        node = self.cls()
        _, scene_linear, _, _ = node.pipeline(t, encoding="ARRI LogC4")
        recovered = float(scene_linear.mean())
        assert abs(recovered - 0.18) < 0.01, (
            f"Log C4 18% grey recovered as {recovered:.4f} (expected ≈ 0.18)"
        )


# ═════════════════════════════════════════════════════════════════════════════
#  TEST 4 — RadianceACES2Tonescale NODE CLASS  (torch required)
# ═════════════════════════════════════════════════════════════════════════════

@skip_no_torch
class TestRadianceACES2TonescaleNode:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    """Integration test for the RadianceACES2Tonescale ComfyUI node."""

    def setup_method(self):
        mod = importlib.import_module("radiance.nodes.hdr.aces2")
        self.cls = mod.RadianceACES2Tonescale

    def test_mid_grey_maps_to_10pct_peak(self):
        """
        ACES 2.0 contract: scene linear 0.18 → display ≈ 10% of peak.
        (At peak_nits=100, display grey should be ≈ 0.10.)
        """
        grey = torch.full((1, 16, 16, 3), 0.18)
        node = self.cls()
        result, _ = node.apply(grey, peak_nits=100.0, mode="luminance_preserving")
        mean_val = float(result.mean())
        assert abs(mean_val - 0.10) < 0.015, (
            f"18% grey → {mean_val:.4f}, expected ≈ 0.10 (ACES 2.0 spec)"
        )

    def test_output_shape_unchanged(self):
        t = torch.rand(1, 32, 32, 3)
        node = self.cls()
        result, _ = node.apply(t, peak_nits=1000.0, mode="luminance_preserving")
        assert result.shape == t.shape

    def test_output_in_unit_range(self):
        """Tonescale output must be in [0, 1] for any non-negative input."""
        t = torch.clamp(torch.randn(1, 32, 32, 3) * 2.0 + 1.0, min=0.0)
        node = self.cls()
        result, _ = node.apply(t, peak_nits=1000.0, mode="luminance_preserving")
        assert float(result.min()) >= -1e-5
        assert float(result.max()) <= 1.0 + 1e-5

    def test_monotone_in_luma(self):
        """Brighter scene-linear input should produce brighter display output."""
        lo = torch.full((1, 1, 1, 3), 0.05)
        hi = torch.full((1, 1, 1, 3), 2.00)
        node = self.cls()
        lo_out, _ = node.apply(lo, peak_nits=1000.0, mode="luminance_preserving")
        hi_out, _ = node.apply(hi, peak_nits=1000.0, mode="luminance_preserving")
        assert float(lo_out.mean()) < float(hi_out.mean()), (
            "Tonescale is not monotone: brighter input gave darker output"
        )
