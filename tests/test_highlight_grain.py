"""The torch highlight grain pass against RUDRA 0.9.0-beta.2's numpy/OpenCV one.

`_reference` is the numpy/OpenCV implementation Radiance ported from RUDRA,
kept verbatim: if RUDRA changes its grain correction, update it here and the
parity test shows what the torch port has to follow.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
cv2 = pytest.importorskip("cv2")

from radiance.model.highlight_grain import settle_highlight_grain  # noqa: E402

_LUMA = np.array([0.2627, 0.6780, 0.0593], dtype=np.float64)


def _reference(hdr_nits, sdr_srgb, knee=0.9, softness=0.04, sigma=2.0):
    code = sdr_srgb.max(axis=-1).astype(np.float64)
    ramp = np.clip((code - (knee - softness)) / (2.0 * softness), 0.0, 1.0)
    ramp = ramp * ramp * (3.0 - 2.0 * ramp)
    if not np.any(ramp > 0.0):
        return hdr_nits
    size = 7
    flat = np.zeros(code.shape, dtype=np.float64)
    for signal in (sdr_srgb.max(axis=-1), sdr_srgb @ _LUMA):
        signal = signal.astype(np.float64)
        mean = cv2.boxFilter(signal, -1, (size, size), normalize=True, borderType=cv2.BORDER_REFLECT)
        second = cv2.boxFilter(signal * signal, -1, (size, size), normalize=True,
                               borderType=cv2.BORDER_REFLECT)
        spread = np.sqrt(np.maximum(second - mean * mean, 0.0))
        flat = np.maximum(flat, np.clip((3.0 - spread * 255.0) / 2.0, 0.0, 1.0))
    flat = flat * flat * (3.0 - 2.0 * flat)
    weight = ramp * flat
    if not np.any(weight > 0.0):
        return hdr_nits
    luma = np.maximum(hdr_nits @ _LUMA, 0.0)
    blur = lambda x: cv2.GaussianBlur(x, (0, 0), sigmaX=sigma, borderType=cv2.BORDER_REFLECT)
    settled = blur(flat * luma) / np.maximum(blur(flat), 1e-6)
    target = luma + weight * (settled - luma)
    gain = np.where(luma > 1e-6, target / np.maximum(luma, 1e-6), 1.0)
    return np.nan_to_num(hdr_nits * gain[..., None], nan=0.0, posinf=0.0, neginf=0.0)


def _frame(seed, h=72, w=96):
    """A flat SDR highlight (sky), a textured area, and grainy reconstructed nits."""
    rng = np.random.default_rng(seed)
    sdr = np.empty((h, w, 3), np.float32)
    sdr[: h // 2] = [0.97, 0.95, 0.9]
    sdr[h // 2:] = rng.uniform(0.2, 1.0, (h - h // 2, w, 3))
    hdr = (sdr.astype(np.float64) ** 2.2 * 800.0 * rng.uniform(0.8, 1.2, (h, w, 1))).astype(np.float32)
    return hdr, sdr


def test_matches_the_numpy_reference_on_a_frame():
    hdr, sdr = _frame(0)
    want = _reference(hdr, sdr)
    got = settle_highlight_grain(torch.from_numpy(hdr), torch.from_numpy(sdr)).numpy()
    assert float(np.abs(want - hdr).max()) > 1.0, "the fixture must exercise the correction"
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-4)


def test_a_batch_matches_frame_by_frame():
    frames = [_frame(seed) for seed in (1, 2, 3)]
    hdr = torch.from_numpy(np.stack([f[0] for f in frames]))
    sdr = torch.from_numpy(np.stack([f[1] for f in frames]))
    got = settle_highlight_grain(hdr, sdr).numpy()
    for i, (h, s) in enumerate(frames):
        np.testing.assert_allclose(got[i], _reference(h, s), rtol=1e-6, atol=1e-4)


def test_no_highlight_leaves_the_frame_untouched():
    hdr, sdr = _frame(4)
    sdr = sdr * 0.5
    out = settle_highlight_grain(torch.from_numpy(hdr), torch.from_numpy(sdr))
    assert torch.equal(out, torch.from_numpy(hdr))


@pytest.mark.parametrize("h,w", [(1, 1), (3, 3), (3, 12), (5, 40)])
def test_images_smaller_than_the_filters_match_too(h, w):
    """OpenCV mirrors again when the 17-tap blur is wider than the image."""
    rng = np.random.default_rng(h * 100 + w)
    sdr = rng.uniform(0.9, 1.0, (h, w, 3)).astype(np.float32)
    hdr = (sdr * 900.0 * rng.uniform(0.8, 1.2, (h, w, 1))).astype(np.float32)
    got = settle_highlight_grain(torch.from_numpy(hdr), torch.from_numpy(sdr)).numpy()
    np.testing.assert_allclose(got, _reference(hdr, sdr), rtol=1e-6, atol=1e-4)
