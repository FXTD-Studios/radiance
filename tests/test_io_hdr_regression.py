"""Regression tests for the HDR/EXR I/O fixes (review findings C-1, C-2, C-3).

These guard against three production-breaking bugs that existed in v3.1.0:

  C-1  `_np_to_tensor` divided any array with max > 2.0 by 255/65535,
       silently crushing scene-linear EXR/HDR data ~255x on load.
  C-2  `_save_exr` swallowed all failures and could write a 0-byte file
       while reporting success.
  C-3  `_save_exr` wrote only R/G/B (dropping alpha) and crashed on
       single-channel matte input.

Run inside an environment with torch + OpenEXR (or OpenCV built with EXR
support), e.g. the ComfyUI venv:  pytest tests/test_io_hdr_regression.py
"""
import importlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")

# Only the tests marked @pytest.mark.real_torch below need genuine tensors; the
# rest run fine against conftest's stub. Opt out of the automatic module-level
# skip so they keep running on the no-torch CI matrix.
RADIANCE_TORCH_GATED = True

nodes_io = importlib.import_module("radiance.nodes.io.write")


# ── C-1: scene-linear values must survive the tensor conversion ─────────────

@pytest.mark.real_torch
def test_np_to_tensor_preserves_hdr_values():
    """Values above 2.0 must NOT be normalized away."""
    arr = np.array([[[0.0, 1.0, 5.0], [10.0, 50.0, 123.4]]], dtype=np.float32)
    t = nodes_io._np_to_tensor(arr)
    # _np_to_tensor adds exactly one leading batch dim: (1,2,3) -> (1,1,2,3).
    assert tuple(t.shape) == (1, 1, 2, 3)
    out = t[0].cpu().numpy()
    np.testing.assert_allclose(out, arr, rtol=0, atol=0)
    assert float(out.max()) == pytest.approx(123.4, abs=1e-4)


@pytest.mark.real_torch
def test_np_to_tensor_does_not_touch_low_range():
    arr = np.array([[[0.25, 0.5, 0.75]]], dtype=np.float32)
    out = nodes_io._np_to_tensor(arr)[0].cpu().numpy()
    np.testing.assert_allclose(out, arr, rtol=0, atol=0)


# ── C-3: EXR channel mapping handles 1 / 3 / 4 channels, never drops alpha ──

def test_exr_channels_single_channel_matte():
    arr = np.full((4, 4), 0.7, dtype=np.float32)
    names = [n for n, _ in nodes_io._exr_channels(arr)]
    assert names == ["R", "G", "B"]  # grayscale broadcast, no crash


def test_exr_channels_rgb():
    arr = np.zeros((4, 4, 3), dtype=np.float32)
    assert [n for n, _ in nodes_io._exr_channels(arr)] == ["R", "G", "B"]


def test_exr_channels_rgba_preserves_alpha():
    arr = np.zeros((4, 4, 4), dtype=np.float32)
    arr[..., 3] = 0.42
    chans = dict(nodes_io._exr_channels(arr))
    assert set(chans) == {"R", "G", "B", "A"}
    np.testing.assert_allclose(chans["A"], 0.42)


def test_exr_channels_rejects_bad_channel_count():
    with pytest.raises(ValueError):
        nodes_io._exr_channels(np.zeros((4, 4, 2), dtype=np.float32))


# ── C-2/C-3: round-trip write (requires OpenEXR or cv2 EXR support) ─────────

def _exr_writable():
    try:
        import OpenEXR  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        import cv2  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _exr_writable(), reason="no EXR backend available")
@pytest.mark.real_torch
def test_save_exr_rgba_roundtrip(tmp_path):
    arr = np.random.rand(8, 8, 4).astype(np.float32)
    arr[..., :3] *= 12.0  # scene-linear highlights well above 1.0
    path = tmp_path / "rgba.exr"
    nodes_io._save_exr(arr, path, half=False)
    assert path.exists() and path.stat().st_size > 0  # C-2: never empty
    img, mask = nodes_io._read_exr_single(str(path))
    assert mask is not None, "alpha channel was dropped (C-3)"
    rgb = img[0].cpu().numpy()
    assert float(rgb.max()) > 2.0, "HDR magnitude lost on round-trip (C-1)"


@pytest.mark.skipif(not _exr_writable(), reason="no EXR backend available")
@pytest.mark.real_torch
def test_save_exr_single_channel_does_not_crash(tmp_path):
    matte = np.full((8, 8), 0.6, dtype=np.float32)
    path = tmp_path / "matte.exr"
    nodes_io._save_exr(matte, path, half=False)  # must not raise (C-3)
    assert path.exists() and path.stat().st_size > 0


def test_save_exr_raises_when_no_backend(tmp_path, monkeypatch):
    """C-2: with no working backend we must raise, never write a 0-byte file."""
    import builtins

    real_import = builtins.__import__

    def _block(name, *args, **kwargs):
        if name in ("OpenEXR", "Imath", "cv2"):
            raise ImportError(f"blocked {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block)
    path = tmp_path / "fail.exr"
    with pytest.raises(RuntimeError):
        nodes_io._save_exr(np.zeros((4, 4, 3), dtype=np.float32), path, half=False)
    assert not (path.exists() and path.stat().st_size == 0), "left a 0-byte file"


# ── #5: MASK -> EXR alpha wiring (RadianceWrite) ────────────────────────────

@pytest.mark.real_torch
def test_coerce_mask_to_alpha_shapes():
    """Mask normalizes to (N,H,W), broadcasting a single mask across frames."""
    m = torch.full((1, 8, 8), 0.3)
    a = nodes_io._coerce_mask_to_alpha(m, n=1, h=8, w=8)
    assert a is not None and a.shape == (1, 8, 8)
    assert abs(float(a[0].mean()) - 0.3) < 1e-5
    a3 = nodes_io._coerce_mask_to_alpha(torch.full((1, 8, 8), 0.5), n=3, h=8, w=8)
    assert a3.shape == (3, 8, 8)
    assert nodes_io._coerce_mask_to_alpha(None, 1, 8, 8) is None


@pytest.mark.skipif(not _exr_writable(), reason="no EXR backend available")
@pytest.mark.real_torch
def test_mask_written_as_exr_alpha(tmp_path):
    """An RGB frame + mask is written as a 4-channel RGBA EXR with alpha preserved."""
    rgb = (np.random.rand(8, 8, 3).astype(np.float32)) * 5.0  # HDR
    alpha = nodes_io._coerce_mask_to_alpha(torch.full((1, 8, 8), 0.3), n=1, h=8, w=8)
    rgba = np.concatenate([rgb, alpha[0][..., None]], axis=-1)
    path = tmp_path / "rgba_from_mask.exr"
    nodes_io._save_exr(rgba, path, half=False)
    _, mask = nodes_io._read_exr_single(str(path))
    assert mask is not None, "alpha channel missing from written EXR"
