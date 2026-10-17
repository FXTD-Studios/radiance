import numpy as np
import pytest

from radiance.core.normal_map import detect_encoding, prepare_normal


def _frame(vec, shape=(2, 2)):
    return np.broadcast_to(np.asarray(vec, np.float32), shape + (3,)).copy()


def test_signed_input_is_detected_and_unit_length():
    unit, disp, info = prepare_normal(_frame([0.0, 0.0, -2.0]))
    assert info["input"] == "signed"
    np.testing.assert_allclose(unit[0, 0], [0, 0, -1])
    np.testing.assert_allclose(disp[0, 0], [0.5, 0.5, 0.0])


def test_encoded_input_is_decoded():
    # (0.5, 0.5, 1.0) is the flat "straight at the camera" normal in a PNG.
    unit, disp, info = prepare_normal(_frame([0.5, 0.5, 1.0]))
    assert info["input"] == "unsigned_0_1"
    np.testing.assert_allclose(unit[0, 0], [0, 0, 1])
    np.testing.assert_allclose(disp[0, 0], [0.5, 0.5, 1.0])


def test_vectors_are_renormalised():
    unit, _, _ = prepare_normal(_frame([3.0, -4.0, 0.0]))
    np.testing.assert_allclose(unit[0, 0], [0.6, -0.8, 0.0], atol=1e-6)


def test_invalid_pixels_are_counted_and_grey():
    f = _frame([0.0, 0.0, 1.0])
    f[0, 0] = [np.nan, 0, 1]
    f[0, 1] = [0, 0, 0]
    unit, disp, info = prepare_normal(f)
    assert info["invalid_pixels"] == 2 and info["pixels"] == 4
    np.testing.assert_allclose(unit[0, 0], 0)
    np.testing.assert_allclose(disp[0, 1], 0.5)
    assert np.isfinite(unit).all() and np.isfinite(disp).all()


def test_alpha_is_ignored_and_shape_checked():
    f = np.concatenate([_frame([0, 0, 1.0]), np.zeros((2, 2, 1), np.float32)], axis=-1)
    assert prepare_normal(f)[0].shape == (2, 2, 3)
    with pytest.raises(ValueError):
        prepare_normal(np.zeros((2, 2), np.float32))
    with pytest.raises(ValueError):
        prepare_normal(np.zeros((2, 2, 2), np.float32))


def test_all_invalid_frame_does_not_crash():
    _, _, info = prepare_normal(np.full((2, 2, 3), np.nan, np.float32))
    assert info["invalid_pixels"] == 4 and info["input"] == "unsigned_0_1"


def test_signed_pass_without_negatives_is_still_signed():
    # A world-space pass of a floor: (0, 1, 0) everywhere, no negative component.
    unit, _, info = prepare_normal(_frame([0.0, 1.0, 0.0]))
    assert info["input"] == "signed"
    np.testing.assert_allclose(unit[0, 0], [0, 1, 0])


def test_encoded_flat_normal_is_not_taken_as_signed():
    assert detect_encoding(_frame([0.5, 0.5, 1.0])) == "unsigned_0_1"


def test_forced_encoding_skips_detection():
    unit, _, info = prepare_normal(_frame([0.5, 0.5, 1.0]), "signed")
    assert info["input"] == "signed"
    np.testing.assert_allclose(unit[0, 0], np.array([0.5, 0.5, 1.0]) / np.linalg.norm([0.5, 0.5, 1.0]), rtol=1e-6)
    with pytest.raises(ValueError):
        prepare_normal(_frame([0, 0, 1.0]), "bogus")


def test_viewer_decides_once_per_sequence():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "nodes" / "monitor" / "viewer.py").read_text(encoding="utf-8")
    assert "prepare_normal(frame, encoding)" in src
