"""A Deliver master is graded with the same maths as the viewer.

The viewer grades on the GPU with js/radiance_grade.js (gradePixelFull is its
CPU form, held to the real shader by js/tests/grade_render.test.mjs). A Deliver
master is graded in Python: delivery/handler.py maps the payload with
viewer_grade_kwargs and runs apply_grading. The Python grade still had the old
viewer maths, so masters stopped matching the screen when the viewer was fixed:
additive Temperature/Tint clamped at zero, linear contrast, Luma Mix restoring
the ungraded luminance, and the tone-mapper ACEScct matrices.

tests/fixtures/viewer_grade_parity.json is generated from radiance_grade.js by
js/tests/viewer_grade_fixture.mjs, and js/tests/viewer_grade_fixture.test.mjs
fails if the committed file stops matching it, so the two grades cannot drift.
"""
import json
import pathlib

import numpy as np
import pytest

_FIXTURE = pathlib.Path(__file__).resolve().parent / "fixtures" / "viewer_grade_parity.json"


def _load():
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


def _grade(grading, pixels):
    from radiance.color.grading import apply_grading, viewer_grade_kwargs

    img = np.asarray(pixels, dtype=np.float32).reshape(1, -1, 3)
    out = apply_grading(img, **viewer_grade_kwargs(grading))
    return out.reshape(-1, 3).astype(np.float64)


@pytest.mark.parametrize("case", _load()["cases"], ids=lambda c: c["name"])
def test_python_grade_matches_the_viewer(case):
    fixture = _load()
    got = _grade(case["grading"], fixture["pixels"])
    want = np.asarray(case["expected"], dtype=np.float64)
    # 1e-5, relative above 1: float32 pixels against float64 maths.
    tol = 1e-5 * np.maximum(1.0, np.abs(want))
    bad = np.abs(got - want) > tol
    assert not bad.any(), (
        f"{case['name']}: Python {got[bad.any(axis=1)].tolist()} vs viewer "
        f"{want[bad.any(axis=1)].tolist()} for pixels "
        f"{np.asarray(fixture['pixels'])[bad.any(axis=1)].tolist()}"
    )


def test_the_fixture_covers_every_grade_control():
    keys = set()
    for case in _load()["cases"]:
        keys |= set(case["grading"])
    for k in ("exposure", "temperature", "tint", "offset", "lift", "gain", "gamma", "contrast", "pivot",
              "saturation", "shadows", "highlights", "colorBoost", "hue_shift", "lumaMix", "logShadow",
              "logMidtone", "logHighlight", "printerR", "curveTable", "secondaryCurveTable", "colorScience",
              "sourceGamut"):
        assert k in keys, f"no parity case exercises {k}"


def test_an_old_grade_without_a_pivot_pivots_on_grey():
    """Saved grades from before the pivot was sent: a missing pivot is 0.18."""
    got = _grade({"contrast": 1.8}, [[0.18, 0.18, 0.18]])
    np.testing.assert_allclose(got, [[0.18, 0.18, 0.18]], atol=1e-6)


def test_white_balance_is_a_gain_not_an_offset():
    """Temperature 0.1 turned black dark red and any Tint clamped negatives."""
    black = _grade({"temperature": 0.1}, [[0.0, 0.0, 0.0]])
    assert np.all(black == 0.0)
    neg = _grade({"tint": 0.001}, [[-0.05, 0.2, 0.2]])
    assert neg[0, 0] < -0.04


def test_alpha_is_left_alone():
    from radiance.color.grading import apply_grading, viewer_grade_kwargs

    rgba = np.full((2, 2, 4), 0.5, dtype=np.float32)
    rgba[..., 3] = 0.25
    out = apply_grading(rgba, **viewer_grade_kwargs({"exposure": 1.0, "temperature": 0.3, "contrast": 1.3,
                                                     "shadows": 0.4, "lumaMix": 0.5, "saturation": 1.4}))
    assert out.shape == rgba.shape
    assert np.allclose(out[..., 3], 0.25)
