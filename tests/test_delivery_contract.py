"""The delivery payload contract, enforced in both directions.

`js/radiance_viewer.js` builds the `grading` object that `POST /radiance/deliver`
receives, and `delivery/handler.py` reads keys back out of it. Nothing tied the
two together, and they drifted: the viewer sent 11 keys while the handler read 6
more that never arrived — `shadows`, `highlights`, `hue_shift`, `lut_name`,
`lut_intensity`, `gamut_compression` — each silently falling back to its identity
default. All six are live viewer controls driving real shader uniforms.

The failure mode is the worst kind. A colourist sets Shadows −0.60 and
Highlights +0.35, presses RENDER, and gets a master with neither. The endpoint
returns 200. The HUD prints EXPORT COMPLETE. The JS comment two lines above the
payload reads "must match viewer for what-you-see = what-you-export". Nothing
anywhere says otherwise until someone A/Bs the delivered file.

These tests parse both sides and diff them, so the next key added to one has to
be added to the other.
"""
import pathlib
import re

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _src(rel):
    return (_ROOT / rel).read_text(encoding="utf-8")


def _js_payload_keys():
    """Keys of the `grading: { ... }` literal in the /radiance/deliver POST."""
    js = _src("js/radiance_viewer.js")
    start = js.index("'/radiance/deliver'")
    grading_at = js.index("grading: {", start)
    depth, i = 0, js.index("{", grading_at)
    for end in range(i, len(js)):
        if js[end] == "{":
            depth += 1
        elif js[end] == "}":
            depth -= 1
            if depth == 0:
                break
    body = js[i + 1:end]
    # Only top-level keys: skip anything nested inside a call or array.
    keys, depth = [], 0
    for line in body.split("\n"):
        stripped = line.strip()
        if depth == 0:
            m = re.match(r"^([A-Za-z_]\w*)\s*:", stripped)
            if m:
                keys.append(m.group(1))
        depth += line.count("{") + line.count("[") + line.count("(")
        depth -= line.count("}") + line.count("]") + line.count(")")
    return set(keys)


def _handler_read_keys():
    """Every key the handler pulls out of `grading`."""
    py = _src("delivery/handler.py")
    return set(re.findall(r"grading\.get\(\s*['\"]([A-Za-z_]\w*)['\"]", py))


def _declared_contract():
    py = _src("delivery/handler.py")
    block = py[py.index("GRADE_PAYLOAD_KEYS = ("):]
    block = block[:block.index(")")]
    return set(re.findall(r"['\"]([A-Za-z_]\w*)['\"]", block))


def test_every_key_the_handler_reads_is_actually_sent():
    """The defect: six grade controls exported at their default, silently."""
    missing = _handler_read_keys() - _js_payload_keys()
    assert not missing, (
        f"delivery/handler.py reads {sorted(missing)} out of the grading payload, "
        "but js/radiance_viewer.js never sends them. Each one falls back to its "
        "identity default, so the control does nothing on export while the "
        "viewer shows it applied."
    )


def test_every_key_the_viewer_sends_is_actually_read():
    """The mirror defect: `tint` was sent and read nowhere."""
    ignored = _js_payload_keys() - _handler_read_keys()
    assert not ignored, (
        f"js/radiance_viewer.js sends {sorted(ignored)} and delivery/handler.py "
        "never reads them, so those viewer controls are dropped from the master."
    )


def test_the_declared_contract_matches_both_sides():
    """GRADE_PAYLOAD_KEYS is the thing a human reads; keep it honest."""
    declared = _declared_contract()
    assert declared == _js_payload_keys(), (
        f"GRADE_PAYLOAD_KEYS and the JS payload disagree: "
        f"only-in-contract={sorted(declared - _js_payload_keys())}, "
        f"only-in-js={sorted(_js_payload_keys() - declared)}"
    )


def test_a_short_payload_is_reported_not_swallowed(caplog):
    import logging

    from radiance.delivery import handler

    handler._warned_missing_grade_keys = False
    with caplog.at_level(logging.WARNING):
        handler._warn_on_missing_grade_keys({"exposure": 0.0})
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "missing" in text and "shadows" in text

    # Once per process, not once per frame.
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        handler._warn_on_missing_grade_keys({"exposure": 0.0})
    assert not caplog.records
    handler._warned_missing_grade_keys = False


def test_a_complete_payload_is_silent(caplog):
    import logging

    from radiance.delivery import handler

    handler._warned_missing_grade_keys = False
    payload = {k: 0 for k in handler.GRADE_PAYLOAD_KEYS}
    with caplog.at_level(logging.WARNING):
        handler._warn_on_missing_grade_keys(payload)
    assert not caplog.records


# ── Temp / tint now use the viewer's model, not a fabricated Kelvin ─────────

@pytest.mark.real_torch
def test_tint_reaches_the_graded_pixels():
    """
    `tint` was sent by the viewer and read by nobody: apply_grade had no tint
    parameter at all, so the green/magenta axis was dropped from every master.
    """
    np = pytest.importorskip("numpy")
    from radiance.color.grading import apply_grading

    img = np.full((4, 4, 3), 0.5, dtype=np.float32)
    out = apply_grading(img, tint_shift=0.25)
    assert not np.allclose(out, img), "tint_shift did nothing"
    # The shader does `shift.g -= tint`: green down, red and blue untouched.
    assert out[0, 0, 1] < img[0, 0, 1] - 0.2
    assert abs(float(out[0, 0, 0] - img[0, 0, 0])) < 1e-6
    assert abs(float(out[0, 0, 2] - img[0, 0, 2])) < 1e-6


@pytest.mark.real_torch
def test_temp_shift_matches_the_shader_not_a_kelvin_curve():
    np = pytest.importorskip("numpy")
    from radiance.color.grading import apply_grading

    img = np.full((2, 2, 3), 0.5, dtype=np.float32)
    out = apply_grading(img, temp_shift=0.2)
    # applyTempTint: shift.r += temp; shift.b -= temp; green untouched.
    assert abs(float(out[0, 0, 0]) - 0.7) < 1e-5
    assert abs(float(out[0, 0, 1]) - 0.5) < 1e-6
    assert abs(float(out[0, 0, 2]) - 0.3) < 1e-5


@pytest.mark.real_torch
def test_white_balance_does_not_touch_alpha():
    np = pytest.importorskip("numpy")
    from radiance.color.grading import apply_grading

    rgba = np.full((2, 2, 4), 0.5, dtype=np.float32)
    out = apply_grading(rgba, temp_shift=0.3, tint_shift=-0.2)
    assert abs(float(out[0, 0, 3]) - 0.5) < 1e-6, "alpha was shifted by white balance"


def test_handler_no_longer_fabricates_kelvin_from_the_slider():
    py = _src("delivery/handler.py")
    assert "6500.0 + _temp_internal * 3500.0" not in py, (
        "the handler is back to inventing a Kelvin value from the viewer's "
        "additive [-2, 2] slider, which applies a different curve than the shader"
    )
    assert "temp_shift" in py and "tint_shift" in py
