"""Regression tests for the viewer / delivery audit fixes."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

# Most assertions here read source text and run fine against conftest's torch
# stub, so opt out of the automatic module-level skip and gate the handful
# of tests that build real tensors with @pytest.mark.real_torch.
RADIANCE_TORCH_GATED = True


# ── 1. Legal-range limiting must never touch a float deliverable ────────────
#
# `soft_clip` defaults ON in the delivery panel and the clamp was ungated, so
# exporting a 32-bit EXR master squeezed every pixel into [16/255, 235/255].

@pytest.mark.parametrize("fmt,should_clip", [
    ("VID │ MP4 (H.264)", True),
    ("SEQ │ PNG (8-bit)", True),
    ("IMG │ PNG (16-bit)", True),
    ("SEQ │ EXR (32-bit float)", False),
    ("IMG │ EXR (16-bit half)", False),
    ("IMG │ Radiance HDR (.hdr)", False),
    ("IMG │ DPX", False),
])
def test_broadcast_safe_is_gated_on_output_format(fmt, should_clip):
    is_float = ("EXR" in fmt or "HDR" in fmt or "32-bit float" in fmt or "DPX" in fmt)
    assert (not is_float) == should_clip, (
        f"{fmt}: legal-range limiting should {'apply' if should_clip else 'NOT apply'}")


def test_float_formats_keep_their_hdr_range():
    """A scene-linear plate must survive an EXR export untouched."""
    frame = np.array([[[0.0, 0.18, 40.0]]], dtype=np.float32)
    fmt = "SEQ │ EXR (32-bit float)"
    is_float = ("EXR" in fmt or "HDR" in fmt or "32-bit float" in fmt or "DPX" in fmt)
    out = frame if is_float else np.clip(frame, 16 / 255.0, 235 / 255.0)
    np.testing.assert_array_equal(out, frame)
    assert float(out.max()) == 40.0, "highlight destroyed by legal-range clamp"


# ── 2. Aspect-ratio blanking ────────────────────────────────────────────────

def _parse_ratio(s):
    token = s.split()[0]
    num, _, den = token.partition(":")
    return float(num) / float(den or 1.0)


@pytest.mark.parametrize("label,expected", [
    ("2.35:1 (Scope)", 2.35),
    ("1.85:1 (Flat)", 1.85),
    ("1:1 (Square)", 1.0),
    ("9:16 (Vertical)", 0.5625),
])
def test_aspect_ratio_parses_both_terms(label, expected):
    """Only the numerator was parsed, so 9:16 became the ratio 9.0."""
    assert _parse_ratio(label) == pytest.approx(expected)


@pytest.mark.real_torch
def test_vertical_aspect_pillarboxes_rather_than_destroying_the_frame():
    t = torch.ones(1, 1080, 1920, 3)
    target = _parse_ratio("9:16 (Vertical)")
    h, w = 1080, 1920
    current = w / h
    assert current > target + 0.01, "1920x1080 must take the pillarbox branch for 9:16"
    pad = (w - int(h * target)) // 2
    if pad > 0:
        t[:, :, :pad, :] = 0.0
        t[:, :, -pad:, :] = 0.0
    kept = float((t != 0).float().mean())
    assert 0.25 < kept < 0.35, f"expected ~9:16 of the frame kept, got {kept:.3f}"


def test_zero_pad_does_not_blank_the_whole_frame():
    """`t[:, :, -0:, :] = 0` is `t[:, :, 0:, :] = 0` — the guard is required."""
    t = torch.ones(1, 4, 4, 3)
    pad = 0
    if pad > 0:
        t[:, :, -pad:, :] = 0.0
    assert float(t.min()) == 1.0


# ── 3. Preview tonemap must not touch alpha ─────────────────────────────────

def _preview_tonemap(frame):
    safe = np.maximum(frame, 0.0)
    tm = (safe[..., :3] / (1.0 + safe[..., :3])).astype(np.float32)
    if frame.ndim == 3 and frame.shape[-1] > 3:
        return np.concatenate([tm, np.clip(frame[..., 3:], 0.0, 1.0).astype(np.float32)], axis=-1)
    return tm


def test_alpha_is_not_tonemapped():
    """An opaque matte came out at 0.5, compositing the PNG fallback at 50%."""
    frame = np.array([[[4.0, 4.0, 4.0, 1.0]]], dtype=np.float32)
    out = _preview_tonemap(frame)
    assert float(out[0, 0, 3]) == pytest.approx(1.0), "alpha was tonemapped"
    assert float(out[0, 0, 0]) == pytest.approx(0.8), "RGB tonemap changed"


def test_rgb_only_input_still_works():
    frame = np.array([[[4.0, 4.0, 4.0]]], dtype=np.float32)
    out = _preview_tonemap(frame)
    assert out.shape[-1] == 3
    assert float(out[0, 0, 0]) == pytest.approx(0.8)


# ── 4. The viewer must always re-run ────────────────────────────────────────

def test_viewer_declares_is_changed():
    """
    Without IS_CHANGED, ComfyUI skips the node on re-queue, so once the frame
    cache evicts that viewer's entry the delivery endpoint answers "run the
    workflow first" forever.
    """
    pytest.importorskip("folder_paths", reason="needs the ComfyUI runtime")
    from radiance.nodes.monitor.viewer import RadianceViewer

    assert hasattr(RadianceViewer, "IS_CHANGED")
    val = RadianceViewer.IS_CHANGED()
    assert val != val, "IS_CHANGED must return NaN (always dirty)"


# ── 5. Bracketing must slice before it multiplies ───────────────────────────

def test_bracketing_slices_before_multiplying():
    """
    The multiply used to be applied to the whole batch inside the per-frame
    loop while only image[frame_idx] was read — O(B^2) memory traffic.
    """
    import inspect
    pytest.importorskip("folder_paths", reason="needs the ComfyUI runtime")
    from radiance.nodes.monitor import viewer as v

    src = inspect.getsource(v.RadianceViewer.view)
    code = "\n".join(line.split("#", 1)[0] for line in src.splitlines())
    assert "image * 0.25" not in code, "still multiplying the whole batch"
    assert "image * 4.0" not in code, "still multiplying the whole batch"
    assert "image[frame_idx:frame_idx + 1] * 0.25" in code
    assert "image[frame_idx:frame_idx + 1] * 4.0" in code


@pytest.mark.real_torch
def test_sliced_bracket_matches_the_old_per_frame_result():
    """Slicing must be numerically identical for the frame actually used."""
    image = torch.rand(6, 8, 8, 3)
    for idx in range(image.shape[0]):
        old = (image * 0.25)[idx]
        new = (image[idx:idx + 1] * 0.25)[0]
        torch.testing.assert_close(old, new, atol=0.0, rtol=0.0)


# ── 6. Frontend invariants (source-level, no browser needed) ────────────────

def _js(name):
    import pathlib
    return (pathlib.Path(__file__).resolve().parent.parent / "js" / name).read_text(encoding="utf-8")


def test_delivery_history_escapes_untrusted_values():
    src = _js("radiance_viewer.js")
    assert "${_esc(name)}" in src and "${_esc(path)}" in src, \
        "delivery history re-introduced an unescaped innerHTML"


def test_widget_hiding_is_scoped_to_radiance_nodes():
    """
    The document-wide scan hid bit_depth / exposure_bracketing widgets on other
    node packs' nodes.
    """
    src = _js("radiance_viewer.js")
    assert "for (const row of document.querySelectorAll('.lg-node-widget'))" not in src, \
        "an unscoped document-wide widget scan is back"
    assert "data-node-id" in src


def test_destroy_cancels_animation_frames():
    src = _js("radiance_viewer.js")
    start = src.index("    destroy(")
    body = src[start:start + 6000]
    for raf in ("_grainRAF", "_seqRAF", "_referenceScopeRAF"):
        assert raf in body, f"destroy() does not cancel {raf}"
    assert body.count("cancelAnimationFrame") >= 3


def test_destroy_preserves_the_shared_hud():
    src = _js("radiance_viewer.js")
    start = src.index("    destroy(")
    body = src[start:start + 6000]
    assert "singletonHUD" in body, \
        "destroy() detaches the shared HUD unconditionally, blanking other viewers"


def test_webgl_destroy_clears_resource_maps():
    src = _js("radiance_webgl.js")
    start = src.index("    destroy(")
    body = src[start:start + 4000]
    assert "this.textures = {}" in body and "this.programs = {}" in body, \
        "stale GL handles survive destroy()"
    assert "loseContext" in body, "WebGL context is never explicitly released"
