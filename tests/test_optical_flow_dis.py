"""The DIS solver, and the ~8 px ceiling it was added to get past.

Pyramidal Lucas-Kanade is bounded by its integration window: the coarsest
pyramid level has to stay larger than the 15x15 window, so coarse-to-fine has a
hard reach, and past it the *field* thins out while the median stays roughly
right. That is the worst way for a flow solver to fail, because the number you
would check looks fine and the mask you propagate tears.

Measured on the same multi-scale plate `test_optical_flow.py` uses, fraction of
the interior landing within half a pixel:

    px      3     8    12    16    20
    LK    100%   84%   71%   58%   29%
    DIS   100%  100%  100%  100%   99%

On an aperiodic plate — closer to grain and real detail than overlapping
sinusoids — LK is down to 9% by 12 px while DIS is still at 100%.

Both fail past roughly 28 px on these fixtures, but that is the fixture rather
than the solver: at that displacement the patterns are ambiguous. The claim
worth making is that the working range went from about 8 px to about 20.
"""
import pathlib

import pytest
import torch

cv2 = pytest.importorskip("cv2")

from radiance.nodes.vfx.multipass.core import (  # noqa: E402
    _optical_flow,
    _optical_flow_dis,
    _optical_flow_lk,
)


@pytest.fixture(scope="module")
def plate():
    h, w = 96, 260
    ys, xs = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing="ij",
    )
    return (0.5
            + 0.20 * torch.sin(xs / 7.0)
            + 0.15 * torch.cos(ys / 5.0)
            + 0.10 * torch.sin((xs + ys) / 11.0)
            + 0.05 * torch.sin(xs / 2.5)).unsqueeze(0)


def _pair(plate, dx=0, dy=0, h=64, w=128):
    return plate[:, dy:dy + h, dx:dx + w], plate[:, 0:h, 0:w]


def _density(u, shift, margin=20):
    """The property mask propagation needs: most of the field, not the mean."""
    interior = u[:, margin:-margin, margin:-margin]
    return float(((interior - shift).abs() < 0.5).float().mean())


# ── the reason it exists ────────────────────────────────────────────────────

@pytest.mark.parametrize("shift", [12, 16, 20])
def test_dis_holds_a_dense_field_where_lucas_kanade_thins_out(plate, shift):
    a, b = _pair(plate, dx=shift)
    dis = _density(_optical_flow_dis(a, b)[0], shift)
    lk = _density(_optical_flow_lk(a, b)[0], shift)
    assert dis >= 0.95, f"{shift} px: DIS field only {dis * 100:.0f}% within 0.5 px"
    assert dis > lk + 0.2, (
        f"{shift} px: DIS {dis * 100:.0f}% vs LK {lk * 100:.0f}% — the new solver "
        "is not buying what it was added for"
    )


@pytest.mark.parametrize("shift", [3, 8, 12, 16, 20])
def test_dis_recovers_the_displacement(plate, shift):
    a, b = _pair(plate, dx=shift)
    u, _ = _optical_flow_dis(a, b)
    assert float(u[:, 20:-20, 20:-20].median()) == pytest.approx(shift, rel=0.05)


# ── the thing that would silently break every caller ────────────────────────

def test_dis_uses_the_same_sign_convention_as_lucas_kanade(plate):
    """OpenCV's calc(I0, I1) is flow I0 to I1. Get the argument order wrong and
    the field is exactly negated, which warps every mask the wrong way while
    looking like a plausible flow field."""
    a, b = _pair(plate, dx=8)
    u_dis, _ = _optical_flow_dis(a, b)
    u_lk, _ = _optical_flow_lk(a, b)
    assert float(u_dis.median()) * float(u_lk.median()) > 0, "the solvers disagree about sign"


def test_a_vertical_move_lands_in_v_and_not_u(plate):
    a, b = _pair(plate, dy=6)
    u, v = _optical_flow_dis(a, b)
    assert float(v[:, 20:-20, 20:-20].median()) == pytest.approx(6, rel=0.1)
    assert abs(float(u[:, 20:-20, 20:-20].median())) < 0.5


# ── HDR, which is the whole point of this package ───────────────────────────

def test_scene_linear_input_survives_the_8_bit_quantisation(plate):
    """DIS only takes 8-bit, so the pair is quantised on the way in. The
    normalisation has to be computed across both frames together: scaling each
    by its own min and max moves the picture between them and invents flow."""
    hdr = plate * 40.0 - 3.0          # over-range and negative
    a, b = _pair(hdr, dx=16)
    u, _ = _optical_flow_dis(a, b)
    assert float(u[:, 20:-20, 20:-20].median()) == pytest.approx(16, rel=0.05)
    assert _density(u, 16) >= 0.95


def test_a_pair_scaled_differently_per_frame_would_be_caught():
    """Guards the shared-normalisation rule directly: a constant exposure
    offset between the two frames must not become flow."""
    g = torch.Generator().manual_seed(0)
    base = torch.rand(1, 64, 96, generator=g)
    a = base * 4.0
    b = base * 4.0 + 1.5              # same picture, one stop brighter
    u, v = _optical_flow_dis(a, b)
    assert float(u.abs().median()) < 0.5, "an exposure change was read as motion"
    assert float(v.abs().median()) < 0.5


# ── degenerate input ────────────────────────────────────────────────────────

def test_no_second_frame_is_zero_flow(plate):
    u, v = _optical_flow_dis(plate[:, :64, :128], None)
    assert u.shape == (1, 64, 128) and torch.count_nonzero(u) == 0
    assert torch.count_nonzero(v) == 0


def test_a_flat_pair_reports_no_motion_rather_than_noise():
    flat = torch.full((1, 32, 32), 0.5)
    u, v = _optical_flow_dis(flat, flat)
    assert float(u.abs().max()) == 0.0 and float(v.abs().max()) == 0.0


def test_non_finite_input_does_not_produce_non_finite_flow():
    nan = torch.full((1, 32, 32), float("nan"))
    u, v = _optical_flow_dis(nan, nan)
    assert torch.isfinite(u).all() and torch.isfinite(v).all()


def test_a_batch_comes_back_as_a_batch(plate):
    a, b = _pair(plate, dx=8)
    u, v = _optical_flow_dis(a.repeat(3, 1, 1), b.repeat(3, 1, 1))
    assert u.shape[0] == 3 and v.shape[0] == 3
    assert torch.allclose(u[0], u[2])


# ── the dispatcher ──────────────────────────────────────────────────────────

def test_auto_picks_dis_when_opencv_is_available(plate):
    a, b = _pair(plate, dx=16)
    assert torch.equal(_optical_flow(a, b, method="auto")[0], _optical_flow_dis(a, b)[0])


def test_the_solver_can_still_be_pinned_to_lucas_kanade(plate):
    a, b = _pair(plate, dx=8)
    assert torch.equal(
        _optical_flow(a, b, method="lucas-kanade")[0],
        _optical_flow_lk(a, b)[0],
    )


def test_the_quality_dial_reaches_dis_and_not_only_lucas_kanade():
    """The Optical Flow node has one Fast / Medium / Ultra widget and two
    solvers behind it. When DIS became the default, that widget only fed
    Lucas-Kanade's integration radius, so all three positions did the same
    thing and nothing said so. Three labels have to map to three distinct DIS
    presets, in the same order.
    """
    import ast
    src = pathlib.Path(__file__).resolve().parent.parent / "nodes" / "vfx" / "motion.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))
    mapping = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict) and {"Fast", "Medium", "Ultra"} == {
            k.value for k in node.keys if isinstance(k, ast.Constant)
        }:
            values = [v.value for v in node.values if isinstance(v, ast.Constant)]
            if all(isinstance(v, str) for v in values):
                mapping = dict(zip(
                    [k.value for k in node.keys if isinstance(k, ast.Constant)], values))
    assert mapping is not None, "no Fast/Medium/Ultra to DIS preset mapping in motion.py"
    assert len(set(mapping.values())) == 3, (
        f"the three widget positions collapse to {sorted(set(mapping.values()))} — "
        "at least two of them do nothing"
    )
    order = ["ultrafast", "fast", "medium"]
    assert [order.index(mapping[k]) for k in ("Fast", "Medium", "Ultra")] == [0, 1, 2], (
        f"the dial is not monotonic in quality: {mapping}"
    )


def test_every_named_preset_is_one_dis_actually_has(plate):
    """A preset name DIS does not know falls back to medium silently, which is
    the same widget-does-nothing failure wearing a different hat."""
    a, b = _pair(plate, dx=8)
    for name in ("ultrafast", "fast", "medium"):
        u, _ = _optical_flow_dis(a, b, preset=name)
        assert float(u[:, 20:-20, 20:-20].median()) == pytest.approx(8, rel=0.05), name


def test_the_dispatcher_passes_the_preset_through(plate, monkeypatch):
    seen = {}
    import radiance.nodes.vfx.multipass.core as core
    real = core._optical_flow_dis
    monkeypatch.setattr(core, "_optical_flow_dis",
                        lambda f1, f2, preset="medium": seen.setdefault("preset", preset)
                        or real(f1, f2, preset=preset))
    a, b = _pair(plate, dx=8)
    core._optical_flow(a, b, method="dis", preset="ultrafast")
    assert seen["preset"] == "ultrafast"


def test_the_dispatcher_falls_back_when_opencv_is_missing(plate, monkeypatch):
    """A broken environment should lose accuracy, not raise."""
    import builtins
    real_import = builtins.__import__

    def no_cv2(name, *a, **kw):
        if name == "cv2":
            raise ImportError("simulated: no OpenCV")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", no_cv2)
    a, b = _pair(plate, dx=8)
    u, _ = _optical_flow(a, b, method="auto")
    assert torch.equal(u, _optical_flow_lk(a, b)[0])
