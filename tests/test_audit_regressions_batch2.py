"""Regression tests for the Batch 2 (consolidation) audit fixes.

The theme of this batch is that the repo already contained a correct version of
almost everything it got wrong -- the fixes were consolidation, not invention.
These tests pin the invariants that would have caught each defect in one or two
lines, and that did not previously exist anywhere in the suite.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

# The stub is a real ModuleType whose attributes are all MagicMocks, so both
# `isinstance(torch, MagicMock)` and `hasattr(torch.zeros(1), "shape")` say
# "real torch" under it. Ask the stub directly instead.
_REAL_TORCH = not getattr(torch, "__radiance_stub__", False)
requires_torch = pytest.mark.skipif(not _REAL_TORCH, reason="requires real torch")

# This module gates torch per test, so conftest's automatic module-level
# skip must leave it alone: the source-text assertions below run fine
# against the stub and are worth keeping on the no-torch CI matrix.
RADIANCE_TORCH_GATED = True


# ── 1. Colour matrices: row sums ARE the white point ────────────────────────
#
# DWG_TO_XYZ and AWG4_TO_XYZ both had a wrong third row, so a neutral picked up
# a green/yellow cast. `M @ [1,1,1] == white_XYZ` is a one-line check that
# catches it, and no such check existed.

_D65_XYZ = np.array([0.95047, 1.00000, 1.08883])


@pytest.mark.parametrize("attr,cls_path", [
    ("DWG_TO_XYZ", "DaVinciWideGamut"),
    ("AWG4_TO_XYZ", "ARRIWideGamut4"),
])
def test_wide_gamut_matrix_white_point(attr, cls_path):
    from radiance.hdr import color as hc

    M = np.asarray(getattr(getattr(hc, cls_path), attr), dtype=np.float64)
    np.testing.assert_allclose(M.sum(axis=1), _D65_XYZ, atol=1e-3,
                               err_msg=f"{cls_path}.{attr} does not encode a D65 white point")


def test_srgb_matrix_white_point_still_holds():
    """The control: this matrix was always right and must stay right."""
    from radiance.hdr import color as hc

    M = np.asarray(hc._SRGB_TO_XYZ, dtype=np.float64)
    np.testing.assert_allclose(M.sum(axis=1), _D65_XYZ, atol=1e-4)


@pytest.mark.parametrize("cls_path,attr", [
    ("DaVinciWideGamut", "DWG_TO_XYZ"),
    ("ARRIWideGamut4", "AWG4_TO_XYZ"),
])
def test_neutral_survives_conversion_to_acescg(cls_path, attr):
    """
    AP1 is D60 and these spaces are D65; without chromatic adaptation a white
    input arrives in ACEScg as a visibly tinted triplet.
    """
    from radiance.hdr import color as hc

    M = np.asarray(getattr(getattr(hc, cls_path), attr), dtype=np.float64)
    adapted = (np.ones(3) @ M.T) @ np.asarray(hc._ADAPT_D65_TO_D60, np.float64).T
    out = adapted @ np.asarray(hc._XYZ_TO_AP1, np.float64).T
    assert out.max() - out.min() < 5e-3, f"neutral came out tinted: {out}"


# ── 2. Transfer functions: continuity, monotonicity, round trip, spec ───────
#
# Five piecewise curves were discontinuous or off-spec at their own cut point.
# Each is caught by evaluating at cut +/- eps -- two lines, and absent.

_SPEC_18_PCT = {
    "logc3": 0.391007,
    "slog3": 0.410,
    "vlog": 0.423,
    "canonlog3": 0.343371,
    "acescct": 0.413589,
    "davinci_intermediate": 0.336043,
    "log3g10": 1.0 / 3.0,
}


@pytest.mark.parametrize("name,expected", sorted(_SPEC_18_PCT.items()))
def test_transfer_curve_midgray_matches_spec(name, expected):
    from radiance.color import transfer as tr

    enc = getattr(tr, f"linear_to_{name}")
    got = float(enc(np.array([0.18], dtype=np.float32))[0])
    assert abs(got - expected) < 2e-3, f"{name}: 18% grey encodes to {got}, spec says {expected}"


@pytest.mark.parametrize("name", sorted(_SPEC_18_PCT) + ["logc4"])
def test_transfer_curve_is_continuous_and_monotonic(name):
    """No jump at the cut point, and no reversal anywhere in the domain."""
    from radiance.color import transfer as tr

    enc = getattr(tr, f"linear_to_{name}")
    x = np.linspace(0.0, 4.0, 200001).astype(np.float32)
    y = enc(x.copy())
    d = np.diff(y)
    assert bool((d >= -1e-6).all()), f"{name} encode is not monotonic"
    # A genuine cut-point discontinuity shows up as a step far larger than the
    # sampling grid's own step. DaVinci Intermediate's was 0.064.
    assert float(np.abs(d).max()) < 5e-3, f"{name} has a discontinuity at its cut point"


@pytest.mark.parametrize("name", sorted(_SPEC_18_PCT) + ["logc4"])
def test_transfer_curve_round_trips(name):
    from radiance.color import transfer as tr

    enc = getattr(tr, f"linear_to_{name}")
    dec = getattr(tr, f"{name}_to_linear")
    x = np.linspace(0.0, 4.0, 20001).astype(np.float32)
    np.testing.assert_allclose(dec(enc(x.copy())), x, atol=1e-4)


@requires_torch
@pytest.mark.parametrize("name", ["davinci_intermediate", "canonlog3", "slog3", "log3g10"])
def test_tensor_variant_matches_numpy(name):
    """The torch and numpy forms are separate code; they must not drift apart."""
    from radiance.color import transfer as tr

    x = np.linspace(-0.05, 4.0, 5001).astype(np.float32)
    np_enc = getattr(tr, f"linear_to_{name}")(x.copy())
    t_enc = getattr(tr, f"tensor_linear_to_{name}")(torch.from_numpy(x.copy())).numpy()
    np.testing.assert_allclose(np_enc, t_enc, atol=1e-5)


def test_luts_delegates_to_transfer():
    """
    color/luts.py used to carry its own copies of these curves. The copies
    disagreed with color/transfer.py, and only the broken ones were wired into
    the VAE and IO paths.
    """
    from radiance.color import luts, transfer as tr

    x = np.linspace(0.0, 4.0, 2001).astype(np.float32)
    np.testing.assert_array_equal(
        luts._lut_davinci_intermediate(x.copy()), tr.linear_to_davinci_intermediate(x.copy()))
    np.testing.assert_array_equal(
        luts._lut_log3g10(x.copy()), tr.linear_to_log3g10(x.copy()))


def test_display_luts_do_not_promote_to_float64():
    """Two LUTs returned float64 for float32 input, doubling frame memory."""
    from radiance.color import luts

    x = np.linspace(0.0, 4.0, 1001).astype(np.float32)
    for name, fn in luts._LUT_FUNCTIONS.items():
        if not callable(fn):
            continue
        out = fn(x.copy())
        assert out.dtype == np.float32, f"{name} promoted float32 input to {out.dtype}"


# ── 3. AgX must not flatten the highlights ──────────────────────────────────

@requires_torch
def test_agx_sigmoid_is_monotonic_above_midgrey():
    """
    The old curve reduced to x/(2x) == 0.5 for every input above mid-grey, so a
    sky at 5.0 and a specular at 100.0 tone-mapped identically.
    """
    pytest.importorskip("radiance.hdr.tonemap")

    p = 1.7
    x_log = torch.linspace(0.5, 1.0, 1000)
    d = x_log - 0.5
    x_sig = 0.5 + d / torch.pow(1.0 + torch.pow((2.0 * d.abs()).clamp(min=1e-8), p), 1.0 / p)
    assert len(torch.unique(x_sig.round(decimals=6))) > 500, \
        "highlights collapse to a single value"
    assert bool((torch.diff(x_sig) > 0).all()), "curve is not strictly increasing"


# ── 4. HDR tonescale: midtones must not track the peak ──────────────────────

def _tonescale(rgb, peak):
    from radiance.hdr.color import ACES2OutputTransform

    return ACES2OutputTransform()._apply_tonescale_drt(
        rgb, peak_luminance=peak, is_hdr=peak > 100
    )


def test_midgrey_tracks_the_aces_reference_not_the_peak():
    """
    18% grey rendered at 1.80 nits on a 1000-nit target and 0.45 on a 4000-nit
    one -- raising the peak made the whole picture darker.

    The original fix asserted the midtone was peak-INDEPENDENT (within 1 nit
    across peaks). That was the right instinct and slightly too strong: ACES
    2.0 publishes 14.512 nits at 1000 and 16.824 at 4000, so it does rise a
    little -- about 2.3 nits, deliberately -- while the highlights extend by
    3000. Pinning the published values covers the original regression and the
    "grey scales with peak" one in the same assertion.
    """
    from radiance.hdr.tonescale import aces2_midgrey_nits

    rgb = np.full((1, 1, 1, 3), 0.18, dtype=np.float32)
    for peak in (100.0, 1000.0, 4000.0):
        nits = float(_tonescale(rgb.copy(), peak)[0, 0, 0, 0] * 100.0)
        assert nits == pytest.approx(aces2_midgrey_nits(peak), abs=0.05), \
            f"18% grey at {nits:.2f} nits on a {peak:.0f}-nit target"


def test_tonescale_reaches_requested_peak():
    """The old shoulder topped out at 2137 nits when asked for 4000."""
    rgb = np.full((1, 1, 1, 3), 10000.0, dtype=np.float32)
    out = float(_tonescale(rgb.copy(), 4000.0)[0, 0, 0, 0] * 100.0)
    assert out > 3900.0, f"peak only reached {out} nits of 4000"


def test_tonescale_has_no_knee_discontinuity():
    """tanh(0)==0 made the two branches disagree by 10% of the white scale."""
    x = np.linspace(0.0, 40.0, 200001, dtype=np.float32)
    rgb = np.stack([x] * 3, axis=-1)[None, None]
    y = _tonescale(rgb, 1000.0)[0, 0, :, 0]
    d = np.diff(y)
    assert bool((d >= -1e-6).all()), "tonescale is not monotonic"
    assert float(np.abs(d).max()) < 1e-2, "step discontinuity at the shoulder knee"


# ── 5. Tiling: a border edge must never be ramped ───────────────────────────

@requires_torch
@pytest.mark.parametrize("H,W,tile,overlap", [
    (1024, 1024, 128, 16), (128, 128, 128, 16), (2048, 2048, 512, 128), (300, 500, 128, 32),
])
def test_tiled_accumulation_reconstructs_borders(H, W, tile, overlap):
    """
    Four independent tilers ramped from exactly 0 on image-border edges. Since
    the accumulated weight is divided out, those pixels normalised to 0 -- a
    black frame around the output.
    """
    from radiance.core.tiling import blend_weight_2d, edge_overlaps_from_coords

    step = max(1, tile - overlap)
    out = torch.zeros(1, 1, H, W)
    acc = torch.zeros(1, 1, H, W)
    coords = []
    for y in range(0, H, step):
        for x in range(0, W, step):
            y1 = min(y, H - tile) if H > tile else 0
            x1 = min(x, W - tile) if W > tile else 0
            coords.append((y1, min(y1 + tile, H), x1, min(x1 + tile, W)))
    for (y1, y2, x1, x2) in dict.fromkeys(coords):
        ot, ob, ol, orr = edge_overlaps_from_coords(y1, y2, x1, x2, H, W, overlap)
        w = blend_weight_2d(y2 - y1, x2 - x1, ot, ob, ol, orr)
        out[:, :, y1:y2, x1:x2] += 0.5 * w
        acc[:, :, y1:y2, x1:x2] += w

    assert float(acc.min()) > 1e-3, "some pixel accumulated ~zero weight"
    res = (out / acc)[0, 0]
    np.testing.assert_allclose(res.numpy(), 0.5, atol=1e-5,
                               err_msg="constant input was not reconstructed exactly")


@requires_torch
def test_border_edges_get_no_ramp():
    from radiance.core.tiling import edge_overlaps_from_coords

    # top-left tile of a grid: top and left are borders
    assert edge_overlaps_from_coords(0, 128, 0, 128, 512, 512, 16)[:3:2] == (0, 0)
    # interior tile: every edge ramps
    assert all(v > 0 for v in edge_overlaps_from_coords(128, 256, 128, 256, 512, 512, 16))


def test_overlap_is_clamped_below_half_tile():
    """overlap >= tile_size collapses the stride to 1 -- effectively a hang."""
    from radiance.core.tiling import clamp_overlap

    assert clamp_overlap(128, 512) == 64
    assert clamp_overlap(128, 16) == 16
    assert clamp_overlap(128, 64) == 64


# ── 6. Model caches must be bounded ─────────────────────────────────────────

def test_lru_cache_survives_size_zero():
    """RADIANCE_CACHE_SIZE=0 used to raise KeyError out of the loader."""
    from radiance.model.cache import LRUCache

    c = LRUCache(max_size=0)
    c.put("a", object())      # must not raise
    assert len(c) == 0


def test_lru_cache_evicts():
    from radiance.model.cache import LRUCache

    c = LRUCache(max_size=2)
    for k in "abc":
        c.put(k, k)
    assert len(c) == 2 and c.get("a") is None and c.get("c") == "c"


def test_gpu_cache_moves_evicted_model_off_device():
    from radiance.model.cache import GPUModelCache

    moved = []

    class FakeModel:
        def to(self, device):
            moved.append(device)
            return self

    c = GPUModelCache(max_size=1)
    c.put("m1", FakeModel())
    c.put("m2", FakeModel())
    assert moved == ["cpu"], "evicted model was dropped without leaving VRAM"


def test_no_unbounded_module_level_model_caches():
    """Guard against a new `_X_CACHE = {}` creeping back in."""
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parent.parent
    pat = re.compile(r"^_[A-Z0-9_]*CACHE[^=\n]*=\s*\{\s*\}\s*$", re.M)
    offenders = []
    for path in root.rglob("*.py"):
        if set(path.parts) & {"tests", "build", "dist", ".codex-backup-20260719", "scripts"}:
            continue
        for m in pat.finditer(path.read_text(encoding="utf-8", errors="ignore")):
            offenders.append(f"{path.relative_to(root)}: {m.group(0).strip()}")
    assert not offenders, f"unbounded module-level caches: {offenders}"
