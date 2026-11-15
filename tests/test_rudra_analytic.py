"""RUDRA's analytic SDR -> HDR route (radiance.model.rudra_analytic) against RUDRA itself.

The goldens in tests/fixtures/rudra_analytic are the RUDRA native core's own
output (FXTD-Studios/RUDRA commit 74f03b6, built and run on these frames):
the clip fill, the composite, the anchor, chroma carry, grain settle and
exposure, as the RUDRA app's master writes them. The port uses Euclidean
distances where the app uses a chamfer, so it is held to a tolerance, in
stops, rather than bit for bit.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("cv2")

from radiance.model import rudra_analytic as ra  # noqa: E402

FIX = Path(__file__).parent / "fixtures" / "rudra_analytic"
INDEX = json.loads((FIX / "index.json").read_text(encoding="utf-8"))
GOLD = np.load(FIX / "golden.npz")

# Measured on the committed goldens: worst pixel 0.057 stops, worst case mean 0.0009.
MAX_STOPS, MEAN_STOPS, P99_STOPS = 0.08, 0.003, 0.03


def _sdr(name):
    return GOLD[f"sdr_{name}"].astype(np.float32) / np.float32(255.0)


def _options(o):
    o = dict(o)
    peaks = o.pop("clip_peaks", [])
    return ra.AnalyticOptions(**o, clip_peaks=[ra.ClipPeak(*p) for p in peaks])


@pytest.mark.parametrize("case", sorted(INDEX["cases"]))
def test_matches_rudra_native(case):
    meta = INDEX["cases"][case]
    r = ra.analytic_master(_sdr(meta["sdr"]), _options(meta["options"]))
    gold = GOLD[f"lin_{case}"].astype(np.float32) / 1024.0
    mine = np.log2(np.maximum(r.image, 0) + 1e-4)
    err = np.abs(mine - gold)
    assert err.max() < MAX_STOPS, f"{case}: worst pixel {err.max():.3f} stops from RUDRA"
    assert err.mean() < MEAN_STOPS, f"{case}: mean {err.mean():.4f} stops from RUDRA"
    assert np.percentile(err, 99) < P99_STOPS
    assert r.info.partial_pct == pytest.approx(meta["partial_pct"], abs=1e-3)
    assert r.info.full_pct == pytest.approx(meta["full_pct"], abs=1e-3)
    assert sorted(g.pixels for g in r.info.regions if g.overridden) == sorted(meta["overridden_regions"])


def test_every_source_curve_has_a_monotone_inverse():
    for curve in ra.SOURCE_CURVES:
        ramp = np.repeat(np.linspace(0, 1, 256, dtype=np.float32)[:, None], 3, 1)
        v = ra.baseline_network(ramp, curve)[:, 0]
        assert np.all(np.diff(v[1:]) >= -1e-9), curve
    assert not ra.curve_knots("aces").any(), "ACES is the inverse itself: no correction"


def test_bt1886_puts_sdr_white_at_203_nits():
    nits = float(ra.baseline_network(np.full((1, 1, 3), 254 / 255, np.float32), "bt1886")[0, 0, 0]) * 10000
    assert 190 < nits < 203


def test_strength_zero_and_fill_off_are_the_plain_inverse():
    sdr = _sdr("synth")
    plain = ra.analytic_master(sdr, ra.AnalyticOptions(clip_fill=False, anchor=False, carry_chroma=False,
                                                       settle_grain=False))
    zero = ra.analytic_master(sdr, ra.AnalyticOptions(strength=0.0, anchor=False, carry_chroma=False,
                                                      settle_grain=False))
    assert np.allclose(plain.image, zero.image, rtol=1e-5, atol=1e-8)
    assert not plain.filled.any()
    assert plain.estimated.any(), "fully clipped pixels are reported even with the fill off"
    filled = ra.analytic_master(sdr, ra.AnalyticOptions(anchor=False, carry_chroma=False, settle_grain=False))
    on = filled.filled > 0
    assert on.any()
    assert np.array_equal(filled.image[~on], plain.image[~on]), "the fill touches only the pixels it reports"
    assert (filled.image[on].max(-1) >= plain.image[on].max(-1) - 1e-6).all(), "never below what clipped"


def test_exposure_is_a_plain_gain():
    sdr = _sdr("dark")
    a = ra.analytic_master(sdr, ra.AnalyticOptions())
    b = ra.analytic_master(sdr, ra.AnalyticOptions(exposure_ev=1.0))
    assert np.allclose(b.image, a.image * 2.0, rtol=1e-5, atol=1e-7)


def test_nothing_clipped_means_no_fill():
    sdr = _sdr("dark")
    r = ra.analytic_master(sdr, ra.AnalyticOptions())
    assert r.info.gain_log2 is None and not r.filled.any() and not r.estimated.any()


def test_highlight_detail_only_acts_near_white():
    sdr = _sdr("sunset")
    off = ra.near_clip_detail(sdr, 1.0)
    on = ra.near_clip_detail(sdr, 2.0)
    assert off is None
    luma = sdr @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    assert not on[luma < 0.86].any()
    assert np.abs(on).max() <= 1.0


def test_fill_missing_keeps_known_values_and_fills_the_rest():
    planes = np.zeros((1, 9, 9), np.float32)
    planes[0, :, :3] = 1.0
    wt = np.zeros((9, 9), np.float32)
    wt[:, :3] = 1.0
    out = ra.fill_missing(planes, wt)
    assert np.array_equal(out[0, :, :3], planes[0, :, :3])
    assert np.allclose(out[0, :, 3:], 1.0, atol=1e-5), "a constant field fills as itself"


def test_clip_peaks_parse_and_refuse():
    assert ra.parse_clip_peaks("") == []
    assert ra.parse_clip_peaks('{"x": 0.5, "y": 0.25, "nits": 2500}') == [ra.ClipPeak(0.5, 0.25, 2500.0)]
    with pytest.raises(ValueError):
        ra.parse_clip_peaks("[{'x': 1}]")
    with pytest.raises(ValueError):
        ra.parse_clip_peaks('[{"x": 2, "y": 0.5, "nits": 1000}]')


def test_unknown_curve_is_refused():
    with pytest.raises(ValueError):
        ra.analytic_master(_sdr("dark"), ra.AnalyticOptions(source_curve="sRGB"))


def _real_torch():
    try:
        import torch
    except ImportError:
        return None
    return torch if getattr(torch, "__file__", None) and hasattr(torch, "from_numpy") else None


@pytest.mark.skipif(_real_torch() is None, reason="needs real torch")
def test_node_runs_a_batch_and_reports():
    torch = _real_torch()
    from radiance.nodes.hdr.rudra_analytic import NODE_CLASS_MAPPINGS, RadianceSDRToHDRAnalytic
    assert "RadianceSDRToHDRAnalytic" in NODE_CLASS_MAPPINGS
    inputs = RadianceSDRToHDRAnalytic.INPUT_TYPES()
    assert inputs["required"]["source_curve"][1]["default"].startswith("BT.1886")
    sdr = torch.from_numpy(np.stack([_sdr("synth"), _sdr("synth")]))
    node = RadianceSDRToHDRAnalytic()
    image, filled, estimated, report, mattes, regions = node.convert(
        sdr, "BT.1886 (graded / broadcast video)", 1.0, True, 1.0, 0.0, "Linear (Rec.709)",
        clip_peaks='[{"x": 0.75, "y": 0.6667, "nits": 2500}]')
    assert image.shape == sdr.shape and filled.shape == sdr.shape[:3] and estimated.shape == sdr.shape[:3]
    direct = ra.analytic_master(_sdr("synth"), ra.AnalyticOptions(clip_peaks=[ra.ClipPeak(0.75, 0.6667, 2500)]))
    assert np.allclose(image[0].numpy(), direct.image, atol=1e-6)
    assert "set by clip_peaks" in report and "BT.1886" in report
    assert len(mattes["frames"]) == 2 and mattes["encoding"] == "Linear (Rec.709)"
    assert regions.shape == sdr.shape and "pixels: preserved" in report
    pq, *_ = node.convert(sdr[:1], "Filmic (Hable)", 1.0, True, 1.0, 0.0, "PQ (HDR10)")
    assert float(pq.max()) <= 0.7518 + 1e-3, "PQ stays under the 1000-nit peak"


# ── phase 2: region, calibration, reference, Region EV, mask ────────────────
GOLD2 = np.load(FIX / "golden_phase2.npz")
P2 = INDEX["phase2"]


def _gold2(name):
    return GOLD2[f"lin_{name}"].astype(np.float32) / 1024.0


def _stops(img, gold):
    return np.abs(np.log2(np.maximum(img, 0) + 1e-4) - gold)


@pytest.mark.parametrize("case", sorted(P2["cases"]))
def test_phase2_matches_rudra_native(case):
    meta = P2["cases"][case]
    o = dict(meta["options"])
    sdr = _sdr(meta["sdr"])
    if "region" in o:
        o["region"] = ra.Region(**o["region"])
    if "calibration" in o:
        o["calibration"] = [ra.CalibrationPoint(int(c), float(n)) for c, n in o["calibration"]]
    if o.pop("reference", False):
        o["reference"] = ra.fit_reference(sdr, GOLD2["reference_sunset"].astype(np.float32))
    r = ra.analytic_master(sdr, ra.AnalyticOptions(**o))
    err = _stops(r.image, _gold2(case))
    assert err.max() < MAX_STOPS and err.mean() < MEAN_STOPS, f"{case}: {err.max():.3f} / {err.mean():.4f} stops"


def test_reference_fit_matches_rudra():
    fit = ra.fit_reference(_sdr("sunset"), GOLD2["reference_sunset"].astype(np.float32))
    want = P2["reference_fit"]
    assert abs(fit.codes_seen - want["codes_seen"]) <= 2
    assert ra.reference_exposure(fit, "bt1886") == pytest.approx(want["exposure"], abs=0.02)
    assert fit.residual_mean == pytest.approx(want["residual_mean"], abs=0.01)
    assert np.all(np.diff(fit.target_log2_nits) >= 0), "the fitted curve is monotone"


def test_non_monotone_anchors_are_refused():
    bad = [ra.CalibrationPoint(60, 50.0), ra.CalibrationPoint(200, 10.0)]
    assert not ra.calibration_is_monotone(bad, "bt1886")
    with pytest.raises(ValueError, match="monotone"):
        ra.analytic_master(_sdr("sunset"), ra.AnalyticOptions(calibration=bad))


def test_one_anchor_is_an_exposure_shift():
    k = ra.calibration_knots([ra.CalibrationPoint(128, 40.0)], "bt1886")
    assert np.allclose(k, k[0])
    nits = float(ra.source_code_nits("bt1886", 128)) * 2 ** float(k[128])
    assert nits == pytest.approx(40.0, rel=1e-4), "exact at the anchor"


def test_calibration_by_position_reads_the_frame():
    sdr = _sdr("dark")
    pts = ra.parse_calibration('[{"x": 0.5, "y": 0.5, "nits": 2}]', sdr)
    h, w = sdr.shape[:2]
    want = round(float((np.clip(sdr[h // 2, w // 2], 0, 1) * np.array([0.2126, 0.7152, 0.0722])).sum()) * 255)
    assert pts == [ra.CalibrationPoint(want, 2.0)]
    with pytest.raises(ValueError):
        ra.parse_calibration('[{"code": 1, "nits": 1}, {"code": 2, "nits": 2}, {"code": 3, "nits": 3}, {"code": 4, "nits": 4}]')


def test_region_ev_matches_rudra():
    base = (2.0 ** (GOLD["lin_sunset_default"].astype(np.float64) / 1024.0) - 1e-4) * 203.0
    bands = [ra.RegionBand(*b) for b in P2["region_ev"]["bands"]]
    out = ra.apply_region_ev(base, bands, 1.0, exposure_ev=0.3) / 203.0
    assert _stops(out, _gold2("regionev")).max() < 0.005
    masks = [GOLD2["mask0"] / 255.0, None, GOLD2["mask2"] / 255.0]
    out = ra.apply_region_ev(base, bands, 0.5, masks=masks) / 203.0
    assert _stops(out, _gold2("regionev_masks")).max() < 0.005


def test_region_ev_at_zero_changes_nothing():
    base = np.random.default_rng(1).uniform(0, 5000, (8, 8, 3))
    assert ra.apply_region_ev(base, ra.DEFAULT_REGION_BANDS, 1.0) is base


def test_outside_the_region_and_the_mask_is_the_plain_inverse_bit_for_bit():
    sdr = _sdr("synth")
    plain = ra.analytic_master(sdr, ra.AnalyticOptions(clip_fill=False, anchor=False, carry_chroma=False,
                                                       settle_grain=False)).image
    band = ra.Region(band=True, band_low=0.6, band_high=0.95, band_softness=0.05)
    r = ra.analytic_master(sdr, ra.AnalyticOptions(region=band, anchor=False, carry_chroma=False, settle_grain=False))
    out = r.region_weight == 0
    assert out.any() and np.array_equal(r.image[out], plain[out])
    limit = np.zeros(sdr.shape[:2], np.float32)
    limit[:, : sdr.shape[1] // 2] = 1.0
    m = ra.analytic_master(sdr, ra.AnalyticOptions(limit=limit, anchor=False, carry_chroma=False, settle_grain=False))
    assert np.array_equal(m.image[:, sdr.shape[1] // 2:], plain[:, sdr.shape[1] // 2:])
    assert not m.filled[:, sdr.shape[1] // 2:].any()
    assert (m.image[:, : sdr.shape[1] // 2] != plain[:, : sdr.shape[1] // 2]).any()


def test_bad_region_is_refused():
    with pytest.raises(ValueError, match="band"):
        ra.analytic_master(_sdr("dark"), ra.AnalyticOptions(region=ra.Region(band=True, band_low=0.9, band_high=0.5)))


@pytest.mark.skipif(_real_torch() is None, reason="needs real torch")
def test_phase2_nodes_wire_together():
    torch = _real_torch()
    from radiance.nodes.hdr.rudra_analytic import (
        RadianceRudraCalibrate, RadianceRudraRegion, RadianceRudraRegionEV, RadianceSDRToHDRAnalytic)
    sdr = torch.from_numpy(_sdr("sunset")[None])
    (region,) = RadianceRudraRegion().build("Band", 0.75, 0.1, 1.0, 0.15, 0.1, 1.0, 0.85, 1.0, 0.1)
    cal, summary = RadianceRudraCalibrate().build(sdr, '[{"code": 128, "nits": 40}]', 0)
    assert "code 128 = 40 nits" in summary
    ref = torch.from_numpy(GOLD2["reference_sunset"].astype(np.float32)[None])
    cal_ref, summary_ref = RadianceRudraCalibrate().build(sdr, "", 0, reference=ref)
    assert "reference fit" in summary_ref and "reference" in cal_ref
    mask = torch.zeros(1, 180, 320)
    mask[:, :, :160] = 1
    node = RadianceSDRToHDRAnalytic()
    image, filled, _, report, *_ = node.convert(sdr, "BT.1886 (graded / broadcast video)", 1.0, True, 1.0, 0.0,
                                            "Linear (Rec.709)", region=region, calibration=cal, mask=mask)
    direct = ra.analytic_master(_sdr("sunset"), ra.AnalyticOptions(
        region=region, calibration=[ra.CalibrationPoint(128, 40.0)], limit=mask[0].numpy()))
    assert np.allclose(image[0].numpy(), direct.image, atol=1e-6)
    assert "region: band" in report and "calibration: 1 anchor" in report and "mask" in report
    assert not filled[0, :, 160:].any()
    (graded,) = RadianceRudraRegionEV().grade(image, 0.0, 1.0, 400, 2000, 1.0, 2000, 8000, 0.0, 0.05, 12, 0.0)
    want = ra.apply_region_ev(image[0].numpy().astype(np.float64) * 203, [ra.RegionBand(400, 2000, 1.0)], 1.0) / 203
    assert np.allclose(graded[0].numpy(), want, atol=1e-6)
    (same,) = RadianceRudraRegionEV().grade(image, 0.0, 1.0, 400, 2000, 0.0, 2000, 8000, 0.0, 0.05, 12, 0.0)
    assert torch.equal(same, image)


# ── phase 3: mattes, pixel classes, regions picture, EV brackets, master write ──
GOLD3 = np.load(FIX / "golden_phase3.npz")
P3 = INDEX["phase3"]


def _p3(case):
    meta = P3["cases"][case]
    o = dict(meta["options"])
    if "region" in o:
        o["region"] = ra.Region(**o["region"])
    r = ra.analytic_master(_sdr(meta["sdr"]), ra.AnalyticOptions(**o))
    gen = GOLD3["generated_dark"].astype(np.float32) / 255.0 if meta["generated"] else None
    return meta, r, gen


@pytest.mark.parametrize("case", sorted(P3["cases"]))
def test_phase3_mattes_and_classes_match_rudra_native(case):
    meta, r, gen = _p3(case)
    m = r.mattes
    g = lambda k: GOLD3[f"p3_{case}_{k}"]
    n = m.cls.size
    # A near-white pixel can carry a fill gain of ~1e-7 stops in RUDRA's float maths and none in
    # numpy's (no visible difference in the picture), which flips its matte; measured: 6 of 19 200
    # pixels in one case, none in the rest.
    lim = 0.001 * n
    assert (np.abs(m.highlight - g("highlight").astype(np.float32)) > 2e-3).sum() <= lim
    assert (np.abs(m.shadow - g("shadow").astype(np.float32)) > 2e-3).sum() <= lim
    assert (m.changed != g("changed")).sum() <= lim
    if f"p3_{case}_region" in GOLD3.files:
        assert np.abs(m.region - g("region").astype(np.float32)).max() < 2e-3
    else:
        assert m.region is None
    mm = m.with_generated(gen)
    assert (mm.cls != g("class")).sum() <= lim
    agree = mm.cls == g("class")
    assert np.abs(mm.confidence - g("confidence").astype(np.float32))[agree].max() < 0.01
    assert np.allclose(mm.share, meta["share"], atol=1e-3)
    assert np.abs(m.grey.astype(int) - g("grey")).max() <= 1
    for ev in ra.BRACKET_STOPS:
        d = np.abs(ra.exposure_bracket(r.image, ev).astype(int) - g(f"ev{ev:+.0f}"))
        # the picture's own parity (worst 0.06 stop, in the deep shadows) shows at -2 EV as a code or two
        assert d.max() <= 2 and (d > 0).mean() < 0.05, f"{case} {ev:+.0f} EV"


def test_brackets_of_rudras_own_picture_match_rudras_brackets():
    lin = np.exp2(GOLD2["lin_region_hs"].astype(np.float32) / 1024.0) - np.float32(1e-4)
    for ev in ra.BRACKET_STOPS:
        d = np.abs(ra.exposure_bracket(lin, ev).astype(int) - GOLD3[f"p3_region_hs_ev{ev:+.0f}"])
        assert d.max() <= 1 and (d > 0).mean() < 0.02, f"{ev:+.0f} EV"   # the golden is log2-quantised to 1/1024 stop


def test_classes_follow_the_rules():
    sdr = np.array([[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]], np.float32)
    base = np.full((1, 5, 3), 0.01, np.float32)
    rec = base.copy()
    rec[0, 1] *= 2 ** 0.05      # follows: under 0.1 stop
    rec[0, 2] *= 2.0            # reconstructed: one stop where the SDR has detail
    rec[0, 3] *= 2.0            # estimated: one stop where the SDR clipped
    rec[0, 4] *= 2.0
    cls, conf = ra.classify_pixels(sdr, rec, base)
    assert cls.tolist() == [[0, 1, 2, 3, 3]]
    assert conf[0, :2].tolist() == [1.0, 1.0]
    assert conf[0, 2] == pytest.approx(0.5, abs=1e-3) and conf[0, 3] == pytest.approx(0.25, abs=1e-3)
    gen = np.array([[0, 0, 1, 0, 0]], np.float32)
    cls, conf = ra.classify_pixels(sdr, rec, base, gen)
    assert cls[0, 2] == 4 and conf[0, 2] == 0.0


def test_generated_split_sums_to_generated():
    _, r, _ = _p3("default")
    h, w = r.mattes.cls.shape
    gen = np.zeros((h, w), np.float32)
    gen[:, : w // 2] = 0.7
    ch = r.mattes.channels(gen)
    assert np.array_equal(ch["rudra.generated.highlight"] + ch["rudra.generated.shadow"], ch["rudra.generated"])
    assert set(np.unique(ch["rudra.class"])) <= {0.0, 0.25, 0.5, 0.75, 1.0}
    pic = ra.regions_picture(r.mattes, gen)
    assert pic.shape == (h, w, 3) and pic.dtype == np.uint8
    assert np.allclose(pic[0, 0], np.round(255 * (0.15 * 0.0 + 0.85 * ra.PIXEL_CLASS_COLOUR[4])), atol=40)


@pytest.mark.skipif(_real_torch() is None, reason="needs real torch")
def test_master_write_writes_what_rudra_writes(tmp_path):
    pytest.importorskip("OpenEXR")
    pytest.importorskip("PIL")
    torch = _real_torch()
    from radiance.nodes.hdr.rudra_analytic import RadianceRudraMasterWrite, RadianceSDRToHDRAnalytic
    sdr = torch.from_numpy(_sdr("sunset")[None])
    image, _, _, _, mattes, _ = RadianceSDRToHDRAnalytic().convert(
        sdr, "BT.1886 (graded / broadcast video)", 1.0, True, 1.0, 0.0, "Linear ACEScg (AP1)")
    src = tmp_path / "plate.png"
    src.write_bytes(b"not really a png, but bytes to hash")
    gen = torch.zeros(1, 180, 320)
    gen[:, :20, :20] = 1
    path, side_text = RadianceRudraMasterWrite().write(
        image, "shot", "From the mattes", "32-bit Float", True, True, mattes=mattes, generated_mask=gen,
        source_path=str(src), output_path=str(tmp_path), frame_index=7)
    names = sorted(p.name for p in tmp_path.iterdir() if p.name != "plate.png")
    assert names == ["shot.0007.ev+0.png", "shot.0007.ev+2.png", "shot.0007.ev-2.png", "shot.0007.exr",
                     "shot.0007.json", "shot.0007.mattes.exr", "shot.0007.regions.png"]
    side = json.loads(side_text)
    import hashlib
    assert side["source_sha256"] == hashlib.sha256(src.read_bytes()).hexdigest()
    assert side["container"] == "ACEScg (AP1)" and side["settings"]["source_curve"] == "bt1886"
    assert side["brackets"]["preview_vs_export"]["max_code_delta"] <= 1, "a float master reads back as written"
    assert side["pixel_classes"]["generated_pct"] == pytest.approx(100 * 400 / (180 * 320), abs=1e-3)
    assert "rudra.generated.highlight" in side["mattes"]["channels"]
    import OpenEXR
    with OpenEXR.File(str(tmp_path / "shot.0007.exr")) as f:
        hdr = f.header()
        assert hdr["rudra:sourceSha256"] == side["source_sha256"] and hdr["rudra:mattes"] == "shot.0007.mattes.exr"
        assert abs(hdr["chromaticities"][0] - 0.713) < 1e-4
    with OpenEXR.File(str(tmp_path / "shot.0007.mattes.exr"), separate_channels=True) as f:
        ch = f.channels()
        assert {"rudra.class", "rudra.confidence", "rudra.highlight", "rudra.shadow", "rudra.changed",
                "rudra.generated"} <= set(ch)
    with pytest.raises(ValueError, match="scene-linear"):
        RadianceRudraMasterWrite().write(image, "pq", "From the mattes", "16-bit Half Float", True, False,
                                         mattes={"frames": mattes["frames"], "encoding": "PQ (HDR10)"},
                                         output_path=str(tmp_path))
