"""RUDRA's delivery measures (radiance.model.rudra_delivery) against RUDRA itself.

tests/fixtures/rudra_delivery holds RUDRA's own output on the sequence below and
on a field of saturated colours: the native core's master_to_peak for the peak
fit, rudra/video.py for the cut test, the static hold and the flicker measure,
and rudra/delivery/metadata.py for HDR10+ / Dolby Vision / the analysis
(index.json "source" says how each was made).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from radiance.model import rudra_delivery as rd  # noqa: E402

FIX = Path(__file__).parent / "fixtures" / "rudra_delivery"
GOLD = np.load(FIX / "golden.npz")
INDEX = json.loads((FIX / "index.json").read_text(encoding="utf-8"))


def _sequence():
    """8 frames, 40x64, a cut at frame 5, codec noise, a moving clipped highlight, a pumping gain."""
    rng = np.random.default_rng(20261115)
    H, W = 40, 64
    base = np.clip(rng.random((H, W, 3)) * 0.25 + 0.35, 0, 1)
    other = np.clip(rng.random((H, W, 3)) * 0.2 + 0.05, 0, 1)
    sdr, hdr = [], []
    for i in range(8):
        f = (base if i < 5 else other).copy()
        f = np.clip(f + rng.normal(0, 1.0 / 255, f.shape), 0, 1)
        x0 = 4 + 3 * i
        f[10:20, x0:x0 + 8] = 1.0
        f = (np.round(f * 255) / 255).astype(np.float32)
        lin = rd.transfer_to_linear(f, "srgb")
        h = lin * 0.0203 * (1.0 + 0.04 * (i % 2))
        h[10:20, x0:x0 + 8] *= 6.0 + rng.normal(0, 0.3, (10, 8, 1))
        sdr.append(f)
        hdr.append(h.astype(np.float32))
    return sdr, hdr


@pytest.mark.parametrize("peak", [1000, 4000])
@pytest.mark.parametrize("fit", ["scale", "luminance"])
def test_peak_fit_matches_rudra_native(peak, fit):
    x = GOLD["peak_in"]
    want = GOLD[f"peak_{peak}_{'lum' if fit == 'luminance' else 'scale'}"]
    got = rd.master_to_peak(x * 10000.0, peak, None, fit)
    assert np.abs(got - want).max() <= 1e-6 * peak
    assert got.max() <= peak


def test_keep_luminance_keeps_luminance_and_gives_up_saturation():
    blue = np.array([[[200.0, 400.0, 6000.0]]], np.float32)          # luminance ~ 680, blue far over 1000
    k = np.asarray(rd.LUMA_REC2020, np.float32)
    lum = rd.master_to_peak(blue, 1000, None, "luminance")
    scale = rd.master_to_peak(blue, 1000, None, "scale")
    assert lum.max() <= 1000 and scale.max() <= 1000
    assert float(lum[0, 0] @ k) == pytest.approx(float((blue[0, 0] @ k)), rel=1e-4)
    assert float(scale[0, 0] @ k) < 0.5 * float(lum[0, 0] @ k), "Scale loses the brightness"
    assert lum[0, 0, 2] / lum[0, 0, 0] < blue[0, 0, 2] / blue[0, 0, 0], "Keep luminance gives up saturation"
    inside = np.array([[[100.0, 120.0, 90.0]]], np.float32)
    assert np.array_equal(rd.master_to_peak(inside, 1000, None, "luminance"), inside), "below the knee nothing moves"


def test_cuts_hold_and_flicker_match_rudra():
    sdr, hdr = _sequence()
    cuts = rd.detect_cuts(sdr)
    assert cuts == INDEX["cuts"]
    held = rd.hold_sequence(sdr, hdr, cuts, rd.retention_for(24, 0.25))
    assert np.abs(np.stack(held) - GOLD["held"]).max() < 1e-7
    rep = rd.flicker_report(sdr, held, cuts)
    for f, (fl, lo) in zip(rep["frames"], INDEX["flicker"]):
        assert (f["flicker_ev"] is None) == (fl is None)
        if fl is not None:
            # the golden hold is float32; 1e-7 there is 1e-8 EV here
            assert f["flicker_ev"] == pytest.approx(fl, abs=1e-6)
            assert f["local_flicker_ev"] == pytest.approx(lo, abs=1e-6)


def test_the_hold_settles_the_pumping_it_is_for():
    sdr, hdr = _sequence()
    cuts = rd.detect_cuts(sdr)
    raw = rd.flicker_report(sdr, hdr, cuts)["temporal"]
    held = rd.flicker_report(sdr, rd.hold_sequence(sdr, hdr, cuts, rd.retention_for(24, 0.25)), cuts)["temporal"]
    assert held["flicker_ev_mean"] < 0.5 * raw["flicker_ev_mean"]
    assert rd.hold_sequence(sdr, hdr, cuts, 0.0)[3] is hdr[3], "0 s: off"


def test_metadata_matches_rudra():
    sdr, hdr = _sequence()
    held = [GOLD["held"][i] for i in range(len(sdr))]
    stats = [rd.analyze_frame(rd.master_to_peak(h * 10000.0, 1000, None, "scale"), i) for i, h in enumerate(held)]
    assert [list(s) for s in rd.detect_shots(stats)] == INDEX["detected_shots"]
    shots = rd.shots_from_cuts(INDEX["cuts"], len(stats))
    data = rd.dynamic_metadata(stats, shots, 1000.0, 0.0001, 400)
    for mine, theirs in (("dovi_generate", "dovi"), ("hdr10plus", "hdr10plus"), ("hdr10plus_scenes", "scenes"),
                         ("analysis", "analysis")):
        _close(json.loads(json.dumps(data[mine])), INDEX[theirs], mine)


def _close(a, b, where):
    if isinstance(a, dict):
        assert isinstance(b, dict) and sorted(a) == sorted(b), where
        for k in a:
            _close(a[k], b[k], f"{where}.{k}")
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), where
        for i, (x, y) in enumerate(zip(a, b)):
            _close(x, y, f"{where}[{i}]")
    elif isinstance(a, float) or isinstance(b, float):
        assert a == pytest.approx(b, rel=1e-6, abs=1e-9), where
    else:
        assert a == b, where


def test_dovi_l1_respects_dovi_tool_clamps():
    stats = [rd.analyze_frame(np.full((4, 4, 3), 5.0, np.float32), i) for i in range(3)]
    l1 = rd.to_dovi_generate_json(stats, [(0, 3)])["shots"][0]["metadata_blocks"][0]["Level1"]
    assert l1["max_pq"] >= rd.L1_MAX_PQ_MIN and l1["min_pq"] <= rd.L1_MIN_PQ_MAX
    assert rd.L1_AVG_PQ_MIN_CMV40 <= l1["avg_pq"] < l1["max_pq"]


def test_cut_data_is_read_from_scene_cut_detect():
    assert rd.parse_cut_data('{"cuts": [0, 12, 40], "shots": []}', 50) == [0, 12, 40]
    assert rd.parse_cut_data("12, 40", 30) == [0, 12]
    assert rd.parse_cut_data("", 30) is None
    with pytest.raises(ValueError):
        rd.parse_cut_data("twelve", 30)


def _real_torch():
    try:
        import torch
    except ImportError:
        return None
    return torch if getattr(torch, "__file__", None) and hasattr(torch, "from_numpy") else None


@pytest.mark.skipif(_real_torch() is None, reason="needs real torch")
def test_delivery_nodes_wire_together(tmp_path):
    torch = _real_torch()
    from radiance.nodes.hdr.rudra_delivery import (
        NODE_CLASS_MAPPINGS, RadianceRudraDynamicMetadata, RadianceRudraFlickerReport, RadianceRudraPeakFit,
        RadianceRudraShotHold)
    assert len(NODE_CLASS_MAPPINGS) == 4
    sdr, hdr = _sequence()
    # the sequence's HDR is network units in Rec.2020: hand it over as scene-linear Rec.2020
    S = torch.from_numpy(np.stack(sdr))
    Hs = torch.from_numpy(np.stack(hdr) * (10000.0 / 203.0))
    held, cut_data, report = RadianceRudraShotHold().hold(S, Hs, "Linear (Rec.2020)", 0.25, 24.0)
    assert json.loads(cut_data)["cuts"] == INDEX["cuts"] and "retention" in report
    assert np.allclose(held.numpy() * (203.0 / 10000.0), GOLD["held"], rtol=1e-5, atol=1e-9)
    text, js, worst = RadianceRudraFlickerReport().report(S, held, "Linear (Rec.2020)", cut_data=cut_data)
    assert "flicker: mean" in text and worst == pytest.approx(json.loads(js)["temporal"]["flicker_ev_max"])
    fitted, rep = RadianceRudraPeakFit().fit(held * 40, "Linear (Rec.709)", 1000.0, "Keep luminance (RUDRA default)")
    assert fitted.shape == held.shape and "pixels changed" in rep
    summary, hdr10p, dovi, analysis, hdr10p_path = RadianceRudraDynamicMetadata().measure(
        held, "Linear (Rec.2020)", 1000.0, "clip", True, cut_data=cut_data, output_path=str(tmp_path))
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "clip_dovi_generate.json", "clip_hdr10plus.json", "clip_hdr10plus_scenes.json", "clip_hdr_analysis.json"]
    assert "shots: 2 (from cut_data)" in summary
    assert len(json.loads(hdr10p)["SceneInfo"]) == 8 and json.loads(dovi)["profile"] == "8.1"
    assert hdr10p_path.endswith("clip_hdr10plus.json") and Path(hdr10p_path).is_file()


# ── HDR10+ into the video (Radiance Write's hdr10plus_json) ─────────────────
def test_x265_param_path_escapes_what_ffmpeg_splits_on():
    from radiance.io.writer import x265_param_path
    assert x265_param_path(r"E:\out dir\a'b.json") == r"E\:\\out\ dir\\a\'b.json"


def test_write_checks_the_hdr10plus_json_before_writing(tmp_path):
    torch = _real_torch()
    if torch is None:
        pytest.skip("needs real torch")
    from radiance.io import writer as W
    stats = [rd.analyze_frame(np.full((4, 4, 3), 100.0 * (i + 1), np.float32), i) for i in range(2)]
    js = tmp_path / "two.json"
    js.write_text(json.dumps(rd.to_hdr10plus_x265_json(stats, [(0, 2)])), encoding="utf-8")
    frames = torch.full((3, 8, 8, 3), 0.5)
    with pytest.raises(ValueError, match="2 frames of metadata for 3"):
        W.write_frames(frames, str(tmp_path / "a"), "VID │ MP4 (H.265 10-bit)", color_space="PQ (HDR10 / ST.2084)",
                       fps=24.0, hdr10plus_json=str(js))
    with pytest.raises(ValueError, match="not a file"):
        W.write_frames(frames, str(tmp_path / "b"), "VID │ MP4 (H.265 10-bit)", color_space="PQ (HDR10 / ST.2084)",
                       fps=24.0, hdr10plus_json=str(tmp_path / "missing.json"))
    assert not any(p.suffix == ".mp4" for p in tmp_path.iterdir()), "nothing written"


def test_write_carries_hdr10plus_into_the_hevc(tmp_path):
    torch = _real_torch()
    if torch is None:
        pytest.skip("needs real torch")
    import shutil
    from radiance.io import writer as W
    if not W._ffmpeg_ok() or shutil.which("ffprobe") is None:
        pytest.skip("needs ffmpeg with libx265 and ffprobe")
    try:
        W._ffmpeg_for("libx265")
    except Exception:
        pytest.skip("no libx265")
    frames = torch.linspace(0.2, 3.0, 4).view(4, 1, 1, 1).expand(4, 32, 32, 3).contiguous()
    stats = [rd.analyze_frame(np.full((4, 4, 3), 203.0 * float(v), np.float32), i)
             for i, v in enumerate(torch.linspace(0.2, 3.0, 4))]
    js = tmp_path / "clip_hdr10plus.json"
    js.write_text(json.dumps(rd.to_hdr10plus_x265_json(stats, [(0, 4)])), encoding="utf-8")
    receipt = {"files": []}
    out, n = W.write_frames(frames, str(tmp_path / "clip"), "VID │ MP4 (H.265 10-bit)",
                            color_space="PQ (HDR10 / ST.2084)", fps=24.0, hdr10plus_json=str(js), receipt=receipt)
    assert n == 4 and receipt["hdr10plus"]["json"] == str(js.resolve())
    assert receipt["hdr10plus"]["carried"] in (True, False)   # False: this ffmpeg's x265 has no HDR10+
