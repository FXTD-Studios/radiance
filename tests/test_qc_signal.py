"""QC on display and scene-linear images (4.0): an HDR master is not 'clipped'.

Until 4.0 the test setup replaced the analysis with mocks, so QC never ran on
a real image here, and it reported every ACEScg or PQ master as clipped and
out of gamut. These run the real analysis.
"""
from __future__ import annotations

import json

import torch

from radiance.nodes.color.qc import RadianceQC


def _run(image, **kw):
    out = RadianceQC().run(mode="Analyze", image=image, **kw)
    out = out.get("result", out) if isinstance(out, dict) else out
    return out[1], json.loads(out[2]), out[3]


def _hdr():
    g = torch.linspace(0.0, 1.0, 64)
    img = torch.stack([g * 6.0, g * 3.0, g * 0.5], -1).expand(32, 64, 3).clone()
    return img.unsqueeze(0).repeat(2, 1, 1, 1)


def test_a_clean_sdr_image_passes():
    text, report, status = _run(torch.full((1, 32, 64, 3), 0.5), enable_noise_check=False,
                                enable_artifacts_check=False)
    assert "PASS" in status and report["signal"] == "display"


def test_sdr_clipping_still_fails():
    img = torch.full((1, 32, 64, 3), 0.5)
    img[0, :4] = 1.2
    text, report, status = _run(img, signal="Display (0-1)", enable_noise_check=False,
                                enable_artifacts_check=False)
    assert "FAIL" in status and "CLIPPED" in text
    assert report["frames"][0]["checks"]["clipped_whites"]["status"] == "FAIL"


def test_an_hdr_master_passes_with_its_peak_in_stops():
    text, report, status = _run(_hdr(), enable_noise_check=False, enable_artifacts_check=False)
    assert "PASS" in status, text
    assert report["signal"] == "scene-linear"
    assert "CLIPPED" not in text and "GAMUT" not in text
    peak = report["frames"][0]["checks"]["peak"]
    assert abs(peak["peak"] - 6.0) < 1e-4
    assert abs(peak["stops_above_grey"] - 5.06) < 0.01
    assert report["frames"][0]["checks"]["clipped_whites"]["status"] == "SKIP"


def test_negative_values_in_scene_linear_warn_without_failing():
    img = _hdr()
    img[:, :2, :2] = -0.01
    text, report, status = _run(img, enable_noise_check=False, enable_artifacts_check=False)
    assert "PASS" in status and "NEGATIVE" in text
    assert report["frames"][0]["checks"]["crushed_blacks"]["status"] == "WARN"


def test_nan_still_fails_in_scene_linear():
    img = _hdr()
    img[0, 0, 0, 0] = float("nan")
    text, report, status = _run(img, enable_noise_check=False, enable_artifacts_check=False)
    assert "FAIL" in status and "NON-FINITE" in text


def test_the_hdr_overlay_does_not_paint_highlights_red():
    out = RadianceQC().run(mode="Analyze", image=_hdr(), overlay_opacity=1.0, enable_noise_check=False,
                           enable_artifacts_check=False)
    out = out.get("result", out) if isinstance(out, dict) else out
    assert torch.allclose(out[0], _hdr())
