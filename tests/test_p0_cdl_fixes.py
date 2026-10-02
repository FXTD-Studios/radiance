"""v3.5.3 P0 regression tests: ASC CDL export (FIX-006).

Files are read back by OpenColorIO, an independent reader, and the colour it
applies is compared with the ASC CDL formula and with Radiance's CDL node.
Nuke and Resolve acceptance is manual (see KNOWN_ISSUES).
"""
import json
import os

import numpy as np
import pytest

ocio = pytest.importorskip("PyOpenColorIO")

from radiance.io.formats import build_cdl_xml, write_cdl_file, ASC_CDL_NAMESPACE
from radiance.color.grading import grading_to_cdl

SLOPE = [1.2, 0.9, 1.05]
OFFSET = [0.02, -0.01, 0.0]
POWER = [0.9, 1.1, 1.0]
SAT = 0.8


def _asc(rgb, slope=SLOPE, offset=OFFSET, power=POWER, sat=SAT):
    x = np.clip(rgb * np.array(slope) + np.array(offset), 0, None) ** np.array(power)
    luma = x @ np.array([0.2126, 0.7152, 0.0722])
    return luma[..., None] + sat * (x - luma[..., None])


def _ocio_apply(path, rgb, ccid=""):
    cfg = ocio.Config.CreateRaw()
    t = ocio.FileTransform(src=str(path), cccId=ccid, interpolation=ocio.INTERP_LINEAR)
    cpu = cfg.getProcessor(t).getDefaultCPUProcessor()
    out = rgb.astype(np.float32).copy()
    cpu.applyRGB(out)
    return out


RGB = np.array([[0.1, 0.2, 0.3], [0.5, 0.4, 0.3], [0.8, 0.7, 0.9], [0.18, 0.18, 0.18]], dtype=np.float32)


@pytest.mark.parametrize("ext", ["cdl", "cc", "ccc"])
def test_ocio_reads_every_container_with_the_written_values(tmp_path, ext):
    path = write_cdl_file(str(tmp_path / f"shot_010.{ext}"), SLOPE, OFFSET, POWER, SAT, cc_id="shot_010")
    t = ocio.CDLTransform.CreateFromFile(path, "shot_010" if ext != "cc" else "")
    np.testing.assert_allclose(t.getSlope(), SLOPE, atol=1e-6)
    np.testing.assert_allclose(t.getOffset(), OFFSET, atol=1e-6)
    np.testing.assert_allclose(t.getPower(), POWER, atol=1e-6)
    assert t.getSat() == pytest.approx(SAT, abs=1e-6)
    out = _ocio_apply(path, RGB, "shot_010" if ext == "ccc" else "")
    np.testing.assert_allclose(out, _asc(RGB), atol=2e-5)


def test_document_structure_and_namespace():
    import xml.etree.ElementTree as ET
    root = ET.fromstring(build_cdl_xml(SLOPE, OFFSET, POWER, SAT, cc_id="◎ My Grade"))
    ns = {"a": ASC_CDL_NAMESPACE}
    assert root.tag == f"{{{ASC_CDL_NAMESPACE}}}ColorDecisionList"
    cc = root.find("a:ColorDecision/a:ColorCorrection", ns)
    assert cc is not None and cc.get("id") == "My_Grade"
    assert cc.find("a:SOPNode/a:Slope", ns) is not None
    assert cc.find("a:SatNode/a:Saturation", ns) is not None


@pytest.mark.parametrize("kw", [dict(power=[0, 1, 1]), dict(slope=[-1, 1, 1]), dict(saturation=float("nan")),
                                dict(offset=[0, 0])])
def test_invalid_values_are_rejected(kw):
    args = dict(slope=SLOPE, offset=OFFSET, power=POWER, saturation=SAT)
    args.update(kw)
    with pytest.raises(ValueError):
        build_cdl_xml(**args)


torch = pytest.importorskip("torch")


def test_cdl_export_node_round_trips_through_ocio_and_matches_cdl_node(tmp_path):
    if not hasattr(torch, "einsum"):
        pytest.skip("real torch required")
    from radiance.nodes.color.cdl import RadianceCDLExport, RadianceCDLTransform, RadianceCDLImport
    (path,) = RadianceCDLExport().save(str(tmp_path / "grade.cdl"), *SLOPE, *OFFSET, *POWER, SAT)
    assert os.path.isfile(path)
    ocio_out = _ocio_apply(path, RGB)
    node_out, _ = RadianceCDLTransform().apply(torch.from_numpy(RGB).view(1, 1, 4, 3), *SLOPE, *OFFSET,
                                               *POWER, SAT)
    np.testing.assert_allclose(node_out.numpy().reshape(4, 3), ocio_out, atol=2e-5)
    data = json.loads(RadianceCDLImport().load(path)[0])
    np.testing.assert_allclose(data["slope"], SLOPE, atol=1e-6)


def test_cdl_import_fails_on_missing_or_bad_file(tmp_path):
    from radiance.nodes.color.cdl import RadianceCDLImport
    with pytest.raises(FileNotFoundError):
        RadianceCDLImport().load(str(tmp_path / "nope.cdl"))
    bad = tmp_path / "bad.cdl"
    bad.write_text("<not xml")
    with pytest.raises(ValueError):
        RadianceCDLImport().load(str(bad))


def test_viewer_grade_to_cdl_matches_the_viewer_math():
    """grading_to_cdl must reproduce apply_grading for the SOP subset."""
    from radiance.color.grading import apply_grading
    grading = {"exposure": 0.5, "gain": [1.1, 0.95, 1.0], "offset": [0.01, 0.0, -0.01],
               "gamma": [1.2, 0.9, 1.0], "saturation": 1.0}
    cdl = grading_to_cdl(grading)
    assert cdl["exact"]
    img = RGB.reshape(1, 4, 3)
    viewer = apply_grading(img, exposure=0.5, gain_rgb=grading["gain"], offset_rgb=grading["offset"],
                           gamma_rgb=grading["gamma"])
    sop = _asc(img, cdl["slope"], cdl["offset"], cdl["power"], 1.0)
    np.testing.assert_allclose(viewer, sop, atol=1e-5)


def test_viewer_grade_reports_what_cdl_cannot_carry():
    cdl = grading_to_cdl({"lift": [0.05, 0, 0], "contrast": 1.2})
    assert not cdl["exact"]
    assert set(cdl["not_represented"]) >= {"lift", "contrast"}
