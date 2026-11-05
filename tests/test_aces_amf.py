"""ACES Metadata File (AMF v2.0) sidecars: RUDRA's documents, byte for byte,
valid against the Academy's schema, and written by Write beside ACES EXRs."""
from pathlib import Path

import numpy as np
import pytest

from radiance.io.aces_amf import AmfRequest, aces_amf_xml, amf_for_files, make_amf_uuid

DATA = Path(__file__).parent / "data" / "aces_amf"
RUDRA_VIEW = ("ACES 2.0 SDR 100 nits Rec.709 BT.1886: the view RUDRA's appearance "
              "constraint holds the master to")


def _fixed(acescg, **kw):
    return AmfRequest(description="RUDRA learned ACES 2.0 master", created_utc="2026-11-02T10:15:00Z",
                      amf_uuid="urn:uuid:11111111-2222-4333-8444-555555555555",
                      pipeline_uuid="urn:uuid:66666666-7777-4888-9999-aaaaaaaaaaaa",
                      acescg=acescg, output_description=RUDRA_VIEW, **kw)


CASES = {
    "acescg_sequence.amf": _fixed(True, clip_name="SH010", sequence=("SH010.######.exr", 1001, 1096)),
    "ap0_bare.amf": _fixed(False),
    "ap0_file_escaped.amf": _fixed(False, clip_name='still <A&B> "x"', file="still_A&B.exr"),
    "acescg_author.amf": _fixed(True, author_name="FXTD Studios", author_email="studio@example.com"),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_matches_rudra_byte_for_byte(name):
    assert aces_amf_xml(CASES[name]) == (DATA / "golden" / name).read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def schema():
    etree = pytest.importorskip("lxml.etree")
    return etree.XMLSchema(etree.parse(str(DATA / "acesMetadataFile.xsd")))


def _valid(schema, text):
    from lxml import etree
    return schema.validate(etree.fromstring(text.encode("utf-8")))


@pytest.mark.parametrize("name", sorted(CASES))
def test_rudra_goldens_are_schema_valid(schema, name):
    assert _valid(schema, (DATA / "golden" / name).read_text(encoding="utf-8")), schema.error_log


def test_radiance_documents_are_schema_valid(schema):
    for paths, label in ((["/o/shot_v0001_1001.exr", "/o/shot_v0001_1002.exr"], "ACEScg"),
                         (["/o/still.exr"], "ACES2065-1")):
        _, text = amf_for_files(paths, label)
        assert _valid(schema, text), schema.error_log


def test_a_broken_document_is_caught(schema):
    bad = aces_amf_xml(_fixed(True)).replace("<aces:majorVersion>2</aces:majorVersion>", "")
    assert not _valid(schema, bad)


def test_uuids_are_version_4_urns():
    import re
    pat = re.compile(r"urn:uuid:[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}")
    ids = {make_amf_uuid() for _ in range(50)}
    assert len(ids) == 50 and all(pat.fullmatch(i) for i in ids)


def test_a_sequence_gets_one_amf_named_after_it():
    path, text = amf_for_files([f"/o/shot_v0001_{f}.exr" for f in (1001, 1002, 1003)], "ACEScg")
    assert path == Path("/o/shot_v0001.amf")
    assert '<aces:sequence idx="#" min="1001" max="1003">shot_v0001_####.exr</aces:sequence>' in text
    assert "ACEScg_to_ACES" in text


def test_ap0_has_no_input_transform_and_other_spaces_get_none():
    _, text = amf_for_files(["/o/still.exr"], "ACES2065-1")
    assert "inputTransform" not in text and "<aces:file>still.exr</aces:file>" in text
    assert amf_for_files(["/o/still.exr"], "Linear Rec.709 (sRGB)") is None
    assert amf_for_files(["/o/still.png"], "ACEScg") is None


def _write(tmp_path, fmt, colour, **kw):
    torch = pytest.importorskip("torch")
    from radiance.io.writer import write_frames
    receipt = {}
    img = torch.from_numpy(np.full((2, 4, 4, 3), 0.5, np.float32))
    write_frames(img, str(tmp_path), fmt, filename="shot", color_space=colour,
                 working_space="ACEScg", receipt=receipt, **kw)
    return receipt


@pytest.mark.real_torch
def test_write_puts_an_amf_beside_an_acescg_exr_sequence(tmp_path):
    pytest.importorskip("OpenEXR", reason="EXR writing")
    receipt = _write(tmp_path, "SEQ │ EXR (16-bit half)", "ACEScg", aces_amf=True)
    amfs = list(tmp_path.rglob("*.amf"))
    assert len(amfs) == 1 and any(e["writer"] == "ACES AMF v2.0" for e in receipt["files"])
    assert "min=\"1001\" max=\"1002\"" in amfs[0].read_text()


@pytest.mark.real_torch
def test_write_warns_instead_of_writing_an_amf_for_a_display_space(tmp_path):
    receipt = _write(tmp_path, "SEQ │ PNG (16-bit)", "sRGB", aces_amf=True)
    assert not list(tmp_path.rglob("*.amf"))
    assert any("no AMF written" in w for w in receipt.get("warnings", []))


@pytest.mark.real_torch
def test_write_writes_no_amf_by_default(tmp_path):
    _write(tmp_path, "SEQ │ PNG (16-bit)", "sRGB")
    assert not list(tmp_path.rglob("*.amf"))
