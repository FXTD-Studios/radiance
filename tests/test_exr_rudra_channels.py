"""EXR MultiPart writes RUDRA's mask channels into the beauty part, named as
RUDRA names them (rudra.reconstructed, rudra.generated)."""
import importlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")
OpenEXR = pytest.importorskip("OpenEXR")


def test_writer_adds_extra_channels_to_a_part(tmp_path):
    io = importlib.import_module("radiance.hdr.io")
    beauty = np.random.default_rng(0).random((8, 10, 3)).astype(np.float32)
    rec = np.zeros((8, 10), np.float32)
    rec[2:5, 3:6] = 1.0
    f = tmp_path / "m.exr"
    r = io.write_exr_multipart_report(str(f), {"beauty": beauty}, "32-bit Float", "ZIP",
                                      extra_channels={"beauty": {"rudra.reconstructed": rec}})
    assert r["complete"] and r["mode"] == "multipart"
    part = OpenEXR.File(str(f), separate_channels=True).parts[0]
    ch = part.channels
    assert {"R", "G", "B", "rudra.reconstructed"} <= set(ch)
    assert np.array_equal(np.asarray(ch["rudra.reconstructed"].pixels), rec)


def test_writer_refuses_a_mismatched_channel(tmp_path):
    io = importlib.import_module("radiance.hdr.io")
    r = io.write_exr_multipart_report(str(tmp_path / "x.exr"), {"beauty": np.zeros((4, 4, 3), np.float32)},
                                      extra_channels={"beauty": {"rudra.generated": np.zeros((2, 2))}})
    assert not r["complete"]


def test_node_inputs_write_both_channels(tmp_path):
    w = importlib.import_module("radiance.nodes.io.write")
    opt = w.RadianceEXRMultiPart.INPUT_TYPES()["optional"]
    assert opt["rudra_reconstructed"][0] == "MASK" and opt["rudra_generated"][0] == "MASK"
    beauty = torch.rand(2, 6, 8, 3)
    rec = torch.zeros(2, 6, 8)
    rec[:, 1:3, 1:3] = 1
    gen = torch.zeros(1, 6, 8)
    path, manifest = w.RadianceEXRMultiPart().write_multipart(
        "rudra", beauty, "16-bit Half Float", "ZIP", output_path=str(tmp_path),
        rudra_reconstructed=rec, rudra_generated=gen)
    ch = OpenEXR.File(path, separate_channels=True).parts[0].channels
    assert "rudra.reconstructed" in ch and "rudra.generated" in ch
