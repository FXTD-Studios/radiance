"""v3.5.3 P0 regression tests: delivery receipts, DPX and Viewer colour
interpretation (FIX-015, 016, 017, 018)."""
import json
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not hasattr(torch, "einsum"):
    pytest.skip("real torch required", allow_module_level=True)
oiio = pytest.importorskip("OpenImageIO")


# ── FIX-015 / FIX-016: multi-part EXR receipts ──────────────────────────────

from radiance.hdr import io as hdr_io


def _parts():
    return {"beauty": np.random.rand(4, 4, 3).astype(np.float32),
            "depth": np.random.rand(4, 4, 1).astype(np.float32)}


def test_multipart_receipt_lists_the_real_file(tmp_path):
    r = hdr_io.write_exr_multipart_report(str(tmp_path / "f.exr"), _parts())
    assert r["mode"] == "multipart" and r["complete"]
    assert r["files"] == [str(tmp_path / "f.exr")] and os.path.isfile(r["files"][0])
    assert {k: v["status"] for k, v in r["parts"].items()} == {"beauty": "written", "depth": "written"}


def _break_multipart(monkeypatch):
    import OpenEXR

    class _Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("simulated multi-part failure")
    monkeypatch.setattr(OpenEXR, "File", _Boom)


def test_strict_multipart_failure_writes_nothing_and_says_so(tmp_path, monkeypatch):
    _break_multipart(monkeypatch)
    r = hdr_io.write_exr_multipart_report(str(tmp_path / "f.exr"), _parts())
    assert r["mode"] == "failed" and not r["complete"] and r["files"] == []
    assert hdr_io.write_exr_multipart(str(tmp_path / "g.exr"), _parts()) is False
    assert list(tmp_path.iterdir()) == []


def test_fallback_receipt_lists_per_part_files(tmp_path, monkeypatch):
    _break_multipart(monkeypatch)
    monkeypatch.setattr(hdr_io, "write_exr_robust",
                        lambda path, arr, *a, **k: (open(path, "wb").write(b"x") and True))
    r = hdr_io.write_exr_multipart_report(str(tmp_path / "f.exr"), _parts(), allow_fallback=True)
    assert r["mode"] == "per_part" and r["complete"]
    assert sorted(os.path.basename(f) for f in r["files"]) == ["f.beauty.exr", "f.depth.exr"]
    assert not (tmp_path / "f.exr").exists()


def test_fallback_with_a_failed_part_is_incomplete(tmp_path, monkeypatch):
    _break_multipart(monkeypatch)

    def _write(path, arr, *a, **k):
        if path.endswith(".depth.exr"):
            return False
        open(path, "wb").write(b"x")
        return True
    monkeypatch.setattr(hdr_io, "write_exr_robust", _write)
    r = hdr_io.write_exr_multipart_report(str(tmp_path / "f.exr"), _parts(), allow_fallback=True)
    assert not r["complete"]
    assert r["parts"]["depth"]["status"] == "failed"
    assert [os.path.basename(f) for f in r["files"]] == ["f.beauty.exr"]


def _node(tmp_path, monkeypatch, **kw):
    from radiance.nodes.io import write as W
    monkeypatch.setattr(W, "get_safe_output_dir", lambda base, p, allow_absolute=True: str(tmp_path))
    node = W.RadianceEXRMultiPart()
    beauty = torch.rand(2, 4, 4, 3)
    return node.write_multipart("shot", beauty, depth=torch.rand(2, 4, 4, 3), **kw)


def test_multipart_node_returns_a_manifest_of_real_files(tmp_path, monkeypatch):
    path, manifest = _node(tmp_path, monkeypatch)
    m = json.loads(manifest)
    assert m["complete"] and len(m["files"]) == 2
    assert all(os.path.isfile(f) for f in m["files"]) and path == m["files"][0]


def test_multipart_node_strict_fails_an_incomplete_write(tmp_path, monkeypatch):
    _break_multipart(monkeypatch)
    with pytest.raises(RuntimeError, match="Incomplete write"):
        _node(tmp_path, monkeypatch)


def test_multipart_node_non_strict_reports_the_fallback_files(tmp_path, monkeypatch):
    _break_multipart(monkeypatch)
    from radiance.nodes.io import write as W
    monkeypatch.setattr(hdr_io, "write_exr_robust",
                        lambda path, arr, *a, **k: (open(path, "wb").write(b"x") and True))
    path, manifest = _node(tmp_path, monkeypatch, strict=False)
    m = json.loads(manifest)
    assert m["mode"] == ["per_part"] and len(m["files"]) == 4
    assert path.endswith(".beauty.exr") and os.path.isfile(path)


def test_multipart_node_rejects_duplicate_part_names(tmp_path, monkeypatch):
    from radiance.nodes.io import write as W
    monkeypatch.setattr(W, "get_safe_output_dir", lambda base, p, allow_absolute=True: str(tmp_path))
    with pytest.raises(ValueError, match="Duplicate"):
        W.RadianceEXRMultiPart().write_multipart("s", torch.rand(1, 4, 4, 3), depth=torch.rand(1, 4, 4, 3),
                                                 custom_1=torch.rand(1, 4, 4, 3), custom_1_name="depth")


def test_write_receipt_lists_exactly_this_writes_files(tmp_path):
    from radiance.io import writer as W
    seq = tmp_path / "seq"
    seq.mkdir()
    (seq / "unrelated_from_another_write.exr").write_bytes(b"x")
    receipt = {"files": []}
    W.write_frames(image=torch.rand(3, 4, 4, 3), output_path=str(seq), format="SEQ │ EXR (16-bit half)",
                   start_frame=1001, receipt=receipt)
    names = [os.path.basename(e["path"]) for e in receipt["files"]]
    assert names == ["seq_1001.exr", "seq_1002.exr", "seq_1003.exr"]
    assert [e["frame"] for e in receipt["files"]] == [1001, 1002, 1003]
    assert all(e["writer"] == "OpenEXR" for e in receipt["files"])
    assert receipt["complete"] and receipt["written_frames"] == 3


# ── FIX-017: DPX metadata matches the pixels ────────────────────────────────

def _dpx(tmp_path, img, color_space, working="Linear Rec.709 (sRGB)"):
    from radiance.io import writer as W
    W.write_frames(image=img, output_path=str(tmp_path / "plate"), format="IMG │ DPX",
                   color_space=color_space, working_space=working)
    path = next(tmp_path.glob("*.dpx"))
    spec = oiio.ImageInput.open(str(path)).spec()
    return path, spec


def test_dpx_pass_through_does_not_claim_linear(tmp_path):
    _, spec = _dpx(tmp_path, torch.full((1, 4, 4, 3), 0.5), "Linear (pass-through)")
    assert spec.getattribute("dpx:Transfer") == "User defined"


def test_dpx_log_encoding_is_labelled_log(tmp_path):
    _, spec = _dpx(tmp_path, torch.full((1, 4, 4, 3), 0.5), "ARRI LogC3")
    assert spec.getattribute("dpx:Transfer") == "Logarithmic"


def test_dpx_rec709_is_labelled_709(tmp_path):
    path, spec = _dpx(tmp_path, torch.full((1, 4, 4, 3), 0.18), "Rec.709 (BT.1886)")
    assert spec.getattribute("dpx:Transfer") == "ITU-R 709-4"
    assert spec.getattribute("dpx:Colorimetric") == "ITU-R 709-4"


def test_dpx_refuses_to_clip_linear_hdr(tmp_path):
    img = torch.full((1, 4, 4, 3), 0.5)
    img[0, 0, 0] = 4.0
    with pytest.raises(ValueError, match="above 1.0"):
        _dpx(tmp_path, img, "Linear Rec.709 (sRGB)")


# ── FIX-018: Viewer source interpretation through export ────────────────────

def _srgb_encode(x):
    x = np.asarray(x, np.float64)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


@pytest.fixture
def deliver(tmp_path, monkeypatch):
    import asyncio
    import importlib
    import sys
    from radiance import cache
    from radiance.delivery import handler
    saved = {n: sys.modules[n] for n in list(sys.modules) if n == "aiohttp" or n.startswith("aiohttp.")}
    for n in saved:
        del sys.modules[n]
    try:
        web = importlib.import_module("aiohttp.web")
    except ImportError:
        sys.modules.update(saved)
        pytest.skip("aiohttp not installed")
    monkeypatch.setattr(handler, "web", web)
    root = tmp_path / "out"
    root.mkdir()
    monkeypatch.setattr(handler.folder_paths, "get_output_directory", lambda: str(root))

    class _Req:
        def __init__(self, p):
            self.p = p

        async def json(self):
            return self.p

    from tests.test_delivery_endpoints import IDENTITY_GRADE

    def run(frames, source, settings):
        key = f"p0-{len(list(root.rglob('*')))}-{id(frames)}"
        cache._viewer_cache_set(key, frames)
        if source is not None:
            cache._viewer_source_set(key, *source)
        payload = {"instance_id": key, "grading": dict(IDENTITY_GRADE),
                   "settings": dict({"soft_clip": False, "smart_versioning": True, "filename": "S"}, **settings)}
        resp = asyncio.run(handler.radiance_deliver_endpoint(_Req(payload)))
        assert resp.status == 200, resp.body
        return root
    yield run
    for n in [n for n in list(sys.modules) if n == "aiohttp" or n.startswith("aiohttp.")]:
        del sys.modules[n]
    sys.modules.update(saved)


SRGB_SRC = ("srgb", "sRGB Encoded Rec.709 (sRGB)")


def test_srgb_source_delivered_as_srgb_is_not_double_encoded(deliver):
    """The headline FIX-018 case: a ComfyUI IMAGE out as sRGB PNG is unchanged."""
    from PIL import Image
    root = deliver(torch.full((1, 4, 4, 3), 0.5), SRGB_SRC,
                   {"format": "Image Sequence — PNG (8-bit)", "colorSpace": "sRGB (Standard)"})
    px = Image.open(next(root.rglob("*.png"))).getpixel((0, 0))[0]
    assert px == 128, f"0.5 sRGB came back as {px}/255 (double-encoded would be {round(_srgb_encode(0.5)*255)})"


def test_srgb_source_delivered_as_linear_exr_is_decoded(deliver):
    root = deliver(torch.full((1, 4, 4, 3), 0.5), SRGB_SRC,
                   {"format": "Image Sequence — EXR (32-bit)", "colorSpace": "Linear (sRGB)"})
    v = oiio.ImageBuf(str(next(root.rglob("*.exr")))).getpixel(0, 0)[0]
    assert v == pytest.approx(0.21404, abs=1e-4)


def test_linear_source_delivered_as_srgb_is_encoded_once(deliver):
    from PIL import Image
    root = deliver(torch.full((1, 4, 4, 3), 0.18), ("linear", "Linear Rec.709 (sRGB)"),
                   {"format": "Image Sequence — PNG (8-bit)", "colorSpace": "sRGB (Standard)"})
    px = Image.open(next(root.rglob("*.png"))).getpixel((0, 0))[0]
    assert px == round(float(_srgb_encode(0.18)) * 255)


def test_acescg_source_exr_is_labelled_acescg(deliver):
    root = deliver(torch.full((1, 4, 4, 3), 0.18), ("linear", "ACEScg"),
                   {"format": "Image Sequence — EXR (32-bit)", "colorSpace": "Linear (sRGB)"})
    buf = oiio.ImageBuf(str(next(root.rglob("*.exr"))))
    assert buf.getpixel(0, 0)[0] == pytest.approx(0.18, abs=1e-6)
    assert "ACEScg" in str(buf.spec().getattribute("radiance:colorspace"))


def test_unrecorded_source_falls_back_to_the_viewer_auto_rule(deliver):
    """No record (older cache): 0-1 values are a display-encoded ComfyUI IMAGE."""
    from PIL import Image
    root = deliver(torch.full((1, 4, 4, 3), 0.5), None,
                   {"format": "Image Sequence — PNG (8-bit)", "colorSpace": "sRGB (Standard)"})
    assert Image.open(next(root.rglob("*.png"))).getpixel((0, 0))[0] == 128
