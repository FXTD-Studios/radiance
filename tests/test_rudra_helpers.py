"""RUDRA's still colour / orientation reading and AI fill prompts (radiance.model.rudra_still,
radiance.model.rudra_prompts) against RUDRA itself.

The expected readings were produced by RUDRA's native read_still_colour
(native/media/src/still_metadata.cpp, commit 74f03b6) on the files _files() builds.
"""
from __future__ import annotations

import io
import struct
import zlib

import numpy as np
import pytest

from radiance.model import rudra_prompts as rp  # noqa: E402
from radiance.model import rudra_still as rs  # noqa: E402

P3 = (0.5151, 0.2412, -0.0011, 0.2920, 0.6922, 0.0419, 0.1571, 0.0666, 0.7841)
ADOBE = (0.6097, 0.3111, 0.0195, 0.2053, 0.6257, 0.0609, 0.1492, 0.0632, 0.7446)
ODD = (0.5, 0.3, 0.0, 0.3, 0.6, 0.1, 0.15, 0.1, 0.7)


def _icc(colorants, desc):
    tags = [(sig, b"XYZ \0\0\0\0" + b"".join(struct.pack(">i", round(v * 65536)) for v in xyz))
            for sig, xyz in zip((b"rXYZ", b"gXYZ", b"bXYZ"), (colorants[0:3], colorants[3:6], colorants[6:9]))]
    d = desc.encode() + b"\0"
    tags.append((b"desc", b"desc\0\0\0\0" + struct.pack(">I", len(d)) + d + b"\0" * 79))
    off, table, data = 128 + 4 + 12 * len(tags), b"", b""
    for sig, body in tags:
        body += b"\0" * (-len(body) % 4)
        table += sig + struct.pack(">II", off + len(data), len(body))
        data += body
    hdr = bytearray(128)
    struct.pack_into(">I", hdr, 0, off + len(data))
    hdr[12:16], hdr[16:20], hdr[20:24], hdr[36:40] = b"mntr", b"RGB ", b"XYZ ", b"acsp"
    return bytes(hdr) + struct.pack(">I", len(tags)) + table + data


def _chunk(t, d):
    return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)


def _exif(orient, le=True):
    e16, e32 = ("<H", "<I") if le else (">H", ">I")
    return ((b"II*\0" if le else b"MM\0*") + struct.pack(e32, 8) + struct.pack(e16, 1) + struct.pack(e16, 0x0112)
            + struct.pack(e16, 3) + struct.pack(e32, 1) + struct.pack(e16, orient) + b"\0\0" + struct.pack(e32, 0))


def _png(extra, w=3, h=2):
    raw = b"".join(b"\0" + bytes(range(y * w * 3, y * w * 3 + w * 3)) for y in range(h))
    return (b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + b"".join(_chunk(t, d) for t, d in extra) + _chunk(b"IDAT", zlib.compress(raw)) + _chunk(b"IEND", b""))


def _chrm(c):
    return struct.pack(">8I", *[round(x * 100000) for x in list(c[6:8]) + list(c[0:6])])


def _files():
    out = {
        "png_none": _png([]),
        "png_srgb": _png([(b"sRGB", b"\0")]),
        "png_cicp9": _png([(b"cICP", bytes([9, 16, 0, 1]))]),
        "png_cicp12": _png([(b"cICP", bytes([12, 13, 0, 1]))]),
        "png_cicp5": _png([(b"cICP", bytes([5, 1, 0, 1]))]),
        "png_icc_p3": _png([(b"iCCP", b"Display P3\0\0" + zlib.compress(_icc(P3, "Display P3")))]),
        "png_icc_odd": _png([(b"iCCP", b"odd\0\0" + zlib.compress(_icc(ODD, "Odd RGB")))]),
        "png_chrm_adobe": _png([(b"cHRM", _chrm((0.64, 0.33, 0.21, 0.71, 0.15, 0.06, 0.3127, 0.3290)))]),
        "png_exif6": _png([(b"eXIf", _exif(6))]),
    }
    try:
        from PIL import Image
    except ImportError:
        return out
    img = Image.fromarray((np.arange(4 * 6 * 3) % 256).astype(np.uint8).reshape(4, 6, 3))
    for name, fmt, kw in (("jpeg_adobe_8", "JPEG", dict(icc_profile=_icc(ADOBE, "Adobe RGB (1998)"),
                                                          exif=b"Exif\0\0" + _exif(8, le=False))),
                          ("tiff_p3_3", "TIFF", dict(icc_profile=_icc(P3, "P3"), tiffinfo={274: 3})),
                          ("webp_p3_5", "WEBP", dict(icc_profile=_icc(P3, "P3 webp"), exif=_exif(5), lossless=True))):
        b = io.BytesIO()
        img.save(b, fmt, **kw)
        out[name] = b.getvalue()
    return out


# (primaries, tagged, recognised, tag, profile, orientation): RUDRA native's reading of each file
RUDRA_NATIVE = {
    "jpeg_adobe_8": ("adobe_rgb", True, True, "icc", "Adobe RGB (1998)", 8),
    "png_chrm_adobe": ("adobe_rgb", True, True, "chrm", "cHRM chromaticities", 1),
    "png_cicp12": ("p3d65", True, True, "cicp", "cICP colour primaries 12", 1),
    "png_cicp5": ("rec709", True, False, "cicp", "cICP colour primaries 5", 1),
    "png_cicp9": ("rec2020", True, True, "cicp", "cICP colour primaries 9", 1),
    "png_exif6": ("rec709", False, True, "", "", 6),
    "png_icc_odd": ("rec709", True, False, "icc", "Odd RGB", 1),
    "png_icc_p3": ("p3d65", True, True, "icc", "Display P3", 1),
    "png_none": ("rec709", False, True, "", "", 1),
    "png_srgb": ("rec709", True, True, "srgb", "sRGB", 1),
    "tiff_p3_3": ("p3d65", True, True, "icc", "P3", 3),
    "webp_p3_5": ("p3d65", True, True, "icc", "P3 webp", 5),
}


@pytest.mark.parametrize("name", sorted(RUDRA_NATIVE))
def test_still_colour_matches_rudra_native(name):
    files = _files()
    if name not in files:
        pytest.skip("needs Pillow to build this file")
    c = rs.read_still_colour(files[name])
    assert (c.primaries, c.tagged, c.recognised, c.tag, c.profile, c.orientation) == RUDRA_NATIVE[name]


def _native_orientation(a, o):
    """native apply_orientation, pixel by pixel: the stored pixel shown at (x, y)."""
    h, w = a.shape[:2]
    sw, sh = (h, w) if o >= 5 else (w, h)
    out = np.zeros((sh, sw) + a.shape[2:], a.dtype)
    for y in range(sh):
        for x in range(sw):
            sx, sy = {1: (x, y), 2: (w - 1 - x, y), 3: (w - 1 - x, h - 1 - y), 4: (x, h - 1 - y),
                      5: (y, x), 6: (y, h - 1 - x), 7: (w - 1 - y, h - 1 - x), 8: (w - 1 - y, x)}[o]
            out[y, x] = a[sy, sx]
    return out


@pytest.mark.parametrize("o", range(1, 9))
def test_orientation_matches_rudra(o):
    a = np.arange(3 * 5 * 2).reshape(3, 5, 2)
    assert np.array_equal(rs.apply_orientation(a, o), _native_orientation(a, o))


def test_p3_to_rec709_matrix():
    m = rs.matrix_to_rec709("p3d65")
    assert np.allclose(m.sum(1), 1.0, atol=1e-6), "white stays white"
    assert np.allclose(m[0], [1.2249, -0.2249, 0.0], atol=1e-3)
    assert np.array_equal(rs.matrix_to_rec709("rec709"), np.eye(3))


def _frame(h=90, w=160):
    return np.full((h, w, 3), 90, np.uint8)


def test_prompt_suggestions_follow_rudras_rules():
    sky = _frame()
    sky[:, :] = (60, 90, 150)
    sky[:25] = 255                                   # large, at the top, blue around it
    s = rp.suggest([sky])
    assert s["suggestions"][0]["id"] == "sky_blue" and "at the top" in s["suggestions"][0]["reason"]
    sun = _frame()
    sun[:] = (200, 120, 60)
    yy, xx = np.mgrid[:90, :160]
    sun[(yy - 60) ** 2 + (xx - 100) ** 2 < 36] = 255   # small, round, warm around it
    assert rp.suggest([sun])["suggestions"][0]["id"] == "sun_disc"
    window = np.full((90, 160, 3), 20, np.uint8)
    window[40:70, 60:100] = 255                       # a rectangle in a dark surround
    assert rp.suggest([window])["suggestions"][0]["id"] == "window_daylight"
    glints = _frame()
    for x in range(10, 150, 20):
        glints[45, x] = 255
    assert rp.suggest([glints])["suggestions"][0]["id"] == "specular_glints"
    dark = _frame()
    dark[60:] = 0
    assert any(x["id"] == "shadow_detail" for x in rp.suggest([dark], tau_low=4)["suggestions"])
    assert rp.suggest([_frame()])["suggestions"] == []
    assert rp.cache_key("  clear blue sky ") == rp.cache_key("clear blue sky")


def _real_torch():
    try:
        import torch
    except ImportError:
        return None
    return torch if getattr(torch, "__file__", None) and hasattr(torch, "from_numpy") else None


@pytest.mark.skipif(_real_torch() is None, reason="needs real torch")
def test_load_still_and_the_analytic_source_wire_together(tmp_path):
    torch = _real_torch()
    pytest.importorskip("cv2")
    from radiance.nodes.hdr.rudra_analytic import RadianceSDRToHDRAnalytic
    from radiance.nodes.hdr.rudra_helpers import NODE_CLASS_MAPPINGS, RadianceRudraFillPrompt, RadianceRudraLoadStill
    assert len(NODE_CLASS_MAPPINGS) == 2
    # a 16-bit PNG, Display P3 by cICP, portrait by EXIF orientation 6
    import cv2
    im = np.zeros((4, 6, 3), np.uint16)
    im[..., 0] = 65535                             # red, BGR order in cv2: stored as blue
    im[0, 0] = (0, 0, 0)
    ok, enc = cv2.imencode(".png", im)
    data = bytes(enc)
    i = data.index(b"IDAT") - 4
    data = data[:i] + _chunk(b"cICP", bytes([12, 13, 0, 1])) + _chunk(b"eXIf", _exif(6)) + data[i:]
    p = tmp_path / "phone.png"
    p.write_bytes(data)
    image, alpha, source, note = RadianceRudraLoadStill().load("", path=str(p))
    assert image.shape == (1, 6, 4, 3), "orientation 6: portrait"
    assert source["primaries"] == "p3d65" and "Display P3" in note and "16-bit" in note
    assert float(image[0, 0, -1, 2]) == 0.0 and float(image[0, 1, 1, 2]) == 1.0
    assert torch.all(alpha == 1)
    node = RadianceSDRToHDRAnalytic()
    p3 = node.convert(image, "BT.1886 (graded / broadcast video)", 1.0, True, 1.0, 0.0, "Linear (Rec.709)",
                      source=source)
    plain = node.convert(image, "BT.1886 (graded / broadcast video)", 1.0, True, 1.0, 0.0, "Linear (Rec.709)")
    m = torch.from_numpy(rs.matrix_to_rec709("p3d65").astype(np.float32))
    assert torch.allclose(p3[0], plain[0] @ m.T, atol=1e-5), "mastered in P3, delivered in Rec.709"
    assert "source: Display P3" in p3[3]
    assert p3[4]["frames"][0].settings["source_space"] == "p3d65"
    sky = np.full((1, 90, 160, 3), 0.35, np.float32)
    sky[..., 2] = 0.6
    sky[:, :25] = 1.0
    hi, sh, rep = RadianceRudraFillPrompt().prompt(torch.from_numpy(sky), "Suggest from the frames", "None")
    assert hi == rp.BY_ID["sky_blue"].text and sh == "" and "Clear blue sky" in rep
    hi, sh, _ = RadianceRudraFillPrompt().prompt(torch.from_numpy(sky), "Flame (highlight)", "Shadow detail (shadow)",
                                                 extra="with sparks")
    assert hi == rp.BY_ID["flame"].text + ", with sparks" and sh == rp.BY_ID["shadow_detail"].text
