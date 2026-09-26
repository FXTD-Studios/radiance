"""Read/Write release matrix (3.5.0).

Every Write format is written through the node and read back through the
Read node, and every colour space the two nodes share round-trips. The
expectation per format is what that container can physically hold, not
what happens to come out.
"""
import os
import glob

import numpy as np
import pytest
import torch

W, H = 64, 48


def _pattern():
    x = np.linspace(0.0, 1.0, W, dtype=np.float32)
    img = np.tile(x[None, :, None], (H, 1, 3)).copy()
    img[0:8, :, :] = np.array([0.18, 0.18, 0.18], np.float32)      # mid grey
    img[8:16, 0:16] = [16.0, 8.0, 4.0]                               # HDR highlight
    img[8:16, 16:32] = [-0.05, 0.02, -0.01]                          # negative (out of gamut)
    img[16:24, 0:16] = [1.0, 0.0, 0.0]
    img[16:24, 16:32] = [0.0, 1.0, 0.0]
    img[16:24, 32:48] = [0.0, 0.0, 1.0]
    alpha = np.tile(np.linspace(0, 1, W, dtype=np.float32)[None, :], (H, 1))
    return torch.from_numpy(img)[None], torch.from_numpy(alpha)[None]


@pytest.fixture(scope="module")
def nodes():
    import radiance.nodes.io.write as w
    return w


def _write(w, tmp, fmt, cs="Linear (pass-through)", mask=None, **kw):
    img, _ = _pattern()
    out = str(tmp / "out")
    res = w.RadianceWrite().write(img, out, fmt, color_space=cs, mask=mask, overwrite=True, **kw)
    files = sorted(f for f in glob.glob(str(tmp / "**" / "*"), recursive=True) if os.path.isfile(f))
    assert files, f"{fmt}: nothing written ({res})"
    return files[-1]


def _read(w, path, cs="Auto / Linear (pass-through)", **kw):
    img, mask, info = w.RadianceRead().read(path=path, media_type="Image", color_space=cs, **kw)
    return img[0].numpy(), (mask[0].numpy() if mask is not None else None), info


# format, expected max abs error on the [0,1] region, holds >1, holds <0, alpha
FORMATS = [
    ("IMG │ EXR (32-bit float)", 0.0, True, True, True),
    ("IMG │ EXR (16-bit half)", 1e-3, True, True, True),
    ("IMG │ TIFF (32-bit float)", 0.0, True, True, False),
    ("IMG │ TIFF (16-bit)", 1.0 / 65535 + 1e-6, False, False, False),
    ("IMG │ PNG (16-bit)", 1.0 / 65535 + 1e-6, False, False, True),
    ("IMG │ PNG (8-bit)", 1.0 / 255 + 1e-6, False, False, True),
    ("IMG │ DPX", 1.0 / 1023 + 1e-4, False, False, False),
    ("IMG │ Radiance HDR (.hdr)", 0.02, True, False, False),
    ("IMG │ JPEG", 0.08, False, False, False),
    ("IMG │ WEBP", 0.08, False, False, False),
]


@pytest.mark.parametrize("fmt,tol,hdr,neg,alpha", FORMATS, ids=[f[0].split("│ ")[1] for f in FORMATS])
def test_format_round_trip(nodes, tmp_path, fmt, tol, hdr, neg, alpha):
    src, a = _pattern()
    src = src[0].numpy()
    path = _write(nodes, tmp_path, fmt, mask=a if alpha else None)
    got, mask, _ = _read(nodes, path)
    assert got.shape == (H, W, 3), f"{fmt}: shape {got.shape}"
    ldr = (src >= 0) & (src <= 1)
    if "JPEG" in fmt or "WEBP" in fmt:
        err = float(np.abs(got[24:] - src[24:]).mean())   # smooth ramp only
    else:
        err = float(np.abs(got[ldr] - src[ldr]).max())
    assert err <= tol + 1e-7, f"{fmt}: [0,1] error {err} > {tol}"
    hi = got[8:16, 0:16]
    if hdr:
        assert np.allclose(hi, [16.0, 8.0, 4.0], rtol=0.02), f"{fmt}: highlight {hi[0,0]}"
    else:
        assert float(hi.max()) <= 1.0 + 1e-6
    ng = got[8:16, 16:32]
    if neg:
        assert np.allclose(ng, [-0.05, 0.02, -0.01], atol=1e-3), f"{fmt}: negatives {ng[0,0]}"
    if alpha:
        assert mask is not None, f"{fmt}: alpha lost"
        assert float(np.abs(mask - a[0].numpy()).max()) <= max(tol, 1e-6) + 1e-6, f"{fmt}: alpha error"


# Write colour space -> the Read colour space that must invert it.
PAIRS = [
    ("sRGB", "sRGB"),
    ("Rec.709 (BT.1886)", "Rec.709 (BT.1886)"),
    ("Rec.709", "Rec.709 (camera OETF)"),
    ("Rec.2020", "Rec.2020 (BT.2020 OETF)"),
    ("P3-D65 (Gamma 2.6)", "P3-D65 (Gamma 2.6)"),
    ("PQ (HDR10 / ST.2084)", "PQ (ST.2084)"),
    ("HLG (Hybrid Log-Gamma)", "HLG (BT.2100)"),
    ("Linear Rec.2020", "Linear Rec.2020"),
    ("Linear P3-D65", "Linear P3-D65"),
    ("ACEScg", "ACEScg"),
    ("ACES2065-1", "ACES2065-1"),
    ("ACEScct", "ACEScct"),
    ("ARRI LogC4", "ARRI LogC4"),
    ("ARRI LogC3", "ARRI LogC3"),
    ("Sony S-Log3", "Sony S-Log3"),
    ("Sony S-Log3 S-Gamut3", "Sony S-Log3 S-Gamut3"),
    ("Panasonic V-Log", "Panasonic V-Log"),
    ("Canon Log 3", "Canon Log 3"),
    ("RED Log3G10", "RED Log3G10"),
    ("DaVinci Intermediate", "DaVinci Intermediate"),
]


@pytest.mark.parametrize("wcs,rcs", PAIRS, ids=[p[0] for p in PAIRS])
def test_colour_space_round_trip(nodes, tmp_path, wcs, rcs):
    """Encode on Write, decode on Read: the working values come back, colour
    patches included (a transfer-only decode passes on greys and fails here)."""
    src, _ = _pattern()
    src = src[0].numpy()
    path = _write(nodes, tmp_path, "IMG │ EXR (32-bit float)", cs=wcs)
    got, _, info = _read(nodes, path, cs=rcs)
    region = (slice(24, None), slice(None))   # 0..1 ramp
    err = float(np.abs(got[region] - src[region]).max())
    assert err < 2e-3, f"{wcs} -> {rcs}: ramp error {err}"
    assert abs(got[0:8].mean() - 0.18) < 2e-3, f"{wcs} -> {rcs}: grey {got[0:8].mean()}"
    patches = got[16:24, 0:48:16]
    want = src[16:24, 0:48:16]
    # Display encodings clip what falls outside their gamut; the Rec.709
    # primaries are inside every gamut here.
    assert float(np.abs(patches - want).max()) < 3e-3, f"{wcs} -> {rcs}: primaries {patches[0]}"


@pytest.mark.parametrize("working", ["ACEScg", "Linear Rec.2020", "ACES2065-1"])
def test_working_space_round_trip(nodes, tmp_path, working):
    """A plate read into ACEScg and written from ACEScg comes back identical."""
    src, _ = _pattern()
    img = src.clone()
    out = str(tmp_path / "o")
    nodes.RadianceWrite().write(img, out, "IMG │ EXR (32-bit float)", color_space="ARRI LogC4",
                                working_space=working, overwrite=True)
    path = glob.glob(str(tmp_path / "*.exr"))[0]
    got, _, info = nodes.RadianceRead().read(path=path, media_type="Image", color_space="ARRI LogC4",
                                             working_space=working)
    region = (slice(16, None), slice(None))
    assert float(np.abs(got[0].numpy()[region] - src[0].numpy()[region]).max()) < 2e-3


def test_logc4_plate_lands_in_rec709_not_awg4(nodes, tmp_path):
    """Pre-3.5 read undid the LogC4 curve only: primaries stayed AWG4."""
    from radiance.color import encodings as enc
    red709 = np.zeros((4, 4, 3), np.float32); red709[..., 0] = 0.5
    logc, _ = enc.encode(red709, "ARRI LogC4", use_ocio=True)
    import OpenEXR
    p = str(tmp_path / "logc4.exr")
    OpenEXR.File({"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage},
                 {c: np.ascontiguousarray(logc[..., i]) for i, c in enumerate("RGB")}).write(p)
    got, _, info = _read(nodes, p, cs="ARRI LogC4")
    assert np.allclose(got[0, 0], [0.5, 0.0, 0.0], atol=2e-3), got[0, 0]
    assert any("ARRI LogC4" in t for t in json_loads(info)["colour_transform"])


def json_loads(info):
    import json
    return json.loads(info)


def test_exr_carries_chromaticities_and_auto_read_honours_them(nodes, tmp_path):
    src, _ = _pattern()
    nodes.RadianceWrite().write(src, str(tmp_path / "aces"), "IMG │ EXR (32-bit float)",
                                color_space="ACES2065-1", overwrite=True)
    path = glob.glob(str(tmp_path / "*.exr"))[0]
    import OpenEXR
    hdr = OpenEXR.File(path).header()
    assert np.allclose(hdr["chromaticities"], (0.7347, 0.2653, 0.0, 1.0, 0.0001, -0.077, 0.32168, 0.33767), atol=1e-4)
    assert hdr["oiio:ColorSpace"] == "ACES2065-1"
    # Auto: the file says AP0, the working space is Rec.709 -> converted back.
    got, _, info = _read(nodes, path)
    assert float(np.abs(got[16:24, 0:48:16] - src[0].numpy()[16:24, 0:48:16]).max()) < 2e-3
    assert "chromaticities say AP0" in " ".join(json_loads(info)["colour_transform"])


def test_ocio_colorspace_free_text_both_ways(nodes, tmp_path):
    src, _ = _pattern()
    nodes.RadianceWrite().write(src, str(tmp_path / "o"), "IMG │ EXR (32-bit float)",
                                ocio_colorspace="lin_ap0", overwrite=True)   # an alias
    path = glob.glob(str(tmp_path / "*.exr"))[0]
    got, _, info = nodes.RadianceRead().read(path=path, media_type="Image",
                                             ocio_colorspace="ACES2065-1")
    assert float(np.abs(got[0].numpy()[16:] - src[0].numpy()[16:]).max()) < 1e-4
    assert "OCIO [studio-config" in " ".join(json_loads(info)["colour_transform"])
    with pytest.raises(Exception, match="not in config"):
        nodes.RadianceWrite().write(src, str(tmp_path / "x"), "IMG │ EXR (32-bit float)",
                                    ocio_colorspace="No Such Space", overwrite=True)


@pytest.mark.parametrize("fmt", ["IMG │ TIFF (32-bit float)", "IMG │ TIFF (16-bit)", "IMG │ DPX", "IMG │ WEBP"])
def test_alpha_in_every_format_that_has_it(nodes, tmp_path, fmt):
    src, a = _pattern()
    path = _write(nodes, tmp_path, fmt, mask=a)
    _, mask, _ = _read(nodes, path)
    assert mask is not None and float(mask.max()) > 0.9 and float(mask.min()) < 0.1, fmt


def test_16bit_grey_png_is_read_at_16_bits(nodes, tmp_path):
    """Pillow's I;16 -> RGB convert saturates at 255: depth maps read white."""
    import cv2
    ramp = np.tile(np.linspace(0, 65535, W).astype(np.uint16)[None, :], (H, 1))
    p = str(tmp_path / "depth.png")
    cv2.imwrite(p, ramp)
    got, _, _ = _read(nodes, p)
    assert np.allclose(got[0, :, 0], ramp[0] / 65535.0, atol=1e-5), got[0, :4, 0]


def test_half_float_tiff_is_read(nodes, tmp_path):
    import tifffile
    src, _ = _pattern()
    p = str(tmp_path / "h.tif")
    tifffile.imwrite(p, src[0].numpy().astype(np.float16), photometric="rgb")
    got, _, _ = _read(nodes, p)
    assert np.allclose(got[8, 0], [16.0, 8.0, 4.0], rtol=1e-3)


def test_integer_writes_round_not_truncate(nodes, tmp_path):
    img = torch.full((1, 4, 4, 3), 0.999)   # 254.745 -> 255, 65469.3 -> 65469
    for fmt, scale in (("IMG │ PNG (8-bit)", 255), ("IMG │ PNG (16-bit)", 65535)):
        d = tmp_path / fmt.split("(")[1][:2]
        nodes.RadianceWrite().write(img, str(d / "r"), fmt, overwrite=True)
        p = glob.glob(str(d / "*.png"))[0]
        got, _, _ = _read(nodes, p)
        assert abs(float(got.max()) * scale - round(0.999 * scale)) < 0.01, fmt


VIDEO = [
    ("VID │ MOV (ProRes 422 HQ)", 0.01),
    ("VID │ MOV (ProRes 4444)", 0.01),
    ("VID │ MP4 (H.265 10-bit)", 0.03),
    ("VID │ MP4 (H.264)", 0.04),
    ("VID │ MOV (DNxHR HQ)", 0.04),
]


def _video_src(n=6):
    src, a = _pattern()
    img = src[0].numpy()[None].repeat(n, 0).clip(0, 1)
    return torch.from_numpy(img), a.repeat(n, 1, 1)


@pytest.mark.parametrize("fmt,tol", VIDEO, ids=[v[0].split("│ ")[1] for v in VIDEO])
def test_video_tags_matrix_and_round_trip(nodes, tmp_path, fmt, tol):
    """A Rec.709 delivery: BT.709 matrix (not swscale's BT.601), BT.709 tags,
    decoded back by the Read node from the tags alone (color_space Auto)."""
    from radiance.core import video as V
    frames, a = _video_src()
    if "DNxHR" in fmt:   # DNxHR's minimum raster is 256x120
        frames = torch.nn.functional.interpolate(frames.movedim(-1, 1), scale_factor=4, mode="nearest").movedim(1, -1)
        a = torch.nn.functional.interpolate(a[:, None], scale_factor=4, mode="nearest")[:, 0]
    nodes.RadianceWrite().write(frames, str(tmp_path / "clip"), fmt, color_space="Rec.709 (BT.1886)",
                                fps=24.0, quality=10, mask=a if "4444" in fmt else None, overwrite=True)
    path = [f for f in glob.glob(str(tmp_path / "*")) if os.path.isfile(f)][0]
    info = V.probe(path)
    assert info.color_primaries == "bt709" and info.color_transfer == "bt709", (info.color_primaries, info.color_transfer)
    assert info.color_space == "bt709", info.color_space
    img, mask, meta = nodes.RadianceRead().read(path=path, media_type="Video", start_frame=0)
    got = img[0].numpy()
    want = frames[0].numpy()
    # Saturated primaries are where a BT.601/709 matrix mix-up shows (hue shift).
    k = got.shape[0] // H
    prim = got[18 * k:22 * k, [4 * k, 20 * k, 36 * k]]
    assert float(np.abs(prim - want[18 * k:22 * k, [4 * k, 20 * k, 36 * k]]).max()) < 0.06, prim[0]
    ramp_err = float(np.abs(got[30 * k:40 * k, 4 * k:60 * k] - want[30 * k:40 * k, 4 * k:60 * k]).mean())
    assert ramp_err < tol, ramp_err
    if "4444" in fmt:
        assert mask is not None and float(np.abs(mask[0].numpy() - a[0].numpy()).mean()) < 0.01


def test_hdr10_pq_video_is_tagged_hdr10(nodes, tmp_path):
    from radiance.core import video as V
    frames, _ = _video_src(3)
    nodes.RadianceWrite().write(frames * 4.0, str(tmp_path / "hdr"), "VID │ MP4 (H.265 10-bit)",
                                color_space="PQ (HDR10 / ST.2084)", fps=24.0, overwrite=True)
    path = glob.glob(str(tmp_path / "*.mp4"))[0]
    info = V.probe(path)
    assert (info.color_primaries, info.color_transfer, info.color_space) == ("bt2020", "smpte2084", "bt2020nc")
    img, _, meta = nodes.RadianceRead().read(path=path, media_type="Video", start_frame=0)
    # Auto follows the smpte2084 tag back to linear: 18% grey * 4 = 0.72
    assert abs(float(img[0, 0:8].mean()) - 0.72) < 0.03


def test_sequence_frame_range_step_and_missing(nodes, tmp_path):
    frames, _ = _video_src(10)
    for i in range(10):
        frames[i] *= (i + 1) / 10.0
    out = tmp_path / "seq"
    nodes.RadianceWrite().write(frames, str(out), "SEQ │ EXR (16-bit half)", start_frame=1001, overwrite=True)
    files = sorted(glob.glob(str(out / "*.exr")))
    assert len(files) == 10 and files[0].endswith("_1001.exr") and files[-1].endswith("_1010.exr")
    os.rename(files[4], files[4] + ".bak")           # frame 1005 missing
    img, _, info = nodes.RadianceRead().read(path=files[0], start_frame=1002, end_frame=1008,
                                             frame_step=2, missing_frames="Black")
    # 1002, 1004, 1006(black? no: 1006 exists), 1008
    assert img.shape[0] == 4
    levels = [round(float(img[k, 0:8].mean()) / 0.18 * 10) for k in range(4)]
    assert levels == [2, 4, 6, 8], levels
    img, _, _ = nodes.RadianceRead().read(path=files[0], start_frame=1004, end_frame=1006,
                                          missing_frames="Black")
    assert img.shape[0] == 3 and float(img[1].abs().max()) == 0.0
    with pytest.raises(RuntimeError):
        nodes.RadianceRead().read(path=files[0], start_frame=1004, end_frame=1006, missing_frames="Error")


# ── every remaining widget does what its tooltip says ──────────────────────

@pytest.mark.parametrize("comp", ["ZIP", "ZIPS", "PIZ", "RLE", "Uncompressed", "PXR24", "B44", "B44A", "DWAA", "DWAB"])
def test_exr_compression_is_what_the_header_says(nodes, tmp_path, comp):
    import OpenEXR
    src, _ = _pattern()
    nodes.RadianceWrite().write(src, str(tmp_path / "c"), "IMG │ EXR (16-bit half)", exr_compression=comp, overwrite=True)
    p = glob.glob(str(tmp_path / "*.exr"))[0]
    got = str(OpenEXR.File(p).header()["compression"]).split(".")[-1].upper()
    want = {"Uncompressed": "NO_COMPRESSION"}.get(comp, comp + "_COMPRESSION")
    assert got == want, (comp, got)


def test_write_naming_version_padding_overwrite_proxy(nodes, tmp_path):
    src, _ = _pattern()
    w = nodes.RadianceWrite()
    w.write(src, str(tmp_path), "IMG │ PNG (8-bit)", filename="shot010_comp", version=3, overwrite=False)
    w.write(src, str(tmp_path), "IMG │ PNG (8-bit)", filename="shot010_comp", version=3, overwrite=False)
    names = sorted(os.path.basename(f) for f in glob.glob(str(tmp_path / "*.png")))
    assert names == ["shot010_comp_v0003.png", "shot010_comp_v0003_001.png"], names
    w.write(src.repeat(3, 1, 1, 1), str(tmp_path / "seq"), "SEQ │ PNG (16-bit)", start_frame=7,
            frame_padding=6, proxy_scale=0.5, overwrite=True)
    seq = sorted(os.path.basename(f) for f in glob.glob(str(tmp_path / "seq" / "*.png")))
    assert seq == ["seq_000007.png", "seq_000008.png", "seq_000009.png"], seq
    img, _, _ = _read(nodes, str(tmp_path / "seq" / seq[0]), )
    assert img.shape[:2] == (H // 2, W // 2)


def test_jpeg_quality_changes_the_file(nodes, tmp_path):
    src, _ = _pattern()
    sizes = []
    for q in (0, 40):
        d = tmp_path / str(q)
        nodes.RadianceWrite().write(src, str(d / "j"), "IMG │ JPEG", quality=q, overwrite=True)
        sizes.append(os.path.getsize(glob.glob(str(d / "*.jpg"))[0]))
    assert sizes[0] > sizes[1], sizes


def test_video_fps_and_audio(nodes, tmp_path):
    from radiance.core import video as V
    frames, _ = _video_src(12)
    sr = 48000
    audio = {"waveform": torch.zeros(1, 2, sr // 2), "sample_rate": sr}
    nodes.RadianceWrite().write(frames, str(tmp_path / "a"), "VID │ MOV (ProRes 422 HQ)", fps=23.976,
                                audio=audio, overwrite=True)
    p = glob.glob(str(tmp_path / "*.mov"))[0]
    info = V.probe(p)
    assert abs(info.fps_float - 23.976) < 1e-3, info.fps_float
    import subprocess
    streams = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0", p],
                             capture_output=True, text=True).stdout.split()
    assert "audio" in streams, streams


def test_read_premultiplied_raw_layer_on_error_proxy(nodes, tmp_path):
    import OpenEXR
    # premultiplied RGBA: colour 0.5 at alpha 0.5 is really 1.0
    rgb = np.full((8, 8), 0.5, np.float32)
    p = str(tmp_path / "pm.exr")
    OpenEXR.File({"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage},
                 {"R": rgb, "G": rgb, "B": rgb, "A": np.full((8, 8), 0.5, np.float32),
                  "depth.Z": np.full((8, 8), 7.0, np.float32)}).write(p)
    img, _, _ = nodes.RadianceRead().read(path=p, premultiplied=True)
    assert abs(float(img.mean()) - 1.0) < 1e-6
    img, _, _ = nodes.RadianceRead().read(path=p, premultiplied=False)
    assert abs(float(img.mean()) - 0.5) < 1e-6
    img, _, info = nodes.RadianceRead().read(path=p, layer="depth")
    assert abs(float(img.mean()) - 7.0) < 1e-6, info
    img, _, _ = nodes.RadianceRead().read(path=p, color_space="sRGB", raw=True)
    assert abs(float(img.mean()) - 0.5) < 1e-6          # raw: no decode
    img, _, _ = nodes.RadianceRead().read(path=p, proxy_scale=0.5)
    assert img.shape[1:3] == (4, 4)
    img, _, info = nodes.RadianceRead().read(path=str(tmp_path / "missing.exr"), on_error="Black frame")
    assert float(img.abs().max()) == 0.0 and '"kind": "error"' in info
    with pytest.raises(RuntimeError):
        nodes.RadianceRead().read(path=str(tmp_path / "missing.exr"), on_error="Error")


def test_max_video_frames_and_step(nodes, tmp_path):
    frames, _ = _video_src(12)
    nodes.RadianceWrite().write(frames, str(tmp_path / "v"), "VID │ MOV (ProRes 422 HQ)", fps=24.0, overwrite=True)
    p = glob.glob(str(tmp_path / "*.mov"))[0]
    img, _, _ = nodes.RadianceRead().read(path=p, start_frame=0, max_video_frames=5)
    assert img.shape[0] == 5
    img, _, _ = nodes.RadianceRead().read(path=p, start_frame=2, end_frame=10, frame_step=4)
    assert img.shape[0] == 3        # 2, 6, 10


def test_pass_through_video_is_not_tagged_linear():
    """Pass-through writes the IMAGE's values untouched; stamping them with a
    linear transfer made players decode ordinary display-encoded frames as
    linear light. An encoding the user actually asked for is still tagged."""
    from radiance.io.writer import OutputColour
    passthrough = OutputColour("Linear (pass-through)").ffmpeg_tags()
    assert "-color_trc" not in passthrough
    assert passthrough[passthrough.index("-color_primaries") + 1] == "bt709"
    srgb = OutputColour("sRGB").ffmpeg_tags()
    assert srgb[srgb.index("-color_trc") + 1] == "iec61966-2-1"
    linear = OutputColour("Linear Rec.709 (sRGB)").ffmpeg_tags()
    assert linear[linear.index("-color_trc") + 1] == "linear"
