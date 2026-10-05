"""Characterization of the four .rhdr writers, before they share one encoder.

The .rhdr sidecar is read by one parser (js/radiance_viewer.js _parseRHDR) but
written in four places that were each written by hand:

* nodes/monitor/viewer.py _process_frame, fp16 and fp32 branches
* nodes/monitor/viewer.py zdepth sidecar
* hdr/vae.py RadianceVAE4KDecode._save_rhdr (also called by
  nodes/generate/engine.py)

These tests pin what each one writes today: the 12-byte header, every sample
of the payload, the file name and the zlib level, so the writers can be moved
onto a shared encoder without changing a byte the viewer sees.

The zlib level is pinned on purpose. The viewer stores (level 0) because
compressing cost 264 ms per 1080p frame there; the VAE export uses level 6.
Changing either is an optimization decision, not part of the refactor.

Inputs stay inside the fp16 range except where a writer clamps today. The VAE
and zdepth writers do not clamp, and what they do above 65504 is a bug, not a
contract to keep.
"""
import os
import re
import struct
import tempfile
import zlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")
RADIANCE_TORCH_GATED = True

ZLIB_STORED = b"\x78\x01"   # zlib header written by level 0
ZLIB_DEFAULT = b"\x78\x9c"  # zlib header written by levels 2-6

# One row of RGB samples: negative, mid-grey, a bright HDR value, and a value
# past the fp16 maximum (65504) that only the viewer frame writer clamps.
VIEWER_ROW = [[-2.0, 0.5, 1e3], [1e5, 0.25, 3.0]]
VAE_ROW = [[-2.0, 0.5, 1e3], [1e4, 0.25, 3.0]]


@pytest.fixture(autouse=True)
def _real_shared_modules():
    """Undo stand-ins other test modules leave for the modules the viewer uses."""
    import importlib
    import sys
    stubbed = {}
    for name in ("radiance.path_utils", "radiance.color_utils", "radiance.hdr.utils"):
        mod = sys.modules.get(name)
        if mod is not None and getattr(mod, "__file__", None) is None:
            stubbed[name] = mod
            del sys.modules[name]
            importlib.import_module(name)
    from radiance.core.system.path_utils import safe_join as real_safe_join
    rebound = []
    for name in ("radiance.nodes.monitor.viewer",):
        mod = sys.modules.get(name)
        if mod is not None and getattr(mod, "safe_join", None) is not real_safe_join:
            rebound.append((mod, mod.safe_join))
            mod.safe_join = real_safe_join
    try:
        yield
    finally:
        for mod, previous in rebound:
            mod.safe_join = previous
        for name, mod in stubbed.items():
            sys.modules[name] = mod


@pytest.fixture
def temp_out(monkeypatch):
    import folder_paths
    from radiance.nodes.monitor import viewer as _viewer
    d = tempfile.mkdtemp()
    seen = set()
    for mod in (folder_paths, _viewer.folder_paths):
        if id(mod) in seen:
            continue
        seen.add(id(mod))
        monkeypatch.setattr(mod, "get_temp_directory", lambda: d, raising=False)
    return d


def _read(path):
    """Header fields, the zlib header bytes, and the payload as stored."""
    raw = open(path, "rb").read()
    magic, w, h, c, flags = struct.unpack("<4sHHHH", raw[:12])
    assert magic == b"RHDR"
    dtype = np.float32 if flags == 1 else np.float16
    payload = np.frombuffer(zlib.decompress(raw[12:]), dtype=dtype)
    assert payload.size == w * h * c, "payload size must match the header, or the JS parser rejects it"
    return (w, h, c, flags), raw[12:14], payload.reshape(h, w, c)


def _image(row, height=4):
    """A (1, H, 2, 3) IMAGE whose every row is ``row``."""
    return torch.tensor(row, dtype=torch.float32).view(1, 1, 2, 3).expand(1, height, 2, 3).contiguous()


def _rgba(row, height=4):
    rgb = np.broadcast_to(np.asarray(row, dtype=np.float32), (height, 2, 3))
    return np.concatenate([rgb, np.ones((height, 2, 1), np.float32)], axis=-1)


# ── viewer frame sidecar ─────────────────────────────────────────────────────

@pytest.mark.real_torch
def test_viewer_half_frame_is_rgba_fp16_clamped_and_stored(temp_out):
    from radiance.nodes.monitor.viewer import RadianceViewer
    e = RadianceViewer().view(_image(VIEWER_ROW), unique_id="rw1")["ui"]["radiance_images"][0]
    assert re.fullmatch(r"Radiance_viewer_[0-9a-f]{12}_0\.rhdr", e["hdr_filename"])
    assert e["hdr_sidecar"] == e["hdr_filename"]
    header, zhead, px = _read(os.path.join(temp_out, e["hdr_filename"]))
    assert header == (2, 4, 4, 0), "RGB is padded to RGBA; flags 0 is fp16"
    assert zhead == ZLIB_STORED
    expected = np.clip(_rgba(VIEWER_ROW), -65504.0, 65504.0).astype(np.float16)
    np.testing.assert_array_equal(px, expected)


@pytest.mark.real_torch
def test_viewer_full_frame_is_rgba_fp32_unclamped_and_stored(temp_out):
    from radiance.nodes.monitor.viewer import RadianceViewer
    e = RadianceViewer().view(_image(VIEWER_ROW), float_precision="Full (32-bit)",
                              unique_id="rw2")["ui"]["radiance_images"][0]
    assert re.fullmatch(r"Radiance_viewer_[0-9a-f]{12}_0\.rhdr", e["hdr_filename"])
    header, zhead, px = _read(os.path.join(temp_out, e["hdr_filename"]))
    assert header == (2, 4, 4, 1), "flags 1 is fp32"
    assert zhead == ZLIB_STORED
    np.testing.assert_array_equal(px, _rgba(VIEWER_ROW))


@pytest.mark.real_torch
def test_viewer_rgba_frame_keeps_its_own_alpha(temp_out):
    from radiance.nodes.monitor.viewer import RadianceViewer
    rgba = np.array([[[0.5, 2.0, 0.25, 0.0], [1.5, 0.5, 4.0, 0.75]]], dtype=np.float32).repeat(4, axis=0)
    e = RadianceViewer().view(torch.from_numpy(rgba).unsqueeze(0), float_precision="Full (32-bit)",
                              unique_id="rw3")["ui"]["radiance_images"][0]
    header, _, px = _read(os.path.join(temp_out, e["hdr_filename"]))
    assert header == (2, 4, 4, 1)
    np.testing.assert_array_equal(px, rgba)


@pytest.mark.real_torch
def test_viewer_8bit_writes_no_float_sidecar(temp_out):
    from radiance.nodes.monitor.viewer import RadianceViewer
    e = RadianceViewer().view(_image(VIEWER_ROW), bit_depth="8-bit (Fast)",
                              unique_id="rw4")["ui"]["radiance_images"][0]
    assert e["hdr_filename"] is None
    assert not [f for f in os.listdir(temp_out) if f.endswith(".rhdr")]


# ── viewer zdepth sidecar ────────────────────────────────────────────────────

@pytest.mark.real_torch
@pytest.mark.parametrize("precision, flags, dtype", [("", 0, np.float16), ("Full (32-bit)", 1, np.float32)])
def test_viewer_depth_sidecar_is_one_channel_raw_depth(temp_out, precision, flags, dtype):
    from radiance.nodes.monitor.viewer import RadianceViewer
    depth = torch.tensor([0.5, 7.5, 120.0, 3000.0]).view(1, 4, 1, 1).expand(1, 4, 2, 1).contiguous()
    entries = RadianceViewer().view(_image(VAE_ROW), float_precision=precision, zdepth=depth,
                                    unique_id="rw5")["ui"]["radiance_images"]
    (e,) = [x for x in entries if x.get("is_zdepth")]
    assert re.fullmatch(r"Radiance_zdepth_[0-9a-f]{12}_0_float\.rhdr", e["hdr_sidecar"])
    header, zhead, px = _read(os.path.join(temp_out, e["hdr_sidecar"]))
    assert header == (2, 4, 1, flags)
    assert zhead == ZLIB_STORED
    expected = depth[0].numpy().astype(dtype)
    np.testing.assert_array_equal(px, expected)


# ── HDR VAE export (RadianceVAE4KDecode._save_rhdr) ──────────────────────────

def _vae_frame():
    return np.broadcast_to(np.asarray(VAE_ROW, dtype=np.float32), (3, 2, 3)).copy()


@pytest.mark.real_torch
@pytest.mark.parametrize("precision, flags, dtype", [
    ("f16", 0, np.float16),
    ("f32", 1, np.float32),
    ("anything else", 0, np.float16),   # only "f32" selects fp32
])
def test_vae_export_header_payload_and_name(precision, flags, dtype):
    from radiance.hdr.vae import RadianceVAE4KDecode
    d = tempfile.mkdtemp()
    frame = _vae_frame()
    name = RadianceVAE4KDecode._save_rhdr(frame, d, prefix="radiance_4k_f0007", precision=precision)
    assert re.fullmatch(r"radiance_4k_f0007_[0-9a-f]{12}\.rhdr", name)
    assert os.listdir(d) == [name]
    header, zhead, px = _read(os.path.join(d, name))
    assert header == (2, 3, 3, flags), "the VAE writes RGB, no alpha padding"
    assert zhead == ZLIB_DEFAULT
    np.testing.assert_array_equal(px, frame.astype(dtype))


@pytest.mark.real_torch
def test_vae_export_default_prefix_and_precision():
    from radiance.hdr.vae import RadianceVAE4KDecode
    d = tempfile.mkdtemp()
    name = RadianceVAE4KDecode._save_rhdr(_vae_frame(), d)
    assert re.fullmatch(r"radiance_4k_[0-9a-f]{12}\.rhdr", name)
    assert _read(os.path.join(d, name))[0][3] == 0


@pytest.mark.real_torch
def test_vae_export_skips_frames_wider_than_the_uint16_header():
    from radiance.hdr.vae import RadianceVAE4KDecode
    d = tempfile.mkdtemp()
    assert RadianceVAE4KDecode._save_rhdr(np.zeros((1, 65536, 3), np.float32), d) is None
    assert os.listdir(d) == []


@pytest.mark.real_torch
def test_vae_export_returns_none_instead_of_raising():
    from radiance.hdr.vae import RadianceVAE4KDecode
    missing = os.path.join(tempfile.mkdtemp(), "does-not-exist")
    assert RadianceVAE4KDecode._save_rhdr(_vae_frame(), missing) is None
