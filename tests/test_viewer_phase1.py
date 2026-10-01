"""Viewer / Lite Viewer phase 1 (3.5.0): what the pixels are, how they travel.

* every frame is tagged sRGB-encoded or linear (+ OCIO colour space); Auto
  reads a ComfyUI IMAGE in 0-1 as sRGB and anything outside it as linear
* float frames travel as fp16 by default (fp32 on request), clamped so large
  values do not become inf
* the PNG previews are display-referred: sRGB untouched, linear through
  OpenColorIO ACES 2.0 SDR (the same view the float path shows)
* the source fps reaches the payload
* an unchanged viewer no longer marks everything downstream dirty
"""
import os
import pathlib
import struct
import tempfile
import zlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")
RADIANCE_TORCH_GATED = True

try:
    import PyOpenColorIO as _ocio  # noqa: F401
    # Other test modules install a stand-in under this name; only the real
    # library (a file on disk) counts.
    HAS_OCIO = bool(getattr(_ocio, "__file__", None)) and hasattr(_ocio, "ColorSpaceTransform")
except Exception:  # noqa: BLE001
    HAS_OCIO = False


@pytest.fixture(autouse=True)
def _real_shared_modules():
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


def _png(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGBA"))


def _rhdr(path):
    raw = pathlib.Path(path).read_bytes()
    magic, w, h, c, flags = struct.unpack("<4sHHHH", raw[:12])
    assert magic == b"RHDR"
    data = zlib.decompress(raw[12:])
    arr = np.frombuffer(data, dtype=np.float32 if flags == 1 else np.float16)
    return flags, arr.reshape(h, w, c).astype(np.float32)


# ── tagging ──────────────────────────────────────────────────────────────────

@pytest.mark.real_torch
def test_auto_reads_comfy_images_as_srgb_and_hdr_as_linear():
    from radiance.nodes.monitor.viewer import resolve_viewer_input_space as r
    assert r("Auto", torch.rand(2, 8, 8, 3)) == ("srgb", "sRGB Encoded Rec.709 (sRGB)")
    hdr = torch.rand(2, 8, 8, 3); hdr[0, 0, 0, 0] = 3.0
    assert r("Auto", hdr) == ("linear", "Linear Rec.709 (sRGB)")
    neg = torch.rand(1, 8, 8, 3); neg[0, 1, 1, 1] = -0.2
    assert r("Auto", neg)[0] == "linear"
    assert r("ACEScg", torch.rand(1, 4, 4, 3)) == ("linear", "ACEScg")
    assert r("sRGB (ComfyUI IMAGE)", hdr)[0] == "srgb"
    # alpha above 1 is not colour and must not flip the decision
    rgba = torch.rand(1, 4, 4, 4) * 0.9; rgba[..., 3] = 2.0
    assert r("Auto", rgba)[0] == "srgb"


@pytest.mark.real_torch
def test_every_frame_is_tagged_and_fps_defaults_to_24(temp_out):
    from radiance.nodes.monitor.viewer import RadianceViewer
    res = RadianceViewer().view(torch.rand(3, 16, 24, 3), unique_id="p1")
    ui = res["ui"]
    assert ui["source_encoding"] == ["srgb"]
    assert ui["fps"] == [24.0]
    main = [e for e in ui["radiance_images"] if not e.get("is_compare")]
    assert len(main) == 3
    assert all(e["source_encoding"] == "srgb" and e["preview_encoding"] == "display" for e in main)


@pytest.mark.real_torch
def test_fps_comes_from_the_widget_or_the_video(temp_out):
    from radiance.nodes.monitor.viewer import RadianceViewer
    assert RadianceViewer().view(torch.rand(2, 8, 8, 3), fps=29.97, unique_id="p2")["ui"]["fps"] == [29.97]

    class _Comp:
        images = torch.rand(2, 8, 8, 3)
        frame_rate = 23.976

    class _Video:
        def get_components(self):
            return _Comp()

    assert RadianceViewer().view(_Video(), unique_id="p3")["ui"]["fps"] == [pytest.approx(23.976)]


# ── transport precision ──────────────────────────────────────────────────────

@pytest.mark.real_torch
def test_half_is_the_default_precision_and_full_is_available(temp_out):
    from radiance.nodes.monitor.viewer import RadianceViewer
    img = torch.rand(1, 8, 8, 3); img[0, 0, 0, 0] = 1e5
    res = RadianceViewer().view(img, unique_id="p4")
    e = res["ui"]["radiance_images"][0]
    assert e["hdr_fp32"] is False
    flags, arr = _rhdr(os.path.join(temp_out, e["hdr_filename"]))
    assert flags == 0
    assert np.isfinite(arr).all(), "1e5 must clamp to the fp16 range, not become inf"
    assert arr[0, 0, 0] == pytest.approx(65504.0)

    res = RadianceViewer().view(img, float_precision="Full (32-bit)", unique_id="p5")
    e = res["ui"]["radiance_images"][0]
    flags, arr = _rhdr(os.path.join(temp_out, e["hdr_filename"]))
    assert flags == 1 and arr[0, 0, 0] == pytest.approx(1e5)


def test_the_new_widgets_are_visible_and_not_named_bit_depth():
    """The JS force-hides any widget called bit_depth on this node."""
    from radiance.nodes.monitor.viewer import RadianceViewer
    opt = RadianceViewer.INPUT_TYPES()["optional"]
    assert list(opt)[-3:] == ["input_space", "float_precision", "fps"], "appended, positions kept"
    assert "bit_depth" not in opt
    assert opt["float_precision"][1]["default"] == "Half (16-bit)"


# ── previews ─────────────────────────────────────────────────────────────────

@pytest.mark.real_torch
def test_srgb_preview_is_untouched_and_rounded(temp_out):
    from radiance.nodes.monitor.viewer import RadianceViewer
    img = torch.full((1, 8, 8, 3), 0.5)
    e = RadianceViewer().view(img, unique_id="p6")["ui"]["radiance_images"][0]
    px = _png(os.path.join(temp_out, e["filename"]))
    assert px[0, 0, 0] == 128, "0.5 must round to 128 (truncation gave 127)"


@pytest.mark.real_torch
@pytest.mark.skipif(not HAS_OCIO, reason="OpenColorIO not installed")
def test_linear_preview_goes_through_aces2_like_the_float_path(temp_out):
    from radiance.nodes.monitor.viewer import RadianceViewer
    img = torch.full((1, 8, 8, 3), 0.18); img[0, 0, 0] = 4.0          # linear by Auto
    e = RadianceViewer().view(img, unique_id="p7")["ui"]["radiance_images"][0]
    assert e["source_encoding"] == "linear"
    px = _png(os.path.join(temp_out, e["filename"]))
    assert abs(int(px[4, 4, 0]) - 89) <= 1, f"0.18 through ACES 2.0 SDR is 89/255, got {px[4, 4, 0]}"


@pytest.mark.real_torch
def test_frames_prepared_together_come_back_in_order_with_their_own_pixels(temp_out):
    """Several frames are prepared at once; the payload keeps the clip's order."""
    from radiance.nodes.monitor.viewer import RadianceViewer
    values = [0.5 + i for i in range(12)]                                  # linear by Auto
    img = torch.tensor(values).view(-1, 1, 1, 1).expand(-1, 8, 8, 3).contiguous()
    entries = RadianceViewer().view(img, unique_id="p8")["ui"]["radiance_images"]
    assert [e["frame"] for e in entries] == list(range(len(values)))
    for e, v in zip(entries, values):
        _, px = _rhdr(os.path.join(temp_out, e["hdr_sidecar"]))
        assert px[4, 4, 0] == v, f"frame {e['frame']} holds {px[4, 4, 0]}, expected {v}"


@pytest.mark.real_torch
def test_depth_frames_prepared_together_come_back_in_order(temp_out):
    from radiance.nodes.monitor.viewer import RadianceViewer
    values = [0.05 * (i + 1) for i in range(12)]
    img = torch.full((12, 8, 8, 3), 0.5)
    depth = torch.tensor(values).view(-1, 1, 1, 1).expand(-1, 8, 8, 3).contiguous()
    entries = [e for e in RadianceViewer().view(img, zdepth=depth, unique_id="p9")["ui"]["radiance_images"]
               if e.get("is_zdepth")]
    assert [e["frame"] for e in entries] == list(range(len(values)))
    for e, v in zip(entries, values):
        assert e["depth_range"][0] == pytest.approx(v), f"depth frame {e['frame']} holds another frame's depth"
        _, px = _rhdr(os.path.join(temp_out, e["hdr_sidecar"]))
        assert px[4, 4, 0] == pytest.approx(v, abs=1e-3)


@pytest.mark.skipif(not HAS_OCIO, reason="OpenColorIO not installed")
def test_display_preview_is_exact_ocio():
    from radiance.color.display_preview import aces2_processor, display_preview
    rng = np.random.default_rng(0)
    frame = (rng.random((300, 400, 3), dtype=np.float32) ** 2) * 8
    out = display_preview(frame, "linear", "ACEScg")
    ref = np.ascontiguousarray(frame.reshape(-1, 3))
    aces2_processor("ACEScg").applyRGB(ref)
    assert np.abs(out - np.clip(ref.reshape(frame.shape), 0, 1)).max() < 1e-6


@pytest.mark.skipif(not HAS_OCIO, reason="OpenColorIO not installed")
def test_display_preview_takes_a_frame_with_planar_channels():
    """ComfyUI's VAE decode returns movedim(1, -1): channels stay in planes, and
    element-wise nodes keep that layout. OCIO failed every such HDR frame with
    'function only supports C-contiguous (row-major) arrays'."""
    from radiance.color.analysis import safe_tensor_to_numpy
    from radiance.color.display_preview import display_preview
    image = (torch.rand(1, 3, 300, 400) * 8).movedim(1, -1)
    assert not image.is_contiguous()
    out = display_preview(safe_tensor_to_numpy(image[0]), "linear", "ACEScg")
    ref = display_preview(safe_tensor_to_numpy(image[0].contiguous()), "linear", "ACEScg")
    assert np.array_equal(out, ref)


# ── Lite Viewer, removed ─────────────────────────────────────────────────────

def test_the_lite_viewer_node_is_gone():
    """Removed in 3.5.0: the Viewer's Simple mode replaces it. The frontend
    converts a saved Lite Viewer into a Viewer when a graph loads
    (js/tests/viewer_compare_mode.test.mjs checks that conversion)."""
    import radiance
    from pathlib import Path
    assert "RadianceLiteViewer" not in radiance.NODE_CLASS_MAPPINGS
    assert not (Path(radiance.__file__).parent / "nodes" / "monitor" / "lite_viewer.py").exists()


# ── caching ──────────────────────────────────────────────────────────────────

@pytest.mark.real_torch
def test_video_audio_is_preserved_and_cleaned_with_preview_files(temp_out):
    import wave
    from types import SimpleNamespace
    from radiance.nodes.monitor.viewer import RadianceViewer

    samples = torch.tensor([[[-1.0, 0.0, 1.0], [0.5, -0.5, 0.0]]])
    original = samples.clone()

    class Video:
        calls = 0

        def get_components(self):
            self.calls += 1
            return SimpleNamespace(images=torch.full((2, 8, 8, 3), 0.5),
                                   frame_rate=24, audio={"waveform": samples, "sample_rate": 48000})

    video = Video()
    result = RadianceViewer().view(video, unique_id="audio-preview")
    assert video.calls == 1, "decoding a VIDEO repeatedly duplicates work"
    first = RadianceViewer.IS_CHANGED(image=video, unique_id="audio-preview")
    samples.mul_(0.5)
    second = RadianceViewer.IS_CHANGED(image=video, unique_id="audio-preview")
    assert first != second, "audio changes must invalidate cached preview output"
    samples.copy_(original)
    assert result["ui"]["audio_fps"] == [24]
    filename = os.path.join(temp_out, result["ui"]["audio"][0]["filename"])
    with wave.open(filename, "rb") as stream:
        assert (stream.getnchannels(), stream.getframerate(), stream.getsampwidth()) == (2, 48000, 2)
        pcm = np.frombuffer(stream.readframes(3), dtype="<i2").reshape(3, 2)
    np.testing.assert_array_equal(pcm, [[-32767, 16383], [0, -16383], [32767, 0]])
    assert torch.equal(samples, original)
    result = RadianceViewer().view(torch.zeros(1, 8, 8, 3), unique_id="audio-preview")
    assert result["ui"]["audio"] == []
    assert not os.path.exists(filename)

@pytest.mark.real_torch
def test_an_unchanged_viewer_is_not_always_dirty(temp_out):
    from radiance.nodes.monitor.viewer import RadianceViewer
    img = torch.rand(2, 8, 8, 3)
    first = RadianceViewer.IS_CHANGED(image=img, unique_id="c1")
    assert first != first, "nothing cached yet: must be NaN (always run)"
    RadianceViewer().view(img, unique_id="c1")
    a = RadianceViewer.IS_CHANGED(image=img, unique_id="c1", exposure_bracketing=False)
    b = RadianceViewer.IS_CHANGED(image=img.clone(), unique_id="c1", exposure_bracketing=False)
    assert a == b, "same pixels and widgets must fingerprint the same"
    c = RadianceViewer.IS_CHANGED(image=img + 0.01, unique_id="c1", exposure_bracketing=False)
    assert c != a
    d = RadianceViewer.IS_CHANGED(image=img, unique_id="c1", exposure_bracketing=True)
    assert d != a, "a widget change must re-run the viewer"


@pytest.mark.real_torch
def test_the_viewers_never_modify_the_image_they_pass_through(temp_out):
    """OCIO applies in place; a view of the input used to be handed to it, so
    the IMAGE output downstream came back as display values."""
    from radiance.nodes.monitor.viewer import RadianceViewer
    img = torch.full((2, 16, 16, 3), 0.18); img[:, 0, 0, 0] = 4.0
    before = img.clone()
    out = RadianceViewer().view(img, unique_id="m1")["result"][0]
    assert torch.equal(img, before), "the viewer overwrote its input"
    assert torch.equal(out, before)
