"""V5 viewer: the regions input (Check > Rebuilt by AI)."""

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



@pytest.mark.real_torch
def test_regions_input_is_last_and_optional():
    from radiance.nodes.monitor.viewer import RadianceViewer
    opt = RadianceViewer.INPUT_TYPES()["optional"]
    assert list(opt)[-1] == "regions", "appended, so saved graphs keep their slots"
    assert opt["regions"][0] == "IMAGE"


@pytest.mark.real_torch
def test_regions_are_written_as_pngs_and_are_not_timeline_frames(temp_out):
    from radiance.nodes.monitor.viewer import RadianceViewer
    img = torch.rand(3, 16, 24, 3)
    reg = torch.zeros(2, 16, 24, 3)
    reg[:, :, 18:, :] = torch.tensor([0.25, 0.78, 0.86])
    res = RadianceViewer().view(img, regions=reg, unique_id="rg1")
    entries = res["ui"]["radiance_images"]
    regions = [e for e in entries if e.get("is_regions")]
    main = [e for e in entries if not any(e.get(k) for k in ("is_regions", "is_compare", "is_zdepth", "is_normal"))]
    assert len(main) == 3
    assert [e["frame"] for e in regions] == [0, 1]
    for e in regions:
        assert e["type"] == "temp" and e["filename"].endswith(".png")
        assert "source_encoding" not in e, "a picture, not a frame to be displayed through a view"
        px = _png(os.path.join(temp_out, e["filename"]))
        assert px.shape[:2] == (16, 24)
        assert tuple(px[0, 20, :3]) == (64, 199, 219)
        assert tuple(px[0, 2, :3]) == (0, 0, 0)


@pytest.mark.real_torch
def test_no_regions_input_writes_nothing_extra(temp_out):
    from radiance.nodes.monitor.viewer import RadianceViewer
    res = RadianceViewer().view(torch.rand(1, 8, 8, 3), unique_id="rg2")
    assert not [e for e in res["ui"]["radiance_images"] if e.get("is_regions")]
