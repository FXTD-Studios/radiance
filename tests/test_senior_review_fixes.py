"""Fixes from the 4.0 senior user review (20 Nov 2026) that have no file of their own."""
from __future__ import annotations

import pytest


def test_the_region_rerender_is_in_the_menu_again_and_still_experimental():
    """4.0 hid it (a port of the 12 Nov research build); 4.1 runs the published v5 and v4.3."""
    from radiance.nodes.hdr.rudra_rerender import RadianceSDRToHDRAIFillRerender as N
    assert not getattr(N, "DEV_ONLY", False) and N.EXPERIMENTAL is True


@pytest.mark.parametrize("module, cls, key", [
    ("radiance.nodes.color.ocio", "RadianceOCIOContext", "config_path"),
    ("radiance.nodes.color.cdl", "RadianceCDLImport", "file_path"),
    ("radiance.nodes.pipeline.audio", "RadianceAudioCut", "audio_filepath"),
    ("radiance.nodes.pipeline.audio", "RadianceAudioTranscribe", "audio_filepath"),
])
def test_no_path_default_pretends_to_be_a_real_file(module, cls, key):
    import importlib
    spec = getattr(importlib.import_module(module), cls).INPUT_TYPES()["required"][key][1]
    assert spec["default"] == "" and spec.get("placeholder")


def test_an_empty_cdl_path_says_so():
    from radiance.nodes.color.cdl import RadianceCDLImport
    with pytest.raises(ValueError, match="file_path is empty"):
        RadianceCDLImport().load(file_path="")


def test_an_empty_ocio_context_loads_the_active_config():
    pytest.importorskip("PyOpenColorIO")
    from radiance.nodes.color import ocio as node
    if not node.HAS_OCIO:
        pytest.skip("OCIO manager stubbed")
    out = node.RadianceOCIOContext().set_context(config_path="")
    ctx = out["result"][0]
    assert ctx["status"] == "active" and ctx["path"]


def test_the_output_check_names_where_the_shipped_workflows_save(monkeypatch, tmp_path):
    import shutil
    from radiance.core import first_start as fs
    monkeypatch.setattr(shutil, "disk_usage", lambda p: type("U", (), {"free": 500 * 1024 ** 3})())
    (tmp_path / "radiance_output").mkdir()
    c = fs.check_output(tmp_path / "radiance_output")
    assert "shipped workflows" in c["detail"] and "outside ComfyUI" not in c["detail"]
