"""The Gizmos page's library: what is saved, what this start-up loaded,
import, export and delete (to gizmos/.trash, never erased)."""
from __future__ import annotations

import json

import pytest

from radiance.nodes import gizmo as gz


@pytest.fixture
def library(tmp_path, monkeypatch):
    monkeypatch.setattr(gz, "GIZMOS_DIR", str(tmp_path))
    monkeypatch.setattr(gz, "LOADED_GIZMOS", {})
    return tmp_path


def _gizmo(name="chroma_ab", **kw):
    return {"name": name, "display_name": kw.pop("display_name", "Chroma Aberration"),
            "inputs": [{"name": "image", "type": "IMAGE", "target_node_id": 1, "target_input": "image"}],
            "outputs": [{"name": "image", "type": "IMAGE", "target_node_id": 1, "target_output_slot": 0}],
            "widgets": [{"name": "strength", "type": "FLOAT", "default": 0.6, "target_node_id": 1, "target_widget": "s"}],
            "nodes": [{"id": 1, "type": "ImageBlur", "widgets": {}}], "links": [], **kw}


def test_a_saved_gizmo_is_listed_and_flagged_until_a_restart_loads_it(library):
    saved = gz._save_gizmo(_gizmo())
    assert saved == {"name": "chroma_ab", "display_name": "Chroma Aberration", "replaced": False}
    [g] = gz.gizmo_summaries()
    assert g["loaded"] is False
    assert g["class_name"] == "RadianceGizmo_chroma_ab"
    assert [p["name"] for p in g["inputs"]] == ["image"] and g["knobs"] == ["strength"]
    assert g["node_types"] == ["ImageBlur"] and g["node_count"] == 1
    # What ComfyUI's start-up does: it registers the class and records it.
    classes, names = gz.load_dynamic_gizmos()
    assert "RadianceGizmo_chroma_ab" in classes
    assert gz.gizmo_summaries()[0]["loaded"] is True


def test_resaving_keeps_the_previous_definition(library):
    gz._save_gizmo(_gizmo())
    assert gz._save_gizmo(_gizmo(description="v2"))["replaced"] is True
    assert (library / "chroma_ab.gizmo.bak").is_file()
    assert json.loads((library / "chroma_ab.gizmo").read_text())["description"] == "v2"


@pytest.mark.parametrize("bad", [[], {"name": "x"}, {"name": "", "nodes": [{"id": 1}]},
                                 {"name": "x", "nodes": [{"id": 1}], "links": "no"}])
def test_an_import_that_is_not_a_gizmo_is_refused(library, bad):
    with pytest.raises(ValueError):
        gz._save_gizmo(bad)
    assert not list(library.glob("*.gizmo"))


def test_names_are_made_safe_and_paths_cannot_escape(library):
    assert gz._save_gizmo(_gizmo(name="../evil gizmo"))["name"] == "___evil_gizmo"
    assert (library / "___evil_gizmo.gizmo").is_file()
    assert gz._gizmo_path("../___evil_gizmo") is None
    assert gz._gizmo_path("___evil_gizmo") == str(library / "___evil_gizmo.gizmo")
    assert gz._gizmo_path("missing") is None


def test_an_unreadable_file_is_listed_with_its_error(library):
    (library / "broken.gizmo").write_text("{not json")
    [g] = gz.gizmo_summaries()
    assert g["name"] == "broken" and "cannot be read" in g["error"]


def test_routes_list_import_export_delete(library, monkeypatch):
    """The four routes against a real aiohttp server."""
    import asyncio
    import importlib
    import sys
    saved = {n: sys.modules[n] for n in list(sys.modules) if n == "aiohttp" or n.startswith("aiohttp.")}
    for n in saved:
        del sys.modules[n]
    try:
        web = importlib.import_module("aiohttp.web")
        from aiohttp.test_utils import TestClient, TestServer
    except ImportError:
        sys.modules.update(saved)
        pytest.skip("aiohttp is not installed")
    try:
        monkeypatch.setattr(gz, "web", web)

        async def run():
            app = web.Application()
            app.router.add_get("/list", gz.list_gizmos_api)
            app.router.add_get("/file", gz.gizmo_file_api)
            app.router.add_post("/import", gz.import_gizmo_api)
            app.router.add_post("/delete", gz.delete_gizmo_api)
            async with TestClient(TestServer(app)) as c:
                r = await c.post("/import", json=_gizmo())
                assert r.status == 200 and (await r.json())["name"] == "chroma_ab"
                r = await c.post("/import", json={"name": "nope"})
                assert r.status == 400
                listed = await (await c.get("/list")).json()
                assert [g["name"] for g in listed["gizmos"]] == ["chroma_ab"]
                assert listed["folder"] == str(library)
                r = await c.get("/file", params={"name": "chroma_ab"})
                assert r.status == 200 and json.loads(await r.read())["name"] == "chroma_ab"
                assert (await c.get("/file", params={"name": "../x"})).status == 404
                r = await c.post("/delete", json={"name": "chroma_ab"})
                body = await r.json()
                assert body["success"] and ".trash" in body["trash"]
                assert not (library / "chroma_ab.gizmo").exists()
                assert list((library / ".trash").glob("chroma_ab.*.gizmo"))
                assert (await (await c.get("/list")).json())["gizmos"] == []
        asyncio.run(run())
    finally:
        for n in [n for n in list(sys.modules) if n == "aiohttp" or n.startswith("aiohttp.")]:
            del sys.modules[n]
        sys.modules.update(saved)
