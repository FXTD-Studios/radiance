"""The Radiance page (js/studio.html) in a real browser, against the real routes.

The page is served next to the actual workspace.py handlers (the ones the
conftest PromptServer stub recorded), on a temporary library that holds a copy
of the official workflow cards plus one project with three shots. ComfyUI's own
/system_stats, /history and /free are small fakes. The canvas is a parent page
that frames the page and records the postMessage traffic.

Set STUDIO_SCREENSHOTS=<dir> to keep a screenshot of every screen.
"""
from __future__ import annotations

import asyncio
import base64
import importlib
import json
import os
import shutil
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("playwright.sync_api")

from radiance.nodes.pipeline import workspace as ws_mod  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
JS = REPO / "js"
IMAGES = REPO / "docs" / "images"


PHOTOS = Path(__file__).resolve().parent / "fixtures" / "studio"  # OpenEXR sample photographs, BSD


def _photo(name: str) -> bytes:
    return (PHOTOS / f"{name}.jpg").read_bytes()


def _cover(name: str) -> str:
    return "data:image/png;base64," + base64.b64encode((IMAGES / "cards" / name).read_bytes()).decode()


def _aiohttp_names():
    return [n for n in list(sys.modules) if n == "aiohttp" or n.startswith("aiohttp.")]


@pytest.fixture
def real_web():
    """The real aiohttp.web for this test; the conftest stub goes back after."""
    saved = {n: sys.modules[n] for n in _aiohttp_names()}
    for name in saved:
        del sys.modules[name]
    try:
        web = importlib.import_module("aiohttp.web")
    except ImportError as exc:
        sys.modules.update(saved)
        pytest.skip(f"aiohttp is not installed: {exc}")
    assert callable(getattr(web, "json_response", None))
    yield web
    for name in _aiohttp_names():
        del sys.modules[name]
    sys.modules.update(saved)


@pytest.fixture
def studio(tmp_path, monkeypatch, real_web):

    root = (tmp_path / "workflows").resolve()
    shutil.copytree(REPO / "workflows" / "official", root / "official")
    inp = (tmp_path / "input").resolve(); inp.mkdir()
    out = (tmp_path / "output").resolve(); out.mkdir()
    monkeypatch.setattr(ws_mod, "web", real_web)
    monkeypatch.setattr(ws_mod, "WORKFLOW_DIR", str(root))
    monkeypatch.setattr(ws_mod, "_WORKFLOW_DIR_RESOLVED", root)
    monkeypatch.setattr(ws_mod, "_ASSETS_BINS_PATH", root / "_assets_bins.json")
    monkeypatch.setattr(ws_mod, "_THUMB_CACHE", root / ".asset_thumbs")
    monkeypatch.setattr(ws_mod.folder_paths, "get_input_directory", lambda: str(inp))
    monkeypatch.setattr(ws_mod.folder_paths, "get_output_directory", lambda: str(out))
    from radiance.nodes import gizmo as gizmo_mod
    from radiance.nodes.pipeline import setup_routes
    monkeypatch.setattr(gizmo_mod, "web", real_web)
    monkeypatch.setattr(setup_routes, "web", real_web)
    (tmp_path / "no_gizmos").mkdir()
    monkeypatch.setattr(gizmo_mod, "GIZMOS_DIR", str(tmp_path / "no_gizmos"))

    for name, photo in (("SH010_grade_v004.jpg", "window"), ("SH020_fill_v003.jpg", "candlelight"),
                        ("SH030_sky_v002.jpg", "mountain")):
        (out / name).write_bytes(_photo(photo))
    for name, photo in (("tree_plate.jpg", "tree"), ("cannon_insert.jpg", "cannon")):
        (inp / name).write_bytes(_photo(photo))

    # Every route the modules recorded. A test that swapped sys.modules["server"] before
    # one of them was first imported leaves its routes on that stub, so read each module's own.
    import server
    servers = []
    for mod in (server, ws_mod, gizmo_mod, setup_routes):
        ps = getattr(mod, "PromptServer", None)
        if ps is not None and getattr(getattr(ps.instance, "routes", None), "registered", None) is not None \
                and all(ps is not s for s in servers):
            servers.append(ps)

    app = real_web.Application(client_max_size=64 * 1024 * 1024)
    seen = set()
    for ps in servers:
        for method, path, fn in ps.instance.routes.registered:
            if path.startswith("/radiance/") and (method, path) not in seen:
                seen.add((method, path))
                app.router.add_route(method, path, fn)

    posted = []

    async def system_stats(_r):
        return real_web.json_response({"devices": [{"name": "cuda:0 NVIDIA GeForce RTX 4090 : cudaMallocAsync",
                                                    "vram_total": 24 * 2**30, "vram_free": 17 * 2**30}]})

    async def history(_r):
        now = time.time() * 1000
        return real_web.json_response({
            "p1": {"prompt": [1], "status": {"status_str": "success", "messages": [
                ["execution_start", {"timestamp": now - 600_000}], ["execution_success", {"timestamp": now - 588_000}]]},
                "outputs": {"9": {"images": [{"filename": "SH010_grade_v004.jpg", "subfolder": "", "type": "output"}]}}},
            "p2": {"prompt": [2], "status": {"status_str": "error", "messages": [
                ["execution_start", {"timestamp": now - 60_000}],
                ["execution_error", {"timestamp": now - 58_000, "node_type": "Radiance Master Write",
                                     "exception_message": "No space left on the output drive"}]]}, "outputs": {}},
        })

    async def free(r):
        posted.append(("free", await r.json()))
        return real_web.json_response({})

    async def view(r):
        return real_web.FileResponse(out / Path(r.query["filename"]).name)

    async def canvas(_r):
        return real_web.Response(content_type="text/html", text="""<!DOCTYPE html><html><body style="margin:0;background:#000">
<iframe id="f" src="/extensions/radiance/studio.html#home" style="border:0;width:100vw;height:100vh"></iframe>
<script>window.got=[];addEventListener('message',e=>{window.got.push(e.data);
if(e.data&&e.data.type==='radiance_save_project_version'){e.source.postMessage({type:'radiance_project_action_result',success:true},location.origin);}});</script>
</body></html>""")

    app.router.add_get("/system_stats", system_stats)
    app.router.add_get("/history", history)
    app.router.add_post("/free", free)
    app.router.add_get("/view", view)
    app.router.add_get("/canvas", canvas)
    app.router.add_static("/extensions/radiance/", str(JS))

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    loop = asyncio.new_event_loop()
    runner = real_web.AppRunner(app)
    ready = threading.Event()

    def serve():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(runner.setup())
        loop.run_until_complete(real_web.TCPSite(runner, "127.0.0.1", port).start())
        ready.set()
        loop.run_forever()

    t = threading.Thread(target=serve, daemon=True)
    t.start()
    assert ready.wait(10)
    base = f"http://127.0.0.1:{port}"

    # A project with three shots, saved through the real save route.
    import urllib.request

    def post(path, payload):
        req = urllib.request.Request(base + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())

    graph = json.dumps({"nodes": [{"id": 1, "type": "RadianceViewer"}], "links": []})
    for name, cover in (("SH010_grade_v003", "colour_grade.png"), ("SH010_grade_v004", "colour_grade.png"),
                        ("SH020_fill_v003", "sdr_remaster_ai_fill.png"), ("SH030_sky_v002", "sdr_to_hdr_analytic.png")):
        assert post("/radiance/workflows/save", {"filename": f"HDR_Test/{name}.rad", "content": graph,
                                                 "message": "Saved", "author": "Ahmed", "stats": {"node_count": 1},
                                                 "preview_image": _cover(cover)})["success"]
    post("/radiance/projects/hdr-test/shots/SH010/status", {"status": "Approved"})
    post("/radiance/projects/hdr-test/shots/SH020/status", {"status": "Review"})

    yield type("Studio", (), {"base": base, "posted": posted, "root": root, "out": out})

    loop.call_soon_threadsafe(loop.stop)
    t.join(5)


def _shot(page, name):
    d = os.environ.get("STUDIO_SCREENSHOTS")
    if d:
        Path(d).mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(Path(d) / f"studio_{name}.png"))


def test_studio_screens_against_the_real_routes(studio):
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900},
                                device_scale_factor=2 if os.environ.get("STUDIO_SCREENSHOTS") else 1)
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.add_init_script("try{localStorage.setItem('radiance_artist','Ahmed')}catch(e){}")
        page.goto(studio.base + "/canvas")
        f = page.frame_locator("#f")

        # Home: greeting, the shot waiting for review and the failed render, real covers.
        f.locator("h1").first.wait_for()
        assert "Ahmed" in f.locator("h1").first.inner_text()
        f.locator(".st-todo-item", has_text="SH020 is waiting for review").wait_for()
        f.locator(".st-todo-item", has_text="A render failed").wait_for()
        f.locator(".st-start").first.wait_for()
        assert f.locator(".st-start").count() == 3
        assert "RTX 4090" in f.locator("#stGpuName").inner_text()
        assert f.locator('#stNav [data-go="shots"] .st-count').inner_text() == "1"
        page.wait_for_timeout(400)
        _shot(page, "home")

        # Workflows: all the official cards plus the four saved ones, filters, details.
        f.locator('#stNav [data-go="workflows"]').click()
        f.locator(".st-tile").first.wait_for()
        total = f.locator(".st-grid .st-tile").count()
        official = len(list((REPO / "workflows" / "official").rglob("*.rad")))
        assert total == official + 4
        assert "T2I" in f.locator(".st-grid .st-tile").first.inner_text()      # the starter (PR #107)
        f.locator('.st-chip[data-id="api"]').click()
        assert f.locator(".st-grid .st-tile").count() == len(list((REPO / "workflows" / "official" / "API").glob("*.rad")))
        f.locator('.st-chip[data-id="all"]').click()
        f.locator('.st-grid .st-tile', has_text="SDR → HDR Analytic").click()
        assert "SDR → HDR Analytic" in f.locator(".st-rail h2").inner_text()
        page.wait_for_timeout(400)
        _shot(page, "workflows")

        # Open it: the canvas gets radiance_load_workflow with the graph, then a close.
        f.locator('.st-rail [data-act="open-wf"]').click()
        page.wait_for_function("window.got.some(m => m.type === 'radiance_load_workflow')")
        page.wait_for_function("window.got.some(m => m.type === 'radiance_close_dashboard')")
        msg = page.evaluate("window.got.find(m => m.type === 'radiance_load_workflow')")
        graph = msg["content"] if isinstance(msg["content"], dict) else json.loads(msg["content"])
        assert len(graph["nodes"]) >= 5

        # Search jumps from Home to Workflows.
        page.goto(studio.base + "/canvas")
        f = page.frame_locator("#f")
        f.locator("#stSearch").fill("veo")
        f.locator(".st-grid .st-tile", has_text="Veo 3").wait_for()
        assert f.locator(".st-grid .st-tile").count() == 1
        f.locator("#stSearch").fill("")

        # Assets: the input and output files, an HDR filter, a bin.
        f.locator('#stNav [data-go="assets"]').click()
        f.locator(".st-grid .st-tile").first.wait_for()
        assert f.locator(".st-grid .st-tile").count() == 5
        page.on("dialog", lambda d: d.accept("Hero plates"))
        f.locator('.st-rail [data-act="new-bin"]').click()
        f.locator('.st-rail .st-chip[aria-pressed="true"]', has_text="Hero plates").wait_for()
        page.wait_for_timeout(300)
        _shot(page, "assets")

        # Shots: three shots with their statuses; approve SH020 through the real route.
        f.locator('#stNav [data-go="shots"]').click()
        f.locator("button.st-tr").first.wait_for()
        assert f.locator("button.st-tr").count() == 3
        assert f.locator("button.st-tr", has_text="SH010").locator(".st-pill").inner_text() == "Approved"
        f.locator("button.st-tr", has_text="SH020").click()
        assert f.locator(".st-rail h2").inner_text() == "SH020"
        page.wait_for_timeout(300)
        _shot(page, "shots")
        f.locator('.st-rail [data-act="status"][data-status="Approved"]').click()
        f.locator("button.st-tr", has_text="SH020").locator(".st-pill", has_text="Approved").wait_for()
        status = json.loads((studio.root / "HDR_Test" / ".shot_status.json").read_text())
        assert status["SH020"] == "Approved"

        # Save canvas as a version: the request goes to the canvas, the answer comes back.
        f.locator('[data-act="save-version"]').click()
        page.wait_for_function("window.got.some(m => m.type === 'radiance_save_project_version' && m.projectId === 'hdr-test')")
        f.locator("#stToast", has_text="Saved the canvas").wait_for()

        # New shot (4.0): the header button asks for a name, adds the shot to the
        # project as Not started, and asks the canvas for an empty graph.
        assert f.locator("#stNew").inner_text().strip() == "New shot"
        f.locator("#stNew").click()
        dlg = f.locator("#stNewShot [role=dialog]")
        dlg.wait_for()
        name = f.locator("#stNewShotName")
        assert name.input_value() == "SH040"                     # the next free shot after SH030
        name.fill("sh20")
        assert "already in this project" in f.locator("#stNewShotErr").inner_text()
        name.fill("intro")
        assert "SH and 2 to 5 digits" in f.locator("#stNewShotErr").inner_text()
        name.fill("45")
        assert f.locator("#stNewShotFile").inner_text() == "SH045_v001"
        _shot(page, "new_shot")
        f.locator('#stNewShot button[type="submit"]').click()
        page.wait_for_function("window.got.some(m => m.type === 'radiance_new_shot' && m.shot === 'SH045' && m.projectId === 'hdr-test')")
        assert f.locator("#stNewShot").count() == 0
        row = f.locator("button.st-tr", has_text="SH045")
        row.locator(".st-pill", has_text="Not started").wait_for()
        assert row.get_attribute("aria-pressed") == "true"
        rail = f.locator(".st-rail").inner_text()
        assert "SH045_v001" in rail and "Approve" not in rail and "VERSIONS" not in rail.upper().split("NOTES")[0]
        planned = json.loads((studio.root / "HDR_Test" / ".planned_shots.json").read_text())
        assert [p["shot"] for p in planned] == ["SH045"]
        _shot(page, "shots_new_shot")
        # Escape closes the dialog, not the Radiance page.
        closes = page.evaluate("window.got.filter(m => m.type === 'radiance_close_dashboard').length")
        f.locator('[data-act="new-shot"]').click()
        dlg.wait_for()
        page.keyboard.press("Escape")
        assert f.locator("#stNewShot").count() == 0
        page.wait_for_timeout(200)
        assert page.evaluate("window.got.filter(m => m.type === 'radiance_close_dashboard').length") == closes

        # Renders: ComfyUI's history, the failure first.
        f.locator('#stNav [data-go="renders"]').click()
        f.locator(".st-render").nth(1).wait_for()
        assert "No space left" in f.locator(".st-render").nth(1).inner_text()
        page.wait_for_timeout(300)
        _shot(page, "renders")

        # Free memory goes to ComfyUI's /free.
        f.locator("#stFree").click()
        f.locator("#stToast", has_text="Freed GPU memory").wait_for()
        assert studio.posted and studio.posted[0][1] == {"unload_models": True, "free_memory": True}

        assert errors == []
        browser.close()
