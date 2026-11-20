"""The Radiance page's Gizmos and Models screens in a real browser, against the real routes.

Same harness as test_studio_browser.py. Gizmos: a temporary gizmos/ folder with
one gizmo loaded at start-up and one saved since; add to canvas, export,
import, delete. Models: the setup check on a temporary models folder with a
fake GPU; "Download missing" goes through the real queue with the transfer
replaced, and the workflow turns Ready.

Set STUDIO_SCREENSHOTS=<dir> to keep a screenshot of both screens.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("playwright.sync_api")

from test_studio_browser import _shot, real_web, studio  # noqa: E402,F401  (fixtures)

from radiance.core import setup_check as sc  # noqa: E402
from radiance.nodes import gizmo as gz  # noqa: E402


def _gizmo(name, display, loaded_types=("ImageBlur", "ImageSharpen")):
    return {"name": name, "display_name": display, "category": "FXTD STUDIOS/Radiance/◎ Gizmos",
            "description": f"{display}: a test gizmo.",
            "inputs": [{"name": "image", "type": "IMAGE", "target_node_id": 1, "target_input": "image"}],
            "outputs": [{"name": "image", "type": "IMAGE", "target_node_id": 2, "target_output_slot": 0}],
            "widgets": [{"name": "strength", "type": "FLOAT", "default": 0.6, "target_node_id": 1, "target_widget": "s"}],
            "nodes": [{"id": i + 1, "type": t, "widgets": {}} for i, t in enumerate(loaded_types)], "links": []}


@pytest.fixture
def library(studio, real_web, tmp_path, monkeypatch):
    from radiance.nodes.pipeline import setup_routes
    monkeypatch.setattr(gz, "web", real_web)
    monkeypatch.setattr(setup_routes, "web", real_web)
    gdir = tmp_path / "gizmos"
    gdir.mkdir()
    monkeypatch.setattr(gz, "GIZMOS_DIR", str(gdir))
    monkeypatch.setattr(gz, "LOADED_GIZMOS", {})
    gz._save_gizmo(_gizmo("bloom_kit", "Bloom Kit"))
    gz.load_dynamic_gizmos()                                   # loaded at "start-up"
    gz._save_gizmo(_gizmo("chroma_ab", "Chroma Aberration"))    # saved since
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(sc, "models_dir", lambda: models)
    from radiance.core import first_start
    monkeypatch.setattr(first_start, "default_output", lambda: tmp_path / "radiance_output")
    monkeypatch.setattr(sc, "_folder_dirs", lambda folder: [models / folder])
    packs = set()
    monkeypatch.setattr(sc, "pack_installed", lambda mid: mid in packs)
    monkeypatch.setattr(sc, "hardware", lambda: {
        "gpus": [{"name": "NVIDIA GeForce RTX 4090", "kind": "cuda", "vram": 25_757_220_864}],
        "ram": 68_000_000_000, "ram_free": 41_000_000_000, "disk_free": 812_000_000_000,
        "torch": "2.12.1+cu130", "cuda": "13.0", "comfyui": "0.46.0", "radiance": "4.0.0",
        "os": "Windows 11", "python": "3.12.10"})
    monkeypatch.delenv("RADIANCE_ALLOW_DOWNLOADS", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)

    def pack(mid, job):
        (models / "radiance" / mid.split("/", 1)[1]).mkdir(parents=True, exist_ok=True)
        packs.add(mid)

    def meta(url):
        return {"revision": "a" * 40, "sha256": "b" * 64, "size": 2_000_000}

    def fetch(url, dest, *, progress, **kw):
        progress(1_000_000, 2_000_000)
        Path(dest).write_bytes(b"x" * 16)
        progress(2_000_000, 2_000_000)
        return dest
    monkeypatch.setattr(sc, "QUEUE", sc.DownloadQueue(fetcher=fetch, pack_downloader=pack, metadata=meta))
    return type("L", (), {"gizmos": gdir, "models": models, "packs": packs})


@pytest.mark.timeout(120)
def test_gizmos_and_models_screens(studio, library, tmp_path):
    from playwright.sync_api import sync_playwright
    import os

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900},
                                device_scale_factor=2 if os.environ.get("STUDIO_SCREENSHOTS") else 1)
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("dialog", lambda d: d.accept())
        page.goto(studio.base + "/canvas")
        f = page.frame_locator("#f")

        # Gizmos: both listed by name, the one saved since start-up asks for a restart.
        f.locator('#stNav [data-go="gizmos"]').click()
        f.locator(".st-grid .st-tile").nth(1).wait_for()
        assert f.locator(".st-grid .st-tile").count() == 2
        assert "Bloom Kit" in f.locator(".st-grid .st-tile").first.inner_text()
        f.locator(".st-notice", has_text="1 gizmo saved since ComfyUI started").wait_for()
        assert "In the node menu" in f.locator(".st-grid .st-tile").first.inner_text()
        f.locator(".st-grid .st-tile", has_text="Chroma Aberration").click()
        assert f.locator('.st-rail [data-act="add-gizmo"]').is_disabled()
        f.locator(".st-grid .st-tile", has_text="Bloom Kit").click()
        assert "strength" in f.locator(".st-rail").inner_text()
        assert "ImageSharpen" in f.locator(".st-rail").inner_text()
        page.wait_for_timeout(300)
        _shot(page, "gizmos")

        # Add to canvas: the canvas gets the node type to create.
        f.locator('.st-rail [data-act="add-gizmo"]').click()
        page.wait_for_function("window.got.some(m => m.type === 'radiance_add_node')")
        msg = page.evaluate("window.got.find(m => m.type === 'radiance_add_node')")
        assert msg["nodeType"] == "RadianceGizmo_bloom_kit" and msg["title"] == "Bloom Kit"

        # Search filters gizmos too.
        f.locator("#stSearch").fill("chroma")
        page.wait_for_timeout(300)
        assert f.locator(".st-grid .st-tile").count() == 1
        f.locator("#stSearch").fill("")
        page.wait_for_timeout(300)

        # Import a shared .gizmo, then delete it: it goes to .trash.
        shared = tmp_path / "glow.gizmo"
        shared.write_text(json.dumps(_gizmo("glow", "Glow")))
        f.locator("#stGizmoFile").set_input_files(str(shared))
        f.locator(".st-grid .st-tile", has_text="Glow").wait_for()
        assert (library.gizmos / "glow.gizmo").is_file()
        f.locator(".st-grid .st-tile", has_text="Glow").click()
        f.locator('.st-rail [data-act="delete-gizmo"]').click()
        f.locator("#stToast", has_text="Glow deleted").wait_for()
        assert not (library.gizmos / "glow.gizmo").exists()
        assert list((library.gizmos / ".trash").glob("glow.*.gizmo"))

        # Delete straight from the library grid: the tile's trash button (4.0) ...
        for name, title in (("halo", "Halo"), ("haze", "Haze")):
            (library.gizmos / f"{name}.gizmo").write_text(json.dumps(_gizmo(name, title)))
        f.locator('[data-act="reload"]').first.click()
        tile = f.locator(".st-tile-wrap", has_text="Halo")
        tile.wait_for()
        tile.hover()
        _shot(page, "gizmos_tile_delete")
        tile.locator('.st-tile-del[data-act="delete-gizmo"]').click()
        f.locator("#stToast", has_text="Halo deleted").wait_for()
        assert not (library.gizmos / "halo.gizmo").exists()
        assert list((library.gizmos / ".trash").glob("halo.*.gizmo"))
        assert f.locator(".st-tile-wrap", has_text="Halo").count() == 0
        # ... and the Delete key on a focused tile.
        f.locator(".st-grid .st-tile", has_text="Haze").focus()
        page.keyboard.press("Delete")
        f.locator("#stToast", has_text="Haze deleted").wait_for()
        assert not (library.gizmos / "haze.gizmo").exists()
        assert f.locator(".st-grid .st-tile").count() == 2

        # Models: this computer, Radiance's folder made, what each workflow needs.
        f.locator('#stNav [data-go="models"]').click()
        f.locator(".st-setup-row").first.wait_for()
        assert (library.models / "radiance").is_dir()
        txt = f.locator(".st-page").inner_text()
        assert "NVIDIA GeForce RTX 4090" in txt and "812 GB" in txt
        assert "Runs every local workflow" in txt
        row = f.locator(".st-setup-row", has_text="SDR → HDR").filter(has_not_text="Analytic").filter(has_not_text="AI Fill").first
        assert "1 to download" in row.inner_text()
        # First-start checks: the server's four and the browser's display check, each with a status.
        checks = f.locator(".st-checks .st-check")
        assert checks.count() == 5
        assert [checks.nth(i).get_attribute("data-check") for i in range(5)] == ["ocio", "ffmpeg", "output", "hub", "display"]
        assert "Hugging Face access" in f.locator('.st-check[data-check="hub"]').inner_text()
        out_fix = f.locator('.st-check[data-check="output"] [data-act="setup-fix"]')
        if out_fix.count():
            out_fix.click()
            f.locator('.st-check[data-check="output"] .st-pill.ok, .st-check[data-check="output"] .st-pill.warn').wait_for()
            assert not f.locator('.st-check[data-check="output"] [data-act="setup-fix"]').count()
            assert (tmp_path / "radiance_output").is_dir()
        page.wait_for_timeout(300)
        _shot(page, "models")
        f.locator(".st-checks").scroll_into_view_if_needed()
        _shot(page, "models_checks")

        # Download what SDR → HDR needs: RUDRA ACES 2, into models/radiance.
        row.locator('[data-act="dl-workflow"]').click()
        f.locator(".st-setup-row.is-ready", has_text="SDR → HDR").filter(has_not_text="Analytic") \
            .filter(has_not_text="Remaster").first.wait_for(timeout=10000)
        assert "radiance/RUDRA_ACES_2" in library.packs
        assert (library.models / "radiance" / "RUDRA_ACES_2").is_dir()

        # A workflow file goes into the folder ComfyUI reads, pinned from the Hub.
        # T2I is the starter since 20 Nov (PR #107): FLUX.2 Klein 4B.
        start = f.locator(".st-setup-row", has_text="T2I")
        start.locator('[data-act="dl-workflow"]').click()
        f.locator(".st-setup-row.is-ready", has_text="T2I").wait_for(timeout=15000)
        for rel in ("diffusion_models/flux-2-klein-4b-fp8.safetensors", "text_encoders/qwen_3_4b.safetensors",
                    "vae/flux2-vae.safetensors"):
            assert (library.models / rel).is_file(), rel
        page.wait_for_timeout(300)
        _shot(page, "models_after")
        assert errors == []
        browser.close()
