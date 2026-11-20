"""The setup check: every workflow's models, this machine, and the downloads.

The catalog is built from the shipped workflows (tools/build_model_catalog.py)
and must match them; the scan reads the models folders; the queue downloads
one file at a time through core/model_fetch.fetch, pinned to a commit and a
SHA-256, into the folder ComfyUI reads, and Radiance's own models through their
own downloaders into models/radiance. Nothing here touches the network: the
Hub's metadata and the transfers are replaced by fakes.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from radiance.core import setup_check as sc

ROOT = Path(__file__).resolve().parents[1]


def _tool():
    import importlib.util
    spec = importlib.util.spec_from_file_location("build_model_catalog", ROOT / "tools" / "build_model_catalog.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── The catalog ─────────────────────────────────────────────────────────────

def test_the_catalog_matches_the_shipped_workflows():
    assert _tool().main(["--check"]) == 0, "run python tools/build_model_catalog.py"


def test_every_local_workflow_is_in_the_catalog_with_a_source_for_each_model():
    cat = sc.load_catalog()
    local = {p.stem for p in (ROOT / "workflows").glob("*.json") if not p.name.startswith("api_")}
    assert set(cat["workflows"]) == local
    for stem, w in cat["workflows"].items():
        for mid in w["required"] + w["optional"]:
            if sc.is_pack(mid):
                assert mid in cat["radiance_packs"], (stem, mid)
                continue
            m = cat["models"][mid]
            assert m["url"].startswith("https://huggingface.co/") and "/resolve/" in m["url"], mid
            assert m["folder"] and m["file"] == mid.rsplit("/", 1)[1], mid


def test_the_catalog_knows_what_each_workflow_loads():
    w = sc.load_catalog()["workflows"]
    assert w["sdr_to_hdr"]["required"] == ["radiance/RUDRA_ACES_2"]
    assert w["sdr_to_hdr_analytic"]["required"] == []                       # no model at all
    assert set(w["multipass_relight"]["required"]) == {"radiance/moge2", "radiance/marigold_appearance",
                                                      "radiance/marigold_lighting"}
    assert {"radiance/rudra_genfill_v3", "radiance/wan_vace_base"} <= set(w["sdr_remaster_ai_fill"]["required"])
    assert "radiance/wan_vace_text_encoder" in w["sdr_remaster_ai_fill"]["optional"]
    # The Starter builds its HDR master with RUDRA ACES 2; its depth map is optional.
    assert {"radiance/RUDRA_ACES_2", "diffusion_models/flux-2-klein-4b-fp8.safetensors"} <= set(w["t2i_i2i_hdr"]["required"])
    assert "radiance/depth_anything_v2_large" in w["t2i_i2i_hdr"]["optional"]
    # Wan 2.2's loader pairs the low-noise expert itself.
    assert "diffusion_models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors" in w["wan_hdr"]["required"]
    # A bypassed node's model is optional, an alternative is not required.
    assert "loras/minimax_h3_fl2v_turbo_8step_v1.1_768p_comfyui_bf16.safetensors" in w["minimax_h3_hdr"]["optional"]
    assert "diffusion_models/minimax_h3_fl2va_bf16.safetensors" in w["minimax_h3_hdr"]["optional"]
    assert "radiance/rudra_genfill_v3" in w["t2i_i2i_hdr"]["optional"]
    gated = sc.load_catalog()["models"]["diffusion_models/flux3-image-klein-base-bf16.safetensors"]
    assert gated["gated"] is True


# ── The scan ────────────────────────────────────────────────────────────────

@pytest.fixture
def models(tmp_path, monkeypatch):
    root = tmp_path / "models"
    root.mkdir()
    monkeypatch.setattr(sc, "models_dir", lambda: root)
    monkeypatch.setattr(sc, "_folder_dirs", lambda folder: [root / folder])
    installed = set()
    monkeypatch.setattr(sc, "pack_installed", lambda mid: mid in installed)
    monkeypatch.setattr(sc, "hardware", lambda: {"gpus": [{"name": "NVIDIA GeForce RTX 4070", "kind": "cuda",
                                                         "vram": 12_000_000_000}], "disk_free": 10**12})
    monkeypatch.delenv("RADIANCE_ALLOW_DOWNLOADS", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    return type("M", (), {"root": root, "packs": installed})


def test_scan_creates_the_radiance_folder_and_says_what_is_missing(models):
    result = sc.scan(create_radiance_dir=True)
    assert (models.root / "radiance").is_dir()
    rows = {w["id"]: w for w in result["workflows"]}
    assert rows["sdr_to_hdr_analytic"]["ready"] is True
    assert rows["sdr_to_hdr"]["missing"] == ["radiance/RUDRA_ACES_2"]
    assert result["core"][0]["id"] == "radiance/RUDRA_ACES_2"
    assert result["summary"]["workflows"] == len(rows)

    models.packs.add("radiance/RUDRA_ACES_2")
    vae = models.root / "vae" / "flux2-vae.safetensors"
    vae.parent.mkdir(parents=True)
    vae.write_bytes(b"x" * 10)
    result = sc.scan()
    rows = {w["id"]: w for w in result["workflows"]}
    assert rows["sdr_to_hdr"]["ready"] is True
    assert "vae/flux2-vae.safetensors" not in rows["t2i_i2i_hdr"]["missing"]
    assert result["models"]["vae/flux2-vae.safetensors"]["installed"] is True
    assert "vae/flux2-vae.safetensors" in rows["t2i_i2i_hdr"]["required"]


def test_the_ai_fill_says_it_needs_a_16_gb_gpu(models):
    rows = {w["id"]: w for w in sc.scan()["workflows"]}
    assert "16 GB" in rows["sdr_remaster_ai_fill"]["gpu_note"]
    assert rows["sdr_to_hdr"]["gpu_note"] == ""


# ── Downloads ───────────────────────────────────────────────────────────────

def _wait(q, mid, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        job = next((j for j in q.jobs() if j["id"] == mid), None)
        if job and job["status"] not in ("queued", "running"):
            return job
        time.sleep(0.02)
    raise AssertionError(f"{mid} did not finish")


def test_a_file_is_pinned_from_the_hub_then_fetched_into_its_folder(models):
    calls = {}

    def meta(url):
        calls["meta"] = url
        return {"revision": "a" * 40, "sha256": "b" * 64, "size": 11}

    def fetch(url, dest, *, sha256, size, label, progress, should_cancel):
        calls["fetch"] = (url, dest, sha256, size)
        progress(5, 11)
        Path(dest).write_bytes(b"x" * 11)
        progress(11, 11)
        return dest

    q = sc.DownloadQueue(fetcher=fetch, metadata=meta)
    mid = "text_encoders/qwen_3_4b.safetensors"
    assert q.enqueue([mid]) == [mid]
    job = _wait(q, mid)
    assert job["status"] == "done", job
    url, dest, sha, size = calls["fetch"]
    assert url == ("https://huggingface.co/Comfy-Org/flux2-klein/resolve/" + "a" * 40 +
                   "/split_files/text_encoders/qwen_3_4b.safetensors")
    assert Path(dest) == models.root / "text_encoders" / "qwen_3_4b.safetensors"
    assert (sha, size) == ("b" * 64, 11)
    assert job["done"] == job["total"] == 11
    assert q.enqueue([mid]) == []                                     # installed: nothing to do


def test_a_radiance_model_goes_through_its_own_downloader(models):
    got = []
    q = sc.DownloadQueue(pack_downloader=lambda mid, job: got.append(mid))
    q.enqueue(["radiance/RUDRA_ACES_2"])
    assert _wait(q, "radiance/RUDRA_ACES_2")["status"] == "done"
    assert got == ["radiance/RUDRA_ACES_2"]


def test_a_failure_is_reported_and_unknown_ids_are_refused(models):
    def meta(url):
        from radiance.core.model_fetch import ModelFetchError
        raise ModelFetchError("The repository is gated (HTTP 401).")
    q = sc.DownloadQueue(fetcher=lambda *a, **k: None, metadata=meta)
    mid = "diffusion_models/flux3-image-klein-base-bf16.safetensors"
    q.enqueue([mid])
    job = _wait(q, mid)
    assert job["status"] == "failed" and "gated" in job["error"]
    with pytest.raises(KeyError):
        q.enqueue(["https://evil.example/model.safetensors"])
    with pytest.raises(KeyError):
        q.enqueue(["../../etc/passwd"])


def test_not_enough_disk_stops_before_downloading(models, monkeypatch):
    monkeypatch.setattr(sc, "_disk_free", lambda p: 1000)
    fetched = []
    q = sc.DownloadQueue(fetcher=lambda *a, **k: fetched.append(1),
                         metadata=lambda url: {"revision": "a" * 40, "sha256": "b" * 64, "size": 10**9})
    mid = "vae/flux2-vae.safetensors"
    q.enqueue([mid])
    job = _wait(q, mid)
    assert job["status"] == "failed" and "disk space" in job["error"] and not fetched


def test_cancel_stops_a_running_download(models):
    import threading
    started = threading.Event()

    def fetch(url, dest, *, should_cancel, **kw):
        started.set()
        while not should_cancel():
            time.sleep(0.01)
        from radiance.core.model_fetch import ModelFetchError
        raise ModelFetchError("download cancelled")

    q = sc.DownloadQueue(fetcher=fetch, metadata=lambda url: {"revision": "a" * 40, "sha256": "b" * 64, "size": 1})
    mid = "vae/flux2-vae.safetensors"
    q.enqueue([mid, "vae/qwen_image_vae.safetensors"])
    assert started.wait(5)
    assert q.cancel(mid) is True
    assert _wait(q, mid)["status"] == "cancelled"


def test_downloads_off_means_no_queue(models, monkeypatch):
    monkeypatch.setenv("RADIANCE_ALLOW_DOWNLOADS", "0")
    with pytest.raises(PermissionError):
        sc.DownloadQueue().enqueue(["radiance/RUDRA_ACES_2"])


def test_missing_for_a_workflow_lists_only_what_is_not_installed(models):
    models.packs.add("radiance/RUDRA_ACES_2")
    assert sc.missing_for("sdr_to_hdr") == []
    assert "radiance/rudra_genfill_v3" in sc.missing_for("sdr_remaster_ai_fill")
    assert "radiance/wan_vace_text_encoder" not in sc.missing_for("sdr_remaster_ai_fill")
    assert "radiance/wan_vace_text_encoder" in sc.missing_for("sdr_remaster_ai_fill", include_optional=True)


def test_hub_metadata_reads_the_lfs_digest_and_the_commit(monkeypatch):
    class Resp:
        headers = {"X-Repo-Commit": "c" * 40, "X-Linked-Etag": '"' + "d" * 64 + '"', "X-Linked-Size": "123"}

    class Opener:
        def open(self, req, timeout):
            assert req.get_method() == "HEAD"
            return Resp()
    monkeypatch.setattr(sc.urllib.request, "build_opener", lambda *a: Opener())
    assert sc.hub_metadata("https://huggingface.co/a/b/resolve/main/x.safetensors") == {
        "revision": "c" * 40, "sha256": "d" * 64, "size": 123}


def test_the_hardware_report_names_the_basics():
    hw = sc.hardware()
    for key in ("gpus", "python", "os", "disk_free", "radiance"):
        assert key in hw
    json.dumps(hw)                                                        # the route sends it as JSON


def test_fetch_reports_progress_and_can_be_cancelled(tmp_path, monkeypatch):
    """core/model_fetch.fetch: the two hooks the setup check uses."""
    import hashlib
    import io
    from radiance.core import model_fetch as mf
    payload = b"radiance" * 300_000
    digest = hashlib.sha256(payload).hexdigest()

    class Resp(io.BytesIO):
        status = 200
        headers = {"Content-Length": str(len(payload))}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    monkeypatch.setattr(mf.urllib.request, "urlopen", lambda *a, **k: Resp(payload))
    monkeypatch.delenv("RADIANCE_ALLOW_DOWNLOADS", raising=False)
    seen = []
    dest = tmp_path / "m.safetensors"
    mf.fetch("https://example.invalid/m", str(dest), sha256=digest, size=len(payload),
             progress=lambda d, t: seen.append((d, t)))
    assert dest.read_bytes() == payload and seen[-1] == (len(payload), len(payload))
    dest2 = tmp_path / "n.safetensors"
    with pytest.raises(mf.ModelFetchError, match="cancelled"):
        mf.fetch("https://example.invalid/m", str(dest2), sha256=digest, size=len(payload), should_cancel=lambda: True)
    assert not dest2.exists()


def test_a_file_listed_twice_keeps_the_link_the_workflow_uses():
    text = ("- vae: [flux2-vae](https://huggingface.co/a/gated/resolve/main/vae/flux2-vae.safetensors) 🔒\n"
            "- vae: [flux2-vae](https://huggingface.co/b/open/resolve/main/split_files/vae/flux2-vae.safetensors) (used)\n")
    link = _tool()._links(text)["flux2-vae.safetensors"]
    assert link["url"].startswith("https://huggingface.co/b/open/") and not link["gated"]

