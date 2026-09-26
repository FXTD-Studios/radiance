"""Every model a node downloads is pinned, and the shared fetch keeps its
promises (3.5.0). No network: the live check is tools/pin_models.py."""
import hashlib
import http.server
import importlib.util
import os
import re
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HEX64 = re.compile(r"[0-9a-f]{64}")
HF_PINNED = re.compile(r"https://huggingface\.co/[^/]+/[^/]+/resolve/[0-9a-f]{40}/.+")


def _pin_tool():
    spec = importlib.util.spec_from_file_location("pin_models", ROOT / "tools" / "pin_models.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_single_file_download_is_pinned():
    files, repos = _pin_tool().collect()
    assert len(files) >= 70 and len(repos) >= 5
    for label, url, sha, size in files:
        assert sha and HEX64.fullmatch(sha), label
        assert isinstance(size, int) and size > 0, label
        if url.startswith("https://huggingface.co/"):
            assert HF_PINNED.fullmatch(url), f"{label}: not pinned to a commit: {url}"
    for label, repo, rev in repos:
        assert re.fullmatch(r"[0-9a-f]{40}", rev), label


def test_read_models_catalogue_flags_gated_repositories():
    from radiance.config.model_map import RADIANCE_MODEL_MAP as M
    gated = {k for k, v in M.items() if v.get("gated")}
    assert "flux2-dev.safetensors" in gated
    assert "ae.safetensors" not in gated, "the FLUX.1 VAE comes from an ungated repackage"
    assert "black-forest-labs/FLUX.1-dev" not in M["ae.safetensors"]["url"]


def test_hat_has_no_download_and_says_where_to_get_it():
    pytest.importorskip("torch")
    from radiance.nodes.upscale import upscale as up
    for key in ("hat_l_x4", "hat_l_x2"):
        e = up._UPSCALE_MODEL_REGISTRY[key]
        assert e["url"] is None and e["manual_url"].startswith("https://github.com/XPixelGroup/HAT")


def test_depth_and_upscaler_pipelines_load_a_pinned_commit():
    pytest.importorskip("torch")
    from radiance.nodes.vfx import depth
    from radiance.nodes.upscale import upscale as up
    assert set(depth.DEPTH_REVISIONS) == set(depth.DEPTH_MODELS.values())
    assert re.fullmatch(r"[0-9a-f]{40}", up._SD_X4_REVISION)


def test_downloads_are_on_unless_turned_off(monkeypatch):
    from radiance.core import consent
    for var in ("RADIANCE_ALLOW_DOWNLOADS", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.delenv(var, raising=False)
    assert consent.downloads_allowed() is True
    monkeypatch.setenv("RADIANCE_ALLOW_DOWNLOADS", "0")
    assert consent.downloads_allowed() is False


# ── fetch() against a real local HTTP server ────────────────────────────────

PAYLOAD = os.urandom(3 * 1024 * 1024 + 17)
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()


class _Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        rng = self.headers.get("Range")
        start = int(rng.split("=")[1].rstrip("-")) if rng else 0
        body = PAYLOAD[start:]
        self.send_response(206 if rng else 200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self.server.ranges.append(rng)


@pytest.fixture
def server(monkeypatch):
    for var in ("RADIANCE_ALLOW_DOWNLOADS", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.delenv(var, raising=False)
    srv = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    srv.ranges = []
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv
    srv.shutdown()


def test_fetch_downloads_verifies_and_resumes(server, tmp_path):
    from radiance.core.model_fetch import fetch
    url = f"http://127.0.0.1:{server.server_port}/m.bin"
    dest = tmp_path / "m.bin"
    (tmp_path / "m.bin.part").write_bytes(PAYLOAD[:1_000_000])     # an interrupted run
    assert fetch(url, str(dest), sha256=DIGEST, size=len(PAYLOAD)) == str(dest)
    assert dest.read_bytes() == PAYLOAD and not (tmp_path / "m.bin.part").exists()
    assert server.ranges == ["bytes=1000000-"]
    # Installed and of the right size: no second request.
    fetch(url, str(dest), sha256=DIGEST, size=len(PAYLOAD))
    assert len(server.ranges) == 1


def test_fetch_never_installs_a_file_that_does_not_match(server, tmp_path):
    from radiance.core.model_fetch import ModelFetchError, fetch
    url = f"http://127.0.0.1:{server.server_port}/m.bin"
    with pytest.raises(ModelFetchError, match="CHECKSUM MISMATCH"):
        fetch(url, str(tmp_path / "m.bin"), sha256="0" * 64, size=len(PAYLOAD))
    assert not list(tmp_path.iterdir())
