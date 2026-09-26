"""First-use download of the pixel SDR-to-HDR checkpoint (no network, no torch)."""
from __future__ import annotations

import hashlib
import io
import logging

import pytest

from radiance.core import consent
from radiance.model import pixel_download as pd


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


@pytest.fixture
def env(monkeypatch, tmp_path):
    for k in ("RADIANCE_ALLOW_DOWNLOADS", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr("radiance.model.paths.radiance_model_dirs", lambda: [tmp_path / "radiance"])
    pd._reset_for_tests()
    yield tmp_path / "radiance"
    pd._reset_for_tests()


def _serve(monkeypatch, payload: bytes, calls: list):
    def fake_urlopen(req, timeout=None):
        calls.append(req.full_url)
        return _Resp(payload)
    monkeypatch.setattr(pd.urllib.request, "urlopen", fake_urlopen)


def _pin(monkeypatch, payload: bytes):
    monkeypatch.setattr(pd, "PIXEL_SIZE", len(payload))
    monkeypatch.setattr(pd, "PIXEL_SHA256", hashlib.sha256(payload).hexdigest())


def test_url_is_pinned_to_a_commit():
    assert pd.PIXEL_REVISION in pd.PIXEL_URL
    assert pd.PIXEL_URL.endswith("/" + pd.PIXEL_FILENAME)
    assert len(pd.PIXEL_REVISION) == 40 and len(pd.PIXEL_SHA256) == 64


def test_consent_default_is_download_unless_a_caller_asks_first(env):
    """3.5.0: models download on first use; a caller can still ask first."""
    assert consent.downloads_allowed() is True
    assert consent.downloads_allowed(default=False) is False


@pytest.mark.parametrize("var,val", [
    ("RADIANCE_ALLOW_DOWNLOADS", "0"), ("HF_HUB_OFFLINE", "1"), ("TRANSFORMERS_OFFLINE", "1"),
])
def test_opt_out_is_respected(env, monkeypatch, var, val, caplog):
    monkeypatch.setenv(var, val)
    calls = []
    _serve(monkeypatch, b"x", calls)
    with caplog.at_level(logging.WARNING, logger="radiance.model.pixel_download"):
        assert pd.download_pixel_checkpoint() is None
        assert pd.download_pixel_checkpoint() is None
    assert calls == []
    assert sum("automatic downloads are off" in r.getMessage() for r in caplog.records) == 1


def test_explicit_allow_beats_offline_flag(env, monkeypatch):
    monkeypatch.setenv("RADIANCE_ALLOW_DOWNLOADS", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    payload = b"weights" * 10
    _pin(monkeypatch, payload)
    calls = []
    _serve(monkeypatch, payload, calls)
    assert pd.download_pixel_checkpoint() is not None


def test_download_lands_verified_in_models_dir(env, monkeypatch):
    payload = b"radiance-pixel" * 100
    _pin(monkeypatch, payload)
    calls = []
    _serve(monkeypatch, payload, calls)
    path = pd.download_pixel_checkpoint()
    assert path == env / pd.PIXEL_FILENAME
    assert path.read_bytes() == payload
    assert not (env / (pd.PIXEL_FILENAME + ".part")).exists()
    assert calls == [pd.PIXEL_URL]
    # Already installed: no second fetch.
    assert pd.download_pixel_checkpoint() == path
    assert len(calls) == 1


@pytest.mark.parametrize("served", [b"tampered-bytes!", b"short"])
def test_bad_file_is_refused_and_not_installed(env, monkeypatch, served):
    good = b"tampered-bytes?"
    _pin(monkeypatch, good)
    calls = []
    _serve(monkeypatch, served, calls)
    assert pd.download_pixel_checkpoint() is None
    assert not (env / pd.PIXEL_FILENAME).exists()
    assert not (env / (pd.PIXEL_FILENAME + ".part")).exists()
    # One attempt per process after a failure.
    assert pd.download_pixel_checkpoint() is None
    assert len(calls) == 1


def test_network_error_is_reported_not_raised(env, monkeypatch, caplog):
    def boom(req, timeout=None):
        raise OSError("no route to host")
    monkeypatch.setattr(pd.urllib.request, "urlopen", boom)
    with caplog.at_level(logging.ERROR, logger="radiance.model.pixel_download"):
        assert pd.download_pixel_checkpoint() is None
    assert any(pd.PIXEL_PAGE_URL in r.getMessage() for r in caplog.records)


@pytest.mark.real_torch
def test_resolver_falls_back_to_download(env, monkeypatch):
    px = pytest.importorskip("radiance.pixel_sdr2hdr")
    target = env / pd.PIXEL_FILENAME
    monkeypatch.setattr(pd, "download_pixel_checkpoint", lambda **k: target)
    assert px.resolve_pixel_checkpoint("") == target
    installed = env / "sdr2hdr_pixel_image.pt"
    env.mkdir(parents=True, exist_ok=True)
    installed.write_bytes(b"x")
    monkeypatch.setattr(pd, "download_pixel_checkpoint", lambda **k: pytest.fail("no fetch"))
    assert px.resolve_pixel_checkpoint("") == installed.resolve()
