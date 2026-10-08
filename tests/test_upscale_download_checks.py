"""Download consent and on-disk integrity for the upscale model files.

B13: RadianceAIUpscale._download_model ignored the legacy
RADIANCE_UPSCALE_OFFLINE=1 opt-out that the rest of the upscale pack honours.

B29: a model file already on disk was trusted when its size matched (or, in
nodes/upscale, whenever it existed), so a corrupt or swapped file with the
right name was loaded silently. fetch() now checks an existing file against
its pinned SHA-256 once and records the pass next to the file, so a large
file is not hashed again on every run.
"""
import hashlib
import io
import os
import sys
import types
import urllib.error

import pytest

from radiance.core import model_fetch as MF
from radiance.core.consent import ALLOW_ENV, LEGACY_UPSCALE_OFFLINE_ENV

PAYLOAD = b"radiance-weights-" * 4096
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()
URL = "https://example.invalid/weights.pth"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in (ALLOW_ENV, LEGACY_UPSCALE_OFFLINE_ENV, "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.delenv(var, raising=False)
    MF._VERIFIED.clear()
    yield
    MF._VERIFIED.clear()


class _Resp(io.BytesIO):
    status = 200

    def __init__(self, body):
        super().__init__(body)
        self.headers = {"Content-Length": str(len(body))}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


@pytest.fixture
def server(monkeypatch):
    """A fake urlopen serving PAYLOAD; records each request."""
    calls = []

    def _urlopen(req, timeout=None):
        calls.append(req.full_url)
        return _Resp(PAYLOAD)

    monkeypatch.setattr(MF.urllib.request, "urlopen", _urlopen)
    return calls


@pytest.fixture
def no_network(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("a download was attempted")
    monkeypatch.setattr(MF.urllib.request, "urlopen", _boom)


def _count_hashes(monkeypatch):
    count = []
    real = MF._sha256_of

    def _counting(path):
        count.append(path)
        return real(path)
    monkeypatch.setattr(MF, "_sha256_of", _counting)
    return count


# ── B29: fetch() verifies a file that is already on disk ────────────────────

class TestExistingFileIsVerified:

    def test_a_corrupt_file_of_the_right_size_is_downloaded_again(self, server, tmp_path):
        dest = tmp_path / "w.pth"
        dest.write_bytes(b"\0" * len(PAYLOAD))          # right size, wrong bytes
        assert MF.fetch(URL, str(dest), sha256=DIGEST, size=len(PAYLOAD)) == str(dest)
        assert dest.read_bytes() == PAYLOAD
        assert len(server) == 1

    def test_a_corrupt_file_without_a_known_size_is_downloaded_again(self, server, tmp_path):
        dest = tmp_path / "w.pth"
        dest.write_bytes(b"not the model")
        MF.fetch(URL, str(dest), sha256=DIGEST)
        assert dest.read_bytes() == PAYLOAD

    def test_with_downloads_off_a_corrupt_file_is_an_error_not_a_load(self, monkeypatch, no_network, tmp_path):
        monkeypatch.setenv(ALLOW_ENV, "0")
        dest = tmp_path / "w.pth"
        dest.write_bytes(b"\0" * len(PAYLOAD))
        with pytest.raises(MF.ModelFetchError) as e:
            MF.fetch(URL, str(dest), sha256=DIGEST, size=len(PAYLOAD), label="Real-ESRGAN x4+")
        msg = str(e.value)
        assert "does not match" in msg and DIGEST in msg
        assert str(dest) in msg and URL in msg and ALLOW_ENV in msg
        assert dest.exists(), "the user's file is reported, not deleted"

    def test_with_downloads_off_a_wrong_size_file_says_so(self, monkeypatch, no_network, tmp_path):
        monkeypatch.setenv(ALLOW_ENV, "0")
        dest = tmp_path / "w.pth"
        dest.write_bytes(b"truncated")
        with pytest.raises(MF.ModelFetchError, match="does not match"):
            MF.fetch(URL, str(dest), sha256=DIGEST, size=len(PAYLOAD))

    def test_the_legacy_upscale_flag_also_blocks_the_replacement(self, monkeypatch, no_network, tmp_path):
        monkeypatch.setenv(LEGACY_UPSCALE_OFFLINE_ENV, "1")
        dest = tmp_path / "w.pth"
        dest.write_bytes(b"\0" * len(PAYLOAD))
        with pytest.raises(MF.ModelFetchError, match="does not match"):
            MF.fetch(URL, str(dest), sha256=DIGEST, size=len(PAYLOAD),
                     legacy_offline_env=LEGACY_UPSCALE_OFFLINE_ENV)

    def test_a_good_file_is_hashed_once_then_trusted(self, monkeypatch, no_network, tmp_path):
        dest = tmp_path / "w.pth"
        dest.write_bytes(PAYLOAD)
        hashes = _count_hashes(monkeypatch)
        for _ in range(3):
            assert MF.fetch(URL, str(dest), sha256=DIGEST, size=len(PAYLOAD)) == str(dest)
        assert len(hashes) == 1

    def test_the_pass_survives_a_restart(self, monkeypatch, no_network, tmp_path):
        dest = tmp_path / "w.pth"
        dest.write_bytes(PAYLOAD)
        MF.fetch(URL, str(dest), sha256=DIGEST, size=len(PAYLOAD))
        MF._VERIFIED.clear()                          # a new ComfyUI process
        hashes = _count_hashes(monkeypatch)
        MF.fetch(URL, str(dest), sha256=DIGEST, size=len(PAYLOAD))
        assert hashes == [], "the recorded pass is reused, the file is not hashed again"

    def test_a_file_changed_after_the_pass_is_checked_again(self, monkeypatch, server, tmp_path):
        dest = tmp_path / "w.pth"
        dest.write_bytes(PAYLOAD)
        MF.fetch(URL, str(dest), sha256=DIGEST, size=len(PAYLOAD))
        MF._VERIFIED.clear()
        dest.write_bytes(b"\1" * len(PAYLOAD))       # swapped in place, same size
        st = os.stat(dest)
        os.utime(dest, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
        MF.fetch(URL, str(dest), sha256=DIGEST, size=len(PAYLOAD))
        assert dest.read_bytes() == PAYLOAD and len(server) == 1

    def test_a_pass_recorded_for_another_digest_does_not_count(self, monkeypatch, server, tmp_path):
        dest = tmp_path / "w.pth"
        dest.write_bytes(b"\1" * len(PAYLOAD))
        other = hashlib.sha256(dest.read_bytes()).hexdigest()
        # The file passes against the digest it was pinned to before...
        MF.fetch(URL, str(dest), sha256=other, size=len(PAYLOAD))
        MF._VERIFIED.clear()
        # ...and the pin moves: the old record must not vouch for it.
        MF.fetch(URL, str(dest), sha256=DIGEST, size=len(PAYLOAD))
        assert dest.read_bytes() == PAYLOAD

    def test_a_fresh_download_records_its_pass(self, monkeypatch, server, tmp_path):
        dest = tmp_path / "w.pth"
        MF.fetch(URL, str(dest), sha256=DIGEST, size=len(PAYLOAD))
        MF._VERIFIED.clear()
        hashes = _count_hashes(monkeypatch)
        MF.fetch(URL, str(dest), sha256=DIGEST, size=len(PAYLOAD))
        assert hashes == [] and len(server) == 1

    def test_a_read_only_folder_still_verifies(self, monkeypatch, no_network, tmp_path):
        dest = tmp_path / "w.pth"
        dest.write_bytes(PAYLOAD)

        def _deny(*a, **k):
            raise PermissionError("read-only")
        monkeypatch.setattr(MF, "_write_record", _deny)
        hashes = _count_hashes(monkeypatch)
        MF.fetch(URL, str(dest), sha256=DIGEST, size=len(PAYLOAD))
        MF.fetch(URL, str(dest), sha256=DIGEST, size=len(PAYLOAD))
        assert len(hashes) == 1, "the in-process record still saves the second hash"


# ── B29: nodes/upscale goes through the same check ──────────────────────────

@pytest.mark.real_torch
class TestUpscaleRegistryFiles:

    def _entry(self, monkeypatch, tmp_path):
        from radiance.nodes.upscale import upscale as up
        monkeypatch.setattr(up, "_get_models_dir", lambda sub: str(tmp_path))
        key = "realesrgan_x4plus"
        info = dict(up._UPSCALE_MODEL_REGISTRY[key], url=URL, sha256=DIGEST, size=len(PAYLOAD))
        monkeypatch.setitem(up._UPSCALE_MODEL_REGISTRY, key, info)
        return up, key, tmp_path / info["filename"]

    def test_a_corrupt_installed_file_is_not_returned(self, monkeypatch, no_network, tmp_path):
        monkeypatch.setenv(ALLOW_ENV, "0")
        up, key, dest = self._entry(monkeypatch, tmp_path)
        dest.write_bytes(b"\0" * len(PAYLOAD))
        assert up._download_upscale_model(key) is None

    def test_a_corrupt_installed_file_is_replaced(self, monkeypatch, server, tmp_path):
        up, key, dest = self._entry(monkeypatch, tmp_path)
        dest.write_bytes(b"\0" * len(PAYLOAD))
        assert up._download_upscale_model(key) == str(dest)
        assert dest.read_bytes() == PAYLOAD

    def test_a_good_installed_file_is_returned_without_a_download(self, monkeypatch, no_network, tmp_path):
        monkeypatch.setenv(ALLOW_ENV, "0")
        up, key, dest = self._entry(monkeypatch, tmp_path)
        dest.write_bytes(PAYLOAD)
        assert up._download_upscale_model(key) == str(dest)

    def test_a_hand_installed_model_with_no_pin_is_still_used(self, monkeypatch, tmp_path):
        from radiance.nodes.upscale import upscale as up
        monkeypatch.setattr(up, "_get_models_dir", lambda sub: str(tmp_path))
        dest = tmp_path / up._UPSCALE_MODEL_REGISTRY["hat_l_x4"]["filename"]
        dest.write_bytes(b"hat weights")
        assert up._download_upscale_model("hat_l_x4") == str(dest)

    def test_the_unused_duplicate_checker_is_gone(self):
        from radiance.nodes.upscale import upscale as up
        assert not hasattr(up, "_verify_or_report_sha256")


# ── B13: RadianceAIUpscale honours RADIANCE_UPSCALE_OFFLINE ─────────────────

@pytest.mark.real_torch
class TestAIUpscaleHonoursTheLegacyOptOut:

    def _patch_comfy(self, monkeypatch, tmp_path):
        fp = types.ModuleType("folder_paths")
        fp.get_full_path = lambda *a, **k: None
        fp.get_folder_paths = lambda *a, **k: [str(tmp_path)]
        cu = types.ModuleType("comfy.utils")
        monkeypatch.setitem(sys.modules, "folder_paths", fp)
        monkeypatch.setitem(sys.modules, "comfy.utils", cu)
        comfy = sys.modules.get("comfy") or types.ModuleType("comfy")
        monkeypatch.setattr(comfy, "utils", cu, raising=False)
        monkeypatch.setitem(sys.modules, "comfy", comfy)

    def test_download_model_refuses(self, monkeypatch, no_network, tmp_path):
        from radiance.image import upscale as ai
        monkeypatch.setenv(LEGACY_UPSCALE_OFFLINE_ENV, "1")
        node = ai.RadianceAIUpscale()
        assert node._download_model("RealESRGAN_x4plus", str(tmp_path / "x.pth")) is False
        assert not (tmp_path / "x.pth").exists()

    def test_load_model_says_why(self, monkeypatch, no_network, tmp_path):
        from radiance.image import upscale as ai
        self._patch_comfy(monkeypatch, tmp_path)
        if hasattr(ai._MODEL_CACHE, "clear"):
            ai._MODEL_CACHE.clear()
        monkeypatch.setenv(LEGACY_UPSCALE_OFFLINE_ENV, "1")
        model, info = ai.RadianceAIUpscale()._load_model("RealESRGAN_x4plus", auto_download=True)
        assert model is None
        assert "automatic downloads are off" in info
        assert "RealESRGAN_x4plus" in info and str(tmp_path) in info

    def test_explicit_consent_beats_the_legacy_flag(self, monkeypatch, tmp_path):
        from radiance.image import upscale as ai
        monkeypatch.setenv(LEGACY_UPSCALE_OFFLINE_ENV, "1")
        monkeypatch.setenv(ALLOW_ENV, "1")

        def _offline(*a, **k):
            raise urllib.error.URLError("no network in tests")
        monkeypatch.setattr(MF.urllib.request, "urlopen", _offline)
        node = ai.RadianceAIUpscale()
        # Consent given, so the fetch is attempted (and fails offline).
        assert node._download_model("RealESRGAN_x4plus", str(tmp_path / "x.pth")) is False
        assert "download failed" in (node._last_download_error or "")
