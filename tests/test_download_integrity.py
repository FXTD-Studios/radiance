"""Downloads that skipped the integrity check until 4.0 (OPEN_QUESTIONS B17).

MoGe-2 is checked against its pinned SHA-256, each Marigold file against the
hash the Hub reports at the pinned commit, and the ACES config manager's
download goes through core.model_fetch (consent, .part file, pinned SHA-256).
"""
import hashlib
import io
import os
from types import SimpleNamespace
from unittest import mock

import pytest

huggingface_hub = pytest.importorskip("huggingface_hub")

from radiance.nodes.vfx.multipass import estimate_models as em  # noqa: E402


# ── MoGe-2 ──────────────────────────────────────────────────────────────────

def _fake_moge_download(repo_id, filename, revision, local_dir):
    p = os.path.join(local_dir, filename)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "wb") as fh:
        fh.truncate(em.MOGE_SIZE)
    return p


def test_a_moge_download_with_the_wrong_hash_is_deleted(tmp_path, monkeypatch):
    monkeypatch.setenv("RADIANCE_ALLOW_DOWNLOADS", "1")
    with mock.patch.object(em, "_geometry_dirs", return_value=[tmp_path]), \
         mock.patch.object(em, "_sha256", return_value="0" * 64), \
         mock.patch.object(huggingface_hub, "hf_hub_download", _fake_moge_download):
        with pytest.raises(em.EstimateModelError, match="CHECKSUM MISMATCH"):
            em.ensure_moge(True)
    assert list(tmp_path.iterdir()) == [], "an unverified file or its staging folder was kept"


# ── Marigold ────────────────────────────────────────────────────────────────

NAME = em.MARIGOLD_LIGHTING


def _git_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _install(target, contents):
    for rel, data in contents.items():
        p = target / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)


def _hub_infos(contents, lfs_files):
    infos = []
    for rel, data in contents.items():
        lfs = SimpleNamespace(sha256=hashlib.sha256(data).hexdigest()) if rel in lfs_files else None
        infos.append(SimpleNamespace(path=rel, blob_id=_git_blob(data), lfs=lfs))
    return infos


def _contents():
    return {rel: f"content of {rel}".encode() for rel in em.MARIGOLD_SOURCES[NAME]["files"]}


LFS = {f for f in em.MARIGOLD_SOURCES[NAME]["files"] if f.endswith(".safetensors")}


def test_marigold_files_matching_the_hub_hashes_pass(tmp_path):
    contents = _contents()
    _install(tmp_path, contents)
    with mock.patch.object(huggingface_hub.HfApi, "get_paths_info",
                           return_value=_hub_infos(contents, LFS)) as info:
        em._verify_marigold(tmp_path, NAME)
    assert info.call_args.kwargs["revision"] == em.MARIGOLD_SOURCES[NAME]["revision"]


@pytest.mark.parametrize("which", ["lfs", "small"])
def test_a_marigold_file_that_differs_is_deleted(tmp_path, which):
    contents = _contents()
    hub = _hub_infos(contents, LFS)
    rel = next(iter(LFS)) if which == "lfs" else "model_index.json"
    contents[rel] = b"tampered"
    _install(tmp_path, contents)
    with mock.patch.object(huggingface_hub.HfApi, "get_paths_info", return_value=hub):
        with pytest.raises(em.EstimateModelError, match=rel.replace(".", r"\.")):
            em._verify_marigold(tmp_path, NAME)
    assert not (tmp_path / rel).exists()
    assert all((tmp_path / r).exists() for r in contents if r != rel)


def test_marigold_verification_that_cannot_reach_the_hub_deletes_nothing(tmp_path):
    contents = _contents()
    _install(tmp_path, contents)
    with mock.patch.object(huggingface_hub.HfApi, "get_paths_info", side_effect=OSError("offline")):
        with pytest.raises(em.EstimateModelError, match="could not read the file hashes"):
            em._verify_marigold(tmp_path, NAME)
    assert all((tmp_path / r).exists() for r in contents)


def test_ensure_marigold_verifies_after_downloading(tmp_path, monkeypatch):
    monkeypatch.setenv("RADIANCE_ALLOW_DOWNLOADS", "1")
    contents = _contents()

    def fake_snapshot(repo_id, revision, allow_patterns, local_dir):
        _install(tmp_path, contents)

    with mock.patch.object(em, "_marigold_dirs", return_value=[tmp_path]), \
         mock.patch.object(huggingface_hub, "snapshot_download", fake_snapshot), \
         mock.patch.object(em, "_verify_marigold") as verify:
        assert em.ensure_marigold(NAME, True) == tmp_path
    verify.assert_called_once_with(tmp_path, NAME)


def test_a_marigold_download_whose_check_failed_is_checked_again_next_time(tmp_path, monkeypatch):
    # 4.0 beta: when the hash lookup failed (rate limit, network blip) the
    # files stayed and the next queue found them complete and used them
    # without ever checking them.
    monkeypatch.setenv("RADIANCE_ALLOW_DOWNLOADS", "1")
    contents = _contents()

    def fake_snapshot(repo_id, revision, allow_patterns, local_dir):
        _install(tmp_path, contents)

    with mock.patch.object(em, "_marigold_dirs", return_value=[tmp_path]), \
         mock.patch.object(huggingface_hub, "snapshot_download", fake_snapshot):
        with mock.patch.object(huggingface_hub.HfApi, "get_paths_info", side_effect=OSError("429")):
            with pytest.raises(em.EstimateModelError, match="could not read the file hashes"):
                em.ensure_marigold(NAME, True)
        assert em.find_marigold(NAME) is None
        with mock.patch.object(huggingface_hub, "snapshot_download") as again, \
             mock.patch.object(huggingface_hub.HfApi, "get_paths_info",
                               return_value=_hub_infos(contents, LFS)) as info:
            assert em.ensure_marigold(NAME, True) == tmp_path
        again.assert_not_called()          # the files are there, only the check reruns
        info.assert_called_once()
        assert em.find_marigold(NAME) == tmp_path


def test_a_hand_installed_marigold_is_used_without_a_check(tmp_path):
    _install(tmp_path, _contents())
    with mock.patch.object(em, "_marigold_dirs", return_value=[tmp_path]), \
         mock.patch.object(huggingface_hub.HfApi, "get_paths_info") as info:
        assert em.ensure_marigold(NAME, False) == tmp_path
    info.assert_not_called()


def test_an_unchecked_marigold_with_downloads_off_says_how_to_proceed(tmp_path, monkeypatch):
    monkeypatch.setenv("RADIANCE_ALLOW_DOWNLOADS", "0")
    _install(tmp_path, _contents())
    (tmp_path / em.MARIGOLD_UNVERIFIED).write_text("")
    with mock.patch.object(em, "_marigold_dirs", return_value=[tmp_path]):
        with pytest.raises(em.EstimateModelError, match=em.MARIGOLD_UNVERIFIED.replace(".", r"\.")):
            em.ensure_marigold(NAME, False)


# ── ACES config manager ─────────────────────────────────────────────────────

def _aces():
    from radiance.hdr.ocio import ACESConfigManager
    return ACESConfigManager()


def test_the_aces_download_respects_download_consent(tmp_path, monkeypatch):
    monkeypatch.setenv("RADIANCE_ALLOW_DOWNLOADS", "0")
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: pytest.fail("downloaded without consent"))
    path, msg = _aces()._download_aces_config(str(tmp_path))
    assert path == "" and "downloads are off" in msg
    assert not (tmp_path / "config.ocio").exists()


class _Resp(io.BytesIO):
    status = 200
    headers = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_a_corrupt_aces_download_installs_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("RADIANCE_ALLOW_DOWNLOADS", "1")
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _Resp(b"x" * 100))
    path, msg = _aces()._download_aces_config(str(tmp_path))
    assert path == "" and "CHECKSUM MISMATCH" in msg
    assert os.listdir(tmp_path) == []


def test_a_verified_aces_download_is_installed(tmp_path, monkeypatch):
    mgr = _aces()
    payload = b"ocio_profile_version: 2.5\n"
    monkeypatch.setenv("RADIANCE_ALLOW_DOWNLOADS", "1")
    monkeypatch.setattr(type(mgr), "ACES2_CONFIG_SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(type(mgr), "ACES2_CONFIG_SIZE", len(payload))
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _Resp(payload))
    path, msg = mgr._download_aces_config(str(tmp_path))
    assert path == str(tmp_path / "config.ocio") and "Successfully" in msg
    assert (tmp_path / "config.ocio").read_bytes() == payload


# ── Whisper (B18) ───────────────────────────────────────────────────────────

def _whisper_cli(tmp_path, monkeypatch, allow, cached):
    from radiance.nodes.pipeline import audio
    monkeypatch.setenv("RADIANCE_ALLOW_DOWNLOADS", "1" if allow else "0")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setitem(__import__("sys").modules, "whisper", None)   # the CLI alone
    if cached:
        (tmp_path / "whisper").mkdir()
        (tmp_path / "whisper" / "base.pt").write_bytes(b"w")
    ran = []

    def fake_run(cmd, **kw):
        ran.append(cmd)
        raise FileNotFoundError

    monkeypatch.setattr(audio.subprocess, "run", fake_run)
    return audio, ran


def test_whisper_does_not_download_without_consent(tmp_path, monkeypatch):
    audio, ran = _whisper_cli(tmp_path, monkeypatch, allow=False, cached=False)
    with pytest.raises(RuntimeError, match="downloads are off"):
        audio._transcribe_whisper_cli("a.wav", "base", "auto")
    assert ran == []


def test_cached_whisper_weights_need_no_consent(tmp_path, monkeypatch):
    audio, ran = _whisper_cli(tmp_path, monkeypatch, allow=False, cached=True)
    audio._transcribe_whisper_cli("a.wav", "base", "auto")
    assert len(ran) == 1


def test_whisper_downloads_by_default(tmp_path, monkeypatch):
    audio, ran = _whisper_cli(tmp_path, monkeypatch, allow=True, cached=False)
    audio._transcribe_whisper_cli("a.wav", "base", "auto")
    assert len(ran) == 1


@pytest.mark.parametrize("size, cached_as", [("large", "large-v3.pt"), ("large-v3", "large-v3.pt"),
                                             ("medium", "medium.pt")])
def test_whisper_cli_finds_cached_weights_under_their_release_name(tmp_path, monkeypatch, size, cached_as):
    # whisper caches "large" as large-v3.pt; 4.0 beta looked for large.pt
    # when only the CLI was installed and refused a model it already had.
    audio, ran = _whisper_cli(tmp_path, monkeypatch, allow=False, cached=False)
    (tmp_path / "whisper").mkdir()
    (tmp_path / "whisper" / cached_as).write_bytes(b"w")
    audio._transcribe_whisper_cli("a.wav", size, "auto")
    assert len(ran) == 1
