"""Behaviour tests for ``radiance/loader_utils.py``.

Every test here asserts on a *real* return value, a real branch outcome or a
real error message.  ComfyUI (`comfy`, `folder_paths`) is not installed, so the
module-level references inside ``radiance.loader_utils`` (and, where the real
helper is exercised, ``radiance.model.detect``) are replaced per-test with
``monkeypatch.setattr`` — fixture-scoped, restored automatically, never leaked
to another test file.
"""
from __future__ import annotations

import hashlib
import logging
import os
import types

import pytest
import torch

import radiance.loader_utils as L
from radiance.config.model_map import RADIANCE_MODEL_MAP, CHECKPOINT_PRESETS
from radiance.model.cache import (
    _unet_cache, _clip_cache, _vae_cache, _audio_vae_cache,
)
from radiance.model.detect import _BASE_VRAM, _DTYPE_MULT, estimate_vram_usage

# ── Import guard (rule 4): a silently-degraded import must fail loudly here,
#    never let a test below pass by accident. ────────────────────────────────
assert L.__file__.endswith("loader_utils.py"), L.__file__
for _required in (
    "_sha256_file", "_download_model", "ensure_model_exists",
    "get_available_vram", "get_total_vram", "file_fingerprint",
    "resolve_divider", "apply_checkpoint_preset", "resolve_architecture",
    "setup_offload_mode", "estimate_vram_for_load", "_require_baked_vae",
    "construct_audio_vae", "load_unet_and_baked_vae", "load_clip_stack",
    "load_standalone_vae", "apply_lora_stack",
):
    assert callable(getattr(L, _required)), f"loader_utils.{_required} missing"


# ─────────────────────────────────────────────────────────────────────────────
#  Fixtures / fakes
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _isolated_model_caches():
    """The four LRU caches are process-wide singletons — isolate every test."""
    caches = (_unet_cache, _clip_cache, _vae_cache, _audio_vae_cache)
    for c in caches:
        c.clear()
    yield
    for c in caches:
        c.clear()


@pytest.fixture
def loader_logs():
    """Capture records from the 'radiance.loader' logger.

    ``logging.getLogger("radiance").propagate`` is False (radiance/core/logging.py),
    so pytest's own caplog (a root handler) never sees these records.
    """
    logger = logging.getLogger("radiance.loader")
    records: list[logging.LogRecord] = []

    class _Collect(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Collect(level=logging.DEBUG)
    old_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)


def msgs(records) -> list[str]:
    return [r.getMessage() for r in records]


class FakeFolderPaths:
    """Stand-in for ComfyUI's ``folder_paths`` module."""

    def __init__(self, files=None, folders=None, has_names_and_paths=True,
                 filename_list_raises=0):
        self.files = dict(files or {})            # (folder_type, name) -> abs path
        # folder_type -> [dir, ...]  (a bare string is accepted for brevity)
        self.folders = {
            ft: ([d] if isinstance(d, str) else list(d))
            for ft, d in (folders or {}).items()
        }
        self.folder_names_and_paths = (
            {ft: (list(d), set()) for ft, d in self.folders.items()}
            if has_names_and_paths else {}
        )
        self.filename_list_calls: list[str] = []
        self._filename_list_raises = filename_list_raises
        self.full_path_calls: list[tuple] = []

    def get_full_path(self, folder_type, name):
        self.full_path_calls.append((folder_type, name))
        return self.files.get((folder_type, name))

    def get_folder_paths(self, folder_type):
        return self.folders.get(folder_type, [])

    def get_filename_list(self, folder_type):
        self.filename_list_calls.append(folder_type)
        if self._filename_list_raises > 0:
            self._filename_list_raises -= 1
            raise RuntimeError("comfy listing unavailable")
        return sorted(n for (ft, n) in self.files if ft == folder_type)


class FakeBar:
    def __init__(self, **kw):
        self.kwargs = kw
        self.total = None
        self.updates: list[float] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def update(self, n):
        self.updates.append(n)


class FakeTqdm:
    def __init__(self):
        self.bars: list[FakeBar] = []

    def tqdm(self, **kw):
        bar = FakeBar(**kw)
        self.bars.append(bar)
        return bar


class FakeModel:
    """UNET patcher stand-in — must accept ``cached_patcher_init`` assignment."""

    def __init__(self, tag="model"):
        self.tag = tag


def make_comfy(**overrides):
    """Build a fake ``comfy`` namespace with the attributes loader_utils uses."""
    sd = types.SimpleNamespace(
        load_diffusion_model=None,
        load_diffusion_model_state_dict=None,
        load_checkpoint_guess_config=None,
        load_state_dict_guess_config=None,
        load_checkpoint_vae_patcher="VAE_PATCHER_FN",
        load_clip=None,
        load_lora_for_models=None,
        VAE=None,
    )
    utils = types.SimpleNamespace(load_torch_file=None, state_dict_prefix_replace=None)
    # The real API: a module-level vram_state read by load_models_gpu().
    # (set_lowvram_mode never existed in ComfyUI; see setup_offload_mode.)
    import enum

    class _VRAMState(enum.Enum):
        NORMAL_VRAM = 3
        LOW_VRAM = 2
        NO_VRAM = 1

    mm = types.SimpleNamespace(VRAMState=_VRAMState, vram_state=_VRAMState.NORMAL_VRAM)
    comfy = types.SimpleNamespace(sd=sd, utils=utils, model_management=mm)
    for key, val in overrides.items():
        target, _, attr = key.partition("__")
        setattr(getattr(comfy, target), attr, val)
    return comfy


def write_file(path, data=b"weights"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return str(path)


# ─────────────────────────────────────────────────────────────────────────────
#  _sha256_file
# ─────────────────────────────────────────────────────────────────────────────

class TestSha256File:
    def test_digest_matches_hashlib(self, tmp_path):
        data = b"radiance" * 5000
        p = tmp_path / "blob.bin"
        p.write_bytes(data)
        assert L._sha256_file(str(p)) == hashlib.sha256(data).hexdigest()

    def test_small_chunk_size_gives_same_digest(self, tmp_path):
        data = os.urandom(4096)
        p = tmp_path / "blob.bin"
        p.write_bytes(data)
        assert L._sha256_file(str(p), chunk=7) == L._sha256_file(str(p))

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.bin"
        p.write_bytes(b"")
        assert L._sha256_file(str(p)) == hashlib.sha256(b"").hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
#  _download_model
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def dl_env(monkeypatch):
    """A fake server behind radiance.core.model_fetch (the downloader
    _download_model delegates to since 3.5.0) + folder_paths."""
    import io
    import radiance.core.model_fetch as MF
    state = types.SimpleNamespace(
        urls=[], payload=b"MODELDATA", raises=None, headers=[],
        fp=FakeFolderPaths(folders={"vae": "/models/vae"}),
    )

    class _Resp(io.BytesIO):
        def __init__(self, data, status=200):
            super().__init__(data)
            self.status = status
            self.headers = {"Content-Length": str(len(data))}
        def __enter__(self):
            return self
        def __exit__(self, *a):
            self.close()

    def fake_urlopen(req, timeout=None):
        state.urls.append(req.full_url)
        state.headers.append(dict(req.header_items()))
        if state.raises is not None:
            raise state.raises
        rng = req.get_header("Range")
        if rng:
            start = int(rng.split("=")[1].rstrip("-"))
            return _Resp(state.payload[start:], status=206)
        return _Resp(state.payload)

    monkeypatch.setattr(MF.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(L, "folder_paths", state.fp)
    monkeypatch.delenv("RADIANCE_ALLOW_DOWNLOADS", raising=False)
    monkeypatch.delenv("RADIANCE_LOADER_OFFLINE", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    return state


GOOD = hashlib.sha256(b"MODELDATA").hexdigest()


class TestDownloadModel:
    def test_success_moves_into_place(self, dl_env, tmp_path):
        target = tmp_path / "sub" / "model.safetensors"
        assert L._download_model("http://host/m.safetensors", str(target), "vae", GOOD) is True
        assert target.read_bytes() == b"MODELDATA"
        assert not (tmp_path / "sub" / "model.safetensors.part").exists()
        assert dl_env.urls == ["http://host/m.safetensors"]

    def test_an_unpinned_download_is_refused(self, dl_env, tmp_path, loader_logs):
        """3.5.0: nothing is downloaded without a SHA-256 to check it against."""
        assert L._download_model("http://host/m", str(tmp_path / "m.bin"), "vae") is False
        assert dl_env.urls == []
        assert any("no SHA-256 is pinned" in m for m in msgs(loader_logs))

    def test_sha256_pin_is_case_and_whitespace_insensitive(self, dl_env, tmp_path):
        digest = "  " + GOOD.upper() + "\n"
        assert L._download_model("http://host/u", str(tmp_path / "m.bin"), "vae", digest) is True

    def test_checksum_mismatch_discards_download(self, dl_env, tmp_path, loader_logs):
        target = tmp_path / "m.bin"
        bad = "0" * 64
        assert L._download_model("http://host/u", str(target), "vae", bad) is False
        assert not target.exists(), "corrupt download must not land at the real path"
        assert not (tmp_path / "m.bin.part").exists()
        errs = [m for m in msgs(loader_logs) if "CHECKSUM MISMATCH" in m]
        assert errs and bad in errs[0] and GOOD in errs[0]

    def test_a_preexisting_file_of_the_right_size_is_kept(self, dl_env, tmp_path):
        target = tmp_path / "m.bin"
        target.write_bytes(b"GOOD-OLD-MODEL")
        assert L._download_model("http://host/u", str(target), "vae", "0" * 64, size=len(b"GOOD-OLD-MODEL")) is True
        assert target.read_bytes() == b"GOOD-OLD-MODEL" and dl_env.urls == []

    def test_transport_failure_keeps_the_part_file_and_the_next_run_resumes(self, dl_env, tmp_path, loader_logs):
        target = tmp_path / "m.bin"
        (tmp_path / "m.bin.part").write_bytes(b"MODEL")         # an interrupted earlier run
        dl_env.raises = OSError("connection reset")
        assert L._download_model("http://host/x", str(target), "vae", GOOD) is False
        assert not target.exists() and (tmp_path / "m.bin.part").exists()
        assert any("connection reset" in m for m in msgs(loader_logs))
        dl_env.raises = None
        assert L._download_model("http://host/x", str(target), "vae", GOOD) is True
        assert dl_env.headers[-1].get("Range") == "bytes=5-"
        assert target.read_bytes() == b"MODELDATA"

    def test_hugging_face_requests_carry_the_token(self, dl_env, tmp_path, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf_test")
        url = "https://huggingface.co/org/repo/resolve/" + "a" * 40 + "/m.bin"
        assert L._download_model(url, str(tmp_path / "m.bin"), "vae", GOOD) is True
        assert dl_env.headers[-1].get("Authorization") == "Bearer hf_test"

    def test_a_gated_repository_says_how_to_get_access(self, dl_env, tmp_path, loader_logs, monkeypatch):
        import urllib.error
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
        dl_env.raises = urllib.error.HTTPError("u", 401, "Unauthorized", {}, None)
        url = "https://huggingface.co/black-forest-labs/FLUX.2-dev/resolve/" + "a" * 40 + "/m.bin"
        assert L._download_model(url, str(tmp_path / "m.bin"), "vae", GOOD) is False
        text = " ".join(msgs(loader_logs))
        assert "gated" in text and "https://huggingface.co/black-forest-labs/FLUX.2-dev" in text and "HF_TOKEN" in text

    def test_downloads_off_stops_before_any_request(self, dl_env, tmp_path, monkeypatch):
        monkeypatch.setenv("RADIANCE_ALLOW_DOWNLOADS", "0")
        assert L._download_model("http://host/u", str(tmp_path / "m.bin"), "vae", GOOD) is False
        assert dl_env.urls == []

    def test_refresh_failure_is_retried_once(self, dl_env, tmp_path):
        """A raising first get_filename_list is caught and retried (2 calls)."""
        dl_env.fp._filename_list_raises = 1
        assert L._download_model("http://host/u", str(tmp_path / "m.bin"), "vae", GOOD) is True
        assert dl_env.fp.filename_list_calls == ["vae", "vae"]

    def test_persistent_refresh_failure_should_still_report_success(self, dl_env, tmp_path):
        dl_env.fp._filename_list_raises = 2
        target = tmp_path / "m.bin"
        result = L._download_model("http://host/u", str(target), "vae", GOOD)
        assert target.read_bytes() == b"MODELDATA"   # the file really is there
        assert result is True


# ─────────────────────────────────────────────────────────────────────────────
#  ensure_model_exists
# ─────────────────────────────────────────────────────────────────────────────

class TestEnsureModelExists:
    @pytest.fixture(autouse=True)
    def _downloads_on(self, monkeypatch):
        # conftest turns downloads off for the whole suite; these tests mock
        # the downloader and need the 3.5.0 default (on).
        monkeypatch.delenv("RADIANCE_ALLOW_DOWNLOADS", raising=False)
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    @pytest.mark.parametrize("name", ["", None, "None"])
    def test_empty_selection_returns_none_without_touching_comfy(self, monkeypatch, name):
        fp = FakeFolderPaths()
        monkeypatch.setattr(L, "folder_paths", fp)
        assert L.ensure_model_exists(name, "vae", auto_download=True) is None
        assert fp.full_path_calls == []

    def test_existing_file_is_returned(self, monkeypatch, tmp_path):
        real = write_file(tmp_path / "ae.safetensors")
        monkeypatch.setattr(L, "folder_paths",
                            FakeFolderPaths(files={("vae", "ae.safetensors"): real}))
        assert L.ensure_model_exists("ae.safetensors", "vae") == real

    def test_registered_but_missing_on_disk_without_download_is_none(self, monkeypatch, tmp_path):
        ghost = str(tmp_path / "gone.safetensors")
        monkeypatch.setattr(L, "folder_paths",
                            FakeFolderPaths(files={("vae", "gone.safetensors"): ghost}))
        assert L.ensure_model_exists("gone.safetensors", "vae") is None

    def test_unknown_name_is_not_downloaded(self, monkeypatch):
        calls = []
        monkeypatch.setattr(L, "folder_paths", FakeFolderPaths())
        monkeypatch.setattr(L, "_download_model", lambda *a, **k: calls.append(a) or True)
        assert L.ensure_model_exists("not-in-the-map.safetensors", "vae",
                                     auto_download=True) is None
        assert calls == []

    def test_auto_download_off_never_downloads_a_known_model(self, monkeypatch):
        calls = []
        monkeypatch.setattr(L, "folder_paths", FakeFolderPaths())
        monkeypatch.setattr(L, "_download_model", lambda *a, **k: calls.append(a) or True)
        assert L.ensure_model_exists("flux1-schnell-fp8.safetensors",
                                     "diffusion_models", auto_download=False) is None
        assert calls == []

    def test_folder_type_mismatch_blocks_download(self, monkeypatch):
        """flux1-schnell is a diffusion_models entry — asking for it as a vae must not download."""
        calls = []
        monkeypatch.setattr(L, "folder_paths", FakeFolderPaths(folders={"vae": "/m/vae"}))
        monkeypatch.setattr(L, "_download_model", lambda *a, **k: calls.append(a) or True)
        assert RADIANCE_MODEL_MAP["flux1-schnell-fp8.safetensors"]["type"] == "diffusion_models"
        assert L.ensure_model_exists("flux1-schnell-fp8.safetensors", "vae",
                                     auto_download=True) is None
        assert calls == []

    def test_successful_download_returns_target_path(self, monkeypatch, tmp_path):
        monkeypatch.setitem(RADIANCE_MODEL_MAP, "sub/dir/test.safetensors",
                            {"url": "http://h/t.safetensors", "type": "vae",
                             "sha256": "abc123"})
        monkeypatch.setattr(L, "folder_paths",
                            FakeFolderPaths(folders={"vae": str(tmp_path)}))
        seen = {}

        def fake_dl(url, target_path, folder_type, expected_sha256=None, size=None):
            seen.update(url=url, target=target_path, ft=folder_type, sha=expected_sha256)
            return True

        monkeypatch.setattr(L, "_download_model", fake_dl)
        out = L.ensure_model_exists("sub/dir/test.safetensors", "vae", auto_download=True)
        assert out == os.path.join(str(tmp_path), "sub", "dir", "test.safetensors")
        assert seen == {"url": "http://h/t.safetensors", "target": out,
                        "ft": "vae", "sha": "abc123"}

    def test_windows_style_separators_are_normalised(self, monkeypatch, tmp_path):
        monkeypatch.setitem(RADIANCE_MODEL_MAP, r"sub\win.safetensors",
                            {"url": "http://h/w", "type": "vae"})
        monkeypatch.setattr(L, "folder_paths",
                            FakeFolderPaths(folders={"vae": str(tmp_path)}))
        monkeypatch.setattr(L, "_download_model", lambda *a, **k: True)
        assert L.ensure_model_exists(r"sub\win.safetensors", "vae", True) == \
            os.path.join(str(tmp_path), "sub", "win.safetensors")

    def test_failed_download_returns_none(self, monkeypatch, tmp_path):
        monkeypatch.setitem(RADIANCE_MODEL_MAP, "t.safetensors",
                            {"url": "http://h/t", "type": "vae"})
        monkeypatch.setattr(L, "folder_paths",
                            FakeFolderPaths(folders={"vae": str(tmp_path)}))
        monkeypatch.setattr(L, "_download_model", lambda *a, **k: False)
        assert L.ensure_model_exists("t.safetensors", "vae", True) is None

    @pytest.mark.parametrize("flag", ["1", "true", "YES", " on ", "True"])
    def test_offline_mode_blocks_download(self, monkeypatch, tmp_path, flag, loader_logs):
        monkeypatch.setenv("RADIANCE_LOADER_OFFLINE", flag)
        monkeypatch.setitem(RADIANCE_MODEL_MAP, "t.safetensors",
                            {"url": "http://h/t", "type": "vae"})
        monkeypatch.setattr(L, "folder_paths",
                            FakeFolderPaths(folders={"vae": str(tmp_path)}))
        called = []
        monkeypatch.setattr(L, "_download_model", lambda *a, **k: called.append(a) or True)
        assert L.ensure_model_exists("t.safetensors", "vae", True) is None
        assert called == []
        assert any("RADIANCE_LOADER_OFFLINE" in m for m in msgs(loader_logs))

    @pytest.mark.parametrize("flag", ["0", "false", "off", ""])
    def test_non_offline_values_still_download(self, monkeypatch, tmp_path, flag):
        monkeypatch.setenv("RADIANCE_LOADER_OFFLINE", flag)
        monkeypatch.setitem(RADIANCE_MODEL_MAP, "t.safetensors",
                            {"url": "http://h/t", "type": "vae"})
        monkeypatch.setattr(L, "folder_paths",
                            FakeFolderPaths(folders={"vae": str(tmp_path)}))
        monkeypatch.setattr(L, "_download_model", lambda *a, **k: True)
        assert L.ensure_model_exists("t.safetensors", "vae", True) == \
            os.path.join(str(tmp_path), "t.safetensors")


# ─────────────────────────────────────────────────────────────────────────────
#  VRAM helpers / fingerprints / divider
# ─────────────────────────────────────────────────────────────────────────────

class TestVramHelpers:
    def test_available_vram_converts_bytes_to_gb(self, monkeypatch):
        seen = []
        monkeypatch.setattr(L, "get_torch_device", lambda: "cuda:3")
        monkeypatch.setattr(L, "_get_free_vram", lambda dev: seen.append(dev) or 6 * 1024 ** 3)
        assert L.get_available_vram() == 6.0
        assert seen == ["cuda:3"]

    def test_total_vram_converts_bytes_to_gb(self, monkeypatch):
        monkeypatch.setattr(L, "get_torch_device", lambda: "cuda:0")
        monkeypatch.setattr(L, "_get_total_vram", lambda dev: 3 * 1024 ** 3 + 512 * 1024 ** 2)
        assert L.get_total_vram() == pytest.approx(3.5)


class TestFileFingerprint:
    def test_encodes_mtime_and_size(self, tmp_path):
        p = tmp_path / "f.bin"
        p.write_bytes(b"12345")
        st = os.stat(p)
        assert L.file_fingerprint(str(p)) == f"{st.st_mtime:.0f}:{5}"

    def test_changes_when_content_size_changes(self, tmp_path):
        p = tmp_path / "f.bin"
        p.write_bytes(b"12345")
        first = L.file_fingerprint(str(p))
        p.write_bytes(b"1234567890")
        assert L.file_fingerprint(str(p)) != first
        assert L.file_fingerprint(str(p)).endswith(":10")

    def test_missing_file_is_nostat(self, tmp_path):
        assert L.file_fingerprint(str(tmp_path / "absent")) == "nostat"


class TestResolveDivider:
    def test_unicode_console(self, monkeypatch):
        monkeypatch.setattr(L, "supports_unicode", lambda: True)
        assert L.resolve_divider() == "│"

    def test_ascii_fallback(self, monkeypatch):
        monkeypatch.setattr(L, "supports_unicode", lambda: False)
        assert L.resolve_divider() == "|"


# ─────────────────────────────────────────────────────────────────────────────
#  apply_checkpoint_preset
# ─────────────────────────────────────────────────────────────────────────────

class TestApplyCheckpointPreset:
    def test_custom_preset_is_a_no_op(self):
        info: list[str] = []
        out = L.apply_checkpoint_preset("Custom", "sdxl", "fp16", "fp16", info)
        assert out == ("sdxl", "fp16", "fp16", [])
        assert info == []

    def test_unknown_preset_is_a_no_op(self):
        info: list[str] = []
        out = L.apply_checkpoint_preset("NotAPreset", "sdxl", "fp8_e4m3fn", "fp16", info)
        assert out == ("sdxl", "fp8_e4m3fn", "fp16", [])
        assert info == []

    def test_real_preset_overrides_and_records_transitions(self):
        info: list[str] = []
        cfg = CHECKPOINT_PRESETS["AuraFlow"]
        mt, wd, cd, overrides = L.apply_checkpoint_preset(
            "AuraFlow", "sdxl", "fp8_e4m3fn", "fp32", info)
        assert (mt, wd, cd) == (cfg["model_type"], cfg["weight_dtype"], cfg["clip_dtype"])
        assert (mt, wd, cd) == ("aura_flow", "fp16", "fp16")
        assert overrides == ["model_type: sdxl→aura_flow",
                             "weight_dtype: fp8_e4m3fn→fp16",
                             "clip_dtype: fp32→fp16"]
        assert len(info) == 1
        assert info[0].startswith("Preset 'AuraFlow' (overrode: ")

    def test_preset_matching_current_values_reports_no_overrides(self):
        info: list[str] = []
        mt, wd, cd, overrides = L.apply_checkpoint_preset(
            "AuraFlow", "aura_flow", "fp16", "fp16", info)
        assert overrides == []
        assert info == ["Preset 'AuraFlow' (no overrides)"]
        assert (mt, wd, cd) == ("aura_flow", "fp16", "fp16")

    def test_partial_preset_keeps_current_value_and_warns(self, monkeypatch, loader_logs):
        monkeypatch.setitem(CHECKPOINT_PRESETS, "Partial", {"model_type": "wan"})
        info: list[str] = []
        mt, wd, cd, overrides = L.apply_checkpoint_preset(
            "Partial", "sdxl", "fp8_e4m3fn", "fp32", info)
        assert (mt, wd, cd) == ("wan", "fp8_e4m3fn", "fp32")
        assert overrides == ["model_type: sdxl→wan"]
        warns = [m for m in msgs(loader_logs) if "missing" in m]
        assert warns and "weight_dtype" in warns[0] and "clip_dtype" in warns[0]

    def test_every_shipped_preset_defines_all_hidden_fields(self):
        """Custom is the only legitimate empty preset (its widgets stay visible)."""
        incomplete = {
            name: sorted(set(L._REQUIRED_PRESET_FIELDS) - set(cfg))
            for name, cfg in CHECKPOINT_PRESETS.items()
            if name != "Custom" and not set(L._REQUIRED_PRESET_FIELDS) <= set(cfg)
        }
        assert incomplete == {}


# ─────────────────────────────────────────────────────────────────────────────
#  resolve_architecture
# ─────────────────────────────────────────────────────────────────────────────

class TestResolveArchitecture:
    def test_explicit_type_skips_detection(self, monkeypatch):
        monkeypatch.setattr(L, "detect_model_type",
                            lambda p: pytest.fail("detection must not run"))
        info: list[str] = []
        resolved, detected, fmt = L.resolve_architecture("/m/flux.safetensors", "flux", info)
        assert (resolved, detected, fmt) == ("flux", None, "flux_16ch")
        assert info == ["Latent format: flux_16ch (flux)"]

    def test_auto_detect_success(self, monkeypatch):
        monkeypatch.setattr(L, "detect_model_type", lambda p: "wan_ti2v")
        info: list[str] = []
        resolved, detected, fmt = L.resolve_architecture("/m/w.safetensors",
                                                         "Auto-Detect", info)
        assert (resolved, detected, fmt) == ("wan_ti2v", "wan_ti2v", "wan_ti2v_48ch")
        assert info == ["Auto-detected: wan_ti2v",
                        "Latent format: wan_ti2v_48ch (wan_ti2v)"]

    def test_auto_detect_failure_falls_back_to_sdxl(self, monkeypatch, loader_logs):
        monkeypatch.setattr(L, "detect_model_type", lambda p: None)
        info: list[str] = []
        resolved, detected, fmt = L.resolve_architecture("/m/x.ckpt", "Auto-Detect", info)
        assert resolved == "sdxl"
        assert detected is None
        assert fmt == "sd_4ch"
        assert info[0] == "Auto-detect failed — fallback: sdxl"
        assert any("Falling back to 'sdxl'" in m for m in msgs(loader_logs))

    def test_unknown_architecture_gets_synthesised_format(self, monkeypatch):
        monkeypatch.setattr(L, "detect_model_type", lambda p: "brand_new_arch")
        info: list[str] = []
        resolved, _, fmt = L.resolve_architecture("/m/x", "Auto-Detect", info)
        assert (resolved, fmt) == ("brand_new_arch", "brand_new_arch_4ch")


# ─────────────────────────────────────────────────────────────────────────────
#  setup_offload_mode
# ─────────────────────────────────────────────────────────────────────────────

class TestSetupOffloadMode:
    def test_cpu_offload_returns_cpu_device_without_lowvram(self, monkeypatch):
        comfy = make_comfy()
        monkeypatch.setattr(L, "comfy", comfy)
        info: list[str] = []
        dev = L.setup_offload_mode("cpu_offload", info)
        assert isinstance(dev, torch.device) and dev.type == "cpu"
        mm = comfy.model_management
        assert mm.vram_state is mm.VRAMState.NORMAL_VRAM
        assert info == []

    def test_sequential_enables_lowvram_and_returns_no_device(self, monkeypatch):
        """3.5: the switch is ComfyUI's module-level vram_state. The old code
        called a set_lowvram_mode() that ComfyUI never shipped, so sequential
        offload logged a warning and did nothing on every run."""
        comfy = make_comfy()
        monkeypatch.setattr(L, "comfy", comfy)
        info: list[str] = []
        assert L.setup_offload_mode("sequential", info) is None
        mm = comfy.model_management
        assert mm.vram_state is mm.VRAMState.LOW_VRAM
        assert info == ["Offload: sequential"]

    def test_sequential_keeps_a_stricter_state(self, monkeypatch):
        """--novram must not be relaxed to LOW_VRAM by the node."""
        comfy = make_comfy()
        comfy.model_management.vram_state = comfy.model_management.VRAMState.NO_VRAM
        monkeypatch.setattr(L, "comfy", comfy)
        info: list[str] = []
        L.setup_offload_mode("sequential", info)
        mm = comfy.model_management
        assert mm.vram_state is mm.VRAMState.NO_VRAM
        assert info == ["Offload: sequential"]

    def test_sequential_survives_a_comfy_without_lowvram_support(self, monkeypatch, loader_logs):
        comfy = make_comfy()
        del comfy.model_management.VRAMState   # old comfy build
        monkeypatch.setattr(L, "comfy", comfy)
        info: list[str] = []
        assert L.setup_offload_mode("sequential", info) is None
        assert info == [], "a failed offload switch must not claim success"
        assert any("Could not enable sequential offload" in m for m in msgs(loader_logs))

    def test_none_mode_touches_nothing(self, monkeypatch):
        comfy = make_comfy()
        monkeypatch.setattr(L, "comfy", comfy)
        info: list[str] = []
        assert L.setup_offload_mode("none", info) is None
        mm = comfy.model_management
        assert mm.vram_state is mm.VRAMState.NORMAL_VRAM
        assert info == []


# ─────────────────────────────────────────────────────────────────────────────
#  estimate_vram_for_load
# ─────────────────────────────────────────────────────────────────────────────

class TestEstimateVramForLoad:
    def test_check_off_skips_probing_and_zeroes_readings(self, monkeypatch):
        monkeypatch.setattr(L, "get_available_vram",
                            lambda: pytest.fail("VRAM must not be probed"))
        monkeypatch.setattr(L, "get_total_vram",
                            lambda: pytest.fail("VRAM must not be probed"))
        info: list[str] = []
        est, avail, total = L.estimate_vram_for_load(
            "wan", "fp16", "fp16", False, "Off", "|", info)
        assert est == 17.0                      # 14.0 UNET + 3.0 CLIP
        assert (avail, total) == (0.0, 0.0)
        assert info == []

    def test_lora_flag_adds_half_a_gig(self):
        info: list[str] = []
        with_lora, _, _ = L.estimate_vram_for_load("wan", "fp16", "fp16", True, "Off", "|", info)
        assert with_lora == 17.5

    def test_fp8_weights_shrink_the_estimate(self):
        info: list[str] = []
        est, _, _ = L.estimate_vram_for_load("sdxl", "fp8_e4m3fn", "fp16", True, "Off", "|", info)
        assert est == estimate_vram_usage("sdxl", "fp8_e4m3fn", "fp16", True, False)
        assert est == 5.9                       # 6.5*0.6 + 1.5*1.0 + 0.5

    def test_companion_unets_add_unet_memory_only(self):
        info: list[str] = []
        base, _, _ = L.estimate_vram_for_load("wan", "fp16", "fp16", False, "Off", "|", info)
        twin, _, _ = L.estimate_vram_for_load("wan", "fp16", "fp16", False, "Off", "|", info,
                                              extra_unet_count=1)
        assert twin - base == pytest.approx(_BASE_VRAM["wan"] * _DTYPE_MULT["fp16"])
        assert twin == 31.0

    def test_check_on_reports_readings_and_warns_when_tight(self, monkeypatch):
        monkeypatch.setattr(L, "get_available_vram", lambda: 10.0)
        monkeypatch.setattr(L, "get_total_vram", lambda: 24.0)
        info: list[str] = []
        est, avail, total = L.estimate_vram_for_load(
            "wan", "fp16", "fp16", False, "On", "│", info)
        assert (est, avail, total) == (17.0, 10.0, 24.0)
        assert info[0] == "VRAM: ~17.0 GB needed │ 10.00 GB free / 24.00 GB total"
        assert len(info) == 2 and info[1].startswith("VRAM tight!")

    def test_check_on_stays_quiet_when_there_is_headroom(self, monkeypatch):
        monkeypatch.setattr(L, "get_available_vram", lambda: 40.0)
        monkeypatch.setattr(L, "get_total_vram", lambda: 48.0)
        info: list[str] = []
        est, avail, _ = L.estimate_vram_for_load("wan", "fp16", "fp16", False, "On", "|", info)
        assert est == 17.0 and avail == 40.0
        assert len(info) == 1 and "tight" not in info[0]

    def test_exactly_at_ninety_percent_is_not_tight(self, monkeypatch):
        """est > avail*0.9 — the boundary itself must not warn."""
        monkeypatch.setattr(L, "get_available_vram", lambda: 17.0 / 0.9)
        monkeypatch.setattr(L, "get_total_vram", lambda: 32.0)
        info: list[str] = []
        L.estimate_vram_for_load("wan", "fp16", "fp16", False, "On", "|", info)
        assert len(info) == 1, info

    def test_unreadable_vram_does_not_warn(self, monkeypatch):
        monkeypatch.setattr(L, "get_available_vram", lambda: 0.0)
        monkeypatch.setattr(L, "get_total_vram", lambda: 0.0)
        info: list[str] = []
        L.estimate_vram_for_load("wan", "fp16", "fp16", False, "On", "|", info)
        assert len(info) == 1 and "tight" not in info[0]


# ─────────────────────────────────────────────────────────────────────────────
#  _require_baked_vae / construct_audio_vae
# ─────────────────────────────────────────────────────────────────────────────

class TestRequireBakedVae:
    def test_missing_vae_names_the_checkpoint_and_the_fix(self):
        with pytest.raises(RuntimeError) as exc:
            L._require_baked_vae(None, "wan2.2_ti2v_5B_fp16.safetensors")
        text = str(exc.value)
        assert "wan2.2_ti2v_5B_fp16.safetensors" in text
        assert "no VAE weights baked in" in text
        assert "standalone vae_name" in text

    def test_present_vae_passes(self):
        sentinel = object()
        assert L._require_baked_vae(sentinel, "x.safetensors") is None


class TestConstructAudioVae:
    def test_architecture_without_remap_uses_the_raw_state_dict(self, monkeypatch):
        built = {}

        def fake_vae(sd=None, metadata=None):
            built.update(sd=sd, metadata=metadata)
            return "AUDIO_VAE"

        comfy = make_comfy(
            sd__VAE=fake_vae,
            utils__state_dict_prefix_replace=lambda *a, **k: pytest.fail("no remap expected"),
        )
        monkeypatch.setattr(L, "comfy", comfy)
        sd = {"encoder.weight": 1}
        assert L.construct_audio_vae(sd, {"m": 1}, "minimax") == "AUDIO_VAE"
        assert built["sd"] is sd
        assert built["metadata"] == {"m": 1}

    def test_ltxav_remaps_prefixes_before_building(self, monkeypatch):
        seen = {}

        def fake_replace(sd, remap, filter_keys=False):
            seen.update(sd=sd, remap=remap, filter_keys=filter_keys)
            return {"autoencoder.w": 1}

        built = {}

        def fake_vae(sd=None, metadata=None):
            built.update(sd=sd)
            return "AUDIO_VAE"

        monkeypatch.setattr(L, "comfy", make_comfy(
            sd__VAE=fake_vae, utils__state_dict_prefix_replace=fake_replace))
        original = {"audio_vae.w": 1, "model.diffusion": 2}
        assert L.construct_audio_vae(original, None, "ltxav") == "AUDIO_VAE"
        assert seen["sd"] is original
        assert seen["remap"] == {"audio_vae.": "autoencoder.", "vocoder.": "vocoder."}
        assert seen["filter_keys"] is True
        assert built["sd"] == {"autoencoder.w": 1}, "must build from the remapped dict"


# ─────────────────────────────────────────────────────────────────────────────
#  load_unet_and_baked_vae
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def unet_env(monkeypatch, tmp_path):
    """A fake comfy whose loaders record their arguments."""
    state = types.SimpleNamespace(
        path=write_file(tmp_path / "unet.safetensors", b"UNETBYTES"),
        diffusion_calls=[], sd_calls=[], ckpt_calls=[], guess_calls=[],
        torch_file_calls=[], vae=types.SimpleNamespace(patcher=None),
        raise_on_load=None,
    )

    def load_diffusion_model(path, model_options=None):
        state.diffusion_calls.append((path, model_options))
        if state.raise_on_load:
            raise state.raise_on_load
        return FakeModel("diffusion")

    def load_diffusion_model_state_dict(sd, model_options=None, metadata=None):
        state.sd_calls.append((sd, model_options, metadata))
        return FakeModel("from_sd")

    def load_checkpoint_guess_config(path, **kw):
        state.ckpt_calls.append((path, kw))
        return (FakeModel("ckpt"), "CLIP", state.vae)

    def load_state_dict_guess_config(sd, **kw):
        state.guess_calls.append((sd, kw))
        return (FakeModel("guess"), "CLIP", state.vae)

    def load_torch_file(path, return_metadata=False):
        state.torch_file_calls.append((path, return_metadata))
        return ({"k": 1}, {"meta": "yes"}) if return_metadata else {"k": 1}

    monkeypatch.setattr(L, "comfy", make_comfy(
        sd__load_diffusion_model=load_diffusion_model,
        sd__load_diffusion_model_state_dict=load_diffusion_model_state_dict,
        sd__load_checkpoint_guess_config=load_checkpoint_guess_config,
        sd__load_state_dict_guess_config=load_state_dict_guess_config,
        sd__VAE=lambda sd=None, metadata=None: "AUDIO_VAE",
        utils__load_torch_file=load_torch_file,
        # LTX-AV namespaces its audio-VAE tensors; the real helper pops them out
        # of the caller's dict, which this mimics.
        utils__state_dict_prefix_replace=lambda sd, remap, filter_keys=False: {
            "autoencoder.w": sd.pop("audio_vae.w", 1)},
    ))
    return state


def call_unet(state, *, unet_name="unet.safetensors", weight_dtype="fp16",
              offload_mode="none", vae_name="None", audio_vae_name="None",
              resolved_type="wan", caching=False, info=None):
    return L.load_unet_and_baked_vae(
        state.path, unet_name, weight_dtype, offload_mode, vae_name,
        audio_vae_name, resolved_type, caching, "|",
        info if info is not None else [],
    )


class TestLoadUnetAndBakedVae:
    def test_plain_load_passes_the_mapped_dtype(self, unet_env):
        info: list[str] = []
        model, vae, audio_vae, t, hit, vt, vhit = call_unet(unet_env, info=info)
        assert isinstance(model, FakeModel) and model.tag == "diffusion"
        assert (vae, audio_vae) == (None, None)
        assert hit is False and vhit is False and vt == 0.0
        assert unet_env.diffusion_calls == [(unet_env.path, {"dtype": torch.float16})]
        assert info[0].startswith("UNET: unet.safetensors [fp16] (")

    # Resolved by name, not at class-body time: CI runs with no torch at all
    # and conftest's stub carries no float8 attributes, so touching them here
    # took the whole file down at collection.
    @pytest.mark.parametrize("dtype,attr", [
        ("fp8_e4m3fn", "float8_e4m3fn"),
        ("fp8_e5m2", "float8_e5m2"),
        ("bf16", "bfloat16"),
        ("fp32", "float32"),
    ])
    def test_every_dtype_name_maps_to_a_torch_dtype(self, unet_env, dtype, attr):
        expected = getattr(torch, attr, None)
        if expected is None:
            pytest.skip(f"this torch has no {attr}")
        call_unet(unet_env, weight_dtype=dtype)
        assert unet_env.diffusion_calls[0][1] == {"dtype": expected}

    def test_unknown_dtype_leaves_the_choice_to_comfy(self, unet_env):
        call_unet(unet_env, weight_dtype="default")
        assert unet_env.diffusion_calls[0][1] == {}

    def test_gguf_ignores_the_dtype_widget(self, unet_env, loader_logs):
        call_unet(unet_env, unet_name="wan2.2.GGUF", weight_dtype="fp8_e4m3fn")
        assert unet_env.diffusion_calls[0][1] == {}, "GGUF carries its own quantisation"
        assert any("GGUF detected" in m for m in msgs(loader_logs))

    def test_second_load_hits_the_cache(self, unet_env):
        call_unet(unet_env, caching=True)
        info: list[str] = []
        model, _, _, t, hit, _, _ = call_unet(unet_env, caching=True, info=info)
        assert hit is True
        assert t == 0.0
        assert len(unet_env.diffusion_calls) == 1, "cached UNET must not reload"
        assert info == ["UNET: unet.safetensors (cached)"]

    def test_caching_off_reloads(self, unet_env):
        call_unet(unet_env, caching=False)
        _, _, _, _, hit, _, _ = call_unet(unet_env, caching=False)
        assert hit is False
        assert len(unet_env.diffusion_calls) == 2

    def test_offload_mode_is_part_of_the_cache_key(self, unet_env):
        call_unet(unet_env, caching=True, offload_mode="none")
        _, _, _, _, hit, _, _ = call_unet(unet_env, caching=True, offload_mode="sequential")
        assert hit is False, "sequential patches the model — must not reuse the plain load"
        assert len(unet_env.diffusion_calls) == 2

    def test_dtype_is_part_of_the_cache_key(self, unet_env):
        call_unet(unet_env, caching=True, weight_dtype="fp16")
        _, _, _, _, hit, _, _ = call_unet(unet_env, caching=True, weight_dtype="bf16")
        assert hit is False
        assert len(unet_env.diffusion_calls) == 2

    def test_touched_file_invalidates_the_cache(self, unet_env, tmp_path):
        call_unet(unet_env, caching=True)
        with open(unet_env.path, "wb") as fh:
            fh.write(b"UNETBYTES-BUT-LONGER")
        _, _, _, _, hit, _, _ = call_unet(unet_env, caching=True)
        assert hit is False
        assert len(unet_env.diffusion_calls) == 2

    def test_baked_vae_extraction(self, unet_env):
        info: list[str] = []
        model, vae, audio_vae, _, _, vt, vhit = call_unet(
            unet_env, vae_name="Baked VAE (from UNET)", info=info)
        assert vae is unet_env.vae
        assert audio_vae is None
        assert vhit is False
        assert unet_env.ckpt_calls[0][0] == unet_env.path
        assert unet_env.ckpt_calls[0][1]["output_vae"] is True
        assert unet_env.ckpt_calls[0][1]["output_clip"] is False
        assert unet_env.diffusion_calls == []
        assert any(i.startswith("VAE: Baked from UNET (") for i in info)

    def test_checkpoint_without_a_baked_vae_fails_loudly(self, unet_env):
        unet_env.vae = None
        with pytest.raises(RuntimeError) as exc:
            call_unet(unet_env, vae_name="Baked VAE (from UNET)")
        text = str(exc.value)
        assert "no VAE weights baked in" in text
        assert "Failed to load UNET 'unet.safetensors'" in text

    def test_baked_vae_is_cached_and_reused(self, unet_env):
        call_unet(unet_env, vae_name="Baked VAE (from UNET)", caching=True)
        info: list[str] = []
        _, vae, _, _, hit, _, vhit = call_unet(
            unet_env, vae_name="Baked VAE (from UNET)", caching=True, info=info)
        assert (hit, vhit) == (True, True)
        assert vae is unet_env.vae
        assert len(unet_env.ckpt_calls) == 1
        assert info == ["UNET: unet.safetensors (cached)", "VAE: Baked from UNET (cached)"]

    def test_evicted_baked_vae_forces_a_full_reload(self, unet_env):
        call_unet(unet_env, vae_name="Baked VAE (from UNET)", caching=True)
        _vae_cache.clear()          # e.g. evicted by another architecture
        _, vae, _, _, hit, _, _ = call_unet(
            unet_env, vae_name="Baked VAE (from UNET)", caching=True)
        assert hit is False, "a cached UNET is useless if its baked VAE is gone"
        assert len(unet_env.ckpt_calls) == 2
        assert vae is unet_env.vae

    def test_audio_vae_extraction_reads_the_checkpoint_once(self, unet_env):
        info: list[str] = []
        model, vae, audio_vae, _, _, _, _ = call_unet(
            unet_env, audio_vae_name="Baked Audio VAE (from UNET)",
            resolved_type="minimax", info=info)
        assert audio_vae == "AUDIO_VAE"
        assert vae is None
        assert unet_env.torch_file_calls == [(unet_env.path, True)]
        assert model.tag == "from_sd"
        assert unet_env.sd_calls[0][0] == {"k": 1}
        assert unet_env.sd_calls[0][2] == {"meta": "yes"}
        assert any(i.startswith("AUDIO VAE: Baked from UNET (") for i in info)

    def test_audio_plus_video_vae_share_a_single_file_read(self, unet_env):
        unet_env.vae = types.SimpleNamespace(patcher=types.SimpleNamespace())
        model, vae, audio_vae, _, _, _, _ = call_unet(
            unet_env, vae_name="Baked VAE (from UNET)",
            audio_vae_name="Baked Audio VAE (from UNET)", resolved_type="ltxav")
        assert unet_env.torch_file_calls == [(unet_env.path, True)], "read the UNET once"
        assert unet_env.ckpt_calls == [], "must not re-read from disk"
        assert len(unet_env.guess_calls) == 1
        assert model.tag == "guess"
        assert vae is unet_env.vae and audio_vae == "AUDIO_VAE"
        assert model.cached_patcher_init[2] == 0
        assert vae.patcher.cached_patcher_init == (
            "VAE_PATCHER_FN", (unet_env.path, None, {"dtype": torch.float16}, {}))

    def test_audio_plus_video_vae_are_both_cached_together(self, unet_env):
        kwargs = dict(vae_name="Baked VAE (from UNET)",
                      audio_vae_name="Baked Audio VAE (from UNET)",
                      resolved_type="ltxav", caching=True)
        call_unet(unet_env, **kwargs)
        info: list[str] = []
        _, vae, audio_vae, _, hit, _, vhit = call_unet(unet_env, info=info, **kwargs)
        assert (hit, vhit) == (True, True)
        assert vae is unet_env.vae and audio_vae == "AUDIO_VAE"
        assert len(unet_env.torch_file_calls) == 1, "nothing may be re-read from disk"
        assert len(unet_env.guess_calls) == 1
        assert info == ["UNET: unet.safetensors (cached)",
                        "VAE: Baked from UNET (cached)",
                        "AUDIO VAE: Baked from UNET (cached)"]

    def test_audio_plus_video_vae_without_a_patcher_is_fine(self, unet_env):
        unet_env.vae = types.SimpleNamespace(patcher=None)
        _, vae, _, _, _, _, _ = call_unet(
            unet_env, vae_name="Baked VAE (from UNET)",
            audio_vae_name="Baked Audio VAE (from UNET)", resolved_type="ltxav")
        assert vae is unet_env.vae

    def test_cached_audio_vae_is_reused(self, unet_env):
        call_unet(unet_env, audio_vae_name="Baked Audio VAE (from UNET)",
                  resolved_type="minimax", caching=True)
        info: list[str] = []
        _, _, audio_vae, _, hit, _, _ = call_unet(
            unet_env, audio_vae_name="Baked Audio VAE (from UNET)",
            resolved_type="minimax", caching=True, info=info)
        assert hit is True and audio_vae == "AUDIO_VAE"
        assert len(unet_env.torch_file_calls) == 1
        assert info == ["UNET: unet.safetensors (cached)",
                        "AUDIO VAE: Baked from UNET (cached)"]

    def test_evicted_audio_vae_forces_a_full_reload(self, unet_env):
        call_unet(unet_env, audio_vae_name="Baked Audio VAE (from UNET)",
                  resolved_type="minimax", caching=True)
        _audio_vae_cache.clear()
        _, _, _, _, hit, _, _ = call_unet(
            unet_env, audio_vae_name="Baked Audio VAE (from UNET)",
            resolved_type="minimax", caching=True)
        assert hit is False
        assert len(unet_env.torch_file_calls) == 2

    def test_loader_failure_is_wrapped_with_the_model_name(self, unet_env):
        unet_env.raise_on_load = ValueError("header too small")
        with pytest.raises(RuntimeError) as exc:
            call_unet(unet_env, unet_name="broken.safetensors")
        text = str(exc.value)
        assert "Failed to load UNET 'broken.safetensors'" in text
        assert "header too small" in text

    def test_failed_load_is_not_cached(self, unet_env):
        unet_env.raise_on_load = ValueError("boom")
        with pytest.raises(RuntimeError):
            call_unet(unet_env, caching=True)
        unet_env.raise_on_load = None
        _, _, _, _, hit, _, _ = call_unet(unet_env, caching=True)
        assert hit is False


# ─────────────────────────────────────────────────────────────────────────────
#  load_clip_stack
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def clip_env(monkeypatch, tmp_path):
    """Wire the real assemble_clip_paths through a fake folder_paths."""
    import radiance.model.detect as detect

    state = types.SimpleNamespace(
        clip_l=write_file(tmp_path / "clip_l.safetensors", b"L"),
        clip_g=write_file(tmp_path / "clip_g.safetensors", b"G"),
        t5=write_file(tmp_path / "t5.safetensors", b"T"),
        unet=write_file(tmp_path / "unet.safetensors", b"U"),
        load_clip_calls=[], ensure_calls=[], raise_on_load=None,
    )
    fp = FakeFolderPaths(files={
        ("text_encoders", "clip_l.safetensors"): state.clip_l,
        ("text_encoders", "clip_g.safetensors"): state.clip_g,
        ("text_encoders", "t5.safetensors"): state.t5,
    }, folders={"embeddings": "/models/embeddings"})
    state.fp = fp

    def load_clip(ckpt_paths=None, embedding_directory=None, clip_type=None,
                  model_options=None):
        state.load_clip_calls.append(
            dict(ckpt_paths=list(ckpt_paths), embedding_directory=embedding_directory,
                 clip_type=clip_type, model_options=model_options))
        if state.raise_on_load:
            raise state.raise_on_load
        return f"CLIP<{len(ckpt_paths)}>"

    monkeypatch.setattr(L, "comfy", make_comfy(sd__load_clip=load_clip))
    monkeypatch.setattr(L, "folder_paths", fp)
    monkeypatch.setattr(detect, "folder_paths", fp)
    monkeypatch.setattr(L, "get_clip_type_enum", lambda arch: f"enum:{arch}")

    def fake_ensure(name, folder_type, auto_download=False):
        state.ensure_calls.append((name, folder_type, auto_download))
        return fp.get_full_path(folder_type, name)

    monkeypatch.setattr(L, "ensure_model_exists", fake_ensure)
    return state


def call_clip(state, *, arch="sdxl", clip_l=None, clip_g=None, t5xxl=None,
              llm_encoder=None, text_projection=None, clip_dtype="fp16",
              offload_mode="none", clip_load_device=None, caching=False,
              auto_download=False, info=None):
    return L.load_clip_stack(
        arch, state.unet, clip_l, clip_g, t5xxl, llm_encoder, text_projection,
        clip_dtype, offload_mode, clip_load_device, caching, "|", auto_download,
        info if info is not None else [],
    )


class TestLoadClipStack:
    def test_no_encoders_names_the_required_slots(self, clip_env):
        with pytest.raises(ValueError) as exc:
            call_clip(clip_env, arch="flux")
        text = str(exc.value)
        # 3.5: required slots are checked first; clip_l is auto-filled from
        # the only matching file, t5xxl has no match and is named.
        assert "flux needs text encoder slot(s) t5xxl" in text

    def test_unknown_architecture_falls_back_to_clip_l_in_the_message(self, clip_env):
        with pytest.raises(ValueError) as exc:
            call_clip(clip_env, arch="totally_unknown")
        assert str(exc.value).rstrip().endswith("clip_l")

    def test_sdxl_loads_both_encoders_in_slot_order(self, clip_env):
        info: list[str] = []
        clip, slots, elapsed, hit = call_clip(
            clip_env, clip_l="clip_l.safetensors", clip_g="clip_g.safetensors", info=info)
        call = clip_env.load_clip_calls[0]
        assert call["ckpt_paths"] == [clip_env.clip_l, clip_env.clip_g]
        assert call["clip_type"] == "enum:sdxl"
        assert call["embedding_directory"] == ["/models/embeddings"]
        assert call["model_options"] == {"dtype": torch.float16}
        assert clip == "CLIP<2>"
        assert slots == ["clip_l", "clip_g"]
        assert hit is False and elapsed >= 0.0
        assert info[0].startswith("CLIP: clip_l+clip_g [fp16] (")

    def test_baked_text_projection_uses_the_unet_path(self, clip_env):
        clip, slots, _, _ = call_clip(
            clip_env, arch="ltxav", llm_encoder="t5.safetensors",
            text_projection="Baked (from UNET)")
        assert clip_env.load_clip_calls[0]["ckpt_paths"] == [clip_env.t5, clip_env.unet]
        assert ("Baked (from UNET)", "text_encoders", False) not in clip_env.ensure_calls
        assert slots == ["llm_encoder", "text_projection"]

    def test_every_non_baked_slot_is_resolved_with_the_download_flag(self, clip_env):
        call_clip(clip_env, clip_l="clip_l.safetensors", clip_g="clip_g.safetensors",
                  auto_download=True)
        assert ("clip_l.safetensors", "text_encoders", True) in clip_env.ensure_calls
        assert ("clip_g.safetensors", "text_encoders", True) in clip_env.ensure_calls
        assert all(c[1] == "text_encoders" for c in clip_env.ensure_calls)

    @pytest.mark.parametrize("dtype,attr", [
        ("fp16", "float16"), ("bf16", "bfloat16"),
        ("fp8_e4m3fn", "float8_e4m3fn"), ("fp32", "float32"),
    ])
    def test_clip_dtypes_map_through(self, clip_env, dtype, attr):
        expected = getattr(torch, attr, None)
        if expected is None:
            pytest.skip(f"this torch has no {attr}")
        call_clip(clip_env, clip_l="clip_l.safetensors", clip_dtype=dtype)
        assert clip_env.load_clip_calls[0]["model_options"]["dtype"] is expected

    def test_default_clip_dtype_sends_empty_model_options(self, clip_env):
        call_clip(clip_env, clip_l="clip_l.safetensors", clip_dtype="default")
        assert clip_env.load_clip_calls[0]["model_options"] == {}

    def test_cpu_offload_device_is_forwarded(self, clip_env):
        dev = torch.device("cpu")
        call_clip(clip_env, clip_l="clip_l.safetensors", clip_load_device=dev,
                  clip_dtype="default")
        assert clip_env.load_clip_calls[0]["model_options"] == {"load_device": dev}

    def test_second_load_hits_the_cache(self, clip_env):
        both = dict(clip_l="clip_l.safetensors", clip_g="clip_g.safetensors")
        call_clip(clip_env, caching=True, **both)
        info: list[str] = []
        clip, slots, elapsed, hit = call_clip(clip_env, caching=True, info=info, **both)
        assert hit is True and elapsed == 0.0 and clip == "CLIP<2>"
        assert len(clip_env.load_clip_calls) == 1
        assert info == ["CLIP: clip_l+clip_g (cached)"]

    def test_sdxl_missing_clip_g_is_filled_from_disk(self, clip_env):
        info: list[str] = []
        clip, slots, _, _ = call_clip(clip_env, clip_l="clip_l.safetensors", info=info)
        assert clip == "CLIP<2>" and "CLIP clip_g: auto -> clip_g.safetensors" in info

    def test_offload_mode_is_part_of_the_clip_cache_key(self, clip_env):
        call_clip(clip_env, clip_l="clip_l.safetensors", caching=True, offload_mode="none")
        _, _, _, hit = call_clip(clip_env, clip_l="clip_l.safetensors", caching=True,
                                 offload_mode="cpu_offload")
        assert hit is False
        assert len(clip_env.load_clip_calls) == 2

    def test_architecture_is_part_of_the_clip_cache_key(self, clip_env):
        call_clip(clip_env, arch="sdxl", clip_l="clip_l.safetensors", caching=True)
        _, _, _, hit = call_clip(clip_env, arch="sd1.5", clip_l="clip_l.safetensors",
                                 caching=True)
        assert hit is False
        assert clip_env.load_clip_calls[1]["clip_type"] == "enum:sd1.5"

    def test_load_failure_is_wrapped(self, clip_env):
        clip_env.raise_on_load = OSError("safetensors header corrupt")
        with pytest.raises(RuntimeError) as exc:
            call_clip(clip_env, clip_l="clip_l.safetensors")
        assert "Failed to load CLIP" in str(exc.value)
        assert "safetensors header corrupt" in str(exc.value)

    def test_missing_encoder_file_is_dropped_from_the_paths(self, clip_env):
        clip, slots, _, _ = call_clip(clip_env, clip_l="clip_l.safetensors",
                                      clip_g="not_installed.safetensors")
        assert clip_env.load_clip_calls[0]["ckpt_paths"] == [clip_env.clip_l]
        assert clip == "CLIP<1>"

    # Was xfail(strict): the defect it documented is fixed.
    def test_unused_slots_are_not_reported_as_used(self, clip_env):
        _, slots, _, _ = call_clip(clip_env, arch="sd1.5",
                                   clip_l="clip_l.safetensors", clip_g="None")
        assert clip_env.load_clip_calls[0]["ckpt_paths"] == [clip_env.clip_l]
        assert slots == ["clip_l"]


# ─────────────────────────────────────────────────────────────────────────────
#  load_standalone_vae
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def vae_env(monkeypatch, tmp_path):
    state = types.SimpleNamespace(
        path=write_file(tmp_path / "ae.safetensors", b"VAEBYTES"),
        vae_calls=[], raise_on_build=None,
    )
    fp = FakeFolderPaths(files={("vae", "ae.safetensors"): state.path})
    state.fp = fp

    def fake_vae(sd=None, metadata=None):
        state.vae_calls.append((sd, metadata))
        if state.raise_on_build:
            raise state.raise_on_build
        return "VAE_OBJ"

    monkeypatch.setattr(L, "comfy", make_comfy(sd__VAE=fake_vae))
    monkeypatch.setattr(L, "folder_paths", fp)
    return state


def sd_meta(path):
    return ({"decoder.weight": path}, {"config": "ltx"})


class TestLoadStandaloneVae:
    def test_missing_vae_raises_with_the_remedy(self, vae_env):
        with pytest.raises(FileNotFoundError) as exc:
            L.load_standalone_vae("absent.safetensors", False, False, "|", [], sd_meta)
        text = str(exc.value)
        assert "VAE not found: 'absent.safetensors'" in text
        assert "auto_download" in text

    def test_happy_path_builds_from_the_callers_metadata_reader(self, vae_env):
        info: list[str] = []
        vae, elapsed, hit = L.load_standalone_vae(
            "ae.safetensors", False, False, "|", info, sd_meta)
        assert vae == "VAE_OBJ"
        assert hit is False and elapsed >= 0.0
        assert vae_env.vae_calls == [({"decoder.weight": vae_env.path}, {"config": "ltx"})]
        assert info[0].startswith("VAE: ae.safetensors (")

    def test_second_load_hits_the_cache(self, vae_env):
        L.load_standalone_vae("ae.safetensors", False, True, "|", [], sd_meta)
        info: list[str] = []
        vae, elapsed, hit = L.load_standalone_vae(
            "ae.safetensors", False, True, "|", info, sd_meta)
        assert (vae, elapsed, hit) == ("VAE_OBJ", 0.0, True)
        assert len(vae_env.vae_calls) == 1
        assert info == ["VAE: ae.safetensors (cached)"]

    def test_caching_off_rebuilds(self, vae_env):
        L.load_standalone_vae("ae.safetensors", False, False, "|", [], sd_meta)
        _, _, hit = L.load_standalone_vae("ae.safetensors", False, False, "|", [], sd_meta)
        assert hit is False
        assert len(vae_env.vae_calls) == 2

    def test_build_failure_is_wrapped_with_the_file_name(self, vae_env):
        vae_env.raise_on_build = KeyError("decoder.conv_in.weight")
        with pytest.raises(RuntimeError) as exc:
            L.load_standalone_vae("ae.safetensors", False, False, "|", [], sd_meta)
        assert "Failed to load VAE 'ae.safetensors'" in str(exc.value)
        assert "decoder.conv_in.weight" in str(exc.value)

    def test_conv_vae_warns_about_the_compression_mismatch(self, monkeypatch, tmp_path):
        path = write_file(tmp_path / "LTX-2.5-Video-VAE-Conv.safetensors", b"V")
        monkeypatch.setattr(L, "comfy", make_comfy(sd__VAE=lambda sd=None, metadata=None: "V"))
        monkeypatch.setattr(L, "folder_paths", FakeFolderPaths(
            files={("vae", "LTX-2.5-Video-VAE-Conv.safetensors"): path}))
        info: list[str] = []
        L.load_standalone_vae("LTX-2.5-Video-VAE-Conv.safetensors", False, False,
                              "|", info, sd_meta)
        assert "16x spatial" in info[0]
        assert "ltx-2.5-video-vae-bf16.safetensors" in info[0]
        assert len(info) == 2

    def test_normal_vae_gets_no_compression_warning(self, vae_env):
        info: list[str] = []
        L.load_standalone_vae("ae.safetensors", False, False, "|", info, sd_meta)
        assert not any("16x spatial" in i for i in info)


# ─────────────────────────────────────────────────────────────────────────────
#  apply_lora_stack
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def lora_env(monkeypatch):
    state = types.SimpleNamespace(
        applied=[], available={"good.safetensors": "/loras/good.safetensors",
                               "second.safetensors": "/loras/second.safetensors"},
        raise_on_apply=None, loaded=[],
    )

    def load_lora_for_models(model, clip, data, model_str, clip_str):
        if state.raise_on_apply:
            raise state.raise_on_apply
        state.applied.append((data, model_str, clip_str))
        return (f"{model}+{data}", f"{clip}+{data}")

    def load_torch_file(path):
        state.loaded.append(path)
        return f"W({os.path.basename(path)})"

    monkeypatch.setattr(L, "comfy", make_comfy(
        sd__load_lora_for_models=load_lora_for_models,
        utils__load_torch_file=load_torch_file,
    ))
    monkeypatch.setattr(L, "folder_paths", FakeFolderPaths(files={
        ("loras", name): path for name, path in state.available.items()}))
    return state


class TestApplyLoraStack:
    def test_empty_stack_passes_the_model_through_untouched(self, lora_env):
        info: list[str] = []
        model, clip, applied, out = L.apply_lora_stack("M", "C", None, "skip", "|", info)
        assert (model, clip) == ("M", "C")
        assert applied == [] and out is None
        assert info == []

    def test_zero_strength_entry_is_skipped_entirely(self, lora_env):
        info: list[str] = []
        model, clip, applied, out = L.apply_lora_stack(
            "M", "C", [("good.safetensors", 0, 0)], "skip", "|", info)
        assert (model, clip, applied, out) == ("M", "C", [], None)
        assert lora_env.loaded == []
        assert info == []

    def test_clip_only_strength_still_applies(self, lora_env):
        model, clip, applied, out = L.apply_lora_stack(
            "M", "C", [("good.safetensors", 0, 0.8)], "skip", "|", [])
        assert lora_env.applied == [("W(good.safetensors)", 0, 0.8)]
        assert out == [("good.safetensors", 0, 0.8)]

    def test_successful_application_rebinds_model_and_clip(self, lora_env):
        info: list[str] = []
        model, clip, applied, out = L.apply_lora_stack(
            "M", "C", [("good.safetensors", 1.0, 0.5)], "skip", "|", info)
        assert model == "M+W(good.safetensors)"
        assert clip == "C+W(good.safetensors)"
        assert applied == [{"name": "good.safetensors", "model_str": 1.0, "clip_str": 0.5}]
        assert out == [("good.safetensors", 1.0, 0.5)]
        assert info[0].startswith("LoRA 1: good.safetensors [m=1.0 c=0.5] (")

    def test_chained_loras_apply_in_order(self, lora_env):
        info: list[str] = []
        model, clip, applied, out = L.apply_lora_stack(
            "M", "C", [("good.safetensors", 1.0, 1.0), ("second.safetensors", 0.5, 0.5)],
            "skip", "|", info)
        assert model == "M+W(good.safetensors)+W(second.safetensors)"
        assert [e["name"] for e in applied] == ["good.safetensors", "second.safetensors"]
        assert out == [("good.safetensors", 1.0, 1.0), ("second.safetensors", 0.5, 0.5)]
        assert info[1].startswith("LoRA 2: second.safetensors")

    def test_missing_lora_is_skipped_and_the_rest_still_apply(self, lora_env, loader_logs):
        info: list[str] = []
        model, clip, applied, out = L.apply_lora_stack(
            "M", "C", [("ghost.safetensors", 1.0, 1.0), ("good.safetensors", 1.0, 1.0)],
            "skip", "|", info)
        assert model == "M+W(good.safetensors)"
        assert [e["name"] for e in applied] == ["good.safetensors"]
        assert info[0] == "LoRA 1: ghost.safetensors (not found)"
        assert info[1].startswith("LoRA 2: good.safetensors")
        assert any("LoRA not found: 'ghost.safetensors'" in m for m in msgs(loader_logs))

    def test_missing_lora_raises_when_configured_to(self, lora_env):
        with pytest.raises(FileNotFoundError) as exc:
            L.apply_lora_stack("M", "C", [("ghost.safetensors", 1.0, 1.0)],
                               "raise", "|", [])
        assert str(exc.value) == "LoRA not found: 'ghost.safetensors'"

    def test_broken_lora_is_skipped(self, lora_env, loader_logs):
        lora_env.raise_on_apply = RuntimeError("shape mismatch in lora_up")
        info: list[str] = []
        model, clip, applied, out = L.apply_lora_stack(
            "M", "C", [("good.safetensors", 1.0, 1.0)], "skip", "|", info)
        assert (model, clip) == ("M", "C"), "a failed LoRA must not corrupt the model"
        assert applied == [] and out is None
        assert info == ["LoRA 1: good.safetensors (failed)"]
        assert any("shape mismatch in lora_up" in m for m in msgs(loader_logs))

    def test_broken_lora_raises_when_configured_to(self, lora_env):
        lora_env.raise_on_apply = RuntimeError("shape mismatch in lora_up")
        with pytest.raises(RuntimeError) as exc:
            L.apply_lora_stack("M", "C", [("good.safetensors", 1.0, 1.0)],
                               "raise", "|", [])
        text = str(exc.value)
        assert "Failed to apply LoRA 'good.safetensors'" in text
        assert "shape mismatch in lora_up" in text


def test_flux_with_only_t5_gets_clip_l_filled_not_mochi():
    """3.5 live log: a Flux Loader with clip_l empty loaded MochiTEModel_."""
    from radiance.model.detect import autofill_required_clip_slots
    files = ["clip_l.safetensors", "t5xxl_fp16.safetensors", "t5xxl_fp8_e4m3fn_scaled.safetensors",
             "umt5_xxl_fp8_e4m3fn_scaled.safetensors"]
    slots, filled, missing = autofill_required_clip_slots(
        "flux", files, clip_l="None", clip_g="None", t5xxl="t5xxl_fp16.safetensors",
        llm_encoder="None", text_projection="None")
    assert filled == {"clip_l": "clip_l.safetensors"} and not missing
    # Two t5xxl candidates: a choice for the user, not a guess.
    slots, filled, missing = autofill_required_clip_slots(
        "flux", files, clip_l="clip_l.safetensors", clip_g="None", t5xxl="None",
        llm_encoder="None", text_projection="None")
    assert missing == ["t5xxl"]
    # Architectures without a requirement are untouched.
    assert autofill_required_clip_slots("wan", files, t5xxl="None")[2] == []
