"""Release audit F4: what users make lives in ComfyUI's user directory.

A Registry update replaces the package folder; the workflow library and saved
gizmos used to live inside it and went with it."""
import json
from pathlib import Path

import pytest

from radiance.core import user_data as ud


@pytest.fixture
def package(tmp_path):
    pkg = tmp_path / "pkg" / "workflows"
    (pkg / "official" / "Local").mkdir(parents=True)
    (pkg / "official" / "Local" / "start.rad").write_bytes(b"RADZ-start")
    (pkg / "official" / "API" ).mkdir()
    (pkg / "official" / "API" / "veo.rad").write_bytes(b"RADZ-veo")
    (pkg / "start.json").write_text("{}", encoding="utf-8")            # a shipped template
    (pkg / "HDR_Test").mkdir()
    (pkg / "HDR_Test" / "SH010_v001.rad").write_bytes(b"RADZ-user")      # a user's save
    (pkg / "_assets_bins.json").write_text("[]", encoding="utf-8")       # a user's bins
    return pkg


@pytest.fixture
def user_dir(tmp_path, monkeypatch):
    import folder_paths
    home = tmp_path / "ComfyUI" / "user"
    monkeypatch.setattr(folder_paths, "get_user_directory", lambda: str(home), raising=False)
    monkeypatch.delenv("RADIANCE_USER_DATA_IN_PACKAGE", raising=False)
    return home / "radiance"


def test_without_comfyui_the_package_folders_are_used(package, monkeypatch):
    import folder_paths
    monkeypatch.delattr(folder_paths, "get_user_directory", raising=False)
    assert ud.user_root() is None
    assert ud.library_dir(package) == package


def test_the_library_moves_to_the_user_directory(package, user_dir):
    lib = ud.library_dir(package)
    assert lib == user_dir / "workflows"
    assert (lib / "HDR_Test" / "SH010_v001.rad").read_bytes() == b"RADZ-user"
    assert (lib / "_assets_bins.json").exists()
    assert not (lib / "start.json").exists(), "shipped templates are not user content"
    assert (lib / "official" / "Local" / "start.rad").read_bytes() == b"RADZ-start"
    assert (package / "HDR_Test" / "SH010_v001.rad").exists(), "the originals are left in place"


def test_migration_runs_once_and_never_overwrites(package, user_dir):
    lib = ud.library_dir(package)
    (lib / "HDR_Test" / "SH010_v001.rad").write_bytes(b"RADZ-edited-in-new-place")
    (package / "HDR_Test" / "SH020_v001.rad").write_bytes(b"RADZ-later")
    ud.library_dir(package)
    assert (lib / "HDR_Test" / "SH010_v001.rad").read_bytes() == b"RADZ-edited-in-new-place"
    assert not (lib / "HDR_Test" / "SH020_v001.rad").exists(), "migration is a one-time move"


def test_official_cards_follow_the_package(package, user_dir):
    lib = ud.library_dir(package)
    official = lib / "official"
    assert ud.sync_official(package / "official", official) is False, "unchanged package: no work"
    (package / "official" / "Local" / "start.rad").write_bytes(b"RADZ-start-4.0.1")
    (package / "official" / "API" / "veo.rad").unlink()
    assert ud.sync_official(package / "official", official) is True
    assert (official / "Local" / "start.rad").read_bytes() == b"RADZ-start-4.0.1"
    assert not (official / "API" / "veo.rad").exists(), "a card no longer shipped is removed"
    assert json.loads((official / ud.OFFICIAL_MANIFEST).read_text(encoding="utf-8"))


def test_gizmos_move_too(tmp_path, user_dir):
    pkg = tmp_path / "pkg" / "gizmos"
    pkg.mkdir(parents=True)
    (pkg / "bloom.gizmo").write_text('{"name": "bloom"}', encoding="utf-8")
    target = ud.gizmos_dir(pkg)
    assert target == user_dir / "gizmos"
    assert (target / "bloom.gizmo").exists() and (pkg / "bloom.gizmo").exists()


def test_the_escape_hatch_keeps_data_in_the_package(package, user_dir, monkeypatch):
    monkeypatch.setenv("RADIANCE_USER_DATA_IN_PACKAGE", "1")
    assert ud.library_dir(package) == package
