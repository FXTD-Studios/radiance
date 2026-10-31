"""Phase 5: route and storage safety (audit E-07, E-10, E-12).

Pure-Python helpers only, so these run in the light CI lane (no aiohttp, no
torch). The route-level behaviour sits beside them in test_workspace_api.py.
"""
import json
import os
from pathlib import Path

import pytest

from radiance.nodes.pipeline import workspace as ws
from radiance.delivery import handler as dh


# ── E-10: atomic saves that say when they fail ──────────────────────────────

def test_atomic_write_replaces_whole_and_leaves_no_partial(tmp_path):
    target = tmp_path / "bins.json"
    target.write_text("old")
    ws._atomic_write_text(target, "new")
    assert target.read_text() == "new"
    assert [p.name for p in tmp_path.iterdir()] == ["bins.json"]


def test_a_failed_atomic_write_keeps_the_old_file(tmp_path, monkeypatch):
    target = tmp_path / "bins.json"
    target.write_text("old")

    def boom(src, dst):
        raise OSError("disk full")
    monkeypatch.setattr(ws.os, "replace", boom)
    with pytest.raises(OSError):
        ws._atomic_write_text(target, "new")
    assert target.read_text() == "old"
    assert [p.name for p in tmp_path.iterdir()] == ["bins.json"]


def test_save_bins_and_shot_status_raise_instead_of_swallowing(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "_ASSETS_BINS_PATH", tmp_path / "no_such_dir" / "bins.json")
    # parent missing: the write cannot succeed, and the caller must hear about it
    with pytest.raises(OSError):
        ws._save_bins([{"id": "a"}])


def test_corrupt_bins_are_kept_aside_before_defaulting(tmp_path, monkeypatch):
    path = tmp_path / "bins.json"
    path.write_text("{not json")
    monkeypatch.setattr(ws, "_ASSETS_BINS_PATH", path)
    assert ws._load_bins() == []
    kept = [p for p in tmp_path.iterdir() if ".corrupt-" in p.name]
    assert len(kept) == 1 and kept[0].read_text() == "{not json"


def test_valid_bins_load_without_a_copy(tmp_path, monkeypatch):
    path = tmp_path / "bins.json"
    path.write_text(json.dumps([{"id": "a"}]))
    monkeypatch.setattr(ws, "_ASSETS_BINS_PATH", path)
    assert ws._load_bins() == [{"id": "a"}]
    assert [p.name for p in tmp_path.iterdir()] == ["bins.json"]


# ── E-12: decompression and upload limits ──────────────────────────────────

def test_rad_v2_round_trips_normally():
    blob = ws._pack_rad_v2('{"nodes": []}', {"a": 1})
    graph, meta = ws._unpack_rad_v2(blob)
    assert graph == '{"nodes": []}' and meta == {"a": 1}


def test_rad_v2_decompression_bomb_is_refused():
    big = "0" * (ws.MAX_WORKFLOW_SIZE_BYTES + 4096)       # compresses to ~50 KB
    blob = ws._pack_rad_v2(big, {})
    assert len(blob) < 1024 * 1024
    with pytest.raises(ValueError, match="limit"):
        ws._unpack_rad_v2(blob)


def test_upload_limit_defaults_and_reads_the_environment(monkeypatch):
    monkeypatch.delenv("RADIANCE_UPLOAD_MAX_MB", raising=False)
    assert ws._upload_limit_bytes() == 4096 * 1024 * 1024
    monkeypatch.setenv("RADIANCE_UPLOAD_MAX_MB", "10")
    assert ws._upload_limit_bytes() == 10 * 1024 * 1024
    monkeypatch.setenv("RADIANCE_UPLOAD_MAX_MB", "junk")
    assert ws._upload_limit_bytes() == 4096 * 1024 * 1024


# ── E-07: output containment follows symlinks ──────────────────────────────

def test_path_inside_accepts_the_root_and_children(tmp_path):
    root = tmp_path / "output"
    (root / "shot").mkdir(parents=True)
    assert dh._path_inside(str(root), str(root))
    assert dh._path_inside(str(root / "shot" / "new_dir"), str(root))


def test_path_inside_rejects_siblings_and_parents(tmp_path):
    root = tmp_path / "output"
    root.mkdir()
    (tmp_path / "output_evil").mkdir()
    assert not dh._path_inside(str(tmp_path / "output_evil"), str(root))
    assert not dh._path_inside(str(tmp_path), str(root))
    assert not dh._path_inside(str(root / ".." / "x"), str(root))


def test_a_dot_dot_prefixed_folder_is_still_inside(tmp_path):
    root = tmp_path / "output"
    (root / "..shots").mkdir(parents=True)
    assert dh._path_inside(str(root / "..shots"), str(root))


@pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
def test_a_symlink_out_of_the_output_folder_is_rejected(tmp_path):
    root = tmp_path / "output"
    root.mkdir()
    elsewhere = tmp_path / "other_volume"
    elsewhere.mkdir()
    (root / "link").symlink_to(elsewhere, target_is_directory=True)
    assert not dh._path_inside(str(root / "link" / "master"), str(root))
