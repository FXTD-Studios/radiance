"""Concurrent writes with overwrite off get distinct, intact files (package audit R1).

The writer chose a free name by looking at the disk only, so two writes that
resolved their paths before either produced a file both picked the same name,
both reported success, and the second replaced the first. Names are now
claimed while a write runs. These tests hold both writes at a barrier after
their paths are chosen, which is the interleaving that lost a file before.
"""

import threading

import numpy as np
import pytest
import torch
from PIL import Image

from radiance.io import writer as W
from radiance.io.writer import write_frames


def _run_two(fn_a, fn_b):
    out, errors = {}, []

    def run(key, fn):
        try:
            out[key] = fn()
        except Exception as exc:  # noqa: BLE001 - reported below
            errors.append(exc)
    threads = [threading.Thread(target=run, args=("a", fn_a)), threading.Thread(target=run, args=("b", fn_b))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert not errors, errors
    return out


@pytest.fixture
def held_after_naming(monkeypatch):
    """Both writes pass path selection, then wait for each other before writing."""
    barrier = threading.Barrier(2, timeout=20)
    real_pil = W._save_pil_image

    def save_pil(*args, **kwargs):
        barrier.wait()
        return real_pil(*args, **kwargs)
    monkeypatch.setattr(W, "_save_pil_image", save_pil)
    return barrier


def _plate(value):
    return torch.full((1, 4, 4, 3), value)


def test_two_still_writes_get_two_files(tmp_path, held_after_naming):
    def write(value):
        return lambda: write_frames(image=_plate(value), output_path=str(tmp_path), format="IMG │ PNG (8-bit)",
                                    filename="Shot", version=1, color_space="Linear (pass-through)",
                                    overwrite=False)
    out = _run_two(write(0.1), write(0.8))
    paths = {out["a"][0], out["b"][0]}
    assert len(paths) == 2, f"both writes reported {paths}"
    pngs = sorted(tmp_path.glob("*.png"))
    assert [p.name for p in pngs] == ["Shot_v0001.png", "Shot_v0001_001.png"]
    values = sorted(int(np.asarray(Image.open(p))[0, 0, 0]) for p in pngs)
    assert values == [26, 204], f"a write was lost or mixed: {values}"


def test_two_sequence_writes_get_two_stems(tmp_path, held_after_naming):
    def write(value):
        return lambda: write_frames(image=torch.full((2, 4, 4, 3), value), output_path=str(tmp_path),
                                    format="SEQ │ PNG (8-bit)", filename="Plate", version=1,
                                    color_space="Linear (pass-through)", overwrite=False)
    _run_two(write(0.1), write(0.8))
    dirs = sorted(p for p in tmp_path.iterdir() if p.is_dir())
    names = sorted(f.name for d in dirs for f in d.glob("*.png"))
    assert len(names) == 4, names
    stems = {n.rsplit("_", 1)[0] for n in names}
    assert len(stems) == 2, f"the two sequences share a stem: {names}"
    by_stem = {}
    for d in dirs:
        for f in d.glob("*.png"):
            by_stem.setdefault(f.name.rsplit("_", 1)[0], set()).add(int(np.asarray(Image.open(f))[0, 0, 0]))
    assert sorted(len(v) for v in by_stem.values()) == [1, 1], f"frames of the two writes are mixed: {by_stem}"


def test_a_claim_is_released_when_the_write_ends(tmp_path):
    base = str(tmp_path / "Clip")
    with W._reservation_scope():
        first = W.resolve_output_path(base, ".mov", False)
        seen = {}
        t = threading.Thread(target=lambda: seen.setdefault("other", _resolve_in_scope(base)))
        t.start(); t.join(10)
    assert first.name == "Clip.mov"
    assert seen["other"].name == "Clip_001.mov", "a second write took the name the first one holds"
    # Nothing was written and the claims ended: the name is free again.
    assert W.resolve_output_path(base, ".mov", False).name == "Clip.mov"
    assert not W._RESERVED


def _resolve_in_scope(base):
    with W._reservation_scope():
        return W.resolve_output_path(base, ".mov", False)


def test_a_failed_write_releases_its_name(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(W, "_save_pil_image", boom)
    with pytest.raises(OSError):
        write_frames(image=_plate(0.5), output_path=str(tmp_path), format="IMG │ PNG (8-bit)",
                     filename="Shot", version=1, color_space="Linear (pass-through)", overwrite=False)
    assert not W._RESERVED
    monkeypatch.undo()
    path, _ = write_frames(image=_plate(0.5), output_path=str(tmp_path), format="IMG │ PNG (8-bit)",
                           filename="Shot", version=1, color_space="Linear (pass-through)", overwrite=False)
    assert path.endswith("Shot_v0001.png")


def test_overwrite_on_is_unchanged(tmp_path):
    for value in (0.1, 0.8):
        write_frames(image=_plate(value), output_path=str(tmp_path), format="IMG │ PNG (8-bit)",
                     filename="Shot", version=1, color_space="Linear (pass-through)", overwrite=True)
    assert [p.name for p in tmp_path.glob("*.png")] == ["Shot_v0001.png"]


def test_two_deliveries_of_one_shot_get_two_versions(tmp_path):
    from radiance.delivery.handler import claim_next_version, release_version
    (tmp_path / "Shot_v0001.mov").write_bytes(b"x")
    a = claim_next_version(str(tmp_path), "Shot")
    b = claim_next_version(str(tmp_path), "Shot")
    assert (a, b) == ("v0002", "v0003")
    release_version(str(tmp_path), "Shot", a)
    release_version(str(tmp_path), "Shot", b)
    assert claim_next_version(str(tmp_path), "Shot") == "v0002"
    release_version(str(tmp_path), "Shot", "v0002")
