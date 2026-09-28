"""The RHDR export's path-traversal guard has to actually run.

hdr/vae.py imported safe_join as `.path_utils`, a module that does not exist
under hdr/, so the ImportError fallback to os.path.join always ran: a prefix
like "../x" wrote outside the output folder.
"""
import os

import numpy as np
import pytest

pytestmark = pytest.mark.real_torch

vae = pytest.importorskip("radiance.hdr.vae")


def _save(out_dir, prefix):
    img = np.ones((4, 4, 3), dtype=np.float32)
    return vae.RadianceVAE4KDecode._save_rhdr(img, str(out_dir), prefix=prefix)


def test_a_plain_prefix_writes_inside_the_folder(tmp_path):
    name = _save(tmp_path, "radiance_4k")
    assert name and (tmp_path / name).is_file()


def test_a_traversing_prefix_writes_nothing_outside(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    assert _save(out, "../escaped") is None
    assert not any(p.name.startswith("escaped") for p in tmp_path.iterdir())
    assert os.listdir(out) == []
