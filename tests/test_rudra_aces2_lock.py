"""The lock (stage 6b) RUDRA ACES 2.0 is released with, against RUDRA's oracle.

fixtures/rudra_aces2/lock_parity_v1.bin is RUDRA's own golden (Apache 2.0,
native/tests/golden/aces_lock/, written by RUDRA's tools/make_lock_golden.py from
training/aces_partial_tighten.py): a made-up frame through a non-separable toy
view, the oracle's dequantized target, and its results for the registered arm
(strength 0.5, dequantized), the full lock and the integer-code target. The
native lock is held to the same file. No footage, no model.
"""
import importlib
import json
import struct
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")
torch = pytest.importorskip("torch")

GOLDEN = Path(__file__).parent / "fixtures" / "rudra_aces2" / "lock_parity_v1.bin"


def _mod():
    return importlib.import_module("radiance.model.rudra_aces2")


class MixingView:
    """RUDRA's toy view: a channel-mixing matrix, then y / (1 + y)."""
    M = np.array([[0.90, 0.08, 0.02], [0.02, 0.96, 0.02], [0.01, 0.04, 0.95]], np.float64)

    def forward(self, x):
        y = np.maximum(np.asarray(x, np.float64), 0) @ self.M.T
        return (y / (1 + y)).astype(np.float32)


def _golden():
    data = GOLDEN.read_bytes()
    assert data[:4] == b"RLK1"
    pos = 4

    def take(fmt):
        nonlocal pos
        v = struct.unpack_from(fmt, data, pos)
        pos += struct.calcsize(fmt)
        return v

    def arr(dt, shape):
        nonlocal pos
        a = np.frombuffer(data, np.dtype(dt), int(np.prod(shape)), pos).reshape(shape)
        pos += a.nbytes
        return a

    h, w, n = take("<iii")
    g = {"source": arr("<f4", (h, w, 3)), "constrained": arr("<f4", (h, w, 3)), "weight": arr("<f4", (h, w)),
         "dq": arr("<f8", (h, w, 3)), "used": arr("u1", (h, w, 3)).astype(bool), "cases": []}
    for _ in range(n):
        strength, dequant, changed, kept, raised, dq_ch = take("<diiiii")
        g["cases"].append({"strength": strength, "dequant": bool(dequant),
                           "meta": (changed, kept, raised, dq_ch), "result": arr("<f4", (h, w, 3))})
    assert pos == len(data)
    return g


def test_dequantized_target_matches_rudra():
    m, g = _mod(), _golden()
    dq, used = m.dequantize_recorded(g["source"], region=m.partial_support(g["source"], g["weight"]))
    assert np.array_equal(used, g["used"])
    np.testing.assert_allclose(dq, g["dq"], rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("case", [0, 1, 2], ids=["registered", "full-lock", "integer-codes"])
def test_lock_matches_rudra(case):
    m, g = _mod(), _golden()
    c = g["cases"][case]
    out, d = m.lock_observed(g["source"], g["constrained"], g["weight"], MixingView(), strength=c["strength"],
                             iters=4, target="dequant" if c["dequant"] else "source")
    assert (d["changed_pixels"], d["kept_pixels"], d["strength_raised_pixels"], d["dequant_channels"]) == c["meta"]
    np.testing.assert_allclose(out, c["result"], rtol=1e-6, atol=1e-9)


def test_the_lock_keeps_the_contract():
    """Outside partial support bit-exact; every recorded channel within 1 code of the
    source and every clipped channel still clipped where it changed a pixel."""
    m, g = _mod(), _golden()
    src, cur, w = g["source"], g["constrained"], g["weight"]
    out, _ = m.lock_observed(src, cur, w, MixingView())
    sup = m.partial_support(src, w)
    assert np.array_equal(out[~sup], cur[~sup])
    changed = (out != cur).any(-1)
    disp = MixingView().forward(out[changed])
    s = src[changed]
    rec = s < m.FULL_CLIP
    assert (np.abs(np.where(rec, disp - s, 0)) * 255 <= 1.0 + 1e-4).all()
    assert (np.where(~rec, disp, 1.0) >= m.FULL_CLIP - 1e-6).all()


def _manifest(major=3, lock=True, **lock_kw):
    m = _mod()
    out = {"contract": m.CONTRACT, "contract_major": major,
           "ocio": {"config": m.OCIO_CONFIG, "view": m.OCIO_VIEW, "display": m.OCIO_DISPLAY}}
    if lock:
        out["lock"] = {"name": "lock_dqs050g", "mode": "lock", "strength": 0.5, "target": "dequant",
                       "iters": 4, "band_codes": 0.0, "dq_iters": 200, "dq_max_lap": 1.5, **lock_kw}
    return out


def test_manifest_major_3_needs_its_lock_and_major_2_has_none():
    m = _mod()
    assert m.lock_settings(_manifest()) == {"strength": 0.5, "target": "dequant", "iters": 4,
                                            "dq_iters": 200, "dq_max_lap": 1.5, "name": "lock_dqs050g"}
    assert m.lock_settings(_manifest(major=2, lock=False)) is None
    m._check_manifest(_manifest())
    m._check_manifest(_manifest(major=2, lock=False))
    with pytest.raises(ValueError, match="lock"):
        m._check_manifest(_manifest(major=3, lock=False))
    with pytest.raises(ValueError, match="lock"):
        m._check_manifest(_manifest(major=2, lock=True))


@pytest.mark.parametrize("bad", [{"mode": "bisect"}, {"band_codes": 2.0}, {"target": "x"}, {"iters": 9},
                                 {"strength": 0.0}, {"strength": 1.5}])
def test_a_lock_this_build_does_not_implement_is_refused(bad):
    with pytest.raises(ValueError, match="lock"):
        _mod()._check_manifest(_manifest(**bad))


def test_the_published_manifest_is_accepted():
    """The 2.0 package's manifest as published (Hugging Face fxtdstudios/rudra-models
    @ab8da1b2, models/RUDRA_ACES_2/manifest.json): its contract, view and lock."""
    m = _mod()
    pub = json.loads((Path(__file__).parent / "fixtures" / "rudra_aces2" / "manifest_2.0.json").read_text())
    m._check_manifest(pub)
    assert m.lock_settings(pub)["name"] == "lock_dqs050g"


# ── the 2.0 package on the hub, and an installed 1.0 ─────────────────────────

def test_the_aces2_package_is_2_0_at_its_publishing_commit():
    dl = importlib.import_module("radiance.model.rudra_download")
    assert dl.ACES2.revision == "ab8da1b2db74099eb45cc39585df1d2edf7b3e34"
    files = {rel: (size, digest) for rel, size, digest in dl.ACES2.files}
    assert files["rudra_g1.ts"][1].startswith("7082eb869d265d6e")      # RUDRA docs/MODELS.md
    assert files["rudra_g1.onnx"][1].startswith("880e7b698a099b8d")
    pub = json.loads((Path(__file__).parent / "fixtures" / "rudra_aces2" / "manifest_2.0.json").read_text())
    assert pub["g1_sha256"] == files["rudra_g1.ts"][1] and pub["onnx_sha256"] == files["rudra_g1.onnx"][1]


def _small_package(dl, tmp_path, content: bytes):
    import hashlib
    return dl.Package(name="t", title="T", repo_dir="", default_allowed=True, size_mb=0,
                      files=(("g1.ts", len(content), hashlib.sha256(content).hexdigest()),))


def test_a_same_size_file_with_another_hash_is_not_installed(tmp_path):
    """RUDRA ACES 2.0's .ts has 1.0's size: an installed 1.0 must not pass for 2.0."""
    dl = importlib.import_module("radiance.model.rudra_download")
    new, old = b"model 2.0 bytes!", b"model 1.0 bytes!"
    pkg = _small_package(dl, tmp_path, new)
    (tmp_path / "g1.ts").write_bytes(old)
    assert not dl.package_complete(tmp_path, pkg)
    (tmp_path / "g1.ts").write_bytes(new)
    assert dl.package_complete(tmp_path, pkg)


def test_the_download_replaces_a_same_size_file_with_another_hash(tmp_path, monkeypatch):
    dl = importlib.import_module("radiance.model.rudra_download")
    new, old = b"model 2.0 bytes!", b"model 1.0 bytes!"
    pkg = _small_package(dl, tmp_path, new)
    (tmp_path / "g1.ts").write_bytes(old)
    fetched = []

    def fake_fetch(url, dest, size=0, report=None):
        fetched.append(url)
        Path(dest).write_bytes(new)

    monkeypatch.setattr(dl, "_fetch", fake_fetch)
    dl.download_to(pkg, tmp_path, log=lambda *_: None)
    assert len(fetched) == 1 and (tmp_path / "g1.ts").read_bytes() == new
    dl.download_to(pkg, tmp_path, log=lambda *_: None)
    assert len(fetched) == 1                                     # verified: not fetched again
