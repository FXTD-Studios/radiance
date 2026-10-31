"""RUDRA ACES 2 (RUDRA 1.0.1's learned ACES 2.0 model) in Radiance.

* the pinned download: sizes, SHA-256, consent;
* stages 4-6 of the staged contract, ported from RUDRA's reference;
* the full pipeline against goldens made by RUDRA's own ``staged_reference``
  with the released package (skipped when the package is not installed);
* the Universal and Recover wiring, with a stand-in network so it runs
  without the package.
"""
import importlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ocio = pytest.importorskip("PyOpenColorIO")
cv2 = pytest.importorskip("cv2")

FIXTURES = Path(__file__).parent / "fixtures" / "rudra_aces2"


def _mod():
    return importlib.import_module("radiance.model.rudra_aces2")


def _dl():
    return importlib.import_module("radiance.model.rudra_download")


def _installed_package():
    try:
        return _mod().resolve_aces2_package("")
    except Exception:  # noqa: BLE001
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  download
# ─────────────────────────────────────────────────────────────────────────────

class TestDownload:
    def test_packages_are_pinned_to_a_commit_and_hashes(self):
        dl = _dl()
        assert len(dl.REVISION) == 40
        for pkg in dl.PACKAGES.values():
            assert pkg.files
            for rel, size, digest in pkg.files:
                assert size > 0 and len(digest) == 64
                assert dl.REVISION in pkg.url(rel)

    def test_aces2_is_small_and_on_by_default_genfill_is_ask_first(self):
        dl = _dl()
        assert dl.ACES2.default_allowed and dl.ACES2.size_mb <= 5
        assert not dl.GENFILL_V3.default_allowed

    def test_refused_when_downloads_are_off(self, monkeypatch, tmp_path):
        dl = _dl()
        dl._reset_for_tests()
        monkeypatch.setenv("RADIANCE_ALLOW_DOWNLOADS", "0")
        called = []
        monkeypatch.setattr(dl, "_fetch", lambda url, dest: called.append(url))
        assert dl.download_package(dl.ACES2) is None
        assert not called

    def _fake_pkg(self, dl, payloads):
        import hashlib
        files = tuple((name, len(data), hashlib.sha256(data).hexdigest()) for name, data in payloads.items())
        return dl.Package("FAKE_PKG", "fake", "models/FAKE", files, 1, True)

    def test_verified_download_lands_in_models_radiance(self, monkeypatch, tmp_path):
        dl = _dl()
        dl._reset_for_tests()
        monkeypatch.setenv("RADIANCE_ALLOW_DOWNLOADS", "1")
        payloads = {"a.json": b'{"x": 1}', "sub/b.bin": b"\x00\x01\x02" * 100}
        pkg = self._fake_pkg(dl, payloads)
        paths = importlib.import_module("radiance.model.paths")
        monkeypatch.setattr(paths, "radiance_model_dirs", lambda: [tmp_path])

        def fetch(url, dest):
            rel = url.split("/models/FAKE/", 1)[1]
            Path(dest).write_bytes(payloads[rel])
        monkeypatch.setattr(dl, "_fetch", fetch)
        folder = dl.download_package(pkg)
        assert folder == tmp_path / "FAKE_PKG"
        assert dl.package_complete(folder, pkg, verify_hash=True)
        assert dl.find_package(pkg) == folder

    def test_a_changed_file_is_never_installed(self, monkeypatch, tmp_path):
        dl = _dl()
        dl._reset_for_tests()
        monkeypatch.setenv("RADIANCE_ALLOW_DOWNLOADS", "1")
        pkg = self._fake_pkg(dl, {"a.bin": b"right"})
        paths = importlib.import_module("radiance.model.paths")
        monkeypatch.setattr(paths, "radiance_model_dirs", lambda: [tmp_path])
        monkeypatch.setattr(dl, "_fetch", lambda url, dest: Path(dest).write_bytes(b"wrong"))
        assert dl.download_package(pkg) is None
        assert not (tmp_path / "FAKE_PKG" / "a.bin").exists()
        assert not (tmp_path / "FAKE_PKG" / "a.bin.part").exists()


# ─────────────────────────────────────────────────────────────────────────────
#  stages
# ─────────────────────────────────────────────────────────────────────────────

class TestStages:
    def test_view_is_the_pinned_aces2_sdr_view(self):
        v = _mod().aces2_view()
        assert v.identity["config"] == "cg-config-v4.0.0_aces-v2.0_ocio-v2.5"
        assert v.identity["view"] == "ACES 2.0 - SDR 100 nits (Rec.709)"
        codes = np.random.default_rng(0).random((64, 3)).astype(np.float32) * 0.9 + 0.05
        back = v.forward(v.inverse(codes))
        assert np.abs(back - codes).max() < 1e-4

    def test_matrices_match_ocio(self):
        m = _mod()
        cfg = ocio.Config.CreateFromBuiltinConfig(m.OCIO_CONFIG)
        for dst, mat in (("ACES2065-1", m.M_AP1_TO_AP0), ("Linear Rec.709 (sRGB)", m.M_AP1_TO_REC709)):
            p = cfg.getProcessor("ACEScg", dst).getDefaultCPUProcessor()
            rgb = np.array([[0.2, 0.5, 0.9]], np.float32)
            got = rgb.copy()
            p.applyRGB(got)
            assert np.allclose(got[0], mat @ rgb[0], atol=1e-5)

    def test_smoothing_leaves_everything_outside_full_clip_bit_exact(self):
        m = _mod()
        rng = np.random.default_rng(1)
        src = rng.random((40, 50, 3)).astype(np.float32) * 0.9
        src[10:30, 10:30] = 1.0
        pred = (rng.random((40, 50, 3)) * 20).astype(np.float32)
        out, full = m.smooth_clipped(pred, src)
        assert np.array_equal(out[~full], pred[~full])
        interior = full.copy()
        interior[:13] = interior[27:] = False
        interior[:, :13] = interior[:, 27:] = False
        # flat in the eroded interior: one colour
        assert np.ptp(out[interior], axis=0).max() < 1e-4

    def test_constraint_brings_the_view_within_one_code(self):
        m = _mod()
        v = m.aces2_view()
        src = np.full((4, 4, 3), 0.95, np.float32)
        base = v.inverse(src)
        pred = base * 3.0                         # far too bright for codes below clip
        out, stats = m.constrain_colour(src, pred, base, np.ones((4, 4), np.float32), v)
        assert np.abs(v.forward(out) - src).max() * 255 <= 1.0 + 1e-3
        assert stats["support_pixels"] == 16

    def test_trims_identity_and_strength_zero_is_the_inverse(self):
        m = _mod()
        rng = np.random.default_rng(2)
        master = rng.random((8, 8, 3)).astype(np.float32) * 4
        base = rng.random((8, 8, 3)).astype(np.float32)
        mask = (rng.random((8, 8)) > 0.5).astype(np.float32)
        assert m.apply_trims(master, base, mask) is master
        assert np.allclose(m.apply_trims(master, base, mask, strength=0.0), base, atol=1e-6)

    def test_peak_trim_rolls_off_only_reconstructed_pixels(self):
        m = _mod()
        master = np.full((2, 2, 3), 20.0, np.float32)   # 20 x diffuse white = 4060 nits
        mask = np.array([[1, 0], [1, 0]], np.float32)
        out = m.apply_trims(master, master, mask, peak_nits=1000.0)
        cap = 1000.0 / 203.0
        y = out @ m.LUMA_AP1.astype(np.float32)
        assert (y[:, 0] <= cap + 1e-4).all()
        assert np.allclose(out[:, 1], master[:, 1])

    def test_linear_input_is_encoded_for_a_24_display(self):
        m = _mod()
        x = torch.full((2, 2, 3), 0.25)
        assert np.allclose(m.sdr_codes_from_image(x, "None"), 0.25 ** (1 / 2.4))
        assert np.allclose(m.sdr_codes_from_image(x, "sRGB"), 0.25)


# ─────────────────────────────────────────────────────────────────────────────
#  the released package against RUDRA's reference
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(_installed_package() is None,
                    reason="RUDRA ACES 2 not installed (models/radiance/RUDRA_ACES_2 or RADIANCE_RUDRA_ACES2)")
@pytest.mark.parametrize("name", ["syn", "photo"])
def test_matches_rudra_staged_reference(name):
    m = _mod()
    g = np.load(FIXTURES / f"golden_{name}.npz")
    pkg = m.load_aces2_package(download=False)
    r = m.run_frame(pkg, g["source"])
    assert np.array_equal(r.mask, g["mask"])
    rel = np.abs(r.acescg - g["acescg"]) / (np.abs(g["acescg"]) + 1e-3)
    assert rel.max() <= 1e-5


def test_goldens_keep_the_codes_contract():
    """RUDRA's output on 8-bit codes keeps highlight codes within one. (The
    synthetic probe is not on the 8-bit grid and has saturated near-black
    channels, where the ACES 2.0 inverse itself does not round-trip.)"""
    m = _mod()
    v = m.aces2_view()
    for name in ("photo",):
        g = np.load(FIXTURES / f"golden_{name}.npz")
        hi = (g["mask"] > 0)
        err = np.abs(v.forward(g["acescg"]) - g["source"]).max(-1) * 255
        assert err[hi].max() <= 1.0 + 1e-3


# ─────────────────────────────────────────────────────────────────────────────
#  node wiring with a stand-in network
# ─────────────────────────────────────────────────────────────────────────────

class _FakeG1(torch.nn.Module):
    """restored_source = source, no fields; weight on fully clipped pixels only."""

    def forward(self, source, baseline):
        w = (source.amin(1, keepdim=True) >= 254.5 / 255).float()
        z = torch.zeros_like(source)
        return source, z + 0.5, z + 1.0, w


@pytest.fixture
def fake_package(monkeypatch):
    m = _mod()
    pkg = m.ACES2Package(folder=Path("."), manifest={}, g1=_FakeG1(), device=torch.device("cpu"),
                         forbid_new_negative_y=True,
                         smoothing={"name": "hybrid64", "sigma": 64.0, "ramp": 1.0, "start": 0.0,
                                    "mode": "hybrid"},
                         identity={"package": "fake"})
    monkeypatch.setattr(m, "load_aces2_package", lambda *a, **k: pkg)
    monkeypatch.setattr(m, "resolve_aces2_package", lambda *a, **k: Path("."))
    return pkg


def _img():
    rng = np.random.default_rng(3)
    x = rng.random((1, 24, 32, 3)).astype(np.float32) * 0.8
    x[0, 5:15, 5:15] = 1.0
    return torch.from_numpy(np.rint(x * 255) / 255).float()


def test_universal_runs_aces2_and_reports_it(fake_package):
    uu = importlib.import_module("radiance.nodes.hdr.uplift_universal")
    m = _mod()
    img = _img()
    out, hi, sh, hc, sc, report, recon = uu.RadianceSDRToHDRUniversal().convert(
        img, "sRGB", 1000.0, "adaptive", 0.75, 2.0, 0.85, "Linear ACES2065-1 (AP0)",
        learned_backend="RUDRA ACES 2")
    assert "path: RUDRA ACES 2" in report and "learned recovery: applied" in report
    assert recon.shape == (1, 24, 32) and float(recon.max()) == 1.0
    # Outside the reconstructed mask the master is the plain ACES 2.0 inverse.
    base = m.aces2_view().inverse(img[0].numpy())
    expect = m.acescg_to(base, m.M_AP1_TO_AP0)
    keep = recon[0].numpy() == 0
    assert np.abs(out[0].numpy()[keep] - expect[keep]).max() < 1e-4


def test_universal_auto_prefers_aces2_when_installed(fake_package):
    uu = importlib.import_module("radiance.nodes.hdr.uplift_universal")
    *_, report, _ = uu.RadianceSDRToHDRUniversal().convert(
        _img(), "sRGB", 1000.0, "adaptive", 0.75, 2.0, 0.85, "Linear")
    assert "RUDRA ACES 2" in report


def test_universal_auto_without_aces2_keeps_the_old_path(monkeypatch):
    uu = importlib.import_module("radiance.nodes.hdr.uplift_universal")
    monkeypatch.setattr(uu, "_aces2_available", lambda *a: False)
    *_, report, recon = uu.RadianceSDRToHDRUniversal().convert(
        _img(), "sRGB", 1000.0, "adaptive", 0.75, 2.0, 0.85, "Linear")
    assert "RUDRA ACES 2" not in report.splitlines()[1]


def test_expand_never_runs_a_model(fake_package, monkeypatch):
    uu = importlib.import_module("radiance.nodes.hdr.uplift_universal")
    monkeypatch.setattr(uu._RudraRecoveryCore, "_aces2_reconstruct",
                        staticmethod(lambda *a, **k: pytest.fail("model ran in Expand")))
    uu.RadianceSDRToHDRUniversal().convert(_img(), "sRGB", 1000.0, "adaptive", 0.75, 2.0, 0.85,
                                           "Linear", processing_mode="Expand")


def test_recover_runs_aces2_with_six_outputs(fake_package):
    uu = importlib.import_module("radiance.nodes.hdr.uplift_universal")
    res = uu.RadianceSDRToHDRRecover().recover(_img(), "sRGB", 1000.0, 0.98, 0.05, 1.0, 1.0, "PQ (HDR10)")
    assert len(res) == 6
    out = res[0]
    assert float(out.max()) <= 0.7519      # PQ of 1000 nits
    assert torch.equal(res[3], res[5])


def test_recover_explicit_aces2_raises_when_missing(monkeypatch):
    uu = importlib.import_module("radiance.nodes.hdr.uplift_universal")
    m = _mod()

    def missing(*a, **k):
        raise FileNotFoundError("RUDRA ACES 2 is not installed")
    monkeypatch.setattr(m, "load_aces2_package", missing)
    with pytest.raises(RuntimeError, match="RUDRA ACES 2"):
        uu.RadianceSDRToHDRRecover().recover(_img(), "sRGB", 1000.0, 0.98, 0.05, 1.0, 1.0, "Linear",
                                             learned_backend="RUDRA ACES 2")


def test_pq_places_diffuse_white_at_reference_white():
    uu = importlib.import_module("radiance.nodes.hdr.uplift_universal")
    from radiance.color.ops import linear_to_pq_bt2408
    m = _mod()
    white = torch.from_numpy(np.linalg.inv(m.M_AP1_TO_REC709).astype(np.float32) @ np.ones(3, np.float32))
    pq = uu._encode_acescg(white.view(1, 1, 1, 3), "PQ (HDR10)", 10.0, 2.03)
    expect = linear_to_pq_bt2408(torch.ones(1, 1, 1, 3), peak_nits=1000.0, reference_white_nits=203.0)
    assert torch.allclose(pq, expect, atol=1e-4)
