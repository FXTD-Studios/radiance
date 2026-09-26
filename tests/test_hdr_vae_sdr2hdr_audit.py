"""3.5.0 audit of HDR VAE Decode and SDR -> HDR (Universal / Recover / Encode).

Pins, against OpenColorIO and the ITU documents, every defect this audit fixed:

* Rec.709 -> ACES2065-1 matrix off by 1.2 % (hdr/vae.py and color/ops.py)
* one reference-white convention package wide: linear 1.0 = 203 nits
  (Universal / Recover Linear used 1.0 = 100 nits, so re-encoding their EXR
  with HDR Encode or Write doubled the brightness)
* HLG: reference white at 75 %, the BT.2100 1000-nit transcode OCIO uses
  (HDR Encode put reference white at 100 % and clipped every highlight)
* radiance_meta surviving a sampler (Auto log-inverted diffused latents)
* camera log targets carrying their camera gamut, not Rec.709
"""
import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not isinstance(getattr(torch, "__version__", None), str):
    pytest.skip("torch not installed", allow_module_level=True)

from radiance.color import encodings as E  # noqa: E402
from radiance.color import ops  # noqa: E402
from radiance.hdr import decode_meta as dm  # noqa: E402
from radiance.nodes.hdr import uplift_universal as uu  # noqa: E402

try:
    import PyOpenColorIO as OCIO  # noqa: N814
except Exception:  # noqa: BLE001
    OCIO = None

_CKPT_CANDIDATES = [
    os.environ.get("RADIANCE_SDR2HDR_PIXEL", ""),
    "/mnt/user-data/uploads/ComfyUI/models/radiance/sdr2hdr_pixel_image.pt",
]
CKPT = next((p for p in _CKPT_CANDIDATES if p and Path(p).is_file()), None)

PQ_203 = 0.580688   # ST.2084 code value of 203 cd/m²


def _uni(img, **kw):
    args = dict(inverse_oetf="sRGB", peak_nits=1000.0, knee_mode="manual", knee=0.75,
                shoulder_gamma=2.0, temporal_smoothing=0.0, output_encoding="Linear",
                processing_mode="Expand")
    args.update(kw)
    return uu.RadianceSDRToHDRUniversal().convert(img, **args)


def _ocio_builtin(name):
    p = OCIO.Config.CreateRaw().getProcessor(OCIO.BuiltinTransform(name))
    return p.getDefaultCPUProcessor()


# ── matrices ─────────────────────────────────────────────────────────────────

def test_ap0_matrices_match_ocio_and_are_inverses():
    ref = E.gamut_matrix("Rec.709", "AP0")
    assert np.allclose(ops.M_REC709_TO_ACES2065_1.numpy(), ref, atol=1e-6)
    assert np.allclose(ops.M_ACES2065_1_TO_REC709.numpy(), E.gamut_matrix("AP0", "Rec.709"), atol=1e-6)
    prod = ops.M_REC709_TO_ACES2065_1 @ ops.M_ACES2065_1_TO_REC709
    assert torch.allclose(prod, torch.eye(3), atol=1e-6)


def test_vae_ap0_matrices_are_the_corrected_ones():
    import radiance.hdr.vae as vae
    assert np.allclose(vae._REC709_TO_AP0.T.numpy(), E.gamut_matrix("Rec.709", "AP0"), atol=1e-6)
    assert np.allclose(vae._AP0_TO_REC709.T.numpy(), E.gamut_matrix("AP0", "Rec.709"), atol=1e-6)


# ── HLG / PQ reference white ─────────────────────────────────────────────────

def test_hlg_transcode_puts_reference_white_at_75_percent():
    v = ops.linear_to_hlg_bt2100(torch.ones(1, 3))
    assert float(v[0, 0]) == pytest.approx(0.75, abs=5e-4)
    grey = ops.linear_to_hlg_bt2100(torch.full((1, 3), 0.18))
    assert float(grey[0, 0]) == pytest.approx(0.436, abs=2e-3)


@pytest.mark.skipif(OCIO is None, reason="OpenColorIO not installed")
def test_hlg_and_pq_match_ocio_rec2100_displays():
    """OCIO display-linear: XYZ with 1.0 = 100 nits. Radiance: 1.0 = 203 nits."""
    to_xyz = torch.tensor(E.rgb_to_xyz("Rec.2020"), dtype=torch.float32)
    rng = np.random.default_rng(3)
    lin = torch.tensor(rng.uniform(0.0, 4.5, (64, 3)), dtype=torch.float32)
    xyz = (lin * 2.03) @ to_xyz.T
    hlg_ocio = _ocio_builtin("DISPLAY - CIE-XYZ-D65_to_REC.2100-HLG-1000nit")
    pq_ocio = _ocio_builtin("DISPLAY - CIE-XYZ-D65_to_REC.2100-PQ")
    ref_hlg = np.array([hlg_ocio.applyRGB(list(map(float, r))) for r in xyz.numpy()])
    ref_pq = np.array([pq_ocio.applyRGB(list(map(float, r))) for r in xyz.numpy()])
    got_hlg = ops.linear_to_hlg_bt2100(lin).numpy()
    got_pq = ops.linear_to_pq_bt2408(lin, peak_nits=10000.0).numpy()
    ok = ref_hlg.max(axis=1) < 0.999          # OCIO and ours both clip at 1.0
    assert np.abs(got_hlg[ok] - ref_hlg[ok]).max() < 2e-3
    assert np.abs(got_pq - ref_pq).max() < 1e-3


def test_hlg_numpy_and_torch_agree_and_round_trip():
    x = np.random.default_rng(1).uniform(0.0, 4.0, (256, 3)).astype(np.float32)
    code_np, _ = E.encode(x, "HLG (BT.2100)", "Linear Rec.2020")
    code_t = ops.linear_to_hlg_bt2100(torch.tensor(x)).numpy()
    assert np.abs(code_np - code_t).max() < 1e-4
    back = ops.hlg_bt2100_to_linear(torch.tensor(code_t)).numpy()
    ok = code_t.max(axis=1) < 0.999
    assert np.abs(back[ok] - x[ok]).max() < 2e-3


def test_hdr_encode_hlg_and_pq_place_reference_white_by_bt2408():
    from radiance.nodes.hdr.delivery import RadianceHDREncode
    white = torch.ones(1, 2, 2, 3)
    (pq,) = RadianceHDREncode().encode(white, "PQ (HDR10)")
    (hlg,) = RadianceHDREncode().encode(white, "HLG (Broadcast)")
    assert float(pq[0, 0, 0, 0]) == pytest.approx(PQ_203, abs=1e-4)
    assert float(hlg[0, 0, 0, 0]) == pytest.approx(0.75, abs=5e-4)
    (bright,) = RadianceHDREncode().encode(white * 3.0, "HLG (Broadcast)")
    assert float(bright.max()) < 1.0, "a 609-nit highlight must not clip in HLG"


# ── Universal / Recover output convention ────────────────────────────────────

def test_universal_linear_sdr_white_is_one_and_pq_is_203_nits():
    white = torch.ones(1, 2, 2, 3)
    lin = _uni(white)[0]
    pq = _uni(white, output_encoding="PQ (HDR10)")[0]
    assert float(lin.max()) == pytest.approx(1.0, abs=1e-4)
    assert float(pq.max()) == pytest.approx(PQ_203, abs=1e-4)


@pytest.mark.parametrize("enc,hdr_fmt", [("PQ (HDR10)", "PQ (HDR10)"), ("HLG", "HLG (Broadcast)")])
def test_universal_linear_reencodes_to_its_own_delivery_output(enc, hdr_fmt):
    """The same picture, whether Universal encodes it or HDR Encode does."""
    from radiance.nodes.hdr.delivery import RadianceHDREncode
    ramp = torch.linspace(0.0, 1.0, 64).view(1, 1, 64, 1).repeat(1, 2, 1, 3)
    ramp[..., 0] *= 0.8
    direct = _uni(ramp, output_encoding=enc)[0]
    linear = _uni(ramp)[0]
    (via,) = RadianceHDREncode().encode(linear, hdr_fmt, apply_bt2020=True)
    assert torch.allclose(direct, via, atol=2e-4)


def test_universal_ap0_is_the_linear_output_in_ap0():
    img = torch.rand(1, 4, 4, 3)
    lin = _uni(img)[0]
    ap0 = _uni(img, output_encoding="Linear ACES2065-1 (AP0)")[0]
    assert torch.allclose(ap0, ops.apply_matrix_3x3(lin, ops.M_REC709_TO_ACES2065_1), atol=1e-6)


def test_recover_uses_the_same_convention(monkeypatch):
    import radiance.pixel_sdr2hdr as px
    monkeypatch.setattr(px, "resolve_pixel_checkpoint", lambda p="": Path("/tmp/x.pt"))
    monkeypatch.setattr(px, "predict_pixel_sdr2hdr",
                        lambda srgb, **k: torch.full_like(srgb[..., :3], 0.05))
    img = torch.full((1, 2, 2, 3), 0.5)
    rec = uu.RadianceSDRToHDRRecover()
    out = rec.recover(img, "None", 1000.0, 0.98, 0.001, 1.0, 1.0, "Linear")[0]
    # unclipped pixel: SDR display level (0.5 * 100 nits) in 203-nit units
    assert float(out[0, 0, 0, 0]) == pytest.approx(0.5 / 2.03, abs=1e-5)
    pq = rec.recover(img, "None", 1000.0, 0.98, 0.001, 1.0, 1.0, "PQ (HDR10)")[0]
    assert float(pq[0, 0, 0, 0]) == pytest.approx(
        float(ops.linear_to_pq_bt2408(torch.tensor([0.5 / 2.03]))[0]), abs=1e-5)


def test_report_states_the_output_unit():
    report = _uni(torch.ones(1, 2, 2, 3))[5]
    assert "linear 1.0 = 203 nits" in report


# ── latent fingerprint ───────────────────────────────────────────────────────

def test_fingerprint_matches_only_the_encoded_latent():
    lat = torch.randn(1, 4, 8, 8)
    meta = {"hdr_mode": "Compress (Log)", "pad_h": 8, "latent_fingerprint": dm.latent_fingerprint(lat)}
    assert dm.radiance_meta_is_live(meta, lat)
    assert dm.radiance_meta_is_live(meta, lat.clone())
    assert not dm.radiance_meta_is_live(meta, lat * 0.9 + 0.1 * torch.randn_like(lat))
    assert not dm.radiance_meta_is_live({"hdr_mode": "Compress (Log)"}, lat)   # unstamped
    samples, live = dm.verify_radiance_meta({"samples": torch.randn(1, 4, 8, 8),
                                             "radiance_meta": meta})
    assert not live
    assert "hdr_mode" not in samples["radiance_meta"]
    assert samples["radiance_meta"]["pad_h"] == 8
    assert meta["hdr_mode"] == "Compress (Log)", "caller's dict must not be mutated"


# ── engine: camera gamut on log targets ──────────────────────────────────────

def _engine():
    from radiance.hdr.vae import RadianceVAE4KDecode
    return RadianceVAE4KDecode()


def test_log_target_applies_camera_gamut_like_ocio():
    eng = _engine()
    srgb = torch.tensor([[[[0.9, 0.2, 0.1], [0.5, 0.5, 0.5]]]])
    lin = eng._vae_output_to_target(srgb, "Linear", "Clip (SDR)", 0.0, False, 12.0,
                                    source_space="sRGB", hdr_output=True, display_tonemap="None")
    for target in ("ARRI LogC4", "ARRI LogC3", "Sony S-Log3", "Panasonic V-Log",
                   "DaVinci Intermediate", "RED Log3G10"):
        got = eng._vae_output_to_target(srgb, target, "Clip (SDR)", 0.0, False, 12.0,
                                        source_space="sRGB", hdr_output=True,
                                        display_tonemap="None")
        ref, _ = E.encode(lin[0, 0].numpy(), target, "Linear Rec.709 (sRGB)")
        assert np.abs(got[0, 0].numpy() - ref).max() < 2e-5, target


def test_compress_log_round_trip_to_its_own_log_space_is_exact():
    """Working gamut == target gamut: no matrix, so the round trip stays exact."""
    from radiance.hdr.vae import RadianceVAE4KEncode
    eng = _engine()
    code = torch.tensor([[[[0.2, 0.45, 0.7], [0.9, 0.3, 0.1]]]])
    lin = RadianceVAE4KEncode._get_log_to_linear()["ARRI LogC4"](code)
    log_in = RadianceVAE4KEncode._get_linear_to_log()["ARRI LogC4"](lin)
    out = eng._vae_output_to_target(log_in, "ARRI LogC4", "Compress (Log)", 0.0, False, 12.0,
                                    source_space="ARRI LogC4", hdr_output=True,
                                    display_tonemap="None", working_gamut="ARRI Wide Gamut 4")
    assert torch.allclose(out, code, atol=5e-3)   # shoulder/denoise are near-identity here


def test_encode_converts_a_log_source_to_rec709_for_sdr_modes():
    from radiance.hdr.vae import RadianceVAE4KEncode
    enc = RadianceVAE4KEncode()
    code = torch.tensor([[[[0.5, 0.35, 0.3]]]])
    sdr = enc._prepare_for_vae(code, "ARRI LogC4", 0.0, "Clip (SDR)")
    lin_awg4 = RadianceVAE4KEncode._get_log_to_linear()["ARRI LogC4"](code)
    ref709 = lin_awg4[0, 0].numpy() @ E.gamut_matrix("ARRI Wide Gamut 4", "Rec.709").T
    from radiance.color.transfer import tensor_linear_to_srgb
    ref = tensor_linear_to_srgb(torch.tensor(ref709).clamp(min=0)).clamp(0, 1)
    assert torch.allclose(sdr[0, 0, 0], ref.float(), atol=1e-5)
    assert dm._encode_working_gamut("ARRI LogC4", "Compress (Log)") == "ARRI Wide Gamut 4"
    assert dm._encode_working_gamut("ARRI LogC4", "Clip (SDR)") == "Rec.709"


# ── wrapper: batch vs video ──────────────────────────────────────────────────

def test_pixel_path_batch_mode_follows_the_latent_not_the_batch(monkeypatch):
    from radiance.nodes.generate.engine import RadianceHDRVAEDecode
    seen = []

    def fake_convert(self, image, *a, **kw):
        seen.append(kw["batch_mode"])
        return (image[..., :3], None, None, None, None, "path: x\nlearned recovery: applied")

    monkeypatch.setattr(uu.RadianceSDRToHDRUniversal, "convert", fake_convert)
    RadianceHDRVAEDecode._pixel_hdr(torch.rand(4, 2, 2, 3), 1000.0, is_video=False)
    RadianceHDRVAEDecode._pixel_hdr(torch.rand(4, 2, 2, 3), 1000.0, is_video=True)
    assert seen == ["Independent Images", "Video Frames"]


# ── the real pixel checkpoint ────────────────────────────────────────────────

@pytest.mark.skipif(CKPT is None, reason="sdr2hdr_pixel_image.pt not available")
def test_real_checkpoint_recovers_clipped_highlights_within_the_peak(monkeypatch):
    monkeypatch.setenv("RADIANCE_SDR2HDR_PIXEL", CKPT)
    torch.manual_seed(0)
    h = w = 96
    yy, xx = torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w), indexing="ij")
    img = torch.stack([0.2 + 0.6 * xx, 0.25 + 0.5 * yy, 0.3 + 0.2 * xx * yy], dim=-1)
    sun = ((xx - 0.7) ** 2 + (yy - 0.3) ** 2) < 0.02
    img[sun] = 1.0
    img = img.unsqueeze(0).clamp(0, 1)
    for peak in (1000.0, 4000.0):
        out, hmask, _, hconf, _, report = _uni(img, processing_mode="Hybrid", peak_nits=peak,
                                               knee_mode="adaptive")
        assert "learned recovery: applied" in report, report
        assert torch.isfinite(out).all()
        y = uu._luma(out)
        assert float(y.max()) <= peak / 203.0 + 1e-3, "exceeded the mastering peak"
        assert float(y[0][sun].mean()) > 1.5, "clipped sun was not reconstructed above SDR white"
        expand = _uni(img, processing_mode="Expand", peak_nits=peak, knee_mode="adaptive")[0]
        untouched = hmask[0] < 1e-6
        assert torch.allclose(out[0][untouched], expand[0][untouched], atol=1e-5)
