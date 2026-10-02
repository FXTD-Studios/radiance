"""v3.5.3 P0 regression tests: colour and QC fixes (FIX-001, 003, 004, 005, 007, 014)."""
import json
import inspect

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not hasattr(torch, "einsum"):
    pytest.skip("real torch required", allow_module_level=True)


# ── FIX-004 / FIX-005: White Balance ─────────────────────────────────────────

from radiance.nodes.color.colorspace import (
    RadianceWhiteBalance, _build_bradford_matrix, _M_709_TO_XYZ,
)


def _xyz_white(x, y):
    return np.array([x / y, 1.0, (1 - x - y) / y])


def test_white_balance_accepts_ocio_context():
    """FIX-004: the declared optional input binds without a TypeError."""
    assert "ocio_context" in RadianceWhiteBalance.INPUT_TYPES()["optional"]
    assert "ocio_context" in inspect.signature(RadianceWhiteBalance.apply).parameters
    img = torch.full((1, 4, 4, 3), 0.18)
    out, _ = RadianceWhiteBalance().apply(img, "Manual RGB Gain", ocio_context=None)
    assert out.shape == img.shape


def test_bradford_maps_source_white_to_destination_white_in_xyz():
    """FIX-005: RGB D65 white (1,1,1) must land on the D50 white in XYZ, i.e. the
    matrix is RGB->XYZ->CAT->RGB, not the cone matrix applied to RGB."""
    M = _build_bradford_matrix("D65", "D50").double()
    rgb_white = torch.ones(3, dtype=torch.float64)
    adapted_xyz = (_M_709_TO_XYZ @ (M @ rgb_white)).numpy()
    target = _xyz_white(0.3457, 0.3585)
    np.testing.assert_allclose(adapted_xyz, target, atol=2e-4)


def test_bradford_matches_published_d65_to_d50_xyz_matrix():
    """XYZ-space Bradford D65->D50 against Lindbloom's published matrix."""
    from radiance.nodes.color.colorspace import _bradford_xyz
    M = _bradford_xyz((0.31271, 0.32902), (0.34567, 0.35850)).numpy()
    ref = np.array([[1.0478112, 0.0228866, -0.0501270],
                    [0.0295424, 0.9904844, -0.0170491],
                    [-0.0092345, 0.0150436, 0.7521316]])
    np.testing.assert_allclose(M, ref, atol=5e-4)


def test_white_balance_preserves_alpha():
    img = torch.rand(2, 4, 4, 4)
    out, _ = RadianceWhiteBalance().apply(img, "Illuminant Adapt", src_illuminant="D65", dst_illuminant="A")
    assert out.shape == img.shape
    assert torch.equal(out[..., 3], img[..., 3])
    assert not torch.allclose(out[..., :3], img[..., :3])


def test_white_balance_does_not_modify_input():
    img = torch.rand(1, 4, 4, 3)
    ref = img.clone()
    RadianceWhiteBalance().apply(img, "Temperature / Tint", temperature=3200.0)
    assert torch.equal(img, ref)


# ── FIX-001: ACES 2.0 tone scale reference behaviour ────────────────────────

from radiance.hdr.tonescale import ACES2TonescaleParams, aces2_tonescale_nits
from radiance.nodes.hdr.aces2 import (
    RadianceACES2Tonescale, RadianceACES2OutputTransformFull, RadianceACES2Compliance,
)

ocio = pytest.importorskip("PyOpenColorIO")


def _pq_decode_nits(e):
    m1, m2 = 2610 / 16384, 2523 / 4096 * 128
    c1, c2, c3 = 3424 / 4096, 2413 / 4096 * 32, 2392 / 4096 * 32
    ep = np.power(e, 1 / m2)
    return 10000 * np.power(np.maximum(ep - c1, 0) / (c2 - c3 * ep), 1 / m1)


@pytest.mark.parametrize("peak", [500, 1000, 2000, 4000])
def test_reference_tonescale_matches_ocio_aces2(peak):
    """The reference tone scale equals OCIO's ACES 2.0 Output Transform on neutrals."""
    from radiance.hdr import aces2_ocio
    ok, why = aces2_ocio.reference_available()
    if not ok:
        pytest.skip(why)
    ys = np.array([0.01, 0.18, 1.0, 10.0, 100.0], dtype=np.float32)
    rgb = np.repeat(ys[:, None], 3, axis=1)
    out = aces2_ocio.apply_reference(rgb, "ACEScg", f"ACES 2.0 HDR (Rec.2100 PQ {peak} nits)") if peak != 500 else None
    if out is None:
        cfg = aces2_ocio._config()
        p = cfg.getProcessor("ACEScg", "Rec.2100-PQ - Display", "ACES 2.0 - HDR 500 nits (Rec.2020)",
                             ocio.TRANSFORM_DIR_FORWARD).getDefaultCPUProcessor()
        out = rgb.copy(); p.applyRGB(out)
    nits_ocio = _pq_decode_nits(out[:, 1].astype(np.float64))
    nits_ours = np.array([aces2_tonescale_nits(y, peak) for y in ys])
    np.testing.assert_allclose(nits_ours, nits_ocio, rtol=2e-4, atol=2e-3)


def test_tonescale_node_default_places_grey_per_peak():
    """FIX-001: the node default used to put grey at 10% of peak (100 nits at 1000)."""
    node = RadianceACES2Tonescale()
    img = torch.full((1, 4, 4, 3), 0.18)
    for peak, nits in [(100.0, 10.0), (1000.0, 14.512), (4000.0, 16.824)]:
        out, info = node.apply(img, peak, "luminance_preserving")
        assert float(out.mean()) * peak == pytest.approx(nits, abs=0.01)
        assert "reference" in info


def test_tonescale_creative_grey_is_labelled():
    out, info = RadianceACES2Tonescale().apply(torch.full((1, 2, 2, 3), 0.18), 1000.0,
                                                "luminance_preserving", grey_target=0.10)
    assert float(out.mean()) == pytest.approx(0.10, abs=1e-3)
    assert "not ACES 2.0 reference" in info


def test_tonescale_preserves_alpha():
    img = torch.rand(1, 4, 4, 4) * 2
    out, _ = RadianceACES2Tonescale().apply(img, 1000.0, "per_channel")
    assert torch.equal(out[..., 3], img[..., 3])


# ── FIX-002: Output Transform through the pinned OCIO reference ─────────────

def _need_reference():
    from radiance.hdr import aces2_ocio
    ok, why = aces2_ocio.reference_available()
    if not ok:
        pytest.skip(why)


@pytest.mark.parametrize("ot", [
    "ACES 2.0 SDR (sRGB/Rec.709)", "ACES 2.0 SDR (P3-D65)",
    "ACES 2.0 HDR (Rec.2100 PQ 1000 nits)", "ACES 2.0 HDR (Rec.2100 PQ 4000 nits)",
    "ACES 2.0 HDR (Rec.2100 HLG)", "ACES 2.0 Cinema (DCI-P3 D65)",
])
def test_output_transform_auto_equals_independent_ocio(ot):
    _need_reference()
    from radiance.hdr.aces2_ocio import OUTPUT_VIEWS
    scene = torch.rand(2, 6, 6, 3) * 8.0
    out, info = RadianceACES2OutputTransformFull().transform(scene, "ACEScg", ot)
    assert "ACES 2.0 reference" in info and "APPROXIMATION" not in info
    # Independent path: a fresh config object from the same pinned built-in.
    cfg = ocio.Config.CreateFromFile("ocio://studio-config-v4.0.0_aces-v2.0_ocio-v2.5")
    display, view = OUTPUT_VIEWS[ot]
    cpu = cfg.getProcessor("ACEScg", display, view, ocio.TRANSFORM_DIR_FORWARD).getDefaultCPUProcessor()
    ref = scene.numpy().astype(np.float32).reshape(-1, 3).copy()
    cpu.applyRGB(ref)
    np.testing.assert_allclose(out.numpy().reshape(-1, 3), np.clip(ref, 0, 1), atol=1e-6)


def test_output_transform_neutral_grey_sdr_is_ten_nits():
    _need_reference()
    out, _ = RadianceACES2OutputTransformFull().transform(
        torch.full((1, 1, 1, 3), 0.18), "ACEScg", "ACES 2.0 SDR (sRGB/Rec.709)")
    v = float(out[0, 0, 0, 1])
    lin = ((v + 0.055) / 1.055) ** 2.4
    assert lin * 100 == pytest.approx(10.0, abs=0.01)


def test_output_transform_approximation_is_labelled():
    out, info = RadianceACES2OutputTransformFull().transform(
        torch.rand(1, 4, 4, 3), "ACEScg", "ACES 2.0 SDR (sRGB/Rec.709)", engine="Radiance approximation")
    assert info.startswith("APPROXIMATION")


def test_output_transform_d60_cinema_has_no_reference():
    with pytest.raises(RuntimeError, match="OCIO reference unavailable"):
        RadianceACES2OutputTransformFull().transform(
            torch.rand(1, 4, 4, 3), "ACEScg", "ACES 2.0 Cinema (DCI-P3 D60)", engine="OCIO reference")
    _, info = RadianceACES2OutputTransformFull().transform(
        torch.rand(1, 4, 4, 3), "ACEScg", "ACES 2.0 Cinema (DCI-P3 D60)")
    assert info.startswith("APPROXIMATION")


def test_output_transform_preserves_alpha_and_input():
    _need_reference()
    img = torch.rand(1, 4, 4, 4)
    ref = img.clone()
    out, _ = RadianceACES2OutputTransformFull().transform(img, "ACEScg", "ACES 2.0 SDR (sRGB/Rec.709)")
    assert torch.equal(out[..., 3], img[..., 3])
    assert torch.equal(img, ref)


# ── FIX-003: no unsupported S-2126 claim ────────────────────────────────────

def test_no_s2126_claim_in_node_surface():
    from radiance.nodes.hdr import aces2
    names = aces2.NODE_DISPLAY_NAME_MAPPINGS
    assert "S-2126" not in names["RadianceACES2Compliance"]
    assert "S-2126" not in RadianceACES2Compliance.DESCRIPTION
    assert "RadianceACES2Compliance" in aces2.NODE_CLASS_MAPPINGS  # ID kept for saved graphs


# ── FIX-007: Policy Guard and strict QC reject bad input ────────────────────

from radiance.nodes.color.qc import RadiancePolicyGuard, RadianceQC


def _guard(image, **kw):
    return RadiancePolicyGuard().run(mode="Guard", image=image, **kw)


def test_policy_guard_fails_nan_frames():
    img = torch.full((3, 8, 8, 3), 0.4)
    img[2, 0, 0, 0] = float("nan")
    _, passed, data1, _, _ = _guard(img)
    assert passed is False
    assert "non_finite_values" in data1


def test_policy_guard_fails_inf():
    img = torch.full((1, 8, 8, 3), 0.4)
    img[0, 1, 1, 1] = float("inf")
    assert _guard(img)[1] is False


@pytest.mark.parametrize("policy", ["{not json", "[]", '{"max_clipping": "a lot"}',
                                    '{"max_peak_nits": -5}', '{"description": "nothing"}',
                                    '{"max_clipping": NaN}'])
def test_policy_guard_rejects_malformed_policy(policy):
    with pytest.raises(ValueError, match="Policy Guard"):
        _guard(torch.full((1, 8, 8, 3), 0.4), policy=policy)


def test_policy_guard_rejects_unloadable_policy_file(tmp_path):
    with pytest.raises(ValueError, match="could not be loaded"):
        RadiancePolicyGuard().run(mode="Preset", image=torch.zeros(1, 2, 2, 3),
                                  policy_file=str(tmp_path / "missing.json"))
    bad = tmp_path / "bad.json"
    bad.write_text('{"max_clipping": -1}')
    with pytest.raises(ValueError, match="invalid"):
        RadiancePolicyGuard().run(mode="Preset", image=torch.zeros(1, 2, 2, 3), policy_file=str(bad))


@pytest.mark.parametrize("shape", [(8, 8), (1, 8, 8, 2), (0, 8, 8, 3)])
def test_policy_guard_rejects_invalid_tensors(shape):
    with pytest.raises(ValueError):
        _guard(torch.zeros(shape))


def test_policy_guard_valid_policy_still_passes():
    pol = '{"max_peak_nits": 1000, "max_clipping": 0.5, "max_black_crush": 0.5}'
    assert _guard(torch.full((2, 8, 8, 3), 0.4), policy=pol)[1] is True


@pytest.fixture
def real_defects(monkeypatch):
    """conftest stubs radiance.image.defects with MagicMocks; QC needs the real one."""
    import importlib.util
    from pathlib import Path
    import radiance.nodes.color.qc as qc
    spec = importlib.util.spec_from_file_location(
        "_radiance_real_defects", Path(qc.__file__).resolve().parents[2] / "image" / "defects.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(qc, "defects", mod)
    return mod


def test_strict_qc_fails_nan_and_blocks(real_defects):
    img = torch.full((2, 8, 8, 3), 0.5)
    img[1, 2, 2, 2] = float("nan")
    _, _, js, status = RadianceQC().run("Analyze", image=img)
    assert "FAIL" in status
    assert json.loads(js)["frames"][1]["checks"]["finite"]["status"] == "FAIL"
    with pytest.raises(RuntimeError):
        RadianceQC().run("Analyze", image=img, fail_on_errors=True)


def test_strict_qc_clean_image_passes(real_defects):
    _, _, _, status = RadianceQC().run("Analyze", image=torch.full((2, 8, 8, 3), 0.5))
    assert "PASS" in status


def test_strict_qc_blocks_on_invalid_input():
    with pytest.raises(RuntimeError, match="channels"):
        RadianceQC().run("Analyze", image=torch.zeros(1, 4, 4, 2), fail_on_errors=True)
    with pytest.raises(ValueError):
        RadianceQC().run("Analyze", image=None, fail_on_errors=True)


# ── FIX-014: 16-bit float + dithering ───────────────────────────────────────

from radiance.image.upscale import RadianceBitDepthConvert


@pytest.mark.parametrize("dither", ["None", "Ordered", "Blue Noise", "Random", "Floyd-Steinberg"])
def test_16bit_float_never_integer_quantises(dither):
    img = torch.linspace(-0.25, 4.0, 3 * 16 * 16).reshape(1, 16, 16, 3)
    out, info = RadianceBitDepthConvert().convert(img, "16-bit Float", dithering=dither)
    assert torch.equal(out, img.half().float())
    assert float(out.max()) > 1.0 and float(out.min()) < 0.0
    if dither != "None":
        assert "not applied" in info


def test_8bit_dither_still_quantises():
    img = torch.rand(1, 8, 8, 3)
    out, _ = RadianceBitDepthConvert().convert(img, "8-bit", dithering="Ordered")
    assert torch.allclose(out * 255, (out * 255).round(), atol=1e-4)
