"""apply_input_transform lands camera log in the linear Rec.709 working space.

It used to decode the log curve only, so an ARRI LogC4 plate came out as
linear AWG4 that every caller (RadianceHDRAnalysis luma, color_utils users)
then treated as linear Rec.709: neutrals were right, saturated colours were
not. These tests hold the decode to the published primaries and, when a real
OpenColorIO is installed, to the ACES studio config.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

if getattr(torch, "__radiance_stub__", False):  # light lane: no real tensors
    pytest.skip("needs real torch", allow_module_level=True)

from radiance.color import encodings as E
from radiance.color import transfer as T
from radiance.color.pipeline import apply_input_transform, apply_output_transform


def _npm(prims):
    """RGB -> XYZ from CIE xy primaries and white (SMPTE RP 177)."""
    (r, g, b, w) = prims

    def xyz(x, y):
        return np.array([x / y, 1.0, (1.0 - x - y) / y])

    m = np.stack([xyz(*r), xyz(*g), xyz(*b)], axis=1)
    return m * np.linalg.solve(m, xyz(*w))


REC709 = ((0.640, 0.330), (0.300, 0.600), (0.150, 0.060), (0.3127, 0.3290))
# Published camera primaries (all D65 white): ARRI LogC4 spec, ARRI LogC3
# white paper, Sony S-Gamut3.Cine, Panasonic V-Log/V-Gamut reference manual,
# Blackmagic DaVinci Wide Gamut.
CAMERA = {
    "ARRI LogC4": (((0.7347, 0.2653), (0.1424, 0.8576), (0.0991, -0.0308), (0.3127, 0.3290)),
                   T.linear_to_logc4),
    "ARRI LogC3": (((0.6840, 0.3130), (0.2210, 0.8480), (0.0861, -0.1020), (0.3127, 0.3290)),
                   T.linear_to_logc3),
    "Sony S-Log3": (((0.766, 0.275), (0.225, 0.800), (0.089, -0.087), (0.3127, 0.3290)),
                    T.linear_to_slog3),
    "Panasonic V-Log": (((0.730, 0.280), (0.165, 0.840), (0.100, -0.030), (0.3127, 0.3290)),
                        T.linear_to_vlog),
    "DaVinci Intermediate": (((0.8000, 0.3130), (0.1682, 0.9877), (0.0790, -0.1155), (0.3127, 0.3290)),
                             T.linear_to_davinci_intermediate),
}


@pytest.mark.parametrize("cs", sorted(CAMERA))
def test_a_camera_primary_lands_on_its_rec709_vector(cs):
    prims, encode = CAMERA[cs]
    # A pure camera-gamut primary at 18 % (above every toe), log encoded.
    lin = np.array([0.18, 0.0, 0.0], np.float32)
    code = np.asarray(encode(lin[None, None, :]), np.float32)
    code[..., 1:] = np.asarray(encode(np.zeros((1, 1, 2), np.float32)), np.float32)

    out = apply_input_transform(torch.from_numpy(code), cs).numpy()[0, 0]

    # Same white on both sides, so the direct NPM product is the conversion;
    # the package routes through the ACES white with CAT02 as the studio
    # config does, which moves it by under 0.7 % of the primary.
    expected = (np.linalg.inv(_npm(REC709)) @ _npm(prims))[:, 0] * 0.18
    assert np.allclose(out, expected, atol=0.18 * 0.01), (cs, out, expected)
    # The old curve-only decode returned the camera primary unchanged.
    assert abs(out[1]) > 0.005 and out[0] > 0.2


@pytest.mark.parametrize("cs", sorted(CAMERA) + ["ACEScct"])
def test_neutrals_stay_neutral(cs):
    code = torch.full((1, 2, 2, 3), 0.45)
    out = apply_input_transform(code, cs).numpy()
    assert np.allclose(out[..., 0], out[..., 1], rtol=1e-4)
    assert np.allclose(out[..., 1], out[..., 2], rtol=1e-4)


def test_acescct_decodes_to_rec709_like_acescg_does():
    lin_ap1 = np.array([[[0.3, 0.1, 0.05]]], np.float32)
    code = torch.from_numpy(np.asarray(T.linear_to_acescct(lin_ap1), np.float32))
    via_cct = apply_input_transform(code, "ACEScct").numpy()
    via_cg = apply_input_transform(torch.from_numpy(lin_ap1), "ACEScg").numpy()
    assert np.allclose(via_cct, via_cg, atol=1e-4)


@pytest.mark.parametrize("cs", sorted(CAMERA) + ["ACEScct"])
def test_output_transform_inverts_input_transform(cs):
    lin = torch.tensor([[[[0.4, 0.2, 0.1], [0.05, 0.3, 0.6], [1.5, 0.9, 0.2]]]])
    back = apply_input_transform(apply_output_transform(lin, cs), cs)
    assert torch.allclose(back, lin, rtol=2e-3, atol=2e-4)


def test_alpha_is_left_alone():
    code = torch.tensor([[[[0.6, 0.4, 0.3, 0.25]]]])
    out = apply_input_transform(code, "ARRI LogC4")
    assert out.shape == code.shape
    assert float(out[..., 3]) == pytest.approx(T.logc4_to_linear(np.float32(0.25)), rel=1e-5)


_OCIO_NAMES = {
    "ARRI LogC4": "ARRI LogC4",
    "ARRI LogC3": "ARRI LogC3 (EI800)",
    "Sony S-Log3": "S-Log3 S-Gamut3.Cine",
    "Panasonic V-Log": "V-Log V-Gamut",
    "DaVinci Intermediate": "DaVinci Intermediate WideGamut",
    "ACEScct": "ACEScct",
}


@pytest.mark.skipif(not E.HAS_OCIO or not getattr(E._OCIO, "__file__", None),
                    reason="needs a real OpenColorIO")
@pytest.mark.parametrize("cs", sorted(_OCIO_NAMES))
def test_matches_the_aces_studio_config(cs):
    code = np.array([[[0.6, 0.4, 0.3], [0.45, 0.45, 0.45], [0.3, 0.5, 0.55]]], np.float32)
    try:
        ref = E.ocio_apply(code, _OCIO_NAMES[cs], "Linear Rec.709 (sRGB)",
                           "ocio://studio-config-latest")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"studio config lacks {cs}: {exc}")
    out = apply_input_transform(torch.from_numpy(code), cs).numpy()
    assert np.allclose(out, ref, rtol=2e-3, atol=2e-3), (cs, out, ref)
