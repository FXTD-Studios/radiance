"""The analytic colour path agrees with OpenColorIO's ACES studio config.

A machine without OCIO decodes a camera file through radiance.color.encodings'
own transfer functions and primaries matrices. These tests hold that path to
the OCIO result for every encoding OCIO also defines, so the same plate reads
the same on every install.
"""
import numpy as np
import pytest

from radiance.color import encodings as E

pytestmark = pytest.mark.skipif(not E.HAS_OCIO, reason="OpenColorIO not installed")

RNG = np.random.default_rng(7)
# In-gamut for every camera gamut here: Rec.709 colours, greys, and HDR greys.
SAMPLES = np.concatenate([
    RNG.uniform(0.0, 1.0, (300, 3)),
    np.repeat(np.array([[0.0], [0.001], [0.18], [1.0], [4.0], [12.0]]), 3, axis=1),
]).astype(np.float32)

WITH_OCIO = [n for n, e in E.ENCODINGS.items() if e.ocio]


@pytest.mark.parametrize("name", WITH_OCIO)
@pytest.mark.parametrize("working", ["Linear Rec.709 (sRGB)", "ACEScg"])
def test_analytic_encode_matches_ocio(name, working):
    enc = E.ENCODINGS[name]
    x = SAMPLES
    if enc.display_referred or name.startswith("Rec.709"):
        # Display encodings have no meaning outside [0, 1] of their own
        # gamut; OCIO clamps there, the analytic path mirrors. Compare the
        # colours both can represent.
        lin = x @ E.gamut_matrix(E.ENCODINGS[working].gamut, enc.gamut).T.astype(np.float32)
        x = x[((lin >= 0) & (lin <= 1)).all(axis=1)]
    ana, _ = E.encode(x, name, working, use_ocio=False)
    ref, _ = E.encode(x, name, working, use_ocio=True)
    live = ref < 0.999            # OCIO clamps some log curves at 1.0
    assert float(np.abs(ana - ref)[live].max()) < 1e-5, name


@pytest.mark.parametrize("name", list(E.ENCODINGS))
def test_every_encoding_round_trips(name):
    enc = E.ENCODINGS[name]
    lim = 3.7 if name.startswith("HLG") else 1.0      # HLG headroom ~3.8x diffuse white
    x = np.clip(SAMPLES[:300], 0.0, lim)
    for use_ocio in (False, True):
        coded, _ = E.encode(x, name, use_ocio=use_ocio)
        back, _ = E.decode(coded, name, use_ocio=use_ocio)
        assert float(np.abs(back - x).max()) < 2e-4, (name, use_ocio)


def test_pq_places_reference_white_at_203_nits():
    code, _ = E.encode(np.ones((1, 3), np.float32), "PQ (ST.2084)", "Linear Rec.2020")
    assert abs(float(code[0, 0]) - 0.5807) < 1e-3     # ST.2084(203 cd/m2)


def test_hlg_places_reference_white_at_75_percent():
    code, _ = E.encode(np.ones((1, 3), np.float32), "HLG (BT.2100)", "Linear Rec.2020")
    # BT.2100 transcode at the 1000-nit reference display: 0.74988, the same
    # value OpenColorIO's Rec.2100-HLG display gives (BT.2408 rounds to 75 %).
    assert abs(float(code[0, 0]) - 0.75) < 5e-4


def test_primaries_matrices_match_ocio_exactly():
    for g, name in (("Rec.2020", "Linear Rec.2020"), ("AP1", "ACEScg"), ("AP0", "ACES2065-1"),
                    ("P3-D65", "Linear P3-D65")):
        m = E.gamut_matrix("Rec.709", g)
        ref, _ = E.encode(np.eye(3, dtype=np.float32), name, use_ocio=True)
        assert np.allclose(m.T, ref, atol=2e-5), g
