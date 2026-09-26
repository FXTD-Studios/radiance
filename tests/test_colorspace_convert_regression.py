"""Regression tests for RadianceColorSpaceConvert (2026-08 audit).

What this guards against
------------------------
1. The node's inline LogC3 disagreed with the ARRI spec: 18% grey encoded to
   0.417 instead of 0.391, black to 0.271 instead of 0.093 -- and its decode
   disagreed with its own encode, so 0.18 round-tripped to 0.060 (~1.5 stops).
2. Ten of the sixteen advertised spaces had NO analytical implementation and
   silently returned the input unchanged whenever no OCIO config was loaded,
   which is the default install.
3. The conftest OCIO mock reported is_loaded as a truthy MagicMock, so the
   OCIO path "succeeded" with a no-op applyRGB and every colour test through
   this node validated an identity transform.

Every advertised space must (a) actually change the pixels, (b) round-trip to
scene linear within tolerance, and (c) hit the published 18%-grey code value
where one exists.
"""
import importlib

import pytest

torch = pytest.importorskip("torch")

radiance = importlib.import_module("radiance")
CSC = radiance.NODE_CLASS_MAPPINGS["RadianceColorSpaceConvert"]

LINEAR = "Linear sRGB (D65)"
ALL_SPACES = [s for s in CSC._COLOR_SPACES if s != LINEAR]

# Published 18%-grey code values (curve conventions as documented per space).
#
# AUDIT-FIX: this table used to hold eight entries for sixteen advertised
# spaces. The other eight were covered only by test_forward_is_not_identity and
# test_round_trip, which a wrong-but-invertible curve passes without complaint,
# so half the node had no published-value pin at all while the README claimed
# all sixteen landed on published values. Every space the node offers is now
# here, and test_the_published_values_agree_with_colour_science re-derives
# every one of them from an independent implementation so a mistyped digit in
# this table cannot pass either.
#
# Where each number comes from:
#   ACEScg                 gamut matrix only, no transfer curve. A neutral is
#                          on the Rec.709 and the AP1 neutral axis alike, so
#                          0.18 in is 0.18 out; a matrix that lost its white
#                          point normalisation moves it.
#   sRGB                   IEC 61966-2-1 inverse EOTF.
#   Rec.709 (OETF)         ITU-R BT.709-6: 1.099 * 0.18^0.45 - 0.099.
#   Rec.709 / BT.1886      ITU-R BT.1886 inverse EOTF, L_W = 1, L_B = 0:
#                          0.18^(1/2.4).
#   ACEScc / ACEScct       S-2014-003 / S-2016-001, (log2(0.18) + 9.72)/17.52
#                          on near-neutral AP1 grey; 0.18 is above both
#                          curves' linear segments so the two agree here.
#   LogC3 (EI800)          ARRI LogC3 spec; 0.391007 is the value ARRI
#                          publishes for 18% grey at every EI.
#   LogC4                  ARRI LogC4 spec (Alexa 35).
#   F-Log2                 Fujifilm F-Log2 spec; 18% grey shares LogC3's
#                          0.391007 by design, it is not a copy-paste slip.
#   C-Log3                 Canon Log 3 v1.2.
#   Log3G10                RED Log3G10 v2; 18% grey is exactly 1/3 by
#                          construction, which is the curve's own anchor.
#   DaVinci Intermediate   Blackmagic DaVinci Intermediate spec.
#   BMD Film Gen5          Blackmagic Film Generation 5 spec.
#   V-Log                  Panasonic V-Log spec (the 42.3 IRE grey patch).
#   N-Log                  Nikon N-Log spec; 0.18 is below the 0.328 cut, so
#                          this lands on the cube-root segment.
GREY18_REFERENCE = {
    "ACEScg": 0.180000,
    "sRGB (OETF encoded)": 0.461356,
    "Rec.709 (OETF encoded)": 0.409008,
    "Rec.709 / BT.1886": 0.489437,
    "ACEScc": 0.413588,
    "ACEScct": 0.413588,
    "LogC3 (ARRI EI800)": 0.391007,
    "LogC4 (ARRI Alexa 35)": 0.278396,
    "F-Log2 (Fujifilm)": 0.391007,
    "C-Log3 (Canon)": 0.343389,
    "Log3G10 (RED IPP2)": 0.333333,
    "DaVinci Intermediate": 0.336043,
    "BMD Film Gen5": 0.383562,
    "V-Log (Panasonic)": 0.423311,
    "N-Log (Nikon)": 0.363668,
}


def _convert(img, src, dst):
    node = CSC()
    return node.apply(image=img, src_space=src, dst_space=dst,
                      direction="Forward", strength=1.0)[0]


@pytest.fixture()
def hdr_image():
    base = torch.rand(1, 24, 24, 3) * 4.0
    base[0, 0] = 18.0        # HDR highlight
    base[0, 1] = 0.0         # black
    base[0, 2] = 0.18        # 18% grey
    return base


@pytest.mark.real_torch
@pytest.mark.parametrize("space", ALL_SPACES)
def test_forward_is_not_identity(space):
    """A conversion that returns its input unchanged is the P0 this file exists for.

    Saturated input, not neutral grey: a gamut matrix (ACEScg) maps the
    neutral axis to itself, so grey would pass an identity check even when
    the matrix is applied correctly.
    """
    img = torch.tensor([0.6, 0.18, 0.05]).expand(1, 4, 4, 3).contiguous()
    out = _convert(img, LINEAR, space)
    assert not torch.allclose(out, img, atol=1e-4), (
        f"{space}: forward conversion returned the input unchanged")


@pytest.mark.real_torch
@pytest.mark.parametrize("space", ALL_SPACES)
def test_round_trip(space, hdr_image):
    encoded = _convert(hdr_image, LINEAR, space)
    back = _convert(encoded, space, LINEAR)
    # Display-referred and clipping curves cannot represent 18.0; compare on
    # the representable region only.
    lossy_above_1 = space in (
        "sRGB (OETF encoded)", "Rec.709 (OETF encoded)", "Rec.709 / BT.1886",
        "ACEScc",  # negatives pin to the min code value by spec
    )
    mask = (hdr_image <= 1.0) & (hdr_image >= 0.0) if lossy_above_1 \
        else torch.ones_like(hdr_image, dtype=torch.bool)
    err = (back - hdr_image).abs()[mask].max().item()
    assert err < 5e-3, f"{space}: round-trip error {err:.5f}"


@pytest.mark.real_torch
@pytest.mark.parametrize("space,ref", sorted(GREY18_REFERENCE.items()))
def test_grey18_reference(space, ref):
    img = torch.full((1, 2, 2, 3), 0.18)
    out = _convert(img, LINEAR, space)
    got = out[0, 0, 0, 0].item()
    # 1e-4, not the old 2e-3: at 2e-3 a fourth-decimal typo in either the table
    # or a curve constant passes, and every value here is published to six.
    assert abs(got - ref) < 1e-4, (
        f"{space}: 18% grey encoded to {got:.6f}, published value is {ref:.6f}")


@pytest.mark.real_torch
def test_every_offered_space_has_a_published_grey_pin():
    """The gap this file was reopened for.

    GREY18_REFERENCE held eight of the sixteen advertised spaces. The eight
    without an entry were covered only by the not-identity and round-trip
    checks above, and a curve that is wrong but invertible passes both, so the
    node could ship a mis-specified transfer function with a green suite. A new
    entry in _COLOR_SPACES must arrive with its published grey value or fail
    here.
    """
    missing = [s for s in ALL_SPACES if s not in GREY18_REFERENCE]
    assert not missing, (
        f"{missing} are offered by RadianceColorSpaceConvert but have no "
        "published 18%-grey value pinned, so only round-trip consistency is "
        "checked and a wrong-but-invertible curve would pass")
    unknown = [s for s in GREY18_REFERENCE if s not in ALL_SPACES]
    assert not unknown, f"{unknown} are pinned but the node no longer offers them"


#: Which colour-science entry point encodes each space, by name. Nothing here
#: is looked up until _colour_science_grey18() has the real package loaded.
_COLOUR_SCIENCE_ENCODINGS = {
    "ACEScc": ("log", "ACEScc"),
    "ACEScct": ("log", "ACEScct"),
    "LogC3 (ARRI EI800)": ("log", "ARRI LogC3"),
    "LogC4 (ARRI Alexa 35)": ("log", "ARRI LogC4"),
    "F-Log2 (Fujifilm)": ("log", "F-Log2"),
    "C-Log3 (Canon)": ("log", "Canon Log 3"),
    "Log3G10 (RED IPP2)": ("log", "Log3G10"),
    "V-Log (Panasonic)": ("log", "V-Log"),
    "N-Log (Nikon)": ("log", "N-Log"),
    "sRGB (OETF encoded)": ("model", "eotf_inverse_sRGB"),
    "Rec.709 (OETF encoded)": ("model", "oetf_BT709"),
    "DaVinci Intermediate": ("model", "oetf_DaVinciIntermediate"),
    "BMD Film Gen5": ("model", "oetf_BlackmagicFilmGeneration5"),
}

#: colour-science has no entry point for these two, so they are spelled out.
#:   ACEScg is a gamut matrix with no transfer curve: a neutral is on the
#:   Rec.709 and the AP1 neutral axis alike, so 0.18 comes back untouched.
#:   Rec.709 / BT.1886 is the BT.1886 inverse EOTF with L_W = 1 and L_B = 0,
#:   which collapses to a pure 2.4 power law.
_SPELLED_OUT_GREY18 = {
    "ACEScg": 0.18,
    "Rec.709 / BT.1886": 0.18 ** (1.0 / 2.4),
}


def _colour_science_grey18():
    """18% grey per space, computed by colour-science, or (None, reason).

    Two things get in the way of a plain ``import colour`` here.
    test_node_smoke.py installs a bare stub module named ``colour`` into
    sys.modules at import time, and pytest imports every test module during
    collection, so by the time this runs the stub is already in place and the
    cross-check would quietly skip. And colour-science resolves its own
    submodules lazily, so every value has to be computed while the real
    package is still in sys.modules, not afterwards. Hence: take the stub out,
    compute everything, put the stub back exactly as it was.
    """
    import importlib
    import sys

    existing = sys.modules.get("colour")
    borrowed = existing is None or not hasattr(existing, "LOG_ENCODINGS")
    saved = {}
    if borrowed:
        saved = {name: mod for name, mod in sys.modules.items()
                 if name == "colour" or name.startswith("colour.")}
        for name in saved:
            del sys.modules[name]
    try:
        colour = importlib.import_module("colour")
        if not hasattr(colour, "LOG_ENCODINGS"):
            return None, "colour-science is not installed"
        values = dict(_SPELLED_OUT_GREY18)
        for space, (kind, name) in _COLOUR_SCIENCE_ENCODINGS.items():
            fn = (colour.LOG_ENCODINGS[name] if kind == "log"
                  else getattr(colour.models, name))
            values[space] = float(fn(0.18))
        return values, colour.__version__
    except ImportError:
        return None, "colour-science is not installed"
    finally:
        if borrowed:
            for name in [n for n in sys.modules
                         if n == "colour" or n.startswith("colour.")]:
                del sys.modules[name]
            sys.modules.update(saved)


@pytest.mark.real_torch
def test_the_published_values_agree_with_colour_science():
    """Cross-check every literal in GREY18_REFERENCE against another codebase.

    The table above is hand-transcribed, and a hand-transcribed number is
    exactly the kind of thing that ships wrong. Worse, the obvious way to
    "fix" a failing pin is to paste in whatever the node printed, which turns
    the pin straight back into a tautology. colour-science implements the same
    published curves independently, so a digit that has drifted towards the
    implementation fails here.
    """
    values, note = _colour_science_grey18()
    if values is None:
        pytest.skip(note)

    missing = [s for s in GREY18_REFERENCE if s not in values]
    assert not missing, (
        f"{missing} have no independent cross-check; add one or say in the "
        "table comment why the value cannot be established elsewhere")

    for space, ref in sorted(GREY18_REFERENCE.items()):
        theirs = values[space]
        assert abs(theirs - ref) < 1e-4, (
            f"{space}: this file pins 18% grey at {ref:.6f}, colour-science "
            f"{note} computes {theirs:.6f} from the same published curve")


@pytest.mark.real_torch
def test_logc3_black_reference():
    """Black must land at f = 0.0928, not the 0.271 the old inline curve gave."""
    img = torch.zeros(1, 2, 2, 3)
    out = _convert(img, LINEAR, "LogC3 (ARRI EI800)")
    assert abs(out[0, 0, 0, 0].item() - 0.092809) < 1e-3


@pytest.mark.real_torch
def test_the_ocio_path_cannot_reintroduce_identity():
    """Nothing in this file may end up validating a no-op transform.

    The original hazard: conftest's OCIO manager mock was a bare MagicMock, so
    is_loaded was truthy, _try_ocio got a MagicMock "processor" whose applyRGB
    did nothing, and every colour assertion here passed on the untouched input.

    This used to be pinned as "the manager must report is_loaded False", which
    held while tests/conftest.py stubbed radiance.radiance_ocio
    unconditionally. It stubs it only when PyOpenColorIO is absent now, so on a
    lane that has PyOpenColorIO the manager is real and legitimately loaded and
    that assertion fails for a reason that has nothing to do with this node.
    The invariant that actually matters survives the change: whatever comes
    back from the OCIO path, it is not the input.
    """
    from radiance.radiance_ocio import get_ocio_manager

    mgr = get_ocio_manager()
    if not mgr.is_loaded:
        # No config, or conftest's stub. Either way the analytical curves are
        # what every other assertion in this file is measuring.
        return

    node = CSC()
    img = torch.tensor([0.6, 0.18, 0.05]).expand(1, 4, 4, 3).contiguous()
    for space in ALL_SPACES:
        out = node._try_ocio(img, LINEAR, space)
        assert out is None or not torch.allclose(out, img, atol=1e-4), (
            f"{space}: the OCIO path returned the input unchanged, so every "
            "colour assertion in this file is validating an identity transform")


@pytest.mark.real_torch
def test_srgb_and_rec709_oetf_differ():
    """Rec.709 OETF was previously aliased to the sRGB curve."""
    img = torch.full((1, 2, 2, 3), 0.18)
    srgb = _convert(img, LINEAR, "sRGB (OETF encoded)")
    r709 = _convert(img, LINEAR, "Rec.709 (OETF encoded)")
    assert abs(srgb[0, 0, 0, 0].item() - r709[0, 0, 0, 0].item()) > 0.02


@pytest.mark.real_torch
def test_negative_scene_linear_survives_log_spaces():
    """Slightly-negative scene-linear (grain, matte edges) must not explode."""
    img = torch.full((1, 2, 2, 3), -0.02)
    for space in ("LogC3 (ARRI EI800)", "LogC4 (ARRI Alexa 35)", "DaVinci Intermediate"):
        out = _convert(img, LINEAR, space)
        assert torch.isfinite(out).all(), f"{space}: non-finite output for negative input"
