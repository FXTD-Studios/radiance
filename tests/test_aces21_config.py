"""ACES 2.1 (OCIO 2.6's studio config v5.0.0) for the ACES 2 reference transforms.

Radiance uses the newest ACES 2 studio config the installed OCIO builds in. ACES
2.1 keeps ACES 2.0's Output Transforms: every view the two configs share gives
the same values, so moving to it changes no rendered pixel; the one display it
renamed is found under its new name. RUDRA ACES 2 stays on the CG config v4.0.0
it was trained on.
"""
import importlib

import numpy as np
import pytest

ocio = pytest.importorskip("PyOpenColorIO")

V5 = "studio-config-v5.0.0_aces-v2.1_ocio-v2.6"
V4 = "studio-config-v4.0.0_aces-v2.0_ocio-v2.5"


def _builtin(name):
    return name in {c[0] for c in ocio.BuiltinConfigRegistry().getBuiltinConfigs()}


def test_the_newest_studio_config_is_used():
    a = importlib.import_module("radiance.hdr.aces2_ocio")
    expect = V5 if _builtin(V5) else V4
    assert a.PINNED_CONFIG == "ocio://" + expect
    dp = importlib.import_module("radiance.color.display_preview")
    assert dp.ACES2_CONFIG == expect


@pytest.mark.skipif(not _builtin(V5), reason="OCIO 2.6 or newer needed for ACES 2.1")
def test_aces_2_1_renders_every_output_like_aces_2_0():
    a = importlib.import_module("radiance.hdr.aces2_ocio")
    v4 = ocio.Config.CreateFromBuiltinConfig(V4)
    rng = np.random.default_rng(0)
    x = rng.lognormal(-1.5, 2.2, (50000, 3)).astype(np.float32)
    for name, (display, view) in a.OUTPUT_VIEWS.items():
        got = a.apply_reference(x, "ACEScg", name)
        ref = x.copy()
        v4.getProcessor("ACEScg", display, view, ocio.TRANSFORM_DIR_FORWARD).getDefaultCPUProcessor().applyRGB(ref)
        np.testing.assert_array_equal(got, ref, err_msg=name)


def test_rudra_aces2_keeps_its_own_pin():
    m = importlib.import_module("radiance.model.rudra_aces2")
    assert m.OCIO_CONFIG == "cg-config-v4.0.0_aces-v2.0_ocio-v2.5"
