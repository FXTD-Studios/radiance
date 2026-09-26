"""Color section, 3.5.0: controls that did nothing or less than their name.

LUT / LUT Blend log_space, HueCorrect grade_info, OCIO Context.
"""
import json
import os

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.real_torch


def _identity_cube(path, n=33):
    lines = [f"LUT_3D_SIZE {n}"]
    g = np.linspace(0.0, 1.0, n)
    for b in g:
        for gg in g:
            for r in g:
                lines.append(f"{r:.6f} {gg:.6f} {b:.6f}")
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def _ramp():
    x = torch.tensor([0.0, 0.01, 0.18, 0.5, 1.0, 4.0])
    return torch.stack([x, x * 0.9, x * 0.8], dim=-1).reshape(1, 1, 6, 3)


# ── LUT: log_space encodes to the LUT's real input ─────────────────────────

@pytest.mark.parametrize("enc", ["ARRI LogC3", "Sony S-Log3", "ACEScct", "Panasonic V-Log"])
def test_lut_log_space_is_the_camera_curve(tmp_path, enc):
    from radiance.color.encodings import encode
    from radiance.color.lut import RadianceLUTApply
    cube = _identity_cube(tmp_path / "id.cube")
    img = _ramp()
    out = RadianceLUTApply().apply_lut(img, cube, 1.0, True, enc, False, "Tetrahedral")[0]
    ref, _ = encode(img.numpy(), enc)
    np.testing.assert_allclose(out.numpy(), np.clip(ref, 0, 1), atol=2e-3)


def test_lut_logc3_grey_lands_where_arri_says(tmp_path):
    from radiance.color.lut import RadianceLUTApply
    cube = _identity_cube(tmp_path / "id.cube")
    grey = torch.full((1, 1, 1, 3), 0.18)
    out = RadianceLUTApply().apply_lut(grey, cube, 1.0, True, "ARRI LogC3", False, "Tetrahedral")[0]
    assert out[0, 0, 0, 0].item() == pytest.approx(0.391, abs=2e-3)      # LogC3 EI800 18 % grey


def test_lut_working_space_changes_the_primaries(tmp_path):
    from radiance.color.lut import RadianceLUTApply
    cube = _identity_cube(tmp_path / "id.cube")
    red = torch.tensor([[[[0.18, 0.02, 0.02]]]])
    a = RadianceLUTApply().apply_lut(red, cube, 1.0, True, "ARRI LogC3", False, "Tetrahedral",
                                     "Linear Rec.709 (sRGB)")[0]
    b = RadianceLUTApply().apply_lut(red, cube, 1.0, True, "ARRI LogC3", False, "Tetrahedral",
                                     "ACEScg")[0]
    assert not torch.allclose(a, b, atol=1e-3)


def test_lut_legacy_exponent_names_still_run_the_same(tmp_path):
    from radiance.color.lut import RadianceLUTApply
    cube = _identity_cube(tmp_path / "id.cube")
    img = torch.full((1, 2, 2, 3), -0.5)
    out = RadianceLUTApply().apply_lut(img, cube, 1.0, True, "Log10", False, "Trilinear")[0]
    assert torch.allclose(out, torch.full_like(out, 10 ** -0.5), atol=2e-3)


def test_lut_legacy_names_pass_validation_but_are_not_offered():
    from radiance.color.lut import RadianceLUTApply, RadianceLUTBlend
    menu = RadianceLUTApply.INPUT_TYPES()["optional"]["log_encoding"][0]
    assert "Log10" not in menu and "ARRI LogC3" in menu
    for cls in (RadianceLUTApply, RadianceLUTBlend):
        assert cls.VALIDATE_INPUTS(log_encoding="Log10") is True
        assert cls.VALIDATE_INPUTS(log_encoding="ARRI LogC4") is True
        assert cls.VALIDATE_INPUTS(log_encoding="Nope") is not True


def test_lut_keeps_alpha(tmp_path):
    from radiance.color.lut import RadianceLUTApply
    cube = _identity_cube(tmp_path / "id.cube")
    img = torch.cat([torch.full((1, 2, 2, 3), 0.18), torch.full((1, 2, 2, 1), 0.25)], dim=-1)
    for enc in ("ARRI LogC3", "Log10"):
        out = RadianceLUTApply().apply_lut(img, cube, 1.0, True, enc, False, "Trilinear")[0]
        assert out.shape[-1] == 4 and torch.allclose(out[..., 3], torch.full((1, 2, 2), 0.25))


def test_lut_blend_encodes_both_lookups(tmp_path):
    from radiance.color.encodings import encode
    from radiance.color.lut import RadianceLUTBlend
    cube = _identity_cube(tmp_path / "id.cube")
    img = _ramp()
    out = RadianceLUTBlend().blend_luts(img, cube, cube, 0.5, "Linear", log_space=True,
                                        log_encoding="Sony S-Log3",
                                        interpolation="Tetrahedral")[0]
    ref, _ = encode(img.numpy(), "Sony S-Log3")
    np.testing.assert_allclose(out.numpy(), np.clip(ref, 0, 1), atol=2e-3)


def test_lut_widget_order_is_unchanged():
    # Saved graphs store widget values by position; working_space is appended.
    from radiance.color.lut import RadianceLUTApply, RadianceLUTBlend
    opt = list(RadianceLUTApply.INPUT_TYPES()["optional"])
    assert opt == ["log_encoding", "clamp_output", "interpolation", "working_space"]
    opt = list(RadianceLUTBlend.INPUT_TYPES()["optional"])
    assert opt == ["strength", "clamp_output", "log_space", "log_encoding", "interpolation",
                   "working_space"]


# ── HueCorrect ─────────────────────────────────────────────────────────────

def test_huecorrect_passes_grade_info_on():
    from radiance.nodes.color.curves import RadianceHueCurves
    img = torch.rand(1, 4, 4, 3)
    out, info = RadianceHueCurves().apply(img, "Hue vs Saturation", "[[0,0.1],[1,0.1]]", 1.0,
                                          grade_info='{"node":"up"}')
    data = json.loads(info)
    assert data["upstream"] == '{"node":"up"}' and data["mode"] == "Hue vs Saturation"
    assert RadianceHueCurves.RETURN_NAMES == ("image", "grade_info")


def test_huecorrect_bad_curve_is_an_error_not_a_silent_no_op():
    from radiance.nodes.color.curves import RadianceHueCurves
    with pytest.raises(ValueError, match="control_points"):
        RadianceHueCurves().apply(torch.rand(1, 2, 2, 3), "Hue vs Hue", "[[0,0.1],[1,0.1]", 1.0)


def test_huecorrect_default_is_still_identity():
    from radiance.nodes.color.curves import RadianceHueCurves
    spec = RadianceHueCurves.INPUT_TYPES()["required"]["control_points"][1]["default"]
    img = torch.rand(1, 4, 4, 3) * 0.9 + 0.05
    out, _ = RadianceHueCurves().apply(img, "Hue vs Hue", spec, 1.0)
    assert torch.allclose(out, img, atol=1e-4)


# ── OCIO Context ───────────────────────────────────────────────────────────

def _studio_config(tmp_path):
    ocio = pytest.importorskip("PyOpenColorIO")
    cfg = ocio.Config.CreateFromBuiltinConfig("studio-config-latest")
    path = tmp_path / "studio.ocio"
    path.write_text(cfg.serialize())
    return str(path)


def test_ocio_context_runs_as_an_output_node():
    from radiance.nodes.color.ocio import RadianceOCIOContext
    assert RadianceOCIOContext.OUTPUT_NODE is True
    assert list(RadianceOCIOContext.INPUT_TYPES()["required"]) == ["config_path"]


def test_ocio_context_loads_and_reports(tmp_path):
    from radiance.nodes.color.ocio import RadianceOCIOContext
    from radiance.radiance_ocio import get_ocio_manager
    path = _studio_config(tmp_path)
    out = RadianceOCIOContext().set_context(path, working_space="ACEScg")   # stale widget ok
    ctx = out["result"][0]
    assert ctx["status"] == "active" and os.path.samefile(ctx["path"], path)
    assert os.path.samefile(get_ocio_manager().config_path, path)
    assert "colour spaces" in out["ui"]["text"][0]


def test_ocio_context_missing_file_is_an_error(tmp_path):
    pytest.importorskip("PyOpenColorIO")
    from radiance.nodes.color.ocio import RadianceOCIOContext
    with pytest.raises(FileNotFoundError):
        RadianceOCIOContext().set_context(str(tmp_path / "nope.ocio"))


def test_color_space_convert_reloads_the_wired_config(tmp_path):
    pytest.importorskip("PyOpenColorIO")
    from radiance.nodes.color.colorspace import RadianceColorSpaceConvert
    from radiance.nodes.color.ocio import RadianceOCIOContext
    from radiance.radiance_ocio import get_ocio_manager
    path = _studio_config(tmp_path)
    ctx = RadianceOCIOContext().set_context(path)["result"][0]
    mgr = get_ocio_manager()
    mgr.config_path = "something else"          # another node loaded a different config
    RadianceColorSpaceConvert().apply(torch.rand(1, 2, 2, 3), "Linear sRGB (D65)", "ACEScg",
                                      ocio_context=ctx)
    assert os.path.samefile(mgr.config_path, path)
