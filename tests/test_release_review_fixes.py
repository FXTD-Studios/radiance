"""Regression tests for the defects found in the pre-release review.

Six of these came out of the new functional harness (tests/test_node_functional.py)
the first time every node's FUNCTION was actually called, and out of independent
verification of the reviewers' highest-severity claims. Each one had shipped:
none of them raised anywhere the existing suite was looking.
"""
import json
import pathlib

import pytest

torch = pytest.importorskip("torch")

RADIANCE_TORCH_GATED = True

_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _src(rel):
    return (_ROOT / rel).read_text(encoding="utf-8")


# ── 1. Path.with_suffix truncated dotted filenames ──────────────────────────

@pytest.mark.real_torch
@pytest.mark.parametrize("filename,expected", [
    ("sh010.comp", "sh010.comp_v0001.exr"),
    ("plate.v2", "plate.v2_v0001.exr"),
    ("bg.matte", "bg.matte_v0001.exr"),
    ("plain", "plain_v0001.exr"),
])
def test_dotted_filenames_survive_the_version_suffix(tmp_path, filename, expected):
    """
    `Path(base).with_suffix(ext)` REPLACES everything after the last dot in the
    final component. Dotted stems are routine in VFX naming, and `base` here is
    "<dir>/<filename>_v0001" — so "sh010.comp_v0001" had a "suffix" of
    ".comp_v0001" and every version collapsed onto one file. With overwrite
    defaulting to True, each render destroyed the previously approved one and
    reported success with the truncated path.
    """
    from radiance.nodes.io.write import RadianceWrite

    base = str(tmp_path / f"{filename}_v0001")
    out = RadianceWrite()._out_path(base, ".exr", overwrite=True)
    assert out.name == expected, f"{filename!r} was truncated to {out.name!r}"


@pytest.mark.real_torch
def test_out_path_rejects_an_empty_name_with_a_readable_message(tmp_path):
    """It used to raise `ValueError: PosixPath('.') has an empty name`."""
    from radiance.nodes.io.write import RadianceWrite

    with pytest.raises(ValueError) as excinfo:
        RadianceWrite()._out_path("", ".exr", overwrite=True)
    msg = str(excinfo.value)
    assert "filename" in msg and "empty name" not in msg


@pytest.mark.real_torch
def test_out_path_does_not_double_the_extension(tmp_path):
    from radiance.nodes.io.write import RadianceWrite

    out = RadianceWrite()._out_path(str(tmp_path / "shot.exr"), ".exr", overwrite=True)
    assert out.name == "shot.exr"


def test_digital_cinema_write_has_a_usable_output_path_default():
    from radiance.nodes.io.write import RadianceDigitalCinemaWrite

    default = RadianceDigitalCinemaWrite.INPUT_TYPES()["required"]["output_path"][1]["default"]
    assert default, "output_path defaults to '', so the node crashes on a bare queue"


# ── 2. A 2-channel tensor declared as IMAGE ─────────────────────────────────

@pytest.mark.real_torch
def test_stabilizer_displacement_map_is_a_valid_comfyui_image():
    """
    RETURN_TYPES declares displacements_xy as IMAGE, and a ComfyUI IMAGE is
    (B, H, W, 3|4). It returned (B, H, W, 2), so wiring it into PreviewImage or
    any downstream node indexed a channel that did not exist.
    """
    from radiance.nodes.vfx.plate import RadianceSubpixelStabilizer

    seq = torch.rand(3, 32, 32, 3)
    _, disp = RadianceSubpixelStabilizer().apply(seq, 0, 16)
    assert disp.ndim == 4 and disp.shape[-1] in (3, 4), \
        f"displacements_xy has shape {tuple(disp.shape)}"
    assert torch.equal(disp[..., 2], torch.zeros_like(disp[..., 2])), \
        "the unused third channel should stay zero so the map reads as red=x/green=y"


# ── 3. json.loads("") on a widget default ───────────────────────────────────

@pytest.mark.real_torch
@pytest.mark.parametrize("payload", ["", "   ", "not json", "[1,2,3]"])
def test_scene_cut_split_explains_bad_cut_data(payload):
    """
    `cut_data` is a STRING widget with no default, so it arrives as "" for
    anyone who drops the node and queues before wiring the detector. That used
    to surface as a bare JSONDecodeError traceback.
    """
    from radiance.nodes.ai.scene_cut import RadianceSceneCutSplit

    images = torch.rand(4, 16, 16, 3)
    if payload.strip() and payload != "not json" and payload.startswith("["):
        with pytest.raises(ValueError, match="JSON object"):
            RadianceSceneCutSplit().split(images, payload, 0)
        return
    if payload == "not json":
        with pytest.raises(ValueError, match="not valid JSON"):
            RadianceSceneCutSplit().split(images, payload, 0)
        return
    # Empty input degrades to the documented "no shots" result rather than raising.
    result = RadianceSceneCutSplit().split(images, payload, 0)
    assert isinstance(result, tuple) and len(result) == 5
    assert json.loads(result[4]).get("error") == "no shots in cut_data"


# ── 4. The model cache stopped being subscriptable ──────────────────────────

def test_model_cache_supports_dict_style_access():
    """
    nodes/upscale/upscale.py did `_FACE_MODEL_CACHE[key]` after the
    dict -> GPUModelCache refactor. TypeError was swallowed by a broad
    `except Exception` logging at DEBUG, so RetinaFace detection silently never
    ran — face restore fell through to the Haar cascade and still reported
    "Faces detected: 0" as success.
    """
    from radiance.model.cache import GPUModelCache

    cache = GPUModelCache(max_size=2)
    cache.put("k", 42)
    assert cache["k"] == 42
    with pytest.raises(KeyError):
        cache["absent"]
    cache["via_setitem"] = 7
    assert cache.get("via_setitem") == 7


def test_no_module_subscripts_a_cache_that_only_had_get():
    src = _src("nodes/upscale/upscale.py")
    assert "_FACE_MODEL_CACHE[" not in src, \
        "use .get(); the bare subscript was dead code for months"


# ── 5. torch.quantile's 2**24 cap kills 4K multipass ────────────────────────

@pytest.mark.real_torch
@pytest.mark.parametrize("h,w,label", [(540, 960, "540p"), (2160, 3840, "UHD")])
def test_albedo_retinex_survives_a_4k_plate(h, w, label):
    """
    `reshape(B, -1)` gives H*W*3 elements per row. UHD is 24,883,200, past
    torch.quantile's 2**24 limit, so this raised
    "quantile() input tensor is too large" and took the Multipass Master node
    with it on every 4K frame.
    """
    from radiance.nodes.vfx.multipass.core import _albedo_retinex

    img = torch.rand(1, h, w, 3)
    out = _albedo_retinex(img, img.mean(-1))
    assert out.shape == img.shape
    assert torch.isfinite(out).all()


def test_the_quantile_guard_is_present_where_the_cap_applies():
    src = _src("nodes/vfx/multipass/core.py")
    assert "2 ** 24" in src or "2**24" in src, "the quantile size guard is gone"


# ── 6. The shipped workflow failed ComfyUI's prompt validation ──────────────

def test_shipped_workflows_match_current_widget_types():
    """
    `peak_nits` became a string combo; the bundled workflow still carried the
    JSON number 1000. ComfyUI's validate_inputs does `if val not in type_input`,
    and 1000 != "1000", so queueing the shipped template errored before
    execution with `value_not_in_list`.
    """
    from radiance.nodes.hdr.delivery import RadianceHDREncode

    spec = RadianceHDREncode.INPUT_TYPES()
    entry = (spec.get("required") or {}).get("peak_nits") \
        or (spec.get("optional") or {}).get("peak_nits")
    assert entry is not None, "peak_nits vanished from RadianceHDREncode"
    options = entry[0]
    doc = json.loads(_src("workflows/official/wan22_t2v_hdr_universal.json"))
    seen = 0
    for node in doc.get("nodes", []):
        if node.get("type") != "RadianceHDREncode":
            continue
        seen += 1
        value = (node.get("widgets_values") or [None, None])[1]
        assert value in options, (
            f"the shipped workflow sends peak_nits={value!r} ({type(value).__name__}), "
            f"which is not in {options}"
        )
    assert seen, "the RadianceHDREncode node vanished from the shipped workflow"
