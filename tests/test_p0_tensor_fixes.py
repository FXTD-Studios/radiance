"""v3.5.3 P0 regression tests: alpha, batch and immutability (FIX-008, 009, 010).

Runs real node classes from the registry on a 3-frame RGBA batch whose frames
differ, and checks through the public FUNCTION that

  * colour nodes leave alpha bit-identical (FIX-008),
  * every frame comes back and frames are not copies of frame 0 (FIX-009),
  * the input tensors are not modified (FIX-010).
"""
import pytest

torch = pytest.importorskip("torch")
if not hasattr(torch, "einsum"):
    pytest.skip("real torch required", allow_module_level=True)


def _frames(c=4, b=3, hdr=False):
    g = torch.Generator().manual_seed(0)
    x = torch.rand(b, 16, 16, c, generator=g) * 0.8 + 0.05
    for i in range(b):
        x[i, ..., :3] *= (0.5 + 0.4 * i)
    if hdr:
        x[..., :3] *= 2.5
    if c == 4:
        x[..., 3] = torch.linspace(0.1, 0.9, 16).view(1, 1, 16).expand(b, 16, 16)
    return x


def _defaults(cls, **override):
    kw = {}
    for k, v in (cls.INPUT_TYPES().get("required") or {}).items():
        t = v[0]
        o = v[1] if len(v) > 1 and isinstance(v[1], dict) else {}
        if isinstance(t, (list, tuple)):
            kw[k] = o.get("default", t[0])
        elif t in ("INT", "FLOAT", "BOOLEAN", "STRING"):
            kw[k] = o.get("default", {"INT": 0, "FLOAT": 0.0, "BOOLEAN": False, "STRING": ""}[t])
    kw.update(override)
    return kw


def _run(node_id, image_arg="image", hdr=False, **override):
    import radiance
    cls = radiance.NODE_CLASS_MAPPINGS[node_id]
    img = _frames(hdr=hdr)
    ref = img.clone()
    kw = _defaults(cls, **override)
    kw[image_arg] = img
    out = getattr(cls(), cls.FUNCTION)(**kw)
    out = out.get("result", out) if isinstance(out, dict) else out
    assert torch.equal(img, ref), f"{node_id} modified its input in place"
    return ref, out


# Colour / exposure / normalise nodes, with widget settings that change RGB.
ALPHA_CASES = [
    ("RadianceWhiteBalance", "image", {"mode": "Illuminant Adapt", "dst_illuminant": "A"}),
    ("RadianceACES2Tonescale", "image", {}),
    ("RadianceACES2ReachGamutCompress", "image", {}),
    ("RadianceACES2OutputTransformFull", "image", {}),
    ("RadianceACESTransform", "image", {}),
    ("RadianceColorSpaceConvert", "image", {}),
    ("RadianceHDRColorConvert", "image", {}),
    ("RadianceARRIWideGamut4", "image", {}),
    ("RadianceDaVinciWideGamut", "image", {}),
    ("RadianceCDLTransform", "image", {"slope_r": 1.3, "power_g": 0.8, "saturation": 0.7}),
    ("RadianceCurves", "image", {}),
    ("RadianceHueCurves", "image", {}),
    ("RadianceGrade", "image", {"preset": "Bleach Bypass"}),
    ("RadianceFloat32ColorCorrect", "image", {"exposure": 2.0, "contrast": 1.5}),
    ("RadianceFloat32Convert", "image", {"normalize": True, "source_gamma": 2.2}),
    ("RadianceGPUTensorOps", "image", {"operation": "Normalize"}),
    ("RadianceHDRShadowHighlight", "image", {}),
    ("RadianceACES2OutputTransform", "image", {}),
    ("RadianceHDRPerChannelNorm", "image", {}),
    ("RadianceHighlightSynthesis", "image", {}),
    ("RadianceRelightEngine", "image", {"normal_map": _frames(c=3)}),
    ("RadianceHDRSynthesisEngine", "image", {}),
    ("RadianceVideoHDRDecode", "image", {}),
    ("RadianceGradeApply", "image", {"exposure": 1.0, "offset": 0.0, "lift": 0.1, "gamma": 1.2,
                                     "gain": 1.1, "contrast": 1.3, "pivot": 0.18}),
]


@pytest.mark.parametrize("node_id,arg,override", ALPHA_CASES, ids=[c[0] for c in ALPHA_CASES])
def test_colour_nodes_keep_alpha_every_frame_and_input(node_id, arg, override):
    import radiance
    if node_id not in radiance.NODE_CLASS_MAPPINGS:
        pytest.skip(f"{node_id} not registered here")
    try:
        ref, out = _run(node_id, arg, hdr=True, **override)
    except TypeError as e:
        if "required positional argument" in str(e):
            pytest.skip(f"needs inputs beyond the widgets: {e}")
        raise
    image = out[0]
    assert image.shape[:3] == ref.shape[:3], f"{node_id}: {tuple(image.shape)} for {tuple(ref.shape)}"
    assert image.shape[-1] == 4, f"{node_id} dropped alpha"
    assert torch.equal(image[..., 3], ref[..., 3].to(image.dtype)), f"{node_id} changed alpha"
    # FIX-009: later frames are processed, not copies of frame 0
    assert not torch.allclose(image[1, ..., :3], image[0, ..., :3]), f"{node_id}: frame 1 == frame 0"


def test_shadow_highlight_processes_every_frame_like_a_single_frame():
    """FIX-009: frame i of the batch equals running frame i alone."""
    import radiance
    cls = radiance.NODE_CLASS_MAPPINGS["RadianceHDRShadowHighlight"]
    clip = _frames(c=3, hdr=True)
    (batch,) = cls().recover(clip)
    for i in range(clip.shape[0]):
        (single,) = cls().recover(clip[i:i + 1])
        torch.testing.assert_close(batch[i], single[0])


def test_hdr360_projects_every_frame():
    import radiance
    cls = radiance.NODE_CLASS_MAPPINGS["RadianceHDR360Generate"]
    clip = _frames(c=4)
    pano, uv = cls().generate(clip, output_width=512, output_height=256, exposure_adjust=1.0)
    assert pano.shape[0] == 3 and pano.shape[-1] == 4
    assert not torch.allclose(pano[2, ..., :3], pano[0, ..., :3])
    # exposure is colour: alpha in [0, 1] after the projection, never x2
    assert float(pano[..., 3].max()) <= 1.0 + 1e-5


def test_float32_convert_normalise_does_not_touch_upstream():
    """FIX-010: `.float()` on a float32 tensor is the same tensor; the per-frame
    normalise divided the upstream node's output in place."""
    import radiance
    cls = radiance.NODE_CLASS_MAPPINGS["RadianceFloat32Convert"]
    img = _frames(c=3, hdr=True)
    ref = img.clone()
    (out,) = cls().convert(img, normalize=True)
    assert torch.equal(img, ref)
    assert float(out[0].max()) <= 1.0 + 1e-6
