"""VFX efficiency fixes (3.5.0): same output, bounded memory.

Six nodes built their work for the whole clip at once or looped in Python per
pixel. Measured on 24 frames of 1024x576 (CPU): Motion Blur ran out of memory
at default settings, Multipass Relight was killed out of memory, Linear Matting
took 107 s, HDR Stitch 24 s, and Floyd-Steinberg dither 8 s per frame. Depth Map
Generator kept every full-resolution depth frame, and a 3-channel copy, on the
GPU. Each now works a few frames at a time (on the GPU when there is one) and
uses separable filters or a wavefront where the maths allows.

These tests pin the outputs to the old maths, written out here as references,
and check that the chunked paths give the same result as a single pass.
"""
from __future__ import annotations


import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not isinstance(getattr(torch, "__version__", None), str):
    pytest.skip("needs real torch", allow_module_level=True)

import torch.nn.functional as F  # noqa: E402

pytestmark = pytest.mark.real_torch


@pytest.fixture
def tiny_chunks(monkeypatch):
    """Force one frame per chunk, so every chunk boundary is exercised."""
    import radiance.core.tensor.chunking as ch
    monkeypatch.setattr(ch, "chunk_budget_bytes", lambda device: 1)
    return ch


# ── chunking helper ──────────────────────────────────────────────────────────

def test_chunks_cover_the_batch_once():
    from radiance.core.tensor.chunking import chunks, frames_per_chunk
    assert list(chunks(7, 3)) == [(0, 3), (3, 6), (6, 7)]
    assert list(chunks(0, 4)) == []
    assert frames_per_chunk(10, 10, 3, 4.0, torch.device("cpu")) >= 1


# ── Motion Blur ──────────────────────────────────────────────────────────────

def test_shipped_motion_blur_chunked_equals_one_pass(monkeypatch):
    """The Motion Blur in the menu is nodes/vfx/motion_blur.py (vector blur);
    film/camera.py holds a rival class under the same key that is not loaded."""
    import radiance.core.tensor.chunking as ch
    from radiance.nodes.vfx.motion_blur import RadianceMotionBlur
    torch.manual_seed(5)
    img = torch.rand(3, 30, 40, 3) * 2
    vec = torch.randn(3, 30, 40, 3) * 5
    one = RadianceMotionBlur().apply(img, vec, 270.0, 8, False)[0]
    monkeypatch.setattr(ch, "chunk_budget_bytes", lambda device: 1)
    many = RadianceMotionBlur().apply(img, vec, 270.0, 8, False)[0]
    assert one.shape == img.shape and float((one - many).abs().max()) < 1e-6


# ── Linear Matting ───────────────────────────────────────────────────────────

def _guided_reference(image, mask, radius, eps):
    r = radius * 2 + 1
    pad = r // 2
    I = image.permute(0, 3, 1, 2)
    p = mask.unsqueeze(1)
    box = lambda x: F.avg_pool2d(x, r, stride=1, padding=pad)  # noqa: E731 - the old 2-D pool
    mI, mp = box(I), box(p)
    a = (box(I * p) - mI * mp) / (box(I * I) - mI * mI + eps)
    b = mp - a * mI
    q = (box(a) * I + box(b)).mean(1, keepdim=True).clamp(0, 1)
    return q.squeeze(1), image * q.permute(0, 2, 3, 1)


@pytest.mark.parametrize("radius", [0, 3, 12])
def test_matting_separable_filter_equals_the_2d_pool(radius, tiny_chunks):
    from radiance.nodes.vfx.masking import RadianceLinearMatting
    torch.manual_seed(1)
    img = torch.rand(3, 48, 64, 3) * 1.5
    mask = (torch.rand(3, 48, 64) > 0.5).float()
    alpha, fg = RadianceLinearMatting().apply(img, mask, "GuidedFilter", radius, 1e-3)
    ra, rf = _guided_reference(img, mask, radius, 1e-3)
    assert float((alpha - ra).abs().max()) < 1e-4
    assert float((fg - rf).abs().max()) < 1e-4


def test_matting_resizes_a_mask_of_another_size():
    """Load Image gives a 64x64 mask for an image with no alpha; this raised."""
    from radiance.nodes.vfx.masking import RadianceLinearMatting
    alpha, fg = RadianceLinearMatting().apply(torch.rand(1, 90, 120, 3), torch.zeros(1, 64, 64), "GuidedFilter", 4, 1e-3)
    assert alpha.shape == (1, 90, 120) and fg.shape == (1, 90, 120, 3)


def test_matting_accepts_one_mask_for_a_clip():
    from radiance.nodes.vfx.masking import RadianceLinearMatting
    img = torch.rand(4, 32, 32, 3)
    m = torch.zeros(1, 32, 32); m[:, 8:24, 8:24] = 1
    alpha, _ = RadianceLinearMatting().apply(img, m, "GuidedFilter", 4, 1e-3)
    assert alpha.shape == (4, 32, 32)


def test_matting_trimap_keeps_known_interiors_and_refines_only_the_band():
    from radiance.nodes.vfx.masking import RadianceLinearMatting
    image = torch.full((2, 32, 32, 3), 0.5)
    mask = torch.zeros(1, 32, 32)
    mask[:, :, 16:] = 1
    alpha, foreground = RadianceLinearMatting().apply(image, mask, "GuidedFilter", 3, 0.01)
    assert torch.all(alpha[:, :, :13] == 0)
    assert torch.all(alpha[:, :, 19:] == 1), "known foreground must survive at image borders"
    assert torch.any((alpha[:, :, 13:19] > 0) & (alpha[:, :, 13:19] < 1))
    torch.testing.assert_close(foreground, image * alpha.unsqueeze(-1))
    unchanged, _ = RadianceLinearMatting().apply(image, mask, "GuidedFilter", 0, 0.01)
    torch.testing.assert_close(unchanged, mask.expand(2, -1, -1), rtol=0, atol=0)


def test_matting_rejects_a_mismatched_mask_batch():
    from radiance.nodes.vfx.masking import RadianceLinearMatting
    with pytest.raises(ValueError, match="mask batch"):
        RadianceLinearMatting().apply(torch.zeros(3, 8, 8, 3), torch.zeros(2, 8, 8), "GuidedFilter", 2, 0.01)


# ── Multipass Relight ────────────────────────────────────────────────────────

@pytest.mark.parametrize("light_type", ["Directional", "Point"])
def test_relight_chunked_equals_one_pass(light_type, monkeypatch):
    import radiance.core.tensor.chunking as ch
    from radiance.nodes.vfx.multipass.relight_comp import RadianceMultipassRelight
    torch.manual_seed(2)
    B, H, W = 3, 24, 32
    kw = dict(albedo=torch.rand(B, H, W, 3), normal_map=torch.rand(B, H, W, 3),
              roughness=torch.rand(B, H, W, 1), ao=torch.rand(1, H, W, 1),
              depth_map=torch.rand(B, H, W, 1), light_type=light_type, output_premultiplied=True,
              alpha=torch.rand(B, H, W, 1))
    one = RadianceMultipassRelight().relight(**kw)
    monkeypatch.setattr(ch, "chunk_budget_bytes", lambda device: 1)
    many = RadianceMultipassRelight().relight(**kw)
    for a, b in zip(one[:5], many[:5]):
        assert a.shape == (B, H, W, 3)
        # Not torch.equal: a vectorised reduction can round differently with
        # the batch's memory alignment, and did once in about 50 runs.
        assert float((a - b).abs().max()) < 1e-6


def test_relight_decides_normal_encoding_on_the_whole_clip(tiny_chunks):
    """A clip whose later frame has signed normals is signed throughout, as
    before; a per-chunk decision would decode the frames differently."""
    from radiance.nodes.vfx.multipass.relight_comp import RadianceMultipassRelight, _decode_normal_map
    n = torch.rand(2, 8, 8, 3)
    n[1] = n[1] * 2 - 1
    out = RadianceMultipassRelight().relight(torch.ones(2, 8, 8, 3), n, ambient=0.0)[1]
    whole = _decode_normal_map(n, "OpenGL (Y-Up)")
    light = torch.nn.functional.normalize(torch.tensor([-0.35, 0.45, 1.0]), dim=0)
    ndotl = (whole * light).sum(-1).clamp(0, 1)
    assert float((out[..., 0] - ndotl).abs().max()) < 1e-5


# ── HDR Stitch ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("mode", ["Linear_Laplacian", "Linear_Gaussian", "Standard"])
def test_stitch_separable_feather_equals_the_2d_pool(mode, tiny_chunks):
    from radiance.nodes.vfx.inpaint import RadianceHDRCrop, RadianceHDRStitch
    torch.manual_seed(3)
    img = torch.randn(2, 60, 90, 3) + 0.5
    m = torch.zeros(2, 60, 90); m[:, 20:35, 30:55] = 1
    ci, cm, sd = getattr(RadianceHDRCrop(), RadianceHDRCrop.FUNCTION)(img, m, 1.5, 8)
    stitched, blend = RadianceHDRStitch().apply(img, ci * 2, cm, sd, mode, 6)
    orig_mask = torch.zeros(2, 60, 90)
    orig_mask[:, sd["ymin"]:sd["ymax"], sd["xmin"]:sd["xmax"]] = cm
    ref_blend = F.avg_pool2d(orig_mask.unsqueeze(1), 13, stride=1, padding=6).squeeze(1).clamp(0, 1)
    assert float((blend - ref_blend).abs().max()) < 1e-5
    assert stitched.shape == img.shape
    if mode in ("Standard", "Linear_Gaussian"):
        replacement = img.clone()
        replacement[:, sd["ymin"]:sd["ymax"], sd["xmin"]:sd["xmax"]] = ci * 2
        expected = torch.lerp(img, replacement, ref_blend.unsqueeze(-1))
        torch.testing.assert_close(stitched, expected)
        hard, _ = RadianceHDRStitch().apply(img, ci * 2, cm, sd, mode, 0)
        assert not torch.allclose(stitched, hard), "feather radius must affect the composite"
    # Far from the crop the plate is untouched (clamped at 0 for Laplacian).
    far = img[:, :5, :5]
    expect = far.clamp(min=0) if mode == "Linear_Laplacian" else far
    assert float((stitched[:, :5, :5] - expect).abs().max()) < 1e-5


# ── Floyd-Steinberg ──────────────────────────────────────────────────────────

def _fs_scanline(img, levels):
    """The pre-3.5.0 scan-line loop, verbatim arithmetic."""
    step = 1.0 / (levels - 1)
    out = img.float().numpy().copy()
    B, H, W, C = out.shape
    for b in range(B):
        for c in range(C):
            p = out[b, :, :, c]
            for y in range(H):
                for x in range(W):
                    old = p[y, x]
                    new = np.round(old / step) * step
                    err = old - new
                    p[y, x] = new
                    if x + 1 < W:
                        p[y, x + 1] += err * 7 / 16
                    if y + 1 < H:
                        if x > 0:
                            p[y + 1, x - 1] += err * 3 / 16
                        p[y + 1, x] += err * 5 / 16
                        if x + 1 < W:
                            p[y + 1, x + 1] += err * 1 / 16
    return torch.from_numpy(out)


@pytest.mark.parametrize("shape", [(1, 1, 9, 3), (1, 7, 1, 3), (2, 21, 33, 3)])
@pytest.mark.parametrize("bits", [1, 4, 8])
def test_floyd_steinberg_is_bit_identical_to_the_scanline_loop(shape, bits):
    from radiance.nodes.color.colorspace import RadianceBitDepthDegrade
    torch.manual_seed(4)
    img = torch.rand(*shape) * 1.2 - 0.1
    levels = max(2, 2 ** bits - 1)
    original = img.clone()
    actual = RadianceBitDepthDegrade._floyd_steinberg(img, levels)
    assert torch.equal(img, original), "dithering modified the upstream image"
    assert torch.equal(actual, _fs_scanline(original, levels))


# ── Depth Map Generator ──────────────────────────────────────────────────────

class _FakeProcessor:
    size = {"height": 28, "width": 28}
    ensure_multiple_of = 14
    image_mean = (0.5, 0.5, 0.5)
    image_std = (0.25, 0.25, 0.25)


class _FakeDepthModel(torch.nn.Module):
    """Depth = mean of the normalised input, at the model resolution."""
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(1))
        self.calls = []

    def forward(self, pixel_values):
        self.calls.append(tuple(pixel_values.shape))
        return type("O", (), {"predicted_depth": pixel_values.mean(1) * self.w})()


def _run_depth(monkeypatch, img, **kw):
    from radiance.nodes.vfx import depth
    model = _FakeDepthModel()
    monkeypatch.setattr(depth, "download_and_load_model", lambda size, dev: (model, _FakeProcessor()))
    out = depth.RadianceDepthMapGenerator().generate_depth(img, "Small (25M - Fast)", use_gpu=False, **kw)[0]
    return out, model


def test_depth_batches_frames_and_returns_a_cpu_image(monkeypatch):
    img = torch.rand(5, 40, 60, 3)
    out, model = _run_depth(monkeypatch, img)
    assert out.shape == (5, 40, 60, 3) and out.device.type == "cpu"
    assert max(c[0] for c in model.calls) > 1, "frames still go through the model one at a time"
    assert float(out.min()) == pytest.approx(0.0, abs=1e-6) and float(out.max()) == pytest.approx(1.0, abs=1e-6)


def test_depth_chunked_equals_one_pass(monkeypatch):
    import radiance.core.tensor.chunking as ch
    img = torch.rand(4, 30, 44, 3) * 1.3
    one, _ = _run_depth(monkeypatch, img, blur_edges=1.0)
    monkeypatch.setattr(ch, "chunk_budget_bytes", lambda device: 1)
    many, _ = _run_depth(monkeypatch, img, blur_edges=1.0)
    assert float((one - many).abs().max()) < 1e-6


def test_depth_resize_size_matches_the_processor():
    from radiance.nodes.vfx.depth import _dpt_resize_size
    try:
        from transformers.models.dpt.image_processing_dpt import get_resize_output_image_size
    except Exception:  # noqa: BLE001 - transformers absent or a different layout
        pytest.skip("transformers DPT processor not importable")
    for h, w in [(2160, 3840), (1086, 1920), (576, 1024), (1080, 1080), (300, 2000)]:
        ref = get_resize_output_image_size(torch.zeros(3, h, w), (518, 518), True, 14)
        ref = (ref.height, ref.width) if hasattr(ref, "height") else tuple(ref)   # SizeDict (v5) or tuple (v4)
        assert _dpt_resize_size(h, w, 518, 518, 14) == tuple(int(v) for v in ref)


# ── Phase 2: per-frame nodes, a few frames at a time ─────────────────────────

def _chunked_equals_one_pass(monkeypatch, fn):
    import radiance.core.tensor.chunking as ch
    one = fn()
    monkeypatch.setattr(ch, "chunk_budget_bytes", lambda device: 1)
    many = fn()
    one = one if isinstance(one, tuple) else (one,)
    many = many if isinstance(many, tuple) else (many,)
    for a, b in zip(one, many):
        if isinstance(a, torch.Tensor):
            assert a.shape == b.shape
            assert float((a - b).abs().max()) < 1e-5


def _clip(c=3, b=3, h=32, w=44, seed=6):
    torch.manual_seed(seed)
    return torch.rand(b, h, w, c) * 2


@pytest.mark.parametrize("node,call", [
    ("LensDistortion", lambda N, x: N().apply(x, -0.2, 0.05, 1.1, 0.5, 0.5, "border", False)),
    ("LensDistortion", lambda N, x: N().apply(x, -0.2, 0.05, 1.1, 0.5, 0.5, "zeros", True)),
    ("ChromaticAberration", lambda N, x: N().apply(x, 0.01, 0.0, -0.01, 0.5, 0.5, False)),
    ("AnamorphicStreaks", lambda N, x: N().apply(x, 0.8, 16, 0.2, 0.5, 1.0, 1.0, "Diagonal +45", 3.0)),
    ("FilmGrain", lambda N, x: N().apply(x, 1.0, 0.05, 0.1, 0.0, -0.1, True, 42)),
])
def test_optics_nodes_chunked_equal_one_pass(node, call, monkeypatch):
    import radiance.nodes.vfx.optics as optics
    x = _clip(c=4)
    _chunked_equals_one_pass(monkeypatch, lambda: call(getattr(optics, f"Radiance{node}"), x))


def test_lens_distortion_invert_runs_and_is_identity_at_zero():
    """invert passed a tensor as full_like's fill value and raised on every call."""
    from radiance.nodes.vfx.optics import RadianceLensDistortion
    x = _clip()
    out, st = RadianceLensDistortion().apply(x, 0.0, 0.0, 1.0, 0.5, 0.5, "border", True)
    assert float((out - x).abs().max()) < 1e-4
    assert st.shape == (3, 32, 44, 3)
    RadianceLensDistortion().apply(x, -0.5, 0.0, 1.0, 0.5, 0.5, "zeros", True)   # strong barrel: no raise


def test_film_grain_seed_still_gives_the_same_grain():
    """The noise is still one full-clip draw per channel from the global
    generator, so a seed reproduces the grain it gave before chunking."""
    from radiance.nodes.vfx.optics import RadianceFilmGrain
    x = _clip()
    a = RadianceFilmGrain().apply(x, 1.0, 0.05, 0.1, 0.0, -0.1, False, 7)[0]
    b = RadianceFilmGrain().apply(x, 1.0, 0.05, 0.1, 0.0, -0.1, False, 7)[0]
    c = RadianceFilmGrain().apply(x, 1.0, 0.05, 0.1, 0.0, -0.1, False, 8)[0]
    assert torch.equal(a, b) and not torch.equal(a, c)
    # Red channel, recomputed from the same draw: noise * strength, blurred at size 1.1.
    torch.manual_seed(7)
    red = torch.randn(3, 1, 32, 44) * 0.05
    red = RadianceFilmGrain._gaussian_blur_2d(red, 1.1)[:, 0]
    assert float((a[..., 0] - (x[..., 0] + red)).abs().max()) < 1e-5


@pytest.mark.parametrize("shape", ["Circle", "Hexagon"])
def test_depth_of_field_chunked_equals_one_pass(shape, monkeypatch):
    from radiance.film.camera import RadianceDepthOfField
    x = _clip()
    d = torch.rand(1, 16, 22, 3)
    _chunked_equals_one_pass(monkeypatch, lambda: RadianceDepthOfField().apply_dof(
        x, 10.0, d, 0.4, 0.1, shape, 2.0, False, use_gpu=False))


def test_stabilizer_finds_the_shift_and_chunks_agree(monkeypatch):
    from radiance.nodes.vfx.plate import RadianceSubpixelStabilizer
    torch.manual_seed(8)
    base = torch.rand(1, 48, 64, 3)
    seq = torch.cat([torch.roll(base, shifts=(i, 2 * i), dims=(1, 2)) for i in range(4)])
    _, disp = RadianceSubpixelStabilizer().apply(seq, 0, 64)
    assert disp[3, 0, 0, 0].item() == pytest.approx(-6.0, abs=0.05)
    assert disp[3, 0, 0, 1].item() == pytest.approx(-3.0, abs=0.05)
    assert disp[0].abs().max().item() == 0.0
    _chunked_equals_one_pass(monkeypatch, lambda: RadianceSubpixelStabilizer().apply(seq, 1, 64))


def test_grain_matcher_one_reference_frame(monkeypatch):
    from radiance.nodes.vfx.plate import RadianceHDRGrainMatcher
    tgt = _clip(b=5); ref = _clip(b=1, seed=9)
    _chunked_equals_one_pass(monkeypatch, lambda: RadianceHDRGrainMatcher().apply(tgt, ref, 1.0, 3, 1.0, 1.0, 1.0))


def test_relight_engine_and_composite_chunked_equal_one_pass(monkeypatch):
    from radiance.nodes.hdr.synthesis import RadianceRelightEngine
    from radiance.nodes.vfx.multipass.relight_comp import RadianceMultipassComposite
    x = _clip(c=4); nm = _clip(b=1, seed=10)
    _chunked_equals_one_pass(monkeypatch, lambda: RadianceRelightEngine().apply(
        x, nm, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.1))
    fg = _clip(); al = torch.rand(3, 32, 44, 1)
    _chunked_equals_one_pass(monkeypatch, lambda: RadianceMultipassComposite().composite(
        fg, al, background=_clip(b=1, seed=11), light_wrap=0.5, shadow_mask=torch.rand(3, 32, 44, 1)))


def _moving_clip(n=4, h=40, w=56):
    torch.manual_seed(3)
    base = torch.rand(1, h, w, 3)
    return torch.cat([torch.roll(base, (i, 2 * i), (1, 2)) for i in range(n)])


@pytest.mark.parametrize("solver", ["DIS", "Lucas-Kanade"])
def test_optical_flow_chunked_equals_one_pass(solver, monkeypatch):
    if solver == "DIS":
        pytest.importorskip("cv2")
    from radiance.nodes.vfx.motion import RadianceOpticalFlow
    x = _moving_clip()
    _chunked_equals_one_pass(monkeypatch, lambda: RadianceOpticalFlow().analyze(x, "Medium", 1.0, True, solver))


def test_optical_flow_lucas_kanade_is_solved_one_pair_at_a_time(monkeypatch):
    """LK's batched solve is not exactly its per-pair solve, so pairs stay single."""
    import radiance.nodes.vfx.motion as motion
    seen = []
    real = motion._optical_flow

    def spy(a, b, **kw):
        seen.append(a.shape[0])
        return real(a, b, **kw)
    monkeypatch.setattr(motion, "_optical_flow", spy)
    vec, vis, _ = motion.RadianceOpticalFlow().analyze(_moving_clip(5), "Fast", 1.0, False, "Lucas-Kanade")
    assert seen == [1, 1, 1, 1]
    assert float(vec[0].abs().max()) == 0.0 and float(vis.abs().max()) == 0.0


# ── Phase 3: the I/O-bound nodes ─────────────────────────────────────────────

def test_exr_passes_writer_parallel_files_equal_serial(tmp_path, monkeypatch):
    OpenEXR = pytest.importorskip("OpenEXR")
    import numpy as np
    import radiance.nodes.vfx.multipass.master as master
    torch.manual_seed(4)
    passes = {"beauty": torch.rand(4, 24, 32, 4), "normal": torch.rand(1, 24, 32, 3),
              "depth": torch.rand(4, 24, 32, 1)}
    dirs = {}
    for threads in (1, 4):
        monkeypatch.setattr(master, "_EXR_WRITE_THREADS", threads)
        d = tmp_path / f"t{threads}"
        first = master.RadianceEXRPassesWriter().write_passes(passes, "p", output_path=str(d), frame_index=10)[0]
        assert first.endswith("p.0010.exr")
        dirs[threads] = d
    names = sorted(p.name for p in dirs[1].iterdir())
    assert names == sorted(p.name for p in dirs[4].iterdir()) == [f"p.{i:04d}.exr" for i in range(10, 14)]
    for n in names:
        a = OpenEXR.File(str(dirs[1] / n), separate_channels=True).parts[0].channels
        b = OpenEXR.File(str(dirs[4] / n), separate_channels=True).parts[0].channels
        assert a.keys() == b.keys()
        for k in a:
            assert np.array_equal(a[k].pixels, b[k].pixels)


def test_compression_artifacts_any_size_and_same_noise_per_seed(monkeypatch):
    """A side that is not a multiple of block_size raised ValueError; encoding in
    parallel must not change the seeded noise."""
    import numpy as np
    from radiance.film.camera import RadianceCompressionArtifacts, _block_average
    x = _clip(c=4, b=3, h=100, w=100).clamp(0, 1)
    out = RadianceCompressionArtifacts().apply_artifacts(x, "Both", 60, 8, True, 32, 0.02, 9)[0]
    assert out.shape == x.shape and torch.equal(out[..., 3], x[..., 3])
    a = np.arange(35, dtype=np.float32).reshape(5, 7, 1)
    b = _block_average(a, 4)
    assert b[4, 6, 0] == pytest.approx(a[4:, 4:, 0].mean()) and b[0, 0, 0] == pytest.approx(a[:4, :4, 0].mean())
    import radiance.film.camera as cam
    monkeypatch.setattr(cam.os, "cpu_count", lambda: 1)
    serial = RadianceCompressionArtifacts().apply_artifacts(x, "Both", 60, 8, True, 32, 0.02, 9)[0]
    assert torch.equal(out, serial)


def test_scene_cut_analyses_each_frame_once(monkeypatch):
    import numpy as np
    import radiance.nodes.ai.scene_cut as sc
    calls = []
    real = sc._edge_map
    monkeypatch.setattr(sc, "_edge_map", lambda f: calls.append(1) or real(f))
    rng = np.random.default_rng(1)
    frames = rng.random((7, 30, 40, 3), dtype=np.float32)
    _, scores = sc.detect_cuts(frames, 0.5, 1, "combined")
    assert len(calls) == 7
    for i in range(6):
        want = 0.6 * sc._confidence(sc._histogram_diff(frames[i], frames[i + 1]), sc.HISTOGRAM_CUT_REFERENCE) \
            + 0.4 * sc._confidence(sc._edge_diff(frames[i], frames[i + 1]), sc.EDGE_CUT_REFERENCE)
        assert scores[i] == np.float32(want)
