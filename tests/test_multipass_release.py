"""Multipass for compositors (4.0 release review).

What a VFX artist gets when the passes land in Nuke, checked on files written
to disk and read back:

* normals are signed -1..1 (N.R/G/B), not 0..1 colour;
* depth Z is far away where there is no surface, never 0 (at the lens);
* the plate's encoding is detected (an EXR plate is not read as sRGB);
* a sequence keeps one field of view, so depth and position do not breathe;
* the camera is in the header (identity worldToCamera, FOV, focal in mm);
* AO has no dark frame border; P, forward/backward, one-channel scalars,
  chromaticities; depth stays 32-bit in EXR MultiPart;
* Marigold passes take the plate's edges back (guided upsampling);
* Relight can add a light to the plate and sees it in perspective.
"""
from __future__ import annotations

import json
import math
from unittest import mock

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not isinstance(getattr(torch, "__version__", None), str):  # stubbed torch lane
    pytest.skip("needs real torch", allow_module_level=True)
OpenEXR = pytest.importorskip("OpenEXR")

from radiance.hdr import io as hio  # noqa: E402
from radiance.nodes.vfx.multipass import estimate as est  # noqa: E402
from radiance.nodes.vfx.multipass import estimate_models as em  # noqa: E402

pytestmark = pytest.mark.real_torch

H, W = 24, 40


def _intrinsics(fov_deg: float) -> torch.Tensor:
    fx = 0.5 / math.tan(math.radians(fov_deg) / 2)
    return torch.tensor([[fx, 0, 0.5], [0, fx * W / H, 0.5], [0, 0, 1.0]])


class _FakeMoGe:
    """A wall 2 m away, facing the camera; sky in the top two rows. Without a
    given FOV each frame 'estimates' a different one, as MoGe does."""
    version = "v2"

    def __init__(self, free_fovs=(50.0, 62.0, 71.0, 44.0)):
        self.free_fovs = list(free_fovs)
        self.fov_args = []

    def infer(self, bchw, fov_x=None, **kw):
        i = len(self.fov_args)
        self.fov_args.append(fov_x)
        fov = fov_x if fov_x is not None else self.free_fovs[i % len(self.free_fovs)]
        K = _intrinsics(fov)
        fx, fy = float(K[0, 0]) * W, float(K[1, 1]) * H
        ys = torch.arange(H, dtype=torch.float32) + 0.5 - H / 2
        xs = torch.arange(W, dtype=torch.float32) + 0.5 - W / 2
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        z = torch.full_like(gx, 2.0)
        pts = torch.stack([gx / fx * z, gy / fy * z, z], -1)[None]        # OpenCV
        n = torch.tensor([0.0, 0.0, -1.0]).expand(1, H, W, 3)
        mask = torch.ones(1, H, W, dtype=torch.bool)
        mask[:, :2] = False
        return {"points": pts, "depth": pts[..., 2], "normal": n, "intrinsics": K[None], "mask": mask}


class _FakeIID:
    def to(self, device):
        return self

    def target_names(self, which):
        return ("albedo", "material") if which == "appearance" else ("albedo", "shading", "residual")

    def predict(self, which, x, **kw):
        _, _, h, w = x.shape
        if which == "appearance":
            mat = torch.stack([torch.full((h, w), 0.3), torch.full((h, w), 0.1), torch.zeros(h, w)])
            return torch.stack([torch.full((3, h, w), 0.5), mat])
        lin = est.srgb_to_linear(x[0])
        return torch.stack([torch.full((3, h, w), 0.5), lin, torch.zeros(3, h, w)])


def _run(beauty, moge=None, **kw):
    moge = moge or _FakeMoGe()
    with mock.patch.object(em, "load_moge", return_value=moge), \
         mock.patch.object(em, "load_marigold", return_value=_FakeIID()), \
         mock.patch.object(em, "release_marigold"):
        out = est.RadianceMultipassEstimate().estimate(beauty, **kw)
    return out, json.loads(out[-1]), moge


def _plate(n=3, peak=0.9):
    g = torch.Generator().manual_seed(5)
    return torch.rand(n, H, W, 3, generator=g) * peak


def _write_passes(passes, tmp_path, **kw):
    from radiance.nodes.vfx.multipass.master import RadianceEXRPassesWriter
    path, manifest = RadianceEXRPassesWriter().write_passes(passes, "shot", output_path=str(tmp_path), **kw)
    return path, json.loads(manifest)


def _read(path):
    with OpenEXR.File(path, separate_channels=True) as f:
        part = f.parts[0]
        return dict(part.header), {k: np.array(v.pixels) for k, v in part.channels.items()}


# ── Plate encoding ───────────────────────────────────────────────────────────

def test_auto_reads_an_hdr_plate_as_linear_and_an_image_as_srgb():
    hdr = _plate(1, peak=3.0)
    out, info, _ = _run(hdr, geometry=False, materials=False, lighting=False, batch_is_sequence=False)
    assert info["encoding"] == "Linear (scene) (Auto)"
    assert torch.allclose(out[0]["beauty"], hdr.clamp(min=0))          # linear kept, not decoded again
    sdr = _plate(1)
    out, info, _ = _run(sdr, geometry=False, materials=False, lighting=False, batch_is_sequence=False)
    assert info["encoding"] == "sRGB (display) (Auto)"
    assert torch.allclose(out[0]["beauty"], est.srgb_to_linear(sdr))


def test_auto_is_the_default_and_a_named_encoding_is_kept():
    spec = est.RadianceMultipassEstimate.INPUT_TYPES()["required"]["beauty_encoding"]
    assert spec[1]["default"] == "Auto"
    assert est.resolve_beauty_encoding(_plate(1, 3.0), "sRGB (display)") == "sRGB (display)"


# ── One camera per shot ──────────────────────────────────────────────────────

def test_a_sequence_holds_the_first_frames_field_of_view():
    out, info, moge = _run(_plate(4), materials=False, lighting=False)
    assert moge.fov_args[0] is None                                      # estimated once
    assert all(a == pytest.approx(50.0, abs=1e-3) for a in moge.fov_args[1:])
    assert info["geometry"]["fov_x_degrees"] == [50.0] * 4
    assert out[0]["_camera"]["fov_source"].startswith("estimated on frame 1")


def test_independent_images_keep_their_own_field_of_view():
    _, info, moge = _run(_plate(3), materials=False, lighting=False, batch_is_sequence=False)
    assert moge.fov_args == [None, None, None]
    assert info["geometry"]["fov_x_degrees"] == [50.0, 62.0, 71.0]


def test_a_given_field_of_view_is_used_on_every_frame():
    _, info, moge = _run(_plate(2), materials=False, lighting=False, fov_x_degrees=54.4)
    assert moge.fov_args == [54.4, 54.4]
    assert info["geometry"]["fov_source"] == "given"


# ── The EXR a compositor opens ───────────────────────────────────────────────

@pytest.fixture
def written(tmp_path):
    out, info, _ = _run(_plate(2), fov_x_degrees=54.4)
    path, manifest = _write_passes(out[0], tmp_path, frame_index=1001)
    header, ch = _read(path)
    return out, header, ch


def test_normals_are_signed_in_the_file(written):
    out, _, ch = written
    assert {"N.R", "N.G", "N.B"} <= set(ch)
    assert not any(k.startswith("normal.") for k in ch)
    n = np.stack([ch["N.R"], ch["N.G"], ch["N.B"]], -1)
    assert np.allclose(n[10, 10], [0.0, 0.0, 1.0], atol=1e-5)            # facing the camera: +Z
    assert np.allclose(np.linalg.norm(n, axis=-1), 1.0, atol=1e-4)
    # the IMAGE output keeps the 0..1 display encoding
    assert torch.allclose(out[7][0, 10, 10], torch.tensor([0.5, 0.5, 1.0]), atol=1e-5)


def test_directx_normals_are_flipped_back_to_y_up(tmp_path):
    n = np.array([0.2, 0.3, 0.9327], np.float32)
    enc_gl = (n * 0.5 + 0.5)
    enc_dx = (n * np.array([1, -1, 1], np.float32)) * 0.5 + 0.5
    for enc, conv in ((enc_gl, "OpenGL (Y-Up)"), (enc_dx, "DirectX (Y-Down)")):
        got = hio.signed_normals(np.broadcast_to(enc, (2, 2, 3)), conv, encoded=True)[0, 0]
        assert np.allclose(got, n / np.linalg.norm(n), atol=1e-3)


def test_renderer_signed_normals_pass_through():
    n = np.array([[[-0.6, 0.0, 0.8]]], np.float32)
    assert np.allclose(hio.signed_normals(n), n, atol=1e-6)


def test_sky_depth_is_far_not_at_the_lens(written):
    _, header, ch = written
    z = ch["Z"]
    assert z.dtype == np.float32
    assert float(z[0, 5]) == pytest.approx(hio.EMPTY_DEPTH_Z)          # sky rows
    assert float(z[10, 10]) == pytest.approx(2.0, abs=1e-5)             # the wall
    assert header["rad_depth_empty"] == pytest.approx(hio.EMPTY_DEPTH_Z)


def test_the_camera_is_in_the_header(written):
    _, header, _ = written
    assert np.allclose(header["worldToCamera"], np.eye(4))
    assert header["rad_fov_x_degrees"] == pytest.approx(54.4, abs=1e-3)
    assert header["rad_focal_length_mm"] == pytest.approx(18.0 / math.tan(math.radians(54.4) / 2), abs=1e-3)
    assert header["rad_haperture_mm"] == 36.0
    assert header["rad_vaperture_mm"] == pytest.approx(36.0 * H / W, abs=1e-3)
    assert "camera space" in header["rad_camera_space"]


def test_layers_carry_the_names_nuke_reads(written):
    _, header, ch = written
    for name in ("P.R", "P.G", "P.B", "backward.u", "backward.v", "forward.u", "forward.v",
                 "ao.Y", "roughness.Y", "metallic.Y", "curvature.Y", "geometry_mask.Y",
                 "albedo.R", "diffuse_lighting.R", "R", "G", "B", "A", "Z"):
        assert name in ch, name
    for gone in ("world_position.R", "MV.X", "ao.R", "ao.G", "roughness.B"):
        assert gone not in ch, gone
    assert float(ch["P.B"][10, 10]) == pytest.approx(-2.0, abs=1e-5)    # in front of the camera
    assert tuple(np.round(header["chromaticities"], 4)) == hio.REC709_CHROMATICITIES


def test_forward_vectors_point_to_the_next_frame():
    rng = np.random.default_rng(0)
    import cv2
    base = cv2.GaussianBlur(rng.random((H + 40, W + 40, 3)).astype(np.float32), (0, 0), 1.5)
    f1 = base[20:20 + H, 20:20 + W]
    f2 = base[20:20 + H, 17:17 + W]                                     # content moves 3 px right
    out, _, _ = _run(torch.from_numpy(np.stack([f1, f2])), geometry=False, materials=False, lighting=False)
    fwd = out[0]["motion_forward"][0, 6:18, 8:32].numpy()
    bwd = out[0]["motion_vector"][1, 6:18, 8:32].numpy()
    assert float(np.median(fwd[..., 0])) == pytest.approx(3.0, abs=0.3)
    assert float(np.median(bwd[..., 0])) == pytest.approx(-3.0, abs=0.3)
    assert float(out[0]["motion_forward"][-1].abs().max()) == 0.0      # no next frame


def test_the_hdr_master_can_be_the_files_beauty(tmp_path):
    out, _, _ = _run(_plate(1), materials=False, lighting=False, batch_is_sequence=False)
    master = torch.full((1, H, W, 3), 4.5)
    path, _ = _write_passes(out[0], tmp_path, beauty=master)
    _, ch = _read(path)
    assert float(ch["R"][5, 5]) == pytest.approx(4.5)
    with pytest.raises(ValueError, match="the beauty input"):
        _write_passes(out[0], tmp_path, beauty=torch.ones(1, H + 2, W, 3))


def test_multi_part_layout_uses_the_same_names(tmp_path):
    out, _, _ = _run(_plate(1), materials=False, lighting=False, batch_is_sequence=False)
    path, _ = _write_passes(out[0], tmp_path, exr_layout="Multi-part")
    with OpenEXR.File(path, separate_channels=True) as f:
        parts = {p.name(): {k: np.array(c.pixels) for k, c in p.channels.items()} for p in f.parts}
    assert set(parts["normal"]) == {"N.R", "N.G", "N.B"}
    assert set(parts["depth"]) == {"Z"} and parts["depth"]["Z"].dtype == np.float32
    assert float(parts["depth"]["Z"][0, 0]) == pytest.approx(hio.EMPTY_DEPTH_Z)
    assert set(parts["ao"]) == {"ao.Y"}


def test_exr_multipart_node_signs_normals_and_keeps_depth_float(tmp_path):
    from radiance.nodes.io.write import RadianceEXRMultiPart
    out, _, _ = _run(_plate(1), materials=False, lighting=False, batch_is_sequence=False)
    path, _ = RadianceEXRMultiPart().write_multipart(
        "plate", out[1], "16-bit Half Float", "ZIP", depth=out[8], normal=out[7],
        output_path=str(tmp_path), frame_index=1001)
    with OpenEXR.File(path, separate_channels=True) as f:
        parts = {p.name(): {k: np.array(c.pixels) for k, c in p.channels.items()} for p in f.parts}
    z = parts["depth"]["Z"]
    assert z.dtype == np.float32
    assert float(z[0, 0]) == pytest.approx(hio.EMPTY_DEPTH_Z)
    assert float(parts["normal"]["N.B"][10, 10]) == pytest.approx(1.0, abs=1e-3)
    assert float(parts["normal"]["N.R"][10, 10]) == pytest.approx(0.0, abs=1e-3)
    assert parts["beauty"]["R"].dtype == np.float16              # colour stays as asked


def test_read_aovs_reads_the_file_back(tmp_path):
    from radiance.nodes.vfx.multipass.aov_reader import RadianceMultipassAOVReader
    out, _, _ = _run(_plate(2), materials=False, lighting=False)
    path, _ = _write_passes(out[0], tmp_path)
    try:
        res = RadianceMultipassAOVReader().read_passes(path)
    except Exception as exc:  # noqa: BLE001 - reader backend missing in this lane
        pytest.skip(f"AOV reader unavailable here: {exc}")
    passes = res[0]
    assert float(passes["normal"][0, 10, 10, 2]) == pytest.approx(1.0, abs=1e-3)      # signed N
    assert float(passes["world_position"][0, 10, 10, 2]) == pytest.approx(-2.0, abs=1e-4)
    assert "motion_vector" in passes["_present"]


# ── AO and detail ────────────────────────────────────────────────────────────

def test_ao_has_no_dark_frame_border():
    Hh, Ww = 64, 96
    f = Ww / (2 * math.tan(math.radians(30)))
    ys = torch.arange(Hh, dtype=torch.float32) + 0.5 - Hh / 2
    xs = torch.arange(Ww, dtype=torch.float32) + 0.5 - Ww / 2
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    z = torch.full_like(gx, 5.0)
    P = torch.stack([gx / f * z, -gy / f * z, -z], -1)[None]             # OpenGL, flat wall
    N = torch.tensor([0.0, 0.0, 1.0]).expand(1, Hh, Ww, 3)
    valid = torch.ones(1, Hh, Ww, dtype=torch.bool)
    ao = est.gtao(P, N, valid, torch.tensor([f]), 0.5)
    assert float(ao.min()) > 0.98, float(ao.min())                      # was 0.84 in the corners


def test_guided_upsampling_takes_the_plates_edges():
    guide = torch.zeros(1, 32, 64)
    guide[:, :, 32:] = 1.0                                               # a hard edge in the plate
    soft = torch.linspace(0, 1, 64).clamp(0, 1).view(1, 1, 64, 1).expand(1, 32, 64, 1)
    soft = ((torch.arange(64).float() - 24) / 16).clamp(0, 1).view(1, 1, 64, 1).expand(1, 32, 64, 1)
    sharp = est.guided_upsample(soft.contiguous(), guide, radius=4)
    jump = lambda x: float(x[0, 16, 33, 0] - x[0, 16, 30, 0])            # noqa: E731
    assert jump(sharp) > jump(soft) * 1.5
    flat = torch.full((1, 32, 64, 1), 0.4)
    assert torch.allclose(est.guided_upsample(flat, guide, 4), flat, atol=1e-4)


def test_material_detail_runs_only_when_the_plate_is_larger(monkeypatch):
    seen = []
    real = est.guided_upsample
    monkeypatch.setattr(est, "guided_upsample", lambda *a, **k: seen.append(a[2]) or real(*a, **k))
    _, info, _ = _run(_plate(1), geometry=False, lighting=False, batch_is_sequence=False, material_resolution=16)
    assert seen and set(seen) == {3}                                    # 40 px plate / 16 px model
    assert info["materials"]["detail"] == "guided filter, radius 3 px"
    seen.clear()
    _, info, _ = _run(_plate(1), geometry=False, lighting=False, batch_is_sequence=False, material_resolution=768)
    assert not seen and info["materials"]["detail"] == "resize only"
    _, info, _ = _run(_plate(1), geometry=False, lighting=False, batch_is_sequence=False,
                      material_resolution=16, material_detail=False)
    assert info["materials"]["detail"] == "resize only"


# ── Relight ──────────────────────────────────────────────────────────────────

def test_add_light_keeps_the_plate():
    from radiance.nodes.vfx.multipass.relight_comp import RadianceMultipassRelight
    beauty = torch.rand(1, 8, 8, 3)
    albedo = torch.full((1, 8, 8, 3), 0.5)
    normal = torch.tensor([0.5, 0.5, 1.0]).expand(1, 8, 8, 3)
    node = RadianceMultipassRelight()
    off = node.relight(albedo, normal, beauty=beauty, intensity=0.0, ambient=1.0,
                       relight_output="Add light to beauty")[0]
    assert torch.allclose(off, beauty, atol=1e-6)                       # no new light: the plate itself
    on = node.relight(albedo, normal, beauty=beauty, intensity=1.0, specular_intensity=0.0,
                      light_x=0.0, light_y=0.0, light_z=1.0, relight_output="Add light to beauty")[0]
    assert torch.allclose(on, beauty + 0.5, atol=1e-5)                  # + albedo x N.L x 1
    with pytest.raises(ValueError, match="needs the beauty"):
        node.relight(albedo, normal, relight_output="Add light to beauty")


def test_specular_is_seen_in_perspective_with_a_position_pass():
    from radiance.nodes.vfx.multipass.relight_comp import RadianceMultipassRelight
    n = 33
    xs = torch.linspace(-2.0, 2.0, n)
    P = torch.stack([xs.view(1, n).expand(n, n), -xs.view(n, 1).expand(n, n), torch.full((n, n), -2.0)], -1)[None]
    albedo = torch.full((1, n, n, 3), 0.2)
    normal = torch.tensor([0.5, 0.5, 1.0]).expand(1, n, n, 3)            # a wall facing the camera
    kw = dict(roughness=torch.full((1, n, n, 3), 0.2), light_type="Directional",
              light_x=0.0, light_y=0.0, light_z=1.0, ambient=0.0)
    node = RadianceMultipassRelight()
    ortho = node.relight(albedo, normal, **kw)[2]
    persp = node.relight(albedo, normal, world_position=P, **kw)[2]
    # Orthographic: the same highlight everywhere. Perspective: it peaks where
    # the camera ray meets the wall head-on (the centre) and falls off.
    assert float(ortho.std()) < 1e-6
    assert float(persp[0, n // 2, n // 2, 0]) > float(persp[0, 0, 0, 0]) * 2
