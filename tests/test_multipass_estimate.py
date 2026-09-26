"""Multipass Estimate: pass maths, model plumbing and the RADIANCE_PASSES contract.

The models themselves are replaced with fakes that return known geometry and
intrinsics, so what is tested is everything Radiance does with a prediction:
axis conversion, AO, curvature, the lighting fit, consent, and the pass
bundle Relight and the EXR writer consume. The real-model check (MoGe-2 and
Marigold on a photo) is a manual release step, recorded in the CHANGELOG.
"""
from __future__ import annotations

import math
import os
import unittest
from unittest import mock

import pytest

torch = pytest.importorskip("torch")
if not isinstance(getattr(torch, "__version__", None), str):  # stubbed torch lane
    pytest.skip("needs real torch", allow_module_level=True)

from radiance.nodes.vfx.multipass import estimate as est  # noqa: E402
from radiance.nodes.vfx.multipass import estimate_models as em  # noqa: E402

pytestmark = pytest.mark.real_torch


def _pinhole(H, W, fov_deg=60.0):
    f = W / (2 * math.tan(math.radians(fov_deg) / 2))
    ys = torch.arange(H, dtype=torch.float32) + 0.5 - H / 2
    xs = torch.arange(W, dtype=torch.float32) + 0.5 - W / 2
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return f, gx, gy


def _sphere(H=128, W=128, R=1.0, dist=3.0):
    f, gx, gy = _pinhole(H, W)
    d = torch.nn.functional.normalize(torch.stack([gx / f, -gy / f, -torch.ones_like(gx)], -1), dim=-1)
    c = torch.tensor([0.0, 0.0, -dist])
    b = (d * c).sum(-1)
    disc = b * b - (c * c).sum() + R * R
    hit = disc > 0
    t = b - torch.sqrt(disc.clamp(min=0))
    P = d * t.unsqueeze(-1)
    N = torch.nn.functional.normalize(P - c, dim=-1)
    return torch.tensor([f]), P[None], N[None], (-P[..., 2])[None], hit[None]


class TestConventions(unittest.TestCase):
    def test_opencv_to_opengl_flips_y_and_z(self):
        v = torch.tensor([[1.0, 2.0, 3.0]])
        self.assertEqual(est.opencv_to_opengl(v).tolist(), [[1.0, -2.0, -3.0]])

    def test_normal_encoding_facing_camera_is_blue(self):
        n = torch.tensor([[[[0.0, 0.0, 1.0]]]])
        self.assertTrue(torch.allclose(est.encode_normals(n, "OpenGL (Y-Up)"), torch.tensor([0.5, 0.5, 1.0])))
        up = torch.tensor([[[[0.0, 1.0, 0.0]]]])
        self.assertAlmostEqual(float(est.encode_normals(up, "OpenGL (Y-Up)")[..., 1]), 1.0)
        self.assertAlmostEqual(float(est.encode_normals(up, "DirectX (Y-Down)")[..., 1]), 0.0)

    def test_srgb_round_trip(self):
        x = torch.linspace(0, 1, 101)
        self.assertTrue(torch.allclose(est.linear_to_srgb(est.srgb_to_linear(x)), x, atol=1e-5))

    def test_focal_from_normalised_intrinsics(self):
        K = torch.tensor([[[0.5, 0, 0.5], [0, 0.9, 0.5], [0, 0, 1]]])
        self.assertAlmostEqual(float(est.focal_px(K, 640)[0]), 320.0)


class TestGeometryPasses(unittest.TestCase):
    def test_ao_open_on_a_plane_and_dark_in_a_corner(self):
        H = W = 128
        f, gx, gy = _pinhole(H, W)
        valid = torch.ones(1, H, W, dtype=torch.bool)
        z = torch.full_like(gx, 3.0)
        P = torch.stack([gx / f * z, -gy / f * z, -z], -1)[None]
        N = torch.tensor([0.0, 0.0, 1.0]).expand(1, H, W, 3)
        vis = est.gtao(P, N, valid, torch.tensor([f]), 0.5)
        self.assertGreater(float(vis.mean()), 0.97)

        # Back wall at z=-4 meeting a floor at y=-1: a 90 degree crease.
        tf = torch.where(gy > 1e-3, f / gy.clamp(min=1e-3), torch.full_like(gy, 1e9))
        z = torch.minimum(torch.full_like(gx, 4.0), tf)
        P = torch.stack([gx / f * z, -gy / f * z, -z], -1)[None]
        floor = (P[..., 1] < -0.999)
        N = torch.where(floor.unsqueeze(-1), torch.tensor([0.0, 1.0, 0.0]), torch.tensor([0.0, 0.0, 1.0]))
        vis = est.gtao(P, N, valid, torch.tensor([f]), 0.5)
        seam = int(H / 2 + f / 4)
        self.assertLess(float(vis[0, seam - 1, W // 2]), 0.75)
        self.assertGreater(float(vis[0, 5, W // 2]), 0.95)

    def test_sphere_mean_curvature_is_one_over_radius(self):
        f, P, N, depth, hit = _sphere(R=1.0)
        k = est.mean_curvature(N, depth, hit, f, sigma_px=0.0)
        self.assertAlmostEqual(float(k[0][hit[0]].median()), 1.0, delta=0.15)

    def test_depth_edges_and_fill(self):
        z = torch.ones(1, 8, 8)
        z[:, :, 4:] = 5.0
        e = est.depth_edges(z, torch.ones_like(z, dtype=torch.bool))
        self.assertTrue(bool(e[0, 0, 3]) and bool(e[0, 0, 4]))
        self.assertFalse(bool(e[0, 0, 0]))
        x = torch.ones(1, 8, 8)
        x[e] = 0.0
        filled = est.fill_edges(x, e)
        self.assertTrue(torch.allclose(filled, torch.ones_like(filled)))

    def test_flying_pixels_do_not_occlude(self):
        H = W = 64
        f, gx, gy = _pinhole(H, W)
        z = torch.full_like(gx, 3.0)
        z[:, 31:33] = 2.9               # a thin ridge of in-between depths
        P = torch.stack([gx / f * z, -gy / f * z, -z], -1)[None]
        N = torch.tensor([0.0, 0.0, 1.0]).expand(1, H, W, 3)
        valid = torch.ones(1, H, W, dtype=torch.bool)
        ridge = torch.zeros_like(valid)
        ridge[:, :, 31:33] = True
        blocked = est.gtao(P, N, valid, torch.tensor([f]), 0.5)
        ignored = est.gtao(P, N, valid, torch.tensor([f]), 0.5, occluder_ok=~ridge)
        self.assertLess(float(blocked[0, 32, 28]), float(ignored[0, 32, 28]))
        self.assertGreater(float(ignored[0, 32, 28]), 0.97)


class TestLightingFit(unittest.TestCase):
    def test_recovers_scales(self):
        g = torch.Generator().manual_seed(0)
        d = torch.rand(32, 32, 3, generator=g)
        r = torch.rand(32, 32, 3, generator=g) * 0.2
        beauty = 0.8 * d + 1.5 * r
        a, b, err = est.fit_lighting(beauty, d, r, torch.ones(32, 32, dtype=torch.bool))
        self.assertAlmostEqual(a, 0.8, places=4)
        self.assertAlmostEqual(b, 1.5, places=4)
        self.assertLess(err, 1e-5)

    def test_scales_never_negative(self):
        g = torch.Generator().manual_seed(1)
        d = torch.rand(16, 16, 3, generator=g)
        r = torch.rand(16, 16, 3, generator=g)
        beauty = d - 0.3 * r
        a, b, _ = est.fit_lighting(beauty, d, r, torch.ones(16, 16, dtype=torch.bool))
        self.assertGreaterEqual(a, 0.0)
        self.assertGreaterEqual(b, 0.0)


class TestConsent(unittest.TestCase):
    def test_widget_and_environment(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            for k in ("RADIANCE_ALLOW_DOWNLOADS", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
                os.environ.pop(k, None)
            self.assertTrue(em.downloads_permitted(True))
            self.assertFalse(em.downloads_permitted(False))
            os.environ["RADIANCE_ALLOW_DOWNLOADS"] = "0"
            self.assertFalse(em.downloads_permitted(True))
            os.environ["RADIANCE_ALLOW_DOWNLOADS"] = "1"
            self.assertFalse(em.downloads_permitted(False))
            os.environ.pop("RADIANCE_ALLOW_DOWNLOADS")
            os.environ["HF_HUB_OFFLINE"] = "1"
            self.assertFalse(em.downloads_permitted(True))

    def test_refusal_names_the_file_and_the_source(self):
        with mock.patch.object(em, "find_moge", return_value=None), \
             mock.patch.object(em, "_first_writable", return_value=None):
            with self.assertRaisesRegex(em.EstimateModelError, "moge_2_vitl_normal_fp16"):
                em.ensure_moge(False)

    def test_sources_are_pinned(self):
        self.assertRegex(em.MOGE_REVISION, r"^[0-9a-f]{40}$")
        self.assertRegex(em.MOGE_SHA256, r"^[0-9a-f]{64}$")
        for src in em.MARIGOLD_SOURCES.values():
            self.assertRegex(src["revision"], r"^[0-9a-f]{40}$")
            self.assertIn("unet/diffusion_pytorch_model.fp16.safetensors", src["files"])
        self.assertIn("vae/diffusion_pytorch_model.fp16.safetensors",
                      em.MARIGOLD_SOURCES[em.MARIGOLD_APPEARANCE]["files"])


# ── Node with fake models ──────────────────────────────────────────────────

class _FakeMoGe:
    version = "v2"
    load_device = torch.device("cpu")
    dtype = torch.float32

    def infer(self, bchw, **kw):
        B, _, H, W = bchw.shape
        f, gx, gy = _pinhole(H, W, 60.0)
        z = torch.full_like(gx, 2.0)
        pts = torch.stack([gx / f * z, gy / f * z, z], -1)[None].expand(B, -1, -1, -1)   # OpenCV
        n = torch.tensor([0.0, 0.0, -1.0]).expand(B, H, W, 3)                            # facing camera (OpenCV)
        K = torch.tensor([[f / W, 0, 0.5], [0, f / H, 0.5], [0, 0, 1]]).expand(B, 3, 3)
        mask = torch.ones(B, H, W, dtype=torch.bool)
        mask[:, :2] = False                                                               # "sky" rows
        return {"points": pts, "depth": pts[..., 2], "normal": n, "intrinsics": K, "mask": mask}


class _FakeIID:
    def __init__(self):
        self.calls = []

    def to(self, device):
        return self

    def target_names(self, which):
        return ("albedo", "material") if which == "appearance" else ("albedo", "shading", "residual")

    def predict(self, which, x, **kw):
        self.calls.append(which)
        _, _, H, W = x.shape
        if which == "appearance":
            alb = torch.full((3, H, W), 0.5)
            mat = torch.stack([torch.full((H, W), 0.3), torch.full((H, W), 0.9), torch.zeros(H, W)])
            return torch.stack([alb, mat])
        lin = est.srgb_to_linear(x[0])
        return torch.stack([torch.full((3, H, W), 0.5), lin / 0.5 * 0.5, torch.zeros(3, H, W)])


class TestNode(unittest.TestCase):
    def _run(self, **kw):
        fake_iid = _FakeIID()
        with mock.patch.object(em, "load_moge", return_value=_FakeMoGe()), \
             mock.patch.object(em, "load_marigold", return_value=fake_iid), \
             mock.patch.object(em, "release_marigold"):
            beauty = torch.rand(2, 16, 24, 3, generator=torch.Generator().manual_seed(3)) * 0.9
            out = est.RadianceMultipassEstimate().estimate(beauty, **kw)
        return beauty, out, fake_iid

    def test_contract(self):
        node = est.RadianceMultipassEstimate
        self.assertEqual(len(node.RETURN_TYPES), len(node.RETURN_NAMES))
        self.assertEqual(len(node.RETURN_NAMES), len(node.OUTPUT_TOOLTIPS))
        inputs = node.INPUT_TYPES()
        for removed in ("emission", "transmission", "reflection_mask", "segmentation_id", "highpass"):
            self.assertNotIn(removed, node.RETURN_NAMES)
        self.assertIn("moge_model", inputs["optional"])

    def test_passes_bundle(self):
        beauty, out, iid = self._run()
        passes = out[0]
        for name in ("beauty", "albedo", "roughness", "metallic", "diffuse_lighting", "specular_lighting",
                     "normal", "depth", "world_position", "ao", "curvature", "geometry_mask", "motion_vector", "alpha"):
            self.assertIn(name, passes["_present"])
            self.assertEqual(tuple(passes[name].shape), (2, 16, 24, 3), name)
        # sRGB plate -> linear beauty
        self.assertTrue(torch.allclose(passes["beauty"], est.srgb_to_linear(beauty)))
        # metric geometry, OpenGL axes: in front of the camera is -z
        self.assertAlmostEqual(float(passes["depth"][0, 8, 12, 0]), 2.0, places=4)
        self.assertLess(float(passes["world_position"][0, 8, 12, 2]), 0.0)
        # facing the camera: encoded (0.5, 0.5, 1)
        self.assertTrue(torch.allclose(passes["normal"][0, 8, 12], torch.tensor([0.5, 0.5, 1.0]), atol=1e-5))
        # sky rows carry no geometry
        self.assertEqual(float(passes["geometry_mask"][0, 0, 0, 0]), 0.0)
        self.assertEqual(float(passes["depth"][0, 0, 0, 0]), 0.0)
        # materials straight from the model, albedo linearised
        self.assertAlmostEqual(float(passes["roughness"][0, 5, 5, 0]), 0.3, places=5)
        self.assertAlmostEqual(float(passes["metallic"][0, 5, 5, 0]), 0.9, places=5)
        self.assertAlmostEqual(float(passes["albedo"][0, 5, 5, 0]), float(est.srgb_to_linear(torch.tensor(0.5))), places=5)
        # diffuse + specular rebuild the beauty where the fake decomposition is exact
        rebuilt = passes["diffuse_lighting"] + passes["specular_lighting"]
        self.assertTrue(torch.allclose(rebuilt[:, 2:], passes["beauty"][:, 2:], atol=1e-4))
        # one model after the other, not interleaved per frame
        self.assertEqual(iid.calls, ["appearance", "appearance", "lighting", "lighting"])
        self.assertIn('"rebuild_error"', out[-1])

    def test_groups_can_be_switched_off(self):
        _, out, iid = self._run(materials=False, lighting=False, geometry=False, batch_is_sequence=False)
        self.assertEqual(sorted(out[0]["_present"]), ["alpha", "beauty"])
        self.assertEqual(iid.calls, [])
        self.assertIn("not_run", out[-1])

    def test_feeds_relight_and_writer(self):
        from radiance.nodes.vfx.multipass.relight_comp import RadianceMultipassRelight
        from radiance.nodes.vfx.multipass.master import RadianceEXRPassesWriter
        import tempfile
        _, out, _ = self._run()
        p = out[0]
        relit = RadianceMultipassRelight().relight(
            p["albedo"], p["normal"], roughness=p["roughness"], metallic=p["metallic"], ao=p["ao"],
            world_position=p["world_position"], light_type="Point", light_x=0.0, light_y=0.0, light_z=0.0,
        )[0]
        self.assertTrue(torch.isfinite(relit).all())
        self.assertGreater(float(relit.mean()), 0.0)
        try:
            path = RadianceEXRPassesWriter().write_passes(p, "estimate", output_path=tempfile.mkdtemp())[0]
        except (ImportError, RuntimeError) as exc:  # no EXR backend in this lane
            self.skipTest(str(exc))
        self.assertTrue(path.endswith(".exr"))


def test_moge_download_leaves_no_staging_folder(tmp_path, monkeypatch):
    """hf_hub_download's lock/metadata files used to stay behind in
    models/geometry_estimation/.radiance_download."""
    monkeypatch.setenv("RADIANCE_ALLOW_DOWNLOADS", "1")

    def fake_download(repo_id, filename, revision, local_dir):
        p = os.path.join(local_dir, filename)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        os.makedirs(os.path.join(local_dir, ".cache", "huggingface"), exist_ok=True)
        with open(p, "wb") as fh:
            fh.truncate(em.MOGE_SIZE)
        return p

    import huggingface_hub
    with mock.patch.object(em, "_geometry_dirs", return_value=[tmp_path]), \
         mock.patch.object(huggingface_hub, "hf_hub_download", fake_download):
        path = em.ensure_moge(True)
    assert path == tmp_path / em.MOGE_FILENAME and path.stat().st_size == em.MOGE_SIZE
    assert sorted(p.name for p in tmp_path.iterdir()) == [em.MOGE_FILENAME]
