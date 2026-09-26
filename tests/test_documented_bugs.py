"""The bugs KNOWN_ISSUES listed under "Found while documenting every input"
(3.5.0), one test each. Every test fails on the code before the fix."""
import json
import os
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import torch


# ── ControlNet Apply ─────────────────────────────────────────────────────────

class _CN:
    """Records what ComfyUI's ControlNet.set_cond_hint receives."""
    def __init__(self, takes_vae=True):
        self.args, self.prev, self.takes_vae = None, None, takes_vae

    def copy(self):
        c = type(self)(self.takes_vae)
        c.args = self.args
        return c

    def set_extra_arg(self, k, v):
        pass

    def set_previous_controlnet(self, p):
        self.prev = p


class _CNVae(_CN):
    def set_cond_hint(self, hint, strength, pct, vae=None, extra_concat=[]):
        self.args = (tuple(hint.shape), vae)
        return self


class _CNOld(_CN):
    def set_cond_hint(self, hint, strength, pct):
        self.args = (tuple(hint.shape), "no-vae-arg")
        return self


def test_controlnet_apply_hint_is_channels_first_and_vae_is_passed():
    from radiance.nodes.generate.loader import RadianceControlNetApply
    cond = [[torch.zeros(1, 4, 8), {}]]
    img = torch.rand(1, 32, 48, 3)
    out = RadianceControlNetApply().apply_controlnet(cond, _CNVae(), img, 1.0, 0.0, 1.0, "auto", vae="VAE")[0]
    assert out[0][1]["control"].args == ((1, 3, 32, 48), "VAE")
    old = RadianceControlNetApply().apply_controlnet(cond, _CNOld(), img, 1.0, 0.0, 1.0)[0]
    assert old[0][1]["control"].args == ((1, 3, 32, 48), "no-vae-arg")
    assert "vae" in RadianceControlNetApply.INPUT_TYPES()["optional"]


# ── Upscale Video ────────────────────────────────────────────────────────────

def test_upscale_video_runs_on_the_compute_device(monkeypatch):
    import radiance.nodes.upscale.upscale as up
    seen = {}

    class Stop(Exception):
        pass

    def fake_build(**kw):
        seen.update(kw)
        raise Stop
    monkeypatch.setattr(up, "_compute_device", lambda t: torch.device("meta"))
    monkeypatch.setattr(up, "_build_upscale_fn", fake_build)
    with pytest.raises(Stop):
        up.RadianceUpscaleVideo().upscale_video(torch.rand(3, 16, 16, 3))
    assert seen["device"] == torch.device("meta")


def test_upscale_video_single_frame_keeps_its_settings(monkeypatch):
    import radiance.nodes.upscale.upscale as up
    seen = {}

    def fake_image(self, images, **kw):
        seen.update(kw)
        return images, images[..., :1], "RadianceUpscaleImage"
    monkeypatch.setattr(up.RadianceUpscaleImage, "upscale_image", fake_image)
    up.RadianceUpscaleVideo().upscale_video(
        torch.rand(1, 16, 16, 3), sharpness_boost=0.4, model_tier="tier3_creative (x)",
        enhancement_prompt="film", diffusion_steps=9)
    assert (seen["sharpness_boost"], seen["model_tier"], seen["enhancement_prompt"], seen["diffusion_steps"]) \
        == (0.4, "tier3_creative (x)", "film", 9)


# ── Multipass Relight, point light with a depth pass ────────────────────────

@pytest.mark.parametrize("near_is_white", [True, False])
def test_relight_point_light_near_pixels_are_nearer(near_is_white):
    from radiance.nodes.vfx.multipass.relight_comp import RadianceMultipassRelight
    H, W = 8, 16
    depth = torch.zeros(1, H, W, 3)
    near, far = (1.0, 0.0) if near_is_white else (0.0, 1.0)
    depth[:, :, : W // 2] = near
    depth[:, :, W // 2:] = far
    normals = torch.zeros(1, H, W, 3)
    normals[..., 0:2] = 0.5
    normals[..., 2] = 1.0                       # facing the camera
    out = RadianceMultipassRelight().relight(
        albedo=torch.ones(1, H, W, 3), normal_map=normals, depth_map=depth,
        light_type="Point", light_x=0.0, light_y=0.0, light_z=10.0, ambient=0.0,
        specular_intensity=0.0, depth_near_is_white=near_is_white)[0]
    left = float(out[:, :, W // 2 - 1].mean())
    right = float(out[:, :, W // 2].mean())
    assert left > 2 * right, "near pixels must be lit more strongly by a light in front"


# ── Video Batch Decode ───────────────────────────────────────────────────────

class _VAE:
    def __init__(self):
        self.got = None

    def decode(self, samples):
        self.got = samples.clone()
        return torch.zeros(1, 8, 8, 3)


def test_video_batch_decode_does_not_rescale_sampler_output():
    from radiance.nodes.video.t2v import RadianceVideoBatchDecode
    vae = _VAE()
    lat = torch.randn(1, 4, 8, 8)
    RadianceVideoBatchDecode().decode(vae, {"samples": lat}, dit_config=json.dumps({"latent_scale": 0.476986}))
    assert vae.got is not None and torch.equal(vae.got, lat)


# ── Video Model Info ─────────────────────────────────────────────────────────

class _Fmt:
    def __init__(self, ch):
        self.latent_channels = ch


class WAN22:
    def __init__(self, ch):
        self.latent_format = _Fmt(ch)


class _Patcher:
    def __init__(self, ch):
        self.model = WAN22(ch)


def test_model_info_reads_wan_channels_and_keeps_a_matching_preset():
    from radiance.nodes.video.t2v import RadianceVideoModelInfo
    spec = json.loads(RadianceVideoModelInfo().inspect(_Patcher(48), "LTX-Video (128ch)")[1])
    assert spec["model_name"] == "Wan2.2-TI2V-5B (48ch)" and spec["channels"] == 48
    spec = json.loads(RadianceVideoModelInfo().inspect(_Patcher(16), "Wan2.2-T2V-14B (16ch)")[1])
    assert spec["model_name"] == "Wan2.2-T2V-14B (16ch)"
    spec = json.loads(RadianceVideoModelInfo().inspect(_Patcher(16), "LTX-Video (128ch)")[1])
    assert spec["model_name"] == "Wan2.1 (16ch)"


# ── Video HDR Decode ─────────────────────────────────────────────────────────

def test_video_hdr_decode_white_lands_on_peak_and_preview_reaches_white():
    from radiance.nodes.video.hdr import RadianceVideoHDRDecode, _pq_encode
    img = torch.cat([torch.ones(1, 4, 4, 3), torch.full((1, 4, 4, 3), 0.46)])
    meta = '{"peak_nits":1000,"gamut":"BT.709"}'
    hdr, sdr, _ = RadianceVideoHDRDecode().decode(img, meta, "Reinhard")
    assert float(hdr[0].max()) == pytest.approx(float(_pq_encode(torch.tensor(0.1))), abs=1e-4)   # 1000 nits
    assert float(sdr[0].max()) == pytest.approx(1.0, abs=1e-4)
    srgb, _, _ = RadianceVideoHDRDecode().decode(img, meta, "Reinhard", output_eotf="sRGB / BT.1886")
    lin, _, _ = RadianceVideoHDRDecode().decode(img, meta, "Reinhard", output_eotf="Linear")
    assert float(srgb[0].max()) == pytest.approx(1.0, abs=1e-4)
    assert not torch.allclose(srgb, lin)


# ── HDR Color Pipeline ───────────────────────────────────────────────────────

def test_color_pipeline_converts_every_pair_and_adapts_once():
    from radiance.nodes.hdr import colorspace as C
    node = C.RadianceHDRColorPipeline()
    rgb = torch.tensor([[[[0.8, 0.2, 0.1]]]])
    for a in C._PRIMARIES_LIST:
        for b in C._PRIMARIES_LIST:
            if a != b:
                m = np.array(C._primaries_matrix(a, b))
                assert np.abs(m @ np.array(C._primaries_matrix(b, a)) - np.eye(3)).max() < 1e-3, (a, b)
    p3 = node.pipeline(rgb, "Linear (none)" if "Linear (none)" in C._EOTF_MAP else list(C._EOTF_MAP)[0], 0.0,
                       "Rec.709 (sRGB)", "DCI-P3 (D65)")[1]
    assert not torch.allclose(p3, rgb), "Rec.709 to P3 passed through unchanged"
    enc = "Linear (none)" if "Linear (none)" in C._EOTF_MAP else list(C._EOTF_MAP)[0]
    plain = node.pipeline(rgb, enc, 0.0, "Rec.709 (sRGB)", "ACEScg", "None")[1]
    twice = node.pipeline(rgb, enc, 0.0, "Rec.709 (sRGB)", "ACEScg", "D65_to_D60")[1]
    assert torch.equal(plain, twice), "Rec.709 to ACEScg adapted the white point twice"
    d65 = np.array([0.3127 / 0.3290, 1.0, (1 - 0.3127 - 0.3290) / 0.3290])
    d60 = np.array([0.32168 / 0.33767, 1.0, (1 - 0.32168 - 0.33767) / 0.33767])
    assert np.abs(np.array(C._BRADFORD_CAT["D65_to_D60"]) @ d65 - d60).max() < 1e-4


# ── Digital Cinema Read ──────────────────────────────────────────────────────

def test_digital_cinema_read_starts_a_video_at_its_first_frame(monkeypatch):
    import radiance.nodes.io.write as w
    seen = []

    def fake_read(self, **kw):
        seen.append(kw)
        return torch.zeros(1, 4, 4, 3), torch.zeros(1, 4, 4), {}
    monkeypatch.setattr(w.RadianceRead, "read", fake_read)
    w.RadianceDigitalCinemaRead().read("clip.mp4", "Auto", 1, 10, "sRGB (Standard)")
    w.RadianceDigitalCinemaRead().read("clip.mp4", "Video", 5, 0, "sRGB (Standard)")
    w.RadianceDigitalCinemaRead().read("shot.####.exr", "Sequence", 1001, 10, "sRGB (Standard)")
    assert (seen[0]["start_frame"], seen[0]["max_video_frames"]) == (0, 10)
    assert seen[1]["start_frame"] == 4
    assert (seen[2]["start_frame"], seen[2]["end_frame"]) == (1001, 1010)


# ── EXR MultiPart ────────────────────────────────────────────────────────────

def test_exr_multipart_writes_every_frame(tmp_path):
    OpenEXR = pytest.importorskip("OpenEXR")
    from radiance.nodes.io.write import RadianceEXRMultiPart
    beauty = torch.rand(3, 8, 12, 3)
    depth = torch.rand(1, 8, 12, 3)
    first = RadianceEXRMultiPart().write_multipart("mp", beauty, depth=depth,
                                                   output_path=str(tmp_path), frame_index=7)[0]
    assert os.path.basename(first) == "mp.0007.exr"
    assert sorted(os.listdir(tmp_path)) == ["mp.0007.exr", "mp.0008.exr", "mp.0009.exr"]
    f = OpenEXR.File(str(tmp_path / "mp.0009.exr"), separate_channels=True)
    parts = {p.name(): p for p in f.parts} if callable(getattr(f.parts[0], "name", None)) else None
    names = [getattr(p, "name", None) for p in f.parts]
    assert len(f.parts) == 2, names
    beauty_px = f.parts[0].channels["R"].pixels
    assert np.allclose(beauty_px, beauty[2, ..., 0].numpy(), atol=2e-3)


# ── Policy Guard and QC ──────────────────────────────────────────────────────

def test_policy_guard_reads_the_signal_it_is_given():
    from radiance.nodes.color.qc import RadiancePolicyGuard, POLICY_SIGNALS
    from radiance.color.encodings import _pq_encode
    white = torch.ones(1, 4, 4, 3)
    g = RadiancePolicyGuard()
    rep = lambda out: json.loads(out[2])["stats"]["peak_nits"]
    assert rep(g.run(image=white)) == pytest.approx(100.0)
    assert rep(g.run(image=white, signal=POLICY_SIGNALS[1])) == pytest.approx(203.0)
    pq1000 = torch.from_numpy(np.asarray(_pq_encode(np.full((1, 4, 4, 3), 1000 / 203), 203.0)))
    assert rep(g.run(image=pq1000, signal=POLICY_SIGNALS[2])) == pytest.approx(1000.0, rel=1e-3)
    out = g.run(image=white * 12, max_peak_nits=1000.0, signal=POLICY_SIGNALS[1])
    assert out[1] is False     # 12 x 203 nits is over 1000


def test_qc_fail_on_errors_stops_the_graph(monkeypatch):
    import importlib.util
    import radiance.nodes.color.qc as qc
    # conftest stubs radiance.image.defects; this test needs the real checks.
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "image", "defects.py")
    spec = importlib.util.spec_from_file_location("_real_defects", path)
    real = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(real)
    monkeypatch.setattr(qc, "defects", real)
    img = torch.full((1, 32, 32, 3), 1.5)            # every pixel above white: fails the clip check
    report = qc.RadianceQC().run("Analyze", img, fail_on_errors=False)
    assert "FAIL" in str(report[3]).upper() and "ERROR" not in str(report[3]).upper()
    with pytest.raises(RuntimeError, match="fail_on_errors"):
        qc.RadianceQC().run("Analyze", img, fail_on_errors=True)


def test_synthesis_guidance_nits_use_203():
    from radiance.nodes.hdr.synthesis import RadianceHDRSynthesisEngine
    torch.manual_seed(0)
    img = torch.rand(1, 32, 32, 3)
    mask = torch.ones(1, 32, 32)
    a = RadianceHDRSynthesisEngine().synthesize(img, 10.0, 3, 0.8)[0]
    b = RadianceHDRSynthesisEngine().synthesize(img, 1.0, 3, 0.8, guidance_mask=mask, guidance_nits=2030.0)[0]
    assert torch.allclose(a, b, atol=1e-5)


# ── AMF ──────────────────────────────────────────────────────────────────────

def test_amf_description_is_escaped():
    from radiance.nodes.hdr.aces2 import _build_amf, _parse_amf
    xml = _build_amf("A&B <clip>", "IDT", "ODT", 1000, 0.0001, "grade <v2> & notes")
    ET.fromstring(xml.encode())
    assert "grade <v2> & notes" in _parse_amf(xml)["description"]


# ── Regional prompts ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("mode", ["Additive", "Replace"])
def test_chained_regions_keep_their_own_strength(mode):
    from radiance.nodes.generate.regional import RadianceRegionalPrompt
    base = [[torch.zeros(1, 2, 4), {}]]
    r1 = [[torch.ones(1, 2, 4), {}]]
    r2 = [[torch.full((1, 2, 4), 2.0), {}]]
    n = RadianceRegionalPrompt()
    one = n.apply(base, r1, x=0.0, y=0.0, w=0.5, h=0.5, region_strength=0.9, global_strength=0.5, merge_mode=mode)[0]
    two = n.apply(one, r2, x=0.5, y=0.5, w=0.5, h=0.5, region_strength=0.7, global_strength=0.2, merge_mode=mode)[0]
    regions = [d for _, d in two if "area" in d]
    globals_ = [d for _, d in two if "area" not in d]
    assert sorted(d["strength"] for d in regions) == [0.7, 0.9]
    assert [d["strength"] for d in globals_] == [0.2]
    if mode == "Replace":
        m = globals_[0]["mask"]
        assert float(m[0, 10, 10]) == 0.0 and float(m[0, 200, 200]) == 0.0 and float(m[0, 10, 200]) == 1.0


# ── Sampler defaults ─────────────────────────────────────────────────────────

def test_sampler_sdr_defaults_match_the_widgets():
    import inspect
    from radiance.nodes.generate import sampler as s
    cls = next(c for c in vars(s).values() if isinstance(c, type) and "sdr_blend" in
               str(getattr(c, "INPUT_TYPES", lambda: {})()))
    opt = cls.INPUT_TYPES()["optional"]
    fn = getattr(cls, cls.FUNCTION)
    params = inspect.signature(fn).parameters
    for name in ("sdr_blend", "sdr_inject_steps", "sdr_decay"):
        assert params[name].default == opt[name][1]["default"], name
