"""RUDRA AI fill v3 in Radiance: the port of rudra/genfill and its node.

The Wan VACE model itself needs a GPU and the base weights, so the model is
replaced by a stand-in that "draws" a brighter fill; everything around it
(windows, masks, brightness, the round-trip-safe blend, the node) is real.
"""
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("PyOpenColorIO")
pytest.importorskip("cv2")


def _g():
    return importlib.import_module("radiance.model.rudra_genfill")


def _view():
    return importlib.import_module("radiance.model.rudra_aces2").aces2_view()


class TestPieces:
    def test_encoding_round_trips_scene_light(self):
        g = _g()
        x = np.array([0.0, 1e-4, 0.18, 1.0, 37.0, 4000.0], np.float32)
        y = g.encode(x)
        assert y.min() >= -1 and y.max() <= 1
        assert np.allclose(g.decode(y), x, rtol=1e-4, atol=1e-6)

    @pytest.mark.parametrize("n", [1, 5, 33, 34, 70])
    def test_windows_write_every_frame_once(self, n):
        g = _g()
        plan = g.plan_windows(n, 33, 8)
        written = [i for _, new, end in plan for i in range(new, end)]
        assert written == list(range(n))
        assert all(end - start <= 33 for start, _, end in plan)

    def test_working_size_is_a_multiple_of_16_within_area(self):
        g = _g()
        h, w = g.working_size(1080, 1920, 832 * 480)
        assert h % 16 == 0 and w % 16 == 0 and h * w <= 832 * 480

    def test_blend_keeps_every_code(self):
        g = _g()
        v = _view()
        codes = np.full((8, 8, 3), 200 / 255, np.float32)
        codes[2:6, 2:6] = 1.0
        master = v.inverse(codes)
        fill = master * 6.0                  # brighter everywhere
        weight = np.ones((8, 8), np.float32)
        out, alpha = g.r_safe_blend(master, fill, weight, v)
        before = np.rint(np.clip(v.forward(master), 0, 1) * 255)
        after = np.rint(np.clip(v.forward(out), 0, 1) * 255)
        assert np.array_equal(before, after)
        assert (alpha[2:6, 2:6] > 0).all()          # clipped: free to brighten
        assert alpha[0, 0] < 1.0                    # below clip: pulled back

    def test_brightness_estimator_format_is_checked(self):
        g = _g()
        with pytest.raises(ValueError):
            g.BrightnessEstimator({"format": "something else"})

    def test_lora_merges_without_peft(self, tmp_path, monkeypatch):
        g = _g()
        from safetensors.torch import save_file
        import builtins
        real_import = builtins.__import__

        def no_peft(name, *a, **k):
            if name.startswith("peft"):
                raise ImportError("no peft")
            return real_import(name, *a, **k)
        monkeypatch.setattr(builtins, "__import__", no_peft)

        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attn1 = torch.nn.Module()
                self.attn1.to_q = torch.nn.Linear(6, 6, bias=False)

        tr = torch.nn.Module()
        tr.blocks = torch.nn.ModuleList([Block()])
        w0 = tr.blocks[0].attn1.to_q.weight.detach().clone()
        a, b = torch.randn(2, 6), torch.randn(6, 2)
        f = tmp_path / "l.safetensors"
        save_file({"blocks.0.attn1.to_q.lora_A.weight": a, "blocks.0.attn1.to_q.lora_B.weight": b}, str(f),
                  metadata={"format": "rudra_genfill_lora_v1", "rank": "2", "alpha": "4",
                            "targets": json.dumps(["to_q"])})
        meta = g.load_lora(tr, f)
        assert meta["applied_with"].startswith("merged")
        assert torch.allclose(tr.blocks[0].attn1.to_q.weight, w0 + 2.0 * (b @ a), atol=1e-5)

    def test_not_a_rudra_lora_is_refused(self, tmp_path):
        g = _g()
        from safetensors.torch import save_file
        f = tmp_path / "x.safetensors"
        save_file({"w": torch.zeros(1)}, str(f), metadata={"format": "kohya"})
        with pytest.raises(ValueError, match="not a RUDRA genfill LoRA"):
            g.load_lora(torch.nn.Linear(1, 1), f)


class _FakeModel:
    """Stands in for WanFill: returns the context video 3 stops brighter."""
    multiple = 16
    identity = {"base": "fake"}

    class _P:
        def get(self, text):
            return None

        def release(self):
            pass

    prompts = _P()

    def __init__(self):
        self.calls = []

    def context(self, video, mask):
        self.video = video
        return video

    def denoise(self, ctx, s, regions, progress=None):
        self.calls.append((ctx.shape, s.prompt_highlight))
        if progress:
            progress(1, 1)
        return ctx

    def decode_latents(self, z):
        g = _g()
        lin = g.decode(z.numpy())
        return torch.from_numpy(g.encode(lin * 8.0).astype(np.float32))


def _masters(n=3):
    v = _view()
    out = []
    for i in range(n):
        c = np.full((48, 64, 3), 0.5, np.float32)
        c[10:30, 20 + i:40 + i] = 1.0               # a clipped area that moves
        out.append(v.inverse(c))
    return out


def test_fill_frames_only_touches_full_clip_and_keeps_codes():
    g = _g()
    v = _view()
    masters = _masters()
    model = _FakeModel()
    outs, alphas, rep = g.fill_frames(masters, model, v, g.FillSettings(steps=1),
                                      g.MaskSettings(), window=9, overlap=2, max_area=48 * 64)
    assert rep["stats"]["windows_generated"] == 1
    for m, o, a in zip(masters, outs, alphas):
        codes = np.rint(np.clip(v.forward(m), 0, 1) * 255)
        assert np.array_equal(codes, np.rint(np.clip(v.forward(o), 0, 1) * 255))
        full = (codes >= 255).all(-1)
        assert (a[~full] == 0).all()
        assert np.array_equal(o[a == 0], m[a == 0])     # untouched: bit-identical
        y = lambda x: x @ g.AP1_Y  # noqa: E731
        assert (y(o) <= y(m) * 2 ** 3.0 + 1e-3).all()   # never more than 3 stops over


def test_brightness_offset_moves_the_fill(tmp_path):
    g = _g()
    v = _view()
    masters = _masters(1)
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(14, 32), torch.nn.GELU(), torch.nn.Linear(32, 32),
                              torch.nn.GELU(), torch.nn.Linear(32, 1))
    state = {"format": "rudra_genfill_brightness_v1", "features": g.FEATURES,
             "mu": [0.0] * 14, "sd": [1.0] * 14,
             "weights": {k: t.tolist() for k, t in net.state_dict().items()}}
    est = g.BrightnessEstimator(state)
    lo = g.fill_frames(masters, _FakeModel(), v, g.FillSettings(steps=1), g.MaskSettings(), window=9,
                       overlap=2, max_area=48 * 64,
                       brightness=g.BrightnessSettings(est, offset=-3.0))[0][0]
    hi = g.fill_frames(masters, _FakeModel(), v, g.FillSettings(steps=1), g.MaskSettings(), window=9,
                       overlap=2, max_area=48 * 64,
                       brightness=g.BrightnessSettings(est, offset=+1.0))[0][0]
    assert (hi @ g.AP1_Y).sum() > (lo @ g.AP1_Y).sum()


# ─────────────────────────────────────────────────────────────────────────────
#  node
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def node_env(monkeypatch, tmp_path):
    g = _g()
    pkg = tmp_path / "rudra_genfill_v3"
    pkg.mkdir()
    monkeypatch.setattr(g, "resolve_package", lambda *a: pkg)
    monkeypatch.setattr(g, "resolve_base", lambda *a: str(tmp_path / "base"))
    model = _FakeModel()
    monkeypatch.setattr(g, "get_model", lambda *a, **k: model)
    monkeypatch.setattr(g, "check_runtime", lambda: None)
    return model


def _linear_master(n=2):
    a = importlib.import_module("radiance.model.rudra_aces2")
    ms = _masters(n)
    return torch.from_numpy(np.stack([a.acescg_to(m, a.M_AP1_TO_REC709) for m in ms]))


def test_node_fills_and_reports(node_env):
    mod = importlib.import_module("radiance.nodes.hdr.rudra_aifill")
    img = _linear_master()
    out, gen, recon, report = mod.RadianceSDRToHDRAIFill().fill(
        img, "Linear (Rec.709)", "soft clouds", 0.0, use_brightness_estimator=False,
        steps=1, window=9, overlap=2, max_area=48 * 64)
    assert out.shape == img.shape and gen.shape == img.shape[:3]
    assert float(gen.max()) > 0
    keep = gen == 0
    assert torch.equal(out[keep], img[keep])
    assert "plausible, not recovered" in report
    assert node_env.calls[0][1] == "soft clouds"


def test_node_with_nothing_clipped_returns_the_input(node_env):
    mod = importlib.import_module("radiance.nodes.hdr.rudra_aifill")
    img = torch.full((1, 16, 16, 3), 0.1)
    out, gen, recon, report = mod.RadianceSDRToHDRAIFill().fill(img, "Linear (Rec.709)", "", 0.0)
    assert out is img and float(gen.max()) == 0
    assert "nothing to fill" in report
    assert not node_env.calls


def test_node_says_how_to_install_when_missing(monkeypatch):
    g = _g()
    mod = importlib.import_module("radiance.nodes.hdr.rudra_aifill")
    monkeypatch.setattr(g, "resolve_package", lambda *a: None)
    monkeypatch.setattr(g, "check_runtime", lambda: None)
    monkeypatch.setenv("RADIANCE_ALLOW_DOWNLOADS", "0")
    with pytest.raises(RuntimeError, match="rudra_genfill_v3"):
        mod.RadianceSDRToHDRAIFill().fill(_linear_master(1), "Linear (Rec.709)", "", 0.0)


def test_first_run_downloads_package_and_base(monkeypatch, tmp_path):
    """Nothing installed: the node fetches the AI fill package and the Wan base itself."""
    g = _g()
    dl = importlib.import_module("radiance.model.rudra_download")
    mod = importlib.import_module("radiance.nodes.hdr.rudra_aifill")
    monkeypatch.setattr(g, "resolve_package", lambda *a: None)
    monkeypatch.setattr(g, "resolve_base", lambda *a: None)
    monkeypatch.setattr(g, "check_runtime", lambda: None)
    fetched = []

    def fake_download(pkg, **kw):
        fetched.append(pkg.name)
        return tmp_path / pkg.dirname
    monkeypatch.setattr(dl, "download_package", fake_download)
    model = _FakeModel()
    monkeypatch.setattr(g, "get_model", lambda *a, **k: model)
    mod.RadianceSDRToHDRAIFill().fill(_linear_master(1), "Linear (Rec.709)", "", 0.0,
                                      use_brightness_estimator=False, steps=1, window=9,
                                      overlap=2, max_area=48 * 64)
    assert fetched == ["rudra_genfill_v3", "wan_vace_base"]


def test_old_diffusers_is_named_before_any_download(monkeypatch):
    g = _g()
    import builtins
    real = builtins.__import__

    def old(name, *a, **k):
        if name == "diffusers" and a and a[2] and "WanVACETransformer3DModel" in a[2]:
            raise ImportError("cannot import name WanVACETransformer3DModel")
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", old)
    with pytest.raises(RuntimeError, match="diffusers 0.35"):
        g.check_runtime()


# ─────────────────────────────────────────────────────────────────────────────
#  RUDRA 1.1-1.2 (v3 additions) and AI fill v4 (region re-render), from RUDRA
#  feat/genfill-v4-finish 72c4a62. The composite tests are RUDRA's own
#  (tests/test_genfill_core.py), on the port's names.
# ─────────────────────────────────────────────────────────────────────────────

class _CountingModel(_FakeModel):
    """_FakeModel that records the working size of every context it is given."""

    def context(self, video, mask):
        self.calls.append(("ctx", tuple(video.shape[-2:])))
        return video

    def denoise(self, ctx, s, regions, progress=None):
        if progress:
            progress(1, 1)
        return ctx


def test_prompt_strength_is_checked():
    g = _g()
    assert g.checked_prompt_strength("1.5") == 1.5
    for bad in (-0.1, 2.5, float("nan")):
        with pytest.raises(ValueError):
            g.checked_prompt_strength(bad)
    assert g.FillSettings().prompt_strength == 1.0


def test_focus_boxes_one_per_far_apart_area():
    g = _g()
    union = np.zeros((1080, 3840), bool)
    union[100:140, 200:260] = True          # a lamp at the left
    union[900:950, 3500:3580] = True        # one at the right
    one = g.focus_box(union)
    assert one is None                       # one box round both is most of the frame
    boxes = g.focus_boxes(union)
    assert boxes is not None and len(boxes) == 2
    for y0, y1, x0, x1 in boxes:
        assert (y1 - y0) >= g.FOCUS_MIN_SIDE and (x1 - x0) >= g.FOCUS_MIN_SIDE
    assert boxes[0][3] < boxes[1][2]          # they do not overlap
    assert g.focus_boxes(np.zeros((64, 64), bool)) is None


def test_focus_areas_draws_each_box_at_its_own_size():
    g = _g()
    v = _view()
    c = np.full((256, 1024, 3), 0.5, np.float32)
    c[20:40, 20:40] = 1.0
    c[200:230, 980:1010] = 1.0
    masters = [v.inverse(c)]
    model = _CountingModel()
    outs, alphas, rep = g.fill_frames(masters, model, v, g.FillSettings(steps=1), g.MaskSettings(),
                                      window=9, overlap=2, max_area=256 * 256, focus="areas")
    f = rep["stats"]["focus"]
    assert f["mode"] == "areas" and len(f["boxes"]) == 2
    assert len([c for c in model.calls if c[0] == "ctx"]) == 2          # one model run per box
    assert f["detail_gain"] > 1.0                                        # more pixels per frame pixel
    codes = np.rint(np.clip(v.forward(masters[0]), 0, 1) * 255)
    assert np.array_equal(codes, np.rint(np.clip(v.forward(outs[0]), 0, 1) * 255))
    with pytest.raises(ValueError, match="focus"):
        g.fill_frames(masters, model, v, g.FillSettings(steps=1), g.MaskSettings(), focus="nope")


class _FlickerModel(_FakeModel):
    """Draws the middle frame of three 3 stops brighter than the others: a one-frame flicker."""

    def decode_latents(self, z):
        g = _g()
        lin = g.decode(z.numpy())
        gain = np.ones((1, 1, lin.shape[2], 1, 1), np.float32) * 4.0
        gain[:, :, 1] = 32.0
        return torch.from_numpy(g.encode(lin * gain).astype(np.float32))


def test_fallback_spikes_returns_the_master_for_a_one_frame_flicker():
    g = _g()
    v = _view()
    masters = [v.inverse(np.where(np.arange(64)[None, :, None] < 32, 1.0, 0.5) * np.ones((48, 64, 3), np.float32))
               for _ in range(3)]
    kw = dict(window=9, overlap=2, max_area=48 * 64)
    ms = g.MaskSettings(max_gain_stops=6.0)
    _, keep, rep_keep = g.fill_frames(masters, _FlickerModel(), v, g.FillSettings(steps=1), ms, **kw)
    outs, alphas, rep = g.fill_frames(masters, _FlickerModel(), v, g.FillSettings(steps=1), ms,
                                      fallback="spikes", **kw)
    assert rep_keep["stats"]["fallback_frames"] == 0 and keep[1].any()
    assert rep["stats"]["fallback_frames"] == 1
    assert [f["fell_back"] for f in rep["frames"]] == [None, "flicker", None]
    assert not alphas[1].any() and np.array_equal(outs[1], masters[1])
    assert alphas[0].any() and alphas[2].any()


# ── composite (v4) ──────────────────────────────────────────────────────────

def test_plain_blend_moves_codes_inside_and_nothing_outside():
    g = _g()
    rng = np.random.default_rng(1)
    master = rng.uniform(0.5, 1.2, (30, 40, 3)).astype(np.float32)
    fill = (master * 3.0).astype(np.float32)
    weight = np.zeros((30, 40), np.float32)
    weight[5:20, 10:30] = 1.0
    weight[20, 10:30] = 0.5
    out, alpha = g.plain_blend(master, fill, weight)
    assert (out[weight == 0] == master[weight == 0]).all()
    assert np.allclose(out[5:20, 10:30], fill[5:20, 10:30])
    assert np.allclose(out[20, 10:30], 2.0 * master[20, 10:30])
    assert (alpha == weight).all()


def test_condition_of_bt1886_is_the_display_light_in_ap1():
    g = _g()
    codes = np.zeros((2, 3, 3), np.uint8)
    codes[0, 0] = 255
    codes[0, 1] = 128
    cond = g.condition_of(codes, "bt1886", view=None)
    assert cond.shape == codes.shape and cond.dtype == np.float32
    assert np.allclose(cond[0, 0], 1.0, atol=2e-3)
    assert np.allclose(cond[0, 1], (128 / 255) ** 2.4, atol=2e-3)
    assert (cond[1] == 0).all()


def test_finish_rerender_is_off_by_default_and_sets_hue_level_and_time():
    g = _g()
    rng = np.random.default_rng(3)
    h, w = 96, 128
    codes = np.full((h, w, 3), 120, np.uint8)
    codes[20:80, 30:110] = 254
    codes[20:80, 30:50] = 246
    grey = np.power(codes.astype(np.float32) / 255.0, 2.4)
    source = grey * np.array([1.0, 1.05, 1.05], np.float32)
    master = grey * 0.9
    fill = (rng.uniform(1.0, 2.0, (h, w, 1)) * np.array([0.6, 1.0, 2.0])).astype(np.float32)
    weight = np.zeros((h, w), np.float32)
    weight[20:80, 30:110] = 1.0
    inside = weight > 0
    lum = lambda x: x @ g.LUMA_AP1  # noqa: E731
    assert g.finish_rerender(fill, master, source, codes, weight) is fill
    white = g.white_scale(master, codes)
    assert abs(white - 0.9) < 1e-3
    out = g.finish_rerender(fill, master, source, codes, weight, hue="source", level=3.0, white=white)
    assert np.array_equal(out[~inside], fill[~inside])
    assert abs(float(np.median(lum(out[30:70, 70:100]))) / white - 3.0) < 0.15
    assert float(np.median(lum(out[30:70, 34:42]))) < 0.8 * float(np.median(lum(out[30:70, 70:100])))
    bright = g.finish_rerender(fill * 64.0, master, source, codes, weight, level=3.0, white=white)
    top = 3.0 * 2.0 ** g.RERENDER_CEILING_STOPS
    assert abs(float(np.median(lum(bright[30:70, 70:100]))) / white - top) < 0.05 * top
    edge = g.finish_rerender(fill, master, source, np.full_like(codes, 240), weight, hue="source", level=3.0,
                             white=white)
    assert np.allclose(edge[inside], master[inside], rtol=1e-3, atol=1e-5)
    core = out[30:70, 70:100].reshape(-1, 3).mean(0)
    assert abs(core[1] / core[0] - 1.05) < 0.01 and abs(core[2] / core[0] - 1.05) < 0.01
    state = {}
    a = g.finish_rerender(fill, master, source, codes, weight, smooth=0.8, state=state)
    assert np.allclose(a[inside], fill[inside], rtol=1e-4)
    b = g.finish_rerender(fill * 2.0, master, source, codes, weight, smooth=0.8, state=state)
    assert np.allclose(np.log2(lum(b[inside]) / lum(a[inside])), 0.2, atol=1e-3)
    dark = fill.copy()
    dark[50:54, 80:84] *= 0.25                     # a two-stop dark smudge in the model's detail
    plain = g.finish_rerender(dark, master, source, codes, weight, level=3.0, white=white)
    dipped = g.finish_rerender(dark, master, source, codes, weight, level=3.0, white=white, dip=0.3)
    assert lum(dipped[51:53, 81:83]).mean() > lum(plain[51:53, 81:83]).mean()   # the dip limit lifts it
    with pytest.raises(ValueError):
        g.finish_rerender(fill, master, source, codes, weight, hue="other")


def test_neutral_weight_leaves_a_saturated_highlight_to_the_master():
    g = _g()
    codes = np.array([[[254, 254, 254], [245, 254, 254], [255, 190, 120], [255, 210, 200], [0, 0, 0]]], np.uint8)
    w = g.neutral_weight(codes)[0]
    assert w[0] == 1.0 and w[1] == 1.0
    assert w[2] == 0.0
    assert 0.0 < w[3] < 1.0
    assert w[4] == 0.0


def test_rudra_app_v4_settings_are_the_apps():
    g = _g()
    assert g.RUDRA_APP_V4 == dict(hue="source", colour=0.3, level=3.3, smooth=0.8, dip=0.3)
    assert g.RUDRA_APP_V4_MASK == dict(tau_high=240, tau_low=-1, feather=8)
    g.Rerender(**g.RUDRA_APP_V4).check()
    with pytest.raises(ValueError):
        g.Rerender(condition="hlg").check()


# ── region re-render through fill_frames and its node ───────────────────────

def _door(n=2, h=64, w=96):
    """A source with a near-white door (code 250) and a sunset patch (red clipped, blue low), and a
    BT.1886 master made from it at SDR white 0.81."""
    g = _g()
    codes = []
    for _ in range(n):
        c = np.full((h, w, 3), 110, np.uint8)
        c[10:50, 10:40] = 250
        c[10:50, 60:90] = (255, 190, 120)
        codes.append(c)
    masters = [0.81 * g.condition_of(c, "bt1886", None) for c in codes]
    return codes, masters


def test_fill_frames_rerender_changes_only_the_near_white_region():
    g = _g()
    codes, masters = _door()
    ms = g.MaskSettings(max_gain_stops=0.0, **g.RUDRA_APP_V4_MASK)
    outs, alphas, rep = g.fill_frames(masters, _FakeModel(), None, g.FillSettings(steps=1), ms, window=9,
                                      overlap=2, max_area=64 * 96, rerender=g.Rerender(**g.RUDRA_APP_V4),
                                      source_codes=codes)
    assert rep["rerender"]["level"] == 3.3 and "NOT kept" in rep["contract"]
    assert abs(rep["stats"]["rerender_white"] - 0.81) < 0.02
    for m, o, a in zip(masters, outs, alphas):
        assert np.array_equal(o[a == 0], m[a == 0])           # outside: the master, bit for bit
        assert a[20:40, 18:32].min() > 0.99                    # the door is re-rendered
        assert not a[20:40, 68:82].any()                       # the sunset is left to the master
        y = lambda x: x @ g.LUMA_AP1  # noqa: E731
        assert np.median(y(o[20:40, 18:32])) > 1.5 * np.median(y(m[20:40, 18:32]))
    with pytest.raises(ValueError, match="source"):
        g.fill_frames(masters, _FakeModel(), None, g.FillSettings(steps=1), ms, rerender=g.Rerender())


def test_v4_lora_is_found_in_models_radiance(monkeypatch, tmp_path):
    g = _g()
    paths = importlib.import_module("radiance.model.paths")
    monkeypatch.setattr(paths, "radiance_model_dirs", lambda: [tmp_path])
    monkeypatch.delenv(g.V4_ENV, raising=False)
    assert g.resolve_v4_lora() is None
    f = tmp_path / g.V4_DIRNAME / "rudra_genfill.safetensors"
    f.parent.mkdir()
    f.write_bytes(b"x")
    assert g.resolve_v4_lora() == f
    assert g.resolve_v4_lora(str(f.parent)) == f
    monkeypatch.setenv(g.V4_ENV, str(f))
    assert g.resolve_v4_lora() == f


def _rerender_node():
    return importlib.import_module("radiance.nodes.hdr.rudra_rerender").RadianceSDRToHDRAIFillRerender()


def _door_tensors(n=2):
    a = importlib.import_module("radiance.model.rudra_aces2")
    codes, masters = _door(n)
    img = torch.from_numpy(np.stack([a.acescg_to(m, a.M_AP1_TO_REC709) for m in masters]))
    src = torch.from_numpy(np.stack(codes).astype(np.float32) / 255.0)
    return img, src


def test_rerender_node_says_why_the_lora_is_not_downloaded(monkeypatch):
    g = _g()
    monkeypatch.setattr(g, "resolve_v4_lora", lambda *a: None)
    img, src = _door_tensors(1)
    with pytest.raises(FileNotFoundError, match="not downloaded.*v4.3"):
        _rerender_node().rerender(img, "Linear (Rec.709)", src, "")


def test_rerender_node_with_nothing_bright_returns_the_input():
    img = torch.full((1, 16, 16, 3), 0.1)
    out, gen, recon, report = _rerender_node().rerender(img, "Linear (Rec.709)", torch.full((1, 16, 16, 3), 0.3), "")
    assert out is img and float(gen.max()) == 0 and "nothing to re-render" in report


def test_rerender_node_runs_with_the_app_finish(monkeypatch, tmp_path):
    g = _g()
    lora = tmp_path / "rudra_genfill.safetensors"
    lora.write_bytes(b"x")
    monkeypatch.setattr(g, "resolve_v4_lora", lambda *a: lora)
    monkeypatch.setattr(g, "resolve_package", lambda *a: tmp_path)
    monkeypatch.setattr(g, "resolve_base", lambda *a: str(tmp_path / "base"))
    monkeypatch.setattr(g, "check_runtime", lambda: None)
    model = _FakeModel()
    model.identity = {"lora_sha256": "0" * 64}
    got = {}

    def fake_get_model(*a, **k):
        got.update(k)
        return model
    monkeypatch.setattr(g, "get_model", fake_get_model)
    img, src = _door_tensors(2)
    out, gen, recon, report = _rerender_node().rerender(img, "Linear (Rec.709)", src, "an open door", steps=1)
    assert got["lora"] == lora
    assert out.shape == img.shape and float(gen.max()) > 0
    keep = gen == 0
    assert torch.equal(out[keep], img[keep])
    assert "EXPERIMENTAL" in report and "NOT the gated v4 LoRA" in report and '"level": 3.3' in report
    assert model.calls and model.calls[0][1] == "an open door"


def test_v3_node_passes_focus_fallback_and_strength(node_env, monkeypatch):
    g = _g()
    mod = importlib.import_module("radiance.nodes.hdr.rudra_aifill")
    seen = {}
    real = g.fill_frames

    def spy(*a, **k):
        seen.update(k)
        seen["s"] = a[3]
        return real(*a, **k)
    monkeypatch.setattr(g, "fill_frames", spy)
    out, gen, recon, report = mod.RadianceSDRToHDRAIFill().fill(
        _linear_master(), "Linear (Rec.709)", "", 0.0, use_brightness_estimator=False, steps=1, window=9,
        overlap=2, max_area=48 * 64, prompt_strength=1.5, focus="region", fallback="spikes")
    assert seen["focus"] == "region" and seen["fallback"] == "spikes" and seen["s"].prompt_strength == 1.5
    assert "fallback spikes" in report and "focus:" in report


def test_v3_node_new_inputs_come_last():
    """Saved workflows store widget values by position: the RUDRA 1.1 inputs are appended."""
    mod = importlib.import_module("radiance.nodes.hdr.rudra_aifill")
    names = list(mod.RadianceSDRToHDRAIFill.INPUT_TYPES()["optional"])
    assert names[-3:] == ["prompt_strength", "focus", "fallback"]
    assert names.index("keep_loaded") == len(names) - 4
