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
