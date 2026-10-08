"""RGBA (4-channel) input on the preview and upscale paths (B23).

FlipbookGIF and PreviewServer crashed on RGBA: Pillow cannot write RGBA as
a GIF frame built with mode="RGB" or as a JPEG. They are preview outputs that
cannot show alpha, so they now drop it on purpose; the passthrough keeps it.

Tier 2 (spandrel / basicsr) upscaling fed only RGB to the model and returned
three channels into a four-channel accumulator, so RGBA crashed. Alpha is now
resized to the colour output's size and carried through.

Face Restore pasted a three-channel restored crop into an RGBA frame by
repeating channels, which wrote the red channel into alpha. The original alpha
is now kept.
"""
import io

import pytest

pytestmark = pytest.mark.real_torch


def _rgba(b=2, h=24, w=32, seed=0):
    import torch
    g = torch.Generator().manual_seed(seed)
    img = torch.rand(b, h, w, 4, generator=g)
    # A recognisable alpha: a left-to-right ramp, never equal to any colour.
    img[..., 3] = torch.linspace(0.05, 0.95, w).view(1, 1, w).expand(b, h, w)
    return img


# ── Preview outputs: alpha is dropped deliberately ───────────────────────────

class TestFlipbookGIF:

    def test_rgba_writes_a_gif(self, tmp_path):
        from PIL import Image
        from radiance.nodes.monitor.realtime import RadianceFlipbookGIF
        imgs = _rgba()
        path = str(tmp_path / "flip.gif")
        out, status = RadianceFlipbookGIF().export_gif(imgs, path, fps=12.0, max_width=16)
        assert out is imgs, "the passthrough keeps the alpha"
        assert status.startswith("GIF written"), status
        with Image.open(path) as gif:
            assert gif.n_frames == 2 and gif.size == (16, 12)

    def test_rgba_matches_its_rgb(self, tmp_path):
        from PIL import Image
        from radiance.nodes.monitor.realtime import RadianceFlipbookGIF
        imgs = _rgba(b=1)
        a, b = str(tmp_path / "a.gif"), str(tmp_path / "b.gif")
        RadianceFlipbookGIF().export_gif(imgs, a, fps=12.0, max_width=64, dither=False)
        RadianceFlipbookGIF().export_gif(imgs[..., :3].contiguous(), b, fps=12.0, max_width=64, dither=False)
        with Image.open(a) as ga, Image.open(b) as gb:
            assert ga.convert("RGB").tobytes() == gb.convert("RGB").tobytes()


class TestPreviewServer:

    def _serve(self, monkeypatch, imgs, **kw):
        from radiance.nodes.monitor import realtime as rt
        monkeypatch.setattr(rt, "_ensure_server", lambda port, name: None)
        monkeypatch.setattr(rt, "_shutdown_servers", lambda keep_port=None: 0)
        out, url = rt.RadiancePreviewServer().serve(imgs, port=18999, stream_name="rgba_test", **kw)
        return out, url, rt._PREVIEW_BUFFER.pop("rgba_test", b"")

    def test_rgba_is_served_as_jpeg(self, monkeypatch):
        from PIL import Image
        imgs = _rgba()
        out, url, jpeg = self._serve(monkeypatch, imgs)
        assert out is imgs and url.startswith("http://localhost:")
        with Image.open(io.BytesIO(jpeg)) as im:
            assert im.format == "JPEG" and im.mode == "RGB" and im.size == (32, 24)

    def test_rgba_with_resize(self, monkeypatch):
        from PIL import Image
        _, _, jpeg = self._serve(monkeypatch, _rgba(), resize_width=16)
        with Image.open(io.BytesIO(jpeg)) as im:
            assert im.size == (16, 12)

    def test_frame_to_jpeg_accepts_rgba(self):
        import numpy as np
        from radiance.nodes.monitor.realtime import _frame_to_jpeg
        assert _frame_to_jpeg(np.random.rand(8, 8, 4).astype("float32"))[:2] == b"\xff\xd8"


# ── Tier 2 upscale: alpha is resized and carried through ─────────────────────

class _NearestSR:
    """A stand-in 2x SR network that accepts exactly three channels."""

    def __init__(self, scale=2):
        import torch.nn as nn
        import torch.nn.functional as F

        class _Net(nn.Module):
            def forward(self, x):
                assert x.shape[1] == 3, f"the model was fed {x.shape[1]} channels"
                return F.interpolate(x, scale_factor=scale, mode="nearest")

        self.net = _Net()


class _Descriptor:
    """A spandrel ModelDescriptor look-alike (not an nn.Module)."""

    def __init__(self, net):
        self.model = net


def _expected_alpha(img, scale):
    import torch.nn.functional as F
    return F.interpolate(img[..., 3:4].permute(0, 3, 1, 2), scale_factor=scale,
                         mode="bilinear", align_corners=False).clamp(0, 1).permute(0, 2, 3, 1)


class TestTier2Alpha:

    @pytest.mark.parametrize("wrap", ["spandrel", "basicsr"])
    def test_tiled_tier2_upscale_keeps_alpha(self, monkeypatch, wrap):
        import torch
        from radiance.nodes.upscale import upscale as up
        net = _NearestSR().net
        model = _Descriptor(net) if wrap == "spandrel" else net
        monkeypatch.setattr(up, "_load_tier2", lambda key, scale, device: model)
        fn, label = up._build_upscale_fn("tier2_quality (SwinIR-L — transformer quality)", 2,
                                         torch.device("cpu"))
        assert "Tier 2" in label
        img = _rgba(b=1, h=20, w=20)
        out, _conf = up.tiled_upscale(img, fn, scale=2, tile_size=20, overlap=0)
        assert out.shape == (1, 40, 40, 4)
        assert torch.allclose(out[..., 3:], _expected_alpha(img, 2), atol=1e-5)
        rgb = torch.nn.functional.interpolate(img[..., :3].permute(0, 3, 1, 2), scale_factor=2,
                                              mode="nearest").permute(0, 2, 3, 1)
        assert torch.allclose(out[..., :3], rgb, atol=1e-5)

    def test_spandrel_infer_matches_the_colour_size(self):
        import torch
        from radiance.nodes.upscale import upscale as up
        tile = _rgba(b=1, h=8, w=12)
        y = up._spandrel_infer(_Descriptor(_NearestSR(scale=4).net), tile, torch.device("cpu"))
        assert y.shape == (1, 32, 48, 4)
        assert torch.allclose(y[..., 3:], _expected_alpha(tile, 4), atol=1e-5)

    def test_rgb_is_unchanged(self):
        import torch
        from radiance.nodes.upscale import upscale as up
        tile = _rgba(b=1, h=8, w=8)[..., :3].contiguous()
        y = up._spandrel_infer(_Descriptor(_NearestSR().net), tile, torch.device("cpu"))
        assert y.shape == (1, 16, 16, 3)


# ── Face Restore: the original alpha is kept ─────────────────────────────────

class TestFaceRestoreAlpha:

    def test_composite_keeps_alpha(self):
        import torch
        from radiance.nodes.upscale import upscale as up
        img = _rgba(b=1, h=32, w=32)[0]
        face = torch.full((16, 16, 3), 0.5)
        out = up._composite_face(img, face, (8, 8, 24, 24), blend_radius=4)
        assert torch.equal(out[..., 3], img[..., 3])
        assert not torch.equal(out[8:24, 8:24, :3], img[8:24, 8:24, :3]), "the face was composited"

    def test_composite_rgb_is_unchanged(self):
        import torch
        from radiance.nodes.upscale import upscale as up
        img = _rgba(b=1, h=32, w=32)[0, ..., :3].contiguous()
        face = torch.full((16, 16, 3), 0.5)
        out = up._composite_face(img, face, (8, 8, 24, 24), blend_radius=0)
        assert out.shape == img.shape
        assert torch.allclose(out[16, 16], torch.full((3,), 0.5))

    def test_restore_faces_keeps_alpha(self, monkeypatch):
        import torch
        from radiance.nodes.upscale import upscale as up
        monkeypatch.setattr(up, "_detect_faces", lambda frame, **k: [(4, 4, 28, 28)])
        monkeypatch.setattr(up, "_load_face_restore_model", lambda key, device: object())
        monkeypatch.setattr(up, "_restore_face_crop",
                            lambda m, crop, key, fidelity_weight, device: crop[..., :3] * 0.5)
        imgs = _rgba(b=2, h=32, w=32)
        out, mask, info = up.RadianceUpscaleFaceRestore().restore_faces(
            imgs, face_model="codeformer", colour_correct=False)
        assert out.shape == imgs.shape
        assert torch.equal(out[..., 3], imgs[..., 3])
        assert "Faces restored    : 2" in info


# ── Focus Peaking, Split View, Contact Sheet, Video Assembler (TEN-007) ─────
# KNOWN_ISSUES listed these as raising on 4-channel input until 4.0.

def _rgba4(b=2, h=16, w=24):
    import torch
    return torch.rand(b, h, w, 4, generator=torch.Generator().manual_seed(0))


def test_focus_peaking_takes_rgba_and_keeps_its_alpha():
    import torch
    from radiance.nodes.monitor.realtime import RadianceFocusPeaking
    x = _rgba4()
    passthrough, peak = RadianceFocusPeaking().peak(x, 0.2, "Red", 0.85)
    assert passthrough is x
    assert peak.shape == x.shape
    assert torch.equal(peak[..., 3], x[..., 3])


@pytest.mark.parametrize("b_channels", [3, 4])
@pytest.mark.parametrize("mode", ["wipe_h", "wipe_v", "side_by_side", "diff"])
def test_split_view_compares_the_colour_of_rgba_input(mode, b_channels):
    from radiance.nodes.monitor.realtime import RadianceSplitView
    a = _rgba4()
    b = _rgba4()[..., :b_channels]
    (out,) = RadianceSplitView().compare(a, b, mode, 0.5)
    assert out.shape == (2, 16, 24, 3)


def test_contact_sheet_takes_rgba4():
    from radiance.nodes.monitor.realtime import RadianceContactSheet
    sheet, cols, rows = RadianceContactSheet().sheet(_rgba4(b=3), 32, 2)
    assert sheet.shape[-1] == 3 and (cols, rows) == (2, 2)


def test_video_assembler_joins_rgb_and_rgba_frames():
    import torch
    from radiance.nodes.video.hdr import RadianceVideoAssembler
    node = RadianceVideoAssembler()
    rgb = torch.rand(1, 16, 24, 3)
    rgba = _rgba4(b=1)
    node.assemble(rgb, "ten007", 2, reset=True)
    video, n, complete = node.assemble(rgba, "ten007", 2)
    assert complete and n == 2 and video.shape == (2, 16, 24, 4)
    assert torch.equal(video[0, ..., :3], rgb[0]) and torch.all(video[0, ..., 3] == 1)
    assert torch.equal(video[1], rgba[0])


def test_an_external_upscale_model_sees_rgb_and_alpha_is_carried(monkeypatch):
    # A ComfyUI UPSCALE_MODEL is an RGB network. Every channel was passed to
    # comfy.utils.tiled_scale, so an RGBA plate reached the model with 4.
    import torch
    import torch.nn.functional as F
    import comfy.utils
    from radiance.nodes.upscale import upscale as up

    seen = []

    def fake_tiled_scale(x, model, tile_x, tile_y, overlap, upscale_amount, pbar=None):
        seen.append(x.shape[1])
        return F.interpolate(x, scale_factor=upscale_amount, mode="nearest")

    monkeypatch.setattr(comfy.utils, "tiled_scale", fake_tiled_scale, raising=False)
    fn, _ = up._build_upscale_fn("auto", 2, torch.device("cpu"), upscale_model=object())
    tile = _rgba4(b=1, h=8, w=12)
    out = fn(tile)
    assert seen == [3]
    assert out.shape == (1, 16, 24, 4)
    assert torch.equal(out[..., :3], F.interpolate(tile[..., :3].permute(0, 3, 1, 2), scale_factor=2,
                                                   mode="nearest").permute(0, 2, 3, 1))


# ── review fixes ────────────────────────────────────────────────────────────

def test_video_assembler_turns_grey_frames_into_grey_rgb_not_cyan():
    import torch
    from radiance.nodes.video.hdr import RadianceVideoAssembler
    node = RadianceVideoAssembler()
    grey = torch.full((1, 4, 4, 1), 0.25)
    node.assemble(torch.rand(1, 4, 4, 3), "review-grey", 2, reset=True)
    video, _, _ = node.assemble(grey, "review-grey", 2)
    assert torch.allclose(video[1], torch.full((4, 4, 3), 0.25))


def test_video_assembler_refuses_channel_counts_it_cannot_reconcile():
    import torch
    from radiance.nodes.video.hdr import RadianceVideoAssembler
    node = RadianceVideoAssembler()
    node.assemble(torch.rand(1, 4, 4, 3), "review-odd", 2, reset=True)
    with pytest.raises(ValueError, match="channel"):
        node.assemble(torch.rand(1, 4, 4, 2), "review-odd", 2)
    node.assemble(torch.rand(1, 4, 4, 3), "review-odd", 1, reset=True)


def test_face_composite_keeps_the_alpha_of_grey_plus_alpha():
    import torch
    from radiance.nodes.upscale import upscale as up
    img = torch.rand(32, 32, 2)
    face = torch.rand(16, 16, 3)
    out = up._composite_face(img, face, (8, 8, 24, 24), blend_radius=0)
    assert torch.equal(out[..., 1], img[..., 1])


@pytest.mark.parametrize("native", [True, False])
def test_seedvr2_sees_rgb_and_carries_alpha(native):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from radiance.nodes.upscale import upscale as up

    seen = []

    class Native:
        device = torch.device("cpu")

        def upscale_batch(self, x, prompt, num_inference_steps):
            seen.append(x.shape[-1])
            return F.interpolate(x.permute(0, 3, 1, 2), scale_factor=4).permute(0, 2, 3, 1)

    class Generic:
        def __call__(self, prompt, image, num_inference_steps):
            seen.append(len(image.getbands()))
            w, h = image.size
            return type("R", (), {"images": [image.resize((w * 4, h * 4), Image.NEAREST)]})()

    tile = _rgba4(b=1, h=8, w=8)
    out = up._seedvr2_infer(Native() if native else Generic(), tile)
    assert seen == [3]
    assert out.shape == (1, 32, 32, 4)
