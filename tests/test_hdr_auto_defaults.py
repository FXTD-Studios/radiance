"""End-user defaults: colour, OCIO, RUDRA and HDR are detected automatically.

Radiance ships to people who drop a node and queue. These pin the defaults a
new node gets (saved workflows keep their own widget values), so that HDR
input is recognised without a setting and the HDR path is on by default:

* Read, Viewer: colour space Auto.
* SDR -> HDR Universal / Recover: learned backend Auto (RUDRA ACES 2).
* RUDRA ACES 2 may download on first use; nothing has to be set.
* Write: a 16-bit half EXR, scene-linear, so an HDR master is not clipped.
* Upscale 32-bit, Upscale By Size, Downscale 32-bit: input colour space Auto,
  so a linear HDR plate is not decoded as sRGB.
* AI Upscale: mode Auto, Refine (HDR) when values exceed 1.0.
"""
import pytest
import torch

from radiance.image import upscale as U


def _default(cls, name):
    spec = cls.INPUT_TYPES()
    for section in ("required", "optional"):
        if name in (spec.get(section) or {}):
            entry = spec[section][name]
            opts = entry[1] if len(entry) > 1 and isinstance(entry[1], dict) else {}
            if "default" in opts:
                return opts["default"]
            # ComfyUI shows the first entry of a combo without a default.
            return entry[0][0] if isinstance(entry[0], (list, tuple)) else None
    raise KeyError(name)


@pytest.mark.parametrize("cls", [U.RadianceProUpscale, U.RadianceUpscaleBySize, U.RadianceDownscale32bit])
def test_resize_nodes_detect_the_input_space(cls):
    assert _default(cls, "input_color_space") == "Auto"


def test_ai_upscale_detects_hdr():
    assert _default(U.RadianceAIUpscale, "mode") == "Auto"


@pytest.mark.parametrize("peak, expected", [(0.9, "Standard"), (1.0, "Standard"), (4.93, "Refine (HDR)")])
def test_ai_upscale_auto_picks_the_mode_from_the_values(peak, expected):
    img = torch.zeros(1, 4, 4, 3)
    img[0, 1, 1, 0] = peak
    assert U.resolve_ai_upscale_mode(img, "Auto") == expected


def test_ai_upscale_auto_ignores_alpha_and_keeps_a_named_mode():
    img = torch.zeros(1, 4, 4, 4)
    img[..., 3] = 2.0                               # alpha is not colour
    assert U.resolve_ai_upscale_mode(img, "Auto") == "Standard"
    assert U.resolve_ai_upscale_mode(img, "Normalize (HDR)") == "Normalize (HDR)"


class _Nearest4x(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.interpolate(x, scale_factor=4, mode="nearest")


def test_ai_upscale_auto_keeps_hdr_highlights(monkeypatch):
    """Auto on an HDR frame runs Refine (HDR): log1p in, expm1 out, so the
    highlight comes back above 1.0 and the info says which mode ran."""
    node = U.RadianceAIUpscale()
    monkeypatch.setattr(node, "_load_model", lambda *a, **k: (_Nearest4x(), "ok"))
    img = torch.full((1, 8, 8, 3), 0.18)
    img[0, 2:4, 2:4, :] = 4.93
    out, info = node.upscale(img, model_name="RealESRGAN_x4plus", mode="Auto", tile_size=512)
    assert out.shape == (1, 32, 32, 3)
    assert float(out.max()) == pytest.approx(4.93, rel=1e-4)
    assert "[Refine (HDR) (Auto)]" in info


def test_the_hdr_pipeline_is_on_by_default():
    from radiance.nodes.io.write import RadianceRead, RadianceWrite
    from radiance.nodes.monitor.viewer import RadianceViewer
    from radiance.nodes.hdr.uplift_universal import RadianceSDRToHDRUniversal, RadianceSDRToHDRRecover
    from radiance.nodes.generate.engine import RadianceHDRVAEDecode

    assert _default(RadianceRead, "color_space").startswith("Auto")
    assert _default(RadianceViewer, "input_space") == "Auto"
    for cls in (RadianceSDRToHDRUniversal, RadianceSDRToHDRRecover):
        assert _default(cls, "learned_backend") == "Auto"
        assert _default(cls, "output_encoding") == "Linear"
    assert _default(RadianceSDRToHDRUniversal, "processing_mode") == "Hybrid"
    assert _default(RadianceSDRToHDRUniversal, "rudra_blend") > 0.0
    # Auto: exact log inversion for an HDR Encode latent, sampler-safe SDR
    # otherwise (SDR -> HDR Universal makes the HDR after it).
    assert _default(RadianceHDRVAEDecode, "decode_mode") == "Auto (Recommended)"
    assert "EXR" in _default(RadianceWrite, "format")
    assert _default(RadianceWrite, "color_space") == "Linear (pass-through)"


def test_rudra_aces2_downloads_by_default(monkeypatch):
    from radiance.core.consent import downloads_allowed
    from radiance.model.rudra_download import ACES2
    for env in ("RADIANCE_ALLOW_DOWNLOADS", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.delenv(env, raising=False)
    assert ACES2.default_allowed is True
    assert downloads_allowed(default=ACES2.default_allowed) is True
