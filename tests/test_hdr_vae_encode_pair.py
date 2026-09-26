"""VAE Encode (HDR) is registered and pairs with VAE Decode (HDR).

Found by a real-VAE check on a fresh install: the two HDR latent encoders on
the menu fed decoders retired in 3.5.0, so their latents came back clipped,
and the encoder HDR VAE Decode's log inversion was written for
(RadianceVAE4KEncode) was never registered.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.real_torch


def test_encoder_is_registered_next_to_the_decoder():
    from radiance.nodes.generate import NODE_CLASS_MAPPINGS as gen
    from radiance.nodes.branding import NODE_SECTIONS
    assert NODE_SECTIONS["RadianceHDRVAEEncode"] == NODE_SECTIONS["RadianceHDRVAEDecode"]
    assert "RadianceHDRVAEEncode" in gen and "RadianceHDRVAEDecode" in gen


def test_default_hdr_coding_is_the_invertible_one():
    from radiance.nodes.generate.engine import RadianceHDRVAEEncode
    choices, cfg = RadianceHDRVAEEncode.INPUT_TYPES()["optional"]["hdr_mode"]
    assert cfg["default"] == "Compress (Log)" and "Compress (Log)" in choices
    import inspect
    sig = inspect.signature(RadianceHDRVAEEncode.encode)
    assert sig.parameters["hdr_mode"].default == "Compress (Log)"   # API prompts omit it


def test_mean_sampling_accepts_a_plain_tensor_posterior():
    """torch.Tensor has a .mean METHOD; it used to be returned as the latent."""
    from radiance.hdr import vae as vmod

    class FSM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))

        def encode(self, x):                      # ComfyUI returns a tensor
            return x[:, :3, ::8, ::8] + self.w

    class FakeVAE:
        first_stage_model = FSM()

        def encode(self, pixels):
            raise AssertionError("fallback should not be needed")

    out = vmod._encode_with_sampling_mode(FakeVAE(), torch.rand(1, 16, 16, 3), "mean")
    assert isinstance(out, torch.Tensor) and out.shape == (1, 3, 2, 2)


def test_legacy_encoders_are_hidden_and_stop_with_the_replacement():
    """They stay registered so saved graphs open, but running one would render
    clipped with no warning, so it stops and names VAE Encode (HDR)."""
    from radiance.nodes.hdr.encoder import RadianceHDRLatentEncoder, RadianceHDRTurboEncoder
    for cls in (RadianceHDRLatentEncoder, RadianceHDRTurboEncoder):
        assert cls.DEPRECATED is True
        assert cls.DESCRIPTION.startswith("Legacy.")
        assert "VAE Encode (HDR)" in cls.DESCRIPTION
        with pytest.raises(RuntimeError, match="VAE Encode \\(HDR\\)"):
            getattr(cls(), cls.FUNCTION)(image=torch.zeros(1, 8, 8, 3), vae=object())


def test_legacy_aces_output_transform_is_hidden_but_still_works():
    from radiance.hdr.color import ACES2OutputTransform
    assert ACES2OutputTransform.DEPRECATED is True
    import numpy as np
    out = ACES2OutputTransform()._apply_tonescale_drt(
        np.full((1, 1, 1, 3), 0.18, dtype=np.float32), peak_luminance=100.0, is_hdr=False)
    assert 0.05 < float(out[0, 0, 0, 0]) < 0.2        # ACES 2.0 SDR grey ~10 nits


class _PixelVAE:
    """A VAE whose latent is the image itself, at 1/8 size: enough to drive
    the HDR encode/decode wrappers without model weights."""
    downscale_ratio = 8
    latent_channels = 4

    def encode(self, pixels):
        x = pixels[..., :3].movedim(-1, 1)
        x = torch.nn.functional.avg_pool2d(x, 8)
        return torch.cat([x, torch.zeros_like(x[:, :1])], dim=1)

    def decode(self, latent):
        x = torch.nn.functional.interpolate(latent[:, :3], scale_factor=8, mode="nearest")
        return x.movedim(1, -1)


def test_paired_decode_does_not_warn_that_correct_output_is_wrong(caplog):
    """The pair carries a linear source in LogC4 and says so in radiance_meta.
    The decoder used to warn 'output colors will be WRONG' on every run of it."""
    import logging
    from radiance.nodes.generate.engine import RadianceHDRVAEEncode, RadianceHDRVAEDecode
    img = torch.rand(1, 64, 64, 3) * 4.0          # scene-linear, above 1.0
    enc = RadianceHDRVAEEncode()
    latent = getattr(enc, enc.FUNCTION)(pixels=img, vae=_PixelVAE(), source_space="Linear")[0]
    assert latent.get("radiance_meta", {}).get("hdr_mode") == "Compress (Log)"
    dec = RadianceHDRVAEDecode()
    with caplog.at_level(logging.WARNING):
        res = getattr(dec, dec.FUNCTION)(samples=latent, vae=_PixelVAE(), target_space="Linear")
    out = (res["result"] if isinstance(res, dict) else res)[0]
    assert not [r for r in caplog.records if "LogC4" in r.getMessage() and r.levelno >= logging.WARNING]
    assert float(out.max()) > 1.5, "the pair should return HDR, not a clipped image"
