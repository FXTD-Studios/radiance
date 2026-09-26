"""VideoHDRConditioner reaches the model; VideoHDRDecode honours the gamut.

The conditioner used to append words to a "text" key that ComfyUI
conditioning never carries, so its output equalled its input. The decoder
printed the metadata gamut and never converted primaries, so a BT.2020 PQ
signal carried Rec.709 primaries.
"""
import json

import numpy as np
import torch

from radiance.nodes.video.hdr import RadianceVideoHDRConditioner, RadianceVideoHDRDecode


class _Clip:
    def __init__(self, width=8):
        self.width = width
        self.seen = []

    def tokenize(self, text):
        self.seen.append(text)
        return text

    def encode_from_tokens(self, tokens, return_pooled=False):
        return torch.ones(1, 3, self.width)


def _cond(width=8):
    return [[torch.zeros(1, 5, width), {"pooled_output": torch.zeros(1, width)}]]


def test_with_clip_the_descriptors_are_concatenated():
    clip = _Clip()
    out, meta = RadianceVideoHDRConditioner().condition(_cond(), "1000", "BT.2020", "PQ (ST.2084)",
                                                       clip=clip, token_strength=0.5)
    t = out[0][0]
    assert t.shape == (1, 8, 8)
    assert float(t[:, 5:].max()) == 0.5
    assert json.loads(meta)["applied_to_model"] is True
    assert "1000" in clip.seen[0]


def test_without_clip_it_says_nothing_reached_the_model():
    cond = _cond()
    out, meta = RadianceVideoHDRConditioner().condition(cond, "1000", "BT.2020", "PQ (ST.2084)")
    assert out[0][0] is cond[0][0]
    m = json.loads(meta)
    assert m["applied_to_model"] is False and "clip not connected" in m["note"]


def test_width_mismatch_is_reported_not_forced():
    _, meta = RadianceVideoHDRConditioner().condition(_cond(8), "1000", "BT.2020", "PQ (ST.2084)",
                                                     clip=_Clip(width=4))
    assert json.loads(meta)["applied_to_model"] is False


def _decode(gamut):
    img = torch.zeros(1, 2, 2, 3)
    img[..., 0] = 1.0  # pure Rec.709 red
    meta = json.dumps({"peak_nits": 1000, "gamut": gamut})
    return RadianceVideoHDRDecode().decode(img, meta, "Pass-through", output_eotf="Linear",
                                           gamut_clip=False)


def test_bt2020_target_converts_primaries():
    hdr, _, report = _decode("BT.2020")
    px = hdr[0, 0, 0]
    # Rec.709 red in BT.2020 has green and blue components.
    assert float(px[1]) > 0 and float(px[2]) > 0
    assert "Rec.709 -> BT.2020" in report


def test_bt709_target_is_untouched():
    hdr, _, report = _decode("BT.709")
    px = hdr[0, 0, 0]
    assert float(px[1]) == 0 and float(px[2]) == 0
    assert "no conversion" in report


def test_hdr_diagnostics_decodes_display_encoded_input():
    from radiance.nodes.hdr.smart import RadianceHDRDiagnostics
    img = torch.full((1, 8, 8, 3), 0.5)
    lin = RadianceHDRDiagnostics().diagnose(img, colorspace="Linear (sRGB)")
    srgb = RadianceHDRDiagnostics().diagnose(img, colorspace="sRGB")
    assert abs(lin[3] - 0.5 * 203.0) < 1e-3
    assert abs(srgb[3] - 0.21404 * 203.0) < 0.1
    assert '"colorspace": "sRGB"' in srgb[0]
