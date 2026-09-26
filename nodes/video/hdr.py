# ============================================================
# FXTD STUDIOS — Radiance
# nodes_video_hdr.py  —  HDR Video Generation Pipeline
# ============================================================
# Connects any DiT video model running in ComfyUI to Radiance's
# HDR color science stack.
#
# Nodes
# -----
#   RadianceVideoHDRConditioner
#     Inject HDR display metadata (peak nits, gamut, EOTF, mastering
#     display) directly into the text conditioning token stream so the
#     video model generates HDR-aware content from the first denoising step.
#
#   RadianceVideoHDRDecode
#     Post-process DiT video output latents through the full Radiance
#     HDR pipeline: tone-map → ACES / OCIO colour transform → PQ/HLG
#     encode → policy guard.  Accepts frame batches or 5-D latents.
#
#   RadianceVideoPromptBuilder
#     Structured prompt templating with HDR-specific lighting / camera /
#     mood descriptors optimised for LTX-Video and HunyuanVideo.
#
#   RadianceVideoFrameRouter
#     Route individual frames from a decoded video tensor to downstream
#     per-frame grading nodes (CDL, LUT, policy guard) then reassemble.
# ============================================================

from radiance.config.constants import VERSION as __version__  # noqa: E402

import logging
logger = logging.getLogger("radiance.video.hdr")
import json
import math
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ---------------------------------------------------------------------------
# HDR metadata vocabulary
# ---------------------------------------------------------------------------

EOTF_OPTIONS    = ["PQ (ST.2084)", "HLG (BT.2100)", "Linear", "sRGB / BT.1886"]
GAMUT_OPTIONS   = ["BT.2020", "P3-D65", "P3-DCI", "BT.709", "ACEScg", "ACES2065-1"]
PEAK_NITS       = [100, 203, 400, 600, 1000, 4000, 10000]

# Text tokens injected into prompts for each gamut / EOTF combination.
# These have been empirically found to improve HDR-aware generation in
# LTX-2.x and HunyuanVideo when appended to the user prompt.
_GAMUT_TOKENS: Dict[str, str] = {
    "BT.2020":    "wide color gamut rec2020 vivid saturated",
    "P3-D65":     "DCI-P3 cinema color cinema grade",
    "P3-DCI":     "digital cinema DCI projection vibrant",
    "BT.709":     "standard dynamic range sRGB broadcast accurate",
    "ACEScg":     "aces linear light vfx reference",
    "ACES2065-1": "aces2065 archival linear reference wide primaries",
}

_EOTF_TOKENS: Dict[str, str] = {
    "PQ (ST.2084)": "HDR10 PQ perceptual quantizer specular highlights",
    "HLG (BT.2100)": "HLG hybrid log gamma broadcast HDR",
    "Linear":        "linear light EXR openexr 32bit",
    "sRGB / BT.1886": "sRGB gamma corrected standard dynamic range",
}

_PEAK_TOKENS: Dict[int, str] = {
    100:   "100 nits standard SDR",
    203:   "203 nits HLG reference",
    400:   "400 nits HDR bright highlights",
    600:   "600 nits bright specular HDR",
    1000:  "1000 nits HDR10 specular bright",
    4000:  "4000 nits ultra bright specular HDR cinema",
    10000: "10000 nits extreme HDR specular bright light",
}

_CAMERA_TOKENS = {
    "Handheld documentary": "handheld camera documentary natural light organic movement",
    "Locked off cinematic": "locked tripod cinematographic composition cinematic steady",
    "Slow push-in":         "slow dolly push in cinematic atmospheric tension",
    "Drone aerial":         "aerial drone high altitude wide establishing sweeping",
    "Tracking shot":        "tracking follow shot dynamic motion parallel subject",
    "Static time-lapse":    "static camera time lapse motion blur sky movement",
    "None":                 "",
}

_MOOD_TOKENS = {
    "Golden hour":      "golden hour warm light long shadows sunset cinematic",
    "Blue hour / dusk": "blue hour twilight dusk cool cinematic atmospheric",
    "Night":            "night low light neon moonlight dark cinematic",
    "Overcast flat":    "overcast diffuse soft light flat even natural",
    "High contrast":    "high contrast dramatic chiaroscuro deep shadows bright highlights",
    "Neon / cyberpunk": "neon lights cyberpunk urban night vivid saturated rain reflections",
    "Natural daylight":  "natural daylight neutral sun balanced outdoor",
    "None":              "",
}


# ---------------------------------------------------------------------------
# Tone-map helper (software fallback, no torch required for pixel path)
# ---------------------------------------------------------------------------

def _reinhard_tonemap(x: "torch.Tensor", peak: float = 1.0) -> "torch.Tensor":
    """Global Reinhard operator: x / (1 + x/peak)."""
    return x / (1.0 + x / max(peak, 1e-7))


def _reinhard_extended(x: "torch.Tensor", unit: float, white: float) -> "torch.Tensor":
    """Extended Reinhard in units of `unit`: L (1 + L / W^2) / (1 + L), with
    L = x / unit and W = white / unit, returned in units of `unit`. It maps
    `white` exactly to `unit`, so the brightest input lands on the target
    rather than at half of it as plain Reinhard does. With W <= 1 there is no
    headroom to compress and the curve is the identity."""
    unit = max(float(unit), 1e-12)
    w = float(white) / unit
    lum = x / unit
    if w <= 1.0 + 1e-9:
        return x
    return lum * (1.0 + lum / (w * w)) / (1.0 + lum) * unit


def _srgb_encode(x: "torch.Tensor") -> "torch.Tensor":
    """IEC 61966-2-1 sRGB inverse EOTF on display-relative linear light."""
    x = x.clamp(0.0, 1.0)
    return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * x.clamp(min=1e-12).pow(1.0 / 2.4) - 0.055)


def _pq_encode(x: "torch.Tensor") -> "torch.Tensor":
    """
    BT.2100 PQ EOTF (signal→display).
    Input: linear light normalised to [0,1] where 1 = 10 000 nits.
    Output: PQ code value [0,1].
    """
    m1, m2 = 0.1593017578125, 78.84375
    c1, c2, c3 = 0.8359375, 18.8515625, 18.6875
    xp = x.clamp(0).pow(m1)
    return ((c1 + c2 * xp) / (1 + c3 * xp)).pow(m2)


def _hlg_encode(x: "torch.Tensor") -> "torch.Tensor":
    """BT.2100 HLG OETF. Input: linear [0,1]. Output: HLG signal [0,1]."""
    a, b, c = 0.17883277, 0.28466892, 0.55991073
    out = torch.where(
        x <= 1.0 / 12.0,
        (3.0 * x).sqrt(),
        a * (12.0 * x - b).clamp(min=1e-8).log() + c,
    )
    return out.clamp(0, 1)


# ===========================================================================
# Node: RadianceVideoHDRConditioner
# ===========================================================================

def _rec709_to(gamut: str):
    """Row-vector matrix from linear Rec.709 to ``gamut`` and a report line."""
    import numpy as np
    from radiance.color import matrices as M
    if gamut == "BT.709":
        return None, "Rec.709 (no conversion)"
    if gamut == "BT.2020":
        return M.SRGB_TO_REC2020, "Rec.709 -> BT.2020"
    if gamut in ("P3-D65", "P3-DCI"):
        note = "Rec.709 -> P3-D65"
        if gamut == "P3-DCI":
            note += " (P3-DCI treated as P3-D65 primaries, no DCI white adaptation)"
        return M.ACESCG_TO_P3D65 @ M.SRGB_TO_ACESCG, note
    if gamut == "ACEScg":
        return M.SRGB_TO_ACESCG, "Rec.709 -> ACEScg (AP1)"
    if gamut == "ACES2065-1":
        return np.linalg.inv(M.ACES_AP0_TO_AP1) @ M.SRGB_TO_ACESCG, "Rec.709 -> ACES2065-1 (AP0)"
    return None, f"{gamut}: unknown, left as Rec.709"


class RadianceVideoHDRConditioner:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Video"
    DESCRIPTION = "Condition a video model on HDR metadata for luminance-aware sampling."
    """
    Add HDR descriptors to an already-encoded conditioning.

    A CONDITIONING is token embeddings; appending words to it needs the text
    encoder. With ``clip`` connected, the descriptors are encoded and
    concatenated onto every entry (ComfyUI's ConditioningConcat), scaled by
    token_strength. Without ``clip`` the conditioning passes through
    unchanged and hdr_metadata_json says so. No ComfyUI model reads the
    metadata dict; it is carried for RadianceVideoHDRDecode.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING", {
                    "tooltip": "Encoded positive prompt. The HDR descriptor embeddings are concatenated "
                               "onto each entry (needs clip)."}),
                "peak_nits": ([str(n) for n in PEAK_NITS], {"default": "1000",
                    "tooltip": "Mastering peak in nits: adds a luminance descriptor to the tokens and is "
                               "stored as peak_nits in hdr_metadata_json for RadianceVideoHDRDecode."}),
                "target_gamut": (GAMUT_OPTIONS, {"default": "BT.2020",
                    "tooltip": "Adds a gamut descriptor to the tokens and is stored as gamut in "
                               "hdr_metadata_json (RadianceVideoHDRDecode converts to it)."}),
                "eotf": (EOTF_OPTIONS, {"default": "PQ (ST.2084)",
                    "tooltip": "Adds a transfer-function descriptor to the tokens and is stored in "
                               "hdr_metadata_json. RadianceVideoHDRDecode uses its own output_eotf."}),
            },
            "optional": {
                "clip": ("CLIP", {"tooltip": "Text encoder used for positive. Required for the "
                                             "descriptors to reach the model."}),
                "camera_move": (list(_CAMERA_TOKENS.keys()), {"default": "None",
                    "tooltip": "Adds camera-movement words to the descriptor tokens. None adds nothing."}),
                "mood": (list(_MOOD_TOKENS.keys()), {"default": "None",
                    "tooltip": "Adds lighting-mood words to the descriptor tokens. None adds nothing."}),
                "extra_hdr_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Additional HDR descriptors appended to conditioning tokens",
                }),
                "inject_metadata_embedding": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Store the HDR metadata in the conditioning for Radiance nodes. "
                               "No ComfyUI model reads it.",
                }),
                "token_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Scale of the concatenated descriptor embeddings (1.0 = as encoded). "
                               "Needs clip.",
                }),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("positive", "hdr_metadata_json")
    FUNCTION = "condition"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Video"

    def condition(
        self,
        positive,
        peak_nits: str = "1000",
        target_gamut: str = "BT.2020",
        eotf: str = "PQ (ST.2084)",
        clip=None,
        camera_move: str = "None",
        mood: str = "None",
        extra_hdr_prompt: str = "",
        inject_metadata_embedding: bool = True,
        token_strength: float = 1.0,
    ):
        nits = int(peak_nits)
        meta = {
            "peak_nits": nits,
            "gamut": target_gamut,
            "eotf": eotf,
            "camera": camera_move,
            "mood": mood,
            "token_strength": token_strength,
        }

        token_parts = [
            _GAMUT_TOKENS.get(target_gamut, ""),
            _EOTF_TOKENS.get(eotf, ""),
            _PEAK_TOKENS.get(nits, f"{nits} nits HDR"),
            _CAMERA_TOKENS.get(camera_move, ""),
            _MOOD_TOKENS.get(mood, ""),
            extra_hdr_prompt.strip(),
        ]
        hdr_tokens = ", ".join(p for p in token_parts if p)
        meta["hdr_tokens"] = hdr_tokens

        extra_embed = None
        if clip is not None and hdr_tokens and token_strength > 0 and HAS_TORCH:
            with torch.no_grad():
                extra_embed = clip.encode_from_tokens(clip.tokenize(hdr_tokens))
            if isinstance(extra_embed, (tuple, list)):
                extra_embed = extra_embed[0]
            extra_embed = extra_embed * float(token_strength)

        out_cond = []
        applied = False
        for cond_tensor, cond_dict in positive:
            new_dict = dict(cond_dict)
            if inject_metadata_embedding:
                new_dict["radiance_hdr"] = meta
            tok = cond_tensor
            if extra_embed is not None and extra_embed.shape[-1] == cond_tensor.shape[-1]:
                e = extra_embed.to(cond_tensor.device, cond_tensor.dtype)
                if e.shape[0] != tok.shape[0]:
                    e = e[:1].expand(tok.shape[0], -1, -1)
                tok = torch.cat((tok, e), dim=1)
                applied = True
            out_cond.append([tok, new_dict])

        if applied:
            meta["applied_to_model"] = True
        elif clip is None:
            meta["applied_to_model"] = False
            meta["note"] = "clip not connected: conditioning unchanged, descriptors not encoded"
        elif not hdr_tokens or token_strength <= 0:
            meta["applied_to_model"] = False
            meta["note"] = "nothing to add (no descriptors or token_strength 0)"
        else:
            meta["applied_to_model"] = False
            meta["note"] = "clip width does not match the conditioning; descriptors not added"
        if not applied:
            logger.info("[VideoHDRConditioner] %s", meta["note"])
        return (out_cond, json.dumps(meta, indent=2))


# ===========================================================================
# Node: RadianceVideoHDRDecode
# ===========================================================================

class RadianceVideoHDRDecode:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Video"
    DESCRIPTION = ("Encode decoded sRGB video frames (IMAGE, not latents) to an HDR signal at the "
                   "metadata's peak nits and gamut, with PQ or HLG output and an SDR preview.")
    """
    Post-process DiT video output through the Radiance HDR pipeline.

    Accepts:
      • A pixel-space IMAGE tensor [B, H, W, 3] (already decoded by ComfyUI VAE)
      • OR a raw latent LATENT (will apply simple linear decode)

    Pipeline:
      1. Exposure normalise to target peak nits
      2. Tone-map (Reinhard or pass-through)
      3. Gamut clip / compress
      4. EOTF encode (PQ or HLG)
      5. Output: HDR signal tensor + SDR preview tensor + metadata JSON

    For full ACES/OCIO colour management, wire the output through the
    existing RadianceColorspaceTransform nodes after this node.
    """

    TONEMAP_MODES = ["Reinhard", "Linear clip", "Pass-through"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Decoded video frames, display-referred sRGB Rec.709 in [0, 1]. Linearised "
                               "with a pure 2.2 gamma; 1.0 is mapped to peak_nits."}),
                "hdr_metadata_json": ("STRING", {
                    "multiline": False,
                    "default": '{"peak_nits":1000,"gamut":"BT.2020","eotf":"PQ (ST.2084)"}',
                    "tooltip": "JSON from RadianceVideoHDRConditioner or manually entered",
                }),
                "tonemap": (cls.TONEMAP_MODES, {"default": "Reinhard",
                    "tooltip": "Reinhard: extended Reinhard whose white point is the brightest input "
                               "(1.0 lifted by a positive exposure_compensation_ev), so that value lands "
                               "exactly on peak_nits and the highlights above it roll off; at 0 EV or "
                               "less there is nothing to compress. Linear clip: clamp at 10,000 nits. "
                               "Pass-through: no curve (clamped at 10,000 nits by the encode)."}),
            },
            "optional": {
                "exposure_compensation_ev": ("FLOAT", {
                    "default": 0.0, "min": -6.0, "max": 6.0, "step": 0.1,
                    "tooltip": "EV adjustment before tone-mapping",
                }),
                "output_eotf": (EOTF_OPTIONS, {"default": "PQ (ST.2084)",
                    "tooltip": "Encoding of hdr_image. PQ: ST 2084 code values (1.0 = 10,000 nits). HLG: "
                               "BT.2100 OETF. Linear: clamped linear light normalised to 10,000 nits. "
                               "sRGB / BT.1886: the sRGB curve on light relative to peak_nits (1.0 = peak), "
                               "an SDR signal."}),
                "sdr_preview_nits": ("FLOAT", {
                    "default": 100.0, "min": 1.0, "max": 203.0,
                    "tooltip": "Nits shown as white-ish mid-range in sdr_preview: light is measured in "
                               "units of this value and rolled off with extended Reinhard so the brightest "
                               "input reaches display white (1.0). Lower = brighter preview.",
                }),
                "gamut_clip": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Clamp to [0, 1] of the encode container after the primaries "
                               "conversion (negatives from out-of-gamut colours, values above peak)",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("hdr_image", "sdr_preview", "decode_report")
    FUNCTION = "decode"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Video"

    def decode(
        self,
        image,
        hdr_metadata_json: str,
        tonemap: str,
        exposure_compensation_ev: float = 0.0,
        output_eotf: str = "PQ (ST.2084)",
        sdr_preview_nits: float = 100.0,
        gamut_clip: bool = True,
    ):
        report = [f"=== RadianceVideoHDRDecode v{__version__} ==="]

        try:
            meta = json.loads(hdr_metadata_json)
        except Exception:
            meta = {"peak_nits": 1000, "eotf": output_eotf}

        peak_nits = float(meta.get("peak_nits", 1000))
        report.append(f"Peak nits  : {peak_nits}")
        report.append(f"Tone-map   : {tonemap}")
        report.append(f"Output EOTF: {output_eotf}")
        report.append(f"EV offset  : {exposure_compensation_ev:+.2f}")

        if not HAS_TORCH:
            report.append("ERROR: torch not available — passing image through")
            return (image, image, "\n".join(report))

        # image: [B, H, W, 3] in [0,1] sRGB (ComfyUI convention)
        x = image.float()
        report.append(f"Input shape: {list(x.shape)}")

        # 1. Convert sRGB [0,1] → approximate scene-linear
        #    Simple gamma 2.2 linearise (full ACES path uses dedicated nodes)
        x_lin = x.clamp(0, 1).pow(2.2)

        # 2. Exposure compensation
        if exposure_compensation_ev != 0.0:
            x_lin = x_lin * (2.0 ** exposure_compensation_ev)

        # 2b. Primaries: generator output is Rec.709. The metadata gamut was
        #     only ever printed; an HDR10 signal needs BT.2020 primaries.
        gamut = str(meta.get("gamut", "BT.2020"))
        mat, gamut_note = _rec709_to(gamut)
        if mat is not None:
            m = torch.as_tensor(mat, dtype=x_lin.dtype, device=x_lin.device)
            x_lin = torch.einsum("...c,dc->...d", x_lin[..., :3], m)
        report.append(f"Gamut      : {gamut_note}")

        # 3. Scale to nit-normalised [0,1] where 1 = 10 000 nits
        #    Generator output is typically in [0,1] ≡ 100 nits SDR,
        #    so scale to target peak.
        x_nit = x_lin * (peak_nits / 10000.0)

        # 4. Tone-map
        #    The brightest input is 1.0 lifted by a positive EV; Reinhard puts
        #    it exactly on peak_nits. Plain x / (1 + x / peak) put input white
        #    at half of peak_nits.
        peak_n = peak_nits / 10000.0
        brightest = peak_n * (2.0 ** max(exposure_compensation_ev, 0.0))
        if tonemap == "Reinhard":
            x_tm = _reinhard_extended(x_nit.clamp(min=0.0), peak_n, brightest)
        elif tonemap == "Linear clip":
            x_tm = x_nit.clamp(0, 1)
        else:
            x_tm = x_nit      # Pass-through — may clip during EOTF

        # 5. Gamut clip
        if gamut_clip:
            x_tm = x_tm.clamp(0, 1)

        # 6. EOTF encode
        if output_eotf == "PQ (ST.2084)":
            x_enc = _pq_encode(x_tm)
        elif output_eotf == "HLG (BT.2100)":
            x_enc = _hlg_encode(x_tm)
        elif output_eotf == "sRGB / BT.1886":
            # An SDR signal: light relative to peak_nits through the sRGB
            # curve. It used to be the same clamped linear light as "Linear".
            x_enc = _srgb_encode(x_tm / peak_n)
        else:
            x_enc = x_tm.clamp(0, 1)

        # 7. SDR preview: light in units of sdr_preview_nits, rolled off so the
        #    brightest input reaches display white, then the 2.2 gamma the input
        #    was linearised with. It used to stay unnormalised at 1/10,000 of
        #    display scale, peaking near 0.12.
        sdr_unit = sdr_preview_nits / 10000.0
        x_sdr_lin = _reinhard_extended(x_nit.clamp(min=0.0), sdr_unit, max(brightest, sdr_unit)) / sdr_unit
        x_sdr = x_sdr_lin.clamp(0, 1).pow(1.0 / 2.2)

        report.append(f"HDR out range: [{x_enc.min():.4f}, {x_enc.max():.4f}]")
        report.append(f"SDR prev range: [{x_sdr.min():.4f}, {x_sdr.max():.4f}]")
        report.append(f"Output frames: {x_enc.shape[0]}")

        return (x_enc, x_sdr, "\n".join(report))


# ===========================================================================
# Node: RadianceVideoPromptBuilder
# ===========================================================================

class RadianceVideoPromptBuilder:
    """
    Structured prompt builder for HDR video generation.

    Combines subject, action, location, mood, lighting, camera movement,
    and HDR-specific descriptors into a single optimised prompt string
    ready for LTX-Video, HunyuanVideo, or Wan2.1.

    Also outputs a negative prompt with common video generation artefact
    suppressors.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "subject": ("STRING", {
                    "multiline": False,
                    "default": "a person walking through a neon-lit cityscape",
                    "tooltip": "Main subject and action. Placed first in the positive prompt.",
                }),
                "peak_nits": ([str(n) for n in PEAK_NITS], {"default": "1000",
                    "tooltip": "Adds a peak-luminance descriptor (e.g. '1000 nits HDR10 ...') to the "
                               "prompt. Prompt text only."}),
                "target_gamut": (GAMUT_OPTIONS, {"default": "BT.2020",
                    "tooltip": "Adds a gamut descriptor to the prompt. Prompt text only."}),
                "eotf": (EOTF_OPTIONS, {"default": "PQ (ST.2084)",
                    "tooltip": "Adds a transfer-function descriptor to the prompt. Prompt text only."}),
            },
            "optional": {
                "camera_move": (list(_CAMERA_TOKENS.keys()), {"default": "Slow push-in",
                    "tooltip": "Adds camera-movement words after the mood words. None adds nothing."}),
                "mood": (list(_MOOD_TOKENS.keys()), {"default": "Neon / cyberpunk",
                    "tooltip": "Adds lighting-mood words right after the subject. None adds nothing."}),
                "style_suffix": ("STRING", {
                    "multiline": True,
                    "default": "photorealistic, 8K, film grain, anamorphic lens",
                    "tooltip": "Free text appended at the end of the positive prompt.",
                }),
                "suppress_artefacts": ("BOOLEAN", {"default": True,
                    "tooltip": "On: negative_prompt is a fixed list of common video artefacts. "
                               "Off: negative_prompt is empty."}),
                "print_prompt": ("BOOLEAN", {"default": False,
                    "tooltip": "Also write both prompts to the ComfyUI console log."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt")
    FUNCTION = "build"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Video"
    DESCRIPTION = "Build structured video prompts combining text, style, and motion cues."
    OUTPUT_NODE = True

    # Standard video-generation negative descriptors
    _NEG_BASE = (
        "watermark, text, logo, subtitles, low quality, blurry, pixelated, "
        "noise artifacts, compression artifacts, oversaturated, washed out, "
        "flickering, temporal inconsistency, jitter, duplicate frames, "
        "low contrast, flat lighting, sdr, overexposed, clipped highlights, "
        "banding, aliasing, distorted faces"
    )

    def build(
        self,
        subject: str,
        peak_nits: str,
        target_gamut: str,
        eotf: str,
        camera_move: str = "None",
        mood: str = "None",
        style_suffix: str = "",
        suppress_artefacts: bool = True,
        print_prompt: bool = False,
    ):
        nits = int(peak_nits)
        parts = [subject.strip()]

        for key, tokens in [
            (mood,        _MOOD_TOKENS),
            (camera_move, _CAMERA_TOKENS),
        ]:
            tok = tokens.get(key, "")
            if tok:
                parts.append(tok)

        # HDR tokens
        hdr_parts = [
            _GAMUT_TOKENS.get(target_gamut, ""),
            _EOTF_TOKENS.get(eotf, ""),
            _PEAK_TOKENS.get(nits, f"{nits} nits HDR"),
        ]
        parts.extend(p for p in hdr_parts if p)

        if style_suffix.strip():
            parts.append(style_suffix.strip())

        positive = ", ".join(parts)
        negative = self._NEG_BASE if suppress_artefacts else ""

        if print_prompt:
            logger.info(f"[RadianceVideoPromptBuilder]\nPositive: {positive}\nNegative: {negative}")

        return (positive, negative)


# ===========================================================================
# Node: RadianceVideoFrameRouter
# ===========================================================================

class RadianceVideoFrameRouter:
    """
    Extract individual frames from a decoded video IMAGE tensor
    [B, H, W, 3], route them through per-frame HDR grading, then
    reassemble into a video tensor.

    This node outputs a single frame per call.  To process all frames,
    use ComfyUI's native batch iteration.

    Outputs:
      frame_image   — [1, H, W, 3] single frame for grading
      frame_index   — which frame was extracted
      total_frames  — total frames in input batch
      passthrough   — original tensor unchanged (for branch wiring)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_image": ("IMAGE", {"tooltip": "Frame batch to pick from. Also returned unchanged "
                                                     "on passthrough."}),
                "frame_index": ("INT", {
                    "default": 0, "min": 0, "max": 4096,
                    "tooltip": "0-based frame to extract. Past the end it wraps or clamps to the last "
                               "frame, per wrap_index.",
                }),
            },
            "optional": {
                "wrap_index": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If frame_index >= total_frames, wrap around (modulo). Off: clamp to "
                               "the last frame.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "IMAGE")
    RETURN_NAMES = ("frame_image", "frame_index", "total_frames", "passthrough")
    FUNCTION = "route"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Video"
    DESCRIPTION = "Route individual video frames to different processing branches."

    def route(self, video_image, frame_index: int = 0, wrap_index: bool = True):
        total = video_image.shape[0]
        if wrap_index:
            idx = frame_index % max(total, 1)
        else:
            idx = min(frame_index, total - 1)

        frame = video_image[idx:idx+1]
        return (frame, idx, total, video_image)


# ===========================================================================
# Node: RadianceVideoAssembler
# ===========================================================================

class RadianceVideoAssembler:
    """
    Collect per-frame IMAGE tensors into a single video batch tensor
    [N_frames, H, W, 3] ready for RadianceVideoWriter / nodes_video_io.py.

    Connect the frame_image output from RadianceVideoFrameRouter (after
    per-frame grading) into this node's frame input on every iteration.
    The node accumulates frames in a stateful list and flushes when
    flush=True or when received frames == expected_total_frames.
    """

    _STORE: Dict[str, List] = {}    # key → list of frame tensors

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frame": ("IMAGE", {"tooltip": "Frame (or batch) to append to this session's buffer. "
                                               "Buffers live in memory and are lost on restart."}),
                "session_key": ("STRING", {"default": "video_session_0",
                    "tooltip": "Name of the accumulation buffer. Use a different key per clip being "
                               "assembled in parallel."}),
                "expected_total_frames": ("INT", {"default": 24, "min": 1,
                    "tooltip": "Output is complete, and the buffer cleared, once this many inputs have "
                               "arrived. Counts executions, not images, if frame is a batch."}),
            },
            "optional": {
                "flush": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Force output of accumulated frames now, even if incomplete",
                }),
                "reset": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Clear accumulated frames for this session key",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "BOOLEAN")
    RETURN_NAMES = ("video_image", "frames_accumulated", "is_complete")
    FUNCTION = "assemble"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Video"
    DESCRIPTION = "Assemble processed video frames back into a temporal sequence."

    def assemble(self, frame, session_key: str, expected_total_frames: int,
                 flush: bool = False, reset: bool = False):
        if reset:
            self._STORE[session_key] = []

        bucket = self._STORE.setdefault(session_key, [])
        bucket.append(frame.cpu() if HAS_TORCH else frame)

        n = len(bucket)
        complete = n >= expected_total_frames or flush

        if complete and HAS_TORCH:
            video = torch.cat(bucket, dim=0)
            self._STORE[session_key] = []  # reset after flush
            return (video, n, True)

        # Not ready yet — return what we have so far
        if HAS_TORCH and bucket:
            video = torch.cat(bucket, dim=0)
        else:
            video = frame
        return (video, n, complete)


# ===========================================================================
# Registration
# ===========================================================================

NODE_CLASS_MAPPINGS = {
    "RadianceVideoHDRConditioner": RadianceVideoHDRConditioner,
    "RadianceVideoHDRDecode":      RadianceVideoHDRDecode,
    "RadianceVideoPromptBuilder":  RadianceVideoPromptBuilder,
    "RadianceVideoFrameRouter":    RadianceVideoFrameRouter,
    "RadianceVideoAssembler":      RadianceVideoAssembler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceVideoHDRConditioner": "◎ Radiance Video HDR Conditioner",
    "RadianceVideoHDRDecode":      "◎ Radiance Video HDR Decode",
    "RadianceVideoPromptBuilder":  "◎ Radiance Video Prompt Builder",
    "RadianceVideoFrameRouter":    "◎ Radiance Video Frame Router",
    "RadianceVideoAssembler":      "◎ Radiance Video Assembler",
}
