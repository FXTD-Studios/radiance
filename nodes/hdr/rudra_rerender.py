"""
◎ Radiance SDR → HDR AI Fill v4: region re-render (RUDRA, experimental)
═══════════════════════════════════════════════════════════════════════

RUDRA AI fill v4 (RUDRA ``feat/genfill-v4-finish`` 72c4a62, 12 Nov 2026).
AI fill v3 draws only inside pixels that are fully clipped, often a fraction of
a per cent of the frame. The region re-render draws a whole bright region
again: a window, an open door, a sky parked just under white.

* The regions come from the SOURCE, the SDR the master was made from (codes at
  or above 240), not from the master's own SDR view.
* Wan2.1-VACE-1.3B with the v4 LoRA draws the region; the fill is finished as
  RUDRA's app does it: the source's colour (30 % of the model's kept), a level
  of 3.3 x the master's SDR white crossfaded in from code 240 to 253, the
  model's detail on it with dark detail at most 0.3 stops deep, edge rules, and
  0.8 of the previous frame's level kept. It acts only where the source is near
  white in every channel; a saturated highlight is left to the master.
* Inside the region the SDR's codes are NOT kept, unlike v3. Outside it every
  pixel is the master's, bit for bit.

Experimental and off unless wired: v4 has passed RUDRA's gates on one shot
(09V) and the held-out set, and fails the structure gate on a second shot. It
was tested on RUDRA's analytic route (the SDR read as BT.1886). The v4 LoRA is
not published; it is never downloaded. What it draws is a plausible guess:
check every frame.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

import comfy.model_management
import comfy.utils

from radiance.nodes.hdr.rudra_aifill import ENCODINGS, _from_acescg, _to_acescg

logger = logging.getLogger("radiance.nodes.hdr.rudra_rerender")

CONDITIONS = ["bt1886", "aces"]                 # rudra_genfill.Rerender.CONDITIONS
FINISHES = ["RUDRA app (12 Nov 2026)", "off (the model as registered)"]


def source_codes(source: torch.Tensor, n: int, H: int, W: int) -> list:
    """The SDR source (display-referred IMAGE, 0..1) -> n frames of 8-bit codes (H, W, 3) uint8 at the
    master's size, the way RUDRA reads it (area resize, then rounded codes). One source frame with
    several master frames is used for all of them."""
    import cv2
    src = source.float()
    if src.dim() == 3:
        src = src.unsqueeze(0)
    m = int(src.shape[0])
    if m not in (1, n):
        raise ValueError(f"the source has {m} frames and the master {n}: give the SDR the master was made from, "
                         "frame for frame")
    out = []
    for i in range(n):
        f = src[min(i, m - 1), ..., :3].cpu().numpy().astype(np.float32)
        if f.shape[:2] != (H, W):
            f = cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)
        out.append(np.rint(np.clip(f, 0.0, 1.0) * 255.0).astype(np.uint8))
    return out


class RadianceSDRToHDRAIFillRerender:
    """RUDRA AI fill v4: a bright region of an HDR master drawn again by Wan VACE + the v4 LoRA."""

    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    DESCRIPTION = (
        "RUDRA AI fill v4, region re-render (experimental): draws a whole bright region of an HDR master "
        "again (a window, an open door, a sky parked just under white), with RUDRA's app finish. Inside the "
        "region the SDR's codes are not kept; outside it every pixel is the master's. Needs the v4 LoRA, "
        "which is not published. Plausible, not recovered."
    )
    FUNCTION = "rerender"
    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "STRING")
    RETURN_NAMES = ("image", "generated_mask", "reconstructed_mask", "report")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The HDR master (scene-linear, 1.0 = diffuse white), a batch read as "
                                               "ordered frames of one shot. RUDRA tested v4 on its analytic route."}),
                "image_encoding": (ENCODINGS, {"default": "Linear (Rec.709)",
                    "tooltip": "How the master is encoded. The result leaves in the same encoding."}),
                "source": ("IMAGE", {"tooltip": "The SDR the master was made from (display-referred, 0..1), frame "
                                                "for frame. The regions are its codes at 240 and above. Resized to "
                                                "the master's size if it differs."}),
                "prompt_highlight": ("STRING", {"default": "", "multiline": True,
                    "tooltip": "What is in the bright area (e.g. 'an overcast sky through an open door'). Empty: "
                               "texture with no subject. A new prompt loads the 11 GB umT5 text encoder once."}),
            },
            "optional": {
                "reconstructed_mask": ("MASK", {"tooltip": "The master's reconstructed_mask; the output becomes "
                                                           "max(it, generated)."}),
                "condition": (CONDITIONS, {"default": "bt1886",
                    "tooltip": "How the model reads the source. bt1886: code ^ 2.4, RUDRA's analytic route (what "
                               "v4 was gated on). aces: the ACES 2.0 inverse, for a master from the RUDRA ACES 2 route."}),
                "finish": (FINISHES, {"default": FINISHES[0],
                    "tooltip": "RUDRA app: source colour (0.3 of the model's), level 3.3 x SDR white, smoothing "
                               "0.8, dark detail at most 0.3 stops, near-white sources only. off: the model's own "
                               "colour and level, as registered on 9 Nov (it failed the level and flicker gates)."}),
                "steps": ("INT", {"default": 30, "min": 4, "max": 80, "step": 1,
                    "tooltip": "Denoising steps per window. RUDRA uses 30."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF,
                    "tooltip": "Noise seed. The same seed and settings give the same fill."}),
                "lora": ("STRING", {"default": "",
                    "tooltip": "The v4 LoRA (rudra_genfill.safetensors of research_genfill_v4_20261109), or its "
                               "folder. Empty searches RADIANCE_RUDRA_GENFILL_V4, then "
                               "models/radiance/rudra_genfill_v4/. Never downloaded: it is not published."}),
                "base_model": ("STRING", {"default": "",
                    "tooltip": "Wan2.1-VACE-1.3B-diffusers folder. Empty searches RADIANCE_WAN_VACE, "
                               "models/radiance/ and models/diffusers/."}),
                "fill_package": ("STRING", {"default": "",
                    "tooltip": "The AI fill v3 folder, used here for its prompt cache only (the base prompt's "
                               "embedding, so no 11 GB text encoder for an empty prompt)."}),
                "allow_download": ("BOOLEAN", {"default": True,
                    "tooltip": "Download the Wan base model (7.7 GB) and the v3 package's prompt cache (271 MB) "
                               "if missing. Never the v4 LoRA. RADIANCE_ALLOW_DOWNLOADS=0 always wins."}),
                "keep_loaded": ("BOOLEAN", {"default": True,
                    "tooltip": "Keep Wan VACE on the GPU between runs. Off frees it after this node."}),
            },
        }

    @torch.no_grad()
    def rerender(self, image, image_encoding, source, prompt_highlight, reconstructed_mask=None,
                 condition="bt1886", finish=FINISHES[0], steps=30, seed=0, lora="", base_model="",
                 fill_package="", allow_download=True, keep_loaded=True):
        from radiance.model import rudra_genfill as G

        img = image.float()
        if img.dim() == 3:
            img = img.unsqueeze(0)
        n, H, W = (int(v) for v in img.shape[:3])
        rgb, extra = img[..., :3], img[..., 3:]
        prior = (reconstructed_mask.float().reshape(-1, H, W) if reconstructed_mask is not None
                 else torch.zeros((n, H, W)))
        if prior.shape[0] == 1 and n > 1:
            prior = prior.expand(n, H, W)
        if n == 0:
            z = img[..., 0]
            return (img, z, z, "frames: 0")

        ms = G.MaskSettings(max_gain_stops=0.0, **G.RUDRA_APP_V4_MASK).check()
        codes = source_codes(source, n, H, W)
        if not any(int(c.max(-1).max()) >= ms.tau_high for c in codes):
            z = torch.zeros((n, H, W))
            return (image, z, torch.maximum(prior.cpu(), z),
                    f"RUDRA AI fill v4: nothing to re-render — no source code at {ms.tau_high} or above.")

        lora_path = G.resolve_v4_lora(lora)
        if lora_path is None:                    # a missing file, not a fault: FileNotFoundError
            raise FileNotFoundError(
                "RUDRA AI fill v4 (region re-render) needs its LoRA, which is experimental and not published. "
                "Put rudra_genfill.safetensors of research_genfill_v4_20261109 in "
                f"ComfyUI/models/radiance/{G.V4_DIRNAME}/, set {G.V4_ENV}, or give its path in 'lora'. "
                "For clipped pixels only, use SDR → HDR AI Fill (RUDRA v3).")
        G.check_runtime()
        allow = None if allow_download else False
        package = G.ensure_package(fill_package, allow=allow)
        if package is None:                      # no v3 prompt cache: one beside the v4 LoRA
            package = Path(lora_path).parent
        base = G.ensure_base(base_model, allow=allow)
        if base is None:
            raise RuntimeError(
                f"Wan2.1-VACE-1.3B-diffusers is not installed and could not be downloaded. Download "
                f"https://huggingface.co/{G.BASE_REPO} into ComfyUI/models/radiance/{G.BASE_DIRNAME}/ "
                f"(or models/diffusers/), or set {G.BASE_ENV}.")

        masters = [_to_acescg(rgb[i].cpu().numpy(), image_encoding) for i in range(n)]
        view = None
        if condition == "aces":
            from radiance.model.rudra_aces2 import aces2_view
            view = aces2_view()

        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        device = comfy.model_management.get_torch_device()
        model = G.get_model(base, Path(package), device, keep_loaded=bool(keep_loaded), lora=Path(lora_path))

        s = G.FillSettings(steps=int(steps), seed=int(seed), prompt_highlight=str(prompt_highlight or "").strip())
        for text in {s.prompt, s.prompt_highlight} - {""}:
            model.prompts.get(text)
        model.prompts.release()

        app = finish == FINISHES[0]
        rer = G.Rerender(condition=str(condition), **(G.RUDRA_APP_V4 if app else {})).check()
        windows = len(G.plan_windows(n, 33, 8))
        bar = comfy.utils.ProgressBar(max(1, windows * int(steps)))

        def progress(**o):
            if "step" in o:
                bar.update_absolute(int((o["window"] - 1 + o["step"] / max(1, o["steps"])) * int(steps)))

        try:
            outs, alphas, rep = G.fill_frames(masters, model, view, s, ms, window=33, overlap=8,
                                              max_area=832 * 480, progress=progress,
                                              rerender=rer, source_codes=codes)
        finally:
            if not keep_loaded:
                G.release_model()

        result = torch.from_numpy(np.stack([_from_acescg(o, image_encoding) for o in outs])).to(img.device)
        gen = torch.from_numpy(np.stack(alphas)).to(img.device)
        keep = (gen <= 0).unsqueeze(-1)          # outside the region: the input, bit for bit
        result = torch.where(keep, rgb, result.to(rgb.dtype))
        if extra.shape[-1] > 0:
            result = torch.cat([result, extra], dim=-1)
        recon = torch.maximum(prior.to(gen.device), gen)
        st = rep["stats"]
        sha = (rep.get("model") or {}).get("lora_sha256")
        lines = [
            "RUDRA AI fill v4, region re-render (EXPERIMENTAL; Wan2.1-VACE-1.3B + v4 LoRA) — plausible, not recovered",
            f"LoRA: {lora_path} ({'the gated v4 LoRA' if sha == G.GENFILL_V4_SHA256 else 'NOT the gated v4 LoRA: sha256 ' + str(sha)[:16]})",
            f"condition: {condition}; finish: {'RUDRA app ' + json.dumps(G.RUDRA_APP_V4) if app else 'off'}",
            f"frames: {st['frames']}, windows: {st['windows']} ({st['windows_generated']} generated), "
            f"working size {st['working_size'][1]}x{st['working_size'][0]}"
            + (f", master SDR white {st['rerender_white']:g}" if "rerender_white" in st else ""),
            f"generated pixels: {st['generated_pixels']} ({100 * st['generated_fraction']:.3f} %)",
            f"frames flagged for flicker (> {G.TEMPORAL_FLAG} stops added): {st['flagged_frames']}",
            "contract: inside the region the SDR codes are NOT kept; outside it every pixel is the master's",
            json.dumps({"model": rep["model"], "settings": rep["settings"], "rerender": rep["rerender"]}, default=str),
        ]
        return (result, gen, recon, "\n".join(lines))


NODE_CLASS_MAPPINGS = {"RadianceSDRToHDRAIFillRerender": RadianceSDRToHDRAIFillRerender}
NODE_DISPLAY_NAME_MAPPINGS = {"RadianceSDRToHDRAIFillRerender": "◎ Radiance SDR → HDR AI Fill v4 Re-render (RUDRA, experimental)"}
