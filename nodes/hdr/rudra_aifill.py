"""
◎ Radiance SDR → HDR AI Fill (RUDRA AI fill v3)
════════════════════════════════════════════════

Draws the shape inside fully clipped highlights of an HDR master and sets
their brightness, the way RUDRA 1.0's AI fill does:

* Wan2.1-VACE-1.3B (Wan-AI, Apache 2.0) with the FXTD LoRA draws the SHAPE,
  in windows of 33 frames that overlap by 8 so a clip joins without seams;
* a small estimator (``brightness.json``) sets HOW BRIGHT each clipped area
  was; ``brightness_offset`` moves it by up to ±3 stops;
* the fill is blended in only where the source is fully clipped, never more
  than 3 stops over the master, and never so that an 8-bit code of the
  master's ACES 2.0 SDR view changes. Every other pixel is copied bit for bit.

Region prompts (``prompt_highlight``, ``prompt_shadow``) steer what is drawn
inside each region, SwitchHDR-style; empty uses RUDRA's base prompt.

Wire it after SDR → HDR Universal or Recover running RUDRA ACES 2 (Linear or
AP0 output). What it draws is plausible, not recovered: review every frame.
The base model (``Wan2.1-VACE-1.3B-diffusers``, 7.7 GB) and the RUDRA AI fill
v3 package (271 MB, Apache 2.0) download themselves the first time the node
runs on a frame with clipped highlights. Needs diffusers >= 0.35 (installed
with Radiance) and about 16 GB of VRAM.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

import comfy.model_management
import comfy.utils

logger = logging.getLogger("radiance.nodes.hdr.rudra_aifill")

ENCODINGS = ["Linear (Rec.709)", "Linear ACES2065-1 (AP0)", "ACEScg"]
WINDOWS = [33, 17, 9, 49]


def _to_acescg(rgb: np.ndarray, encoding: str) -> np.ndarray:
    from radiance.model.rudra_aces2 import M_AP1_TO_AP0, M_REC709_TO_AP1
    if encoding == "ACEScg":
        return rgb.astype(np.float32)
    m = np.linalg.inv(M_AP1_TO_AP0) if encoding.startswith("Linear ACES2065-1") else M_REC709_TO_AP1
    return (rgb.astype(np.float64) @ m.T).astype(np.float32)


def _from_acescg(rgb: np.ndarray, encoding: str) -> np.ndarray:
    from radiance.model.rudra_aces2 import M_AP1_TO_AP0, M_AP1_TO_REC709
    if encoding == "ACEScg":
        return rgb.astype(np.float32)
    m = M_AP1_TO_AP0 if encoding.startswith("Linear ACES2065-1") else M_AP1_TO_REC709
    return (rgb.astype(np.float64) @ m.T).astype(np.float32)


class RadianceSDRToHDRAIFill:
    """RUDRA AI fill v3 on an HDR master: shape from Wan VACE + LoRA, brightness estimated."""

    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    DESCRIPTION = (
        "RUDRA AI fill v3: draws the shape inside fully clipped highlights of an HDR master "
        "(Wan2.1-VACE-1.3B + the FXTD LoRA) and sets their brightness with RUDRA's estimator. "
        "Region prompts, ±3 stop brightness offset. No 8-bit code of the master's ACES 2.0 SDR "
        "view changes; untouched pixels are bit-identical. Plausible, not recovered."
    )
    FUNCTION = "fill"
    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "STRING")
    RETURN_NAMES = ("image", "generated_mask", "reconstructed_mask", "report")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The HDR master from SDR → HDR Universal or Recover running RUDRA "
                                               "ACES 2 (scene-linear, 1.0 = diffuse white). A batch is read as "
                                               "ordered frames of one shot."}),
                "image_encoding": (ENCODINGS, {"default": "Linear (Rec.709)",
                    "tooltip": "How the master is encoded: the Universal/Recover Linear output, their AP0 output, "
                               "or ACEScg. The result leaves in the same encoding."}),
                "prompt_highlight": ("STRING", {"default": "", "multiline": True,
                    "tooltip": "What to draw inside clipped highlights (e.g. 'soft clouds in a bright blue sky'). "
                               "Empty uses RUDRA's base prompt. A new prompt loads the 11 GB umT5 text encoder "
                               "on the CPU once; its embedding is cached."}),
                "brightness_offset": ("FLOAT", {"default": 0.0, "min": -3.0, "max": 3.0, "step": 0.05,
                    "tooltip": "Stops added to every clipped area's estimated brightness (RUDRA's Highlight "
                               "brightness). Still capped at max_gain_stops over the master."}),
            },
            "optional": {
                "reconstructed_mask": ("MASK", {"tooltip": "The reconstructed_mask from Universal/Recover; the "
                                                           "reconstructed_mask output becomes max(it, generated)."}),
                "fill_shadows": ("BOOLEAN", {"default": False,
                    "tooltip": "Also fill fully crushed shadows (code 0). Off by default, as in RUDRA 1.0: the "
                               "registered gates covered highlights."}),
                "prompt_shadow": ("STRING", {"default": "", "multiline": True,
                    "tooltip": "What to draw inside crushed shadows, when fill_shadows is on."}),
                "highlight_context_code": ("INT", {"default": 250, "min": 200, "max": 255, "step": 1,
                    "tooltip": "8-bit code from which the model sees a highlight as context (tau_high). The fill is "
                               "still blended only at code 255."}),
                "max_gain_stops": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 6.0, "step": 0.25,
                    "tooltip": "The fill is never more than this many stops brighter than the master (0 = no cap)."}),
                "shape_gain": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "How strongly the generated shape varies the brightness inside a clipped area "
                               "(0 = flat at the estimate)."}),
                "use_brightness_estimator": ("BOOLEAN", {"default": True,
                    "tooltip": "v3: brightness from the estimator plus the offset. Off: the model's own brightness "
                               "(RUDRA v1-v2 behaviour)."}),
                "steps": ("INT", {"default": 30, "min": 4, "max": 80, "step": 1,
                    "tooltip": "Denoising steps per window (flow matching, UniPC). RUDRA uses 30."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF,
                    "tooltip": "Noise seed. The same seed and settings give the same fill."}),
                "guidance": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 8.0, "step": 0.1,
                    "tooltip": "Above 1: classifier-free guidance against negative_prompt (twice the compute). "
                               "RUDRA's gates were run at 1."}),
                "negative_prompt": ("STRING", {"default": "", "multiline": False,
                    "tooltip": "What to steer away from; used only when guidance is above 1."}),
                "window": (WINDOWS, {"default": 33,
                    "tooltip": "Frames per generation window (4k+1). RUDRA uses 33."}),
                "overlap": ("INT", {"default": 8, "min": 0, "max": 16, "step": 1,
                    "tooltip": "Frames the next window re-reads from the previous one, so windows join without a "
                               "seam. Must be below window."}),
                "max_area": ("INT", {"default": 832 * 480, "min": 256 * 256, "max": 1280 * 720, "step": 1024,
                    "tooltip": "Working size in pixels (aspect kept); the fill is brought back to full size. "
                               "RUDRA uses 832 x 480."}),
                "base_model": ("STRING", {"default": "",
                    "tooltip": "Wan2.1-VACE-1.3B-diffusers folder. Empty searches RADIANCE_WAN_VACE, "
                               "models/radiance/ and models/diffusers/."}),
                "fill_package": ("STRING", {"default": "",
                    "tooltip": "RUDRA AI fill v3 folder (rudra_genfill.safetensors, brightness.json, prompt_cache). "
                               "Empty searches models/radiance/rudra_genfill_v3 and RADIANCE_RUDRA_GENFILL."}),
                "allow_download": ("BOOLEAN", {"default": True,
                    "tooltip": "Download what is missing on first use, into models/radiance: the AI fill v3 "
                               "package (271 MB) and the Wan2.1-VACE-1.3B base model (7.7 GB; its 11.4 GB text "
                               "encoder only when a custom prompt is first used). Pinned and hash-checked; "
                               "resumes if interrupted. RADIANCE_ALLOW_DOWNLOADS=0 always wins."}),
                "keep_loaded": ("BOOLEAN", {"default": True,
                    "tooltip": "Keep Wan VACE on the GPU between runs. Off frees it after this node."}),
            },
        }

    @torch.no_grad()
    def fill(self, image, image_encoding, prompt_highlight, brightness_offset,
             reconstructed_mask=None, fill_shadows=False, prompt_shadow="",
             highlight_context_code=250, max_gain_stops=3.0, shape_gain=1.0,
             use_brightness_estimator=True, steps=30, seed=0, guidance=1.0, negative_prompt="",
             window=33, overlap=8, max_area=832 * 480, base_model="", fill_package="",
             allow_download=True, keep_loaded=True):
        from radiance.model import rudra_genfill as G
        from radiance.model.rudra_aces2 import aces2_view

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

        masters = [_to_acescg(rgb[i].cpu().numpy(), image_encoding) for i in range(n)]
        view = aces2_view()
        clipped = sum(int((np.clip(view.forward(G._small(m, 512)), 0, 1) >= 254.5 / 255).all(-1).sum())
                      for m in masters)
        if clipped == 0 and not fill_shadows:
            report = "RUDRA AI fill v3: nothing to fill — no fully clipped highlights in the master's SDR view."
            z = torch.zeros((n, H, W))
            return (image, z, torch.maximum(prior.cpu(), z), report)

        # Everything below downloads itself on first use (RADIANCE_ALLOW_DOWNLOADS=0 or
        # allow_download off stops that, with a message naming the files and folders).
        G.check_runtime()
        allow = None if allow_download else False
        package = G.ensure_package(fill_package, allow=allow)
        if package is None:
            from radiance.model.rudra_download import GENFILL_V3
            raise RuntimeError(
                "RUDRA AI fill v3 is not installed and could not be downloaded. Put "
                "rudra_genfill.safetensors, brightness.json and prompt_cache/ in "
                f"ComfyUI/models/radiance/{GENFILL_V3.dirname}/ (from {GENFILL_V3.page_url}) "
                "or set RADIANCE_RUDRA_GENFILL.")
        base = G.ensure_base(base_model, allow=allow)
        if base is None:
            raise RuntimeError(
                f"Wan2.1-VACE-1.3B-diffusers is not installed and could not be downloaded. Download "
                f"https://huggingface.co/{G.BASE_REPO} into ComfyUI/models/radiance/{G.BASE_DIRNAME}/ "
                f"(or models/diffusers/), or set {G.BASE_ENV}.")

        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        device = comfy.model_management.get_torch_device()
        model = G.get_model(base, Path(package), device, keep_loaded=bool(keep_loaded))

        s = G.FillSettings(steps=int(steps), seed=int(seed), guidance=float(guidance),
                           negative=str(negative_prompt or ""),
                           prompt_highlight=str(prompt_highlight or ""),
                           prompt_shadow=str(prompt_shadow or "") if fill_shadows else "")
        for text in {s.prompt, s.prompt_highlight, s.prompt_shadow, s.negative} - {""}:
            model.prompts.get(text)
        model.prompts.release()

        ms = G.MaskSettings(tau_high=int(highlight_context_code), tau_low=4 if fill_shadows else -1,
                            max_gain_stops=float(max_gain_stops)).check()
        brightness = None
        if use_brightness_estimator:
            bpath = Path(package) / "brightness.json"
            brightness = G.BrightnessSettings(G.BrightnessEstimator.load(bpath), float(brightness_offset),
                                              float(shape_gain),
                                              identity={"estimator": str(bpath),
                                                        "estimator_sha256": G.sha256_file(bpath)})
        windows = len(G.plan_windows(n, int(window), min(int(overlap), int(window) - 1)))
        bar = comfy.utils.ProgressBar(max(1, windows * int(steps)))

        def progress(**o):
            if "step" in o:
                bar.update_absolute((o["window"] - 1) * int(steps) + o["step"])

        try:
            outs, alphas, rep = G.fill_frames(masters, model, view, s, ms, window=int(window),
                                              overlap=min(int(overlap), int(window) - 1),
                                              max_area=int(max_area), brightness=brightness,
                                              progress=progress)
        finally:
            if not keep_loaded:
                G.release_model()

        result = torch.from_numpy(np.stack([_from_acescg(o, image_encoding) for o in outs])).to(img.device)
        # Untouched pixels bit-identical: copy the input wherever nothing was generated.
        gen = torch.from_numpy(np.stack(alphas)).to(img.device)
        keep = (gen <= 0).unsqueeze(-1)
        result = torch.where(keep, rgb, result.to(rgb.dtype))
        if extra.shape[-1] > 0:
            result = torch.cat([result, extra], dim=-1)
        recon = torch.maximum(prior.to(gen.device), gen)
        st = rep["stats"]
        lines = [
            "RUDRA AI fill v3 (Wan2.1-VACE-1.3B + FXTD LoRA, Apache 2.0) — plausible, not recovered",
            f"frames: {st['frames']}, windows: {st['windows']} ({st['windows_generated']} generated), "
            f"working size {st['working_size'][1]}x{st['working_size'][0]}",
            f"generated pixels: {st['generated_pixels']} ({100 * st['generated_fraction']:.3f} %)",
            f"brightness: {'estimator, offset %+.2f stops' % brightness_offset if brightness else 'model'}; "
            f"cap {max_gain_stops:g} stops over the master",
            f"frames flagged for flicker (> {G.TEMPORAL_FLAG} stops added): {st['flagged_frames']}",
            "contract: no 8-bit code of the master's ACES 2.0 SDR view changed; untouched pixels bit-identical",
            json.dumps({"model": rep["model"], "settings": rep["settings"]}, default=str),
        ]
        return (result, gen, recon, "\n".join(lines))


NODE_CLASS_MAPPINGS = {"RadianceSDRToHDRAIFill": RadianceSDRToHDRAIFill}
NODE_DISPLAY_NAME_MAPPINGS = {"RadianceSDRToHDRAIFill": "◎ Radiance SDR → HDR AI Fill (RUDRA v3)"}
