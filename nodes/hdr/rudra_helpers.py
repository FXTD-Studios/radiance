"""
◎ RUDRA input helpers: Load Still, Fill Prompt
══════════════════════════════════════════════

* **◎ Load Still (RUDRA)** -- loads an SDR still the way RUDRA does: every code
  kept (8 or 16 bit), the EXIF orientation applied, and the file's own colour
  tag read (ICC profile, PNG cICP / sRGB / cHRM). A Display P3 phone photo is
  then mastered from Display P3 by ◎ SDR → HDR Analytic (RUDRA) instead of
  being read as Rec.709, and a portrait one is not sideways.
* **◎ RUDRA Fill Prompt** -- RUDRA's prompt presets for the AI fill and its
  suggestion from the shot's clipped areas (sky, sun, lamp, window, glints,
  flame, shadow detail), with no model. Wire its outputs into SDR → HDR AI
  Fill's prompt boxes.

Ported from FXTD-Studios/RUDRA (Apache 2.0): radiance.model.rudra_still and
radiance.model.rudra_prompts.
"""
from __future__ import annotations

import hashlib
import logging
import os

import numpy as np
import torch

from radiance.model.rudra_prompts import BY_ID, PRESETS, cached, suggest
from radiance.model.rudra_still import apply_orientation, read_still_colour

logger = logging.getLogger("radiance.nodes.hdr.rudra_helpers")

_STILL_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp")


def _input_files():
    try:
        import folder_paths
        d = folder_paths.get_input_directory()
        return sorted(f for f in os.listdir(d)
                      if os.path.isfile(os.path.join(d, f)) and f.lower().endswith(_STILL_EXT)) or [""]
    except Exception:
        return [""]


def _resolve(image: str, path: str) -> str:
    from radiance.path_utils import strip_path_quotes
    p = strip_path_quotes(path or "")
    if p:
        return p
    try:
        import folder_paths
        return folder_paths.get_annotated_filepath(image)
    except Exception:
        return image


def decode_still(data: bytes):
    """(H, W, C) float32 codes 0..1 as stored (no orientation, no colour change), and the bit depth."""
    import cv2
    try:
        im = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED) if data else None
    except cv2.error:
        im = None
    if im is None:
        raise ValueError("cannot decode this still (PNG, JPEG, TIFF, WebP or BMP)")
    if im.ndim == 2:
        im = im[..., None]
    if im.shape[-1] >= 3:
        order = [2, 1, 0] + list(range(3, im.shape[-1]))   # BGR(A) -> RGB(A)
        im = im[..., order]
    if im.dtype == np.uint8:
        return im.astype(np.float32) / 255.0, 8
    if im.dtype == np.uint16:
        return im.astype(np.float32) / 65535.0, 16
    if im.dtype in (np.float32, np.float64):
        raise ValueError("this still holds float pixels (HDR); read it with Radiance Read, not Load Still")
    raise ValueError(f"unsupported pixel type {im.dtype}")


class RadianceRudraLoadStill:
    """An SDR still with its EXIF orientation applied and its colour tag read, as RUDRA reads stills."""

    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    DESCRIPTION = (
        "Loads an SDR still as RUDRA does: every code kept (8 or 16 bit), the EXIF orientation applied, and "
        "the colour tag read (ICC profile, PNG cICP / sRGB / cHRM): Rec.709 / sRGB, Display P3, Rec.2020 or "
        "Adobe RGB. Wire `source` into SDR → HDR Analytic (RUDRA) so the master is made from the right "
        "primaries. A profile RUDRA does not know is reported and read as Rec.709, never guessed."
    )
    FUNCTION = "load"
    RETURN_TYPES = ("IMAGE", "MASK", "RUDRA_SOURCE", "STRING")
    RETURN_NAMES = ("image", "alpha", "source", "note")
    OUTPUT_TOOLTIPS = ("The still's codes, oriented, 0..1 (not converted).",
                       "Its alpha (1 where it has none).",
                       "Its primaries, for SDR → HDR Analytic (RUDRA)'s source input.",
                       "What was found and what is done, in words.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (_input_files(), {"image_upload": True,
                                           "tooltip": "A still in ComfyUI's input folder (or upload one)."}),
            },
            "optional": {
                "path": ("STRING", {"default": "", "tooltip": "A file anywhere instead. Empty: the image above."}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, image, path=""):
        p = _resolve(image, path)
        try:
            h = hashlib.sha256()
            with open(p, "rb") as f:
                h.update(f.read())
            return h.hexdigest()
        except OSError:
            return ""

    @classmethod
    def VALIDATE_INPUTS(cls, image, path=""):
        p = _resolve(image, path)
        return True if p and os.path.isfile(p) else f"no such file: {p or image}"

    def load(self, image, path=""):
        p = _resolve(image, path)
        if not p or not os.path.isfile(p):
            raise FileNotFoundError(f"[Load Still] no such file: {p or image}")
        with open(p, "rb") as f:
            data = f.read()
        if not data:
            raise FileNotFoundError(f"[Load Still] {p} is empty: no image in it")
        colour = read_still_colour(data)
        codes, bits = decode_still(data)
        codes = np.ascontiguousarray(apply_orientation(codes, colour.orientation))
        if codes.shape[-1] == 1:
            rgb, alpha = np.repeat(codes, 3, -1), np.ones(codes.shape[:2], np.float32)
        elif codes.shape[-1] == 2:
            rgb, alpha = np.repeat(codes[..., :1], 3, -1), codes[..., 1]
        else:
            rgb = codes[..., :3]
            alpha = codes[..., 3] if codes.shape[-1] > 3 else np.ones(codes.shape[:2], np.float32)
        note = f"{os.path.basename(p)}: {bits}-bit, {rgb.shape[1]}x{rgb.shape[0]}; {colour.note()}"
        source = {"primaries": colour.primaries, "note": colour.note(), "file": os.path.abspath(p),
                  "tagged": colour.tagged, "recognised": colour.recognised, "orientation": colour.orientation}
        return (torch.from_numpy(np.ascontiguousarray(rgb))[None], torch.from_numpy(np.ascontiguousarray(alpha))[None],
                source, note)


_PRESET_CHOICES = ["Suggest from the frames"] + [f"{p.label} ({p.region})" for p in PRESETS] + ["None"]
_CHOICE_ID = {f"{p.label} ({p.region})": p.id for p in PRESETS}


def _prompt_cache_dir():
    try:
        from radiance.model.rudra_genfill import resolve_package
        pkg = resolve_package("")
        return os.path.join(str(pkg), "prompt_cache") if pkg else None
    except Exception:
        return None


class RadianceRudraFillPrompt:
    """RUDRA's prompt presets and its suggestion for the AI fill's highlight and shadow prompts."""

    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    DESCRIPTION = (
        "Prompts for SDR → HDR AI Fill. 'Suggest' looks at the frames' fully clipped areas with no model: "
        "a large one at the top is a sky (blue, overcast or sunset by the colour around it), a small round "
        "one the sun or a lamp, a bright rectangle in a dark surround a window, many specks glints, a warm "
        "area that changes size from frame to frame a flame; with fill_shadows, crushed blacks ask for "
        "shadow detail. Every suggestion says why. Or pick one of RUDRA's nine presets. Presets already "
        "embedded in the AI fill's prompt cache run without loading the 11 GB text encoder."
    )
    FUNCTION = "prompt"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt_highlight", "prompt_shadow", "report")
    OUTPUT_TOOLTIPS = ("For AI Fill's prompt_highlight.", "For AI Fill's prompt_shadow (empty unless asked).",
                       "What was found and why each prompt was picked.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The SDR frames the fill will work on (a few frames of a shot let "
                                               "a flame show itself)."}),
                "highlight": (_PRESET_CHOICES, {"default": _PRESET_CHOICES[0],
                    "tooltip": "The highlight prompt: RUDRA's suggestion, a preset, or none."}),
                "shadow": (["Suggest from the frames", "Shadow detail (shadow)", "None"], {"default": "None",
                    "tooltip": "The shadow prompt, for AI Fill with fill_shadows on."}),
            },
            "optional": {
                "highlight_context_code": ("INT", {"default": 250, "min": 200, "max": 256,
                    "tooltip": "AI Fill's highlight_context_code (tau_high): clipped is code >= max(this, 254)."}),
                "extra": ("STRING", {"default": "", "multiline": True,
                    "tooltip": "Added after the highlight prompt (a detail of your own: 'with a plane trail')."}),
            },
        }

    @torch.no_grad()
    def prompt(self, image, highlight, shadow, highlight_context_code=250, extra=""):
        img = image if image.dim() == 4 else image.unsqueeze(0)
        step = max(1, img.shape[0] // 8)            # at most 8 frames looked at, spread over the batch
        frames = [np.rint(np.clip(img[i, ..., :3].float().cpu().numpy(), 0, 1) * 255).astype(np.uint8)
                  for i in range(0, img.shape[0], step)][:8]
        want_shadow = shadow == "Suggest from the frames"
        res = suggest(frames, int(highlight_context_code), 4 if want_shadow else -1)
        ranked = res["suggestions"]
        lines = [f"clipped: {100 * res['clipped_fraction']:.2f} % of the frame ({len(frames)} frames looked at)"]
        for s in ranked:
            lines.append(f"  {s['label']}: {s['reason']}")
        if not ranked:
            lines.append("  nothing fully clipped: the fill has nothing to fill here")
        hi = ""
        if highlight == "Suggest from the frames":
            pick = next((s for s in ranked if s["region"] == "highlight"), None)
            hi = pick["text"] if pick else ""
            lines.append(f"highlight: {pick['label'] if pick else 'none (nothing clipped)'}")
        elif highlight in _CHOICE_ID:
            hi = BY_ID[_CHOICE_ID[highlight]].text
            lines.append(f"highlight: preset {highlight}")
        sh = ""
        if want_shadow:
            pick = next((s for s in ranked if s["region"] == "shadow"), None)
            sh = pick["text"] if pick else ""
            lines.append(f"shadow: {pick['label'] if pick else 'none (no crushed blacks)'}")
        elif shadow == "Shadow detail (shadow)":
            sh = BY_ID["shadow_detail"].text
        extra = (extra or "").strip()
        if extra and hi:
            hi = f"{hi}, {extra}"
        cache = _prompt_cache_dir()
        if cache:
            for label, text in (("highlight", hi), ("shadow", sh)):
                if text:
                    lines.append(f"{label} prompt {'is in' if cached(text, cache) else 'is not in'} the prompt cache"
                                 + ("" if cached(text, cache) else " (the fill embeds it once with umT5)"))
        return (hi, sh, "\n".join(lines))


NODE_CLASS_MAPPINGS = {
    "RadianceRudraLoadStill": RadianceRudraLoadStill,
    "RadianceRudraFillPrompt": RadianceRudraFillPrompt,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceRudraLoadStill": "◎ Load Still (RUDRA)",
    "RadianceRudraFillPrompt": "◎ RUDRA Fill Prompt",
}
