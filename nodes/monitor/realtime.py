"""
nodes_realtime_preview.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Radiance v3 — Pillar 05 · Real-Time Preview

Nodes
─────
  RadianceFalseColor      Exposure false-color overlay (cinema / broadcast style)
  RadianceFocusPeaking    Highlight in-focus regions with a Sobel edge overlay
  RadianceSplitView       A/B side-by-side or wipe comparison of two images
  RadianceContactSheet    Thumbnail grid from an IMAGE batch
  RadianceFlipbookGIF     Animated GIF export from a batch
  RadianceFrameStamp      Burn timecode, frame number, and custom text into frames
  RadiancePreviewServer   HTTP server — serve latest frame as JPEG to a browser

All nodes pass the original IMAGE through unchanged on their first output so
they can sit inline in any pipeline without breaking the graph.  The second
output is always the visualization (or status string for the server node).
"""

from __future__ import annotations

import io
import json
import logging
import math
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from radiance.color_utils import (
    luma_bt709       as _luma,
    tensor_to_numpy  as _to_numpy,
    numpy_to_tensor  as _to_tensor,
)
from radiance.path_utils import resolve_output_path, strip_path_quotes
from radiance.color import transfer

log = logging.getLogger("radiance.preview")

# ── PIL (required for GIF, JPEG, text rendering) ─────────────────────────────
try:
    from PIL import Image as _PilImage, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    log.warning("nodes_realtime_preview: Pillow not found — GIF/JPEG/text nodes degraded")


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers imported from color_utils:
#   _luma, _to_numpy, _to_tensor

def _to_batch_numpy(t: torch.Tensor) -> np.ndarray:
    """(B,H,W,3) tensor → float32 numpy, all frames."""
    arr = t.detach().cpu().float().numpy()
    return arr if arr.ndim == 4 else arr[np.newaxis]


def _sobel_mag(gray: np.ndarray) -> np.ndarray:
    """Approximate Sobel magnitude via finite differences. (H,W) → (H,W)."""
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    return np.sqrt(gx ** 2 + gy ** 2)


# ═════════════════════════════════════════════════════════════════════════════
# NODE 1 — RadianceFalseColor
# ═════════════════════════════════════════════════════════════════════════════

# Cinema-style false color exposure zones (luma value → display color)
# Ranges use linear [0, 1] display-referred luminance.
# Colors are (R, G, B) float32.
_FALSE_COLOR_ZONES = [
    # (luma_lo, luma_hi, color_rgb, label)
    (0.000, 0.010, (0.0,  0.0,  0.0),  "Black clip"),
    (0.010, 0.040, (0.13, 0.0,  0.35), "Deep shadow"),
    (0.040, 0.100, (0.0,  0.0,  0.80), "Shadow"),
    (0.100, 0.165, (0.0,  0.55, 0.85), "Low midtone"),
    # 18% grey ± ~0.5 stop — the "good exposure" indicator zone
    (0.165, 0.235, (0.0,  0.85, 0.15), "18% grey (correct)"),
    (0.235, 0.700, None,               "Normal (passthrough)"),
    (0.700, 0.800, (0.95, 0.85, 0.0),  "Upper highlight"),
    (0.800, 0.900, (1.0,  0.45, 0.0),  "Highlight warning"),
    (0.900, 0.990, (0.9,  0.15, 0.0),  "Near clip"),
    (0.990, 9999,  (1.0,  1.0,  1.0),  "White clip (zebra)"),
]


def _apply_false_color(
    rgb: np.ndarray,
    strength: float = 1.0,
    hdr_peak: float = 1.0,
    input_encoding: str = "Linear Rec.709",
) -> np.ndarray:
    """
    Decode the selected transfer and map linear luminance to false-color zones.

    Parameters
    ──────────
    rgb      : (H, W, 3/4), Rec.709 primaries; alpha is preserved
    strength : blend factor (0 = passthrough, 1 = full false color)
    hdr_peak : normalize by this before zone lookup (1.0 for SDR)
    input_encoding : Linear Rec.709, sRGB, or the Rec.709 camera OETF
    """
    if hdr_peak <= 0:
        raise ValueError("False Color hdr_peak must be positive")
    source = rgb[..., :3]
    if input_encoding == "Linear Rec.709":
        linear = source
    elif input_encoding == "sRGB":
        linear = transfer.srgb_to_linear(source)
    elif input_encoding == "Rec.709":
        linear = transfer.rec709_to_linear(source)
    else:
        raise ValueError(f"Unknown false-colour input encoding: {input_encoding}")
    if strength == 0:
        return rgb.copy()
    luma = np.clip(_luma(linear) / hdr_peak, 0.0, None)
    out = source.copy()

    for lo, hi, color, _ in _FALSE_COLOR_ZONES:
        if color is None:
            continue  # passthrough zone — keep original
        mask = (luma >= lo) & (luma < hi)
        if not mask.any():
            continue
        c = np.array(color, dtype=np.float32)
        out[mask] = (1.0 - strength) * source[mask] + strength * c

    # Zebra stripe pattern for clipped highlights (alternate columns)
    clip_mask = luma >= 0.990
    if clip_mask.any():
        h, w = clip_mask.shape
        stripe = (np.arange(w) // 8 % 2).astype(bool)
        stripe_mask = clip_mask & stripe[np.newaxis, :]
        out[stripe_mask] = (1.0 - strength) * source[stripe_mask]

    result = rgb.copy()
    result[..., :3] = np.clip(out, 0.0, 1.0)
    return result


class RadianceFalseColorMonitor:
    """
    Cinema / broadcast-style exposure false color overlay.

    Maps display-referred luminance zones to distinctive colours so
    colorists can instantly identify correct exposure, crushed blacks,
    and clipping without reading numbers.

    Zone legend (display-referred luma)
    ────────────────────────────────────
      Black  < 1%      Absolute black / signal loss
      Purple 1–4%      Deep shadow / pedestal
      Blue   4–10%     Shadow detail
      Teal   10–16.5%  Low midtone
      Green  16.5–23.5% ◀ 18% grey — correct exposure indicator
      (none) 23.5–70%  Normal tones (pass-through)
      Yellow 70–80%    Bright — watch headroom
      Orange 80–90%    Highlight warning
      Red    90–99%    Near clip — take action
      Zebra  ≥ 99%     Clipped / blown — alternating black stripes
    """

    CATEGORY = "FXTD STUDIOS/Radiance/◎ QC & Debug"
    DESCRIPTION = "Display a false-colour exposure monitor for on-set or grading use."
    FUNCTION     = "apply"
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("passthrough", "false_color")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Rec.709-primary image to check, returned unchanged on passthrough. "
                               "Choose its input encoding below; zones measure linear luminance. "
                               "Convert other primaries into Rec.709 before connecting."}),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Blend factor. 0 = off, 1 = full false color.",
                }),
            },
            "optional": {
                "hdr_peak": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 100.0, "step": 0.1,
                    "tooltip": "Luma is divided by this before the zone lookup (1.0 for SDR). For "
                               "scene-linear HDR at 1.0 = 203 nits, a 1000-nit peak is about 4.93.",
                }),
                "input_encoding": (["Linear Rec.709", "sRGB", "Rec.709"], {
                    "default": "Linear Rec.709",
                    "tooltip": "Transfer encoding of the input. sRGB and Rec.709 are decoded before "
                               "measuring luminance; Linear Rec.709 preserves legacy behaviour. "
                               "The false-colour output is an SDR diagnostic image, not HDR mastering.",
                }),
            },
        }

    def apply(self, image: torch.Tensor, strength: float, hdr_peak: float = 1.0,
              input_encoding: str = "Linear Rec.709"):
        frames = _to_batch_numpy(image)
        fc_frames = np.zeros_like(frames)

        for b in range(frames.shape[0]):
            fc_frames[b] = _apply_false_color(frames[b], strength, hdr_peak, input_encoding)

        log.debug("FalseColor: %d frame(s), strength=%.2f, peak=%.1f", frames.shape[0], strength, hdr_peak)
        return (image, _to_tensor(fc_frames))


# ═════════════════════════════════════════════════════════════════════════════
# NODE 2 — RadianceFocusPeaking
# ═════════════════════════════════════════════════════════════════════════════

def _focus_peak(
    rgb:       np.ndarray,
    threshold: float,
    color:     tuple[float, float, float],
    strength:  float,
) -> np.ndarray:
    """Overlay focus peaking on (H,W,3) image."""
    gray     = _luma(rgb)
    mag      = _sobel_mag(gray)
    # Normalise magnitude to [0,1]
    mag_max  = mag.max()
    if mag_max > 1e-6:
        mag = mag / mag_max

    in_focus = mag >= threshold   # (H, W) bool mask
    out = rgb.copy()
    c   = np.array(color, dtype=np.float32)
    out[in_focus] = (1.0 - strength) * rgb[in_focus] + strength * c
    return np.clip(out, 0.0, 1.0)


class RadianceFocusPeaking:
    """
    Focus peaking monitor — highlight in-focus (sharp) regions with a
    coloured overlay using Sobel edge magnitude.

    Higher Sobel magnitude → sharper transition → in focus.
    Pixels above the threshold are coloured with the peaking colour;
    soft / out-of-focus areas keep their original appearance.

    Typical settings
    ────────────────
      threshold 0.15–0.25  : moderate sensitivity (most footage)
      threshold 0.05–0.10  : very sensitive (catches micro-focus variations)
      threshold 0.40+      : only hard edges (fast lens, close subject)
    """

    CATEGORY = "FXTD STUDIOS/Radiance/◎ QC & Debug"
    DESCRIPTION = "Overlay focus-peaking highlights on edges for sharpness assessment."
    FUNCTION     = "peak"
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("passthrough", "focus_peak")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Image to check, returned unchanged on passthrough. Edges are measured on "
                               "BT.709 luma, normalised to the strongest edge in each frame."}),
                "threshold": ("FLOAT", {
                    "default": 0.20, "min": 0.01, "max": 1.0, "step": 0.01,
                    "tooltip": "Normalised Sobel magnitude above which a pixel is considered in-focus.",
                }),
                "peak_color": (["Red", "Green", "White", "Yellow", "Cyan"], {
                    "default": "Red",
                    "tooltip": "Overlay colour painted on pixels above threshold.",
                }),
                "strength": ("FLOAT", {
                    "default": 0.85, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Blend factor for the peaking overlay.",
                }),
            },
        }

    _COLORS = {
        "Red":    (1.0, 0.0, 0.0),
        "Green":  (0.0, 1.0, 0.0),
        "White":  (1.0, 1.0, 1.0),
        "Yellow": (1.0, 1.0, 0.0),
        "Cyan":   (0.0, 1.0, 1.0),
    }

    def peak(self, image: torch.Tensor, threshold: float, peak_color: str, strength: float):
        frames = _to_batch_numpy(image)
        pk_frames = np.zeros_like(frames)
        color = self._COLORS.get(peak_color, (1.0, 0.0, 0.0))

        for b in range(frames.shape[0]):
            pk_frames[b] = _focus_peak(frames[b], threshold, color, strength)

        log.debug("FocusPeaking: %d frame(s), thr=%.2f, color=%s", frames.shape[0], threshold, peak_color)
        return (image, _to_tensor(pk_frames))


# ═════════════════════════════════════════════════════════════════════════════
# NODE 3 — RadianceSplitView
# ═════════════════════════════════════════════════════════════════════════════

def _split_view(
    a:        np.ndarray,
    b:        np.ndarray,
    mode:     str,
    position: float,
) -> np.ndarray:
    """
    Combine two (H,W,3) images into a comparison view.

    Modes
    ─────
    side_by_side : left half from A, right half from B
    wipe_h       : horizontal wipe — position controls the split column
    wipe_v       : vertical wipe — position controls the split row
    diff         : abs(A - B) × 4 amplified difference
    """
    H, W, _ = a.shape
    # Resize b to match a if needed
    if b.shape[:2] != (H, W):
        if HAS_PIL:
            b_pil = _PilImage.fromarray((np.clip(b, 0, 1) * 255).astype(np.uint8))
            b_pil = b_pil.resize((W, H), _PilImage.LANCZOS)
            b = np.array(b_pil, dtype=np.float32) / 255.0
        else:
            b = b[:H, :W]   # rough crop

    if mode == "side_by_side":
        half = W // 2
        out = np.concatenate([a[:, :half], b[:, half:]], axis=1)

    elif mode == "wipe_h":
        split_col = int(np.clip(position, 0.0, 1.0) * W)
        out = a.copy()
        out[:, split_col:] = b[:, split_col:]
        # Draw a white guide line
        if 0 < split_col < W:
            out[:, split_col - 1: split_col + 1] = [1.0, 1.0, 1.0]

    elif mode == "wipe_v":
        split_row = int(np.clip(position, 0.0, 1.0) * H)
        out = a.copy()
        out[split_row:] = b[split_row:]
        if 0 < split_row < H:
            out[split_row - 1: split_row + 1] = [1.0, 1.0, 1.0]

    elif mode == "diff":
        diff = np.abs(a.astype(np.float32) - b.astype(np.float32)) * 4.0
        out  = np.clip(diff, 0.0, 1.0)

    else:
        out = a

    return out


class RadianceSplitView:
    """
    A/B comparison viewer.

    Wire the pre-grade image into image_a and the post-grade result into
    image_b. Choose wipe mode and adjust position to compare.

    Modes
    ─────
    side_by_side : left = A, right = B (position unused)
    wipe_h       : horizontal split — slider controls split column
    wipe_v       : vertical split   — slider controls split row
    diff         : 4× amplified absolute difference (great for catching subtle changes)
    """

    CATEGORY = "FXTD STUDIOS/Radiance/◎ QC & Debug"
    DESCRIPTION = "Side-by-side or wipe comparison between two images or versions."
    FUNCTION     = "compare"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("comparison",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE", {"tooltip": "Original / reference image."}),
                "image_b": ("IMAGE", {"tooltip": "Processed / graded image."}),
                "mode": (["wipe_h", "wipe_v", "side_by_side", "diff"], {
                    "default": "wipe_h",
                    "tooltip": "wipe_h: A left, B right of a vertical line. wipe_v: A above, B below. "
                               "side_by_side: left half of A beside right half of B (fixed 50% split). "
                               "diff: |A - B| x 4, clipped to [0, 1].",
                }),
                "position": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Wipe line position as a fraction of width (wipe_h) or height (wipe_v): "
                               "0 = all B, 1 = all A. Unused in side_by_side and diff.",
                }),
            },
        }

    def compare(self, image_a: torch.Tensor, image_b: torch.Tensor, mode: str, position: float):
        batch_a = _to_batch_numpy(image_a)
        batch_b = _to_batch_numpy(image_b)
        n = max(batch_a.shape[0], batch_b.shape[0])

        out = np.zeros((n, *batch_a.shape[1:]), dtype=np.float32)
        for i in range(n):
            a = batch_a[min(i, batch_a.shape[0] - 1)]
            b = batch_b[min(i, batch_b.shape[0] - 1)]
            out[i] = _split_view(a, b, mode, position)

        log.debug("SplitView: %d frame(s), mode=%s", n, mode)
        return (_to_tensor(out),)


# ═════════════════════════════════════════════════════════════════════════════
# NODE 4 — RadianceContactSheet
# ═════════════════════════════════════════════════════════════════════════════

class RadianceContactSheet:
    """
    Generate a thumbnail contact sheet from an IMAGE batch.

    Arranges all frames in a grid, optionally labelling each cell with its
    frame index.  Useful for at-a-glance review of a rendered sequence before
    sending it to a compositor or client.
    """

    CATEGORY = "FXTD STUDIOS/Radiance/◎ QC & Debug"
    DESCRIPTION = "Render a contact sheet grid of multiple images for review."
    FUNCTION     = "sheet"
    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("contact_sheet", "grid_cols", "grid_rows")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Frame batch to lay out, left to right then top to bottom. Clamped to "
                               "[0, 1] and resized through 8-bit, so display-encoded input is expected."}),
                "thumb_width": ("INT", {
                    "default": 160, "min": 32, "max": 512, "step": 8,
                    "tooltip": "Width of each thumbnail in pixels.",
                }),
                "max_cols": ("INT", {
                    "default": 8, "min": 1, "max": 32,
                    "tooltip": "Maximum number of columns. Rows are computed automatically.",
                }),
            },
            "optional": {
                "label_frames": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Print the frame index below each thumbnail.",
                }),
                "background": (["Black", "Grey", "White"], {"default": "Black",
                    "tooltip": "Fill colour for empty cells and label strips (Grey is 64/255)."}),
            },
        }

    _BG = {"Black": (0, 0, 0), "Grey": (64, 64, 64), "White": (255, 255, 255)}

    def sheet(
        self,
        images: torch.Tensor,
        thumb_width: int,
        max_cols: int,
        label_frames: bool = True,
        background: str = "Black",
    ):
        frames = _to_batch_numpy(images)          # (B, H, W, 3)
        B, H, W, _ = frames.shape

        aspect = H / max(W, 1)
        th = int(thumb_width * aspect)
        tw = thumb_width
        label_h = 14 if label_frames else 0

        cols = min(B, max_cols)
        rows = math.ceil(B / cols)

        bg = self._BG.get(background, (0, 0, 0))
        sheet = np.full((rows * (th + label_h), cols * tw, 3),
                        [v / 255.0 for v in bg], dtype=np.float32)

        for i, frame in enumerate(frames):
            r, c = divmod(i, cols)
            y0 = r * (th + label_h)
            x0 = c * tw

            # Resize thumbnail via PIL when available
            thumb_f32 = np.clip(frame, 0, 1)
            if HAS_PIL:
                pil_img = _PilImage.fromarray((thumb_f32 * 255).astype(np.uint8))
                pil_img = pil_img.resize((tw, th), _PilImage.LANCZOS)
                thumb_f32 = np.array(pil_img, dtype=np.float32) / 255.0
            else:
                # Simple crop + skip
                thumb_f32 = thumb_f32[:th, :tw]
                pad_h = th - thumb_f32.shape[0]
                pad_w = tw - thumb_f32.shape[1]
                if pad_h > 0 or pad_w > 0:
                    thumb_f32 = np.pad(thumb_f32, ((0, pad_h), (0, pad_w), (0, 0)))

            sheet[y0: y0 + th, x0: x0 + tw] = thumb_f32

            # Label
            if label_frames and HAS_PIL:
                lbl_arr = np.full((label_h, tw, 3), [v / 255.0 for v in bg], dtype=np.float32)
                lbl_pil = _PilImage.fromarray((lbl_arr * 255).astype(np.uint8))
                draw = ImageDraw.Draw(lbl_pil)
                try:
                    font = ImageFont.load_default()
                except Exception:
                    font = None
                draw.text((2, 1), f"#{i}", fill=(200, 200, 200), font=font)
                sheet[y0 + th: y0 + th + label_h, x0: x0 + tw] = \
                    np.array(lbl_pil, dtype=np.float32) / 255.0

        log.info("ContactSheet: %d frames -> %dx%d grid (%dpx thumbs)", B, cols, rows, tw)
        return (_to_tensor(sheet), cols, rows)


# ═════════════════════════════════════════════════════════════════════════════
# NODE 5 — RadianceFlipbookGIF
# ═════════════════════════════════════════════════════════════════════════════

class RadianceFlipbookGIF:
    """
    Export an IMAGE batch as an animated GIF for quick preview sharing.

    GIF is 8-bit palette-quantised, so it is not colour-accurate — it is
    purely for review and communication, not colour grading.  The node also
    returns a status string with the output path.
    """

    CATEGORY = "FXTD STUDIOS/Radiance/◎ QC & Debug"
    DESCRIPTION = "Export a sequence as an animated GIF flipbook for quick review."
    FUNCTION     = "export_gif"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("passthrough", "status")
    OUTPUT_NODE  = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Frames to animate, returned unchanged on passthrough. Clamped to [0, 1] "
                               "and quantised to a 256-colour palette, so display-encoded input is expected."}),
                "save_path": ("STRING", {
                    "default": "preview/flipbook.gif",
                    "tooltip": (
                        "Output .gif path. A relative path is written under ComfyUI's "
                        "output/ folder; absolute paths are used as given. The "
                        "directory is created automatically."
                    ),
                }),
                "fps": ("FLOAT", {
                    "default": 12.0, "min": 1.0, "max": 60.0, "step": 0.5,
                    "tooltip": "Playback speed.  GIF frame delay = 1000/fps ms.",
                }),
                "max_width": ("INT", {
                    "default": 480, "min": 64, "max": 1920, "step": 8,
                    "tooltip": "Resize frames to this width (preserves aspect ratio). "
                               "Smaller = smaller file.",
                }),
            },
            "optional": {
                "loop": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Loop the animation indefinitely.",
                }),
                "dither": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable Floyd-Steinberg dithering for smoother gradients.",
                }),
            },
        }

    def export_gif(
        self,
        images:    torch.Tensor,
        save_path: str,
        fps:       float,
        max_width: int,
        loop:      bool = True,
        dither:    bool = True,
    ):
        if not HAS_PIL:
            log.error("FlipbookGIF: Pillow not installed — cannot write GIF")
            return (images, "ERROR: Pillow not installed")

        frames = _to_batch_numpy(images)  # (B, H, W, 3)
        B, H, W, _ = frames.shape

        # Compute resize dimensions
        if W > max_width:
            scale = max_width / W
            new_w = max_width
            new_h = int(H * scale)
        else:
            new_w, new_h = W, H

        delay_ms = int(1000 / max(fps, 0.1))
        dith_mode = _PilImage.Dither.FLOYDSTEINBERG if dither else _PilImage.Dither.NONE

        pil_frames: list[_PilImage.Image] = []
        for frame in frames:
            arr = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
            pil_img = _PilImage.fromarray(arr, mode="RGB")
            if new_w != W or new_h != H:
                pil_img = pil_img.resize((new_w, new_h), _PilImage.LANCZOS)
            pil_frames.append(pil_img.convert("P", dither=dith_mode))

        path = Path(resolve_output_path(save_path))
        if not path.suffix:
            path = path.with_suffix(".gif")
        path.parent.mkdir(parents=True, exist_ok=True)

        pil_frames[0].save(
            str(path),
            format      = "GIF",
            save_all    = True,
            append_images = pil_frames[1:],
            duration    = delay_ms,
            loop        = 0 if loop else 1,
            optimize    = False,
        )

        size_kb = path.stat().st_size // 1024
        status = (
            f"GIF written: {path} | {B} frames @ {fps:.1f} fps | "
            f"{new_w}×{new_h}px | {size_kb} KB"
        )
        log.info(status)
        return (images, status)


# ═════════════════════════════════════════════════════════════════════════════
# NODE 6 — RadianceFrameStamp
# ═════════════════════════════════════════════════════════════════════════════

class RadianceFrameStamp:
    """
    Burn timecode, frame number, and custom text into frames.

    Timecode formats
    ────────────────
    Non-drop (NDF): HH:MM:SS:FF  — exact integer fps (24, 25, 30, 48, 50, 60).
    Drop-frame (DF): HH:MM:SS;FF — SMPTE 12M for 29.97 and 59.94 fps.
        Drops frame numbers 00 and 01 at the start of every minute, except
        every 10th minute, to keep the TC clock aligned with wall-clock time.

    The stamp is rendered as white text with a semi-transparent black drop-shadow
    for legibility on both bright and dark backgrounds.

    Text placement options
    ──────────────────────
    top_left | top_right | bottom_left | bottom_right | center
    """

    # Drop-frame applies at these nominal rates (29.97 ≈ 30000/1001, 59.94 ≈ 60000/1001)
    _DF_RATES = {30: 2, 60: 4}   # nominal_fps → drop_frames_per_minute

    CATEGORY = "FXTD STUDIOS/Radiance/◎ QC & Debug"
    DESCRIPTION = "Burn frame number, timecode, and metadata into an image for dailies."
    FUNCTION     = "stamp"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("stamped",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Float frames to stamp. Values outside the overlay and the source alpha "
                               "are preserved, including negative and HDR values. Text RGB is 230/255 "
                               "in the input encoding; no colour transform is applied."}),
                "start_frame": ("INT", {
                    "default": 1001, "min": 0, "max": 999999,
                    "tooltip": "Frame number of the first frame in the batch.",
                }),
                "fps": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 120.0, "step": 0.001,
                    "tooltip": "Frames per second (used for timecode calculation).",
                }),
            },
            "optional": {
                "drop_frame": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Use SMPTE drop-frame timecode (DF).  "
                        "Only meaningful at 29.97 or 59.94 fps.  "
                        "Uses ';' separator instead of ':' for the frame field."
                    ),
                }),
                "show_frame_number": ("BOOLEAN", {"default": True,
                    "tooltip": "Burn the frame number as '# 001001' (6 digits)."}),
                "show_timecode":     ("BOOLEAN", {"default": True,
                    "tooltip": "Burn the SMPTE timecode of the frame number at fps. "
                               "Frame 0 is 00:00:00:00."}),
                "custom_text": ("STRING", {
                    "default": "",
                    "tooltip": "Additional text burned into each frame (e.g. shot name, version).",
                }),
                "position": (["bottom_left", "bottom_right", "top_left", "top_right", "center"], {
                    "default": "bottom_left",
                    "tooltip": "Where the text block sits in the frame.",
                }),
                "font_scale": ("FLOAT", {
                    "default": 1.0, "min": 0.5, "max": 4.0, "step": 0.1,
                    "tooltip": "Text size: 14 px x scale (minimum 12 px) with DejaVu Sans Mono. Without "
                               "that font, Pillow's fixed default font is used and only the margins scale.",
                }),
                "opacity": ("FLOAT", {
                    "default": 0.85, "min": 0.1, "max": 1.0, "step": 0.05,
                    "tooltip": "Text opacity; the dark backing box uses 0.6 x this.",
                }),
            },
        }

    @staticmethod
    def _frame_to_tc(frame: int, fps: float, drop_frame: bool = False) -> str:
        """
        Convert an absolute frame number to a SMPTE timecode string.

        Non-drop  → HH:MM:SS:FF
        Drop-frame → HH:MM:SS;FF  (29.97 / 59.94 only)

        Drop-frame algorithm (SMPTE 12M):
          nominal_fps ∈ {30, 60}; drop_count ∈ {2, 4}
          d  = drop_count per non-10th minute
          D  = frames_per_10min  = nominal_fps*600 - 9*d
          hh = frame_count // (6*D)
          remaining after extracting hours
          mm = remaining // D (then sub-minute arithmetic)
        """
        fps_int = max(1, round(fps))

        if drop_frame and fps_int in RadianceFrameStamp._DF_RATES:
            d = RadianceFrameStamp._DF_RATES[fps_int]   # frames dropped per non-10th minute
            frames_per_10min = fps_int * 600 - 9 * d    # 17982 @ 30 DF, 35964 @ 60 DF

            # Extract hours first (each hour = 6 × 10-min blocks)
            frames_per_hour = 6 * frames_per_10min
            hh = frame // frames_per_hour
            frame %= frames_per_hour

            # Extract 10-minute blocks
            ten_min = frame // frames_per_10min
            frame %= frames_per_10min

            # Within a 10-minute block the first minute is "full" (no drop),
            # subsequent minutes each lose d frames at their start.
            frames_per_min_full  = fps_int * 60                  # first minute of each 10
            frames_per_min_short = fps_int * 60 - d              # remaining 9 minutes

            if frame < frames_per_min_full:
                # First minute of the 10-min block — no drop
                mm_sub = 0
                ff = frame % fps_int
                ss = frame // fps_int
            else:
                frame -= frames_per_min_full
                mm_sub = frame // frames_per_min_short + 1   # +1 because minute 0 already consumed
                frame %= frames_per_min_short
                # Re-add the dropped frames at the start of this minute
                frame += d
                ff = frame % fps_int
                ss = frame // fps_int

            mm = ten_min * 10 + mm_sub
            return f"{hh:02d}:{mm:02d}:{ss:02d};{ff:02d}"

        # ── Non-drop (NDF) ────────────────────────────────────────────────────
        ff = frame % fps_int
        secs = frame // fps_int
        ss = secs % 60
        mm = (secs // 60) % 60
        hh = secs // 3600
        return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"

    def stamp(
        self,
        images:             torch.Tensor,
        start_frame:        int,
        fps:                float,
        drop_frame:         bool  = False,
        show_frame_number:  bool  = True,
        show_timecode:      bool  = True,
        custom_text:        str   = "",
        position:           str   = "bottom_left",
        font_scale:         float = 1.0,
        opacity:            float = 0.85,
    ):
        frames = _to_batch_numpy(images)
        B, H, W, _ = frames.shape
        out = frames.copy()

        if not HAS_PIL:
            log.warning("FrameStamp: Pillow not installed — returning unstamped frames")
            return (_to_tensor(out),)

        try:
            font_size = max(12, int(14 * font_scale))
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", font_size)
            except Exception:
                font = ImageFont.load_default()
        except Exception:
            font = None

        for b in range(B):
            frame_num = start_frame + b
            lines: list[str] = []
            if show_frame_number:
                lines.append(f"# {frame_num:06d}")
            if show_timecode:
                lines.append(self._frame_to_tc(frame_num, fps, drop_frame=drop_frame))
            if custom_text.strip():
                lines.append(custom_text.strip())

            if not lines:
                continue

            # Rasterise only the overlay. The source stays float, including
            # negative values, HDR highlights and its original alpha channel.
            pil_img = _PilImage.new("RGBA", (W, H), (0, 0, 0, 0))
            draw = ImageDraw.Draw(pil_img, "RGBA")

            text = "\n".join(lines)
            margin = int(8 * font_scale)
            line_h = int(font_size * 1.25)
            text_h = len(lines) * line_h
            bbox_w = max(len(l) for l in lines) * int(font_size * 0.62)

            if "bottom" in position:
                ty = H - text_h - margin * 2
            elif "center" in position:
                ty = (H - text_h) // 2
            else:
                ty = margin

            if "right" in position:
                tx = W - bbox_w - margin
            elif "center" in position:
                tx = (W - bbox_w) // 2
            else:
                tx = margin

            # Draw semi-transparent background box
            pad = 3
            bg_alpha = int(opacity * 0.6 * 255)
            draw.rectangle(
                [tx - pad, ty - pad, tx + bbox_w + pad, ty + text_h + pad],
                fill=(0, 0, 0, bg_alpha),
            )

            # Draw shadow then text
            text_alpha = int(opacity * 255)
            for i, line in enumerate(lines):
                y = ty + i * line_h
                draw.text((tx + 1, y + 1), line, fill=(0, 0, 0, text_alpha // 2), font=font)
                draw.text((tx, y), line,     fill=(230, 230, 230, text_alpha),   font=font)

            overlay = np.asarray(pil_img, dtype=np.float32) / 255.0
            coverage = overlay[..., 3:4]
            out[b, ..., :3] = frames[b, ..., :3] * (1.0 - coverage) + overlay[..., :3] * coverage

        log.debug("FrameStamp: %d frame(s) stamped from #%d", B, start_frame)
        return (_to_tensor(out),)


# ═════════════════════════════════════════════════════════════════════════════
# NODE 7 — RadiancePreviewServer
# ═════════════════════════════════════════════════════════════════════════════

# Global frame buffer — shared between node instances and the HTTP handler
_PREVIEW_BUFFER: dict[str, bytes] = {}    # stream_name → JPEG bytes
_PREVIEW_META:   dict[str, dict] = {}     # stream_name → metadata dict
_SERVERS:        dict[int, HTTPServer] = {}  # port -> running HTTPServer
_SERVER_LOCK     = threading.Lock()

_HTML_PAGE = """\
<!DOCTYPE html><html><head>
<title>Radiance Preview — {name}</title>
<meta charset="utf-8"/>
<meta http-equiv="refresh" content="{interval}"/>
<style>
  body {{ margin:0; background:#111; display:flex; flex-direction:column;
         align-items:center; font-family:monospace; color:#aaa; }}
  img {{ max-width:100vw; max-height:90vh; margin-top:8px; border:1px solid #333; }}
  .info {{ font-size:11px; padding:4px 0; }}
</style></head><body>
<img src="/frame/{name}" alt="frame"/>
<div class="info">{name} — {meta} — auto-refresh {interval}s</div>
</body></html>
"""


def _make_handler(name_filter: str):
    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            log.debug("PreviewServer: %s %s", self.command, self.path)

        def do_GET(self):
            path = self.path.rstrip("/")

            if path in ("", "/", "/index"):
                # Serve the HTML auto-refresh page
                names = list(_PREVIEW_BUFFER.keys()) or [name_filter]
                tgt = name_filter if name_filter in names else (names[0] if names else "default")
                meta = _PREVIEW_META.get(tgt, {})
                meta_str = " | ".join(f"{k}:{v}" for k, v in meta.items())
                body = _HTML_PAGE.format(name=tgt, meta=meta_str, interval=1).encode()
                self._respond(200, "text/html", body)

            elif path.startswith("/frame/"):
                # Serve the JPEG for a named stream
                sname = path[len("/frame/"):]
                jpeg  = _PREVIEW_BUFFER.get(sname) or _PREVIEW_BUFFER.get(name_filter)
                if jpeg:
                    self._respond(200, "image/jpeg", jpeg)
                else:
                    self._respond(204, "text/plain", b"No frame yet")

            elif path == "/health":
                self._respond(200, "application/json",
                              json.dumps({"status": "ok",
                                          "streams": list(_PREVIEW_BUFFER)}).encode())
            else:
                self._respond(404, "text/plain", b"Not found")

        def _respond(self, code: int, ct: str, body: bytes):
            self.send_response(code)
            self.send_header("Content-Type", ct)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    return _Handler


def _ensure_server(port: int, stream_name: str) -> Optional[str]:
    """Start an HTTP server on `port` if one isn't already running.

    Returns None on success, or the bind error as a string.

    BUG-FIX: this used to catch OSError, log it, and return None either way,
    and `serve()` then unconditionally returned "http://localhost:<port>/" on
    its server_url output. A port already in use therefore handed the user a
    working-looking URL for a server that was never listening, or, worse,
    for somebody else's server on that port. The caller now gets the failure
    and says so on the output socket.
    """
    with _SERVER_LOCK:
        if port in _SERVERS:
            return None
        try:
            handler  = _make_handler(stream_name)
            server   = HTTPServer(("127.0.0.1", port), handler)
            thread   = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            _SERVERS[port] = server
            log.info("PreviewServer started on http://127.0.0.1:%d", port)
            return None
        except OSError as exc:
            log.error("PreviewServer: cannot bind port %d — %s", port, exc)
            return f"cannot bind port {port}: {exc}"


def _shutdown_servers(keep_port: Optional[int] = None) -> int:
    """Stop and drop every preview server except `keep_port`.

    BUG-FIX: _SERVERS was never pruned and nothing ever called shutdown(), so
    changing the port widget started a second HTTPServer and leaked the first
    one, its serve_forever thread and its listening socket, for the lifetime of
    the ComfyUI process. Re-queueing after a few port edits left a handful of
    stale servers still answering on old ports with stale frames.
    """
    stopped = 0
    with _SERVER_LOCK:
        for old_port in [p for p in _SERVERS if p != keep_port]:
            server = _SERVERS.pop(old_port, None)
            if server is None:
                continue
            try:
                server.shutdown()
                server.server_close()
                stopped += 1
                log.info("PreviewServer on port %d stopped", old_port)
            except Exception as exc:
                log.warning("PreviewServer: could not stop port %d, %s", old_port, exc)
    return stopped


def _frame_to_jpeg(arr: np.ndarray, quality: int = 85) -> bytes:
    """Convert (H,W,3) float32 [0,1] to JPEG bytes."""
    if not HAS_PIL:
        return b""
    u8  = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    pil = _PilImage.fromarray(u8)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


class RadiancePreviewServer:
    """
    HTTP preview server — serve the most recent processed frame as JPEG to
    any browser or monitoring tool on the local workstation.

    After the node executes, open:
        http://localhost:<port>/

    The page auto-refreshes each second.  Multiple named streams can run on
    the same port (access each at /frame/<stream_name>).

    The HTTP server runs in a background daemon thread and survives between
    ComfyUI graph executions — it only needs to start once per port.
    """

    CATEGORY = "FXTD STUDIOS/Radiance/◎ QC & Debug"
    DESCRIPTION = "Run a local HTTP preview server for browser-based image review."
    FUNCTION     = "serve"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("passthrough", "server_url")
    OUTPUT_NODE  = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Frames to publish, returned unchanged on passthrough. Only the last frame "
                               "is served, clamped to [0, 1] as an 8-bit JPEG."}),
                "port": ("INT", {
                    "default": 8765, "min": 1024, "max": 65535,
                    "tooltip": "TCP port for the preview HTTP server.",
                }),
                "stream_name": ("STRING", {
                    "default": "radiance",
                    "tooltip": "Stream identifier.  Access at /frame/<stream_name>.",
                }),
            },
            "optional": {
                "jpeg_quality": ("INT", {
                    "default": 85, "min": 20, "max": 99,
                    "tooltip": "JPEG compression quality (20=small, 99=lossless-ish).",
                }),
                "resize_width": ("INT", {
                    "default": 0, "min": 0, "max": 3840,
                    "tooltip": "Resize frame before serving (0 = original size). "
                               "Smaller = faster over network.",
                }),
                "enabled": ("BOOLEAN", {"default": True,
                    "tooltip": "Off: skip publishing this frame. A server already started keeps running "
                               "and serves the previous frame."}),
            },
        }

    def serve(
        self,
        images:      torch.Tensor,
        port:        int,
        stream_name: str,
        jpeg_quality: int = 85,
        resize_width: int = 0,
        enabled:     bool = True,
    ):
        if not enabled:
            return (images, "PreviewServer disabled")

        if not HAS_PIL:
            return (images, "ERROR: Pillow not installed — cannot serve JPEG")

        # Stop any server this node left running on a different port before
        # binding the new one, so editing the port widget does not leak the
        # previous HTTPServer and its thread.
        _shutdown_servers(keep_port=port)

        bind_error = _ensure_server(port, stream_name)
        if bind_error:
            # Do not hand back a URL for a server that is not listening.
            return (images, f"ERROR: PreviewServer not started, {bind_error}")

        frames = _to_batch_numpy(images)
        # Serve only the last frame of the batch (most recently processed)
        frame = frames[-1]

        # Optional resize
        if resize_width > 0 and frame.shape[1] != resize_width:
            H, W = frame.shape[:2]
            new_h = int(H * resize_width / W)
            pil_f = _PilImage.fromarray((np.clip(frame, 0, 1) * 255).astype(np.uint8))
            pil_f = pil_f.resize((resize_width, new_h), _PilImage.LANCZOS)
            frame = np.array(pil_f, dtype=np.float32) / 255.0

        jpeg = _frame_to_jpeg(frame, jpeg_quality)
        _PREVIEW_BUFFER[stream_name] = jpeg
        _PREVIEW_META[stream_name]   = {
            "frames": frames.shape[0],
            "size":   f"{frames.shape[2]}×{frames.shape[1]}",
        }

        url = f"http://localhost:{port}/"
        log.debug("PreviewServer: pushed %d B JPEG to '%s' at %s", len(jpeg), stream_name, url)
        return (images, url)


# ─────────────────────────────────────────────────────────────────────────────
# Registration
# ─────────────────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    # Nodes 1 and 3 of this file were finished, documented and covered by
    # tests/test_realtime_preview.py, and this dict never named them, so
    # ComfyUI never saw them -- the same defect that hid seventeen nodes in the
    # v3 reorganisation (see nodes/aggregate.py). The module mapping is what
    # `fold_in_module_nodes` sweeps, so a node added to this file only ships
    # once it appears here.
    "RadianceFalseColorMonitor": RadianceFalseColorMonitor,
    "RadianceFocusPeaking":  RadianceFocusPeaking,
    "RadianceSplitView":     RadianceSplitView,
    "RadianceContactSheet":  RadianceContactSheet,
    "RadianceFlipbookGIF":   RadianceFlipbookGIF,
    "RadianceFrameStamp":    RadianceFrameStamp,
    "RadiancePreviewServer": RadiancePreviewServer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceFalseColorMonitor": "◎ Radiance False Color Monitor",
    "RadianceFocusPeaking":  "◎ Radiance Focus Peaking",
    "RadianceSplitView":     "◎ Radiance Split View",
    "RadianceContactSheet":  "◎ Radiance Contact Sheet",
    "RadianceFlipbookGIF":   "◎ Radiance Flipbook GIF",
    "RadianceFrameStamp":    "◎ Radiance Frame Stamp",
    "RadiancePreviewServer": "◎ Radiance Preview Server",
}
