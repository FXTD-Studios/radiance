import torch
import numpy as np
import logging
from typing import Tuple, Dict, Any

# Local imports
from .utils import tensor_to_numpy_float32, numpy_to_tensor_float32

logger = logging.getLogger("radiance.hdr.analysis")

# ═══════════════════════════════════════════════════════════════════════════════
#                          ANALYSIS NODES
# ═══════════════════════════════════════════════════════════════════════════════


class HDRHistogram:
    """
    Analyze HDR image histogram with dynamic range statistics, clipping indicators, and stops visualization.
    """

    MODES = ["luminance", "rgb", "log_luminance"]

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (cls.MODES, {"default": "luminance"}),
                "show_clipping": ("BOOLEAN", {"default": True,
                    "tooltip": "Highlight overexposed pixels in red on the waveform/vectorscope overlay.",
                }),
                "stops_range": ("INT", {"default": 14, "min": 8, "max": 24,
                    "tooltip": "Dynamic range to display in the exposure waveform, measured in stops.",
                }),
                "hdr_zone_markers": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Draw vertical lines at 1×, 2×, 4×, 8× EV (scene-linear 1.0/2.0/4.0/8.0) "
                        "on the linear histogram. Essential for HDR authoring: shows where "
                        "highlight data exceeds the SDR clip point."
                    ),
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "FLOAT")
    RETURN_NAMES = ("histogram", "stats", "headroom_pct")
    FUNCTION = "analyze"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Display"
    DESCRIPTION = (
        "v2.4 — Analyze HDR image histogram with dynamic range statistics, clipping indicators, "
        "HDR zone markers, x-axis tick labels on the linear panel, and per-stop gridlines "
        "with EV axis labels on the log panel."
    )

    def analyze(
        self,
        image: torch.Tensor,
        mode: str = "luminance",
        show_clipping: bool = True,
        stops_range: int = 14,
        hdr_zone_markers: bool = True,
    ) -> Tuple[torch.Tensor, str, float]:

        logger.debug("HDRHistogram analyzing...")
        try:
            from PIL import Image, ImageDraw

            img = tensor_to_numpy_float32(image)
            if img.ndim == 4:
                img = img[0]  # Take first image in batch

            # Sanitize Input: Handle NaNs and Infs
            img = np.nan_to_num(img, nan=0.0, posinf=65504.0, neginf=-65504.0)

            # Calculate luminance
            if img.shape[-1] >= 3:
                luma = (
                    0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
                )
            else:
                luma = img[..., 0]

            # Statistics
            min_val = float(np.min(img))
            max_val = float(np.max(img))
            mean_val = float(np.mean(img))

            # Dynamic range in stops
            eps = 1e-10
            # FIX 2: dynamic_range must use the minimum POSITIVE value.
            # Using min_val directly causes log2(max/eps) ≈ 49 stops when
            # min_val is negative (valid HDR wide-gamut data).
            positive_vals = img[img > 0]
            min_positive = float(positive_vals.min()) if positive_vals.size > 0 else float(eps)
            dynamic_range = np.log2(max(max_val, eps) / max(min_positive, eps))

            # Clipping analysis
            clip_low  = float(np.sum(img <= 0) / img.size * 100)
            clip_high = float(np.sum(img >= 1) / img.size * 100)
            # Headroom: pixels above 1.0 (SDR clip) in scene-linear
            headroom_pct_val = float(np.sum(luma > 1.0) / luma.size * 100)

            # FIX 3: stats string had mixed indentation — first separator was
            # flush-left while all data lines were indented, producing a ragged
            # misaligned block. Now consistently dedented.
            stats = (
                "═══════════════════════════════════\n"
                "      HDR IMAGE ANALYSIS            \n"
                "═══════════════════════════════════\n"
                f"Min Value:     {min_val:.6f}\n"
                f"Max Value:     {max_val:.6f}\n"
                f"Mean Value:    {mean_val:.6f}\n"
                f"Dynamic Range: {dynamic_range:.1f} stops\n"
                "───────────────────────────────────\n"
                f"Clipped Low:   {clip_low:.2f}%\n"
                f"Clipped High:  {clip_high:.2f}%\n"
                f"HDR Headroom:  {headroom_pct_val:.2f}% (>1.0 EV)\n"
                "═══════════════════════════════════"
            )

            # ═══════════════════════════════════════════════════════════════
            # CREATE HISTOGRAM WITH PIL (NO MATPLOTLIB)
            # ═══════════════════════════════════════════════════════════════
            logger.debug("HDRHistogram rendering with PIL...")

            # Canvas size
            hist_w, hist_h = 1000, 600
            bg_color = (26, 26, 46)  # Dark blue background

            # Create image
            hist_img = Image.new("RGB", (hist_w, hist_h), bg_color)
            draw = ImageDraw.Draw(hist_img)

            # Graph area
            margin = 60
            graph_w = hist_w - margin * 2
            graph_h = (hist_h - margin * 3) // 2
            graph_top1 = margin
            graph_top2 = margin + graph_h + margin

            # ── LINEAR HISTOGRAM ──
            draw.rectangle(
                [
                    margin - 5,
                    graph_top1 - 5,
                    margin + graph_w + 5,
                    graph_top1 + graph_h + 5,
                ],
                fill=(22, 33, 62),
            )

            # Compute histogram bins
            num_bins = 256
            bin_width = graph_w / num_bins

            if mode == "rgb" and img.shape[-1] >= 3:
                # RGB mode - draw each channel
                # FIX 4: Previous code halved RGB values to fake transparency
                # (alpha_color = c//2). PIL has no real alpha blend on RGB canvases,
                # so channels just painted over each other at 50% brightness —
                # producing muddy dim bars with no actual overlap visualisation.
                # Fix: render each channel into a separate RGBA layer at full
                # intensity and composite with Image.alpha_composite for correct
                # additive overlap (R+G = yellow, R+B = magenta, G+B = cyan).
                from PIL import Image as _PILImage
                colors_rgba = [(255, 60, 60, 160), (60, 255, 60, 160), (60, 60, 255, 160)]
                # Convert base graph to RGBA for compositing
                hist_img_rgba = hist_img.convert("RGBA")
                colors = [(255, 80, 80), (80, 255, 80), (80, 80, 255)]  # kept for draw ref
                for ch_idx, (color, color_a) in enumerate(zip(colors, colors_rgba)):
                    channel = img[..., ch_idx].flatten()
                    channel = np.clip(channel, 0, max(1, max_val))
                    hist_vals, _ = np.histogram(
                        channel, bins=num_bins, range=(0, max(1, max_val))
                    )
                    max_count = max(hist_vals.max(), 1)
                    # Draw channel onto its own transparent layer
                    ch_layer = _PILImage.new("RGBA", (hist_w, hist_h), (0, 0, 0, 0))
                    ch_draw = ImageDraw.Draw(ch_layer)
                    for i, count in enumerate(hist_vals):
                        bar_h = int((count / max_count) * graph_h * 0.9)
                        x1 = margin + int(i * bin_width)
                        x2 = margin + int((i + 1) * bin_width)
                        y1 = graph_top1 + graph_h - bar_h
                        y2 = graph_top1 + graph_h
                        ch_draw.rectangle([x1, y1, x2, y2], fill=color_a)
                    hist_img_rgba = _PILImage.alpha_composite(hist_img_rgba, ch_layer)
                hist_img = hist_img_rgba.convert("RGB")
                draw = ImageDraw.Draw(hist_img)  # refresh draw handle after conversion
            else:
                # Luminance mode
                hist_vals, _ = np.histogram(
                    luma.flatten(), bins=num_bins, range=(0, max(1, max_val))
                )
                max_count = max(hist_vals.max(), 1)

                for i, count in enumerate(hist_vals):
                    bar_h = int((count / max_count) * graph_h * 0.9)
                    x1 = margin + int(i * bin_width)
                    x2 = margin + int((i + 1) * bin_width)
                    y1 = graph_top1 + graph_h - bar_h
                    y2 = graph_top1 + graph_h
                    draw.rectangle([x1, y1, x2, y2], fill=(233, 69, 96))

            # Clipping indicators
            if show_clipping:
                draw.line(
                    [(margin, graph_top1), (margin, graph_top1 + graph_h)],
                    fill=(0, 255, 255),
                    width=2,
                )
                # SDR clip at 1.0 / max_val
                white_x = margin + int(graph_w * min(1.0 / max(max_val, 1), 1.0))
                draw.line(
                    [(white_x, graph_top1), (white_x, graph_top1 + graph_h)],
                    fill=(255, 255, 0),
                    width=2,
                )

            # HDR zone markers: vertical lines at 1×, 2×, 4×, 8× scene-linear
            if hdr_zone_markers and max_val > 1.0:
                zone_colors = [
                    (255, 200,  60),   # 1.0 EV  (SDR limit) — amber
                    (255, 140,  20),   # 2.0 EV  (+1 stop)   — orange
                    (255,  80,  20),   # 4.0 EV  (+2 stops)  — deep orange
                    (220,  40,  40),   # 8.0 EV  (+3 stops)  — red
                ]
                zone_levels = [1.0, 2.0, 4.0, 8.0]
                zone_labels = ["+0EV", "+1EV", "+2EV", "+3EV"]
                for zval, zcolor, zlabel in zip(zone_levels, zone_colors, zone_labels):
                    if zval <= max_val:
                        zx = margin + int(graph_w * min(zval / max(max_val, 1e-8), 1.0))
                        draw.line(
                            [(zx, graph_top1), (zx, graph_top1 + graph_h)],
                            fill=zcolor, width=1,
                        )
                        if zx + 2 < margin + graph_w:
                            draw.text((zx + 2, graph_top1 + 4), zlabel, fill=zcolor)

            # v2.4: X-axis tick labels for the linear histogram.
            # Draw value ticks at the bottom edge of the graph panel at key
            # scene-linear levels: 0, 0.5, 1.0 (SDR clip), 2, 4, 8, 16 (if in range).
            # Each tick gets a small vertical mark and a numeric label below it.
            _lin_ticks = [v for v in [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
                          if v <= max_val * 1.02]
            _tick_y_base = graph_top1 + graph_h
            for tv in _lin_ticks:
                tx = margin + int(graph_w * min(tv / max(max_val, 1e-8), 1.0))
                # Tick mark (3 px tall)
                draw.line([(tx, _tick_y_base), (tx, _tick_y_base + 4)],
                          fill=(100, 100, 120), width=1)
                # Label — "1.0" gets special amber colour (SDR clip)
                tick_label = f"{tv:.1f}" if tv < 1.0 else f"{tv:.0f}"
                tick_color = (255, 200, 60) if tv == 1.0 else (150, 150, 180)
                draw.text((tx - 7, _tick_y_base + 5), tick_label, fill=tick_color)

            # Title
            draw.text(
                (margin, graph_top1 - 25), "Linear Histogram", fill=(255, 255, 255)
            )

            # ── LOG HISTOGRAM (STOPS) ──
            draw.rectangle(
                [
                    margin - 5,
                    graph_top2 - 5,
                    margin + graph_w + 5,
                    graph_top2 + graph_h + 5,
                ],
                fill=(22, 33, 62),
            )

            # Compute log histogram
            log_luma = np.log2(np.maximum(luma, eps))
            min_stop = -8
            max_stop = stops_range - 8
            log_bins = stops_range * 10

            log_hist, log_edges = np.histogram(
                log_luma.flatten(), bins=log_bins, range=(min_stop, max_stop)
            )
            log_max = max(log_hist.max(), 1)
            log_bin_width = graph_w / log_bins

            for i, count in enumerate(log_hist):
                bar_h = int((count / log_max) * graph_h * 0.9)
                x1 = margin + int(i * log_bin_width)
                x2 = margin + int((i + 1) * log_bin_width)
                y1 = graph_top2 + graph_h - bar_h
                y2 = graph_top2 + graph_h
                draw.rectangle([x1, y1, x2, y2], fill=(15, 52, 96))

            # v2.4: Per-stop vertical gridlines on the log histogram.
            # Every full stop gets a faint rule + axis label at the bottom.
            # Even stops get slightly brighter lines for easier reading at a glance.
            _stop_label_y = graph_top2 + graph_h + 3
            for _s in range(int(min_stop), int(max_stop) + 1):
                _sx = margin + int((_s - min_stop) / (max_stop - min_stop) * graph_w)
                _line_col = (50, 50, 68) if (_s % 2 == 0) else (38, 38, 52)
                draw.line([(_sx, graph_top2), (_sx, graph_top2 + graph_h)],
                          fill=_line_col, width=1)
                # Label every 2 stops (or every stop if range ≤ 12)
                if (max_stop - min_stop) <= 12 or (_s % 2 == 0):
                    _sl = f"{_s:+d}" if _s != 0 else "0 EV"
                    _sc = (233, 69, 96) if _s == 0 else (100, 100, 130)
                    draw.text((_sx - 7, _stop_label_y), _sl, fill=_sc)

            # Middle gray marker (0 stops = 18% gray) — draw ON TOP of gridlines
            mid_x = margin + int(graph_w * (-min_stop) / (max_stop - min_stop))
            draw.line(
                [(mid_x, graph_top2), (mid_x, graph_top2 + graph_h)],
                fill=(233, 69, 96),
                width=2,
            )

            # v2.4: EV axis label at the top-left of the log panel
            draw.text((margin, graph_top2 + 3), "EV (log₂)", fill=(120, 120, 150))

            # Title
            draw.text(
                (margin, graph_top2 - 25),
                f"Log Histogram ({stops_range} stops)",
                fill=(255, 255, 255),
            )

            # Bottom info
            draw.text(
                (margin, hist_h - 30),
                f"DR: {dynamic_range:.1f} stops | Min: {min_val:.3f} | Max: {max_val:.3f}",
                fill=(180, 180, 200),
            )

            # Convert to tensor
            hist_np = np.array(hist_img).astype(np.float32) / 255.0
            logger.debug("HDRHistogram done.")
            # FIX 5: wrap in batch dimension — ComfyUI IMAGE is (B, H, W, C).
            # numpy_to_tensor_float32 returns (H, W, C) for a single frame;
            # without unsqueeze(0) downstream nodes crash on .shape batch index.
            hist_tensor = numpy_to_tensor_float32(hist_np).unsqueeze(0)
            return (hist_tensor, stats, float(headroom_pct_val))

        except Exception as e:
            logger.error(f"HDRHistogram error: {e}")
            import traceback

            traceback.print_exc()

            # Return RED ERROR IMAGE so user knows it failed
            if image.dim() == 4:
                B, H, W, C = image.shape
                err_img = torch.zeros((B, H, W, C), dtype=torch.float32)
                err_img[..., 0] = 1.0  # Red
            else:
                H, W, C = image.shape
                err_img = torch.zeros((H, W, C), dtype=torch.float32)
                err_img[..., 0] = 1.0  # Red

            return (err_img, f"Error: {str(e)}", 0.0)


# =============================================================================
# NODE MAPPINGS
# FIX 1: NODE_CLASS_MAPPINGS was completely absent — HDRHistogram was invisible
# to ComfyUI and could not be used in any workflow.
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "RadianceHDRHistogram": HDRHistogram,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceHDRHistogram": "◎ HDR Histogram",
}
