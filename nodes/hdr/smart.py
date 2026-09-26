"""
nodes_hdr_smart.py — Radiance intelligent HDR pre-processing nodes.

  1. RadianceHDRAutoLogSelect        — histogram-based log format picker +
                                        model-aware compression_ratio preset.
  2. RadianceHDRDiagnostics          — structured JSON diagnostics panel (P6 observability).
"""
from __future__ import annotations

import json
import math
import torch
import torch.nn.functional as F
import logging

from radiance.color.luma import luma_rec2020_tensor as _luma_rec2020

logger = logging.getLogger("radiance.hdr_smart")

# Structured diagnostics logger — all HDR nodes write here so users can
# redirect this single logger to a file / remote sink without touching the
# rest of Radiance's output.
diag_logger = logging.getLogger("radiance.diagnostics")


# ─────────────────────────────────────────────────────────────────────────────
# Per-model preset table  (Priorities 1 & 2)
# ─────────────────────────────────────────────────────────────────────────────
#
# Each entry maps a model-family key → recommended HDR parameters:
#   compression_ratio  float [0,1]  — soft-knee compression for _hdr_soft_compress
#   norm_center        float        — ±N·σ window for RadianceHDRPerChannelNorm
#   vae_spatial_factor int          — VAE spatial downscale (latent size = img / factor)
#   vae_temporal_factor int         — temporal downscale for video VAEs (1 = image)
#   latent_channels    int          — number of latent channels
#   notes              str          — brief rationale
#
# Rationale for video model values:
#   LTX-Video  — uses tone_map_compression_ratio natively; 0.5 is the library default.
#   Flux       — image model, large latent space (16ch), latent std slightly lower than
#                SD3; 0.5 works well and is validated against the sRGB→ACEScg path.
#   CogVideoX  — 3-D causal VAE (4T × 8S factor); latent distribution tighter than Flux
#                so compression can be lower (0.45).  Temporal dim needs careful handling.
#   Wan 2.1    — WanVideo uses its own "Flow-VAE" with a different latent mean target;
#                slightly higher compression (0.6) avoids VAE saturation on highlights.
#   SD 3.5     — same SD3 VAE, 16ch latents, close to Flux defaults.
#   SDXL       — classic 4ch SD VAE; tighter dynamic range, lower norm_center.
#   HunyuanVideo — large video model; latent conventions close to Wan.

RADIANCE_MODEL_PRESETS: dict[str, dict] = {
    "ltx-video": {
        "compression_ratio":   0.50,
        "norm_center":         3.0,
        "vae_spatial_factor":  8,
        "vae_temporal_factor": 8,
        "latent_channels":     128,
        "notes": "LTX-Video native tone_map_compression_ratio default.",
    },
    "flux": {
        "compression_ratio":   0.50,
        "norm_center":         3.0,
        "vae_spatial_factor":  8,
        "vae_temporal_factor": 1,
        "latent_channels":     16,
        "notes": "Flux.1 image model — validated against Bradford sRGB derivation.",
    },
    "cogvideox": {
        "compression_ratio":   0.45,
        "norm_center":         3.0,
        "vae_spatial_factor":  8,
        "vae_temporal_factor": 4,
        "latent_channels":     16,
        "notes": "CogVideoX 3D causal VAE; tighter latent distribution → lower ratio.",
    },
    "wan": {
        "compression_ratio":   0.60,
        "norm_center":         3.5,
        "vae_spatial_factor":  8,
        "vae_temporal_factor": 4,
        "latent_channels":     16,
        "notes": "Wan 2.1 Flow-VAE; different latent mean target → higher compression.",
    },
    "hunyuanvideo": {
        "compression_ratio":   0.60,
        "norm_center":         3.5,
        "vae_spatial_factor":  8,
        "vae_temporal_factor": 4,
        "latent_channels":     16,
        "notes": "HunyuanVideo; similar latent conventions to Wan.",
    },
    "sd3": {
        "compression_ratio":   0.50,
        "norm_center":         3.0,
        "vae_spatial_factor":  8,
        "vae_temporal_factor": 1,
        "latent_channels":     16,
        "notes": "SD 3 / SD 3.5 — same VAE as Flux, identical defaults.",
    },
    "sdxl": {
        "compression_ratio":   0.40,
        "norm_center":         2.5,
        "vae_spatial_factor":  8,
        "vae_temporal_factor": 1,
        "latent_channels":     4,
        "notes": "SDXL classic 4ch VAE; tighter DR headroom → lower ratio.",
    },
    "sd1.5": {
        "compression_ratio":   0.35,
        "norm_center":         2.5,
        "vae_spatial_factor":  8,
        "vae_temporal_factor": 1,
        "latent_channels":     4,
        "notes": "SD 1.x / 2.x — narrowest latent dynamic range.",
    },
}

# Aliases so partial model name strings resolve correctly
_MODEL_ALIASES: dict[str, str] = {
    "ltx":        "ltx-video",
    "flux1":      "flux",
    "flux.1":     "flux",
    "cogvideo":   "cogvideox",
    "cogvideox5b":"cogvideox",
    "wanvideo":   "wan",
    "wan2":       "wan",
    "wan2.1":     "wan",
    "hunyuan":    "hunyuanvideo",
    "sd3.5":      "sd3",
    "sd3":        "sd3",
    "stable-diffusion-3": "sd3",
    "sdxl":       "sdxl",
    "sd1":        "sd1.5",
    "sd2":        "sd1.5",
}

def _resolve_model(hint: str) -> dict | None:
    """Look up preset by model name / alias (case-insensitive, partial match)."""
    key = hint.strip().lower()
    if key in RADIANCE_MODEL_PRESETS:
        return RADIANCE_MODEL_PRESETS[key]
    if key in _MODEL_ALIASES:
        return RADIANCE_MODEL_PRESETS[_MODEL_ALIASES[key]]
    # Partial substring match
    for alias, canonical in _MODEL_ALIASES.items():
        if alias in key:
            return RADIANCE_MODEL_PRESETS[canonical]
    for canonical in RADIANCE_MODEL_PRESETS:
        if canonical in key:
            return RADIANCE_MODEL_PRESETS[canonical]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────
# _luma_rec2020 imported from color_utils (Rec.2020 coefficients, keepdim=True)

def _luma(img: torch.Tensor) -> torch.Tensor:
    """Rec.2020/scene-linear luminance (Y) from an (..., 3) tensor. keepdim=True."""
    return _luma_rec2020(img, keepdim=True)


def _percentile(t: torch.Tensor, q: float) -> float:
    """Scalar percentile of a flat float tensor."""
    k = max(1, int(round(q / 100.0 * t.numel())))
    return float(torch.kthvalue(t.reshape(-1), k).values)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  RadianceHDRAutoLogSelect
# ─────────────────────────────────────────────────────────────────────────────

_LOG_FORMAT_KNEE = {
    # format_name: (knee_stop_above_18grey, description)
    # "knee stop" = how many stops above 0.18 the knee/shoulder begins
    "LogC4":   (14.0, "ARRI ALEXA 35 — widest gamut, best for 14+ stop footage"),
    "SLog3":   (15.0, "Sony — 15-stop range, best for very bright highlights"),
    "VLog":    (12.0, "Panasonic — 12-stop, best for VariCam / GH-series footage"),
    "LogC3":   (13.0, "ARRI ALEXA classic — 13-stop, good general purpose"),
    "ACEScct": (10.0, "ACES creative transform — moderate HDR, grading-friendly"),
}

_LOG_TO_COMPRESSION = {
    # Pre-tested compression_ratio values that work well per format
    "LogC4":   0.6,
    "SLog3":   0.65,
    "VLog":    0.55,
    "LogC3":   0.55,
    "ACEScct": 0.45,
}


def _detect_log_format(image: torch.Tensor) -> str:
    """
    Histogram-based log format selection.

    Strategy:
      1. Compute luminance for the whole batch.
      2. Find the 95th-percentile luminance as a proxy for "peak highlight".
      3. Compute stops above 18% grey: stops = log2(peak / 0.18).
      4. Pick the format whose knee stop count is the closest fit ≥ required stops.
         This ensures highlights are packed into the format's safe range.

    Returns the format name string.
    """
    Y = _luma(image.reshape(-1, image.shape[-1]))    # (N, 1)
    peak95 = _percentile(Y.clamp(min=1e-7), 95.0)

    if peak95 <= 0.0:
        return "ACEScct"  # no signal — use moderate format

    stops_needed = torch.log2(torch.tensor(peak95 / 0.18)).item()
    logger.info("[AutoLogSelect] 95th-pct luma=%.4f  stops_needed=%.1f", peak95, stops_needed)

    # Pick format with smallest knee ≥ stops_needed (or the widest if nothing fits)
    best_fmt   = "LogC4"
    best_delta = float("inf")
    for fmt, (knee, _) in _LOG_FORMAT_KNEE.items():
        delta = knee - stops_needed
        if 0 <= delta < best_delta:
            best_delta = delta
            best_fmt   = fmt

    logger.info("[AutoLogSelect] selected format: %s (knee=%.0f stops)", best_fmt,
                _LOG_FORMAT_KNEE[best_fmt][0])
    return best_fmt


class RadianceHDRAutoLogSelect:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    DESCRIPTION = "Automatically select the optimal log encoding for an HDR input."
    """
    ◎ Radiance HDR Auto Log Format Selector

    Analyses the input image histogram and automatically picks the log
    encoding format whose dynamic-range knee best fits the scene's peak
    highlight level.

    When an optional *model_hint* is supplied (e.g. "flux", "cogvideox",
    "wan2.1"), the node consults the RADIANCE_MODEL_PRESETS table and
    overrides the compression_ratio with the model-tuned value.  This lets
    users wire a single string constant ("flux") to override the
    compression_ratio without manual lookup.

    Outputs:
      • log_format        — string name ("LogC4", "SLog3", …)
      • compression_ratio — recommended ratio (model preset if hint given,
                            otherwise log-format default)
      • stops_detected    — estimated stops above 18% grey (diagnostics)
      • model_preset_used — which model key was resolved (empty = no hint)

    Usage:
        image → RadianceHDRAutoLogSelect → log_format, compression_ratio
                                         → RadianceHDRTurboEncoder
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Scene-linear HDR image (18% grey = 0.18). Its 95th-percentile luma sets how many stops above grey the log format must hold. The image itself is not changed."}),
                "override": (
                    ["auto", "LogC4", "SLog3", "VLog", "LogC3", "ACEScct"],
                    {"default": "auto", "tooltip": "auto = choose the log format with the smallest highlight range that still fits the image. Any other choice forces that format; stops_detected is still measured."},
                ),
            },
            "optional": {
                "model_hint": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": (
                        "Optional model name (e.g. 'flux', 'cogvideox', 'wan2.1', 'sdxl'). "
                        "When set, the compression_ratio output is taken from the "
                        "RADIANCE_MODEL_PRESETS table instead of the log-format default."
                    ),
                }),
            },
        }

    RETURN_TYPES  = ("STRING", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES  = ("log_format", "compression_ratio", "stops_detected", "model_preset_used")
    FUNCTION      = "select"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"

    def select(self, image: torch.Tensor, override: str, model_hint: str = ""):
        if override != "auto":
            fmt = override
        else:
            fmt = _detect_log_format(image)

        Y      = _luma(image.reshape(-1, image.shape[-1]))
        peak95 = max(float(_percentile(Y.clamp(min=1e-7), 95.0)), 1e-7)
        stops  = float(torch.log2(torch.tensor(peak95 / 0.18)).item())

        # Try model preset first; fall back to per-format default
        preset_key = ""
        if model_hint.strip():
            preset = _resolve_model(model_hint)
            if preset is not None:
                comp_ratio = preset["compression_ratio"]
                # Recover canonical key for display
                key = model_hint.strip().lower()
                if key in RADIANCE_MODEL_PRESETS:
                    preset_key = key
                elif key in _MODEL_ALIASES:
                    preset_key = _MODEL_ALIASES[key]
                else:
                    for alias, canonical in _MODEL_ALIASES.items():
                        if alias in key:
                            preset_key = canonical
                            break
                    else:
                        for canonical in RADIANCE_MODEL_PRESETS:
                            if canonical in key:
                                preset_key = canonical
                                break
                logger.info("[AutoLogSelect] model_hint=%r → preset=%s  compression_ratio=%.2f",
                            model_hint, preset_key, comp_ratio)
            else:
                comp_ratio = _LOG_TO_COMPRESSION.get(fmt, 0.5)
                logger.warning("[AutoLogSelect] model_hint=%r not found in presets; "
                               "using log-format default %.2f", model_hint, comp_ratio)
        else:
            comp_ratio = _LOG_TO_COMPRESSION.get(fmt, 0.5)

        logger.info("[AutoLogSelect] fmt=%s  stops=%.2f  compression_ratio=%.2f",
                    fmt, stops, comp_ratio)
        return (fmt, comp_ratio, stops, preset_key)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  RadianceHDRDiagnostics  (P6 — structured observability)
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_psnr_linear(image: torch.Tensor, compression_ratio: float) -> float:
    """
    Estimate the round-trip PSNR that the soft-knee codec will achieve on
    *this specific image* at *this compression_ratio*.

    Method: run the actual compress → decompress in-memory (no VAE, no GPU
    latency) and compare the result to the original.  This gives a realistic
    bound because it isolates codec error from VAE reconstruction error.

    Returns dB (float).  Returns inf for SDR images with ratio=0.
    """
    from radiance.nodes.hdr.encoder import _hdr_soft_compress, _hdr_soft_decompress
    img_pos = image.clamp(min=0.0)
    comp    = _hdr_soft_compress(img_pos, compression_ratio)
    recon   = _hdr_soft_decompress(comp,  compression_ratio)
    peak    = float(img_pos.max())
    if peak < 1e-7:
        return float("inf")
    mse = float(((img_pos - recon) ** 2).mean())
    if mse < 1e-14:
        return float("inf")
    return 10.0 * math.log10((peak ** 2) / mse)


class RadianceHDRDiagnostics:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    DESCRIPTION = "Run full HDR diagnostic checks. Outputs a JSON report, estimated PSNR, peak stops, and live metric floats (peak_nit, ev_range, clipped_pct, is_hdr) — replaces the separate RadianceHDRAnalysis node."
    """
    ◎ Radiance HDR Diagnostics Panel

    Collects all key HDR pipeline metrics into a single structured JSON
    report and emits them to the 'radiance.diagnostics' logger so users
    can redirect the output to a file or remote sink.

    Useful for:
      • Tuning compression_ratio without guesswork — see exact PSNR estimate.
      • Confirming the model preset was resolved correctly.
      • Verifying coherence map quality before starting a long render.
      • CI regression: pipe the JSON output into a test assertion.

    Inputs (all optional except image):
        image             — The HDR input (before compression), used to compute
                            peak stops, histogram stats, and PSNR estimate.
        compression_ratio — Ratio currently in use (for PSNR estimate).
        model_preset_used — String key from RadianceHDRAutoLogSelect
                            (diagnostic label only).
        stats_json        — JSON string from RadianceHDRPerChannelNorm
                            (shows channel mean / std after normalisation).
        coherence_map     — Optional IMAGE for coherence analysis
                            (shows mean coherence score).

    Outputs:
        report_json       — Structured JSON string with all metrics.
        psnr_estimate     — Estimated round-trip PSNR in dB (FLOAT, wire to
                            a ComfyUI display node for live monitoring).
        peak_stops        — Stops above 18% grey at 95th percentile luma.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "HDR image before compression, encoded as set by colorspace. Metrics treat linear 1.0 as 203 nits."}),
            },
            "optional": {
                "compression_ratio": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Must match the value used in RadianceHDRTurboEncoder.",
                }),
                "model_preset_used": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Resolved model key from AutoLogSelect.",
                }),
                "stats_json": ("STRING", {
                    "default": "",
                    "forceInput": True,
                    "tooltip": "JSON from RadianceHDRPerChannelNorm (optional).",
                }),
                "coherence_map": ("IMAGE", {"tooltip": "Optional coherence map (0 to 1). Only its mean is reported, as coherence_mean in the JSON."}),
                "colorspace": (
                    ["Linear (sRGB)", "ACEScg", "sRGB", "Rec.709"],
                    {"default": "Linear (sRGB)",
                     "tooltip": "Input colour space for nit/EV-range estimation."},
                ),
            },
        }

    RETURN_TYPES  = ("STRING", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "BOOLEAN")
    RETURN_NAMES  = ("report_json", "psnr_estimate", "peak_stops",
                      "peak_nit", "ev_range", "clipped_pct", "is_hdr")
    FUNCTION      = "diagnose"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    OUTPUT_NODE   = True   # marks this as a terminal / sink node in ComfyUI

    def diagnose(
        self,
        image: torch.Tensor,
        compression_ratio: float = 0.5,
        model_preset_used: str   = "",
        stats_json: str          = "",
        coherence_map: torch.Tensor | None = None,
        colorspace: str = "Linear (sRGB)",
    ):
        # ── Colour space ──────────────────────────────────────────────────────
        # The widget was read by nothing: an sRGB-encoded image was measured
        # as if its code values were light, and luma always used BT.2020
        # weights. Display-encoded input is decoded first, and luma uses the
        # weights of the input's primaries.
        image = image[..., :3].float()
        if colorspace in ("sRGB", "Rec.709"):
            x = image.clamp(0.0, 1.0)
            if colorspace == "sRGB":
                image = torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
            else:  # BT.1886 display, gamma 2.4
                image = x ** 2.4
        luma_w = [0.2722287, 0.6740818, 0.0536895] if colorspace == "ACEScg" \
            else [0.2126, 0.7152, 0.0722]

        # ── Luma stats ────────────────────────────────────────────────────────
        w   = image.new_tensor(luma_w)
        Y   = (image * w).sum(dim=-1)          # (B, H, W)
        eps = 1e-7

        peak_linear  = float(image.clamp(min=0).max())
        peak95_luma  = float(torch.kthvalue(
            Y.clamp(min=eps).reshape(-1),
            max(1, int(0.95 * Y.numel())),
        ).values)
        peak_stops   = float(torch.log2(torch.tensor(max(peak95_luma, eps) / 0.18)).item())
        mean_luma    = float(Y.mean())
        min_luma     = float(Y.min())

        # ── Codec PSNR estimate ───────────────────────────────────────────────
        try:
            psnr_db = _estimate_psnr_linear(image, compression_ratio)
        except Exception:
            psnr_db = float("nan")

        # ── Channel stats (from PerChannelNorm) ───────────────────────────────
        channel_stats: dict = {}
        if stats_json.strip():
            try:
                channel_stats = json.loads(stats_json)
            except json.JSONDecodeError:
                channel_stats = {"error": "invalid JSON"}

        # ── Coherence map mean ────────────────────────────────────────────────
        coherence_mean = float("nan")
        if coherence_map is not None:
            coherence_mean = float(coherence_map.clamp(0, 1).mean())

        # ── Model preset details ──────────────────────────────────────────────
        preset_detail: dict = {}
        if model_preset_used.strip():
            p = _resolve_model(model_preset_used)
            if p:
                preset_detail = {
                    "compression_ratio":   p["compression_ratio"],
                    "norm_center":         p["norm_center"],
                    "vae_spatial_factor":  p["vae_spatial_factor"],
                    "vae_temporal_factor": p["vae_temporal_factor"],
                    "latent_channels":     p["latent_channels"],
                }

        # ── Assemble report ───────────────────────────────────────────────────
        report = {
            "radiance_diagnostics": {
                "image_shape":        list(image.shape),
                "colorspace":         colorspace,
                "peak_linear":        round(peak_linear,  4),
                "peak95_luma":        round(peak95_luma,  4),
                "peak_stops_above_18pct_grey": round(peak_stops, 2),
                "mean_luma":          round(mean_luma, 4),
                "min_luma":           round(float(min_luma), 4),
                "compression_ratio":  compression_ratio,
                "psnr_estimate_db":   round(psnr_db, 1) if math.isfinite(psnr_db) else "inf",
                "coherence_mean":     round(coherence_mean, 4) if math.isfinite(coherence_mean) else None,
                "model_preset_used":  model_preset_used or None,
                "model_preset_detail": preset_detail or None,
                "channel_stats":      channel_stats or None,
            }
        }
        report_str = json.dumps(report, indent=2)

        # ── Emit to structured diagnostics logger ─────────────────────────────
        diag_logger.info(
            "HDR_DIAG peak_stops=%.2f psnr_estimate=%.1f dB "
            "compression_ratio=%.2f coherence=%.3f model=%s",
            peak_stops,
            psnr_db if math.isfinite(psnr_db) else -1.0,
            compression_ratio,
            coherence_mean if math.isfinite(coherence_mean) else -1.0,
            model_preset_used or "—",
        )
        logger.debug("[Diagnostics] full report:\n%s", report_str)

        # ── HDRAnalysis-equivalent metrics (BT.2408: 203 nit = linear 1.0) ──────
        _NIT_ANCHOR = 203.0
        _img_lin  = image[..., :3].clamp(min=0.0).float()
        _w        = _img_lin.new_tensor(luma_w)
        _Y_flat   = (_img_lin * _w).sum(dim=-1).reshape(-1)
        _n        = _Y_flat.numel()
        _p01      = float(_Y_flat.kthvalue(max(1, int(0.01 * _n))).values)
        _p99      = float(_Y_flat.kthvalue(max(1, int(0.99 * _n))).values)
        _peak_nit = float(_img_lin.max()) * _NIT_ANCHOR
        _ev_range = float(math.log2(max(_p99, 1e-7) / max(_p01, 1e-7)))
        _clipped  = float((_img_lin > 1.0).float().mean()) * 100.0
        _is_hdr   = bool(_peak_nit > _NIT_ANCHOR)
        return (
            report_str,
            float(psnr_db) if math.isfinite(psnr_db) else 0.0,
            peak_stops,
            _peak_nit,
            _ev_range,
            _clipped,
            _is_hdr,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Node registry
# ─────────────────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "RadianceHDRAutoLogSelect":        RadianceHDRAutoLogSelect,
    "RadianceHDRDiagnostics":          RadianceHDRDiagnostics,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceHDRAutoLogSelect":        "◎ Radiance HDR Auto Log Selector",
    "RadianceHDRDiagnostics":          "◎ Radiance HDR Diagnostics",
}
