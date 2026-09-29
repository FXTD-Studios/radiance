# ============================================================
# FXTD STUDIOS — Radiance
# t2v.py  —  Text-to-Video & Image-to-Video Wrappers
# ============================================================
# Unified wrapper layer over any DiT video model loaded in ComfyUI.
# Supports: LTX-Video 2.x, HunyuanVideo, Wan2.1, CogVideoX, Mochi-1.
#
# Node overview
# -------------
#   RadianceVideoModelInfo    — inspect a loaded MODEL's latent spec
#   RadianceVideoLatentNoise  — generate correctly-shaped noise for any DiT
#   RadianceT2VPipeline       — text  → video latents (full sampling loop)
#   RadianceI2VPipeline       — image → video latents (image-conditioned)
#   RadianceVideoSampler      — low-level sampler shared by T2V and I2V
#   RadianceVideoBatchDecode   — decode video latents → IMAGE frame batch
#   RadianceVideoCondMerge    — merge character + HDR + text conditionings
#   RadianceVideoExport       — route frame batch to video writer / EXR
#
# ComfyUI integration notes
# -------------------------
# All sampling is done through ComfyUI's comfy.sample / comfy.samplers
# infrastructure — no custom CUDA kernels or model weights are shipped.
# When running in test environments without ComfyUI, every node falls
# back gracefully and returns zero tensors + descriptive error text.
#
# I2V conditioning strategy (per model)
# --------------------------------------
#   LTX-Video   — first frame injected via img2vid conditioning API
#                  (model.apply_model receives image_start tensor)
#   HunyuanVideo — first-frame encoded latent concatenated to noise
#   Wan2.1       — CLIP vision features injected into cross-attention
#   CogVideoX    — reference latent prepended to the latent sequence
#   Generic      — encode image, inject as first frame, pad remaining
# ============================================================

from radiance.config.constants import VERSION as __version__  # noqa: E402

import logging
logger = logging.getLogger("radiance.video.t2v")
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple


def _hdr_prompt_descriptors(peak_nits, gamut, eotf):
    """Describe the requested look; this does not encode or master pixels."""
    gamuts = {"BT.2020": "wide color gamut rec2020 vivid",
              "P3-D65": "Display P3 D65 cinema color",
              "P3-DCI": "DCI-P3 theatrical cinema color",
              "BT.709": "Rec709 standard color gamut",
              "ACEScg": "aces linear light vfx reference"}
    transfers = {"PQ (ST.2084)": "HDR10 PQ specular highlights",
                 "HLG (BT.2100)": "HLG broadcast HDR",
                 "Linear": "linear EXR 32bit",
                 "sRGB / BT.1886": "sRGB BT1886 standard dynamic range"}
    parts = [gamuts.get(gamut, ""), transfers.get(eotf, "")]
    if peak_nits is not None:
        nits = int(peak_nits)
        parts.append(f"{nits} nits {'HDR' if nits > 100 else 'SDR'}")
    return ", ".join(part for part in parts if part)


try:
    import PIL  # noqa: F401
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# ---------------------------------------------------------------------------
# ComfyUI runtime imports (graceful fallback)
# ---------------------------------------------------------------------------
try:
    import comfy.samplers
    import comfy.sample
    import comfy.utils
    HAS_COMFY = True
except ImportError:
    HAS_COMFY = False

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

# DiT latent specs. This import used to also ask dit.py for _wrap and
# _to_tensor, which it never defined, so it always failed and every preset
# silently got the same 16 ch /8 /4 fallback spec.
from radiance.nodes.video.dit import _MODEL_SPECS, MODEL_NAMES, _get_spec


def _wrap(t):
    return {"samples": t}


def _to_tensor(latent):
    return latent["samples"] if isinstance(latent, dict) else latent


# ---------------------------------------------------------------------------
# ComfyUI sampler / scheduler name lists
# ---------------------------------------------------------------------------

_SAMPLERS = [
    "euler", "euler_ancestral", "heun", "heunpp2", "dpm_2", "dpm_2_ancestral",
    "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde",
    "dpmpp_sde_gpu", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu",
    "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", "lcm", "ipndm", "ipndm_v",
    "deis", "ddim", "uni_pc", "uni_pc_bh2",
]
_SCHEDULERS = [
    "normal", "karras", "exponential", "sgm_uniform", "simple",
    "ddim_uniform", "beta",
]

# Per-model recommended defaults
_MODEL_DEFAULTS: Dict[str, Dict] = {
    "LTX-Video (128ch)":    {"sampler": "euler", "scheduler": "normal",   "steps": 25, "cfg": 3.5},
    "HunyuanVideo (16ch)": {"sampler": "euler", "scheduler": "simple",   "steps": 50, "cfg": 7.0},
    "Wan2.1 (16ch)":        {"sampler": "dpmpp_2m", "scheduler": "karras","steps": 30, "cfg": 5.0},
    "CogVideoX (16ch)":     {"sampler": "ddim", "scheduler": "ddim_uniform","steps": 50,"cfg": 6.0},
    "Mochi-1 (12ch)":       {"sampler": "euler", "scheduler": "normal",   "steps": 64, "cfg": 4.5},
    "Wan2.2-TI2V-5B (48ch)": {"sampler": "uni_pc", "scheduler": "simple", "steps": 20, "cfg": 5.0},
    "HunyuanVideo-1.5 (32ch)": {"sampler": "euler", "scheduler": "simple", "steps": 20, "cfg": 6.0},
    "SD-VAE (4ch)":         {"sampler": "euler", "scheduler": "karras",   "steps": 20, "cfg": 7.0},
}


# ===========================================================================
# ALBABIT-FIX: shared dit_config / noise-shape helpers (were triplicated
# across RadianceVideoLatentNoise, RadianceT2VPipeline, RadianceI2VPipeline)
# ===========================================================================

def _resolve_dit_config(dit_config, steps=0, cfg=0.0,
                        sampler_name="euler", scheduler="normal"):
    """Parse dit_config JSON and resolve model spec + effective sampling params."""
    try:
        cfg_dict = json.loads(dit_config) if dit_config.strip() not in ("", "{}") else {}
    except Exception:
        cfg_dict = {}
    spec        = _get_spec(cfg_dict.get("model_name", "")) if cfg_dict else _get_spec("LTX-Video (128ch)")
    spec.update(cfg_dict)
    model_name  = spec.get("model_name", "LTX-Video (128ch)")
    defaults    = _MODEL_DEFAULTS.get(model_name, {})
    eff_steps   = steps        if steps   > 0   else defaults.get("steps",    25)
    eff_cfg     = cfg          if cfg     > 0.0  else defaults.get("cfg",     7.0)
    eff_sampler = sampler_name or defaults.get("sampler",    "euler")
    eff_sched   = scheduler    or defaults.get("scheduler",  "normal")
    return spec, model_name, eff_steps, eff_cfg, eff_sampler, eff_sched


def _build_noise_shape(spec, width, height, frames, batch_size=1):
    """Compute latent noise tensor shape from model spec and pixel dimensions."""
    sc   = spec.get("spatial_compression", 8)
    tc   = spec.get("temporal_compression", 4)
    ch   = spec.get("channels", 16)
    temp = spec.get("temporal", True)
    lh   = max(1, height // sc)
    lw   = max(1, width  // sc)
    lt   = max(1, math.ceil(frames / tc)) if temp else 1
    if temp and lt > 1:
        return (batch_size, ch, lt, lh, lw)
    return (batch_size, ch, lh, lw)


# ===========================================================================
# Shared sampling helper
# ===========================================================================

def _comfy_sample(model, noise, steps, cfg, sampler_name, scheduler,
                   positive, negative, latent_image, denoise=1.0, seed=0):
    """
    Thin wrapper around comfy.sample.sample_custom (modern ComfyUI API).
    Falls back to returning the noise unchanged when ComfyUI unavailable.
    """
    if not HAS_COMFY or not HAS_TORCH:
        return noise   # test / no-ComfyUI path

    # ALBABIT-FIX: migrate from legacy KSampler to sample_custom, matching
    # RadianceSamplerPro. ComfyUI v0.26.0 CFGGuider.sample() calls .is_nested
    # directly on the latent — LATENT dicts lack this attribute, raw tensors do not.
    if isinstance(latent_image, dict):
        latent_image = latent_image.get("samples", noise)

    try:
        # Sigma computation replicates KSampler.set_steps() / calculate_sigmas(),
        # including penultimate-sigma discard and partial-denoise slicing.
        _DISCARD_PENULTIMATE = {"dpm_2", "dpm_2_ancestral", "uni_pc", "uni_pc_bh2"}
        sampler_obj = comfy.samplers.sampler_object(sampler_name)
        model_sampling = model.get_model_object("model_sampling")

        def _calc_sigmas(n):
            _n = n + (1 if sampler_name in _DISCARD_PENULTIMATE else 0)
            sigs = comfy.samplers.calculate_sigmas(model_sampling, scheduler, _n)
            if sampler_name in _DISCARD_PENULTIMATE:
                sigs = torch.cat([sigs[:-2], sigs[-1:]])
            return sigs

        if denoise <= 0.0:
            sigmas = torch.FloatTensor([])
        elif denoise >= 0.9999:
            sigmas = _calc_sigmas(steps)
        else:
            sigmas = _calc_sigmas(int(steps / denoise))[-(steps + 1):]
        sigmas = sigmas.to(noise.device)

        samples = comfy.sample.sample_custom(
            model, noise, cfg, sampler_obj, sigmas,
            positive, negative, latent_image, seed=seed,
        )
        # ALBABIT-FIX: guard against NaN/Inf that silently produce black/corrupt frames.
        # Mirrors the same guard in RadianceSamplerPro.sample().
        try:
            if not torch.isfinite(samples).all():
                _bad = int((~torch.isfinite(samples)).sum().item())
                logger.warning(
                    "[RadianceVideoSampler] %d non-finite value(s) (NaN/Inf) in sampled "
                    "latent — sanitizing via nan_to_num. Check CFG, scheduler, and precision.",
                    _bad,
                )
                samples = torch.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception as _nf_err:
            logger.debug("[RadianceVideoSampler] finite-check skipped: %s", _nf_err)
        return samples
    except Exception as exc:
        # Returning `noise` here turned every sampling failure into a
        # plausible-looking success: the caller wrapped raw noise in a LATENT,
        # printed "Sampled: [...]" and the preview decoded to static, so an OOM
        # or a bad sampler/scheduler pair was indistinguishable from a finished
        # render. torch.cuda.OutOfMemoryError subclasses RuntimeError subclasses
        # Exception, so the broad handler caught exactly the case that matters.
        # Sampling has no meaningful degraded result, so the failure propagates.
        logger.error("[RadianceT2V] sampling failed: %s", exc)
        raise


def _make_noise(shape, seed: int = 0) -> "torch.Tensor":
    """Reproducible gaussian noise."""
    if not HAS_TORCH:
        return None
    gen = torch.Generator()
    gen.manual_seed(seed % (2**32))
    return torch.randn(shape, generator=gen)


# Inference only. .eval() does not clear requires_grad on parameters, so an
# unguarded forward still builds and retains an autograd graph.
@torch.no_grad()
def _vae_encode(vae, image_tensor) -> "torch.Tensor":
    """Encode an IMAGE tensor [B,H,W,3] through the VAE.

    A failed encode propagates. It used to return a 4-channel zero latent,
    which every I2V strategy then blended in as if it were the image.
    """
    if not HAS_COMFY or not HAS_TORCH:
        B, H, W, C = image_tensor.shape
        return torch.zeros(B, 4, H // 8, W // 8)
    try:
        return vae.encode(image_tensor[:, :, :, :3])
    except Exception as exc:
        raise RuntimeError(f"[RadianceI2V] VAE encode of the reference image failed: {exc}") from exc


# Inference only. .eval() does not clear requires_grad on parameters, so an
# unguarded forward still builds and retains an autograd graph.
@torch.no_grad()
def _vae_decode(vae, latent) -> "torch.Tensor":
    """Decode latents → IMAGE tensor.

    A 5-D video latent is handed to the VAE as 5-D: comfy.sd.VAE.decode()
    computes ``dims = samples_in.ndim - 2`` and only reaches its video path
    (and its decode_tiled_3d memory fallback) when that is 3.
    """
    if not HAS_COMFY or not HAS_TORCH:
        return latent
    try:
        samples = _to_tensor(latent)
        return vae.decode(samples)
    except Exception as exc:
        # Returning a 64x64 black tensor made an OOM look like a decode: the
        # caller clamped it, counted its frames and reported success, so a long
        # clip came back as black tiles with no failing node in the graph.
        # torch.cuda.OutOfMemoryError subclasses RuntimeError subclasses
        # Exception, so the broad handler swallowed the case that matters most.
        # Callers that can survive without pixels catch this and say so in their
        # report; the rest let it stop the graph.
        logger.error("[RadianceT2V] VAE decode failed: %s", exc)
        raise


# ---------------------------------------------------------------------------
# VAE compression probes + tiled decode
# ---------------------------------------------------------------------------
# Defaults copied from ComfyUI's own VAEDecodeTiled node (nodes.py) so a
# Radiance tiled decode tiles the same way the stock node does. All four are
# PIXEL-space sizes; they are divided by the VAE's compression before the call.
_TILE_SIZE_PX = 512
_TEMPORAL_TILE_PX = 64
_TEMPORAL_OVERLAP_PX = 8


def _vae_spatial_compression(vae, fallback: int = 8) -> int:
    """Pixel width per latent column, from the VAE when it will say.

    Note the spelling: comfy.sd.VAE calls it ``spacial_compression_decode``.
    Third-party VAE-like objects are not guaranteed to implement it, so this
    mirrors the defensive getattr/callable/try pattern used in hdr/vae.py.
    """
    fn = getattr(vae, "spacial_compression_decode", None) if vae is not None else None
    if callable(fn):
        try:
            value = fn()
            if value:
                return max(1, int(value))
        except Exception as exc:
            logger.debug("[RadianceT2V] spacial_compression_decode() unusable: %s", exc)
    return max(1, int(fallback or 8))


def _vae_temporal_compression(vae) -> Optional[int]:
    """Pixel frames per latent frame, or None when the VAE is not temporal.

    comfy.sd.VAE derives this from ``upscale_ratio[0]``, which for every video
    VAE is ``lambda a: max(0, a * k - (k - 1))``, so k pixel frames per latent
    frame after the first, i.e. (T - 1) * k + 1 pixel frames for T latent
    frames. It returns None for image VAEs.
    """
    fn = getattr(vae, "temporal_compression_decode", None) if vae is not None else None
    if callable(fn):
        try:
            value = fn()
            if value:
                return max(1, int(value))
        except Exception as exc:
            logger.debug("[RadianceT2V] temporal_compression_decode() unusable: %s", exc)
    return None


def _expected_pixel_frames(latent_frames: int, temporal_compression: Optional[int]) -> int:
    """Pixel frames a causal video VAE returns for `latent_frames` latent frames.

    (T - 1) * k + 1, which is every video VAE's temporal upscale_ratio in
    comfy/sd.py bar one: the MiniMax H3 taehv uses (T - 2) // 5 * 17 + 5, and
    its temporal_compression_decode() rounds to 3. This is a cross-check for the
    report, never the returned count, so that VAE gets a MISMATCH note rather
    than a wrong number. Do not "correct" it by dropping the check.
    """
    if not temporal_compression or temporal_compression <= 1:
        return int(latent_frames)
    return max(0, int(latent_frames) * int(temporal_compression) - (int(temporal_compression) - 1))


# Inference only. .eval() does not clear requires_grad on parameters, so an
# unguarded forward still builds and retains an autograd graph.
@torch.no_grad()
def _vae_decode_tiled(vae, samples, tile_overlap_px: int,
                      spatial_compression: int,
                      temporal_compression: Optional[int]):
    """Decode through comfy.sd.VAE.decode_tiled(), the public tiled entry point.

    Signature (comfy/sd.py): decode_tiled(samples, tile_x=None, tile_y=None,
    overlap=None, tile_t=None, overlap_t=None). Every one of those is a LATENT
    size, not a pixel size. ComfyUI's VAEDecodeTiled divides each pixel-space
    widget by the VAE's compression before calling. Passing pixel sizes through
    unconverted makes the tile larger than the latent, so nothing tiles and the
    VRAM ceiling the operator was trying to stay under is unchanged.

    tile_t/overlap_t are omitted (None) for a non-temporal VAE; comfy's
    dispatcher then never reaches its 3-D branch, which is what we want for an
    image latent.

    Returns (frames, args) so the caller reports the arguments that were used
    rather than the ones it asked for: the overlap clamp below silently lowers
    any tile_overlap above 128px.
    """
    tile_px = _TILE_SIZE_PX
    overlap_px = max(0, int(tile_overlap_px))
    if tile_px < overlap_px * 4:
        overlap_px = tile_px // 4

    temporal_overlap_px = _TEMPORAL_OVERLAP_PX
    if _TEMPORAL_TILE_PX < temporal_overlap_px * 2:
        temporal_overlap_px = temporal_overlap_px // 2

    if temporal_compression:
        tile_t = max(2, _TEMPORAL_TILE_PX // temporal_compression)
        overlap_t = max(1, min(tile_t // 2, temporal_overlap_px // temporal_compression))
    else:
        tile_t = None
        overlap_t = None

    args = {
        "tile_x": max(1, tile_px // spatial_compression),
        "tile_y": max(1, tile_px // spatial_compression),
        "overlap": max(0, overlap_px // spatial_compression),
        "tile_t": tile_t,
        "overlap_t": overlap_t,
    }
    return vae.decode_tiled(samples, **args), args


def _frames_to_batch(frames):
    """Collapse a video VAE's (B, T, H, W, C) output to an IMAGE batch.

    comfy.sd.VAE.decode() and decode_tiled() both end with ``movedim(1, -1)``,
    so a 5-D result is (B, T, H, W, C), channels last already. Treating it as
    (B, C, T, H, W) and permuting (the shape this file used to assume) shuffled
    the axes and produced an unreadable preview.
    """
    if not HAS_TORCH or not isinstance(frames, torch.Tensor) or frames.dim() != 5:
        return frames
    b, t, h, w, c = frames.shape
    return frames.reshape(b * t, h, w, c)


def _require_video_model(model, node_name):
    """Refuse an image model before it reaches the sampler.

    Handed an image model (Flux, Z-Image, SDXL ...), the pipeline built a 5-D
    video latent and the failure surfaced deep inside the model's forward as
    "too many values to unpack (expected 4)", which names neither the node nor
    the cause. Found by the live 3.5 run. The latent format ComfyUI attaches to
    every loaded model says how many dimensions it samples in; a video model
    is 3 (T, H, W). Models that do not expose it are let through.
    """
    try:
        dims = model.model.latent_format.latent_dimensions
    except AttributeError:
        return
    if dims is not None and int(dims) < 3:
        fmt = type(model.model.latent_format).__name__
        raise ValueError(
            f"{node_name} needs a video model (Wan, LTX-Video, HunyuanVideo, "
            f"Cosmos, Mochi, CogVideoX, MiniMax H3). The connected model samples "
            f"{dims}-D latents ({fmt}), i.e. it is an image model."
        )


def _align_spec_to_model(spec, model, vae, report):
    """Make the noise shape match the connected model and VAE.

    Without a dit_config the pipelines defaulted to the LTX-Video spec
    (128 ch, /32), so a Wan or HunyuanVideo run built a latent the model
    could not take. The model's latent_format and the VAE's own compression
    are the truth; they win over the table whenever they are readable, and
    every override is written to the report.
    """
    spec = dict(spec)
    changes = []
    try:
        ch = int(model.model.latent_format.latent_channels)
        if ch and ch != spec.get("channels"):
            changes.append(f"channels {spec.get('channels')}->{ch}")
            spec["channels"] = ch
    except Exception:
        pass
    sc = 0
    fn = getattr(vae, "spacial_compression_decode", None) if vae is not None else None
    if callable(fn):
        try:
            sc = int(fn() or 0)
        except Exception:
            sc = 0
    if sc > 1 and sc != spec.get("spatial_compression"):
        changes.append(f"spatial /{spec.get('spatial_compression')}->/{sc}")
        spec["spatial_compression"] = sc
    tc = _vae_temporal_compression(vae)
    if tc and tc != spec.get("temporal_compression"):
        changes.append(f"temporal /{spec.get('temporal_compression')}->/{tc}")
        spec["temporal_compression"] = tc
    if changes:
        report.append("Latent spec from model/VAE: " + ", ".join(changes))
    return spec


def _model_concat_channels(model, latent_channels):
    """Extra input channels the diffusion model takes for image concat, or 0.

    Wan-family models expose it as patch_embedding in-channels minus the
    latent channels (Wan 2.1 I2V: 36 - 16 = 20). Text-to-video checkpoints and
    Wan 2.2 TI2V 5B have none. Unknown architectures report 0 so auto falls
    back to a strategy that works with any model.
    """
    try:
        dm = model.model.diffusion_model
        w = dm.patch_embedding.weight
        return max(int(w.shape[1]) - int(latent_channels), 0)
    except Exception:
        return 0


# ===========================================================================
# Node: RadianceVideoModelInfo
# ===========================================================================

class RadianceVideoModelInfo:
    """
    Inspect a ComfyUI MODEL object and produce a DiT config JSON describing
    its latent space.  Attempts to auto-detect the model type from its class
    name / config attributes; falls back to the user-selected preset.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "Video diffusion model to inspect. It is passed through unchanged "
                               "on the model output."}),
                "model_preset": (MODEL_NAMES, {"default": "LTX-Video (128ch)",
                    "tooltip": "Latent spec used when auto-detection finds nothing. Detection reads "
                               "the model class name (ltx, hunyuan, wan, cogvideo, mochi) and the latent "
                               "channel count the model reports. A preset from the detected family that "
                               "matches the channels is kept; otherwise the matching one is used."}),
            },
            "optional": {
                "override_channels": ("INT", {"default": 0, "min": 0, "max": 512,
                    "tooltip": "Replace the latent channel count written to dit_config. "
                               "0 keeps the preset's value."}),
                "override_latent_scale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0,
                    "tooltip": "Replace the preset's latent_scale in dit_config. Reported for reference; "
                               "Video Batch Decode does not apply it, since sampler output is already in "
                               "VAE space. 0 keeps the preset's value."}),
                "print_info": ("BOOLEAN", {"default": False,
                    "tooltip": "Also write the info report to the ComfyUI console log."}),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING", "STRING")
    RETURN_NAMES = ("model", "dit_config", "info_report")
    FUNCTION = "inspect"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Video"
    DESCRIPTION = "Display configuration and parameter info for a loaded video model."

    def inspect(self, model, model_preset: str,
                override_channels: int = 0,
                override_latent_scale: float = 0.0,
                print_info: bool = False):

        spec = dict(_get_spec(model_preset))
        spec["model_name"] = model_preset

        # Auto-detect from model class name
        cls_name = type(model).__name__.lower() if model is not None else ""
        model_inner = getattr(model, "model", None)
        inner_name  = type(model_inner).__name__.lower() if model_inner else ""
        combined    = cls_name + " " + inner_name

        # The class name picks the family; the latent channel count the model
        # itself reports (ComfyUI's latent_format) picks the preset inside it.
        # A preset the user chose from the right family is kept. Every Wan
        # model used to be forced to Wan2.1 (16ch), including the 48-channel
        # Wan2.2 TI2V 5B, over whatever model_preset said.
        families = (
            ("ltx", ("LTX-Video (128ch)",)),
            ("hunyuan", ("HunyuanVideo (16ch)", "HunyuanVideo-1.5 (32ch)")),
            ("wan", ("Wan2.1 (16ch)", "Wan2.2-T2V-14B (16ch)", "Wan2.2-I2V-14B (16ch)",
                     "Wan2.2-TI2V-5B (48ch)")),
            ("cogvideo", ("CogVideoX (16ch)",)),
            ("mochi", ("Mochi-1 (12ch)",)),
        )
        latent_format = getattr(model_inner, "latent_format", None)
        reported_ch = getattr(latent_format, "latent_channels", None)
        reported_ch = int(reported_ch) if isinstance(reported_ch, (int, float)) and reported_ch > 0 else None

        detected = None
        for key, presets in families:
            if key not in combined:
                continue
            presets = [p for p in presets if p in MODEL_NAMES]
            def fits(name):
                return reported_ch is None or int(_get_spec(name).get("channels", 0)) == reported_ch
            if model_preset in presets and fits(model_preset):
                detected = model_preset
            else:
                detected = next((p for p in presets if fits(p)), presets[0] if presets else None)
            break

        if detected and detected != model_preset:
            spec = dict(_get_spec(detected))
            spec["model_name"] = detected

        if override_channels > 0:
            spec["channels"] = override_channels
        if override_latent_scale > 0:
            spec["latent_scale"] = override_latent_scale

        info = [
            f"=== RadianceVideoModelInfo v{__version__} ===",
            f"Preset      : {model_preset}",
            f"Auto-detect : {detected or '(no match)'}",
            f"Final spec  :",
            f"  channels           = {spec['channels']}",
            f"  spatial_compression= {spec['spatial_compression']}",
            f"  temporal_compression= {spec.get('temporal_compression', 1)}",
            f"  latent_scale       = {spec['latent_scale']}",
            f"  has_temporal       = {spec.get('temporal', False)}",
        ]
        if print_info:
            logger.info("\n".join(info))

        defaults = _MODEL_DEFAULTS.get(spec["model_name"], {})
        spec["defaults"] = defaults

        return (model, json.dumps(spec), "\n".join(info))


# ===========================================================================
# Node: RadianceVideoLatentNoise
# ===========================================================================

class RadianceVideoLatentNoise:
    """
    Generate correctly-shaped Gaussian noise for a given DiT video model
    and target resolution / frame count.

    For temporal models the output shape is [B, C, T, H//sc, W//sc].
    For non-temporal (image) models it is [B, C, H//sc, W//sc].
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dit_config": ("STRING", {
                    "default": "{}",
                    "tooltip": "JSON from Video Model Info",
                }),
                "width":  ("INT", {"default": 512,  "min": 64, "max": 4096, "step": 8,
                    "tooltip": "Target frame width in pixels. Divided (rounded down) by the spec's "
                               "spatial compression to size the latent."}),
                "height": ("INT", {"default": 512,  "min": 64, "max": 4096, "step": 8,
                    "tooltip": "Target frame height in pixels. Divided (rounded down) by the spec's "
                               "spatial compression to size the latent."}),
                "frames": ("INT", {"default": 25,   "min": 1,  "max": 512,
                    "tooltip": "Target pixel frames. The latent gets ceil(frames / temporal compression) "
                               "frames; use a multiple of the compression plus 1 (e.g. 25, 49) to decode "
                               "back to exactly this count."}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 16,
                    "tooltip": "Number of independent noise samples in the batch."}),
                "seed":   ("INT", {"default": 0,   "min": 0,  "max": 2**31,
                    "tooltip": "Seed for the CPU noise generator; the same seed and shape give "
                               "identical noise."}),
            },
            "optional": {
                "noise_scale": ("FLOAT", {
                    "default": 1.0, "min": 0.01, "max": 4.0, "step": 0.01,
                    "tooltip": "Multiply noise standard deviation (1.0 = unit Gaussian)",
                }),
            },
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("noise_latent", "shape_report")
    FUNCTION = "generate"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Video"
    DESCRIPTION = ("Correctly shaped, seeded i.i.d. Gaussian latent noise for a video model "
                   "(independent per frame, as every video sampler expects).")

    def generate(self, dit_config: str, width: int, height: int, frames: int,
                 batch_size: int, seed: int, noise_scale: float = 1.0):

        try:
            cfg = json.loads(dit_config)
        except Exception:
            cfg = {}

        spec = _get_spec(cfg.get("model_name", "")) if cfg else _get_spec("LTX-Video (128ch)")
        if cfg:
            spec.update(cfg)

        shape = _build_noise_shape(spec, width, height, frames, batch_size)
        sc = spec.get("spatial_compression", 8)  # for report
        tc = spec.get("temporal_compression", 4)  # for report

        noise = _make_noise(shape, seed)
        if noise is not None:
            noise = noise * noise_scale

        report = [
            f"=== RadianceVideoLatentNoise ===",
            f"Model     : {cfg.get('model_name', 'unknown')}",
            f"Input     : {width}×{height}  {frames}f",
            f"Latent    : {list(shape)}",
            f"SC×TC     : {sc}×{tc}",
            f"Seed      : {seed}",
        ]

        if noise is None:
            import numpy as np_
            noise_np = np_.random.default_rng(seed).standard_normal(shape).astype("float32")
            noise = noise_np   # numpy path for non-torch environments
            return ({"samples": noise}, "\n".join(report))

        return (_wrap(noise), "\n".join(report))


# ===========================================================================
# Node: RadianceVideoCondMerge
# ===========================================================================

class RadianceVideoCondMerge:
    """
    Merge up to three conditioning inputs (text, character, HDR) into a
    single positive conditioning tensor.

    Merge modes
    -----------
    concat    — append tokens from each conditioning in order
    weighted  — scale each conditioning by its weight then sum / average
    priority  — use the first non-empty conditioning as base, patch in others
    """

    MODES = ["concat", "weighted", "priority"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_conditioning": ("CONDITIONING", {
                    "tooltip": "Base conditioning, usually the encoded prompt. The output keeps its "
                               "entries; the other inputs are merged into them."}),
                "merge_mode": (cls.MODES, {"default": "concat",
                    "tooltip": "concat: append the other inputs' tokens (times their weights) after the "
                               "text tokens. weighted: weighted average of the token tensors. priority: "
                               "keep the text tokens, only copy missing dict keys from the others."}),
            },
            "optional": {
                "character_conditioning": ("CONDITIONING", {
                    "tooltip": "Optional conditioning (e.g. a character or identity embedding) merged "
                               "per merge_mode."}),
                "hdr_conditioning": ("CONDITIONING", {
                    "tooltip": "Optional conditioning (e.g. from Video HDR Conditioner) merged "
                               "per merge_mode."}),
                "text_weight":      ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Weight of text_conditioning in weighted mode. concat and priority keep "
                               "the text tokens unscaled."}),
                "character_weight": ("FLOAT", {"default": 0.75,"min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Multiplier on the character tokens in concat and weighted modes. "
                               "Ignored in priority mode."}),
                "hdr_weight":       ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Multiplier on the HDR tokens in concat and weighted modes. "
                               "Ignored in priority mode."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("merged_conditioning", "merge_report")
    FUNCTION = "merge"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Video"
    DESCRIPTION = "Merge multiple video conditioning signals into a unified tensor."

    def merge(self, text_conditioning, merge_mode,
              character_conditioning=None, hdr_conditioning=None,
              text_weight=1.0, character_weight=0.75, hdr_weight=0.5):

        sources = [
            (text_conditioning,      text_weight,      "text"),
            (character_conditioning, character_weight, "character"),
            (hdr_conditioning,       hdr_weight,       "hdr"),
        ]
        sources = [(c, w, n) for c, w, n in sources if c is not None]

        report = [f"=== RadianceVideoCondMerge === mode={merge_mode}",
                  f"Sources: {[n for _,_,n in sources]}"]

        if not sources:
            return (text_conditioning, "\n".join(report))

        if len(sources) == 1 or not HAS_TORCH:
            return (sources[0][0], "\n".join(report))

        if merge_mode == "concat":
            merged = self._concat(sources)
        elif merge_mode == "weighted":
            merged = self._weighted(sources)
        else:
            merged = self._priority(sources)

        report.append(f"Output pairs: {len(merged)}")
        return (merged, "\n".join(report))

    def _concat(self, sources):
        """Concatenate token sequences from all sources."""
        result = []
        # Use the text conditioning as base pairs
        base_cond, base_weight, _ = sources[0]
        for i, (base_t, base_d) in enumerate(base_cond):
            new_d = dict(base_d)
            tokens = base_t
            for extra_cond, w, name in sources[1:]:
                if i < len(extra_cond):
                    ext_t, ext_d = extra_cond[i]
                    # Project extra tokens to same token_dim if needed
                    if ext_t.shape[-1] != tokens.shape[-1]:
                        try:
                            ext_t = F.interpolate(
                                ext_t.float().unsqueeze(0),
                                size=(ext_t.shape[-2], tokens.shape[-1]),
                                mode="bilinear", align_corners=False,
                            ).squeeze(0).to(tokens.dtype)
                        except Exception:
                            continue
                    tokens = torch.cat([tokens, ext_t * w], dim=1)
                    # Merge extra dicts
                    for k, v in ext_d.items():
                        if k not in new_d:
                            new_d[k] = v
            result.append([tokens, new_d])
        return result

    def _weighted(self, sources):
        """Weighted sum of token tensors (mean-pooled to same length)."""
        result = []
        base_cond, base_weight, _ = sources[0]
        for i, (base_t, base_d) in enumerate(base_cond):
            acc = base_t * base_weight
            new_d = dict(base_d)
            total_w = base_weight
            for extra_cond, w, name in sources[1:]:
                if i < len(extra_cond):
                    ext_t, ext_d = extra_cond[i]
                    # Pool or pad to match base seq_len
                    tgt_len = acc.shape[1]
                    if ext_t.shape[1] != tgt_len:
                        ext_t = F.adaptive_avg_pool1d(
                            ext_t.float().permute(0, 2, 1),
                            tgt_len,
                        ).permute(0, 2, 1).to(acc.dtype)
                    acc = acc + ext_t * w
                    total_w += w
            result.append([acc / max(total_w, 1e-8), new_d])
        return result

    def _priority(self, sources):
        """Use first source as base, patch extra dict keys from others."""
        result = []
        base_cond, _, _ = sources[0]
        for i, (base_t, base_d) in enumerate(base_cond):
            new_d = dict(base_d)
            for extra_cond, _, name in sources[1:]:
                if i < len(extra_cond):
                    _, ext_d = extra_cond[i]
                    for k, v in ext_d.items():
                        if k not in new_d:
                            new_d[k] = v
            result.append([base_t, new_d])
        return result


# ===========================================================================
# Node: RadianceVideoSampler
# ===========================================================================

class RadianceVideoSampler:
    """
    Low-level video sampler — shared engine behind T2V and I2V pipelines.

    Accepts pre-built noise + conditioning and runs the ComfyUI sampling
    loop via _comfy_sample(). An optional cfg_schedule_json (a JSON float
    array) overrides the static CFG value with its first value. Tiling is
    automatically restored after sampling.
    """
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Video"
    DESCRIPTION = "Run the diffusion sampler to generate video latents from pre-built noise and conditioning."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Video diffusion model to sample with."}),
                "positive": ("CONDITIONING", {"tooltip": "Positive (prompt) conditioning."}),
                "negative": ("CONDITIONING", {"tooltip": "Negative conditioning, used by classifier-free "
                                                         "guidance."}),
                "latent_noise": ("LATENT", {
                    "tooltip": "Noise latent, e.g. from Video Latent Noise. It is used as both the "
                               "noise and the start latent, so its shape must match the model."}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 200,
                    "tooltip": "Sampling steps. Ignored when dit_config carries a model_name."}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 30.0, "step": 0.1,
                    "tooltip": "Classifier-free guidance scale. Replaced by the model default when "
                               "dit_config carries a model_name, then by the first value of "
                               "cfg_schedule_json."}),
                "sampler_name": (_SAMPLERS, {"default": "euler",
                    "tooltip": "ComfyUI sampler. Ignored when dit_config carries a model_name."}),
                "scheduler": (_SCHEDULERS, {"default": "normal",
                    "tooltip": "ComfyUI sigma scheduler. Ignored when dit_config carries a model_name."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31,
                    "tooltip": "Seed for the sampler's own noise (ancestral and SDE samplers). The "
                               "initial noise comes from latent_noise."}),
            },
            "optional": {
                # ALBABIT-FIX: dit_config promoted from required to optional.
                # When connected (RadianceVideoModelInfo), model-specific defaults
                # (steps/cfg/sampler/scheduler) override the manual widgets.
                # When absent, widget values are used as-is.
                "dit_config": ("STRING", {
                    "default": "{}",
                    "tooltip": "JSON from Video Model Info — when connected, overrides steps/cfg/sampler/scheduler with model-specific defaults.",
                }),
                "cfg_schedule_json": ("STRING", {
                    "default": "",
                    "tooltip": "JSON float array, e.g. [4.0]. Its first value overrides CFG.",
                }),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Fraction of the noise schedule to run (1.0 = full). The start latent is "
                               "latent_noise itself, so below 1.0 this is not a video-to-video strength."}),
                # The "tiling" widget is deliberately not offered. It drove
                # `model.model.set_tiling(True)`, and no ComfyUI version defines
                # set_tiling (or enable_tiling) on a model or a VAE, so the probe
                # never matched, so the control reported "Tiling: enabled" while
                # changing nothing. ComfyUI has no model-level tiled sampling to
                # wire it to; tiling exists only on the VAE decode, which
                # RadianceVideoBatchDecode exposes. `tiling` stays in the
                # signature below so a saved workflow that still sends it loads,
                # and setting it says plainly that it does nothing.
            },
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("samples", "sampler_report")
    FUNCTION = "sample"

    def sample(self, model, positive, negative, latent_noise,
               steps, cfg, sampler_name, scheduler, seed,
               dit_config="{}", cfg_schedule_json="", denoise=1.0, tiling=False):

        # ALBABIT-FIX: resolve effective sampling params — dit_config wins when connected.
        # A real dit_config always carries a "model_name" key from RadianceVideoModelInfo.
        # An absent/empty dit_config ("{}") means the node is not connected → use widgets.
        try:
            _dc = json.loads(dit_config) if dit_config.strip() not in ("", "{}") else {}
        except Exception:
            _dc = {}

        if _dc.get("model_name"):
            _, model_name, eff_steps, eff_cfg, eff_sampler, eff_sched = _resolve_dit_config(
                dit_config, 0, 0.0, "", "",
            )
            _from_dit_config = True
        else:
            eff_steps, eff_cfg, eff_sampler, eff_sched = steps, cfg, sampler_name, scheduler
            model_name = "manual"
            _from_dit_config = False

        # Parse CFG schedule — applied on top of whichever source won above
        cfg_eff = eff_cfg
        _cfg_from_schedule = False  # ALBABIT-FIX: track actual parse success for the report label
        if cfg_schedule_json.strip():
            try:
                parsed = json.loads(cfg_schedule_json)
                if isinstance(parsed, (list, tuple)) and len(parsed) > 0:
                    cfg_eff = float(parsed[0])
                    _cfg_from_schedule = True
            except Exception as _cfg_err:
                # ALBABIT-FIX: warn instead of silently falling back to static CFG
                logger.warning(
                    "[RadianceVideoSampler] cfg_schedule_json parse failed (%s) — "
                    "using static CFG value %.2f.", _cfg_err, eff_cfg,
                )

        _cfg_src = " (schedule)" if _cfg_from_schedule else (" (dit_config)" if _from_dit_config else "")
        report = [
            f"=== RadianceVideoSampler v{__version__} ===",
            f"Model    : {model_name}",
            f"Sampler  : {eff_sampler}",
            f"Scheduler: {eff_sched}",
            f"Steps    : {eff_steps}",
            f"CFG      : {cfg_eff}{_cfg_src}",
            f"Denoise  : {denoise}",
            f"Seed     : {seed}",
        ]

        if not HAS_TORCH:
            report.append("torch unavailable — returning noise unchanged")
            return (latent_noise, "\n".join(report))

        # `tiling` is retired, not honoured: the old probe looked for
        # model.model.set_tiling(), which exists in no ComfyUI release, so the
        # branch never ran and the "Tiling: enabled" line above it was a lie.
        # A workflow saved before the widget was withdrawn can still send the
        # value, so say what happened to it rather than ignoring it silently.
        if tiling:
            report.append(
                "Tiling: NOT APPLIED, ComfyUI has no model-level tiled sampling; "
                "this input is retired. Tile the VAE decode instead "
                "(RadianceVideoBatchDecode.tile_decode)."
            )

        noise = _to_tensor(latent_noise)
        samples = _comfy_sample(
            model, noise, eff_steps, cfg_eff, eff_sampler, eff_sched,
            positive, negative, latent_noise, denoise=denoise, seed=seed,
        )
        report.append(f"Output shape: {list(samples.shape) if HAS_TORCH else 'N/A'}")
        return (_wrap(samples), "\n".join(report))


# ===========================================================================
# Node: RadianceT2VPipeline
# ===========================================================================

class RadianceT2VPipeline:
    """
    Full Text-to-Video pipeline — one node from prompt to video IMAGE batch.

    Sampling is performed via _comfy_sample() — the same module-level helper
    used by RadianceVideoSampler. Per-model defaults (sampler, scheduler,
    steps, CFG) are resolved from dit_config via _resolve_dit_config().
    You can override every parameter explicitly.
    """
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Video"
    DESCRIPTION = "End-to-end text-to-video generation pipeline with HDR support."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Video diffusion model. Image models (2-D latents) "
                                               "are rejected."}),
                "clip": ("CLIP", {"tooltip": "Text encoder matching the model, used for both prompts."}),
                "vae": ("VAE", {"tooltip": "VAE matching the model. Its compression sets the latent "
                                           "shape and it decodes preview_frames."}),
                "positive_prompt": ("STRING", {
                    "multiline": True,
                    "default": "cinematic HDR video, stunning visuals, 4K, film grain",
                    "tooltip": "What to generate. HDR descriptors from peak_nits, target_gamut and "
                               "hdr_eotf are appended unless hdr_strength is 0.",
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "watermark, blurry, low quality, sdr, flickering",
                    "tooltip": "What to steer away from, encoded with the same text encoder.",
                }),
                "width":  ("INT", {"default": 768,  "min": 64, "max": 4096, "step": 8,
                    "tooltip": "Output width in pixels, rounded down to a multiple of the VAE's "
                               "spatial compression."}),
                "height": ("INT", {"default": 512,  "min": 64, "max": 4096, "step": 8,
                    "tooltip": "Output height in pixels, rounded down to a multiple of the VAE's "
                               "spatial compression."}),
                "frames": ("INT", {"default": 25,   "min": 1,  "max": 512,
                    "tooltip": "Requested frames. The latent holds ceil(frames / temporal compression) "
                               "frames; use a multiple of the compression plus 1 (e.g. 25, 49, 81) to "
                               "get exactly this count back."}),
                "seed":   ("INT", {"default": 0,    "min": 0,  "max": 2**31,
                    "tooltip": "Seed for the initial noise and the sampler."}),
            },
            "optional": {
                "dit_config": ("STRING", {"default": "{}",
                    "tooltip": "JSON from Video Model Info. When it carries a model_name, that "
                               "model's defaults replace steps, cfg, sampler_name and scheduler."}),
                "character_conditioning": ("CONDITIONING", {
                    "tooltip": "Optional conditioning whose tokens are appended to the positive prompt "
                               "at weight 0.75. Skipped if its embedding width differs."}),
                "cfg_schedule_json": ("STRING", {"default": "",
                    "tooltip": "JSON float array, e.g. [4.0]. Only the first value is "
                               "used, as a static CFG override; CFG does not vary per step."}),
                "steps": ("INT", {"default": 0, "min": 0, "max": 200,
                    "tooltip": "Sampling steps. 0 uses the model default (the LTX-Video preset's 25 when "
                               "no dit_config is connected). Ignored when dit_config carries a model_name."}),
                "cfg":   ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0, "step": 0.1,
                    "tooltip": "Guidance scale. 0 uses the model default (the LTX-Video preset's 3.5 when "
                               "no dit_config is connected). cfg_schedule_json overrides it."}),
                "sampler_name": (_SAMPLERS, {"default": "euler",
                    "tooltip": "ComfyUI sampler. Ignored when dit_config carries a model_name."}),
                "scheduler":    (_SCHEDULERS, {"default": "normal",
                    "tooltip": "ComfyUI sigma scheduler. Ignored when dit_config carries a model_name."}),
                "peak_nits":    ([str(n) for n in [100,203,400,600,1000,4000,10000]],
                                  {"default": "1000",
                                   "tooltip": "Adds '<n> nits HDR' to the prompt. Prompt text only, no "
                                              "pixel change; 100 requests an SDR look."}),
                "target_gamut": (["BT.2020","P3-D65","P3-DCI","BT.709","ACEScg"],
                                  {"default": "BT.2020",
                                   "tooltip": "Adds the selected gamut descriptor to the prompt. "
                                              "Prompt text only, no pixel conversion."}),
                "hdr_eotf":     (["PQ (ST.2084)","HLG (BT.2100)","Linear","sRGB / BT.1886"],
                                  {"default": "PQ (ST.2084)",
                                   "tooltip": "Adds a transfer-function descriptor to the prompt. Prompt "
                                              "text only, no pixel encoding."}),
                "hdr_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Prompt weight of the appended HDR descriptors (gamut, EOTF, peak nits), "
                               "as (text:weight) with weight = 2 x strength: 0.5 is neutral, "
                               "1.0 doubles their emphasis, 0 leaves them out."}),
            },
        }

    RETURN_TYPES = ("LATENT", "IMAGE", "CONDITIONING", "STRING")
    RETURN_NAMES = ("video_latent", "preview_frames", "positive_cond", "pipeline_report")
    FUNCTION = "generate"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Video"

    def generate(self, model, clip, vae, positive_prompt, negative_prompt,
                 width, height, frames, seed,
                 dit_config="{}", character_conditioning=None,
                 cfg_schedule_json="", steps=0, cfg=0.0,
                 sampler_name="euler", scheduler="normal",
                 peak_nits="1000", target_gamut="BT.2020",
                 hdr_eotf="PQ (ST.2084)", hdr_strength=0.5):
        _require_video_model(model, "RadianceT2VPipeline")

        report = [f"=== RadianceT2VPipeline v{__version__} ===",
                  f"Prompt : {positive_prompt[:80]}...",
                  f"Size   : {width}×{height}  {frames}f  seed={seed}"]

        # ALBABIT-FIX: dit_config wins when connected (model_name key present),
        # mirroring RadianceVideoSampler.sample() — JS greys out manual widgets.
        try:
            _dc = json.loads(dit_config) if dit_config.strip() not in ("", "{}") else {}
        except Exception:
            _dc = {}
        if _dc.get("model_name"):
            spec, model_name, eff_steps, eff_cfg, eff_sampler, eff_sched = _resolve_dit_config(
                dit_config, 0, 0.0, "", "")
            _from_dit_config = True
        else:
            spec, model_name, eff_steps, eff_cfg, eff_sampler, eff_sched = _resolve_dit_config(
                dit_config, steps, cfg, sampler_name, scheduler)
            _from_dit_config = False

        # ALBABIT-FIX: cfg_schedule_json was accepted but silently ignored — mirrors
        # RadianceVideoSampler.sample() parsing logic.
        cfg_eff = eff_cfg
        _cfg_from_schedule = False
        if cfg_schedule_json.strip():
            try:
                _parsed = json.loads(cfg_schedule_json)
                if isinstance(_parsed, (list, tuple)) and len(_parsed) > 0:
                    cfg_eff = float(_parsed[0])
                    _cfg_from_schedule = True
            except Exception as _cfg_err:
                logger.warning(
                    "[RadianceT2VPipeline] cfg_schedule_json parse failed (%s) — "
                    "using static CFG %.2f.", _cfg_err, eff_cfg,
                )

        _cfg_src = " (schedule)" if _cfg_from_schedule else (" (dit_config)" if _from_dit_config else "")
        report.append(f"Model  : {model_name}")
        report.append(f"Steps={eff_steps}  CFG={cfg_eff}{_cfg_src}  {eff_sampler}/{eff_sched}")
        spec = _align_spec_to_model(spec, model, vae, report)

        # --- Build positive conditioning ---
        hdr_tokens = self._hdr_tokens(peak_nits, target_gamut, hdr_eotf, hdr_strength)
        full_prompt = positive_prompt.strip()
        if hdr_tokens:
            full_prompt = full_prompt.rstrip(", ") + ", " + hdr_tokens

        pos_cond = self._encode_text(clip, full_prompt)
        neg_cond = self._encode_text(clip, negative_prompt)

        # Merge character conditioning if provided
        if character_conditioning is not None and pos_cond is not None:
            pos_cond = self._merge_cond(pos_cond, character_conditioning, 0.75)

        # --- Generate noise ---
        noise_shape = _build_noise_shape(spec, width, height, frames)

        noise = _make_noise(noise_shape, seed)
        if noise is None:
            report.append("ERROR: torch unavailable")
            return (_wrap(self._zero_latent(noise_shape)),
                    self._zero_image(height, width, frames),
                    pos_cond or [], "\n".join(report))

        noise_latent = _wrap(noise)
        report.append(f"Noise shape: {list(noise_shape)}")

        # --- Sample ---
        samples = _comfy_sample(
            model, noise, eff_steps, cfg_eff, eff_sampler, eff_sched,
            pos_cond or [], neg_cond or [], noise_latent,
            denoise=1.0, seed=seed,
        )
        video_latent = _wrap(samples)
        report.append(f"Sampled: {list(samples.shape)}")

        # --- Decode preview ---
        preview, preview_error = self._decode_preview(vae, video_latent, frames)
        if preview_error:
            report.append(
                f"Preview frames: DECODE FAILED ({preview_error}). "
                f"preview_frames is a black placeholder, not pixels. "
                f"video_latent is unaffected; decode it with "
                f"RadianceVideoBatchDecode (tile_decode=True for long clips)."
            )
        else:
            report.append(f"Preview frames: {preview.shape[0]}")

        return (video_latent, preview, pos_cond or [], "\n".join(report))

    # ------------------------------------------------------------------
    # Inference only. .eval() does not clear requires_grad on parameters, so an
# unguarded forward still builds and retains an autograd graph.
    @torch.no_grad()
    def _encode_text(self, clip, text):
        if clip is None or not HAS_TORCH:
            return []
        try:
            tokens = clip.tokenize(text)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            return [[cond, {"pooled_output": pooled}]]
        except Exception as exc:
            logger.warning(f"[T2V] text encode failed: {exc}")
            return []

    def _hdr_tokens(self, peak_nits, gamut, eotf, strength):
        if strength < 0.01:
            return ""
        text = _hdr_prompt_descriptors(peak_nits, gamut, eotf)
        if not text:
            return ""
        weight = round(2.0 * float(strength), 2)
        # ComfyUI's tokenizers parse (text:weight) for CLIP and T5 alike.
        return text if abs(weight - 1.0) < 1e-6 else f"({text}:{weight})"

    def _merge_cond(self, base, extra, weight):
        if not HAS_TORCH:
            return base
        merged = []
        for i, (bt, bd) in enumerate(base):
            new_d = dict(bd)
            if i < len(extra):
                et, _ = extra[i]
                if et.shape[-1] == bt.shape[-1]:
                    tok = torch.cat([bt, et * weight], dim=1)
                else:
                    tok = bt
            else:
                tok = bt
            merged.append([tok, new_d])
        return merged

    # VAE decode for the preview image -- no gradients are ever needed here.

    @torch.no_grad()

    def _decode_preview(self, vae, latent, target_frames):
        """Decode the preview batch. Returns (frames, failure_text_or_None).

        The preview is secondary to video_latent, so a decode failure must not
        throw away a finished sample, but the caller has to know it happened.
        The old bare `except Exception` returned a 64x64 black batch that the
        report then counted as "Preview frames: N", so an OOM on a long clip
        left black tiles labelled as a success.
        """
        if not HAS_TORCH:
            return self._zero_image(64, 64, target_frames), "torch unavailable"
        try:
            decoded = _vae_decode(vae, latent)
            # comfy.sd.VAE.decode() returns (B, T, H, W, C) for a video VAE: it
            # movedim(1, -1)s the channel axis before returning. Reading it as
            # (B, C, T, H, W) and permuting scrambled the preview.
            decoded = _frames_to_batch(decoded)
            return decoded.clamp(0, 1), None
        except Exception as exc:
            logger.error("[RadianceT2VPipeline] preview decode failed: %s", exc)
            return (self._zero_image(64, 64, target_frames),
                    f"{type(exc).__name__}: {exc}")

    def _zero_latent(self, shape):
        if HAS_TORCH:
            return torch.zeros(shape)
        return shape

    def _zero_image(self, h, w, n):
        if HAS_TORCH:
            return torch.zeros(n, h, w, 3)
        return None


# ===========================================================================
# Node: RadianceI2VPipeline
# ===========================================================================

class RadianceI2VPipeline:
    """
    Full Image-to-Video pipeline — one reference image → video IMAGE batch.

    Sampling is performed via _comfy_sample() — the same module-level helper
    used by RadianceVideoSampler and RadianceT2VPipeline.

    I2V conditioning strategy (applied automatically by model type)
    ---------------------------------------------------------------
    LTX-Video      — reference image injected via model's img2vid API;
                     first latent frame is clamped to the image encoding
    HunyuanVideo   — encoded image latent concatenated channel-wise to noise
    Wan2.1         — CLIP vision embedding injected into cross-attention
    CogVideoX      — reference latent prepended to the video latent sequence
    Generic        — image encoded and blended into the first latent frame

    Outputs
    -------
    video_latent   — raw DiT latent (route to RadianceVideoHDRDecode)
    preview_frames — quick VAE-decoded IMAGE batch
    pipeline_report — diagnostic text
    """
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Video"
    DESCRIPTION = "End-to-end image-to-video generation pipeline with motion control."

    # concat_channels and clip_vision_inject write the conditioning keys
    # ComfyUI's own WanImageToVideo writes (concat_latent_image + concat_mask,
    # clip_vision_output), so a model with an image-concat input or a CLIP
    # vision projection reads them. first_frame_lock and prepend_latent blend
    # the image latent into the starting noise and work with any video model.
    I2V_STRATEGIES = [
        "auto", "first_frame_lock", "concat_channels", "clip_vision_inject",
        "prepend_latent",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Video diffusion model. Image models are rejected; a "
                                               "model with an image-concat input (Wan I2V) enables "
                                               "concat_channels."}),
                "clip": ("CLIP", {"tooltip": "Text encoder matching the model, used for both prompts."}),
                "vae": ("VAE", {"tooltip": "VAE matching the model. Encodes the reference image, sets "
                                           "the latent shape and decodes preview_frames."}),
                "reference_image": ("IMAGE", {
                    "tooltip": "Start frame (display-referred sRGB, one image). Its width and height set "
                               "the video size, rounded down to the VAE's spatial compression."}),
                "positive_prompt": ("STRING", {
                    "multiline": True,
                    "default": "smooth camera motion, cinematic HDR, 4K",
                    "tooltip": "What should happen in the shot. ', <n> nits HDR, <gamut>' is appended "
                               "when peak_nits is above 100.",
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "watermark, blurry, flickering, sdr",
                    "tooltip": "What to steer away from, encoded with the same text encoder.",
                }),
                "frames": ("INT", {"default": 25, "min": 1, "max": 512,
                    "tooltip": "Requested frames. The latent holds ceil(frames / temporal compression) "
                               "frames; use a multiple of the compression plus 1 (e.g. 25, 49, 81) to "
                               "get exactly this count back."}),
                "seed":   ("INT", {"default": 0,  "min": 0, "max": 2**31,
                    "tooltip": "Seed for the initial noise and the sampler."}),
            },
            "optional": {
                "dit_config": ("STRING", {"default": "{}",
                    "tooltip": "JSON from Video Model Info. When it carries a model_name, that "
                               "model's defaults replace steps, cfg, sampler_name and scheduler."}),
                "character_conditioning": ("CONDITIONING", {
                    "tooltip": "Optional conditioning whose tokens are appended to the positive prompt "
                               "at weight 0.75. Skipped if its embedding width differs."}),
                "cfg_schedule_json": ("STRING", {"default": "",
                    "tooltip": "JSON float array. Only the first value is used, as a static CFG "
                               "override; CFG does not vary per step."}),
                "i2v_strategy": (cls.I2V_STRATEGIES, {"default": "auto",
                    "tooltip": "auto: concat_channels when the model has an image-concat input "
                               "(Wan 2.1 I2V, Wan 2.2 I2V 14B), else first_frame_lock. "
                               "clip_vision_inject needs clip_vision_output connected."}),
                "clip_vision_output": ("CLIP_VISION_OUTPUT", {
                    "tooltip": "From CLIP Vision Encode. Added to both conditionings as "
                               "clip_vision_output, which Wan 2.1 I2V reads (clip_fea)."}),
                "image_strength": ("FLOAT", {
                    "default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "first_frame_lock and prepend_latent only: blend weight of the image "
                               "latent into the first latent frame; also lowers denoise to "
                               "1 - 0.4 x strength. Ignored by concat_channels and clip_vision_inject.",
                }),
                "motion_strength": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "first_frame_lock and prepend_latent only: scales the starting noise "
                               "to 0.2 + 0.8 x strength of its unit amplitude. Ignored by "
                               "concat_channels and clip_vision_inject.",
                }),
                "steps": ("INT", {"default": 0, "min": 0, "max": 200,
                    "tooltip": "Sampling steps. 0 uses the model default (the LTX-Video preset's 25 when "
                               "no dit_config is connected). Ignored when dit_config carries a model_name."}),
                "cfg":   ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0, "step": 0.1,
                    "tooltip": "Guidance scale. 0 uses the model default (the LTX-Video preset's 3.5 when "
                               "no dit_config is connected). cfg_schedule_json overrides it."}),
                "sampler_name": (_SAMPLERS, {"default": "euler",
                    "tooltip": "ComfyUI sampler. Ignored when dit_config carries a model_name."}),
                "scheduler":    (_SCHEDULERS, {"default": "normal",
                    "tooltip": "ComfyUI sigma scheduler. Ignored when dit_config carries a model_name."}),
                "peak_nits":    ([str(n) for n in [100,203,400,600,1000,4000,10000]],
                                  {"default": "1000",
                                   "tooltip": "Adds the selected peak brightness to the prompt; 100 "
                                              "requests an SDR look. No pixel change."}),
                "target_gamut": (["BT.2020","P3-D65","P3-DCI","BT.709","ACEScg"],
                                  {"default": "BT.2020",
                                   "tooltip": "Adds the selected gamut descriptor to the prompt at "
                                              "every peak brightness. No pixel conversion."}),
                "hdr_eotf":     (["PQ (ST.2084)","HLG (BT.2100)","Linear","sRGB / BT.1886"],
                                  {"default": "PQ (ST.2084)",
                                   "tooltip": "Adds the selected transfer-function descriptor to "
                                              "the prompt. No pixel encoding."}),
            },
        }

    RETURN_TYPES = ("LATENT", "IMAGE", "STRING")
    RETURN_NAMES = ("video_latent", "preview_frames", "pipeline_report")
    FUNCTION = "generate"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Video"

    def generate(self, model, clip, vae, reference_image,
                 positive_prompt, negative_prompt, frames, seed,
                 dit_config="{}", character_conditioning=None,
                 cfg_schedule_json="", i2v_strategy="auto",
                 clip_vision_output=None,
                 image_strength=0.85, motion_strength=0.5,
                 steps=0, cfg=0.0, sampler_name="euler", scheduler="normal",
                 peak_nits="1000", target_gamut="BT.2020", hdr_eotf="PQ (ST.2084)"):
        _require_video_model(model, "RadianceI2VPipeline")

        report = [f"=== RadianceI2VPipeline v{__version__} ===",
                  f"Strategy       : {i2v_strategy}",
                  f"Image strength : {image_strength}",
                  f"Motion strength: {motion_strength}",
                  f"Frames         : {frames}  seed={seed}"]

        # ALBABIT-FIX: dit_config wins when connected (model_name key present),
        # mirroring RadianceVideoSampler.sample() — JS greys out manual widgets.
        try:
            _dc = json.loads(dit_config) if dit_config.strip() not in ("", "{}") else {}
        except Exception:
            _dc = {}
        if _dc.get("model_name"):
            spec, model_name, eff_steps, eff_cfg, eff_sampler, eff_sched = _resolve_dit_config(
                dit_config, 0, 0.0, "", "")
            _from_dit_config = True
        else:
            spec, model_name, eff_steps, eff_cfg, eff_sampler, eff_sched = _resolve_dit_config(
                dit_config, steps, cfg, sampler_name, scheduler)
            _from_dit_config = False

        # ALBABIT-FIX: cfg_schedule_json was accepted but silently ignored — mirrors
        # RadianceVideoSampler.sample() parsing logic.
        cfg_eff = eff_cfg
        _cfg_from_schedule = False
        if cfg_schedule_json.strip():
            try:
                _parsed = json.loads(cfg_schedule_json)
                if isinstance(_parsed, (list, tuple)) and len(_parsed) > 0:
                    cfg_eff = float(_parsed[0])
                    _cfg_from_schedule = True
            except Exception as _cfg_err:
                logger.warning(
                    "[RadianceI2VPipeline] cfg_schedule_json parse failed (%s) — "
                    "using static CFG %.2f.", _cfg_err, eff_cfg,
                )

        _cfg_src = " (schedule)" if _cfg_from_schedule else (" (dit_config)" if _from_dit_config else "")
        report.append(f"Model  : {model_name}")
        report.append(f"Steps={eff_steps}  CFG={cfg_eff}{_cfg_src}  {eff_sampler}/{eff_sched}")
        spec = _align_spec_to_model(spec, model, vae, report)

        # Resolve I2V strategy from what the model can read, not its name.
        concat_ch = _model_concat_channels(model, spec.get("channels", 16))
        strategy = i2v_strategy
        if strategy == "auto":
            strategy = "concat_channels" if concat_ch else "first_frame_lock"
        if strategy == "concat_channels" and not concat_ch:
            raise ValueError(
                "i2v_strategy concat_channels needs a model with an image-concat input "
                "(Wan 2.1 I2V, Wan 2.2 I2V 14B, ...). The connected model has none, so "
                "the image would be ignored. Use first_frame_lock or auto.")
        if strategy == "clip_vision_inject" and clip_vision_output is None:
            raise ValueError(
                "i2v_strategy clip_vision_inject needs clip_vision_output connected "
                "(Load CLIP Vision -> CLIP Vision Encode). Nothing was injected.")
        report.append(f"Resolved strategy: {strategy}"
                      + (f" (model concat input: {concat_ch} ch)" if concat_ch else ""))

        _, img_h, img_w, _ = reference_image.shape
        noise_shape = _build_noise_shape(spec, img_w, img_h, frames)
        ch = spec.get("channels", 16)  # needed by _first_frame_lock

        noise = _make_noise(noise_shape, seed)
        if noise is None:
            report.append("ERROR: torch unavailable")
            return (_wrap(self._zero(noise_shape)),
                    self._zero_image(img_h, img_w, frames),
                    "\n".join(report))

        pos_cond = self._encode_text(clip, positive_prompt, peak_nits, target_gamut, hdr_eotf)
        neg_cond = self._encode_text(clip, negative_prompt)
        if character_conditioning is not None and pos_cond:
            pos_cond = self._merge_cond(pos_cond, character_conditioning, 0.75)

        denoise = 1.0
        if strategy in ("concat_channels", "clip_vision_inject"):
            # The image rides in the conditioning; sampling starts from pure
            # noise at full denoise, as with ComfyUI's WanImageToVideo.
            motion_noise = noise
            if strategy == "concat_channels" or concat_ch:
                concat_latent, concat_mask = self._wan_concat(
                    vae, reference_image, frames, noise_shape)
                extra = {"concat_latent_image": concat_latent, "concat_mask": concat_mask}
                pos_cond = self._set_cond(pos_cond, extra)
                neg_cond = self._set_cond(neg_cond, extra)
                report.append(f"Image concat: {list(concat_latent.shape)} (first frame kept)")
            if clip_vision_output is not None:
                cv = {"clip_vision_output": clip_vision_output}
                pos_cond = self._set_cond(pos_cond, cv)
                neg_cond = self._set_cond(neg_cond, cv)
                report.append("CLIP vision: injected")
            start_latent = _wrap(torch.zeros_like(noise))
            report.append("image_strength / motion_strength: not used by this strategy")
        else:
            img_latent = _to_tensor(_vae_encode(vae, reference_image)) if vae is not None else None
            if img_latent is None:
                raise ValueError(f"i2v_strategy {strategy} needs a VAE to encode the reference image.")
            if img_latent.dim() == 5:
                img_latent = img_latent[:, :, 0]
            report.append(f"Image latent: {list(img_latent.shape)}")
            # Scale noise by motion_strength (less noise = less motion)
            motion_noise = noise * (0.2 + motion_strength * 0.8)
            if strategy == "first_frame_lock":
                motion_noise = self._first_frame_lock(
                    motion_noise, img_latent, image_strength, noise_shape, ch)
            else:
                motion_noise = self._prepend_latent(motion_noise, img_latent, image_strength)
            start_latent = _wrap(motion_noise)
            denoise = 1.0 - (image_strength * 0.4)   # stronger image -> less denoising
            if clip_vision_output is not None:
                report.append("CLIP vision: connected but not used by " + strategy)

        report.append(f"Final noise shape: {list(motion_noise.shape)}")

        # --- Sample ---
        samples = _comfy_sample(
            model, motion_noise, eff_steps, cfg_eff, eff_sampler, eff_sched,
            pos_cond, neg_cond, start_latent,
            denoise=denoise, seed=seed,
        )
        video_latent = _wrap(samples)
        report.append(f"Sampled: {list(samples.shape)}")

        # --- Decode preview ---
        preview, preview_error = self._decode_preview(vae, video_latent, frames)
        if preview_error:
            report.append(
                f"Preview frames: DECODE FAILED ({preview_error}). "
                f"preview_frames is a black placeholder, not pixels. "
                f"video_latent is unaffected; decode it with "
                f"RadianceVideoBatchDecode (tile_decode=True for long clips)."
            )
        else:
            report.append(f"Preview frames: {preview.shape[0]}")

        return (video_latent, preview, "\n".join(report))

    # ------------------------------------------------------------------
    # Inference only. .eval() does not clear requires_grad on parameters, so an
# unguarded forward still builds and retains an autograd graph.
    @torch.no_grad()
    def _encode_text(self, clip, text, peak_nits=None, gamut=None, eotf=None):
        if clip is None or not HAS_TORCH:
            return []
        descriptors = _hdr_prompt_descriptors(peak_nits, gamut, eotf)
        hdr_sfx = f", {descriptors}" if descriptors else ""
        try:
            tokens = clip.tokenize(text + hdr_sfx)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            return [[cond, {"pooled_output": pooled}]]
        except Exception as exc:
            logger.warning("[nodes_t2v_pipeline] _encode_text: %s", exc)
            return []

    def _merge_cond(self, base, extra, weight):
        if not HAS_TORCH:
            return base
        merged = []
        for i, (bt, bd) in enumerate(base):
            new_d = dict(bd)
            if i < len(extra):
                et, _ = extra[i]
                if et.shape[-1] == bt.shape[-1]:
                    tok = torch.cat([bt, et * weight], dim=1)
                else:
                    tok = bt
            else:
                tok = bt
            merged.append([tok, new_d])
        return merged

    # --- Strategy implementations ---

    def _first_frame_lock(self, noise, img_latent, strength, shape, ch):
        """Lock the first frame of the video latent to the image encoding."""
        if img_latent is None or not HAS_TORCH:
            return noise
        try:
            if noise.dim() == 5:
                # Resize img_latent to match spatial dims
                il = img_latent.float()
                if il.shape[1] != ch:
                    rep = math.ceil(ch / il.shape[1])
                    il = il.repeat(1, rep, 1, 1)[:, :ch]
                lh, lw = noise.shape[-2], noise.shape[-1]
                if il.shape[-2] != lh or il.shape[-1] != lw:
                    il = F.interpolate(il, size=(lh, lw), mode="bilinear", align_corners=False)
                # Lock first temporal frame
                locked = noise.clone()
                locked[:, :, 0] = locked[:, :, 0] * (1 - strength) + il * strength
                return locked
            else:
                lh, lw = noise.shape[-2], noise.shape[-1]
                il = F.interpolate(img_latent.float(), size=(lh, lw),
                                    mode="bilinear", align_corners=False)
                if il.shape[1] != ch:
                    rep = math.ceil(ch / il.shape[1])
                    il = il.repeat(1, rep, 1, 1)[:, :ch]
                return noise * (1 - strength) + il * strength
        except Exception as exc:
            logger.warning(f"[I2V first_frame_lock] {exc}")
            return noise

    @staticmethod
    def _set_cond(cond, values):
        """Copy-on-write equivalent of node_helpers.conditioning_set_values."""
        out = []
        for t, d in cond or []:
            nd = dict(d)
            nd.update(values)
            out.append([t, nd])
        return out

    @torch.no_grad()
    def _wan_concat(self, vae, image, frames, noise_shape):
        """The image-concat conditioning ComfyUI's WanImageToVideo builds.

        The reference fills frame 0, the rest is 0.5 grey; the sequence is
        VAE-encoded, and concat_mask is 0 on the first latent frame (keep)
        and 1 elsewhere (generate).
        """
        if vae is None:
            raise ValueError("concat_channels needs a VAE to encode the reference image.")
        lh, lw = noise_shape[-2], noise_shape[-1]
        sc = _vae_spatial_compression(vae, 8)
        h, w = lh * sc, lw * sc
        ref = image[:1, :, :, :3].movedim(-1, 1).float()
        ref = F.interpolate(ref, size=(h, w), mode="bilinear", align_corners=False).movedim(1, -1)
        seq = torch.full((max(int(frames), 1), h, w, 3), 0.5, dtype=ref.dtype, device=ref.device)
        seq[:1] = ref
        latent = _to_tensor(_vae_encode(vae, seq))
        t_lat = latent.shape[2] if latent.dim() == 5 else noise_shape[2] if len(noise_shape) == 5 else 1
        mask = torch.ones((1, 1, t_lat, latent.shape[-2], latent.shape[-1]),
                          dtype=latent.dtype, device=latent.device)
        mask[:, :, :1] = 0.0
        return latent, mask

    def _prepend_latent(self, noise, img_latent, strength):
        """Prepend image latent as first frame(s) of video latent."""
        if img_latent is None or noise.dim() != 5 or not HAS_TORCH:
            return noise
        try:
            lh, lw = noise.shape[-2], noise.shape[-1]
            ch     = noise.shape[1]
            il = F.interpolate(img_latent.float(), size=(lh, lw),
                                mode="bilinear", align_corners=False)
            if il.shape[1] != ch:
                rep = math.ceil(ch / il.shape[1])
                il = il.repeat(1, rep, 1, 1)[:, :ch]
            il_t = il.unsqueeze(2) * strength   # [B,C,1,H,W]
            # Replace first temporal frame
            result = noise.clone()
            result[:, :, :1] = result[:, :, :1] * (1 - strength) + il_t
            return result
        except Exception as exc:
            logger.warning(f"[I2V prepend_latent] {exc}")
            return noise

    def _decode_preview(self, vae, latent, target_frames):
        """Decode the preview batch. Returns (frames, failure_text_or_None).

        Same contract as RadianceT2VPipeline._decode_preview: a failed decode
        keeps video_latent but must never be reported as decoded frames. The
        old bare `except Exception` returned a 64x64 black batch that the report
        counted as "Preview frames: N", so an OOM looked like a finished render.
        """
        if not HAS_TORCH:
            return None, "torch unavailable"
        try:
            decoded = _vae_decode(vae, latent)
            # comfy.sd.VAE.decode() returns (B, T, H, W, C) for a video VAE: it
            # movedim(1, -1)s the channel axis before returning. Reading it as
            # (B, C, T, H, W) and permuting scrambled the preview.
            decoded = _frames_to_batch(decoded)
            return decoded.clamp(0, 1), None
        except Exception as exc:
            logger.error("[RadianceI2VPipeline] preview decode failed: %s", exc)
            return (torch.zeros(target_frames, 64, 64, 3),
                    f"{type(exc).__name__}: {exc}")

    def _zero(self, shape):
        return torch.zeros(shape) if HAS_TORCH else shape

    def _zero_image(self, h, w, n):
        return torch.zeros(n, h, w, 3) if HAS_TORCH else None


# ===========================================================================
# Node: RadianceVideoBatchDecode
# ===========================================================================

class RadianceVideoBatchDecode:
    """
    Decode a DiT video LATENT tensor into an IMAGE frame batch.

    Handles both temporal (5-D) and spatial (4-D) latents.
    Applies optional per-model latent scale correction before decoding.
    Outputs IMAGE [N_frames, H, W, 3] ready for RadianceVideoHDRDecode,
    RadianceVideoFrameRouter, or any standard ComfyUI image node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE", {"tooltip": "VAE matching the model that produced the latent."}),
                "latent": ("LATENT", {
                    "tooltip": "Video latent from a sampler or pipeline. A 5-D latent is decoded through "
                               "the VAE's temporal path."}),
            },
            "optional": {
                "dit_config": ("STRING", {"default": "{}",
                    "tooltip": "JSON from Video Model Info. Its compression values are used only "
                               "if the VAE does not report its own. Its latent_scale is not applied: a "
                               "ComfyUI sampler already returns the latent in VAE space."}),
                "tile_decode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Route the decode through the VAE's own tiled entry point "
                               "(comfy.sd.VAE.decode_tiled) to cut peak VRAM on large "
                               "videos. Leave off: an untiled decode already falls back "
                               "to tiling by itself when it runs out of memory.",
                }),
                "tile_overlap": ("INT", {
                    "default": 64, "min": 0, "max": 256,
                    "tooltip": "Pixel overlap between spatial tiles (higher = smoother "
                               "seams). Only read when tile_decode is on.",
                }),
                "output_linear": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Convert the VAE's display-referred sRGB frames to scene-linear "
                               "(inverse sRGB transfer). Off: sRGB clamped to [0, 1], as VAE Decode.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("frames", "frame_count", "decode_report")
    FUNCTION = "decode"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Video"
    DESCRIPTION = "Batch-decode video latents to pixel frames with memory management."

    def decode(self, vae, latent, dit_config="{}", tile_decode=False,
               tile_overlap=64, output_linear=False):

        report = [f"=== RadianceVideoBatchDecode v{__version__} ==="]

        try:
            cfg_dict = json.loads(dit_config) if dit_config.strip() not in ("", "{}") else {}
        except Exception:
            cfg_dict = {}

        samples = _to_tensor(latent)
        report.append(f"Latent shape : {list(samples.shape)}")

        # latent_scale is not applied. A ComfyUI sampler returns its latent
        # already in VAE space (the model's process_latent_out undoes the
        # scaling before the LATENT leaves the sampler), and VAE.decode takes
        # it as is, as ComfyUI's own VAE Decode does. Dividing again scaled
        # HunyuanVideo, CogVideoX and SD latents twice.
        scale = cfg_dict.get("latent_scale", 1.0)
        if abs(float(scale) - 1.0) > 1e-5:
            report.append(f"latent_scale {scale} in dit_config not applied (sampler output is in VAE space)")

        is_5d = HAS_TORCH and isinstance(samples, torch.Tensor) and samples.dim() == 5
        latent_batch = int(samples.shape[0]) if is_5d else None
        latent_frames = int(samples.shape[2]) if is_5d else None

        # Compression factors come from the VAE itself when it will say, and
        # from the dit_config spec only as a fallback: the loaded VAE is the
        # authority on its own latent geometry, a preset JSON is a guess.
        spatial_comp = _vae_spatial_compression(vae, cfg_dict.get("spatial_compression", 8))
        temporal_comp = _vae_temporal_compression(vae)
        if temporal_comp is None and is_5d:
            temporal_comp = cfg_dict.get("temporal_compression") or None
            if temporal_comp:
                report.append(
                    f"Temporal compression: {temporal_comp} (from dit_config, this VAE "
                    f"has no temporal_compression_decode())"
                )
        elif temporal_comp:
            report.append(f"Temporal compression: {temporal_comp} (from VAE)")

        # The 5-D latent goes to the VAE as 5-D. comfy.sd.VAE.decode() computes
        # dims = samples_in.ndim - 2, so only a 5-D input reaches its video path
        # and its decode_tiled_3d() fallback, which shrinks the temporal tile
        # until one tile fits the memory budget. That fallback IS the
        # unlimited-duration decode. Flattening [B,C,T,H,W] to [B*T,C,H,W]
        # first forced the 2-D image path: temporal tiling never engaged, and a
        # video VAE with causal 3-D convs either raised or decoded each latent
        # frame in isolation, discarding the temporal upsample and returning
        # B*T frames instead of (T-1)*temporal_compression+1.
        _tiled_fn = getattr(vae, "decode_tiled", None) if vae is not None else None
        if tile_decode and callable(_tiled_fn):
            try:
                frames, tiled_args = _vae_decode_tiled(
                    vae, samples, tile_overlap, spatial_comp, temporal_comp)
                report.append(
                    "Tile decode: vae.decode_tiled() "
                    + "  ".join(f"{k}={v}" for k, v in tiled_args.items())
                    + " (latent units)"
                    + ("" if temporal_comp else "; spatial only, non-temporal VAE")
                )
            except Exception as exc:
                # Same rule as _vae_decode: a failed tiled decode must not turn
                # into a silently wrong picture.
                logger.error("[RadianceVideoBatchDecode] tiled decode failed: %s", exc)
                raise
        else:
            if tile_decode:
                # Requirement of this node's contract: never let a requested
                # mitigation quietly not happen. The predecessor probed for
                # enable_tiling/set_tiling, which exist on no ComfyUI VAE, and
                # printed "Tile decode: enabled" regardless.
                report.append(
                    "Tile decode: UNAVAILABLE, this VAE exposes no decode_tiled(); "
                    "decoding untiled. comfy.sd.VAE.decode() still falls back to "
                    "tiling on its own if it runs out of memory."
                )
            frames = _vae_decode(vae, _wrap(samples))

        if HAS_TORCH and isinstance(frames, torch.Tensor):
            # A video VAE returns (B, T, H, W, C): both decode() and
            # decode_tiled() movedim(1, -1) before returning. Collapse to the
            # IMAGE batch shape ComfyUI expects, exactly as the stock VAEDecode
            # node does.
            frames = _frames_to_batch(frames)
            if frames.dim() == 3:
                frames = frames.unsqueeze(0)
            frames = frames.clamp(0, 1)
            if output_linear:
                # It used to only skip the clamp; nothing ever linearised.
                frames = torch.where(frames <= 0.04045, frames / 12.92,
                                     ((frames + 0.055) / 1.055) ** 2.4)
                report.append("Output : scene-linear (inverse sRGB)")
            n = int(frames.shape[0])
            # Report what came back, not what was hoped for. The old line read
            # "Decoded {T} frames from 5D latent" while the function returned
            # B*T, so the number in the report never described the output.
            if is_5d:
                # The batch axis survives the decode, so the whole IMAGE batch is
                # B * ((T-1)*temporal_compression + 1), not one clip's worth.
                expected = latent_batch * _expected_pixel_frames(latent_frames, temporal_comp)
                note = "" if n == expected else (
                    f"  [CHECK: {latent_batch}x{latent_frames} latent frames at "
                    f"temporal compression {temporal_comp or 1} imply {expected}; "
                    f"frame_count is the {n} actually returned]"
                )
                report.append(
                    f"Decoded {n} pixel frames from {latent_frames} latent frames"
                    f" x {latent_batch} batch{note}"
                )
            report.append(f"Output : {n} frames  {list(frames.shape[1:])}")
        else:
            n = 0
            report.append(
                "Decode UNAVAILABLE: the VAE returned no tensor "
                f"(got {type(frames).__name__}). ComfyUI is required for a real "
                "decode; frames is the latent passed straight through, NOT pixels."
            )

        return (frames, n, "\n".join(report))


# ===========================================================================
# Node: RadianceVideoExport
# ===========================================================================

class RadianceVideoExport:
    """
    Route a decoded video IMAGE batch to the appropriate output format.

    Modes
    -----
    passthrough  — return frames unchanged (connect to video writer nodes)
    hdr_decode   — apply PQ/HLG decode via RadianceVideoHDRDecode inline
    exr_sequence — save each frame as a 32-bit float EXR (OpenEXR, else OpenCV)
    preview_gif  — save a small GIF preview and return frame batch

    An empty output_folder writes to ComfyUI's output directory. A write that
    fails raises; it used to fall back to a PNG writer that could not write
    RGB and still reported "Saved N EXR frames".

    The node is designed as the final stage in the pipeline:
      T2V / I2V → VideoBatchDecode → VideoExport → (VideoWriter / EXR IO)
    """

    MODES = ["passthrough", "hdr_decode", "exr_sequence", "preview_gif"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE", {
                    "tooltip": "Decoded frames. EXR writes the values unchanged as 32-bit float; the GIF "
                               "clamps to [0, 1] display-referred."}),
                "mode": (cls.MODES, {"default": "passthrough",
                    "tooltip": "passthrough: return frames. hdr_decode: run Video HDR Decode "
                               "(Reinhard, PQ out) and return the HDR signal. exr_sequence / preview_gif: "
                               "also write files. Only the last two write to disk."}),
            },
            "optional": {
                "hdr_metadata_json": ("STRING", {
                    "default": '{"peak_nits":1000,"eotf":"PQ (ST.2084)"}',
                    "tooltip": "hdr_decode only. Its peak_nits and gamut (BT.2020 if absent) are used; "
                               "the eotf key is ignored, output is always PQ.",
                }),
                "output_folder": ("STRING", {"default": "",
                    "tooltip": "Empty: ComfyUI's output folder."}),
                "filename_prefix": ("STRING", {"default": "radiance_video",
                    "tooltip": "File name start: <prefix>_<6-digit frame>.exr, or <prefix>_preview.gif "
                               "(overwritten on each run)."}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0,
                    "tooltip": "GIF playback rate (frame duration 1000 / fps ms). preview_gif only."}),
                "frame_offset": ("INT", {"default": 0, "min": 0,
                    "tooltip": "Added to the frame number in EXR file names. exr_sequence only."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("frames", "frame_count", "export_report")
    FUNCTION = "export"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Video"
    DESCRIPTION = ("Pass decoded video frames through, HDR-encode them, or write them as a 32-bit "
                   "float EXR sequence or a small GIF preview. No movie codecs: use a video writer node.")
    OUTPUT_NODE = True

    def export(self, frames, mode, hdr_metadata_json="{}",
               output_folder="", filename_prefix="radiance_video",
               fps=24.0, frame_offset=0):

        report = [f"=== RadianceVideoExport v{__version__} ===",
                  f"Mode   : {mode}",
                  f"Frames : {frames.shape[0] if HAS_TORCH else '?'}"]

        if mode == "passthrough":
            n = frames.shape[0] if HAS_TORCH else 0
            return (frames, n, "\n".join(report))

        if mode == "hdr_decode" and HAS_TORCH:
            try:
                from radiance.nodes.video.hdr import RadianceVideoHDRDecode
            except ImportError:
                RadianceVideoHDRDecode = None
            if RadianceVideoHDRDecode is not None:
                decoder = RadianceVideoHDRDecode()
                hdr_frames, sdr_frames, _ = decoder.decode(
                    frames, hdr_metadata_json, "Reinhard")
                report.append("HDR decode applied")
                return (hdr_frames, frames.shape[0], "\n".join(report))

        if mode in ("exr_sequence", "preview_gif") and not output_folder.strip():
            try:
                import folder_paths  # type: ignore
                output_folder = folder_paths.get_output_directory()
            except Exception as exc:
                raise ValueError(f"{mode}: output_folder is empty and ComfyUI's output "
                                 f"directory is unavailable ({exc})") from exc

        if mode == "exr_sequence":
            saved = self._save_exr_sequence(frames, output_folder,
                                              filename_prefix, frame_offset, report)
            report.append(f"Saved {saved} EXR frames (32-bit float) to {output_folder}")

        if mode == "preview_gif":
            if not HAS_PIL:
                raise RuntimeError("preview_gif needs Pillow, which is not installed")
            gif_path = self._save_gif(frames, output_folder, filename_prefix, fps, report)
            report.append(f"GIF saved: {gif_path}")

        n = frames.shape[0] if HAS_TORCH else 0
        return (frames, n, "\n".join(report))

    def _save_exr_sequence(self, frames, folder, prefix, offset, report):
        from radiance.hdr.io import write_exr_robust
        os.makedirs(folder, exist_ok=True)
        saved = 0
        for i in range(frames.shape[0]):
            arr = frames[i].detach().cpu().float().numpy()
            out_path = os.path.join(folder, f"{prefix}_{offset+i:06d}.exr")
            if not write_exr_robust(out_path, arr, "32-bit Float"):
                raise RuntimeError(
                    f"EXR write failed for {out_path} after {saved} frames: neither "
                    f"OpenEXR nor OpenCV's EXR codec could write it.")
            saved += 1
        return saved

    def _save_gif(self, frames, folder, prefix, fps, report):
        os.makedirs(folder, exist_ok=True)
        from PIL import Image as PILImage
        import numpy as np
        duration_ms = int(1000 / fps)
        pil_frames = []
        scale = min(1.0, 320 / frames.shape[2])  # max 320px wide
        for i in range(frames.shape[0]):
            arr = (frames[i].cpu().clamp(0,1).float().numpy() * 255).astype("uint8")
            img = PILImage.fromarray(arr[:,:,:3])
            if scale < 1.0:
                nw = int(arr.shape[1] * scale)
                nh = int(arr.shape[0] * scale)
                img = img.resize((nw, nh), PILImage.LANCZOS)
            pil_frames.append(img.convert("P", palette=PILImage.ADAPTIVE))
        gif_path = os.path.join(folder, f"{prefix}_preview.gif")
        pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:],
                           loop=0, duration=duration_ms)
        return gif_path


# ALBABIT-FIX: NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS removed.
# Live registry is nodes/video/__init__.py — this file's mappings were never
# read by ComfyUI and had drifted (stale display names, missing 4 HDR nodes).
