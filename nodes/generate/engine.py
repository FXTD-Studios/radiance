"""
nodes_engine.py — Radiance Engine Nodes

CHANGELOG v3.0.1 (audit + scored-review fixes):
  ROUND 1 — structural audit:
  - Issue 1: _SCENE_REFERRED guard now uses exact set membership (target_space in _SCENE_REFERRED)
             instead of fragile substring matching — prevents false positives on future space names.
  - Issue 2: Removed dead Radiancev3_MasterHub class (was defined but never registered in
             NODE_CLASS_MAPPINGS — invisible to ComfyUI).
  - Issue 3: Removed duplicate RadianceLUTApply class (canonical version is color/lut.py;
             local copy risked silent divergence). Import from .color if needed.
  - Issue 4: Fast decoder load wrapped in try/except — wrong model type or missing weights
             now logs a clear error and falls back to standard VAE instead of crashing.
  - Issue 5: Metadata timestamp now uses UTC (datetime.now(timezone.utc)) — unambiguous
             across timezones and daylight savings.
  ROUND 2 — scored review:
  - HIGH   : model_type NameError in except block — initialised to "unknown" sentinel
             before the try so the except handler can always format it safely.
             Raised a clear ValueError when samples["samples"] is None.
  - MEDIUM : SD 1.x misidentified as SDXL — 4-channel branch now inspects the VAE's
             first_stage_model class name for "xl"/"sdxl"; warns and falls back to
             "sdxl" weights for unrecognised 4-channel VAEs.
  - MEDIUM : _explicit set replaced with inspect.signature(self.apply).parameters —
             auto-syncs when new params are added; can never drift out of sync.
  - MEDIUM : metadata "version" field updated to "3.0.1" to match file version.
  - LOW    : @torch.no_grad() added to apply() — explicit intent, guards future paths.

CHANGELOG v3.0.2 (review fixes):
  - Fix 1: JS widget sync now matches "◎ Radiance HDR VAE Decode" display name
           (amber badge, display_tonemap graying, hdr_output warning)
  - Fix 2: force_hdr_decode respects the user's widget value (default True for this node)
           No longer hardcoded to True — passes user choice through to engine
  - Fix 3: Metadata JSON now includes alpha_restored, source_space, hdr_output fields
  - Fix 4: source_space, hdr_output, and all inherited params now explicitly declared
           in apply() method signature (no more ghost params through **kwargs)

CHANGELOG v3.0.0 vs v2.5:
  BUG FIXES:
    - BUG 1: RadianceHDRVAEDecode now exposes display_tonemap widget (v2.3.8 req).
             Without it, Compress(Log) mode silently used Reinhard regardless of intent.
    - BUG 2: hdr_scale_factor now guarded — warns and skips multiplication when
             target_space is display-referred ([0,1] sRGB/Raw), preventing blown output.
    - BUG 3: apply() target_space default aligned to vae.py v2.3.8 default ("sRGB").
             Was hardcoded "Linear" — silent mismatch with inherited INPUT_TYPES UI default.
    - BUG 4: NDI apply() guards image tensor shape before [0] indexing (IndexError).
    - BUG 5: NDI singleton tracks current stream_name and recreates sender on change.
    - BUG 7: RadianceHDRVAEDecode adds metadata STRING output (decode settings).
    - BUG 8: **kwargs now forwarded to engine.decode() so new vae.py params are passed.
    - BUG 9: numpy duplicate import removed from NDI apply().

  NEW NODES:
    - RadianceHDRColorPipeline: full Radiance color pipeline as standalone node.
      Any IMAGE + colorspace_in + colorspace_out → converted IMAGE.
    - RadianceHDRAnalysis: scene-linear image → peak nit, EV range, clip %, zone stats.
      Enables conditional workflow logic (e.g. auto-clamp if peak > 1000 nit).
    - RadianceLUTApply: apply a .cube LUT file to any IMAGE with strength control.
             (canonical implementation: color/lut.py — registered via color/__init__.py)

  NDI UPGRADES:
    - frame_rate exposed for NDI timing metadata.
    - connected BOOLEAN output — lets workflow branch when NDI is unavailable.
"""

import inspect
import logging
import math
import json
import datetime
from datetime import timezone as _tz
import torch
import numpy as np

from radiance.config.constants import VERSION
from radiance.hdr.vae import RadianceVAE4KEncode, RadianceVAE4KDecode
from radiance.hdr.decode_meta import (
    LOG_SPACE_GAMUT,
    TARGET_SPACE_GAMUT,
    _gamut_mat_t,
    strip_hdr_meta,
    verify_radiance_meta,
)
from radiance.color.transfer import (
    tensor_linear_to_logc4,
    tensor_linear_to_slog3,
)
from radiance.color.pipeline import apply_input_transform, INPUT_COLORSPACES

logger = logging.getLogger("radiance.engine")

# Scene-referred spaces — hdr_scale_factor is valid for these
_SCENE_REFERRED = {
    "Linear", "ACEScg", "ACES 2065-1", "Rec.2020 Linear",
    "ARRI LogC3", "ARRI LogC4", "Sony S-Log3", "Panasonic V-Log",
    "DaVinci Intermediate", "RED Log3G10",
}

_LOG_TARGETS = set(LOG_SPACE_GAMUT)


def _scale_rgb(image: torch.Tensor, gain: float) -> torch.Tensor:
    """Multiply RGB only; alpha and any extra channels pass through."""
    if image.shape[-1] <= 3:
        return image * gain
    return torch.cat([image[..., :3] * gain, image[..., 3:]], dim=-1)


def _linear709_to_target(image: torch.Tensor, target: str) -> torch.Tensor:
    """Linear Rec.709 -> a scene-referred target (gamut, then log curve)."""
    if target in ("Linear", "Raw"):
        return image
    rgb, extra = image[..., :3], image[..., 3:]
    mat = _gamut_mat_t("Rec.709", TARGET_SPACE_GAMUT.get(target, "Rec.709"))
    if mat is not None:
        rgb = (rgb.reshape(-1, 3) @ mat.to(rgb.device, rgb.dtype)).reshape(rgb.shape)
    if target in _LOG_TARGETS:
        from radiance.color import transfer as _tr
        curve = {
            "ARRI LogC3": _tr.tensor_linear_to_logc3,
            "ARRI LogC4": _tr.tensor_linear_to_logc4,
            "Sony S-Log3": _tr.tensor_linear_to_slog3,
            "Panasonic V-Log": _tr.tensor_linear_to_vlog,
            "DaVinci Intermediate": _tr.tensor_linear_to_davinci_intermediate,
            "RED Log3G10": _tr.tensor_linear_to_log3g10,
        }[target]
        rgb = curve(rgb)
    return torch.cat([rgb, extra], dim=-1) if extra.shape[-1] else rgb


# ═══════════════════════════════════════════════════════════════════════════════
#                    NODE 1: RADIANCE HDR VAE DECODE
# ═══════════════════════════════════════════════════════════════════════════════

class RadianceHDRVAEEncode(RadianceVAE4KEncode):
    """
    ◎ Radiance HDR VAE Encode -- the partner of HDR VAE Decode.

    Encodes a scene-linear (or log, or display) image into a VAE latent and
    stamps the latent with what it did: the HDR coding, the source space and
    a fingerprint of the latent itself. HDR VAE Decode in Auto reads that and
    inverts the coding exactly, so values above 1.0 survive the VAE; once a
    sampler has touched the latent the fingerprint no longer matches and
    Decode treats it as an ordinary diffusion latent.

    This was the only encoder whose latent HDR VAE Decode could invert, and
    it was never registered: the two HDR latent encoders on the menu (HDR
    Latent Encoder, HDR Turbo Encoder) fed decoders retired in 3.5.0, so
    their latents came back clipped. Measured through the SD VAE on an HDR
    plate peaking at 7.75: Compress (Log) returns highlights within 0.03
    stops (median) and keeps 97% of the values above 1.0.
    """

    DESCRIPTION = (
        "HDR-aware VAE encode, the partner of VAE Decode (HDR). Compress (Log) "
        "(default) keeps values above 1.0 through the VAE: VAE Decode (HDR) in "
        "Auto inverts it exactly. Tiled for 4K+ and video latents."
    )

    @classmethod
    def INPUT_TYPES(cls):
        types = super().INPUT_TYPES()
        req = types["required"]
        req["pixels"] = ("IMAGE", {"tooltip": (
            "Image to encode, in the encoding named by source_space (scene-linear by "
            "default, values above 1.0 allowed). It is linearised, exposed and HDR-coded "
            "per hdr_mode before the VAE sees it.")})
        req["vae"] = ("VAE", {"tooltip": (
            "VAE of the model the latent is meant for. Pair it with the same VAE in "
            "VAE Decode (HDR) so the HDR coding can be inverted.")})
        opt = types["optional"]
        choices, cfg = opt["hdr_mode"]
        opt["hdr_mode"] = (choices, {**cfg, "default": "Compress (Log)",
            "tooltip": ("How HDR is carried through the VAE. Compress (Log): values "
                        "above 1.0 survive and VAE Decode (HDR) inverts them exactly "
                        "(recommended). Soft Clip: gentler roll-off, less exact. "
                        "Clip (SDR): ordinary SDR encode. Passthrough: no coding.")})
        return types

    def encode(self, pixels, vae, source_space="Linear", hdr_mode="Compress (Log)", **kwargs):
        # The widget default only reaches graphs built in the UI; an API prompt
        # that omits an optional input gets the Python default, which on the
        # base class is Soft Clip. Keep the two in step.
        return super().encode(pixels, vae, source_space=source_space,
                              hdr_mode=hdr_mode, **kwargs)


class RadianceHDRVAEDecode:
    """
    ◎ Radiance HDR VAE Decode

    Thin wrapper around RadianceVAE4KDecode that adds:
      • hdr_scale_factor  — linear multiplier for scene-referred output
      • display_tonemap   — exposed from vae.py v2.3.8 (required for Compress(Log))
      • metadata output   — decode settings as a JSON string for downstream nodes
      • All vae.py params passed through correctly, no silent drops
      • explicit sampler-safe and direct-HDR decode contracts
      • Alpha restoration verified from the actual output tensor
      • source_space, hdr_output, and all vae.py params explicitly declared
    """

    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    DESCRIPTION = ("Decode a latent with an ordinary VAE into the chosen output space. Auto inverts the "
                   "HDR coding of latents straight from VAE Encode (HDR); Direct HDR returns scene-linear values above 1.0.")
    FUNCTION = "apply"
    RETURN_TYPES  = ("IMAGE", "STRING")
    RETURN_NAMES  = ("image", "metadata")
    OUTPUT_TOOLTIPS = (
        "Decoded image tensor.",
        "JSON string with decode settings and applied hdr_scale_factor.",
    )

    # Class-level engine cache — avoids re-instantiating RadianceVAE4KDecode
    # on every apply() call (which is a pure no-op object creation but adds
    # needless Python overhead on every ComfyUI execution).
    _engine: "RadianceVAE4KDecode | None" = None

    @classmethod
    def INPUT_TYPES(cls):
        # Inherit all inputs from the production decode node
        types = RadianceVAE4KDecode.INPUT_TYPES()
        types["required"]["samples"] = ("LATENT", {"tooltip": (
            "Latent to decode: from a sampler, or straight from VAE Encode (HDR), whose "
            "embedded metadata lets Auto invert the HDR coding exactly.")})
        types["required"]["vae"] = ("VAE", {"tooltip": (
            "VAE matching the model (and the VAE used by VAE Encode (HDR), if any).")})

        # BUG 3 FIX: explicitly set default to sRGB to match vae.py v2.3.8
        if "target_space" in types.get("required", {}):
            types["required"]["target_space"][1]["default"] = "sRGB"
            types["required"]["target_space"][1]["tooltip"] = (
                "Output colour space. Sampler-safe decodes deliver it as chosen: sRGB is "
                "display-ready for Preview/Save; Linear, ACEScg, ACES 2065-1 and Rec.2020 "
                "Linear are linear light in that gamut; a camera log target applies its "
                "gamut and curve (ARRI LogC4 = AWG4 + LogC4, Sony S-Log3 = S-Gamut3.Cine). "
                "Direct HDR output is scene-linear: sRGB and Raw become Linear there."
            )

        # Decode mode owns the force_hdr_decode safety decision.
        types["optional"].pop("force_hdr_decode", None)

        # Safe defaults for ordinary sampler latents. Direct HDR mode below
        # explicitly selects the log/linear contract at execution time.
        if "hdr_mode" in types.get("optional", {}):
            types["optional"]["hdr_mode"][1]["default"] = "Clip (SDR)"

        if "source_space" in types.get("optional", {}):
            types["optional"]["source_space"][1]["default"] = "sRGB"
            types["optional"]["source_space"][1]["tooltip"] = (
                "Kept for saved workflows. The log curve and camera gamut are read from "
                "the latent's HDR Encode metadata; this widget is not used."
            )
        if "display_tonemap" in types.get("optional", {}):
            types["optional"]["display_tonemap"][1]["default"] = "None"

        if "hdr_output" in types.get("optional", {}):
            types["optional"]["hdr_output"][1]["default"] = False
        if "export_rhdr" in types.get("optional", {}):
            types["optional"]["export_rhdr"][1]["default"] = False
        if "rhdr_precision" in types.get("optional", {}):
            types["optional"]["rhdr_precision"][1]["default"] = "f32"

        # Hide crop_padding and processing_mode from the UI — both have safe
        # defaults ("", "sequential") and only add noise for HDR decode users.
        types["optional"].pop("crop_padding", None)
        types["optional"].pop("processing_mode", None)

        # Add hdr_scale_factor to optional section
        types.setdefault("optional", {})
        types["optional"]["hdr_scale_factor"] = (
            "FLOAT",
            {
                "default": 1.0,
                "min": 0.1,
                "max": 10.0,
                "step": 0.05,
                "tooltip": (
                    "Linear gain on the Direct HDR output (RGB only, alpha untouched). "
                    "Not applied to sampler-safe SDR decodes or to log targets; the "
                    "metadata output says when it was skipped."
                ),
            },
        )
        # Append new widgets after the existing v3.0 controls so legacy
        # widgets_values arrays retain their original positional mapping.
        types["optional"]["decode_mode"] = (
            ["Auto (Recommended)", "Sampler (SDR-safe)", "Direct HDR"],
            {
                "default": "Auto (Recommended)",
                "tooltip": (
                    "Auto: Direct HDR when the latent comes straight from VAE Encode (HDR) "
                    "(its fingerprint still matches), sampler-safe for anything a sampler "
                    "touched. Sampler: standard SDR decode, never log inversion. Direct HDR: "
                    "scene-linear output above 1.0, no display tonemap; a VAE Encode (HDR) latent "
                    "is log-inverted exactly, any other latent is decoded and its clipped "
                    "highlights are reconstructed by the pixel SDR -> HDR model. The metadata "
                    "output names the path that ran."
                ),
            },
        )
        # 3.5: mastering peak for the RUDRA pixel reconstruction that Direct
        # HDR runs on ordinary (non log-encoded) latents. Appended after
        # decode_mode so saved widgets_values keep their positions.
        types["optional"]["hdr_peak_nits"] = (
            "FLOAT",
            {
                "default": 1000.0, "min": 200.0, "max": 10000.0, "step": 50.0,
                "tooltip": (
                    "Direct HDR on an ordinary sampler latent: the mastering peak the "
                    "reconstructed highlights are limited to. Output is linear with SDR "
                    "white = 1.0 = 203 nits (BT.2408), so the peak sits at peak/203 "
                    "(4.93 at 1000). Not used when the latent is log-encoded by HDR Encode."
                ),
            },
        )
        # ALBABIT-FIX: Restored from previous radiance version. Accepts
        # RadianceResolution's crop_bbox output to crop off model-alignment
        # padding (e.g. LTX's 32px turning a 1920x1080 request into 1920x1088)
        # right here, instead of needing a separate crop node after this one.
        types["optional"]["crop_bbox"] = (
            "BOUNDING_BOX",
            {
                "forceInput": True,
                "tooltip": (
                    "Optional: connect RadianceResolution's crop_bbox output to "
                    "crop off model-alignment padding (e.g. 1920x1088 -> "
                    "1920x1080) after decode."
                ),
            },
        )
        return types

    def apply(
        self,
        samples: dict,
        vae,
        target_space: str = "sRGB",     # BUG 3 FIX: aligned to vae.py v2.3.8 default
        tile_size: str = "Auto",
        overlap: int = 128,
        exposure_adjust: float = 0.0,
        alpha=None,
        hdr_mode: str = "Clip (SDR)",
        display_tonemap: str = "None",
        source_space: str = "sRGB",
        hdr_output: bool = False,
        inverse_tonemap: bool = False,
        target_stops: float = 12.0,
        crop_padding: str = "",
        export_rhdr: bool = False,
        rhdr_precision: str = "f32",
        processing_mode: str = "sequential",
        decode_noise_scale: float = 0.0,
        decode_mode: str = "Auto (Recommended)",
        hdr_scale_factor: float = 1.0,
        crop_bbox: dict = None,
        hdr_peak_nits: float = 1000.0,             # ALBABIT-FIX: broadcast-resolution crop from RadianceResolution
        **kwargs,                           # BUG 8 FIX: forward remaining params
    ):
        # Lazily instantiate once; RadianceVAE4KDecode is stateless so one
        # shared instance is safe across all ComfyUI graph executions.
        if RadianceHDRVAEDecode._engine is None:
            RadianceHDRVAEDecode._engine = RadianceVAE4KDecode()
        engine = RadianceHDRVAEDecode._engine

        # Guard: if the user connected an IMAGE (LoadImage etc.) to the 'samples'
        # input instead of a LATENT, give a clear message instead of a cryptic
        # VAE crash.
        if not isinstance(samples, dict):
            raise RuntimeError(
                "This node needs a LATENT, but the input is an IMAGE.\n\n"
                "Make sure you connect a VAE Encode or a Sampler node before this one."
            )
        if "samples" not in samples:
            raise RuntimeError(
                "The LATENT connected to this node is missing its data.\n\n"
                f"Keys found: {list(samples.keys())}\n"
                "Make sure you connect a valid LATENT output."
            )
        latent_tensor = samples["samples"]
        if getattr(latent_tensor, "is_nested", False):
            # ALBABIT-FIX: AV latent (e.g. MiniMax H3). This node only does
            # video (HDR/tonemap); peels the video stream the same way
            # ComfyUI's own LTXVSeparateAVLatent does (unbind()[0] = video).
            # Decode audio separately via a native VAEDecodeAudio fed from
            # the same latent and vae.
            latent_tensor = latent_tensor.unbind()[0]
            samples = {**samples, "samples": latent_tensor}
            logger.info(
                "[RadianceHDRVAEDecode] AV latent detected, decoding the video "
                "stream only. Use a native VAEDecodeAudio (same latent/vae "
                "inputs) for the audio track."
            )
        if not isinstance(latent_tensor, torch.Tensor) or latent_tensor.ndim not in (4, 5):
            raise RuntimeError(
                "The LATENT 'samples' value must be a 4D image latent or 5D video latent tensor."
            )

        # Track whether alpha was provided (for metadata / downstream use)
        alpha_provided = alpha is not None
        is_video = latent_tensor.ndim == 5

        # 3.5.0: samplers copy the latent dict and keep radiance_meta, so its
        # hdr_mode used to survive KSampler and send Auto down the log path on
        # a diffused latent (blown-out frames). Only a latent whose fingerprint
        # still matches what HDR Encode produced keeps its HDR coding keys.
        samples, meta_live = verify_radiance_meta(samples)
        radiance_meta = samples.get("radiance_meta") or {}
        encoded_hdr_mode = radiance_meta.get("hdr_mode") if isinstance(radiance_meta, dict) else None
        log_encoded = encoded_hdr_mode in {"Compress (Log)", "Soft Clip"}

        # "Direct HDR / RUDRA" is the pre-3.5 name of this mode; saved
        # workflows still carry it.
        if decode_mode in {"Direct HDR", "Direct HDR / RUDRA"}:
            effective_decode_mode = "Direct HDR"
        elif decode_mode == "Sampler (SDR-safe)":
            effective_decode_mode = "Sampler (SDR-safe)"
        else:  # Auto: direct only when live HDR Encode metadata says so
            effective_decode_mode = "Direct HDR" if log_encoded else "Sampler (SDR-safe)"

        # ALBABIT-FIX: presence only known at execution time. Used below to
        # compute log_overexposure_risk for the frontend's post-execution marker.
        radiance_meta_present = log_encoded

        # Forward extra vae.py params via **kwargs, stripping any key already
        # passed explicitly to avoid "multiple values for keyword argument".
        _sig_params = set(inspect.signature(self.apply).parameters)
        _sig_params.discard("kwargs")       # the **kwargs catch-all itself
        safe_kwargs = {k: v for k, v in kwargs.items() if k not in _sig_params}

        # Which HDR path Direct HDR takes. Log inversion is only correct on a
        # latent HDR Encode log-coded (live radiance_meta). Anything else is
        # decoded to SDR and its clipped highlights are reconstructed to
        # scene-linear by the pixel SDR -> HDR model.
        direct = effective_decode_mode == "Direct HDR"
        pixel_hdr = direct and not log_encoded
        want_rhdr = bool(export_rhdr)
        hdr_target = target_space if target_space in _SCENE_REFERRED else "Linear"
        post_exposure = 0.0
        notes = []
        if pixel_hdr:
            # Decode plain SDR; exposure is applied after reconstruction, in
            # scene-linear, so a positive trim no longer clips highlights
            # before the model sees them.
            post_exposure = float(exposure_adjust)
            eng = dict(hdr_mode="Clip (SDR)", source_space="sRGB", target_space="sRGB",
                       display_tonemap="None", hdr_output=False, force_hdr_decode=False,
                       exposure_adjust=0.0, inverse_tonemap=False,
                       decode_noise_scale=0.0)
            hdr_path = "pixel SDR->HDR"
        elif direct:
            log_source = radiance_meta.get("source_space")
            if encoded_hdr_mode == "Compress (Log)" and log_source not in LOG_SPACE_GAMUT:
                log_source = "ARRI LogC4"   # HDR Encode's curve for non-log sources
            eng = dict(hdr_mode=encoded_hdr_mode, source_space=log_source or "sRGB",
                       target_space=hdr_target, display_tonemap="None", hdr_output=True,
                       force_hdr_decode=True, exposure_adjust=exposure_adjust,
                       inverse_tonemap=False, decode_noise_scale=decode_noise_scale)
            hdr_path = "log inversion (radiance_meta)"
        else:
            # Sampler-safe: a diffused latent is display-referred SDR. The
            # visible target_space is honoured (hdr_output follows it, so
            # "Linear" really is linear), and inverse_tonemap keeps the range
            # it creates instead of being clamped straight back to 1.0.
            samples = strip_hdr_meta(samples)
            eng = dict(hdr_mode="Clip (SDR)", source_space="sRGB", target_space=target_space,
                       display_tonemap="None",
                       hdr_output=(target_space != "sRGB") or bool(inverse_tonemap),
                       force_hdr_decode=False, exposure_adjust=exposure_adjust,
                       inverse_tonemap=inverse_tonemap, decode_noise_scale=0.0)
            hdr_path = "sampler SDR"
            if want_rhdr:
                notes.append("RHDR export skipped: the sampler-safe decode is display-referred")
                want_rhdr = False
            if hdr_scale_factor != 1.0:
                notes.append("hdr_scale_factor applies to Direct HDR output only")

        result = engine.decode(
            samples=samples,
            vae=vae,
            tile_size=tile_size,
            overlap=overlap,
            alpha=alpha,
            target_stops=target_stops,
            crop_padding=crop_padding,
            export_rhdr=False,          # written below, after scale and crop
            rhdr_precision=rhdr_precision,
            processing_mode=processing_mode,
            **eng,
            **safe_kwargs,
        )

        image = result[0] if isinstance(result, (tuple, list)) else result
        engine_meta = {}
        if isinstance(result, (tuple, list)) and len(result) > 1:
            try:
                engine_meta = json.loads(result[1]) if isinstance(result[1], str) else dict(result[1])
            except (json.JSONDecodeError, TypeError, ValueError):
                engine_meta = {}
        latent_format = result[2] if isinstance(result, (tuple, list)) and len(result) > 2 else None

        out_space = eng["target_space"]
        pixel_report = None
        if pixel_hdr:
            image, pixel_report, hdr_path = self._pixel_hdr(
                image, float(hdr_peak_nits), is_video=is_video)
            if post_exposure != 0.0:
                image = _scale_rgb(image, 2.0 ** post_exposure)
            image = _linear709_to_target(image, hdr_target)
            out_space = hdr_target

        # hdr_scale_factor: Direct HDR only, RGB only (it used to scale alpha).
        scale_applied = False
        if direct and hdr_scale_factor != 1.0:
            if out_space in _LOG_TARGETS:
                notes.append("hdr_scale_factor not applied to a log-encoded target")
            else:
                image = _scale_rgb(image, float(hdr_scale_factor))
                scale_applied = True

        # ALBABIT-FIX: crop off model-alignment padding using
        # RadianceResolution's crop_bbox. 3.5.0: clamped to the image.
        if crop_bbox:
            H, W = int(image.shape[1]), int(image.shape[2])
            bx = min(max(int(crop_bbox.get("x", 0)), 0), W)
            by = min(max(int(crop_bbox.get("y", 0)), 0), H)
            bw = min(max(int(crop_bbox.get("width", W)), 0), W - bx)
            bh = min(max(int(crop_bbox.get("height", H)), 0), H - by)
            if bw > 0 and bh > 0:
                image = image[:, by:by + bh, bx:bx + bw, :]
            else:
                notes.append(f"crop_bbox {crop_bbox} lies outside the {W}x{H} image; not applied")

        # RHDR: Direct HDR only, from the final scene-linear image, so it
        # carries the same scale, crop and target as the IMAGE output.
        engine_meta.pop("rhdr_export", None)
        if want_rhdr and direct:
            if out_space in _LOG_TARGETS:
                notes.append("RHDR export needs a linear target; skipped for a log target")
            else:
                names = self._write_rhdr(image, rhdr_precision)
                if names:
                    engine_meta["rhdr_export"] = names[0] if len(names) == 1 else names
                    engine_meta["rhdr_precision"] = rhdr_precision
        elif want_rhdr:
            notes.append("RHDR export skipped: this decode ran sampler-safe SDR")

        for n in notes:
            logger.info("[RadianceHDRVAEDecode] %s", n)

        engine_meta.update({
            "node": "RadianceHDRVAEDecode",
            "version": VERSION,
            "decode_mode": effective_decode_mode,
            "decode_mode_requested": decode_mode,
            "hdr_path": hdr_path,
            "radiance_meta_live": bool(meta_live),
            "target_space": out_space,
            "hdr_mode": eng["hdr_mode"],
            "source_space": eng["source_space"],
            "exposure_adjust": exposure_adjust,
            "hdr_scale_factor": hdr_scale_factor if scale_applied else "N/A",
            "alpha_restored": bool(alpha_provided and image.shape[-1] == 4),
            "hdr_output": bool(direct or eng["hdr_output"]),
            "latent_format": latent_format or engine_meta.get("latent_format", "unknown"),
            "timestamp": datetime.datetime.now(_tz.utc).isoformat(timespec="seconds"),
        })
        if direct:
            engine_meta["linear_convention"] = (
                "1.0 = SDR diffuse white = 203 nits (BT.2408)" if pixel_hdr
                else "1.0 = the encoded source's white")
        if notes:
            engine_meta["notes"] = notes
        if pixel_report:
            engine_meta["hdr_peak_nits"] = float(hdr_peak_nits)
            engine_meta["sdr_to_hdr_report"] = pixel_report
        meta = json.dumps(engine_meta, indent=2)

        log_overexposure_risk = (eng["hdr_mode"] == "Compress (Log)" and not radiance_meta_present
                                 and not pixel_hdr)
        return {
            "ui": {
                "log_overexposure_risk": [log_overexposure_risk],
            },
            "result": (image, meta),
        }

    @staticmethod
    def _write_rhdr(image: torch.Tensor, precision: str):
        try:
            import folder_paths
            out_dir = folder_paths.get_temp_directory()
        except Exception as exc:  # noqa: BLE001 - export is best effort
            logger.warning("[RadianceHDRVAEDecode] RHDR export failed: %s", exc)
            return []
        names = []
        n = int(image.shape[0])
        for i in range(n):
            prefix = f"radiance_4k_f{i:04d}" if n > 1 else "radiance_4k"
            fn = RadianceVAE4KDecode._save_rhdr(
                image[i, ..., :3].float().cpu().numpy(), out_dir,
                prefix=prefix, precision=precision)
            if fn:
                names.append(fn)
        return names

    @staticmethod
    def _pixel_hdr(sdr: torch.Tensor, peak_nits: float, is_video: bool = False):
        """Decoded SDR -> scene-linear HDR through SDR -> HDR Universal.

        Hybrid mode with the Auto backend: deterministic expansion to BT.2408
        reference white, plus RUDRA pixel reconstruction inside the clipped
        highlights when the checkpoint is installed. Without the checkpoint
        the result is the deterministic expansion, and the returned path
        says so rather than claiming learned recovery.
        """
        from radiance.nodes.hdr.uplift_universal import RadianceSDRToHDRUniversal
        out, _m, _s, _h, _sc, report = RadianceSDRToHDRUniversal().convert(
            sdr.float().clamp(0.0, 1.0) if sdr.shape[-1] <= 3
            else torch.cat([sdr[..., :3].float().clamp(0.0, 1.0), sdr[..., 3:].float()], dim=-1),
            "sRGB", float(peak_nits), "adaptive", 0.75, 2.0, 0.85, "Linear",
            reference_white_nits=203.0,
            # 3.5.0: from the latent, not the batch size. A batch of four
            # independent images used to be treated as a clip, which smoothed
            # the adaptive knee across unrelated pictures.
            batch_mode="Video Frames" if is_video and sdr.shape[0] > 1 else "Independent Images",
            processing_mode="Hybrid", learned_backend="Auto",
            pixel_recovery_mode="highlights",
        )
        learned = "learned recovery: applied" in report
        path_line = next((l[6:] for l in report.splitlines() if l.startswith("path: ")), "")
        hdr_path = (f"pixel SDR->HDR model ({path_line})" if learned
                    else "SDR->HDR expansion only (no pixel checkpoint installed)")
        return out, report, hdr_path


# ═══════════════════════════════════════════════════════════════════════════════
#                    NODE 2: RADIANCE HDR ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

class RadianceHDRAnalysis:
    """
    ◎ Radiance HDR Analysis

    Analyses a scene-linear IMAGE tensor and outputs HDR metrics used by the
    Radiance Viewer exposure strip — now available as ComfyUI node outputs for
    conditional workflow logic.

    Output values:
      peak_nit    — estimated peak luminance in cd/m² (BT.2408: 203 nit = linear 1.0)
      ev_range    — dynamic range in stops between p01 and p99
      clipped_pct — percentage of pixels above scene-linear 1.0
      is_hdr      — True if peak_nit > 203 (above SDR white)
      stats_json  — full statistics as JSON string

    Typical use:
      Feed scene-linear output of Radiance Decode → RadianceHDRAnalysis →
      route to RadianceHDRColorPipeline or conditional tonemapper based on is_hdr.
    """

    CATEGORY = "FXTD STUDIOS/Radiance/◎ Generate"
    DESCRIPTION = "Analyse HDR image statistics: peak luminance, clipping, histogram."
    FUNCTION = "analyse"
    RETURN_TYPES  = ("FLOAT", "FLOAT", "FLOAT", "BOOLEAN", "STRING")
    RETURN_NAMES  = ("peak_nit", "ev_range", "clipped_pct", "is_hdr", "stats_json")
    OUTPUT_TOOLTIPS = (
        "Estimated peak luminance in cd/m² (BT.2408 anchor: 203 nit = linear 1.0).",
        "Dynamic range in stops (EV) between p01 and p99 luma.",
        "Percentage of pixels above scene-linear 1.0 (clipped for SDR display).",
        "True when peak_nit > 203 — image contains HDR content above SDR white.",
        "Full zone statistics as JSON.",
    )

    # BT.2408 SDR anchor: 203 cd/m² = scene-linear 1.0
    _NIT_ANCHOR = 203.0
    _MAX_SAMPLES = 500_000   # performance cap — subsample large tensors

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": (
                    "Image to measure; set colorspace to match its encoding. Luma above "
                    "1.0 counts as above SDR white (203 nits). Images over 500k pixels "
                    "are measured on a strided subsample.")}),
            },
            "optional": {
                "colorspace": (
                    INPUT_COLORSPACES,
                    {
                        "default": "Linear (sRGB)",
                        "tooltip": (
                            "Colorspace of the incoming image. "
                            "'Linear (sRGB)' is the correct setting for Radiance Decode output. "
                            "Log spaces are linearised first for accurate nit estimation."
                        ),
                    },
                ),
            },
        }

    @torch.no_grad()
    def analyse(self, image: torch.Tensor, colorspace: str = "Linear (sRGB)"):
        # Linearise if needed (log/sRGB input)
        if colorspace != "Linear (sRGB)" and colorspace != "ACEScg":
            linear = apply_input_transform(image, colorspace)
        else:
            linear = image

        # Flatten to (N, 3) for fast percentile computation
        flat = linear[..., :3].reshape(-1, 3).float()
        N = flat.shape[0]

        # Subsample if very large
        if N > self._MAX_SAMPLES:
            step = max(1, N // self._MAX_SAMPLES)
            flat = flat[::step]

        if colorspace == "ACEScg":
            luma = 0.272229 * flat[:, 0].clamp(min=0.0) \
                 + 0.674082 * flat[:, 1].clamp(min=0.0) \
                 + 0.053689 * flat[:, 2].clamp(min=0.0)
        else:
            luma = 0.2126 * flat[:, 0].clamp(min=0.0) \
                 + 0.7152 * flat[:, 1].clamp(min=0.0) \
                 + 0.0722 * flat[:, 2].clamp(min=0.0)

        p01, p50, p99, p999 = torch.quantile(
            luma, torch.tensor([0.01, 0.50, 0.99, 0.999], device=luma.device)
        ).tolist()
        peak = float(luma.max())

        # EV range (p01 → p99) — avoid log(0)
        ev_range = math.log2(max(p99, 1e-6) / max(p01, 1e-6)) if p01 > 1e-6 else 0.0

        # Clipped pixels (above scene-linear 1.0)
        clipped_pct = float((luma > 1.0).float().mean()) * 100.0

        # Peak nit estimate
        peak_nit = peak * self._NIT_ANCHOR

        is_hdr = bool(peak_nit > self._NIT_ANCHOR)

        stats = {
            "node": "RadianceHDRAnalysis",
            "colorspace": colorspace,
            "pixels_sampled": len(luma),
            "p01": round(p01, 6),
            "p50": round(p50, 6),
            "p99": round(p99, 6),
            "p99.9": round(p999, 6),
            "peak_linear": round(peak, 6),
            "peak_nit": round(peak_nit, 2),
            "ev_range": round(ev_range, 3),
            "clipped_pct": round(clipped_pct, 4),
            "is_hdr": is_hdr,
            "nit_anchor": self._NIT_ANCHOR,
        }
        stats_json = json.dumps(stats, indent=2)

        logger.info(
            f"[HDRAnalysis] peak={peak_nit:.0f} nit, "
            f"EV={ev_range:.1f}, clip={clipped_pct:.2f}%, hdr={is_hdr}"
        )

        return (float(peak_nit), float(ev_range), float(clipped_pct), is_hdr, stats_json)


# ═══════════════════════════════════════════════════════════════════════════════
# NOTE (Issue 3 fix): RadianceLUTApply was previously defined here but is NOT
# registered in this file's NODE_CLASS_MAPPINGS. The canonical, full-featured
# implementation lives in color/lut.py and is registered via color/__init__.py.
# The local copy has been removed to eliminate the risk of the two diverging
# silently. Import from color/ if you need it:
#   from .color import RadianceLUTApply
# ═══════════════════════════════════════════════════════════════════════════════
#                    NODE 5: RADIANCE NDI SENDER
# ═══════════════════════════════════════════════════════════════════════════════

class RadianceNDISender:
    """
    ◎ Radiance NDI Sender

    Real-time streaming node to push frames to OBS, Resolume, Nuke,
    or any NDI receiver via NewTek NDI SDK.

    - Singleton tracks stream_name and recreates the sender on name change.
    - Guards against an empty image batch.
    - frame_rate exposed for NDI timing metadata.
    - connected BOOLEAN output for workflow branching.

    3.5.0: the legacy latent "turbo" decode path was retired with the rest of
    the latent RUDRA decoders. The node streams the IMAGE it is given; decode
    upstream with HDR VAE Decode (Direct HDR) or SDR → HDR Universal.

    Note: NDIlib must be installed separately.
      pip install ndi-python
    """

    CATEGORY = "FXTD STUDIOS/Radiance/◎ Generate"
    DESCRIPTION = "Stream frames in real time over the NDI network video protocol."
    FUNCTION = "apply"
    OUTPUT_NODE = True
    RETURN_TYPES  = ("IMAGE", "BOOLEAN")
    RETURN_NAMES  = ("image", "connected")
    OUTPUT_TOOLTIPS = (
        "Pass-through image (unchanged).",
        "True if NDI frame was sent successfully this call.",
    )

    # Singleton NDI state
    _ndi_send_instance = None
    _ndi_video_frame   = None
    _ndi_stream_name   = None     # BUG 5 FIX: track active stream name
    _ndi_lock = __import__("threading").RLock()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": (
                    "Frames to stream, every frame of the batch in order. Sent as 8-bit "
                    "BGRA, so with encoding None the values must already be display-ready 0-1.")}),
                "stream_name": (
                    "STRING",
                    {"default": "Radiance ComfyUI",
                     "tooltip": "NDI source name shown to receivers. Changing it recreates the sender."},
                ),
                "encoding": (
                    ["None (SDR)", "S-Log3 (HDR)", "LogC4 (HDR)"],
                    {"default": "None (SDR)",
                     "tooltip": (
                         "Curve applied before 8-bit quantisation. None sends values as-is (clipped "
                         "to 0-1). S-Log3 and LogC4 expect scene-linear input and apply only the log "
                         "curve (no gamut change), so the receiver must decode with the same curve.")},
                ),
                "enable_streaming": ("BOOLEAN", {"default": True,
                    "tooltip": "Enable real-time NDI streaming during generation. Requires NDI SDK installed."
                }),
                "frame_rate": (
                    "FLOAT",
                    {
                        "default": 24.0,
                        "min": 1.0,
                        "max": 120.0,
                        "step": 1.0,
                        "tooltip": "Frame rate written into NDI video frame metadata.",
                    },
                ),
            },
        }

    def apply(
        self,
        image: torch.Tensor,
        stream_name: str,
        encoding: str,
        enable_streaming: bool,
        frame_rate: float = 24.0,
        **_legacy,   # latent_in / vae / turbo_mode / model_meta from pre-3.5 workflows
    ):
        if not enable_streaming:
            return (image, False)

        # BUG 4 FIX: guard empty batch
        if image is None or image.shape[0] == 0:
            logger.warning("[Radiance NDI] Received empty image batch — skipping.")
            return (image, False)

        if _legacy.get("turbo_mode"):
            logger.warning(
                "[Radiance NDI] turbo_mode is no longer available (the latent RUDRA "
                "decoders were retired in 3.5.0); streaming the connected image."
            )

        def _encode(frame):
            if encoding == "S-Log3 (HDR)":
                return tensor_linear_to_slog3(frame)
            if encoding == "LogC4 (HDR)":
                return tensor_linear_to_logc4(frame)
            return frame

        # ── NDI dispatch ───────────────────────────────────────────────────────
        connected = False
        try:
            import NDIlib as ndi
        except ImportError:
            logger.warning(
                "[Radiance NDI] NDIlib not installed. "
                "Install with: pip install ndi-python"
            )
            return (image, False)

        # BUG 5 FIX: recreate sender if stream_name changed
        with RadianceNDISender._ndi_lock:
            if (RadianceNDISender._ndi_send_instance is None
                    or RadianceNDISender._ndi_stream_name != stream_name):
                if RadianceNDISender._ndi_send_instance is not None:
                    ndi.send_destroy(RadianceNDISender._ndi_send_instance)
                    RadianceNDISender._ndi_send_instance = None
                if not ndi.initialize():
                    logger.error("[Radiance NDI] ndi.initialize() failed.")
                    return (image, False)
                desc = ndi.SendCreate()
                desc.p_ndi_name = stream_name
                # Paced by the SDK at frame_rate, so a batch plays as video
                # instead of arriving as a burst.
                desc.clock_video = True
                RadianceNDISender._ndi_send_instance = ndi.send_create(desc)
                RadianceNDISender._ndi_video_frame   = ndi.VideoFrameV2()
                RadianceNDISender._ndi_stream_name   = stream_name
                logger.info(f"[Radiance NDI] Sender created: '{stream_name}'")

        # Every frame of the batch is sent. Only image[0] used to go out, so a
        # decoded clip streamed as one still. The wire format is 8-bit BGRA;
        # the log encodings exist to fit HDR into those 8 bits.
        vf = RadianceNDISender._ndi_video_frame
        for idx in range(image.shape[0]):
            img_np = _encode(image[idx].clone()).cpu().float().numpy()
            H, W, C = img_np.shape
            img_8bit = np.clip(img_np * 255.0 + 0.5, 0, 255).astype(np.uint8)
            alpha_ch = (img_8bit[..., 3:4] if C >= 4
                        else np.full((H, W, 1), 255, dtype=np.uint8))
            img_bgra = np.ascontiguousarray(np.concatenate(
                [img_8bit[..., 2:3], img_8bit[..., 1:2], img_8bit[..., 0:1], alpha_ch], axis=2,
            ))
            vf.xres = W
            vf.yres = H
            vf.FourCC = ndi.FOURCC_VIDEO_TYPE_BGRA
            vf.p_data = img_bgra
            vf.line_stride_in_bytes = W * 4
            vf.frame_rate_N = int(frame_rate * 1000)
            vf.frame_rate_D = 1000
            ndi.send_send_video_v2(RadianceNDISender._ndi_send_instance, vf)
        connected = True
        logger.info("[Radiance NDI] sent %d frame(s) to '%s' (8-bit BGRA, %s)",
                    image.shape[0], stream_name, encoding)

        return (image, connected)


# NOTE (Issue 2 fix): Radiancev3_MasterHub was previously defined here but was
# never added to NODE_CLASS_MAPPINGS — it was invisible to ComfyUI and constituted
# dead code. It has been removed. If you want to ship this node, implement it
# fully and register it in NODE_CLASS_MAPPINGS before release.

# ═══════════════════════════════════════════════════════════════════════════════
#                          NODE REGISTRATION
# ═══════════════════════════════════════════════════════════════════════════════

NODE_CLASS_MAPPINGS = {
    "RadianceHDRVAEDecode":        RadianceHDRVAEDecode,
    "RadianceHDRAnalysis":         RadianceHDRAnalysis,
    # RadianceLUTApply is the canonical key — registered in color/__init__.py
    "RadianceNDISender":           RadianceNDISender,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceHDRVAEDecode":        "◎ Radiance HDR VAE Decode",
    "RadianceHDRAnalysis":         "◎ Radiance HDR Analysis",
    "RadianceNDISender":           "◎ Radiance NDI Sender",
}

import atexit
def _ndi_cleanup():
    if RadianceNDISender._ndi_send_instance is not None:
        try:
            import NDIlib as ndi
            ndi.send_destroy(RadianceNDISender._ndi_send_instance)
        except Exception as _exc:
            logger.debug(
                "[Radiance] _ndi_cleanup(): ignoring %s from `import NDIlib as ndi`: %s",
                type(_exc).__name__, _exc,
            )
atexit.register(_ndi_cleanup)
