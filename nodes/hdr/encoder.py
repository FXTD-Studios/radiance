"""
nodes_hdr_encoder.py — Radiance HDR encoding nodes.

Architecture note (LTX-Video alignment):
    LTX-Video's VAE was trained on display-referred images in [0, 1].  Feeding
    it log-encoded (LogC4 / S-Log3) data was a conceptual mismatch: the model
    never saw log-space inputs during training, so highlight tones were decoded
    incorrectly.

    The corrected approach mirrors LTX-Video's `tone_map_compression_ratio`:
      1. Apply a soft-knee Reinhard compression IN LINEAR LIGHT that smoothly
         maps HDR highlights into [0, 1] without posterising them.
      2. `compression_ratio` blends between hard-clamp (0.0) and full Reinhard
         (1.0) — identical semantics to LTX-Video's parameter.
      3. The image fed to vae.encode() is always display-referred [0, 1].

    Per-channel VAE normalisation (RadianceHDRPerChannelNorm) is a separate
    node that mirrors `vae_per_channel_normalize=True` from LTX-Video:
    it computes per-frame per-channel mean/std, normalises the image, and
    stores the stats so the decoder node can invert the normalisation.
"""

import torch
import logging

logger      = logging.getLogger("radiance.hdr_encoder")
diag_logger = logging.getLogger("radiance.diagnostics")


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _hdr_soft_compress(img: torch.Tensor, compression_ratio: float) -> torch.Tensor:
    """
    Soft-knee HDR compression tuned for display-referred VAE encoding.

    Maps scene-linear HDR content into [0, 1] via a blend of hard-clamp
    and Reinhard tone-mapping.

    Args:
        img:               Linear image tensor (..., C), values in [0, ∞).
        compression_ratio: 0.0 → hard clamp [0, 1] (no tone mapping).
                           1.0 → full Reinhard (x / (1 + x)).
                           Values in between blend linearly.

    Returns:
        Tone-mapped tensor in [0, 1].
    """
    img_pos = img.clamp(min=0.0)           # negative values → 0 (sub-black)
    reinhard = img_pos / (1.0 + img_pos)   # Reinhard: maps [0,∞) → [0,1)
    clamped  = img_pos.clamp(max=1.0)      # hard clamp

    if compression_ratio <= 0.0:
        return clamped
    if compression_ratio >= 1.0:
        return reinhard

    return clamped * (1.0 - compression_ratio) + reinhard * compression_ratio


#: Ceiling for the inverse of the soft-compress curve, in the module's
#: convention where 1.0 == 100 nits.
#:
#: DECOMPRESS-POLE FIX: the maths below is right -- Reinhard's inverse IS
#: unbounded as y -> 1 -- but the input clamp of ``1.0 - 1e-7`` chose to
#: evaluate it one ten-millionth from the pole. At r=0.5 that returns ~2.5e6
#: and on the r>=1 branch ~1e7, so any VAE-decoded pixel landing on code 1.0,
#: which is every specular and every practical light source, decoded to a
#: quarter of a billion nits. Both figures also overflow fp16 (max 65504), so
#: an f16 .rhdr or EXR written from them stores Inf, not a big number.
#: 10,000.0 linear is 1,000,000 nits: two orders above the brightest HDR
#: mastering target anyone grades to, so no legitimate highlight is touched,
#: and it still round-trips through fp16 as a finite value.
_DECOMPRESS_MAX_LINEAR = 10_000.0


def _hdr_soft_decompress(img: torch.Tensor, compression_ratio: float) -> torch.Tensor:
    """
    Inverse of _hdr_soft_compress — recovers scene-linear HDR from VAE-decoded output.

    The forward function blends hard-clamp and Reinhard:
        y = (1−r)·clamp(x,0,1) + r·x/(1+x)

    This has two regimes (breakpoint at x=1, encoded as y_break = 1 − r/2):

    For y > y_break  (came from HDR region x > 1):
        x = (y − 1 + r) / (1 − y)

    For y ≤ y_break  (came from SDR region x ≤ 1):
        Solve (1−r)x² + (1−y)x − y = 0  →  positive root:
        x = [−(1−y) + sqrt((1−y)² + 4(1−r)y)] / (2(1−r))
        Special case r=1: degenerates to x = y/(1−y)  (Reinhard inverse)

    For r=0 (hard clamp): x = y  (identity for [0,1])
    For r=1 (Reinhard):   x = y/(1−y+ε)  everywhere

    Verified: compress(decompress(y, r), r) ≈ y for r ∈ {0, 0.25, 0.5, 0.75, 1.0}
    """
    r = float(compression_ratio)

    if r <= 0.0:
        return img.clamp(0.0, 1.0)

    eps = 1e-7
    y   = img.clamp(0.0, 1.0 - eps)   # clamp so denominators stay positive

    if r >= 1.0:
        # Pure Reinhard inverse
        return (y / (1.0 - y + eps)).clamp(max=_DECOMPRESS_MAX_LINEAR)

    # Breakpoint: values above this came from x > 1 (HDR side of clamp)
    y_break = 1.0 - r * 0.5

    # ── HDR branch: x = (y − 1 + r) / (1 − y) ──────────────────────────────
    x_hdr  = (y - 1.0 + r) / (1.0 - y + eps)

    # ── SDR branch: positive root of quadratic ───────────────────────────────
    one_minus_r = 1.0 - r
    one_minus_y = 1.0 - y
    disc   = one_minus_y ** 2 + 4.0 * one_minus_r * y
    x_sdr  = (-one_minus_y + torch.sqrt(disc.clamp(min=0.0))) / (2.0 * one_minus_r + eps)

    return torch.where(y > y_break, x_hdr, x_sdr).clamp(max=_DECOMPRESS_MAX_LINEAR)


def _compute_channel_stats(image: torch.Tensor):
    """
    Compute per-channel mean and std over the spatial dimensions.

    Args:
        image: (B, H, W, C) or (H, W, C) float tensor.

    Returns:
        mean: (C,) tensor
        std:  (C,) tensor (clamped to ≥ 1e-6)
    """
    # flatten to (N, C) where N = B*H*W or H*W
    flat = image.reshape(-1, image.shape[-1])
    mean = flat.mean(dim=0)
    std  = flat.std(dim=0).clamp(min=1e-6)
    return mean, std


# ─────────────────────────────────────────────────────────────────────────────
# RadianceHDRTurboEncoder  (redesigned)
# ─────────────────────────────────────────────────────────────────────────────

_LEGACY_ENCODER_STOP = (
    "this encoder is retired: the latent decoders it fed were removed in "
    "Radiance 3.5.0, so nothing can decode its latent back to HDR and the "
    "graph would render clipped. Replace it with 'VAE Encode (HDR)' (same "
    "image and vae inputs), which 'VAE Decode (HDR)' inverts exactly."
)


class RadianceHDRTurboEncoder:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    DEPRECATED = True   # hidden from the menu; saved graphs still open
    DESCRIPTION = (
        "Legacy. Its TurboDecoder was retired, so no Radiance node decodes this "
        "latent back to HDR (it comes back compressed and clipped). For an HDR "
        "latent use VAE Encode (HDR), which VAE Decode (HDR) inverts exactly."
    )
    """
    ◎ Radiance HDR Turbo Encoder

    Compresses scene-linear HDR images into the [0, 1] range expected by
    LTX-Video's display-referred VAE, then encodes to latents.

    compression_ratio mirrors LTX-Video's `tone_map_compression_ratio`:
      0.0 = hard clamp (preserves mid-tones perfectly, clips highlights)
      0.5 = gentle knee (good balance for 4–6 stop overexposure)
      1.0 = full Reinhard (maximum highlight recovery, ~1 stop luminance shift)

    For strongly over-exposed HDR footage (10+ stops), values 0.7–1.0 work
    best.  For near-SDR content, 0.0–0.2 avoids unnecessary luminance shift.

    Previously this node log-encoded to LogC4/SLog3 before the VAE, which
    was a training-domain mismatch — the VAE never saw log-space data.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "vae": ("VAE",),
                "compression_ratio": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": (
                        "0.0 = hard clamp, 1.0 = full Reinhard tone-map. "
                        "Mirrors LTX-Video tone_map_compression_ratio."
                    ),
                }),
                "exposure_offset": ("FLOAT", {
                    "default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1,
                    "tooltip": "EV offset applied before compression (scene-linear).",
                }),
            }
        }

    RETURN_TYPES  = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION      = "encode"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"

    # VAE encode builds an autograd graph unless guarded: .eval() does not
    # freeze parameters, so requires_grad stays True on every weight.
    @torch.no_grad()
    def encode(self, image: torch.Tensor, vae, compression_ratio: float = 0.5,
               exposure_offset: float = 0.0):
        raise RuntimeError(f"[HDR Turbo Encoder] {_LEGACY_ENCODER_STOP}")

    def _legacy_encode(self, image: torch.Tensor, vae, compression_ratio: float, exposure_offset: float):
        # 1. Exposure offset in scene-linear light
        img = image * (2.0 ** exposure_offset)

        # 2. Soft-knee compression → display-referred [0, 1]
        #    This matches LTX-Video's VAE training domain.
        img_compressed = _hdr_soft_compress(img, compression_ratio)

        # 3. Encode with VAE (now receives display-referred data as trained)
        latent = vae.encode(img_compressed)

        peak_in  = float(img.max())
        peak_out = float(img_compressed.max())
        logger.info(
            "[HDR Encoder] compression_ratio=%.2f  exposure_offset=%.1f EV  "
            "peak_linear=%.3f → compressed=%.3f",
            compression_ratio, exposure_offset, peak_in, peak_out,
        )
        diag_logger.info(
            "HDR_ENCODE compression_ratio=%.2f exposure_offset=%.1f "
            "peak_linear=%.3f peak_compressed=%.3f",
            compression_ratio, exposure_offset, peak_in, peak_out,
        )
        return ({"samples": latent},)


# ─────────────────────────────────────────────────────────────────────────────
# RadianceHDRPerChannelNorm  (new — mirrors vae_per_channel_normalize)
# ─────────────────────────────────────────────────────────────────────────────

class RadianceHDRPerChannelNorm:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    DESCRIPTION = "Per-channel normalise HDR tensors for stable latent space encoding."
    """
    ◎ Radiance HDR Per-Channel Normalizer

    Mirrors LTX-Video's `vae_per_channel_normalize=True`.

    LTX-Video computes per-frame per-channel mean and std, then normalises
    the image before VAE encoding.  Without this, HDR frames with strong
    colour casts (golden-hour, fire, neon) can saturate the VAE's latent
    distribution and decode with colour banding.

    This node:
      1. Computes per-channel mean (μ) and std (σ) over all spatial pixels.
      2. Normalises: x_norm = (x − μ) / σ
      3. Re-centres to [0, 1] for VAE compatibility using a fixed window:
            x_vae = clamp((x_norm + norm_center) / (2 × norm_center), 0, 1)
         Default norm_center=3.0 captures ±3σ — suitable for most HDR content.
      4. Outputs the normalised image + raw stats strings so a downstream
         RadianceHDRPerChannelDenorm node can invert the normalisation after
         VAE decode.

    Connect:
        image → RadianceHDRPerChannelNorm → image_norm → RadianceHDRTurboEncoder
                                          → stats_json → (store in metadata node)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":       ("IMAGE", {"tooltip": "HDR image or batch, usually scene-linear. One mean and std per channel is measured over the whole batch; values beyond +/- norm_center std are clipped in the output."}),
                "norm_center": ("FLOAT", {
                    "default": 3.0, "min": 1.0, "max": 8.0, "step": 0.5,
                    "tooltip": "Window half-width in σ units mapped to [0,1]. 3.0 captures ±3σ.",
                }),
            }
        }

    RETURN_TYPES  = ("IMAGE", "STRING")
    RETURN_NAMES  = ("image_norm", "stats_json")
    FUNCTION      = "normalize"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"

    def normalize(self, image: torch.Tensor, norm_center: float):
        mean, std = _compute_channel_stats(image)   # (C,), (C,)

        # Broadcast (C,) → same shape as image for subtraction
        mu  = mean.reshape(1, 1, 1, -1)
        sig = std.reshape(1, 1, 1, -1)

        x_norm = (image - mu) / sig
        # Map ±norm_center·σ → [0, 1]
        x_vae  = ((x_norm + norm_center) / (2.0 * norm_center)).clamp(0.0, 1.0)

        # Serialise stats for downstream de-normalisation
        import json
        stats = {
            "mean":        mean.tolist(),
            "std":         std.tolist(),
            "norm_center": norm_center,
        }
        stats_json = json.dumps(stats)

        logger.info("[HDR PerChannelNorm] mean=%s  std=%s", mean.tolist(), std.tolist())
        return (x_vae, stats_json)


class RadianceHDRPerChannelDenorm:
    """
    ◎ Radiance HDR Per-Channel De-Normalizer

    Inverts the normalisation applied by RadianceHDRPerChannelNorm.
    Feed the stats_json from the Norm node and the decoded image to recover
    the original scene-linear scale.

    x_linear = x_norm * σ + μ
             = ((x_vae * 2 * norm_center) − norm_center) * σ + μ
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":      ("IMAGE", {"tooltip": "Normalised image (0 to 1), typically the VAE-decoded result of an image that went through HDR Per-Channel Norm."}),
                "stats_json": ("STRING", {"forceInput": True,
                    "tooltip": "stats_json output of HDR Per-Channel Norm (per-channel mean, std and norm_center). Used to undo that normalisation.",
                }),
            }
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("image_linear",)
    FUNCTION      = "denormalize"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    DESCRIPTION = "Invert per-channel normalisation to recover HDR tensor values."

    def denormalize(self, image: torch.Tensor, stats_json: str):
        import json

        # `stats_json` is forceInput, so an empty or malformed value means the
        # Norm node is not wired up. Say that, rather than surfacing a bare
        # JSONDecodeError with a character offset into a string the user never
        # typed.
        try:
            stats = json.loads(stats_json)
            mean_v, std_v = stats["mean"], stats["std"]
            norm_center = float(stats["norm_center"])
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError(
                "RadianceHDRPerChannelDenorm: `stats_json` must be the stats "
                "output of RadianceHDRPerChannelNorm — connect that node's "
                f"stats_json socket to this input. ({type(exc).__name__}: {exc})"
            ) from exc

        mean = torch.tensor(mean_v, dtype=image.dtype, device=image.device)
        std = torch.tensor(std_v, dtype=image.dtype, device=image.device)

        mu  = mean.reshape(1, 1, 1, -1)
        sig = std.reshape(1, 1, 1, -1)

        # Invert: [0,1] → [-norm_center, norm_center] σ → linear
        x_norm   = image * (2.0 * norm_center) - norm_center
        x_linear = x_norm * sig + mu

        logger.info("[HDR PerChannelDenorm] restored mean=%s  std=%s", mean.tolist(), std.tolist())
        return (x_linear,)


# ─────────────────────────────────────────────────────────────────────────────
# CONSOLIDATED: RadianceHDRTurboDecoder removed (Consolidate 1 — 2026-04-26)
#
# The simple vae.decode() + soft-knee-decompress path is superseded by
# ◎ Radiance HDR VAE Decode in Direct HDR mode; learned SDR→HDR recovery
# lives in ◎ Radiance SDR → HDR Universal / Recover (RUDRA pixel model).
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Node registry
# ─────────────────────────────────────────────────────────────────────────────



# ─────────────────────────────────────────────────────────────────────────────
# RadianceHDRLatentEncoder  ·  unified replacement for TurboEncoder + NativeHDREncoder
# ─────────────────────────────────────────────────────────────────────────────

class RadianceHDRLatentEncoder:
    """
    ◎ Radiance HDR Latent Encoder

    Single node that unifies the former TurboEncoder (soft-knee Reinhard,
    LTX-Video–aligned) and NativeHDREncoder (log1p calibration) into one
    surface with a mode selector.

    Modes
    ─────
    Soft-Knee (LTX)   Preferred for LTX-Video, WanVideo, and any VAE trained
                       on display-referred [0, 1] footage.  Uses an analytically
                       invertible Reinhard blend.  compression_ratio mirrors
                       LTX-Video's tone_map_compression_ratio.
    Log Calibration   Fallback for Flux / SDXL VAEs that tolerate a wider input
                       range.  Applies log1p + mean/std distribution calibration.

    Channel normalise toggle
    ────────────────────────
    When enabled, per-channel mean/std normalisation (mirrors LTX-Video's
    vae_per_channel_normalize=True) is applied before the VAE.  The stats are
    returned as a JSON string so RadianceHDRPerChannelDenorm can invert them after
    VAE decode.  Leave off for SDR content.
    """

    CATEGORY    = "FXTD STUDIOS/Radiance/◎ HDR"
    FUNCTION    = "encode"
    RETURN_TYPES  = ("LATENT", "STRING")
    RETURN_NAMES  = ("latent",  "channel_stats")
    DEPRECATED = True   # hidden from the menu; saved graphs still open
    DESCRIPTION = (
        "Legacy. The latent decoders this fed were retired in 3.5.0, so no Radiance "
        "node decodes this latent back to HDR (VAE Decode (HDR) returns it clipped at "
        "1.0). For an HDR latent use VAE Encode (HDR), which VAE Decode (HDR) inverts "
        "exactly. Kept so saved workflows load."
    )

    _MODES = ["Soft-Knee (LTX)", "Log Calibration"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":             ("IMAGE",),
                "vae":               ("VAE",),
                "mode":              (cls._MODES, {"default": "Soft-Knee (LTX)"}),
            },
            "optional": {
                # ── Soft-Knee params ─────────────────────────────────────────
                "compression_ratio": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "[Soft-Knee] 0 = hard clamp, 1 = full Reinhard. "
                               "Mirrors LTX-Video tone_map_compression_ratio.",
                }),
                "exposure_offset":   ("FLOAT", {
                    "default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1,
                    "tooltip": "[Soft-Knee] EV offset applied in scene-linear before compression.",
                }),
                # ── Log Calibration params ───────────────────────────────────
                "energy_normalization": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1,
                    "tooltip": "[Log Calibration] Scale applied before log1p encoding. "
                               "Higher values compress brighter highlights more aggressively.",
                }),
                # ── Shared ───────────────────────────────────────────────────
                "normalize_channels": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Apply per-channel mean/std normalisation before VAE encode "
                               "(mirrors LTX-Video vae_per_channel_normalize). "
                               "Wire channel_stats → RadianceHDRPerChannelDenorm to invert.",
                }),
                "norm_center": ("FLOAT", {
                    "default": 3.0, "min": 1.0, "max": 8.0, "step": 0.5,
                    "tooltip": "[Channel Norm] Window half-width in σ mapped to [0,1]. "
                               "3.0 captures ±3σ — suitable for most HDR content.",
                }),
            },
        }

    # VAE encode builds an autograd graph unless guarded: .eval() does not
    # freeze parameters, so requires_grad stays True on every weight.
    def encode(self, image=None, vae=None, *args, **kwargs):
        raise RuntimeError(f"[HDR Latent Encoder] {_LEGACY_ENCODER_STOP}")

    @torch.no_grad()
    def _legacy_encode(
        self,
        image: "torch.Tensor",
        vae,
        mode: str = "Soft-Knee (LTX)",
        compression_ratio: float = 0.5,
        exposure_offset: float = 0.0,
        energy_normalization: float = 1.0,
        normalize_channels: bool = False,
        norm_center: float = 3.0,
    ):
        import json, math
        img = image.clamp(min=0.0).float()

        # ── Mode-specific compression → [0, 1] ──────────────────────────────
        if mode == "Soft-Knee (LTX)":
            img = img * (2.0 ** exposure_offset)
            img_compressed = _hdr_soft_compress(img, compression_ratio)
        else:  # Log Calibration
            import torch
            img_encoded = torch.log1p(img * energy_normalization) / math.log(10.0)
            mean = img_encoded.mean()
            std  = img_encoded.std().clamp(min=1e-6)
            img_compressed = ((img_encoded - mean) / std).clamp(0.0, 1.0)

        stats_json = ""

        # ── Optional per-channel normalisation ──────────────────────────────
        if normalize_channels:
            import json as _json
            mean_ch, std_ch = _compute_channel_stats(img_compressed)
            mu  = mean_ch.reshape(1, 1, 1, -1)
            sig = std_ch.reshape(1, 1, 1, -1)
            x_norm = (img_compressed - mu) / sig
            img_compressed = ((x_norm + norm_center) / (2.0 * norm_center)).clamp(0.0, 1.0)
            stats_json = _json.dumps({
                "mean": mean_ch.tolist(), "std": std_ch.tolist(), "norm_center": norm_center,
            })

        # ── VAE encode ───────────────────────────────────────────────────────
        # ComfyUI VAEs typically work on (B, H, W, C) → they permute internally.
        latent = vae.encode(img_compressed)

        logger.info(
            "HDRLatentEncoder mode=%s  peak_in=%.3f  peak_out=%.3f  ch_norm=%s",
            mode, float(image.max()), float(img_compressed.max()), normalize_channels,
        )
        return ({"samples": latent}, stats_json)


NODE_CLASS_MAPPINGS = {
    "RadianceHDRTurboEncoder":      RadianceHDRTurboEncoder,      # kept as legacy reference
    "RadianceHDRPerChannelNorm":    RadianceHDRPerChannelNorm,     # kept as legacy reference
    "RadianceHDRPerChannelDenorm":  RadianceHDRPerChannelDenorm,   # kept as legacy reference
    "RadianceHDRLatentEncoder":     RadianceHDRLatentEncoder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceHDRTurboEncoder":      "◎ Radiance HDR Turbo Encoder",
    "RadianceHDRPerChannelNorm":    "◎ Radiance HDR Per-Channel Norm",
    "RadianceHDRPerChannelDenorm":  "◎ Radiance HDR Per-Channel Denorm",
    "RadianceHDRLatentEncoder":     "◎ HDR Latent Encoder",
}
