"""
◎ Radiance SDR → HDR Recover / Universal
═════════════════════════════════════════

One-click SDR→HDR conversion for any image or video (frame batch).

Pipeline
--------
1. Inverse OETF          — decode the SDR transfer curve to scene-linear.
2. Adaptive soft-knee    — per-frame knee from a luma percentile (or manual),
   inverse tone mapping    EMA-smoothed across frames so video doesn't flicker.
3. Highlight expansion   — hue-preserving luminance expansion above the knee
                           up to peak_nits, hard shoulder controlled by gamma.
4. Output transform      — linear Rec.709, ACES2065-1/AP0 for EXR, or
                           Rec.2020 PQ / HLG delivery encoding. Every output
                           follows BT.2408: linear 1.0 = reference white
                           (203 nits default), the convention HDR Encode,
                           Write and OpenColorIO's Rec.2100 displays share.

Works on IMAGE tensors of shape [B,H,W,C]; a video is simply a batch of
frames, so the same node handles stills, batches, and footage. B == 1
automatically behaves as a still image (temporal smoothing is a no-op).

Pure GPU math by default — no checkpoints required. The learned path is
RUDRA in its pixel form: a direct SDR-pixel → scene-linear network
(``radiance.pixel_sdr2hdr``, checkpoint ``sdr2hdr_pixel_image.pt``) for stills
and independent frames, and the motion-aligned temporal residual model
(``radiance.temporal_rudra``) over 5/7/9 adjacent RGB frames for ordered
video. Neither needs a VAE. Universal exposes Expand, Recover, and Hybrid
modes; its learned modes reconstruct only clipped highlights/crushed shadows
and fall back to deterministic expansion when no checkpoint is installed, and
say so in the ``report`` output. The dedicated Recover node is the same
learned path without the deterministic fallback: it raises when it cannot
run. The latent-space RUDRA decoders (VAE + distilled decoder) were retired
in 3.5.0.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch

from radiance.color.ops import (
    M_REC709_TO_ACES2065_1,
    M_REC709_TO_BT2020,
    apply_matrix_3x3,
    linear_to_hlg_bt2100,
    linear_to_pq_bt2408,
)

logger = logging.getLogger("radiance.nodes.hdr.uplift_universal")

_EPS = 1e-6


# ─────────────────────────────────────────────────────────────────────────────
#  Transfer-function helpers
# ─────────────────────────────────────────────────────────────────────────────

def _inverse_oetf(img: torch.Tensor, curve: str) -> torch.Tensor:
    """Decode an SDR-encoded image to scene-linear [0, 1]."""
    if curve == "sRGB":
        return torch.where(img <= 0.04045, img / 12.92,
                           ((img + 0.055) / 1.055).clamp(min=_EPS) ** 2.4)
    if curve == "Rec.709":
        return torch.where(img < 0.0812, img / 4.5,
                           ((img + 0.099) / 1.099).clamp(min=_EPS) ** (1.0 / 0.45))
    if curve == "Gamma 2.2":
        return img.clamp(min=0.0) ** 2.2
    if curve == "Gamma 2.4":
        return img.clamp(min=0.0) ** 2.4
    return img  # "None" — already linear


def _luma(rgb: torch.Tensor) -> torch.Tensor:
    """Rec.709 luminance from linear RGB, shape [..., H, W]."""
    return (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2])


# ─────────────────────────────────────────────────────────────────────────────
#  Core expansion math (kept as free functions so they are unit-testable)
# ─────────────────────────────────────────────────────────────────────────────

#: torch.quantile's historic size cap is on the input tensor's TOTAL element
#: count, not on the length of the reduced axis. A single 8K frame is 33.2M
#: pixels, so the default adaptive knee mode raised "quantile() input tensor is
#: too large" on anything past roughly 4K -- on the node whose entire purpose is
#: film-resolution HDR work.
#:
#: QUANTILE-GUARD FIX: the guard used to compare the cap against `flat.shape[1]`
#: alone, so a 16-frame 1080p batch passed it at n = 2,073,600 and then handed
#: torch 33,177,600 elements, twice the cap, on the adaptive-knee path that is
#: this node's default. Compare against numel. Newer torch (2.6+) lifted the cap
#: on CPU, which is why this never showed up in a test run on a current build;
#: the guard still has to be right for the versions that enforce it and for any
#: backend that reinstates it.
_QUANTILE_MAX_ELEMS = 2 ** 24


def _row_quantile(flat: torch.Tensor, percentile: float) -> torch.Tensor:
    """
    Per-row quantile that works at any resolution.

    Falls back to ``torch.kthvalue`` (which has no size cap) for rows too large
    for ``torch.quantile``. kthvalue returns the exact k-th order statistic
    rather than interpolating between neighbours; on image-sized inputs the two
    agree to ~1e-5, which is far below the precision a tone-mapping knee needs.
    """
    n = flat.shape[1]
    if n == 0:
        return flat.new_zeros((flat.shape[0],))
    if flat.numel() <= _QUANTILE_MAX_ELEMS:
        return torch.quantile(flat, percentile, dim=1)
    k = max(1, min(n, int(round(float(percentile) * (n - 1))) + 1))
    return torch.kthvalue(flat, k, dim=1).values


def _adaptive_knees(luma: torch.Tensor, percentile: float,
                    smoothing: float) -> torch.Tensor:
    """
    Per-frame knee = `percentile` of frame luma, EMA-smoothed along the batch
    (time) axis with weight `smoothing` in [0, 1). smoothing=0 → per-frame.
    Returns a tensor of shape [B].
    """
    b = luma.shape[0]
    if b == 0:                                   # empty batch — nothing to do
        return luma.new_zeros((0,))
    flat = luma.reshape(b, -1)
    q = _row_quantile(flat, percentile).clamp(0.05, 0.99)
    if smoothing <= 0.0 or b == 1:
        return q
    knees = torch.empty_like(q)
    ema = q[0]
    for i in range(b):
        ema = smoothing * ema + (1.0 - smoothing) * q[i]
        knees[i] = ema
    return knees


def _soft_knee_expand(luma: torch.Tensor, knee: torch.Tensor,
                      peak_scale: float, shoulder_gamma: float) -> torch.Tensor:
    """
    Inverse tone map: identity below the knee, expansion from the knee to
    `peak_scale` at L == 1, with the gradient continuous across the join.
    `knee` broadcasts per frame [B]. Returns the expanded luminance.

    The join used to be a bare power curve, ``k + (peak - k) * t**gamma``.
    That is continuous in value and discontinuous in slope: below the knee the
    curve is the identity, gradient 1; immediately above it the gradient is
    ``(peak - k) * gamma * t**(gamma - 1) / (1 - k)``, which for any gamma > 1
    goes to *zero* as t → 0. Measured at the shipped defaults (knee 0.75,
    peak_scale 10.0):

        shoulder_gamma   gradient below   gradient above
              1.0             1.000            37.05
              1.6             1.000             0.083
              2.5             1.000             0.000

    A gradient that drops from 1.0 to 0.08 across a single code value is a
    visible ridge in any smooth gradient crossing the knee — a sky, a skin
    falloff, the soft edge of a practical — and in adaptive mode the knee moves
    per frame, so on video the ridge crawls. BT.2446 Method B specifies the
    opposite: "the gradient of the exponential function is set to unity at the
    breakpoint", and suggests a Bezier blend "to avoid artefacts at the join".
    This module's own `_soft_peak_limit` has always been C¹ at its knee; only
    the expansion was not.

    The shoulder is now

        f(t) = s·t + (1 − s)·t^gamma,   s = (1 − k) / (peak_scale − k)

    which satisfies f(0) = 0, f(1) = 1 and f'(0) = s — and s is exactly the
    value that makes the gradient in luminance space equal 1 at the knee. It
    is monotonic for gamma ≥ 1 and keeps `shoulder_gamma`'s meaning: how hard
    the expansion ramps once past the knee.

    gamma = 1 collapses to a straight line from (k, k) to (1, peak_scale),
    which has an unavoidable gradient step — a linear ramp to peak is what
    that asks for. Values at or above 2 land within 0.2% of unity.
    """
    k = knee.view(-1, *([1] * (luma.dim() - 1)))
    span = (1.0 - k).clamp(min=_EPS)
    t = ((luma - k) / span).clamp(0.0, 1.0)

    # The gradient the shoulder must start with for the join to be C¹.
    s = (span / (peak_scale - k).clamp(min=_EPS)).clamp(0.0, 1.0)

    g = max(float(shoulder_gamma), 1.0)
    shaped = s * t + (1.0 - s) * t.clamp(min=0.0) ** g

    expanded = k + (peak_scale - k) * shaped
    return torch.where(luma > k, expanded, luma)


def _soft_peak_limit(rgb: torch.Tensor, peak_scale: float,
                     knee_ratio: float = 0.9) -> torch.Tensor:
    """Hue-preserving max-channel shoulder bounded by ``peak_scale``.

    Pixels whose brightest channel is below ``knee_ratio * peak_scale`` are
    unchanged. Brighter ones are scaled, all three channels by one gain, so the
    brightest channel approaches the peak smoothly; a final numerical guard
    guarantees that no channel of the learned reconstruction exceeds the
    mastering target.

    The shoulder used to act on luminance. Luminance weights red at 0.21 and
    blue at 0.07, so a saturated highlight could sit under the peak in
    luminance with one channel far above it: measured on a user's sunset,
    10% of pixels had a channel over a 1,000-nit peak and red reached
    3,500 nits. HDR10 encodes channels, not luminance, so every one of those
    pixels would have clipped per channel at the encoder, shifting its hue.
    Bounding the max channel also bounds luminance (``Y <= max(R,G,B)``).
    """
    peak = max(float(peak_scale), _EPS)
    knee = peak * float(knee_ratio)
    rgb = rgb.clamp(min=0.0)
    m = rgb.amax(dim=-1)
    span = max(peak - knee, _EPS)
    compressed = knee + span * torch.tanh((m - knee).clamp(min=0.0) / span)
    target_m = torch.where(m > knee, compressed, m).clamp(max=peak)
    gain = target_m / m.clamp(min=_EPS)
    limited = rgb * gain.unsqueeze(-1)

    # Floating-point roundoff can leave a channel a few ulps above the target.
    limited_m = limited.amax(dim=-1).clamp(min=_EPS)
    guard = torch.clamp(peak / limited_m, max=1.0)
    return limited * guard.unsqueeze(-1)


def _shadow_mask(luma: torch.Tensor, threshold: float) -> torch.Tensor:
    """Soft mask for crushed-shadow recovery/QC, 1 at black and 0 at threshold."""
    t = max(float(threshold), _EPS)
    return ((t - luma) / t).clamp(0.0, 1.0)


def _clipped_highlight_mask(sdr_rgb: torch.Tensor,
                            threshold: float) -> torch.Tensor:
    """Mask display-referred pixels that contain little recoverable headroom.

    Channel clipping is included alongside luma so saturated red, green, or
    blue highlights are not missed. The smoothstep ramp is exactly zero below
    ``threshold`` and reaches one at SDR code value 1.0.
    """
    t = min(max(float(threshold), 0.0), 1.0 - _EPS)
    evidence = torch.maximum(_luma(sdr_rgb), sdr_rgb.max(dim=-1).values)
    x = ((evidence - t) / max(1.0 - t, _EPS)).clamp(0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _pixel_checkpoint_available(checkpoint_path: str) -> bool:
    """
    Whether a direct-pixel checkpoint can actually be resolved.

    `resolve_pixel_checkpoint` searches, in order: the explicit path, the
    RADIANCE_SDR2HDR_PIXEL environment variable, then models/radiance. The
    Auto backend used to gate on `bool(pixel_checkpoint.strip())` instead of
    asking the resolver, so a user with an installed checkpoint but an empty
    path widget -- the stock defaults -- never got the direct-pixel model at
    all. It silently fell through to legacy VAE RUDRA or plain expansion,
    contradicting both the `learned_backend` tooltip ("Auto prefers ... the
    direct-pixel model ...") and the `pixel_checkpoint` tooltip ("Empty
    searches models/radiance and RADIANCE_SDR2HDR_PIXEL").
    """
    try:
        from radiance.pixel_sdr2hdr import resolve_pixel_checkpoint
    except Exception as exc:  # noqa: BLE001 — optional dependency chain
        logger.debug("Direct-pixel backend unavailable: %s", exc)
        return False
    try:
        return resolve_pixel_checkpoint(str(checkpoint_path)) is not None
    except Exception as exc:  # noqa: BLE001
        logger.debug("Direct-pixel checkpoint lookup failed: %s", exc)
        return False


def _encode_output(hdr: torch.Tensor, output_encoding: str,
                   peak_scale: float, white_scale: float = 1.0) -> torch.Tensor:
    """Mastering output transform.

    ``hdr`` is the internal working signal: display-linear Rec.709 with
    1.0 == 100 nits. Every output leaves in Radiance's one convention,
    BT.2408: linear 1.0 == reference white (``white_scale * 100`` nits), the
    same convention HDR Encode, Write's ``hdr_reference_nits``, HDR
    Diagnostics and OpenColorIO's Rec.2100 displays use.

    3.5.0: Linear and AP0 used to leave in the internal 1.0 == 100 nit
    units, so SDR white sat at 2.03 and a Linear EXR re-encoded to PQ by
    HDR Encode or Write (1.0 == 203 nits) came out twice as bright as this
    node's own PQ output. HLG used to put the mastering peak at signal 1.0,
    which placed reference white at 69.7 % instead of BT.2408's 75 %.
    """
    ws = max(float(white_scale), _EPS)
    norm = hdr / ws
    ref_nits = ws * 100.0
    if output_encoding.startswith("PQ"):
        delivery = apply_matrix_3x3(norm, M_REC709_TO_BT2020).clamp(min=0.0)
        return linear_to_pq_bt2408(delivery, peak_nits=float(peak_scale) * 100.0,
                                   reference_white_nits=ref_nits)
    if output_encoding == "HLG":
        delivery = apply_matrix_3x3(norm, M_REC709_TO_BT2020).clamp(min=0.0)
        return linear_to_hlg_bt2100(delivery, reference_white_nits=ref_nits)
    if output_encoding == "Linear ACES2065-1 (AP0)":
        return apply_matrix_3x3(norm, M_REC709_TO_ACES2065_1)
    return norm


# ─────────────────────────────────────────────────────────────────────────────
#  Shared learned-recovery core
# ─────────────────────────────────────────────────────────────────────────────

class _RudraRecoveryCore:
    """Shared learned-recovery execution used by Recover and Universal."""

    @staticmethod
    def _temporal_reconstruct(
        source: torch.Tensor,
        deterministic: torch.Tensor,
        highlight_mask: torch.Tensor,
        shadow_mask: torch.Tensor,
        peak_scale: float,
        window_size: int,
        checkpoint_path: str,
        preserve_source_outside: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from radiance.temporal_rudra import (
            load_temporal_rudra_weights,
            recover_temporal_residual,
        )

        if source.shape[0] < 5:
            raise RuntimeError("temporal RUDRA requires at least five ordered frames")
        model = load_temporal_rudra_weights(checkpoint_path, source.device)
        if model is None:
            from radiance.model.paths import describe_search
            from radiance.temporal_rudra import (
                TEMPORAL_CHECKPOINT_ENV, TEMPORAL_CHECKPOINT_PATTERNS,
            )
            raise RuntimeError(
                "no temporal checkpoint installed; expected "
                + describe_search(TEMPORAL_CHECKPOINT_PATTERNS[:1], TEMPORAL_CHECKPOINT_ENV)
            )
        out, h_conf, s_conf = recover_temporal_residual(
            source, deterministic, highlight_mask, shadow_mask, model,
            peak_scale, window_size, preserve_source_outside,
        )
        return _soft_peak_limit(out, peak_scale), h_conf, s_conf

    @staticmethod
    def _pixel_reconstruct(
        sdr_linear: torch.Tensor,
        base_hdr: torch.Tensor,
        mask: torch.Tensor,
        checkpoint_path: str,
        blend: float,
        peak_scale: float,
        tile_size: int,
        tile_overlap: int,
        recovery_mode: str,
        strength: float,
        highlight_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the direct-pixel SDR2HDRNet and blend it into evidence masks.

        The network consumes canonical sRGB code values and returns
        scene-linear RGB with 1.0 == 10,000 nits, in the input's primaries
        (Rec.709 here). Radiance's working space is scene-linear Rec.709 with
        1.0 == 100 nits, so only the scale changes.

        Inside clipped highlights the network supplies BRIGHTNESS only; the
        colour comes from the source. A clipped channel carries no information
        about its own value, and the network rebuilds each channel separately
        from a per-channel inverse tone curve that is steepest exactly at the
        clip, so where red has clipped and green has not, red alone is pushed
        up. On a sunset whose channels clip in turn (red, then green, then blue)
        that drew a false-colour ring at each clip boundary, cast white areas
        red (R about 2.2x G) and quadrupled chroma noise on the glints. The
        released checkpoints were also trained on a corpus that almost never
        clipped (RUDRA's legacy -1 EV render), so clipped colour is outside
        what they learned. Taking the learned luminance and the source
        chromaticity leaves no hue to invent, and the learned lift is let in
        as the source goes to white (see ``fullness``), never below the
        deterministic base. Shadows keep the network's own colour: that is
        where it was trained on it.
        """
        from radiance.pixel_sdr2hdr import linear_to_srgb, predict_pixel_sdr2hdr

        canonical_srgb = linear_to_srgb(sdr_linear.clamp(0.0, 1.0))
        predicted = predict_pixel_sdr2hdr(
            canonical_srgb, checkpoint_path=checkpoint_path,
            tile_size=int(tile_size), tile_overlap=int(tile_overlap),
            recovery_mode=str(recovery_mode), strength=float(strength),
        )
        # There used to be a Rec.2020 -> Rec.709 matrix here. The network is a
        # per-channel mapping and never changes primaries (RUDRA's inference
        # writes its output as-is), so the matrix only over-saturated, and on
        # a warm highlight it multiplied the red cast by another 1.5x.
        recovered = (predicted * 100.0).clamp(min=0.0)

        # How far the source went to white: the MEDIAN channel's code value.
        # One clipped channel says the pixel is bright in that primary and
        # nothing about how bright; the network's answer there is driven by
        # that channel's inverse curve alone, and switching it on at the first
        # clipped channel doubled luminance in one step (80 -> 185 nits across
        # a smooth sky) and drew a hard plateau edge round the sun. Two
        # channels at clip is a highlight the tone curve has flattened, and
        # that is where learned brightness is let in, fading in from code 0.85.
        median_code = canonical_srgb.median(dim=-1).values
        fullness = ((median_code - 0.85) / 0.15).clamp(0.0, 1.0)
        fullness = fullness * fullness * (3.0 - 2.0 * fullness)
        base_y = _luma(base_hdr).clamp(min=_EPS)
        lift = (_luma(recovered) - base_y).clamp(min=0.0)   # never darker than the base
        target_y = base_y + lift * fullness
        hue_kept = base_hdr.clamp(min=0.0) * (target_y / base_y).unsqueeze(-1)
        if highlight_mask is None:
            share = torch.ones_like(mask)
        else:
            share = (highlight_mask / mask.clamp(min=_EPS)).clamp(0.0, 1.0)
        share = share.unsqueeze(-1)
        recovered = hue_kept * share + recovered * (1.0 - share)

        recovered = _soft_peak_limit(recovered, peak_scale)
        weight = (mask * float(blend)).unsqueeze(-1)
        # Limit the LEARNED signal only, then blend -- do not re-limit the
        # result. _soft_peak_limit is a tanh compressor and not idempotent, so
        # limiting twice pulled highlights below the peak; and re-limiting the
        # blend altered pixels where the mask is zero, breaking the contract
        # that pixels outside the recovery masks are preserved exactly. The
        # peak guarantee still holds: the blend is convex and both inputs
        # respect peak_scale.
        return base_hdr * (1.0 - weight) + recovered * weight


# ─────────────────────────────────────────────────────────────────────────────
#  Dedicated learned recovery product
# ─────────────────────────────────────────────────────────────────────────────

class RadianceSDRToHDRRecover(_RudraRecoveryCore):
    """Recover clipped highlights and crushed shadows with the RUDRA pixel model.

    Unlike deterministic expansion, this node is explicitly reconstructive and
    it does not fall back: with no checkpoint installed it raises, so a graph
    that asks for learned recovery never silently ships expansion. Stills and
    independent frames run the direct-pixel network; ordered video (5+
    frames, batch_mode "Video Frames") runs the motion-aligned temporal
    residual model when its checkpoint is installed and otherwise the pixel
    model per frame. Pixels outside the recovery masks are preserved exactly.
    """

    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    DESCRIPTION = (
        "Learned SDR→HDR reconstruction for clipped highlights and crushed "
        "shadows (RUDRA pixel model, temporal residual model for video). No "
        "VAE needed; raises instead of falling back when no checkpoint is "
        "installed. Pixels outside the recovery masks are preserved exactly."
    )
    FUNCTION = "recover"
    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "MASK", "MASK")
    RETURN_NAMES = ("image", "highlight_mask", "shadow_mask",
                    "highlight_confidence", "shadow_confidence")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "SDR still, image batch, or ordered video frames."}),
                "inverse_oetf": (["sRGB", "Rec.709", "Gamma 2.2", "Gamma 2.4", "None"], {"default": "sRGB",
                    "tooltip": "Transfer curve the SDR input is encoded with, decoded to linear before recovery. "
                               "None treats the input as already linear. Input is clamped to 0-1 first."}),
                "peak_nits": ("FLOAT", {"default": 1000.0, "min": 200.0, "max": 10000.0, "step": 50.0,
                    "tooltip": "Mastering display peak in nits. Recovered highlights keep the source hue and no "
                               "channel exceeds it; it is also the PQ encode peak."}),
                "highlight_threshold": ("FLOAT", {"default": 0.98, "min": 0.8, "max": 0.999, "step": 0.001,
                    "tooltip": "SDR code-value threshold used to identify clipped luma or RGB channels."}),
                "shadow_threshold": ("FLOAT", {"default": 0.05, "min": 0.001, "max": 0.5, "step": 0.005,
                    "tooltip": "Linear-luma threshold used to identify crushed shadows."}),
                "highlight_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Scales the clipped-highlight recovery mask: 1 applies the learned highlights fully "
                               "inside the mask, 0 leaves highlights at their decoded SDR level."}),
                "shadow_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Scales the crushed-shadow recovery mask: 1 applies the learned shadow detail fully "
                               "inside the mask, 0 leaves shadows at their decoded SDR level."}),
                "output_encoding": (["Linear", "Linear ACES2065-1 (AP0)", "PQ (HDR10)", "HLG"], {"default": "Linear",
                    "tooltip": "Linear: scene-linear Rec.709, 1.0 = reference_white_nits. AP0: the same in ACES2065-1. "
                               "PQ (HDR10) and HLG: Rec.2020 delivery code values (BT.2100)."}),
            },
            "optional": {
                "batch_mode": (["Independent Images", "Video Frames"], {"default": "Independent Images",
                    "tooltip": "Video Frames treats the batch as an ordered clip and prefers the temporal model."}),
                "pixel_checkpoint": ("STRING", {"default": "",
                    "tooltip": "Direct-pixel .pt checkpoint. Empty searches models/radiance and RADIANCE_SDR2HDR_PIXEL, and downloads the default RUDRA model (~5 MB) on first use unless RADIANCE_ALLOW_DOWNLOADS=0."}),
                "pixel_tile_size": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 64,
                    "tooltip": "Tile size used only when the whole frame does not fit in memory. The model normalises over its input, so whole-frame inference is more accurate and is always tried first."}),
                "pixel_tile_overlap": ("INT", {"default": 64, "min": 0, "max": 512, "step": 16,
                    "tooltip": "Overlap in pixels between tiles, feathered to hide seams. Used only when tiling "
                               "kicks in; must be smaller than pixel_tile_size."}),
                "temporal_window": ([5, 7, 9], {"default": 5,
                    "tooltip": "Adjacent frames used by the temporal residual model."}),
                "temporal_checkpoint": ("STRING", {"default": "",
                    "tooltip": "Optional temporal checkpoint path. Empty uses RADIANCE_TEMPORAL_RUDRA or models/radiance."}),
                "reference_white_nits": ("FLOAT", {
                    "default": 203.0, "min": 100.0, "max": 1000.0, "step": 1.0,
                    "tooltip": (
                        "Output convention: linear 1.0 = this many nits (BT.2408 reference "
                        "white, 203). The same value HDR Encode and Write use, so Linear "
                        "output re-encodes to the same PQ/HLG this node writes. Recover "
                        "keeps unclipped pixels at their SDR display level (SDR 1.0 = 100 "
                        "nits); only the recovered highlights and shadows change."
                    ),
                }),
            },
        }

    @torch.no_grad()
    def recover(self, image: torch.Tensor, inverse_oetf: str,
                peak_nits: float, highlight_threshold: float,
                shadow_threshold: float, highlight_strength: float,
                shadow_strength: float, output_encoding: str,
                batch_mode: str = "Independent Images",
                pixel_checkpoint: str = "", pixel_tile_size: int = 512,
                pixel_tile_overlap: int = 64,
                temporal_window: int = 5, temporal_checkpoint: str = "",
                reference_white_nits: float = 203.0,
                **_legacy):
        # `_legacy` swallows rudra_size / vae / model_meta from pre-3.5 graphs.
        if _legacy.get("vae") is not None:
            logger.info("[SDR → HDR Recover] a VAE is connected but no longer used; "
                        "recovery runs on the pixel model.")
        img = torch.nan_to_num(image.float(), nan=0.0, posinf=1.0, neginf=0.0)
        if img.dim() == 3:
            img = img.unsqueeze(0)
        if img.shape[0] == 0:                   # empty batch → empty result
            zeros = img[..., 0]
            return (img, zeros, zeros, zeros, zeros)
        rgb, extra = img[..., :3], img[..., 3:]
        peak_scale = max(float(peak_nits), 100.0) / 100.0
        lin = _inverse_oetf(rgb.clamp(0.0, 1.0), inverse_oetf)
        highlights = _clipped_highlight_mask(rgb.clamp(0.0, 1.0), highlight_threshold)
        shadows = _shadow_mask(_luma(lin).clamp(0.0, 1.0), shadow_threshold)
        h_weighted = highlights * float(highlight_strength)
        s_weighted = shadows * float(shadow_strength)
        recovery_mask = torch.maximum(h_weighted, s_weighted)

        errors: List[str] = []
        hdr = None
        h_conf = s_conf = None
        if img.shape[0] >= 5 and batch_mode == "Video Frames":
            try:
                knees = _adaptive_knees(_luma(lin).clamp(0.0, 1.0), 0.75, 0.85)
                expanded_luma = _soft_knee_expand(
                    _luma(lin).clamp(0.0, 1.0), knees, peak_scale, 1.6,
                )
                deterministic = lin * (
                    expanded_luma / _luma(lin).clamp(min=_EPS)
                ).unsqueeze(-1)
                hdr, h_conf, s_conf = self._temporal_reconstruct(
                    lin, deterministic, h_weighted, s_weighted,
                    peak_scale, int(temporal_window), str(temporal_checkpoint), True,
                )
            except Exception as exc:  # noqa: BLE001 — try the pixel model next
                errors.append(f"temporal model: {exc}")
                hdr = None
        if hdr is None:
            try:
                mode = ("all" if highlight_strength > 0 and shadow_strength > 0
                        else "shadows" if shadow_strength > 0 else "highlights")
                hdr = self._pixel_reconstruct(
                    lin, lin, recovery_mask, str(pixel_checkpoint), 1.0,
                    peak_scale, int(pixel_tile_size), int(pixel_tile_overlap),
                    mode, 1.0, highlight_mask=h_weighted,
                )
                h_conf, s_conf = h_weighted, s_weighted
            except Exception as exc:  # noqa: BLE001 — convert to an actionable node error
                errors.append(f"pixel model: {exc}")
                raise RuntimeError(
                    "SDR → HDR Recover could not run the learned path: "
                    + "; ".join(errors)
                    + ". Install sdr2hdr_pixel_image.pt in models/radiance (or set "
                    "RADIANCE_SDR2HDR_PIXEL), or use SDR → HDR Universal, which "
                    "falls back to deterministic expansion."
                ) from exc

        white_scale = min(max(float(reference_white_nits), 100.0) / 100.0, peak_scale)
        out = _encode_output(hdr, output_encoding, peak_scale, white_scale)
        if extra.shape[-1] > 0:
            out = torch.cat([out, extra], dim=-1)
        return (out, highlights, shadows, h_conf, s_conf)


# ─────────────────────────────────────────────────────────────────────────────
#  Universal orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class RadianceSDRToHDRUniversal(_RudraRecoveryCore):
    """Orchestrate deterministic expansion, learned recovery, or a hybrid."""

    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    DESCRIPTION = (
        "SDR→HDR orchestrator: deterministic Expand, learned Recover, or "
        "Hybrid. Recover and Hybrid run the RUDRA pixel model (or the temporal "
        "model on ordered video); with no checkpoint installed they fall back "
        "to Expand, and the report output says so. Professional output "
        "transforms and separate highlight/shadow masks. No VAE needed."
    )
    FUNCTION = "convert"
    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "MASK", "MASK", "STRING")
    RETURN_NAMES = ("image", "highlight_mask", "shadow_mask",
                    "highlight_confidence", "shadow_confidence", "report")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "SDR still, image batch, or video frames [B,H,W,C]."}),
                "inverse_oetf": (["sRGB", "Rec.709", "Gamma 2.2", "Gamma 2.4", "None"], {"default": "sRGB",
                    "tooltip": "Transfer curve the SDR input is encoded with, decoded to linear before expansion. "
                               "None treats the input as already linear. Input is clamped to 0-1 first."}),
                "peak_nits": ("FLOAT", {
                    "default": 1000.0, "min": 200.0, "max": 10000.0, "step": 50.0,
                    "tooltip": (
                        "Mastering display peak. This is the ceiling the output "
                        "is limited to and the value the PQ/HLG encode targets — "
                        "not where SDR white lands. See reference_white_nits."
                    ),
                }),
                "reference_white_nits": ("FLOAT", {
                    "default": 203.0, "min": 100.0, "max": 1000.0, "step": 1.0,
                    "tooltip": (
                        "Where SDR diffuse white (code 1.0) lands, in nits. "
                        "203 is the ITU-R BT.2408 HDR Reference White — a white "
                        "shirt, a page, a cloud. Headroom above it belongs to "
                        "speculars. Raise it for a brighter grade; setting it to "
                        "peak_nits restores the pre-3.4 behaviour of slamming "
                        "SDR white to the display peak. Also the output unit: "
                        "Linear/AP0 1.0 = this many nits, matching HDR Encode "
                        "and Write's hdr_reference_nits."
                    ),
                }),
                "knee_mode": (["adaptive", "manual"], {"default": "adaptive",
                    "tooltip": "adaptive: the knee is measured per frame as a percentile of linear luma (see knee). "
                               "manual: knee is used directly as a fixed linear-luma level."}),
                "knee": ("FLOAT", {"default": 0.75, "min": 0.05, "max": 0.99, "step": 0.01,
                    "tooltip": "Where expansion starts. adaptive: the luma percentile (0.75 = 75th percentile of the "
                               "frame). manual: a linear-luma level. Below the knee the image is unchanged."}),
                "shoulder_gamma": ("FLOAT", {
                    "default": 2.0, "min": 1.0, "max": 6.0, "step": 0.05,
                    "tooltip": (
                        "How hard highlights ramp once past the knee. The join "
                        "keeps a continuous gradient at any value ≥ 1; 2.0 and "
                        "above sit within 0.2% of unity there. 1.0 is a "
                        "straight line to peak and steps the gradient."
                    ),
                }),
                "temporal_smoothing": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 0.98, "step": 0.01,
                    "tooltip": "Frame-to-frame smoothing of the adaptive knee to stop flicker (0 = per-frame knee, "
                               "higher = steadier). Used only with knee_mode adaptive and batch_mode Video Frames."}),
                "output_encoding": (["Linear", "Linear ACES2065-1 (AP0)", "PQ (HDR10)", "HLG"], {"default": "Linear",
                    "tooltip": "Linear: scene-linear Rec.709, 1.0 = reference_white_nits. AP0: the same in ACES2065-1. "
                               "PQ (HDR10) and HLG: Rec.2020 delivery code values (BT.2100)."}),
            },
            "optional": {
                "rudra_blend": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "How much of the learned reconstruction is blended into the recovery masks. 0 disables it."}),
                "batch_mode": (["Independent Images", "Video Frames"], {"default": "Independent Images",
                    "tooltip": "Video Frames treats the batch as an ordered clip: enables temporal knee smoothing "
                               "and the temporal learned backend. Independent Images processes each frame alone."}),
                "shadow_threshold": ("FLOAT", {"default": 0.05, "min": 0.001, "max": 0.5, "step": 0.005,
                    "tooltip": "Linear-luma level below which shadows count as crushed. Shapes the "
                               "shadow_mask output; changes the image only when pixel_recovery_mode "
                               "is 'all' or 'shadows'."}),
                "processing_mode": (["Expand", "Recover", "Hybrid"], {"default": "Hybrid",
                    "tooltip": "Expand: deterministic only. Recover: learned reconstruction only. Hybrid: expansion plus masked learned recovery."}),
                "highlight_threshold": ("FLOAT", {"default": 0.98, "min": 0.8, "max": 0.999, "step": 0.001,
                    "tooltip": "Clipping threshold for Recover/Hybrid. Appended for saved-workflow compatibility."}),
                "temporal_window": ([5, 7, 9], {"default": 5,
                    "tooltip": "Adjacent video frames used by temporal RUDRA."}),
                "temporal_checkpoint": ("STRING", {"default": "",
                    "tooltip": "Optional Phase 3 checkpoint path. Empty uses the configured default."}),
                "learned_backend": (["Auto", "Direct Pixel", "Temporal"], {"default": "Auto",
                    "tooltip": "Auto: temporal model on ordered video when installed, else the direct-pixel model. "
                               "Direct Pixel: pixel model per frame. Temporal: temporal model only (video, 5+ frames)."}),
                "pixel_checkpoint": ("STRING", {"default": "",
                    "tooltip": "Direct-pixel .pt checkpoint. Empty searches models/radiance and RADIANCE_SDR2HDR_PIXEL, and downloads the default RUDRA model (~5 MB) on first use unless RADIANCE_ALLOW_DOWNLOADS=0."}),
                "pixel_tile_size": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 64,
                    "tooltip": "Tile size used only when the whole frame does not fit in memory. The model normalises over its input, so whole-frame inference is more accurate and is always tried first."}),
                "pixel_tile_overlap": ("INT", {"default": 64, "min": 0, "max": 512, "step": 16,
                    "tooltip": "Overlap in pixels between tiles, feathered to hide seams. Used only when tiling "
                               "kicks in; must be smaller than pixel_tile_size."}),
                "pixel_recovery_mode": (["highlights", "all", "shadows", "off"], {"default": "highlights",
                    "tooltip": "Highlights is safest and avoids hallucinating chroma in deep shadows."}),
                "pixel_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Multiplier on the pixel model's learned residual over its own inverse tone curve "
                               "(0 = curve only, 1 = as trained, 2 = doubled). Direct Pixel backend only."}),
            },
        }

    @torch.no_grad()
    def convert(self, image: torch.Tensor, inverse_oetf: str, peak_nits: float,
                knee_mode: str, knee: float, shoulder_gamma: float,
                temporal_smoothing: float, output_encoding: str,
                # Appended rather than slotted in beside peak_nits, for the same
                # reason highlight_threshold below is: ComfyUI passes required
                # inputs by keyword, so widget order is free, but a positional
                # caller — every existing test, and any script driving the node
                # directly — binds by position.
                reference_white_nits: float = 203.0,
                rudra_blend: float = 1.0,
                batch_mode: str = "Independent Images",
                shadow_threshold: float = 0.05,
                processing_mode: str = "Hybrid",
                highlight_threshold: float = 0.98,
                temporal_window: int = 5,
                temporal_checkpoint: str = "",
                learned_backend: str = "Auto",
                pixel_checkpoint: str = "",
                pixel_tile_size: int = 512,
                pixel_tile_overlap: int = 64,
                pixel_recovery_mode: str = "highlights",
                pixel_strength: float = 1.0,
                **_legacy):
        # `_legacy` swallows vae / rudra_size / model_meta from pre-3.5 graphs,
        # and "Legacy RUDRA" as a learned_backend value maps to Auto below.
        if _legacy.get("vae") is not None:
            logger.info("[SDR→HDR Universal] a VAE is connected but no longer used; "
                        "learned recovery runs on the pixel model.")
        # RESIDENT-COPY FIX: the `.clone()` here was a whole extra full-size
        # frame batch, and it bought nothing -- out-of-place nan_to_num already
        # returns a new tensor, so the caller's IMAGE is never written through.
        # On a 1080p frame that is 25 MB of the roughly 125 MB this node used to
        # keep resident across five simultaneous copies.
        img = torch.nan_to_num(image.float(), nan=0.0, posinf=1.0, neginf=0.0)
        if img.dim() == 3:                      # single HWC frame → batch of 1
            img = img.unsqueeze(0)
        if img.shape[0] == 0:                   # empty batch → empty result
            zeros = img[..., 0]
            return (img, zeros, zeros, zeros, zeros, "mode: empty batch\nframes: 0")

        rgb, extra = img[..., :3], img[..., 3:]
        peak_scale = max(peak_nits, 100.0) / 100.0

        # Where SDR diffuse white lands, as distinct from the display ceiling.
        #
        # The expansion used to target `peak_scale` at SDR code 1.0, so a white
        # shirt or a page of text came out at the full mastering peak: measured
        # 1000.00 nits at the shipped defaults, against the 203 nits ITU-R
        # BT.2408 defines as HDR Reference White. Nearly five times reference,
        # on the value the eye uses to judge exposure for the whole image.
        #
        # Below the knee the SDR signal keeps its SDR display level (1.0 ==
        # 100 nits); from the knee up it is expanded so SDR code 1.0 lands on
        # reference white. Unlike BT.2446 Method B this does not also lift the
        # midtones by ~2x: shadows and skin stay where the SDR grade put them
        # and only the top of the range opens up. `peak_nits` is the ceiling
        # and the encode target, `reference_white_nits` is where SDR white
        # sits, and the range between them is headroom for speculars the
        # learned path recovers.
        white_scale = max(float(reference_white_nits), 100.0) / 100.0
        white_scale = min(white_scale, peak_scale)

        # 1 ── decode to scene-linear
        lin = _inverse_oetf(rgb.clamp(0.0, 1.0), inverse_oetf)

        # 2 ── knee per frame (adaptive + temporally smoothed, or manual)
        luma = _luma(lin).clamp(0.0, 1.0)
        if knee_mode == "adaptive":
            smoothing = float(temporal_smoothing) if batch_mode == "Video Frames" else 0.0
            knees = _adaptive_knees(luma, float(knee), smoothing)
        else:
            knees = torch.full((img.shape[0],), float(knee),
                               dtype=lin.dtype, device=lin.device)

        # 3 ── hue-preserving highlight expansion, up to reference white
        luma_exp = _soft_knee_expand(luma, knees, white_scale, float(shoulder_gamma))
        gain = luma_exp / luma.clamp(min=_EPS)
        expanded_hdr = lin * gain.unsqueeze(-1)

        mask = ((luma_exp - luma) / max(white_scale - 1.0, _EPS)).clamp(0.0, 1.0)

        # RESIDENT-COPY FIX: `gain` and `luma_exp` are dead from here, and this
        # is where they have to be released rather than at the end of the
        # method: the measured peak of convert() is the line below, where
        # _clipped_highlight_mask's full-size clamp copy is allocated while
        # every intermediate above is still referenced. Releasing each
        # intermediate at its last use is what makes the returned batch the
        # only full-size tensor alive by the time the output is encoded.
        del gain, luma_exp

        shadows = _shadow_mask(luma, float(shadow_threshold))
        clipped = _clipped_highlight_mask(rgb.clamp(0.0, 1.0), highlight_threshold)

        # Expand is deterministic and never invokes a learned model. Recover
        # starts from decoded SDR and changes only clipped highlights/crushed
        # shadows. Hybrid starts from the deterministic expansion and blends
        # learned reconstruction only inside those evidence masks.
        mode = processing_mode if processing_mode in {"Expand", "Recover", "Hybrid"} else "Hybrid"
        hdr = lin if mode == "Recover" else expanded_hdr
        recovery_applied = False
        h_conf = torch.zeros_like(clipped)
        s_conf = torch.zeros_like(shadows)

        # What actually ran, reported rather than left to the console.
        #
        # Every learned backend below is wrapped in try/except and logs a
        # warning on the way past, so a user with no checkpoint installed gets
        # deterministic Expand output from Hybrid or Recover — bit-identical,
        # with five normal-looking outputs and nothing on the graph to say the
        # node's headline feature never engaged. Measured on a clean install:
        # Hybrid and Recover both return exactly the Expand result.
        #
        # `report` names the path that executed and, when the learned path did
        # not, why. It is a STRING output so it can be wired to a preview or
        # read at a glance, and it is appended last so existing links, which
        # ComfyUI stores by index, are unaffected.
        attempts: List[str] = []
        path = "deterministic expansion"

        # 3b ── temporal or direct-pixel reconstruction.
        wants_recovery = mode in {"Recover", "Hybrid"}
        if wants_recovery and rudra_blend > 0.0:
            backend = learned_backend if learned_backend in {
                "Auto", "Direct Pixel", "Temporal",
            } else "Auto"
            is_clip = img.shape[0] > 1 and batch_mode == "Video Frames"

            # A trained temporal model is the preferred video backend.
            if is_clip and backend in {"Auto", "Temporal"}:
                try:
                    hdr, h_conf, s_conf = self._temporal_reconstruct(
                        lin, expanded_hdr,
                        clipped * float(rudra_blend),
                        shadows * float(rudra_blend),
                        peak_scale, int(temporal_window), str(temporal_checkpoint),
                        mode == "Recover",
                    )
                    recovery_applied = True
                    path = "temporal RUDRA"
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Temporal RUDRA unavailable (%s).", exc)
                    attempts.append(f"temporal RUDRA unavailable ({exc})")
            elif backend == "Temporal":
                attempts.append(
                    "temporal backend needs batch_mode 'Video Frames' and at least five frames"
                )

            # The direct-pixel checkpoint supports stills and frame batches.
            # Frames are independent, so a clip may need downstream deflicker.
            # Ask the resolver, not the widget: an installed checkpoint in
            # models/radiance (or RADIANCE_SDR2HDR_PIXEL) is found with the
            # path left blank, which is the default.
            if not recovery_applied and backend in {"Auto", "Direct Pixel"}:
                if backend == "Direct Pixel" or _pixel_checkpoint_available(pixel_checkpoint):
                    try:
                        recovery_mask = torch.maximum(clipped, shadows)
                        if pixel_recovery_mode == "highlights":
                            recovery_mask = clipped
                        elif pixel_recovery_mode == "shadows":
                            recovery_mask = shadows
                        elif pixel_recovery_mode == "off":
                            recovery_mask = torch.zeros_like(clipped)
                        hdr = self._pixel_reconstruct(
                            lin, hdr, recovery_mask, str(pixel_checkpoint),
                            float(rudra_blend), peak_scale, int(pixel_tile_size),
                            int(pixel_tile_overlap), str(pixel_recovery_mode),
                            float(pixel_strength),
                            highlight_mask=(clipped if pixel_recovery_mode in {"highlights", "all"}
                                            else torch.zeros_like(clipped)),
                        )
                        h_conf = clipped * float(rudra_blend) if pixel_recovery_mode in {"highlights", "all"} else torch.zeros_like(clipped)
                        s_conf = shadows * float(rudra_blend) if pixel_recovery_mode in {"shadows", "all"} else torch.zeros_like(shadows)
                        recovery_applied = True
                        path = f"direct-pixel ({pixel_recovery_mode})"
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Direct-pixel recovery unavailable (%s).", exc)
                        attempts.append(f"direct-pixel unavailable ({exc})")
                else:
                    from radiance.pixel_sdr2hdr import describe_pixel_checkpoint_search
                    attempts.append(
                        "no pixel checkpoint installed; expected "
                        + describe_pixel_checkpoint_search()
                    )

        # Universal always produces usable HDR. Recover mode therefore falls
        # back to deterministic expansion if the learned path cannot run.
        if mode == "Recover" and not recovery_applied:
            hdr = expanded_hdr

        # RESIDENT-COPY FIX: from here only `hdr` and the extra channels are
        # needed. `hdr` aliases one of `lin` / `expanded_hdr` unless a learned
        # backend replaced it, so dropping these names frees whichever of the
        # two is not the one in use, and both once `hdr` itself is consumed.
        n_frames = int(img.shape[0])
        has_extra = extra.shape[-1] > 0
        del lin, expanded_hdr, luma, rgb
        if not has_extra:
            del extra, img

        # 4 ── output encoding (linear 1.0 == reference white, BT.2408)
        out = _encode_output(hdr, output_encoding, peak_scale, white_scale)
        del hdr

        if has_extra:                           # pass alpha / extra channels through
            out = torch.cat([out, extra], dim=-1)
            del extra, img

        report = self._build_report(
            mode=mode, path=path, recovery_applied=recovery_applied,
            attempts=attempts, frames=n_frames,
            reference_white_nits=float(reference_white_nits),
            peak_nits=float(peak_nits), output_encoding=str(output_encoding),
            rudra_blend=float(rudra_blend),
        )
        return (out, mask, shadows, h_conf, s_conf, report)

    @staticmethod
    def _build_report(*, mode, path, recovery_applied, attempts, frames,
                      reference_white_nits, peak_nits, output_encoding,
                      rudra_blend) -> str:
        """One human-readable line per fact about what this run actually did.

        Recover and Hybrid are the reason this node is called Universal, and on
        a machine with no checkpoint installed they are bit-identical to Expand
        — silently, because every backend is wrapped in try/except and only
        logs. This is the output that makes the difference visible on the graph
        instead of in a console the user is not reading.
        """
        lines = [
            f"mode: {mode}",
            f"path: {path}",
            f"frames: {frames}",
            f"reference white: {reference_white_nits:.0f} nits "
            f"(BT.2408 reference is 203)",
            f"mastering peak: {peak_nits:.0f} nits",
            f"output: {output_encoding} (linear 1.0 = {reference_white_nits:.0f} nits, "
            f"peak = {peak_nits / max(reference_white_nits, 1.0):.2f})",
        ]
        if output_encoding == "HLG" and peak_nits > 1000.0:
            lines.append("note: HLG is referenced to a 1000-nit display (BT.2100/BT.2408); "
                         "highlights above 1000 nits clip in the HLG signal")

        if mode == "Expand":
            lines.append("learned recovery: not requested")
            return "\n".join(lines)

        if recovery_applied:
            lines.append("learned recovery: applied")
            return "\n".join(lines)

        # Requested but did not run. Say why, and say what it cost.
        if rudra_blend <= 0.0:
            why = "rudra_blend is 0"
        elif attempts:
            why = "; ".join(attempts)
        else:
            why = ("no learned checkpoint found — install "
                   "sdr2hdr_pixel_image.pt in models/radiance or set "
                   "RADIANCE_SDR2HDR_PIXEL")

        lines.append(f"learned recovery: NOT APPLIED — {why}")
        lines.append(
            f"result: identical to Expand. {mode} did nothing this run."
        )
        logger.warning(
            "[SDR→HDR Universal] %s mode produced deterministic Expand output: %s",
            mode, why,
        )
        return "\n".join(lines)


NODE_CLASS_MAPPINGS = {
    "RadianceSDRToHDRRecover": RadianceSDRToHDRRecover,
    "RadianceSDRToHDRUniversal": RadianceSDRToHDRUniversal,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceSDRToHDRRecover": "◎ Radiance SDR → HDR Recover",
    "RadianceSDRToHDRUniversal": "◎ Radiance SDR → HDR Universal",
}
