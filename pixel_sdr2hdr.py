"""Direct 8-bit SDR to scene-linear HDR image and video models.

Unlike the latent decoders, these modules consume decoded SDR RGB pixels.  The
image network predicts a residual over a deterministic inverse-tone-map
baseline plus explicit highlight and shadow recovery masks.  The temporal
network is deliberately small and refines image-model predictions across a
short clip without requiring a multi-billion-parameter video backbone.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch

from radiance.model.cache import GPUModelCache
import torch.nn as nn
import torch.nn.functional as F


# Bounded LRU -- was an unbounded dict holding GPU-resident models.
_MODEL_CACHE = GPUModelCache(max_size=2)


def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055).pow(2.4))


def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * x.pow(1.0 / 2.4) - 0.055)


def canonicalize_sdr(sdr: torch.Tensor, transfer: str = "srgb", value_range: str = "full") -> torch.Tensor:
    """Convert common 8-bit SDR encodings to the model's canonical sRGB code."""
    x = sdr.float()
    if value_range == "limited":
        x = (x - 16.0 / 255.0) / (219.0 / 255.0)
    elif value_range != "full":
        raise ValueError("value_range must be 'full' or 'limited'")
    x = x.clamp(0.0, 1.0)
    if transfer == "srgb":
        return x
    if transfer == "rec709":
        linear = torch.where(x < 0.081, x / 4.5, ((x + 0.099) / 1.099).pow(1.0 / 0.45))
    elif transfer == "gamma22":
        linear = x.pow(2.2)
    elif transfer == "gamma24":
        linear = x.pow(2.4)
    else:
        raise ValueError("transfer must be srgb, rec709, gamma22, or gamma24")
    return linear_to_srgb(linear)


def inverse_aces_approx(display_linear: torch.Tensor) -> torch.Tensor:
    """Invert the ACES approximation used by ``prepare_training_data.py``.

    Values at exactly one are clipped and unknowable; clamping them just below
    one produces a finite baseline which the learned residual can extend.
    """
    y = display_linear.clamp(0.0, 0.995)
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    qa = y * c - a
    qb = y * d - b
    qc = y * e
    disc = (qb.square() - 4.0 * qa * qc).clamp_min(0.0)
    root_a = (-qb - torch.sqrt(disc)) / (2.0 * qa).clamp(max=-1e-7)
    root_b = (-qb + torch.sqrt(disc)) / (2.0 * qa).clamp(max=-1e-7)
    return torch.maximum(root_a, root_b).clamp_min(0.0)


# ── Model ────────────────────────────────────────────────────────────────────
#
# Ported verbatim from the RUDRA reference implementation (rudra/sdr2hdr.py,
# fxtdstudios/RUDRA) so every published checkpoint builds and runs exactly as it
# was trained: the v5 backbone, the shadow-gated shipped model (shadow_v1 and
# its seeds), the residual-scale gate and the tone-curve head. Radiance used to
# carry a pre-gate copy that could not load shadow_v1 at all. Keep this block in
# step with RUDRA; everything Radiance-specific lives below it.

# How the corpus renders its SDR side, and therefore what the analytic baseline
# has to undo.
#
# training/prepare_training_data.py applies an exposure offset BEFORE the ACES
# curve. The baseline is the inverse of that render, so it must apply the
# opposite offset after inverting the curve. Getting this wrong is a whole-stop
# error that looks like a grading choice rather than a bug, which is why the
# factor is derived from the offset here instead of being written as a literal.
#
# LEGACY_CORPUS_EV is -1 EV, which is what every checkpoint up to and including
# sdr2hdr_shadow_v1 was trained against. It has a cost that took a long time to
# see: halving the scene radiance before a curve that saturates near 1.0 means
# the SDR side almost never clips. Median clipped fraction on that corpus is
# 0.000%, and 52.6% of frames have no clipped pixel at all, so the hardest part
# of the task -- what is above a blown highlight -- barely occurs in training.
#
# CLIPPING_CORPUS_EV is 0 EV: the SDR side clips the way delivered SDR clips.
# Corpora rendered that way record it, checkpoints trained on them carry it in
# their config, and nothing has to be remembered.
LEGACY_CORPUS_EV = -1.0
CLIPPING_CORPUS_EV = 0.0


def sdr_to_baseline_hdr(sdr: torch.Tensor,
                        corpus_ev: float = LEGACY_CORPUS_EV) -> torch.Tensor:
    """Map sRGB SDR to the normalised scene-linear convention of the corpus.

    ``corpus_ev`` is the exposure offset the corpus applied before its tone
    curve. The default is the legacy -1 EV, so every existing checkpoint keeps
    the baseline it was trained against and a caller that does not know about
    this gets the old behaviour exactly.

    The scale is ``2**(-corpus_ev) * 203/10000``: the first term undoes the
    render's exposure, the second places diffuse white. At the legacy -1 EV that
    is ``2 * 203/10000``, which is the constant this replaces.
    """
    display_linear = srgb_to_linear(sdr)
    scale = (2.0 ** (-float(corpus_ev))) * (203.0 / 10000.0)
    return inverse_aces_approx(display_linear) * scale


def luminance(x: torch.Tensor) -> torch.Tensor:
    return 0.2126 * x[:, 0:1] + 0.7152 * x[:, 1:2] + 0.0722 * x[:, 2:3]


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        self.block = nn.Sequential(
            nn.GroupNorm(groups, channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


def frame_conditioning_stats(sdr: torch.Tensor, sdr_y: torch.Tensor) -> torch.Tensor:
    """Seven per-frame scalars the pooled conv features represent badly.

    Four describe headroom -- how much of the scene the SDR container could not
    hold -- and three describe the condition the input arrived in. The 29 Aug
    2026 measurement is why both halves are here: an oracle global residual
    scale is bimodal in headroom on CLEAN input (median 0.125 below 1 000 nits,
    0.969 above) and flat near full strength on DEGRADED input regardless of
    headroom, because degradation destroys information the analytic baseline
    cannot recover whatever the scene's range. A head that saw only headroom
    would switch itself off on exactly the degraded low-range frames that need
    it most.

    Every statistic is differentiable: the clipping fractions are sigmoids
    rather than thresholds, or no gradient would reach the head at all.
    """
    hi = torch.sigmoid((sdr_y - 0.98) * 200.0).mean(dim=(1, 2, 3))     # blown
    lo = torch.sigmoid((0.02 - sdr_y) * 200.0).mean(dim=(1, 2, 3))     # crushed
    mean = sdr_y.mean(dim=(1, 2, 3))
    std = sdr_y.flatten(1).std(dim=1)

    # Banding, JPEG blocking and a resampled tone curve all move luma's
    # high-frequency energy; 4:2:0 shows up in chroma's, not luma's.
    lap = (sdr_y[:, :, 1:-1, 1:-1] * 4.0
           - sdr_y[:, :, :-2, 1:-1] - sdr_y[:, :, 2:, 1:-1]
           - sdr_y[:, :, 1:-1, :-2] - sdr_y[:, :, 1:-1, 2:])
    hf = lap.abs().mean(dim=(1, 2, 3))
    chroma = sdr - sdr_y
    chf = (chroma[..., 1:] - chroma[..., :-1]).abs().mean(dim=(1, 2, 3))

    flat = sdr_y.flatten(1)
    if flat.shape[1] > 1_000_000:            # torch.quantile has a size ceiling
        flat = flat[:, :: flat.shape[1] // 1_000_000 + 1]
    span = (torch.quantile(flat, 0.99, dim=1) - torch.quantile(flat, 0.50, dim=1))
    return torch.stack([hi, lo, mean, std, hf, chf, span], dim=1)


class ConditionGate(nn.Module):
    """One residual scale per frame, predicted from headroom and degradation.

    The per-pixel priors below decide WHERE to reconstruct; nothing in the
    network decided HOW MUCH, for the frame as a whole. That is the defect the
    29 Aug 2026 benchmark found: on clean input v5 loses 3.0 dB to its own
    analytic baseline, concentrated entirely in low-dynamic-range frames (the
    60 worst average -10.8 dB at a median reference peak of 238 nits, the 60
    best +3.4 dB at 19 590 nits), because a 238-nit studio interior gets the
    same treatment as a 20 000-nit sunset.

    An oracle scale is worth +5.84 dB on clean and +0.29 dB on hard AT THE SAME
    TIME. No constant can do it: clean wants ~0.125 and hard wants ~1.1, and the
    constant that fixes clean throws away 85% of the hard gain. So the scale has
    to be predicted per frame, which is all this module does.

    A fresh head emits exactly 1.0, so enabling it changes nothing until it is
    trained -- the final layer is zeroed and its bias set so that
    sigmoid(bias) * alpha_max == 1.
    """

    def __init__(self, channels: int, hidden: int = 64, alpha_max: float = 1.25,
                 stats: int = 7):
        super().__init__()
        self.alpha_max = float(alpha_max)
        self.mlp = nn.Sequential(
            nn.Linear(channels * 2 + stats, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        last = self.mlp[-1]
        nn.init.zeros_(last.weight)
        nn.init.constant_(last.bias, math.log(1.0 / (self.alpha_max - 1.0)))

    def forward(self, features: torch.Tensor, sdr: torch.Tensor,
                sdr_y: torch.Tensor) -> torch.Tensor:
        # Average pooling alone would miss the thing that matters most: a
        # specular highlight is a max, not a mean, and it is a few pixels wide.
        pooled = torch.cat((features.mean(dim=(2, 3)),
                            features.amax(dim=(2, 3))), dim=1)
        stats = frame_conditioning_stats(sdr, sdr_y).to(pooled.dtype)
        alpha = torch.sigmoid(self.mlp(torch.cat((pooled, stats), dim=1)))
        return (alpha * self.alpha_max).view(-1, 1, 1, 1)


class ShadowGate(nn.Module):
    """One number per frame: how much of the shadow prior to let through.

    Measured on 429 held-out frames, 1 Sep 2026. Disabling the shadow arm
    entirely (`--recovery-mode highlights`) moves clean input by **+3.51 dB** --
    the regression against the analytic baseline does not shrink, it inverts to
    +0.51 dB -- and costs **1.10 dB** on degraded input. The shadow path is the
    whole of the clean regression and most of the degraded-input gain at the
    same time, because crushed shadows are what a bad tone curve produces and
    what a well-graded frame does not have.

    That makes this a far easier problem than the continuous residual scale of
    `ConditionGate`. It is BINARY, and its correct setting is exactly the
    clean-versus-degraded axis -- the one thing about a frame that is partially
    detectable (75.5% under frame-grouped cross-validation, against 3% of the
    variance explained for the continuous target). At that accuracy a switch is
    worth +2.65 dB on clean for 0.27 dB on hard.

    Output 1.0 reproduces `recovery_mode="all"` exactly and 0.0 reproduces
    `recovery_mode="highlights"` exactly, so the two ends of this dial are the
    two configurations that were measured.

    An untrained head emits **sigmoid(4) = 0.982**, not 1.0 -- so enabling this
    flag on a shipped checkpoint changes the composite by under 1% before any
    training. That is deliberate. A bias large enough to make the output exactly
    1.0 would put the sigmoid where its derivative is ~1e-6, and a head that
    cannot receive gradient at initialisation is the failure that cost two
    8,000-step runs on the residual-scale gate. 0.982 keeps sigmoid' at 0.018,
    which is trainable. tests/test_shadow_gate_2026_09_01.py pins both the
    weight and the 1% bound.
    """

    def __init__(self, channels: int, hidden: int = 64, stats: int = 7):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels * 2 + stats, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        last = self.mlp[-1]
        nn.init.zeros_(last.weight)
        nn.init.constant_(last.bias, 4.0)      # sigmoid(4) = 0.982, effectively "on"

    def forward(self, features: torch.Tensor, sdr: torch.Tensor,
                sdr_y: torch.Tensor) -> torch.Tensor:
        pooled = torch.cat((features.mean(dim=(2, 3)), features.amax(dim=(2, 3))), dim=1)
        stats = frame_conditioning_stats(sdr, sdr_y).to(pooled.dtype)
        return torch.sigmoid(self.mlp(torch.cat((pooled, stats), dim=1))).view(-1, 1, 1, 1)


@dataclass
class SDR2HDROutput:
    hdr: torch.Tensor
    baseline: torch.Tensor
    log_residual: torch.Tensor
    highlight_mask: torch.Tensor
    shadow_mask: torch.Tensor
    highlight_logits: torch.Tensor
    shadow_logits: torch.Tensor
    # (B,1,1,1) when the conditioning gate is enabled, else None. Worth logging:
    # it is the number the 29 Aug measurement says the model was missing.
    residual_scale: torch.Tensor | None = None
    # (B,1,1,1) when the shadow gate is enabled. 1.0 == recovery_mode "all",
    # 0.0 == recovery_mode "highlights"; the two measured endpoints.
    shadow_weight: torch.Tensor | None = None
    # The analytic inverse-ACES baseline, before any learned curve correction.
    # Equal to ``baseline`` unless the model has a CurveHead. Evaluation scores
    # "gain" against THIS, the only baseline that is genuinely "nothing learned".
    analytic_baseline: torch.Tensor | None = None
    # (B, 1 + knots) log2 corrections from the CurveHead, else None.
    curve_params: torch.Tensor | None = None


class CurveHead(nn.Module):
    """Estimates, per frame, how this SDR was tone-mapped, and undoes it.

    The analytic baseline inverts exactly one curve: Narkowicz ACES at the
    corpus exposure. Real SDR comes through camera LUTs, filmic curves, AgX,
    plain clipping, and an unknown exposure. On the 23 Sep 2026 out-of-generator
    bench (Hable + H.264) the shipped model, riding that one inverse, was worse
    than the inverse alone on 270 of 429 frames -- and outside the learned masks
    ``preserve_outside`` hands every pixel to that wrong inverse.

    This head looks at the whole frame once (a 128-px thumbnail: a small conv
    stack plus a soft luma histogram and clip statistics) and predicts a
    correction in log2 radiance: a global exposure plus a piecewise-linear
    function of the SDR code value, sampled at ``knots`` evenly spaced codes and
    applied per channel. Bounded (tanh) and zero-initialised, so a model with a
    fresh CurveHead reproduces the analytic baseline exactly and a checkpoint
    without one is untouched.
    """

    def __init__(self, knots: int = 8, hidden: int = 64, bins: int = 32,
                 max_exposure_stops: float = 3.0, max_knot_stops: float = 2.0):
        super().__init__()
        self.knots, self.bins = int(knots), int(bins)
        self.max_exposure = float(max_exposure_stops)
        self.max_knot = float(max_knot_stops)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.SiLU(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(32 + self.bins + 6, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1 + self.knots),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, sdr: torch.Tensor) -> torch.Tensor:
        longest = max(sdr.shape[-2:])
        view = sdr
        if longest > 128:
            s = 128.0 / longest
            view = F.interpolate(sdr, size=(max(1, round(sdr.shape[-2] * s)),
                                            max(1, round(sdr.shape[-1] * s))), mode="area")
        pooled = self.conv(view).mean(dim=(2, 3))
        y = luminance(view).flatten(1)
        centres = torch.linspace(0.0, 1.0, self.bins, device=sdr.device, dtype=sdr.dtype)
        width = 1.0 / (self.bins - 1)
        hist = torch.exp(-((y[:, :, None] - centres) / width) ** 2).mean(dim=1)
        hist = hist / hist.sum(dim=1, keepdim=True).clamp_min(1e-6)
        stats = torch.stack((
            y.mean(1), y.std(1),
            torch.sigmoid((y - 0.98) * 200.0).mean(1),
            torch.sigmoid((0.02 - y) * 200.0).mean(1),
            view.amax(dim=(1, 2, 3)), view.flatten(1).median(dim=1).values,
        ), dim=1)
        raw = self.mlp(torch.cat((pooled, hist, stats), dim=1))
        exposure = torch.tanh(raw[:, :1]) * self.max_exposure
        knots = torch.tanh(raw[:, 1:]) * self.max_knot
        return torch.cat((exposure, knots), dim=1)

    def correction_log2(self, sdr: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Per-pixel, per-channel log2 correction for ``sdr`` under ``params``."""
        k = self.knots
        pos = sdr.clamp(0.0, 1.0) * (k - 1)
        lo = pos.floor().clamp(max=k - 2).long()
        frac = pos - lo.to(pos.dtype)
        knots = params[:, 1:]
        b = sdr.shape[0]
        flat_lo = lo.reshape(b, -1)
        v0 = torch.gather(knots, 1, flat_lo).reshape_as(sdr)
        v1 = torch.gather(knots, 1, flat_lo + 1).reshape_as(sdr)
        return params[:, :1, None, None] + v0 + (v1 - v0) * frac


class SDR2HDRNet(nn.Module):
    """Compact U-Net for direct SDR pixel to HDR radiance recovery."""

    def __init__(self, base_channels: int = 32, log_scale: float = 16.0, max_hdr: float = 4.0,
                 corpus_ev: float = LEGACY_CORPUS_EV,
                 gate_conditioning: bool = False,
                 shadow_conditioning: bool = False,
                 curve_head: bool = False):
        super().__init__()
        c = base_channels
        # Off by default so every checkpoint written before 29 Aug 2026 still
        # loads with strict=True: when it is off no parameters are created and
        # the state dict is byte-identical to the old one.
        self.gate_conditioning = bool(gate_conditioning)
        self.shadow_conditioning = bool(shadow_conditioning)
        self.curve_head = bool(curve_head)
        self.log_scale = float(log_scale)
        self.max_hdr = float(max_hdr)
        # Which corpus convention this network was trained against. Carried on
        # the model so inference cannot use a baseline the weights never saw:
        # a checkpoint whose config omits it is legacy, which is correct for
        # every checkpoint that existed when this was added.
        self.corpus_ev = float(corpus_ev)
        self.stem = nn.Conv2d(6, c, 3, padding=1)
        self.enc1 = nn.Sequential(ResidualBlock(c), ResidualBlock(c))
        self.down1 = nn.Conv2d(c, c * 2, 3, stride=2, padding=1)
        self.enc2 = nn.Sequential(ResidualBlock(c * 2), ResidualBlock(c * 2))
        self.down2 = nn.Conv2d(c * 2, c * 4, 3, stride=2, padding=1)
        self.mid = nn.Sequential(ResidualBlock(c * 4), ResidualBlock(c * 4))
        self.up2 = nn.Conv2d(c * 6, c * 2, 3, padding=1)
        self.dec2 = nn.Sequential(ResidualBlock(c * 2), ResidualBlock(c * 2))
        self.up1 = nn.Conv2d(c * 3, c, 3, padding=1)
        self.dec1 = nn.Sequential(ResidualBlock(c), ResidualBlock(c))
        self.head = nn.Conv2d(c, 5, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.gate = ConditionGate(c * 4) if self.gate_conditioning else None
        self.shadow_gate = ShadowGate(c * 4) if self.shadow_conditioning else None
        # Off by default for the same reason as the gates: no parameters, and
        # every earlier checkpoint loads strict and computes exactly what it did.
        self.curve = CurveHead() if self.curve_head else None

    @classmethod
    def from_config(cls, config: dict | None, **overrides) -> "SDR2HDRNet":
        """Build the architecture a checkpoint's own config describes.

        Five call sites used to spell `SDR2HDRNet(base_channels=...)` by hand,
        so any new architectural flag had to be remembered five times or a
        strict load would fail somewhere far from the change. It is one place
        now.
        """
        config = config or {}
        kwargs = {
            "base_channels": int(config.get("base_channels", 32)),
            "gate_conditioning": bool(config.get("gate_conditioning", False)),
            "shadow_conditioning": bool(config.get("shadow_conditioning", False)),
            "curve_head": bool(config.get("curve_head", False)),
        }
        for key in ("log_scale", "max_hdr", "corpus_ev"):
            if config.get(key) is not None:
                kwargs[key] = float(config[key])
        kwargs.update(overrides)
        return cls(**kwargs)

    def baseline_hdr(self, sdr: torch.Tensor, params: torch.Tensor | None = None
                     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """(baseline, analytic_baseline, curve_params) for this SDR.

        Without a CurveHead the two baselines are the same tensor. With one,
        the analytic inverse is scaled per pixel by 2**correction, where the
        correction is estimated from the whole frame (or taken from ``params``,
        which is how tiled inference keeps one curve across every tile).
        """
        analytic = sdr_to_baseline_hdr(sdr, self.corpus_ev)
        if self.curve is None:
            return analytic, analytic, None
        if params is None:
            params = self.curve(sdr)
        corrected = analytic * torch.exp2(self.curve.correction_log2(sdr, params))
        return corrected, analytic, params

    @torch.no_grad()
    def predict_curve(self, sdr: torch.Tensor) -> torch.Tensor | None:
        """The frame's curve correction, once, for tiled inference and the
        viewer. None without a CurveHead."""
        if self.curve is None:
            return None
        return self.curve(sdr.float().clamp(0.0, 1.0))

    def encode(self, sdr: torch.Tensor, baseline: torch.Tensor):
        x = torch.cat((sdr, baseline), dim=1)
        e1 = self.enc1(self.stem(x))
        e2 = self.enc2(self.down1(e1))
        return e1, e2, self.mid(self.down2(e2))

    @torch.no_grad()
    def predict_shadow_weight(self, sdr: torch.Tensor, max_side: int = 512) -> torch.Tensor | None:
        """The per-frame shadow weight, from the whole frame, computed once.

        Same contract as predict_residual_scale: pooled features from a
        downscaled view because the judgement is global, statistics from the
        NATIVE frame because the evidence that an input arrived degraded lives
        at the pixel scale and an area resize is a low-pass filter over it.
        """
        if self.shadow_gate is None:
            return None
        sdr = sdr.float().clamp(0.0, 1.0)
        view = sdr
        longest = max(sdr.shape[-2:])
        if longest > max_side:
            scale = max_side / longest
            view = F.interpolate(sdr, size=(max(1, round(sdr.shape[-2] * scale)),
                                            max(1, round(sdr.shape[-1] * scale))), mode="area")
        _, _, m = self.encode(view, self.baseline_hdr(view)[0])
        return self.shadow_gate(m, sdr, luminance(sdr))

    @torch.no_grad()
    def predict_residual_scale(self, sdr: torch.Tensor, max_side: int = 512) -> torch.Tensor | None:
        """The per-frame scale, computed once, from the WHOLE frame.

        Tiled inference would otherwise hand each tile its own scale, computed
        from that tile's statistics -- and the scale is a property of the
        frame, not of a 512-pixel window. A tile of sky inside a dim interior
        would read as a high-headroom frame and reconstruct hard, which is the
        exact failure the head exists to prevent, reintroduced one tile at a
        time. Callers that tile must compute this once and pass it to every
        tile as ``residual_scale``.

        Downscaling first is safe and deliberate: every statistic the head
        reads is a pooled or fractional quantity, and it makes the extra pass
        cost a rounding error against the tiles themselves.
        """
        if self.gate is None:
            return None
        sdr = sdr.float().clamp(0.0, 1.0)
        view = sdr
        longest = max(sdr.shape[-2:])
        if longest > max_side:
            scale = max_side / longest
            size = (max(1, round(sdr.shape[-2] * scale)),
                    max(1, round(sdr.shape[-1] * scale)))
            view = F.interpolate(sdr, size=size, mode="area")
        _, _, m = self.encode(view, self.baseline_hdr(view)[0])
        # Pooled features come from the downscaled view -- alpha is a whole-frame
        # judgement and that is what makes the encode affordable. The statistics
        # come from the NATIVE frame, because the evidence that an input arrived
        # degraded lives at the pixel scale and an area resize is a low-pass
        # filter over it. Measured on 27 held-out frames on 29 Aug 2026: chroma
        # high-frequency energy separates clean from degraded at 1.13 sd at
        # native resolution and 0.60 sd -- with the sign inverted -- in the
        # 512-pixel view. Same reason luma high-frequency energy falls from
        # 0.19 sd to 0.01.
        return self.gate(m, sdr, luminance(sdr))

    def forward(
        self,
        sdr: torch.Tensor,
        preserve_outside: bool = False,
        recovery_mode: str = "all",
        residual_strength: float | torch.Tensor = 1.0,
        residual_scale: float | torch.Tensor | None = None,
        shadow_weight: float | torch.Tensor | None = None,
        curve_params: torch.Tensor | None = None,
    ) -> SDR2HDROutput:
        if sdr.ndim != 4 or sdr.shape[1] != 3:
            raise ValueError(f"Expected SDR tensor (B,3,H,W), got {tuple(sdr.shape)}")
        sdr = sdr.float().clamp(0.0, 1.0)
        baseline, analytic, curve_params = self.baseline_hdr(sdr, curve_params)
        e1, e2, m = self.encode(sdr, baseline)
        u2 = F.interpolate(m, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        u2 = self.dec2(self.up2(torch.cat((u2, e2), dim=1)))
        u1 = F.interpolate(u2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        features = self.dec1(self.up1(torch.cat((u1, e1), dim=1)))
        raw = self.head(features)

        residual = raw[:, :3]
        highlight_logits = raw[:, 3:4]
        shadow_logits = raw[:, 4:5]
        highlight = torch.sigmoid(highlight_logits)
        shadow = torch.sigmoid(shadow_logits)
        base_log = torch.log1p(baseline * self.log_scale)
        sdr_y = luminance(sdr)
        highlight_prior = torch.sigmoid((sdr_y - 0.82) * 24.0)
        shadow_prior = torch.sigmoid((0.10 - sdr_y) * 24.0)
        # Preserve the physically strong inverse-tone-map baseline through
        # ordinary midtones. Most learned capacity is applied where clipping or
        # crushed shadows made the SDR mapping non-invertible.
        if self.shadow_gate is not None and shadow_weight is None:
            shadow_weight = self.shadow_gate(m, sdr, sdr_y)
        if shadow_weight is not None and not isinstance(shadow_weight, torch.Tensor):
            shadow_weight = torch.full((sdr.shape[0], 1, 1, 1), float(shadow_weight),
                                       device=sdr.device, dtype=sdr.dtype)
        if shadow_weight is not None:
            # Only the residual GATE is weighted, never the learned masks that
            # drive preserve_outside -- because that is exactly what
            # `--recovery-mode highlights` did, and the two ends of this dial
            # have to be the two configurations that were measured.
            shadow_prior = shadow_prior * shadow_weight.to(sdr.device, sdr.dtype)
        if recovery_mode == "all":
            residual_gate = torch.maximum(highlight_prior, shadow_prior)
        elif recovery_mode == "highlights":
            residual_gate = highlight_prior
        elif recovery_mode == "shadows":
            residual_gate = shadow_prior
        elif recovery_mode == "off":
            residual_gate = torch.zeros_like(highlight_prior)
        else:
            raise ValueError("recovery_mode must be all, highlights, shadows, or off")
        if isinstance(residual_strength, torch.Tensor):
            # Per-pixel ITM strength (mask-driven local inverse tone mapping):
            # (H,W), (1,1,H,W) or (B,1,H,W) in [0, 2]. Built from artist masks
            # via rudra.delivery.controls.itm_strength_map.
            strength = residual_strength.to(device=sdr.device, dtype=residual_gate.dtype)
            while strength.ndim < 4:
                strength = strength.unsqueeze(0)
            if strength.shape[-2:] != residual_gate.shape[-2:]:
                strength = F.interpolate(strength, size=residual_gate.shape[-2:],
                                         mode="bilinear", align_corners=False)
            residual_gate = residual_gate * strength.clamp(0.0, 2.0)
        else:
            residual_gate = residual_gate * float(residual_strength)
        # The learned per-frame scale multiplies whatever the caller asked for,
        # so the Studio's strength slider and the artist mask stay a control ON
        # TOP of the model's own judgement rather than a replacement for it.
        if self.gate is not None:
            if residual_scale is None:
                residual_scale = self.gate(m, sdr, sdr_y)
            elif not isinstance(residual_scale, torch.Tensor):
                residual_scale = torch.full((sdr.shape[0], 1, 1, 1), float(residual_scale),
                                            device=sdr.device, dtype=sdr.dtype)
            residual_gate = residual_gate * residual_scale.to(sdr.device, sdr.dtype)
        else:
            residual_scale = None
        pred_log = (base_log + residual * residual_gate).clamp(0.0, torch.log1p(torch.tensor(
            self.max_hdr * self.log_scale, device=sdr.device, dtype=sdr.dtype
        )))
        pred = torch.expm1(pred_log) / self.log_scale
        if preserve_outside:
            recovery = torch.maximum(highlight, shadow)
            pred = baseline + recovery * (pred - baseline)
        return SDR2HDROutput(
            pred, baseline, residual, highlight, shadow,
            highlight_logits, shadow_logits, residual_scale, shadow_weight,
            analytic, curve_params,
        )


class TemporalHDRRefiner(nn.Module):
    """Small residual 3D CNN for short-clip temporal consistency.

    Input layouts are ``(B,T,3,H,W)``.  The frozen or jointly trained image
    model supplies the initial HDR frames; this module only learns a bounded
    log-radiance correction using neighboring frames.
    """

    def __init__(self, channels: int = 24, log_scale: float = 16.0):
        super().__init__()
        self.log_scale = float(log_scale)
        self.net = nn.Sequential(
            nn.Conv3d(6, channels, (3, 3, 3), padding=1), nn.SiLU(),
            nn.Conv3d(channels, channels, (3, 3, 3), padding=1), nn.SiLU(),
            nn.Conv3d(channels, channels, (3, 3, 3), padding=1), nn.SiLU(),
            nn.Conv3d(channels, 3, (3, 3, 3), padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, sdr: torch.Tensor, image_hdr: torch.Tensor) -> torch.Tensor:
        if sdr.shape != image_hdr.shape or sdr.ndim != 5 or sdr.shape[2] != 3:
            raise ValueError("Expected matching SDR/HDR tensors with shape (B,T,3,H,W)")
        x = torch.cat((sdr, image_hdr), dim=2).permute(0, 2, 1, 3, 4)
        correction = self.net(x).permute(0, 2, 1, 3, 4)
        base_log = torch.log1p(image_hdr.clamp_min(0.0) * self.log_scale)
        return torch.expm1((base_log + 0.25 * torch.tanh(correction)).clamp_min(0.0)) / self.log_scale


def recovery_masks(sdr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Soft luminance masks used as supervision for recovery regions."""
    y = luminance(sdr)
    highlight = torch.sigmoid((y - 0.82) * 24.0)
    shadow = torch.sigmoid((0.10 - y) * 24.0)
    return highlight, shadow


def _gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    px, tx = pred[..., :, 1:] - pred[..., :, :-1], target[..., :, 1:] - target[..., :, :-1]
    py, ty = pred[..., 1:, :] - pred[..., :-1, :], target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(px, tx) + F.l1_loss(py, ty)


def sdr2hdr_loss(output: SDR2HDROutput, sdr: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    """Stable HDR recovery objective in log-radiance and masked regions."""
    scale = 16.0
    pred_log = torch.log1p(output.hdr.clamp_min(0.0) * scale)
    target_log = torch.log1p(target.clamp_min(0.0) * scale)
    hi, sh = recovery_masks(sdr)
    base = F.l1_loss(pred_log, target_log)
    highlight = ((pred_log - target_log).abs() * hi).sum() / (hi.sum() * 3.0 + 1e-6)
    shadow = ((pred_log - target_log).abs() * sh).sum() / (sh.sum() * 3.0 + 1e-6)
    mask = F.binary_cross_entropy_with_logits(output.highlight_logits, hi) + F.binary_cross_entropy_with_logits(output.shadow_logits, sh)
    edge = _gradient_loss(pred_log, target_log)
    pred_chroma = output.hdr / (output.hdr.sum(1, keepdim=True) + 1e-4)
    target_chroma = target / (target.sum(1, keepdim=True) + 1e-4)
    chroma = F.l1_loss(pred_chroma, target_chroma)
    recovery = torch.maximum(hi, sh)
    outside = ((pred_log - target_log).abs() * (1.0 - recovery)).mean()
    residual_outside = (output.log_residual.abs() * (1.0 - recovery)).mean()
    total = (base + 0.50 * highlight + 0.20 * shadow + 0.10 * chroma +
             0.10 * edge + 0.05 * mask + 0.05 * outside + 0.10 * residual_outside)
    return {
        "total": total, "log_l1": base, "highlight": highlight,
        "shadow": shadow, "chroma": chroma, "edge": edge,
        "mask": mask, "outside": outside, "residual_outside": residual_outside,
    }


def temporal_consistency_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Match adjacent-frame log-radiance changes without assuming optical flow."""
    p = torch.log1p(pred.clamp_min(0.0) * 16.0)
    t = torch.log1p(target.clamp_min(0.0) * 16.0)
    return F.l1_loss(p[:, 1:] - p[:, :-1], t[:, 1:] - t[:, :-1])


PIXEL_CHECKPOINT_ENV = "RADIANCE_SDR2HDR_PIXEL"
#: Search order. RUDRA's shipped model first (the v5 backbone plus the trained
#: shadow gate), then its other seeds and the ungated backbone as Hugging Face
#: publishes them, flat in models/radiance or under the ``sdr2hdr/`` folder a
#: ``hf download --local-dir models/radiance`` creates. The legacy ``.pt``
#: names the training scripts emit come last. ``sdr2hdr_temporal_v1`` is not
#: an SDR2HDRNet and is deliberately not matched.
PIXEL_CHECKPOINT_PATTERNS = (
    "sdr2hdr_shadow_v1.safetensors",
    "sdr2hdr/sdr2hdr_shadow_v1.safetensors",
    "sdr2hdr_shadow_*.safetensors",
    "sdr2hdr/sdr2hdr_shadow_*.safetensors",
    "sdr2hdr_image_v5.safetensors",
    "sdr2hdr/sdr2hdr_image_v5.safetensors",
    "sdr2hdr_pixel_image.pt",
    "sdr2hdr_pixel_image_*.pt",
    "sdr2hdr_image_*.pt",
    "sdr2hdr_pixel*.pt",
    "sdr2hdr_image_*.safetensors",
    "sdr2hdr/sdr2hdr_image_*.safetensors",
)


def resolve_pixel_checkpoint(checkpoint_path: str = "") -> Path | None:
    """Resolve an explicit or installed direct-pixel SDR-to-HDR checkpoint.

    Order: the explicit path, ``RADIANCE_SDR2HDR_PIXEL``, then every
    ``models/radiance`` folder ComfyUI knows about (see
    :mod:`radiance.model.paths`).

    When none of those finds a file, the default checkpoint is downloaded
    into ``models/radiance`` on first use (see
    :mod:`radiance.model.pixel_download`; off with ``RADIANCE_ALLOW_DOWNLOADS=0``).
    A missing explicit path already falls back to the default search with a
    warning, so it falls back to the default download the same way.
    """
    from radiance.model.paths import find_radiance_checkpoint
    found = find_radiance_checkpoint(
        PIXEL_CHECKPOINT_PATTERNS, explicit_path=checkpoint_path or "",
        env_var=PIXEL_CHECKPOINT_ENV,
    )
    if found is not None:
        return found
    from radiance.model.pixel_download import download_pixel_checkpoint
    return download_pixel_checkpoint()


def describe_pixel_checkpoint_search() -> str:
    from radiance.model.paths import describe_search
    return describe_search(PIXEL_CHECKPOINT_PATTERNS[:1], PIXEL_CHECKPOINT_ENV)


def read_pixel_checkpoint(path: str | Path) -> tuple[dict, dict]:
    """``(state_dict, config)`` from a RUDRA checkpoint, ``.safetensors`` or ``.pt``.

    Hugging Face publishes ``.safetensors`` with the architecture config in the
    file's metadata; the training scripts write ``.pt`` dicts with ``model`` and
    ``config`` keys. Both are read without executing code: safetensors has no
    code path, and ``.pt`` is loaded with ``weights_only=True`` because the path
    comes from the user-editable ``pixel_checkpoint`` widget, so a workflow can
    aim it at any file on disk.
    """
    path = Path(path)
    if path.suffix.lower() == ".safetensors":
        from safetensors import safe_open
        from safetensors.torch import load_file
        with safe_open(str(path), framework="pt") as handle:
            metadata = handle.metadata() or {}
        try:
            config = json.loads(metadata.get("config") or "{}")
        except ValueError:
            config = {}
        return load_file(str(path)), (config if isinstance(config, dict) else {})
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), dict):
        config = checkpoint.get("config") or {}
        return checkpoint["model"], (config if isinstance(config, dict) else {})
    return checkpoint, {}


def build_pixel_model(state: dict, config: dict) -> SDR2HDRNet:
    """Build the network the checkpoint describes and load it strictly.

    The config says which optional heads exist. Older configs predate those
    flags, so the state dict is the tiebreaker: a ``shadow_gate.*`` key means
    the gate is there whatever the config forgot to say.
    """
    if any(k.startswith("net.") for k in state) and not any(k.startswith("stem.") for k in state):
        raise ValueError(
            "this is the RUDRA temporal refiner (sdr2hdr_temporal_v1), not an "
            "SDR2HDRNet image model; use sdr2hdr_shadow_v1.safetensors")
    config = dict(config or {})
    if "stem.weight" in state and "base_channels" not in config:
        config["base_channels"] = int(state["stem.weight"].shape[0])
    for flag, prefix in (("shadow_conditioning", "shadow_gate."),
                         ("gate_conditioning", "gate."),
                         ("curve_head", "curve.")):
        if any(k.startswith(prefix) for k in state):
            config[flag] = True
    model = SDR2HDRNet.from_config(config)
    model.load_state_dict(state, strict=True)
    return model.eval()


def load_pixel_sdr2hdr_weights(checkpoint_path: str, device: torch.device) -> SDR2HDRNet | None:
    """Load and cache a direct-pixel checkpoint on the requested device."""
    resolved = resolve_pixel_checkpoint(checkpoint_path)
    if resolved is None:
        return None
    key = (str(resolved), str(device))
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached
    state, config = read_pixel_checkpoint(resolved)
    model = build_pixel_model(state, config)
    model.to(device).eval()
    _MODEL_CACHE.put(key, model)
    return model


def _tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    starts = list(range(0, length - tile_size + 1, stride))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


def _tile_weight(height: int, width: int, overlap: int, y: int, x: int,
                 full_h: int, full_w: int, device: torch.device) -> torch.Tensor:
    wy = torch.ones(height, device=device, dtype=torch.float32)
    wx = torch.ones(width, device=device, dtype=torch.float32)
    fy, fx = min(overlap, height // 2), min(overlap, width // 2)
    if y > 0 and fy:
        wy[:fy] = torch.linspace(1e-3, 1.0, fy, device=device)
    if y + height < full_h and fy:
        wy[-fy:] = torch.linspace(1.0, 1e-3, fy, device=device)
    if x > 0 and fx:
        wx[:fx] = torch.linspace(1e-3, 1.0, fx, device=device)
    if x + width < full_w and fx:
        wx[-fx:] = torch.linspace(1.0, 1e-3, fx, device=device)
    return (wy[:, None] * wx[None, :])[None, None]


# Whole-frame inference is preferred over tiles. Every ResidualBlock opens
# with GroupNorm, whose statistics are taken over the whole input, so a tile is
# normalised by its own contents: measured on a 1920x1088 sunset, 512-px tiles
# moved the recovered highlights by 1.4 stops at the 99th percentile against
# the same frame run whole. RUDRA's own viewer runs untiled for the same
# reason. Tiles remain the fallback when the frame does not fit.
#
# Cost, measured: ~1.5 KB of activations per pixel in fp32 (3.0 GB at
# 1920x1088), about half that under CUDA bf16 autocast.
_WHOLE_FRAME_BYTES_PER_PIXEL_CUDA = 900
_WHOLE_FRAME_MAX_PIXELS_CPU = 2_600_000


def _whole_frame_fits(frame: torch.Tensor) -> bool:
    pixels = int(frame.shape[-2]) * int(frame.shape[-1])
    if frame.is_cuda:
        try:
            free, _ = torch.cuda.mem_get_info(frame.device)
        except Exception:  # noqa: BLE001 - no query, be conservative
            return False
        return pixels * _WHOLE_FRAME_BYTES_PER_PIXEL_CUDA < 0.8 * free
    return pixels <= _WHOLE_FRAME_MAX_PIXELS_CPU


@torch.inference_mode()
def _frame_parameters(model: SDR2HDRNet, frame: torch.Tensor) -> dict:
    """The per-frame decisions, made once from the whole frame.

    The residual-scale gate, the shadow gate and the curve head each judge the
    frame as a whole. Left to run per tile, a tile of sky inside a dim interior
    reads as a high-headroom frame and each tile inverts its own curve, so the
    seams show. RUDRA's own inference computes them once and hands them to every
    tile; so does this.
    """
    return {
        "residual_scale": model.predict_residual_scale(frame),
        "shadow_weight": model.predict_shadow_weight(frame),
        "curve_params": model.predict_curve(frame),
    }


@torch.inference_mode()
def _predict_frame(model: SDR2HDRNet, frame: torch.Tensor, tile_size: int,
                   overlap: int, recovery_mode: str, strength: float) -> torch.Tensor:
    _, _, height, width = frame.shape
    params = _frame_parameters(model, frame)
    whole = tile_size <= 0 or (height <= tile_size and width <= tile_size) \
        or _whole_frame_fits(frame)
    if whole:
        amp = torch.autocast("cuda", dtype=torch.bfloat16) if frame.is_cuda else contextlib.nullcontext()
        try:
            with amp:
                return model(frame, recovery_mode=recovery_mode,
                             residual_strength=strength, **params).hdr.float()
        except torch.cuda.OutOfMemoryError:
            if tile_size <= 0:
                raise
            torch.cuda.empty_cache()     # fall through to tiles
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("pixel tile overlap must be >= 0 and smaller than tile size")
    result = torch.zeros((1, 3, height, width), device=frame.device, dtype=torch.float32)
    weights = torch.zeros((1, 1, height, width), device=frame.device, dtype=torch.float32)
    for y in _tile_starts(height, tile_size, overlap):
        for x in _tile_starts(width, tile_size, overlap):
            tile = frame[..., y:min(y + tile_size, height), x:min(x + tile_size, width)]
            amp = torch.autocast("cuda", dtype=torch.bfloat16) if tile.is_cuda else contextlib.nullcontext()
            with amp:
                prediction = model(tile, recovery_mode=recovery_mode,
                                   residual_strength=strength, **params).hdr.float()
            weight = _tile_weight(tile.shape[-2], tile.shape[-1], overlap, y, x,
                                  height, width, frame.device)
            result[..., y:y + tile.shape[-2], x:x + tile.shape[-1]] += prediction * weight
            weights[..., y:y + tile.shape[-2], x:x + tile.shape[-1]] += weight
    return result / weights.clamp_min(1e-6)


@torch.inference_mode()
def predict_pixel_sdr2hdr(sdr_bhwc: torch.Tensor, checkpoint_path: str = "",
                          tile_size: int = 512, tile_overlap: int = 64,
                          recovery_mode: str = "highlights",
                          strength: float = 1.0) -> torch.Tensor:
    """Run the installed image model on an IMAGE batch.

    The returned BHWC tensor follows the model contract: scene-linear RGB,
    where 1.0 equals 10,000 nits, in the SAME primaries as the input. The
    network is a per-channel mapping trained on SDR rendered channel-wise from
    its HDR target, so it does not change gamut; RUDRA's own inference writes
    its output untouched. Frames are processed independently until a
    compatible temporal direct-pixel model is trained.
    """
    if sdr_bhwc.ndim != 4 or sdr_bhwc.shape[-1] < 3:
        raise ValueError(f"Expected IMAGE [B,H,W,C], got {tuple(sdr_bhwc.shape)}")
    model = load_pixel_sdr2hdr_weights(checkpoint_path, sdr_bhwc.device)
    if model is None:
        raise RuntimeError(
            "no direct-pixel checkpoint; set pixel_checkpoint, or install "
            + describe_pixel_checkpoint_search()
        )
    # STREAM-FIX: one frame in flight, written straight into the returned
    # batch. This used to list-comprehend every predicted frame onto the model
    # device and then torch.stack them, so a clip cost the whole prediction
    # twice, on top of a full-clip `frames` copy that only ever needed one
    # frame at a time. Peak above the returned batch is now one frame.
    batch, height, width = sdr_bhwc.shape[0], sdr_bhwc.shape[1], sdr_bhwc.shape[2]
    out = torch.empty((batch, height, width, 3),
                      dtype=torch.float32, device=sdr_bhwc.device)
    for index in range(batch):
        frame = (sdr_bhwc[index:index + 1, ..., :3]
                 .float().clamp(0.0, 1.0).permute(0, 3, 1, 2))
        prediction = _predict_frame(model, frame, int(tile_size), int(tile_overlap),
                                    recovery_mode, float(strength))
        out[index] = prediction[0].permute(1, 2, 0).to(out.device)
        del frame, prediction
    return out
