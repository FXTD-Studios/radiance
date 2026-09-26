"""
nodes_hdr_inception.py — Radiance HDR Generation Seeds

CHANGES vs original:
  RadianceHDRPromptMultiplier — REMOVED.
  ════════════════════════════
  This node multiplied the conditioning tensor (text embeddings) by up to
  10,000×. Text embeddings are normalised vectors in semantic space. Scaling
  them by large factors pushes the conditioning out of the distribution the
  model was trained on, degrading image quality. It does NOT make the model
  generate brighter or more HDR content — the conditioning magnitude is
  normalised internally by attention softmax. The node was scientifically
  incorrect and should not be exposed to users.

  RadianceHDRLatentInit — REMOVED.
  ══════════════════════════════
  Previous version created a tensor of all zeros, which produces degenerate
  results. Superseded by native ComfyUI latent generation with proper noise
  initialisation.

  RadianceHDRLatentBlend — REMOVED.
  ══════════════════════════════
  Blend two latent tensors with HDR energy weighting. Removed alongside
  RadianceHDRLatentInit; use RadianceHDRBlendValidator for quality checks.
"""

import torch
import logging

logger = logging.getLogger("radiance.hdr_inception")


# ==============================================================================
# RadianceHDRBlendValidator
# ==============================================================================


class RadianceHDRBlendValidator:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    DESCRIPTION = "Validate that blended HDR layers maintain energy and gamut constraints."
    """
    ◎ Radiance HDR Blend Validator

A/B quality metric node for validating whether HDR processing
actually improves output quality over a baseline.

    Computes three objective metrics between the baseline (no blend) and the
    HDR-blended image:

      1. SSIM (Structural Similarity)
         Measures structural fidelity.  Closer to 1.0 = more similar.
         Values < 0.95 indicate significant structural change.

      2. Luminance histogram divergence (Jensen-Shannon)
         Measures how differently the two images distribute brightness.
         Higher JS divergence = the blend shifted the tonal distribution more.
         Expected: HDR blend should push JS divergence > 0.01 (measurable shift).

      3. Dynamic range delta (stops)
         DR_b − DR_a where DR = log2(p99 / (p1 + ε)).
         Positive = HDR blend expanded dynamic range.  Negative = compressed it.

    The node outputs:
      • winner          — passes through whichever image has better HDR properties
                           (higher dynamic range by default; override with metric)
      • report_json     — full metric breakdown as JSON string for logging
      • ssim            — structural similarity score (FLOAT)
      • js_divergence   — tonal distribution change (FLOAT)
      • dr_delta_stops  — dynamic range change in stops (FLOAT)

    Usage:
        Without blend → image_a (baseline)
        With    blend → image_b (HDR blend)
        Both → RadianceHDRBlendValidator → winner (the better one) + metrics
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a":     ("IMAGE", {"tooltip": "Baseline (no HDR blend)"}),
                "image_b":     ("IMAGE", {"tooltip": "HDR-blended output"}),
                "win_metric":  (
                    ["dynamic_range", "ssim", "js_divergence"],
                    {"default": "dynamic_range",
                     "tooltip": "Which metric decides the winner."},
                ),
            }
        }

    RETURN_TYPES  = ("IMAGE", "STRING", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES  = ("winner", "report_json", "ssim", "js_divergence", "dr_delta_stops")
    FUNCTION      = "validate"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"

    # ── Metric helpers (static, torch-only) ──────────────────────────────────

    #: SSIM window, in pixels. 11x11 is the size Wang et al. (2004) specify.
    _SSIM_WINDOW = 11

    #: Below this the blend has changed structure, not just range, so the
    #: baseline is the safer pick. See validate()'s win_metric == "ssim".
    _SSIM_SAFE = 0.95

    @staticmethod
    def _ssim(a: torch.Tensor, b: torch.Tensor) -> float:
        """Windowed single-scale SSIM over the luminance channel.

        SSIM-HDR FIX. Two defects, both of which made this function return a
        number that says nothing about the HDR content the node exists to
        validate:

          1. It clamped luma to [0,1] first. Two HDR images whose luma exceeds
             1.0 over most of the frame both clamp to a flat field of 1.0, so
             mu = 1, sigma = 0, sigma_ab = 0, and the expression collapses to
             (2+C1)(0+C2) / ((2+C1)(0+C2)) == EXACTLY 1.0, for any two images
             whatever. The docstring's "values < 0.95 indicate significant
             structural change" was unreachable. The luma is now scaled by the
             larger of the two images' peaks, which puts both on [0,1] with
             their structure intact and keeps C1/C2 meaningful (they are
             defined for a dynamic range of 1.0).
          2. It computed ONE global mean and variance over the whole frame.
             That is not SSIM, it is a global correlation coefficient: it is
             blind to local structure, which is the entire point of the metric.
             Statistics are now computed over an 11x11 sliding window and the
             SSIM map averaged, as Wang et al. specify.
        """
        w = a.new_tensor([0.2126, 0.7152, 0.0722])
        la = (a[..., :3].float() * w).sum(-1).clamp(min=0.0)
        lb = (b[..., :3].float() * w).sum(-1).clamp(min=0.0)

        # Shared scale: normalises range without flattening HDR structure.
        peak = max(float(la.max()), float(lb.max()), 1e-6)
        la, lb = la / peak, lb / peak

        if la.dim() == 2:                      # (H, W) -> (1, H, W)
            la, lb = la.unsqueeze(0), lb.unsqueeze(0)
        la = la.unsqueeze(1)                   # (B, 1, H, W)
        lb = lb.unsqueeze(1)

        C1, C2 = 0.01 ** 2, 0.03 ** 2
        win = min(RadianceHDRBlendValidator._SSIM_WINDOW,
                  la.shape[-2], la.shape[-1])
        if win < 2:
            # Too small to window; fall back to whole-frame statistics rather
            # than raise. Still not clamped, so it is at least honest.
            mu_a, mu_b = la.mean(), lb.mean()
            sigma_a2 = ((la - mu_a) ** 2).mean()
            sigma_b2 = ((lb - mu_b) ** 2).mean()
            sigma_ab = ((la - mu_a) * (lb - mu_b)).mean()
            num = (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)
            den = (mu_a ** 2 + mu_b ** 2 + C1) * (sigma_a2 + sigma_b2 + C2)
            return float(num / den)

        pool = torch.nn.functional.avg_pool2d
        mu_a = pool(la, win, stride=1)
        mu_b = pool(lb, win, stride=1)
        mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
        sigma_a2 = pool(la * la, win, stride=1) - mu_a2
        sigma_b2 = pool(lb * lb, win, stride=1) - mu_b2
        sigma_ab = pool(la * lb, win, stride=1) - mu_ab

        num = (2 * mu_ab + C1) * (2 * sigma_ab + C2)
        den = (mu_a2 + mu_b2 + C1) * (sigma_a2 + sigma_b2 + C2)
        return float((num / den).mean())

    @staticmethod
    def _js_divergence(a: torch.Tensor, b: torch.Tensor, bins: int = 256) -> float:
        """Jensen-Shannon divergence of the two luminance histograms (in [0,∞))."""
        w = a.new_tensor([0.2126, 0.7152, 0.0722])
        la = (a[..., :3] * w).sum(-1).reshape(-1).clamp(min=0.0)
        lb = (b[..., :3] * w).sum(-1).reshape(-1).clamp(min=0.0)
        # Normalise to [0,1] for histogram range
        vmax = max(float(la.max()), float(lb.max()), 1.0)
        la, lb = la / vmax, lb / vmax

        ha = torch.histc(la, bins=bins, min=0.0, max=1.0) + 1e-8
        hb = torch.histc(lb, bins=bins, min=0.0, max=1.0) + 1e-8
        ha, hb = ha / ha.sum(), hb / hb.sum()
        hm = 0.5 * (ha + hb)

        def _kl(p, q):
            return (p * (p / q).log()).sum()

        return float(0.5 * _kl(ha, hm) + 0.5 * _kl(hb, hm))

    @staticmethod
    def _dynamic_range_stops(img: torch.Tensor) -> float:
        """Estimate dynamic range in stops from p1 to p99 of luminance."""
        w   = img.new_tensor([0.2126, 0.7152, 0.0722])
        lum = (img[..., :3] * w).sum(-1).reshape(-1).clamp(min=1e-8)
        n   = lum.numel()
        lo  = float(torch.kthvalue(lum, max(1, int(0.01 * n))).values)
        hi  = float(torch.kthvalue(lum, max(1, int(0.99 * n))).values)
        if lo <= 0 or hi <= 0:
            return 0.0
        return float(torch.log2(torch.tensor(hi / lo)).item())

    # ─────────────────────────────────────────────────────────────────────────

    def validate(
        self,
        image_a: torch.Tensor,
        image_b: torch.Tensor,
        win_metric: str,
    ):
        import json as _json

        # Resize b to match a if needed (safety guard)
        if image_b.shape != image_a.shape:
            image_b = torch.nn.functional.interpolate(
                image_b.permute(0, 3, 1, 2),
                size=(image_a.shape[1], image_a.shape[2]),
                mode="bilinear", align_corners=False,
            ).permute(0, 2, 3, 1)

        ssim_val = self._ssim(image_a, image_b)
        js_val   = self._js_divergence(image_a, image_b)
        dr_a     = self._dynamic_range_stops(image_a)
        dr_b     = self._dynamic_range_stops(image_b)
        dr_delta = dr_b - dr_a

        # Decide winner
        if win_metric == "dynamic_range":
            winner_img = image_b if dr_delta > 0 else image_a
            winner_lbl = "image_b (HDR blend)" if dr_delta > 0 else "image_a (baseline)"
        elif win_metric == "ssim":
            # SSIM-WINNER FIX: this used to return image_a unconditionally and
            # label it "higher SSIM", a comparison never made: SSIM is a single
            # symmetric number between the two images, so there is no "higher"
            # one to pick. The only decision one number supports is a
            # threshold, so that is what it now is, and the label carries the
            # measured value rather than an invented comparison. Above the
            # threshold the blend left structure intact and its extra range is
            # free, so it wins; below it the blend distorted structure and the
            # baseline is the safer output.
            if ssim_val >= self._SSIM_SAFE:
                winner_img = image_b
                winner_lbl = (
                    f"image_b (HDR blend, SSIM {ssim_val:.4f} >= "
                    f"{self._SSIM_SAFE}: structure preserved)"
                )
            else:
                winner_img = image_a
                winner_lbl = (
                    f"image_a (baseline, SSIM {ssim_val:.4f} < "
                    f"{self._SSIM_SAFE}: blend altered structure)"
                )
        else:  # js_divergence — higher = more tonal variety
            winner_img = image_b if js_val > 0.01 else image_a
            winner_lbl = "image_b (larger tonal shift)" if js_val > 0.01 else "image_a (blend negligible)"

        report = {
            "node":            "RadianceHDRBlendValidator",
            "ssim":            round(ssim_val, 6),
            "js_divergence":   round(js_val, 6),
            "dr_a_stops":      round(dr_a, 3),
            "dr_b_stops":      round(dr_b, 3),
            "dr_delta_stops":  round(dr_delta, 3),
            "winner":          winner_lbl,
            "recommendation":  (
                "Increase blend ratio — HDR effect is weak (JS<0.01, ΔDR<0.1 stop)"
                if js_val < 0.01 and abs(dr_delta) < 0.1 else
                "Blend is effective — HDR distribution shifted measurably"
                if dr_delta > 0 else
                "Blend compressed dynamic range — consider reducing blend ratio"
            ),
        }
        report_json = _json.dumps(report, indent=2)

        logger.info(
            "[HDRBlendValidator] SSIM=%.4f  JS=%.4f  ΔDR=%+.2f stops  winner=%s",
            ssim_val, js_val, dr_delta, winner_lbl,
        )
        return (winner_img, report_json, ssim_val, js_val, dr_delta)


NODE_CLASS_MAPPINGS = {
    "RadianceHDRBlendValidator":    RadianceHDRBlendValidator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceHDRBlendValidator":    "◎ Radiance HDR Blend Validator",
}
