import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import Tuple, Dict, Any, List, Optional

# Local imports
from .utils import tensor_to_numpy_float32, numpy_to_tensor_float32
from radiance.core.tensor.alpha import alpha_passthrough

logger = logging.getLogger("radiance.hdr.processing")

# FIX #3: scipy imported at module scope (was imported inside method bodies x3,
# re-executing on every call through the GPU fallback and CPU paths).
try:
    from scipy.ndimage import (
        gaussian_filter as _scipy_gaussian_filter,
        laplace as _scipy_laplace,
    )

    SCIPY_AVAILABLE = True
except ImportError:
    _scipy_gaussian_filter = None
    SCIPY_AVAILABLE = False

GPU_UTILS_AVAILABLE = False
try:
    # Try to import GPU utils if available in the package
    from ..gpu_utils import gpu_laplacian_pyramid_blend, gpu_local_contrast

    GPU_UTILS_AVAILABLE = True
except ImportError as _exc:
    logger.debug(
        "[Radiance] hdr/processing.py: ignoring %s from `from ..gpu_utils import gpu_laplacian_pyramid_blend, gpu_loc…`: %s",
        type(_exc).__name__, _exc,
    )

# ═══════════════════════════════════════════════════════════════════════════════
#                          HDR EXPOSURE BLENDING
# ═══════════════════════════════════════════════════════════════════════════════


class HDRExposureBlend:
    """
    Blend multiple exposures (bracketing) for extended dynamic range. Takes highlights from low exposure and shadows from high exposure for optimal color grading.
    """

    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"

    BLEND_METHODS = [
        "Mertens Fusion",
        "Luminance Weighted",
        "Shadow/Highlight Mask",
        "Exposure Weighted",
        "Laplacian Pyramid",
    ]

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "low_exposure": ("IMAGE", {
                    "tooltip": "Darker bracket, the source of highlight detail. Should be linear; "
                    "every frame is processed; a single bracket frame broadcasts across the batch.",
                }),  # Darker image - good highlights
                "high_exposure": ("IMAGE", {
                    "tooltip": "Brighter bracket, the source of shadow detail. Should be linear and "
                    "the same size as low_exposure, with matching batch length or one frame.",
                }),  # Brighter image - good shadows
                "blend_method": (cls.BLEND_METHODS, {
                    "default": "Mertens Fusion",
                    "tooltip": "How brackets are merged after EV compensation. Mertens: per-pixel "
                    "contrast/saturation/exposedness weights. Exposure Weighted favours original "
                    "bracket values away from black and clipping, then combines compensated linear values.",
                }),
            },
            "optional": {
                "mid_exposure": ("IMAGE", {
                    "tooltip": "Optional middle bracket, compensated by exposure_offset_mid. Used "
                    "by Mertens Fusion and Exposure Weighted; the other methods ignore it.",
                }),  # Optional middle exposure
                "shadow_weight": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1,
                     "tooltip": "Relative weight of high_exposure in the shadow zone (low_exposure "
                     "luma below 0.25). Shadow/Highlight Mask method only."},
                ),
                "highlight_weight": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1,
                     "tooltip": "Relative weight of low_exposure in the highlight zone (low_exposure "
                     "luma above 0.75). Shadow/Highlight Mask method only."},
                ),
                "transition_smoothness": (
                    "FLOAT",
                    {"default": 0.3, "min": 0.05, "max": 1.0, "step": 0.05,
                     "tooltip": "Width in luma of the crossfade between brackets; higher is softer. "
                     "Used by Luminance Weighted and Shadow/Highlight Mask only."},
                ),
                "exposure_offset_low": (
                    "FLOAT",
                    {"default": -2.0, "min": -6.0, "max": 0.0, "step": 0.5,
                     "tooltip": "EV of low_exposure relative to 0 EV. It is multiplied by 2^-EV "
                     "(-2 brightens it 4x) so all brackets share one linear scale."},
                ),
                "exposure_offset_high": (
                    "FLOAT",
                    {"default": 2.0, "min": 0.0, "max": 6.0, "step": 0.5,
                     "tooltip": "EV of high_exposure relative to 0 EV. It is multiplied by 2^-EV "
                     "(+2 darkens it 4x) so all brackets share one linear scale."},
                ),
                "exposure_offset_mid": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -6.0,
                        "max": 6.0,
                        "step": 0.5,
                        "tooltip": "EV of the optional middle exposure relative to 0 EV. "
                        "Applied as compensation so all three exposures are normalised "
                        "to the same scale before fusion.",
                    },
                ),
                "ghost_removal": ("BOOLEAN", {"default": False,
                    "tooltip": "Align brackets by global translation (up to 64 px). Mertens Fusion also "
                    "down-weights pixels that disagree between brackets (moving objects); other methods only align.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("blended_hdr", "blend_mask", "blend_info")
    FUNCTION = "blend_exposures"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    DESCRIPTION = "Blend multiple exposures (bracketing) for extended dynamic range. Takes highlights from low exposure and shadows from high exposure for optimal color grading."

    def _calculate_luminance(self, img: np.ndarray) -> np.ndarray:
        """Calculate Rec.709 luminance."""
        return 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]

    def _mertens_weights(
        self,
        img: np.ndarray,
        contrast_weight: float = 1.0,
        saturation_weight: float = 1.0,
        exposure_weight: float = 1.0,
    ) -> np.ndarray:
        """Calculate Mertens exposure fusion weights."""
        gray = self._calculate_luminance(img)
        if SCIPY_AVAILABLE:
            laplacian = np.abs(_scipy_laplace(gray))
        else:
            # Fallback to approximate if scipy is missing
            laplacian = np.abs(
                np.gradient(np.gradient(gray, axis=0), axis=0)
                + np.gradient(np.gradient(gray, axis=1), axis=1)
            )
        contrast = laplacian**contrast_weight

        # Saturation measure
        mean_rgb = np.mean(img, axis=-1)
        saturation = np.sqrt(np.mean((img - mean_rgb[..., np.newaxis]) ** 2, axis=-1))
        saturation = saturation**saturation_weight

        # Well-exposedness measure (Gaussian centered at 0.5)
        sigma = 0.2
        exposedness = np.exp(-0.5 * ((img - 0.5) / sigma) ** 2)
        exposedness = np.prod(exposedness, axis=-1) ** exposure_weight

        # Combined weight
        weight = contrast * saturation * exposedness + 1e-10
        return weight

    # ── Ghost Removal ─────────────────────────────────────────────────────────

    def _phase_correlation_shift(
        self, ref: np.ndarray, src: np.ndarray
    ) -> Tuple[float, float]:
        """
        Estimate the global (dy, dx) translation between ref and src using
        phase-correlation in the frequency domain.

        Algorithm (Kuglin & Hines 1975):
          1. Compute FFT of both images (luma channel only)
          2. Compute normalised cross-power spectrum: R = F·conj(G) / |F·conj(G)|
          3. Inverse FFT → cross-correlation surface
          4. Peak location == sub-pixel shift via parabolic interpolation

        Returns:
            (dy, dx) in pixels. Sub-pixel accurate to ~0.25px.
        """
        luma_ref = (0.2126 * ref[..., 0] + 0.7152 * ref[..., 1] + 0.0722 * ref[..., 2]).astype(np.float32)
        luma_src = (0.2126 * src[..., 0] + 0.7152 * src[..., 1] + 0.0722 * src[..., 2]).astype(np.float32)

        # Hann window reduces spectral leakage at image edges
        H, W = luma_ref.shape
        hann_y = np.hanning(H).reshape(-1, 1).astype(np.float32)
        hann_x = np.hanning(W).reshape(1, -1).astype(np.float32)
        window = hann_y * hann_x

        F_ref = np.fft.fft2(luma_ref * window)
        F_src = np.fft.fft2(luma_src * window)

        cross_power = F_ref * np.conj(F_src)
        denom = np.abs(cross_power) + 1e-10
        r = np.fft.ifft2(cross_power / denom).real

        # Peak in the correlation surface
        peak_idx = np.unravel_index(np.argmax(r), r.shape)
        py, px = int(peak_idx[0]), int(peak_idx[1])

        # Parabolic sub-pixel refinement around the peak
        def _parabolic_refinement(arr, idx, size):
            if 1 <= idx < size - 1:
                y0, y1, y2 = arr[idx - 1], arr[idx], arr[idx + 1]
                denom_p = 2.0 * y1 - y0 - y2
                if abs(denom_p) > 1e-8:
                    return idx + 0.5 * (y0 - y2) / denom_p
            return float(idx)

        sub_y = _parabolic_refinement(r[:, px], py, H)
        sub_x = _parabolic_refinement(r[py, :], px, W)

        # Wrap: shifts > half the image are negative (other direction)
        dy = sub_y if sub_y <= H / 2 else sub_y - H
        dx = sub_x if sub_x <= W / 2 else sub_x - W

        # Clamp to reasonable alignment range (±64px) — larger shifts indicate
        # a completely different framing, which ghost removal cannot fix
        dy = max(-64, min(64, dy))
        dx = max(-64, min(64, dx))

        return float(dy), float(dx)

    def _warp_translation(
        self, img: np.ndarray, dy: float, dx: float
    ) -> np.ndarray:
        """
        Apply a sub-pixel translation to img using bilinear interpolation.
        Uses torch F.grid_sample for quality and speed.

        Args:
            img: (H, W, C) float32
            dy, dx: shift in pixels (positive = shift image down/right)

        Returns:
            (H, W, C) warped image, edge pixels filled with nearest-border value.
        """
        H, W, C = img.shape
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()  # (1, C, H, W)

        # Normalised shift: grid_sample uses coordinates in [-1, 1]
        norm_dy = 2.0 * dy / H
        norm_dx = 2.0 * dx / W

        # Identity grid
        theta = torch.tensor(
            [[1.0, 0.0, -norm_dx],
             [0.0, 1.0, -norm_dy]],
            dtype=torch.float32
        ).unsqueeze(0)  # (1, 2, 3)

        grid = F.affine_grid(theta, t.shape, align_corners=False)
        warped = F.grid_sample(t, grid, mode="bilinear", padding_mode="border", align_corners=False)
        return warped.squeeze(0).permute(1, 2, 0).numpy()  # (H, W, C)

    def _ghost_weight_mask(
        self,
        images: List[np.ndarray],
        sigma: float = 0.1,
    ) -> List[np.ndarray]:
        """
        Compute per-pixel ghost confidence weights for a list of aligned images.

        Method:
          1. Compute the per-pixel median across all brackets.
          2. For each bracket, measure the absolute deviation from the median.
          3. Convert deviation to a confidence weight via a Gaussian:
               w = exp(-0.5 * (deviation / sigma)²)
          4. Where deviation is small (consistent across brackets) → weight ≈ 1
             Where deviation is large (ghost/motion) → weight → 0

        Args:
            images: List of (H, W, C) aligned float32 arrays
            sigma:  Gaussian width for deviation → confidence mapping.
                    0.1 = strict (only very consistent pixels get full weight)
                    0.2 = lenient (tolerates minor inter-frame differences)

        Returns:
            List of (H, W) float32 ghost masks, one per input image.
        """
        if len(images) < 2:
            return [np.ones(images[0].shape[:2], dtype=np.float32)]

        # Convert to luma for deviation measurement
        lumas = [
            (0.2126 * im[..., 0] + 0.7152 * im[..., 1] + 0.0722 * im[..., 2])
            for im in images
        ]
        stack = np.stack(lumas, axis=0)  # (N, H, W)
        median = np.median(stack, axis=0)  # (H, W)

        masks = []
        for luma in lumas:
            deviation = np.abs(luma - median)
            # Gaussian confidence: low deviation = high confidence
            confidence = np.exp(-0.5 * (deviation / (sigma + 1e-8)) ** 2)
            masks.append(confidence.astype(np.float32))

        return masks

    def _align_and_deghost(
        self,
        images: List[np.ndarray],
        ref_idx: int = -1,
        ghost_sigma: float = 0.12,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Align all exposure brackets to one reference and compute ghost masks.

        Args:
            images:      List of (H, W, 3) float32 exposure brackets
            ref_idx:     Index of reference frame (-1 = auto: median luminance)
            ghost_sigma: Ghost deviation threshold (smaller = stricter)

        Returns:
            (aligned_images, ghost_masks)
            aligned_images: List of (H, W, 3) warped to reference frame
            ghost_masks:    List of (H, W) confidence weights in [0, 1]
        """
        # Auto reference selection: use the bracket whose mean luminance is
        # closest to the median of all bracket mean-luminances
        if ref_idx < 0:
            means = [img.mean() for img in images]
            median_mean = np.median(means)
            ref_idx = int(np.argmin([abs(m - median_mean) for m in means]))

        ref = images[ref_idx]
        aligned = []

        for i, img in enumerate(images):
            if i == ref_idx:
                aligned.append(img)
                continue
            dy, dx = self._phase_correlation_shift(ref, img)
            if abs(dy) > 0.25 or abs(dx) > 0.25:  # Only warp if shift is significant
                warped = self._warp_translation(img, dy, dx)
                logger.debug(
                    f"[HDR Ghost Removal] Bracket {i}: aligned shift dy={dy:.2f}px dx={dx:.2f}px"
                )
            else:
                warped = img
            aligned.append(warped)

        ghost_masks = self._ghost_weight_mask(aligned, sigma=ghost_sigma)
        return aligned, ghost_masks

    def _mertens_fusion(self, images: list, weights: list = None) -> np.ndarray:
        """Mertens exposure fusion algorithm."""
        if weights is None:
            weights = [self._mertens_weights(img) for img in images]

        # Normalize weights
        weight_sum = sum(weights)
        weight_sum = np.maximum(weight_sum, 1e-10)
        normalized_weights = [w / weight_sum for w in weights]

        # Weighted blend
        result = np.zeros_like(images[0])
        for img, w in zip(images, normalized_weights):
            result += img * w[..., np.newaxis]

        return result

    def _luminance_weighted_blend(
        self, low_exp: np.ndarray, high_exp: np.ndarray, transition: float = 0.3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Luminance-weighted blending for natural HDR merge."""
        # Calculate luminance of both images
        lum_low = self._calculate_luminance(low_exp)
        lum_high = self._calculate_luminance(high_exp)

        # Create smooth transition mask based on luminance
        # We use the average luminance for a more balanced cross-fade
        lum_avg = (lum_low + lum_high) / 2.0

        # Sigmoid-based transition
        mid_point = 0.5
        x = (lum_avg - mid_point) / transition
        low_weight = 1.0 / (1.0 + np.exp(-x))  # High weight for bright areas
        high_weight = 1.0 - low_weight

        # Blend
        result = (
            low_exp * low_weight[..., np.newaxis]
            + high_exp * high_weight[..., np.newaxis]
        )

        mask = low_weight[..., np.newaxis].repeat(3, axis=-1)
        return result, mask

    def _shadow_highlight_blend(
        self,
        low_exp: np.ndarray,
        high_exp: np.ndarray,
        shadow_weight: float,
        highlight_weight: float,
        transition: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract shadows from high exposure, highlights from low exposure."""
        lum_low = self._calculate_luminance(low_exp)

        # Shadow mask (where low exposure is dark)
        shadow_threshold = 0.25
        shadow_mask = np.clip((shadow_threshold - lum_low) / transition + 0.5, 0, 1)

        # Highlight mask (where low exposure is bright)
        highlight_threshold = 0.75
        highlight_mask = np.clip(
            (lum_low - highlight_threshold) / transition + 0.5, 0, 1
        )

        # Midtone mask (remainder)
        midtone_mask = 1.0 - shadow_mask - highlight_mask
        midtone_mask = np.maximum(midtone_mask, 0)

        # Blend: shadows from high_exp, highlights from low_exp, midtones averaged
        result = (
            high_exp * shadow_mask[..., np.newaxis] * shadow_weight
            + low_exp * highlight_mask[..., np.newaxis] * highlight_weight
            + (low_exp + high_exp) / 2 * midtone_mask[..., np.newaxis]
        )

        # Normalize
        total_weight = (
            shadow_mask * shadow_weight
            + highlight_mask * highlight_weight
            + midtone_mask
        )
        total_weight = np.maximum(total_weight, 1e-10)
        result = result / total_weight[..., np.newaxis]

        # Visualization mask
        mask = np.stack([shadow_mask, midtone_mask, highlight_mask], axis=-1)
        return result, mask

    def _laplacian_pyramid_blend(
        self, low_exp: np.ndarray, high_exp: np.ndarray, levels: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Multi-scale Laplacian pyramid blending for seamless HDR merge."""

        # Use GPU-accelerated version if available
        if GPU_UTILS_AVAILABLE and torch.cuda.is_available():
            try:
                device = torch.device("cuda")
                low_tensor = torch.from_numpy(low_exp).to(device)
                high_tensor = torch.from_numpy(high_exp).to(device)

                # FIX #5: create a soft blending mask instead of a hard binary one.
                # The CPU path softens the mask through the Gaussian pyramid; here we
                # apply avg_pool2d to approximate that smoothing before passing to the
                # GPU blend function, preventing hard seam artefacts at the 50% boundary.
                lum = (
                    0.2126 * low_tensor[..., 0]
                    + 0.7152 * low_tensor[..., 1]
                    + 0.0722 * low_tensor[..., 2]
                )
                mask_hard = (lum > 0.5).float()
                # Soft-blur the mask (~31px kernel ≈ one pyramid level of smoothing)
                mask_soft = (
                    torch.nn.functional.avg_pool2d(
                        mask_hard.unsqueeze(0).unsqueeze(0),
                        kernel_size=31,
                        stride=1,
                        padding=15,
                    )
                    .squeeze(0)
                    .squeeze(0)
                )
                mask = mask_soft.unsqueeze(-1).expand_as(low_tensor)

                result_tensor = gpu_laplacian_pyramid_blend(
                    low_tensor, high_tensor, mask, levels
                )
                result = result_tensor.cpu().numpy()
                mask_np = mask_soft.cpu().numpy()[..., np.newaxis].repeat(3, axis=-1)
                return result, mask_np
            except Exception as e:
                logger.debug(f"GPU Laplacian blend failed, falling back to CPU: {e}")

        # CPU fallback using scipy Gaussian pyramid
        # FIX #3: scipy imported at module scope; removed inline import here.

        def build_gaussian_pyramid(img, levels):
            pyramid = [img]
            for _ in range(levels - 1):
                blurred = _scipy_gaussian_filter(pyramid[-1], sigma=2)
                downsampled = blurred[::2, ::2]
                pyramid.append(downsampled)
            return pyramid

        def build_laplacian_pyramid(gaussian_pyr):
            laplacian = []
            for i in range(len(gaussian_pyr) - 1):
                upsampled = np.repeat(
                    np.repeat(gaussian_pyr[i + 1], 2, axis=0), 2, axis=1
                )
                # Handle size mismatch
                h, w = gaussian_pyr[i].shape[:2]
                upsampled = upsampled[:h, :w]
                laplacian.append(gaussian_pyr[i] - upsampled)
            laplacian.append(gaussian_pyr[-1])
            return laplacian

        # Build pyramids
        g_low = build_gaussian_pyramid(low_exp, levels)
        g_high = build_gaussian_pyramid(high_exp, levels)
        l_low = build_laplacian_pyramid(g_low)
        l_high = build_laplacian_pyramid(g_high)

        # Create mask based on luminance
        lum = self._calculate_luminance(low_exp)
        mask = (lum > 0.5).astype(np.float32)
        mask_pyr = build_gaussian_pyramid(
            mask[..., np.newaxis].repeat(3, axis=-1), levels
        )

        # Blend pyramids
        blended_laplacian = []
        for l_l, l_h, m in zip(l_low, l_high, mask_pyr):
            blended = l_l * m + l_h * (1 - m)
            blended_laplacian.append(blended)

        # Reconstruct from blended Laplacian pyramid
        result = blended_laplacian[-1]
        for i in range(len(blended_laplacian) - 2, -1, -1):
            upsampled = np.repeat(np.repeat(result, 2, axis=0), 2, axis=1)
            h, w = blended_laplacian[i].shape[:2]
            upsampled = upsampled[:h, :w]
            result = upsampled + blended_laplacian[i]

        return result, mask[..., np.newaxis].repeat(3, axis=-1)

    def blend_exposures(
        self,
        low_exposure: torch.Tensor,
        high_exposure: torch.Tensor,
        blend_method: str = "Mertens Fusion",
        mid_exposure: torch.Tensor = None,
        shadow_weight: float = 1.0,
        highlight_weight: float = 1.0,
        transition_smoothness: float = 0.3,
        exposure_offset_low: float = -2.0,
        exposure_offset_high: float = 2.0,
        exposure_offset_mid: float = 0.0,
        ghost_removal: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, str]:

        if blend_method not in self.BLEND_METHODS:
            raise ValueError(f"Unknown exposure blend method: {blend_method}")
        # FIX-008: the fusion weighted and exposed every channel, so a matte
        # came back scaled (1.02 for a 0.9 alpha). Brackets are fused on RGB;
        # the low bracket's alpha is passed through.
        _brs = [t for t in (low_exposure, high_exposure, mid_exposure) if t is not None]
        if any(t.shape[-1] == 4 for t in _brs):
            src = low_exposure if low_exposure.shape[-1] == 4 else next(t for t in _brs if t.shape[-1] == 4)
            alpha = (src if src.ndim == 4 else src.unsqueeze(0))[..., 3:4]

            def _rgb(t):
                return t[..., :3] if t is not None and t.shape[-1] == 4 else t
            image, mask, report = self.blend_exposures(
                _rgb(low_exposure), _rgb(high_exposure), blend_method, _rgb(mid_exposure),
                shadow_weight, highlight_weight, transition_smoothness, exposure_offset_low,
                exposure_offset_high, exposure_offset_mid, ghost_removal)
            if alpha.shape[0] == 1 and image.shape[0] > 1:
                alpha = alpha.expand(image.shape[0], -1, -1, -1)
            if alpha.shape[:3] == image.shape[:3]:
                image = torch.cat([image, alpha.to(image.dtype)], dim=-1)
            return image, mask, report
        if low_exposure.ndim == 3:
            low_exposure = low_exposure.unsqueeze(0)
        if high_exposure.ndim == 3:
            high_exposure = high_exposure.unsqueeze(0)
        if mid_exposure is not None and mid_exposure.ndim == 3:
            mid_exposure = mid_exposure.unsqueeze(0)
        brackets = [low_exposure, high_exposure]
        if mid_exposure is not None:
            brackets.append(mid_exposure)
        if any(x.ndim != 4 or x.shape[0] == 0 for x in brackets):
            raise ValueError("Exposure brackets must be non-empty IMAGE batches (B,H,W,C).")
        batch = max(x.shape[0] for x in brackets)
        if any(x.shape[1:] != low_exposure.shape[1:] or x.shape[0] not in (1, batch) for x in brackets):
            raise ValueError("Exposure brackets must have matching dimensions and equal batch lengths, or one frame to broadcast.")
        if batch > 1:
            images, masks, reports = [], [], []
            for i in range(batch):
                low = low_exposure[i:i+1] if low_exposure.shape[0] > 1 else low_exposure
                high = high_exposure[i:i+1] if high_exposure.shape[0] > 1 else high_exposure
                mid = mid_exposure
                if mid is not None and mid.shape[0] > 1:
                    mid = mid[i:i+1]
                image, mask, report = self.blend_exposures(
                    low, high, blend_method, mid, shadow_weight, highlight_weight,
                    transition_smoothness, exposure_offset_low, exposure_offset_high,
                    exposure_offset_mid, ghost_removal)
                images.append(image)
                masks.append(mask)
                reports.append(f"Frame {i + 1}: {report}")
            return torch.cat(images), torch.cat(masks), "\n".join(reports)

        low_np = tensor_to_numpy_float32(low_exposure)
        high_np = tensor_to_numpy_float32(high_exposure)

        if ghost_removal:
            brackets_count = 2 if mid_exposure is None else 3
            logger.info(
                "[HDR Ghost Removal] Running phase-correlation alignment + ghost masking "
                f"on {brackets_count} brackets."
            )

        # Handle batch dimension
        if low_np.ndim == 4:
            low_np = low_np[0]
            high_np = high_np[0]

        # Apply exposure compensation to bring to common scale
        low_np = low_np * (2.0 ** (-exposure_offset_low))  # Brighten low exposure
        high_np = high_np * (2.0 ** (-exposure_offset_high))  # Darken high exposure

        # ── Ghost Removal ──────────────────────────────────────────────────────
        ghost_masks = None  # populated below when ghost_removal=True

        # Perform blending based on method
        if blend_method in ("Mertens Fusion", "Exposure Weighted"):
            images = [low_np, high_np]
            offsets = [exposure_offset_low, exposure_offset_high]
            if mid_exposure is not None:
                mid_np = tensor_to_numpy_float32(mid_exposure)
                if mid_np.ndim == 4:
                    mid_np = mid_np[0]
                # FIX #2: apply EV compensation so mid_exposure is normalised to
                # the same 0-EV scale as low_np and high_np.
                # Previously mid_np was inserted raw with no offset applied.
                mid_np = mid_np * (2.0 ** (-exposure_offset_mid))
                images.insert(1, mid_np)
                offsets.insert(1, exposure_offset_mid)

            if ghost_removal:
                images, ghost_masks = self._align_and_deghost(images)

            if blend_method == "Exposure Weighted":
                # Reliability comes from the original bracket, before EV
                # compensation: black and saturated channels carry no detail.
                weights = [np.maximum(0.0, 1.0 - 2.0 * np.abs(img * (2.0 ** ev) - 0.5)).prod(axis=-1)
                           for img, ev in zip(images, offsets)]
                if ghost_masks is not None:
                    weights = [w * gm for w, gm in zip(weights, ghost_masks)]
                total = sum(weights)
                missing = total <= 1e-10
                weights = [np.where(missing, 1.0, w) for w in weights]
                total = sum(weights)
                result = sum(img * (w / total)[..., None] for img, w in zip(images, weights))
                mask = np.repeat((weights[0] / total)[..., None], 3, axis=-1)
            else:
                mertens_weights = [self._mertens_weights(img) for img in images]
                if ghost_masks is not None:
                    mertens_weights = [w * gm for w, gm in zip(mertens_weights, ghost_masks)]
                result = self._mertens_fusion(images, weights=mertens_weights)
                mask = np.ones_like(result) * 0.5

        elif blend_method == "Luminance Weighted":
            if ghost_removal:
                [low_np, high_np], _ = self._align_and_deghost([low_np, high_np])
            result, mask = self._luminance_weighted_blend(
                low_np, high_np, transition_smoothness
            )

        elif blend_method == "Shadow/Highlight Mask":
            if ghost_removal:
                [low_np, high_np], _ = self._align_and_deghost([low_np, high_np])
            result, mask = self._shadow_highlight_blend(
                low_np, high_np, shadow_weight, highlight_weight, transition_smoothness
            )



        elif blend_method == "Laplacian Pyramid":
            if ghost_removal:
                [low_np, high_np], _ = self._align_and_deghost([low_np, high_np])
            result, mask = self._laplacian_pyramid_blend(low_np, high_np)

        # Ensure valid range for 32-bit HDR
        result = np.maximum(result, 0.0)


        # Calculate blend statistics
        positive = result[result > 0]
        dynamic_range = (np.log2(np.maximum(result.max(), 1e-10) / np.maximum(positive.min(), 1e-10))
                         if positive.size else 0.0)

        info = f"Method: {blend_method} | DR: {dynamic_range:.1f} stops | Range: [{result.min():.3f}, {result.max():.3f}]"

        return (
            numpy_to_tensor_float32(result),
            numpy_to_tensor_float32(mask.astype(np.float32)),
            info,
        )


class HDRShadowHighlightRecovery:
    """
    Recover shadow and highlight detail from a single HDR image for better color grading flexibility.
    """

    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Linear HDR image (values above 1.0 allowed). Every frame of a batch "
                    "is processed; alpha passes through unchanged.",
                }),
                "shadow_amount": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05,
                     "tooltip": "Shadow gain at black: pixels are multiplied by up to 1 + amount, "
                     "fading out with luma (see shadow_tone). 0 = off."},
                ),
                "highlight_amount": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05,
                     "tooltip": "Highlight compression above highlight_tone, never clipping. At the "
                     "default, luma 1.0 is scaled by 0.8 and brighter values more. 0 = off."},
                ),
            },
            "optional": {
                "shadow_tone": (
                    "FLOAT",
                    {"default": 0.25, "min": 0.0, "max": 0.5, "step": 0.01,
                     "tooltip": "Linear-luma width of the shadow lift. It decays exponentially and is "
                     "down to about 5% at this luma."},
                ),
                "highlight_tone": (
                    "FLOAT",
                    {"default": 0.75, "min": 0.5, "max": 1.0, "step": 0.01,
                     "tooltip": "Linear luma above which highlight compression starts; it keeps "
                     "increasing past 1.0 for super-whites."},
                ),
                "color_correction": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1,
                     "tooltip": "Desaturates lifted shadows to counter the colour boost of the lift "
                     "(up to 30% at 1.0, deepest shadows most). 0 = off."},
                ),
                "local_contrast": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.1,
                     "tooltip": "Large-radius local contrast (about 50 px blur of luma). Positive "
                     "adds, negative flattens, 0 = off."},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("recovered_image",)
    FUNCTION = "recover"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    DESCRIPTION = "Recover shadow and highlight detail from a single HDR image for better color grading flexibility."

    def recover(
        self,
        image: torch.Tensor,
        shadow_amount: float = 0.5,
        highlight_amount: float = 0.5,
        shadow_tone: float = 0.25,
        highlight_tone: float = 0.75,
        color_correction: float = 0.5,
        local_contrast: float = 0.0,
    ) -> Tuple[torch.Tensor]:
        # FIX-009: only image[0] was processed and a 1-frame batch returned,
        # so a clip lost every frame after the first. FIX-008: the gains were
        # applied to all channels, alpha included. Every frame is processed
        # now, on RGB only, with alpha passed through.
        arr = tensor_to_numpy_float32(image)
        if arr.ndim == 3:
            arr = arr[None]
        if arr.ndim != 4 or arr.shape[-1] < 3:
            raise ValueError(f"HDR Shadow / Highlight Recovery needs an RGB(A) IMAGE, got {tuple(arr.shape)}")
        frames = []
        for f in arr:
            rgb = self._recover_frame(np.ascontiguousarray(f[..., :3]), shadow_amount, highlight_amount,
                                      shadow_tone, highlight_tone, color_correction, local_contrast)
            frames.append(np.concatenate([rgb, f[..., 3:]], axis=-1) if f.shape[-1] > 3 else rgb)
        return (numpy_to_tensor_float32(np.stack(frames).astype(np.float32)),)

    @staticmethod
    def _recover_frame(
        img: np.ndarray,
        shadow_amount: float,
        highlight_amount: float,
        shadow_tone: float,
        highlight_tone: float,
        color_correction: float,
        local_contrast: float,
    ) -> np.ndarray:
        # Calculate luminance
        lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
        lum = np.maximum(lum, 1e-10)

        # Shadow recovery
        # New HDR-safe mask: 1.0 at 0, smooth falloff to 0.0 at shadow_tone
        shadow_ratio = lum / (shadow_tone + 1e-6)
        shadow_mask = np.exp(-3.0 * shadow_ratio)  # Exponential decay

        shadow_boost = 1.0 + shadow_amount * shadow_mask

        # Highlight recovery
        # New HDR-safe mask: 0.0 below tone, smooth rise to 1.0+ above

        # Normalized position above threshold
        h_pos = (lum - highlight_tone) / (1.0 - highlight_tone + 1e-6)

        # Sigmoid-like activation that works for > 1.0
        # h_pos: 0.0 at highlight_tone
        # rises to 1.0 at h_pos=1.0 (original lum=1.0)
        # continues rising for h_pos > 1.0 (super-whites)

        highlight_mask = np.maximum(0.0, h_pos)

        # Apply reduction (soft compression, non-clamping)
        highlight_reduce = 1.0 / (1.0 + highlight_amount * highlight_mask * 0.5)

        # Apply to image
        result = img * shadow_boost[..., np.newaxis] * highlight_reduce[..., np.newaxis]

        # Color correction to reduce oversaturation in lifted shadows
        if color_correction > 0:
            new_lum = (
                0.2126 * result[..., 0]
                + 0.7152 * result[..., 1]
                + 0.0722 * result[..., 2]
            )
            new_lum = np.maximum(new_lum, 1e-10)
            sat_factor = 1.0 - (shadow_mask * color_correction * 0.3)
            result = new_lum[..., np.newaxis] + sat_factor[..., np.newaxis] * (
                result - new_lum[..., np.newaxis]
            )

        # Local contrast enhancement
        if local_contrast != 0:
            # Use GPU-accelerated local contrast if available
            if GPU_UTILS_AVAILABLE and torch.cuda.is_available():
                try:
                    device = torch.device("cuda")
                    result_tensor = torch.from_numpy(result.astype(np.float32)).to(
                        device
                    )
                    result_tensor = gpu_local_contrast(
                        result_tensor, sigma=50.0, amount=local_contrast
                    )
                    result = result_tensor.cpu().numpy().astype(np.float32)
                except Exception as e:
                    logger.debug(f"GPU local contrast failed, falling back to CPU: {e}")
                    # FIX #3: use module-level _scipy_gaussian_filter
                    local_lum = _scipy_gaussian_filter(lum, sigma=50)
                    detail = lum / (local_lum + 1e-10)
                    contrast_boost = 1.0 + local_contrast * (detail - 1.0)
                    result = result * contrast_boost[..., np.newaxis]
            else:
                # FIX #3: use module-level _scipy_gaussian_filter
                local_lum = _scipy_gaussian_filter(lum, sigma=50)
                detail = lum / (local_lum + 1e-10)
                contrast_boost = 1.0 + local_contrast * (detail - 1.0)
                result = result * contrast_boost[..., np.newaxis]

        return result.astype(np.float32)


# FIX-008: exposure / gamma / lift / normalise are colour ops; alpha passes through.
@alpha_passthrough("image")
class GPUTensorOps:
    """
    GPU-accelerated HDR operations. Fast exposure, gamma, and normalization.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Image to process, usually linear float. The whole batch is processed.",
                }),
                "operation": (
                    ["Exposure", "Gamma", "Lift/Gain", "Normalize", "Clamp"],
                    {"default": "Exposure",
                     "tooltip": "Exposure: x * 2^value. Gamma: sign-preserving x^(1/value). "
                     "Lift/Gain: adds value (an offset only, no gain). Normalize: per-frame min/max "
                     "to 0-1. Clamp: to 0-1."},
                ),
            },
            "optional": {
                "value": (
                    "FLOAT",
                    {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1,
                     "tooltip": "Exposure: stops. Gamma: gamma (0 means 2.2, anything below 0.1 "
                     "including negatives is treated as 0.1). Lift/Gain: offset added. Ignored by "
                     "Normalize and Clamp."},
                ),
                "force_gpu": ("BOOLEAN", {"default": True,
                    "tooltip": "Run on CUDA when available. Off always runs on the CPU. Results are the same either way.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("processed_image", "performance_info")
    FUNCTION = "process"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    DESCRIPTION = (
        "GPU-accelerated HDR operations. Fast exposure, gamma, and normalization."
    )

    def process(
        self,
        image: torch.Tensor,
        operation: str,
        value: float = 0.0,
        force_gpu: bool = True,
    ) -> Tuple[torch.Tensor, str]:

        import time

        start_time = time.time()

        # Determine device
        if force_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        img = image.to(device)

        # Apply operation
        if operation == "Exposure":
            result = img * (2.0**value)
        elif operation == "Gamma":
            # FIX (VFX Audit): Previous code used torch.clamp(img, 0.001, None) which:
            #  1. Destroyed negative wide-gamut values (clipped to 0.001).
            #  2. Added a 0.001 pedestal to pure black (visible on 10-bit+ displays).
            # Fix: use sign-preserving power, consistent with Float32ColorCorrect.
            gamma = max(0.1, value) if value != 0 else 2.2
            result = torch.sign(img) * torch.pow(torch.clamp(torch.abs(img), min=1e-12), 1.0 / gamma)
        elif operation == "Lift/Gain":
            result = img + value  # Simple lift
        elif operation == "Normalize":
            # FIX #4: normalise each frame independently.
            # Previous: img.min()/img.max() computed the global extremes across the
            # entire batch, so frames with a narrower tonal range were not expanded
            # to [0, 1] — they were compressed relative to the brightest/darkest frame.
            result = torch.zeros_like(img)
            for b in range(img.shape[0]):
                min_val = img[b].min()
                max_val = img[b].max()
                result[b] = (img[b] - min_val) / (max_val - min_val + 1e-10)
        elif operation == "Clamp":
            result = torch.clamp(img, 0.0, 1.0)
        else:
            result = img

        result = result.cpu()

        elapsed = (time.time() - start_time) * 1000
        # FIX #6: report the device actually used, not always the GPU name.
        # Previous code called get_device_name(0) unconditionally even when
        # force_gpu=False or CUDA was unavailable and we ran on CPU.
        if force_gpu and torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
        else:
            device_name = "CPU"
        info = f"Device: {device_name} | Op: {operation} | Time: {elapsed:.2f}ms"

        return (result, info)


# =============================================================================
# NODE MAPPINGS
# FIX #1: NODE_CLASS_MAPPINGS was absent — all three nodes were invisible to
# ComfyUI and could not be used in any workflow.
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "RadianceHDRExposureBlend": HDRExposureBlend,
    "RadianceHDRShadowHighlight": HDRShadowHighlightRecovery,
    "RadianceGPUTensorOps": GPUTensorOps,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceHDRExposureBlend": "◎ HDR Exposure Blend",
    "RadianceHDRShadowHighlight": "◎ HDR Shadow/Highlight Recovery",
    "RadianceGPUTensorOps": "◎ GPU Tensor Ops",
}
