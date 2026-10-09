from __future__ import annotations

import logging
import math
import torch
import torch.nn.functional as F

logger = logging.getLogger("radiance")


class RadianceDenoise:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Generate"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image or frame batch to denoise, float32 in any encoding (scene-linear HDR above 1.0 is kept). A batch of more than one frame enables the temporal options."}),
                "filter_type": (["Bilateral", "Guided"], {"default": "Bilateral",
                    "tooltip": "The core spatial denoising algorithm. Bilateral preserves edges; Guided runs faster, preserves finer detail, and avoids halos."}),
                "d": ("INT", {"default": 9, "min": 1, "max": 50,
                    "tooltip": "Base filter radius in pixels. It sets the band split (d/2 and 1.5 x d box blurs) and the per-band filter radius (d/4, d/2, d); Bilateral caps each radius at 8 px."
                }),
                "sigmaColor": (
                    "FLOAT",
                    {"default": 0.15, "min": 0.0, "max": 10.0, "step": 0.01,
                     "tooltip": (
                         "Color similarity threshold. High = smoother, but can lose edge detail. "
                         "For HDR images, scale is auto-adjusted if hdr_auto_sigma is ON. "
                         "This is bypassed if auto_profiling is enabled."
                     )},
                ),
                "sigmaSpace": (
                    "FLOAT",
                    {"default": 75.0, "min": 0.1, "max": 500.0, "step": 0.5,
                     "tooltip": "Gaussian spatial falloff in pixels for Bilateral weights and Guided local statistics. Higher values give flatter weights within the filter radius."},
                ),
                "hdr_auto_sigma": (
                    "BOOLEAN",
                    {"default": True,
                     "tooltip": (
                         "Multiplies the colour threshold by the peak value of the whole batch when that peak exceeds 1.01, "
                         "so HDR input gets a proportionally wider threshold. No effect on 0-1 images."
                     )},
                ),
                "auto_profiling": (
                    "BOOLEAN",
                    {"default": False,
                     "tooltip": "Replaces sigmaColor with a measured noise level: the lowest luma standard deviation among 32 px patches, times profile_multiplier. Other settings are unchanged."}
                ),
                "profile_multiplier": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1,
                     "tooltip": "Adjusts the strength of the automatically profiled noise signature. Raise if noise remains; lower if details get soft."}
                ),
                "luma_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                     "tooltip": "Multiplier for spatial denoise strength on brightness (Luma). Lower this (e.g. 0.2) to keep natural fine grain."}
                ),
                "chroma_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                     "tooltip": "Multiplier for spatial denoise strength on colors (Chroma). Raise this to wash away annoying chromatic noise."}
                ),
                "high_freq_denoise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                     "tooltip": "Denoise strength for fine details and high-frequency pixel grain."}
                ),
                "mid_freq_denoise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                     "tooltip": "Denoise strength for medium textures and compression artifacts."}
                ),
                "low_freq_denoise": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05,
                     "tooltip": "Denoise strength for large coarse gradient splotches."}
                ),
                "joint_chroma_guidance": (
                    "BOOLEAN",
                    {"default": True,
                     "tooltip": "Enables Joint Guided filtering. Uses the sharp structural boundaries of the Luma channel to guide Chroma smoothing, preventing color bleeding."}
                ),
                "temporal_blend": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 0.95, "step": 0.05,
                     "tooltip": "Enables multi-frame temporal de-flickering. 0.0 is off. Higher values blend adjacent frames to stabilize video."}
                ),
                "temporal_radius": (
                    "INT",
                    {"default": 1, "min": 1, "max": 4, "step": 1,
                     "tooltip": "Temporal search window size. 1 searches 1 prev/next frame. 2 searches 2 prev/next frames, etc. Higher values de-flicker better but are slower."}
                ),
                "temporal_threshold": (
                    "FLOAT",
                    {"default": 0.05, "min": 0.01, "max": 0.5, "step": 0.01,
                     "tooltip": "Flicker delta threshold. Lower values prevent ghosting/trailing by only blending static or slow-moving areas."}
                ),
                "motion_compensation": (
                    "BOOLEAN",
                    {"default": True,
                     "tooltip": "Before temporal blending, aligns each neighbour frame to this one: motion is estimated coarse to fine by 8x8 block matching (up to about 30 pixels per frame on large frames, less on small ones) and the neighbour is warped, edges repeating. Off blends neighbours where they are."}
                ),
                "detail_recovery": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                     "tooltip": "Restores the high-frequency part of the removed residual (radius d/2), retaining low-frequency denoising. 0 = off, 1 = full detail residual. Fine noise can return along with texture; this is not a full original-image blend."}
                ),
                "sharpen_strength": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05,
                     "tooltip": "Adds a subtle post-sharpening (unsharp mask) to recover perceived edge crispness."}
                ),
                "view_mode": (
                    ["Denoised", "Noise Residual", "Luma (Y)", "Chroma (Cb/Cr)"],
                    {"default": "Denoised",
                     "tooltip": "Diagnostic view options. 'Noise Residual' is extremely helpful to see exactly what details are being removed."}
                )
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "denoise"

    DESCRIPTION = (
        "Ultimate professional-grade 32-bit float spatial-temporal denoising node. "
        "Splits Luma and Chroma, offers automatic hands-free noise profiling, "
        "applies 3-band multiscale frequency decomposition, provides joint guided chroma filtering, "
        "and implements multi-frame block-matching motion-compensated temporal stabilization."
    )

    @staticmethod
    def _rgb_to_ycrcb_t(rgb: torch.Tensor) -> torch.Tensor:
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        y = 0.2126 * r + 0.7152 * g + 0.0722 * b
        cr = (r - y) / 1.5748
        cb = (b - y) / 1.8556
        return torch.stack([y, cr, cb], dim=-1)

    @staticmethod
    def _ycrcb_to_rgb_t(ycrcb: torch.Tensor) -> torch.Tensor:
        y, cr, cb = ycrcb[..., 0], ycrcb[..., 1], ycrcb[..., 2]
        r = y + 1.5748 * cr
        b = y + 1.8556 * cb
        g = y - 0.468124273 * cr - 0.187324273 * cb
        return torch.stack([r, g, b], dim=-1)

    @staticmethod
    def _box_blur(x_bchw: torch.Tensor, radius: int) -> torch.Tensor:
        radius = max(1, int(radius))
        kernel = 2 * radius + 1
        return F.avg_pool2d(
            x_bchw,
            kernel_size=kernel,
            stride=1,
            padding=radius,
            count_include_pad=False,
        )

    @classmethod
    def _guided_filter_t(cls, guide: torch.Tensor, src: torch.Tensor, radius: int, eps: float,
                         sigma_space: float = 75.0) -> torch.Tensor:
        radius = max(1, int(radius))
        offsets = torch.arange(-radius, radius + 1, device=src.device, dtype=src.dtype)
        kernel = torch.exp(-0.5 * (offsets / max(float(sigma_space), 1e-6)).square())
        kernel = kernel / kernel.sum()

        def convolve(x):
            channels = x.shape[1]
            horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
            vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
            x = F.conv2d(x, horizontal, padding=(0, radius), groups=channels)
            return F.conv2d(x, vertical, padding=(radius, 0), groups=channels)

        # Normalise truncated windows at image edges, matching the existing
        # box statistics without inventing reflected image structure.
        normalizer = convolve(torch.ones_like(guide[:, :1]))

        def mean(x):
            return convolve(x) / normalizer

        mean_i = mean(guide)
        mean_p = mean(src)
        corr_i = mean(guide * guide)
        corr_ip = mean(guide * src)
        var_i = corr_i - mean_i * mean_i
        cov_ip = corr_ip - mean_i * mean_p
        a = cov_ip / (var_i + eps)
        b = mean_p - a * mean_i
        return mean(a) * guide + mean(b)

    @classmethod
    def _bilateral_filter_t(cls, src: torch.Tensor, radius: int, sigma_color: float, sigma_space: float,
                            guide: torch.Tensor | None = None) -> torch.Tensor:
        """Bilateral filter; with ``guide``, a joint (cross) bilateral whose
        range weights come from the guide, so chroma edges follow luma edges."""
        radius = max(1, min(int(radius), 8))
        sigma_color = max(float(sigma_color), 1e-6)
        sigma_space = max(float(sigma_space), 1e-6)
        padded = F.pad(src, (radius, radius, radius, radius), mode="reflect")
        g = guide if guide is not None else src
        gpad = F.pad(g, (radius, radius, radius, radius), mode="reflect") if guide is not None else padded
        acc = torch.zeros_like(src)
        wsum = torch.zeros_like(src)
        center = g
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                shifted = padded[
                    :,
                    :,
                    radius + dy : radius + dy + src.shape[-2],
                    radius + dx : radius + dx + src.shape[-1],
                ]
                gshift = gpad[
                    :,
                    :,
                    radius + dy : radius + dy + src.shape[-2],
                    radius + dx : radius + dx + src.shape[-1],
                ]
                spatial = math.exp(-float(dx * dx + dy * dy) / (2.0 * sigma_space * sigma_space))
                range_w = torch.exp(-((gshift - center) ** 2) / (2.0 * sigma_color * sigma_color))
                weight = range_w * spatial
                acc = acc + shifted * weight
                wsum = wsum + weight
        return acc / wsum.clamp(min=1e-8)

    @classmethod
    def _denoise_band_t(
        cls,
        band_bchw: torch.Tensor,
        strength: float,
        filter_type: str,
        radius: int,
        sigma_color: float,
        sigma_space: float,
        guidance_bchw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if strength <= 0.0:
            return band_bchw
        if filter_type == "Guided":
            guide = guidance_bchw if guidance_bchw is not None else band_bchw
            filtered = cls._guided_filter_t(guide, band_bchw, max(1, radius),
                                             max(1e-6, sigma_color * sigma_color), sigma_space)
        else:
            # The luma guide used to reach only the Guided filter, so with the
            # default Bilateral joint_chroma_guidance did nothing.
            filtered = cls._bilateral_filter_t(band_bchw, max(1, radius), sigma_color, sigma_space,
                                               guidance_bchw)
        return band_bchw + float(strength) * (filtered - band_bchw)

    @staticmethod
    def _noise_profile_t(luma: torch.Tensor, patch_size: int = 32) -> float:
        # luma: (B,1,H,W)
        B, _, H, W = luma.shape
        if H < patch_size or W < patch_size:
            return float(luma.std().clamp(min=0.015).item())
        patches = F.unfold(luma, kernel_size=patch_size, stride=patch_size)
        if patches.numel() == 0:
            return 0.015
        stds = patches.std(dim=1)
        valid = stds[stds > 1e-4]
        if valid.numel() == 0:
            return 0.015
        return float(valid.min().item())

    #: Motion search: offsets tried at the coarsest pyramid level and at each
    #: finer one, block size, pyramid depth. +-4 at 1/8 scale locks onto
    #: motion of about +-36 px; each finer level refines by +-2.
    _MC_COARSE_SEARCH = 4
    _MC_SEARCH = 2
    _MC_BLOCK = 8
    _MC_LEVELS = 3
    _MC_MIN_SIDE = 16

    @staticmethod
    def _mc_warp(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """Sample `img` (1, C, H, W) at p + flow(p); flow (1, 2, H, W) is (dy, dx)
        in pixels. Edges repeat."""
        _, _, H, W = img.shape
        ys = torch.arange(H, device=img.device, dtype=img.dtype).view(H, 1).expand(H, W)
        xs = torch.arange(W, device=img.device, dtype=img.dtype).view(1, W).expand(H, W)
        gy = (ys + flow[0, 0]) * (2.0 / max(H - 1, 1)) - 1.0
        gx = (xs + flow[0, 1]) * (2.0 / max(W - 1, 1)) - 1.0
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)
        return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=True)

    @staticmethod
    def _mc_shift(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """_mc_warp for a whole-pixel flow, as a gather: (1, 1, H, W) only."""
        _, _, H, W = img.shape
        fy, fx = flow[0, 0].long(), flow[0, 1].long()
        ys = (torch.arange(H, device=img.device).view(H, 1) + fy).clamp_(0, H - 1)
        xs = (torch.arange(W, device=img.device).view(1, W) + fx).clamp_(0, W - 1)
        return img.reshape(-1)[ys * W + xs].view(1, 1, H, W)

    @staticmethod
    def _mc_per_pixel(block_flow: torch.Tensor, block: int, h: int, w: int) -> torch.Tensor:
        return block_flow.repeat_interleave(block, 2).repeat_interleave(block, 3)[..., :h, :w]

    @classmethod
    def _motion_compensate_t(cls, frame: torch.Tensor, neighbor: torch.Tensor) -> torch.Tensor:
        """Warp `neighbor` (H, W, C) onto `frame` (H, W, C).

        3.x picked the best of the nine 1-pixel offsets per pixel, so only
        motion of about a pixel per frame was followed. Since 4.0 this is
        hierarchical block matching: one motion vector per 8x8 block, found on
        a pyramid from coarse to fine. The coarsest level tries every vector
        within +-4, each finer level those within +-2 of the coarser estimate,
        by the mean absolute difference over the block (no move wins ties). The neighbour is then warped with edge
        repeat, so nothing wraps around from the far side as torch.roll did.
        """
        f = frame[..., :3].float().mean(-1)[None, None]
        n = neighbor[..., :3].float().mean(-1)[None, None]
        pyramid = [(f, n)]
        while (len(pyramid) <= cls._MC_LEVELS
               and min(pyramid[-1][0].shape[-2:]) >= 2 * cls._MC_MIN_SIDE):
            pf, pn = pyramid[-1]
            pyramid.append((F.avg_pool2d(pf, 2), F.avg_pool2d(pn, 2)))

        b = cls._MC_BLOCK
        block_flow = None
        for lf, ln in reversed(pyramid):
            h, w = lf.shape[-2:]
            hb, wb = -(-h // b), -(-w // b)
            if block_flow is None:
                block_flow = lf.new_zeros(1, 2, hb, wb)
                r = cls._MC_COARSE_SEARCH
            else:
                # One level up is half the size: vectors double.
                block_flow = F.interpolate(block_flow, size=(hb, wb), mode="nearest") * 2.0
                r = cls._MC_SEARCH
            # The flow stays whole pixels, so trials are gathers, not resamples.
            base = cls._mc_per_pixel(block_flow, b, h, w)
            offsets = [(0, 0)] + [(dy, dx) for dy in range(-r, r + 1)
                                  for dx in range(-r, r + 1) if (dy, dx) != (0, 0)]
            best_cost = best = None
            for dy, dx in offsets:
                off = base.new_tensor([dy, dx]).view(1, 2, 1, 1)
                diff = (lf - cls._mc_shift(ln, base + off)).abs()
                cost = F.avg_pool2d(diff, b, stride=b, ceil_mode=True)
                trial = block_flow + off
                if best_cost is None:
                    best_cost, best = cost, trial
                    continue
                better = cost < best_cost
                best_cost = torch.where(better, cost, best_cost)
                best = torch.where(better, trial, best)
            block_flow = best

        H, W = frame.shape[:2]
        flow = cls._mc_per_pixel(block_flow, b, H, W)
        src = neighbor.float().permute(2, 0, 1).unsqueeze(0)
        return cls._mc_warp(src, flow)[0].permute(1, 2, 0).to(neighbor.dtype)

    def _denoise_gpu(
        self,
        image: torch.Tensor,
        d: int,
        sigmaColor: float,
        sigmaSpace: float,
        hdr_auto_sigma: bool,
        filter_type: str,
        auto_profiling: bool,
        profile_multiplier: float,
        luma_strength: float,
        chroma_strength: float,
        high_freq_denoise: float,
        mid_freq_denoise: float,
        low_freq_denoise: float,
        joint_chroma_guidance: bool,
        temporal_blend: float,
        temporal_radius: int,
        temporal_threshold: float,
        motion_compensation: bool,
        detail_recovery: float,
        sharpen_strength: float,
        view_mode: str,
    ):
        img = image.float()
        original = img
        B, H, W, C = img.shape

        if B > 1 and temporal_blend > 0.0:
            temporal = []
            for i in range(B):
                frame = img[i]
                blended = frame.clone()
                total = torch.ones_like(frame[..., :3] if C >= 3 else frame)
                neighbors = []
                for dist in range(1, temporal_radius + 1):
                    if i - dist >= 0:
                        neighbors.append((img[i - dist], dist))
                    if i + dist < B:
                        neighbors.append((img[i + dist], dist))
                for neighbor, dist in neighbors:
                    aligned = self._motion_compensate_t(frame, neighbor) if motion_compensation else neighbor
                    active_frame = frame[..., :3] if C >= 3 else frame
                    active_neighbor = aligned[..., :3] if C >= 3 else aligned
                    diff = (active_frame - active_neighbor).abs()
                    weight = torch.exp(-((diff / max(temporal_threshold, 1e-6)) ** 2))
                    weight = weight * (temporal_blend / max(len(neighbors), 1) / float(dist))
                    if C >= 3:
                        blended_rgb = blended[..., :3] + weight * active_neighbor
                        blended = torch.cat([blended_rgb, blended[..., 3:]], dim=-1) if C == 4 else blended_rgb
                    else:
                        blended = blended + weight * aligned
                    total = total + weight
                if neighbors:
                    if C >= 3:
                        rgb = blended[..., :3] / total.clamp(min=1e-8)
                        blended = torch.cat([rgb, blended[..., 3:]], dim=-1) if C == 4 else rgb
                    else:
                        blended = blended / total.clamp(min=1e-8)
                temporal.append(blended)
            img = torch.stack(temporal, dim=0)

        if C >= 3:
            ycrcb = self._rgb_to_ycrcb_t(img[..., :3])
            y = ycrcb[..., 0].unsqueeze(1)
            cr = ycrcb[..., 1].unsqueeze(1)
            cb = ycrcb[..., 2].unsqueeze(1)
            frame_max = float(img[..., :3].amax().item())
            base_sigma = self._noise_profile_t(y) * profile_multiplier if auto_profiling else sigmaColor
            if hdr_auto_sigma and frame_max > 1.01:
                base_sigma *= frame_max

            r_mid = max(1, d // 2)
            r_low = max(2, int(d * 1.5))

            def _bands(ch: torch.Tensor):
                low = self._box_blur(ch, r_low)
                mid_raw = self._box_blur(ch, r_mid)
                return low, mid_raw - low, ch - mid_raw

            y_low, y_mid, y_high = _bands(y)
            y_high_d = self._denoise_band_t(y_high, high_freq_denoise, filter_type, max(1, d // 4), base_sigma * luma_strength, sigmaSpace)
            y_mid_d = self._denoise_band_t(y_mid, mid_freq_denoise, filter_type, max(1, d // 2), base_sigma * luma_strength, sigmaSpace)
            y_low_d = self._denoise_band_t(y_low, low_freq_denoise, filter_type, max(1, d), base_sigma * luma_strength, sigmaSpace)
            denoised_y = y_high_d + y_mid_d + y_low_d

            guide_high = y_high_d if joint_chroma_guidance else None
            guide_mid = y_mid_d if joint_chroma_guidance else None
            guide_low = y_low_d if joint_chroma_guidance else None

            chroma_out = []
            for chroma in (cr, cb):
                c_low, c_mid, c_high = _bands(chroma)
                c_high_d = self._denoise_band_t(c_high, high_freq_denoise, filter_type, max(1, d // 4), base_sigma * chroma_strength, sigmaSpace, guide_high)
                c_mid_d = self._denoise_band_t(c_mid, mid_freq_denoise, filter_type, max(1, d // 2), base_sigma * chroma_strength, sigmaSpace, guide_mid)
                c_low_d = self._denoise_band_t(c_low, low_freq_denoise, filter_type, max(1, d), base_sigma * chroma_strength, sigmaSpace, guide_low)
                chroma_out.append(c_high_d + c_mid_d + c_low_d)

            denoised_ycrcb = torch.cat([denoised_y, chroma_out[0], chroma_out[1]], dim=1).permute(0, 2, 3, 1)
            denoised_rgb = self._ycrcb_to_rgb_t(denoised_ycrcb)
            denoised = torch.cat([denoised_rgb, img[..., 3:]], dim=-1) if C == 4 else denoised_rgb
        else:
            gray = img.permute(0, 3, 1, 2)
            frame_max = float(gray.amax().item())
            base_sigma = self._noise_profile_t(gray) * profile_multiplier if auto_profiling else sigmaColor
            if hdr_auto_sigma and frame_max > 1.01:
                base_sigma *= frame_max
            r_mid = max(1, d // 2)
            r_low = max(2, int(d * 1.5))
            low = self._box_blur(gray, r_low)
            mid_raw = self._box_blur(gray, r_mid)
            mid = mid_raw - low
            high = gray - mid_raw
            denoised = (
                self._denoise_band_t(high, high_freq_denoise, filter_type, max(1, d // 4), base_sigma * luma_strength, sigmaSpace)
                + self._denoise_band_t(mid, mid_freq_denoise, filter_type, max(1, d // 2), base_sigma * luma_strength, sigmaSpace)
                + self._denoise_band_t(low, low_freq_denoise, filter_type, max(1, d), base_sigma * luma_strength, sigmaSpace)
            ).permute(0, 2, 3, 1)

        if detail_recovery > 0.0:
            residual = (original - denoised).permute(0, 3, 1, 2)
            detail = residual - self._box_blur(residual, max(1, d // 2))
            denoised = denoised + detail_recovery * detail.permute(0, 2, 3, 1)

        if sharpen_strength > 0.0:
            rgb_or_gray = denoised[..., :3] if C >= 3 else denoised
            blurred = self._box_blur(rgb_or_gray.permute(0, 3, 1, 2), 2).permute(0, 2, 3, 1)
            sharp = rgb_or_gray + sharpen_strength * (rgb_or_gray - blurred)
            denoised = torch.cat([sharp, denoised[..., 3:]], dim=-1) if C == 4 else sharp

        if view_mode == "Noise Residual":
            final = original - denoised + 0.5
        elif view_mode == "Luma (Y)" and C >= 3:
            yv = self._rgb_to_ycrcb_t(denoised[..., :3])[..., :1]
            final = yv.expand(-1, -1, -1, 3)
            if C == 4:
                final = torch.cat([final, original[..., 3:]], dim=-1)
        elif view_mode == "Chroma (Cb/Cr)" and C >= 3:
            yc = self._rgb_to_ycrcb_t(denoised[..., :3])
            chroma = torch.cat([torch.full_like(yc[..., :1], 0.5), yc[..., 1:2], yc[..., 2:3]], dim=-1)
            final = self._ycrcb_to_rgb_t(chroma)
            if C == 4:
                final = torch.cat([final, original[..., 3:]], dim=-1)
        else:
            final = denoised

        return (final,)

    def denoise(
        self,
        image,
        d,
        sigmaColor,
        sigmaSpace,
        hdr_auto_sigma=True,
        filter_type="Bilateral",
        auto_profiling=False,
        profile_multiplier=1.0,
        luma_strength=1.0,
        chroma_strength=1.0,
        high_freq_denoise=1.0,
        mid_freq_denoise=1.0,
        low_freq_denoise=0.5,
        joint_chroma_guidance=True,
        temporal_blend=0.0,
        temporal_radius=1,
        temporal_threshold=0.05,
        motion_compensation=True,
        detail_recovery=0.0,
        sharpen_strength=0.0,
        view_mode="Denoised"
    ):
        return self._denoise_gpu(
            image, d, sigmaColor, sigmaSpace, hdr_auto_sigma, filter_type,
            auto_profiling, profile_multiplier, luma_strength, chroma_strength,
            high_freq_denoise, mid_freq_denoise, low_freq_denoise,
            joint_chroma_guidance, temporal_blend, temporal_radius,
            temporal_threshold, motion_compensation, detail_recovery,
            sharpen_strength, view_mode,
        )


NODE_CLASS_MAPPINGS = {"RadianceDenoise": RadianceDenoise}

NODE_DISPLAY_NAME_MAPPINGS = {"RadianceDenoise": "◎ Radiance 32-bit Denoise"}
