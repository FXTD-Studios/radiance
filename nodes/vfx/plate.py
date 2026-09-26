import torch
import torch.nn.functional as F
import numpy as np
import logging

from radiance.core.tensor.chunking import FrameSink, chunks, compute_device, frames_per_chunk

logger = logging.getLogger("radiance.vfx.plate")

class RadianceHDRGrainMatcher:
    """
    ◎ Radiance HDR Grain Matcher
    
    Extracts high-frequency film grain from a reference plate in log2 exposure space
    and maps it matching the target image's exposure range to prevent highlight burnout.
    """
    
    DESCRIPTION = "Transfer grain from a reference plate onto a clean image: the reference's fine detail is high-pass filtered in log2 space and added to the target in log2, so it scales with exposure and does not burn out highlights."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target": ("IMAGE", {"tooltip": "Clean RGB image or sequence to receive grain, ideally scene-linear. Negative values are clamped to 0."}),
                "reference": ("IMAGE", {"tooltip": "Grainy plate to take grain from. Must match the target's width and height; frames are reused in a cycle if it is shorter than the target. All high-frequency detail is transferred, so use a flat, defocused area if possible."}),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05, "tooltip": "Multiplier on the extracted grain. 0 = no grain, 1 = matched to the reference."}),
                "kernel_size": ("INT", {"default": 3, "min": 1, "max": 15, "step": 2, "tooltip": "Box-blur size in pixels used to split grain from the image. Larger values capture coarser grain (and more image detail); 1 extracts nothing."}),
                "r_gain": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "Extra grain multiplier for the red channel. 1.0 = unchanged."}),
                "g_gain": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "Extra grain multiplier for the green channel. 1.0 = unchanged."}),
                "b_gain": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "Extra grain multiplier for the blue channel. 1.0 = unchanged."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("grained_image",)
    FUNCTION = "apply"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ VFX/Plate Prep"

    def apply(self, target: torch.Tensor, reference: torch.Tensor, intensity: float, kernel_size: int, r_gain: float, g_gain: float, b_gain: float):
        B, H, W, C = target.shape
        ref_B = reference.shape[0]
        # log2 prevents grain extraction from being influenced by absolute light levels
        eps = 1e-4
        pad = kernel_size // 2

        # 3.5.0: each reference frame's grain is computed once (it used to be
        # recomputed for every target frame: 240 times for a one-frame
        # reference and a 240-frame clip), and targets are grained a few frames
        # at a time on the GPU instead of all at once on the CPU (about 6x the
        # clip in memory). Frames cycle through the reference as before.
        device = compute_device()
        gains = torch.tensor([r_gain, g_gain, b_gain], device=device).view(1, 1, 1, 3)
        grain_cache = {}

        def grain_of(idx: int) -> torch.Tensor:
            if idx not in grain_cache:
                ref_frame = reference[idx].to(device, torch.float32).permute(2, 0, 1).unsqueeze(0)  # 1, C, H, W
                ref_log = torch.log2(ref_frame.clamp(min=0.0) + eps)
                ref_smooth = F.avg_pool2d(ref_log, kernel_size, stride=1, padding=pad)   # box-filter smooth
                grain_cache[idx] = (ref_log - ref_smooth).permute(0, 2, 3, 1)           # high-frequency grain
            return grain_cache[idx]

        out = FrameSink((B, H, W, C))
        per = frames_per_chunk(H, W, C, 5.0, device)
        for a, b in chunks(B, per):
            needed = [i % ref_B for i in range(a, b)]
            # Keep only the grains this chunk uses (all of them when the
            # reference is shorter than a chunk).
            for k in [k for k in grain_cache if k not in needed]:
                del grain_cache[k]
            grain_tensor = torch.cat([grain_of(k) for k in needed], dim=0)             # n, H, W, C
            # In place where the values allow it, so the chunk holds about two
            # frame-sized buffers instead of six.
            target_log = target[a:b].to(device, torch.float32).clamp(min=0.0).add_(eps).log2_()
            grained_log = grain_tensor.mul_(intensity).mul_(gains).add_(target_log)
            del target_log
            # Convert back from log2 space
            out.put(a, b, torch.pow(2.0, grained_log).sub_(eps).clamp_(min=0.0))
            del grain_tensor, grained_log

        logger.info(f"[HDR Grain Matcher] Extracted and re-applied grain (Intensity: {intensity})")
        return (out.value,)


class RadianceSubpixelStabilizer:
    """
    ◎ Radiance Subpixel Stabilizer
    
    Stabilizes an image sequence batch to a selected reference anchor frame.
    Calculates displacements using high-precision sub-pixel FFT Phase Correlation in pure PyTorch.
    """
    
    DESCRIPTION = "Stabilise an image sequence to one anchor frame using sub-pixel FFT phase correlation. Corrects translation only (no rotation or scale); also outputs the per-frame shift as an image (red = x, green = y, in pixels)."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image sequence to stabilise. Shifts are measured on the channel average of each frame."}),
                "anchor_frame": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1, "tooltip": "Zero-based index of the frame every other frame is aligned to. Values past the end use the last frame."}),
                "max_shift": ("INT", {"default": 64, "min": 4, "max": 512, "step": 4, "tooltip": "Largest correction applied, in pixels per axis. A larger measured shift is clamped to this rather than rejected."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("stabilized_sequence", "displacements_xy")
    FUNCTION = "apply"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ VFX/Plate Prep"

    def apply(self, image: torch.Tensor, anchor_frame: int, max_shift: int):
        B, H, W, C = image.shape
        # 3.5.0: frames go through the FFT, the peak search and the warp a few
        # at a time on the GPU. This used to be one frame at a time with seven
        # blocking .item() reads each, on the CPU. The sub-pixel step is the
        # same double-precision arithmetic on the same five samples, and ties
        # resolve to the first maximum in row-major order as before.
        device = compute_device()
        anchor_idx = min(anchor_frame, B - 1)

        # Hann window to minimise FFT boundary leakage
        hann_y = torch.hann_window(H, device=device).unsqueeze(1)
        hann_x = torch.hann_window(W, device=device).unsqueeze(0)
        window = hann_y * hann_x
        # Reference anchor frame, greyscale for correlation
        ref_frame = image[anchor_idx].to(device, torch.float32).mean(dim=-1)
        F_ref = torch.fft.fft2(ref_frame * window)

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing="ij",
        )

        stabilized = FrameSink((B, H, W, C))
        shifts = torch.zeros((B, 3), dtype=torch.float32)       # dx, dy, 0 per frame
        per = frames_per_chunk(H, W, C, 8.0, device)
        for a, b in chunks(B, per):
            frames = image[a:b].to(device, torch.float32)                              # n, H, W, C
            F_cur = torch.fft.fft2(frames.mean(dim=-1) * window)
            # Cross-power spectrum R = (F_ref * conj(F_cur)) / |F_ref * conj(F_cur)|
            cross = F_ref * torch.conj(F_cur)
            r = torch.fft.ifft2(cross / (torch.abs(cross) + 1e-12)).real               # n, H, W
            del F_cur, cross
            n = b - a
            peak = torch.argmax(r.reshape(n, -1), dim=1)                               # first maximum
            py, px = peak // W, peak % W
            # The peak and its four neighbours (wrapped), read back in one go.
            ys = torch.stack([(py - 1) % H, (py + 1) % H, py, py, py], dim=1)
            xs = torch.stack([px, px, px, (px - 1) % W, (px + 1) % W], dim=1)
            vals = r[torch.arange(n, device=device).unsqueeze(1), ys, xs].double().cpu().tolist()
            pys, pxs = py.cpu().tolist(), px.cpu().tolist()
            del r

            dxs, dys = [], []
            for j in range(n):
                i = a + j
                if i == anchor_idx:
                    dxs.append(0.0); dys.append(0.0)
                    continue
                dy_int, dx_int = pys[j], pxs[j]
                # Wrap around coordinates
                if dy_int > H // 2:
                    dy_int -= H
                if dx_int > W // 2:
                    dx_int -= W
                dy_sub, dx_sub = float(dy_int), float(dx_int)
                # Sub-pixel refinement: parabola through the peak and its neighbours
                val_n1_0, val_p1_0, val_0_0, val_0_n1, val_0_p1 = vals[j]
                denom_y = val_n1_0 + val_p1_0 - 2 * val_0_0
                if abs(denom_y) > 1e-5:
                    dy_sub += (val_n1_0 - val_p1_0) / (2 * denom_y)
                denom_x = val_0_n1 + val_0_p1 - 2 * val_0_0
                if abs(denom_x) > 1e-5:
                    dx_sub += (val_0_n1 - val_0_p1) / (2 * denom_x)
                # Clamp maximum shifts to prevent wild drift on noise
                dxs.append(max(min(dx_sub, float(max_shift)), -float(max_shift)))
                dys.append(max(min(dy_sub, float(max_shift)), -float(max_shift)))

            # Translate each frame with grid_sample ([-1, 1] grid, shift in pixels / half size)
            dx_t = torch.tensor(dxs, device=device, dtype=torch.float32).view(n, 1, 1)
            dy_t = torch.tensor(dys, device=device, dtype=torch.float32).view(n, 1, 1)
            grid = torch.stack([grid_x - dx_t / (W / 2.0), grid_y - dy_t / (H / 2.0)], dim=-1)
            warp = F.grid_sample(frames.permute(0, 3, 1, 2), grid, mode="bicubic",
                                 padding_mode="border", align_corners=True).permute(0, 2, 3, 1)
            if a <= anchor_idx < b:
                warp[anchor_idx - a] = frames[anchor_idx - a]      # the anchor is passed through
            stabilized.put(a, b, warp)
            shifts[a:b, 0] = torch.tensor(dxs)
            shifts[a:b, 1] = torch.tensor(dys)
            del frames, grid, warp

        # Displacement map (diagnostic): red = x, green = y in pixels, blue 0.
        # 3 channels, as a ComfyUI IMAGE must be.
        displacements = shifts.view(B, 1, 1, 3).expand(B, H, W, 3).contiguous()

        logger.info(f"[Subpixel Stabilizer] Anchored sequence to frame {anchor_idx}.")
        return (stabilized.value, displacements)
