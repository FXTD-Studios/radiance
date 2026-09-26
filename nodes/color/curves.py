import torch
import logging
import json
from typing import Tuple

logger = logging.getLogger("radiance.curves")


def _rgb_to_hsl(img: torch.Tensor) -> torch.Tensor:
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    cmax = torch.max(img, dim=-1).values
    cmin = torch.min(img, dim=-1).values
    delta = cmax - cmin
    L = (cmax + cmin) * 0.5
    eps = 1e-7
    S = torch.where(delta < eps, torch.zeros_like(delta),
                    delta / (1.0 - torch.abs(2.0 * L - 1.0).clamp(max=1.0 - eps) + eps))
    H = torch.zeros_like(delta)
    mask_r = (cmax == r) & (delta > eps)
    mask_g = (cmax == g) & (delta > eps)
    mask_b = (cmax == b) & (delta > eps)
    H[mask_r] = ((g[mask_r] - b[mask_r]) / (delta[mask_r] + eps)) % 6.0
    H[mask_g] = (b[mask_g] - r[mask_g]) / (delta[mask_g] + eps) + 2.0
    H[mask_b] = (r[mask_b] - g[mask_b]) / (delta[mask_b] + eps) + 4.0
    H = (H / 6.0) % 1.0
    return torch.stack([H, S, L], dim=-1)


def _hsl_to_rgb_fast(hsl: torch.Tensor) -> torch.Tensor:
    H, S, L = hsl[..., 0:1], hsl[..., 1:2], hsl[..., 2:3]
    kR = torch.zeros_like(H)
    kG = torch.full_like(H, 8.0)
    kB = torch.full_like(H, 4.0)

    def _chan(k):
        t = (k + H * 12.0) % 12.0
        a = S * torch.clamp(torch.min(L, 1.0 - L), 0.0, 1.0)
        return L - a * torch.clamp(torch.min(t - 3.0, 9.0 - t).clamp(-1.0, 1.0), -1.0, 1.0)

    return torch.cat([_chan(kR), _chan(kG), _chan(kB)], dim=-1)


def _apply_curve_1d(x: torch.Tensor, ctrl_x: list, ctrl_y: list) -> torch.Tensor:
    if len(ctrl_x) < 2:
        return x
    xs = torch.tensor(ctrl_x, dtype=x.dtype, device=x.device)
    ys = torch.tensor(ctrl_y, dtype=x.dtype, device=x.device)
    result = torch.empty_like(x)
    slope0 = (ys[1] - ys[0]) / (xs[1] - xs[0] + 1e-9)
    result = ys[0] + slope0 * (x - xs[0])
    for i in range(len(xs) - 1):
        x0, x1 = xs[i], xs[i + 1]
        y0, y1 = ys[i], ys[i + 1]
        slope = (y1 - y0) / (x1 - x0 + 1e-9)
        mask = (x >= x0) & (x < x1)
        result = torch.where(mask, y0 + slope * (x - x0), result)
    slope_end = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2] + 1e-9)
    mask_end = x >= xs[-1]
    result = torch.where(mask_end, ys[-1] + slope_end * (x - xs[-1]), result)
    return result


class RadianceHueCurves:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Color"
    DESCRIPTION = "Per-hue selective colour adjustment: a piecewise-linear curve over hue shifts hue, saturation or lightness of the colours it targets. HDR values above 1.0 are scaled down for the HSL maths and restored afterwards."
    FUNCTION = "apply"
    # 3.5.0: grade_info output appended (the input used to be dropped), so a
    # chain of grading nodes keeps its record through HueCorrect. Appending
    # keeps saved links to `image` intact.
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "grade_info")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image to adjust, RGB in any encoding. Pixels with luma above 1.0 are normalised for the HSL conversion and rescaled afterwards."}),
                "mode": (["Hue vs Hue", "Hue vs Saturation", "Hue vs Luminance"], {"default": "Hue vs Hue", "tooltip": "What the curve's output changes for each input hue: hue rotation, HSL saturation, or HSL lightness."}),
                "control_points": ("STRING", {
                    "default": "[[0.0,0.0],[0.167,0.0],[0.333,0.0],[0.5,0.0],[0.667,0.0],[0.833,0.0],[1.0,0.0]]",
                    "multiline": False,
                    "tooltip": "JSON list of [hue, amount] points, hue 0 to 1 (0 red, 0.333 green, 0.667 blue, 1 red again). Amount is added: hue shift in turns, or a saturation or lightness delta; 0 everywhere (the default) changes nothing.",
                }),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "Multiplier on the curve's amounts. 0 = no change, 1 = as drawn, 2 = doubled."}),
            },
            "optional": {
                "grade_info": ("STRING", {"forceInput": True, "tooltip": "Upstream grade metadata. Not used for the adjustment; stored as the 'upstream' field of the grade_info output."}),
            },
        }

    def apply(self, image: torch.Tensor, mode: str, control_points: str,
              strength: float = 1.0, grade_info: str = None):
        img = image.clone()
        luma = img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722
        hdr_scale = luma.clamp(min=1.0)
        safe_img = img / hdr_scale.unsqueeze(-1).clamp(min=1.0)

        # 3.5.0: a curve that does not parse is an error. It used to fall back
        # to a flat curve, so a typo gave an unchanged image and no message.
        try:
            pts = json.loads(control_points)
            if isinstance(pts[0], (int, float)):
                pts = [[pts[i], pts[i + 1]] for i in range(0, len(pts) - 1, 2)]
            pts = sorted(([float(p[0]), float(p[1])] for p in pts), key=lambda p: p[0])
            ctrl_x = [p[0] for p in pts]
            ctrl_y = [p[1] for p in pts]
            if len(ctrl_x) < 2:
                raise ValueError("at least two points are needed")
        except Exception as exc:
            raise ValueError(
                f"HueCorrect: control_points is not a list of [hue, amount] points "
                f"({exc}). Example: [[0.0,0.0],[0.333,0.1],[1.0,0.0]]") from exc

        hsl = _rgb_to_hsl(safe_img)
        if mode == "Hue vs Hue":
            delta = _apply_curve_1d(hsl[..., 0], ctrl_x, ctrl_y) * strength
            hsl = torch.stack([(hsl[..., 0] + delta) % 1.0, hsl[..., 1], hsl[..., 2]], dim=-1)
        elif mode == "Hue vs Saturation":
            delta = _apply_curve_1d(hsl[..., 0], ctrl_x, ctrl_y) * strength
            hsl = torch.stack([hsl[..., 0], (hsl[..., 1] + delta).clamp(0.0, 1.0), hsl[..., 2]], dim=-1)
        elif mode == "Hue vs Luminance":
            delta = _apply_curve_1d(hsl[..., 0], ctrl_x, ctrl_y) * strength
            hsl = torch.stack([hsl[..., 0], hsl[..., 1], (hsl[..., 2] + delta).clamp(0.0, 1.0)], dim=-1)

        graded = _hsl_to_rgb_fast(hsl)
        graded = graded * hdr_scale.unsqueeze(-1)
        info = json.dumps({
            "node": "RadianceHueCurves", "mode": mode, "control_points": pts,
            "strength": strength, "upstream": grade_info,
        })
        return (graded, info)


class RadianceCurves:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Color"
    DESCRIPTION = "RGB curve grading with piecewise-linear control points: a master curve on all three channels, then per-channel red, green and blue curves. Values beyond the end points are extrapolated along the last segment, so HDR values are not clipped."
    FUNCTION = "apply"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "grade_info")

    @classmethod
    def INPUT_TYPES(cls):
        _default_pts = "[[0.0,0.0],[0.25,0.25],[0.5,0.5],[0.75,0.75],[1.0,1.0]]"
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image to grade. Curves act on the values as they arrive, so point positions are in that encoding (0 to 1 for display-referred sRGB)."}),
                "master": ("STRING", {"default": _default_pts, "multiline": False, "tooltip": "JSON list of [in, out] points applied to R, G and B before the per-channel curves. The default diagonal is identity."}),
                "red": ("STRING", {"default": _default_pts, "multiline": False, "tooltip": "JSON list of [in, out] points for the red channel, applied after the master curve. The default diagonal is identity."}),
                "green": ("STRING", {"default": _default_pts, "multiline": False, "tooltip": "JSON list of [in, out] points for the green channel, applied after the master curve. The default diagonal is identity."}),
                "blue": ("STRING", {"default": _default_pts, "multiline": False, "tooltip": "JSON list of [in, out] points for the blue channel, applied after the master curve. The default diagonal is identity."}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Mix between the input (0) and the fully curved result (1)."}),
            },
            "optional": {
                "grade_info_in": ("STRING", {"forceInput": True, "tooltip": "Upstream grade metadata. Not used for the grade; it is stored as the 'upstream' field of the grade_info output."}),
            },
        }

    @staticmethod
    def _parse_pts(s: str):
        try:
            pts = json.loads(s)
            if isinstance(pts[0], (int, float)):
                pts = [[pts[i], pts[i + 1]] for i in range(0, len(pts) - 1, 2)]
            xs = [float(p[0]) for p in pts]
            ys = [float(p[1]) for p in pts]
            return xs, ys
        except Exception:
            return [0.0, 1.0], [0.0, 1.0]

    def apply(self, image: torch.Tensor, master: str, red: str, green: str,
              blue: str, strength: float = 1.0, grade_info_in: str = None):
        img = image.clone()
        mx, my = self._parse_pts(master)
        rx, ry = self._parse_pts(red)
        gx, gy = self._parse_pts(green)
        bx, by = self._parse_pts(blue)

        def _is_identity(xs, ys):
            return all(abs(x - y) < 1e-6 for x, y in zip(xs, ys))

        def _apply(chan, xs, ys):
            if _is_identity(xs, ys):
                return chan
            return _apply_curve_1d(chan, xs, ys)

        if not _is_identity(mx, my):
            img = _apply_curve_1d(img, mx, my)
        r = _apply(img[..., 0], rx, ry)
        g = _apply(img[..., 1], gx, gy)
        b = _apply(img[..., 2], bx, by)
        result = torch.stack([r, g, b], dim=-1)

        if strength < 1.0:
            result = torch.lerp(image, result, strength)

        grade_info = json.dumps({
            "node": "RadianceCurves", "strength": strength,
            "curves": {"master": {"x": mx, "y": my}, "red": {"x": rx, "y": ry},
                       "green": {"x": gx, "y": gy}, "blue": {"x": bx, "y": by}},
            "upstream": grade_info_in,
        })
        return (result, grade_info)
