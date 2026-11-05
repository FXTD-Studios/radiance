"""RUDRA ACES 2: the learned ACES 2.0 SDR -> scene-linear reconstruction.

Port of RUDRA 1.0.1's staged contract (``rudra_native_aces_v2_staged``, major
2; ``tools/export_native_v2.py``, ``native/core/src/aces_v2.cpp``). The
network stage ("G1") is one TorchScript module; everything colour-managed is
OpenColorIO's built-in ACES 2.0 CG config, exactly as the checkpoint pins it:

    1. baseline  = inverse_view(source)                     OCIO, CPU
    2. G1(source, baseline) -> restored_source, highlight_fields, full_fields, weight
    3. restored  = inverse_view(restored_source)            OCIO, CPU
    4. compose                                              per-pixel arithmetic
    5. clipped smoothing (hybrid64)                         fully clipped interiors flat
    6. appearance constraint                                ACES 2.0 SDR view keeps every code within 1
    7. mask = weight where (weight > 0 and max source > 0.5) else 0

``source`` is display RGB codes in [0, 1], Rec.709 primaries, Rec.1886 (what an
8-bit SDR file holds). The result is an ACEScg master, signed and unclamped,
whose ACES 2.0 SDR 100-nit view reproduces the source codes; scene-linear 1.0
is diffuse white (203 nits), RUDRA's convention and Radiance's
``reference_white_nits`` default. ``mask`` is ``rudra.reconstructed``: where
radiance was estimated rather than taken from the source.

Stages 5 and 6 are ported line for line from RUDRA's NumPy reference (float64
where it is float64) so a master made here matches one made by the RUDRA app.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger("radiance.model.rudra_aces2")

CONTRACT = "rudra_native_aces_v2_staged"
CONTRACT_MAJOR = 2
OCIO_CONFIG = "cg-config-v4.0.0_aces-v2.0_ocio-v2.5"
OCIO_DISPLAY = "Rec.1886 Rec.709 - Display"
OCIO_VIEW = "ACES 2.0 - SDR 100 nits (Rec.709)"
OCIO_CONFIG_SHA256 = "24ec81841048fc5db160a7bad882263246183385c5d49d0e86e11464917ead15"
PACKAGE_ENV = "RADIANCE_RUDRA_ACES2"
DIFFUSE_WHITE_NITS = 203.0

LUMA_AP1 = np.array([.2722287168, .6740817658, .0536895174], np.float64)
FULL_CLIP = 254.5 / 255.0

# ACEScg (AP1) <-> ACES2065-1 (AP0) and linear Rec.709, the CG config's matrices.
M_AP1_TO_AP0 = np.array([[0.6954522414, 0.1406786965, 0.1638690622],
                         [0.0447945634, 0.8596711185, 0.0955343182],
                         [-0.0055258826, 0.0040252103, 1.0015006723]], np.float64)
M_AP1_TO_REC709 = np.array([[1.7050509927, -0.6217921207, -0.0832588720],
                            [-0.1302564175, 1.1408047366, -0.0105483191],
                            [-0.0240033568, -0.1289689761, 1.1529723329]], np.float64)
M_REC709_TO_AP1 = np.linalg.inv(M_AP1_TO_REC709)


# ─────────────────────────────────────────────────────────────────────────────
#  OCIO view
# ─────────────────────────────────────────────────────────────────────────────

class ACES2View:
    """The ACES 2.0 SDR 100-nit Rec.709 view of the built-in CG config (CPU)."""

    def __init__(self, threads: Optional[int] = None):
        import PyOpenColorIO as ocio
        self.threads = max(1, min(16, threads or (os.cpu_count() or 1)))
        config = ocio.Config.CreateFromBuiltinConfig(OCIO_CONFIG)
        transform = ocio.DisplayViewTransform(src="ACEScg", display=OCIO_DISPLAY, view=OCIO_VIEW)
        self._forward = config.getProcessor(transform).getDefaultCPUProcessor()
        transform.setDirection(ocio.TRANSFORM_DIR_INVERSE)
        self._inverse = config.getProcessor(transform).getDefaultCPUProcessor()
        self.config_sha256 = hashlib.sha256(config.serialize().encode()).hexdigest()
        self.identity = {"config": OCIO_CONFIG, "display": OCIO_DISPLAY, "view": OCIO_VIEW,
                         "working_space": "ACEScg", "ocio_version": ocio.__version__,
                         "config_sha256": self.config_sha256}
        if self.config_sha256 != OCIO_CONFIG_SHA256:
            logger.warning("[RUDRA ACES 2] OCIO %s builds %s with a different hash than the model pins "
                           "(%s); results may differ slightly from the RUDRA app.",
                           ocio.__version__, OCIO_CONFIG, OCIO_CONFIG_SHA256[:12])

    def _process(self, processor, out: np.ndarray) -> None:
        if self.threads == 1 or len(out) < 262144:
            processor.applyRGB(out)
        else:
            with ThreadPoolExecutor(self.threads) as pool:
                list(pool.map(processor.applyRGB, np.array_split(out, self.threads)))

    def inverse(self, display_rgb: np.ndarray) -> np.ndarray:
        a = np.asarray(display_rgb, np.float32)
        if np.any((a < 0) | (a > 1)):
            raise ValueError("inverse requires display RGB codes in [0,1]")
        out = np.ascontiguousarray(a).reshape(-1, 3).copy()
        self._process(self._inverse, out)
        return out.reshape(a.shape)

    def forward(self, acescg: np.ndarray) -> np.ndarray:
        a = np.asarray(acescg, np.float32)
        out = np.ascontiguousarray(a).reshape(-1, 3).copy()
        self._process(self._forward, out)
        return out.reshape(a.shape)


_VIEW: Optional[ACES2View] = None


def aces2_view() -> ACES2View:
    global _VIEW
    if _VIEW is None:
        _VIEW = ACES2View()
    return _VIEW


# ─────────────────────────────────────────────────────────────────────────────
#  Package
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ACES2Package:
    folder: Path
    manifest: dict
    g1: torch.jit.ScriptModule
    device: torch.device
    forbid_new_negative_y: bool
    smoothing: Optional[dict]
    identity: dict = field(default_factory=dict)


_PACKAGES: dict = {}


def resolve_aces2_package(explicit: str = "") -> Optional[Path]:
    from radiance.model.rudra_download import ACES2, find_package
    return find_package(ACES2, explicit, PACKAGE_ENV)


def _check_manifest(m: dict) -> None:
    if m.get("contract") != CONTRACT or int(m.get("contract_major", -1)) != CONTRACT_MAJOR:
        raise ValueError(f"not a {CONTRACT} v{CONTRACT_MAJOR} package "
                         f"(contract {m.get('contract')!r} v{m.get('contract_major')})")
    ocio_cfg = m.get("ocio", {})
    if ocio_cfg.get("config") != OCIO_CONFIG or ocio_cfg.get("view") != OCIO_VIEW \
            or ocio_cfg.get("display") != OCIO_DISPLAY:
        raise ValueError(f"package pins a view this build does not implement: {ocio_cfg}")
    sm = m.get("clipped_smoothing")
    if sm is not None and (sm.get("name") != "hybrid64" or float(sm.get("sigma", 0)) != 64.0
                           or float(sm.get("ramp", 0)) != 1.0 or sm.get("mode") != "hybrid"):
        raise ValueError(f"package asks for a clipped smoothing this build does not implement: {sm}")
    ac = m.get("appearance_constraint") or {}
    if ac and (int(ac.get("max_display_error_codes", 1)) != 1 or not ac.get("include_fully_clipped", True)):
        raise ValueError(f"package asks for a constraint this build does not implement: {ac}")


def load_aces2_package(explicit: str = "", device: Optional[torch.device] = None,
                       download: bool = True) -> ACES2Package:
    """Load (and on first use download) the RUDRA ACES 2 package. Raises when it cannot."""
    folder = resolve_aces2_package(explicit)
    if folder is None and download:
        from radiance.model.rudra_download import ACES2, download_package
        folder = download_package(ACES2)
    if folder is None:
        from radiance.model.rudra_download import ACES2
        raise FileNotFoundError(
            f"RUDRA ACES 2 is not installed. Put its files ({', '.join(f for f, _, _ in ACES2.files)}) in "
            f"ComfyUI/models/radiance/{ACES2.name}/ (from {ACES2.page_url}) or set {PACKAGE_ENV}.")
    device = torch.device(device) if device is not None else torch.device("cpu")
    key = (str(folder.resolve()), str(device))
    if key in _PACKAGES:
        return _PACKAGES[key]
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    _check_manifest(manifest)
    g1_path = folder / manifest.get("g1_file", "rudra_g1.ts")
    from radiance.model.rudra_download import sha256_file
    digest = sha256_file(g1_path)
    if manifest.get("g1_sha256") and digest != manifest["g1_sha256"]:
        raise ValueError(f"{g1_path}: sha256 {digest} does not match the manifest's {manifest['g1_sha256']}")
    g1 = torch.jit.load(str(g1_path), map_location=device).eval()
    ac = manifest.get("appearance_constraint") or {}
    pkg = ACES2Package(
        folder=folder, manifest=manifest, g1=g1, device=device,
        forbid_new_negative_y=bool(ac.get("forbid_new_negative_y", False)),
        smoothing=manifest.get("clipped_smoothing"),
        identity={"package": manifest.get("name", folder.name), "title": manifest.get("title", "RUDRA ACES 2"),
                  "model_contract": manifest.get("model_contract", {}).get("id"),
                  "g1_sha256": digest, "checkpoint_sha256": manifest.get("checkpoint_sha256")},
    )
    if len(_PACKAGES) >= 2:
        _PACKAGES.pop(next(iter(_PACKAGES)))
    _PACKAGES[key] = pkg
    logger.info("[RUDRA ACES 2] loaded %s on %s", folder, device)
    return pkg


# ─────────────────────────────────────────────────────────────────────────────
#  Stages 4-7
# ─────────────────────────────────────────────────────────────────────────────

def staged_compose(source, baseline, restored, hl, full, weight):
    """Stage 4, BCHW float32 tensors. Mirrors RUDRA's staged_compose exactly."""
    scale = restored.abs().amax(1, keepdim=True) + .001
    top = source.amax(1, keepdim=True)
    high = (weight > 0) & (top > .5)
    full_support = (weight > 0) & (source.amin(1, keepdim=True) >= FULL_CLIP)
    shadow = (weight > 0) & (top <= .5)
    out = restored + weight * (scale * hl)
    out = torch.where(full_support, restored + weight * (scale * full), out)
    shadow_blend = torch.where(weight >= 1, baseline, weight * baseline + (1 - weight) * restored)
    out = torch.where(shadow, shadow_blend, torch.where(high, out, restored))
    return out, torch.where(high, weight, torch.zeros_like(weight))


def smooth_clipped(pred: np.ndarray, source: np.ndarray, sigma: float = 64.0, ramp: float = 1.0,
                   start: float = 0.0, threshold: float = FULL_CLIP, mode: str = "hybrid"):
    """Stage 5 (RUDRA training/aces_clipped_smoothing.py). Inside fully clipped
    source pixels the SDR holds no spatial information: keep each pixel's
    chroma ratio, replace log2 luminance with a support-normalised Gaussian
    average (flat in each eroded interior), blended in from the boundary.
    Pixels outside the fully clipped support come back bit-exact."""
    import cv2
    pred = np.asarray(pred, np.float64)
    full = np.asarray(source).min(-1) >= threshold
    y = pred @ LUMA_AP1
    ok = full & (y > 1e-10)
    if not ok.any():
        return pred.astype(np.float32), full
    ly = np.where(ok, np.log2(np.maximum(y, 1e-10)), 0.0)
    n = 0
    if mode == "component":
        n, labels = cv2.connectedComponents(full.astype(np.uint8), connectivity=8)
        sums = np.bincount(labels[ok], ly[ok], minlength=n)
        counts = np.bincount(labels[ok], minlength=n)
        means = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
        ls = np.where(ok, means[labels], ly)
    else:
        k = int(4 * sigma) | 1
        num = cv2.GaussianBlur(ly, (k, k), sigma)
        den = cv2.GaussianBlur(ok.astype(np.float64), (k, k), sigma)
        ls = np.where(den > 1e-6, num / np.maximum(den, 1e-6), ly)
        if mode == "hybrid":
            interior = cv2.erode(full.astype(np.uint8), np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8),
                                 iterations=2, borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool) & ok
            n, labels = cv2.connectedComponents(interior.astype(np.uint8), connectivity=8)
            if n > 1:
                sums = np.bincount(labels[interior], ls[interior], minlength=n)
                counts = np.bincount(labels[interior], minlength=n)
                ls = np.where(interior, (sums / np.maximum(counts, 1))[labels], ls)
    d = cv2.distanceTransform(full.astype(np.uint8), cv2.DIST_L2, 5)
    w = np.clip((d - start) / ramp, 0, 1) * ok
    scale = np.where(ok, 2.0 ** (w * (ls - ly)), 1.0)
    out = pred * scale[..., None]
    out[~full] = pred[~full]
    if mode == "hybrid" and n > 1:
        for c in range(3):
            sums = np.bincount(labels[interior], out[..., c][interior], minlength=n)
            out[..., c] = np.where(interior, (sums / np.maximum(counts, 1))[labels], out[..., c])
    return out.astype(np.float32), full


def constrain_colour(source, prediction, baseline, weight, view, include_fully_clipped=True,
                     forbid_new_negative_y=False):
    """Stage 6 (RUDRA training/aces_colour_backtrack.py): on highlight support,
    halve the step from the baseline until the ACES 2.0 SDR view is within one
    8-bit code of the source (nine tries, then the baseline)."""
    source = np.asarray(source, np.float32)
    prediction = np.asarray(prediction, np.float32)
    baseline = np.asarray(baseline, np.float32)
    weight = np.asarray(weight, np.float32)
    result = prediction.copy()
    support = (weight > 0) & (source.max(-1) > .5)
    if not include_fully_clipped:
        support &= source.min(-1) < FULL_CLIP
    alpha = np.ones(source.shape[:2], np.float32)
    active = support.copy()
    guard = forbid_new_negative_y
    baseline_y_ok = (baseline @ LUMA_AP1 >= 0) if guard else None
    for step in range(9):
        if not active.any():
            break
        indices = np.flatnonzero(active)
        codes = view.forward(result.reshape(-1, 3)[indices])
        error = np.abs(codes - source.reshape(-1, 3)[indices]).max(-1) * 255
        ok = error <= 1
        if guard:
            ok &= (result.reshape(-1, 3)[indices] @ LUMA_AP1 >= 0) | ~baseline_y_ok.reshape(-1)[indices]
        active.reshape(-1)[indices[ok]] = False
        if not active.any():
            break
        alpha[active] = alpha[active] * .5 if step < 8 else 0
        result[active] = baseline[active] + alpha[active, None] * (prediction[active] - baseline[active])
    stats = {"support_pixels": int(support.sum()),
             "changed_pixels": int((support & (alpha < 1)).sum()),
             "baseline_fallback_pixels": int((support & (alpha == 0)).sum()),
             "remaining_display_errors": 0, "inverse_limit_pixels": 0}
    # RUDRA 1.1.0: of the pixels still more than one code off, those where the
    # plain ACES inverse is off too (near-white tinted codes such as 253,255,255,
    # which the ACES 2.0 view sends to white) are the view's limit, not the
    # reconstruction's.
    idx = np.flatnonzero(support)
    if idx.size:
        err = np.abs(view.forward(result.reshape(-1, 3)[idx]) - source.reshape(-1, 3)[idx]).max(-1) * 255
        failing = idx[err > 1]
        if failing.size:
            base_err = np.abs(view.forward(baseline.reshape(-1, 3)[failing])
                              - source.reshape(-1, 3)[failing]).max(-1) * 255
            stats["inverse_limit_pixels"] = int((base_err > 1).sum())
            stats["remaining_display_errors"] = int((base_err <= 1).sum())
    return result, stats


# ─────────────────────────────────────────────────────────────────────────────
#  RUDRA 1.1.0 stages (native/core/src/aces_v2.cpp): the highlight ceiling, the
#  user's highlight / shadow bands and Clean shadows
# ─────────────────────────────────────────────────────────────────────────────

def inverse_white_luma(view: "ACES2View") -> float:
    """AP1 luminance of the view's inverse of display white (1, 1, 1): RUDRA's
    highlight ceiling (about 128 in ACEScg for ACES 2.0 SDR 100 nits)."""
    w = view.inverse(np.ones((1, 1, 3), np.float32)).reshape(3).astype(np.float64)
    return float(w @ LUMA_AP1)


def cap_luminance(rgb: np.ndarray, luma_max: float) -> Tuple[np.ndarray, int]:
    """Scale every pixel brighter than ``luma_max`` (AP1 luminance) down to it,
    hue kept. Returns (capped copy, pixels capped)."""
    if not luma_max > 0:
        return rgb, 0
    y = rgb.astype(np.float64) @ LUMA_AP1
    over = y > luma_max
    if not over.any():
        return rgb, 0
    out = rgb.copy()
    out[over] = (rgb[over].astype(np.float64) * (luma_max / y[over])[:, None]).astype(rgb.dtype)
    return out, int(over.sum())


BAND_RAMP_CODES = 2


def user_band_weight(top: np.ndarray, tau_high: int = 254, tau_low: int = 5) -> np.ndarray:
    """1 where the source's brightest channel (``top``, 0..1) is at or above
    ``tau_high`` or at or below ``tau_low`` (8-bit codes), 0 in between, with a
    smoothstep over 2 codes. tau_high 256 keeps no highlights, tau_low -1 no shadows."""
    code = np.asarray(top, np.float32) * 255.0
    ramp = float(BAND_RAMP_CODES)
    hi = np.zeros_like(code)
    lo = np.zeros_like(code)
    if tau_high <= 255:
        t = np.clip((code - (tau_high - ramp)) / ramp, 0.0, 1.0)
        hi = t * t * (3.0 - 2.0 * t)
    if tau_low >= 0:
        t = np.clip((code - tau_low) / ramp, 0.0, 1.0)
        lo = 1.0 - t * t * (3.0 - 2.0 * t)
    return np.maximum(hi, lo).astype(np.float32)


def apply_user_band(prediction: np.ndarray, baseline: np.ndarray, source: np.ndarray,
                    tau_high: int = 254, tau_low: int = 5) -> Tuple[np.ndarray, np.ndarray, dict]:
    """The learned result only in the bands, the exact plain ACES inverse
    between (which the ACES 2.0 view returns to the source pixel for pixel)."""
    w = user_band_weight(source.max(-1), tau_high, tau_low)
    out = baseline + w[..., None] * (prediction - baseline)
    out = np.where((w >= 1.0)[..., None], prediction, np.where((w <= 0.0)[..., None], baseline, out))
    stats = {"learned_pixels": int((w >= 1).sum()), "ramp_pixels": int(((w > 0) & (w < 1)).sum()),
             "analytic_pixels": int((w <= 0).sum())}
    return out.astype(np.float32), w, stats


def clean_shadows(acescg: np.ndarray, source: np.ndarray, max_code: int = 40, radius: int = 3,
                  range_stops: float = 0.25) -> Tuple[np.ndarray, int]:
    """Edge-preserving denoise of the deep shadows: weight 1 where the source's
    brightest channel is at or below ``max_code``, 0 from ``max_code + 8``; each
    pixel there becomes a mean of its neighbours weighted by distance (sigma
    radius / 2) and by the difference in log2 luminance (sigma range_stops).
    Nothing is invented: only what the neighbourhood holds."""
    h, w_ = acescg.shape[:2]
    t = np.clip((source.max(-1) * 255.0 - float(max_code)) / 8.0, 0.0, 1.0)
    weight = 1.0 - t * t * (3.0 - 2.0 * t)
    active = weight > 0
    if not active.any():
        return acescg, 0
    rgb = acescg.astype(np.float64)
    logy = np.log2(np.maximum(rgb @ LUMA_AP1, 1e-9))
    ss = 2.0 * (0.5 * radius) ** 2
    rs = 2.0 * range_stops * range_stops
    acc = np.zeros_like(rgb)
    wsum = np.zeros((h, w_), np.float64)
    pad = radius
    rgb_p = np.pad(rgb, ((pad, pad), (pad, pad), (0, 0)))
    logy_p = np.pad(logy, pad)
    valid_p = np.pad(np.ones((h, w_), bool), pad)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            ys, xs = pad + dy, pad + dx
            nb = rgb_p[ys:ys + h, xs:xs + w_]
            ok = valid_p[ys:ys + h, xs:xs + w_]
            d = logy_p[ys:ys + h, xs:xs + w_] - logy
            k = np.exp(-float(dx * dx + dy * dy) / ss) * np.exp(-(d * d) / rs) * ok
            acc += k[..., None] * nb
            wsum += k
    mean = acc / np.maximum(wsum, 1e-300)[..., None]
    out = rgb + weight[..., None] * (mean - rgb)
    out = np.where(active[..., None], out, rgb)
    return out.astype(np.float32), int(active.sum())


# ─────────────────────────────────────────────────────────────────────────────
#  One frame
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FrameResult:
    acescg: np.ndarray          # H,W,3 float32 ACEScg master
    mask: np.ndarray            # H,W float32 rudra.reconstructed
    baseline: np.ndarray        # H,W,3 float32 plain ACES 2.0 inverse
    tiled: bool = False
    stats: dict = field(default_factory=dict)


# ALBABIT-FIX: run_batch runs frames on threads; G1 takes one at a time and
# hands its outputs back on the CPU, so VRAM stays that of a single frame
# (0.64 GB at 1080p, 1.9 GB without the lock).
_G1_LOCK = threading.Lock()


def _g1(pkg: ACES2Package, src: torch.Tensor, base: torch.Tensor):
    with _G1_LOCK, torch.inference_mode():
        return tuple(t.float().cpu() for t in pkg.g1(src.to(pkg.device), base.to(pkg.device)))


def _g1_tiled(pkg: ACES2Package, src: torch.Tensor, base: torch.Tensor, tile: int, overlap: int):
    """G1 over overlapping tiles, each tile's centre kept. An approximation of the
    whole-frame result near tile seams (the context kernels reach 65 px)."""
    _, _, h, w = src.shape
    outs = [torch.zeros((1, c, h, w)) for c in (3, 3, 3, 1)]
    step = max(1, tile - 2 * overlap)
    for y0 in range(0, h, step):
        for x0 in range(0, w, step):
            ya, xa = max(0, y0 - overlap), max(0, x0 - overlap)
            yb, xb = min(h, y0 + step + overlap), min(w, x0 + step + overlap)
            res = _g1(pkg, src[..., ya:yb, xa:xb], base[..., ya:yb, xa:xb])
            ye, xe = min(h, y0 + step), min(w, x0 + step)
            for o, r in zip(outs, res):
                o[..., y0:ye, x0:xe] = r[..., y0 - ya:ye - ya, x0 - xa:xe - xa]
    return tuple(outs)


def run_frame(pkg: ACES2Package, source_hwc: np.ndarray, view: Optional[ACES2View] = None,
              smoothing: bool = True, constrain: bool = True, tile: int = 1024,
              overlap: int = 128, band: Optional[dict] = None,
              clean: Optional[dict] = None, ceiling: bool = True) -> FrameResult:
    """The full staged pipeline on one H x W x 3 frame of display codes in [0, 1].

    RUDRA 1.1.0 additions: the highlight ceiling (on: nothing brighter than the
    inverse of display white), ``band`` ({"tau_high", "tau_low"}: the learned
    result only in highlights and shadows) and ``clean`` ({"max_code",
    "radius", "range_stops"}: Clean shadows). Both off by default."""
    view = view or aces2_view()
    src = np.ascontiguousarray(np.clip(source_hwc, 0.0, 1.0), np.float32)
    baseline = view.inverse(src)                                                 # stage 1
    limit = inverse_white_luma(view) if ceiling else 0.0
    if ceiling:
        baseline, _ = cap_luminance(baseline, limit)
    x = torch.from_numpy(src).permute(2, 0, 1)[None]
    b = torch.from_numpy(baseline).permute(2, 0, 1)[None]
    tiled = False
    try:
        rs, hl, full, weight = _g1(pkg, x, b)                                     # stage 2
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.warning("[RUDRA ACES 2] whole frame does not fit; running G1 in %d px tiles.", tile)
        rs, hl, full, weight = _g1_tiled(pkg, x, b, tile, overlap)
        tiled = True
    restored_source = rs[0].permute(1, 2, 0).numpy()
    restored = view.inverse(np.clip(restored_source, 0.0, 1.0))                  # stage 3
    if ceiling:
        restored, _ = cap_luminance(restored, limit)
    with torch.inference_mode():
        composed, mask = staged_compose(x, b, torch.from_numpy(restored).permute(2, 0, 1)[None],
                                        hl, full, weight)                         # stage 4
    out = composed[0].permute(1, 2, 0).numpy().astype(np.float32)
    stats = {}
    if smoothing and pkg.smoothing is not None:                                 # stage 5
        sm = pkg.smoothing
        out, _ = smooth_clipped(out, src, sigma=float(sm["sigma"]), ramp=float(sm["ramp"]),
                                start=float(sm.get("start", 0.0)), mode=sm.get("mode", "hybrid"))
    mask_np = mask[0, 0].numpy().astype(np.float32)
    extra = {}
    if band:                                                                    # stage 5b
        out, bw, extra["band"] = apply_user_band(out, baseline, src, int(band.get("tau_high", 254)),
                                                 int(band.get("tau_low", 5)))
        mask_np = mask_np * bw                       # the analytic band was not reconstructed
    if clean:
        out, extra["clean_shadow_pixels"] = clean_shadows(
            out, src, int(clean.get("max_code", 40)), int(clean.get("radius", 3)),
            float(clean.get("range_stops", 0.25)))
    if ceiling:
        out, extra["ceiling_pixels"] = cap_luminance(out, limit)
    if constrain:                                                               # stage 6
        out, stats = constrain_colour(src, out, baseline, weight[0, 0].numpy(), view,
                                      include_fully_clipped=True,
                                      forbid_new_negative_y=pkg.forbid_new_negative_y)
    if not np.isfinite(out).all():
        raise ValueError("RUDRA ACES 2 produced non-finite values")
    stats.update(extra)
    return FrameResult(out.astype(np.float32), mask_np,                         # stage 7
                       baseline.astype(np.float32), tiled, stats)


# ─────────────────────────────────────────────────────────────────────────────
#  Trims (RUDRA native/core/src/aces_v2_trims.cpp)
# ─────────────────────────────────────────────────────────────────────────────

def apply_trims(master: np.ndarray, baseline: np.ndarray, mask: np.ndarray, strength: float = 1.0,
                saturation: float = 1.0, peak_nits: float = 0.0, exposure_ev: float = 0.0,
                contrast: float = 1.0) -> np.ndarray:
    """A colourist's trims on the master, as the RUDRA app applies them.
    strength 0..1.5 (0 = plain inverse, 1 = the model), saturation 0..2 and a
    soft peak in nits only on reconstructed pixels; exposure and contrast on
    the whole master. Defaults return the master unchanged."""
    if strength == 1.0 and saturation == 1.0 and peak_nits <= 0 and exposure_ev == 0.0 and contrast == 1.0:
        return master
    s = float(np.clip(strength, 0.0, 1.5))
    sat = float(np.clip(saturation, 0.0, 2.0))
    gain = float(2.0 ** np.clip(exposure_ev, -3.0, 3.0))
    con = float(np.clip(contrast, 0.5, 1.5))
    rgb = (baseline.astype(np.float64) + s * (master.astype(np.float64) - baseline)) * gain
    if con != 1.0:
        y = rgb @ LUMA_AP1
        k = np.where(y > 0, 0.18 * np.power(np.maximum(y, 1e-30) / 0.18, con) / np.maximum(y, 1e-30), 1.0)
        rgb = rgb * k[..., None]
    w = np.clip(mask.astype(np.float64), 0.0, 1.0)[..., None]
    on = w[..., 0] > 0
    if on.any():
        y = (rgb @ LUMA_AP1)[..., None]
        if sat != 1.0:
            r2 = y + sat * (rgb - y)
            rgb = np.where(on[..., None], rgb + w * (r2 - rgb), rgb)
            y = (rgb @ LUMA_AP1)[..., None]
        if peak_nits > 0:
            cap = peak_nits / DIFFUSE_WHITE_NITS
            knee = 0.75 * cap
            room = cap - knee
            y2 = knee + room * (1.0 - np.exp(-(y - knee) / room))
            k = np.where(on[..., None] & (y > knee), 1.0 + w * (y2 / np.maximum(y, 1e-30) - 1.0), 1.0)
            rgb = rgb * k
    return rgb.astype(np.float32)


def acescg_to(rgb: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return (rgb.astype(np.float64) @ matrix.T).astype(np.float32)


def sdr_codes_from_image(rgb: torch.Tensor, inverse_oetf: str) -> np.ndarray:
    """IMAGE RGB [H,W,3] -> the display codes the model reads (Rec.1886).

    Encoded SDR is read code-for-code, as the RUDRA app reads it: the model
    was trained and gated on 8-bit codes, and re-quantising them through
    another curve would break its sub-code restoration. Linear input
    (``None``) is encoded for a gamma-2.4 display first.
    """
    x = rgb.detach().float().clamp(0.0, 1.0).cpu().numpy()
    if inverse_oetf == "None":
        x = np.power(x, 1.0 / 2.4)
    return np.ascontiguousarray(x, np.float32)


def run_batch(pkg: ACES2Package, codes: np.ndarray, progress: Optional[Callable[[int], None]] = None,
              workers: int = 1, stats: Optional[list] = None,
              **kw) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """codes [B,H,W,3] -> (acescg [B,H,W,3], mask [B,H,W], baseline [B,H,W,3], tiled).
    ``stats``, when a list, receives each frame's stage statistics in order."""
    view = aces2_view()
    out, masks, bases, tiled = [], [], [], False
    # ALBABIT-FIX: TorchScript runs G1's first call at a frame size unoptimised,
    # with other bits; a blank frame laid out as run_frame's takes it instead.
    # Frames too big for the GPU are tiled one at a time, in order, as before.
    blank = torch.from_numpy(np.zeros(codes.shape[1:], np.float32)).permute(2, 0, 1)[None]
    try:
        _g1(pkg, blank, blank)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        workers = 1
    # ALBABIT-FIX: stages 5-6 run on one core, so `workers` frames go at once
    # (241 frames of 1080p in "Universal": 392 s -> 95 s); each keeps its bits.
    with ThreadPoolExecutor(max(1, workers)) as pool:
        for r in pool.map(lambda c: run_frame(pkg, c, view, **kw), codes):
            out.append(r.acescg)
            masks.append(r.mask)
            bases.append(r.baseline)
            tiled |= r.tiled
            if stats is not None:
                stats.append(r.stats)
            if progress:
                progress(1)
    return np.stack(out), np.stack(masks), np.stack(bases), tiled
