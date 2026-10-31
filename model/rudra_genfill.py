"""RUDRA AI fill v3: the shape inside fully clipped highlights, drawn by
Wan2.1-VACE-1.3B with the FXTD LoRA; the brightness set by a small estimator.

Port of RUDRA 1.0.1 ``rudra/genfill`` (encoding, masks, composite, brightness,
model, run) working on in-memory frames instead of EXR folders. The steps and
numbers are RUDRA's; what it guarantees is RUDRA's:

* Codes kept: no 8-bit code of the master's ACES 2.0 SDR 100-nit view changes
  (``r_safe_blend``). Pixels the fill does not touch are copied bit for bit.
* Highlights only by default: the fill is blended only where the source is
  fully clipped (code 255), and adds at most ``max_gain_stops`` (3) over the
  master. Shadows (code 0) only when asked for.
* Marked: every generated pixel's blend weight is returned (``rudra.generated``).
* What it draws is plausible, not recovered.

The master is RUDRA's: scene-linear ACEScg, 1.0 = diffuse white, whose ACES
2.0 SDR view is the source (what SDR -> HDR Universal / Recover make with the
RUDRA ACES 2 backend). Needs ``diffusers`` >= 0.35 (WanVACE) at run time;
``peft`` is used for the LoRA when installed, otherwise the LoRA is merged into
the weights directly.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("radiance.model.rudra_genfill")

AP1_Y = np.array([.2722287168, .6740817658, .0536895174], np.float32)
BASE_PROMPT = ("scene-referred high dynamic range footage, natural light, physically plausible "
               "highlights and shadow detail")
MAX_SEQUENCE_LENGTH = 512
BASE_REPO = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
BASE_DIRNAME = "Wan2.1-VACE-1.3B-diffusers"
BASE_ENV = "RADIANCE_WAN_VACE"
PACKAGE_ENV = "RADIANCE_RUDRA_GENFILL"


# ─────────────────────────────────────────────────────────────────────────────
#  encoding: scene-linear ACEScg <-> the VAE's [-1, 1] (normalised ACEScct)
# ─────────────────────────────────────────────────────────────────────────────

M_DEFAULT = 2.0 ** 14
_A, _B = 10.5402377416545, 0.0729055341958355
_X_BREAK = 0.0078125
_C_BREAK = 0.155251141552511


def _cct_scalar(x: float) -> float:
    return _A * x + _B if x <= _X_BREAK else (math.log2(x) + 9.72) / 17.52


def acescct(x):
    lin = _A * x + _B
    lg = (np.log2(np.clip(x, _X_BREAK, None)) + 9.72) / 17.52
    return np.where(x <= _X_BREAK, lin, lg)


def acescct_inverse(c):
    lin = (c - _B) / _A
    lg = 2.0 ** (np.clip(c, _C_BREAK, None) * 17.52 - 9.72)
    return np.where(c <= _C_BREAK, lin, lg)


def encode(x, M: float = M_DEFAULT):
    lo, hi = _cct_scalar(0.0), _cct_scalar(M)
    v = (acescct(np.clip(x, 0.0, M)) - lo) / (hi - lo)
    return v * 2.0 - 1.0


def decode(y, M: float = M_DEFAULT):
    lo, hi = _cct_scalar(0.0), _cct_scalar(M)
    v = np.clip((y + 1.0) * 0.5, 0.0, 1.0)
    return np.clip(acescct_inverse(v * (hi - lo) + lo), 0.0, M)


# ─────────────────────────────────────────────────────────────────────────────
#  masks
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MaskSettings:
    tau_high: int = 250          # 8-bit code; >= is highlight context for the model (256 = none)
    tau_low: int = -1            # 8-bit code; <= is shadow context (-1 = none)
    dilate: int = 4
    ema: float = 0.7
    feather: int = 0
    blend_high: int = 255        # the fill is blended only at max code >= this
    blend_low: int = 0           # ... and, with shadows on, at max code <= this
    max_gain_stops: float = 3.0

    def check(self):
        if not (-1 <= self.tau_low < self.tau_high <= 256):
            raise ValueError(f"need -1 <= tau_low < tau_high <= 256 (got {self.tau_low}, {self.tau_high})")
        if not (0 <= self.blend_high <= 256 and -1 <= self.blend_low <= 255):
            raise ValueError("need 0 <= blend_high <= 256 and -1 <= blend_low <= 255")
        if not (0.0 <= self.max_gain_stops <= 20.0):
            raise ValueError("max_gain_stops must be in [0, 20]")
        if not (0.0 <= self.ema < 1.0):
            raise ValueError("ema must be in [0, 1)")
        return self


def _dilate(m: np.ndarray, r: int) -> np.ndarray:
    if r <= 0 or not m.any():
        return m
    import cv2
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(m.astype(np.uint8), k) > 0


def frame_masks(codes: np.ndarray, s: MaskSettings) -> Tuple[np.ndarray, np.ndarray]:
    mx = codes.max(-1)
    hi = mx >= s.tau_high if s.tau_high <= 255 else np.zeros(mx.shape, bool)
    lo = mx <= s.tau_low if s.tau_low >= 0 else np.zeros(mx.shape, bool)
    hi, lo = _dilate(hi, s.dilate), _dilate(lo, s.dilate)
    return hi, lo & ~hi


class TemporalMask:
    """Per-pixel EMA of the masks across frames (frames in order)."""

    def __init__(self, s: MaskSettings):
        self.s = s.check()
        self._hi = self._lo = None

    def __call__(self, codes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        hi, lo = frame_masks(codes, self.s)
        hi, lo = hi.astype(np.float32), lo.astype(np.float32)
        if self._hi is not None and self._hi.shape == hi.shape and self.s.ema > 0:
            a = self.s.ema
            hi = a * self._hi + (1 - a) * hi
            lo = a * self._lo + (1 - a) * lo
        self._hi, self._lo = hi, lo
        h, l = hi >= 0.5, lo >= 0.5
        return h, l & ~h


def blend_core(codes: np.ndarray, s: MaskSettings) -> np.ndarray:
    mx = codes.max(-1)
    core = mx >= s.blend_high if s.blend_high <= 255 else np.zeros(mx.shape, bool)
    if s.tau_low >= 0 and s.blend_low >= 0:
        core = core | (mx <= s.blend_low)
    return core


def feather(mask: np.ndarray, radius: int) -> np.ndarray:
    m = mask.astype(np.float32)
    if radius <= 0 or not mask.any():
        return m
    import cv2
    grown = _dilate(mask, radius).astype(np.float32)
    soft = cv2.GaussianBlur(grown, (0, 0), radius / 2.0)
    return np.maximum(m, np.clip(soft, 0, 1)) * grown


# ─────────────────────────────────────────────────────────────────────────────
#  composite: the round-trip-safe blend
# ─────────────────────────────────────────────────────────────────────────────

def _codes(view, rgb: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(view.forward(np.ascontiguousarray(rgb, np.float32)), 0.0, 1.0) * 255.0).astype(np.int16)


def r_safe_blend(master: np.ndarray, fill: np.ndarray, weight: np.ndarray, view,
                 iterations: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    """out = master + alpha * (fill - master) with alpha <= weight chosen so the
    master's SDR view keeps every 8-bit code (bisection; alpha 0 always does)."""
    if master.shape != fill.shape or master.shape[:2] != weight.shape:
        raise ValueError("master, fill and weight must agree in size")
    out = np.array(master, np.float32, copy=True)
    alpha = np.zeros(weight.shape, np.float32)
    idx = np.nonzero(weight > 0)
    if not len(idx[0]):
        return out, alpha
    m = master[idx].astype(np.float32)
    d = fill[idx].astype(np.float32) - m
    w = np.clip(weight[idx].astype(np.float32), 0.0, 1.0)
    ref = _codes(view, m)
    good = (_codes(view, m + w[:, None] * d) == ref).all(-1)
    best = np.where(good, w, 0.0).astype(np.float32)
    bad = np.nonzero(~good)[0]
    if len(bad):
        lo = np.zeros(len(bad), np.float32)
        hi = w[bad].copy()
        mb, db, rb = m[bad], d[bad], ref[bad]
        for _ in range(iterations):
            mid = (lo + hi) * 0.5
            g = (_codes(view, mb + mid[:, None] * db) == rb).all(-1)
            lo = np.where(g, mid, lo)
            hi = np.where(g, hi, mid)
        best[bad] = lo
    alpha[idx] = best
    out[idx] = m + best[:, None] * d
    return out, alpha


# ─────────────────────────────────────────────────────────────────────────────
#  brightness (v3): how bright a clipped area was
# ─────────────────────────────────────────────────────────────────────────────

FEATURES = ["log2_area", "level", "min_chan", "all_three", "ring1_rel", "ring1_near", "ring2_rel", "slope",
            "ring_std", "compact", "frame_rel", "frame_clipped", "centroid_y", "ring_sat"]
MIN_PIXELS = 16


def log2_lum(rgb: np.ndarray) -> np.ndarray:
    return np.log2(np.maximum(rgb @ AP1_Y, 1e-6)).astype(np.float32)


def components(codes: np.ndarray, min_pixels: int = MIN_PIXELS):
    import cv2
    core = (codes.max(-1) >= 255).astype(np.uint8)
    n, lab = cv2.connectedComponents(core, connectivity=8)
    ids = [c for c in range(1, n) if (lab == c).sum() >= min_pixels]
    return lab.astype(np.int32), ids


def _ring(mask: np.ndarray, r0: int, r1: int) -> np.ndarray:
    import cv2
    k = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))  # noqa: E731
    outer = cv2.dilate(mask.astype(np.uint8), k(r1)) > 0
    inner = cv2.dilate(mask.astype(np.uint8), k(r0)) > 0 if r0 > 0 else mask
    return outer & ~inner


def frame_features(codes: np.ndarray, master: np.ndarray):
    ly = log2_lum(master)
    mx = codes.max(-1)
    return {"ly": ly, "median": float(np.median(ly)), "clipped": float((mx >= 255).mean()), "mx": mx}


def component_features(codes: np.ndarray, master: np.ndarray, mask: np.ndarray, frame=None) -> np.ndarray:
    import cv2
    fr = frame or frame_features(codes, master)
    ly, mx = fr["ly"], fr["mx"]
    h, w = mask.shape
    area = float(mask.sum())
    level = float(ly[mask].mean())
    cm = codes[mask].astype(np.float32)
    min_chan = float(cm.min(-1).mean() / 255.0)
    all_three = float((cm.min(-1) >= 255).mean())
    r1, r2 = _ring(mask, 0, 3), _ring(mask, 4, 10)
    r1 &= mx < 255
    r2 &= mx < 255
    l1 = float(ly[r1].mean()) if r1.any() else level
    l2 = float(ly[r2].mean()) if r2.any() else l1
    near = float((mx[r1] >= 250).mean()) if r1.any() else 1.0
    std = float(ly[r1 | r2].std()) if (r1 | r2).any() else 0.0
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    per = float(sum(cv2.arcLength(c, True) for c in cnts)) or 1.0
    ys = np.nonzero(mask)[0]
    rr = master[r1] if r1.any() else master[mask]
    sat = float(((rr.max(-1) - rr.min(-1)) / np.maximum(rr.max(-1), 1e-6)).mean())
    return np.array([np.log2(area / (h * w)), level, min_chan, all_three, l1 - level, near, l2 - level,
                     (l1 - l2) / 5.5, std, min(1.0, 4 * np.pi * area / per ** 2), fr["median"] - level,
                     fr["clipped"], float(ys.mean() / h), sat], np.float32)


class BrightnessEstimator:
    """features -> b (stops over the master), >= 0. Inference only (14 -> 32 -> 32 -> 1 MLP)."""

    def __init__(self, state: dict):
        import torch
        if state.get("format") != "rudra_genfill_brightness_v1" or state.get("features") != FEATURES:
            raise ValueError("not a RUDRA brightness estimator for these features")
        self.net = torch.nn.Sequential(torch.nn.Linear(len(FEATURES), 32), torch.nn.GELU(),
                                       torch.nn.Linear(32, 32), torch.nn.GELU(), torch.nn.Linear(32, 1))
        self.net.load_state_dict({k: torch.tensor(v) for k, v in state["weights"].items()})
        self.mu = np.array(state["mu"], np.float32)
        self.sd = np.array(state["sd"], np.float32)
        self.meta = state.get("meta", {})

    def predict(self, feats: np.ndarray) -> np.ndarray:
        import torch
        x = torch.from_numpy(((np.atleast_2d(feats) - self.mu) / self.sd).astype(np.float32))
        with torch.no_grad():
            return torch.nn.functional.softplus(self.net(x)[:, 0]).numpy()

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class BrightnessSettings:
    estimator: Optional[BrightnessEstimator] = None
    offset: float = 0.0
    shape_gain: float = 1.0
    ema: float = 0.7
    identity: dict = field(default_factory=dict)


def excess_map(codes: np.ndarray, master: np.ndarray, estimator: Optional[BrightnessEstimator],
               offset: float = 0.0):
    lab, ids = components(codes)
    out = np.full(codes.shape[:2], np.nan, np.float32)
    if not ids:
        return out, lab
    fr = frame_features(codes, master)
    feats = np.stack([component_features(codes, master, lab == c, fr) for c in ids])
    b = estimator.predict(feats) if estimator is not None else np.zeros(len(ids), np.float32)
    for c, bc in zip(ids, b):
        out[lab == c] = max(0.0, float(bc) + offset)
    return out, lab


def compose(master: np.ndarray, generated: np.ndarray, b_map: np.ndarray, labels: np.ndarray,
            shape_gain: float = 1.0) -> np.ndarray:
    out = np.array(master, np.float32, copy=True)
    lm, lg = log2_lum(master), log2_lum(generated)
    for c in np.unique(labels[np.isfinite(b_map)]):
        s = (labels == c) & np.isfinite(b_map)
        rel = lg[s] - lm[s]
        shape = rel - rel.mean()
        ly = lm[s] + np.maximum(0.0, b_map[s] + shape_gain * shape)
        g = generated[s]
        yg = np.maximum(g @ AP1_Y, 1e-6)[:, None]
        ratio = np.where(yg > 1e-5, g / yg, master[s] / np.maximum(master[s] @ AP1_Y, 1e-6)[:, None])
        out[s] = ratio * (2.0 ** ly)[:, None]
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  model: Wan2.1-VACE components + RUDRA LoRA (mirrors diffusers' WanVACEPipeline)
# ─────────────────────────────────────────────────────────────────────────────

def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class PromptBank:
    """umT5 prompt embeddings, cached on disk by text (the 11 GB encoder loads at most once)."""

    def __init__(self, base, cache, device="cpu", dtype=None):
        import torch
        self.base, self.device, self.dtype = base, device, dtype or torch.bfloat16
        self.cache = Path(cache) if cache else None
        self._mem: Dict[str, object] = {}
        self._encoder = None

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha256(text.strip().encode()).hexdigest()[:24]

    def _load_encoder(self):
        from transformers import AutoTokenizer, UMT5EncoderModel
        ensure_text_encoder(self.base)
        tok = AutoTokenizer.from_pretrained(self.base, subfolder="tokenizer")
        enc = UMT5EncoderModel.from_pretrained(self.base, subfolder="text_encoder", torch_dtype=self.dtype)
        self._encoder = (tok, enc.to(self.device).eval())

    def release(self):
        import torch
        self._encoder = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get(self, text: str):
        import torch
        k = self.key(text)
        if k in self._mem:
            return self._mem[k]
        f = self.cache / f"{k}.pt" if self.cache else None
        if f and f.exists():
            e = torch.load(f, map_location="cpu", weights_only=True)
            self._mem[k] = e
            return e
        if self.base is None:
            raise KeyError(f"no embeddings for prompt {text!r} and no text encoder to make them")
        if self._encoder is None:
            logger.info("[RUDRA AI fill] encoding a new prompt with umT5 (loaded once, on %s)", self.device)
            self._load_encoder()
        tok, enc = self._encoder
        with torch.no_grad():
            t = tok([" ".join(text.split())], padding="max_length", max_length=MAX_SEQUENCE_LENGTH,
                    truncation=True, add_special_tokens=True, return_attention_mask=True, return_tensors="pt")
            n = int(t.attention_mask.gt(0).sum())
            e = enc(t.input_ids.to(self.device), t.attention_mask.to(self.device)).last_hidden_state.float().cpu()
        e[:, n:] = 0
        if f:
            try:
                f.parent.mkdir(parents=True, exist_ok=True)
                torch.save(e, f)
                f.with_suffix(".json").write_text(json.dumps({"text": text, "tokens": n}))
            except OSError as exc:
                logger.debug("prompt cache not written: %s", exc)
        self._mem[k] = e
        return e


@dataclass
class FillSettings:
    steps: int = 30
    seed: int = 0
    conditioning_scale: float = 1.0
    guidance: float = 1.0
    negative: str = ""
    prompt: str = BASE_PROMPT
    prompt_highlight: str = ""
    prompt_shadow: str = ""


def load_lora(transformer, path) -> dict:
    """The RUDRA genfill LoRA onto *transformer*: through peft when it is
    installed (what RUDRA does), else merged into the Linear weights
    (W += alpha/r * B @ A), which computes the same function."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file
    with safe_open(str(path), "pt") as f:
        meta = f.metadata() or {}
    if meta.get("format") != "rudra_genfill_lora_v1":
        raise ValueError(f"{path}: not a RUDRA genfill LoRA")
    rank, alpha = int(meta["rank"]), float(meta["alpha"])
    state = load_file(str(path))
    try:
        from peft import LoraConfig
        from peft.utils import set_peft_model_state_dict
    except ImportError:
        LoraConfig = None
    if LoraConfig is not None:
        cfg = LoraConfig(r=rank, lora_alpha=alpha, target_modules=json.loads(meta["targets"]),
                         init_lora_weights=True)
        transformer.add_adapter(cfg, adapter_name="rudra_genfill")
        res = set_peft_model_state_dict(transformer, state, adapter_name="rudra_genfill")
        missing = [k for k in (res.missing_keys if res else []) if "lora_" in k]
        if missing:
            raise ValueError(f"{path}: LoRA weights missing for {len(missing)} modules (first {missing[0]})")
        meta["applied_with"] = "peft"
    else:
        modules = dict(transformer.named_modules())
        scale = alpha / rank
        done = 0
        for key in state:
            if not key.endswith(".lora_A.weight"):
                continue
            name = key[: -len(".lora_A.weight")]
            mod = modules.get(name)
            b_key = name + ".lora_B.weight"
            if mod is None or not hasattr(mod, "weight") or b_key not in state:
                raise ValueError(f"{path}: no module {name} in the base transformer")
            with torch.no_grad():
                delta = (state[b_key].float() @ state[key].float()) * scale
                mod.weight.copy_((mod.weight.float() + delta.to(mod.weight.device)).to(mod.weight.dtype))
            done += 1
        if not done:
            raise ValueError(f"{path}: no LoRA weights found")
        meta["applied_with"] = f"merged ({done} modules)"
    for p in transformer.parameters():
        p.requires_grad_(False)
    return meta


class WanFill:
    def __init__(self, vae, transformer, scheduler, prompts: PromptBank, device="cpu", dtype=None):
        import torch
        self.vae, self.transformer, self.scheduler, self.prompts = vae, transformer, scheduler, prompts
        self.device, self.dtype = torch.device(device), dtype or torch.bfloat16
        self.t_scale = 2 ** sum(vae.config.temperal_downsample)
        self.s_scale = 2 ** len(vae.config.temperal_downsample)
        self.patch = transformer.config.patch_size[1]
        z = vae.config.z_dim
        self._mean = torch.tensor(vae.config.latents_mean, dtype=torch.float32).view(1, z, 1, 1, 1)
        self._inv_std = 1.0 / torch.tensor(vae.config.latents_std, dtype=torch.float32).view(1, z, 1, 1, 1)
        self.identity: dict = {}

    @property
    def multiple(self) -> int:
        return self.s_scale * self.patch

    @classmethod
    def load(cls, base, lora=None, device="cuda", dtype=None, flow_shift: float = 3.0,
             prompt_cache=None, text_device: Optional[str] = None):
        import torch
        try:
            from diffusers import AutoencoderKLWan, WanVACETransformer3DModel
            from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
        except ImportError as exc:
            raise ImportError("RUDRA AI fill needs diffusers >= 0.35 (WanVACE): "
                              "pip install -U 'diffusers>=0.35'") from exc
        dtype = dtype or torch.bfloat16
        vae = AutoencoderKLWan.from_pretrained(base, subfolder="vae", torch_dtype=torch.float32).to(device).eval()
        tr = WanVACETransformer3DModel.from_pretrained(base, subfolder="transformer", torch_dtype=dtype)
        lora_meta = load_lora(tr, lora) if lora else {}
        tr = tr.to(device).eval()
        sch = UniPCMultistepScheduler.from_pretrained(base, subfolder="scheduler")
        sch = UniPCMultistepScheduler.from_config(sch.config, flow_shift=flow_shift)
        pb = PromptBank(base, prompt_cache, device=text_device or device, dtype=dtype)
        self = cls(vae, tr, sch, pb, device=device, dtype=dtype)
        self.identity = {"base": str(base), "lora": str(lora) if lora else None,
                         "lora_sha256": sha256_file(lora) if lora else None,
                         "lora_applied_with": lora_meta.get("applied_with"), "flow_shift": flow_shift,
                         "dtype": str(dtype).replace("torch.", ""), "vae_dtype": "float32"}
        return self

    def to(self, device):
        """Move to *device* (the GPU cache calls this on eviction)."""
        self.vae.to(device)
        self.transformer.to(device)
        return self

    def encode_pixels(self, video):
        import torch
        with torch.no_grad():
            z = self.vae.encode(video.to(self.device, torch.float32)).latent_dist.mode()
        return (z.float() - self._mean.to(z.device)) * self._inv_std.to(z.device)

    def decode_latents(self, z):
        import torch
        z = z.to(self.device, torch.float32) / self._inv_std.to(self.device) + self._mean.to(self.device)
        with torch.no_grad():
            return self.vae.decode(z, return_dict=False)[0].float().clamp(-1, 1)

    def mask_latent(self, mask):
        import torch.nn.functional as F
        _, _, n, h, w = mask.shape
        s = self.s_scale
        nf = (n + self.t_scale - 1) // self.t_scale
        m = mask[0, 0].float().view(n, h // s, s, w // s, s).permute(2, 4, 0, 1, 3).flatten(0, 1)
        return F.interpolate(m.unsqueeze(0), size=(nf, h // s, w // s), mode="nearest-exact")

    def context(self, video, mask):
        import torch
        _, _, n, h, w = video.shape
        if (n - 1) % self.t_scale or h % self.multiple or w % self.multiple:
            raise ValueError(f"frames must be 4k+1 and sizes multiples of {self.multiple} (got {n}x{h}x{w})")
        m = (mask > 0.5).float().to(self.device)
        v = video.to(self.device, torch.float32)
        inactive = self.encode_pixels(v * (1 - m))
        reactive = self.encode_pixels(v * m)
        return torch.cat([inactive, reactive, self.mask_latent(m).to(inactive)], dim=1)

    def _region_weights(self, masks: dict, shape) -> dict:
        import torch
        import torch.nn.functional as F
        out = {}
        f, h, w = shape[2:]
        for k, m in masks.items():
            if m is None:
                continue
            m = m.float()
            n = m.shape[2]
            groups = [m[:, :, :1]] + [m[:, :, 1 + 4 * i:1 + 4 * (i + 1)] for i in range((n - 1) // 4)]
            g = torch.cat([x.amax(2, keepdim=True) for x in groups], 2)
            out[k] = F.adaptive_max_pool3d(g, (f, h, w)).to(self.device)
        return out

    def _predict(self, x, t, embeds, ctx, scale):
        return self.transformer(hidden_states=x.to(self.dtype), timestep=t.expand(x.shape[0]),
                                encoder_hidden_states=embeds.to(self.device, self.dtype),
                                control_hidden_states=ctx.to(self.dtype), control_hidden_states_scale=scale,
                                return_dict=False)[0].float()

    def denoise(self, ctx, s: FillSettings, region_masks: Optional[dict] = None, progress=None):
        import torch
        with torch.no_grad():
            n_layers = len(self.transformer.config.vace_layers)
            scale = torch.full((n_layers,), float(s.conditioning_scale), device=self.device, dtype=self.dtype)
            g = torch.Generator("cpu").manual_seed(int(s.seed))
            shape = (1, self.transformer.config.in_channels) + tuple(ctx.shape[2:])
            x = torch.randn(shape, generator=g, dtype=torch.float32).to(self.device)
            base = self.prompts.get(s.prompt)
            regions = []
            weights = self._region_weights(region_masks or {}, shape)
            for key, text in (("highlight", s.prompt_highlight), ("shadow", s.prompt_shadow)):
                if text.strip() and key in weights and weights[key].any():
                    regions.append((weights[key], self.prompts.get(text)))
            neg = self.prompts.get(s.negative) if s.guidance > 1 else None
            self.scheduler.set_timesteps(s.steps, device=self.device)
            for i, t in enumerate(self.scheduler.timesteps):
                eps = self._predict(x, t, base, ctx, scale)
                for w, e in regions:
                    eps = eps * (1 - w) + self._predict(x, t, e, ctx, scale) * w
                if neg is not None:
                    un = self._predict(x, t, neg, ctx, scale)
                    eps = un + s.guidance * (eps - un)
                x = self.scheduler.step(eps, t, x, return_dict=False)[0]
                if progress:
                    progress(i + 1, len(self.scheduler.timesteps))
            return x


# ─────────────────────────────────────────────────────────────────────────────
#  run: frames in memory
# ─────────────────────────────────────────────────────────────────────────────

def plan_windows(n: int, window: int, overlap: int) -> List[Tuple[int, int, int]]:
    """Frames 0..n-1 -> [(start, first_new, end)]; every frame written once."""
    if window < 1 or (window - 1) % 4:
        raise ValueError("window must be 4k+1")
    if not 0 <= overlap < window:
        raise ValueError("overlap must be in [0, window)")
    out, written = [], 0
    while written < n:
        start = max(0, written - overlap) if written else 0
        end = min(n, start + window)
        out.append((start, written, end))
        written = end
    return out


def pad_len(k: int) -> int:
    return k if (k - 1) % 4 == 0 else k + (4 - (k - 1) % 4)


def working_size(h: int, w: int, max_area: int, multiple: int = 16) -> Tuple[int, int]:
    s = min(1.0, math.sqrt(max_area / float(h * w)))
    return (max(multiple, int(h * s // multiple) * multiple),
            max(multiple, int(w * s // multiple) * multiple))


def temporal_excess(prev, cur) -> Optional[float]:
    if prev is None:
        return None
    sel = (cur[2] > 0) | (prev[2] > 0)
    if not sel.any() or prev[0].shape != cur[0].shape:
        return None
    lum = lambda x: np.log2(np.maximum(x @ np.array([0.2722, 0.6741, 0.0537], np.float32), 1e-4))  # noqa: E731
    d_fill = np.abs(lum(cur[1]) - lum(prev[1]))[sel].mean()
    d_master = np.abs(lum(cur[0]) - lum(prev[0]))[sel].mean()
    return float(max(0.0, d_fill - d_master))


def _small(x: np.ndarray, side: int = 256) -> np.ndarray:
    import cv2
    h, w = x.shape[:2]
    s = min(1.0, side / float(max(h, w)))
    return cv2.resize(x, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else x


TEMPORAL_FLAG = 0.25


def fill_frames(masters: List[np.ndarray], model, view, s: FillSettings, ms: MaskSettings,
                window: int = 33, overlap: int = 8, max_area: int = 832 * 480,
                brightness: Optional[BrightnessSettings] = None,
                progress: Optional[Callable[..., None]] = None) -> Tuple[List[np.ndarray], List[np.ndarray], dict]:
    """ACEScg masters (H,W,3) in frame order -> (filled masters, generated weights, report).

    RUDRA's run.fill_master step for step, frames in memory: the model sees the
    exact inverse of the master's SDR codes (log-encoded) at a working size,
    windows of `window` frames overlapping by `overlap` (filled frames from the
    previous window come in as kept frames, so windows join without a seam),
    and the fill is blended back only where the source holds no information,
    never changing an 8-bit code of the master's SDR view.
    """
    import cv2
    import torch
    if not masters:
        return [], [], {"frames": 0}
    H, W = masters[0].shape[:2]
    for m in masters:
        if m.shape[:2] != (H, W):
            raise ValueError("every frame must have the first frame's size")
    h, w = working_size(H, W, max_area, model.multiple)
    plan = plan_windows(len(masters), window, overlap)
    tmask = TemporalMask(ms)
    prev_b = None
    masks: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    gen_work: Dict[int, np.ndarray] = {}
    codes_all: Dict[int, np.ndarray] = {}
    outs: List[Optional[np.ndarray]] = [None] * len(masters)
    alphas: List[Optional[np.ndarray]] = [None] * len(masters)
    stats = {"frames": len(masters), "windows": len(plan), "generated_pixels": 0, "region_pixels": 0,
             "windows_generated": 0, "working_size": [h, w], "size": [H, W]}
    per_frame = []
    prev_small = None
    for wi, (start, new, end) in enumerate(plan):
        idxs = list(range(start, end))
        for i in idxs:
            if i not in codes_all:
                codes_all[i] = np.rint(np.clip(view.forward(masters[i]), 0, 1) * 255).astype(np.uint8)
            if i not in masks:
                masks[i] = tmask(codes_all[i])
        need = any(masks[i][0].any() or masks[i][1].any() for i in range(new, end))
        n_pad = pad_len(len(idxs))
        padded = idxs + [idxs[-1]] * (n_pad - len(idxs))
        dec = None
        if need:
            vid = np.zeros((n_pad, h, w, 3), np.float32)
            msk = np.zeros((n_pad, h, w), np.float32)
            regions_np = {"highlight": np.zeros((n_pad, h, w), np.float32),
                          "shadow": np.zeros((n_pad, h, w), np.float32)}
            for k, i in enumerate(padded):
                if i < new and i in gen_work:
                    vid[k] = gen_work[i]
                    continue
                analytic = view.inverse(codes_all[i].astype(np.float32) / 255.0)
                vid[k] = cv2.resize(encode(analytic).astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
                region = (masks[i][0] | masks[i][1]).astype(np.float32)
                msk[k] = cv2.resize(region, (w, h), interpolation=cv2.INTER_AREA) > 0
                for key, j in (("highlight", 0), ("shadow", 1)):
                    regions_np[key][k] = cv2.resize(masks[i][j].astype(np.float32), (w, h),
                                                    interpolation=cv2.INTER_AREA) > 0
            video = torch.from_numpy(vid).permute(3, 0, 1, 2)[None]
            mt = torch.from_numpy(msk)[None, None]
            regions = {k: torch.from_numpy(v)[None, None] for k, v in regions_np.items()}

            def step(a, b, _wi=wi):
                if progress:
                    progress(window=_wi + 1, windows=len(plan), step=a, steps=b)

            ctx = model.context(video, mt)
            z = model.denoise(ctx, s, regions, progress=step)
            dec = model.decode_latents(z)[0].permute(1, 2, 3, 0).cpu().numpy()
            stats["windows_generated"] += 1
        for k, i in enumerate(idxs):
            if i < new:
                continue
            master = masters[i]
            codes = codes_all[i]
            region = masks[i][0] | masks[i][1]
            if need and region.any():
                gen_work[i] = dec[k]
                full = cv2.resize(dec[k], (W, H), interpolation=cv2.INTER_CUBIC)
                fill_lin = decode(full.astype(np.float32)).astype(np.float32)
                if brightness is not None:
                    b_map, lab = excess_map(codes, master, brightness.estimator, brightness.offset)
                    if prev_b is not None and prev_b.shape == b_map.shape and brightness.ema > 0:
                        both = np.isfinite(b_map) & np.isfinite(prev_b)
                        b_map[both] = brightness.ema * prev_b[both] + (1 - brightness.ema) * b_map[both]
                    prev_b = b_map
                    fill_lin = compose(master, fill_lin, b_map, lab, brightness.shape_gain)
                if ms.max_gain_stops > 0:
                    lum = lambda x: np.maximum(x @ AP1_Y, 1e-6)  # noqa: E731
                    lim = lum(master) * np.float32(2.0 ** ms.max_gain_stops)
                    yf = lum(fill_lin)
                    fill_lin = fill_lin * np.where(yf > lim, lim / yf, 1.0)[..., None].astype(np.float32)
                core = blend_core(codes, ms) & region
                weight = feather(core, ms.feather) if ms.feather > 0 else core.astype(np.float32)
                out, alpha = r_safe_blend(master, fill_lin, weight, view)
            else:
                if need:
                    gen_work[i] = dec[k]
                out, alpha = master, np.zeros((H, W), np.float32)
            outs[i], alphas[i] = out, alpha
            cur_small = (_small(master), _small(out), _small(alpha.astype(np.float32)))
            temporal = temporal_excess(prev_small, cur_small)
            prev_small = cur_small
            per_frame.append({"frame": i, "generated_fraction": float((alpha > 0).mean()),
                              "temporal_excess_stops": temporal,
                              "flagged": bool(temporal is not None and temporal > TEMPORAL_FLAG)})
            stats["generated_pixels"] += int((alpha > 0).sum())
            stats["region_pixels"] += int(region.sum())
            if progress:
                progress(frame=i + 1, frames=len(masters))
        for i in list(gen_work):
            if i < end - overlap:
                del gen_work[i]
        for i in list(masks):
            if i < start:
                del masks[i]
        for i in list(codes_all):
            if i < start:
                del codes_all[i]
    stats["flagged_frames"] = sum(1 for f in per_frame if f["flagged"])
    stats["generated_fraction"] = stats["generated_pixels"] / float(max(1, len(masters) * H * W))
    report = {"model": getattr(model, "identity", {}), "settings": asdict(s), "mask": asdict(ms),
              "brightness": None if brightness is None else {**brightness.identity, "offset": brightness.offset,
                                                             "shape_gain": brightness.shape_gain},
              "window": window, "overlap": overlap, "max_area": max_area, "stats": stats, "frames": per_frame,
              "contract": "no 8-bit code of the master SDR view changed (r_safe_blend); untouched pixels bit-identical"}
    return outs, alphas, report


# ─────────────────────────────────────────────────────────────────────────────
#  finding the files
# ─────────────────────────────────────────────────────────────────────────────

def resolve_base(explicit: str = "") -> Optional[str]:
    """Wan2.1-VACE-1.3B-diffusers: explicit folder, RADIANCE_WAN_VACE, models/radiance/,
    models/diffusers/ (ComfyUI's), or None."""
    from radiance.model.paths import radiance_model_dirs
    cands = []
    for v in (explicit, os.environ.get(BASE_ENV, "")):
        v = (v or "").strip().strip('"').strip("'")
        if v:
            cands.append(Path(v).expanduser())
    for d in radiance_model_dirs():
        cands += [d / BASE_DIRNAME, d.parent / "diffusers" / BASE_DIRNAME]
    for c in cands:
        if (c / "model_index.json").is_file() and (c / "transformer").is_dir() and (c / "vae").is_dir():
            return str(c)
    return None


def resolve_package(explicit: str = "") -> Optional[Path]:
    from radiance.model.rudra_download import GENFILL_V3, find_package
    return find_package(GENFILL_V3, explicit, PACKAGE_ENV)


def check_runtime() -> None:
    """Fail before any download when this Python cannot run Wan VACE."""
    try:
        import diffusers
        from diffusers import AutoencoderKLWan, WanVACETransformer3DModel  # noqa: F401
    except ImportError as exc:
        import sys
        ver = getattr(sys.modules.get("diffusers"), "__version__", "not installed")
        raise RuntimeError(
            f"RUDRA AI fill needs diffusers 0.35 or newer (found {ver}). Run: "
            f'"{sys.executable}" -m pip install -U "diffusers>=0.35" and restart ComfyUI.') from exc


def ensure_package(explicit: str = "", allow: Optional[bool] = None) -> Optional[Path]:
    """The AI fill v3 package, downloaded on first use (271 MB) unless downloads are off."""
    found = resolve_package(explicit)
    if found is not None:
        return found
    from radiance.model.rudra_download import GENFILL_V3, download_package
    return download_package(GENFILL_V3, allow=allow)


def ensure_base(explicit: str = "", allow: Optional[bool] = None) -> Optional[str]:
    """Wan2.1-VACE-1.3B-diffusers, downloaded on first use into
    models/radiance/ (7.7 GB: VAE, transformer, scheduler, tokenizer; the text
    encoder comes later, only for a custom prompt) unless downloads are off."""
    found = resolve_base(explicit)
    if found is not None:
        return found
    from radiance.model.rudra_download import WAN_BASE, download_package
    got = download_package(WAN_BASE, allow=allow)
    return str(got) if got is not None else None


def ensure_text_encoder(base) -> None:
    """The umT5 text encoder next to *base*: fetched (11.4 GB) the first time a
    prompt that is not cached has to be encoded."""
    if base is None:
        return
    folder = Path(str(base))
    if not folder.is_dir():
        return                                    # a hub id: transformers fetches it itself
    from radiance.model.rudra_download import WAN_TEXT_ENCODER, download_package, package_complete
    te = folder / "text_encoder"
    if package_complete(folder, WAN_TEXT_ENCODER) or (
            (te / "config.json").is_file() and any(te.glob("*.safetensors"))
            and not any(te.glob("*.part"))):
        return
    logger.info("[RUDRA AI fill] a new prompt needs the umT5 text encoder; fetching it once (11.4 GB).")
    if download_package(WAN_TEXT_ENCODER, target=folder) is None:
        raise RuntimeError(
            "The prompt is not cached and the umT5 text encoder is missing. Leave the prompt empty "
            f"(RUDRA's base prompt is cached), allow downloads, or put text_encoder/ from "
            f"https://huggingface.co/{BASE_REPO} in {folder}.")


_MODEL: Dict[str, object] = {}


def get_model(base: str, package: Path, device, keep_loaded: bool = True) -> WanFill:
    key = f"{base}|{package}|{device}"
    if key in _MODEL:
        return _MODEL[key]
    release_model()
    model = WanFill.load(base, package / "rudra_genfill.safetensors", device=device,
                         prompt_cache=package / "prompt_cache", text_device="cpu")
    if keep_loaded:
        _MODEL[key] = model
    return model


def release_model() -> None:
    for m in list(_MODEL.values()):
        try:
            m.to("cpu")
        except Exception:  # noqa: BLE001
            pass
    _MODEL.clear()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
