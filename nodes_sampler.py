import torch
import time
import math
import logging
import gc  # ALBABIT FIX: Added garbage collection import for pre-flight cleanup
import inspect
from typing import Tuple, Dict, Any, Optional, List
from dataclasses import dataclass, field

import comfy.samplers
import comfy.sample
import comfy.model_management
import comfy.utils
from .gpu_memory import cleanup_gpu_memory

try:
    from comfy.nested_tensor import NestedTensor as _NestedTensor
    _HAS_NESTED_TENSOR = True
except ImportError:
    _NestedTensor = None
    _HAS_NESTED_TENSOR = False


def _sample_custom_compat(
    model,
    noise,
    cfg,
    sampler,
    sigmas,
    positive,
    negative,
    latent_image,
    noise_mask=None,
    callback=None,
    disable_pbar=False,
    seed=None,
):
    """Call ComfyUI sample_custom across positional/keyword API variants."""
    kwargs = {
        "model": model,
        "noise": noise,
        "cfg": cfg,
        "sampler": sampler,
        "sigmas": sigmas,
        "positive": positive,
        "negative": negative,
        "latent_image": latent_image,
        "noise_mask": noise_mask,
        "callback": callback,
        "disable_pbar": disable_pbar,
        "seed": seed,
    }
    try:
        params = inspect.signature(comfy.sample.sample_custom).parameters
        filtered = {k: v for k, v in kwargs.items() if k in params}
        return comfy.sample.sample_custom(**filtered)
    except (TypeError, ValueError):
        return comfy.sample.sample_custom(
            model,
            noise,
            cfg,
            sampler,
            sigmas,
            positive,
            negative,
            latent_image,
            noise_mask=noise_mask,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=seed,
        )

DYNAMIC_GUIDANCE_EARLY_MULTIPLIER = 0.6                                              
DYNAMIC_GUIDANCE_LATE_MULTIPLIER = 0.95                                                                         
DYNAMIC_GUIDANCE_EARLY_THRESHOLD = 0.2                               
DYNAMIC_GUIDANCE_LATE_THRESHOLD = 0.9                                  
DYNAMIC_GUIDANCE_RAMP_WIDTH = 0.05                                 

GUIDANCE_RESCALE_PHI = 0.0                                              

SIGMA_DISCONTINUITY_THRESHOLD = 0.01

PAG_DEFAULT_SCALE = 0.0                       
PAG_LAYER_NAMES = ["middle_block"]                               

CFG_PLUS_PLUS_DEFAULT_SCALE = 1.6                           

CFG_GUIDANCE_MODELS = {"wan", "hunyuan_video"}

DYNAMIC_CFG_EARLY_MULTIPLIER = 1.2                                              
DYNAMIC_CFG_LATE_MULTIPLIER = 0.7                                                
DYNAMIC_CFG_EARLY_THRESHOLD = 0.15                      
DYNAMIC_CFG_LATE_THRESHOLD = 0.85                      

MODEL_TYPES = [
    "auto",
    "flux",
    "sd3",
    "sd35",
    "sdxl",
    "sd15",
    "wan",
    "ltxv",
    "ltxav",                                     
    "hunyuan_video",
    "lumina2",
    "z_image",
    "chroma",
]

VIDEO_MODEL_TYPES = {"wan", "ltxv", "ltxav", "hunyuan_video", "cosmos"}

GUIDANCE_EMBED_MODELS = {"flux", "chroma", "lumina2", "z_image", "ltxv"}

CFG_GUIDED_MODELS = {"wan", "hunyuan_video", "sdxl", "sd15", "sd3", "sd35", "ltxav"}

MODEL_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "flux": {
        "cfg": 1.0,
        "scheduler": "simple",
        "guidance": 3.5,
        "shift": 1.0,
        "sampler": "euler",
        "denoise_range": (0.3, 1.0),
    },
    "sd3": {
        "cfg": 4.5,
        "scheduler": "sgm_uniform",
        "guidance": 0.0,
        "shift": 1.0,
        "sampler": "dpmpp_2m",
        "denoise_range": (0.2, 1.0),
    },
    "sd35": {
        "cfg": 4.5,
        "scheduler": "sgm_uniform",
        "guidance": 0.0,
        "shift": 1.0,
        "sampler": "dpmpp_2m",
        "denoise_range": (0.2, 1.0),
    },
    "sdxl": {
        "cfg": 7.0,
        "scheduler": "karras",
        "guidance": 0.0,
        "shift": 1.0,
        "sampler": "dpmpp_2m",                                                               
        "denoise_range": (0.3, 1.0),
    },
    "sd15": {
        "cfg": 7.0,
        "scheduler": "normal",
        "guidance": 0.0,
        "shift": 1.0,
        "sampler": "dpmpp_2m",                                                               
        "denoise_range": (0.3, 1.0),
    },

    "wan": {
        "cfg": 6.0,
        "scheduler": "simple",
        "guidance": 0.0,
        "shift": 8.0,
        "sampler": "euler",
        "denoise_range": (0.3, 1.0),
        "guidance_type": "cfg",                                              
    },
    "ltxv": {
        "cfg": 1.0,
        "scheduler": "simple",
        "guidance": 3.5,                                   
        "shift": 2.37,
        "sampler": "euler",
        "denoise_range": (0.3, 1.0),
        "guidance_type": "embedding",
    },

    "ltxav": {
        "cfg": 3.0,
        "scheduler": "beta",
        "guidance": 0.0,
        "shift": 3.0,                                                     
        "sampler": "euler",
        "denoise_range": (0.3, 1.0),
        "guidance_type": "cfg",
    },
    "hunyuan_video": {
        "cfg": 6.0,
        "scheduler": "simple",
        "guidance": 0.0,
        "shift": 7.0,
        "sampler": "euler",
        "denoise_range": (0.3, 1.0),
        "guidance_type": "cfg",                                                       
    },
    "lumina2": {
        "cfg": 1.0,
        "scheduler": "simple",
        "guidance": 3.5,                                    
        "shift": 6.0,
        "sampler": "euler",
        "denoise_range": (0.3, 1.0),
        "guidance_type": "embedding",
    },
    "z_image": {
        "cfg": 1.0,
        "scheduler": "simple",
        "guidance": 3.5,                                            
        "shift": 3.0,
        "sampler": "euler",
        "denoise_range": (0.3, 1.0),
        "guidance_type": "embedding",
    },
    "chroma": {
        "cfg": 1.0,
        "scheduler": "simple",
        "guidance": 0.0,
        "shift": 1.0,
        "sampler": "euler",
        "denoise_range": (0.3, 1.0),
    },
}

PREVIEW_METHODS = ["None", "TAESD", "Latent2RGB"]

NOISE_TYPES = ["Gaussian", "Perlin", "Uniform", "Spectral", "Brownian"]

CLIP_TARGETS = ["Auto", "clip_l", "clip_g", "t5xxl"]

MULTI_COND_MODES = ["Off", "average", "weighted", "sequential"]

TILE_BLEND_MODES = ["feather", "average", "gaussian"]

class SamplerMode:

    STANDARD = "Standard"
    PHASE_SHIFT_DPM = "Phase-Shift (Euler→DPM)"
    PHASE_SHIFT_SGM = "Phase-Shift (Euler→SGM)"
    CFG_PLUS_PLUS = "CFG++ (Perpendicular)"

    ALL = [STANDARD, PHASE_SHIFT_DPM, PHASE_SHIFT_SGM, CFG_PLUS_PLUS]

    @classmethod
    def is_phase_shift(cls, mode: str) -> bool:
        return mode in (cls.PHASE_SHIFT_DPM, cls.PHASE_SHIFT_SGM)

    @classmethod
    def is_cfg_plus_plus(cls, mode: str) -> bool:
        return mode == cls.CFG_PLUS_PLUS

logger = logging.getLogger("radiance.sampler")

class SigmaCache:
    MAX_ENTRIES = 32

    def __init__(self):
        from collections import OrderedDict
        self._cache: "OrderedDict[tuple, torch.Tensor]" = OrderedDict()

    @staticmethod
    def _make_key(model, scheduler: str, total_steps: int) -> tuple:
        try:
            ms = model.get_model_object("model_sampling")
            sigma_max = ms.sigma_max.item() if hasattr(ms, "sigma_max") else 0.0
            sigma_min = ms.sigma_min.item() if hasattr(ms, "sigma_min") else 0.0

            config_name = "unknown"
            try:
                if hasattr(model, "model") and hasattr(model.model, "model_config"):
                    config_name = type(model.model.model_config).__name__
            except (AttributeError, RuntimeError):
                pass

            return (config_name, round(sigma_max, 6), round(sigma_min, 6), scheduler, total_steps)
        except (AttributeError, RuntimeError):
            logger.debug("SigmaCache: failed to build config key, cache disabled for this model")
            return (id(model), scheduler, total_steps, time.time())

    def get(self, model, scheduler: str, total_steps: int) -> Optional[torch.Tensor]:
        key = self._make_key(model, scheduler, total_steps)
        if key in self._cache:
            self._cache.move_to_end(key)   
            return self._cache[key]
        return None

    def put(self, model, scheduler: str, total_steps: int, sigmas: torch.Tensor) -> None:
        key = self._make_key(model, scheduler, total_steps)
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self.MAX_ENTRIES:
                evicted = next(iter(self._cache))  
                del self._cache[evicted]
                logger.debug(f"SigmaCache: evicted LRU entry, {len(self._cache)} remaining")
        self._cache[key] = sigmas

    def clear(self) -> None:
        count = len(self._cache)
        self._cache.clear()
        if count:
            logger.debug(f"SigmaCache: cleared {count} entries")

_sigma_cache = SigmaCache()

def detect_model_type(model) -> str:

    try:

        diffusion_model = model.get_model_object("diffusion_model")
        if diffusion_model is not None:
            model_cls = type(diffusion_model).__name__.lower()
            model_module = (
                type(diffusion_model).__module__.lower()
                if hasattr(type(diffusion_model), "__module__")
                else ""
            )
            full_path = f"{model_module}.{model_cls}"

            if "wan" in model_cls or "wan" in model_module:
                return "wan"

            if (
                "ltxav" in model_cls
                or ("lightricks" in model_module and hasattr(diffusion_model, "recombine_audio_and_video_latents"))
            ):
                return "ltxav"
            if (
                "ltxv" in model_cls
                or "lightricks" in model_module
            ):
                return "ltxv"
            if "hunyuan" in model_cls and (
                "video" in model_cls or "video" in model_module
            ):
                return "hunyuan_video"

            if "lumina" in full_path:

                if (
                    hasattr(diffusion_model, "hidden_size")
                    and diffusion_model.hidden_size >= 3840
                ):
                    return "z_image"
                return "lumina2"
            if "chroma" in full_path:
                return "chroma"

            if "mmdit" in model_cls or "sd3" in model_cls:
                try:
                    if (
                        hasattr(diffusion_model, "in_channels")
                        and diffusion_model.in_channels >= 16
                    ):
                        return "sd35"
                except (AttributeError, RuntimeError):
                    pass
                return "sd3"

            if "sdxl" in model_cls or hasattr(diffusion_model, "label_emb"):
                return "sdxl"

        try:
            model_config = model.model.model_config if hasattr(model, "model") else None
            if model_config is not None:
                config_cls = type(model_config).__name__
                config_map = {
                    "WAN21": "wan",
                    "WAN22": "wan",
                    "LTXV": "ltxv",
                    "LTXAV": "ltxav",                        
                    "HunyuanVideo": "hunyuan_video",
                    "Lumina2": "lumina2",
                    "ZImage": "z_image",
                    "Chroma": "chroma",
                    "ChromaRadiance": "chroma",
                    "Flux": "flux",
                    "FluxSchnell": "flux",
                    "FluxInpaint": "flux",
                    "Flux2": "flux",
                }
                for pattern, model_type in config_map.items():
                    if pattern in config_cls:
                        return model_type
        except (AttributeError, RuntimeError):
            pass

        model_sampling = model.get_model_object("model_sampling")
        has_flux_attrs = hasattr(model_sampling, "shift") or hasattr(
            model_sampling, "flux_shift"
        )

        if has_flux_attrs:
            try:
                sigma_max = (
                    model_sampling.sigma_max.item()
                    if hasattr(model_sampling, "sigma_max")
                    else 1.0
                )
                if sigma_max <= 1.5:
                    return "flux"
            except (AttributeError, RuntimeError):
                return "flux"

        return "sd15"

    except (AttributeError, RuntimeError) as e:
        logger.warning(f"Model type detection failed: {e}, defaulting to sd15")
        return "sd15"

def get_model_defaults(model_type: str) -> Dict[str, Any]:

    return MODEL_DEFAULTS.get(model_type, MODEL_DEFAULTS["sd15"])

def gradual_sigma_blend(
    sigmas_a: torch.Tensor, sigmas_b: torch.Tensor, blend_steps: int = 3
) -> torch.Tensor:

    if blend_steps <= 0 or len(sigmas_a) == 0 or len(sigmas_b) == 0:
        return sigmas_b

    result = sigmas_b.clone()

    last_sigma_a = sigmas_a[-1].item()

    blend_steps = min(blend_steps, len(result) - 1)

    result[0] = last_sigma_a

    for i in range(1, blend_steps):

        t = i / blend_steps

        blend_factor = 0.5 * (1.0 - math.cos(math.pi * t))

        result[i] = (
            last_sigma_a * (1.0 - blend_factor) + sigmas_b[i].item() * blend_factor
        )

    logger.debug(
        f"Sigma blend: {blend_steps} steps, "
        f"transition {last_sigma_a:.4f} → {sigmas_b[blend_steps-1].item():.4f}"
    )
    return result

def log_tensor(name: str, tensor: Optional[torch.Tensor]) -> None:

    if tensor is None:
        logger.debug(f"{name}: None")
        return

    if _HAS_NESTED_TENSOR and isinstance(tensor, _NestedTensor):
        try:
            shapes = [tuple(t.shape) for t in tensor.tensors]
            logger.debug(f"{name} (NestedTensor): sub-shapes={shapes}")
        except Exception:
            logger.debug(f"{name}: NestedTensor (shape logging unavailable)")
        return
    try:
        t = tensor.float()
        logger.debug(
            f"{name}: Shape={list(t.shape)} | "
            f"Range=[{t.min().item():.3f}, {t.max().item():.3f}] | "
            f"Mean={t.mean().item():.3f} | Std={t.std().item():.3f}"
        )
    except (RuntimeError, ValueError) as e:
        logger.warning(f"{name}: Error logging stats ({e})")

class SigmaIndexer:

    def __init__(self, total_steps: int, base_sigmas: torch.Tensor):
        self.total_steps = total_steps
        self.base_sigmas = base_sigmas
        self.num_sigmas = len(base_sigmas)

        self.offset = total_steps - (self.num_sigmas - 1)

        assert self.num_sigmas <= total_steps + 1, (
            f"SigmaIndexer: schedule length {self.num_sigmas} exceeds "
            f"steps+1 ({total_steps + 1}). A sigma correction probably "
            f"appended instead of replacing."
        )
        assert self.offset >= 0, (
            f"SigmaIndexer: negative offset {self.offset} — schedule has "
            f"{self.num_sigmas} sigmas for {total_steps} steps"
        )

    def to_local(self, global_step: int) -> int:

        return global_step - self.offset

    def get_stage_sigmas(
        self, global_start: int, global_end: int
    ) -> Optional[torch.Tensor]:

        local_start = self.to_local(global_start)
        local_end = self.to_local(global_end)

        if local_end < 0:
            return None

        safe_start = max(0, min(local_start, self.num_sigmas - 1))
        safe_end = max(0, min(local_end + 1, self.num_sigmas))                

        if safe_start >= safe_end:
            return None

        return self.base_sigmas[safe_start:safe_end]

    def get_sigma_at(self, global_step: int) -> Optional[float]:

        local = self.to_local(global_step)
        if 0 <= local < self.num_sigmas:
            return self.base_sigmas[local].item()
        return None

@dataclass
class SamplingStage:

    index: int                          
    global_start: int                     
    global_end: int                   
    model: Any                         
    sampler_name: str                          
    scheduler_name: str                  
    is_phase_shifted: bool = False                                            
    is_blend_point: bool = False                                            

def apply_flux_guidance(cond: List, guidance_value: float) -> List:

    result = []
    for c in cond:
        new_dict = c[1].copy()                                        
        new_dict["guidance"] = guidance_value
        result.append([c[0], new_dict])
    return result

def compute_dynamic_guidance(
    base_guidance: float,
    step: int,
    total_steps: int,
    denoise: float,
) -> float:

    g_low = base_guidance * DYNAMIC_GUIDANCE_EARLY_MULTIPLIER
    g_high = base_guidance

    denoising_steps = max(
        1, int(total_steps * denoise) if denoise < 1.0 else total_steps
    )
    denoising_start = total_steps - denoising_steps
    progress = max(0.0, min(1.0, (step - denoising_start) / denoising_steps))

    RAMP = DYNAMIC_GUIDANCE_RAMP_WIDTH
    EARLY_T = DYNAMIC_GUIDANCE_EARLY_THRESHOLD
    LATE_T = DYNAMIC_GUIDANCE_LATE_THRESHOLD

    if progress < EARLY_T - RAMP:
        return g_low

    elif progress < EARLY_T + RAMP:
        t = (progress - (EARLY_T - RAMP)) / (2 * RAMP)
        blend = 0.5 * (1.0 - math.cos(math.pi * t))
        return g_low + (g_high - g_low) * blend

    elif progress < LATE_T - RAMP:
        return g_high

    elif progress < LATE_T + RAMP:
        g_late = g_high * DYNAMIC_GUIDANCE_LATE_MULTIPLIER
        t = (progress - (LATE_T - RAMP)) / (2 * RAMP)
        blend = 0.5 * (1.0 - math.cos(math.pi * t))
        return g_high + (g_late - g_high) * blend

    else:
        return g_high * DYNAMIC_GUIDANCE_LATE_MULTIPLIER

def compute_dynamic_cfg(
    base_cfg: float,
    step: int,
    total_steps: int,
    denoise: float,
) -> float:

    denoising_steps = max(1, int(total_steps * denoise) if denoise < 1.0 else total_steps)
    denoising_start = total_steps - denoising_steps
    progress = max(0.0, min(1.0, (step - denoising_start) / denoising_steps))

    RAMP = DYNAMIC_GUIDANCE_RAMP_WIDTH                               
    EARLY_T = DYNAMIC_CFG_EARLY_THRESHOLD
    LATE_T = DYNAMIC_CFG_LATE_THRESHOLD

    cfg_early = base_cfg * DYNAMIC_CFG_EARLY_MULTIPLIER
    cfg_late = base_cfg * DYNAMIC_CFG_LATE_MULTIPLIER

    if progress < EARLY_T - RAMP:
        return cfg_early
    elif progress < EARLY_T + RAMP:
        t = (progress - (EARLY_T - RAMP)) / (2 * RAMP)
        blend = 0.5 * (1.0 - math.cos(math.pi * t))
        return cfg_early + (base_cfg - cfg_early) * blend
    elif progress < LATE_T - RAMP:
        return base_cfg
    elif progress < LATE_T + RAMP:
        t = (progress - (LATE_T - RAMP)) / (2 * RAMP)
        blend = 0.5 * (1.0 - math.cos(math.pi * t))
        return base_cfg + (cfg_late - base_cfg) * blend
    else:
        return cfg_late

def compute_base_sigmas(
    model,
    scheduler_name: str,
    total_steps: int,
    primary_scheduler: str,
    flux_shift: float,
    denoise: float,
    cache: SigmaCache,
) -> torch.Tensor:

    cached = cache.get(model, scheduler_name, total_steps)
    if cached is not None:
        bs = cached
    else:

        ms = model.get_model_object("model_sampling")
        bs = comfy.samplers.calculate_sigmas(ms, scheduler_name, total_steps)

        if flux_shift != 1.0 and scheduler_name != primary_scheduler:
            bs = flux_shift_sigmas(bs, flux_shift)

        bs = correct_sigma_end(bs)

        assert len(bs) == total_steps + 1, (
            f"compute_base_sigmas: schedule length {len(bs)} != steps+1 "
            f"({total_steps + 1}) for scheduler '{scheduler_name}'. "
            f"A sigma correction may have appended instead of replacing."
        )

        cache.put(model, scheduler_name, total_steps, bs)

    if denoise < 1.0:
        n = len(bs) - 1
        if n > 0:
            bs = bs[max(0, int(n * (1.0 - denoise))):]

    return bs

WORKFLOW_PRESETS = [
    "None (Custom)",
    "→ Flux txt2img",
    "→ Flux img2img",
    "→ Flux Inpaint",
    "→ Flux High-Res Fix",
    "→ Flux Fast (12 steps)",
    "→ Flux Quality (28 steps)",
    "→ Flux Cinematic (30 steps)",

    "→ Flux Schnell (4 steps)",
    "→ SD3.5 Turbo (4 steps)",
    "→ Flux Ultra Fast (8 steps)",

    "▶ WAN txt2vid (30 steps)",
    "▶ WAN img2vid (20 steps)",
    "▶ LTX-Video (25 steps)",
    "▶ LTX 2.3 LowRes (32 steps)",                                                
    "▶ LTX 2.3 HighRes (40 steps)",                                                     
    "▶ HunyuanVideo (30 steps)",

    "◈ Draft (4-step / AYS)",
    "◈ Fast (8-step / AYS)",
    "◈ Balanced (20-step)",
    "◈ Quality (35-step)",
    "◈ Cinema (60-step)",

    "◈ z_image (25 steps)",
    "◈ Lumina2 (25 steps)",
]

PRESET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "→ Flux txt2img": {
        "steps": 25,
        "cfg": 1.0,
        "sampler": "euler",
        "scheduler": "simple",
        "denoise": 1.0,
        "flux_shift": 1.0,
        "flux_guidance": 3.5,
    },
    "→ Flux img2img": {
        "steps": 20,
        "cfg": 1.0,
        "sampler": "euler",
        "scheduler": "simple",
        "denoise": 0.75,
        "flux_shift": 1.0,
        "flux_guidance": 3.5,
    },
    "→ Flux Inpaint": {
        "steps": 25,
        "cfg": 1.0,
        "sampler": "euler",
        "scheduler": "simple",
        "denoise": 1.0,
        "flux_shift": 1.0,
        "flux_guidance": 4.0,
    },
    "→ Flux High-Res Fix": {
        "steps": 20,
        "cfg": 1.0,
        "sampler": "euler",
        "scheduler": "simple",
        "denoise": 0.5,
        "flux_shift": 3.0,
        "flux_guidance": 3.5,
    },
    "→ Flux Fast (12 steps)": {
        "steps": 12,
        "cfg": 1.0,
        "sampler": "euler",
        "scheduler": "simple",
        "denoise": 1.0,
        "flux_shift": 1.0,
        "flux_guidance": 3.5,
    },
    "→ Flux Quality (28 steps)": {
        "steps": 28,
        "cfg": 1.0,
        "sampler": "euler",
        "scheduler": "simple",
        "denoise": 1.0,
        "flux_shift": 1.0,
        "flux_guidance": 4.0,
    },
    "→ Flux Cinematic (30 steps)": {
        "steps": 30,
        "cfg": 1.0,
        "sampler": "euler",
        "scheduler": "simple",
        "denoise": 1.0,
        "flux_shift": 1.0,
        "flux_guidance": 4.0,
    },

    "→ Flux Schnell (4 steps)": {
        "steps": 4,
        "cfg": 1.0,
        "sampler": "euler",
        "scheduler": "simple",
        "denoise": 1.0,
        "flux_shift": 1.0,
        "flux_guidance": 0.0,                            
    },
    "→ SD3.5 Turbo (4 steps)": {
        "steps": 4,
        "cfg": 1.6,
        "sampler": "euler",
        "scheduler": "sgm_uniform",
        "denoise": 1.0,
        "flux_shift": 1.0,
        "flux_guidance": 0.0,
    },
    "→ Flux Ultra Fast (8 steps)": {
        "steps": 8,
        "cfg": 1.0,
        "sampler": "euler",
        "scheduler": "simple",
        "denoise": 1.0,
        "flux_shift": 1.0,
        "flux_guidance": 2.0,
    },

    "▶ WAN txt2vid (30 steps)": {
        "steps": 30,
        "cfg": 6.0,
        "sampler": "euler",
        "scheduler": "simple",
        "denoise": 1.0,
        "flux_shift": 8.0,
        "flux_guidance": 0.0,
    },
    "▶ WAN img2vid (20 steps)": {
        "steps": 20,
        "cfg": 6.0,
        "sampler": "euler",
        "scheduler": "simple",
        "denoise": 0.75,
        "flux_shift": 8.0,
        "flux_guidance": 0.0,
    },
    "▶ LTX-Video (25 steps)": {
        "steps": 25,
        "cfg": 1.0,
        "sampler": "euler",
        "scheduler": "simple",
        "denoise": 1.0,
        "flux_shift": 2.37,
        "flux_guidance": 0.0,
    },

    "▶ LTX 2.3 LowRes (32 steps)": {
        "steps": 32,
        "cfg": 3.0,
        "sampler": "euler",
        "scheduler": "beta",
        "denoise": 1.0,
        "flux_shift": 3.0,
        "flux_guidance": 0.0,
        "force_full_denoise_steps": True,
        "force_exact_steps": True,
        "model_type": "ltxav",
    },
    "▶ LTX 2.3 HighRes (40 steps)": {
        "steps": 40,
        "cfg": 3.0,
        "sampler": "euler",
        "scheduler": "beta",
        "denoise": 0.45,
        "flux_shift": 6.0,
        "flux_guidance": 0.0,
        "force_full_denoise_steps": True,
        "force_exact_steps": True,
        "model_type": "ltxav",
    },
    "▶ HunyuanVideo (30 steps)": {
        "steps": 30,
        "cfg": 6.0,
        "sampler": "euler",
        "scheduler": "simple",
        "denoise": 1.0,
        "flux_shift": 7.0,
        "flux_guidance": 0.0,
    },

    "◈ Draft (4-step / AYS)": {
        "steps": 4, "cfg": 1.0, "sampler": "euler", "scheduler": "simple",
        "denoise": 1.0, "flux_shift": 1.0, "flux_guidance": 3.5,
        "ays_schedule": True,
    },
    "◈ Fast (8-step / AYS)": {
        "steps": 8, "cfg": 1.0, "sampler": "euler", "scheduler": "simple",
        "denoise": 1.0, "flux_shift": 1.0, "flux_guidance": 3.5,
        "ays_schedule": True,
    },
    "◈ Balanced (20-step)": {
        "steps": 20, "cfg": 1.0, "sampler": "euler", "scheduler": "simple",
        "denoise": 1.0, "flux_shift": 1.0, "flux_guidance": 3.5,
        "ays_schedule": False,
    },
    "◈ Quality (35-step)": {
        "steps": 35, "cfg": 1.0, "sampler": "euler", "scheduler": "simple",
        "denoise": 1.0, "flux_shift": 1.0, "flux_guidance": 4.0,
        "sampler_mode": "Phase-Shift (Euler\u2192SGM)", "ays_schedule": False,
    },
    "◈ Cinema (60-step)": {
        "steps": 60, "cfg": 1.0, "sampler": "euler", "scheduler": "simple",
        "denoise": 1.0, "flux_shift": 1.5, "flux_guidance": 4.5,
        "sampler_mode": "Phase-Shift (Euler\u2192SGM)", "ays_schedule": False,
    },

    "◈ z_image (25 steps)": {
        "steps": 25,
        "cfg": 1.0,
        "sampler": "euler",
        "scheduler": "simple",
        "denoise": 1.0,
        "flux_shift": 3.0,
        "flux_guidance": 3.5,
        "model_type": "z_image",
    },
    "◈ Lumina2 (25 steps)": {
        "steps": 25,
        "cfg": 1.0,
        "sampler": "euler",
        "scheduler": "simple",
        "denoise": 1.0,
        "flux_shift": 6.0,
        "flux_guidance": 3.5,
        "model_type": "lumina2",
    },
}

def flux_shift_sigmas(sigmas: torch.Tensor, shift: float) -> torch.Tensor:

    if shift <= 0:
        raise ValueError(f"flux_shift must be > 0, got {shift}")

    if shift == 1.0:
        return sigmas

    denominator = 1.0 + (shift - 1.0) * sigmas

    denominator = torch.clamp(denominator, min=1e-6)

    shifted = shift * sigmas / denominator
    return shifted

def get_flux_sigmas(
    model, scheduler: str, steps: int, denoise: float, shift: float = 1.0,
    force_full: bool = False, force_exact: bool = False,
) -> torch.Tensor:

    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if denoise < 0.0 or denoise > 1.0:
        raise ValueError(f"denoise must be in [0.0, 1.0], got {denoise}")
    if denoise <= 0.0:
        return torch.tensor([0.0])

    model_sampling = model.get_model_object("model_sampling")

    if force_exact and 0.0 < denoise < 1.0:

        total_steps = max(1, int(round(steps / denoise)))
    else:
        total_steps = steps

    sigmas = comfy.samplers.calculate_sigmas(model_sampling, scheduler, total_steps)

    if shift != 1.0:
        sigmas = flux_shift_sigmas(sigmas, shift)

    if denoise < 1.0 and not force_full:
        if force_exact:

            sigmas = sigmas[-(steps + 1):]
        else:
            total_s = len(sigmas) - 1
            if total_s > 0:
                start_step = max(0, int(total_s * (1.0 - denoise)))
                sigmas = sigmas[start_step:]

    return sigmas

def validate_step_range(
    start_step: int, end_step: int, steps: int, context: str = ""
) -> Tuple[int, int]:

    original = (start_step, end_step)

    start = max(0, min(start_step, steps))
    end = max(0, min(end_step, steps))

    if start > end:
        logger.warning(f"{context}start_step ({start}) > end_step ({end}), swapping")
        start, end = end, start

    if original != (start, end):
        logger.debug(f"{context}Step range adjusted: {original} → ({start}, {end})")

    return start, end

def apply_pag_to_model(model, pag_scale: float):

    if pag_scale <= 0:
        return model

    try:

        model_pag = model.clone()

        if hasattr(model_pag, "model_options"):
            model_pag.model_options = model_pag.model_options.copy()
            model_pag.model_options["pag_scale"] = pag_scale

        def pag_attention_patch(q, k, v, extra_options):

            cond_or_uncond = extra_options.get("cond_or_uncond", [0])
            block_type = extra_options.get("block_type", "unknown")

            if 1 not in cond_or_uncond or block_type != "middle":
                return q, k, v

            k_out = k.clone()
            v_out = v.clone()

            num_cond = len(cond_or_uncond)
            batch_size = q.shape[0]
            chunk_size = batch_size // num_cond if num_cond > 0 else batch_size

            for idx, cond_type in enumerate(cond_or_uncond):
                if cond_type == 1:                 
                    start = idx * chunk_size
                    end = min(start + chunk_size, batch_size)

                    k_out[start:end] = q[start:end]
                    v_out[start:end] = q[start:end]

            return q, k_out, v_out

        model_pag.set_model_attn1_patch(pag_attention_patch)

        logger.info(f"PAG applied with scale {pag_scale} (attention hook active)")
        return model_pag

    except (AttributeError, RuntimeError, TypeError) as e:

        logger.warning(f"Failed to apply PAG: {e}, using original model")
        return model

AYS_ANCHORS = {
    "sdxl": [14.615, 6.315, 3.771, 1.181, 0.468, 0.131, 0.029, 0.0],
    "sd15": [
        14.615,
        6.475,
        3.861,
        2.697,
        1.886,
        1.396,
        0.963,
        0.652,
        0.399,
        0.152,
        0.029,
        0.0,
    ],

    "flux": [1.0, 0.90, 0.70, 0.45, 0.22, 0.08, 0.02, 0.0],
    "sd3": [14.615, 6.291, 3.438, 1.566, 0.741, 0.288, 0.079, 0.0],

    "wan": [1.0, 0.92, 0.78, 0.58, 0.35, 0.15, 0.04, 0.0],

    "ltxv": [1.0, 0.88, 0.68, 0.42, 0.20, 0.07, 0.015, 0.0],
}

def get_ays_sigmas(model_type: str, steps: int) -> Optional[torch.Tensor]:

    key = model_type if model_type in AYS_ANCHORS else None
    if key is None:

        if model_type in ("sd35",):
            key = "sd3"
        elif model_type in ("chroma",):
            key = "flux"
        elif model_type in ("hunyuan_video",):
            key = "wan"                                      
        elif model_type in ("lumina2", "z_image"):
            key = "flux"                        
        else:
            return None

    if key in ("flux", "wan", "ltxv"):
        logger.info(
            f"AYS for {model_type}: using experimental anchors (not from original paper)"
        )

    anchors = AYS_ANCHORS[key]
    n_anchors = len(anchors)

    if steps + 1 <= n_anchors:

        indices = torch.linspace(0, n_anchors - 1, steps + 1).long()
        return torch.tensor([anchors[i] for i in indices])

    result = torch.zeros(steps + 1)
    anchor_t = torch.tensor(anchors)

    log_anchors = torch.log(torch.clamp(anchor_t[:-1], min=1e-6))                   
    log_anchors = torch.cat([log_anchors, torch.tensor([-12.0])])                 

    x_anchors = torch.linspace(0, 1, len(log_anchors))
    x_result = torch.linspace(0, 1, steps + 1)

    for i in range(steps + 1):
        t = x_result[i].item()

        idx = 0
        while idx < len(x_anchors) - 2 and x_anchors[idx + 1] < t:
            idx += 1

        t_local = (t - x_anchors[idx].item()) / max(
            1e-8, (x_anchors[idx + 1] - x_anchors[idx]).item()
        )
        t_local = max(0.0, min(1.0, t_local))

        log_val = log_anchors[idx] * (1 - t_local) + log_anchors[idx + 1] * t_local
        result[i] = math.exp(log_val.item())

    result[-1] = 0.0

    return result

def guidance_rescale_cfg(
    cond_output: torch.Tensor,
    uncond_output: torch.Tensor,
    cfg: float,
    phi: float = 0.7,
) -> torch.Tensor:

    if phi <= 0.0 or cfg <= 1.0:
        return uncond_output + cfg * (cond_output - uncond_output)

    guided = uncond_output + cfg * (cond_output - uncond_output)

    dims = list(range(1, guided.ndim))                         
    guided_std = guided.std(dim=dims, keepdim=True) + 1e-6
    cond_std = cond_output.std(dim=dims, keepdim=True) + 1e-6

    rescaled = guided * (cond_std / guided_std)

    return rescaled * phi + guided * (1.0 - phi)

def correct_sigma_end(
    sigmas: torch.Tensor, target_end: float = 0.0
) -> torch.Tensor:

    if len(sigmas) < 2:
        return sigmas

    if sigmas[-1].item() > 0.0 and target_end == 0.0:

        result = sigmas.clone()
        result[-1] = target_end
        return result

    return sigmas

def apply_cfg_plus_plus(cfg: float, sigma: torch.Tensor, sigma_max: float) -> float:

    if sigma_max <= 0:
        return cfg

    if isinstance(sigma, torch.Tensor):
        sigma_val = sigma.item() if sigma.numel() == 1 else sigma[0].item()
    else:
        sigma_val = float(sigma)

    progress = 1.0 - (sigma_val / sigma_max)
    progress = max(0.0, min(1.0, progress))

    cos_factor = (1.0 + math.cos(math.pi * progress)) / 2.0

    effective_cfg = cfg * cos_factor + 1.0 * (1.0 - cos_factor)

    return effective_cfg

def build_sigma_report(
    detected_type: str,
    steps: int,
    scheduler: str,
    flux_shift: float,
    denoise: float,
    sigmas: torch.Tensor,
    sampler_mode: str,
    sorted_splits: List[int],
    stage_timings: List[Tuple[int, int, int, str, float]],
    total_time: float,
    ays_active: bool = False,
    frames: Optional[int] = None,
) -> str:

    video_tag = f" | Frames: {frames}" if frames is not None and frames > 1 else ""
    lines = [
        f"═══ Radiance Sampler Pro v4.2 ═══",
        f"Model: {detected_type} | Steps: {steps} | Scheduler: {scheduler}{'  [AYS]' if ays_active else ''}{video_tag}",
        f"Shift: {flux_shift} | Denoise: {denoise} | Mode: {sampler_mode}",
    ]

    if len(sigmas) > 0:
        lines.append(
            f"Sigma range: [{sigmas[0].item():.4f} → {sigmas[-1].item():.4f}] "
            f"({len(sigmas)} values)"
        )

    lines.append(f"Stages: {max(0, len(sorted_splits) - 1)}")

    if stage_timings:
        lines.append("─── Per-Stage Timing ───")
        for stage_num, s_start, s_end, samp, t in stage_timings:
            stage_steps = s_end - s_start
            speed = stage_steps / t if t > 0.001 else 0
            lines.append(
                f"  Stage {stage_num}: steps {s_start}→{s_end} [{samp}] "
                f"= {t:.3f}s ({speed:.1f} it/s)"
            )

    total_sampling_steps = (
        sum(s_end - s_start for _, s_start, s_end, _, _ in stage_timings)
        if stage_timings
        else steps
    )
    overall_speed = total_sampling_steps / total_time if total_time > 0.001 else 0
    lines.append(f"Total: {total_time:.2f}s ({overall_speed:.1f} it/s)")

    return "\n".join(lines)

def _temporally_correlate(
    noise_fn, shape: tuple, device: torch.device, alpha: float = 0.6,
    seed: Optional[int] = None,
) -> torch.Tensor:

    B, C, T, H, W = shape
    frame_shape = (B, C, H, W)

    frames = []
    if seed is not None:
        torch.manual_seed(seed)
    prev = noise_fn(frame_shape, device)
    for f in range(T):
        if seed is not None:
            torch.manual_seed(seed + f + 1)                                   
        curr = noise_fn(frame_shape, device)
        blended = alpha * prev + math.sqrt(1 - alpha ** 2) * curr
        frames.append(blended)
        prev = blended

    result = torch.stack(frames, dim=2)                     

    result = result - result.mean()
    std = result.std().clamp(min=1e-6)
    return result / std

def _perlin_noise(shape: tuple, device: torch.device, seed: Optional[int] = None) -> torch.Tensor:

    if len(shape) == 5 and shape[2] > 1:
        return _temporally_correlate(_perlin_noise_2d, shape, device, seed=seed)

    return _perlin_noise_2d(shape, device)

def _perlin_noise_2d(shape: tuple, device: torch.device) -> torch.Tensor:

    noise = torch.zeros(shape, device=device)
    amplitude = 1.0
    frequency = 1.0
    spatial_h, spatial_w = shape[-2], shape[-1]

    for _ in range(4):             
        n = torch.randn(shape, device=device) * amplitude

        if spatial_w > 1:
            kernel = max(1, int(spatial_w / frequency))

            leading = n.shape[:-2]
            n_flat = n.reshape(-1, 1, spatial_h, spatial_w)

            pad = kernel // 2
            pooled = torch.nn.functional.avg_pool2d(
                n_flat, kernel_size=kernel, stride=1, padding=pad
            )

            pooled = pooled[..., :spatial_h, :spatial_w]

            n = pooled.reshape(*leading, spatial_h, spatial_w)

        noise = noise + n
        amplitude *= 0.5
        frequency *= 2.0

    noise = noise - noise.mean()
    std = noise.std().clamp(min=1e-6)
    return noise / std

def _spectral_noise(shape: tuple, device: torch.device, seed: Optional[int] = None) -> torch.Tensor:

    if len(shape) == 5 and shape[2] > 1:
        return _temporally_correlate(_spectral_noise_2d, shape, device, seed=seed)

    return _spectral_noise_2d(shape, device)

def _spectral_noise_2d(shape: tuple, device: torch.device) -> torch.Tensor:

    white = torch.randn(shape, device=device)
    fft = torch.fft.rfft2(white)

    freqs_h = torch.fft.fftfreq(shape[-2], device=device).abs()
    freqs_w = torch.fft.rfftfreq(shape[-1], device=device).abs()
    freq_grid = torch.sqrt(
        freqs_h.unsqueeze(-1) ** 2 + freqs_w.unsqueeze(0) ** 2
    ).clamp(min=1e-6)
    weight = 1.0 / freq_grid                      

    weight[0, 0] = 0.0

    for _ in range(fft.ndim - 2):
        weight = weight.unsqueeze(0)

    filtered = fft * weight
    result = torch.fft.irfft2(filtered, s=(shape[-2], shape[-1]))

    result = result - result.mean()
    std = result.std().clamp(min=1e-6)
    return result / std

def _brownian_noise(
    shape: tuple, device: torch.device, frames: Optional[int] = None,
    seed: Optional[int] = None,
) -> torch.Tensor:

    if len(shape) == 5 and shape[2] > 1:
        B, C, T, H, W = shape
        alpha = 0.7                                          

        frame_shape = (B, C, H, W)
        noises = []
        if seed is not None:
            torch.manual_seed(seed)
        prev = torch.randn(frame_shape, device=device)
        for f in range(T):
            if seed is not None:
                torch.manual_seed(seed + f + 1)
            curr = alpha * prev + math.sqrt(1 - alpha ** 2) * torch.randn(frame_shape, device=device)
            noises.append(curr)
            prev = curr

        return torch.stack(noises, dim=2)

    if frames is None or (len(shape) == 4 and shape[0] == 1):
        return _spectral_noise(shape, device)

    alpha = 0.7
    noises = []
    prev = torch.randn(shape[1:], device=device)
    for _ in range(shape[0]):
        curr = alpha * prev + math.sqrt(1 - alpha ** 2) * torch.randn(shape[1:], device=device)
        noises.append(curr)
        prev = curr
    return torch.stack(noises, dim=0)

def generate_noise(
    latent_samples: torch.Tensor,
    seed: int,
    noise_type: str = "Gaussian",
    frames: Optional[int] = None,
) -> torch.Tensor:

    shape = latent_samples.shape
    device = latent_samples.device
    dtype = latent_samples.dtype

    is_video = len(shape) == 5 and shape[2] > 1
    if is_video and noise_type in ("Gaussian", "Uniform"):
        B, C, T, H, W = shape
        frame_shape = (B, C, H, W)
        frame_noises = []
        for f in range(T):
            torch.manual_seed(seed + f)
            if noise_type == "Gaussian":
                frame_noises.append(torch.randn(frame_shape, device=device, dtype=dtype))
            else:           
                frame_noises.append(
                    (torch.rand(frame_shape, device=device, dtype=dtype) * 2 - 1) * (3 ** 0.5)
                )
        return torch.stack(frame_noises, dim=2)                     

    torch.manual_seed(seed)

    try:
        if noise_type == "Gaussian":
            return torch.randn(shape, device=device, dtype=dtype)
        elif noise_type == "Uniform":

            return (torch.rand(shape, device=device, dtype=dtype) * 2 - 1) * (3 ** 0.5)
        elif noise_type == "Perlin":
            return _perlin_noise(shape, device, seed=seed).to(dtype)
        elif noise_type == "Spectral":
            return _spectral_noise(shape, device, seed=seed).to(dtype)
        elif noise_type == "Brownian":
            return _brownian_noise(shape, device, frames=frames, seed=seed).to(dtype)
    except Exception as e:
        logger.warning(f"[Noise] Failed to generate {noise_type} noise ({e}), falling back to Gaussian")

    return torch.randn(shape, device=device, dtype=dtype)

def merge_conditionings(
    cond_a: List,
    cond_b: List,
    mode: str = "average",
    weight_b: float = 0.5,
    split_step: int = 0,
    total_steps: int = 20,
) -> List:

    if not cond_b or mode == "Off":
        return cond_a
    if mode in ("average", "weighted"):
        w_b = weight_b if mode == "weighted" else 0.5
        w_a = 1.0 - w_b
        merged = []
        max_len = max(len(cond_a), len(cond_b))
        for i in range(max_len):
            a = cond_a[i % len(cond_a)]
            b = cond_b[i % len(cond_b)]
            tensor_a = a[0]
            tensor_b = b[0]

            if tensor_a.shape[1] != tensor_b.shape[1]:
                target_len = max(tensor_a.shape[1], tensor_b.shape[1])
                def _pad(t, n):
                    if t.shape[1] < n:
                        pad = t[:, -1:, :].expand(-1, n - t.shape[1], -1)
                        return torch.cat([t, pad], dim=1)
                    return t
                tensor_a = _pad(tensor_a, target_len)
                tensor_b = _pad(tensor_b, target_len)
            blended = tensor_a * w_a + tensor_b * w_b

            pooled_a = a[1].get("pooled_output", None) if len(a) > 1 else None
            pooled_b = b[1].get("pooled_output", None) if len(b) > 1 else None
            extra = dict(a[1]) if len(a) > 1 else {}
            if pooled_a is not None and pooled_b is not None:
                try:
                    extra["pooled_output"] = pooled_a * w_a + pooled_b * w_b
                except Exception:
                    pass                                              
            merged.append([blended, extra])
        return merged
    elif mode == "sequential":

        frac = float(split_step) / max(1, total_steps)
        cond_a_timed = []
        for c in cond_a:
            entry = [c[0], dict(c[1]) if len(c) > 1 else {}]
            entry[1]["start_percent"] = 0.0
            entry[1]["end_percent"] = frac
            cond_a_timed.append(entry)
        cond_b_timed = []
        for c in cond_b:
            entry = [c[0], dict(c[1]) if len(c) > 1 else {}]
            entry[1]["start_percent"] = frac
            entry[1]["end_percent"] = 1.0
            cond_b_timed.append(entry)
        return cond_a_timed + cond_b_timed
    return cond_a

def route_conditioning(cond: List, target_key: str) -> List:

    if not target_key or target_key == "Auto":
        return cond
    routed = []
    for entry in cond:
        new_entry = [entry[0], dict(entry[1]) if len(entry) > 1 else {}]
        new_entry[1]["encoder_target"] = target_key
        routed.append(new_entry)
    return routed

def tile_sample(
    model,
    noise: torch.Tensor,
    latent_samples: torch.Tensor,
    positive: List,
    negative: List,
    sigmas: torch.Tensor,
    sampler_obj,
    seed: int,
    tile_size: int = 128,                        
    tile_overlap: int = 16,                      
    tile_blend: str = "feather",
    noise_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if latent_samples.ndim != 4:
        raise ValueError(
            f"[Radiance] tile_sample requires 4D latent (B, C, H, W), "
            f"got {latent_samples.ndim}D. Video tile sampling is not supported."
        )

    B, C, H, W = latent_samples.shape
    step = max(1, tile_size - tile_overlap)
    device = latent_samples.device

    output = torch.zeros_like(latent_samples)
    weight = torch.zeros((B, 1, H, W), device=device, dtype=latent_samples.dtype)

    tile_coords = []
    for y in range(0, H, step):
        for x in range(0, W, step):
            y1 = min(y, H - tile_size) if H > tile_size else 0
            x1 = min(x, W - tile_size) if W > tile_size else 0
            y2 = min(y1 + tile_size, H)
            x2 = min(x1 + tile_size, W)
            tile_coords.append((y1, y2, x1, x2))

    tile_coords = list(dict.fromkeys(tile_coords))

    for idx, (y1, y2, x1, x2) in enumerate(tile_coords):
        t_latent = latent_samples[:, :, y1:y2, x1:x2]
        t_noise = noise[:, :, y1:y2, x1:x2]

        try:
            t_out = _sample_custom_compat(
                model=model,
                noise=t_noise,
                cfg=1.0,
                sampler=sampler_obj,
                sigmas=sigmas,
                positive=positive,
                negative=negative,
                latent_image=t_latent,
                noise_mask=noise_mask[:, :, y1:y2, x1:x2] if noise_mask is not None else None,
                callback=None,
                disable_pbar=True,
                seed=seed + idx,
            )
        except Exception as e:
            logger.warning(f"[TileSample] Tile ({y1},{y2},{x1},{x2}) failed: {e} — using input")
            t_out = t_latent

        if t_out.device != device or t_out.dtype != latent_samples.dtype:
            t_out = t_out.to(device=device, dtype=latent_samples.dtype)

        th = y2 - y1
        tw = x2 - x1

        if tile_blend == "feather":

            wy = torch.ones(th, device=device)
            wx = torch.ones(tw, device=device)
            fade = min(tile_overlap, th // 2, tw // 2)
            if fade > 0:
                ramp = (1 - torch.cos(torch.linspace(0, math.pi, fade, device=device))) / 2
                wy[:fade] = ramp
                wy[-fade:] = ramp.flip(0)
                wx[:fade] = ramp
                wx[-fade:] = ramp.flip(0)
            w_tile = (wy.unsqueeze(1) * wx.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
        elif tile_blend == "gaussian":
            sigma_h = th / 4.0
            sigma_w = tw / 4.0
            yg = torch.arange(th, device=device).float() - th / 2
            xg = torch.arange(tw, device=device).float() - tw / 2
            w_tile = torch.exp(-(yg.unsqueeze(1)**2 / (2*sigma_h**2) +
                                  xg.unsqueeze(0)**2 / (2*sigma_w**2)))
            w_tile = w_tile.unsqueeze(0).unsqueeze(0)
        else:                                                       
            w_tile = torch.ones((1, 1, th, tw), device=device, dtype=latent_samples.dtype)

        output[:, :, y1:y2, x1:x2] += t_out * w_tile
        weight[:, :, y1:y2, x1:x2] += w_tile

    weight = weight.clamp(min=1e-6)
    output = output / weight

    return output

def _build_latent_meta(
    detected_type: str,
    steps: int,
    scheduler: str,
    flux_shift: float,
    denoise: float,
    sigmas: torch.Tensor,
    ays_active: bool,
    pag_active: bool,
    noise_type: str,
    tile_mode: bool,
    multi_cond_mode: str,
    clip_target: str,
    seed: int,
    total_time_ms: int,
    latent_format: str = "",
    frames: Optional[int] = None,
) -> str:

    import json
    sigma_min = float(sigmas[sigmas > 0].min()) if (sigmas > 0).any() else 0.0
    sigma_max = float(sigmas.max())
    is_video = frames is not None and frames > 1
    meta = {
        "version": "4.2",
        "detected_arch": detected_type,
        "is_video": is_video,
        "frames": frames if is_video else None,
        "steps": steps,
        "scheduler": scheduler,
        "flux_shift": flux_shift,
        "denoise": denoise,
        "sigma_min": round(sigma_min, 5),
        "sigma_max": round(sigma_max, 5),
        "ays_active": ays_active,
        "pag_active": pag_active,
        "noise_type": noise_type,
        "tile_mode": tile_mode,
        "multi_cond_mode": multi_cond_mode,
        "clip_target": clip_target,
        "latent_format": latent_format or "unknown",
        "seed": seed,
        "time_ms": total_time_ms,
    }
    return json.dumps(meta, indent=2)

class RadianceSamplerPro:

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "preset": (WORKFLOW_PRESETS, {"default": "None (Custom)"}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 200, "step": 1}),
                "start_step": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 200,
                        "step": 1,
                        "tooltip": "Start step (0 = beginning)",
                    },
                ),
                "end_step": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 200,
                        "step": 1,
                        "tooltip": "End step (0 = use total steps)",
                    },
                ),
                "cfg": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1},
                ),
                "sampler": (comfy.samplers.KSampler.SAMPLERS,),
                "sampler_mode": (SamplerMode.ALL, {"default": SamplerMode.STANDARD}),
                "phase_split": (
                    "FLOAT",
                    {"default": 0.40, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "scheduler_mode": (
                    ["Manual", "Auto (Match Steps)"],
                    {"default": "Manual"},
                ),
                "denoise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "flux_shift": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.1},
                ),
                "flux_guidance": (
                    "FLOAT",
                    {"default": 3.5, "min": 0.0, "max": 20.0, "step": 0.1},
                ),
                "flux_guidance_profile": (
                    ["Static", "Dynamic (Creative Start/End)"],
                    {"default": "Static"},
                ),
                "add_noise": ("BOOLEAN", {"default": True}),
                "return_with_leftover_noise": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),

                "pag_scale": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 5.0,
                        "step": 0.1,
                        "tooltip": "PAG strength (0=off). Perturbs attention for better prompt adherence.",
                    },
                ),

                "model_type": (MODEL_TYPES, {"default": "auto"}),
                "sigma_blend_steps": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 10,
                        "step": 1,
                        "tooltip": "Smooth sigma transition steps at phase-shift boundary",
                    },
                ),

                "ays_schedule": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Use AYS (Align Your Steps) research-optimized sigma schedule. Best at 8-15 steps.",
                    },
                ),
                "guidance_rescale_phi": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Guidance rescale (Imagen). 0=off, 0.7=recommended for SDXL. Prevents oversaturation at high CFG.",
                    },
                ),

                "preview_method": (PREVIEW_METHODS, {"default": "None"}),

                "noise_type": (
                    NOISE_TYPES,
                    {"default": "Gaussian",
                     "tooltip": ("Noise generation algorithm. Perlin=coherent structure, "
                                 "Spectral=pink/1f noise, Brownian=video-correlated, Uniform=flat distribution.")},
                ),

                "multi_cond_mode": (
                    MULTI_COND_MODES,
                    {"default": "Off",
                     "tooltip": "Merge positive + positive_2 conditioning. average/weighted blend tensors; sequential splits by step range."},
                ),
                "cond_weight_b": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                     "tooltip": "Weight of positive_2 conditioning when mode='weighted'."},
                ),

                "conditioning_clip_target": (
                    CLIP_TARGETS,
                    {"default": "Auto",
                     "tooltip": "Route conditioning to a specific encoder slot (clip_l, clip_g, t5xxl). Auto = no routing."},
                ),

                "tile_mode": (
                    "BOOLEAN",
                    {"default": False,
                     "tooltip": "Enable tiled sampling for memory-efficient high-resolution generation."},
                ),
                "tile_size": (
                    "INT",
                    {"default": 128, "min": 32, "max": 1024, "step": 32,
                     "tooltip": "Tile size in latent pixels (128 latent ≈ 1024px output with VAE factor 8)."},
                ),
                "tile_overlap": (
                    "INT",
                    {"default": 16, "min": 0, "max": 256, "step": 8,
                     "tooltip": "Overlap between adjacent tiles to reduce seam artifacts."},
                ),
                "tile_blend": (
                    TILE_BLEND_MODES,
                    {"default": "feather",
                     "tooltip": "Seam blending method. feather=cosine fade, gaussian=bell curve, average=uniform."},
                ),

                "force_full_denoise_steps": (
                    "BOOLEAN",
                    {"default": False,
                     "tooltip": ""},
                ),
                "force_exact_steps": (
                    "BOOLEAN",
                    {"default": False,
                     "tooltip": ""},
                ),
                "video_normalization_factors": (
                    "STRING",
                    {"default": "1,1,1,1,1,1,1,1",
                     "tooltip": ""},
                ),
                "audio_normalization_factors": (
                    "STRING",
                    {"default": "1,1,0.25,1,1,0.25,1,1",
                     "tooltip": ""},
                ),
            },
            "optional": {
                "refiner_model": ("MODEL",),
                "refiner_start_step": (
                    "INT",
                    {"default": 20, "min": 0, "max": 200, "step": 1},
                ),
                "noise_override": ("LATENT",),

                "sigmas_override": (
                    "SIGMAS",
                    {"tooltip": "Inject a pre-computed sigma schedule. Bypasses all internal sigma computation."},
                ),

                "latent_format": (
                    "STRING",
                    {"default": "",
                     "tooltip": "Wire from Radiance Loader latent_format output. Used for channel validation."},
                ),

                "positive_2": (
                    "CONDITIONING",
                    {"tooltip": "Secondary positive conditioning. Merged with primary via multi_cond_mode."},
                ),
            },
        }

    RETURN_TYPES = ("LATENT", "SIGMAS", "STRING", "STRING")
    RETURN_NAMES = ("latent", "sigmas", "sigma_report", "latent_meta")
    OUTPUT_TOOLTIPS = (
        "Denoised latent ready for VAE decode.",
        "The sigma schedule used — chain to another sampler or inspect.",
        "Human-readable timing and schedule report.",
        "JSON telemetry: arch, steps, scheduler, noise_type, tile_mode, seed, time_ms, etc.",
    )
    FUNCTION = "sample"
    CATEGORY = "FXTD Studios/Radiance/Generate"
    DESCRIPTION = (
        "v4.2 — Universal diffusion sampler. Auto-detects model type (Flux, SD3, SDXL, "
        "WAN, LTX, HunyuanVideo, Lumina2, Chroma). Phase-shift sampling, AYS schedules, "
        "PAG, dynamic guidance, tiled sampling, multi-conditioning, noise types, refiner chain."
    )

    def sample(
        self,
        model,
        positive: List,
        negative: List,
        latent_image: Dict[str, torch.Tensor],
        preset: str,
        steps: int,
        start_step: int,
        end_step: int,
        cfg: float,
        sampler: str,
        sampler_mode: str,
        phase_split: float,
        scheduler: str,
        scheduler_mode: str,
        denoise: float,
        flux_shift: float,
        flux_guidance: float,
        flux_guidance_profile: str,
        add_noise: bool,
        return_with_leftover_noise: bool,
        seed: int,
        pag_scale: float = 0.0,

        model_type: str = "auto",
        sigma_blend_steps: int = 0,

        ays_schedule: bool = False,
        guidance_rescale_phi: float = 0.0,

        preview_method: str = "None",

        noise_type: str = "Gaussian",
        multi_cond_mode: str = "Off",
        cond_weight_b: float = 0.5,
        conditioning_clip_target: str = "Auto",
        tile_mode: bool = False,
        tile_size: int = 128,
        tile_overlap: int = 16,
        tile_blend: str = "feather",

        refiner_model=None,
        refiner_start_step: int = 20,
        noise_override: Optional[Dict[str, torch.Tensor]] = None,
        sigmas_override: Optional[torch.Tensor] = None,
        latent_format: str = "",
        positive_2: Optional[List] = None,

        force_full_denoise_steps: bool = False,
        force_exact_steps: bool = False,
        video_normalization_factors: str = "1,1,1,1,1,1,1,1",
        audio_normalization_factors: str = "1,1,0.25,1,1,0.25,1,1",
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, str, str]:

        # ALBABIT-FIX: Console alert for sigmas_override
        if sigmas_override is not None:
            print("\033[93m[Radiance] sigmas_override connected. UI parameters for Steps, Denoise, Scheduler, and Shift will be IGNORED. The sampler will strictly follow the external sigmas schedule.\033[0m")
            logger.warning("[Radiance] sigmas_override is active and will take precedence.")

        t_start = time.time()
        timings: Dict[str, float] = {}

        is_ltx_av = False
        ltxav_obj = None
        try:
            inner_model = model.get_model_object("diffusion_model")
            if inner_model.__class__.__name__ == "LTXAVModel":
                is_ltx_av = True
                ltxav_obj = inner_model
                logger.info(
                    "[Radiance] LTX-AV detected. Volume adjustment will be applied at the end of the phase."
                )
        except Exception:
            pass                                      

        try:
            v_norm = [float(f.strip()) for f in video_normalization_factors.split(",")]
        except ValueError:
            logger.warning("[Radiance] video_normalization_factors parse failed — using 1.0")
            v_norm = [1.0]
        try:
            a_norm = [float(f.strip()) for f in audio_normalization_factors.split(",")]
        except ValueError:
            logger.warning("[Radiance] audio_normalization_factors parse failed — using 0.25")
            a_norm = [0.25]

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if preset != "None (Custom)":
            logger.info(f"[Radiance] Preset '{preset}' active but relying strictly on UI parameters.")

        flux_shift = max(0.01, flux_shift)

        detected_type = detect_model_type(model) if model_type == "auto" else model_type

        if preset == "None (Custom)" and model_type == "auto":
            defaults = get_model_defaults(detected_type)

            if cfg == 1.0 and detected_type != "flux":
                cfg = defaults.get("cfg", cfg)

            if flux_guidance == 3.5 and detected_type != "flux":
                model_default_guidance = defaults.get("guidance", flux_guidance)
                if model_default_guidance > 0 or defaults.get("guidance_type") != "embedding":
                    flux_guidance = model_default_guidance

            default_shift = defaults.get("shift", 1.0)
            if flux_shift == 1.0 and default_shift != 1.0:
                flux_shift = default_shift
                logger.info(
                    f"Auto-applied shift={flux_shift} for {detected_type} "
                    f"(widget was at default 1.0)"
                )

            default_sampler = defaults.get("sampler", sampler)
            if sampler == "euler" and default_sampler != "euler":
                sampler = default_sampler
                logger.info(
                    f"Auto-applied sampler={sampler} for {detected_type}"
                )

            logger.info(
                f"Auto-detected model type: {detected_type} (CFG={cfg}, guidance={flux_guidance}, "
                f"shift={flux_shift}, scheduler={scheduler})"
            )
        else:
            logger.info(f"Model type: {detected_type}")

        if scheduler_mode == "Auto (Match Steps)":
            defaults = get_model_defaults(detected_type)
            auto_scheduler = defaults.get("scheduler", scheduler)
            if auto_scheduler != scheduler:
                logger.info(
                    f"Auto scheduler: {scheduler} → {auto_scheduler} (optimal for {detected_type})"
                )
                scheduler = auto_scheduler

        if is_ltx_av or detected_type == "ltxav":

            if scheduler == "karras":
                logger.warning(
                    "[Radiance] Karras scheduler is designed for classic diffusion models (SD1.5/SDXL). LTX 2.3 uses Flow Matching which requires linear or beta distributions. Karras may cause artifacts or washed-out results."
                )
            elif scheduler == "exponential":
                logger.warning(
                    "[Radiance] Exponential scheduler concentrates steps at the end of the generation. With LTX 2.3, this often leads to blurry or temporally unstable video results."
                )
            if sampler == "uni_pc" and scheduler in ("sgm_uniform", "ddim_uniform"):
                logger.warning(
                    f"[Radiance] LTX 2.3: uni_pc + {scheduler} can cause severe structural "
                    f"instability or 'melting' artifacts in motion. Proceed with caution."
                )
            if sampler == "euler_cfg_pp" and force_full_denoise_steps:
                logger.warning(
                    "[Radiance] 'euler_cfg_pp' sampler is extremely aggressive when 'force_full_denoise_steps' is True. It is highly recommended to disable 'force_full_denoise_steps' or use 'euler' instead."
                )

            if noise_type.lower() not in ("gaussian", "uniform"):
                logger.warning(
                    f"[Radiance] LTX 2.3: '{noise_type}' noise type produces abstract "
                    f"results. Forcing to Gaussian."
                )
                noise_type = "Gaussian"

            if tile_mode:
                logger.warning(
                    "[Radiance] Tile Sampling is incompatible with Video models. Forcing to False."
                )
                tile_mode = False

            if preview_method != "None":
                logger.warning(
                    "[Radiance] Preview Method is incompatible with Video models. Forcing to None."
                )
                preview_method = "None"

        start_step, end_step = validate_step_range(
            start_step, end_step, steps, "[Radiance] "
        )

        t0 = time.time()
        latent = latent_image
        latent_samples = latent["samples"]

        if latent_samples.ndim == 5:
            batch_size, channels, frames, height, width = latent_samples.shape
        else:
            batch_size, channels, height, width = latent_samples.shape
            frames = None
        timings["latent_copy"] = time.time() - t0

        if detected_type in VIDEO_MODEL_TYPES and latent_samples.ndim == 4:
            b, c, h, w = latent_samples.shape
            latent_samples = latent_samples.unsqueeze(2)                           
            frames = 1
            logger.warning(
                f"[Radiance] Video model '{detected_type}' received a 4D latent "
                f"[{b},{c},{h},{w}] — auto-reshaped to 5D [{b},{c},1,{h},{w}] "
                f"(frames=1). For proper video generation, use a video latent node "
                f"(e.g., WAN EmptyLatentVideo) with the desired frame count."
            )

        if latent_format:

            fmt_lower = latent_format.lower()
            expected_ch = 16 if any(k in fmt_lower for k in (
                "flux", "sd3", "16ch", "wan", "ltx", "hunyuan",
                "z_image", "lumina",                                               
            )) else 4
            if channels != expected_ch:
                logger.warning(
                    f"[v4.0] Latent has {channels} channels but latent_format='{latent_format}' "
                    f"suggests {expected_ch}ch. Check VAE/model mismatch."
                )

        t0 = time.time()
        device = comfy.model_management.get_torch_device()
        noise_mask = latent.get("noise_mask", None)

        if noise_override is not None:
            noise = noise_override["samples"]
            expected_shape = latent_samples.shape
            if noise.shape != expected_shape:
                raise ValueError(
                    f"noise_override shape {noise.shape} does not match "
                    f"latent shape {expected_shape}"
                )
        else:

            if noise_type == "Gaussian":
                noise = comfy.sample.prepare_noise(latent_samples, seed, None)
            else:
                noise = generate_noise(latent_samples, seed, noise_type, frames=frames)
                logger.info(f"[v4.0] Using {noise_type} noise generator")

        noise = noise.to(device)
        timings["prepare_noise"] = time.time() - t0

        if latent_samples.ndim == 5 and noise.ndim == 4:
            noise = noise.unsqueeze(2)
            logger.debug("[Radiance] Auto-reshaped noise 4D→5D to match video latent.")
        if latent_samples.ndim == 5 and noise_mask is not None and noise_mask.ndim == 4:
            noise_mask = noise_mask.unsqueeze(2)
            logger.debug("[Radiance] Auto-reshaped noise_mask 4D→5D to match video latent.")

        t0 = time.time()
        if sigmas_override is not None:
            sigmas = sigmas_override.to(device if hasattr(device, '__str__') else "cpu")
            logger.info(
                f"[v4.0] Using external sigmas_override ({len(sigmas)} values, "
                f"range [{sigmas[0].item():.4f} \u2192 {sigmas[-1].item():.4f}])"
            )

            sigmas = correct_sigma_end(sigmas)
        else:
            try:
                if ays_schedule:

                    ays_sigmas = get_ays_sigmas(detected_type, steps)
                    if ays_sigmas is not None:
                        sigmas = ays_sigmas

                        if denoise < 1.0:
                            total_s = len(sigmas) - 1
                            if total_s > 0:
                                start_s = max(0, int(total_s * (1.0 - denoise)))
                                sigmas = sigmas[start_s:]
                        logger.info(
                            f"Using AYS schedule for {detected_type} ({steps} steps)"
                        )
                    else:
                        logger.info(
                            f"AYS not available for {detected_type}, using standard schedule"
                        )
                        sigmas = get_flux_sigmas(
                            model, scheduler, steps, denoise, flux_shift,
                            force_full=force_full_denoise_steps,
                            force_exact=force_exact_steps,
                        )
                else:
                    sigmas = get_flux_sigmas(
                        model, scheduler, steps, denoise, flux_shift,
                        force_full=force_full_denoise_steps,
                        force_exact=force_exact_steps,
                    )

                sigmas = correct_sigma_end(sigmas)

            except ValueError as e:
                logger.error(f"Failed to calculate sigmas: {e}")
                raise
        timings["sigma_calc"] = time.time() - t0

        log_tensor("Sigmas", sigmas)

        if len(sigmas) <= 1:
            logger.warning("[Radiance] Sigma schedule is trivial, returning input unchanged")
            report = build_sigma_report(
                detected_type,
                steps,
                scheduler,
                flux_shift,
                denoise,
                sigmas,
                sampler_mode,
                [],
                [],
                0.0,
                frames=frames,
            )
            return (latent_image.copy(), sigmas, report, "{}")

        work_latent = latent_samples.to(device)
        log_tensor("Work Latent (Start)", work_latent)

        # ALBABIT-FIX: Sync total_steps to sigmas_override if provided to prevent indexer mismatch
        if sigmas_override is not None:
            total_steps = max(1, len(sigmas) - 1)
        else:
            total_steps = max(1, steps)

        if multi_cond_mode != "Off" and positive_2 is not None:
            positive = merge_conditionings(
                positive, positive_2,
                mode=multi_cond_mode,
                weight_b=cond_weight_b,
                split_step=int(total_steps * phase_split),
                total_steps=total_steps,
            )
            logger.info(f"[v4.0] Conditionings merged (mode={multi_cond_mode}, weight_b={cond_weight_b:.2f})")

        if conditioning_clip_target != "Auto":
            positive = route_conditioning(positive, conditioning_clip_target)
            logger.info(f"[v4.0] Conditioning routed to {conditioning_clip_target}")

        primary_sampler = sampler
        secondary_sampler: Optional[str] = None

        secondary_scheduler: Optional[str] = None
        split_step = -1

        effective_end = end_step if end_step > 0 else total_steps
        effective_end = min(effective_end, total_steps)
        effective_start = min(start_step, effective_end)

        splits = {effective_start, effective_end}

        is_cfg_plus_plus = SamplerMode.is_cfg_plus_plus(sampler_mode)
        if is_cfg_plus_plus:
            logger.info(f"CFG++ (Perpendicular) mode active with base CFG {cfg}")

            if detected_type == "flux" and cfg <= 1.05:
                logger.warning(
                    f"CFG++ enabled but CFG is {cfg}. For Flux, CFG++ requires CFG > 1.0"
                )

        if pag_scale > 0:
            model = apply_pag_to_model(model, pag_scale)

        if guidance_rescale_phi > 0.0 and cfg > 1.0:
            phi = guidance_rescale_phi
            model = (
                model.clone() if pag_scale <= 0 else model
            )                                

            def guidance_rescale_patch(args):

                cond = args["cond_denoised"]
                uncond = args["uncond_denoised"]
                cfg_val = args["cond_scale"]

                guided = uncond + cfg_val * (cond - uncond)

                dims = list(range(1, guided.ndim))
                guided_std = guided.std(dim=dims, keepdim=True).clamp(min=1e-6)
                cond_std = cond.std(dim=dims, keepdim=True).clamp(min=1e-6)

                rescaled = guided * (cond_std / guided_std)
                return rescaled * phi + guided * (1.0 - phi)

            model.set_model_sampler_cfg_function(guidance_rescale_patch)
            logger.info(f"Guidance Rescale applied (phi={phi:.2f})")

        if SamplerMode.is_phase_shift(sampler_mode) and detected_type in VIDEO_MODEL_TYPES:
            logger.warning(
                f"[v4.2] Phase-Shift mode is not supported for video model '{detected_type}' — "
                f"sampler switching mid-schedule causes temporal discontinuities. "
                f"Falling back to Standard mode."
            )
            sampler_mode = SamplerMode.STANDARD

        if SamplerMode.is_phase_shift(sampler_mode):

            if sigma_blend_steps == 0:
                sigma_blend_steps = 3
                logger.info("Phase-Shift: auto-enabled 3 sigma blend steps (was 0)")

            if sampler_mode == SamplerMode.PHASE_SHIFT_DPM:
                secondary_sampler = "dpmpp_2m"
                secondary_scheduler = None                       

                flow_match_types = GUIDANCE_EMBED_MODELS | {"wan", "hunyuan_video"}
                if detected_type in flow_match_types:
                    logger.warning(
                        f"WARNING: DPM solvers are generally incompatible with {detected_type}'s "
                        f"Flow Matching and may produce severe noise/grain. "
                        f"Recommended to use Euler or Phase-Shift SGM."
                    )
            elif sampler_mode == SamplerMode.PHASE_SHIFT_SGM:

                secondary_sampler = sampler                     
                secondary_scheduler = "sgm_uniform"                              
            else:
                secondary_sampler = sampler

            split_step = int(total_steps * max(0.0, min(1.0, phase_split)))

            if effective_start < split_step < effective_end:
                splits.add(split_step)
                phase2_label = secondary_sampler or sampler
                if secondary_scheduler:
                    phase2_label = f"{phase2_label}+{secondary_scheduler}"
                logger.info(
                    f"Phase-Shift: {primary_sampler} (0-{split_step}) → "
                    f"{phase2_label} ({split_step}-{total_steps})"
                )

                if sigma_blend_steps > 0:
                    logger.info(f"Sigma blend: {sigma_blend_steps} steps at transition")

        if refiner_model is not None:
            refiner_step = max(0, min(refiner_start_step, total_steps))
            if effective_start < refiner_step < effective_end:
                splits.add(refiner_step)
                logger.info(f"Refiner starts at step {refiner_step}")

        is_dynamic = "Dynamic" in flux_guidance_profile and detected_type in GUIDANCE_EMBED_MODELS

        is_dynamic_cfg = "Dynamic" in flux_guidance_profile and detected_type in CFG_GUIDED_MODELS

        if is_dynamic:

            denoising_steps = (
                int(total_steps * denoise) if denoise < 1.0 else total_steps
            )
            denoising_start = total_steps - denoising_steps

            idx_20 = denoising_start + int(
                denoising_steps * DYNAMIC_GUIDANCE_EARLY_THRESHOLD
            )
            idx_90 = denoising_start + int(
                denoising_steps * DYNAMIC_GUIDANCE_LATE_THRESHOLD
            )

            if effective_start < idx_20 < effective_end:
                splits.add(idx_20)
            if effective_start < idx_90 < effective_end:
                splits.add(idx_90)
            logger.info(
                f"Dynamic Guidance Active (effective range: steps {denoising_start}-{total_steps}, "
                f"early={idx_20}, late={idx_90})"
            )

        elif is_dynamic_cfg:

            denoising_steps = (
                int(total_steps * denoise) if denoise < 1.0 else total_steps
            )
            denoising_start = total_steps - denoising_steps

            idx_15 = denoising_start + int(denoising_steps * DYNAMIC_CFG_EARLY_THRESHOLD)
            idx_85 = denoising_start + int(denoising_steps * DYNAMIC_CFG_LATE_THRESHOLD)

            if effective_start < idx_15 < effective_end:
                splits.add(idx_15)
            if effective_start < idx_85 < effective_end:
                splits.add(idx_85)
            logger.info(
                f"Dynamic CFG Active for {detected_type} (effective range: steps "
                f"{denoising_start}-{total_steps}, boost→{idx_15}, taper→{idx_85})"
            )

        sorted_splits = sorted(
            s for s in splits if effective_start <= s <= effective_end
        )

        pbar_ref = None
        use_custom_preview = False
        if preview_method != "None":

            if preview_method == "TAESD":
                try:

                    from comfy.taesd.taesd import TAESDDecoder              
                except (ImportError, AttributeError):
                    logger.warning(
                        "[Radiance] TAESD unavailable, fallback to Latent2RGB"
                    )
                    preview_method = "Latent2RGB"

            try:
                pbar_ref = comfy.utils.ProgressBar(total_steps)
                use_custom_preview = True
                logger.debug(f"Preview callback active: {preview_method}")
            except (AttributeError, TypeError) as e:
                logger.warning(f"Failed to create preview callback: {e}")

                preview_method = "None"

        def create_phase_callback(phase_start_step):
            def callback(step, x0, x1, total_steps):

                global_step = step + phase_start_step

                if pbar_ref:
                    pbar_ref.update_absolute(global_step + 1, total_steps, (x0,))

            return callback

        current_latent = work_latent
        prev_stage_sigmas: Optional[torch.Tensor] = None                      

        try:
            _ms = model.get_model_object("model_sampling")
            _untrimmed = comfy.samplers.calculate_sigmas(_ms, scheduler, total_steps)
            if flux_shift != 1.0:
                _untrimmed = flux_shift_sigmas(_untrimmed, flux_shift)
            _untrimmed = correct_sigma_end(_untrimmed)
            _sigma_cache.put(model, scheduler, total_steps, _untrimmed)
        except Exception as _e:
            logger.debug(f"Pre-seed cache skipped: {_e}")

        def _get_base_sigmas(mdl, sched: str = scheduler) -> torch.Tensor:
            # ALBABIT-FIX: Directly return the overridden sigmas to ensure SigmaIndexer matches sizes
            if sigmas_override is not None and sched == scheduler and mdl is model:
                return sigmas
            return compute_base_sigmas(
                mdl, sched, total_steps, scheduler, flux_shift, denoise, _sigma_cache
            )

        t0 = time.time()

        stage_timings: List[Tuple[int, int, int, str, float]] = []

        planned_stages: List[SamplingStage] = []
        for si in range(len(sorted_splits) - 1):
            gs = sorted_splits[si]
            ge = sorted_splits[si + 1]
            if gs >= ge:
                continue

            stage_model = model
            if refiner_model is not None and gs >= refiner_start_step:
                stage_model = refiner_model

            stage_sampler = primary_sampler
            stage_scheduler = scheduler
            is_shifted = False
            is_blend = False

            if SamplerMode.is_phase_shift(sampler_mode) and gs >= split_step:
                is_shifted = True
                if secondary_sampler:
                    stage_sampler = secondary_sampler
                if secondary_scheduler:
                    stage_scheduler = secondary_scheduler
                if gs == split_step and sigma_blend_steps > 0:
                    is_blend = True

            planned_stages.append(
                SamplingStage(
                    index=si,
                    global_start=gs,
                    global_end=ge,
                    model=stage_model,
                    sampler_name=stage_sampler,
                    scheduler_name=stage_scheduler,
                    is_phase_shifted=is_shifted,
                    is_blend_point=is_blend,
                )
            )

        for ps in planned_stages:
            label = f"{ps.sampler_name}+{ps.scheduler_name}"
            if ps.is_blend_point:
                label += " [blend]"
            logger.info(
                f"Plan Stage {ps.index + 1}: steps {ps.global_start}→{ps.global_end} [{label}]"
            )

        try:
            if tile_mode and latent_samples.ndim == 4:
                # FIX: tile_mode runs INSTEAD of the staged denoising loop.
                logger.info(
                    f"[v4.0] Tile sampling: size={tile_size}, "
                    f"overlap={tile_overlap}, blend={tile_blend}"
                )
                t_tile = time.time()
                _tile_sampler_obj = comfy.samplers.sampler_object(primary_sampler)
                current_latent = tile_sample(
                    model=model,
                    noise=noise,
                    latent_samples=work_latent,
                    positive=positive,
                    negative=negative,
                    sigmas=sigmas,
                    sampler_obj=_tile_sampler_obj,
                    seed=seed,
                    tile_size=tile_size,
                    tile_overlap=tile_overlap,
                    tile_blend=tile_blend,
                    noise_mask=noise_mask,
                )
                # ALBABIT-FIX: Prevent AttributeError on NestedTensor which lacks 'is_cuda'
                if hasattr(current_latent, "is_cuda"):
                    samples = current_latent.cpu() if current_latent.is_cuda else current_latent
                else:
                    samples = current_latent
                
                timings["tile_sampling"] = time.time() - t_tile
                timings["sampling"] = timings["tile_sampling"]
                logger.info(
                    f"[v4.0] Tile sampling done in {timings['tile_sampling']:.2f}s"
                )

            for plan_idx, stage in enumerate(planned_stages):
                if tile_mode and latent_samples.ndim == 4:
                    break  # tile_sample already ran above
                t_stage = time.time()
                i = stage.index
                s_start = stage.global_start
                s_end = stage.global_end
                current_model = stage.model
                current_sampler = stage.sampler_name
                current_scheduler = stage.scheduler_name

                stage_positive = positive
                if is_dynamic:
                    effective_guidance = compute_dynamic_guidance(
                        flux_guidance, s_start, total_steps, denoise
                    )
                    stage_positive = apply_flux_guidance(positive, effective_guidance)
                    logger.debug(
                        f"Dynamic Guidance @ step {s_start}: {effective_guidance:.2f}"
                    )

                elif detected_type in GUIDANCE_EMBED_MODELS:

                    stage_positive = apply_flux_guidance(positive, flux_guidance)

                logger.info(
                    f"Stage {i+1}: Steps {s_start}-{s_end} | "
                    f"Sampler: {current_sampler} | Scheduler: {current_scheduler}"
                )

                try:

                    base_sigmas = _get_base_sigmas(current_model, current_scheduler)

                    indexer = SigmaIndexer(total_steps, base_sigmas)
                    stage_sigmas = indexer.get_stage_sigmas(s_start, s_end)

                    if stage_sigmas is None:

                        continue

                    if (
                        stage.is_blend_point
                        and prev_stage_sigmas is not None
                    ):
                        stage_sigmas = gradual_sigma_blend(
                            prev_stage_sigmas, stage_sigmas, sigma_blend_steps
                        )

                    if len(stage_sigmas) < 2:
                        logger.warning(
                            f"Stage {i+1}: Insufficient sigmas ({len(stage_sigmas)}), skipping"
                        )
                        continue

                    is_last_stage = plan_idx == len(planned_stages) - 1
                    if (
                        return_with_leftover_noise
                        and is_last_stage
                        and len(stage_sigmas) >= 3                                   
                        and stage_sigmas[-1].item() == 0.0
                    ):
                        stage_sigmas = stage_sigmas[:-1]
                        logger.debug("Stripped terminal sigma for leftover noise")

                    sampler_obj = comfy.samplers.sampler_object(current_sampler)

                    if is_cfg_plus_plus and len(stage_sigmas) > 0:
                        sigma_max = (
                            base_sigmas[0].item() if len(base_sigmas) > 0 else 1.0
                        )
                        effective_cfg = apply_cfg_plus_plus(
                            cfg, stage_sigmas[0], sigma_max
                        )
                        logger.debug(
                            f"CFG++: {cfg:.2f} → {effective_cfg:.2f} at sigma {stage_sigmas[0].item():.4f}"
                        )
                    elif is_dynamic_cfg:

                        effective_cfg = compute_dynamic_cfg(
                            cfg, s_start, total_steps, denoise
                        )
                        logger.debug(
                            f"Dynamic CFG @ step {s_start}: {cfg:.2f} → {effective_cfg:.2f}"
                        )
                    else:
                        effective_cfg = cfg

                    stage_latent = current_latent

                    is_first_stage = plan_idx == 0
                    if is_first_stage and add_noise:
                        stage_noise = noise
                        logger.debug("Stage 1: sampler will add initial noise")
                    else:
                        stage_noise = torch.zeros_like(noise)
                        if is_first_stage:
                            logger.debug("Stage 1: add_noise=False, no noise")
                        else:
                            logger.debug(f"Stage {i+1}: continuation, no extra noise")

                    if (
                        _HAS_NESTED_TENSOR
                        and isinstance(current_latent, _NestedTensor)
                        and noise_type.lower() != "gaussian"
                    ):
                        try:
                            _dev = comfy.model_management.get_torch_device()
                            new_noises = []
                            for sub_t in current_latent.tensors:
                                if is_first_stage and add_noise:
                                    sub_noise = generate_noise(sub_t, seed, noise_type, frames=frames)
                                else:
                                    sub_noise = torch.zeros_like(sub_t)
                                new_noises.append(sub_noise)
                            stage_noise = _NestedTensor(tuple(new_noises))
                        except Exception as _ne:
                            logger.warning(
                                f"[Radiance] LTX-AV NestedTensor noise rebuild failed: {_ne}. "
                            )

                    comfy.model_management.load_model_gpu(current_model)

                    result = _sample_custom_compat(
                        model=current_model,
                        noise=stage_noise,
                        cfg=effective_cfg,
                        sampler=sampler_obj,
                        sigmas=stage_sigmas,
                        positive=stage_positive,
                        negative=negative,
                        latent_image=stage_latent,
                        noise_mask=noise_mask,
                        callback=create_phase_callback(s_start),
                        disable_pbar=use_custom_preview,
                        seed=seed,
                    )
                    current_latent = result

                    prev_stage_sigmas = indexer.get_stage_sigmas(s_start, s_end)
                    log_tensor(f"Stage {i+1} Output", current_latent)

                    if plan_idx < len(planned_stages) - 1:
                        next_stage = planned_stages[plan_idx + 1]
                        next_base = _get_base_sigmas(
                            next_stage.model, next_stage.scheduler_name
                        )
                        next_indexer = SigmaIndexer(total_steps, next_base)
                        next_sigma = next_indexer.get_sigma_at(next_stage.global_start)

                        if (
                            next_sigma is not None
                            and len(stage_sigmas) > 0
                        ):
                            expected_sigma = stage_sigmas[-1].item()
                            sigma_diff = abs(expected_sigma - next_sigma)
                            if sigma_diff > SIGMA_DISCONTINUITY_THRESHOLD:
                                logger.warning(
                                    f"Sigma discontinuity: {expected_sigma:.4f} → {next_sigma:.4f} "
                                    f"(Δ={sigma_diff:.4f})"
                                )

                    stage_time = time.time() - t_stage
                    stage_timings.append(
                        (i + 1, s_start, s_end, current_sampler, stage_time)
                    )

                except (RuntimeError, ValueError) as e:
                    logger.error(f"Error in Stage {i+1}: {e}")
                    raise

            # ALBABIT-FIX: Prevent AttributeError on NestedTensor which lacks 'is_cuda'
            if hasattr(current_latent, "is_cuda"):
                samples = current_latent.cpu() if current_latent.is_cuda else current_latent
            else:
                samples = current_latent

            timings["sampling"] = time.time() - t0

            # FIX: tile_mode 4D now handled earlier in the stage loop.
            if tile_mode and latent_samples.ndim == 5:
                logger.warning(
                    "[Radiance] Tile sampling ignored for 5D video latents."
                )

        finally:

            if "noise" in locals():
                try: del noise
                except UnboundLocalError: pass
            if "work_latent" in locals():
                try: del work_latent
                except UnboundLocalError: pass

            cleanup_gpu_memory()

        if is_ltx_av and ltxav_obj is not None and _HAS_NESTED_TENSOR:
            try:
                final_t = samples["samples"] if isinstance(samples, dict) else samples

                if isinstance(final_t, _NestedTensor):
                    final_t_raw = final_t.tensors
                else:
                    final_t_raw = final_t

                v_s, a_s = ltxav_obj.separate_audio_and_video_latents(final_t_raw, None)
                v_s = v_s.detach()
                a_s = a_s.detach()

                f_v = v_norm[-1] if v_norm else 1.0
                f_a = a_norm[-1] if a_norm else 1.0
                if f_v != 1.0:
                    v_s = v_s * f_v
                if f_a != 1.0:
                    a_s = a_s * f_a

                recombined = _NestedTensor(
                    ltxav_obj.recombine_audio_and_video_latents(v_s, a_s)
                )
                if isinstance(samples, dict):
                    samples["samples"] = recombined
                else:
                    samples = recombined

                del v_s, a_s, final_t_raw
                logger.info(
                    f"[Radiance] LTX-AV normalization applied "
                    f"(video×{f_v:.3f}, audio×{f_a:.3f})"
                )
            except Exception as _av_err:
                logger.warning(
                    f"[Radiance] LTX-AV recombination failed: {_av_err}. "
                )

        t0 = time.time()
        out = latent.copy()
        out["samples"] = samples
        timings["output_prep"] = time.time() - t0

        total_time = time.time() - t_start

        logger.info(
            f"Sampling complete: {steps} steps, {total_time:.2f}s total, "
            f"{timings['sampling']:.2f}s sampling"
        )
        for stage_num, s_start_t, s_end_t, samp, t in stage_timings:
            logger.info(
                f"  Stage {stage_num}: steps {s_start_t}→{s_end_t} [{samp}] = {t:.3f}s"
            )

        output_sigmas = (
            sigmas if sigmas is not None and len(sigmas) > 0 else torch.tensor([0.0])
        )

        sigma_report = build_sigma_report(
            detected_type,
            steps,
            scheduler,
            flux_shift,
            denoise,
            output_sigmas,
            sampler_mode,
            sorted_splits,
            stage_timings,
            total_time,
            ays_active=ays_schedule,
            frames=frames,
        )

        latent_meta = _build_latent_meta(
            detected_type=detected_type,
            steps=steps,
            scheduler=scheduler,
            flux_shift=flux_shift,
            denoise=denoise,
            sigmas=output_sigmas,
            ays_active=ays_schedule,
            pag_active=(pag_scale > 0),
            noise_type=noise_type,
            tile_mode=tile_mode,
            multi_cond_mode=multi_cond_mode,
            clip_target=conditioning_clip_target,
            seed=seed,
            total_time_ms=int(total_time * 1000),
            latent_format=latent_format,
            frames=frames,
        )

        return (out, output_sigmas, sigma_report, latent_meta)

NODE_CLASS_MAPPINGS = {
    "RadianceSamplerPro": RadianceSamplerPro,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceSamplerPro": "◎ Radiance Sampler Pro",
}
