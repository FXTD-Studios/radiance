import torch
import time

try:
    from radiance.config.constants import VERSION as _RADIANCE_VERSION
except Exception:  # noqa: BLE001 - imported outside the package (tests, tools)
    _RADIANCE_VERSION = "unknown"
import math
import logging
import gc
import json
from typing import Tuple, Dict, Any, Optional, List
from dataclasses import dataclass, field
from radiance.core.tiling import blend_weight_2d, edge_overlaps_from_coords, clamp_overlap

import comfy.samplers
import comfy.sample
import comfy.model_management
import comfy.utils
try:
    from .tensor_contract import ensure_4d, ensure_5d
except (ImportError, ValueError):
    from tensor_contract import ensure_4d, ensure_5d


try:
    from comfy.nested_tensor import NestedTensor as _NestedTensor
    _HAS_NESTED_TENSOR = True
except ImportError:
    _NestedTensor = None
    _HAS_NESTED_TENSOR = False

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

DYNAMIC_CFG_EARLY_MULTIPLIER = 1.2
DYNAMIC_CFG_LATE_MULTIPLIER = 0.7                                                
DYNAMIC_CFG_EARLY_THRESHOLD = 0.15                      
DYNAMIC_CFG_LATE_THRESHOLD = 0.85                      

MODEL_TYPES = [
    "auto",
    "flux",
    # ALBABIT-FIX: flux2 / flux2-klein were absent — Loader uses them, Sampler silently fell back to "flux"
    "flux2", "flux2-klein",
    "sd3",
    # ALBABIT-FIX: renamed "sd35" → "sd3.5" to match Loader, model/detect.py, prompt.py
    "sd3.5",
    "sdxl",
    # ALBABIT-FIX: renamed "sd15" → "sd1.5" -- same rationale as sd3.5 above,
    # converges on the Loader/model/detect.py form instead of diverging from it.
    "sd1.5",
    "wan",
    # ALBABIT-FIX: WAN 2.2 TI2V-5B (48ch VAE, distinct from "wan"'s 16ch) —
    # real bug fix, see model/detect.py.
    "wan_ti2v",
    "ltxv",
    "ltxav",
    "hunyuan_video",
    "lumina2",
    "z_image",
    "chroma",
    "cosmos",
    "cogvideox",
    "mochi",  # ALBABIT-FIX: Mochi-1 — match Resolution/Loader model types
    "minimax",  # ALBABIT-FIX: MiniMax H3, matches Resolution/Loader/Prompt model types
    # 3.5: ComfyUI 0.32 families (see model/detect.py).
    "qwen_image", "krea2", "hunyuan_image", "hunyuan_video_15",
    "hidream", "omnigen2", "longcat_image", "kandinsky5", "kandinsky5_image",
    "qwen_image21",  # ALBABIT-FIX: Qwen-Image 2.1, matches Resolution/Loader model types
]

VIDEO_MODEL_TYPES = {"wan", "wan_ti2v", "ltxv", "ltxav", "hunyuan_video", "cosmos", "cogvideox", "mochi", "minimax",
                     "hunyuan_video_15", "kandinsky5"}

# ALBABIT-FIX: flux2/flux2-klein use guidance_embed like flux (not external CFG)
# ALBABIT-FIX: lumina2 removed -- its official workflow uses a plain KSampler
# cfg, no guidance-embed node (unlike Flux's FluxGuidance) -- see CFG_GUIDED_MODELS
# ALBABIT-FIX: z_image removed too -- exact same situation as lumina2 (its
# official workflow's KSampler uses cfg=4, no guidance-embed node either),
# apparently missed when lumina2 got the same fix. Confirmed against
# Comfy-Org's own bundled "image_z_image.json" template directly.
# 3.5: LongCat-Image is a Flux transformer and its official template drives it
# through FluxGuidance (4.0) like Flux.1, with cfg held at 4 under CFGNorm.
GUIDANCE_EMBED_MODELS = {"flux", "flux2", "flux2-klein", "ltxv", "longcat_image"}

# ALBABIT-FIX: "sd35" renamed to "sd3.5" for consistency with Loader/detect.py
# ALBABIT-FIX: lumina2 added -- classic external CFG, confirmed via its
# official example workflow (plain KSampler cfg=4, no guidance-embed node)
# ALBABIT-FIX: z_image added -- same evidence class as lumina2 above.
# ALBABIT-FIX: wan_ti2v added -- same CFG-guided convention as "wan" (its
# official workflow's KSampler uses a real cfg value, no guidance-embed node).
CFG_GUIDED_MODELS = {"wan", "wan_ti2v", "hunyuan_video", "sdxl", "sd1.5", "sd3", "sd3.5", "ltxav", "cogvideox", "mochi", "lumina2", "z_image",
                     # 3.5: every one of these runs a plain KSampler / CFGGuider with
                     # a real cfg in its official Comfy-Org template.
                     "qwen_image", "krea2", "hunyuan_image", "hunyuan_video_15",
                     "hidream", "omnigen2", "kandinsky5", "kandinsky5_image",
                     "qwen_image21"}  # ALBABIT-FIX: same evidence, its template's KSampler cfg

# ALBABIT-FIX: "minimax" belongs in neither set above on purpose. Its reference
# pipeline uses BasicGuider, which has no cfg input and no guidance-embed
# mechanism either; cfg is pinned inert via MODEL_DEFAULTS instead of routed
# through either widget family.

MODEL_DEFAULTS: Dict[str, Dict[str, Any]] = {
    # ALBABIT-FIX: steps=20 added, verified against Comfy-Org's official
    # Flux.1 Dev workflow template.
    "flux": {
        "cfg": 1.0,
        "scheduler": "simple",
        "guidance": 3.5,
        "shift": 1.0,
        "sampler": "euler",
        "steps": 20,
    },
    # ALBABIT-FIX: Flux.2 Dev and Flux.2 Klein — guidance_embed models like Flux.1,
    # same sampling defaults (scheduler=simple, cfg=1.0, guidance_embed). guidance
    # verified against BFL's own example code (4.0, not Flux.1's 3.5). Klein's
    # value is a fallback for when model_meta isn't connected -- Base (undistilled,
    # guidance=4.0) and distilled (guidance~1.0) are architecturally identical and
    # only distinguishable via model_meta's unet_file (see refine_distillation_from_meta).
    # ALBABIT-FIX: steps=20 added, verified against Comfy-Org's official
    # Flux.2 Dev/Klein workflow templates.
    "flux2": {
        "cfg": 1.0,
        "scheduler": "simple",
        "guidance": 4.0,
        "shift": 1.0,
        "sampler": "euler",
        "steps": 20,
    },
    "flux2-klein": {
        "cfg": 1.0,
        "scheduler": "simple",
        "guidance": 4.0,
        "shift": 1.0,
        "sampler": "euler",
        "steps": 20,
    },
    # ALBABIT-FIX: cfg 4.5->5.45, sampler dpmpp_2m->euler, steps=30 -- all
    # verified directly against the official SD3 Medium example workflow's
    # embedded JSON (sd3_simple_example.png, comfyanonymous/ComfyUI_examples).
    "sd3": {
        "cfg": 5.45,
        "scheduler": "sgm_uniform",
        "guidance": 0.0,
        "shift": 1.0,
        "sampler": "euler",
        "steps": 30,
    },
    # ALBABIT-FIX: renamed from "sd35" to "sd3.5" for consistency with Loader/detect.py.
    # cfg/sampler verified against Comfy-Org's own official SD3.5 Large workflow
    # (sd3.5-t2i-fp8-scaled-workflow.json) -- sampler was "dpmpp_2m" (wrong,
    # should be "euler"); cfg confirmed against Albabit's own ComfyUI workflow (4.0).
    # ALBABIT-FIX: steps=20 added, verified against Comfy-Org's official
    # SD3.5 Large workflow template (same source already used for cfg/sampler).
    "sd3.5": {
        "cfg": 4.0,
        "scheduler": "sgm_uniform",
        "guidance": 0.0,
        "shift": 1.0,
        "sampler": "euler",
        "steps": 20,
    },
    # ALBABIT-FIX: cfg 7.0->8.0, sampler dpmpp_2m->euler, scheduler
    # karras->normal, matching ComfyUI's own official SDXL example workflow
    # (sdxl_simple_example.json). steps=20 added from the same file (base
    # stage runs steps 0-20 of a nominal 25-step schedule with the optional
    # refiner stage disabled by default -- we don't have a 2-stage refiner
    # split, so 20 is the actual number of steps that workflow runs).
    "sdxl": {
        "cfg": 8.0,
        "scheduler": "normal",
        "guidance": 0.0,
        "shift": 1.0,
        "sampler": "euler",
        "steps": 20,
    },
    # ALBABIT-FIX: cfg 7.0->8.0, sampler dpmpp_2m->euler, steps=20 -- all
    # verified against ComfyUI's own default startup workflow (default.json,
    # v1-5-pruned-emaonly, the graph shown on first launch).
    "sd1.5": {
        "cfg": 8.0,
        "scheduler": "normal",
        "guidance": 0.0,
        "shift": 1.0,
        "sampler": "euler",
        "steps": 20,
    },

    "wan": {
        "cfg": 6.0,
        "scheduler": "simple",
        "guidance": 0.0,
        "shift": 8.0,
        # ALBABIT-FIX: euler -> uni_pc, confirmed by 2 official Comfy-Org
        # workflows (Wan 2.1 1.3B T2V and Wan 2.1 14B I2V 720P). steps=20
        # added, from the same 14B I2V workflow (its KSampler uses steps=20).
        "sampler": "uni_pc",
        "steps": 20,
        "guidance_type": "cfg",
    },
    # ALBABIT-FIX: WAN 2.2 TI2V-5B -- verified against Comfy-Org's official
    # bundled "video_wan2_2_5B_ti2v.json" workflow template. Same scheduler/
    # shift/sampler/steps as "wan" above, only cfg genuinely differs (5 vs 6).
    "wan_ti2v": {
        "cfg": 5.0,
        "scheduler": "simple",
        "guidance": 0.0,
        "shift": 8.0,
        "sampler": "uni_pc",
        "steps": 20,
        "guidance_type": "cfg",
    },
    # ALBABIT-FIX: steps=30 added (was previously "faible confiance" from a
    # Lightricks model card, now upgraded to "haute" -- confirmed by
    # ComfyUI's own official LTX Video example workflow, corroborated by
    # Lightricks' own 13B-dev first-pass config). The 2B-0.9.6-dev config
    # suggests 40 instead -- our entry doesn't distinguish 2B/13B currently.
    "ltxv": {
        "cfg": 1.0,
        "scheduler": "simple",
        "guidance": 3.5,
        "shift": 2.37,
        "sampler": "euler",
        "steps": 30,
        "guidance_type": "embedding",
    },

    "ltxav": {
        "cfg": 3.0,
        "scheduler": "beta",
        "guidance": 0.0,
        "shift": 3.0,                                                     
        "sampler": "euler",
        "guidance_type": "cfg",
    },
    # ALBABIT-FIX: steps=20 added, from the same official ComfyUI HunyuanVideo
    # workflow already used for shift/sampler/scheduler. Note: Tencent's own
    # CLI README recommends 50 steps -- a real divergence between the
    # ComfyUI-native default and the creator's own recommendation, not
    # resolved here (kept internally consistent with the single source
    # already used for this architecture's other values).
    "hunyuan_video": {
        "cfg": 6.0,
        "scheduler": "simple",
        "guidance": 0.0,
        "shift": 7.0,
        "sampler": "euler",
        "steps": 20,
        "guidance_type": "cfg",
    },
    # ALBABIT-FIX: Lumina2's official example workflow shows a plain KSampler
    # cfg=4 with no guidance-embed node at all (unlike Flux's FluxGuidance) --
    # it's classic external CFG, not embedded guidance. cfg 1.0->4.0,
    # sampler euler->res_multistep, guidance_type embedding->cfg, steps=25
    # added (matches the workflow's saved value; its own Note claims "36
    # steps" as the official recommendation but the workflow itself uses 25).
    "lumina2": {
        "cfg": 4.0,
        "scheduler": "simple",
        "guidance": 0.0,
        "shift": 6.0,
        "sampler": "res_multistep",
        "steps": 25,
        "guidance_type": "cfg",
    },
    # ALBABIT-FIX: steps=25 verified against Comfy-Org's official Z-Image
    # (Base) workflow template -- its Turbo variant uses 8 steps instead, see
    # refine_distillation_from_meta() below. Same template's KSampler also
    # showed cfg=1.0/sampler="euler"/guidance_type="embedding" here were all
    # wrong -- plain KSampler cfg=4, sampler="res_multistep", no
    # guidance-embed node at all (exact same situation lumina2 was already
    # fixed for, apparently missed for z_image at the time -- both share the
    # same "lumina2" CLIPLoader type, consistent with a related architecture).
    "z_image": {
        "cfg": 4.0,
        "scheduler": "simple",
        "guidance": 0.0,
        "shift": 3.0,
        "sampler": "res_multistep",
        "steps": 25,
        "guidance_type": "cfg",
    },
    # ALBABIT-FIX: steps=20 added, verified against ComfyUI's own official
    # Cosmos-1.0 7B example workflow.
    "cosmos": {
        "cfg": 7.0,
        "scheduler": "simple",
        "guidance": 0.0,
        "shift": 3.0,
        "sampler": "euler",
        "steps": 20,
        "guidance_type": "cfg",
    },
    # ── v2.6.0: New video models ──────────────────────────────────────────────
    # ALBABIT-FIX: steps=50 added, verified against THUDM's official
    # CogVideoX-5b model card (num_inference_steps=50, guidance_scale=6 --
    # cfg was already exact).
    "cogvideox": {
        "cfg": 6.0,
        "scheduler": "simple",
        "guidance": 0.0,
        "shift": 8.0,
        "sampler": "euler",
        "steps": 50,
        "guidance_type": "cfg",
    },
    # ALBABIT-FIX: cfg/scheduler/steps verified against lodestones' own official
    # Chroma1-HD ComfyUI workflow (cfg was 1.0, wrong; scheduler was "simple",
    # wrong -- should be "beta"). "steps" is a generic (non-distillation)
    # fallback -- new for this architecture, see refine_distillation_from_meta's
    # docstring/_configure_model_and_defaults for how it's resolved.
    "chroma": {
        "cfg": 3.8,
        "scheduler": "beta",
        "guidance": 0.0,
        "shift": 1.0,
        "sampler": "euler",
        "steps": 26,
    },
    # ALBABIT-FIX: Mochi-1 (Genmo) — 12ch video VAE, T5XXL text encoder.
    # steps=64 added, verified against Genmo's official Mochi 1 model card
    # (num_inference_steps=64, cfg_schedule=[4.5]*64 -- cfg was already exact).
    "mochi": {
        "cfg": 4.5,
        "scheduler": "simple",
        "guidance": 0.0,
        "shift": 6.0,
        "sampler": "euler",
        "steps": 64,
        "guidance_type": "cfg",
    },
    # ALBABIT-FIX: verified against Comfy-Org's official T2V workflow
    # template (KSamplerSelect=res_multistep, BasicScheduler=simple/20
    # steps). BasicGuider has zero widgets, so cfg=1.0 just pins it inert,
    # same trick flux/ltxv use. No shift node in the reference pipeline.
    "minimax": {
        "cfg": 1.0,
        "scheduler": "simple",
        "guidance": 0.0,
        "shift": 1.0,
        "sampler": "res_multistep",
        "steps": 20,
    },
    # ── 3.5: ComfyUI 0.32 families ─────────────────────────────────────────
    # Each entry is read off Comfy-Org's official workflow template for the
    # model (templates/<name>.json in Comfy-Org/workflow_templates) unless
    # stated otherwise.
    # Qwen-Image: the current official template ships the 8-step Lightning
    # LoRA (cfg 1, 8 steps); these are the base-model values from the same
    # graph without the LoRA (ModelSamplingAuraFlow shift 3.1, cfg 2.5).
    "qwen_image": {
        "cfg": 2.5,
        "scheduler": "simple",
        "guidance": 0.0,
        "shift": 3.1,
        "sampler": "euler",
        "steps": 20,
        "guidance_type": "cfg",
    },
    # ALBABIT-FIX: Qwen-Image 2.1: image_qwen_image_2_1_t2i.json (KSampler 25
    # steps, cfg 1, euler/simple, no shift node: the model keeps its own 0.69).
    # Qwen's own pipeline runs 40-50 steps.
    "qwen_image21": {
        "cfg": 1.0,
        "scheduler": "simple",
        "guidance": 0.0,
        "shift": 1.0,
        "sampler": "euler",
        "steps": 25,
        "guidance_type": "cfg",
    },
    # Krea 2 Turbo: image_krea2_turbo_t2i.json (KSampler 8 steps, cfg 1, euler/simple).
    "krea2": {
        "cfg": 1.0,
        "scheduler": "simple",
        "guidance": 0.0,
        "shift": 1.0,
        "sampler": "euler",
        "steps": 8,
        "guidance_type": "cfg",
    },
    # HunyuanImage 2.1: no Comfy-Org template yet. shift 5.0 is the model's
    # own sampling_settings in comfy/supported_models.py; cfg 3.5 / 50 steps
    # are the Tencent reference defaults.
    "hunyuan_image": {
        "cfg": 3.5,
        "scheduler": "simple",
        "guidance": 0.0,
        "shift": 5.0,
        "sampler": "euler",
        "steps": 50,
        "guidance_type": "cfg",
    },
    # HunyuanVideo 1.5: video_hunyuan_video_1.5_720p_t2v.json (CFGGuider 6,
    # ModelSamplingSD3 shift 7, euler/simple 20 steps).
    "hunyuan_video_15": {
        "cfg": 6.0,
        "scheduler": "simple",
        "guidance": 0.0,
        "shift": 7.0,
        "sampler": "euler",
        "steps": 20,
        "guidance_type": "cfg",
    },
    # HiDream-I1 Full: hidream_i1_full.json (KSampler 50 steps, cfg 5,
    # uni_pc/simple, ModelSamplingSD3 shift 3).
    "hidream": {
        "cfg": 5.0,
        "scheduler": "simple",
        "guidance": 0.0,
        "shift": 3.0,
        "sampler": "uni_pc",
        "steps": 50,
        "guidance_type": "cfg",
    },
    # OmniGen2: image_omnigen2_t2i.json (DualCFGGuider 5 / 2, euler/simple 20 steps).
    "omnigen2": {
        "cfg": 5.0,
        "scheduler": "simple",
        "guidance": 0.0,
        "shift": 1.0,
        "sampler": "euler",
        "steps": 20,
        "guidance_type": "cfg",
    },
    # LongCat-Image: image_longcat_text_to_image.json (FluxGuidance 4,
    # KSampler cfg 4 with CFGNorm, euler/simple 20 steps).
    "longcat_image": {
        "cfg": 4.0,
        "scheduler": "simple",
        "guidance": 4.0,
        "shift": 1.0,
        "sampler": "euler",
        "steps": 20,
    },
    # Kandinsky 5 video: video_kandinsky5_t2v.json (KSampler 50 steps, cfg 5,
    # euler_ancestral/beta, ModelSamplingSD3 shift 5).
    "kandinsky5": {
        "cfg": 5.0,
        "scheduler": "beta",
        "guidance": 0.0,
        "shift": 5.0,
        "sampler": "euler_ancestral",
        "steps": 50,
        "guidance_type": "cfg",
    },
    # Kandinsky 5 image: image_kandinsky5_t2i.json (KSampler 50 steps,
    # cfg 3.5, euler/simple, ModelSamplingSD3 shift 3).
    "kandinsky5_image": {
        "cfg": 3.5,
        "scheduler": "simple",
        "guidance": 0.0,
        "shift": 3.0,
        "sampler": "euler",
        "steps": 50,
        "guidance_type": "cfg",
    },
    # ALBABIT-FIX: previously fell back to "sd1.5" (cfg=7.0/dpmpp_2m/normal) --
    # verified against AuraFlow's own official ComfyUI workflow, which
    # contradicts all three. No shift node present (unlike Lumina2, which
    # reuses the same ModelSamplingAuraFlow node but at shift=6.0 -- confirmed
    # NOT applicable to AuraFlow's own workflow, checked directly).
    "aura_flow": {
        "cfg": 3.48,
        "scheduler": "sgm_uniform",
        "guidance": 0.0,
        "shift": 1.0,
        "sampler": "euler",
        "steps": 20,
    },
    # ALBABIT-FIX: previously fell back to "sd1.5" -- cfg/sampler verified
    # against multiple independent community sources (weaker than AuraFlow's
    # direct official workflow, moderate confidence). scheduler/shift kept at
    # sd1.5-equivalent values, no better source found. steps=20 added, from
    # the diffusers pipeline's own default parameter (no official ComfyUI
    # workflow found for PixArt Sigma -- moderate confidence, same tier as cfg).
    "pixart": {
        "cfg": 4.5,
        "scheduler": "normal",
        "guidance": 0.0,
        "shift": 1.0,
        "sampler": "dpmpp_2m",
        "steps": 20,
    },
}

PREVIEW_METHODS = ["None", "TAESD", "Latent2RGB"]

NOISE_TYPES = ["Gaussian", "Perlin", "Uniform", "Spectral", "Brownian", "Simplex", "Voronoi", "Curl"]

CLIP_TARGETS = ["Auto", "clip_l", "clip_g", "t5xxl"]

MULTI_COND_MODES = ["Off", "average", "weighted", "sequential"]

TILE_BLEND_MODES = ["feather", "average", "gaussian"]

class SamplerMode:

    STANDARD = "Standard"
    PHASE_SHIFT_DPM = "Phase-Shift (Euler >> DPM)"
    PHASE_SHIFT_SGM = "Phase-Shift (Euler >> SGM)"
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
            except (AttributeError, RuntimeError) as _exc:
                logger.debug(
                    "[Radiance] _make_key(): ignoring %s from `if hasattr(model, 'model') and hasattr(model.model, 'model_c…`: %s",
                    type(_exc).__name__, _exc,
                )

            return (config_name, round(sigma_max, 6), round(sigma_min, 6), scheduler, total_steps)
        except (AttributeError, RuntimeError):
            # BUG-SIGMACACHE-KEY FIX: removed time.time() from the fallback key.
            # Including it created a unique throwaway entry every call, filling
            # the 32-slot LRU with non-reusable entries for unusual models.
            logger.debug("SigmaCache: failed to build config key, using id-based fallback")
            return (id(model), scheduler, total_steps)

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

# ═══════════════════════════════════════════════════════════════════════════════
#                    MODEL REGISTRY & DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

class RadianceModelRegistry:
    """Registry for model architecture detection logic."""
    _detectors = []

    @classmethod
    def register(cls, priority: int = 10):
        def decorator(func):
            cls._detectors.append((priority, func))
            # Sort by priority (lower number = higher priority)
            cls._detectors.sort(key=lambda x: x[0])
            return func
        return decorator

    @classmethod
    def detect(cls, model) -> str:
        for _, detector in cls._detectors:
            try:
                res = detector(model)
                if res: return res
            except Exception as e:
                logger.debug(f"Detector {detector.__name__} failed: {e}")
        return "sd1.5"

@RadianceModelRegistry.register(priority=1)
def detect_by_config(model) -> Optional[str]:
    """Detect model type by explicit config class name."""
    try:
        if not hasattr(model, "model") or not hasattr(model.model, "model_config"):
            return None
        config_cls = type(model.model.model_config).__name__
        config_map = {
            "WAN21": "wan", "WAN22": "wan",
            "LTXV": "ltxv", "LTXAV": "ltxav",
            "HunyuanVideo": "hunyuan_video",
            "Lumina2": "lumina2", "ZImage": "z_image",
            "Chroma": "chroma", "ChromaRadiance": "chroma",
            # ALBABIT-FIX: Flux2 config class maps to "flux2", not "flux"
            "Flux": "flux", "FluxSchnell": "flux", "FluxInpaint": "flux", "Flux2": "flux2",
            "CogVideoX": "cogvideox", "CogVideo": "cogvideox",
            "Mochi": "mochi",  # ALBABIT-FIX: Mochi-1 config class detection
            "MiniMaxH3": "minimax",  # ALBABIT-FIX: MiniMax H3 config class detection
            # ALBABIT-FIX: without it, detect_by_sampling reported Qwen-Image 2.1 as "flux".
            "QwenImage21": "qwen_image21",
        }
        for pattern, mtype in config_map.items():
            if pattern in config_cls: return mtype
    except Exception as exc:
        logger.warning("[nodes_sampler] detect_by_config: %s", exc)
    return None

@RadianceModelRegistry.register(priority=5)
def detect_by_architecture(model) -> Optional[str]:
    """Detect model type by inspecting diffusion model class and modules."""
    try:
        diffusion_model = model.get_model_object("diffusion_model")
        if diffusion_model is None: return None
        
        model_cls = type(diffusion_model).__name__.lower()
        model_module = type(diffusion_model).__module__.lower()
        full_path = f"{model_module}.{model_cls}"

        if "wan" in model_cls or "wan" in model_module: return "wan"
        if "ltxav" in model_cls or ("lightricks" in model_module and hasattr(diffusion_model, "recombine_audio_and_video_latents")):
            return "ltxav"
        if "ltxv" in model_cls or "lightricks" in model_module: return "ltxv"
        if "hunyuan" in model_cls and ("video" in model_cls or "video" in model_module):
            return "hunyuan_video"
        if "cogvideo" in model_cls or "cogvideo" in model_module: return "cogvideox"
        if "mochi" in model_cls or "mochi" in model_module: return "mochi"  # ALBABIT-FIX
        if "minimax" in model_cls or "minimax" in model_module: return "minimax"  # ALBABIT-FIX
        if "lumina" in full_path:
            if hasattr(diffusion_model, "hidden_size") and diffusion_model.hidden_size >= 3840:
                return "z_image"
            return "lumina2"
        if "chroma" in full_path: return "chroma"
        
        if "mmdit" in model_cls or "sd3" in model_cls:
            # ALBABIT-FIX: renamed to "sd3.5" for consistency with Loader/detect.py
            if hasattr(diffusion_model, "in_channels") and diffusion_model.in_channels >= 16:
                return "sd3.5"
            return "sd3"
        
        if "sdxl" in model_cls or hasattr(diffusion_model, "label_emb"): return "sdxl"
    except Exception as exc:
        logger.warning("[nodes_sampler]: %s", exc)
    return None

@RadianceModelRegistry.register(priority=10)
def detect_by_sampling(model) -> Optional[str]:
    """Detect model type by sampling attributes (e.g. Flux shift)."""
    try:
        ms = model.get_model_object("model_sampling")
        if hasattr(ms, "shift") or hasattr(ms, "flux_shift"):
            sigma_max = ms.sigma_max.item() if hasattr(ms, "sigma_max") else 1.0
            if sigma_max <= 1.5: return "flux"
    except Exception as exc:
        logger.warning("[nodes_sampler] detect_by_sampling: %s", exc)
    return None

def detect_model_type(model) -> str:
    """Universal model architecture detector."""
    return RadianceModelRegistry.detect(model)

def get_model_defaults(model_type: str) -> Dict[str, Any]:

    return MODEL_DEFAULTS.get(model_type, MODEL_DEFAULTS["sd1.5"])


def parse_model_meta(model_meta: str) -> Tuple[str, str]:
    """Parse the Loader's model_meta JSON, returning (arch, unet_file). Empty
    strings on missing/malformed input -- callers should treat that as
    "no extra info available", not an error."""
    if not model_meta:
        return "", ""
    try:
        meta = json.loads(model_meta)
        return meta.get("arch", "") or "", meta.get("unet_file", "") or ""
    except Exception:
        return "", ""


def refine_distillation_from_meta(detected_type: str, unet_file: str) -> Optional[Dict[str, Any]]:
    """
    Some checkpoints need settings that differ from their model_type's generic
    default -- only unet_file's exact filename can tell them apart. Verified
    against official model cards. Not every override includes every key (e.g.
    Krea Dev is guidance-only, BFL gives no steps recommendation) -- callers
    must not assume "steps"/"cfg" are always present. Returns None when not
    applicable, leaving the generic MODEL_DEFAULTS fallback in place.
    """
    if not unet_file:
        return None
    name = unet_file.lower()
    if detected_type == "flux2-klein":
        is_distilled = "base" not in name
        return {"guidance": 1.0, "steps": 4} if is_distilled else {"guidance": 4.0, "steps": 50}
    if detected_type == "flux" and "schnell" in name:
        return {"guidance": 0.0, "steps": 4}
    if detected_type == "flux" and "krea" in name:
        return {"guidance": 4.5}
    if detected_type == "sdxl" and "turbo" in name:
        return {"cfg": 1.0, "steps": 1, "sampler": "euler_ancestral"}
    # ALBABIT-FIX: cfg=1.6 (not the "pure" diffusers guidance_scale=0.0
    # translation) to match the Sampler's own pre-existing "[F] SD3.5 Turbo
    # (4 steps)" preset, already tuned in practice.
    if detected_type == "sd3.5" and "turbo" in name:
        return {"cfg": 1.6, "steps": 4}
    # ALBABIT-FIX: verified against Comfy-Org's official Z-Image Turbo
    # workflow template -- KSampler cfg=1/steps=8 (sampler stays
    # "res_multistep", inherited unchanged from MODEL_DEFAULTS["z_image"]
    # above, same for both Base and Turbo).
    if detected_type == "z_image" and "turbo" in name:
        return {"cfg": 1.0, "steps": 8}
    return None

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
    # The stats below cost a full float copy and four GPU syncs; the f-string
    # used to be built even when DEBUG was off, i.e. on every run.
    if not logger.isEnabledFor(logging.DEBUG):
        return

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

    # At cfg <= 1.0 there is no unconditional pass to shape. Boosting 1.0 to
    # 1.2 switched that pass ON for the early steps, doubling their cost for
    # a change the model was never tuned for.
    if base_cfg <= 1.0:
        return base_cfg

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
    force_full_denoise: bool = False, 
    force_exact_steps: bool = False, 
) -> torch.Tensor:

    # ALBABIT-FIX: Properly define how many schedule steps to generate conceptually.
    if force_exact_steps and 0.0 < denoise < 1.0:
        calc_steps = max(1, int(round(total_steps / denoise)))
    else:
        calc_steps = total_steps

    cached = cache.get(model, scheduler_name, calc_steps)
    if cached is not None:
        bs = cached
    else:
        ms = model.get_model_object("model_sampling")
        bs = comfy.samplers.calculate_sigmas(ms, scheduler_name, calc_steps)

        if flux_shift != 1.0 and scheduler_name != primary_scheduler:
            bs = flux_shift_sigmas(bs, flux_shift)

        bs = correct_sigma_end(bs)
        cache.put(model, scheduler_name, calc_steps, bs)

    # ALBABIT-FIX: ALWAYS truncate the schedule based on denoise. 
    # force_full does NOT bypass truncation, it only forces the final value to 0.0 later
    if denoise < 1.0:
        if force_exact_steps:
            bs = bs[-(total_steps + 1):]
        else:
            n = len(bs) - 1
            if n > 0:
                start_step = max(0, int(n * (1.0 - denoise)))
                bs = bs[start_step:]
                
    if force_full_denoise and len(bs) > 0:
        bs[-1] = 0.0

    return bs

WORKFLOW_PRESETS = [
    # ALBABIT-FIX: "None" renamed "Auto" -- it hides advanced widgets like
    # "None" used to, but shows the model_meta-driven live values (cfg/
    # sampler/scheduler/steps/model_type) instead of hiding everything.
    "Auto",
    "Custom",
    "[F] Flux txt2img",
    "[F] Flux img2img",
    "[F] Flux Inpaint",
    "[F] Flux High-Res Fix",
    "[F] Flux Fast (12 steps)",
    "[F] Flux Quality (28 steps)",
    "[F] Flux Cinematic (30 steps)",

    "[F] Flux Schnell (4 steps)",
    "[F] SD3.5 Turbo (4 steps)",
    "[F] Flux Ultra Fast (8 steps)",

    "[V] WAN txt2vid (30 steps)",
    "[V] WAN img2vid (20 steps)",
    "[V] LTX-Video (25 steps)",
    "[V] LTX 2.3 LowRes (20 steps)",
    "[V] LTX 2.3 HighRes (40 steps)",
    "[V] LTX 2.5 LowRes (20 steps)",
    "[V] LTX 2.5 HighRes (40 steps)",
    "[V] HunyuanVideo (30 steps)",
    "[V] MiniMax H3 T2V (20 steps)",

    "[Q] Draft (4-step / AYS)",
    "[Q] Fast (8-step / AYS)",
    "[Q] Balanced (20-step)",
    "[Q] Quality (35-step)",
    "[Q] Cinema (60-step)",

    "[Q] z_image (25 steps)",
    "[Q] Lumina2 (25 steps)",
]

def flux_shift_sigmas(sigmas: torch.Tensor, shift: float) -> torch.Tensor:

    if shift <= 0:
        raise ValueError(f"flux_shift must be > 0, got {shift}")

    if shift == 1.0:
        return sigmas

    denominator = 1.0 + (shift - 1.0) * sigmas

    denominator = torch.clamp(denominator, min=1e-6)

    shifted = shift * sigmas / denominator
    return shifted

def get_sd_turbo_sigmas(model, steps: int, denoise: float) -> torch.Tensor:
    """
    Mirrors ComfyUI's own SDTurboScheduler node (comfy_extras/nodes_custom_sampler.py)
    exactly. SD-Turbo/SDXL-Turbo are only distilled at 10 fixed discrete timesteps
    (99, 199, ..., 999), not across the continuous sigma space the standard
    schedulers (karras/normal/simple/...) sample from -- using one of those on a
    Turbo checkpoint asks the model to denoise at noise levels it was never
    trained to be good at.
    """
    model_sampling = model.get_model_object("model_sampling")
    start_step = 10 - int(10 * denoise)
    timesteps = torch.flip(torch.arange(1, 11) * 100 - 1, (0,))[start_step:start_step + steps]
    sigmas = model_sampling.sigma(timesteps)
    return torch.cat([sigmas, sigmas.new_zeros([1])])

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

    # ALBABIT-FIX: ALWAYS truncate the schedule based on denoise. 
    # force_full does NOT prevent truncation, it only ensures the last sigma is 0.0
    if denoise < 1.0:
        if force_exact:
            sigmas = sigmas[-(steps + 1):]
        else:
            total_s = len(sigmas) - 1
            if total_s > 0:
                start_step = max(0, int(total_s * (1.0 - denoise)))
                sigmas = sigmas[start_step:]

    # RADIANCE-AUDIT v2.6.0 [CRITICAL]: force_full must run for denoise==1.0
    # too so Flow-Matching schedules with non-zero terminal sigma get
    # clamped when the user asked for terminal_sigma_to_zero.
    if force_full and len(sigmas) > 0:
        sigmas[-1] = 0.0

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

def apply_pag_to_model(model, pag_scale: float, cfg: Optional[float] = None):

    if pag_scale <= 0:
        return model

    # DEFECT: the patch below only acts on the batch slices whose cond_or_uncond
    # entry is 1, i.e. the uncond pass. ComfyUI skips the uncond pass entirely at
    # cfg <= 1.0, and even when it is forced to run, the CFG combine at cfg == 1.0
    # is `uncond + 1.0 * (cond - uncond)` == cond, so a perturbed uncond cannot
    # reach the output. The "PAG applied at scale" log used to fire at
    # registration regardless, and this sampler's own cfg default is 1.0 (Flux),
    # so the common case was a log line claiming PAG ran while nothing happened.
    # Say so at registration, where the user can still act on it.
    if cfg is not None and cfg <= 1.0:
        logger.warning(
            "[Radiance] pag_scale=%.2f has no effect at cfg=%.2f. PAG here perturbs "
            "the unconditional pass, which ComfyUI does not run at cfg <= 1.0 (and "
            "which cancels out of the CFG combine at exactly 1.0). Raise cfg above "
            "1.0 to use PAG, or set pag_scale to 0 to silence this.",
            pag_scale, cfg,
        )
        return model

    try:

        model_pag = model.clone()

        if hasattr(model_pag, "model_options"):
            model_pag.model_options = model_pag.model_options.copy()
            model_pag.model_options["pag_scale"] = pag_scale

        def pag_attention_patch(q, k, v, extra_options):

            cond_or_uncond = extra_options.get("cond_or_uncond", [0])

            # ComfyUI's attn1 patch sets extra_options["block"] = ("middle", i)
            # plus "block_index". There is no "block_type" key -- this used to
            # read `extra_options.get("block_type", "unknown")` and bail unless
            # it equalled "middle", so the default never matched and the patch
            # returned q, k, v unmodified on EVERY call while the line below
            # still logged "PAG applied with scale ...". Read the key that
            # exists, and tolerate either shape.
            block = extra_options.get("block")
            block_name = block[0] if isinstance(block, (tuple, list)) and block else block

            if 1 not in cond_or_uncond or block_name != "middle":
                return q, k, v

            k_out = k.clone()
            v_out = v.clone()

            num_cond = len(cond_or_uncond)
            batch_size = q.shape[0]
            # BUG-PAG-CHUNK FIX: use ceil division so the last element is always
            # perturbed when batch_size is not cleanly divisible by num_cond.
            # Integer // silently truncated the final batch slice.
            chunk_size = math.ceil(batch_size / num_cond) if num_cond > 0 else batch_size

            for idx, cond_type in enumerate(cond_or_uncond):
                if cond_type == 1:
                    start = idx * chunk_size
                    end = min(start + chunk_size, batch_size)

                    # Blend by pag_scale rather than replacing outright.
                    # `pag_scale` was previously used only as an on/off gate and
                    # stashed in model_options where nothing read it, so 0.1 and
                    # 5.0 produced bit-identical results.
                    #
                    # NOTE: this is a self-attention perturbation, not the full
                    # Ahn et al. 2024 method -- true PAG needs a third forward
                    # pass combined as eps_u + s(eps_c - eps_u) + s_pag(eps_c -
                    # eps_p), which ComfyUI's patch API cannot express here. The
                    # tooltip says so.
                    #
                    # DEFECT: this used to clamp to 1.0 while the widget is
                    # min 0.0 / max 5.0, so every value from 1.0 to 5.0 produced
                    # a bit-identical render and four fifths of the slider was
                    # dead. The blend is a lerp toward q; past w=1 it keeps
                    # going in the same direction, which is what a scale above
                    # full substitution should mean. Values <= 1.0 are
                    # unchanged, so existing graphs render identically.
                    w = max(float(pag_scale), 0.0)
                    k_out[start:end] = k[start:end] * (1.0 - w) + q[start:end] * w
                    v_out[start:end] = v[start:end] * (1.0 - w) + q[start:end] * w

            return q, k_out, v_out

        model_pag.set_model_attn1_patch(pag_attention_patch)

        logger.info(
            "PAG-style self-attention perturbation applied at scale %.3f. This "
            "is not the full Ahn et al. 2024 method (no separate perturbed "
            "forward pass); treat it as a mid-block attention guidance term.",
            pag_scale,
        )
        return model_pag

    except (AttributeError, RuntimeError, TypeError) as e:

        logger.warning(f"Failed to apply PAG: {e}, using original model")
        return model

AYS_ANCHORS = {
    "sdxl": [14.615, 6.315, 3.771, 1.181, 0.468, 0.131, 0.029, 0.0],
    "sd1.5": [14.615, 6.475, 3.861, 2.697, 1.886, 1.396, 0.963, 0.652, 0.399, 0.152, 0.029, 0.0],

    "flux": [1.0, 0.90, 0.70, 0.45, 0.22, 0.08, 0.02, 0.0],
    "sd3": [14.615, 6.291, 3.438, 1.566, 0.741, 0.288, 0.079, 0.0],

    "wan": [1.0, 0.92, 0.78, 0.58, 0.35, 0.15, 0.04, 0.0],

    "ltxv": [1.0, 0.88, 0.68, 0.42, 0.20, 0.07, 0.015, 0.0],
}

def get_ays_sigmas(model_type: str, steps: int) -> Optional[torch.Tensor]:

    key = model_type if model_type in AYS_ANCHORS else None
    if key is None:

        # ALBABIT-FIX: renamed from "sd35" to "sd3.5"
        if model_type in ("sd3.5",):
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
        f"═══ Radiance Sampler v{_RADIANCE_VERSION} ═══",
        f"Model: {detected_type} | Target Steps: {steps} | Scheduler: {scheduler}{'  [AYS]' if ays_active else ''}{video_tag}",
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
            # ALBABIT-FIX: Display 1-based step counts for user clarity (1 to N) instead of 0-based
            stage_steps = s_end - s_start
            speed = stage_steps / t if t > 0.001 else 0
            lines.append(
                f"  Stage {stage_num}: Steps {s_start + 1}→{s_end} [{samp}] "
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

# ── Per-frame seeding ────────────────────────────────────────────────────────
#
# Two defects lived in every `torch.manual_seed(seed + f)` in this file.
#
#   1. Overflow. The sampler's `seed` widget is min 0, max 0xFFFFFFFFFFFFFFFF.
#      `seed + f` on a 5D latent with T >= 2 therefore exceeds the 64-bit range
#      torch.manual_seed accepts and raises
#      "RuntimeError: Overflow when unpacking long". The Uniform/Gaussian
#      branch in generate_noise sits above that function's try/except, so it
#      was not even caught: seed=2**64-1 with any of Uniform, Perlin, Spectral,
#      Brownian, Simplex, Voronoi or Curl hard-failed the node.
#
#   2. Correlation between neighbouring seeds. `seed + f` makes run S frame f
#      bit-identical to run S+1 frame f-1, so incrementing the seed produced a
#      one-frame temporal SHIFT of the same noise rather than an independent
#      draw. Anyone stepping the seed to explore variations was re-rendering
#      the same noise field.
#
# derive_frame_seed mixes seed and frame through splitmix64's finalizer, which
# decorrelates adjacent inputs and stays inside [0, 2**64-1] by construction.
_SEED_MASK_64 = 0xFFFFFFFFFFFFFFFF
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15

#: Uniform noise is scaled to unit variance: Var(U[-s, s]) = s^2/3, so s = sqrt(3).
_UNIFORM_HALF_RANGE = 3.0 ** 0.5


def _mix64(x: int) -> int:
    """splitmix64 finalizer. Avalanches every input bit across all 64 outputs."""
    x &= _SEED_MASK_64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _SEED_MASK_64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _SEED_MASK_64
    return (x ^ (x >> 31)) & _SEED_MASK_64


def derive_frame_seed(seed: Optional[int], frame: int, stream: int = 0) -> int:
    """A per-frame seed in [0, 2**64-1], independent across both seed and frame.

    *stream* separates independent noise sequences that share a frame index
    (e.g. the two draws per step in a temporally correlated walk).
    """
    base = int(seed if seed is not None else 0) & _SEED_MASK_64
    offset = ((int(frame) + 1) * _SPLITMIX_GAMMA + int(stream)) & _SEED_MASK_64
    return _mix64(base ^ _mix64(offset))


def _fill_frame_(dst: torch.Tensor, kind: str, generator) -> torch.Tensor:
    """Fill *dst* in place with one frame of noise, allocating nothing.

    Writing straight into a slice of the output buffer is what keeps peak
    memory at one frame instead of T frames; see generate_noise.
    """
    try:
        if kind == "Gaussian":
            return dst.normal_(generator=generator)
        return dst.uniform_(-_UNIFORM_HALF_RANGE, _UNIFORM_HALF_RANGE, generator=generator)
    except RuntimeError:
        # Narrow dtypes (fp8) have no in-place normal_/uniform_ kernel. One
        # frame-sized float32 temporary is still O(1/T) of the old list.
        if kind == "Gaussian":
            src = torch.randn(tuple(dst.shape), device=dst.device, generator=generator)
        else:
            src = (
                torch.rand(tuple(dst.shape), device=dst.device, generator=generator) * 2 - 1
            ) * _UNIFORM_HALF_RANGE
        return dst.copy_(src)


def _frame_generator(device) -> torch.Generator:
    """A torch.Generator bound to *device*, falling back to CPU if unsupported."""
    try:
        return torch.Generator(device=device)
    except (RuntimeError, TypeError):
        return torch.Generator()


def _temporally_correlate(
    noise_fn, shape: tuple, device: torch.device, alpha: float = 0.6,
    seed: Optional[int] = None,
) -> torch.Tensor:

    B, C, T, H, W = shape
    frame_shape = (B, C, H, W)

    # MEMORY: this used to append T tensors of (B, C, H, W) to a Python list and
    # then torch.stack them, so the list and the stacked copy were alive
    # together, i.e. 2x the sequence at the moment of the stack on top of
    # whatever the sampler already held. Write each frame straight into a
    # pre-allocated buffer instead: peak is the buffer plus two working frames,
    # independent of T.
    result = torch.empty(shape, device=device)

    if seed is not None:
        torch.manual_seed(derive_frame_seed(seed, -1, stream=1))
    prev = noise_fn(frame_shape, device)
    scale = math.sqrt(1 - alpha ** 2)
    for f in range(T):
        if seed is not None:
            torch.manual_seed(derive_frame_seed(seed, f))
        curr = noise_fn(frame_shape, device)
        # prev is carried into the next iteration, so it has to stay a tensor
        # of its own rather than an alias of the output slice.
        prev = alpha * prev + scale * curr
        result[:, :, f].copy_(prev)
    del prev, curr

    result -= result.mean()
    std = result.std().clamp(min=1e-6)
    return result.div_(std)

def _perlin_noise(shape: tuple, device: torch.device, seed: Optional[int] = None) -> torch.Tensor:

    if len(shape) == 5 and shape[2] > 1:
        return _temporally_correlate(_perlin_noise_2d, shape, device, seed=seed)

    # RADIANCE-AUDIT v2.6.0 [IMPORTANT]: honour seed in the 2D path too.
    if seed is not None:
        torch.manual_seed(seed)
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

    # RADIANCE-AUDIT v2.6.0 [IMPORTANT]: honour seed in the 2D path too.
    if seed is not None:
        torch.manual_seed(seed)
    return _spectral_noise_2d(shape, device)

_freq_grid_cache: Dict[tuple, torch.Tensor] = {}

def _get_freq_grid(H: int, W: int, device: torch.device) -> torch.Tensor:
    """Return a cached 1/f frequency weight grid for spectral noise."""
    key = (H, W, str(device))
    if key not in _freq_grid_cache:
        freqs_h = torch.fft.fftfreq(H, device=device).abs()
        freqs_w = torch.fft.rfftfreq(W, device=device).abs()
        grid = torch.sqrt(
            freqs_h.unsqueeze(-1) ** 2 + freqs_w.unsqueeze(0) ** 2
        ).clamp(min=1e-6)
        weight = 1.0 / grid
        weight[0, 0] = 0.0  # zero DC component
        _freq_grid_cache[key] = weight
        # Evict oldest entry if cache grows too large (e.g. many resolutions)
        if len(_freq_grid_cache) > 32:
            oldest = next(iter(_freq_grid_cache))
            del _freq_grid_cache[oldest]
    return _freq_grid_cache[key]


def _spectral_noise_2d(shape: tuple, device: torch.device) -> torch.Tensor:

    white = torch.randn(shape, device=device)
    fft = torch.fft.rfft2(white)

    # PERF-FREQGRID FIX: cache the frequency weight grid per (H, W, device)
    # to avoid recomputing it on every frame in video generation.
    weight = _get_freq_grid(shape[-2], shape[-1], device)

    # Broadcast to match fft dims (B*C, H, W/2+1) when fft.ndim > 2
    w = weight
    for _ in range(fft.ndim - 2):
        w = w.unsqueeze(0)

    filtered = fft * w
    result = torch.fft.irfft2(filtered, s=(shape[-2], shape[-1]))

    result = result - result.mean()
    std = result.std().clamp(min=1e-6)
    return result / std

def _brownian_noise(
    shape: tuple, device: torch.device, frames: Optional[int] = None,
    seed: Optional[int] = None,
) -> torch.Tensor:

    # MEMORY: both walks below used to build a Python list of T frames and then
    # torch.stack it, holding the list and the stacked copy at once. They now
    # write each step into a pre-allocated buffer, so peak is the buffer plus
    # the single carried frame. Seeds go through derive_frame_seed; see the
    # note above it for the overflow and adjacent-seed defects that fixes.
    if len(shape) == 5 and shape[2] > 1:
        B, C, T, H, W = shape
        alpha = 0.7

        frame_shape = (B, C, H, W)
        out = torch.empty(shape, device=device)
        if seed is not None:
            torch.manual_seed(derive_frame_seed(seed, -1, stream=1))
        prev = torch.randn(frame_shape, device=device)
        scale = math.sqrt(1 - alpha ** 2)
        for f in range(T):
            if seed is not None:
                torch.manual_seed(derive_frame_seed(seed, f))
            prev = alpha * prev + scale * torch.randn(frame_shape, device=device)
            out[:, :, f].copy_(prev)
        return out

    if frames is None or (len(shape) == 4 and shape[0] == 1):
        return _spectral_noise(shape, device, seed=seed)

    # RADIANCE-AUDIT v2.6.0 [IMPORTANT]: seed the 4D-with-frames path. The
    # old code used the global torch RNG state unmodified, so the same
    # seed could produce different output depending on upstream calls.
    alpha = 0.7
    out = torch.empty(shape, device=device)
    if seed is not None:
        torch.manual_seed(derive_frame_seed(seed, -1, stream=1))
    prev = torch.randn(shape[1:], device=device)
    scale = math.sqrt(1 - alpha ** 2)
    for f in range(shape[0]):
        if seed is not None:
            torch.manual_seed(derive_frame_seed(seed, f))
        prev = alpha * prev + scale * torch.randn(shape[1:], device=device)
        out[f].copy_(prev)
    return out


# ── v2.4 Phase 4: New noise generators (pure PyTorch, no native deps) ─────────

def _simplex_noise(shape: tuple, device: torch.device,
                   seed: Optional[int] = None) -> torch.Tensor:
    """
    Approximate Simplex-style noise via trilinear hash grid.
    Faster than Perlin for high-resolution latents; smoother frequency spectrum.
    Based on Stefan Gustavson's method adapted for GPU tensors.
    """
    if seed is not None:
        torch.manual_seed(seed)

    if len(shape) == 5:
        B, C, T, H, W = shape
        # MEMORY: the list comprehension here held all T frames alive at once
        # and then torch.stack doubled that. Write each frame into the output
        # buffer as it is produced. Seed via derive_frame_seed: `(seed or 0) + t`
        # overflowed at seed=2**64-1 and shifted rather than changed the field
        # between adjacent seeds.
        out = torch.empty(shape, device=device)
        for t in range(T):
            out[:, :, t].copy_(
                _simplex_noise((B, C, H, W), device, seed=derive_frame_seed(seed, t))
            )
        return out

    # Work on last 2 dims (spatial)
    out_shape = shape
    if len(shape) == 4:
        B, C, H, W = shape
    else:
        return torch.randn(shape, device=device)  # fallback for unknown shapes

    # RADIANCE-AUDIT v2.6.0 [CRITICAL]: salt the hash with the seed so
    # different seeds actually produce different simplex noise. The old
    # 4D path derived everything from grid_x/grid_y and was fully
    # deterministic regardless of seed.
    seed_salt = int(seed) if seed is not None else 0
    octave_salt = (seed_salt * 2654435761) & 0xFFFFFFFF

    octaves = 4
    persistence = 0.5
    lacunarity = 2.0
    result = torch.zeros(B, C, H, W, device=device)
    amplitude = 1.0
    frequency = 1.0
    max_amplitude = 0.0

    for octave_idx in range(octaves):
        # RADIANCE-AUDIT v2.6.0 [CRITICAL]: per-octave jitter driven by the
        # seeded RNG so the grid itself shifts across seeds.
        jitter_y = torch.rand((), device=device).item() if seed is not None else 0.0
        jitter_x = torch.rand((), device=device).item() if seed is not None else 0.0
        ys = torch.linspace(0, frequency, H, device=device) + jitter_y
        xs = torch.linspace(0, frequency, W, device=device) + jitter_x
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')

        # Hash-based gradient (deterministic within an octave,
        # seed-dependent across octaves)
        octave_offset = (octave_salt + octave_idx * 0x9E3779B1) & 0xFFFFFFFF
        ix = (grid_x.long() + (octave_offset & 0xFFFF)) % 255
        iy = (grid_y.long() + ((octave_offset >> 16) & 0xFFFF)) % 255
        hash_val = ((ix * 1619 + iy * 31337 + octave_offset) & 0xFFFF).float() / 32768.0 - 1.0
        fx = grid_x - grid_x.long().float()
        fy = grid_y - grid_y.long().float()

        # Fade (smoothstep — quintic for C2 continuity)
        u = fx * fx * fx * (fx * (fx * 6 - 15) + 10)
        v = fy * fy * fy * (fy * (fy * 6 - 15) + 10)

        # Gradient vectors from hash angle
        angle = hash_val * 2.0 * math.pi
        gx = torch.cos(angle)
        gy = torch.sin(angle)

        # BUG-SIMPLEX-GRAD FIX: sum all four bilinear corner contributions.
        # The old code only computed the (0,0) corner, producing ~25% amplitude
        # and no smooth tiling across cell boundaries.
        #   corner (0,0): local coords (fx,     fy    )
        #   corner (1,0): local coords (fx-1,   fy    )
        #   corner (0,1): local coords (fx,     fy-1  )
        #   corner (1,1): local coords (fx-1,   fy-1  )
        g00 = (gx * fx       + gy * fy)       * (1.0 - u) * (1.0 - v)
        g10 = (gx * (fx - 1) + gy * fy)       *        u  * (1.0 - v)
        g01 = (gx * fx       + gy * (fy - 1)) * (1.0 - u) *        v
        g11 = (gx * (fx - 1) + gy * (fy - 1)) *        u  *        v
        grad = (g00 + g10 + g01 + g11).unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)

        result += amplitude * grad
        max_amplitude += amplitude
        amplitude *= persistence
        frequency *= lacunarity

    return (result / max_amplitude)


def _voronoi_noise(shape: tuple, device: torch.device,
                   seed: Optional[int] = None) -> torch.Tensor:
    """
    Cellular / Voronoi noise. Each pixel value = distance to nearest random point.
    Produces an organic, cellular texture pattern useful for breaking up uniform noise.
    """
    if len(shape) == 5:
        B, C, T, H, W = shape
        # MEMORY + SEED: same list-plus-stack and `(seed or 0) + t` defects as
        # the simplex path above. See derive_frame_seed.
        out = torch.empty(shape, device=device)
        for t in range(T):
            out[:, :, t].copy_(
                _voronoi_noise((B, C, H, W), device, seed=derive_frame_seed(seed, t))
            )
        return out

    if len(shape) == 4:
        B, C, H, W = shape
    else:
        return torch.randn(shape, device=device)

    n_points = max(8, (H * W) // 256)  # ~1 point per 16×16 cell

    # BUG-VORONOI-SEED FIX: use a dedicated Generator so the point placement
    # is seed-deterministic regardless of external RNG state.  The old
    # torch.manual_seed() call was overridden by entropy consumed in the frame
    # loop before points_y/points_x were sampled.
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed) if seed is not None else 0)
    points_y = torch.rand(n_points, device=device, generator=gen) * H
    points_x = torch.rand(n_points, device=device, generator=gen) * W

    ys = torch.arange(H, device=device).float()
    xs = torch.arange(W, device=device).float()
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')

    # Distance to nearest voronoi point - Memory-safe chunked vectorization
    # AUDIT FIX: Process points in batches of 256 to prevent OOM on 2K/4K latents
    min_dist_sq = torch.full((H, W), float('inf'), device=device)
    batch_size = 256
    
    for i in range(0, n_points, batch_size):
        end = min(i + batch_size, n_points)
        py = points_y[i:end].view(-1, 1, 1)
        px = points_x[i:end].view(-1, 1, 1)
        
        dy = grid_y.unsqueeze(0) - py
        dx = grid_x.unsqueeze(0) - px
        chunk_dist_sq = dy * dy + dx * dx
        
        min_dist_sq = torch.minimum(min_dist_sq, torch.min(chunk_dist_sq, dim=0)[0])
        del dy, dx, chunk_dist_sq

    min_dist = torch.sqrt(min_dist_sq)

    # Normalise to [-1, 1]
    d_max = min_dist.max().clamp(min=1e-6)
    noise_2d = (min_dist / d_max) * 2.0 - 1.0

    return noise_2d.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)


def _curl_noise(shape: tuple, device: torch.device,
                seed: Optional[int] = None) -> torch.Tensor:
    """
    Divergence-free Curl noise derived from the gradient of a Perlin potential field.
    Produces swirling, fluid-like patterns. Excellent for organic motion blur or
    fire/smoke latent initializations.
    """
    if seed is not None:
        torch.manual_seed(seed)

    if len(shape) == 5:
        B, C, T, H, W = shape
        # Already writes into a pre-allocated buffer. The seed derivation is the
        # fix here: `(seed or 0) + t` overflowed torch.manual_seed at
        # seed=2**64-1 and made adjacent seeds a one-frame shift of each other.
        result = torch.empty(shape, device=device)
        for t in range(T):
            result[:, :, t] = _curl_noise((B, C, H, W), device, seed=derive_frame_seed(seed, t))
        return result

    if len(shape) == 4:
        B, C, H, W = shape
    else:
        return torch.randn(shape, device=device)

    eps = 1.0 / max(H, W)

    # Generate potential field as spectral noise (acts as the stream function)
    potential = _spectral_noise_2d(shape, device)  # (B, C, H, W)

    # Compute curl via finite differences on the potential field
    # curl_x =  dPhi/dy,  curl_y = -dPhi/dx
    phi = potential

    # Pad for boundary conditions
    phi_pad = torch.nn.functional.pad(phi, (1, 1, 1, 1), mode='replicate')
    dPhi_dy = (phi_pad[:, :, 2:, 1:-1] - phi_pad[:, :, :-2, 1:-1]) / (2 * eps)
    dPhi_dx = (phi_pad[:, :, 1:-1, 2:] - phi_pad[:, :, 1:-1, :-2]) / (2 * eps)

    # Alternate channels between x and y components for multi-channel latents
    result = torch.zeros_like(potential)
    for c in range(C):
        result[:, c] = dPhi_dy[:, c] if c % 2 == 0 else -dPhi_dx[:, c]

    # Normalise
    rms = result.pow(2).mean().sqrt().clamp(min=1e-6)
    return result / rms


# ── End v2.4 noise generators ──────────────────────────────────────────────────

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
        # MEMORY: this branch built a Python list of T (B, C, H, W) tensors and
        # then torch.stack'd it, so the list and the stacked result were alive
        # simultaneously -- two full copies of the whole sequence at the peak,
        # before the sampler's own latent/work/noise/stage buffers are counted.
        # Allocate the output once and fill each frame in place. Peak is now one
        # copy of the sequence, flat in T.
        #
        # SEED: `torch.manual_seed(seed + f)` lived ABOVE the try/except below,
        # so the overflow at seed=2**64-1 was not even caught. It also made
        # adjacent seeds a one-frame shift of one another. derive_frame_seed
        # fixes both; a dedicated Generator additionally stops this from
        # stomping the global RNG state that the rest of the graph shares.
        out = torch.empty(shape, device=device, dtype=dtype)
        gen = _frame_generator(out.device)
        for f in range(T):
            gen.manual_seed(derive_frame_seed(seed, f))
            _fill_frame_(out[:, :, f], noise_type, gen)
        return out

    torch.manual_seed(int(seed) & _SEED_MASK_64)

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
        elif noise_type == "Simplex":
            return _simplex_noise(shape, device, seed=seed).to(dtype)
        elif noise_type == "Voronoi":
            return _voronoi_noise(shape, device, seed=seed).to(dtype)
        elif noise_type == "Curl":
            return _curl_noise(shape, device, seed=seed).to(dtype)
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
    sigmas: Optional[torch.Tensor] = None,
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
                except Exception as exc:
                    logger.warning("[nodes_sampler] _pad: %s", exc)
            merged.append([blended, extra])
        return merged
    elif mode == "sequential":
        # BUG-SEQCOND-FRAC FIX: ComfyUI start_percent/end_percent are sigma-space
        # fractions (fraction of the noise-level trajectory), NOT step-count
        # fractions. At non-uniform schedulers (Karras, SGM) the midpoint in
        # steps is not the midpoint in sigma. Derive the fraction from the
        # actual sigma schedule when available.
        if sigmas is not None and len(sigmas) > 1:
            sigma_max = float(sigmas[0].clamp(min=1e-8))
            split_idx = max(0, min(split_step, len(sigmas) - 1))
            frac = 1.0 - float(sigmas[split_idx]) / sigma_max
        else:
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
    """No-op. Kept so existing graphs and imports keep working.

    DEFECT: this used to write ``new_entry[1]["encoder_target"] = target_key``
    and the sampler logged "Conditioning routed to clip_g". Nothing reads
    ``encoder_target``: not ComfyUI 0.32.0, not 0.36.0, and nothing in Radiance.
    The widget reported success and did nothing.

    It cannot be made to work from here either. Which text encoder produced an
    embedding is decided at ENCODE time; by the time a CONDITIONING list reaches
    a sampler it is a finished tensor with no encoder identity left to change.
    Routing to a specific slot means encoding with that slot's CLIP up front
    (a separate CLIPTextEncode per encoder, e.g. ComfyUI's CLIPTextEncodeSDXL),
    which is a graph change, not a sampler flag.

    Returns *cond* unchanged and warns, rather than returning a copy carrying a
    key that only makes the no-op harder to notice.
    """
    if not target_key or target_key == "Auto":
        return cond

    logger.warning(
        "[Radiance] conditioning_clip_target='%s' is ignored. Which text encoder "
        "produced a CONDITIONING is fixed when it is encoded, so a sampler cannot "
        "re-route it. Encode per-encoder instead (e.g. CLIPTextEncodeSDXL for "
        "clip_l/clip_g, or a dedicated T5 encode for t5xxl) and feed that in. "
        "Set this back to 'Auto' to silence this warning.",
        target_key,
    )
    return cond

# ══════════════════════════════════════════════════════════════════════════════
#  Overlapping temporal windows, long-video sampling
# ══════════════════════════════════════════════════════════════════════════════
#
# A video model attends across the temporal axis, so slicing a long latent into
# independent chunks and denoising each on its own produces visible seams and
# content that drifts between chunks. The established fix is overlapping
# windows with per-step blending: the clip is covered by windows that share
# some latent frames, EVERY denoising step is evaluated per window, and the
# overlapping frames are blended at that step before the next one begins.
#
# The per-step part is the whole point. Blending windows that have each been
# denoised to completion re-creates exactly the seams the technique exists to
# remove, because by then the two windows have committed to different content.
#
# The weighting below is the temporal form of what RadianceUpscaleVideo already
# does for pixels (nodes/upscale/upscale.py), including its half-sample ramp
# offset, which fixed a real black-frame bug: a ramp of the literal form
# sin(pi * i / (2 * half)) is 0 at i == 0, so with overlap=1 the single shared
# frame received a zero weight from both neighbours and normalised to black.
# Offsetting by half a sample keeps the shape and makes every weight strictly
# positive. The squaring is taken from core/tiling.py's spatial ramp: sin^2 and
# cos^2 sum to exactly 1, so two overlapping ramps are a partition of unity
# before any normalisation, not merely after it.


def plan_temporal_windows(
    total_frames: int, window_size: int, overlap: int
) -> List[Tuple[int, int]]:
    """Window bounds ``[(f0, f1), ...]`` covering ``range(total_frames)``.

    The first window starts at frame 0 and the last ends at *total_frames*, so
    every frame is covered. Windows are returned in ascending order and no
    window is contained in another. A clip that fits in one window returns a
    single full-length window, which is what makes windowing a no-op below the
    threshold.

    Pure function of its three arguments, so the same clip and settings always
    produce the same plan, on any machine and on any rerun.
    """
    total_frames = int(total_frames)
    window_size = int(window_size)
    if total_frames <= 0:
        return []
    if window_size <= 0 or total_frames <= window_size:
        return [(0, total_frames)]

    # Shared with the spatial tilers: an overlap >= window collapses the stride
    # to 1 and turns the run into total_frames windows.
    overlap = clamp_overlap(window_size, overlap)
    stride = max(1, window_size - overlap)

    last_start = total_frames - window_size
    starts: List[int] = []
    cursor = 0
    while True:
        start = min(cursor, last_start)
        if not starts or start > starts[-1]:
            starts.append(start)
        if start + window_size >= total_frames:
            break
        cursor += stride

    # Keep full context and the requested overlap at the final join. The
    # weights support its intersection with two preceding windows.
    return [(s, s + window_size) for s in starts]


def temporal_window_weights(
    windows: List[Tuple[int, int]], index: int, device=None, dtype=None
) -> torch.Tensor:
    """Blend weight per frame for ``windows[index]``, shaped (1, 1, L, 1, 1).

    Ramp lengths come from the ACTUAL overlap with each neighbour rather than
    from the requested overlap, because the last window is pulled back to end
    on the final frame and can therefore share more frames with its predecessor
    than the setting asks for. Each new window attenuates all earlier ones by
    its complementary fade, keeping the total weight at one even when three
    windows overlap.

    Every weight is strictly positive, so dividing by the accumulated weight is
    safe and no frame can normalise to zero.
    """
    f0, f1 = windows[index]
    length = f1 - f0
    weight = torch.ones(length, device=device, dtype=dtype or torch.float32)
    if length <= 1:
        return weight.view(1, 1, length, 1, 1)

    lead = windows[index - 1][1] - f0 if index > 0 else 0
    lead = max(lead, 0)
    if lead > 0:
        i = torch.arange(lead, device=device, dtype=weight.dtype)
        weight[:lead] = torch.sin(math.pi * (i + 0.5) / (2 * lead)) ** 2

    # Each new window attenuates all earlier contributors by the complement
    # of its fade-in, keeping their total at one even in a three-way join.
    for next_index in range(index + 1, len(windows)):
        next_start = windows[next_index][0]
        if next_start >= f1:
            break
        fade_length = windows[next_index - 1][1] - next_start
        count = f1 - next_start
        i = torch.arange(count, device=device, dtype=weight.dtype)
        weight[next_start - f0:] *= torch.cos(math.pi * (i + 0.5) / (2 * fade_length)) ** 2

    return weight.view(1, 1, length, 1, 1)


def slice_conds_temporally(c: Dict[str, Any], f0: int, f1: int, total_frames: int) -> Dict[str, Any]:
    """Copy of the model kwargs *c* with temporal entries cut to ``[f0, f1)``.

    Text embeddings carry no temporal axis and are shared by every window
    untouched. Entries that DO carry one -- an image-to-video model's
    ``c_concat`` conditioning frames, for instance -- have to follow the window
    or the model is told about frames it is not being shown.

    Anything that is not a 5D tensor whose temporal dim matches the clip is
    passed through unchanged, which is the safe default: a wrong slice is worse
    than no slice.
    """
    out = {}
    for key, value in c.items():
        if (
            isinstance(value, torch.Tensor)
            and value.ndim == 5
            and value.shape[2] == total_frames
        ):
            out[key] = value[:, :, f0:f1]
        else:
            out[key] = value
    return out


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
    cfg: float = 1.0,
) -> torch.Tensor:

    import comfy.sample as cs

    if latent_samples.ndim != 4:
        raise ValueError(
            f"[Radiance] tile_sample requires 4D latent (B, C, H, W), "
            f"got {latent_samples.ndim}D. Video tile sampling is not supported."
        )

    B, C, H, W = latent_samples.shape
    # An overlap >= tile_size collapses the stride to 1 and turns this into
    # millions of tile inferences; only one of five tilers checked for it.
    tile_overlap = clamp_overlap(tile_size, tile_overlap)
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
            t_out = cs.sample_custom(
                model,
                t_noise,
                cfg=cfg,                                                         
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
            # DEFECT: this used to set `t_out = t_latent`, the UN-DENOISED input
            # slice, feather it into the output and let the run report success.
            # The condition that makes a tile fail is almost always OOM at high
            # resolution, which is the exact condition tiling exists to avoid,
            # so the failure mode was a finished plate with a rectangle of raw
            # latent noise in it and nothing above INFO to say so. A delivery
            # pipeline has to fail the node instead.
            raise RuntimeError(
                f"[Radiance] Tiled sampling failed on tile {idx + 1}/{len(tile_coords)} "
                f"at latent rows {y1}:{y2}, cols {x1}:{x2} (tile_size={tile_size}, "
                f"tile_overlap={tile_overlap}).\n\n"
                f"The node fails rather than substituting the un-denoised input for "
                f"this tile, which would have left raw latent noise in the output.\n\n"
                f"If this is an out-of-memory error, lower tile_size or tile_overlap.\n\n"
                f"Original error: {e}"
            ) from e

        if t_out.device != device or t_out.dtype != latent_samples.dtype:
            t_out = t_out.to(device=device, dtype=latent_samples.dtype)

        th = y2 - y1
        tw = x2 - x1

        if tile_blend == "feather":
            # Border-aware feather via the shared helper. The previous inline
            # ramp started at exactly 0 and was applied to all four edges of
            # every tile, including edges lying on the image border -- those
            # pixels are covered by no other tile, so dividing by the
            # accumulated weight gave 0/1e-6 == 0 and produced a black band
            # around the whole latent (8 image pixels wide after the VAE).
            ov_t, ov_b, ov_l, ov_r = edge_overlaps_from_coords(
                y1, y2, x1, x2, H, W, tile_overlap
            )
            w_tile = blend_weight_2d(
                th, tw, ov_t, ov_b, ov_l, ov_r,
                device=device, dtype=latent_samples.dtype,
            )
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
    sdr_blend: float = 0.0,
    sdr_inject_steps: int = 0,
    sdr_decay: float = 0.65,
) -> str:

    import json
    sigma_min = float(sigmas[sigmas > 0].min()) if (sigmas > 0).any() else 0.0
    sigma_max = float(sigmas.max())
    is_video = frames is not None and frames > 1
    meta = {
        "version": _RADIANCE_VERSION,
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
    if sdr_blend > 0.0:
        meta["sdr_conditioning"] = {
            "sdr_blend": round(sdr_blend, 4),
            "sdr_inject_steps": sdr_inject_steps,
            "sdr_decay": round(sdr_decay, 4),
        }
    return json.dumps(meta, indent=2)
