


# ═══════════════════════════════════════════════════════════════════════════════
#  Radiance Read Models  —  nodes_loader.py
#  v3.1.1: Deduplicated — all shared utilities live in loader_utils.py (SSOT).
#  This file contains only ComfyUI node class definitions and registration.
# ═══════════════════════════════════════════════════════════════════════════════

import json
import logging
import os
import time

import torch
import folder_paths
import comfy.utils
import comfy.model_management
from comfy.cldm.control_types import UNION_CONTROLNET_TYPES

# ── Single Source of Truth: all model-detection / latent / cache utilities ──
from ...loader_utils import (
    RADIANCE_MODEL_MAP,
    CHECKPOINT_PRESETS,
    VIDEO_PRESET_NAMES,
    VIDEO_MODEL_TYPES,
    LATENT_CHANNELS,
    file_fingerprint as _file_fingerprint,
    ensure_model_exists as _ensure_model_exists,
    resolve_divider,
    apply_checkpoint_preset,
    resolve_architecture,
    setup_offload_mode,
    estimate_vram_for_load,
    load_unet_and_baked_vae,
    load_clip_stack,
    load_standalone_vae,
    construct_audio_vae,
    apply_lora_stack,
    _unet_cache,
    _clip_cache,
    _vae_cache,
    _audio_vae_cache,
)
from radiance.model.cache import LRUCache

# ALBABIT-FIX: dedicated cache for RadianceVideoLoader's latent upscale model
# extra (LTX 2.3 / HunyuanVideo SR). Kept separate from the core unet/clip/vae
# caches so it doesn't compete for eviction with the main pipeline models.
_upscale_model_cache = LRUCache()

logger = logging.getLogger("radiance.loader")
from radiance.core.logging import print_premium_loader_hud




# ═══════════════════════════════════════════════════════════════════════════════
#                     RADIANCE LORA STACK NODE  (v3.0)
# ═══════════════════════════════════════════════════════════════════════════════

class RadianceLoraStack:
    """
    Compose up to 5 LoRAs into a LORA_STACK for use with RadianceUnifiedLoader.
    Can accept an upstream LORA_STACK to chain stacks.
    """

    @classmethod
    def INPUT_TYPES(cls):
        lora_list = ["None"] + folder_paths.get_filename_list("loras")
        lora_slot = lambda tooltip: (
            lora_list,
            {"default": "None", "tooltip": tooltip},
        )
        str_slot = lambda tooltip: (
            "FLOAT",
            {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05,
             "tooltip": tooltip},
        )
        return {
            "required": {},
            "optional": {
                "lora_stack":     ("LORA_STACK", {"default": None,
                    "tooltip": "Chain an upstream LORA_STACK before these LoRAs."}),
                "lora_1":         lora_slot("LoRA 1"),
                "lora_1_model":   str_slot("LoRA 1 model strength"),
                "lora_1_clip":    str_slot("LoRA 1 CLIP strength"),
                "lora_2":         lora_slot("LoRA 2"),
                "lora_2_model":   str_slot("LoRA 2 model strength"),
                "lora_2_clip":    str_slot("LoRA 2 CLIP strength"),
                "lora_3":         lora_slot("LoRA 3"),
                "lora_3_model":   str_slot("LoRA 3 model strength"),
                "lora_3_clip":    str_slot("LoRA 3 CLIP strength"),
                "lora_4":         lora_slot("LoRA 4"),
                "lora_4_model":   str_slot("LoRA 4 model strength"),
                "lora_4_clip":    str_slot("LoRA 4 CLIP strength"),
                "lora_5":         lora_slot("LoRA 5"),
                "lora_5_model":   str_slot("LoRA 5 model strength"),
                "lora_5_clip":    str_slot("LoRA 5 CLIP strength"),
            },
        }

    RETURN_TYPES = ("LORA_STACK",)
    RETURN_NAMES = ("lora_stack",)
    FUNCTION = "build_stack"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Generate"
    DESCRIPTION = (
        "Compose up to 5 LoRAs into an accumulating LORA_STACK. "
        "Chain multiple stacks together. Feed into Radiance Read Models."
    )

    def build_stack(
        self,
        lora_stack=None,
        lora_1="None", lora_1_model=1.0, lora_1_clip=1.0,
        lora_2="None", lora_2_model=1.0, lora_2_clip=1.0,
        lora_3="None", lora_3_model=1.0, lora_3_clip=1.0,
        lora_4="None", lora_4_model=1.0, lora_4_clip=1.0,
        lora_5="None", lora_5_model=1.0, lora_5_clip=1.0,
    ) -> tuple:
        stack = list(lora_stack) if lora_stack else []
        for name, ms, cs in [
            (lora_1, lora_1_model, lora_1_clip),
            (lora_2, lora_2_model, lora_2_clip),
            (lora_3, lora_3_model, lora_3_clip),
            (lora_4, lora_4_model, lora_4_clip),
            (lora_5, lora_5_model, lora_5_clip),
        ]:
            if name and name != "None":
                stack.append((name, float(ms), float(cs)))
        return (stack,)


# ═══════════════════════════════════════════════════════════════════════════════
#                     RADIANCE READ MODELS v2.1.0
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_TYPES = [
    "Auto-Detect",
    "flux", "sd3", "sd3.5",
    "sdxl", "sd1.5",
    "hunyuan_video", "wan", "ltxv",  # ALBABIT-FIX: "ltx" → "ltxv" — matches sampler_utils.py
    # ALBABIT-FIX: LTX 2.3 (audio) — distinct VRAM profile from "ltxv" (LTX
    # Video 2B/13B). Was only present in a dead duplicate MODEL_TYPES list
    # above, never in the active one, so "ltxav" was unselectable in Custom
    # mode despite being used by the "LTX Video 2.3" presets.
    "ltxav",
    # ALBABIT-FIX: WAN 2.2 TI2V-5B (48ch VAE, distinct from "wan"'s 16ch) —
    # real bug fix, see model/detect.py.
    "wan_ti2v",
    "lumina2", "z_image",
    "pixart", "aura_flow",
    # ALBABIT-FIX: Cosmos / CogVideoX / Mochi — match Resolution/Sampler model types
    "cosmos", "cogvideox", "mochi",
    # ALBABIT-FIX: Chroma (distilled Flux) and Flux.2 / Flux.2 Klein
    "chroma", "flux2", "flux2-klein",
    # ALBABIT-FIX: MiniMax H3, a joint video+audio DiT, 24ch video latent.
    "minimax",
    # 3.5: ComfyUI 0.32 families (see model/detect.py for the tables).
    "qwen_image", "krea2", "hunyuan_image", "hunyuan_video_15",
    "hidream", "omnigen2", "longcat_image", "kandinsky5", "kandinsky5_image",
]

WEIGHT_DTYPES = ["default", "fp8_e4m3fn", "fp8_e5m2", "fp16", "bf16", "fp32"]
CLIP_DTYPES   = ["default", "fp16", "bf16", "fp8_e4m3fn", "fp32"]
OFFLOAD_MODES = ["none", "cpu_offload", "sequential"]


def _find_wan_moe_companion(unet_name):
    # ALBABIT-FIX: WAN 2.2 uses two expert UNETs (high_noise + low_noise).
    # Derive the companion filename by swapping the noise tag.
    name_lower = unet_name.lower()
    for tag, companion_tag in (("high_noise", "low_noise"), ("low_noise", "high_noise")):
        idx = name_lower.find(tag)
        if idx != -1:
            return unet_name[:idx] + companion_tag + unet_name[idx + len(tag):]
    return None


class RadianceUnifiedLoader:
    """
    Universal diffusion model loader v3.3 — Streamlined and Optimized.
    Zero canvas clutter: LoRAs and ControlNets decoupled to external modular nodes.
    Outputs: MODEL, CLIP, VAE, LORA_STACK.
    """

    @classmethod
    def INPUT_TYPES(cls):
        clip_list = ["None"] + folder_paths.get_filename_list("text_encoders")
        clip_slot = lambda tip: (clip_list, {"default": "None", "tooltip": tip})
        # ALBABIT-FIX: "Baked (from UNET)" — LTX 2.3's text_embedding_projection
        # weights ship inside the main UNET checkpoint (see assemble_clip_paths).
        text_projection_list = ["None", "Baked (from UNET)"] + folder_paths.get_filename_list("text_encoders")
        text_projection_slot = lambda tip: (text_projection_list, {"default": "None", "tooltip": tip})
        # ALBABIT-FIX: AuraFlow has no standalone text encoder file either
        # (official workflow uses CheckpointLoaderSimple only, its HF repo
        # only ships a diffusers-format folder). Same "Baked (from UNET)"
        # mechanism as text_projection, on the t5xxl slot (AuraFlow's real
        # encoder is T5, not clip_l, see assemble_clip_paths/CLIP_SLOT_ORDER).
        t5xxl_list = ["None", "Baked (from UNET)"] + folder_paths.get_filename_list("text_encoders")
        t5xxl_slot = lambda tip: (t5xxl_list, {"default": "None", "tooltip": tip})

        return {
            "required": {
                # ── Preset ──
                "preset": (
                    # ALBABIT-FIX: image loader only lists image-model presets
                    # (video presets are exclusive to RadianceVideoLoader).
                    [k for k in CHECKPOINT_PRESETS if k not in VIDEO_PRESET_NAMES],
                    {"default": "Custom",
                     "tooltip": "Quick-configure for common architectures. "
                                "Overrides model_type, dtypes, offload_mode, and hints "
                                "which CLIP slots are needed."},
                ),
                # ── UNET ──
                "unet_name": (
                    folder_paths.get_filename_list("diffusion_models"),
                    {"tooltip": "Main diffusion model (UNET / DiT / Transformer)."},
                ),
                "weight_dtype": (
                    WEIGHT_DTYPES,
                    {"default": "default",
                     "tooltip": "UNET weight precision. fp8_e4m3fn saves ~40% VRAM vs fp16."},
                ),
                # ── Architecture ──
                "model_type": (
                    # ALBABIT-FIX: image loader only lists image-model
                    # model_types (video model_types are exclusive to
                    # RadianceVideoLoader).
                    [m for m in MODEL_TYPES if m not in VIDEO_MODEL_TYPES],
                    {"default": "Auto-Detect",
                     "tooltip": "'Auto-Detect' reads the checkpoint's key names to determine "
                                "architecture. Override manually if detection fails."},
                ),
                # ── VAE ──
                # ALBABIT-FIX: "Baked VAE (from UNET)" lets checkpoint-style
                # files (e.g. SD3.5) skip the standalone vae_name file, same
                # mechanism as RadianceVideoLoader's LTX 2.3. Appended, not
                # prepended, so the raw combo default stays unchanged for
                # architectures with real separate VAE files.
                "vae_name": (
                    folder_paths.get_filename_list("vae") + ["Baked VAE (from UNET)"],
                    {"tooltip": "VAE for encoding/decoding latents. "
                                "'Baked VAE (from UNET)' extracts it from the checkpoint, "
                                "for architectures whose standard release bundles the VAE "
                                "into the main file instead of shipping it separately."},
                ),
            },
            "optional": {
                # ── Named CLIP slots ──
                "clip_l":      clip_slot(
                    "CLIP-L (text encoder). Used by: SD1.5, SDXL, Flux, SD3."),
                "clip_g":      clip_slot(
                    "CLIP-G (text encoder). Used by: SDXL, SD3, SD3.5."),
                "t5xxl":       t5xxl_slot(
                    "T5-XXL (text encoder). Used by: Flux, SD3, SD3.5, Wan, PixArt, LTX (pre-2.3). "
                    "'Baked (from UNET)' loads it from the main checkpoint -- AuraFlow ships no "
                    "standalone text encoder file."),
                "llm_encoder": clip_slot(
                    "LLM encoder. Used by: HunyuanVideo (Llava-Llama3), LTX 2.3 (Gemma 3), "
                    "Lumina2 (Gemma-2), Z-Image (Qwen3), Flux.2 (Mistral-3/Qwen3)."),
                "text_projection": text_projection_slot(
                    "Text projection matrix. Used by: LTX 2.3 (with Gemma 3 llm_encoder). "
                    "'Baked (from UNET)' loads it from the main LTX 2.3 checkpoint, like the native LTXV Audio Text Encoder Loader."),
                # ── CLIP precision ──
                "clip_dtype": (
                    CLIP_DTYPES,
                    {"default": "default",
                     "tooltip": "CLIP weight precision. Independent from UNET. "
                                "For Flux T5XXL: fp8 saves ~4.7 GB vs fp16."},
                ),
                # ── Offload ──
                "offload_mode": (
                    OFFLOAD_MODES,
                    {"default": "none",
                     "tooltip": "none = GPU only. "
                                "cpu_offload = CLIP loaded to CPU RAM. "
                                "sequential = enable ComfyUI sequential CPU offload (8–12 GB GPUs)."},
                ),
                # ── LoRA (external stack input) ──
                "lora_stack":      ("LORA_STACK", {"default": None,
                    "tooltip": "Accept a LORA_STACK from RadianceLoraStack node."}),
                # ── Options ──
                "check_vram":  (["On", "Off"], {"default": "On",
                    "tooltip": "Estimate VRAM before load and warn if tight."}),
                "use_cache":   (["On", "Off"], {"default": "On",
                    "tooltip": "Cache loaded models. Skips disk I/O when re-running "
                               "with the same files. Cache auto-invalidates if files change."}),
                "lora_on_error": (["warn", "raise"], {"default": "raise",
                    "tooltip": "'warn' skips failed LoRA and continues. "
                               " 'raise' stops execution."}),
                "auto_download": ("BOOLEAN", {"default": True,
                    "tooltip": "If a selected model is missing and is one Radiance knows, download it on first run from its pinned Hugging Face source, checked against its SHA-256 before it is installed (large: 4 to 60 GB). Gated repositories (FLUX.2-dev, FLUX.2-klein 9B, LTX-2.5) need their licence accepted on Hugging Face and HF_TOKEN set. RADIANCE_ALLOW_DOWNLOADS=0 always stops downloads."}),
            },
        }

    RETURN_TYPES  = ("MODEL", "CLIP", "VAE", "LORA_STACK", "STRING")
    RETURN_NAMES  = ("model", "clip", "vae", "lora_stack", "model_meta")
    FUNCTION      = "load_radiance_stack"
    CATEGORY      = "FXTD STUDIOS/Radiance/◎ Generate"
    DESCRIPTION   = (
        "Universal loader v3.3 — streamlined to be extremely visual and modular. "
        "Auto-detects architecture, auto-tunes weights/offload, supports "
        "chainable LORA_STACK, and outputs JSON model metadata."
    )

    @staticmethod
    def _load_vae_sd_metadata(vae_path):
        # ALBABIT-FIX: base loader does not need VAE metadata.
        return comfy.utils.load_torch_file(vae_path), None

    def load_radiance_stack(
        self,
        preset,
        unet_name,
        weight_dtype,
        model_type,
        vae_name,
        clip_l="None",
        clip_g="None",
        t5xxl="None",
        llm_encoder="None",
        text_projection="None",
        clip_dtype="default",
        offload_mode="none",
        lora_stack=None,
        check_vram="On",
        use_cache="On",
        lora_on_error="raise",
        auto_download=True,
    ):
        def _none(val):
            return None if val in ("None", "", None) else val

        clip_l = _none(clip_l)
        clip_g = _none(clip_g)
        t5xxl = _none(t5xxl)
        llm_encoder = _none(llm_encoder)
        text_projection = _none(text_projection)

        load_start  = time.time()
        info_lines  = []
        caching     = use_cache == "On"

        # ════════════════════════════════════════════════════════════════
        # 0. APPLY PRESET
        # ════════════════════════════════════════════════════════════════
        model_type, weight_dtype, clip_dtype, overrides = apply_checkpoint_preset(
            preset, model_type, weight_dtype, clip_dtype, info_lines
        )

        # ════════════════════════════════════════════════════════════════
        # 1. RESOLVE ARCHITECTURE
        # ════════════════════════════════════════════════════════════════
        unet_path = _ensure_model_exists(unet_name, "diffusion_models", auto_download)
        if not unet_path:
            raise FileNotFoundError(
                f"❌ UNET not found: '{unet_name}'. Enable auto_download or install it manually."
            )

        divider = resolve_divider()

        resolved_type, detected_type, latent_fmt = resolve_architecture(
            unet_path, model_type, info_lines
        )

        # ════════════════════════════════════════════════════════════════
        # 2. OFFLOAD MODE
        # ════════════════════════════════════════════════════════════════
        clip_load_device = setup_offload_mode(offload_mode, info_lines)

        # ════════════════════════════════════════════════════════════════
        # 3. VRAM ESTIMATION
        # ════════════════════════════════════════════════════════════════
        has_loras = bool(lora_stack)
        est, avail, total = estimate_vram_for_load(
            resolved_type, weight_dtype, clip_dtype, has_loras, check_vram, divider, info_lines
        )

        # ════════════════════════════════════════════════════════════════
        # 4. LOAD UNET  (+ optional baked VAE extraction, mtime + size cache key)
        # ════════════════════════════════════════════════════════════════
        # No audio_vae_name slot on the base loader -- always passed "None".
        model, vae, _audio_vae, unet_time, unet_cache_hit, vae_time, vae_cache_hit = load_unet_and_baked_vae(
            unet_path, unet_name, weight_dtype, offload_mode, vae_name, "None", resolved_type,
            caching, divider, info_lines
        )

        # ════════════════════════════════════════════════════════════════
        # 5. LOAD CLIP  (named slots → ordered paths → mtime cache key)
        # ════════════════════════════════════════════════════════════════
        clip, clip_slot_used, clip_time, clip_cache_hit = load_clip_stack(
            resolved_type, unet_path, clip_l, clip_g, t5xxl, llm_encoder, text_projection,
            clip_dtype, offload_mode, clip_load_device, caching, divider, auto_download, info_lines
        )

        # ════════════════════════════════════════════════════════════════
        # 6. LOAD STANDALONE VAE  (mtime cache key) — skipped if baked
        # ════════════════════════════════════════════════════════════════
        extract_vae = (vae_name == "Baked VAE (from UNET)")
        if not extract_vae:
            vae, vae_time, vae_cache_hit = load_standalone_vae(
                vae_name, auto_download, caching, divider, info_lines, self._load_vae_sd_metadata
            )

        # ════════════════════════════════════════════════════════════════
        # 7. APPLY LoRA STACK
        #    Priority: upstream lora_stack input stack
        # ════════════════════════════════════════════════════════════════
        model, clip, applied_loras, out_lora_stack = apply_lora_stack(
            model, clip, lora_stack, lora_on_error, divider, info_lines
        )

        # ════════════════════════════════════════════════════════════════
        # 8. BUILD OUTPUTS
        # ════════════════════════════════════════════════════════════════
        total_ms = round((time.time() - load_start) * 1000)
        cache_info = f" (cache: U{_unet_cache.size} C{_clip_cache.size} V{_vae_cache.size})" if caching else ""
        summary = f"System load complete in {total_ms / 1000:.1f}s{cache_info}"
        logger.info(summary)
        info_lines.append(summary)

        # Draw a beautiful independent console HUD card that represents visual Apple-style dashboard loading
        print_premium_loader_hud(
            preset=preset,
            overrides=overrides if preset in CHECKPOINT_PRESETS else None,
            resolved_type=resolved_type,
            latent_fmt=latent_fmt,
            est_vram=est,
            avail_vram=avail if check_vram == "On" else 0.0,
            total_vram=total if check_vram == "On" else 0.0,
            unet_name=unet_name,
            unet_dtype=weight_dtype,
            unet_time=unet_time,
            unet_cached=unet_cache_hit,
            clip_slots=clip_slot_used,
            clip_dtype=clip_dtype,
            clip_time=clip_time,
            clip_cached=clip_cache_hit,
            vae_name=vae_name,
            vae_time=vae_time,
            vae_cached=vae_cache_hit,
            loras=applied_loras,
            total_time_ms=total_ms,
            caching=caching
        )

        load_info = "\n".join(info_lines)

        # model_meta — structured JSON for downstream QC / analytics nodes
        model_meta = {
            "arch":          resolved_type,
            "detected":      detected_type is not None,
            "unet_file":     unet_name,
            "weight_dtype":  weight_dtype,
            "clip_slots":    clip_slot_used,
            "clip_dtype":    clip_dtype,
            "offload_mode":  offload_mode,
            "latent_ch":     LATENT_CHANNELS.get(resolved_type, 4),
            "latent_format": latent_fmt,
            "vram_est_gb":   est,
            "loras":         applied_loras,
            "load_ms":       total_ms,
            "cached_unet":   unet_cache_hit,
        }

        return (
            model,
            clip,
            vae,
            out_lora_stack,
            json.dumps(model_meta, sort_keys=True),
        )


class RadianceVideoLoader(RadianceUnifiedLoader):
    """Video-oriented loader (LTX, Wan, HunyuanVideo, ...). Adds:
    - VAE loading also reads .safetensors metadata so ComfyUI picks the
      correct internal VAE architecture (e.g. LTX 2.3's video VAE).
    - "Baked VAE (from UNET)" option: extracts VAE directly from the
      checkpoint instead of a standalone file.
    - Optional Audio VAE (baked or standalone) for LTX 2.3 audio.
    - Optional latent upscale model (e.g. LTX 2.3 / HunyuanVideo SR)."""

    @staticmethod
    def _load_vae_sd_metadata(vae_path):
        # ALBABIT-FIX: read metadata so comfy.sd.VAE selects the correct
        # internal config (e.g. LTX 2.3 video VAE) instead of falling back
        # to the default config and raising a state_dict size mismatch.
        return comfy.utils.load_torch_file(vae_path, return_metadata=True)

    @classmethod
    def INPUT_TYPES(cls):
        types = super().INPUT_TYPES()

        # ALBABIT-FIX: update unet_name tooltip to document WAN 2.2 dual-expert auto-detect.
        types["required"]["unet_name"][1]["tooltip"] = (
            "Main diffusion model (UNET / DiT / Transformer).\n"
            "For WAN 2.2, select either the high_noise or low_noise file — "
            "the companion expert is detected automatically. "
            "The 'model' output always carries the high_noise expert "
            "and 'model_low_noise' always carries the low_noise expert, "
            "regardless of which file is selected."
        )

        # ALBABIT-FIX: video loader only lists video-model presets
        # (image presets are exclusive to RadianceUnifiedLoader).
        _, preset_kwargs = types["required"]["preset"]
        types["required"]["preset"] = (
            [k for k in CHECKPOINT_PRESETS if k == "Custom" or k in VIDEO_PRESET_NAMES],
            preset_kwargs,
        )

        # ALBABIT-FIX: video loader only lists video-model model_types
        # (image model_types are exclusive to RadianceUnifiedLoader).
        _, model_type_kwargs = types["required"]["model_type"]
        types["required"]["model_type"] = (
            [m for m in MODEL_TYPES if m == "Auto-Detect" or m in VIDEO_MODEL_TYPES],
            model_type_kwargs,
        )

        # ALBABIT-FIX: "Baked VAE (from UNET)" lets LTX 2.3 checkpoints that
        # embed their own VAE skip the standalone vae_name file entirely.
        vae_files = folder_paths.get_filename_list("vae")
        types["required"]["vae_name"] = (
            ["Baked VAE (from UNET)"] + vae_files,
            {"default": "Baked VAE (from UNET)",
             "tooltip": "VAE for encoding/decoding latents. "
                        "'Baked VAE (from UNET)' extracts it from the checkpoint."},
        )

        # ALBABIT-FIX: optional standalone/baked Audio VAE (LTX 2.3 audio).
        ckpt_files = folder_paths.get_filename_list("checkpoints")
        audio_vae_list = ["None", "Baked Audio VAE (from UNET)"] + sorted(set(ckpt_files + vae_files))

        # ALBABIT-FIX: optional latent upscale model (LTX 2.3 / HunyuanVideo SR).
        upscale_list = ["None"] + (folder_paths.get_filename_list("latent_upscale_models") or [])

        # ALBABIT-FIX: rebuild "optional" so audio_vae_name / upscale_model_name
        # appear right after vae_name (matching the legacy Radiance layout)
        # instead of at the bottom of the node.
        old_optional = types["optional"]
        types["optional"] = {
            "audio_vae_name": (
                audio_vae_list,
                {"default": "None",
                 "tooltip": "Audio VAE for LTX 2.3. Choose 'Baked' or a standalone safetensors file."},
            ),
            "upscale_model_name": (
                upscale_list,
                {"default": "None",
                 "tooltip": "Latent Upscale Model (e.g. for LTX 2.3 or HunyuanVideo)."},
            ),
            **old_optional,
        }

        return types

    # ALBABIT-FIX: MODEL_LOW_NOISE slot reserved for WAN 2.2 dual-expert (MoE) support.
    # Returns None until the loading strategy is confirmed with upstream dev.
    RETURN_TYPES  = ("MODEL", "MODEL", "CLIP", "VAE", "VAE", "LORA_STACK", "LATENT_UPSCALE_MODEL", "STRING")
    RETURN_NAMES  = ("model", "model_low_noise", "clip", "vae", "audio_vae", "lora_stack", "upscale_model", "model_meta")
    DESCRIPTION   = (
        "Video loader v3.3 — for LTX 2.3, Wan, HunyuanVideo, etc. "
        "Supports Baked/standalone VAE, optional Audio VAE, and optional "
        "latent upscale model, on top of the universal loader features."
    )

    def load_radiance_stack(
        self,
        preset,
        unet_name,
        weight_dtype,
        model_type,
        vae_name,
        audio_vae_name="None",
        upscale_model_name="None",
        clip_l="None",
        clip_g="None",
        t5xxl="None",
        llm_encoder="None",
        text_projection="None",
        clip_dtype="default",
        offload_mode="none",
        lora_stack=None,
        check_vram="On",
        use_cache="On",
        lora_on_error="raise",
        auto_download=True,
    ):
        def _none(val):
            return None if val in ("None", "", None) else val

        clip_l = _none(clip_l)
        clip_g = _none(clip_g)
        t5xxl = _none(t5xxl)
        llm_encoder = _none(llm_encoder)
        text_projection = _none(text_projection)

        load_start  = time.time()
        info_lines  = []
        caching     = use_cache == "On"

        # ════════════════════════════════════════════════════════════════
        # 0. APPLY PRESET
        # ════════════════════════════════════════════════════════════════
        model_type, weight_dtype, clip_dtype, overrides = apply_checkpoint_preset(
            preset, model_type, weight_dtype, clip_dtype, info_lines
        )

        # ════════════════════════════════════════════════════════════════
        # 1. RESOLVE ARCHITECTURE
        # ════════════════════════════════════════════════════════════════
        unet_path = _ensure_model_exists(unet_name, "diffusion_models", auto_download)
        if not unet_path:
            raise FileNotFoundError(
                f"❌ UNET not found: '{unet_name}'. Enable auto_download or install it manually."
            )

        divider = resolve_divider()

        resolved_type, detected_type, latent_fmt = resolve_architecture(
            unet_path, model_type, info_lines
        )

        # ════════════════════════════════════════════════════════════════
        # 2. OFFLOAD MODE
        # ════════════════════════════════════════════════════════════════
        clip_load_device = setup_offload_mode(offload_mode, info_lines)

        # ════════════════════════════════════════════════════════════════
        # 3. VRAM ESTIMATION
        # ════════════════════════════════════════════════════════════════
        has_loras = bool(lora_stack)
        # Detect companion early so VRAM estimate accounts for both UNETs.
        companion_name = _find_wan_moe_companion(unet_name)
        est, avail, total = estimate_vram_for_load(
            resolved_type, weight_dtype, clip_dtype, has_loras, check_vram, divider, info_lines,
            extra_unet_count=1 if companion_name else 0,
        )

        # ════════════════════════════════════════════════════════════════
        # 4. LOAD UNET  (+ optional baked VAE / Audio VAE extraction)
        # ════════════════════════════════════════════════════════════════
        model, vae, audio_vae, unet_time, unet_cache_hit, vae_time, vae_cache_hit = load_unet_and_baked_vae(
            unet_path, unet_name, weight_dtype, offload_mode, vae_name, audio_vae_name, resolved_type,
            caching, divider, info_lines
        )

        # ════════════════════════════════════════════════════════════════
        # 4b. LOAD WAN 2.2 MoE COMPANION UNET (auto-detect high/low_noise pair)
        # ════════════════════════════════════════════════════════════════
        model_low_noise = None
        # companion_name already computed above for VRAM estimate accuracy.
        if companion_name:
            companion_path = _ensure_model_exists(companion_name, "diffusion_models", auto_download)
            if companion_path:
                model_low_noise, _, _, _, _, _, _ = load_unet_and_baked_vae(
                    companion_path, companion_name, weight_dtype, offload_mode,
                    "None", "None", resolved_type, caching, divider, info_lines
                )
            else:
                logger.warning(f"WAN 2.2 companion UNET not found: '{companion_name}'")
                info_lines.append(f"COMPANION UNET: '{companion_name}' not found — MODEL_LOW_NOISE will be None")

        # ALBABIT-FIX: ensure model = high_noise expert, model_low_noise = low_noise expert,
        # regardless of which file the user picked in the unet_name widget.
        if model_low_noise is not None and "low_noise" in unet_name.lower():
            model, model_low_noise = model_low_noise, model

        # ════════════════════════════════════════════════════════════════
        # 5. LOAD CLIP  (named slots → ordered paths → mtime cache key)
        # ════════════════════════════════════════════════════════════════
        clip, clip_slot_used, clip_time, clip_cache_hit = load_clip_stack(
            resolved_type, unet_path, clip_l, clip_g, t5xxl, llm_encoder, text_projection,
            clip_dtype, offload_mode, clip_load_device, caching, divider, auto_download, info_lines
        )

        # ════════════════════════════════════════════════════════════════
        # 6. LOAD STANDALONE VAE  (mtime cache key) — skipped if baked
        # ════════════════════════════════════════════════════════════════
        extract_vae       = (vae_name == "Baked VAE (from UNET)")
        extract_audio_vae = (audio_vae_name == "Baked Audio VAE (from UNET)")

        if not extract_vae:
            vae, vae_time, vae_cache_hit = load_standalone_vae(
                vae_name, auto_download, caching, divider, info_lines, self._load_vae_sd_metadata
            )

        # ════════════════════════════════════════════════════════════════
        # 6b. LOAD STANDALONE AUDIO VAE — skipped if baked or "None"
        # ════════════════════════════════════════════════════════════════
        if not extract_audio_vae and audio_vae_name != "None":
            t0 = time.time()
            audio_vae_path = (folder_paths.get_full_path("checkpoints", audio_vae_name)
                               or folder_paths.get_full_path("vae", audio_vae_name))

            if audio_vae_path:
                audio_vae_fp  = _file_fingerprint(audio_vae_path)
                audio_vae_key = f"audio_vae:{audio_vae_path}:{audio_vae_fp}"

                if caching and _audio_vae_cache.has(audio_vae_key):
                    audio_vae = _audio_vae_cache.get(audio_vae_key)
                    logger.info(f"Audio VAE loaded from cache: {audio_vae_name}")
                    info_lines.append(f"AUDIO VAE: {audio_vae_name} (cached)")
                else:
                    try:
                        sd, metadata = comfy.utils.load_torch_file(audio_vae_path, return_metadata=True)
                        audio_vae = construct_audio_vae(sd, metadata, resolved_type)
                        av_time = time.time() - t0
                        logger.info(f"Audio VAE loaded {divider} {audio_vae_name} {divider} {av_time:.1f}s")
                        info_lines.append(f"AUDIO VAE: {audio_vae_name} ({av_time:.1f}s)")
                        if caching:
                            _audio_vae_cache.put(audio_vae_key, audio_vae)
                    except Exception as e:
                        logger.error(f"❌ Failed to load Audio VAE '{audio_vae_name}': {e}")
                        info_lines.append(f"AUDIO VAE: load failed ({e})")
            else:
                logger.warning(f"❌ Audio VAE not found: '{audio_vae_name}'")
                info_lines.append(f"AUDIO VAE: '{audio_vae_name}' not found")

        # ════════════════════════════════════════════════════════════════
        # 6c. LOAD LATENT UPSCALE MODEL — skipped if "None"
        # ════════════════════════════════════════════════════════════════
        upscale_model = None
        if upscale_model_name and upscale_model_name != "None":
            t0 = time.time()
            upscale_path = folder_paths.get_full_path("latent_upscale_models", upscale_model_name)
            if not upscale_path:
                logger.warning(f"❌ Latent Upscale Model not found: '{upscale_model_name}'")
                info_lines.append(f"UPSCALE MODEL: '{upscale_model_name}' not found")
            else:
                upscale_fp  = _file_fingerprint(upscale_path)
                upscale_key = f"upscale_model:{upscale_path}:{upscale_fp}"

                if caching and _upscale_model_cache.has(upscale_key):
                    upscale_model = _upscale_model_cache.get(upscale_key)
                    logger.info(f"Latent Upscale Model loaded from cache: {upscale_model_name}")
                    info_lines.append(f"UPSCALE MODEL: {upscale_model_name} (cached)")
                else:
                    try:
                        sd, metadata = comfy.utils.load_torch_file(upscale_path, safe_load=True, return_metadata=True)

                        # ALBABIT-FIX: native Hunyuan/LTX upscale model routing.
                        # Imported locally to avoid hard deps on older ComfyUI.
                        if "blocks.0.block.0.conv.weight" in sd:
                            from comfy.ldm.hunyuan_video.upsampler import HunyuanVideo15SRModel
                            config = {
                                "in_channels": sd["in_conv.conv.weight"].shape[1],
                                "out_channels": sd["out_conv.conv.weight"].shape[0],
                                "hidden_channels": sd["in_conv.conv.weight"].shape[0],
                                "num_blocks": len([k for k in sd.keys() if k.startswith("blocks.") and k.endswith(".block.0.conv.weight")]),
                                "global_residual": False,
                            }
                            upscale_model = HunyuanVideo15SRModel("720p", config)
                            upscale_model.load_sd(sd)
                        elif "up.0.block.0.conv1.conv.weight" in sd:
                            from comfy.ldm.hunyuan_video.upsampler import HunyuanVideo15SRModel
                            sd = {key.replace("nin_shortcut", "nin_shortcut.conv", 1): value for key, value in sd.items()}
                            config = {
                                "z_channels": sd["conv_in.conv.weight"].shape[1],
                                "out_channels": sd["conv_out.conv.weight"].shape[0],
                                "block_out_channels": tuple(
                                    sd[f"up.{i}.block.0.conv1.conv.weight"].shape[0]
                                    for i in range(len([k for k in sd.keys() if k.startswith("up.") and k.endswith(".block.0.conv1.conv.weight")]))
                                ),
                            }
                            upscale_model = HunyuanVideo15SRModel("1080p", config)
                            upscale_model.load_sd(sd)
                        elif "post_upsample_res_blocks.0.conv2.bias" in sd:
                            from comfy.ldm.lightricks.latent_upsampler import LatentUpsampler
                            # ALBABIT-FIX: scoped import -- a bare `import comfy.ops` here
                            # would shadow the top-level `comfy` name for this whole function
                            # (Python scoping), breaking the Audio VAE load right above.
                            from comfy.ops import disable_weight_init
                            from comfy.model_patcher import CoreModelPatcher
                            config = json.loads(metadata["config"])
                            # ALBABIT-FIX: from_config() requires `operations` (upstream API
                            # change) and the raw nn.Module needs wrapping in a ModelPatcher --
                            # LTXVLatentUpsampler reads `.load_device` off the returned object.
                            # Same pattern as comfy_extras/nodes_hunyuan.py for this class.
                            upscale_model = LatentUpsampler.from_config(config, operations=disable_weight_init).to(
                                dtype=comfy.model_management.vae_dtype(allowed_dtypes=[torch.bfloat16, torch.float32])
                            )
                            comfy.model_management.archive_model_dtypes(upscale_model)
                            upscale_model_patcher = CoreModelPatcher(
                                upscale_model,
                                load_device=comfy.model_management.get_torch_device(),
                                offload_device=comfy.model_management.unet_offload_device(),
                            )
                            upscale_model.load_state_dict(sd, assign=upscale_model_patcher.is_dynamic())
                            upscale_model = upscale_model_patcher
                        else:
                            logger.warning(f"❌ Unrecognized upscale model architecture for: '{upscale_model_name}'")
                            info_lines.append(f"UPSCALE MODEL: unrecognized architecture ({upscale_model_name})")

                        if upscale_model is not None:
                            up_time = time.time() - t0
                            logger.info(f"Latent Upscale Model loaded {divider} {upscale_model_name} {divider} {up_time:.1f}s")
                            info_lines.append(f"UPSCALE MODEL: {upscale_model_name} ({up_time:.1f}s)")
                            if caching:
                                _upscale_model_cache.put(upscale_key, upscale_model)
                    except Exception as e:
                        logger.error(f"❌ Failed to load Latent Upscale Model '{upscale_model_name}': {e}")
                        info_lines.append(f"UPSCALE MODEL: load failed ({e})")

        # ════════════════════════════════════════════════════════════════
        # 7. APPLY LoRA STACK
        # ════════════════════════════════════════════════════════════════
        model, clip, applied_loras, out_lora_stack = apply_lora_stack(
            model, clip, lora_stack, lora_on_error, divider, info_lines
        )

        # ════════════════════════════════════════════════════════════════
        # 8. BUILD OUTPUTS
        # ════════════════════════════════════════════════════════════════
        total_ms = round((time.time() - load_start) * 1000)
        cache_info = f" (cache: U{_unet_cache.size} C{_clip_cache.size} V{_vae_cache.size})" if caching else ""
        summary = f"System load complete in {total_ms / 1000:.1f}s{cache_info}"
        logger.info(summary)
        info_lines.append(summary)

        print_premium_loader_hud(
            preset=preset,
            overrides=overrides if preset in CHECKPOINT_PRESETS else None,
            resolved_type=resolved_type,
            latent_fmt=latent_fmt,
            est_vram=est,
            avail_vram=avail if check_vram == "On" else 0.0,
            total_vram=total if check_vram == "On" else 0.0,
            unet_name=unet_name,
            unet_dtype=weight_dtype,
            unet_time=unet_time,
            unet_cached=unet_cache_hit,
            clip_slots=clip_slot_used,
            clip_dtype=clip_dtype,
            clip_time=clip_time,
            clip_cached=clip_cache_hit,
            vae_name=vae_name,
            vae_time=vae_time,
            vae_cached=vae_cache_hit,
            loras=applied_loras,
            total_time_ms=total_ms,
            caching=caching
        )

        model_meta = {
            "arch":          resolved_type,
            "detected":      detected_type is not None,
            "unet_file":     unet_name,
            "weight_dtype":  weight_dtype,
            "clip_slots":    clip_slot_used,
            "clip_dtype":    clip_dtype,
            "offload_mode":  offload_mode,
            "latent_ch":     LATENT_CHANNELS.get(resolved_type, 4),
            "latent_format": latent_fmt,
            "vram_est_gb":   est,
            "loras":         applied_loras,
            "audio_vae":     audio_vae_name if audio_vae is not None else None,
            "upscale_model": upscale_model_name if upscale_model is not None else None,
            "wan_moe_companion": companion_name if model_low_noise is not None else None,
            "load_ms":       total_ms,
            "cached_unet":   unet_cache_hit,
        }

        return (
            model,
            model_low_noise,
            clip,
            vae,
            audio_vae,
            out_lora_stack,
            upscale_model,
            json.dumps(model_meta, sort_keys=True),
        )


# ═══════════════════════════════════════════════════════════════════════════════
class RadianceControlNetApply:
    """
    Advanced ControlNet application node for the Radiance suite.
    - Gracefully bypasses if CONTROL_NET is None (prevents AttributeError crashes).
    - Supports standard start/end percent and strength controls.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", {
                    "tooltip": "Conditioning to attach the ControlNet to (usually the positive). An existing ControlNet on it is chained, not replaced."}),
                "control_net": ("CONTROL_NET", {
                    "tooltip": "ControlNet model from a ControlNet loader. Strength 0 returns the conditioning unchanged."}),
                "image": ("IMAGE", {
                    "tooltip": "Control hint image (edges, depth, pose and so on) in display-referred 0-1 values."}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05,
                    "tooltip": "Global strength of the control effect."}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Percentage of the generation where control starts (0.0 = beginning)."}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Percentage of the generation where control ends (1.0 = end)."}),
                "control_type": (["auto"] + list(UNION_CONTROLNET_TYPES.keys()), {"default": "auto",
                    "tooltip": "For Union ControlNets (like Flux), select the specific control mode (Canny, Depth, etc.)."}),
            },
            "optional": {
                "vae": ("VAE", {
                    "tooltip": "The model's VAE. Needed by ControlNets that take a VAE-encoded hint "
                    "(SD3, Flux and other DiT ControlNets); leave unconnected for SD1.5 / SDXL ones."}),
            },
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)  # FIX 7: was missing — ComfyUI showed generic labels
    FUNCTION = "apply_controlnet"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Generate"
    DESCRIPTION = "Apply a ControlNet conditioning signal to the Radiance sampler."

    def apply_controlnet(self, conditioning, control_net, image, strength, start_percent, end_percent,
                         control_type="auto", vae=None):
        # 1. Graceful Bypass: If no control_net or zero strength, just return the input conditioning.
        if control_net is None:
            logger.info("◎ Radiance Control: No ControlNet connected. Bypassing.")
            return (conditioning, )
            
        if strength == 0:
            return (conditioning, )

        # 2. Set Union ControlNet Type (if applicable)
        control_net = control_net.copy()
        type_number = UNION_CONTROLNET_TYPES.get(control_type, -1)
        if type_number >= 0:
            control_net.set_extra_arg("control_type", [type_number])
        else:
            control_net.set_extra_arg("control_type", [])

        # 3. Advanced Application (Standard ComfyUI Logic with added safety)
        # ComfyUI's ControlNets take the hint channels-first, as its own
        # ControlNet Apply passes it (IMAGE is B,H,W,C). The hint used to go in
        # as B,H,W,C and a VAE was never passed, so ControlNets that encode the
        # hint with the VAE could not work.
        control_hint = image.movedim(-1, 1)

        def with_hint(cn):
            import inspect
            try:
                takes_vae = "vae" in inspect.signature(cn.set_cond_hint).parameters
            except (TypeError, ValueError):
                takes_vae = False
            kw = {"vae": vae} if takes_vae else {}   # a ComfyUI older than the vae argument
            return cn.set_cond_hint(control_hint, strength, (start_percent, end_percent), **kw)

        c = []
        try:
            for t in conditioning:
                n = [t[0], t[1].copy()]
                c_net = with_hint(control_net.copy())
                if 'control' in n[1]:
                    c_net.set_previous_controlnet(n[1]['control'])
                n[1]['control'] = c_net
                n[1]['control_apply_strength'] = strength
                c.append(n)
            return (c, )
        except Exception as e:
            logger.error(f"❌ Radiance Control: Failed to apply ControlNet: {e}")
            return (conditioning, )


# ═══════════════════════════════════════════════════════════════════════════════
#                         NODE REGISTRATION
# ═══════════════════════════════════════════════════════════════════════════════

# FIX 1: NODE_CLASS_MAPPINGS keys must be plain ASCII identifiers.
# The ◎ prefix belongs only in NODE_DISPLAY_NAME_MAPPINGS (the user-visible label).
# Having it in the type key breaks ComfyUI workflow JSON serialization and node lookup.
# Compatibility aliases.
#
# These two keys are older names for the node beside them. They have to stay
# registered or a workflow saved against the old key fails to load with a red
# "missing node" box, but they must not appear in the menu twice — two search
# results for one node is exactly the confusion the naming cleanup removed.
#
# A subclass rather than a second reference to the same class: `DEPRECATED` is
# a class attribute, and ComfyUI reads it off the class it was handed, so
# setting it on `RadianceUnifiedLoader` itself would hide the canonical node
# too. The subclass carries the flag and inherits every behaviour.
class RadianceImageLoader(RadianceUnifiedLoader):
    """Deprecated alias of RadianceUnifiedLoader. Loads, does not list."""
    DEPRECATED = True


class RadianceControlApply(RadianceControlNetApply):
    """Deprecated alias of RadianceControlNetApply. Loads, does not list."""
    DEPRECATED = True


NODE_CLASS_MAPPINGS = {
    "RadianceUnifiedLoader": RadianceUnifiedLoader,
    "RadianceImageLoader":   RadianceImageLoader,   # deprecated alias
    "RadianceVideoLoader":   RadianceVideoLoader,
    "RadianceLoraStack":     RadianceLoraStack,
    "RadianceControlNetApply": RadianceControlNetApply,
    "RadianceControlApply":    RadianceControlApply,  # deprecated alias
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceUnifiedLoader": "◎ Radiance Read Models",
    "RadianceImageLoader":   "◎ Radiance Image Loader",
    "RadianceVideoLoader":   "◎ Radiance Video Loader",
    "RadianceLoraStack":     "◎ Radiance LoRA Stack",
    "RadianceControlNetApply": "◎ Radiance ControlNet Apply",
    "RadianceControlApply":    "◎ Radiance Control",
}
