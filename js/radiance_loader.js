/**
 * radiance_loader.js
 * v3.3 — Streamlined Radiance Loader Widget sync, dynamic visibility & smart auto-fill
 */

import { app } from "../../scripts/app.js";

import {
    forceWidgetReinsert as _forceWidgetReinsert,
    setWidgetVisible as _setWidgetVisible,
    getWidget,
} from "./radiance_widget_utils.js";

// Widget helpers now live in radiance_widget_utils.js; this module's only
// local difference was the "combo" fallback type, which is passed through.
function setWidgetVisible(widget, visible, node) {
    _setWidgetVisible(widget, visible, node, { fallbackType: "combo" });
}

// Node definition identifiers
const LOADER_NODES = ["RadianceUnifiedLoader", "RadianceImageLoader", "RadianceVideoLoader"];

// Dynamic visibility rules for CLIP slots per preset.
// ALBABIT-FIX: keys sorted alphabetically ("Custom" pinned first) to match
// the preset dropdown order (config/model_map.py's CHECKPOINT_PRESETS).
const PRESET_SLOTS = {
    "Custom": ["clip_l", "clip_g", "t5xxl", "llm_encoder", "text_projection"],
    // ALBABIT-FIX: AuraFlow's real encoder is a T5 variant (comfy.text_encoders.
    // aura_t5), not clip_l -- matches CLIP_SLOT_ORDER["aura_flow"] (model/detect.py).
    "AuraFlow": ["t5xxl"],
    "Chroma": ["t5xxl"],
    "CogVideoX": ["t5xxl"],
    "Cosmos World": ["t5xxl"],
    "Flux.1": ["clip_l", "t5xxl"],
    "Flux.1 (Low VRAM)": ["clip_l", "t5xxl"],
    "Flux.2": ["llm_encoder"],
    "Flux.2 (Low VRAM)": ["llm_encoder"],
    "HunyuanVideo": ["clip_l", "llm_encoder"],
    // ALBABIT-FIX: text_projection dropped -- pre-2.3 LTX only ever loads 1
    // CLIP file (comfy/text_encoders/lt.py's ltxv_te, T5-only). Providing a
    // 2nd file (text_projection) would route comfy.sd's CLIPType.LTXV branch
    // into ltxav_te (Gemma-based, LTX 2.3 only) instead -- the wrong encoder
    // for these architectures, not just an unused widget.
    "LTX Video": ["llm_encoder"],
    "LTX Video (Low VRAM)": ["llm_encoder"],
    "LTX Video 2.3": ["llm_encoder", "text_projection"],
    "LTX Video 2.3 (Low VRAM)": ["llm_encoder", "text_projection"],
    // ALBABIT-FIX: unlike 2.3, the 2.5 text encoder file bakes the projection
    // layer in (confirmed via safetensors header: a text_embedding_projection
    // tensor group lives inside gemma4-12b-with-proj-*.safetensors) -- only
    // one CLIP slot needed, same as pre-2.3 "LTX Video".
    "LTX Video 2.5": ["llm_encoder"],
    "LTX Video 2.5 (Low VRAM)": ["llm_encoder"],
    "Lumina2": ["llm_encoder"],
    // ALBABIT-FIX: MiniMax H3's conditioning encoder is a single Qwen3-VL-32B
    // checkpoint. One llm_encoder slot, same shape as Flux.2/Z-Image.
    "MiniMax H3": ["llm_encoder"],
    "MiniMax H3 (Low VRAM)": ["llm_encoder"],
    "Mochi": ["t5xxl"],
    "PixArt Sigma": ["t5xxl"],
    "SD 1.5": ["clip_l"],
    "SD3.5": ["clip_l", "clip_g", "t5xxl"],
    "SDXL": ["clip_l", "clip_g"],
    "Wan 2.1": ["t5xxl"],
    "Wan 2.1 (Low VRAM)": ["t5xxl"],
    "Wan 2.2": ["t5xxl"],
    "Wan 2.2 (Low VRAM)": ["t5xxl"],
    "Wan 2.2 TI2V": ["t5xxl"],
    "Z-Image": ["llm_encoder"],
};

// Full hints configs for automatic local file selection matching.
// ALBABIT-FIX: keys sorted alphabetically to match PRESET_SLOTS/the preset
// dropdown order (config/model_map.py's CHECKPOINT_PRESETS).
const PRESET_CONFIGS = {
    "AuraFlow": {
        "unet_hints":    ["auraflow", "aura_flow", "aura-flow"],
        // ALBABIT-FIX: official AuraFlow example workflow (fal/AuraFlow-v0.2,
        // extracted from aura_flow_0.2_example.png) uses CheckpointLoaderSimple
        // -- no separate VAELoader node at all. No standalone AuraFlow VAE file
        // found; "Baked VAE (from UNET)" listed first, old guesses kept as fallback.
        "vae_hints":     ["Baked VAE (from UNET)", "aura_vae", "sd_vae"],
        // ALBABIT-FIX: AuraFlow's real encoder is a T5 variant (comfy.text_encoders.
        // aura_t5.AuraT5Model, TEModel.T5_XL) -- "clip_l" hints never matched any
        // real file. No standalone encoder file found either (fal/AuraFlow-v0.2's
        // HF repo only has a generic diffusers-format text_encoder/ folder, not a
        // distinct ComfyUI-ready filename) -- "Baked (from UNET)" on t5xxl instead.
        "clip_hints":    {
            "t5xxl": ["Baked (from UNET)"],
        },
    },
    "Chroma": {
        "unet_hints":    ["chroma-unlocked", "chroma_unlocked", "chroma"],
        "vae_hints":     ["ae.safetensors", "flux_ae", "ae_"],
        "clip_hints":    {
            "t5xxl": ["t5xxl_fp16", "t5xxl_fp8_e4m3fn", "t5xxl"],
        },
    },
    "CogVideoX": {
        "unet_hints":    ["cogvideox-5b", "cogvideox_5b", "CogVideoX", "cogvideox"],
        "vae_hints":     ["cogvideox_vae", "cogvideox-vae", "cogvideo_vae"],
        "clip_hints":    {
            "t5xxl": ["t5xxl_fp16", "t5xxl_fp8_e4m3fn", "t5xxl"],
        },
    },
    "Cosmos World": {
        "unet_hints":    ["cosmos-1_0-diffusion", "Cosmos-1_0", "cosmos_world", "cosmos"],
        "vae_hints":     ["cosmos_vae", "cosmos-tokenizer", "cosmos"],
        "clip_hints":    {
            // ALBABIT-FIX: Cosmos uses the "old" T5-XXL (T5 1.0) encoder,
            // distinct from the t5xxl_fp8/fp16 (T5 1.1) used by Flux/SD3/etc.
            // Prioritize oldt5_xxl_*, fall back to t5xxl_* if absent.
            "t5xxl": ["oldt5_xxl_fp8_e4m3fn", "oldt5_xxl_fp16", "oldt5_xxl", "t5xxl_fp8_e4m3fn", "t5xxl_fp16", "t5xxl"],
        },
    },
    // ALBABIT-FIX: Dev and Schnell merged -- architecturally identical (no
    // single_blocks-style split like Flux.2 Klein), and the Sampler's
    // model_meta mechanism already tells them apart by filename for
    // guidance/steps. unet_hints combines both lists, Dev first.
    "Flux.1": {
        "unet_hints":    [
            "flux1-dev-fp8", "flux1-dev", "flux1-krea-dev", "krea-dev", "flux-dev",
            "flux1-schnell-fp8", "flux1-schnell", "flux-schnell",
        ],
        "vae_hints":     ["ae.safetensors", "flux_ae", "ae_"],
        "clip_hints":    {
            "clip_l": ["clip_l.safetensors", "clip_l"],
            "t5xxl":  ["t5xxl_fp16", "t5xxl_fp8_e4m3fn", "t5xxl"],
        },
    },
    "Flux.1 (Low VRAM)": {
        "unet_hints":    ["flux1-dev-fp8", "flux1-dev", "flux1-krea-dev", "krea-dev", "flux-dev"],
        "vae_hints":     ["ae.safetensors", "flux_ae", "ae_"],
        "clip_hints":    {
            "clip_l": ["clip_l.safetensors", "clip_l"],
            "t5xxl":  ["t5xxl_fp16", "t5xxl_fp8_e4m3fn", "t5xxl"],
        },
        // ALBABIT-FIX: "Low VRAM" presets force offload_mode — expose the
        // widget so the user can still override it (e.g. on a higher-VRAM
        // GPU where cpu_offload is unnecessarily slow).
        "extra_widgets": ["offload_mode"],
        "offload_mode": "cpu_offload",
    },
    // ALBABIT-FIX: Dev and Klein merged into one preset, Auto-Detect tells
    // them apart on its own. Quality-first (full precision, then -fp8),
    // same philosophy as MiniMax H3's presets below. "Flux.2 (Low VRAM)"
    // inverts this and never lists a full-precision file, even as a
    // fallback, leaving the widget unmatched instead.
    "Flux.2": {
        "unet_hints": [
            "flux2-dev.safetensors",
            "flux-2-klein-9b.safetensors", "flux-2-klein-base-9b.safetensors",
            "flux-2-klein-4b.safetensors", "flux-2-klein-base-4b.safetensors",
            "flux2_dev_fp8mixed.safetensors",
            "flux-2-klein-9b-fp8.safetensors", "flux-2-klein-base-9b-fp8.safetensors",
            "flux-2-klein-4b-fp8.safetensors", "flux-2-klein-base-4b-fp8.safetensors",
            "klein-9b-kv", "klein-base", "klein-9b", "klein-4b",
            "flux2-dev", "flux2_dev", "flux.2-dev",
            "flux2-klein", "flux2_klein", "flux.2-klein", "klein",
        ],
        // full_encoder_small_decoder is a lighter/faster decoder (same
        // encoder) -- only used if the full-quality flux2-vae isn't present.
        "vae_hints":     ["flux2-vae", "flux2_vae", "flux2_ae", "full_encoder_small_decoder"],
        "clip_hints":    {
            // Dev's encoder (Mistral) -- also the fallback when unet_name
            // isn't a recognized Klein size (see clip_size_hints, which
            // takes priority whenever a Klein 4B/9B file is detected).
            "llm_encoder": [
                "mistral_3_small_flux2_bf16.safetensors", "mistral_3_small_flux2_fp8.safetensors",
                "mistral_3_small_flux2_fp4_mixed.safetensors",
                "mistral_3_small_flux2_bf16", "mistral_3_small_flux2_fp8", "mistral_3_small_flux2", "mistral_3", "mistral",
            ],
        },
        // ALBABIT-FIX: Klein's encoder (Qwen) must match its size (4B->qwen_3_4b,
        // 9B->qwen_3_8b*) -- resolved dynamically from the size token detected
        // in unet_name. Quality-first: bf16 before fp8mixed/fp4mixed/fp4_flux2.
        "clip_size_hints": {
            "9b": [
                "qwen_3_8b.safetensors", "qwen_3_8b_fp8mixed.safetensors", "qwen_3_8b_fp4mixed.safetensors",
                "qwen_3_8b", "qwen3_8b",
            ],
            "4b": [
                "qwen_3_4b.safetensors", "qwen_3_4b_fp4_flux2.safetensors",
                "qwen_3_4b", "qwen3_4b",
            ],
        },
    },
    "Flux.2 (Low VRAM)": {
        "unet_hints": [
            "flux2_dev_fp8mixed.safetensors",
            "flux-2-klein-9b-fp8.safetensors", "flux-2-klein-base-9b-fp8.safetensors",
            "flux-2-klein-4b-fp8.safetensors", "flux-2-klein-base-4b-fp8.safetensors",
        ],
        "vae_hints":     ["flux2-vae", "flux2_vae", "flux2_ae", "full_encoder_small_decoder"],
        "clip_hints":    {
            "llm_encoder": [
                "mistral_3_small_flux2_fp8.safetensors", "mistral_3_small_flux2_fp4_mixed.safetensors",
            ],
        },
        "clip_size_hints": {
            "9b": ["qwen_3_8b_fp8mixed.safetensors", "qwen_3_8b_fp4mixed.safetensors"],
            "4b": ["qwen_3_4b_fp4_flux2.safetensors"],
        },
        // ALBABIT-FIX: "Low VRAM" presets force offload_mode — expose the
        // widget so the user can still override it (e.g. on a higher-VRAM
        // GPU where cpu_offload is unnecessarily slow).
        "extra_widgets": ["offload_mode"],
        "offload_mode": "cpu_offload",
    },
    "HunyuanVideo": {
        "unet_hints":    ["hunyuan_video", "hunyuanvideo", "hyvideo"],
        "vae_hints":     ["hunyuan_video_vae", "hunyuan_vae", "hyvideo_vae"],
        "clip_hints":    {
            "llm_encoder": ["llava_llama3", "llm_llava", "hunyuan_llm"],
            "clip_l":      ["clip_l.safetensors", "clip_l"],
        },
    },
    // ALBABIT-FIX: 2B and 13B merged into one preset, identical vae_hints/
    // clip_hints, weight_dtype "default" (comfy.sd auto-picks per file size).
    // unet_hints/upscale_hints list 13B-specific patterns first so the old
    // bare "ltxv" 2B fallback can't out-match a 13B file via substring.
    "LTX Video": {
        "unet_hints":    ["ltx-video-13b", "ltxv-13b", "ltx_13b", "ltx-video-2b", "ltxv-2b", "ltx_video", "ltxv"],
        "vae_hints":     ["Baked VAE (from UNET)", "ltxvideo_vae", "ltx_vae", "ltxv_vae", "causal_vae"],
        "clip_hints":    {
            "llm_encoder": ["t5xxl_fp8_e4m3fn", "t5xxl_fp16", "t5xxl"],
        },
        "extra_widgets": ["upscale_model_name"],
        // ALBABIT-FIX: spatial (resolution) and temporal (motion) upscalers
        // are separate files, confirmed on Lightricks' HF repo. 0.9.8 first
        // as the newer generation.
        "upscale_hints": ["ltxv-spatial-upscaler-0.9.8", "ltxv-spatial-upscaler-0.9.7", "ltxv-13b", "ltx_13b", "ltxv", "ltx_video", "latent_upsampler", "upsampler"],
        // ALBABIT-FIX: 2B's standard workflow is a single Sampler -- no
        // latent-upscale stage -- so leave upscale_model_name on "None"
        // instead of pre-filling a file the typical pipeline doesn't use.
        // 13B keeps the real upscale_hints above (2-stage LowRes+HighRes
        // pipeline, confirmed via the reference 13B workflow).
        "upscale_size_hints": {
            "2b": [],
            "13b": ["ltxv-spatial-upscaler-0.9.8", "ltxv-spatial-upscaler-0.9.7", "ltxv-13b", "ltx_13b", "ltxv", "ltx_video", "latent_upsampler", "upsampler"],
        },
    },
    "LTX Video (Low VRAM)": {
        "unet_hints":    ["ltx-video-13b", "ltxv-13b", "ltx_13b", "ltx-video-2b", "ltxv-2b", "ltx_video", "ltxv"],
        "vae_hints":     ["Baked VAE (from UNET)", "ltxvideo_vae", "ltx_vae", "ltxv_vae", "causal_vae"],
        "clip_hints":    {
            "llm_encoder": ["t5xxl_fp8_e4m3fn", "t5xxl_fp16", "t5xxl"],
        },
        // ALBABIT-FIX: offload_mode exposed + defaulted, same convention as
        // Flux.1/Flux.2/LTX Video 2.3's own "(Low VRAM)" siblings.
        "extra_widgets": ["upscale_model_name", "offload_mode"],
        "offload_mode": "cpu_offload",
        // ALBABIT-FIX: same real spatial-upscaler hints as "LTX Video" above.
        "upscale_hints": ["ltxv-spatial-upscaler-0.9.8", "ltxv-spatial-upscaler-0.9.7", "ltxv-13b", "ltx_13b", "ltxv", "ltx_video", "latent_upsampler", "upsampler"],
        // ALBABIT-FIX: 2B's standard workflow is a single Sampler -- no
        // latent-upscale stage -- so leave upscale_model_name on "None"
        // instead of pre-filling a file the typical pipeline doesn't use.
        // 13B keeps the real upscale_hints above (2-stage LowRes+HighRes
        // pipeline, confirmed via the reference 13B workflow).
        "upscale_size_hints": {
            "2b": [],
            "13b": ["ltxv-spatial-upscaler-0.9.8", "ltxv-spatial-upscaler-0.9.7", "ltxv-13b", "ltx_13b", "ltxv", "ltx_video", "latent_upsampler", "upsampler"],
        },
    },
    "LTX Video 2.3": {
        // ALBABIT-FIX: distilled-1.1 added in second position — auto-matches if user has it
        "unet_hints":    ["ltx-2.3-22b-dev.safetensors", "ltx-2.3-22b-distilled-1.1.safetensors", "ltx-2.3-22b-distilled.safetensors", "ltx-2.3-22b-dev", "ltx-2.3", "ltx_2.3"],
        "vae_hints":     ["LTX23_video_vae_bf16.safetensors", "LTX23_video_vae", "ltx23_video", "ltx_23_video", "Baked VAE (from UNET)"],
        "audio_vae_hints": ["LTX23_audio_vae_bf16.safetensors", "LTX23_audio_vae", "ltx23_audio", "ltx_23_audio"],
        "clip_hints":    {
            "llm_encoder":     ["gemma_3_12B_it.safetensors", "gemma_3_12B_it_fp4", "gemma_3_12B_it", "gemma_3", "gemma"],
            // ALBABIT-FIX: fall back to "Baked (from UNET)" if the standalone
            // text_projection file isn't present (mirrors the Low VRAM preset).
            "text_projection": ["ltx-2.3_text_projection_bf16.safetensors", "ltx-2.3_text_projection", "text_projection", "Baked (from UNET)"],
        },
        "extra_widgets": ["upscale_model_name", "audio_vae_name"],
        // ALBABIT-FIX: x2-1.0 was removed from the Lightricks repo, replaced
        // by x2-1.1. Kept here only so a local file from before the removal
        // still auto-matches. Deliberately not in RADIANCE_MODEL_MAP, its
        // old download URL 404s now.
        "upscale_hints": ["ltx-2.3-spatial-upscaler-x2-1.1.safetensors", "ltx-2.3-spatial-upscaler-x2-1.0.safetensors", "ltx-2.3", "ltx_2.3", "latent_upsampler", "upsampler"],
    },
    "LTX Video 2.3 (Low VRAM)": {
        // ALBABIT-FIX: distilled-1.1 replaces dev-fp8 as primary Low VRAM model
        "unet_hints":    ["ltx-2.3-22b-distilled-1.1.safetensors", "ltx-2.3-22b-distilled.safetensors", "ltx-2.3-22b-distilled-1.1", "ltx-2.3-22b-distilled", "ltx-2.3-22b-dev-fp8.safetensors", "ltx-2.3-22b-dev-fp8", "ltx-2.3", "ltx_2.3"],
        "vae_hints":     ["Baked VAE (from UNET)", "LTX23_video_vae", "ltx23_video", "ltx_23_video"],
        // ALBABIT-FIX: without this, autoFillPresetFiles() falls back to
        // audio_vae_name = "None" (no hints), so extract_audio_vae is False
        // and the AUDIO_VAE output stays None, failing downstream with
        // "Audio VAE model is required" (nodes_lt_audio.py).
        "audio_vae_hints": ["Baked Audio VAE (from UNET)"],
        "clip_hints":    {
            "llm_encoder":     ["gemma_3_12B_it_fp4_mixed.safetensors", "gemma_3_12B_it_fp4", "gemma_3_12B_it", "gemma_3", "gemma"],
            "text_projection": ["Baked (from UNET)"],
        },
        // ALBABIT-FIX: "Low VRAM" presets force offload_mode — expose the
        // widget so the user can still override it (e.g. on a higher-VRAM
        // GPU where cpu_offload is unnecessarily slow).
        "extra_widgets": ["upscale_model_name", "offload_mode"],
        "offload_mode": "cpu_offload",
        "upscale_hints": ["ltx-2.3-spatial-upscaler-x2-1.1.safetensors", "ltx-2.3-spatial-upscaler-x2-1.0.safetensors", "ltx-2.3", "ltx_2.3", "latent_upsampler", "upsampler"],
    },
    "LTX Video 2.5": {
        // ALBABIT-FIX: Dev listed first (max quality, flexible CFG per the HF
        // repo's own README), Distilled as fallback -- same convention as LTX
        // 2.3's own preset. bf16 preferred over the comfy-int8-convrot
        // quantization within each tier.
        "unet_hints":    ["ltx-2.5-22b-dev-transformer-bf16.safetensors", "ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors", "ltx-2.5-22b-distilled-transformer-bf16.safetensors", "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors", "ltx-2.5-22b-dev", "ltx-2.5-22b-distilled", "ltx-2.5", "ltx_2.5"],
        // ALBABIT-FIX: the 2.5 transformer has no baked VAE (confirmed via
        // safetensors header), "Baked VAE" kept as a low-priority fallback
        // only. "video-vae-conv" deliberately NOT hinted -- different (16x/4x)
        // compression Resolution can't detect per-file, see project_radiance_ltx25.
        "vae_hints":     ["ltx-2.5-video-vae-bf16.safetensors", "ltx-2.5-video-vae", "ltx_2.5_video", "Baked VAE (from UNET)"],
        "audio_vae_hints": ["ltx-2.5-audio-vae-bf16.safetensors", "ltx-2.5-audio-vae", "ltx_2.5_audio"],
        "clip_hints":    {
            "llm_encoder": ["gemma4-12b-with-proj-ltx-2.5-bf16.safetensors", "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors", "gemma4-12b-with-proj-ltx-2.5", "gemma4", "gemma_4"],
        },
        "extra_widgets": ["upscale_model_name", "audio_vae_name"],
        // ALBABIT-FIX: spatial upscaler only -- matches 2.3's convention (the
        // 2x latent upscale feeding the HighRes stage). The new temporal
        // upscaler (ltx-2.5-latent-temporal-upscaler-x2) isn't wired into any
        // Radiance node yet -- not hinted here, see project_radiance_ltx25 memory.
        "upscale_hints": ["ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors", "ltx-2.5-spatial-upscaler", "ltx-2.5", "ltx_2.5", "latent_upsampler", "upsampler"],
    },
    "LTX Video 2.5 (Low VRAM)": {
        // ALBABIT-FIX: Distilled int8 first (lightest real option), same
        // "lightest file first" convention as LTX 2.3's Low VRAM sibling.
        "unet_hints":    ["ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors", "ltx-2.5-22b-distilled-transformer-bf16.safetensors", "ltx-2.5-22b-distilled", "ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors", "ltx-2.5", "ltx_2.5"],
        // ALBABIT-FIX: "video-vae-conv" was hinted FIRST here (the repo's own
        // "lighter" recommendation) -- removed, wrong 16x/4x compression
        // Resolution can't detect per-file. See project_radiance_ltx25 memory.
        "vae_hints":     ["ltx-2.5-video-vae-bf16.safetensors", "ltx-2.5-video-vae", "ltx_2.5_video", "Baked VAE (from UNET)"],
        "audio_vae_hints": ["ltx-2.5-audio-vae-bf16.safetensors", "ltx-2.5-audio-vae", "ltx_2.5_audio"],
        "clip_hints":    {
            "llm_encoder": ["gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors", "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors", "gemma4-12b-with-proj-ltx-2.5", "gemma4", "gemma_4"],
        },
        // ALBABIT-FIX: "Low VRAM" presets force offload_mode -- expose the
        // widget so the user can still override it, same convention as every
        // other "(Low VRAM)" sibling in this file.
        "extra_widgets": ["upscale_model_name", "audio_vae_name", "offload_mode"],
        "offload_mode": "cpu_offload",
        "upscale_hints": ["ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors", "ltx-2.5-spatial-upscaler", "ltx-2.5", "ltx_2.5", "latent_upsampler", "upsampler"],
    },
    "Lumina2": {
        "unet_hints":    ["lumina2", "lumina-2", "lumina_2"],
        // ALBABIT-FIX: official Lumina2 example workflow (lumina_2.safetensors,
        // extracted from lumina2_basic_example.png) uses CheckpointLoaderSimple
        // -- no separate VAELoader node at all. No standalone Lumina2 VAE file
        // found; "Baked VAE (from UNET)" listed first, old guesses kept as fallback.
        "vae_hints":     ["Baked VAE (from UNET)", "ae.safetensors", "flux_ae", "sd3_vae", "sd_vae", "lumina_vae"],
        "clip_hints":    {
            "llm_encoder": ["gemma_2_2b", "gemma2_2b", "gemma_2"],
        },
    },
    // ALBABIT-FIX: quality-first for "MiniMax H3", same philosophy as
    // Flux.2's Dev-first unet_hints above. bf16 exceeds a 32GB card's native
    // capacity but still runs via ComfyUI's automatic lowvram partial-load,
    // at a measured real-world cost of about +37% time on a 5090. "MiniMax
    // H3 (Low VRAM)" never lists bf16, even as a fallback.
    "MiniMax H3": {
        "unet_hints": [
            "minimax_h3_fl2va_bf16.safetensors", "minimax_h3_fl2va_bf16",
            "minimax_h3_fl2va_pruned_int8_convrot.safetensors", "minimax_h3_fl2va_pruned_int8_convrot",
            "minimax_h3_fl2va_pruned_fp8_scaled.safetensors", "minimax_h3_fl2va_pruned_fp8_scaled",
            "minimax_h3_fl2va_int8_convrot.safetensors", "minimax_h3_fl2va_int8_convrot",
            "minimax_h3_fl2va_pruned_bf16.safetensors", "minimax_h3_fl2va_pruned_bf16",
            "minimax_h3_fl2va",
        ],
        "vae_hints":     ["minimax_h3_video_vae_fp16.safetensors", "minimax_h3_video_vae"],
        "audio_vae_hints": ["minimax_h3_audio_vae_fp32.safetensors", "minimax_h3_audio_vae"],
        "clip_hints":    {
            // ALBABIT-FIX: bf16 first, matching the UNET's quality-first
            // priority above. Live A/B test (5090): swapping the text
            // encoder's precision cost near-zero extra time, since it only
            // runs once (prompt encoding) versus the UNET's cost every
            // sampling step.
            "llm_encoder": [
                "qwen3vl_32b_minimax_h3_bf16.safetensors", "qwen3vl_32b_minimax_h3_bf16",
                "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "qwen3vl_32b_minimax_h3_nvfp4_awq",
                "qwen3vl_32b_minimax_h3_int8_convrot.safetensors", "qwen3vl_32b_minimax_h3_int8_convrot",
                "qwen3vl_32b_minimax_h3", "qwen3vl",
            ],
        },
        "extra_widgets": ["audio_vae_name"],
    },
    "MiniMax H3 (Low VRAM)": {
        "unet_hints": [
            "minimax_h3_fl2va_pruned_int8_convrot.safetensors", "minimax_h3_fl2va_pruned_int8_convrot",
            "minimax_h3_fl2va_pruned_fp8_scaled.safetensors", "minimax_h3_fl2va_pruned_fp8_scaled",
            "minimax_h3_fl2va_pruned",
        ],
        "vae_hints":     ["minimax_h3_video_vae_fp16.safetensors", "minimax_h3_video_vae"],
        "audio_vae_hints": ["minimax_h3_audio_vae_fp32.safetensors", "minimax_h3_audio_vae"],
        "clip_hints":    {
            "llm_encoder": [
                "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "qwen3vl_32b_minimax_h3_nvfp4_awq",
                "qwen3vl_32b_minimax_h3_int8_convrot.safetensors", "qwen3vl_32b_minimax_h3_int8_convrot",
            ],
        },
        "extra_widgets": ["audio_vae_name", "offload_mode"],
        "offload_mode": "cpu_offload",
    },
    "Mochi": {
        "unet_hints":    ["mochi_preview", "mochi-1-preview", "genmo_mochi", "mochi"],
        "vae_hints":     ["mochi_vae", "mochi-vae"],
        "clip_hints":    {
            // ALBABIT-FIX: prioritize fp16 t5xxl for Mochi, fp8 as fallback.
            "t5xxl": ["t5xxl_fp16", "t5xxl_fp8_e4m3fn", "t5xxl"],
        },
    },
    "PixArt Sigma": {
        "unet_hints":    ["pixart_sigma", "pixart-sigma", "PixArt-Sigma"],
        "vae_hints":     ["pixart_sigma_sdxlvae", "sdxl_vae", "sd_vae", "pixart_vae", "vae-ft-mse"],
        "clip_hints":    {
            "t5xxl": ["t5xxl_fp16", "t5xxl_fp8_e4m3fn", "t5xxl"],
        },
    },
    "SD 1.5": {
        "unet_hints":    ["v1-5", "v1_5", "sd15", "sd-1-5", "sd_1.5"],
        "vae_hints":     ["vae-ft-mse", "sd15_vae", "kl-f8"],
        "clip_hints":    {
            "clip_l": ["clip_l.safetensors", "clip_l"],
        },
    },
    // ALBABIT-FIX: Large, Large Turbo, and Medium merged into one preset,
    // identical vae_hints/clip_hints, so a combined unet_hints list is
    // enough (Large first, flagship). Turbo is Large's distilled variant,
    // Sampler tells it apart by filename, unaffected by this merge.
    "SD3.5": {
        "unet_hints":    [
            "sd3.5_large", "sd3-5_large", "sd3.5_large_turbo", "sd3.5_turbo", "sd3-5_turbo",
            "sd3.5_medium", "sd3-5_medium",
        ],
        // ALBABIT-FIX: every official SD3.5 release ships VAE baked into the
        // checkpoint, no standalone file found anywhere. "Baked VAE (from
        // UNET)" listed first as the auto-fill default; old hints kept as
        // a fallback in case a standalone file turns up for some user.
        "vae_hints":     ["Baked VAE (from UNET)", "sd3_vae", "sd3.5_vae", "sd3"],
        "clip_hints":    {
            "clip_l": ["clip_l.safetensors", "clip_l"],
            "clip_g": ["clip_g.safetensors", "clip_g"],
            "t5xxl":  ["t5xxl_fp16", "t5xxl_fp8_e4m3fn", "t5xxl"],
        },
    },
    // ALBABIT-FIX: Base and Turbo merged -- same reasoning as SD3.5 above.
    "SDXL": {
        "unet_hints":    ["sd_xl_base", "sdxl_base", "sdxl-base", "sdxl_turbo", "sdxl-turbo", "turbo"],
        "vae_hints":     ["sdxl_vae", "vae-ft-mse", "xl_vae"],
        "clip_hints":    {
            "clip_l": ["clip_l.safetensors", "clip_l"],
            "clip_g": ["clip_g.safetensors", "clip_g"],
        },
    },
    // ALBABIT-FIX: 1.3B/14B size resolved via unet_size_priority
    // (_resolveUnetMatch), not bare "14b"/"1.3b" hints (bug caught by
    // Albabit: those also matched Wan 2.2's own 14B-class files, no
    // architecture awareness in findMatchingFile). VAE/T5 are shared
    // across both sizes on HF, no size handling needed there.
    "Wan 2.1": {
        "unet_hints":    ["wan2.1", "wan_2.1", "wan-2.1", "Wan2.1"],
        "unet_size_priority": ["14b", "1.3b"],
        "vae_hints":     ["wan_2.1_vae", "wan2.1_vae", "wan_vae", "wan2_vae", "open_wan"],
        "clip_hints":    {
            "t5xxl": ["umt5_xxl", "umt5-xxl", "umt5xxl", "t5xxl"],
        },
    },
    "Wan 2.1 (Low VRAM)": {
        "unet_hints":    ["wan2.1", "wan_2.1", "wan-2.1", "Wan2.1"],
        "unet_size_priority": ["14b", "1.3b"],
        "vae_hints":     ["wan_2.1_vae", "wan2.1_vae", "wan_vae", "wan2_vae", "open_wan"],
        "clip_hints":    {
            "t5xxl": ["umt5_xxl", "umt5-xxl", "umt5xxl", "t5xxl"],
        },
        // ALBABIT-FIX: offload_mode exposed + defaulted, same convention as
        // Flux.1/Flux.2/LTX Video's own "(Low VRAM)" siblings.
        "extra_widgets": ["offload_mode"],
        "offload_mode": "cpu_offload",
    },
    "Wan 2.2": {
        "unet_hints":    [
            "wan2.2_t2v_high_noise_14B_fp8_scaled",
            "wan2.2_i2v_high_noise_14B_fp8_scaled",
            "wan2.2_t2v_high_noise",
            "wan2.2_i2v_high_noise",
            "wan2.2_high_noise",
            "wan2.2",
            "wan_2.2",
            "wan-2.2",
            "Wan2.2",
        ],
        "vae_hints":     ["wan_2.1_vae", "wan2.1_vae", "wan_vae", "wan2_vae", "open_wan"],
        "clip_hints":    {
            "t5xxl": ["umt5_xxl", "umt5-xxl", "umt5xxl", "t5xxl"],
        },
        // ALBABIT-FIX: either the high_noise or low_noise UNET works --
        // _find_wan_moe_companion() (nodes_loader.py) auto-loads the other
        // one server-side regardless of which one is picked here.
        "companion_linked": true,
    },
    "Wan 2.2 (Low VRAM)": {
        "unet_hints":    [
            "wan2.2_t2v_high_noise_14B_fp8_scaled",
            "wan2.2_i2v_high_noise_14B_fp8_scaled",
            "wan2.2_t2v_high_noise",
            "wan2.2_i2v_high_noise",
            "wan2.2_high_noise",
            "wan2.2",
            "wan_2.2",
            "wan-2.2",
            "Wan2.2",
        ],
        "vae_hints":     ["wan_2.1_vae", "wan2.1_vae", "wan_vae", "wan2_vae", "open_wan"],
        "clip_hints":    {
            "t5xxl": ["umt5_xxl", "umt5-xxl", "umt5xxl", "t5xxl"],
        },
        "companion_linked": true,
        // ALBABIT-FIX: offload_mode exposed + defaulted, same convention as
        // every other "(Low VRAM)" sibling -- both high_noise and low_noise
        // UNETs (~14B each) get offloaded together.
        "extra_widgets": ["offload_mode"],
        "offload_mode": "cpu_offload",
    },
    "Wan 2.2 TI2V": {
        "unet_hints":    ["wan2.2_ti2v_5B_fp16", "wan2.2_ti2v_5B", "wan2.2_ti2v"],
        "vae_hints":     ["wan2.2_vae", "wan_2.2_vae"],
        "clip_hints":    {
            "t5xxl": ["umt5_xxl", "umt5-xxl", "umt5xxl", "t5xxl"],
        },
    },
    "Z-Image": {
        "unet_hints":    ["z_image", "z-image", "zimage"],
        "vae_hints":     ["flux_vae", "ae.safetensors", "flux_ae", "sd3_vae", "sd_vae"],
        "clip_hints":    {
            "llm_encoder": ["qwen_3_4b", "qwen3_4b", "qwen_3"],
        },
    },
};

const ALL_CLIP_WIDGETS = ["clip_l", "clip_g", "t5xxl", "llm_encoder", "text_projection"];

/**
 * Fuzzy search through options values list for any item containing the hints
 */
function findMatchingFile(hints, values) {
    if (!hints || !values) return null;
    for (const hint of hints) {
        const h = hint.toLowerCase();
        for (const val of values) {
            if (val.toLowerCase().includes(h)) {
                return val;
            }
        }
    }
    return null;
}

/**
 * Extract a size token like "4b"/"9b" from a filename (e.g. Flux.2 Klein
 * variants), used to pick a differently-sized paired text encoder.
 */
function _detectSizeToken(filename) {
    const m = /(?:^|[-_])(\d+b)(?:[-_.]|$)/i.exec((filename || "").toLowerCase());
    return m ? m[1] : null;
}

/**
 * Like _detectSizeToken, but only returns a token the preset actually knows
 * about (config.clip_size_hints) -- an unrelated UNET (e.g. Cosmos "...7B...")
 * can contain a size-shaped substring without being a real Klein size.
 */
function _knownSizeToken(config, filename) {
    const token = _detectSizeToken(filename);
    return (token && config?.clip_size_hints?.[token]) ? token : null;
}

/**
 * Resolve the CLIP hint list for a given slot, applying a preset's
 * clip_size_hints (size-aware override) when present, falling back to its
 * static clip_hints otherwise.
 */
function _resolveClipHints(node, config, wName) {
    if (wName === "llm_encoder" && config?.clip_size_hints) {
        const sizeToken = _knownSizeToken(config, getWidget(node, "unet_name")?.value || "");
        if (sizeToken) {
            return config.clip_size_hints[sizeToken];
        }
    }
    return config?.clip_hints?.[wName];
}

/**
 * Resolve which file (if any) currently matches a CLIP slot's hints.
 */
function _resolveClipMatch(node, config, wName) {
    const w = getWidget(node, wName);
    const hints = _resolveClipHints(node, config, wName);
    return (w && hints && w.options?.values) ? findMatchingFile(hints, w.options.values) : null;
}

/**
 * Like _resolveClipHints, but for upscale_model_name (e.g. LTX Video 2B/13B):
 * a preset's upscale_size_hints lets a known size (e.g. "2b") resolve to an
 * empty list -- the typical 2B workflow is a single Sampler with no latent
 * upscale stage, unlike 13B's 2-stage LowRes+upscale+HighRes pipeline -- so
 * the widget correctly defaults to "None" instead of a real (but unused)
 * upscaler file. Uses hasOwnProperty (not truthiness) so an empty-array entry
 * still counts as "known" and isn't skipped in favor of the static fallback.
 */
function _resolveUpscaleHints(node, config) {
    if (config?.upscale_size_hints) {
        const sizeToken = _detectSizeToken(getWidget(node, "unet_name")?.value || "");
        if (sizeToken && Object.prototype.hasOwnProperty.call(config.upscale_size_hints, sizeToken)) {
            return config.upscale_size_hints[sizeToken];
        }
    }
    return config?.upscale_hints;
}

/**
 * Resolve unet_name's auto-fill pick. Without unet_size_priority, behaves
 * exactly like a plain findMatchingFile() call (existing behavior for every
 * other preset, unchanged). With it (e.g. Wan 2.1's 1.3B/14B): a bare size
 * token must NEVER be tested against the full file list on its own -- it
 * would just as happily match an unrelated architecture's same-size file
 * (Wan 2.2's own 14B-class high_noise/low_noise checkpoints, under the
 * "Wan 2.1" preset -- the exact bug Albabit caught live). So size tokens are
 * only considered among files that already matched one of unet_hints
 * (architecture identity first, size preference second, never combined into
 * one flat scan).
 */
function _resolveUnetMatch(config, values) {
    if (!config?.unet_hints || !values) return null;
    if (!config.unet_size_priority) return findMatchingFile(config.unet_hints, values);
    const candidates = values.filter(v => config.unet_hints.some(h => v.toLowerCase().includes(h.toLowerCase())));
    for (const token of config.unet_size_priority) {
        const match = candidates.find(v => v.toLowerCase().includes(token.toLowerCase()));
        if (match) return match;
    }
    return candidates[0] || null;
}

/**
 * Performs client-side smart auto-fill matching of files based on selected preset
 */
function autoFillPresetFiles(node, cleanPreset) {
    const config = PRESET_CONFIGS[cleanPreset];
    if (!config) return;

    // 1. Match UNET
    const unetW = getWidget(node, "unet_name");
    if (unetW && unetW.options?.values && config.unet_hints) {
        const matched = _resolveUnetMatch(config, unetW.options.values);
        if (matched) unetW.value = matched;
    }

    // 2. Match VAE
    const vaeW = getWidget(node, "vae_name");
    if (vaeW && vaeW.options?.values && config.vae_hints) {
        const matched = findMatchingFile(config.vae_hints, vaeW.options.values);
        if (matched) vaeW.value = matched;
    }

    // 3. Match CLIP Slots
    for (const wName of ALL_CLIP_WIDGETS) {
        const clipW = getWidget(node, wName);
        if (!clipW) continue;

        const hints = _resolveClipHints(node, config, wName);
        if (hints && clipW.options?.values) {
            const matched = findMatchingFile(hints, clipW.options.values);
            clipW.value = matched || "None";
        } else {
            clipW.value = "None";
        }
    }

    // 4. Match Audio VAE (LTX 2.3, Radiance Video Loader only)
    const audioVaeW = getWidget(node, "audio_vae_name");
    if (audioVaeW && audioVaeW.options?.values) {
        if (config.audio_vae_hints) {
            const matched = findMatchingFile(config.audio_vae_hints, audioVaeW.options.values);
            audioVaeW.value = matched || "None";
        } else {
            audioVaeW.value = "None";
        }
    }

    // 5. Match Latent Upscale Model (Radiance Video Loader only)
    const upscaleW = getWidget(node, "upscale_model_name");
    if (upscaleW && upscaleW.options?.values) {
        const upscaleHints = _resolveUpscaleHints(node, config);
        if (upscaleHints) {
            const matched = findMatchingFile(upscaleHints, upscaleW.options.values);
            upscaleW.value = matched || "None";
        } else {
            upscaleW.value = "None";
        }
    }

    // 6. Default offload_mode (visible widget for "Low VRAM" presets — see
    // extra_widgets — but the user can change it afterward; not preset-locked)
    const offloadW = getWidget(node, "offload_mode");
    if (offloadW) {
        offloadW.value = config.offload_mode || "none";
    }
}

// ALBABIT-FIX: unet_name/vae_name/CLIP slots/audio_vae_name/upscale_model_name
// stay visible and editable while a preset is active (unlike model_type/
// weight_dtype/clip_dtype, hidden by design — loader_utils.py). Flag any that
// no longer match what autoFillPresetFiles() would pick right now, with a "✎"
// label marker (same pattern as radiance_sampler.js/radiance_prompt.js).
const PRESET_MARKER = " ✎";
// ALBABIT-FIX: "🧲" marks unet_name only when its model_meta output actually
// reaches a live Sampler (_isModelMetaConnected below), not just when the
// file is recognized. llm_encoder's own "🧲" under clip_size_hints is a
// different, Loader-internal fact with no Sampler relation, left ungated.
const LINKED_MARKER = " 🧲";
// ALBABIT-FIX: for companion_linked presets (Wan 2.2), "⛓" marks unet_name
// instead of "✎" to signal its companion is auto-loaded server-side, not a
// manual mistake. Purely Loader-internal, unlike "🧲" above, not gated on
// model_meta being connected.
const COMPANION_MARKER = " ⛓";

/**
 * Mirrors radiance_sampler.js's _findModelMetaSourceNode() in reverse: does
 * this Loader's own model_meta OUTPUT actually reach a live downstream node
 * right now? An output can fan out to several links (one Loader feeding
 * multiple Samplers) -- true as soon as at least one target is real and not
 * muted/bypassed (mode 2/4, same exclusion the Sampler-side lookup applies
 * to its origin node).
 */
function _isModelMetaConnected(node) {
    const output = node.outputs?.find(o => o.name === "model_meta");
    if (!output?.links || output.links.length === 0) return false;
    return output.links.some(linkId => {
        const link = app.graph.links[linkId];
        if (!link) return false;
        const targetNode = app.graph.getNodeById(link.target_id);
        return !!(targetNode && targetNode.mode !== 2 && targetNode.mode !== 4);
    });
}

// unet_name/vae_name are left untouched by autoFillPresetFiles() on no
// match; CLIP slots/audio_vae/upscale fall back to "None" instead.
const NO_NONE_FALLBACK_FIELDS = new Set(["unet_name", "vae_name"]);

function _markFileWidget(widget, markerText) {
    if (!widget) return false;
    if (widget._radOrigLabel === undefined && !markerText) return false;
    if (widget._radOrigLabel === undefined) widget._radOrigLabel = widget.label ?? widget.name;
    const wanted = markerText ? widget._radOrigLabel + markerText : widget._radOrigLabel;
    if (widget.label === wanted) return false;
    widget.label = wanted;
    return true;
}

function updatePresetDivergenceMarkers(node) {
    if (!node.widgets) return;
    const presetW = getWidget(node, "preset");
    const presetVal = presetW ? presetW.value : "Custom";
    const cleanPreset = presetVal ? presetVal.replace("→ ", "").replace("▶ ", "").replace("◈ ", "").trim() : "Custom";
    const config = cleanPreset === "Custom" ? null : PRESET_CONFIGS[cleanPreset];
    const activeSlots = config ? (PRESET_SLOTS[cleanPreset] || ALL_CLIP_WIDGETS) : [];
    const unetVal = getWidget(node, "unet_name")?.value || "";
    const unetRecognized = !!findMatchingFile(config?.unet_hints, [unetVal]);
    // ALBABIT-FIX: unetVariantLinked (any recognized file -- unet_name always
    // feeds the Sampler's model_meta magnets) vs clipLinked (llm_encoder also
    // depends on it, Klein-only) -- previously conflated under one check.
    const unetVariantLinked = !!(config && unetRecognized);
    const clipLinked = !!(config?.clip_size_hints && unetRecognized);
    const companionLinked = !!(config?.companion_linked && unetRecognized);
    // ALBABIT-FIX: gates "🧲" specifically -- "⛓" and llm_encoder's own "🧲"
    // (clipLinked) are Loader-internal facts, unaffected by whether anything
    // downstream is actually listening.
    const modelMetaConnected = _isModelMetaConnected(node);

    let changed = false;

    // resolveMatch overrides the plain findMatchingFile(hints, ...) lookup --
    // needed for unet_name, whose match must go through _resolveUnetMatch()
    // (unet_size_priority-aware) instead, same resolver autoFillPresetFiles()
    // itself uses, so the "✎" divergence check never disagrees with what
    // auto-fill would actually pick.
    const check = (widgetName, hints, resolveMatch) => {
        const w = getWidget(node, widgetName);
        if (!w) return;
        let markerText = null;
        if (config) {
            if (!hints || hints.length === 0) {
                if (String(w.value) !== "None") markerText = PRESET_MARKER;
            } else {
                const matched = resolveMatch ? resolveMatch() : findMatchingFile(hints, w.options?.values);
                if (matched !== null) {
                    if (String(w.value) !== String(matched)) markerText = PRESET_MARKER;
                } else if (!NO_NONE_FALLBACK_FIELDS.has(widgetName)) {
                    if (String(w.value) !== "None") markerText = PRESET_MARKER;
                }
            }
        }
        if (_markFileWidget(w, markerText)) changed = true;
    };

    // unet_name shows a link marker instead of the divergence marker while
    // companionLinked/unetVariantLinked -- the point is to signal a
    // relationship (auto-loaded companion, recognized preset variant), not
    // flag a mistake. companionLinked implies unetVariantLinked too (both
    // require unetRecognized). "⛓" always shows when companionLinked (a
    // Loader-internal fact); "🧲" -- for both the plain unetVariantLinked
    // case and as an addition alongside "⛓" -- only shows on top of that
    // when modelMetaConnected is also true. Without a live downstream
    // Sampler, a plain unetVariantLinked (non-companion) file falls back to
    // the normal "✎" divergence check instead, same treatment as any other
    // non-magnet widget.
    if (companionLinked) {
        const marker = COMPANION_MARKER + (modelMetaConnected ? LINKED_MARKER : "");
        if (_markFileWidget(getWidget(node, "unet_name"), marker)) changed = true;
    } else if (unetVariantLinked && modelMetaConnected) {
        if (_markFileWidget(getWidget(node, "unet_name"), LINKED_MARKER)) changed = true;
    } else {
        const unetW = getWidget(node, "unet_name");
        check("unet_name", config?.unet_hints, () => _resolveUnetMatch(config, unetW?.options?.values));
    }
    check("vae_name", config?.vae_hints);
    for (const wName of ALL_CLIP_WIDGETS) {
        if (config && !activeSlots.includes(wName)) continue; // hidden slot
        if (wName === "llm_encoder" && clipLinked) {
            const w = getWidget(node, "llm_encoder");
            const matched = _resolveClipMatch(node, config, "llm_encoder");
            const markerText = matched === null ? null
                : (String(w.value) === String(matched) ? LINKED_MARKER : PRESET_MARKER);
            if (_markFileWidget(w, markerText)) changed = true;
            continue;
        }
        check(wName, config ? _resolveClipHints(node, config, wName) : null);
    }
    check("audio_vae_name", config?.audio_vae_hints);
    check("upscale_model_name", config ? _resolveUpscaleHints(node, config) : null);

    if (changed) node.setDirtyCanvas(true, true);
}

/**
 * ALBABIT-FIX: Vue 3's virtual-DOM differ reuses the existing widget component
 * instance when the same object reference stays in node.widgets — a plain
 * `splice(0, 0)` no-op notifies Vue "something changed" but Vue doesn't
 * re-read `type`/`options.hidden` on that reused instance, so a widget
 * restored from `type === "hidden"` can stay invisible/zero-height (e.g. when
 * switching back to "Custom"). Removing and re-inserting the widget at the
 * same index forces Vue to destroy and remount its component.
 */
/**
 * Collapsible widget visibility helper
 */
/**
 * Recalculate node dimensions and refresh the canvas layout cleanly
 */
function refreshNodeSize(node) {
    if (!node.computeSize) return;

    const sz = node.computeSize();
    // ALBABIT-FIX: directly mutating node.size[i] updates LiteGraph but Vue's
    // node component never observes it, keeping the old height forever.
    // node.setSize(...) is the API Vue's resize handling actually reacts to.
    node.setSize([Math.max(node.size[0], sz[0]), sz[1]]);
    app.graph.setDirtyCanvas(true, true);
}

function updateLoaderUI(node, forceAutoFill = false) {
    if (!node.widgets) return;

    const presetW = getWidget(node, "preset");
    if (!presetW) return;

    const presetVal = presetW.value;
    // Strip cosmetic menu symbol prefixes (e.g. arrow icons)
    const cleanPreset = presetVal ? presetVal.replace("→ ", "").replace("▶ ", "").replace("◈ ", "").trim() : "Custom";

    const isCustom = cleanPreset === "Custom";

    if (isCustom) {
        // SHOW ALL WIDGETS
        node.widgets.forEach(w => {
            setWidgetVisible(w, true, node);
        });
        refreshNodeSize(node);
        updatePresetDivergenceMarkers(node);
        return;
    }

    // --- Specific Model Preset mode ---
    // Hide: model_type, weight_dtype, clip_dtype, offload_mode (preset manages them)
    // Hide general utilities to keep the UI clean: check_vram, use_cache, lora_on_error, auto_download
    // Show only: preset, unet_name, vae_name (unless baked, see below), and active CLIP slots!
    const activeSlots = PRESET_SLOTS[cleanPreset] || ALL_CLIP_WIDGETS;
    const extraWidgets = (PRESET_CONFIGS[cleanPreset] && PRESET_CONFIGS[cleanPreset].extra_widgets) || [];

    if (forceAutoFill) {
        autoFillPresetFiles(node, cleanPreset);
    }

    node.widgets.forEach(w => {
        if (w.name === "preset" || w.name === "unet_name") {
            setWidgetVisible(w, true, node);
        } else if (w.name === "vae_name") {
            // ALBABIT-FIX: hide vae_name once resolved to "Baked VAE (from
            // UNET)", nothing left to choose. Stays visible for a real
            // standalone file instead. "Custom" mode always shows it.
            setWidgetVisible(w, w.value !== "Baked VAE (from UNET)", node);
        } else if (ALL_CLIP_WIDGETS.includes(w.name)) {
            const shouldShow = activeSlots.includes(w.name);
            setWidgetVisible(w, shouldShow, node);
        } else if (extraWidgets.includes(w.name)) {
            setWidgetVisible(w, true, node);
        } else {
            setWidgetVisible(w, false, node);
        }
    });

    refreshNodeSize(node);
    updatePresetDivergenceMarkers(node);
}

// ALBABIT-FIX: app.registerExtension's nodeCreated/loadedGraphNode wraps
// callbacks AFTER Vue (Nodes 2.0) has already mounted the widgets, so the
// wrapped callback never fires on user interaction. beforeRegisterNodeDef +
// prototype.onNodeCreated/onConfigure (same as radiance_sampler.js) hooks
// before widget construction instead, so Vue captures the wrapped version.
app.registerExtension({
    name: "Radiance.UnifiedLoaderSync",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (!LOADER_NODES.includes(nodeData.name)) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            const node = this;
            const presetW = getWidget(node, "preset");
            const modelTypeW = getWidget(node, "model_type");

            if (presetW) {
                const origPresetCallback = presetW.callback;
                presetW.callback = function(value) {
                    if (origPresetCallback) origPresetCallback.call(this, value);
                    // Trigger updates and execute smart file matching
                    setTimeout(() => updateLoaderUI(node, true), 10);
                };
            }

            if (modelTypeW) {
                const origModelCallback = modelTypeW.callback;
                modelTypeW.callback = function(value) {
                    if (origModelCallback) origModelCallback.call(this, value);
                    setTimeout(() => updateLoaderUI(node, false), 10);
                };
            }

            // ALBABIT-FIX: for clip_size_hints presets (Flux.2 Klein), keep
            // llm_encoder in sync when the user manually changes unet_name.
            // No-op for every other preset.
            const unetW = getWidget(node, "unet_name");
            if (unetW) {
                const origUnetCallback = unetW.callback;
                unetW.callback = function(value) {
                    if (origUnetCallback) origUnetCallback.call(this, value);
                    setTimeout(() => {
                        const presetVal = getWidget(node, "preset")?.value;
                        const cleanPreset = presetVal ? presetVal.replace("→ ", "").replace("▶ ", "").replace("◈ ", "").trim() : "Custom";
                        const cfg = PRESET_CONFIGS[cleanPreset];
                        if (cfg?.clip_size_hints) {
                            const clipW = getWidget(node, "llm_encoder");
                            const matched = _resolveClipMatch(node, cfg, "llm_encoder");
                            if (clipW && matched) {
                                clipW.value = matched;
                                node.setDirtyCanvas(true, true);
                            }
                        }
                        updatePresetDivergenceMarkers(node);
                    }, 10);
                };
            }

            // Apply initial layout folding on creation, unless onConfigure
            // (loaded workflow) is about to do it with the restored values.
            setTimeout(() => {
                if (node._configuredByLoad) return;
                updateLoaderUI(node, false);
            }, 50);

            // File widgets aren't individually wrapped, so poll for manual
            // edits (same pattern as radiance_sampler.js/radiance_prompt.js).
            // Calls only the lightweight marker check, never updateLoaderUI()
            // itself, which remounts every widget on each call.
            node._presetMarkerInterval = setInterval(() => updatePresetDivergenceMarkers(node), 250);
            const origOnRemoved = node.onRemoved;
            node.onRemoved = function () {
                if (node._presetMarkerInterval) {
                    clearInterval(node._presetMarkerInterval);
                    node._presetMarkerInterval = null;
                }
                if (origOnRemoved) origOnRemoved.apply(this, arguments);
            };

            return r;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            this._configuredByLoad = true;
            const node = this;
            // ALBABIT-FIX: a single 100ms reapply can fire before
            // graph.configure() finishes applying the saved "preset" value,
            // folding as "Custom" with nothing to correct it after. Mirrors
            // radiance_sampler.js: 150ms for Vue's first layout pass, 600ms
            // as a safety net for heavier workflows.
            setTimeout(() => updateLoaderUI(node, false), 150);
            setTimeout(() => updateLoaderUI(node, false), 600);
            return r;
        };
    }
});
