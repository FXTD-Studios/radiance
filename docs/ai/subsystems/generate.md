<!-- project-mapper:generated -->
# Generation, model management, and downloads

ID: generate (plus sdr2hdr model side). Snapshot: `7376f9e`. Coverage: partial (static).

## Responsibility and boundaries

This component covers the unified model loader, architecture detection, model caches, the
Radiance sampler, LoRA, the prompt encoder, model downloads, and the RUDRA SDR to HDR models.
Device placement and memory are delegated to ComfyUI's `comfy.model_management`.

## Implementation index

| Source/symbol | Role | Status |
| --- | --- | --- |
| `nodes/generate/loader.py:RadianceUnifiedLoader` and `VideoLoader` | Widgets from `folder_paths.get_filename_list`. Finds the WAN 2.2 MoE companion file (`_find_wan_moe_companion`) | observed |
| `loader_utils.py` | `comfy.sd.load_diffusion_model`, `load_clip`, `VAE`, `load_lora_for_models`. Cache keys; `ensure_model_exists`; `setup_offload_mode` | observed |
| `model/detect.py:detect_model_type` (~:528) | Reads safetensors header keys and shapes only. Ordered heuristics, then ComfyUI `model_config_from_unet`. The Loader's fallback is `"sdxl"` | observed |
| `model/detect.py` tables | `LATENT_CHANNELS`, `VAE_SPATIAL_FACTOR`, `CLIP_SLOT_ORDER`, `CLIP_SLOTS_REQUIRED` | observed |
| `model/cache.py` | `LRUCache` (size from `RADIANCE_CACHE_SIZE`, default 2) and `GPUModelCache` (eviction moves to CPU and calls `empty_cache`) | observed |
| `nodes/generate/sampler.py:RadianceSamplerPro.sample` (~:1533) | Presets, then model defaults, then 4D/5D latent, noise, sigmas, then clone and patch, then `comfy.sample.sample_custom` | observed |
| `sampler_utils.py:RadianceModelRegistry` (`detect_by_config` ~:584) | Model-type detection on the MODEL object, used when `model_meta` isn't connected | observed; Flux2 bug |
| `config/model_map.py:RADIANCE_MODEL_MAP` | 71 entries: filename to {pinned HF url, sha256, size, folder type, gated} | observed (agent) |
| `core/model_fetch.py:fetch` | Requires a sha256 pin. Re-checks consent. `.part` file with Range resume. HF bearer token. `os.replace` after the digest matches | observed |
| `core/consent.py:downloads_allowed` | **Default allow.** Off with `RADIANCE_ALLOW_DOWNLOADS=0`, `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, or the legacy flags | observed |
| `nodes/hdr/uplift_universal.py`, `pixel_sdr2hdr.py`, `temporal_rudra.py`, `model/pixel_download.py` | RUDRA SDR to HDR. The checkpoint is downloaded on first use (gated by consent) | observed (agent) |

## Contracts

- **Families with real code support** (detection plus tables): FLUX.1, FLUX.2 Dev/Klein, Chroma,
  SD1.5, SDXL, SD3, WAN 2.1/2.2 (including TI2V 48ch), LTX/LTX-AV, HunyuanVideo/1.5,
  HunyuanImage 2.1, Lumina2, Z-Image, Qwen-Image/2.1, and others. Download entries exist only
  for the FLUX, SDXL, LTX, MiniMax, and Qwen families. There are none for WAN, Hunyuan, or Z-Image.
- **Cache keys** include the file's `mtime:size` fingerprint and dtype/offload options. The
  sampler clones the ModelPatcher before patching, so cached models aren't mutated.
- **Sampler outputs:** `LATENT, SIGMAS, SIGMAS(remaining), IMAGE(sigma plot)`.
- **HDR doesn't enter sampling.** The only HDR-related input is the optional SDR reference anchor.

## Important paths

- **Offload `sequential`** sets the global `comfy.model_management.vram_state = LOW_VRAM`
  for the whole process and never restores it.
- **Cleanup:** the sampler calls `torch.cuda.empty_cache()` in `finally`. No `soft_empty_cache`
  call was found. Radiance's LRU caches aren't tied to ComfyUI's "free memory" action (inferred).

## Tests

About 28 files: `tests/test_sampler_*` (CPU harness `tests/_sampler_harness.py`),
`test_resolution_*`, `test_prompt_*`, `test_model_*`, `test_loader_*`, `test_qwen_*`.
`test_download_consent*.py` and `test_pixel_download.py` monkeypatch `urlopen`. Not run.

## Open questions

See OPEN_QUESTIONS: the Flux2 to "flux" detection order, `_encode_sdr_reference` indexing a
tensor, the global `vram_state`, `image/upscale` ignoring `RADIANCE_UPSCALE_OFFLINE`, and the
ignored return value of `validate_runtime_dependencies`.
