[← Back to the README](../README.md)

# Models and downloads

| Model | Best for | Loader preset | Speed | Downloads |
|---|---|---|---|---|
| [FLUX.2 Klein](#flux2-klein-starter-workflow) | Getting started, fast drafts | Flux.2 | 4 steps | partly automatic |
| [FLUX 3 Image Klein](#flux-3-image-klein) | Quality, editing with up to 10 references, layouts | Flux.3 Klein | 16 steps (distilled) · 50 (base) | manual, gated |
| [Qwen-Image 2.1](#qwen-image-21) | Editing with up to 16 references, transparency | Qwen-Image 2.1 | 25 steps | automatic |
| [RUDRA](#rudra-sdr-to-hdr) | Turning SDR images into HDR | (SDR → HDR nodes) | one pass | automatic |

Every image model has a **(Low VRAM)** preset that picks smaller files and offloads to the CPU. To turn off automatic downloads, set `RADIANCE_ALLOW_DOWNLOADS=0` before starting ComfyUI; every automatic download is checked against a file hash.

<a id="flux2-klein-starter-workflow"></a>
<details>
<summary><b>FLUX.2 Klein (starter workflow): files</b></summary>
<br>

| File | Download | Save in |
|---|---|---|
| `flux-2-klein-4b-fp8.safetensors` | [FLUX.2 Klein 4B distilled](https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-fp8/blob/main/flux-2-klein-4b-fp8.safetensors) | `models/diffusion_models/` |
| `flux2-vae.safetensors` | [FLUX.2 VAE](https://huggingface.co/Comfy-Org/flux2-dev/blob/main/split_files/vae/flux2-vae.safetensors) | `models/vae/` |
| `qwen_3_4b.safetensors` | [Qwen3 4B text encoder](https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-4b/blob/main/split_files/text_encoders/qwen_3_4b.safetensors) | `models/text_encoders/` |

The starter workflow is set up for the distilled model; the `base` variant needs different sampling settings.

</details>

<a id="flux-3-image-klein"></a>
<details>
<summary><b>FLUX 3 Image Klein: files and tips</b></summary>
<br>

Black Forest Labs' 5.6B open-weights model. **Distilled** is fast (16 steps, CFG 1); **base** is for the highest fidelity and fine-tuning (50 steps, CFG 3.5). Needs **ComfyUI 0.41.1** or a build from 15 October 2026 or later.

| File | Save in |
|---|---|
| `flux3-image-klein-*.safetensors` (distilled) or `flux3-image-klein-base-*.safetensors` (base): bf16, int8 or nvfp4. BFL's first names also work: `flux3_image_klein_sd_*` (distilled), `flux3_image_klein_*` (base) | `models/diffusion_models/` |
| `qwen3vl_4b_bf16.safetensors` or `qwen3vl_4b_int8_convrot.safetensors` | `models/text_encoders/` |
| `flux2-vae.safetensors` (the FLUX.2 VAE) | `models/vae/` |

Download from [black-forest-labs/flux-3-image-klein-comfy](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy). The repository is gated: accept the FLUX Kommunity licence on Hugging Face first. The Loader does not download these files.

- **8 to 12 GB cards:** use **Flux.3 Klein (Low VRAM)**.
- **Edit:** connect up to 10 images to the Prompt's `images` and the VAE, and refer to them as `<ref_image_0>`, `<ref_image_1>`…
- **Layout boxes:** in the Viewer, **Pro tools › Tools › FLUX 3 Layout Boxes**. Draw boxes, describe each, click **Send to Prompt**, and refer to them as `<obj_1>`, `<obj_2>`…
- **Prompt upsampler (optional):** connect the Loader's `upsampler_clip` output to the Prompt's `upsampler_clip` and turn on `upsample_prompt`: the Flux.3 Klein presets then pick `qwen3.5_9b_flux3_prompt_upsampler_*` in the Loader's `upsampler` (back to None when it is off, so it is not loaded for nothing).

</details>

<a id="qwen-image-21"></a>
<details>
<summary><b>Qwen-Image 2.1: files and tips</b></summary>
<br>

Qwen's image model with its own RGBA VAE, so it can make and edit images with transparency. The Loader downloads the official files on first use, or place them yourself:

| File | Download | Save in |
|---|---|---|
| `qwen_image_2.1_bf16.safetensors` or `qwen_image_2.1_int8_convrot.safetensors` | [Model](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/tree/main/diffusion_models) | `models/diffusion_models/` |
| `qwen3vl_8b_bf16.safetensors`, `_int8_convrot` or `_w4a8` | [Text encoder](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/tree/main/text_encoders) | `models/text_encoders/` |
| `qwen_image_2.1_vae_bf16.safetensors` | [VAE](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/tree/main/vae) | `models/vae/` |

- **Edit:** connect images to the Prompt's `images` and refer to them as `<image1>`, `<image2>`…
- **Avoid seed 0:** edits at that seed fail or oversaturate (in ComfyUI's own nodes too). The Sampler's default is 1.
- **More detail:** the template runs 25 steps; Qwen's own pipeline uses 40 to 50.
- **Prompt upsampler (optional):** the `qwen3.5_9b_qwen_image_2.1_pe_*` files are prompt enhancers, not text encoders. Connect the Loader's `upsampler_clip` to the Prompt's and turn on `upsample_prompt`: the Qwen-Image 2.1 presets then pick the text-to-image one (`pe_t2i`) in the Loader's `upsampler`; choose `pe_i2i` there for editing.

</details>

<a id="rudra-sdr-to-hdr"></a>
<details>
<summary><b>RUDRA (SDR to HDR): files and licence</b></summary>
<br>

[RUDRA 1.0](https://huggingface.co/fxtdstudios/rudra-models) is the learned model behind **SDR → HDR Universal**, **SDR → HDR Recover** and **SDR → HDR AI Fill**.

- **RUDRA ACES 2** (2.0, default, 2.2 MB, downloads on first use): the learned ACES 2.0 reconstruction, with the lock it was released with (RUDRA 1.2). SDR in, an ACEScg master out (diffuse white = 1.0 = 203 nits) whose ACES 2.0 SDR view keeps the source's 8-bit codes. On held-out footage it is within 0.1 code of the plain ACES 2.0 inverse at the delivered exposure and better at −3 and +3 EV. It is the inverse of the ACES 2.0 SDR view, so it suits SDR rendered through ACES 2.0; on a grade made another way, bright areas just under white can come out very bright. Universal and Recover use it whenever it can run; the `reconstructed_mask` output says where light was estimated.
- **RUDRA AI fill v3** (271 MB, plus the 7.7 GB Wan base model; both download the first time the AI Fill node runs): a LoRA for Wan2.1-VACE-1.3B that draws the shape inside fully clipped highlights, plus an estimator that sets their brightness (±3 stops). Region prompts. Needs about 16 GB of VRAM. Experimental: what it draws is plausible, not recovered.
- **RUDRA AI fill v4, region re-render** (experimental, not published): the **SDR → HDR AI Fill v4 Re-render** node draws a whole bright region again (a window, a door, a sky just under white) from the SDR source, with RUDRA's app finish. Put its LoRA in `models/radiance/rudra_genfill_v4/`; it is never downloaded.

| Files | Download | Save in |
|---|---|---|
| `manifest.json`, `parity.json`, `rudra_g1.ts`, `rudra_g1.onnx` | [RUDRA ACES 2](https://huggingface.co/fxtdstudios/rudra-models/tree/ab8da1b2db74099eb45cc39585df1d2edf7b3e34/models/RUDRA_ACES_2) | `models/radiance/RUDRA_ACES_2/` |
| `rudra_genfill.safetensors`, `brightness.json`, `prompt_cache/` | [AI fill v3](https://huggingface.co/fxtdstudios/rudra-models/tree/v1.0.0/models/rudra_genfill_v3) | `models/radiance/rudra_genfill_v3/` |
| the whole repository | [Wan2.1-VACE-1.3B-diffusers](https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B-diffusers) | `models/radiance/Wan2.1-VACE-1.3B-diffusers/` |
| `sdr2hdr_shadow_v1.safetensors` (RUDRA 3.x, fallback) | [Browse](https://huggingface.co/fxtdstudios/RUDRA/tree/main/sdr2hdr) | `models/radiance/` |

**Nothing to set up.** Installing Radiance through ComfyUI-Manager or the Comfy Registry installs the libraries from `requirements.txt` (OpenColorIO 2.5+, diffusers 0.35+), and `install.py` checks them (printing the pip line for anything missing) and fetches RUDRA ACES 2; if you install by hand, it downloads on the first SDR → HDR run instead. The AI fill's files download the first time that node runs on a frame with clipped highlights: the LoRA package and the Wan base model (VAE, transformer, tokenizer), and the 11.4 GB umT5 text encoder only the first time you type a custom prompt (the default prompt is cached). Every file is pinned and hash-checked, and an interrupted download resumes. To keep everything offline, set `RADIANCE_ALLOW_DOWNLOADS=0` and put the files above in place yourself.

**Licence:** the RUDRA 1.0 models (ACES 2, AI fill v3) are Apache 2.0, so commercial use is allowed; Wan2.1-VACE-1.3B is Apache 2.0 from Wan-AI. The older RUDRA 3.x weights (`sdr2hdr_shadow_v1` and the temporal model) stay non-commercial ([weights licence](https://huggingface.co/fxtdstudios/RUDRA/blob/main/LICENSE)).

</details>

Review each model's licence before use.
