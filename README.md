<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/radiance-logo-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="assets/radiance-logo-light.svg">
  <img src="assets/radiance-logo-light.svg" width="420" alt="Radiance">
</picture>

### HDR, colour and VFX review for ComfyUI

Generate in HDR, grade and review like a finishing suite, and deliver masters to Nuke and DaVinci Resolve,<br>without leaving ComfyUI.

[![Version](https://img.shields.io/badge/version-4.0.0-c8a96e?style=flat-square)](CHANGELOG.md)
[![Status](https://img.shields.io/badge/status-beta-orange?style=flat-square)](CHANGELOG.md)
[![Installs](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.comfy.org%2Fnodes%2Fradiance&query=%24.downloads&label=installs&color=blue&style=flat-square)](https://registry.comfy.org/nodes/radiance)
[![Comfy Registry](https://img.shields.io/badge/Comfy_Registry-Radiance-orange?style=flat-square)](https://registry.comfy.org/nodes/radiance)
[![Platforms](https://img.shields.io/badge/Windows%20·%20macOS%20·%20Linux-5C7CFA?style=flat-square)](#before-you-start)
[![License](https://img.shields.io/badge/license-GPL--3.0-green?style=flat-square)](LICENSE)

**[Install](#installation)** &nbsp;·&nbsp; **[Quick start](#quick-start)** &nbsp;·&nbsp; **[Models](#choose-a-model)** &nbsp;·&nbsp; **[Viewer](#using-the-viewer)** &nbsp;·&nbsp; **[Deliver](#delivering-masters)** &nbsp;·&nbsp; **[Help](#support)**

<br>

<img src="viewer.png" width="920" alt="The Radiance Viewer: picture, grade controls and scopes">

</div>

<br>

## What Radiance does

147 visible nodes, grouped around four jobs:

<table>
<tr>
<td width="25%" valign="top">

**Generate**

FLUX 3 Klein, Qwen-Image 2.1, FLUX.2, Wan, LTX Video and more, from one Loader with presets. The Sampler and Resolution set themselves for the model.

</td>
<td width="25%" valign="top">

**Make it HDR**

SDR → HDR with the RUDRA model, HDR VAE encode and decode, tone mapping, ACES 2.0 and OpenColorIO, EXR in and out.

</td>
<td width="25%" valign="top">

**Review**

A pro Viewer: grade, scopes, A/B compare, notes on frames, and a QC pass that checks every frame for you.

</td>
<td width="25%" valign="top">

**Deliver**

EXR, ProRes and HDR10 / HLG masters with your grade baked in, a render queue, and hand-off to Nuke and Resolve.

</td>
</tr>
</table>

## What's new in 4.0

> [!NOTE]
> 4.0 is in **beta** on the `version-4-Beta` branch. Coming from 3.x? Read the [upgrade notes](CHANGELOG.md) first.

| New | What you get |
|---|---|
| **FLUX 3 Image Klein** | Base and distilled, with reference editing and layout boxes you draw in the Viewer |
| **Qwen-Image 2.1** | Text-to-image and editing with up to 16 references, and images with transparency |
| **Prompt upsampler** | Turns a short prompt into the detailed one FLUX 3 Klein and Qwen-Image 2.1 were trained on |
| **Deliver Master** | Graded-master export from the Viewer, with a render queue and HDR10 / HLG |
| **Viewer workspaces** | Review, Color, Inspect, Compare and QC, each showing only what that job needs |
| **Compare and QC** | Scene-linear difference for HDR plates, onion skin, and a QC workspace that lists problem frames |

<details>
<summary><b>Everything else in 4.0</b></summary>
<br>

- **Review tools:** bookmarks, notes and drawings on frames, saved with the node, and an exportable review report.
- **Channels:** Depth and 3D Normal beside R, G, B and A; compare inputs at their own frame rate.
- **HDR10:** choose the mastering display (P3-D65 or Rec.2020, 1000 or 4000 nits); MaxCLL / MaxFALL are measured from the frames.
- **Safer files:** saves, sidecars and DCC frames are written atomically, sequence writes warn about stale frames, and a remote DCC bridge needs signed requests.

The full list is in the [changelog](CHANGELOG.md).

</details>

## Installation

**With ComfyUI Manager** (recommended)

1. Open **ComfyUI Manager** and search for **Radiance**.
2. Click **Install**.
3. Restart ComfyUI and refresh your browser.

<details>
<summary><b>Install manually with Git</b></summary>
<br>

Run these with the Python that runs ComfyUI:

```sh
cd ComfyUI/custom_nodes
git clone https://github.com/fxtd-studios/radiance.git
cd radiance
python -m pip install -r requirements.txt
```

**Windows portable:** use ComfyUI's bundled `python_embeded\python.exe` instead of `python`.

**To try the 4.0 beta**, switch branch and install the requirements again:

```sh
git checkout version-4-Beta
python -m pip install -r requirements.txt
```

Restart ComfyUI and refresh your browser.

</details>

<details>
<summary><b>Update</b></summary>
<br>

Click **Update** in ComfyUI Manager, or inside the `radiance` folder:

```sh
git pull
python -m pip install -r requirements.txt
```

Restart ComfyUI and refresh your browser. Back up your workflows before a major update.

</details>

## Quick start

<table>
<tr>
<td width="50%" valign="top">

### Turn an image into HDR

**Read → SDR → HDR Universal → Viewer**

1. Load an image with **Read**.
2. Connect it to **SDR → HDR Universal**.
3. Connect that to **Viewer** and press **Run**.
4. Add **Write** to save it as EXR.

The HDR model downloads the first time.

</td>
<td width="50%" valign="top">

### Generate an image

1. Drag the [starter workflow](workflows/start.json) onto the canvas.
2. Pick the model files in the **Loader** ([which files?](#flux2-klein-starter-workflow)).
3. Type your prompt and press **Run**.
4. Review it in the **Viewer**; add **Write** to save.

For another model, pick its preset in the Loader. The Sampler sets its steps and CFG for you.

</td>
</tr>
</table>

<div align="center">
<img src="basic_workflow.png" width="920" alt="The starter workflow: Loader, Prompt, Sampler and Viewer">
</div>

## Choose a model

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
| `flux3_image_klein_sd_*.safetensors` (distilled) or `flux3_image_klein_*.safetensors` (base): bf16, int8 or nvfp4 | `models/diffusion_models/` |
| `qwen3vl_4b_bf16.safetensors` or `qwen3vl_4b_int8_convrot.safetensors` | `models/text_encoders/` |
| `flux2-vae.safetensors` (the FLUX.2 VAE) | `models/vae/` |

Download from [black-forest-labs/flux-3-image-klein-comfy](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy). The repository is gated: accept the FLUX Kommunity licence on Hugging Face first. The Loader does not download these files.

- **8 to 12 GB cards:** use **Flux.3 Klein (Low VRAM)**.
- **Edit:** connect up to 10 images to the Prompt's `images` and the VAE, and refer to them as `<ref_image_0>`, `<ref_image_1>`…
- **Layout boxes:** in the Viewer, **Tools > FLUX 3 Layout Boxes**. Draw boxes, describe each, click **Send to Prompt**, and refer to them as `<obj_1>`, `<obj_2>`…
- **Prompt upsampler (optional):** load `qwen3.5_9b_flux3_prompt_upsampler_*` with **Load CLIP** (type `flux3`), connect it to the Prompt's `upsampler_clip` and turn on `upsample_prompt`.

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
- **Prompt upsampler (optional):** the `qwen3.5_9b_qwen_image_2.1_pe_*` files are prompt enhancers, not text encoders. Load one with **Load CLIP** (type `qwen_image`), connect it to `upsampler_clip` and turn on `upsample_prompt`.

</details>

<a id="rudra-sdr-to-hdr"></a>
<details>
<summary><b>RUDRA (SDR to HDR): files and licence</b></summary>
<br>

[RUDRA](https://huggingface.co/fxtdstudios/RUDRA) is the learned model behind **SDR → HDR Universal** and **SDR → HDR Recover**. The default downloads on first use.

| File | Download | Save in |
|---|---|---|
| `sdr2hdr_shadow_v1.safetensors` | [Default model](https://huggingface.co/fxtdstudios/RUDRA/resolve/main/sdr2hdr/sdr2hdr_shadow_v1.safetensors) | `models/radiance/` |
| Other checkpoints | [Browse](https://huggingface.co/fxtdstudios/RUDRA/tree/main/sdr2hdr) | `models/radiance/` |

**Licence:** RUDRA weights are for non-commercial use ([weights licence](https://huggingface.co/fxtdstudios/RUDRA/blob/main/LICENSE)), separate from the Radiance code licence.

</details>

Review each model's licence before use.

## Example workflows

Drag a file onto the ComfyUI canvas to open it.

| Workflow | What it does |
|---|---|
| [Starter](workflows/start.json) | Generate an image and review it |
| [SDR to HDR](workflows/sdr_to_hdr.json) | Turn an image into HDR and save EXR |
| [Colour grading](workflows/colour_grade.json) | Grade an image and save it |
| [HDR VAE](workflows/hdr_vae_encode_decode.json) | Encode, decode and compare an HDR image |
| [Multipass relighting](workflows/multipass_relight.json) | Estimate passes and relight |
| [HDR delivery](workflows/hdr_delivery.json) | Make EXR, HDR10 and SDR outputs |

## Using the Viewer

The Viewer opens in **Simple** mode: picture, compare and playback. Switch to **Advanced** in its title bar for the workspaces.

| Workspace | Use it to |
|---|---|
| **Review** | Play, bookmark frames, write notes, export a review report |
| **Color** | Grade, mask, add effects, read the scopes |
| **Inspect** | Read pixel values, channels and analysis |
| **Compare** | Wipe, difference (display or scene-linear), onion skin and blink against a reference |
| **QC** | Check every frame for NaN/Inf, range, alpha and flicker, and jump to the bad ones |

<details>
<summary><b>Handy keys</b> (hover the Viewer first; <b>H</b> or <b>?</b> lists them all)</summary>
<br>

| Key | Does |
|---|---|
| **F** | Fit the picture |
| **X** | Compare on or off |
| **D** | Bookmark this frame |
| **,** and **.** | Previous / next bookmark |
| **T** | Write a note on this frame |
| **Alt + drag** | Draw on the frame |
| **Alt + 1…7** | Open a panel (Inspector … QC) |

</details>

- **Tools > FLUX 3 Layout Boxes** draws layout boxes on the picture and sends them to the Prompt.
- **File > Deliver Master…** exports the graded result (see below).
- Viewer adjustments change the preview only, until you deliver or add the same processing to the workflow.
- WebGL is the default renderer. WebGPU is experimental and lists what it can't do yet.

## Delivering masters

- **Write** saves images, sequences (EXR, PNG, TIFF, DPX) and video from the graph.
- **Deliver Master** (Viewer > **File > Deliver Master…**) exports with the Viewer's grade baked in: ACES, camera log, PQ (HDR10) or HLG, with CDL and AMF sidecars and automatic versioning.
- **Render queue:** set up a version, **Add to Queue**, repeat, then **Start Queue**. A failed job doesn't stop the rest.
- **HDR:** PQ and HLG need 10-bit or float, so 8-bit formats are disabled for them. H.265 10-bit carries HDR10 metadata with the mastering display you choose; scene-linear 1.0 sits at 203 nits (BT.2408).
- **Checks before writing:** frames with NaN or infinite pixels are refused, and HDR QC reports the peak and how much clips.

## Before you start

| You need | Why |
|---|---|
| A recent **ComfyUI** | FLUX 3 Klein needs 0.41.1 or later; the prompt upsampler needs a Generate Text node with a system prompt |
| **ffmpeg** | For video: on the PATH, the `imageio-ffmpeg` package, or `RADIANCE_FFMPEG` pointing at it |
| **OpenColorIO 2.5+** | For the reference ACES 2.0 output transform (older versions fall back to an approximation and say so) |
| **Disk space** | Some tools download models on first use; Multipass Estimate downloads the most |

<details>
<summary><b>Known limits</b></summary>
<br>

- Long-video windowing is experimental and off by default; some scenes distort when split.
- Depth and material passes are estimates, so results vary and video may flicker.
- If ComfyUI has cleared its temporary previews, run the workflow again.

The full list is in [KNOWN_ISSUES.md](KNOWN_ISSUES.md).

</details>

## Documentation

- **[Radiance website](https://www.fxtdstudios.com/radiance-comfyui.html)**: guides and tutorials
- **[Wiki](https://github.com/fxtd-studios/radiance/wiki)**: how-tos and troubleshooting
- **[Node reference](docs/nodes/README.md)**: every node and input
- **[Changelog](CHANGELOG.md)**: what changed and upgrade notes

Tip: in ComfyUI, hover over any node or input to read what it does.

## Support

Something not working? Try these first:

1. Restart ComfyUI and refresh your browser.
2. Look in the ComfyUI console for errors or missing packages.
3. Make sure the requirements went into ComfyUI's own Python.

Still stuck? [Open an issue](https://github.com/fxtd-studios/radiance/issues) with your Radiance and ComfyUI versions, your operating system, the error message and a small workflow that shows the problem.

## License

Radiance is licensed under [GPL-3.0](LICENSE). Model weights have their own licences; check them before you use or share them.

<div align="center">
<br>
Made by <a href="https://fxtdstudios.com"><b>FXTD Studios</b></a>
</div>
