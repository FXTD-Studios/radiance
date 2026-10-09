<!-- project-mapper:generated -->
# Architecture map and indexes

Snapshot: `7376f9e`. Status: static source analysis, not a runtime trace. Labels are defined in
[START_HERE](START_HERE.md#evidence-labels).

## System boundary

- **Host:** ComfyUI. Radiance imports `comfy.*`, `folder_paths`, `node_helpers`, `comfy_extras.*`,
  `comfy_api.latest`, and `server.PromptServer`. None of these is vendored. The tests stub them
  (`tests/conftest.py:_make_comfy_stubs`). Implemented.
- **First-party Python:** the repo root is the `radiance` package (`pyproject.toml`
  `package-dir {"radiance": "."}`). Implemented.
- **First-party browser code:** `js/`. `js/vendor/ocio/` is a vendored OpenColorIO 2.5 WASM build (excluded from analysis).
- **External processes and services:**
  - `ffmpeg`/`ffprobe` subprocesses.
  - Hugging Face, GitHub, and OpenAI (audio transcription) over the network.
  - Nuke socket server (`scripts/start_nuke_server.py`, which runs inside Nuke).
  - Resolve import worker (`tools/resolve_import.py`, a subprocess).
  - Node-started sockets: preview `HTTPServer`, DCC bridge, NDI.

  All Implemented.
- **Not present:** a PHP backend, a database, or a standalone UI process. Implemented (by absence: no `*.php`, no DB driver imports found).

## Layering (Implemented, from import statements)

```
nodes/<group>  ->  hdr/, color/, io/, image/, film/, delivery/, model/, loader_utils, sampler_utils
hdr/, image/   ->  color/, core/tensor, gpu/
io/            ->  core/exr, core/video, core/ffmpeg, core/formats, color/encodings
color/         ->  leaves: matrices, transfer, ops, luma
```

- No module under `color/`, `hdr/`, `core/tensor`, or `gpu/` imports `nodes/`, `io/`, or `delivery/`. `io/reader.py` and `io/writer.py` don't import the node layer, and tests enforce that.
- **Exceptions to the layering (Implemented):**
  - `delivery/handler.py` imports `radiance.image.upscale.RadianceAIUpscale`.
  - `nodes/pipeline/dcc.py` and `studio_integrations.py` import private helpers from `nodes/io/write.py`.
- The root shims `color_utils.py`, `tensor_contract.py`, `gpu_utils.py`, `path_utils.py`, and `secret_utils.py` re-export the newer modules. `color_utils` warns on import but is still used.

## Subsystem index

| ID | Responsibility | Entry/source evidence | State owner | Coverage | Note |
| --- | --- | --- | --- | --- | --- |
| entry, registry | Startup, group catalog, merge, fold-in, branding, gizmo loading | `__init__.py`, `nodes/catalog.py`, `nodes/registry.py`, `nodes/aggregate.py`, `nodes/branding.py` | process env, `<repo>/gizmos` | inspected | [registry](subsystems/registry.md) |
| color | Curves, matrices, encodings, LUT/CDL, OCIO setup and routes | `color/*`, `radiance_ocio.py`, `hdr/ocio.py`, `hdr/aces2_ocio.py` | OCIO current config (process global) | partial | [color-hdr](subsystems/color-hdr.md) |
| hdr-vae | Log-coded HDR VAE encode/decode, latent metadata | `hdr/vae.py`, `hdr/decode_meta.py`, `nodes/generate/engine.py` | `radiance_meta` in LATENT dicts | partial | [hdr-vae](subsystems/hdr-vae.md) |
| io, delivery | Read/Write engines, EXR, video, sequences; viewer export endpoint | `io/reader.py`, `io/writer.py`, `core/exr.py`, `core/video.py`, `delivery/handler.py` | output files, `radiance_sessions.json` | partial | [io-delivery](subsystems/io-delivery.md) |
| generate | Loader, detection, caches, sampler, LoRA, prompt, downloads, RUDRA | `loader_utils.py`, `model/*`, `nodes/generate/*`, `sampler_utils.py`, `core/model_fetch.py` | `model/cache.py` LRU singletons | partial | [generate](subsystems/generate.md) |
| vfx | Depth, flow, optics, matting, plate, multipass, relight, film camera | `nodes/vfx/*`, `nodes/vfx/multipass/*`, `film/*` | `GPUModelCache`, HF cache, `models/geometry_estimation`, `models/radiance/marigold` | partial | [vfx-multipass](subsystems/vfx-multipass.md) |
| upscale, qc | Upscale tiers, face restore, bit depth, QC, policy | `nodes/upscale/upscale.py`, `image/upscale.py`, `nodes/color/qc.py`, `image/defects.py` | `GPUModelCache` instances | partial | [upscale-qc-monitor](subsystems/upscale-qc-monitor.md) |
| monitor-realtime | False colour, peaking, split, contact sheet, stamp, GIF, preview server | `nodes/monitor/realtime.py` | `_PREVIEW_BUFFER`, `_SERVERS` (in memory) | partial | [upscale-qc-monitor](subsystems/upscale-qc-monitor.md) |
| video, ai, audio | T2V/I2V, latent specs, HDR video conditioning and decode, windowing, scene cut, audio | `nodes/video/*`, `nodes/ai/scene_cut.py`, `nodes/pipeline/audio.py` | `VideoAssembler._STORE` (in memory) | partial | [video-temporal](subsystems/video-temporal.md) |
| dcc, gizmo-runtime | Nuke/Resolve send, DCC bridge, NDI, subgraph executor | `nodes/pipeline/{dcc,studio_integrations}.py`, `tools/*`, `scripts/start_nuke_server.py`, `nodes/gizmo.py` | `~/.radiance/dcc_token`, bridge threads | partial | [dcc-pipeline](subsystems/dcc-pipeline.md) |
| viewer, workspace, frontend | Viewer node and cache, routes, dashboards, `.rad` storage, WebGL/WebGPU | `nodes/monitor/viewer.py`, `cache.py`, `nodes/pipeline/workspace.py`, `js/*` | viewer LRU, `<repo>/workflows` | partial | [frontend-routes](subsystems/frontend-routes.md) |
| validation | pytest, node tests, CI, release gates | `tests/conftest.py`, `.github/workflows/*.yml`, `tools/*` | n/a | inspected (structure) | [validation](subsystems/validation.md) |

## Workflow index

| ID | Workflow | Detail | Diagram |
| --- | --- | --- | --- |
| W1 | ComfyUI startup and node registration | [WORKFLOWS](WORKFLOWS.md#w1-comfyui-startup-and-node-registration) | [startup](diagrams/flows/startup.mmd) |
| W2, W3 | Read media; Write and export | [W2](WORKFLOWS.md#w2-load-media-read-node), [W3](WORKFLOWS.md#w3-write-and-export-write-node) | [read-write](diagrams/flows/read-write.mmd) |
| W4 | HDR generation round trip | [W4](WORKFLOWS.md#w4-hdr-generation-round-trip-starter-workflow-workflowsstartjson) | [hdr-generation](diagrams/flows/hdr-generation.mmd) |
| W5, W6 | Review in the Viewer; Deliver | [W5](WORKFLOWS.md#w5-review-in-the-viewer), [W6](WORKFLOWS.md#w6-deliver-from-the-viewer) | [viewer-deliver](diagrams/flows/viewer-deliver.mmd) |
| W7 | Workspace save and restore | [W7](WORKFLOWS.md#w7-workspace-save-and-restore-a-workflow) | in text only |
| W8 | Model download | [W8](WORKFLOWS.md#w8-model-download-consent-gated) | [model-download](diagrams/flows/model-download.mmd) |
| W9 | Multipass estimate, relight, and pass export | [W9](WORKFLOWS.md#w9-multipass-estimate-relight-and-pass-export) | [multipass](diagrams/flows/multipass.mmd) |
| W10 | Video generation | [W10](WORKFLOWS.md#w10-video-generation-t2v-i2v-and-export) | [video](diagrams/flows/video.mmd) |
| W11 | Send to Nuke or Resolve | [W11](WORKFLOWS.md#w11-send-to-nuke-or-resolve) | [dcc](diagrams/flows/dcc.mmd) |
| W12 | Upscale | [W12](WORKFLOWS.md#w12-upscale-image-or-video) | in text only |

Overview diagrams: [mind map](diagrams/project-mindmap.mmd) and [lifecycle flow](diagrams/runtime-flow.mmd).

## Relationships

| From | Type | To | Contract | Label |
| --- | --- | --- | --- | --- |
| entry | import side effect | registry | `from .nodes.registry import ...` (`__init__.py:38`) runs `nodes/__init__.py`, which loads every group before any env, logging, or OCIO setup | Implemented (mapper) |
| entry | call | color | `configure_ocio()` after the groups are loaded. Never fatal | Implemented |
| registry | import | `nodes/<group>` | One ImportError drops the whole group (WARNING). The health check then logs an ERROR | Implemented |
| nodes/generate | call | hdr-vae | HDR VAE nodes wrap `RadianceVAE4KEncode/Decode` (`engine.py:133, 186`) | Implemented (mapper) |
| nodes/io | call | io | `RadianceRead.read` calls `read_frames`; `RadianceWrite.write` calls `write_frames` | Implemented |
| viewer | data | frontend | `ui` output plus a `/view?type=temp` fetch of `.rhdr`/EXR/PNG. No `send_sync` | Implemented |
| frontend | http | delivery | `POST /radiance/deliver`; JS polls `/radiance/progress` | Implemented |
| delivery | data | viewer | Frames come from the viewer cache | Implemented |
| delivery | call | io, upscale | `write_frames`; optional `RadianceAIUpscale` for 2x | Implemented (mapper) |
| vfx | call | hdr (io) | `EXRPassesWriter` writes through `hdr/io.write_exr_openexr`, not `io/writer` | Implemented (trace) |
| video | call | hdr (io) | `VideoExport` writes EXR through `hdr/io.write_exr_robust` and GIF through PIL | Implemented (trace) |
| dcc | call | io | `NukeSend` and `DaVinciSend` use `io/writer._save_exr` / PIL directly, not `write_frames` | Implemented (mapper) |
| dcc | socket | Nuke | HMAC-framed TCP to `start_nuke_server.py` | Implemented (trace) |
| dcc | subprocess | Resolve | `tools/resolve_import.py`, JSON on stdin, 30 s timeout | Implemented (trace) |
| gizmo-runtime | call | ComfyUI node classes | Calls node classes directly, bypassing the ComfyUI executor | Implemented (trace) |
| generate, vfx, upscale | network | HF/GitHub | Several download mechanisms with different integrity guarantees (see OPEN_QUESTIONS B17) | Implemented |
| audio | network | OpenAI | `AudioTranscribe` uploads audio when the API backend is chosen. No consent gate | Implemented (trace) |

## Coverage

| Area | Status | Evidence | Missing or excluded |
| --- | --- | --- | --- |
| Entry, registry, catalog | inspected | all group `__init__` files, registry, catalog, branding | gizmo class generation details |
| color, hdr, OCIO | partial | module headers and key symbols; targeted reads of `vae.py` | `hdr/color.py`, `processing.py`, `panorama.py` bodies |
| io, core, delivery | partial | read/write dispatch, EXR, video, delivery endpoint | DPX/OIIO edge paths; `core/logging.py` internals |
| generate, model, sampler | partial | loader, detect, cache, main sampler path | `prompt.py`, `regional.py`, `resolution.py`, `denoise.py`, `energy.py` internals |
| vfx, multipass, film | partial | every node class, models, `RADIANCE_PASSES`, writer | shipped workflow not executed; RollingShutter body |
| upscale, qc, realtime | partial | backend chain, HDR wrap, caches, QC rules, preview server | spandrel dtype defaults; SeedVR2/facexlib download internals |
| video, ai, audio | partial | every node, latent specs, windowing code, scene cut, audio backends | ComfyUI noise-scaling semantics per family |
| dcc, gizmo runtime | partial | Nuke/Resolve/bridge/NDI protocols, executor | NDI SDK behaviour; Resolve API behaviour |
| JS frontend | partial | extension registry, renderer selection, data flow | most of the 23k-line `radiance_viewer.js` body |
| tests, CI, release | inspected (structure) | conftest, CI YAML, pyproject, tools | test bodies; **no tests run** |
| Excluded | excluded | none | `js/vendor/`, `ACES/config.ocio` beyond its header, images, `workflows/*.json` beyond their node types, `scripts/training/*` beyond a summary, generated `docs/nodes/` |

"Partial" means the main entry points, contracts, and failure paths were traced, but not every function body.

## Documentation drift

Kept once, in [OPEN_QUESTIONS § Documentation drift](OPEN_QUESTIONS.md#documentation-drift).
