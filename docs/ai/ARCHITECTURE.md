<!-- project-mapper:generated -->
# Architecture map

Snapshot: `7376f9e`, clean worktree. Status: static source analysis only, no runtime trace.

## System boundary

- **Host:** ComfyUI. Radiance imports `comfy.*`, `folder_paths`, `node_helpers`,
  `comfy_extras.*`, `comfy_api.latest`, and `server.PromptServer`. None of these is vendored.
  The tests stub all of them (`tests/conftest.py:_make_comfy_stubs`).
- **First-party Python:** the repo root is the `radiance` package
  (`pyproject.toml` `package-dir {"radiance": "."}`, with 26 packages listed).
- **First-party browser code:** `js/` (ES modules, served by ComfyUI).
  `js/vendor/ocio/` is a vendored OpenColorIO 2.5 WASM build.
- **External processes and services:**
  - `ffmpeg`/`ffprobe` subprocesses
  - Hugging Face and GitHub downloads
  - a Nuke socket server (`scripts/start_nuke_server.py`, which runs inside Nuke)
  - a DaVinci Resolve import worker (`tools/resolve_import.py`, run as a subprocess)
  - an optional local preview `http.server` (`nodes/monitor/realtime.py`)
  - a DCC bridge TCP socket (`nodes/pipeline/dcc.py`)
- **Not present:** a PHP backend, a database, or a standalone UI process.

## Layering (observed import direction)

```
nodes/<group>  ->  hdr/, color/, io/, image/, film/, delivery/, model/, loader_utils, sampler_utils
hdr/, image/   ->  color/, core/tensor, gpu/
io/            ->  core/exr, core/video, core/ffmpeg, core/formats, color/encodings
color/         ->  (leaves: matrices, transfer, ops, luma)
```

No module under `color/`, `hdr/`, `core/tensor`, or `gpu/` imports `nodes/`, `io/`, or `delivery/`.
`io/reader.py` and `io/writer.py` don't import the node layer, and tests enforce that
(`tests/test_reader_layering.py`, `test_writer_layering.py`).

The root-level `color_utils.py`, `tensor_contract.py`, `gpu_utils.py`, `path_utils.py`, and
`secret_utils.py` are compatibility shims that re-export `color/`, `core/tensor/contract`,
`gpu/ops`, and `core/system/*`. `color_utils` emits a DeprecationWarning but is still
imported by `io/reader.py`, `io/writer.py`, and `nodes/monitor/realtime.py`.

## Component index

| ID | Responsibility | Entry/source evidence | State owner | Detail |
| --- | --- | --- | --- | --- |
| entry | Env setup, OCIO config, node load, health banner | `__init__.py:_load_comfyui_nodes`, `report_node_load_health` | process env (`OCIO`, `OPENCV_IO_ENABLE_OPENEXR`) | [registry](subsystems/registry.md) |
| registry | Group catalog, merge, fold-in, branding, gizmos | `nodes/catalog.py`, `nodes/registry.py:load_node_mappings`, `nodes/aggregate.py:fold_in_module_nodes`, `nodes/branding.py`, `nodes/gizmo.py` | `<repo>/gizmos/*.gizmo` | [registry](subsystems/registry.md) |
| color | Transfer curves, matrices, encodings, LUT/CDL, OCIO setup | `color/ops.py`, `color/transfer.py`, `color/encodings.py`, `color/ocio_setup.py`, `radiance_ocio.py` | OCIO current config (global) | [color-hdr](subsystems/color-hdr.md) |
| hdr | HDR VAE engine, tonemap, ACES 2.0, OCIO nodes, processing | `hdr/vae.py`, `hdr/decode_meta.py`, `hdr/tonemap.py`, `hdr/aces2_ocio.py`, `nodes/hdr/*` | `radiance_meta` on latents | [hdr-vae](subsystems/hdr-vae.md), [color-hdr](subsystems/color-hdr.md) |
| io | Read/Write engines, EXR, video, sequences | `io/reader.py:read_frames`, `io/writer.py:write_frames`, `core/exr.py`, `core/video.py`, `core/ffmpeg.py` | files under the ComfyUI output dir or absolute paths | [io-delivery](subsystems/io-delivery.md) |
| delivery | Viewer export endpoint: grade, QC, write, sidecars | `delivery/handler.py:radiance_deliver_endpoint` | `radiance_sessions.json` (atomic) | [io-delivery](subsystems/io-delivery.md) |
| generate | Unified loader, model detection, sampler, prompt, LoRA, model downloads | `loader_utils.py`, `model/detect.py`, `nodes/generate/sampler.py`, `sampler_utils.py`, `config/model_map.py`, `core/model_fetch.py` | `model/cache.py` LRU singletons | [generate](subsystems/generate.md) |
| sdr2hdr | Learned SDR to HDR (RUDRA) plus heuristics | `nodes/hdr/uplift_universal.py`, `pixel_sdr2hdr.py`, `temporal_rudra.py`, `model/pixel_download.py` | per-module GPU caches | [generate](subsystems/generate.md) |
| vfx | Depth, flow, optics, masking, multipass, inpaint | `nodes/vfx/*`, `nodes/vfx/multipass/*`, `film/camera.py` | `model/cache.py` | (not detailed; see OPEN_QUESTIONS) |
| upscale | Tiled and AI upscalers | `nodes/upscale/upscale.py`, `image/upscale.py` | `GPUModelCache` instances | (not detailed) |
| video | Video model helpers, T2V/I2V pipelines, HDR video | `nodes/video/t2v.py`, `nodes/video/hdr.py` | none found | (not detailed) |
| viewer | Viewer node, frame cache, `.rhdr` sidecars, progress | `nodes/monitor/viewer.py:RadianceViewer.view`, `cache.py` | `cache.py` viewer LRU (2 GiB default) | [frontend-routes](subsystems/frontend-routes.md) |
| workspace | Workflow library, projects, assets (REST) | `nodes/pipeline/workspace.py` | `<repo>/workflows/`, `.versions/`, `_assets_bins.json` | [frontend-routes](subsystems/frontend-routes.md) |
| dcc | Nuke, Resolve, DCC bridge | `nodes/pipeline/dcc.py`, `nodes/pipeline/studio_integrations.py`, `tools/nuke_connector.py`, `core/dcc_auth.py` | `~/.radiance/dcc_token` | (not detailed) |
| frontend | Viewer UI, WebGL/WebGPU renderers, widgets, dashboards | `js/radiance_viewer.js`, `js/radiance_webgl.js`, `js/radiance_workspace.js` | browser localStorage | [frontend-routes](subsystems/frontend-routes.md) |
| validation | pytest, node tests, CI, release gates | `tests/conftest.py`, `.github/workflows/ci.yml`, `tools/check_release_ready.py` | n/a | [validation](subsystems/validation.md) |

## Relationships

| From | Type | To | Contract | Evidence/status |
| --- | --- | --- | --- | --- |
| entry | import | registry | First load happens as a side effect of `from .nodes.registry import ...` (`__init__.py:38`). `_load_comfyui_nodes` re-imports the cached `.nodes` (`required=True`) and folds group failures into the health check | `__init__.py:38, 44-71`, observed |
| entry | call | color | `configure_ocio()` runs **after** the groups have already been imported (via `__init__.py:38`). Never fatal | `__init__.py:_configure_ocio`, observed |
| registry | import | all `nodes/<group>` | Any ImportError drops the whole group (WARNING) | `nodes/registry.py:131-138`, observed |
| nodes/generate | call | hdr | HDR VAE nodes wrap `RadianceVAE4KEncode/Decode` | `nodes/generate/engine.py:133,186`, observed |
| nodes/io | call | io | `RadianceRead.read` to `read_frames`; `RadianceWrite.write` to `write_frames` | `nodes/io/write.py:471,855`, observed |
| viewer | data | frontend | ComfyUI `ui` output plus `/view?type=temp` file fetch (`.rhdr`/EXR/PNG). No `send_sync` | `viewer.py:~703-884`, observed |
| frontend | http | delivery | `POST /radiance/deliver`, polls `GET /radiance/progress` | `js/radiance_viewer.js:4645`, observed |
| delivery | data | viewer | Reads frames from the viewer cache (`cache._viewer_cache_get`) | `delivery/handler.py`, observed |
| delivery | call | io | `write_frames(...)` | `delivery/handler.py:876`, observed |
| frontend | http | workspace | `/radiance/workflows/*`, `/projects/*`, `/assets*` | dashboards in `js/*.html/.mjs`, observed |
| generate | call | network | `core/model_fetch.fetch`, gated by `core/consent.downloads_allowed` | `loader_utils.ensure_model_exists`, observed |
| sdr2hdr | call | network | `model/pixel_download` is consent-gated but does not use `fetch()` | report, inferred (not re-read) |
| hdr (ACESConfigManager) | call | network | Fetches the OCIO config from GitHub and sets `$OCIO` | `hdr/ocio.py:597,719`, observed; consent gating unknown |

## Views

[Mind map](diagrams/project-mindmap.mmd) |
[Runtime flow](diagrams/runtime-flow.mmd) | [Workflows](WORKFLOWS.md)

## Coverage and uncertainty

| Area | Status | Inspected evidence | Missing/excluded |
| --- | --- | --- | --- |
| Entry, registry, catalog | inspected | `__init__.py`, `nodes/__init__.py`, `registry.py`, `catalog.py`, all group `__init__` | gizmo runtime behaviour |
| color/, hdr/, OCIO | partial | module headers, key symbols, `vae.py` targeted reads | `hdr/color.py`, `processing.py`, `panorama.py` bodies |
| io/, core/, delivery | partial | read/write dispatch, EXR, video, delivery endpoint | DPX and OIIO edge paths; `core/logging.py` internals |
| generate, model, sampler | partial | loader, detect, cache, sampler main path | `prompt.py`, `regional.py`, `resolution.py`, `denoise.py`, `energy.py` internals |
| vfx, upscale, video, ai, pipeline/audio, studio | uninspected beyond registration | node keys and imports only | behaviour, models, contracts |
| JS frontend | partial | extension registry, renderer selection, data flow | most of the 23k-line `radiance_viewer.js` |
| tests, CI, release | inspected (structure) | conftest, CI YAML, pyproject, tools | test bodies; no tests executed |
| Excluded | excluded | none | `js/vendor/`, `ACES/config.ocio` (only the header was read), `*.png`, `workflows/*.json` (only node types were grepped), `scripts/training/*` (summary only) |

## Confirmed documentation drift

See [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md#documentation-drift). Each item there is a
contradiction between a doc or comment and the source.
