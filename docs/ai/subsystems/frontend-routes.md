<!-- project-mapper:generated -->
# Frontend, viewer data flow, and HTTP routes

ID: frontend (plus viewer, workspace). Snapshot: `7376f9e`. Coverage: partial. `js/radiance_viewer.js` (23k lines) was only grepped and read in bounded slices. Labels: see [START_HERE](../START_HERE.md#evidence-labels).

## Responsibility and boundaries

ComfyUI auto-loads every `.js` file in `js/` (`WEB_DIRECTORY`). Extensions register with
`app.registerExtension`. The Python side serves aiohttp routes on `PromptServer.instance.routes`.
None of the routes has authentication.

## Implementation index

| Source/symbol | Role | Label |
| --- | --- | --- |
| `js/radiance_viewer.js` (ext `FXTD.RadianceViewer`, hook ~:21679, `onExecuted` ~:21863) | Viewer UI, grading, scopes, OCIO menu, sequence paging, delivery dialog | Implemented |
| `js/radiance_webgl.js` | Default renderer (`RadianceWebGLRenderer`). Falls back to a 2D canvas | Implemented |
| `js/radiance_webgpu.js` | Opt-in renderer (`localStorage.radiance_prefer_webgpu === '1'`). Has no masks, qualifiers, or OCIO | Implemented |
| `js/radiance_renderer.js`, `js/radiance_grade.js` | Renderer base class. One grade definition emitted as JS, GLSL, and WGSL | Implemented |
| `js/radiance_ocio.js` and `js/vendor/ocio/` | OCIO 2.5 WASM (~4.7 MB, lazy). Its GLSL is spliced into the WebGL shader | Implemented |
| `js/radiance_workspace.js`, `workspace_dashboard.html`, `project_manager_dashboard.*`, `assets_dashboard.*` | `.rad` pack/unpack and dashboard iframes | Implemented |
| `nodes/monitor/viewer.py:RadianceViewer.view` (:383) | Writes `.rhdr`, EXR, and PNG files to the temp dir and fills the viewer cache. Returns a `ui` dict | Implemented |
| `nodes/monitor/viewer.py:_radiance_route_once` (~:1441) | Route idempotency guard | Implemented |
| `nodes/pipeline/workspace.py` | Workflow, project, and asset REST API. `WORKFLOW_DIR = <repo>/workflows` (:218) | Implemented |
| `nodes/monitor/realtime.py` | A separate `http.server` on 127.0.0.1 (frames, health). Not PromptServer | Implemented (trace) |

## Routes (all under `/radiance/`)

| Area | Routes | Handler file |
| --- | --- | --- |
| Viewer | `GET progress` | `nodes/monitor/viewer.py` |
| Delivery | `POST deliver` | `delivery/handler.py` |
| OCIO | `GET ocio/config`, `GET ocio/displays`, `POST ocio/load`, `POST ocio/bake` | `radiance_ocio.py` |
| Media | `GET media/layers`, `info`, `preview`, `poster`, `resolve_write` | `nodes/io/write.py` |
| Prompt | `GET prompt/presets` | `nodes/generate/prompt.py` |
| Gizmos | `POST gizmos/create`, `GET gizmos/list` | `nodes/gizmo.py` |
| Workflows | `pack`, `unpack`, `save`, `list`, `get`, `delete`, `restore`, `history`, `preview` | `nodes/pipeline/workspace.py` |
| Projects | `projects`, `projects/recent`, `projects/dashboard`, `{id}/versions`, `outputs`, `notes`, `save-version`, `export-package`, `shots/{shot}/status` | `nodes/pipeline/workspace.py` |
| Assets | `assets`, `assets/thumb`, `assets/bins`, `assets/bins/{id}`, `assets/upload` | `nodes/pipeline/workspace.py` |

The exact line for each route is listed in the agent trace that this summary was built from.
Run `grep -n "routes\.\(get\|post\)\|_route(" <file>` to find a route again.

## Contracts

- **Python to JS data:** the ComfyUI `ui` output (`radiance_images`, `source_encoding`,
  `source_colorspace`, `fps`, `instance_id`, …), and files fetched through
  `/view?type=temp`. There are **no `send_sync` events**.
- **`.rhdr` format:** a header of `"RHDR"` followed by w, h, channels, and flags as uint16, then
  stored zlib data. Flag 0 means fp16 (RGBA16F) and flag 1 means fp32. Frames are padded to RGBA
  and decoded in Web Workers. Encoded by `core/rhdr.py` (fix pass); the zdepth `.rhdr` is written but
  the JS loads the zdepth PNG.
- **Where colour happens:** the realtime view transform and grade run in shaders. Python only
  tags the source encoding and builds a display-referred PNG preview
  (`color/display_preview.py`).
- **Path guards:**
  - Workspace: `_resolve_safe_path` under `WORKFLOW_DIR`, `.rad` only, 50 MB and 256 zip entries.
  - `/media/*`: must be inside input, output, temp, or models, or under `RADIANCE_READ_ROOTS`.
  - `/ocio/load`: must be inside `RADIANCE_OCIO_ROOTS` or `$OCIO`.
  - `/deliver`: must be inside the output dir.

## Persistence

- **Workflows:** `<repo>/workflows/**/*.rad` with a `.rad.json` sidecar, and `.versions/*.vN.rad` (50 max).
- **Projects:** `.shot_status.json` and `.review_notes.json`.
- **Assets:** `_assets_bins.json` and `.asset_thumbs/`. Uploads go to `<ComfyUI input>/radiance_assets/`.
- **Gizmos:** `<repo>/gizmos/`.

User data lives inside the package checkout. See OPEN_QUESTIONS.

Workspace details below are Implemented unless labelled; all come from the agent trace of `nodes/pipeline/workspace.py`.

- **`.rad` formats:**
  - v1: plain JSON.
  - v2: `RAD!` header, meta JSON, a zlib graph, and a trailing SHA-256 (`_pack_rad_v2` :554).
  - v3: `RADZ` plus a ZIP holding `manifest.json`, `workflow.json`, `assets/`, and an optional `preview.png` (`_pack_rad_v3` :415).
  - Unpack limits are 256 entries and 50 MB of *declared* size. v2 `zlib.decompress` has no bound (Inferred).
- **Node run:** `RadianceProjectManager` (an OUTPUT_NODE) packs the hidden `PROMPT` as `{filename}_{artist}_v{NNNN}.rad`. That `PROMPT` is ComfyUI API format, which has no `nodes` key, so the recorded `node_count` is probably 0. The tests use UI-format graphs (Inferred).
- **Projects** are derived by `rglob("*.rad")`. The project is `metadata.project`, else `show`, else the first folder, else `GENERAL`. Shot and version come from the regexes `\bsh[-_ ]?(\d{2,5})\b` and `\bv(\d{3,4})\b`. Names with `_` separators probably fail the `\b` match (Inferred).
- **Shot status:** `workflows/<project>/.shot_status.json`, values WIP, Review, Approved, Retake, Final. `project.name` comes from file metadata and is joined without a containment check (Inferred risk).
- **Asset bins:** `_assets_bins.json`. Asset ids are `sha1(path)[:16]`. Scanning covers input and output, capped at 600.
- **Export package:** an in-memory ZIP of the `.rad` files plus a manifest. It includes no versions, outputs, or assets.
- **Concurrency:** no locks and no temp-and-rename on any workspace write.
- **Dashboard user:** `/radiance/projects/dashboard` returns a hard-coded user name (`workspace.py:960`).

## Tests

35 `js/tests/*.test.mjs` files run under `node --test`. The GPU/DOM tests use Playwright
Chromium and are skipped when it's missing. `grade_wgsl_gpu` uses Deno WebGPU. On the Python
side: `tests/test_viewer_*`, `test_workspace_api.py`, `test_gizmo_storage.py`. Not run.
