<!-- project-mapper:generated; verify against source before editing code -->
# Radiance: start here

Snapshot: 2026-10-05; source revision `7376f9e` on `version-4-Beta` (no source changes since).
Coverage: full inventory and static source analysis. Every node group and subsystem has a note.
Nothing was run, imported, or tested. This map is a navigation aid, not a substitute for the source.

## Evidence labels

| Label | Meaning |
| --- | --- |
| **Implemented** | Seen directly in code or config at the cited location. "(mapper)" means the map author re-read it. "(trace)" means it came from a read-only sub-agent trace and was not re-read. "(review)" means it was re-read for [CODE_REVIEW.md](CODE_REVIEW.md) |
| **Inferred** | Plausible but not fully traced. The note says what would verify it |
| **Unknown** | Not enough evidence. The note says what to inspect |
| **Proposed** | A suggestion, not current behaviour. Kept only in [OPEN_QUESTIONS § Proposed](OPEN_QUESTIONS.md#proposed-not-implemented) |

## Purpose and execution

Radiance (v3.5.4) is a ComfyUI custom-node package for HDR, colour management, VFX, review,
and delivery. It publishes 156 nodes and ships a browser viewer and dashboards. ComfyUI imports
the repo root as the `radiance` package (Implemented, mapper):

- `__init__.py:38` imports `.nodes.registry`, which runs `nodes/__init__.py` **first**. That loads all 10 groups in `nodes/catalog.py:NODE_GROUPS`, plus gizmos and branding.
- Only after that does `__init__.py` set up logging, the environment, dependency checks, and OCIO. Group code that reads `$OCIO` or Radiance env defaults at import time sees them unset.
- It then exports `NODE_CLASS_MAPPINGS` and serves `WEB_DIRECTORY = ./js`.

There is no standalone app, CLI, PHP backend, or database. The HTTP API is aiohttp routes on
ComfyUI's `PromptServer`. Three optional sockets start only when their node runs: the preview
`HTTPServer`, the DCC bridge, and NDI.

## Read next, only as needed

| Task | Note | Source starting point |
| --- | --- | --- |
| Add or rename a node, or a node is missing | [registry](subsystems/registry.md) | `nodes/catalog.py`, `nodes/<group>/__init__.py` |
| Colour maths, OCIO, ACES, log curves | [color-hdr](subsystems/color-hdr.md) | `color/ops.py`, `color/encodings.py`, `color/ocio_setup.py` |
| HDR VAE encode/decode, log profiles | [hdr-vae](subsystems/hdr-vae.md) | `hdr/vae.py`, `hdr/decode_meta.py` |
| Read/Write, EXR, video files, delivery | [io-delivery](subsystems/io-delivery.md) | `io/reader.py:read_frames`, `io/writer.py:write_frames` |
| Loader, sampler, model detection, downloads | [generate](subsystems/generate.md) | `loader_utils.py`, `nodes/generate/sampler.py` |
| Depth, flow, optics, multipass, relight | [vfx-multipass](subsystems/vfx-multipass.md) | `nodes/vfx/`, `nodes/vfx/multipass/estimate.py` |
| Upscale, QC, policy, review utilities | [upscale-qc-monitor](subsystems/upscale-qc-monitor.md) | `nodes/upscale/upscale.py`, `nodes/color/qc.py` |
| Video T2V/I2V, temporal windows, scene cut, audio | [video-temporal](subsystems/video-temporal.md) | `nodes/video/t2v.py`, `sampler_utils.py:plan_temporal_windows` |
| Nuke, Resolve, DCC bridge, NDI, gizmo runtime | [dcc-pipeline](subsystems/dcc-pipeline.md) | `nodes/pipeline/studio_integrations.py`, `nodes/gizmo.py` |
| Viewer, dashboards, routes, workspace storage | [frontend-routes](subsystems/frontend-routes.md) | `nodes/monitor/viewer.py`, `js/radiance_viewer.js`, `nodes/pipeline/workspace.py` |
| Tests, CI, release | [validation](subsystems/validation.md) | `tests/conftest.py`, `.github/workflows/ci.yml` |

## Contracts that must not change accidentally

- **IMAGE** is float32 `(B,H,W,C)` with C in {1,3,4}. Alpha is straight. MASK is `(B,H,W)`. The working space defaults to scene-linear Rec.709 and is unbounded (`core/tensor/alpha.py`, `color/encodings.py:WORKING_SPACES`). Implemented.
- **Node keys** in `NODE_CLASS_MAPPINGS` are the saved-workflow API. The floor is `config/constants.py:EXPECTED_MIN_NODE_COUNT = 156`, checked against `tests/node_keys_snapshot.json`. Implemented.
- **`radiance_meta` on HDR latents** carries a fingerprint. The decoder drops the HDR keys once a sampler changes the latent (`hdr/decode_meta.py:verify_radiance_meta`). Implemented.
- **`.rhdr` sidecar**: `RHDR`, then w/h/c/flags as uint16, then zlib data. Flag 0 means fp16 (always clamped to ±65504), 1 means fp32. One encoder, `core/rhdr.py`, used by `nodes/monitor/viewer.py` (frames and zdepth) and `hdr/vae.py:_save_rhdr`. One reader, `js/radiance_viewer.js`. Change the two together; `tests/test_rhdr_writers.py` pins what each writer produces. Implemented (fix pass).
- **`RADIANCE_PASSES`**: depth is metric with far = larger, normals are +Z toward the camera, motion is +y up. Other VFX nodes use different conventions; see [vfx-multipass](subsystems/vfx-multipass.md#contracts). Implemented.
- **Branding overwrites every node's `CATEGORY`** (`nodes/branding.py:apply_radiance_branding`). Implemented.
- **Downloads** are allowed by default; `RADIANCE_ALLOW_DOWNLOADS=0` turns most of them off. The Loader and the built-in upscale tiers use `core/model_fetch.fetch`, which requires a SHA-256. Other live paths pin a revision but don't check a hash, or have no consent gate at all. See OPEN_QUESTIONS B17 for the list. Implemented (mapper).

## Validation (from CI and CONTRIBUTING; not run during mapping)

```
pip install -e ".[test]"
python -m pytest tests/ --tb=short --timeout=60 -q   # CI light lane, no torch
python -m pytest tests/ --tb=short --timeout=120 -q  # full lane: torch, OpenEXR, OCIO, ffmpeg
node --test js/tests/*.test.mjs                      # or: npm test
ruff check --isolated --select E9,F82,F63,F7 --exclude tests,scripts .
```

## Map navigation

[Architecture and indexes](ARCHITECTURE.md) | [Workflows](WORKFLOWS.md) |
[Open questions, drift, bugs, proposals](OPEN_QUESTIONS.md) | [Code review](CODE_REVIEW.md) | [Snapshot](MAP_MANIFEST.json) |
[Mind map](diagrams/project-mindmap.mmd) | [Lifecycle flow](diagrams/runtime-flow.mmd)

Before changing code, check freshness against `MAP_MANIFEST.json` and read the cited source.
`docs/DEVELOPMENT.md` is the human development log. `KNOWN_ISSUES.md` lists accepted limitations.
