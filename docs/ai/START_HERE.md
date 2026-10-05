<!-- project-mapper:generated; verify against source before editing code -->
# Radiance: start here

Snapshot: 2026-10-05T07:01Z; revision `7376f9e` on `version-4-Beta`; worktree clean.
Coverage: full inventory, static source analysis only. Nothing was run, imported, or tested.
This map is a navigation aid, not a substitute for the source.

## Purpose and execution

Radiance (v3.5.3) is a ComfyUI custom-node package for HDR, colour management,
VFX, review, and delivery. It publishes 156 nodes and ships a browser viewer and
dashboards. ComfyUI imports the repo root as the `radiance` package:

- `__init__.py` sets up the environment and configures OCIO.
- It then loads `radiance.nodes`, which imports the 10 groups listed in `nodes/catalog.py:NODE_GROUPS`.
- It exports `NODE_CLASS_MAPPINGS` and serves `WEB_DIRECTORY = ./js`.

There is no standalone app, CLI, or PHP backend. The HTTP API is aiohttp routes on
ComfyUI's `PromptServer`.

## Read next, only as needed

| Task | Component note | Source starting point |
| --- | --- | --- |
| Add or rename a node, or a node is missing | [registry](subsystems/registry.md) | `nodes/catalog.py:NODE_GROUPS`, `nodes/<group>/__init__.py` |
| Colour maths, OCIO, ACES, log curves | [color-hdr](subsystems/color-hdr.md) | `color/ops.py`, `color/encodings.py`, `color/ocio_setup.py` |
| HDR VAE encode/decode, log profiles | [hdr-vae](subsystems/hdr-vae.md) | `hdr/vae.py:RadianceVAE4KEncode/RadianceVAE4KDecode` |
| Read/Write nodes, EXR, video, delivery | [io-delivery](subsystems/io-delivery.md) | `io/reader.py:read_frames`, `io/writer.py:write_frames` |
| Loader, sampler, model detection, downloads | [generate](subsystems/generate.md) | `loader_utils.py`, `nodes/generate/sampler.py:RadianceSamplerPro` |
| Viewer, dashboards, HTTP routes | [frontend-routes](subsystems/frontend-routes.md) | `nodes/monitor/viewer.py:RadianceViewer`, `js/radiance_viewer.js` |
| Tests, CI, release | [validation](subsystems/validation.md) | `tests/conftest.py`, `.github/workflows/ci.yml` |

## Contracts that must not change accidentally

- **IMAGE** is float32 `(B,H,W,C)` with C in {1,3,4}. Alpha is straight. MASK is `(B,H,W)`. The working space defaults to scene-linear Rec.709 and is unbounded (`core/tensor/alpha.py`, `color/encodings.py:WORKING_SPACES`).
- **Node keys** in `NODE_CLASS_MAPPINGS` are the saved-workflow API. The count floor is `config/constants.py:EXPECTED_MIN_NODE_COUNT = 156`, checked against `tests/node_keys_snapshot.json`.
- **`radiance_meta` on HDR latents** carries a fingerprint. The decoder drops the HDR keys once a sampler changes the latent (`hdr/decode_meta.py:verify_radiance_meta`).
- **`.rhdr` viewer sidecar**: a 12-byte header (`RHDR`, then w/h/c/flags as uint16) followed by zlib data. Flag 0 means fp16, 1 means fp32. Writer is `nodes/monitor/viewer.py`, reader is `js/radiance_viewer.js`.
- **Branding overwrites every node's `CATEGORY`** to `FXTD STUDIOS/Radiance/<section>` (`nodes/branding.py:apply_radiance_branding`).
- **Downloads** go through `core/model_fetch.fetch`, which needs a pinned SHA-256. They are allowed by default. `RADIANCE_ALLOW_DOWNLOADS=0` turns them off.

## Validation (from CI and CONTRIBUTING; none of these were run during mapping)

```
pip install -e ".[test]"
python -m pytest tests/ --tb=short --timeout=60 -q   # CI light lane, no torch
python -m pytest tests/ --tb=short --timeout=120 -q  # full lane: torch, OpenEXR, OCIO, ffmpeg
node --test js/tests/*.test.mjs                      # or: npm test
ruff check --isolated --select E9,F82,F63,F7 --exclude tests,scripts .
```

## Map navigation

[Architecture](ARCHITECTURE.md) | [Workflows](WORKFLOWS.md) |
[Unknowns and concerns](OPEN_QUESTIONS.md) | [Snapshot](MAP_MANIFEST.json) |
[Mind map](diagrams/project-mindmap.mmd) | [Runtime flow](diagrams/runtime-flow.mmd)

Before changing code, check freshness against `MAP_MANIFEST.json` and read the relevant
implementation and tests. Don't load the full map for an unrelated task.
`docs/DEVELOPMENT.md` is the human development log. `KNOWN_ISSUES.md` lists accepted limitations.
