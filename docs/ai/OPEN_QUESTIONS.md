<!-- project-mapper:generated -->
# Open questions, documentation drift, and concerns

Snapshot: `7376f9e`. Every item is labelled.
- **Confirmed** means it was re-read in the source during mapping.
- **Reported** means a mapping sub-agent traced it and it was not re-read.

These items are not a security audit or a test result.

## Documentation drift

| # | Claim | Source evidence | Status |
| --- | --- | --- | --- |
| D1 | README: "147 visible nodes" | 156 registry keys (`config/constants.py:EXPECTED_MIN_NODE_COUNT`, `nodes/branding.py:NODE_SECTIONS`) | Confirmed mismatch; the intended meaning of "visible" is unknown |
| D2 | `CONTRIBUTING.md:133`: CI skips GPU tests with `-m "not gpu"` | `ci.yml` never passes that flag | Confirmed |
| D3 | `CONTRIBUTING.md:154`: version is read from pyproject via `importlib.metadata` | `config/constants.py:5` hard-codes `VERSION` | Confirmed |
| D4 | `CONTRIBUTING.md:16`: `--timeout=30` | CI uses 60 and 120 | Confirmed |
| D5 | `pyproject.toml:179-180` package-data `workflows/TEMPLATES/*` | The folder doesn't exist | Confirmed |
| D6 | Write UI note: "DNxHR is written into an MXF container" (`nodes/io/write.py:~1576`) | Writer uses `.mov` (`io/writer.py:~907`) | Confirmed |
| D7 | `hdr/vae.py:~158`: "same names as model/detect.py's latent formats" | Labels differ (for example `sd3_8ch` vs `sd3_16ch`; 128ch is `ltx_128ch` for Flux.2) | Confirmed |
| D8 | (withdrawn) `color/ocio_setup.py:17` "Nothing is downloaded" is accurate for `configure_ocio()`. The separate ACESConfigManager node does download; see B17 | `hdr/ocio.py:597, 715, 719` | Not drift |
| D9 | `core/exr.py` docstring promises an OpenCV fallback for EXR reads | No such fallback in the probe order | Reported |
| D10 | Viewer comment ~L3713: "WebGPU-preferred" | WebGPU is opt-in through localStorage | Reported |
| D11 | `tools/gpu_acceptance.py:110-115` lists `RADIANCE_GRADE_INFO`, `RADIANCE_CDL`, … | Those types aren't used in `nodes/` | Reported |
| D12 | `tools/check_mojibake.py` exists as a release check | Not run by CI or any test | Reported |

## Suspected bugs (evidence-backed, not reproduced)

| # | Concern | Evidence | Status |
| --- | --- | --- | --- |
| B1 | `.rhdr` export path-traversal guard is dead: `from .path_utils import safe_join` inside `hdr/` points to a module that doesn't exist, so it always falls back to `os.path.join` | `hdr/vae.py:~2698`; `hdr/path_utils.py` is absent | Confirmed |
| B2 | Flux.2 models detected as `"flux"`: dict order puts `"Flux"` before `"Flux2"` and the match is a substring test | `sampler_utils.py:~597,605` | Confirmed (hidden when `model_meta` is connected) |
| B3 | `_encode_sdr_reference` does `vae.encode(ref)["samples"]`. `comfy.sd.VAE.encode` returns a tensor | `nodes/generate/sampler.py:~1504` | Confirmed code; the runtime failure is inferred |
| B4 | Delivery adds two versions: the handler builds `prefix_v02`, then `write_frames` appends `_v0001` (`version` defaults to 1) | `delivery/handler.py:~470,876`; `io/writer.py:1266,1323` | Confirmed |
| B5 | Image sequences always report fps 24.0 | `io/reader.py:~769` | Confirmed |
| B6 | No HDR10 static metadata (mastering display, MaxCLL) on PQ/HLG video. PQ can go into 8-bit H.264 | No matches for `master-display`/`max-cll`/`x265-params` in non-test code | Confirmed absence |
| B7 | Offload `sequential` permanently sets global `vram_state = LOW_VRAM` | `loader_utils.py:~240` | Confirmed |
| B8 | `core/errors.py` (`RadianceError`, `handle_node_errors`) isn't used by production code | grep finds only `exceptions.py` and `core/__init__` | Confirmed |
| B9 | Without ffprobe every video read fails (imageio-ffmpeg ships no ffprobe) | `core/video.py:~656-675` | Reported |
| B10 | `raw=True` on video still tag-decodes the colour but labels it raw | `io/reader.py:~1274, 829` | Reported |
| B11 | OIIO-only extensions other than `.dpx` fall through to Pillow | `io/reader.py:_read_image`, `core/formats.py:~131` | Reported |
| B12 | A non-zero ffmpeg exit leaves a partial video file | `io/writer.py:~1019` | Reported |
| B13 | `image/upscale._download_model` doesn't pass `legacy_offline_env`, so `RADIANCE_UPSCALE_OFFLINE` is ignored there | `image/upscale.py:~1916` | Reported |
| B14 | `model/pixel_download.py` bypasses `fetch()`. It is unclear whether the hash is checked after the write | `model/pixel_download.py:~83` | Reported / unknown |
| B15 | `validate_runtime_dependencies` return value is ignored | `__init__.py:132` | Confirmed |
| B17 | Downloads outside `core/model_fetch.fetch` (no enforced SHA-256 pin): multipass `hf_hub_download`/`urlretrieve`, a `torch.hub.load` of DSINE that executes remote code, `estimate_models.hf_hub_download`, RUDRA `pixel_download`, ACESConfigManager. Consent gating per path not verified | `nodes/vfx/multipass/core.py:290, 321, 556`; `estimate_models.py:200`; `model/pixel_download.py:83`; `hdr/ocio.py:715` | Confirmed call sites; gating unknown |
| B16 | `color/pipeline.apply_input_transform` applies only the curve for log spaces (no camera gamut matrix). It's used for nit estimation | `color/pipeline.py:~55`, `nodes/generate/engine.py:~705` | Reported |

## Design and ownership questions

- **User data lives inside the package checkout.** `WORKFLOW_DIR = <repo>/workflows`
  (`nodes/pipeline/workspace.py:218`) and `GIZMOS_DIR = <repo>/gizmos`, created at import
  (`nodes/gizmo.py:48-49`). What happens on `git pull` or a ComfyUI Manager reinstall is unknown.
- **No auth on `/radiance/*` routes.** This is acceptable only while ComfyUI binds to localhost.
  `/assets/upload` silently overwrites existing files. `/deliver` containment uses `abspath`,
  not `realpath`.
- **Five separate OCIO config resolvers** (see [color-hdr](subsystems/color-hdr.md)). There is
  no single owner.
- **Logger `propagate=False`.** It's unknown whether ComfyUI's log capture or UI sees Radiance logs.
- **Raw `fetch('/radiance/...')` in the JS** instead of `api.fetchApi`. This may break when
  ComfyUI is served under a sub-path. Inferred.

## Not inspected (next bounded checks)

- VFX and multipass internals (`nodes/vfx/multipass/core.py`, `estimate*.py`): model downloads,
  pass contracts.
- `nodes/video/t2v.py`, `nodes/video/hdr.py`: the video sampling path and the frame and fps contract.
- `nodes/pipeline/dcc.py` and `studio_integrations.py`: the bridge protocol and Resolve worker lifecycle.
- `nodes/upscale/upscale.py` (3,121 lines): tiling and cache behaviour.
- The body of `js/radiance_viewer.js` past the data-flow entry points.
