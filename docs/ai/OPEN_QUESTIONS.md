<!-- project-mapper:generated -->
# Open questions, documentation drift, suspected bugs, and proposals

Snapshot: `7376f9e`. Labels are defined in [START_HERE](START_HERE.md#evidence-labels).
**Implemented (mapper)** means the map author re-read the code. **Implemented (trace)** means it
comes from a read-only sub-agent trace and was not re-read. No item here is a test result or a
security audit, and nothing was reproduced at runtime.

## Documentation drift

These are confirmed contradictions between a document or comment and the code.

| # | Claim | Code evidence | Label |
| --- | --- | --- | --- |
| D1 | README: "147 visible nodes" | 156 registry keys (`config/constants.py:EXPECTED_MIN_NODE_COUNT`, `nodes/branding.py:NODE_SECTIONS`) | Implemented (mapper). What "visible" means is Unknown |
| D2 | `CONTRIBUTING.md:133`: CI skips GPU tests with `-m "not gpu"` | `ci.yml` never passes that flag | Implemented (mapper) |
| D3 | `CONTRIBUTING.md:154`: version read via `importlib.metadata` | `config/constants.py:5` hard-codes `VERSION` | Implemented (mapper) |
| D4 | `CONTRIBUTING.md:16`: `--timeout=30` | CI uses 60 and 120 | Implemented (mapper) |
| D5 | `pyproject.toml:179-180` package-data `workflows/TEMPLATES/*` | The folder doesn't exist | Implemented (mapper) |
| D6 | Write UI: "DNxHR is written into an MXF container" (`nodes/io/write.py:~1576`) | Writer uses `.mov` (`io/writer.py:~907`) | Implemented (mapper) |
| D7 | `hdr/vae.py:~158`: "same names as model/detect.py's latent formats" | Labels differ (`sd3_8ch` vs `sd3_16ch`; 128ch is `ltx_128ch` even for Flux.2) | Implemented (mapper) |
| D9 | `core/exr.py` docstring promises an OpenCV EXR read fallback | Not in the probe order | Implemented (trace) |
| D10 | Viewer comment ~L3713: "WebGPU-preferred" | WebGPU is opt-in through localStorage | Implemented (trace) |
| D11 | `tools/gpu_acceptance.py:110-115` lists `RADIANCE_GRADE_INFO`, `RADIANCE_CDL`, … | Those types aren't used in `nodes/` | Implemented (trace) |
| D12 | `tools/check_mojibake.py` presented as a release check | Not run by CI or any test | Implemented (trace) |
| D13 | `nodes/video/character.py` header lists `RadianceCharacterAnchor`, Checker, and Gallery nodes | None are defined; the module exports an empty mapping | Implemented (mapper) |
| D14 | `nodes/vfx/depth.py:60` docstring: "Cache stores models on CPU" | The model is cached on the target device, fp16 on CUDA (:122-123) | Implemented (trace) |
| D15 | LensDistortion tooltip: "exact closed-form inverse" (`optics.py:106`) | Divides by the polynomial at the output radius, so it is an approximation (:144-150) | Implemented (trace). The numeric error is Inferred |
| D16 | Upscale Tiler tooltip: "output clamped to 0-1" | The Reinhard HDR path keeps values above 1 | Implemented (trace) |
| D17 | `blend_mode` offers `laplacian_pyramid` | Falls back to the Gaussian feather (`nodes/upscale/upscale.py:503, 520`) | Implemented (mapper) |
| D18 | `scene_cut.py` tooltip calls the edge method "Sobel" | Uses plain finite differences | Implemented (trace) |
| D19 | `start_nuke_server.py:320-321` comment implies HMAC fixed replay | The HMAC covers only the command, with no nonce or timestamp | Implemented (trace) |

D8 was withdrawn in the previous revision. The `ocio_setup.py` docstring is accurate.

## Suspected bugs

These are backed by code evidence but not reproduced. The consequence is Inferred unless stated otherwise.

| # | Concern | Evidence | Label |
| --- | --- | --- | --- |
| B1 | The `.rhdr` export path-traversal guard is dead. It imports `safe_join` from a `hdr/path_utils` that doesn't exist, so it always uses `os.path.join` | `hdr/vae.py:~2698` | Implemented (mapper) |
| B2 | Flux.2 is detected as `"flux"`: the substring test checks `"Flux"` before `"Flux2"` | `sampler_utils.py:~597, 605` | Implemented (mapper). Hidden when `model_meta` is connected |
| B3 | `_encode_sdr_reference` indexes `["samples"]` on what `comfy.sd.VAE.encode` returns, which is a plain tensor | `nodes/generate/sampler.py:~1504` | Code Implemented (mapper). Failure Inferred |
| B4 | Delivery adds two version suffixes (`_v02`, then `_v0001`) | `delivery/handler.py:~470, 876`; `io/writer.py:1266, 1323` | Implemented (mapper) |
| B5 | Image sequences always report fps 24.0 | `io/reader.py:~769` | Implemented (mapper) |
| B6 | No HDR10 mastering/MaxCLL metadata on PQ/HLG video. PQ can go into 8-bit H.264 | grep finds no such code outside tests | Implemented (mapper, by absence) |
| B7 | Offload `sequential` permanently sets global `vram_state = LOW_VRAM` | `loader_utils.py:~240` | Implemented (mapper) |
| B8 | `core/errors.py` is unused by production code | grep | Implemented (mapper) |
| B9 | Without ffprobe every video read fails | `core/video.py:~656-675` | Implemented (trace) |
| B10 | `raw=True` on video still decodes using the colour tags | `io/reader.py:~1274, 829` | Implemented (trace) |
| B11 | OIIO-only extensions other than `.dpx` fall through to Pillow | `io/reader.py:_read_image` | Implemented (trace) |
| B12 | A non-zero ffmpeg exit leaves a partial video file | `io/writer.py:~1019` | Implemented (trace) |
| B13 | `RadianceAIUpscale._download_model` ignores `RADIANCE_UPSCALE_OFFLINE` | `image/upscale.py:~1916` | Implemented (trace) |
| B15 | The `validate_runtime_dependencies` result is ignored | `__init__.py:132` | Implemented (mapper) |
| B16 | `color/pipeline.apply_input_transform` applies only the log curve, with no gamut matrix | `color/pipeline.py:~55` | Implemented (trace) |
| B17 | **Download integrity varies.** `fetch()` requires a SHA-256 (Loader, upscale tiers). RUDRA `pixel_download` is consent-gated and SHA-256 checked (mapper). MoGe-2 checks size only, and its `MOGE_SHA256` is read only by a test and `tools/pin_models.py` (trace). Marigold checks file presence only (trace). Depth Anything, SD-x4, and character CLIP use a pinned `from_pretrained` revision (trace). **ACESConfigManager** downloads from GitHub with no consent check in `hdr/ocio.py` (mapper). The DSINE `torch.hub.load` (remote code) is reachable only through `_legacy_extract`, which only a test calls (mapper) | cited files | Mixed, as marked |
| B18 | `AudioTranscribe` uploads audio to OpenAI and may download whisper weights with no consent gate | `nodes/pipeline/audio.py`; `core.consent` not imported | Implemented (trace) |
| B19 | `NukeConnector` treats a read timeout after a successful send as success | `tools/nuke_connector.py:176-186` | Implemented (mapper) |
| B20 | `NukeSend` and `DaVinciSend` don't reject path separators in `filename` | `studio_integrations.py:181-184, 365-366` | Code Implemented (mapper). Traversal Inferred |
| B21 | The DCC bridge has no authentication when `RADIANCE_ALLOW_REMOTE_BRIDGE` is set. Its threads are uncapped and `stop_server` is never called | `nodes/pipeline/dcc.py` | Implemented (trace) |
| B22 | Delivery's 2x upscale can silently ship bicubic, because `RadianceAIUpscale` returns bicubic on any exception | `delivery/handler.py:705`; `image/upscale.py:_fallback_upscale` | Implemented (mapper and trace). Silence Inferred |
| B23 | RGBA is broken on more review and upscale paths than KNOWN_ISSUES lists: FlipbookGIF, PreviewServer, spandrel Tier 2, FaceRestore alpha | `realtime.py`, `nodes/upscale/upscale.py:~908, 2710` | Inferred (trace) |
| B24 | Motion and depth conventions differ between VFX nodes: Estimate motion is +y up while OpticalFlow and MaskPropagator use +y down; Relight and Composite expect near = white | `estimate.py:652`; `motion.py:107`; `relight_comp.py` | Implemented (trace). The effect on chained nodes is Inferred |
| B25 | `T2V` and `VideoSampler` pass the noise tensor as the start latent too | `nodes/video/t2v.py:999-1001, 1176-1179` | Code Implemented (trace). Effect on EPS models Unknown (check `model_sampling.noise_scaling`) |
| B26 | `AudioTranscribe` keeps the original segment start on every split chunk | `nodes/pipeline/audio.py:transcribe` | Implemented (trace) |
| B27 | `ProjectManager` packs the API-format `PROMPT`, which has no `nodes` key, so `node_count` is probably 0. The shot/version regexes probably fail on `_`-separated names | `nodes/pipeline/workspace.py:135-184, 656-776` | Inferred (trace) |
| B29 | `_verify_or_report_sha256` is defined but never called; files already on disk are trusted when their size matches | `nodes/upscale/upscale.py:212`; `core/model_fetch.fetch` | Implemented (mapper) |
| B28 | `.shot_status.json` path is built from file metadata `project.name` without a containment check | `workspace.py:_shot_status_path` | Inferred (trace) |

The B14 question from the previous revision is resolved and folded into B17: `pixel_download` does verify SHA-256.

## Design and ownership questions (Unknown)

- **User data lives inside the package checkout** (`<repo>/workflows`, `<repo>/gizmos`). What happens on `git pull` or a ComfyUI Manager reinstall?
- **No auth on `/radiance/*` routes.** This is acceptable only while ComfyUI binds to localhost. `/assets/upload` overwrites existing files. `/deliver` containment uses `abspath`, not `realpath`.
- **Five OCIO config resolvers** with no single owner (see [color-hdr](subsystems/color-hdr.md#contracts)).
- **Logger `propagate=False`:** does ComfyUI's log capture see Radiance logs? And do group-import WARNINGs (emitted before `setup_radiance_logging`) reach the console formatted as intended?
- **Raw `fetch('/radiance/...')` in the JS:** does it break when ComfyUI is served under a sub-path? (Inferred.)
- **`image/upscale.py` sets `KMP_DUPLICATE_LIB_OK=TRUE` at import** for the whole ComfyUI process. Is that intended?
- **Gizmo executor** bypasses the ComfyUI executor and has no tests. Which node types are known to work inside a gizmo?

## Proposed (not implemented)

These are recommendations only. None of them describes current behaviour, and none has been
agreed with the maintainers.

- **P1. One download path.** Route MoGe, Marigold, and ACESConfigManager through `core/model_fetch.fetch` or an equivalent consent and SHA-256 check (addresses B17 and B18).
- **P2. One OCIO resolver.** Have every consumer ask `color/ocio_setup` for the active config.
- **P3. Move user data out of the package dir**, for example under ComfyUI's user directory.
- **P4. Shared convention constants** for depth polarity and motion-vector axis, and convert at `RADIANCE_PASSES` boundaries (addresses B24).
- **P5. Atomic media writes** (temp file and `os.replace`) for EXR, video, and workspace files (addresses B12 and the workspace concurrency gap).
- **P6. Test gaps:** RadianceAIUpscale, the gizmo executor, RGBA on review tools, and the shipped workflow JSONs executing end to end.

## Not inspected (next bounded checks)

- `hdr/color.py`, `hdr/processing.py`, `hdr/panorama.py`, and `hdr/recovery.py` bodies.
- `nodes/generate/prompt.py`, `regional.py`, `resolution.py`, `denoise.py`, and `energy.py` internals.
- The `js/radiance_viewer.js` body past the data-flow entry points.
- Third-party download behaviour (SeedVR2, facexlib, whisper) and spandrel dtype defaults.
