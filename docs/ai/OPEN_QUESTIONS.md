<!-- project-mapper:generated -->
# Open questions, documentation drift, suspected bugs, and proposals

Snapshot: `7376f9e`. Labels are defined in [START_HERE](START_HERE.md#evidence-labels).
**Implemented (mapper)** means the map author re-read the code. **Implemented (trace)** means it
comes from a read-only sub-agent trace and was not re-read. **Implemented (review)** means it was
re-read during the code review in [CODE_REVIEW.md](CODE_REVIEW.md). No item here is a test result or a
security audit, and nothing was reproduced at runtime.

Severity and fix suggestions live in [CODE_REVIEW.md](CODE_REVIEW.md). Where an item has a review
finding, its ID is given in brackets, for example [P1-1]. Items without one were not re-reviewed.

**Fix pass (version-4-Beta, after `b645400`):** items marked **Fixed** were fixed with a regression
test that failed on the old code; the commit is named in the row. See
[CODE_REVIEW § Fix status](CODE_REVIEW.md#fix-status). Line numbers in fixed rows describe the code
before the fix.

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
| B1 | **Fixed in `2af342b`.** The `.rhdr` export path-traversal guard is dead. It imports `safe_join` from a `hdr/path_utils` that doesn't exist, so it always uses `os.path.join`. **Not exploitable today:** both callers pass internal prefixes | `hdr/vae.py:~2698`; callers `:3038`, `:3387` | Implemented (mapper) [P3-4] |
| B2 | **Fixed in `1766a81`.** Flux.2 is detected as `"flux"`: the substring test checks `"Flux"` before `"Flux2"` | `sampler_utils.py:~597, 605` | Implemented (mapper). Hidden when `model_meta` is connected [P2-1] |
| B3 | **Fixed in `3314513`, `55afb69`.** `_encode_sdr_reference` indexes `["samples"]` on what `comfy.sd.VAE.encode` returns, which is a plain tensor. The test double `_FakeVAE` returns a dict, so tests miss it | `nodes/generate/sampler.py:~1504`; `tests/test_sdr_conditioning.py:354-367` | Code Implemented (mapper). Runtime failure Inferred [P1-1] |
| B4 | Delivery added two version suffixes (`_v02`, then `_v0001`). Fixed in 4.0: one, `Shot_v0002` | `delivery/handler.py:get_next_version`; `io/writer.py:write_frames` | Fixed [P3-1] |
| B5 | Image sequences always reported fps 24.0 (fixed in 4.0: the first frame's EXR `framesPerSecond` or DPX/Cineon rate; info carries `fps` and `fps_source`) | `io/reader.py:~769` | Fixed in 4.0 |
| B6 | No HDR10 mastering/MaxCLL metadata on PQ/HLG video; PQ could go into 8-bit H.264 (fixed in 4.0: 8-bit PQ/HLG refused; libx265 PQ gets `hdr10=1`, P3-D65 1000-nit master-display and measured max-cll) | grep finds no such code outside tests | Fixed in 4.0 |
| B7 | **Still open:** the fix `6b5bd9c` was reverted in `33a30cb` (see CODE_REVIEW P2-4); the tooltip now says the effect lasts for the session. Offload `sequential` permanently sets global `vram_state = LOW_VRAM` | `loader_utils.py:~240` | Implemented (mapper) [P2-4] |
| B8 | `core/errors.py` is unused by production code | grep | Implemented (mapper) |
| B9 | Without ffprobe every video read failed (fixed in 4.0: `_probe_with_ffmpeg` parses ffmpeg's banner; frame count estimated) | `core/video.py:~656-675` | Fixed in 4.0 |
| B10 | `raw=True` on video still decoded using the colour tags (fixed in 4.0: raw skips the tag decode and OCIO override) | `io/reader.py:~1274, 829` | Fixed in 4.0 |
| B11 | OIIO-only extensions other than `.dpx` fell through to Pillow (fixed in 4.0: `_needs_oiio` / `_read_oiio`) | `io/reader.py:_read_image` | Fixed in 4.0 |
| B12 | **Fixed in `e4f084c`.** A non-zero ffmpeg exit leaves a partial video file. The file is removed after a timeout (`:1010`) or a mid-write exception such as a cancel (`:986`), but the non-zero-exit branch only raises (`:1015`). CODE_REVIEW.md had marked this fixed; that was wrong and has been corrected there | `io/writer.py:986, 1010, 1015` | Implemented (mapper) |
| B13 | `RadianceAIUpscale._download_model` ignored `RADIANCE_UPSCALE_OFFLINE` (fixed in 4.0) | `image/upscale.py:~1916` | Fixed in 4.0 |
| B15 | The `validate_runtime_dependencies` result was ignored (fixed in 4.0: `check_runtime_dependencies` warns, never raises) | `__init__.py:132` | Fixed in 4.0 |
| B16 | `color/pipeline.apply_input_transform` applied only the log curve, with no gamut matrix (fixed in 4.0: camera gamut to Rec.709 through `encodings.gamut_matrix`; `apply_output_transform` mirrors it) | `color/pipeline.py:~55` | Fixed in 4.0 |
| B17 | **Download integrity varies.** Since 4.0: MoGe-2 is checked against `MOGE_SHA256`; each Marigold file against the Hub's hash at the pinned commit; ACESConfigManager goes through `core/model_fetch.fetch` with a pinned SHA-256 and the consent check. `fetch()` requires a SHA-256 (Loader, upscale tiers). RUDRA `pixel_download` is consent-gated and SHA-256 checked. **Still open:** the multipass `_MODEL_REGISTRY` (Depth Anything V2 `.pth`, DSINE) downloads from `main` with no pinned hash (consent-gated; the digest is logged); pin it with `tools/pin_models.py` on a networked machine. Depth Anything, SD-x4 and character CLIP `from_pretrained` use a pinned revision. The DSINE `torch.hub.load` (remote code) is reachable only through `_legacy_extract`, which only a test calls | cited files | Mixed, as marked |
| B18 | `AudioTranscribe` uploads audio to OpenAI when that backend is chosen (or Auto with a key and no local whisper). Since 4.0 the whisper weight download (local package and CLI) asks for download consent first | `nodes/pipeline/audio.py:_require_whisper_consent` | Implemented (tests) |
| B19 | `NukeConnector` treated a read timeout after a successful send as success (fixed in 4.0: `UNCONFIRMED`, not OK) | `tools/nuke_connector.py:176-186` | Fixed in 4.0 |
| B20 | (withdrawn) `NukeSend` and `DaVinciSend` don't reject path separators in `filename`, but `nuke_folder` / `resolve_folder` are user widgets too, so a separator gives no reach the user doesn't already have | `studio_integrations.py:95-100, 298-303` | Not a defect |
| B21 | The DCC bridge had no authentication when `RADIANCE_ALLOW_REMOTE_BRIDGE` is set (fixed in 4.0: a remote `queue` must be signed with the shared DCC token, which is never sent; HTTP requests are refused). Since 4.0 connections are capped at 16 and `stop_server` runs at exit; a sequence export works | `nodes/pipeline/dcc.py:86-101, 117-149` | Implemented (review) [P2-6] |
| B22 | **Fixed in `b242fdf`.** Delivery's 2x upscale can silently ship bicubic, because `RadianceAIUpscale` returns bicubic on any exception | `delivery/handler.py:691-715`; `image/upscale.py:2241, 2377-2379, 2393, 2536-2545` | Implemented (review): the delivery warning only fires on an exception, which `upscale` never lets escape [P1-2] |
| B23 | RGBA was broken on FlipbookGIF, PreviewServer, spandrel Tier 2, FaceRestore alpha (fixed in 4.0: previews drop alpha, Tier 2 resizes it, Face Restore keeps it). An external `UPSCALE_MODEL` with RGBA is unchecked | `realtime.py`, `nodes/upscale/upscale.py:~908, 2710` | Fixed in 4.0 |
| B24 | Motion and depth conventions differ between VFX nodes: Estimate motion is +y up while OpticalFlow and MaskPropagator use +y down; Relight and Composite expect near = white | `estimate.py:652`; `motion.py:107`; `relight_comp.py` | Implemented (trace). The effect on chained nodes is Inferred |
| B25 | (reclassified) `T2V` and `VideoSampler` pass the noise tensor as the start latent too. This is documented in the `latent_noise` tooltip (`t2v.py:883-884`), so it is intended behaviour, not a defect. Effect on EPS models is still Unknown | `nodes/video/t2v.py:883-884, 999-1001` | Documented behaviour. See CODE_REVIEW §3 |
| B26 | `AudioTranscribe` kept the original segment start on every split chunk (fixed in 4.0: time shared by characters, `_split_long_segments`) | `nodes/pipeline/audio.py:transcribe` | Fixed in 4.0 |
| B27 | **Fixed in `16afaf6`.** `ProjectManager` packs the API-format `PROMPT`, which has no `nodes` key, so saved metadata is empty and reopening from the library probably fails | `nodes/pipeline/workspace.py:155-157, 279-281`; `js/radiance_workspace.js:855` | Code Implemented (review). Reopen failure Inferred [P2-2] |
| B27b | **Fixed in `16afaf6`.** The shot and version regexes start with `\b`, and `_` is a word character, so the node's own names (`sh010_v002`, `comp_artist_v0003`) match neither, giving shot `GENERAL` and version `v001` | `nodes/pipeline/workspace.py:662-675`, name built at `:146` | Implemented (mapper: both patterns tested with Python `re`) [P2-3] |
| B28 | **Fixed in `945458a`, `77075f5`.** `.shot_status.json` path is built from file metadata `project.name` without a containment check | `workspace.py:745-753, 797-798, 815-821` | Implemented (review) [P3-3] |
| B29 | `_verify_or_report_sha256` was never called; files on disk were trusted when their size matched (fixed in 4.0: `fetch` checks an existing file once and records the pass in `<file>.radiance-sha256`; the dead function is gone) | `nodes/upscale/upscale.py:212`; `core/model_fetch.fetch` | Fixed in 4.0 |
| B30 | **Fixed in `2af342b`.** `.rhdr` fp16 export overflows: `hdr/vae.py` casts to float16 without the ±65504 clamp `viewer.py` applies, so values above 65504 become `inf` | `hdr/vae.py:2709`; compare `nodes/monitor/viewer.py:870` | Implemented (mapper) [P3-4] |
| B31 | **Fixed in `c4492a8`.** `/radiance/media/*` is a file-existence oracle: `os.path.isfile` runs before the allowed-root check, so a missing path returns 404 (path echoed) and an existing one 403 | `nodes/io/write.py:1414-1422` | Implemented (mapper) [P3-2] |
| B32 | **Fixed in `fb88615`.** `/radiance/assets/upload` overwrites an existing file of the same name, non-atomically | `nodes/pipeline/workspace.py:1866-1890` | Implemented (review) [P2-5] |

The B14 question from the previous revision is resolved and folded into B17: `pixel_download` does verify SHA-256.

## Design and ownership questions (Unknown)

- **User data lives inside the package checkout** (`<repo>/workflows`, `<repo>/gizmos`). What happens on `git pull` or a ComfyUI Manager reinstall?
- **No auth on `/radiance/*` routes.** This is acceptable only while ComfyUI binds to localhost. `/deliver` containment uses `abspath`, not `realpath`.
- **More structural debt** (dashboard re-reads every `.rad` per request, error JSON returned with HTTP 200, test doubles with the wrong contract): see [CODE_REVIEW §2](CODE_REVIEW.md#2-structural-debt-not-defects-today).
- **Five OCIO config resolvers** with no single owner (see [color-hdr](subsystems/color-hdr.md#contracts)).
- **Logger `propagate=False`:** does ComfyUI's log capture see Radiance logs? And do group-import WARNINGs (emitted before `setup_radiance_logging`) reach the console formatted as intended?
- **Raw `fetch('/radiance/...')` in the JS:** does it break when ComfyUI is served under a sub-path? (Inferred.)
- **`image/upscale.py` sets `KMP_DUPLICATE_LIB_OK=TRUE` at import** for the whole ComfyUI process. Is that intended?
- **Gizmo executor** bypasses the ComfyUI executor and has no tests. Which node types are known to work inside a gizmo?

## Proposed (not implemented)

These are recommendations only. None of them describes current behaviour, and none has been
agreed with the maintainers.

- **P1. One download path.** Done in 4.0 for MoGe, Marigold, ACESConfigManager and whisper (B17, B18). Left: pin the multipass `_MODEL_REGISTRY` hashes.
- **P2. One OCIO resolver.** Done in 4.0: `color/ocio_setup.active_config_path()` / `active_config()` answer for `color/encodings.ocio_config` (Write/Read), `hdr/ocio._resolve_config` and `radiance_ocio.discover_ocio_config`. The ACES 2.0 reference renders (`hdr/aces2_ocio`, `color/display_preview`) stay pinned to studio v4.0.0 on purpose; `ACESConfigManager._find_existing_config` is a disk search for its own action, not a default.
- **P3. Move user data out of the package dir**, for example under ComfyUI's user directory.
- **P4. Shared convention constants** for depth polarity and motion-vector axis, and convert at `RADIANCE_PASSES` boundaries (addresses B24).
- **P5. Atomic media writes** (temp file and `os.replace`) for EXR, video, and workspace files (addresses B12 and the workspace concurrency gap).
- **P6. Test gaps:** RadianceAIUpscale, the gizmo executor, RGBA on review tools, and the shipped workflow JSONs executing end to end.

## Not inspected (next bounded checks)

- `hdr/color.py`, `hdr/processing.py`, `hdr/panorama.py`, and `hdr/recovery.py` bodies.
- `nodes/generate/prompt.py`, `regional.py`, `resolution.py`, `denoise.py`, and `energy.py` internals.
- The `js/radiance_viewer.js` body past the data-flow entry points.
- Third-party download behaviour (SeedVR2, facexlib, whisper) and spandrel dtype defaults.
