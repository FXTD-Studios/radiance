<!-- project-mapper:generated -->
# Upscale, QC, and realtime monitor nodes

ID: upscale (plus qc, monitor-realtime). Snapshot: `7376f9e`. Coverage: partial. Agent trace of `nodes/upscale/upscale.py` (3,121 lines), `image/upscale.py` (2,694), `image/defects.py`, `nodes/color/qc.py`, and `nodes/monitor/realtime.py`, read with grep and bounded reads. The mapper re-read the unused sha256 helper, the delivery 2x upscale call, the blend modes, and the `HTTPServer` choice. Labels are explained in [START_HERE](../START_HERE.md#evidence-labels).

## Responsibility and boundaries

This subsystem covers resolution changes (AI and classical), bit depth, face restoration, image QC
and policy checks, and the review utilities in `realtime.py`. The Viewer node itself is in
[frontend-routes](frontend-routes.md).

## Implementation index

| Source/symbol | Role | Label |
| --- | --- | --- |
| `nodes/upscale/upscale.py:_build_upscale_fn` (:1254) | Backend chain, tried in order: (1) a connected `UPSCALE_MODEL` via `comfy.utils.tiled_scale`; (2) Tier 3, SeedVR2 (external package) or diffusers SD-x4; (3) Tier 2, HAT-L/SwinIR-L via spandrel; (4) Tier 1, built-in Real-ESRGAN `_RRDBNet`; (5) bicubic, labelled "NOT an AI upscale" | Implemented |
| `upscale.py:_hdr_to_sr_domain` (:471) / `_hdr_from_sr_domain` (:489) | HDR wrap: Reinhard x/(1+x) before the model and y/(1-y) after. Not log or PQ | Implemented |
| `upscale.py:tiled_upscale` (:564), `_tile_weight_map` (:499) | Tiles with reflect padding. `linear` uses `core/tiling.blend_weight_2d`. **`laplacian_pyramid` is listed but falls back to the Gaussian feather** | Implemented (mapper) |
| `upscale.py:upscale_video` (:2132) | Overlapping temporal windows, Lucas-Kanade alignment of overlap frames, Laplacian blend. Not a temporal model, so flicker remains. The output accumulator is a whole-clip CPU tensor | Implemented |
| `upscale.py` FaceRestore (:2891) | RetinaFace (facexlib), falling back to a Haar cascade. CodeFormer or GFPGAN v1.4. **Output clamped to [0,1]** | Implemented |
| `image/upscale.py` ProUpscale / BySize / Downscale32bit | Separable 32-bit kernels (lanczos, mitchell, …). No clamp. CPU numpy tiling above 64 MP | Implemented |
| `image/upscale.py:RadianceAIUpscale` (:1782) | spandrel, with a SUPIR bridge. Modes `Standard`, `Refine (HDR)` (log1p), `Normalize (HDR)`. **Any exception returns bicubic instead of raising** | Implemented |
| `image/upscale.py` BitDepthConvert (:1517) | Float targets keep HDR. Integer targets clamp. Several dither modes | Implemented |
| `nodes/color/qc.py:RadianceQC` (:208) | Uses `image/defects.py` metrics. **FAIL** on NaN/Inf, below `black_threshold`, above `white_threshold` (default 1.0, so any HDR value fails), or gamut over 1%. Raises only if `fail_on_errors` | Implemented |
| `nodes/color/qc.py:RadiancePolicyGuard` (:482) | Presets: Broadcast SDR, Cinema P3-PQ, OTT HDR10, Social. Score = 100 − 25·errors − 5·warnings. A policy failure never raises. "P3 gamut" is a range check | Implemented |
| `nodes/monitor/realtime.py` | FalseColor, FocusPeaking, SplitView, ContactSheet, FrameStamp, FlipbookGIF, PreviewServer | Implemented |
| `realtime.py` PreviewServer (:1016, server :969) | Single-threaded `HTTPServer` on 127.0.0.1 (default port 8765). `/`, `/frame/<name>` (JPEG of the last frame, clamped), `/health`. In memory, no disk writes. Shut down only when restarted on another port. No atexit hook | Implemented (mapper) |

## Contracts

- **HDR is not preserved by every upscale path.**
  - Kept: the Tiler, Image, and Video Reinhard wrap; `iu` 32-bit kernels; BitDepth float targets.
  - Lost or clamped: FaceRestore, the Tiler ColourFix mode, AIUpscale `Standard` mode (values above 1 are fed raw into an SDR model), and integer BitDepth targets.
  - All Implemented.
- **Alpha:** Real-ESRGAN and SD-x4 upscale alpha separately. spandrel and basicsr keep only RGB, so RGBA through Tier 2 probably raises (Inferred). FaceRestore pastes 3-channel crops into RGBA with a `repeat`, so alpha inside each face box is corrupted (Implemented).
- **Downloads:** `_download_upscale_model` uses `core/model_fetch.fetch` with the `RADIANCE_UPSCALE_OFFLINE` legacy flag. `_verify_or_report_sha256` is defined but never called (mapper). `AIUpscale._download_model` doesn't pass the legacy flag. SD-x4 uses `from_pretrained` with a pinned revision. SeedVR2 and facexlib may download on their own, outside the consent gate (Inferred).
- **`image/upscale.py` sets `KMP_DUPLICATE_LIB_OK=TRUE` at import**, for the whole process (mapper).
- **Delivery's 2x option** calls `RadianceAIUpscale` (`delivery/handler.py:705`). AIUpscale falls back to bicubic silently, so a delivery could ship a bicubic 2x master with no warning (Inferred).
- **RGBA on review tools:** FocusPeaking, SplitView wipes, and ContactSheet fail on 4 channels, as `KNOWN_ISSUES.md:22` records. FlipbookGIF and PreviewServer probably fail too and are not listed there (Inferred).

## Tests (inspected by agent, not run)

- `tests/test_upscale.py` (43), `test_upscale_honesty.py`, `test_p0_upscale_fixes.py` (bicubic offline geometry), `test_upscale_kernels.py`, `test_image_upscale.py`.
- `test_policy_guard.py` (30), `test_realtime_preview.py` (90).
- RadianceQC is covered in `test_p0_color_fixes.py` and `test_documented_bugs.py`.
- **AIUpscale has no tests.** No test covers RGBA on FocusPeaking, SplitView, or ContactSheet.

## Open questions

See [OPEN_QUESTIONS](../OPEN_QUESTIONS.md): B22 (silent bicubic in delivery), B29 (unused sha256 verifier), D16/D17 (tooltip and blend-mode drift), B23 (RGBA in Tier 2 and FaceRestore), B13 (offline flag).
