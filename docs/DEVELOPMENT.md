# Radiance development record

For contributors and reviewers: what has been verified, and what is open
before and after this release.


Radiance is **suitable for a controlled pilot; full production approval is open**. Everything below is measured rather than
asserted; the full history is in the [changelog](../CHANGELOG.md), and per-defect
detail in [KNOWN_ISSUES.md](../KNOWN_ISSUES.md).

## Verified

### Viewer feedback fixes — 2026-09-27

- Imported H.264/AAC video: Chromium decoded audio, looped beyond the two-second
  fixture duration, and restored the server-backed source through ComfyUI's
  `loadGraphData` workflow transition. Playback resumes on an explicit Play action.
- Executed 48-frame sequence: media references and audio restored after graph
  replacement; forward playback measured frame 21 at audio time 0.860 s (24 fps).
  Audio pauses during buffering and reverse playback, and follows seeks and rate.
- Keyboard: Space ignored outside an unselected Viewer and while typing; selected
  Viewer playback works. The two-Viewer browser harness verifies hover ownership
  and that leaving the Viewer releases shortcuts.
- Frame upload now follows decoded-video callbacks and avoids asynchronous bitmap
  queues. This removes redundant work; smooth playback of large HDR sequences on
  all hardware is not an acceptance claim.
- Limits: IMAGE batches have no audio. Executed previews reference ComfyUI temp
  files and need a rerun if those files are removed. Imported videos are retained
  under `input/radiance-viewer`; sharing a workflow requires sharing its media.

Standing properties of the shipped package. Pinned by a test unless the row
says otherwise. An audit number that no test holds is a number that can
quietly stop being true.

## Production readiness check — 3.5.0 (2026-09-25)

Base commit `328d297`, with the local production fixes described below.
**Release approval remains open.** Passing this checkpoint does not resolve
all the feature limitations in `KNOWN_ISSUES.md` or certify every model family.

| Check | Measured result |
| :-- | :-- |
| Full Python suite | 4692 passed, 91 skipped, 3 slow tests deselected; 28 subtests passed. Rechecked after Resolve DLL discovery and the ControlNet window guard (2026-09-26). Windows, Python 3.12.12, real torch / OpenEXR / OCIO / OpenImageIO. |
| Coverage | 68.42% including branches on the current worktree; configured floor 54%. |
| Exhaustive video tests | All 3 slow tests passed separately. |
| Console encoding | 6 additional regressions passed for ASCII, Windows cp1252 and UTF-8, including exception tracebacks. |
| JavaScript / browser | 324 passed, 5 skipped, no failures. Four checks require Deno; one is the opt-in hardware check. Includes the new sampler UI regressions. |
| GPU graph | Real RTX 4080 SUPER, Z-Image Turbo at 512x512: Radiance Sampler Pro → HDR VAE Decode (Direct HDR) → Write (32-bit EXR). ComfyUI reported success in 60.04 s. |
| GPU output | `output/radiance_acceptance/production_20260925_001.exr`: 512x512 RGB float32, all finite, range 0..4.926108, Rec.709 chromaticities and linear encoding metadata. Prompt ID `05e730a3-d982-4d7a-9869-c3a9d435eae6`. |
| Packaging | Wheel rebuilt after the fixes; all 47 required frontend assets and OCIO notices verified present; no tests or scratch data included. |
| Static checks | Syntax / undefined-name lint and release hygiene checks passed. |

Confirmed product fixes in this pass:

- Nuke live handoff (2026-09-26): installed Nuke 17.1v2 runs with the
  interactive license (`-i -t`). Its render-only license request still fails.
  The listener's GUI-only calls failed in terminal mode; dispatch now runs
  synchronously there, with node edits on the calling main thread. The real
  signed Send-to-Nuke path created a Read node for frames 1001-1003, raw mode,
  and preserved RGB values 0.18, 1.0 and 4.0. A separate 512x512 production
  EXR read also passed. GUI dispatch is tested with a stand-in; the live
  application test used terminal mode. The Nuke license/import blocker is closed.
  All 26 DCC regressions passed, including GUI/terminal dispatch and same-thread
  terminal startup. Static/release checks passed. The wheel was rebuilt and
  checked against the current listener, worker and sampler, with all 47 assets.
  This follow-up postdates the full-suite/coverage baseline in the table above.
- Windowed ControlNet now fails before a model evaluation can receive control
  tensors for a different frame range. Unwindowed and whole-clip-window
  ControlNet remain supported. Window/sampler/reference/DCC checks: 105 passed.
- Resolve DLL discovery (2026-09-26): the isolated Windows worker sets
  `PYTHONHOME=sys.base_prefix` before importing fusionscript. This fixes the
  reproduced native crash in both installed Python environments without changing
  ComfyUI's environment. DCC suites: 23 passed, including worker isolation.
  A live Resolve Studio test through the ComfyUI Python worker imported an EXR
  still and a 121-frame sequence into `Radiance_Acceptance_20260926_1413`.
  Media Pool frame counts were 1 and 121. The prior project was restored and
  the temporary headless Resolve process exited. Nuke verification is recorded above.
- Full-context overlap candidate (2026-09-26): keep the pulled-back final
  window instead of dropping its predecessor. Each incoming window's fade
  attenuates every earlier contributor, preserving unity even at a three-way
  join. All windows retain the requested length. Window/sampler regressions:
  88 passed; window/reference checks: 63 passed (overlapping counts).
  Syntax checks passed. Prompt `5d206d6c-2338-42d1-92f3-930414f314fd`
  completed in 545.28 s and exported 121 finite frames. Maximum adjacent-frame
  mean difference fell to 0.02927 (old native-schedule windowing: 0.07024),
  but sampled join frames still show ghosting. This fails visual acceptance.
  A 241-frame follow-up uses 31 latent frames per window, matching the
  successful stock render's temporal context, with overlap 8. Prompt
  `0f6d30be-9457-4ad9-b6f4-9a4f34d8d4b5` completed and exported 241 finite
  1280x704 frames. It still failed visual review: overlapping boats/backgrounds
  around frames 97-121, with different framing on either side. Maximum mean
  adjacent difference was 0.06093 at frame 169. Elapsed time was 19m20s,
  including GPU contention from another job; this is not a throughput benchmark.
  A diagnostic 121-frame run now uses 10 Euler ancestral steps followed by
  10 Euler steps with continuous sigmas and no fresh initial noise on the
  second stage. Prompt `a4cd3969-9a5e-43f0-a63a-01bc0fe36fad` stopped during
  its second stage with no exported frames. It was resubmitted as
  `6a40fb19-737b-4d6b-9135-6a0be89a0cf0`, which completed in 348.97 s.
  All 121 frames were finite; maximum mean adjacent difference was 0.02000.
  Sampled frames still showed a change in the boat, framing and background
  across the join, so visual acceptance failed. A separate shuffled-noise
  diagnostic (`67d1c236-34b0-4a40-a11c-b0b8ed3a21ea`) also completed with
  121 finite frames, maximum mean adjacent difference 0.02890. Detailed join
  samples showed a crossfade between scene layouts, with severe final-frame
  artefacts. It failed visual acceptance and remains a scratch experiment;
  no shuffled-noise default or new sampling mode was added.
  This tests an early stochastic phase, informed by
  [FlowLong](https://arxiv.org/html/2605.20910v1), using existing ComfyUI samplers;
  it is not a claim to reproduce that paper's method or results.
- The full candidate suite found a Windows bridge failure: the oversized-line
  response was lost to a connection reset. The bridge now shuts down sending
  and drains unread input for at most 0.25 s before closing. Eleven DCC tests
  passed, including different excess payload sizes and bounded handler exit.
  This follows the shutdown-before-close pattern in Microsoft's
  [Winsock guidance](https://learn.microsoft.com/en-us/windows/win32/winsock/graceful-shutdown-linger-options-and-socket-closure-2).
  Full recheck passed: 4688 tests, 91 skipped, 3 deselected, 28 subtests;
  coverage 68.08%. Current wheel verified against sampler, blend, bridge,
  UI and worker source, with all 47 assets and no test/scratch files.
  The larger-context and stochastic-phase runs failed visual review.
  Temporal-window tooltips and the run warning now state the observed risk
  and recommend 0 for production. No visual approval is inferred.

- Sampler Pro extra-shift correction (2026-09-26): Auto/Custom now preserve
  `flux_shift=1`, because the model schedule already includes its native shift.
  Explicit extra-shift values still apply. Sampler and reference regressions:
  209 passed; full Python suite: 4674 passed, 91 skipped, 3 deselected,
  28 subtests passed. Release hygiene and syntax checks passed. Rebuilt wheel
  includes all 47 runtime assets and the Resolve worker; its sampler matches
  current source. This run did not remeasure coverage.
  All three slow video tests also passed separately.
  Before the correction, full-resolution unwindowed Radiance also failed
  visual review (prompt `4174a015-5dd5-4adb-b3a5-029a21eb611c`), so windowing
  alone does not explain the original failure. All 121 exports were finite,
  but showed severe structure/colour artefacts. Using the native schedule
  (prompt `0ed44e1a-5bf0-4960-be6b-c639ee712a11`) restored a coherent sailboat
  in six sampled frames: 121 finite exports, maximum adjacent-frame mean
  absolute difference 0.02792 (stock 0.02613), versus 0.11317 before correction.
  A separate check against real ComfyUI ModelSamplingDiscreteFlow confirms
  the corrected default schedule matches all 21 native shift-8 sigma values
  exactly. This validates schedule parity, not full motion or model-family
  acceptance. The windowed native-schedule comparison (prompt
  `714b894e-84c7-4dc3-b931-f8544e045f1e`) removed the severe colour artifacts,
  but the scene changed across the join. Maximum adjacent-frame mean absolute
  difference was 0.07024 at frame 62; windowed visual acceptance still failed.
- Window overlap investigation (2026-09-26): allowing a shorter final window
  preserved the requested overlap but failed real visual acceptance. Prompt
  `a967eb92-da07-4e5b-a15b-36b4b7a5a023` completed in 382.97 s and exported
  121 finite frames; sampled images showed ghosting at a join and severe
  tail-frame artefacts. This planner change was reverted. The existing
  full-size-window planner remains, including its known reduced-overlap
  limitation. Windowed-video production acceptance is still open. Temporary
  server stopped after the queue completed.
- Frontend schedule correction (2026-09-26): the UI no longer auto-writes
  native model shifts into the extra-shift widget, and Wan presets select 1.
  Existing saved explicit values are preserved; use 1 to avoid adding another
  shift to a native schedule. Full JavaScript suite: 324 passed, 5 skipped
  (4 Deno/WebGPU checks and the opt-in hardware test). Nine new tests exercise
  metadata updates and Wan preset values. Final Python recheck: 4674 passed,
  91 skipped, 3 deselected, 28 subtests passed; coverage 68.07%. Final wheel
  rebuilt and verified against current sampler, planner, UI and worker source;
  all 47 runtime assets present and no test/scratch files included.

- **Reproduced video blocker at full settings (2026-09-26):** Radiance
  windowed sampling at 1280x704, 121 frames, 20 steps failed visual review
  with severe artefacts despite a coherent stock baseline. Prompt
  `372ba47d-77e0-49ec-a47e-7bde61707fed`: 232.85 s sampling, 296.79 s total,
  121 finite float exports. The requested overlap 4 became one latent frame
  after planner pruning. Maximum adjacent-frame mean absolute difference
  was 0.1133 near the join (stock 0.0261); these statistics support the visual
  finding but do not identify its cause. Next isolate unwindowed Sampler Pro
  at identical full settings before attributing this entirely to windowing.
  Temporary server stopped; failed acceptance is retained in KNOWN_ISSUES.

- Decode isolation (2026-09-26): the low-resolution failed video also shows
  artefacts with stock KSampler and stock VAEDecode, removing Radiance's
  decoder from that comparison. Prompt `e9ab6d66-901f-4d11-9d73-3d41c34a6a3d`
  completed successfully but failed visual review. A new stock baseline
  follows the official Wan 2.2 template's 1280x704, 121-frame, 20-step,
  CFG 5, uni_pc/simple, shift-8 settings. Prompt
  `54757a09-9c26-4f2b-bf8e-eb45853b34bf` completed in 276.43 s and exported
  121 frames. Six sampled frames showed a coherent red sailboat scene,
  without the severe low-resolution artefacts. This establishes a usable
  stock visual baseline, not full motion certification or Radiance windowing
  acceptance. Stock VAEDecode output was exported as raw display-encoded
  float values for comparison (not a scene-linear mastered deliverable).
  The temporary server was stopped after review. Template reference:
  https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_wan2_2_5B_ti2v.json

- **Video visual acceptance failed (2026-09-26).** Contact-sheet inspection
  of the 129-frame, 256x256, eight-step Wan run showed severe colour/structure
  artefacts and large brightness shifts. A matched low-step stock KSampler
  run and a Radiance run with windowing disabled also showed artefacts, so
  this does not isolate a windowing defect. Stock prompt:
  `384d97c7-9ab0-4d1c-b783-b1a56b2f3c32`; unwindowed Radiance prompt:
  `121f5e11-a60c-4cb1-9c13-ef2a0716a814`. The successful execution and EXR
  checks below remain valid only as numerical/transport tests. A known-good
  model workflow at sufficient resolution/steps is needed before temporal
  quality and seam acceptance can pass. No video production approval is
  inferred from this run. Temporary server stopped after comparison.

- Real windowed-video acceptance (2026-09-26): installed Wan 2.2 TI2V 5B,
  UMT5 and Wan 2.2 VAE ran on RTX 4080 SUPER through Sampler Pro, temporal
  windowing, Radiance tiled video decode and Write. Requested 129 frames at
  256x256; 33 latent frames were evaluated as three overlapping windows of
  16, overlap 4. Eight sampling steps completed in 12.79 s; the generation
  and decode prompt completed in 21.23 s. The sampler resolved to uni_pc
  with shift 8 through its model defaults. Sequence export then reused the
  cached result (0.87 s). All 129 RGB float EXRs, frames 1001–1129, read back
  at the expected dimensions with finite values in 0..1 and nonidentical
  adjacent frames. Generation prompt: `918224da-7fe1-49fb-9411-e7816a5f634a`;
  sequence export: `d2e5b8ec-bd0e-42ec-9d41-a00d845d52b9`.
  Files: `output/radiance_acceptance/video_window_20260926/`.
  This proves execution across real model windows, not seam-free visual
  quality, long-form duration, high-resolution memory scaling or every video
  model. Eight-step images were not quality-certified. Downloads were disabled;
  the temporary ComfyUI server was stopped after validation.

- Hardware WebGL acceptance (2026-09-26): all 18 Viewer colour checks passed
  in Chromium with the hardware backend, including OCIO, wide-gamut inputs,
  scope signal and viewer-only exposure. The test verifies the renderer is a
  named hardware GPU rather than SwiftShader. Run it on Windows with
  `RADIANCE_TEST_HARDWARE_GPU=1 node --test js/tests/viewer_color.test.mjs`
  (set the environment variable with the syntax appropriate to the shell).
  This checks colour/interaction correctness; sustained playback rate remains
  unmeasured. Default test mode continues to use SwiftShader.

- Denoise detail recovery now restores the high-frequency removed residual
  rather than undoing all denoising. Constant-offset and texture regressions
  plus spatial checks: 7 passed. Real RTX 4080 SUPER comparison suite, now
  including full denoising with recovery and alpha: 6 passed. Generated
  reference checks: 3 passed; syntax checks passed. Nonzero recovery changes
  appearance; default 0 is unchanged. This postdates the full-suite baseline.

- RTX 4080 SUPER acceptance with the production ComfyUI interpreter (torch
  2.12.1+cu130): five real CUDA tests passed, comparing guided spatial filtering,
  trimap matting and feathered stitching against CPU, including chunk boundaries.
  At 1920x1080, three warmed runs measured median 6.17 ms for the guided
  spatial helper (radius 9, sigma 4), 11.86 ms for matting (radius 12) and
  9.84 ms for standard stitching (radius 16). Incremental peak allocated
  GPU memory was 223.64, 183.82 and 143.73 MiB respectively. All outputs
  were finite. The guided measurement starts with GPU-resident input; matting
  and stitching start with CPU input. These are single-frame operation
  measurements, not full denoising or end-to-end video throughput.

- Linear Matting uses trimap_dilation to form an unknown edge band and
  preserves known foreground/background interiors, including image borders.
  Radius 0 preserves the clamped input mask; incompatible mask batches fail
  clearly. VFX regression suites: 58 passed; generated-reference checks:
  3 passed; syntax checks passed. This postdates the full-suite baseline.

- Cinema D60 has its own AP1-to-P3 matrix at ACES white, replacing the
  duplicated D65 conversion. Tests derive the expected colourimetry from
  primary chromaticities and verify distinct coloured-patch output and
  neutral-grey preservation. ACES tests: 85 passed; generated-reference
  checks: 3 passed; syntax checks passed. This is a correction within the
  existing rendering pipeline, not an independent certification of the
  entire ACES output transform. White-point reference:
  https://docs.acescentral.com/system-components/output-transforms/parameters/

- Guided denoising uses sigmaSpace for Gaussian-weighted local statistics.
  Independent full-2D convolution checks verify its separable computation,
  including edge normalization, constant HDR colours and tiny images.
  Targeted checks plus the node execution harness: 130 passed, 42 skipped;
  generated-reference checks: 3 passed; syntax checks passed. This fix
  postdates the full-suite baseline above.

- Dynamic sampler guidance adds ramp-end boundaries so middle/late targets
  are reached when the step range includes them. Harness tests capture the
  actual conditioning/CFG passed to sampling, verify reduced-denoise and
  partial ranges, and check sigma continuity. Sampler regression suites:
  143 passed; generated-reference checks: 3 passed; syntax checks passed.

- False Colour decodes explicitly selected sRGB or Rec.709 inputs before
  measuring linear luminance. Independent encoded middle-grey values verify
  the exposure zone; node tests verify encoding forwarding across a batch.
  Zebra strength, zero-strength HDR passthrough and RGBA alpha preservation
  are covered. Preview and generated-reference suites: 93 passed; syntax
  checks passed. Linear Rec.709 remains the default for saved workflows.

- Standard inpaint stitching now applies its feathered mask; radius 0 retains
  the previous hard-mask result. Frame Stamp renders only the overlay through
  Pillow, preserving float HDR/negative values elsewhere and source alpha.
  Realtime-preview and VFX regression suites: 129 passed; generated-reference
  checks: 3 passed; syntax checks passed.

- Video look controls: every offered gamut, EOTF and peak reaches T2V and I2V
  text conditioning; negative prompts remain untouched. Prompt descriptors do
  not establish physical output mastering. Video regression suites: 58 passed.
- Synthesis chroma preservation now blends neutral and proportional highlight
  energy at equal luminance. Its default 0.8 changes coloured highlights;
  1.0 retains the former proportional result. HDR control suite: 25 passed;
  documentation and adjacent HDR regressions: 25 passed. Syntax checks passed.

- WebGL built-in display views convert ACEScg, ACES2065-1, linear Rec.2020
  and linear P3-D65 source primaries. Browser colour patches agree with
  OCIO-derived Rec.709 references; linear exports preserve source values.
- Follow-up HDR controls: ClipDetector preserves channel mode with feathering;
  ExposureBlend preserves batches and weights unclipped original brackets;
  HDR Monitor respects its gamma switch for Exposure + Gamma. Targeted
  regression checks cover colour-channel selection, batch ordering, singleton
  brackets, clipped highlights, middle exposures, black inputs and alpha.
  Follow-up run: 179 passed, 42 skipped across the affected HDR suites and
  node execution harness; these counts overlap the full-suite baseline above.

- Floyd–Steinberg dithering no longer modifies the upstream image; its
  arithmetic matches the scan-line reference on NumPy 1 and 2.
- Video encodes stamp colour metadata on filtered frames as well as codec
  options, preserving Rec.709 and HDR10 tags with the installed FFmpeg build.
- CDL sidecars use UTF-8 on Windows.
- Two-component lighting fits compare boundary solutions even when the
  least-squares solver returns a non-negative but inferior singular solution.
- Console messages and tracebacks are escaped for the actual stream encoding;
  an arrow in a log message previously interrupted a real Windows export.
- The wheel now includes the Viewer's OpenColorIO JavaScript, WASM and notices.

Validation fixes: local test-helper imports avoid an unrelated installed
`tests` package; log assertions capture the non-propagating Radiance logger;
model-free tests explicitly isolate installed checkpoints or choose Expand;
Windows memory tests measure peak working set; browser tests select SwiftShader
through ANGLE. CI now covers release branches, runs the full job on mirrors,
applies the project's coverage floor there, and gates syntax/undefined-name
checks. These workflow changes have been checked locally, not run on GitHub yet.

Remaining production work: resolve or safely expose the documented inert
controls and Viewer colour/backend gaps; verify long-video GPU behaviour and
long-video visual acceptance; run the changed CI on the release revision and
reconcile the public release branch. Version remains 3.5.0; nothing published.

Workstation DCC check: Nuke 17.0v4 is installed, but a headless EXR read
probe exited with noLicenseFound (no product licence or login token). No
Nuke handoff was exercised. DaVinci Resolve and its scripting module are
installed. A headless Resolve launch succeeded, but importing its native
scripting DLL crashed both the base and ComfyUI Python interpreters with
Windows access violation 0xC0000005. The Send node now runs that API in a
separate worker. The real failure was reproduced through this boundary:
the host survived and returned an actionable failure status. Nine DCC
integration tests passed, including real worker success, abrupt process exit
and timeout handling. The later DLL-discovery correction and successful live
Resolve import are recorded above. The subsequent Nuke 17.1v2 handoff also passed.

Packaging follow-up: rebuilt the wheel with the Resolve worker and Guided
spatial fix. Extracted the worker from that wheel and launched it standalone
against a test scripting API, verifying media paths and the returned result.
CI now explicitly checks worker inclusion. DCC and Guided regression suites:
14 passed. Release hygiene and diff checks passed. Live Resolve import was
subsequently verified with the DLL-discovery correction recorded above.

| Area | What is verified |
| :-- | :-- |
| **EXR** | 32-bit float round-trips bit-exactly through EXR and TIFF, negatives and over-range highlights included, the clamp-free HDR claim, as a write-and-read rather than an assertion. 16-bit half holds to 1e-3. |
| **Colour** | All 16 colour spaces are pinned to a published 18%-grey value: 15 transfer curves plus the linear identity, each held to 1e-4 and cross-checked against colour-science's independent implementation of the same specification. Both ACES 2.0 tone scales place 18% scene grey where the ACES Output Transform publishes it: 10.000 / 13.193 / 14.512 / 15.747 / 16.824 nits for peaks of 100 / 500 / 1000 / 2000 / 4000, held to 1e-6, with intermediate peaks checked against the geometric-mean form of the log-log rule. The shipped table is compared against a transcription of the published one rather than against the interpolator that reads it, which is what made the old test unfalsifiable. HLG keeps its BT.2408 anchor. The OCIO bake is checked against OCIO's own CPU processor, exactly rather than approximately. |
| **Transfer** | PQ encodes absolute luminance against ST.2084's fixed 10 000 cd/m² ceiling, pinned at five mastering peaks and against an absolute-luminance ladder, and cross-checked against the package's other PQ encoder. Rec.709 and Rec.2020 are the real BT.709-6 and BT.2020-2 OETFs with the BT.2020 primaries matrix, pinned by value, and every offered output colour space is asserted to change the data, so a missing conversion cannot pass as a successful write. |
| **Video** | Frame counts are exact from 1 to 100 frames across H.264, H.265 10-bit and ProRes 422 HQ, by encoding and reading back real media. The default suite covers 17 lengths per codec, chosen around the 1/2/3 degenerate cases and both sides of every GOP boundary, and asserts the identity and order of each frame as well as the count, at `core.video` and again at the Read node. The exhaustive 1-to-100 sweep runs under `-m slow`. Sequences read correctly by frame number for `####`, `%04d` and explicit ranges. |
| **Duration** | The write path, the HDR VAE encode and decode, the viewer and the sampler's noise generation all hold a working window rather than the clip. Measured, not asserted: writing 32 frames and writing 512 frames peak within 0.1 MB of each other, and enabling a colour transform costs 1.6 MB rather than a second copy of the shot. The VAE's decode overhead is flat at 5.7 MB from 4 frames to 32 where it used to grow by a whole extra clip. Sequence length is bounded by disk. Generation is the exception and has its own control, see below. |
| **Memory** | Flat across 150 consecutive 1080p runs, an audit measurement rather than a standing test. |
| **Security** | `weights_only` loads, no `shell=True`, and every model download pinned: a Hugging Face commit or the original release file, plus the file's SHA-256 and size, checked before the file is installed (`core/model_fetch.py`; `tools/pin_models.py` re-checks every pin online, 79 of 79 on 2026-09-25). Models download on first use through one gate every downloader shares (`core/consent.py`); `RADIANCE_ALLOW_DOWNLOADS=0` or the Hugging Face offline flags stop all of them. Gated repositories use the user's `HF_TOKEN`. Nodes never write into the ComfyUI install directory. |
| **Catalog** | All 156 nodes declare their menu section explicitly; a test fails if a registered node is missing from the table. Withholding a node from the menu requires a named entry with a written reason a test reads and checks the length of. Separately, an AST walk of every file in the distribution finds every `NODE_CLASS_MAPPINGS` and asserts each class in it is registered as the class that ships, so a node stranded in a package the catalog does not load turns the suite red. That is how 26 finished nodes stayed out of the menu until 3.4.0. |
| **Isolation** | Every one of the eleven node groups imports with `aiohttp` and `server` blocked, proven in a subprocess rather than for one hand-listed module. The blocker uses `find_spec`; it previously used `find_module`, which Python 3.12 removed, so on the 3.12 leg of the matrix it silently blocked nothing and the test passed while measuring nothing. The harness now proves it is blocking before it reports anything. |
| **Layering** | `radiance/io/writer.py` and `radiance/io/reader.py` import nothing above them, checked by AST walk *and* by running them in a bare interpreter with no ComfyUI present. |
| **Suite** | See the dated production readiness check above for current counts, coverage, GPU acceptance and explicit verification limits. |

## Open

**Blocking a release**

- [x] **One full GPU render in live ComfyUI.** Passed on 2026-09-25 with
      Z-Image Turbo, Radiance Sampler Pro, HDR VAE Decode (Direct HDR), and
      Write to 32-bit float EXR on RTX 4080 SUPER. See the measured output and
      prompt ID above. This covers one image model, not every video/model path.
- [ ] **The public repo is behind, and it is not a fast-forward.**
      `fxtdstudios/radiance` `main` (`16e885f`) is 5 commits the beta line
      never took. They were reviewed hunk by hunk on 2026-09-23: PR #18's
      two real fixes are ported (Write reports its files to ComfyUI history;
      no upper version caps in the platform requirements or `pyproject.toml`),
      the rest are already fixed here, patch files that no longer exist, or
      are features (movable controls panel, extra Save nodes) left for later.
      PR #20 edits a README section that no longer exists. What remains is
      the publishing decision: replace public `main` with this line.

**Needs a GPU to confirm**

- [ ] **Long-video windowing is unproven on a real model.** `temporal_window` on
      the sampler denoises a long clip in overlapping latent windows, blended at
      every step rather than after each window is finished, so peak VRAM follows
      the window size instead of the clip length. CPU tests pin the parts that
      are checkable without a model: the schedule covers every frame at full
      weight, the blend weights form a partition of unity, step accounting and
      seeding are stable across a rerun, and a single window reproduces the
      unwindowed result. What CPU cannot show is whether the output is
      temporally coherent on a real DiT. Default is 0, off, so nothing changes
      until it is switched on deliberately. Treat it as ready to test, not ready
      to deliver from, until a long clip has run on the 4080.

**Correctness**

- [x] **Nuke / Resolve handoff and the bridge (3.5.0).** Push to Nuke failed on every default install (the listener requires a signed command; the key had to be set in both environments): shared `~/.radiance/dcc_token` fallback (`core/dcc_auth.py`, mirrored in the Nuke script). Node names made valid for Nuke, quoted `.nk` paths, `input_space` on both Send nodes, Resolve Media Pool import through its scripting API, bridge line cap; "MCP Bridge" relabelled "DCC Bridge" (key unchanged). `tests/test_dcc_send.py` runs the real listener against a stand-in Nuke. Still to do on real software: one push into Nuke 15 and one import into Resolve Studio on the workstation.

- [x] **Every node that needs a model downloads it (3.5.0).** Downloads are on by default; one fetch (`core/model_fetch.py`) pins, verifies, resumes and carries the Hugging Face token. Audit of every source: the Read Models catalogue (60 files) was on unpinned `main` with no digests, one URL did not exist (flux1-schnell-fp8) and the FLUX.1 VAE came from a gated repo (now the byte-identical ungated repackage); the upscale registry's Hugging Face mirrors were wrong or private and HAT-L's GitHub URLs 404 (HAT-L is now a documented manual install, Tier 2 uses SwinIR-L at 4x until it is there); AI Upscale wrote straight to the final path with no check; SeedVR2 pointed at a repository that does not exist; Depth Anything and the SD x4 pipeline loaded `main`. All pinned, 79 of 79 verified online by `tools/pin_models.py`. Live: Real-ESRGAN x2 and SwinIR-L downloaded, verified and ran in ComfyUI from an empty models folder; Depth Anything Small loaded at its pinned commit; a gated file stops with the two steps to get access. `tests/test_model_pins.py`.

- [x] **The bugs found while documenting every input (3.5.0).** All thirteen in KNOWN_ISSUES fixed: ControlNet Apply, Upscale Video, Multipass Relight point + depth, Video Batch Decode, Video HDR Decode, HDR Color Pipeline (plus its wrong D65/D60 and ACEScg-to-Rec.709 matrices), Video Model Info, Digital Cinema Read, EXR MultiPart, Policy Guard / QC `fail_on_errors` / Synthesis nits, AMF escaping, regional prompt chaining, sampler SDR defaults. `tests/test_documented_bugs.py` (18, each failing on the old code); live ComfyUI run of HDR Decode, Color Pipeline, EXR MultiPart and Policy Guard: success. The "controls that do nothing" list stays for the post-3.5.0 pass.

- [x] **Legacy latent RUDRA decoders retired (3.5.0).** The video decoders
      trained on stills, the truncated `ltx-video` full decoder and the
      per-model checkpoint matrix are gone with them. Learned SDR → HDR is the
      pixel model, one checkpoint for every model family, plus the temporal
      residual model for video.
- [x] **The Viewer rendered black (3.5.0), two causes, both verified live on
      ComfyUI 0.32 / frontend 1.48.** (1) The WebGL renderer's `setMask` and
      its mask uniforms addressed a `this.mask` object that the v3.1 refactor
      had replaced with flat fields, so every `render()` threw before the draw
      call; hidden while WebGPU auto-upgraded, exposed when 3.4.0 made WebGL
      the default. (2) The Vue node frontend grew the node to the height of
      the sidebar and inspector content (1180x760 became 1480x2286), the
      canvas stretched with it, and `resize()` never refit, so the frame was
      centred below the visible area. The container now has `contain: size`
      and `resize()` refits an auto-fitted view. Pinned by browser tests that
      load a real frame through `onExecuted`, read the canvas back, and host
      the viewer in an auto-height parent.
- [x] **Read / Write colour management and precision (3.5.0).** Every
      encoding decodes and encodes transfer AND primaries to a selectable
      working space (Linear Rec.709, ACEScg, Linear Rec.2020, Linear P3-D65,
      ACES2065-1), through OpenColorIO's ACES studio config or any
      `ocio_colorspace` / `ocio_config`, with an analytic fallback held to
      OCIO in tests. Video uses the correct YUV matrix and is tagged; EXR and
      DPX carry their colour metadata; 16-bit grey, half-float TIFF and
      alpha read correctly. 65 write-and-read-back tests.
- [x] **Honest release pass (3.5.0).** Every control, option and output was
      checked against the code that reads it. Fixed: I2V strategies now write
      the conditioning keys ComfyUI's Wan models read; T2V/I2V latents follow
      the connected model instead of an LTX default; Video HDR Conditioner and
      Decode reach the model and convert gamut; upscale reports what actually
      ran and Face Restore `auto` restores; mask propagation follows motion;
      Bezier roto, motion-blur energy, Policy Guard (all frames, HDR peak),
      Diagnostics colorspace, Digital Cinema Read colorspace and fps, Audio
      Cut and Camera Sync errors, Regional `Replace`, NDI batches, joint
      bilateral chroma. SAM withheld; ViTMatte/RVM removed. Each has a test.
- [x] **HDR VAE Decode and SDR → HDR (3.5.0).** Auto no longer log-inverts a
      latent a sampler touched (HDR Encode now fingerprints its latent);
      hidden widgets no longer steer the decode; Direct HDR honours
      scene-referred targets, applies exposure after reconstruction and writes
      RHDR from the returned image. One HDR convention everywhere: linear
      1.0 = 203 nits (BT.2408), HLG the BT.2100 1000-nit transcode OCIO uses,
      camera log targets in their camera gamut, AP0 matrix corrected. Checked
      against OpenColorIO and the shipped pixel checkpoint.
- [x] **Clean install from the registry package (3.5.0).** Packed with
      `comfy node pack`, installed into a fresh ComfyUI 0.32 on Python 3.13:
      157 nodes, OCIO configured, the RUDRA model fetched and applied on first
      run, the suite green there. VAE Encode (HDR) registered as the encoder
      VAE Decode (HDR) inverts; the two legacy HDR latent encoders labelled.
- [x] **RUDRA pixel model: no false colour, no over-peak channels (3.5.0).**
      From a user report on a clipped Flux.2 sunset. Recovered highlights keep
      the source colour (no rings, no red cast, source-level chroma noise),
      no channel exceeds `peak_nits`, the whole frame runs untiled when it
      fits, and every published checkpoint loads, with the shipped
      `sdr2hdr_shadow_v1` as the default and the auto-download.
- [x] **Documentation complete (3.5.0).** Every menu node has a description
      and every input a tooltip, written from the code and spot-checked by a
      separate reviewer (43 of 45 sampled tooltips accurate, the other two
      reworded). The node reference in `docs/nodes` is generated and tested
      for freshness; five example workflows were run from their saved files
      on a clean install. About 45 controls found not to match their names
      are listed in KNOWN_ISSUES for the next pass.
- [x] **Slow tests are opt-in (3.5.0).** The 1-to-100 video frame-count
      sweep ran on every `pytest` although it is marked `slow`; it is now
      deselected by default (`pytest -m slow` runs it, and CI runs it in its
      own step) and runs in parallel. Default suite about 68 s faster.
- [x] **Release feature test from the registry package (3.5.0).** Packed
      with `comfy node pack` (238 files, 3.5 MB, no tests or dev tools),
      installed into a clean ComfyUI 0.32 on Python 3.13 the way
      ComfyUI-Manager does it: 158 nodes, OCIO configured, the RUDRA model
      and MoGe-2 fetched on first use. Run through the API: SDR → HDR
      Universal to 32-bit EXR (HDR to 4.9, 19 % of pixels above 1.0) and the
      Viewer; Grade, CDL, colour-space convert, OCIO and tone map to 16-bit
      PNG; VAE Encode (HDR) to VAE Decode (HDR) with a real SD VAE (median
      error 0.09 stops, 96 % of over-range values kept); 8-frame H.264 and
      ProRes 422 HQ (8 frames each); Multipass Estimate to EXR passes, Read
      AOVs and Relight. In a headless browser every node creates, and the
      graph saves and reloads, with no Radiance console error. Three bugs
      this found are fixed (see the changelog).
- [x] **Release clean-up (3.5.0).** Removed 26 files (21, then 5 in a second pass) that nothing loaded,
      called or documented, including a stale offline manual that ComfyUI
      loaded as an extension on every page in git installs and a front-end
      extension for a node that does not exist. `tools/check_release_ready.py`
      is fixed and now runs in the suite, so version, node count, licence
      and packaging drift fail a test.
- [x] **Multipass passes are real or removed (3.5.0).** Multipass Extract's
      image-filter passes (Retinex albedo, blur-difference specular, contrast
      roughness, emission, transmission, reflection, k-means object ID) are
      gone. Multipass Estimate predicts geometry with MoGe-2 and materials and
      lighting with Marigold IID, computes GTAO and metric curvature from the
      geometry, and fits the lighting to the plate. Run on the real models on
      a photo: FOV, metric depth, normals, AO and curvature checked visually
      and on analytic scenes (plane, 90 degree crease, unit sphere), lighting
      rebuilds the plate to 11 percent RMS.
- [x] **Viewer and Lite Viewer, phase 1 (3.5.0).**
  - **Colour.** The node tags every frame: a ComfyUI IMAGE is shown exactly as ComfyUI shows it, and a linear source goes through OpenColorIO ACES 2.0. A normal image used to be read as linear and sRGB-encoded twice, which washed it out, and its input colour space was guessed from brightness.
  - **View menu.** Every entry is real (ACES 2.0 and 1.3 through OCIO, sRGB, Rec.709 BT.1886). The same view is baked into the PNG previews.
  - **Units.** Everything reads 203 nits for 1.0, and the DaVinci Intermediate curve matches the spec again.
  - **Compare and playback.** Compare works on WebGL and follows the playhead. Revisited frames no longer go black. Playback follows the source fps. Timecode is SMPTE, with drop-frame at 29.97 and 59.94.
  - **Export and transport.** The graded EXR is scene-linear and tagged with its primaries. Frames travel as half-float by default, and an unchanged viewer no longer re-runs everything downstream.
  - **Lite Viewer** (the node was removed later in 3.5.0, see below). Readout, clip check and diff use float source values. 1:1 is exact on scaled displays. It has play/loop.
- [x] **Cinematic Encoder, step 1 (3.5.0).** Long SDXL prompts keep their camera and lighting (no `BREAK`, no 77-token cut); Wan, Flux.2, Z-Image, Lumina2, Qwen-Image, AuraFlow and every other T5 / LLM encoder get prose instead of SDXL tags; the subject is never rewritten; Flux, Flux.2 and MiniMax skip the unused negative encode.
- [x] **Cinematic Encoder, step 2 (3.5.0).** No menu label reaches the encoder with brackets (they were 1.1x weights); prose reads as English; Wan / LTX keep their negative; "text" only on CLIP-only models; presets apply in API workflows; empty default prompt; token count ignores padding; one preset table, served to the JS. `tests/test_prompt_quality.py` (248); the functional suite runs the encoder with a stand-in CLIP. Node score 76 to 94.
- [x] **Color: controls that did nothing (3.5.0).** LUT / LUT Blend `log_space` encodes to the LUT's real input (camera log, ACEScct, PQ...) with a `working_space`; legacy Log10 / Log2 / Ln still run. HueCorrect gets a `grade_info` output and errors on a bad curve. OCIO Context is an output node that raises on a bad config and feeds Color Space Convert; `working_space` removed. `tests/test_color_controls.py` (18). Next sections of the list: HDR, Review, VFX, Generate, Video.
- [x] **Upscale: controls that did nothing (3.5.0).** Exact lanczos / lanczos4 / mitchell / catrom / hermite / gaussian on every path (GPU matmul, CPU, tiled; one weight builder), Downscale `antialiasing` is the prefilter width (0.5 = previous behaviour), Auto colour space detects HDR. `tests/test_upscale_kernels.py` (46).
- [x] **Viewer phase 2 (3.5.0).**
  - **Scopes.** Waveform, vectorscope and histogram measure the picture as displayed (graded, through the active view), not the ungraded texture or the 8-bit canvas. The vectorscope is BT.709 Cb/Cr with 75 % and 100 % targets and a skin line.
  - **Warnings.** False colour uses ARRI's bands on the Rec.709 signal. Clip and gamut warnings run before the display clamp, so they fire.
  - **Viewer-only exposure and gamma.** `f/` and `γ` in the viewer bar (and `-` / `=`, `0` to reset) change only what you see, never scopes, readout or export.
  - **Pixels.** The canvas is device-pixel sized; zoom above 1:1 is nearest-neighbour by default. Output is dithered, and a P3 monitor gets a Display P3 view and canvas.
  - **Keys and transport.** Keys go to the viewer under the pointer only. In/out (`I` / `O`), J/K/L shuttle, ping-pong and play-once, play every frame with a dropped-frame count.
- [x] **Viewer player checked end to end (3.5.0).** Load, paused seeks, fast scrub, arrow keys, Space, J/K/L, in/out loop, ping-pong, play-once, whole-clip wrap, Home/End and memory over a full loop, on MP4, a 240-frame clip and a PNG sequence, with the frame read back from the pixels. Fixed: playback freezing at the end of the range, the node growing without limit, J on a directly loaded video, and the software-backed 2D canvas. Playback fps on real hardware is still to be measured (the check machine renders WebGL in software).
- [x] **Viewer Simple / Advanced, one compare, Lite Viewer removed (3.5.0).** A switch in the title bar: Simple is picture, transport and compare; Advanced is every panel. Saved per node; graphs saved before it open in Advanced. The Lite Viewer node is deleted; a saved one is converted to a Viewer in Simple mode when the graph loads (before ComfyUI's missing-node check), links re-pointed by socket name. Compare is one controller (A, B, Wipe, Diff, Blink, pin and release) behind every button in both modes. Checked in a browser with B = the clip through ImageInvert, so every mode has a known picture: 29 of 29 checks, including B following the playhead, a pin surviving a new run, save and reload, and an old Lite Viewer graph running with its widget values in place.
- [x] **VFX efficiency, the severe six (3.5.0).** An audit of the 31 VFX nodes (report in the project: `vfx_efficiency_3.5.0.md`) found six that broke or crawled at production size. Fixed, each checked against the old maths: Motion Blur (the vector blur in the menu: +1 GB -> +0.48 GB, identical; the unregistered rival in film/camera.py, since deleted: out of memory -> 3.7 s), Multipass Relight (killed out of memory -> 4.0 s, output identical), Linear Matting (107 s -> 13.5 s, separable box filters, identical to 2e-6), HDR Stitch (23.6 s -> 3.1 s), Floyd-Steinberg dither (8.2 s a frame -> 0.06 s a frame in a batch, bit-identical), Depth Map Generator (batched, GPU preprocessing identical to the processor, fp16 on CUDA, depth no longer kept on the GPU). CPU timings; the GPU gains are larger and still to be measured on the 4080. `tests/test_vfx_efficiency.py` (28).
- [x] **VFX efficiency, phase 2 (3.5.0).** Film Grain, Depth of Field, Lens Distortion, Chromatic Aberration, Anamorphic Streaks, Subpixel Stabilizer, HDR Grain Matcher, Relight Engine, Multipass Composite and Optical Flow (DIS) work in chunks on the GPU through `core/tensor/chunking.py` (`FrameSink` keeps a one-chunk result without a copy). Checked against the old code (identical, or within 7e-6 for the stabilizer), and on 64 frames of 1024x576 peak memory falls by 10 % to 64 %. Lens Distortion's Invert crash fixed. The unloaded rival Film Grain and Motion Blur classes in `radiance.film` are deleted. API run of the nine image nodes chained on 8 frames: success. `tests/test_vfx_efficiency.py` (40).
- [x] **VFX efficiency, phase 3 (3.5.0).** EXR Passes Writer writes frames in parallel and converts passes per frame (files identical, 180 compared; 2.1x on 2 cores), Compression Artifacts encodes in parallel into the output (2.4x, half the memory, identical, same seeded noise) and no longer crashes on sizes that are not a multiple of `block_size`, Scene Cut Detect analyses each frame once (1.9x, bit-identical). `tests/test_vfx_efficiency.py` (43).
- [ ] **VFX efficiency, the rest.** Multipass Estimate's geometry passes and batched MoGe / Marigold (needs its models to test); GPU timings on the RTX 4080 for all three phases. See the project report.
- [x] **Sequential offload never engaged.** `setup_offload_mode("sequential")`
      called a `comfy.model_management.set_lowvram_mode` that ComfyUI never
      shipped, so it warned and did nothing on every run. It sets ComfyUI's
      `vram_state` now.

**Structural debt**

- [ ] **Split the remaining monoliths.** Largest first: `hdr/vae.py` (3527
      lines), `nodes/upscale/upscale.py` (3029), `image/upscale.py` (2741),
      `nodes/generate/sampler.py` (2111), `nodes/pipeline/workspace.py` (1913),
      `sampler_utils.py` (1897), `nodes/generate/prompt.py` (2166),
      `nodes/io/write.py` (1261), `nodes/monitor/viewer.py` (1233).

**Test coverage**

Current measurement: **67.49% including branches** on the full local suite.
The configured floor is 54%; the updated full CI job enforces it. Every node has structural coverage, though note what that does and does not mean: the smoke
tests check that the method named by `FUNCTION` exists, they do not call it.
Calling every registered node is `test_node_functional.py`'s job, and it now
runs in CI.

- [ ] **`image/upscale.py`, 1111 statements, 31%.** The one module that cannot
      be finished on CPU: what remains is `RadianceAIUpscale`, the SUPIR path,
      and the tiling code, all of which need model weights or a GPU. Newly
      registered in 3.4.0, so this is the first release in which its coverage
      counts for anything.

**Colour management**

- [x] **Sampler speed (3.5.0).** No second model load per stage, no silent
      cfg 1.0 → base CFG (the extra unconditional pass that doubled turbo
      runs), no cfg boost at 1.0 in Dynamic CFG, no gc / cache flush before
      sampling, no debug statistics with DEBUG off. Flux with an empty
      `clip_l` no longer falls back to Mochi's T5 encoder, and the Loader
      reports when the weights cannot stay in VRAM.
- [x] **OCIO is configured automatically (3.5.0).** OpenColorIO is a
      required dependency (installed by ComfyUI-Manager via `requirements.txt`,
      or by `install.py`). At startup Radiance uses `$OCIO` when you have one
      and otherwise OpenColorIO's built-in ACES 2.0 studio config (55
      colorspaces: every ACES space, the major camera logs, Rec.709 / Rec.2020
      / P3 / PQ / HLG), written to `ACES/studio-config.ocio`, exported as
      `$OCIO` for the process and made OCIO's current config, so every node
      resolves the same names. Nothing is downloaded (the old startup fetch
      from GitHub is gone). `RadianceColorSpaceConvert` now maps its names to
      that config (the old targets existed in no config, so OCIO never ran)
      and `RadianceHDROCIOTransform`'s defaults resolve. OpenCV's EXR codec is
      forced on (`OPENCV_IO_ENABLE_OPENEXR=1`) before anything imports cv2.


### 2026-09-26 — full-refresh candidate and release-gate review

- Rejected moving-window and both first-frame-anchor experiments after real
  121-frame visual review. Restored the prior sampler; saved rejected sources
  as non-importable scratch text. A 61-frame stock control was coherent.
- Isolated full stochastic refresh above sigma 0.8, Euler below, and half-window
  overlap produced clean sampled joins on the 121-frame sailboat test. Longer
  and different-scene acceptance remains pending. Native LCM then Euler matched
  the diagnostic numerical trajectory; no production sampler was added.
- Clean full suite: 4,695 passed, 91 skipped, 3 slow deselected, 28 subtests;
  68.07% branch-inclusive coverage. Separate slow suite: 3 passed. JavaScript:
  324 passed, 5 skipped. The preceding failed run was invalidated by importable
  scratch backups and an incompatible CUDA visibility test setting.
- Fixed publish installation of nonexistent Imath>=3.1.0 (OpenEXR supplies its
  binding), completed publish-test dependencies, separated CI failure artifacts
  by OS, and selected the full public-repository matrix by repository visibility.


### 2026-09-26 — merged release checks and video acceptance results

- Merged main through 2a55b5d. The resulting code, 842317b, passed all 17
  hosted CI jobs on both the branch and draft PR #89. Local validation:
  4,698 Python tests passed, 91 skipped, 3 slow deselected, 28 subtests;
  68.07% branch coverage. JavaScript: 324 passed, 5 skipped.
- Full-refresh boat samples were coherent at 121 and 241 frames, but train
  windows failed at both 16 and 31 latent frames. The 121-frame unsplit train
  control was coherent. Native causal windows also changed train geometry.
  The 241-frame low-resolution global control had severe texture artifacts;
  no refinement was attempted. All research sampler changes remain excluded.
- Nuke, Resolve and release CI gates are closed. Long-video visual acceptance
  remains open; draft review and publication cannot be completed honestly yet.
