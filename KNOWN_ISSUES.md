# Known Issues — Radiance 3.5

Tracked limitations and tech debt. Full production approval remains open: the
video visual-acceptance failure below blocks production approval of temporal windowing. Other limitations
apply to the specific workflows and backends described here.

## Architecture / tech debt

- **Route registration keeps an idempotency guard.** aiohttp routes register
  through `_radiance_route_once`, which records what it has already registered
  on the PromptServer singleton, so a module imported twice cannot crash
  startup with "method HEAD is already registered". The dual-module structure
  that made this necessary is gone — the legacy `nodes_*.py` layer was removed
  in 3.3.0 and there is one entry point now — but ComfyUI puts `custom_nodes/`
  on `sys.path`, so a user script doing `import viewer` can still produce a
  second copy of a module. One file (`nodes/monitor/viewer.py`) carries the
  guard; two more read the same registry. Kept as cheap insurance, no longer a
  symptom of anything.

- **Masks and Qualifiers are inert on the WebGPU backend.** `js/radiance_webgpu.js`
  carries no WGSL for either one. The shared base class, `js/radiance_renderer.js`,
  implements `setMask` and `setQualifier` and stores the state, so the Viewer's
  calls succeed and nothing changes on screen: both tabs are fully interactive
  and completely inert on that backend. WebGPU is opt-in since 3.4.0
  (`localStorage.radiance_prefer_webgpu = "1"`); WebGL, the default, renders
  both.
  OCIO has the same WGSL gap and at least says so, returning "OpenColorIO needs
  the WebGL backend; this renderer has no WGSL path for it"; these two say
  nothing. *Planned fix:* none scheduled. The two resolutions are porting the
  shaders to WGSL, or disabling the tabs on WebGPU with that message. The gap is
  listed in `js/tests/backend_parity.test.mjs`, which fails if this entry and
  the shipped source stop agreeing, so neither resolution can land without
  updating this list.
  Beyond masks, qualifiers and OCIO (3.5.0 audit), the WebGPU backend also
  differs from WebGL in: no sRGB decode for 8-bit input; a 3D LUT read with the
  wrong stride; the red curve applied to every channel; a transposed hue
  matrix; misregistered bloom; and no heatmap, gamut/clip warnings, scope
  signal, viewer f-stop or scene-linear graded EXR, and compare shows only
  wipe there (B, Difference and Blink need `setCompareShow`, WebGL only).
  `FEATURE_PARITY` now says
  false. Porting these to WGSL is the remaining work; until then WebGL is the
  default and the tested path.

- **Monolithic files.** `hdr/vae.py` (3324 lines), `nodes/io/write.py` (2972)
  and `nodes/monitor/viewer.py` (1220, with dynamic `globals().update()`
  injection) carry several responsibilities each and are the hardest files to
  change safely. The `nodes/io/write.py` split is also what unblocks
  `delivery/handler.py`, which still reaches up into the node layer for
  `RadianceWrite`. Tracked in the README's Status & to-do list.

- **Learned SDR → HDR is one pixel model.** The latent-space RUDRA decoders
  (per-model `rudra_turbo_decoder_*` / `rudra_full_decoder_*` checkpoints,
  the video decoders trained on stills, and the truncated `ltx-video` full
  decoder) were retired in 3.5.0. `SDR → HDR Universal` and `Recover` run the
  pixel model (`sdr2hdr_shadow_v1.safetensors`) on any image or frame batch and the
  temporal residual model on ordered video when its checkpoint is installed.
  The pixel model processes video frames independently; a clip may need
  downstream deflicker until the temporal checkpoint ships.

## Placeholders and labelled limits (3.5.0 honest release pass)

- **SAM Loader / SAM Mask Generator are not shipped.** Radiance bundles no
  SAM runtime; the generator used to return discs drawn around the click
  points. Both nodes are hidden from the menu (`DEPRECATED`), still load in
  saved graphs, and raise a clear error when executed. *Planned fix:* a real
  SAM2 backend, or removal in 4.0.
- **Upscale `confidence` output is geometric.** 1 at tile centres, lower
  toward tile edges. It locates seam blending; it does not measure
  hallucination, which none of the backends report.
- **Upscale Video is windowed, not temporal.** Tier 1/2 models upscale each
  frame independently; overlap frames are flow-aligned and blended at window
  seams. GAN flicker between frames is not removed.
- **`cfg_schedule_json` is a static override.** T2V, I2V and Video Sampler use
  its first value as CFG. A per-step schedule needs a sampler CFG hook and is
  not implemented.
- **Video HDR Conditioner needs `clip`.** Without it the conditioning passes
  through unchanged and `hdr_metadata_json.applied_to_model` is false.
- **I2V on Wan 2.2 TI2V 5B uses `first_frame_lock`.** That model has no
  image-concat input; ComfyUI's own path for it is `Wan22ImageToVideoLatent`
  (image in the latent with a noise mask), which the pipeline does not yet
  reproduce.
- **Audio Cut `method` is honoured by librosa only.** scipy finds energy
  onsets and ffmpeg finds silence ends whatever the method; the report says
  which ran. `max_cuts` keeps an evenly spaced subset (no strength is
  reported by any backend).
- **Camera Sync reads JSON only.** Alembic (`.abc`) raises with a message.
- **HDR Synthesis Engine does not reconstruct clipped detail.** It lifts the
  low-pass base toward `energy_target`; `recovery_iters` is the pyramid
  depth. Learned recovery is SDR → HDR Universal / Recover.
- **Relight Engine adds one directional light.** Lambert + Blinn-Phong added
  to the image, not an albedo relight, and no environment map.
- **SDR → HDR Expand keeps SDR midtones at their SDR display level.** Below
  the knee the signal stays at 1.0 = 100 nits (18 % grey at 18 nits) and only
  the knee-to-white span opens up to reference white; BT.2446 Method B would
  also lift midtones by ~2x. Raise `reference_white_nits` or grade after for a
  brighter mid-scale.
- **Pixel SDR → HDR runs frame by frame.** Ordered video gets the temporal
  model only when its checkpoint is installed; otherwise downstream deflicker
  may be needed. `pixel_tile_size` changes the result slightly (each tile sees
  less context); keep it at 512 or above. On CPU a 1080p frame takes ~19 s
  (0.16 s for Expand); GPU timing is still to be measured on the 4080.
- **HDR through a VAE is lossy in the highlights.** VAE Encode (HDR) and VAE
  Decode (HDR) invert each other exactly, but the VAE between them does not:
  the log-coded latent's small errors grow as the curve steepens. Measured
  with the SD 1.5 VAE on a plate peaking at 8.4: median error 0.09 stops,
  96 percent of values above 1.0 stay above 1.0, and the brightest speculars
  can overshoot (to 24 on that plate). Midtones hold; specular peaks are not
  reliable after a VAE pass. *Planned fix:* none in the codec; it is the
  VAE's reconstruction.
- **Multipass Estimate passes are predictions from one image.** MoGe-2's
  metric scale is an estimate (typically within 10 to 20 percent indoors), and
  so is the field of view unless `fov_x_degrees` is given. Marigold was
  trained mostly on synthetic interiors: it can call painted walls slightly
  metallic, and it sees values above 1.0 clipped, so on HDR plates the
  lighting fit leaves clipped pixels out and `specular_lighting` under-reads
  the brightest highlights. The lighting passes add back up to the plate only
  approximately; `info` gives the error per frame (about 10 percent RMS on a
  clean photo, much worse on a clipped sunset). Video is estimated frame by
  frame; a fixed seed limits flicker but does not remove it. AO is screen
  space: nothing outside the frame or behind the visible surface occludes.
  *Planned fix:* temporal models when they are released; until then, render
  real passes where they exist.
- **HLG is referenced to a 1000-nit display.** BT.2100 / BT.2408 / OCIO
  convention: highlights mastered above 1000 nits clip in the HLG signal. Use
  PQ for 4000- and 10000-nit masters.

## VFX nodes: memory and speed (3.5.0)

The six nodes that broke or crawled at production size, ten more that held the clip several times over, and the EXR Passes Writer, Compression Artifacts and Scene Cut Detect are fixed (see the changelog). Still to convert: Multipass Estimate's geometry passes (normals, GTAO, curvature) hold the whole clip on the CPU and its models run one frame at a time; it needs its models to test. Multipass Composite returns four full-size images, so with every input connected a long clip needs about eleven clip-sized buffers in RAM; split long 4K clips for it. RTX 4080 SUPER follow-up verified CPU/GPU agreement for matting and standard stitching and measured their single-frame 1080p latency (see docs/DEVELOPMENT.md). GPU timings for the other operations and long clips remain unverified.

## Found while documenting every input (3.5.0)

Writing a tooltip for all 1,259 inputs meant reading the code behind each
one. The bugs it found are fixed (see the changelog, "Bugs found while
documenting every input"). These controls still do not do what their name
says; each tooltip says what really happens.

**Controls that do nothing or less than their name**

- Do nothing: Sampler `conditioning_clip_target`; Bit Depth Degrade
  `restore_from_quantized`.
- Only label a report: ACES Compliance `output_type` and `peak_nits`; Color
  Space Info `scene_referred` and `peak_nits`.
- Narrower than named: `creative_white_scale` is a linear gain;
  CFG++ "(Perpendicular)" is a cosine cfg
  scale.
- Denoise: `motion_compensation` is a ±1 px search.

*Planned fix:* one pass that either implements each control or removes it,
with a test per item, section by section. Done so far: Color (LUT and LUT
Blend `log_space`, HueCurves `grade_info`, OCIO Context) and Upscale (resampling
kernels, Downscale `antialiasing`, Auto colour space). The production pass also
fixed ClipDetector channel selection with soft edges, ExposureBlend weighting
and batch preservation, and HDR Monitor’s gamma switch for Exposure + Gamma.
Video prompt controls now describe every offered gamut, transfer function and
peak in both T2V and I2V. These are look descriptors, not pixel mastering.
Synthesis chroma preservation now blends neutral highlight energy with
proportional RGB energy at the same luminance.
Standard inpaint stitching now uses the feathered mask in its composite.
FrameStamp rasterises only its overlay and composites onto float pixels,
preserving the source alpha and unclipped values outside the overlay.
False Colour now decodes its selected sRGB or Rec.709 input encoding before
linear-luminance measurements, preserves alpha and applies strength to zebra
stripes. Its default remains linear Rec.709; other primaries must be converted
upstream and the diagnostic output is SDR.
Dynamic sampler guidance now includes ramp-end stage boundaries so runs with
enough steps reach the middle and late targets. Short or partial runs may
omit phases; the schedule is evaluated at stage boundaries.
Guided denoising now uses `sigmaSpace` for Gaussian-weighted local statistics;
truncated windows are normalized at image edges and HDR values remain float.
Detail recovery restores only the high-frequency removed residual, retaining
low-frequency denoising. Fine noise can return along with texture.
Cinema D60 now converts AP1 into P3 at ACES white (approximately D60),
distinct from the D65 output conversion. Select the matching display calibration.
Linear Matting now constructs an unknown trimap band using the dilation
radius, preserves known foreground/background regions and refines the band.

## Viewer: remaining (phase 3)

- **No HDR output.** The canvas is SDR (sRGB or Display P3, 8-bit, dithered). A Rec.2100 PQ / HLG canvas needs a float16 HDR drawing buffer in both the WebGL and the 2D compositing canvas; browser support is still settling and it cannot be verified without an HDR display. Use HDR Monitor or an external HDR display for HDR review.
- **Compare B side.** It is the node's display-referred preview, baked with ACES 2.0 or untouched for sRGB. Switching the A side to another view does not re-render B. A B of a different size is stretched to A. Side-by-side is 2D-fallback only.
- **Blink rate is fixed** at two flips a second; there is no control for it yet.
- **Wide-gamut display backend scope.** WebGL now converts tagged ACEScg, ACES2065-1, linear Rec.2020 and linear P3-D65 into Rec.709 before the built-in sRGB, Rec.709 and Filmic views. OCIO handles its own source conversion; linear EXR exports retain source primaries. This correction applies only to the WebGL backend; other backends remain unverified for these source gamuts.
- **Annotations are screen-space.** They do not follow pan and zoom.
- **Reverse video is seeked, not played.** J on a video loaded straight into the Viewer steps back one seek at a time. On long-GOP files (typical H.264) each seek decodes from the previous keyframe, so reverse can run below the set rate. Image sequences and clips from a Read node are not affected.
- **Playback rate not yet measured on a GPU.** The player's correctness is checked in a browser with software WebGL, which tops out near 3 fps; real-time rate at 24 fps and above is to be confirmed on the RTX 4080 before tagging.
- **No audio.** Playback is picture only.
- **Pro scope "False colour" mode** is a display-level zone map, not the ARRI bands of the viewer overlay.

## Models that cannot download on their own (3.5.0)

Every other model downloads on first use (see the README, "Models every other
node downloads").

- **HAT-L** (Upscale Tier 2) is published only on Google Drive, so there is no
  pinned download. Install it by hand from the HAT page; until then Tier 2
  uses SwinIR-L at 4x and Real-ESRGAN at 2x, and says so in the log.
- **Gated repositories** (FLUX.2-dev, FLUX.2-klein 9B and base 9B, LTX-2.5)
  download only after the licence is accepted on Hugging Face and `HF_TOKEN`
  is set; the node stops with both steps until then.
- **SeedVR2** needs the ComfyUI-SeedVR2_VideoUpscaler node pack, which fetches
  its own weights; without it Tier 3 uses the SD x4 upscaler.

## Minor

- **Three legacy nodes are hidden, not removed.** HDR Latent Encoder and HDR
  Turbo Encoder stop with a message naming VAE Encode (HDR), because their
  decoders were retired and their latents would render clipped. ACES 2.0
  Output Transform (Legacy) still works. All three stay registered only so
  saved graphs open. *Planned:* remove them in 3.6.
- **Learned highlight recovery restores brightness, not colour.** RUDRA's
  released SDR → HDR checkpoints (v5, shadow_v1 and seeds) were trained on
  SDR rendered at -1 EV, which almost never clipped (median clipped fraction
  0.000%). In blown areas their per-channel output is outside what they
  learned, so Radiance takes only the learned luminance there, keeps the
  source colour, and fades it in as the source goes to white. A highlight
  clipped in one channel only (a saturated red light) gets the deterministic
  expansion, not a learned guess. *Planned fix:* a RUDRA checkpoint trained
  on RUDRA's clipping corpus (0 EV render), after which the colour can be
  handed back to the model.
- **The pixel model has a hard edge where the clip mask starts.** The
  highlight mask ramps over SDR codes `highlight_threshold`..1.0 (0.98 by
  default), so on noisy or dithered 8-bit input the mask edge speckles.
  Lowering `highlight_threshold` to about 0.9 widens the ramp.

- **`RadianceSDRToHDRUniversal`'s learned modes need the RUDRA checkpoint,
  which now downloads itself (3.5.0).** On first use, with no
  RUDRA checkpoint installed, Radiance fetches `sdr2hdr_shadow_v1` (~5 MB, pinned
  commit, SHA-256 checked) into `models/radiance`. What remains: on a machine
  with `RADIANCE_ALLOW_DOWNLOADS=0`, `HF_HUB_OFFLINE=1`, no network, or no
  writable models folder, `Recover` and `Hybrid` still produce output
  bit-identical to `Expand`. That is a safe fallback; the node's `report`
  output says which path ran and why, the console logs the manual download
  link, and the dedicated `Recover` node raises instead. A failed download is
  tried once per ComfyUI session; restart to try again.


- **Naming overlap:** `Grade` / `Grade Apply` / `Apply Grade Info` read similarly;
  to be clarified during the Color cleanup.

- **Scene-cut `edge` method is weak on soft and grainy footage.** It compares
  gradient magnitudes, so it cannot see a cut between two frames that both lack
  edges — a soft gradient cutting to a different soft gradient, or black
  cutting to white — and heavy grain still moves it more than a subtle cut
  does, because a gradient operator is a high-pass filter and grain is
  high-frequency. The 5 px pre-blur reduces that but does not remove it. This
  is why `combined` is the default and is weighted 60/40 toward the histogram.
  Use `histogram` alone on soft material; use `edge` alone only for the case
  the histogram cannot see — the palette held, the framing changed. *Planned
  fix:* none scheduled. The two methods cover each other's blind spots, and a
  metric that handled both would be a different detector — feature matching or
  a learned embedding — not a tuned constant.

## Environment notes (not bugs)

- **GPU optimized kernels** prefer a `cu130` PyTorch build; on `cu128` the
  `comfy_kitchen` CUDA backend falls back to "eager" (works, slightly slower fp8/fp4).
- **`Imath`** is provided by the `OpenEXR` wheel — do not add a separate `Imath`
  dependency (PyPI `imath` maxes at 0.0.2).

## Workstation DCC acceptance (2026-09-26)

Nuke 17.1v2 acceptance now passes using the installed interactive license
(`-i -t`); a render-only terminal launch does not have a matching license.
The real signed handoff created a Read node for frames 1001-1003 and preserved
scene-linear values 0.18, 1.0 and 4.0. The listener now avoids GUI-only calls
in terminal mode and keeps node edits on its main thread. GUI event dispatch
is covered by regression tests; the live test used terminal mode.
Resolve's scripting DLL crash (0xC0000005) was resolved by setting
PYTHONHOME to the worker's running Python installation before loading the DLL.
This is scoped to the isolated worker. Live Resolve Studio acceptance imported
one EXR still and a 121-frame EXR sequence into a separate test project;
the Media Pool reported 1 and 121 frames respectively. Colour/timeline round
trips are outside these import checks.

## Video visual acceptance (2026-09-26)

**Production blocker:** at the official-size 1280x704, 121-frame, 20-step
settings, stock KSampler produced a coherent sailboat while Radiance Sampler
Pro with temporal_window=16 and overlap=4 produced severe artefacts. Prompt
`372ba47d-77e0-49ec-a47e-7bde61707fed` completed numerically, but failed visual
review. The planner reduced this 31-latent-frame case to two windows sharing
only one frame. Maximum adjacent-frame mean absolute difference was 0.1133
near the join, versus 0.0261 for stock. Neither sampler nor windowing quality
is approved by this result. The same-size unwindowed path also failed visual
review (prompt `4174a015-5dd5-4adb-b3a5-029a21eb611c`), so windowing alone
is not the cause. Auto/Custom's redundant extra schedule shift has been
corrected. A native-schedule unwindowed comparison restored coherent imagery,
but the windowed run still changed scene near the join (maximum adjacent-frame
mean difference 0.07024 at frame 62). An experiment preserving overlap with
a shorter final window produced ghosting and severe tail artefacts, so it was
reverted. The current full-size planner keeps the final overlap and uses
three-way complementary fades. This removed the large numerical jump
(maximum mean frame difference 0.02927) but still showed join ghosting in
the 16-frame-window test. Windowed-video acceptance remains open; this experimental feature is disabled by default.
The 241-frame test with 31-frame windows also failed visual review, showing
doubled boats/backgrounds around frames 97-121. Larger context alone does
not fix coherence. An early-stochastic/late-deterministic diagnostic completed
with 121 finite frames and maximum mean adjacent difference 0.02000, but the
boat, framing and background still changed across the join. It also failed
visual acceptance. Tooltips and an execution warning now recommend
`temporal_window=0` for production; valid files and smooth numerical statistics
do not establish visual continuity.

A separate shuffled-initial-noise diagnostic also failed: its 121 finite
frames still showed a scene transition and severe final-frame artefacts.
This experiment was not added to the product. A different windowing strategy
and real visual acceptance are still required before approving this feature.

The real 129-frame Wan 2.2 test completed and exported valid EXRs, but its
256x256, eight-step images showed severe artefacts. Stock ComfyUI and
unwindowed Radiance comparisons at those settings also failed visual review.
This is not evidence of usable video quality or seam-free windowing; a
stock run at the official template settings (1280x704, 121 frames, 20 steps)
has since produced coherent sampled images. Radiance must now be compared
at those same settings before its temporal quality can be approved.
