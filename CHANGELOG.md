# Changelog

All notable changes to FXTD Radiance will be documented in this file.

## [4.0.0] - Unreleased (beta, branch version-4-Beta)

A major release because saved graphs and studio scripts can see it: four
hidden placeholder nodes are gone, delivered file names change, a remote DCC
bridge needs signed requests, and three controls that did nothing now work. Everything
else is fixes. Read the upgrade notes before moving a production setup.

### Upgrade notes

- **Removed nodes.** SAM Loader, SAM Mask Generator, HDR Latent Encoder and
  HDR Turbo Encoder were hidden placeholders that raised when run. A saved
  graph that still holds one loads with that node missing; delete it. Use
  ComfyUI's SAM nodes and the HDR VAE Encode node instead. ACES 2.0 Output
  Transform (Legacy) stays, hidden.
- **Delivery file names** carry one version suffix: `Shot_v0002.mov`, where
  3.x wrote `Shot_v02_v0001.mov`. The counter continues from 3.x names in the
  same folder. Smart versioning off writes `Shot_v0001`. The reported version
  is four digits too (`v0002`).
- **DCC bridge on a network address** (`RADIANCE_ALLOW_REMOTE_BRIDGE=1`):
  `queue` must be signed with the shared DCC token from
  `RADIANCE_DCC_AUTH_TOKEN` or `~/.radiance/dcc_token` (the one the Nuke
  listener already uses): `ts`, a single-use `nonce` and `sig`, as
  `radiance.core.dcc_auth.sign_queue` builds them. The token itself is never
  sent. Loopback, `ping` and `status` are unchanged.
- **The DCC bridge refuses HTTP.** A connection whose first line is an HTTP
  request is closed, so a web page can no longer queue a prompt through a
  loopback bridge with `fetch()`.
- **Controls that now act:** Bit Depth Degrade `restore_from_quantized`,
  Sampler CFG++ (Perpendicular), and Denoise `motion_compensation` (only with
  `temporal_blend` above 0, which is not the default). A saved graph that set
  them gets a different result from 3.x, which ignored or under-did them.
- **One default OCIO config.** The Write/Read colour options, the HDR OCIO
  nodes and the OCIO manager now all use `$OCIO` when it loads, else the ACES
  studio config Radiance sets up. Before, two of them fell back to the bundled
  CG config, whose colour space names differ.
- **HDR video needs 10 bits.** PQ or HLG with "MP4 (H.264)" or "MOV (DNxHR
  HQ)" now stops with an error naming H.265 10-bit, ProRes 422 HQ and ProRes
  4444; before, it wrote an 8-bit file tagged as HDR. PQ H.265 files are now
  HDR10: P3-D65 1000-nit mastering display and MaxCLL / MaxFALL measured from
  the frames.
- **Camera log input** (`apply_input_transform`, used by HDR Analysis on log
  and ACEScct) now converts the camera gamut to Rec.709 primaries after the
  curve; it decoded the curve only. Neutrals are unchanged, saturated colours
  and the measured peak and clipping on saturated log content change.
- **Installed upscale and face models are checked once.** The first run after
  upgrading hashes each one against its pinned SHA-256 (records the pass in
  `<file>.radiance-sha256`, tied to the file's inode and change time). A file
  that does not match is moved to `<file>.radiance-mismatch` and the pinned
  one downloaded, or reported when downloads are off; it is never used
  silently or overwritten.
- **Nuke push** reports `UNCONFIRMED` when Nuke does not reply before the read
  timeout, drops the connection, or is busy (the listener now answers
  `PENDING` after 10 s); it reported `OK`, or `FAILED` for a busy Nuke whose
  command still ran. Update `scripts/start_nuke_server.py` in Nuke too.
- **Downloads.** The ACES config manager's download needs download consent
  (`RADIANCE_ALLOW_DOWNLOADS=0` refuses it) and is checked against a pinned
  SHA-256. Whisper model downloads ask for consent too (weights already in
  whisper's cache, including `large` as `large-v3.pt`, need none).

### Removed

- SAM Loader, SAM Mask Generator, HDR Latent Encoder, HDR Turbo Encoder (see
  above). The node count is 152.

### Changed

- **CFG++ (Perpendicular)** keeps only the part of (cond - uncond)
  orthogonal to the conditional prediction, per batch item (projected
  guidance). It used to be only a cosine cfg schedule per stage, which stays.
  It applies on the refiner's steps too. A cfg function another patch set
  (LTX-AV `audio_cfg`) is left in place.
- **Bit Depth Degrade `restore_from_quantized`** dequantises: eight 3x3
  smoothing passes, each clamped to the values the pixel could have come
  from. The other outputs and the metrics then measure the restored image.
- **Denoise `motion_compensation`** is hierarchical block matching (8x8
  blocks, ±4 at the coarsest level, about ±30 px per frame on large frames)
  with an edge-repeating warp, instead of the best of nine 1-pixel offsets
  with wrap-around.
- **Relabelled, by design:** ACES Compliance `peak_nits` and the Legacy
  output transform's `creative_white_scale` say exactly what they do. Sampler
  `conditioning_clip_target`, Color Space Info `scene_referred` and
  `peak_nits` were already labelled truthfully. KNOWN_ISSUES lists all seven
  as resolved.
- **`hdr/vae.py` split, first slice:** `TileEngine` moved to
  `hdr/vae_tiling.py` unchanged; `hdr/vae.py` re-exports it.
- **`.rhdr` sidecars** are written by one encoder, `core/rhdr.py`, shared by
  the Viewer (fp16, fp32, depth) and the HDR VAE export.

### Fixed

- **SDR reference conditioning** crashed with every real VAE (it expected a
  dict from `vae.encode`). It takes a tensor or a dict, encodes RGB only, and
  warns when a video VAE drops reference frames.
- **Delivery** reports status "partial" with a warning when its 2x AI upscale
  fell back to bicubic.
- **Flux.2** is detected as `flux2` when `model_meta` is not connected
  (longest config pattern first).
- **Project Manager** saves graphs the library can reopen and parse; shot and
  version names split on any non-alphanumeric boundary.
- **Asset upload** keeps an existing file (the new one becomes `name_1.ext`)
  and removes its partial file when aborted.
- **`/radiance/media/*`** answers the same for a missing file and one outside
  the allowed roots, so it no longer reveals which files exist.
- **Shot status** cannot be written outside the workflow library.
- **Video encodes** go to a hidden sibling file that replaces the output only
  on success, so a failed, timed-out or cancelled encode leaves any previous
  master intact; masters keep normal file permissions and overwrite off never
  clobbers.
- **A video written from a stream of unknown length** kept only as many
  frames as its audio was long; the audio is now padded instead.
- **fp16 `.rhdr` writes** clamp to ±65504 everywhere (the VAE export and the
  depth sidecar wrote inf), and the VAE export's path guard works again.
- **MoGe-2** is checked against its pinned SHA-256 after download, and each
  **Marigold** file against the Hub's hash at the pinned commit; a mismatch
  deletes the file. A Marigold download whose check could not run (rate
  limit, network) is checked again on the next run instead of being used.
- **Reading:**
  - Image sequences report the frame rate their EXR, DPX or Cineon header
    declares (24 only as a stated fallback; DPX's float rates snap, so 23.976
    is 24000/1001); every sequence said 24.
  - Video reads work with only ffmpeg installed (imageio-ffmpeg ships no
    ffprobe): the stream is probed from ffmpeg's banner.
  - `raw` on a video skips the colour-tag decode and any OCIO override, as it
    does for images.
  - Formats only OpenImageIO reads (Cineon, RLA, IFF, ARRIRAW and others) go
    to OpenImageIO instead of failing in Pillow, with an install hint when it
    is missing; `.pic` (Radiance RGBE) is read as float HDR.
- **RGBA:** Tier 2 upscale, SeedVR2 and an external `UPSCALE_MODEL` carry alpha
  (resized to the output; the model sees RGB), Face Restore keeps the input
  alpha (it wrote the face's red into alpha), Focus Peaking keeps alpha, and
  Split View, Contact Sheet, Flipbook GIF and Preview Server accept RGBA and
  show RGB (they raised, or the GIF's colours were scrambled). Video Assembler
  joins RGB and RGBA frames (TEN-007).
- **AI Upscale** honours `RADIANCE_UPSCALE_OFFLINE` and says why a model was
  not downloaded.
- **DCC Bridge:** export from an image sequence works (it always failed on an
  unknown colour space name); at most 16 connections are served at once (8 per
  address), a request line must arrive within 30 s, and the server stops
  when ComfyUI exits.
- **Audio Transcribe:** chunks of a split segment each get their own start
  and end; they all started at the segment start.
- **Startup** ends on a warning naming any missing required dependency, and
  a failure inside the dependency check no longer stops Radiance loading.
- **OCIO:** a broken `$OCIO` falls back the same way everywhere (the ACES
  Config Manager's Detect included), a config set after the first use reaches
  the writer, an edited config file is read again, and the OCIO manager
  accepts an `ocio://` URI (used when the package folder is read-only).

### Known limits

- Sequential offload still switches ComfyUI to LOW_VRAM for the rest of the
  session (P2-4); nothing Radiance can hook is scoped to one prompt.
- The multipass model registry (Depth Anything V2, DSINE) downloads from
  `main` without a pinned hash; it is consent-gated and logs the digest.
- GPU timings and DCC round trips need hardware and were not run for this
  beta.

## [3.5.4] - 2026-10-02

Registry release. 3.5.0, 3.5.2 and 3.5.3 were published but held as
"flagged" by the Comfy Registry security scan, so the registry kept serving
2.3.3. No node behaviour changes; this release removes what the scan flagged.

### Changed

- **Environment variables:** `OPENCV_IO_ENABLE_OPENEXR` and
  `KMP_DUPLICATE_LIB_OK` are set in one place only,
  `configure_runtime_environment()` in `config/env.py`, at package init. The
  duplicate writes in `__init__.py`, the Viewer and `image/upscale.py` are
  gone. `KMP_DUPLICATE_LIB_OK` is a default, so a value you export wins.
- **ACES config download** no longer rewrites `$OCIO` for the whole ComfyUI
  session. The node returns the config path; wire it into OCIO Context, or
  export `OCIO` before launch to make it the default.
- **Resolve import:** `PYTHONHOME` is passed only to the Resolve worker
  process on Windows, not set inside it.
- **Nuke bridge hint:** the "Nuke not reachable" message now suggests
  `import runpy; runpy.run_path('<path>/start_nuke_server.py')` instead of
  `exec(open(...).read())`.
- **DSINE manual download** link points at the Hugging Face checkpoint
  (`baegwangbin/DSINE`) the auto-downloader already uses, not Google Drive.
- Three inline `__import__()` calls replaced with normal imports.

## [3.5.3] - 2026-10-02

Maintenance release: the P0 correctness fixes from the 29 September 2026
audit (FIX-001 to FIX-018). Every fix has a regression test in
`tests/test_p0_*.py`; 51 of those tests fail on 3.5.2.

### Upgrade notes

- OpenColorIO 2.5 or newer is required (`opencolorio>=2.5.0`). The ACES 2.0
  Output Transform runs the Academy reference from OCIO's built-in
  `studio-config-v4.0.0_aces-v2.0_ocio-v2.5`.
- ACES 2.0 Tonescale: new nodes default to the ACES 2.0 reference tone
  scale (`grey_target` 0). Saved workflows keep their stored 0.10, which is
  now labelled a creative curve in `curve_info`; set it to 0 for the
  reference.
- EXR Multi-Part is strict by default: an incomplete file stops the node.
  Turn `strict` off to get the per-part fallback, listed in the new
  `manifest` output.
- CDL Import raises on a missing or unreadable file instead of returning an
  identity grade.

### Fixed

- **ACES 2.0** (FIX-001, 002, 003): the Tonescale node defaulted to grey at
  10% of peak (100 nits on a 1000-nit master); it now runs the ACES 2.0
  tone scale, matching OCIO to 1e-3 nits. The Output Transform runs the OCIO
  reference (`engine` Auto / OCIO reference / Radiance approximation; the
  approximation is labelled). "S-2126 Compliance" is now "ACES 2.0 Output
  Check", a per-frame comparison with the reference (node ID unchanged).
- **White Balance** (FIX-004, 005): accepts its `ocio_context` input; the
  Bradford adaptation runs in XYZ from Rec.709 RGB; 6500 K is D65 (daylight
  locus), so the default is neutral; alpha passes through.
- **ASC CDL** (FIX-006): one writer for CDL Export, the delivery sidecar and
  the AMF, producing standard `.cdl` / `.cc` / `.ccc` documents in
  `urn:ASC:CDL:v1.01`, read back by OpenColorIO in the tests. The Viewer
  grade maps to SOP correctly (power = 1/gamma, offset x gain, exposure in
  slope); the Viewer's CDL import is the inverse of its export.
- **Policy Guard and QC** (FIX-007): NaN/Inf frames, malformed or
  unloadable policies and invalid tensors fail instead of passing; strict
  QC (`fail_on_errors`) also stops on invalid input and analysis errors.
- **Alpha** (FIX-008): more than 20 colour, exposure and normalisation nodes leave
  alpha untouched; several of them used to raise on RGBA.
- **Batches** (FIX-009): HDR Shadow / Highlight Recovery, HDR 360 Generate
  and the legacy ACES 2.0 Output Transform process every frame.
- **Inputs** (FIX-010): Float32 Convert's normalise no longer divides the
  upstream tensor in place.
- **Upscale** (FIX-011, 012, 013): the 8x confidence map matches the image;
  Upscale Video's 8x mode no longer raises, its temporal windows blend the
  same source frame, and flow compensation (default on) no longer raises on
  multi-window clips; the Tiler and the pre-denoise no longer clip HDR.
- **Bit Depth Convert** (FIX-014): 16-bit Float with dithering stays float
  instead of becoming an 8-bit quantise.
- **EXR delivery** (FIX-015, 016): multi-part writes report what is on
  disk; EXR Multi-Part and Write EXR Passes return a `manifest`; Write lists
  exactly the files it produced and the backend that wrote each one.
- **DPX** (FIX-017): pass-through is tagged "User defined" instead of
  linear; linear DPX refuses to clip values above 1.0.
- **Viewer to delivery** (FIX-018): the export knows what the Viewer's
  pixels are. A display-encoded source is linearised before grading, as the
  Viewer does, so an sRGB IMAGE delivered as sRGB is no longer encoded
  twice and linear EXRs carry the source's colour space.

## [3.5.2] - 2026-09-28

3.5.1 did not reach the Comfy Registry: its tag pointed at a commit that
still carried version 3.5.0. 3.5.2 is the first Registry release with the
3.5.1 changes below, plus these.

### Added

- Qwen-Image Edit 2511: Loader preset with the official template files, and
  Sampler Pro defaults for it.
- Prompt reference images for Qwen-Image 2.1, Qwen-Image Edit (3 at most) and
  Flux.2 Dev / Klein editing, through the native encoders, with a `vae` input
  and a `latent` output sized on image_1.

### Fixed

- Qwen-Image latents are sampled in 5D as the native KSampler does, and a
  single-frame 5D latent is tiled again in tile_mode, takes a 4D
  noise_override, and decodes as independent images rather than a clip.
- Qwen-Image Edit reference images reach the encoder as RGB; an alpha channel
  broke Qwen2.5-VL's normalisation. The negative reuses the positive's
  reference latents instead of VAE-encoding every image again.
- Reference images on a plain Qwen-Image checkpoint log a warning. The Prompt's
  `vae` input follows `model_meta`, so saved workflows keep its slot.
- Sampler Pro names the Prompt's empty latent output instead of calling it an
  IMAGE, and warns on seed 0 with Qwen-Image 2.1 (default seed is now 1).
- Bypassed nodes are followed to the input ComfyUI actually forwards when the
  Prompt and Sampler read the Loader's model_meta.

## [3.5.1] - 2026-09-28

### Fixed

- Viewer shortcuts now require current hover, selection or fullscreen, including
  undo and transport controls. Leaving an unselected Viewer releases the keys.
- Viewer video playback keeps audio, and connected VIDEO inputs carry an audio
  preview synchronized with forward sequence playback. Reverse playback is silent.
- Workflow switches restore Viewer media references. Imported videos are stored
  in ComfyUI's input folder; executed frame previews still depend on temp files.
- Native video rendering follows decoded frames and avoids queued bitmap copies.
  The export menu now directs video/sequence exports to Write nodes.
- Nuke's terminal listener avoids GUI-only functions and runs node edits
  synchronously on its main thread. Signed EXR handoff verified in Nuke 17.1v2.
  The wheel now includes the standalone listener, with a CI inclusion check.
- Temporal windowing now rejects full-clip ControlNet outputs before invoking
  the model with mismatched frame ranges. Unwindowed ControlNet is unchanged.
- Resolve's Windows import worker selects its own Python installation before
  loading the scripting DLL, fixing the reproduced native crash. Live Studio
  acceptance imported a still and a 121-frame EXR sequence successfully.
- Temporal planning keeps full-sized final windows and the requested overlap.
  Complementary fades now handle three-window joins without changing total
  weight. Windowed-video visual acceptance remains open (see KNOWN_ISSUES).
- DCC Bridge briefly drains oversized requests after shutting down sending,
  avoiding Windows connection resets that could discard the error reply.
  Drain time is bounded at 0.25 seconds.

- Sampler Pro Auto/Custom preserve `flux_shift=1` (no extra shift). Model
  schedules already include their native shift; automatically adding the
  architecture default again distorted the schedule. Explicit extra shifts
  remain supported. The UI also stops auto-writing the native model shift;
  Wan presets now select 1. Existing saved explicit values are preserved.

- Denoise detail recovery adds back the high-frequency removed residual
  instead of blending the entire original image over the denoised result.
  Default 0 is unchanged; 1 now retains low-frequency denoising.
- Linear Matting uses trimap_dilation to define an unknown band, keeping known
  foreground/background interiors fixed while refining edges. Radius 0
  preserves the mask; incompatible mask batches produce a clear error.
- Cinema D60 uses a P3 matrix at ACES white instead of reusing the D65 matrix.
  The choice now changes the colour conversion as labelled.
- Guided denoising uses sigmaSpace for Gaussian-weighted local statistics
  instead of ignoring the setting. Image-edge windows are normalized, and
  negative/HDR values remain unclipped.
- Run Resolve's native scripting API in an isolated worker. A native DLL
  crash or stalled connection now reports an import failure instead of
  terminating or indefinitely blocking ComfyUI; exported media remains intact.
- Dynamic sampler guidance includes ramp-end boundaries, reaching the intended
  middle and late guidance/CFG targets instead of holding halfway values.
  Partial sampling ranges retain their requested limits.
- False Colour measures linear luminance using an explicit input-encoding
  setting (linear Rec.709, sRGB or Rec.709). Zebra stripes obey strength,
  RGBA inputs preserve alpha and strength 0 preserves all source pixels.
- Standard inpaint stitching uses feather_radius in the composite, matching
  its returned blend mask. Set the radius to 0 for the former hard-mask blend.
- Frame Stamp composites its text overlay onto float pixels without converting
  the plate to 8-bit. Source alpha and untouched HDR/negative pixels survive.
- T2V and I2V include every offered gamut, transfer function and peak-brightness
  setting in positive prompt conditioning, including P3-DCI and 100-nit SDR.
  Negative prompts remain unchanged. These controls do not master output pixels.
- Synthesis chroma preservation now controls neutral versus proportional RGB
  highlight brightening at equal luminance. Its default 0.8 now visibly affects
  coloured highlights; use 1.0 for the previous proportional lift.
- WebGL built-in views convert tagged ACES and wide-gamut linear sources to
  Rec.709 display primaries while preserving linear export values.
- Clip Detector uses the selected Any / All / Luma statistic for soft masks.
- Exposure Blend processes every frame, supports singleton brackets, uses
  original bracket reliability for Exposure Weighted (including the middle
  bracket), and handles all-black images without a reduction error.
- HDR Monitor honours gamma_correct_sdr for Exposure + Gamma.

- Prevent Floyd–Steinberg dithering from modifying its input image; preserve
  scan-line arithmetic across NumPy versions.
- Preserve video colour tags through FFmpeg filtering, write CDL sidecars as
  UTF-8, and make console diagnostics safe on Windows legacy encodings.
- Select the best feasible multipass lighting fit for singular decompositions.
- Include the Viewer’s bundled OpenColorIO runtime and licence notices in wheels.
- Repair test imports, log capture, installed-model isolation, Windows memory
  measurements and Chromium renderer settings. Release branches now run CI;
  full-dependency coverage and syntax checks gate the release.

### Features and changes

### Upgrade note

1. **The latent-space RUDRA decoders are gone.** `◎ Radiance HDR VAE Decode`
    no longer has `rudra_decoder`, `decoder_size` or `model_meta`; it decodes
    through the model's VAE in every mode, and the decode mode formerly named
    "Direct HDR / RUDRA" is now "Direct HDR" (the old value is migrated on
    load). `SDR → HDR Universal` loses its `vae`, `rudra_size` and
    `model_meta` inputs and its "Legacy RUDRA" backend; `SDR → HDR Recover`
    loses `vae`, `rudra_size` and `model_meta` and gains the pixel controls.
    `NDI Sender` loses `turbo_mode`, `latent_in`, `vae` and `model_meta`. A
    saved graph still loads (the frontend drops the missing widgets and the
    nodes accept the stale keyword arguments), but check widget values on those
    three nodes once, since ComfyUI stores them by position. The per-model
    `rudra_turbo_decoder_*` / `rudra_full_decoder_*` files in `models/radiance`
    are no longer read and can be deleted. `fast_vae.py`, `model/vae.py`, the
    latent training scripts and the `rudra` dataset tools moved to
    `_to_delete/legacy-latent-rudra-20260923/`.
2. **Learned SDR → HDR is one pixel model.** `SDR → HDR Universal` and
    `Recover` run RUDRA's pixel-space network (`sdr2hdr_shadow_v1.safetensors`,
    downloaded on first use) on any image or frame batch, with the temporal
    residual model preferred on ordered video when its checkpoint exists.

### Fixed

- **Send to Nuke, Send to DaVinci Resolve and the bridge, checked end to end (3.5.0).**
  - *Push to Nuke never worked out of the box.* The Nuke listener refuses
    unsigned commands, and the key came only from `RADIANCE_DCC_AUTH_TOKEN`.

### Added

- **Every node and every input is documented.** All 149 menu nodes have a
  description and all 1,259 inputs a tooltip.

### Changed

- **README is for users; the development record moved** to
  `docs/DEVELOPMENT.md`.

### Removed

- **Release clean-up: files nothing loaded, called or documented.**
  `core/param_memory.py` and other stale files were removed.

## [3.4.x unreleased work, folded into 3.5.0]

### Upgrade note

Four changes alter what an unchanged graph produces. Read these before updating
 a project in flight.
