# Changelog

All notable changes to FXTD Radiance will be documented in this file.

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
