# Changelog

All notable changes to FXTD Radiance will be documented in this file.

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
