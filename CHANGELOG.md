# Changelog

All notable changes to FXTD Radiance will be documented in this file.

## [3.5.0] - 2026-09-27

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
   downloaded on first use) on any
   image or frame batch, with the temporal residual model preferred on ordered
   video when its checkpoint exists. Backends are `Auto` / `Direct Pixel` /
   `Temporal`. Checkpoints are found in every `models/radiance` folder
   ComfyUI knows about, including `extra_model_paths.yaml`, or through
   `RADIANCE_SDR2HDR_PIXEL` / `RADIANCE_TEMPORAL_RUDRA`; the `report` output
   names the path that ran.

3. **HDR VAE Decode's Direct HDR mode runs the pixel model after the VAE.**
   On a plain latent (anything that did not come from VAE Encode (HDR), i.e. every
   sampler output) Direct HDR now decodes sampler-safe sRGB through the
   model's VAE and hands it to the RUDRA pixel SDR → HDR model, returning
   scene-linear HDR (1.0 = 203 nit reference white) with highlights up to the
   new `hdr_peak_nits` input (default 1000). The log inversion is used only on latents that carry VAE Encode (HDR)'s
   `radiance_meta`, the one case where it is correct. `metadata` names the
   path (`hdr_path`) and carries the SDR → HDR report.
4. **SAM Loader and SAM Mask Generator are hidden.** No SAM runtime ships;
   the generator drew discs around the click points. Both stay registered so
   saved graphs open, and raise a clear error when executed. Use a SAM2 node
   pack and feed its MASK into Radiance.
5. **Behaviour changes from the honest release pass** (below): Motion Blur's
   `energy_conservation` default now conserves energy instead of brightening
   the frame; Regional Prompt `Replace` keeps the global prompt outside the
   region; Upscale Face Restore `auto` now restores faces; Audio Cut and
   Camera Sync raise on a bad path instead of returning defaults; Video
   Export raises when an EXR cannot be written.
6. **One reference-white convention for HDR linear (BT.2408).** Linear light
   is 1.0 = reference white (203 nits by default) everywhere: HDR Encode,
   Write's `hdr_reference_nits`, HDR Diagnostics, and now `SDR → HDR
   Universal`, `SDR → HDR Recover` and HDR VAE Decode's Direct HDR. Universal
   and Recover `Linear` / `AP0` output used 1.0 = 100 nits (SDR white at
   2.03); the same image is now 2.03x smaller in value and identical in nits,
   and its PQ / HLG match HDR Encode exactly. Recover gains
   `reference_white_nits`. **HLG** everywhere (HDR Encode, HDR Monitor, Read,
   Write, Universal, Recover) is the BT.2100 transcode for the 1000-nit
   reference display that OpenColorIO's Rec.2100-HLG display uses: reference
   white at 75 %, 18 % grey at 43.6 %; HDR Encode used to put reference white
   at 100 % and clip everything above it. **Camera log targets** of the VAE
   Decode nodes now carry their camera gamut (LogC4 = AWG4, S-Log3 =
   S-Gamut3.Cine, ...), so their colours change in any colour-managed host.

7. **The Viewers show what the node says the pixels are.** `◎ Radiance Viewer` gains `input_space` (Auto / sRGB / Linear Rec.709 / ACEScg / Linear Rec.2020 / Linear P3-D65 / ACES2065-1), `float_precision` (Half by default; the old hidden default was 32-bit) and `fps`. `◎ Radiance Lite Viewer` gains `input_space` and `fps`.
   - **Auto.** A ComfyUI IMAGE in 0-1 is sRGB and is shown untouched; anything above 1.0 or below 0 is linear and is shown through OpenColorIO ACES 2.0 SDR (the Viewer's WebAssembly OCIO, loaded on demand).
   - **Brightness change.** Ordinary images now look like ComfyUI's own preview, darker and with more contrast than before. Linear material now looks like ACES 2.0 rather than the Narkowicz fit (18 % grey at 0.349 instead of 0.556).
   - **View menu.** It is now Auto / ACES 2.0 / ACES 1.3 / sRGB / Rec.709 (BT.1886) / Filmic (approx.). The PQ and HLG entries, which did nothing, are gone until the HDR canvas lands.
   - **Browser storage.** Input space and output LUT are no longer remembered in browser storage.

8. **Viewer keys changed (phase 2).** Keys now reach only the viewer under the pointer, and follow the RV / Nuke / NLE layout.
   - `A` toggles alpha, `X` cycles compare (`Alt+X` clears in/out), `Y` shows luma.
   - `J` / `K` / `L` shuttle, `I` / `[` set in, `O` / `]` set out, Home / End jump to in / out.
   - `-` / `=` change the viewer-only f-stop and `0` resets viewer f-stop and gamma. `0` no longer resets the grade.
   - `Shift+K` toggles focus peaking; the sequence dock's tools moved to Shift+letter.
   - Zoom above 1:1 is nearest-neighbour by default (it was always smoothed).

9. **Multipass Extract is retired; Multipass Estimate replaces it.** Its
   material and lighting passes were image filters (Retinex albedo, image
   minus blur as "specular", roughness from local contrast, emission and
   transmission from saturation and brightness, a k-means "object ID") and
   its geometry passes had no real scale. The node is hidden from the menu; a
   saved graph still opens and stops with a message naming the replacement.
   Multipass Estimate's outputs differ (see Added), so reconnect them.
10. **Multipass Relight reads `ao` as renderers write it, 1 = open.** It used
    to read the pass as an occlusion amount, which inverted real AO loaded
    through Read AOVs. With nothing connected the default is still fully
    open. A hand-made occlusion mask now needs inverting once.
11. **Fixes that change the output of a saved graph** (details under Fixed,
    "Bugs found while documenting every input"): Multipass Relight's point
    light with a depth pass is lit the right way round; Video Batch Decode no
    longer divides HunyuanVideo, CogVideoX and SD latents by `latent_scale`;
    Video HDR Decode puts input white on `peak_nits`, its SDR preview reaches
    display white and its `sRGB / BT.1886` output has a curve; HDR Color
    Pipeline converts every primaries pair and its D65/D60 adaptation and
    ACEScg-to-Rec.709 matrices are corrected; Digital Cinema Read reads a
    video's first frame; EXR MultiPart writes every frame of a batch; Radiance
    QC with `fail_on_errors` stops the graph; Synthesis `guidance_nits` is on
    the 203-nit scale.
12. **Models download on first use.** Nothing to set: a node that needs a
    model fetches it, pinned and SHA-256 checked. Set
    `RADIANCE_ALLOW_DOWNLOADS=0` to keep the old ask-first behaviour on a
    metered or air-gapped machine. Gated models need `HF_TOKEN`.

### Fixed

- **Send to Nuke, Send to DaVinci Resolve and the bridge, checked end to end (3.5.0).**
  - *Push to Nuke never worked out of the box.* The Nuke listener refuses
    unsigned commands, and the key came only from `RADIANCE_DCC_AUTH_TOKEN`,
    which had to be set identically in both programs' environments and is
    unset by default. Both sides now fall back to `~/.radiance/dcc_token`,
    created with a random 256-bit key the first time either needs it, so on
    one machine it works with no setup; commands are still HMAC-signed.
    Checked by running the real listener script against a stand-in Nuke.
  - *A file name with a hyphen or space broke the Nuke push*, after the EXRs
    were written: the Read node name went to Nuke unchanged, and Nuke
    accepts only letters, digits and underscores. The name is made valid now,
    and a failed push is reported in `status` instead of stopping the graph.
  - *Paths with quotes lost them* on the way to Nuke (stripped as if the
    payload were code; it is JSON). *The `.nk` snippet* now quotes the path,
    so a folder with a space loads, and sets `raw` like the push does.
  - *Send to DaVinci Resolve* can import into the open project's Media Pool
    (`import_to_media_pool`, off by default), through Resolve's scripting API;
    a numbered batch arrives as one clip. When Resolve is not running or
    external scripting is off, `status` says which.
  - *`input_space`* on both nodes: the tensor used to be written unchanged
    to every format, so either the EXR or the TIFF / PNG was in the wrong
    encoding. Scene-linear or sRGB input is now converted to what each format
    holds; the default, As is, keeps the old output.
  - *MCP Bridge is now DCC Bridge.* It was never the Model Context Protocol
    (it is Radiance's own TCP channel for queueing workflows and exporting
    frames), and the name read as if an AI client could connect to it. Only
    the label changed; the node key stays `RadianceMCP`, so saved graphs load.
  - *DCC Bridge*: a request line is capped at 4 MB (a client that sent no
    newline grew memory without bound), a failed Nuke push no longer raises,
    and its Resolve target says the Send to DaVinci Resolve node can import.
  - Tests: `tests/test_dcc_send.py` (7).

- **Models download automatically, pinned and verified (3.5.0).** Every node
  that needs a model now fetches it on first use; `RADIANCE_ALLOW_DOWNLOADS=0`
  or the Hugging Face offline flags still stop all downloads. One downloader,
  `core/model_fetch.py`, installs a file only after its SHA-256 matches, writes
  to `.part` and resumes after an interruption, shows ComfyUI's progress bar,
  and sends the user's `HF_TOKEN` to gated repositories (a refusal names the
  page to accept and the token to set). Found by auditing every source online:
  - *Read Models* (`auto_download`, now on by default): all 60 catalogue files
    were on unpinned `main` with no digest. Each is pinned to a commit with its
    SHA-256 and size. `flux1-schnell-fp8` pointed at a file that does not
    exist (now Kijai's `flux1-schnell-fp8-e4m3fn`), and the FLUX.1 VAE came
    from the gated FLUX.1-dev repo (now Comfy-Org's ungated copy, the same
    file byte for byte).
  - *Upscale*: none of the nine files had a digest; the Hugging Face mirrors
    tried first were missing or private, so every download fell back to a
    second URL. Each now comes from its original release file with its
    SHA-256. HAT-L's GitHub URLs return 404 (it is published only on Google
    Drive): it is a manual install, and Tier 2 uses SwinIR-L at 4x until it
    is present. SeedVR2's second attempt loaded a repository that does not
    exist; it now says to install the SeedVR2 node pack. The SD x4 pipeline is
    pinned to a commit.
  - *AI Upscale* downloaded SUPIR and Real-ESRGAN straight to the final path,
    with no check, so an interrupted download left a truncated model that was
    loaded next time. Pinned and verified now.
  - *Depth Map Generator* loaded Depth Anything V2 from `main`; pinned to a
    commit per size.
  - `tools/pin_models.py` re-checks every pin against its source (79 of 79 on
    2026-09-25). Tests: `tests/test_model_pins.py`; the loader and consent
    tests follow the new default.

- **Bugs found while documenting every input.** The tooltip pass listed these
  in KNOWN_ISSUES as owed; each is fixed and pinned by a test that fails on
  the old code (`tests/test_documented_bugs.py`, 18).
  - *ControlNet Apply* passed the hint image as B,H,W,C where ComfyUI's
    ControlNets take it channels-first, and passed no VAE, so ControlNets that
    encode the hint (SD3, Flux, DiT) could not work. The hint is channels-first
    and an optional `vae` input is passed on.
  - *Upscale Video* ran the built-in models on the CPU for clips (it used the
    frames' device, always the CPU in ComfyUI); it uses the compute device. A
    one-frame clip now keeps `model_tier`, `sharpness_boost`,
    `enhancement_prompt` and `diffusion_steps`.
  - *Multipass Relight*, point light with `depth_map` and no
    `world_position`: the depth flip was on the wrong setting, so near pixels
    sat behind far ones either way. Near is now toward the camera.
  - *Video Batch Decode* divided the latent by dit_config's `latent_scale`,
    although a ComfyUI sampler already returns it in VAE space, so
    HunyuanVideo, CogVideoX and SD latents were scaled twice. It is reported
    and not applied.
  - *Video HDR Decode*: Reinhard put input white at half of `peak_nits`; it
    is now extended Reinhard with the brightest input (1.0, lifted by a
    positive EV) on `peak_nits`. The SDR preview peaked near 0.12; it now
    reaches display white. `sRGB / BT.1886` output was the same clamped linear
    light as Linear; it is the sRGB curve relative to `peak_nits`.
  - *HDR Color Pipeline*: only five primaries pairs converted and the rest
    passed through silently; every pair of the listed primaries converts, with
    Bradford adaptation where the whites differ. `chromatic_adaptation` on a
    conversion that already adapts (to or from ACEScg) is skipped with a
    warning instead of applied twice. Found on the way: its D65/D60 Bradford
    matrices put D65 white 1.2 % off D60, and its ACEScg-to-Rec.709 matrix was
    4e-3 off the inverse; both corrected.
  - *Video Model Info* forced every Wan model to Wan2.1 (16ch), the 48-channel
    Wan2.2 TI2V 5B included, over `model_preset`. The latent channel count the
    model reports picks the preset, and a matching preset the user chose is
    kept.
  - *Digital Cinema Read* skipped a video's first frame: `start_frame` counts
    from 1 but went to the reader as a 0-based offset. It is converted.
  - *EXR MultiPart* wrote only the first frame of a batch; every frame is
    written, `<prefix>.<frame_index + n>.exr`, in parallel. A one-frame AOV is
    used for every frame. Depth stays one Z channel from the first channel.
  - *Policy Guard* read every input at 1.0 = 100 nits and PQ or HLG code
    values as linear light. A `signal` option reads Display SDR, Scene-linear
    (1.0 = 203 nits), PQ or HLG. *Radiance QC* `fail_on_errors` only added
    "(BLOCKING)" to the status; it now stops the graph with the report.
    *Synthesis* `guidance_nits` used 100 nits = 1.0; it uses 203.
  - *AMF* description (and clip name) went into the XML unescaped, so `<` or
    `&` broke the file; they are escaped.
  - *Regional prompts*: each chained node reset the strength of every earlier
    region to its own `global_strength`, and in Replace mode dropped the
    earlier cut-out. Earlier regions keep their strength and area, and the
    global keeps every cut.
  - *Sampler* `sdr_blend` and `sdr_inject_steps` defaulted to 0 in Python and
    0.35 / 6 in the widgets, so an API prompt that left them out ran with SDR
    guidance off. The defaults match.

- **HDR Color Pipeline crashed on PQ input.** It still passed `peak_nits` to
  a PQ decode that had dropped the argument on purpose, so every PQ run
  raised TypeError. PQ is absolute; `pq_peak_nits` no longer reaches it.
- **`slow` tests ran on every plain `pytest`.** The 1-to-100 frame-count
  sweep (300 encoded clips) is marked `slow` and the README said slow tests
  were deselected by default, but nothing deselected them: it cost 68 s of a
  245 s suite locally and in CI. `pyproject.toml` now sets
  `addopts = "-m 'not slow'"`; `pytest -m slow` runs them, and CI's
  full-dependency job runs them in their own step so the sweep is not lost.
  The sweep itself now encodes and decodes on a thread pool (its time is
  process start-up, not codec work): 68 s to 40 s on 2 cores, more on a
  workstation. Every length is still its own encoder-produced file.
- **VAE Decode (HDR) told users of the recommended pair that correct output
  was wrong.** Every run of VAE Encode (HDR) into VAE Decode (HDR) with a
  linear source logged "output colors will be WRONG". The encoder carries a
  non-log source in ARRI LogC4 and records it in the latent, so the LogC4
  decode is the exact inverse. The warning now appears only when a latent
  without encoder metadata is decoded in Compress (Log), and says what to
  pick instead.
- **Pass-through video was tagged as linear light.** Write's default
  "Linear (pass-through)" writes the IMAGE unchanged, which for ordinary
  ComfyUI images is display-encoded, but tagged MP4 / ProRes with a linear
  transfer, so players that honour the tag decoded it wrong. The transfer is
  now left unspecified for pass-through; an encoding you choose is still
  tagged.
- **Multipass Estimate's MoGe download left a `.radiance_download` folder**
  of Hugging Face lock and metadata files in `models/geometry_estimation`.
- **Four model loads skipped the download consent gate.** Depth Map
  Generator (Depth Anything V2; Base and Large are CC-BY-NC-4.0), the SD x4
  upscaler (~2.4 GB), the SeedVR2 fallback and character-consistency CLIP
  called `from_pretrained` directly, so they downloaded even with
  `RADIANCE_ALLOW_DOWNLOADS=0`. Without consent they now load only a copy
  already in the Hugging Face cache; Depth Map Generator says what to set or
  where to get the model, and character matching falls back to colour
  histograms as before.

- **Viewer and Lite Viewer, phase 1 (3.5.0).**
  - *Standard images washed out.* Float frames were always read as linear, so an sRGB IMAGE was encoded twice. The renderer now folds the node's tag into `isLinearTexture`, so the shader, scopes and bloom all see it.
  - *Input colour space guessed from brightness.* The "midgrey fingerprint" decoded ordinary photos as camera log, and the guess was saved to browser storage. `_userSetIDT` was never set, so a manual choice was overwritten. LogC4 metadata was read as LogC3.
  - *Encoded twice.* Rec.709 (the camera OETF) and every log view had the sRGB OETF applied on top. Rec.709 is now the inverse of BT.1886, applied once.
  - *DaVinci Intermediate.* The shader's "BUG-1 FIX" had put 18 % grey at 0.447. The Blackmagic spec, OCIO and Radiance's Python all give 0.336, and the shader now does too.
  - *Nits.* The heatmap used 1.0 = 100 nits and the PQ waveform plotted 1.0 at 10,000 nits. Both now use 203, like the probe and the nodes.
  - *Frames black on replay.* Each upload deleted the previous frame's cached texture. `setFrame` also bypassed the cache.
  - *Compare.* WebGL had no `loadCompareTexture`, so the base-class stub threw on compare_image and Pin Frame. Compare frames now follow the playhead.
  - *Playback.* The fps and loop controls sat after an unconditional return. They are now in the viewer bar, the source fps comes from the node, and the three fps values are one. Timecode is SMPTE with an integer timebase (29.97 used to print frame 30) and drop-frame at 29.97 / 59.94.
  - *Graded EXR.* It carried the display transform, overlays and wipe, alpha 1, and no colour metadata. It is now the grade in scene-linear, with source alpha and `chromaticities`.
  - *OCIO default.* The one-click start button loaded ACES 1.3 CG and now loads the ACES 2.0 studio config.
  - *PNG previews.* Linear frames got x/(1+x) with no display encoding (dark), and 8-bit conversion truncated. Both viewers now bake the OCIO ACES 2.0 view (exact, threaded) and round. The bake also no longer overwrites the IMAGE passed downstream.
  - *Always dirty.* `IS_CHANGED` returned NaN on every queue, re-running every node downstream of a viewer. It now fingerprints the inputs, and is NaN only while the delivery cache has no frames for the node.
  - *Lite Viewer.* Readout, clip check and diff read an fp16 float proxy, so they show source values at source coordinates. The canvas is in device pixels, so 1:1 is exact on scaled displays; it was sized from a bordered box, 0.3 % off. B is scaled to A, diff has a gain, and play/loop run at the source fps. Frames load progressively.
  - Tests: `tests/test_viewer_phase1.py` (11), and `js/tests/viewer_color.test.mjs` and `js/tests/lite_viewer.test.mjs` (removed with the Lite Viewer), which read real pixels back from Chromium.

- **Legacy nodes off the menu.** HDR Latent Encoder and HDR Turbo Encoder are
  hidden (`DEPRECATED`) and now raise when run, naming VAE Encode (HDR): a
  graph using them used to render clipped with no warning, since their
  decoders were retired. ACES 2.0 Output Transform (Legacy) is hidden and keeps
  working; ACES 2.0 Output Transform replaces it. All three stay registered so
  saved graphs open, and the registry's 2.3.3 had none of them. 157 nodes
  registered, 149 in the menu.
- **One version everywhere.** `pyproject.toml`, `config/constants.py`,
  `package.json`, the CHANGELOG and the README badge all read 3.5.0, and
  `tests/test_version_sync.py` fails if they drift. What users saw still said
  older releases: the Sampler's node description and its report header
  ("Radiance Sampler v3.0.0") and metadata, the QC report header, the AMF
  exported by Deliver, the Viewer's description, and module versions of 3.1.0
  to 3.2.2 in the prompt, audio and video nodes. They all derive from the one
  constant now.
- **`pyproject.toml` for the registry.** Python 3.10 to 3.13 (ComfyUI's own
  floor; 3.9 was listed and cannot run ComfyUI), a description of what the pack
  does, Homepage / Changelog / Models links, and the `full` extra without four
  packages nothing imports (anthropic, google-generativeai, accelerate, peft).
  The registry currently serves 2.3.3, so 3.5.0 is the next registry release.
- **README for end users.** New Quick start, Updating, Upgrading from 2.x or
  3.4, Settings (every environment variable a user may set) and
  Troubleshooting sections. The overview no longer advertises the retired
  Turbo / Full HDR decoders, the optical-flow limitation is two lines with the
  measurements in KNOWN_ISSUES, and the verification record and release
  checklist moved to a collapsed "Development status" section at the end.
  Upgrade notes 2 and 3 above name the current model and encoder.
- **HDR latents could not be decoded back to HDR: VAE Encode (HDR) is now the
  encoder VAE Decode (HDR) inverts.** Found on a clean install with a real VAE
  (SD `vae-ft-mse`) and an HDR plate peaking at 7.75. The two HDR latent
  encoders on the menu, HDR Latent Encoder and HDR Turbo Encoder, fed latent
  decoders retired in 3.5.0; nothing stamped their latents, so VAE Decode (HDR)
  took them as ordinary diffusion latents and returned them clipped at 1.0,
  midtones 2.5 to 14 stops off. The encoder its Auto / Direct HDR log inversion
  was written for, `RadianceVAE4KEncode`, was implemented and tested but never
  registered. It is now `RadianceHDRVAEEncode`, shown as **VAE Encode (HDR)**
  beside VAE Decode (HDR), defaulting to Compress (Log) in both the widget and
  the method signature (API prompts omit optional inputs). Measured through
  the SD VAE: highlights within 0.03 stops (median), 97% of the range above 1.0
  kept, midtones within 0.04 stops. The old two stay registered so saved
  workflows load, relabelled "(Legacy)" with a description that says why.
  157 nodes. `tests/test_hdr_vae_encode_pair.py`.
- **VAE Encode (HDR) crashed with `latent_sampling = mean` on any ComfyUI
  VAE.** `torch.Tensor` has a `mean` method, and the posterior check tested
  `hasattr(posterior, "mean")` before `isinstance(posterior, Tensor)`, so the
  bound method became the latent. Tensors are checked first.
- **Fresh install verified from the registry package.** Packed with
  `comfy node pack` exactly as the registry builds it, installed into a clean
  ComfyUI 0.32 on Python 3.13 with `requirements.txt` and `install.py`: 157
  nodes load in 0.9 s, OCIO configures itself from the built-in ACES studio
  config, the first SDR → HDR run downloads the RUDRA model and applies it, the
  sampler-safe decode is bit-identical to ComfyUI's own VAE Decode, and the
  suite passes there too (4,187 tests, OpenCV 5.0, NumPy 2.5).
- **OpenEXR is skipped on Python 3.14.** It has no 3.14 wheel, so a source
  build there failed the whole `pip install -r` and nothing installed. The
  requirements files and `pyproject.toml` carry `python_version < "3.14"`, and
  `install.py` skips it there; EXR goes through OpenImageIO, then OpenCV.
- **README install section.** Says what the registry install does by itself
  (dependencies, OCIO, the RUDRA model), the Windows clone pointed at the beta
  repo, and the GPU acceptance tool is marked git-install only (the registry
  package excludes it).
- **RUDRA weights licence stated correctly.** The README called the RUDRA
  weights Apache-2.0. RUDRA's code is; its trained weights have been
  non-commercial since 5 Sep 2026 (RUDRA `NOTICE` and `checkpoints/LICENSE`),
  because the HdM-HDR training data is licensed for academic use only. The
  README model section now says so and links the terms, the download log
  states it on install, and the docstrings that repeated "Apache-2.0" are
  corrected. Radiance's own code licence is unchanged.
- **RUDRA pixel model: false colour, red cast and over-peak channels in
  recovered highlights.** Reported on a clean Flux.2 sunset (Hybrid, default
  settings, v5): false-colour rings around the sun lined up with where each
  channel clipped, white areas came out with red about 2.2x green, glints had
  4x the source's chroma noise, and 10% of pixels had a single channel above
  the 1,000-nit peak (up to 3,500 nits). Reproduced through the node at
  R/G 2.29, 9.4% over peak, 4,094 nits, 6x chroma noise. Four causes, all
  fixed:
  - *A Rec.2020 → Rec.709 matrix on a model that never changes primaries.*
    The network is a per-channel mapping trained on SDR rendered
    channel-wise; RUDRA's own inference writes its output as-is. The matrix
    only over-saturated, and on a warm highlight multiplied the red cast by
    another 1.5x. Removed.
  - *Per-channel colour in blown areas.* The inverse tone curve is steepest
    at the clip, so where red had clipped and green had not, red alone was
    pushed up (the analytic baseline alone gives R/G 6.2 in that band against
    1.26 in the source); each clip boundary became a ring. The shipped
    checkpoints were also trained on a corpus that almost never clipped. In
    clipped highlights Radiance now takes the learned luminance and keeps the
    source chromaticity, lets the lift in as the source goes to white (median
    channel from code 0.85), and never goes darker than the deterministic
    base. Shadows keep the model's colour, where it was trained on it.
  - *The peak limiter measured luminance.* `_soft_peak_limit` now bounds the
    brightest channel with one hue-preserving gain, so nothing the learned
    path returns can clip per channel in an HDR10 encode.
  - *Tiles shifted the result.* Every block opens with GroupNorm, so a 512-px
    tile was normalised by its own contents: 1.4 stops at p99 in the
    highlights against the whole frame. The model now runs whole-frame when
    it fits (measured ~1.5 KB/pixel fp32, half under bf16) and falls back to
    tiles, including on CUDA out-of-memory.
  After, on the same frame: R/G 1.00 in the clipped core, no channel above
  peak, chroma noise equal to the source, clipped-core luminance 765 nits
  (Expand alone: 193).
- **Every published RUDRA checkpoint loads, and the shipped one is the
  default.** Radiance carried a pre-gate copy of `SDR2HDRNet` and read only
  `.pt`, while Hugging Face publishes `.safetensors` and names
  `sdr2hdr_shadow_v1` (v5 plus a trained shadow gate) as the shipped model,
  which Radiance could not build. The network is now ported from RUDRA's
  reference (shadow gate, residual-scale gate, curve head, `from_config`,
  corpus EV), verified bit-identical to it on shadow_v1, v5 and the legacy
  `.pt`. Checkpoints load from `.safetensors` (config from the file's
  metadata) or `.pt` (`weights_only`); heads the config forgot are read off
  the state dict; the temporal refiner is refused by name. The per-frame
  gates run once per frame, not per tile, as in RUDRA. Search order and the
  auto-download now put `sdr2hdr_shadow_v1.safetensors` first (pinned
  commit, SHA-256 checked); the flat and `sdr2hdr/` layouts are both found.
  `tests/test_pixel_rudra_parity.py`.
- **The RUDRA pixel model downloads itself.** SDR → HDR Universal and
  SDR → HDR Recover used to need a RUDRA checkpoint installed by hand;
  without it Universal quietly fell back to plain expansion. Radiance now
  fetches it on first use (~5 MB) from
  `huggingface.co/fxtdstudios/RUDRA`, pinned to a commit and verified by size
  and SHA-256 before it is moved into `models/radiance` (new
  `model/pixel_download.py`). Because it is small and first-party it is on by
  default, unlike the large third-party weights, which still ask first. Off
  with `RADIANCE_ALLOW_DOWNLOADS=0`, `HF_HUB_OFFLINE=1` or
  `TRANSFORMERS_OFFLINE=1`; `core/consent.downloads_allowed` gained a
  `default` argument and honours the two offline flags. One attempt per
  session, one "downloads are off" message per session, and the test suite
  runs with downloads off. `tests/test_pixel_download.py` covers the pin, the
  opt-outs, verification failures and network errors.
- **README: model and workflow links.** Direct download link for the RUDRA
  model, the auto-download and opt-out, and a table for both shipped
  workflows with a link to every model file each one needs.
- **Final release review (3.5.0).** Run against a real ComfyUI 0.32 / frontend 1.48.
  - *The shipped start graph did not queue.* The frontend restores widgets by position, and `workflows/start.json` predated inputs added since (`audio_cfg` on the Sampler, `crop_to_broadcast_resolution` on Resolution, the viewers' `input_space` / `fps`, the encoder's `negative_mode`). Everything after each new input moved one slot: the Sampler's `audio_cfg` received "euler", its extension threw `samplerMode.includes is not a function`, and ComfyUI rejected the prompt. HDR VAE Decode also had no VAE connected, and the Flux loader had no `clip_l`. The file is re-saved from the frontend itself, the VAE is wired, `clip_l` is set, and Resolution uses the Flux 16-channel latent. It now passes ComfyUI's prompt validation.
  - *README said 131 nodes.* The log prints 156, so the install check told users a correct install had failed.
  - *CI red on the lightweight lane.* Two new test files imported torch submodules at module scope, which fails collection against the conftest stub (it is a module, not a package), and a handful of torch-only tests in files that gate per test were not marked `real_torch`. They now skip cleanly without torch and run on the full lane; the lightweight lane passes (1817 passed, coverage 22.6 %).
  - Tests: `tests/test_shipped_workflows.py` re-derives each Radiance node's widget order from `INPUT_TYPES` and checks every saved value against its input's type, so the next new input that shifts a shipped workflow fails the suite.

- **Cinematic Encoder, step 1 (3.5.0).** Checked against ComfyUI 0.32's real tokenizers.
  - *SDXL / SD1.5 long prompts lost their ending.* The node inserted A1111's `BREAK`, which ComfyUI does not support, so the word "break" was encoded; then anything past 77 tokens was cut. Camera, lens and lighting come after the subject, so they were what got cut. Both steps are gone: ComfyUI chunks long CLIP prompts itself and T5 / LLM tokenizers take any length. Prompts past a model's trained window (Flux / Wan 512, SD3 256) are logged, never cut.
  - *Modern encoders detected as SDXL.* Wan (umt5xxl), Flux.2 (mistral3_24b), Z-Image / Flux.2 Klein (qwen3_4b / 8b), Lumina2 (gemma2_2b), Qwen-Image / HunyuanVideo 1.5 (qwen25_7b) and AuraFlow (pile_t5xl) got SDXL tag prompts. With `model_meta` wired, the loader's `flux2`, `z_image`, `lumina2`, `chroma`, `qwen_image`, `hidream`, `cosmos` and others still did. Every encoder except SD1.5 / SD2 / SDXL now gets prose, including ones added later.
  - *The subject was rewritten.* A prompt containing "accurate", "precise" or "visualize" had "design", "process", "generation" and others swapped for unrelated phrases. The subject is now passed through untouched.
  - *Negative encoded for nothing.* New `negative_mode` (Auto / Always encode / Zero). Auto returns a zeroed negative, as ComfyUI's ConditioningZeroOut, on Flux, Flux.2 and MiniMax H3 when no custom negative is typed: ComfyUI skips the negative at CFG 1, so that text-encoder pass was wasted. Choose Always encode to run those models above CFG 1 with a negative. The widget is appended, so saved workflows load unchanged.
  - Tests: `tests/test_prompt_encoder_fixes.py` (27, all failing on the old encoder).

- **Upscale section: controls that did nothing (3.5.0).**
  - *The method menu did nothing on the default path.* Upscale 32-bit (untiled), Upscale By Size and Downscale 32-bit on CUDA ran lanczos, lanczos4, mitchell and catrom as torch bicubic, and hermite and gaussian as bilinear; only the tiled CPU path had the real kernels. Every path now resamples with the named kernel from one weight builder, so GPU, CPU and tiled give the same picture (within 1e-5). On the GPU it is one matmul per axis; 1080p to 4K lanczos takes 0.6 s on a CPU, and frames too large for the matrix fall back to a per-tap gather.
  - *Downscale 32-bit `antialiasing` was never read.* It is the prefilter width now: 0.5 (the default) is the kernel stretched by the scale factor, which is what the node always did, 0 turns the prefilter off (sharp, aliases), 1 doubles it (softest). Alpha no longer goes through the sRGB curve on the CUDA path.
  - *"Auto" colour space meant Linear.* Auto now reads the image: Linear when any value is outside 0-1 (HDR), sRGB otherwise, and Upscale 32-bit's `info` says which it chose.
  - Tests: `tests/test_upscale_kernels.py` (46); live ComfyUI run of the three nodes: success.

- **Color section: controls that did nothing (3.5.0).** From the "controls that do nothing" list in KNOWN_ISSUES.
  - *LUT and LUT Blend `log_space` was a plain exponent.* On, it raised the pixels to 10^x, 2^x or e^x, which matches no camera curve, so a LogC3 to Rec.709 show LUT could not be fed correctly. It now encodes the scene-linear image into the LUT's input (`log_encoding`: ARRI LogC3 / LogC4, S-Log3, V-Log, Log3G10, Canon Log 3, DaVinci Intermediate, ACEScct, sRGB, PQ, HLG...) through the same encoder Read and Write use, OCIO when available. New `working_space` (appended, so saved widget order holds) gives the image's primaries. 18 % grey through an identity LUT reads 0.391, ARRI's LogC3 value. Saved graphs holding Log10 / Log2 / Ln still validate and render as before; those names are no longer offered. Alpha no longer goes through the exponent.
  - *HueCorrect dropped `grade_info`.* It now has a `grade_info` output (appended) carrying the mode, points, strength and the upstream record. A curve that does not parse is an error; it used to give an unchanged image with no message.
  - *OCIO Context never ran.* It was not an output node and nothing read its output, so ComfyUI skipped it and the config was never loaded. It is an output node now, shows the loaded config on the node, raises when the file is missing or OCIO rejects it, and its `ocio_context` output is an optional input of Color Space Convert, which reloads that config if another replaced it. `working_space`, stored and never read, is removed; a saved graph that still sends it runs.
  - Tests: `tests/test_color_controls.py` (18); live ComfyUI run of all four with a legacy `Log10` graph: success.

- **Cinematic Encoder, step 2 (3.5.0).** Found while scoring every node for the 3.5 freeze.
  - *Menu labels were encoded as weights.* ComfyUI reads brackets as emphasis, so "Close-Up (CU)" gave `CU` 1.1x weight and "f/2.8 (Cinematic Separation)" did the same to its description: on SDXL for every bracketed entry (about 40 % of the menus), on T5 / LLM models for style and film stock. Every menu value now goes through a label table and reaches the encoder without a bracket. Brackets you type in the prompt are left alone.
  - *Prose read badly.* "through a 85mm", "Rule of Thirds of a man", "Medium Close-Up MCU of", "bathed in Studio Strobe 3-Point Setup" and an "Aesthetic:" header. Each framing and lighting entry now has its own wording ("A low-angle hero shot of a detective. ... Lit by hard film noir lighting."), with "a" / "an" chosen by sound.
  - *Wan and LTX negatives were cut to Soft.* They run real CFG and their reference workflows use a long negative, but the node treated them like Flux and the UI marked the field "no CFG, ignored". Only Flux, Flux.2 and MiniMax H3 are cut now.
  - *"text" in every auto-negative.* It suppressed signs, titles and captions on Flux, SD3, Qwen-Image, HiDream and the other models that can draw lettering. It is added on SD1.5 / SD2 / SDXL only.
  - *Presets did nothing outside the UI.* The 7 widget fields were filled by the frontend, so a workflow queued through the API got film stock, shutter and aspect only. When all 7 are still "None" the preset is now applied in Python. A widget you set is kept.
  - *An untouched node encoded "A cinematic scene...".* `base_prompt` is empty with a placeholder; an empty prompt is refused with a message.
  - *`token_count` counted padding* on Wan, Flux.2, Qwen, Gemma and AuraFlow encoders. It now counts on the encoder the model reads, with each tokenizer's own pad id.
  - The preset table lives in Python only; `js/radiance_prompt.js` reads it from `/radiance/prompt/presets` instead of keeping a copy.
  - Tests: `tests/test_prompt_quality.py` (248). The functional suite now has a stand-in CLIP, so the Cinematic Encoder and Video Prompt Builder run there instead of being skipped.

- **Synced with upstream (3.5.0).**
  - *MiniMax H3 image guides crashed at 720p* (beta PR #75, merged). Keyframe latents were not padded to the 2x2 patch, so 45 latent rows failed in `patchify_video`. Resolution now aligns MiniMax H3 to 32 px and `crop_bbox` crops back after decode.
  - *Write left nothing in ComfyUI history* (from public PR #18). As an output node with no `ui` block, `/history` had no record of what Write saved. Write and Digital Cinema Write now list every file in `radiance_files`; PNG / JPEG / WEBP inside the output directory also go to `images` for the preview. The node still returns `(saved, count)` as `result`.
  - *Platform requirements downgraded ComfyUI's packages* (from public PR #18). `requirements_{windows,linux,mac_silicon}.txt` and `pyproject.toml` capped `Pillow<12`, `transformers<5` and nine others, so installing them into ComfyUI's environment could downgrade what ComfyUI or other packs had. Lower bounds only now, matching `requirements.txt`.
  - Tests: `tests/test_write_history.py` (5).

- **Viewer phase 2 (3.5.0).**
  - *Scopes measured the wrong thing.* Waveform, vectorscope, histogram and the reference scopes read the ungraded source texture or the 8-bit canvas (with overlays in it). They now read a display-signal render: graded, through the active view, with no overlays, false colour or viewer exposure. Checked: a 0.5 grey reads 128 on the scope with false colour on.
  - *Vectorscope.* Targets sat at the wrong angles. It is BT.709 Cb/Cr of the encoded signal now (100 % red at 102.9°), with 75 % and 100 % boxes and the skin line, on the CPU and GPU scopes alike.
  - *Warnings never fired.* Clip and gamut ran after the display clamp. Clip now flags the pre-clamp display value (≥ 0.999, red) and all-black (blue); gamut flags negative scene-linear (magenta). The grade functions no longer clamp negatives away first.
  - *False colour.* Custom bands replaced by ARRI's, on the Rec.709 signal of scene luma (18 % grey = 40.9 %, green).
  - *Viewer exposure and gamma were grade state.* They are viewer-only now, like Nuke: checked at screen 175, scope 128, export 0.214 for +1 stop on 0.214.
  - *Zoom and sharpness.* Nearest-neighbour did nothing above 1:1 and the main Viewer's canvas was CSS-pixel sized. The canvas is device-pixel sized and zoom is crisp by default.
  - *Banding.* 8-bit output is TPDF-dithered.
  - *P3.* The `colorSpace` attribute WebGL ignores is gone. On a P3 monitor a Display menu offers Display P3, which uses the OCIO P3 view and a `display-p3` drawing buffer and canvas.
  - *Keys hit every viewer.* The viewer and dock handlers listened on the whole page and fired inside text fields. They now act only for the active viewer, skip inputs and Ctrl/Cmd combinations.
  - *Transport.* In/out points, J/K/L shuttle, ping-pong and play-once, play every frame (waits for each frame) or realtime with a dropped-frame count.
  - *WebGPU.* `FEATURE_PARITY` is false and the gaps are listed in KNOWN_ISSUES; WebGL stays the default.
  - Tests: `js/tests/viewer_color.test.mjs` grows to 16 browser checks (shader compile, ARRI green, gamut, scope signal, viewer f-stop, DPR 2 crisp zoom, key scoping, ping-pong, P3).

- **Rival Film Grain and Motion Blur classes deleted (3.5.0).** `radiance.film` declared its own `RadianceFilmGrain` (film-stock profiles, `film/grain.py`) and `RadianceMotionBlur` (directional / radial / zoom, in `film/camera.py`) under the keys the VFX menu already ships from `nodes/vfx/optics.py` and `nodes/vfx/motion_blur.py`. ComfyUI never loaded them; they were kept only until the owner chose. The shipped nodes stay, so no saved graph changes; `film/grain.py` and the film Motion Blur class are removed.

- **VFX nodes that broke or crawled at production size (3.5.0).** Found by timing every VFX node on 24 frames of 1024x576 (CPU, 6 GB). Each fix was checked against the old maths.
  - *Motion Blur held the clip several times over.* The Motion Blur in the menu (`nodes/vfx/motion_blur.py`, vector blur) integrated the whole clip at once on the CPU with a copy of the vectors: +1 GB for 24 frames of 1024x576. It now works a few frames at a time on the GPU: +0.48 GB, output identical.
  - *Multipass Relight was killed out of memory.* The whole clip was shaded at once on the CPU with about 25 frame-sized temporaries, and unconnected passes were full frames of a constant. It now shades a few frames at a time on the GPU with broadcast defaults: 4.0 s. Output identical across 96 combinations of light, normal convention, premultiply, beauty mix and passes. The 0..1-or-signed normal decision is still made on the whole clip.
  - *Linear Matting took 4.5 s a frame.* Its six box means were square 2-D pools (cost grows with radius squared). They are now two 1-D passes, which is the same zero-padded mean, on the GPU in chunks: 107 s to 13.5 s on this machine, peak +1.8 GB to +0.65 GB, output within 2e-6.
  - *HDR Stitch took 24 s to paste a small crop.* Same cause in the feather, plus two full-frame pyramids for the whole clip at once: now 3.1 s and +0.66 GB, output within 5e-6.
  - *Floyd-Steinberg dither looped over every pixel in Python* (8 s a 1024x576 frame, about 2 minutes a 4K frame). It now processes one anti-diagonal of every frame and channel at a time, with the loop's arithmetic and the loop's order of error additions, so the result is bit-identical: 0.44 s for one frame, 1.5 s for 24.
  - *Depth Map Generator kept the clip's depth on the GPU.* Every full-resolution depth frame, then a 3-channel copy of all of them, stayed in VRAM (about 32 GB for 240 frames of 4K), and frames went through the Hugging Face processor on the CPU one at a time. Frames are now preprocessed on the device with the processor's own steps (torchvision's uint8 bicubic resize, so the input matches it to 2e-7), run through the model up to 8 at a time, and moved to the CPU as they are produced; normalisation is unchanged. On CUDA the model runs in fp16: against fp32 the Small and Base models differ by 0.03-0.07 % of the depth range on average, with no NaNs.
  - New `core/tensor/chunking.py`: frames-per-chunk from free VRAM (or 1 GB on CPU) and the compute device. The timings above are CPU; GPU gains are larger and still to be measured on the RTX 4080.
  - *Linear Matting crashed on a mask of another size.* Load Image returns a 64x64 mask for an image with no alpha; the mask is now resized to the image.
  - Tests: `tests/test_vfx_efficiency.py` (28), pinning each node to its old maths and chunked runs to single-pass runs.

- **VFX nodes held the clip several times over (3.5.0, phase 2).** Ten more nodes built every intermediate for the whole clip at once on the CPU. They now work a few frames at a time on the GPU (or the CPU when there is none; Optical Flow's DIS solver is OpenCV on the CPU), and set-up work that is the same for every frame is done once. Output checked against the old code: identical for Optical Flow's vectors, Lens Distortion, Chromatic Aberration, Depth of Field, HDR Grain Matcher, Relight Engine and Multipass Composite; within 2.4e-7 for Anamorphic Streaks and Film Grain; within 7e-6 for Subpixel Stabilizer. A seed gives the same grain as before. CPU, 1024x576, peak memory above the input:

  | Node | 24 frames | 64 frames |
  |---|---|---|
  | Lens Distortion | +768 MB to +356 MB, 2.3x faster | +2.0 GB to +0.9 GB |
  | Depth of Field | +1.16 GB to +0.98 GB | +3.1 GB to +1.2 GB |
  | Anamorphic Streaks | +657 MB to +495 MB | +1.7 GB to +1.2 GB |
  | Film Grain | +1.14 GB to +1.09 GB | +3.0 GB to +2.1 GB |
  | Chromatic Aberration | +389 MB to +287 MB, 3x faster | +1.0 GB to +0.9 GB |
  | Subpixel Stabilizer | 1.4x faster (batched FFT, no per-frame `.item()` reads) | +959 MB to +880 MB |
  | HDR Grain Matcher | +1.03 GB to +0.66 GB, 1.4x faster (each reference frame's grain computed once) | +2.6 GB to +1.2 GB |
  | Relight Engine | +1.27 GB to +0.71 GB | +3.3 GB to +1.2 GB |
  | Optical Flow | +826 MB to +609 MB, 1.3x faster (DIS built once per chunk, not per pair) | +2.2 GB to +1.3 GB |
  | Multipass Composite | +1.30 GB to +1.20 GB (four full outputs) | both out of memory on this 8 GB machine with all seven inputs connected |

  - *Lens Distortion with Invert crashed.* It passed a tensor as the fill value of `full_like`. Fixed; the inverse keeps the sign of tiny values.
  - New `FrameSink` in `core/tensor/chunking.py` collects chunk results on the CPU, keeping a one-chunk result as it is so a short clip costs no more than before.
  - Optical Flow's Lucas-Kanade fallback still solves one pair at a time: its batched solve differs from the per-pair one by up to 0.07 px.
  - Tests: `tests/test_vfx_efficiency.py` grows to 40.

- **EXR Passes Writer, Compression Artifacts and Scene Cut Detect ran one frame after another (3.5.0, phase 3).** CPU, 1024x576:
  - *EXR Passes Writer* writes frames in parallel (OpenEXR compresses outside the GIL; up to 8 at once) and converts each pass one frame at a time instead of copying the whole clip first. 24 frames with beauty, normal, albedo and depth: 11.1 s to 5.3 s on this 2-core machine, and it scales with cores. Every file's header, channels and pixels are identical to before (180 files compared across layouts, bit depths and compressions).
  - *Compression Artifacts* encodes frames in parallel and writes them straight into the output instead of a list that was then stacked: 64 frames 5.3 s to 2.2 s, +888 MB to +485 MB. Identical output, and the seeded noise is still drawn frame by frame in order.
  - *Compression Artifacts crashed in JPEG or Both mode* when a side was not a multiple of `block_size` (100x100 with the default 8 raised ValueError). Blocks cut off at the right or bottom edge are now averaged over the pixels they have; whole-block images are unchanged.
  - *Scene Cut Detect analysed every frame twice*, once in each pair it belongs to. Each frame's histograms and edge map are now computed once: 64 frames 4.3 s to 2.3 s, cuts and scores bit-identical.
  - Tests: `tests/test_vfx_efficiency.py` grows to 43. Three chunked-versus-one-pass checks compare within 1e-6 instead of bit for bit: a vectorised reduction can round differently with the batch's memory alignment, and one did once in about 50 runs.

- **Roto removed (3.5.0).** `◎ Vector Mask Draw (Roto)` (`RadianceVectorMaskDraw`) typed polygon or spline points as text, with no way to draw on the image, so it was slower to use than ComfyUI's own mask editor or a SAM / matting node. It is deleted; 156 nodes load. A saved graph that used it shows it as a missing node. The Video Mask Propagator that shared its file stays, now in `nodes/vfx/mask_propagate.py`.

- **Viewer: Simple / Advanced, one compare, Lite Viewer removed (3.5.0).**
  - *Simple mode.* A switch in the Viewer's title bar. Simple shows the picture, a transport (play, step, scrub, frame) and compare; Advanced shows every panel as before. A new Viewer opens in Simple; the choice is saved with the node. Graphs saved before the switch open in Advanced, as they looked. Choosing Advanced grows a small node to 1180 x 760.
  - *Compare did not work as labelled.* A/B grabbed a still of A over a connected `compare_image`, so A was compared with itself. Wipe with no B showed the ungraded source. Difference and Blink drew nothing on the default WebGL path, and Blink started playback. One controller now drives every compare control in both modes: A, B, Wipe, Diff (|A − B| × 4) and Blink (two flips a second), drawn in the shader. B is the `compare_image` frame under the playhead; with none, **Pin A as B** keeps the current frame, read at image resolution so it lines up at any zoom, and it survives a new run. **Release B** returns to the input.
  - *Lite Viewer removed.* Simple mode does its job, and two viewers meant two frontends and two sets of compare bugs. `◎ Radiance Lite Viewer` (`RadianceLiteViewer`) is deleted: `nodes/monitor/lite_viewer.py`, `js/radiance_lite_viewer.js` and their tests. 157 nodes load. A graph saved with one opens with a Radiance Viewer in its place, in Simple mode: the frontend rewrites the node before ComfyUI's missing-node check, keeps `input_space` and `fps`, and re-points its links by socket name (root graph and subgraphs). `workflows/start.json` uses a Viewer in Simple mode instead.
  - *Context loss reported on every graph load.* Removing a viewer releases its WebGL context on purpose, and the context-lost handler logged that as an error and kept the context restorable. It now ignores its own release.
  - *Docs.* The README key table listed A for compare and L for luma; compare is X, luma is Y, A is alpha, and J / K / L are the shuttle.
  - Tests: `js/tests/viewer_compare_mode.test.mjs` (13, including the load-time conversion). `tests/test_viewer_phase1.py` checks the node is gone.

- **Viewer player, checked end to end (3.5.0).** Driven in a browser on a bar-coded 96- and 240-frame clip and a 48-frame PNG sequence, reading the frame number back from the pixels.
  - *Playback froze at the end of the range.* The loop checked whether the in point was loaded whatever the loop mode, so ping-pong and play-once waited at the out point for a frame they would never show, and a whole-clip loop longer than the 16-frame paging window waited for a frame 0 that had been paged out. One function now decides the next frame for both the check and the step; a loop wrap moves the playhead so the window re-centres and loads the in point first.
  - *The node grew without limit.* The viewer widget sized itself from the node (height minus 110), so LiteGraph grew the node every frame: 760 to 5688 px in 3 s, with the picture pushed off screen. The widget has a fixed minimum height now and the node keeps its size.
  - *J did nothing on a video loaded straight into the Viewer.* Browsers cannot play a video element backwards; J now steps back by seeking, at the playback rate, and wraps when looping. K, L and Space stop it.
  - *Every frame went through the CPU.* The main 2D canvas asked for `willReadFrequently`, which makes Chrome keep it in software, so the WebGL frame was copied back to the CPU on every draw. The flag is gone; the pixel readouts read the WebGL buffer.
  - Tests: `js/tests/playback_step.test.mjs` (8). The browser run passes 20 of 21 checks on the 240-frame clip, including bounded memory over a full loop; the one left is the frame rate, which this software-GL machine cannot reach.

- **HDR VAE Decode and SDR → HDR audit (3.5.0).**
  - *Auto log-inverted sampled latents.* Samplers copy the latent dict, so HDR
    Encode's `radiance_meta` survived KSampler and Auto (and even Sampler
    mode, through the engine's metadata override) ran the log decompression
    over a diffused latent: blown-out frames on every img2img graph started
    from HDR Encode. Encode now stamps a latent fingerprint; decode trusts the
    HDR coding keys only when it still matches, keeps the padding keys, and
    reports `radiance_meta_live`.
  - *Rec.709 → ACES2065-1 was off by 1.2 %* in `hdr/vae.py` and
    `color/ops.py` (Universal's AP0 output), and the two AP0 matrices were not
    inverses. Both now match OpenColorIO to 1e-7; P3-D65 → Rec.2020 corrected
    too.
  - *Hidden widgets that still steered the decode.* `hdr_output`,
    `hdr_scale_factor` and `inverse_tonemap` kept acting while the frontend
    hid them. Every mode now decides them itself: Sampler mode honours the
    visible `target_space` (Linear really is linear) and keeps the range
    `inverse_tonemap` creates; Direct HDR honours scene-referred targets
    (ACEScg, ACES 2065-1, Rec.2020, camera logs). `source_space` is read from
    the latent. Auto shows the controls of both paths.
  - *Direct HDR details.* `exposure_adjust` is applied in scene-linear after
    the pixel model instead of clipping highlights before it;
    `hdr_scale_factor` no longer scales alpha; RHDR is written after scale and
    crop, from the image the node returns, and Auto can export it;
    `crop_bbox` is clamped to the image; a batch of stills is no longer
    treated as a clip by the pixel model (the latent decides).
  - *Log targets were transfer-only* and log sources fed camera-gamut linear
    to SDR modes; see upgrade note 6. A Compress (Log) round trip back to its
    own log space stays exact.
  - Video alpha was assigned to the wrong frames when the batch was > 1, and
    the pre-tonemap scene-linear copy was cloned on every Compress (Log)
    decode even with no RHDR export.
  - HDR Encode / HDR Monitor HLG reference white (upgrade note 6).
  - Measured on the shipped pixel checkpoint: every Universal and Recover
    control changes the output, clipped highlights reach the mastering peak
    and never exceed it, pixels outside the recovery mask equal Expand.
    `tests/test_hdr_vae_sdr2hdr_audit.py` pins all of it against OCIO.

- **Sampler speed (from the 3.5 live logs).**
  - *A second model load per stage.* Every stage called `load_model_gpu()`
    and then `sample_custom`, which loads the model again with inference
    memory reserved; the log shows each model "prepared for dynamic VRAM
    loading" twice, and a Z-Image run spent ~4.5 s outside its 8-step loop
    against ~1.9 s for ComfyUI's own path. Removed.
  - *cfg 1.0 silently became the base CFG.* In the Auto preset a cfg of 1.0
    was replaced by the architecture's base value (Z-Image 4.0, Wan 6.0, ...)
    whenever model_meta was not connected, so turbo / distilled checkpoints
    ran an unconditional pass on every step: twice the compute and the wrong
    look. It is now applied only when model_meta names the checkpoint, and
    suggested in the log otherwise. Dynamic CFG no longer lifts 1.0 to 1.2.
  - *~0.5 s before the first step.* `gc.collect()` plus two
    `torch.cuda.empty_cache()` calls ran on every sample; removed (ComfyUI
    manages the cache between prompts). `log_tensor` built its statistics
    (a float copy and four GPU syncs per tensor) with DEBUG off; it returns
    early now.
  - *Flux ran on the wrong text encoder.* With `clip_l` empty, ComfyUI's
    loader picks Mochi's T5 encoder for a lone `t5xxl` (the log shows
    `MochiTEModel_` and `target_arch='wan'` for a Flux graph). The Loader
    now fills a required slot from the one matching file on disk
    (`clip_l.safetensors`) or stops naming the slot (Flux, SDXL,
    HunyuanVideo, HiDream, Kandinsky 5).
  - *VRAM warning with the real size.* The table estimated Flux dev bf16 at
    16.5 GB; its weights are 22.7 GB. The Loader now reads the model's own
    size and, when it exceeds the GPU, says the weights will be streamed
    from system RAM every step and what fp8 would bring them to.

- **OCIO sets itself up; OpenEXR is always on.** Nothing to configure after
  install. OpenColorIO moves to required dependencies (and `install.py`
  installs OpenEXR and OpenColorIO if ComfyUI-Manager did not). At startup
  `radiance.color.ocio_setup` keeps a valid `$OCIO`, otherwise activates
  OpenColorIO's built-in ACES 2.0 studio config: written once to
  `ACES/studio-config.ocio`, exported as `$OCIO` for the process, and set as
  OCIO's current config so every node agrees. A broken `$OCIO` is reported
  and bypassed, never overwritten. OCIO < 2.2 falls back to the bundled CG
  config. The startup download of a config from GitHub (no consent) is
  removed. `OPENCV_IO_ENABLE_OPENEXR` is forced to `1` at the top of the
  package (a `0` in the shell used to win, and OpenCV caches it at first
  use). `RadianceColorSpaceConvert`'s OCIO name map pointed at names no
  config had, so OCIO never ran; it now targets the studio config's names
  (camera logs stay analytic, per that node's transfer-only convention).

- **Read / Write colour management and precision.** Checked parameter by
  parameter and format by format; `tests/test_io_release_matrix.py` (65
  tests) writes through the Write node and reads back through the Read node.
  - *Camera logs decoded to the wrong gamut.* Read undid the transfer only, so
    LogC4, S-Log3, V-Log, Canon Log 3, Log3G10 and DaVinci Intermediate
    plates stayed in the camera's own primaries (AWG4 red read as a
    different red in a Rec.709 or ACEScg graph). Every encoding now goes to
    the working space, transfer and primaries, through OpenColorIO's ACES
    studio config when OCIO is installed and an analytic path otherwise; the
    two agree within 2e-6 on every encoding (`tests/test_color_encodings.py`).
  - *New inputs on both nodes:* `working_space` (Linear Rec.709, ACEScg,
    Linear Rec.2020, Linear P3-D65, ACES2065-1), `ocio_colorspace` (any
    name or alias from the config), `ocio_config` (path or `ocio://`, else
    `$OCIO`, else the built-in studio config) and `hdr_reference_nits`.
    New encodings: Linear Rec.709 / Rec.2020 / P3-D65, ACES2065-1, P3-D65
    gamma 2.6, Rec.709 camera OETF, Rec.2020 OETF, Sony S-Log3 S-Gamut3;
    Write gains V-Log, Canon Log 3, Log3G10, DaVinci Intermediate and ACEScct.
  - *PQ and HLG were 5x off Radiance's own HDR scale.* Write encoded PQ with
    1.0 = 1000 nits and HLG with 1.0 = peak; both now place scene-linear 1.0
    at `hdr_reference_nits` (203, BT.2408, as the SDR → HDR nodes do), and
    both convert to Rec.2020 primaries first (they wrote Rec.709 primaries
    into an HDR10 file).
  - *Video was encoded with the BT.601 matrix and no tags.* ffmpeg's RGB →
    YUV defaulted to BT.601, so a pure red came back (255, 24, 0) in any
    BT.709-aware player; nothing was tagged. Every codec now uses the BT.709
    or BT.2020 matrix, is tagged with primaries, transfer and matrix (HDR10
    as bt2020 / smpte2084 / bt2020nc), and is fed 16-bit RGB (H.265 10-bit
    got 8-bit before). ProRes 4444 carries the mask as alpha. DNxHR HQ is
    written as `.mov` as its label says (was `.mxf`). `broadcast_safe` no
    longer squeezes video to legal range twice.
  - *EXR / DPX carry their colour.* EXR gets `chromaticities`,
    `oiio:ColorSpace` and (for ACES2065-1) `acesImageContainerFlag`; DPX
    gets transfer / colorimetric. Read on `Auto` honours an EXR's
    chromaticities, so an ACES2065-1 EXR lands in the working space.
  - *Precision.* 16-bit grey PNG/TIFF (depth maps, mattes) read almost
    entirely white through Pillow's 8-bit convert; half-float TIFF did not
    read at all; DPX and grey+alpha files dropped their alpha; 8- and 16-bit
    writes truncated instead of rounding. All fixed. Alpha now writes to
    TIFF, DPX and WEBP as well as EXR and PNG. EXR compression gains PXR24,
    B44 and B44A. `.hdr` and 16-bit PNG writes check their result. A decode
    that fails raises (it logged and passed file values on as linear).
  - `layer: depth` finds a `depth.Z` channel, as in Nuke.

- **Honest release pass.** Every control, option and output was checked
  against the code that consumes it; the ones that did nothing, did
  something else, or reported success after falling back were fixed, and the
  few that cannot be made real in this release are labelled as what they
  are.
  - *Video.* I2V `concat_channels` concatenated image channels onto the noise
    (no model accepts that) and `clip_vision_inject` wrote a key nothing
    reads; both now write what ComfyUI's WanImageToVideo writes
    (`concat_latent_image` + `concat_mask`, `clip_vision_output` from a new
    input), `auto` picks from the model's actual inputs, and a strategy that
    cannot work raises. The T2V/I2V pipelines built LTX-shaped latents (128
    ch, /32) for every model because the DiT spec import always failed; the
    real table is used and the latent shape follows the connected model and
    VAE (Wan 2.2 TI2V 5B and HunyuanVideo 1.5 added). `hdr_strength` is now
    the prompt weight of the HDR descriptors. Video HDR Conditioner appended
    words to a key encoded conditioning never has; with `clip` connected it
    now encodes and concatenates them, and says when it could not. Video HDR
    Decode converts Rec.709 to the metadata gamut before PQ/HLG (it printed
    the gamut and never converted). Batch Decode `output_linear` linearises.
    Video Export's EXR fallback wrote PNGs that failed on RGB and still
    reported EXR; it writes EXR or raises.
  - *Upscale.* A diffusion run with no diffusion model was labelled a
    diffusion upscale; `pass_info` now says what ran on how many tiles, and a
    bicubic fallback says it is not an AI upscale. `model_tier auto` uses the
    content analysis and `mode balanced` sharpens, as their tooltips said.
    Face Restore `auto` never restored (the default label contains "skip"),
    counted crops pasted back unchanged as restored, and ignored fidelity on
    the spandrel path. Upscale Video crashed whenever `sharpness_boost` > 0.
    The confidence output is documented as the tile-geometry weight it is.
  - *VFX.* Video Mask Propagator read the wrong frame's flow with the wrong
    sign (IoU 0.6 after one frame, 0.0 after three; now > 0.92). Vector Mask
    `Bezier_Spline` drew the polygon; it draws a closed spline. Motion Blur's
    `energy_conservation` rescaled the whole frame by one highlight. Camera
    Sync ignored the `shutter` key its tooltip documents and returned a
    default camera for a missing file or an `.abc`. Linear Matting listed
    ViTMatte and RVM, which never existed. DSINE loaded any checkpoint with
    `strict=False` and fetched code through torch.hub without download
    consent; both fixed, and the passes record which normals were used.
  - *QC / IO / audio.* Policy Guard checked only frame 0 and enforced
    `max_peak_nits` only below 200 nits. HDR Diagnostics ignored its
    `colorspace` input. Digital Cinema Read never wrote the `colorspace` key
    Linear Check reads, never used `fps_override`, and offered one input
    colorspace. Audio Cut returned `[]` for a missing file or a failed
    backend, which is also its answer for a quiet track.
  - *Generate.* Denoise `joint_chroma_guidance` reached only the Guided
    filter; Bilateral (the default) is now a joint bilateral. Regional Prompt
    `Replace` dropped the global prompt from the whole frame. NDI Sender
    streamed only the first frame of a batch.
- **Found by the live 3.5 run on ComfyUI 0.32.** HDR Stitch collapsed a
  one-frame batch (`squeeze(0)`) before the Laplacian blend and failed on
  every single image. T2V / I2V handed an image model a 5-D latent and died
  inside the model's forward; they now refuse it by name. Digital Cinema
  Read executed an empty path as an 8x8 black shot. `RadianceAIUpscale`
  downloaded weights without download consent.

- **The Viewer rendered black on the default (WebGL) backend.**
  `RadianceWebGLRenderer.setMask()` and the mask uniform upload wrote and read
  `this.mask.type` etc., but the v3.1 refactor (`19a440c`) moved the mask into
  the shared base class as flat fields and no constructor created `this.mask`
  any more. Every `render()` after a frame loaded threw
  `TypeError: Cannot set properties of undefined (setting 'type')` inside
  `setMask`, before the draw call, so the canvas stayed black. It went
  unnoticed for three months because WebGPU auto-upgraded on every machine
  with `navigator.gpu`; 3.4.0 made WebGL the default and the Viewer went
  black for everyone. The WebGL renderer now delegates to the base class.
  `js/tests/sequence_load.test.mjs` builds a real Viewer in Chromium, feeds it
  a PNG plus fp32 RHDR sidecar through `onExecuted`, and asserts the canvas is
  not black on the first frame and after a scrub; it failed before the fix.
- **The Viewer framed its image off-screen.** The second half of the black
  viewer, found live on ComfyUI 0.32 / frontend 1.48 once the render path was
  fixed. The sidebar and inspector reported their full content height
  (2019 px) with `min-height: auto`; the Vue node frontend sizes a node to
  its DOM widget's content, so a 1180x760 Viewer node became 1480x2286 and
  the canvas column stretched to 1637 px. `fitToView()` centred the frame on
  that canvas, below the visible part of the node, and `resize()` changed the
  canvas size without touching pan or zoom, so the view never recovered. The
  container now carries `contain: size` (the node's size drives the viewer,
  never the reverse; panels scroll inside it), and `resize()` refits a view
  that is still auto-fitted and keeps the centre point of one the user has
  zoomed or panned. The browser test hosts a second viewer in an auto-height
  parent and fails at 1954 px without the containment.
- **Sequential offload never engaged.** `setup_offload_mode("sequential")`
  called `comfy.model_management.set_lowvram_mode`, which ComfyUI has never
  shipped, so it logged "Could not enable sequential offload" and did nothing
  on every run. It now sets ComfyUI's module-level `vram_state` to
  `LOW_VRAM`, which `load_models_gpu()` reads, and leaves a stricter
  `NO_VRAM` alone.
- **The pixel checkpoint was only found under one name.** The resolver looked
  for `sdr2hdr_pixel_image.pt` and `sdr2hdr_image_50k.pt`; the training
  scripts emit `sdr2hdr_pixel_image_v1_50k.pt` and friends. Any
  `sdr2hdr_pixel*.pt` is found now, the preferred name first.
- **The Recover node reported only the last failure.** With no temporal
  checkpoint and no pixel checkpoint it now names both reasons, and it falls
  back from the temporal model to the pixel model per frame rather than
  failing a clip outright.

### Added

- **Every node and every input is documented.** All 149 menu nodes have a
  description and all 1,259 inputs a tooltip, each written from the code that
  reads it: units, range, colour space, what the default does, and what a
  control does not do. Before: 14 nodes had no description and 561 inputs no
  tooltip. About 80 existing tooltips and descriptions that were wrong were
  corrected (gamma directions, the Split View position, the Grade temperature
  scale, nits examples at 100 instead of 203, a LoRA environment variable
  that does not exist, and more). `tests/test_node_docs.py` fails when a new
  node or input lands undocumented.
- **Node reference, `docs/nodes/`.** One page per menu section with every
  node's inputs (type, default, range, meaning) and outputs, generated from
  the nodes by `tools/build_node_reference.py`, so it cannot drift:
  `tests/test_node_reference.py` fails when it is stale, when the README's
  node map disagrees with it, or when a local link in the docs is broken.
- **Five example workflows**, listed in ComfyUI's Templates browser: SDR to
  HDR, HDR through a VAE, Multipass Estimate and relight, a colour grade, and
  HDR delivery (EXR, tagged HDR10 and ACES 2.0 SDR from one master). Each was
  loaded from its saved file into a clean install and run. Tests check every
  Radiance node in a shipped workflow exists and is not retired, every
  required socket is connected, and no workflow carries a machine path.
- **Multipass Estimate.** Render-style passes from a plate, from trained
  models only.
  - *Geometry, MoGe-2 ViT-L (Microsoft, MIT), through ComfyUI's native MoGe:*
    metric depth (metres), camera-space world position (OpenGL axes, metres),
    normals (OpenGL or DirectX), a surface mask, and the estimated field of
    view. A model from Load MoGe Model can be connected instead.
  - *Computed from that geometry:* ambient occlusion as the GTAO integral
    (cosine-weighted, radius in metres, 1 = open), using normals taken from
    the point map so the occlusion plane matches the surface, and with the
    flying pixels along silhouettes excluded as occluders; and mean
    curvature in 1/metre, undefined (0) across depth edges. Checked on
    analytic scenes: a plane facing the camera is open (0.99), a 90 degree
    crease closes to about 0.6, a unit sphere reads curvature 1.0.
  - *Materials, Marigold IID Appearance v1.1 (ETH Zurich, OpenRAIL++-M):*
    albedo (linear), roughness, metallic.
  - *Lighting, Marigold IID Lighting v1.1:* diffuse and specular (residual)
    light. The model predicts them only up to scale, so they are fitted to the
    linear plate per frame by non-negative least squares; `info` reports the
    scales and the rebuild error (11 percent RMS on an indoor photo).
  - *Measured:* backward motion vectors from DIS optical flow, in pixels,
    +y up.
  - Weights download on first run (about 4.9 GB, pinned commits) while the
    node's `download_missing_models` switch is on; `RADIANCE_ALLOW_DOWNLOADS=0`
    or `HF_HUB_OFFLINE=1` always stops them. On the CPU only one model is held
    in memory at a time, the Marigold text encoder is used once to cache the
    empty-prompt embedding beside the weights, and all frames go through one
    model before the next.
  - Removed with the old node: emission, transmission, reflection mask,
    segmentation ID, highpass, the tone masks and the blur-difference
    "specular". None of them measured what its name said.
  - `diffusers>=0.33` and `accelerate>=0.26` are now dependencies (install.py
    installs them for ComfyUI-Manager installs).
- **ComfyUI 0.32 model families.** Qwen-Image, Krea 2, HiDream-I1, OmniGen2,
  LongCat-Image, Kandinsky 5 (video and image), HunyuanImage 2.1 (64ch, 32px)
  and HunyuanVideo 1.5 (32ch, 16px / 4 frames) are detected from their
  checkpoint keys, carry latent, VAE-factor, text-encoder-slot, VRAM and
  sampler defaults (each read off the model's official Comfy-Org workflow
  template, or its `sampling_settings` in `comfy/supported_models.py` where no
  template exists), and appear in the Loader's model list and the
  Resolution presets.
- **ComfyUI-native detection fallback.** When none of Radiance's key
  heuristics match, `detect_model_type` hands a shape-only view of the
  checkpoint to `comfy.model_detection.model_config_from_unet` and maps the
  answer back, so a checkpoint the running ComfyUI can load is never reported
  as unknown; families without a Radiance table are named in the log.
- **`radiance.model.paths`.** One resolver for Radiance's own checkpoints:
  registers `models/radiance` with `folder_paths` (so `extra_model_paths.yaml`
  works), searches every registered folder, and describes where it looked
  when nothing is found.

### Changed

- **README is for users; the development record moved** to
  `docs/DEVELOPMENT.md`. The node map links to the generated reference, and
  its hand-written table, which had drifted (Project Manager under Core, QC
  under Color), is replaced.

### Removed

- **Release clean-up: files nothing loaded, called or documented.**
  `core/param_memory.py` (a SQLite parameter-history node never registered
  in the catalog), `lut_utils.py` (no importer), the empty `nodes/training`
  group and its unused `sdr_degradation.py`, `extras/nuke_scripts/radiance_client.py`
  (a client for a `RadianceNukeServer` node that does not exist; the Nuke
  path is `scripts/start_nuke_server.py`), `js/radiance_vfx_multipass.js`
  (styled a `RadianceVFXMultipass` node that does not exist),
  `js/docs/` (a stale offline manual quoting 121 nodes that ComfyUI loaded
  as an extension on every page in git installs), `scripts/resolve_bridge.py`
  (posted a placeholder graph of nonexistent nodes),
  `scripts/radiance_batch_convert.py`, the Poly Haven / AmbientCG HDRI
  downloaders and the Wan LoRA data-prep scripts (hard-coded local drive
  paths, VAE training that was retired), `tools/make_validation_contact_sheet.py`,
  `tools/clean_release.py` (its dead-file list named files already gone),
  `rpacks/SDXL_Standard.rpack`, and three unreferenced images (`icon.png`,
  `Viewer_shortcut.png`, `radiance_workspace.png`). The unused
  `DYNAMIC_EXEC_ENABLED` flag in the Nuke listener went with them; nothing
  read it. A second pass removed five more: `nodes/hdr/patch.py` and
  `nodes/monitor/scopes.py` (empty modules that registered nothing),
  `tools/validate_hdr_pipeline.py` (imported a `radiance_color` module that
  no longer exists, so it could not run), and `scripts/build_js.js` with
  `package-lock.json` (minified the frontend into a `build/` folder nothing
  loads; `package.json` keeps only `npm test`).
- **`tools/check_release_ready.py` works again and runs in the suite.** It
  had crashed since `license` became an SPDX string, looked for README
  headings that no longer exist, and flagged local `__pycache__` as release
  content. It now checks the SPDX licence and its file, that pyproject, the
  runtime `VERSION` and the README node badge agree, that every listed
  package and package-data pattern exists, and that no generated file is
  committed.
- `fast_vae.py` (latent RUDRA decoders and their loader), `model/vae.py`,
  `tools/build_rudra_cache.py`, `tools/validate_rudra_dataset.py`,
  `tools/compute_descriptor_stats.py` (all imported a `rudra` package that
  was never in the tree), `scripts/training/train_turbo_decoder.py`,
  `dataset_hdr.py`, `verify_unified_config.py`, `train_all_models.bat`,
  `train_pipeline.bat`, and the tests that covered them. The
  `RADIANCE_TURBO_DECODER` environment variable is no longer read.

## [3.4.x unreleased work, folded into 3.5.0]

### Upgrade note

Four changes alter what an unchanged graph produces. Read these before updating
a project in flight.

1. **PQ output moves by a factor of ten at any mastering peak below 10 000
   nits.** `RadianceHDREncode` in PQ (HDR10) normalised scene-linear by
   `peak_nits` where ST.2084 requires the fixed 10 000 cd/m² ceiling. At the
   shipped default of 1000 nits, diffuse white encoded at PQ 0.8290, which a
   conforming display renders at 2030 cd/m² instead of the 203 the widget
   promises. The error cancelled at `peak_nits = 10000`, which is the single
   value the old test used, and it cancelled again on a round trip through
   Radiance, so it was invisible except against another tool. Any PQ master
   graded before this release was delivered ten times too bright and needs
   re-rendering. Decoding moves the same way in the opposite direction:
   `RadianceColorSpaceConvert` reading a real HDR10 file was ten times dark.
2. **Rec.709 and Rec.2020 output actually converts now.** `Rec.709` previously
   wrote untouched scene-linear, roughly 2.2 stops dark in the midtones,
   because the function it called did not exist and a blanket except returned
   the array unchanged. `Rec.2020` applied a 1/2.4 power curve with a hard clip
   to [0,1], which is neither BT.2020 primaries nor a BT.2020 transfer function
   and destroyed all highlight headroom on float formats. Both are now the real
   OETFs, and Rec.2020 applies the primaries matrix. A colour space that cannot
   be applied now fails the write instead of silently writing linear.
3. **Twenty-five nodes appear in the menu that were not there before**, taking
   the registered count from 131 to 156. Nothing was renamed or removed; these
   are finished classes that four implementation packages declared and the
   catalog never loaded.
4. **`pq_bt2408_to_linear` and `_eotf_pq` lost their `peak_nits` parameter**,
   and `_pq_encode` / `_torch_pq_encode` in `nodes/hdr/aces2.py` lost theirs.
   In every case the parameter was either applying the wrong normaliser or
   accepted and ignored. Removing rather than correcting them means a stale
   keyword call raises instead of quietly changing exposure.

### Added

- **Long-video sampling in overlapping latent windows.** `RadianceSamplerPro`
  gains `temporal_window` and `temporal_overlap`. Above zero, a 5D video latent
  longer than the window is denoised window by window with the overlaps
  cross-faded **at every denoising step** rather than after each window is
  finished, which is the difference between a seam you cannot see and one you
  can. Peak VRAM then follows the window size instead of the clip length. The
  default is 0, off, so no existing graph changes. The weighting follows the
  pattern `RadianceUpscaleVideo` already uses for pixels, including its
  half-sample ramp offset. CPU tests pin the schedule, the partition of unity,
  the step accounting, seed stability and that a single window reproduces the
  unwindowed result exactly; whether the output is temporally coherent on a real
  DiT needs a GPU and is recorded as open in the README.

- **The viewer pages frames instead of holding them.** A bounded frame window,
  16 frames or 768 MB by default, fetches on demand as the playhead moves and
  evicts what falls outside, with bounded fetch concurrency.

### Changed

- **The write path streams.** Sequence and video writes transform and write one
  frame at a time, and video frames are piped into ffmpeg's stdin as they are
  produced rather than stacked into one buffer first. Measured peak for 32
  frames against 512 frames differs by 0.1 MB where it used to differ by 360 MB,
  and enabling a colour transform now costs 1.6 MB rather than a second copy of
  the whole shot, so a colour space no longer halves the longest shot you can
  write. A streaming read path sits beside the batch one, so a read, transform
  and write pipeline never holds the sequence; the batch entry points are
  unchanged because the node layer genuinely does hand ComfyUI a batch.

- **The HDR VAE encode and decode stream.** Both wrote every frame into a list
  and then concatenated, so peak was two copies of the clip. They now write into
  one pre-allocated buffer and release as they go: decode overhead measured flat
  at 5.7 MB from 4 frames to 32, where it previously grew by a whole extra clip,
  and the RHDR sidecar accumulator is gone entirely because sidecars are written
  as each frame lands. Video encode and decode buffers now live on
  `comfy.model_management.intermediate_device()`, which is where `comfy.sd.VAE`
  already puts its own, so this is a device change for the video paths. The
  still paths are untouched.

- **Noise generation writes into a pre-allocated buffer.** Every 5D noise type
  built a Python list of per-frame tensors and then stacked it, so the list and
  the stack were alive together, and `stage_noise` allocated a fresh full-size
  zero tensor per stage.

### Fixed

- **Two regional-conditioning nodes were non-functional at their own defaults.**
  `RadianceRegionalPrompt` and `RadianceRegionalGrid` wrote a fractional `area`
  tuple without the `"percentage"` marker ComfyUI requires, so the floats
  reached `get_area_and_mult` and raised `TypeError: narrow(): argument 'start'
  must be int, not float` on any sampler.

- **The sampler returned the un-sampled input latent and logged success.** When
  `start_step >= end_step` the stage split collapsed to one element and the
  sampling loop never ran, so an empty-latent run decoded to flat grey while the
  node reported "Sampling complete". Reachable at `start_step=10, end_step=10`,
  at `start_step=20, end_step=5`, and at `start_step=50` with `steps=20`.
  `validate_step_range` existed for this and was imported but never called.

- **A maximum seed crashed video noise generation.** The 5D branch looped
  `torch.manual_seed(seed + f)` against a widget whose max is 2**64 - 1, so the
  top of the range raised `RuntimeError: Overflow when unpacking long` on every
  non-Gaussian noise type, outside the surrounding try. Adjacent seeds also
  produced shifted rather than independent sequences, since run S frame f was
  bit-identical to run S+1 frame f-1.

- **`tile_mode` silently discarded six settings**, breaking out of the stage
  loop so phase shift, refiner, dynamic guidance, start and end step and
  `add_noise` never applied, and a failed tile was replaced with the
  **un-denoised input slice**, feathered into the output and reported as a
  success. An OOM on one tile, the condition tiling exists to avoid, put raw
  latent noise into a finished plate.

- **The viewer showed an 8-bit tonemapped 2048px proxy labelled `FP32`.** When
  the RHDR path failed, and it had three silent ways to fail, the badge still
  read FP32 while a colourist graded against the fallback PNG. The badge now
  says what is actually on screen.

- **The viewer wrote roughly 67 GB per 500-frame shot**, two thirds of it for a
  status label: exposure bracketing was hardcoded on with no widget, writing a
  full 32-bit EXR, an fp32 RHDR and an rpick for each of three exposures per
  frame, of which the frontend fetched only the 8-bit PNG.

- **Temp files leaked.** The zdepth RHDR sidecar was stored under a key the
  purge loop did not collect, orphaning one file per depth frame forever, and
  `RadianceLiteViewer` wrote a full-resolution PNG per frame with no purge path
  at all despite a docstring promising compact previews.

- **`RadiancePreviewServer` returned a working-looking URL after a failed bind**,
  and leaked the previous `HTTPServer` whenever the port changed.

- **Video encode dropped the crop and the alpha.** The 4D-VAE video path
  hardcoded `pad_h`/`pad_w` to zero in the clip-level metadata, so decode skipped
  the crop and returned frames padded to the VAE factor with reflect-padded
  garbage at the bottom, and it discarded the alpha channel even with
  `alpha_handling="Preserve"`, returning solid white that decode then
  composited over the real matte.

- **`RadianceHighlightSynthesis` blacked out highlights.** Its Soft Light blend
  used the W3C formula, which is only valid on [0,1], so a 5.0 highlight went
  negative and clamped to pure black. The node exists to expand highlights.

- **`RadianceHDRExpansion` applied the inverse OETF to alpha**, returning 0.2140
  for an alpha of 0.5, because the transfer ran over the full tensor before RGB
  was sliced off.

- **`RadianceHDRBlendValidator`'s SSIM returned exactly 1.0 on HDR content**
  regardless of what it was comparing: it was a global statistic, not windowed,
  and clamped luma to [0,1] first, so two images above 1.0 everywhere both
  flattened to mean 1 and variance 0. Its "higher SSIM" winner branch also chose
  `image_a` unconditionally without computing a second value.

- **`_hdr_soft_decompress` mapped the top code value to 2.5 million linear.**
  Reinhard's inverse is genuinely unbounded at 1.0; the missing bound was the
  hazard, since any VAE-decoded specular landing at code 1.0 became 250 million
  nits.

- **OCIO could never reach the bundled ACES config.** `_resolve_config` returned
  whatever `OCIO.GetCurrentConfig()` gave it, and in OCIO v2 that never returns
  None and never raises: with `$OCIO` unset it hands back a built-in default, so
  the shipped `radiance/ACES/config.ocio` and the one the Download ACES 2.0
  button installs were both unreachable. Separately, `RadianceHDROCIOTransform`
  raised at its own default colorspace names, which are absent from that config,
  and its pre-validation read only colorspace names while OCIO resolves aliases,
  so it rejected two of the three names its own tooltip recommends.

- **An animated energy mask was collapsed to frame 0**, silently, because the
  reshape folded the time axis into the batch axis and the batch-mismatch guard
  then expanded frame 0 across the clip.

- **`ensure_4d` flattened video into the batch axis at `debug` level**, which
  ComfyUI's default INFO does not print, so a WAN latent reaching the sampler
  with image-type detection became independent stills with no visible trace.

- **The non-finite guard could not fire on the path that needed it most**:
  `torch.isfinite` raises on a `NestedTensor`, and the exception was swallowed
  into an invisible debug line, so an LTX-AV or MiniMax H3 run with a CFG blowup
  shipped NaNs unchecked.

- **`RadianceVideoExport`, delivery and several nodes reported success for work
  they did not do.** Delivery now returns `status: "partial"` with the list of
  transforms that failed, rather than "EXPORT COMPLETE" over a master delivered
  at 1x because a model was missing, and it validates the format and colourspace
  before grading rather than raising after the whole batch has been processed.

- **`bake_grade` linearised an already-linear master**, applying the sRGB EOTF
  when the colour space was `Linear (sRGB)` and logging it as correct. The other
  half of the same defect made the bake inert for `sRGB (Standard)`, because the
  writer re-encoded it straight back.

- **A video write ignored `overwrite=False` and ran ffmpeg with `-y`**, so
  re-queueing destroyed an approved master, and it never created its output
  directory despite the docstring promising it. ffmpeg's stderr was captured and
  discarded, so every failure surfaced as a bare exit status, and a hardcoded
  600 second timeout killed long encodes and left a truncated file over the one
  `-y` had already replaced.

- **`exr_compression` was threaded through four layers and never used.** Every
  EXR was ZIP, so choosing DWAA for a long dailies sequence silently cost
  roughly five times the size and time.

- **`overwrite=False` renamed sequence frames individually**, so a partial
  re-render produced a mix of `shot_1001_001.exr` and `shot_1005.exr` that was
  neither the old sequence nor the new one and no longer matched the `%04d`
  glob. The collision is now decided once for the sequence.

- **A relative output path resolved against the ComfyUI install directory**,
  which the repo has a containment helper for that the flagship writer was not
  using.

- **`RadianceDigitalCinemaRead` ignored `frame_limit` at its own default.** A
  VFX sequence numbered from 1001 against the shim's `start_frame` default of 1
  produced `end < start`, and the resolver fell through to reading the entire
  shot into RAM: asking for 10 frames of a 100-frame sequence read all 100.

- **`enable_video` with an image model type produced unrelated stills** and
  reported them as a video with a frame count and a duration, skipping stride
  validation, and `batch_size` was ignored for video without the warning the
  same node gives for every other ignored widget.

- **Several controls did nothing.** `pag_scale` was clamped to 1.0 against a
  widget max of 5.0, and PAG logged "applied" at registration even at
  `cfg <= 1.0` where ComfyUI runs no uncond pass and the patch cannot act;
  `noise_alpha_end` had no ramp anywhere in the file despite the tooltip
  promising cosine interpolation; `conditioning_clip_target` wrote a key nothing
  reads and logged a route; `sigmas_override` logged "bypassing internal sigma
  computation" and then forced the terminal sigma to zero, silently fully
  denoising a deliberate leftover-noise pass; `guidance_rescale_phi` was skipped
  at `cfg <= 1.0`, which is the sampler's own default; and `ip_image` and
  `ip_weight` wrote a conditioning key ComfyUI never promotes while the node
  reported `ip_adapter: enabled`.

- **`decode_to_linear_realtime` accepted two denoise parameters and discarded
  them**, skipping the log-space highlight denoise that `hdr/vae.py`'s own
  docstring calls the key to clean HDR.

- **A cross-architecture RUDRA decoder substitution never reached the report**,
  so a decoder for a different VAE could run while the node said "learned
  recovery: applied".

- **`_scene_linear_for_rhdr` was per-instance state on a node ComfyUI reuses**,
  so a run that exited early left it populated for the next one.

- **`detect_vae_factor` returned 0** for a VAE exposing only
  `latent_format.scale_factor`, a value-normalisation constant rather than a
  spatial factor, giving a zero-size buffer and a division by zero.

- **The stale root-level `recovery.py`** was an unimported duplicate of
  `hdr/recovery.py` whose docstring claimed fixes the shipped module does not
  have, so anyone auditing "recovery.py" read the wrong file. Moved to
  `_to_delete/`.

- **PQ encode and decode used the mastering peak as ST.2084's normaliser.**
  See the upgrade note. `color/ops.py` now clips absolute luminance at
  `peak_nits` and normalises by `PQ_MAX_NITS`, a named constant, and the
  behaviour is pinned at five mastering peaks, against an absolute-luminance
  ladder computed from the recommendation's rational constants, and against
  the package's other PQ encoder, which it had been disagreeing with by 10x.

- **`RadianceVAE4KDecode` hung ComfyUI on a shipped default.**
  `TileEngine.compute_tiles` computed `stride = tile_size - overlap` and looped
  `pos += stride` with the only exit being full coverage, so any overlap at or
  above the tile size never advanced and appended tiles until memory ran out,
  with no error and nothing in the log. The temporal call site did not clamp its
  overlap where the spatial ones did, and `temporal_size "2"` with the default
  `temporal_overlap` of 2 was enough to reach it. `compute_tiles` now rejects a
  non-positive stride and the call site clamps with a warning.

- **`color_utils.linear_to_rec709` did not exist.** Written, along with
  `linear_to_rec2020`, their inverses, the torch forms, and the Rec.709 to
  BT.2020 primaries matrix in numpy. See the upgrade note.

- **Applying a LoRA permanently modified the loader's cached model.**
  `ModelPatcher.clone()` shares the underlying module and its parameter
  storage, so the in-place `weight.add_(delta)` wrote through to the cache: the
  delta accumulated on every queue, survived bypassing or deleting the node,
  and only a ComfyUI restart cleared it. On fp8 weights it raised partway
  through and left the shared model half-patched. The node now records patches
  on the clone through ComfyUI's own `add_patches`, which also handles
  quantised weights. Key parsing was rewritten at the same time: it assumed
  `lora_unet_` prefixes and `lora_down`/`lora_up` naming, so a PEFT or
  diffusers LoRA applied zero deltas and logged that as a success with the
  strength widget silently ignored. Zero matched deltas is now an error.

- **`preview_method` did nothing, and selecting it disabled ComfyUI's own
  preview.** It imported `TAESDDecoder` from `comfy.taesd.taesd`, a name that
  exists in no ComfyUI release, so TAESD always fell through to Latent2RGB, and
  neither branch decoded anything. The callback then sent a raw latent where
  ComfyUI expects a format/image/size triple, raising server-side on every step
  that emitted a preview, while `disable_pbar` turned off the working progress
  bar. Both methods now go through ComfyUI's `latent_preview`.

- **`RadianceVideoBatchDecode` flattened 5D latents to 4D before decoding**,
  which forced the 2D image path and threw away ComfyUI's native temporal
  tiling, the mechanism that lets a long clip decode at all. The returned frame
  count was wrong with it: a causal video VAE produces `(T-1)*k+1` pixel frames,
  not `T`. Its two mitigation widgets were inert, probing for `set_tiling` and
  `enable_tiling` methods that ComfyUI has never defined, and `tile_overlap` was
  accepted and never read. 5D latents now reach `vae.decode` intact,
  `tile_decode` routes to `decode_tiled(tile_t=...)`, and the report quotes the
  count actually returned.

- **A VAE decode failure became a 64x64 black frame batch reported as success.**
  `_decode_preview` and `_vae_decode` caught every exception, including
  `OutOfMemoryError`, and returned black; `_comfy_sample` returned the raw input
  noise, which decodes to static, while the pipeline reported a successful
  sample. All three now surface the failure.

- **Twenty-six finished nodes never reached ComfyUI's menu.** `NODE_GROUPS`
  lists only `radiance.nodes.*`, and the folding sweep walks only packages
  already in the catalog, so four implementation packages declaring complete
  mappings were never read: all of `image/upscale.py`, fifteen HDR nodes, three
  film-look nodes, plus three classes missing from their own module's mapping.
  Twenty-five are now registered. The twenty-sixth, `RadianceHDRHistogram`,
  emits a 5D IMAGE that ComfyUI cannot render and is withheld with that reason
  recorded rather than shipped broken.

- **The HLG EOTF carried a second, wrong expression for its linear segment**,
  computed and unused, that anyone reconciling the two would have "fixed" the
  working branch to match.

### Changed

- **CI runs the tests it has.** `test-full`, the only job that installs real
  torch, OpenEXR and OpenColorIO, was gated on `github.repository ==
  'fxtdstudios/radiance'`; development happens on `radiance-beta`, so the lane
  that executes every claim in the README's Verified table had never run. On the
  lightweight lane the suite is 1483 passed and 1577 skipped at 22% coverage;
  with real dependencies it is 3667 passed and 84 skipped at 59.6%. The gate now
  keys on the fork status of the event rather than a repository name.

- **One coverage floor instead of three.** CI passed `--cov-fail-under=15`,
  `pyproject.toml` said 40, and the README claimed 53%. The CLI flag won, so CI
  was gating at 15 against a measured 22. The floor now lives only in
  `pyproject.toml`, is enforced only on the lane that can execute the code, and
  is set just under the measured figure.

- **The README's Verified table states what the suite reports**, not what it
  would have reported had the lane run. Rows that overstated their coverage were
  rewritten and the shortfall moved to Open.

### Tests

- **The ACES 2.0 grey test asserted a table against the interpolator that reads
  it.** At an anchor the interpolation parameter is zero, so it returned the
  table entry by construction: replacing the published values with fabricated
  ones left the suite green. The published table is now transcribed into the
  test as literal data from the ACES Output Transform, the shipped table is
  compared against that transcription, and the interpolator is exercised between
  anchors where interpolation actually happens.

- **Eight of the sixteen colour spaces had no published-value pin**, only
  round-trip and not-identity checks that a wrong but invertible curve passes.
  All fifteen non-identity spaces are now pinned to six decimals at 1e-4 and
  cross-checked against colour-science.

- **The 1-to-100 frame-count claim had no test.** There was one fixed count of
  8, ProRes only, and H.265 10-bit was never encoded. There is now a sweep
  across all three codecs asserting frame identity and order as well as count,
  with the exhaustive version under `-m slow`. The claim was true, it was simply
  untested.

- **The functional test converted real bugs into skips.** Any exception whose
  message contained "requires", "not found", "does not exist", "is required for"
  or "must not be empty" became a skip, which swallows a genuine argument
  contract error as readily as a missing model file. Substring matching on free
  text cannot tell those apart, so it was replaced with a structural classifier:
  an absence has to be signalled by type, and the four nodes that need an
  explicit gate have a predicate that interrogates the environment rather than a
  sentence.

- **The route-registration guard had no assertion**, and the shared test double
  it ran against kept no state, so the ComfyUI startup crash it was written for
  was unreachable. It now registers into a real aiohttp dispatcher and counts
  routes.

- **The import-isolation blocker was inert on Python 3.12**, which is in the CI
  matrix: it used `find_module`, removed in that version, so the test passed
  while blocking nothing. Ported to `find_spec`, and the harness now proves it
  is blocking before reporting anything.

- **A stub leaked across test modules.** `test_node_smoke.py` installed a stub
  `colour` whenever the real package had not yet been imported, which is not the
  same question as whether it is installed, and the stub then shadowed the real
  one for every module collected afterwards. Stubs are now installed only for
  packages that are genuinely absent.

- Removed a constant-folding assertion that could not fail, and an early return
  that made the withheld-node reason check vacuous.

- **RUDRA graded each tile and each frame separately.** A decoder with
  dynamic-range conditioning (`dr_dim`) infers its conditioning vector from
  whatever tensor it is handed, and the tiled and chunked decode paths handed
  it spatial tiles and four-frame chunks. Neighbouring tiles of one frame were
  therefore graded apart, up to the FiLM bound of 5% gain and 0.05 in log code,
  roughly half a stop, and a clip's exposure breathed frame to frame. The
  vector is now resolved once from the whole latent, by
  `fast_vae.rudra_condition_for()`, and passed down to every tile and chunk in
  `fast_vae.decode_to_linear_realtime` and in both `hdr/vae.py` decode paths.
  A clip shares one vector across its frames; a batch of independent stills
  keeps one per image, which is the correct behaviour there. Decoders without
  conditioning are called exactly as before, with a single argument, so any
  `nn.Module` still works as a `turbo_decoder`. Measured on a split-exposure
  latent: the tile conditioning vectors differed by 8.3 in norm, and tiled
  decode carried 29 times the error of the same test with an unconditioned
  decoder. Pinned by 15 tests.

- **The LTX stills warning fired on every LTX checkpoint, retrained or not.**
  It now reads the frame count the training run stamped into the safetensors
  metadata and warns only when the checkpoint declares one frame or declares
  nothing.

- **The OCIO CPU transform was an exact identity.** `apply_ocio_transform()`
  called `applyRGB()` on a Python list, which OCIO transforms into a temporary
  and discards, then wrote the untouched input back. The same defect the LUT
  bake carried. It now applies to a contiguous float32 array. The function has
  no callers yet, so no shipped output was affected.

- **Denoise did nothing on delivery.** `delivery/handler.py` passed a `uint16`
  array to OpenCV's bilateral filter, which accepts 8U and 32F only, and the
  resulting exception was swallowed at DEBUG level. A colourist could set
  Denoise, get HTTP 200 and "EXPORT COMPLETE", and receive a master with no
  denoise and no warning. It now filters in float32, and a failure logs at
  WARNING.

- **Version backups evicted the newest instead of the oldest.**
  `_create_version_backup` sorted filenames lexicographically, so past ten
  versions `v10` sorted before `v2` and capacity eviction destroyed a newer
  backup while keeping an older one. Sorting is now by version number.

- **Every saved v3 workflow advertised a preview it did not have.** The pack
  tested `"preview_image" in metadata` rather than its value, and `save` always
  sets the key, so an empty string wrote a zero-byte `preview.png`,
  `/workflows/list` reported `has_preview: True`, and `/workflows/preview`
  returned 200 with no bytes.

- **Legacy v1 workflows under 14 bytes lost their metadata.** The record reader
  required the v2 binary header length from every container, but a v1 `.rad` is
  plain-text JSON of any length, so a short one fell into the error path and
  its sidecar was discarded.

- **A completed download could be reported as a failure.** The ComfyUI listing
  refresh sat inside the same `try` as the download, after the verified file
  had already been moved into place, so a refresh that kept failing sent a
  checksum-clean model down the failure path and the loader reported it
  missing. The refresh now warns and the download reports success.

- **The loader named CLIP encoders it never read.** `clip_slot_used` tested
  `val is not None`, which counts ComfyUI's empty-combo sentinel `"None"` as a
  loaded encoder. It now walks the architecture's own slot order and applies
  the same emptiness test the path assembly uses.

- **A refused delivery left the progress bar spinning.** Three validation
  refusals returned before any progress update, and the client clears its poll
  interval only on `done` or `error`.

### Changed

- **The write engine moved out of the node layer.** `radiance/io/writer.py`
  now holds the format tables, the output colour-space application, every save
  backend and `write_frames` itself; `RadianceWrite.write` is a signature and a
  delegation, and `delivery/handler.py` calls the engine directly instead of
  instantiating the node to save a file. `nodes/io/write.py` drops from 2972 to
  2201 lines and re-imports the moved helpers under their old private names, so
  the reader, `RadianceEXRMultiPart` and `RadianceDigitalCinemaWrite` are
  untouched. The node's parameter list is unchanged, so saved workflows keep
  loading.

  Coercing a file path or a VideoHelperSuite dict to frames still needs the
  decoders, which belong to the reader — the engine takes a `read_media`
  callback for that rather than importing upward, and the delivery path does
  not supply one because it hands the writer a tensor.

  Verified byte-for-byte across 42 cases: every write format, every output
  colour space, and each of the named behaviours. The only differences are the
  DPX header's creation timestamp, which differs between two runs of identical
  code.

### Added

- **The delivery path is tested without ComfyUI**, which is what moving the
  write engine down a floor was for. Twenty-nine tests over
  `delivery/handler.py`: the two lookup tables are checked against the writer's
  own format and colour-space tables rather than a list retyped in the test —
  that mismatch is exactly what shipped once, as a TypeError the handler
  swallowed into an HTTP 200 — plus version numbering, the atomic session log,
  the ACES sidecar's CDL values, and a master written end to end from the same
  call the endpoint makes. 10% to 22%.

- **Optical flow is DIS now**, which roughly doubles the motion it can follow
  and runs about eleven times faster. Measured on the existing test plate, the
  fraction of the field landing within half a pixel: Lucas–Kanade 100 / 84 / 71
  / 58 / 29 percent at 3 / 8 / 12 / 16 / 20 px, DIS 100 / 100 / 100 / 100 / 99.
  On an aperiodic plate, closer to grain than to sinusoids, Lucas–Kanade is
  down to 9% by 12 px where DIS is still at 100%. Lucas–Kanade remains
  selectable on the Optical Flow node and is the automatic fallback if OpenCV
  cannot be imported. `RadianceOpticalFlow`'s docstring has claimed DIS for
  some time while the node ran Lucas–Kanade; that is now true rather than
  aspirational.

- **First tests for `image/upscale.py`**, which was 1111 statements at 0% and
  is reached from `delivery/handler.py`. Nineteen of them, over bit depth
  conversion, resampling and sharpen — the parts that requantise and resample,
  which is where an HDR range gets quietly destroyed. The float depths must not
  clamp; the integer depths must clamp and land on their quantisation steps;
  dithering must be reproducible at a fixed seed; an over-range plate must
  still be over-range after a downscale; and the sRGB decode must only run when
  the input is called sRGB. 28% covered now, the remainder being the paths that
  need model weights.

- **The browser test lane covers every Viewer panel, not five of them.** The
  harness listed the panels it drove, so it exercised the framing, scope,
  probe, OCIO and view sections while nine others — primaries, effects, lens,
  curves, qualifiers, masks, prompt, terminal and the HDR preview widget — were
  held by source assertions. The list is now discovered from the class, so a
  panel added later joins the check by existing, and the test asserts a floor
  on the count so it cannot quietly shrink back. Building all fourteen found no
  defect; driving the mask type selector found one worth pinning, and the new
  assertion fails if the `parseInt` on it is ever dropped.

- **The read engine moved below the node layer**, to `radiance/io/reader.py`.
  The decoders, the colour-space decode, path-kind detection, the image,
  sequence and video readers, the write-input coercion helpers and the UI probe
  all left `nodes/io/write.py`, which is now 1262 lines — 2972 before the
  writer moved, 2201 after it. `RadianceRead.read` is a signature and a
  delegation; the browse widget stays in the node layer because resolving a
  filename in ComfyUI's input directory needs `folder_paths`. Every private
  name is re-exported, so `RadianceEXRMultiPart`, the digital-cinema nodes, the
  HTTP routes and the two `nodes/pipeline` modules that reach in for
  `_read_sequence` and `_load_video_to_numpy` are untouched. No node key, no
  widget name and no widget order changed, so saved workflows are unaffected.
  Verified byte-for-byte across 215 cases — 18 fixtures covering every image,
  EXR, sequence and video kind, crossed with every input colour space and every
  read option — with no hash moving.

- **The emitted WGSL is compiled and compared against the JS it comes from.**
  `radiance_grade.js` emits GLSL and WGSL from the same file as its JS
  functions; only the GLSL half was ever compiled, because headless Chromium
  has no `navigator.gpu` at all -- absent rather than blocked, under every flag
  combination tried, while WebGL2 works in the same browser. Deno ships WebGPU
  and Mesa's lavapipe provides a software Vulkan device, so the WGSL now runs
  the same 60-case matrix as the GLSL, on a runner with no GPU. Worst deviation
  3.1e-7 against a 2e-6 tolerance. Reverting the WGSL to any of the three ways
  the WebGPU path used to disagree with WebGL -- flat lift, unguarded gamma,
  power-form contrast -- turns the suite red.
- **`tests/test_writer_layering.py`.** Six tests that make the write engine's
  independence checkable rather than asserted in a docstring: the engine's AST
  carries no node-layer import at any depth, it imports and writes a file in a
  bare interpreter with no ComfyUI, the handler neither imports nor instantiates
  the node, the node's signature still matches the engine's, and the handler's
  keyword arguments are all real parameters — which is the delivery bug that
  shipped, as a test.
- **The viewer panels are built and operated in a browser.** The panel code was
  held by source assertions, which cannot tell you a panel builds; the week the
  Viewer node rendered with no UI, every text-level check passed. The panels
  now load against stubbed ComfyUI modules, get built, and get driven --
  selects changed, toggles clicked -- with the instance state, the persisted
  setting and the control label all checked afterwards.

### Fixed

- **Every OCIO LUT baked in Python was an exact identity.** `bake_colorspace_lut`
  and `bake_display_view_lut` looped over the cube calling
  `applyRGB(python_list)`. OCIO's binding converts a list to a temporary
  buffer, transforms that and discards it, so the list is never modified and
  each lattice point was written back unchanged — the result was bit-identical
  to its own input. 18% grey through ACEScg → sRGB Display came out as 0.18
  instead of 0.47. That LUT is served over `POST /radiance/ocio/bake` and
  loaded into the Viewer's renderer, so choosing a display transform from the
  OCIO dropdown logged "[OCIO] Active" and changed nothing on screen. The bake
  now applies to the array, which OCIO does modify in place, and matches its
  own CPU processor exactly. It is also about 35x faster, and the
  `hasattr(applyRGB)` branch guarding an unreachable "batch API if available"
  fallback is gone with it. Found by writing the first tests this module has
  ever had; `tests/test_ocio_manager.py` fails if the identity returns.
- **OCIO LUT cache keys could collide.** The key was a colon-joined string of
  unescaped names, so a display or view containing a colon could hash to the
  same key as a different transform and be served the wrong LUT.

- **The writer-layering tests now run on CI instead of failing and skipping.**
  Both bare-interpreter tests assumed the repository directory is named
  `radiance` -- true of a working copy, false on GitHub, which checks out as
  `radiance-beta/radiance-beta`. One failed with `ModuleNotFoundError: No
  module named 'radiance'`; the other swallowed the same error as "a
  dependency is missing" and reported a skip, so the claim that the write
  engine imports without ComfyUI was unverified on CI in both directions. They
  now load the package from its `__init__` path under an explicit module name,
  and the skip guard is limited to named third-party packages.

- **The safe-area caption no longer describes the previous preset.** Switching
  to Legacy 480-line redrew the boxes at 90/80 correctly but left the note
  beneath reading "SMPTE ST 2046-1 and EBU R 95 specify the same two boxes".
  The guide was right and the standard named under it was wrong, which in a QC
  overlay is the worse half. Found by driving the panel, not by reading it.

## [3.4.0] - 2026-08-17

A Viewer release. Everything below is in the Viewer unless it says otherwise.

**Upgrade note.** Two defaults change.

**WebGPU is now opt-in.** It used to upgrade automatically wherever the browser
exposed `navigator.gpu`, so nobody chose it — and the backend it switched people
to is the one that does not implement masks, qualifiers, the HDR heatmap or
OpenColorIO, and whose grade maths differed from WebGL on lift, gamma and
contrast. Defaulting to the backend with the missing features, then explaining
the gaps with in-panel banners, is a worse product than defaulting to the one
that works. View → Framing & Guides → Backend turns it back on, and states what
stops working when you do.

**Safe-area boxes move.** The Viewer drew the modern 93% action box against the
legacy 80% title box — a pairing no published standard specifies. The title box
is now 90%. If you have been placing graphics against the old inner box they
were inset by 10% for a delivery that asks for 5%.

### Added

- **OpenColorIO 2.5 config support.** Load a show's config and the Display and
  View menus come from it; picking a view changes the picture, through OCIO's
  own generated GLSL. Bundled ACES 1.3 and 2.0 configs for anyone without a
  config of their own. With nothing loaded the built-in ACES 1.3 pipeline runs
  exactly as before — OCIO is a capability, not a dependency. Real OpenColorIO
  compiled to WebAssembly, vendored under `js/vendor/ocio/` (BSD-3-Clause).
- **Pixel probe** (PROBE tab). Cursor, region and full-frame sampling; source or
  rendered values; RGBA, luminance, EV, cd/m², HSV and hex; min, max, mean and
  median per channel. NaN, Inf and negative counts are excluded from every
  statistic and reported separately.
- **Scope scales.** 10-bit and 12-bit code value, percent, millivolts, and nits
  for ST.2084 and HLG, with Data/Video levels and a switch for whether the
  scopes measure before or after the viewer colour transforms. Every graticule
  line carries its number; the panel states what is being measured. IRE is
  deliberately not offered.
- **HDR heatmap**, banded to ITU-R BT.2408 with a white line at 200–206 nits.
- **Safe areas labelled with their standard**, an **aspect-ratio matte**,
  **nearest-neighbour magnification on `N`**, and **frames / seconds / timecode**.

### Fixed

- The **HDR heatmap did not exist**: "HDR Heatmap" and "False Color" were one
  toggle under two menu labels, and the map read display luma on ARRI stops.
- **The grade was implemented four times** — WebGL GLSL, WebGPU WGSL, the WebGPU
  CPU readback and the `.cube` export — and no two agreed on lift, gamma or
  contrast. The CPU readback used a different contrast curve entirely and
  produced NaN at pivot 0. All four are now emitted from `js/radiance_grade.js`.
- The scopes' **"HDR ruler" placed its nit lines with a Reinhard curve** as a
  stand-in for the display transform, while the pixels had come through the
  actual ACES transform: "203 nit" drew at half height regardless.
- **Safe areas were drawn on the canvas, not the picture** — at any zoom or pan
  other than an exact fit they measured the viewport.
- Scene-linear black rendered as **NaN or ~1e16** through OCIO views
  (`pow(0, y≤0)` is undefined in GLSL), and every HDR view rendered **solid
  black** (`OES_texture_float_linear` must be enabled, not merely supported).
- `load_trained_turbo_decoder()` raised **NameError on every call** — the
  decoder classes were imported in `train()` and nowhere else.

### Testing

240 JavaScript tests, including two headless-WebGL2 harnesses that compile the
real shaders and compare them against the CPU implementations they were
generated from. Both defects in the OCIO list above were found by rendering, not
by reading — each compiled cleanly and reported nothing.

## [3.3.0] - 2026-08-16

**Upgrade note — read before updating mid-project.** Five changes alter what an
unchanged graph does. None is a preference; each corrects something that was
measurably wrong. But if you are partway through a job, finish it on 3.2.1.

1. **ACES 2.0 output moves.** 18% grey now lands where the standard puts it —
   10.000 nits at a 100-nit peak, 14.512 at 1000 — instead of ~18 nits
   everywhere (legacy transform) or 10% of peak (Daniele Evo node, which put
   grey at 400 nits on a 4000-nit master). In signal terms: sRGB 0.3492 rather
   than 0.4610, PQ 0.3298 rather than 0.3478. **Re-check any master graded
   against the old midtone.** HLG is unchanged — it is anchored to BT.2408.
2. **Scene-cut `threshold` is now `cut_confidence`**, on a calibrated 0–1 scale
   that means the same thing for `histogram`, `edge` and `combined`: 0.5 is "as
   different as a hard cut", lower is more sensitive. It was a fraction of the
   clip's own maximum, then briefly an absolute inter-frame distance whose
   useful value differed by method. The widget was renamed each time
   deliberately — a saved value would otherwise have been silently
   reinterpreted. Existing workflows reset to the new default and need
   re-tuning. The `edge` metric itself changed with it; see Fixed.
3. **Model downloads now refuse by default.** Set `RADIANCE_ALLOW_DOWNLOADS=1`
   to restore automatic fetching. Previously a first queue could pull 67 MB to
   2.4 GB with no prompt.
4. **Tier-3 upscale at 2x returns a different image** — the whole frame rather
   than the top-left quarter of a 4x render.
5. **CDL Export and Flipbook GIF write to `output/`** instead of resolving
   their relative defaults against the ComfyUI install directory.
6. **The node menu gains twenty entries.** Twenty finished nodes had never been
   published; nothing you already had moves or changes. See Added.
7. **Chained Energy Mask nodes now stack.** Two of them in series, or two
   branches joined by Conditioning Combine, previously threw one mask away
   silently. They add now. A graph with two Energy Masks will sample
   differently — and, for the first time, the way it looks like it should.


### Added

- **Twenty nodes published that had never reached the menu.** Every node group's
  `__init__.py` hand-copied a selection of its modules' node keys into the
  mapping ComfyUI reads, and nothing compared the two. Whatever nobody
  remembered to copy did not exist as far as ComfyUI was concerned — not
  broken, not disabled, simply invisible. Seventeen complete nodes were in that
  state, plus three compatibility alias keys:

  `RadianceHDRTurboEncoder`, `RadianceHDRPerChannelNorm`,
  `RadianceHDRPerChannelDenorm`, `RadianceACESMetadataFile`,
  `RadianceACES2Compliance`, `RadianceColorSpaceInfo`,
  `RadianceLuminanceGuidance`, `RadianceHDRBlendValidator`,
  `RadianceHDRAnalysis`, `RadianceNDISender`, `RadianceShotGradeRouter`,
  `RadianceAudioCut`, `RadianceAudioTranscribe`, `RadianceLinearCheck`,
  `RadianceCinemaStudio`, `RadianceCameraSync`, `RadianceVideoPromptBuilder`.

  This is the third time the same defect has surfaced — issue #40 was a sampler
  feature no node could reach, and `RadianceGradeApply` was a finished node
  whose only trace was a menu-section override. `nodes/aggregate.py` inverts
  the default: a group now sweeps its own modules, so writing the node is
  enough to ship it, and withholding one takes a named `WITHHELD_NODES` entry
  that a test reads back. Catalog 111 → 131.

- **Compatibility aliases load without cluttering the menu.**
  `RadianceImageLoader`, `RadianceControlApply` and `RadianceWorkspace` are
  older keys for nodes that still exist. They ship as `DEPRECATED` subclasses:
  a workflow saved against the old key opens normally, and node search shows
  one entry per node rather than two. (Subclasses because `DEPRECATED` is read
  off the class — pointing both keys at the same class object would have hidden
  the canonical node too.)

- **`RadianceGradeApply` reaches the menu**, published as **Bake Viewer Grade**.
  It bakes the Viewer's grading maths into the tensor — a complete node with 16
  documented inputs and a valid contract, listed in `viewer.py`'s own mappings
  since v3, but never transcribed into `nodes/monitor/__init__.py`. It was
  found because `nodes/branding.py` carried a menu-section override for it:
  someone had classified a node nobody could place. Catalog 110 → 111.

  Its label also settles the `Grade` / `Grade Apply` / `Apply Grade Info`
  overlap — the menu now reads Grade, Grade Match, Apply Grade Info and Bake
  Viewer Grade.

- **Menu sections are declared, not guessed.** `NODE_SECTIONS` in
  `nodes/branding.py` names the section for all 111 registered nodes.
  `classify_menu_section`'s keyword rules ran in order and returned on the
  first match, so `"mask"` was tested for VFX several rules before
  `"conditioning"` was tested for Generate — which is how `RadianceEnergyMask`,
  a node that feeds the sampler, landed in the VFX menu. The rules remain only
  as a fallback for undeclared nodes, and they now log a warning when they run.
  `tests/test_menu_sections.py` fails if a registered node is missing from the
  table, so the next one is caught at commit time rather than by a user
  hunting for their node.


- **Energy Mask node (`RadianceEnergyMask`).** Energy-Prioritized Sampling has
  been in the sampler since v3.1, but nothing ever wrote the
  `radiance_energy_mask` key it reads, so no graph could reach it (#40). This
  node writes it: connect CONDITIONING and a MASK, set `priority`, and the
  masked region samples at an effective `cfg x (1 + priority)` while the rest
  of the frame keeps the sampler's `cfg`. Negative priority softens a region
  instead. Menu: Generate.

### Fixed

- **EPS crashed on video latents.** The reader unpacked `B, C, H, W` from the
  denoised prediction, which raises `ValueError` on the 5-D `(B, C, T, H, W)`
  latents WAN, Hunyuan and LTX produce. The mask is now aligned rank-agnostically
  and broadcasts across frames. Nobody had hit this because the feature was
  unreachable.
- **EPS crashed on a mask batch that was neither 1 nor the latent batch.**
  `mask.expand(B, ...)` cannot broadcast 3 -> 4; it now falls back to the first
  mask rather than raising mid-sample.
- The resized mask is cached per latent geometry instead of being
  re-interpolated on every sampling step.
- `RadianceEnergyMask` is pinned to the Generate menu section; the keyword
  classifier would otherwise file anything named "...Mask" under VFX, away from
  the sampler it feeds.
- **Chaining two Energy Mask nodes lost one of the masks, silently.** In series,
  the second `attach_energy_mask` overwrote the key on every conditioning entry,
  so the downstream node won; joined by Conditioning Combine, the sampler
  stopped at the first entry carrying the key, so the upstream node won. The
  node's own docstring described only the second case, and got it backwards for
  the first. Layers accumulate now and the sampler sums `priority x mask` across
  all of them, with the combined modifier floored at 0 so a stack of negative
  priorities suppresses guidance instead of inverting it.

- **The scene-cut `edge` metric had no meaningful threshold at all.** It was a
  mean absolute difference of gradient magnitudes, which scales with the
  footage's own contrast and texture rather than with how different two frames
  are. Measured on one synthetic cut, re-graded:

  | contrast | absolute (old) | relative (new) |
  | --- | --- | --- |
  | 100% | 0.0369 | 0.2950 |
  | 50%  | 0.0185 | 0.2950 |
  | 25%  | 0.0092 | 0.2950 |

  The same cut, four times fainter. A threshold tuned on a bright exterior
  found nothing in a dim interior, and grain on a detailed frame (0.039)
  outscored a real cut in soft content (0.003). It is a relative (Bray-Curtis)
  distance between edge maps now, bounded in [0, 1], with a 5 px pre-blur so a
  high-pass metric stops measuring film grain. `combined` was the worst
  casualty — `0.6 * histogram + 0.4 * edge` added two quantities with no shared
  unit, so the nominal 40% edge contribution was a couple of percent; both
  terms are calibrated confidences now and the weights mean what they say.
  `KNOWN_ISSUES.md` records what the edge method still cannot see.

### Fixed

- **Nodes wrote into the ComfyUI install directory.** CDL Export defaulted its
  `file_path` to `grading/shot_01_output.cdl` and Flipbook GIF defaulted
  `save_path` to `preview/flipbook.gif` — bare relative paths resolved with
  `os.path.abspath()`, which anchors to the process working directory. For
  ComfyUI that is the install root, so running either node on its default
  created `grading/` and `preview/` folders inside the ComfyUI installation
  instead of writing to `output/`. Both now route through the new
  `resolve_output_path()`; relative paths land under `output/`, absolute paths
  are honoured as typed, and `..` traversal is rejected. CDL Import gained the
  matching `resolve_input_path()`, which searches `input/` then `output/` so a
  CDL you just exported is found by the stock widget value.
- **`INPUT_TYPES()` created a directory.** `RadianceLUTApply.get_lut_files()`
  called `os.makedirs(models/luts)` from inside the widget enumerator, which
  ComfyUI invokes for every node at startup and on every `/object_info`
  request. It now reports "No LUTs found" and leaves the filesystem alone.
- **`models_dir` was checked for truthiness, not type.** Anywhere
  `folder_paths.models_dir` is joined, a non-string value now falls back to
  `~/.cache/radiance` instead of being coerced into a path. Affects
  `nodes/upscale`, `nodes/vfx/multipass`, `temporal_rudra`, `radiance_ocio`
  and `color/lut`.
- **`RADIANCE_LOG_LEVEL` did nothing.** The startup shortfall error has told
  users to "re-run with RADIANCE_LOG_LEVEL=DEBUG for tracebacks" since 3.2.0,
  but nothing read the variable — `setup_radiance_logging()` was always called
  at INFO. It now resolves the level from the environment (level name or
  number, falling back rather than raising on garbage), so the advice printed
  by the error message is true.

### Fixed

- **Both ACES 2.0 tone scales now hit the published reference.** Radiance
  shipped two of them and they disagreed by up to 24x.
  `RadianceACES2Tonescale` used the real Daniele Evo curve but pinned 18% grey
  to a flat 10% *of peak*, so grey tracked the display: 10 nits at SDR, 100 at
  1000, 400 at 4000. `RadianceACES2OutputTransform` used a log-contrast + tanh
  approximation that held grey at ~18 nits on every peak — right in kind,
  ~0.85 stop bright at SDR.

  There was no judgement call: ACES 2.0 publishes the mapping. An ACES value of
  0.18 lands at 10.000 / 13.193 / 14.512 / 15.747 / 16.824 nits at peaks of
  100 / 500 / 1000 / 2000 / 4000 — the midtone barely moves while the
  highlights extend. `hdr/tonescale.py` carries that table; the Daniele Evo
  parameterisation derives its grey target from it, and the legacy transform
  solves an input gain that puts grey on the same value while keeping its
  highlight roll-off.

  **Upgrade note.** SDR and PQ output changes: 18% grey encodes to sRGB 0.3492
  instead of 0.4610, and to PQ 0.3298 instead of 0.3478. Both new values are
  independently derivable — 0.3492 is the sRGB encode of 10 nits, 0.3298 is the
  PQ signal for 14.512 nits. Re-check any master graded against the old
  midtone. HLG is deliberately unchanged: it is anchored to BT.2408 reference
  grey (signal 0.38, ~26 nits) and passes a synthetic peak into the curve, so
  the ACES table does not apply to it.

- **Scene-cut thresholds were relative to the batch.** `detect_cuts` divided
  every clip's scores by that clip's own maximum before comparing to
  `threshold`, so the largest inter-frame delta in *any* footage was forced to
  1.0. The threshold widget meant something different on every clip, and
  cut-free footage always reported a cut. The comparison is now against the raw
  distance. `_make_plot` was already comparing raw scores, so the rendered plot
  and the returned cut list could contradict each other on the same input; they
  now agree. Note the scales differ by method — histogram runs 0–2, edge runs
  roughly 0–0.5 — which the normalisation had been hiding.
- **Tier-3 upscale ignored `scale`.** The diffusion backends are fixed 4x, and
  nothing conformed their output, so `tiled_upscale` cropped a 4x render down
  to the requested 2x size — the user got the top-left quarter of the frame
  (measured correlation with the input: 0.0024). The output is now resampled to
  the requested ratio.
- **Optical flow was single-scale Lucas–Kanade.** Brightness constancy was
  linearised as `It = f2 - f1`, valid only within a pixel or two, so mask
  propagation was effectively static on real motion: ~100% recovery at 1 px,
  17% at 3 px, 1% at 5 px. It is now coarse-to-fine over an image pyramid,
  warping by the accumulated flow and solving for the residual at each level.
  1–5 px displacements now recover to within 10%, with 84–100% of the field
  inside half a pixel. Above ~8 px is still unreliable.
- **Model weights downloaded without asking.** Real-ESRGAN, HAT-L, SwinIR,
  Depth Anything V2 and DSINE — 67 MB to 2.4 GB — began fetching the moment a
  graph was queued. `nodes/upscale` had an opt-*out*
  (`RADIANCE_UPSCALE_OFFLINE=1`); `nodes/vfx/multipass` had no gate at all. Both
  now go through `radiance.core.consent`, which defaults to refusing with a
  message naming the file, its size and where to put it. Set
  `RADIANCE_ALLOW_DOWNLOADS=1` to permit them; the legacy opt-out is still
  honoured.

### Removed

- **Six internal write-ups.** `AUDIT.md`, `FULL_AUDIT_2026-08.md`,
  `BATCH3_REPORT.md`, `RELEASE_REVIEW.md`, `VIEWER_REPORT.md` and
  `PACKAGE_REVIEW.md` were point-in-time reports that had already drifted from
  the code — several listed as open bugs that had been fixed months earlier,
  which is how the tiled-VAE blend stayed on the backlog after it was
  corrected. What matters lives in the README's Status & to-do list,
  `KNOWN_ISSUES.md` and this changelog; `.gitignore` now keeps their filenames
  out.
- **`gpu_acceptance_report.md` is no longer tracked.** `tools/gpu_acceptance.py`
  writes it on every run — a generated artifact that should never have been
  committed. Now ignored.

- **The legacy `nodes_*.py` layer.** Forty root modules were pure re-export
  shims over `nodes/`; they are deleted. Six were still the real home of their
  code and moved into the package, which also reverses a dependency that ran
  backwards — the organized `nodes/` package had been importing *from* the
  layer it was introduced to replace:

  | was | is |
  | --- | --- |
  | `nodes_io.py` | `nodes/io/write.py` |
  | `nodes_sampler.py` | `nodes/generate/sampler.py` |
  | `nodes_loader.py` | `nodes/generate/loader.py` |
  | `nodes_workspace.py` | `nodes/pipeline/workspace.py` |
  | `nodes_realtime_preview.py` | `nodes/monitor/realtime.py` |
  | `nodes_gizmo.py` | `nodes/gizmo.py` |

  `radiance.nodes_*` imports no longer resolve. Node keys are untouched, so no
  saved workflow is affected — every one of the 111 pre-existing keys was
  compared class-by-class and widget-by-widget against a snapshot taken before
  the move. The package also drops to a single entry point: the separate
  `.nodes_radiance_viewer` spec was publishing `RadianceViewer` alongside
  `radiance.nodes.monitor`, the one genuine double-import the layer had.
- **Dead `.comfyignore` entries.** Thirty-two of its paths no longer existed. A
  packaging exclusion list naming files that are already gone reads as
  protection it is not providing, so the file was rewritten around what is
  actually there plus deliberate glob guards.

- **The `docs/` folder and its tooling.** Documentation now lives at
  [www.fxtdstudios.com](https://www.fxtdstudios.com) rather than in the
  repository. Removed with it: `tests/test_docs_coverage.py`,
  `tools/generate_docs.py`, `scripts/build_website_docs.py`, and the
  `docs/dev/` handling in `tools/clean_release.py`.

  Note what this gives up — `generate_docs.py` built the node reference from
  the live `NODE_CLASS_MAPPINGS` and a test failed when the two drifted. That
  guarantee is gone, so the published node reference is now maintained by hand.
  Node `DESCRIPTION` strings and per-input tooltips still ship inside the
  package and are shown by ComfyUI on hover, so the in-app reference stays
  accurate on its own.

### Changed

- **All links point at [www.fxtdstudios.com](https://www.fxtdstudios.com).**
  Standardizes the website address to `www.fxtdstudios.com` in the README, the
  `Documentation` URL in `pyproject.toml`, and the in-product links in the
  Workspace node, the project-manager launcher, the Viewer and the workspace
  dashboard.

### Housekeeping

- `.gitignore` now covers `grading/`, `preview/` and `MagicMock/` — all three
  were untracked but unignored, one `git add -A` away from being committed.

### Tests

- `tests/test_energy_prioritized_sampling.py` replaces the two EPS tests in
  `test_sampler_regression.py`. Those re-implemented the parser and the modifier
  arithmetic inside the test body and asserted on their own copies, so they
  passed without executing any of `nodes_sampler.py` — including through both
  crashes above. The new tests drive the real node and the real CFG patch.
- `tests/test_output_path_anchoring.py` pins the path fixes above, and adds
  two ratchets: no node may default a writable path to a bare relative
  string without routing it through a resolver, and `INPUT_TYPES()` may not
  touch the filesystem.
- `tests/test_log_level.py` covers the environment variable above, including
  a check that the string in the startup error still names a real flag.
- The README gains a **Status & to-do** section: what is blocking a release,
  the correctness backlog, structural debt, and what is verified done.
- `tests/test_backlog_fixes.py`, `tests/test_download_consent.py` and
  `tests/test_optical_flow.py` cover the fixes above. The tiled-VAE blend is
  pinned too — it had already been fixed months earlier but never tested, which
  is why the audit notes still listed it as open.
- **First JavaScript coverage in the project.** 27 tests over the shared
  `escapeHtml` and widget-visibility helpers, on Node's built-in
  `node --test` — no test framework added — wired into CI as a `js-test` job
  and available as `npm test`.

## [3.2.1] - 2026-08-09 ("Full Audit")

**Upgrade note.** Colour output changes for any graph that used
`RadianceColorSpaceConvert` with a camera-log or ACES space — those
conversions previously did nothing (see below). Re-check affected masters.
16/32-bit TIFF writes now require `tifffile` (`pip install tifffile`) instead
of silently writing 8-bit.

### Added

- **Inline video preview on Read.** MP4/MOV/WebM the browser can decode play
  directly on the node (`/radiance/media/preview`, with seeking); ProRes,
  DNxHR and other production codecs fall back to a first-frame poster
  (`/radiance/media/poster`). Same allowed-roots policy as the info routes.
- **Resolved-path readout on Write.** The node now shows exactly what will
  land on disk — output_path + filename + version + format combined by the
  same Python logic that writes, via `/radiance/media/resolve_write` — plus
  a note line ("frame 1 only" for IMG formats with a batch, "DNxHR is written
  into an MXF container", sequence start frame). A contract test writes real
  files and fails if prediction and reality drift.

### HDR family

- **SDR to HDR Expand: `smoothness` did nothing.** The feathering mask was
  computed and then never applied — a dead control since the node shipped.
  It now feathers the expansion onset as its tooltip promises.
- **SDR to HDR Prepare: the feathered inpainting mask drifted up-left** by
  ~2× the feather radius (32 px at the default 16) with dark bands on the
  right/bottom — the blur loop re-padded asymmetrically after passes 2–3.
  The AI was being guided to repair pixels offset from the actual clipped
  highlights. Symmetric per-pass padding; centroid pinned by test.
- **Fast-VAE tiled decode: tiles were butt-joined.** Each tile decoded with
  overlap context but pasted hard against its neighbour; VAE decoders are
  not shift-invariant at their borders, so seams showed on flat gradients.
  Tiles now accumulate under a raised-cosine weight and normalise — a test
  proves tiled output equals untiled output exactly.

### Viewer

- **VRAM frame cache now evicts by bytes, not just frame count.** The LRU
  held 4–24 frames regardless of size — 400 MB at 1080p but ~7 GB at 8K,
  an out-of-memory long before the count limit was reached. Eviction now
  also respects a byte budget (0.5–3 GB, scaled to machine memory).
- **Frames beyond the GPU's texture limit fail loudly, before upload.**
  8K DCI (8192 px) sits exactly on many GPUs' `MAX_TEXTURE_SIZE`; anything
  over it used to produce a black frame and a cryptic GL error code. The
  viewer now reports the frame size, the GPU's limit, and that `proxy_scale`
  is the way to view it — full-resolution data is unaffected.
- Removed stale `MAX_BATCH_SIZE`/`MAX_IMAGE_DIMENSION` duplicates from
  `hdr/io.py`; `viewer_utils.py` (9999 frames / 16384 px) is the single
  source.

### Changed

- **`overwrite` on Write now defaults to OFF.** Destroying an existing file
  must be an explicit choice; when off, a unique suffix is appended instead.
- Read greys out `color_space` when `raw (no transform)` is selected — the
  reader ignores it, and the UI now says so instead of looking live.
- Write hides `broadcast_safe` for float formats (EXR/HDR/32f TIFF/DPX),
  matching the Python side which already refuses to legal-range-clamp them;
  `version` displays as `v0001` alongside the number.

### Fixed

- **Colour Space Convert did real conversions for only 6 of its 16 spaces.**
  Without an OCIO config — the default install — ACEScc, ACEScct, LogC4,
  F-Log2, C-Log3, Log3G10, DaVinci Intermediate, BMD Film Gen5, V-Log and
  N-Log silently returned the input unchanged. The node's own inline LogC3
  was also wrong: 18% grey encoded to 0.417 instead of ARRI's 0.391, and its
  decode disagreed with its own encode, so a LogC3 round trip lost ~1.5 stops.
  All curves now come from `color/transfer.py` / `color/luts.py` (verified
  against published 18%-grey code values, round-trip exact), Rec.709 OETF is
  the real BT.709 camera curve rather than an alias of sRGB, ACEScc/ACEScct
  apply the Rec.709↔AP1 gamut matrix, and a space with no analytical path
  raises instead of passing pixels through untouched.
- **16-bit and 32-bit float TIFF writes refuse to downgrade.** `tifffile` is
  the only writer for these formats but was never declared as a dependency;
  without it a 32-bit float request silently produced an 8-bit clipped file
  (the 16-bit path at least logged a warning). Both now raise with an install
  hint, the reader warns when it cannot probe TIFF depth, and `tifffile` is
  listed by the Environment Guard.
- **Directory and glob sequence reads always came back empty.** The file list
  was sliced by list index with the widget's frame-number defaults
  (`files[1001:99999]`). Frame numbers are now parsed from filenames and the
  window is reconciled against the range on disk, matching the `####` path.
- **The test suite could not see colour-node bugs.** The conftest OCIO mock
  reported `is_loaded` as a truthy `MagicMock`, so nodes "converted" via a
  no-op mock processor and every colour test through them validated an
  identity transform. The mock now reports `is_loaded = False` and a
  regression test pins it.

## [3.2.0] - 2026-07-29 ("Audit Release")

A three-week audit, five independent review passes, and a new test harness that
calls every node. The theme throughout: almost nothing here raised an error. The
code ran, reported success, and produced the wrong result — which is why the
existing 1,500-test suite had nothing to catch.

**Upgrade note.** Colour output changes on several paths. If you have approved
masters made with 3.1.x, re-check them before conforming new work against them —
particularly anything graded through the Viewer's delivery panel, tone-mapped
with AgX, or exported through the ACES 2.0 Cinema or HLG transforms.

### Added

- **109 nodes, up from 100.** Nine complete node classes were written and never
  listed in any mapping dict, so they never reached ComfyUI's menu:
  Bit-Depth Degrade, Policy Guard, LUT Apply, LUT Blend, Digital Cinema Read,
  Digital Cinema Write, Flipbook GIF, Preview Server and ControlNet Apply.
- **Per-node functional tests.** `tests/test_node_functional.py` builds inputs
  from each node's own `INPUT_TYPES` and calls its `FUNCTION`, checking return
  arity, IMAGE/MASK shape and dtype, and that inputs are not mutated in place.
  80 nodes execute, 30 skip with a stated reason, none fail.
- **A delivery payload contract.** `GRADE_PAYLOAD_KEYS` plus a test that parses
  the viewer's JS and the handler's Python and diffs them in both directions.
- **`core/ffmpeg.py`** — resolves ffmpeg from `RADIANCE_FFMPEG`, then PATH, then
  the binary `imageio-ffmpeg` ships.
- **`core/video.py`** — one video decoder for the whole package. Probes the
  container for frame rate, frame count, bit depth, alpha, colour range, matrix,
  primaries, transfer characteristics, rotation, field order and start timecode,
  then decodes through a single raw ffmpeg pipe. `tests/test_video_read.py`
  encodes real ProRes 4444, ProRes 422 HQ and H.264 fixtures and decodes them
  back — 73 tests.
- **Frame ranges on video.** `start_frame`, `end_frame` and `frame_step` now
  apply to clips as well as sequences, selecting on the decoder's own frame
  counter so the range is exact for long-GOP codecs too.
- **Six more input colour spaces on Read** — Rec.709 (BT.1886), Canon Log 3,
  RED Log3G10, PQ (ST.2084), HLG (BT.2100). The inverses were already in
  `color/transfer.py`; only the menu was missing them.
- **`core/tiling.py`** — shared tile blending weights with correct border
  handling.
- **A real-torch gate in the test suite.** The MagicMock torch stub defeated
  every self-skip idiom in use; CI had been red for seventeen days behind a
  stale `--ignore` list.
- **`core/formats.py`** — the extension tables are built from what the installed
  backends actually register, not from a list someone typed. 9 image extensions
  became 64 on a stock install. Installing OpenImageIO adds DPX, Cineon, ARRI
  and camera raw without a code change, and an unsupported file now says which
  package would open it.
- **`core/exr.py`** — multi-part and multi-layer EXR. A Nuke or Arnold render
  with `diffuse`, `specular`, `Z` and `N` reads every layer through a `layer`
  widget, populated from the file itself by `/radiance/media/layers`.
- **Sequence auto-detection.** Picking `sh010.1004.png` reads the whole
  sequence and reports its real range, the way Nuke's Read does.
  `media_type = Image` is the escape hatch.
- **A third output, `info`** — JSON describing what was actually read:
  resolution, frame count and range, bit depth, codec, EXR layers and windows,
  colour tags, timecode. Nuke's metadata tab, as a wire. Appended last, so
  workflows saved against the two-output version keep working.
- **`on_error`, `raw` and `premultiplied` on Read.** Nuke's error policy, raw
  bypass and unpremultiply, with the same defaults Nuke uses.
- **The Read node hides widgets that do not apply** to the detected media type,
  and draws the file's format, range and layers on itself.

### Fixed — colour

- **The delivery panel dropped half the grade.** Shadows, Highlights, Hue, LUT
  and gamut compression were read by the exporter and never sent by the viewer,
  so each exported at its identity default while the viewer showed it applied.
  Tint was sent and read nowhere. Temperature used a Kelvin white-balance
  multiply against the viewer's additive slider — a different curve entirely.
- **ACES 2.0 Cinema** flat-lined above 0.4 scene-linear at a peak white of
  ~27 nits instead of 48. **ACES 2.0 HLG** placed diffuse white at the display
  peak: 18% grey rendered at ~127 nits where BT.2408 specifies ~26.
- **AgX** applied its inset matrix untransposed, tinting every neutral (channel
  spread 0.042 at 18% grey, 0.116 at 16.0), and double transfer-encoded its
  output, raising the black floor to ~12/255 so pure black was unreachable.
- **Alpha was tone-mapped, expanded and graded.** A 50% matte came back at
  0.607, 0.214 or 1.059 depending on the path.
- **Sony S-Log3's toe** used half the specified slope with a spurious offset:
  black encoded to 0.127921 instead of 0.092864 (~36 code values at 10-bit).
- **Colour Space Convert silently skipped the primaries transform** for
  ACES2065-1, DCI-P3 and Display-P3 — three of twelve advertised spaces
  returned the Rec.709 result unchanged.
- DaVinci Intermediate, Canon Log 3 and RED Log3G10 rewritten to spec; the
  DaVinci Wide Gamut and ARRI Wide Gamut 4 matrices had wrong third rows;
  Bradford D65↔D60 adaptation added; the tone-scale shoulder rebuilt.

### Fixed — output and data integrity

- **Writing a versioned file could destroy the previous version.** `_out_path`
  used `Path.with_suffix()`, which replaces everything after the last dot, so
  `sh010.comp_v0001` and `sh010.comp_v0002` both wrote to `sh010.exr`. With
  overwrite on by default, each render silently replaced the last approved one.
- Legal-range limiting was ungated, squeezing 32-bit EXR masters into
  [16/255, 235/255]. Vertical aspect ratios cropped instead of pillarboxing.
- A wheel built on a working machine shipped that developer's own `.rad` shot
  files.
- The session log is now written atomically under a lock; concurrent deliveries
  used to lose entries and a crash mid-write discarded the whole history.

### Fixed — video read

The Read node's video path did not behave the way any other application in a
facility behaves. Every item below is measured against the fixtures in
`tests/test_video_read.py`.

- **ProRes 4444 alpha was decoded and thrown away.** Frames came back through an
  RGB-only reader, so a vendor plate with a matte arrived as three channels and
  the `mask` output was always zeros. The file's alpha now reaches `mask`.
- **The second decoder quantised everything to 8 bits.** `_load_video_to_numpy`
  — used by the DCC handoff paths — tried OpenCV first, and OpenCV returns 8-bit
  BGR regardless of the source. A 12-bit ProRes 4444 came back on an exact 1/255
  grid, losing four bits per component. It now shares one decoder with Read.
- **Decoding wrote a PNG per frame to a temp directory and read them back.** A
  4-second 1080p ProRes 422 HQ clip: 25.0 s and roughly 600 MB of scratch files,
  against 12.9 s and nothing on disk now.
- **A decode that failed part-way through returned a short clip with no error.**
  OpenCV's read loop just stopped. Both a truncated file and a non-zero ffmpeg
  exit now raise, and the message quotes ffmpeg's own stderr.
- **Every failure in Read became an 8×8 black frame.** A missing plate, a
  corrupt MOV and an unrecognised path all logged an error, returned black, and
  left the node green — so the graph carried on and wrote a master out of it.
  Read now raises. A Read with no path set still returns a frame, because a node
  just added to the canvas is not a failure.
- **The container's colour tags were never read.** A Rec.709 delivery was passed
  through as if it were scene-linear, so every downstream exposure, blur and
  blend operated on gamma-encoded values. On Auto, a tagged file now decodes
  through the matching curve and says which. An untagged file still passes
  through, but warns instead of doing it silently.
- **`nb_frames` of 0 was reported as the frame count** for MXF and the many MOVs
  that carry no count. It now falls back to `duration × fps` and marks the
  number estimated.
- **`r_frame_rate` parsing raised on a bare `"30"`** and took the width and
  height down with it, because one `except` covered the whole probe.
- **The recognised-extension list had seven entries**, so `.m2ts`, `.mts`,
  `.r3d`, `.braw` and friends were classified "unknown" and handed to the image
  reader. One list now serves the browser, the detector and the decoder.
- **The fixed 300-second decode timeout** turned any long clip into a spurious
  failure. There is no cap by default.

### Fixed — reading files at all

- **Nine hand-typed image extensions decided what the node would open.** TGA,
  SGI, PPM, PGM, JP2, PCX and ICO were classified "unknown" and refused —
  measured, every one of them decoded correctly through the reader underneath.
  They never reached it.
- **A multi-layer EXR was rejected, and the message blamed the file.** A Nuke or
  Arnold render with AOVs — the normal output of both — raised "which
  RadianceRead does not support". Every layer now reads.
- **A depth-only or data-only EXR raised.** A Z pass is a render output, not a
  malformed file.
- **The EXR reader ignored the display window.** An overscan render came back at
  the data-window resolution and offset with no warning. It is now conformed to
  the display window; `raw` keeps the overscan. This was on the known-limitations
  page.
- **A sequence's alpha was discarded**, exactly like the ProRes 4444 case:
  `img_t, _ = _read_image(p)` for every frame. An RGBA PNG or EXR sequence came
  back with an empty mask.
- **A sequence numbered from anything but 1001 read nothing**, because
  `start_frame` defaults to the VFX convention. A start outside the range that
  exists on disk is now treated as unset, and said out loud.
- **The frontend's video-extension list disagreed with Python in both
  directions** — it listed `.webp`, which is a still, and omitted `.mxf` — so
  the wrong widgets were shown for the files this pack exists to open. One list
  now, with a test that fails if they drift.

### Fixed — performance and stability

- **The built-in upscalers ran on the CPU.** They selected `images.device`,
  which is always CPU for a ComfyUI IMAGE — roughly ten minutes a frame at 4K
  against about five seconds on a GPU.
- **The delivery export blocked ComfyUI's websocket** for its whole duration:
  grading, filters, a model upscale and ffmpeg all ran on the event loop. The
  Project Manager dashboard did the same while walking the output tree.
- `overlap_temporal=1` — the widget minimum, and the value the tooltip
  recommends — produced fully black frames at every window boundary.
- `torch.quantile`'s 2²⁴-element limit made the Multipass Master node fail on
  every 4K plate.
- The Sampler patched the loader's cached ModelPatcher in place whenever
  `cfg <= 1.0` (the Flux default), leaking a stale patch into every later run;
  and the second CFG patch overwrote the first rather than wrapping it.
- Real-ESRGAN loaded with `strict=False` against mismatched layer names: 8 of
  702 tensors matched and the model ran at near-random initialisation while the
  log reported a successful load.
- Tile blending ramped image borders as if they were seams, darkening the frame
  perimeter.
- Four unbounded GPU-resident model dictionaries replaced with bounded LRU
  caches; `RADIANCE_CACHE_SIZE=0` no longer raises.

### Fixed — controls that did nothing

- **PAG** guarded on an `extra_options` key ComfyUI never sets, so the patch was
  a no-op on every call while logging that it had been applied, and `pag_scale`
  was a gate rather than a strength.
- **Restart sampling** ran after the schedule had finished, passed its noised
  latent through the wrong argument, used the wrong noise variance, and stopped
  before returning to σ=0.
- **`blend_mode`** appeared exactly once in the tiling function — in its own
  signature.
- **`chromatic_adaptation`** returned bit-identical output for every option.
- **The Read node's RELOAD button** never rendered: `reload` was declared as a
  hidden input, where ComfyUI only populates magic-string keys.

### Security

- Removed arbitrary code execution from the Nuke bridge. The guard was a
  substring blocklist that a single space defeated (`open (` does not contain
  `open(`), and that module chaining through the permitted `json` global
  defeated outright. Structured commands and literal values only.
- `torch.load(weights_only=True)` on every user-reachable checkpoint path.
- `/radiance/ocio/load` accepted any absolute path on the host; it is now
  contained to the bundled config, `$OCIO`, `RADIANCE_OCIO_ROOTS` and the
  ComfyUI models tree.
- Delivery history in the viewer is HTML-escaped.

### Changed

- Startup reports a shortfall as an ERROR naming each failed module. It used to
  print "successfully loaded N nodes" whether N was 109 or 12.
- `colour-science`, `einops` and `torchsde` removed from the runtime
  dependencies — nothing imported them. `OpenImageIO` added to the three
  platform requirements files, where it was missing despite being required for
  DPX.
- One implementation each of `escapeHtml` and the widget helpers, replacing five
  and six diverged copies.
- 49 exception handlers that silently swallowed failures now log at DEBUG with
  the operation and exception type.

### Known limitations

- The ACES 2.0 tone scale is a log-space contrast of 1.55 with a tanh shoulder,
  not the Daniele Evo curve the specification defines. 18% grey therefore sits
  about 0.84 stop above the ACES 2.0 reference on SDR, and HLG diffuse white
  lands at signal 0.915 rather than 0.75. The normalisation defects around it
  are fixed; the curve itself is not yet the published one.
- `blend_mode="laplacian_pyramid"` falls back to the Gaussian feather and logs
  that it has done so.
- `chromatic_adaptation` has no effect — the adaptation is baked into the
  precomputed conversion matrices. The widget warns when set to a non-default.
- Optical flow is single-scale Lucas–Kanade despite the DIS reference in its
  docstring; it recovers about 1% of a 5-pixel displacement.
- Scene-cut detection normalises scores by the batch maximum, so the threshold
  has no absolute meaning and cut-free footage still reports cuts.

## [3.1.2] - 2026-07-02 ("GPU-First Release Candidate")

GPU-first completion pass for HDR, RUDRA decode, denoise, motion/flow, upscale, and VFX finishing paths, plus the missing HDR tone-map node migration.

### Fixed

- **HDR Tone Map now loads from the organized HDR package.** `RadianceHDRToneMap` and `RadianceHDRExpandDynamicRange` are registered through `nodes/hdr/`, so saved workflows and the HDR menu can resolve the tone-map node again.
- **HDR Tone Map is GPU-first end to end.** The implementation no longer falls back to NumPy or forces CUDA results back to CPU; tone mapping now stays on the selected Torch device.
- **RUDRA compatibility restored on the public decoder API.** `radiance.model.vae` now reuses the maintained fast decoder implementation, including dynamic-range conditioning (`dr_dim` / `dr_proj`), predictor fallback, checkpoint inference, and hardened explicit-checkpoint loading.
- **Version metadata synchronized.** Python package metadata, runtime constants, README badge, and `package.json` now agree on `3.1.2`.

### Changed

- ACES/HDR color operations, denoise, optical-flow motion blur, multipass helpers, and upscale inference paths were tightened to prefer Torch/GPU execution and avoid unnecessary CPU transfers.
- The node-key snapshot was updated for the intentional HDR tone-map registry restoration.

### Testing

- Restored and enabled the RUDRA compatibility tests that were previously skipped.
- Release verification run performed with real Torch, OpenEXR, OpenColorIO, and colour-science dependencies.

## [3.1.1] - 2026-06-02 ("Fidelity and Trust Release")

Correctness, safety, and delivery-trust fixes for the 32-bit / HDR / EXR pipeline, plus real-AOV ingestion and hardened model loading. Closes the critical data-integrity and security findings from the pre-release engineering review.

### Fixed

- **HDR/EXR load no longer crushes scene-linear data.** The image-to-tensor path previously divided any array whose brightest pixel exceeded 2.0 by 255 (or 65535), silently darkening every real HDR/EXR plate ~255x. Normalization is now driven by the source format: integer formats normalize by their type max; float formats (EXR/HDR/float-TIFF) pass through untouched. Read -> Write of an EXR is now lossless.
- **EXR writer can no longer write a silent 0-byte or downgraded file.** It raises a clear error when no EXR backend is available instead of reporting a phantom success.
- **EXR writer preserves alpha and single-channel mattes** (1->RGB grayscale, 3->RGB, 4->RGBA), and no longer crashes on `(H,W)` matte input.
- **`RadianceWrite` surfaces failures** by re-raising on write error (node turns red) instead of swallowing the exception.
- **Sampler non-finite guard.** `RadianceSamplerPro` detects NaN/Inf in the sampled latent, warns, and sanitizes with `nan_to_num` so blown CFG/precision runs are visible instead of shipping black frames.
- **HDR VAE Decode metadata output.** The node now emits its decode-settings JSON as a second `STRING` output (previously built and discarded).
- Removed internal imports of deprecated shim paths, eliminating load-time `DeprecationWarning`s.

### Security

- **`torch.load` hardened.** All shippable checkpoint loads use `weights_only=True` (VAE, fast VAE, multipass depth/normal models, training data prep), preventing arbitrary-code-execution from malicious `.ckpt`/`.pth` files.
- **Model downloads are atomic + integrity-checked.** The loader and multipass downloaders write to a temp file and atomically move into place, with optional SHA-256 verification and `RADIANCE_LOADER_OFFLINE` / `RADIANCE_UPSCALE_OFFLINE` switches for airgapped setups.

### Added

- **Multipass: AOV Reader.** Reads a real multilayer/AOV OpenEXR (Arnold, Redshift, Karma, Cycles, V-Ray) and splits its named layers into the same outputs as the Multipass Master extractor, so ground-truth render passes flow straight into the EXR-passes writer and relight/comp chain. Scene-linear values preserved; missing layers come through black.
- **Alpha output on the EXR write node.** `RadianceWrite` gained an optional `mask` input, written as the EXR alpha channel (RGBA) for EXR formats.
- **Upscaler HDR + color handling.** Image and video upscalers gained `hdr_mode` (Reinhard tonemap round-trip) and `color_encoding` (linear<->sRGB / linear<->LogC3 OETF round-trip).

### Documentation

- Website docs now render Mermaid workflow diagrams as themed graphs (previously shown as raw code) via `scripts/build_website_docs.py`.

### Testing & CI

- Full `pytest tests/` passes against real torch + OpenEXR + OpenColorIO + colour-science (1347 passed, 34 skipped).
- CI installs real runtime dependencies and runs the whole suite; the publish gate runs the full suite. Removed a stale CI import of a non-existent module.
- Added `tests/test_io_hdr_regression.py` covering the load/write/mask fixes.

## [3.1.0] - 2026-05-25 ("Temporal, Viewer, and Registry Release")

### Added

- Viewer timeline upgrades: filmstrip thumbnails, pinned-frame A/B comparison, and OCIO display transform controls.
- Four color-science nodes: Hue Curves, RGB Curves, White Balance using Bradford adaptation, and Color Space Convert with OCIO-first behavior plus analytical fallback.
- Organized package namespace for color, HDR, I/O, monitor, pipeline, training, upscale, video, VFX, and generation nodes while preserving legacy flat module imports.
- GitHub CI and registry publish workflows for the v3 release path.

### Fixed

- Printer Lights viewer listener leak after tab switches and UI rebuilds.
- HDR VAE decode category placement and repeated engine construction.
- VAE encode/decode/roundtrip docstring placement, category labels, and mode-aware NaN/Inf sanitation.
- Fast VAE trained decoder cache now keys by latent channel count and model mode.
- Viewer grading edge cases, including `luma_mix` fast-path handling, CLog3 output clamping, ACES matrix allocation, vectorized saturation, and bounded progress tracking.

### Release

- Updated Registry metadata for `radiance` v3.1.0 under publisher `fxtdstudios`.
- Restored GitHub/Registry image assets referenced by README and `pyproject.toml`.
- Added `.comfyignore` so Registry packages exclude tests, local scratch files, build caches, and review documents.

---

## [3.0.1] - 2026-05-07 ("Color Science Precision & Demo Tooling")

### Added

- **BT.1886 EOTF** (`nodes_hdr_colorspace.py`): Added `"BT.1886 (TV γ2.4)"` to `_EOTF_MAP`
  — ITU-R BT.1886 display EOTF (γ 2.4) for Rec.709 broadcast reference monitors. Distinct
  from `"Rec.709 (OETF)"` (camera signal encoding) and `"Gamma 2.4"` (generic power curve).
- **Rec.709 / BT.1886 IDT** (`nodes_colorscience.py`): Added `"Rec.709 / BT.1886"` to
  `_COLOR_SPACES` — the correct Input Device Transform for display-referred Rec.709 material.
  OCIO mapping: `"Output - Rec.709"`. Analytical fallback: γ 2.4 linearise / γ 1/2.4 encode.
- **◎ Radiance Bit-Depth Degrade** (`nodes_colorscience.py`): Quantize float images to N-bit
  (4–16) precision. Dither modes: none (hard clip), triangular TPDF, Floyd-Steinberg error
  diffusion. Outputs: quantized image, amplified error delta, banding mask, metrics JSON
  (PSNR dB, max error, dynamic range loss in stops). Critical for demo workflows:
  ARRI LogC / Venice RAW → 8-bit → AI HDR reconstruct → EXR.

### Tests

- **`tests/test_colorscience_v3.py`** (32 tests): BT.1886 decode/encode round-trip,
  `_COLOR_SPACES` contents, `RadianceBitDepthDegrade` node registration, quantisation
  correctness (4-bit ≤16 levels, 8-bit ≤256, 16-bit near-lossless), PSNR monotonicity
  with increasing bit depth, TPDF vs hard-clip divergence, Floyd-Steinberg completion,
  banding mask binary property, alpha channel preservation.

### Changed

- `_COLOR_SPACES` in `nodes_colorscience.py` grouped into labelled sections (scene-linear,
  display-referred SDR, camera log/raw) for clarity.
- `_EOTF_MAP` in `nodes_hdr_colorspace.py` restructured with inline comments separating
  SDR EOTFs, HDR EOTFs, and camera log curves.

---

## [3.0.0] - 2026-04-30 ("The Full Pipeline & Intelligence Update")

### Added — ACES 2.0 Full Pipeline (Pillar 03)

- **◎ Radiance ACES 2.0 RRT+ODT** (`nodes_aces2.py`): Analytical ACES 2.0 Reference
  Rendering Transform with four Output Display Transforms (sRGB D65, DCI-P3 D65,
  Rec.2020 PQ HDR10, Rec.2020 HLG). No external CTL toolchain required.
- **◎ Radiance ACES Input Transform** (`nodes_aces2.py`): 25 IDT matrices covering
  ARRI ALEXA, RED, Sony Venice, Canon C-series, Nikon Z, Panasonic Varicam, Blackmagic,
  and GoPro camera systems.
- **◎ Radiance ACES LMT** (`nodes_aces2.py`): Five LMT presets — Linear, Blue Light Fix,
  Golden, Desaturate Highlights, and Kodak-2383 emulation.
- **◎ Radiance ACES CDL** (`nodes_aces2.py`): ASC CDL v1.2 in ACES working space — slope,
  offset, power, saturation with Rec.709 luma weighting and optional clamp.

### Added — Studio Integrations (Pillar 04)

- **◎ Radiance DaVinci Send** (`nodes_studio_integrations.py`): One-click image/sequence
  export to DaVinci Resolve shared folder with configurable bit depth (8/16 bit, EXR).
- **◎ Radiance Nuke Send** (`nodes_studio_integrations.py`): Write a .nk snippet that
  imports the current image into a Nuke session via Read node.
- **◎ Radiance Shot Metadata** (`nodes_studio_integrations.py`): Attach scene/shot/take
  labels, camera info, and lens data as a JSON sidecar and EXIF-compatible comment.
- **◎ Radiance ASC CDL Export** (`nodes_studio_integrations.py`): Write ASC CDL v1.2 XML
  (single `<ColorCorrection>`) or JSON override file from slope/offset/power/saturation
  values — standard interchange with Resolve, SCRATCH, and Nuke.

### Added — Real-Time Preview (Pillar 05)

- **◎ Radiance False Color Monitor** (`nodes_realtime_preview.py`): Cinema-style false
  colour overlay with 10 configurable exposure zones (adjustable luma thresholds and
  zone colours). Toggle between false-colour and source view per node.
- **◎ Radiance Focus Peaking** (`nodes_realtime_preview.py`): Sobel edge highlight overlay
  for critical focus evaluation. Configurable threshold and peak colour.
- **◎ Radiance Split View** (`nodes_realtime_preview.py`): Side-by-side or top-bottom
  wipe comparison between two images, with adjustable split position and guide line.
- **◎ Radiance Contact Sheet** (`nodes_realtime_preview.py`): Auto-tiled contact sheet
  from a batch of images — configurable grid columns and thumbnail padding.
- **◎ Radiance Flipbook GIF** (`nodes_realtime_preview.py`): Assemble batch frames into an
  animated GIF at configurable FPS and scale for quick motion previews.
- **◎ Radiance Frame Stamp** (`nodes_realtime_preview.py`): Burn timecode, frame number,
  and optional label overlay into images. SMPTE 12M timecode — both NDF (24/25/30/48/50 fps)
  and drop-frame (29.97 DF with d=2; 59.94 DF with d=4). `;` separator for DF, `:` for NDF.
- **◎ Radiance Preview Server** (`nodes_realtime_preview.py`): Ephemeral HTTP server
  (daemon thread) serving the last processed frame as JPEG — lets any browser on the LAN
  see the current result without streaming infrastructure.

### Added — AI Assist Layer (Pillar 06)

- **◎ Radiance Auto Grade** (`nodes_ai_assist.py`): Zone-based ASC CDL matching — aligns
  a source image to a reference still by independently correcting shadow offset, midtone
  slope, highlight power, and global saturation. Strength blend 0–1.
- **◎ Radiance CLIP Match** (`nodes_ai_assist.py`): Cosine similarity match between source
  and a pool of reference images using CLIP ViT-B/32 embeddings. Falls back to
  Bhattacharyya histogram similarity when `transformers` is unavailable. Returns the best
  matching reference image plus a similarity score.
- **◎ Radiance Continuity Check** (`nodes_ai_assist.py`): Shot-to-shot continuity analysis
  — flags luma drift, colour cast, saturation drift, contrast drift, and histogram
  dissimilarity between consecutive frames. Outputs a JSON report + clean/dirty flag.
- **◎ Radiance Grade Prompt** (`nodes_ai_assist.py`): Natural language CDL grading — parse
  35 intent rules ("warmer", "crushed blacks", "pull the highlights", etc.) with intensity
  modifiers (very / slightly / a touch / less). Six preset looks: film, bleach, day-for-
  night, instagram, neon, noir.

### Added — Test Infrastructure (Pillar 07)

- `tests/test_cdl.py`: CDL math, XML round-trip, and node registration tests.
- `tests/test_scene_cut.py`: Scene-cut detection — histogram diff, Sobel edge diff,
  combined method, min-shot-frame enforcement.
- `tests/test_temporal_coherence.py`: Flicker removal and temporal smoothing helpers
  (`_luminance_per_frame`, `_rolling_median`, `_gaussian_smooth`, `_smooth_track`).
- `tests/test_colorscience.py`: Bradford matrix algebra, `_xy_to_XYZ`, and illuminant
  chromaticity coverage — 18 tests (Bradford matrix tests torch-gated).
- `tests/test_curves.py`: HSL round-trip for grey, saturated primaries, and random
  images; hue/saturation range checks.
- `tests/test_optics.py`: Lens distortion, chromatic aberration, anamorphic streaks, and
  vignette — registration, input structure, and return-type validation.
- `conftest.py`: Extended torch stub with 20+ additional attributes so module-level
  `torch.*` calls in all nodes succeed at collection time without real torch.

### Added — AI & Pipeline Nodes (Pillar 08)

- **◎ Radiance LLM Driver** (`nodes_llm_driver.py`): Model-agnostic LLM backend
  configuration — Claude (Anthropic), GPT-4o (OpenAI), Gemini (Google), Ollama (local).
  Single driver JSON wire connects to any downstream AI node.
- **◎ Radiance LLM Prompt** (`nodes_llm_driver.py`): Send text prompts to the configured
  backend; returns response string and echoes driver config.
- **◎ Radiance LLM Image Query** (`nodes_llm_driver.py`): Vision-capable multimodal query
  (base-64 PNG encoded) — falls back to text-only for non-vision backends.
- **◎ Radiance Agent Shot Analyst** (`nodes_agent_pipeline.py`): Per-shot technical analysis
  — exposure, contrast, saturation, dynamic range, and scene classification via LLM.
- **◎ Radiance Agent Grade Advisor** (`nodes_agent_pipeline.py`): AI-driven CDL suggestions
  from shot analysis; outputs slope/offset/power/saturation JSON.
- **◎ Radiance Agent QC Check** (`nodes_agent_pipeline.py`): Automated quality gate —
  runs policy rules and returns pass/fail flag + violation report.
- **◎ Radiance Agent Pipeline** (`nodes_agent_pipeline.py`): Shot Analyst → Grade Advisor →
  QC Check orchestration chain in a single node.
- **◎ Radiance Studio Knowledge Base** (`nodes_knowledge_base.py`): CLIP/HSV image vector
  store with semantic query — index reference stills and retrieve by similarity.
- **◎ Radiance MCP Bridge** (`nodes_mcp_bridge.py`): JSON-RPC MCP server exposing Radiance
  as a tool endpoint; includes client stubs for Nuke, Maya, and ShotGrid.
- **◎ Radiance Policy Preset** (`nodes_policy_guard.py`): Six delivery policy presets —
  Broadcast SDR, Cinema DCP, Streaming HDR, Web SDR, Custom. Outputs JSON policy dict.
- **◎ Radiance Policy Guard** (`nodes_policy_guard.py`): Per-frame delivery compliance gate
  — evaluates peak, clipping, black crush, saturation, and metadata presence against policy.
  Returns original image (pass-through), compliance report STRING, and pass/fail BOOLEAN.
- **◎ Radiance Render Dispatch** (`nodes_render_dispatch.py`): Submit render jobs to
  Deadline, Tractor, and OpenCue; poll status; cancel; list queued jobs.
- **◎ Radiance Audio Cut** (`nodes_audio_cut.py`): Beat/onset detection from audio waveform
  — outputs cut timecodes and Whisper speech transcription (when available).

### Added — DiT Video Model Integration (Tier 1)

- **◎ Radiance DiT Model Config** (`nodes_dit_adapter.py`): Unified architecture config for
  SD-VAE, SDXL-VAE, LTX-Video (128ch), HunyuanVideo (16ch), Wan 2.1 (16ch), CogVideoX
  (16ch), and Mochi-1 (12ch). Outputs spec JSON for downstream adapter nodes.
- **◎ Radiance DiT Latent Adapter** (`nodes_dit_adapter.py`): SD-VAE latent ↔ DiT latent
  channel projection with per-architecture mean/std normalisation. Adaptive channel project
  supports expand (tile) and contract (fold) modes.
- **◎ Radiance DiT Latent Norm** (`nodes_dit_adapter.py`): Per-architecture normalisation
  and denormalisation of DiT latents. Forward and inverse pass.
- **◎ Radiance DiT Inspect** (`nodes_dit_adapter.py`): Latent shape / stats inspector
  reporting channels, spatial dimensions, dtype, min/max/mean.
- **◎ Radiance DiT Frame Split / Merge** (`nodes_dit_adapter.py`): Split video latents
  into per-frame chunks and reassemble — enables frame-parallel DiT workflows.
- **◎ Radiance HDR Video Conditioner** (`nodes_video_hdr.py`): HDR conditioning for DiT
  video models — builds latent sequence from EXR frames with scene-linear encoding.
- **◎ Radiance HDR Video Decoder** (`nodes_video_hdr.py`): Decode DiT video latents back
  to HDR image sequence with optional PQ/HLG EOTF application.
- **◎ Radiance HDR Video Prompt Builder** (`nodes_video_hdr.py`): Construct HDR-aware CLIP
  prompt strings from scene metadata (peak nits, colour space, HDR format).
- **◎ Radiance HDR Video Assembler** (`nodes_video_hdr.py`): Assemble decoded frames into
  an EXR sequence or MP4 (ProRes 4444 / H.265 10-bit).

### Added — Character Consistency System

- **◎ Radiance Character Anchor** (`nodes_character.py`): Build a character profile from a
  reference image — CLIP embedding, HSV histogram, face embedding (facexlib), and metadata.
  Saves to JSON sidecar; returns profile STRING for downstream enforcement.
- **◎ Radiance Character Enforce** (`nodes_character.py`): Inject character embedding into
  CONDITIONING via token-append, IP-Adapter strength, or text-blend strategies.
- **◎ Radiance Character Checker** (`nodes_character.py`): Per-frame cosine similarity check
  against a stored profile. Outputs similarity score, pass/fail flag, and report JSON.
- **◎ Radiance Character Blend** (`nodes_character.py`): Weighted blend of two character
  profiles — useful for gradual style evolution across a sequence.
- **◎ Radiance Character Gallery** (`nodes_character.py`): Load all character profiles from
  a folder; display as structured JSON gallery for selection.
- **◎ Radiance Character Score Timeline** (`nodes_character.py`): Per-shot similarity scores
  serialised as JSON timeline for consistency trend analysis.

### Added — T2V & I2V Pipeline

- **◎ Radiance T2V Wrapper** (`nodes_t2v_pipeline.py`): Text-to-video inference wrapper for
  LTX-Video, HunyuanVideo, Wan 2.1, and CogVideoX. Unified prompt/resolution/steps API.
- **◎ Radiance I2V Wrapper** (`nodes_t2v_pipeline.py`): Image-to-video inference with first-
  frame conditioning; supports motion bucket / flow magnitude parameters.
- **◎ Radiance Video Sampler** (`nodes_t2v_pipeline.py`): Low-level DiT sampler with
  sigma schedule, CFG, and optional restart sampling.
- **◎ Radiance Batch Decode** (`nodes_t2v_pipeline.py`): Decode a batch of video latents to
  pixel frames using the architecture-appropriate VAE.
- **◎ Radiance Video Export** (`nodes_t2v_pipeline.py`): Write decoded frames to ProRes
  4444, DNxHR, H.265, or EXR sequence; embeds timecode metadata.

### Added — AI Upscaler (nodes_upscale.py)

- **◎ Radiance Upscale Tiler** (`nodes_upscale.py`): Anti-seam tiling engine with
  Gaussian-weighted overlap (≥20%), 4-level Laplacian pyramid blending, and per-tile
  confidence map. Handles arbitrary resolution with no visible tile boundaries.
- **◎ Radiance Upscale Image** (`nodes_upscale.py`): Three-tier upscaler for images.
  - Tier 1 — Real-ESRGAN (RRDB, no basicsr): `realesrgan_x4plus`, `x4plus_anime`, `x2plus`
  - Tier 2 — Transformer SOTA (spandrel): HAT-L ×4/×2, SwinIR-L ×4
  - Tier 3 — Diffusion creative: SD ×4 upscaler (stabilityai), SeedVR2 one-step video
  - Auto tier: content classifier (noise/sharpness/saturation/ai-likelihood) selects tier
- **◎ Radiance Upscale Video** (`nodes_upscale.py`): Temporal-coherent video upscaler.
  SeedVR2-style 4n+1 overlapping windows, sinusoidal ramp weights at window edges,
  Lucas-Kanade optical-flow warp for cross-window reference alignment, Laplacian pyramid
  blend at seams. Eliminates inter-chunk flicker.
- **◎ Radiance Upscale Router** (`nodes_upscale.py`): Content-aware tier selector — analyse
  any image/video and route to the appropriate upscale node automatically.
- **◎ Radiance Upscale Face Restore** (`nodes_upscale.py`): Post-upscale face enhancement.
  Face detection: facexlib RetinaFace → OpenCV Haar cascade fallback. Restoration: spandrel
  (auto-detects CodeFormer / GFPGAN from checkpoint) → basicsr CodeFormer → GFPGANer →
  identity. Gaussian-feather composite. `fidelity_weight` controls realism vs. fidelity.
- **◎ Radiance Upscale Colour Fix** (`nodes_upscale.py`): Histogram-match colour-drift
  correction post-diffusion. Fast path: `torch.searchsorted` vectorised CDF inversion
  (~100× faster than pixel-loop). Strength 0–1 blend with original.

### Added — Test Coverage (Pillar 07 sweep)

- `tests/test_upscale.py` (120 assertions): Gaussian kernel normalisation/symmetry, weight
  map positivity, bicubic output shapes, `tiled_upscale` shape/confidence, histogram-match
  identity/strength-0 noop, content classifier keys, tier recommendation, quantisation math,
  model registry completeness, node registration (6 nodes).
- `tests/test_llm_driver.py`: `_driver_from_json` parse + fallback, `_resolve_api_key` env
  resolution, `_DISPATCH` routing for all 4 backends (Claude/GPT/Gemini/Ollama), INPUT_TYPES
  field names, 3-node registration.
- `tests/test_policy_guard.py`: `_luma` BT.709 coefficients, saturation range, gamut
  fraction, `_analyse` key coverage, `_evaluate` tuple return (passed, violations, score),
  metadata presence checks.
- `tests/test_dit_adapter.py`: `_get_spec` for 6 architectures, channel projection shape,
  `_apply_norm` round-trip, config/adapter/norm/inspect/split/merge INPUT_TYPES.
- `tests/test_character.py`: `_has`, HSV histogram, `_cosine_sim` identity/orthogonal/
  opposite, all 6 character node INPUT_TYPES and RETURN_TYPES.
- `tests/test_colorscience_v3.py` *(see v3.0.1)*: BT.1886 decode/encode round-trip,
  `_COLOR_SPACES` contents, `RadianceBitDepthDegrade` quantisation, PSNR monotonicity,
  dither divergence, FS dither completion, banding mask binary property.
- `tests/test_timecode.py` *(Pillar 09 — 68 tests)*: Full SMPTE 12M timecode suite.
  NDF 24/25/30 fps boundary frames; DF 29.97 (d=2) covering frames 0, 1799, 1800, 3597–3599,
  17981–17983, 107891–107892; DF 59.94 (d=4) equivalent boundaries; monotonicity within
  short minutes; dropped-frame absence at non-10th-minute starts; `;` vs `:` separator;
  `_DF_RATES` constant validation; `RadianceFrameStamp` INPUT_TYPES / RETURN_TYPES.

### Added — CI / Packaging (Pillar 10)

- **`.github/workflows/ci.yml`**: Full GitHub Actions CI matrix — Python 3.9 · 3.10 · 3.11 · 3.12
  on `ubuntu-latest`. Installs headless test deps (`opencv-python-headless`, `pytest-cov`),
  runs `pytest` with `--cov`, uploads XML coverage to Codecov (3.11 only), and attaches pytest
  logs as artifacts on failure. Concurrency group cancels in-flight runs on new push.
- **`.github/workflows/ci.yml` — smoke job**: Independent import smoke-test that verifies 11 key
  node modules (`nodes_scene_cut`, `nodes_ai_assist`, `nodes_aces2`, `nodes_realtime_preview`,
  `nodes_llm_driver`, `nodes_policy_guard`, `nodes_upscale`, `nodes_dit_adapter`,
  `nodes_character`, …) import cleanly with minimal stubs — catches syntax errors and bad
  top-level imports independently of the pytest run.
- **`.github/workflows/ci.yml` — lint-config job**: Validates `pyproject.toml` (TOML parse +
  required table check) and all `.github/workflows/*.yml` files (YAML parse) on every push.
- **`.github/workflows/publish.yml`**: Publish workflow triggered by version tags (`v*.*.*`) or
  `workflow_dispatch`. Runs full test gate, then calls `Comfy-Org/publish-node-action@v1` with
  `REGISTRY_ACCESS_TOKEN` secret, then creates a GitHub Release with notes extracted from
  `CHANGELOG.md`. Pre-release tags (`-rc*`) set `prerelease: true`.
- **`pyproject.toml`** — expanded and tightened:
  - Python classifiers now enumerate 3.9 · 3.10 · 3.11 · 3.12 explicitly.
  - Runtime `dependencies` list pinned with minimum versions (`Pillow>=9.0`, `numpy>=1.22`, …).
  - `opencv-python` moved from runtime to `[full]` extra; `opencv-python-headless>=4.7` added
    to `[test]` extra (server-safe, no GUI dependency).
  - `[test]` extra adds `pytest-cov>=4.0` and `pytest-timeout>=2.1`.
  - `[full]` extra extended to cover `diffusers`, `accelerate`, `peft`, `anthropic`, `openai`,
    `google-generativeai`.
  - `Bug Tracker` URL corrected to GitHub Issues.
  - `[tool.coverage.run]` and `[tool.coverage.report]` sections added (`fail_under = 0`,
    ready to raise once baseline is measured).
  - Stale `tb = "short"` (unknown pytest option) removed; `integration` marker added.

### Changed

- All display names standardised to `◎ Radiance …` prefix across all 7 new nodes_*.py files.
- `tests/test_coherence_prior.py` skip guard now also checks `HAS_TORCH` so it correctly
  skips when the conftest torch stub is active (prevents false-pass on stub MagicMocks).

## [2.6.0] - 2026-04-17 ("The HDR Science & Fast VAE Update")

### Added — Color Science

- **◎ Radiance ACES 2.0 Transform** (`nodes_colorscience.py`): Full analytical ACES 2.0
  Reference Rendering Transform (RRT) + four Output Display Transforms — no external CTL
  toolchain required.
  - RRT: cubic Bézier rational polynomial approximation matching CTL within ±0.3%
  - ODT **sRGB D65** — standard monitor / SDR web output
  - ODT **DCI-P3 D65** — digital cinema / wide-gamut display
  - ODT **Rec.2020 PQ (HDR10)** — ST 2084 absolute luminance up to 10,000 nits
  - ODT **Rec.2020 HLG** — ARIB STD-B67 broadcast HDR (BBC/NHK)
  - Full gamut pipeline: ACEScg → XYZ (D60) → D65 Bradford → target display primaries
  - `exposure_offset` (stops), `saturation`, `peak_nits` (cd/m²), `grade_info` passthrough

- **◎ Radiance White Balance** (`nodes_colorscience.py`): Bradford chromatic adaptation in
  three modes — Temperature/Tint (Kang 2002 xy approx), Illuminant-to-Illuminant CAT
  matrix, and Manual RGB gain. Full HDR-safe, no clamping above 1.0.

- **◎ Radiance Color Space Convert** (`nodes_colorscience.py`): OCIO-powered color space
  conversion with artist-friendly dropdown. Falls back to analytical transforms (sRGB OETF,
  LogC3, ACEScg gamut) when no OCIO config is present. Forward and Inverse directions.
  Supports: Linear sRGB, ACEScg, ACEScc, ACEScct, sRGB, Rec.709, LogC3, LogC4, F-Log2,
  C-Log3, Log3G10, DaVinci Intermediate, BMD Film Gen5, V-Log, N-Log.

### Added — AgX Tone Mapping (Full Pipeline)

- **AgX Full Pipeline** (`hdr/tonemap.py`): Replaced the previous smoothstep approximation
  with a mathematically correct AgX operator (Blender/Troy Sobotka, BSD-licensed).
  - `_AGX_M_IN` / `_AGX_M_OUT` gamut matrices (sRGB ↔ AgX working space)
  - Log₂ inset over working range: −10 … +6.5 EV
  - Sigmoid contrast curve fitted to the AgX CDL
  - Full GPU path via `torch.einsum` (no CPU fallback needed for AgX)
  - CPU fallback via NumPy using the same three-stage pipeline
  - Fixes previous hue shifts in saturated reds/blues under the agx operator

### Added — HDR Generation & Enhancement

- **◎ Radiance HDR Enhancer** (`nodes_hdr_inception.py`): Pixel-space generative HDR
  expansion for scene-linear images. No sampler required for standalone use.
  - Luminance-weighted soft highlight roll-off via configurable knee function
  - Calibrated Gaussian noise injection into near-clip specular regions (sub-pixel detail)
  - Luminance-weighted blend back to original (`blend_strength`)
  - Outputs: `enhanced_image` (IMAGE) + `headroom_mask` (MASK) + `stats` (JSON)
  - `headroom_pct`, `delta_stops` reported in stats JSON
  - Optional `lora_name` hint for HDR LoRA workflows (informational, load via Load LoRA node)

- **◎ Radiance HDR Latent Init** (`nodes_hdr_inception.py`): Properly seeded Gaussian noise
  latent (N(0, σ²)) for HDR-aware generation — replaces zero-latent which caused degenerate
  sampler output. Configurable `sigma` (1.0 = SDXL/SD3, 14.6 = Flux, 3.5 = LCM).

- **◎ Radiance HDR Latent Blend** (`nodes_hdr_inception.py`): Per-channel blend between a
  reference HDR latent and a noise latent. Steers generation toward a tonal distribution
  without full img2img. Per-channel weight overrides via comma-separated string.

### Added — HDR Analysis

- **HDR Histogram: Headroom Zone Markers** (`hdr/analysis.py`): `hdr_zone_markers` input
  draws calibrated vertical lines at 1× (+0 EV), 2× (+1 EV), 4× (+2 EV), 8× (+3 EV)
  scene-linear on the linear histogram panel. Essential for HDR authoring visibility above
  the SDR clip point.
- **HDR Histogram: `headroom_pct` Output** (`hdr/analysis.py`): New FLOAT output pin
  reporting the percentage of luminance pixels above 1.0 (scene-linear SDR ceiling).
  Connect directly to downstream math nodes or display.

### Added — Tier 4: TurboDecoder Training Pipeline

- **RadianceTurboDecoder Architecture** (`fast_vae.py`, `hdr/fast_vae.py`): ~2M parameter
  convolutional decoder that maps VAE latents directly to log-coded images, replacing the
  engineered `_denoise_log_highlights()` + `_soft_log_shoulder()` combination with a
  learned equivalent. Supports Flux (16ch) and SDXL (4ch) latent formats.

- **HDRLogLoss** (`train_turbo_decoder.py`): Four-component HDR-aware training loss:
  - L1 in log space (perceptually uniform tonal weight)
  - MSE in log space (sharpness / outlier penalty)
  - Highlight penalty above configurable `knee` (extra weight on near-clip regions)
  - Structural gradient loss (L1 on image gradients, preserves edge fidelity)

- **EMA Weight Tracking** (`train_turbo_decoder.py`): Exponential moving average of decoder
  weights (`decay=0.999`). EMA weights are saved separately as `turbo_decoder_ema_stepN.pth`
  for smooth, stable inference.

- **`train_turbo_decoder.py`** — Full training CLI:
  - AdamW + Cosine Annealing LR scheduler
  - Configurable steps (50k useful / 200k production), batch size, grad clip
  - Resume from checkpoint, JSONL training log, eval PSNR-log every 2k steps
  - Checkpoint save every 5k steps (full + EMA-only inference weights)

- **`dataset_hdr.py`** — HDR pair dataset generator: produces `.npz` (latent, log-coded
  target) pairs from raw HDR EXRI sources for supervised TurboDecoder distillation.

- **`nodes_fast_vae.py`** — ComfyUI inference node for loading a trained TurboDecoder
  checkpoint and running fast log-domain VAE decoding in production workflows.

---

## [2.5.1] - 2026-04-12 ("The New Architecture Update")

### Added
- **Node Architecture**: Unified `◎ Radiance HDR VAE Decode` to officially use the Radiance 4K production engine. Replaces placeholder with production-grade tiling and log-domain math.
- **Node: ◎ Radiance NDI Sender** — Upgraded with **High-Fidelity Log-Encoding** (LogC4/S-Log3) ensuring full HDR dynamic range is preserved when streaming to external apps in 8-bit.
- **Optics Refinement**: Added professional Boundary Handling (`zeros`, `reflection`, `border`) and `Invert` math to the Lens Distortion and Chromatic Aberration suite.

## [2.5.0] - 2026-04-12 ("The Optics & Architecture Update")

### Added
- **Node: ◎ Radiance Lens Distortion** — True barrel/pincushion distortion with proper ST-Map output generation. Validated for 32-bit float coordinate mapping and edge-wrap prevention.
- **Node: ◎ Radiance Chromatic Aberration** — Real-world spectral dispersion shifting (R/G/B) simulating glass Index of Refraction (IoR), operating from the specified optical center.
- **Node: ◎ Radiance Anamorphic Streaks** — High-end cinematic flare generation employing cascaded anisotropic blurs to organically bloom clipped highlights within the HDR/32-bit linear signal.
- **Node: ◎ Radiance SDR to HDR Expand** — Inverse OETF logic combined with mathematical power-curve spline expansion, successfully recovering squashed SDR highlights into full Scene-Linear data capable of pushing 4,000+ nits dynamically.
- **Node: ◎ Radiance Relight Engine** — True 32-bit fp geometric re-lighting node. Ingests any RGB normal map and applies physically-based Lambertian diffuse and Blinn-Phong specular passes. Allows additive relighting of 2D generative sequences safely.
- **Node: ◎ Radiance HDR VAE Decode** — Intercepts Latent tensor data during decode and bypasses traditional StableDiffusion/Flux clipping mechanisms, mapping latent energies directly to fp32 values to retain dynamic range.
- **Node: ◎ Radiance NDI Sender** — Native PyTorch-to-NDI streaming bridge. Streams PyTorch tensor surfaces as BGRA video frames directly over local network to OBS, Resolume, or Nuke via NewTek NDI SDK.

## [2.4.2] - 2026-04-10 (The Temporal & Intelligence Update)

### Added
- **Viewer: Precision Scopes Suite** — GPU-accelerated **CIE 1931 xy Chromaticity** scope with spectral locus and gamut target overlays (Rec.709, P3, Rec.2020).
- **Viewer: HDR Waveform Upgrade** — Full linear-to-ST.2084 (PQ) non-linear mapping for waveforms. Provides visibility for details up to 10,000 nits without clamping.
- **Viewer: HDR HUD Graticules** — Calibrated horizontal lines and labels for 100, 400, 1000, and 4000 nits in the waveform panel.
- **Node: ◎ Radiance Chromaticity** — Backend analytical node for CIE 1931 visualization. Supports batch processing and gamut target comparison.
- **Node: ◎ Radiance Curves & Hue Curves** — Full piecewise-linear RGB/Master curves and HSL-based secondary color correction engine for 32-bit float pipelines.
- **Viewer: Filmstrip Timeline** — A scrollable row of 40×28px frame thumbnails above the transport scrubber. Click any thumbnail to jump directly to that frame. Active frame is highlighted in accent blue. Max 60 thumbnails with step sub-sampling for long sequences.
- **Viewer: Pin Frame (A/B Wipe)** — New 📌 button in the transport panel. Freezes the current rendered frame as the B-side reference for instant A/B wipe comparison. Click again to release. Integrated with existing wipe renderer path.
- **Viewer: OCIO Display Transform Panel** — New panel in VIEW tab; 37 ACES transforms available when OCIO config is loaded.
- **Viewer: Filmstrip Flicker Heatmap** — Per-frame colored dot (green→magenta) on each filmstrip thumbnail showing inter-frame luma delta; ✂ marker and red border on detected scene cuts.
- **◎ Radiance Scene Cut Detector** — Multi-signal hard/soft cut detection: luma histogram distance, frame delta, LAB chroma shift, and flash guard. Outputs IMAGE + MASK + JSON report.
- **◎ Radiance Temporal Color Lock** — LAB-space batch colour stabiliser. Anchors every frame’s L*a*b* statistics to a reference frame with per-scene auto-reset via cut_mask.
- **◎ Radiance Deflicker Pro** — Professional luminance deflickering: Gain Only (fast), Percentile (selective), and Luma Match (full CDF histogram remapping).
- **◎ Radiance Frame Interpolation** — Temporal upsampler with Linear, Optical Flow (Farneback via OpenCV), and RIFE (external binary) modes. Factor 2–8×.
- **Sampler: CogVideoX + StepVideo** — Full `MODEL_DEFAULTS` entries, `VIDEO_MODEL_TYPES` membership, `CFG_GUIDED_MODELS`, and `detect_model_type()` class + config detection.
- **Sampler: Noise Library Expansion** — Three new pure-PyTorch noise generators added to `NOISE_TYPES`: Simplex (octaved hash-grid), Voronoi (cellular distance field), Curl (divergence-free from spectral potential).
- **◎ Radiance LoRA Scheduler** — Generate per-step LoRA strength schedules using Catmull-Rom / Linear / Step interpolation between user-defined control points. Outputs `FLOAT_LIST` + JSON.
- **◎ Radiance Regional Prompt** — Add a spatial region with its own conditioning via bbox or MASK. Additive / Replace merge modes. Chainable.
- **◎ Radiance Regional Grid** — Divide image into a columns×rows grid; assign a separate text prompt to each cell; encodes inline via CLIP input.
- **◎ Radiance EXR Multi-Part** — Write a Nuke/Resolve-compatible multi-part EXR v2 file with named AOV parts: beauty (RGBA), depth (Z), normal (NX/NY/NZ), albedo, and 2 custom layers. Fallback to per-layer EXRs if OpenEXR multi-part is unavailable.
- **◎ Radiance Render Queue** — Submit parametric sweep runs to the ComfyUI queue. Define `{node_id: {field: [values]}}` and choose Product / Zip / Chain combination modes. Dry-run preview. Returns JSON job manifest.
- **Radiance Write: Remote Output** — `RadianceDigitalCinemaWrite` gains optional `remote_path` input supporting UNC (`\\server\share`) and S3 (`s3://bucket/key`) output via `boto3`.
- **.rad v3 Archive Format** — New ZIP-based `.rad` v3 container (`RADZ` magic). Stores `workflow.json`, `manifest.json` (asset list + metadata), and optional `assets/` directory. Full backward-compat reader handles v1 (plain text), v2 (binary+SHA256), and v3 (ZIP) automatically via `_unpack_any_rad()`.

### Fixed
- **Viewer: Printer Lights Listener Leak (B-14)** — `makePrinterStrip()` was attaching permanent `document.addEventListener('mousemove'/'mouseup')` calls every time the grade panel was rebuilt. Fixed by switching to `AbortController`-scoped listeners.
- **Viewer: Printer Lights UX** — Added EV stop readout per channel (e.g. `+0.40EV`), combined RGB status badge, and a RST button to reset all channels simultaneously.

## [2.3.3] - 2026-04-01


### Fixed
- **A/B Split-Wipe Fix**: Repaired the broken A/B comparison functionality in the Radiance Viewer. The wipe mode now correctly uses the GPU shader and features a draggable split-line with A/B labels.
- **Node Cleanup**: Removed the redundant `RadianceSaveEXR` node in favor of the new `◎ Radiance Write` node.
- **Package Hardening**: Resolved JSON serialization errors in QC reports and standardized environment variables.

## [2.3.2] - 2026-03-30

### Added
- **Smart Overwrite Protection**: `Radiance Write` and `EXR Save` now feature automated index detection to prevent accidental file destruction.
- **Universal Digital Cinema I/O**: Consolidated Video, Image Sequence, and Single Image handling into a high-performance unified pipeline.
- **Terminal HUD & Live REPL**: Nuke-style Python interaction directly inside the viewer for real-time data inspection.
- **Interactive Mask Editor**: Non-destructive brush masking in `◎ Radiance Load Image`.

### Fixed
- **Robust Pipeline Validation**: Enhanced null-safety and input checking for high-load production environments.
- **◎ Radiance Grade Match**: Optimized shot-to-shot color statistics transfer using CIE L*a*b* mean/std math.

## [2.3] - 2026-03-29


## [2.2.2] - 2026-03-29

### Fixed
- **Comfy Registry Metadata**: Removed restrictive `Environment :: GPU :: NVIDIA CUDA` classifier from `pyproject.toml` to allow installation on CPU and macOS (MPS) systems.
- **Registry Icon**: Renamed `RADIANCE_ICON.png` to `flogo.png` and updated `pyproject.toml` to ensure the logo appears correctly in the Comfy Registry.

## [2.2.1] - 2026-03-28

### Fixed
- **Save Overwrite Protection** (`hdr/io.py`, `nodes_io.py`): `Radiance Save EXR/HDR` and `Radiance Write` now default to `start_frame=0`, enabling automatic index detection to prevent accidental file overwrites.
- **Null Safety**: Added robust input validation to sequence loading and saving methods to prevent crashes when passed empty paths or invalid indices from third-party nodes.

### Changed
- **Branding Consistency**: `RadianceApplyGradeInfo` display name updated to `◎ Radiance Apply Grade Info` to match the suite's naming convention.
- **Repository Cleanup**: Removed legacy test scripts and temporary debug directories (`MagicMock`) to streamline the package for GitHub and Comfy Registry.

## [2.2] - 2026-03-18

### Changed
- Major version update to 2.2.
- Refined terminal and viewer UI.
- Improved console noise filtering and error stabilization.

## [2.1.1] - 2026-03-17

### Changed
- Maintenance update to version 2.1.1.

## [2.1.0] - 2026-03-10

### Added — Radiance Viewer v2.2 (Terminal & UX Overhaul)
- **TERMINAL HUD Tab** (`radiance_viewer.js`): Live Python REPL embedded directly in the viewer.
  - Persistent namespace (`_TERMINAL_NS`) — variables survive between executions.
  - 30-second timeout guard via `threading.Thread` to prevent ComfyUI event-loop freeze.
  - Pre-injected context: `math`, `os`, `torch`, `np`, `json`, `time`, `folder_paths`.
  - `Reset Namespace` button wipes state on demand.
  - Snippet dropdown with common helper presets.
- **Documentation Links** in Radiance Workspace node (two new buttons: `📖 Docs — www.fxtdstudios.com` and `🌐 FXTD Studios — www.fxtdstudios.com`) and in the Terminal tab status bar.

### Changed — Radiance Viewer UX (5 Design Improvements)
- **Toolbar Group Labels**: 9 labeled clusters (`FILE · GRADE · VIEW · CH · NAV · ANALYSIS · COMPARE · SCOPES · ANNOTATE · MEASURE`) with hairline separator rules.
- **Panel Label Typography**: All HUD sub-labels bumped to `11px` + `letter-spacing: 0.06em` for improved legibility.
- **Bottom Dock Colors**: `TERMINAL` tab now renders in `#00a8ff` (brand blue); `SCRIPT EDITOR` in `#7a92b0` (muted slate). `RUN AUTOMATION` button color unified to match.
- **Film Stock Preset Active State**: Clicking a film stock pill now shows a blue-glow border indicator (`rgba(0,168,255,0.45)`) — selected state is always visible.
- **Right Panel Height**: `tabContentContainer` uses `flex:1 + overflow-y:auto` — grading controls fill the full panel height instead of leaving a generic black void below.

### Changed — Nodes
- **◎ Radiance Depth Map** (renamed from `◎ Depth Map Generator`): Display name updated in `nodes_depth.py` — existing workflows unaffected as the internal class name `RadianceDepthMapGenerator` is unchanged.
- **Output Path Consistency**: Widget parameter name unified to `output_path` across:
  - `RadianceWrite` node (`nodes_io.py`) — was `subfolder`.
  - `RadianceSaveEXR` node (`hdr/io.py`) — was `custom_path`.

### Fixed
- Added missing `import json` to `nodes_radiance_viewer.py`.
- Restored accidentally dropped `exportToCDL()` function declaration after Terminal injection.

### Added — Radiance v2.1.0 (The Professional Suite)
- **◎ Radiance 32-bit Denoise** (`nodes_denoise.py`): Edge-preserving bilateral filter for 32-bit float images.
- **◎ Radiance Reroute / Reroute+** (`nodes_layout.py`): Compact visual reroute nodes with auto-type detection and custom labels.
- **◎ Radiance Load Image** (`nodes_radiance_mask.py`): Enhanced image loader with integrated soft-brush mask editor and non-destructive companion mask storage.
- **◎ Show Text (Radiance)** (`nodes_text.py`): Utility node for displaying any data type (string, JSON, etc.) directly on the node UI.
- **Prompt Enhancer Integration**: `Cinematic Prompt Machine` now supports grammar-aware prompt enhancement (Natural, Descriptive, Cinematic styles).

### Improved
- **Metadata Management**: Improved EXR/PNG metadata handling across I/O nodes.
- **Dependency Validation**: enhanced `check_dependencies` in `__init__.py` for clearer installation guidance.

### Added — Radiance Viewer v2.1 (Viewer Overhaul)
- **fp32 Pick Buffer Sidecar** (`#5`): Each frame now saves a zlib-compressed fp32 `.rpick` sidecar (max 256px) for accurate scene-linear HDR color picking — true EV readout at cursor
- **OES_texture_half_float** (`#3`): Explicitly enables the WebGL `OES_texture_half_float` extension; falls back gracefully if unavailable. Half-float upload path now active by default
- **GPU Histogram** (`#6`): Press **H** in the viewer for a GPU-rendered 256-bin per-channel histogram with log scale for HDR images and a dotted white line at `x=1.0` to mark the SDR ceiling
- **LRU Frame Cache** (`#8`): Up to 8 GPU textures cached by frame ID. Scrubbing through frames no longer re-uploads data that is already in VRAM
- **Linear False Color / Zebra** (`#9`): `linearFalseColor` flag evaluates false-color and zebra thresholds in scene-linear space (pre-OETF) for accurate stop-level analysis
- **Display-P3 / HDR Monitor Detection** (`#10`): `RadianceWebGLRenderer.initDisplayP3()` detects P3 and Rec.2020 displays via CSS media queries and configures canvas `colorSpace` accordingly
- **CDL Export + Import** (`#7`): 💾 menu gains **Export CDL (Grade)** and **Import CDL (Grade)** — bidirectional ASC CDL v1.2 XML roundtrip recognized by Nuke, DaVinci Resolve, and OCIO pipelines

### Added — New Nodes (9 nodes)

#### Color
- **◎ Radiance Grade Match** (`nodes_grade.py`): Dedicated shot-to-shot LAB mean/std match grading. Connect `source` + `reference` → matched image + grade_info JSON
- **◎ Apply Grade Info** (`nodes_grade.py`): Replay a `grade_info` JSON string (from Radiance Grade) onto any new image with a `strength` blend slider. Closes the grade roundtrip
- **◎ Radiance LUT Bake** (`nodes_lut.py`): Generate a 33³ `.cube` LUT file from any Radiance Grade parameter set. Compatible with DaVinci Resolve, Nuke, Premiere Pro, and OCIO
- **◎ Radiance LUT Apply** (`nodes_lut.py`): Load and apply any external `.cube` LUT file via trilinear interpolation with `strength` blend control

#### Video / Temporal
- **◎ Radiance Temporal Smooth** (`nodes_temporal.py`): Per-pixel exponential moving average (EMA) across batch frames to remove inter-frame flicker and AI grain. Motion-aware masking preserves sharp moving areas
- **◎ Radiance Flicker Analyze** (`nodes_temporal.py`): Measures frame-to-frame luma delta and outputs a JSON flicker report (flicker index, max delta, per-frame means). Use before/after Temporal Smooth to benchmark improvement

#### Scopes / Analysis
- **◎ Radiance False Color** (`nodes_scopes.py`): Bake a 7-zone calibrated false-color exposure visualization as an IMAGE for headless/batch pipeline use. Zones: Crushed / Under / Dim / Correct / Over / Hot / Clipped. Configurable `is_linear`, `blend`, and `exposure_offset`

#### Compositing
- **◎ Radiance Blend Composite** (`nodes_overlay.py`): Two-layer compositor with 8 blend modes — Normal, Add, Screen, Multiply, Overlay, Soft Light, Difference, Divide. Optional `MASK` pin for per-pixel coverage. HDR-safe (no mid-pipeline clamping)

### Improved — Existing Nodes
- **◎ Radiance Grade** (`nodes_grade.py`): Added `reference_image` + `match_strength` optional inputs for auto LAB match grading. Added `preset_file` for loading external JSON preset libraries. `grade_info` output is now a full JSON dump of all 12 grade parameters (was plain text)
- **Sampler Pro v3.6**: Fixed BUG-28 (sigma schedule off-by-one causing negative step indices), BUG-33 (AYS Flux anchors), BUG-20 (deep-copy in guidance), and several noise-injection double-application bugs

### Fixed
- **Viewer Pick Buffer**: HDR color picker now reads true scene-linear fp32 values from `.rpick` sidecar instead of tonemapped 8-bit pixel values
- **Viewer Histogram**: Histogram scope now GPU-rendered in scene-linear space instead of display-encoded space
- **Pixel Loupe**: Enhanced HDR support and magnification controls
- **QC Analysis**: Tonemapping applied before defect detection for more accurate results

---

## [1.2.1] - 2026-02-10

### Changed
- **Code Organization** - Moved `defects.py` to `radiance/image/defects.py` for better modularity.
- **Cleanup** - Removed temporary dev files (`fix_indent.py`, `fix_indent_2.py`).
- **Dependencies** - Updated `nodes_qc.py` to import defects from the new location.

## [1.2.0] - 2026-02-09

### Added - Radiance Studio
- **RadianceQC Node (`RadianceQC`)**
  - Automated technical quality control for VFX workflows.
  - Analyzes Levels (Crushed Blacks/Clipped Whites), Gamut violations, and Noise floor.
  - Generates visual overlay (Red=Clipped, Blue=Crushed) and detailed text report.
- **Project Management System**
  - **Project Settings Node (`RadianceProjectSettings`)**: Define root, sequence, shot, and version.
  - **Smart EXR Saver**: `RadianceSaveEXR` now accepts `RADIANCE_PROJECT` input to auto-generate paths (`Root/Seq/Shot/vXX`).
- **Nuke-Style Backdrops**
  - Right-click context menu "Radiance Studio > Create Backdrop".
  - Wraps selected nodes in a professional, color-coded group using industry standard colors.
  - "Auto-Align Nodes" command for quick organization.

## [1.1.1] - 2026-02-07

### Critical Fixes
- **HDR Clamping Fixes** - Removed unintended 8-bit clamping in CPU-based effects (`apply_bloom`, `apply_chromatic_aberration`, `RadianceProFilmEffects` sharpening).
- **EXR Saver Stability** - Fixed generic `SystemError` crash in OpenEXR writer; fallback mechanism now robustly handles writer failures.
- **EXR Absolute Paths** - `RadianceSaveEXR` now supports absolute paths in `subfolder` input (e.g. for external drive export).

## [1.1.0] - 2026-02-03

### Added
- **Temporal Grain Engine v2** - Frame-coherent grain for video with smooth transitions
  - `generate_temporal_grain()` function with prime decorrelation seed system
  - Per-channel R/G/B intensity controls in `FXTDTemporalGrain` node
  - Temporal smoothness blending between frames
  
- **LUT Engine Upgrade**
  - Tetrahedral interpolation for more accurate 3D LUT lookups
  - `interpolation` parameter (Trilinear/Tetrahedral) in `RadianceLUTApply`
  
- **Comprehensive Test Suite**
  - `test_comprehensive.py` with 18 tests covering 32-bit precision and sampler validation
  - 100% pass rate on all tests

### Fixed
- Phase-Shift sampler override now correctly uses selected sampler
- Dynamic Guidance math corrected for Low → High → Low profile
- **Canon Log3** - Fixed coefficients to match Canon specification (18% gray → 0.343)
- **Panasonic V-Log** - Fixed decode coefficient typo for proper roundtrip precision

### Security & Production Hardening
- **Thread-safe caching** - Added `threading.RLock()` to LUT, depth model, and processor caches
- **Exception handling** - Replaced 14 bare `except:` blocks with specific exception types + logging
- **Logging infrastructure** - Added `logging.getLogger()` to all modules for proper debugging
- **LUT cache eviction** - Bounded cache size (32 max) with FIFO eviction to prevent memory leaks

---

## [1.0.0] - 2026-01-15

### Initial Release
- 55 professional nodes for HDR processing
- Film grain with 30+ camera sensors, 20+ film stocks
- Industry-standard scopes (Histogram, Waveform, Vectorscope)
- GPU-accelerated processing
- EXR/HDR file support
- ACES 2.0 Output Transform
- LUT support with caching
- Radiance Pro Viewer with Nuke/Flame-style shortcuts
