# Radiance 4.0 beta: hardware acceptance

The last step before 4.0 goes to `main`. Everything here needs a GPU, a real
ComfyUI install, Nuke, Resolve or an HDR display, so CI cannot run it. The
automated suites (pytest, both lanes, ruff, JS) are green on `version-4-Beta`.

Run on the workstation (RTX 4080 SUPER, Windows), ComfyUI with this branch in
`custom_nodes/radiance`. Tick each box, and note the prompt id or file for
anything that fails.

## 0. Install and load

- [ ] ComfyUI starts; the log ends `Radiance: successfully loaded 152 nodes (v4.0.0)` and no
      missing-dependency warning (or one that names a package you expect to be missing).
- [ ] A saved 3.x graph loads. One that used SAM Loader, SAM Mask Generator, HDR Latent
      Encoder or HDR Turbo Encoder shows those nodes missing and nothing else broken.
- [ ] `OCIO` unset: the log names the built-in ACES studio config. Set `OCIO` to a broken
      file: Read, Write, HDR OCIO nodes and the ACES Config Manager's Detect all fall back
      to the same config.

## 1. Generation (GPU)

- [ ] **CFG++ (Perpendicular)** on SDXL and Flux at cfg 5-7, same seed as Standard: less
      saturation and burn at high cfg, no artefacts. Repeat with `refiner_model` connected.
- [ ] **Sequential offload**: still switches ComfyUI to LOW_VRAM for the session
      (known limit, P2-4). Confirm the tooltip warning matches what happens.
- [ ] **Video, unwindowed** (`temporal_window=0`) at 1280x704, 121 frames, 20 steps:
      coherent next to stock KSampler. Windowed video stays experimental.
- [ ] **8x upscale on CUDA** (Image, Tiler, Video) with Real-ESRGAN: geometry right,
      no seams (FIX-011/012).

## 2. Image processing

- [ ] **Denoise `motion_compensation`** with `temporal_blend` > 0 on a panning plate:
      no ghosting on moving edges, versus off. Note the time per frame at 1080p and 4K.
- [ ] **Bit Depth Degrade `restore_from_quantized`** on an 8-bit banded gradient:
      bands smoothed, no new detail invented.
- [ ] **RGBA plate** (premultiplied EXR with alpha) through Upscale Tier 1, Tier 2 and an
      external `UPSCALE_MODEL`, Face Restore, Focus Peaking, Split View, Contact Sheet,
      Flipbook GIF and Preview Server: none raise, alpha survives where the changelog says.
- [ ] **First run after upgrading** hashes each installed upscale and face model once
      (log: "checking ... against its pinned SHA-256"); the second run does not.
      A `.radiance-sha256` file sits next to each. A model of your own saved under a
      registry name is moved to `<name>.radiance-mismatch`, not overwritten.

## 3. Read and Write

- [ ] **HDR10**: Write PQ to "MP4 (H.265 10-bit)". `ffprobe -show_frames` (or MediaInfo)
      shows mastering display P3-D65 1000 nits and MaxCLL / MaxFALL. Plays as HDR on an
      HDR display. PQ to "MP4 (H.264)" stops with the 10-bit error and writes nothing.
- [ ] **ffmpeg only**: with ffprobe off PATH (imageio-ffmpeg only), Read loads an H.264
      and a ProRes 4444 clip with the right size, fps and alpha.
- [ ] **Sequences**: a DPX or EXR sequence written at 25 fps reports 25; a PNG sequence
      reports 24 as the default. Cineon (`.cin`) reads through OpenImageIO.
- [ ] **`raw` on video**: the picture is the file's code values, not decoded.
- [ ] **Delivery** into a folder holding 3.x masters (`Shot_v02_v0001.mov`): the next
      file is `Shot_v0003.mov`, with one suffix.
- [ ] **Audio**: a video from a stream of unknown length with short audio keeps every frame.

## 4. Downloads (empty models folders)

- [ ] Multipass Estimate downloads MoGe-2 and Marigold, verifies them, and runs.
      Pull the network during the Marigold check: the next run checks again.
- [ ] ACES Config Manager "Download ACES 2.0" installs a config that verifies.
- [ ] Audio Transcribe with `RADIANCE_ALLOW_DOWNLOADS=0` and Whisper `large` cached as
      `large-v3.pt`: runs. With nothing cached: refuses with the consent message.
- [ ] Depth Anything V2 and DSINE still download from `main` (not pinned, known limit);
      note the logged digests so they can be pinned.

## 5. DCC

- [ ] **Push to Nuke** (Nuke 17.1, listener from this branch's
      `scripts/start_nuke_server.py`): Read node created, scene-linear 0.18 / 1.0 / 4.0
      preserved. Stop the listener mid-push, and push while Nuke is busy for over 10 s
      (a long render): both read `UNCONFIRMED`, not `OK` or `FAILED`.
- [ ] **CDL** `.cdl`, `.cc`, `.ccc` from CDL Export load in Nuke (OCIOCDLTransform) and
      Resolve with the same grade (FIX-006).
- [ ] **Resolve import** of one EXR still and an EXR sequence (Media Pool frame counts).
- [ ] **DCC Bridge, Export Frames, source Sequence**: writes EXRs (it always failed before).
- [ ] **DCC Bridge, remote** (`RADIANCE_ALLOW_REMOTE_BRIDGE=1`, bind 0.0.0.0) from a
      second machine holding the same `~/.radiance/dcc_token`:

      ```python
      import json, socket
      from radiance.core.dcc_auth import sign_queue
      s = socket.create_connection(("WORKSTATION", 1987))
      s.sendall((json.dumps(sign_queue(prompt)) + "\n").encode())
      print(s.recv(65536))
      ```

      A signed prompt queues; an unsigned one, a replayed one and a browser `fetch()` to
      the port are refused.

## 6. Viewer

- [ ] Playback at 24 fps and above on the RTX 4080 (not measured yet), 1080p and
      4K sequences, with the Scopes tab open and closed.
- [ ] Scopes on the GPU: a frame with a few clipped speculars shows them at the
      100% line on the waveform; the sidebar keeps up while scrubbing (note the
      update time at 1080p and 4K from the console if it falls back to 1024 px).
- [ ] Two viewers in one graph: each keeps its own panel; deleting one leaves
      the other working; memory in Task Manager drops back after deleting.
- [ ] A video file (H.264 and ProRes) loaded into the viewer: arrows, ‹ › and
      the scrubber move the picture frame by frame; the frame rate is read.
- [ ] Grade, then reload the workflow: the grade comes back. Export .cube and
      .cdl: they match the screen in Resolve and Nuke (OCIOFileTransform).
- [ ] Deliver a graded master: it matches the viewer.
- [ ] Restart ComfyUI and reopen a workflow: the viewer says its files are
      gone instead of showing a blank canvas.
- [ ] With WebGPU chosen (Settings (⚙) > Renderer, then reload), the Masks tab opens
      with the notice that names that control; back on WebGL it works.

## Sign-off

When every box is ticked or each failure has an issue, merge `version-4-Beta` to `main`
(squash merge recommended), set the CHANGELOG date and tag `v4.0.0`.
