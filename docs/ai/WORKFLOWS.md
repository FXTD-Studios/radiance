<!-- project-mapper:generated -->
# Workflows

Snapshot: `7376f9e`. These are static traces from entry to output. None was executed.
Line numbers are approximate (`~`) and should be re-checked before editing.
Every step is **Implemented** (seen in code) unless marked *(Inferred)* or *(Unknown)*; labels are defined in
[START_HERE](START_HERE.md#evidence-labels). Diagrams are static-analysis maps, not runtime traces.

## W1. ComfyUI startup and node registration

Diagram: [diagrams/flows/startup.mmd](diagrams/flows/startup.mmd)

```
ComfyUI imports custom_nodes/radiance/__init__.py
 -> os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"; _bootstrap_package_context()
 -> imports config/constants, config/dependencies, config/env, core/logging
 -> from .nodes.registry import ...   (__init__.py:38)  runs nodes/__init__.py FIRST:
      for each NODE_GROUPS -> import nodes/<group>/__init__
           explicit imports + dict + fold_in_module_nodes
      + load_dynamic_gizmos()  (<repo>/gizmos/*.gizmo)
      + apply_radiance_branding (CATEGORY overwrite)
      Route modules register aiohttp routes here, at import time.
 -> setup_radiance_logging()                      core/logging.py (propagate=False)
 -> configure_runtime_environment()               config/env.py
 -> validate_runtime_dependencies(logger)         config/dependencies.py (return value ignored)
 -> _configure_ocio() -> color/ocio_setup.configure_ocio   (sets $OCIO if unset; never fatal)
 -> _load_comfyui_nodes(): load_node_mappings([.nodes required])   (cached module, no re-run)
 -> report_node_load_health()  ERROR if failures or < 152
 -> register_run_grouping()    PromptServer on_prompt hook
```

**Failure paths:**
- A group ImportError drops the whole group, logs a WARNING, and the health check logs an ERROR.
- An OCIO setup exception logs a WARNING.
- A failure to import `.nodes` itself is raised (it is `required`).
- Group-import WARNINGs fire before `setup_radiance_logging()` has configured the `radiance` logger, so their formatting and visibility depend on ComfyUI's defaults (inferred).

## W2. Load media: Read node

Diagram: [diagrams/flows/read-write.mmd](diagrams/flows/read-write.mmd)

```
RadianceRead.read                 nodes/io/write.py:~471
 -> _resolve_browse (ComfyUI input dir)
 -> io/reader.read_frames         :~1124  (colour contextvar)
 -> _read_resolved -> core/formats.classify
      image: _read_image (EXR->core/exr, DPX->OIIO, 16-bit->cv2, float TIFF->tifffile, else Pillow)
      video: _read_video -> core/video.probe (ffprobe) -> decode (ffmpeg rgb48le pipe)
      sequence: _resolve_sequence_paths -> threaded per-file reads (fps fixed 24.0)
 -> _apply_input_colorspace (OCIO or color/encodings) -> working space
 -> optional unpremultiply, proxy_scale; MASK from alpha or zeros
 -> (IMAGE, MASK, ..., info JSON)
```

**Failure paths:**
- An error raises `RuntimeError`, or returns an 8×8 black frame when `on_error="Black frame"`.
- If ffprobe is missing, every video read fails.
- A truncated decode raises `VideoTruncatedError`.

## W3. Write and export: Write node

Diagram: [diagrams/flows/read-write.mmd](diagrams/flows/read-write.mmd)

```
RadianceWrite.write              nodes/io/write.py:~855
 -> io/writer.write_frames       :1261  (resolve_output_path; _vNNNN; collision suffix)
 -> transform_stream (OutputColour encode) -> dispatch_write
      EXR (OpenEXR, metadata) | DPX (OIIO) | PNG/TIFF/JPEG/WEBP | video (ffmpeg stdin pipe)
 -> receipt {"ui": {radiance_files, radiance_manifest, images}, "result": (path, count)}
```

**Failure paths:**
- Writes are not atomic. A failed sequence frame leaves the earlier frames on disk.
- A non-zero ffmpeg exit leaves a partial file.
- Errors are logged and re-raised.

## W4. HDR generation round trip (starter workflow `workflows/start.json`)

Diagram: [diagrams/flows/hdr-generation.mmd](diagrams/flows/hdr-generation.mmd)

```
UnifiedLoader (loader_utils: detect arch, comfy.sd.load_*, LRU cache, optional pinned download)
 -> CinematicPromptEncoder (V3 node) -> CONDITIONING
 -> Resolution -> empty LATENT
 -> SamplerPro.sample (clone+patch model; comfy.sample.sample_custom)
 -> HDRVAEDecode (engine.py): verify_radiance_meta fails after sampling -> SDR-safe decode
      "Direct HDR" -> RadianceSDRToHDRUniversal (RUDRA; may download checkpoint)
 -> Viewer  (and DepthMapGenerator in start.json)
```

Encoding an HDR plate without sampling (`workflows/hdr_vae_encode_decode.json`): HDRVAEEncode
log-codes the plate and stamps `radiance_meta`. HDRVAEDecode then reconstructs scene-linear HDR
from the metadata.

## W5. Review in the Viewer

Diagram: [diagrams/flows/viewer-deliver.mmd](diagrams/flows/viewer-deliver.mmd)

```
RadianceViewer.view              nodes/monitor/viewer.py:383
 -> write temp files: .rhdr (fp16/fp32), 32-bit EXR, PNG preview (display_preview), audio
 -> fill viewer cache (cache.py, 2 GiB LRU)
 -> return {"ui": {...}}
JS onExecuted (radiance_viewer.js:~21863)
 -> fetch /view?type=temp -> Web Worker decode .rhdr -> WebGL texture (RGBA16F/32F)
 -> shader grade + view transform (+ OCIO GLSL from WASM when an OCIO view is active)
```

## W6. Deliver from the Viewer

Diagram: [diagrams/flows/viewer-deliver.mmd](diagrams/flows/viewer-deliver.mmd)

```
JS export dialog -> POST /radiance/deliver   delivery/handler.py:337
 -> validate format/colourspace vocabulary (400) and output dir containment (403)
 -> executor: _run_export
      frames from viewer cache -> color.grading.apply_grading -> optional FX/2x upscale
      -> QC + flicker scan -> io/writer.write_frames
      -> sidecars: thumb, CDL (io/formats.write_cdl_file), ACES AMF, _meta.json
      -> radiance_sessions.json (atomic, 500 cap)
 JS polls GET /radiance/progress
```

**Failure paths:**
- An FX or upscale failure becomes a warning, and the response status is `"partial"`.
- A sidecar failure is only logged.
- An exception returns 500 and sets progress to `"error"`.

## W7. Workspace: save and restore a workflow

```
radiance_workspace.js / workspace_dashboard.html
 -> POST /radiance/workflows/save   nodes/pipeline/workspace.py:~1073
      _resolve_safe_path under <repo>/workflows ; .rad (zip v3) + .rad.json ; .versions/ (max 50)
 -> GET /radiance/workflows/list | get | history ; POST restore | delete
```

## W8. Model download (consent-gated)

Diagram: [diagrams/flows/model-download.mmd](diagrams/flows/model-download.mmd)

```
ensure_model_exists (loader_utils:~89) -> name in RADIANCE_MODEL_MAP?
 -> core/consent.downloads_allowed (default True)
 -> core/model_fetch.fetch: require sha256 -> .part with Range resume -> verify -> os.replace
```

There is no cancellation path. Interrupting ComfyUI mid-download leaves a `.part` file, which is
resumed next time.

This is the `fetch` path. ACESConfigManager uses it since 4.0. Multipass (MoGe `hf_hub_download`,
checked against `MOGE_SHA256`; Marigold `snapshot_download`, checked against the Hub's hashes at the
pinned commit), Depth Anything (`from_pretrained`), SD-x4 (`from_pretrained`) and the RUDRA
`pixel_download` use their own mechanism. See OPEN_QUESTIONS B17.

## W9. Multipass estimate, relight, and pass export

Diagram: [diagrams/flows/multipass.mmd](diagrams/flows/multipass.mmd). Shipped graph: `workflows/multipass_relight.json`.

```
RadianceRead -> RadianceMultipassEstimate.estimate      nodes/vfx/multipass/estimate.py:371
   decode beauty to scene-linear
   load_moge -> ensure_moge (hf_hub_download, size check) -> infer per frame      :507-515
   _free_vram(4.5e9) -> load_marigold (snapshot_download) -> appearance, then lighting per frame   :575-613
   GTAO, curvature, lighting least-squares, DIS flow                              :158-648
   -> RADIANCE_PASSES dict + 13 IMAGE outputs + STRING
 -> RadianceEXRPassesWriter.write_passes                  master.py:508
      validate (beauty present, dims, no path in prefix, no lossy codec with data passes)
      -> <prefix>.<frame:04d>.exr from 1001, up to 8 threads, data passes promoted to 32-bit
      -> hdr/io.write_exr_openexr
 -> RadianceMultipassRelight (individual IMAGE passes) -> Viewer
```

**Failure paths:**
- Downloads off (widget, `RADIANCE_ALLOW_DOWNLOADS=0`, or `HF_HUB_OFFLINE=1`) raises `EstimateModelError`, naming the URL and destination.
- A ComfyUI without `comfy.ldm.moge`, or no diffusers, raises `EstimateModelError`.
- Validation failures raise `ValueError`. A write failure raises `RuntimeError`.
- There is no OOM fallback in Estimate.
- *(Inferred)* The shipped graph's Read output may be linear while Estimate's `beauty_encoding` is "sRGB (display)", which would decode the plate twice. Check RadianceRead's output for that widget value.

## W10. Video generation (T2V, I2V) and export

Diagram: [diagrams/flows/video.mmd](diagrams/flows/video.mmd).

```
UnifiedLoader / VideoLoader -> MODEL, CLIP, VAE
 -> T2VPipeline or I2VPipeline                         nodes/video/t2v.py:1095 / :1402
      _require_video_model (latent_dimensions >= 3)   :424
      _align_spec_to_model (dit.py _MODEL_SPECS + model/VAE reality)   :447
      noise latent [B,C,ceil(frames/tc),H,W]  (4D when a single latent frame)
      I2V: auto -> concat_channels (Wan-style extra input) | first_frame_lock
      _comfy_sample -> comfy.sample.sample_custom       :178
 -> VideoBatchDecode (5D to VAE; frames (T-1)*tc+1)    :1753
 -> optional VideoHDRDecode (PQ/HLG) | Write node (movie codecs) | VideoExport (EXR seq / GIF only)
Alternative: SamplerPro with temporal_window > 0 -> make_temporal_window_wrapper (experimental, off by default)
```

**Failure paths:**
- A non-video model raises.
- A NaN latent is replaced with 0 and a warning is logged.
- A failed preview decode returns a black placeholder, and the report says so.
- *(Inferred)* Noise is passed as both `noise` and `latent_image` (`t2v.py:999-1001`). This is probably correct for flow models but unverified for EPS models.
- Windowed sampling failed visual acceptance (`KNOWN_ISSUES.md`).

## W11. Send to Nuke or Resolve

Diagram: [diagrams/flows/dcc.mmd](diagrams/flows/dcc.mmd).

```
RadianceNukeSend.run                      nodes/pipeline/studio_integrations.py:147
 -> input_space conversion (_for_format)
 -> io/writer._save_exr to <nuke_folder>/<filename>[.NNNN].exr  (direct, overwrites)
 -> write <filename>.nk Read snippet
 -> if push_to_nuke: NukeConnector (RCMD + HMAC token) -> start_nuke_server.py inside Nuke
       -> load_exr action sets a Read node file knob
 -> status STRING (never raises)

RadianceDaVinciSend.run                   :339
 -> write TIFF16 / PNG8 / EXR to <resolve_folder>
 -> if import_to_media_pool: subprocess tools/resolve_import.py (JSON stdin, 30 s)
       -> DaVinciResolveScript -> MediaPool.ImportMedia
 -> status STRING
```

**Failure paths:**
- If Nuke isn't running, the status is `CONNECTION_REFUSED`, `TIMEOUT`, or `OS_ERROR`.
- **A read timeout after a successful send is reported as success** (`tools/nuke_connector.py:176-186`).
- For Resolve, a timeout, non-zero exit, or missing result line becomes a "not imported: …" message.

## W12. Upscale image or video

```
RadianceUpscaleImage / Tiler / Video     nodes/upscale/upscale.py
 -> _build_upscale_fn :1254  (UPSCALE_MODEL | SeedVR2/SD-x4 | spandrel | Real-ESRGAN | bicubic)
 -> _hdr_to_sr_domain (Reinhard) -> tiled_upscale :564 (gaussian feather / linear) -> _hdr_from_sr_domain
 -> Video: overlapping windows, LK-aligned overlap frames, Laplacian blend, CPU output accumulator
 -> _backend_report says which tier actually ran
```

**Failure paths:**
- A missing model or download refusal falls to the next tier, ending at bicubic, and the label says so.
- `RadianceAIUpscale` (`image/upscale.py`) returns bicubic on **any** exception.

Not traced end to end: audio cut and transcribe, scene-cut split, gizmo execution inside a
running graph, NDI streaming, and the realtime preview server lifecycle. Their behaviour is
summarised in the subsystem notes.
