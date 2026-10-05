<!-- project-mapper:generated -->
# Workflows

Snapshot: `7376f9e`. These are static traces from entry to output. None was executed.
Line numbers are approximate (`~`) and should be re-checked before editing.

## W1. ComfyUI startup and node registration

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
 -> report_node_load_health()  ERROR if failures or < 156
 -> register_run_grouping()    PromptServer on_prompt hook
```

**Failure paths:**
- A group ImportError drops the whole group, logs a WARNING, and the health check logs an ERROR.
- An OCIO setup exception logs a WARNING.
- A failure to import `.nodes` itself is raised (it is `required`).
- Group-import WARNINGs fire before `setup_radiance_logging()` has configured the `radiance` logger, so their formatting and visibility depend on ComfyUI's defaults (inferred).

## W2. Load media: Read node

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

```
ensure_model_exists (loader_utils:~89) -> name in RADIANCE_MODEL_MAP?
 -> core/consent.downloads_allowed (default True)
 -> core/model_fetch.fetch: require sha256 -> .part with Range resume -> verify -> os.replace
```

There is no cancellation path. Interrupting ComfyUI mid-download leaves a `.part` file, which is
resumed next time.

Not traced: VFX/multipass, upscale, video T2V/I2V, DCC send (Nuke/Resolve), audio, scene cut.
