<!-- project-mapper:generated -->
# VFX nodes and multipass estimation

ID: vfx. Snapshot: `7376f9e`. Coverage: partial. Every module in `nodes/vfx/`, `nodes/vfx/multipass/`, and `film/` was traced by a read-only agent. The mapper re-read the reachability of the DSINE path. Labels are explained in [START_HERE](../START_HERE.md#evidence-labels).

## Responsibility and boundaries

This subsystem covers pixel-space VFX: depth, flow, optics, matting, crop/stitch, plate tools,
mask propagation, and camera metadata. It also covers learned multipass estimation, relighting
and compositing, and multipass EXR read/write. It does not own colour management (see color) or
generic EXR IO (see io). `EXRPassesWriter` writes through `hdr/io.py`, not `io/writer.py`.

## Nodes by kind

| Node | Algorithm | Label |
| --- | --- | --- |
| `RadianceDepthMapGenerator` (`depth.py:171`) | Depth Anything V2 S/B/L through `transformers`, pinned commits. Each frame is z-scored, then the clip is min/max normalised to 0..1 | Implemented |
| `RadianceMultipassEstimate` (`nodes/vfx/multipass/estimate.py:371`) | MoGe-2 (ComfyUI `comfy.ldm.moge`) for geometry and Marigold IID v1.1 (diffusers) for materials and lighting. GTAO, curvature, a least-squares lighting fit, and DIS flow are computed classically | Implemented |
| `RadianceOpticalFlow` (`motion.py:13`) | OpenCV DIS, or pure-torch Lucas-Kanade when cv2 is missing | Implemented |
| `RadianceMotionBlur`, `LensDistortion`, `ChromaticAberration`, `AnamorphicStreaks`, `FilmGrain`, `Vignette` | Classical `grid_sample` or convolution | Implemented |
| `RadianceLinearMatting` (`masking.py:127`) | Guided filter plus trimap band. No learned matting | Implemented |
| (removed in 4.0) `RadianceSAMModelLoader`, `RadianceSAMGenerator` | Were hidden placeholders that raised; no SAM model ships | Removed |
| `RadianceHDRCrop` / `HDRStitch` / `TemporalStitchStabilizer` (`inpaint.py`) | Union bbox crop, then a feathered or Laplacian stitch, then a temporal Gaussian | Implemented |
| `RadianceHDRGrainMatcher`, `SubpixelStabilizer` (`plate.py`) | log2 high-pass grain transfer. FFT phase correlation, translation only | Implemented |
| `RadianceVideoMaskPropagator` (`mask_propagate.py:15`) | Warps keyframe masks along the flow. Fills empty frames only | Implemented |
| `RadianceCameraSync` (`camera.py:9`) | Reads JSON into `RADIANCE_CAMERA`. `.abc` raises | Implemented |
| `RadianceMultipassRelight` / `Composite` (`relight_comp.py`) | Lambert plus GGX relight, then an over composite with depth holdout. Takes **individual IMAGE passes**, not `RADIANCE_PASSES` | Implemented |
| `RadianceMultipassMaster` (`master.py:183`) | `DEPRECATED`. `extract` always raises. `_legacy_extract` is called only from `tests/test_multipass_contracts.py` | Implemented (mapper) |
| `RadianceEXRPassesWriter` (`master.py:508`) | `<prefix>.<frame:04d>.exr` from 1001, up to 8 threads. Data passes are promoted to 32-bit | Implemented |
| `RadianceMultipassAOVReader` (`aov_reader.py:272`) | One EXR frame mapped to `RADIANCE_PASSES` by layer aliases | Implemented |
| `RadianceDepthOfField`, `RollingShutter`, `CompressionArtifacts` (`film/camera.py`) | Classical. DoF expects depth with 0 = near, 1 = far | Implemented |

## Contracts

- **`RADIANCE_PASSES`** is a plain dict of `(B,H,W,3)` float tensors (Implemented, `estimate.py:489-659`):
  - `beauty`: scene-linear.
  - `depth`: metric camera-z in metres, larger means farther, 0 where there is no surface.
  - `world_position`: OpenGL camera space (x right, y up, -z forward).
  - `normal`: camera space, encoded 0..1, +Z toward the camera.
  - `motion_vector`: backward flow in pixels, stored `[u, -v, 0]`, so **+y up**.
  - Also: `ao`, `curvature`, `albedo`, `roughness`, `metallic`, diffuse/specular lighting, `alpha`, and `_`-prefixed metadata keys.
- **Conventions differ between nodes.** The code facts are Implemented. The effect on results is Inferred, and a test chaining the nodes would verify it.
  - `RadianceOpticalFlow` and `MaskPropagator` use **+y down**, so Estimate's `motion_vector` should not feed them directly.
  - Relight and Composite default to `depth_near_is_white=True` with 0..1 depth. Estimate depth is metric, far = larger. DoF wants 0 = near.
- **`RADIANCE_CAMERA`** has `focal_length`, `f_stop`, `shutter_angle`, `transform` (4x4), and frame fields.
  Its only consumer is `RadianceRelightEngine` (`nodes/hdr/synthesis.py:261-266`), which uses only the translation part.
- **No fps handling** anywhere in `nodes/vfx` (Implemented). Frame count is the batch size B.
- **Alpha:** only GrainMatcher uses `@alpha_passthrough`. LensDistortion, MotionBlur, Stabilizer, and Stitch probably warp alpha as if it were a colour channel (Inferred).
- **Chunking** (`core/tensor/chunking.py`) is used by most nodes. MaskPropagator, TemporalStitchStabilizer, and RollingShutter process the whole clip at once (Implemented).

## Models and downloads

| Model | Mechanism | Gate | Integrity | Label |
| --- | --- | --- | --- | --- |
| Depth Anything V2 | `from_pretrained` at a pinned commit, stored in the HF cache | `local_files_only = not downloads_allowed()` | HF revision pin | Implemented |
| MoGe-2 | `hf_hub_download` at a pinned revision into `models/geometry_estimation/` | widget AND `downloads_allowed()` | Size and SHA-256 (`MOGE_SHA256`, since 4.0); a mismatch deletes the download | Implemented |
| Marigold IID | `snapshot_download` at a pinned revision into `models/radiance/marigold/` | same | Each file checked against the Hub's LFS sha256 or git blob id at the pinned commit (4.0); a mismatch deletes it | Implemented |
| DSINE (legacy) | `torch.hub.load(..., trust_repo=True)` and `_download_model` | Partial | Unpinned | Implemented code. **Unreachable in production**: only `_legacy_extract` calls it, and only a test calls that (mapper) |

**Memory:** Depth Anything is cached in VRAM as fp16 on CUDA, which contradicts its docstring
("Cache stores models on CPU"). Marigold is moved back to CPU after a run, and Estimate calls
`_free_vram(device, 4.5e9)` first. Estimate has no OOM fallback. Only DoF and RollingShutter
retry on CPU. All Implemented.

## Important paths

The multipass workflow is traced in [WORKFLOWS W9](../WORKFLOWS.md#w9-multipass-estimate-relight-and-pass-export)
and drawn in [diagrams/flows/multipass.mmd](../diagrams/flows/multipass.mmd).

## Tests (inspected by agent, not run)

- `tests/test_vfx_nodes.py`, `test_vfx_efficiency.py`: chunked output equals one-pass output.
- `test_multipass_contracts.py`, `test_multipass_estimate.py`: uses a fake MoGe/IID.
- `test_optical_flow.py`, `test_optical_flow_dis.py`, `test_optics.py`, `test_exr_multipass_workflow.py`, `test_download_consent*.py`.
- Unknown: whether any test covers AnamorphicStreaks or Vignette numerics, the RollingShutter body, `RadianceRelightEngine` camera semantics, or the shipped workflow executing end to end.

## Open questions

See [OPEN_QUESTIONS](../OPEN_QUESTIONS.md): B24 (sign conventions), B17 (MoGe hash unused), D14 (cache docstring), D15 (LensDistortion "exact inverse").
