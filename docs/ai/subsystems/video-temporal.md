<!-- project-mapper:generated -->
# Video generation, temporal windowing, scene cut, and audio

ID: video (plus ai, audio). Snapshot: `7376f9e`. Coverage: partial. Agent trace of `nodes/video/*`, `nodes/ai/scene_cut.py`, `nodes/pipeline/audio.py`, the temporal parts of `temporal_rudra.py`, and the sampler's temporal window. The mapper re-read the `character.py` header and the audio key resolution. Labels are explained in [START_HERE](../START_HERE.md#evidence-labels).

## Responsibility and boundaries

This subsystem covers video-diffusion helpers and the T2V/I2V pipelines (`nodes/video/t2v.py`),
the latent spec table (`dit.py`), HDR prompt conditioning and video HDR decode (`nodes/video/hdr.py`),
long-video temporal windowing (in the generate sampler), scene-cut detection, and audio tools.
Model loading belongs to [generate](generate.md). Video file decode and encode belong to [io-delivery](io-delivery.md).

## Implementation index

| Source/symbol | Role | Label |
| --- | --- | --- |
| `nodes/video/t2v.py:_comfy_sample` (:178) | `sampler_object` + `calculate_sigmas` + `sample_custom`. NaN is replaced with 0 and a warning is logged | Implemented |
| `t2v.py:_require_video_model` (:424), `_align_spec_to_model` (:447) | Rejects non-3D latents. The model and VAE's own channels and compression override the preset | Implemented |
| `VideoModelInfo`, `VideoLatentNoise`, `VideoCondMerge`, `VideoSampler`, `T2VPipeline`, `I2VPipeline`, `VideoBatchDecode`, `VideoExport` | Pipeline nodes. See the agent trace for each node's I/O | Implemented |
| `I2VPipeline` strategies | `auto` picks `concat_channels` (Wan-style, detected from the model's extra input channels) or `first_frame_lock`. Also `clip_vision_inject` and `prepend_latent`. Dispatch is by **capability, not model name**, despite the header at t2v.py:26-33 | Implemented |
| `VideoExport` (:1936) | Writes EXR frames via `hdr/io.write_exr_robust` and a GIF via PIL. **No movie codecs.** Does not use `io/writer` | Implemented |
| `nodes/video/dit.py:_MODEL_SPECS` (:46) | Latent spec table: LTX 128ch /32 /8; Hunyuan; Wan 2.1/2.2 (MoE flag); TI2V 48ch; CogVideoX; Mochi | Implemented |
| `nodes/video/hdr.py` | `VideoHDRConditioner` (adds descriptor tokens and `radiance_hdr` metadata that no model reads), `VideoHDRDecode` (gamma 2.2, then gamut, then Reinhard, then PQ/HLG), `FrameRouter`, `Assembler` (class-level `_STORE`, in memory), `PromptBuilder` | Implemented |
| `nodes/video/character.py` | Helpers only. The header lists `RadianceCharacterAnchor`, Checker, and Gallery nodes that **do not exist**. `.npz` profiles are loaded with `allow_pickle=True` | Implemented (mapper re-read header) |
| `nodes/generate/sampler.py:make_temporal_window_wrapper` (:438) | UNet wrapper that blends windows at every step. 5D latents only. Raises if ControlNet is present. `temporal_window` defaults to 0 (off) | Implemented |
| `sampler_utils.py:plan_temporal_windows` (:2045), `temporal_window_weights` (:2087), `slice_conds_temporally` (:2128) | Window planner, sin²/cos² complementary fades, cond slicing | Implemented |
| `nodes/ai/scene_cut.py` | Histogram L1, plus a finite-difference edge measure (the tooltip says "Sobel"), combined 0.6/0.4. Cut when score ≥ 0.5 and at least 12 frames since the last cut | Implemented |
| `nodes/pipeline/audio.py` | `AudioCut`: librosa, then scipy, then ffmpeg `silencedetect`. `AudioTranscribe`: local whisper, the OpenAI API (uploads the audio), or the whisper CLI | Implemented |

## Contracts

- **Frames** are an IMAGE batch `[N,H,W,C]`. Video latents are `[B,C,T,H,W]`. A one-frame latent becomes **4D** (`t2v.py:169-171`). Implemented.
- **Frame count:** the noise latent has `ceil(frames/tc)` frames, and decode expects `(T-1)·tc+1`. A request that isn't `k·tc+1` comes back shorter (Inferred; for example 24 frames returns 21). `Resolution` validates `stride·k+1` (`resolution.py:1235-1255`).
- **fps** is a `Fraction` in `core/video.VideoInfo`, a float at the Read node output (exact value in `metadata_json.fps_exact`), and a FLOAT widget wherever a node needs it. **No video node carries fps between nodes.** Implemented.
- **Long-video windowing is experimental and off by default.** `KNOWN_ISSUES.md` (line 4 and the 2026-09-26 section) records failed visual acceptance: join ghosting and doubled objects. Implemented (doc quote) and code location identified.
- **AudioTranscribe** gets its key from `resolve_secret`, where the env var named by `openai_api_key_env` takes precedence over the widget. The whisper weight download (package and CLI) asks for download consent since 4.0; the OpenAI upload happens only when that backend is chosen, or on Auto with a key and no local whisper. Errors are returned as transcript text. Implemented.

## Important paths

T2V is traced in [WORKFLOWS W10](../WORKFLOWS.md#w10-video-generation-t2v-i2v-and-export) and drawn in [diagrams/flows/video.mmd](../diagrams/flows/video.mmd).

## Tests (inspected by agent, not run)

- `tests/test_video_batch_decode_5d.py`, `test_i2v_strategies.py`, `test_video_hdr_nodes.py`, `test_video_pipeline.py`.
- `test_long_video_windowing.py` (28 tests on partition of unity, coverage, and determinism), `test_temporal_chunking.py`, `test_vae_temporal_auto_size.py`, `test_temporal_rudra.py`.
- `test_scene_cut*.py`, `test_timecode.py`, `test_ndi_sender_batch.py`, `test_video_frame_counts.py` (slow, needs real ffmpeg).
- Unknown: tests for `VideoAssembler`, `FrameRouter`, or `PromptBuilder`.

## Open questions

See [OPEN_QUESTIONS](../OPEN_QUESTIONS.md): B25 (noise passed as the start latent), B26 (transcribe segment start), B18 (transcribe has no consent gate), D13 (phantom character nodes).
