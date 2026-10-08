<!-- project-mapper:generated -->
# HDR VAE engine and log profiles

ID: hdr-vae. Snapshot: `7376f9e`. Coverage: partial (targeted reads of `hdr/vae.py`, 3,597 lines). Labels: see [START_HERE](../START_HERE.md#evidence-labels).

## Responsibility and boundaries

Moves scene-linear HDR pixels through a standard ComfyUI VAE by log-coding them, and
decodes back with metadata-driven reconstruction. Does not own model loading
(that is generate) or sampling.

## Implementation index

| Source/symbol | Role | Label |
| --- | --- | --- |
| `hdr/vae.py:EXTENDED_LOG_SPACES` (~:200) | Six log profiles: LogC3, LogC4, S-Log3, V-Log, DaVinci Intermediate, Log3G10 | Implemented |
| `hdr/vae.py:LOG_PROFILE_HDR_PARAMS` (~:236), `DECODE_NOISE_SCALE_PER_PROFILE` (~:335) | Per-profile shoulder, denoise, and noise parameters | Implemented |
| `hdr/vae.py:LATENT_FORMAT_MAP` (:153) | Latent channel count to format label | Implemented; disagrees with `model/detect.py` |
| `hdr/vae.py:detect_vae_factor`, `detect_latent_format` | Spatial factor and latent format from the VAE object | Implemented |
| `hdr/vae.py:RadianceVAE4KEncode._prepare_for_vae` (~:1255) | Linearise, expose, clamp the floor, log-encode (the source's own curve, or LogC4) | Implemented |
| `hdr/vae.py:TileEngine` (~:869) | Tiled encode/decode with cosine blend | Implemented |
| `hdr/vae.py:RadianceVAE4KDecode.decode` (~:2733), `_vae_output_to_target` (~:2283) | Shoulder, highlight denoise, inverse curve, optional display tonemap, target space | Implemented |
| `hdr/vae.py:_save_rhdr` (~:2666) | Optional `.rhdr` sidecar, written through `core/rhdr.py` (zlib level 6) | Implemented; `safe_join` guard restored and fp16 clamped in the fix pass |
| `hdr/decode_meta.py:verify_radiance_meta` (~:116), `LOG_SPACE_GAMUT` | Latent metadata contract and fingerprint | Implemented |
| `nodes/generate/engine.py` (:133, :186, `apply` ~:331) | `RadianceHDRVAEEncode` / `RadianceHDRVAEDecode` node wrappers. Auto routing and the "Direct HDR" path | Implemented |

## Contracts

- **`hdr_mode` values:** `Clip (SDR)`, `Soft Clip` (tanh knee 0.85), `Compress (Log)`, and
  `Passthrough` (sRGB, clamped to [-0.05, 1.5]).
- **Encode output:** `(LATENT, alpha IMAGE (B,H,W,1), metadata, latent_format, quality_report)`.
  The latent dict carries `radiance_meta` = {pad, vae_factor, latent_format, source_space,
  hdr_mode, working_gamut, latent_fingerprint}.
- **Decode:** if the fingerprint doesn't match (the latent went through a sampler), the HDR keys
  are dropped. Live metadata overrides the widgets. With no metadata, Log and SoftClip fall back
  to Clip unless `force_hdr_decode` is set. `hdr_output=True` keeps values above 1.
- **3D (video) VAEs** are detected by `latent_dim == 3`. Temporal compression comes from
  `temporal_compression_decode()`.

## Important paths

- **Encode, then sample, then decode:** the fingerprint fails, so the node takes the SDR-safe
  decode. "Direct HDR" then rebuilds highlights through `RadianceSDRToHDRUniversal`, which can
  trigger the RUDRA checkpoint download (`engine.py:_pixel_hdr`).
- **Round trip with no sampler:** the metadata survives and HDR is reconstructed from the log curve.

## Tests

About 25 files: `tests/test_hdr_vae_*`, `test_vae_*`, `test_sdr_to_hdr*`, `test_pixel_*`,
and `test_integration_hdr_pipeline.py`. `test_hdr_vae_sdr2hdr_audit.py` skips without a local
`.pt` weight file. Not run.

## Open questions

- `LATENT_FORMAT_MAP` labels (`ltx_128ch` for Flux.2 Klein, `cascade_32ch`, `sd3_8ch`) don't
  match the labels in `model/detect.py` (`sd3_16ch`, …), even though a comment at :158 says they
  match. It's unknown which consumers compare these strings.
- `hdr/vae.py` still has its own gamut matrices (~:350-384) that duplicate `color/matrices`.
