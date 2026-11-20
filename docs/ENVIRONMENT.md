# Environment variables

Radiance reads these when ComfyUI starts. Set them in the shell, service or
launcher that starts ComfyUI, then restart it. None is needed for a normal
install: every one has a working default.

## Models and downloads

| Variable | Default | Effect |
| --- | --- | --- |
| `RADIANCE_ALLOW_DOWNLOADS` | on | `0` turns off every model download; the Models page then lists files to copy by hand. |
| `RADIANCE_LOADER_OFFLINE` | off | `1` stops the generation loaders downloading (kept for older setups; prefer `RADIANCE_ALLOW_DOWNLOADS=0`). |
| `RADIANCE_UPSCALE_OFFLINE` | off | `1` stops the upscale nodes downloading (kept for older setups). |
| `HF_TOKEN` | none | Hugging Face token for gated models. `hf auth login` stores one instead; Radiance only checks that one exists. |
| `RADIANCE_RUDRA_ACES2` | `models/radiance/RUDRA_ACES_2` | Folder of the RUDRA ACES 2 package, when it lives elsewhere. |
| `RADIANCE_RUDRA_GENFILL` | `models/radiance/` | Folder of the RUDRA AI fill v3 package. |
| `RADIANCE_RUDRA_GENFILL_V4` | `models/radiance/` | Folder or file of the experimental AI fill v4 LoRA. |
| `RADIANCE_WAN_VACE` | `models/radiance/` | Folder of Wan2.1-VACE-1.3B (diffusers) for the AI fill. |
| `RADIANCE_SDR2HDR_PIXEL` | `models/radiance/` | Checkpoint of the per-pixel SDR → HDR model. |
| `RADIANCE_TEMPORAL_RUDRA` | `models/radiance/` | Checkpoint of the temporal RUDRA model. |

## Files, folders and the server

| Variable | Default | Effect |
| --- | --- | --- |
| `OCIO` | built-in ACES 2.0 studio config | Your OCIO config. Radiance uses it when it loads; the Models page warns when it has no ACES 2.0 views. |
| `RADIANCE_READ_ROOTS` | ComfyUI's input, output, temp and models folders | Extra folders (separated by `;` on Windows, `:` elsewhere) the Viewer and Read routes may open, such as a NAS with plates. |
| `RADIANCE_UPLOAD_MAX_MB` | `4096` | Largest file the Assets page accepts, in MB. |
| `RADIANCE_VIEWER_CACHE_BYTES` | 2 GiB | System memory the Viewer's frame cache may hold (frames kept for Deliver). |
| `RADIANCE_USER_DATA_IN_PACKAGE` | off | `1` keeps workflows and gizmos inside the package folder (the 3.x layout) instead of `ComfyUI/user/radiance`. |
| `RADIANCE_FFMPEG` | ffmpeg on PATH, then the imageio-ffmpeg build | Full path of the ffmpeg to use; when set, only this one is used. |
| `RADIANCE_FFPROBE` | ffprobe on PATH, then next to ffmpeg | Full path of the ffprobe to use. |

## DCC bridges (Nuke, MCP)

| Variable | Default | Effect |
| --- | --- | --- |
| `RADIANCE_NUKE_HOST` / `RADIANCE_NUKE_PORT` | `127.0.0.1` / `1986` | Where Radiance finds Nuke's command server. |
| `RADIANCE_NUKE_BIND_HOST` | `127.0.0.1` | Address the Nuke-side server listens on. Use `0.0.0.0` only when Nuke runs on another machine. |
| `RADIANCE_COMFY_URL` | `http://127.0.0.1:8188` | ComfyUI's address, for the bridges that call back into it. |
| `RADIANCE_ALLOW_REMOTE_BRIDGE` | off | `1` lets the DCC bridge listen on a non-local address. |
| `RADIANCE_DCC_AUTH_TOKEN` | `~/.radiance/dcc_token` | Shared secret the DCC clients sign with. Prefer the token file; a variable can show up in process listings. |

## Logging and behaviour

| Variable | Default | Effect |
| --- | --- | --- |
| `RADIANCE_LOG_LEVEL` | `INFO` | `DEBUG` prints tracebacks for nodes that fail to load. |
| `RADIANCE_LOG_THEME` | `pro` | Console style: `pro`, `minimalist`, `classic`, `compact`, `cyberpunk`, `matrix`. |
| `RADIANCE_PROFILE` | off | `1` logs timings (`[Radiance PERF]`) for the nodes that measure themselves. |
| `RADIANCE_STRICT` | off | `1` makes the prompt nodes stop when their preset lists fail validation, instead of logging it. |

## Development only

`RADIANCE_UPDATE_DOCS`, `RADIANCE_UPDATE_SNAPSHOT`, `RADIANCE_DISABLE_TORCH_GATE`,
`RADIANCE_DEFAULT_YEAR` and `RADIANCE_RUDRA_ACES2_GOLDEN_PACKAGE` are read by the
tests and the documentation tools; see [DEVELOPMENT.md](DEVELOPMENT.md).
