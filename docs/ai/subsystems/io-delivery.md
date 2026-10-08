<!-- project-mapper:generated -->
# Media IO and delivery

ID: io (plus delivery). Snapshot: `7376f9e`. Coverage: partial (static). Labels: see [START_HERE](../START_HERE.md#evidence-labels).

## Responsibility and boundaries

- `io/reader.py` and `io/writer.py` are the engines. They don't depend on the node layer.
- `nodes/io/write.py` holds the node classes (Read, Write, EXRMultiPart, DigitalCinemaRead/Write),
  `browse` resolution, and the `/radiance/media/*` routes. It re-exports the engines' private
  helpers under their legacy names.
- `delivery/handler.py` is the viewer's export endpoint.
- Not owned here: colour maths (color), and the PQ/HLG tensor encode (`nodes/hdr/delivery.py`,
  which writes no files).

## Implementation index

| Source/symbol | Role | Label |
| --- | --- | --- |
| `nodes/io/write.py:RadianceRead.read` (~:471) | Resolves `browse` against the ComfyUI input dir, then calls `read_frames` | Implemented |
| `io/reader.py:read_frames` (~:1124), `_read_resolved` (~:1223) | Classify the path, read, optional unpremultiply, proxy scale. Errors become `RuntimeError("RadianceRead failed ...")` unless `on_error="Black frame"` | Implemented |
| `io/reader.py:_read_image` (~:342) | `.exr` uses `core/exr`; `.hdr` uses cv2; `.dpx` uses OIIO; 16-bit PNG/TIFF use cv2; float TIFF uses tifffile; everything else uses Pillow (8-bit) | Implemented |
| `core/exr.py` | EXR probe order: OpenEXR 3 `File` (multipart), then legacy `InputFile`, then OIIO. Layer pick (`_pick_layer`); data window conformed to display window | Implemented |
| `core/video.py:decode` (~:462), `probe` (~:295) | ffprobe JSON, then one ffmpeg rawvideo pipe at rgb48le/rgba64le. Frame-accurate `select=` filter. fps is a `Fraction` | Implemented |
| `core/ffmpeg.py` | Binary lookup: `RADIANCE_FFMPEG`, then PATH, then imageio-ffmpeg. `ffmpeg_with_encoder` | Implemented |
| `io/reader.py` sequence path (`_resolve_sequence_paths` ~:511) | `%0Nd`, `####`, glob, or directory. Up to 8 decode threads. **fps is hard-coded to 24.0** (~:769) | Implemented |
| `nodes/io/write.py:RadianceWrite.write` (~:855) | Calls `io/writer.write_frames` | Implemented |
| `io/writer.py:write_frames` (:1261), `dispatch_write` (~:1441) | Colour out (`OutputColour`), then the per-format writer. Versioning `_vNNNN` (`version` defaults to 1; Delivery passes its counter since 4.0, one suffix). A stream of unknown length pads short audio (`apad` + `-shortest`) | Implemented |
| `io/writer.py:_save_exr` (~:597) | OpenEXR `File` write with workflow/prompt/colourspace/chromaticities metadata. The cv2 fallback drops metadata | Implemented |
| `io/writer.py:_save_video_ffmpeg` (~:847) | H.264, H.265 10-bit, ProRes 422 HQ/4444, DNxHR HQ (`.mov`). Writes trc/primaries/matrix tags. **No HDR10 mastering/MaxCLL metadata** | Implemented |
| `delivery/handler.py:radiance_deliver_endpoint` (:337) | `POST /radiance/deliver`: viewer cache, then grade, optional FX/upscale, QC, `write_frames`, then sidecars (thumb, CDL, AMF, `_meta.json`) | Implemented |
| `core/errors.py` | `RadianceError` hierarchy and `handle_node_errors`. **Not used by production code** | Implemented |

## Contracts

- **Write paths:** relative paths are anchored under the ComfyUI output dir with `safe_join`, and
  `..` is rejected (`core/system/path_utils.resolve_output_path`). **Absolute paths pass through.**
  The Write node's default is `~/radiance_output`.
- **Delivery output** must be inside the ComfyUI output dir, otherwise 403. The check uses
  `abspath`/`relpath`, not `realpath`.
- **Write receipt:** `{"ui": {"radiance_files", "radiance_manifest", "images"}, "result": (path, count)}`.
- **No atomic media writes.** Partial video files are removed on timeout or exception, but not
  on a non-zero ffmpeg exit. The only atomic writes are the delivery session log and model
  downloads (`.part` then `os.replace`).
- **Error surface:** plain `RuntimeError`, `ValueError`, or `ImportError`, plus `EXRReadError`,
  `VideoDecodeError`, and `VideoTruncatedError`. HTTP routes return JSON errors.

## Tests

About 29 IO files (`tests/test_exr_*`, `test_io*`, `test_read_*`, `test_write*`, `test_video_*`,
`test_timecode*`, …) and about 10 delivery/DCC files. `test_video_frame_counts.py` is `slow` and
needs ffmpeg. Not run.

## Open questions

See OPEN_QUESTIONS: DNxHR extension mismatch, sequence fps,
missing ffprobe, `raw=True` on video, OIIO-only extensions, and fps written as a float.
