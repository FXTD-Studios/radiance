import os
import glob
import json
import time
import logging
import datetime
import subprocess  # nosec B404
import tempfile
from typing import Dict, Any, Optional, List, Tuple

import torch
import numpy as np
import cv2

from . import color_utils
try:
    from .path_utils import safe_join, get_safe_output_dir, get_safe_input_path
except ImportError:
    from path_utils import safe_join, get_safe_output_dir, get_safe_input_path

import folder_paths

try:
    from .hdr.io import write_exr_robust, write_hdr_rgbe
except ImportError:
    try:
        from hdr.io import write_exr_robust, write_hdr_rgbe
    except ImportError:
        write_exr_robust = None
        write_hdr_rgbe = None


logger = logging.getLogger("◎ Radiance.io")


class RadianceType(str):
    def __ne__(self, __value: object) -> bool:
        return False

image_video_type = RadianceType("IMAGE,VIDEO")

# ═══════════════════════════════════════════════════════════════════════════════
#                         COLOR SPACE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

INPUT_COLORSPACES = [
    "Linear (sRGB)",
    "sRGB (Standard)",
    "ARRI LogC3",
    "ARRI LogC4",
    "Sony S-Log3",
    "Panasonic V-Log",
    "DaVinci Intermediate",
    "ACEScg",
    "ACEScct",
]


def apply_input_transform(img_tensor: torch.Tensor, colorspace: str) -> torch.Tensor:
    if colorspace == "Linear (sRGB)" or colorspace == "ACEScg": return img_tensor
    elif colorspace == "sRGB (Standard)": return color_utils.tensor_srgb_to_linear(img_tensor)
    
    device = img_tensor.device
    img_np = img_tensor.cpu().numpy()
    transform_map = {
        "ARRI LogC3": color_utils.logc3_to_linear,
        "ARRI LogC4": color_utils.logc4_to_linear,
        "Sony S-Log3": color_utils.slog3_to_linear,
        "Panasonic V-Log": color_utils.vlog_to_linear,
        "DaVinci Intermediate": color_utils.davinci_intermediate_to_linear,
        "ACEScct": color_utils.acescct_to_linear,
    }
    fn = transform_map.get(colorspace)
    out_np = fn(img_np) if fn else img_np
    return torch.from_numpy(out_np).to(device)


def apply_output_transform(img_tensor: torch.Tensor, colorspace: str, broadcast_safe: bool = True) -> torch.Tensor:
    is_display_space = colorspace in ["sRGB (Standard)"] or ("sRGB" in colorspace and "Linear" not in colorspace)
    if broadcast_safe and is_display_space:
        a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
        x = img_tensor
        img_tensor = torch.clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)

    if colorspace == "Linear (sRGB)" or colorspace == "ACEScg":
        return torch.clamp(img_tensor, 0.0, 1.0) if broadcast_safe else img_tensor
    elif colorspace == "sRGB (Standard)":
        return color_utils.tensor_linear_to_srgb(img_tensor)

    device = img_tensor.device
    img_np = img_tensor.cpu().numpy()
    transform_map = {
        "ARRI LogC3": color_utils.linear_to_logc3,
        "ARRI LogC4": color_utils.linear_to_logc4,
        "Sony S-Log3": color_utils.linear_to_slog3,
        "Panasonic V-Log": color_utils.linear_to_vlog,
        "DaVinci Intermediate": color_utils.linear_to_davinci_intermediate,
        "ACEScct": color_utils.linear_to_acescct,
    }
    fn = transform_map.get(colorspace)
    out_np = fn(img_np) if fn else img_np
    return torch.from_numpy(out_np).to(device)

# ═══════════════════════════════════════════════════════════════════════════════
#                         HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_audio_ffmpeg(video_path: str) -> dict | None:
    try:
        probe_cmd = ["ffprobe", "-v", "quiet", "-select_streams", "a:0", "-show_entries", "stream=codec_type,sample_rate,channels", "-of", "json", video_path]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0: return None
        probe_data = json.loads(result.stdout)
        streams = probe_data.get("streams", [])
        if not streams: return None
        sample_rate = int(streams[0].get("sample_rate", 44100))
        channels = int(streams[0].get("channels", 2))
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        extract_cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_f32le", "-ar", str(sample_rate), "-ac", str(channels), tmp_path]
        subprocess.run(extract_cmd, capture_output=True, timeout=120)
        if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) < 100: return None
        import struct
        with open(tmp_path, "rb") as f: data = f.read()
        os.unlink(tmp_path)
        data_offset = data.find(b"data")
        if data_offset == -1: return None
        data_offset += 4
        data_size = struct.unpack_from("<I", data, data_offset)[0]
        data_offset += 4
        raw_audio = data[data_offset : data_offset + data_size]
        n_samples = len(raw_audio) // 4
        samples = np.frombuffer(raw_audio, dtype=np.float32)
        samples_per_channel = n_samples // channels
        samples = samples[: samples_per_channel * channels]
        waveform = samples.reshape(samples_per_channel, channels).T
        waveform_tensor = torch.from_numpy(waveform.copy()).unsqueeze(0)
        return {"waveform": waveform_tensor, "sample_rate": sample_rate}
    except Exception: return None

def find_video_path(data: Any) -> Optional[str]:
    if isinstance(data, str):
        if data.lower().endswith((".mp4", ".mov", ".gif", ".webp", ".avi", ".mkv", ".webm")): return data
        return None
    if isinstance(data, dict):
        for key in ["source", "filename", "video", "gif", "full_path"]:
            if key in data:
                res = find_video_path(data[key])
                if res: return res
        for val in data.values():
            res = find_video_path(val)
            if res: return res
    if isinstance(data, list):
        for item in data:
            res = find_video_path(item)
            if res: return res
    return None

def _load_video_frames(video_path: str, input_colorspace: str = "sRGB (Standard)") -> torch.Tensor | None:
    if not os.path.exists(video_path): return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    if not frames: return None
    frames_np = np.stack(frames).astype(np.float32) / 255.0
    return apply_input_transform(torch.from_numpy(frames_np), input_colorspace)


def _unique_path(path: str) -> str:
    """Return a non-existing path by appending _0001, _0002, ... if needed."""
    if not os.path.exists(path):
        return path
    directory, filename = os.path.split(path)
    stem, ext = os.path.splitext(filename)
    idx = 1
    while True:
        candidate = os.path.join(directory, f"{stem}_{idx:04d}{ext}")
        if not os.path.exists(candidate):
            return candidate
        idx += 1


def _unique_dir(path: str) -> str:
    """Return a non-existing directory path by appending _0001, _0002, ... if needed."""
    if not os.path.exists(path):
        return path
    idx = 1
    while True:
        candidate = f"{path}_{idx:04d}"
        if not os.path.exists(candidate):
            return candidate
        idx += 1


def _history_outputs(path_or_dir: str) -> List[Dict[str, str]]:
    """Build Comfy history output entries for files written by Radiance."""
    if not path_or_dir:
        return []
    if os.path.isdir(path_or_dir):
        files = [
            os.path.join(path_or_dir, name)
            for name in sorted(os.listdir(path_or_dir))
            if os.path.isfile(os.path.join(path_or_dir, name))
        ]
    else:
        files = [path_or_dir] if os.path.isfile(path_or_dir) else []

    output_root = os.path.abspath(folder_paths.get_output_directory())
    entries = []
    for fpath in files:
        abs_path = os.path.abspath(fpath)
        try:
            rel = os.path.relpath(abs_path, output_root)
            if rel.startswith(".." + os.sep) or rel == "..":
                raise ValueError
            subfolder = os.path.dirname(rel).replace(os.sep, "/")
            filename = os.path.basename(rel)
            entries.append({"filename": filename, "subfolder": subfolder, "type": "output"})
        except ValueError:
            entries.append({"filename": os.path.basename(abs_path), "subfolder": "", "type": "output", "fullpath": abs_path})
    return entries

# ═══════════════════════════════════════════════════════════════════════════════
#                         NODE: DIGITAL CINEMA READ (UNIVERSAL)
# ═══════════════════════════════════════════════════════════════════════════════

class RadianceDigitalCinemaRead:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_path": ("STRING", {"default": "C:/Footage/shot_01.mp4"}),
                "start_frame": ("INT", {"default": 1, "min": 1}),
                "frame_limit": ("INT", {"default": 0, "min": 0, "tooltip": "0 = Load all"}),
                "input_colorspace": (INPUT_COLORSPACES, {"default": "sRGB (Standard)"}),
                "fps_override": ("FLOAT", {"default": 0.0, "min": 0.0, "tooltip": "0 = Auto (Detect from Video or 24.0 for Seq)"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT", "FLOAT", "AUDIO", "STRING")
    RETURN_NAMES = ("IMAGE", "MASK", "frame_count", "width", "height", "fps", "audio", "video")
    FUNCTION = "read"
    CATEGORY = "FXTD Studios/Radiance/IO"

    def _load_single_exr(self, file_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Robust EXR loader with multiple backends for high-bitrate cinema workflows."""
        img, alpha_img, depth_img = None, None, None
        try:
            import OpenEXR, Imath
            exr_file = OpenEXR.InputFile(file_path)
            header = exr_file.header()
            dw = header["dataWindow"]
            width, height = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
            channels = list(header["channels"].keys())
            pt = Imath.PixelType(Imath.PixelType.FLOAT)
            if "R" in channels and "G" in channels and "B" in channels:
                r = np.frombuffer(exr_file.channel("R", pt), dtype=np.float32).copy().reshape(height, width)
                g = np.frombuffer(exr_file.channel("G", pt), dtype=np.float32).copy().reshape(height, width)
                b = np.frombuffer(exr_file.channel("B", pt), dtype=np.float32).copy().reshape(height, width)
                alpha_img = np.frombuffer(exr_file.channel("A", pt), dtype=np.float32).copy().reshape(height, width) if "A" in channels else np.ones((height, width), dtype=np.float32)
                img = np.stack([r, g, b], axis=-1)
            elif "Y" in channels:
                y = np.frombuffer(exr_file.channel("Y", pt), dtype=np.float32).copy().reshape(height, width)
                img = np.stack([y, y, y], axis=-1)
                alpha_img = np.frombuffer(exr_file.channel("A", pt), dtype=np.float32).copy().reshape(height, width) if "A" in channels else np.ones((height, width), dtype=np.float32)
            
            for dname in ["Z", "Depth", "depth", "z"]:
                if dname in channels:
                    depth_img = np.frombuffer(exr_file.channel(dname, pt), dtype=np.float32).copy().reshape(height, width)
                    break
            exr_file.close()
            return img, alpha_img, depth_img
        except: pass

        try:
            import imageio.v3 as iio
            img = iio.imread(file_path)
            return np.asarray(img, dtype=np.float32), None, None
        except: pass
        return None, None, None

    def read(self, source_path, start_frame, frame_limit, input_colorspace, fps_override=0.0):
        if isinstance(source_path, (list, tuple)):
            source_path = source_path[0] if source_path else ""
        source_path = str(source_path or "").strip()
        if not source_path:
            raise ValueError("source_path is empty")
        source_path = get_safe_input_path(folder_paths.get_input_directory(), source_path, allow_absolute=True)
        
        is_video = source_path.lower().endswith((".mp4", ".mov", ".gif", ".webp", ".avi", ".mkv", ".webm"))
        
        if is_video:
            cap = cv2.VideoCapture(source_path)
            if not cap.isOpened(): raise IOError(f"Could not open: {source_path}")
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Smart FPS Detection
            detected_fps = cap.get(cv2.CAP_PROP_FPS)
            if detected_fps < 1.0 or detected_fps > 1000.0: detected_fps = 24.0
            fps = fps_override if fps_override > 0 else detected_fps
            
            cv2_start = max(0, start_frame - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, cv2_start)
            frames_to_read = frame_limit if frame_limit > 0 else (total_frames - cv2_start)
            frames_to_read = min(frames_to_read, total_frames - cv2_start)

            frames = []
            for _ in range(int(frames_to_read)):
                ret, frame = cap.read()
                if not ret: break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            
            frames_np = np.stack(frames).astype(np.float32) / 255.0
            images = apply_input_transform(torch.from_numpy(frames_np), input_colorspace)
            mask = torch.ones((len(frames), height, width), dtype=torch.float32)
            audio = _extract_audio_ffmpeg(source_path)
            return (images, mask, len(frames), width, height, float(fps), audio, source_path)
        else:
            if os.path.isdir(source_path): folder, pattern = source_path, "*"
            else: folder, pattern = os.path.dirname(source_path), os.path.basename(source_path)
            
            files = sorted(glob.glob(os.path.join(folder, pattern)))
            files = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg", ".exr", ".hdr", ".tiff", ".tif"))]
            
            startIndex = 0
            if start_frame > 1:
                # Try to find frame by number
                for idx, f in enumerate(files):
                    if str(start_frame).zfill(4) in os.path.basename(f) or str(start_frame) in os.path.basename(f):
                        startIndex = idx
                        break
            
            files = files[startIndex:]
            if frame_limit > 0: files = files[:frame_limit]
            
            images, masks = [], []
            img_w, img_h = 0, 0
            for fpath in files:
                ext = os.path.splitext(fpath)[1].lower()
                img = None
                mask_np = None
                if ext in [".exr", ".hdr"]:
                    try:
                        rgb_np, alpha_np, _ = self._load_single_exr(fpath)
                        if rgb_np is not None:
                             img = rgb_np
                             mask_np = alpha_np if alpha_np is not None else np.ones(img.shape[:2], dtype=np.float32)
                    except: pass
                
                if img is None:
                    img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
                    if img is None: continue
                    if img.shape[-1] == 4:
                        mask_np = img[..., 3].astype(np.float32) / (255.0 if img.dtype == np.uint8 else 65535.0)
                        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                    else:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        mask_np = np.ones(img.shape[:2], dtype=np.float32)

                if img.dtype == np.uint8: img = img.astype(np.float32) / 255.0
                elif img.dtype == np.uint16: img = img.astype(np.float32) / 65535.0
                else: img = img.astype(np.float32)

                if img_w == 0: img_h, img_w = img.shape[:2]
                images.append(torch.from_numpy(img))
                masks.append(torch.from_numpy(mask_np))
            
            if not images: raise ValueError(f"No valid images found in: {source_path}")
            
            # Finalize batch
            batch_img = apply_input_transform(torch.stack(images), input_colorspace)
            batch_mask = torch.stack(masks)

            # Smart Sequence FPS Detection
            seq_fps = 24.0
            fps_file = os.path.join(folder, "fps.txt")
            if os.path.exists(fps_file):
                try:
                    with open(fps_file, "r") as f:
                        seq_fps = float(f.read().strip().split()[0]) # Handle "24 fps" or just "24"
                except: pass
            
            fps = fps_override if fps_override > 0 else seq_fps
            return (batch_img, batch_mask, len(images), img_w, img_h, float(fps), None, source_path)

# ═══════════════════════════════════════════════════════════════════════════════
#                         NODE: DIGITAL CINEMA WRITE (UNIVERSAL)
# ═══════════════════════════════════════════════════════════════════════════════

WRITE_FORMATS = [
    "Image Sequence — EXR (32-bit)",
    "Image Sequence — Radiance HDR (.hdr)",
    "Image Sequence — PNG (16-bit)",
    "Image Sequence — PNG (8-bit)",
    "Image Sequence — JPEG",
    "Video — MP4 (H.264)",
    "Video — MP4 (H.265 10-bit)",
    "Video — MOV (ProRes 422 HQ)",
    "Video — MOV (ProRes 4444)",
    "Video — MOV (ProRes 4444 XQ)",
    "Video — MOV (ProRes 4444 HDR Log)",
    "GIF", "WEBP",
]

COMPRESSIONS = ["ZIP", "ZIPS", "PIZ", "RLE", "None", "PXR24", "B44", "B44A", "DWAA", "DWAB"]
BIT_DEPTHS = ["16-bit Half Float", "32-bit Float"]
ALPHA_MODES = ["None", "From Image", "Solid White", "Solid Black"]

class RadianceDigitalCinemaWrite:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (image_video_type,),
                "filename_prefix": ("STRING", {"default": "◎ Radiance"}),
                "write_mode": (["Video", "Sequence", "Single Image"], {"default": "Video"}),
                "output_format": (WRITE_FORMATS, {"default": "Video — MP4 (H.265 10-bit)"}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0}),
                "quality": ("INT", {"default": 10, "min": 0, "max": 100}),
                "output_color_space": (INPUT_COLORSPACES, {"default": "sRGB (Standard)"}),
                "broadcast_safe": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "output_path": ("STRING", {"default": ""}),
                "start_frame": ("INT", {"default": 1, "min": 0}),
                "bit_depth": (BIT_DEPTHS, {"default": "32-bit Float"}),
                "compression": (COMPRESSIONS, {"default": "ZIP"}),
                "alpha_mode": (ALPHA_MODES, {"default": "From Image"}),
                "custom_metadata": ("STRING", {"default": "", "multiline": True}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "write"
    OUTPUT_NODE = True
    CATEGORY = "FXTD Studios/Radiance/IO"

    def write(self, **kwargs):
        _, written_path, _ = RadianceWrite().write(**kwargs)
        return {"ui": {"images": _history_outputs(written_path)}, "result": ()}

class RadianceWrite:
    def write(self, image, filename_prefix, write_mode="Video", output_format="", fps=24.0, quality=10, 
              output_color_space="sRGB (Standard)", broadcast_safe=True, audio=None, 
              output_path="", start_frame=1, bit_depth="32-bit Float", compression="ZIP", 
              alpha_mode="From Image", custom_metadata="", prompt=None, extra_pnginfo=None):
        
        if hasattr(image, "get_components"): image = image.get_components().images
        elif isinstance(image, dict) and "samples" in image: image = image["samples"]
        
        if not isinstance(image, torch.Tensor):
            vpath = find_video_path(image)
            image = _load_video_frames(vpath) if vpath else None
        
        if image is None: raise ValueError("No image/video data")

        full_out = get_safe_output_dir(folder_paths.get_output_directory(), output_path, allow_absolute=True)
        images_out = apply_output_transform(image, output_color_space, broadcast_safe)
        images_np = images_out.cpu().numpy()
        ts = int(time.time())

        # Build Metadata
        meta = {
            "software": "Radiance v2.3",
            "created": datetime.datetime.now().isoformat(),
            "colorspace": output_color_space,
        }
        if custom_metadata:
            for line in custom_metadata.strip().split("\n"):
                if "=" in line:
                    k, v = line.split("=", 1)
                    meta[k.strip()] = v.strip()

        if write_mode == "Video" and any(x in output_format for x in ["Video", "GIF", "WEBP"]):
            res = self._write_video(images_np, filename_prefix, output_format, fps, quality, output_color_space, full_out, ts, audio, broadcast_safe)
        elif write_mode == "Single Image":
            # For Single Image, we only take the first frame and save without sequence naming
            res = self._write_sequence(images_np[:1], filename_prefix, output_format, quality, full_out, ts, start_frame, 4, False, bit_depth, compression, meta, alpha_mode, is_single_image=True)
        else:
            res = self._write_sequence(images_np, filename_prefix, output_format, quality, full_out, ts, start_frame, 4, True, bit_depth, compression, meta, alpha_mode)
        
        return (image, res, res)

    def _write_video(self, images_np, prefix, fmt, fps, quality, color_space, output_dir, ts, audio, broadcast_safe):
        import imageio.v3 as iio

        fpath_mp4 = _unique_path(os.path.join(output_dir, f"{prefix}_{ts}.mp4"))
        fpath_mov = _unique_path(os.path.join(output_dir, f"{prefix}_{ts}.mov"))

        if "H.264" in fmt:
            # ── H.264 / AVC — 8-bit, imageio path ──────────────────────────────
            data = (np.clip(images_np, 0, 1) * 255).astype(np.uint8)
            iio.imwrite(
                fpath_mp4, data,
                fps=fps, codec="libx264", pixelformat="yuv420p",
                macro_block_size=1,
            )
            fpath = fpath_mp4

        elif "H.265" in fmt:
            # ── H.265 / HEVC — true 10-bit, ffmpeg subprocess ─────────────────
            # imageio.v3 cannot convert uint16 RGB to yuv420p10le natively.
            # What it actually does: uint16 → uint8 (fires "Lossy conversion"
            # warning), then passes 8-bit frames to ffmpeg with pixelformat
            # yuv420p10le. ffmpeg interprets uint8 values (0-255) as 10-bit
            # samples (0-1023), so 255 looks like ~25% brightness → BLACK video.
            #
            # Correct approach: pipe raw uint8 RGB24 frames to ffmpeg stdin and
            # let ffmpeg do the RGB24→yuv420p10le conversion internally.
            # CRF range for H.265: 0 (lossless) – 51 (worst). quality 0-100 → CRF.
            crf_h265 = max(0, min(51, int((1.0 - quality / 100.0) * 51)))
            frames_u8 = (np.clip(images_np, 0, 1) * 255).astype(np.uint8)
            h, w = frames_u8.shape[1], frames_u8.shape[2]
            cmd = [
                "ffmpeg", "-y",
                "-f", "rawvideo", "-vcodec", "rawvideo",
                "-s", f"{w}x{h}", "-pix_fmt", "rgb24",
                "-r", str(fps),
                "-i", "pipe:0",
                "-vcodec", "libx265",
                "-pix_fmt", "yuv420p10le",   # true 10-bit output
                "-crf", str(crf_h265),
                "-preset", "slow",
                "-tag:v", "hvc1",            # Apple/QuickTime compatibility
                "-movflags", "+faststart",
                fpath_mp4,
            ]
            raw = frames_u8.tobytes()
            result = subprocess.run(cmd, input=raw, capture_output=True, timeout=600)  # nosec B603
            if result.returncode != 0:
                logger.error(
                    f"[RadianceWrite] H.265 encode failed:\n"
                    f"{result.stderr.decode(errors='replace')}"
                )
                raise RuntimeError("ffmpeg H.265 encode failed — see log for details")
            fpath = fpath_mp4

        elif "ProRes 4444" in fmt:
            # ── ProRes 4444 — true 12-bit, ffmpeg subprocess ──────────────────
            # Same imageio limitation: it cannot handle uint16 → yuv444p12le.
            # Fix: pipe uint16 RGB48LE frames to ffmpeg stdin; ffmpeg converts
            # to yuv444p12le internally for true ProRes 4444 12-bit encode.
            frames_u16 = (np.clip(images_np, 0, 1) * 65535).astype(np.uint16)
            h, w = frames_u16.shape[1], frames_u16.shape[2]
            cmd = [
                "ffmpeg", "-y",
                "-f", "rawvideo", "-vcodec", "rawvideo",
                "-s", f"{w}x{h}", "-pix_fmt", "rgb48le",
                "-r", str(fps),
                "-i", "pipe:0",
                "-vcodec", "prores_ks",
                "-profile:v", "4",           # ProRes 4444 profile
                "-pix_fmt", "yuv444p12le",   # true 12-bit 4:4:4
                "-vendor", "apl0",
                "-bits_per_mb", "8000",
                fpath_mov,
            ]
            raw = frames_u16.tobytes()
            result = subprocess.run(cmd, input=raw, capture_output=True, timeout=600)  # nosec B603
            if result.returncode != 0:
                logger.error(
                    f"[RadianceWrite] ProRes encode failed:\n"
                    f"{result.stderr.decode(errors='replace')}"
                )
                raise RuntimeError("ffmpeg ProRes encode failed — see log for details")
            fpath = fpath_mov

        else:
            # ── Fallback — H.264 8-bit ──────────────────────────────────────────
            data = (np.clip(images_np, 0, 1) * 255).astype(np.uint8)
            iio.imwrite(
                fpath_mp4, data,
                fps=fps, codec="libx264", pixelformat="yuv420p",
                macro_block_size=1,
            )
            fpath = fpath_mp4

        if audio:
            self._mux_audio(fpath, audio)
        return fpath

    def _write_sequence(self, images_np, prefix, fmt, quality, output_dir, ts, start, padding, use_ts, bdepth, comp, meta, alpha_mode, is_single_image=False):
        if is_single_image:
            target = output_dir
        else:
            target = _unique_dir(os.path.join(output_dir, f"{prefix}_{ts}")) if (use_ts and len(images_np) > 1) else output_dir
        
        os.makedirs(target, exist_ok=True)
        
        paths = []
        for i, frame in enumerate(images_np):
            num = str(start + i).zfill(padding)
            frame_meta = {**meta, "frame": start + i}
            
            if "EXR" in fmt:
                ext = ".exr"
                fname = f"{prefix}{ext}" if is_single_image else f"{prefix}.{num}{ext}"
                fpath = _unique_path(os.path.join(target, fname))
                if write_exr_robust:
                    success = write_exr_robust(fpath, frame, bdepth, comp, frame_meta)
                    if not success: cv2.imwrite(fpath, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).astype(np.float32))
                else:
                    cv2.imwrite(fpath, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).astype(np.float32))
            elif "HDR" in fmt:
                ext = ".hdr"
                fname = f"{prefix}{ext}" if is_single_image else f"{prefix}.{num}{ext}"
                fpath = _unique_path(os.path.join(target, fname))
                if write_hdr_rgbe: write_hdr_rgbe(fpath, frame)
                else: cv2.imwrite(fpath, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).astype(np.float32))
            elif "JPEG" in fmt:
                ext = ".jpg"
                fname = f"{prefix}{ext}" if is_single_image else f"{prefix}.{num}{ext}"
                fpath = _unique_path(os.path.join(target, fname))
                cv2.imwrite(fpath, cv2.cvtColor((np.clip(frame, 0, 1)*255).astype(np.uint8), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality * 10])
            else: # PNG
                ext = ".png"
                fname = f"{prefix}{ext}" if is_single_image else f"{prefix}.{num}{ext}"
                fpath = _unique_path(os.path.join(target, fname))
                data = (np.clip(frame, 0, 1)*65535).astype(np.uint16) if "16-bit" in fmt else (np.clip(frame, 0, 1)*255).astype(np.uint8)
                cv2.imwrite(fpath, cv2.cvtColor(data, cv2.COLOR_RGB2BGR))
            paths.append(fpath)
            
        return paths[0] if len(paths) == 1 else target

    def _mux_audio(self, video_path, audio):
        try:
            waveform, sr = audio.get("waveform"), audio.get("sample_rate", 44100)
            if waveform is None: return
            import struct
            wav_np = waveform.squeeze(0).cpu().numpy()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp: tmp_wav = tmp.name
            with open(tmp_wav, "wb") as f:
                f.write(b"RIFF" + struct.pack("<I", 36 + wav_np.size*4) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 3, wav_np.shape[0], sr, sr*wav_np.shape[0]*4, wav_np.shape[0]*4, 32) + b"data" + struct.pack("<I", wav_np.size*4) + wav_np.T.flatten().astype(np.float32).tobytes())
            tmp_out = video_path + ".tmp" + os.path.splitext(video_path)[1]
            subprocess.run(["ffmpeg", "-y", "-i", video_path, "-i", tmp_wav, "-c:v", "copy", "-c:a", "aac", "-shortest", tmp_out], capture_output=True)
            if os.path.exists(tmp_out): os.replace(tmp_out, video_path)
            os.unlink(tmp_wav)
        except: pass


class SaveImage16bit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "Radiance"}),
            },
            "optional": {
                "output_path": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "FXTD Studios/Radiance/IO"

    def save(self, image, filename_prefix="Radiance", output_path=""):
        _, written_path, _ = RadianceWrite().write(
            image=image,
            filename_prefix=filename_prefix,
            write_mode="Single Image",
            output_format="Image Sequence — PNG (16-bit)",
            output_path=output_path,
            output_color_space="sRGB (Standard)",
            broadcast_safe=True,
        )
        return {"ui": {"images": _history_outputs(written_path)}, "result": ()}


class RadianceSaveEXR:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "Radiance"}),
                "bit_depth": (BIT_DEPTHS, {"default": "32-bit Float"}),
                "compression": (COMPRESSIONS, {"default": "ZIP"}),
            },
            "optional": {
                "output_path": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "FXTD Studios/Radiance/IO"

    def save(self, image, filename_prefix="Radiance", bit_depth="32-bit Float", compression="ZIP", output_path=""):
        _, written_path, _ = RadianceWrite().write(
            image=image,
            filename_prefix=filename_prefix,
            write_mode="Single Image",
            output_format="Image Sequence — EXR (32-bit)",
            output_path=output_path,
            output_color_space="Linear (sRGB)",
            broadcast_safe=False,
            bit_depth=bit_depth,
            compression=compression,
        )
        return {"ui": {"images": _history_outputs(written_path)}, "result": ()}


class RadianceSaveHDR:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "Radiance"}),
            },
            "optional": {
                "output_path": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "FXTD Studios/Radiance/IO"

    def save(self, image, filename_prefix="Radiance", output_path=""):
        _, written_path, _ = RadianceWrite().write(
            image=image,
            filename_prefix=filename_prefix,
            write_mode="Single Image",
            output_format="Image Sequence — Radiance HDR (.hdr)",
            output_path=output_path,
            output_color_space="Linear (sRGB)",
            broadcast_safe=False,
        )
        return {"ui": {"images": _history_outputs(written_path)}, "result": ()}

NODE_CLASS_MAPPINGS = {
    "RadianceDigitalCinemaRead": RadianceDigitalCinemaRead,
    "RadianceDigitalCinemaWrite": RadianceDigitalCinemaWrite,
    "◎ RadianceDigitalCinemaRead": RadianceDigitalCinemaRead,
    "◎ RadianceDigitalCinemaWrite": RadianceDigitalCinemaWrite,
    "SaveImage16bit": SaveImage16bit,
    "RadianceSaveEXR": RadianceSaveEXR,
    "RadianceSaveHDR": RadianceSaveHDR,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceDigitalCinemaRead": "◎ Radiance Read",
    "RadianceDigitalCinemaWrite": "◎ Radiance Write",
    "◎ RadianceDigitalCinemaRead": "◎ Radiance Read",
    "◎ RadianceDigitalCinemaWrite": "◎ Radiance Write",
    "SaveImage16bit": "◎ Save Image 16-bit",
    "RadianceSaveEXR": "◎ Radiance Save EXR",
    "RadianceSaveHDR": "◎ Radiance Save HDR",
}
