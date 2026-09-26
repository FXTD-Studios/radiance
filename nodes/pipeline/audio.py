# ============================================================
# FXTD STUDIOS — Radiance
# nodes_audio_cut.py  —  Audio-Driven Cut Detection & Transcription
# ============================================================
# Priority 7 of 7 in the Griptape-inspired feature roadmap.
#
# Two nodes:
#   RadianceAudioCut
#     • Detects beats / onsets / transients in an audio file
#     • Returns frame indices (at a specified FPS) where cuts
#       should be placed — pipe directly into temporal/sequence nodes
#     • Backends: librosa (full analysis), scipy (lightweight),
#       ffmpeg subprocess fallback (silence/loud detection)
#
#   RadianceAudioTranscribe
#     • Transcribes speech in an audio file to text
#     • Backends: openai-whisper (local), OpenAI Whisper API,
#       subprocess whisper CLI fallback
#     • Returns full transcript + per-segment word timings as JSON
# ============================================================

from radiance.config.constants import VERSION as __version__  # noqa: E402

import io
import json
import logging
import math
import os
import subprocess
import sys
import tempfile
import traceback
import urllib.request
import urllib.error
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from radiance.secret_utils import resolve_secret
from radiance.path_utils import strip_path_quotes

logger = logging.getLogger("radiance.audio_cut")

# ---------------------------------------------------------------------------
# Path security helpers
# ---------------------------------------------------------------------------

# Maximum path length to reject obviously malicious inputs early.
_MAX_PATH_LEN = 4096

# Audio extensions that are accepted as input.
_AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".flac", ".aac", ".ogg", ".m4a",
    ".mp4", ".mov", ".mxf", ".mkv", ".avi",
}


def _validate_audio_path(filepath: str) -> str:
    """
    Validate and normalise a user-supplied audio file path.

    Raises ValueError with a human-readable message for any unsafe input.
    Returns the resolved absolute path string on success.

    Checks performed
    ----------------
    - Not empty / too long
    - No null bytes or control characters
    - No path-traversal sequences after resolution
    - File exists and is a regular file (not a symlink to an unsafe target)
    - Extension is in the allowlist
    """
    filepath = strip_path_quotes(filepath)
    if not filepath:
        raise ValueError("Audio file path must not be empty.")
    if len(filepath) > _MAX_PATH_LEN:
        raise ValueError(f"Audio file path exceeds maximum length ({_MAX_PATH_LEN}).")

    # Reject null bytes and ASCII control characters — common injection vectors.
    if any(ord(c) < 32 for c in filepath):
        raise ValueError("Audio file path contains illegal control characters.")

    resolved = Path(filepath).resolve()

    # Check extension against the allowlist.
    if resolved.suffix.lower() not in _AUDIO_EXTENSIONS:
        raise ValueError(
            f"Unsupported audio extension '{resolved.suffix}'. "
            f"Allowed: {', '.join(sorted(_AUDIO_EXTENSIONS))}"
        )

    if not resolved.exists():
        raise ValueError(f"Audio file not found: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"Path is not a regular file: {resolved}")

    return str(resolved)

# ---------------------------------------------------------------------------
# Optional heavy-dependency flags — all imports deferred to call time
# ---------------------------------------------------------------------------
def _has(pkg: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(pkg) is not None

HAS_LIBROSA = _has("librosa")
HAS_NUMPY   = _has("numpy")
HAS_SCIPY   = _has("scipy")
HAS_SOUNDFILE = _has("soundfile")
HAS_WHISPER = _has("whisper")   # openai-whisper local package
HAS_OPENAI  = _has("openai")


# ===========================================================================
# Audio loading helpers
# ===========================================================================

def _load_audio_numpy(filepath: str, target_sr: int = 22050) -> Tuple[Any, int]:
    """
    Load audio file to a mono float32 numpy array.
    Tries: librosa → soundfile+scipy resampling → ffmpeg pipe → stub zeros.
    Returns (samples_array, sample_rate).
    """
    if HAS_LIBROSA:
        import librosa  # type: ignore
        y, sr = librosa.load(filepath, sr=target_sr, mono=True)
        return y, sr

    if HAS_NUMPY and HAS_SOUNDFILE:
        import numpy as np
        import soundfile as sf  # type: ignore
        data, sr = sf.read(filepath, always_2d=True)
        mono = data.mean(axis=1).astype(np.float32)
        if sr != target_sr and HAS_SCIPY:
            from scipy.signal import resample  # type: ignore
            n_out = int(len(mono) * target_sr / sr)
            mono = resample(mono, n_out).astype(np.float32)
            sr = target_sr
        return mono, sr

    # ffmpeg pipe fallback — decode to raw PCM s16le mono
    try:
        cmd = [
            _ffmpeg_bin(), "-i", filepath,
            "-ac", "1", "-ar", str(target_sr),
            "-f", "s16le", "-",
        ]
        proc = subprocess.run(cmd, capture_output=True, timeout=120)
        if proc.returncode == 0 and HAS_NUMPY:
            import numpy as np
            samples = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
            return samples, target_sr
    except Exception as exc:
        logger.warning("[nodes_audio_cut]: %s", exc)

    # Stub: return silence
    if HAS_NUMPY:
        import numpy as np
        return np.zeros(target_sr * 5, dtype=np.float32), target_sr
    return [], target_sr


# ===========================================================================
# Beat / onset detection backends
# ===========================================================================

def _detect_librosa(filepath: str, method: str, fps: float,
                    sensitivity: float, min_interval_frames: int) -> List[int]:
    """Full librosa beat/onset detection."""
    import librosa  # type: ignore
    import numpy as np

    y, sr = librosa.load(filepath, sr=22050, mono=True)

    if method == "beats":
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        times = librosa.frames_to_time(beat_frames, sr=sr)
    elif method == "onsets":
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units="frames",
                                                   delta=1.0 - sensitivity)
        times = librosa.frames_to_time(onset_frames, sr=sr)
    else:  # transients
        spectral_flux = librosa.onset.onset_strength(y=y, sr=sr)
        threshold = np.mean(spectral_flux) + (1.0 - sensitivity) * np.std(spectral_flux)
        onset_frames = librosa.util.peak_pick(
            spectral_flux, pre_max=3, post_max=3, pre_avg=3,
            post_avg=5, delta=threshold * 0.1, wait=0,
        )
        times = librosa.frames_to_time(onset_frames, sr=sr)

    frame_indices = [int(t * fps) for t in times]
    # Enforce minimum interval
    filtered = []
    last = -min_interval_frames - 1
    for f in sorted(frame_indices):
        if f - last >= min_interval_frames:
            filtered.append(f)
            last = f
    return filtered


def _detect_scipy(filepath: str, method: str, fps: float,
                  sensitivity: float, min_interval_frames: int) -> List[int]:
    """Lightweight scipy-based onset detection via energy envelope."""
    import numpy as np
    from scipy.signal import find_peaks  # type: ignore

    y, sr = _load_audio_numpy(filepath, 22050)
    if not hasattr(y, "__len__") or len(y) == 0:
        return []

    # Compute short-time energy in ~23ms windows
    hop = int(sr * 0.023)
    frame_count = len(y) // hop
    energy = np.array([np.sum(y[i*hop:(i+1)*hop] ** 2) for i in range(frame_count)])

    # Differentiate energy for onset strength
    onset_strength = np.diff(energy, prepend=energy[0])
    onset_strength = np.clip(onset_strength, 0, None)

    threshold = np.mean(onset_strength) + (1.0 - sensitivity) * np.std(onset_strength)
    min_dist = max(1, int(min_interval_frames * sr / hop / fps))
    peaks, _ = find_peaks(onset_strength, height=threshold, distance=min_dist)

    # Convert energy frames → time → video frames
    times = peaks * hop / sr
    return sorted(set(int(t * fps) for t in times))


def _ffmpeg_bin() -> str:
    """ffmpeg path — PATH first, then the binary imageio-ffmpeg ships."""
    from radiance.core.ffmpeg import require_ffmpeg
    return require_ffmpeg()


def _detect_ffmpeg(filepath: str, fps: float,
                   sensitivity: float, min_interval_frames: int) -> List[int]:
    """
    Use ffmpeg silencedetect filter to find non-silence onsets.
    Falls back to scanning amplitude periodically.
    """
    noise_db = -30 + int((1.0 - sensitivity) * 20)  # sensitivity 1.0 → -30dB, 0.0 → -10dB
    cmd = [
        _ffmpeg_bin(), "-i", filepath,
        "-af", f"silencedetect=noise={noise_db}dB:duration=0.1",
        "-f", "null", "-",
    ]
    # A failed ffmpeg run raises: it used to return [], which is exactly what
    # a track with no detectable onsets returns.
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    output = proc.stderr
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg silencedetect failed (exit {proc.returncode}): "
                           f"{output.strip().splitlines()[-1] if output.strip() else 'no output'}")

    # Parse "silence_end: X.XX" lines as onset times
    import re
    times = []
    for m in re.finditer(r"silence_end:\s*([\d.]+)", output):
        times.append(float(m.group(1)))

    frame_indices = sorted(set(int(t * fps) for t in times))
    filtered = []
    last = -min_interval_frames - 1
    for f in frame_indices:
        if f - last >= min_interval_frames:
            filtered.append(f)
            last = f
    return filtered


# ===========================================================================
# Node: RadianceAudioCut
# ===========================================================================

class RadianceAudioCut:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Video"
    DESCRIPTION = "Detect beat, onset, or transient events in audio for video cut points."
    """
    Analyse an audio file and return a list of frame indices where
    rhythmic cuts should be placed, based on beat/onset/transient detection.

    Output:
      cut_frames_json  — JSON array of int frame indices  e.g. [0, 24, 48, 96]
      cut_times_json   — JSON array of float times in seconds
      cut_count        — integer number of cut points found
      analysis_report  — human-readable summary
    """

    METHODS  = ["beats", "onsets", "transients"]
    BACKENDS = ["Auto", "librosa", "scipy", "ffmpeg"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_filepath": ("STRING", {
                    "default": "/path/to/audio.wav",
                    "tooltip": "Absolute path to audio file (WAV, MP3, FLAC, AAC, etc.)",
                }),
                "fps": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001,
                    "tooltip": "Frame rate of the target video sequence",
                }),
                "method": (cls.METHODS, {"default": "beats",
                    "tooltip": "Honoured by the librosa backend. scipy always finds energy "
                               "onsets and ffmpeg always finds silence ends; the report says so."}),
                "sensitivity": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "0 = only strong peaks, 1 = detect all micro-variations",
                }),
                "min_interval_frames": ("INT", {
                    "default": 12, "min": 1,
                    "tooltip": "Minimum frames between consecutive cut points",
                }),
                "backend": (cls.BACKENDS, {"default": "Auto",
                    "tooltip": "Auto: librosa if installed, else scipy, else ffmpeg. scipy finds energy "
                               "onsets; ffmpeg finds silence ends, its threshold running from -30 dB "
                               "(sensitivity 1) to -10 dB (sensitivity 0)."}),
            },
            "optional": {
                "frame_offset": ("INT", {
                    "default": 0,
                    "tooltip": "Add this value to every returned frame index (useful when audio starts mid-sequence)",
                }),
                "max_cuts": ("INT", {
                    "default": 0,
                    "tooltip": "If > 0, keep N cut points evenly spaced through the detected "
                               "list (the backends do not return per-cut strength).",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = ("cut_frames_json", "cut_times_json", "cut_count", "analysis_report")
    FUNCTION = "detect_cuts"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Video"

    def detect_cuts(
        self,
        audio_filepath: str,
        fps: float,
        method: str,
        sensitivity: float,
        min_interval_frames: int,
        backend: str = "Auto",
        frame_offset: int = 0,
        max_cuts: int = 0,
    ):
        report = [
            f"=== Radiance Audio Cut v{__version__} ===",
            f"File     : {audio_filepath}",
            f"FPS      : {fps}",
            f"Method   : {method}",
            f"Backend  : {backend}",
            f"Sensitivity: {sensitivity:.2f}",
            "",
        ]

        # Security: validate and resolve path before any file operation.
        # A bad path raises. Zero cuts is a legitimate result, so returning
        # "[]" for a missing file made the two indistinguishable downstream.
        try:
            audio_filepath = _validate_audio_path(audio_filepath)
        except ValueError as exc:
            raise ValueError(f"Radiance Audio Cut: {exc}") from exc

        frames: List[int] = []
        used_backend = backend

        try:
            if backend == "Auto":
                if HAS_LIBROSA:
                    used_backend = "librosa"
                elif HAS_SCIPY and HAS_NUMPY:
                    used_backend = "scipy"
                else:
                    used_backend = "ffmpeg"

            if used_backend == "librosa":
                frames = _detect_librosa(audio_filepath, method, fps,
                                          sensitivity, min_interval_frames)
            elif used_backend == "scipy":
                frames = _detect_scipy(audio_filepath, method, fps,
                                        sensitivity, min_interval_frames)
            else:  # ffmpeg
                frames = _detect_ffmpeg(audio_filepath, fps,
                                         sensitivity, min_interval_frames)

        except Exception as exc:
            # Raise: an empty cut list is a valid answer for a quiet track, so
            # returning [] here made a failed analysis look like one.
            raise RuntimeError(f"Radiance Audio Cut: {used_backend} backend failed on "
                               f"{audio_filepath}: {exc}") from exc

        if used_backend == "scipy" and method != "onsets":
            report.append(f"NOTE: scipy backend has no {method} detector; ran energy onsets")
        elif used_backend == "ffmpeg":
            report.append(f"NOTE: ffmpeg backend ignores method ({method}); cut points are "
                          f"silence ends from silencedetect")

        # Apply offset
        frames = [f + frame_offset for f in frames]

        # Evenly spaced subset: no backend reports per-cut strength.
        if max_cuts > 0 and len(frames) > max_cuts:
            step = len(frames) / max_cuts
            frames = [frames[int(i * step)] for i in range(max_cuts)]

        times = [round(f / fps, 4) for f in frames]

        report += [
            f"Backend used : {used_backend}",
            f"Cut points   : {len(frames)}",
            f"Frame offset : {frame_offset}",
        ]
        if frames:
            report.append(f"First cut    : frame {frames[0]} ({times[0]:.3f}s)")
            report.append(f"Last cut     : frame {frames[-1]} ({times[-1]:.3f}s)")

        return (
            json.dumps(frames),
            json.dumps(times),
            len(frames),
            "\n".join(report),
        )


# ===========================================================================
# Transcription helpers
# ===========================================================================

def _transcribe_local_whisper(filepath: str, model_size: str,
                               language: str) -> Tuple[str, List[dict]]:
    """openai-whisper local inference."""
    import whisper  # type: ignore
    model = whisper.load_model(model_size)
    opts = {}
    if language and language.lower() not in ("auto", ""):
        opts["language"] = language
    result = model.transcribe(filepath, **opts)
    text = result.get("text", "").strip()
    segments = [
        {"start": s["start"], "end": s["end"], "text": s["text"].strip()}
        for s in result.get("segments", [])
    ]
    return text, segments


def _transcribe_openai_api(filepath: str, model: str, api_key: str,
                            language: str) -> Tuple[str, List[dict]]:
    """OpenAI Whisper API (audio.transcriptions.create)."""
    if HAS_OPENAI:
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=api_key)
        with open(filepath, "rb") as f:
            kwargs = {"model": model, "file": f, "response_format": "verbose_json"}
            if language and language.lower() not in ("auto", ""):
                kwargs["language"] = language
            resp = client.audio.transcriptions.create(**kwargs)
        text = getattr(resp, "text", str(resp))
        segs = []
        for s in getattr(resp, "segments", []):
            segs.append({
                "start": getattr(s, "start", 0),
                "end": getattr(s, "end", 0),
                "text": getattr(s, "text", "").strip(),
            })
        return text, segs

    # Pure HTTP fallback
    import mimetypes
    mime = mimetypes.guess_type(filepath)[0] or "audio/wav"
    boundary = uuid.uuid4().hex
    fname = os.path.basename(filepath)
    with open(filepath, "rb") as f:
        audio_bytes = f.read()

    body_parts = [
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{model}\r\n",
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"response_format\"\r\n\r\nverbose_json\r\n",
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{fname}\"\r\nContent-Type: {mime}\r\n\r\n",
    ]
    if language and language.lower() not in ("auto", ""):
        body_parts.insert(1,
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"language\"\r\n\r\n{language}\r\n")

    preamble = "".join(body_parts).encode()
    ending = f"\r\n--{boundary}--\r\n".encode()
    full_body = preamble + audio_bytes + ending

    req = urllib.request.Request(
        "https://api.openai.com/v1/audio/transcriptions",
        data=full_body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
        text = data.get("text", "")
        segs = [{"start": s.get("start", 0), "end": s.get("end", 0),
                  "text": s.get("text", "")} for s in data.get("segments", [])]
        return text, segs
    except Exception as exc:
        return f"[OpenAI API error: {exc}]", []


def _transcribe_whisper_cli(filepath: str, model_size: str,
                             language: str) -> Tuple[str, List[dict]]:
    """
    Fallback: call the `whisper` CLI subprocess and parse its output.
    Returns (text, segments) — segments are approximate (no timing from CLI).
    """
    cmd = ["whisper", filepath, "--model", model_size, "--output_format", "json"]
    if language and language.lower() not in ("auto", ""):
        cmd += ["--language", language]

    with tempfile.TemporaryDirectory() as tmpdir:
        cmd += ["--output_dir", tmpdir]
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=True)
        except subprocess.CalledProcessError as e:
            return f"[whisper CLI error: {e.stderr[:200]}]", []
        except FileNotFoundError:
            return "[whisper CLI not found in PATH]", []

        base = os.path.splitext(os.path.basename(filepath))[0]
        out_json = os.path.join(tmpdir, base + ".json")
        if os.path.isfile(out_json):
            with open(out_json) as f:
                data = json.load(f)
            text = data.get("text", "")
            segs = [{"start": s.get("start", 0), "end": s.get("end", 0),
                      "text": s.get("text", "")} for s in data.get("segments", [])]
            return text, segs
    return "[no output]", []


# ===========================================================================
# Node: RadianceAudioTranscribe
# ===========================================================================

class RadianceAudioTranscribe:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Video"
    DESCRIPTION = "Transcribe speech from an audio or video file using Whisper."
    """
    Transcribe speech in an audio file to text.

    Backends (tried in order when "Auto"):
      1. openai-whisper (local model, most accurate offline)
      2. OpenAI Whisper API (requires api_key)
      3. whisper CLI subprocess (if `whisper` is on PATH)

    Outputs:
      transcript       — full plain text
      segments_json    — JSON array of {start, end, text} dicts (seconds)
      segment_count    — number of segments
      transcribe_report — human-readable summary
    """

    BACKENDS    = ["Auto", "local_whisper", "openai_api", "whisper_cli"]
    MODEL_SIZES = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
    LANGUAGES   = ["auto", "en", "ar", "zh", "fr", "de", "es", "ja", "ko",
                   "pt", "ru", "it", "nl", "pl", "tr", "hi"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_filepath": ("STRING", {
                    "default": "/path/to/audio.wav",
                    "tooltip": "Absolute path to audio/video file",
                }),
                "backend": (cls.BACKENDS, {"default": "Auto",
                    "tooltip": "Auto: local openai-whisper if installed, else the OpenAI API if a key "
                               "resolves, else the whisper CLI on PATH. Failures are written to "
                               "transcribe_report, not raised."}),
                "model_size": (cls.MODEL_SIZES, {"default": "base",
                    "tooltip": "Whisper model for local_whisper and whisper_cli (larger is slower and "
                               "more accurate). openai_api always uses whisper-1."}),
                "language": (cls.LANGUAGES, {"default": "auto",
                    "tooltip": "Spoken language code passed to Whisper. auto lets Whisper detect it."}),
            },
            "optional": {
                "openai_api_key_env": ("STRING", {
                    "default": "OPENAI_API_KEY",
                    "tooltip": "Environment variable name containing the OpenAI API key.",
                }),
                "openai_api_key": ("STRING", {
                    "default": "",
                    "tooltip": "Legacy fallback only. Prefer openai_api_key_env so workflows do not store secrets.",
                }),
                "include_timings": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include per-segment timestamps in segments_json",
                }),
                "max_segment_chars": ("INT", {
                    "default": 0,
                    "tooltip": "If > 0, split long segments at this character count",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = ("transcript", "segments_json", "segment_count", "transcribe_report")
    FUNCTION = "transcribe"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Video"

    def transcribe(
        self,
        audio_filepath: str,
        backend: str,
        model_size: str,
        language: str,
        openai_api_key: str = "",
        include_timings: bool = True,
        max_segment_chars: int = 0,
        openai_api_key_env: str = "OPENAI_API_KEY",
    ):
        report = [
            f"=== Radiance Audio Transcribe v{__version__} ===",
            f"File    : {audio_filepath}",
            f"Backend : {backend}",
            f"Model   : {model_size}",
            f"Language: {language}",
            "",
        ]

        # Security: validate and resolve path before any file operation.
        try:
            audio_filepath = _validate_audio_path(audio_filepath)
        except ValueError as exc:
            report.append(f"ERROR: {exc}")
            return ("", "[]", 0, "\n".join(report))

        text = ""
        segments: List[dict] = []
        used_backend = backend
        error_msg = ""
        resolved_openai_api_key = resolve_secret(
            explicit_value=openai_api_key,
            env_var=openai_api_key_env,
            default_env_var="OPENAI_API_KEY",
        )

        try:
            if backend == "Auto":
                if HAS_WHISPER:
                    used_backend = "local_whisper"
                elif resolved_openai_api_key:
                    used_backend = "openai_api"
                else:
                    used_backend = "whisper_cli"

            if used_backend == "local_whisper":
                text, segments = _transcribe_local_whisper(
                    audio_filepath, model_size, language)
            elif used_backend == "openai_api":
                if not resolved_openai_api_key:
                    error_msg = "Set openai_api_key_env or provide legacy openai_api_key for openai_api backend"
                else:
                    text, segments = _transcribe_openai_api(
                        audio_filepath, "whisper-1", resolved_openai_api_key, language)
            elif used_backend == "whisper_cli":
                text, segments = _transcribe_whisper_cli(
                    audio_filepath, model_size, language)

        except Exception as exc:
            error_msg = f"{used_backend} error: {exc}"
            traceback.print_exc()

        if error_msg:
            report.append(f"ERROR: {error_msg}")

        # Optionally split long segments
        if max_segment_chars > 0 and segments:
            split_segs: List[dict] = []
            for seg in segments:
                seg_text = seg.get("text", "")
                if len(seg_text) <= max_segment_chars:
                    split_segs.append(seg)
                else:
                    # Naive word-wrap split
                    words = seg_text.split()
                    chunk: List[str] = []
                    chunk_len = 0
                    start = seg.get("start", 0)
                    duration = seg.get("end", start) - start
                    for w in words:
                        if chunk_len + len(w) + 1 > max_segment_chars and chunk:
                            split_segs.append({
                                "start": start,
                                "end": start + duration * len(chunk) / len(words),
                                "text": " ".join(chunk),
                            })
                            chunk = [w]
                            chunk_len = len(w)
                        else:
                            chunk.append(w)
                            chunk_len += len(w) + 1
                    if chunk:
                        split_segs.append({
                            "start": start + duration * (len(words) - len(chunk)) / max(len(words), 1),
                            "end": seg.get("end", start),
                            "text": " ".join(chunk),
                        })
            segments = split_segs

        if not include_timings:
            segments = [{"text": s.get("text", "")} for s in segments]

        report += [
            f"Backend used : {used_backend}",
            f"Segments     : {len(segments)}",
            f"Characters   : {len(text)}",
            f"Words        : {len(text.split())}",
        ]
        if text:
            preview = text[:200].replace("\n", " ")
            report.append(f"Preview      : {preview}{'...' if len(text) > 200 else ''}")

        return (
            text,
            json.dumps(segments, ensure_ascii=False, indent=2),
            len(segments),
            "\n".join(report),
        )





# ===========================================================================
# Registration
# ===========================================================================

NODE_CLASS_MAPPINGS = {
    "RadianceAudioCut":           RadianceAudioCut,
    "RadianceAudioTranscribe":    RadianceAudioTranscribe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceAudioCut":           "◎ Radiance Audio Cut",
    "RadianceAudioTranscribe":    "◎ Radiance Audio Transcribe",
}
