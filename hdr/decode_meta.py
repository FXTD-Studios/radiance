"""HDR VAE decode contract helpers (3.5.0).

Pure torch, no ComfyUI import, so the decode wrapper (nodes/generate/engine.py)
and the engine (hdr/vae.py) share one implementation and both stay testable.

* camera gamut of each log space and of each decode target
* the latent fingerprint that tells a live HDR Encode latent from a sampled
  copy still carrying its ``radiance_meta`` dict
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Dict, Optional, Tuple

import torch

logger = logging.getLogger("radiance")


# ─────────────────────────────────────────────────────────────────────────────
# 3.5.0: camera gamuts for the log spaces
# ─────────────────────────────────────────────────────────────────────────────
# A camera log encoding is a transfer curve AND a gamut: LogC4 is AWG4 +
# LogC4, S-Log3 is S-Gamut3.Cine + S-Log3 and so on. The log targets used to
# apply the curve alone to Rec.709 linear, so an "ARRI LogC4" EXR carried
# Rec.709 primaries under a LogC4 label and turned desaturated in any
# colour-managed host. Names and primaries come from color/encodings, which
# matches OpenColorIO's ACES studio config to 1e-6.
LOG_SPACE_GAMUT: Dict[str, str] = {
    "ARRI LogC3": "ARRI Wide Gamut 3",
    "ARRI LogC4": "ARRI Wide Gamut 4",
    "Sony S-Log3": "S-Gamut3.Cine",
    "Panasonic V-Log": "V-Gamut",
    "DaVinci Intermediate": "DaVinci Wide Gamut",
    "RED Log3G10": "REDWideGamutRGB",
}
#: Gamut each decode target_space is delivered in.
TARGET_SPACE_GAMUT: Dict[str, str] = {
    "Linear": "Rec.709", "sRGB": "Rec.709", "Raw": "Rec.709",
    "ACEScg": "AP1", "ACES 2065-1": "AP0", "Rec.2020 Linear": "Rec.2020",
    **LOG_SPACE_GAMUT,
}
@functools.lru_cache(maxsize=64)
def _gamut_mat_cpu(src: str, dst: str) -> torch.Tensor:
    from radiance.color.encodings import gamut_matrix
    return torch.tensor(gamut_matrix(src, dst), dtype=torch.float32).T.contiguous()


def _gamut_mat_t(src: str, dst: str) -> Optional[torch.Tensor]:
    """Row-vector (already transposed) src -> dst gamut matrix, None if identity."""
    if src == dst:
        return None
    return _gamut_mat_cpu(src, dst)


# ─────────────────────────────────────────────────────────────────────────────
# 3.5.0: is radiance_meta still describing this latent?
# ─────────────────────────────────────────────────────────────────────────────
# ComfyUI's samplers copy the latent dict (``latent.copy()``) and replace only
# "samples", so the radiance_meta HDR Encode stamped rides through KSampler,
# SamplerCustom and Radiance's own sampler untouched. Decode then saw
# hdr_mode="Compress (Log)" on a freshly diffused, sRGB-like latent and ran
# the log decompression over it: blown-out frames on every img2img graph that
# started from HDR Encode. The keys that say how the latent is coded are only
# trusted when the latent is provably the one Encode produced.
_HDR_META_KEYS = ("hdr_mode", "source_space", "working_gamut")


def latent_fingerprint(t: torch.Tensor) -> Dict[str, Any]:
    """Cheap identity of a latent tensor: shape plus three global moments."""
    f = t.detach().float()
    n = f.numel()
    return {
        "shape": [int(d) for d in t.shape],
        "mean": float(f.mean()) if n else 0.0,
        "abs_mean": float(f.abs().mean()) if n else 0.0,
        "std": float(f.std()) if n > 1 else 0.0,
    }


def radiance_meta_is_live(meta: Any, t: Any) -> bool:
    """True when ``meta`` was stamped on exactly this latent (no sampler since)."""
    if not isinstance(meta, dict) or not isinstance(t, torch.Tensor):
        return False
    fp = meta.get("latent_fingerprint")
    if not isinstance(fp, dict):
        return False
    try:
        if [int(d) for d in fp["shape"]] != [int(d) for d in t.shape]:
            return False
        cur = latent_fingerprint(t)
        for k in ("mean", "abs_mean", "std"):
            a, b = float(fp[k]), float(cur[k])
            if abs(a - b) > 1e-4 * max(abs(a), abs(b)) + 1e-6:
                return False
        return True
    except (KeyError, TypeError, ValueError, RuntimeError):
        return False


def strip_hdr_meta(samples: Dict[str, Any]) -> Dict[str, Any]:
    """Copy of ``samples`` whose radiance_meta no longer claims an HDR coding.

    Padding and VAE-factor entries are kept: a sampler preserves the latent's
    geometry, so auto-crop stays correct.
    """
    meta = samples.get("radiance_meta")
    if not isinstance(meta, dict) or not any(k in meta for k in _HDR_META_KEYS):
        return samples
    out = dict(samples)
    out["radiance_meta"] = {k: v for k, v in meta.items() if k not in _HDR_META_KEYS}
    return out


def verify_radiance_meta(samples: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """(samples, live): strips HDR coding keys when the fingerprint does not match."""
    meta = samples.get("radiance_meta")
    if not isinstance(meta, dict) or not meta:
        return samples, False
    if radiance_meta_is_live(meta, samples.get("samples")):
        return samples, True
    if any(meta.get(k) for k in _HDR_META_KEYS):
        logger.info(
            "[Radiance HDR Decode] this latent's HDR Encode metadata "
            "(hdr_mode=%s) no longer matches it: the latent went through a "
            "sampler or was edited. Decoding it as a standard diffusion latent.",
            meta.get("hdr_mode"),
        )
    return strip_hdr_meta(samples), False


def _encode_working_gamut(source_space: str, hdr_mode: str) -> str:
    """Gamut of the linear light HDR Encode put into the latent."""
    if hdr_mode == "Compress (Log)" and source_space in LOG_SPACE_GAMUT:
        return LOG_SPACE_GAMUT[source_space]
    return "Rec.709"
