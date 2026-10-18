import collections
import os
import threading
import logging
import torch
from typing import Dict, Any, Optional

logger = logging.getLogger("radiance.cache")

_VIEWER_CACHE_MAX = 8

#: Byte budget for the viewer frame cache.
#:
#: The cache used to be bounded by ENTRY COUNT alone. Eight entries sounds
#: modest until each one is a 240-frame 4K RGBA fp32 plate (~31 GB), so the
#: bound placed no real limit on memory at all. Entries are also now stored as
#: detached CPU copies: previously the tensor was cached as-is, so a CUDA
#: IMAGE pinned that VRAM for the life of the process (invisible to ComfyUI's
#: model manager, and the sampler would OOM later with no obvious culprit),
#: and because the node returns the *same object* it caches, any downstream
#: in-place op silently mutated the plate that /radiance/deliver would export.
_VIEWER_CACHE_MAX_BYTES = int(os.environ.get("RADIANCE_VIEWER_CACHE_BYTES", 2 * 1024 ** 3))

_VIEWER_CACHE: collections.OrderedDict = collections.OrderedDict()
_VIEWER_CACHE_LOCK = threading.Lock()

#: AUDIT-FIX (R2): which frame set a cache entry holds. Every insert gets a
#: new number, so a delivery queued for one run can tell that the Viewer has
#: since been given other frames (a re-run replaces the entry under the same
#: key) and refuse, instead of exporting the new pixels under the old job's
#: name, range and grade.
_VIEWER_GENERATION: Dict[str, int] = {}
_VIEWER_GENERATION_NEXT = [0]


def _tensor_nbytes(t: Any) -> int:
    try:
        return int(t.numel()) * int(t.element_size())
    except Exception:
        return 0


def _viewer_cache_set(key: str, value: torch.Tensor) -> None:
    """
    Thread-safe LRU insert into the viewer cache.

    Stores a detached CPU copy and evicts on a byte budget as well as an entry
    count, so the cache can neither pin VRAM nor grow without bound.
    """
    nbytes = _tensor_nbytes(value)
    oversize = nbytes > _VIEWER_CACHE_MAX_BYTES
    try:
        if isinstance(value, torch.Tensor):
            if oversize and value.device.type == "cpu":
                # A plate over the whole budget is kept as it is, not copied:
                # the copy doubled the memory of a long 4K shot at exactly the
                # moment it was largest. Only an in-place change downstream can
                # now reach it before delivery, which beats running out of RAM.
                value = value.detach()
            else:
                value = value.detach().to("cpu", copy=True)
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("[Radiance] Could not copy viewer frame to CPU: %s", exc)
    with _VIEWER_CACHE_LOCK:
        if key in _VIEWER_CACHE:
            _VIEWER_CACHE.move_to_end(key)
        _VIEWER_CACHE[key] = value
        _VIEWER_GENERATION_NEXT[0] += 1
        _VIEWER_GENERATION[key] = _VIEWER_GENERATION_NEXT[0]

        total = sum(_tensor_nbytes(v) for v in _VIEWER_CACHE.values())
        while len(_VIEWER_CACHE) > 1 and (
            len(_VIEWER_CACHE) > _VIEWER_CACHE_MAX or total > _VIEWER_CACHE_MAX_BYTES
        ):
            evicted_key, evicted_val = _VIEWER_CACHE.popitem(last=False)
            _VIEWER_GENERATION.pop(evicted_key, None)
            total -= _tensor_nbytes(evicted_val)
            logger.debug(
                "[Radiance] Cache evicted node %s (entries=%d, %.1f MB in use, "
                "budget %.1f MB)", evicted_key, len(_VIEWER_CACHE),
                total / 1e6, _VIEWER_CACHE_MAX_BYTES / 1e6,
            )
            del evicted_val

        if oversize:
            logger.warning(
                "[Radiance] A single viewer frame set is %.1f MB, larger than the "
                "whole %.1f MB cache budget, so it is cached without a copy. Raise "
                "RADIANCE_VIEWER_CACHE_BYTES if delivery of this shot needs a private copy.",
                nbytes / 1e6, _VIEWER_CACHE_MAX_BYTES / 1e6,
            )


def _viewer_cache_generation(key: str) -> Optional[int]:
    """The generation of the frame set cached under ``key``, or None."""
    with _VIEWER_CACHE_LOCK:
        return _VIEWER_GENERATION.get(key) if key in _VIEWER_CACHE else None


def _viewer_cache_get_with_generation(key: str):
    """``(frames, generation)`` read together, or ``(None, None)``."""
    with _VIEWER_CACHE_LOCK:
        if key in _VIEWER_CACHE:
            _VIEWER_CACHE.move_to_end(key)
            return _VIEWER_CACHE[key], _VIEWER_GENERATION.get(key)
        return None, None


def _viewer_cache_get(key: str) -> Optional[torch.Tensor]:
    """Thread-safe LRU lookup."""
    with _VIEWER_CACHE_LOCK:
        if key in _VIEWER_CACHE:
            _VIEWER_CACHE.move_to_end(key)
            return _VIEWER_CACHE[key]
        return None


# What the cached Viewer pixels are: {instance_id: (encoding, colorspace)},
# encoding "srgb" (display-encoded ComfyUI IMAGE) or "linear", colorspace the
# Viewer's resolved input space. The delivery export reads it so the master is
# encoded from what the pixels really are (FIX-018).
_VIEWER_SOURCE: collections.OrderedDict = collections.OrderedDict()
_VIEWER_SOURCE_LOCK = threading.Lock()


def _viewer_source_set(key: str, encoding: str, colorspace: str) -> None:
    with _VIEWER_SOURCE_LOCK:
        _VIEWER_SOURCE[key] = (str(encoding), str(colorspace))
        _VIEWER_SOURCE.move_to_end(key)
        while len(_VIEWER_SOURCE) > 256:
            _VIEWER_SOURCE.popitem(last=False)


def _viewer_source_get(key: str):
    """``(encoding, colorspace)`` recorded for ``key``, or None."""
    with _VIEWER_SOURCE_LOCK:
        return _VIEWER_SOURCE.get(key)


# Stores active export progress: {instance_id: {current, total, status, message}}
_VIEWER_PROGRESS_MAX = 32
_VIEWER_PROGRESS: collections.OrderedDict = collections.OrderedDict()
_VIEWER_PROGRESS_LOCK = threading.Lock()


def _progress_set(key: str, value: Dict[str, Any]) -> None:
    """Thread-safe LRU insert into the progress store."""
    with _VIEWER_PROGRESS_LOCK:
        if key in _VIEWER_PROGRESS:
            _VIEWER_PROGRESS.move_to_end(key)
        _VIEWER_PROGRESS[key] = value
        while len(_VIEWER_PROGRESS) > _VIEWER_PROGRESS_MAX:
            _VIEWER_PROGRESS.popitem(last=False)


def _progress_get(key: str) -> Dict[str, Any]:
    """Thread-safe lookup; returns idle sentinel when key is absent."""
    with _VIEWER_PROGRESS_LOCK:
        return dict(_VIEWER_PROGRESS.get(key, {
            "current": 0, "total": 100, "status": "idle", "message": "Waiting...",
        }))
