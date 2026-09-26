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
    try:
        if isinstance(value, torch.Tensor):
            value = value.detach().to("cpu", copy=True)
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("[Radiance] Could not copy viewer frame to CPU: %s", exc)

    nbytes = _tensor_nbytes(value)
    with _VIEWER_CACHE_LOCK:
        if key in _VIEWER_CACHE:
            _VIEWER_CACHE.move_to_end(key)
        _VIEWER_CACHE[key] = value

        total = sum(_tensor_nbytes(v) for v in _VIEWER_CACHE.values())
        while len(_VIEWER_CACHE) > 1 and (
            len(_VIEWER_CACHE) > _VIEWER_CACHE_MAX or total > _VIEWER_CACHE_MAX_BYTES
        ):
            evicted_key, evicted_val = _VIEWER_CACHE.popitem(last=False)
            total -= _tensor_nbytes(evicted_val)
            logger.debug(
                "[Radiance] Cache evicted node %s (entries=%d, %.1f MB in use, "
                "budget %.1f MB)", evicted_key, len(_VIEWER_CACHE),
                total / 1e6, _VIEWER_CACHE_MAX_BYTES / 1e6,
            )
            del evicted_val

        if nbytes > _VIEWER_CACHE_MAX_BYTES:
            logger.warning(
                "[Radiance] A single viewer frame set is %.1f MB, larger than the "
                "whole %.1f MB cache budget. Raise RADIANCE_VIEWER_CACHE_BYTES if "
                "delivery of this shot needs the cache.",
                nbytes / 1e6, _VIEWER_CACHE_MAX_BYTES / 1e6,
            )


def _viewer_cache_get(key: str) -> Optional[torch.Tensor]:
    """Thread-safe LRU lookup."""
    with _VIEWER_CACHE_LOCK:
        if key in _VIEWER_CACHE:
            _VIEWER_CACHE.move_to_end(key)
            return _VIEWER_CACHE[key]
        return None


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
