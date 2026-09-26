"""LRU model cache — prevents redundant model loading across node invocations."""
from __future__ import annotations

import logging
from collections import OrderedDict

from radiance.config.env import ENV, get_env_int

logger = logging.getLogger("radiance.model.cache")

_DEFAULT_CACHE_SIZE = get_env_int(ENV.RADIANCE_CACHE_SIZE, 2)


class LRUCache:
    """Least-Recently-Used cache with O(1) get/put/evict."""

    def __init__(self, max_size: int = _DEFAULT_CACHE_SIZE):
        self._cache: OrderedDict[str, object] = OrderedDict()
        self._max_size = max_size

    def get(self, key: str):
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key: str, obj) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            # `while ... and self._cache` rather than a single `if`: with
            # RADIANCE_CACHE_SIZE=0 the old form called popitem() on an empty
            # OrderedDict and raised KeyError out of the loader, failing the
            # whole render. Size 0 now simply means "do not cache".
            while len(self._cache) >= self._max_size and self._cache:
                evicted, victim = self._cache.popitem(last=False)
                self._on_evict(evicted, victim)
            if self._max_size <= 0:
                return
        self._cache[key] = obj

    def _on_evict(self, key: str, obj) -> None:
        logger.info("Cache evicted: %s", key)

    def pop(self, key: str):
        """Explicitly drop an entry, running eviction handling. Returns it."""
        obj = self._cache.pop(key, None)
        if obj is not None:
            self._on_evict(key, obj)
        return obj

    def has(self, key: str) -> bool:
        return key in self._cache

    def clear(self) -> None:
        for key, obj in list(self._cache.items()):
            self._on_evict(key, obj)
        self._cache.clear()
        logger.info("Model cache cleared")

    def __getitem__(self, key: str):
        """Dict-style read, so a `cache[key]` left over from the dict this
        class replaced raises KeyError rather than TypeError.

        It mattered: nodes/upscale/upscale.py did `_FACE_MODEL_CACHE[key]`
        after the dict -> GPUModelCache refactor. TypeError was swallowed by a
        broad `except Exception` that logged at DEBUG, so RetinaFace detection
        silently never ran and face restore fell through to the Haar cascade
        while still reporting success.
        """
        value = self.get(key)
        if value is None and not self.has(key):
            raise KeyError(key)
        return value

    def __setitem__(self, key: str, obj) -> None:
        self.put(key, obj)

    def __contains__(self, key: str) -> bool:
        return key in self._cache

    def __len__(self) -> int:
        return len(self._cache)

    @property
    def size(self) -> int:
        return len(self._cache)


class GPUModelCache(LRUCache):
    """
    LRU cache for models that live in VRAM.

    Radiance had twelve module-level dicts holding loaded upscalers, face
    restorers, depth models and diffusion pipelines, none of which ever evicted
    anything -- a session that touched several upscale tiers pinned gigabytes
    for the lifetime of the ComfyUI process, and ComfyUI's own model_management
    could not reclaim it because it did not own those objects.

    Eviction here moves the module back to CPU before dropping the reference,
    so the VRAM is actually released rather than merely unreferenced.
    """

    def _on_evict(self, key: str, obj) -> None:
        try:
            to_cpu = getattr(obj, "to", None)
            if callable(to_cpu):
                obj.to("cpu")
        except Exception as exc:  # pragma: no cover - best effort
            logger.debug("Could not move evicted model %s to CPU: %s", key, exc)
        logger.info("GPU model cache evicted: %s", key)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover
            pass


_unet_cache = LRUCache()
_clip_cache = LRUCache()
_vae_cache = LRUCache()
_audio_vae_cache = LRUCache()


def get_model_cache(kind: str) -> LRUCache:
    """Return one of the three singleton caches."""
    return {"unet": _unet_cache, "clip": _clip_cache, "vae": _vae_cache}.get(kind, _unet_cache)


def clear_model_caches() -> None:
    _unet_cache.clear()
    _clip_cache.clear()
    _vae_cache.clear()
