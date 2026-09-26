"""Lightweight opt-in performance timing helpers for Radiance nodes."""
from __future__ import annotations

import os
import time
from typing import Any, Optional


def profiling_enabled() -> bool:
    return os.environ.get("RADIANCE_PROFILE", "").strip().lower() in {"1", "true", "yes", "on"}


def perf_start(device: Optional[Any] = None) -> Optional[float]:
    if not profiling_enabled():
        return None
    _sync_cuda(device)
    return time.perf_counter()


def perf_finish(logger: Any, label: str, start: Optional[float], device: Optional[Any] = None) -> None:
    if start is None:
        return
    _sync_cuda(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    logger.info("[Radiance PERF] %s %.2f ms", label, elapsed_ms)


def _sync_cuda(device: Optional[Any]) -> None:
    if device is None:
        return
    try:
        import torch

        if torch.cuda.is_available() and getattr(device, "type", None) == "cuda":
            torch.cuda.synchronize(device)
    except Exception:
        return
