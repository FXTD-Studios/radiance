"""Centralized error hierarchy and input validation."""
from __future__ import annotations

import functools
import logging
from typing import Any, Callable, Optional, TypeVar, Union

logger = logging.getLogger("radiance.core.errors")


class RadianceError(Exception):
    """Base exception for all Radiance errors."""

    def __init__(self, message: str, node_name: Optional[str] = None):
        self.node_name = node_name
        prefix = f"[{node_name}] " if node_name else "[Radiance] "
        super().__init__(f"{prefix}{message}")


class RadianceImageError(RadianceError):
    """Raised when image input is invalid or corrupted."""


class RadiancePathError(RadianceError):
    """Raised when file path is invalid or inaccessible."""


class RadianceGPUError(RadianceError):
    """Raised when GPU operation fails."""


class RadianceDependencyError(RadianceError):
    """Raised when a required dependency is missing."""


T = TypeVar("T")


def handle_node_errors(
    node_name: str, fallback: Optional[Any] = None, reraise: bool = True
) -> Callable:
    """Decorator for consistent error handling in ComfyUI nodes."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return func(*args, **kwargs)
            except RadianceError:
                if reraise:
                    raise
                logger.exception("[%s] Operation failed", node_name)
                return fallback
            except MemoryError as e:
                logger.error("[%s] Out of memory: %s", node_name, e)
                if reraise:
                    raise RadianceGPUError(
                        "Insufficient memory. Try reducing image size or batch count.",
                        node_name=node_name,
                    ) from e
                return fallback
            except FileNotFoundError as e:
                logger.error("[%s] File not found: %s", node_name, e)
                if reraise:
                    raise RadiancePathError(
                        f"File not found: {e.filename}", node_name=node_name
                    ) from e
                return fallback
            except PermissionError as e:
                logger.error("[%s] Permission denied: %s", node_name, e)
                if reraise:
                    raise RadiancePathError(
                        f"Permission denied: {e.filename}", node_name=node_name
                    ) from e
                return fallback
            except Exception as e:
                logger.exception("[%s] Unexpected error: %s", node_name, e)
                if reraise:
                    raise RadianceError(
                        f"Unexpected error: {e}", node_name=node_name
                    ) from e
                return fallback

        return wrapper

    return decorator


def validate_image_input(
    image: Any,
    node_name: str,
    min_dims: int = 2,
    max_dims: int = 4,
    require_batch: bool = True,
) -> None:
    import torch

    if image is None:
        raise RadianceImageError("No image input provided", node_name=node_name)

    if not isinstance(image, torch.Tensor):
        raise RadianceImageError(
            f"Expected torch.Tensor, got {type(image).__name__}", node_name=node_name
        )

    ndim = len(image.shape)
    if ndim < min_dims or ndim > max_dims:
        raise RadianceImageError(
            f"Expected {min_dims}-{max_dims}D tensor, got {ndim}D", node_name=node_name
        )

    if require_batch and ndim == 3:
        raise RadianceImageError(
            "Missing batch dimension. Expected shape (B, H, W, C)", node_name=node_name
        )

    if any(s == 0 for s in image.shape):
        raise RadianceImageError(
            f"Image has zero dimension: {image.shape}", node_name=node_name
        )


def validate_positive(
    value: Union[int, float], name: str, node_name: str, allow_zero: bool = False
) -> None:
    if allow_zero:
        if value < 0:
            raise RadianceError(
                f"{name} must be >= 0, got {value}", node_name=node_name
            )
    else:
        if value <= 0:
            raise RadianceError(f"{name} must be > 0, got {value}", node_name=node_name)


def validate_range(
    value: Union[int, float],
    name: str,
    node_name: str,
    min_val: Optional[Union[int, float]] = None,
    max_val: Optional[Union[int, float]] = None,
) -> None:
    if min_val is not None and value < min_val:
        raise RadianceError(
            f"{name} must be >= {min_val}, got {value}", node_name=node_name
        )
    if max_val is not None and value > max_val:
        raise RadianceError(
            f"{name} must be <= {max_val}, got {value}", node_name=node_name
        )
