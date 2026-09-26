"""Core shared utilities for Radiance.

The core package intentionally exposes a broad convenience API while keeping
imports lazy. Startup modules such as logging and dependency checks must not
pull in tensor libraries before dependency validation has a chance to run.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    "RadianceError": "radiance.core.errors",
    "RadianceImageError": "radiance.core.errors",
    "RadiancePathError": "radiance.core.errors",
    "RadianceGPUError": "radiance.core.errors",
    "RadianceDependencyError": "radiance.core.errors",
    "handle_node_errors": "radiance.core.errors",
    "validate_image_input": "radiance.core.errors",
    "validate_positive": "radiance.core.errors",
    "validate_range": "radiance.core.errors",
    "setup_radiance_logging": "radiance.core.logging",
    "print_box": "radiance.core.logging",
    "print_table": "radiance.core.logging",
    "TextTheme": "radiance.core.logging",
    "supports_color": "radiance.core.logging",
    "ensure_4d": "radiance.core.tensor.contract",
    "ensure_5d": "radiance.core.tensor.contract",
    "tensor_to_numpy_float32": "radiance.core.tensor.convert",
    "numpy_to_tensor_float32": "radiance.core.tensor.convert",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module 'radiance.core' has no attribute {name!r}")

    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


__all__ = list(_EXPORT_MODULES)
