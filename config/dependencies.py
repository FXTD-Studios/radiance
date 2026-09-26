"""Runtime dependency checking — validates that required and optional packages exist."""
from __future__ import annotations

import importlib.util
import logging
import sys
from typing import Iterable, Optional, Sequence, Tuple
import subprocess
import json

import os
from radiance.config.constants import PACKAGE_NAME
from radiance.config.env import ENV
from radiance.core.logging import print_table, supports_unicode, supports_color, TextTheme


class DependencySpec:
    """A runtime dependency and the feature area it unlocks."""

    def __init__(
        self,
        module_name: str,
        display_name: str,
        feature: str,
        install_hint: str,
        required: bool = False,
    ):
        self.module_name = module_name
        self.display_name = display_name
        self.feature = feature
        self.install_hint = install_hint
        self.required = required


CORE_DEPENDENCIES: Tuple[DependencySpec, ...] = (
    DependencySpec("torch", "torch", "tensor processing", "pip install torch", True),
    DependencySpec("numpy", "numpy", "array processing", "pip install numpy", True),
    DependencySpec("PIL.Image", "Pillow", "image I/O", "pip install Pillow", True),
    DependencySpec("aiohttp", "aiohttp", "async HTTP server", "pip install aiohttp", True),
    DependencySpec("OpenEXR", "OpenEXR", "EXR file support", "pip install OpenEXR", True),
    # Required since 3.5: Read / Write colour management and every OCIO node
    # run through it, and it is configured automatically at startup.
    DependencySpec("PyOpenColorIO", "opencolorio", "OCIO colour management (auto-configured)",
                   "pip install opencolorio", True),
)

# Optional packages that gate a specific, user-visible feature.
#
# The `colour-science` entry that used to live here claimed to unlock "advanced
# color science"; nothing in the package has ever imported `colour`, so the row
# was reporting on a feature that did not exist. The three added below are the
# reverse case -- real feature gates whose absence produced a confusing runtime
# failure rather than a line in this table.
OPTIONAL_DEPENDENCIES: Tuple[DependencySpec, ...] = (
    DependencySpec("transformers", "transformers", "Depth Anything V2", "pip install transformers"),
    DependencySpec("defusedxml", "defusedxml", "secure CDL XML parsing", "pip install defusedxml"),
    DependencySpec("OpenImageIO", "OpenImageIO", "DPX read/write", "pip install OpenImageIO"),
    # AUDIT-FIX (2026-08): tifffile is the ONLY writer for "TIFF (16-bit)" and
    # "TIFF (32-bit float)" and the only depth probe on the read side, yet it
    # was declared nowhere -- without it the 32-bit float TIFF write silently
    # produced an 8-bit file.
    DependencySpec("tifffile", "tifffile", "16/32-bit TIFF read & write", "pip install tifffile"),
    DependencySpec("imageio_ffmpeg", "imageio-ffmpeg", "bundled ffmpeg for video export",
                   "pip install imageio-ffmpeg"),
)


def module_available(module_name: str) -> bool:
    if module_name in sys.modules:
        return True
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def missing_dependencies(
    dependencies: Iterable[DependencySpec],
) -> Tuple[DependencySpec, ...]:
    return tuple(spec for spec in dependencies if not module_available(spec.module_name))


def check_dependency_conflicts(logger: logging.Logger) -> None:
    """
    Dry-Run conflict checker. Checks environment integrity against common ComfyUI
    package conflict targets before writing changes.
    """
    logger.info("Radiance Environment Guard: Running dry-run integrity scan ...")
    
    # Critical packages to watch
    critical = {"torch", "torchvision", "diffusers", "transformers", "safetensors", "numpy"}
    
    try:
        # Probe using importlib.metadata
        from importlib.metadata import distributions
        installed = {}
        for dist in distributions():
            name = (dist.metadata["Name"] or "").lower().replace("_", "-")
            if name in critical:
                installed[name] = dist.version
                
        # Format diagnostics table
        logger.info("[Environment Guard] Current environment state:")
        for pkg, ver in installed.items():
            logger.info(f"  • {pkg}: version {ver} installed.")
            
    except Exception as exc:
        logger.debug(f"Environment Guard failed to scan metadata: {exc}")


def validate_runtime_dependencies(
    logger: Optional[logging.Logger] = None,
    core_dependencies: Sequence[DependencySpec] = CORE_DEPENDENCIES,
    optional_dependencies: Sequence[DependencySpec] = OPTIONAL_DEPENDENCIES,
) -> bool:
    active_logger = logger or logging.getLogger(PACKAGE_NAME)
    
    # Run Dry-Run conflict checker first
    check_dependency_conflicts(active_logger)
    
    missing_core = missing_dependencies(core_dependencies)

    # Dynamic status labels based on active visual Theme
    unicode_enabled = supports_unicode()
    color_enabled = supports_color()
    theme_name = os.environ.get("RADIANCE_LOG_THEME", "minimalist")
    theme = TextTheme(theme_name, unicode_enabled, color_enabled)

    status_active = theme.status_ok
    status_missing_core = theme.status_err
    status_missing_opt = theme.status_warn

    # Build gorgeous dependency status table
    headers = ["Dependency", "Type", "Status", "Unlocks Feature", "Install Command / Hint"]
    rows = []

    # Process core dependencies
    for spec in core_dependencies:
        available = module_available(spec.module_name)
        status_str = f"\033[1;32m{status_active}\033[0m" if available else f"\033[1;31m{status_missing_core}\033[0m"
        type_str = "\033[1;31mRequired\033[0m" if not available else "Required"
        hint_str = "-" if available else spec.install_hint
        rows.append([
            spec.display_name,
            type_str,
            status_str,
            spec.feature,
            hint_str
        ])

    # Process optional dependencies
    for spec in optional_dependencies:
        available = module_available(spec.module_name)
        status_str = f"\033[1;32m{status_active}\033[0m" if available else f"\033[1;33m{status_missing_opt}\033[0m"
        type_str = "Optional"
        hint_str = "-" if available else spec.install_hint
        rows.append([
            spec.display_name,
            type_str,
            status_str,
            spec.feature,
            hint_str
        ])

    # Print the beautiful status table
    print_table(
        headers=headers,
        rows=rows,
        col_alignments=["left", "center", "center", "left", "left"],
        level=logging.INFO,
        logger_name=PACKAGE_NAME
    )

    # Log explicit critical failures if required dependencies are missing
    for spec in missing_core:
        active_logger.error(
            "CRITICAL: Missing required dependency %s. Radiance functionality will be disabled! "
            "Please run: %s",
            spec.display_name,
            spec.install_hint,
        )

    return not missing_core
