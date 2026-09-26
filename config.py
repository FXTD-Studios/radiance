"""Backward-compatible re-exports. New code should import from radiance.config."""
from radiance.config.constants import (
    PACKAGE_NAME,
    PACKAGE_DISPLAY_NAME,
    VERSION,
    AUTHOR,
    WEB_DIRECTORY,
)
from radiance.config.env import configure_runtime_environment
from radiance.config.dependencies import (
    DependencySpec,
    CORE_DEPENDENCIES,
    OPTIONAL_DEPENDENCIES,
    module_available,
    missing_dependencies,
    validate_runtime_dependencies,
)

__all__ = [
    "PACKAGE_NAME",
    "PACKAGE_DISPLAY_NAME",
    "VERSION",
    "AUTHOR",
    "WEB_DIRECTORY",
    "DependencySpec",
    "CORE_DEPENDENCIES",
    "OPTIONAL_DEPENDENCIES",
    "configure_runtime_environment",
    "missing_dependencies",
    "module_available",
    "validate_runtime_dependencies",
]
