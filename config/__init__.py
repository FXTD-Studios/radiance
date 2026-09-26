"""Configuration package — single source of truth for all Radiance settings."""
from radiance.config.constants import (
    PACKAGE_NAME,
    PACKAGE_DISPLAY_NAME,
    VERSION,
    AUTHOR,
    WEB_DIRECTORY,
)
from radiance.config.env import (
    configure_runtime_environment,
    get_env,
    get_env_int,
    get_env_bool,
    ENV,
)
from radiance.config.dependencies import (
    DependencySpec,
    CORE_DEPENDENCIES,
    OPTIONAL_DEPENDENCIES,
    module_available,
    missing_dependencies,
    validate_runtime_dependencies,
)
from radiance.config.model_map import (
    RADIANCE_MODEL_MAP,
    CHECKPOINT_PRESETS,
)

__all__ = [
    "PACKAGE_NAME",
    "PACKAGE_DISPLAY_NAME",
    "VERSION",
    "AUTHOR",
    "WEB_DIRECTORY",
    "configure_runtime_environment",
    "get_env",
    "get_env_int",
    "get_env_bool",
    "ENV",
    "DependencySpec",
    "CORE_DEPENDENCIES",
    "OPTIONAL_DEPENDENCIES",
    "module_available",
    "missing_dependencies",
    "validate_runtime_dependencies",
    "RADIANCE_MODEL_MAP",
    "CHECKPOINT_PRESETS",
]
