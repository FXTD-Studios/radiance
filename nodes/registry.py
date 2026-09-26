"""Shared node registry loading utilities."""
from __future__ import annotations

from dataclasses import dataclass
import importlib
import logging
from collections.abc import Mapping
from types import ModuleType
from typing import Any, Dict, Iterable, Optional, Tuple

from radiance.nodes.branding import apply_radiance_branding

NODE_CLASS_MAPPINGS_ATTR = "NODE_CLASS_MAPPINGS"
NODE_DISPLAY_NAME_MAPPINGS_ATTR = "NODE_DISPLAY_NAME_MAPPINGS"


@dataclass(frozen=True)
class NodeModuleSpec:
    """Import target for a module that may export ComfyUI node mappings."""

    import_path: str
    package: Optional[str] = None
    required: bool = False

    @property
    def label(self) -> str:
        if self.package:
            return f"{self.package}:{self.import_path}"
        return self.import_path


@dataclass(frozen=True)
class NodeLoadFailure:
    """A failed module import captured during registry loading."""

    source: NodeModuleSpec
    error: Exception


@dataclass(frozen=True)
class NodeLoadResult:
    """Merged node mappings and diagnostics from a registry load."""

    class_mappings: Dict[str, Any]
    display_name_mappings: Dict[str, Any]
    loaded_modules: Tuple[str, ...]
    failures: Tuple[NodeLoadFailure, ...]


def package_root_from_node_package(package_name: str) -> Optional[str]:
    """
    Resolve the package root from a node package name.

    Examples:
      radiance.nodes.color -> radiance
      nodes.color          -> None
    """

    marker = ".nodes"
    if marker not in package_name:
        return None

    root = package_name.split(marker, 1)[0]
    return root or None


def root_module(
    module_name: str,
    package_root: Optional[str],
    required: bool = False,
) -> NodeModuleSpec:
    """Build an import spec for a legacy flat module at the package root."""

    if package_root and not module_name.startswith(f"{package_root}."):
        module_name = f"{package_root}.{module_name}"
    return NodeModuleSpec(module_name, required=required)


def root_modules(
    module_names: Iterable[str],
    package_root: Optional[str],
    required: bool = False,
) -> Tuple[NodeModuleSpec, ...]:
    """Build import specs for several package-root modules."""

    return tuple(
        root_module(module_name, package_root=package_root, required=required)
        for module_name in module_names
    )


def child_module(
    module_name: str,
    package: str,
    required: bool = False,
) -> NodeModuleSpec:
    """Build a relative import spec for a child module/package."""

    import_path = module_name if module_name.startswith(".") else f".{module_name}"
    return NodeModuleSpec(import_path, package=package, required=required)


def child_modules(
    module_names: Iterable[str],
    package: str,
    required: bool = False,
) -> Tuple[NodeModuleSpec, ...]:
    """Build relative import specs for several child modules/packages."""

    return tuple(
        child_module(module_name, package=package, required=required)
        for module_name in module_names
    )


def load_node_mappings(
    sources: Iterable[NodeModuleSpec],
    logger: Optional[logging.Logger] = None,
    context: str = "Radiance nodes",
    fail_fast: bool = False,
) -> NodeLoadResult:
    """Import node modules and merge their exported mapping dictionaries."""

    active_logger = logger or logging.getLogger("radiance.nodes")
    class_mappings: Dict[str, Any] = {}
    display_name_mappings: Dict[str, Any] = {}
    loaded_modules = []
    failures = []

    for source in sources:
        try:
            module = importlib.import_module(source.import_path, package=source.package)
        except Exception as exc:  # pragma: no cover - exercised by optional deps
            failures.append(NodeLoadFailure(source=source, error=exc))
            _log_import_failure(active_logger, context, source, exc)
            if fail_fast or source.required:
                raise
            continue

        _merge_module_mappings(module, class_mappings, display_name_mappings, active_logger)
        loaded_modules.append(module.__name__)

    apply_radiance_branding(class_mappings, display_name_mappings)

    return NodeLoadResult(
        class_mappings=class_mappings,
        display_name_mappings=display_name_mappings,
        loaded_modules=tuple(loaded_modules),
        failures=tuple(failures),
    )


def load_node_group(
    package_name: str,
    root_source_modules: Iterable[str] = (),
    child_source_modules: Iterable[str] = (),
    logger: Optional[logging.Logger] = None,
    required: bool = False,
) -> NodeLoadResult:
    """Load a node group from package-root modules and local child modules."""

    sources = root_modules(
        root_source_modules,
        package_root=package_root_from_node_package(package_name),
        required=required,
    ) + child_modules(
        child_source_modules,
        package=package_name,
        required=required,
    )
    return load_node_mappings(sources, logger=logger, context=package_name)


def _merge_module_mappings(
    module: ModuleType,
    class_mappings: Dict[str, Any],
    display_name_mappings: Dict[str, Any],
    logger: logging.Logger,
) -> None:
    """Merge mapping dictionaries from a module into the aggregate registry."""

    classes = getattr(module, NODE_CLASS_MAPPINGS_ATTR, {})
    display_names = getattr(module, NODE_DISPLAY_NAME_MAPPINGS_ATTR, {})

    if not isinstance(classes, Mapping):
        logger.warning("%s.%s is not a mapping", module.__name__, NODE_CLASS_MAPPINGS_ATTR)
        return
    if not isinstance(display_names, Mapping):
        logger.warning("%s.%s is not a mapping", module.__name__, NODE_DISPLAY_NAME_MAPPINGS_ATTR)
        display_names = {}

    duplicate_keys = sorted(set(class_mappings).intersection(classes))
    if duplicate_keys:
        logger.debug(
            "%s overrides %d existing node registrations: %s",
            module.__name__,
            len(duplicate_keys),
            ", ".join(duplicate_keys[:10]),
        )

    class_mappings.update(classes)
    display_name_mappings.update(display_names)


def _log_import_failure(
    logger: logging.Logger,
    context: str,
    source: NodeModuleSpec,
    error: Exception,
) -> None:
    """Log import failures consistently across entry point and node groups.

    BUG FIX: optional-module failures used to log at DEBUG, which is
    invisible in ComfyUI's default console output. A single broken
    submodule inside a node *group* (e.g. radiance.nodes.hdr, which bundles
    ~18 nodes behind one import) silently dropped every node in that group
    with no visible trace — surfacing as "missing node" reports with no
    clue why. WARNING keeps startup non-fatal (required=False groups still
    don't crash ComfyUI) while making the failure and the exact module that
    failed visible by default.
    """

    if source.required:
        logger.error("%s: failed to load required module %s: %s", context, source.label, error)
    else:
        logger.warning("%s: skipping optional module %s (%s)", context, source.label, error)
    logger.debug("Import failure details for %s", source.label, exc_info=True)


__all__ = [
    "NODE_CLASS_MAPPINGS_ATTR",
    "NODE_DISPLAY_NAME_MAPPINGS_ATTR",
    "NodeLoadFailure",
    "NodeLoadResult",
    "NodeModuleSpec",
    "child_module",
    "child_modules",
    "load_node_group",
    "load_node_mappings",
    "package_root_from_node_package",
    "root_module",
    "root_modules",
]
