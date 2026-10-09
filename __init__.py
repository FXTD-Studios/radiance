"""Radiance — HDR/VFX/Color pipeline for ComfyUI."""
from __future__ import annotations

import logging
import os
import sys

# OPENCV_IO_ENABLE_OPENEXR is forced once, by configure_runtime_environment()
# below (config/env.py RUNTIME_ENV_FORCED). OpenCV reads it at the first EXR
# read or write, not at import, and no EXR is touched before that call runs.


def _bootstrap_package_context() -> None:
    """Make relative imports reliable when ComfyUI loads this file directly."""

    global __package__, __path__

    if not __package__:
        __package__ = "radiance"
        __path__ = [os.path.dirname(os.path.abspath(__file__))]

    sys.modules.setdefault("radiance", sys.modules.get(__name__, type(sys)(__name__)))


_bootstrap_package_context()

from .config.constants import (
    AUTHOR,
    EXPECTED_MIN_NODE_COUNT,
    VERSION,
    WEB_DIRECTORY,
)
from .config.dependencies import (
    CORE_DEPENDENCIES,
    missing_dependencies,
    validate_runtime_dependencies,
)
from .config.env import configure_runtime_environment
from .core.logging import register_run_grouping, setup_radiance_logging
from .nodes.registry import NodeLoadResult, NodeModuleSpec, load_node_mappings

logger = setup_radiance_logging()


def _load_comfyui_nodes() -> NodeLoadResult:
    """Load the organized node catalog."""

    # One entry point, not two. `.nodes_radiance_viewer` was a second spec
    # publishing RadianceViewer alongside `radiance.nodes.monitor` — the one
    # genuine dual-publish the legacy layer had, and a live instance of the
    # double import that `_radiance_route_once` exists to survive. The monitor
    # group publishes the viewer; nothing else needs to.
    entrypoint_modules = (
        NodeModuleSpec(".nodes", package=__name__, required=True),
    )
    result = load_node_mappings(
        entrypoint_modules,
        logger=logger,
        context="Radiance entry point",
    )

    # Fold in the catalog's own group failures. Without this the entry point
    # sees only its two top-level specs, so a group that dropped 18 nodes on an
    # optional-dependency error looked like a clean load from up here.
    nested = getattr(sys.modules.get(f"{__name__}.nodes"), "NODE_LOAD_FAILURES", ())
    if nested:
        result = NodeLoadResult(
            class_mappings=result.class_mappings,
            display_name_mappings=result.display_name_mappings,
            loaded_modules=result.loaded_modules,
            failures=tuple(result.failures) + tuple(nested),
        )
    return result


def report_node_load_health(
    load_result: NodeLoadResult,
    expected_minimum: int = EXPECTED_MIN_NODE_COUNT,
    log: "logging.Logger | None" = None,
) -> bool:
    """Log the outcome of a registry load at a severity that matches reality.

    The startup banner used to be an unconditional INFO -- "successfully loaded
    N nodes" -- with no comparison against anything. A group import failure
    drops every node in that group, so an 88% shortfall printed the same
    reassuring line as a clean start, and the only trace was a WARNING several
    hundred lines earlier that most users never scroll back to.

    Returns True when the load looks healthy.
    """

    active = log or logger
    loaded = len(load_result.class_mappings)
    healthy = True

    if load_result.failures:
        healthy = False
        active.error(
            "Radiance: %d node module(s) failed to import; every node they "
            "export is missing from ComfyUI.",
            len(load_result.failures),
        )
        for failure in load_result.failures:
            active.error(
                "  - %s: %s: %s",
                failure.source.label,
                type(failure.error).__name__,
                failure.error,
            )

    if loaded < expected_minimum:
        healthy = False
        shortfall = expected_minimum - loaded
        active.error(
            "Radiance: loaded %d of at least %d expected nodes (v%s) - %d "
            "missing (%.0f%% of the catalog). This is a failed start, not a "
            "small one: check the import errors above, then re-run with "
            "RADIANCE_LOG_LEVEL=DEBUG for tracebacks.",
            loaded,
            expected_minimum,
            __version__,
            shortfall,
            100.0 * shortfall / max(expected_minimum, 1),
        )
    else:
        active.info(
            "Radiance: successfully loaded %d nodes (v%s)", loaded, __version__
        )

    return healthy


def check_runtime_dependencies(log: "logging.Logger | None" = None) -> tuple:
    """Run the dependency check and act on its result; never raises.

    Returns the display names of missing required dependencies (empty when
    all are present or the check itself failed). Until 4.0 the result was
    discarded, and an exception inside the check stopped Radiance loading.
    ComfyUI still starts either way: nodes that need a missing package fail
    on their own import or use, and the node load report says which.
    """
    active = log or logger
    try:
        if validate_runtime_dependencies(active):
            return ()
        missing = missing_dependencies(CORE_DEPENDENCIES)
    except Exception as exc:  # noqa: BLE001 - a broken check must not block node registration
        active.warning(
            "Radiance: the runtime dependency check failed (%s: %s); loading nodes anyway.",
            type(exc).__name__, exc,
        )
        return ()
    names = tuple(spec.display_name for spec in missing)
    if names:
        active.warning(
            "Radiance: required dependencies missing: %s. Nodes that need them will fail to "
            "load or run. Install them (%s) and restart ComfyUI.",
            ", ".join(names),
            "; ".join(spec.install_hint for spec in missing),
        )
    return names


configure_runtime_environment()
_MISSING_REQUIRED_DEPENDENCIES = check_runtime_dependencies(logger)


def _configure_ocio() -> None:
    """Automatic OCIO: $OCIO if set, else the ACES studio config. No setup."""
    try:
        from .color.ocio_setup import configure_ocio, summary
        state = configure_ocio()
        (logger.info if state.get("configured") else logger.warning)("[Radiance OCIO] %s", summary())
    except Exception as exc:  # noqa: BLE001 - never block node registration
        logger.warning("[Radiance OCIO] automatic setup failed: %s", exc)


_configure_ocio()

_LOAD_RESULT = _load_comfyui_nodes()
NODE_CLASS_MAPPINGS = _LOAD_RESULT.class_mappings
NODE_DISPLAY_NAME_MAPPINGS = _LOAD_RESULT.display_name_mappings

__version__ = VERSION
__author__ = AUTHOR
__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
    "report_node_load_health",
]

report_node_load_health(_LOAD_RESULT)
if _MISSING_REQUIRED_DEPENDENCIES:
    # Repeated after the load banner: the dependency table is printed
    # hundreds of lines earlier, where it is easy to miss.
    logger.warning(
        "Radiance: started without required dependencies: %s (see the dependency table above).",
        ", ".join(_MISSING_REQUIRED_DEPENDENCIES),
    )
logger.debug("Radiance Viewer JavaScript extension enabled")

# Mark each prompt run with a console separator (no-op if the hook is absent).
register_run_grouping()
