"""Declarative catalog of node groups exposed by Radiance."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from radiance.config.env import ENV, get_env_bool
from radiance.nodes.registry import NodeModuleSpec


@dataclass(frozen=True)
class NodeGroupSpec:
    """A loadable node group and the feature flag that controls it."""

    module_path: str
    env_flag: Optional[str] = None

    def is_enabled(self) -> bool:
        if self.env_flag is None:
            return True
        return get_env_bool(self.env_flag, False)

    def as_module_spec(self) -> NodeModuleSpec:
        return NodeModuleSpec(self.module_path)


# Node GROUPS only, never implementation packages.
#
# `radiance.image`, `radiance.hdr`, `radiance.film` and `radiance.color` each
# declare a full NODE_CLASS_MAPPINGS in their __init__.py, and for a long time
# nothing in the load chain read any of them: they are not listed here, and
# `fold_in_module_nodes` cannot reach them either, because its `_leaf_modules`
# walks only the __path__ of a package already in this tuple and skips
# sub-packages. Twenty-three finished nodes were invisible for that reason
# until 2026-09-18.
#
# Adding them here is the obvious fix and it is the wrong one: a package loaded
# as a group publishes everything it declares, so a rival class under a key
# another group ships (as `radiance.film` had for RadianceFilmGrain and
# RadianceMotionBlur until 3.5.0) would silently replace a shipping node.
#
# The registration layer is `radiance/nodes/<group>/__init__.py`: it imports
# the implementation classes it wants and names them, one key at a time. See
# nodes/color/__init__.py, which has published radiance.color.lut that way
# since 2026-08.
NODE_GROUPS: Tuple[NodeGroupSpec, ...] = (
    NodeGroupSpec("radiance.nodes.color"),
    NodeGroupSpec("radiance.nodes.hdr"),
    NodeGroupSpec("radiance.nodes.io"),
    NodeGroupSpec("radiance.nodes.vfx"),
    NodeGroupSpec("radiance.nodes.pipeline"),
    NodeGroupSpec("radiance.nodes.monitor"),
    NodeGroupSpec("radiance.nodes.upscale"),
    NodeGroupSpec("radiance.nodes.video"),
    NodeGroupSpec("radiance.nodes.ai"),
    NodeGroupSpec("radiance.nodes.generate"),
)


def enabled_node_group_specs() -> Tuple[NodeModuleSpec, ...]:
    """Return import specs for the currently enabled node groups."""

    return tuple(group.as_module_spec() for group in NODE_GROUPS if group.is_enabled())


__all__ = ["NODE_GROUPS", "NodeGroupSpec", "enabled_node_group_specs"]
