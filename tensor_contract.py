"""Backward-compatible re-exports. New code should import from radiance.core.tensor."""
from radiance.core.tensor.contract import ensure_4d, ensure_5d

__all__ = ["ensure_4d", "ensure_5d"]
