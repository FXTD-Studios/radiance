"""Backward-compatible re-exports. New code should import from radiance.core.system."""
from radiance.core.system.secret_utils import resolve_secret, normalize_env_name

__all__ = ["resolve_secret", "normalize_env_name"]
