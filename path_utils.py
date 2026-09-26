"""Backward-compatible re-exports. New code should import from radiance.core.system."""
from radiance.core.system.path_utils import (
    safe_join,
    validate_output_path,
    get_safe_output_dir,
    get_safe_input_path,
    get_next_index,
    strip_path_quotes,
    resolve_output_path,
    resolve_input_path,
)

__all__ = ["safe_join", "validate_output_path", "get_safe_output_dir", "get_safe_input_path", "get_next_index", "strip_path_quotes", "resolve_output_path", "resolve_input_path"]
