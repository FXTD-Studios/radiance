"""Backward-compatible re-exports. New code should import from radiance.gpu."""
from radiance.gpu.ops import (
    gpu_laplacian_pyramid_blend,
    gpu_local_contrast,
)

__all__ = ["gpu_laplacian_pyramid_blend", "gpu_local_contrast"]
