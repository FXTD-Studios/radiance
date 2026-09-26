"""Model loading, caching, and architecture detection."""
from radiance.model.cache import LRUCache, get_model_cache, clear_model_caches
from radiance.model.paths import (
    radiance_model_dirs,
    find_radiance_checkpoint,
    describe_search,
)
from radiance.model.detect import (
    detect_model_type,
    LATENT_CHANNELS,
    latent_format,
    CLIP_SLOT_ORDER,
    assemble_clip_paths,
    get_clip_type_enum,
    estimate_vram_usage,
)

__all__ = [
    "radiance_model_dirs",
    "find_radiance_checkpoint",
    "describe_search",
    "LRUCache",
    "get_model_cache",
    "clear_model_caches",
    "detect_model_type",
    "LATENT_CHANNELS",
    "latent_format",
    "CLIP_SLOT_ORDER",
    "assemble_clip_paths",
    "get_clip_type_enum",
    "estimate_vram_usage",
]
