# ============================================================
# FXTD STUDIOS — Radiance
# nodes_character.py  —  Character Consistency System
# ============================================================
# Consolidation (v3.0):  6 nodes → 3 multi-mode nodes
#
#   RadianceCharacterAnchor  (mode: Anchor | Enforce)
#     Anchor  — encode reference image → character profile
#     Enforce — inject profile embedding into CONDITIONING
#
#   RadianceCharacterChecker  (mode: Check | Blend)
#     Check   — compare frame against anchor; report cosine similarity
#     Blend   — mix two anchors into a composite profile
#
#   RadianceCharacterGallery  (mode: Gallery | Timeline)
#     Gallery  — list / load .npz profiles from disk + registry
#     Timeline — summarise per-frame consistency scores across a shot
#
# Backward-compat aliases are exported for all 6 retired class names.
#
# Embedding backends (tried in priority order)
# -------------------------------------------
#   1. ComfyUI CLIP model (passed in as optional input)
#   2. transformers CLIP  (openai/clip-vit-base-patch32)
#   3. Colour HSV histogram  (36-D, no dependencies)
# ============================================================

from radiance.config.constants import VERSION as __version__  # noqa: E402

import io
import json
import logging
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("radiance")

# ---------------------------------------------------------------------------
# Lazy optional imports
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def _has(pkg):
    import importlib.util
    return importlib.util.find_spec(pkg) is not None


HAS_TRANSFORMERS = _has("transformers")

# ---------------------------------------------------------------------------
# In-memory character registry (survives within a session)
# ---------------------------------------------------------------------------
_REGISTRY: Dict[str, Dict] = {}   # name → profile dict


# ===========================================================================
# Embedding helpers
# ===========================================================================

def _tensor_to_pil(image_tensor) -> "PILImage.Image":
    """Convert ComfyUI IMAGE tensor [1,H,W,3] or [H,W,3] to PIL."""
    if not HAS_TORCH or not HAS_PIL:
        raise RuntimeError("torch and Pillow are required for image processing")
    t = image_tensor
    if t.dim() == 4:
        t = t[0]
    arr = (t.clamp(0, 1).cpu().numpy() * 255).astype("uint8")
    return PILImage.fromarray(arr, mode="RGB")


def _hsv_histogram(pil_img, bins: int = 12) -> "np.ndarray":
    """Fast 36-D HSV histogram embedding — no ML dependencies."""
    import numpy as _np
    img_rgb = pil_img.convert("RGB").resize((128, 128))
    arr = _np.array(img_rgb, dtype=_np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    maxc = _np.maximum(_np.maximum(r, g), b)
    minc = _np.minimum(_np.minimum(r, g), b)
    v = maxc
    s = _np.where(maxc > 1e-7, (maxc - minc) / maxc, 0.0)
    delta = maxc - minc + 1e-8
    h = _np.where(maxc == r, (g - b) / delta % 6,
        _np.where(maxc == g, (b - r) / delta + 2, (r - g) / delta + 4)) / 6.0
    h = h % 1.0
    hist_h, _ = _np.histogram(h.ravel(), bins=bins, range=(0, 1))
    hist_s, _ = _np.histogram(s.ravel(), bins=bins, range=(0, 1))
    hist_v, _ = _np.histogram(v.ravel(), bins=bins, range=(0, 1))
    feat = _np.concatenate([hist_h, hist_s, hist_v]).astype(_np.float32)
    norm = _np.linalg.norm(feat)
    return feat / (norm + 1e-8)


def _clip_embed_transformers(pil_img) -> "np.ndarray":
    """512-D CLIP image embedding via HuggingFace transformers."""
    from transformers import CLIPProcessor, CLIPModel  # type: ignore
    import numpy as _np
    from radiance.core.consent import downloads_allowed
    model_id = "openai/clip-vit-base-patch32"
    # Downloads on first use unless RADIANCE_ALLOW_DOWNLOADS=0 or an offline
    # flag is set; then only a cached copy is used (the caller falls back to
    # colour histograms when this raises).
    local_only = not downloads_allowed()
    revision = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"   # pinned commit of openai/clip-vit-base-patch32
    model     = CLIPModel.from_pretrained(model_id, revision=revision, local_files_only=local_only)
    processor = CLIPProcessor.from_pretrained(model_id, revision=revision, local_files_only=local_only)
    inputs = processor(images=pil_img, return_tensors="pt")
    with (torch.no_grad() if HAS_TORCH else _dummy_ctx()):
        feats = model.get_image_features(**inputs)
    feat = feats[0].float().numpy()
    norm = _np.linalg.norm(feat)
    return feat / (norm + 1e-8)


class _dummy_ctx:
    """No-op context manager for numpy-only path."""
    def __enter__(self): return self
    def __exit__(self, *a): pass


def _embed_image(pil_img, clip_model=None, clip_processor=None) -> Tuple["np.ndarray", str]:
    """
    Embed a PIL image; returns (embedding_array, backend_used).
    Tries: ComfyUI CLIP → transformers CLIP → HSV histogram.
    """
    if HAS_NUMPY:
        import numpy as _np
        if clip_model is not None and clip_processor is not None and HAS_TORCH:
            try:
                inputs = clip_processor(images=pil_img, return_tensors="pt")
                with torch.no_grad():
                    feat = clip_model.get_image_features(**inputs)[0].float().cpu().numpy()
                norm = _np.linalg.norm(feat)
                return feat / (norm + 1e-8), "comfyui_clip"
            except Exception as exc:
                logger.warning("[nodes_character] _embed_image comfyui: %s", exc)
        if HAS_TRANSFORMERS and HAS_TORCH:
            try:
                return _clip_embed_transformers(pil_img), "transformers_clip"
            except Exception as exc:
                logger.warning("[nodes_character] _embed_image transformers: %s", exc)
        return _hsv_histogram(pil_img), "hsv_histogram"
    raise RuntimeError("numpy is required for character embedding")


def _cosine_sim(a: "np.ndarray", b: "np.ndarray") -> float:
    """Cosine similarity between two 1-D numpy arrays."""
    import numpy as _np
    na = _np.linalg.norm(a)
    nb = _np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(_np.dot(a, b) / (na * nb))


# ===========================================================================
# Profile persistence helpers
# ===========================================================================

def _save_profile(profile: dict, filepath: str):
    import numpy as _np
    data = {
        "embedding": profile["embedding"],
        "hist":      profile.get("hist", _np.zeros(36, dtype=_np.float32)),
        "meta":      _np.array([json.dumps(profile.get("meta", {}))]),
    }
    _np.savez_compressed(filepath, **data)


def _load_profile(filepath: str) -> dict:
    import numpy as _np
    npz = _np.load(filepath, allow_pickle=True)
    meta_str = str(npz["meta"][0]) if "meta" in npz else "{}"
    return {
        "embedding": npz["embedding"],
        "hist":      npz.get("hist", _np.zeros(36, dtype=_np.float32)),
        "meta":      json.loads(meta_str),
    }


# ===========================================================================
# Registration
# ===========================================================================

NODE_CLASS_MAPPINGS = {}

NODE_DISPLAY_NAME_MAPPINGS = {}
