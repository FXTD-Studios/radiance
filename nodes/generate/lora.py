"""
nodes_hdr_lora.py — Radiance HDR LoRA ComfyUI nodes.
════════════════════════════════════════════════════════════════════════════════

Two nodes that wire the Radiance HDR LoRA workflow into any ComfyUI pipeline:

  1. RadianceHDRLoRALoader
     • Loads a .safetensors LoRA file trained by train_hdr_lora.py.
     • Reads Radiance metadata (model_name, compression_ratio, rank, alpha)
       embedded at export time.
     • Passes the raw MODEL through so downstream samplers see the patched
       weights.  Outputs compression_ratio so it can be wired directly into
       RadianceHDREncoder without manual entry.

  2. RadianceHDRLoRAApply
     • Applies a LoRA dict (from the Loader, or any other source) to a MODEL
       with a user-adjustable strength multiplier.
     • If model_hint is wired, it cross-checks against the preset table and
       warns when the LoRA was trained on a different model family.
     • Outputs the patched MODEL and the effective compression_ratio (from
       LoRA metadata, or RADIANCE_MODEL_PRESETS if not found).

Pipeline wiring example
────────────────────────
  [CheckpointLoader] ──MODEL──▶ [RadianceHDRLoRAApply] ──MODEL──▶ [KSampler]
                                        ▲                              │
  [RadianceHDRLoRALoader] ─lora_dict──┘                              │
         └── compression_ratio ──▶ [RadianceHDREncoder] ◀── image ──┘

The MODEL output from RadianceHDRLoRAApply replaces the base checkpoint MODEL
everywhere in the graph so no other nodes need changing.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import torch

from radiance.core.errors import RadianceError
from radiance.path_utils import strip_path_quotes

logger = logging.getLogger("radiance.hdr_lora")
diag_logger = logging.getLogger("radiance.diagnostics")

# ─────────────────────────────────────────────────────────────────────────────
# Lazy safetensors import — optional at import time, required when loading
# ─────────────────────────────────────────────────────────────────────────────

def _require_safetensors():
    try:
        from safetensors.torch import load_file, save_file  # noqa: F401
        return load_file
    except ImportError:
        raise ImportError(
            "safetensors is required for RadianceHDRLoRALoader.  "
            "Install it with: pip install safetensors"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Import shared presets from nodes_hdr_smart (or fall back gracefully)
# ─────────────────────────────────────────────────────────────────────────────

try:
    from radiance.nodes.hdr.smart import RADIANCE_MODEL_PRESETS, _resolve_model
    _HAS_PRESETS = True
except ImportError:
        logger.warning(
            "nodes_hdr_smart not available — model preset lookup disabled in LoRA nodes."
        )
        RADIANCE_MODEL_PRESETS: dict = {}
        _HAS_PRESETS = False

        def _resolve_model(hint: str) -> str | None:  # type: ignore[misc]
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_LORA_METADATA_KEYS = {
    "radiance_model_name",
    "radiance_compression_ratio",
    "radiance_rank",
    "radiance_alpha",
    "radiance_version",
}


def _parse_lora_metadata(raw_meta: dict) -> dict:
    """Extract and coerce Radiance metadata from a safetensors metadata dict.

    safetensors stores ALL metadata values as strings, so we cast carefully.
    Returns a dict with Python-typed values (float, int, str).
    """
    out: dict[str, Any] = {}
    for key in _LORA_METADATA_KEYS:
        val = raw_meta.get(key)
        if val is None:
            continue
        if key in ("radiance_compression_ratio", "radiance_alpha"):
            try:
                out[key] = float(val)
            except (ValueError, TypeError) as _exc:
                logger.debug(
                    "[Radiance] _parse_lora_metadata(): ignoring %s from `out[key] = float(val)`: %s",
                    type(_exc).__name__, _exc,
                )
        elif key in ("radiance_rank",):
            try:
                out[key] = int(val)
            except (ValueError, TypeError) as _exc:
                logger.debug(
                    "[Radiance] _parse_lora_metadata(): ignoring %s from `out[key] = int(val)`: %s",
                    type(_exc).__name__, _exc,
                )
        else:
            out[key] = str(val)
    return out


# (down suffix, up suffix) for every LoRA key layout we accept.  Order matters:
# the first suffix that matches a key wins, so longer/more specific forms are
# listed before the bare ones.
_LORA_KEY_SUFFIXES: tuple[tuple[str, str], ...] = (
    (".lora_down.weight",         ".lora_up.weight"),           # kohya_ss
    (".lora_A.default.weight",    ".lora_B.default.weight"),    # PEFT named adapter
    (".lora_A.weight",            ".lora_B.weight"),            # PEFT / diffusers
    (".lora.down.weight",         ".lora.up.weight"),           # diffusers (dotted)
    ("_lora.down.weight",         "_lora.up.weight"),           # diffusers (legacy)
    (".lora_linear_layer.down.weight", ".lora_linear_layer.up.weight"),
    (".lora_A",                   ".lora_B"),                   # mochi
)

# Namespace prefixes that wrap the real module path in a LoRA export.  Stripping
# them leaves a name that can be matched against the model's own state-dict keys.
_LORA_NAME_PREFIXES: tuple[str, ...] = (
    "lora_unet_",
    "lora_te1_",
    "lora_te2_",
    "lora_te_",
    "lora_transformer_",
    "base_model.model.",
    "diffusion_model.",
    "transformer.",
    "unet.",
)


def _strip_lora_name_prefix(name: str) -> str:
    """Drop the LoRA namespace prefix from a module name, if it carries one."""
    for prefix in _LORA_NAME_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _iter_lora_pairs(lora_tensors: dict):
    """Yield ``(module_name, down, up, alpha)`` for every LoRA pair present.

    *module_name* keeps its namespace prefix; _lookup_weight_key tries it both
    with and without, because some architectures genuinely contain a submodule
    called ``transformer`` and stripping unconditionally would lose it.

    DEFECT GUARD: this used to slice ``k[len("lora_unet_"):]`` with no
    ``startswith`` check and to recognise only ``.lora_down.weight`` /
    ``.lora_up.weight``.  A PEFT/diffusers export (``diffusion_model.*`` or
    ``transformer.*`` with ``lora_A``/``lora_B``) produced zero pairs, so the
    node applied nothing and reported success, and a text-encoder key
    (``lora_te_*``) was mangled into a nonsense module path by the blind slice.
    """
    for key in sorted(lora_tensors):
        for down_suffix, up_suffix in _LORA_KEY_SUFFIXES:
            if not key.endswith(down_suffix):
                continue
            base = key[: -len(down_suffix)]
            up = lora_tensors.get(base + up_suffix)
            if up is not None:
                yield (
                    base,
                    lora_tensors[key],
                    up,
                    lora_tensors.get(base + ".alpha"),
                )
            break


def _build_weight_key_index(weight_keys) -> dict[str, str]:
    """Index ``*.weight`` state-dict keys by every name a LoRA might use.

    kohya_ss flattens '.' to '_' in module paths and diffusers keeps them
    dotted, and ComfyUI's own state dict carries a ``diffusion_model.`` wrapper
    that LoRA exports usually omit.  Exact matches always win over the
    normalised variants so an ambiguous ``_`` vs ``.`` name cannot shadow a
    real key.
    """
    exact: dict[str, str] = {}
    fuzzy: dict[str, str] = {}
    for key in weight_keys:
        if not key.endswith(".weight"):
            continue
        base = key[: -len(".weight")]
        exact.setdefault(base, key)

        variants = {base.replace(".", "_")}
        for wrapper in ("model.diffusion_model.", "diffusion_model.", "model."):
            if base.startswith(wrapper):
                short = base[len(wrapper):]
                variants.add(short)
                variants.add(short.replace(".", "_"))
                break
        for variant in variants:
            fuzzy.setdefault(variant, key)

    index = dict(fuzzy)
    index.update(exact)
    return index


def _lookup_weight_key(index: dict, name: str):
    """Resolve a LoRA module name to a state-dict weight key, or None.

    Tried in order: the name as exported, its '_'-flattened form, then the same
    two with the LoRA namespace prefix removed.  The prefixed forms go first so
    a model that really does have a ``transformer`` submodule wins over the
    ``transformer.`` namespace reading of the same string.
    """
    stripped = _strip_lora_name_prefix(name)
    for candidate in (name, name.replace(".", "_"), stripped, stripped.replace(".", "_")):
        key = index.get(candidate)
        if key is not None:
            return key
    return None


def _lora_delta(down, up, alpha, weight_shape):
    """Return ``(alpha / rank) * up @ down`` reshaped to *weight_shape*, float32.

    DEFECT GUARD: the delta is computed and handed to the caller, never written
    into a live weight here.  ``Tensor.add_`` has no ``torch.float8_e4m3fn``
    kernel, so an fp8_e4m3fn UNET used to raise partway through the key loop and
    leave the shared module half patched.
    """
    if down.ndim < 1 or up.ndim < 1:
        return None
    rank = int(down.shape[0])
    if rank == 0:
        return None

    alpha_val = float(alpha) if alpha is not None else float(rank)
    scale = alpha_val / rank

    mat_down = down.to(dtype=torch.float32).flatten(start_dim=1)
    mat_up = up.to(dtype=torch.float32).flatten(start_dim=1)
    if mat_up.shape[1] != mat_down.shape[0]:
        return None

    expected = 1
    for dim in weight_shape:
        expected *= int(dim)

    delta = torch.mm(mat_up, mat_down)
    if delta.numel() != expected:
        return None
    return (scale * delta).reshape(tuple(int(d) for d in weight_shape))


def _apply_lora_via_patcher(patcher, lora_tensors: dict, strength: float) -> int:
    """Record LoRA deltas on a ComfyUI ModelPatcher clone via ``add_patches``.

    DEFECT GUARD: ``ModelPatcher.clone()`` shares the underlying nn.Module and
    its nn.Parameter storage (``get_clone_model_override()`` returns
    ``self.model`` itself), so writing deltas into ``module.weight`` corrupted
    the loader's cached model permanently and accumulated another delta on every
    queue.  ``add_patches`` records the diff on the clone's own patch table and
    ComfyUI applies it at weight-load time, which also keeps quantised (fp8)
    weights out of the arithmetic.
    """
    model_sd = patcher.model.state_dict()
    index = _build_weight_key_index(model_sd.keys())

    patches: dict = {}
    unmatched: list[str] = []
    for name, down, up, alpha in _iter_lora_pairs(lora_tensors):
        key = _lookup_weight_key(index, name)
        if key is None:
            unmatched.append(name)
            continue
        delta = _lora_delta(down, up, alpha, model_sd[key].shape)
        if delta is None:
            unmatched.append(name)
            continue
        patches[key] = ("diff", (delta,))

    if unmatched:
        logger.debug(
            "RadianceHDRLoRAApply: %d LoRA key(s) had no matching model weight: %s",
            len(unmatched), ", ".join(unmatched[:8]),
        )

    # add_patches() silently drops keys that are not in the model state dict, so
    # trust its return value rather than len(patches) for the applied count.
    return len(patcher.add_patches(patches, strength))


def _apply_lora_in_place(model, lora_tensors: dict, strength: float) -> int:
    """Write LoRA deltas into a plain module tree.

    Only reached for objects that are not a ComfyUI ModelPatcher (raw
    nn.Module pipelines and tests).  RadianceHDRLoRAApply deep-copies those
    before calling in, so the caller's module is still left untouched.
    """
    target = model
    for attr in ("model", "unet", "diffusion_model"):
        if hasattr(target, attr):
            target = getattr(target, attr)

    modules_by_key: dict[str, Any] = {}
    for name, module in target.named_modules():
        weight = getattr(module, "weight", None)
        if isinstance(weight, torch.Tensor):
            modules_by_key[f"{name}.weight"] = module

    index = _build_weight_key_index(modules_by_key.keys())

    applied = 0
    for name, down, up, alpha in _iter_lora_pairs(lora_tensors):
        key = _lookup_weight_key(index, name)
        if key is None:
            logger.debug("LoRA module not found in model: %s", name)
            continue
        module = modules_by_key[key]
        weight = module.weight
        delta = _lora_delta(down, up, alpha, weight.shape)
        if delta is None:
            continue
        with torch.no_grad():
            merged = weight.to(dtype=torch.float32) + (
                strength * delta.to(device=weight.device)
            )
            weight.copy_(merged.to(dtype=weight.dtype))
        applied += 1
    return applied


def _apply_lora_to_model(model, lora_tensors: dict, strength: float) -> int:
    """Apply lora down/up weight pairs to *model*, returning the number applied.

    The update is ``W += strength * (alpha / rank) * up @ down``.  On a ComfyUI
    ModelPatcher it is recorded as a patch rather than merged eagerly; see
    ``_apply_lora_via_patcher`` for why.
    """
    add_patches = getattr(model, "add_patches", None)
    if callable(add_patches) and getattr(model, "model", None) is not None:
        applied = _apply_lora_via_patcher(model, lora_tensors, strength)
    else:
        applied = _apply_lora_in_place(model, lora_tensors, strength)

    logger.info("RadianceHDRLoRAApply: applied %d LoRA delta(s) at strength=%.3f", applied, strength)
    diag_logger.info(
        "HDR_LORA_APPLY applied=%d strength=%.3f",
        applied, strength
    )
    return applied


# ─────────────────────────────────────────────────────────────────────────────
# Node 1: RadianceHDRLoRALoader
# ─────────────────────────────────────────────────────────────────────────────

class RadianceHDRLoRALoader:
    """Load a Radiance HDR LoRA .safetensors file.

    Reads the Radiance metadata embedded at training time so the
    compression_ratio flows automatically into RadianceHDREncoder.

    Outputs
    -------
    lora_dict         — opaque dict of tensors + metadata (pass to LoRAApply)
    compression_ratio — float extracted from LoRA metadata (or 0.5 default)
    model_name        — string model family the LoRA was trained on
    metadata_json     — full metadata as a JSON string for display/debugging
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "/path/to/radiance_hdr_lora.safetensors",
                    "tooltip": "Path to a Radiance HDR LoRA .safetensors file (surrounding quotes are stripped); a relative path resolves from ComfyUI's working directory. Required: a blank path raises an error."
                }),
            },
            "optional": {
                "fallback_compression_ratio": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "compression_ratio output by this node when the LoRA metadata has none. HDR LoRA Apply does not see this value; it falls back to 0.5 on its own.",
                }),
            },
        }

    RETURN_TYPES    = ("LORA_DICT", "FLOAT",              "STRING",     "STRING")
    RETURN_NAMES    = ("lora_dict", "compression_ratio",  "model_name", "metadata_json")
    FUNCTION        = "load"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Generate"
    DESCRIPTION     = (
        "Load a Radiance HDR LoRA .safetensors and extract embedded training metadata. "
        "Wire compression_ratio → RadianceHDREncoder to guarantee consistent HDR response."
    )

    def load(
        self,
        lora_path: str,
        fallback_compression_ratio: float = 0.5,
    ):
        load_file = _require_safetensors()

        lora_path = strip_path_quotes(lora_path)
        if not lora_path:
            raise ValueError("RadianceHDRLoRALoader: lora_path must not be empty.")
        if not os.path.isfile(lora_path):
            raise FileNotFoundError(
                f"RadianceHDRLoRALoader: file not found — {lora_path}"
            )

        # Load tensors + raw string metadata
        from safetensors import safe_open  # type: ignore
        tensors: dict[str, torch.Tensor] = {}
        raw_meta: dict[str, str] = {}

        with safe_open(lora_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
            raw_meta = dict(f.metadata()) if f.metadata() else {}

        meta = _parse_lora_metadata(raw_meta)

        compression_ratio = float(meta.get("radiance_compression_ratio", fallback_compression_ratio))
        model_name        = str(meta.get("radiance_model_name", "unknown"))

        # Build rich metadata JSON for display node / console
        display_meta = {
            "lora_path":          lora_path,
            "model_name":         model_name,
            "compression_ratio":  compression_ratio,
            "rank":               meta.get("radiance_rank", "?"),
            "alpha":              meta.get("radiance_alpha", "?"),
            "radiance_version":   meta.get("radiance_version", "?"),
            "tensor_count":       len(tensors),
            "raw_metadata":       raw_meta,
        }
        metadata_json = json.dumps(display_meta, indent=2)

        logger.info(
            "Loaded Radiance HDR LoRA: model=%s  ratio=%.3f  tensors=%d  path=%s",
            model_name, compression_ratio, len(tensors), lora_path,
        )
        diag_logger.info(
            "HDR_LORA_LOAD model=%s compression_ratio=%.3f rank=%s tensors=%d",
            model_name, compression_ratio, meta.get("radiance_rank", "?"), len(tensors),
        )

        lora_dict = {
            "tensors":  tensors,
            "metadata": meta,
            "path":     lora_path,
        }

        return (lora_dict, compression_ratio, model_name, metadata_json)


# ─────────────────────────────────────────────────────────────────────────────
# Node 2: RadianceHDRLoRAApply
# ─────────────────────────────────────────────────────────────────────────────

class RadianceHDRLoRAApply:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ HDR"
    """Apply a Radiance HDR LoRA to a diffusion MODEL.

    The low-rank delta (strength × alpha/rank × B@A) is recorded on a clone of
    the incoming MODEL through ComfyUI's own ModelPatcher.add_patches(), so the
    upstream loader's cached model is never touched and the same graph can be
    re-queued without deltas piling up.

    It is safe to chain multiple LoRA applies — each one accumulates deltas.

    Inputs
    ------
    model             — ComfyUI MODEL (from CheckpointLoader or prior LoRAApply)
    lora_dict         — LORA_DICT from RadianceHDRLoRALoader
    strength          — scaling multiplier (0 = skip, 1 = full, 1.5 = amplified)
    model_hint        — optional string to cross-check the LoRA's intended model
                        family.  Emits a WARNING if mismatched — does NOT block.

    Outputs
    -------
    MODEL             — patched MODEL (replaces the input MODEL in the graph)
    compression_ratio — from LoRA metadata (wire to RadianceHDREncoder)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":    ("MODEL", {
                    "tooltip": "Diffusion model to patch. It is cloned and the LoRA deltas are added as ComfyUI patches, so the upstream model is left untouched.",
                }),
                "lora_dict": ("LORA_DICT", {
                    "tooltip": "LoRA tensors and metadata from Radiance HDR LoRA Loader. If no tensor matches a model weight the node raises an error.",
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "LoRA strength multiplier. 1.0 = trained weight. 0 = no effect.",
                }),
            },
            "optional": {
                "model_hint": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "ltx-video / flux / wan / sdxl / …",
                    "tooltip": "Cross-checks the LoRA's trained model against this hint and warns if mismatched.",
                }),
            },
        }

    RETURN_TYPES  = ("MODEL",  "FLOAT")
    RETURN_NAMES  = ("MODEL",  "compression_ratio")
    FUNCTION      = "apply"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Generate"
    DESCRIPTION   = (
        "Apply a Radiance HDR LoRA to a diffusion model. "
        "Outputs the patched MODEL and the matching compression_ratio for RadianceHDREncoder."
    )

    def apply(
        self,
        model,
        lora_dict: dict,
        strength: float = 1.0,
        model_hint: str = "",
    ):
        import copy

        tensors = lora_dict.get("tensors", {})
        meta    = lora_dict.get("metadata", {})

        lora_model_name     = str(meta.get("radiance_model_name", "unknown"))
        compression_ratio   = float(meta.get("radiance_compression_ratio", 0.5))

        # ── Cross-check model hint ─────────────────────────────────────────
        if model_hint and model_hint.strip():
            resolved = _resolve_model(model_hint.strip())
            if resolved and resolved != lora_model_name:
                logger.warning(
                    "RadianceHDRLoRAApply: LoRA was trained on '%s' but model_hint='%s' "
                    "resolved to '%s'.  Applying anyway — results may be inconsistent.",
                    lora_model_name, model_hint, resolved,
                )
            elif _HAS_PRESETS and resolved is None:
                logger.info(
                    "RadianceHDRLoRAApply: model_hint='%s' not found in preset table; "
                    "using compression_ratio=%.3f from LoRA metadata.",
                    model_hint, compression_ratio,
                )
            # If hint resolved to lora_model_name → all good, no warning needed

        # ── Skip if strength == 0 ─────────────────────────────────────────
        if abs(strength) < 1e-6:
            logger.info("RadianceHDRLoRAApply: strength=0, skipping apply.")
            return (model, compression_ratio)

        # ── Clone the ComfyUI model wrapper ───────────────────────────────
        # The clone shares the underlying nn.Module with the original by
        # design; it is the patch table that is per-clone.  _apply_lora_to_model
        # therefore records deltas instead of writing them into the weights.
        try:
            patched_model = model.clone()
        except AttributeError:
            # Fallback for non-ComfyUI model objects (tests, custom pipelines).
            # Those have no patch table, so they get a real copy to merge into.
            patched_model = copy.deepcopy(model)

        # ── Apply LoRA deltas ──────────────────────────────────────────────
        applied = _apply_lora_to_model(patched_model, tensors, strength)

        if applied == 0:
            # A LoRA that matched nothing is a failed run, not a quiet no-op:
            # the strength widget is ignored and the output MODEL is the input
            # MODEL. This used to be logged at INFO as "applied 0 LoRA delta(s)"
            # and read as success.
            raise RadianceError(
                f"no LoRA delta matched this model, 0 of {len(tensors)} tensor(s) "
                f"from {lora_dict.get('path', 'the supplied LORA_DICT')!r} could be "
                "paired with a model weight. The LoRA key layout or the model "
                "family is wrong; strength was ignored and the MODEL is unpatched.",
                node_name="RadianceHDRLoRAApply",
            )

        diag_logger.info(
            "HDR_LORA_APPLY_DONE lora_model=%s hint=%s strength=%.3f compression_ratio=%.3f",
            lora_model_name, model_hint or "none", strength, compression_ratio,
        )

        return (patched_model, compression_ratio)


# ─────────────────────────────────────────────────────────────────────────────
# ComfyUI registration
# ─────────────────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "RadianceHDRLoRALoader": RadianceHDRLoRALoader,
    "RadianceHDRLoRAApply":  RadianceHDRLoRAApply,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceHDRLoRALoader": "◎ Radiance HDR LoRA Loader",
    "RadianceHDRLoRAApply":  "◎ Radiance HDR LoRA Apply",
}
