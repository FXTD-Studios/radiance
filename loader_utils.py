"""Model loading orchestration — downloading, detection, CLIP assembly, caching.

This module is the public API for loading models. It delegates data to
config/model_map.py (URLs, presets) and model/detect.py (architecture heuristics),
keeping only the loading logic that requires ComfyUI runtime imports.
"""
from __future__ import annotations

import os
import hashlib
import time
import logging

import torch
import folder_paths
import comfy.sd
import comfy.utils
import comfy.model_management

from radiance.config.model_map import RADIANCE_MODEL_MAP, CHECKPOINT_PRESETS, VIDEO_PRESET_NAMES, VIDEO_MODEL_TYPES
from radiance.model.detect import (
    detect_model_type,
    latent_format,
    assemble_clip_paths,
    get_clip_type_enum,
    estimate_vram_usage,
    LATENT_CHANNELS,
    CLIP_SLOT_ORDER,
    AUDIO_VAE_KEY_REMAP,
    _BASE_VRAM,
    _DTYPE_MULT,
)
from radiance.model.cache import get_model_cache, _unet_cache, _clip_cache, _vae_cache, _audio_vae_cache
from radiance.core.logging import supports_unicode
from comfy.model_management import get_free_memory as _get_free_vram, get_total_memory as _get_total_vram, get_torch_device

logger = logging.getLogger("radiance.loader")


def _sha256_file(path: str, chunk: int = 1 << 20) -> str:
    """Return the hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _refresh_listing(target_path: str, folder_type: str) -> None:
    # A failure to refresh ComfyUI's listing is a stale menu, not a failed
    # download, and reporting it as one made the loader claim a model was
    # missing that was on disk.
    _exc = None
    try:
        folder_paths.folder_names_and_paths[folder_type]
        folder_paths.get_filename_list(folder_type)
    except Exception:
        try:
            folder_paths.get_filename_list(folder_type)   # one retry, as before
            _exc = None
        except Exception as _retry_exc:
            _exc = _retry_exc
    if _exc is not None:
        logger.warning(
            "Radiance: %s downloaded, but refreshing the %s listing failed (%s). "
            "Restart ComfyUI if the file does not appear in the menu.",
            os.path.basename(target_path), folder_type, _exc,
        )


def _download_model(url: str, target_path: str, folder_type: str,
                    expected_sha256: str | None = None, size: int | None = None) -> bool:
    """Download a model through radiance.core.model_fetch: pinned SHA-256,
    written to ``.part`` and renamed only after the digest matches, resumed
    after an interruption, and sent with the Hugging Face token for gated
    repositories. Returns True when the file is in place."""
    from radiance.core.model_fetch import ModelFetchError, fetch
    from radiance.core.consent import LEGACY_LOADER_OFFLINE_ENV
    try:
        fetch(url, target_path, sha256=expected_sha256 or "", size=size,
              label=os.path.basename(target_path), legacy_offline_env=LEGACY_LOADER_OFFLINE_ENV)
    except ModelFetchError as e:
        logger.error("%s", e)
        return False
    _refresh_listing(target_path, folder_type)
    return True


def ensure_model_exists(name: str, folder_type: str, auto_download: bool = True) -> str | None:
    """Path of `name` in `folder_type`; with `auto_download` (the default) a
    model in RADIANCE_MODEL_MAP that is not installed is fetched first.
    RADIANCE_ALLOW_DOWNLOADS=0, RADIANCE_LOADER_OFFLINE=1 or the Hugging Face
    offline flags stop the download."""
    if not name or name == "None":
        return None

    path = folder_paths.get_full_path(folder_type, name)
    if path and os.path.exists(path):
        return path

    if auto_download and name in RADIANCE_MODEL_MAP:
        from radiance.core.consent import LEGACY_LOADER_OFFLINE_ENV, downloads_allowed
        if not downloads_allowed(legacy_offline_env=LEGACY_LOADER_OFFLINE_ENV):
            logger.error(
                "Radiance: '%s' is not installed and downloads are off (RADIANCE_ALLOW_DOWNLOADS=0, "
                "RADIANCE_LOADER_OFFLINE=1 or an offline flag). Place the file in models/%s or unset it.",
                name, folder_type,
            )
            return None
        res = RADIANCE_MODEL_MAP[name]
        if res["type"] == folder_type:
            base_dir = folder_paths.get_folder_paths(folder_type)[0]
            target_path = os.path.join(base_dir, *name.replace("\\", "/").split("/"))
            if _download_model(res["url"], target_path, folder_type, res.get("sha256"), res.get("size")):
                return target_path

    return None


def get_available_vram() -> float:
    """Return free GPU memory in GB."""
    return _get_free_vram(get_torch_device()) / (1024 ** 3)


def get_total_vram() -> float:
    """Return total GPU memory in GB."""
    return _get_total_vram(get_torch_device()) / (1024 ** 3)


def file_fingerprint(path: str) -> str:
    try:
        st = os.stat(path)
        return f"{st.st_mtime:.0f}:{st.st_size}"
    except OSError:
        return "nostat"


# ---------------------------------------------------------------------------
# Shared load_radiance_stack pipeline steps (used by RadianceUnifiedLoader and
# RadianceVideoLoader). Each step appends human-readable lines to the caller's
# info_lines list (mutated in place) for the HUD output.
# ---------------------------------------------------------------------------


def resolve_divider() -> str:
    """Return the log-line separator, falling back to ASCII on non-Unicode consoles."""
    return "│" if supports_unicode() else "|"


def _apply_preset_field(cfg, field, key, cur, overrides):
    new = cfg.get(key, cur)
    if new != cur:
        overrides.append(f"{field}: {cur}→{new}")
    return new


# ALBABIT-FIX: these 3 fields are hidden while a preset is active, so every
# preset must define them all -- a missing key would silently fall back to
# the hidden widget's (possibly stale) value via _apply_preset_field's
# cfg.get(key, cur).
_REQUIRED_PRESET_FIELDS = ("model_type", "weight_dtype", "clip_dtype")


def apply_checkpoint_preset(
    preset: str,
    model_type: str,
    weight_dtype: str,
    clip_dtype: str,
    info_lines: list[str],
) -> tuple[str, str, str, list[str]]:
    """Apply a CHECKPOINT_PRESETS entry, overriding model_type/weight_dtype/clip_dtype."""
    overrides: list[str] = []
    if preset not in ("Custom",) and preset in CHECKPOINT_PRESETS:
        cfg = CHECKPOINT_PRESETS[preset]

        missing = [k for k in _REQUIRED_PRESET_FIELDS if k not in cfg]
        if missing:
            logger.warning(
                f"[Radiance] Preset '{preset}' is missing {missing} in CHECKPOINT_PRESETS "
                "-- falling back to the current (hidden, possibly stale) widget value "
                "for those fields instead of a preset-defined one."
            )

        model_type   = _apply_preset_field(cfg, "model_type",   "model_type",   model_type,   overrides)
        weight_dtype = _apply_preset_field(cfg, "weight_dtype",  "weight_dtype", weight_dtype, overrides)
        clip_dtype   = _apply_preset_field(cfg, "clip_dtype",    "clip_dtype",   clip_dtype,   overrides)
        # ALBABIT-FIX: offload_mode is NOT preset-overridden — it's exposed
        # as a visible widget for "Low VRAM" presets (JS sets its default
        # to match the preset, but the user can change it, e.g. on a
        # higher-VRAM GPU where cpu_offload is unnecessarily slow).

        msg = f"Preset '{preset}'" + (f" (overrode: {', '.join(overrides)})" if overrides else " (no overrides)")
        logger.info(msg)
        info_lines.append(msg)

    return model_type, weight_dtype, clip_dtype, overrides


def resolve_architecture(
    unet_path: str,
    model_type: str,
    info_lines: list[str],
) -> tuple[str, str | None, str]:
    """Resolve the model architecture, running Auto-Detect heuristics if requested."""
    detected_type = None
    if model_type == "Auto-Detect":
        detected_type = detect_model_type(unet_path)
        if detected_type:
            resolved_type = detected_type
            info_lines.append(f"Auto-detected: {resolved_type}")
        else:
            resolved_type = "sdxl"   # safe fallback
            logger.warning(
                "Architecture auto-detect failed. Falling back to 'sdxl'. "
                "Set model_type manually if this is wrong."
            )
            info_lines.append("Auto-detect failed — fallback: sdxl")
    else:
        resolved_type = model_type

    latent_fmt = latent_format(resolved_type)
    lat_msg = f"Latent format: {latent_fmt} ({resolved_type})"
    logger.info(lat_msg)
    info_lines.append(lat_msg)

    return resolved_type, detected_type, latent_fmt


def setup_offload_mode(offload_mode: str, info_lines: list[str]) -> torch.device | None:
    """Apply the offload_mode setting, returning the CLIP load_device override (if any)."""
    if offload_mode == "sequential":
        # ComfyUI has never shipped set_lowvram_mode(); the call raised
        # AttributeError inside this try and logged "Could not enable
        # sequential offload" on every run. The supported switch is the
        # module-level vram_state, which load_models_gpu() reads on every
        # load: LOW_VRAM streams weights to the GPU per layer, the same thing
        # the --lowvram flag selects.
        try:
            mm = comfy.model_management
            if mm.vram_state not in (mm.VRAMState.LOW_VRAM, mm.VRAMState.NO_VRAM):
                mm.vram_state = mm.VRAMState.LOW_VRAM
            logger.info("Sequential CPU offload enabled (ComfyUI LOW_VRAM state)")
            info_lines.append("Offload: sequential")
        except Exception as e:
            logger.warning(f"Could not enable sequential offload: {e}")

    # FIX 6: ComfyUI model_options["load_device"] expects torch.device, not str.
    return torch.device("cpu") if offload_mode == "cpu_offload" else None


def estimate_vram_for_load(
    resolved_type: str,
    weight_dtype: str,
    clip_dtype: str,
    has_loras: bool,
    check_vram: str,
    divider: str,
    info_lines: list[str],
    extra_unet_count: int = 0,
) -> tuple[float, float, float]:
    """Estimate VRAM usage and, if check_vram=="On", log available/total VRAM.

    extra_unet_count: additional UNETs beyond the primary (e.g. 1 for WAN 2.2 MoE companion).
    """
    est = estimate_vram_usage(resolved_type, weight_dtype, clip_dtype, has_loras, False)
    if extra_unet_count > 0:
        # Companion UNETs add UNET memory only (same type/dtype, no extra CLIP).
        unet_gb = _BASE_VRAM.get(resolved_type, 8.0) * _DTYPE_MULT.get(weight_dtype, 1.0)
        est = round(est + unet_gb * extra_unet_count, 1)

    if check_vram == "On":
        avail = get_available_vram()
        total = get_total_vram()
        vram_msg = f"VRAM: ~{est} GB needed {divider} {avail:.2f} GB free / {total:.2f} GB total"
        logger.info(vram_msg)
        info_lines.append(vram_msg)
        if avail > 0 and est > avail * 0.9:
            # ALBABIT-FIX: est assumes a traditional full load -- with
            # DynamicVRAM/async offload active (ComfyUI 0.28+), real usage can
            # be far lower, so this warning may not actually apply.
            warn = (f"VRAM tight! {est} GB estimated, {avail:.2f} GB free. "
                    f"Consider fp8 dtype or cpu_offload -- or ignore if DynamicVRAM "
                    f"is active (actual usage may be much lower).")
            logger.warning(warn)
            info_lines.append(warn)
    else:
        avail = 0.0
        total = 0.0

    return est, avail, total


# ALBABIT-FIX: comfy's loaders silently return vae=None for a checkpoint with
# no baked VAE instead of raising, so callers used to log a fake success and
# only fail much later at VAE Decode with a confusing NoneType error.
def _require_baked_vae(vae, unet_name: str):
    if vae is None:
        raise RuntimeError(
            f"'{unet_name}' has no VAE weights baked in — "
            f"select a standalone vae_name file instead of "
            f"'Baked VAE (from UNET)'."
        )


def construct_audio_vae(sd: dict, metadata, resolved_type: str) -> comfy.sd.VAE:
    """Build a comfy.sd.VAE from a raw audio-VAE state dict.

    Applies the architecture's key remap first if AUDIO_VAE_KEY_REMAP has one
    (LTX-AV's audio_vae./vocoder. checkpoint namespace); most architectures
    (e.g. MiniMax H3) have none, since comfy.sd.VAE() already auto-detects
    their audio VAE from the raw, unprefixed keys.

    When a remap applies, state_dict_prefix_replace(..., filter_keys=True)
    pops the matched keys out of the CALLER's `sd` in place, on top of
    returning the renamed copy used here. load_unet_and_baked_vae relies on
    this so its own baked-VAE/model extraction doesn't see the audio-VAE
    weights mixed back in.
    """
    remap = AUDIO_VAE_KEY_REMAP.get(resolved_type)
    if remap:
        sd = comfy.utils.state_dict_prefix_replace(sd, remap, filter_keys=True)
    return comfy.sd.VAE(sd=sd, metadata=metadata)


def load_unet_and_baked_vae(
    unet_path: str,
    unet_name: str,
    weight_dtype: str,
    offload_mode: str,
    vae_name: str,
    audio_vae_name: str,
    resolved_type: str,
    caching: bool,
    divider: str,
    info_lines: list[str],
):
    """Load the UNET, optionally extracting a baked VAE/Audio VAE from the checkpoint.

    Returns ``(model, vae, audio_vae, unet_time, unet_cache_hit, vae_time, vae_cache_hit)``.
    ``vae``/``audio_vae`` stay ``None`` unless ``vae_name``/``audio_vae_name``
    request extraction from the checkpoint (callers that load a standalone VAE
    instead simply ignore them).
    """
    t0 = time.time()
    unet_fp = file_fingerprint(unet_path)
    # ALBABIT-FIX: include offload_mode — "sequential" patches the model
    # for lowvram at load time (set_lowvram_mode), so a cached UNET
    # loaded under a different offload_mode would silently keep its
    # stale patching.
    unet_key = f"unet:{unet_path}:{weight_dtype}:{offload_mode}:{unet_fp}"

    extract_vae       = (vae_name == "Baked VAE (from UNET)")
    extract_audio_vae = (audio_vae_name == "Baked Audio VAE (from UNET)")
    baked_vae_key       = f"vae:baked:{unet_path}:{unet_fp}"
    baked_audio_vae_key = f"audio_vae:baked:{unet_path}:{unet_fp}"

    vae = None
    audio_vae = None
    vae_time = 0.0
    vae_cache_hit = False
    unet_time = 0.0

    unet_cache_hit = caching and _unet_cache.has(unet_key)
    if unet_cache_hit:
        if extract_vae and not _vae_cache.has(baked_vae_key):
            unet_cache_hit = False
        if extract_audio_vae and not _audio_vae_cache.has(baked_audio_vae_key):
            unet_cache_hit = False

    if unet_cache_hit:
        model = _unet_cache.get(unet_key)
        logger.info(f"UNET loaded from cache: {unet_name}")
        info_lines.append(f"UNET: {unet_name} (cached)")

        if extract_vae:
            vae = _vae_cache.get(baked_vae_key)
            vae_cache_hit = True
            logger.info("Baked VAE loaded from cache")
            info_lines.append("VAE: Baked from UNET (cached)")
        if extract_audio_vae:
            audio_vae = _audio_vae_cache.get(baked_audio_vae_key)
            logger.info("Baked Audio VAE loaded from cache")
            info_lines.append("AUDIO VAE: Baked from UNET (cached)")
    else:
        model_options = {}
        is_gguf = unet_name.lower().endswith(".gguf")
        if is_gguf:
            logger.info(f"GGUF detected: {unet_name} (embedded quant)")
        else:
            dtype_map = {
                "fp8_e4m3fn": torch.float8_e4m3fn,
                "fp8_e5m2":   torch.float8_e5m2,
                "fp16":       torch.float16,
                "bf16":       torch.bfloat16,
                "fp32":       torch.float32,
            }
            if weight_dtype in dtype_map:
                model_options["dtype"] = dtype_map[weight_dtype]

        try:
            if extract_audio_vae:
                # ALBABIT-FIX: read the UNET state dict once and reuse it
                # for both the audio VAE extraction and the model/baked-VAE
                # load below (load_state_dict_guess_config /
                # load_diffusion_model_state_dict), instead of reading the
                # (often multi-GB) UNET file from disk twice.
                t_av0 = time.time()
                sd, metadata = comfy.utils.load_torch_file(unet_path, return_metadata=True)
                audio_vae = construct_audio_vae(sd, metadata, resolved_type)
                av_time = time.time() - t_av0
                logger.info("Audio VAE extracted natively from UNET")
                info_lines.append(f"AUDIO VAE: Baked from UNET ({av_time:.1f}s)")
                if caching:
                    _audio_vae_cache.put(baked_audio_vae_key, audio_vae)

                if extract_vae:
                    out = comfy.sd.load_state_dict_guess_config(
                        sd, output_vae=True, output_clip=False,
                        output_clipvision=False, model_options=model_options,
                        metadata=metadata,
                    )
                    model = out[0]
                    model.cached_patcher_init = (
                        comfy.sd.load_checkpoint_guess_config,
                        (unet_path, False, False, False, None, True, model_options, {}),
                        0,
                    )
                    t_vae0 = time.time()
                    vae = out[2]
                    _require_baked_vae(vae, unet_name)
                    if getattr(vae, "patcher", None) is not None:
                        vae.patcher.cached_patcher_init = (
                            comfy.sd.load_checkpoint_vae_patcher,
                            (unet_path, None, model_options, {}),
                        )
                    vae_time = time.time() - t_vae0
                    logger.info("VAE extracted natively from UNET")
                    info_lines.append(f"VAE: Baked from UNET ({vae_time:.1f}s)")
                    if caching:
                        _vae_cache.put(baked_vae_key, vae)
                else:
                    model = comfy.sd.load_diffusion_model_state_dict(sd, model_options=model_options, metadata=metadata)
                    model.cached_patcher_init = (comfy.sd.load_diffusion_model, (unet_path, model_options))
            elif extract_vae:
                # ALBABIT-FIX: load_checkpoint_guess_config to extract the
                # baked VAE natively from the checkpoint.
                out = comfy.sd.load_checkpoint_guess_config(
                    unet_path, output_vae=True, output_clip=False,
                    output_clipvision=False, model_options=model_options,
                )
                model = out[0]
                t_vae0 = time.time()
                vae = out[2]
                _require_baked_vae(vae, unet_name)
                vae_time = time.time() - t_vae0
                logger.info("VAE extracted natively from UNET")
                info_lines.append(f"VAE: Baked from UNET ({vae_time:.1f}s)")
                if caching:
                    _vae_cache.put(baked_vae_key, vae)
            else:
                model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)

            unet_time = time.time() - t0
            logger.info(f"UNET loaded {divider} {unet_name} [{weight_dtype}] {divider} {unet_time:.1f}s")
            info_lines.append(f"UNET: {unet_name} [{weight_dtype}] ({unet_time:.1f}s)")
            if caching:
                _unet_cache.put(unet_key, model)
        except Exception as e:
            raise RuntimeError(f"❌ Failed to load UNET '{unet_name}': {e}")

    _warn_if_weights_exceed_vram(model, unet_name, weight_dtype, info_lines)
    return model, vae, audio_vae, unet_time, unet_cache_hit, vae_time, vae_cache_hit


def _warn_if_weights_exceed_vram(model, unet_name: str, weight_dtype: str, info_lines: list) -> None:
    """Say, with real numbers, when the UNET cannot stay resident in VRAM.

    The table estimate said 16.5 GB for Flux dev at bf16; the model is 22.7 GB
    (3.5 live log). When the weights exceed the card, ComfyUI streams them from
    system RAM on every step, which is the dominant cost of the whole run. fp8
    halves the weights (Flux dev: ~11.9 GB) and keeps them resident.
    """
    try:
        size_gb = float(model.model_size()) / (1024 ** 3)
        total_gb = float(_get_total_vram()) / (1024 ** 3)
    except Exception:  # noqa: BLE001 - diagnostics only
        return
    if total_gb <= 0 or size_gb <= total_gb * 0.85:
        return
    fp8_gb = size_gb / 2.0 if weight_dtype in ("default", "bf16", "fp16") else size_gb
    msg = (f"{unet_name}: {size_gb:.1f} GB of weights [{weight_dtype}] on a {total_gb:.0f} GB GPU. "
           f"They cannot stay in VRAM, so ComfyUI streams them from system RAM every step "
           f"(typically several times slower per step).")
    if fp8_gb < size_gb:
        msg += f" weight_dtype fp8_e4m3fn brings them to ~{fp8_gb:.1f} GB and keeps them resident."
    logger.warning(msg)
    info_lines.append("⚠ " + msg)


def load_clip_stack(
    resolved_type: str,
    unet_path: str,
    clip_l: str | None,
    clip_g: str | None,
    t5xxl: str | None,
    llm_encoder: str | None,
    text_projection: str | None,
    clip_dtype: str,
    offload_mode: str,
    clip_load_device: torch.device | None,
    caching: bool,
    divider: str,
    auto_download: bool,
    info_lines: list[str],
):
    """Assemble and load the CLIP stack for ``resolved_type``.

    Returns ``(clip, clip_slot_used, clip_time, clip_cache_hit)``.
    """
    t0 = time.time()

    # A missing required slot silently selects a different text encoder
    # (Flux with only t5xxl became Mochi's T5). Fill it when exactly one file
    # on disk matches, otherwise stop with the slot named.
    from radiance.model.detect import autofill_required_clip_slots
    _slots, _filled, _missing = autofill_required_clip_slots(
        resolved_type, folder_paths.get_filename_list("text_encoders"),
        clip_l=clip_l, clip_g=clip_g, t5xxl=t5xxl, llm_encoder=llm_encoder,
        text_projection=text_projection)
    for _slot, _file in _filled.items():
        logger.warning("[Radiance] %s needs %s; it was empty, using %s.", resolved_type, _slot, _file)
        info_lines.append(f"CLIP {_slot}: auto -> {_file}")
    if _missing:
        raise ValueError(
            f"❌ {resolved_type} needs text encoder slot(s) {', '.join(_missing)}, which are "
            f"empty. Without them ComfyUI picks a different encoder for the files it "
            f"is given (e.g. Flux with only t5xxl loads Mochi's T5) and the prompt is "
            f"encoded wrongly. Select the file(s) in the Loader.")
    clip_l, clip_g, t5xxl = _slots["clip_l"], _slots["clip_g"], _slots["t5xxl"]
    llm_encoder, text_projection = _slots["llm_encoder"], _slots["text_projection"]

    # Ensure all selected CLIPs exist/downloaded
    for slot, val in [("clip_l", clip_l), ("clip_g", clip_g),
                      ("t5xxl", t5xxl), ("llm_encoder", llm_encoder),
                      ("text_projection", text_projection)]:
         if val != "Baked (from UNET)":
             ensure_model_exists(val, "text_encoders", auto_download)

    clip_paths = assemble_clip_paths(resolved_type, unet_path=unet_path, clip_l=clip_l, clip_g=clip_g, t5xxl=t5xxl, llm_encoder=llm_encoder, text_projection=text_projection)

    if not clip_paths:
        raise ValueError(
            f"❌ No CLIP encoders provided for architecture '{resolved_type}'. "
            f"Fill the required slot(s): "
            f"{', '.join(CLIP_SLOT_ORDER.get(resolved_type, ['clip_l']))}"
        )

    clip_fps     = ":".join(file_fingerprint(p) for p in clip_paths)
    # ALBABIT-FIX: include offload_mode — a cached CLIP keeps the
    # load_device it was first loaded with, so switching offload_mode
    # without this would silently reuse a CLIP stuck on CPU (or GPU).
    clip_key     = f"clip:{':'.join(clip_paths)}:{resolved_type}:{clip_dtype}:{offload_mode}:{clip_fps}"

    # Report the slots the assembly actually walked, in its order. Testing
    # `val is not None` counted ComfyUI's empty-combo sentinel "None" as a
    # loaded encoder, so the HUD named encoders that were never read.
    _slot_values = {"clip_l": clip_l, "clip_g": clip_g, "t5xxl": t5xxl,
                    "llm_encoder": llm_encoder, "text_projection": text_projection}
    clip_slot_used = [
        slot for slot in CLIP_SLOT_ORDER.get(resolved_type, list(_slot_values))
        if _slot_values.get(slot) not in (None, "", "None")
    ]

    clip_time = 0.0
    clip_cache_hit = caching and _clip_cache.has(clip_key)
    if clip_cache_hit:
        clip = _clip_cache.get(clip_key)
        logger.info(f"CLIP loaded from cache: {' + '.join(clip_slot_used)}")
        info_lines.append(f"CLIP: {'+'.join(clip_slot_used)} (cached)")
    else:
        clip_type_enum = get_clip_type_enum(resolved_type)
        clip_model_opts = {}
        if clip_load_device:
            clip_model_opts["load_device"] = clip_load_device
        dtype_map = {
            "fp16": torch.float16, "bf16": torch.bfloat16,
            "fp8_e4m3fn": torch.float8_e4m3fn, "fp32": torch.float32,
        }
        if clip_dtype in dtype_map:
            clip_model_opts["dtype"] = dtype_map[clip_dtype]

        try:
            clip = comfy.sd.load_clip(
                ckpt_paths=clip_paths,
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
                clip_type=clip_type_enum,
                model_options=clip_model_opts if clip_model_opts else {},
            )
            clip_time = time.time() - t0
            logger.info(
                f"CLIP loaded {divider} {' + '.join(clip_slot_used)} "
                f"[{clip_dtype}] {divider} {clip_time:.1f}s"
            )
            info_lines.append(
                f"CLIP: {'+'.join(clip_slot_used)} [{clip_dtype}] ({clip_time:.1f}s)"
            )
            if caching:
                _clip_cache.put(clip_key, clip)
        except Exception as e:
            raise RuntimeError(f"❌ Failed to load CLIP: {e}")

    return clip, clip_slot_used, clip_time, clip_cache_hit


def load_standalone_vae(
    vae_name: str,
    auto_download: bool,
    caching: bool,
    divider: str,
    info_lines: list[str],
    load_sd_metadata_fn,
):
    """Load a standalone VAE file.

    ``load_sd_metadata_fn`` is the caller's ``_load_vae_sd_metadata`` (per-class
    polymorphism: the base loader returns ``(sd, None)``, the video loader
    returns ``(sd, metadata)`` so ``comfy.sd.VAE`` can pick the correct internal
    config for LTX 2.3).

    Returns ``(vae, vae_time, vae_cache_hit)``.
    """
    # ALBABIT-FIX: "conv" VAE uses a different compression (16x/4x) than the
    # default LTX 2.5 VAE (32x/8x) Resolution's factors assume, silently
    # halving output resolution. Delisted from presets/model_map; this warns
    # if picked manually anyway. See project_radiance_ltx25 memory.
    if "ltx-2.5-video-vae-conv" in vae_name.lower():
        warn = (
            f"⚠ '{vae_name}' uses a different VAE compression (16x spatial / "
            f"4x temporal) than LTX 2.5's default VAE (32x/8x, what "
            f"Resolution's 'LTXV (128ch)' assumes). The decoded video will "
            f"come out at half the intended resolution. Use "
            f"'ltx-2.5-video-vae-bf16.safetensors' instead."
        )
        logger.warning(warn)
        info_lines.append(warn)

    t0 = time.time()
    vae_path = ensure_model_exists(vae_name, "vae", auto_download)
    if not vae_path:
        raise FileNotFoundError(f"❌ VAE not found: '{vae_name}'. Enable auto_download or install it manually.")

    vae_fp  = file_fingerprint(vae_path)
    vae_key = f"vae:{vae_path}:{vae_fp}"

    vae_time = 0.0
    vae_cache_hit = caching and _vae_cache.has(vae_key)
    if vae_cache_hit:
        vae = _vae_cache.get(vae_key)
        logger.info(f"VAE loaded from cache: {vae_name}")
        info_lines.append(f"VAE: {vae_name} (cached)")
    else:
        try:
            sd, vae_metadata = load_sd_metadata_fn(vae_path)
            vae = comfy.sd.VAE(sd=sd, metadata=vae_metadata)
            vae_time = time.time() - t0
            logger.info(f"VAE loaded {divider} {vae_name} {divider} {vae_time:.1f}s")
            info_lines.append(f"VAE: {vae_name} ({vae_time:.1f}s)")
            if caching:
                _vae_cache.put(vae_key, vae)
        except Exception as e:
            raise RuntimeError(f"❌ Failed to load VAE '{vae_name}': {e}")

    return vae, vae_time, vae_cache_hit


def apply_lora_stack(
    model,
    clip,
    lora_stack,
    lora_on_error: str,
    divider: str,
    info_lines: list[str],
):
    """Apply a chained LORA_STACK to ``model``/``clip``.

    Returns ``(model, clip, applied_loras, out_lora_stack)``, where
    ``out_lora_stack`` is the list of applied ``(name, model_str, clip_str)``
    tuples (ready to pass through as a ``LORA_STACK`` output), or ``None`` if
    no LoRAs were applied.
    """
    combined_loras = list(lora_stack) if lora_stack else []

    applied_loras = []
    for i, (lora_name, model_str, clip_str) in enumerate(combined_loras, 1):
        if model_str == 0 and clip_str == 0:
            continue
        lora_path = folder_paths.get_full_path("loras", lora_name)
        if not lora_path:
            msg = f"LoRA not found: '{lora_name}'"
            if lora_on_error == "raise":
                raise FileNotFoundError(msg)
            logger.warning(f"{msg} — Skipping.")
            info_lines.append(f"LoRA {i}: {lora_name} (not found)")
            continue
        try:
            t0 = time.time()
            lora_data = comfy.utils.load_torch_file(lora_path)
            model, clip = comfy.sd.load_lora_for_models(
                model, clip, lora_data, model_str, clip_str
            )
            elapsed = time.time() - t0
            logger.info(
                f"LoRA {i} applied {divider} {lora_name} [model={model_str}, clip={clip_str}] {divider} {elapsed:.1f}s"
            )
            info_lines.append(
                f"LoRA {i}: {lora_name} [m={model_str} c={clip_str}] ({elapsed:.1f}s)"
            )
            applied_loras.append({"name": lora_name, "model_str": model_str,
                                  "clip_str": clip_str})
        except Exception as e:
            msg = f"Failed to apply LoRA '{lora_name}': {e}"
            if lora_on_error == "raise":
                raise RuntimeError(f"❌ {msg}")
            logger.warning(f"{msg} — Skipping.")
            info_lines.append(f"LoRA {i}: {lora_name} (failed)")

    out_lora_stack = [(e["name"], e["model_str"], e["clip_str"])
                      for e in applied_loras] if applied_loras else None

    return model, clip, applied_loras, out_lora_stack
