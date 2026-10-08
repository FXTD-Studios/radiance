import torch
import time
import math
import logging
import gc
import errno
from typing import Tuple, Dict, Any, Optional, List
from dataclasses import dataclass, field

import comfy.samplers
import comfy.sample
import comfy.model_management
import comfy.utils
from radiance.core.tiling import clamp_overlap
try:
    from ...tensor_contract import ensure_4d, ensure_5d
except (ImportError, ValueError):
    from tensor_contract import ensure_4d, ensure_5d


try:
    from comfy.nested_tensor import NestedTensor as _NestedTensor
    _HAS_NESTED_TENSOR = True
except ImportError:
    _NestedTensor = None
    _HAS_NESTED_TENSOR = False

try:
    from ...sampler_utils import (
        DYNAMIC_GUIDANCE_EARLY_MULTIPLIER, DYNAMIC_GUIDANCE_LATE_MULTIPLIER, 
        DYNAMIC_GUIDANCE_EARLY_THRESHOLD, DYNAMIC_GUIDANCE_LATE_THRESHOLD, 
        DYNAMIC_GUIDANCE_RAMP_WIDTH, GUIDANCE_RESCALE_PHI, 
        SIGMA_DISCONTINUITY_THRESHOLD, PAG_DEFAULT_SCALE, PAG_LAYER_NAMES, 
        CFG_PLUS_PLUS_DEFAULT_SCALE,
        DYNAMIC_CFG_EARLY_MULTIPLIER, DYNAMIC_CFG_LATE_MULTIPLIER,
        DYNAMIC_CFG_EARLY_THRESHOLD, DYNAMIC_CFG_LATE_THRESHOLD, MODEL_TYPES, 
        VIDEO_MODEL_TYPES, GUIDANCE_EMBED_MODELS, CFG_GUIDED_MODELS, 
        PREVIEW_METHODS, NOISE_TYPES, CLIP_TARGETS,
        TILE_BLEND_MODES, SamplerMode, SigmaCache, _sigma_cache,
        RadianceModelRegistry, detect_by_config, detect_by_architecture,
        detect_by_sampling, detect_model_type, get_model_defaults,
        parse_model_meta, refine_distillation_from_meta,
        gradual_sigma_blend, log_tensor, SigmaIndexer, SamplingStage,
        apply_flux_guidance, compute_dynamic_guidance, compute_dynamic_cfg,
        compute_base_sigmas, WORKFLOW_PRESETS, flux_shift_sigmas, get_flux_sigmas, get_sd_turbo_sigmas,
        validate_step_range, apply_pag_to_model, AYS_ANCHORS, get_ays_sigmas,
        guidance_rescale_cfg, correct_sigma_end, apply_cfg_plus_plus,
        build_sigma_report, _temporally_correlate, _perlin_noise, _perlin_noise_2d,
        _spectral_noise, _get_freq_grid, _spectral_noise_2d, _brownian_noise,
        _simplex_noise, _voronoi_noise, _curl_noise, generate_noise,
        route_conditioning, tile_sample,
        plan_temporal_windows, temporal_window_weights, slice_conds_temporally,
        MODEL_DEFAULTS,
    )
except (ImportError, ValueError):
    from sampler_utils import (
        DYNAMIC_GUIDANCE_EARLY_MULTIPLIER, DYNAMIC_GUIDANCE_LATE_MULTIPLIER,
        DYNAMIC_GUIDANCE_EARLY_THRESHOLD, DYNAMIC_GUIDANCE_LATE_THRESHOLD,
        DYNAMIC_GUIDANCE_RAMP_WIDTH, GUIDANCE_RESCALE_PHI,
        SIGMA_DISCONTINUITY_THRESHOLD, PAG_DEFAULT_SCALE, PAG_LAYER_NAMES,
        CFG_PLUS_PLUS_DEFAULT_SCALE,
        DYNAMIC_CFG_EARLY_MULTIPLIER, DYNAMIC_CFG_LATE_MULTIPLIER,
        DYNAMIC_CFG_EARLY_THRESHOLD, DYNAMIC_CFG_LATE_THRESHOLD, MODEL_TYPES,
        VIDEO_MODEL_TYPES, GUIDANCE_EMBED_MODELS, CFG_GUIDED_MODELS,
        PREVIEW_METHODS, NOISE_TYPES, CLIP_TARGETS,
        TILE_BLEND_MODES, SamplerMode, SigmaCache, _sigma_cache,
        RadianceModelRegistry, detect_by_config, detect_by_architecture,
        detect_by_sampling, detect_model_type, get_model_defaults,
        parse_model_meta, refine_distillation_from_meta,
        gradual_sigma_blend, log_tensor, SigmaIndexer, SamplingStage,
        apply_flux_guidance, compute_dynamic_guidance, compute_dynamic_cfg,
        compute_base_sigmas, WORKFLOW_PRESETS, flux_shift_sigmas, get_flux_sigmas, get_sd_turbo_sigmas,
        validate_step_range, apply_pag_to_model, AYS_ANCHORS, get_ays_sigmas,
        guidance_rescale_cfg, correct_sigma_end, apply_cfg_plus_plus,
        build_sigma_report, _temporally_correlate, _perlin_noise, _perlin_noise_2d,
        _spectral_noise, _get_freq_grid, _spectral_noise_2d, _brownian_noise,
        _simplex_noise, _voronoi_noise, _curl_noise, generate_noise,
        route_conditioning, tile_sample,
        plan_temporal_windows, temporal_window_weights, slice_conds_temporally,
        MODEL_DEFAULTS,
    )

logger = logging.getLogger("radiance.sampler")


def _is_invalid_progress_stream_error(exc: BaseException) -> bool:
    """Detect Windows/tqdm/wandb progress-stream failures without hiding model errors."""

    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, OSError):
            err_no = getattr(current, "errno", None)
            msg = str(current)
            if err_no == errno.EINVAL or "Errno 22" in msg or "Invalid argument" in msg:
                return True
        current = current.__cause__ or current.__context__
    return False


def _is_channel_mismatch_error(exc: BaseException) -> Optional[str]:
    """Detect the classic PyTorch conv/linear channel-count mismatch message.

    Returns the matched substring (for logging) or None. This is the error
    shape torch raises when a latent's channel count doesn't match what the
    connected model's first layer expects (e.g. a 4ch SDXL-shaped latent fed
    into a 16ch Flux model) — the most common cause of "dimension not
    matching" reports after switching a Load Preset without also updating
    RadianceResolution's model_type/latent_channels to match.
    """
    msg = str(exc)
    for needle in ("to have {} channels", "channels, but got", "size mismatch", "shapes cannot be multiplied"):
        if needle in msg:
            return msg
    # torch's exact wording varies by op ("expected input[...] to have N
    # channels, but got M channels instead"); match the stable middle part.
    if "channels" in msg and ("expected" in msg or "got" in msg):
        return msg
    return None


def _count_non_finite(t) -> Optional[int]:
    """Non-finite element count for one tensor, or None if it cannot be checked."""
    try:
        if not torch.is_floating_point(t):
            return 0
        return int((~torch.isfinite(t)).sum().item())
    except (TypeError, RuntimeError, AttributeError):
        return None


def assert_latent_finite(samples) -> None:
    """Fail the node when the sampled latent carries NaN or Inf.

    Two defects lived in the old inline version of this check.

    First, it called `torch.isfinite(_s)` straight on the latent. That raises
    TypeError on a comfy.nested_tensor.NestedTensor, and the surrounding
    `except Exception` funnelled it into a `logger.debug`, which ComfyUI's
    default INFO level does not print. So on LTX-AV and MiniMax H3, the packed
    multi-modality runs where a CFG blowup is most likely and hardest to spot,
    the guard could not fire at all and said nothing about it. Unpack and check
    each stream instead.

    Second, the remedy on the tensor path was
    `nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)` under a single WARNING.
    Latent 0.0 is not black, it is a mid-grey-ish smear once decoded, so a run
    that blew up came out as a plausible-looking WRONG plate that a review pass
    can easily miss. Sanitising hides the failure precisely when it matters. A
    delivery pipeline has to stop, so this raises.
    """
    if samples is None:
        return

    parts = []
    # isinstance() needs a real class: some ComfyUI builds (and the test stubs)
    # expose comfy.nested_tensor with NestedTensor set to None.
    _nested_cls = _NestedTensor if isinstance(_NestedTensor, type) else None
    if _nested_cls is not None and isinstance(samples, _nested_cls):
        parts = [(f"stream {i}", t) for i, t in enumerate(samples.tensors)]
    elif isinstance(samples, torch.Tensor):
        parts = [("latent", samples)]
    else:
        # Unknown container: say so at WARNING, where it is actually visible,
        # rather than debug.
        logger.warning(
            "[RadianceSamplerPro] Could not run the NaN/Inf guard: sampled latent is "
            "a %s, which is neither a Tensor nor a NestedTensor. The output is "
            "UNCHECKED.", type(samples).__name__,
        )
        return

    bad_streams = []
    for label, t in parts:
        count = _count_non_finite(t)
        if count is None:
            logger.warning(
                "[RadianceSamplerPro] Could not run the NaN/Inf guard on %s (%s). "
                "That part of the output is UNCHECKED.", label, type(t).__name__,
            )
            continue
        if count:
            bad_streams.append(f"{label}: {count} of {t.numel()} value(s)")

    if bad_streams:
        raise RuntimeError(
            "[RadianceSamplerPro] The sampled latent contains NaN or Inf "
            f"({'; '.join(bad_streams)}).\n\n"
            "This is a diverged run, usually a CFG blowup, fp16/bf16 overflow, or a "
            "degenerate sigma schedule. Lower cfg (and flux_guidance / pag_scale if "
            "set), check the scheduler, or run the model in a wider precision.\n\n"
            "The node fails rather than substituting zeros: latent 0.0 is not black, "
            "so a sanitized blowup decodes to a plausible-looking wrong plate."
        )


ENERGY_MASK_KEY = "radiance_energy_mask"
ENERGY_PRIORITY_KEY = "radiance_energy_priority"
ENERGY_LAYERS_KEY = "radiance_energy_layers"


def _collect_energy_layers(positive) -> List[Tuple["torch.Tensor", float]]:
    """Every energy layer carried by a CONDITIONING list, oldest first.

    Written by RadianceEnergyMask. Layers stack: two of those nodes in series,
    or two branches joined by Conditioning Combine, both yield two layers and
    the sampler sums their contributions.

    Two shapes have to be told apart, and the layer tuple's identity is what
    does it. `attach_energy_mask` builds one `(mask, priority)` tuple per call
    and stores that same object on every entry, so:

      * one node fanned across four conditioning entries → the same layer seen
        four times → one layer;
      * two nodes combined → two distinct layers → both applied.

    Identity survives the shallow `dict(meta)` copies ComfyUI's conditioning
    helpers make. A node that deep-copied conditioning would defeat it, but
    that would give the masks new tensors too, so the dedupe token —
    `(id(mask), priority)` — degrades to "applied once", never to double
    counting.

    Kept at module scope so the tests can exercise the real parser instead of
    re-implementing it: the original tests copied this loop into the test body
    and asserted on their own copy, which meant they passed no matter what
    this file did.
    """
    if not isinstance(positive, list):
        return []

    layers: List[Tuple["torch.Tensor", float]] = []
    seen = set()

    def _add(mask, raw_priority):
        if not isinstance(mask, torch.Tensor) or mask.numel() == 0:
            logger.warning(
                "[Energy Guidance] Ignoring energy layer: expected a non-empty tensor, got %r.",
                type(mask).__name__,
            )
            return
        try:
            priority = float(raw_priority)
        except (TypeError, ValueError):
            logger.warning("[Energy Guidance] Non-numeric energy priority; defaulting to 1.0.")
            priority = 1.0

        token = (id(mask), priority)
        if token in seen:
            return
        seen.add(token)
        layers.append((mask.detach(), priority))

    for entry in positive:
        if not isinstance(entry, (tuple, list)) or len(entry) < 2:
            continue
        cond_dict = entry[1]
        if not isinstance(cond_dict, dict):
            continue

        stack = cond_dict.get(ENERGY_LAYERS_KEY)
        if isinstance(stack, (list, tuple)) and stack:
            for layer in stack:
                if isinstance(layer, (tuple, list)) and len(layer) >= 2:
                    _add(layer[0], layer[1])
                else:
                    logger.warning(
                        "[Energy Guidance] Ignoring malformed energy layer %r.",
                        type(layer).__name__,
                    )
            continue

        # Conditioning written by 3.3.0-beta, or by hand, carries only the
        # singular keys.
        if ENERGY_MASK_KEY in cond_dict:
            _add(cond_dict[ENERGY_MASK_KEY], cond_dict.get(ENERGY_PRIORITY_KEY, 1.0))

    return layers


def _make_energy_cfg_patch(layers, latent_shapes=None):
    """Build the EPS sampler_post_cfg_function from one or more energy layers.

    *layers* is a sequence of ``(mask, priority)`` pairs, as returned by
    `_collect_energy_layers`.

    *latent_shapes* is the per-modality shape list captured from the work
    latent at registration time (mirrors how the LTX-AV dual-CFG patch above
    gets its own video/audio split from `work_latent.unbind()`): None for a
    plain latent, or ``[tuple(p.shape) for p in work_latent.unbind()]`` when
    nested. Lets EPS mask the video stream of a packed multi-modality latent
    (e.g. LTX-AV) instead of no-op'ing on it; see `_energy_prioritized_cfg_patch`.

    Scales the guidance vector (cond − uncond) by ``1 + Σ priority_i · mask_i``,
    leaving unmasked regions at the sampler's own cfg.

    Summing is what makes chained Energy Mask nodes stack. It also means a
    stack of negative priorities can drive the modifier below zero, which would
    *invert* guidance rather than suppress it — a much louder artefact than the
    user asked for — so the sum is floored at −1 before the +1, i.e. the
    modifier is clamped to ``[0, ∞)``.

    Shape handling is deliberately rank-agnostic. The previous version
    unpacked `B, C, H_l, W_l = cond.shape`, which raises ValueError on the
    5-D (B, C, T, H, W) latents every video model here produces — WAN,
    Hunyuan, LTX — so EPS would have crashed the sampler on video the moment
    a producer existed. It also called `mask.expand(B, ...)` on a mask whose
    batch was neither 1 nor B, which expand() cannot do.
    """
    _eps_layers = [
        (m.detach() if hasattr(m, "detach") else m, float(p)) for m, p in layers
    ]
    # Resizing the masks is per-step work that only depends on the latent
    # geometry, which never changes mid-sample. Cache the summed field.
    _cache: Dict[Any, Any] = {}
    _warned_bad_shape = [False]

    def _energy_field(cond):
        """Σ priority_i · mask_i, resampled to the latent grid."""
        key = (tuple(cond.shape), str(cond.device), str(cond.dtype))
        cached = _cache.get(key)
        if cached is not None:
            return cached

        import torch.nn.functional as F

        B = cond.shape[0]
        H_l, W_l = cond.shape[-2], cond.shape[-1]
        # Latent temporal length, for 5D (B, C, T, H, W) video latents.
        T_l = cond.shape[2] if cond.ndim == 5 else None

        total = None
        for mask, priority in _eps_layers:
            m = mask.to(device=cond.device, dtype=cond.dtype)
            # MASK is (H,W) or (B,H,W); a latent-shaped mask may arrive as
            # (B,1,H,W) and a video mask as (B,1,T,H,W). Only the trailing two
            # dims are spatial, so fold everything else into the batch axis.
            n_slices = 1
            for d in m.shape[:-2]:
                n_slices *= int(d)
            m = m.reshape(n_slices, 1, m.shape[-2], m.shape[-1])

            if m.shape[-2] != H_l or m.shape[-1] != W_l:
                m = F.interpolate(m, size=(H_l, W_l), mode="bilinear", align_corners=False)

            # DEFECT: an ANIMATED mask was silently collapsed to its first frame.
            # For a (1, 1, 21, 480, 832) per-frame video mask the reshape above
            # folds T into dim 0, giving 21 slices; the old code then saw
            # `m.shape[0] != B` (21 != 1), took the `m[:1].expand(B, ...)` branch
            # meant for genuine batch mismatches, and applied FRAME 0's mask to
            # all 21 latent frames. No warning, and the result looks plausible.
            #
            # Recognise the animated case first: when the folded slice count
            # matches the latent's temporal length, those slices ARE the temporal
            # axis and belong on dim 2, not dim 0.
            if T_l is not None and m.shape[0] == T_l * B and T_l > 1:
                # (B*T, 1, H, W) -> (B, 1, T, H, W)
                m = m.reshape(B, T_l, 1, H_l, W_l).permute(0, 2, 1, 3, 4)
                contribution = m * priority
                total = contribution if total is None else total + contribution
                continue

            if m.shape[0] != B:
                if T_l is not None and T_l > 1 and m.shape[0] > 1:
                    logger.warning(
                        "[Energy Guidance] Mask has %d frame(s) but the latent has "
                        "%d temporal frame(s) and batch %d. Using the first frame's "
                        "mask for the whole clip. Supply a mask with %d frames to "
                        "animate it.",
                        m.shape[0], T_l, B, T_l * B,
                    )
                # expand() only broadcasts from 1. For any other mismatch (a
                # 3-frame mask against a 4-latent batch) fall back to the first
                # mask rather than raising in the middle of sampling.
                m = m[:1].expand(B, -1, -1, -1)

            if cond.ndim > 4:
                # (B,1,H,W) → (B,1,1,…,H,W) so the modifier broadcasts across
                # the temporal axis as well as channels.
                m = m.reshape(B, 1, *([1] * (cond.ndim - 4)), H_l, W_l)

            contribution = m * priority
            total = contribution if total is None else total + contribution

        if total is None:
            total = torch.zeros((), device=cond.device, dtype=cond.dtype)

        _cache.clear()
        _cache[key] = total
        return total

    def _boosted_cond(cond, uncond):
        """cond, pushed further from uncond in high-energy mask regions."""
        eps_modifier = (1.0 + _energy_field(cond)).clamp_min(0.0)
        return uncond + (cond - uncond) * eps_modifier

    def _energy_prioritized_cfg_patch(args):
        cond = args["cond_denoised"]
        uncond = args["uncond_denoised"]
        cfg_val = args["cond_scale"]

        # ALBABIT-FIX: a packed multi-modality latent (e.g. LTX-AV) arrives
        # as a flat (B, 1, N) tensor with no spatial grid to overlay the mask
        # against. Unpack (using shapes captured at registration time) to
        # recover the video stream's real grid, mask that alone, leave any
        # other stream (audio) untouched, then re-pack.
        if latent_shapes and len(latent_shapes) > 1 and cond.ndim < 4 and cond.shape == uncond.shape:
            try:
                cond_parts = comfy.utils.unpack_latents(cond, latent_shapes)
                uncond_parts = comfy.utils.unpack_latents(uncond, latent_shapes)
                video_cond, video_uncond = cond_parts[0], uncond_parts[0]
                if video_cond.ndim >= 4 and video_cond.shape == video_uncond.shape:
                    cond_parts[0] = _boosted_cond(video_cond, video_uncond)
                    packed_cond, _ = comfy.utils.pack_latents(cond_parts)
                    return uncond + cfg_val * (packed_cond - uncond)
            except Exception as _exc:
                logger.debug(
                    "[Energy Guidance] packed-latent unpack failed (%s: %s); "
                    "falling back to the unsupported-shape warning.",
                    type(_exc).__name__, _exc,
                )

        if cond.ndim < 4 or cond.shape != uncond.shape:
            if not _warned_bad_shape[0]:
                logger.warning(
                    "[Energy Guidance] Latent shape %s isn't supported (need "
                    "4+ dims, or a multi-modality latent with video as the "
                    "first stream). EPS has no effect this run.", tuple(cond.shape),
                )
                _warned_bad_shape[0] = True
            return args["denoised"]

        return uncond + cfg_val * (_boosted_cond(cond, uncond) - uncond)

    return _energy_prioritized_cfg_patch


def make_temporal_window_wrapper(
    window_size: int,
    overlap: int,
    previous_wrapper=None,
    on_plan=None,
):
    """Build the UNet wrapper that runs each denoising step in temporal windows.

    Registered with ModelPatcher.set_model_unet_function_wrapper, so ComfyUI
    calls it in place of `model.apply_model` for every model evaluation
    (comfy/samplers.py, `calc_cond_batch`). One call == one denoising step for
    one conditioning batch, which is what makes the blend below a PER-STEP
    blend: the windows are recombined before the sampler takes its next step,
    never after a window has been denoised to completion.

    Why the wrapper and not an outer per-step loop over sample_custom: the
    sampler keeps state across steps (dpmpp_2m carries the previous denoised
    estimate, ancestral samplers carry their noise schedule), and restarting it
    once per window per step would silently degrade every multi-step solver to
    first order. Here the solver sees one continuous trajectory over the whole
    clip and only the model evaluation underneath it is windowed, so the
    schedule, the step count, the callback and the seed all behave exactly as
    they do unwindowed.

    What this does and does not bound: peak activation memory becomes a
    function of *window_size* rather than of clip length, because attention
    only ever runs over one window. The latent itself is still allocated in
    full, which is comparatively small, so this lifts the wall that long video
    actually hits without pretending the latent is free.

    *previous_wrapper* is composed rather than clobbered, so a wrapper another
    node installed first still runs.
    """
    def wrapper(apply_model, args):
        x = args["input"]
        t = args["timestep"]
        c = args["c"]

        if previous_wrapper is not None:
            def _call(xx, tt, cc):
                inner = dict(args)
                inner["input"], inner["timestep"], inner["c"] = xx, tt, cc
                return previous_wrapper(apply_model, inner)
        else:
            def _call(xx, tt, cc):
                return apply_model(xx, tt, **cc)

        # Only 5D (B, C, T, H, W) video latents have a temporal axis to window.
        # Packed multi-modality latents (LTX-AV, MiniMax H3) reach here already
        # flattened to (B, 1, N) and are passed straight through.
        if not isinstance(x, torch.Tensor) or x.ndim != 5:
            return _call(x, t, c)

        total_frames = x.shape[2]
        windows = plan_temporal_windows(total_frames, window_size, overlap)
        if on_plan is not None:
            on_plan(windows, total_frames)
        if len(windows) <= 1:
            # Single window: the same call the unwindowed path makes, so the
            # result is bit-identical rather than merely close.
            return _call(x, t, c)

        # ComfyUI computes `c['control']` from the FULL latent before calling
        # this wrapper (comfy/samplers.py, just above the model_function_wrapper
        # branch), so a ControlNet's hint tensors are clip-length while the
        # model is handed one window. Already-computed ControlBase outputs
        # cannot be sliced generically here.
        if c.get("control") is not None:
            raise ValueError(
                "Temporal windowing cannot align full-clip ControlNet outputs "
                "with individual windows. Set temporal_window=0 for ControlNet runs."
            )

        accum = torch.zeros_like(x)
        weights = torch.zeros(
            (1, 1, total_frames, 1, 1), device=x.device, dtype=x.dtype
        )

        for wi, (f0, f1) in enumerate(windows):
            w = temporal_window_weights(
                windows, wi, device=x.device, dtype=x.dtype
            )
            out_w = _call(
                x[:, :, f0:f1],
                t,
                slice_conds_temporally(c, f0, f1, total_frames),
            )
            accum[:, :, f0:f1] += out_w * w
            weights[:, :, f0:f1] += w
            del out_w

        # Weights form a partition of unity, including three-window joins.
        # Normalize floating-point accumulation before returning.
        return accum / weights

    return wrapper


def _resolve_latent_previewer(model, preview_method: str):
    """Build a ComfyUI latent previewer for *preview_method*, or None.

    DEFECT GUARD: the "TAESD" branch used to import a `TAESDDecoder` symbol that
    has never existed in comfy.taesd.taesd (the module exports TAESD), so it
    always fell into the except-ImportError path and silently downgraded to
    "Latent2RGB", and neither branch decoded anything, because the imported
    symbol was never used. latent_preview.get_previewer() is ComfyUI's own
    implementation: it loads the taesd weights out of models/vae_approx, falls
    back to Latent2RGB by itself when they are missing, and returns an object
    whose decode_latent_to_preview_image() yields the (format, PIL.Image,
    max_size) triple the /progress socket requires.
    """
    if not preview_method or preview_method == "None":
        return None

    try:
        import latent_preview  # ComfyUI top-level module
        from comfy.cli_args import args as comfy_args, LatentPreviewMethod
    except ImportError as exc:
        logger.warning(
            "[Radiance] Latent preview unavailable (%s: %s).", type(exc).__name__, exc
        )
        return None

    method = LatentPreviewMethod.from_string(preview_method.lower())
    if method is None:
        logger.warning(
            "[Radiance] Unknown preview_method '%s'; no preview will be sent.",
            preview_method,
        )
        return None

    latent_format = getattr(getattr(model, "model", None), "latent_format", None)
    if latent_format is None:
        logger.warning(
            "[Radiance] Model exposes no latent_format; cannot build a previewer."
        )
        return None

    device = getattr(model, "load_device", None)
    if device is None:
        device = comfy.model_management.get_torch_device()

    # get_previewer() reads the global --preview-method. Override it for the
    # duration of the call so the node's widget wins, then put the launch
    # setting back so other nodes in the same prompt are unaffected.
    previous_method = getattr(comfy_args, "preview_method", None)
    try:
        comfy_args.preview_method = method
        previewer = latent_preview.get_previewer(device, latent_format)
    except Exception as exc:
        logger.warning(
            "[Radiance] Could not build the '%s' previewer (%s: %s).",
            preview_method, type(exc).__name__, exc,
        )
        return None
    finally:
        if previous_method is not None:
            comfy_args.preview_method = previous_method

    if previewer is None:
        logger.warning(
            "[Radiance] ComfyUI has no previewer for this model's latent format "
            "with preview_method='%s'.", preview_method,
        )
    return previewer


def _make_phase_preview_callback(pbar, previewer, phase_start_step: int):
    """Progress callback that reports a global step and a real preview image.

    DEFECT GUARD: the old callback handed ProgressBar.update_absolute a bare
    `(x0,)` 1-tuple of raw latents. server.py's send_image_with_metadata()
    indexes image_data[2], so every previewed step raised IndexError server
    side. decode_latent_to_preview_image() returns the 3-tuple the protocol
    defines (PreviewImageTuple in comfy_execution/progress.py).
    """

    def callback(step, x0, x1, total_steps):
        global_step = step + phase_start_step
        preview = None
        if previewer is not None:
            try:
                latent = x0
                if getattr(latent, "is_nested", False):
                    latent = latent.tensors[0]
                preview = previewer.decode_latent_to_preview_image("JPEG", latent)
            except Exception as exc:
                logger.debug(
                    "[Radiance] Preview decode failed at step %d (%s: %s); "
                    "sending progress without an image.",
                    global_step, type(exc).__name__, exc,
                )
        # ALBABIT-FIX: the bar keeps its own total, every stage's iterations.
        # total_steps is this stage's alone, which filled it after stage one.
        pbar.update_absolute(global_step + 1, preview=preview)

    return callback


def _sample_custom_progress_safe(context: str, **kwargs):
    """Run Comfy sampling, retrying once with progress output disabled if stderr is broken."""

    try:
        return comfy.sample.sample_custom(**kwargs)
    except Exception as exc:
        if _is_invalid_progress_stream_error(exc) and not kwargs.get("disable_pbar"):
            retry_kwargs = dict(kwargs)
            retry_kwargs["disable_pbar"] = True
            retry_kwargs["callback"] = None
            logger.warning(
                "[%s] Comfy progress output failed with OSError(22). "
                "Retrying with progress display disabled; sampling settings are unchanged.",
                context,
            )
            return comfy.sample.sample_custom(**retry_kwargs)

        # BUG FIX: RuntimeErrors from a latent/model channel-count mismatch
        # (e.g. RadianceResolution's model_type wasn't updated after switching
        # a Load Preset to a different architecture) surface as a raw, cryptic
        # PyTorch shape error. Re-raise with the original error preserved as
        # the cause, plus an actionable hint, instead of hiding or altering it.
        if isinstance(exc, RuntimeError) and _is_channel_mismatch_error(exc):
            raise RuntimeError(
                f"[{context}] Sampling failed with a tensor shape/channel mismatch "
                f"(original error below). This usually means the LATENT's channel "
                f"count doesn't match what the connected MODEL expects — most often "
                f"because RadianceResolution's 'model_type' (and/or 'latent_channels') "
                f"still matches a previous model after the checkpoint/Load Preset was "
                f"switched to a different architecture (e.g. SDXL's 4ch vs Flux's 16ch). "
                f"Check that RadianceResolution's model_type matches the model currently "
                f"loaded, then regenerate the latent.\n\nOriginal error: {exc}"
            ) from exc
        raise

class RadianceSamplerPro:
    """
    Universal diffusion sampler — Flux, SD3, SDXL, WAN, LTX,
    HunyuanVideo, Lumina2, Chroma.  Supports phase-shift sampling, AYS
    schedules, PAG, tiled sampling, multi-conditioning, noise types,
    refiner chaining, and restart (IRES) sampling.
    """
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Generate"
    DESCRIPTION = "Universal diffusion sampler with phase-shift, PAG, AYS, tiling, restart, and multi-model support."

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "model": ("MODEL", {"tooltip": (
                    "Diffusion model to sample with. It is cloned before any patch (PAG, guidance "
                    "rescale, SDR anchor, windowing), so the loader's model is not modified.")}),
                "positive": ("CONDITIONING", {"tooltip": (
                    "Prompt conditioning to steer toward. For guidance-embedded models (Flux, Flux.2, "
                    "LTXV, LongCat) flux_guidance is written into it.")}),
                "negative": ("CONDITIONING", {"tooltip": (
                    "Conditioning to steer away from. Only evaluated when cfg is above 1.0; at 1.0 "
                    "ComfyUI skips the negative pass.")}),
                "latent_image": ("LATENT", {"tooltip": (
                    "Starting latent: empty for text-to-image, VAE-encoded for img2img (lower denoise). "
                    "A noise_mask on it limits sampling to the masked area.")}),
                "preset": (WORKFLOW_PRESETS, {"default": "Auto", "tooltip": (
                    "Auto and Custom let the node replace cfg, flux_guidance, steps and "
                    "sampler with the detected model's defaults while they sit at their widget defaults. "
                    "A named preset only fills the widgets in the UI; at run time the widget values are used as-is.")}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 200, "step": 1,
                    "tooltip": "Total denoising steps. More steps = higher quality but slower. 20–30 is typical for most samplers."
                }),
                "start_step": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 200,
                        "step": 1,
                        "tooltip": "Start step (0 = beginning)",
                    },
                ),
                "end_step": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 200,
                        "step": 1,
                        "tooltip": "End step (0 = use total steps)",
                    },
                ),
                "cfg": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1,
                     "tooltip": (
                         "Classifier-free guidance scale. 1.0 runs one model pass per step and ignores the "
                         "negative; above 1.0 adds a negative pass (about twice the time). Keep 1.0 for Flux "
                         "and distilled or turbo models and use flux_guidance instead.")},
                ),
                "audio_cfg": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "tooltip": "LTX-AV only (e.g. LTX 2.5): separate CFG scale for the audio "
                                   "half of the latent. 0 = same as cfg (single-CFG, pre-2.5 behavior).",
                    },
                ),
                "sampler": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": (
                    "ComfyUI sampling algorithm. With preset Auto or Custom, 'euler' is replaced by the "
                    "detected model's recommended sampler, if it has one.")}),
                "sampler_mode": (SamplerMode.ALL, {"default": SamplerMode.STANDARD, "tooltip": (
                    "Standard: one sampler throughout. Phase-Shift: switch at phase_split to dpmpp_2m "
                    "(DPM) or to the same sampler on the sgm_uniform schedule (SGM); not used for video "
                    "models. CFG++: scales cfg toward 1.0 by a cosine of each stage's starting sigma, so a "
                    "plain single-stage run keeps cfg unchanged.")}),
                "phase_split": (
                    "FLOAT",
                    {"default": 0.40, "min": 0.0, "max": 1.0, "step": 0.05,
                     "tooltip": "Phase-Shift modes only: fraction of the total steps at which the second sampler takes over (0.4 of 20 steps = step 8)."},
                ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": (
                    "ComfyUI noise schedule used to build the sigmas. Ignored when ays_schedule, "
                    "sigmas_override or the SD/SDXL Turbo schedule is in use.")}),
                "scheduler_mode": (
                    ["Manual", "Auto (Match Steps)"],
                    {"default": "Manual", "tooltip": (
                        "Manual uses the scheduler widget. Auto replaces it with the detected model's "
                        "default scheduler (it does not depend on the step count).")},
                ),
                "denoise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                     "tooltip": (
                         "Fraction of the noise schedule to run. 1.0 starts from pure noise; lower values "
                         "skip the noisiest steps and keep more of latent_image (img2img). Fewer steps run "
                         "unless force_exact_steps is on.")},
                ),
                "flux_shift": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.1,
                     "tooltip": (
                         "Extra time-shift applied to the sigma schedule, shift*s / (1 + (shift-1)*s), on top "
                         "of the model's own shift. 1.0 = off; higher spends more steps at high noise. Intended "
                         "for flow-matching models (sigmas 0-1). Auto/Custom preserve this value.")},
                ),
                "flux_guidance": (
                    "FLOAT",
                    {"default": 3.5, "min": 0.0, "max": 20.0, "step": 0.1,
                     "tooltip": (
                         "Embedded guidance written into the positive conditioning for Flux, Flux.2, LTXV and "
                         "LongCat; ignored by other models. Auto/Custom replace 3.5 with the model default "
                         "(for example 0 for Schnell).")},
                ),
                "flux_guidance_profile": (
                    ["Static", "Dynamic (Creative Start/End)"],
                    {"default": "Static", "tooltip": (
                        "Static: one value for the whole run. Dynamic: splits the run into stages; "
                        "guidance-embedded models start at 0.6x flux_guidance and end slightly lower, CFG models "
                        "start at 1.2x cfg and end at 0.7x (cfg above 1.0 only). Stage boundaries include "
                        "ramp ends so the middle reaches 1.0x when the step range includes it. "
                        "Short or partial runs may omit phases. Ignored in tile_mode.")},
                ),
                # ALBABIT-FIX: 1 and fixed by default. Seed 0 breaks Qwen-Image 2.1 edits,
                # native nodes too; fixed reruns reproduce the last image.
                "seed": ("INT", {"default": 1, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                    "control_after_generate": "fixed",
                    "tooltip": "Random seed for reproducible results. Use the control below it (randomize / increment / fixed) to vary the seed between runs."
                }),

                "pag_scale": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 5.0,
                        "step": 0.1,
                        "tooltip": "Perturbed-attention guidance strength (0 = off). It perturbs the unconditional pass, so it only works with cfg above 1.0.",
                    },
                ),

                "model_type": (MODEL_TYPES, {"default": "auto", "tooltip": (
                    "Model family used for defaults, guidance handling and schedules. auto detects it from "
                    "the model (or from model_meta when connected); set it by hand if detection is wrong.")}),
                "sigma_blend_steps": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 10,
                        "step": 1,
                        "tooltip": "Smooth sigma transition steps at phase-shift boundary",
                    },
                ),

                "guidance_rescale_phi": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Guidance rescale (Imagen). 0=off, 0.7=recommended for SDXL. Prevents oversaturation at high CFG.",
                    },
                ),

                "preview_method": (PREVIEW_METHODS, {"default": "None", "tooltip": (
                    "Live preview during sampling. TAESD needs the taesd weights in models/vae_approx and "
                    "falls back to Latent2RGB without them. None sends no Radiance preview, only ComfyUI's progress bar.")}),

                "noise_type": (
                    NOISE_TYPES,
                    {"default": "Gaussian",
                     "tooltip": ("Noise generation algorithm. Perlin=coherent structure, "
                                 "Spectral=pink/1f noise, Brownian=video-correlated, Uniform=flat distribution.")},
                ),

                "conditioning_clip_target": (
                    CLIP_TARGETS,
                    {"default": "Auto",
                     "tooltip": ("Has no effect. Kept so saved workflows load; any value other than Auto only logs a "
                                 "warning. Choose the text encoder at encode time instead (e.g. CLIPTextEncodeSDXL).")},
                ),

                "add_noise": ("BOOLEAN", {"default": True,
                    "tooltip": "Inject fresh noise at the start of sampling. Disable for img2img-style passes that should preserve structure."
                }),
                "return_with_leftover_noise": ("BOOLEAN", {"default": False,
                    "tooltip": "Return the latent with residual noise un-removed. Useful for multi-pass workflows."
                }),
                "ays_schedule": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Use AYS (Align Your Steps) research-optimized sigma schedule. Best at 8-15 steps.",
                    },
                ),

                "tile_mode": (
                    "BOOLEAN",
                    {"default": False,
                     "tooltip": "Enable tiled sampling for memory-efficient high-resolution generation."},
                ),
                "tile_size": (
                    "INT",
                    {"default": 128, "min": 32, "max": 1024, "step": 32,
                     "tooltip": "Tile size in latent pixels (128 latent ≈ 1024px output with VAE factor 8)."},
                ),
                "tile_overlap": (
                    "INT",
                    {"default": 16, "min": 0, "max": 256, "step": 8,
                     "tooltip": "Overlap between adjacent tiles to reduce seam artifacts."},
                ),
                "tile_blend": (
                    TILE_BLEND_MODES,
                    {"default": "feather",
                     "tooltip": "Seam blending method. feather=cosine fade, gaussian=bell curve, average=uniform."},
                ),

                "terminal_sigma_to_zero": (
                    "BOOLEAN",
                    {"default": False,
                     "tooltip": "Ensure terminal step reaches zero noise even on truncated image-to-image runs. Vital for Flow Matching models."},
                ),
                "force_exact_steps": (
                    "BOOLEAN",
                    {"default": False,
                     "tooltip": "Ensure precise step count in image-to-image runs, adjusting calculations rather than purely truncating stages."},
                ),

                # ALBABIT-FIX: Removed UI normalization factors entirely. Native LTX handles recombination cleanly.
            },
            "optional": {
                "refiner_model": ("MODEL", {"tooltip": (
                    "Optional second model that takes over from refiner_start_step with the same conditioning, "
                    "so it must accept the same text-encoder width. PAG, guidance rescale and the SDR anchor are "
                    "not applied to it. Ignored in tile_mode.")}),
                "refiner_start_step": (
                    "INT",
                    {"default": 20, "min": 0, "max": 200, "step": 1,
                     "tooltip": "Step index (0-based, out of steps) where refiner_model takes over. At or above the last step being run, the refiner never runs."},
                ),
                "noise_override": ("LATENT", {"tooltip": (
                    "Use this latent's samples as the initial noise instead of generating it from seed and "
                    "noise_type. Its shape must match latent_image exactly.")}),

                "sigmas_override": (
                    "SIGMAS",
                    {"tooltip": "Inject a pre-computed sigma schedule. Bypasses all internal sigma computation."},
                ),

                # ── Workflow-compat absorbers (positions 38/39/40 in old saves) ─
                # Old workflow JSONs serialised two JS button widgets (null, null)
                # and the preset_info text widget at these positions. STRING type
                # absorbs null/any-string without a validation error, so the new
                # feature inputs below land at 41/42/43 where they are absent
                # from old JSON → ComfyUI falls back to their defaults.
                "_js_export_btn": ("STRING", {"default": "", "multiline": False,
                    "tooltip": "JS serialization placeholder (not user-editable)."}),
                "_js_import_btn": ("STRING", {"default": "", "multiline": False,
                    "tooltip": "JS serialization placeholder (not user-editable)."}),
                "_js_preset_info": ("STRING", {"default": "", "multiline": False,
                    "tooltip": "JS serialization placeholder (not user-editable)."}),

                # ── Restart sampling ─────────────────────────────────────────
                "restart_count": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4,
                        "step": 1,
                        "tooltip": (
                            "Number of restart iterations at each restart_schedule sigma. "
                            "0 = disabled. 1–2 restarts add ~5% extra steps but measurably "
                            "improve high-frequency detail on Flux and WAN."
                        ),
                    },
                ),

                # ── Noise alpha schedule ──────────────────────────────────────
                "noise_alpha_start": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": (
                            "Weight of the selected noise_type against Gaussian at step 0 "
                            "(1.0 = pure noise_type; no effect with Gaussian). The noise is "
                            "injected once, so the start/end cosine ramp is read only at "
                            "start_step: on a run from step 0 only this value matters."
                        ),
                    },
                ),
                "noise_alpha_end": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": (
                            "Weight of the selected noise_type at the final step of the ramp. "
                            "The noise is injected once, at start_step, so this only has an "
                            "effect when start_step is above 0 (partial or multi-pass runs)."
                        ),
                    },
                ),

                # ── Custom AYS anchors ────────────────────────────────────────
                "custom_ays_anchors": (
                    "SIGMAS",
                    {
                        "tooltip": (
                            "Optional model-specific AYS anchor schedule. "
                            "Overrides the built-in AYS tables when ays_schedule=True. "
                            "Must be a monotonically decreasing tensor ending at 0."
                        ),
                    },
                ),

                # ── Restart sigma schedule (socket-only — link a SIGMAS node) ──
                "restart_schedule": (
                    "SIGMAS",
                    {
                        "tooltip": (
                            "Optional list of sigma levels at which to re-inject noise "
                            "and re-denoise (Restart / IRES style). Improves fine detail "
                            "at fixed step count. Requires restart_count > 0."
                        ),
                    },
                ),
                # ── SDR conditioning ─────────────────────────
                "sdr_reference": ("IMAGE", {"tooltip": (
                    "Optional structure reference: a display-referred 0-1 image, VAE-encoded with sdr_vae and "
                    "resized to the latent. Needs sdr_vae connected and sdr_blend above 0.")}),
                "sdr_vae": ("VAE", {"tooltip": "VAE used to encode sdr_reference. Use the VAE that matches the model."}),
                "sdr_blend": (
                    "FLOAT",
                    {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05,
                     "tooltip": (
                         "Weight of the encoded reference: it is mixed into the starting latent at this weight, "
                         "and is the starting weight of the per-step anchor. 0 disables SDR conditioning.")},
                ),
                "sdr_inject_steps": (
                    "INT",
                    {"default": 6, "min": 0, "max": 100, "step": 1,
                     "tooltip": (
                         "Number of model evaluations after CFG in which the denoised result is pulled toward the "
                         "reference (0 = only the starting-latent mix). Multi-evaluation samplers use these up "
                         "faster than one per step.")},
                ),
                "sdr_decay": (
                    "FLOAT",
                    {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.05,
                     "tooltip": "Per-evaluation falloff of the anchor: weight = sdr_blend x sdr_decay^n. Lower fades faster; 1.0 holds it constant."},
                ),
                # ── Model-aware auto defaults ─────────────────────────────────
                "model_meta": (
                    "STRING",
                    {
                        "default": "", "forceInput": True,
                        "tooltip": "Optional: connect the Loader's model_meta output. "
                                   "Only used when preset='Auto'/'Custom' and model_type='auto'. "
                                   "Refines cfg/guidance/steps beyond what the loaded model's "
                                   "architecture alone can tell -- e.g. distinguishing Flux.2 Klein "
                                   "Base from Klein distilled, which are architecturally identical.",
                    },
                ),
                # ── Long-video temporal windowing ─────────────────────────────
                # Off by default and inert on image latents, so no existing
                # graph changes behaviour.
                #
                # Appended AFTER every pre-existing optional input on purpose:
                # ComfyUI restores widget VALUES from an old workflow JSON by
                # position, so inserting these anywhere earlier would shift
                # restart_count, the noise alpha pair, the SDR block and
                # model_meta onto each other's saved values. Same reasoning as
                # the _js_* absorbers above.
                "temporal_window": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 512,
                        "step": 1,
                        "tooltip": (
                            "Experimental long-video windowing; may produce ghosting "
                            "or scene changes at joins. Use 0 for production work. "
                            "ControlNet requires 0 or a window covering the whole clip. "
                            "0 = off (whole clip denoised at "
                            "once, the previous behaviour). Above 0, this many LATENT "
                            "frames are denoised per window, with the windows blended "
                            "at every step, so peak VRAM follows the window size "
                            "instead of the clip length. Only applies to 5D video "
                            "latents longer than the window. 16-32 is a usual range; "
                            "smaller windows save more memory and give the model less "
                            "temporal context."
                        ),
                    },
                ),
                "temporal_overlap": (
                    "INT",
                    {
                        "default": 4,
                        "min": 0,
                        "max": 256,
                        "step": 1,
                        "tooltip": (
                            "Latent frames shared between neighbouring windows, used "
                            "to cross-fade them at every denoising step. Clamped to "
                            "half of temporal_window. More overlap increases compute "
                            "but does not guarantee coherent joins; 0 means hard cuts between "
                            "windows. Ignored when temporal_window is 0."
                        ),
                    },
                ),

            },
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("LATENT", "SIGMAS", "SIGMAS", "IMAGE")
    RETURN_NAMES = ("latent", "sigmas", "sigmas_remaining", "sigma_plot")
    OUTPUT_TOOLTIPS = (
        "Denoised latent ready for VAE decode.",
        "The full sigma schedule used — chain to another sampler or inspect.",
        "Unused tail of the sigma schedule after end_step. Chain directly to a refiner or upscaler sampler.",
        "Visual plot of the sigma schedule as an IMAGE. Blank if matplotlib is unavailable.",
    )
    FUNCTION = "sample"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Generate"
    DESCRIPTION = (
        "Universal diffusion sampler. Auto-detects model type (Flux, SD3, SDXL, "
        "WAN, LTX, HunyuanVideo, Lumina2, Chroma). Phase-shift sampling, AYS schedules, "
        "PAG, dynamic guidance, tiled sampling, multi-conditioning, noise types, refiner chain. "
        "Restart sampling (IRES style), noise alpha schedule, sigma plot output, "
        "custom AYS anchors, sigmas_remaining for multi-node chains. "
        "AVControl-style SDR reference conditioning: encode an SDR reference through the VAE "
        "and blend it into the initial latent + per-step post-CFG anchor for HDR structure preservation."
    )

    def _apply_presets(self, preset, **kwargs):
        if preset in ("Auto", "Custom"):
            return kwargs

        if preset not in WORKFLOW_PRESETS:
            logger.warning(f"[Radiance] Preset '{preset}' not found.")
            return kwargs

        # ALBABIT-FIX: presets fill the widgets in the front end, which marks later
        # edits ✎. Forcing their values here overrode those edits (cfg 1 -> 3 on
        # LTX HighRes: a second forward pass per step, 2.5x slower).
        logger.info(f"[Radiance] Preset '{preset}' active — relying strictly on UI parameters.")
        return kwargs

    @staticmethod
    def _resolved_values_ui(cfg, flux_guidance, flux_shift, sampler, steps):
        # ALBABIT-FIX: _configure_model_and_defaults() may change these without the
        # widgets showing it; onExecuted (radiance_sampler.js) syncs them to what
        # the run used, as Resolution does with its computed size.
        return {
            "resolved_cfg": [cfg], "resolved_flux_guidance": [flux_guidance],
            "resolved_flux_shift": [flux_shift], "resolved_sampler": [sampler],
            "resolved_steps": [steps],
        }

    @staticmethod
    def _model_chain_loras(prompt, unique_id):
        # ALBABIT-FIX: model_meta lists the Loader's own LoRAs; these are the
        # ones native loaders apply between the Loader and this node.
        loras = []
        node = prompt.get(str(unique_id)) if prompt else None
        while node:
            link = node["inputs"].get("model")
            node = prompt.get(str(link[0])) if isinstance(link, list) else None
            if node and node["class_type"] in ("LoraLoaderModelOnly", "LoraLoader"):
                name = node["inputs"].get("lora_name")
                if isinstance(name, str):
                    loras.append(name)
        return loras

    def _configure_model_and_defaults(self, model, model_type, preset, model_meta="", chain_loras=(), **kwargs):
        detected_type = detect_model_type(model) if model_type == "auto" else model_type

        # model_meta's arch is the Loader's own (already-correct) resolution --
        # prefer it over re-detecting from the loaded MODEL object, which can't
        # distinguish Flux.2 Dev from Flux.2 Klein either. Only applies in auto
        # mode, never overriding an explicit manual model_type.
        meta_arch, meta_unet_file, meta_loras = parse_model_meta(model_meta)
        if model_type == "auto" and meta_arch:
            detected_type = meta_arch

        # ALBABIT-FIX: "and model_type == auto" used to also gate this block --
        # redundant on top of the per-field "still at generic default" checks
        # below, which already protect any field the user set by hand,
        # regardless of how model_type itself was determined.
        if preset in ("Auto", "Custom"):
            defaults = get_model_defaults(detected_type)
            distilled = refine_distillation_from_meta(detected_type, meta_unet_file, [*meta_loras, *chain_loras])

            # Apply defaults if user has them at "default" values -- distilled
            # (e.g. SDXL/SD3.5 Turbo) takes priority over the generic default.
            # cfg 1.0 is both the widget default and the correct value for every
            # distilled / turbo checkpoint (Z-Image Turbo, SDXL Turbo, ...). It
            # used to be replaced by the architecture's base CFG whenever it
            # read 1.0, which on a turbo model is the wrong look AND a second
            # (unconditional) forward pass on every step, silently doubling
            # the time. It is only replaced now when model_meta names the exact
            # checkpoint, so the choice is known to be right; otherwise it is
            # kept and the base value is suggested.
            if kwargs.get('cfg') == 1.0 and detected_type != "flux":
                suggested = (distilled or {}).get("cfg", defaults.get("cfg", 1.0))
                if meta_unet_file and suggested != 1.0:
                    kwargs['cfg'] = suggested
                    logger.info(f"Auto-applied cfg={suggested} for {detected_type} "
                                f"({meta_unet_file})")
                elif suggested != 1.0:
                    logger.info(
                        f"cfg=1.0 kept for {detected_type}: without model_meta the "
                        f"checkpoint cannot be told apart from a distilled one. The base "
                        f"model's usual cfg is {suggested} (2 forward passes per step); "
                        f"connect the Loader's model_meta to apply it automatically.")

            # ALBABIT-FIX: no longer skipped for Flux.1, which blocked Schnell's
            # guidance 0; a plain "flux" still resolves to 3.5.
            if kwargs.get('flux_guidance') == 3.5:
                model_default_guidance = (distilled or {}).get(
                    "guidance", defaults.get("guidance", kwargs['flux_guidance']))
                if model_default_guidance > 0 or defaults.get("guidance_type") != "embedding":
                    kwargs['flux_guidance'] = model_default_guidance

            # ALBABIT-FIX: "steps" isn't always present in either dict (e.g.
            # Krea Dev's override is guidance-only; most architectures have no
            # generic default at all, steps stays manual) -- distilled takes
            # priority, defaults is a generic per-architecture fallback (only
            # a few architectures define one, e.g. Chroma).
            if kwargs.get('steps') == 20:
                model_default_steps = (distilled or {}).get("steps", defaults.get("steps"))
                if model_default_steps is not None:
                    kwargs['steps'] = model_default_steps
                    logger.info(f"Auto-applied steps={kwargs['steps']} for {detected_type} (from model_meta)")

            # The model's scheduler already includes its native shift. This
            # widget is an extra shift, so 1.0 must remain the identity.

            # ALBABIT-FIX: didn't check distilled -- SDXL Turbo's sampler
            # override (euler_ancestral) was silently ignored here, unlike
            # cfg/steps above which already checked it first.
            default_sampler = (distilled or {}).get("sampler", defaults.get("sampler", kwargs['sampler']))
            if kwargs.get('sampler') == "euler" and default_sampler != "euler":
                kwargs['sampler'] = default_sampler
                logger.info(f"Auto-applied sampler={kwargs['sampler']} for {detected_type}")

        # Auto-match scheduler
        if kwargs.get('scheduler_mode') == "Auto (Match Steps)":
            defaults = get_model_defaults(detected_type)
            auto_scheduler = defaults.get("scheduler", kwargs['scheduler'])
            if auto_scheduler != kwargs['scheduler']:
                logger.info(f"Auto scheduler: {kwargs['scheduler']} → {auto_scheduler}")
                kwargs['scheduler'] = auto_scheduler

        return detected_type, kwargs, meta_unet_file

    def _prepare_noise(self, latent_samples, seed, noise_type, noise_override, device, frames):
        if noise_override is not None:
            noise = noise_override["samples"]
            # A single-frame 5D latent (Qwen-Image) and its 4D form hold the
            # same noise; an override made from an empty 4D latent still fits.
            if (torch.is_tensor(noise) and noise.ndim == 4 and latent_samples.ndim == 5
                    and latent_samples.shape[2] == 1):
                noise = noise.unsqueeze(2)
            if noise.shape != latent_samples.shape:
                raise ValueError(f"noise_override shape {noise.shape} mismatch with latent {latent_samples.shape}")
        else:
            if noise_type == "Gaussian":
                noise = comfy.sample.prepare_noise(latent_samples, seed, None)
            else:
                noise = generate_noise(latent_samples, seed, noise_type, frames=frames)
        
        return noise.to(device)

    def _prepare_sigmas(self, model, detected_type, steps, denoise, flux_shift,
                        scheduler, ays_schedule, terminal_sigma_to_zero,
                        force_exact_steps, sigmas_override, device,
                        custom_ays_anchors=None, use_sd_turbo_schedule=False):
        if sigmas_override is not None:
            # ALBABIT-FIX: restore informational log when sigmas_override is active
            logger.info(
                "[Radiance] sigmas_override active — bypassing internal sigma computation. "
                "steps / denoise / scheduler / flux_shift / AYS and related settings are ignored."
            )
            sigmas = sigmas_override.to(device)

            # DEFECT: this used to run correct_sigma_end(sigmas) unconditionally,
            # one line after logging that internal sigma computation was being
            # bypassed. correct_sigma_end forces a non-zero terminal sigma to 0,
            # so a custom SIGMAS deliberately ending at e.g. 0.03 for a
            # leftover-noise multi-pass was silently fully denoised and the
            # second pass had nothing left to work on. "Bypassing" has to mean
            # bypassing. terminal_sigma_to_zero is the widget that asks for the
            # clamp, so honour it and nothing else.
            if terminal_sigma_to_zero:
                sigmas = correct_sigma_end(sigmas)
                logger.info(
                    "[Radiance] terminal_sigma_to_zero=True, clamping the supplied "
                    "schedule's final sigma to 0."
                )
            elif len(sigmas) > 0 and float(sigmas[-1]) > 0.0:
                logger.info(
                    "[Radiance] Supplied schedule ends at sigma=%.5f; leaving it "
                    "un-clamped so residual noise survives for a following pass. "
                    "Enable terminal_sigma_to_zero to force a full denoise.",
                    float(sigmas[-1]),
                )
            target_steps = max(1, len(sigmas) - 1)
            return sigmas, target_steps

        try:
            if ays_schedule:
                # Custom AYS anchors take priority over the built-in tables
                if custom_ays_anchors is not None:
                    sigmas = custom_ays_anchors.to(device)
                    logger.info(f"Using custom AYS anchors ({len(sigmas)} values)")
                else:
                    sigmas = get_ays_sigmas(detected_type, steps)

                if sigmas is not None:
                    if denoise < 1.0:
                        total_s = len(sigmas) - 1
                        if total_s > 0:
                            start_s = max(0, int(total_s * (1.0 - denoise)))
                            sigmas = sigmas[start_s:]
                    if custom_ays_anchors is None:
                        logger.info(f"Using AYS schedule for {detected_type} ({steps} steps)")
                else:
                    sigmas = get_flux_sigmas(
                        model, scheduler, steps, denoise, flux_shift,
                        force_full=terminal_sigma_to_zero, force_exact=force_exact_steps,
                    )
            elif use_sd_turbo_schedule:
                # ALBABIT-FIX: SD-Turbo/SDXL-Turbo's own discrete 10-timestep
                # schedule (see get_sd_turbo_sigmas) -- only reached when the
                # user hasn't explicitly asked for AYS instead.
                sigmas = get_sd_turbo_sigmas(model, steps, denoise)
                logger.info(f"Using SD-Turbo discrete schedule ({steps} steps, denoise={denoise:.2f})")
            else:
                sigmas = get_flux_sigmas(
                    model, scheduler, steps, denoise, flux_shift,
                    force_full=terminal_sigma_to_zero, force_exact=force_exact_steps,
                )

            sigmas = correct_sigma_end(sigmas)
            return sigmas, steps
        except ValueError as e:
            logger.error(f"Failed to calculate sigmas: {e}")
            raise

    # ── Noise alpha ramp ──────────────────────────────────────────────────────

    @staticmethod
    def _noise_alpha_at(
        step: int, total_steps: int, alpha_start: float, alpha_end: float
    ) -> float:
        """Cosine interpolation from *alpha_start* to *alpha_end* across the run.

        DEFECT: noise_alpha_end used to appear only inside the
        `_noise_alpha_ramp_active` predicate and an f-string. The tooltip and
        the comment beside it both promised a cosine interpolation from start to
        end across the denoising trajectory, and no such interpolation existed
        anywhere in this file, so the widget was inert.

        This is the promised curve. What it can be evaluated at is bounded by
        the architecture: the noise tensor is mixed into the latent once, by
        `noise_scaling(sigma[0], noise, latent)` inside ComfyUI's sampler, so
        there is exactly one injection point per run and the ramp is sampled
        there. That point is step 0 for a normal run, where the result is
        alpha_start and nothing changes, and `start_step` for a partial-denoise
        or multi-pass run, where the ramp does real work. An ancestral sampler's
        own per-step re-noising is generated inside ComfyUI and cannot be
        intercepted from a node; `sample()` warns when the end value has no
        injection point that can see it.
        """
        if total_steps <= 0:
            return alpha_start
        t = min(max(float(step) / float(total_steps), 0.0), 1.0)
        # (1 - cos(pi*t)) / 2: 0 at t=0, 1 at t=1, zero-derivative at both ends.
        w = (1.0 - math.cos(math.pi * t)) / 2.0
        return alpha_start + (alpha_end - alpha_start) * w

    # ── Sigma plot ────────────────────────────────────────────────────────────

    @staticmethod
    def _build_sigma_plot(sigmas: torch.Tensor, detected_type: str) -> torch.Tensor:
        """Render sigma schedule as a small (256×128) IMAGE tensor (B,H,W,3).
        Falls back to a blank IMAGE if matplotlib is unavailable."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import io
            import numpy as np

            vals = sigmas.cpu().float().numpy()
            steps = list(range(len(vals)))

            fig, ax = plt.subplots(figsize=(3.2, 1.6), dpi=80)
            ax.plot(steps, vals, color="#3266ad", linewidth=1.5)
            ax.fill_between(steps, vals, alpha=0.12, color="#3266ad")
            ax.set_xlabel("step", fontsize=7)
            ax.set_ylabel("sigma", fontsize=7)
            ax.set_title(f"sigma schedule  [{detected_type}]", fontsize=7)
            ax.tick_params(labelsize=6)
            fig.tight_layout(pad=0.4)

            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            plt.close(fig)
            buf.seek(0)

            from PIL import Image as PILImage
            img = PILImage.open(buf).convert("RGB")
            arr = np.array(img).astype(np.float32) / 255.0
            return torch.from_numpy(arr).unsqueeze(0)  # (1, H, W, 3)
        except Exception as e:
            logger.debug(f"sigma_plot: matplotlib unavailable or failed ({e}) — returning blank")
            return torch.zeros(1, 64, 128, 3)

    # ── Restart sampling ──────────────────────────────────────────────────────

    @staticmethod
    def _apply_restarts(
        latent: torch.Tensor,
        model,
        positive,
        negative,
        sampler_obj,
        sigmas: torch.Tensor,
        restart_schedule: torch.Tensor,
        restart_count: int,
        cfg: float,
        seed: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        IRES / Restart sampling: at each sigma level in restart_schedule,
        re-inject noise scaled to that sigma and run restart_count extra
        denoising steps from that level down to the next sigma in the
        main schedule.
        """
        if restart_count <= 0 or len(restart_schedule) == 0 or len(sigmas) < 2:
            return latent

        result = latent
        s_vals = sigmas.to(device)

        for r_sigma in restart_schedule.to(device):
            r_val = float(r_sigma)
            if r_val <= 0.0:
                continue

            # Find where this sigma sits in the main schedule
            diffs = (s_vals - r_val).abs()
            idx = int(diffs.argmin().item())
            if idx >= len(s_vals) - 1:
                continue

            # Run the restart segment all the way back DOWN to the end of the
            # schedule.
            #
            # This used to be `s_vals[idx : idx+restart_count+2]`, which stops a
            # couple of steps in and leaves the latent partially noised. Measured
            # on a 20-step Karras schedule (14.615 -> 0.0292): restart_sigma 2.0
            # with count 2 returned a latent still at sigma 0.7913, and
            # restart_sigma 5.0 left it at 2.6415. `_apply_restarts` is called
            # after the main schedule has already finished, so the segment has to
            # reach 0 or the output is simply noise.
            sub_sigmas = s_vals[idx:]
            if len(sub_sigmas) < 2:
                continue

            for _ in range(restart_count):
                # Xu et al. 2023 (Restart Sampling), Alg. 2: jumping from
                # sigma_min back to sigma_max adds noise of variance
                # sigma_max^2 - sigma_min^2. The old code used `randn * r_val`,
                # i.e. a standard deviation of r_val with no subtraction.
                sigma_max = float(r_val)
                sigma_min = float(sub_sigmas[-1])
                std = math.sqrt(max(sigma_max ** 2 - sigma_min ** 2, 0.0))
                noisy = result + torch.randn_like(result) * std
                try:
                    result = _sample_custom_progress_safe(
                        "RadianceSamplerPro restart",
                        model=model,
                        # ComfyUI applies model_sampling.noise_scaling(sigma0,
                        # noise, latent_image) internally, which for EPS models
                        # is `sigma0 * noise + latent_image`. Passing the noised
                        # latent as `noise=` therefore amplified the SIGNAL by
                        # (1 + sigma0) and scaled the noise by sigma0 on top --
                        # at restart_sigma 2.0 that is a 3x signal gain with 4x
                        # the intended noise. Hand it the already-noised latent
                        # and a zero noise tensor instead.
                        noise=torch.zeros_like(noisy),
                        cfg=cfg,
                        sampler=sampler_obj,
                        sigmas=sub_sigmas,
                        positive=positive,
                        negative=negative,
                        latent_image=noisy,
                        noise_mask=None,
                        callback=None,
                        disable_pbar=True,
                        seed=seed,
                    )
                except Exception as e:
                    logger.warning(f"Restart sampling failed at sigma={r_val:.4f}: {e}")
                    break

        return result

    # ── AVControl — SDR Reference Conditioning ───────────────────────────────

    # Inference only. .eval() does not clear requires_grad on parameters, so an
# unguarded forward still builds and retains an autograd graph.
    @torch.no_grad()
    def _encode_sdr_reference(self, ref, vae, work):
        # comfy.sd.VAE.encode returns the latent tensor; only the VAEEncode node
        # wraps it in {"samples": ...}. Indexing the tensor with "samples" made
        # every real VAE crash the sample. Accept both, and pass RGB the way
        # VAEEncode does: a VAE given an alpha channel fails.
        if ref.shape[-1] > 3:
            ref = ref[..., :3]
        res = vae.encode(ref)
        ref_latent = res["samples"] if isinstance(res, dict) else res
        if ref_latent.ndim == 5:
            # A video VAE encodes the still as a (B, C, T, h, w) clip; its
            # first latent frame is the image. A temporal VAE reads a batch of
            # references as one clip, so the later frames are dropped: say so.
            if ref_latent.shape[2] > 1:
                logger.warning(
                    "[SDR Conditioning] sdr_vae encoded the reference as a clip of %d latent "
                    "frames; only the first conditions the sample. Pass one image.",
                    ref_latent.shape[2])
            ref_latent = ref_latent[:, :, 0]
        B = work.shape[0]
        if ref_latent.shape[0] == 1 and B > 1:
            ref_latent = ref_latent.expand(B, -1, -1, -1)
            
        target_H, target_W = work.shape[-2], work.shape[-1]
        if ref_latent.shape[-2] != target_H or ref_latent.shape[-1] != target_W:
            import torch.nn.functional as F
            ref_latent = F.interpolate(ref_latent, size=(target_H, target_W), mode="nearest")
            
        target_C = work.shape[1]
        current_C = ref_latent.shape[1]
        if current_C > target_C:
            ref_latent = ref_latent[:, :target_C, :, :]
        elif current_C < target_C:
            import torch
            pad = torch.zeros(ref_latent.shape[0], target_C - current_C, ref_latent.shape[2], ref_latent.shape[3],
                              device=ref_latent.device, dtype=ref_latent.dtype)
            ref_latent = torch.cat([ref_latent, pad], dim=1)
            
        if work.ndim == 5:
            T = work.shape[2]
            ref_latent = ref_latent.unsqueeze(2).expand(-1, -1, T, -1, -1)
            
        return ref_latent.to(dtype=work.dtype, device=work.device)



    def sample(
        self,
        model,
        positive: List,
        negative: List,
        latent_image: Dict[str, torch.Tensor],
        preset: str,
        steps: int,
        start_step: int,
        end_step: int,
        cfg: float,
        audio_cfg: float,
        sampler: str,
        sampler_mode: str,
        phase_split: float,
        scheduler: str,
        scheduler_mode: str,
        denoise: float,
        flux_shift: float,
        flux_guidance: float,
        flux_guidance_profile: str,
        add_noise: bool,
        return_with_leftover_noise: bool,
        seed: int,
        pag_scale: float = 0.0,
        model_type: str = "auto",
        sigma_blend_steps: int = 0,
        ays_schedule: bool = False,
        guidance_rescale_phi: float = 0.0,
        preview_method: str = "None",
        noise_type: str = "Gaussian",
        conditioning_clip_target: str = "Auto",
        tile_mode: bool = False,
        tile_size: int = 128,
        tile_overlap: int = 16,
        tile_blend: str = "feather",
        refiner_model=None,
        refiner_start_step: int = 20,
        noise_override: Optional[Dict[str, torch.Tensor]] = None,
        sigmas_override: Optional[torch.Tensor] = None,
        latent_format: str = "",
        terminal_sigma_to_zero: bool = False,
        force_exact_steps: bool = False,
        # ── Workflow-compat absorbers (swallow old JS-serialised widget values) ─
        _js_export_btn=None,
        _js_import_btn=None,
        _js_preset_info=None,
        # ── Custom AYS anchors, restart sampling, noise alpha schedule ─────────
        custom_ays_anchors: Optional[torch.Tensor] = None,
        restart_schedule: Optional[torch.Tensor] = None,
        restart_count: int = 0,
        noise_alpha_start: float = 1.0,
        noise_alpha_end: float = 1.0,
        temporal_window: int = 0,
        temporal_overlap: int = 4,
        sdr_reference: Optional[torch.Tensor] = None,
        sdr_vae: Optional[Any] = None,
        # Same defaults as the widgets (0.35, 6). They were 0.0 and 0 here, so
        # an API prompt that connected an SDR reference but left these out
        # ran with SDR guidance off.
        sdr_blend: float = 0.35,
        sdr_inject_steps: int = 6,
        sdr_decay: float = 0.65,
        model_meta: str = "",
        prompt=None,
        unique_id=None,
    ) -> Tuple:

        t_start = time.time()
        timings: Dict[str, float] = {}

        # ALBABIT-FIX: refiner_model swaps only the MODEL mid-run, never the
        # CONDITIONING -- a mismatched context_dim crashes deep in the sampling
        # loop with a cryptic PyTorch shape error. Fail early and clearly
        # instead (e.g. SDXL Base's 2048-dim clip_l+clip_g conditioning vs
        # Refiner's 1280-dim clip_g-only).
        if refiner_model is not None and positive:
            try:
                refiner_dim = refiner_model.model.model_config.unet_config.get("context_dim")
                cond_dim = positive[0][0].shape[-1]
            except AttributeError:
                refiner_dim = cond_dim = None
            if refiner_dim is not None and refiner_dim != cond_dim:
                raise RuntimeError(
                    f"refiner_model expects conditioning with {refiner_dim} dimensions, "
                    f"but positive/negative were encoded with {cond_dim}.\n\n"
                    f"Encode the prompt with a CLIP compatible with refiner_model "
                    f"(e.g. SDXL Refiner needs its own clip_g-only encode, not the "
                    f"Base model's clip_l+clip_g)."
                )

        # 1. Apply Presets
        params = self._apply_presets(preset, 
            steps=steps, cfg=cfg, sampler=sampler, scheduler=scheduler, 
            denoise=denoise, flux_shift=flux_shift, flux_guidance=flux_guidance,
            sampler_mode=sampler_mode, ays_schedule=ays_schedule, model_type=model_type)
        
        # Unpack params
        steps, cfg, sampler, scheduler = params['steps'], params['cfg'], params['sampler'], params['scheduler']
        denoise, flux_shift, flux_guidance = params['denoise'], params['flux_shift'], params['flux_guidance']
        sampler_mode, ays_schedule, model_type = params['sampler_mode'], params['ays_schedule'], params['model_type']

        # 2. Model Detection & Default Calibration
        detected_type, params, meta_unet_file = self._configure_model_and_defaults(model, model_type, preset,
            model_meta=model_meta, chain_loras=self._model_chain_loras(prompt, unique_id),
            cfg=cfg, flux_guidance=flux_guidance, flux_shift=flux_shift,
            sampler=sampler, scheduler=scheduler, scheduler_mode=scheduler_mode, steps=steps)

        cfg, flux_guidance, flux_shift = params['cfg'], params['flux_guidance'], params['flux_shift']
        sampler, scheduler, steps = params['sampler'], params['scheduler'], params['steps']

        # ALBABIT-FIX: SD-Turbo/SDXL-Turbo were only distilled at 10 fixed
        # discrete timesteps -- standard continuous schedulers pick sigma
        # values the model was never trained to be good at (see
        # get_sd_turbo_sigmas). Same "sdxl" + "turbo" filename detection
        # already used for the cfg/steps/sampler override above.
        use_sd_turbo_schedule = detected_type == "sdxl" and "turbo" in meta_unet_file.lower()

        # The seed default moved to 1 for this, but decrement, randomize or a
        # typed 0 still reach seed 0; say why the edit breaks when one does.
        if detected_type == "qwen_image21" and int(seed) == 0:
            logger.warning(
                "[RadianceSamplerPro] seed 0 on Qwen-Image 2.1: edits at this seed "
                "fail or come out oversaturated, with the native nodes too. Use any "
                "other seed."
            )

        # 3. Latent Preparation
        # The Prompt's latent output is empty without reference images; say
        # so instead of calling the missing latent an IMAGE.
        if latent_image is None:
            raise RuntimeError(
                "No latent reached latent_image.\n\n"
                "If it comes from the Prompt's latent output, connect a reference "
                "image to image_1: that output is empty without one. For text to "
                "image, connect an Empty Latent Image instead."
            )
        if not isinstance(latent_image, dict) or "samples" not in latent_image:
            _got = type(latent_image).__name__
            raise RuntimeError(
                "This node needs a LATENT, but the input is an IMAGE.\n\n"
                "Make sure you connect a VAE Encode or a Sampler node before this one."
            )
        t0 = time.time()
        latent_samples = latent_image["samples"]
        noise_mask = latent_image.get("noise_mask")  # Optional inpainting mask
        
        # Standardize to 5D for processing if it's a video model, otherwise keep 4D
        is_video = detected_type in VIDEO_MODEL_TYPES
        if is_video:
            # ALBABIT-FIX: packed AV latents (LTX-AV, MiniMax H3) are handled natively
            # below; ensure_5d now rejects anything that is not a plain Tensor.
            if not getattr(latent_samples, "is_nested", False):
                latent_samples = ensure_5d(latent_samples, "RadianceSamplerPro")
            frames = latent_samples.shape[2]
            logger.warning(
                "[RadianceSamplerPro] Video model '%s' detected. "
                "For native video-model parity, especially LTX/LTX 2.3, prefer "
                "RadianceVideoSampler. SamplerPro remains a universal/experimental "
                "sampler and may differ because guidance, noise, callbacks, and "
                "staged sampling are still Radiance-controlled.",
                detected_type,
            )
            if sigmas_override is not None:
                logger.warning(
                    "[RadianceSamplerPro] sigmas_override only replaces the sigma "
                    "schedule. It does not make SamplerPro identical to the native "
                    "video sampler path."
                )
        # ALBABIT-FIX: Qwen-Image samples in Wan 2.1's 5D format (B, C, 1, H, W).
        # Squeezed to 4D, its process_in broadcast the latent into 16 frames of
        # noise. The latent format decides, as in comfy.sample.fix_empty_latent_channels.
        elif model.get_model_object("latent_format").latent_dimensions == 3:
            latent_samples = ensure_5d(latent_samples, "RadianceSamplerPro")
            frames = None
        else:
            latent_samples = ensure_4d(latent_samples, "RadianceSamplerPro")
            frames = None
        # An image model sampling in 5D (Qwen-Image) still carries one frame:
        # the batch holds independent images, and the frame axis is the only
        # difference from 4D, so tiling applies to it as it does to 4D.
        tileable = tile_mode and (
            latent_samples.ndim == 4
            or (not is_video and latent_samples.ndim == 5 and latent_samples.shape[2] == 1)
        )
        timings["latent_prep"] = time.time() - t0

        device = comfy.model_management.get_torch_device()
        work_latent = latent_samples.to(device)

        # Blend SDR reference into initial latent if provided
        sdr_latent = None
        if sdr_reference is not None and sdr_vae is not None and sdr_blend > 0.0:
            sdr_latent = self._encode_sdr_reference(sdr_reference, sdr_vae, work_latent)
            work_latent = (1.0 - sdr_blend) * work_latent + sdr_blend * sdr_latent
            logger.info(f"[SDR Conditioning] Blended SDR reference into initial latent with strength {sdr_blend:.2f}")

        # 4. Noise & Sigmas Preparation
        t0 = time.time()
        noise = self._prepare_noise(latent_samples, seed, noise_type, noise_override, device, frames)

        # FEAT-NOISE-ALPHA: the blend itself is deferred until the step range is
        # resolved, because the ramp is a function of where on the trajectory the
        # noise is actually injected. See the _noise_alpha_at call further down.
        _noise_alpha_ramp_active = (
            abs(noise_alpha_start - 1.0) > 1e-4
            or abs(noise_alpha_end - 1.0) > 1e-4
            or abs(noise_alpha_start - noise_alpha_end) > 1e-4
        ) and noise_type != "Gaussian"

        timings["prepare_noise"] = time.time() - t0

        t0 = time.time()
        sigmas, target_total_steps = self._prepare_sigmas(
            model, detected_type, steps, denoise,
            flux_shift, scheduler, ays_schedule, terminal_sigma_to_zero,
            force_exact_steps, sigmas_override, device,
            custom_ays_anchors=custom_ays_anchors,
            use_sd_turbo_schedule=use_sd_turbo_schedule,
        )
        timings["prepare_sigmas"] = time.time() - t0

        # 5. LTX-AV Detection (Specific logic for audio/video separation)
        is_ltx_av = False
        ltxav_obj = None
        try:
            inner_model = model.get_model_object("diffusion_model")
            if hasattr(inner_model, "separate_audio_and_video_latents") and \
               hasattr(inner_model, "recombine_audio_and_video_latents"):
                is_ltx_av = True
                ltxav_obj = inner_model
                logger.info("[Radiance] LTX-AV detected. Latents will be processed natively.")
        except Exception as exc:
            logger.warning("[nodes_sampler]: %s", exc)

        # No gc.collect() / empty_cache() here: ComfyUI manages its own cache
        # between prompts, and a full collection of ComfyUI's heap plus a CUDA
        # cache flush right before sampling cost ~0.5 s per run (3.5 live log:
        # 0.53 s between "Auto-applied" and the first stage) and made the first
        # step re-allocate everything.
        log_tensor("Sigmas", sigmas)

        if len(sigmas) <= 1:
            logger.warning("[Radiance] Sigma schedule is trivial, returning input unchanged")
            report = build_sigma_report(
                detected_type,
                steps,
                scheduler,
                flux_shift,
                denoise,
                sigmas,
                sampler_mode,
                [],
                [],
                0.0,
                frames=frames,
            )
            blank_plot = torch.zeros(1, 64, 128, 3)
            return {
                "ui": self._resolved_values_ui(cfg, flux_guidance, flux_shift, sampler, steps),
                "result": (latent_image.copy(), sigmas, torch.tensor([0.0]), blank_plot),
            }
        log_tensor("Work Latent (Start)", work_latent)



        if conditioning_clip_target != "Auto":
            # DEFECT: this used to log "Conditioning routed to clip_g" after a
            # call that only wrote an `encoder_target` key nothing in ComfyUI or
            # Radiance reads. route_conditioning now warns and returns the
            # conditioning untouched, so this must not claim a routing happened.
            positive = route_conditioning(positive, conditioning_clip_target)

        primary_sampler = sampler
        secondary_sampler: Optional[str] = None

        secondary_scheduler: Optional[str] = None
        split_step = -1

        # DEFECT: this used to be a bare
        #     effective_end   = min(end_step or target_total_steps, target_total_steps)
        #     effective_start = min(start_step, effective_end)
        # with no validation at all, even though sampler_utils.validate_step_range
        # exists for exactly this and is imported at the top of this file. Any
        # range that collapsed to a single value -- start_step=10/end_step=10,
        # start_step=20/end_step=5, or start_step=50 with steps=20 and
        # end_step=0, all reachable because both widgets are min 0 / max 200 --
        # left `splits` with one element, so the stage loop below never ran and
        # the node returned the UN-SAMPLED INPUT LATENT while logging
        # "Sampling complete". Silent, and indistinguishable downstream from a
        # model that produced nothing.
        #
        # Resolve the "0 = use total steps" widget convention first, then clamp
        # and swap through the shared validator, then refuse an empty range
        # outright rather than pretending to have sampled it.
        effective_end = end_step if end_step > 0 else target_total_steps
        effective_start, effective_end = validate_step_range(
            start_step, effective_end, target_total_steps,
            context="[RadianceSamplerPro] ",
        )

        if effective_start >= effective_end:
            raise RuntimeError(
                f"Sampler: Empty step range: start_step={start_step}, "
                f"end_step={end_step} resolve to steps {effective_start}-{effective_end} "
                f"of {target_total_steps}, so there is nothing to sample.\n\n"
                f"end_step=0 means 'run to the end'. Set start_step below end_step "
                f"(or below `steps` when end_step is 0).\n\n"
                f"The node raises instead of returning the input latent unchanged, "
                f"which is what it used to do while logging 'Sampling complete'."
            )

        splits = {effective_start, effective_end}

        # FEAT-NOISE-ALPHA: blend the structured noise toward Gaussian using the
        # ramp value at the step where the noise is actually injected.
        if _noise_alpha_ramp_active:
            _alpha = self._noise_alpha_at(
                effective_start, target_total_steps, noise_alpha_start, noise_alpha_end
            )
            if _alpha < 1.0 - 1e-4:
                _gaussian = torch.randn_like(noise)
                if noise_override is None:
                    # We generated this tensor, so blend in place: the Gaussian
                    # draw is then the only extra full-size allocation and it is
                    # freed immediately.
                    noise = noise.lerp_(_gaussian, 1.0 - _alpha)
                else:
                    # noise_override hands us the caller's LATENT tensor, and on
                    # a CPU run `.to(device)` returns that same object. An
                    # in-place blend would mutate the upstream node's output and
                    # corrupt every other branch fed from it.
                    noise = torch.lerp(noise, _gaussian, 1.0 - _alpha)
                del _gaussian
            logger.info(
                "Noise alpha ramp: %.2f -> %.2f over %d steps, injecting at step %d "
                "-> alpha=%.3f (%s x %.2f + Gaussian x %.2f)",
                noise_alpha_start, noise_alpha_end, target_total_steps,
                effective_start, _alpha, noise_type, _alpha, 1.0 - _alpha,
            )
            if (
                abs(noise_alpha_start - noise_alpha_end) > 1e-4
                and effective_start == 0
            ):
                logger.warning(
                    "[Radiance] noise_alpha_end=%.2f has no effect on this run. The "
                    "noise tensor is mixed into the latent once, at step %d, where the "
                    "ramp still reads noise_alpha_start=%.2f. Raise start_step to "
                    "inject further down the trajectory, or leave the two equal.",
                    noise_alpha_end, effective_start, noise_alpha_start,
                )

        is_cfg_plus_plus = SamplerMode.is_cfg_plus_plus(sampler_mode)
        if is_cfg_plus_plus:
            logger.info(f"CFG++ (Perpendicular) mode active with base CFG {cfg}")

            if detected_type == "flux" and cfg <= 1.05:
                logger.warning(
                    f"CFG++ enabled but CFG is {cfg}. For Flux, CFG++ requires CFG > 1.0"
                )

        # Clone ONCE, before anything patches it.
        #
        # Each block below used to guard its own clone on a condition that
        # assumed an earlier block had already cloned -- and the first of those
        # only ran when `guidance_rescale_phi > 0 AND cfg > 1.0`. Flux defaults
        # cfg to 1.0, so a user following this node's own tooltip
        # ("recommended 0.7") patched the LOADER'S CACHED ModelPatcher in place.
        # The patch, its `_step_counter` and its captured `_sdr_ref_lat` closure
        # then persisted into every later queue and every other branch fed from
        # that MODEL, and survived the user disconnecting the input.
        model = model.clone()
        # Same reason as above: temporal windowing patches the refiner too, and
        # patching the loader's cached ModelPatcher in place would leak the
        # wrapper into every later queue fed from that MODEL.
        if refiner_model is not None:
            refiner_model = refiner_model.clone()

        # ── Long-video temporal windowing ────────────────────────────────────
        # Registered on the clone above, and on the refiner when one is present,
        # so both halves of a refiner chain window the same way. Off by default;
        # inert unless the latent is 5D and longer than the window, so no
        # existing graph changes.
        if temporal_window > 0:
            if not is_video or frames is None:
                logger.warning(
                    "[Radiance] temporal_window=%d is ignored: this run's latent is "
                    "not a 5D video latent (detected model type '%s'). Windowing has "
                    "no temporal axis to split.",
                    temporal_window, detected_type,
                )
            elif frames <= temporal_window:
                logger.info(
                    "[Radiance] temporal_window=%d covers the whole clip (%d latent "
                    "frames), so sampling runs unwindowed and is bit-identical to "
                    "temporal_window=0.",
                    temporal_window, frames,
                )
            else:
                logger.warning(
                    "[Radiance] Temporal windowing is experimental and can produce "
                    "ghosting or scene changes at joins. Set temporal_window=0 "
                    "for production work."
                )
                _win_plan = plan_temporal_windows(
                    frames, temporal_window, temporal_overlap
                )
                _eff_overlap = clamp_overlap(temporal_window, temporal_overlap)
                if _eff_overlap != temporal_overlap:
                    logger.warning(
                        "[Radiance] temporal_overlap=%d clamped to %d (half of "
                        "temporal_window=%d). A larger overlap collapses the stride "
                        "and would run one window per frame.",
                        temporal_overlap, _eff_overlap, temporal_window,
                    )
                logger.info(
                    "[Radiance] Temporal windowing active: %d latent frames -> %d "
                    "windows of %d (overlap %d). Every denoising step is evaluated "
                    "per window and the overlaps are blended before the next step, "
                    "so peak activation memory follows the window, not the clip. "
                    "Step count, schedule, seed and progress are unchanged.",
                    frames, len(_win_plan), temporal_window, _eff_overlap,
                )
                for _wm in (model, refiner_model):
                    if _wm is None:
                        continue
                    _prev = None
                    if hasattr(_wm, "model_options"):
                        _prev = _wm.model_options.get("model_function_wrapper")
                    _wm.set_model_unet_function_wrapper(
                        make_temporal_window_wrapper(
                            temporal_window, temporal_overlap, previous_wrapper=_prev
                        )
                    )

        if pag_scale > 0:
            # cfg goes in so apply_pag_to_model can say at registration that the
            # patch cannot fire at cfg <= 1.0, instead of logging success and
            # doing nothing. This node's own cfg default is 1.0.
            model = apply_pag_to_model(model, pag_scale, cfg=cfg)

        if guidance_rescale_phi > 0.0 and cfg <= 1.0:
            # DEFECT: this branch used to have no else and no warning. Guidance
            # rescale divides by the guided std and is only meaningful when CFG
            # actually pushed the result away from the conditional prediction;
            # at cfg <= 1.0 ComfyUI runs no uncond pass and `denoised` IS
            # `cond_denoised`, so the patch would be a no-op. That is the right
            # thing to do, but this sampler's cfg default is 1.0 and the
            # widget's own tooltip recommends 0.7, so the common case was a user
            # setting it and getting nothing with no indication why.
            logger.warning(
                "[Radiance] guidance_rescale_phi=%.2f is ignored at cfg=%.2f. "
                "Guidance rescale corrects the over-saturation CFG introduces, so it "
                "needs cfg > 1.0 to have anything to correct. Raise cfg, or set "
                "guidance_rescale_phi to 0 to silence this.",
                guidance_rescale_phi, cfg,
            )

        if guidance_rescale_phi > 0.0 and cfg > 1.0:
            phi = guidance_rescale_phi

            # ALBABIT-FIX: was on set_model_sampler_cfg_function (noise-space
            # single slot), but this math is denoised-space -- cfg_result
            # silently became x - denoised. Moved to the correct API,
            # set_model_sampler_post_cfg_function (comfy/samplers.py:600-603),
            # and now reads args["denoised"] directly instead of recomputing.
            def guidance_rescale_patch(args):
                guided = args["denoised"]
                cond = args["cond_denoised"]

                dims = list(range(1, guided.ndim))
                guided_std = guided.std(dim=dims, keepdim=True).clamp(min=1e-6)
                cond_std = cond.std(dim=dims, keepdim=True).clamp(min=1e-6)

                rescaled = guided * (cond_std / guided_std)
                return rescaled * phi + guided * (1.0 - phi)

            model.set_model_sampler_post_cfg_function(guidance_rescale_patch)
            logger.info(f"Guidance Rescale applied (phi={phi:.2f})")

        if sdr_latent is not None and sdr_inject_steps > 0:
            _step_counter = [0]
            _sdr_ref_lat = sdr_latent.detach()

            # ALBABIT-FIX: same wrong-API bug as guidance_rescale_patch above.
            # Moved to set_model_sampler_post_cfg_function, a list ComfyUI
            # chains automatically, so the manual existing_cfg_fn chaining
            # hack (compensating for the old single-slot clobbering) is gone.
            def _sdr_post_cfg_patch(args):
                denoised = args["denoised"]
                step = _step_counter[0]
                _step_counter[0] += 1
                if step >= sdr_inject_steps:
                    return denoised
                blend = sdr_blend * (sdr_decay ** step)
                if blend < 1e-5:
                    return denoised
                
                ref = _sdr_ref_lat
                if hasattr(ref, "to"):
                    ref = ref.to(device=denoised.device, dtype=denoised.dtype)
                
                ref_shape = tuple(ref.shape)
                denoised_shape = tuple(denoised.shape)
                if ref_shape != denoised_shape:
                    if ref.ndim == denoised.ndim:
                        if ref_shape[0] == 1 and denoised_shape[0] > 1:
                            ref = ref.expand_as(denoised)
                        else:
                            return denoised
                    else:
                        return denoised
                return (1.0 - blend) * denoised + blend * ref

            model.set_model_sampler_post_cfg_function(_sdr_post_cfg_patch)
            logger.info(f"[SDR Conditioning] Registered post-CFG SDR anchor patch (steps={sdr_inject_steps}, blend={sdr_blend:.2f}, decay={sdr_decay:.2f})")

        # ── Energy-Prioritized Sampling (EPS) Detection ──────────────────────
        # The producer is RadianceEnergyMask (nodes/generate/energy.py). Before
        # v3.2.1 there was no producer at all, so this branch was unreachable
        # from any graph (#40).
        energy_layers = _collect_energy_layers(positive)
        active_layers = [(m, p) for m, p in energy_layers if p != 0.0]

        if active_layers:
            # ALBABIT-FIX: a post-CFG function, like the two patches above: its result
            # is denoised-space, which the cfg function slot does not take.
            #
            # ALBABIT-FIX: same video/audio split source as the LTX-AV dual-CFG
            # patch below (work_latent.unbind()), so EPS can mask the video
            # stream of a packed multi-modality latent instead of no-op'ing.
            _energy_latent_shapes = None
            if getattr(work_latent, "is_nested", False):
                _energy_latent_shapes = [tuple(p.shape) for p in work_latent.unbind()]
            model.set_model_sampler_post_cfg_function(
                _make_energy_cfg_patch(active_layers, latent_shapes=_energy_latent_shapes)
            )
            logger.info(
                "[Energy Guidance] Registered Energy-Prioritized Sampling (EPS) patch "
                "(%d layer(s), priorities=%s)",
                len(active_layers), ", ".join(f"{p:.2f}" for _, p in active_layers),
            )
        elif energy_layers:
            logger.info(
                "[Energy Guidance] %d energy mask(s) attached, all at priority=0 — "
                "EPS is a no-op, not registering a patch.", len(energy_layers),
            )

        # ── LTX-AV Dual CFG (audio_cfg) ──────────────────────────────────────
        # ALBABIT-FIX: mirrors comfy's own LTXVDualCFGGuider (comfy_extras/
        # nodes_lt.py) -- separate CFG for the video/audio halves of a packed
        # LTX-AV latent. Noise-space slot, so it composes cleanly with the 3
        # post-cfg patches above (a different slot, applied after this one).
        if is_ltx_av and _HAS_NESTED_TENSOR and audio_cfg > 0.0 and not math.isclose(audio_cfg, cfg):
            _v_numel = None
            if getattr(work_latent, "is_nested", False):
                parts = work_latent.unbind()
                if len(parts) >= 2:
                    _v_numel = math.prod(parts[0].shape[1:])
            if _v_numel is None:
                logger.warning(
                    "[Radiance] audio_cfg=%.2f set but the latent isn't a nested "
                    "audio+video tensor -- ignoring (using cfg=%.2f for both).",
                    audio_cfg, cfg,
                )
            else:
                _video_cfg, _audio_cfg, _v = cfg, audio_cfg, _v_numel

                def _ltx_av_dual_cfg_patch(args):
                    cond, uncond = args["cond"], args["uncond"]
                    out = uncond + (cond - uncond) * _video_cfg
                    out[..., _v:] = uncond[..., _v:] + (cond[..., _v:] - uncond[..., _v:]) * _audio_cfg
                    return out

                model.set_model_sampler_cfg_function(_ltx_av_dual_cfg_patch, disable_cfg1_optimization=True)
                logger.info(
                    "[Radiance] LTX-AV dual CFG active (video_cfg=%.2f, audio_cfg=%.2f).",
                    _video_cfg, _audio_cfg,
                )

        if SamplerMode.is_phase_shift(sampler_mode) and detected_type in VIDEO_MODEL_TYPES:
            logger.warning(
                f"Phase-Shift mode is not supported for video model '{detected_type}' — "
                f"sampler switching mid-schedule causes temporal discontinuities. "
                f"Falling back to Standard mode."
            )
            sampler_mode = SamplerMode.STANDARD

        if SamplerMode.is_phase_shift(sampler_mode):

            if sigma_blend_steps == 0:
                sigma_blend_steps = 3
                logger.info("Phase-Shift: auto-enabled 3 sigma blend steps (was 0)")

            if sampler_mode == SamplerMode.PHASE_SHIFT_DPM:
                secondary_sampler = "dpmpp_2m"
                secondary_scheduler = None                       

                flow_match_types = GUIDANCE_EMBED_MODELS | {"wan", "wan_ti2v", "hunyuan_video"}
                if detected_type in flow_match_types:
                    logger.warning(
                        f"WARNING: DPM solvers are generally incompatible with {detected_type}'s "
                        f"Flow Matching and may produce severe noise/grain. "
                        f"Recommended to use Euler or Phase-Shift SGM."
                    )
            elif sampler_mode == SamplerMode.PHASE_SHIFT_SGM:

                secondary_sampler = sampler                     
                secondary_scheduler = "sgm_uniform"                              
            else:
                secondary_sampler = sampler

            split_step = int(target_total_steps * max(0.0, min(1.0, phase_split)))

            if effective_start < split_step < effective_end:
                splits.add(split_step)
                phase2_label = secondary_sampler or sampler
                if secondary_scheduler:
                    phase2_label = f"{phase2_label}+{secondary_scheduler}"
                logger.info(
                    f"Phase-Shift: {primary_sampler} (0-{split_step}) → "
                    f"{phase2_label} ({split_step}-{target_total_steps})"
                )

                if sigma_blend_steps > 0:
                    logger.info(f"Sigma blend: {sigma_blend_steps} steps at transition")

        if refiner_model is not None:
            refiner_step = max(0, min(refiner_start_step, target_total_steps))
            if effective_start < refiner_step < effective_end:
                splits.add(refiner_step)
                logger.info(f"Refiner starts at step {refiner_step}")

        is_dynamic = "Dynamic" in flux_guidance_profile and detected_type in GUIDANCE_EMBED_MODELS

        is_dynamic_cfg = "Dynamic" in flux_guidance_profile and detected_type in CFG_GUIDED_MODELS

        if is_dynamic:

            denoising_steps = (
                int(target_total_steps * denoise) if denoise < 1.0 else target_total_steps
            )
            denoising_start = target_total_steps - denoising_steps

            idx_20 = denoising_start + int(
                denoising_steps * DYNAMIC_GUIDANCE_EARLY_THRESHOLD
            )
            idx_90 = denoising_start + int(
                denoising_steps * DYNAMIC_GUIDANCE_LATE_THRESHOLD
            )

            if effective_start < idx_20 < effective_end:
                splits.add(idx_20)
            if effective_start < idx_90 < effective_end:
                splits.add(idx_90)
            logger.info(
                f"Dynamic Guidance Active (effective range: steps {denoising_start}-{target_total_steps}, "
                f"early={idx_20}, late={idx_90})"
            )

        elif is_dynamic_cfg:

            denoising_steps = (
                int(target_total_steps * denoise) if denoise < 1.0 else target_total_steps
            )
            denoising_start = target_total_steps - denoising_steps

            idx_15 = denoising_start + int(denoising_steps * DYNAMIC_CFG_EARLY_THRESHOLD)
            idx_85 = denoising_start + int(denoising_steps * DYNAMIC_CFG_LATE_THRESHOLD)

            if effective_start < idx_15 < effective_end:
                splits.add(idx_15)
            if effective_start < idx_85 < effective_end:
                splits.add(idx_85)
            logger.info(
                f"Dynamic CFG Active for {detected_type} (effective range: steps "
                f"{denoising_start}-{target_total_steps}, boost→{idx_15}, taper→{idx_85})"
            )

        if is_dynamic or is_dynamic_cfg:
            # Threshold-only splits sample the midpoint of each cosine ramp
            # and hold it for the entire next stage, never reaching its target.
            thresholds = (DYNAMIC_GUIDANCE_EARLY_THRESHOLD, DYNAMIC_GUIDANCE_LATE_THRESHOLD) if is_dynamic else (
                DYNAMIC_CFG_EARLY_THRESHOLD, DYNAMIC_CFG_LATE_THRESHOLD)
            for threshold in thresholds:
                ramp_end = denoising_start + math.ceil(denoising_steps * (threshold + DYNAMIC_GUIDANCE_RAMP_WIDTH))
                if effective_start < ramp_end < effective_end:
                    splits.add(ramp_end)

        sorted_splits = sorted(
            s for s in splits if effective_start <= s <= effective_end
        )

        # ALBABIT-FIX: the node's progress bar on every run, one step per
        # iteration of every stage. It only existed with a live preview:
        # sample_custom draws its bar in the console, never on the node.
        pbar_ref = comfy.utils.ProgressBar(len(sigmas) - 1)
        previewer_ref = None
        use_custom_preview = False
        if preview_method != "None":
            previewer_ref = _resolve_latent_previewer(model, preview_method)

            if previewer_ref is not None:
                use_custom_preview = True
                logger.debug(f"Preview callback active: {preview_method}")
            else:
                # DEFECT GUARD: disable_pbar was wired to a flag that went True
                # whenever the widget was not "None", so a previewer that could
                # not be built left the run with no Radiance preview AND
                # ComfyUI's console progress bar switched off. use_custom_preview
                # stays False, so the console bar stays on.
                logger.warning(
                    "[Radiance] preview_method='%s' produced no previewer; "
                    "sampling runs without a preview.",
                    preview_method,
                )

        def create_phase_callback(phase_start_step):
            return _make_phase_preview_callback(
                pbar_ref, previewer_ref, phase_start_step
            )

        current_latent = work_latent
        prev_stage_sigmas: Optional[torch.Tensor] = None
        _ltxav_mr_base = None
        _ltxav_mr_orig = None
        # MEMORY: one shared all-zero noise buffer for every stage that does not
        # inject noise, allocated on first use. See the stage loop.
        zero_noise: Optional[torch.Tensor] = None

        # RADIANCE-AUDIT v2.6.0 [IMPORTANT]: removed dead pre-seed cache
        # block. It wrote under (primary_scheduler, target_total_steps)
        # but the cache is only consulted for *secondary* schedulers via
        # compute_base_sigmas, which uses a different calc_steps key when
        # force_exact_steps is on. The write never produced a hit.

        def _get_base_sigmas(mdl, sched: str = scheduler) -> torch.Tensor:
            if sched == scheduler and mdl is model:
                return sigmas
            return compute_base_sigmas(
                mdl, sched, target_total_steps, scheduler, flux_shift, denoise, _sigma_cache,
                force_full_denoise=terminal_sigma_to_zero,
                force_exact_steps=force_exact_steps
            )

        t0 = time.time()

        stage_timings: List[Tuple[int, int, int, str, float]] = []

        planned_stages: List[SamplingStage] = []
        for si in range(len(sorted_splits) - 1):
            gs = sorted_splits[si]
            ge = sorted_splits[si + 1]
            if gs >= ge:
                continue

            stage_model = model
            if refiner_model is not None and gs >= refiner_start_step:
                stage_model = refiner_model

            stage_sampler = primary_sampler
            stage_scheduler = scheduler
            is_shifted = False
            is_blend = False

            if SamplerMode.is_phase_shift(sampler_mode) and gs >= split_step:
                is_shifted = True
                if secondary_sampler:
                    stage_sampler = secondary_sampler
                if secondary_scheduler:
                    stage_scheduler = secondary_scheduler
                if gs == split_step and sigma_blend_steps > 0:
                    is_blend = True

            planned_stages.append(
                SamplingStage(
                    index=si,
                    global_start=gs,
                    global_end=ge,
                    model=stage_model,
                    sampler_name=stage_sampler,
                    scheduler_name=stage_scheduler,
                    is_phase_shifted=is_shifted,
                    is_blend_point=is_blend,
                )
            )

        for ps in planned_stages:
            label = f"{ps.sampler_name}+{ps.scheduler_name}"
            if ps.is_blend_point:
                label += " [blend]"
            
            # ALBABIT-FIX: Display logical 1-based steps for clarity in logs instead of 0-based
            logger.info(
                f"Plan Stage {ps.index + 1}: Steps {ps.global_start + 1}→{ps.global_end} [{label}]"
            )

        try:
            if tileable:
                # DEFECT: tile_mode called tile_sample once with the FULL sigma
                # schedule and the primary sampler, then `break`-ed out of the
                # stage loop. Six settings were discarded in silence:
                # phase-shift, refiner_model, dynamic guidance, start_step,
                # end_step and add_noise -- and flux_guidance never reached the
                # conditioning at all, because the only apply_flux_guidance call
                # lives inside the loop that was skipped. The one INFO log
                # mentioned none of it.
                #
                # Tiling runs as a single stage by construction (each tile is
                # denoised end to end independently), so the settings that need
                # several stages genuinely cannot apply. Everything that can be
                # honoured now is; everything that cannot is named in a WARNING
                # rather than dropped.
                _tile_unsupported = []
                if SamplerMode.is_phase_shift(sampler_mode):
                    _tile_unsupported.append(
                        f"sampler_mode={sampler_mode} (needs a mid-schedule sampler "
                        f"switch; tiling denoises each tile in one pass)"
                    )
                if refiner_model is not None:
                    _tile_unsupported.append(
                        f"refiner_model (would need to swap models at step "
                        f"{refiner_start_step}, same reason)"
                    )
                if is_dynamic:
                    _tile_unsupported.append(
                        "flux_guidance_profile='Dynamic' (per-stage guidance ramp; "
                        "the static flux_guidance value is used instead)"
                    )
                if is_dynamic_cfg:
                    _tile_unsupported.append(
                        "flux_guidance_profile='Dynamic' CFG ramp (the static cfg "
                        "value is used instead)"
                    )
                if _tile_unsupported:
                    logger.warning(
                        "[Radiance] tile_mode=True ignores %d setting(s), because "
                        "tiled sampling is single-stage: %s. Turn tile_mode off to "
                        "use them.",
                        len(_tile_unsupported), "; ".join(_tile_unsupported),
                    )

                # start_step / end_step: slice the schedule the same way a stage
                # would, instead of always running the whole thing.
                _tile_indexer = SigmaIndexer(target_total_steps, sigmas)
                _tile_sigmas = _tile_indexer.get_stage_sigmas(
                    effective_start, effective_end
                )
                if _tile_sigmas is None or len(_tile_sigmas) < 2:
                    _tile_sigmas = sigmas

                # add_noise: honoured, not ignored. The sampler adds
                # noise_scaling(sigma0, noise, latent), so zeros means "start
                # from the latent as given", which is what add_noise=False asks
                # for on an img2img-style pass.
                _tile_noise = noise if add_noise else torch.zeros_like(noise)

                # flux_guidance: applied here too. Guidance-embedded models
                # (Flux, LTXV) read it off the conditioning, so skipping this
                # ran them at whatever guidance the encode happened to carry.
                _tile_positive = positive
                if detected_type in GUIDANCE_EMBED_MODELS:
                    _tile_positive = apply_flux_guidance(positive, flux_guidance)

                logger.info(
                    "Tile sampling: size=%d, overlap=%d, blend=%s | "
                    "steps %d-%d of %d | sampler=%s | add_noise=%s",
                    tile_size, tile_overlap, tile_blend,
                    effective_start + 1, effective_end, target_total_steps,
                    primary_sampler, add_noise,
                )
                t_tile = time.time()
                _tile_sampler_obj = comfy.samplers.sampler_object(primary_sampler)
                effective_cfg = cfg
                if is_cfg_plus_plus:
                    sigma_max = sigmas[0].item() if len(sigmas) > 0 else 1.0
                    effective_cfg = apply_cfg_plus_plus(cfg, _tile_sigmas[0], sigma_max)

                current_latent = tile_sample(
                    model=model,
                    noise=_tile_noise,
                    latent_samples=work_latent,
                    positive=_tile_positive,
                    negative=negative,
                    sigmas=_tile_sigmas,
                    sampler_obj=_tile_sampler_obj,
                    seed=seed,
                    tile_size=tile_size,
                    tile_overlap=tile_overlap,
                    tile_blend=tile_blend,
                    noise_mask=noise_mask,
                    cfg=effective_cfg,
                )

                timings["tile_sampling"] = time.time() - t_tile
                timings["sampling"] = timings["tile_sampling"]
                logger.info(
                    f"Tile sampling done in {timings['tile_sampling']:.2f}s"
                )

            # ALBABIT-FIX: KSampler packs the LTX-AV NestedTensor to (B, 1, flat_N), so
            # memory_required counted 128x the real area and split the conditioning into
            # 2 forward passes per step (2.5x slower at HighRes). Correct it on the model.
            if is_ltx_av and hasattr(model, "model"):
                _ltxav_mr_base = model.model
                _ltxav_mr_orig = _ltxav_mr_base.memory_required
                def _ltxav_memory_required(input_shape, cond_shapes={}):
                    if len(input_shape) == 3 and input_shape[1] == 1 and input_shape[2] > 500_000:
                        return _ltxav_mr_orig([input_shape[0], 1, input_shape[2] // 128], cond_shapes)
                    return _ltxav_mr_orig(input_shape, cond_shapes)
                _ltxav_mr_base.memory_required = _ltxav_memory_required

            for plan_idx, stage in enumerate(planned_stages):
                if tileable:
                    break
                t_stage = time.time()
                i = stage.index
                s_start = stage.global_start
                s_end = stage.global_end
                current_model = stage.model
                current_sampler = stage.sampler_name
                current_scheduler = stage.scheduler_name

                stage_positive = positive
                if is_dynamic:
                    effective_guidance = compute_dynamic_guidance(
                        flux_guidance, s_start, target_total_steps, denoise
                    )
                    stage_positive = apply_flux_guidance(positive, effective_guidance)
                    logger.debug(
                        f"Dynamic Guidance @ step {s_start}: {effective_guidance:.2f}"
                    )

                elif detected_type in GUIDANCE_EMBED_MODELS:

                    stage_positive = apply_flux_guidance(positive, flux_guidance)

                logger.info(
                    f"Stage {i+1}: Steps {s_start + 1}-{s_end} | "
                    f"Sampler: {current_sampler} | Scheduler: {current_scheduler}"
                )

                try:

                    base_sigmas = _get_base_sigmas(current_model, current_scheduler)

                    indexer = SigmaIndexer(target_total_steps, base_sigmas)
                    stage_sigmas = indexer.get_stage_sigmas(s_start, s_end)

                    if stage_sigmas is None:

                        continue

                    if (
                        stage.is_blend_point
                        and prev_stage_sigmas is not None
                    ):
                        stage_sigmas = gradual_sigma_blend(
                            prev_stage_sigmas, stage_sigmas, sigma_blend_steps
                        )

                    if len(stage_sigmas) < 2:
                        logger.warning(
                            f"Stage {i+1}: Insufficient sigmas ({len(stage_sigmas)}), skipping"
                        )
                        continue

                    is_last_stage = plan_idx == len(planned_stages) - 1
                    if (
                        return_with_leftover_noise
                        and is_last_stage
                        and len(stage_sigmas) >= 3                                   
                        and stage_sigmas[-1].item() == 0.0
                    ):
                        stage_sigmas = stage_sigmas[:-1]
                        logger.debug("Stripped terminal sigma for leftover noise")

                    sampler_obj = comfy.samplers.sampler_object(current_sampler)

                    if is_cfg_plus_plus and len(stage_sigmas) > 0:
                        sigma_max = (
                            base_sigmas[0].item() if len(base_sigmas) > 0 else 1.0
                        )
                        effective_cfg = apply_cfg_plus_plus(
                            cfg, stage_sigmas[0], sigma_max
                        )
                        logger.debug(
                            f"CFG++: {cfg:.2f} → {effective_cfg:.2f} at sigma {stage_sigmas[0].item():.4f}"
                        )
                    elif is_dynamic_cfg:

                        effective_cfg = compute_dynamic_cfg(
                            cfg, s_start, target_total_steps, denoise
                        )
                        logger.debug(
                            f"Dynamic CFG @ step {s_start}: {cfg:.2f} → {effective_cfg:.2f}"
                        )
                    else:
                        effective_cfg = cfg

                    stage_latent = current_latent

                    is_first_stage = plan_idx == 0
                    if is_first_stage and add_noise:
                        stage_noise = noise
                        logger.debug("Stage 1: sampler will add initial noise")
                    else:
                        # MEMORY: this was `torch.zeros_like(noise)` every time
                        # round, so an N-stage run allocated N full copies of the
                        # sequence and let the old ones sit until the GC got to
                        # them. sample_custom only reads the noise tensor, so one
                        # zero buffer serves every stage that needs one.
                        if zero_noise is None:
                            zero_noise = torch.zeros_like(noise)
                        stage_noise = zero_noise
                        if is_first_stage:
                            logger.debug("Stage 1: add_noise=False, no noise")
                        else:
                            logger.debug(f"Stage {i+1}: continuation, no extra noise")

                    # RADIANCE-AUDIT v2.6.0 [IMPORTANT]: this block used to
                    # require noise_type != "gaussian", so a Gaussian
                    # NestedTensor multi-stage path fed a plain Tensor as
                    # stage_noise against a NestedTensor latent and
                    # crashed inside sample_custom. Rebuild whenever the
                    # latent is a NestedTensor, regardless of noise_type.
                    if (
                        _HAS_NESTED_TENSOR
                        and isinstance(current_latent, _NestedTensor)
                    ):
                        try:
                            new_noises = []
                            for idx, sub_t in enumerate(current_latent.tensors):
                                if is_first_stage and add_noise:
                                    # ALBABIT-FIX: use pre-seeded noise sub-tensor from
                                    # _prepare_noise instead of torch.randn_like, which
                                    # ignores the seed and produces non-deterministic noise
                                    # mismatched from native RandomNoise — root cause of
                                    # black bars in LTX-AV NestedTensor inference.
                                    if (
                                        isinstance(noise, _NestedTensor)
                                        and idx < len(noise.tensors)
                                    ):
                                        sub_noise = noise.tensors[idx]
                                    elif noise_type.lower() == "gaussian":
                                        sub_noise = torch.randn_like(sub_t)
                                    else:
                                        sub_noise = generate_noise(
                                            sub_t, seed, noise_type, frames=frames
                                        )
                                else:
                                    sub_noise = torch.zeros_like(sub_t)
                                new_noises.append(sub_noise)
                            stage_noise = _NestedTensor(tuple(new_noises))
                        except Exception as _ne:
                            logger.warning(
                                f"[Radiance] LTX-AV NestedTensor noise rebuild failed: {_ne}. "
                            )

                    # No explicit load_model_gpu() here. sample_custom loads the
                    # model itself with the inference memory it needs reserved;
                    # the extra call loaded it first with no reservation, so
                    # ComfyUI prepared it twice per stage (seen in the 3.5 live
                    # log: "prepared for dynamic VRAM loading" x2) and, outside
                    # dynamic VRAM, filled the card and then had to evict weights
                    # to make room for activations.
                    result = _sample_custom_progress_safe(
                        f"RadianceSamplerPro stage {i+1}",
                        model=current_model,
                        noise=stage_noise,
                        cfg=effective_cfg,
                        sampler=sampler_obj,
                        sigmas=stage_sigmas,
                        positive=stage_positive,
                        negative=negative,
                        latent_image=stage_latent,
                        noise_mask=noise_mask,
                        callback=create_phase_callback(s_start),
                        disable_pbar=use_custom_preview,
                        seed=seed,
                    )
                    current_latent = result

                    prev_stage_sigmas = indexer.get_stage_sigmas(s_start, s_end)
                    log_tensor(f"Stage {i+1} Output", current_latent)

                    if plan_idx < len(planned_stages) - 1:
                        next_stage = planned_stages[plan_idx + 1]
                        next_base = _get_base_sigmas(
                            next_stage.model, next_stage.scheduler_name
                        )
                        next_indexer = SigmaIndexer(target_total_steps, next_base)
                        next_sigma = next_indexer.get_sigma_at(next_stage.global_start)

                        if (
                            next_sigma is not None
                            and len(stage_sigmas) > 0
                        ):
                            expected_sigma = stage_sigmas[-1].item()
                            sigma_diff = abs(expected_sigma - next_sigma)
                            if sigma_diff > SIGMA_DISCONTINUITY_THRESHOLD:
                                logger.warning(
                                    f"Sigma discontinuity: {expected_sigma:.4f} → {next_sigma:.4f} "
                                    f"(Δ={sigma_diff:.4f})"
                                )

                    stage_time = time.time() - t_stage
                    stage_timings.append(
                        (i + 1, s_start, s_end, current_sampler, stage_time)
                    )

                except (RuntimeError, ValueError) as e:
                    logger.error(f"Error in Stage {i+1}: {e}")
                    raise

            if hasattr(current_latent, "is_cuda"):
                samples = current_latent.cpu() if current_latent.is_cuda else current_latent
            else:
                samples = current_latent

            # MEMORY: `.cpu()` above materialises a second full copy of the
            # sequence. The device copy is dead from here on, but nothing
            # dropped the references, so both sat in memory through the rest of
            # the function and the `finally` block only ever freed `work_latent`
            # and `noise`. Release the device-side copies as soon as the CPU
            # one exists, which is the peak moment of the whole run.
            if samples is not current_latent:
                current_latent = None
                stage_latent = None
                work_latent = None
                zero_noise = None
                noise = None

            if not tileable:
                timings["sampling"] = time.time() - t0

            if tile_mode and not tileable:
                logger.warning(
                    "[Radiance] Tile sampling ignored for 5D video latents."
                )

        finally:

            if _ltxav_mr_base is not None and _ltxav_mr_orig is not None:
                _ltxav_mr_base.memory_required = _ltxav_mr_orig

            if "noise" in locals():
                try: del noise
                except UnboundLocalError: pass
            if "work_latent" in locals():
                try: del work_latent
                except UnboundLocalError: pass

            if torch.cuda.is_available():
                torch.cuda.empty_cache()


        if is_ltx_av and ltxav_obj is not None and _HAS_NESTED_TENSOR:
            try:
                final_t = samples["samples"] if isinstance(samples, dict) else samples

                if isinstance(final_t, _NestedTensor):
                    final_t_raw = final_t.tensors
                else:
                    final_t_raw = final_t

                # ALBABIT-FIX: Removed internal normalization multipliers entirely. LTX natively recombines.
                v_s, a_s = ltxav_obj.separate_audio_and_video_latents(final_t_raw, None)
                v_s = v_s.detach()
                a_s = a_s.detach()

                recombined = _NestedTensor(
                    ltxav_obj.recombine_audio_and_video_latents(v_s, a_s)
                )
                if isinstance(samples, dict):
                    samples["samples"] = recombined
                else:
                    samples = recombined

                del v_s, a_s, final_t_raw
                logger.info("[Radiance] LTX-AV recombination applied (native scaling).")
            except Exception as _av_err:
                logger.warning(
                    f"[Radiance] LTX-AV recombination failed: {_av_err}. "
                )

        t0 = time.time()
        out = latent_image.copy()
        out["samples"] = samples
        timings["output_prep"] = time.time() - t0

        total_time = time.time() - t_start

        logger.info(
            f"Sampling complete: {target_total_steps} total target steps, {total_time:.2f}s total, "
            f"{timings['sampling']:.2f}s sampling"
        )
        for stage_num, s_start_t, s_end_t, samp, t in stage_timings:
            logger.info(
                f"  Stage {stage_num}: steps {s_start_t + 1}→{s_end_t} [{samp}] = {t:.3f}s"
            )

        # ── Restart sampling (IRES style) ─────────────────────────────────────
        if restart_count > 0 and restart_schedule is not None and len(restart_schedule) > 0:
            try:
                _rs_sampler = comfy.samplers.sampler_object(primary_sampler)
                raw_samples = out["samples"].to(device)
                raw_samples = self._apply_restarts(
                    raw_samples, model, positive, negative,
                    _rs_sampler, sigmas, restart_schedule,
                    restart_count, cfg, seed, device,
                )
                out["samples"] = raw_samples.cpu()
                logger.info(
                    f"Restart sampling applied: {restart_count} restart(s) at "
                    f"{len(restart_schedule)} sigma level(s)"
                )
            except Exception as _rs_err:
                logger.warning(f"Restart sampling failed: {_rs_err} — using unmodified output")

        output_sigmas = (
            sigmas if sigmas is not None and len(sigmas) > 0 else torch.tensor([0.0])
        )

        # ── sigmas_remaining: the unused schedule tail after effective_end ───
        _eff_end_idx = min(effective_end, len(output_sigmas))
        sigmas_remaining = output_sigmas[_eff_end_idx:] if _eff_end_idx < len(output_sigmas) else torch.tensor([0.0])

        # ── Non-finite guard ───────────────────────────────────────────────────
        assert_latent_finite(out.get("samples") if isinstance(out, dict) else None)

        # ── sigma_plot IMAGE ──────────────────────────────────────────────────
        sigma_plot = self._build_sigma_plot(output_sigmas, detected_type)

        return {
            "ui": self._resolved_values_ui(cfg, flux_guidance, flux_shift, sampler, steps),
            "result": (out, output_sigmas, sigmas_remaining, sigma_plot),
        }

NODE_CLASS_MAPPINGS = {
    "RadianceSamplerPro": RadianceSamplerPro,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceSamplerPro": "◎ Radiance Sampler",
}
