"""Dimension contract helpers — safe 4D/5D tensor reshaping for image/video."""
import torch
import logging

logger = logging.getLogger("radiance.core.tensor.contract")


def _reject_non_tensor(tensor, context_label: str, want: str):
    """Raise a readable TypeError for anything that is not a plain torch.Tensor.

    DEFECT: these helpers only ever touched `.ndim`, then called `.unsqueeze`
    or `.permute`. A comfy.nested_tensor.NestedTensor (LTX-AV, MiniMax H3)
    exposes `.ndim` and `.shape` but neither of those methods, so it sailed
    past the rank checks and died on a bare
    `AttributeError: 'NestedTensor' object has no attribute 'unsqueeze'`
    with no mention of which node or which input was at fault. Say what
    arrived and what to do about it instead.
    """
    if isinstance(tensor, torch.Tensor):
        return
    raise TypeError(
        f"[{context_label}] Expected a torch.Tensor to reshape to {want}, got "
        f"{type(tensor).__name__}. Packed multi-modality latents "
        f"(NestedTensor, e.g. LTX-AV or MiniMax H3) cannot be reshaped by rank: "
        f"unpack the video stream first, or route this latent to a node that "
        f"handles the packed form natively."
    )


def ensure_5d(tensor: torch.Tensor, context_label: str = "Unknown") -> torch.Tensor:
    if getattr(tensor, "ndim", None) == 5 and isinstance(tensor, torch.Tensor):
        return tensor
    _reject_non_tensor(tensor, context_label, "5D (B, C, T, H, W)")
    if tensor.ndim == 4:
        logger.debug("[%s] Auto-reshaping 4D -> 5D (B, C, 1, H, W)", context_label)
        return tensor.unsqueeze(2)
    if tensor.ndim == 5:
        return tensor
    raise ValueError(
        f"[{context_label}] Expected 4D or 5D tensor, got {tensor.ndim}D (shape={tensor.shape})"
    )


def ensure_4d(tensor: torch.Tensor, context_label: str = "Unknown") -> torch.Tensor:
    if getattr(tensor, "ndim", None) == 4 and isinstance(tensor, torch.Tensor):
        return tensor
    _reject_non_tensor(tensor, context_label, "4D (B, C, H, W)")
    if tensor.ndim == 5:
        B, C, F, H, W = tensor.shape
        if F == 1:
            logger.debug("[%s] Auto-reshaping 5D(F=1) -> 4D (B, C, H, W)", context_label)
            return tensor.squeeze(2)
        # DEFECT: this flatten was logged at debug, which ComfyUI's default
        # INFO level does not print. Folding T into the batch axis destroys
        # every temporal relationship the latent carries: a WAN
        # (1, 16, 21, 60, 104) latent reaching a node whose model detection
        # landed on an image type silently became 21 unrelated images, and the
        # only trace was a log line nobody saw. Rank changes that discard the
        # temporal axis are a WARNING, so the operator sees them in the
        # console the run happens in.
        logger.warning(
            "[%s] Latent rank change: 5D (B=%d, C=%d, T=%d, H=%d, W=%d) flattened to "
            "4D (B*T=%d, C=%d, H=%d, W=%d). The %d temporal frames are now independent "
            "batch items and all temporal coherence is lost. If this is a video latent, "
            "the receiving node's model_type is set to an image architecture.",
            context_label, B, C, F, H, W, B * F, C, H, W, F,
        )
        return tensor.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
    if tensor.ndim == 4:
        return tensor
    raise ValueError(
        f"[{context_label}] Expected 4D or 5D tensor, got {tensor.ndim}D (shape={tensor.shape})"
    )
