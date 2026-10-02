"""RGB / alpha separation for IMAGE tensors (B, H, W, C).

Colour and exposure operations work on RGB. Alpha is coverage, not colour: a
grade, an exposure push, a normalisation or a colour-space transform must not
touch it unless the operation targets alpha on purpose.

    rgb, alpha = split_rgb_alpha(image)
    rgb = grade(rgb)
    return merge_rgb_alpha(rgb, alpha)
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch


def split_rgb_alpha(image: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Return ``(rgb, alpha)``. ``alpha`` is None when the image has no 4th channel.

    One-channel images are returned unchanged with no alpha; channels beyond
    the fourth are not supported and raise.
    """
    c = image.shape[-1]
    if c <= 3:
        return image, None
    if c == 4:
        return image[..., :3], image[..., 3:4]
    raise ValueError(
        f"Expected an IMAGE with 1, 3 or 4 channels, got {c}. "
        "Split extra channels before this node."
    )


def merge_rgb_alpha(rgb: torch.Tensor, alpha: Optional[torch.Tensor]) -> torch.Tensor:
    """Re-attach an untouched alpha to a processed RGB tensor.

    The alpha is moved to the RGB tensor's device and dtype; if the RGB was
    resized, the alpha must already match its spatial size.
    """
    if alpha is None:
        return rgb
    if alpha.shape[:-1] != rgb.shape[:-1]:
        raise ValueError(
            f"Alpha {tuple(alpha.shape)} does not match RGB {tuple(rgb.shape)}."
        )
    return torch.cat([rgb, alpha.to(device=rgb.device, dtype=rgb.dtype)], dim=-1)


def rgb_only(fn: Callable[[torch.Tensor], torch.Tensor], image: torch.Tensor) -> torch.Tensor:
    """Apply ``fn`` to the RGB channels of ``image`` and pass alpha through."""
    rgb, alpha = split_rgb_alpha(image)
    return merge_rgb_alpha(fn(rgb), alpha)


__all__ = ["split_rgb_alpha", "merge_rgb_alpha", "rgb_only"]
