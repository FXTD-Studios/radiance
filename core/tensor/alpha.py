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


def alpha_passthrough(image_arg: str = "image", outputs=(0,), strip=()):
    """Class decorator for colour nodes: run FUNCTION on RGB, keep alpha.

    The IMAGE argument ``image_arg`` is split; the node computes on RGB only
    and its IMAGE results at the ``outputs`` indices get the untouched alpha
    back (when their batch and size still match the input). A node that
    already returns 4 channels is left as it is. IMAGE arguments named in
    ``strip`` (references, mattes) are reduced to RGB as well.
    """
    import functools

    def wrap(cls):
        fname = cls.FUNCTION
        inner = getattr(cls, fname)

        import inspect
        params = [p for p in inspect.signature(inner).parameters if p != "self"]
        pos = params.index(image_arg) if image_arg in params else None

        @functools.wraps(inner)
        def run(self, *args, **kwargs):
            args = list(args)
            if image_arg in kwargs:
                img = kwargs[image_arg]
            elif pos is not None and pos < len(args):
                img = args[pos]
            else:
                img = None
            alpha = None
            if isinstance(img, torch.Tensor) and img.dim() == 4 and img.shape[-1] == 4:
                rgb, alpha = img[..., :3], img[..., 3:4]
                if image_arg in kwargs:
                    kwargs = dict(kwargs)
                    kwargs[image_arg] = rgb
                else:
                    args[pos] = rgb
            for name in strip:
                if name in kwargs:
                    t = kwargs[name]
                    if isinstance(t, torch.Tensor) and t.dim() == 4 and t.shape[-1] == 4:
                        kwargs = dict(kwargs)
                        kwargs[name] = t[..., :3]
                elif name in params and params.index(name) < len(args):
                    k = params.index(name)
                    t = args[k]
                    if isinstance(t, torch.Tensor) and t.dim() == 4 and t.shape[-1] == 4:
                        args[k] = t[..., :3]
            out = inner(self, *args, **kwargs)
            if alpha is None:
                return out
            ui = None
            res = out
            if isinstance(out, dict):
                if "result" not in out:
                    return out   # ui-only result, nothing to re-attach
                ui, res = out, out["result"]
            res = list(res)
            for i in outputs:
                t = res[i] if i < len(res) else None
                if (isinstance(t, torch.Tensor) and t.dim() == 4 and t.shape[-1] == 3
                        and t.shape[:3] == alpha.shape[:3]):
                    res[i] = merge_rgb_alpha(t, alpha)
            res = tuple(res)
            if ui is not None:
                ui = dict(ui)
                ui["result"] = res
                return ui
            return res

        run.__radiance_alpha_passthrough__ = True
        setattr(cls, fname, run)
        return cls

    return wrap


__all__.append("alpha_passthrough")
