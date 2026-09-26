"""Luma/chroma utility functions — single-source for BT.709 / Rec.2020 luma."""
from __future__ import annotations

import numpy as np
import torch


def luma_bt709(arr: np.ndarray) -> np.ndarray:
    return (0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2])


def luma_bt709_tensor(t: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
    w = t.new_tensor([0.2126, 0.7152, 0.0722])
    return (t[..., :3] * w).sum(dim=-1, keepdim=keepdim)


def luma_rec2020_tensor(t: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
    w = t.new_tensor([0.2627, 0.6780, 0.0593])
    return (t[..., :3] * w).sum(dim=-1, keepdim=keepdim)
