"""Tensor-numpy conversion helpers with type safety."""
import numpy as np
import torch


def tensor_to_numpy_float32(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(np.float32)


def numpy_to_tensor_float32(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array.astype(np.float32))


def tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    arr = t.detach().cpu().float().numpy()
    return arr[0] if arr.ndim == 4 else arr


def numpy_to_tensor(arr: np.ndarray) -> torch.Tensor:
    if arr.ndim == 3:
        arr = arr[np.newaxis]
    return torch.from_numpy(arr.astype(np.float32))
