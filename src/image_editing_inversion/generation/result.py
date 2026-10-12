"""Reconstructed and edited RGB pixel tensors for one sample."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class EditResult:
    """CPU uint8 RGB tensors in contiguous [3, height, width] layout."""

    reconstructed: torch.Tensor
    edited: torch.Tensor
