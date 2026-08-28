"""Latent target matching without pixel reconstruction."""

import torch.nn.functional as functional
from torch import Tensor


def jepa_latent_loss(predicted: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
    error = functional.smooth_l1_loss(predicted, target.detach(), reduction="none").mean(-1)
    if mask is None:
        return error.mean()
    weights = mask.to(error.dtype)
    return (error * weights).sum() / weights.sum().clamp_min(1.0)
