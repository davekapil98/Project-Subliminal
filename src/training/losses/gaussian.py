"""Diagonal Gaussian negative log likelihood for body uncertainty."""

import math

import torch
from torch import Tensor


def gaussian_nll(mean: Tensor, log_variance: Tensor, target: Tensor) -> Tensor:
    log_variance = log_variance.clamp(-8.0, 5.0)
    squared = (target - mean).square() * torch.exp(-log_variance)
    return 0.5 * (squared + log_variance + math.log(2.0 * math.pi)).mean()
