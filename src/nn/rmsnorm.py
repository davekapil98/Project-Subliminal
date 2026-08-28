"""Readable RMSNorm implementation used by all Stage 0 Transformers."""

import torch
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """Normalize each token by its root-mean-square magnitude."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, inputs: Tensor) -> Tensor:
        # The reduction is evaluated in FP32 for stable mixed-precision behavior.
        squared_mean = inputs.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = inputs * torch.rsqrt(squared_mean + self.eps).to(inputs.dtype)
        return normalized * self.weight.to(inputs.dtype)
