"""RMSNorm equation: x / sqrt(mean(x**2) + eps) * learned_scale."""

import torch
from torch import Tensor, nn


class LearningRMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, tokens: Tensor) -> Tensor:
        rms = torch.sqrt(tokens.square().mean(dim=-1, keepdim=True) + self.eps)
        return tokens / rms * self.scale
