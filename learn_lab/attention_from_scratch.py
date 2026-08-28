"""Single-head scaled dot-product attention with an optional causal mask."""

import math

import torch
from torch import Tensor, nn


class LearningAttention(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)

    def forward(self, tokens: Tensor, *, causal: bool = False) -> tuple[Tensor, Tensor]:
        query, key, value = self.query(tokens), self.key(tokens), self.value(tokens)
        scores = query @ key.transpose(-2, -1) / math.sqrt(tokens.shape[-1])
        if causal:
            allowed = torch.ones(
                tokens.shape[1], tokens.shape[1], dtype=torch.bool, device=tokens.device
            ).tril()
            scores = scores.masked_fill(~allowed, float("-inf"))
        weights = scores.softmax(dim=-1)
        return weights @ value, weights
