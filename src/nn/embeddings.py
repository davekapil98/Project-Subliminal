"""Numeric and continuous-time token embeddings."""

import math

import torch
from torch import Tensor, nn


class NumericTokenEmbedding(nn.Module):
    """Project one typed numeric feature vector into one Transformer token."""

    def __init__(self, input_dim: int, d_model: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.projection(inputs).unsqueeze(-2)


class FourierTimeEmbedding(nn.Module):
    """Embed scalar flow time with fixed Fourier features plus a learned MLP."""

    def __init__(self, d_model: int, fourier_dim: int = 32) -> None:
        super().__init__()
        if fourier_dim % 2:
            raise ValueError("fourier_dim must be even")
        frequencies = torch.exp(
            torch.linspace(math.log(1.0), math.log(1_000.0), fourier_dim // 2)
        )
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.projection = nn.Sequential(
            nn.Linear(fourier_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, flow_time: Tensor) -> Tensor:
        phases = flow_time.float().reshape(-1, 1) * self.frequencies.reshape(1, -1)
        features = torch.cat((phases.sin(), phases.cos()), dim=-1)
        return self.projection(features)
