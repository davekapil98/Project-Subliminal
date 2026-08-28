"""Small learned destination router; it does not make task plans."""

import torch
from torch import Tensor, nn


class TinyOrchestrator(nn.Module):
    def __init__(self, bus_dim: int = 64, hidden_dim: int = 64, routes: int = 8) -> None:
        super().__init__()
        self.router = nn.Sequential(
            nn.Linear(bus_dim + 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, routes),
        )

    def forward(self, request_token: Tensor, routing_features: Tensor) -> Tensor:
        """Return route logits from request latent plus priority/age/confidence."""

        if routing_features.shape[-1] != 3:
            raise ValueError("routing_features must contain priority, age, confidence")
        return self.router(torch.cat((request_token, routing_features), dim=-1))
