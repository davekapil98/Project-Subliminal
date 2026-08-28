"""Toy dynamics model with mean and explicitly bounded log variance."""

import torch
from torch import Tensor, nn


class ToyGaussianDynamics(nn.Module):
    def __init__(self, state_width: int = 12, action_width: int = 6) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_width + action_width, 32),
            nn.SiLU(),
            nn.Linear(32, state_width * 2),
        )

    def forward(self, state: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        mean, log_variance = self.network(torch.cat((state, action), dim=-1)).chunk(2, -1)
        return mean, log_variance.clamp(-8.0, 5.0)
