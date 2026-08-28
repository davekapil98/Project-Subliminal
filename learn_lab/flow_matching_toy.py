"""Toy flow matching on vectors: noise at t=0, data at t=1."""

import torch
from torch import Tensor, nn


class ToyVectorField(nn.Module):
    def __init__(self, width: int = 6, hidden: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(width + 1, hidden), nn.SiLU(), nn.Linear(hidden, width)
        )

    def forward(self, point: Tensor, flow_time: Tensor) -> Tensor:
        return self.network(torch.cat((point, flow_time[:, None]), dim=-1))


def make_flow_example(data: Tensor, noise: Tensor, flow_time: Tensor) -> tuple[Tensor, Tensor]:
    point = (1.0 - flow_time[:, None]) * noise + flow_time[:, None] * data
    target_velocity = data - noise
    return point, target_velocity
