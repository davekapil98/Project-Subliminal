"""Typed action trajectory adapter for the JEPA world predictor."""

from torch import Tensor, nn


class ActionAdapter(nn.Module):
    def __init__(self, action_dim: int, d_model: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(action_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )

    def forward(self, actions: Tensor) -> Tensor:
        return self.projection(actions)
