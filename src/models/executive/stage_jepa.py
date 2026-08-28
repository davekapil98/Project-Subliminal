"""Auxiliary next-task-stage latent predictor."""

from torch import Tensor, nn


class StageJEPAPredictor(nn.Module):
    def __init__(self, d_model: int, bus_dim: int) -> None:
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, bus_dim)
        )

    def forward(self, executive_hidden: Tensor) -> Tensor:
        return self.predictor(executive_hidden)
