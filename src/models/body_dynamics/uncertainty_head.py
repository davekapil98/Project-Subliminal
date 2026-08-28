"""Stable Gaussian mean/log-variance prediction."""

from torch import Tensor, nn


class GaussianStateHead(nn.Module):
    def __init__(self, d_model: int, state_dim: int) -> None:
        super().__init__()
        self.mean_residual = nn.Linear(d_model, state_dim)
        self.log_variance = nn.Linear(d_model, state_dim)
        nn.init.zeros_(self.mean_residual.weight)
        nn.init.zeros_(self.mean_residual.bias)
        nn.init.zeros_(self.log_variance.weight)
        nn.init.constant_(self.log_variance.bias, -2.0)

    def forward(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        return self.mean_residual(hidden), self.log_variance(hidden).clamp(-8.0, 5.0)
