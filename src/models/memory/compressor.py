"""Learned compression of retrieved episodic entries."""

from torch import Tensor, nn


class MemoryCompressor(nn.Module):
    def __init__(self, bus_dim: int, d_model: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(bus_dim, d_model), nn.SiLU(), nn.Linear(d_model, bus_dim)
        )

    def forward(self, retrieved: Tensor, weights: Tensor) -> Tensor:
        weighted = (retrieved * weights.unsqueeze(-1)).sum(dim=1)
        return self.network(weighted)
