"""Adapters between module hidden widths and the common latent bus."""

from torch import Tensor, nn


class BusAdapter(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 512) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.projection(inputs)
