"""SwiGLU feed-forward layer."""

import torch.nn.functional as functional
from torch import Tensor, nn


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.gate_and_value = nn.Linear(d_model, hidden_dim * 2, bias=False)
        self.output = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        gate, value = self.gate_and_value(inputs).chunk(2, dim=-1)
        return self.output(self.dropout(functional.silu(gate) * value))
