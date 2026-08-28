"""JEPA latent predictor used for masked and temporal Stage 0 objectives."""

import torch
from torch import Tensor, nn

from nn.rmsnorm import RMSNorm
from nn.transformer_block import TransformerBlock


class JEPALatentPredictor(nn.Module):
    def __init__(
        self,
        bus_dim: int = 64,
        d_model: int = 192,
        depth: int = 2,
        num_heads: int = 6,
    ) -> None:
        super().__init__()
        self.input = nn.Linear(bus_dim, d_model)
        self.mask_token = nn.Parameter(Tensor(1, 1, d_model))
        nn.init.normal_(self.mask_token, std=0.02)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads) for _ in range(depth)]
        )
        self.norm = RMSNorm(d_model)
        self.output = nn.Linear(d_model, bus_dim)

    def forward(self, context_tokens: Tensor, predict_mask: Tensor) -> Tensor:
        if predict_mask.shape != context_tokens.shape[:2]:
            raise ValueError("predict_mask must have shape [B, N]")
        hidden = self.input(context_tokens)
        hidden = torch_where_mask(predict_mask, self.mask_token, hidden)
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(self.norm(hidden))


def torch_where_mask(mask: Tensor, replacement: Tensor, values: Tensor) -> Tensor:
    # Keep the [B, N, 1] condition explicit; torch.where broadcasts it over D.
    return torch.where(mask.unsqueeze(-1), replacement.expand_as(values), values)
