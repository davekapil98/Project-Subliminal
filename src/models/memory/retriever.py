"""Differentiable relevance, recency, and confidence scoring."""

import torch
from torch import Tensor, nn


class MemoryRetriever(nn.Module):
    def __init__(self, bus_dim: int, d_model: int) -> None:
        super().__init__()
        self.query = nn.Linear(bus_dim, d_model)
        self.entry = nn.Linear(bus_dim, d_model)
        self.recency_weight = nn.Parameter(torch.tensor(0.1))
        self.confidence_weight = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        query: Tensor,
        entries: Tensor,
        recency: Tensor,
        confidence: Tensor,
    ) -> Tensor:
        projected_query = nn.functional.normalize(self.query(query), dim=-1)
        projected_entries = nn.functional.normalize(self.entry(entries), dim=-1)
        relevance = (projected_entries * projected_query[:, None]).sum(dim=-1)
        return (
            relevance
            + self.recency_weight * recency
            + self.confidence_weight * confidence
        )
