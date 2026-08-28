"""Tiny controller for external memory reads, writes, and compression."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from models.memory.compressor import MemoryCompressor
from models.memory.retriever import MemoryRetriever


@dataclass
class MemoryControllerOutput:
    scores: Tensor
    retrieved_entries: Tensor
    retrieved_scores: Tensor
    compressed: Tensor
    write_logits: Tensor


class TinyMemoryController(nn.Module):
    """Neural policy around external storage; entries are not model weights."""

    def __init__(
        self,
        *,
        bus_dim: int = 64,
        d_model: int = 128,
        top_k: int = 4,
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.retriever = MemoryRetriever(bus_dim, d_model)
        self.compressor = MemoryCompressor(bus_dim, d_model)
        self.write_gate = nn.Sequential(
            nn.Linear(bus_dim * 2, d_model), nn.SiLU(), nn.Linear(d_model, 4)
        )

    def forward(
        self,
        query: Tensor,
        entries: Tensor,
        recency: Tensor,
        confidence: Tensor,
    ) -> MemoryControllerOutput:
        if entries.ndim != 3 or entries.shape[0] != query.shape[0]:
            raise ValueError("entries must have shape [B, M, bus_dim]")
        scores = self.retriever(query, entries, recency, confidence)
        count = min(self.top_k, entries.shape[1])
        retrieved_scores, indices = scores.topk(count, dim=1)
        gather_indices = indices.unsqueeze(-1).expand(-1, -1, entries.shape[-1])
        retrieved = entries.gather(1, gather_indices)
        weights = torch.softmax(retrieved_scores, dim=1)
        compressed = self.compressor(retrieved, weights)
        write_logits = self.write_gate(torch.cat((query, compressed), dim=-1))
        return MemoryControllerOutput(
            scores=scores,
            retrieved_entries=retrieved,
            retrieved_scores=retrieved_scores,
            compressed=compressed,
            write_logits=write_logits,
        )
