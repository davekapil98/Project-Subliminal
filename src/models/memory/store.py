"""Small external episodic store used by Stage 0 integration tests."""

from dataclasses import dataclass
import time

import torch
from torch import Tensor


@dataclass(frozen=True)
class MemoryRecord:
    embedding: Tensor
    timestamp: float
    confidence: float
    kind: str
    metadata: dict[str, str]


class InMemoryStore:
    def __init__(self, bus_dim: int, *, capacity: int = 128) -> None:
        self.bus_dim = bus_dim
        self.capacity = capacity
        self._records: list[MemoryRecord] = []

    def __len__(self) -> int:
        return len(self._records)

    def write(
        self,
        embedding: Tensor,
        *,
        confidence: float,
        kind: str,
        metadata: dict[str, str] | None = None,
        timestamp: float | None = None,
    ) -> None:
        if embedding.shape != (self.bus_dim,):
            raise ValueError(f"memory embedding must have shape [{self.bus_dim}]")
        self._records.append(
            MemoryRecord(
                embedding=embedding.detach().cpu().clone(),
                timestamp=time.time() if timestamp is None else timestamp,
                confidence=float(confidence),
                kind=kind,
                metadata=dict(metadata or {}),
            )
        )
        if len(self._records) > self.capacity:
            self._records = self._records[-self.capacity :]

    def tensors(self, *, now: float | None = None) -> tuple[Tensor, Tensor, Tensor]:
        if not self._records:
            raise RuntimeError("memory store is empty")
        now = time.time() if now is None else now
        entries = torch.stack([record.embedding for record in self._records])
        ages = torch.tensor([max(0.0, now - record.timestamp) for record in self._records])
        recency = torch.exp(-ages / 60.0)
        confidence = torch.tensor([record.confidence for record in self._records])
        return entries, recency, confidence
