"""Tiny action-conditioned JEPA world predictor.

Shapes:
    world_tokens: [B, N, bus_dim]
    action_candidates: [B, K, H, 6]
    future_tokens: [B, K, N, bus_dim]
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from models.jepa_world.action_adapter import ActionAdapter
from models.jepa_world.heads import WorldPredictionHeads
from nn.rmsnorm import RMSNorm
from nn.transformer_block import TransformerBlock


@dataclass
class JEPAWorldOutput:
    future_tokens: Tensor
    event_logits: Tensor
    log_variance: Tensor


class TinyJEPAWorldPredictor(nn.Module):
    def __init__(
        self,
        *,
        bus_dim: int = 64,
        action_dim: int = 6,
        d_model: int = 192,
        depth: int = 4,
        num_heads: int = 6,
    ) -> None:
        super().__init__()
        self.world_adapter = nn.Linear(bus_dim, d_model)
        self.action_adapter = ActionAdapter(action_dim, d_model)
        self.type_embedding = nn.Parameter(Tensor(2, d_model))
        nn.init.normal_(self.type_embedding, std=0.02)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads) for _ in range(depth)]
        )
        self.norm = RMSNorm(d_model)
        self.heads = WorldPredictionHeads(d_model, bus_dim)

    def forward(self, world_tokens: Tensor, action_candidates: Tensor) -> JEPAWorldOutput:
        if world_tokens.ndim != 3:
            raise ValueError("world_tokens must have shape [B, N, D]")
        if action_candidates.ndim != 4 or action_candidates.shape[-1] != 6:
            raise ValueError("action_candidates must have shape [B, K, H, 6]")
        batch, candidates, horizon, _ = action_candidates.shape
        if world_tokens.shape[0] != batch:
            raise ValueError("world and action candidate batches must match")
        world_count = world_tokens.shape[1]

        repeated_world = world_tokens[:, None].expand(-1, candidates, -1, -1)
        repeated_world = repeated_world.reshape(batch * candidates, world_count, -1)
        actions = action_candidates.reshape(batch * candidates, horizon, 6)
        world_hidden = self.world_adapter(repeated_world) + self.type_embedding[0]
        action_hidden = self.action_adapter(actions) + self.type_embedding[1]
        hidden = torch.cat((world_hidden, action_hidden), dim=1)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.norm(hidden)
        predicted_world = hidden[:, :world_count]
        pooled = hidden.mean(dim=1)
        heads = self.heads(predicted_world, pooled)
        return JEPAWorldOutput(
            future_tokens=heads["future_tokens"].reshape(
                batch, candidates, world_count, -1
            ),
            event_logits=heads["event_logits"].reshape(batch, candidates, -1),
            log_variance=heads["log_variance"].reshape(batch, candidates, 1),
        )
