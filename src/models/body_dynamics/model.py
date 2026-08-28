"""Tiny SO-101-specific numeric body dynamics model.

Shapes:
    body_state: [B, 12] containing q and qdot
    action_candidates: [B, K, H, 6] relative position commands
    mean/log_variance: [B, K, H, 12]
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from models.body_dynamics.uncertainty_head import GaussianStateHead
from nn.rmsnorm import RMSNorm
from nn.transformer_block import TransformerBlock


@dataclass
class BodyDynamicsOutput:
    mean: Tensor
    log_variance: Tensor


class TinyBodyDynamics(nn.Module):
    def __init__(
        self,
        *,
        joints: int = 6,
        d_model: int = 192,
        depth: int = 3,
        num_heads: int = 6,
        control_hz: float = 30.0,
    ) -> None:
        super().__init__()
        self.joints = joints
        self.state_dim = joints * 2
        self.control_hz = control_hz
        self.state_embedding = nn.Sequential(
            nn.Linear(self.state_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        self.action_embedding = nn.Linear(joints, d_model)
        self.type_embedding = nn.Parameter(torch.randn(2, d_model) * 0.02)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads) for _ in range(depth)]
        )
        self.norm = RMSNorm(d_model)
        self.head = GaussianStateHead(d_model, self.state_dim)

    def forward(self, body_state: Tensor, action_candidates: Tensor) -> BodyDynamicsOutput:
        if body_state.ndim != 2 or body_state.shape[-1] != self.state_dim:
            raise ValueError(f"body_state must have shape [B, {self.state_dim}]")
        if action_candidates.ndim != 4 or action_candidates.shape[-1] != self.joints:
            raise ValueError("action_candidates must have shape [B, K, H, joints]")
        batch, candidates, horizon, _ = action_candidates.shape
        if body_state.shape[0] != batch:
            raise ValueError("body state and action batches must match")

        repeated_state = body_state.repeat_interleave(candidates, dim=0)
        actions = action_candidates.reshape(batch * candidates, horizon, self.joints)
        state_token = self.state_embedding(repeated_state).unsqueeze(1)
        state_token = state_token + self.type_embedding[0]
        action_tokens = self.action_embedding(actions) + self.type_embedding[1]
        hidden = torch.cat((state_token, action_tokens), dim=1)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.norm(hidden[:, 1:])
        residual, log_variance = self.head(hidden)

        current_q = repeated_state[:, : self.joints]
        baseline_q = current_q[:, None] + actions.cumsum(dim=1)
        baseline_qdot = actions * self.control_hz
        baseline = torch.cat((baseline_q, baseline_qdot), dim=-1)
        mean = baseline + residual
        return BodyDynamicsOutput(
            mean=mean.reshape(batch, candidates, horizon, self.state_dim),
            log_variance=log_variance.reshape(
                batch, candidates, horizon, self.state_dim
            ),
        )
