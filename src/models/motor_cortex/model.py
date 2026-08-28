"""Tiny flow-matching Motor Cortex.

The model consumes only numeric actuator state and a physical motor goal. It has
no language, text-token, image, or world-latent input path.

Shapes:
    body_state: [B, 18] (q, qdot, previous command)
    motor_goal: [B, 15] (q_goal, qdot_goal, duration, two constraints)
    noisy_actions / vector_field: [B, H, 6]
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from models.motor_cortex.action_head import ActionVectorFieldHead
from nn.attention import GroupedQueryAttention
from nn.embeddings import FourierTimeEmbedding, NumericTokenEmbedding
from nn.rmsnorm import RMSNorm
from nn.swiglu import SwiGLU


class ConditionalRMSNorm(nn.Module):
    """RMSNorm modulated by a flow-time conditioning vector."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.modulation = nn.Linear(d_model, d_model * 2)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, inputs: Tensor, condition: Tensor) -> Tensor:
        scale, shift = self.modulation(condition).chunk(2, dim=-1)
        return self.norm(inputs) * (1.0 + scale[:, None]) + shift[:, None]


class FlowTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        kv_heads = max(1, num_heads // 2)
        self.attention_norm = ConditionalRMSNorm(d_model)
        self.attention = GroupedQueryAttention(d_model, num_heads, kv_heads)
        self.ffn_norm = ConditionalRMSNorm(d_model)
        self.ffn = SwiGLU(d_model, int(d_model * 8 / 3))

    def forward(self, inputs: Tensor, condition: Tensor) -> Tensor:
        hidden = inputs + self.attention(self.attention_norm(inputs, condition))
        return hidden + self.ffn(self.ffn_norm(hidden, condition))


@dataclass
class MotorCortexOutput:
    actions: Tensor
    confidence: Tensor


class TinyMotorCortex(nn.Module):
    def __init__(
        self,
        *,
        state_dim: int = 18,
        goal_dim: int = 15,
        joints: int = 6,
        horizon: int = 16,
        d_model: int = 256,
        depth: int = 4,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.joints = joints
        self.horizon = horizon
        self.state_embedding = NumericTokenEmbedding(state_dim, d_model)
        self.goal_embedding = NumericTokenEmbedding(goal_dim, d_model)
        self.action_embedding = nn.Linear(joints, d_model)
        self.type_embedding = nn.Parameter(torch.randn(3, d_model) * 0.02)
        self.time_embedding = FourierTimeEmbedding(d_model)
        self.blocks = nn.ModuleList(
            [FlowTransformerBlock(d_model, num_heads) for _ in range(depth)]
        )
        self.norm = RMSNorm(d_model)
        self.action_head = ActionVectorFieldHead(d_model, joints)

    def forward(
        self,
        body_state: Tensor,
        motor_goal: Tensor,
        noisy_actions: Tensor,
        flow_time: Tensor,
    ) -> Tensor:
        if noisy_actions.ndim != 3 or noisy_actions.shape[1:] != (
            self.horizon,
            self.joints,
        ):
            raise ValueError(f"noisy_actions must have shape [B, {self.horizon}, {self.joints}]")
        state = self.state_embedding(body_state) + self.type_embedding[0]
        goal = self.goal_embedding(motor_goal) + self.type_embedding[1]
        action = self.action_embedding(noisy_actions) + self.type_embedding[2]
        hidden = torch.cat((state, goal, action), dim=1)
        condition = self.time_embedding(flow_time)
        for block in self.blocks:
            hidden = block(hidden, condition)
        hidden = self.norm(hidden)
        return self.action_head(hidden[:, 2:])

    @torch.no_grad()
    def sample(
        self,
        body_state: Tensor,
        motor_goal: Tensor,
        *,
        candidates: int = 4,
        steps: int = 8,
        noise_scale: float = 0.15,
        goal_guidance: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> MotorCortexOutput:
        if steps < 1 or candidates < 1:
            raise ValueError("steps and candidates must be positive")
        batch = body_state.shape[0]
        expanded_state = body_state.repeat_interleave(candidates, dim=0)
        expanded_goal = motor_goal.repeat_interleave(candidates, dim=0)
        actions = torch.randn(
            batch * candidates,
            self.horizon,
            self.joints,
            device=body_state.device,
            dtype=body_state.dtype,
            generator=generator,
        ) * noise_scale
        delta = 1.0 / steps
        for index in range(steps):
            flow_time = torch.full(
                (batch * candidates,),
                index / steps,
                device=body_state.device,
                dtype=body_state.dtype,
            )
            actions = actions + delta * self(
                expanded_state, expanded_goal, actions, flow_time
            )

        if goal_guidance:
            current_q = expanded_state[:, : self.joints]
            goal_q = expanded_goal[:, : self.joints]
            per_step = (goal_q - current_q) / self.horizon
            nominal = per_step[:, None].expand(-1, self.horizon, -1)
            actions = actions + goal_guidance * nominal
        actions = actions.reshape(batch, candidates, self.horizon, self.joints)
        confidence = torch.exp(-actions.square().mean(dim=(-1, -2)))
        return MotorCortexOutput(actions=actions, confidence=confidence)
