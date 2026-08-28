"""Tiny decoder-Transformer Executive and structured subgoal head.

The final learned query token causally reads TASK_GOAL, WORLD_STATE, body/world
predictions, current robot state, and optional memory. It emits typed physical
fields and never emits servo commands.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from models.executive.stage_jepa import StageJEPAPredictor
from models.executive.structured_head import StructuredSubgoalHead
from nn.rmsnorm import RMSNorm
from nn.transformer_block import TransformerBlock


@dataclass
class ExecutiveOutput:
    q_goal: Tensor
    qdot_goal: Tensor
    duration: Tensor
    constraints: Tensor
    decision_logits: Tensor
    next_stage_latent: Tensor

    def motor_goal_tensor(self) -> Tensor:
        return torch.cat(
            (self.q_goal, self.qdot_goal, self.duration, self.constraints), dim=-1
        )


class TinyExecutive(nn.Module):
    def __init__(
        self,
        *,
        bus_dim: int = 64,
        joints: int = 6,
        d_model: int = 256,
        depth: int = 6,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.joints = joints
        self.bus_adapter = nn.Linear(bus_dim, d_model)
        self.robot_adapter = nn.Linear(joints * 2, d_model)
        self.body_adapter = nn.Linear(joints * 2, d_model)
        self.event_adapter = nn.Linear(3, d_model)
        self.type_embedding = nn.Parameter(torch.randn(6, d_model) * 0.02)
        self.output_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, num_kv_heads=max(1, num_heads // 4)) for _ in range(depth)]
        )
        self.norm = RMSNorm(d_model)
        self.structured_head = StructuredSubgoalHead(d_model, joints)
        self.stage_jepa = StageJEPAPredictor(d_model, bus_dim)

    def forward(
        self,
        task_semantic: Tensor,
        world_tokens: Tensor,
        robot_state: Tensor,
        *,
        memory_tokens: Tensor | None = None,
        body_predictions: Tensor | None = None,
        world_event_logits: Tensor | None = None,
    ) -> ExecutiveOutput:
        batch = task_semantic.shape[0]
        tokens = [
            self.bus_adapter(task_semantic).unsqueeze(1) + self.type_embedding[0],
            self.bus_adapter(world_tokens) + self.type_embedding[1],
            self.robot_adapter(robot_state).unsqueeze(1) + self.type_embedding[2],
        ]
        if memory_tokens is not None:
            tokens.append(self.bus_adapter(memory_tokens) + self.type_embedding[3])
        if body_predictions is not None:
            body_summary = body_predictions.mean(dim=(-3, -2))
            tokens.append(self.body_adapter(body_summary).unsqueeze(1) + self.type_embedding[4])
        if world_event_logits is not None:
            event_summary = world_event_logits.mean(dim=1)
            tokens.append(self.event_adapter(event_summary).unsqueeze(1) + self.type_embedding[5])
        tokens.append(self.output_query.expand(batch, -1, -1))
        hidden = torch.cat(tokens, dim=1)
        for block in self.blocks:
            hidden = block(hidden, causal=True)
        output_hidden = self.norm(hidden[:, -1])
        fields = self.structured_head(output_hidden, robot_state[:, : self.joints])
        return ExecutiveOutput(
            q_goal=fields["q_goal"],
            qdot_goal=fields["qdot_goal"],
            duration=fields["duration"],
            constraints=fields["constraints"],
            decision_logits=fields["decision_logits"],
            next_stage_latent=self.stage_jepa(output_hidden),
        )
