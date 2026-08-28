"""Validated numeric subgoal fields emitted by the Executive."""

import torch
import torch.nn.functional as functional
from torch import Tensor, nn


class StructuredSubgoalHead(nn.Module):
    def __init__(self, d_model: int, joints: int = 6, decisions: int = 5) -> None:
        super().__init__()
        self.joints = joints
        self.fields = nn.Linear(d_model, joints * 2 + 3)
        self.decision = nn.Linear(d_model, decisions)

    def forward(self, hidden: Tensor, current_q: Tensor) -> dict[str, Tensor]:
        fields = self.fields(hidden)
        q_delta = torch.tanh(fields[:, : self.joints]) * 0.35
        qdot = torch.tanh(fields[:, self.joints : self.joints * 2]) * 1.5
        duration = functional.softplus(fields[:, self.joints * 2 : self.joints * 2 + 1])
        constraints = torch.sigmoid(fields[:, -2:])
        return {
            "q_goal": current_q + q_delta,
            "qdot_goal": qdot,
            "duration": duration + 0.1,
            "constraints": constraints,
            "decision_logits": self.decision(hidden),
        }
