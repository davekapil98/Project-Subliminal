"""Deterministic numeric plant for Stage 0 integration tests only."""

import torch
from torch import Tensor

from control.safety import SafetyLayer, SafetyLimits


class MockSO101:
    def __init__(
        self,
        *,
        batch_size: int = 1,
        control_hz: float = 30.0,
        response_gain: float = 0.9,
    ) -> None:
        self.control_hz = control_hz
        self.response_gain = response_gain
        self.q = torch.zeros(batch_size, 6)
        self.qdot = torch.zeros(batch_size, 6)
        self.previous_command = torch.zeros(batch_size, 6)
        self.safety = SafetyLayer(
            SafetyLimits.conservative_stage0(), control_hz=control_hz
        )

    def state(self) -> Tensor:
        return torch.cat((self.q, self.qdot), dim=-1)

    def motor_state(self) -> Tensor:
        return torch.cat((self.q, self.qdot, self.previous_command), dim=-1)

    def execute_relative(self, action_prefix: Tensor) -> Tensor:
        safe = self.safety.filter_chunk(self.q, action_prefix)
        for command in safe.unbind(dim=1):
            applied = command * self.response_gain
            self.q = self.q + applied
            self.qdot = applied * self.control_hz
            self.previous_command = command
        return self.state()

    def hold(self) -> None:
        self.qdot.zero_()
        self.previous_command.zero_()
