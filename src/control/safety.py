"""Hard deterministic joint, velocity, and command safety limits."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class SafetyLimits:
    joint_min: Tensor
    joint_max: Tensor
    max_velocity: Tensor
    max_relative_command: Tensor

    @classmethod
    def conservative_stage0(cls) -> "SafetyLimits":
        # Synthetic limits only. Physical SO-101 limits must come from the
        # calibrated URDF and hardware safety configuration before deployment.
        return cls(
            joint_min=torch.full((6,), -2.5),
            joint_max=torch.full((6,), 2.5),
            max_velocity=torch.full((6,), 1.5),
            max_relative_command=torch.full((6,), 0.05),
        )


class SafetyLayer:
    def __init__(self, limits: SafetyLimits, *, control_hz: float = 30.0) -> None:
        self.limits = limits
        self.control_hz = control_hz

    def filter_chunk(self, current_q: Tensor, relative_actions: Tensor) -> Tensor:
        """Clamp a relative action chunk while propagating its safe joint state."""

        if current_q.ndim != 2 or current_q.shape[-1] != 6:
            raise ValueError("current_q must have shape [B, 6]")
        if relative_actions.ndim != 3 or relative_actions.shape[-1] != 6:
            raise ValueError("relative_actions must have shape [B, H, 6]")
        device, dtype = current_q.device, current_q.dtype
        joint_min = self.limits.joint_min.to(device=device, dtype=dtype)
        joint_max = self.limits.joint_max.to(device=device, dtype=dtype)
        max_delta = torch.minimum(
            self.limits.max_relative_command.to(device=device, dtype=dtype),
            self.limits.max_velocity.to(device=device, dtype=dtype) / self.control_hz,
        )
        position = current_q
        safe_steps: list[Tensor] = []
        for step in relative_actions.unbind(dim=1):
            step = step.clamp(-max_delta, max_delta)
            next_position = (position + step).clamp(joint_min, joint_max)
            safe_step = next_position - position
            safe_steps.append(safe_step)
            position = next_position
        return torch.stack(safe_steps, dim=1)
