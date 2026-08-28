"""Conditional-flow helpers for action-chunk training."""

import torch
import torch.nn.functional as functional
from torch import Tensor


def flow_matching_batch(
    target_actions: Tensor,
    *,
    noise: Tensor | None = None,
    flow_time: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Create linear probability-path samples and target vector fields.

    At time zero the trajectory is Gaussian noise; at time one it is the
    demonstrated action chunk. The target vector field is constant for this
    simple conditional optimal-transport path.
    """

    batch = target_actions.shape[0]
    noise = torch.randn_like(target_actions) if noise is None else noise
    flow_time = (
        torch.rand(batch, device=target_actions.device, dtype=target_actions.dtype)
        if flow_time is None
        else flow_time
    )
    interpolation = flow_time.reshape(batch, *([1] * (target_actions.ndim - 1)))
    noisy_actions = (1.0 - interpolation) * noise + interpolation * target_actions
    target_velocity = target_actions - noise
    return noisy_actions, flow_time, target_velocity


def flow_matching_loss(predicted_velocity: Tensor, target_velocity: Tensor) -> Tensor:
    return functional.mse_loss(predicted_velocity, target_velocity)
