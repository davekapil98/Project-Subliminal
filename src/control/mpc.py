"""Deterministic model-predictive candidate scoring."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class MPCWeights:
    goal: float = 4.0
    velocity: float = 0.5
    smoothness: float = 0.25
    joint_limit: float = 10.0
    uncertainty: float = 0.5
    collision: float = 20.0
    task: float = 1.0


def score_action_candidates(
    *,
    actions: Tensor,
    body_mean: Tensor,
    body_log_variance: Tensor,
    q_goal: Tensor,
    qdot_goal: Tensor,
    joint_min: Tensor,
    joint_max: Tensor,
    collision_probability: Tensor,
    task_cost: Tensor | None = None,
    weights: MPCWeights = MPCWeights(),
) -> Tensor:
    """Return lower-is-better costs shaped ``[B, K]``."""

    final_q = body_mean[:, :, -1, :6]
    final_qdot = body_mean[:, :, -1, 6:]
    goal_error = (final_q - q_goal[:, None]).square().mean(dim=-1)
    velocity_error = (final_qdot - qdot_goal[:, None]).square().mean(dim=-1)
    if actions.shape[2] >= 3:
        jerk = actions.diff(dim=2).diff(dim=2).square().mean(dim=(-1, -2))
    else:
        jerk = torch.zeros_like(goal_error)
    predicted_q = body_mean[..., :6]
    below = (joint_min - predicted_q).clamp_min(0.0)
    above = (predicted_q - joint_max).clamp_min(0.0)
    limit_penalty = (below.square() + above.square()).mean(dim=(-1, -2))
    uncertainty = body_log_variance.exp().mean(dim=(-1, -2))
    cost = (
        weights.goal * goal_error
        + weights.velocity * velocity_error
        + weights.smoothness * jerk
        + weights.joint_limit * limit_penalty
        + weights.uncertainty * uncertainty
        + weights.collision * collision_probability
    )
    if task_cost is not None:
        cost = cost + weights.task * task_cost
    return cost


def choose_candidates(actions: Tensor, costs: Tensor) -> tuple[Tensor, Tensor]:
    """Gather the minimum-cost action chunk for each batch element."""

    best = costs.argmin(dim=1)
    batch_indices = torch.arange(actions.shape[0], device=actions.device)
    return actions[batch_indices, best], best
