"""Stage 0 deterministic pose-to-joint goal placeholder."""

from torch import Tensor


def validate_joint_goal(q_goal: Tensor) -> Tensor:
    if q_goal.ndim != 2 or q_goal.shape[-1] != 6:
        raise ValueError("q_goal must have shape [B, 6]")
    return q_goal
