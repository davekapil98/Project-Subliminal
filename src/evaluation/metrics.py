"""Small offline metrics used by Stage 0 tests and reports."""

import torch
from torch import Tensor


def final_joint_error(actual_q: Tensor, goal_q: Tensor) -> Tensor:
    return torch.linalg.vector_norm(actual_q - goal_q, dim=-1).mean()


def prediction_rmse(predicted: Tensor, actual: Tensor) -> Tensor:
    return (predicted - actual).square().mean().sqrt()
