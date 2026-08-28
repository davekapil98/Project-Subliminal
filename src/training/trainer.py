"""Small reference training step with gradient accumulation semantics."""

from collections.abc import Callable

import torch
from torch import Tensor, nn

from training.precision import PrecisionPolicy


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_closure: Callable[[], Tensor],
    *,
    precision: PrecisionPolicy,
    gradient_clip_norm: float | None = 1.0,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with precision.autocast():
        loss = loss_closure()
    if not torch.isfinite(loss):
        raise FloatingPointError("training loss is not finite")
    loss.backward()
    if gradient_clip_norm is not None:
        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
    optimizer.step()
    return float(loss.detach())
