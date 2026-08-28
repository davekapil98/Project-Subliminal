"""Small helpers around parameter-producing linear layers."""

from torch import nn


def count_parameters(module: nn.Module, *, trainable_only: bool = True) -> int:
    """Return the number of scalar parameters in ``module``."""

    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad or not trainable_only
    )
