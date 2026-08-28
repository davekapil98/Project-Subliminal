"""Exponential-moving-average target encoder for JEPA training."""

from copy import deepcopy

import torch
from torch import nn


class EMATargetEncoder(nn.Module):
    def __init__(self, online_encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = deepcopy(online_encoder).eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, online_encoder: nn.Module, momentum: float = 0.996) -> None:
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        online = dict(online_encoder.named_parameters())
        for name, target_parameter in self.encoder.named_parameters():
            target_parameter.lerp_(online[name].detach(), 1.0 - momentum)
        online_buffers = dict(online_encoder.named_buffers())
        for name, target_buffer in self.encoder.named_buffers():
            if name in online_buffers:
                target_buffer.copy_(online_buffers[name])

    @torch.no_grad()
    def forward(self, *args: object, **kwargs: object) -> object:
        return self.encoder(*args, **kwargs)
