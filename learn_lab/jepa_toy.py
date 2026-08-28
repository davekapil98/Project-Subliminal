"""Toy JEPA: predict held-out target embeddings, never target pixels."""

from copy import deepcopy

import torch
from torch import Tensor, nn


class ToyJEPA(nn.Module):
    def __init__(self, input_width: int = 8, latent_width: int = 16) -> None:
        super().__init__()
        self.context_encoder = nn.Sequential(
            nn.Linear(input_width, latent_width), nn.SiLU(), nn.Linear(latent_width, latent_width)
        )
        self.target_encoder = deepcopy(self.context_encoder)
        self.predictor = nn.Linear(latent_width, latent_width)
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad_(False)

    def loss(self, context: Tensor, target: Tensor) -> Tensor:
        predicted = self.predictor(self.context_encoder(context))
        with torch.no_grad():
            target_latent = self.target_encoder(target)
        return (predicted - target_latent).square().mean()

    @torch.no_grad()
    def update_target(self, momentum: float = 0.99) -> None:
        for online, target in zip(
            self.context_encoder.parameters(), self.target_encoder.parameters(), strict=True
        ):
            target.lerp_(online, 1.0 - momentum)
