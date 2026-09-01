"""Action-free temporal JEPA objective admitted by the Stage 1.5 gate."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn
from torch.nn import functional as F

from models.jepa_encoder.model import MultimodalJEPAEncoder
from models.jepa_encoder.predictor import JEPALatentPredictor
from models.jepa_encoder.target_encoder import EMATargetEncoder


@dataclass(frozen=True)
class TemporalJEPAOutput:
    predicted_tokens: Tensor
    target_tokens: Tensor


class ActionFreeTemporalJEPA(nn.Module):
    """Predict future visual latents without accepting an action tensor.

    Context proprioception is observed history. The target encoder receives
    zero proprioception so its stopped-gradient target is determined by future
    images and camera availability, never by a current or future action field.
    """

    def __init__(
        self,
        encoder: MultimodalJEPAEncoder,
        predictor: JEPALatentPredictor,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.target_encoder = EMATargetEncoder(encoder)

    def train(self, mode: bool = True) -> "ActionFreeTemporalJEPA":
        super().train(mode)
        self.target_encoder.eval()
        return self

    def forward(
        self,
        context_rgb: Tensor,
        future_rgb: Tensor,
        proprioception: Tensor,
        camera_valid: Tensor,
    ) -> TemporalJEPAOutput:
        if context_rgb.shape != future_rgb.shape:
            raise ValueError("context and future RGB tensors must have equal shapes")
        context = self.encoder(
            context_rgb,
            proprioception,
            camera_valid=camera_valid,
        ).world_tokens
        target_proprio = proprioception.new_zeros(proprioception.shape)
        target = self.target_encoder(
            future_rgb,
            target_proprio,
            camera_valid=camera_valid,
        ).world_tokens
        predict_mask = context.new_zeros(context.shape[:2], dtype=bool)
        predicted = self.predictor(context, predict_mask)
        return TemporalJEPAOutput(
            predicted_tokens=F.layer_norm(predicted, (predicted.shape[-1],)),
            target_tokens=F.layer_norm(target, (target.shape[-1],)),
        )

    def update_target(self, momentum: float) -> None:
        self.target_encoder.update(self.encoder, momentum=momentum)
