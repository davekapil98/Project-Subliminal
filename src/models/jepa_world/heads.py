"""World-event and uncertainty prediction heads."""

from torch import Tensor, nn


class WorldPredictionHeads(nn.Module):
    def __init__(self, d_model: int, bus_dim: int, event_count: int = 3) -> None:
        super().__init__()
        self.future = nn.Linear(d_model, bus_dim)
        self.events = nn.Linear(d_model, event_count)
        self.log_variance = nn.Linear(d_model, 1)

    def forward(self, world_hidden: Tensor, pooled: Tensor) -> dict[str, Tensor]:
        return {
            "future_tokens": self.future(world_hidden),
            "event_logits": self.events(pooled),
            "log_variance": self.log_variance(pooled).clamp(-8.0, 5.0),
        }
