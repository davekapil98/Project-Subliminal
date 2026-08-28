"""Structured task-semantics classifier heads."""

from torch import Tensor, nn


class IntentEntityHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        *,
        intents: int = 8,
        entities: int = 16,
        attributes: int = 16,
    ) -> None:
        super().__init__()
        self.intent = nn.Linear(d_model, intents)
        self.entity = nn.Linear(d_model, entities)
        self.attribute = nn.Linear(d_model, attributes)

    def forward(self, hidden: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return self.intent(hidden), self.entity(hidden), self.attribute(hidden)
