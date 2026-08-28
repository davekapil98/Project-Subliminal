"""Deterministic schema and freshness guard."""

from bus.message import BusMessage
from bus.validation import PacketValidator


class SchemaGuard:
    def __init__(self, *, bus_dim: int = 512, max_age_seconds: float = 1.0) -> None:
        self.validator = PacketValidator(
            bus_dim=bus_dim, max_age_seconds=max_age_seconds
        )

    def accept(self, message: BusMessage, *, now: float | None = None) -> BusMessage:
        self.validator.validate(message, now=now)
        return message
