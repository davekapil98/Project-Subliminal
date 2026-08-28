"""Stateful validation such as staleness and sequence monotonicity."""

from __future__ import annotations

import time

from bus.message import BusMessage
from bus.schemas import SchemaError, validate_message


class PacketValidator:
    def __init__(self, *, bus_dim: int = 512, max_age_seconds: float = 1.0) -> None:
        self.bus_dim = bus_dim
        self.max_age_seconds = max_age_seconds
        self._latest_sequence: dict[tuple[str, str], int] = {}

    def validate(self, message: BusMessage, *, now: float | None = None) -> None:
        validate_message(message, bus_dim=self.bus_dim)
        now = time.time() if now is None else now
        if now - message.header.timestamp > self.max_age_seconds:
            raise SchemaError("stale packet")
        key = (message.header.source, message.header.message_type.value)
        previous = self._latest_sequence.get(key)
        if previous is not None and message.header.sequence_id <= previous:
            raise SchemaError("duplicate or out-of-order packet")
        self._latest_sequence[key] = message.header.sequence_id
