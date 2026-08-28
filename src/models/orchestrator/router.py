"""Synchronous Stage 0 message router with schema validation."""

from dataclasses import dataclass
from typing import Callable

from bus.message import BusMessage
from models.orchestrator.schema_guard import SchemaGuard


MessageHandler = Callable[[BusMessage], BusMessage | None]


@dataclass(frozen=True)
class RouteResult:
    accepted: bool
    response: BusMessage | None
    destination: str


class MessageRouter:
    def __init__(self, *, bus_dim: int = 512, max_age_seconds: float = 1.0) -> None:
        self.guard = SchemaGuard(bus_dim=bus_dim, max_age_seconds=max_age_seconds)
        self._handlers: dict[str, MessageHandler] = {}

    def register(self, destination: str, handler: MessageHandler) -> None:
        if not destination:
            raise ValueError("destination must be non-empty")
        self._handlers[destination] = handler

    def route(self, message: BusMessage, *, now: float | None = None) -> RouteResult:
        accepted = self.guard.accept(message, now=now)
        destination = accepted.header.destination
        if destination not in self._handlers:
            raise KeyError(f"no handler registered for {destination}")
        response = self._handlers[destination](accepted)
        if response is not None:
            if response.header.correlation_id != message.header.correlation_id:
                raise ValueError("response correlation_id does not match request")
        return RouteResult(True, response, destination)
