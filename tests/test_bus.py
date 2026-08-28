import time

import pytest
import torch

from bus import BusMessage, MessageHeader, MessageType, SchemaError, validate_message
from bus.validation import PacketValidator
from models.orchestrator import MessageRouter


def task_message(*, sequence: int = 1, timestamp: float | None = None) -> BusMessage:
    return BusMessage(
        MessageHeader.create(
            MessageType.TASK_GOAL,
            "language",
            "executive",
            sequence,
            timestamp=time.time() if timestamp is None else timestamp,
            correlation_id="request-1",
        ),
        {"semantic_token": torch.randn(2, 8), "intent_id": torch.tensor([1, 2])},
        {"label": "toy"},
    )


def test_message_round_trip_preserves_header_tensors_and_metadata() -> None:
    message = task_message()
    decoded = BusMessage.from_json(message.to_json())
    assert decoded.header == message.header
    assert decoded.metadata == message.metadata
    torch.testing.assert_close(decoded.tensors["semantic_token"], message.tensors["semantic_token"])
    validate_message(decoded, bus_dim=8)


def test_world_state_rejects_wrong_bus_width() -> None:
    message = BusMessage(
        MessageHeader.create(MessageType.WORLD_STATE, "jepa", "executive", 1),
        {"world_tokens": torch.randn(1, 4, 7)},
    )
    with pytest.raises(SchemaError, match="last dimension"):
        validate_message(message, bus_dim=8)


def test_packet_validator_rejects_stale_and_duplicate_packets() -> None:
    now = time.time()
    validator = PacketValidator(bus_dim=8, max_age_seconds=0.5)
    with pytest.raises(SchemaError, match="stale"):
        validator.validate(task_message(timestamp=now - 1.0), now=now)
    validator.validate(task_message(sequence=2, timestamp=now), now=now)
    with pytest.raises(SchemaError, match="duplicate"):
        validator.validate(task_message(sequence=2, timestamp=now), now=now)


def test_router_retains_request_correlation() -> None:
    router = MessageRouter(bus_dim=8)

    def handle(message: BusMessage) -> BusMessage:
        return BusMessage(
            MessageHeader.create(
                MessageType.TASK_GOAL,
                "executive",
                "language",
                1,
                correlation_id=message.header.correlation_id,
            ),
            message.tensors,
        )

    router.register("executive", handle)
    message = task_message()
    result = router.route(message, now=message.header.timestamp)
    assert result.accepted
    assert result.response is not None
    assert result.response.header.correlation_id == message.header.correlation_id
