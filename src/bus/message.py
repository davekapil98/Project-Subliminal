"""Serializable bus message and header types."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import time
import uuid
from typing import Any

import torch
from torch import Tensor

from bus.schemas import MessageType


@dataclass(frozen=True, slots=True)
class MessageHeader:
    message_type: MessageType
    source: str
    destination: str
    timestamp: float
    sequence_id: int
    priority: int = 0
    confidence: float = 1.0
    frame_id: str = "world"
    correlation_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    @classmethod
    def create(
        cls,
        message_type: MessageType,
        source: str,
        destination: str,
        sequence_id: int,
        **kwargs: Any,
    ) -> "MessageHeader":
        return cls(
            message_type=message_type,
            source=source,
            destination=destination,
            timestamp=float(kwargs.pop("timestamp", time.time())),
            sequence_id=sequence_id,
            **kwargs,
        )


@dataclass(slots=True)
class BusMessage:
    """A header plus typed numeric payload and JSON-compatible metadata."""

    header: MessageHeader
    tensors: dict[str, Tensor]
    metadata: dict[str, Any] = field(default_factory=dict)
    validity_mask: Tensor | None = None

    def to_json(self) -> str:
        header = asdict(self.header)
        header["message_type"] = self.header.message_type.value
        body: dict[str, Any] = {
            "header": header,
            "tensors": {name: _tensor_to_dict(value) for name, value in self.tensors.items()},
            "metadata": self.metadata,
            "validity_mask": (
                _tensor_to_dict(self.validity_mask) if self.validity_mask is not None else None
            ),
        }
        return json.dumps(body, allow_nan=False, separators=(",", ":"))

    @classmethod
    def from_json(cls, encoded: str) -> "BusMessage":
        body = json.loads(encoded)
        header_data = dict(body["header"])
        header_data["message_type"] = MessageType(header_data["message_type"])
        validity = body.get("validity_mask")
        return cls(
            header=MessageHeader(**header_data),
            tensors={
                name: _tensor_from_dict(value) for name, value in body["tensors"].items()
            },
            metadata=body.get("metadata", {}),
            validity_mask=_tensor_from_dict(validity) if validity is not None else None,
        )


def _tensor_to_dict(tensor: Tensor) -> dict[str, Any]:
    detached = tensor.detach().cpu()
    dtype = str(detached.dtype).removeprefix("torch.")
    return {"dtype": dtype, "shape": list(detached.shape), "values": detached.tolist()}


def _tensor_from_dict(encoded: dict[str, Any]) -> Tensor:
    dtype_name = encoded["dtype"]
    if not hasattr(torch, dtype_name):
        raise ValueError(f"unsupported tensor dtype: {dtype_name}")
    tensor = torch.tensor(encoded["values"], dtype=getattr(torch, dtype_name))
    expected_shape = tuple(encoded["shape"])
    if tensor.shape != expected_shape:
        tensor = tensor.reshape(expected_shape)
    return tensor
