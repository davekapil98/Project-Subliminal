"""Runtime validation for physical and latent message contracts."""

from __future__ import annotations

from enum import Enum
import math
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from bus.message import BusMessage


class MessageType(str, Enum):
    TASK_GOAL = "TASK_GOAL"
    WORLD_STATE = "WORLD_STATE"
    MOTOR_GOAL = "MOTOR_GOAL"
    ACTION_CANDIDATES = "ACTION_CANDIDATES"
    BODY_PREDICTION = "BODY_PREDICTION"
    WORLD_PREDICTION = "WORLD_PREDICTION"
    MEMORY_QUERY = "MEMORY_QUERY"
    MEMORY_RESULT = "MEMORY_RESULT"
    EXECUTION_RESULT = "EXECUTION_RESULT"


class SchemaError(ValueError):
    pass


def validate_message(message: "BusMessage", *, bus_dim: int = 512) -> None:
    """Raise ``SchemaError`` if a bus packet violates its typed contract."""

    header = message.header
    if not header.source or not header.destination:
        raise SchemaError("source and destination are required")
    if not math.isfinite(header.timestamp):
        raise SchemaError("timestamp must be finite")
    if header.sequence_id < 0:
        raise SchemaError("sequence_id must be non-negative")
    if not 0.0 <= header.confidence <= 1.0:
        raise SchemaError("confidence must be between zero and one")
    if not header.correlation_id:
        raise SchemaError("correlation_id is required")

    tensors = message.tensors
    validators = {
        MessageType.TASK_GOAL: _task_goal,
        MessageType.WORLD_STATE: lambda values: _world_state(values, bus_dim),
        MessageType.MOTOR_GOAL: _motor_goal,
        MessageType.ACTION_CANDIDATES: _action_candidates,
        MessageType.BODY_PREDICTION: _body_prediction,
        MessageType.WORLD_PREDICTION: lambda values: _world_prediction(values, bus_dim),
        MessageType.MEMORY_QUERY: lambda values: _memory_query(values, bus_dim),
        MessageType.MEMORY_RESULT: lambda values: _memory_result(values, bus_dim),
        MessageType.EXECUTION_RESULT: _execution_result,
    }
    validators[header.message_type](tensors)
    for name, tensor in tensors.items():
        if not isinstance(tensor, Tensor):
            raise SchemaError(f"{name} must be a torch.Tensor")
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise SchemaError(f"{name} contains non-finite values")


def _require(tensors: dict[str, Tensor], name: str, rank: int) -> Tensor:
    if name not in tensors:
        raise SchemaError(f"missing required tensor: {name}")
    tensor = tensors[name]
    if tensor.ndim != rank:
        raise SchemaError(f"{name} must have rank {rank}, got shape {tuple(tensor.shape)}")
    return tensor


def _last_dim(tensor: Tensor, expected: int, name: str) -> None:
    if tensor.shape[-1] != expected:
        raise SchemaError(f"{name} last dimension must be {expected}, got {tensor.shape[-1]}")


def _same_batch(*tensors: Tensor) -> None:
    batches = {tensor.shape[0] for tensor in tensors}
    if len(batches) != 1:
        raise SchemaError(f"payload batch dimensions disagree: {sorted(batches)}")


def _task_goal(tensors: dict[str, Tensor]) -> None:
    semantic = _require(tensors, "semantic_token", 2)
    _require(tensors, "intent_id", 1)
    _same_batch(semantic, tensors["intent_id"])


def _world_state(tensors: dict[str, Tensor], bus_dim: int) -> None:
    world = _require(tensors, "world_tokens", 3)
    _last_dim(world, bus_dim, "world_tokens")
    if "robot_state" in tensors:
        _last_dim(_require(tensors, "robot_state", 2), 18, "robot_state")


def _motor_goal(tensors: dict[str, Tensor]) -> None:
    required = [
        _require(tensors, "q_goal", 2),
        _require(tensors, "qdot_goal", 2),
        _require(tensors, "duration", 2),
        _require(tensors, "constraints", 2),
    ]
    _last_dim(required[0], 6, "q_goal")
    _last_dim(required[1], 6, "qdot_goal")
    _last_dim(required[2], 1, "duration")
    _same_batch(*required)


def _action_candidates(tensors: dict[str, Tensor]) -> None:
    actions = _require(tensors, "actions", 4)
    _last_dim(actions, 6, "actions")


def _body_prediction(tensors: dict[str, Tensor]) -> None:
    mean = _require(tensors, "mean", 4)
    log_variance = _require(tensors, "log_variance", 4)
    _last_dim(mean, 12, "mean")
    if mean.shape != log_variance.shape:
        raise SchemaError("body mean and log_variance shapes must match")


def _world_prediction(tensors: dict[str, Tensor], bus_dim: int) -> None:
    future = _require(tensors, "future_tokens", 4)
    events = _require(tensors, "event_logits", 3)
    _last_dim(future, bus_dim, "future_tokens")
    _last_dim(events, 3, "event_logits")
    _same_batch(future, events)


def _memory_query(tensors: dict[str, Tensor], bus_dim: int) -> None:
    query = _require(tensors, "query", 2)
    _last_dim(query, bus_dim, "query")


def _memory_result(tensors: dict[str, Tensor], bus_dim: int) -> None:
    entries = _require(tensors, "entries", 3)
    scores = _require(tensors, "scores", 2)
    _last_dim(entries, bus_dim, "entries")
    if entries.shape[:2] != scores.shape:
        raise SchemaError("memory entries and scores must agree on [B, M]")


def _execution_result(tensors: dict[str, Tensor]) -> None:
    state = _require(tensors, "actual_state", 2)
    residual = _require(tensors, "prediction_residual", 2)
    _last_dim(state, 12, "actual_state")
    _last_dim(residual, 12, "prediction_residual")
    _same_batch(state, residual)
