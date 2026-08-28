"""Canonical multi-embodiment episode contract from master spec v1.3."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import torch
from torch import Tensor


VALID_DOMAINS = frozenset({"sim", "real"})


def _finite_vector(name: str, value: Tensor) -> None:
    if value.ndim != 1 or value.numel() < 1:
        raise ValueError(f"{name} must be a non-empty one-dimensional tensor")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")


@dataclass(frozen=True)
class EpisodeMetadata:
    episode_id: str
    source_dataset: str
    source_version: str
    source_url: str
    license: str
    redistribution_terms: str
    domain: str
    robot_id: str
    embodiment: str
    task: str
    success: bool | None
    quality: float
    collection_method: str
    native_action_semantics: str
    source_policy: str | None = None
    simulator_family: str | None = None
    simulator_version: str | None = None
    fps: float | None = None
    camera_names: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        required = {
            "episode_id": self.episode_id,
            "source_dataset": self.source_dataset,
            "source_version": self.source_version,
            "source_url": self.source_url,
            "license": self.license,
            "redistribution_terms": self.redistribution_terms,
            "robot_id": self.robot_id,
            "embodiment": self.embodiment,
            "task": self.task,
            "collection_method": self.collection_method,
            "native_action_semantics": self.native_action_semantics,
        }
        missing = [name for name, value in required.items() if not value.strip()]
        if missing:
            raise ValueError(f"required metadata is empty: {', '.join(missing)}")
        if self.domain not in VALID_DOMAINS:
            raise ValueError("domain must be explicitly 'sim' or 'real'")
        if self.domain == "sim" and not self.simulator_family:
            raise ValueError("sim episodes require simulator_family")
        if not 0.0 <= self.quality <= 1.0:
            raise ValueError("quality must be in [0, 1]")
        if self.fps is not None and (not math.isfinite(self.fps) or self.fps <= 0):
            raise ValueError("fps must be finite and positive when provided")
        if len(set(self.camera_names)) != len(self.camera_names):
            raise ValueError("camera_names must be unique")

    @property
    def failure(self) -> bool | None:
        return None if self.success is None else not self.success


@dataclass(frozen=True)
class Observation:
    timestamp: float
    q: Tensor
    qdot: Tensor
    previous_command: Tensor
    rgb: dict[str, Tensor] = field(default_factory=dict)
    imu: Tensor | None = None
    actuator_telemetry: Tensor | None = None
    validity: dict[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(self.timestamp):
            raise ValueError("observation timestamp must be finite")
        for name, value in (
            ("q", self.q),
            ("qdot", self.qdot),
            ("previous_command", self.previous_command),
        ):
            _finite_vector(name, value)
        if not (self.q.shape == self.qdot.shape == self.previous_command.shape):
            raise ValueError("q, qdot and previous_command must use the same embodiment width")
        if self.imu is not None:
            _finite_vector("imu", self.imu)
        for name, image in self.rgb.items():
            if not name or image.ndim != 3 or image.shape[0] not in {1, 3, 4}:
                raise ValueError("RGB entries require a camera name and [C,H,W] tensor")


@dataclass(frozen=True)
class Action:
    timestamp: float
    native: Tensor
    task_space: Tensor | None = None
    task_space_frame: str | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.timestamp):
            raise ValueError("action timestamp must be finite")
        _finite_vector("native action", self.native)
        if self.task_space is not None:
            _finite_vector("task-space action", self.task_space)
            if not self.task_space_frame:
                raise ValueError("task_space_frame is required with task_space action")
        elif self.task_space_frame is not None:
            raise ValueError("task_space_frame cannot be set without task_space action")


@dataclass(frozen=True)
class CanonicalEpisode:
    metadata: EpisodeMetadata
    observations: tuple[Observation, ...]
    actions: tuple[Action, ...]
    language: tuple[str, ...] = ()
    scene_metadata: tuple[dict[str, Any], ...] = ()

    def validate(self) -> None:
        if len(self.observations) != len(self.actions) + 1:
            raise ValueError("an episode requires one more observation than action")
        if not self.actions:
            raise ValueError("an episode requires at least one transition")
        timestamps = [observation.timestamp for observation in self.observations]
        if timestamps != sorted(timestamps) or len(set(timestamps)) != len(timestamps):
            raise ValueError("observation timestamps must be strictly increasing")
        state_width = self.observations[0].q.numel()
        if any(observation.q.numel() != state_width for observation in self.observations):
            raise ValueError("observation embodiment width cannot change within an episode")
        for index, action in enumerate(self.actions):
            if not timestamps[index] <= action.timestamp <= timestamps[index + 1]:
                raise ValueError("each action timestamp must bracket its transition")
        if self.scene_metadata and len(self.scene_metadata) not in {1, len(self.observations)}:
            raise ValueError("scene_metadata must be episode-level or aligned to observations")
        if any(not item.strip() for item in self.language):
            raise ValueError("language annotations cannot be empty strings")
