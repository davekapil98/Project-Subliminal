"""Canonical episode contract shared by future dataset converters."""

from dataclasses import dataclass, field
from typing import Any

from torch import Tensor


@dataclass(frozen=True)
class EpisodeMetadata:
    episode_id: str
    source_dataset: str
    source_version: str
    license: str
    robot_id: str
    embodiment: str
    task: str
    success: bool | None
    quality: float
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError("episode_id is required")
        if not self.source_dataset or not self.source_version or not self.license:
            raise ValueError("dataset source, version, and license are required")
        if not 0.0 <= self.quality <= 1.0:
            raise ValueError("quality must be in [0, 1]")


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
        for name, value in (
            ("q", self.q),
            ("qdot", self.qdot),
            ("previous_command", self.previous_command),
        ):
            if value.shape[-1] != 6:
                raise ValueError(f"{name} must end in six SO-101 joints")


@dataclass(frozen=True)
class Action:
    timestamp: float
    native: Tensor
    task_space: Tensor | None = None

    def __post_init__(self) -> None:
        if self.native.shape[-1] != 6:
            raise ValueError("native SO-101 action must end in six joints")


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
        timestamps = [observation.timestamp for observation in self.observations]
        if timestamps != sorted(timestamps) or len(set(timestamps)) != len(timestamps):
            raise ValueError("observation timestamps must be strictly increasing")
        for index, action in enumerate(self.actions):
            if not timestamps[index] <= action.timestamp <= timestamps[index + 1]:
                raise ValueError("each action timestamp must bracket its transition")
