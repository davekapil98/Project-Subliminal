"""In-memory canonical episode containers used by Isaac and offline tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


OBSERVATION_KEYS = (
    "timestamp_sim_s",
    "timestamp_sensor_s",
    "joint_position_rad",
    "joint_position_true_rad",
    "joint_velocity_rad_s",
    "joint_velocity_true_rad_s",
    "previous_command_rad",
    "imu_linear_acceleration_m_s2",
    "imu_angular_velocity_rad_s",
    "applied_joint_torque_nm",
    "object_pose_env_wxyz",
    "contact_force_by_vial_n",
    "grasp_contact_bool",
    "collision_bool",
    "success_bool",
    "hard_limit_bool",
)

ACTION_KEYS = (
    "action_requested_joint_position_rad",
    "action_native_joint_position_rad",
    "action_relative_joint_rad",
    "actuation_delay_steps",
)


@dataclass(frozen=True)
class EpisodePayload:
    """One immutable canonical episode ready for atomic serialization."""

    episode_id: str
    metadata: dict[str, Any]
    arrays: dict[str, np.ndarray]
    rear_rgb: np.ndarray
    segmentation: np.ndarray | None = None

    def validate(self) -> None:
        missing = sorted(set(OBSERVATION_KEYS + ACTION_KEYS) - set(self.arrays))
        if missing:
            raise ValueError(f"episode {self.episode_id} is missing arrays: {', '.join(missing)}")
        steps = int(self.arrays["action_native_joint_position_rad"].shape[0])
        observations = steps + 1
        for key in OBSERVATION_KEYS:
            if self.arrays[key].shape[0] != observations:
                raise ValueError(f"{key} must have T+1={observations} samples")
        for key in ACTION_KEYS:
            if self.arrays[key].shape[0] != steps:
                raise ValueError(f"{key} must have T={steps} samples")
        for key in (
            "joint_position_rad",
            "joint_position_true_rad",
            "joint_velocity_rad_s",
            "joint_velocity_true_rad_s",
            "previous_command_rad",
            "applied_joint_torque_nm",
            "action_requested_joint_position_rad",
            "action_native_joint_position_rad",
            "action_relative_joint_rad",
        ):
            if self.arrays[key].shape[-1] != 6:
                raise ValueError(f"{key} must end in six SO-101 joints")
        for key in ("imu_linear_acceleration_m_s2", "imu_angular_velocity_rad_s"):
            if self.arrays[key].shape[-1] != 3:
                raise ValueError(f"{key} must end in XYZ")
        if self.arrays["object_pose_env_wxyz"].shape[-1] != 7:
            raise ValueError("object_pose_env_wxyz must end in XYZ + WXYZ")
        if self.rear_rgb.shape[0] != observations or self.rear_rgb.shape[-1] != 3:
            raise ValueError("rear_rgb must have shape [T+1, H, W, 3]")
        if self.rear_rgb.dtype != np.uint8:
            raise ValueError("rear_rgb must use uint8 pixels")
        if self.segmentation is not None and self.segmentation.shape[0] != observations:
            raise ValueError("segmentation must align with observation frames")
        timestamps = self.arrays["timestamp_sim_s"]
        if timestamps.ndim != 1 or np.any(np.diff(timestamps) <= 0):
            raise ValueError("timestamp_sim_s must be one-dimensional and strictly increasing")
        if not np.isfinite(timestamps).all():
            raise ValueError("timestamps must be finite")


@dataclass(frozen=True)
class VectorEpisodeBatch:
    """A synchronized vector rollout with time-major arrays [time, env, ...]."""

    observations: dict[str, np.ndarray]
    actions: dict[str, np.ndarray]
    rear_rgb: np.ndarray
    segmentation: np.ndarray | None

    @property
    def num_envs(self) -> int:
        return int(self.rear_rgb.shape[1])

    @property
    def steps(self) -> int:
        return int(self.actions["action_native_joint_position_rad"].shape[0])

    def episode(self, env_index: int, episode_id: str, metadata: dict[str, Any]) -> EpisodePayload:
        if not 0 <= env_index < self.num_envs:
            raise IndexError(env_index)
        arrays = {key: np.ascontiguousarray(value[:, env_index]) for key, value in self.observations.items()}
        arrays.update({key: np.ascontiguousarray(value[:, env_index]) for key, value in self.actions.items()})
        payload = EpisodePayload(
            episode_id=episode_id,
            metadata=metadata,
            arrays=arrays,
            rear_rgb=np.ascontiguousarray(self.rear_rgb[:, env_index]),
            segmentation=(
                None if self.segmentation is None else np.ascontiguousarray(self.segmentation[:, env_index])
            ),
        )
        payload.validate()
        return payload
