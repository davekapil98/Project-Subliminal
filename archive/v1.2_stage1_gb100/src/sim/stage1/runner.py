"""Backend-independent vector rollout collection and sensor perturbation."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .policies import RandomizedActuation, SafeJointPolicy
from .records import ACTION_KEYS, OBSERVATION_KEYS, VectorEpisodeBatch
from .worlds import WorldProfile


class VectorBackend(Protocol):
    num_envs: int
    joint_limits: np.ndarray
    default_joint_position: np.ndarray

    def reset(self, seed: int) -> dict[str, np.ndarray]: ...

    def step(self, action: np.ndarray) -> dict[str, np.ndarray]: ...


def _ensure_rgb(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    if value.ndim != 4 or value.shape[-1] < 3:
        raise ValueError(f"rear camera must return [N,H,W,C], got {value.shape}")
    if np.issubdtype(value.dtype, np.floating):
        scale = 255.0 if value.max(initial=0.0) <= 1.0 else 1.0
        value = np.clip(value * scale, 0, 255)
    return np.ascontiguousarray(value[..., :3].astype(np.uint8))


def collect_vector_episode(
    backend: VectorBackend,
    policy_name: str,
    profile: WorldProfile,
    steps: int,
    control_hz: int,
    seed: int,
    record_segmentation: bool,
) -> VectorEpisodeBatch:
    """Collect a fixed-length, synchronized vector episode batch."""

    rng = np.random.default_rng(seed + 101)
    raw = backend.reset(seed)
    num_envs = backend.num_envs
    policy = SafeJointPolicy(
        policy_name,
        backend.joint_limits,
        backend.default_joint_position,
        num_envs,
        steps,
        control_hz,
        seed + 211,
    )
    initial_command = np.asarray(raw["joint_position_true_rad"], dtype=np.float32)
    actuation = RandomizedActuation(
        initial_command,
        profile.action_delay_steps,
        profile.action_deadband_rad,
        seed + 307,
    )
    accel_bias = rng.normal(0.0, profile.imu_accel_noise_std_m_s2 * 0.5, size=(num_envs, 3))
    gyro_bias = rng.normal(0.0, profile.imu_gyro_noise_std_rad_s * 0.5, size=(num_envs, 3))

    observations: dict[str, list[np.ndarray]] = {key: [] for key in OBSERVATION_KEYS}
    actions: dict[str, list[np.ndarray]] = {key: [] for key in ACTION_KEYS}
    rgb_frames: list[np.ndarray] = []
    segmentation_frames: list[np.ndarray] | None = [] if record_segmentation else None

    def append_observation(frame: dict[str, np.ndarray], previous_command: np.ndarray, index: int) -> None:
        q_true = np.asarray(frame["joint_position_true_rad"], dtype=np.float32)
        qdot_true = np.asarray(frame["joint_velocity_true_rad_s"], dtype=np.float32)
        accel_true = np.asarray(frame["imu_linear_acceleration_m_s2"], dtype=np.float32)
        gyro_true = np.asarray(frame["imu_angular_velocity_rad_s"], dtype=np.float32)
        sim_time = np.full(num_envs, index / float(control_hz), dtype=np.float64)
        jitter = rng.uniform(-profile.timestamp_jitter_s, profile.timestamp_jitter_s, size=num_envs)
        sensor_time = sim_time + jitter
        if index == 0:
            sensor_time = np.maximum(sensor_time, 0.0)

        observations["timestamp_sim_s"].append(sim_time)
        observations["timestamp_sensor_s"].append(sensor_time)
        observations["joint_position_true_rad"].append(q_true)
        observations["joint_velocity_true_rad_s"].append(qdot_true)
        observations["joint_position_rad"].append(
            q_true + rng.normal(0.0, profile.encoder_noise_std_rad, size=q_true.shape).astype(np.float32)
        )
        observations["joint_velocity_rad_s"].append(
            qdot_true + rng.normal(0.0, profile.velocity_noise_std_rad_s, size=qdot_true.shape).astype(np.float32)
        )
        observations["previous_command_rad"].append(np.asarray(previous_command, dtype=np.float32))
        observations["imu_linear_acceleration_m_s2"].append(
            accel_true
            + accel_bias.astype(np.float32)
            + rng.normal(0.0, profile.imu_accel_noise_std_m_s2, size=accel_true.shape).astype(np.float32)
        )
        observations["imu_angular_velocity_rad_s"].append(
            gyro_true
            + gyro_bias.astype(np.float32)
            + rng.normal(0.0, profile.imu_gyro_noise_std_rad_s, size=gyro_true.shape).astype(np.float32)
        )
        for key in (
            "applied_joint_torque_nm",
            "object_pose_env_wxyz",
            "contact_force_by_vial_n",
            "grasp_contact_bool",
            "collision_bool",
            "success_bool",
            "hard_limit_bool",
        ):
            observations[key].append(np.asarray(frame[key]))
        rgb_frames.append(_ensure_rgb(frame["rear_rgb"]))
        if segmentation_frames is not None:
            if "segmentation" not in frame:
                raise ValueError("segmentation recording was requested but the backend did not provide it")
            segmentation_frames.append(np.asarray(frame["segmentation"]))

    previous_command = initial_command.copy()
    append_observation(raw, previous_command, 0)
    for step in range(steps):
        observed_q = observations["joint_position_rad"][-1]
        requested = policy.step(observed_q, step)
        applied, delays = actuation.apply(requested)
        actions["action_requested_joint_position_rad"].append(requested)
        actions["action_native_joint_position_rad"].append(applied)
        actions["action_relative_joint_rad"].append(applied - observed_q)
        actions["actuation_delay_steps"].append(delays)
        raw = backend.step(applied)
        previous_command = applied
        append_observation(raw, previous_command, step + 1)

    observation_arrays = {key: np.stack(value, axis=0) for key, value in observations.items()}
    action_arrays = {key: np.stack(value, axis=0) for key, value in actions.items()}
    batch = VectorEpisodeBatch(
        observations=observation_arrays,
        actions=action_arrays,
        rear_rgb=np.stack(rgb_frames, axis=0),
        segmentation=None if segmentation_frames is None else np.stack(segmentation_frames, axis=0),
    )
    return batch
