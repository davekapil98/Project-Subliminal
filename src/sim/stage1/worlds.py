"""Named, reproducible SO-101 world and randomization profiles."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class WorldProfile:
    name: str
    description: str
    rack_pose_range: dict[str, tuple[float, float]]
    vial_pose_range: dict[str, tuple[float, float]]
    rack_placement_probability: float
    robot_start_offset_rad: float
    light_exposure_range: tuple[float, float]
    camera_position_jitter_m: float
    camera_rotation_jitter_rad: float
    actuator_gain_scale: tuple[float, float]
    joint_friction_scale: tuple[float, float]
    object_mass_scale: tuple[float, float]
    object_friction_range: tuple[float, float]
    action_delay_steps: tuple[int, int]
    action_deadband_rad: float
    encoder_noise_std_rad: float
    velocity_noise_std_rad_s: float
    imu_accel_noise_std_m_s2: float
    imu_gyro_noise_std_rad_s: float
    timestamp_jitter_s: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


WORLD_PROFILES: dict[str, WorldProfile] = {
    "motor_bench": WorldProfile(
        name="motor_bench",
        description="Centered rack, small start perturbations, and mild dynamics variation for Body/Motor data.",
        rack_pose_range={"x": (-0.01, 0.01), "y": (-0.005, 0.005), "yaw": (-0.12, 0.12)},
        vial_pose_range={"x": (-0.015, 0.015), "y": (-0.008, 0.008), "roll": (-0.08, 0.08), "yaw": (0.0, 0.0)},
        rack_placement_probability=0.0,
        robot_start_offset_rad=0.04,
        light_exposure_range=(-0.5, 0.5),
        camera_position_jitter_m=0.003,
        camera_rotation_jitter_rad=0.01,
        actuator_gain_scale=(0.92, 1.08),
        joint_friction_scale=(0.85, 1.15),
        object_mass_scale=(0.9, 1.1),
        object_friction_range=(0.65, 0.9),
        action_delay_steps=(0, 1),
        action_deadband_rad=0.001,
        encoder_noise_std_rad=0.001,
        velocity_noise_std_rad_s=0.003,
        imu_accel_noise_std_m_s2=0.03,
        imu_gyro_noise_std_rad_s=0.002,
        timestamp_jitter_s=0.0003,
    ),
    "randomized_objects": WorldProfile(
        name="randomized_objects",
        description="Scattered vials with randomized camera, lighting, mechanics, and actuator response.",
        rack_pose_range={"x": (-0.05, 0.05), "y": (-0.025, 0.025), "yaw": (-0.6, 0.6)},
        vial_pose_range={"x": (-0.055, 0.055), "y": (-0.025, 0.025), "roll": (-0.35, 0.35), "yaw": (-0.6, 0.6)},
        rack_placement_probability=0.2,
        robot_start_offset_rad=0.12,
        light_exposure_range=(-3.0, 1.0),
        camera_position_jitter_m=0.02,
        camera_rotation_jitter_rad=0.05,
        actuator_gain_scale=(0.78, 1.22),
        joint_friction_scale=(0.6, 1.45),
        object_mass_scale=(0.65, 1.4),
        object_friction_range=(0.35, 1.25),
        action_delay_steps=(0, 2),
        action_deadband_rad=0.0025,
        encoder_noise_std_rad=0.0025,
        velocity_noise_std_rad_s=0.008,
        imu_accel_noise_std_m_s2=0.08,
        imu_gyro_noise_std_rad_s=0.006,
        timestamp_jitter_s=0.0008,
    ),
    "rack_task": WorldProfile(
        name="rack_task",
        description="Vials-to-rack task distribution with broad scene and sensor randomization.",
        rack_pose_range={"x": (-0.04, 0.04), "y": (-0.015, 0.015), "yaw": (-0.5, 0.5)},
        vial_pose_range={"x": (-0.045, 0.045), "y": (-0.02, 0.02), "roll": (-0.3, 0.3), "yaw": (-0.2, 0.2)},
        rack_placement_probability=0.1,
        robot_start_offset_rad=0.08,
        light_exposure_range=(-3.0, 1.0),
        camera_position_jitter_m=0.018,
        camera_rotation_jitter_rad=0.045,
        actuator_gain_scale=(0.8, 1.2),
        joint_friction_scale=(0.65, 1.35),
        object_mass_scale=(0.7, 1.3),
        object_friction_range=(0.4, 1.15),
        action_delay_steps=(0, 2),
        action_deadband_rad=0.002,
        encoder_noise_std_rad=0.002,
        velocity_noise_std_rad_s=0.006,
        imu_accel_noise_std_m_s2=0.06,
        imu_gyro_noise_std_rad_s=0.004,
        timestamp_jitter_s=0.0006,
    ),
    "recovery_states": WorldProfile(
        name="recovery_states",
        description="Partially completed and disturbed rack scenes for failures and recovery trajectories.",
        rack_pose_range={"x": (-0.055, 0.055), "y": (-0.025, 0.025), "yaw": (-0.7, 0.7)},
        vial_pose_range={"x": (-0.06, 0.06), "y": (-0.03, 0.03), "roll": (-0.5, 0.5), "yaw": (-0.8, 0.8)},
        rack_placement_probability=0.8,
        robot_start_offset_rad=0.18,
        light_exposure_range=(-3.5, 1.5),
        camera_position_jitter_m=0.022,
        camera_rotation_jitter_rad=0.06,
        actuator_gain_scale=(0.72, 1.25),
        joint_friction_scale=(0.55, 1.55),
        object_mass_scale=(0.6, 1.5),
        object_friction_range=(0.3, 1.3),
        action_delay_steps=(1, 3),
        action_deadband_rad=0.003,
        encoder_noise_std_rad=0.003,
        velocity_noise_std_rad_s=0.01,
        imu_accel_noise_std_m_s2=0.1,
        imu_gyro_noise_std_rad_s=0.008,
        timestamp_jitter_s=0.001,
    ),
}


OBJECT_NAMES = ("vial_1", "vial_2", "vial_3", "rack_left")
SO101_JOINT_NAMES = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw")
