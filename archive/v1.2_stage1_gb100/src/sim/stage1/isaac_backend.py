"""Concrete vector backend that exposes exact Isaac scene state to the recorder."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
import sim_to_real_so101.tasks  # noqa: F401

from .worlds import OBJECT_NAMES, SO101_JOINT_NAMES


TASK_ID = "Lerobot-So101-Teleop-Vials-To-Rack-DR"


def _numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _rgb_uint8(value: np.ndarray) -> np.ndarray:
    rgb = np.asarray(value)[..., :3]
    if np.issubdtype(rgb.dtype, np.floating):
        rgb = rgb * (255.0 if rgb.max(initial=0.0) <= 1.0 else 1.0)
    return np.clip(rgb, 0, 255).astype(np.uint8)


class IsaacSO101Backend:
    def __init__(self, env_cfg: Any, record_segmentation: bool) -> None:
        self.env = gym.make(TASK_ID, cfg=env_cfg)
        self.unwrapped = self.env.unwrapped
        self.scene = self.unwrapped.scene
        self.device = self.unwrapped.device
        self.num_envs = int(self.unwrapped.num_envs)
        self.record_segmentation = record_segmentation
        self.robot = self.scene["robot"]
        joint_ids, joint_names = self.robot.find_joints(list(SO101_JOINT_NAMES), preserve_order=True)
        if tuple(joint_names) != SO101_JOINT_NAMES:
            raise RuntimeError(f"SO-101 joint order mismatch: {joint_names}")
        self.joint_ids = joint_ids
        limits = _numpy(self.robot.data.soft_joint_pos_limits[0, joint_ids])
        self.joint_limits = limits.astype(np.float32)
        default = _numpy(self.robot.data.default_joint_pos[0, joint_ids])
        self.default_joint_position = default.astype(np.float32)
        self._observation: dict[str, Any] = {}

    def close(self) -> None:
        self.env.close()

    def reset(self, seed: int) -> dict[str, np.ndarray]:
        result = self.env.reset(seed=seed)
        self._observation = result[0] if isinstance(result, tuple) else result
        return self.snapshot()

    def step(self, action: np.ndarray) -> dict[str, np.ndarray]:
        action_tensor = torch.as_tensor(action, device=self.device, dtype=torch.float32)
        self._observation, _reward, _terminated, _truncated, _info = self.env.step(action_tensor)
        return self.snapshot()

    def _subtask(self, name: str) -> np.ndarray:
        group = self._observation.get("subtask_terms")
        if not isinstance(group, dict) or name not in group:
            raise RuntimeError(f"official workshop observation is missing exact subtask label {name!r}")
        value = _numpy(group[name])
        return value.reshape(self.num_envs).astype(bool)

    def _contact_forces(self) -> np.ndarray:
        force_matrix = self.scene["contact_grasp"].data.force_matrix_w
        if force_matrix is None:
            raise RuntimeError("filtered per-vial contact force matrix is unavailable")
        forces = _numpy(force_matrix)
        # Isaac Lab 2.3.2 contract: [env, sensor_body, filtered_vial, xyz].
        magnitudes = np.linalg.norm(forces, axis=-1)
        while magnitudes.ndim > 2:
            magnitudes = magnitudes.max(axis=1)
        if magnitudes.shape != (self.num_envs, 3):
            raise RuntimeError(f"unexpected per-vial contact shape: {magnitudes.shape}")
        return magnitudes.astype(np.float32)

    def _collision(self) -> np.ndarray:
        sensor = self.scene["robot_contacts"]
        forces = _numpy(sensor.data.net_forces_w)
        magnitudes = np.linalg.norm(forces, axis=-1)
        body_names = list(sensor.body_names)
        keep = [
            index
            for index, name in enumerate(body_names)
            if not any(expected in name.lower() for expected in ("base", "jaw", "gripper"))
        ]
        if not keep:
            raise RuntimeError("robot collision sensor did not resolve any non-base arm bodies")
        return (magnitudes[:, keep].max(axis=1) > 1.0).astype(bool)

    def snapshot(self) -> dict[str, np.ndarray]:
        joint_ids = self.joint_ids
        q = _numpy(self.robot.data.joint_pos[:, joint_ids]).astype(np.float32)
        qdot = _numpy(self.robot.data.joint_vel[:, joint_ids]).astype(np.float32)
        torque = _numpy(self.robot.data.applied_torque[:, joint_ids]).astype(np.float32)
        imu = self.scene["gripper_imu"].data
        accel = _numpy(imu.lin_acc_b).reshape(self.num_envs, 3).astype(np.float32)
        gyro = _numpy(imu.ang_vel_b).reshape(self.num_envs, 3).astype(np.float32)

        object_poses = []
        origins = self.scene.env_origins
        for name in OBJECT_NAMES:
            asset = self.scene[name]
            position = asset.data.root_pos_w - origins
            object_poses.append(torch.cat((position, asset.data.root_quat_w), dim=-1))
        pose_array = _numpy(torch.stack(object_poses, dim=1)).astype(np.float32)

        camera_output = self.scene["rear_camera"].data.output
        rgb = _numpy(camera_output["rgb"])
        segmentation = None
        if self.record_segmentation:
            segmentation = _numpy(camera_output["instance_id_segmentation_fast"])
            if segmentation.shape[-1:] == (1,):
                segmentation = segmentation[..., 0]

        contact_forces = self._contact_forces()
        grasp = (contact_forces.max(axis=1) > 2.0) | self._subtask("vial_grasped")
        success = self._subtask("vial_placed")
        low = self.joint_limits[:, 0][None]
        high = self.joint_limits[:, 1][None]
        hard_limit = ((q <= low + 0.015) | (q >= high - 0.015)).any(axis=1)
        result = {
            "joint_position_true_rad": q,
            "joint_velocity_true_rad_s": qdot,
            "applied_joint_torque_nm": torque,
            "imu_linear_acceleration_m_s2": accel,
            "imu_angular_velocity_rad_s": gyro,
            "object_pose_env_wxyz": pose_array,
            "contact_force_by_vial_n": contact_forces,
            "grasp_contact_bool": grasp.astype(bool),
            "collision_bool": self._collision(),
            "success_bool": success.astype(bool),
            "hard_limit_bool": hard_limit.astype(bool),
            "rear_rgb": rgb,
        }
        if segmentation is not None:
            result["segmentation"] = segmentation
        return result

    def contract_report(self) -> dict[str, Any]:
        snapshot = self.snapshot()
        sensor_types = {name: type(sensor).__name__ for name, sensor in self.scene.sensors.items()}
        camera_names = [name for name, class_name in sensor_types.items() if "Camera" in class_name]
        imu_names = [name for name, class_name in sensor_types.items() if "Imu" in class_name]
        if camera_names != ["rear_camera"]:
            raise RuntimeError(f"expected only rear_camera as an RGB sensor, found {camera_names}")
        if imu_names != ["gripper_imu"]:
            raise RuntimeError(f"expected only gripper_imu as an IMU sensor, found {imu_names}")

        rgb = _rgb_uint8(snapshot["rear_rgb"])
        per_env_mean = rgb.mean(axis=(1, 2, 3))
        per_env_std = rgb.std(axis=(1, 2, 3))
        blank_envs = np.flatnonzero(per_env_std < 1.0).tolist()
        if blank_envs:
            raise RuntimeError(f"rear tiled-camera frames are blank/nearly constant in environments {blank_envs}")
        for key in (
            "joint_position_true_rad",
            "imu_linear_acceleration_m_s2",
            "imu_angular_velocity_rad_s",
            "object_pose_env_wxyz",
        ):
            if not np.isfinite(snapshot[key]).all():
                raise RuntimeError(f"non-finite values in required sensor/label tensor {key}")

        segmentation_info = None
        if self.record_segmentation:
            segmentation_info = self.scene["rear_camera"].data.info.get("instance_id_segmentation_fast")
            if not isinstance(segmentation_info, dict) or "idToLabels" not in segmentation_info:
                raise RuntimeError("instance segmentation is missing its idToLabels mapping")
        return {
            "num_envs": self.num_envs,
            "joint_names": list(SO101_JOINT_NAMES),
            "joint_position_shape": list(snapshot["joint_position_true_rad"].shape),
            "rear_camera_shape": list(snapshot["rear_rgb"].shape),
            "rear_camera_dtype": str(snapshot["rear_rgb"].dtype),
            "rear_camera_per_env_pixel_mean": per_env_mean.tolist(),
            "rear_camera_per_env_pixel_std": per_env_std.tolist(),
            "gripper_imu_accel_shape": list(snapshot["imu_linear_acceleration_m_s2"].shape),
            "gripper_imu_gyro_shape": list(snapshot["imu_angular_velocity_rad_s"].shape),
            "object_names": list(OBJECT_NAMES),
            "object_pose_shape": list(snapshot["object_pose_env_wxyz"].shape),
            "contact_force_shape": list(snapshot["contact_force_by_vial_n"].shape),
            "scene_sensor_types": sensor_types,
            "rgb_input_names": camera_names,
            "imu_names": imu_names,
            "rgb_input_count": len(camera_names),
            "imu_count": len(imu_names),
            "segmentation_enabled": self.record_segmentation,
            "segmentation_mapping_present": segmentation_info is not None,
        }
