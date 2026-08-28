from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sim.stage1.config import load_stage1_config
from sim.stage1.runner import collect_vector_episode
from sim.stage1.validator import verify_run_directory
from sim.stage1.worlds import WORLD_PROFILES
from sim.stage1.writer import RunWriter, verify_episode_directory


ROOT = Path(__file__).resolve().parents[2]


class FakeBackend:
    num_envs = 2
    joint_limits = np.repeat(np.array([[-1.5, 1.5]], dtype=np.float32), 6, axis=0)
    default_joint_position = np.zeros(6, dtype=np.float32)

    def __init__(self) -> None:
        self.q = np.zeros((self.num_envs, 6), dtype=np.float32)
        self.qdot = self.q.copy()
        self.index = 0

    def reset(self, seed: int) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        self.q = rng.uniform(-0.05, 0.05, size=(self.num_envs, 6)).astype(np.float32)
        self.qdot.fill(0)
        self.index = 0
        return self._frame()

    def step(self, action: np.ndarray) -> dict[str, np.ndarray]:
        previous = self.q.copy()
        self.q += 0.25 * (action - self.q)
        self.qdot = (self.q - previous) * 30.0
        self.index += 1
        return self._frame()

    def _frame(self) -> dict[str, np.ndarray]:
        n = self.num_envs
        rgb = np.zeros((n, 16, 20, 3), dtype=np.uint8)
        rgb[..., 0] = 20 + self.index
        rgb[1, ..., 1] = 100
        poses = np.zeros((n, 4, 7), dtype=np.float32)
        poses[..., 3] = 1.0
        contacts = np.zeros((n, 3), dtype=np.float32)
        contacts[0, 0] = float(self.index)
        return {
            "joint_position_true_rad": self.q.copy(),
            "joint_velocity_true_rad_s": self.qdot.copy(),
            "applied_joint_torque_nm": np.abs(self.qdot) * 0.1,
            "imu_linear_acceleration_m_s2": np.full((n, 3), 9.81, dtype=np.float32),
            "imu_angular_velocity_rad_s": self.qdot[:, :3].copy(),
            "object_pose_env_wxyz": poses,
            "contact_force_by_vial_n": contacts,
            "grasp_contact_bool": contacts.max(axis=1) > 2.0,
            "collision_bool": np.zeros(n, dtype=bool),
            "success_bool": np.array([self.index >= 3, False]),
            "hard_limit_bool": np.zeros(n, dtype=bool),
            "rear_rgb": rgb,
            "segmentation": np.full((n, 16, 20), self.index, dtype=np.uint32),
        }


def _batch():
    return collect_vector_episode(
        FakeBackend(),
        "smooth_random",
        WORLD_PROFILES["motor_bench"],
        steps=5,
        control_hz=30,
        seed=123,
        record_segmentation=True,
    )


def test_vector_runner_has_canonical_t_plus_one_contract() -> None:
    first = _batch()
    second = _batch()
    assert first.rear_rgb.shape == (6, 2, 16, 20, 3)
    assert first.segmentation is not None and first.segmentation.shape == (6, 2, 16, 20)
    assert first.observations["joint_position_rad"].shape == (6, 2, 6)
    assert first.observations["imu_linear_acceleration_m_s2"].shape == (6, 2, 3)
    assert first.observations["object_pose_env_wxyz"].shape == (6, 2, 4, 7)
    assert first.actions["action_native_joint_position_rad"].shape == (5, 2, 6)
    for key in first.observations:
        np.testing.assert_array_equal(first.observations[key], second.observations[key])
    for key in first.actions:
        np.testing.assert_array_equal(first.actions[key], second.actions[key])


def test_writer_commits_resumes_and_detects_tampering(tmp_path: Path) -> None:
    config = load_stage1_config(ROOT / "configs/simulation/stage1_smoke.toml")
    config = replace(
        config,
        camera=replace(config.camera, video_backend="npz"),
        dataset=replace(config.dataset, min_free_disk_gib=0.001, max_output_gib=1.0),
    )
    batch = _batch()
    payloads = [
        batch.episode(index, f"smoke_e{index:06d}", {"task": "test", "success": bool(index == 0)})
        for index in range(2)
    ]
    writer = RunWriter(tmp_path, "test_run", config, {"test": True})
    assert writer.write_batch(payloads) == {"written": 2, "skipped": 0}
    writer.finalize("complete")

    report = verify_run_directory(tmp_path / "test_run")
    assert report["episodes"] == 2
    resumed = RunWriter(tmp_path, "test_run", config, {"test": True})
    assert resumed.committed_count == 2
    assert resumed.write_batch(payloads) == {"written": 0, "skipped": 2}

    telemetry = tmp_path / "test_run/episodes/smoke_e000000/telemetry.npz"
    with telemetry.open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="size|SHA-256"):
        verify_episode_directory(telemetry.parent)
