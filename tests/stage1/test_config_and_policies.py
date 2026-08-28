from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sim.stage1.config import load_stage1_config, validate_run_id
from sim.stage1.policies import RandomizedActuation, SafeJointPolicy
from sim.stage1.worlds import WORLD_PROFILES


ROOT = Path(__file__).resolve().parents[2]


def test_production_plan_is_pinned_and_prioritized() -> None:
    config = load_stage1_config(ROOT / "configs/simulation/stage1_gb100.toml")
    assert config.runtime.isaac_lab_version == "2.3.2"
    assert config.runtime.workshop_commit == "ce807d99724cb65671abec01f908a2fcb4a6eab7"
    assert config.runtime.physics_hz == 240
    assert config.runtime.control_hz == config.camera.fps == 30
    assert config.runtime.decimation == 8
    assert config.total_episodes == 12_288
    assert config.total_transitions == 1_548_288
    assert [job.priority for job in config.jobs] == sorted(job.priority for job in config.jobs)
    assert all(job.record_rgb for job in config.jobs)
    assert {job.world for job in config.jobs} == set(WORLD_PROFILES)


def test_run_id_rejects_path_traversal() -> None:
    assert validate_run_id("stage1_20260828T120000Z") == "stage1_20260828T120000Z"
    for value in ("", "../escape", "has/slash", ".hidden"):
        with pytest.raises(ValueError):
            validate_run_id(value)


@pytest.mark.parametrize("name", ["smooth_random", "motor_sweep", "task_attempt", "failure_recovery"])
def test_policy_stays_inside_limits_and_is_deterministic(name: str) -> None:
    limits = np.repeat(np.array([[-1.5, 1.5]], dtype=np.float32), 6, axis=0)
    default = np.zeros(6, dtype=np.float32)
    kwargs = dict(
        name=name,
        joint_limits=limits,
        default_position=default,
        num_envs=4,
        steps=90,
        control_hz=30,
        seed=77,
    )
    first = SafeJointPolicy(**kwargs)
    second = SafeJointPolicy(**kwargs)
    q = np.zeros((4, 6), dtype=np.float32)
    for step in range(90):
        a = first.step(q, step)
        b = second.step(q, step)
        np.testing.assert_array_equal(a, b)
        assert np.all(a >= -1.5) and np.all(a <= 1.5)
        q = a


def test_actuation_delay_deadband_is_seeded() -> None:
    initial = np.zeros((3, 6), dtype=np.float32)
    first = RandomizedActuation(initial, (0, 3), 0.002, seed=91)
    second = RandomizedActuation(initial, (0, 3), 0.002, seed=91)
    requested = np.full((3, 6), 0.2, dtype=np.float32)
    for _ in range(5):
        a, delay_a = first.apply(requested)
        b, delay_b = second.apply(requested)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(delay_a, delay_b)
