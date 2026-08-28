"""Simulation package with lazy Stage 0 exports.

Keeping the package initializer dependency-free lets host-side Stage 1 config
checks run before the pinned Isaac container (and its PyTorch install) exists.
"""

from __future__ import annotations

from typing import Any


__all__ = ["MockSO101", "Stage0RobotBrain", "Stage0StepResult", "Stage0SystemConfig"]


def __getattr__(name: str) -> Any:
    if name == "MockSO101":
        from sim.mock_robot import MockSO101

        return MockSO101
    if name in {"Stage0RobotBrain", "Stage0StepResult", "Stage0SystemConfig"}:
        from sim.stage0_system import Stage0RobotBrain, Stage0StepResult, Stage0SystemConfig

        return {
            "Stage0RobotBrain": Stage0RobotBrain,
            "Stage0StepResult": Stage0StepResult,
            "Stage0SystemConfig": Stage0SystemConfig,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
