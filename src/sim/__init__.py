"""Simulation-independent Stage 0 mock environment exports.

The production baseline is public-dataset-first. This package contains only the
small deterministic plant used to prove integration and safety ordering.
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
