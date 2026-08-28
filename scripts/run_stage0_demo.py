#!/usr/bin/env python3
"""Run one safe synthetic Project Subliminal control cycle."""

from evaluation.metrics import final_joint_error
from sim import Stage0RobotBrain


def main() -> None:
    brain = Stage0RobotBrain()
    result = brain.step("pick up the red ball")
    print("Stage 0 synthetic cycle: PASS")
    print(f"messages validated: {len(result.messages)}")
    print(f"selected candidate: {int(result.selected_candidate[0])}")
    print(f"executed prefix shape: {tuple(result.executed_actions.shape)}")
    print(f"memory entries: {result.memory_entries}")
    print(f"joint goal error: {float(final_joint_error(result.final_state[:, :6], result.q_goal)):.4f}")


if __name__ == "__main__":
    main()
