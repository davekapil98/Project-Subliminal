import torch

from bus import BusMessage
from sim import Stage0RobotBrain


def test_complete_mocked_bus_loop_is_safe_and_serializable() -> None:
    brain = Stage0RobotBrain()
    result = brain.step("pick up the red ball")
    assert set(result.messages) == {
        "task_goal",
        "world_state",
        "motor_goal",
        "action_candidates",
        "body_prediction",
        "world_prediction",
        "execution_result",
    }
    for encoded in result.messages.values():
        decoded = BusMessage.from_json(encoded)
        assert decoded.header.correlation_id
    assert result.executed_actions.shape == (1, 2, 6)
    assert result.executed_actions.abs().max() <= 0.05 + 1e-7
    assert torch.isfinite(result.final_state).all()
    assert torch.isfinite(result.prediction_residual).all()
    assert result.memory_entries == 1
    assert result.route_logits.shape == (1, 8)


def test_seeded_stage0_cycles_are_deterministic() -> None:
    first = Stage0RobotBrain().step("reach the blue cube")
    second = Stage0RobotBrain().step("reach the blue cube")
    torch.testing.assert_close(first.candidate_costs, second.candidate_costs)
    torch.testing.assert_close(first.executed_actions, second.executed_actions)
    torch.testing.assert_close(first.final_state, second.final_state)
