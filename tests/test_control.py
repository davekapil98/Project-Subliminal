import torch

from control.mpc import choose_candidates, score_action_candidates
from control.safety import SafetyLayer, SafetyLimits


def test_safety_clamps_delta_velocity_and_joint_limits() -> None:
    limits = SafetyLimits.conservative_stage0()
    safety = SafetyLayer(limits, control_hz=30.0)
    current = torch.tensor([[2.49, 0.0, 0.0, 0.0, 0.0, -2.49]])
    unsafe = torch.full((1, 3, 6), 1.0)
    safe = safety.filter_chunk(current, unsafe)
    assert safe.abs().max() <= 0.05 + 1e-7
    positions = current[:, None] + safe.cumsum(dim=1)
    assert (positions <= limits.joint_max + 1e-7).all()
    assert (positions >= limits.joint_min - 1e-7).all()


def test_mpc_prefers_candidate_closer_to_goal() -> None:
    actions = torch.zeros(1, 2, 3, 6)
    body_mean = torch.zeros(1, 2, 3, 12)
    body_mean[:, 0, -1, :6] = 0.1
    body_mean[:, 1, -1, :6] = 1.0
    costs = score_action_candidates(
        actions=actions,
        body_mean=body_mean,
        body_log_variance=torch.full_like(body_mean, -4.0),
        q_goal=torch.zeros(1, 6),
        qdot_goal=torch.zeros(1, 6),
        joint_min=torch.full((6,), -2.5),
        joint_max=torch.full((6,), 2.5),
        collision_probability=torch.zeros(1, 2),
    )
    _, selected = choose_candidates(actions, costs)
    assert selected.item() == 0
