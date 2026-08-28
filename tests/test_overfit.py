"""Stage 0 gate: every tiny neural module must memorize a fixed tiny batch."""

from collections.abc import Callable

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from models.body_dynamics import TinyBodyDynamics
from models.executive import TinyExecutive
from models.jepa_encoder import JEPALatentPredictor, TinyJEPAEncoder
from models.jepa_world import TinyJEPAWorldPredictor
from models.language_speech import ByteTokenizer, TinyLanguageSpeech
from models.memory import TinyMemoryController
from models.motor_cortex import TinyMotorCortex
from models.orchestrator import TinyOrchestrator


def assert_overfits(
    model: nn.Module,
    objective: Callable[[], Tensor],
    *,
    steps: int = 60,
    required_fraction: float = 0.45,
) -> tuple[float, float]:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.015, weight_decay=0.0)
    with torch.no_grad():
        initial = float(objective())
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = objective()
        assert torch.isfinite(loss)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    with torch.no_grad():
        final = float(objective())
    assert final < initial * required_fraction, (initial, final)
    return initial, final


def test_all_eight_stage0_modules_overfit_tiny_batches() -> None:
    torch.manual_seed(11)

    encoder = TinyJEPAEncoder(
        image_size=8,
        patch_size=4,
        max_views=1,
        d_model=8,
        depth=1,
        num_heads=2,
        world_tokens=2,
        bus_dim=8,
    )
    predictor = JEPALatentPredictor(bus_dim=8, d_model=8, depth=1, num_heads=2)
    jepa = nn.ModuleDict({"encoder": encoder, "predictor": predictor})
    images = torch.randn(1, 1, 3, 8, 8)
    proprio = torch.randn(1, 18)
    jepa_target = torch.randn(1, 2, 8)

    def jepa_objective() -> Tensor:
        world = encoder(images, proprio).world_tokens
        predicted = predictor(world, torch.ones(1, 2, dtype=torch.bool))
        return functional.mse_loss(predicted, jepa_target)

    assert_overfits(jepa, jepa_objective)

    world_model = TinyJEPAWorldPredictor(bus_dim=8, d_model=8, depth=1, num_heads=2)
    world_tokens = torch.randn(1, 2, 8)
    action_candidates = torch.randn(1, 1, 2, 6) * 0.02
    world_target = torch.randn(1, 1, 2, 8)
    assert_overfits(
        world_model,
        lambda: functional.mse_loss(
            world_model(world_tokens, action_candidates).future_tokens, world_target
        ),
    )

    motor = TinyMotorCortex(horizon=2, d_model=8, depth=1, num_heads=2)
    motor_state = torch.randn(1, 18)
    motor_goal = torch.randn(1, 15)
    noisy_actions = torch.randn(1, 2, 6)
    flow_time = torch.tensor([0.4])
    velocity_target = torch.randn(1, 2, 6)
    assert_overfits(
        motor,
        lambda: functional.mse_loss(
            motor(motor_state, motor_goal, noisy_actions, flow_time), velocity_target
        ),
    )

    body = TinyBodyDynamics(d_model=8, depth=1, num_heads=2)
    state = torch.randn(1, 12)
    body_actions = torch.randn(1, 1, 2, 6) * 0.01
    with torch.no_grad():
        body_target = body(state, body_actions).mean + 0.15
    assert_overfits(
        body,
        lambda: functional.mse_loss(body(state, body_actions).mean, body_target),
    )

    executive = TinyExecutive(bus_dim=8, d_model=8, depth=1, num_heads=2)
    task = torch.randn(1, 8)
    scene = torch.randn(1, 2, 8)
    robot_state = torch.zeros(1, 12)
    executive_target = torch.full((1, 6), 0.2)
    assert_overfits(
        executive,
        lambda: functional.mse_loss(
            executive(task, scene, robot_state).q_goal, executive_target
        ),
    )

    language = TinyLanguageSpeech(
        bus_dim=8,
        d_model=8,
        semantic_depth=1,
        conformer_depth=1,
        num_heads=2,
    )
    text_ids, text_valid = ByteTokenizer().encode(["pick red ball"])
    semantic_target = torch.randn(1, 8)

    def language_objective() -> Tensor:
        output = language(text_ids=text_ids, text_valid=text_valid)
        return functional.cross_entropy(output.intent_logits, torch.tensor([3])) + functional.mse_loss(
            output.semantic_token, semantic_target
        )

    assert_overfits(language, language_objective)

    memory = TinyMemoryController(bus_dim=8, d_model=8, top_k=2)
    query = torch.randn(1, 8)
    entries = torch.randn(1, 3, 8)
    recency = torch.rand(1, 3)
    confidence = torch.rand(1, 3)
    assert_overfits(
        memory,
        lambda: functional.cross_entropy(
            memory(query, entries, recency, confidence).scores, torch.tensor([1])
        ),
    )

    orchestrator = TinyOrchestrator(bus_dim=8, hidden_dim=8)
    route_token = torch.randn(1, 8)
    route_features = torch.randn(1, 3)
    assert_overfits(
        orchestrator,
        lambda: functional.cross_entropy(
            orchestrator(route_token, route_features), torch.tensor([5])
        ),
    )
