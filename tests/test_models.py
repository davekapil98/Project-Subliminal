import torch

from models.body_dynamics import TinyBodyDynamics
from models.executive import TinyExecutive
from models.jepa_encoder import EMATargetEncoder, JEPALatentPredictor, TinyJEPAEncoder
from models.jepa_world import TinyJEPAWorldPredictor
from models.language_speech import ByteTokenizer, TinyLanguageSpeech
from models.memory import TinyMemoryController
from models.motor_cortex import TinyMotorCortex, flow_matching_batch, flow_matching_loss
from models.orchestrator import TinyOrchestrator


def assert_finite_gradient(model: torch.nn.Module) -> None:
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_jepa_encoder_predictor_and_ema_target() -> None:
    encoder = TinyJEPAEncoder(
        image_size=16,
        patch_size=8,
        max_views=2,
        d_model=16,
        depth=1,
        num_heads=4,
        world_tokens=2,
        bus_dim=8,
    )
    images = torch.randn(2, 2, 3, 16, 16)
    proprio = torch.randn(2, 18)
    output = encoder(images, proprio, camera_valid=torch.tensor([[1, 1], [1, 0]]))
    assert output.world_tokens.shape == (2, 2, 8)
    assert output.pose.shape == (2, 2, 7)
    (output.world_tokens.square().mean() + output.object_logits.square().mean() + output.pose.square().mean()).backward()
    assert_finite_gradient(encoder)

    predictor = JEPALatentPredictor(bus_dim=8, d_model=16, depth=1, num_heads=4)
    prediction = predictor(output.world_tokens.detach(), torch.tensor([[0, 1], [1, 0]], dtype=torch.bool))
    assert prediction.shape == output.world_tokens.shape
    prediction.square().mean().backward()
    assert_finite_gradient(predictor)
    target = EMATargetEncoder(encoder)
    target.update(encoder, momentum=0.9)
    assert not any(parameter.requires_grad for parameter in target.parameters())


def test_world_motor_and_body_models_have_typed_finite_outputs() -> None:
    world_tokens = torch.randn(2, 3, 8)
    actions = torch.randn(2, 2, 3, 6) * 0.01
    world = TinyJEPAWorldPredictor(
        bus_dim=8, d_model=16, depth=1, num_heads=4
    )
    world_output = world(world_tokens, actions)
    assert world_output.future_tokens.shape == (2, 2, 3, 8)
    assert world_output.event_logits.shape == (2, 2, 3)
    (world_output.future_tokens.square().mean() + world_output.event_logits.square().mean() + world_output.log_variance.square().mean()).backward()
    assert_finite_gradient(world)

    motor = TinyMotorCortex(horizon=3, d_model=16, depth=1, num_heads=4)
    body_state = torch.randn(2, 18)
    goal = torch.randn(2, 15)
    target_actions = torch.randn(2, 3, 6) * 0.02
    noisy, flow_time, target_velocity = flow_matching_batch(target_actions)
    velocity = motor(body_state, goal, noisy, flow_time)
    assert velocity.shape == target_actions.shape
    flow_matching_loss(velocity, target_velocity).backward()
    assert_finite_gradient(motor)
    sampled = motor.sample(body_state, goal, candidates=2, steps=2)
    assert sampled.actions.shape == (2, 2, 3, 6)
    assert sampled.confidence.shape == (2, 2)

    dynamics = TinyBodyDynamics(d_model=16, depth=1, num_heads=4)
    body_output = dynamics(body_state[:, :12], actions)
    assert body_output.mean.shape == (2, 2, 3, 12)
    assert body_output.log_variance.shape == body_output.mean.shape
    (body_output.mean.square().mean() + body_output.log_variance.square().mean()).backward()
    assert_finite_gradient(dynamics)


def test_executive_language_memory_and_orchestrator_outputs() -> None:
    language = TinyLanguageSpeech(
        bus_dim=8,
        d_model=16,
        semantic_depth=1,
        conformer_depth=1,
        num_heads=4,
    )
    tokenizer = ByteTokenizer()
    text_ids, text_valid = tokenizer.encode(["pick red ball", "move left"])
    language_output = language(text_ids=text_ids, text_valid=text_valid)
    assert language_output.semantic_token.shape == (2, 8)
    language_loss = (
        language_output.semantic_token.square().mean()
        + language_output.intent_logits.square().mean()
        + language_output.entity_logits.square().mean()
        + language_output.attribute_logits.square().mean()
    )
    language_loss.backward()
    assert_finite_gradient(language)

    executive = TinyExecutive(bus_dim=8, d_model=16, depth=1, num_heads=4)
    robot_state = torch.randn(2, 12)
    executive_output = executive(
        language_output.semantic_token.detach(),
        torch.randn(2, 3, 8),
        robot_state,
        memory_tokens=torch.randn(2, 2, 8),
        body_predictions=torch.randn(2, 2, 3, 12),
        world_event_logits=torch.randn(2, 2, 3),
    )
    assert executive_output.motor_goal_tensor().shape == (2, 15)
    executive_loss = (
        executive_output.motor_goal_tensor().square().mean()
        + executive_output.decision_logits.square().mean()
        + executive_output.next_stage_latent.square().mean()
    )
    executive_loss.backward()
    assert_finite_gradient(executive)

    memory = TinyMemoryController(bus_dim=8, d_model=16, top_k=2)
    memory_output = memory(
        torch.randn(2, 8),
        torch.randn(2, 4, 8),
        torch.rand(2, 4),
        torch.rand(2, 4),
    )
    assert memory_output.retrieved_entries.shape == (2, 2, 8)
    (memory_output.scores.square().mean() + memory_output.compressed.square().mean() + memory_output.write_logits.square().mean()).backward()
    assert_finite_gradient(memory)

    orchestrator = TinyOrchestrator(bus_dim=8, hidden_dim=16)
    route_logits = orchestrator(torch.randn(2, 8), torch.randn(2, 3))
    assert route_logits.shape == (2, 8)
    route_logits.square().mean().backward()
    assert_finite_gradient(orchestrator)
