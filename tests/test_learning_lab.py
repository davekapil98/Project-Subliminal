import torch

from learn_lab.attention_from_scratch import LearningAttention
from learn_lab.flow_matching_toy import ToyVectorField, make_flow_example
from learn_lab.gaussian_dynamics_toy import ToyGaussianDynamics
from learn_lab.jepa_toy import ToyJEPA
from learn_lab.rmsnorm_from_scratch import LearningRMSNorm
from learn_lab.transformer_from_scratch import LearningTransformerBlock


def test_learning_implementations_are_independent_and_differentiable() -> None:
    tokens = torch.randn(2, 4, 8, requires_grad=True)
    normalized = LearningRMSNorm(8)(tokens)
    attended, weights = LearningAttention(8)(normalized, causal=True)
    transformed = LearningTransformerBlock(8)(attended)
    assert weights.shape == (2, 4, 4)

    jepa = ToyJEPA(input_width=8, latent_width=8)
    jepa_loss = jepa.loss(torch.randn(2, 8), torch.randn(2, 8))
    jepa.update_target()
    flow = ToyVectorField(width=6, hidden=8)
    point, target_velocity = make_flow_example(
        torch.ones(2, 6), torch.zeros(2, 6), torch.tensor([0.25, 0.75])
    )
    flow_loss = (flow(point, torch.tensor([0.25, 0.75])) - target_velocity).square().mean()
    dynamics = ToyGaussianDynamics()
    mean, log_variance = dynamics(torch.randn(2, 12), torch.randn(2, 6))
    loss = transformed.square().mean() + jepa_loss + flow_loss + mean.square().mean() + log_variance.square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert tokens.grad is not None
