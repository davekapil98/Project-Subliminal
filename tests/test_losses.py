import torch

from models.motor_cortex.flow import flow_matching_batch
from training.losses import gaussian_nll, jepa_latent_loss


def test_flow_matching_path_endpoints_and_velocity() -> None:
    target = torch.ones(2, 3, 6)
    noise = torch.zeros_like(target)
    at_zero, _, velocity = flow_matching_batch(
        target, noise=noise, flow_time=torch.zeros(2)
    )
    at_one, _, _ = flow_matching_batch(
        target, noise=noise, flow_time=torch.ones(2)
    )
    torch.testing.assert_close(at_zero, noise)
    torch.testing.assert_close(at_one, target)
    torch.testing.assert_close(velocity, target - noise)


def test_jepa_and_gaussian_losses_are_finite_and_differentiable() -> None:
    predicted = torch.randn(2, 3, 4, requires_grad=True)
    target = torch.randn_like(predicted)
    latent_loss = jepa_latent_loss(predicted, target)
    mean = torch.randn(2, 3, 12, requires_grad=True)
    log_variance = torch.zeros_like(mean, requires_grad=True)
    uncertainty_loss = gaussian_nll(mean, log_variance, torch.randn_like(mean))
    total = latent_loss + uncertainty_loss
    total.backward()
    assert torch.isfinite(total)
    assert predicted.grad is not None and mean.grad is not None
