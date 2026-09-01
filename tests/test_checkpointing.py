from pathlib import Path

import torch

from models.executive import TinyExecutive
from training.checkpointing import (
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
from training.precision import resolve_precision


def make_executive() -> TinyExecutive:
    return TinyExecutive(bus_dim=8, d_model=16, depth=1, num_heads=4)


def test_checkpoint_round_trip_includes_reproducibility_metadata(tmp_path: Path) -> None:
    torch.manual_seed(3)
    model = make_executive()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        step=12,
        config={"bus_dim": 8},
        dataset_manifest={"name": "synthetic"},
        normalization={"q_mean": [0.0] * 6},
        precision="fp32",
        git_commit="test-commit",
        training_state={"sampler_step": 7, "rng": capture_rng_state()},
    )
    restored = make_executive()
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    metadata = load_checkpoint(
        checkpoint_path, model=restored, optimizer=restored_optimizer
    )
    assert metadata["step"] == 12
    assert metadata["format_version"] == 2
    assert metadata["dataset_manifest"]["name"] == "synthetic"
    assert metadata["git_commit"] == "test-commit"
    assert metadata["training_state"]["sampler_step"] == 7
    assert not list(tmp_path.glob("*.part"))
    for expected, actual in zip(model.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(expected, actual)


def test_checkpoint_rng_state_restores_the_next_random_draw() -> None:
    torch.manual_seed(91)
    state = capture_rng_state()
    expected = torch.rand(5)
    torch.manual_seed(1234)
    restore_rng_state(state)
    torch.testing.assert_close(torch.rand(5), expected)


def test_precision_auto_falls_back_safely_on_cpu() -> None:
    policy = resolve_precision(device="cpu", precision="auto")
    assert policy.device.type == "cpu"
    assert policy.parameter_dtype == torch.float32
    assert policy.autocast_dtype is None
    with policy.autocast():
        result = torch.ones(2) + 1
    assert result.dtype == torch.float32
