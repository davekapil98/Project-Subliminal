from pathlib import Path
import tomllib

import torch
import torch.nn.functional as functional

from bus import BusMessage
from models.jepa_encoder import TinyJEPAEncoder
from models.language_speech import TinyLanguageSpeech
from sim import Stage0RobotBrain, Stage0SystemConfig
from training.checkpointing import load_checkpoint, save_checkpoint


ROOT = Path(__file__).resolve().parents[1]


def compact_config(*, bus_dim: int) -> Stage0SystemConfig:
    return Stage0SystemConfig(
        bus_dim=bus_dim,
        d_model=16,
        num_heads=4,
        image_size=16,
        world_tokens=2,
        horizon=2,
        candidates=2,
        flow_steps=1,
        execute_prefix=1,
    )


def test_invalid_camera_content_cannot_change_jepa_world_tokens() -> None:
    torch.manual_seed(19)
    encoder = TinyJEPAEncoder(
        image_size=16,
        patch_size=8,
        max_views=2,
        d_model=16,
        depth=2,
        num_heads=4,
        world_tokens=2,
        bus_dim=8,
    ).eval()
    first = torch.randn(1, 2, 3, 16, 16)
    second = first.clone()
    second[:, 1] = torch.randn_like(second[:, 1]) * 1000
    valid = torch.tensor([[True, False]])
    proprio = torch.randn(1, 18)
    with torch.no_grad():
        before = encoder(first, proprio, camera_valid=valid).world_tokens
        after = encoder(second, proprio, camera_valid=valid).world_tokens
    torch.testing.assert_close(before, after, atol=1e-6, rtol=1e-6)


def test_toy_audio_to_intent_path_is_finite_and_trainable() -> None:
    model = TinyLanguageSpeech(
        mel_bins=80,
        bus_dim=8,
        d_model=16,
        semantic_depth=1,
        conformer_depth=1,
        num_heads=4,
    )
    log_mel = torch.randn(2, 6, 80)
    valid = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    output = model(log_mel=log_mel, audio_valid=valid)
    loss = functional.cross_entropy(output.intent_logits, torch.tensor([1, 2]))
    loss.backward()
    assert output.semantic_token.shape == (2, 8)
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.conformer.parameters())


def test_full_mock_loop_accepts_the_fixed_512d_bus() -> None:
    result = Stage0RobotBrain(compact_config(bus_dim=512)).step("pick up the red ball")
    world_message = BusMessage.from_json(result.messages["world_state"])
    assert world_message.tensors["world_tokens"].shape[-1] == 512
    assert result.route_logits.shape == (1, 8)
    assert torch.isfinite(result.final_state).all()


def test_all_eight_module_state_dicts_round_trip_together(tmp_path: Path) -> None:
    model = Stage0RobotBrain(compact_config(bus_dim=8))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "all-eight.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        step=3,
        config={"bus_dim": 8},
        dataset_manifest={"dataset_id": "synthetic-stage0"},
        normalization={"q_mean": [0.0] * 6},
        precision="fp32",
        git_commit="test",
    )
    restored = Stage0RobotBrain(compact_config(bus_dim=8))
    load_checkpoint(path, model=restored)
    for name, expected in model.state_dict().items():
        torch.testing.assert_close(expected, restored.state_dict()[name])


def test_v13_budget_is_exactly_1024m_and_stage0_uses_512d_adapter_target() -> None:
    with (ROOT / "configs/models/v1_3_budget.toml").open("rb") as handle:
        budget = tomllib.load(handle)
    modules = [value for key, value in budget.items() if key != "system"]
    assert sum(module["parameters_m"] for module in modules) == 1024
    assert budget["system"]["total_parameters_m"] == 1024
    assert budget["system"]["bus_dim"] == 512
    with (ROOT / "configs/models/stage0_tiny.toml").open("rb") as handle:
        tiny = tomllib.load(handle)
    assert tiny["common"]["bus_dim"] == 512
