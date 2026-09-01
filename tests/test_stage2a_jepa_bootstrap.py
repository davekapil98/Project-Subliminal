import hashlib
import inspect
import json
import runpy
from pathlib import Path
import tomllib

import numpy as np
import pytest
import torch

from models.jepa_encoder import (
    ActionFreeTemporalJEPA,
    JEPALatentPredictor,
    MultimodalJEPAEncoder,
)
from training.losses.jepa import jepa_latent_loss


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/training/stage2a_jepa_real_bootstrap.toml"
GATE_PATH = ROOT / "data/manifests/stage1_5_droid_visual.value_gate.json"
OBJECTS_PATH = (
    ROOT / "configs/datasets/registry/stage1_5_visual_subset_v2.objects.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config() -> dict:
    with CONFIG_PATH.open("rb") as handle:
        return tomllib.load(handle)


def _driver() -> dict:
    return runpy.run_path(str(ROOT / "scripts/train_stage2a_jepa_real_bootstrap.py"))


def test_stage2a_contract_is_pinned_to_the_narrow_stage1_5_admission() -> None:
    config = _config()
    gate = json.loads(GATE_PATH.read_text(encoding="utf-8"))
    assert config["stage"] == "2A"
    assert config["status"] == "frozen_before_nondecision_preflight"
    assert config["admission"]["evidence_sha256"] == _sha256(GATE_PATH)
    assert config["admission"]["required_use"] == gate["admitted_uses"][0]
    assert set(config["admission"]["still_excluded_uses"]) == set(
        gate["still_excluded_uses"]
    )
    assert config["data"]["object_manifest_sha256"] == _sha256(OBJECTS_PATH)
    assert config["data"]["action_fields_allowed"] is False
    assert config["data"]["source_weights"] == [1.0, 1.0, 1.0]
    assert config["long_run"] == {
        "authorized": False,
        "optimizer_steps": 0,
        "reason": config["long_run"]["reason"],
    }
    assert config["evaluation"]["during_training_split"].endswith(":validation")
    assert len(config["evaluation"]["locked_until_final_evaluation"]) == 3


def test_full_profile_matches_master_spec_scale_without_allocating_weights() -> None:
    config = _config()
    counts = _driver()["parameter_counts"](
        config["profiles"]["rtx3060_benchmark"]
    )
    assert 240_000_000 <= counts["encoder"] <= 280_000_000
    assert counts["trainable"] == counts["encoder"] + counts["predictor"]
    assert counts["checkpoint_total"] == (
        counts["trainable"] + counts["ema_target"]
    )
    training = config["profiles"]["rtx3060_benchmark"]["training"]
    assert training["microbatch_size"] == 1
    assert training["precision"] == "fp16"
    assert training["minimum_gpu_memory_bytes"] == 11_000_000_000


def test_multimodal_temporal_jepa_is_action_free_masked_and_differentiable() -> None:
    torch.manual_seed(7)
    encoder = MultimodalJEPAEncoder(
        image_size=16,
        patch_size=8,
        max_views=3,
        proprio_dim=27,
        d_model=32,
        visual_depth=1,
        fusion_depth=1,
        num_heads=4,
        world_tokens=2,
        bus_dim=16,
        activation_checkpointing=True,
    )
    model = ActionFreeTemporalJEPA(
        encoder,
        JEPALatentPredictor(bus_dim=16, d_model=32, depth=1, num_heads=4),
    ).train()
    assert model.target_encoder.training is False
    assert not any(value.requires_grad for value in model.target_encoder.parameters())
    assert "action" not in inspect.signature(model.forward).parameters

    context = torch.randn(2, 3, 3, 16, 16)
    future = torch.randn(2, 3, 3, 16, 16)
    proprio = torch.randn(2, 27)
    camera_valid = torch.tensor([[True, False, True], [True, True, True]])
    output = model(context, future, proprio, camera_valid)
    assert output.predicted_tokens.shape == output.target_tokens.shape == (2, 2, 16)
    loss = jepa_latent_loss(output.predicted_tokens, output.target_tokens)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(value.grad is not None for value in model.encoder.parameters())
    assert all(value.grad is None for value in model.target_encoder.parameters())

    with pytest.raises(ValueError, match="at least one valid camera"):
        model.encoder(context, proprio, camera_valid=torch.zeros(2, 3, dtype=torch.bool))


def test_balanced_sampler_resume_and_camera_dropout_are_exact_and_safe() -> None:
    driver = _driver()
    sampler = driver["BalancedSourceSampler"](2026)
    sampler.sample(driver["DROID"], 2304, 5)
    valid = np.asarray([[True, True, True], [True, False, True]], dtype=np.bool_)
    dropped, removed = sampler.camera_dropout(valid, 0.99)
    assert removed >= 1
    assert dropped.any(axis=1).all()
    assert not dropped[~valid].any()

    state = sampler.state_dict()
    expected_indices = sampler.sample(driver["PROJECT_IRA"], 720, 8)
    expected_mask, expected_removed = sampler.camera_dropout(valid, 0.15)
    resumed = driver["BalancedSourceSampler"](999)
    resumed.load_state_dict(state)
    assert np.array_equal(
        resumed.sample(driver["PROJECT_IRA"], 720, 8), expected_indices
    )
    actual_mask, actual_removed = resumed.camera_dropout(valid, 0.15)
    assert np.array_equal(actual_mask, expected_mask)
    assert actual_removed == expected_removed
