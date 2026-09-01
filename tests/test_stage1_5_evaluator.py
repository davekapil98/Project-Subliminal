import copy
import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def _evaluator():
    path = ROOT / "scripts/evaluate_stage1_5_droid_visual.py"
    spec = importlib.util.spec_from_file_location("stage1_5_evaluator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_source_local_sampler_matches_batches_after_branch_clone() -> None:
    evaluator = _evaluator()
    common = evaluator.SourceBatchSampler(13)
    common.sample(evaluator.PROJECT_IRA, 720, 32)
    common.sample(evaluator.ARMNETBENCH, 400, 32)
    baseline = common.clone()
    treatment = common.clone()

    assert np.array_equal(
        baseline.sample(evaluator.PROJECT_IRA, 720, 32),
        treatment.sample(evaluator.PROJECT_IRA, 720, 32),
    )
    treatment.sample(evaluator.DROID, 2304, 32)
    assert np.array_equal(
        baseline.sample(evaluator.ARMNETBENCH, 400, 32),
        treatment.sample(evaluator.ARMNETBENCH, 400, 32),
    )


def test_gate_aggregation_requires_droid_value_and_bounded_forgetting() -> None:
    evaluator = _evaluator()
    passing = [
        {
            "droid_test_relative_improvement": improvement,
            "droid_validation_relative_improvement": improvement / 2,
            "so101_relative_forgetting": {
                evaluator.PROJECT_IRA: project,
                evaluator.ARMNETBENCH: arm,
            },
        }
        for improvement, project, arm in (
            (0.2, 0.03, 0.05),
            (0.1, 0.11, 0.04),
            (-0.01, 0.02, 0.12),
        )
    ]
    result = evaluator.aggregate_gate(
        passing,
        minimum_positive_seed_count=2,
        forgetting_relative_tolerance=0.10,
    )
    assert result["passed"] is True
    assert result["positive_droid_test_seed_count"] == 2
    assert result["median_droid_test_relative_improvement"] == 0.1

    failing = copy.deepcopy(passing)
    failing[1]["so101_relative_forgetting"][evaluator.PROJECT_IRA] = 0.20
    failing[2]["so101_relative_forgetting"][evaluator.PROJECT_IRA] = 0.12
    result = evaluator.aggregate_gate(
        failing,
        minimum_positive_seed_count=2,
        forgetting_relative_tolerance=0.10,
    )
    assert result["passed"] is False
    assert result["criteria"]["project_ira_forgetting_within_tolerance"] is False


def test_visual_gate_model_is_action_free_and_branch_hashes_match() -> None:
    evaluator = _evaluator()
    model_config = {
        "image_size": 64,
        "patch_size": 8,
        "max_views": 3,
        "proprio_dim": 27,
        "d_model": 32,
        "depth": 1,
        "num_heads": 4,
        "world_tokens": 4,
        "bus_dim": 16,
        "predictor_depth": 1,
    }
    torch.manual_seed(13)
    common = evaluator.VisualJEPAGateModel(model_config)
    baseline = copy.deepcopy(common)
    treatment = copy.deepcopy(common)
    assert (
        evaluator.model_state_sha256(common)
        == evaluator.model_state_sha256(baseline)
        == evaluator.model_state_sha256(treatment)
    )

    prediction = common(
        torch.zeros(2, 3, 3, 64, 64),
        torch.zeros(2, 27),
        torch.ones(2, 3, dtype=torch.bool),
    )
    assert prediction.shape == (2, 4, 16)
    assert "action" not in evaluator.VisualJEPAGateModel.forward.__annotations__
