import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/training/stage1_5_droid_visual.toml"
PROTOCOL_PATH = ROOT / "configs/training/stage1_5_droid_visual_protocol_v3.toml"
OBJECTS_PATH = (
    ROOT / "configs/datasets/registry/stage1_5_visual_subset_v2.objects.json"
)
EVIDENCE_PATH = ROOT / "data/manifests/stage1_5_droid_visual.value_gate.json"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_stage1_5_gate_passes_every_frozen_criterion() -> None:
    report = _json(EVIDENCE_PATH)
    assert report["gate"] == "stage1.5_action_free_visual_jepa_value_and_forgetting"
    assert report["protocol_revision"] == 3
    assert report["configuration"]["seeds"] == [13, 29, 47]
    assert report["configuration"]["common_pretraining_steps"] == 300
    assert report["configuration"]["matched_comparison_steps_per_branch"] == 240
    assert report["configuration"]["batch_size"] == 32

    aggregate = report["aggregate"]
    assert aggregate["passed"] is True
    assert all(aggregate["criteria"].values())
    assert aggregate["positive_droid_test_seed_count"] == 3
    assert aggregate["median_droid_test_relative_improvement"] > 0.07
    assert aggregate["median_droid_validation_relative_improvement"] > 0.07
    assert all(
        value <= report["configuration"]["forgetting_relative_tolerance"]
        for value in aggregate["median_so101_relative_forgetting"].values()
    )
    assert report["visual_value_gate_passed"] is True
    assert report["admission_decision"] == "admitted"
    assert report["admitted_uses"] == [
        "jepa_encoder_action_free_temporal_pretraining"
    ]
    assert report["admission_blockers"] == []


def test_stage1_5_evidence_preserves_matched_branch_and_action_boundary() -> None:
    report = _json(EVIDENCE_PATH)
    assert len(report["per_seed"]) == 3
    for row in report["per_seed"]:
        assert len(row["branch_start_checkpoint_sha256"]) == 64
        assert row["droid_test_relative_improvement"] > 0.0
        assert row["common_training"]["source_updates"] == {
            "armnetbench_so101_v01": 150,
            "project_ira_so101_v1": 150,
        }
        assert row["baseline_training"]["source_updates"] == {
            "armnetbench_so101_v01": 120,
            "project_ira_so101_v1": 120,
        }
        assert row["treatment_training"]["source_updates"] == {
            "armnetbench_so101_v01": 80,
            "droid_raw_1_0_1": 80,
            "project_ira_so101_v1": 80,
        }
        assert row["final_model_sha256"]["baseline"] != (
            row["final_model_sha256"]["treatment"]
        )

    excluded = set(report["still_excluded_uses"])
    assert "SO-101 motor targets" in excluded
    assert "SO-101 body dynamics" in excluded
    assert "cross-embodiment native-action mixing" in excluded
    assert "action-conditioned JEPA World admission" in excluded
    assert report["input_evidence"]["privacy"] == {
        "action_fields_included": False,
        "episode_identifiers_recorded": False,
        "raw_metadata_values_recorded": False,
    }
    caches = report["input_evidence"]["caches"]
    assert len(caches) == 7
    assert sum(entry["samples"] for entry in caches.values()) == 4672
    assert all(entry["contains_action_fields"] is False for entry in caches.values())
    assert all(entry["maximum_horizon_seconds"] <= 0.6 + 1e-9 for entry in caches.values())


def test_stage1_5_evidence_is_bound_to_frozen_reviewable_inputs() -> None:
    evidence = _json(EVIDENCE_PATH)["input_evidence"]
    assert evidence["base_config"]["sha256"] == _sha256(CONFIG_PATH)
    assert evidence["active_protocol"]["sha256"] == _sha256(PROTOCOL_PATH)
    assert evidence["object_manifest"]["sha256"] == _sha256(OBJECTS_PATH)
    assert all(
        len(entry["sha256"]) == 64 for entry in evidence["caches"].values()
    )
