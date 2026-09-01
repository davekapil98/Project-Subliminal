import hashlib
import json
import math
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/training/stage2a_jepa_real_bootstrap.toml"
GATE_PATH = ROOT / "data/manifests/stage1_5_droid_visual.value_gate.json"
EVIDENCE_PATH = (
    ROOT / "data/manifests/stage2a_jepa_real_bootstrap.preflight.json"
)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_stage2a_preflight_is_bound_to_the_frozen_commit_and_inputs() -> None:
    evidence = _json(EVIDENCE_PATH)
    assert evidence["kind"] == "nondecision_engineering_preflight"
    assert evidence["git_commit"] == "3e40124efbefb0d371fd7f7dd46c088220f0c20b"
    assert evidence["config"]["sha256"] == _sha256(CONFIG_PATH)
    assert evidence["admission_evidence"]["sha256"] == _sha256(GATE_PATH)
    assert evidence["admission_evidence"]["admitted_use"] == (
        "jepa_encoder_action_free_temporal_pretraining"
    )
    assert len(evidence["cache_manifest"]["sha256"]) == 64


def test_stage2a_preflight_passes_without_promoting_or_opening_locked_tests() -> None:
    evidence = _json(EVIDENCE_PATH)
    assert evidence["passed"] is True
    assert all(evidence["criteria"].values())
    assert evidence["long_run_authorized"] is False
    assert evidence["locked_test_splits_accessed"] == []
    assert evidence["source_optimizer_steps"] == {
        "armnetbench_so101_v01": 3,
        "droid_raw_1_0_1": 3,
        "project_ira_so101_v1": 3,
    }
    assert evidence["camera_views_dropped"] > 0
    assert evidence["checkpoint"]["format_version"] == 2
    assert evidence["checkpoint"]["resumed_at_optimizer_step"] == 4
    assert evidence["checkpoint"]["final_optimizer_step"] == 9
    assert evidence["checkpoint"]["artifact_paths_committed"] is False


def test_stage2a_preflight_metrics_are_finite_noncollapsed_and_improve() -> None:
    evidence = _json(EVIDENCE_PATH)
    first = evidence["training"]["before_checkpoint"]["loss_first_last"][0]
    final = evidence["training"]["after_resume"]["loss_first_last"][-1]
    assert final < first
    validation = evidence["droid_validation"]
    assert all(math.isfinite(value) for value in validation.values())
    assert validation["target_feature_std"] > 0.0
    assert validation["prediction_feature_std"] > 0.0
    assert validation["sensor_drop_relative_degradation"] < 0.10
    full = evidence["parameter_counts"]["rtx3060_benchmark"]
    assert 240_000_000 <= full["encoder"] <= 280_000_000
    assert full["trainable"] == full["encoder"] + full["predictor"]


def test_stage2a_checkpoints_caches_and_secrets_are_git_ignored() -> None:
    paths = (
        "artifacts/runs/jepa_encoder/stage2a_real_bootstrap/preflight/checkpoints/last.pt",
        "data/cache/stage1_5_visual/droid_raw_1_0_1.train.npz",
        ".env",
    )
    for path in paths:
        result = subprocess.run(
            ("git", "-C", str(ROOT), "check-ignore", "--quiet", path),
            check=False,
        )
        assert result.returncode == 0, path
