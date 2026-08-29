import json
from pathlib import Path
import subprocess
import tomllib


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "configs/datasets/registry/project_ira_so101_v1.toml"
MANIFEST_PATH = ROOT / "data/manifests/project_ira_so101_v1.json"
QUALIFICATION_PATH = ROOT / "data/manifests/project_ira_so101_v1.qualification.json"
SPLIT_PATH = ROOT / "data/splits/project_ira_so101_v1.json"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_qualification_evidence_matches_registry_and_manifest() -> None:
    with REGISTRY_PATH.open("rb") as handle:
        registry = tomllib.load(handle)
    manifest = _json(MANIFEST_PATH)
    report = _json(QUALIFICATION_PATH)

    assert report["passed"] is True
    assert report["gate"] == "stage1.1_source_qualification"
    assert report["source_revision"] == registry["dataset"]["revision"] == manifest["revision"]
    assert report["source_status"] == manifest["status"] == "validated"
    assert report["admission_decision"] == "validated_for_stage1; not_yet_admitted_to_training"
    assert report["qualified_subset_bytes"] == registry["qualified_subset"]["local_subset_bytes"]
    assert report["source_validation"]["episodes"] == 930
    assert report["source_validation"]["frames"] == 844208
    assert report["source_validation"]["tasks"] == 93
    assert report["source_validation"]["null_values"] == 0
    assert report["source_validation"]["nonfinite_values"] == 0
    assert report["canonical_sample"]["decoded_rgb_frames"] == 6
    assert set(report["video_validation"]) == {"desk_view", "wrist_left"}

    registry_files = {
        item["path"]: (item["size"], item["sha256"]) for item in registry["qualified_files"]
    }
    report_files = {
        item["path"]: (item["size"], item["sha256"]) for item in report["verified_files"]
    }
    assert report_files == registry_files
    assert manifest["checksum"] == f"sha256:{registry['dataset']['primary_trajectory_sha256']}"


def test_prompt_group_split_is_complete_disjoint_and_leakage_resistant() -> None:
    split = _json(SPLIT_PATH)
    groups = split["task_indices"]
    train = set(groups["train"])
    validation = set(groups["validation"])
    test = set(groups["test"])
    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)
    assert train | validation | test == set(range(93))
    assert split["episode_counts"] == {"train": 750, "validation": 90, "test": 90}
    assert sum(split["episode_counts"].values()) == 930
    assert split["group_key"] == "task_index"
    family_counts = split["task_family_prompt_counts"]
    expected_families = {"desk_cleanup", "dice_throw", "fetch_ball", "sort_lego_color"}
    assert set(family_counts) == expected_families
    assert all(
        counts["validation"] > 0 and counts["test"] > 0 for counts in family_counts.values()
    )
    for family, (first, last) in split["task_family_ranges_inclusive"].items():
        for split_name, task_indices in groups.items():
            actual = sum(first <= task_index <= last for task_index in task_indices)
            assert actual == family_counts[family][split_name]


def test_qualified_raw_and_cleaned_samples_are_git_ignored() -> None:
    raw_card = "data/raw/public_real/project_ira_so101/revision/README.md"
    raw_trajectory = "data/raw/public_real/project_ira_so101/revision/data/file.parquet"
    cleaned_sample = "data/cleaned/public_real/project_ira_so101/revision/sample.json"
    for path in (raw_card, raw_trajectory, cleaned_sample):
        result = subprocess.run(
            ("git", "-C", str(ROOT), "check-ignore", "--quiet", path),
            check=False,
        )
        assert result.returncode == 0, path
