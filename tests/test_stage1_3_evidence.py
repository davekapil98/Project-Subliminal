import hashlib
import json
from pathlib import Path
import subprocess
import tomllib


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "configs/datasets/registry/so101_ma_multitask_700.toml"
OBJECTS_PATH = ROOT / "configs/datasets/registry/so101_ma_multitask_700.objects.json"
MANIFEST_PATH = ROOT / "data/manifests/so101_ma_multitask_700.json"
QUALIFICATION_PATH = ROOT / "data/manifests/so101_ma_multitask_700.qualification.json"
VALUE_GATE_PATH = ROOT / "data/manifests/so101_ma_multitask_700.value_gate.json"
SPLIT_PATH = ROOT / "data/splits/so101_ma_multitask_700.json"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_qualification_matches_registry_objects_and_manifest() -> None:
    with REGISTRY_PATH.open("rb") as handle:
        registry = tomllib.load(handle)
    objects = _json(OBJECTS_PATH)
    manifest = _json(MANIFEST_PATH)
    report = _json(QUALIFICATION_PATH)

    assert report["passed"] is True
    assert report["gate"] == "stage1.3_source_qualification"
    assert report["source_revision"] == registry["dataset"]["revision"] == manifest["revision"]
    assert report["source_status"] == manifest["status"] == "validated"
    assert report["admission_decision"] == "not_admitted"
    assert report["qualified_subset"]["verified_object_count"] == 13
    assert report["qualified_subset"]["verified_bytes"] == 567845886
    assert registry["objects"]["manifest_sha256"] == _sha256(OBJECTS_PATH)
    assert len(objects["objects"]) == 13
    assert sum(item["size"] for item in objects["objects"]) == 567845886
    assert manifest["checksum"] == f"sha256:{_sha256(OBJECTS_PATH)}"
    assert len(report["upstream_sources"]) == 7
    assert all(source["license"] == "Apache-2.0" for source in report["upstream_sources"])


def test_integrity_video_and_canonical_evidence_is_complete() -> None:
    report = _json(QUALIFICATION_PATH)
    source = report["source_validation"]
    assert source["episodes"] == 700
    assert source["frames"] == 358006
    assert source["tasks"] == 7
    assert source["null_values"] == source["nonfinite_values"] == 0
    assert source["task_text_source_column"] == "__index_level_0__"
    assert source["task_text_normalized"] is True
    assert source["published_statistics_trusted"] is False
    assert source["native_action_physical_units_proven"] is False
    assert source["task_space_conversion_allowed"] is False
    assert source["merge_provenance"]["upstream_source_count"] == 7

    video = report["video_validation"]
    codecs = {
        camera["codec"]
        for episode in video.values()
        for camera in episode.values()
    }
    assert codecs == {"h264", "av1"}
    assert all(
        camera["pixel_format"] == "yuv420p"
        for episode in video.values()
        for camera in episode.values()
    )
    assert video["447"]["top"]["unused_frames"] == 9
    assert video["447"]["left_wrist"]["unused_frames"] == 9
    assert video["299"]["top"]["unused_frames"] == 0
    sample = report["canonical_sample"]
    assert sample["episodes"] == 3
    assert sample["observations"] == 9
    assert sample["actions"] == 6
    assert sample["decoded_rgb_frames"] == 18
    assert len(sample["sha256"]) == 64


def test_source_block_split_is_complete_disjoint_and_leakage_resistant() -> None:
    split = _json(SPLIT_PATH)
    episodes = {
        name: set(indices) for name, indices in split["episode_indices"].items()
    }
    assert episodes["train"].isdisjoint(episodes["validation"])
    assert episodes["train"].isdisjoint(episodes["test"])
    assert episodes["validation"].isdisjoint(episodes["test"])
    assert set().union(*episodes.values()) == set(range(700))
    assert split["episode_counts"] == {
        "train": 560,
        "validation": 70,
        "test": 70,
    }
    assert split["frame_counts"] == {
        "train": 287520,
        "validation": 35497,
        "test": 34989,
    }
    assert sum(split["frame_counts"].values()) == 358006
    blocks = {
        name: {
            (item["task_index"], item["source_episode_block_10"])
            for item in values
        }
        for name, values in split["blocks"].items()
    }
    assert len(blocks["train"]) == 56
    assert len(blocks["validation"]) == len(blocks["test"]) == 7
    assert blocks["train"].isdisjoint(blocks["validation"])
    assert blocks["train"].isdisjoint(blocks["test"])
    assert blocks["validation"].isdisjoint(blocks["test"])
    expected = (("train", 80), ("validation", 10), ("test", 10))
    assert all(
        set(split["task_episode_counts"][name].values()) == {count}
        for name, count in expected
    )


def test_matched_value_gate_records_negative_admission_decision() -> None:
    report = _json(VALUE_GATE_PATH)
    assert report["gate"] == "stage1.3_tiny_body_dynamics_value_and_forgetting"
    assert report["configuration"]["seeds"] == [13, 29, 47]
    assert report["configuration"]["matched_comparison_steps_per_branch"] == 180
    assert "fairness_control" in report["methodology"]
    assert report["aggregate"]["median_simulation_relative_improvement"] > 0.49
    forgetting = report["aggregate"]["median_real_source_relative_forgetting"]
    assert forgetting["project_ira_so101_v1"] > 0.10
    assert forgetting["armnetbench_so101_v01"] < 0.10
    assert report["body_value_gate_passed"] is False
    assert report["admission_decision"] == "not_admitted"
    assert report["admitted_uses"] == []
    relative_registry = REGISTRY_PATH.relative_to(ROOT).as_posix()
    assert report["input_evidence"]["registries_sha256"][relative_registry] == _sha256(REGISTRY_PATH)


def test_qualified_raw_and_cleaned_samples_are_git_ignored() -> None:
    revision = "d4ae15a1044198bced5b7401123888068033451b"
    paths = (
        f"data/raw/public_sim/so101_ma_multitask_700/{revision}/README.md",
        f"data/raw/public_sim/so101_ma_multitask_700/{revision}/data/chunk-000/file-000.parquet",
        f"data/raw/public_sim/so101_ma_multitask_700/{revision}/videos/top.mp4",
        f"data/cleaned/public_sim/so101_ma_multitask_700/{revision}/sample.json",
        ".env",
    )
    for path in paths:
        result = subprocess.run(
            ("git", "-C", str(ROOT), "check-ignore", "--quiet", path),
            check=False,
        )
        assert result.returncode == 0, path
