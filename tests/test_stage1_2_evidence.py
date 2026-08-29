import hashlib
import json
from pathlib import Path
import subprocess
import tomllib


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "configs/datasets/registry/armnetbench_so101_v01.toml"
OBJECTS_PATH = ROOT / "configs/datasets/registry/armnetbench_so101_v01.objects.json"
MANIFEST_PATH = ROOT / "data/manifests/armnetbench_so101_v01.json"
QUALIFICATION_PATH = ROOT / "data/manifests/armnetbench_so101_v01.qualification.json"
SPLIT_PATH = ROOT / "data/splits/armnetbench_so101_v01.json"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_qualification_evidence_matches_registry_objects_and_manifest() -> None:
    with REGISTRY_PATH.open("rb") as handle:
        registry = tomllib.load(handle)
    objects = _json(OBJECTS_PATH)
    manifest = _json(MANIFEST_PATH)
    report = _json(QUALIFICATION_PATH)

    assert report["passed"] is True
    assert report["gate"] == "stage1.2_source_qualification"
    assert report["source_revision"] == registry["dataset"]["revision"] == manifest["revision"]
    assert report["source_status"] == manifest["status"] == "validated"
    assert report["admission_decision"] == "validated_for_stage1; not_yet_admitted_to_training"
    assert report["qualified_subset"]["verified_object_count"] == 128
    assert report["qualified_subset"]["verified_bytes"] == 664617401
    assert report["qualified_subset"]["object_manifest_sha256"] == _sha256(OBJECTS_PATH)
    assert registry["objects"]["manifest_sha256"] == _sha256(OBJECTS_PATH)
    assert len(objects["objects"]) == 128
    assert sum(item["size"] for item in objects["objects"]) == 664617401
    assert manifest["checksum"] == f"sha256:{_sha256(OBJECTS_PATH)}"

    source = report["source_validation"]
    assert source["episodes"] == 2499
    assert source["frames"] == 1127881
    assert source["tasks"] == 8
    assert source["outcome_counts"] == {
        "failure": 1532,
        "suboptimal": 52,
        "successful": 915,
    }
    assert source["terminal_done_count"] == 2499
    assert source["terminal_success_reward_count"] == 915
    assert source["null_values"] == source["nonfinite_values"] == 0
    assert source["published_statistics_trusted"] is False
    assert source["published_statistics_mismatched_counts"] == {
        "episode_index": 1147055,
        "index": 1147055,
        "timestamp": 1147055,
    }


def test_real_video_and_canonical_outcome_evidence_is_complete() -> None:
    report = _json(QUALIFICATION_PATH)
    video = report["video_validation"]
    assert set(video) == {"front", "top", "wrist"}
    assert all(values["codec"] == "av1" for values in video.values())
    assert all(values["pixel_format"] == "yuv420p" for values in video.values())
    assert all(values["fps"] == 20.0 for values in video.values())
    assert (video["front"]["width"], video["front"]["height"]) == (1024, 576)
    assert (video["top"]["width"], video["top"]["height"]) == (1024, 576)
    assert (video["wrist"]["width"], video["wrist"]["height"]) == (1280, 720)

    sample = report["canonical_sample"]
    assert sample["episodes"] == 3
    assert sample["outcome_classes"] == ["suboptimal", "failure", "successful"]
    assert sample["observations"] == 9
    assert sample["actions"] == 6
    assert sample["decoded_rgb_frames"] == 27
    assert len(sample["sha256"]) == 64


def test_task_policy_split_is_complete_disjoint_and_leakage_resistant() -> None:
    split = _json(SPLIT_PATH)
    episodes = {
        name: set(indices) for name, indices in split["episode_indices"].items()
    }
    assert episodes["train"].isdisjoint(episodes["validation"])
    assert episodes["train"].isdisjoint(episodes["test"])
    assert episodes["validation"].isdisjoint(episodes["test"])
    assert set().union(*episodes.values()) == set(range(2499))
    assert split["episode_counts"] == {"train": 1889, "validation": 320, "test": 290}
    assert split["frame_counts"] == {
        "train": 865218,
        "validation": 143349,
        "test": 119314,
    }
    assert sum(split["frame_counts"].values()) == 1127881
    assert split["group_keys"] == ["task_index", "policy_type"]

    cells = {
        name: {(item["task_index"], item["policy_type"]) for item in values}
        for name, values in split["cells"].items()
    }
    assert len(cells["train"]) == 48
    assert len(cells["validation"]) == len(cells["test"]) == 8
    assert cells["train"].isdisjoint(cells["validation"])
    assert cells["train"].isdisjoint(cells["test"])
    assert cells["validation"].isdisjoint(cells["test"])
    expected_cells = {
        (task_index, policy_type)
        for task_index in range(8)
        for policy_type in split["policy_order"]
    }
    assert set().union(*cells.values()) == expected_cells

    expected_outcomes = {"failure", "suboptimal", "successful"}
    expected_tasks = set(split["task_family_order"])
    expected_policies = set(split["policy_order"])
    for name in ("validation", "test"):
        assert set(split["outcome_counts"][name]) == expected_outcomes
        assert set(split["task_episode_counts"][name]) == expected_tasks
        assert set(split["policy_episode_counts"][name]) == expected_policies


def test_qualified_raw_and_cleaned_samples_are_git_ignored() -> None:
    revision = "2e5e89aee0e7f081078d9d6ab3b279fc4b83ea84"
    paths = (
        f"data/raw/public_real/armnetbench_so101/{revision}/README.md",
        f"data/raw/public_real/armnetbench_so101/{revision}/data/chunk-000/file-029.parquet",
        f"data/raw/public_real/armnetbench_so101/{revision}/videos/front.mp4",
        f"data/cleaned/public_real/armnetbench_so101/{revision}/sample.json",
        ".env",
    )
    for path in paths:
        result = subprocess.run(
            ("git", "-C", str(ROOT), "check-ignore", "--quiet", path),
            check=False,
        )
        assert result.returncode == 0, path
