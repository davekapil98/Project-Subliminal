import hashlib
import json
from pathlib import Path
import subprocess
import tomllib
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "configs/datasets/registry/droid_raw_1_0_1.toml"
OBJECTS_PATH = ROOT / "configs/datasets/registry/droid_raw_1_0_1.objects.json"
MANIFEST_PATH = ROOT / "data/manifests/droid_raw_1_0_1.json"
QUALIFICATION_PATH = ROOT / "data/manifests/droid_raw_1_0_1.qualification.json"
SPLIT_PATH = ROOT / "data/splits/droid_raw_1_0_1.json"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _all_keys(value: Any) -> list[str]:
    if isinstance(value, dict):
        return list(value) + [key for item in value.values() for key in _all_keys(item)]
    if isinstance(value, list):
        return [key for item in value for key in _all_keys(item)]
    return []


def test_qualification_matches_registry_objects_and_manifest() -> None:
    with REGISTRY_PATH.open("rb") as handle:
        registry = tomllib.load(handle)
    objects = _json(OBJECTS_PATH)
    manifest = _json(MANIFEST_PATH)
    report = _json(QUALIFICATION_PATH)

    assert report["passed"] is True
    assert report["gate"] == "stage1.4_source_qualification"
    assert report["source_revision"] == registry["dataset"]["revision"] == manifest["revision"]
    assert report["source_status"] == manifest["status"] == "validated"
    assert report["source_priority"] == manifest["priority"] == "B"
    assert report["admission_decision"] == "validated_for_stage1; not_yet_admitted_to_training"
    assert registry["objects"]["manifest_sha256"] == _sha256(OBJECTS_PATH)
    assert len(objects["objects"]) == 59
    assert sum(item["size"] for item in objects["objects"]) == 75738594
    assert manifest["checksum"] == f"sha256:{_sha256(OBJECTS_PATH)}"
    verified = report["verified_objects"]
    assert verified["verified_object_count"] == 59
    assert verified["verified_bytes"] == 75738594
    assert verified["role_counts"] == {
        "license": 1,
        "metadata": 26,
        "trajectory": 26,
        "video": 6,
    }


def test_hdf5_video_canonical_and_privacy_evidence_is_complete() -> None:
    report = _json(QUALIFICATION_PATH)
    source = report["source_validation"]
    assert source["episodes"] == 26
    assert source["frames"] == 7817
    assert source["transitions"] == 7791
    assert source["labs"] == 13
    assert source["outcomes"] == {"failure": 13, "success": 13}
    assert source["null_values"] == source["nonfinite_values"] == 0
    assert source["stale_controller_terminal_labels"] == 2
    assert source["state_width"] == 8
    assert source["native_action_width"] == 7
    assert source["task_space_conversion_allowed"] is False

    video = report["video_validation"]
    assert video["validated_streams"] == 6
    assert video["decoded_frames"] == 18
    assert video["alignment"] == "frame_index_not_container_timestamp"
    streams = [
        stream
        for outcome in video["streams"].values()
        for stream in outcome.values()
    ]
    assert {stream["codec"] for stream in streams} == {"h264"}
    assert {stream["pixel_format"] for stream in streams} == {"yuv420p"}
    assert all(stream["frames"] == stream["trajectory_frames"] - 1 for stream in streams)
    assert all(len(stream["decoded_frame_indices"]) == 3 for stream in streams)
    assert sum(sample["decoded_rgb_frames"] for sample in report["canonical_samples"]) == 12
    assert {sample["state_width"] for sample in report["canonical_samples"]} == {8}
    assert {sample["native_action_width"] for sample in report["canonical_samples"]} == {7}

    assert report["privacy"]["committed_evidence_contains_identity_values"] is False
    assert not {"user", "user_id"}.intersection(_all_keys(report))


def test_lab_split_is_complete_disjoint_and_balanced() -> None:
    split = _json(SPLIT_PATH)
    splits = split["splits"]
    labs = {name: set(values["labs"]) for name, values in splits.items()}
    assert labs["train"].isdisjoint(labs["validation"])
    assert labs["train"].isdisjoint(labs["test"])
    assert labs["validation"].isdisjoint(labs["test"])
    assert len(set().union(*labs.values())) == 13
    assert {name: values["episode_count"] for name, values in splits.items()} == {
        "train": 58234,
        "validation": 8517,
        "test": 8145,
    }
    assert splits["train"]["outcome_counts"] == {"success": 46491, "failure": 11743}
    assert splits["validation"]["outcome_counts"] == {"success": 6781, "failure": 1736}
    assert splits["test"]["outcome_counts"] == {"success": 6468, "failure": 1677}
    assert sum(values["episode_count"] for values in splits.values()) == 74896
    assert all(values["outcome_counts"]["success"] > 0 for values in splits.values())
    assert all(values["outcome_counts"]["failure"] > 0 for values in splits.values())


def test_raw_cleaned_and_secret_paths_are_git_ignored() -> None:
    paths = (
        "data/raw/public_real/droid_raw_1_0_1/IPRL/failure/trajectory.h5",
        "data/raw/public_real/droid_raw_1_0_1/IPRL/failure/recordings/MP4/video.mp4",
        "data/cleaned/public_real/droid_raw_1_0_1/sample.json",
        ".env",
    )
    for path in paths:
        result = subprocess.run(
            ("git", "-C", str(ROOT), "check-ignore", "--quiet", path),
            check=False,
        )
        assert result.returncode == 0, path
