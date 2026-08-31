import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tomllib
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/training/stage1_5_droid_visual.toml"
OBJECTS_PATH = (
    ROOT / "configs/datasets/registry/stage1_5_visual_subset.objects.json"
)
DROID_STAGE1_4_OBJECTS_PATH = (
    ROOT / "configs/datasets/registry/droid_raw_1_0_1.objects.json"
)
PROJECT_SPLIT_PATH = ROOT / "data/splits/project_ira_so101_v1.json"
ARM_SPLIT_PATH = ROOT / "data/splits/armnetbench_so101_v01.json"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _planner() -> Any:
    path = ROOT / "scripts/prepare_stage1_5_visual_subset.py"
    spec = importlib.util.spec_from_file_location("stage1_5_planner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _episode_prefix(object_name: str) -> str:
    if "/recordings/" in object_name:
        return object_name.split("/recordings/", 1)[0]
    return object_name.rsplit("/", 1)[0]


def _all_keys(value: Any) -> list[str]:
    if isinstance(value, dict):
        return list(value) + [key for item in value.values() for key in _all_keys(item)]
    if isinstance(value, list):
        return [key for item in value for key in _all_keys(item)]
    return []


def test_protocol_is_frozen_action_free_and_matched() -> None:
    with CONFIG_PATH.open("rb") as handle:
        config = tomllib.load(handle)

    assert config["stage"] == "1.5"
    assert config["status"] == "frozen_before_acquisition"
    assert config["acquisition"]["cap_bytes"] == 20_000_000_000
    assert config["acquisition"]["minimum_free_reserve_bytes"] == 5_000_000_000
    assert config["selection"]["droid"]["expected_episodes"] == 416
    assert config["selection"]["droid"]["quota_per_lab_outcome"] == 16
    assert config["selection"]["droid"]["samples_per_episode"] == 8
    assert config["representation"]["action_policy"].startswith("No action input")
    assert config["training"]["seeds"] == [13, 29, 47]
    assert config["training"]["matched_comparison_steps_per_branch"] == 240
    assert config["evaluation"]["forgetting_relative_tolerance"] == 0.10
    assert config["evaluation"]["minimum_positive_seed_count"] == 2
    assert "SO-101 motor targets" in config["scope"]["explicitly_excluded_uses"]
    assert "cross-embodiment native-action mixing" in config["scope"]["explicitly_excluded_uses"]


def test_so101_ranges_have_the_declared_frozen_split_membership() -> None:
    project = _json(PROJECT_SPLIT_PATH)
    project_train_tasks = set(project["task_indices"]["train"])
    project_test_tasks = set(project["task_indices"]["test"])
    assert {episode // 10 for episode in range(0, 90)} <= project_train_tasks
    assert {episode // 10 for episode in range(880, 890)} <= project_test_tasks

    arm = _json(ARM_SPLIT_PATH)
    arm_train = set(arm["episode_indices"]["train"])
    arm_test = set(arm["episode_indices"]["test"])
    assert set(range(1549, 1599)) <= arm_train
    assert set(range(610, 628)) <= arm_test
    assert set(range(1549, 1599)).isdisjoint(arm_test)
    assert set(range(610, 628)).isdisjoint(arm_train)


def test_droid_selector_is_deterministic_complete_and_excludes_prior_episodes() -> None:
    planner = _planner()
    prefix_a = "release/LAB/success/date/episode-a"
    prefix_b = "release/LAB/success/date/episode-b"
    prefix_c = "release/LAB/success/date/episode-c"
    listed = []
    for prefix in (prefix_a, prefix_b, prefix_c):
        listed.extend(
            {"name": f"{prefix}/recordings/MP4/camera-{index}.mp4"}
            for index in range(3)
        )
    listed.append(
        {"name": f"{prefix_a}/recordings/MP4/camera-0-stereo.mp4"}
    )
    listed.pop(7)  # prefix_c is incomplete and cannot be selected.

    selected, inventory = planner.select_episode_prefixes(
        dataset_id="droid_raw_1_0_1",
        lab="LAB",
        outcome="success",
        listed_videos=reversed(listed),
        excluded_prefixes={prefix_a},
        quota=1,
    )
    assert selected == [prefix_b]
    assert inventory["stereo_video_objects"] == 1
    assert inventory["complete_eligible_episode_prefixes"] == 1


def test_exact_object_plan_is_under_cap_complete_and_pii_local() -> None:
    with CONFIG_PATH.open("rb") as handle:
        config = tomllib.load(handle)
    plan = _json(OBJECTS_PATH)

    assert plan["status"] == "frozen_before_acquisition"
    assert plan["config_sha256"] == _sha256(CONFIG_PATH)
    assert plan["selection"]["droid_episode_count"] == 416
    assert plan["selection"]["stage1_4_episode_prefixes_excluded"] == 26
    acquisition = plan["acquisition"]
    assert acquisition["cap_bytes"] == 20_000_000_000
    assert acquisition["selected_bytes"] < acquisition["cap_bytes"]
    assert acquisition["headroom_bytes"] == (
        acquisition["cap_bytes"] - acquisition["selected_bytes"]
    )
    assert acquisition["object_count"] == len(plan["objects"]) == 2222
    assert acquisition["dataset_object_counts"] == {
        "armnetbench_so101_v01": 131,
        "droid_raw_1_0_1": 2081,
        "project_ira_so101_v1": 10,
    }

    selected_droid: dict[str, list[dict[str, Any]]] = {}
    for item in plan["objects"]:
        if item["provider"] == "gcs":
            assert item["generation"] and len(item["md5"]) == 32
        else:
            assert len(item["sha256"]) == 64
            assert item["revision"] and item["repository_id"]
        if item["dataset_id"] == "droid_raw_1_0_1" and item["episode_selector"]:
            selected_droid.setdefault(item["episode_selector"], []).append(item)
    assert len(selected_droid) == 416
    for objects in selected_droid.values():
        assert len(objects) == 5
        roles = [item["role"] for item in objects]
        assert roles.count("metadata") == 1
        assert roles.count("trajectory") == 1
        assert roles.count("video") == 3

    previous = _json(DROID_STAGE1_4_OBJECTS_PATH)
    previous_prefixes = {
        _episode_prefix(item["object_name"])
        for item in previous["objects"]
        if item["role"] in {"metadata", "trajectory", "video"}
    }
    selected_prefixes = {
        _episode_prefix(item["object_name"])
        for item in plan["objects"]
        if item["dataset_id"] == "droid_raw_1_0_1"
        and item["episode_selector"]
    }
    assert previous_prefixes.isdisjoint(selected_prefixes)
    assert plan["privacy"]["raw_metadata_contents_committed"] is False
    assert not {"user", "user_id"}.intersection(_all_keys(plan))
    assert len(config["so101_video_objects"]) == 10


def test_stage1_5_raw_cache_artifacts_and_secrets_are_git_ignored() -> None:
    paths = (
        "data/raw/public_real/droid_raw_1_0_1/LAB/success/episode/trajectory.h5",
        "data/raw/public_real/droid_raw_1_0_1/LAB/success/episode/recordings/MP4/camera.mp4",
        "data/cache/stage1_5_visual/droid_train.npz",
        "artifacts/runs/stage1_5/checkpoint.pt",
        ".env",
    )
    for path in paths:
        result = subprocess.run(
            ("git", "-C", str(ROOT), "check-ignore", "--quiet", path),
            check=False,
        )
        assert result.returncode == 0, path
