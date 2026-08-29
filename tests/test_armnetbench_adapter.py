from dataclasses import replace
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from data.adapters.armnetbench_so101 import (
    ArmnetBenchSO101Adapter,
    ArmnetBenchSourceSpec,
    QualifiedObject,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs/datasets/registry/armnetbench_so101_v01.toml"


def _fixture_source(tmp_path: Path) -> tuple[ArmnetBenchSO101Adapter, Path]:
    root = tmp_path / "source"
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    base_spec = ArmnetBenchSourceSpec.from_toml(REGISTRY)
    total_episodes = 3
    frames_per_episode = 3
    total_frames = total_episodes * frames_per_episode
    trajectory_path = "data/chunk-000/file-000.parquet"
    spec = replace(
        base_spec,
        total_episodes=total_episodes,
        total_frames=total_frames,
        total_tasks=1,
        task_families=("tool_insert",),
        outcome_classes=("successful", "failure", "suboptimal"),
        policy_types=("diffusion", "act", "teleoperated"),
        outcome_counts={"successful": 1, "failure": 1, "suboptimal": 1},
        qualified_episode_indices=(0, 1, 2),
        qualified_objects=(QualifiedObject(trajectory_path, 0, "fixture", "trajectory_table"),),
    )
    features: dict[str, object] = {
        "observation.state": {"names": list(spec.joint_names)},
        "action": {"names": list(spec.joint_names)},
        "next.reward": {"dtype": "float32"},
        "next.done": {"dtype": "bool"},
    }
    for camera_name, camera in spec.cameras.items():
        features[camera.feature_key] = {
            "info": {
                "video.width": camera.width,
                "video.height": camera.height,
                "video.codec": camera.codec,
                "video.pix_fmt": camera.pixel_format,
                "video.fps": int(camera.fps),
            }
        }
    info = {
        "codebase_version": "v3.0",
        "robot_type": spec.robot_id,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "fps": 20,
        "features": features,
    }
    (root / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
    (root / "meta/stats.json").write_text(
        json.dumps(
            {
                name: {"count": [total_frames]}
                for name in ("observation.state", "action", "timestamp", "episode_index", "index")
            }
        ),
        encoding="utf-8",
    )
    pq.write_table(
        pa.table({"task_index": pa.array([0]), "task": pa.array(["insert the tool"])}),
        root / "meta/tasks.parquet",
    )

    outcomes = ("suboptimal", "failure", "successful")
    policies = ("diffusion", "act", "teleoperated")
    episode_rows: list[dict[str, object]] = []
    data_rows: list[dict[str, object]] = []
    global_index = 0
    video_starts = (0.0, 0.20, 0.35)
    for episode_index, (outcome, policy_type) in enumerate(zip(outcomes, policies, strict=True)):
        start = global_index
        successful = outcome == "successful"
        for frame_index in range(frames_per_episode):
            state = [
                float(episode_index),
                float(frame_index),
                2.0,
                3.0,
                4.0,
                5.0,
            ]
            data_rows.append(
                {
                    "observation.state": state,
                    "action": [value + 0.25 for value in state],
                    "timestamp": frame_index / 20.0,
                    "frame_index": frame_index,
                    "episode_index": episode_index,
                    "index": global_index,
                    "task_index": 0,
                    "next.reward": float(successful and frame_index == frames_per_episode - 1),
                    "next.done": frame_index == frames_per_episode - 1,
                }
            )
            global_index += 1
        episode_row: dict[str, object] = {
            "episode_index": episode_index,
            "tasks": ["insert the tool"],
            "length": frames_per_episode,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": start,
            "dataset_to_index": global_index,
            "success": int(successful),
            "success_class": outcome,
            "policy_repo_id": "" if policy_type == "teleoperated" else f"org/{policy_type}",
            "policy_type": policy_type,
        }
        for camera_name in spec.camera_names:
            prefix = f"videos/observation.images.{camera_name}"
            episode_row[f"{prefix}/chunk_index"] = 0
            episode_row[f"{prefix}/file_index"] = 0
            episode_row[f"{prefix}/from_timestamp"] = video_starts[episode_index]
            episode_row[f"{prefix}/to_timestamp"] = video_starts[episode_index] + 0.15
        episode_rows.append(episode_row)

    pq.write_table(
        pa.Table.from_pylist(episode_rows),
        root / "meta/episodes/chunk-000/file-000.parquet",
    )
    vector_type = pa.list_(pa.field("element", pa.float32()))
    data_schema = pa.schema(
        [
            pa.field("observation.state", vector_type),
            pa.field("action", vector_type),
            pa.field("timestamp", pa.float32()),
            pa.field("frame_index", pa.int64()),
            pa.field("episode_index", pa.int64()),
            pa.field("index", pa.int64()),
            pa.field("task_index", pa.int64()),
            pa.field("next.reward", pa.float32()),
            pa.field("next.done", pa.bool_()),
        ]
    )
    pq.write_table(
        pa.Table.from_pylist(data_rows, schema=data_schema),
        root / trajectory_path,
    )
    return ArmnetBenchSO101Adapter(root, spec), root


def test_registry_pins_release_license_units_labels_and_bounded_scope() -> None:
    spec = ArmnetBenchSourceSpec.from_toml(REGISTRY)
    assert spec.revision == "2e5e89aee0e7f081078d9d6ab3b279fc4b83ea84"
    assert spec.release_tag == "v1.0"
    assert spec.license == "Apache-2.0"
    assert spec.status == "validated"
    assert spec.units["shoulder_pan"] == "degree"
    assert spec.units["gripper"] == "normalized_percent_0_100"
    assert spec.coordinate_frames["task_space"].startswith("not provided")
    assert spec.total_episodes == 2499
    assert spec.total_frames == 1127881
    assert spec.outcome_counts == {"successful": 915, "failure": 1532, "suboptimal": 52}
    assert len(spec.qualified_objects) == 128
    assert sum(item.size for item in spec.qualified_objects) == 664617401
    assert sum(item.role == "trajectory_table" for item in spec.qualified_objects) == 120


def test_full_integrity_outcomes_video_gaps_and_canonical_conversion(tmp_path: Path) -> None:
    adapter, _ = _fixture_source(tmp_path)
    result = adapter.validate_source()
    assert result["episodes"] == 3
    assert result["frames"] == 9
    assert result["outcome_counts"] == {
        "failure": 1,
        "suboptimal": 1,
        "successful": 1,
    }
    assert result["terminal_done_count"] == 3
    assert result["terminal_success_reward_count"] == 1
    assert all(
        metadata["unreferenced_gap_count"] == 1
        for metadata in result["video_metadata"].values()
    )

    episode = adapter.canonical_episode(0, max_transitions=2)
    episode.validate()
    assert len(episode.observations) == 3
    assert len(episode.actions) == 2
    assert episode.metadata.success is False
    assert episode.metadata.quality == 0.5
    assert episode.metadata.extra["success_class"] == "suboptimal"
    assert episode.metadata.extra["imitation_eligible"] is False
    assert episode.metadata.extra["prediction_eligible"] is True
    assert episode.metadata.native_action_semantics.startswith("Absolute calibrated")
    assert torch.isfinite(episode.observations[0].qdot).all()
    torch.testing.assert_close(episode.observations[0].previous_command, episode.observations[0].q)
    assert episode.observations[0].validity == {"front": False, "top": False, "wrist": False}


def test_integrity_rejects_misaligned_timestamps(tmp_path: Path) -> None:
    adapter, root = _fixture_source(tmp_path)
    path = root / "data/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    rows = table.to_pylist()
    rows[1]["timestamp"] = 0.07
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    with pytest.raises(ValueError, match="timestamps"):
        adapter.validate_source()


def test_integrity_rejects_reward_label_disagreement(tmp_path: Path) -> None:
    adapter, root = _fixture_source(tmp_path)
    path = root / "data/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    rows = table.to_pylist()
    rows[2]["next.reward"] = 1.0
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    with pytest.raises(ValueError, match="next.reward"):
        adapter.validate_source()


def test_integrity_rejects_referenced_video_overlap(tmp_path: Path) -> None:
    adapter, root = _fixture_source(tmp_path)
    path = root / "meta/episodes/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    rows = table.to_pylist()
    for camera_name in adapter.spec.camera_names:
        prefix = f"videos/observation.images.{camera_name}"
        rows[1][f"{prefix}/from_timestamp"] = 0.10
        rows[1][f"{prefix}/to_timestamp"] = 0.25
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    with pytest.raises(ValueError, match="overlap"):
        adapter.validate_source()
