from dataclasses import replace
import json
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from data.adapters.so101_ma_multitask_700 import (
    SO101MAMultiTaskAdapter,
    SO101MAMultiTaskSourceSpec,
    VideoSegment,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs/datasets/registry/so101_ma_multitask_700.toml"


def _fixture_source(tmp_path: Path) -> tuple[SO101MAMultiTaskAdapter, Path]:
    root = tmp_path / "source"
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    base_spec = SO101MAMultiTaskSourceSpec.from_toml(REGISTRY)
    frames_per_episode = 2
    total_frames = 700 * frames_per_episode
    upstream_sources = tuple(
        replace(source, episodes=100, frames=100 * frames_per_episode)
        for source in base_spec.upstream_sources
    )
    spec = replace(
        base_spec,
        total_frames=total_frames,
        qualified_objects=(),
        qualified_episode_indices=(0,),
        extra_video_episode=0,
        extra_video_frames_per_camera=1,
        upstream_sources=upstream_sources,
    )
    features: dict[str, object] = {
        "observation.state": {"names": list(spec.joint_names)},
        "action": {"names": list(spec.joint_names)},
    }
    for camera_name, camera in spec.cameras.items():
        features[camera.feature_key] = {
            "info": {
                "video.width": camera.width,
                "video.height": camera.height,
                "video.codec": "mixed",
                "video.pix_fmt": camera.pixel_format,
                "video.fps": int(camera.fps),
            }
        }
    info = {
        "codebase_version": "v3.0",
        "robot_type": spec.robot_id,
        "total_episodes": 700,
        "total_frames": total_frames,
        "total_tasks": 7,
        "fps": 10,
        "features": features,
    }
    (root / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
    (root / "meta/stats.json").write_text(
        json.dumps(
            {
                name: {"count": [total_frames + 1]}
                for name in (
                    "observation.state",
                    "action",
                    "timestamp",
                    "episode_index",
                    "index",
                )
            }
        ),
        encoding="utf-8",
    )
    pq.write_table(
        pa.table(
            {
                "task_index": pa.array(range(7), type=pa.int64()),
                spec.task_text_source_column: pa.array(spec.task_labels),
            }
        ),
        root / "meta/tasks.parquet",
    )
    provenance = {
        "output_repo_id": spec.repository_id,
        "total_episodes": 700,
        "total_frames": total_frames,
        "fps": 10,
        "total_tasks": 7,
        "derivation": "synthetic_test_fixture",
        "dropped_optional_feature_groups": [
            "action.radian_urdf0",
            "observation.state.radian_urdf0",
            "observation.ee_pos.robot_xyzrpy",
            "observation.gripper_binary",
        ],
        "sources": [
            {
                "repo_id": source.declared_repository_id,
                "episodes": source.episodes,
                "frames": source.frames,
            }
            for source in upstream_sources
        ],
    }
    (root / "meta/scrape_collection_manifest.json").write_text(
        json.dumps(provenance), encoding="utf-8"
    )

    episode_rows: list[dict[str, object]] = []
    data_rows: list[dict[str, object]] = []
    global_index = 0
    for episode_index in range(700):
        task_index = episode_index // 100
        start = global_index
        for frame_index in range(frames_per_episode):
            state = [
                float(task_index),
                float(episode_index % 100),
                float(frame_index),
                3.0,
                4.0,
                5.0,
            ]
            data_rows.append(
                {
                    "observation.state": state,
                    "action": [value + 0.25 for value in state],
                    "timestamp": frame_index / 10.0,
                    "frame_index": frame_index,
                    "episode_index": episode_index,
                    "index": global_index,
                    "task_index": task_index,
                }
            )
            global_index += 1
        episode_row: dict[str, object] = {
            "episode_index": episode_index,
            "tasks": [spec.task_labels[task_index]],
            "length": frames_per_episode,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": start,
            "dataset_to_index": global_index,
        }
        for camera_name in spec.camera_names:
            prefix = f"videos/observation.images.{camera_name}"
            episode_row[f"{prefix}/chunk_index"] = 0
            episode_row[f"{prefix}/file_index"] = episode_index
            episode_row[f"{prefix}/from_timestamp"] = 0.0
            episode_row[f"{prefix}/to_timestamp"] = (
                frames_per_episode
                + (1 if episode_index == spec.extra_video_episode else 0)
            ) / spec.fps
        episode_rows.append(episode_row)

    pq.write_table(
        pa.Table.from_pylist(episode_rows),
        root / "meta/episodes/chunk-000/file-000.parquet",
    )
    vector_type = pa.list_(pa.field("element", pa.float32()), 6)
    schema = pa.schema(
        [
            pa.field("observation.state", vector_type),
            pa.field("action", vector_type),
            pa.field("timestamp", pa.float32()),
            pa.field("frame_index", pa.int64()),
            pa.field("episode_index", pa.int64()),
            pa.field("index", pa.int64()),
            pa.field("task_index", pa.int64()),
        ]
    )
    pq.write_table(
        pa.Table.from_pylist(data_rows, schema=schema),
        root / "data/chunk-000/file-000.parquet",
    )
    return SO101MAMultiTaskAdapter(root, spec), root


def test_registry_pins_provenance_unknown_units_and_bounded_scope() -> None:
    spec = SO101MAMultiTaskSourceSpec.from_toml(REGISTRY)
    assert spec.revision == "d4ae15a1044198bced5b7401123888068033451b"
    assert spec.license == "Apache-2.0"
    assert spec.status == "validated"
    assert spec.total_episodes == 700
    assert spec.total_frames == 358006
    assert spec.full_tree_bytes == 3822803256
    assert len(spec.upstream_sources) == 7
    assert all(source.license == "Apache-2.0" for source in spec.upstream_sources)
    assert "unpublished" in spec.units["shoulder_pan"]
    assert "no task-space action" in spec.coordinate_frames["task_space"]
    assert sum(item.size for item in spec.qualified_objects) == 567845886


def test_full_integrity_task_normalization_and_canonical_conversion(
    tmp_path: Path,
) -> None:
    adapter, _ = _fixture_source(tmp_path)
    assert adapter.tasks() == dict(enumerate(adapter.spec.task_labels))
    result = adapter.validate_source()
    assert result["episodes"] == 700
    assert result["frames"] == 1400
    assert result["task_text_source_column"] == "__index_level_0__"
    assert result["task_text_normalized"] is True
    assert result["published_statistics_trusted"] is False
    assert result["native_action_physical_units_proven"] is False
    assert result["task_space_conversion_allowed"] is False
    assert set(result["task_episode_counts"].values()) == {100}

    episode = adapter.canonical_episode(0, max_transitions=1)
    episode.validate()
    assert len(episode.observations) == 2
    assert len(episode.actions) == 1
    assert episode.metadata.success is None
    assert episode.metadata.quality is None
    assert episode.metadata.extra["imitation_eligible"] is False
    assert episode.metadata.extra["prediction_eligible"] is True
    assert episode.metadata.extra["unused_video_frames_per_camera"] == 1
    torch.testing.assert_close(
        episode.observations[0].previous_command,
        episode.observations[0].q,
    )
    assert episode.observations[0].validity == {
        "top": False,
        "left_wrist": False,
    }


def test_integrity_rejects_nonfinite_native_actions(tmp_path: Path) -> None:
    adapter, root = _fixture_source(tmp_path)
    path = root / "data/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    rows = table.to_pylist()
    rows[3]["action"][2] = float("nan")
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    with pytest.raises(ValueError, match="NaN or Inf"):
        adapter.validate_source()


def test_video_probe_decode_and_aligned_interval_guard(tmp_path: Path) -> None:
    path = tmp_path / "fixture.mp4"
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=10)
        stream.width = 16
        stream.height = 12
        stream.pix_fmt = "yuv420p"
        for index in range(3):
            pixels = np.zeros((12, 16, 3), dtype=np.uint8)
            pixels[:, :, index % 3] = 255
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = index
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    probe = SO101MAMultiTaskAdapter.probe_video(path)
    assert probe["codec"] == "h264"
    assert probe["pixel_format"] == "yuv420p"
    segment = VideoSegment(
        0, "fixture", path, 0.0, 0.2, 0.3, 2, 1, 16, 12, 10.0
    )
    frame = SO101MAMultiTaskAdapter.decode_rgb_frame(segment, 0.1)
    assert frame.shape == (3, 12, 16)
    assert frame.dtype == torch.uint8
    with pytest.raises(ValueError, match="outside aligned"):
        SO101MAMultiTaskAdapter.decode_rgb_frame(segment, 0.2)
