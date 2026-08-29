from dataclasses import replace
import json
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from data.adapters.project_ira_so101 import (
    ProjectIRASO101Adapter,
    ProjectIRASourceSpec,
    VideoSegment,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs/datasets/registry/project_ira_so101_v1.toml"


def _fixture_source(tmp_path: Path) -> tuple[ProjectIRASO101Adapter, Path]:
    root = tmp_path / "source"
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    base_spec = ProjectIRASourceSpec.from_toml(REGISTRY)
    total_episodes = 20
    frames_per_episode = 3
    total_frames = total_episodes * frames_per_episode
    spec = replace(
        base_spec,
        total_episodes=total_episodes,
        total_frames=total_frames,
        total_tasks=2,
        qualified_files=(),
        qualified_episode_indices=(0,),
    )
    info = {
        "codebase_version": "v3.0",
        "robot_type": spec.robot_id,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 2,
        "fps": 30,
        "features": {
            "observation.state": {"names": list(spec.joint_names)},
            "action": {"names": list(spec.joint_names)},
        },
    }
    (root / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
    pq.write_table(
        pa.table({"task_index": pa.array([0, 1]), "task": pa.array(["task zero", "task one"])}),
        root / "meta/tasks.parquet",
    )

    episode_rows: list[dict[str, object]] = []
    data_rows: list[dict[str, object]] = []
    global_index = 0
    for episode_index in range(total_episodes):
        task_index = 0 if episode_index < 10 else 1
        task = f"task {'zero' if task_index == 0 else 'one'}"
        start = global_index
        for frame_index in range(frames_per_episode):
            state = [float(episode_index), float(frame_index), 2.0, 3.0, 4.0, 5.0]
            action = [value + 0.25 for value in state]
            data_rows.append(
                {
                    "observation.state": state,
                    "action": action,
                    "timestamp": frame_index / 30.0,
                    "frame_index": frame_index,
                    "episode_index": episode_index,
                    "index": global_index,
                    "task_index": task_index,
                }
            )
            global_index += 1
        episode_rows.append(
            {
                "episode_index": episode_index,
                "tasks": [task],
                "length": frames_per_episode,
                "dataset_from_index": start,
                "dataset_to_index": global_index,
                "videos/observation.images.desk_view/chunk_index": 0,
                "videos/observation.images.desk_view/file_index": 0,
                "videos/observation.images.desk_view/from_timestamp": episode_index / 10,
                "videos/observation.images.desk_view/to_timestamp": (episode_index + 1) / 10,
                "videos/observation.images.wrist_left/chunk_index": 0,
                "videos/observation.images.wrist_left/file_index": 0,
                "videos/observation.images.wrist_left/from_timestamp": episode_index / 10,
                "videos/observation.images.wrist_left/to_timestamp": (episode_index + 1) / 10,
            }
        )
    pq.write_table(pa.Table.from_pylist(episode_rows), root / "meta/episodes/chunk-000/file-000.parquet")
    data_schema = pa.schema(
        [
            pa.field("observation.state", pa.list_(pa.float32())),
            pa.field("action", pa.list_(pa.float32())),
            pa.field("timestamp", pa.float32()),
            pa.field("frame_index", pa.int64()),
            pa.field("episode_index", pa.int64()),
            pa.field("index", pa.int64()),
            pa.field("task_index", pa.int64()),
        ]
    )
    pq.write_table(pa.Table.from_pylist(data_rows, schema=data_schema), root / "data/chunk-000/file-000.parquet")
    return ProjectIRASO101Adapter(root, spec), root


def test_registry_pins_source_license_native_units_and_scope() -> None:
    spec = ProjectIRASourceSpec.from_toml(REGISTRY)
    assert spec.revision == "bf6f568e9ec218fff838fb266e161724ec4f7e2c"
    assert spec.license == "CC-BY-SA-4.0"
    assert spec.status == "validated"
    assert spec.units["shoulder_pan"] == "degree"
    assert spec.units["gripper"] == "normalized_percent_0_100"
    assert spec.coordinate_frames["task_space"].startswith("not provided")
    assert spec.total_episodes == 930
    assert spec.total_frames == 844208
    assert sum(item.size for item in spec.qualified_files) == 435441545
    assert spec.task_family_ranges == {
        "desk_cleanup": (0, 24),
        "dice_throw": (25, 36),
        "fetch_ball": (37, 46),
        "sort_lego_color": (47, 92),
    }


def test_full_integrity_checks_and_canonical_conversion(tmp_path: Path) -> None:
    adapter, _ = _fixture_source(tmp_path)
    result = adapter.validate_source()
    assert result["episodes"] == 20
    assert result["frames"] == 60
    assert result["prompt_episode_counts"] == [10]

    episode = adapter.canonical_episode(0, max_transitions=2)
    episode.validate()
    assert len(episode.observations) == 3
    assert len(episode.actions) == 2
    assert episode.metadata.success is None
    assert episode.metadata.quality is None
    assert episode.metadata.native_action_semantics.startswith("Absolute calibrated")
    assert torch.isfinite(episode.observations[0].qdot).all()
    torch.testing.assert_close(episode.observations[0].previous_command, episode.observations[0].q)
    assert episode.observations[0].validity == {"desk_view": False, "wrist_left": False}


def test_integrity_rejects_nonfinite_native_actions(tmp_path: Path) -> None:
    adapter, root = _fixture_source(tmp_path)
    path = root / "data/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    rows = table.to_pylist()
    rows[4]["action"][2] = float("nan")
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    with pytest.raises(ValueError, match="NaN or Inf"):
        adapter.validate_source()


def test_video_probe_and_rgb_decode(tmp_path: Path) -> None:
    path = tmp_path / "fixture.mp4"
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=30)
        stream.width = 16
        stream.height = 12
        stream.pix_fmt = "yuv420p"
        for index in range(30):
            pixels = np.zeros((12, 16, 3), dtype=np.uint8)
            pixels[:, :, 0] = 255
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = index
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    probe = ProjectIRASO101Adapter.probe_video(path)
    assert probe["codec"] == "h264"
    assert probe["pixel_format"] == "yuv420p"
    assert (probe["width"], probe["height"], probe["fps"]) == (16, 12, 30.0)
    segment = VideoSegment("fixture", path, 0.0, 1.0, 16, 12, 30.0)
    frame = ProjectIRASO101Adapter.decode_rgb_frame(segment, 1 / 30)
    assert frame.shape == (3, 12, 16)
    assert frame.dtype == torch.uint8
    assert np.isfinite(frame.numpy()).all()
