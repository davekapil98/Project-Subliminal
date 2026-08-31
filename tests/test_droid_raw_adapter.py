from dataclasses import replace
import json
from pathlib import Path

import av
import h5py
import numpy as np
import pytest
import torch

from data.adapters.droid_raw import DROIDObject, DROIDRawAdapter, DROIDSourceSpec


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs/datasets/registry/droid_raw_1_0_1.toml"


def _fixture_adapter(
    tmp_path: Path, *, controller_matches: bool = True
) -> tuple[DROIDRawAdapter, Path]:
    root = tmp_path / "source"
    episode_dir = root / "IPRL/failure/fixture"
    episode_dir.mkdir(parents=True)
    metadata_path = episode_dir / "metadata_fixture.json"
    trajectory_path = episode_dir / "trajectory.h5"
    metadata_path.write_text(
        json.dumps(
            {
                "lab": "IPRL",
                "success": False,
                "trajectory_length": 3,
                "uuid": "fixture-uuid",
                "current_task": "Move object",
                "hdf5_path": "failure/fixture/trajectory.h5",
                "wrist_cam_serial": "wrist",
                "ext1_cam_serial": "exterior-one",
                "ext2_cam_serial": "exterior-two",
                "wrist_cam_extrinsics": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                "ext1_cam_extrinsics": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
                "ext2_cam_extrinsics": [2.0, 2.1, 2.2, 2.3, 2.4, 2.5],
                "user": "must not escape",
                "user_id": "must not escape either",
            }
        ),
        encoding="utf-8",
    )
    with h5py.File(trajectory_path, "w") as handle:
        handle.attrs["success"] = False
        handle.attrs["failure"] = True
        handle.attrs["current_task"] = "Move object"
        handle.create_dataset(
            "observation/robot_state/joint_positions",
            data=np.arange(21, dtype=np.float64).reshape(3, 7) / 10,
        )
        handle.create_dataset(
            "observation/robot_state/joint_velocities",
            data=np.ones((3, 7), dtype=np.float64),
        )
        handle.create_dataset(
            "observation/robot_state/gripper_position",
            data=np.array([0.0, 0.25, 0.5]),
        )
        handle.create_dataset(
            "action/cartesian_velocity", data=np.full((3, 6), 0.5)
        )
        handle.create_dataset(
            "action/gripper_velocity", data=np.array([0.1, 0.2, 0.3])
        )
        handle.create_dataset(
            "action/joint_position",
            data=np.arange(21, dtype=np.float64).reshape(3, 7) / 20,
        )
        handle.create_dataset(
            "action/gripper_position", data=np.array([0.2, 0.4, 0.6])
        )
        handle.create_dataset(
            "observation/timestamp/control/step_start",
            data=np.array([1_000.0, 1_100.0, 1_200.0]),
        )
        handle.create_dataset(
            "observation/timestamp/control/control_start",
            data=np.array([1_001.0, 1_101.0, 1_201.0]),
        )
        handle.create_dataset(
            "observation/timestamp/skip_action", data=np.array([False, True, False])
        )
        terminal_success = not controller_matches
        handle.create_dataset(
            "observation/controller_info/success",
            data=np.array([False, False, terminal_success]),
        )
        handle.create_dataset(
            "observation/controller_info/failure",
            data=np.array([False, False, not terminal_success]),
        )
        camera_fixtures = (
            ("wrist", 0, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
            ("exterior-one", 1, [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]),
            ("exterior-two", 1, [2.0, 2.1, 2.2, 2.3, 2.4, 2.5]),
        )
        for serial, camera_type, extrinsics in camera_fixtures:
            handle.create_dataset(
                f"observation/camera_type/{serial}",
                data=np.full(3, camera_type, dtype=np.int64),
            )
            handle.create_dataset(
                f"observation/camera_extrinsics/{serial}_left",
                data=np.tile(np.asarray(extrinsics), (3, 1)),
            )
            handle.create_dataset(
                f"observation/timestamp/cameras/{serial}_estimated_capture",
                data=np.array([990, 1_090, 1_190], dtype=np.int64),
            )

    base_spec = DROIDSourceSpec.from_toml(REGISTRY)
    objects = (
        DROIDObject(
            path="IPRL/failure/fixture/metadata_fixture.json",
            object_name="fixture-metadata",
            generation="1",
            size=metadata_path.stat().st_size,
            md5="0" * 32,
            sha256="0" * 64,
            role="metadata",
            lab="IPRL",
            outcome="failure",
        ),
        DROIDObject(
            path="IPRL/failure/fixture/trajectory.h5",
            object_name="fixture-trajectory",
            generation="1",
            size=trajectory_path.stat().st_size,
            md5="0" * 32,
            sha256="0" * 64,
            role="trajectory",
            lab="IPRL",
            outcome="failure",
        ),
    )
    adapter = object.__new__(DROIDRawAdapter)
    adapter.root = root
    adapter.spec = replace(base_spec, objects=objects)
    return adapter, trajectory_path


def test_registry_pins_complete_inventory_license_and_bounded_scope() -> None:
    spec = DROIDSourceSpec.from_toml(REGISTRY)
    assert spec.revision == "1.0.1"
    assert spec.collection_code_revision == "33ae6a67274f36d2e29525b86f23a56616ef43a7"
    assert spec.license == "CC-BY-4.0"
    assert spec.priority == "B"
    assert spec.status == "validated"
    assert (spec.indexed_episodes, spec.indexed_successes, spec.indexed_failures) == (
        74896,
        59740,
        15156,
    )
    assert len(spec.labs) == 13
    assert len(spec.objects) == 59
    assert sum(item.size for item in spec.objects) == 75738594
    assert spec.schema["canonical_state_width"] == 8
    assert spec.schema["native_action_width"] == 7


def test_integrity_redaction_stale_label_audit_and_canonicalization(
    tmp_path: Path,
) -> None:
    adapter, _ = _fixture_adapter(tmp_path, controller_matches=False)
    assert adapter.metadata_record("IPRL", "failure") == {
        "lab": "IPRL",
        "success": False,
        "trajectory_length": 3,
        "current_task": "Move object",
    }
    cell = adapter._inspect_cell("IPRL", "failure")
    assert cell["controller_terminal_matches_authoritative"] is False
    assert cell["skip_action_transitions"] == 1

    episode = adapter.canonical_episode("IPRL", "failure")
    episode.validate()
    assert episode.metadata.episode_id == "droid_raw_1_0_1:IPRL:failure"
    assert episode.metadata.success is False
    assert episode.metadata.quality == 0.0
    assert episode.metadata.extra["imitation_eligible"] is False
    assert episode.metadata.extra["prediction_eligible"] is True
    assert episode.metadata.extra["imitation_weight"] == 0.0
    assert episode.metadata.extra["prediction_weight"] == 1.0
    assert episode.language == ("Move object",)
    assert len(episode.observations) == 3
    assert len(episode.actions) == 2
    assert episode.observations[0].q.shape == (8,)
    assert episode.actions[0].native.shape == (7,)
    assert len(episode.scene_metadata) == 3
    assert set(episode.scene_metadata[0]["camera_extrinsics"]) == {
        "wrist",
        "exterior_1",
        "exterior_2",
    }
    assert episode.scene_metadata[0]["camera_extrinsics"]["wrist"] == {
        "translation_m": [0.0, 0.1, 0.2],
        "rotation_euler_xyz_rad": [0.3, 0.4, 0.5],
        "source_eye": "left",
        "target_frame": "robot_base",
    }
    assert episode.scene_metadata[0]["source_timestamps"] == {
        "control_step_start_unix_ms": 1000,
        "control_start_unix_ms": 1001,
        "camera_estimated_capture_unix_ms": {
            "wrist": 990,
            "exterior_1": 990,
            "exterior_2": 990,
        },
    }
    assert episode.scene_metadata[0]["source_recorded_joint_velocity_rad_s"] == [
        1.0
    ] * 7
    torch.testing.assert_close(episode.observations[0].qdot[:7], torch.ones(7))
    torch.testing.assert_close(
        episode.observations[0].previous_command, episode.observations[0].q
    )
    assert episode.observations[1].validity["action_executed"] is False


def test_timestamp_regression_is_rejected(tmp_path: Path) -> None:
    adapter, trajectory_path = _fixture_adapter(tmp_path)
    with h5py.File(trajectory_path, "r+") as handle:
        handle[adapter.spec.schema["timestamp_path"]][1] = 900.0
    with pytest.raises(ValueError, match="strictly increasing"):
        adapter._inspect_cell("IPRL", "failure")


def test_video_probe_and_frame_index_decode(tmp_path: Path) -> None:
    path = tmp_path / "fixture.mp4"
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=60)
        stream.width = 16
        stream.height = 12
        stream.pix_fmt = "yuv420p"
        for index in range(3):
            pixels = np.zeros((12, 16, 3), dtype=np.uint8)
            pixels[:, :, index] = 255
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = index
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    probe = DROIDRawAdapter.probe_video(path)
    assert probe["codec"] == "h264"
    assert probe["pixel_format"] == "yuv420p"
    assert probe["frames"] == 3
    frame = DROIDRawAdapter.decode_rgb_frame(path, 2)
    assert frame.shape == (3, 12, 16)
    assert frame.dtype == torch.uint8
    with pytest.raises(ValueError, match="no frame"):
        DROIDRawAdapter.decode_rgb_frame(path, 3)
