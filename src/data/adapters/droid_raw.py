"""Strict adapter for the pinned DROID raw v1.0.1 qualification subset."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tomllib
from typing import Any

import numpy as np
import torch

from data.canonical_schema import Action, CanonicalEpisode, EpisodeMetadata, Observation


OUTCOMES = ("failure", "success")
PRIVATE_METADATA_FIELDS = frozenset({"user", "user_id"})


def _hdf5() -> Any:
    try:
        import h5py
    except ImportError as error:  # pragma: no cover - exercised without the data extra
        raise RuntimeError(
            "DROID raw support requires the 'data' optional dependency (h5py)"
        ) from error
    return h5py


def _video() -> Any:
    try:
        import av
    except ImportError as error:  # pragma: no cover - exercised without the data extra
        raise RuntimeError(
            "DROID video support requires the 'data' optional dependency (av)"
        ) from error
    return av


@dataclass(frozen=True)
class DROIDObject:
    path: str
    object_name: str
    generation: str
    size: int
    md5: str
    sha256: str
    role: str
    lab: str | None = None
    outcome: str | None = None
    camera: str | None = None


@dataclass(frozen=True)
class DROIDCameraSpec:
    name: str
    width: int
    height: int
    codec: str
    pixel_format: str
    container_fps: float


@dataclass(frozen=True)
class DROIDSourceSpec:
    dataset_id: str
    bucket: str
    release_prefix: str
    revision: str
    source_url: str
    collection_code_revision: str
    license: str
    redistribution_terms: str
    domain: str
    robot_id: str
    embodiment: str
    data_format: str
    collection_method: str
    native_action_semantics: str
    priority: str
    status: str
    nominal_control_hz: float
    indexed_episodes: int
    indexed_successes: int
    indexed_failures: int
    modalities: tuple[str, ...]
    camera_names: tuple[str, ...]
    outcome_classes: tuple[str, ...]
    labs: tuple[str, ...]
    known_limitations: tuple[str, ...]
    inventory_pages: int
    inventory_metadata_bytes: int
    lab_outcome_counts: dict[str, dict[str, int]]
    object_manifest_path: str
    object_manifest_sha256: str
    qualified_objects: int
    qualified_subset_bytes: int
    qualified_episodes: int
    units: dict[str, str]
    coordinate_frames: dict[str, str]
    schema: dict[str, Any]
    cameras: dict[str, DROIDCameraSpec]
    train_labs: tuple[str, ...]
    validation_labs: tuple[str, ...]
    test_labs: tuple[str, ...]
    objects: tuple[DROIDObject, ...]

    @classmethod
    def from_toml(cls, path: Path) -> DROIDSourceSpec:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        dataset = raw["dataset"]
        inventory = raw["inventory"]
        objects_config = raw["objects"]
        repository_root = path.resolve().parents[3]
        object_path = repository_root / objects_config["manifest_path"]
        object_bytes = object_path.read_bytes()
        actual_sha256 = hashlib.sha256(object_bytes).hexdigest()
        if actual_sha256 != objects_config["manifest_sha256"]:
            raise ValueError("DROID object manifest checksum differs from the registry")
        object_manifest = json.loads(object_bytes)
        if object_manifest.get("dataset_id") != dataset["dataset_id"]:
            raise ValueError("DROID object manifest dataset_id differs from the registry")
        objects = tuple(DROIDObject(**item) for item in object_manifest["objects"])
        cameras = {
            name: DROIDCameraSpec(name=name, **values)
            for name, values in raw["cameras"].items()
        }
        splitting = raw["splitting"]
        return cls(
            dataset_id=dataset["dataset_id"],
            bucket=dataset["bucket"],
            release_prefix=dataset["release_prefix"],
            revision=dataset["revision"],
            source_url=dataset["source_url"],
            collection_code_revision=dataset["collection_code_revision"],
            license=dataset["license"],
            redistribution_terms=dataset["redistribution_terms"],
            domain=dataset["domain"],
            robot_id=dataset["robot_id"],
            embodiment=dataset["embodiment"],
            data_format=dataset["data_format"],
            collection_method=dataset["collection_method"],
            native_action_semantics=dataset["native_action_semantics"],
            priority=dataset["priority"],
            status=dataset["status"],
            nominal_control_hz=float(dataset["nominal_control_hz"]),
            indexed_episodes=int(dataset["indexed_episodes"]),
            indexed_successes=int(dataset["indexed_successes"]),
            indexed_failures=int(dataset["indexed_failures"]),
            modalities=tuple(dataset["modalities"]),
            camera_names=tuple(dataset["camera_names"]),
            outcome_classes=tuple(dataset["outcome_classes"]),
            labs=tuple(dataset["labs"]),
            known_limitations=tuple(dataset["known_limitations"]),
            inventory_pages=int(inventory["pages"]),
            inventory_metadata_bytes=int(inventory["metadata_bytes"]),
            lab_outcome_counts={
                lab: {outcome: int(count) for outcome, count in counts.items()}
                for lab, counts in inventory["labs"].items()
            },
            object_manifest_path=objects_config["manifest_path"],
            object_manifest_sha256=objects_config["manifest_sha256"],
            qualified_objects=int(objects_config["qualified_objects"]),
            qualified_subset_bytes=int(objects_config["qualified_subset_bytes"]),
            qualified_episodes=int(objects_config["qualified_episodes"]),
            units={key: str(value) for key, value in raw["units"].items()},
            coordinate_frames={
                key: str(value) for key, value in raw["coordinate_frames"].items()
            },
            schema=dict(raw["schema"]),
            cameras=cameras,
            train_labs=tuple(splitting["train_labs"]),
            validation_labs=tuple(splitting["validation_labs"]),
            test_labs=tuple(splitting["test_labs"]),
            objects=objects,
        )


class DROIDRawAdapter:
    """Read, validate and canonicalize the bounded DROID qualification subset."""

    def __init__(self, root: Path, spec: DROIDSourceSpec) -> None:
        self.root = root
        self.spec = spec
        self._validate_registry()

    @classmethod
    def from_registry(cls, root: Path, registry_path: Path) -> DROIDRawAdapter:
        return cls(root=root, spec=DROIDSourceSpec.from_toml(registry_path))

    def _validate_registry(self) -> None:
        if self.spec.domain != "real" or self.spec.priority != "B":
            raise ValueError("DROID must remain a Priority B real-data source")
        if self.spec.status != "validated":
            raise ValueError("the pinned DROID qualification source must be validated")
        if self.spec.outcome_classes != ("success", "failure"):
            raise ValueError("DROID outcome classes differ from the pinned registry")
        if set(self.spec.cameras) != set(self.spec.camera_names):
            raise ValueError("camera specifications differ from camera_names")
        if set(self.spec.lab_outcome_counts) != set(self.spec.labs):
            raise ValueError("inventory lab counts differ from the pinned lab list")
        if self.spec.indexed_successes + self.spec.indexed_failures != self.spec.indexed_episodes:
            raise ValueError("DROID outcome counts do not sum to the episode inventory")
        inventory_counts = Counter()
        for counts in self.spec.lab_outcome_counts.values():
            inventory_counts.update(counts)
        if inventory_counts != Counter(
            success=self.spec.indexed_successes, failure=self.spec.indexed_failures
        ):
            raise ValueError("per-lab DROID inventory does not sum to global counts")
        split_labs = (
            set(self.spec.train_labs),
            set(self.spec.validation_labs),
            set(self.spec.test_labs),
        )
        if any(left & right for index, left in enumerate(split_labs) for right in split_labs[index + 1 :]):
            raise ValueError("DROID collection-lab splits overlap")
        if set().union(*split_labs) != set(self.spec.labs):
            raise ValueError("DROID collection-lab splits do not cover every lab")
        if len(self.spec.objects) != self.spec.qualified_objects:
            raise ValueError("DROID qualification object count differs from the registry")
        if sum(item.size for item in self.spec.objects) != self.spec.qualified_subset_bytes:
            raise ValueError("DROID qualification byte count differs from the registry")
        roles = Counter(item.role for item in self.spec.objects)
        if roles != Counter(metadata=26, trajectory=26, video=6, license=1):
            raise ValueError("DROID qualification object roles differ from the registry")
        for item in self.spec.objects:
            expected_name = (
                item.path if item.role == "license" else f"{self.spec.release_prefix}/{item.path}"
            )
            if item.role == "license":
                expected_name = "robotics/droid/1.0.0/CC-BY-4.0"
            if item.object_name != expected_name:
                raise ValueError("DROID object path does not match its pinned object name")

    def _objects_for(self, lab: str, outcome: str, role: str) -> tuple[DROIDObject, ...]:
        if lab not in self.spec.labs or outcome not in OUTCOMES:
            raise KeyError(f"unknown qualification cell {lab!r}/{outcome!r}")
        return tuple(
            item
            for item in self.spec.objects
            if item.lab == lab and item.outcome == outcome and item.role == role
        )

    def _episode_files(self, lab: str, outcome: str) -> tuple[Path, Path]:
        metadata_objects = self._objects_for(lab, outcome, "metadata")
        trajectory_objects = self._objects_for(lab, outcome, "trajectory")
        if len(metadata_objects) != 1 or len(trajectory_objects) != 1:
            raise ValueError(f"{lab}/{outcome} must have one metadata/HDF5 pair")
        metadata_path = self.root / metadata_objects[0].path
        trajectory_path = self.root / trajectory_objects[0].path
        if metadata_path.parent != trajectory_path.parent:
            raise ValueError(f"{lab}/{outcome} metadata and HDF5 are not co-located")
        return metadata_path, trajectory_path

    def _metadata(self, lab: str, outcome: str) -> dict[str, Any]:
        metadata_path, _ = self._episode_files(lab, outcome)
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    def metadata_record(self, lab: str, outcome: str) -> dict[str, Any]:
        """Return a deliberately minimal, PII-free metadata view."""

        metadata = self._metadata(lab, outcome)
        return {
            "lab": metadata.get("lab"),
            "success": metadata.get("success"),
            "trajectory_length": metadata.get("trajectory_length"),
            "current_task": metadata.get("current_task"),
        }

    def _inspect_cell(self, lab: str, outcome: str) -> dict[str, Any]:
        h5py = _hdf5()
        metadata_path, trajectory_path = self._episode_files(lab, outcome)
        metadata = self._metadata(lab, outcome)
        expected_success = outcome == "success"
        safe_required = (
            "lab",
            "success",
            "trajectory_length",
            "current_task",
            "hdf5_path",
            "wrist_cam_extrinsics",
            "ext1_cam_extrinsics",
            "ext2_cam_extrinsics",
        )
        if any(key not in metadata for key in safe_required):
            raise ValueError(f"{lab}/{outcome} metadata lacks a required field")
        if metadata["lab"] != lab or bool(metadata["success"]) != expected_success:
            raise ValueError(f"{lab}/{outcome} metadata outcome or lab disagrees with its path")
        relative_hdf5 = str(trajectory_path.relative_to(self.root / lab))
        if metadata["hdf5_path"] != relative_hdf5:
            raise ValueError(f"{lab}/{outcome} metadata hdf5_path disagrees with its pair")
        task = str(metadata["current_task"]).strip()
        if not task:
            raise ValueError(f"{lab}/{outcome} has an empty task annotation")
        serials = tuple(str(metadata.get(name, "")).strip() for name in (
            "wrist_cam_serial", "ext1_cam_serial", "ext2_cam_serial"
        ))
        if any(not serial for serial in serials) or len(set(serials)) != 3:
            raise ValueError(f"{lab}/{outcome} must declare three unique cameras")

        schema = self.spec.schema
        required_paths = (
            "state_joint_path", "state_joint_velocity_path", "state_gripper_path",
            "action_cartesian_velocity_path", "action_gripper_velocity_path",
            "command_joint_position_path", "command_gripper_position_path",
            "timestamp_path", "control_start_path", "skip_action_path",
            "controller_success_path", "controller_failure_path",
        )
        with h5py.File(trajectory_path, "r") as handle:
            missing = [schema[name] for name in required_paths if schema[name] not in handle]
            if missing:
                raise ValueError(f"{lab}/{outcome} lacks required HDF5 paths: {missing}")
            root_success = bool(handle.attrs.get("success", False))
            root_failure = bool(handle.attrs.get("failure", False))
            if root_success == root_failure or root_success != expected_success:
                raise ValueError(f"{lab}/{outcome} HDF5 root outcome disagrees with its path")
            if str(handle.attrs.get("current_task", "")).strip() != task:
                raise ValueError(f"{lab}/{outcome} task differs between JSON and HDF5")
            length = int(handle[schema["state_joint_path"]].shape[0])
            if length != int(metadata["trajectory_length"]):
                raise ValueError(f"{lab}/{outcome} trajectory length differs from metadata")
            wrong_lengths: list[str] = []

            def check_length(name: str, value: Any) -> None:
                if isinstance(value, h5py.Dataset) and value.ndim and value.shape[0] != length:
                    wrong_lengths.append(name)

            handle.visititems(check_length)
            if wrong_lengths:
                raise ValueError(f"{lab}/{outcome} contains unaligned HDF5 arrays")
            arrays = {
                "joint_position": np.asarray(handle[schema["state_joint_path"]]),
                "joint_velocity": np.asarray(handle[schema["state_joint_velocity_path"]]),
                "gripper_position": np.asarray(handle[schema["state_gripper_path"]]),
                "cartesian_velocity": np.asarray(handle[schema["action_cartesian_velocity_path"]]),
                "gripper_velocity": np.asarray(handle[schema["action_gripper_velocity_path"]]),
                "joint_command": np.asarray(handle[schema["command_joint_position_path"]]),
                "gripper_command": np.asarray(handle[schema["command_gripper_position_path"]]),
            }
            expected_shapes = {
                "joint_position": (length, int(schema["expected_joint_width"])),
                "joint_velocity": (length, int(schema["expected_joint_width"])),
                "gripper_position": (length,),
                "cartesian_velocity": (length, 6),
                "gripper_velocity": (length,),
                "joint_command": (length, int(schema["expected_joint_width"])),
                "gripper_command": (length,),
            }
            if any(arrays[name].shape != shape for name, shape in expected_shapes.items()):
                raise ValueError(f"{lab}/{outcome} core array shapes differ from the registry")
            if any(not np.isfinite(value).all() for value in arrays.values()):
                raise ValueError(f"{lab}/{outcome} contains non-finite state or action values")
            timestamps = np.asarray(handle[schema["timestamp_path"]], dtype=np.float64)
            deltas = np.diff(timestamps)
            if length < 2 or not np.isfinite(timestamps).all() or not np.all(deltas > 0):
                raise ValueError(f"{lab}/{outcome} control timestamps are not strictly increasing")
            camera_group = handle["observation/camera_type"]
            if set(camera_group.keys()) != set(serials):
                raise ValueError(f"{lab}/{outcome} camera serials differ between metadata and HDF5")
            camera_roles = (
                ("wrist", serials[0], "wrist_cam_extrinsics", 0),
                ("exterior_1", serials[1], "ext1_cam_extrinsics", 1),
                ("exterior_2", serials[2], "ext2_cam_extrinsics", 1),
            )
            camera_extrinsics: dict[str, np.ndarray[Any, Any]] = {}
            for role, serial, metadata_key, expected_type in camera_roles:
                camera_types = np.asarray(camera_group[serial], dtype=np.int64)
                if not np.all(camera_types == expected_type):
                    raise ValueError(f"{lab}/{outcome} {role} camera type is inconsistent")
                extrinsics_path = (
                    f"{schema['camera_extrinsics_group']}/{serial}_left"
                )
                if extrinsics_path not in handle:
                    raise ValueError(f"{lab}/{outcome} lacks {role} left-camera extrinsics")
                extrinsics = np.asarray(handle[extrinsics_path], dtype=np.float64)
                metadata_extrinsics = np.asarray(metadata[metadata_key], dtype=np.float64)
                if extrinsics.shape != (length, 6) or metadata_extrinsics.shape != (6,):
                    raise ValueError(f"{lab}/{outcome} {role} extrinsics must be [N,6]")
                if not np.isfinite(extrinsics).all() or not np.isfinite(metadata_extrinsics).all():
                    raise ValueError(f"{lab}/{outcome} {role} extrinsics are non-finite")
                if not np.allclose(metadata_extrinsics, extrinsics[0], rtol=0.0, atol=1e-9):
                    raise ValueError(
                        f"{lab}/{outcome} {role} metadata extrinsics differ from HDF5"
                    )
                camera_extrinsics[role] = extrinsics
            controller_success = np.asarray(handle[schema["controller_success_path"]], dtype=bool)
            controller_failure = np.asarray(handle[schema["controller_failure_path"]], dtype=bool)
            if np.any(controller_success & controller_failure):
                raise ValueError(f"{lab}/{outcome} controller flags contain simultaneous outcomes")
            skip_actions = np.asarray(handle[schema["skip_action_path"]], dtype=bool)
            state = np.column_stack((arrays["joint_position"], arrays["gripper_position"]))
            native_action = np.column_stack((arrays["cartesian_velocity"], arrays["gripper_velocity"]))
            command = np.column_stack((arrays["joint_command"], arrays["gripper_command"]))
            return {
                "qualification_id": f"{self.spec.dataset_id}:{lab}:{outcome}",
                "lab": lab,
                "outcome": outcome,
                "task": task,
                "frames": length,
                "transitions": length - 1,
                "skip_action_transitions": int(skip_actions[:-1].sum()),
                "controller_terminal_success": bool(controller_success[-1]),
                "controller_terminal_failure": bool(controller_failure[-1]),
                "controller_terminal_matches_authoritative": bool(
                    controller_success[-1] == expected_success
                    and controller_failure[-1] == (not expected_success)
                ),
                "dt_seconds_min": float(deltas.min() / 1000.0),
                "dt_seconds_median": float(np.median(deltas) / 1000.0),
                "dt_seconds_max": float(deltas.max() / 1000.0),
                "state_min": state.min(axis=0).tolist(),
                "state_max": state.max(axis=0).tolist(),
                "native_action_min": native_action.min(axis=0).tolist(),
                "native_action_max": native_action.max(axis=0).tolist(),
                "derived_command_min": command.min(axis=0).tolist(),
                "derived_command_max": command.max(axis=0).tolist(),
                "camera_extrinsics_min": {
                    role: values.min(axis=0).tolist()
                    for role, values in camera_extrinsics.items()
                },
                "camera_extrinsics_max": {
                    role: values.max(axis=0).tolist()
                    for role, values in camera_extrinsics.items()
                },
            }

    def validate_source(self) -> dict[str, Any]:
        episode_uuids = [
            str(self._metadata(lab, outcome).get("uuid", "")).strip()
            for lab in self.spec.labs
            for outcome in OUTCOMES
        ]
        if any(not value for value in episode_uuids):
            raise ValueError("a DROID qualification episode lacks its source UUID")
        if len(set(episode_uuids)) != len(episode_uuids):
            raise ValueError("the DROID qualification subset contains duplicate episodes")
        cells = [self._inspect_cell(lab, outcome) for lab in self.spec.labs for outcome in OUTCOMES]
        frames = sum(int(cell["frames"]) for cell in cells)
        transitions = sum(int(cell["transitions"]) for cell in cells)
        dts_min = [float(cell["dt_seconds_min"]) for cell in cells]
        dts_median = [float(cell["dt_seconds_median"]) for cell in cells]
        dts_max = [float(cell["dt_seconds_max"]) for cell in cells]

        def bounds(key: str, reduction: Any) -> list[float]:
            return reduction(np.asarray([cell[key] for cell in cells]), axis=0).tolist()

        return {
            "episodes": len(cells),
            "frames": frames,
            "transitions": transitions,
            "labs": len({cell["lab"] for cell in cells}),
            "outcomes": dict(Counter(str(cell["outcome"]) for cell in cells)),
            "tasks": len({cell["task"] for cell in cells}),
            "skip_action_transitions": sum(int(cell["skip_action_transitions"]) for cell in cells),
            "stale_controller_terminal_labels": sum(
                not bool(cell["controller_terminal_matches_authoritative"]) for cell in cells
            ),
            "state_width": int(self.spec.schema["canonical_state_width"]),
            "native_action_width": int(self.spec.schema["native_action_width"]),
            "state_min": bounds("state_min", np.min),
            "state_max": bounds("state_max", np.max),
            "native_action_min": bounds("native_action_min", np.min),
            "native_action_max": bounds("native_action_max", np.max),
            "derived_command_min": bounds("derived_command_min", np.min),
            "derived_command_max": bounds("derived_command_max", np.max),
            "dt_seconds_min": min(dts_min),
            "dt_seconds_median_of_episode_medians": float(np.median(dts_median)),
            "dt_seconds_max": max(dts_max),
            "null_values": 0,
            "nonfinite_values": 0,
            "duplicate_episodes": 0,
            "redacted_fields": sorted(PRIVATE_METADATA_FIELDS),
            "authoritative_outcome_sources": [
                "bucket_path", "metadata.success", "hdf5_root.success", "hdf5_root.failure"
            ],
            "controller_info_policy": "audit_only",
            "camera_calibration": {
                "preserved": True,
                "roles": list(self.spec.camera_names),
                "source_eye": "left",
                "vector": "[x, y, z, Euler-Rx, Euler-Ry, Euler-Rz]",
                "translation_unit": "meter",
                "rotation_unit": "radian_euler_xyz",
                "target_frame": "robot_base",
                "per_observation": True,
            },
            "native_action_semantics": self.spec.native_action_semantics,
            "task_space_conversion_allowed": False,
            "cells": cells,
        }

    def video_object(self, lab: str, outcome: str, camera: str) -> DROIDObject:
        if camera not in self.spec.cameras:
            raise KeyError(f"unknown camera {camera!r}")
        matches = tuple(
            item for item in self._objects_for(lab, outcome, "video") if item.camera == camera
        )
        if len(matches) != 1:
            raise ValueError(f"{lab}/{outcome}/{camera} has no unique qualified video")
        return matches[0]

    @staticmethod
    def probe_video(path: Path) -> dict[str, Any]:
        av = _video()
        with av.open(str(path), mode="r") as container:
            if not container.streams.video:
                raise ValueError(f"no video stream in {path}")
            stream = container.streams.video[0]
            context = stream.codec_context
            duration = (
                float(stream.duration * stream.time_base)
                if stream.duration is not None
                else float(container.duration / av.time_base)
            )
            return {
                "codec": context.name,
                "pixel_format": context.pix_fmt or (context.format.name if context.format else None),
                "width": int(context.width),
                "height": int(context.height),
                "fps": float(stream.average_rate),
                "frames": int(stream.frames) if stream.frames else None,
                "duration_seconds": duration,
            }

    def validate_video(self, lab: str, outcome: str, camera: str) -> dict[str, Any]:
        item = self.video_object(lab, outcome, camera)
        probe = self.probe_video(self.root / item.path)
        expected = self.spec.cameras[camera]
        expected_values = {
            "codec": expected.codec,
            "pixel_format": expected.pixel_format,
            "width": expected.width,
            "height": expected.height,
        }
        for key, value in expected_values.items():
            if probe[key] != value:
                raise ValueError(f"{lab}/{outcome}/{camera} {key} differs from the registry")
        if not np.isclose(probe["fps"], expected.container_fps, rtol=0.0, atol=1e-6):
            raise ValueError(f"{lab}/{outcome}/{camera} declared fps differs from the registry")
        cell = self._inspect_cell(lab, outcome)
        if probe["frames"] != int(cell["transitions"]):
            raise ValueError(f"{lab}/{outcome}/{camera} frame count is not N-1")
        return probe | {"alignment": "frame_index", "trajectory_frames": int(cell["frames"])}

    @staticmethod
    def decode_rgb_frame(path: Path, frame_index: int) -> torch.Tensor:
        if frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        av = _video()
        with av.open(str(path), mode="r") as container:
            stream = container.streams.video[0]
            for index, frame in enumerate(container.decode(stream)):
                if index == frame_index:
                    array = frame.to_ndarray(format="rgb24")
                    return torch.from_numpy(array.copy()).permute(2, 0, 1)
        raise ValueError(f"video has no frame at index {frame_index}")

    def canonical_episode(
        self,
        lab: str,
        outcome: str,
        *,
        max_transitions: int | None = None,
        include_rgb: bool = False,
    ) -> CanonicalEpisode:
        h5py = _hdf5()
        cell = self._inspect_cell(lab, outcome)
        metadata_path, trajectory_path = self._episode_files(lab, outcome)
        del metadata_path
        available = int(cell["transitions"])
        transitions = available if max_transitions is None else min(max_transitions, available)
        if transitions < 1:
            raise ValueError("canonicalization requires at least one transition")
        count = transitions + 1
        schema = self.spec.schema
        metadata = self._metadata(lab, outcome)
        serial_by_role = {
            "wrist": str(metadata["wrist_cam_serial"]),
            "exterior_1": str(metadata["ext1_cam_serial"]),
            "exterior_2": str(metadata["ext2_cam_serial"]),
        }
        with h5py.File(trajectory_path, "r") as handle:
            q = np.column_stack((
                np.asarray(handle[schema["state_joint_path"]][:count], dtype=np.float64),
                np.asarray(handle[schema["state_gripper_path"]][:count], dtype=np.float64),
            ))
            source_joint_velocities = np.asarray(
                handle[schema["state_joint_velocity_path"]][:count],
                dtype=np.float64,
            )
            timestamps_ms = np.asarray(handle[schema["timestamp_path"]][:count], dtype=np.float64)
            control_start_ms = np.asarray(
                handle[schema["control_start_path"]][:count], dtype=np.int64
            )
            timestamps = (timestamps_ms - timestamps_ms[0]) / 1000.0
            gripper_velocity = np.empty(count, dtype=np.float64)
            gripper_velocity[1:] = np.diff(q[:, -1]) / np.diff(timestamps)
            gripper_velocity[0] = gripper_velocity[1]
            qdot = np.column_stack((source_joint_velocities, gripper_velocity))
            commands = np.column_stack((
                np.asarray(handle[schema["command_joint_position_path"]][:count], dtype=np.float64),
                np.asarray(handle[schema["command_gripper_position_path"]][:count], dtype=np.float64),
            ))
            previous_commands = np.empty_like(q)
            previous_commands[0] = q[0]
            previous_commands[1:] = commands[:transitions]
            native_actions = np.column_stack((
                np.asarray(handle[schema["action_cartesian_velocity_path"]][:transitions], dtype=np.float64),
                np.asarray(handle[schema["action_gripper_velocity_path"]][:transitions], dtype=np.float64),
            ))
            skipped = np.asarray(handle[schema["skip_action_path"]][:transitions], dtype=bool)
            camera_extrinsics = {
                role: np.asarray(
                    handle[
                        f"{schema['camera_extrinsics_group']}/{serial}_left"
                    ][:count],
                    dtype=np.float64,
                )
                for role, serial in serial_by_role.items()
            }
            camera_capture_ms = {
                role: np.asarray(
                    handle[
                        f"observation/timestamp/cameras/{serial}_estimated_capture"
                    ][:count],
                    dtype=np.int64,
                )
                for role, serial in serial_by_role.items()
            }

        rgb_by_observation: list[dict[str, torch.Tensor]] = [dict() for _ in range(count)]
        if include_rgb:
            for camera in self.spec.camera_names:
                item = self.video_object(lab, outcome, camera)
                path = self.root / item.path
                for index in range(transitions):
                    rgb_by_observation[index][camera] = self.decode_rgb_frame(path, index)

        observations = tuple(
            Observation(
                timestamp=float(timestamps[index]),
                q=torch.tensor(q[index], dtype=torch.float32),
                qdot=torch.tensor(qdot[index], dtype=torch.float32),
                previous_command=torch.tensor(previous_commands[index], dtype=torch.float32),
                rgb=rgb_by_observation[index],
                validity={
                    "state": True,
                    "action_executed": bool(index >= transitions or not skipped[index]),
                    **{
                        f"rgb_{camera}": camera in rgb_by_observation[index]
                        for camera in self.spec.camera_names
                    },
                },
            )
            for index in range(count)
        )
        actions = tuple(
            Action(
                timestamp=float(timestamps[index]),
                native=torch.tensor(native_actions[index], dtype=torch.float32),
            )
            for index in range(transitions)
        )
        success = outcome == "success"
        episode = CanonicalEpisode(
            metadata=EpisodeMetadata(
                episode_id=str(cell["qualification_id"]),
                source_dataset=self.spec.dataset_id,
                source_version=self.spec.revision,
                source_url=self.spec.source_url,
                license=self.spec.license,
                redistribution_terms=self.spec.redistribution_terms,
                domain=self.spec.domain,
                robot_id=self.spec.robot_id,
                embodiment=self.spec.embodiment,
                task=str(cell["task"]),
                success=success,
                quality=1.0 if success else 0.0,
                collection_method=self.spec.collection_method,
                native_action_semantics=self.spec.native_action_semantics,
                source_policy="human_teleoperation",
                fps=self.spec.nominal_control_hz,
                camera_names=self.spec.camera_names,
                extra={
                    "lab": lab,
                    "outcome": outcome,
                    "source_frames": int(cell["frames"]),
                    "canonical_transitions": transitions,
                    "skip_action_transitions": int(skipped.sum()),
                    "controller_terminal_matches_authoritative": bool(
                        cell["controller_terminal_matches_authoritative"]
                    ),
                    "imitation_eligible": success,
                    "imitation_weight": 1.0 if success else 0.0,
                    "prediction_eligible": True,
                    "prediction_weight": 1.0,
                    "eligible_modules": [
                        "jepa_encoder",
                        "jepa_world",
                        "executive",
                        "language",
                        "universal_action_pretraining",
                    ],
                    "language_annotation_source": "metadata.current_task",
                    "unit_conventions": self.spec.units,
                    "coordinate_frames": self.spec.coordinate_frames,
                },
            ),
            observations=observations,
            actions=actions,
            language=(str(cell["task"]),),
            scene_metadata=tuple(
                {
                    "source_frame_index": index,
                    "source_timestamps": {
                        "control_step_start_unix_ms": int(timestamps_ms[index]),
                        "control_start_unix_ms": int(control_start_ms[index]),
                        "camera_estimated_capture_unix_ms": {
                            role: int(values[index])
                            for role, values in camera_capture_ms.items()
                        },
                    },
                    "source_recorded_joint_velocity_rad_s": source_joint_velocities[
                        index
                    ].tolist(),
                    "camera_extrinsics": {
                        role: {
                            "translation_m": values[index, :3].tolist(),
                            "rotation_euler_xyz_rad": values[index, 3:].tolist(),
                            "source_eye": "left",
                            "target_frame": "robot_base",
                        }
                        for role, values in camera_extrinsics.items()
                    }
                }
                for index in range(count)
            ),
        )
        episode.validate()
        return episode
