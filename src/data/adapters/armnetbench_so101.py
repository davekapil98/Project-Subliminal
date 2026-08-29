"""Pinned LeRobot-v3 adapter for the ArmnetBench v0.1 SO-101 release."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import tomllib
from typing import Any

import numpy as np
import torch

from data.canonical_schema import Action, CanonicalEpisode, EpisodeMetadata, Observation


REQUIRED_DATA_COLUMNS = (
    "observation.state",
    "action",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
    "next.reward",
    "next.done",
)

REQUIRED_EPISODE_COLUMNS = (
    "episode_index",
    "tasks",
    "length",
    "data/chunk_index",
    "data/file_index",
    "dataset_from_index",
    "dataset_to_index",
    "success",
    "success_class",
    "policy_repo_id",
    "policy_type",
)


def _arrow() -> tuple[Any, Any]:
    try:
        import pyarrow as arrow
        import pyarrow.parquet as parquet
    except ImportError as error:  # pragma: no cover - exercised by packaging, not data tests
        raise RuntimeError("ArmnetBench Parquet support requires the 'data' optional dependency") from error
    return arrow, parquet


def _video() -> Any:
    try:
        import av
    except ImportError as error:  # pragma: no cover - exercised by packaging, not data tests
        raise RuntimeError("ArmnetBench video support requires the 'data' optional dependency") from error
    return av


@dataclass(frozen=True)
class QualifiedObject:
    path: str
    size: int
    sha256: str
    role: str


@dataclass(frozen=True)
class CameraSpec:
    name: str
    feature_key: str
    width: int
    height: int
    codec: str
    pixel_format: str
    fps: float


@dataclass(frozen=True)
class VideoSegment:
    camera_name: str
    path: Path
    from_timestamp: float
    to_timestamp: float
    width: int
    height: int
    fps: float


@dataclass(frozen=True)
class ArmnetBenchSourceSpec:
    dataset_id: str
    repository_id: str
    revision: str
    release_tag: str
    source_url: str
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
    fps: float
    total_episodes: int
    total_frames: int
    total_tasks: int
    full_snapshot_bytes: int
    hub_used_storage_bytes: int
    modalities: tuple[str, ...]
    task_families: tuple[str, ...]
    camera_names: tuple[str, ...]
    outcome_classes: tuple[str, ...]
    policy_types: tuple[str, ...]
    known_limitations: tuple[str, ...]
    units: dict[str, str]
    coordinate_frames: dict[str, str]
    joint_names: tuple[str, ...]
    cameras: dict[str, CameraSpec]
    outcome_counts: dict[str, int]
    qualified_episode_indices: tuple[int, ...]
    test_policy_offset: int
    validation_policy_offset: int
    object_manifest_path: Path
    object_manifest_sha256: str
    qualified_objects: tuple[QualifiedObject, ...]

    @classmethod
    def from_toml(cls, path: Path) -> "ArmnetBenchSourceSpec":
        with path.open("rb") as handle:
            raw = tomllib.load(handle)
        dataset = raw["dataset"]
        project_root = path.resolve().parents[3]
        object_manifest_path = project_root / raw["objects"]["manifest_path"]
        object_manifest = json.loads(object_manifest_path.read_text(encoding="utf-8"))
        cameras = {
            name: CameraSpec(name=name, **values) for name, values in raw["cameras"].items()
        }
        return cls(
            dataset_id=dataset["dataset_id"],
            repository_id=dataset["repository_id"],
            revision=dataset["revision"],
            release_tag=dataset["release_tag"],
            source_url=dataset["source_url"],
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
            fps=float(dataset["fps"]),
            total_episodes=int(dataset["total_episodes"]),
            total_frames=int(dataset["total_frames"]),
            total_tasks=int(dataset["total_tasks"]),
            full_snapshot_bytes=int(dataset["full_snapshot_bytes"]),
            hub_used_storage_bytes=int(dataset["hub_used_storage_bytes"]),
            modalities=tuple(dataset["modalities"]),
            task_families=tuple(dataset["task_families"]),
            camera_names=tuple(dataset["camera_names"]),
            outcome_classes=tuple(dataset["outcome_classes"]),
            policy_types=tuple(dataset["policy_types"]),
            known_limitations=tuple(dataset["known_limitations"]),
            units=dict(raw["units"]),
            coordinate_frames=dict(raw["coordinate_frames"]),
            joint_names=tuple(raw["schema"]["joint_names"]),
            cameras=cameras,
            outcome_counts={name: int(raw["outcomes"][name]) for name in dataset["outcome_classes"]},
            qualified_episode_indices=tuple(raw["qualified_subset"]["episode_indices"]),
            test_policy_offset=int(raw["splitting"]["test_policy_offset"]),
            validation_policy_offset=int(raw["splitting"]["validation_policy_offset"]),
            object_manifest_path=object_manifest_path,
            object_manifest_sha256=raw["objects"]["manifest_sha256"],
            qualified_objects=tuple(QualifiedObject(**item) for item in object_manifest["objects"]),
        )


class ArmnetBenchSO101Adapter:
    """Validate ArmnetBench data and preserve outcome/policy semantics in canonical episodes."""

    def __init__(self, root: Path, spec: ArmnetBenchSourceSpec) -> None:
        self.root = root
        self.spec = spec
        self._episodes_cache: Any | None = None
        self._tasks_cache: dict[int, str] | None = None
        self._data_cache: Any | None = None

    @classmethod
    def from_registry(cls, root: Path, registry_path: Path) -> "ArmnetBenchSO101Adapter":
        return cls(root, ArmnetBenchSourceSpec.from_toml(registry_path))

    def _read_table(self, relative_path: str, *, columns: tuple[str, ...] | None = None) -> Any:
        _, parquet = _arrow()
        return parquet.read_table(self.root / relative_path, columns=list(columns) if columns else None)

    def _episode_table(self) -> Any:
        if self._episodes_cache is None:
            self._episodes_cache = self._read_table("meta/episodes/chunk-000/file-000.parquet")
        return self._episodes_cache

    def _trajectory_objects(self) -> tuple[QualifiedObject, ...]:
        return tuple(item for item in self.spec.qualified_objects if item.role == "trajectory_table")

    def _data_table(self) -> Any:
        if self._data_cache is None:
            arrow, parquet = _arrow()
            tables = [
                parquet.read_table(self.root / item.path, columns=list(REQUIRED_DATA_COLUMNS))
                for item in self._trajectory_objects()
            ]
            self._data_cache = arrow.concat_tables(tables)
        return self._data_cache

    def tasks(self) -> dict[int, str]:
        if self._tasks_cache is None:
            rows = self._read_table("meta/tasks.parquet").to_pylist()
            self._tasks_cache = {int(row["task_index"]): str(row["task"]) for row in rows}
        return self._tasks_cache

    def episode_record(self, episode_index: int) -> dict[str, Any]:
        if not 0 <= episode_index < self.spec.total_episodes:
            raise IndexError(f"episode_index must be in [0, {self.spec.total_episodes})")
        record = self._episode_table().slice(episode_index, 1).to_pylist()[0]
        if int(record["episode_index"]) != episode_index:
            raise ValueError("episode metadata rows are not ordered by episode_index")
        return record

    def episode_records(self) -> list[dict[str, Any]]:
        return self._episode_table().to_pylist()

    def _validate_video_metadata(
        self, episodes: Any, lengths: np.ndarray
    ) -> dict[str, dict[str, int | float]]:
        results: dict[str, dict[str, int | float]] = {}
        for camera_name in self.spec.camera_names:
            prefix = f"videos/observation.images.{camera_name}"
            chunk_indices = np.asarray(episodes[f"{prefix}/chunk_index"].to_numpy(), dtype=np.int64)
            file_indices = np.asarray(episodes[f"{prefix}/file_index"].to_numpy(), dtype=np.int64)
            starts = np.asarray(episodes[f"{prefix}/from_timestamp"].to_numpy(), dtype=np.float64)
            stops = np.asarray(episodes[f"{prefix}/to_timestamp"].to_numpy(), dtype=np.float64)
            if not np.array_equal(chunk_indices, np.zeros(self.spec.total_episodes, dtype=np.int64)):
                raise ValueError(f"{camera_name} video chunks differ from the pinned single-chunk layout")
            if not np.allclose(stops - starts, lengths / self.spec.fps, rtol=0.0, atol=1e-8):
                raise ValueError(f"{camera_name} video segment durations do not match episode lengths")
            previous_stop: dict[int, float] = {}
            gap_count = 0
            maximum_gap_seconds = 0.0
            for file_index, start, stop in zip(file_indices, starts, stops, strict=True):
                if file_index in previous_stop:
                    difference = float(start) - previous_stop[int(file_index)]
                    if difference < -1e-8:
                        raise ValueError(f"{camera_name} packed-video segments overlap")
                    if difference > 1e-8:
                        gap_count += 1
                        maximum_gap_seconds = max(maximum_gap_seconds, difference)
                if file_index not in previous_stop and not np.isclose(start, 0.0, rtol=0.0, atol=1e-8):
                    raise ValueError(f"{camera_name} packed-video file does not start at timestamp zero")
                previous_stop[int(file_index)] = float(stop)
            results[camera_name] = {
                "file_count": len(previous_stop),
                "unreferenced_gap_count": gap_count,
                "maximum_unreferenced_gap_seconds": maximum_gap_seconds,
            }
        return results

    def validate_source(self) -> dict[str, Any]:
        """Run full metadata/trajectory integrity checks and outcome-label validation."""

        info = json.loads((self.root / "meta/info.json").read_text(encoding="utf-8"))
        expected_info = {
            "codebase_version": "v3.0",
            "robot_type": self.spec.robot_id,
            "total_episodes": self.spec.total_episodes,
            "total_frames": self.spec.total_frames,
            "total_tasks": self.spec.total_tasks,
            "fps": int(self.spec.fps),
        }
        for key, expected in expected_info.items():
            if info.get(key) != expected:
                raise ValueError(f"meta/info.json {key!r} drifted: {info.get(key)!r} != {expected!r}")
        features = info.get("features", {})
        if tuple(features["observation.state"]["names"]) != self.spec.joint_names:
            raise ValueError("state joint order differs from the pinned registry")
        if tuple(features["action"]["names"]) != self.spec.joint_names:
            raise ValueError("action joint order differs from the pinned registry")
        if features["next.reward"]["dtype"] != "float32" or features["next.done"]["dtype"] != "bool":
            raise ValueError("reward/done feature types differ from the pinned schema")
        for camera_name, camera in self.spec.cameras.items():
            feature = features[camera.feature_key]
            published = feature["info"]
            expected_camera = {
                "video.width": camera.width,
                "video.height": camera.height,
                "video.codec": camera.codec,
                "video.pix_fmt": camera.pixel_format,
                "video.fps": int(camera.fps),
            }
            for key, expected in expected_camera.items():
                if published.get(key) != expected:
                    raise ValueError(f"{camera_name} {key} differs from the pinned registry")

        episodes = self._episode_table()
        tasks = self.tasks()
        if episodes.num_rows != self.spec.total_episodes:
            raise ValueError("episode count differs from the pinned registry")
        if len(tasks) != self.spec.total_tasks or set(tasks) != set(range(self.spec.total_tasks)):
            raise ValueError("task indices must be unique and contiguous")
        for column in REQUIRED_EPISODE_COLUMNS:
            if episodes[column].null_count:
                raise ValueError(f"episode metadata column {column!r} contains nulls")

        episode_indices = np.asarray(episodes["episode_index"].to_numpy(), dtype=np.int64)
        lengths = np.asarray(episodes["length"].to_numpy(), dtype=np.int64)
        from_indices = np.asarray(episodes["dataset_from_index"].to_numpy(), dtype=np.int64)
        to_indices = np.asarray(episodes["dataset_to_index"].to_numpy(), dtype=np.int64)
        if not np.array_equal(episode_indices, np.arange(self.spec.total_episodes)):
            raise ValueError("episode indices are not unique, ordered and contiguous")
        if lengths.sum() != self.spec.total_frames:
            raise ValueError("episode lengths do not sum to total_frames")
        if from_indices[0] != 0 or to_indices[-1] != self.spec.total_frames:
            raise ValueError("episode ranges do not cover the full trajectory table")
        if not np.array_equal(from_indices[1:], to_indices[:-1]):
            raise ValueError("episode ranges contain gaps or overlaps")
        if not np.array_equal(to_indices - from_indices, lengths):
            raise ValueError("episode range widths differ from declared lengths")

        metadata_tasks = episodes["tasks"].to_pylist()
        if any(len(values) != 1 for values in metadata_tasks):
            raise ValueError("each episode must contain exactly one canonical task instruction")
        reverse_tasks = {text: index for index, text in tasks.items()}
        if len(reverse_tasks) != self.spec.total_tasks:
            raise ValueError("canonical task texts must be unique")
        try:
            metadata_task_indices = np.asarray(
                [reverse_tasks[values[0]] for values in metadata_tasks], dtype=np.int64
            )
        except KeyError as error:
            raise ValueError("episode metadata contains an unknown canonical task") from error

        success = np.asarray(episodes["success"].to_numpy(), dtype=np.int64)
        success_classes = [str(value) for value in episodes["success_class"].to_pylist()]
        policy_types = [str(value) for value in episodes["policy_type"].to_pylist()]
        policy_repo_ids = [str(value) for value in episodes["policy_repo_id"].to_pylist()]
        if set(success_classes) != set(self.spec.outcome_classes):
            raise ValueError("outcome classes differ from the pinned registry")
        outcome_counts = Counter(success_classes)
        if dict(outcome_counts) != self.spec.outcome_counts:
            raise ValueError("outcome class counts differ from the pinned registry")
        expected_success = np.asarray(
            [value == "successful" for value in success_classes], dtype=np.int64
        )
        if not np.array_equal(success, expected_success):
            raise ValueError("binary success must be one only for strict successful outcomes")
        if set(policy_types) != set(self.spec.policy_types):
            raise ValueError("policy types differ from the pinned registry")
        for policy_type, policy_repo_id, outcome in zip(
            policy_types, policy_repo_ids, success_classes, strict=True
        ):
            if policy_type == "teleoperated":
                if policy_repo_id or outcome != "successful":
                    raise ValueError("teleoperated episodes must be successful with an empty policy repo")
            elif not policy_repo_id:
                raise ValueError("learned-policy episodes require policy_repo_id")

        video_metadata = self._validate_video_metadata(episodes, lengths)

        data = self._data_table()
        if data.num_rows != self.spec.total_frames:
            raise ValueError("trajectory row count differs from the pinned registry")
        expected_types = {
            "observation.state": "list<element: float>",
            "action": "list<element: float>",
            "timestamp": "float",
            "frame_index": "int64",
            "episode_index": "int64",
            "index": "int64",
            "task_index": "int64",
            "next.reward": "float",
            "next.done": "bool",
        }
        for column, expected_type in expected_types.items():
            actual_type = str(data.schema.field(column).type)
            if actual_type != expected_type:
                raise ValueError(f"{column} type drifted: {actual_type!r} != {expected_type!r}")
            if data[column].null_count:
                raise ValueError(f"{column} contains nulls")

        episode_index = np.asarray(data["episode_index"].to_numpy(), dtype=np.int64)
        frame_index = np.asarray(data["frame_index"].to_numpy(), dtype=np.int64)
        global_index = np.asarray(data["index"].to_numpy(), dtype=np.int64)
        task_index = np.asarray(data["task_index"].to_numpy(), dtype=np.int64)
        timestamps = np.asarray(data["timestamp"].to_numpy(), dtype=np.float64)
        rewards = np.asarray(data["next.reward"].to_numpy(), dtype=np.float32)
        done = np.asarray(data["next.done"].to_numpy(), dtype=np.bool_)
        expected_episode = np.repeat(np.arange(self.spec.total_episodes), lengths)
        expected_frame = global_index - np.repeat(from_indices, lengths)
        if not np.array_equal(global_index, np.arange(self.spec.total_frames)):
            raise ValueError("global trajectory index is not unique and contiguous")
        if not np.array_equal(episode_index, expected_episode):
            raise ValueError("trajectory rows are not grouped by declared episode ranges")
        if not np.array_equal(frame_index, expected_frame):
            raise ValueError("frame indices do not restart at zero for each episode")
        if not np.array_equal(task_index, np.repeat(metadata_task_indices, lengths)):
            raise ValueError("trajectory task indices differ from episode task text")
        if not np.allclose(timestamps, frame_index / self.spec.fps, rtol=0.0, atol=1e-5):
            raise ValueError("timestamps are not aligned to frame_index/fps")

        states = np.asarray(data["observation.state"].to_pylist(), dtype=np.float32)
        actions = np.asarray(data["action"].to_pylist(), dtype=np.float32)
        expected_shape = (self.spec.total_frames, len(self.spec.joint_names))
        if states.shape != expected_shape or actions.shape != expected_shape:
            raise ValueError(f"state/action arrays must both have shape {expected_shape}")
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise ValueError("state/action arrays contain NaN or Inf")
        if not np.isfinite(rewards).all() or not np.isin(rewards, (0.0, 1.0)).all():
            raise ValueError("rewards must be finite binary values")

        expected_done = frame_index == np.repeat(lengths - 1, lengths)
        if not np.array_equal(done, expected_done):
            raise ValueError("next.done must be true only on each final kept frame")
        expected_reward = done & np.repeat(success.astype(bool), lengths)
        if not np.array_equal(rewards, expected_reward.astype(np.float32)):
            raise ValueError("next.reward must be one only on terminal strict-success frames")

        within_episode = frame_index > 0
        action_delta = np.abs(actions[within_episode] - actions[np.flatnonzero(within_episode) - 1])
        stats = json.loads((self.root / "meta/stats.json").read_text(encoding="utf-8"))
        published_counts = {
            name: int(stats[name]["count"][0])
            for name in ("observation.state", "action", "timestamp", "episode_index", "index")
        }
        mismatched_published_counts = {
            name: count for name, count in published_counts.items() if count != self.spec.total_frames
        }
        return {
            "episodes": int(episodes.num_rows),
            "frames": int(data.num_rows),
            "tasks": len(tasks),
            "fps": self.spec.fps,
            "state_width": states.shape[1],
            "action_width": actions.shape[1],
            "state_min": states.min(axis=0).tolist(),
            "state_max": states.max(axis=0).tolist(),
            "action_min": actions.min(axis=0).tolist(),
            "action_max": actions.max(axis=0).tolist(),
            "max_abs_action_step": action_delta.max(axis=0).tolist(),
            "outcome_counts": dict(sorted(outcome_counts.items())),
            "policy_counts": dict(sorted(Counter(policy_types).items())),
            "task_episode_counts": {
                self.spec.task_families[index]: int((metadata_task_indices == index).sum())
                for index in range(self.spec.total_tasks)
            },
            "terminal_done_count": int(done.sum()),
            "terminal_success_reward_count": int(rewards.sum()),
            "video_metadata": video_metadata,
            "null_values": 0,
            "nonfinite_values": 0,
            "published_statistics_counts": published_counts,
            "published_statistics_mismatched_counts": mismatched_published_counts,
            "published_statistics_trusted": False,
        }

    def video_segment(self, episode_index: int, camera_name: str) -> VideoSegment:
        if camera_name not in self.spec.cameras:
            raise KeyError(f"unknown camera {camera_name!r}")
        record = self.episode_record(episode_index)
        prefix = f"videos/observation.images.{camera_name}"
        chunk_index = int(record[f"{prefix}/chunk_index"])
        file_index = int(record[f"{prefix}/file_index"])
        camera = self.spec.cameras[camera_name]
        return VideoSegment(
            camera_name=camera_name,
            path=self.root / prefix / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.mp4",
            from_timestamp=float(record[f"{prefix}/from_timestamp"]),
            to_timestamp=float(record[f"{prefix}/to_timestamp"]),
            width=camera.width,
            height=camera.height,
            fps=camera.fps,
        )

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
            raw_codec = context.name
            codec = "av1" if raw_codec in {"av1", "libdav1d"} else raw_codec
            pixel_format = context.pix_fmt or (context.format.name if context.format else None)
            return {
                "codec": codec,
                "decoder": raw_codec,
                "pixel_format": pixel_format,
                "width": int(context.width),
                "height": int(context.height),
                "fps": float(stream.average_rate),
                "frames": int(stream.frames) if stream.frames else None,
                "duration_seconds": duration,
            }

    def validate_video(self, segment: VideoSegment) -> dict[str, Any]:
        camera = self.spec.cameras[segment.camera_name]
        probe = self.probe_video(segment.path)
        expected = {
            "codec": camera.codec,
            "pixel_format": camera.pixel_format,
            "width": camera.width,
            "height": camera.height,
        }
        for key, value in expected.items():
            if probe[key] != value:
                raise ValueError(f"{segment.camera_name} {key} drifted: {probe[key]!r} != {value!r}")
        if not np.isclose(probe["fps"], camera.fps, rtol=0.0, atol=1e-6):
            raise ValueError(f"{segment.camera_name} fps differs from the pinned registry")
        if probe["duration_seconds"] + 1 / camera.fps < segment.to_timestamp:
            raise ValueError(f"{segment.camera_name} video does not cover the episode segment")
        return probe

    @staticmethod
    def decode_rgb_frame(segment: VideoSegment, timestamp: float) -> torch.Tensor:
        if not segment.from_timestamp <= timestamp < segment.to_timestamp:
            raise ValueError("decode timestamp lies outside the episode video segment")
        av = _video()
        with av.open(str(segment.path), mode="r") as container:
            stream = container.streams.video[0]
            container.seek(int(timestamp / stream.time_base), stream=stream, any_frame=False, backward=True)
            for frame in container.decode(stream):
                frame_time = float(frame.pts * frame.time_base) if frame.pts is not None else timestamp
                if frame_time + 0.5 / segment.fps < timestamp:
                    continue
                array = frame.to_ndarray(format="rgb24")
                expected_shape = (segment.height, segment.width, 3)
                if array.shape != expected_shape:
                    raise ValueError(
                        f"decoded {segment.camera_name} frame has shape {array.shape}, expected {expected_shape}"
                    )
                return torch.from_numpy(array.copy()).permute(2, 0, 1)
        raise ValueError(f"could not decode {segment.camera_name} at {timestamp:.9f}s")

    def canonical_episode(
        self,
        episode_index: int,
        *,
        max_transitions: int | None = None,
        include_rgb: bool = False,
    ) -> CanonicalEpisode:
        record = self.episode_record(episode_index)
        start = int(record["dataset_from_index"])
        length = int(record["length"])
        transitions = length - 1 if max_transitions is None else min(max_transitions, length - 1)
        if transitions < 1:
            raise ValueError("canonical samples require at least one transition")
        rows = self._data_table().slice(start, transitions + 1)
        states = np.asarray(rows["observation.state"].to_pylist(), dtype=np.float32)
        source_actions = np.asarray(rows["action"].to_pylist(), dtype=np.float32)
        timestamps = np.asarray(rows["timestamp"].to_numpy(), dtype=np.float64)
        task_indices = np.asarray(rows["task_index"].to_numpy(), dtype=np.int64)
        if not np.all(task_indices == task_indices[0]):
            raise ValueError("task index changes inside an episode")

        qdot = np.zeros_like(states)
        qdot[1:] = np.diff(states, axis=0) / np.diff(timestamps)[:, None]
        qdot[0] = qdot[1]
        previous_commands = np.empty_like(states)
        previous_commands[0] = states[0]
        previous_commands[1:] = source_actions[:-1]

        segments = {
            camera: self.video_segment(episode_index, camera) for camera in self.spec.camera_names
        }
        observations: list[Observation] = []
        for index, timestamp in enumerate(timestamps):
            rgb: dict[str, torch.Tensor] = {}
            if include_rgb:
                for camera, segment in segments.items():
                    rgb[camera] = self.decode_rgb_frame(
                        segment, segment.from_timestamp + float(timestamp)
                    )
            observations.append(
                Observation(
                    timestamp=float(timestamp),
                    q=torch.from_numpy(states[index].copy()),
                    qdot=torch.from_numpy(qdot[index].copy()),
                    previous_command=torch.from_numpy(previous_commands[index].copy()),
                    rgb=rgb,
                    validity={camera: camera in rgb for camera in self.spec.camera_names},
                )
            )

        actions = tuple(
            Action(timestamp=float(timestamps[index]), native=torch.from_numpy(source_actions[index].copy()))
            for index in range(transitions)
        )
        success_class = str(record["success_class"])
        quality = {"successful": 1.0, "suboptimal": 0.5, "failure": 0.0}[success_class]
        policy_type = str(record["policy_type"])
        policy_repo_id = str(record["policy_repo_id"])
        task = self.tasks()[int(task_indices[0])]
        collection_method = (
            "human teleoperation" if policy_type == "teleoperated" else "learned-policy physical rollout"
        )
        metadata = EpisodeMetadata(
            episode_id=f"{self.spec.dataset_id}:{episode_index:06d}",
            source_dataset=self.spec.dataset_id,
            source_version=self.spec.revision,
            source_url=self.spec.source_url,
            license=self.spec.license,
            redistribution_terms=self.spec.redistribution_terms,
            domain=self.spec.domain,
            robot_id=self.spec.robot_id,
            embodiment=self.spec.embodiment,
            task=task,
            success=bool(record["success"]),
            quality=quality,
            collection_method=collection_method,
            native_action_semantics=self.spec.native_action_semantics,
            source_policy=policy_repo_id or policy_type,
            fps=self.spec.fps,
            camera_names=self.spec.camera_names,
            extra={
                "source_episode_index": episode_index,
                "source_task_index": int(task_indices[0]),
                "source_dataset_from_index": start,
                "source_length": length,
                "success_class": success_class,
                "policy_type": policy_type,
                "policy_repo_id": policy_repo_id,
                "imitation_eligible": success_class == "successful",
                "prediction_eligible": True,
                "unit_conventions": self.spec.units,
                "coordinate_frames": self.spec.coordinate_frames,
            },
        )
        episode = CanonicalEpisode(
            metadata=metadata,
            observations=tuple(observations),
            actions=actions,
            language=(task,),
        )
        episode.validate()
        return episode
