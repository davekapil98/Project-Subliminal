"""Pinned LeRobot-v3 adapter for the Project IRA SO-101 dataset."""

from __future__ import annotations

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
)


def _parquet() -> Any:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:  # pragma: no cover - exercised by packaging, not data tests
        raise RuntimeError("Project IRA Parquet support requires the 'data' optional dependency") from error
    return parquet


def _video() -> Any:
    try:
        import av
    except ImportError as error:  # pragma: no cover - exercised by packaging, not data tests
        raise RuntimeError("Project IRA video support requires the 'data' optional dependency") from error
    return av


@dataclass(frozen=True)
class QualifiedFile:
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
class ProjectIRASourceSpec:
    dataset_id: str
    repository_id: str
    revision: str
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
    full_repository_bytes: int
    primary_trajectory_sha256: str
    modalities: tuple[str, ...]
    task_families: tuple[str, ...]
    task_family_ranges: dict[str, tuple[int, int]]
    camera_names: tuple[str, ...]
    known_limitations: tuple[str, ...]
    units: dict[str, str]
    coordinate_frames: dict[str, str]
    joint_names: tuple[str, ...]
    cameras: dict[str, CameraSpec]
    qualified_files: tuple[QualifiedFile, ...]
    qualified_episode_indices: tuple[int, ...]

    @classmethod
    def from_toml(cls, path: Path) -> "ProjectIRASourceSpec":
        with path.open("rb") as handle:
            raw = tomllib.load(handle)
        dataset = raw["dataset"]
        cameras = {
            name: CameraSpec(name=name, **values) for name, values in raw["cameras"].items()
        }
        files = tuple(QualifiedFile(**values) for values in raw["qualified_files"])
        return cls(
            dataset_id=dataset["dataset_id"],
            repository_id=dataset["repository_id"],
            revision=dataset["revision"],
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
            full_repository_bytes=int(dataset["full_repository_bytes"]),
            primary_trajectory_sha256=dataset["primary_trajectory_sha256"],
            modalities=tuple(dataset["modalities"]),
            task_families=tuple(dataset["task_families"]),
            task_family_ranges={
                name: (int(bounds["first"]), int(bounds["last"]))
                for name, bounds in raw["task_family_ranges"].items()
            },
            camera_names=tuple(dataset["camera_names"]),
            known_limitations=tuple(dataset["known_limitations"]),
            units=dict(raw["units"]),
            coordinate_frames=dict(raw["coordinate_frames"]),
            joint_names=tuple(raw["schema"]["joint_names"]),
            cameras=cameras,
            qualified_files=files,
            qualified_episode_indices=tuple(raw["qualified_subset"]["episode_indices"]),
        )


@dataclass(frozen=True)
class VideoSegment:
    camera_name: str
    path: Path
    from_timestamp: float
    to_timestamp: float
    width: int
    height: int
    fps: float


class ProjectIRASO101Adapter:
    """Validate native Project IRA data and convert selected episodes without guessing frames."""

    def __init__(self, root: Path, spec: ProjectIRASourceSpec) -> None:
        self.root = root
        self.spec = spec

    @classmethod
    def from_registry(cls, root: Path, registry_path: Path) -> "ProjectIRASO101Adapter":
        return cls(root, ProjectIRASourceSpec.from_toml(registry_path))

    def _read_table(self, relative_path: str, *, columns: tuple[str, ...] | None = None) -> Any:
        return _parquet().read_table(self.root / relative_path, columns=list(columns) if columns else None)

    def _episode_table(self) -> Any:
        return self._read_table("meta/episodes/chunk-000/file-000.parquet")

    def _data_table(self) -> Any:
        return self._read_table("data/chunk-000/file-000.parquet", columns=REQUIRED_DATA_COLUMNS)

    def tasks(self) -> dict[int, str]:
        rows = self._read_table("meta/tasks.parquet").to_pylist()
        return {int(row["task_index"]): str(row["task"]) for row in rows}

    def episode_record(self, episode_index: int) -> dict[str, Any]:
        if not 0 <= episode_index < self.spec.total_episodes:
            raise IndexError(f"episode_index must be in [0, {self.spec.total_episodes})")
        return self._episode_table().slice(episode_index, 1).to_pylist()[0]

    def validate_source(self) -> dict[str, Any]:
        """Run full-table schema, alignment, finiteness and leakage-key checks."""

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

        episodes = self._episode_table()
        data = self._data_table()
        tasks = self.tasks()
        if episodes.num_rows != self.spec.total_episodes:
            raise ValueError("episode count differs from the pinned registry")
        if data.num_rows != self.spec.total_frames:
            raise ValueError("frame count differs from the pinned registry")
        if len(tasks) != self.spec.total_tasks or set(tasks) != set(range(self.spec.total_tasks)):
            raise ValueError("task indices must be unique and contiguous")

        expected_types = {
            "observation.state": "list<element: float>",
            "action": "list<element: float>",
            "timestamp": "float",
            "frame_index": "int64",
            "episode_index": "int64",
            "index": "int64",
            "task_index": "int64",
        }
        for column, expected_type in expected_types.items():
            actual_type = str(data.schema.field(column).type)
            if actual_type != expected_type:
                raise ValueError(f"{column} type drifted: {actual_type!r} != {expected_type!r}")
        for column in REQUIRED_DATA_COLUMNS:
            if data[column].null_count:
                raise ValueError(f"{column} contains nulls")

        lengths = np.asarray(episodes["length"].to_numpy(), dtype=np.int64)
        from_indices = np.asarray(episodes["dataset_from_index"].to_numpy(), dtype=np.int64)
        to_indices = np.asarray(episodes["dataset_to_index"].to_numpy(), dtype=np.int64)
        if lengths.sum() != self.spec.total_frames:
            raise ValueError("episode lengths do not sum to total_frames")
        if from_indices[0] != 0 or to_indices[-1] != self.spec.total_frames:
            raise ValueError("episode ranges do not cover the full trajectory table")
        if not np.array_equal(from_indices[1:], to_indices[:-1]):
            raise ValueError("episode ranges contain gaps or overlaps")
        if not np.array_equal(to_indices - from_indices, lengths):
            raise ValueError("episode range widths differ from declared lengths")

        episode_index = np.asarray(data["episode_index"].to_numpy(), dtype=np.int64)
        frame_index = np.asarray(data["frame_index"].to_numpy(), dtype=np.int64)
        global_index = np.asarray(data["index"].to_numpy(), dtype=np.int64)
        task_index = np.asarray(data["task_index"].to_numpy(), dtype=np.int64)
        timestamp = np.asarray(data["timestamp"].to_numpy(), dtype=np.float64)
        expected_episode = np.repeat(np.arange(self.spec.total_episodes), lengths)
        expected_frame = global_index - np.repeat(from_indices, lengths)
        if not np.array_equal(global_index, np.arange(self.spec.total_frames)):
            raise ValueError("global index is not unique and contiguous")
        if not np.array_equal(episode_index, expected_episode):
            raise ValueError("trajectory rows are not grouped by declared episode ranges")
        if not np.array_equal(frame_index, expected_frame):
            raise ValueError("frame indices do not restart at zero for each episode")
        if not np.allclose(timestamp, frame_index / self.spec.fps, rtol=0.0, atol=1e-5):
            raise ValueError("timestamps are not aligned to frame_index/fps")

        states = np.asarray(data["observation.state"].to_pylist(), dtype=np.float32)
        actions = np.asarray(data["action"].to_pylist(), dtype=np.float32)
        expected_shape = (self.spec.total_frames, len(self.spec.joint_names))
        if states.shape != expected_shape or actions.shape != expected_shape:
            raise ValueError(f"state/action arrays must both have shape {expected_shape}")
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise ValueError("state/action arrays contain NaN or Inf")

        episode_tasks = task_index[from_indices]
        task_counts = np.bincount(episode_tasks, minlength=self.spec.total_tasks)
        if not np.array_equal(task_counts, np.full(self.spec.total_tasks, 10)):
            raise ValueError("each pinned prompt must map to exactly ten episodes")
        metadata_tasks = episodes["tasks"].to_pylist()
        for episode, source_task in enumerate(episode_tasks):
            if metadata_tasks[episode] != [tasks[int(source_task)]]:
                raise ValueError(f"episode {episode} task text/index mismatch")

        within_episode = frame_index > 0
        action_delta = np.abs(actions[within_episode] - actions[np.flatnonzero(within_episode) - 1])
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
            "null_values": 0,
            "nonfinite_values": 0,
            "prompt_episode_counts": sorted(set(task_counts.tolist())),
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
            pixel_format = context.pix_fmt or (context.format.name if context.format else None)
            return {
                "codec": context.name,
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
        task = self.tasks()[int(task_indices[0])]
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
            success=None,
            quality=None,
            collection_method=self.spec.collection_method,
            native_action_semantics=self.spec.native_action_semantics,
            fps=self.spec.fps,
            camera_names=self.spec.camera_names,
            extra={
                "source_episode_index": episode_index,
                "source_task_index": int(task_indices[0]),
                "source_dataset_from_index": start,
                "source_length": length,
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
