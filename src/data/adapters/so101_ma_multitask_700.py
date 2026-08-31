"""Pinned LeRobot-v3 adapter for the SO101 MA MultiTask 700 simulation release."""

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
)

REQUIRED_EPISODE_COLUMNS = (
    "episode_index",
    "tasks",
    "length",
    "data/chunk_index",
    "data/file_index",
    "dataset_from_index",
    "dataset_to_index",
)

DROPPED_UPSTREAM_COLUMNS = (
    "action.radian_urdf0",
    "observation.state.radian_urdf0",
    "observation.ee_pos.robot_xyzrpy",
    "observation.gripper_binary",
)


def _arrow() -> tuple[Any, Any]:
    try:
        import pyarrow as arrow
        import pyarrow.parquet as parquet
    except ImportError as error:  # pragma: no cover - exercised by packaging
        raise RuntimeError("SO101 MA Parquet support requires the 'data' optional dependency") from error
    return arrow, parquet


def _video() -> Any:
    try:
        import av
    except ImportError as error:  # pragma: no cover - exercised by packaging
        raise RuntimeError("SO101 MA video support requires the 'data' optional dependency") from error
    return av


@dataclass(frozen=True)
class QualifiedObject:
    path: str
    size: int
    sha256: str
    role: str


@dataclass(frozen=True)
class UpstreamSource:
    declared_repository_id: str
    resolved_repository_id: str
    revision: str
    license: str
    task_index: int
    episodes: int
    frames: int


@dataclass(frozen=True)
class CameraSpec:
    name: str
    feature_key: str
    width: int
    height: int
    allowed_codecs: tuple[str, ...]
    pixel_format: str
    fps: float


@dataclass(frozen=True)
class VideoSegment:
    episode_index: int
    camera_name: str
    path: Path
    from_timestamp: float
    aligned_to_timestamp: float
    declared_to_timestamp: float
    aligned_frames: int
    unused_frames: int
    width: int
    height: int
    fps: float


@dataclass(frozen=True)
class SO101MAMultiTaskSourceSpec:
    dataset_id: str
    repository_id: str
    revision: str
    source_url: str
    license: str
    redistribution_terms: str
    domain: str
    robot_id: str
    embodiment: str
    simulator_family: str
    simulator_version: str
    data_format: str
    collection_method: str
    native_action_semantics: str
    priority: str
    status: str
    fps: float
    total_episodes: int
    total_frames: int
    total_tasks: int
    full_tree_files: int
    full_tree_bytes: int
    hub_used_storage_bytes: int
    modalities: tuple[str, ...]
    task_families: tuple[str, ...]
    task_labels: tuple[str, ...]
    camera_names: tuple[str, ...]
    known_limitations: tuple[str, ...]
    units: dict[str, str]
    coordinate_frames: dict[str, str]
    joint_names: tuple[str, ...]
    task_text_source_column: str
    cameras: dict[str, CameraSpec]
    qualified_episode_indices: tuple[int, ...]
    extra_video_episode: int
    extra_video_frames_per_camera: int
    source_episode_block_size: int
    test_block_offset: int
    validation_block_offset: int
    object_manifest_path: Path
    object_manifest_sha256: str
    qualified_objects: tuple[QualifiedObject, ...]
    upstream_sources: tuple[UpstreamSource, ...]

    @classmethod
    def from_toml(cls, path: Path) -> "SO101MAMultiTaskSourceSpec":
        with path.open("rb") as handle:
            raw = tomllib.load(handle)
        dataset = raw["dataset"]
        project_root = path.resolve().parents[3]
        object_manifest_path = project_root / raw["objects"]["manifest_path"]
        object_manifest = json.loads(object_manifest_path.read_text(encoding="utf-8"))
        cameras = {
            name: CameraSpec(
                name=name,
                feature_key=values["feature_key"],
                width=int(values["width"]),
                height=int(values["height"]),
                allowed_codecs=tuple(values["allowed_codecs"]),
                pixel_format=values["pixel_format"],
                fps=float(values["fps"]),
            )
            for name, values in raw["cameras"].items()
        }
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
            simulator_family=dataset["simulator_family"],
            simulator_version=dataset["simulator_version"],
            data_format=dataset["data_format"],
            collection_method=dataset["collection_method"],
            native_action_semantics=dataset["native_action_semantics"],
            priority=dataset["priority"],
            status=dataset["status"],
            fps=float(dataset["fps"]),
            total_episodes=int(dataset["total_episodes"]),
            total_frames=int(dataset["total_frames"]),
            total_tasks=int(dataset["total_tasks"]),
            full_tree_files=int(dataset["full_tree_files"]),
            full_tree_bytes=int(dataset["full_tree_bytes"]),
            hub_used_storage_bytes=int(dataset["hub_used_storage_bytes"]),
            modalities=tuple(dataset["modalities"]),
            task_families=tuple(dataset["task_families"]),
            task_labels=tuple(dataset["task_labels"]),
            camera_names=tuple(dataset["camera_names"]),
            known_limitations=tuple(dataset["known_limitations"]),
            units=dict(raw["units"]),
            coordinate_frames=dict(raw["coordinate_frames"]),
            joint_names=tuple(raw["schema"]["joint_names"]),
            task_text_source_column=raw["schema"]["task_text_source_column"],
            cameras=cameras,
            qualified_episode_indices=tuple(raw["qualified_subset"]["episode_indices"]),
            extra_video_episode=int(raw["qualified_subset"]["extra_video_episode"]),
            extra_video_frames_per_camera=int(
                raw["qualified_subset"]["extra_video_frames_per_camera"]
            ),
            source_episode_block_size=int(raw["splitting"]["source_episode_block_size"]),
            test_block_offset=int(raw["splitting"]["test_block_offset"]),
            validation_block_offset=int(raw["splitting"]["validation_block_offset"]),
            object_manifest_path=object_manifest_path,
            object_manifest_sha256=raw["objects"]["manifest_sha256"],
            qualified_objects=tuple(
                QualifiedObject(**item) for item in object_manifest["objects"]
            ),
            upstream_sources=tuple(
                UpstreamSource(**item) for item in object_manifest["upstream_sources"]
            ),
        )


class SO101MAMultiTaskAdapter:
    """Validate the merged simulation corpus without inventing units or labels."""

    def __init__(self, root: Path, spec: SO101MAMultiTaskSourceSpec) -> None:
        self.root = root
        self.spec = spec
        self._episodes_cache: Any | None = None
        self._tasks_cache: dict[int, str] | None = None
        self._data_cache: Any | None = None

    @classmethod
    def from_registry(
        cls, root: Path, registry_path: Path
    ) -> "SO101MAMultiTaskAdapter":
        return cls(root, SO101MAMultiTaskSourceSpec.from_toml(registry_path))

    def _read_table(
        self, relative_path: str, *, columns: tuple[str, ...] | None = None
    ) -> Any:
        _, parquet = _arrow()
        return parquet.read_table(
            self.root / relative_path,
            columns=list(columns) if columns else None,
        )

    def _episode_table(self) -> Any:
        if self._episodes_cache is None:
            self._episodes_cache = self._read_table(
                "meta/episodes/chunk-000/file-000.parquet"
            )
        return self._episodes_cache

    def _data_table(self) -> Any:
        if self._data_cache is None:
            self._data_cache = self._read_table(
                "data/chunk-000/file-000.parquet",
                columns=REQUIRED_DATA_COLUMNS,
            )
        return self._data_cache

    def tasks(self) -> dict[int, str]:
        if self._tasks_cache is None:
            table = self._read_table("meta/tasks.parquet")
            if "task" in table.column_names:
                raise ValueError("source task-table defect changed; review the normalization policy")
            expected_columns = {"task_index", self.spec.task_text_source_column}
            if set(table.column_names) != expected_columns:
                raise ValueError(
                    "task table must contain only task_index and the pinned nonstandard text column"
                )
            rows = table.to_pylist()
            self._tasks_cache = {
                int(row["task_index"]): str(row[self.spec.task_text_source_column])
                for row in rows
            }
        return self._tasks_cache

    def episode_record(self, episode_index: int) -> dict[str, Any]:
        if not 0 <= episode_index < self.spec.total_episodes:
            raise IndexError(
                f"episode_index must be in [0, {self.spec.total_episodes})"
            )
        record = self._episode_table().slice(episode_index, 1).to_pylist()[0]
        if int(record["episode_index"]) != episode_index:
            raise ValueError("episode metadata rows are not ordered by episode_index")
        return record

    def episode_records(self) -> list[dict[str, Any]]:
        return self._episode_table().to_pylist()

    def trajectory_arrays(self) -> dict[str, np.ndarray]:
        table = self._data_table()
        return {
            "state": np.asarray(table["observation.state"].to_pylist(), dtype=np.float32),
            "action": np.asarray(table["action"].to_pylist(), dtype=np.float32),
            "timestamp": np.asarray(table["timestamp"].to_numpy(), dtype=np.float64),
            "frame_index": np.asarray(table["frame_index"].to_numpy(), dtype=np.int64),
            "episode_index": np.asarray(
                table["episode_index"].to_numpy(), dtype=np.int64
            ),
            "task_index": np.asarray(table["task_index"].to_numpy(), dtype=np.int64),
        }

    def _validate_merge_provenance(self) -> dict[str, Any]:
        manifest = json.loads(
            (self.root / "meta/scrape_collection_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        expected = {
            "output_repo_id": self.spec.repository_id,
            "total_episodes": self.spec.total_episodes,
            "total_frames": self.spec.total_frames,
            "fps": int(self.spec.fps),
            "total_tasks": self.spec.total_tasks,
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(f"merge provenance {key!r} differs from the registry")
        source_rows = manifest.get("sources", [])
        if len(source_rows) != self.spec.total_tasks:
            raise ValueError("merge provenance must list exactly seven sources")
        for index, (source_row, pin) in enumerate(
            zip(source_rows, self.spec.upstream_sources, strict=True)
        ):
            expected_row = {
                "repo_id": pin.declared_repository_id,
                "episodes": pin.episodes,
                "frames": pin.frames,
            }
            for key, value in expected_row.items():
                if source_row.get(key) != value:
                    raise ValueError(
                        f"upstream source {index} {key!r} differs from the pin"
                    )
            if pin.task_index != index:
                raise ValueError("upstream pins are not ordered by task_index")
        dropped = set(manifest.get("dropped_optional_feature_groups", ()))
        if not set(DROPPED_UPSTREAM_COLUMNS).issubset(dropped):
            raise ValueError("merge no longer declares all pinned dropped action/frame fields")
        return {
            "derivation": manifest["derivation"],
            "upstream_source_count": len(source_rows),
            "declared_repository_ids": [row["repo_id"] for row in source_rows],
            "resolved_repository_ids": [
                source.resolved_repository_id for source in self.spec.upstream_sources
            ],
            "dropped_optional_feature_groups": sorted(dropped),
        }

    def _validate_video_metadata(
        self, episodes: Any, lengths: np.ndarray
    ) -> dict[str, dict[str, int | float]]:
        results: dict[str, dict[str, int | float]] = {}
        for camera_name in self.spec.camera_names:
            prefix = f"videos/observation.images.{camera_name}"
            chunk_indices = np.asarray(
                episodes[f"{prefix}/chunk_index"].to_numpy(), dtype=np.int64
            )
            file_indices = np.asarray(
                episodes[f"{prefix}/file_index"].to_numpy(), dtype=np.int64
            )
            starts = np.asarray(
                episodes[f"{prefix}/from_timestamp"].to_numpy(), dtype=np.float64
            )
            stops = np.asarray(
                episodes[f"{prefix}/to_timestamp"].to_numpy(), dtype=np.float64
            )
            if not np.array_equal(
                chunk_indices, np.zeros(self.spec.total_episodes, dtype=np.int64)
            ):
                raise ValueError(
                    f"{camera_name} video chunks differ from the pinned single-chunk layout"
                )
            expected_durations = lengths / self.spec.fps
            expected_durations[self.spec.extra_video_episode] += (
                self.spec.extra_video_frames_per_camera / self.spec.fps
            )
            if not np.allclose(
                stops - starts, expected_durations, rtol=0.0, atol=1e-8
            ):
                raise ValueError(
                    f"{camera_name} video segment durations differ from the pinned edge-case policy"
                )
            previous_stop: dict[int, float] = {}
            for file_index, start, stop in zip(
                file_indices, starts, stops, strict=True
            ):
                if file_index in previous_stop and not np.isclose(
                    start, previous_stop[int(file_index)], rtol=0.0, atol=1e-8
                ):
                    raise ValueError(
                        f"{camera_name} referenced video segments contain gaps or overlaps"
                    )
                if file_index not in previous_stop and not np.isclose(
                    start, 0.0, rtol=0.0, atol=1e-8
                ):
                    raise ValueError(
                        f"{camera_name} packed-video file does not start at timestamp zero"
                    )
                previous_stop[int(file_index)] = float(stop)
            results[camera_name] = {
                "file_count": len(previous_stop),
                "extra_frame_episode": self.spec.extra_video_episode,
                "unused_frames": self.spec.extra_video_frames_per_camera,
                "unused_seconds": (
                    self.spec.extra_video_frames_per_camera / self.spec.fps
                ),
            }
        return results

    def validate_source(self) -> dict[str, Any]:
        """Validate all numeric rows, metadata, provenance and video intervals."""

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
                raise ValueError(
                    f"meta/info.json {key!r} drifted: {info.get(key)!r} != {expected!r}"
                )
        features = info.get("features", {})
        if tuple(features["observation.state"]["names"]) != self.spec.joint_names:
            raise ValueError("state joint order differs from the pinned registry")
        if tuple(features["action"]["names"]) != self.spec.joint_names:
            raise ValueError("action joint order differs from the pinned registry")
        for camera_name, camera in self.spec.cameras.items():
            published = features[camera.feature_key]["info"]
            expected_camera = {
                "video.width": camera.width,
                "video.height": camera.height,
                "video.codec": "mixed",
                "video.pix_fmt": camera.pixel_format,
                "video.fps": int(camera.fps),
            }
            for key, expected in expected_camera.items():
                if published.get(key) != expected:
                    raise ValueError(f"{camera_name} {key} differs from the registry")
        if any(column in features for column in DROPPED_UPSTREAM_COLUMNS):
            raise ValueError("a dropped action/frame feature unexpectedly reappeared")

        provenance = self._validate_merge_provenance()
        episodes = self._episode_table()
        tasks = self.tasks()
        if episodes.num_rows != self.spec.total_episodes:
            raise ValueError("episode count differs from the pinned registry")
        if tasks != dict(enumerate(self.spec.task_labels)):
            raise ValueError("normalized task labels differ from the pinned registry")
        for column in REQUIRED_EPISODE_COLUMNS:
            if episodes[column].null_count:
                raise ValueError(f"episode metadata column {column!r} contains nulls")

        episode_indices = np.asarray(
            episodes["episode_index"].to_numpy(), dtype=np.int64
        )
        lengths = np.asarray(episodes["length"].to_numpy(), dtype=np.int64)
        from_indices = np.asarray(
            episodes["dataset_from_index"].to_numpy(), dtype=np.int64
        )
        to_indices = np.asarray(
            episodes["dataset_to_index"].to_numpy(), dtype=np.int64
        )
        if not np.array_equal(
            episode_indices, np.arange(self.spec.total_episodes)
        ):
            raise ValueError("episode indices are not unique, ordered and contiguous")
        if lengths.sum() != self.spec.total_frames:
            raise ValueError("episode lengths do not sum to total_frames")
        if from_indices[0] != 0 or to_indices[-1] != self.spec.total_frames:
            raise ValueError("episode ranges do not cover the complete trajectory")
        if not np.array_equal(from_indices[1:], to_indices[:-1]):
            raise ValueError("episode ranges contain gaps or overlaps")
        if not np.array_equal(to_indices - from_indices, lengths):
            raise ValueError("episode range widths differ from declared lengths")

        metadata_tasks = episodes["tasks"].to_pylist()
        if any(len(values) != 1 for values in metadata_tasks):
            raise ValueError("each episode must contain exactly one task label")
        reverse_tasks = {text: index for index, text in tasks.items()}
        try:
            metadata_task_indices = np.asarray(
                [reverse_tasks[values[0]] for values in metadata_tasks],
                dtype=np.int64,
            )
        except KeyError as error:
            raise ValueError("episode metadata contains an unknown task") from error
        episode_task_counts = Counter(metadata_task_indices.tolist())
        if episode_task_counts != Counter({index: 100 for index in range(7)}):
            raise ValueError("every merged source must contribute exactly 100 episodes")
        expected_task_frames = {
            source.task_index: source.frames for source in self.spec.upstream_sources
        }
        actual_task_frames = {
            index: int(lengths[metadata_task_indices == index].sum())
            for index in range(self.spec.total_tasks)
        }
        if actual_task_frames != expected_task_frames:
            raise ValueError("per-task frame totals differ from upstream pins")

        video_metadata = self._validate_video_metadata(episodes, lengths)
        data = self._data_table()
        if data.num_rows != self.spec.total_frames:
            raise ValueError("trajectory row count differs from the registry")
        expected_types = {
            "observation.state": "fixed_size_list<element: float>[6]",
            "action": "fixed_size_list<element: float>[6]",
            "timestamp": "float",
            "frame_index": "int64",
            "episode_index": "int64",
            "index": "int64",
            "task_index": "int64",
        }
        for column, expected_type in expected_types.items():
            actual_type = str(data.schema.field(column).type)
            if actual_type != expected_type:
                raise ValueError(
                    f"{column} type drifted: {actual_type!r} != {expected_type!r}"
                )
            if data[column].null_count:
                raise ValueError(f"{column} contains nulls")

        arrays = self.trajectory_arrays()
        states = arrays["state"]
        actions = arrays["action"]
        frame_index = arrays["frame_index"]
        episode_index = arrays["episode_index"]
        task_index = arrays["task_index"]
        timestamps = arrays["timestamp"]
        global_index = np.asarray(data["index"].to_numpy(), dtype=np.int64)
        expected_episode = np.repeat(
            np.arange(self.spec.total_episodes), lengths
        )
        expected_frame = global_index - np.repeat(from_indices, lengths)
        if not np.array_equal(global_index, np.arange(self.spec.total_frames)):
            raise ValueError("global trajectory index is not unique and contiguous")
        if not np.array_equal(episode_index, expected_episode):
            raise ValueError("trajectory rows are not grouped by episode ranges")
        if not np.array_equal(frame_index, expected_frame):
            raise ValueError("frame indices do not restart at zero per episode")
        if not np.array_equal(
            task_index, np.repeat(metadata_task_indices, lengths)
        ):
            raise ValueError("trajectory task indices differ from episode task text")
        if not np.allclose(
            timestamps, frame_index / self.spec.fps, rtol=0.0, atol=1e-5
        ):
            raise ValueError("timestamps are not aligned to frame_index/fps")
        expected_shape = (
            self.spec.total_frames,
            len(self.spec.joint_names),
        )
        if states.shape != expected_shape or actions.shape != expected_shape:
            raise ValueError(f"state/action arrays must both have shape {expected_shape}")
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise ValueError("state/action arrays contain NaN or Inf")

        within_episode = frame_index > 0
        action_delta = np.abs(
            actions[within_episode]
            - actions[np.flatnonzero(within_episode) - 1]
        )
        stats = json.loads(
            (self.root / "meta/stats.json").read_text(encoding="utf-8")
        )
        published_counts = {
            name: int(values["count"][0])
            for name, values in stats.items()
            if "count" in values
        }
        mismatched_counts = {
            name: count
            for name, count in published_counts.items()
            if count != self.spec.total_frames
        }
        if not mismatched_counts:
            raise ValueError("published statistics unexpectedly became consistent; review the pin")
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
            "task_episode_counts": {
                self.spec.task_families[index]: episode_task_counts[index]
                for index in range(self.spec.total_tasks)
            },
            "task_frame_counts": {
                self.spec.task_families[index]: actual_task_frames[index]
                for index in range(self.spec.total_tasks)
            },
            "task_text_source_column": self.spec.task_text_source_column,
            "task_text_normalized": True,
            "success_labels": "absent",
            "quality_labels": "absent",
            "video_metadata": video_metadata,
            "null_values": 0,
            "nonfinite_values": 0,
            "published_statistics_counts": dict(sorted(published_counts.items())),
            "published_statistics_mismatched_counts": dict(
                sorted(mismatched_counts.items())
            ),
            "published_statistics_trusted": False,
            "merge_provenance": provenance,
            "native_action_semantics": self.spec.native_action_semantics,
            "native_action_physical_units_proven": False,
            "task_space_conversion_allowed": False,
        }

    def video_segment(self, episode_index: int, camera_name: str) -> VideoSegment:
        if camera_name not in self.spec.cameras:
            raise KeyError(f"unknown camera {camera_name!r}")
        record = self.episode_record(episode_index)
        prefix = f"videos/observation.images.{camera_name}"
        chunk_index = int(record[f"{prefix}/chunk_index"])
        file_index = int(record[f"{prefix}/file_index"])
        camera = self.spec.cameras[camera_name]
        aligned_frames = int(record["length"])
        unused_frames = (
            self.spec.extra_video_frames_per_camera
            if episode_index == self.spec.extra_video_episode
            else 0
        )
        start = float(record[f"{prefix}/from_timestamp"])
        return VideoSegment(
            episode_index=episode_index,
            camera_name=camera_name,
            path=(
                self.root
                / prefix
                / f"chunk-{chunk_index:03d}"
                / f"file-{file_index:03d}.mp4"
            ),
            from_timestamp=start,
            aligned_to_timestamp=start + aligned_frames / camera.fps,
            declared_to_timestamp=float(record[f"{prefix}/to_timestamp"]),
            aligned_frames=aligned_frames,
            unused_frames=unused_frames,
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
            pixel_format = (
                context.pix_fmt
                or (context.format.name if context.format else None)
            )
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
        if probe["codec"] not in camera.allowed_codecs:
            raise ValueError(
                f"{segment.camera_name} codec {probe['codec']!r} is not allowed"
            )
        expected = {
            "pixel_format": camera.pixel_format,
            "width": camera.width,
            "height": camera.height,
        }
        for key, value in expected.items():
            if probe[key] != value:
                raise ValueError(
                    f"{segment.camera_name} {key} drifted: {probe[key]!r} != {value!r}"
                )
        if not np.isclose(probe["fps"], camera.fps, rtol=0.0, atol=1e-6):
            raise ValueError(f"{segment.camera_name} fps differs from the registry")
        if probe["duration_seconds"] + 1 / camera.fps < segment.declared_to_timestamp:
            raise ValueError(
                f"{segment.camera_name} file does not cover its declared interval"
            )
        declared_frames = round(
            (segment.declared_to_timestamp - segment.from_timestamp) * segment.fps
        )
        if declared_frames != segment.aligned_frames + segment.unused_frames:
            raise ValueError("video segment unused-frame accounting is inconsistent")
        return probe

    @staticmethod
    def decode_rgb_frame(
        segment: VideoSegment, episode_local_timestamp: float
    ) -> torch.Tensor:
        aligned_duration = segment.aligned_frames / segment.fps
        if not 0.0 <= episode_local_timestamp < aligned_duration:
            raise ValueError("decode timestamp lies outside aligned trajectory frames")
        timestamp = segment.from_timestamp + episode_local_timestamp
        av = _video()
        with av.open(str(segment.path), mode="r") as container:
            stream = container.streams.video[0]
            container.seek(
                int(timestamp / stream.time_base),
                stream=stream,
                any_frame=False,
                backward=True,
            )
            for frame in container.decode(stream):
                frame_time = (
                    float(frame.pts * frame.time_base)
                    if frame.pts is not None
                    else timestamp
                )
                if frame_time + 0.5 / segment.fps < timestamp:
                    continue
                array = frame.to_ndarray(format="rgb24")
                expected_shape = (segment.height, segment.width, 3)
                if array.shape != expected_shape:
                    raise ValueError(
                        f"decoded {segment.camera_name} frame has shape "
                        f"{array.shape}, expected {expected_shape}"
                    )
                return torch.from_numpy(array.copy()).permute(2, 0, 1)
        raise ValueError(
            f"could not decode {segment.camera_name} at {timestamp:.9f}s"
        )

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
        transitions = (
            length - 1
            if max_transitions is None
            else min(max_transitions, length - 1)
        )
        if transitions < 1:
            raise ValueError("canonical samples require at least one transition")
        rows = self._data_table().slice(start, transitions + 1)
        states = np.asarray(
            rows["observation.state"].to_pylist(), dtype=np.float32
        )
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
            camera: self.video_segment(episode_index, camera)
            for camera in self.spec.camera_names
        }
        observations: list[Observation] = []
        for index, timestamp in enumerate(timestamps):
            rgb: dict[str, torch.Tensor] = {}
            if include_rgb:
                for camera, segment in segments.items():
                    rgb[camera] = self.decode_rgb_frame(
                        segment, float(timestamp)
                    )
            observations.append(
                Observation(
                    timestamp=float(timestamp),
                    q=torch.from_numpy(states[index].copy()),
                    qdot=torch.from_numpy(qdot[index].copy()),
                    previous_command=torch.from_numpy(
                        previous_commands[index].copy()
                    ),
                    rgb=rgb,
                    validity={
                        camera: camera in rgb for camera in self.spec.camera_names
                    },
                )
            )
        actions = tuple(
            Action(
                timestamp=float(timestamps[index]),
                native=torch.from_numpy(source_actions[index].copy()),
            )
            for index in range(transitions)
        )
        task_index = int(task_indices[0])
        task = self.tasks()[task_index]
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
            source_policy="Code-as-Policies/SCRAPE demonstration",
            simulator_family=self.spec.simulator_family,
            simulator_version=self.spec.simulator_version,
            fps=self.spec.fps,
            camera_names=self.spec.camera_names,
            extra={
                "source_episode_index": episode_index,
                "source_task_index": task_index,
                "source_task_family": self.spec.task_families[task_index],
                "source_episode_block_10": (
                    (episode_index % 100) // self.spec.source_episode_block_size
                ),
                "source_dataset_from_index": start,
                "source_length": length,
                "success_label_available": False,
                "quality_label_available": False,
                "imitation_eligible": False,
                "prediction_eligible": True,
                "native_action_physical_units_proven": False,
                "task_space_conversion_allowed": False,
                "unit_conventions": self.spec.units,
                "coordinate_frames": self.spec.coordinate_frames,
                "unused_video_frames_per_camera": (
                    self.spec.extra_video_frames_per_camera
                    if episode_index == self.spec.extra_video_episode
                    else 0
                ),
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
