"""Write-once, checksummed Stage 1 raw episode storage."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
from typing import Any, Iterable
import uuid

import numpy as np

from .config import Stage1Config, validate_run_id
from .records import EpisodePayload


GIB = 1024**3


class DatasetQuotaReached(RuntimeError):
    """Raised before admitting an episode that would exceed a storage guard."""


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _encode_video_pyav(path: Path, frames: np.ndarray, fps: int, codec: str, crf: int, preset: str) -> None:
    try:
        import av
    except ImportError as error:
        raise RuntimeError("PyAV is required by camera.video_backend='pyav'") from error

    frames = np.asarray(frames, dtype=np.uint8)
    container = av.open(str(path), mode="w")
    try:
        stream = container.add_stream(codec, rate=fps)
        stream.width = int(frames.shape[2])
        stream.height = int(frames.shape[1])
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(crf), "preset": preset}
        for frame_array in frames:
            frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _parse_checksums(path: Path) -> dict[str, tuple[str, int]]:
    entries: dict[str, tuple[str, int]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("  ")
        if len(parts) != 3:
            raise ValueError(f"malformed checksum line in {path}: {line!r}")
        digest, size_text, name = parts
        if Path(name).name != name or name in entries:
            raise ValueError(f"unsafe or duplicate checksum entry: {name!r}")
        entries[name] = (digest, int(size_text))
    return entries


def verify_episode_directory(path: Path, verify_payload: bool = True) -> dict[str, Any]:
    """Verify commit marker, sizes, hashes, and the canonical telemetry contract."""

    commit_path = path / "COMMITTED.json"
    checksum_path = path / "checksums.sha256"
    if not commit_path.is_file() or not checksum_path.is_file():
        raise ValueError(f"episode is not committed: {path}")
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    if commit.get("episode_id") != path.name:
        raise ValueError(f"episode id/path mismatch at {path}")
    if sha256_file(checksum_path) != commit.get("checksums_file_sha256"):
        raise ValueError(f"checksum manifest digest mismatch at {path}")
    entries = _parse_checksums(checksum_path)
    required = {"metadata.json", "telemetry.npz"}
    if not required.issubset(entries):
        raise ValueError(f"required payload files are missing at {path}")
    if not ({"rear_rgb.mp4", "rear_rgb.npz"} & set(entries)):
        raise ValueError(f"rear RGB payload is missing at {path}")
    for name, (expected_hash, expected_size) in entries.items():
        target = path / name
        if not target.is_file() or target.stat().st_size != expected_size:
            raise ValueError(f"size or file mismatch for {target}")
        if sha256_file(target) != expected_hash:
            raise ValueError(f"SHA-256 mismatch for {target}")

    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("episode_id") != path.name:
        raise ValueError(f"metadata episode id mismatch at {path}")
    if verify_payload:
        with np.load(path / "telemetry.npz", allow_pickle=False) as payload:
            expected_shapes = metadata.get("array_shapes", {})
            if set(payload.files) != set(expected_shapes):
                raise ValueError(f"telemetry keys do not match metadata at {path}")
            for key in payload.files:
                if list(payload[key].shape) != expected_shapes[key]:
                    raise ValueError(f"shape mismatch for {key} at {path}")
            timestamps = payload["timestamp_sim_s"]
            if timestamps.ndim != 1 or np.any(np.diff(timestamps) <= 0):
                raise ValueError(f"non-monotonic timestamps at {path}")
            actions = payload["action_native_joint_position_rad"]
            q = payload["joint_position_rad"]
            if q.shape[0] != actions.shape[0] + 1 or q.shape[-1] != 6:
                raise ValueError(f"invalid transition contract at {path}")
    return metadata


class RunWriter:
    """Owns one append-only generation run and atomically commits episodes."""

    def __init__(
        self,
        output_root: str | Path,
        run_id: str,
        config: Stage1Config,
        provenance: dict[str, Any],
        verify_existing: bool = True,
    ) -> None:
        self.run_id = validate_run_id(run_id)
        self.config = config
        self.root = Path(output_root).expanduser().resolve() / self.run_id
        self.episodes_dir = self.root / "episodes"
        self.staging_dir = self.root / "_staging"
        self._lock = threading.Lock()
        self.max_bytes = int(config.dataset.max_output_gib * GIB)
        self.min_free_bytes = int(config.dataset.min_free_disk_gib * GIB)
        self.writer_threads = config.dataset.writer_threads
        self._committed: dict[str, dict[str, Any]] = {}

        self.episodes_dir.mkdir(parents=True, exist_ok=True)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.root / "run_manifest.json"
        immutable_manifest = {
            "schema_version": "project-subliminal-canonical-episode-v1",
            "run_id": self.run_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "config_sha256": config.digest,
            "config": config.to_dict(),
            "provenance": provenance,
        }
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing.get("config_sha256") != config.digest:
                raise ValueError(f"run {run_id} already exists with a different configuration")
        else:
            write_json_atomic(manifest_path, immutable_manifest)

        # Staging directories are explicitly non-admitted and safe to discard after a crash.
        for stale in self.staging_dir.iterdir():
            if stale.is_dir():
                shutil.rmtree(stale)
            else:
                stale.unlink()

        episode_paths = sorted(path for path in self.episodes_dir.iterdir() if path.is_dir())
        for path in episode_paths:
            metadata = verify_episode_directory(path, verify_payload=verify_existing)
            self._committed[path.name] = metadata
        self._bytes_written = _directory_size(self.root)
        self.rebuild_episode_index()
        self.write_state("ready")

    @property
    def committed_count(self) -> int:
        return len(self._committed)

    @property
    def bytes_written(self) -> int:
        return self._bytes_written

    def has_episode(self, episode_id: str) -> bool:
        return episode_id in self._committed

    def _write_staging(self, payload: EpisodePayload) -> Path:
        payload.validate()
        stage = self.staging_dir / f"{payload.episode_id}.{uuid.uuid4().hex}"
        stage.mkdir(parents=False, exist_ok=False)
        np.savez_compressed(stage / "telemetry.npz", **payload.arrays)
        if self.config.camera.video_backend == "pyav":
            _encode_video_pyav(
                stage / "rear_rgb.mp4",
                payload.rear_rgb,
                self.config.camera.fps,
                self.config.camera.codec,
                self.config.camera.crf,
                self.config.camera.preset,
            )
        else:
            np.savez_compressed(stage / "rear_rgb.npz", rear_rgb=payload.rear_rgb)
        if payload.segmentation is not None:
            np.savez_compressed(stage / "rear_segmentation.npz", segmentation=payload.segmentation)

        metadata = dict(payload.metadata)
        metadata.update(
            {
                "episode_id": payload.episode_id,
                "schema_version": "project-subliminal-canonical-episode-v1",
                "array_shapes": {key: list(value.shape) for key, value in payload.arrays.items()},
                "array_dtypes": {key: str(value.dtype) for key, value in payload.arrays.items()},
                "rear_rgb_shape": list(payload.rear_rgb.shape),
                "rear_rgb_dtype": str(payload.rear_rgb.dtype),
                "segmentation_shape": None if payload.segmentation is None else list(payload.segmentation.shape),
            }
        )
        write_json_atomic(stage / "metadata.json", metadata)

        payload_files = sorted(item for item in stage.iterdir() if item.is_file())
        checksum_lines = []
        for item in payload_files:
            checksum_lines.append(f"{sha256_file(item)}  {item.stat().st_size}  {item.name}")
        checksum_path = stage / "checksums.sha256"
        checksum_path.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
        write_json_atomic(
            stage / "COMMITTED.json",
            {
                "episode_id": payload.episode_id,
                "committed_utc": datetime.now(timezone.utc).isoformat(),
                "checksums_file_sha256": sha256_file(checksum_path),
                "config_sha256": self.config.digest,
            },
        )
        return stage

    def write_episode(self, payload: EpisodePayload) -> str:
        if self.has_episode(payload.episode_id):
            return "skipped"
        stage: Path | None = None
        try:
            stage = self._write_staging(payload)
            staged_bytes = _directory_size(stage)
            with self._lock:
                if payload.episode_id in self._committed:
                    shutil.rmtree(stage)
                    return "skipped"
                free_bytes = shutil.disk_usage(self.root).free
                if free_bytes - staged_bytes < self.min_free_bytes:
                    raise DatasetQuotaReached(
                        f"free disk guard reached: preserve {self.config.dataset.min_free_disk_gib:.1f} GiB"
                    )
                if self._bytes_written + staged_bytes > self.max_bytes:
                    raise DatasetQuotaReached(
                        f"run output guard reached: {self.config.dataset.max_output_gib:.1f} GiB"
                    )
                destination = self.episodes_dir / payload.episode_id
                if destination.exists():
                    raise FileExistsError(f"refusing to overwrite raw episode {destination}")
                os.replace(stage, destination)
                metadata = json.loads((destination / "metadata.json").read_text(encoding="utf-8"))
                self._committed[payload.episode_id] = metadata
                self._bytes_written += staged_bytes
            return "written"
        except Exception:
            if stage is not None and stage.exists():
                shutil.rmtree(stage)
            raise

    def write_batch(self, payloads: Iterable[EpisodePayload]) -> dict[str, int]:
        payload_list = [payload for payload in payloads if not self.has_episode(payload.episode_id)]
        counts = {"written": 0, "skipped": 0}
        if not payload_list:
            counts["skipped"] = len(list(payloads)) if not isinstance(payloads, list) else len(payloads)
            return counts
        with ThreadPoolExecutor(max_workers=self.writer_threads, thread_name_prefix="stage1-writer") as pool:
            futures = {pool.submit(self.write_episode, payload): payload.episode_id for payload in payload_list}
            try:
                for future in as_completed(futures):
                    counts[future.result()] += 1
            except Exception:
                for future in futures:
                    future.cancel()
                raise
        self.rebuild_episode_index()
        self.write_state("recording")
        return counts

    def rebuild_episode_index(self) -> None:
        lines = [json.dumps(_jsonable(self._committed[key]), sort_keys=True) for key in sorted(self._committed)]
        target = self.root / "episodes.jsonl"
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        os.replace(temporary, target)

    def write_state(self, status: str, **extra: Any) -> None:
        write_json_atomic(
            self.root / "run_state.json",
            {
                "run_id": self.run_id,
                "status": status,
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "committed_episodes": self.committed_count,
                "planned_episodes": self.config.total_episodes,
                "bytes_written": self.bytes_written,
                **extra,
            },
        )

    def finalize(self, status: str = "complete", **extra: Any) -> None:
        self.rebuild_episode_index()
        self.write_state(status, **extra)
        if status == "complete":
            write_json_atomic(
                self.root / "RUN_COMPLETE.json",
                {
                    "run_id": self.run_id,
                    "completed_utc": datetime.now(timezone.utc).isoformat(),
                    "committed_episodes": self.committed_count,
                    "config_sha256": self.config.digest,
                },
            )
