"""Strict, dependency-free configuration for the Stage 1 cloud run."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import tomllib
from typing import Any


_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")


@dataclass(frozen=True)
class RuntimeConfig:
    image: str
    isaac_lab_version: str
    workshop_repository: str
    workshop_commit: str
    lerobot_commit: str
    num_envs: int
    preflight_num_envs: int
    physics_hz: int
    control_hz: int
    device: str
    seed: int

    @property
    def decimation(self) -> int:
        return self.physics_hz // self.control_hz


@dataclass(frozen=True)
class CameraConfig:
    name: str
    width: int
    height: int
    fps: int
    video_backend: str
    codec: str
    crf: int
    preset: str


@dataclass(frozen=True)
class ImuConfig:
    name: str
    prim_path: str
    include_gravity: bool


@dataclass(frozen=True)
class DatasetConfig:
    output_root: str
    max_output_gib: float
    min_free_disk_gib: float
    writer_threads: int
    source_dataset: str
    source_version: str
    source_url: str
    license: str
    redistribution_terms: str
    robot_id: str
    embodiment: str


@dataclass(frozen=True)
class JobConfig:
    name: str
    world: str
    policy: str
    episodes: int
    steps: int
    priority: int
    record_rgb: bool
    record_segmentation: bool
    task: str
    instruction: str
    quality: float


@dataclass(frozen=True)
class Stage1Config:
    runtime: RuntimeConfig
    camera: CameraConfig
    imu: ImuConfig
    dataset: DatasetConfig
    jobs: tuple[JobConfig, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def digest(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def total_episodes(self) -> int:
        return sum(job.episodes for job in self.jobs)

    @property
    def total_transitions(self) -> int:
        return sum(job.episodes * job.steps for job in self.jobs)


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"missing [{name}] configuration section")
    return value


def _construct(cls: type, values: dict[str, Any], section: str):
    known = set(cls.__dataclass_fields__)
    unknown = sorted(set(values) - known)
    missing = sorted(known - set(values))
    if unknown:
        raise ValueError(f"unknown keys in {section}: {', '.join(unknown)}")
    if missing:
        raise ValueError(f"missing keys in {section}: {', '.join(missing)}")
    return cls(**values)


def _validate(config: Stage1Config) -> None:
    runtime = config.runtime
    if runtime.num_envs < 2:
        raise ValueError("runtime.num_envs must be at least 2 for parallel generation")
    if not 1 <= runtime.preflight_num_envs <= runtime.num_envs:
        raise ValueError("runtime.preflight_num_envs must be in [1, num_envs]")
    if runtime.physics_hz < 200:
        raise ValueError("runtime.physics_hz must be at least 200 for useful IMU acceleration")
    if runtime.physics_hz % runtime.control_hz:
        raise ValueError("runtime.physics_hz must be divisible by runtime.control_hz")
    if runtime.control_hz != config.camera.fps:
        raise ValueError("camera.fps must match runtime.control_hz for aligned canonical frames")
    if len(runtime.workshop_commit) != 40 or len(runtime.lerobot_commit) < 12:
        raise ValueError("runtime workshop and LeRobot revisions must be pinned commits")

    camera = config.camera
    if camera.name != "rear_camera":
        raise ValueError("the single Stage 1 RGB input must be named rear_camera")
    if camera.width < 64 or camera.height < 64:
        raise ValueError("camera resolution is too small")
    if camera.video_backend not in {"pyav", "npz"}:
        raise ValueError("camera.video_backend must be 'pyav' or 'npz'")
    if not 0 <= camera.crf <= 51:
        raise ValueError("camera.crf must be in [0, 51]")

    if config.imu.name != "gripper_imu" or "/Robot/gripper" not in config.imu.prim_path:
        raise ValueError("the IMU must be named gripper_imu and mounted on Robot/gripper")

    dataset = config.dataset
    if dataset.max_output_gib <= 0 or dataset.min_free_disk_gib <= 0:
        raise ValueError("dataset disk limits must be positive")
    if dataset.writer_threads < 1:
        raise ValueError("dataset.writer_threads must be positive")

    if not config.jobs:
        raise ValueError("at least one generation job is required")
    names: set[str] = set()
    from .worlds import WORLD_PROFILES

    for job in config.jobs:
        if not _RUN_ID.fullmatch(job.name):
            raise ValueError(f"invalid job name: {job.name!r}")
        if job.name in names:
            raise ValueError(f"duplicate job name: {job.name}")
        names.add(job.name)
        if job.world not in WORLD_PROFILES:
            raise ValueError(f"unknown world profile {job.world!r}")
        if job.policy not in {"smooth_random", "motor_sweep", "task_attempt", "failure_recovery"}:
            raise ValueError(f"unknown policy {job.policy!r}")
        if job.episodes < 1 or job.steps < 2:
            raise ValueError(f"job {job.name} needs positive episodes and at least two steps")
        if not 0.0 <= job.quality <= 1.0:
            raise ValueError(f"job {job.name} quality must be in [0, 1]")
        if not job.record_rgb:
            raise ValueError(f"job {job.name} must record the requested rear RGB input")


def load_stage1_config(path: str | Path) -> Stage1Config:
    """Load and validate a Stage 1 TOML file without importing Isaac Lab."""

    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    known_top = {"runtime", "camera", "imu", "dataset", "jobs"}
    unknown_top = sorted(set(raw) - known_top)
    if unknown_top:
        raise ValueError(f"unknown top-level sections: {', '.join(unknown_top)}")

    jobs_raw = raw.get("jobs")
    if not isinstance(jobs_raw, list):
        raise ValueError("configuration must contain one or more [[jobs]] sections")
    config = Stage1Config(
        runtime=_construct(RuntimeConfig, _section(raw, "runtime"), "runtime"),
        camera=_construct(CameraConfig, _section(raw, "camera"), "camera"),
        imu=_construct(ImuConfig, _section(raw, "imu"), "imu"),
        dataset=_construct(DatasetConfig, _section(raw, "dataset"), "dataset"),
        jobs=tuple(_construct(JobConfig, item, "jobs") for item in jobs_raw),
    )
    _validate(config)
    return config


def validate_run_id(run_id: str) -> str:
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError("run_id must use 1-96 letters, numbers, '.', '_' or '-'")
    return run_id
