"""Isaac-launched CLI for hardware verification and Stage 1 recording."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any

from isaaclab.app import AppLauncher

from .config import load_stage1_config, validate_run_id


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Project Subliminal Stage 1 Isaac generator")
    parser.add_argument("--mode", choices=("verify", "record"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--report", type=Path)
    AppLauncher.add_app_launcher_args(parser)
    return parser


ARGS = _parser().parse_args()
CONFIG = load_stage1_config(ARGS.config)
ARGS.enable_cameras = True
ARGS.device = CONFIG.runtime.device
APP_LAUNCHER = AppLauncher(ARGS)
SIMULATION_APP = APP_LAUNCHER.app


# Isaac imports must happen after the application is running.
import imageio.v3 as iio  # noqa: E402
import numpy as np  # noqa: E402

from .isaac_backend import IsaacSO101Backend  # noqa: E402
from .isaac_worlds import build_env_cfg  # noqa: E402
from .provenance import collect_provenance  # noqa: E402
from .runner import collect_vector_episode  # noqa: E402
from .worlds import OBJECT_NAMES, SO101_JOINT_NAMES, WORLD_PROFILES  # noqa: E402
from .writer import DatasetQuotaReached, RunWriter, write_json_atomic  # noqa: E402


def _world_segmentation_requirements() -> dict[str, bool]:
    requirements: dict[str, bool] = {}
    for job in CONFIG.jobs:
        requirements[job.world] = requirements.get(job.world, False) or job.record_segmentation
    return requirements


def _verify_contract(world_name: str, backend: IsaacSO101Backend) -> dict[str, Any]:
    report = backend.contract_report()
    expected_envs = CONFIG.runtime.preflight_num_envs
    expected_camera = [expected_envs, CONFIG.camera.height, CONFIG.camera.width]
    checks = {
        "multiple_so101_instances": report["num_envs"] == expected_envs and expected_envs >= 2,
        "six_canonical_joints": report["joint_names"] == list(SO101_JOINT_NAMES),
        "one_rear_rgb_input": report["rgb_input_names"] == ["rear_camera"]
        and report["rear_camera_shape"][:3] == expected_camera
        and report["rear_camera_shape"][-1] >= 3,
        "all_rear_camera_frames_valid": len(report["rear_camera_per_env_pixel_std"]) == expected_envs
        and min(report["rear_camera_per_env_pixel_std"]) >= 1.0,
        "one_gripper_imu": report["imu_names"] == ["gripper_imu"]
        and report["gripper_imu_accel_shape"] == [expected_envs, 3]
        and report["gripper_imu_gyro_shape"] == [expected_envs, 3],
        "exact_object_poses": report["object_names"] == list(OBJECT_NAMES)
        and report["object_pose_shape"] == [expected_envs, len(OBJECT_NAMES), 7],
        "per_vial_contact_labels": report["contact_force_shape"] == [expected_envs, 3],
        "segmentation_mapping_if_enabled": not report["segmentation_enabled"]
        or report["segmentation_mapping_present"],
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"world {world_name} failed sensor contract: {', '.join(failed)}")
    report["checks"] = checks
    return report


def _preview_grid(frames: np.ndarray) -> np.ndarray:
    frames = np.asarray(frames)[..., :3]
    if np.issubdtype(frames.dtype, np.floating):
        frames = frames * (255.0 if frames.max(initial=0.0) <= 1.0 else 1.0)
    frames = np.clip(frames, 0, 255).astype(np.uint8)
    return np.concatenate([frame for frame in frames], axis=1)


def verify() -> int:
    report_path = ARGS.report or (ARGS.output / "stage1_preflight_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    provenance = collect_provenance(CONFIG)

    worlds: list[dict[str, Any]] = []
    for world_index, (world_name, segmentation) in enumerate(_world_segmentation_requirements().items()):
        profile = WORLD_PROFILES[world_name]
        env_cfg = build_env_cfg(
            CONFIG,
            profile,
            CONFIG.runtime.preflight_num_envs,
            segmentation,
            CONFIG.runtime.seed + world_index * 1000,
        )
        backend = IsaacSO101Backend(env_cfg, segmentation)
        try:
            first = backend.reset(CONFIG.runtime.seed + world_index * 1000)
            backend.step(np.repeat(backend.default_joint_position[None], backend.num_envs, axis=0))
            world_report = _verify_contract(world_name, backend)
            pixels = _preview_grid(first["rear_rgb"])
            preview = report_path.parent / f"preview_{world_name}.png"
            iio.imwrite(preview, pixels)
            world_report.update(
                {
                    "name": world_name,
                    "description": profile.description,
                    "preview": str(preview),
                    "preview_contains_all_preflight_environments": True,
                    "physics_hz": CONFIG.runtime.physics_hz,
                    "control_hz": CONFIG.runtime.control_hz,
                }
            )
            worlds.append(world_report)
        finally:
            backend.close()

    report = {
        "status": "PASS",
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": CONFIG.digest,
        "image": CONFIG.runtime.image,
        "provenance": provenance,
        "requirements": {
            "prebuilt_worlds": len(worlds),
            "parallel_instances_per_world": CONFIG.runtime.preflight_num_envs,
            "camera_inputs_per_robot": 1,
            "gripper_imus_per_robot": 1,
            "recording_started": False,
        },
        "worlds": worlds,
    }
    write_json_atomic(report_path, report)
    print(json.dumps(report, indent=2))
    return 0


def _episode_metadata(
    job: Any,
    profile: Any,
    episode_id: str,
    episode_index: int,
    cycle_seed: int,
    env_index: int,
    batch: Any,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    success = bool(batch.observations["success_bool"][-1, env_index])
    is_task_episode = job.policy in {"task_attempt", "failure_recovery"}
    any_collision = bool(batch.observations["collision_bool"][:, env_index].any())
    any_hard_limit = bool(batch.observations["hard_limit_bool"][:, env_index].any())
    any_grasp = bool(batch.observations["grasp_contact_bool"][:, env_index].any())
    quality = float(max(0.0, job.quality - 0.15 * any_collision - 0.2 * any_hard_limit))
    return {
        "episode_id": episode_id,
        "source_dataset": CONFIG.dataset.source_dataset,
        "source_version": CONFIG.dataset.source_version,
        "source_url": CONFIG.dataset.source_url,
        "license": CONFIG.dataset.license,
        "redistribution_terms": CONFIG.dataset.redistribution_terms,
        "robot_id": CONFIG.dataset.robot_id,
        "embodiment": CONFIG.dataset.embodiment,
        "task": job.task,
        "instruction": job.instruction,
        "policy": job.policy,
        "world": job.world,
        "world_profile": profile.to_dict(),
        "success": success if is_task_episode else None,
        "failure": (not success) if is_task_episode else None,
        "outcome": "success" if success else ("failure" if is_task_episode else "completed"),
        "quality": quality,
        "labels": {
            "any_grasp": any_grasp,
            "any_collision": any_collision,
            "any_hard_limit": any_hard_limit,
            "final_success": success,
        },
        "episode_index": episode_index,
        "cycle_seed": cycle_seed,
        "vector_environment_index": env_index,
        "random_seed_derivation": "runtime.seed + job_order*100000 + cycle_index",
        "steps": job.steps,
        "physics_hz": CONFIG.runtime.physics_hz,
        "control_hz": CONFIG.runtime.control_hz,
        "joint_names": list(SO101_JOINT_NAMES),
        "object_names": list(OBJECT_NAMES),
        "camera": {
            "name": CONFIG.camera.name,
            "placement": "rear stand; official D455 lightbox mount",
            "width": CONFIG.camera.width,
            "height": CONFIG.camera.height,
            "fps": CONFIG.camera.fps,
            "frame_index_array": "timestamp_sim_s",
        },
        "imu": {
            "name": CONFIG.imu.name,
            "prim_path": CONFIG.imu.prim_path,
            "include_gravity": CONFIG.imu.include_gravity,
        },
        "native_action": "absolute SO-101 joint-position target in radians after delay/deadband",
        "normalized_action": "relative joint-position delta in radians; gripper-frame task action unavailable",
        "exact_scene_labels": [
            "object_pose_env_wxyz",
            "contact_force_by_vial_n",
            "grasp_contact_bool",
            "collision_bool",
            "success_bool",
            "hard_limit_bool",
        ],
        "provenance": provenance,
    }


def _segmentation_mapping(backend: IsaacSO101Backend) -> dict[str, Any]:
    mapping = backend.scene["rear_camera"].data.info.get("instance_id_segmentation_fast")
    if not isinstance(mapping, dict) or "idToLabels" not in mapping:
        raise RuntimeError("segmentation recording requires an idToLabels mapping")
    return mapping


def record() -> int:
    if not ARGS.run_id:
        raise ValueError("--run-id is required in record mode")
    run_id = validate_run_id(ARGS.run_id)
    provenance = collect_provenance(CONFIG)
    writer = RunWriter(ARGS.output, run_id, CONFIG, provenance)
    jobs = sorted(enumerate(CONFIG.jobs), key=lambda item: (item[1].priority, item[0]))
    try:
        for job_order, job in jobs:
            profile = WORLD_PROFILES[job.world]
            num_envs = CONFIG.runtime.num_envs
            cycles = math.ceil(job.episodes / num_envs)
            env_cfg = build_env_cfg(
                CONFIG,
                profile,
                num_envs,
                job.record_segmentation,
                CONFIG.runtime.seed + job_order * 100000,
            )
            backend = IsaacSO101Backend(env_cfg, job.record_segmentation)
            try:
                for cycle in range(cycles):
                    start = cycle * num_envs
                    end = min(start + num_envs, job.episodes)
                    episode_indices = list(range(start, end))
                    episode_ids = {index: f"{job.name}_e{index:06d}" for index in episode_indices}
                    missing = [index for index in episode_indices if not writer.has_episode(episode_ids[index])]
                    if not missing:
                        continue
                    cycle_seed = CONFIG.runtime.seed + job_order * 100000 + cycle
                    print(
                        f"[record] job={job.name} cycle={cycle + 1}/{cycles} "
                        f"missing={len(missing)} seed={cycle_seed}",
                        flush=True,
                    )
                    batch = collect_vector_episode(
                        backend,
                        job.policy,
                        profile,
                        job.steps,
                        CONFIG.runtime.control_hz,
                        cycle_seed,
                        job.record_segmentation,
                    )
                    segmentation_mapping = _segmentation_mapping(backend) if job.record_segmentation else None
                    payloads = []
                    for episode_index in missing:
                        env_index = episode_index - start
                        episode_id = episode_ids[episode_index]
                        metadata = _episode_metadata(
                            job,
                            profile,
                            episode_id,
                            episode_index,
                            cycle_seed,
                            env_index,
                            batch,
                            provenance,
                        )
                        if segmentation_mapping is not None:
                            metadata["segmentation"] = {
                                "data_type": "instance_id_segmentation_fast",
                                "id_mapping": segmentation_mapping,
                            }
                            metadata["exact_scene_labels"].append("rear_segmentation")
                        payloads.append(batch.episode(env_index, episode_id, metadata))
                    counts = writer.write_batch(payloads)
                    print(
                        f"[record] committed={counts['written']} skipped={counts['skipped']} "
                        f"total={writer.committed_count}/{CONFIG.total_episodes}",
                        flush=True,
                    )
            finally:
                backend.close()
            writer.write_state("recording", last_completed_job=job.name)
        if writer.committed_count != CONFIG.total_episodes:
            raise RuntimeError(
                f"generation ended with {writer.committed_count}/{CONFIG.total_episodes} committed episodes"
            )
        writer.finalize("complete")
        print(f"[record] complete: {writer.root}", flush=True)
        return 0
    except DatasetQuotaReached as error:
        writer.finalize("quota_reached", reason=str(error))
        print(f"[record] stopped safely at storage guard: {error}", file=sys.stderr, flush=True)
        return 3
    except Exception as error:
        writer.finalize("failed", error=repr(error))
        raise


def main() -> int:
    if ARGS.mode == "verify":
        return verify()
    return record()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        SIMULATION_APP.close()
