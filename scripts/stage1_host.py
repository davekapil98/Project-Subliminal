#!/usr/bin/env python3
"""Host-side checks and reports used by the one-command Stage 1 launcher."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sim.stage1.config import load_stage1_config  # noqa: E402
from sim.stage1.validator import verify_run_directory  # noqa: E402


def _run(*command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, check=False)


def _is_gb100_product(gpu_row: str) -> bool:
    """Accept the die name and NVIDIA's B100/B200 product names."""
    name = gpu_row.split(",", 1)[0].upper()
    return "GB100" in name or re.search(r"\bB(?:100|200)\b", name) is not None


def _preflight(args: argparse.Namespace) -> int:
    config = load_stage1_config(args.config)
    output = args.output.expanduser().resolve()
    failures: list[str] = []
    warnings: list[str] = []

    for binary in ("docker", "nvidia-smi"):
        if shutil.which(binary) is None:
            failures.append(f"required executable is missing: {binary}")

    gpu_rows: list[str] = []
    if shutil.which("nvidia-smi"):
        result = _run(
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        )
        if result.returncode:
            failures.append(f"nvidia-smi failed: {result.stderr.strip()}")
        else:
            gpu_rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            if not gpu_rows:
                failures.append("nvidia-smi found no GPUs")
            if not args.allow_non_gb100 and not any(_is_gb100_product(row) for row in gpu_rows):
                failures.append(
                    "no GB100/B100/B200 GPU detected "
                    "(use --allow-non-gb100 only for a deliberate smoke run)"
                )

    docker_runtime = ""
    if shutil.which("docker"):
        result = _run("docker", "info", "--format", "{{json .Runtimes}}")
        if result.returncode:
            failures.append(f"Docker daemon is unavailable: {result.stderr.strip()}")
        else:
            docker_runtime = result.stdout.strip()
            if "nvidia" not in docker_runtime.lower():
                failures.append("Docker does not report the NVIDIA container runtime")

    try:
        output.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix=".stage1-write-test-", dir=output, delete=True):
            pass
    except OSError as error:
        failures.append(f"output directory is not writable: {error}")

    disk = shutil.disk_usage(output if output.exists() else output.parent)
    minimum = int(config.dataset.min_free_disk_gib * 1024**3)
    planned_ceiling = int((config.dataset.max_output_gib + config.dataset.min_free_disk_gib) * 1024**3)
    if disk.free < minimum:
        failures.append(
            f"only {disk.free / 1024**3:.1f} GiB free; {config.dataset.min_free_disk_gib:.1f} GiB must remain"
        )
    elif disk.free < planned_ceiling:
        warnings.append(
            f"{disk.free / 1024**3:.1f} GiB free is below the configured output ceiling plus reserve "
            f"({config.dataset.max_output_gib + config.dataset.min_free_disk_gib:.1f} GiB); the safe quota may stop early"
        )

    report = {
        "status": "FAIL" if failures else "PASS",
        "config": str(args.config.resolve()),
        "config_sha256": config.digest,
        "container_image": config.runtime.image,
        "gpu": gpu_rows,
        "docker_runtimes": docker_runtime,
        "output": str(output),
        "free_disk_gib": round(disk.free / 1024**3, 1),
        "worlds": list(dict.fromkeys(job.world for job in config.jobs)),
        "parallel_environments": config.runtime.num_envs,
        "preflight_environments": config.runtime.preflight_num_envs,
        "planned_episodes": config.total_episodes,
        "planned_transitions": config.total_transitions,
        "rear_camera": f"1 x {config.camera.width}x{config.camera.height} RGB @ {config.camera.fps} Hz",
        "gripper_imu": f"1 x accel+gyro at {config.imu.prim_path}",
        "failures": failures,
        "warnings": warnings,
    }
    print(json.dumps(report, indent=2))
    return 2 if failures else 0


def _config_value(args: argparse.Namespace) -> int:
    config = load_stage1_config(args.config)
    values: dict[str, Any] = {
        "image": config.runtime.image,
        "output_root": config.dataset.output_root,
        "workshop_repository": config.runtime.workshop_repository,
        "workshop_commit": config.runtime.workshop_commit,
        "lerobot_commit": config.runtime.lerobot_commit,
        "digest": config.digest,
        "total_episodes": config.total_episodes,
    }
    print(values[args.name])
    return 0


def _show_isaac_report(args: argparse.Namespace) -> int:
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if report.get("status") != "PASS":
        raise ValueError("Isaac preflight report is not PASS")
    requirements = report["requirements"]
    print("\nIsaac scene verification: PASS")
    print(f"  Prebuilt worlds: {requirements['prebuilt_worlds']}")
    print(f"  SO-101 instances checked per world: {requirements['parallel_instances_per_world']}")
    print("  Sensor contract per robot: one rear stand RGB camera + one gripper IMU")
    for world in report["worlds"]:
        print(f"  - {world['name']}: preview={world['preview']}")
        for check, passed in world["checks"].items():
            print(f"      [{'OK' if passed else 'FAIL'}] {check}")
    print("  Recording started: NO")
    return 0


def _validate(args: argparse.Namespace) -> int:
    print(json.dumps(verify_run_directory(args.run, verify_payload=not args.fast), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    preflight = commands.add_parser("preflight")
    preflight.add_argument("--config", type=Path, required=True)
    preflight.add_argument("--output", type=Path, required=True)
    preflight.add_argument("--allow-non-gb100", action="store_true")
    preflight.set_defaults(func=_preflight)

    value = commands.add_parser("config-value")
    value.add_argument("--config", type=Path, required=True)
    value.add_argument(
        "name",
        choices=(
            "image",
            "output_root",
            "workshop_repository",
            "workshop_commit",
            "lerobot_commit",
            "digest",
            "total_episodes",
        ),
    )
    value.set_defaults(func=_config_value)

    show = commands.add_parser("show-isaac-report")
    show.add_argument("--report", type=Path, required=True)
    show.set_defaults(func=_show_isaac_report)

    validate = commands.add_parser("validate")
    validate.add_argument("--run", type=Path, required=True)
    validate.add_argument("--fast", action="store_true", help="check structure without re-hashing telemetry payloads")
    validate.set_defaults(func=_validate)
    return parser


if __name__ == "__main__":
    namespace = build_parser().parse_args()
    raise SystemExit(namespace.func(namespace))
