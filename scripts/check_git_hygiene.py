#!/usr/bin/env python3
"""Reject secrets, generated training data, and oversized files from Git."""

from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAX_SOURCE_FILE_BYTES = 25 * 1024 * 1024
PLACEHOLDERS = {".gitkeep", "README.md"}
GENERATED_ROOTS = {
    "artifacts",
    "checkpoints",
    "data",
    "dataset",
    "datasets",
    "lightning_logs",
    "mlruns",
    "output",
    "outputs",
    "recordings",
    "runs",
    "tensorboard",
    "wandb",
}
GENERATED_SUFFIXES = {
    ".arrow",
    ".avi",
    ".ckpt",
    ".engine",
    ".h5",
    ".hdf5",
    ".joblib",
    ".mkv",
    ".mp4",
    ".npy",
    ".npz",
    ".onnx",
    ".parquet",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
    ".safetensors",
    ".tfrecord",
}


def violation_reason(path_text: str, size: int = 0) -> str | None:
    path = PurePosixPath(path_text)
    if path.is_absolute() or ".." in path.parts:
        return "unsafe repository path"
    name = path.name.lower()
    if name == ".env" or name.startswith(".env.") or name.endswith(".env") or name == ".envrc":
        return "environment/secret file"
    if name.endswith((".pem", ".key")) or name.startswith(("credentials", "secrets")):
        return "credential material"
    if path.parts and path.parts[0].lower() in GENERATED_ROOTS and path.name not in PLACEHOLDERS:
        return "generated data/training artifact path"
    if path.suffix.lower() in GENERATED_SUFFIXES:
        return "generated model/data/media file type"
    if size > MAX_SOURCE_FILE_BYTES:
        return f"file exceeds the {MAX_SOURCE_FILE_BYTES // 1024 // 1024} MiB source limit"
    return None


def _git_paths(*arguments: str) -> list[str]:
    result = subprocess.run(
        ("git", "-C", str(PROJECT_ROOT), *arguments),
        check=True,
        capture_output=True,
    )
    return [item.decode("utf-8", "surrogateescape") for item in result.stdout.split(b"\0") if item]


def _paths(mode: str) -> list[str]:
    if mode == "staged":
        return _git_paths("diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR")
    return _git_paths("ls-files", "-z")


def audit(mode: str) -> list[tuple[str, str]]:
    violations: list[tuple[str, str]] = []
    for path_text in _paths(mode):
        local_path = PROJECT_ROOT / path_text
        size = local_path.lstat().st_size if local_path.exists() or local_path.is_symlink() else 0
        reason = violation_reason(path_text, size)
        if reason is not None:
            violations.append((path_text, reason))
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("staged", "tracked"), default="staged")
    args = parser.parse_args()
    violations = audit(args.mode)
    if not violations:
        print(f"Git hygiene audit passed ({args.mode} files).")
        return 0
    print("Refusing Git operation; prohibited files were found:", file=sys.stderr)
    for path, reason in violations:
        print(f"  {path}: {reason}", file=sys.stderr)
    print("Keep secrets and training data outside Git; do not bypass this guard.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
