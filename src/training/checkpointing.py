"""Reproducible checkpoint save/load helpers."""

import os
from pathlib import Path
import tempfile
from typing import Any

import torch
from torch import nn


CHECKPOINT_FORMAT_VERSION = 2


def capture_rng_state() -> dict[str, Any]:
    """Capture torch RNG state needed for an exact local resume."""

    return {
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore RNG state captured by :func:`capture_rng_state`."""

    if "torch_cpu" not in state or "torch_cuda" not in state:
        raise ValueError("checkpoint RNG state is incomplete")
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    config: dict[str, Any],
    dataset_manifest: dict[str, Any],
    normalization: dict[str, Any],
    precision: str,
    git_commit: str = "uncommitted",
    training_state: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "config": config,
        "dataset_manifest": dataset_manifest,
        "normalization": normalization,
        "precision": precision,
        "git_commit": git_commit,
        "training_state": training_state or {},
    }
    with tempfile.NamedTemporaryFile(
        "wb", dir=path.parent, suffix=".pt.part", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if checkpoint.get("format_version") not in {1, CHECKPOINT_FORMAT_VERSION}:
        raise ValueError("unsupported checkpoint format")
    required = {
        "step",
        "model",
        "optimizer",
        "config",
        "dataset_manifest",
        "normalization",
        "precision",
        "git_commit",
    }
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"checkpoint is missing required fields: {sorted(missing)}")
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and checkpoint["optimizer"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    checkpoint.setdefault("training_state", {})
    return checkpoint
