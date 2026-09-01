#!/usr/bin/env python3
"""Preflight or benchmark the admitted Stage 2A action-free JEPA bootstrap."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time
import tomllib
from typing import Any, Iterable

import numpy as np
import torch
from torch import Tensor, nn

from data.dataloaders.stage1_5_visual import Stage15VisualSamples
from models.jepa_encoder import (
    ActionFreeTemporalJEPA,
    JEPALatentPredictor,
    MultimodalJEPAEncoder,
)
from training.checkpointing import (
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
from training.losses.jepa import jepa_latent_loss
from training.precision import PrecisionPolicy, resolve_precision
from training.seed import seed_everything


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/training/stage2a_jepa_real_bootstrap.toml"
PREFLIGHT_OUTPUT_PATH = (
    PROJECT_ROOT / "data/manifests/stage2a_jepa_real_bootstrap.preflight.json"
)
BENCHMARK_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data/manifests/stage2a_jepa_real_bootstrap.rtx3060_benchmark.json"
)
REQUIRED_CLEAN_PATHS = (
    CONFIG_PATH,
    PROJECT_ROOT / "scripts/train_stage2a_jepa_real_bootstrap.py",
    PROJECT_ROOT / "src/models/jepa_encoder/model.py",
    PROJECT_ROOT / "src/models/jepa_encoder/pretraining.py",
    PROJECT_ROOT / "src/training/checkpointing.py",
)

DROID = "droid_raw_1_0_1"
PROJECT_IRA = "project_ira_so101_v1"
ARMNETBENCH = "armnetbench_so101_v01"
SOURCE_ORDER = (DROID, PROJECT_IRA, ARMNETBENCH)
SOURCE_TOKEN = {source: index for index, source in enumerate(SOURCE_ORDER)}
TRAIN_CACHE_FILES = {
    DROID: "droid_raw_1_0_1.train.npz",
    PROJECT_IRA: "project_ira_so101_v1.train.npz",
    ARMNETBENCH: "armnetbench_so101_v01.train.npz",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _write_json_once(path: Path, payload: dict[str, Any]) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            raise FileExistsError(f"refusing to overwrite evidence record {path}")
        return
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(serialized)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _git_commit() -> str:
    result = subprocess.run(
        ("git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _require_frozen_implementation() -> None:
    relative = [path.relative_to(PROJECT_ROOT).as_posix() for path in REQUIRED_CLEAN_PATHS]
    tracked = subprocess.run(
        ("git", "-C", str(PROJECT_ROOT), "ls-files", "--error-unmatch", *relative),
        check=False,
        capture_output=True,
    )
    if tracked.returncode:
        raise RuntimeError("commit the Stage 2A implementation before preflight")
    changed = subprocess.run(
        ("git", "-C", str(PROJECT_ROOT), "diff", "--quiet", "HEAD", "--", *relative),
        check=False,
    )
    if changed.returncode:
        raise RuntimeError("Stage 2A implementation differs from its committed freeze")


@dataclass(frozen=True)
class ProprioNormalization:
    mean: Tensor
    std: Tensor

    def normalize(self, value: Tensor) -> Tensor:
        return (value - self.mean.to(value.device)) / self.std.to(value.device)

    def evidence(self) -> dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "fit_scope": "Stage 1.5 source training cache only",
        }


@dataclass(frozen=True)
class BootstrapData:
    train: dict[str, Stage15VisualSamples]
    droid_validation: Stage15VisualSamples
    normalization: dict[str, ProprioNormalization]
    cache_manifest: dict[str, Any]


class BalancedSourceSampler:
    """Checkpointable source-local sample and camera-dropout RNG streams."""

    def __init__(self, seed: int, sources: Iterable[str] = SOURCE_ORDER) -> None:
        self.generators: dict[str, np.random.Generator] = {}
        for source in sources:
            material = f"stage2a:{seed}:{source}".encode("utf-8")
            source_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
            self.generators[source] = np.random.default_rng(source_seed)
        dropout_seed = int.from_bytes(
            hashlib.sha256(f"stage2a:{seed}:camera_dropout".encode()).digest()[:8],
            "big",
        )
        self.dropout_generator = np.random.default_rng(dropout_seed)

    def sample(self, source: str, count: int, batch_size: int) -> np.ndarray:
        if source not in self.generators or count < 1 or batch_size < 1:
            raise ValueError("invalid balanced-source sampling request")
        return self.generators[source].integers(
            0, count, size=batch_size, dtype=np.int64
        )

    def camera_dropout(
        self, camera_valid: np.ndarray, probability: float
    ) -> tuple[np.ndarray, int]:
        if camera_valid.ndim != 2 or camera_valid.dtype != np.bool_:
            raise ValueError("camera_valid must be a boolean [batch, views] array")
        if not 0.0 <= probability < 1.0:
            raise ValueError("camera dropout probability must be in [0, 1)")
        dropped = camera_valid.copy()
        draw = self.dropout_generator.random(camera_valid.shape)
        dropped &= draw >= probability
        for row in range(len(dropped)):
            provider_valid = np.flatnonzero(camera_valid[row])
            if not len(provider_valid):
                raise ValueError("a source sample has no provider-valid camera")
            if not dropped[row].any():
                retained = int(self.dropout_generator.choice(provider_valid))
                dropped[row, retained] = True
        removed = int(camera_valid.sum() - dropped.sum())
        return dropped, removed

    def state_dict(self) -> dict[str, Any]:
        return {
            "sources": {
                source: copy.deepcopy(generator.bit_generator.state)
                for source, generator in self.generators.items()
            },
            "camera_dropout": copy.deepcopy(
                self.dropout_generator.bit_generator.state
            ),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if set(state.get("sources", {})) != set(self.generators):
            raise ValueError("sampler checkpoint source set differs")
        for source, generator_state in state["sources"].items():
            self.generators[source].bit_generator.state = generator_state
        self.dropout_generator.bit_generator.state = state["camera_dropout"]


def _validate_contract(
    config: dict[str, Any], gate: dict[str, Any], cache: dict[str, Any]
) -> None:
    admission = config["admission"]
    data = config["data"]
    if config.get("stage") != "2A" or config.get("status") != (
        "frozen_before_nondecision_preflight"
    ):
        raise ValueError("unexpected or unfrozen Stage 2A bootstrap config")
    gate_path = PROJECT_ROOT / admission["evidence_path"]
    if sha256_file(gate_path) != admission["evidence_sha256"]:
        raise ValueError("Stage 1.5 admission evidence checksum differs")
    if gate.get("admission_decision") != admission["required_decision"]:
        raise ValueError("Stage 1.5 did not produce the required admission")
    if gate.get("admitted_uses") != [admission["required_use"]]:
        raise ValueError("Stage 1.5 admitted-use boundary differs")
    if gate.get("visual_value_gate_passed") is not True:
        raise ValueError("Stage 1.5 visual gate did not pass")
    if set(admission["still_excluded_uses"]) != set(gate["still_excluded_uses"]):
        raise ValueError("Stage 1.5 excluded-use boundary differs")
    cache_path = PROJECT_ROOT / data["cache_manifest_path"]
    if sha256_file(cache_path) != data["cache_manifest_sha256"]:
        raise ValueError("Stage 1.5 cache manifest checksum differs")
    object_path = PROJECT_ROOT / data["object_manifest_path"]
    if sha256_file(object_path) != data["object_manifest_sha256"]:
        raise ValueError("Stage 1.5 object manifest checksum differs")
    if cache.get("action_fields_included") is not False:
        raise ValueError("Stage 2A input cache must be action-free")
    if list(data["sources"]) != list(SOURCE_ORDER):
        raise ValueError("Stage 2A source order differs")
    if list(data["source_schedule"]) != list(SOURCE_ORDER):
        raise ValueError("Stage 2A balanced source schedule differs")
    if list(data["source_weights"]) != [1.0, 1.0, 1.0]:
        raise ValueError("Stage 2A source weights differ from 1:1:1")
    if data["action_fields_allowed"] is not False:
        raise ValueError("Stage 2A must prohibit action fields")
    if gate["input_evidence"]["privacy"]["action_fields_included"] is not False:
        raise ValueError("Stage 1.5 evidence reports action fields")


def load_data(config: dict[str, Any]) -> BootstrapData:
    gate_path = PROJECT_ROOT / config["admission"]["evidence_path"]
    cache_path = PROJECT_ROOT / config["data"]["cache_manifest_path"]
    gate = _json(gate_path)
    cache = _json(cache_path)
    _validate_contract(config, gate, cache)
    cache_root = cache_path.parent

    for key, entry in cache["caches"].items():
        path = PROJECT_ROOT / entry["path"]
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"cache object checksum differs for {key}")
        if entry.get("contains_action_fields") is not False:
            raise ValueError(f"cache object contains action fields for {key}")

    train = {
        source: Stage15VisualSamples.load(cache_root / filename)
        for source, filename in TRAIN_CACHE_FILES.items()
    }
    droid_validation = Stage15VisualSamples.load(
        cache_root / "droid_raw_1_0_1.validation.npz"
    )
    if sum(len(samples) for samples in train.values()) != int(
        config["data"]["train_samples"]
    ):
        raise ValueError("Stage 2A training sample total differs")
    if len(droid_validation) != int(config["data"]["droid_validation_samples"]):
        raise ValueError("Stage 2A DROID validation sample total differs")

    gate_normalization = gate["input_evidence"]["source_training_normalization"]
    normalization: dict[str, ProprioNormalization] = {}
    for source in SOURCE_ORDER:
        source_stats = gate_normalization[source]
        mean = torch.tensor(source_stats["proprio_mean"], dtype=torch.float32)
        std = torch.tensor(source_stats["proprio_std"], dtype=torch.float32)
        if mean.shape != (24,) or std.shape != (24,) or not torch.all(std > 0):
            raise ValueError(f"invalid pinned proprio normalization for {source}")
        normalization[source] = ProprioNormalization(mean=mean, std=std)
    return BootstrapData(
        train=train,
        droid_validation=droid_validation,
        normalization=normalization,
        cache_manifest=cache,
    )


def build_model(profile: dict[str, Any]) -> ActionFreeTemporalJEPA:
    encoder = MultimodalJEPAEncoder(**profile["encoder"])
    predictor = JEPALatentPredictor(**profile["predictor"])
    if encoder.world_token_count < 1:
        raise ValueError("encoder must produce world tokens")
    return ActionFreeTemporalJEPA(encoder, predictor)


def model_state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def parameter_counts(profile: dict[str, Any]) -> dict[str, int]:
    with torch.device("meta"):
        model = build_model(profile)
    return {
        "encoder": sum(value.numel() for value in model.encoder.parameters()),
        "predictor": sum(value.numel() for value in model.predictor.parameters()),
        "ema_target": sum(
            value.numel() for value in model.target_encoder.parameters()
        ),
        "trainable": sum(
            value.numel() for value in model.parameters() if value.requires_grad
        ),
        "checkpoint_total": sum(value.numel() for value in model.parameters()),
    }


def _source_token(source: str, batch_size: int, device: torch.device) -> Tensor:
    token = torch.zeros(batch_size, len(SOURCE_ORDER), device=device)
    token[:, SOURCE_TOKEN[source]] = 1.0
    return token


def training_batch(
    source: str,
    data: BootstrapData,
    sampler: BalancedSourceSampler,
    *,
    batch_size: int,
    camera_dropout_probability: float,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor, int]:
    samples = data.train[source]
    indices = sampler.sample(source, len(samples), batch_size)
    camera_valid_np, dropped_views = sampler.camera_dropout(
        samples.camera_valid[indices], camera_dropout_probability
    )
    context = torch.from_numpy(samples.context_rgb[indices]).to(
        device=device, dtype=torch.float32
    ) / 255.0
    future = torch.from_numpy(samples.future_rgb[indices]).to(
        device=device, dtype=torch.float32
    ) / 255.0
    camera_valid = torch.from_numpy(camera_valid_np).to(device=device)
    proprio = torch.from_numpy(samples.proprio[indices]).to(device=device)
    proprio = data.normalization[source].normalize(proprio)
    proprio = torch.cat(
        (proprio, _source_token(source, batch_size, device)), dim=-1
    )
    return context, future, proprio, camera_valid, dropped_views


def _feature_std(value: Tensor) -> float:
    return float(value.float().std(dim=(0, 1), correction=0).mean().detach().cpu())


def train_steps(
    model: ActionFreeTemporalJEPA,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    policy: PrecisionPolicy,
    data: BootstrapData,
    sampler: BalancedSourceSampler,
    config: dict[str, Any],
    *,
    start_step: int,
    stop_step: int,
    source_updates: dict[str, int] | None = None,
    dropped_views: int = 0,
) -> tuple[dict[str, Any], dict[str, int], int]:
    training = config["training"]
    schedule = tuple(config["source_schedule"])
    accumulation = int(training["gradient_accumulation_steps"])
    microbatch = int(training["microbatch_size"])
    source_updates = source_updates or {source: 0 for source in SOURCE_ORDER}
    losses: list[float] = []
    target_stds: list[float] = []
    prediction_stds: list[float] = []
    gradient_norms: list[float] = []
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for step in range(start_step, stop_step):
        source = schedule[step % len(schedule)]
        step_loss = 0.0
        for _ in range(accumulation):
            context, future, proprio, camera_valid, removed = training_batch(
                source,
                data,
                sampler,
                batch_size=microbatch,
                camera_dropout_probability=float(
                    config["objective"]["camera_dropout_probability"]
                ),
                device=policy.device,
            )
            dropped_views += removed
            with policy.autocast():
                output = model(context, future, proprio, camera_valid)
                micro_loss = jepa_latent_loss(
                    output.predicted_tokens, output.target_tokens
                )
                loss = micro_loss / accumulation
            if not torch.isfinite(micro_loss):
                raise FloatingPointError("Stage 2A temporal loss is not finite")
            scaler.scale(loss).backward()
            step_loss += float(micro_loss.detach().cpu()) / accumulation
            target_stds.append(_feature_std(output.target_tokens))
            prediction_stds.append(_feature_std(output.predicted_tokens))
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        gradient_norm = nn.utils.clip_grad_norm_(
            (value for value in model.parameters() if value.requires_grad),
            max_norm=float(training["gradient_clip_norm"]),
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("Stage 2A gradient norm is not finite")
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        model.update_target(float(config["objective"]["ema_momentum"]))
        losses.append(step_loss)
        gradient_norms.append(float(gradient_norm.detach().cpu()))
        source_updates[source] += 1
    metrics = {
        "loss_first_last": [losses[0], losses[-1]],
        "loss_mean": float(np.mean(losses)),
        "target_feature_std_mean": float(np.mean(target_stds)),
        "prediction_feature_std_mean": float(np.mean(prediction_stds)),
        "gradient_norm_max": max(gradient_norms),
    }
    return metrics, source_updates, dropped_views


def _deterministic_sensor_drop(camera_valid: np.ndarray, offset: int) -> np.ndarray:
    result = camera_valid.copy()
    for row in range(len(result)):
        valid = np.flatnonzero(result[row])
        if len(valid) > 1:
            result[row, valid[(offset + row) % len(valid)]] = False
    return result


@torch.no_grad()
def evaluate_droid_validation(
    model: ActionFreeTemporalJEPA,
    data: BootstrapData,
    policy: PrecisionPolicy,
    *,
    batch_size: int = 32,
) -> dict[str, float]:
    model.eval()
    samples = data.droid_validation
    sums = {"clean": 0.0, "sensor_drop": 0.0}
    values = 0
    target_stds: list[float] = []
    prediction_stds: list[float] = []
    for start in range(0, len(samples), batch_size):
        stop = min(start + batch_size, len(samples))
        context = torch.from_numpy(samples.context_rgb[start:stop]).to(
            device=policy.device, dtype=torch.float32
        ) / 255.0
        future = torch.from_numpy(samples.future_rgb[start:stop]).to(
            device=policy.device, dtype=torch.float32
        ) / 255.0
        provider_valid = samples.camera_valid[start:stop]
        clean_valid = torch.from_numpy(provider_valid).to(device=policy.device)
        dropped_valid = torch.from_numpy(
            _deterministic_sensor_drop(provider_valid, start)
        ).to(device=policy.device)
        proprio = torch.from_numpy(samples.proprio[start:stop]).to(device=policy.device)
        proprio = data.normalization[DROID].normalize(proprio)
        proprio = torch.cat(
            (proprio, _source_token(DROID, stop - start, policy.device)), dim=-1
        )
        with policy.autocast():
            clean = model(context, future, proprio, clean_valid)
            sensor_drop = model(context, future, proprio, dropped_valid)
        clean_error = nn.functional.smooth_l1_loss(
            clean.predicted_tokens.float(), clean.target_tokens.float(), reduction="sum"
        )
        drop_error = nn.functional.smooth_l1_loss(
            sensor_drop.predicted_tokens.float(),
            sensor_drop.target_tokens.float(),
            reduction="sum",
        )
        sums["clean"] += float(clean_error.cpu())
        sums["sensor_drop"] += float(drop_error.cpu())
        values += clean.target_tokens.numel()
        target_stds.append(_feature_std(clean.target_tokens))
        prediction_stds.append(_feature_std(clean.predicted_tokens))
    clean_loss = sums["clean"] / values
    dropped_loss = sums["sensor_drop"] / values
    return {
        "future_latent_smooth_l1": clean_loss,
        "sensor_drop_future_latent_smooth_l1": dropped_loss,
        "sensor_drop_relative_degradation": (dropped_loss - clean_loss) / clean_loss,
        "target_feature_std": float(np.mean(target_stds)),
        "prediction_feature_std": float(np.mean(prediction_stds)),
    }


def _optimizer(
    model: nn.Module, training: dict[str, Any]
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        (value for value in model.parameters() if value.requires_grad),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )


def _scaler(policy: PrecisionPolicy) -> torch.amp.GradScaler:
    return torch.amp.GradScaler(
        device=policy.device.type,
        enabled=policy.autocast_dtype == torch.float16,
    )


def _checkpoint_metadata(
    config: dict[str, Any], data: BootstrapData
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config_record = {
        "path": CONFIG_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "sha256": sha256_file(CONFIG_PATH),
        "payload": config,
    }
    dataset_record = {
        "path": config["data"]["cache_manifest_path"],
        "sha256": config["data"]["cache_manifest_sha256"],
        "payload": data.cache_manifest,
    }
    normalization = {
        source: stats.evidence() for source, stats in data.normalization.items()
    }
    return config_record, dataset_record, normalization


def run_preflight() -> dict[str, Any]:
    _require_frozen_implementation()
    config = _toml(CONFIG_PATH)
    profile = config["profiles"]["preflight"]
    training = profile["training"]
    policy = resolve_precision(
        device=str(training["device"]), precision=str(training["precision"])
    )
    seed = int(training["seed"])
    seed_everything(seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    data = load_data(config)
    model = build_model(profile).to(policy.device)
    optimizer = _optimizer(model, training)
    scaler = _scaler(policy)
    sampler = BalancedSourceSampler(seed)
    run_config = {
        "training": training,
        "source_schedule": config["data"]["source_schedule"],
        "objective": config["objective"],
    }
    steps = int(training["optimizer_steps"])
    checkpoint_step = steps // 2
    source_updates = {source: 0 for source in SOURCE_ORDER}
    first_metrics, source_updates, dropped_views = train_steps(
        model,
        optimizer,
        scaler,
        policy,
        data,
        sampler,
        run_config,
        start_step=0,
        stop_step=checkpoint_step,
        source_updates=source_updates,
    )

    artifact_root = PROJECT_ROOT / config["checkpoint"]["root"] / "preflight"
    checkpoint_path = artifact_root / "checkpoints" / f"step_{checkpoint_step:08d}.pt"
    config_record, dataset_record, normalization = _checkpoint_metadata(config, data)
    training_state = {
        "sampler": sampler.state_dict(),
        "rng": capture_rng_state(),
        "scaler": scaler.state_dict(),
        "source_updates": source_updates,
        "dropped_views": dropped_views,
    }
    before_hash = model_state_sha256(model)
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        step=checkpoint_step,
        config=config_record,
        dataset_manifest=dataset_record,
        normalization=normalization,
        precision=str(training["precision"]),
        git_commit=_git_commit(),
        training_state=training_state,
    )

    resumed_model = build_model(profile).to(policy.device)
    resumed_optimizer = _optimizer(resumed_model, training)
    resumed_scaler = _scaler(policy)
    metadata = load_checkpoint(
        checkpoint_path,
        model=resumed_model,
        optimizer=resumed_optimizer,
        map_location=policy.device,
    )
    if metadata["config"]["sha256"] != sha256_file(CONFIG_PATH):
        raise ValueError("resumed checkpoint config checksum differs")
    if metadata["dataset_manifest"]["sha256"] != config["data"][
        "cache_manifest_sha256"
    ]:
        raise ValueError("resumed checkpoint dataset checksum differs")
    after_hash = model_state_sha256(resumed_model)
    if before_hash != after_hash:
        raise ValueError("checkpoint model state changed during round-trip")
    resumed_state = metadata["training_state"]
    resumed_sampler = BalancedSourceSampler(seed)
    resumed_sampler.load_state_dict(resumed_state["sampler"])
    restore_rng_state(resumed_state["rng"])
    resumed_scaler.load_state_dict(resumed_state["scaler"])
    second_metrics, source_updates, dropped_views = train_steps(
        resumed_model,
        resumed_optimizer,
        resumed_scaler,
        policy,
        data,
        resumed_sampler,
        run_config,
        start_step=checkpoint_step,
        stop_step=steps,
        source_updates=dict(resumed_state["source_updates"]),
        dropped_views=int(resumed_state["dropped_views"]),
    )
    final_checkpoint = artifact_root / "checkpoints/last.pt"
    final_training_state = {
        "sampler": resumed_sampler.state_dict(),
        "rng": capture_rng_state(),
        "scaler": resumed_scaler.state_dict(),
        "source_updates": source_updates,
        "dropped_views": dropped_views,
    }
    save_checkpoint(
        final_checkpoint,
        model=resumed_model,
        optimizer=resumed_optimizer,
        step=steps,
        config=config_record,
        dataset_manifest=dataset_record,
        normalization=normalization,
        precision=str(training["precision"]),
        git_commit=_git_commit(),
        training_state=final_training_state,
    )
    validation = evaluate_droid_validation(resumed_model, data, policy)
    finite_values = [
        *first_metrics["loss_first_last"],
        *second_metrics["loss_first_last"],
        *validation.values(),
    ]
    criteria = {
        "frozen_inputs_verified": True,
        "equal_source_optimizer_steps": len(set(source_updates.values())) == 1,
        "action_fields_excluded": True,
        "finite_loss_gradients_and_metrics": all(math.isfinite(value) for value in finite_values),
        "camera_dropout_retained_valid_view": dropped_views > 0,
        "checkpoint_model_optimizer_sampler_rng_round_trip": before_hash == after_hash,
        "droid_validation_completed_without_locked_tests": True,
    }
    result = {
        "schema_version": 1,
        "stage": "2A",
        "round": config["round"],
        "kind": "nondecision_engineering_preflight",
        "passed": all(criteria.values()),
        "criteria": criteria,
        "scope": config["preflight_acceptance"]["decision_scope"],
        "git_commit": _git_commit(),
        "config": {
            "path": CONFIG_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256_file(CONFIG_PATH),
        },
        "admission_evidence": {
            "path": config["admission"]["evidence_path"],
            "sha256": config["admission"]["evidence_sha256"],
            "admitted_use": config["admission"]["required_use"],
        },
        "cache_manifest": {
            "path": config["data"]["cache_manifest_path"],
            "sha256": config["data"]["cache_manifest_sha256"],
        },
        "profile": "preflight",
        "resolved_device_type": policy.device.type,
        "precision": str(training["precision"]),
        "parameter_counts": {
            "preflight": parameter_counts(profile),
            "rtx3060_benchmark": parameter_counts(
                config["profiles"]["rtx3060_benchmark"]
            ),
        },
        "optimizer_steps": steps,
        "microbatch_size": int(training["microbatch_size"]),
        "gradient_accumulation_steps": int(
            training["gradient_accumulation_steps"]
        ),
        "source_optimizer_steps": source_updates,
        "camera_views_dropped": dropped_views,
        "training": {
            "before_checkpoint": first_metrics,
            "after_resume": second_metrics,
        },
        "checkpoint": {
            "format_version": int(metadata["format_version"]),
            "round_trip_model_sha256": before_hash,
            "resumed_at_optimizer_step": int(metadata["step"]),
            "final_optimizer_step": steps,
            "artifact_paths_committed": False,
        },
        "droid_validation": validation,
        "locked_test_splits_accessed": [],
        "long_run_authorized": bool(config["long_run"]["authorized"]),
        "next_required_action": "run the frozen full-profile throughput/memory benchmark on the RTX 3060 12 GB machine",
    }
    if not result["passed"]:
        raise RuntimeError("Stage 2A preflight criteria did not all pass")
    _write_json_once(PREFLIGHT_OUTPUT_PATH, result)
    return result


def _require_benchmark_hardware(training: dict[str, Any]) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "the full Stage 2A profile requires a CUDA-enabled PyTorch runtime on the RTX 3060"
        )
    device = torch.device("cuda")
    total_memory = int(torch.cuda.get_device_properties(device).total_memory)
    required = int(training["minimum_gpu_memory_bytes"])
    if total_memory < required:
        raise RuntimeError(
            f"the full profile requires at least {required} GPU bytes; detected {total_memory}"
        )
    return device


def run_benchmark() -> dict[str, Any]:
    _require_frozen_implementation()
    config = _toml(CONFIG_PATH)
    profile = config["profiles"]["rtx3060_benchmark"]
    training = profile["training"]
    device = _require_benchmark_hardware(training)
    policy = resolve_precision(device=str(device), precision=str(training["precision"]))
    seed = int(training["seed"])
    seed_everything(seed)
    data = load_data(config)
    model = build_model(profile).to(device)
    optimizer = _optimizer(model, training)
    scaler = _scaler(policy)
    sampler = BalancedSourceSampler(seed)
    run_config = {
        "training": training,
        "source_schedule": config["data"]["source_schedule"],
        "objective": config["objective"],
    }
    warmup = int(training["warmup_optimizer_steps"])
    measured = int(training["measured_optimizer_steps"])
    torch.cuda.reset_peak_memory_stats(device)
    _, source_updates, dropped_views = train_steps(
        model,
        optimizer,
        scaler,
        policy,
        data,
        sampler,
        run_config,
        start_step=0,
        stop_step=warmup,
    )
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    metrics, source_updates, dropped_views = train_steps(
        model,
        optimizer,
        scaler,
        policy,
        data,
        sampler,
        run_config,
        start_step=warmup,
        stop_step=warmup + measured,
        source_updates=source_updates,
        dropped_views=dropped_views,
    )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    measured_samples = (
        measured
        * int(training["microbatch_size"])
        * int(training["gradient_accumulation_steps"])
    )
    result = {
        "schema_version": 1,
        "stage": "2A",
        "round": config["round"],
        "kind": "rtx3060_full_profile_throughput_memory_benchmark",
        "git_commit": _git_commit(),
        "config_sha256": sha256_file(CONFIG_PATH),
        "profile": "rtx3060_benchmark",
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_total_memory_bytes": int(
            torch.cuda.get_device_properties(device).total_memory
        ),
        "precision": str(training["precision"]),
        "parameter_counts": parameter_counts(profile),
        "warmup_optimizer_steps": warmup,
        "measured_optimizer_steps": measured,
        "measured_samples": measured_samples,
        "elapsed_seconds": elapsed,
        "samples_per_second": measured_samples / elapsed,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "training": metrics,
        "source_optimizer_steps_total": source_updates,
        "camera_views_dropped": dropped_views,
        "long_run_authorized": False,
        "next_required_action": "review feasibility and freeze revision 2 with a long-run budget and numerical promotion thresholds",
    }
    _write_json_once(BENCHMARK_OUTPUT_PATH, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--preflight",
        action="store_true",
        help="run the laptop-scale data/objective/checkpoint/resume qualification",
    )
    mode.add_argument(
        "--benchmark",
        action="store_true",
        help="benchmark the full frozen profile on a CUDA GPU with at least 11 GB",
    )
    args = parser.parse_args()
    result = run_preflight() if args.preflight else run_benchmark()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
