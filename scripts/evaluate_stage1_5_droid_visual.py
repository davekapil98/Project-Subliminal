#!/usr/bin/env python3
"""Run the frozen Stage 1.5 action-free DROID visual value gate."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
import tomllib
from typing import Any, Iterable

import numpy as np
import torch
from torch import Tensor, nn

from data.dataloaders.stage1_5_visual import (
    FrozenVisualTarget,
    SourceNormalization,
    Stage15VisualSamples,
)
from models.jepa_encoder import JEPALatentPredictor, TinyJEPAEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/training/stage1_5_droid_visual.toml"
ACTIVE_PROTOCOL_PATH = (
    PROJECT_ROOT / "configs/training/stage1_5_droid_visual_protocol_v3.toml"
)
OBJECT_MANIFEST_PATH = (
    PROJECT_ROOT
    / "configs/datasets/registry/stage1_5_visual_subset_v2.objects.json"
)
CACHE_ROOT = PROJECT_ROOT / "data/cache/stage1_5_visual"
CACHE_MANIFEST_PATH = CACHE_ROOT / "cache_manifest.json"
OUTPUT_PATH = (
    PROJECT_ROOT / "data/manifests/stage1_5_droid_visual.value_gate.json"
)

DROID = "droid_raw_1_0_1"
PROJECT_IRA = "project_ira_so101_v1"
ARMNETBENCH = "armnetbench_so101_v01"
SOURCE_ORDER = (PROJECT_IRA, ARMNETBENCH, DROID)
SOURCE_TOKEN = {name: index for index, name in enumerate(SOURCE_ORDER)}
CACHE_FILES = {
    (DROID, "train"): "droid_raw_1_0_1.train.npz",
    (DROID, "validation"): "droid_raw_1_0_1.validation.npz",
    (DROID, "test"): "droid_raw_1_0_1.test.npz",
    (PROJECT_IRA, "train"): "project_ira_so101_v1.train.npz",
    (PROJECT_IRA, "test"): "project_ira_so101_v1.test.npz",
    (ARMNETBENCH, "train"): "armnetbench_so101_v01.train.npz",
    (ARMNETBENCH, "test"): "armnetbench_so101_v01.test.npz",
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
            raise FileExistsError(
                f"refusing to overwrite reproducibility record {path}"
            )
        return
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(serialized)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def model_state_sha256(model: nn.Module) -> str:
    """Hash tensor names, shapes, dtypes, and bytes without serialization noise."""

    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class PreparedSplit:
    samples: Stage15VisualSamples
    normalized_target: Tensor

    def __post_init__(self) -> None:
        if self.normalized_target.shape != (len(self.samples), 4, 16):
            raise ValueError("normalized fixed targets must have shape [samples, 4, 16]")
        if self.normalized_target.dtype != torch.float32:
            raise ValueError("normalized fixed targets must be float32")
        if not torch.isfinite(self.normalized_target).all():
            raise ValueError("normalized fixed targets must be finite")


@dataclass(frozen=True)
class PreparedSource:
    dataset_id: str
    train: PreparedSplit
    evaluation: dict[str, PreparedSplit]
    normalization: SourceNormalization


class VisualJEPAGateModel(nn.Module):
    """The frozen tiny context encoder and latent predictor used by the gate."""

    def __init__(self, model_config: dict[str, Any]) -> None:
        super().__init__()
        self.encoder = TinyJEPAEncoder(
            image_size=int(model_config["image_size"]),
            patch_size=int(model_config["patch_size"]),
            max_views=int(model_config["max_views"]),
            proprio_dim=int(model_config["proprio_dim"]),
            d_model=int(model_config["d_model"]),
            depth=int(model_config["depth"]),
            num_heads=int(model_config["num_heads"]),
            world_tokens=int(model_config["world_tokens"]),
            bus_dim=int(model_config["bus_dim"]),
        )
        self.predictor = JEPALatentPredictor(
            bus_dim=int(model_config["bus_dim"]),
            d_model=int(model_config["d_model"]),
            depth=int(model_config["predictor_depth"]),
            num_heads=int(model_config["num_heads"]),
        )

    def forward(
        self,
        context_rgb: Tensor,
        normalized_proprio_with_source: Tensor,
        camera_valid: Tensor,
    ) -> Tensor:
        encoded = self.encoder(
            context_rgb,
            normalized_proprio_with_source,
            camera_valid=camera_valid,
        ).world_tokens
        predict_mask = torch.zeros(
            encoded.shape[:2], dtype=torch.bool, device=encoded.device
        )
        return self.predictor(encoded, predict_mask)


class SourceBatchSampler:
    """Independent deterministic streams make cross-branch batches comparable."""

    def __init__(self, seed: int, sources: Iterable[str] = SOURCE_ORDER) -> None:
        self._generators: dict[str, np.random.Generator] = {}
        for source in sources:
            material = f"stage1.5:{seed}:{source}".encode("utf-8")
            source_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
            self._generators[source] = np.random.default_rng(source_seed)

    def sample(self, source: str, count: int, batch_size: int) -> np.ndarray:
        if source not in self._generators:
            raise KeyError(f"unknown sampling source {source}")
        if count < 1 or batch_size < 1:
            raise ValueError("sampling requires positive source and batch sizes")
        return self._generators[source].integers(
            0, count, size=batch_size, dtype=np.int64
        )

    def clone(self) -> "SourceBatchSampler":
        return copy.deepcopy(self)


def _project_targets(
    samples: Stage15VisualSamples,
    projector: FrozenVisualTarget,
    normalization: SourceNormalization,
    *,
    batch_size: int = 256,
) -> Tensor:
    targets: list[Tensor] = []
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            future = torch.from_numpy(samples.future_rgb[start : start + batch_size])
            target = projector(future)
            targets.append(normalization.normalize_target(target).cpu())
    return torch.cat(targets, dim=0).contiguous()


def _validate_frozen_inputs(
    config: dict[str, Any],
    active_protocol: dict[str, Any],
    cache_manifest: dict[str, Any],
) -> None:
    if config.get("stage") != "1.5" or config.get("gate") != (
        "stage1.5_action_free_visual_jepa_value_and_forgetting"
    ):
        raise ValueError("unexpected Stage 1.5 base configuration")
    if active_protocol.get("protocol_revision") != 3:
        raise ValueError("the active Stage 1.5 protocol must be revision 3")
    if active_protocol.get("status") != "frozen_before_training":
        raise ValueError("the active Stage 1.5 protocol was not frozen before training")
    if active_protocol["base"]["config_sha256"] != sha256_file(CONFIG_PATH):
        raise ValueError("the Stage 1.5 base configuration hash changed")
    if active_protocol["base"]["object_manifest_sha256"] != sha256_file(
        OBJECT_MANIFEST_PATH
    ):
        raise ValueError("the Stage 1.5 object selection hash changed")
    if cache_manifest.get("protocol_revision") != 3:
        raise ValueError("the visual cache was not built under protocol revision 3")
    if cache_manifest.get("action_fields_included") is not False:
        raise ValueError("the Stage 1.5 cache must remain action-free")
    expected_hashes = {
        "config_sha256": sha256_file(CONFIG_PATH),
        "object_manifest_sha256": sha256_file(OBJECT_MANIFEST_PATH),
        "active_protocol_sha256": sha256_file(ACTIVE_PROTOCOL_PATH),
    }
    for key, expected in expected_hashes.items():
        if cache_manifest.get(key) != expected:
            raise ValueError(f"cache manifest {key} differs from the frozen input")
    sampling = active_protocol["sampling"]
    if sampling["minimum_temporal_horizon_seconds"] != 0.5:
        raise ValueError("the minimum visual horizon differs from 0.5 seconds")
    if sampling["maximum_temporal_horizon_seconds"] != 0.6:
        raise ValueError("the maximum visual horizon differs from 0.6 seconds")
    if not config["representation"]["action_policy"].startswith("No action input"):
        raise ValueError("the Stage 1.5 action exclusion changed")


def load_prepared_sources() -> tuple[dict[str, PreparedSource], dict[str, Any]]:
    config = _toml(CONFIG_PATH)
    active_protocol = _toml(ACTIVE_PROTOCOL_PATH)
    cache_manifest = _json(CACHE_MANIFEST_PATH)
    _validate_frozen_inputs(config, active_protocol, cache_manifest)

    loaded: dict[tuple[str, str], Stage15VisualSamples] = {}
    cache_evidence: dict[str, Any] = {}
    for (dataset_id, split), filename in CACHE_FILES.items():
        path = CACHE_ROOT / filename
        manifest_key = f"{dataset_id}:{split}"
        entry = cache_manifest["caches"].get(manifest_key)
        if not isinstance(entry, dict):
            raise ValueError(f"cache manifest lacks {manifest_key}")
        if entry.get("path") != path.relative_to(PROJECT_ROOT).as_posix():
            raise ValueError(f"cache path differs for {manifest_key}")
        actual_hash = sha256_file(path)
        if entry.get("sha256") != actual_hash:
            raise ValueError(f"cache checksum differs for {manifest_key}")
        samples = Stage15VisualSamples.load(
            path, image_size=int(config["representation"]["image_size"])
        )
        if entry.get("samples") != len(samples):
            raise ValueError(f"cache sample count differs for {manifest_key}")
        loaded[(dataset_id, split)] = samples
        cache_evidence[manifest_key] = {
            "path": entry["path"],
            "sha256": actual_hash,
            "samples": len(samples),
            "episodes": int(entry["episodes"]),
            "minimum_horizon_seconds": float(entry["minimum_horizon_seconds"]),
            "maximum_horizon_seconds": float(entry["maximum_horizon_seconds"]),
            "contains_action_fields": False,
        }

    representation = config["representation"]
    projector = FrozenVisualTarget(
        image_size=int(representation["image_size"]),
        pool_size=int(representation["target_pool_size"]),
        target_tokens=int(config["model"]["world_tokens"]),
        target_width=int(config["model"]["bus_dim"]),
        seed=int(representation["target_projection_seed"]),
    )
    sources: dict[str, PreparedSource] = {}
    normalization_evidence: dict[str, Any] = {}
    evaluation_splits = {
        DROID: ("validation", "test"),
        PROJECT_IRA: ("test",),
        ARMNETBENCH: ("test",),
    }
    for dataset_id in SOURCE_ORDER:
        train_samples = loaded[(dataset_id, "train")]
        normalization = SourceNormalization.fit(train_samples, projector)
        train = PreparedSplit(
            train_samples,
            _project_targets(train_samples, projector, normalization),
        )
        evaluation = {
            split: PreparedSplit(
                loaded[(dataset_id, split)],
                _project_targets(loaded[(dataset_id, split)], projector, normalization),
            )
            for split in evaluation_splits[dataset_id]
        }
        sources[dataset_id] = PreparedSource(
            dataset_id=dataset_id,
            train=train,
            evaluation=evaluation,
            normalization=normalization,
        )
        normalization_evidence[dataset_id] = normalization.evidence()

    evidence = {
        "base_config": {
            "path": CONFIG_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256_file(CONFIG_PATH),
        },
        "active_protocol": {
            "path": ACTIVE_PROTOCOL_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256_file(ACTIVE_PROTOCOL_PATH),
        },
        "object_manifest": {
            "path": OBJECT_MANIFEST_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256_file(OBJECT_MANIFEST_PATH),
        },
        "cache_manifest": {
            "path": CACHE_MANIFEST_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256_file(CACHE_MANIFEST_PATH),
        },
        "caches": cache_evidence,
        "source_training_normalization": normalization_evidence,
        "privacy": {
            "episode_identifiers_recorded": False,
            "raw_metadata_values_recorded": False,
            "action_fields_included": False,
        },
    }
    return sources, evidence


def _source_token(dataset_id: str, batch_size: int, device: torch.device) -> Tensor:
    token = torch.zeros(batch_size, len(SOURCE_ORDER), device=device)
    token[:, SOURCE_TOKEN[dataset_id]] = 1.0
    return token


def _batch_tensors(
    source: PreparedSource,
    indices: np.ndarray,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    samples = source.train.samples
    context = torch.from_numpy(samples.context_rgb[indices]).to(
        device=device, dtype=torch.float32
    ) / 255.0
    camera_valid = torch.from_numpy(samples.camera_valid[indices]).to(device=device)
    proprio = torch.from_numpy(samples.proprio[indices]).to(device=device)
    proprio = source.normalization.normalize_proprio(proprio)
    proprio = torch.cat(
        (proprio, _source_token(source.dataset_id, len(indices), device)), dim=-1
    )
    target = source.train.normalized_target[indices].to(device=device)
    return context, proprio, camera_valid, target


def _train_steps(
    model: VisualJEPAGateModel,
    sources: dict[str, PreparedSource],
    schedule: tuple[str, ...],
    sampler: SourceBatchSampler,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    gradient_clip_norm: float,
    device: torch.device,
) -> dict[str, Any]:
    if steps < 1 or not schedule:
        raise ValueError("training requires positive steps and a source schedule")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    first_loss = 0.0
    last_loss = 0.0
    source_updates = {name: 0 for name in schedule}
    for step in range(steps):
        dataset_id = schedule[step % len(schedule)]
        source = sources[dataset_id]
        indices = sampler.sample(dataset_id, len(source.train.samples), batch_size)
        context, proprio, camera_valid, target = _batch_tensors(
            source, indices, device
        )
        prediction = model(context, proprio, camera_valid)
        loss = nn.functional.mse_loss(prediction, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
        optimizer.step()
        last_loss = float(loss.detach().cpu())
        if step == 0:
            first_loss = last_loss
        source_updates[dataset_id] += 1
    return {
        "loss_first_last": [first_loss, last_loss],
        "source_updates": source_updates,
    }


@torch.no_grad()
def _evaluate(
    model: VisualJEPAGateModel,
    source: PreparedSource,
    split_name: str,
    *,
    device: torch.device,
    batch_size: int = 128,
) -> float:
    model.eval()
    split = source.evaluation[split_name]
    squared_error_sum = 0.0
    value_count = 0
    for start in range(0, len(split.samples), batch_size):
        stop = min(start + batch_size, len(split.samples))
        samples = split.samples
        context = torch.from_numpy(samples.context_rgb[start:stop]).to(
            device=device, dtype=torch.float32
        ) / 255.0
        camera_valid = torch.from_numpy(samples.camera_valid[start:stop]).to(
            device=device
        )
        proprio = torch.from_numpy(samples.proprio[start:stop]).to(device=device)
        proprio = source.normalization.normalize_proprio(proprio)
        proprio = torch.cat(
            (proprio, _source_token(source.dataset_id, stop - start, device)), dim=-1
        )
        target = split.normalized_target[start:stop].to(device=device)
        prediction = model(context, proprio, camera_valid)
        squared_error_sum += float((prediction - target).square().sum().cpu())
        value_count += target.numel()
    return squared_error_sum / value_count


def _median(values: Iterable[float]) -> float:
    values_array = np.asarray(list(values), dtype=np.float64)
    if not len(values_array) or not np.isfinite(values_array).all():
        raise ValueError("gate aggregation requires finite, non-empty metrics")
    return float(np.median(values_array))


def aggregate_gate(
    per_seed: list[dict[str, Any]],
    *,
    minimum_positive_seed_count: int,
    forgetting_relative_tolerance: float,
) -> dict[str, Any]:
    if not per_seed:
        raise ValueError("gate aggregation requires at least one seed record")
    droid_test = [float(row["droid_test_relative_improvement"]) for row in per_seed]
    droid_validation = [
        float(row["droid_validation_relative_improvement"]) for row in per_seed
    ]
    median_forgetting = {
        dataset_id: _median(
            row["so101_relative_forgetting"][dataset_id] for row in per_seed
        )
        for dataset_id in (PROJECT_IRA, ARMNETBENCH)
    }
    positive_count = sum(value > 0.0 for value in droid_test)
    median_test = _median(droid_test)
    criteria = {
        "median_droid_test_improvement_positive": median_test > 0.0,
        "minimum_positive_seed_count_met": positive_count
        >= minimum_positive_seed_count,
        "project_ira_forgetting_within_tolerance": median_forgetting[PROJECT_IRA]
        <= forgetting_relative_tolerance,
        "armnetbench_forgetting_within_tolerance": median_forgetting[ARMNETBENCH]
        <= forgetting_relative_tolerance,
    }
    return {
        "median_droid_test_relative_improvement": median_test,
        "median_droid_validation_relative_improvement": _median(droid_validation),
        "positive_droid_test_seed_count": positive_count,
        "median_so101_relative_forgetting": median_forgetting,
        "criteria": criteria,
        "passed": all(criteria.values()),
    }


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _evaluation_metrics(
    model: VisualJEPAGateModel,
    sources: dict[str, PreparedSource],
    device: torch.device,
) -> dict[str, float]:
    return {
        f"{DROID}:validation": _evaluate(
            model, sources[DROID], "validation", device=device
        ),
        f"{DROID}:test": _evaluate(model, sources[DROID], "test", device=device),
        f"{PROJECT_IRA}:test": _evaluate(
            model, sources[PROJECT_IRA], "test", device=device
        ),
        f"{ARMNETBENCH}:test": _evaluate(
            model, sources[ARMNETBENCH], "test", device=device
        ),
    }


def run_experiment(
    *,
    common_steps: int | None = None,
    comparison_steps: int | None = None,
    seeds: tuple[int, ...] | None = None,
    requested_device: str | None = None,
) -> dict[str, Any]:
    config = _toml(CONFIG_PATH)
    training = config["training"]
    evaluation = config["evaluation"]
    common_steps = int(
        training["common_pretraining_steps"] if common_steps is None else common_steps
    )
    comparison_steps = int(
        training["matched_comparison_steps_per_branch"]
        if comparison_steps is None
        else comparison_steps
    )
    seeds = tuple(int(value) for value in (training["seeds"] if seeds is None else seeds))
    if common_steps < 1 or comparison_steps < 1 or not seeds:
        raise ValueError("the experiment requires positive steps and at least one seed")
    device = _resolve_device(requested_device or str(training["device"]))
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.use_deterministic_algorithms(bool(training["deterministic_algorithms"]))
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False

    print("Loading and verifying seven frozen visual caches...", flush=True)
    sources, input_evidence = load_prepared_sources()
    baseline_schedule = tuple(str(value) for value in training["baseline_schedule"])
    treatment_schedule = tuple(str(value) for value in training["treatment_schedule"])
    per_seed: list[dict[str, Any]] = []

    for seed in seeds:
        print(f"Seed {seed}: common pretraining ({common_steps} updates)...", flush=True)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        common = VisualJEPAGateModel(config["model"]).to(device)
        common_sampler = SourceBatchSampler(seed)
        common_training = _train_steps(
            common,
            sources,
            baseline_schedule,
            common_sampler,
            steps=common_steps,
            batch_size=int(training["batch_size"]),
            learning_rate=float(training["common_pretraining_learning_rate"]),
            gradient_clip_norm=float(training["gradient_clip_norm"]),
            device=device,
        )
        common_hash = model_state_sha256(common)
        baseline = copy.deepcopy(common)
        treatment = copy.deepcopy(common)
        baseline_start_hash = model_state_sha256(baseline)
        treatment_start_hash = model_state_sha256(treatment)
        if not common_hash == baseline_start_hash == treatment_start_hash:
            raise RuntimeError("matched branches do not start from one checkpoint")
        baseline_sampler = common_sampler.clone()
        treatment_sampler = common_sampler.clone()

        print(f"Seed {seed}: matched baseline branch ({comparison_steps} updates)...", flush=True)
        baseline_training = _train_steps(
            baseline,
            sources,
            baseline_schedule,
            baseline_sampler,
            steps=comparison_steps,
            batch_size=int(training["batch_size"]),
            learning_rate=float(training["matched_comparison_learning_rate"]),
            gradient_clip_norm=float(training["gradient_clip_norm"]),
            device=device,
        )
        baseline_metrics = _evaluation_metrics(baseline, sources, device)

        print(f"Seed {seed}: matched DROID treatment branch ({comparison_steps} updates)...", flush=True)
        treatment_training = _train_steps(
            treatment,
            sources,
            treatment_schedule,
            treatment_sampler,
            steps=comparison_steps,
            batch_size=int(training["batch_size"]),
            learning_rate=float(training["matched_comparison_learning_rate"]),
            gradient_clip_norm=float(training["gradient_clip_norm"]),
            device=device,
        )
        treatment_metrics = _evaluation_metrics(treatment, sources, device)

        droid_test_key = f"{DROID}:test"
        droid_validation_key = f"{DROID}:validation"
        droid_test_improvement = (
            baseline_metrics[droid_test_key] - treatment_metrics[droid_test_key]
        ) / baseline_metrics[droid_test_key]
        droid_validation_improvement = (
            baseline_metrics[droid_validation_key]
            - treatment_metrics[droid_validation_key]
        ) / baseline_metrics[droid_validation_key]
        forgetting = {
            dataset_id: (
                treatment_metrics[f"{dataset_id}:test"]
                - baseline_metrics[f"{dataset_id}:test"]
            )
            / baseline_metrics[f"{dataset_id}:test"]
            for dataset_id in (PROJECT_IRA, ARMNETBENCH)
        }
        per_seed.append(
            {
                "seed": seed,
                "branch_start_checkpoint_sha256": common_hash,
                "common_training": common_training,
                "baseline_training": baseline_training,
                "treatment_training": treatment_training,
                "normalized_fixed_target_mse": {
                    "baseline": baseline_metrics,
                    "treatment": treatment_metrics,
                },
                "droid_test_relative_improvement": droid_test_improvement,
                "droid_validation_relative_improvement": droid_validation_improvement,
                "so101_relative_forgetting": forgetting,
                "final_model_sha256": {
                    "baseline": model_state_sha256(baseline),
                    "treatment": model_state_sha256(treatment),
                },
            }
        )
        print(
            f"Seed {seed}: DROID test improvement={droid_test_improvement:.6f}; "
            f"Project forgetting={forgetting[PROJECT_IRA]:.6f}; "
            f"Arm forgetting={forgetting[ARMNETBENCH]:.6f}",
            flush=True,
        )

    aggregate = aggregate_gate(
        per_seed,
        minimum_positive_seed_count=int(evaluation["minimum_positive_seed_count"]),
        forgetting_relative_tolerance=float(
            evaluation["forgetting_relative_tolerance"]
        ),
    )
    passed = bool(aggregate["passed"])
    failed_criteria = [
        name for name, value in aggregate["criteria"].items() if not value
    ]
    return {
        "schema_version": 1,
        "stage": "1.5",
        "protocol_revision": 3,
        "gate": "stage1.5_action_free_visual_jepa_value_and_forgetting",
        "visual_value_gate_passed": passed,
        "metric": "source-training-normalized future fixed-target mean squared error",
        "methodology": {
            "target": config["representation"]["target_policy"],
            "normalization": config["representation"]["proprio_policy"],
            "baseline": (
                "from the common SO-101 checkpoint, continue on alternating "
                "Project IRA and ArmnetBench batches"
            ),
            "treatment": (
                "from the identical common checkpoint, continue on deterministic "
                "DROID:Project-IRA:ArmnetBench replay"
            ),
            "fairness_control": training["fairness_control"],
            "sampling_control": (
                "source-local deterministic RNG streams are cloned at the branch; "
                "matching source-update ordinals receive matching sample batches"
            ),
            "optimizer_policy": (
                "both continuation branches use fresh, identically configured "
                "AdamW optimizers from the identical common model checkpoint"
            ),
            "action_policy": config["representation"]["action_policy"],
        },
        "configuration": {
            "model": config["model"],
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in VisualJEPAGateModel(config["model"]).parameters()
            ),
            "target_projection_seed": int(
                config["representation"]["target_projection_seed"]
            ),
            "seeds": list(seeds),
            "common_pretraining_steps": common_steps,
            "matched_comparison_steps_per_branch": comparison_steps,
            "batch_size": int(training["batch_size"]),
            "common_pretraining_learning_rate": float(
                training["common_pretraining_learning_rate"]
            ),
            "matched_comparison_learning_rate": float(
                training["matched_comparison_learning_rate"]
            ),
            "gradient_clip_norm": float(training["gradient_clip_norm"]),
            "baseline_source_schedule": list(baseline_schedule),
            "treatment_source_schedule": list(treatment_schedule),
            "resolved_device_type": device.type,
            "deterministic_algorithms": bool(training["deterministic_algorithms"]),
            "forgetting_relative_tolerance": float(
                evaluation["forgetting_relative_tolerance"]
            ),
            "minimum_positive_seed_count": int(
                evaluation["minimum_positive_seed_count"]
            ),
        },
        "input_evidence": input_evidence,
        "per_seed": per_seed,
        "aggregate": {
            **aggregate,
            "criterion": evaluation["admission_rule"],
        },
        "admission_decision": "admitted" if passed else "not_admitted",
        "admitted_uses": (
            [config["scope"]["admission_candidate"]] if passed else []
        ),
        "still_excluded_uses": config["scope"]["explicitly_excluded_uses"],
        "admission_blockers": (
            []
            if passed
            else [f"frozen gate criterion failed: {name}" for name in failed_criteria]
        ),
        "decision_scope": evaluation["decision_scope"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run one seed and two updates per phase without writing evidence",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default=None,
        help="override the frozen automatic device selection for diagnostics",
    )
    args = parser.parse_args()
    if args.smoke:
        frozen_seeds = tuple(int(value) for value in _toml(CONFIG_PATH)["training"]["seeds"])
        report = run_experiment(
            common_steps=2,
            comparison_steps=2,
            seeds=(frozen_seeds[0],),
            requested_device=args.device,
        )
    else:
        report = run_experiment(requested_device=args.device)
        _write_json_once(OUTPUT_PATH, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
