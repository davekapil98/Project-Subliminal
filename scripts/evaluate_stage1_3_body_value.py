#!/usr/bin/env python3
"""Run the bounded Stage 1.3 SO-101 simulation value/forgetting gate.

The experiment uses source-local, train-only normalization. It does not
assert physical equivalence across the three native motor coordinate systems.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from data.adapters.armnetbench_so101 import (
    ArmnetBenchSO101Adapter,
    ArmnetBenchSourceSpec,
)
from data.adapters.project_ira_so101 import (
    ProjectIRASO101Adapter,
    ProjectIRASourceSpec,
)
from data.adapters.so101_ma_multitask_700 import (
    SO101MAMultiTaskAdapter,
    SO101MAMultiTaskSourceSpec,
)
from models.body_dynamics import TinyBodyDynamics
from qualify_so101_ma_multitask_700 import build_split_record


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = (
    PROJECT_ROOT / "data/manifests/so101_ma_multitask_700.value_gate.json"
)
PROJECT_IRA_REGISTRY = (
    PROJECT_ROOT / "configs/datasets/registry/project_ira_so101_v1.toml"
)
ARMNETBENCH_REGISTRY = (
    PROJECT_ROOT / "configs/datasets/registry/armnetbench_so101_v01.toml"
)
SIM_REGISTRY = (
    PROJECT_ROOT / "configs/datasets/registry/so101_ma_multitask_700.toml"
)
PROJECT_IRA_SPLIT = PROJECT_ROOT / "data/splits/project_ira_so101_v1.json"
ARMNETBENCH_SPLIT = PROJECT_ROOT / "data/splits/armnetbench_so101_v01.json"
PROJECT_IRA_RAW = PROJECT_ROOT / "data/raw/public_real/project_ira_so101"
ARMNETBENCH_RAW = PROJECT_ROOT / "data/raw/public_real/armnetbench_so101"
SIM_RAW = PROJECT_ROOT / "data/raw/public_sim/so101_ma_multitask_700"

SEEDS = (13, 29, 47)
BASELINE_STEPS = 240
TREATMENT_STEPS = 180
BATCH_SIZE = 192
EVALUATION_TRANSITIONS = 12_000
LEARNING_RATE = 3e-4
TREATMENT_LEARNING_RATE = 2e-4
FORGETTING_RELATIVE_TOLERANCE = 0.10
MODEL_CONFIG = {
    "joints": 6,
    "d_model": 32,
    "depth": 1,
    "num_heads": 4,
    "control_hz": 1.0,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _table_arrays(table: Any) -> dict[str, np.ndarray]:
    return {
        "state": np.asarray(
            table["observation.state"].to_pylist(), dtype=np.float32
        ),
        "action": np.asarray(table["action"].to_pylist(), dtype=np.float32),
        "episode_index": np.asarray(
            table["episode_index"].to_numpy(), dtype=np.int64
        ),
        "task_index": np.asarray(
            table["task_index"].to_numpy(), dtype=np.int64
        ),
    }


def _episode_ids_for_tasks(
    episode_index: np.ndarray,
    task_index: np.ndarray,
    selected_tasks: list[int],
) -> np.ndarray:
    _, first_rows = np.unique(episode_index, return_index=True)
    selected = np.isin(task_index[first_rows], selected_tasks)
    return episode_index[first_rows][selected]


@dataclass
class SourceTransitions:
    dataset_id: str
    revision: str
    state: np.ndarray
    action: np.ndarray
    episode_index: np.ndarray
    train_episode_ids: np.ndarray
    test_episode_ids: np.ndarray

    def __post_init__(self) -> None:
        expected = (len(self.episode_index), 6)
        if self.state.shape != expected or self.action.shape != expected:
            raise ValueError(
                f"{self.dataset_id} state/action arrays must have shape {expected}"
            )
        if not np.isfinite(self.state).all() or not np.isfinite(self.action).all():
            raise ValueError(f"{self.dataset_id} contains non-finite numeric values")
        if np.intersect1d(self.train_episode_ids, self.test_episode_ids).size:
            raise ValueError(f"{self.dataset_id} train/test episode sets overlap")
        same_next_episode = self.episode_index[:-1] == self.episode_index[1:]
        transition_rows = np.flatnonzero(same_next_episode)
        self.train_rows = transition_rows[
            np.isin(self.episode_index[transition_rows], self.train_episode_ids)
        ]
        self.test_rows = transition_rows[
            np.isin(self.episode_index[transition_rows], self.test_episode_ids)
        ]
        if not len(self.train_rows) or not len(self.test_rows):
            raise ValueError(f"{self.dataset_id} has an empty transition split")
        train_state = self.state[
            np.isin(self.episode_index, self.train_episode_ids)
        ].astype(np.float64)
        self.q_mean = train_state.mean(axis=0).astype(np.float32)
        q_std = train_state.std(axis=0).astype(np.float32)
        self.q_std = np.maximum(q_std, np.float32(1e-6))

    def selected_test_rows(self, limit: int) -> np.ndarray:
        if len(self.test_rows) <= limit:
            return self.test_rows
        positions = np.linspace(
            0, len(self.test_rows) - 1, num=limit, dtype=np.int64
        )
        return self.test_rows[positions]

    def tensors(self, rows: np.ndarray) -> tuple[Tensor, Tensor, Tensor]:
        q = self.state[rows]
        q_next = self.state[rows + 1]
        previous_rows = rows - 1
        has_previous = (
            (rows > 0)
            & (self.episode_index[previous_rows] == self.episode_index[rows])
        )
        q_previous = np.where(
            has_previous[:, None], self.state[previous_rows], q_next
        )
        q_normalized = (q - self.q_mean) / self.q_std
        qdot_normalized = (q - q_previous) / self.q_std
        relative_command = (self.action[rows] - q) / self.q_std
        q_next_normalized = (q_next - self.q_mean) / self.q_std
        qdot_next_normalized = (q_next - q) / self.q_std
        body_state = np.concatenate(
            (q_normalized, qdot_normalized), axis=1
        ).astype(np.float32)
        target = np.concatenate(
            (q_next_normalized, qdot_next_normalized), axis=1
        ).astype(np.float32)
        action = relative_command[:, None, None, :].astype(np.float32)
        return (
            torch.from_numpy(body_state),
            torch.from_numpy(action),
            torch.from_numpy(target),
        )

    def evidence(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "revision": self.revision,
            "rows": len(self.episode_index),
            "train_episodes": int(len(self.train_episode_ids)),
            "test_episodes": int(len(self.test_episode_ids)),
            "train_transitions": int(len(self.train_rows)),
            "test_transitions": int(len(self.test_rows)),
            "evaluated_test_transitions": int(
                len(self.selected_test_rows(EVALUATION_TRANSITIONS))
            ),
            "normalization": {
                "q_mean": self.q_mean.tolist(),
                "q_std": self.q_std.tolist(),
                "fit_scope": "all rows from the frozen training episodes only",
            },
        }


def load_sources() -> dict[str, SourceTransitions]:
    project_spec = ProjectIRASourceSpec.from_toml(PROJECT_IRA_REGISTRY)
    project_split = _load_json(PROJECT_IRA_SPLIT)
    if project_split["source_revision"] != project_spec.revision:
        raise ValueError("Project IRA split revision differs from its registry")
    project_adapter = ProjectIRASO101Adapter(
        PROJECT_IRA_RAW / project_spec.revision, project_spec
    )
    project_arrays = _table_arrays(project_adapter._data_table())
    project_train = _episode_ids_for_tasks(
        project_arrays["episode_index"],
        project_arrays["task_index"],
        project_split["task_indices"]["train"],
    )
    project_test = _episode_ids_for_tasks(
        project_arrays["episode_index"],
        project_arrays["task_index"],
        project_split["task_indices"]["test"],
    )

    arm_spec = ArmnetBenchSourceSpec.from_toml(ARMNETBENCH_REGISTRY)
    arm_split = _load_json(ARMNETBENCH_SPLIT)
    if arm_split["source_revision"] != arm_spec.revision:
        raise ValueError("ArmnetBench split revision differs from its registry")
    arm_adapter = ArmnetBenchSO101Adapter(
        ARMNETBENCH_RAW / arm_spec.revision, arm_spec
    )
    arm_arrays = _table_arrays(arm_adapter._data_table())

    sim_spec = SO101MAMultiTaskSourceSpec.from_toml(SIM_REGISTRY)
    sim_adapter = SO101MAMultiTaskAdapter(
        SIM_RAW / sim_spec.revision, sim_spec
    )
    sim_arrays = sim_adapter.trajectory_arrays()
    sim_split = build_split_record(sim_adapter)

    return {
        project_spec.dataset_id: SourceTransitions(
            project_spec.dataset_id,
            project_spec.revision,
            project_arrays["state"],
            project_arrays["action"],
            project_arrays["episode_index"],
            project_train,
            project_test,
        ),
        arm_spec.dataset_id: SourceTransitions(
            arm_spec.dataset_id,
            arm_spec.revision,
            arm_arrays["state"],
            arm_arrays["action"],
            arm_arrays["episode_index"],
            np.asarray(arm_split["episode_indices"]["train"], dtype=np.int64),
            np.asarray(arm_split["episode_indices"]["test"], dtype=np.int64),
        ),
        sim_spec.dataset_id: SourceTransitions(
            sim_spec.dataset_id,
            sim_spec.revision,
            sim_arrays["state"],
            sim_arrays["action"],
            sim_arrays["episode_index"],
            np.asarray(sim_split["episode_indices"]["train"], dtype=np.int64),
            np.asarray(sim_split["episode_indices"]["test"], dtype=np.int64),
        ),
    }


def _train_steps(
    model: TinyBodyDynamics,
    schedule: tuple[SourceTransitions, ...],
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> list[float]:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    generator = np.random.default_rng(seed)
    first_loss = 0.0
    last_loss = 0.0
    for step in range(steps):
        source = schedule[step % len(schedule)]
        rows = generator.choice(
            source.train_rows, size=batch_size, replace=True
        )
        body_state, action, target = source.tensors(rows)
        prediction = model(body_state, action).mean[:, 0, 0]
        loss = nn.functional.mse_loss(prediction, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        last_loss = float(loss.detach())
        if step == 0:
            first_loss = last_loss
    return [first_loss, last_loss]


@torch.no_grad()
def _evaluate(model: TinyBodyDynamics, source: SourceTransitions) -> float:
    model.eval()
    rows = source.selected_test_rows(EVALUATION_TRANSITIONS)
    squared_error_sum = 0.0
    value_count = 0
    for start in range(0, len(rows), 512):
        body_state, action, target = source.tensors(rows[start : start + 512])
        prediction = model(body_state, action).mean[:, 0, 0]
        squared_error_sum += float((prediction - target).square().sum())
        value_count += target.numel()
    return squared_error_sum / value_count


def _median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def run_experiment(
    *,
    baseline_steps: int = BASELINE_STEPS,
    treatment_steps: int = TREATMENT_STEPS,
    seeds: tuple[int, ...] = SEEDS,
) -> dict[str, Any]:
    if not seeds or baseline_steps < 1 or treatment_steps < 1:
        raise ValueError("the experiment requires seeds and positive step counts")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.use_deterministic_algorithms(True)
    sources = load_sources()
    project = sources["project_ira_so101_v1"]
    arm = sources["armnetbench_so101_v01"]
    sim = sources["so101_ma_multitask_700"]
    baseline_schedule = (project, arm)
    treatment_schedule = (sim, project, arm)
    per_seed: list[dict[str, Any]] = []

    for seed in seeds:
        torch.manual_seed(seed)
        model = TinyBodyDynamics(**MODEL_CONFIG)
        common_loss = _train_steps(
            model,
            baseline_schedule,
            steps=baseline_steps,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            seed=seed * 10 + 1,
        )
        baseline = copy.deepcopy(model)
        treatment = copy.deepcopy(model)
        baseline_loss = _train_steps(
            baseline,
            baseline_schedule,
            steps=treatment_steps,
            batch_size=BATCH_SIZE,
            learning_rate=TREATMENT_LEARNING_RATE,
            seed=seed * 10 + 2,
        )
        baseline_metrics = {
            source.dataset_id: _evaluate(baseline, source)
            for source in sources.values()
        }
        treatment_loss = _train_steps(
            treatment,
            treatment_schedule,
            steps=treatment_steps,
            batch_size=BATCH_SIZE,
            learning_rate=TREATMENT_LEARNING_RATE,
            seed=seed * 10 + 3,
        )
        treatment_metrics = {
            source.dataset_id: _evaluate(treatment, source)
            for source in sources.values()
        }
        sim_improvement = (
            baseline_metrics[sim.dataset_id]
            - treatment_metrics[sim.dataset_id]
        ) / baseline_metrics[sim.dataset_id]
        forgetting = {
            source.dataset_id: (
                treatment_metrics[source.dataset_id]
                - baseline_metrics[source.dataset_id]
            )
            / baseline_metrics[source.dataset_id]
            for source in (project, arm)
        }
        per_seed.append(
            {
                "seed": seed,
                "common_pretraining_loss_first_last": common_loss,
                "baseline_training_loss_first_last": baseline_loss,
                "treatment_training_loss_first_last": treatment_loss,
                "normalized_next_state_mse": {
                    "baseline": baseline_metrics,
                    "treatment": treatment_metrics,
                },
                "simulation_relative_improvement": sim_improvement,
                "real_source_relative_forgetting": forgetting,
            }
        )

    median_sim_improvement = _median(
        [row["simulation_relative_improvement"] for row in per_seed]
    )
    median_forgetting = {
        source.dataset_id: _median(
            [
                row["real_source_relative_forgetting"][source.dataset_id]
                for row in per_seed
            ]
        )
        for source in (project, arm)
    }
    body_gate_passed = median_sim_improvement > 0.0 and all(
        value <= FORGETTING_RELATIVE_TOLERANCE
        for value in median_forgetting.values()
    )
    blockers = [
        "native actuator calibration and physical units are unpublished; the merged source drops the upstream URDF-radian transform",
        "no published success, failure, quality or official task-success benchmark labels exist",
        "the bounded numeric result does not establish value for the 3.8 GB full video source or for JEPA/world/executive modules",
        "the source-card collection-code URL was unavailable (HTTP 404) at provenance verification",
    ]
    if not body_gate_passed:
        blockers.insert(
            0,
            "the bounded TinyBodyDynamics improvement/forgetting criterion did not pass",
        )
    return {
        "schema_version": 1,
        "gate": "stage1.3_tiny_body_dynamics_value_and_forgetting",
        "body_value_gate_passed": body_gate_passed,
        "metric": "source-normalized one-step next-state mean squared error",
        "methodology": {
            "coordinate_policy": (
                "absolute source action minus current source state, scaled by "
                "the source training-state standard deviation"
            ),
            "normalization_policy": (
                "fit q mean/std on each source's frozen training split only; "
                "apply those source-local statistics to its test split"
            ),
            "baseline": (
                "after common real-source pretraining, continue for the matched "
                "comparison budget on alternating Project IRA and ArmnetBench batches"
            ),
            "treatment": (
                "from the identical common checkpoint, continue for the same "
                "number of updates with deterministic 1:1:1 "
                "SO101-MA:Project-IRA:ArmnetBench replay"
            ),
            "fairness_control": (
                "baseline and treatment start from an identical checkpoint and "
                "receive equal update counts, batch sizes and learning rates"
            ),
            "scope": (
                "bounded diagnostic of reusable normalized numeric dynamics; "
                "not evidence of physical cross-source motor equivalence"
            ),
            "selection": (
                "all training transitions are sampleable; test evaluation uses "
                "at most 12000 evenly spaced transitions per frozen test split"
            ),
        },
        "configuration": {
            "model": "TinyBodyDynamics",
            "model_parameters": MODEL_CONFIG,
            "seeds": list(seeds),
            "common_pretraining_steps": baseline_steps,
            "matched_comparison_steps_per_branch": treatment_steps,
            "batch_size": BATCH_SIZE,
            "common_pretraining_learning_rate": LEARNING_RATE,
            "matched_comparison_learning_rate": TREATMENT_LEARNING_RATE,
            "baseline_source_schedule": [project.dataset_id, arm.dataset_id],
            "treatment_source_schedule": [
                sim.dataset_id,
                project.dataset_id,
                arm.dataset_id,
            ],
            "evaluation_transition_cap": EVALUATION_TRANSITIONS,
            "forgetting_relative_tolerance": FORGETTING_RELATIVE_TOLERANCE,
            "deterministic_algorithms": True,
        },
        "input_evidence": {
            "sources": {
                source.dataset_id: source.evidence()
                for source in sources.values()
            },
            "registries_sha256": {
                path.relative_to(PROJECT_ROOT).as_posix(): sha256_file(path)
                for path in (
                    PROJECT_IRA_REGISTRY,
                    ARMNETBENCH_REGISTRY,
                    SIM_REGISTRY,
                )
            },
            "frozen_real_splits_sha256": {
                path.relative_to(PROJECT_ROOT).as_posix(): sha256_file(path)
                for path in (PROJECT_IRA_SPLIT, ARMNETBENCH_SPLIT)
            },
        },
        "per_seed": per_seed,
        "aggregate": {
            "median_simulation_relative_improvement": median_sim_improvement,
            "median_real_source_relative_forgetting": median_forgetting,
            "criterion": (
                "median simulation improvement must be positive and median "
                "forgetting on each real source must be <= 0.10"
            ),
        },
        "admission_decision": "not_admitted",
        "admitted_uses": [],
        "admission_blockers": blockers,
        "next_eligible_work": [
            "obtain or prove an exact native-to-URDF calibration before any cross-source motor-target use",
            "qualify a success-bearing label source before imitation/executive supervision",
            "run a separate held-out visual/world-model value gate before acquiring the remaining packed videos",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run one seed and two steps per phase without writing evidence",
    )
    args = parser.parse_args()
    if args.smoke:
        report = run_experiment(
            baseline_steps=2, treatment_steps=2, seeds=(SEEDS[0],)
        )
    else:
        report = run_experiment()
        _write_json_once(OUTPUT_PATH, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
