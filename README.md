# Project Subliminal

Project Subliminal is a modular, fully local robot brain for the LeRobot SO-101
arm. The authoritative architecture and methodology are defined by
[`Modular_Robot_Brain_Functional_Master_Spec_v1.3.docx`](./Modular_Robot_Brain_Functional_Master_Spec_v1.3.docx).

The v1.3 baseline is public-dataset-first and requires **no paid cloud
compute**. The RTX 3050 laptop is used for tiny correctness/learning models;
the RTX 3060 12 GB machine is the sequential specialist trainer. Public
synthetic and real multi-embodiment datasets provide broad pretraining before
public and self-collected SO-101 specialization.

## Current project phase

Stage 0 contains small, readable implementations of all eight neural modules,
typed bus interfaces, deterministic safety/MPC primitives, checkpoint/data
contracts, educational reimplementations, tiny-overfit tests, and a complete
mocked closed loop. Its purpose is code correctness, not robotic competence.

Stage 1 under v1.3 is **public dataset acquisition, validation and
canonicalization**. Stage 1.1 and Stage 1.2 are complete. Project IRA SO-101 is
pinned and validated across 844,208 numeric rows with a decoded dual-camera
sample and prompt-group split. ArmnetBench v0.1 SO-101 is pinned and validated
across 1,127,881 rows; all three outcome classes decode from three AV1 cameras,
and a task-policy-cell split prevents rollout-family leakage.

Both sources are `validated`, not yet admitted to training. Held-out
target-model improvement belongs to the later public-data gate, and the
remaining ArmnetBench media is not authorized for bulk download. See
[`docs/stage1_1_project_ira_report.md`](docs/stage1_1_project_ira_report.md)
and
[`docs/stage1_2_armnetbench_report.md`](docs/stage1_2_armnetbench_report.md).

The former v1.2 GB100/Isaac generator is retained only as excluded historical
reference. Its launch entry points are deliberately disabled; see
[`docs/legacy_v12_isaac.md`](docs/legacy_v12_isaac.md).

## Stage 0 quick start

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python scripts/run_stage0_gate.py
.venv/bin/python scripts/run_stage0_demo.py
```

Reproduce the Stage 1 source qualifications (about 435 MB for Project IRA and
665 MB for ArmnetBench, all ignored raw data):

```bash
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python scripts/qualify_project_ira_so101.py --download
.venv/bin/python scripts/qualify_armnetbench_so101.py --download
```

The demo runs this synthetic path:

```text
text -> Language -> TASK_GOAL -> Executive -> MOTOR_GOAL
     -> Motor Cortex candidates -> Body/World predictions
     -> deterministic MPC score -> safety clamp -> mock robot
     -> EXECUTION_RESULT -> memory
```

It does not control physical hardware and does not claim task performance.

## Repository and data rules

- Public sources are registered with exact revision, URL, license, domain,
  embodiment, modalities, action semantics and checksums before training use.
- `data/raw/` is immutable and separates `public_sim`, `public_real`, and
  `own_real` sources.
- Cleaned, normalized, split and model-specific shard tiers are reproducible
  from source revisions, manifests and code.
- Raw datasets, training shards, checkpoints, recordings and `.env` files are
  never committed; small versioned registry/split/preprocessing metadata is.
- Native actions are preserved. Cross-embodiment actions are normalized only
  when units, kinematics and coordinate frames are proven.
- Physical fields remain typed numeric tensors and learned output is always
  wrapped by deterministic safety checks.
- Model architecture, trainers, data conversion, control, evaluation and
  deployment remain separate.
- Every important neural concept receives a readable implementation and test
  before an optimized production implementation.

See [`docs/stage0.md`](docs/stage0.md),
[`docs/v1.3_alignment_report.md`](docs/v1.3_alignment_report.md), and
[`docs/tensor_shapes.md`](docs/tensor_shapes.md).
