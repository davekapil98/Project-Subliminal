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
canonicalization**. Stage 1.1 through Stage 1.5 are complete. Project IRA SO-101 is
pinned and validated across 844,208 numeric rows with a decoded dual-camera
sample and prompt-group split. ArmnetBench v0.1 SO-101 is pinned and validated
across 1,127,881 rows; all three outcome classes decode from three AV1 cameras,
and a task-policy-cell split prevents rollout-family leakage. The SO101 MA
MultiTask simulation release is pinned and structurally validated across
358,006 rows with real H.264/AV1 edge-case decoding and a source-block split.
DROID raw v1.0.1 adds a success-bearing real Franka source: its complete
74,896-episode metadata namespace is inventoried, while a generation- and
checksum-pinned 75.74 MB subset validates 26 HDF5 episodes and six H.264
streams with a collection-lab split and strict identity-field redaction.

All three sources are `validated`, but none is admitted to training. The first
matched-update value gate improved held-out simulation error by 49.65% while
causing 12.90% median Project IRA forgetting, above the 10% limit. SO101 MA is
therefore explicitly `not_admitted`, and its remaining media is not authorized
for bulk download. See
[`docs/stage1_1_project_ira_report.md`](docs/stage1_1_project_ira_report.md)
and
[`docs/stage1_2_armnetbench_report.md`](docs/stage1_2_armnetbench_report.md),
and
[`docs/stage1_3_so101_ma_multitask_report.md`](docs/stage1_3_so101_ma_multitask_report.md).
The DROID qualification and its cross-embodiment boundary are documented in
[`docs/stage1_4_droid_raw_report.md`](docs/stage1_4_droid_raw_report.md).

Stage 1.5 completed its frozen action-free DROID visual value/forgetting gate.
The active 416-episode, 12.94 GB DROID/SO-101 selection is 7.06 GB below its
20 GB cap. DROID replay improved held-out DROID fixed-target error by 7.17%
median across three positive seeds, with 0.56% Project IRA and 0.43%
ArmnetBench median forgetting. The result admits only action-free temporal
visual pretraining for the JEPA encoder; no DROID Franka action is used as an
SO-101 target, and all broader DROID uses remain excluded. See the
[`predeclaration`](docs/stage1_5_droid_visual_predeclaration.md) and
[`final report`](docs/stage1_5_droid_visual_report.md).

Stage 2A has now started with an admitted-real, action-free JEPA encoder
bootstrap. Its committed laptop preflight verified the production data,
shared-view/fusion model, balanced sampler, camera dropout, EMA temporal
objective, atomic checkpoint and exact resume paths. The full 255.46M-parameter
encoder profile is not yet authorized for long training: the next required
step is its frozen throughput/memory benchmark on the RTX 3060 12 GB machine.
See the
[`Stage 2A predeclaration`](docs/stage2a_jepa_real_bootstrap_predeclaration.md)
and [`preflight report`](docs/stage2a_jepa_real_bootstrap_preflight_report.md).

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

Reproduce the Stage 1 source qualifications (about 435 MB for Project IRA,
665 MB for ArmnetBench, 568 MB for SO101 MA and 76 MB for DROID, all ignored
raw data):

```bash
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python scripts/qualify_project_ira_so101.py --download
.venv/bin/python scripts/qualify_armnetbench_so101.py --download
.venv/bin/python scripts/qualify_so101_ma_multitask_700.py --download
.venv/bin/python scripts/qualify_droid_raw.py --download
.venv/bin/python scripts/evaluate_stage1_3_body_value.py
.venv/bin/python scripts/qualify_so101_ma_multitask_700.py
.venv/bin/python scripts/qualify_droid_raw.py
.venv/bin/python scripts/prepare_stage1_5_visual_subset.py --verify
.venv/bin/python scripts/build_stage1_5_visual_cache.py
.venv/bin/python scripts/evaluate_stage1_5_droid_visual.py
.venv/bin/python scripts/train_stage2a_jepa_real_bootstrap.py --preflight
# Run this full-profile benchmark on the RTX 3060 12 GB machine:
.venv/bin/python scripts/train_stage2a_jepa_real_bootstrap.py --benchmark
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
