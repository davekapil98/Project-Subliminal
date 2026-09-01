# Stage 1.5 DROID action-free visual value gate

Decision run: 1 September 2026

Outcome: **passed — narrowly admitted for action-free JEPA encoder temporal
visual pretraining only**.

## Decision

The frozen revision-3 gate passed all four pre-declared criteria. Adding
bounded DROID replay improved DROID test error for all three seeds, with a
7.17% median relative improvement. Median relative forgetting was 0.56% on
Project IRA and 0.43% on ArmnetBench, both well below the 10% ceiling.

This result admits only
`jepa_encoder_action_free_temporal_pretraining`. It does not admit DROID Franka
actions as SO-101 targets, SO-101 motor or body-dynamics supervision,
cross-embodiment native-action mixing, action-conditioned JEPA World training,
Executive training, or Language training.

## Frozen inputs

The decision used protocol revision 3, frozen before model training. Its exact
selection contains 2,222 immutable objects totaling 12,939,284,289 bytes,
7,060,715,711 bytes below the 20 GB cap:

| Source | Selected bytes | Gate role |
|---|---:|---|
| DROID raw 1.0.1 | 10,989,130,414 | treatment train; lab-held-out validation/test |
| Project IRA SO-101 | 779,332,130 | common/baseline/treatment train; forgetting test |
| ArmnetBench SO-101 | 1,170,821,745 | common/baseline/treatment train; forgetting test |

The cache builder produced 4,672 action-free samples from 584 source episodes.
Every sample has a distinct context/future pair in the frozen 0.50–0.60 second
window. DROID uses 288 train, 64 validation, and 64 test episodes; Project IRA
uses 90 train and 10 test episodes; ArmnetBench uses 50 train and 18 test
episodes. Raw objects and derived caches remain Git-ignored. The committed
evidence contains checksums and aggregate statistics, not raw metadata values
or episode identifiers.

## Matched comparison

For each seed (13, 29, 47), a 39,367-parameter TinyJEPAEncoder and
JEPALatentPredictor first received 300 common alternating Project
IRA/ArmnetBench updates. Baseline and treatment were then copied from the same
hashed model state and trained for 240 updates each, batch size 32, with fresh
identically configured AdamW optimizers:

- Baseline: Project IRA, ArmnetBench (120 updates each).
- Treatment: DROID, Project IRA, ArmnetBench (80 updates each).

Source-local deterministic random streams were cloned at the branch point, so
matching source-update ordinals use matching sample batches. Inputs contain
64×64 RGB views, a camera-validity mask, normalized observed proprioceptive
history, and a source token. There is no current action input or action target.
The future target is a frozen seed-1515 projection, not a learned target
encoder. Evaluation is source-training-normalized fixed-target mean squared
error. The decision run resolved to CPU and enabled deterministic algorithms.

## Results

| Seed | DROID test improvement | Project IRA forgetting | ArmnetBench forgetting |
|---:|---:|---:|---:|
| 13 | 3.06% | 0.71% | 0.27% |
| 29 | 7.17% | -0.40% | 0.43% |
| 47 | 8.04% | 0.56% | 0.90% |
| **Median** | **7.17%** | **0.56%** | **0.43%** |

The diagnostic DROID validation improvement was 7.56% median. It did not enter
the decision rule.

| Frozen criterion | Result |
|---|---|
| Median DROID test improvement > 0 | Pass (7.17%) |
| At least 2/3 DROID-positive seeds | Pass (3/3) |
| Median Project IRA forgetting <= 10% | Pass (0.56%) |
| Median ArmnetBench forgetting <= 10% | Pass (0.43%) |

## Reproduction

The exact selection and protocol rationale are in
[`stage1_5_droid_visual_predeclaration.md`](stage1_5_droid_visual_predeclaration.md).
With the frozen raw objects present:

```bash
.venv/bin/python scripts/prepare_stage1_5_visual_subset.py --verify
.venv/bin/python scripts/build_stage1_5_visual_cache.py
.venv/bin/python scripts/evaluate_stage1_5_droid_visual.py
.venv/bin/python -m pytest -q
```

The evaluator verifies the frozen config, protocol, object manifest, cache
manifest, every cache checksum, schema, horizon, and action exclusion before
training. It refuses to overwrite a different decision record. Full numerical
evidence is in
[`stage1_5_droid_visual.value_gate.json`](../data/manifests/stage1_5_droid_visual.value_gate.json).

## What this does not prove

This bounded tiny-model result establishes useful signal under one controlled
representation test. It does not prove task success, large-model scaling,
physical action equivalence, or safe robot control. The next eligible work is
to integrate DROID only into the action-free JEPA encoder pretraining path and
retain held-out SO-101 forgetting checks. Any broader DROID use requires a new
pre-declared gate.
