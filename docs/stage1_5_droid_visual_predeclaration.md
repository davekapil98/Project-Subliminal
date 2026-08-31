# Stage 1.5 pre-declaration — bounded DROID visual value gate

Date frozen: 31 August 2026

## Question and boundary

Stage 1.5 asks one narrow question: does bounded real DROID video improve an
action-free temporal JEPA representation while retaining performance on two
real SO-101 sources? This experiment does not establish cross-embodiment
action equivalence. DROID Franka state and action values cannot become SO-101
Motor, Body, JEPA World, Executive, or Language targets through this gate.

Only `jepa_encoder_action_free_temporal_pretraining` can be admitted, and only
if every frozen criterion below passes. A failed or incomplete run admits
nothing.

## Acquisition ceiling and selection

The hard source-object ceiling is 20,000,000,000 bytes with at least
5,000,000,000 bytes of free disk retained before acquisition. The cap includes
objects already present locally: source licenses, metadata/index files,
trajectory tables, and every selected video pack. Raw files and derived
caches remain Git-ignored. The committed object manifest contains public
provider paths and integrity pins, never raw metadata values.

The resolved selection is 12,687,618,078 bytes across 2,222 objects, leaving
7,312,381,922 bytes of cap headroom. DROID contributes 10,737,464,203 bytes,
Project IRA 779,332,130 bytes, and ArmnetBench 1,170,821,745 bytes. The public
inventory contained one high-ranked IRIS success episode with three videos but
without the full five-object contract; the selector deterministically skipped
it and admitted the next eligible rank. This was detected before acquisition
and before the protocol freeze commit.

DROID selection is frozen at 16 episodes from each of 13 labs and both
success/failure outcomes: 416 episodes total. Within each cell, eligible
episodes have exactly three non-stereo MP4 streams, one metadata JSON, and one
trajectory HDF5. The 26 Stage 1.4 qualification episodes are excluded. Episode
prefixes are ranked by SHA-256 of
`droid_raw_1_0_1:stage1.5:{lab}:{outcome}:{source_episode_prefix}`.

Complete collection labs retain the frozen Stage 1.4 partition: nine train,
CLVR/ILIAD validation, and PennPAL/REAL test. Eight temporal samples are chosen
per episode at a 0.5-second future horizon.

For matched SO-101 forgetting measurements:

- Project IRA uses episodes 0–89 for train and 880–889 for test, with its two
  cameras and an explicit missing-third-camera mask.
- ArmnetBench uses episodes 1549–1598 for train and 610–627 for test, with
  front, top, and wrist cameras.

Only the ten exact packed SO-101 video files in the frozen configuration are
eligible. Their revisions, sizes, and SHA-256 hashes are pinned.

## Representation and comparison

Input views have semantic order exterior-primary, exterior-secondary, wrist;
they are resized to 64×64 and scaled to float32 `[0,1]`. DROID proprioception
uses width 24. SO-101 `q`, `qdot`, and previous command use width 18 and are
zero-padded to 24. Each source is normalized only from its training episodes,
then a three-value source/embodiment one-hot is appended. Native actions are
not inputs.

The future target is deterministic and non-learned: all three future views are
pooled to 16×16 and projected with frozen seed 1515 into four 16-dimensional
tokens. This removes an EMA-target collapse shortcut from the small gate.

Each seed first receives 300 common SO-101 pretraining steps. Baseline and
treatment then start from the identical checkpoint and each receive 240
updates, batch size 32, the same optimizer and learning rate, and deterministic
sampling. Baseline alternates Project IRA and ArmnetBench; treatment alternates
DROID, Project IRA, and ArmnetBench. Seeds are 13, 29, and 47.

## Frozen decision rule

The gate passes only when all conditions hold:

1. Median relative DROID test-error improvement is greater than zero.
2. At least two of three seeds improve DROID test error.
3. Median relative forgetting is at most 10% on the Project IRA test set.
4. Median relative forgetting is at most 10% on the ArmnetBench test set.

Validation diagnostics cannot change the rule, thresholds, split, sample
count, training budget, or seeds. A code or data-contract defect requires a
new explicitly versioned protocol before another decision-bearing run.

## Reproduction order

```bash
.venv/bin/python scripts/prepare_stage1_5_visual_subset.py --plan
.venv/bin/python -m pytest -q tests/test_stage1_5_protocol.py
# Commit the config, planner, exact object manifest, pre-declaration, and tests.
.venv/bin/python scripts/prepare_stage1_5_visual_subset.py --download
.venv/bin/python scripts/prepare_stage1_5_visual_subset.py --verify
```

The selection/configuration commit must predate any new Stage 1.5 download or
model run. The evaluator and final evidence will be added only after that
freeze point.
