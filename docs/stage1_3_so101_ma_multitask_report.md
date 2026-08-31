# Stage 1.3 report — SO101 MA MultiTask simulation qualification and value gate

Date: 31 August 2026

## Decisions

**PASS: Stage 1.3 source qualification is complete.** The public SO101 MA
MultiTask 700 release is structurally validated at immutable revision
`d4ae15a1044198bced5b7401123888068033451b`.

**FAIL: the source is not admitted to training.** A matched-update,
three-seed TinyBodyDynamics experiment improved its held-out normalized
one-step error, but median forgetting on Project IRA was 12.90%, above the
predeclared 10% limit. Unknown native actuator calibration, absent outcome
labels and the lack of a visual/world-model value result independently keep
the source blocked. The remaining full-source videos must not be acquired.

Authoritative source:

- [pinned dataset card](https://huggingface.co/datasets/CoRL2026-CSI/SO101-MA-MultiTask-700epi_10fps/blob/d4ae15a1044198bced5b7401123888068033451b/README.md)
- [official dataset repository](https://huggingface.co/datasets/CoRL2026-CSI/SO101-MA-MultiTask-700epi_10fps)
- [Apache-2.0 license](https://www.apache.org/licenses/LICENSE-2.0)

The source was public and ungated when verified. All seven declared upstream
repositories, their API-resolved IDs, immutable revisions, task/frame counts
and Apache-2.0 licenses are pinned in the object inventory.

## Source contract

| Field | Pinned value |
|---|---|
| Domain / embodiment | Isaac simulation, SO-101 follower, six source actuators |
| Scale | 700 episodes, 358,006 frames, seven tasks, 10 fps |
| Full tree | 47 files, 3,822,803,256 bytes |
| Modalities | Dual-camera RGB, state, action, language and merge provenance |
| Cameras | `top` and `left_wrist`, 640x480 yuv420p at 10 fps |
| Video codecs | Mixed by packed file: H.264 and AV1 are both present |
| Native actions | Absolute source-calibrated motor coordinates |
| Physical units/calibration | Unpublished in the merged release |
| Outcomes | No success, failure, quality or official task-success labels |

The merged release drops explicit upstream URDF-radian, end-effector,
gripper-helper, skill and subtask fields. One upstream source proves that its
native motor coordinates are affinely related to explicit radians, but the
merged source omits that transform. The adapter therefore preserves native
values without conversion and forbids task-space or physical cross-source
motor claims.

The source cards reference `SCRAPE-IsaacLab` collection code, but that URL
returned HTTP 404 at verification. The documented MA sources name Isaac Sim
5.1 and IsaacLab 2.3.2; CAP source cards omit exact simulator versions. These
limitations are provenance facts, not inferred replacements.

## Subset-first acquisition

The exact qualification subset is 567,845,886 bytes in 13 pinned objects:

- all metadata, task, merge-provenance and numeric trajectory objects needed
  to validate every one of the 358,006 rows; and
- six exact video objects for episodes 299, 447 and 599, covering a source
  camera-key remap, the unused-frame edge case and both H.264 and AV1.

All selected objects match their declared size and SHA-256. Raw and cleaned
samples remain under ignored `data/raw/public_sim/` and
`data/cleaned/public_sim/`; only compact registries, manifests, split selectors
and evidence are versioned.

The full source is not downloaded. Its videos dominate the remaining storage,
and the bounded numeric result does not establish their value for JEPA,
world-model or executive training.

## Integrity and canonicalization

The adapter verifies the complete numeric/metadata source:

- episode ranges are ordered, complete, contiguous and non-overlapping;
- global and per-episode frame indices, task indices and timestamps agree;
- all state/action vectors are float32 width six with no null, NaN or Inf;
- each of seven upstream task sources contributes exactly 100 episodes and
  its pinned frame total;
- merge provenance and declared dropped fields agree with all upstream pins;
  and
- native actions remain absolute and source-local; success and quality remain
  unknown.

`meta/tasks.parquet` is nonstandard: task text is stored in
`__index_level_0__`, not `task`. The adapter explicitly normalizes this field
without altering immutable raw data. Published aggregate statistics are also
stale: most counts are 420,171 or 89,862 instead of 358,006. Qualification
recomputes numeric evidence from the trajectory table and records the
published counts as untrusted.

Canonical samples contain source-native `q`, finite-difference `qdot`, the
previous absolute command and two native actions for three observations.
Success and quality are `None`; imitation eligibility is false while
structurally valid source-local prediction eligibility remains true. Three
edge-case episodes produced nine observations, six actions and 18 decoded RGB
frames. The sample summary SHA-256 is
`5080b60ca195269c620178a3926db523058d746a15e081ce5b30dccc31e15088`.

## Video edge cases

Episodes 299 and 447 decode from H.264; episode 599 decodes from AV1. Every
qualified stream passes codec, yuv420p, resolution, 10 fps and declared
interval-coverage checks.

Episode 447 declares nine more frames in each camera interval than its 362
trajectory rows require. Qualification accounts for those frames explicitly
and permits decoding only the aligned trajectory interval. Referenced packed
segments otherwise have no gaps or overlaps.

## Leakage-resistant split

The source publishes only a train split and has no session or random-seed key.
Project Subliminal groups every adjacent ten-episode collection-order block by
task. A rotating assignment holds one complete block per task for validation,
one distinct block per task for test and eight for training.

| Split | Episodes | Frames | Episodes per task |
|---|---:|---:|---:|
| Train | 560 | 287,520 | 80 |
| Validation | 70 | 35,497 | 10 |
| Test | 70 | 34,989 | 10 |

All 70 task/block groups are complete and disjoint, and every task is retained
in each split. The exact episode selectors and groups are versioned under
`data/splits/`.

## TinyBodyDynamics value and forgetting gate

The diagnostic uses the existing TinyBodyDynamics target. It derives a
source-local relative command from absolute action minus current state and
scales state, velocity and command using statistics fitted only on each
source's frozen training split. This does not assert physical coordinate
equivalence.

For each seed, a common checkpoint is trained on alternating Project IRA and
ArmnetBench batches. Two copies then receive the same 180-update continuation
budget, batch size and learning rate:

- control: alternating Project IRA and ArmnetBench;
- treatment: equal-cycle SO101 MA, Project IRA and ArmnetBench replay.

Evaluation uses up to 12,000 evenly spaced transitions from each frozen test
split. Positive forgetting means the treatment is worse than the matched
control.

| Seed | SO101 MA error improvement | Project IRA forgetting | ArmnetBench forgetting |
|---:|---:|---:|---:|
| 13 | 51.47% | 12.90% | 13.32% |
| 29 | 48.03% | 11.39% | 5.50% |
| 47 | 49.65% | 21.87% | 8.80% |
| **Median** | **49.65%** | **12.90%** | **8.80%** |

The rule requires positive median simulation improvement and no more than 10%
median forgetting on either real source. Project IRA exceeds the limit, so the
body-value gate fails. The source remains `validated` as an audited source but
has no admitted training use.

## Reproduce

```bash
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python scripts/qualify_so101_ma_multitask_700.py --download
.venv/bin/python scripts/evaluate_stage1_3_body_value.py
.venv/bin/python scripts/qualify_so101_ma_multitask_700.py
.venv/bin/python -m pytest -q
```

The downloader requests only the 13 pinned qualification objects, verifies
size and SHA-256, and never overwrites immutable raw data. Reproducibility
records refuse a content-changing rewrite.

## Boundary and next step

Stage 1.3 proves source identity, license, provenance, complete numeric
integrity, mixed-codec decoding, canonical conversion and leakage-resistant
splitting. It also records a fair negative training-admission result.

Further use requires a new gate, not reinterpretation of this result. Eligible
next work is to obtain/prove the native-to-URDF calibration, qualify a
success-bearing source, or test a separately bounded visual/world-model subset.
Until then, do not mix this source into training and do not download its
remaining videos.
