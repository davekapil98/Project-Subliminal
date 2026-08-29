# Stage 1.2 report — ArmnetBench SO-101 source qualification

Date: 29 August 2026

## Decision

**PASS: Stage 1.2 is complete.** ArmnetBench v0.1 SO-101 is a validated
Priority-A source at immutable revision
`2e5e89aee0e7f081078d9d6ab3b279fc4b83ea84`. It is not yet admitted to model
training because held-out target-model improvement remains a later public-data
gate.

Authoritative evidence:

- [pinned dataset card](https://huggingface.co/datasets/armnet/armnetbench_v01_lerobot_so101/blob/2e5e89aee0e7f081078d9d6ab3b279fc4b83ea84/README.md)
- [official dataset repository](https://huggingface.co/datasets/armnet/armnetbench_v01_lerobot_so101)
- [official v1.0 revision API](https://huggingface.co/api/datasets/armnet/armnetbench_v01_lerobot_so101/revision/v1.0)
- [pinned LeRobot SO follower unit semantics](https://github.com/huggingface/lerobot/blob/4aaff99be4a1d81568c08c8f0296b41b40c99ec4/src/lerobot/robots/so_follower/so_follower.py)

The official `v1.0` tag and default branch resolved to the same dataset commit
when reverified. The source is public, ungated and Apache-2.0 licensed.

## Source contract

| Field | Pinned value |
|---|---|
| Dataset | `armnet/armnetbench_v01_lerobot_so101` |
| Domain / embodiment | Real SO-101 single-arm human demonstrations and learned-policy physical rollouts |
| License | Apache-2.0; license/notices, change statements and non-endorsement terms preserved |
| Full tree size | 60,592,338,170 bytes; Hub reports 60,643,574,258 stored bytes |
| Dataset scale | 2,499 episodes, 1,127,881 frames, 8 tasks, 20 fps |
| Outcomes | 915 successful, 1,532 failure, 52 suboptimal |
| Collection | 400 successful teleoperated demonstrations plus 2,099 rollouts from seven learned policy families |
| Modalities | Three RGB cameras, six actuator positions, six absolute position commands, English task text, outcome, reward/done and source-policy metadata |
| Cameras | `front` and `top` 1024x576; `wrist` 1280x720; AV1/yuv420p at 20 fps |
| Native units | First five actuator positions in degrees; gripper normalized to 0..100; timestamps in seconds |
| Task-space action | Not published and therefore not inferred |

The registry preserves the three-way `success_class` label. Binary `success`
is true only for strict success; both failure and suboptimal map to false. The
adapter marks strict successes imitation-eligible while retaining every
structurally valid class for prediction, body/world, critic and later recovery
work.

## Subset-first acquisition

The local qualification subset is 664,617,401 bytes, about 1.10% of the full
tree. It contains:

- the pinned card, source schema, tasks, complete episode index and published
  aggregate statistics;
- all 120 Parquet trajectory shards, totalling 53,757,034 bytes, so numeric
  validation covers every published frame; and
- one exact packed AV1 file for each camera, totalling 606,956,828 bytes and
  covering episodes 610–627.

The representative pack includes episode 610 (suboptimal), 611 (failure) and
621 (successful), all from the same tool-insertion task and diffusion-policy
family. This holds task and policy fixed while exercising label semantics.

All 128 acquired objects match their pinned byte size and SHA-256. Raw objects
remain under ignored `data/raw/public_real/`; the decoded canonical summary
remains under ignored `data/cleaned/public_real/`. Neither can enter Git.

## Numeric and label integrity

The adapter recomputes evidence from all 1,127,881 trajectory rows:

- all 2,499 episode ranges are ordered, contiguous, non-overlapping and cover
  the complete trajectory index;
- global indices are unique/contiguous, frame indices restart at zero, task
  indices agree with episode task text, and timestamps match
  `frame_index / 20` within 10 microseconds;
- state and action columns are float32 vectors of width six with no null, NaN
  or Inf values;
- every episode has exactly one terminal `next.done`;
- the 915 unit rewards occur only on terminal strict-success frames;
- binary success, three-way outcome, policy type and policy repository metadata
  agree, including 400 successful teleoperated episodes with empty policy
  repository IDs; and
- source-native actions remain absolute calibrated positions. No unsupported
  camera calibration, task-space frame or physical outlier threshold is
  invented.

The published `meta/stats.json` is stale: action/state counts are 1,127,881,
while timestamp, episode-index and global-index counts are 1,147,055. The
qualification records this mismatch and never treats those aggregate counts as
authoritative.

## Camera and canonical-sample results

All three exact source files pass codec, pixel format, resolution, frame-rate
and episode-segment coverage checks through PyAV:

| Camera | Packed file frames | File duration | Episode-627 segment |
|---|---:|---:|---:|
| `front` | 11,757 | 587.85 s | 333.1–350.6 s |
| `top` | 8,553 | 427.65 s | 333.1–350.6 s |
| `wrist` | 7,076 | 353.8 s | 333.1–350.6 s |

Three aligned observations and two native actions from each representative
outcome episode were converted to the canonical schema. Twenty-seven real RGB
frames decode at the expected camera shapes. Canonical qualities are 1.0 for
successful, 0.5 for suboptimal and 0.0 for failure. The deterministic sample
summary SHA-256 is
`9c0f6705c810a21762dbaa3acc6463acece67ba7e5d8e19c7beae1c429d005ac`.

The packed files contain unreferenced gaps where episodes were removed before
publication. Referenced segments never overlap and each duration exactly
matches episode length divided by 20 fps. The adapter therefore uses explicit
per-episode `from_timestamp`/`to_timestamp` metadata, records 205 front,
204 top and 204 wrist gaps (maximum 18.85 seconds), permits those unreferenced
regions, and rejects referenced overlap.

## Leakage-resistant split

The source publishes only a train split. Project Subliminal freezes an 8×8
task-policy matrix assignment:

- one complete policy-family cell per task goes to test;
- a distinct policy-family cell per task goes to validation; and
- the remaining six cells per task go to train.

Every rollout with the same exact task and policy family stays in one split.
The 48/8/8 cells are complete and disjoint:

| Split | Episodes | Frames | Failure | Suboptimal | Successful |
|---|---:|---:|---:|---:|---:|
| Train | 1,889 | 865,218 | 1,177 | 36 | 676 |
| Validation | 320 | 143,349 | 201 | 10 | 109 |
| Test | 290 | 119,314 | 154 | 6 | 130 |

Every task, policy family and outcome class appears in both held-out splits.
The exact episode selectors and task-policy cells are versioned under
`data/splits/`.

## Comparison with Project IRA

The sources overlap usefully on real SO-101 trajectories, multi-camera RGB,
language, proprioception and native absolute joint-position commands. Their
main value is complementary:

- ArmnetBench adds three-way outcomes, learned-policy failures/suboptimal
  behavior, seven policy families, precision manipulation and a top camera.
- Project IRA adds 93 prompt variants, 930 human demonstrations, different
  objects/environment/camera setup and broader language grounding.

This supports qualification, not automatic mixture. Bulk ArmnetBench media
download and training admission remain blocked until controlled held-out
experiments show target-module improvement without unacceptable forgetting.

## Reproduce

```bash
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python scripts/qualify_armnetbench_so101.py --download
.venv/bin/python -m pytest -q
```

The qualifier downloads only the 128 pinned objects, never overwrites an
existing raw object, verifies every hash, and refuses to rewrite a
reproducibility record with different content.

## Boundary and next step

Stage 1.2 proves that ArmnetBench is pinned, legally reviewed,
checksum-reproducible, fully validated for numeric trajectories,
camera-decodable across every outcome class, canonically convertible and
leakage-resistant. It does not prove learned-model improvement or authorize the
remaining roughly 60 GB download.

The next Stage 1 work should continue the master-spec Priority-A inventory and
define the controlled subset-value experiment comparing ArmnetBench with
Project IRA. Training admission remains a separate gate.
