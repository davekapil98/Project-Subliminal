# Stage 1.1 report — Project IRA SO-101 source qualification

Date: 29 August 2026

## Decision

**PASS: Stage 1.1 is complete.** Project IRA SO-101 is a validated Priority-A
source at immutable revision `bf6f568e9ec218fff838fb266e161724ec4f7e2c`.
It is not yet admitted to model training because held-out improvement is a later
public-data gate.

Authoritative evidence:

- [pinned dataset card](https://huggingface.co/datasets/Project-IRA/TPSoSe2026_Dataset_Full_Merged_Final_LeRobot_SO101_V1/blob/bf6f568e9ec218fff838fb266e161724ec4f7e2c/README.md)
- [official dataset repository](https://huggingface.co/datasets/Project-IRA/TPSoSe2026_Dataset_Full_Merged_Final_LeRobot_SO101_V1)
- [pinned Project IRA pipeline](https://github.com/Project-IRA/interactive-robotic-arm/blob/a339b00b09b8438f6feedcc3a3e42bb299a44034/docs/SO101_PIPELINE.md)
- [pinned LeRobot SO follower unit semantics](https://github.com/huggingface/lerobot/blob/4aaff99be4a1d81568c08c8f0296b41b40c99ec4/src/lerobot/robots/so_follower/so_follower.py)

## Source contract

| Field | Pinned value |
|---|---|
| Dataset | `Project-IRA/TPSoSe2026_Dataset_Full_Merged_Final_LeRobot_SO101_V1` |
| Domain / embodiment | Real SO-101 single-arm teleoperation |
| License | CC BY-SA 4.0; attribution and share-alike obligations preserved |
| Full repository size | 9,329,617,621 bytes |
| Dataset scale | 930 episodes, 844,208 frames, 93 prompt templates, 30 fps |
| Modalities | Two RGB cameras, six actuator positions, six absolute position commands, English task text |
| Cameras | `desk_view` 800x600 and `wrist_left` 640x480, H.264/yuv420p at 30 fps |
| Native units | First five actuator positions in degrees; gripper normalized to 0..100; timestamps in seconds |
| Task-space action | Not published and therefore not inferred |
| Published labels absent | Task success, per-episode quality, camera calibration and task-space frame |

The complete pinned values, limitations, evidence revisions and file hashes are
in `configs/datasets/registry/project_ira_so101_v1.toml`. The cleaning policy is
in `configs/datasets/cleaning_profiles/project_ira_so101_v1.toml`.

## Subset-first acquisition

The local qualification subset is 435,441,545 bytes, about 4.7% of the full
repository. It contains the source card, schema/statistics, all task and episode
metadata, the 20,596,943-byte trajectory table, and video file 000 for both
cameras. This covers episode 0 without downloading the remaining camera files.

All eight acquired objects match their pinned size and SHA-256. Raw source
objects remain under the ignored `data/raw/public_real/` tier. The canonical
sample remains under ignored `data/cleaned/public_real/`; neither can be added
to Git by the repository hygiene policy.

## Integrity results

The source adapter validates the full numeric table, not only episode 0:

- all 930 episode ranges are contiguous, non-overlapping and cover exactly
  844,208 rows;
- global indices are unique/contiguous, frame indices restart at zero, and
  episode indices agree with metadata ranges;
- timestamps agree with `frame_index / 30` within 10 microseconds;
- state and action columns are float32 vectors of width six with zero null,
  NaN or Inf values;
- all 93 task indices/text values are unique and each prompt maps to exactly ten
  episodes;
- native state/action extrema and per-step action deltas are recorded for later
  outlier policy, but no unsupported physical rejection threshold is invented;
- native actions are preserved as absolute source-calibrated commands; and
- the terminal same-frame action is omitted during canonical conversion so N
  observations form N-1 well-defined transitions.

## Camera and canonical-sample results

Both exact source videos pass codec, pixel format, resolution, frame-rate and
episode-segment coverage checks through PyAV:

| Camera | File frames | File duration | Episode-0 segment |
|---|---:|---:|---:|
| `desk_view` | 116,813 | 3,893.7667 s | 0.0–18.3 s |
| `wrist_left` | 76,032 | 2,534.4 s | 0.0–18.3 s |

Three aligned observations and two native actions from episode 0 were converted
to the canonical schema. Six real RGB frames decode with the expected camera
shapes, and metadata keeps source revision, license, task, units, frames and
unknown success/quality explicit. The deterministic sample-summary SHA-256 is
`de54b4c3414e3a66cde1dbcc4247b749129e3a961f76d303d37073f1ca13f557`.

## Leakage-resistant split

The source publishes only a train split. Project Subliminal freezes a derived,
task-family-stratified split by `task_index`, not adjacent frames or individual
episodes. Each exact prompt and its ten episodes stay together, and every task
family is represented in validation and test:

- train: 75 prompt groups / 750 episodes;
- validation: 9 prompt groups / 90 episodes; and
- test: 9 prompt groups / 90 episodes.

This prevents exact language-template leakage while retaining deterministic
reproduction from the dataset ID, source revision, task family and task index.

## Reproduce

```bash
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python scripts/qualify_project_ira_so101.py --download
.venv/bin/python -m pytest -q
```

The qualifier never overwrites an existing raw object. Existing files must
match their pin; a mismatch fails instead of silently repairing raw data.

## Boundary and next step

Stage 1.1 proves that one Priority-A source is pinned, legally reviewed,
checksum-reproducible, structurally valid, camera-decodable and canonically
convertible. It does not prove learned model improvement or authorize bulk
training.

Stage 1.2 should qualify the current ArmnetBench SO-101 revision next because
its success/failure/suboptimal labels complement Project IRA's prompt diversity.
Its current official card and license must be reverified before download.
