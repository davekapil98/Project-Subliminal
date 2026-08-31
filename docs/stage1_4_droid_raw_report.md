# Stage 1.4 report — DROID raw real-robot qualification

Date: 31 August 2026

## Decision

**PASS: Stage 1.4 source qualification is complete.** DROID raw v1.0.1 is
registered as a validated Priority-B real-data source. The complete public
metadata namespace was inventoried, and a balanced 26-episode qualification
subset—one success and one failure from each of 13 collection labs—was pinned
by immutable GCS generation, size, MD5 and SHA-256.

**The source is not admitted to training.** This result proves provenance,
integrity, decoding and canonical conversion; it does not prove target-model
value. DROID uses a Franka Panda rather than SO-101, its raw release is about
8.7 TB, and its native controller coordinates must remain embodiment-specific.
No bulk acquisition or training mixture is authorized by this qualification.

Primary sources:

- [official DROID dataset documentation](https://droid-dataset.github.io/droid/docs/the-droid-dataset/)
- [pinned collection code](https://github.com/droid-dataset/droid/tree/33ae6a67274f36d2e29525b86f23a56616ef43a7)
- [DROID license clarification](https://github.com/droid-dataset/droid/issues/62)
- [target SO-101 calibrated URDF](https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf)

## Source and inventory contract

| Field | Pinned value |
|---|---|
| Release | DROID raw v1.0.1 in the official `gresearch` bucket |
| Collection code | `33ae6a67274f36d2e29525b86f23a56616ef43a7` |
| License | CC BY 4.0; the official marker is adjacent to the RLDS release |
| Domain / embodiment | Real, Franka Panda, seven arm joints plus parallel gripper |
| Metadata inventory | 74,896 episodes: 59,740 success, 15,156 failure, 13 labs |
| Inventory method | Complete metadata-object pagination: 134 pages, 132,872,668 bytes |
| Full raw scale | Approximately 8.7 TB; approximately 5.6 TB without stereo recordings |
| Native state | Seven joint angles in radians plus normalized gripper position |
| Native action | Six normalized Cartesian velocity-like controller values plus gripper velocity |
| Cameras | Wrist 672x376; two exterior cameras 1280x720; H.264/yuv420p |

The public raw metadata contains contributor names and identifiers. They are
necessary only to describe the source risk and are never copied to canonical
episodes, manifests, split selectors or committed qualification evidence.
Derived qualification IDs use only `dataset_id:lab:outcome`.

The complete inventory is an observed public namespace snapshot dated
31 August 2026, not a provider-published immutable manifest. Every object used
for qualification is independently immutable: the 59-object inventory hash is
`e0f8d51151e78387641f8ae1238d5e181887dd6fbfa15d4eee309eb22f6bc69f`.

## Bounded acquisition

The exact subset is 75,738,594 bytes:

- 26 JSON metadata objects and 26 paired HDF5 trajectories;
- six non-stereo MP4 files for both IPRL outcomes and all three camera roles;
  and
- one official CC BY 4.0 license object.

All 59 local objects pass generation-pin presence, byte size, MD5 and SHA-256
verification. Raw files and decoded canonical samples stay under ignored
`data/raw/` and `data/cleaned/`. Only the compact object inventory, source
manifest, aggregate qualification evidence and lab split are versioned.

## Integrity and outcome policy

All 26 trajectories pass strict validation, covering 7,817 source frames and
7,791 canonical transitions:

- the JSON lab/outcome, bucket path and HDF5 root success/failure attributes
  agree, with success/failure exactly exclusive;
- every HDF5 array is aligned to the declared trajectory length;
- required 7D joint, scalar gripper, 6D Cartesian command and derived-command
  shapes are exact and contain no NaN or Inf;
- all three camera serial roles agree between JSON and HDF5;
- native control timestamps are finite and strictly increasing; and
- skip-action transitions are preserved for source-local prediction but marked
  ineligible as executed imitation actions.

DROID's post-hoc relabel operation updates the root outcome and directory but
can leave `controller_info` arrays stale. Two qualification episodes exhibit
this documented condition. Per-step flags are therefore audit-only; path,
JSON and root attributes remain the mutually agreeing authoritative label.
Episodes with outcome disagreement or timestamp regression are rejected, not
silently repaired.

## Canonical conversion

Canonical `q` is width eight: seven observed joint positions plus gripper.
`qdot` is recomputed from strict native timestamps. `previous_command` begins
at the first observed state, then uses the prior derived absolute seven-joint
plus gripper target. The native action remains width seven: six normalized
Cartesian controller values plus gripper velocity.

No task-space conversion is emitted. In particular, DROID Franka actions are
not reinterpreted through the SO-101 calibrated URDF. That URDF defines the
target embodiment, but it supplies no cross-robot action equivalence.

Both IPRL outcomes were canonicalized for two transitions. Each result has
three observations, two actions, 8D state, 7D native action and six decoded
RGB tensors. Success quality is 1.0 and imitation-eligible; failure quality is
0.0 and prediction-only. All structurally valid transitions remain eligible
for source-local prediction, subject to the skip-action mask.

## Video alignment

All six representative MP4s declare H.264, yuv420p and 60 fps. Each contains
exactly one fewer frame than its HDF5 trajectory: 196 video frames for the
197-frame success and 281 for the 282-frame failure. First, middle and last
frames decode for every stream—18 independently hashed RGB tensors total.

The container playback rate is not the approximately 15 Hz control clock.
Raw DROID video is therefore aligned by frame index only. Container timestamps
must never be used as control timestamps.

## Leakage-resistant split

Complete collection labs are the split unit, preventing the same operators,
scenes, cameras and robot instance from crossing boundaries. Validation and
test lab pairs were chosen to have similar sizes and failure rates.

| Split | Labs | Episodes | Success | Failure | Failure rate |
|---|---:|---:|---:|---:|---:|
| Train | 9 | 58,234 | 46,491 | 11,743 | 20.17% |
| Validation | 2 | 8,517 | 6,781 | 1,736 | 20.38% |
| Test | 2 | 8,145 | 6,468 | 1,677 | 20.59% |

The 13 labs are complete and pairwise disjoint, both outcomes occur in every
split, and counts sum exactly to the 74,896-episode inventory. The exact lab
selectors and qualification cells are versioned under `data/splits/`.

## Reproduce

```bash
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python scripts/qualify_droid_raw.py --download
.venv/bin/python scripts/qualify_droid_raw.py
.venv/bin/python -m pytest -q
```

The downloader requests only the 59 pinned immutable GCS generations, keeps a
1 GiB disk reserve, verifies size/MD5/SHA-256 and never overwrites an existing
raw object. Reproducibility records refuse content-changing rewrites.

## Boundary and next step

Stage 1.4 proves source identity, license, a complete outcome/lab inventory,
bounded raw integrity, real H.264 decoding, PII redaction, canonical conversion
and a lab-disjoint split. It does not prove useful transfer to Project
Subliminal's target models.

The next authorized step is a separately specified bounded value and
forgetting gate using these frozen train/validation/test lab groups. It must
show held-out target improvement without unacceptable forgetting on the
existing real SO-101 sources. Until that gate passes, DROID remains validated
but not admitted, and the remaining multi-terabyte release stays blocked.
