# Stage 1 preparation gate report

Date: 2026-08-28

This report audits the portable GB100 deliverable. It does not claim that the
dataset has already been generated. Live Isaac rendering and physics are
deliberately performed by `launch_stage1.sh` on the rented GPU before the
operator can start production recording.

| Requirement | Implementation evidence | Local evidence | GB100 live gate |
|---|---|---|---|
| One copy-and-run launch file | `launch_stage1.sh` and `.launch_stage1_impl.sh` | Shell syntax and confirmation-order test pass | Public launcher runs host + container checks |
| Pinned Isaac/SO-101 runtime | `docker/stage1/Dockerfile` | Pin/static tests pass | Image boots and reports actual commits/assets |
| Different prebuilt worlds | Four profiles and four prioritized jobs in `stage1_gb100.toml` | Strict TOML/profile tests pass | Every unique world is instantiated and stepped |
| Multiple SO-101 instances | 64 production, 2 preflight environments | Config asserts at least two | Actual joint tensor must have the requested batch size |
| One rear stand camera | Camera-free robot USD plus `rear_camera` on official D455 mount | Static single-camera contract test passes | Actual scene sensor list must be exactly `rear_camera`; every environment must render non-blank pixels |
| One gripper IMU | `gripper_imu` targets `Robot/gripper`, physics at 240 Hz | Strict config/static tests pass | Actual accel/gyro tensors must both be `[N,3]` and finite |
| Verify before recording | Live `--mode verify`, JSON report, PNG previews | Verify invocation precedes prompt and record invocation | Any failed world/sensor/label check exits before prompt |
| Button/confirmation start | Enter/q prompt after passing report | Launcher ordering test passes | Only Enter (or explicit `--yes`) reaches record mode |
| Canonical compressed episodes | `records.py`, `runner.py`, `writer.py` | T+1/T shapes and deterministic fake-backend run pass | Smoke TOML writes H.264/NPZ telemetry using live tensors |
| Exact labels | Object poses, per-vial force, grasp, collision, success, hard-limit, segmentation | Shape/schema tests pass | Live label tensors and segmentation ID mapping are mandatory preflight checks |
| Domain randomization | World profiles + Isaac actuator, joint, mass, material, lighting, pose resets + delay/noise model | Policy/randomization determinism tests pass | Isaac EventManager constructs every required term |
| Immutable/resumable raw data | Staging + hashes + commit marker + atomic rename + stable IDs | Commit/resume/corruption tests pass | Active-run marker resumes verified IDs after interruption |
| Provenance and quotas | Full config digest, upstream/source/image revisions, asset-tree hash, seeds, disk guards | Config totals and writer guard paths tested | Actual image/revision/GPU metadata enters run manifest |

## Local gate result

- Complete test suite: **38 passed**.
- Python compilation: **passed**.
- Public launcher, internal launcher, and container entrypoint shell syntax:
  **passed**.
- Git whitespace check: **passed**.
- Bare-host configuration query without third-party Python packages: **passed**.
- Local Isaac execution: **not attempted**, as required by the master
  specification for the 4 GB laptop.

## Required first cloud action

Run the smoke plan first:

```bash
./launch_stage1.sh \
  --config configs/simulation/stage1_smoke.toml \
  --run-id stage1_smoke
```

Inspect the four preview images. After the smoke run and checksum audit pass,
run `./launch_stage1.sh` for the production quotas.
