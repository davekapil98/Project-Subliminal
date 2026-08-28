# Stage 1: GB100 Isaac Sim data generation

Stage 1 is packaged as a copy-and-run workflow. The public entry point is
`./launch_stage1.sh`. It will not start dataset recording until both host and
live Isaac checks pass and the operator presses Enter.

## What is prebuilt

The production plan in `configs/simulation/stage1_gb100.toml` runs four named
world distributions in priority order:

| Priority | World | Policy/data purpose | Episodes | Steps |
|---:|---|---|---:|---:|
| 1 | `motor_bench` | Safe random motion for Body Dynamics | 4,096 | 96 |
| 2 | `randomized_objects` | Smooth motor-goal sweeps | 4,096 | 128 |
| 3 | `rack_task` | Vials-to-rack task attempts | 2,048 | 180 |
| 4 | `recovery_states` | Disturbed/partial failure recovery | 2,048 | 128 |

The total is 12,288 episodes and 1,548,288 transitions. Each production world
runs 64 SO-101 instances in parallel. The final two jobs also store exact
instance segmentation from the same rear camera; segmentation is a label, not
a second camera input.

Every robot instance has exactly:

- one RGB input named `rear_camera`, bound to the external D455 camera on the
  workshop's stand behind the robot;
- one IMU named `gripper_imu`, attached to the `Robot/gripper` rigid-body chain;
- the NVIDIA workshop's filtered jaw-to-vial contact sensor plus a robot contact
  sensor used only for labels.

The wrist camera provided by the upstream workshop is explicitly disabled.

## First GB100 launch

Prerequisites on the rented server are Python 3.11 or newer, an NVIDIA driver,
Docker, NVIDIA Container Toolkit, enough local SSD storage, and outbound access
to NGC, GitHub, and Git LFS. No host-side PyTorch, NumPy, Isaac, or project
installation is required. If NGC asks for authentication, log in before
launching:

```bash
docker login nvcr.io
```

Copy the complete project folder to the server, enter it, and run:

```bash
chmod +x launch_stage1.sh
./launch_stage1.sh
```

The launcher performs this sequence:

1. Check that a GB100/B100/B200 is visible, Docker has the NVIDIA runtime, the
   output path is writable, and the disk reserve is available.
2. Build the container pinned to Isaac Lab 2.3.2, NVIDIA's SO-101 workshop
   commit `ce807d99724cb65671abec01f908a2fcb4a6eab7`, and LeRobot commit
   `e670ac5daf9b76`.
3. Boot every configured world with two parallel arms, take a physics step, and
   verify the real joint, RGB, IMU, object-pose, and contact tensor contracts.
   PNG previews and a JSON report are written under
   `data/raw/isaac_so101/.stage1_preflight/<config-digest>/`.
4. Print `READY TO RECORD` and wait. No episode writer or production simulation
   has started yet. Press Enter to begin, or type `q` and Enter to leave safely.

To perform only steps 1-3:

```bash
./launch_stage1.sh --preflight-only
```

To exercise the complete writer with four tiny episodes before committing to
the production run:

```bash
./launch_stage1.sh \
  --config configs/simulation/stage1_smoke.toml \
  --run-id stage1_smoke
```

`--allow-non-gb100` exists only for deliberate smoke tests on another NVIDIA
GPU. Production defaults refuse a non-GB100-class machine.

## Resume and storage safety

The launcher records the current run in
`data/raw/isaac_so101/.stage1_active_run`. Re-running the same command after a
disconnect or instance restart resumes that run. Use `--new-run` only when an
independent dataset run is intended, or supply an explicit stable `--run-id`.

Raw episodes are never edited or overwritten. Each episode is created under a
run-local `_staging` directory, compressed, hashed, given a `COMMITTED.json`
marker, and atomically renamed into `episodes/`. Resume scans and verifies
committed episodes before skipping them. Incomplete staging data is not an
admitted raw episode and is discarded on restart.

The production configuration stops before the run exceeds 350 GiB or would
leave less than 40 GiB free. A quota stop keeps the active marker and all prior
episodes valid. Add storage and run the launcher again to resume.

After completion, the launcher performs a full SHA-256 and array-contract audit
inside the pinned container. It removes the active-run marker only after that
audit passes.

Copy the raw run off the rental server immediately. For example, from the
destination machine:

```bash
rsync -a --partial --info=progress2 \
  user@gb100-host:/path/to/subliminal/data/raw/isaac_so101/<run-id>/ \
  ./data/raw/isaac_so101/<run-id>/
```

Use `rsync --checksum` for a transfer-level recheck. A second full dataset audit
can be run from any copy of the project with its Stage 1 image already built by
reusing the containerized validation command in `.launch_stage1_impl.sh`.

## Useful launcher options

```text
--preflight-only       verify without recording
--yes                  start automatically after a passing preflight
--run-id ID            select or resume a stable run
--new-run              ignore the active-run marker
--no-build             require the image to exist already
--rebuild              force a clean image rebuild
--output PATH          put raw runs on a mounted data disk
--livestream           enable Isaac WebRTC livestream
--allow-non-gb100      deliberate smoke run only
```

## Cloud-day operating advice

Use the smoke TOML first, inspect every preview, then use the production TOML.
Keep the terminal attached through `tmux` or the provider's equivalent. Watch
free disk space and the `run_state.json` counters. The quotas deliberately put
Body Dynamics and Motor Cortex data first, so a provider interruption still
leaves the highest-priority subset usable.

The task policy is a deterministic, safe scripted attempt generator rather
than a trained expert. Task episodes therefore intentionally contain mixed
success, failure, collision, and recovery outcomes, all labeled from simulator
state. Failed trajectories are suitable for World/Body/critic/recovery uses;
they must not be treated as successful Motor Cortex demonstrations.
