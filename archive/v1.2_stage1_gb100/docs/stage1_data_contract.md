# Stage 1 raw episode contract

Each immutable episode is stored at:

```text
data/raw/isaac_so101/<run-id>/episodes/<episode-id>/
├── metadata.json
├── telemetry.npz
├── rear_rgb.mp4
├── rear_segmentation.npz       # task/recovery jobs only
├── checksums.sha256
└── COMMITTED.json
```

The smoke/test backend may use `rear_rgb.npz`; production uses H.264 MP4 through
PyAV. There are `T+1` observations and `T` actions.

## Telemetry arrays

| Name | Shape | Meaning |
|---|---|---|
| `timestamp_sim_s` | `[T+1]` | Exact uniform simulation time |
| `timestamp_sensor_s` | `[T+1]` | Sensor time including configured jitter |
| `joint_position_rad` | `[T+1,6]` | Encoder-like noisy observation |
| `joint_position_true_rad` | `[T+1,6]` | Exact simulator position |
| `joint_velocity_rad_s` | `[T+1,6]` | Noisy velocity observation |
| `joint_velocity_true_rad_s` | `[T+1,6]` | Exact simulator velocity |
| `previous_command_rad` | `[T+1,6]` | Command active before this observation |
| `imu_linear_acceleration_m_s2` | `[T+1,3]` | Biased/noisy gripper accelerometer |
| `imu_angular_velocity_rad_s` | `[T+1,3]` | Biased/noisy gripper gyroscope |
| `applied_joint_torque_nm` | `[T+1,6]` | Isaac actuator applied effort |
| `object_pose_env_wxyz` | `[T+1,4,7]` | Exact XYZ + WXYZ poses for 3 vials and rack |
| `contact_force_by_vial_n` | `[T+1,3]` | Exact filtered normal jaw contact forces |
| `grasp_contact_bool` | `[T+1]` | Contact/official grasp subtask label |
| `collision_bool` | `[T+1]` | Non-gripper robot contact over threshold |
| `success_bool` | `[T+1]` | Official vial-on-rack subtask label |
| `hard_limit_bool` | `[T+1]` | Joint within 0.015 rad of a soft limit |
| `action_requested_joint_position_rad` | `[T,6]` | Safe policy target before bus effects |
| `action_native_joint_position_rad` | `[T,6]` | Authoritative delayed/deadband target |
| `action_relative_joint_rad` | `[T,6]` | Applied target minus observed position |
| `actuation_delay_steps` | `[T]` | Per-episode sampled delay at 30 Hz |

The joint order is always `Rotation`, `Pitch`, `Elbow`, `Wrist_Pitch`,
`Wrist_Roll`, `Jaw`. Object order is always `vial_1`, `vial_2`, `vial_3`,
`rack_left`.

`metadata.json` records task text, success/failure/outcome, quality, world and
randomization ranges, episode/cycle seeds, environment slot, sensor placement,
simulator/workshop/container/source revisions, the complete asset-tree hash,
and all tensor shapes/dtypes. `run_manifest.json` embeds the complete TOML and
its SHA-256 digest. `episodes.jsonl` is rebuilt from committed metadata so it
cannot admit a half-written directory.
