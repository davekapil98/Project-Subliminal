# Dataset adapters

Adapters read native LeRobot/Parquet+video, HDF5, RLDS/TFRecord and other
source layouts without altering raw files. Each adapter exposes source-native
values plus proven metadata before canonical conversion.

`project_ira_so101.py` is the first production source adapter. It validates the
complete pinned numeric trajectory table, episode/task alignment, source-native
units and actions, camera segment metadata and real H.264 decoding. It does not
invent task-space actions, camera calibration, success labels or quality labels
that the source does not publish.
