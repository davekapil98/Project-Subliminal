# Simulation configurations

- `stage1_gb100.toml` is the production 64-environment, four-world Stage 1 plan.
- `stage1_smoke.toml` exercises the same sensor and writer contract with two
  environments and four tiny episodes.

Both are consumed by the repository-root `launch_stage1.sh`. See
`docs/stage1_gb100.md` before renting the server. Generated raw data belongs in
`data/raw/isaac_so101/<run-id>` and is write-once.
