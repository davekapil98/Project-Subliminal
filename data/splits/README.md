# Frozen data splits

Small train/validation/test selectors are versioned here. Split by episode and,
where possible, scene/task/source session; never split adjacent frames from one
episode. Large ID tables remain external and are represented by deterministic
selectors plus hashes.

ArmnetBench uses a complete 8x8 task-policy matrix: one distinct cell per task
is held for validation and test, and the remaining six cells train. Every
rollout sharing an exact task and policy family stays in one split; both
held-out splits retain all tasks, policies and outcome classes.
