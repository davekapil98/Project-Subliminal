# Frozen data splits

Small train/validation/test selectors are versioned here. Split by episode and,
where possible, scene/task/source session; never split adjacent frames from one
episode. Large ID tables remain external and are represented by deterministic
selectors plus hashes.

ArmnetBench uses a complete 8x8 task-policy matrix: one distinct cell per task
is held for validation and test, and the remaining six cells train. Every
rollout sharing an exact task and policy family stays in one split; both
held-out splits retain all tasks, policies and outcome classes.

SO101 MA MultiTask uses contiguous ten-episode source-order blocks within each
task. One rotating block per task is held for validation, one distinct block
for test and eight for train. All seven tasks remain in every split and no
adjacent source block crosses a boundary.

DROID raw holds complete collection laboratories together. Nine labs train,
two validate and two test; the groups are disjoint, cover all 74,896 inventoried
episodes and retain both outcomes at closely matched failure rates.
