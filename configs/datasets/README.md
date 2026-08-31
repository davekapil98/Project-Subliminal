# Dataset configurations — Stage 1

Master spec v1.3 defines Stage 1 as public dataset acquisition, validation and
canonicalization. Configuration is separated into:

- `registry/`: one pinned source record per dataset revision;
- `cleaning_profiles/`: deterministic, source-specific integrity and cleaning
  rules; and
- later mixture/subset configs that reference registered dataset IDs rather
  than unpinned URLs.

An admitted source must record its exact revision, source URL, license,
redistribution terms, sim/real domain, embodiment, data format, modalities,
camera schema, native action semantics, units/frames, checksum and limitations.

Stage 1 starts subset-first. No bulk download is allowed until the source card,
license, schema, sample decoding, alignment, disk estimate and target-model
value have been reviewed.

`project_ira_so101_v1.toml` is the first validated Priority-A source. Its
source-specific cleaning policy, immutable file pins, qualification report and
prompt-group split complete Stage 1.1 without admitting the source to training
before the later held-out model-improvement gate.

`armnetbench_so101_v01.toml` is the second validated Priority-A source. Its
exact 128-object qualification subset, all numeric trajectories, three-camera
outcome sample, three-way label policy and task-policy split complete Stage 1.2.
The remaining media and any training mixture stay blocked by the held-out
target-model-value gate.

`so101_ma_multitask_700.toml` is the third validated Priority-A source. Its
seven upstream pins, exact 13-object subset, full numeric validation,
mixed-codec edge cases and contiguous source-block split complete Stage 1.3.
Validation does not imply admission: the matched-update TinyBodyDynamics gate
failed the real-source forgetting limit, so the source has no admitted training
use and the remaining media stays blocked.
