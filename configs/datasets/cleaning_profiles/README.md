# Source cleaning profiles

Profiles declare integrity checks, timestamp/alignment policy, known unit/frame
conversions, model-safe derived signals and rejection reasons. They never edit
raw sources in place and never guess units from numeric ranges.

The ArmnetBench profile preserves successful/failure/suboptimal labels, audits
published statistics instead of trusting stale counts, permits only
unreferenced packed-video gaps caused by dropped episodes, and rejects
referenced overlap, timestamp drift and reward/done disagreement.
