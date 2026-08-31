# Versioned dataset metadata

This directory stores small, reviewable metadata—not training samples:

- pinned source and license records;
- checksums and source revisions;
- deterministic preprocessing-run records; and
- compact selectors that reproduce any externally stored large manifest.

Large per-episode tables, decoded media and model shards remain outside Git.
Every tracked metadata file is limited by the repository hygiene guard.

The ArmnetBench qualification record binds the pinned object inventory, full
numeric validation, representative AV1 probes, outcome-class canonical sample,
frozen split and comparison with Project IRA. The decoded sample path and raw
objects remain ignored.

The SO101 MA MultiTask records bind seven upstream revisions, a 13-object
subset, complete numeric/provenance validation, mixed H.264/AV1 edge cases,
the source-block split and a matched-update TinyBodyDynamics value result. Its
source qualification passes, but its separate admission decision is
`not_admitted` because Project IRA forgetting exceeds the configured limit.
