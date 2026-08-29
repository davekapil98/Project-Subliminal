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
