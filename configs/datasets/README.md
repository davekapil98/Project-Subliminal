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
