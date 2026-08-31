# Pinned public dataset registry

One reviewed configuration belongs here for each exact public-source revision.
No source is marked `admitted` until its license, checksum, schema, units,
coordinate frames, action semantics and a small decoded subset are validated.

Priority-A SO-101 sources from master spec v1.3 are investigated first. URLs
and revisions must be verified from the current authoritative dataset card
before a registry entry is committed.

`armnetbench_so101_v01.objects.json` is the exact SHA-256/size inventory for
the bounded Stage 1.2 subset: source metadata, all 120 trajectory shards and one
representative packed file per camera. It is configuration evidence, not raw
dataset content.

`stage1_5_visual_subset.objects.json` freezes the 12,687,618,078-byte bounded
visual value-gate selection before acquisition. Its 2,222 records pin 416
DROID episodes by GCS generation/MD5, the ten selected SO-101 video packs by
revision/SHA-256, and the already-local support objects counted by the 20 GB
cap. It contains no raw metadata values or media.
