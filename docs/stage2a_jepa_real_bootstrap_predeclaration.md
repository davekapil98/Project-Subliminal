# Stage 2A entry — admitted real-data JEPA bootstrap

Protocol revision 1, frozen 1 September 2026 before its non-decision preflight.

## Why this is the next eligible action

Stage 1.5 admitted exactly one new use:
`jepa_encoder_action_free_temporal_pretraining`. Its passing evidence is pinned
by SHA-256 in
[`stage2a_jepa_real_bootstrap.toml`](../configs/training/stage2a_jepa_real_bootstrap.toml).
The v1.3 master specification routes this work to Stage 2A broad representation
pretraining, but requires full specialist training to run on the RTX 3060 12 GB
and requires a throughput/memory benchmark before committing to a long dense
run.

This revision therefore starts Stage 2A with an admitted-real bootstrap. It
implements and locally preflights the production data/objective/checkpoint
path, then stops at the RTX 3060 benchmark boundary. It is not the complete
broad sim-plus-real Stage 2A curriculum: the available SO101-MA simulation
source failed its earlier forgetting gate and is not silently re-admitted.

## Frozen scope

The only training sources are the passing Stage 1.5 subset:

- DROID real Franka video: 2,304 action-free training samples.
- Project IRA real SO-101 video: 720 action-free training samples.
- ArmnetBench real SO-101 video: 400 action-free training samples.

The deterministic source schedule is DROID, Project IRA, ArmnetBench, one
optimizer step each in rotation. This balances by source rather than frame
count. All samples retain the 0.50–0.60 second context/future horizon, camera
availability mask, and source-training-only normalization frozen for Stage
1.5. Raw data, caches, logs, and weight files remain Git-ignored.

No action field may be loaded, passed to the model, predicted, or stored in a
training shard. DROID Franka actions remain excluded from SO-101 Motor Cortex,
Body Dynamics, action-conditioned JEPA World, Executive, and Language uses.

## Production path

`MultimodalJEPAEncoder` separates a shared per-camera visual stack from the
multimodal fusion stack specified by v1.3. It supports provider-valid camera
masks, a proprioceptive history/source token, compact world-token resampling,
and optional per-block activation checkpointing. The RTX 3060 benchmark profile
uses 20 visual blocks and six fusion blocks at width 896, 32 world tokens, and
a 768-wide bus. A six-block temporal predictor is a training component; only
the encoder is eligible for eventual promotion.

The online encoder observes context RGB and normalized observed proprioceptive
history. An exponential-moving-average target encoder observes future RGB and
zero proprioception. Prediction and stopped-gradient target tokens are layer
normalized before smooth-L1 matching. This is action-free latent prediction,
not raw-pixel reconstruction. Target and prediction feature standard deviation
are recorded as collapse monitors.

Camera dropout removes paired context/future views only from cameras the source
actually provides and always retains one view. This exercises the master-spec
sensor-drop acceptance requirement without inventing missing imagery.

## Checkpoint and evaluation boundary

Format-v2 checkpoints are written atomically and contain the online encoder,
predictor, EMA target, optimizer/scaler state, deterministic sampler and RNG
state, full config and hash, cache manifest and hash, source normalization,
precision mode, and Git commit. The preflight must prove a checkpoint can be
loaded and resumed through the same path.

DROID validation is the only evaluation split available during the preflight
and later training. DROID test, Project IRA test, and ArmnetBench test remain
locked until a final predeclared evaluation. Revision 1 cannot promote a
checkpoint regardless of its metrics.

## Hardware gate

The laptop preflight uses a small shape-compatible profile in FP32. The full
profile may only be benchmarked on CUDA hardware reporting at least 11 GB. Its
frozen memory-saving baseline is microbatch 1, accumulation 16, activation
checkpointing, and FP16 autocast with FP32 loss/optimizer state. The benchmark
must record finite gradients, peak allocated/reserved memory, and measured
samples/second.

Long-run steps remain zero and `authorized = false`. After the RTX 3060
benchmark, a revision-2 config must freeze the feasible profile, training
budget, checkpoint cadence, and numerical promotion/forgetting thresholds
before any decision-bearing run.

## Reproduction order

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/train_stage2a_jepa_real_bootstrap.py --preflight
# On the RTX 3060 12 GB machine only:
.venv/bin/python scripts/train_stage2a_jepa_real_bootstrap.py --benchmark
```
