# Stage 2A admitted-real JEPA bootstrap preflight

Run date: 1 September 2026

Outcome: **passed as an engineering preflight; no checkpoint promoted and no
long run authorized**.

## What started

Stage 2A now has an executable action-free temporal pretraining path for the
JEPA encoder. The path is pinned to the passing Stage 1.5 evidence and uses
only DROID, Project IRA, and ArmnetBench visual samples admitted by that gate.
It balances optimizer updates 1:1:1 by source and cannot accept an action
tensor.

The production-scalable encoder follows the v1.3 split between a shared
per-camera visual stack and multimodal fusion/resampling. The frozen RTX 3060
benchmark profile has 255,459,991 encoder parameters and a 43,659,264-parameter
training-only temporal predictor. Its EMA target adds checkpoint state but no
trainable parameters.

## Preflight result

The small, shape-compatible profile ran in FP32 on CPU from frozen commit
`3e40124`. It used nine optimizer steps with two microbatches of two samples
per step:

| Source | Optimizer steps |
|---|---:|
| DROID | 3 |
| Project IRA | 3 |
| ArmnetBench | 3 |

All seven frozen requirements passed. The loader verified the Stage 1.5
admission, object selection, cache manifest, and all seven cache checksums
before constructing the model. Fourteen provider-valid camera views were
dropped during training, and every sample retained at least one camera.

The first training loss was 0.71313. After saving at optimizer step 4,
reconstructing the model and optimizer, restoring model/EMA, sampler, scaler,
and RNG state, and continuing to step 9, the final loss was 0.12903. The model
state before save and after reload had the same SHA-256 digest.

DROID validation was the only evaluation split accessed:

| Metric | Result |
|---|---:|
| Future-latent smooth-L1 | 0.11377 |
| One-camera-drop smooth-L1 | 0.11436 |
| Relative sensor-drop degradation | 0.52% |
| Target feature standard deviation | 0.06276 |
| Prediction feature standard deviation | 0.01818 |

The nonzero feature-variation monitors show that this short preflight did not
produce a constant-token shortcut. These values are engineering diagnostics,
not promotion thresholds or performance claims.

## Preserved boundary

No DROID, Project IRA, or ArmnetBench test split was opened. No action field
was loaded, predicted, or written. The preflight does not admit JEPA World,
Motor Cortex, Body Dynamics, Executive, Language, or cross-embodiment action
training. Checkpoints and caches remain ignored; only the 3.4 KB reviewable
evidence record is committed.

## Next required action

Run the frozen full profile on the RTX 3060 12 GB machine:

```bash
.venv/bin/python scripts/train_stage2a_jepa_real_bootstrap.py --benchmark
```

The command refuses CPU-only PyTorch and GPUs reporting less than 11 GB. It
will record peak allocated/reserved memory, finite-gradient metrics, and
samples per second. Only after reviewing that benchmark may revision 2 freeze
a long-run budget and numerical temporal, sensor-drop, DROID-value, and SO-101
forgetting thresholds. Revision 1 has `long_run.authorized = false` and zero
long-run optimizer steps.

Exact numerical evidence is in
[`stage2a_jepa_real_bootstrap.preflight.json`](../data/manifests/stage2a_jepa_real_bootstrap.preflight.json),
and the frozen methodology is in
[`stage2a_jepa_real_bootstrap_predeclaration.md`](stage2a_jepa_real_bootstrap_predeclaration.md).
