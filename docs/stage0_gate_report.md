# Stage 0 baseline report

Date: 28 August 2026
Environment: Python 3.12.3, PyTorch 2.13.0 CPU

## Result

`23 passed in 1.64s`

The suite proves the Stage 0 code baseline currently covers:

- finite forward and backward passes and expected output shapes;
- tiny-batch overfitting for all eight neural modules;
- JEPA EMA and masked latent prediction;
- flow-path endpoints, sampling, and motor vector-field loss;
- body uncertainty and Gaussian NLL;
- bus schema validation, serialization, staleness, sequence, and correlation;
- deterministic safety filtering and model-predictive candidate selection;
- canonical episode validation and stable episode-level splitting;
- checkpoint save/load with config, data, normalization, precision, and commit
  metadata;
- CPU mixed-precision fallback and deterministic seeded cycles; and
- one complete synthetic language-to-execution-to-memory loop.

Reproduce with:

```bash
.venv/bin/python scripts/run_stage0_gate.py
.venv/bin/python scripts/run_stage0_demo.py
```

This is a code-correctness gate only. It is not evidence of learned robotic
competence or readiness for physical hardware.
